"""Modal script: Gaussian Splatting from a restaurant walkthrough video.

Uses the original 3DGS implementation (graphdeco-inria/gaussian-splatting)
which has its own CUDA rasterizer and avoids gsplat compatibility issues.

Uploads video → extracts frames → COLMAP → 3DGS train → exports .ply
All on a Modal A100 GPU. Downloads result locally.

Usage:
    modal run modal_gaussian_splat.py --video /Users/heng/Downloads/test_rest.mp4
"""
import modal
from pathlib import Path

app = modal.App("gaussian-splat")

# Build image: CUDA 11.8 + COLMAP + original 3DGS repo
gs_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "colmap", "imagemagick", "g++", "ninja-build")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "CC": "gcc", "CXX": "g++"})
    .run_commands(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        " && pip install plyfile tqdm Pillow numpy setuptools wheel opencv-python-headless"
        " && git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /opt/gaussian-splatting"
        " && pip install --no-build-isolation /opt/gaussian-splatting/submodules/diff-gaussian-rasterization"
        " && pip install --no-build-isolation /opt/gaussian-splatting/submodules/simple-knn"
    )
)

vol = modal.Volume.from_name("gs-workspace", create_if_missing=True)
VOLUME_PATH = "/workspace"


@app.function(
    image=gs_image,
    gpu="A100",
    timeout=2400,  # 40 min
    volumes={VOLUME_PATH: vol},
)
def train_gaussian_splat(video_bytes: bytes, video_name: str = "input.mp4"):
    """Full pipeline: frames → COLMAP → 3DGS train → .ply export."""
    import subprocess
    import shutil

    work = Path(VOLUME_PATH) / "gs_run"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1. Save video
    video_path = work / video_name
    video_path.write_bytes(video_bytes)
    print(f"[1/5] Video saved: {len(video_bytes)/1024/1024:.1f}MB")

    # 2. Extract frames (5fps, 800px wide)
    input_dir = work / "input"
    input_dir.mkdir()
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", "fps=5,scale=800:-2",
        "-q:v", "2",
        str(input_dir / "frame_%04d.jpg"),
    ], check=True, capture_output=True)
    n = len(list(input_dir.glob("*.jpg")))
    print(f"[2/5] Extracted {n} frames")

    # 3. Run COLMAP (the 3DGS repo expects a specific directory structure)
    # Structure: <project>/input/<images>  →  COLMAP outputs to <project>/sparse/0/
    sparse_dir = work / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    db_path = work / "database.db"

    # Feature extraction
    result = subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Feature extraction stderr: {result.stderr[-2000:]}")
        raise RuntimeError("COLMAP feature extraction failed")
    print("[3/5] COLMAP feature extraction done")

    # Matching
    result = subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "0",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Matching stderr: {result.stderr[-2000:]}")
        raise RuntimeError("COLMAP matching failed")
    print("       COLMAP matching done")

    # Sparse reconstruction
    result = subprocess.run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--output_path", str(work / "sparse"),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Mapper stderr: {result.stderr[-2000:]}")
        raise RuntimeError("COLMAP mapper failed")
    print("       COLMAP reconstruction done")

    # Undistort images (3DGS expects undistorted images)
    result = subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(work / "undistorted"),
        "--output_type", "COLMAP",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Undistort stderr: {result.stderr[-2000:]}")
        raise RuntimeError("COLMAP undistortion failed")

    # Move undistorted output to where 3DGS expects it
    undist = work / "undistorted"
    # 3DGS expects: <source>/images/ and <source>/sparse/0/
    # undistorter outputs: <output>/images/ and <output>/sparse/
    if (undist / "sparse").exists():
        target_sparse = undist / "sparse" / "0"
        if not target_sparse.exists():
            # Move contents into 0/ subdirectory if needed
            target_sparse.mkdir(parents=True)
            for f in (undist / "sparse").iterdir():
                if f.is_file():
                    shutil.move(str(f), str(target_sparse / f.name))
    print("[4/5] COLMAP pipeline complete")

    # 4. Train 3DGS
    output_dir = work / "output"
    result = subprocess.run([
        "python", "/opt/gaussian-splatting/train.py",
        "-s", str(undist),
        "-m", str(output_dir),
        "--iterations", "7000",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Train stdout: {result.stdout[-3000:]}")
        print(f"Train stderr: {result.stderr[-3000:]}")
        raise RuntimeError("3DGS training failed")
    print("[5/5] Training done")

    # 5. Find the output .ply
    ply_path = output_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply"
    if not ply_path.exists():
        # Try other iteration counts
        plys = list(output_dir.rglob("point_cloud.ply"))
        if not plys:
            raise RuntimeError("No point_cloud.ply found")
        ply_path = plys[-1]  # Take the last (highest iteration)

    out_bytes = ply_path.read_bytes()
    print(f"Export: {ply_path.name} ({len(out_bytes)/1024/1024:.1f}MB)")

    vol.commit()
    return out_bytes, "point_cloud.ply"


@app.local_entrypoint()
def main(video: str = "/Users/heng/Downloads/test_rest.mp4"):
    """Upload video, train on A100, download .ply."""
    p = Path(video)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {video}")

    print(f"Uploading {p.name} ({p.stat().st_size/1024/1024:.1f}MB)...")
    data, name = train_gaussian_splat.remote(p.read_bytes(), p.name)

    out_dir = Path("/Users/heng/Development/DINO/output/gaussian_splat")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    out_path.write_bytes(data)
    print(f"Saved: {out_path} ({len(data)/1024/1024:.1f}MB)")
