"""Modal script: Dense room reconstruction via COLMAP MVS.

Runs COLMAP SfM + patch_match_stereo + stereo_fusion on a Modal A100 GPU
to produce a dense colored point cloud (fused.ply) from a walkthrough video.

Usage:
    modal run modal_colmap_dense.py --video /Users/heng/Downloads/test_rest.mp4
"""
import modal
from pathlib import Path

app = modal.App("colmap-dense")

colmap_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "ffmpeg", "imagemagick", "g++", "ninja-build",
        "cmake", "libboost-all-dev", "libfreeimage-dev", "libgoogle-glog-dev",
        "libgflags-dev", "libglew-dev", "libsqlite3-dev", "libceres-dev",
        "qtbase5-dev", "libqt5opengl5-dev", "libcgal-dev", "libflann-dev",
    )
    .run_commands(
        # Build COLMAP from source with CUDA support for patch_match_stereo
        "git clone --branch 3.9.1 https://github.com/colmap/colmap.git /tmp/colmap"
        " && mkdir /tmp/colmap/build && cd /tmp/colmap/build"
        " && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release"
        " && ninja -j$(nproc) && ninja install"
        " && rm -rf /tmp/colmap"
    )
    .run_commands("pip install plyfile tqdm numpy opencv-python-headless")
)

vol = modal.Volume.from_name("colmap-dense-workspace", create_if_missing=True)
VOLUME_PATH = "/workspace"


@app.function(
    image=colmap_image,
    gpu="A100",
    timeout=2400,  # 40 min
    volumes={VOLUME_PATH: vol},
)
def run_colmap_dense(video_bytes: bytes, video_name: str = "input.mp4"):
    """Full pipeline: frames -> COLMAP SfM -> dense MVS -> fused.ply."""
    import subprocess
    import shutil

    work = Path(VOLUME_PATH) / "dense_run"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1. Save video (sanitize filename to prevent path traversal)
    video_name = Path(video_name).name or "input.mp4"
    video_path = work / video_name
    video_path.write_bytes(video_bytes)
    print(f"[1/7] Video saved: {len(video_bytes) / 1024 / 1024:.1f}MB")

    # 2. Extract frames (5fps, 800px wide)
    input_dir = work / "images"
    input_dir.mkdir()
    try:
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=5,scale=800:-2",
            "-q:v", "2",
            str(input_dir / "frame_%04d.jpg"),
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg stderr: {e.stderr[-2000:]}")
        raise RuntimeError(f"Frame extraction failed: {e.stderr[-500:]}") from e
    n = len(list(input_dir.glob("*.jpg")))
    print(f"[2/7] Extracted {n} frames")

    # 3. COLMAP feature extraction (GPU SIFT)
    db_path = work / "database.db"
    sparse_dir = work / "sparse"
    sparse_dir.mkdir()

    _run_colmap("feature_extractor", [
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ], "Feature extraction")
    print("[3/7] Feature extraction done")

    # 4. Sequential matching (GPU)
    _run_colmap("sequential_matcher", [
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ], "Matching")
    print("[4/7] Matching done")

    # 5. Sparse reconstruction
    _run_colmap("mapper", [
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--output_path", str(sparse_dir),
    ], "Sparse reconstruction")
    print("[5/7] Sparse reconstruction done")

    # Find the reconstruction (usually sparse/0/)
    recon_dir = sparse_dir / "0"
    if not recon_dir.exists():
        recons = sorted(sparse_dir.iterdir())
        if not recons:
            raise RuntimeError("COLMAP mapper produced no reconstruction")
        recon_dir = recons[0]

    # 6. Image undistorter (needed for dense reconstruction)
    dense_dir = work / "dense"
    _run_colmap("image_undistorter", [
        "--image_path", str(input_dir),
        "--input_path", str(recon_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], "Undistortion")

    # Patch match stereo (dense depth maps, GPU)
    _run_colmap("patch_match_stereo", [
        "--workspace_path", str(dense_dir),
        "--PatchMatchStereo.gpu_index", "0",
        "--PatchMatchStereo.geom_consistency", "true",
    ], "Patch match stereo")
    print("[6/7] Dense depth maps done")

    # 7. Stereo fusion -> fused.ply
    fused_ply = dense_dir / "fused.ply"
    _run_colmap("stereo_fusion", [
        "--workspace_path", str(dense_dir),
        "--output_path", str(fused_ply),
    ], "Stereo fusion")
    print("[7/7] Stereo fusion done")

    if not fused_ply.exists():
        raise RuntimeError("stereo_fusion did not produce fused.ply")

    out_bytes = fused_ply.read_bytes()
    print(f"Result: fused.ply ({len(out_bytes) / 1024 / 1024:.1f}MB)")

    vol.commit()
    return out_bytes


def _run_colmap(command: str, args: list, label: str):
    """Run a COLMAP command with error handling."""
    import subprocess

    result = subprocess.run(
        ["colmap", command] + args,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"{label} stdout: {result.stdout[-2000:]}")
        print(f"{label} stderr: {result.stderr[-2000:]}")
        raise RuntimeError(
            f"COLMAP {command} failed (exit {result.returncode}): "
            f"{result.stderr[-500:]}"
        )
    if result.stderr:
        print(f"{label} warnings: {result.stderr[-1000:]}")


@app.local_entrypoint()
def main(video: str = "/Users/heng/Downloads/test_rest.mp4"):
    """Upload video, run dense reconstruction on A100, download fused.ply."""
    p = Path(video)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {video}")

    print(f"Uploading {p.name} ({p.stat().st_size / 1024 / 1024:.1f}MB)...")
    data = run_colmap_dense.remote(p.read_bytes(), p.name)

    out_path = Path("viewer/point_cloud.ply")
    out_path.write_bytes(data)
    print(f"Saved: {out_path} ({len(data) / 1024 / 1024:.1f}MB)")
