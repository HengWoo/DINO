"""Modal script: Dense room reconstruction via SLAM3R (CVPR 2025).

SLAM3R reconstructs dense colored point clouds from monocular video
without camera calibration. Runs on a Modal A100 GPU.

Usage:
    modal run modal_slam3r.py --video /Users/heng/Downloads/test_rest.mp4
"""
import modal
from pathlib import Path

app = modal.App("slam3r")

slam3r_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "g++", "ninja-build", "cmake")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        "pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118"
    )
    .run_commands(
        "pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118"
    )
    .run_commands(
        "git clone https://github.com/PKU-VCL-3DV/SLAM3R.git /opt/slam3r"
    )
    .run_commands(
        "pip install roma scipy einops trimesh matplotlib tqdm"
        " opencv-python-headless 'huggingface-hub[torch]>=0.22' 'pyglet<2' tensorboard open3d plyfile"
    )
    # Build curope CUDA kernels for A100 (sm_80)
    .run_commands(
        "cd /opt/slam3r/slam3r/pos_embed/curope"
        " && sed -i 's/-arch=native/-arch=sm_80/' setup.py"
        " && python setup.py build_ext --inplace"
    )
    # Bake model weights into image (~4GB, downloads once during build)
    .run_commands(
        'python -c "from huggingface_hub import snapshot_download;'
        "snapshot_download('siyan824/slam3r_i2p');"
        "snapshot_download('siyan824/slam3r_l2w')\""
    )
)

vol = modal.Volume.from_name("slam3r-workspace", create_if_missing=True)
VOLUME_PATH = "/workspace"


@app.function(
    image=slam3r_image,
    gpu="A100",
    timeout=3600,  # 60 min
    volumes={VOLUME_PATH: vol},
)
def run_slam3r(video_bytes: bytes, video_name: str = "input.mp4"):
    """Full pipeline: video -> frames -> SLAM3R -> dense PLY."""
    import subprocess
    import shutil
    import glob

    work = Path(VOLUME_PATH) / "slam3r_run"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1. Save video
    video_path = work / video_name
    video_path.write_bytes(video_bytes)
    print(f"[1/3] Video saved: {len(video_bytes) / 1024 / 1024:.1f}MB")

    # 2. Extract frames (5fps, 800px wide)
    frames_dir = work / "frames"
    frames_dir.mkdir()
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", "fps=5,scale=800:-2",
        "-q:v", "2",
        str(frames_dir / "frame_%04d.jpg"),
    ], check=True, capture_output=True)
    n = len(list(frames_dir.glob("*.jpg")))
    print(f"[2/3] Extracted {n} frames")

    # 3. Run SLAM3R reconstruction
    result = subprocess.run([
        "python", "/opt/slam3r/recon.py",
        "--img_dir", str(frames_dir),
        "--test_name", "modal_run",
        "--num_points_save", "2000000",
        "--buffer_strategy", "reservoir",
        "--keyframe_stride", "3",
    ], capture_output=True, text=True, cwd="/opt/slam3r")

    if result.returncode != 0:
        print(f"SLAM3R stdout: {result.stdout[-3000:]}")
        print(f"SLAM3R stderr: {result.stderr[-3000:]}")
        raise RuntimeError(f"SLAM3R reconstruction failed (exit {result.returncode})")
    print("[3/3] SLAM3R reconstruction done")

    # Find the output PLY (SLAM3R outputs *_recon.ply in results/)
    ply_candidates = (
        glob.glob("/opt/slam3r/results/**/*_recon.ply", recursive=True)
        + glob.glob(str(work / "**/*_recon.ply"), recursive=True)
        + glob.glob("/opt/slam3r/results/**/*.ply", recursive=True)
    )
    if not ply_candidates:
        raise RuntimeError("SLAM3R did not produce a PLY file")

    ply_path = Path(ply_candidates[0])
    out_bytes = ply_path.read_bytes()
    print(f"Result: {ply_path.name} ({len(out_bytes) / 1024 / 1024:.1f}MB)")

    vol.commit()
    return out_bytes


@app.local_entrypoint()
def main(video: str = "/Users/heng/Downloads/test_rest.mp4"):
    """Upload video, run SLAM3R on A100, download PLY."""
    p = Path(video)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {video}")

    print(f"Uploading {p.name} ({p.stat().st_size / 1024 / 1024:.1f}MB)...")
    data = run_slam3r.remote(p.read_bytes(), p.name)

    out_path = Path("viewer/point_cloud.ply")
    out_path.write_bytes(data)
    print(f"Saved: {out_path} ({len(data) / 1024 / 1024:.1f}MB)")
