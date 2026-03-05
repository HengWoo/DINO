# 3D Environment Reconstruction Options for Spatial Video Viewer

**Date**: 2026-03-05
**Context**: Restaurant walkthrough video, 24 seconds, 720x1280, processed through DINO spatial pipeline
**Current baseline**: Per-frame sparse point clouds (~13K points/frame) from Depth-Anything-V2-Small with ORB-based ego-motion, accumulated over a sliding window, rendered in Three.js

---

## Table of Contents

1. [Current System Baseline](#1-current-system-baseline)
2. [Gaussian Splatting (3DGS)](#2-gaussian-splatting-3dgs)
3. [NeRF Variants](#3-nerf-variants)
4. [TSDF Fusion](#4-tsdf-fusion)
5. [Dense Monocular Depth Upgrades](#5-dense-monocular-depth-upgrades)
6. [Textured Mesh Reconstruction](#6-textured-mesh-reconstruction)
7. [Hybrid Approaches](#7-hybrid-approaches)
8. [Recommendation Matrix](#8-recommendation-matrix)
9. [Recommended Implementation Path](#9-recommended-implementation-path)

---

## 1. Current System Baseline

**What we have today:**

- **Depth model**: Depth-Anything-V2-Small (25M params, Apache 2.0 license)
- **Depth type**: Relative (affine-invariant), normalized to [0, 1] per frame
- **Point cloud**: ~13K points per frame, sampled on an 8-pixel grid from 720x1280 depth maps
- **Ego-motion**: ORB feature matching with essential matrix decomposition
- **Accumulation**: Sliding window of recent frames transformed to world coordinates
- **Renderer**: Three.js `BufferGeometry` with `Points` material
- **Writer**: Custom binary `.bin` format (see `depth_cloud_writer.py`)

**Key limitations:**
- Relative depth means no consistent scale across frames -- points "breathe" as depth normalization shifts
- ORB ego-motion drifts over time, causing ghosting and misalignment
- Sparse sampling (every 8th pixel) loses fine geometric detail
- Point cloud rendering has visible gaps; no surface interpolation
- No color/texture information in the point cloud

---

## 2. Gaussian Splatting (3DGS)

### Overview

3D Gaussian Splatting represents scenes as collections of 3D Gaussian primitives, each with position, covariance, color (via spherical harmonics), and opacity. Trained via differentiable rendering from multi-view images, it achieves photorealistic novel view synthesis at real-time frame rates.

### Quality Level

- **Photorealistic** for well-captured scenes with sufficient viewpoint coverage
- Handles view-dependent effects (specular reflections on restaurant glass/metal)
- Significantly superior to point clouds -- produces continuous, artifact-free renders
- Quality degrades in regions with sparse viewpoint coverage (e.g., if the camera only passes through once in a straight line, side views will have artifacts)

### Compute Requirements

**COLMAP preprocessing (camera poses + sparse SfM):**
- For a 24-second video at 30fps (~720 frames), COLMAP feature extraction + matching + bundle adjustment: **15-45 minutes** on RTX 3060
- Can subsample to every 3rd-5th frame (~150-240 images) to speed up without significant quality loss
- COLMAP is CPU-bound for bundle adjustment; GPU helps with feature extraction

**3DGS training on RTX 3060 12GB:**
- 30,000 iterations: ~30 minutes
- 7,000-15,000 iterations (reduced, still good quality): ~10-15 minutes
- VRAM usage: fits comfortably at 720p input resolution (auto-rescales if width > 1600px)
- The 12GB VRAM on the 3060 is adequate for this scene scale

**Cloud (A100/H100 on Modal):**
- COLMAP: ~5-10 minutes
- 3DGS training: ~5-8 minutes for 30K iterations
- Total wall-clock including data upload: ~15-20 minutes

### Three.js Viewer Integration

Two mature Three.js Gaussian splat renderers exist:

**[GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)** (Mark Kellogg):
- Most mature and widely used Three.js 3DGS renderer
- Supports `.ply`, `.splat`, and `.ksplat` (compressed) formats
- Progressive loading support
- CPU-based splat sorting (artifacts when moving fast, but acceptable for controlled viewpoints)
- In-memory compression to reduce footprint
- Integrates as a standard Three.js scene object

**[Spark v2.0](https://github.com/sparkjsdev/spark)** (sparkjs.dev):
- Newer, more advanced renderer with Level-of-Detail (LoD) streaming
- Supports `.ply`, `.spz`, `.splat`, `.ksplat`, `.sog` formats
- Designed for "huge worlds" with dynamic splats
- Integrates into Three.js rendering pipeline, fusing splats with mesh objects
- 98%+ WebGL2 device support
- Better suited if we want to combine splats with other Three.js geometry (zone overlays, annotations)

**Integration effort**: Medium. Both libraries provide npm packages. Load the exported `.ply`/`.splat` file, add to the Three.js scene, and the renderer handles the rest. The main work is replacing our current point cloud rendering code with the splat viewer component.

### Dependencies and Setup

- COLMAP (C++ build, or use `pycolmap` via pip)
- Original 3DGS implementation (CUDA, Python, PyTorch) or nerfstudio's Splatfacto
- `gsplat` library (nerfstudio's CUDA rasterizer) -- more memory efficient than original
- Export via `ns-export gaussian-splat` or directly from original implementation

### Pros
- Best visual quality of any approach listed here
- Real-time rendering in browser via WebGL
- Mature ecosystem with multiple Three.js renderers
- Can be trained on RTX 3060 in under an hour
- Actively developed community with continuous improvements

### Cons
- **COLMAP dependency** is the biggest pain point -- can fail on textureless or repetitive regions (common in restaurants with uniform walls/floors)
- Requires sufficient multi-view overlap; a single-pass walkthrough may not provide enough coverage for side-looking novel views
- Training produces large files (~50-200MB for a scene this size)
- CPU-based splat sorting in web renderers can cause frame drops during fast navigation
- Overfitting risk with sequential video frames (too similar viewpoints)

---

## 3. NeRF Variants

### Overview

Neural Radiance Fields encode a scene as a neural network that maps 3D coordinates + viewing direction to color + density. Multiple variants exist with different speed/quality tradeoffs.

### Nerfacto (nerfstudio's recommended method)

**Quality**: Good for real-world scenes. Incorporates per-image appearance embeddings (handles exposure changes), hash-grid encoding from Instant-NGP, and predicted normals.

**Training time on RTX 3060:**
- ~12 minutes for a standard scene
- Uses PyTorch (no custom CUDA required beyond tinycudann)
- Render speed: ~1 FPS (not real-time)

**Export options:**
- Point cloud export (similar to what we have now, but denser)
- Mesh export via marching cubes on the density field
- Cannot directly export to Gaussian splats (one-way conversion exists: NeRF-to-3DGS)

### Instant-NGP

**Quality**: Comparable to Nerfacto but faster training. The hash-grid encoding enables training in seconds to minutes.

**Compute**: Requires tinycudann (NVIDIA-only). RTX 3060 compatible. Training: 2-5 minutes.

**Limitation**: Nerfstudio's implementation diverges from the original. The original Instant-NGP binary from NVIDIA is fast but less flexible.

### Real-Time Rendering Problem

The fundamental issue with NeRFs for our use case: **NeRFs cannot render in real-time in a web browser.** They require neural network inference per pixel per frame. Options:

1. **Pre-render a fixed camera path** -- defeats the purpose of interactive exploration
2. **Export to mesh** -- loses view-dependent effects, quality drops significantly
3. **Convert NeRF to 3DGS** -- nerfstudio supports this (`nerf2gs`), but adds another conversion step
4. **WebGL NeRF renderers** -- exist but are extremely slow (< 1 FPS) and impractical

### Pros
- Well-understood pipeline via nerfstudio
- Nerfacto handles varying lighting/exposure well
- Can export meshes for offline use
- Lower memory footprint than 3DGS during training

### Cons
- **No practical real-time web rendering** -- this is a dealbreaker for our Three.js viewer
- Mesh export quality is significantly lower than direct NeRF rendering
- Still requires COLMAP for camera poses
- Nerfstudio dependency stack is heavy (torch, tinycudann, nerfacc, etc.)
- Converting NeRF to 3DGS adds complexity; better to train 3DGS directly

### Verdict

**Not recommended as a primary approach.** The lack of real-time web rendering makes NeRFs impractical for our interactive Three.js viewer. If real-time rendering is not needed (e.g., pre-rendered flythrough video), Nerfacto is a solid option, but 3DGS dominates for interactive use.

---

## 4. TSDF Fusion

### Overview

Truncated Signed Distance Function (TSDF) integration fuses multiple depth maps into a volumetric representation, then extracts a triangle mesh via Marching Cubes. This is the classical approach used by RGB-D reconstruction systems (KinectFusion, etc.).

### Quality Level

- Produces smooth, watertight triangle meshes
- Quality depends entirely on input depth map quality and camera pose accuracy
- With our current relative depth maps: **poor** (TSDF needs metric depth with consistent scale)
- With metric depth models: **moderate to good** for geometry, texture quality depends on UV mapping
- Thin structures (table legs, chair backs) are problematic -- TSDF has known issues with thin geometry
- Not photorealistic -- produces a textured mesh, not radiance-field quality

### Compute Requirements

**Open3D TSDF integration:**
- GPU-accelerated via Open3D's tensor API
- Processing 720 frames of 720x1280 depth maps: ~2-5 minutes on RTX 3060
- Marching Cubes mesh extraction: seconds
- Voxel resolution tradeoff: finer voxels = more detail but more memory. At 1cm resolution, a restaurant-sized room (~10m x 10m x 3m) = ~300M voxels, which exceeds 12GB. Use 2-5cm voxels.
- ~100Hz integration rate achievable on a GTX 1070 (faster on 3060)

**Alternatives to Open3D:**
- **voxblox** (C++, ROS-oriented): faster but harder to integrate with Python
- **fVDB** (OpenVDB-based): sparse TSDF, more memory efficient for large scenes
- **RGBTSDF**: improved texture quality via depth interpolation and texture constraints

### Integration with Our Existing Pipeline

This is the most straightforward upgrade path because:
- We already produce depth maps per frame
- We already have camera poses (ego-motion)
- TSDF fusion directly consumes depth maps + poses
- Output is a triangle mesh, which Three.js renders natively and efficiently

**Critical prerequisite**: Our depth maps must be **metric** (consistent scale across frames). Our current relative depth normalization breaks TSDF fusion. We would need to upgrade to a metric depth model first (see Section 5).

### Three.js Integration

- Export mesh as `.glb`/`.gltf` with textures
- Three.js `GLTFLoader` handles this natively
- Mesh rendering is far more efficient than point clouds (GPU-optimized triangle rasterization)
- Can add normal maps, ambient occlusion, etc. for enhanced visual quality
- Supports LOD (level of detail) for large meshes

### Dependencies

- `open3d` (pip install, well-maintained)
- Metric depth estimator (see Section 5)
- MeshLab or Open3D for texture baking (optional)

### Pros
- Directly uses our existing depth map pipeline
- Fast processing (minutes, not hours)
- Output is standard triangle mesh -- trivial Three.js integration
- Well-understood, mature technique
- Low dependency count

### Cons
- Quality ceiling is lower than 3DGS or NeRF
- Requires metric depth (upgrade from our current relative depth)
- Texture quality depends on separate UV mapping/baking step
- Thin structures and fine details are lost
- Voxel resolution is a hard tradeoff between detail and memory
- No view-dependent effects (no specular highlights, reflections)

---

## 5. Dense Monocular Depth Upgrades

### Current State: Depth-Anything-V2-Small

- 25M parameters, ViT-S backbone
- Apache 2.0 license (commercial OK)
- Relative/affine-invariant depth only
- Fast inference, low VRAM usage
- Good generalization but limited fine detail

### Upgrade Option 1: Depth-Anything-V2-Large

| Attribute | Small (current) | Large |
|-----------|-----------------|-------|
| Parameters | 25M | 335M |
| License | Apache 2.0 | CC-BY-NC-4.0 (non-commercial) |
| Depth quality | Good | Significantly better fine detail |
| Inference speed (3060) | ~30ms/frame | ~150-200ms/frame |
| VRAM | ~1GB | ~4-6GB |
| Output type | Relative depth | Relative depth |

**Verdict**: Easy swap (same API, just change `model_id`). Noticeably better depth maps. But still relative depth -- does not solve the scale consistency problem for TSDF or accumulated point clouds. The CC-BY-NC-4.0 license is a constraint if this becomes a commercial product.

### Upgrade Option 2: Video Depth Anything (CVPR 2025 Highlight)

This is the most relevant upgrade for our video pipeline:

- Built on Depth-Anything-V2 with a **spatiotemporal head** that enforces temporal consistency
- Processes arbitrarily long videos without quality degradation
- **Key-frame-based inference strategy** for long videos
- Smallest model runs at **30 FPS** in real-time
- Achieves 0.944 delta-1 on KITTI vs. 0.815 for DAv2-Large
- Temporal consistency (TAE): 0.570 on ScanNet vs. 1.140 for DAv2-Large (2x better)

**Impact on our pipeline**: Would eliminate the "breathing" artifacts from per-frame depth normalization. Temporally consistent depth means accumulated point clouds align much better without explicit scale correction.

**Compute**: Similar to DAv2-Large per frame, but processes video chunks for temporal modeling. RTX 3060 feasible with the Small variant.

### Upgrade Option 3: Metric Depth Models

For TSDF fusion or any approach requiring absolute scale:

**ZoeDepth:**
- Extends MiDaS with adaptive metric binning
- Predicts metric depth with scene-aware routing (indoor/outdoor)
- Requires camera intrinsics
- Good zero-shot generalization
- Moderate quality, showing its age (2023)

**UniDepth / UniDepthV2 (2025):**
- Predicts metric 3D point clouds without camera intrinsics (self-predicts camera parameters)
- Best zero-shot cross-domain generalization
- Outperforms Metric3D on NYU and KITTI benchmarks
- ~3GB VRAM for inference
- RTX 3060 feasible
- V2 simplifies the architecture further

**Metric3Dv2:**
- Joint metric depth + surface normal estimation
- Canonical camera space normalization (handles different focal lengths)
- Champion of CVPR 2023 Monocular Depth Challenge
- Slightly more efficient decoder than UniDepth
- Requires more training data but produces normals (useful for mesh reconstruction)

**Apple DepthPro:**
- Pixel-perfect metric depth with sharp boundaries
- Outperforms Metric3D v2 and Depth Anything on in-the-wild scenes
- Higher compute cost
- Apple license restrictions

**Recommendation**: **UniDepthV2** for metric depth. No intrinsics required (simplifies our pipeline), best generalization, and RTX 3060 feasible. Combine with Video Depth Anything for temporal consistency if needed.

### Upgrade Option 4: DUSt3R / MASt3R / MUSt3R

These are not just depth estimators -- they are **multi-view geometric foundation models** that jointly solve depth, camera poses, and 3D reconstruction from unposed images.

**DUSt3R** (2024):
- Takes pairs of images, outputs dense 3D pointmaps
- No camera calibration or poses needed
- Pairwise only; requires global optimization for multi-view

**MASt3R** (2024):
- Adds metric pointmaps + matching head to DUSt3R
- Handles thousands of images
- 30% improvement over prior art on localization benchmarks

**MUSt3R** (CVPR 2025):
- Symmetric architecture, processes all views in a common frame directly
- Multi-layer memory mechanism for scaling to large collections
- Works both offline and online (visual SLAM compatible)
- **High frame-rate inference** for thousands of pointmaps
- State-of-the-art on visual odometry, relative pose, scale estimation, and multi-view depth

**MV-DUSt3R+** (CVPR 2025):
- Single-pass feed-forward for multiple views
- Reconstructs a room from 12 views in 0.89 seconds, multi-room from 20 views in 1.54 seconds

**Impact on our pipeline**: MUSt3R could **replace both our ego-motion estimation AND depth estimation** in one model. It would give us:
- Metric-scale dense 3D pointmaps
- Accurate camera poses (no ORB drift)
- Multi-view consistent geometry

**Compute on RTX 3060:**
- DUSt3R/MASt3R: ~8-12GB VRAM for pairs of 512x512 images. Tight fit on 12GB 3060 at our resolution (720x1280 would need downsampling or tiling).
- MUSt3R: similar baseline but more efficient for many views due to memory mechanism.
- Processing 720 frames: subsample to ~50-100 keyframes, run pairwise or multi-view. Estimate: 10-30 minutes on 3060 depending on subsampling.
- Cloud (A100): 2-5 minutes for the full sequence.

### Processing Time Summary (24-second 720x1280 video, ~720 frames)

| Model | RTX 3060 Time | Output |
|-------|--------------|--------|
| DAv2-Small (current) | ~22s (30ms/frame) | Relative depth |
| DAv2-Large | ~2.5min (200ms/frame) | Relative depth, finer |
| Video Depth Anything-S | ~30s | Temporally consistent relative depth |
| Video Depth Anything-L | ~3min | Temporally consistent relative depth |
| UniDepthV2 | ~3-5min | Metric depth + 3D points |
| MUSt3R (50 keyframes) | ~15-30min | Metric pointmaps + poses |

---

## 6. Textured Mesh Reconstruction

### Poisson Surface Reconstruction

Given a dense, oriented point cloud, Poisson reconstruction produces a smooth, watertight triangle mesh.

**Pipeline:**
1. Generate dense point cloud (from upgraded depth or DUSt3R/MASt3R)
2. Estimate + orient normals (Open3D `estimate_normals` + `orient_normals_consistent_tangent_plane`)
3. Run Poisson reconstruction (`create_from_point_cloud_poisson`)
4. Prune low-density vertices (remove spurious outer shell)
5. Generate UV atlas (MeshLab or xatlas)
6. Bake vertex colors / project images to texture atlas
7. Export as `.glb` with texture

**Quality**: Smooth surfaces, but Poisson always creates closed/watertight meshes which means phantom geometry where there are no observations. The density-based pruning step is critical but imperfect. Fine details are smoothed out.

**Compute on RTX 3060:**
- Point cloud generation: depends on depth model (see Section 5)
- Poisson reconstruction: CPU-bound, 1-5 minutes for ~1M points (octree depth 10-12)
- Texture baking: 2-10 minutes depending on atlas resolution

### Ball Pivoting Algorithm (BPA)

Alternative to Poisson that does not create closed meshes. Better for partial reconstructions (like a single-pass walkthrough where you only see one side of things). Available in Open3D and MeshLab.

### Texture Mapping Approaches

**Per-vertex coloring**: Simplest. Store RGB per vertex from the source image at back-projected pixel. No UV mapping needed. Three.js supports vertex colors natively. Quality is limited by mesh vertex density.

**Texture atlas projection**: Project source images onto the mesh, stitching a texture atlas from the best-viewing-angle image per triangle. Higher quality but complex:
- Requires accurate camera poses for reprojection
- Seam artifacts at atlas boundaries
- MeshLab's "Transfer: Vertex Attributes to Texture" filter handles this
- Open3D does not have built-in texture atlas (requires external tools)

**Neural texture baking**: Newer approaches (e.g., from nerfstudio mesh exports) can bake NeRF-quality colors into mesh textures, but this is more complex.

### Three.js Rendering

- `GLTFLoader` for `.glb` meshes with textures -- standard, well-optimized
- `MeshStandardMaterial` with diffuse + normal maps
- Orders of magnitude more efficient than point cloud rendering
- Supports shadows, ambient occlusion, environment maps
- LOD via `THREE.LOD` for performance optimization
- Draco compression for mesh geometry (reduces file size 10-20x)

### Pros
- Standard triangle mesh -- universally supported, well-optimized rendering
- Texture mapping provides decent visual quality
- Small file sizes with Draco compression
- Can add PBR materials for enhanced realism
- Easy to integrate with existing Three.js scene (zone overlays, annotations, etc.)

### Cons
- Quality ceiling well below 3DGS
- Poisson reconstruction artifacts (phantom geometry, over-smoothing)
- Texture atlas generation is fiddly -- seams, blurring, projection errors
- No view-dependent effects
- Multiple tool pipeline (Open3D -> MeshLab -> export)
- Quality highly dependent on input point cloud density and accuracy

---

## 7. Hybrid Approaches

### Hybrid A: MUSt3R Poses + Gaussian Splatting (Recommended)

**Pipeline:**
1. Extract keyframes from video (~100-200 frames)
2. Run MUSt3R for metric camera poses + dense pointmaps (replaces COLMAP)
3. Initialize 3DGS with MUSt3R pointmaps as seed points
4. Train Gaussian Splatting using MUSt3R poses
5. Export `.ply`/`.splat` file
6. Render in Three.js via GaussianSplats3D or Spark

**Why this is compelling:**
- Eliminates COLMAP entirely (the biggest pain point of standard 3DGS)
- MUSt3R provides metric-scale poses and dense initialization
- 3DGS provides photorealistic rendering
- Both are actively developed (MUSt3R: CVPR 2025, Spark: 2025)

**Prior art**: **InstantSplat** already demonstrated this approach with DUSt3R + 3DGS, achieving 30x speedup over COLMAP-based pipelines and building full scenes in under 1 minute. MUSt3R improves on DUSt3R with native multi-view support.

**Compute (RTX 3060):**
- MUSt3R pose estimation: 15-30 minutes
- 3DGS training: 15-30 minutes
- Total: ~30-60 minutes

**Compute (A100 on Modal):**
- MUSt3R: 2-5 minutes
- 3DGS: 5-8 minutes
- Total: ~10-15 minutes

### Hybrid B: Video Depth Anything + TSDF Fusion (Quick Win)

**Pipeline:**
1. Run Video Depth Anything on the full video (temporally consistent depth)
2. Use existing ORB ego-motion (or upgrade to MUSt3R poses)
3. Feed depth maps + poses into Open3D TSDF integration
4. Extract mesh via Marching Cubes
5. Bake textures from source frames
6. Load textured mesh in Three.js

**Why this is compelling:**
- Minimal changes to our existing pipeline (swap depth model, add TSDF step)
- Video Depth Anything is a drop-in replacement for per-frame depth estimation
- TSDF fusion is well-understood and fast
- Produces a standard mesh that Three.js handles efficiently

**Limitation**: Still relative depth (not metric) unless we add scale estimation. Temporal consistency helps but doesn't solve absolute scale. Could use UniDepthV2 instead for metric depth, but loses the temporal consistency advantage.

**Compute (RTX 3060):**
- Video Depth Anything: 30 seconds - 3 minutes
- TSDF fusion: 2-5 minutes
- Texture baking: 5-10 minutes
- Total: ~10-20 minutes

### Hybrid C: DUSt3R Dense Points + Poisson Mesh (Moderate Effort)

**Pipeline:**
1. Extract ~50-100 keyframes
2. Run MASt3R/MUSt3R for dense metric pointmaps
3. Merge pointmaps into unified point cloud
4. Run Poisson surface reconstruction
5. Texture atlas from source images
6. Export to Three.js

**Compute (RTX 3060):** 20-40 minutes total

### Hybrid D: COLMAP-Free Gaussian Splatting (CF-3DGS)

The CF-3DGS method (CVPR 2024) processes video frames sequentially, growing Gaussians without any SfM preprocessing. It estimates relative poses between consecutive frames using local Gaussian transformations.

**Pros**: No COLMAP, no separate pose estimation step, processes video directly
**Cons**: Quality slightly below COLMAP-initialized 3DGS, sequential processing is slower, less tested on long sequences

---

## 8. Recommendation Matrix

| Approach | Quality | RTX 3060 Feasible | Processing Time (3060) | Viewer Integration | Setup Complexity | Priority |
|----------|---------|-------------------|----------------------|-------------------|------------------|----------|
| **3DGS (COLMAP)** | Excellent | Yes | 45-75 min | Easy (Spark/GS3D) | Medium (COLMAP) | 2 |
| **Hybrid A: MUSt3R + 3DGS** | Excellent | Yes (tight) | 30-60 min | Easy (Spark/GS3D) | Medium | **1 (Top)** |
| **Hybrid B: VDA + TSDF** | Good | Yes | 10-20 min | Trivial (GLTF) | Low | **1 (Quick win)** |
| **NeRF (Nerfacto)** | Good | Yes | 15-20 min | Hard (no RT web) | Medium | 5 (Skip) |
| **TSDF Fusion (metric depth)** | Moderate-Good | Yes | 10-15 min | Trivial (GLTF) | Low | 3 |
| **Dense Depth Upgrade (VDA)** | Moderate | Yes | 1-3 min | Same as current | Trivial | **1 (First step)** |
| **Textured Mesh (Poisson)** | Moderate | Yes | 20-40 min | Easy (GLTF) | Medium | 4 |
| **Hybrid C: DUSt3R + Poisson** | Good | Yes | 20-40 min | Easy (GLTF) | Medium | 3 |
| **CF-3DGS** | Good-Excellent | Yes | 45-90 min | Easy (Spark/GS3D) | Medium | 3 |

**Quality scale**: Moderate < Good < Excellent (photorealistic)

---

## 9. Recommended Implementation Path

### Phase 1: Immediate Upgrade (1-2 days effort)

**Swap Depth-Anything-V2-Small for Video Depth Anything.**

This is the lowest-effort, highest-impact change:
- Drop-in model swap in `depth_estimator.py`
- Temporally consistent depth maps eliminate "breathing" artifacts
- Immediate visible improvement in accumulated point cloud quality
- No pipeline architecture changes required
- RTX 3060: processes full 24s video in under 3 minutes

Code change in `src/dino/spatial/depth_estimator.py`:
```python
# Change from:
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
# To:
DEFAULT_MODEL = "depth-anything/Video-Depth-Anything-Small-hf"  # or equivalent
```

Also consider increasing point density by reducing `grid_step` from 8 to 4 in `depth_cloud_writer.py` (4x more points, ~52K per frame).

### Phase 2: Mesh Upgrade (3-5 days effort)

**Add TSDF fusion to produce textured meshes.**

- Add Open3D TSDF integration step after depth estimation
- Use Video Depth Anything output (or switch to UniDepthV2 for metric depth)
- Extract mesh via Marching Cubes
- Bake textures from source video frames
- Export as `.glb`, load with Three.js `GLTFLoader`
- Massive rendering performance improvement (mesh vs. point cloud)
- Good visual quality, standard format

### Phase 3: Photorealistic Rendering (1-2 weeks effort)

**Implement MUSt3R + Gaussian Splatting hybrid pipeline.**

- Replace ORB ego-motion with MUSt3R pose estimation
- Use MUSt3R dense pointmaps to initialize 3DGS
- Train Gaussian Splatting (via nerfstudio Splatfacto or original implementation)
- Export `.splat`/`.ply` file
- Integrate Spark v2.0 or GaussianSplats3D into Three.js viewer
- This achieves the "DIMENSIONAL spatial-computing" level of fidelity

**Cloud acceleration option**: Run MUSt3R + 3DGS training on Modal (A100) for ~10-15 minute turnaround. Use RTX 3060 for development/iteration.

### Phase 4: Polish (ongoing)

- Implement LOD streaming (Spark v2.0) for mobile/low-end devices
- Add scene editing capabilities (SuperSplat for cleaning up artifacts)
- Investigate NoPoSplat (ICLR 2025) for instant pose-free reconstruction
- Consider MV-DUSt3R+ for sub-second multi-view reconstruction as models improve

---

## References

### Gaussian Splatting & Viewers
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) -- Three.js 3DGS renderer
- [Spark v2.0](https://github.com/sparkjsdev/spark) -- Advanced Three.js 3DGS with LoD streaming
- [Original 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) -- INRIA reference implementation
- [gsplat](https://github.com/nerfstudio-project/gsplat) -- Nerfstudio's CUDA rasterizer
- [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) -- Nerfstudio's 3DGS method

### NeRF
- [Nerfstudio](https://docs.nerf.studio/) -- Modular NeRF framework
- [Nerfacto](https://docs.nerf.studio/nerfology/methods/nerfacto.html) -- Nerfstudio's recommended method

### Multi-View Stereo (DUSt3R family)
- [DUSt3R](https://arxiv.org/abs/2312.14132) -- Dense Unconstrained Stereo 3D Reconstruction
- [MASt3R](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/) -- Matching And Stereo 3D Reconstruction
- [MUSt3R](https://arxiv.org/abs/2503.01661) -- Multi-view Network for Stereo 3D Reconstruction (CVPR 2025)
- [MV-DUSt3R+](https://mv-dust3rp.github.io/) -- Multi-view extension (CVPR 2025)
- [InstantSplat](https://arxiv.org/abs/2403.20309) -- DUSt3R + 3DGS hybrid

### Depth Estimation
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) -- Foundation depth model
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) -- Temporally consistent video depth (CVPR 2025)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) -- Universal metric depth
- [Metric3Dv2](https://github.com/YvanYin/Metric3D) -- Metric depth + normals
- [DepthPro](https://learnopencv.com/depth-pro-monocular-metric-depth/) -- Apple's metric depth

### COLMAP-Free Methods
- [CF-3DGS](https://arxiv.org/abs/2312.07504) -- COLMAP-Free 3D Gaussian Splatting (CVPR 2024)
- [NoPoSplat](https://proceedings.iclr.cc/paper_files/paper/2025/file/857b34d81f0a8bfe3e18879dee3b5086-Paper-Conference.pdf) -- Pose-free Gaussian Splatting (ICLR 2025)

### TSDF & Mesh Reconstruction
- [Open3D TSDF Integration](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html)
- [Open3D Surface Reconstruction](https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html)
- [RGBTSDF](https://www.mdpi.com/2072-4292/16/17/3188) -- Improved color TSDF fusion
