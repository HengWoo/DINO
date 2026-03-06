/**
 * Point cloud renderer for the 3D depth panel.
 * Loads PLY files (SLAM3R/COLMAP dense output) or binary .bin files
 * with accumulated per-frame point clouds.
 * Supports V2 binary format with per-vertex RGB colors (magic 0x44494E4F).
 */
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const FALLBACK_COLOR = [56 / 255, 189 / 255, 248 / 255]; // sky-400
const POINT_SIZE = 3;
const PLY_POINT_SIZE = 1;
const ACCUMULATE_FRAMES = 60; // keep last N frames for dense reconstruction
const SAT_BOOST = 1.3;
const GAMMA = 0.85;

const MAGIC_V2 = 0x44494E4F; // "DINO" in little-endian

export class PointCloudRenderer {
  constructor(scene, sceneWidth, sceneHeight) {
    this.scene = scene;
    this.sceneWidth = sceneWidth || 720;
    this.sceneHeight = sceneHeight || 1280;
    this.buffer = null;
    this.numFrames = 0;
    this.frameIndex = [];
    this.points = null;
    this._geometry = null;
    this._material = null;
    this._scaleFactor = 0;
    this._frameCache = new Map(); // frameIdx -> { positions, colors }
    this._lastIdx = -1;
    this._hasRgb = false;
    this._headerSize = 8; // legacy default
    this._center = null; // {x, y, z} in scaled Three.js coords
    this._bounds = null; // {min: {x,y,z}, max: {x,y,z}} in scaled coords
    this._isPLY = false;
  }

  async loadBinary(url) {
    console.log('[PointCloud] Fetching:', url);
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`Failed to load point cloud: ${resp.status}`);
    this.buffer = await resp.arrayBuffer();
    console.log('[PointCloud] Loaded:', (this.buffer.byteLength / 1024 / 1024).toFixed(1), 'MB');
    this._parseIndex();
    console.log('[PointCloud] Frames:', this.numFrames, 'RGB:', this._hasRgb);
    this._computeScale();
    this._initPoints();
  }

  async loadPLY(url) {
    console.log('[PointCloud] Loading PLY:', url);
    const loader = new PLYLoader();
    const geometry = await new Promise((resolve, reject) => {
      loader.load(url, resolve, undefined, reject);
    });

    const posAttr = geometry.getAttribute('position');
    if (!posAttr) {
      throw new Error('PLY file has no vertex position data — file may be corrupt');
    }
    const colAttr = geometry.getAttribute('color');
    const N = posAttr.count;
    console.log(`[PointCloud] PLY loaded: ${N} points, hasColor: ${!!colAttr}`);

    if (N === 0) {
      console.warn('[PointCloud] PLY file contains no points');
      this._center = { x: 0, y: 0, z: 0 };
      this._bounds = { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
      return;
    }

    // Compute bounding box for auto-scale
    geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    const span = Math.max(
      bb.max.x - bb.min.x,
      bb.max.y - bb.min.y,
      bb.max.z - bb.min.z,
      0.01
    );
    const s = Math.min(this.sceneWidth, this.sceneHeight) * 0.35 / span;
    this._scaleFactor = s;

    // Auto-detect up axis: heuristic assumes the axis with smallest variance
    // is the vertical (thinnest) dimension in SLAM reconstructions.
    // Remap that axis to Three.js Y (up).
    const sampleStep = Math.max(1, Math.floor(N / 2000));
    let sx = 0, sy = 0, sz = 0, sx2 = 0, sy2 = 0, sz2 = 0, cnt = 0;
    for (let i = 0; i < N; i += sampleStep) {
      const px = posAttr.getX(i), py = posAttr.getY(i), pz = posAttr.getZ(i);
      sx += px; sy += py; sz += pz;
      sx2 += px * px; sy2 += py * py; sz2 += pz * pz;
      cnt++;
    }
    const varX = sx2 / cnt - (sx / cnt) ** 2;
    const varY = sy2 / cnt - (sy / cnt) ** 2;
    const varZ = sz2 / cnt - (sz / cnt) ** 2;
    const minVar = Math.min(varX, varY, varZ);
    console.log(`[PointCloud] Axis variance: X=${varX.toFixed(3)}, Y=${varY.toFixed(3)}, Z=${varZ.toFixed(3)}`);

    const positions = new Float32Array(N * 3);
    if (minVar === varY) {
      // Y has smallest variance — assume CV Y-down convention: flip Y and Z for Three.js Y-up
      console.log('[PointCloud] Detected Y-up (CV convention)');
      for (let i = 0; i < N; i++) {
        positions[i * 3]     =  posAttr.getX(i) * s;
        positions[i * 3 + 1] = -posAttr.getY(i) * s;
        positions[i * 3 + 2] = -posAttr.getZ(i) * s;
      }
    } else if (minVar === varZ) {
      // Z has smallest variance: remap (x, z, -y)
      console.log('[PointCloud] Detected Z-up');
      for (let i = 0; i < N; i++) {
        positions[i * 3]     =  posAttr.getX(i) * s;
        positions[i * 3 + 1] =  posAttr.getZ(i) * s;
        positions[i * 3 + 2] = -posAttr.getY(i) * s;
      }
    } else {
      // X has smallest variance (uncommon — catch-all fallback): remap (y, x, z)
      console.log('[PointCloud] Detected X-up');
      for (let i = 0; i < N; i++) {
        positions[i * 3]     =  posAttr.getY(i) * s;
        positions[i * 3 + 1] =  posAttr.getX(i) * s;
        positions[i * 3 + 2] =  posAttr.getZ(i) * s;
      }
    }

    // Process colors with saturation boost + gamma
    const colors = new Float32Array(N * 3);
    if (colAttr) {
      for (let i = 0; i < N; i++) {
        let r = colAttr.getX(i);
        let g = colAttr.getY(i);
        let b = colAttr.getZ(i);
        // Saturation boost
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        r = Math.min(1, gray + (r - gray) * SAT_BOOST);
        g = Math.min(1, gray + (g - gray) * SAT_BOOST);
        b = Math.min(1, gray + (b - gray) * SAT_BOOST);
        // Gamma correction
        colors[i * 3]     = Math.pow(Math.max(0, r), GAMMA);
        colors[i * 3 + 1] = Math.pow(Math.max(0, g), GAMMA);
        colors[i * 3 + 2] = Math.pow(Math.max(0, b), GAMMA);
      }
    } else {
      for (let i = 0; i < N * 3; i += 3) {
        colors[i]     = FALLBACK_COLOR[0];
        colors[i + 1] = FALLBACK_COLOR[1];
        colors[i + 2] = FALLBACK_COLOR[2];
      }
    }

    // Compute center and bounds in transformed coords
    let cx = 0, cy = 0, cz = 0;
    let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity, mnZ = Infinity, mxZ = -Infinity;
    for (let i = 0; i < N; i++) {
      const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
      cx += px; cy += py; cz += pz;
      if (px < mnX) mnX = px; if (px > mxX) mxX = px;
      if (py < mnY) mnY = py; if (py > mxY) mxY = py;
      if (pz < mnZ) mnZ = pz; if (pz > mxZ) mxZ = pz;
    }
    this._center = { x: cx / N, y: cy / N, z: cz / N };
    this._bounds = { min: { x: mnX, y: mnY, z: mnZ }, max: { x: mxX, y: mxY, z: mxZ } };

    // Initialize points mesh with smaller point size for dense PLY
    this._initPoints();
    this._material.size = PLY_POINT_SIZE;
    this._material.opacity = 0.85;
    this._geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this._geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this._geometry.computeBoundingSphere();

    this._isPLY = true;
    console.log(`[PointCloud] PLY ready: scale=${s.toFixed(2)}, center=(${this._center.x.toFixed(0)}, ${this._center.y.toFixed(0)}, ${this._center.z.toFixed(0)})`);
  }

  _parseIndex() {
    const view = new DataView(this.buffer);
    const magic = view.getUint32(0, true);
    let indexOffset;

    if (magic === MAGIC_V2) {
      // V2 header: u32 magic, u16 version, u16 flags, u32 indexOffset
      this._headerSize = 12;
      const version = view.getUint16(4, true);
      if (version !== 2) {
        throw new Error(`Unsupported point cloud format version: ${version}. Expected 2.`);
      }
      const flags = view.getUint16(6, true);
      this._hasRgb = (flags & 1) !== 0;
      indexOffset = view.getUint32(8, true);
      if (indexOffset < this._headerSize || indexOffset > this.buffer.byteLength) {
        throw new Error(`Corrupted point cloud: invalid index offset ${indexOffset} (file size ${this.buffer.byteLength})`);
      }
      // Count frames: each index entry is 12 bytes
      this.numFrames = Math.floor((this.buffer.byteLength - indexOffset) / 12);
    } else {
      // Legacy header: u32 numFrames, u32 indexOffset
      this._headerSize = 8;
      this._hasRgb = false;
      this.numFrames = view.getUint32(0, true);
      indexOffset = view.getUint32(4, true);
    }

    // Read frame index (shared between V1 and V2)
    this.frameIndex = [];
    for (let i = 0; i < this.numFrames; i++) {
      const entryOffset = indexOffset + i * 12;
      const offsetLow = view.getUint32(entryOffset, true);
      const numPoints = view.getUint32(entryOffset + 8, true);
      this.frameIndex.push({ offset: offsetLow, numPoints });
    }
  }

  _computeScale() {
    const sampled = [0, Math.floor(this.numFrames / 2), this.numFrames - 1];
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (const fi of sampled) {
      const entry = this.frameIndex[fi];
      if (!entry || entry.numPoints === 0) continue;
      const dataOffset = entry.offset + 4; // skip numPoints u32
      const end = dataOffset + entry.numPoints * 12;
      if (end > this.buffer.byteLength) {
        console.warn(`[PointCloud] Frame ${fi}: data extends beyond buffer`);
        continue;
      }
      // Use slice() for alignment safety — V2 RGB bytes can cause unaligned offsets
      const fa = new Float32Array(this.buffer.slice(dataOffset, end));
      for (let i = 0; i < entry.numPoints; i++) {
        const b = i * 3;
        if (fa[b] < minX) minX = fa[b]; if (fa[b] > maxX) maxX = fa[b];
        if (fa[b+1] < minY) minY = fa[b+1]; if (fa[b+1] > maxY) maxY = fa[b+1];
        if (fa[b+2] < minZ) minZ = fa[b+2]; if (fa[b+2] > maxZ) maxZ = fa[b+2];
      }
    }
    if (minX === Infinity) {
      console.warn('[PointCloud] No valid points found in sampled frames; defaulting scale to 1');
      this._scaleFactor = 1;
      this._center = { x: 0, y: 0, z: 0 };
      return;
    }
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.01);
    this._scaleFactor = Math.min(this.sceneWidth, this.sceneHeight) * 0.35 / span;
    // Center in Three.js coords: (x*s, z*s, -y*s)
    const s = this._scaleFactor;
    this._center = {
      x: ((minX + maxX) / 2) * s,
      y: ((minZ + maxZ) / 2) * s,
      z: -((minY + maxY) / 2) * s,
    };
    console.log('[PointCloud] Full extent:', span.toFixed(2), 'scale:', this._scaleFactor.toFixed(1),
      'center:', this._center.x.toFixed(0), this._center.y.toFixed(0), this._center.z.toFixed(0));
  }

  _initPoints() {
    // Clean up previous mesh if re-initializing (e.g., switching PLY → BIN)
    if (this.points) this.scene.remove(this.points);
    if (this._geometry) this._geometry.dispose();
    if (this._material) this._material.dispose();

    this._material = new THREE.PointsMaterial({
      size: POINT_SIZE,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.7,
    });
    this._geometry = new THREE.BufferGeometry();
    this._geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(0), 3)
    );
    this._geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(new Float32Array(0), 3)
    );
    this.points = new THREE.Points(this._geometry, this._material);
    this.scene.add(this.points);
  }

  /**
   * Read and scale a single frame's points + colors (cached).
   */
  _getFramePoints(idx) {
    if (this._frameCache.has(idx)) return this._frameCache.get(idx);
    const entry = this.frameIndex[idx];
    if (!entry || entry.numPoints === 0) return null;

    const N = entry.numPoints;
    const dataOffset = entry.offset + 4; // skip numPoints u32
    const xyzEnd = dataOffset + N * 12;
    if (xyzEnd > this.buffer.byteLength) {
      console.warn(`[PointCloud] Frame ${idx}: XYZ data truncated`);
      return null;
    }
    // Use slice() for alignment safety — V2 RGB bytes can cause unaligned offsets
    const raw = new Float32Array(this.buffer.slice(dataOffset, xyzEnd));
    const s = this._scaleFactor || 1;
    const positions = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const b = i * 3;
      positions[b] = raw[b] * s;           // x
      positions[b + 1] = raw[b + 2] * s;   // z -> y (up)
      positions[b + 2] = -raw[b + 1] * s;  // -y -> z
    }

    let colors = null;
    if (this._hasRgb) {
      const rgbOffset = entry.offset + 4 + N * 12; // after numPoints u32 + float32[N*3]
      const rgbEnd = rgbOffset + N * 3;
      if (rgbEnd > this.buffer.byteLength) {
        console.warn(`[PointCloud] Frame ${idx}: RGB data truncated`);
      } else {
        const rgbRaw = new Uint8Array(this.buffer.slice(rgbOffset, rgbEnd));
        colors = new Float32Array(N * 3);
        for (let i = 0; i < N * 3; i++) {
          colors[i] = rgbRaw[i] / 255;
        }
      }
    }

    // Evict old cache entries
    if (this._frameCache.size > ACCUMULATE_FRAMES + 10) {
      const oldest = this._frameCache.keys().next().value;
      this._frameCache.delete(oldest);
    }
    const result = { positions, colors };
    this._frameCache.set(idx, result);
    return result;
  }

  /**
   * Update point cloud — accumulate last N frames for dense reconstruction.
   */
  updateFrame(frameIdx) {
    if (this._isPLY) return; // static PLY — no per-frame updates
    if (!this.buffer || !this.frameIndex.length || !this._geometry) return;
    const idx = Math.min(frameIdx, this.frameIndex.length - 1);
    if (idx === this._lastIdx) return;
    this._lastIdx = idx;

    // Gather points from last ACCUMULATE_FRAMES frames
    const startIdx = Math.max(0, idx - ACCUMULATE_FRAMES + 1);
    const frames = [];
    let totalPoints = 0;
    for (let f = startIdx; f <= idx; f++) {
      try {
        const data = this._getFramePoints(f);
        if (data) {
          frames.push(data);
          totalPoints += data.positions.length / 3;
        }
      } catch (err) {
        console.warn(`[PointCloud] Failed to read frame ${f}:`, err.message);
      }
    }

    if (totalPoints === 0) {
      this._geometry.setAttribute(
        'position', new THREE.BufferAttribute(new Float32Array(0), 3)
      );
      this._geometry.setAttribute(
        'color', new THREE.BufferAttribute(new Float32Array(0), 3)
      );
      this._geometry.attributes.position.needsUpdate = true;
      this._geometry.attributes.color.needsUpdate = true;
      return;
    }

    // Merge all frame arrays
    const mergedPos = new Float32Array(totalPoints * 3);
    const mergedCol = new Float32Array(totalPoints * 3);
    let offset = 0;
    for (const frame of frames) {
      mergedPos.set(frame.positions, offset);
      if (frame.colors) {
        mergedCol.set(frame.colors, offset);
      } else {
        // Fallback: sky-blue for legacy frames
        const n = frame.positions.length;
        for (let i = 0; i < n; i += 3) {
          mergedCol[offset + i] = FALLBACK_COLOR[0];
          mergedCol[offset + i + 1] = FALLBACK_COLOR[1];
          mergedCol[offset + i + 2] = FALLBACK_COLOR[2];
        }
      }
      offset += frame.positions.length;
    }

    this._geometry.setAttribute(
      'position', new THREE.BufferAttribute(mergedPos, 3)
    );
    this._geometry.setAttribute(
      'color', new THREE.BufferAttribute(mergedCol, 3)
    );
    this._geometry.attributes.position.needsUpdate = true;
    this._geometry.attributes.color.needsUpdate = true;
    this._geometry.computeBoundingSphere();
  }

  get scaleFactor() {
    return this._scaleFactor;
  }

  get center() {
    return this._center;
  }

  get bounds() {
    return this._bounds;
  }

  dispose() {
    if (this.points) this.scene.remove(this.points);
    if (this._geometry) this._geometry.dispose();
    if (this._material) this._material.dispose();
    this.buffer = null;
    this.frameIndex = [];
    this._frameCache.clear();
  }
}
