/**
 * Point cloud renderer for the 3D depth panel.
 * Loads binary .bin files and renders accumulated per-frame point clouds.
 */
import * as THREE from 'three';

const POINT_COLOR = 0x38bdf8; // sky-400
const POINT_SIZE = 3;
const ACCUMULATE_FRAMES = 60; // keep last N frames for dense reconstruction

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
    this._frameCache = new Map(); // frameIdx -> Float32Array (scaled)
    this._lastIdx = -1;
  }

  async loadBinary(url) {
    console.log('[PointCloud] Fetching:', url);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load point cloud: ${resp.status}`);
    this.buffer = await resp.arrayBuffer();
    console.log('[PointCloud] Loaded:', (this.buffer.byteLength / 1024 / 1024).toFixed(1), 'MB');
    this._parseIndex();
    console.log('[PointCloud] Frames:', this.numFrames);
    this._computeScale();
    this._initPoints();
  }

  _parseIndex() {
    const view = new DataView(this.buffer);
    this.numFrames = view.getUint32(0, true);
    const indexOffset = view.getUint32(4, true);
    this.frameIndex = [];
    for (let i = 0; i < this.numFrames; i++) {
      const entryOffset = indexOffset + i * 12;
      const offsetLow = view.getUint32(entryOffset, true);
      const numPoints = view.getUint32(entryOffset + 8, true);
      this.frameIndex.push({ offset: offsetLow, numPoints });
    }
  }

  _computeScale() {
    // Sample a few frames to determine extent
    const sampled = [0, Math.floor(this.numFrames / 2), this.numFrames - 1];
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (const fi of sampled) {
      const entry = this.frameIndex[fi];
      if (!entry || entry.numPoints === 0) continue;
      const dataOffset = entry.offset + 4;
      const fa = new Float32Array(this.buffer, dataOffset, entry.numPoints * 3);
      for (let i = 0; i < entry.numPoints; i++) {
        const b = i * 3;
        if (fa[b] < minX) minX = fa[b]; if (fa[b] > maxX) maxX = fa[b];
        if (fa[b+1] < minY) minY = fa[b+1]; if (fa[b+1] > maxY) maxY = fa[b+1];
        if (fa[b+2] < minZ) minZ = fa[b+2]; if (fa[b+2] > maxZ) maxZ = fa[b+2];
      }
    }
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.01);
    this._scaleFactor = Math.min(this.sceneWidth, this.sceneHeight) * 0.35 / span;
    console.log('[PointCloud] Full extent:', span.toFixed(2), 'scale:', this._scaleFactor.toFixed(1));
  }

  _initPoints() {
    this._material = new THREE.PointsMaterial({
      size: POINT_SIZE,
      color: POINT_COLOR,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.7,
    });
    this._geometry = new THREE.BufferGeometry();
    this._geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(0), 3)
    );
    this.points = new THREE.Points(this._geometry, this._material);
    this.scene.add(this.points);
  }

  /**
   * Read and scale a single frame's points (cached).
   */
  _getFramePoints(idx) {
    if (this._frameCache.has(idx)) return this._frameCache.get(idx);
    const entry = this.frameIndex[idx];
    if (!entry || entry.numPoints === 0) return null;

    const dataOffset = entry.offset + 4;
    const raw = new Float32Array(this.buffer, dataOffset, entry.numPoints * 3);
    const s = this._scaleFactor || 1;
    const out = new Float32Array(entry.numPoints * 3);
    for (let i = 0; i < entry.numPoints; i++) {
      const b = i * 3;
      out[b] = raw[b] * s;           // x
      out[b + 1] = raw[b + 2] * s;   // z -> y (up)
      out[b + 2] = -raw[b + 1] * s;  // -y -> z
    }

    // Evict old cache entries
    if (this._frameCache.size > ACCUMULATE_FRAMES + 10) {
      const oldest = this._frameCache.keys().next().value;
      this._frameCache.delete(oldest);
    }
    this._frameCache.set(idx, out);
    return out;
  }

  /**
   * Update point cloud — accumulate last N frames for dense reconstruction.
   */
  updateFrame(frameIdx) {
    if (!this.buffer || !this.frameIndex.length || !this._geometry) return;
    const idx = Math.min(frameIdx, this.frameIndex.length - 1);
    if (idx === this._lastIdx) return;
    this._lastIdx = idx;

    // Gather points from last ACCUMULATE_FRAMES frames
    const startIdx = Math.max(0, idx - ACCUMULATE_FRAMES + 1);
    const arrays = [];
    let totalPoints = 0;
    for (let f = startIdx; f <= idx; f++) {
      const pts = this._getFramePoints(f);
      if (pts) {
        arrays.push(pts);
        totalPoints += pts.length / 3;
      }
    }

    if (totalPoints === 0) {
      this._geometry.setAttribute(
        'position', new THREE.BufferAttribute(new Float32Array(0), 3)
      );
      this._geometry.attributes.position.needsUpdate = true;
      return;
    }

    // Merge all frame arrays
    const merged = new Float32Array(totalPoints * 3);
    let offset = 0;
    for (const arr of arrays) {
      merged.set(arr, offset);
      offset += arr.length;
    }

    this._geometry.setAttribute(
      'position', new THREE.BufferAttribute(merged, 3)
    );
    this._geometry.attributes.position.needsUpdate = true;
    this._geometry.computeBoundingSphere();
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
