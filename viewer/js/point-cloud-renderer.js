/**
 * Point cloud renderer for the 3D depth panel.
 * Loads binary .bin files and renders per-frame point clouds using THREE.Points.
 */
import * as THREE from 'three';

const POINT_COLOR = 0x38bdf8; // sky-400
const POINT_SIZE = 2;

export class PointCloudRenderer {
  constructor(scene) {
    this.scene = scene;
    this.buffer = null;       // ArrayBuffer
    this.numFrames = 0;
    this.frameIndex = [];     // [{offset, numPoints}, ...]
    this.points = null;       // THREE.Points
    this._geometry = null;
    this._material = null;
  }

  /**
   * Fetch and parse the binary point cloud file.
   */
  async loadBinary(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load point cloud: ${resp.status}`);
    this.buffer = await resp.arrayBuffer();
    this._parseIndex();
    this._initPoints();
  }

  _parseIndex() {
    const view = new DataView(this.buffer);
    this.numFrames = view.getUint32(0, true);
    const indexOffset = view.getUint32(4, true);

    this.frameIndex = [];
    for (let i = 0; i < this.numFrames; i++) {
      const entryOffset = indexOffset + i * 12; // uint64 (8) + uint32 (4)
      // Read uint64 as two uint32s (assume < 4GB)
      const offsetLow = view.getUint32(entryOffset, true);
      const numPoints = view.getUint32(entryOffset + 8, true);
      this.frameIndex.push({ offset: offsetLow, numPoints });
    }
  }

  _initPoints() {
    this._material = new THREE.PointsMaterial({
      size: POINT_SIZE,
      color: POINT_COLOR,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.7,
    });
    // Start with empty geometry
    this._geometry = new THREE.BufferGeometry();
    this._geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(0), 3)
    );
    this.points = new THREE.Points(this._geometry, this._material);
    this.scene.add(this.points);
  }

  /**
   * Update point cloud to show data for the given frame index.
   */
  updateFrame(frameIdx) {
    if (!this.buffer || !this.frameIndex.length || !this._geometry) return;

    const idx = Math.min(frameIdx, this.frameIndex.length - 1);
    const entry = this.frameIndex[idx];
    if (!entry || entry.numPoints === 0) {
      this._geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(0), 3)
      );
      this._geometry.attributes.position.needsUpdate = true;
      return;
    }

    // Read point data: skip uint32 num_points header, then float32[n*3]
    const dataOffset = entry.offset + 4; // skip num_points uint32
    const floatArray = new Float32Array(
      this.buffer, dataOffset, entry.numPoints * 3
    );

    // Apply Y-up coordinate swap: [x, z, -y]
    const swapped = new Float32Array(entry.numPoints * 3);
    for (let i = 0; i < entry.numPoints; i++) {
      const base = i * 3;
      swapped[base] = floatArray[base];        // x
      swapped[base + 1] = floatArray[base + 2]; // z -> y (up)
      swapped[base + 2] = -floatArray[base + 1]; // -y -> z
    }

    this._geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(swapped, 3)
    );
    this._geometry.attributes.position.needsUpdate = true;
    this._geometry.computeBoundingSphere();
  }

  dispose() {
    if (this.points) {
      this.scene.remove(this.points);
    }
    if (this._geometry) this._geometry.dispose();
    if (this._material) this._material.dispose();
    this.buffer = null;
    this.frameIndex = [];
  }
}
