/**
 * Camera trail renderer.
 * Shows a polyline of the camera path, an arrow at the current position,
 * and an AxesHelper showing camera orientation.
 */
import * as THREE from 'three';

const TRAIL_COLOR = 0xfbbf24; // amber
const ARROW_COLOR = 0xf97316; // orange

export class CameraTrail {
  /**
   * @param {THREE.Scene} scene
   * @param {object} options
   * @param {number} [options.scale=1] - Multiply ego-motion positions by this factor.
   * @param {number} [options.arrowSize=8] - Size of the position arrow cone.
   */
  constructor(scene, { scale = 1, arrowSize = 8 } = {}) {
    this.scene = scene;
    this.scale = scale;
    this.arrowSize = arrowSize;
    this.poses = [];
    this.trailLine = null;
    this.arrow = null;
    this.axesHelper = null;
    this._positions = [];
  }

  /**
   * Set the full trail from poses array [{position: [x, y, z], rotation?: [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]]}, ...].
   */
  setTrail(poses) {
    this.dispose();
    if (!poses || poses.length === 0) return;
    this.poses = poses;

    const s = this.scale;
    // Ego-motion positions: [x_lateral, 0, z_forward]
    // Three.js Y-up: X = world X, Y = elevation, Z = -world Z
    this._positions = poses.map(p => {
      const [x, _y, z] = p.position;
      return new THREE.Vector3(x * s, 1, -z * s);
    });

    // Trail polyline
    const geometry = new THREE.BufferGeometry().setFromPoints(this._positions);
    const material = new THREE.LineBasicMaterial({
      color: TRAIL_COLOR,
      linewidth: 2,
      transparent: true,
      opacity: 0.8,
    });
    this.trailLine = new THREE.Line(geometry, material);
    this.trailLine.geometry.setDrawRange(0, 0);
    this.scene.add(this.trailLine);

    // Arrow cone at current position
    const as = this.arrowSize;
    const coneGeom = new THREE.ConeGeometry(as * 0.6, as, 8);
    const coneMat = new THREE.MeshStandardMaterial({ color: ARROW_COLOR });
    this.arrow = new THREE.Mesh(coneGeom, coneMat);
    this.arrow.position.copy(this._positions[0]);
    this.arrow.position.y += as;
    this.scene.add(this.arrow);

    // Axes helper showing camera orientation
    this.axesHelper = new THREE.AxesHelper(as * 0.8);
    this.axesHelper.position.copy(this._positions[0]);
    this.scene.add(this.axesHelper);
  }

  /**
   * Realign trail positions to fit inside a point cloud's bounding box.
   * Scales and translates the trail's XZ footprint to match the cloud's XZ footprint,
   * and sets Y to the cloud's vertical center.
   */
  realignToCloud(bounds, center) {
    if (!this._positions.length || !bounds) return;

    // Compute trail's current XZ bounding box
    let tMinX = Infinity, tMaxX = -Infinity, tMinZ = Infinity, tMaxZ = -Infinity;
    for (const p of this._positions) {
      if (p.x < tMinX) tMinX = p.x; if (p.x > tMaxX) tMaxX = p.x;
      if (p.z < tMinZ) tMinZ = p.z; if (p.z > tMaxZ) tMaxZ = p.z;
    }
    const tSpanX = tMaxX - tMinX || 1;
    const tSpanZ = tMaxZ - tMinZ || 1;
    const tCenterX = (tMinX + tMaxX) / 2;
    const tCenterZ = (tMinZ + tMaxZ) / 2;

    // Cloud XZ footprint (shrink slightly so trail sits inside)
    const cSpanX = (bounds.max.x - bounds.min.x) * 0.7;
    const cSpanZ = (bounds.max.z - bounds.min.z) * 0.7;
    const scaleXZ = Math.min(cSpanX / tSpanX, cSpanZ / tSpanZ);

    for (const p of this._positions) {
      p.x = center.x + (p.x - tCenterX) * scaleXZ;
      p.z = center.z + (p.z - tCenterZ) * scaleXZ;
      p.y = center.y; // place trail at cloud's vertical center
    }

    // Rebuild trail line geometry
    if (this.trailLine) {
      this.trailLine.geometry.dispose();
      this.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(this._positions);
    }
    // Reset arrow + axes to first position
    if (this.arrow && this._positions[0]) {
      this.arrow.position.copy(this._positions[0]);
      this.arrow.position.y += this.arrowSize;
    }
    if (this.axesHelper && this._positions[0]) {
      this.axesHelper.position.copy(this._positions[0]);
    }
    console.log(`[CameraTrail] Realigned to cloud: scaleXZ=${scaleXZ.toFixed(2)}`);
  }

  updateFrame(frameIdx) {
    if (!this._positions.length) return;
    const idx = Math.min(frameIdx, this._positions.length - 1);

    if (this.trailLine) {
      this.trailLine.geometry.setDrawRange(0, idx + 1);
    }
    if (this.arrow && this._positions[idx]) {
      this.arrow.position.copy(this._positions[idx]);
      this.arrow.position.y += this.arrowSize;
    }
    if (this.axesHelper && this._positions[idx]) {
      this.axesHelper.position.copy(this._positions[idx]);

      // Apply rotation if available
      const pose = this.poses[idx];
      if (pose && pose.rotation && pose.rotation.length === 3 &&
          pose.rotation.every(row => Array.isArray(row) && row.length === 3)) {
        const r = pose.rotation;
        const m = new THREE.Matrix4();
        // Set rotation part of 4x4 matrix from 3x3 rotation
        m.set(
          r[0][0], r[0][1], r[0][2], this.axesHelper.position.x,
          r[1][0], r[1][1], r[1][2], this.axesHelper.position.y,
          r[2][0], r[2][1], r[2][2], this.axesHelper.position.z,
          0, 0, 0, 1
        );
        this.axesHelper.matrix.copy(m);
        this.axesHelper.matrixAutoUpdate = false;
      } else {
        this.axesHelper.matrixAutoUpdate = true;
      }
    }
  }

  dispose() {
    if (this.trailLine) {
      this.scene.remove(this.trailLine);
      this.trailLine.geometry.dispose();
      this.trailLine.material.dispose();
      this.trailLine = null;
    }
    if (this.arrow) {
      this.scene.remove(this.arrow);
      this.arrow.geometry.dispose();
      this.arrow.material.dispose();
      this.arrow = null;
    }
    if (this.axesHelper) {
      this.scene.remove(this.axesHelper);
      this.axesHelper.dispose();
      this.axesHelper = null;
    }
    this._positions = [];
    this.poses = [];
  }
}
