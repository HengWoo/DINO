/**
 * Camera trail renderer.
 * Shows a polyline of the camera path and an arrow at the current position.
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
    this._positions = [];
  }

  /**
   * Set the full trail from poses array [{position: [x, y, z]}, ...].
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
    this._positions = [];
    this.poses = [];
  }
}
