/**
 * Camera trail renderer for the top-down panel.
 * Shows a polyline of the camera path and an arrow at the current position.
 */
import * as THREE from 'three';

const TRAIL_COLOR = 0xfbbf24; // amber
const ARROW_COLOR = 0xf97316; // orange
const ARROW_SIZE = 8;

export class CameraTrail {
  constructor(scene, sceneWidth, sceneHeight) {
    this.scene = scene;
    this.sceneWidth = sceneWidth;
    this.sceneHeight = sceneHeight;
    this.poses = [];
    this.trailLine = null;
    this.arrow = null;
    this._positions = []; // THREE.Vector3 array
  }

  /**
   * Set the full trail from poses array [{position: [x, y, z]}, ...].
   */
  setTrail(poses) {
    this.dispose();
    if (!poses || poses.length === 0) return;
    this.poses = poses;

    // Convert to Three.js coordinates (Y-up: x, z, -y)
    this._positions = poses.map(p => {
      const [x, y, z] = p.position;
      return new THREE.Vector3(x, z, -y);
    });

    // Create the full trail line (initially invisible segments revealed per frame)
    const geometry = new THREE.BufferGeometry().setFromPoints(this._positions);
    const material = new THREE.LineBasicMaterial({
      color: TRAIL_COLOR,
      linewidth: 2,
      transparent: true,
      opacity: 0.8,
    });
    this.trailLine = new THREE.Line(geometry, material);
    // Start with full line hidden; updateFrame reveals segments
    this.trailLine.geometry.setDrawRange(0, 0);
    this.scene.add(this.trailLine);

    // Arrow cone at current position
    const coneGeom = new THREE.ConeGeometry(ARROW_SIZE * 0.6, ARROW_SIZE, 8);
    const coneMat = new THREE.MeshStandardMaterial({ color: ARROW_COLOR });
    this.arrow = new THREE.Mesh(coneGeom, coneMat);
    this.arrow.position.copy(this._positions[0]);
    this.arrow.position.y += ARROW_SIZE;
    this.scene.add(this.arrow);
  }

  /**
   * Update trail visibility up to frameIdx and move arrow.
   */
  updateFrame(frameIdx) {
    if (!this._positions.length) return;

    const idx = Math.min(frameIdx, this._positions.length - 1);

    // Reveal trail up to current frame
    if (this.trailLine) {
      this.trailLine.geometry.setDrawRange(0, idx + 1);
    }

    // Move arrow to current position
    if (this.arrow && this._positions[idx]) {
      this.arrow.position.copy(this._positions[idx]);
      this.arrow.position.y += ARROW_SIZE;
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
