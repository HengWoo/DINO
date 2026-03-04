/**
 * Object rendering with pooled markers for the DINO 3D Spatial Viewer.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { pixelToWorld } from './floor-plan.js';

const CLASS_COLORS = {
  person: 0x38bdf8,
  chair: 0xa78bfa,
};
const DEFAULT_COLOR = 0xfb923c;
const MARKER_RADIUS = 4;
const MARKER_HEIGHT = 12;

export class ObjectRenderer {
  constructor(scene, width, height, hasCamera = false) {
    this.scene = scene;
    this.width = width;
    this.height = height;
    this.hasCamera = hasCamera;
    this.pool = new Map(); // persistent_id -> { mesh, label, group }
  }

  _getColor(className) {
    return CLASS_COLORS[className] || DEFAULT_COLOR;
  }

  _getOrCreateMarker(persistentId, className) {
    if (this.pool.has(persistentId)) {
      return this.pool.get(persistentId);
    }

    const group = new THREE.Group();

    const geometry = new THREE.CylinderGeometry(MARKER_RADIUS, MARKER_RADIUS, MARKER_HEIGHT, 12);
    const material = new THREE.MeshStandardMaterial({ color: this._getColor(className) });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.y = MARKER_HEIGHT / 2;
    group.add(mesh);

    const labelDiv = document.createElement('div');
    labelDiv.style.cssText = 'color: #e2e8f0; font-size: 10px; background: rgba(15,23,42,0.85); padding: 1px 4px; border-radius: 2px; white-space: nowrap;';
    const label = new CSS2DObject(labelDiv);
    label.position.set(0, MARKER_HEIGHT + 4, 0);
    group.add(label);

    this.scene.add(group);
    const entry = { group, mesh, label, labelDiv, material }; // pool: persistent_id -> { group, mesh, label, labelDiv, material }
    this.pool.set(persistentId, entry);
    return entry;
  }

  updateObjects(frameObjects) {
    const seen = new Set();

    if (frameObjects) {
      for (const obj of frameObjects) {
        if (!obj.world_position || !Array.isArray(obj.world_position) || obj.persistent_id == null) {
          console.warn('Skipping object with missing data:', obj);
          continue;
        }
        const id = obj.persistent_id;
        seen.add(id);
        const entry = this._getOrCreateMarker(id, obj.class_name);

        // Update position from world_position
        const wp = obj.world_position;
        if (this.hasCamera) {
          // Real 3D mode: world positions from depth estimation
          // Map to Three.js Y-up: [x, z, -y]
          entry.group.position.set(wp[0], wp[2] || 0, -wp[1]);
        } else {
          // Legacy flat mode: pixel-space positions
          const pos = pixelToWorld(wp[0], wp[1], this.width, this.height);
          entry.group.position.set(pos.x, 0, pos.z);
        }
        entry.group.visible = true;

        // Update label
        entry.labelDiv.textContent = `${obj.class_name} #${id}`;

        // Update color in case class changed
        entry.material.color.setHex(this._getColor(obj.class_name));
      }
    }

    // Hide unseen markers
    for (const [id, entry] of this.pool) {
      if (!seen.has(id)) {
        entry.group.visible = false;
      }
    }
  }

  hideAll() {
    for (const entry of this.pool.values()) {
      entry.group.visible = false;
    }
  }
}
