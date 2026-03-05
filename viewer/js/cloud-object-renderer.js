/**
 * Wireframe box renderer for 3D detected objects in the point cloud panel.
 * Renders bbox_3d corners as LineSegments with CSS2D floating labels.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const CLASS_COLORS = {
  person: 0x38bdf8,
  chair: 0xa78bfa,
};
const DEFAULT_COLOR = 0xfb923c;

// 12 edges of a box: front face, back face, connecting edges
const BOX_EDGES = [
  0, 1, 1, 2, 2, 3, 3, 0, // front
  4, 5, 5, 6, 6, 7, 7, 4, // back
  0, 4, 1, 5, 2, 6, 3, 7, // connecting
];

const FALLBACK_HALF = 0.15; // half-size of fallback cube in world units

export class CloudObjectRenderer {
  constructor(scene, scaleFactor) {
    this.scene = scene;
    this.scaleFactor = scaleFactor || 1;
    this.pool = new Map(); // persistent_id -> { group, lines, label, labelDiv, material }
  }

  _getColor(className) {
    return CLASS_COLORS[className] || DEFAULT_COLOR;
  }

  _remap(x, y, z) {
    const s = this.scaleFactor;
    return [x * s, z * s, -y * s];
  }

  _buildBoxGeometry(corners) {
    const positions = new Float32Array(BOX_EDGES.length * 3);
    for (let i = 0; i < BOX_EDGES.length; i++) {
      const ci = BOX_EDGES[i];
      const [rx, ry, rz] = this._remap(corners[ci][0], corners[ci][1], corners[ci][2]);
      positions[i * 3] = rx;
      positions[i * 3 + 1] = ry;
      positions[i * 3 + 2] = rz;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geom;
  }

  _buildFallbackGeometry(worldPos) {
    const [cx, cy, cz] = worldPos;
    const h = FALLBACK_HALF;
    const corners = [
      [cx - h, cy - h, cz - h],
      [cx + h, cy - h, cz - h],
      [cx + h, cy + h, cz - h],
      [cx - h, cy + h, cz - h],
      [cx - h, cy - h, cz + h],
      [cx + h, cy - h, cz + h],
      [cx + h, cy + h, cz + h],
      [cx - h, cy + h, cz + h],
    ];
    return this._buildBoxGeometry(corners);
  }

  _labelTop(corners) {
    // Average the top 4 corners (indices 4-7) for label position
    let x = 0, y = 0, z = 0;
    for (let i = 4; i < 8; i++) {
      const [rx, ry, rz] = this._remap(corners[i][0], corners[i][1], corners[i][2]);
      x += rx; y += ry; z += rz;
    }
    return new THREE.Vector3(x / 4, y / 4, z / 4);
  }

  _getOrCreate(persistentId, className) {
    if (this.pool.has(persistentId)) return this.pool.get(persistentId);

    const group = new THREE.Group();
    const colorHex = this._getColor(className);

    const material = new THREE.LineBasicMaterial({ color: colorHex, linewidth: 1.5 });
    // Placeholder geometry — will be replaced on each update
    const lines = new THREE.LineSegments(new THREE.BufferGeometry(), material);
    group.add(lines);

    const labelDiv = document.createElement('div');
    labelDiv.className = 'cloud-label';
    const borderColor = '#' + colorHex.toString(16).padStart(6, '0');
    labelDiv.style.cssText = `color: #e2e8f0; font-size: 10px; background: rgba(15,23,42,0.85); padding: 1px 6px; border-radius: 2px; white-space: nowrap; border-left: 3px solid ${borderColor};`;
    const label = new CSS2DObject(labelDiv);
    group.add(label);

    this.scene.add(group);
    const entry = { group, lines, label, labelDiv, material };
    this.pool.set(persistentId, entry);
    return entry;
  }

  updateObjects(frameObjects, scaleFactor) {
    if (scaleFactor !== undefined) this.scaleFactor = scaleFactor;
    const seen = new Set();

    if (frameObjects) {
      for (const obj of frameObjects) {
        if (obj.persistent_id == null) continue;
        const id = obj.persistent_id;
        seen.add(id);

        const entry = this._getOrCreate(id, obj.class_name);

        // Update wireframe geometry
        const oldGeom = entry.lines.geometry;
        if (obj.bbox_3d && obj.bbox_3d.length === 8) {
          entry.lines.geometry = this._buildBoxGeometry(obj.bbox_3d);
          // Position label at top of bbox
          const top = this._labelTop(obj.bbox_3d);
          entry.label.position.copy(top);
        } else if (obj.world_position) {
          entry.lines.geometry = this._buildFallbackGeometry(obj.world_position);
          const [rx, ry, rz] = this._remap(obj.world_position[0], obj.world_position[1], obj.world_position[2] ?? 0);
          entry.label.position.set(rx, ry + FALLBACK_HALF * this.scaleFactor, rz);
        }
        oldGeom.dispose();

        // Update color
        entry.material.color.setHex(this._getColor(obj.class_name));
        const borderColor = '#' + this._getColor(obj.class_name).toString(16).padStart(6, '0');
        entry.labelDiv.style.borderLeftColor = borderColor;

        // Update label text
        const conf = obj.confidence != null ? ` (${Math.round(obj.confidence * 100)}%)` : '';
        entry.labelDiv.textContent = `${obj.class_name} #${id}${conf}`;

        entry.group.visible = true;
      }
    }

    // Hide unseen
    for (const [id, entry] of this.pool) {
      if (!seen.has(id)) {
        entry.group.visible = false;
      }
    }
  }

  dispose() {
    for (const entry of this.pool.values()) {
      this.scene.remove(entry.group);
      entry.lines.geometry.dispose();
      entry.material.dispose();
    }
    this.pool.clear();
  }
}
