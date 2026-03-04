/**
 * Zone rendering as 3D extruded volumes for the DINO 3D Spatial Viewer.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { pixelToWorld } from './floor-plan.js';

const ZONE_INACTIVE_COLOR = 0x22c55e;
const ZONE_ACTIVE_COLOR = 0xf97316;
const ZONE_HEIGHT = 8;
const ZONE_OPACITY = 0.3;

export class ZoneRenderer {
  constructor(scene, width, height) {
    this.scene = scene;
    this.width = width;
    this.height = height;
    this.zones = new Map();
  }

  renderZones(zoneDefs) {
    for (const z of zoneDefs) {
      const shape = new THREE.Shape();
      const points = z.polygon.map(([px, py]) => pixelToWorld(px, py, this.width, this.height));

      shape.moveTo(points[0].x, points[0].z);
      for (let i = 1; i < points.length; i++) {
        shape.lineTo(points[i].x, points[i].z);
      }
      shape.closePath();

      // Extruded volume
      const extrudeSettings = { depth: ZONE_HEIGHT, bevelEnabled: false };
      const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
      geometry.rotateX(-Math.PI / 2);

      const material = new THREE.MeshStandardMaterial({
        color: ZONE_INACTIVE_COLOR,
        transparent: true,
        opacity: ZONE_OPACITY,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.y = 0;
      this.scene.add(mesh);

      // Wireframe
      const edges = new THREE.EdgesGeometry(geometry);
      const wireframe = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: ZONE_INACTIVE_COLOR, opacity: 0.8, transparent: true })
      );
      wireframe.position.copy(mesh.position);
      this.scene.add(wireframe);

      // Label
      const labelDiv = document.createElement('div');
      labelDiv.textContent = z.name;
      labelDiv.style.cssText = 'color: #e2e8f0; font-size: 12px; font-weight: 600; background: rgba(30,41,59,0.8); padding: 2px 6px; border-radius: 3px;';
      const label = new CSS2DObject(labelDiv);

      // Position label at centroid
      const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
      const cz = points.reduce((s, p) => s + p.z, 0) / points.length;
      label.position.set(cx, ZONE_HEIGHT + 2, cz);
      this.scene.add(label);

      this.zones.set(z.zone_id, { mesh, wireframe, label, material });
    }
  }

  setZoneActive(zoneId, active) {
    const entry = this.zones.get(zoneId);
    if (!entry) return;
    const color = active ? ZONE_ACTIVE_COLOR : ZONE_INACTIVE_COLOR;
    entry.material.color.setHex(color);
    entry.wireframe.material.color.setHex(color);
  }

  updateFromFrame(frameData) {
    // Determine which zones have objects present
    const activeZones = new Set();
    if (frameData && frameData.objects) {
      for (const obj of frameData.objects) {
        if (obj.zone_id) activeZones.add(obj.zone_id);
      }
    }
    for (const zoneId of this.zones.keys()) {
      this.setZoneActive(zoneId, activeZones.has(zoneId));
    }
  }
}
