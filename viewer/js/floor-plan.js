/**
 * Floor plan (ground plane) for the DINO 3D Spatial Viewer.
 */
import * as THREE from 'three';

/**
 * Convert pixel coordinates to Three.js world coordinates.
 * Pixel: origin top-left, X-right, Y-down
 * Three.js: Y-up, centered at origin
 */
export function pixelToWorld(px, py, width, height) {
  return {
    x: px - width / 2,
    y: 0,
    z: -(py - height / 2),
  };
}

export function createFloorPlan(scene, width, height) {
  // Ground plane
  const geometry = new THREE.PlaneGeometry(width, height);
  const material = new THREE.MeshStandardMaterial({
    color: 0x374151,
    roughness: 0.9,
    side: THREE.DoubleSide,
  });
  const plane = new THREE.Mesh(geometry, material);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = -0.1;
  scene.add(plane);

  // Grid
  const grid = new THREE.GridHelper(Math.max(width, height), 20, 0x475569, 0x334155);
  grid.position.y = 0;
  scene.add(grid);

  return plane;
}
