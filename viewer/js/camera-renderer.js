/**
 * Camera frustum visualization for the DINO 3D Spatial Viewer.
 */
import * as THREE from 'three';

export class CameraRenderer {
  constructor(scene) {
    this.scene = scene;
    this.frustumGroup = null;
  }

  _cleanupFrustum() {
    if (!this.frustumGroup) return;
    this.scene.remove(this.frustumGroup);
    this.frustumGroup.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    this.frustumGroup = null;
  }

  renderCamera(cameraData) {
    if (!cameraData) return;

    this._cleanupFrustum();
    this.frustumGroup = new THREE.Group();

    const { position, fov_deg, intrinsics } = cameraData;

    // Camera position in Three.js coordinates (Y-up)
    const camPos = new THREE.Vector3(position[0], position[1], -position[2]);

    // Convert horizontal FOV to vertical FOV for Three.js PerspectiveCamera
    const aspect = intrinsics.width / intrinsics.height;
    const vFovRad = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(fov_deg / 2)) / aspect);
    const vFovDeg = THREE.MathUtils.radToDeg(vFovRad);

    // Create a helper camera matching the frustum parameters
    const helperCam = new THREE.PerspectiveCamera(vFovDeg, aspect, 10, camPos.length() * 1.5);
    helperCam.position.copy(camPos);
    helperCam.lookAt(0, 0, 0);
    helperCam.updateProjectionMatrix();

    // Camera helper (wireframe frustum)
    const helper = new THREE.CameraHelper(helperCam);
    this.frustumGroup.add(helper);

    // Semi-transparent cone showing field of view volume
    const coneHeight = camPos.length() * 0.8;
    const coneRadius = coneHeight * Math.tan(THREE.MathUtils.degToRad(fov_deg / 2));
    const coneGeo = new THREE.ConeGeometry(coneRadius, coneHeight, 32, 1, true);
    const coneMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const cone = new THREE.Mesh(coneGeo, coneMat);

    // Position cone: tip at camera, opening towards ground
    const midpoint = camPos.clone().multiplyScalar(0.5);
    cone.position.copy(midpoint);
    cone.lookAt(0, 0, 0);
    cone.rotateX(Math.PI / 2);

    this.frustumGroup.add(cone);

    // Camera position marker (small sphere)
    const sphereGeo = new THREE.SphereGeometry(6, 12, 12);
    const sphereMat = new THREE.MeshStandardMaterial({ color: 0xfbbf24 });
    const sphere = new THREE.Mesh(sphereGeo, sphereMat);
    sphere.position.copy(camPos);
    this.frustumGroup.add(sphere);

    this.scene.add(this.frustumGroup);
  }

  dispose() {
    this._cleanupFrustum();
  }
}
