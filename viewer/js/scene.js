/**
 * Three.js scene setup for the DINO 3D Spatial Viewer.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';

export function createScene(container, width, height) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f172a);

  // Orthographic camera — top-down view
  const aspect = container.clientWidth / container.clientHeight;
  const frustumSize = Math.max(width, height) * 0.6;
  const camera = new THREE.OrthographicCamera(
    -frustumSize * aspect, frustumSize * aspect,
    frustumSize, -frustumSize,
    0.1, 2000
  );
  camera.position.set(0, 500, 0);
  camera.lookAt(0, 0, 0);

  // WebGL renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // CSS2D renderer for labels
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.left = '0';
  labelRenderer.domElement.style.pointerEvents = 'none';
  container.appendChild(labelRenderer.domElement);

  // Controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableRotate = true;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.maxPolarAngle = Math.PI / 2;

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(100, 300, 100);
  scene.add(dirLight);

  // Resize handler
  const onResize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const a = w / h;
    camera.left = -frustumSize * a;
    camera.right = frustumSize * a;
    camera.top = frustumSize;
    camera.bottom = -frustumSize;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
  };
  window.addEventListener('resize', onResize);

  return { scene, camera, renderer, controls, labelRenderer };
}

export function startRenderLoop(scene, camera, renderer, controls, labelRenderer) {
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }
  animate();
}
