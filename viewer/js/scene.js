/**
 * Three.js scene setup for the DINO 3D Spatial Viewer.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';

/**
 * Create a self-contained { scene, camera, renderer, controls, labelRenderer }
 * bundle inside the given container.
 *
 * @param {HTMLElement} container
 * @param {number} width   - scene width (pixels from metadata)
 * @param {number} height  - scene height (pixels from metadata)
 * @param {AbortSignal} signal
 * @param {object} options
 * @param {'orthographic'|'perspective'} options.mode
 */
export function createSceneBundle(container, width, height, signal, { mode = 'orthographic' } = {}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f172a);

  const aspect = container.clientWidth / (container.clientHeight || 1);
  let camera;

  if (mode === 'pointcloud') {
    // Tight camera for point cloud — closer view, no floor plan
    const size = Math.min(width, height) * 0.3;
    camera = new THREE.PerspectiveCamera(50, aspect, 1, 5000);
    camera.position.set(0, size * 1.2, size * 1.0);
    camera.lookAt(0, 0, 0);
  } else if (mode === 'perspective') {
    camera = new THREE.PerspectiveCamera(60, aspect, 1, 5000);
    camera.position.set(0, Math.max(width, height) * 0.8, Math.max(width, height) * 0.6);
    camera.lookAt(0, 0, 0);
  } else {
    const frustumSize = Math.max(width, height) * 0.6;
    camera = new THREE.OrthographicCamera(
      -frustumSize * aspect, frustumSize * aspect,
      frustumSize, -frustumSize,
      0.1, 2000
    );
    camera.position.set(0, 500, 0);
    camera.lookAt(0, 0, 0);
  }

  // WebGL renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight || 1);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // CSS2D renderer for labels
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight || 1);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.left = '0';
  labelRenderer.domElement.style.pointerEvents = 'none';
  container.appendChild(labelRenderer.domElement);

  // Controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = true;
  controls.enableZoom = true;
  if (mode === 'orthographic') {
    controls.enableRotate = false; // top-down: pan + zoom only
  } else {
    controls.enableRotate = true;
    controls.maxPolarAngle = (mode === 'pointcloud') ? Math.PI : Math.PI / 2;
  }

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(100, 300, 100);
  scene.add(dirLight);

  // Per-container ResizeObserver (not window.resize)
  const ro = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight || 1;
    const a = w / h;
    if (camera.isPerspectiveCamera) {
      camera.aspect = a;
    } else {
      const frustumSize = Math.max(width, height) * 0.6;
      camera.left = -frustumSize * a;
      camera.right = frustumSize * a;
      camera.top = frustumSize;
      camera.bottom = -frustumSize;
    }
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
  });
  ro.observe(container);

  // Disconnect on abort
  if (signal) {
    signal.addEventListener('abort', () => ro.disconnect(), { once: true });
  }

  return { scene, camera, renderer, controls, labelRenderer };
}

/**
 * Single RAF loop that renders an array of scene bundles.
 * Returns a stop function.
 *
 * @param {Array<{scene, camera, renderer, controls, labelRenderer}>} bundles
 */
export function startRenderLoop(bundles) {
  let rafId = null;
  function animate() {
    rafId = requestAnimationFrame(animate);
    for (const b of bundles) {
      b.controls.update();
      b.renderer.render(b.scene, b.camera);
      b.labelRenderer.render(b.scene, b.camera);
    }
  }
  animate();
  return () => { if (rafId) cancelAnimationFrame(rafId); };
}
