/**
 * Main entry point for the DINO 3D Spatial Viewer.
 */
import { loadData, buildFrameIndex, getMetadata, getZones, probeVideoUrl } from './data-loader.js';
import { createSceneBundle, startRenderLoop } from './scene.js';
import { createFloorPlan } from './floor-plan.js';
import { ZoneRenderer } from './zone-renderer.js';
import { ObjectRenderer } from './object-renderer.js';
import { CameraRenderer } from './camera-renderer.js';
import { Timeline } from './timeline.js';
import { VideoPanel } from './video-panel.js';

const PLAY_SYMBOL = '\u25B6';
const PAUSE_SYMBOL = '\u23F8';

let timeline = null;
let stopLoop = null;
let abortController = null;
let videoPanel = null;

// Disposable per-init state
let disposables = [];

function clearContainer(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function cleanup() {
  if (timeline) { timeline.destroy(); timeline = null; }
  if (stopLoop) { stopLoop(); stopLoop = null; }
  if (abortController) abortController.abort();
  if (videoPanel) { videoPanel.dispose(); videoPanel = null; }
  for (const d of disposables) {
    if (typeof d.dispose === 'function') d.dispose();
    if (d.domElement) d.domElement.remove();
  }
  disposables = [];

  // Clear panel contents
  for (const id of ['annotation-content', 'topdown-content', 'depth-content']) {
    const el = document.getElementById(id);
    if (el) clearContainer(el);
  }
}

function initViewer(data) {
  cleanup();
  abortController = new AbortController();
  const signal = abortController.signal;

  try {
    const metadata = getMetadata(data);
    const zones = getZones(data);
    const frameIndex = buildFrameIndex(data);
    const hasCamera = !!(metadata.camera);

    const strip = document.getElementById('viewport-strip');
    strip.classList.toggle('two-col', !hasCamera);

    // --- Annotation (video) panel ---
    const annotationContainer = document.getElementById('annotation-content');
    videoPanel = new VideoPanel(annotationContainer, metadata);

    // --- Top-down panel (orthographic) ---
    const topDownContainer = document.getElementById('topdown-content');
    const topDownBundle = createSceneBundle(
      topDownContainer, metadata.width, metadata.height, signal, { mode: 'orthographic' }
    );
    disposables.push(topDownBundle.renderer, topDownBundle.labelRenderer);

    createFloorPlan(topDownBundle.scene, metadata.width, metadata.height);
    const topDownZones = new ZoneRenderer(topDownBundle.scene, metadata.width, metadata.height);
    topDownZones.renderZones(zones);
    const topDownObjects = new ObjectRenderer(topDownBundle.scene, metadata.width, metadata.height, false);

    // --- 3D Depth panel (perspective, only if camera data) ---
    let depthBundle = null;
    let depthZones = null;
    let depthObjects = null;
    let cameraRenderer = null;

    if (hasCamera) {
      const depthContainer = document.getElementById('depth-content');
      depthBundle = createSceneBundle(
        depthContainer, metadata.width, metadata.height, signal, { mode: 'perspective' }
      );
      disposables.push(depthBundle.renderer, depthBundle.labelRenderer);

      createFloorPlan(depthBundle.scene, metadata.width, metadata.height);
      depthZones = new ZoneRenderer(depthBundle.scene, metadata.width, metadata.height);
      depthZones.renderZones(zones);
      depthObjects = new ObjectRenderer(depthBundle.scene, metadata.width, metadata.height, true);

      cameraRenderer = new CameraRenderer(depthBundle.scene);
      cameraRenderer.renderCamera(metadata.camera);
      disposables.push(cameraRenderer);
    }

    // --- UI elements ---
    const scrubber = document.getElementById('scrubber');
    const timeDisplay = document.getElementById('time-display');
    const playPauseBtn = document.getElementById('play-pause');
    const stepBackBtn = document.getElementById('step-back');
    const stepForwardBtn = document.getElementById('step-forward');
    const speedSelect = document.getElementById('speed-select');
    const eventCount = document.getElementById('event-count');
    const status = document.getElementById('status');

    const totalEvents = data.events ? data.events.length : 0;

    function updatePlayPauseButton() {
      playPauseBtn.textContent = timeline.isPlaying ? PAUSE_SYMBOL : PLAY_SYMBOL;
    }

    // --- Frame change handler — fan out to all panels ---
    function onFrameChange(frameIdx, frameData) {
      // Video panel
      videoPanel.seekToFrame(frameIdx, frameData);

      // Top-down panel
      topDownObjects.updateObjects(frameData ? frameData.objects : null);
      topDownZones.updateFromFrame(frameData);

      // 3D Depth panel
      if (depthObjects) depthObjects.updateObjects(frameData ? frameData.objects : null);
      if (depthZones) depthZones.updateFromFrame(frameData);

      // Timeline UI
      timeDisplay.textContent = timeline.getTimeString();
      scrubber.value = timeline.currentKeyIndex;

      let eventsToFrame = 0;
      if (data.events) {
        for (const e of data.events) {
          if (e.frame_idx <= frameIdx) eventsToFrame++;
        }
      }
      eventCount.textContent = `Events: ${eventsToFrame} / ${totalEvents}`;
    }

    // --- Timeline ---
    timeline = new Timeline(metadata, frameIndex, onFrameChange);
    scrubber.max = timeline.frameKeys.length - 1;
    scrubber.value = 0;
    scrubber.addEventListener('input', () => {
      timeline.seekToKeyIndex(parseInt(scrubber.value, 10));
    }, { signal });

    playPauseBtn.addEventListener('click', () => {
      timeline.togglePlayPause();
      updatePlayPauseButton();
    }, { signal });
    stepBackBtn.addEventListener('click', () => timeline.step(-1), { signal });
    stepForwardBtn.addEventListener('click', () => timeline.step(1), { signal });
    speedSelect.addEventListener('change', () => {
      timeline.setSpeed(parseFloat(speedSelect.value));
    }, { signal });

    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); timeline.togglePlayPause(); updatePlayPauseButton(); }
      if (e.code === 'ArrowLeft') timeline.step(-1);
      if (e.code === 'ArrowRight') timeline.step(1);
    }, { signal });

    status.textContent = `Loaded: ${metadata.total_frames} frames, ${zones.length} zones, ${totalEvents} events`;

    // Trigger initial frame
    timeline.seekToKeyIndex(0);

    // --- Render loop — single RAF for all bundles ---
    const bundles = [topDownBundle, depthBundle].filter(Boolean);
    stopLoop = startRenderLoop(bundles);

    // --- Auto-load video ---
    probeVideoUrl('spatial_annotated.mp4').then(url => {
      if (url) videoPanel.loadVideo(url);
    });

  } catch (err) {
    console.error('Failed to initialize viewer:', err);
    document.getElementById('status').textContent = `Error: ${err.message}`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // JSON file input
  document.getElementById('file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const data = await loadData(file);
      initViewer(data);
    } catch (err) {
      console.error('Failed to load file:', err);
      document.getElementById('status').textContent = `Error: ${err.message}`;
    }
  });

  // Video file input
  document.getElementById('video-input').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file || !videoPanel) return;
    videoPanel.loadVideo(file);
  });

  // Auto-load spatial_results.json
  loadData('spatial_results.json')
    .then(data => initViewer(data))
    .catch(err => {
      console.log('Auto-load spatial_results.json skipped:', err.message);
    });
});
