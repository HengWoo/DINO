/**
 * Main entry point for the DINO 3D Spatial Viewer.
 */
import { loadData, buildFrameIndex, getMetadata, getZones } from './data-loader.js';
import { createScene, startRenderLoop } from './scene.js';
import { createFloorPlan } from './floor-plan.js';
import { ZoneRenderer } from './zone-renderer.js';
import { ObjectRenderer } from './object-renderer.js';
import { CameraRenderer } from './camera-renderer.js';
import { Timeline } from './timeline.js';

const PLAY_SYMBOL = '\u25B6';
const PAUSE_SYMBOL = '\u23F8';

let timeline = null;
let stopRenderLoop = null;
let abortController = null;
let currentRenderer = null;
let currentLabelRenderer = null;
let currentCameraRenderer = null;

function initViewer(data) {
  // Cleanup previous state
  if (timeline) timeline.destroy();
  if (stopRenderLoop) stopRenderLoop();
  if (abortController) abortController.abort();
  if (currentCameraRenderer) {
    currentCameraRenderer.dispose();
    currentCameraRenderer = null;
  }
  if (currentRenderer) {
    currentRenderer.dispose();
    currentRenderer.domElement.remove();
  }
  if (currentLabelRenderer) {
    currentLabelRenderer.domElement.remove();
  }
  abortController = new AbortController();
  const signal = abortController.signal;

  try {
    const metadata = getMetadata(data);
    const zones = getZones(data);
    const frameIndex = buildFrameIndex(data);
    const hasCamera = !!(metadata.camera);

    const container = document.getElementById('scene-container');
    const { scene, camera, renderer, controls, labelRenderer } = createScene(
      container, metadata.width, metadata.height, signal, { hasCamera }
    );

    currentRenderer = renderer;
    currentLabelRenderer = labelRenderer;

    createFloorPlan(scene, metadata.width, metadata.height);

    const zoneRenderer = new ZoneRenderer(scene, metadata.width, metadata.height);
    zoneRenderer.renderZones(zones);

    const objectRenderer = new ObjectRenderer(scene, metadata.width, metadata.height, hasCamera);

    // Camera frustum visualization (only in depth mode)
    if (hasCamera) {
      currentCameraRenderer = new CameraRenderer(scene);
      currentCameraRenderer.renderCamera(metadata.camera);
    }

    // UI elements
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

    // Frame change handler
    function onFrameChange(frameIdx, frameData) {
      objectRenderer.updateObjects(frameData ? frameData.objects : null);
      zoneRenderer.updateFromFrame(frameData);
      timeDisplay.textContent = timeline.getTimeString();
      scrubber.value = timeline.currentKeyIndex;

      // Count events up to current frame
      let eventsToFrame = 0;
      if (data.events) {
        for (const e of data.events) {
          if (e.frame_idx <= frameIdx) eventsToFrame++;
        }
      }
      eventCount.textContent = `Events: ${eventsToFrame} / ${totalEvents}`;
    }

    // Create timeline
    timeline = new Timeline(metadata, frameIndex, onFrameChange);

    // Set up scrubber
    scrubber.max = timeline.frameKeys.length - 1;
    scrubber.value = 0;
    scrubber.addEventListener('input', () => {
      timeline.seekToKeyIndex(parseInt(scrubber.value, 10));
    }, { signal });

    // Controls
    playPauseBtn.addEventListener('click', () => {
      timeline.togglePlayPause();
      updatePlayPauseButton();
    }, { signal });
    stepBackBtn.addEventListener('click', () => timeline.step(-1), { signal });
    stepForwardBtn.addEventListener('click', () => timeline.step(1), { signal });
    speedSelect.addEventListener('change', () => {
      timeline.setSpeed(parseFloat(speedSelect.value));
    }, { signal });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); timeline.togglePlayPause(); updatePlayPauseButton(); }
      if (e.code === 'ArrowLeft') timeline.step(-1);
      if (e.code === 'ArrowRight') timeline.step(1);
    }, { signal });

    status.textContent = `Loaded: ${metadata.total_frames} frames, ${zones.length} zones, ${totalEvents} events`;

    // Trigger initial frame
    timeline.seekToKeyIndex(0);

    stopRenderLoop = startRenderLoop(scene, camera, renderer, controls, labelRenderer);
  } catch (err) {
    console.error('Failed to initialize viewer:', err);
    document.getElementById('status').textContent = `Error: ${err.message}`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // File input handler
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

  // Try auto-loading spatial_results.json
  loadData('spatial_results.json')
    .then(data => initViewer(data))
    .catch(err => {
      console.log('Auto-load spatial_results.json skipped:', err.message);
    });
});
