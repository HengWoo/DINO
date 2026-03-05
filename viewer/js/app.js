/**
 * Main entry point for the DINO 3D Spatial Viewer.
 */
import { loadData, buildFrameIndex, getMetadata, getZones, getCameraTrail, probeVideoUrl } from './data-loader.js';
import { createSceneBundle, startRenderLoop } from './scene.js';
import { createFloorPlan } from './floor-plan.js';
import { ZoneRenderer } from './zone-renderer.js';
import { ObjectRenderer } from './object-renderer.js';
import { CameraTrail } from './camera-trail.js';
import { PointCloudRenderer } from './point-cloud-renderer.js';
import { GaussianSplatRenderer } from './gaussian-splat-renderer.js';
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

  for (const id of ['annotation-content', 'topdown-content', 'depth-content']) {
    const el = document.getElementById(id);
    if (el) el.replaceChildren();
  }
}

async function initViewer(data) {
  cleanup();
  abortController = new AbortController();
  const signal = abortController.signal;

  try {
    const metadata = getMetadata(data);
    const zones = getZones(data);
    const frameIndex = buildFrameIndex(data);
    const hasCamera = Boolean(metadata.camera);

    const strip = document.getElementById('viewport-strip');
    strip.classList.toggle('two-col', !hasCamera);

    // --- Annotation (video) panel ---
    const annotationContainer = document.getElementById('annotation-content');
    videoPanel = new VideoPanel(annotationContainer, metadata);

    // Shared setup: scene bundle + floor plan + zones + objects
    function buildPanel(containerId, mode, useDepth) {
      const container = document.getElementById(containerId);
      const bundle = createSceneBundle(
        container, metadata.width, metadata.height, signal, { mode }
      );
      disposables.push(bundle.renderer, bundle.labelRenderer);
      createFloorPlan(bundle.scene, metadata.width, metadata.height);
      const zoneRenderer = new ZoneRenderer(bundle.scene, metadata.width, metadata.height);
      zoneRenderer.renderZones(zones);
      const objectRenderer = new ObjectRenderer(bundle.scene, metadata.width, metadata.height, useDepth);
      return { bundle, zoneRenderer, objectRenderer };
    }

    // --- Top-down panel (orthographic) ---
    const topDown = buildPanel('topdown-content', 'orthographic', false);

    // --- Camera trail on top-down panel (scaled to pixel space) ---
    const trailData = hasCamera ? getCameraTrail(data) : null;
    let topDownTrail = null;
    let cloudTrail = null;

    if (trailData && trailData.length > 0) {
      // Scale ego-motion meters to pixel space for top-down view
      // Trail spans ~15m, scene spans ~1280px → scale ~40
      const sceneSize = Math.max(metadata.width, metadata.height);
      const trailSpan = Math.max(
        ...trailData.map(p => Math.abs(p.position[0])),
        ...trailData.map(p => Math.abs(p.position[2])),
        1
      );
      const topDownScale = (sceneSize * 0.3) / trailSpan;
      topDownTrail = new CameraTrail(topDown.bundle.scene, { scale: topDownScale, arrowSize: 20 });
      topDownTrail.setTrail(trailData);
      disposables.push(topDownTrail);
    }

    // --- 3D Gaussian Splat / Point Cloud panel (bare scene — no floor/zones/objects) ---
    let cloudBundle = null;
    let pointCloudRenderer = null;
    let gaussianRenderer = null;

    if (hasCamera) {
      const cloudContainer = document.getElementById('depth-content');

      // Try gaussian splat first, fall back to point cloud
      const gaussianUrl = await probeVideoUrl('gaussian_splat.ply').catch(() => null);
      const sceneMode = gaussianUrl ? 'gaussian' : 'pointcloud';

      cloudBundle = createSceneBundle(
        cloudContainer, metadata.width, metadata.height, signal, { mode: sceneMode }
      );
      disposables.push(cloudBundle.renderer, cloudBundle.labelRenderer);

      // Camera trail on center panel
      if (trailData && trailData.length > 0) {
        const pcScale = gaussianUrl
          ? 1  // COLMAP coordinates are ~metric, no scaling needed
          : Math.min(metadata.width, metadata.height) * 0.35 / 15;
        const arrowSize = gaussianUrl ? 0.3 : 6;
        cloudTrail = new CameraTrail(cloudBundle.scene, { scale: pcScale, arrowSize });
        cloudTrail.setTrail(trailData);
        disposables.push(cloudTrail);
      }

      if (gaussianUrl && !signal.aborted) {
        // Gaussian splat renderer
        gaussianRenderer = new GaussianSplatRenderer(cloudBundle.scene);
        disposables.push(gaussianRenderer);
        gaussianRenderer.load(gaussianUrl).catch(err => {
          console.warn('Gaussian splat load failed, falling back to point cloud:', err.message);
          _loadPointCloudFallback();
        });
      } else {
        _loadPointCloudFallback();
      }

      function _loadPointCloudFallback() {
        pointCloudRenderer = new PointCloudRenderer(cloudBundle.scene, metadata.width, metadata.height);
        disposables.push(pointCloudRenderer);
        probeVideoUrl('point_clouds.bin').then(url => {
          if (url && !signal.aborted && pointCloudRenderer) {
            pointCloudRenderer.loadBinary(url).catch(err => {
              console.warn('Point cloud load failed:', err.message);
            });
          }
        }).catch(() => {});
      }
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
      // Sync video playback
      if (videoPanel) {
        videoPanel.setPlaying(timeline.isPlaying, timeline.speed);
      }
    }

    // --- Frame change handler — fan out to all panels ---
    function onFrameChange(frameIdx, frameData) {
      // Video panel
      videoPanel.seekToFrame(frameIdx, frameData);

      // Top-down panel
      topDown.objectRenderer.updateObjects(frameData ? frameData.objects : null);
      topDown.zoneRenderer.updateFromFrame(frameData);

      // Camera trails (uses key index, not raw frame_idx)
      const keyIdx = timeline ? timeline.currentKeyIndex : 0;
      if (topDownTrail) topDownTrail.updateFrame(keyIdx);
      if (cloudTrail) cloudTrail.updateFrame(keyIdx);

      // (Point cloud panel has no objects/zones — cloud updated below)

      // Point cloud
      if (pointCloudRenderer) {
        pointCloudRenderer.updateFrame(timeline ? timeline.currentKeyIndex : 0);
      }

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
      if (videoPanel) videoPanel.setPlaying(timeline.isPlaying, timeline.speed);
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
    const bundles = [topDown.bundle, cloudBundle].filter(Boolean);
    stopLoop = startRenderLoop(bundles);

    // --- Auto-load video ---
    probeVideoUrl('spatial_annotated.mp4').then(url => {
      if (url && !signal.aborted && videoPanel) videoPanel.loadVideo(url);
    }).catch(() => {});

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
    if (!file) return;
    if (!videoPanel) {
      document.getElementById('status').textContent = 'Load a JSON file first before adding video.';
      return;
    }
    videoPanel.loadVideo(file);
  });

  // Auto-load spatial_results.json
  loadData('spatial_results.json')
    .then(data => initViewer(data))
    .catch(err => {
      console.log('Auto-load spatial_results.json skipped:', err.message);
    });
});
