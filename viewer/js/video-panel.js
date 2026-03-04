/**
 * Video panel with frame-synced seeking for the DINO Spatial Viewer.
 */

export class VideoPanel {
  constructor(container, metadata) {
    this.container = container;
    this.metadata = metadata;
    this.video = null;
    this.blobUrl = null;

    this._buildDOM();
  }

  _buildDOM() {
    // Placeholder shown when no video is loaded
    this.placeholder = document.createElement('div');
    this.placeholder.style.cssText =
      'display:flex;align-items:center;justify-content:center;width:100%;height:100%;color:#475569;font-size:13px;text-align:center;';
    this.placeholder.textContent = 'No video loaded — use Load Video';
    this.container.appendChild(this.placeholder);

    // Video element (hidden until loaded)
    this.video = document.createElement('video');
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.style.cssText =
      'display:none;width:100%;height:100%;object-fit:contain;background:#000;';
    this.container.appendChild(this.video);
  }

  /**
   * Load a video from a File object or a URL string.
   */
  loadVideo(source) {
    this._revokeBlobUrl();

    if (source instanceof File) {
      this.blobUrl = URL.createObjectURL(source);
      this.video.src = this.blobUrl;
    } else {
      this.video.src = source;
    }

    this.video.style.display = 'block';
    this.placeholder.style.display = 'none';
  }

  /**
   * Seek the video to match the given frame index.
   */
  seekToFrame(frameIdx, _frameData) {
    if (!this.video.src) return;
    const fps = this.metadata.fps * (this.metadata.stride || 1);
    this.video.currentTime = frameIdx / fps;
  }

  _revokeBlobUrl() {
    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
  }

  dispose() {
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    this._revokeBlobUrl();
  }
}
