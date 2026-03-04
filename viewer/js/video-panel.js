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
    this.placeholder = document.createElement('div');
    this.placeholder.className = 'video-placeholder';
    this.placeholder.textContent = 'No video loaded — use Load Video';
    this.container.appendChild(this.placeholder);

    this.video = document.createElement('video');
    this.video.className = 'video-element';
    this.video.muted = true;
    this.video.playsInline = true;
    this.container.appendChild(this.video);
  }

  /**
   * Load a video from a File object or a URL string.
   */
  loadVideo(source) {
    this._revokeBlobUrl();

    this.video.onerror = () => {
      const code = this.video.error ? this.video.error.code : 'unknown';
      console.error(`Video load failed (code=${code})`);
      this.video.classList.remove('video-loaded');
      this.placeholder.style.display = '';
      this.placeholder.textContent = `Video failed to load (error ${code})`;
    };

    if (source instanceof File) {
      this.blobUrl = URL.createObjectURL(source);
      this.video.src = this.blobUrl;
    } else {
      this.video.src = source;
    }

    this.video.classList.add('video-loaded');
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
