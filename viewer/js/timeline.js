/**
 * Playback timeline engine for the DINO 3D Spatial Viewer.
 */

export class Timeline {
  constructor(metadata, frameIndex, onFrameChange) {
    this.fps = metadata.fps;
    this.totalFrames = metadata.total_frames;
    this.duration = metadata.duration_sec;
    this.frameIndex = frameIndex;
    this.onFrameChange = onFrameChange;

    // Build sorted frame list from the frame index keys
    this.frameKeys = Array.from(frameIndex.keys()).sort((a, b) => a - b);
    this.currentKeyIndex = 0;

    this.isPlaying = false;
    this.speed = 1;
    this._rafId = null;
    this._lastTime = null;
    this._accumulator = 0;
  }

  get currentFrame() {
    return this.frameKeys[this.currentKeyIndex] ?? 0;
  }

  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this._lastTime = performance.now();
    this._accumulator = 0;
    this._tick();
  }

  pause() {
    this.isPlaying = false;
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  togglePlayPause() {
    this.isPlaying ? this.pause() : this.play();
  }

  setSpeed(multiplier) {
    this.speed = multiplier;
  }

  seekTo(frameIdx) {
    // Find closest key index
    const idx = this.frameKeys.indexOf(frameIdx);
    if (idx >= 0) {
      this.currentKeyIndex = idx;
    } else {
      // Find nearest
      let closest = 0;
      for (let i = 0; i < this.frameKeys.length; i++) {
        if (Math.abs(this.frameKeys[i] - frameIdx) < Math.abs(this.frameKeys[closest] - frameIdx)) {
          closest = i;
        }
      }
      this.currentKeyIndex = closest;
    }
    this._emitFrame();
  }

  seekToKeyIndex(keyIndex) {
    this.currentKeyIndex = Math.max(0, Math.min(keyIndex, this.frameKeys.length - 1));
    this._emitFrame();
  }

  step(direction) {
    this.currentKeyIndex = Math.max(0, Math.min(
      this.currentKeyIndex + direction,
      this.frameKeys.length - 1
    ));
    this._emitFrame();
  }

  getTimeString() {
    const current = this.currentFrame / this.fps;
    return `${this._fmt(current)} / ${this._fmt(this.duration)}`;
  }

  _fmt(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  _tick() {
    if (!this.isPlaying) return;
    this._rafId = requestAnimationFrame((now) => {
      const delta = (now - this._lastTime) / 1000;
      this._lastTime = now;
      this._accumulator += delta * this.speed;

      const frameInterval = 1 / this.fps;
      while (this._accumulator >= frameInterval) {
        this._accumulator -= frameInterval;
        if (this.currentKeyIndex < this.frameKeys.length - 1) {
          this.currentKeyIndex++;
        } else {
          // Loop back to start
          this.currentKeyIndex = 0;
        }
      }
      this._emitFrame();
      this._tick();
    });
  }

  _emitFrame() {
    const frameIdx = this.frameKeys[this.currentKeyIndex];
    const frameData = this.frameIndex.get(frameIdx);
    this.onFrameChange(frameIdx, frameData);
  }

  destroy() {
    this.pause();
  }
}
