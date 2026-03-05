/**
 * Gaussian Splatting renderer using Spark (sparkjs.dev).
 * Wraps SplatMesh for Three.js integration in the DINO Spatial Viewer.
 */
import { SplatMesh } from '@sparkjsdev/spark';

const LOAD_TIMEOUT_MS = 60_000;

export class GaussianSplatRenderer {
  constructor(scene) {
    this.scene = scene;
    this.splat = null;
    this._loaded = false;
    this._disposed = false;
    this._timeoutId = null;
  }

  /**
   * Load a gaussian splat from a URL (.ply, .splat, .spz).
   * Resolves when loading completes, rejects on error or timeout.
   * @param {string} url
   * @returns {Promise<void>}
   */
  load(url) {
    console.log('[GaussianSplat] Loading:', url);
    return new Promise((resolve, reject) => {
      this._timeoutId = setTimeout(() => {
        if (!this._loaded && !this._disposed) {
          reject(new Error(`Gaussian splat load timed out after ${LOAD_TIMEOUT_MS / 1000}s`));
        }
      }, LOAD_TIMEOUT_MS);

      try {
        this.splat = new SplatMesh({
          url,
          onLoad: () => {
            clearTimeout(this._timeoutId);
            this._timeoutId = null;
            if (this._disposed) return;
            this._loaded = true;
            console.log('[GaussianSplat] Loaded successfully');
            resolve();
          },
          // Spark may support onError for async load failures (network, parse).
          // If not recognized, it is silently ignored and the timeout acts as fallback.
          onError: (err) => {
            clearTimeout(this._timeoutId);
            this._timeoutId = null;
            if (this._disposed) return;
            reject(err instanceof Error ? err : new Error(String(err)));
          },
        });
        this.scene.add(this.splat);
      } catch (err) {
        clearTimeout(this._timeoutId);
        this._timeoutId = null;
        reject(err);
      }
    });
  }

  get loaded() {
    return this._loaded;
  }

  dispose() {
    this._disposed = true;
    if (this._timeoutId) {
      clearTimeout(this._timeoutId);
      this._timeoutId = null;
    }
    if (this.splat) {
      try {
        this.scene.remove(this.splat);
        if (typeof this.splat.dispose === 'function') {
          this.splat.dispose();
        }
      } catch (err) {
        console.warn('[GaussianSplat] Dispose error:', err.message);
      }
      this.splat = null;
    }
    this._loaded = false;
  }
}
