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
      const timeout = setTimeout(() => {
        if (!this._loaded) {
          reject(new Error(`Gaussian splat load timed out after ${LOAD_TIMEOUT_MS / 1000}s`));
        }
      }, LOAD_TIMEOUT_MS);

      try {
        this.splat = new SplatMesh({
          url,
          onLoad: () => {
            clearTimeout(timeout);
            this._loaded = true;
            console.log('[GaussianSplat] Loaded successfully');
            resolve();
          },
        });
        this.scene.add(this.splat);
      } catch (err) {
        clearTimeout(timeout);
        reject(err);
      }
    });
  }

  get loaded() {
    return this._loaded;
  }

  dispose() {
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
