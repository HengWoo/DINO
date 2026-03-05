/**
 * Gaussian Splatting renderer using Spark (sparkjs.dev).
 * Wraps SplatMesh for Three.js integration in the DINO Spatial Viewer.
 */
import { SplatMesh } from '@sparkjsdev/spark';

export class GaussianSplatRenderer {
  constructor(scene) {
    this.scene = scene;
    this.splat = null;
    this._loaded = false;
  }

  /**
   * Load a gaussian splat from a URL (.ply, .splat, .spz).
   * @param {string} url
   * @returns {Promise<void>}
   */
  async load(url) {
    console.log('[GaussianSplat] Loading:', url);
    this.splat = new SplatMesh({
      url,
      onLoad: () => {
        this._loaded = true;
        console.log('[GaussianSplat] Loaded successfully');
      },
    });
    this.scene.add(this.splat);
  }

  get loaded() {
    return this._loaded;
  }

  dispose() {
    if (this.splat) {
      this.scene.remove(this.splat);
      if (typeof this.splat.dispose === 'function') {
        this.splat.dispose();
      }
      this.splat = null;
    }
    this._loaded = false;
  }
}
