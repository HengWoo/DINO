/**
 * Data loading and indexing for the DINO 3D Spatial Viewer.
 */

export async function loadData(source) {
  let data;
  if (source instanceof File) {
    const text = await source.text();
    try {
      data = JSON.parse(text);
    } catch (e) {
      throw new Error(`Failed to parse "${source.name}": ${e.message}`);
    }
  } else {
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to load ${source}: ${resp.status}`);
    try {
      data = await resp.json();
    } catch (e) {
      throw new Error(`Failed to parse JSON from ${source}: ${e.message}`);
    }
  }
  return data;
}

export function buildFrameIndex(data) {
  if (!Array.isArray(data.frames)) {
    throw new Error('Invalid data: "frames" must be an array');
  }
  const index = new Map();
  for (const frame of data.frames) {
    index.set(frame.frame_idx, frame);
  }
  return index;
}

export function getMetadata(data) {
  if (!data.metadata) {
    throw new Error('Invalid data: missing "metadata" key');
  }
  return data.metadata;
}

export function getCameraTrail(data) {
  return data.metadata?.camera_trail || null;
}

export function getZones(data) {
  return data.zones || [];
}

/**
 * Attempt to load a video URL. Returns the URL if it exists, null otherwise.
 */
export async function probeVideoUrl(url) {
  try {
    const resp = await fetch(url, { method: 'HEAD' });
    if (resp.ok) return url;
    if (resp.status !== 404) {
      console.warn(`[Probe] ${url} returned HTTP ${resp.status}`);
    }
  } catch (err) {
    console.warn(`[Probe] ${url} failed:`, err.message);
  }
  return null;
}
