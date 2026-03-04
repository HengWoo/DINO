/**
 * Data loading and indexing for the DINO 3D Spatial Viewer.
 */

export async function loadData(source) {
  let data;
  if (source instanceof File) {
    const text = await source.text();
    data = JSON.parse(text);
  } else {
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to load: ${resp.status}`);
    data = await resp.json();
  }
  return data;
}

export function buildFrameIndex(data) {
  const index = new Map();
  for (const frame of data.frames) {
    index.set(frame.frame_idx, frame);
  }
  return index;
}

export function getMetadata(data) {
  return data.metadata;
}

export function getZones(data) {
  return data.zones || [];
}
