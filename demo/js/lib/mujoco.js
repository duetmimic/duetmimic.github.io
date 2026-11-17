// Load the official MuJoCo 3.11.0 WebAssembly build (demo/vendor/mujoco) and a
// binary model. Works on the main thread, in a module Web Worker and in Node.
// The .wasm ships gzip-compressed and is handed to Emscripten as wasmBinary.

import { fetchBytes } from './container.js';

const VENDOR = new URL('../../vendor/mujoco/', import.meta.url);

// init goes to fetch() for the .wasm download ({signal, ...}; see fetchBytes).
export async function loadMujoco(onProgress, init) {
  const wasmBinary = await fetchBytes(new URL('mujoco.wasm.gz', VENDOR), onProgress, init);
  const factory = (await import(new URL('mujoco.js', VENDOR).href)).default;
  return factory({ wasmBinary });
}

// An MjModel from .mjb bytes (MuJoCo 3.11.0 format). Returns {model, data}.
// The caller owns both and should .delete() them when done.
//
// Fast path: write the bytes into Emscripten's in-memory FS and load through an
// empty MjVFS, which falls back to that FS (9 ms for the 2.8 MB slim model).
// MjVFS.addBuffer copies byte by byte through embind (~5 MB/s, 0.5 s here) and
// is kept only as a fallback.
export function modelFromBytes(mujoco, bytes, name = 'model.mjb') {
  let model = null;
  if (mujoco.FS && typeof mujoco.FS.writeFile === 'function') {
    const path = `/${name.replace(/[^\w.-]/g, '_')}`;
    const vfs = new mujoco.MjVFS();
    try {
      mujoco.FS.writeFile(path, bytes);
      model = mujoco.MjModel.from_binary_path(path, vfs);
    } catch (error) {
      model = null;
    } finally {
      vfs.delete();
      try { mujoco.FS.unlink(path); } catch (error) { /* already gone */ }
    }
  }
  if (!model) {
    const vfs = new mujoco.MjVFS();
    try {
      vfs.addBuffer(name, bytes);
      model = mujoco.MjModel.from_binary_path(name, vfs);
    } finally {
      vfs.delete();
    }
  }
  if (!model) throw new Error(`MuJoCo could not load ${name}`);
  const data = new mujoco.MjData(model);
  return { model, data };
}
