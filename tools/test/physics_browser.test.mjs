// Browser check of the slim physics model: headless Chromium (Playwright) loads
// the vendored MuJoCo 3.11.0 WASM build and demo/assets/stage2/physics.mjb.gz
// (or DUET_ASSETS, when it lies inside the repository; see paths.mjs) inside
// a module Web Worker, through the demo's own loaders (fetch + gzip
// DecompressionStream), as demo/js/worker.js will. It checks the bytes and the
// counts against demo.json, replays the native trace of PHYSICS_REF
// (paths.mjs: tools/test/physics_wasm_ref.bin.gz for the demo's folder) and
// times mj_step.
//
//   PLAYWRIGHT_BROWSERS_PATH=<browsers> node --test test/physics_browser.test.mjs
//
// Skips when Chromium cannot be launched or the exported files are missing.
// Serves the repository root from this process on PHYSICS_TEST_PORT (default
// 8701, the physics agent's range 8701-8709) and closes it when done.

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { PHYSICS_REF, REPO, served, servedAssets } from './paths.mjs';

const ROOT = fileURLToPath(REPO);
const PORT = Number(process.env.PHYSICS_TEST_PORT || 8701);
const NATIVE_TOLERANCE = 1e-5; // as in physics.test.mjs
const ASSETS = servedAssets(); // '/demo/assets/stage2/'; null: DUET_ASSETS outside ROOT
const NATIVE = served(PHYSICS_REF); // '/tools/test/physics_wasm_ref.bin.gz' for the demo's folder
const NEEDED = ASSETS ? [`${ASSETS}physics.mjb.gz`, `${ASSETS}demo.json`, NATIVE] : [];
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.gz': 'application/octet-stream', // served as stored, like GitHub Pages serves .gz
};

// The page and its module worker, served from memory (no files in the repo).
const PAGE = `<!doctype html><meta charset="utf-8"><title>physics check</title>
<script type="module">
  const worker = new Worker('./worker.js', { type: 'module' });
  window.result = new Promise((resolve) => {
    worker.onmessage = (event) => resolve(event.data);
    worker.onerror = (event) => resolve({ ok: false, error: event.message || 'worker error' });
  });
  worker.postMessage('run');
</script>`;
const WORKER = `
import { fetchBytes, loadContainer, loadJSON } from '/demo/js/lib/container.js';
import { loadMujoco, modelFromBytes } from '/demo/js/lib/mujoco.js';

const hex = (buffer) => Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, '0')).join('');
const maxAbs = (a, b) => { let w = 0; for (let i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i] - b[i])); return w; };

self.onmessage = async () => {
  try {
    const t0 = performance.now();
    const mujoco = await loadMujoco();
    const t1 = performance.now();
    const bytes = await fetchBytes('${ASSETS}physics.mjb.gz');
    const t2 = performance.now();
    const { model, data } = modelFromBytes(mujoco, bytes, 'physics.mjb');
    const t3 = performance.now();
    const demo = await loadJSON('${ASSETS}demo.json');
    const native = await loadContainer('${NATIVE}');
    const counts = {};
    for (const key of Object.keys(demo.physics.counts)) counts[key] = model[key];
    const start = () => {
      mujoco.mj_resetData(model, data);
      data.qpos.set(native.arrays.qpos0);
      data.qvel.set(native.arrays.qvel0);
      mujoco.mj_forward(model, data);
    };
    const pattern = (t) => { const c = data.ctrl; for (let i = 0; i < model.nu; i++) c[i] = ((7 * i + 3 * t) % 11) / 10; };
    const { steps, every } = native.meta;
    const nq = model.nq, nv = model.nv, na = model.na;
    start();
    let worst = 0, ncon = true, finite = true;
    for (let t = 0; t < steps; t++) {
      pattern(t);
      mujoco.mj_step(model, data);
      if ((t + 1) % every) continue;
      const k = (t + 1) / every - 1;
      worst = Math.max(worst,
        maxAbs(data.qpos, native.arrays.qpos.subarray(k * nq, (k + 1) * nq)),
        maxAbs(data.qvel, native.arrays.qvel.subarray(k * nv, (k + 1) * nv)),
        maxAbs(data.act, native.arrays.act.subarray(k * na, (k + 1) * na)));
      ncon = ncon && data.ncon === native.arrays.ncon[k];
      finite = finite && data.qpos.every(Number.isFinite) && data.qvel.every(Number.isFinite);
    }
    start();
    for (let t = 0; t < 50; t++) { pattern(t); mujoco.mj_step(model, data); }
    const n = 500, ts = performance.now();
    for (let t = 0; t < n; t++) { pattern(t); mujoco.mj_step(model, data); }
    const stepMs = (performance.now() - ts) / n;
    self.postMessage({
      ok: true,
      sha256: hex(await crypto.subtle.digest('SHA-256', bytes)),
      rawBytes: bytes.length,
      counts,
      timestep: model.opt.timestep,
      worst, ncon, finite, stepMs,
      loadWasmMs: t1 - t0, fetchModelMs: t2 - t1, parseModelMs: t3 - t2,
      expected: { sha256: native.meta.physics_sha256, counts: demo.physics.counts, timestep: demo.physics.timestep, rawBytes: demo.physics.raw_bytes },
    });
    data.delete();
    model.delete();
  } catch (error) {
    self.postMessage({ ok: false, error: String((error && error.stack) || error) });
  }
};
`;

function serve(root, port) {
  const server = http.createServer(async (request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    if (pathname === '/__physics__/' || pathname === '/__physics__/worker.js') {
      const page = pathname.endsWith('/');
      response.writeHead(200, { 'content-type': page ? TYPES['.html'] : TYPES['.js'], 'cache-control': 'no-store' });
      response.end(page ? PAGE : WORKER);
      return;
    }
    const file = path.join(root, pathname);
    if (!file.startsWith(root)) {
      response.writeHead(403).end();
      return;
    }
    try {
      const body = await readFile(file);
      response.writeHead(200, {
        'content-type': TYPES[path.extname(file)] || 'application/octet-stream',
        'cache-control': 'no-store',
      });
      response.end(body);
    } catch {
      response.writeHead(404).end();
    }
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () => resolve(server));
  });
}

let chromium = null;
let skip = false;
const missing = NEEDED.filter((file) => !existsSync(path.join(ROOT, file)));
if (!ASSETS) skip = 'DUET_ASSETS lies outside the repository this test serves';
else if (missing.length) skip = `missing ${missing.join(', ')}: run python -m duet_export --only physics`;
if (!skip) {
  try {
    ({ chromium } = await import('playwright'));
  } catch (error) {
    skip = `playwright is not installed: ${error.message}`;
  }
}

let browser = null;
let server = null;
after(async () => {
  if (browser) await browser.close();
  if (server) await new Promise((resolve) => server.close(resolve));
});

test('slim physics model in a module worker in headless Chromium', { skip, timeout: 300000 }, async (t) => {
  try {
    browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
  } catch (error) {
    t.skip(`Chromium did not launch (set PLAYWRIGHT_BROWSERS_PATH): ${error.message.split('\n')[0]}`);
    return;
  }
  server = await serve(ROOT, PORT);
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${PORT}/__physics__/`);
  const result = await page.evaluate(() => window.result);
  assert.ok(result.ok, result.error);
  assert.deepEqual(errors, []);
  const { expected } = result;
  assert.equal(result.sha256, expected.sha256, 'browser gunzip of physics.mjb.gz differs from the exported model');
  assert.equal(result.rawBytes, expected.rawBytes);
  assert.deepEqual(result.counts, expected.counts);
  assert.equal(result.timestep, expected.timestep);
  assert.ok(result.finite, 'non-finite state');
  assert.ok(result.ncon, 'contact counts differ from native');
  assert.ok(result.worst <= NATIVE_TOLERANCE, `max |diff| ${result.worst} > ${NATIVE_TOLERANCE}`);
  t.diagnostic(`Chromium worker: WASM vs native max |diff| ${result.worst.toExponential(3)}`);
  t.diagnostic(
    `load wasm ${result.loadWasmMs.toFixed(0)} ms, fetch+gunzip model ${result.fetchModelMs.toFixed(0)} ms, ` +
      `parse model ${result.parseModelMs.toFixed(0)} ms; mj_step ${result.stepMs.toFixed(3)} ms`,
  );
});
