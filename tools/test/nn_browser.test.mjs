// Browser check of the MLP engine: headless Chromium (Playwright) loads
// tools/test/nn_browser.html over http://127.0.0.1, which runs demo/js/nn inside
// a module Web Worker and compares WebGPU (SwiftShader in headless Chromium)
// with the CPU backend on the exported human network.
//
//   PLAYWRIGHT_BROWSERS_PATH=<browsers> node --test test/nn_browser.test.mjs
//
// Skips when Chromium cannot be launched or the exported assets are missing.
// Serves the repository root from this process on NN_TEST_PORT (default 8711,
// the policy agent's range 8711-8719) and closes it when done. SwiftShader is
// slow: allow a few minutes.
//
// NN_TEST_HARDWARE_GPU=1 adds --use-angle=vulkan --use-vulkan=native, which on
// a Linux machine with a Vulkan driver gives the worker the real GPU adapter
// (NVIDIA here: about 3 ms per human inference including the readback, and
// 'auto' picks WebGPU) instead of SwiftShader.

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PORT = Number(process.env.NN_TEST_PORT || 8711);
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.gz': 'application/octet-stream', // served as stored, like GitHub Pages serves .gz
};
const NEEDED = ['demo/assets/r2/human.bin.gz', 'demo/assets/r2/osl.bin.gz', 'tools/traces/nn_reference.bin.gz'];

function serve(root, port) {
  const server = http.createServer(async (request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
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
if (missing.length) skip = `missing ${missing.join(', ')}: run python -m duet_export --only policies`;
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

test('nn engine in a module worker in headless Chromium: WebGPU vs CPU', { skip, timeout: 600000 }, async (t) => {
  try {
    const hardware = process.env.NN_TEST_HARDWARE_GPU ? ['--use-angle=vulkan', '--use-vulkan=native'] : [];
    browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', ...hardware] });
  } catch (error) {
    t.skip(`cannot launch Chromium (set PLAYWRIGHT_BROWSERS_PATH): ${error.message.split('\n')[0]}`);
    return;
  }
  server = await serve(ROOT, PORT);
  const page = await browser.newPage();
  const messages = [];
  page.on('console', (message) => messages.push(message.text()));
  page.on('pageerror', (error) => messages.push(`page error: ${error.message}`));
  await page.goto(`http://127.0.0.1:${PORT}/tools/test/nn_browser.html`);
  let result;
  try {
    await page.waitForFunction(() => window.nnResult !== undefined, null, { timeout: 540000, polling: 500 });
    result = await page.evaluate(() => window.nnResult);
  } finally {
    for (const line of messages) if (!line.startsWith('{')) t.diagnostic(`page: ${line}`);
  }
  assert.ok(!result.error, result.error);
  const ms = (timing) => `${timing.median.toFixed(1)} ms median, ${timing.min.toFixed(1)} ms min`;
  t.diagnostic(`worker navigator.gpu: ${result.workerHasGpu}; adapter: ${result.gpuInfo.adapter}`);
  t.diagnostic(`cpu: load ${result.cpuLoadMs.toFixed(0)} ms, ${ms(result.cpuTiming)}, max |cpu - fp64 ref| ${result.cpuVsRef.toExponential(2)}; ` +
    `int8 kernel ${result.cpuKernel} (race ms ${JSON.stringify(result.cpuRace, (k, v) => (typeof v === 'number' ? +v.toFixed(2) : v))})`);
  t.diagnostic(`webgpu: load+self-test ${result.gpuLoadMs.toFixed(0)} ms (self-test max diff ${result.gpuInfo.selfTest.maxDiff.toExponential(2)}), ` +
    `${ms(result.gpuTiming)}, max |gpu - cpu| ${result.gpuVsCpu.toExponential(2)}, max |gpu - fp64 ref| ${result.gpuVsRef.toExponential(2)}`);
  t.diagnostic(`auto -> ${result.auto.backend} (${result.auto.reason})`);
  t.diagnostic(`osl auto -> ${result.osl.backend}, max |js - fp64 ref| ${result.osl.vsRef.toExponential(2)}, ${ms(result.osl.timing)}`);
  assert.equal(result.workerHasGpu, true);
  assert.ok(result.cpuVsRef <= 1e-4, `cpu vs reference ${result.cpuVsRef}`);
  assert.ok(['packed', 'int8'].includes(result.cpuKernel), `cpu kernel ${result.cpuKernel}`);
  assert.equal(result.kernelBitsDiffer, 0, 'the packed and int8 kernels give identical float32 bits');
  assert.ok(result.gpuVsCpu <= 1e-3, `gpu vs cpu ${result.gpuVsCpu}`);
  assert.ok(result.gpuVsRef <= 1e-3, `gpu vs reference ${result.gpuVsRef}`);
  assert.ok(result.osl.vsRef <= 1e-4, `osl vs reference ${result.osl.vsRef}`);
  assert.equal(result.osl.backend, 'cpu');
});
