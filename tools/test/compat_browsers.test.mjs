// Cross-browser smoke test of the site (Playwright: Chromium, Firefox, WebKit).
// For every engine that is installed and can be launched:
//  1. capabilities the demo relies on: WebGL 2, DecompressionStream, import
//     maps, a module worker with a dynamic import, H.264 playback, WebGPU
//     (reported, not required); the CPU int8 kernel the nn engine picks for
//     the human network in a worker, with both kernels' times (reported);
//  2. the project page: the teaser video plays, "Start interactive demo"
//     inserts the iframe and brings it fully into view, the embedded demo gets
//     ready and starts, W reaches it, Space and the arrows do not scroll the
//     page, "Open full screen" puts the running iframe in full screen (the
//     session goes on; where the engine refuses, the link opens demo/), "Stop
//     demo" removes it;
//  3. demo/ on its own: no console errors, the worker loads MuJoCo and both
//     policies, the nn backend is chosen (the CPU path must work everywhere),
//     Start, W moves the model, W+A, pause stops the clock, R and the Reset
//     button reset, C cycles through the three camera modes; frame rate and
//     real-time factor;
//  4. phone emulation (iPhone 13 in WebKit, Pixel 7 in Chromium, a 390x844
//     touch viewport in Firefox): no horizontal scroll, the project page shows
//     the touch note instead of the key list, the demo loads from the project
//     page and on its own, its panels stay on screen.
//
//   cd tools
//   DUET_COMPAT_TESTS=1 PLAYWRIGHT_BROWSERS_PATH=<ms-playwright> \
//     node --test test/compat_browsers.test.mjs
//
// Install the engines with `npx playwright install chromium firefox webkit`.
// An engine that is not installed, or cannot start, is skipped with the reason.
// Options (environment):
//   COMPAT_ENGINES=chromium,firefox,webkit   the engines to try (default: all)
//   COMPAT_PORT=8831                         port of the in-process server
//   COMPAT_GPU=1     Chromium with the hardware-GPU flags (ANGLE on Vulkan,
//                    WebGPU), Firefox with dom.webgpu.enabled. Without it,
//                    Chromium renders with SwiftShader and nn 'auto' uses the CPU.
//   COMPAT_WEBKIT_LIBS=<dir>  LD_PRELOAD every .so in <dir> into WebKit and skip
//                    Playwright's host check: a user-space stand-in for missing
//                    system libraries (Ubuntu 24.04 lacks libavif16, libgav1-1
//                    and libyuv0; `apt-get download` them and `dpkg-deb -x` into
//                    <dir>). `sudo npx playwright install-deps webkit` is the
//                    proper fix.
//   COMPAT_SHOTS_DIR=<dir>   screenshots (default tools/out/compat/)
//
// The repository root is served on 127.0.0.1 with byte ranges (Safari and
// WebKit only play <video> from servers that answer Range requests, as GitHub
// Pages does) and .gz files as stored, as GitHub Pages sends them. Web fonts
// are blocked with a URL-pattern route so the test does not need the network.
// Do not switch that to a catch-all route(): in WebKit, intercepting every
// request makes the demo's worker fail with NotReadableError ("The I/O read
// operation failed.").

import { after, test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PORT = Number(process.env.COMPAT_PORT || 8831);
const BASE = `http://127.0.0.1:${PORT}`;
const OUT = process.env.COMPAT_SHOTS_DIR || fileURLToPath(new URL('../out/compat/', import.meta.url));
const SKIP = process.env.DUET_COMPAT_TESTS !== '1' && 'set DUET_COMPAT_TESTS=1';
const ENGINES = (process.env.COMPAT_ENGINES || 'chromium,firefox,webkit').split(',').map((s) => s.trim()).filter(Boolean);
const GPU = process.env.COMPAT_GPU === '1';
const FONTS = /fonts\.(googleapis|gstatic)\.com/;
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg',
  '.png': 'image/png', '.mp4': 'video/mp4', '.wasm': 'application/wasm', '.gz': 'application/gzip',
};
const PHONES = { chromium: 'Pixel 7', webkit: 'iPhone 13' };
// The tests wait for the demo itself (window.__duet), not for the load event.
const LOADED = { waitUntil: 'domcontentloaded', timeout: 60000 };

// ------------------------------------------------------------------ server
function serve(root, port) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, BASE);
    let file = path.join(root, decodeURIComponent(url.pathname));
    if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
    if (url.pathname.endsWith('/')) file = path.join(file, 'index.html');
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) { res.writeHead(404); res.end(); return; }
      const headers = {
        'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
        'Cache-Control': 'no-store',
        'Accept-Ranges': 'bytes',
      };
      const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
      if (range && (range[1] !== '' || range[2] !== '')) {
        const start = range[1] === '' ? Math.max(0, st.size - Number(range[2])) : Number(range[1]);
        const end = range[1] !== '' && range[2] !== '' ? Math.min(Number(range[2]), st.size - 1) : st.size - 1;
        if (start > end) { res.writeHead(416, { 'Content-Range': `bytes */${st.size}` }); res.end(); return; }
        res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${st.size}`, 'Content-Length': end - start + 1 });
        if (req.method === 'HEAD') { res.end(); return; }
        fs.createReadStream(file, { start, end }).pipe(res);
        return;
      }
      res.writeHead(200, { ...headers, 'Content-Length': st.size });
      if (req.method === 'HEAD') { res.end(); return; }
      fs.createReadStream(file).pipe(res);
    });
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () => resolve(server));
  });
}

let serverPromise = null;
const ensureServer = () => (serverPromise ||= serve(ROOT, PORT));
after(async () => {
  if (!serverPromise) return;
  const server = await serverPromise.catch(() => null);
  if (server) await new Promise((resolve) => server.close(resolve));
});

// ------------------------------------------------------------------ helpers
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// The useful lines of an error (Playwright's missing-library notice is boxed).
const summarize = (error, n = 8) => String((error && error.message) || error).split('\n')
  .map((line) => line.replace(/[\u2550-\u256c]/g, '').trim()).filter(Boolean).slice(0, n).join(' | ');

let pw = null;
async function playwright() {
  if (!pw) pw = await import('playwright');
  return pw;
}

function webkitEnv() {
  const dir = process.env.COMPAT_WEBKIT_LIBS;
  if (!dir) return undefined;
  const libs = fs.readdirSync(dir).filter((f) => /\.so(\.\d+)*$/.test(f)).map((f) => path.join(dir, f));
  process.env.PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = '1';
  return { ...process.env, LD_PRELOAD: [process.env.LD_PRELOAD, ...libs].filter(Boolean).join(' ') };
}

async function launch(engine) {
  const { chromium, firefox, webkit } = await playwright();
  const type = { chromium, firefox, webkit }[engine];
  if (!type) throw new Error(`unknown engine ${engine}`);
  const exe = type.executablePath();
  if (!exe || !fs.existsSync(exe)) throw new Error(`not installed (npx playwright install ${engine})`);
  if (engine === 'chromium') {
    const args = GPU
      ? ['--use-angle=vulkan', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist']
      : ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];
    return chromium.launch({ args });
  }
  if (engine === 'firefox') {
    return firefox.launch(GPU ? { firefoxUserPrefs: { 'dom.webgpu.enabled': true, 'gfx.webgpu.ignore-blocklist': true } } : {});
  }
  return webkit.launch({ env: webkitEnv() });
}

// A page that blocks the web fonts and records errors from this site.
async function openPage(context) {
  const page = await context.newPage();
  const errors = [];
  await page.route(FONTS, (route) => route.abort());
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('crash', () => errors.push('the page crashed'));
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const where = (m.location() && m.location().url) || '';
    if (FONTS.test(where) || FONTS.test(m.text())) return; // the blocked fonts
    if (where && !where.startsWith(BASE)) return;
    if (/Failed to load resource/.test(m.text()) && !where.startsWith(BASE)) return;
    errors.push(`console: ${m.text()} (${where})`);
  });
  return { page, errors };
}

const state = (target) => target.evaluate(() => window.__duet.state());

async function waitReady(target, timeout = 180000) {
  await target.waitForFunction(() => window.__duet && (window.__duet.state().ready || window.__duet.state().error),
    null, { timeout, polling: 250 });
  const s = await state(target);
  assert.equal(s.error, null, `the demo reported an error: ${s.error}`);
  return s;
}

function moved(a, b) {
  const progress = (b.x - a.x) * Math.cos(a.heading) + (b.y - a.y) * Math.sin(a.heading);
  const turn = Math.atan2(Math.sin(b.heading - a.heading), Math.cos(b.heading - a.heading));
  return { progress, turn };
}

// What the page needs from the engine (main thread and a module worker).
async function capabilities(page) {
  return page.evaluate(async () => {
    const out = {};
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    out.webgl2 = !!gl;
    if (gl) {
      // Firefox already reports a (sanitised) renderer and warns that the debug extension is deprecated.
      const debug = /Firefox\//.test(navigator.userAgent) ? null : gl.getExtension('WEBGL_debug_renderer_info');
      out.renderer = gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
    }
    out.decompressionStream = typeof DecompressionStream === 'function';
    out.importMaps = typeof HTMLScriptElement.supports === 'function' && HTMLScriptElement.supports('importmap');
    out.h264 = document.createElement('video').canPlayType('video/mp4; codecs="avc1.640029"');
    out.webgpu = !!navigator.gpu;
    const source = `self.onmessage = async () => {
      const r = { decompressionStream: typeof DecompressionStream === 'function', webgpu: !!(self.navigator && navigator.gpu) };
      try { const m = await import('${location.origin}/demo/js/nn/network.js'); r.dynamicImport = typeof m.parseNetwork === 'function'; }
      catch (e) { r.dynamicImport = String(e); }
      self.postMessage(r);
    };`;
    try {
      const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
      const worker = new Worker(url, { type: 'module' });
      out.worker = await new Promise((resolve) => {
        const timer = setTimeout(() => resolve({ error: 'no answer in 20 s' }), 20000);
        worker.onerror = (e) => { clearTimeout(timer); resolve({ error: e.message || 'worker error' }); };
        worker.onmessage = (e) => { clearTimeout(timer); resolve(e.data); };
        worker.postMessage(0);
      });
      worker.terminate();
    } catch (e) {
      out.worker = { error: String(e) };
    }
    return out;
  });
}

// The human network on the CPU in a module worker, as the demo's worker loads
// it: the int8 kernel the load-time race kept, both kernels' ms per inference,
// and a median over 5 more runs. JavaScriptCore keeps 'int8', V8 and
// SpiderMonkey keep 'packed' (demo/js/nn/cpu.js).
async function cpuKernels(page) {
  return page.evaluate(async () => {
    const source = `self.onmessage = async () => {
      try {
        const { loadPolicy } = await import('${location.origin}/demo/js/nn/index.js');
        const policy = await loadPolicy('${location.origin}/demo/assets/r2/human.bin.gz', { backend: 'cpu' });
        const x = new Float32Array(policy.inputSize);
        const times = [];
        for (let i = 0; i < 6; i++) { const t = performance.now(); await policy.run(x); times.push(performance.now() - t); }
        times.shift();
        times.sort((a, b) => a - b);
        self.postMessage({ kernel: policy.info.cpuKernel, race: policy.info.cpuRace, medianMs: times[2] });
      } catch (e) { self.postMessage({ error: String(e && e.stack || e) }); }
    };`;
    const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
    const worker = new Worker(url, { type: 'module' });
    try {
      return await new Promise((resolve) => {
        const timer = setTimeout(() => resolve({ error: 'no answer in 120 s' }), 120000);
        worker.onerror = (e) => { clearTimeout(timer); resolve({ error: e.message || 'worker error' }); };
        worker.onmessage = (e) => { clearTimeout(timer); resolve(e.data); };
        worker.postMessage(0);
      });
    } finally {
      worker.terminate();
    }
  });
}

// The frame's box in the viewport once a smooth scroll has settled.
async function settledBox(page, selector) {
  let last = null;
  for (let i = 0; i < 40; i++) {
    const box = await page.evaluate((s) => {
      const r = document.querySelector(s).getBoundingClientRect();
      return { top: r.top, bottom: r.bottom, height: r.height, view: innerHeight, scrollY };
    }, selector);
    if (last && Math.abs(box.top - last.top) < 0.5) return box;
    last = box;
    await sleep(100);
  }
  return last;
}

// ------------------------------------------------------------------ tests
for (const engine of ENGINES) {
  test(`${engine}: site and demo smoke test`, { skip: SKIP, timeout: 900000 }, async (t) => {
    await ensureServer();
    let browser;
    try {
      browser = await launch(engine);
    } catch (error) {
      const hint = engine === 'webkit' && !process.env.COMPAT_WEBKIT_LIBS ? ' (or see COMPAT_WEBKIT_LIBS in this file)' : '';
      t.skip(`${engine} cannot be launched: ${summarize(error)}${hint}`);
      return;
    }
    fs.mkdirSync(OUT, { recursive: true });
    t.diagnostic(`${engine} ${browser.version()}${GPU ? ' (COMPAT_GPU=1)' : ''}`);
    try {
      await t.test('capabilities', async (st) => {
        const context = await browser.newContext();
        try {
          const { page } = await openPage(context);
          await page.goto(`${BASE}/`, LOADED); // an HTML page of this origin (a secure context)
          const caps = await capabilities(page);
          st.diagnostic(JSON.stringify(caps));
          assert.ok(caps.webgl2, 'WebGL 2');
          assert.ok(caps.decompressionStream, 'DecompressionStream');
          assert.ok(caps.importMaps, 'import maps');
          assert.equal(caps.worker.dynamicImport, true, `module worker with a dynamic import: ${JSON.stringify(caps.worker)}`);
          assert.ok(caps.worker.decompressionStream, 'DecompressionStream in a worker');
          const cpu = await cpuKernels(page);
          const ms = (v) => (Number.isFinite(v) ? v.toFixed(1) : String(v));
          st.diagnostic(cpu.error ? `cpu kernels: ${cpu.error}` : `cpu kernel ${cpu.kernel} (race: packed ${ms(cpu.race.packed)} ms, ` +
            `int8 ${ms(cpu.race.int8)} ms; then ${ms(cpu.medianMs)} ms median per inference)`);
          assert.ok(!cpu.error, cpu.error);
          assert.ok(['packed', 'int8'].includes(cpu.kernel), `cpu kernel ${cpu.kernel}`);
        } finally {
          await context.close();
        }
      });

      await t.test('project page: teaser video, Start interactive demo, keys, full screen, Stop', async (st) => {
        const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
        try {
          const { page, errors } = await openPage(context);
          await page.goto(`${BASE}/`, LOADED);
          const h264 = await page.evaluate(() => document.createElement('video').canPlayType('video/mp4; codecs="avc1.640029"'));
          if (h264) {
            await page.waitForFunction(() => {
              const v = document.getElementById('teaser-video');
              return v.readyState >= 2 && !v.paused && v.currentTime > 0.5;
            }, null, { timeout: 15000 });
          } else {
            st.diagnostic(`${engine} cannot play H.264 here: the teaser shows its poster`);
          }
          const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
          assert.ok(overflow <= 0, `no horizontal scroll (${overflow} px)`);
          await page.click('#demo-start');
          const handle = await page.waitForSelector('iframe.demo-iframe', { timeout: 5000 });
          const frame = await handle.contentFrame();
          const t0 = Date.now();
          const ready = await waitReady(frame);
          let s = await state(frame);
          if (!s.running) {
            await sleep(1000);
            s = await state(frame);
          }
          const autoStarted = s.running;
          if (!s.running) {
            await frame.click('#start-btn');
            await frame.waitForFunction(() => window.__duet.state().running, null, { timeout: 5000 });
          }
          st.diagnostic(`embedded demo ready in ${((Date.now() - t0) / 1000).toFixed(1)} s, backend ${ready.backend}, ` +
            `${autoStarted ? 'started by itself' : 'needed a click on Start'}`);
          // Start brought the whole frame into view (site.js revealDemo).
          const box = await settledBox(page, '#demo-frame');
          st.diagnostic(`frame at ${box.top.toFixed(0)}-${box.bottom.toFixed(0)} px in a ${box.view} px window after Start`);
          assert.ok(box.top >= -0.5 && box.bottom <= box.view + 0.5, `the started demo is fully in view (${box.top.toFixed(0)}-${box.bottom.toFixed(0)} of ${box.view})`);
          const scroll0 = await page.evaluate(() => scrollY);
          const s0 = await state(frame);
          await page.keyboard.down('KeyW');
          await page.keyboard.press('Space');
          await page.keyboard.press('Space');
          await page.keyboard.press('ArrowDown');
          await sleep(3000);
          const s1 = await state(frame);
          const scroll1 = await page.evaluate(() => scrollY);
          await page.keyboard.up('KeyW');
          await page.screenshot({ path: path.join(OUT, `${engine}_landing_live.png`) });
          assert.deepEqual(s1.keys, ['KeyW'], 'the key reached the iframe');
          assert.equal(s1.action, 'forward');
          assert.ok(s1.simTime > s0.simTime, 'the embedded simulation advanced');
          assert.equal(scroll1, scroll0, 'Space and the arrow keys do not scroll the project page');
          const w = moved(s0.pose, s1.pose);
          st.diagnostic(`embedded: W for 3 s moved ${w.progress.toFixed(2)} m (${(s1.simTime - s0.simTime).toFixed(2)} s simulated), ${s1.fps.toFixed(0)} fps`);

          // "Open full screen" while the demo runs: the iframe itself goes full
          // screen and the same session goes on. An engine that refuses gets
          // demo/ as a page instead.
          await page.click('#demo-full');
          const fullscreen = await page.waitForFunction(
            () => document.fullscreenElement || document.webkitFullscreenElement || null, null, { timeout: 5000 },
          ).then((h) => h.evaluate((el) => el.className)).catch(() => null);
          if (!fullscreen) {
            await page.waitForURL(/\/demo\/$/, { timeout: 10000 });
            st.diagnostic(`${engine} refused element full screen: the link opened ${page.url()}`);
            assert.deepEqual(errors, []);
            return;
          }
          assert.equal(fullscreen, 'demo-iframe', 'the running iframe is in full screen');
          const f0 = await state(frame);
          await page.keyboard.down('KeyW');
          await sleep(1500);
          const f1 = await state(frame);
          await page.keyboard.up('KeyW');
          await page.screenshot({ path: path.join(OUT, `${engine}_landing_fullscreen.png`) });
          st.diagnostic(`full screen: ${(f1.simTime - f0.simTime).toFixed(2)} s simulated in 1.5 s, keys ${JSON.stringify(f1.keys)}`);
          assert.ok(f1.simTime > f0.simTime, 'the same session goes on in full screen');
          assert.deepEqual(f1.keys, ['KeyW'], 'keys reach the full-screen demo');
          await page.evaluate(() => (document.exitFullscreen ? document.exitFullscreen() : document.webkitExitFullscreen()));
          await page.waitForFunction(() => !(document.fullscreenElement || document.webkitFullscreenElement), null, { timeout: 5000 });

          await page.click('#demo-stop');
          assert.equal(await page.locator('iframe.demo-iframe').count(), 0, 'Stop demo removes the iframe');
          assert.deepEqual(errors, []);
        } finally {
          await context.close();
        }
      });

      await t.test('demo/: load, Start, WASD, pause, reset, camera, frame rate', async (st) => {
        const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
        try {
          const { page, errors } = await openPage(context);
          const t0 = Date.now();
          await page.goto(`${BASE}/demo/`, LOADED);
          const ready = await waitReady(page);
          const loadSeconds = (Date.now() - t0) / 1000;
          const info = await page.evaluate(() => window.__duet.stats());
          st.diagnostic(`ready in ${loadSeconds.toFixed(1)} s; human net ${info.backend.human} (${info.backend.adapter || info.backend.reason}), OSL net ${info.backend.osl}`);
          assert.equal(ready.phase, 'ready');
          assert.ok(ready.paused, 'paused until Start');
          assert.ok(['CPU', 'WebGPU'].includes(ready.backend));
          assert.ok(info.load && Number.isFinite(info.load.wasm) && Number.isFinite(info.load.model), `MuJoCo loaded (${JSON.stringify(info.load)})`);
          // frames.renders counts drawn frames (render on demand; 'drawn' before).
          assert.ok((ready.frames.renders ?? ready.frames.drawn) >= 1, 'the standing model is drawn behind the loader');

          await page.click('#start-btn');
          await page.waitForFunction(() => window.__duet.state().running && !window.__duet.state().paused, null, { timeout: 5000 });
          await sleep(500);
          const s0 = await state(page);
          // Hold W for 3 s of simulated time (a slow CPU path or a loaded
          // machine needs more wall time): the first second of a walk from
          // standing covers little ground.
          await page.keyboard.down('KeyW');
          const tw = Date.now();
          await page.waitForFunction((t) => window.__duet.state().simTime >= t, s0.simTime + 3, { timeout: 45000, polling: 250 })
            .catch(() => {});
          const s1 = await state(page);
          assert.equal(s1.hud.command, 'Forward');
          assert.ok(await page.evaluate(() => document.querySelector('kbd[data-key~="KeyW"]').classList.contains('pressed')), 'kbd W lights up');
          const walk = moved(s0.pose, s1.pose);
          const simulated = s1.simTime - s0.simTime;
          st.diagnostic(`W: ${walk.progress.toFixed(2)} m in ${simulated.toFixed(2)} s simulated (${((Date.now() - tw) / 1000).toFixed(1)} s wall)`);
          assert.ok(simulated >= 2.9, `the simulation advanced 3 s within 45 s of wall time (${simulated.toFixed(2)} s)`);
          assert.ok(walk.progress > 0.3, `walked forward (${walk.progress.toFixed(2)} m)`);
          await page.keyboard.down('KeyA');
          await sleep(1500);
          const s2 = await state(page);
          assert.equal(s2.hud.command, 'Forward, left');
          await page.screenshot({ path: path.join(OUT, `${engine}_demo_walk.png`) });
          await page.keyboard.up('KeyA');
          await page.keyboard.up('KeyW');
          await sleep(300);

          await page.keyboard.press('Space');
          await sleep(300);
          const p0 = await state(page);
          await sleep(800);
          const p1 = await state(page);
          assert.ok(p0.paused && p0.pauseReasons.includes('user'), 'Space pauses');
          assert.equal(p1.steps, p0.steps, 'no steps while paused');
          await page.keyboard.press('Space');
          await sleep(400);
          assert.ok(!(await state(page)).paused, 'Space resumes');

          const r0 = (await state(page)).resets;
          await page.keyboard.press('KeyR');
          await page.waitForFunction((n) => window.__duet.state().resets > n, r0, { timeout: 5000 });
          await page.click('#btn-reset');
          await page.waitForFunction((n) => window.__duet.state().resets > n + 1, r0, { timeout: 5000 });
          assert.ok((await state(page)).simTime < 1, 'back at the start');

          // C cycles through the three camera modes, from whichever the demo
          // starts in, back to it.
          const cameraMode = () => page.evaluate(() => document.getElementById('btn-camera').dataset.mode);
          const firstMode = await cameraMode();
          const modes = [];
          for (let i = 0; i < 3; i++) {
            await page.keyboard.press('KeyC');
            modes.push(await cameraMode());
          }
          assert.deepEqual([firstMode, ...modes.slice(0, 2)].sort(), ['chase', 'follow', 'free'], `camera modes ${firstMode} > ${modes.join(' > ')}`);
          assert.equal(modes[2], firstMode, 'three presses of C come back to the first mode');

          // Frame rate and real-time factor while walking.
          await sleep(500);
          const m0 = await state(page);
          await page.keyboard.down('KeyW');
          const fps = [];
          const tm = Date.now();
          while (Date.now() - tm < 5000) {
            await sleep(1000);
            fps.push((await state(page)).fps);
          }
          await page.keyboard.up('KeyW');
          const m1 = await state(page);
          const stats = await page.evaluate(() => window.__duet.stats());
          const meanFps = fps.slice(1).reduce((a, b) => a + b, 0) / Math.max(1, fps.length - 1);
          st.diagnostic(`${meanFps.toFixed(1)} fps, RTF ${m1.rtf === null ? '-' : m1.rtf.toFixed(2)}, step ${stats.timing.step.toFixed(1)} ms ` +
            `(nn ${stats.timing.policy.toFixed(1)}, physics ${stats.timing.physics.toFixed(1)}), ${m1.backend} inference, ` +
            `walked ${moved(m0.pose, m1.pose).progress.toFixed(2)} m, falls ${m1.falls}`);
          // Speed depends on the machine and its load: reported above, only
          // sanity-checked here.
          assert.ok(meanFps > 1, `the view renders (${meanFps.toFixed(1)} fps)`);
          assert.ok(m1.steps > m0.steps, 'the simulation runs');
          assert.deepEqual(errors, []);
        } finally {
          await context.close();
        }
      });

      await t.test('phone emulation: project page and demo/', async (st) => {
        const { devices } = await playwright();
        const options = PHONES[engine]
          ? { ...devices[PHONES[engine]] }
          : { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, hasTouch: true }; // Firefox: no isMobile
        const context = await browser.newContext(options);
        try {
          const { page, errors } = await openPage(context);
          await page.goto(`${BASE}/`, LOADED);
          const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
          assert.ok(overflow <= 0, `no horizontal scroll on the project page (${overflow} px)`);
          // Touch screens: no key list, and the note under Start names the pad.
          const touch = await page.evaluate(() => ({
            coarse: matchMedia('(pointer: coarse)').matches,
            keyList: getComputedStyle(document.querySelector('.key-list')).display,
            note: document.querySelector('.demo-note').textContent,
          }));
          st.diagnostic(`project page: pointer ${touch.coarse ? 'coarse' : 'fine'}, key list display ${touch.keyList}, note "${touch.note}"`);
          if (touch.coarse) {
            assert.equal(touch.keyList, 'none', 'no key list on a touch screen');
            assert.match(touch.note, /on-screen pad/);
          }
          await page.locator('#demo-start').scrollIntoViewIfNeeded();
          await page.tap('#demo-start');
          const handle = await page.waitForSelector('iframe.demo-iframe', { timeout: 5000 });
          const frame = await handle.contentFrame();
          const ready = await waitReady(frame);
          st.diagnostic(`${PHONES[engine] || '390x844 touch'}: embedded demo ready, backend ${ready.backend}`);
          await page.screenshot({ path: path.join(OUT, `${engine}_phone_embedded.png`) });
          await page.close(); // one simulation at a time

          const { page: demo, errors: demoErrors } = await openPage(context);
          await demo.goto(`${BASE}/demo/`, LOADED);
          await waitReady(demo);
          await demo.tap('#start-btn');
          await demo.waitForFunction(() => window.__duet.state().running, null, { timeout: 5000 });
          await sleep(1000);
          const layout = await demo.evaluate(() => {
            const out = { coarsePointer: matchMedia('(pointer: coarse)').matches };
            for (const [name, selector] of [['topbar', '.topbar'], ['help', '#help'], ['back', '#back-link']]) {
              const node = document.querySelector(selector);
              const style = getComputedStyle(node);
              if (node.hidden || style.display === 'none') continue;
              const b = node.getBoundingClientRect();
              out[name] = { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
            }
            out.width = innerWidth;
            out.height = innerHeight;
            return out;
          });
          st.diagnostic(`demo/ layout: ${JSON.stringify(layout)}`);
          for (const name of ['topbar', 'help', 'back']) {
            const b = layout[name];
            if (!b) continue;
            assert.ok(b.left >= -0.5 && b.top >= -0.5 && b.right <= layout.width + 0.5 && b.bottom <= layout.height + 0.5, `${name} on screen`);
          }
          await demo.screenshot({ path: path.join(OUT, `${engine}_phone_demo.png`) });
          assert.deepEqual([...errors, ...demoErrors], []);
        } finally {
          await context.close();
        }
      });
    } finally {
      await browser.close();
    }
  });
}
