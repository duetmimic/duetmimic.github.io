// Browser end-to-end test of the demo (Playwright, headless Chromium):
//  1. demo/ on its own: loads without console errors; Space on the focused
//     Start button starts it, with the chase camera; real key presses walk
//     and turn; checks the HUD and its first-run hint, the kbd highlights,
//     pause (the clock stops and nothing is drawn), speed levels, camera
//     modes, help, Space on a focused button, reset, clearing the keys on
//     blur and keycap labels for an AZERTY keyboard; measures frame rate and RTF.
//  2. The project page: "Start interactive demo" inserts the iframe, the demo
//     becomes ready and starts with the one-line legend, key presses reach it,
//     a plain mouse wheel scrolls the page while Ctrl + wheel zooms, and the
//     demo pauses (and stops drawing) while scrolled out of view.
//  3. Touch (a phone-sized touch screen): the pad shows, its forward arrow
//     starts the demo and walks while held, and lets go on release.
//  4. Layout: the HUD panels and the touch pad never overlap and the longest
//     readouts fit, at 21 x 15 window sizes, standalone and embedded, with and
//     without touch, in each help mode.
//  5. With E2E_GPU=1, demo/ again on the hardware GPU (ANGLE on Vulkan for
//     WebGL, WebGPU on the GPU adapter): reports backends, frame rate and RTF.
//
//   cd tools
//   DUET_BROWSER_TESTS=1 PLAYWRIGHT_BROWSERS_PATH=<ms-playwright> \
//     [E2E_GPU=1] [E2E_SHOTS_DIR=<folder>] node --test test/e2e_browser.test.mjs
//
// Opt-in (DUET_BROWSER_TESTS=1): it runs the real simulation for a few
// minutes. The default launch renders with SwiftShader, and nn 'auto' skips its
// software WebGPU adapter, so inference runs on the CPU. Walking and turning
// are judged over simulated time (waitSim), so a slow or loaded machine runs
// them longer instead of failing them. Screenshots go to E2E_SHOTS_DIR (default
// tools/out/e2e/). The repository root is served on 127.0.0.1:E2E_PORT
// (default 8741) while the test runs; requests to other hosts (web fonts) are
// blocked so the test does not depend on the network.

import { after, before, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PORT = Number(process.env.E2E_PORT || 8741);
const BASE = `http://127.0.0.1:${PORT}`;
const OUT = process.env.E2E_SHOTS_DIR || fileURLToPath(new URL('../out/e2e/', import.meta.url));
const SKIP = process.env.DUET_BROWSER_TESTS !== '1' && 'set DUET_BROWSER_TESTS=1';
const SOFTWARE = ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];
const HARDWARE = ['--use-angle=vulkan', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'];
const SIM_TIMEOUT_MS = 120000; // wall time allowed for a few simulated seconds
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg',
  '.png': 'image/png', '.mp4': 'video/mp4', '.gz': 'application/gzip', // .gz as GitHub Pages sends it: no Content-Encoding
};

function serve(root, port) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, BASE);
    let file = path.join(root, decodeURIComponent(url.pathname));
    if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
    if (url.pathname.endsWith('/')) file = path.join(file, 'index.html');
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, {
        'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
        'Content-Length': st.size,
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(file).pipe(res);
    });
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () => resolve(server));
  });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A page that blocks other hosts and records console errors and page errors.
async function openPage(browser, viewport = { width: 1280, height: 720 }, contextOptions = null) {
  const page = contextOptions ? await (await browser.newContext(contextOptions)).newPage() : await browser.newPage({ viewport });
  const errors = [];
  await page.route((url) => !url.href.startsWith(BASE), (route) => route.abort());
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const where = (m.location() && m.location().url) || '';
    if (where && !where.startsWith(BASE)) return; // a blocked font or other external file
    if (/Failed to load resource/.test(m.text()) && !where.startsWith(BASE)) return;
    errors.push(`console: ${m.text()} (${where})`);
  });
  return { page, errors };
}

// Forward distance along the starting heading, and the heading change.
function moved(a, b) {
  const progress = (b.x - a.x) * Math.cos(a.heading) + (b.y - a.y) * Math.sin(a.heading);
  const turn = Math.atan2(Math.sin(b.heading - a.heading), Math.cos(b.heading - a.heading));
  return { progress, turn };
}

const state = (target) => target.evaluate(() => window.__duet.state());

async function waitReady(target, timeout = 120000) {
  await target.waitForFunction(() => window.__duet && (window.__duet.state().ready || window.__duet.state().error),
    null, { timeout, polling: 200 });
  const s = await state(target);
  assert.equal(s.error, null, `the demo reported an error: ${s.error}`);
  return s;
}

// Wait until the simulated clock is `seconds` past `from` (s), however long
// that takes in wall time on this machine.
async function waitSim(target, from, seconds, timeout = SIM_TIMEOUT_MS) {
  await target.waitForFunction(([t0, dt]) => window.__duet.state().simTime >= t0 + dt, [from, seconds], { timeout, polling: 50 });
  return state(target);
}

// Hold W for `seconds` of simulated time, sampling fps and RTF from the page.
async function measure(page, target, seconds) {
  const before = await state(target);
  await page.keyboard.down('KeyW');
  const fps = [];
  const t0 = Date.now();
  for (;;) {
    await sleep(1000);
    const s = await state(target);
    fps.push(s.fps);
    if (s.simTime >= before.simTime + seconds || Date.now() - t0 > SIM_TIMEOUT_MS) break;
  }
  await page.keyboard.up('KeyW');
  const after = await state(target);
  const wall = (Date.now() - t0) / 1000;
  return {
    fps: fps.slice(1).reduce((a, b) => a + b, 0) / Math.max(1, fps.length - 1),
    rtf: after.rtf,
    simRate: (after.steps - before.steps) * 0.01 / wall,
    simSeconds: after.simTime - before.simTime,
    ...moved(before.pose, after.pose),
    backend: after.backend,
    falls: after.falls,
  };
}

// Draws over `ms` of wall time (render on demand: none while nothing changes).
async function rendersOver(target, ms) {
  const a = (await state(target)).frames;
  await sleep(ms);
  const b = (await state(target)).frames;
  return { renders: b.renders - a.renders, rafs: b.rafs - a.rafs };
}

let server = null;
let chromium = null;
let devices = null;
const results = [];

describe('demo in headless Chromium', { skip: SKIP }, () => {
  before(async () => {
    ({ chromium, devices } = await import('playwright'));
    fs.mkdirSync(OUT, { recursive: true });
    server = await serve(ROOT, PORT);
  });

  after(async () => {
    if (server) await new Promise((resolve) => server.close(resolve));
  });

  test('standalone demo: load, Start, WASD, HUD, pause, levels, camera, help, reset, AZERTY keycaps', { timeout: 300000 }, async (t) => {
    const browser = await chromium.launch({ args: SOFTWARE });
    try {
      const { page, errors } = await openPage(browser);
      const t0 = Date.now();
      await page.goto(`${BASE}/demo/`);
      const ready = await waitReady(page);
      const loadSeconds = (Date.now() - t0) / 1000;
      assert.equal(ready.phase, 'ready');
      assert.ok(ready.paused, 'paused until Start');
      const loader = await page.evaluate(() => ({
        status: document.getElementById('loader-status').textContent,
        size: document.getElementById('loader-size').textContent,
        sizeHidden: document.getElementById('loader-size').hidden,
        bar: document.getElementById('loader-bar').style.width,
        start: !document.getElementById('start-btn').disabled,
        focused: document.activeElement && document.activeElement.id,
        hudInert: document.getElementById('hud').inert,
        helpHidden: document.getElementById('help').hidden,
        devRows: [document.getElementById('hud-clip'), document.getElementById('hud-time')].map((dd) => dd.parentElement.hidden),
      }));
      assert.equal(loader.status, 'Ready. Press Start, then walk with W A S D.');
      assert.match(loader.size, /About \d+ MB download/);
      assert.ok(loader.sizeHidden, 'the size line gives way to the download count');
      assert.equal(parseFloat(loader.bar), 100, `loader bar at ${loader.bar}`);
      assert.ok(loader.start, 'Start is enabled');
      assert.equal(loader.focused, 'start-btn', 'Start has the keyboard focus');
      assert.ok(loader.hudInert && loader.helpHidden, 'the HUD is inert and the help hidden behind the loader');
      assert.deepEqual(loader.devRows, [true, true], 'Gait clip and Sim time only with ?debug=1');
      assert.equal(ready.hud.command, 'Hold W to walk');
      assert.equal(ready.camera, 'follow', 'the ready screen keeps the front-right view');
      assert.ok(ready.frames.renders >= 1, 'the standing model is drawn behind the loader');
      await page.screenshot({ path: path.join(OUT, 'e2e_demo_ready.png') });

      // Space on the focused Start button starts the demo (its keyup is not cancelled).
      await page.keyboard.press('Space');
      await page.waitForFunction(() => window.__duet.state().running && !window.__duet.state().paused, null, { timeout: 5000 });
      const started = await state(page);
      assert.equal(started.camera, 'chase', 'Start switches to the chase camera');
      assert.equal(started.help, 'full');
      assert.equal(await page.evaluate(() => document.getElementById('hud').inert), false);
      await sleep(600);
      const s0 = await state(page);

      // W for 3 s of simulated time.
      await page.keyboard.down('KeyW');
      await sleep(300);
      const pressed = await page.evaluate(() => document.querySelector('kbd[data-key~="KeyW"]').classList.contains('pressed'));
      assert.ok(pressed, 'kbd W lights up while held');
      const s1 = await waitSim(page, s0.simTime, 3);
      assert.equal(s1.hud.command, 'Forward');
      assert.equal(s1.action, 'forward');
      await page.screenshot({ path: path.join(OUT, 'e2e_demo_walk.png') });
      const walk = moved(s0.pose, s1.pose);
      t.diagnostic(`W: ${walk.progress.toFixed(2)} m forward in ${(s1.simTime - s0.simTime).toFixed(2)} s simulated, turn ${walk.turn.toFixed(2)} rad, RTF ${s1.rtf && s1.rtf.toFixed(2)}`);
      assert.ok(walk.progress > 0.3, `walked forward ${walk.progress.toFixed(2)} m`);

      // W+A for 2 s of simulated time.
      await page.keyboard.down('KeyA');
      const s2 = await waitSim(page, s1.simTime, 2);
      assert.equal(s2.hud.command, 'Forward, left');
      const turn = moved(s1.pose, s2.pose);
      t.diagnostic(`W+A: ${turn.progress.toFixed(2)} m forward, turned ${turn.turn.toFixed(2)} rad in ${(s2.simTime - s1.simTime).toFixed(2)} s simulated`);
      assert.ok(turn.turn > 0.1, `turned left (${turn.turn.toFixed(2)} rad)`);
      await page.screenshot({ path: path.join(OUT, 'e2e_demo_turn.png') });
      await page.keyboard.up('KeyA');
      await page.keyboard.up('KeyW');
      await sleep(400);
      const hud = await state(page);
      assert.equal(hud.hud.command, 'Stand');
      assert.equal(await page.evaluate(() => document.querySelectorAll('kbd.pressed:not([data-key^="Digit"])').length), 0, 'released keys go dark');
      assert.match(hud.hud.rtf, /^\d\.\d\d×$/);
      assert.ok(['CPU', 'WebGPU'].includes(hud.hud.backend));
      assert.equal(hud.hud.note, hud.backend === 'CPU', 'the CPU note shows exactly when inference is on the CPU');
      if (hud.backend === 'CPU') assert.match(hud.hud.noteText, /CPU/);

      // Space pauses: the simulated clock stops and, once the view is still,
      // nothing is drawn; Space resumes without a catch-up burst.
      await page.keyboard.press('Space');
      await sleep(300);
      const p0 = await state(page);
      assert.ok(p0.paused && p0.pauseReasons.includes('user'));
      const toast = await page.evaluate(() => document.getElementById('toast').textContent);
      assert.match(toast, /Paused/);
      const idle = await rendersOver(page, 1000);
      const p1 = await state(page);
      assert.equal(p1.steps, p0.steps, 'no steps while paused');
      t.diagnostic(`paused: ${idle.renders} draws in ${idle.rafs} animation frames`);
      assert.ok(idle.renders <= 1, `nothing drawn while paused (${idle.renders} draws)`);
      await page.keyboard.press('Space');
      await sleep(500);
      const p2 = await state(page);
      assert.ok(!p2.paused && p2.steps > p1.steps, 'resumed');
      assert.ok(p2.steps - p1.steps < 80, `no catch-up burst after the pause (${p2.steps - p1.steps} steps in 0.5 s)`);

      // Speed levels: Digit5 then Digit3; Q (bound, though not listed).
      await page.keyboard.press('Digit5');
      await sleep(300);
      assert.equal((await state(page)).level, 4);
      assert.ok(await page.evaluate(() => document.querySelector('kbd[data-key="Digit5"]').classList.contains('pressed')), 'the selected level is lit');
      await page.keyboard.press('KeyQ');
      await sleep(300);
      assert.equal((await state(page)).level, 3);
      await page.keyboard.press('Digit3');
      await sleep(300);
      assert.equal((await state(page)).level, 2);

      // Camera modes: chase -> follow -> free -> chase, named in the toast.
      const modes = [];
      for (let i = 0; i < 3; i++) {
        await page.keyboard.press('KeyC');
        modes.push(await page.evaluate(() => [document.getElementById('btn-camera').dataset.mode, document.getElementById('toast').textContent]));
      }
      assert.deepEqual(modes.map((m) => m[0]), ['follow', 'free', 'chase']);
      assert.deepEqual(modes.map((m) => m[1]), ['Camera: fixed angle (drag to orbit)', 'Camera: free (drag to orbit, right-drag to pan)', 'Camera: behind the walker']);

      // Help: H hides it, the button shows it; Space on the focused button
      // toggles it (once).
      await page.keyboard.press('KeyH');
      assert.ok(await page.evaluate(() => document.getElementById('help').hidden));
      await page.click('#btn-help');
      assert.ok(await page.evaluate(() => !document.getElementById('help').hidden));
      assert.equal(await page.evaluate(() => document.getElementById('btn-help').getAttribute('aria-expanded')), 'true');
      await page.focus('#btn-help');
      await page.keyboard.press('Space');
      assert.equal((await state(page)).help, 'hidden', 'Space on a focused button presses it');
      await page.keyboard.press('Space');
      assert.equal((await state(page)).help, 'full');
      await page.evaluate(() => document.activeElement.blur());

      // Reset (R) and the button.
      const r0 = await state(page);
      await page.keyboard.press('KeyR');
      await page.waitForFunction((n) => window.__duet.state().resets > n, r0.resets, { timeout: 5000 });
      await page.click('#btn-reset');
      await page.waitForFunction((n) => window.__duet.state().resets > n + 1, r0.resets, { timeout: 5000 });
      const r1 = await state(page);
      assert.ok(r1.simTime < 1, 'back at the start');
      assert.equal(await page.evaluate(() => document.activeElement === document.body), true, 'a clicked button gives the keyboard back');

      // Keys are cleared when the window loses focus.
      await page.keyboard.down('KeyW');
      await sleep(400);
      assert.deepEqual((await state(page)).keys, ['KeyW']);
      await page.evaluate(() => window.dispatchEvent(new Event('blur')));
      await sleep(400);
      const b = await state(page);
      assert.deepEqual(b.keys, []);
      assert.equal(b.hud.command, 'Stand');
      await page.keyboard.up('KeyW');

      // An AZERTY keyboard: the W position types 'z'. Bindings stay put, the
      // keycaps follow the layout, once with a toast.
      await page.evaluate(() => window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW', key: 'z' })));
      const azerty = await page.evaluate(() => ({
        keys: window.__duet.state().keys,
        caps: ['KeyW', 'KeyA', 'KeyS', 'KeyD'].map((c) => document.querySelector(`kbd[data-label="${c}"]`).textContent).join(''),
        toast: document.getElementById('toast').textContent,
      }));
      await page.evaluate(() => window.dispatchEvent(new KeyboardEvent('keyup', { code: 'KeyW', key: 'z' })));
      assert.deepEqual(azerty.keys, ['KeyW']);
      assert.equal(azerty.caps, 'ZQSD');
      assert.match(azerty.toast, /Z Q S D/);

      // Frame rate and real-time factor while walking.
      const m = await measure(page, page, 4);
      results.push(`SwiftShader WebGL, ${m.backend} inference: ${m.fps.toFixed(1)} fps, RTF ${m.rtf.toFixed(2)} ` +
        `(sim ${m.simRate.toFixed(2)} s per s), ${loadSeconds.toFixed(1)} s to ready, falls ${m.falls}`);
      t.diagnostic(results[results.length - 1]);
      assert.deepEqual(errors, []);
    } finally {
      await browser.close();
    }
  });

  test('project page: Start interactive demo -> iframe -> ready -> keys, wheel, off-screen pause', { timeout: 240000 }, async (t) => {
    const browser = await chromium.launch({ args: SOFTWARE });
    try {
      const { page, errors } = await openPage(browser, { width: 1280, height: 900 });
      await page.goto(`${BASE}/`);
      await page.click('#demo-start');
      const handle = await page.waitForSelector('iframe.demo-iframe', { timeout: 5000 });
      const frame = await handle.contentFrame();
      await frame.waitForLoadState('domcontentloaded');
      assert.match(frame.url(), /demo\/\?embed=1$/);
      const ready = await waitReady(frame);
      t.diagnostic(`embedded demo ready: phase ${ready.phase}, backend ${ready.backend}`);
      // Embedded and focused by the parent's click: it starts on its own.
      // Without focus it would wait for a click on Start inside the frame.
      let s = await state(frame);
      if (!s.running) {
        await sleep(1000);
        s = await state(frame);
      }
      if (!s.running) {
        t.diagnostic('the iframe did not have the focus: clicking Start inside it');
        await frame.click('#start-btn');
        await frame.waitForFunction(() => window.__duet.state().running, null, { timeout: 5000 });
      }
      await page.locator('#demo').scrollIntoViewIfNeeded();
      await sleep(500);
      const s0 = await state(frame);
      assert.equal(s0.help, 'compact', 'embedded, the help is the one-line legend');
      assert.equal(s0.camera, 'chase');
      await page.keyboard.down('KeyW');
      const s1 = await waitSim(frame, s0.simTime, 2.5);
      await page.screenshot({ path: path.join(OUT, 'e2e_landing_live.png') });
      await page.keyboard.up('KeyW');
      assert.deepEqual(s1.keys, ['KeyW'], 'the key reached the iframe');
      assert.equal(s1.action, 'forward');
      const w = moved(s0.pose, s1.pose);
      t.diagnostic(`embedded: W for ${(s1.simTime - s0.simTime).toFixed(2)} s simulated moved ${w.progress.toFixed(2)} m, fps ${s1.fps.toFixed(1)}`);
      assert.ok(w.progress > 0.2, `walked ${w.progress.toFixed(2)} m`);

      // A plain wheel over the demo scrolls the page; Ctrl + wheel zooms.
      const box = await handle.boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      const y0 = await page.evaluate(() => scrollY);
      const d0 = (await state(frame)).cameraDistance;
      await page.mouse.wheel(0, 200);
      await sleep(500);
      const y1 = await page.evaluate(() => scrollY);
      assert.ok(y1 > y0, `a plain wheel scrolls the page (${y0} -> ${y1})`);
      assert.ok(Math.abs((await state(frame)).cameraDistance - d0) < 1e-6, 'and leaves the camera alone');
      await page.evaluate((y) => window.scrollTo(0, y), y0);
      await sleep(300);
      const box2 = await handle.boundingBox();
      await page.mouse.move(box2.x + box2.width / 2, box2.y + box2.height / 2);
      await page.keyboard.down('Control');
      await page.mouse.wheel(0, 200);
      await page.keyboard.up('Control');
      await sleep(500);
      assert.equal(await page.evaluate(() => scrollY), y0, 'Ctrl + wheel does not scroll the page');
      assert.ok((await state(frame)).cameraDistance > d0 + 0.05, 'Ctrl + wheel zooms');

      // Scrolled out of view: paused, nothing drawn; back in view: running.
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForFunction(() => document.querySelector('iframe.demo-iframe').contentWindow.__duet.state().pauseReasons.includes('offscreen'), null, { timeout: 5000 });
      const o0 = await state(frame);
      const off = await rendersOver(frame, 1000);
      const o1 = await state(frame);
      assert.equal(o1.steps, o0.steps, 'no steps off screen');
      assert.equal(off.renders, 0, 'nothing drawn off screen');
      await page.locator('#demo').scrollIntoViewIfNeeded();
      await frame.waitForFunction(() => !window.__duet.state().paused, null, { timeout: 5000 });

      // Stop demo removes the iframe.
      await page.click('#demo-stop');
      assert.equal(await page.locator('iframe.demo-iframe').count(), 0);
      assert.deepEqual(errors, []);
    } finally {
      await browser.close();
    }
  });

  test('touch screen: the pad starts the demo, walks while held, lets go', { timeout: 240000 }, async (t) => {
    const browser = await chromium.launch({ args: SOFTWARE });
    try {
      // A phone-sized touch screen; 1x pixels keep SwiftShader quick.
      const { page, errors } = await openPage(browser, null, { ...devices['iPhone 13'], deviceScaleFactor: 1 });
      const cdp = await page.context().newCDPSession(page);
      const touch = async (type, selector) => {
        const b = selector ? await page.locator(selector).boundingBox() : null;
        await cdp.send('Input.dispatchTouchEvent', { type, touchPoints: b ? [{ x: b.x + b.width / 2, y: b.y + b.height / 2, id: 1 }] : [] });
      };
      await page.goto(`${BASE}/demo/`);
      const ready = await waitReady(page);
      const shown = await page.evaluate(() => ({
        touch: document.documentElement.classList.contains('touch'),
        pad: getComputedStyle(document.getElementById('touch-pad')).display,
        status: document.getElementById('loader-status').textContent,
      }));
      assert.ok(shown.touch && shown.pad === 'grid', 'the pad shows on a coarse pointer');
      assert.equal(shown.status, 'Ready. Press Start, then hold ▲ to walk.');
      assert.equal(ready.hud.command, 'Hold ▲ to walk');
      await touch('touchStart', '.pad-up');
      await page.waitForFunction(() => window.__duet.state().running, null, { timeout: 5000 });
      const s0 = await state(page);
      assert.deepEqual(s0.keys, ['KeyW']);
      assert.equal(s0.help, 'hidden', 'touch screens start without the help panel');
      const s1 = await waitSim(page, s0.simTime, 2);
      await page.screenshot({ path: path.join(OUT, 'e2e_touch_walk.png') });
      await touch('touchEnd');
      await sleep(300);
      const s2 = await state(page);
      const w = moved(s0.pose, s1.pose);
      t.diagnostic(`touch: ▲ held for ${(s1.simTime - s0.simTime).toFixed(2)} s simulated moved ${w.progress.toFixed(2)} m`);
      assert.ok(w.progress > 0.2, `walked ${w.progress.toFixed(2)} m`);
      assert.deepEqual(s2.keys, [], 'lifting the finger lets go');
      assert.deepEqual(errors, []);
    } finally {
      await browser.close();
    }
  });

  test('layout: HUD panels and the touch pad never overlap, readouts fit (standalone and embedded)', { timeout: 300000 }, async (t) => {
    const browser = await chromium.launch({ args: SOFTWARE });
    try {
      // [query, touch, help mode]: the help as each start shows it, and full.
      const cases = [['', false, 'full'], ['?embed=1', false, 'compact'], ['?embed=1', false, 'full'], ['', true, 'full'], ['?embed=1', true, 'full']];
      for (const [query, touch, mode] of cases) {
        const { page } = await openPage(browser);
        // No simulation needed: the assets fail to load and the error overlay is hidden below.
        await page.route('**/assets/**', (route) => route.fulfill({ status: 404, body: '' }));
        await page.goto(`${BASE}/demo/${query}`);
        await page.waitForFunction(() => window.__duet && window.__duet.state().error, null, { timeout: 30000 });
        await page.evaluate(([touchMode, helpMode]) => {
          document.getElementById('error').hidden = true;
          document.getElementById('loader').hidden = true;
          document.documentElement.classList.toggle('touch', touchMode);
          document.documentElement.dataset.phase = 'running';
          document.getElementById('hud').inert = false;
          const help = document.getElementById('help');
          help.hidden = false;
          help.dataset.mode = helpMode;
          const longest = {
            'hud-command': 'Backward, turning right', 'hud-speed': '-0.50 m/s, -0.25 rad/s',
            'hud-clip': 'fwd 1.00, +0.50 rad/s', 'hud-time': '999.9 s', 'hud-rtf': '0.58×', 'hud-backend': 'WebGPU',
          };
          for (const [id, text] of Object.entries(longest)) document.getElementById(id).textContent = text;
          document.getElementById('hud-note').hidden = false;
          document.querySelector('#hud-note dd').textContent = "This browser's WebGPU is too slow for per-step inference, so the networks run on the CPU in slow motion. A browser with fast WebGPU (recent Chrome, Edge or Safari) runs it in real time.";
        }, [touch, mode]);
        const problems = [];
        let sizes = 0;
        for (let w = 320; w <= 1920; w += 80) {
          for (let h = 360; h <= 1200; h += 60) {
            await page.setViewportSize({ width: w, height: h });
            sizes += 1;
            const found = await page.evaluate(() => {
              const boxes = {};
              for (const [name, selector] of [['back', '#back-link'], ['top', '.topbar'], ['stats', '.stats'], ['help', '#help'], ['pad', '#touch-pad']]) {
                const node = document.querySelector(selector);
                const style = node && getComputedStyle(node);
                if (!node || style.display === 'none' || style.visibility === 'hidden') continue;
                const b = node.getBoundingClientRect();
                if (b.width > 0) boxes[name] = b;
              }
              const out = [];
              const names = Object.keys(boxes);
              for (let i = 0; i < names.length; i++) {
                for (let j = i + 1; j < names.length; j++) {
                  const a = boxes[names[i]], b = boxes[names[j]];
                  if (a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5) out.push(`${names[i]} overlaps ${names[j]}`);
                }
              }
              for (const [name, b] of Object.entries(boxes)) {
                if (b.left < -0.5 || b.right > innerWidth + 0.5 || b.top < -0.5 || b.bottom > innerHeight + 0.5) out.push(`${name} off screen`);
              }
              for (const dd of document.querySelectorAll('.stats dd')) {
                if (dd.offsetParent && dd.scrollWidth > dd.clientWidth + 1) out.push(`${dd.id || 'note'} cut off`);
              }
              return out;
            });
            if (found.length) problems.push(`${w}x${h}${query}${touch ? ' touch' : ''} ${mode}: ${found.join(', ')}`);
          }
        }
        t.diagnostic(`demo/${query}${touch ? ' (touch)' : ''}, help ${mode}: ${sizes} window sizes checked`);
        assert.deepEqual(problems, []);
        await page.close();
      }
    } finally {
      await browser.close();
    }
  });

  test('hardware GPU (E2E_GPU=1): backends, frame rate, RTF', { skip: process.env.E2E_GPU !== '1' && 'set E2E_GPU=1', timeout: 240000 }, async (t) => {
    const browser = await chromium.launch({ args: HARDWARE });
    try {
      // 'auto' (WebGPU inference on this adapter), then ?backend=cpu: a GPU
      // machine whose browser has no WebGPU.
      for (const query of ['', '?backend=cpu']) {
        const { page, errors } = await openPage(browser);
        await page.goto(`${BASE}/demo/${query}`);
        const gl = await page.evaluate(() => {
          const g = document.createElement('canvas').getContext('webgl2');
          const ext = g && g.getExtension('WEBGL_debug_renderer_info');
          return g ? g.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : g.RENDERER) : 'none';
        });
        await waitReady(page);
        await page.click('#start-btn');
        await sleep(500);
        const m = await measure(page, page, 5);
        const stats = await page.evaluate(() => window.__duet.stats());
        if (!query) await page.screenshot({ path: path.join(OUT, 'e2e_gpu_walk.png') });
        results.push(`hardware${query ? ` ${query}` : ''}: WebGL ${gl}; inference ${m.backend} ` +
          `(${stats.backend.adapter || stats.backend.reason}): ${m.fps.toFixed(1)} fps, RTF ${m.rtf.toFixed(2)}, ` +
          `step ${stats.timing.step.toFixed(2)} ms (nn ${stats.timing.policy.toFixed(2)}, physics ${stats.timing.physics.toFixed(2)}), ` +
          `frame lag ${(1000 * stats.render.lag).toFixed(1)} ms, falls ${m.falls}`);
        t.diagnostic(results[results.length - 1]);
        assert.ok(m.progress > 1, `walked ${m.progress.toFixed(2)} m in ${m.simSeconds.toFixed(2)} s simulated`);
        if (query) assert.equal(m.backend, 'CPU');
        assert.deepEqual(errors, []);
        await page.close();
      }
    } finally {
      await browser.close();
    }
  });
});
