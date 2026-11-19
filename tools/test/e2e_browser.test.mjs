// Browser end-to-end test of the demo (Playwright, headless Chromium):
//  1. demo/ on its own: loads without console errors; Space on the focused
//     Start button starts it, with the chase camera; real key presses walk
//     and turn; checks the HUD and its first-run hint, the kbd highlights,
//     pause (the clock stops and nothing is drawn), speed levels, camera
//     modes, help, Space on a focused button, reset, clearing the keys on
//     blur and keycap labels for an AZERTY keyboard; measures frame rate and RTF.
//  2. The project page: "Start interactive demo" inserts the iframe, the demo
//     becomes ready and starts with the one-line legend, key presses reach it,
//     a mouse drag on the walker inside the frame pulls it, a plain mouse wheel
//     scrolls the page while Ctrl + wheel zooms, and the demo pauses (and stops
//     drawing) while scrolled out of view.
//  3. Touch (a phone-sized touch screen): the pad shows, its forward arrow
//     starts the demo and walks while held, and lets go on release; a
//     one-finger drag on the body pulls it, one on the background orbits; a
//     thumb holding ▲ while another finger drags the body walks and pulls;
//     the sensor-chart button shows the compact card across the top and
//     frames the walker below it (no part of it under the card).
//  4. Layout: the HUD panels, the view toggles (#viewbar), the X-ray legend,
//     the sensor card and the touch pad never overlap and the longest
//     readouts fit, at 21 x 15 window sizes (embedded, also the project page's
//     iframe sizes), standalone and embedded, with and without touch, in each
//     help mode, with and without the X-ray legend and the sensor card (which
//     the toast keeps clear of, and the help fits beside); where not even the
//     compact card fits, it is hidden, and only there; the walker's zone on
//     screen (framed below or beside the right column where that would cover
//     it) is clear of the card and the legend, and a framed walker of the pad.
//  4b. X-ray view: X and #btn-xray toggle it (aria-pressed, a toast, the
//     legend, muscles hidden, arrows from the socket anchor); it stays on
//     through a reset, the camera modes and a pause; a toggle while paused
//     draws once; ?xray=1 starts in it; the help gives way to the legend on a
//     short window and comes back.
//  4c. Sensor charts: G and #btn-sensors toggle them (aria-pressed, a toast,
//     the card with its six rows, the markers); the charts are drawn (inked
//     canvas), keep in step with the pose (their right edge at the play time,
//     once paused) and follow the walk (heel or toe contact in stance); the
//     markers sit on their geoms as drawn (projectToClient(geomPoint)); a
//     paused card does not redraw, a toggle while paused draws once; a reset
//     clears the history; with the X-ray view both show, the card under the
//     legend; ?sensors=1 starts with them; the help gives way to the card.
//  5. Pulling: the cursor shows 'grab' over the body; a real mouse drag on the
//     torso (down, a move over 0.5 s, a hold, up) pulls the walker toward the
//     drag, against a standing baseline, with the camera held still meanwhile;
//     state().perturb is active on both sides during the drag and released
//     after it; target updates go out at most once per animation frame; a drag
//     on the background orbits and pulls nothing; a drag while paused pulls
//     once resumed; the worker's 'perturb-end' ends the drag it names only.
//     With E2E_GPU=1 it runs on the hardware GPU.
//  6. With E2E_GPU=1, demo/ again on the hardware GPU (ANGLE on Vulkan for
//     WebGL, WebGPU on the GPU adapter): reports backends, frame rate and RTF,
//     and the same with the X-ray view, the sensor charts and both on (none
//     slower than off, within noise; the charts' 95th-percentile draw under
//     2 ms).
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

// A pixel (client coordinates in `target`'s document) where a press grabs the
// walker: the nearest to the ribcage's origin on screen (geom 21) that the
// view's pick() hits without slop. {x, y, hit} or null.
function torsoPixel(target) {
  return target.evaluate(() => {
    const d = window.__duet;
    const c = d.project(d.geomPoint(21));
    if (!c) return null;
    for (let r = 0; r <= 48; r += 3) {
      const n = Math.max(1, 2 * r);
      for (let k = 0; k < n; k++) {
        const x = c[0] + r * Math.cos((2 * Math.PI * k) / n), y = c[1] + r * Math.sin((2 * Math.PI * k) / n);
        const hit = d.pick(x, y, 0);
        if (hit && document.elementFromPoint(x, y) === document.querySelector('#viewport canvas')) return { x, y, hit };
      }
    }
    return null;
  });
}

// Whether a press at (x, y) lands on the canvas and grabs nothing there (with
// the slop a mouse press gets).
function onBackground(target, [x, y]) {
  return target.evaluate(([px, py]) => document.elementFromPoint(px, py) === document.querySelector('#viewport canvas')
    && window.__duet.pick(px, py) === null, [x, y]);
}

const wrapDeg = (a) => ((a + 540) % 360) - 180;

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

      // W+A for 2 s of simulated time, from when the reference turns left: the
      // switch waits for the running blend (and, on a loaded machine, for the
      // worker), 0.4-0.9 s simulated after the key press in a browser at RTF
      // 0.25-0.3, while the turn checked is the walk's.
      await page.keyboard.down('KeyA');
      await page.waitForFunction(() => window.__duet.state().hud.command === 'Forward, left', null, { timeout: 10000 });
      const arc = await page.waitForFunction(() => {
        const st = window.__duet.state();
        return /^fwd .*, \+/.test(st.clip || '') ? st : false;
      }, null, { timeout: 20000, polling: 50 });
      const s1a = await arc.jsonValue();
      const s2 = await waitSim(page, s1a.simTime, 2);
      assert.equal(s2.hud.command, 'Forward, left');
      const turn = moved(s1a.pose, s2.pose);
      t.diagnostic(`W+A: the left arc ${(s1a.simTime - s1.simTime).toFixed(2)} s simulated after the key; ${turn.progress.toFixed(2)} m forward, turned ${turn.turn.toFixed(2)} rad in ${(s2.simTime - s1a.simTime).toFixed(2)} s simulated`);
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
      assert.deepEqual(modes.map((m) => m[1]), [
        'Camera: fixed angle (drag the background to orbit)',
        'Camera: free (drag the background to orbit, right-drag to pan)',
        'Camera: behind the walker',
      ]);

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
      // The status that first reports the second reset (sent with it): read
      // later, a loaded machine may have simulated on for a while.
      const second = await page.waitForFunction((n) => {
        const st = window.__duet.state();
        return st.resets > n + 1 ? { simTime: st.simTime, resets: st.resets } : false;
      }, r0.resets, { timeout: 5000, polling: 'raf' });
      const r1 = await second.jsonValue();
      assert.ok(r1.simTime < 1, `back at the start (${r1.simTime.toFixed(2)} s after the reset)`);
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

      // A mouse drag on the walker inside the frame pulls it (once it stands).
      await sleep(1000);
      const frameBox = await handle.boundingBox();
      const tp = await torsoPixel(frame);
      assert.ok(tp, 'a pixel on the torso inside the frame');
      await page.mouse.move(frameBox.x + tp.x, frameBox.y + tp.y);
      await page.mouse.down();
      for (let i = 1; i <= 8; i++) {
        await page.mouse.move(frameBox.x + tp.x + 8 * i, frameBox.y + tp.y);
        await sleep(30);
      }
      const pulling = await state(frame);
      await page.mouse.up();
      assert.ok(pulling.perturb.active && pulling.perturb.started === 1, 'a drag on the walker pulls it, embedded too');
      await frame.waitForFunction(() => !window.__duet.state().perturb.active, null, { timeout: 3000 });

      // A plain wheel over the demo scrolls the page; Ctrl + wheel zooms.
      const box = await handle.boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      const y0 = await page.evaluate(() => scrollY);
      const d0 = (await state(frame)).cameraDistance;
      await page.mouse.wheel(0, 200);
      // Smooth scrolling takes a few frames; at 1-2 fps under load, seconds.
      await page.waitForFunction((y) => scrollY > y, y0, { timeout: 10000 }).catch(() => {});
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
      // The batch the worker was running when the pause came still finishes,
      // and its steps reach the page with the next status (four a second):
      // count from after that.
      await sleep(600);
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

      // One finger on the body pulls it; one on the background orbits (the
      // fixed-angle camera, from the camera button, keeps the orbit).
      const finger = (type, x, y) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: type === 'touchEnd' ? [] : [{ x, y, id: 2 }] });
      const swipe = async (x, y, dx) => {
        await finger('touchStart', x, y);
        for (let i = 1; i <= 8; i++) {
          await finger('touchMove', x + (dx * i) / 8, y);
          await sleep(40);
        }
        return state(page);
      };
      await sleep(1000); // standing again
      const tp = await torsoPixel(page);
      assert.ok(tp, 'a pixel on the torso');
      const during = await swipe(tp.x, tp.y, 60);
      assert.ok(during.perturb.active && during.perturb.started === 1, 'a finger on the body pulls it');
      await finger('touchEnd');
      await page.waitForFunction(() => !window.__duet.state().perturb.active, null, { timeout: 3000 });
      assert.equal((await state(page)).perturb.released, 1, 'lifting the finger lets go of the body');
      await touch('touchStart', '#btn-camera');
      await touch('touchEnd');
      await page.waitForFunction(() => window.__duet.state().camera === 'follow', null, { timeout: 3000 });
      const vp = page.viewportSize();
      const bgTouch = [Math.round(vp.width * 0.15), Math.round(vp.height * 0.4)];
      assert.ok(await onBackground(page, bgTouch), 'the background test point is clear');
      const a0 = (await state(page)).cameraAzimuth;
      await swipe(bgTouch[0], bgTouch[1], 120);
      await finger('touchEnd');
      await sleep(800);
      const a1 = await state(page);
      t.diagnostic(`touch: a swipe on the background orbited ${wrapDeg(a1.cameraAzimuth - a0).toFixed(1)} deg`);
      assert.ok(Math.abs(wrapDeg(a1.cameraAzimuth - a0)) > 15, 'a finger on the background orbits');
      assert.equal(a1.perturb.started, 1, 'and pulls nothing');

      // A thumb on ▲ and a finger on the body: it walks and is pulled. The pad
      // finger is the primary pointer, so the body finger is not; it grabs all
      // the same. (A touchStart or touchMove lists every finger down; the
      // closing touchEnd lists none and lifts them all.)
      const pad = await page.locator('.pad-up').boundingBox();
      const thumb = { x: pad.x + pad.width / 2, y: pad.y + pad.height / 2, id: 3 };
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [thumb] });
      await sleep(600);
      const body = await torsoPixel(page);
      assert.ok(body, 'a pixel on the torso while walking');
      const pullFinger = (dx) => ({ x: body.x + dx, y: body.y, id: 4 });
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [thumb, pullFinger(0)] });
      for (let i = 1; i <= 6; i++) {
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [thumb, pullFinger(-8 * i)] });
        await sleep(40);
      }
      const both = await state(page);
      assert.deepEqual(both.keys, ['KeyW'], 'the thumb keeps walking');
      assert.ok(both.perturb.active && both.perturb.started === 2, 'the other finger pulls the body');
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await page.waitForFunction(() => { const s = window.__duet.state(); return !s.perturb.active && s.keys.length === 0; }, null, { timeout: 3000 });

      // The sensor charts on a phone: the compact card across the top and the
      // walker framed below it (smaller, above the pad), so no part of it (its
      // geoms as drawn, over a second of walking) is under the card. Back to
      // the chase camera first (follow -> free -> chase).
      for (let i = 0; i < 2; i++) {
        await touch('touchStart', '#btn-camera');
        await touch('touchEnd');
      }
      await page.waitForFunction(() => window.__duet.state().camera === 'chase', null, { timeout: 3000 });
      await touch('touchStart', '#btn-sensors');
      await touch('touchEnd');
      await page.waitForFunction(() => window.__duet.state().sensors, null, { timeout: 5000 });
      await touch('touchStart', '.pad-up');
      await sleep(400);
      const phone = await page.evaluate(async () => {
        const demo = await (await fetch('/demo/assets/stage2/demo.json', { cache: 'no-cache' })).json();
        const d = window.__duet, s = d.state();
        const card = document.getElementById('sensor-card').getBoundingClientRect().toJSON();
        const pad = document.getElementById('touch-pad').getBoundingClientRect().toJSON();
        let under = 0, samples = 0, top = Infinity, bottom = -Infinity;
        const t0 = performance.now();
        while (performance.now() - t0 < 1500) {
          for (const id of demo.visual.geom_ids) {
            const p = d.project(d.geomPoint(id));
            if (!p) continue;
            samples += 1;
            top = Math.min(top, p[1]);
            bottom = Math.max(bottom, p[1]);
            if (p[0] > card.left && p[0] < card.right && p[1] > card.top && p[1] < card.bottom) under += 1;
          }
          await new Promise((r) => setTimeout(r, 100));
        }
        return { mode: s.sensorCard, framing: s.framing, card, pad, under, samples, top, bottom, markers: d.sensors().markers.map((m) => m.client) };
      });
      await page.screenshot({ path: path.join(OUT, 'e2e_touch_sensors.png') });
      await touch('touchEnd');
      t.diagnostic(`touch, sensor charts: card ${phone.mode} ${[phone.card.left, phone.card.top, phone.card.right, phone.card.bottom].map(Math.round)}, ` +
        `walker framed at ${phone.framing && phone.framing.scale.toFixed(2)}, its geoms from y ${phone.top.toFixed(0)} to ${phone.bottom.toFixed(0)} (${phone.samples} samples), pad top ${phone.pad.top.toFixed(0)}`);
      assert.equal(phone.mode, 'compact', 'a phone gets the compact card');
      assert.ok(phone.card.left <= 16 && phone.card.right >= 374, 'across the width');
      assert.ok(phone.framing && phone.framing.scale < 1, 'the walker is framed below the card');
      assert.equal(phone.under, 0, `no part of the walker under the card (${phone.under} of ${phone.samples} geom points)`);
      assert.ok(phone.top > phone.card.bottom, 'the walker below the card');
      assert.ok(phone.markers.every((c) => c && c[1] > phone.card.bottom && c[1] < phone.pad.top), 'the sensor markers between the card and the pad');
      await touch('touchStart', '#btn-sensors');
      await touch('touchEnd');
      await page.waitForFunction(() => !window.__duet.state().sensors, null, { timeout: 5000 });
      assert.equal((await state(page)).framing, null, 'the walker is framed as before once the card is off');
      assert.deepEqual(errors, []);
    } finally {
      await browser.close();
    }
  });

  test('layout: HUD panels, the view toggles, the X-ray legend, the sensor card and the touch pad never overlap, readouts fit (standalone and embedded)', { timeout: 1500000 }, async (t) => {
    const browser = await chromium.launch({ args: SOFTWARE });
    try {
      // [query, touch, help mode asked for, X-ray, sensor card]: the help as
      // each start shows it, and full; then all again with the X-ray legend,
      // the sensor card and both, beside which the help shows compact
      // (keyboard) or not at all (touch) where it would overlap
      // (__duet.help(mode) asks for a mode and lays the panels out, as a
      // resize does).
      const starts = [['', false, 'full'], ['?embed=1', false, 'compact'], ['?embed=1', false, 'full'], ['', true, 'full'], ['?embed=1', true, 'full']];
      const cases = [[false, false], [true, false], [false, true], [true, true]].flatMap(([xray, card]) => starts.map((c) => [...c, xray, card]));
      // The project page's iframe sizes (embedded): 16:9 up to 1096 px wide, 4:3 below a 760 px page.
      const LANDING = [[1096, 616], [960, 540], [736, 414], [699, 524], [382, 286], [358, 268], [328, 246]];
      const TOAST = "This browser's WebGPU is too slow for per-step inference, so the networks run on the CPU in slow motion. A browser with fast WebGPU (recent Chrome, Edge or Safari) runs it in real time.";
      const cardModes = { full: 0, compact: 0, none: 0 };
      for (const [query, touch, mode, xray, card] of cases) {
        const { page } = await openPage(browser);
        // No simulation needed: the assets fail to load and the error overlay is hidden below.
        await page.route('**/assets/**', (route) => route.fulfill({ status: 404, body: '' }));
        await page.goto(`${BASE}/demo/${query}`);
        await page.waitForFunction(() => window.__duet && window.__duet.state().error, null, { timeout: 30000 });
        await page.evaluate(([touchMode, xrayOn, cardOn, toastText]) => {
          document.getElementById('error').hidden = true;
          document.getElementById('loader').hidden = true;
          document.documentElement.classList.toggle('touch', touchMode);
          document.documentElement.dataset.phase = 'running';
          document.getElementById('hud').inert = false;
          const longest = {
            'hud-command': 'Backward, curving right', 'hud-speed': '-0.50 m/s, -0.25 rad/s',
            'hud-clip': 'fwd 1.00, +0.50 rad/s', 'hud-time': '999.9 s', 'hud-rtf': '0.58×', 'hud-backend': 'WebGPU',
            'xray-force': '2263', 'xray-moment': '130',
          };
          for (const [id, text] of Object.entries(longest)) document.getElementById(id).textContent = text;
          document.getElementById('hud-note').hidden = false;
          document.querySelector('#hud-note dd').textContent = "This browser's WebGPU is too slow for per-step inference, so the networks run on the CPU in slow motion. A browser with fast WebGPU (recent Chrome, Edge or Safari) runs it in real time.";
          // The longest toast, on screen.
          const toast = document.getElementById('toast');
          toast.textContent = toastText;
          toast.classList.add('show');
          if (xrayOn) {
            document.documentElement.classList.add('xray');
            document.getElementById('xray-legend').hidden = false;
          }
          // The sensor card: app.js places it (and shows or hides it) at every
          // layout while html.sensors is set; empty plots, as before the
          // first frame.
          if (cardOn) document.documentElement.classList.add('sensors');
        }, [touch, xray, card, TOAST]);
        const problems = [];
        const fitted = new Map(); // help mode shown -> window sizes
        const placed = new Map(); // card mode -> window sizes
        const framed = { n: 0, minScale: 1, covered: [] }; // window sizes where the walker was framed clear of the panels, or could not be
        const sizes = [];
        for (let w = 320; w <= 1920; w += 80) for (let h = 360; h <= 1200; h += 60) sizes.push([w, h]);
        if (query) sizes.push(...LANDING);
        for (const [w, h] of sizes) {
          await page.setViewportSize({ width: w, height: h });
          const shown = await page.evaluate((m) => window.__duet.help(m), mode);
          fitted.set(shown, (fitted.get(shown) || 0) + 1);
          const [cardMode, framing, covered] = await page.evaluate(() => [window.__duet.state().sensorCard, window.__duet.state().framing, window.__duet.state().walkerCovered]);
          if (framing) {
            framed.n += 1;
            framed.minScale = Math.min(framed.minScale, framing.scale);
          }
          if (covered) framed.covered.push(`${w}x${h}`);
          if (card) {
            placed.set(cardMode, (placed.get(cardMode) || 0) + 1);
            cardModes[cardMode] += 1;
            // A laptop window or larger always has room for the full card.
            if (!touch && w >= 1100 && h >= 720 && cardMode !== 'full') problems.push(`${w}x${h}${query}: card ${cardMode}, not full`);
          } else if (cardMode !== 'off') {
            problems.push(`${w}x${h}${query}: card ${cardMode} without the sensor charts`);
          }
          const found = await page.evaluate(([cardOn, cardMode]) => {
            const boxes = {};
            const panels = [['back', '#back-link'], ['top', '.topbar'], ['stats', '.stats'], ['help', '#help'], ['pad', '#touch-pad'],
              ['viewbar', '#viewbar'], ['legend', '#xray-legend'], ['card', '#sensor-card'], ['toast', '#toast']];
            for (const [name, selector] of panels) {
              const node = document.querySelector(selector);
              const style = node && getComputedStyle(node);
              if (!node || style.display === 'none' || style.visibility === 'hidden') continue;
              const b = node.getBoundingClientRect();
              if (b.width > 0) boxes[name] = b;
            }
            // The toast passes over the other panels for a moment; it must
            // keep clear of the view toggles, the X-ray legend and the card.
            const mine = new Set(['viewbar', 'legend', 'card']);
            const out = [];
            const names = Object.keys(boxes);
            for (let i = 0; i < names.length; i++) {
              for (let j = i + 1; j < names.length; j++) {
                const [p, q] = [names[i], names[j]];
                if ((p === 'toast' || q === 'toast') && !mine.has(p) && !mine.has(q)) continue;
                const a = boxes[p], b = boxes[q];
                if (a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5) out.push(`${p} overlaps ${q}`);
              }
            }
            for (const [name, b] of Object.entries(boxes)) {
              if (name === 'toast') continue;
              if (b.left < -0.5 || b.right > innerWidth + 0.5 || b.top < -0.5 || b.bottom > innerHeight + 0.5) out.push(`${name} off screen`);
            }
            for (const dd of document.querySelectorAll('.stats dd, #xray-legend dd')) {
              if (dd.offsetParent && dd.scrollWidth > dd.clientWidth + 1) out.push(`${dd.id || dd.textContent} cut off`);
            }
            const legend = document.getElementById('xray-legend');
            if (legend.offsetParent && (legend.scrollWidth > legend.clientWidth + 1 || legend.scrollHeight > legend.clientHeight + 1)) out.push('legend content cut off');
            // The card: shown exactly when placed, its content inside it, and
            // the full card where the compact one is not needed.
            const sc = document.getElementById('sensor-card');
            const scShown = !!boxes.card;
            if (cardOn && scShown !== (cardMode === 'full' || cardMode === 'compact')) out.push(`card shown ${scShown} in mode ${cardMode}`);
            if (scShown && (sc.scrollWidth > sc.clientWidth + 1 || sc.scrollHeight > sc.clientHeight + 1)) out.push('card content cut off');
            if (scShown && sc.dataset.mode !== cardMode) out.push(`card data-mode ${sc.dataset.mode}, state ${cardMode}`);
            // The walker (its zone on screen, framed where the right column
            // would cover it) stays clear of the card and of the legend at
            // the top, and a framed walker of the touch pad and a legend at
            // the bottom.
            const st = window.__duet.state(), z = st.walker;
            const cross = (a, b) => a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;
            if (!z) out.push('no walker zone');
            else {
              // Only the X-ray legend may cover it, and only where no framing
              // keeps 0.4 of its size (the smallest embeds; counted below).
              if (boxes.card && cross(z, boxes.card)) out.push('the card covers the walker');
              if (boxes.legend && cross(z, boxes.legend) && !st.walkerCovered) out.push(`the legend covers the walker${st.framing ? ' (framed)' : ''}`);
              if (st.walkerCovered && !(boxes.legend && cross(z, boxes.legend))) out.push('walkerCovered without a panel over it');
              if (st.framing && boxes.pad && cross(z, boxes.pad)) out.push('the framed walker is under the pad');
              if (st.framing && (z.left < -0.5 || z.right > innerWidth + 0.5 || z.top < -0.5 || z.bottom > innerHeight + 0.5)) out.push('the framed walker leaves the window');
            }
            return out;
          }, [card, cardMode]);
          if (found.length) problems.push(`${w}x${h}${query}${touch ? ' touch' : ''} ${mode}${xray ? ' x-ray' : ''}${card ? ' card' : ''} (help ${shown}, card ${cardMode}): ${found.join(', ')}`);
        }
        t.diagnostic(`demo/${query}${touch ? ' (touch)' : ''}, help ${mode}${xray ? ', X-ray' : ''}${card ? ', sensor card' : ''}: ${sizes.length} window sizes checked; ` +
          `help shown ${JSON.stringify(Object.fromEntries(fitted))}${card ? `; card ${JSON.stringify(Object.fromEntries(placed))}` : ''}` +
          `; walker framed at ${framed.n} sizes${framed.n ? ` (scale down to ${framed.minScale.toFixed(2)})` : ''}` +
          `${framed.covered.length ? `, under the legend with no room to move at ${framed.covered.join(' ')}` : ''}`);
        assert.deepEqual(problems, []);
        await page.close();
      }
      t.diagnostic(`sensor card over all cases: ${JSON.stringify(cardModes)}`);
      // The card shows on every desktop and laptop window, and the compact one on phones.
      assert.ok(cardModes.full > 0 && cardModes.compact > 0, 'both card modes are used');
    } finally {
      await browser.close();
    }
  });

  test('X-ray view: X and the button, legend and arrows, through reset, camera and pause; ?xray=1', { timeout: 300000 }, async (t) => {
    const gpu = process.env.E2E_GPU === '1';
    const browser = await chromium.launch({ args: gpu ? HARDWARE : SOFTWARE });
    try {
      const { page, errors } = await openPage(browser);
      await page.goto(`${BASE}/demo/`);
      await waitReady(page);
      const xrayDom = () => page.evaluate(() => ({
        state: window.__duet.state(), x: window.__duet.xray(), render: window.__duet.stats().render,
        pressed: document.getElementById('btn-xray').getAttribute('aria-pressed'),
        legendHidden: document.getElementById('xray-legend').hidden,
        toast: document.getElementById('toast').textContent,
        canvas: document.querySelector('#viewport canvas').getBoundingClientRect().toJSON(),
      }));
      const pre = await xrayDom();
      assert.ok(!pre.state.xray && pre.pressed === 'false' && pre.legendHidden && pre.render.muscles, 'X-ray starts off');
      await page.click('#start-btn');
      await page.waitForFunction(() => window.__duet.state().running && !window.__duet.state().paused, null, { timeout: 5000 });
      await sleep(800);
      const base = (await xrayDom()).render;

      // The arrows as drawn: from the socket anchor (on the canvas), each
      // min(scale |v|, cap) long along v, or hidden below its minimum.
      const checkArrows = (d, label) => {
        const { x } = d;
        assert.ok(x.on && x.anchor && x.F && x.T, `${label}: the frames carry the wrench`);
        const [cx, cy] = x.anchorClient;
        assert.ok(cx >= d.canvas.left && cx <= d.canvas.right && cy >= d.canvas.top && cy <= d.canvas.bottom, `${label}: the anchor is on screen`);
        for (const [key, v, cap, min] of [['force', x.F, 0.8, 3], ['moment', x.T, 0.6, 0.3]]) {
          const a = x[key], mag = Math.hypot(...v);
          assert.equal(a.shown, mag >= min, `${label}: ${key} shown`);
          if (!a.shown) continue;
          assert.ok(Math.abs(a.length - Math.min(x.scale[key] * mag, cap)) < 1e-6, `${label}: ${key} length ${a.length} for ${mag.toFixed(1)}`);
          assert.ok(Math.hypot(...a.direction.map((c, k) => c - v[k] / mag)) < 1e-6, `${label}: ${key} direction`);
          assert.ok(Math.hypot(...a.origin.map((c, k) => c - x.anchor[k])) < 1e-9, `${label}: ${key} starts at the anchor`);
        }
      };

      // X: on, with its toast, the legend, no muscles, fewer draw calls.
      await page.keyboard.press('KeyX');
      await sleep(400);
      const on = await xrayDom();
      assert.ok(on.state.xray && on.pressed === 'true' && !on.legendHidden && on.x.legend.shown, 'X turns the X-ray view on');
      assert.equal(on.toast, 'X-ray view on');
      assert.ok(!on.render.muscles && on.render.xray, 'the muscles are hidden');
      assert.ok(on.render.calls < base.calls, `fewer draw calls (${on.render.calls}, without X-ray ${base.calls})`);
      checkArrows(on, 'running');
      await page.screenshot({ path: path.join(OUT, 'e2e_xray.png') });

      // Through a reset and the three camera modes: still on, arrows drawn.
      const r0 = (await state(page)).resets;
      await page.keyboard.press('KeyR');
      await page.waitForFunction((n) => window.__duet.state().resets > n, r0, { timeout: 5000 });
      await sleep(600);
      const afterReset = await xrayDom();
      assert.ok(afterReset.state.xray && !afterReset.render.muscles, 'still on after a reset');
      checkArrows(afterReset, 'after a reset');
      for (let i = 0; i < 3; i++) await page.keyboard.press('KeyC');
      await sleep(300);
      const afterCamera = await xrayDom();
      assert.ok(afterCamera.state.xray && afterCamera.state.camera === 'chase', 'still on after the camera modes');

      // Paused: the legend gives the wrench as drawn; each toggle draws once,
      // with the X key or the button; X-ray off brings the muscles back posed.
      await page.keyboard.press('Space');
      await page.waitForFunction(() => window.__duet.state().paused, null, { timeout: 5000 });
      await sleep(700);
      const p0 = await xrayDom();
      assert.ok(p0.state.xray, 'still on while paused');
      assert.equal(p0.x.legend.force, String(Math.round(Math.hypot(...p0.x.F))), 'the legend gives |F| as drawn');
      assert.equal(p0.x.legend.moment, String(Math.round(Math.hypot(...p0.x.T))), 'the legend gives |T| as drawn');
      const settled = await rendersOver(page, 500);
      assert.equal(settled.renders, 0, 'nothing drawn while paused and still');
      const drawsAfter = async (action) => {
        const before = (await state(page)).frames.renders;
        await action();
        await sleep(700);
        return (await state(page)).frames.renders - before;
      };
      const offDraws = await drawsAfter(() => page.keyboard.press('KeyX'));
      const off = await xrayDom();
      assert.equal(offDraws, 1, `a paused toggle draws once (${offDraws})`);
      assert.ok(!off.state.xray && off.pressed === 'false' && off.legendHidden, 'X turns it off while paused');
      assert.ok(off.render.muscles && off.render.segments > 0, 'the muscles are back, posed');
      assert.equal(off.render.calls, base.calls, 'as many draw calls as before X-ray');
      const onDraws = await drawsAfter(() => page.click('#btn-xray'));
      const back = await xrayDom();
      assert.equal(onDraws, 1, `a paused toggle with the button draws once (${onDraws})`);
      assert.ok(back.state.xray && back.pressed === 'true', 'the button turns it on');
      assert.ok(await page.evaluate(() => document.activeElement === document.body), 'the clicked button gives the keyboard back');
      checkArrows(back, 'paused');
      // R while paused: the reset pose stands before its first control step,
      // which is no measurement (frame.initial): no arrows, the legend a dash.
      const pr0 = (await state(page)).resets;
      await page.keyboard.press('KeyR');
      await page.waitForFunction((n) => window.__duet.state().resets > n, pr0, { timeout: 5000 });
      await sleep(500);
      const pr = await xrayDom();
      assert.ok(pr.state.paused && pr.x.F === null && !(pr.x.force && pr.x.force.shown) && !(pr.x.moment && pr.x.moment.shown), 'a reset while paused draws no arrows');
      assert.deepEqual([pr.x.legend.force, pr.x.legend.moment], ['–', '–'], 'and the legend shows dashes');
      await page.keyboard.press('Space');
      await page.waitForFunction(() => !window.__duet.state().paused, null, { timeout: 5000 });
      await page.waitForFunction(() => window.__duet.xray().F !== null, null, { timeout: SIM_TIMEOUT_MS, polling: 50 });

      // The legend's swatches are the arrows' colours as the renderer draws them.
      const swatches = await page.evaluate(() => ({
        force: getComputedStyle(document.querySelector('.xl-swatch.xl-force')).color,
        moment: getComputedStyle(document.querySelector('.xl-swatch.xl-moment')).color,
      }));
      const css = (hex) => `rgb(${[1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(', ')})`;
      const drawn = await page.evaluate(() => window.__duet.xray().colors);
      assert.deepEqual(swatches, { force: css(drawn.force), moment: css(drawn.moment) }, 'the legend\'s swatches are the arrows\' colours');

      // A short window: the full help gives way to the legend (compact), and
      // comes back when there is room again.
      assert.equal((await state(page)).help, 'full');
      await page.setViewportSize({ width: 1280, height: 600 });
      await page.waitForFunction(() => window.__duet.state().help === 'compact', null, { timeout: 3000 });
      const short = await state(page);
      assert.equal(short.helpWanted, 'full', 'the help asked for stays full');
      await page.setViewportSize({ width: 1280, height: 900 });
      await page.waitForFunction(() => window.__duet.state().help === 'full', null, { timeout: 3000 });
      await page.setViewportSize({ width: 1280, height: 720 });
      assert.deepEqual(errors, []);
      await page.close();

      // ?xray=1: on from the start, behind the loader too.
      const second = await openPage(browser);
      await second.page.goto(`${BASE}/demo/?xray=1`);
      await waitReady(second.page);
      await second.page.waitForFunction(() => window.__duet.state().frames.renders >= 1 && window.__duet.xray().anchor, null, { timeout: 10000 });
      const ready = await second.page.evaluate(() => ({
        xray: window.__duet.state().xray, render: window.__duet.stats().render, x: window.__duet.xray(),
        pressed: document.getElementById('btn-xray').getAttribute('aria-pressed'), legend: !document.getElementById('xray-legend').hidden,
      }));
      assert.ok(ready.xray && ready.pressed === 'true' && ready.legend && !ready.render.muscles, '?xray=1 starts in the X-ray view');
      // Behind the loader the walker stands in the reset pose, before its first
      // control step: no measurement, so no arrows and dashes in the legend.
      assert.ok(ready.x.F === null && !(ready.x.force && ready.x.force.shown) && !(ready.x.moment && ready.x.moment.shown), 'no arrows for the reset pose behind the loader');
      assert.deepEqual([ready.x.legend.force, ready.x.legend.moment], ['–', '–'], 'the legend shows dashes behind the loader');
      await second.page.click('#start-btn');
      await sleep(800);
      const started = await second.page.evaluate(() => ({ x: window.__duet.xray(), canvas: document.querySelector('#viewport canvas').getBoundingClientRect().toJSON() }));
      checkArrows(started, '?xray=1');
      // Its first X (off) builds no shader program: they were built behind the
      // loader (the muscles' main and shadow programs, the socket's base one).
      const programs0 = (await second.page.evaluate(() => window.__duet.stats().render)).programs;
      await second.page.keyboard.press('KeyX');
      await sleep(600);
      const offFirst = await second.page.evaluate(() => ({ s: window.__duet.state(), r: window.__duet.stats().render }));
      assert.ok(!offFirst.s.xray && offFirst.r.muscles, '?xray=1: X turns it off');
      assert.equal(offFirst.r.programs, programs0, `?xray=1: the first X off builds no program (${programs0} -> ${offFirst.r.programs})`);
      t.diagnostic(`X-ray: ${on.render.calls} draw calls and ${on.render.triangles} triangles against ${base.calls} and ${base.triangles}; ` +
        `standing: |F| ${Math.hypot(...started.x.F).toFixed(0)} N (${started.x.forceLength.toFixed(3)} m), |T| ${Math.hypot(...started.x.T).toFixed(1)} N m`);
      assert.deepEqual(second.errors, []);
    } finally {
      await browser.close();
    }
  });

  test('sensor charts: G and the button, the card and the markers, through pause, reset and the X-ray view; ?sensors=1', { timeout: 360000 }, async (t) => {
    const gpu = process.env.E2E_GPU === '1';
    const browser = await chromium.launch({ args: gpu ? HARDWARE : SOFTWARE });
    try {
      const { page, errors } = await openPage(browser);
      await page.goto(`${BASE}/demo/`);
      await waitReady(page);
      const dom = () => page.evaluate(() => ({
        state: window.__duet.state(), x: window.__duet.sensors(), render: window.__duet.stats().render,
        pressed: document.getElementById('btn-sensors').getAttribute('aria-pressed'),
        hidden: document.getElementById('sensor-card').hidden,
        toast: document.getElementById('toast').textContent,
      }));
      // Pixels drawn on the live canvas (traces and values) in each row's
      // plot, and on the static one (titles, grid, keys).
      const inked = () => page.evaluate(() => {
        const count = (c, box = null) => {
          const [x, y, w, h] = box || [0, 0, c.width, c.height];
          const d = c.getContext('2d').getImageData(x, y, w, h).data;
          let n = 0;
          for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n += 1;
          return n;
        };
        const [live, still] = ['.sc-live', '.sc-static'].map((s) => document.querySelector(`#sensor-card canvas${s}`));
        const card = window.__duet.sensors().card, k = card.dpr;
        const rows = card.rows.map((r) => ({ key: r.key, inked: count(live, [r.plot.x, r.plot.y, r.plot.w, r.plot.h].map((v) => Math.round(v * k))), width: r.plot.w * k }));
        return { live: count(live), static: count(still), size: [live.width, live.height], rows };
      });
      // Each marker on screen against its geom point as drawn, in one go (one pose).
      const markerErrors = () => page.evaluate(() => {
        const d = window.__duet;
        return d.sensors().markers.map((m) => {
          const want = d.project(d.geomPoint(m.geom_id, m.local));
          return { key: m.key, shown: m.shown, px: m.client && want ? Math.hypot(m.client[0] - want[0], m.client[1] - want[1]) : Infinity };
        });
      });
      const draws = async () => (await page.evaluate(() => window.__duet.stats().sensors)).draws;

      const pre = await dom();
      assert.ok(!pre.state.sensors && pre.pressed === 'false' && pre.hidden && !pre.render.markers, 'the sensor charts start off');
      assert.equal(pre.state.sensorCard, 'off');
      await page.click('#start-btn');
      await page.waitForFunction(() => window.__duet.state().running && !window.__duet.state().paused, null, { timeout: 5000 });
      await sleep(800);
      const base = (await dom()).render;

      // G: on, with its toast, the full card's six rows, the markers.
      await page.keyboard.press('KeyG');
      await sleep(600);
      const on = await dom();
      assert.ok(on.state.sensors && on.pressed === 'true' && !on.hidden && on.x.card.shown, 'G shows the sensor charts');
      assert.equal(on.toast, 'Sensor charts on');
      assert.equal(on.state.sensorCard, 'full');
      assert.deepEqual(on.x.card.rows.map((r) => r.title), ['IMU gyro', 'IMU accel', 'Load cell force', 'Load cell moment', 'Foot contact', 'Encoders']);
      assert.deepEqual(on.x.card.rows.map((r) => r.lines.map((l) => l.name).join(' ')), ['x y z', 'x y z', 'x y z', 'x y z', 'heel toe', 'knee ankle']);
      assert.ok(on.render.markers && on.render.calls === base.calls + 1, `the markers: one more draw call (${on.render.calls}, before ${base.calls})`);
      assert.equal(on.x.markers.length, 6);
      assert.ok(on.x.markers.every((m) => m.shown), 'every marker is drawn');
      const ink = await inked();
      assert.ok(ink.static > 5000, `the titles, grid and keys are drawn (${ink.static} pixels)`);
      assert.ok(ink.live > 200, `the traces so far and the values are drawn (${ink.live} pixels)`);
      for (const m of await markerErrors()) assert.ok(m.px < 1.5, `marker ${m.key} ${m.px.toFixed(3)} px from its geom point`);
      // Every range label inside the canvas (right-aligned beside its plot).
      for (const r of on.x.card.rows) {
        assert.ok(r.ticks.length >= 2, `${r.key}: its range labels`);
        for (const tk of r.ticks) assert.ok(tk.right - tk.width >= 0, `${r.key}: the label ${tk.text} starts at ${(tk.right - tk.width).toFixed(1)} px`);
      }
      // The values as text for screen readers, in the button's description.
      const summary = await page.evaluate(() => ({ text: document.getElementById('sensor-summary').textContent, by: document.getElementById('btn-sensors').getAttribute('aria-describedby') }));
      assert.equal(summary.by, 'sensor-summary');
      assert.match(summary.text, /^IMU gyro, rad\/s: x .+ Encoders, deg: knee .+, ankle .+\.$/, `the summary: ${summary.text}`);

      // Walking: the charts follow (heel or toe loaded in stance), at most 30
      // draws a second, each well under 2 ms.
      await page.keyboard.down('KeyW');
      const w0 = await state(page);
      const seen = { contact: 0, values: new Set(), samples: 0 };
      const d0 = await draws();
      const tw = Date.now();
      while ((await state(page)).simTime < w0.simTime + 3 && Date.now() - tw < SIM_TIMEOUT_MS) {
        const x = (await dom()).x;
        const contact = x.card.rows.find((r) => r.key === 'contact');
        seen.contact = Math.max(seen.contact, ...contact.lines.map((l) => l.value || 0));
        seen.values.add(x.card.rows.map((r) => r.lines.map((l) => l.text).join()).join());
        seen.samples += 1;
        for (const m of await markerErrors()) assert.ok(m.px < 1.5, `walking: marker ${m.key} ${m.px.toFixed(3)} px from its geom point`);
        await sleep(100);
      }
      const walkWall = (Date.now() - tw) / 1000;
      const walkDraws = (await draws()) - d0;
      const walked = await dom();
      // 3 s of walking: every plot has its traces across it.
      const inkWalk = await inked();
      t.diagnostic(`card canvases ${inkWalk.size.join('x')}: ${inkWalk.live} live and ${inkWalk.static} static pixels inked; ` +
        `per plot ${inkWalk.rows.map((r) => `${r.key} ${r.inked}`).join(', ')}`);
      for (const r of inkWalk.rows) assert.ok(r.inked >= r.width, `${r.key}: its traces are drawn (${r.inked} pixels in a plot ${r.width} px wide)`);
      await page.screenshot({ path: path.join(OUT, 'e2e_sensors.png') });
      await page.keyboard.up('KeyW');
      const ms = walked.x.card.stats.ms;
      t.diagnostic(`walking: contact up to ${seen.contact.toFixed(0)} N, ${seen.values.size} different readings in ${seen.samples} samples; ` +
        `${walkDraws} card draws in ${walkWall.toFixed(1)} s (${(walkDraws / walkWall).toFixed(1)} per s), draw ${ms.mean.toFixed(2)} ms mean, ${ms.p95.toFixed(2)} ms p95; ` +
        `history ${walked.x.card.stats.samples} samples over ${walked.x.card.stats.span.toFixed(2)} s`);
      assert.ok(seen.contact > 50, `heel or toe contact in stance (${seen.contact.toFixed(0)} N)`);
      assert.ok(seen.values.size > 5, 'the values change while walking');
      assert.ok(walkDraws / walkWall <= 31, `at most 30 draws a second (${(walkDraws / walkWall).toFixed(1)})`);
      assert.ok(walked.x.card.stats.span >= 2.9, 'the history covers the 3 s drawn');
      // Every control step's sensor row arrived (frame.sensorRows carries those
      // of the steps whose frame the worker skipped): one sample per 10 ms,
      // no break in the lines.
      const hs = walked.x.card.stats;
      t.diagnostic(`stream: ${hs.rows} rows of skipped steps, ${hs.samples} samples over ${hs.span.toFixed(2)} s, ${hs.breaks} breaks; frames ${walked.state.frames.received}`);
      assert.ok(hs.streamed, 'the worker sends the rows of skipped steps');
      assert.equal(hs.breaks, 0, 'no break in the lines');
      assert.ok(Math.abs(hs.span - (hs.samples - 1) * 0.01) < 1e-6, `a sample for every step (${hs.samples} over ${hs.span.toFixed(3)} s)`);

      // Paused: one last drawing at the paused pose, then none; a toggle while
      // paused draws the view once (the markers), and the card once.
      await page.keyboard.press('Space');
      await page.waitForFunction(() => window.__duet.state().paused, null, { timeout: 5000 });
      await sleep(600);
      const p0 = await dom();
      assert.ok(Math.abs(p0.x.card.drawnAt - p0.x.playTime) < 1e-9, `the paused card ends at the pose on screen (${p0.x.card.drawnAt} vs ${p0.x.playTime})`);
      assert.ok(Math.abs(p0.x.card.at - p0.x.playTime) < 1e-9, 'and its values are the pose\'s');
      const heel = p0.x.card.rows.find((r) => r.key === 'contact').lines[0].value;
      assert.ok(Math.abs(heel - p0.x.drawn[16]) < 1e-3 * Math.max(1, Math.abs(heel)), `the heel value is the drawn frame's (${heel} vs ${p0.x.drawn[16]})`);
      const still = await draws();
      const idle = await rendersOver(page, 1000);
      assert.equal(await draws(), still, 'a paused card does not redraw');
      assert.equal(idle.renders, 0, 'nor does the view');
      const viewDraws = async (action) => {
        const before = (await state(page)).frames.renders;
        const c0 = await draws();
        await action();
        await sleep(700);
        return { view: (await state(page)).frames.renders - before, card: (await draws()) - c0 };
      };
      const offDraws = await viewDraws(() => page.keyboard.press('KeyG'));
      const off = await dom();
      assert.ok(!off.state.sensors && off.pressed === 'false' && off.hidden && !off.render.markers, 'G turns them off while paused');
      assert.equal(await page.evaluate(() => document.getElementById('sensor-summary').textContent), '', 'no summary while the card is off');
      assert.equal(off.render.calls, base.calls, 'as many draw calls as before');
      assert.deepEqual(offDraws, { view: 1, card: 0 }, `a paused toggle off draws the view once (${JSON.stringify(offDraws)})`);
      const onDraws = await viewDraws(() => page.click('#btn-sensors'));
      const back = await dom();
      assert.ok(back.state.sensors && back.pressed === 'true' && !back.hidden, 'the button turns them on');
      assert.ok(await page.evaluate(() => document.activeElement === document.body), 'the clicked button gives the keyboard back');
      assert.deepEqual(onDraws, { view: 1, card: 1 }, `a paused toggle on draws the view and the card once (${JSON.stringify(onDraws)})`);
      assert.ok(Math.abs(back.x.card.drawnAt - back.x.playTime) < 1e-9, 'at the pose on screen');
      await page.keyboard.press('Space');
      await page.waitForFunction(() => !window.__duet.state().paused, null, { timeout: 5000 });

      // A reset clears the history: the card starts over with the new epoch.
      await sleep(500);
      const r0 = await dom();
      await page.keyboard.press('KeyR');
      await page.waitForFunction((n) => window.__duet.state().resets > n, r0.state.resets, { timeout: 5000 });
      await page.waitForFunction((e) => { const s = window.__duet.stats().sensors; return s.epoch !== e && s.samples > 0; }, r0.x.card.stats.epoch, { timeout: 5000 });
      const r1 = await dom();
      t.diagnostic(`reset: history ${r0.x.card.stats.samples} samples (${r0.x.card.stats.span.toFixed(2)} s) -> ${r1.x.card.stats.samples} (${r1.x.card.stats.span.toFixed(2)} s), epoch ${r0.x.card.stats.epoch} -> ${r1.x.card.stats.epoch}`);
      assert.ok(r1.x.card.stats.span < 1 && r1.x.card.stats.samples < r0.x.card.stats.samples, 'the reset cleared the history');
      assert.ok(r1.state.sensors && r1.x.card.shown, 'the charts stay on through a reset');

      // With the X-ray view: both on, the card under the legend, the markers over the ghosts.
      await page.keyboard.press('KeyX');
      await sleep(600);
      const both = await page.evaluate(() => ({
        s: window.__duet.state(), x: window.__duet.sensors(), r: window.__duet.stats().render,
        legend: document.getElementById('xray-legend').getBoundingClientRect().toJSON(),
      }));
      assert.ok(both.s.xray && both.s.sensors && both.r.xray && both.r.markers, 'both views on');
      assert.ok(both.x.card.shown && both.x.card.box.top >= both.legend.bottom + 7.5, `the card under the legend (${both.x.card.box.top} vs ${both.legend.bottom})`);
      for (const m of await markerErrors()) assert.ok(m.px < 1.5, `X-ray: marker ${m.key} ${m.px.toFixed(3)} px from its geom point`);
      await page.screenshot({ path: path.join(OUT, 'e2e_sensors_xray.png') });
      // Three camera modes later, both still on.
      for (let i = 0; i < 3; i++) await page.keyboard.press('KeyC');
      await sleep(300);
      const cam = await state(page);
      assert.ok(cam.xray && cam.sensors && cam.camera === 'chase', 'both stay on through the camera modes');
      await page.keyboard.press('KeyX');

      // The full help (510 px tall) gives way to the card (compact) below
      // about 1070 px of height, and comes back above.
      await page.setViewportSize({ width: 1280, height: 600 });
      await page.waitForFunction(() => window.__duet.state().help === 'compact', null, { timeout: 3000 });
      assert.equal((await state(page)).helpWanted, 'full', 'the help asked for stays full');
      await page.setViewportSize({ width: 1280, height: 1100 });
      await page.waitForFunction(() => window.__duet.state().help === 'full', null, { timeout: 3000 });
      await page.setViewportSize({ width: 1280, height: 720 });
      assert.deepEqual(errors, []);
      await page.close();

      // ?sensors=1: on from the start, behind the loader too.
      const second = await openPage(browser);
      await second.page.goto(`${BASE}/demo/?sensors=1`);
      await waitReady(second.page);
      const ready = await second.page.evaluate(() => ({
        s: window.__duet.state(), r: window.__duet.stats().render,
        pressed: document.getElementById('btn-sensors').getAttribute('aria-pressed'), shown: !document.getElementById('sensor-card').hidden,
      }));
      assert.ok(ready.s.sensors && ready.pressed === 'true' && ready.shown && ready.r.markers, '?sensors=1 starts with the sensor charts');
      // Behind the loader: the reset pose before its first step is no
      // measurement (frame.initial): nothing in the history, the contact
      // glyphs open.
      await second.page.waitForFunction(() => window.__duet.state().frames.renders >= 1 && window.__duet.sensors().markers.every((m) => m.shown), null, { timeout: 10000 });
      const behind = await second.page.evaluate(() => ({ st: window.__duet.stats().sensors, m: window.__duet.sensors().markers }));
      assert.ok(behind.st.framesInitial >= 1 && behind.st.samples === 0, `the reset pose is not charted (${JSON.stringify(behind.st)})`);
      assert.ok(behind.m.filter((m) => m.channel >= 0).every((m) => m.fill === 0), 'the contact glyphs are open behind the loader');
      await second.page.click('#start-btn');
      // Half a second of simulated time, however long that takes here.
      await second.page.waitForFunction(() => { const s = window.__duet.stats().sensors; return s.draws > 0 && s.samples > 50; }, null, { timeout: SIM_TIMEOUT_MS, polling: 100 });
      const started = await second.page.evaluate(() => window.__duet.sensors());
      assert.ok(started.card.stats.draws > 0 && started.card.stats.samples > 50, '?sensors=1: the card draws once running');
      assert.deepEqual(second.errors, []);
    } finally {
      await browser.close();
    }
  });

  test('pull: drag a body part to pull the walker; the background still orbits', { timeout: 300000 }, async (t) => {
    const gpu = process.env.E2E_GPU === '1';
    const browser = await chromium.launch({ args: gpu ? HARDWARE : SOFTWARE });
    try {
      const { page, errors } = await openPage(browser);
      // Keep a handle on the worker, to hand the page a 'perturb-end' event below.
      await page.addInitScript(() => {
        const Native = window.Worker;
        window.Worker = class extends Native {
          constructor(...args) {
            super(...args);
            window.__testWorker = this;
          }
        };
      });
      await page.goto(`${BASE}/demo/`);
      await waitReady(page);
      await page.click('#start-btn');
      await page.waitForFunction(() => window.__duet.state().running && !window.__duet.state().paused, null, { timeout: 5000 });
      await sleep(600);

      // Hover: 'grab' over the body, the default cursor over the background.
      const bg = [140, 520];
      assert.ok(await onBackground(page, bg), 'the test point (140, 520) shows the background');
      await page.mouse.move(...bg);
      await sleep(200);
      assert.equal((await state(page)).cursor, '');
      const torso = await torsoPixel(page);
      assert.ok(torso, 'a pixel on the torso grabs it');
      await page.mouse.move(torso.x, torso.y);
      await sleep(200);
      assert.equal((await state(page)).cursor, 'grab', 'the cursor offers to grab the body');

      // Baseline: standing for 1 s of simulated time.
      const b0 = await state(page);
      const b1 = await waitSim(page, b0.simTime, 1);

      // The drag: down on the torso, 15 moves of (+10, -2) px over 0.5 s, a hold
      // for 1 s of simulated time, up.
      const d0 = await state(page);
      await page.mouse.down();
      const down = await state(page);
      assert.ok(down.perturb.active && down.perturb.started === 1, 'the press starts a drag');
      assert.equal(down.cursor, 'grabbing');
      for (let i = 1; i <= 15; i++) {
        await page.mouse.move(torso.x + 10 * i, torso.y - 2 * i);
        await sleep(33);
      }
      const moved = await state(page);
      await page.waitForFunction(() => { const w = window.__duet.state().perturb.worker; return w && w.active; }, null, { timeout: 5000 });
      const samples = [moved];
      const holdStart = Date.now();
      while (samples[samples.length - 1].simTime < moved.simTime + 1 && Date.now() - holdStart < SIM_TIMEOUT_MS) {
        await sleep(100);
        samples.push(await state(page));
      }
      await page.screenshot({ path: path.join(OUT, 'e2e_pull.png') });
      const held = samples[samples.length - 1];
      await page.mouse.up();
      await page.waitForFunction(() => { const p = window.__duet.state().perturb; return !p.active && !(p.worker && p.worker.active); }, null, { timeout: 5000 });
      const after = await state(page);
      assert.notEqual(after.cursor, 'grabbing', 'letting go drops the grabbing cursor');

      // The walker moved toward the drag (horizontally, from the grabbed point
      // to the target), well beyond its standing sway; samples after a fall
      // (which resets the pose) do not count.
      const target = moved.perturb.target, from = torso.hit.point;
      const len = Math.hypot(target[0] - from[0], target[1] - from[1]);
      const dir = [(target[0] - from[0]) / len, (target[1] - from[1]) / len];
      const along = (a, b) => (b.pose.x - a.pose.x) * dir[0] + (b.pose.y - a.pose.y) * dir[1];
      const sway = Math.abs(along(b0, b1));
      const pulled = Math.max(...samples.filter((s) => s.falls === d0.falls).map((s) => along(d0, s)));
      const forces = samples.map((s) => (s.perturb.worker ? s.perturb.worker.force : 0));
      const frames = after.frames.rafs - down.frames.rafs; // animation frames from the press to after the release
      t.diagnostic(`pull on ${JSON.stringify(down.perturb.geom)} (body ${down.perturb.body}): target ${len.toFixed(2)} m away horizontally; ` +
        `the walker moved ${pulled.toFixed(2)} m toward it in ${(held.simTime - d0.simTime).toFixed(2)} s simulated (standing sway ${sway.toFixed(3)} m per s); ` +
        `force up to ${Math.max(...forces).toFixed(0)} N; falls ${held.falls - d0.falls}; ${after.perturb.updates} target updates in ${frames} animation frames`);
      assert.ok(len > 0.2, `the drag put the target ${len.toFixed(2)} m away`);
      assert.ok(pulled > sway + 0.1, `pulled ${pulled.toFixed(2)} m toward the drag (standing: ${sway.toFixed(3)} m)`);
      assert.ok(Math.max(...forces) > 0, 'the worker reported the force');
      assert.ok(Math.abs(wrapDeg(held.cameraAzimuth - d0.cameraAzimuth)) < 2, 'the camera held still during the drag');
      assert.ok(after.perturb.updates >= 1 && after.perturb.updates <= frames, `at most one target update per animation frame (${after.perturb.updates} in ${frames})`);
      assert.equal(after.perturb.released, 1);

      // The background: a drag orbits (the fixed-angle camera keeps it) and
      // pulls nothing.
      await page.keyboard.press('KeyC');
      assert.equal((await state(page)).camera, 'follow');
      await sleep(300);
      assert.ok(await onBackground(page, bg), 'the background test point is still clear');
      const o0 = await state(page);
      await page.mouse.move(...bg);
      await page.mouse.down();
      for (let i = 1; i <= 15; i++) {
        await page.mouse.move(bg[0] + 12 * i, bg[1]);
        await sleep(16);
      }
      await page.mouse.up();
      await sleep(600);
      const o1 = await state(page);
      const orbit = wrapDeg(o1.cameraAzimuth - o0.cameraAzimuth);
      t.diagnostic(`background drag: the camera orbited ${orbit.toFixed(1)} deg`);
      assert.ok(Math.abs(orbit) > 20, `a drag on the background orbits (${orbit.toFixed(1)} deg)`);
      assert.equal(o1.perturb.started, o0.perturb.started, 'and pulls nothing');
      assert.ok(!o1.perturb.active);

      // Paused: the drag registers, the pull acts once resumed.
      await page.keyboard.press('Space');
      await sleep(300);
      const p = await torsoPixel(page);
      assert.ok(p, 'a pixel on the torso');
      await page.mouse.move(p.x, p.y);
      await page.mouse.down();
      for (let i = 1; i <= 8; i++) {
        await page.mouse.move(p.x + 12 * i, p.y);
        await sleep(33);
      }
      await page.waitForFunction(() => { const w = window.__duet.state().perturb.worker; return w && w.active; }, null, { timeout: 5000 });
      const q0 = await state(page);
      await sleep(400);
      const q1 = await state(page);
      assert.ok(q1.paused && q1.steps === q0.steps && q1.perturb.active, 'paused: the drag waits');
      await page.keyboard.press('Space');
      await page.waitForFunction(() => { const s = window.__duet.state(); const w = s.perturb.worker; return !s.paused && w && w.active && w.force > 0; }, null, { timeout: 10000 });
      await page.mouse.up();
      await page.waitForFunction(() => !window.__duet.state().perturb.active, null, { timeout: 5000 });

      // The worker's 'perturb-end' ends the drag it names (by id), not a newer one.
      await sleep(1000); // the walker has caught itself
      const r = await torsoPixel(page);
      assert.ok(r, 'a pixel on the torso');
      await page.mouse.move(r.x, r.y);
      await page.mouse.down();
      const e0 = await state(page);
      assert.ok(e0.perturb.active);
      const endEvent = (id) => page.evaluate((i) => window.__testWorker.onmessage({ data: { type: 'event', event: { type: 'perturb-end', reason: 'reset', id: i } } }), id);
      await endEvent(e0.perturb.id - 1);
      assert.ok((await state(page)).perturb.active, 'a perturb-end for an older drag is ignored');
      await endEvent(e0.perturb.id);
      const e1 = await state(page);
      assert.ok(!e1.perturb.active && e1.perturb.ended === e0.perturb.ended + 1, 'the perturb-end for this drag ends it');
      await page.mouse.up();
      assert.deepEqual(errors, []);
    } finally {
      await browser.close();
    }
  });

  test('hardware GPU (E2E_GPU=1): backends, frame rate, RTF', { skip: process.env.E2E_GPU !== '1' && 'set E2E_GPU=1', timeout: 480000 }, async (t) => {
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
        if (!query) {
          // The X-ray view, the sensor charts and both against the baseline,
          // alternating (the machine is shared): the X-ray view draws less,
          // the charts add a canvas drawing at 30 Hz and a draw call, so none
          // may be slower, within noise. X and G toggle them.
          const runs = { off: [m], xray: [], sensors: [], both: [] };
          const want = { off: [false, false], xray: [true, false], sensors: [false, true], both: [true, true] };
          for (const mode of ['xray', 'off', 'xray', 'sensors', 'off', 'sensors', 'both', 'off', 'both']) {
            const s = await state(page);
            if (s.xray !== want[mode][0]) await page.keyboard.press('KeyX');
            if (s.sensors !== want[mode][1]) await page.keyboard.press('KeyG');
            await sleep(500);
            const now = await state(page);
            assert.deepEqual([now.xray, now.sensors], want[mode], `${mode}: the views`);
            runs[mode].push(await measure(page, page, 4));
            if (mode === 'both' && runs.both.length === 1) await page.screenshot({ path: path.join(OUT, 'e2e_gpu_sensors_xray.png') });
          }
          const chart = (await page.evaluate(() => window.__duet.stats().sensors));
          // RTF as simulated seconds per wall second over each run (the
          // worker's own figure is one sample at its end); noise: the spread
          // between the runs of each mode, at least 0.05 RTF and 2 fps.
          const modes = Object.keys(runs);
          const mean = (list, key) => list.reduce((s, r) => s + r[key], 0) / list.length;
          const spread = (key) => Math.max(...modes.map((k) => Math.max(...runs[k].map((r) => r[key])) - Math.min(...runs[k].map((r) => r[key]))));
          const noise = { fps: Math.max(2, spread('fps')), rtf: Math.max(0.05, spread('simRate')) };
          const runText = (k) => runs[k].map((r) => `${r.fps.toFixed(1)} fps/${r.simRate.toFixed(2)}`).join(', ');
          results.push(`hardware, alternating runs (off ${runs.off.length}, others 2 each): ` +
            modes.map((k) => `${k} ${mean(runs[k], 'fps').toFixed(1)} fps, RTF ${mean(runs[k], 'simRate').toFixed(2)} (${runText(k)})`).join('; ') +
            `; noise ${noise.fps.toFixed(1)} fps, ${noise.rtf.toFixed(2)} RTF; chart draws ${chart.draws}, ${chart.ms.mean.toFixed(2)} ms mean, ${chart.ms.p95.toFixed(2)} ms p95, ${chart.ms.max.toFixed(2)} ms max`);
          t.diagnostic(results[results.length - 1]);
          for (const k of ['xray', 'sensors', 'both']) {
            assert.ok(mean(runs[k], 'fps') >= mean(runs.off, 'fps') - noise.fps, `${k}: frame rate ${mean(runs[k], 'fps').toFixed(1)} fps, without ${mean(runs.off, 'fps').toFixed(1)}`);
            assert.ok(mean(runs[k], 'simRate') >= mean(runs.off, 'simRate') - noise.rtf, `${k}: RTF ${mean(runs[k], 'simRate').toFixed(2)}, without ${mean(runs.off, 'simRate').toFixed(2)}`);
          }
          assert.ok(chart.ms.p95 < 2, `the charts' 95th-percentile draw ${chart.ms.p95.toFixed(2)} ms`);
        }
        assert.deepEqual(errors, []);
        await page.close();
      }
    } finally {
      await browser.close();
    }
  });
});
