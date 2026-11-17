// Browser test of demo/js/render/{frame,scene}.js: headless Chromium renders the
// fixture poses (traces/render_poses.json) with the ORIGINAL bundle model and
// saves 1280x720 screenshots.
//
//   cd tools
//   DUET_BROWSER_TESTS=1 PLAYWRIGHT_BROWSERS_PATH=<ms-playwright> \
//     RENDER_SHOTS_DIR=<folder> node --test test/render_shots.test.mjs
//
// Skipped unless DUET_BROWSER_TESTS=1 (loading the 103 MB model.mjb in wasm takes
// tens of seconds). RENDER_MODEL=slim uses demo/assets/r2/physics.mjb.gz instead.
// It serves the folder holding duetmimic.github.io and osl_deploy on
// 127.0.0.1:RENDER_PORT (default 8731) for the duration of the test.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const PORT = Number(process.env.RENDER_PORT || 8731);
const OUT = process.env.RENDER_SHOTS_DIR || fileURLToPath(new URL('../out/render/', import.meta.url));
const MODEL = process.env.RENDER_MODEL || 'bundle';
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css', '.png': 'image/png',
};

// Shots: absolute azimuth (deg, camera seen from the target, CCW from +x),
// elevation (deg, < 0 looks down), distance (m). The stride pose walks along +x,
// the OSL is its right leg, on the -y side. No view = the demo's default view.
const SHOTS = [
  { file: 'render_default.png', pose: 'stride' },
  { file: 'render_side.png', pose: 'stride', azimuth: -90, elevation: -3, distance: 3.0 },
  { file: 'render_front.png', pose: 'idle', azimuth: 25, elevation: -24, distance: 3.6 },
  { file: 'render_osl.png', pose: 'stride', azimuth: -55, elevation: -8, distance: 1.25, targetZ: 0.42 },
];

function serve(root, port) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://127.0.0.1');
    const file = path.join(root, decodeURIComponent(url.pathname));
    if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
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

test('renderer screenshots in headless Chromium', { skip: process.env.DUET_BROWSER_TESTS !== '1' && 'set DUET_BROWSER_TESTS=1' }, async () => {
  const { chromium } = await import('playwright');
  fs.mkdirSync(OUT, { recursive: true });
  const server = await serve(ROOT, PORT);
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    const url = `http://127.0.0.1:${PORT}/duetmimic.github.io/tools/test/render_pose.html?model=${MODEL}`;
    const t0 = Date.now();
    await page.goto(url);
    await page.waitForFunction(() => window.renderReady || window.renderError, null, { timeout: 240000, polling: 500 });
    const failure = await page.evaluate(() => window.renderError || null);
    assert.equal(failure, null, failure);
    const env = await page.evaluate(() => ({ gl: window.renderTest.glRenderer, model: window.renderTest.modelSeconds }));
    console.log(`# page ready in ${((Date.now() - t0) / 1000).toFixed(1)} s (model ${env.model.toFixed(1)} s), GL: ${env.gl}`);

    for (const shot of SHOTS) {
      const info = await page.evaluate((s) => {
        const t = window.renderTest;
        if (s.targetZ === undefined) return t.shot(s);
        const base = t.shot({ pose: s.pose });
        t.view.setCameraMode('free');
        t.view.setView({ azimuth: s.azimuth, elevation: s.elevation, distance: s.distance, target: [base.root[0], base.root[1], s.targetZ] });
        for (let i = 0; i < 3; i++) t.view.render();
        return { ...base, ...t.view.stats };
      }, shot);
      const file = path.join(OUT, shot.file);
      await page.screenshot({ path: file });
      const bytes = fs.statSync(file).size;
      console.log(`# ${file}: ${info.triangles} triangles, ${info.calls} draw calls, ${info.segments} muscle segments, ${bytes} bytes`);
      assert.ok(info.triangles > 50000 && info.segments > 700, 'scene looks empty');
      assert.ok(bytes > 20000, 'screenshot looks blank');
    }
    const ms = await page.evaluate(() => window.renderTest.timing(20));
    console.log(`# render() + 1-pixel readPixels: ${ms.toFixed(1)} ms per frame at 1280x720 (${env.gl})`);

    const cam = await page.evaluate(() => window.renderTest.cameraChecks());
    console.log(`# camera: follow error ${(cam.follow * 1000).toFixed(2)} mm, orbit offset drift ${(cam.offsetDrift * 1000).toFixed(3)} mm, ` +
      `chase azimuth error ${cam.chaseError.toFixed(2)} deg, free-mode drift ${(cam.freeDrift * 1000).toFixed(3)} mm, ` +
      `free-mode light lag ${(cam.lightLag * 1000).toFixed(2)} mm`);
    assert.ok(cam.follow < 0.01 && cam.offsetDrift < 0.001, 'follow mode');
    assert.ok(cam.chaseError < 3, 'chase mode');
    assert.ok(cam.freeDrift < 1e-6 && cam.lightLag < 0.01, 'free mode: camera stays, light follows the walker');

    // Resize follows the host; dispose removes the canvas.
    await page.setViewportSize({ width: 800, height: 450 });
    await page.waitForFunction(() => window.renderTest.view.renderer.domElement.width === 800 * devicePixelRatio, null, { timeout: 10000 });
    const gone = await page.evaluate(() => { window.renderTest.view.dispose(); return document.querySelector('#viewport canvas') === null; });
    assert.ok(gone, 'dispose() should remove the canvas');
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
