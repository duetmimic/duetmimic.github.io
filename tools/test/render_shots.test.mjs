// Browser test of demo/js/render/{frame,scene}.js: headless Chromium renders the
// fixture poses (traces/render_poses.json) and saves 1280x720 screenshots. It
// also checks the camera modes, and picking
// for the drag-to-pull feature against MuJoCo's own geom frames (pickChecks()
// in render_pose.html), and saves render_pull.png with a pull drawn. Then the
// X-ray view (xrayChecks()): ghosts by role, the socket load arrows against
// MuJoCo's socket anchor, and the exact way back; saves render_xray.png. Then
// the sensor markers (sensorChecks()): each at its sensor in MuJoCo (site or
// joint anchor) and drawn there, the contact glyphs by their force, the
// prewarmed program, the draw calls back to the baseline; saves
// render_sensors.png.
//
//   cd tools
//   DUET_BROWSER_TESTS=1 PLAYWRIGHT_BROWSERS_PATH=<ms-playwright> \
//     RENDER_SHOTS_DIR=<folder> node --test test/render_shots.test.mjs
//
// Skipped unless DUET_BROWSER_TESTS=1. The model is the asset folder's slim
// physics.mjb.gz (paths.mjs; same ids and kinematics as the bundle model);
// RENDER_MODEL=bundle loads the ORIGINAL bundle model.mjb instead (paths.mjs
// BUNDLE, 103 MB: tens of seconds in wasm). It serves the repository on
// 127.0.0.1:RENDER_PORT (default 8731) for the duration of the test, the bundle
// folder at /__bundle__/ (it lies outside the repository: B is in mjlab/) and
// a DUET_ASSETS folder outside the repository at /__assets__/.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { ASSETS, BUNDLE, REPO, servedAssets } from './paths.mjs';

const ROOT = fileURLToPath(REPO);
const PORT = Number(process.env.RENDER_PORT || 8731);
const OUT = process.env.RENDER_SHOTS_DIR || fileURLToPath(new URL('../out/render/', import.meta.url));
const MODEL = process.env.RENDER_MODEL || 'slim';
// URL prefix -> folder, besides the repository at /.
const MOUNTS = {};
if (MODEL === 'bundle') MOUNTS['/__bundle__/'] = fileURLToPath(BUNDLE);
const ASSETS_URL = servedAssets() || '/__assets__/';
if (!servedAssets()) MOUNTS['/__assets__/'] = fileURLToPath(ASSETS);
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

function serve(root, port, mounts = {}) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://127.0.0.1');
    const pathname = decodeURIComponent(url.pathname);
    const prefix = Object.keys(mounts).find((p) => pathname.startsWith(p));
    const base = prefix ? mounts[prefix] : root;
    const file = path.join(base, prefix ? pathname.slice(prefix.length) : pathname);
    if (!file.startsWith(base)) { res.writeHead(403); res.end(); return; }
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
  const server = await serve(ROOT, PORT, MOUNTS);
  // E2E_GPU=1: the hardware GPU (ANGLE on Vulkan), as e2e_browser; else SwiftShader.
  const gpu = process.env.E2E_GPU === '1' ? ['--use-angle=vulkan', '--ignore-gpu-blocklist'] : [];
  const browser = await chromium.launch({ args: [...gpu, '--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    const bundle = MODEL === 'bundle' ? `&bundle=${encodeURIComponent('/__bundle__/model.mjb')}` : '';
    const url = `http://127.0.0.1:${PORT}/tools/test/render_pose.html?model=${MODEL}&assets=${encodeURIComponent(ASSETS_URL)}${bundle}`;
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
      `chase azimuth error ${cam.chaseError.toFixed(2)} deg, elevation error ${cam.chaseElevationError.toFixed(2)} deg, free-mode drift ${(cam.freeDrift * 1000).toFixed(3)} mm, ` +
      `free-mode light lag ${(cam.lightLag * 1000).toFixed(2)} mm`);
    assert.ok(cam.follow < 0.01 && cam.offsetDrift < 0.001, 'follow mode');
    assert.ok(cam.chaseError < 3 && cam.chaseElevationError < 3, 'chase mode');
    assert.ok(cam.freeDrift < 1e-6 && cam.lightLag < 0.01, 'free mode: camera stays, light follows the walker');

    // Picking: grabbed points lie on the named geom in MuJoCo's own frame,
    // within the worker's reach rule, under the cursor; drag targets too.
    const pk = await page.evaluate(() => window.renderTest.pickChecks());
    console.log(`# pick: ${pk.hits} hits in ${pk.picks} cursor positions (${pk.muscleOnly} on muscles only, ${pk.slopHits} more within 4 px), ` +
      `${pk.bodies} bodies, ${pk.ms.toFixed(3)} ms per pick; errors: world ${(pk.world * 1000).toFixed(4)} mm, screen ${pk.screen.toFixed(3)} px, ` +
      `reach ${(pk.reach * 100).toFixed(1)} cm past geom_rbound, drag target ${pk.target.toFixed(3)} px, off its plane ${(pk.plane * 1000).toFixed(4)} mm; ` +
      `overlay ${JSON.stringify(pk.overlay)}`);
    assert.ok(pk.hits > 400 && pk.bodies > 20 && pk.muscleOnly > 0, 'the walker can be grabbed, muscles included');
    assert.ok(pk.world < 1e-4, `a grabbed point is ${pk.world} m off MuJoCo's geom frame`);
    assert.ok(pk.screen < 0.5 && pk.target < 0.5, 'grabbed points and drag targets sit under the cursor');
    assert.ok(pk.reach < 0.05, `a grabbed point lies ${pk.reach} m past geom_rbound (the worker allows 0.05)`);
    assert.ok(pk.plane < 1e-6, 'drag targets lie in the plane through the grabbed point, square to the view');
    assert.ok(pk.ms < 2, `${pk.ms} ms per pick`);
    const o = pk.overlay;
    assert.ok(o.shown && o.grabDot < 0.6 && o.targetDot < 0.6, 'the pull is drawn from the grabbed point to the target');
    assert.ok(Math.abs(o.lineWidth - 3.4) < 0.11, `half the maximum force widens the line half way (${o.lineWidth} px)`);
    assert.deepEqual(o.litBodies, [o.hit.body], 'only the grabbed body glows');
    assert.ok(o.hidden && o.unlit, 'setPerturb(null) clears the pull and the glow');
    await page.evaluate(() => {
      const t = window.renderTest, v = t.view;
      t.shot({ pose: 'stride', azimuth: -60, elevation: -10, distance: 3.0 });
      const xy = v.projectToClient(v.geomPoint(21));
      const hit = v.pick(xy[0], xy[1], 24);
      v.setPerturb({ geom: hit.geom, local: hit.local, target: v.dragTarget(xy[0] + 140, xy[1] - 30, hit), force: 300, maxForce: 400 });
      v.render();
    });
    await page.screenshot({ path: path.join(OUT, 'render_pull.png') });
    await page.evaluate(() => { window.renderTest.view.setPerturb(null); window.renderTest.view.render(); });

    // X-ray view: ghosts and base materials by role, the arrows against
    // MuJoCo's socket anchor for set wrenches, the blend, the glow on ghosts,
    // picking, no compile on the first toggle, and the exact way back.
    const xr = await page.evaluate(() => window.renderTest.xrayChecks());
    console.log(`# x-ray: roles ${JSON.stringify(xr.roles)}; draws ${JSON.stringify(xr.draws)}; anchor ${(xr.anchorWorld * 1000).toFixed(4)} mm, ` +
      `${xr.anchorPx.toFixed(3)} px; arrow origin ${xr.originPx.toFixed(3)} px, direction ${xr.direction.toExponential(2)}, length ${xr.length.toExponential(2)} m; ` +
      `blend alpha ${xr.blend.alpha.toFixed(3)} (wrench error ${xr.blend.error.toExponential(2)}, length ${xr.blend.length.toExponential(2)} m); ` +
      `arrows ${JSON.stringify(xr.arrows)}; glow ${JSON.stringify(xr.glow)}; pick ${JSON.stringify(xr.pick)}; programs added ${xr.programsAdded}; ` +
      `restore ${JSON.stringify(xr.restore)}`);
    assert.deepEqual(xr.errors, []);
    assert.ok(xr.roles.bone > 0 && xr.roles.socket > 0 && xr.roles.hardware > 0 && xr.roles.cover > 0, 'every role is present');
    assert.ok(xr.musclesHidden, 'X-ray hides the muscles');
    assert.ok(xr.draws.xray.calls < xr.draws.base.calls && xr.draws.xray.triangles < xr.draws.base.triangles, 'X-ray draws less');
    assert.equal(xr.programsAdded, 0, 'the first toggle compiles no shader (prewarmed)');
    assert.ok(xr.anchorWorld < 1e-5 && xr.anchorPx < 1, 'the anchor is the socket point of MuJoCo\'s geom frame');
    assert.ok(xr.originPx < 1, 'the arrows start at the socket anchor');
    assert.ok(xr.direction < 1e-6, `arrow direction off by ${xr.direction}`);
    assert.ok(xr.length < 1e-6, `arrow length off by ${xr.length} m`);
    assert.ok(Math.abs(xr.arrows.capped.force.length - 0.8) < 1e-9, 'the force arrow is capped at 0.8 m');
    assert.ok(Math.abs(xr.arrows.capped.moment.length - 0.6) < 1e-9, 'the moment arrow is capped at 0.6 m');
    assert.ok(!xr.arrows.tiny.force.shown && !xr.arrows.tiny.moment.shown && !xr.arrows.swing.moment.shown, 'arrows below their minimum are hidden');
    assert.ok(xr.arrows.swing.force.shown, 'a swing load (50 N down) is drawn');
    assert.ok(xr.blend.alpha > 0.1 && xr.blend.alpha < 0.9 && xr.blend.error < 1e-4 && xr.blend.length < 1e-6, 'the wrench blends with the pose');
    assert.ok(xr.initial && xr.initial.wrench === null && !xr.initial.force && !xr.initial.moment && xr.initial.dot, 'the reset state (frame.initial) draws the dot, no arrows');
    assert.ok(!xr.nonFinite['Fz Infinity'].force && !xr.nonFinite['F NaN'].force && !xr.nonFinite['Tz Infinity'].moment, 'a non-finite wrench draws no arrow');
    assert.deepEqual(xr.glow.litBodies, [xr.glow.body], 'only the grabbed body glows in X-ray');
    assert.ok(xr.glow.litGhosts && xr.glow.litOff && xr.glow.litBack && xr.glow.unlit, 'the glow follows the grab across X-ray toggles');
    assert.ok(xr.pick.hits > 100 && xr.pick.muscleOnly === 0, 'ghost bones can be grabbed, hidden muscles not');
    assert.ok(xr.skipped && xr.restoredIdle, 'muscles skip poses in X-ray and are posed again for the pose on screen');
    const rs = xr.restore;
    assert.ok(rs.materials && rs.shadows && rs.order && rs.muscles && rs.arrowsHidden, 'X-ray off restores the base look');
    assert.deepEqual(rs.draws, rs.base, 'draw calls and triangles are back to the baseline');
    await page.evaluate(() => {
      const t = window.renderTest;
      t.view.setXray(true);
      t.shot({ pose: 'stride', azimuth: -100, elevation: -6, distance: 2.4, wrench: [-90, 50, 700, 30, -20, 18] });
    });
    await page.screenshot({ path: path.join(OUT, 'render_xray.png') });
    await page.evaluate(() => { window.renderTest.view.setXray(false); window.renderTest.view.render(); });

    // Sensor markers: at their sensors and geom points, drawn where projected,
    // contact glyphs by force, no compile on the first toggle, the way back.
    const sn = await page.evaluate(() => window.renderTest.sensorChecks());
    console.log(`# sensors: ${sn.count} markers; at the sensors ${(sn.anchor * 1000).toFixed(4)} mm, at the geom points ${(sn.world * 1000).toFixed(4)} mm, ` +
      `on screen ${sn.screen.toFixed(3)} px; pixels ${JSON.stringify(sn.pixels)}; contact ${JSON.stringify(sn.contact)}; blend ${JSON.stringify(sn.blend)}; ` +
      `draws ${JSON.stringify(sn.draws)} -> off ${JSON.stringify(sn.off)}; programs added ${sn.programsAdded}; picks unchanged ${sn.pickSame}; x-ray ${JSON.stringify(sn.xray)}`);
    assert.deepEqual(sn.errors, []);
    assert.equal(sn.count, 6, 'a marker per OSL sensor');
    assert.ok(sn.anchor < 1e-5 && sn.world < 1e-5, 'each marker sits at its sensor (site or joint anchor; a touch zone\'s where the zone meets the foot) on its drawn geom');
    assert.ok(sn.zoneCentre < 1e-5, 'the touch zones\' centres ride their geom');
    assert.ok(sn.screen < 0.5, 'and on screen where projectToClient puts its geom point');
    assert.ok(sn.pixels.length >= 3, `the glyphs not under another were checked (${sn.pixels.length})`);
    assert.equal(sn.programsAdded, 0, 'the first toggle compiles no shader (prewarmed)');
    assert.equal(sn.draws.on.calls, sn.draws.base.calls + 1, 'the markers are one draw call');
    assert.deepEqual(sn.off, sn.draws.base, 'off: draw calls and triangles back to the baseline');
    assert.ok(sn.blend.alpha > 0.1 && sn.blend.alpha < 0.9 && sn.blend.error < 1e-3, 'the sensor frame blends with the pose');
    assert.ok(sn.pickSame, 'the markers are never picked');
    assert.ok(sn.xray.markers && sn.xray.xray && sn.xray.order, 'with the X-ray view, the markers over the arrows');
    await page.evaluate(() => {
      const t = window.renderTest, v = t.view;
      const ov = t.demo.visual.overlays.markers;
      const heel = ov.find((m) => m.key === 'heel').channels[0], toe = ov.find((m) => m.key === 'toe').channels[0];
      v.setSensorMarkers(true);
      const info = t.shot({ pose: 'stride', sensors: { [heel]: 650, [toe]: 120 } });
      v.setCameraMode('free'); // the camera stays where it is put
      v.setView({ azimuth: -75, elevation: -4, distance: 1.3, target: [info.root[0], info.root[1], 0.36] });
      for (let i = 0; i < 3; i++) v.render();
    });
    await page.screenshot({ path: path.join(OUT, 'render_sensors.png') });
    await page.evaluate(() => { const v = window.renderTest.view; v.setSensorMarkers(false); v.setCameraMode('follow'); v.render(); });

    // Framing (the walker moved and shrunk clear of the panels on phones): the
    // markers project to the mapped point and are drawn there, a pick there
    // grabs the same point, a resize keeps it, null gives the view back.
    const fr = await page.evaluate(() => window.renderTest.framingChecks());
    console.log(`# framing ${JSON.stringify(fr.framing)}: markers ${fr.mapped.toFixed(4)} px from the mapped points, ${fr.pixels} glyph centres drawn there; sizes ${JSON.stringify(fr.sizes)}; ` +
      `pick ${JSON.stringify(fr.pick)}; after a resize ${fr.resized.toFixed(4)} px; cleared ${fr.back.toFixed(6)} px from before`);
    assert.deepEqual(fr.errors, []);
    assert.ok(fr.mapped < 0.01, 'the markers project to the framed points');
    assert.ok(fr.sizes.at && fr.sizes.min && fr.sizes.back, 'the glyphs shrink with the framing (to the minimum scale), and grow back');
    assert.ok(fr.pixels >= 2, `filled glyphs drawn where projected under framing (${fr.pixels})`);
    assert.ok(fr.pick && fr.pick.geom && fr.pick.point < 1e-3, 'a pick under framing grabs the same point');
    assert.ok(fr.calls, 'framing adds no draw call');
    assert.ok(fr.resized < 0.01, 'a resize keeps the framing');
    assert.ok(fr.cleared && fr.back < 1e-6, 'null gives the unframed view back');

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
