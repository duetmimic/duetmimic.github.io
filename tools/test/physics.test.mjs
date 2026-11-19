// Slim physics model checks (demo/assets/stage2/physics.mjb.gz, or DUET_ASSETS):
//   node --test test/physics.test.mjs        (from tools/)
//
// Loads the model through the demo's own loaders (demo/js/lib/{container,mujoco}.js)
// in the vendored MuJoCo 3.11.0 WASM build, checks its counts against demo.json,
// steps it, and compares a 200-step run with the native MuJoCo trace the
// exporter recorded (PHYSICS_REF in paths.mjs: tools/test/physics_wasm_ref.bin.gz
// for the demo's folder, the file of that name inside DUET_ASSETS otherwise).
// Both files come from
//   python -m duet_export --out ../demo/assets/stage2 --only physics
// The WASM-vs-native difference and the step timings print as diagnostics.

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { fetchBytes, loadContainer, loadJSON } from '../../demo/js/lib/container.js';
import { loadMujoco, modelFromBytes } from '../../demo/js/lib/mujoco.js';
import { ASSETS, PHYSICS_REF } from './paths.mjs';

const MODEL = new URL('physics.mjb.gz', ASSETS);
const DEMO = new URL('demo.json', ASSETS);
const NATIVE = PHYSICS_REF;
// WASM vs native x86-64 after 200 steps (0.4 s) of contact-rich motion. Both
// run the same MuJoCo 3.11.0 C code; only compiler and libm rounding differ,
// and contact onsets amplify it (measured on R2, 2026-11-16: 6e-14 at step
// 25, 8.9e-7 at its worst; on B, 2026-11-18: 2.7e-13 and 2.3e-6).
// Deterministic, so the margin only absorbs re-exports.
const NATIVE_TOLERANCE = 1e-5;

const missing = [MODEL, DEMO, NATIVE].filter((url) => !existsSync(fileURLToPath(url)));
const skip = missing.length
  ? `missing ${missing.map((url) => fileURLToPath(url)).join(', ')}: run python -m duet_export --out ${fileURLToPath(ASSETS)} --only physics`
  : false;

let mujoco, model, data, demo, native, bytes;
if (!skip) {
  demo = await loadJSON(DEMO);
  native = await loadContainer(NATIVE);
  bytes = await fetchBytes(MODEL); // gunzipped .mjb
  mujoco = await loadMujoco();
  ({ model, data } = modelFromBytes(mujoco, bytes, 'physics.mjb'));
}
after(() => {
  data?.delete();
  model?.delete();
});

const allFinite = (...arrays) => arrays.every((a) => a.every(Number.isFinite));
const maxAbs = (a, b) => {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
};
// The exporter's control pattern, exact in numpy and JavaScript doubles.
function setPatternCtrl(t) {
  const ctrl = data.ctrl; // re-read views: WASM memory may have grown
  for (let i = 0; i < model.nu; i++) ctrl[i] = ((7 * i + 3 * t) % 11) / 10;
}
// The native trace's start: a mid-stride reference frame of the 1 m/s walk.
function startNative() {
  mujoco.mj_resetData(model, data);
  data.qpos.set(native.arrays.qpos0);
  data.qvel.set(native.arrays.qvel0);
  mujoco.mj_forward(model, data);
}

test('physics.mjb.gz is the model demo.json describes', { skip }, () => {
  const physics = demo.physics;
  assert.equal(bytes.length, physics.raw_bytes);
  const sha = createHash('sha256').update(bytes).digest('hex');
  assert.equal(sha, native.meta.physics_sha256, 'native trace is stale: re-run --only physics');
  for (const [key, value] of Object.entries(physics.counts)) {
    assert.equal(model[key], value, key);
  }
  assert.equal(model.opt.timestep, physics.timestep);
  assert.equal(model.nmeshvert, physics.stripped.sizes.nmeshvert[1]);
  assert.equal(model.nbvh, physics.stripped.sizes.nbvh[1]);
});

test('100 steps from the default pose and from mid-stride stay finite', { skip }, () => {
  mujoco.mj_resetData(model, data);
  for (let t = 0; t < 100; t++) mujoco.mj_step(model, data);
  assert.ok(allFinite(data.qpos, data.qvel, data.act, data.sensordata), 'NaN from the default pose');
  startNative();
  for (let t = 0; t < 100; t++) {
    setPatternCtrl(t);
    mujoco.mj_step(model, data);
  }
  assert.ok(allFinite(data.qpos, data.qvel, data.act, data.sensordata), 'NaN mid-stride');
  assert.ok(data.ncon > 0, 'expected floor contacts mid-stride');
});

test('WASM matches native MuJoCo on the slim model', { skip }, (t) => {
  const { steps, every } = native.meta;
  const [count, nq] = native.arrays.qpos.shape;
  const nv = native.arrays.qvel.shape[1];
  const na = native.arrays.act.shape[1];
  startNative();
  const rows = [];
  let worst = 0;
  for (let step = 0; step < steps; step++) {
    setPatternCtrl(step);
    mujoco.mj_step(model, data);
    if ((step + 1) % every) continue;
    const k = (step + 1) / every - 1;
    const diff = Math.max(
      maxAbs(data.qpos, native.arrays.qpos.subarray(k * nq, (k + 1) * nq)),
      maxAbs(data.qvel, native.arrays.qvel.subarray(k * nv, (k + 1) * nv)),
      maxAbs(data.act, native.arrays.act.subarray(k * na, (k + 1) * na)),
    );
    worst = Math.max(worst, diff);
    rows.push(`step ${step + 1}: ncon ${data.ncon}/${native.arrays.ncon[k]} max|diff| ${diff.toExponential(2)}`);
    assert.equal(data.ncon, native.arrays.ncon[k], `contact count at step ${step + 1}`);
  }
  assert.equal(rows.length, count);
  t.diagnostic(`WASM vs native over ${steps} steps (qpos, qvel, act): max |diff| ${worst.toExponential(3)}`);
  for (const row of rows) t.diagnostic(row);
  assert.ok(worst <= NATIVE_TOLERANCE, `max |diff| ${worst} > ${NATIVE_TOLERANCE}`);
});

test('mj_step timing (diagnostic)', { skip }, (t) => {
  startNative();
  for (let step = 0; step < 50; step++) {
    setPatternCtrl(step);
    mujoco.mj_step(model, data);
  }
  const n = 500;
  const started = performance.now();
  for (let step = 0; step < n; step++) {
    setPatternCtrl(step);
    mujoco.mj_step(model, data);
  }
  const perStep = (performance.now() - started) / n;
  t.diagnostic(`WASM mj_step ${perStep.toFixed(3)} ms (ncon ${data.ncon}); one 10 ms control step = 5 ticks = ${(5 * perStep).toFixed(2)} ms`);
  assert.ok(allFinite(data.qpos, data.qvel));
});
