// SimCore against osl_deploy's own Simulation (node --test).
//
// Needs the ORIGINAL bundle model (paths.mjs BUNDLE, the bundle the exporter
// defaults to; override with DUET_BUNDLE=<folder>) and the git-ignored trace
// written by
//   cd tools && python -m duet_export.trace
// (tools/traces/trace.bin.gz, DUET_TRACE=<file>; see tools/duet_export/trace.py
// for its rows). Without them the trace tests are skipped, and so they are
// when the trace's manifest_sha256 is not demo.json's source.manifest_sha256
// (a stale trace, from another bundle than the assets) or not the bundle's.
//
// Checks, driving everything from the trace's recorded requests and actions
// (no networks run here):
//   (a) the JS GaitReference + request latch reproduce the reference state at
//       every boundary;
//   (b) with the trace's MuJoCo state set and mj_forward, the JS human
//       observation matches the recorded one;
//   (c) the JS OSL observation + previous-command history, fed the trace's
//       states and recorded OSL means, reproduce the recorded OSL inputs;
//   (d) SimCore with stub policies replaying the recorded means: the delayed
//       ctrl is exact, and WASM physics tracks the recorded qpos (divergence
//       reported, exactness not required);
// plus the per-step physics error from recorded states, the reset state, and
// Node timings of SimCore.step with stub policies.
//
// Also, on the slim demo model alone (no trace needed): SimCore's disturbance
// hook clears xfrc_applied and runs before every mj_step.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { fetchBytes, loadContainer, loadJSON } from '../../demo/js/lib/container.js';
import { loadMujoco, modelFromBytes } from '../../demo/js/lib/mujoco.js';
import {
  CommandLatch, CommandSupport, GaitReference, HumanObservation, OslObservation,
  PreviousCommand, SimCore,
} from '../../demo/js/sim/index.js';
import { ASSETS, BUNDLE, TRACE } from './paths.mjs';

const MODEL = new URL('model.mjb', BUNDLE);
const MANIFEST = new URL('manifest.json', BUNDLE);
const DEMO = new URL('demo.json', ASSETS);
const REFERENCE = new URL('reference.bin.gz', ASSETS);
const missing = [MODEL, MANIFEST, TRACE, DEMO, REFERENCE].filter((u) => !existsSync(fileURLToPath(u)));

// A trace is only ground truth for the assets and the model of its own bundle.
const loaded = missing.length ? null : await loadContainer(TRACE);
function staleness() {
  const recorded = loaded.meta.manifest_sha256;
  const demo = JSON.parse(readFileSync(fileURLToPath(DEMO), 'utf8'));
  const assets = demo.source && demo.source.manifest_sha256;
  const bundle = createHash('sha256').update(readFileSync(fileURLToPath(MANIFEST))).digest('hex');
  const short = (sha) => (sha ? sha.slice(0, 12) : 'none');
  if (recorded !== assets) {
    return `stale trace: ${fileURLToPath(TRACE)} was recorded from manifest ${short(recorded)} `
      + `(${loaded.meta.bundle}), the assets from ${short(assets)} (${demo.source && demo.source.bundle}): `
      + 'regenerate with python -m duet_export.trace';
  }
  if (recorded !== bundle) {
    return `the trace's bundle (manifest ${short(recorded)}) is not ${fileURLToPath(BUNDLE)} `
      + `(manifest ${short(bundle)}): set DUET_BUNDLE to the bundle the assets came from`;
  }
  return false;
}
const SKIP = missing.length ? `missing ${missing.map((u) => fileURLToPath(u)).join(', ')}` : staleness();

const F32_EPS = 2 ** -23;
const row = (a, width, i) => a.subarray(width * i, width * (i + 1));

// The bundle model through the Emscripten FS (fast); MjVFS.addBuffer copies a
// 103 MB buffer element by element (about 20 s), so it is only the fallback.
function loadModel(mujoco, bytes) {
  if (mujoco.FS && mujoco.FS.writeFile) {
    const path = '/sim_test_model.mjb';
    mujoco.FS.writeFile(path, bytes);
    const vfs = new mujoco.MjVFS();
    try {
      return mujoco.MjModel.from_binary_path(path, vfs);
    } finally {
      vfs.delete();
      mujoco.FS.unlink(path);
    }
  }
  return modelFromBytes(mujoco, bytes).model;
}

// Max |js - py| and the largest excess over float32 rounding of py, per group.
function compare(js, py, groups, stats) {
  for (const [name, [start, stop]] of groups) {
    const s = stats[name] || (stats[name] = { maxAbs: 0, maxExcess: 0, at: -1 });
    for (let i = start; i < stop; i++) {
      const d = Math.abs(js[i] - py[i]);
      if (!(d <= s.maxAbs)) { s.maxAbs = d; s.at = i; }
      const excess = d - 2 * F32_EPS * Math.abs(py[i]);
      if (excess > s.maxExcess) s.maxExcess = excess;
    }
  }
}

function overall(stats) {
  let maxAbs = 0, maxExcess = 0;
  for (const s of Object.values(stats)) {
    maxAbs = Math.max(maxAbs, s.maxAbs);
    maxExcess = Math.max(maxExcess, s.maxExcess);
  }
  return { maxAbs, maxExcess };
}

const fmt = (x) => (x === 0 ? '0' : x.toExponential(2));

describe('SimCore vs osl_deploy trace', { skip: SKIP }, () => {
  let mujoco, model, demo, reference, T, meta, N;
  const results = {};

  before(async () => {
    mujoco = await loadMujoco();
    const started = performance.now();
    model = loadModel(mujoco, await fetchBytes(MODEL));
    results.modelLoadMs = performance.now() - started;
    demo = await loadJSON(DEMO);
    reference = await loadContainer(REFERENCE);
    T = loaded.arrays;
    meta = loaded.meta;
    N = meta.steps;
  });

  test('(a)-(c) reference state, human and OSL observations from recorded states', () => {
    const sim = demo.sim;
    const nq = sim.model.nq, nv = sim.model.nv, na = sim.model.na, nu = sim.model.nu;
    const ref = new GaitReference(demo.reference, reference.arrays);
    const support = new CommandSupport(demo.reference, ref);
    const latch = new CommandLatch(sim.rebase_steps);
    latch.reset(ref.twist);
    const human = new HumanObservation(sim, demo.reference);
    const osl = new OslObservation(sim.osl.observation);
    const previous = new PreviousCommand(sim.osl.observation.previous_command);
    const data = new mujoco.MjData(model);
    const humanIn = new Float32Array(sim.human.observation.size);
    const oslIn = new Float32Array(sim.osl.observation.size);
    const humanGroups = Object.entries(sim.human.observation.groups);
    const oslGroups = [
      ...sim.osl.observation.terms.map((t) => [t.name, [t.offset, t.offset + t.history * t.frame_dim]]),
      ['last_action', [sim.osl.observation.previous_command.offset, sim.osl.observation.size]],
    ];
    const ints = ['clip', 'previous', 'frame', 'blend_count', 'loops'];
    const floats = { previous_frame: 1, previous_rate: 1, anchor: 3, previous_anchor: 3, catchup: 2, twist: 3 };
    const refStats = { intMismatches: 0, sinceAnchorMismatches: 0, firstMismatch: null };
    for (const key of Object.keys(floats)) refStats[key] = 0;
    const humanStats = {}, oslStats = {};
    let forwardMs = 0;

    const checkReference = (b) => {
      const s = ref.state();
      for (const key of ints) {
        let js = s[key];
        if (key === 'clip' || key === 'previous') js = ref.libraryIndex[js];
        if (js !== T[`ref_${key}`][b]) {
          refStats.intMismatches += 1;
          refStats.firstMismatch ??= { boundary: b, key, js, py: T[`ref_${key}`][b] };
        }
      }
      for (const [key, width] of Object.entries(floats)) {
        const py = row(T[`ref_${key}`], width, b);
        const js = width === 1 ? [s[key]] : s[key];
        for (let i = 0; i < width; i++) refStats[key] = Math.max(refStats[key], Math.abs(js[i] - py[i]));
      }
      if (latch.sinceAnchor !== T.since_anchor[b]) refStats.sinceAnchorMismatches += 1;
    };

    for (let b = 0; b < N; b++) {
      checkReference(b);
      // The recorded boundary state, then mj_forward (as the runner does).
      mujoco.mj_resetData(model, data);
      data.qpos.set(row(T.qpos, nq, b));
      data.qvel.set(row(T.qvel, nv, b));
      data.act.set(row(T.act, na, b));
      data.ctrl.set(row(T.ctrl, nu, b));
      data.qacc_warmstart.set(row(T.qacc_warmstart, nv, b));
      data.time = T.time[b];
      const t0 = performance.now();
      mujoco.mj_forward(model, data);
      forwardMs += performance.now() - t0;
      const v = {};
      for (const key of ['qpos', 'qvel', 'act', 'ctrl', 'actuator_length', 'actuator_velocity',
        'actuator_force', 'sensordata', 'xpos', 'xmat', 'cvel', 'subtree_com']) v[key] = data[key];
      // (b) human input at boundary b.
      assert.ok(human.build(v, ref, humanIn), `non-finite human observation at ${b}`);
      compare(humanIn, row(T.human_obs, humanIn.length, b), humanGroups, humanStats);
      // (c) OSL input as fed at step b.
      if (b === 0) { osl.reset(v.sensordata); previous.reset(); } else osl.observe(v.sensordata);
      assert.ok(osl.assemble(oslIn));
      previous.write(b, oslIn);
      compare(oslIn, row(T.osl_obs, oslIn.length, b), oslGroups, oslStats);
      previous.update(row(T.osl_mean, sim.osl.action.size, b));
      // (a) the step's command boundary: request, consume on the robot's pose
      // after the step's physics (boundary b + 1), advance.
      latch.request(row(T.requested, 3, b));
      latch.consume(ref, support, row(T.qpos, nq, b + 1), sim.root_qpos);
      ref.advance();
    }
    checkReference(N);
    results.reference = refStats;
    results.human = humanStats;
    results.osl = oslStats;
    results.forwardMs = forwardMs / N;

    const h = overall(humanStats), o = overall(oslStats);
    console.log(`\n[sim] trace: ${N} steps (${(N * meta.dt).toFixed(2)} s), model load ${results.modelLoadMs.toFixed(0)} ms, mj_forward ${results.forwardMs.toFixed(3)} ms`);
    console.log(`[sim] (a) reference: integer-key mismatches ${refStats.intMismatches}, since_anchor mismatches ${refStats.sinceAnchorMismatches}; max |diff| ${Object.keys(floats).map((k) => `${k} ${fmt(refStats[k])}`).join(', ')}`);
    console.log(`[sim] (b) human obs: max |diff| ${fmt(h.maxAbs)}, max excess over float32 rounding ${fmt(h.maxExcess)}`);
    for (const [name, s] of Object.entries(humanStats)) console.log(`[sim]     ${name.padEnd(22)} max |diff| ${fmt(s.maxAbs)}  excess ${fmt(s.maxExcess)}`);
    console.log(`[sim] (c) OSL obs: max |diff| ${fmt(o.maxAbs)}, max excess over float32 rounding ${fmt(o.maxExcess)}`);
    for (const [name, s] of Object.entries(oslStats)) console.log(`[sim]     ${name.padEnd(22)} max |diff| ${fmt(s.maxAbs)}  excess ${fmt(s.maxExcess)}`);

    assert.equal(refStats.intMismatches, 0, `reference mismatch ${JSON.stringify(refStats.firstMismatch)}`);
    assert.equal(refStats.sinceAnchorMismatches, 0);
    for (const key of Object.keys(floats)) assert.ok(refStats[key] < 1e-9, `reference ${key} differs by ${refStats[key]}`);
    // Beyond float32 rounding the inputs agree to ~1e-5 (contact forces are
    // the solver's, warm-started identically but in a different float build).
    assert.ok(h.maxExcess < 1e-3, `human observation differs by ${h.maxExcess}`);
    assert.ok(o.maxExcess < 1e-3, `OSL observation differs by ${o.maxExcess}`);
    assert.equal(oslStats.last_action.maxAbs, 0, 'previous-command history must be exact');
  });

  test('per-step physics from recorded states (WASM vs native, one control step)', () => {
    const sim = demo.sim;
    const nq = sim.model.nq, nv = sim.model.nv, na = sim.model.na, nu = sim.model.nu;
    const data = new mujoco.MjData(model);
    let qposErr = 0, qvelErr = 0, actErr = 0, worstAt = -1;
    for (let b = 0; b < N; b++) {
      mujoco.mj_resetData(model, data);
      data.qpos.set(row(T.qpos, nq, b));
      data.qvel.set(row(T.qvel, nv, b));
      data.act.set(row(T.act, na, b));
      data.qacc_warmstart.set(row(T.qacc_warmstart, nv, b));
      data.time = T.time[b];
      // The delayed ctrl is constant over a step here (15 and 5 ticks are whole
      // steps): the ctrl recorded at boundary b + 1 acted on all five ticks.
      data.ctrl.set(row(T.ctrl, nu, b + 1));
      for (let s = 0; s < sim.decimation; s++) mujoco.mj_step(model, data);
      const q = data.qpos, w = data.qvel, a = data.act;
      const pq = row(T.qpos, nq, b + 1), pv = row(T.qvel, nv, b + 1), pa = row(T.act, na, b + 1);
      for (let i = 0; i < nq; i++) { const d = Math.abs(q[i] - pq[i]); if (d > qposErr) { qposErr = d; worstAt = b; } }
      for (let i = 0; i < nv; i++) qvelErr = Math.max(qvelErr, Math.abs(w[i] - pv[i]));
      for (let i = 0; i < na; i++) actErr = Math.max(actErr, Math.abs(a[i] - pa[i]));
    }
    console.log(`[sim] one-step physics (5 ticks from each recorded state): max |qpos diff| ${fmt(qposErr)} (step ${worstAt}), |qvel diff| ${fmt(qvelErr)}, |act diff| ${fmt(actErr)}`);
    results.oneStep = { qposErr, qvelErr, actErr };
    assert.ok(qposErr < 1e-6, `one-step qpos error ${qposErr}`);
  });

  test('(d) SimCore replaying the recorded actions: delay exact, physics tracks', async () => {
    const sim = demo.sim;
    const nq = sim.model.nq, nu = sim.model.nu;
    const data = new mujoco.MjData(model);
    // Widths from demo.json: the human and OSL action sizes, the OSL input and
    // where its previous-command history starts (the sensor terms end there).
    const humanOut = sim.human.action.size, oslOut = sim.osl.action.size;
    const oslSize = sim.osl.observation.size, history = sim.osl.observation.previous_command.offset;
    let k = 0;
    const policies = {
      human: { run: async () => row(T.human_mean, humanOut, k) },
      osl: { run: async () => row(T.osl_mean, oslOut, k) },
    };
    const core = new SimCore({ mujoco, model, data, cfg: demo, reference, policies });
    const v0 = core.views();
    let resetQ = 0, resetObs = 0, resetOsl = 0;
    for (let i = 0; i < nq; i++) resetQ = Math.max(resetQ, Math.abs(v0.qpos[i] - T.qpos[i]));
    for (let i = 0; i < core.humanInput.length; i++) resetObs = Math.max(resetObs, Math.abs(core.humanInput[i] - T.human_obs[i]));
    for (let i = 0; i < history; i++) resetOsl = Math.max(resetOsl, Math.abs(core.oslInput[i] - T.osl_obs[i]));
    console.log(`[sim] reset: max |qpos - trace| ${fmt(resetQ)}, human obs ${fmt(resetObs)}, OSL sensor terms ${fmt(resetOsl)}`);
    assert.equal(resetQ, 0);

    let ctrlErr = 0, busErr = 0, clipMismatch = 0, firstFail = null;
    const checkpoints = [50, 100, 200, 300, 350, 400, 450, 500, 550, 600, 700, 1000, N].filter((c) => c <= N);
    const divergence = [];
    let worstQpos = 0, oslInputErr = 0;
    for (k = 0; k < N; k++) {
      core.requestTwist(row(T.requested, 3, k));
      // The input the OSL net would see this step (history + sensors).
      const info = await core.step();
      if (k < 100) {
        const py = row(T.osl_obs, oslSize, k);
        // core.oslInput now holds the NEXT boundary's sensor terms; compare the
        // last_action history written for this step instead.
        for (let i = history; i < oslSize; i++) oslInputErr = Math.max(oslInputErr, Math.abs(core.oslInput[i] - py[i]));
      }
      const v = core.views();
      const pc = row(T.ctrl, nu, k + 1), pb = row(T.bus, nu, k);
      for (let i = 0; i < nu; i++) {
        ctrlErr = Math.max(ctrlErr, Math.abs(v.ctrl[i] - pc[i]));
        busErr = Math.max(busErr, Math.abs(core.delay.ctrl[i] - pb[i]));
      }
      if (core.reference.libraryIndex[core.reference.clip] !== T.ref_clip[k + 1]) clipMismatch += 1;
      const pq = row(T.qpos, nq, k + 1);
      let e = 0;
      for (let i = 0; i < nq; i++) e = Math.max(e, Math.abs(v.qpos[i] - pq[i]));
      worstQpos = Math.max(worstQpos, e);
      if (checkpoints.includes(k + 1)) {
        const dx = v.qpos[0] - pq[0], dy = v.qpos[1] - pq[1], dz = v.qpos[2] - pq[2];
        divergence.push({ t: ((k + 1) * meta.dt).toFixed(2), qpos: e, root: Math.hypot(dx, dy, dz), siteError: info.siteError, traceSiteError: T.site_error[k] });
      }
      if (info.failed) { firstFail = { step: k, failure: info.failure }; break; }
    }
    // Open loop, the recorded actions cannot correct the tiny WASM/native
    // differences, which grow chaotically through the contacts: the replay
    // tracks for seconds, then drifts off and eventually trips the tracking
    // check. That is expected; the closed loop corrects it.
    console.log(`[sim] (d) replay: delayed ctrl max |diff| ${fmt(ctrlErr)}, command bus ${fmt(busErr)}, clip mismatches ${clipMismatch}, open-loop failure ${firstFail ? `at step ${firstFail.step} (${((firstFail.step + 1) * meta.dt).toFixed(2)} s): ${firstFail.failure}` : 'none'}`);
    for (const d of divergence) {
      console.log(`[sim]     t=${d.t.padStart(5)} s  max |qpos diff| ${fmt(d.qpos)}  root ${fmt(d.root)} m  site error js ${d.siteError.toFixed(4)} / trace ${d.traceSiteError.toFixed(4)}`);
    }
    results.replay = { ctrlErr, busErr, clipMismatch, divergence, worstQpos, firstFail };
    assert.equal(ctrlErr, 0, 'the delayed ctrl must equal the trace exactly');
    assert.equal(busErr, 0, 'the command bus must equal the trace exactly');
    assert.equal(oslInputErr, 0, 'previous-command history must be exact');
    assert.equal(clipMismatch, 0, 'the replay must switch clips where the trace did');
    const at = (seconds) => divergence.find((d) => Number(d.t) === seconds);
    assert.ok(at(1).qpos < 1e-6, `qpos diverged by ${at(1).qpos} after 1 s`);
    assert.ok(at(2).qpos < 1e-4, `qpos diverged by ${at(2).qpos} after 2 s`);
    assert.ok(!firstFail || firstFail.step >= 300, 'the open-loop replay failed within 3 s');
  });

  test('timing: SimCore.step with stub policies (physics + observations)', async () => {
    const data = new mujoco.MjData(model);
    let k = 0, physics = 0, observe = 0;
    // Stub policies replaying the trace's means (they keep it walking; zero
    // means would collapse it), with the trace's requests: idle, then walking.
    const humanOut = demo.sim.human.action.size, oslOut = demo.sim.osl.action.size;
    const policies = {
      human: { run: async () => row(T.human_mean, humanOut, k) },
      osl: { run: async () => row(T.osl_mean, oslOut, k) },
    };
    const core = new SimCore({ mujoco, model, data, cfg: demo, reference, policies });
    for (k = 0; k < 20; k++) await core.step(); // warm up
    core.reset();
    const steps = Math.min(300, N);
    const started = performance.now();
    for (k = 0; k < steps; k++) {
      core.requestTwist(row(T.requested, 3, k));
      const info = await core.step();
      physics += info.timing.physics;
      observe += info.timing.observe;
      if (info.failed) break;
    }
    const total = (performance.now() - started) / k;
    console.log(`[sim] Node timing over ${k} steps: SimCore.step ${total.toFixed(2)} ms (physics: 5 x mj_step ${(physics / k).toFixed(2)} ms; mj_forward + reference + observations ${(observe / k).toFixed(3)} ms)`);
    results.timing = { total, physics: physics / k, observe: observe / k };
    assert.ok(total < 50);
  });
});

// SimCore's disturbance hook (core.disturb, applyDisturbances), on the slim
// demo model with stub policies (zero means): before every mj_step
// xfrc_applied is cleared and the hook called with the tick since the reset;
// its wrench moves the body; without a hook nothing is applied.
const SLIM = new URL('physics.mjb.gz', ASSETS);
const SLIM_SKIP = [SLIM, DEMO, REFERENCE].some((u) => !existsSync(fileURLToPath(u)))
  ? `missing ${fileURLToPath(ASSETS)}: run python -m duet_export` : false;

describe('SimCore disturbance hook', { skip: SLIM_SKIP }, () => {
  let mujoco, model, data, core;

  before(async () => {
    mujoco = await loadMujoco();
    ({ model, data } = modelFromBytes(mujoco, await fetchBytes(SLIM), 'physics.mjb'));
    const demo = await loadJSON(DEMO);
    const reference = await loadContainer(REFERENCE);
    const zeros = (n) => ({ run: async () => new Float32Array(n) });
    core = new SimCore({
      mujoco, model, data, cfg: demo, reference,
      policies: { human: zeros(demo.sim.human.action.size), osl: zeros(demo.sim.osl.action.size) },
    });
  });

  test('clears xfrc_applied, then calls the hook before every mj_step', async () => {
    const PELVIS = 92, PUSH = 300; // N along +x, at the pelvis's center of mass
    const calls = [];
    core.reset();
    core.views().xfrc_applied.fill(7); // stale garbage: cleared before the first tick
    core.disturb = (m, d, tick, v) => {
      assert.equal(m, model);
      assert.equal(d, data);
      assert.ok(v.xfrc_applied.every((x) => x === 0), `cleared before tick ${tick}`);
      calls.push(tick);
      v.xfrc_applied[6 * PELVIS] += PUSH;
    };
    const pushed = [];
    for (let k = 0; k < 20; k++) {
      await core.step();
      pushed.push(core.views().qpos[0]);
    }
    assert.deepEqual(calls, Array.from({ length: 20 * core.sim.decimation }, (_, i) => i));
    assert.equal(core.views().xfrc_applied[6 * PELVIS], PUSH, 'the last tick\'s wrench stays visible');
    // The same 20 steps without the hook: nothing applied, the root drifts less in +x.
    core.disturb = null;
    core.reset();
    for (let k = 0; k < 20; k++) await core.step();
    assert.ok(core.views().xfrc_applied.every((x) => x === 0), 'no hook: nothing applied');
    const shift = pushed.at(-1) - core.views().qpos[0];
    assert.ok(shift > 0.01, `${PUSH} N for 0.2 s moved the root ${shift.toFixed(4)} m along +x`);
  });

  after(() => {
    core = null;
    data?.delete();
    model?.delete();
  });
});
