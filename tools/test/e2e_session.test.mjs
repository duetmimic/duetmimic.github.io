// End-to-end check of demo/js/session.js in Node (node --test), on the real
// exported assets: DemoSession loads MuJoCo, physics.mjb.gz, the reference and
// both policies exactly as the browser worker does, then drives a scripted key
// schedule of 26 s of simulated time, stepped as fast as possible (no real-time
// pacing):
//   idle 1 s, W 4 s, W+A 3 s, W 2 s, D 2 s, S 3 s, idle 2 s,
//   Digit5 then W 3 s, Digit1 then W 2 s, A 2 s, idle 2 s.
// Checks: no fall; the root walks forward along its heading during W, at
// least 0.8x the commanded speed x time, and backward during S, at least
// BACKWARD_FRACTION of it. A walk that starts from a standstill (after
// standing or turning in place) is measured after its first 1.5 s: getting
// from a standstill to walking pace takes 1-1.5 s, and bundle B, backing up
// after the schedule's walk and turn in place, hardly moves in its first
// second. It turns left during W+A and A and right during D. A forward
// restart after a stop that misses the pace check is reported as an incident
// and at most one is tolerated: restarting after backing up occasionally
// hesitates for 2-3 s, a limit of the policy itself (fp32 too) that command
// shaping does not remove. The first walk after a reset is gated in
// DemoSession and must meet it. Prints per-step timings and the achievable
// real-time factor.
//
// Also: the real-time pacer (fake clock); the first walk after a reset (W,
// W+A, W+D and Shift+W pressed at FIRST_WALK_DELAYS after a reset, and S,
// S+A and S+D at level 4 at FIRST_BACK_DELAYS: no fall within
// FIRST_WALK_SECONDS, and the start gate releases each one inside
// SHAPING.startWindow, backing up inside SHAPING.backStartWindow, within
// SHAPING.startWait); command shaping (W while
// backing up stands first, the HUD says so, and the forward walk is served
// within SHAPING.maxHold); a fixed-seed key-masher (the human key model of the
// shaping benchmark: holds of 0.4-4 s, taps, steering while walking, direct
// reversals, stops, level changes and Shift) whose falls must stay within the
// measured bound; and the fall -> relaxed physics -> reset sequence. Loading:
// demo.json "versions" match the asset files (cache busting), and a missing
// asset fails create() at once, naming the file, instead of after the other
// downloads.
//
// Dragging a body part (DemoSession.setPerturb, DragSpring): a 0.35 m pull to
// the walker's left for 0.8 s, on the pelvis while standing and on the chest
// while walking, moves the pelvis toward the target against an identical
// unpulled run (and the standing walker recovers); xfrc_applied carries the
// pull on the grabbed body only while a drag is active and is zero after a
// release and after a reset; the force caps hold; the pull goes on through a
// fall's relaxed collapse and the reset after it ends the drag with
// 'perturb-end' {reason: 'fall-reset'} (a user reset: 'reset'); updates still
// on their way for an ended drag are ignored; invalid requests are rejected
// with 'perturb-end' {reason: 'error'}.
//
// Inference: WebGPU through Dawn (the 'webgpu' package) when it finds a
// hardware adapter, else the CPU. E2E_BACKEND=cpu|webgpu forces one.
//
//   cd tools && nice -n 19 node --test test/e2e_session.test.mjs
// E2E_MASH_SECONDS sets the masher's seconds per seed (default 30 on WebGPU,
// 15 on the CPU).

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs, { existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  DRAG, DemoSession, Pacer, SHAPING, STARTING, STOPPING, clipLabel, speedText,
} from '../../demo/js/session.js';
import { ASSETS } from './paths.mjs';

const NEEDED = ['demo.json', 'physics.mjb.gz', 'reference.bin.gz', 'human.bin.gz', 'osl.bin.gz', 'visual.bin.gz'];
const missing = NEEDED.filter((f) => !existsSync(fileURLToPath(new URL(f, ASSETS))));
const SKIP = missing.length ? `missing ${fileURLToPath(ASSETS)}{${missing.join(',')}}: run python -m duet_export` : false;

const SCHEDULE = [
  { name: 'idle', keys: [], seconds: 1 },
  { name: 'W', keys: ['KeyW'], seconds: 4, forward: true },
  { name: 'W+A', keys: ['KeyW', 'KeyA'], seconds: 3, turns: 'left' },
  { name: 'W', keys: ['KeyW'], seconds: 2, forward: true },
  { name: 'D', keys: ['KeyD'], seconds: 2, turns: 'right' },
  { name: 'S', keys: ['KeyS'], seconds: 3, backward: true },
  { name: 'idle', keys: [], seconds: 2 },
  { name: 'Digit5, W', press: 'Digit5', keys: ['KeyW'], seconds: 3, forward: true },
  { name: 'Digit1, W', press: 'Digit1', keys: ['KeyW'], seconds: 2, forward: true },
  { name: 'A', keys: ['KeyA'], seconds: 2, turns: 'left' },
  { name: 'idle', keys: [], seconds: 2 },
];
const FORWARD_FRACTION = 0.8; // of commanded speed x time, during W
const STARTUP_SECONDS = 1.5; // not counted when a walk starts from a standstill
const TURN_FRACTION = 0.5; // of commanded yaw rate x time, during W+A, A and D
// Of commanded speed x time, during S once under way. Bundle B (2026-11-18),
// S held at level 3 (-0.6 m/s) from a standstill, 1.5-3 s after the key:
// 0.72-0.87x (as the first walk after a reset, after turning in place either
// way, after a stop; WebGPU). The schedule's S follows a walk and a turn in
// place, which leave the body drifting forward: it backed 0.62 m in 3 s
// (0.35x), 0.54-0.55 m of it (0.60-0.61x) after the first 1.5 s (WebGPU and
// CPU).
const BACKWARD_FRACTION = 0.5;
const MAX_HESITANT_RESTARTS = 1; // restarts after a stop that miss the pace check

// The first walk after a reset: these keys, pressed FIRST_WALK_DELAYS (s)
// after a reset, must walk FIRST_WALK_SECONDS without a fall. With the start
// window [0.3, 0.5] they start at idle phase 0.3 (as does any press in the
// first 0.15 s), at 0.4 (pressed at 0.2 s, at once) and at 0.3 one idle cycle
// later (pressed at 0.3 s, held 0.35 s): three distinct starts per key set.
const FIRST_WALK_KEYS = [['KeyW'], ['KeyW', 'KeyA'], ['KeyW', 'KeyD'], ['ShiftLeft', 'KeyW']];
const FIRST_WALK_DELAYS = [0, 0.2, 0.3];
const FIRST_WALK_SECONDS = 4;
// Backing up first, at level 4 (-0.7 m/s, as level 5): pressed 0.22 s and
// 0.25 s after a reset, START_WINDOW would start these at once, at idle phase
// 0.44 and 0.5, where S+A fell (and S at 0.5); SHAPING.backStartWindow holds
// them until 0.3 one idle cycle later (0.43 and 0.4 s). Pressed at once, they
// start at 0.3.
const FIRST_BACK_KEYS = [['KeyS'], ['KeyS', 'KeyA'], ['KeyS', 'KeyD']];
const FIRST_BACK_DELAYS = [0, 0.22, 0.25];
const FIRST_BACK_LEVEL = 3; // 0-based: level 4

// Key-masher: the seeds and the falls allowed in them. Bundle B with its
// command shaping fell 5.7 times per 10 min of this human model ("casual",
// WebGPU, 140 seeds x 30 s pooled over the 2026-11-18 retune checks: 40 falls
// in 70 min; the "stress" profile 7.8: 51 in 65 min; R2's benchmark: 2.7
// without command shaping); at that rate 60 s expect 0.57 falls, and 4 or more
// would happen with probability 0.3 %. On WebGPU seeds 0 and 1 fall twice,
// both times in seed 0.
const MASH_SEEDS = [0, 1];
const MASH_MAX_FALLS = 3;

// Dragging: rendered geoms to grab (model ids, demo.visual.geom_ids), each by
// its origin (local [0, 0, 0]), and the pull of the scripted drags.
const PELVIS = 338; // pelvis_r mesh, body pelvis
const CHEST = 21; // the rib cage, body torso
const HAND = 154; // third metacarpal, right hand
const PULL = { offset: 0.35, hold: 0.8, ramp: 0.1, after: 2.0 }; // m to the left, s, s, s
const PULL_MIN_SHIFT = 0.15; // m: the pelvis's lead over the unpulled run at the release

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const pct = (values, q) => {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
const mean = (values) => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length);

// The shaping benchmark's human key model ("casual" profile). Events
// {t, keys} (the held set from t on) and {t, press} (a level key); a seed's
// first n seconds are the same for any total.
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mashTimeline(seed, total) {
  const rnd = mulberry32(0x9E3779B1 ^ (seed * 7919 + 13));
  const U = (a, b) => a + (b - a) * rnd();
  const logU = (a, b) => Math.exp(U(Math.log(a), Math.log(b)));
  const pick = (arr) => arr[Math.floor(rnd() * arr.length)];
  const ev = [{ t: 0, keys: [] }];
  let t = U(0.3, 1.5);
  let dir = 0; // direction of the last walk (+1 W, -1 S)
  const set = (at, keys) => ev.push({ t: at, keys: [...keys] });
  while (t < total) {
    if (rnd() < 0.12) ev.push({ t, press: pick(['Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'KeyQ', 'KeyE']) });
    const fast = rnd() < 0.15 ? ['ShiftLeft'] : [];
    const r = rnd();
    let T;
    if (r < 0.26) { // walk forward, maybe steering on the way
      T = logU(0.4, 4);
      set(t, ['KeyW', ...fast]);
      if (rnd() < 0.45 && T > 0.8) {
        const t0 = t + U(0.2, T - 0.4), d = Math.min(logU(0.2, 1.5), t + T - t0);
        set(t0, ['KeyW', ...fast, pick(['KeyA', 'KeyD'])]);
        set(t0 + d, ['KeyW', ...fast]);
      }
      dir = 1;
    } else if (r < 0.36) { // arc
      T = logU(0.3, 3);
      set(t, ['KeyW', ...fast, pick(['KeyA', 'KeyD'])]);
      dir = 1;
    } else if (r < 0.48) { // backward, maybe steering
      T = logU(0.4, 3);
      set(t, ['KeyS']);
      if (rnd() < 0.35 && T > 0.8) {
        const t0 = t + U(0.2, T - 0.4), d = Math.min(logU(0.2, 1.5), t + T - t0);
        set(t0, ['KeyS', pick(['KeyA', 'KeyD'])]);
        set(t0 + d, ['KeyS']);
      }
      dir = -1;
    } else if (r < 0.60) { // turn in place
      T = logU(0.3, 2.5);
      set(t, [pick(['KeyA', 'KeyD'])]);
    } else if (r < 0.70) { // tap
      T = logU(0.05, 0.3);
      const k = rnd() < 0.7 ? 'KeyW' : 'KeyS';
      set(t, [k]);
      t += T;
      set(t, []);
      T = logU(0.2, 1.0);
      dir = k === 'KeyW' ? 1 : -1;
    } else if (r < 0.85) { // stop
      T = logU(0.2, 3);
      set(t, []);
    } else if (r < 0.95) { // straight into the other direction
      T = logU(0.3, 3);
      const k = dir >= 0 ? 'KeyS' : 'KeyW';
      set(t, [k, ...(k === 'KeyW' ? fast : [])]);
      dir = k === 'KeyW' ? 1 : -1;
    } else { // jiggle
      const keys = pick([['KeyA', 'KeyD'], ['KeyW', 'KeyS'], ['KeyA', 'KeyW']]);
      const n = 2 + Math.floor(rnd() * 3);
      let tt = t;
      for (let i = 0; i < n; i++) { set(tt, [keys[i % 2]]); tt += logU(0.1, 0.35); }
      T = tt - t;
    }
    t += T;
    if (rnd() < 0.4) { set(t, []); t += logU(0.03, 0.3); } // a finger lift between moves
  }
  ev.sort((a, b) => a.t - b.t);
  return ev.filter((e) => e.t < total);
}

// The world position of a geom's origin now.
function geomPoint(session, geom) {
  const x = session.core.views().geom_xpos;
  return [x[3 * geom], x[3 * geom + 1], x[3 * geom + 2]];
}

// The xfrc_applied rows that are not zero: [[body, |force|, |torque|], ...].
function wrenches(session) {
  const x = session.core.views().xfrc_applied, rows = [];
  for (let b = 0; 6 * b < x.length; b++) {
    const r = x.subarray(6 * b, 6 * b + 6);
    if (r.some((value) => value !== 0)) rows.push([b, Math.hypot(r[0], r[1], r[2]), Math.hypot(r[3], r[4], r[5])]);
  }
  return rows;
}

// From a reset, `total` control steps with `keys` held, recording the pelvis
// (the observation anchor), the grabbed geom's origin and the heading after
// every step. With pull = {from, hold, ramp, offset (world [x, y, z]), path
// (an unpulled run's geom track)}, drags `geom` from step `from` for `hold`
// steps toward path + offset (ramped in over `ramp` steps), then releases.
async function dragRun(session, { keys, total, geom, pull = null }) {
  session.reset();
  session.level = session.ui.default_level;
  session.setKeys(keys);
  const anchor = session.demo.sim.body_ids[0];
  const pelvis = new Float64Array(3 * (total + 1)), point = new Float64Array(3 * (total + 1));
  const heading = new Float64Array(total + 1);
  const record = (k) => {
    const xpos = session.core.views().xpos;
    for (let i = 0; i < 3; i++) pelvis[3 * k + i] = xpos[3 * anchor + i];
    point.set(geomPoint(session, geom), 3 * k);
    heading[k] = session.pose().heading;
  };
  let fellAt = -1, peak = 0;
  const off = session.on('fall', () => { if (fellAt < 0) fellAt = session.steps; });
  const steps0 = session.steps;
  record(0);
  for (let k = 0; k < total; k++) {
    if (pull && fellAt >= 0) {
      // fallen: the reset after it ended the drag, the caller reports it
    } else if (pull && k >= pull.from && k < pull.from + pull.hold) {
      const a = Math.min(1, (k - pull.from + 1) / pull.ramp);
      const target = [0, 1, 2].map((i) => pull.path[3 * k + i] + a * pull.offset[i]);
      assert.equal(session.setPerturb({ geom, local: [0, 0, 0], target }), k === pull.from ? 'start' : 'update');
    } else if (pull && k === pull.from + pull.hold) {
      session.clearPerturb();
    }
    await session.step();
    if (session.drag.active) peak = Math.max(peak, session.drag.force);
    record(k + 1);
  }
  off();
  session.clearPerturb();
  return { pelvis, point, heading, peak, fellAt: fellAt < 0 ? -1 : fellAt - steps0 };
}

// WebGPU (Dawn) with a hardware adapter, else the CPU.
async function pickBackend() {
  const want = process.env.E2E_BACKEND || 'auto';
  if (want === 'cpu') return { backend: 'cpu', why: 'E2E_BACKEND=cpu' };
  try {
    const { create } = await import('webgpu');
    const gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    const software = adapter && adapter.info && adapter.info.isFallbackAdapter;
    if (adapter && (!software || want === 'webgpu')) return { backend: 'webgpu', gpu, why: 'Dawn adapter' };
    if (want === 'webgpu') throw new Error('Dawn found no adapter');
    return { backend: 'cpu', why: software ? 'Dawn offers only a software adapter' : 'Dawn found no adapter' };
  } catch (error) {
    if (want === 'webgpu') throw error;
    return { backend: 'cpu', why: `no WebGPU in Node (${error.message})` };
  }
}

// Cache busting: the exporter writes demo.json "versions", the first 12 hex
// digits of each referenced asset's sha256, and DemoSession fetches
// `${file}?v=${version}`. A file changed by hand, or by a section exported
// with an older exporter, would be served under a stale version.
test('demo.json versions match the asset files it references', { skip: SKIP }, () => {
  const demo = JSON.parse(fs.readFileSync(fileURLToPath(new URL('demo.json', ASSETS)), 'utf8'));
  const files = new Set();
  const walk = (node) => {
    if (Array.isArray(node)) node.forEach(walk);
    else if (node && typeof node === 'object') {
      for (const [key, value] of Object.entries(node)) {
        if (key === 'versions') continue;
        if (key === 'file' && typeof value === 'string') files.add(value);
        else walk(value);
      }
    }
  };
  walk(demo);
  assert.ok(demo.versions, 'demo.json has no "versions": re-run python -m duet_export');
  assert.deepEqual(Object.keys(demo.versions).sort(), [...files].sort(), 'one version per referenced file');
  for (const file of files) {
    const digest = createHash('sha256').update(fs.readFileSync(fileURLToPath(new URL(file, ASSETS)))).digest('hex');
    assert.equal(demo.versions[file], digest.slice(0, 12), `${file} changed after the export wrote its version: re-run python -m duet_export`);
  }
});

describe('DemoSession end to end', { skip: SKIP }, () => {
  let session = null;
  let picked = null;
  let createMs = 0;
  const progress = [];

  before(async () => {
    picked = await pickBackend();
    const t0 = performance.now();
    session = await DemoSession.create({
      assetsBase: ASSETS,
      backend: picked.backend,
      gpu: picked.gpu,
      onProgress: (p) => progress.push({ loaded: p.loaded, total: p.total, stage: p.stage }),
    });
    createMs = performance.now() - t0;
  });

  after(() => {
    if (session) session.dispose();
    session = null;
    if (picked) picked.gpu = null; // dawn.node keeps the process alive while referenced
  });

  test('loads with byte-weighted progress', (t) => {
    const demo = session.demo;
    const listed = demo.physics.bytes + demo.reference.bytes + demo.policies.human.bytes + demo.policies.osl.bytes;
    const last = progress[progress.length - 1];
    assert.ok(progress.length >= 2, 'progress was reported');
    assert.equal(last.loaded, last.total, 'progress ends complete');
    assert.ok(last.total >= listed && last.total < listed + 4e6, `total ${last.total} = listed ${listed} + the wasm`);
    for (let i = 1; i < progress.length; i++) assert.ok(progress[i].loaded >= progress[i - 1].loaded, 'progress never goes back');
    assert.equal(session.backend, picked.backend === 'webgpu' ? 'WebGPU' : 'CPU');
    const s = session.status();
    assert.equal(s.command, 'Stand');
    assert.equal(s.clip, 'idle');
    assert.equal(s.speed, speedText(session.command(), session.ui, session.level));
    t.diagnostic(`create ${createMs.toFixed(0)} ms (${picked.backend}: ${picked.why}; ` +
      `adapter ${s.adapter || '-'}; ${session.policies.human.info.reason}); load times ${JSON.stringify(session.loadTimes, (k, v) => (typeof v === 'number' ? Math.round(v) : v))}`);
  });

  test('pacer: steps owed, debt cap, pause without catch-up, smoothed RTF', () => {
    let now = 1000;
    const pacer = new Pacer({ dt: 0.01, maxLag: 0.25, window: 2, clock: () => now });
    assert.equal(pacer.due(), 0);
    now = 1035;
    assert.equal(pacer.due(), 3);
    pacer.tick(3);
    assert.equal(pacer.due(), 0);
    assert.ok(Math.abs(pacer.msUntilNext() - 5) < 1e-9);
    // 0.4 s behind: the debt is dropped, one step is due, the clock re-anchors.
    now = 1435;
    assert.equal(pacer.due(), 1);
    assert.equal(pacer.dropped, 39);
    pacer.tick(1);
    assert.equal(pacer.due(), 0);
    // Real time for 2 s: RTF 1; then half speed: RTF tends to 0.5.
    pacer.restart(now = 0);
    for (let k = 1; k <= 200; k++) { now = 10 * k; pacer.tick(pacer.due()); pacer.sample(); }
    assert.ok(Math.abs(pacer.rtf - 1) < 0.02, `rtf ${pacer.rtf}`);
    for (let k = 1; k <= 300; k++) { now += 20; pacer.tick(1); pacer.sample(); }
    assert.ok(Math.abs(pacer.rtf - 0.5) < 0.02, `rtf ${pacer.rtf}`);
  });

  test('advance() runs the owed steps; a pause stops the clock', async () => {
    const real = { clock: session.clock, pacer: session.pacer };
    let now = 0;
    session.clock = () => now;
    session.pacer = new Pacer({ dt: session.dt, clock: session.clock });
    try {
      session.reset();
      session.pause(true);
      session.pause(false); // re-anchors the clock at 0
      now = 50;
      assert.equal(await session.advance({ budgetMs: 1e9 }), 5);
      assert.equal(await session.advance({ budgetMs: 1e9 }), 0);
      session.pause(true);
      now = 10050; // 10 s paused
      assert.equal(await session.advance({ budgetMs: 1e9 }), 0);
      assert.equal(session.msUntilNextStep(), Infinity);
      session.pause(false);
      now = 10070;
      assert.equal(await session.advance({ budgetMs: 1e9 }), 2, 'no catch-up burst after a pause');
      now = 11070; // 1 s stall: more than 0.25 s behind, the debt goes
      assert.equal(await session.advance({ budgetMs: 1e9 }), 1);
      assert.ok(session.pacer.dropped >= 99);
    } finally {
      session.clock = real.clock;
      session.pacer = real.pacer;
      session.reset();
    }
  });

  test('scripted WASD schedule, stepped as fast as possible', async (t) => {
    session.reset();
    session.setKeys([]);
    session.level = session.ui.default_level;
    const falls = [];
    const offFall = session.on('fall', (e) => falls.push(e));
    const timing = { policy: [], physics: [], observe: [], step: [] };
    const rows = [];
    const started = performance.now();
    let simSeconds = 0;
    let walked = false; // a gait clip since the last reset
    let pose = session.pose();
    for (const seg of SCHEDULE) {
      if (seg.press) session.handleKey(seg.press);
      session.setKeys(seg.keys);
      const command = session.command();
      const steps = Math.round(seg.seconds / session.dt);
      const resetsBefore = session.resets;
      const fallsBefore = falls.length;
      let progress = 0;
      let turn = 0;
      const trace = new Float64Array(steps + 1); // progress after k steps
      const clips = new Set();
      const walkedBefore = walked;
      for (let k = 0; k < steps; k++) {
        const t0 = performance.now();
        const info = await session.step();
        timing.step.push(performance.now() - t0);
        if (!info.failed) {
          timing.policy.push(info.timing.policy);
          timing.physics.push(info.timing.physics);
          timing.observe.push(info.timing.observe);
        }
        clips.add(clipLabel(session.clips[info.clip]));
        if (session.resets !== resetsBefore) walked = false;
        else if (!session.clips[info.clip].idle) walked = true;
        const next = session.pose();
        if (session.resets === resetsBefore) {
          progress += (next.x - pose.x) * Math.cos(pose.heading) + (next.y - pose.y) * Math.sin(pose.heading);
          turn += wrap(next.heading - pose.heading);
        }
        pose = next;
        trace[k + 1] = progress;
      }
      simSeconds += steps * session.dt;
      const [vx, , wz] = command.twist;
      // From a standstill: after standing or turning in place (no walking speed to carry over).
      const fromRest = rows.length === 0 || rows[rows.length - 1].vx === 0;
      const skip = fromRest ? Math.round(STARTUP_SECONDS / session.dt) : 0;
      rows.push({
        ...seg, command, progress, turn, vx, wz, fromRest, restart: fromRest && walkedBefore,
        counted: progress - trace[skip], countedSeconds: seg.seconds - skip * session.dt,
        fell: falls.length > fallsBefore || session.resets !== resetsBefore,
        clips: [...clips].join(' > '),
      });
    }
    offFall();
    const wall = (performance.now() - started) / 1000;

    for (const r of rows) {
      const steady = (r.forward || r.backward) && r.fromRest
        ? `; after ${STARTUP_SECONDS} s ${r.counted.toFixed(2)} m (${(r.counted / (r.vx * r.countedSeconds)).toFixed(2)}x)` : '';
      t.diagnostic(`${r.name.padEnd(10)} ${r.seconds} s  cmd vx ${r.vx.toFixed(2)} wz ${r.wz.toFixed(2)} (${r.command.label}): ` +
        `forward ${r.progress.toFixed(2)} m (${r.vx ? (r.progress / (r.vx * r.seconds)).toFixed(2) : '-'}x${steady}), ` +
        `turn ${r.turn.toFixed(2)} rad (${r.wz ? (r.turn / (r.wz * r.seconds)).toFixed(2) : '-'}x)${r.fell ? '  FELL' : ''}  clips ${r.clips}`);
    }
    const ms = (values) => `mean ${mean(values).toFixed(2)} p50 ${pct(values, 0.5).toFixed(2)} p95 ${pct(values, 0.95).toFixed(2)} ms`;
    t.diagnostic(`per step: inference ${ms(timing.policy)}; physics ${ms(timing.physics)}; observations ${ms(timing.observe)}; whole step ${ms(timing.step)}`);
    t.diagnostic(`${simSeconds.toFixed(1)} s simulated in ${wall.toFixed(1)} s: achievable RTF ${(simSeconds / wall).toFixed(2)} (${session.backend})`);
    for (const f of falls) t.diagnostic(`FALL at ${f.time.toFixed(2)} s after the reset: ${f.failure}`);
    assert.equal(falls.length, 0, `${falls.length} falls on the schedule`);
    const incidents = [];
    for (const r of rows) {
      if (r.fell) continue; // a reset teleports the root; the fall is reported above
      const label = `${r.name} (${r.seconds} s)`;
      if (r.forward) {
        const want = FORWARD_FRACTION * r.vx * r.countedSeconds;
        const what = r.fromRest ? `after the first ${STARTUP_SECONDS} s` : 'over the segment';
        const message = `${label}: forward ${r.counted.toFixed(2)} m ${what} < ${want.toFixed(2)} m`;
        if (r.counted < want && r.restart) incidents.push(`hesitant restart after a stop: ${message}`);
        else assert.ok(r.counted >= want, message);
      }
      if (r.backward) {
        const want = BACKWARD_FRACTION * r.vx * r.countedSeconds; // vx < 0
        const what = r.fromRest ? `after the first ${STARTUP_SECONDS} s` : 'over the segment';
        assert.ok(r.counted <= want, `${label}: forward ${r.counted.toFixed(2)} m ${what}, want <= ${want.toFixed(2)} m`);
      }
      if (r.turns) {
        // same sign as the command, at least TURN_FRACTION of it
        assert.equal(Math.sign(r.wz), r.turns === 'left' ? 1 : -1, `${label}: commanded yaw rate ${r.wz}`);
        const want = TURN_FRACTION * r.wz * r.seconds;
        if (r.wz > 0) assert.ok(r.turn >= want, `${label}: turned ${r.turn.toFixed(2)} rad, want >= ${want.toFixed(2)}`);
        else assert.ok(r.turn <= want, `${label}: turned ${r.turn.toFixed(2)} rad, want <= ${want.toFixed(2)}`);
      }
    }
    for (const incident of incidents) t.diagnostic(`INCIDENT ${incident}`);
    assert.ok(incidents.length <= MAX_HESITANT_RESTARTS, `${incidents.length} incidents: ${incidents.join('; ')}`);
    // Commands followed the keys and levels.
    assert.equal(rows[2].command.action, 'forward_left');
    assert.equal(rows[4].command.action, 'turn_right');
    assert.equal(rows[5].command.action, 'backward');
    assert.equal(rows[7].command.level, 4);
    assert.equal(rows[8].command.level, 0);
  });

  // The first walk after a reset: the touch pad's ▲ holds W from Start, and a
  // walk held through a fall resumes the moment the walker resets. Out of the
  // reset stance bundle B falls when its first stride starts early in the idle
  // clip, so the start gate holds the first walk until the idle phase is in
  // SHAPING.startWindow (see session.js).
  // Backing up has a window of its own, SHAPING.backStartWindow.
  for (const { label, keySets, delays, level, window } of [
    { label: 'W, W+A, W+D, Shift+W', keySets: FIRST_WALK_KEYS, delays: FIRST_WALK_DELAYS, level: null, window: 'startWindow' },
    { label: 'S, S+A, S+D at level 4', keySets: FIRST_BACK_KEYS, delays: FIRST_BACK_DELAYS, level: FIRST_BACK_LEVEL, window: 'backStartWindow' },
  ]) test(`first walk after a reset: ${label} pressed at several delays never fall`, async (t) => {
    const maxWait = Math.round(SHAPING.startWait / session.dt);
    const [lo, hi] = SHAPING[window];
    const fell = [];
    for (const keys of keySets) {
      const cells = [];
      for (const delay of delays) {
        session.reset();
        session.setKeys([]);
        session.level = level ?? session.ui.default_level;
        const press = Math.round(delay / session.dt);
        for (let k = 0; k < press; k++) await session.step();
        session.setKeys(keys);
        let held = null, phase = null, fall = null;
        const off = session.on('fall', (e) => { if (fall === null) fall = e.time; });
        for (let k = 0; k < Math.round(FIRST_WALK_SECONDS / session.dt) && fall === null; k++) {
          const before = session.core.reference.phase; // the phase this step's request sees
          await session.step();
          if (held === null && !session.core.reference.idle) { held = k; phase = before; }
        }
        off();
        const name = `${keys.join('+')} pressed ${delay} s after the reset`;
        if (fall !== null) {
          cells.push(`${delay} s: FELL`);
          fell.push(`${name}: fell ${(fall - delay).toFixed(2)} s after the press`);
          continue;
        }
        cells.push(`${delay} s: held ${(held * session.dt).toFixed(2)} s, phase ${phase.toFixed(2)}`);
        assert.ok(held !== null && held <= maxWait, `${name}: held back ${held} steps (at most ${maxWait})`);
        assert.ok(phase >= lo && phase <= hi, `${name}: started at idle phase ${phase}, outside [${lo}, ${hi}]`);
      }
      t.diagnostic(`first walk ${keys.join('+')}: ${cells.join('; ')}`);
    }
    assert.deepEqual(fell, [], `${fell.length} first walks fell`);
    session.reset();
    session.setKeys([]);
    session.level = session.ui.default_level;
  });

  test('shaping: W while backing up stands first, says so, and starts within maxHold', async (t) => {
    session.reset();
    session.level = 4; // backward -0.7 m/s, forward 1.2 m/s
    session.setKeys(['KeyS']);
    for (let k = 0; k < 250; k++) await session.step();
    assert.ok(session.core.reference.twist[0] < 0, 'walking backward');
    session.setKeys(['KeyW']);
    assert.equal(session.status().command, STOPPING, 'the HUD says the body stops first');
    assert.equal(session.status().shaping, 'reverse');
    assert.equal(session.status().moving, true);
    const labels = new Set();
    let served = null;
    const limit = Math.round(SHAPING.maxHold / session.dt) + 2;
    for (let k = 1; k <= limit + 50 && served === null; k++) {
      labels.add(session.status().command);
      await session.step();
      assert.equal(session.fall, null, 'no fall');
      if (session.core.reference.twist[0] > 0) served = k;
    }
    t.diagnostic(`forward served ${served} steps after the key; HUD ${[...labels].join(' > ')}`);
    assert.ok(served !== null && served <= limit, `forward served after ${served} steps (at most ${limit})`);
    assert.ok(labels.has(STOPPING) && labels.has(STARTING), 'Stopping…, then Starting…');
    assert.equal(session.status().command, 'Forward');
    assert.equal(session.status().shaping, null);
    // Backing up is never held back, nor is stopping.
    session.setKeys(['KeyS']);
    assert.equal(session.shaper.decide(session.command()).reason, null);
    session.setKeys([]);
    assert.equal(session.status().command, 'Stand');
    session.reset();
    session.level = session.ui.default_level;
  });

  test('S+A backs up curving to the walker\'s left, S+D to its right (car in reverse)', async (t) => {
    for (const [turnKey, side] of [['KeyA', 1], ['KeyD', -1]]) {
      session.reset();
      session.level = session.ui.default_level;
      session.setKeys(['KeyS', turnKey]);
      const start = session.pose();
      const steps = Math.round(4 / session.dt);
      for (let k = 0; k < steps; k++) {
        await session.step();
        assert.equal(session.fall, null, 'no fall');
      }
      const end = session.pose();
      // Displacement in the start pose's frame: x along its facing, y to its left.
      const dx = end.x - start.x, dy = end.y - start.y;
      const c = Math.cos(start.heading), s = Math.sin(start.heading);
      const back = -(dx * c + dy * s), left = -dx * s + dy * c;
      const yaw = wrap(end.heading - start.heading);
      t.diagnostic(`S+${turnKey.slice(3)}: back ${back.toFixed(2)} m, left ${left.toFixed(2)} m, yaw ${yaw.toFixed(2)} rad`);
      assert.ok(back > 0.8, `walked backward (${back.toFixed(2)} m)`);
      assert.ok(side * left > 0.1, `curved to the ${side > 0 ? 'left' : 'right'} (${left.toFixed(2)} m)`);
      assert.ok(side * yaw < 0, `body yawed the other way (${yaw.toFixed(2)} rad)`);
    }
    session.setKeys([]);
    session.reset();
    session.level = session.ui.default_level;
  });

  test('key-masher: falls within the measured bound, holds within their limits', async (t) => {
    const perSeed = Number(process.env.E2E_MASH_SECONDS || (picked.backend === 'webgpu' ? 30 : 15));
    const steps = Math.round(perSeed / session.dt);
    const maxReverse = Math.round(SHAPING.maxHold / session.dt);
    const maxStart = Math.round(SHAPING.startWait / session.dt);
    const labels = new Set();
    let falls = 0, longestReverse = 0, longestStart = 0, held = { reverse: 0, start: 0 };
    const started = performance.now();
    const off = session.on('fall', () => { falls += 1; });
    try {
      for (const seed of MASH_SEEDS) {
        const events = mashTimeline(seed, perSeed);
        session.reset();
        session.setKeys([]);
        session.level = session.ui.default_level;
        let e = 0, run = { reason: null, n: 0 };
        for (let k = 0; k < steps; k++) {
          while (e < events.length && events[e].t <= k * session.dt + 1e-9) {
            const ev = events[e++];
            if (ev.press) session.handleKey(ev.press);
            else session.setKeys(ev.keys);
          }
          if (k % 10 === 0 && !session.fall) {
            const s = session.status();
            labels.add(s.command);
            if (s.shaping) assert.ok([STARTING, STOPPING].includes(s.command), `HUD ${s.command} while ${s.shaping}`);
            else assert.equal(s.command, session.command().label);
          }
          const falling = !!session.fall;
          await session.step();
          const reason = falling ? null : session.shaped && session.shaped.reason;
          if (reason) held[reason] += 1;
          run = reason && reason === run.reason ? { reason, n: run.n + 1 } : { reason, n: reason ? 1 : 0 };
          if (run.reason === 'reverse') longestReverse = Math.max(longestReverse, run.n);
          if (run.reason === 'start') longestStart = Math.max(longestStart, run.n);
        }
      }
    } finally {
      off();
    }
    const simSeconds = MASH_SEEDS.length * perSeed;
    const wall = (performance.now() - started) / 1000;
    t.diagnostic(`masher: seeds ${MASH_SEEDS.join(',')} x ${perSeed} s (${simSeconds} s in ${wall.toFixed(1)} s wall, ${session.backend}): ` +
      `${falls} falls (${(falls / simSeconds * 600).toFixed(1)} per 10 min); held back: reverse ${(held.reverse * session.dt).toFixed(2)} s ` +
      `(longest ${(longestReverse * session.dt).toFixed(2)} s), start gate ${(held.start * session.dt).toFixed(2)} s ` +
      `(longest ${(longestStart * session.dt).toFixed(2)} s); HUD commands seen: ${[...labels].join(', ')}`);
    assert.ok(falls <= MASH_MAX_FALLS, `${falls} falls in ${simSeconds} s of mashing (at most ${MASH_MAX_FALLS})`);
    assert.ok(longestReverse <= maxReverse, `a forward request was held back ${longestReverse} steps (at most ${maxReverse})`);
    assert.ok(longestStart <= maxStart, `the start gate held ${longestStart} steps (at most ${maxStart})`);
    session.reset();
    session.setKeys([]);
    session.level = session.ui.default_level;
  });

  test('a fall relaxes the body for about 1 s, then resets', async () => {
    session.reset();
    session.setKeys([]);
    const events = [];
    const off = session.on('*', (e) => events.push(e));
    const v = session.core.views();
    v.qvel[0] = 6; // shove the root sideways at 6 m/s
    v.qvel[1] = 6;
    for (let n = 0; n < 300 && !events.some((e) => e.type === 'fall'); n++) await session.step();
    const fall = events.find((e) => e.type === 'fall');
    assert.ok(fall, 'the shove made it fall');
    assert.ok(fall.physics, 'the state stayed finite');
    assert.equal(session.status().falling, true);
    const z0 = session.pose().z;
    let steps = 0;
    while (session.fall) {
      await session.step();
      steps += 1;
      assert.ok(session.extract() !== null || !session.drawable);
      if (steps > 500) break;
    }
    off();
    assert.equal(steps, Math.round(1 / session.dt), 'one second of relaxed physics');
    const reset = events.find((e) => e.type === 'reset');
    assert.ok(reset && reset.reason === 'fall', 'reset after the fall');
    assert.ok(events.indexOf(reset) > events.indexOf(fall));
    assert.equal(session.status().falling, false);
    assert.ok(session.status().time < 1e-9, 'back at t = 0');
    assert.ok(Math.abs(session.pose().z - 0.94) < 0.05, `standing again (root z ${session.pose().z.toFixed(3)}, was ${z0.toFixed(3)})`);
  });

  test('drag: a sideways pull on the pelvis (standing) and the chest (walking) moves the walker toward it', async (t) => {
    const n = (seconds) => Math.round(seconds / session.dt);
    const cases = [
      { name: 'standing, pelvis', keys: [], warm: 1.0, geom: PELVIS, recovers: true },
      { name: 'walking, chest', keys: ['KeyW'], warm: 3.0, geom: CHEST, recovers: false },
    ];
    for (const c of cases) {
      const from = n(c.warm), hold = n(PULL.hold), total = from + hold + n(PULL.after);
      const free = await dragRun(session, { keys: c.keys, total, geom: c.geom });
      assert.equal(free.fellAt, -1, `${c.name}: the unpulled run fell`);
      const h = free.heading[from];
      const left = [-Math.sin(h), Math.cos(h), 0];
      const pull = { from, hold, ramp: n(PULL.ramp), offset: left.map((x) => PULL.offset * x), path: free.point };
      const pulled = await dragRun(session, { keys: c.keys, total, geom: c.geom, pull });
      // The pelvis's lead to the left over the unpulled run.
      const lead = (k) => [0, 1, 2].reduce((s, i) => s + (pulled.pelvis[3 * k + i] - free.pelvis[3 * k + i]) * left[i], 0);
      const release = from + hold;
      let most = -Infinity;
      for (let k = from; k <= release; k++) most = Math.max(most, lead(k));
      const fell = pulled.fellAt >= 0 ? `fell ${((pulled.fellAt - release) * session.dt).toFixed(2)} s after the release` : `recovered (no fall in ${PULL.after} s)`;
      t.diagnostic(`${c.name}: ${PULL.offset} m for ${PULL.hold} s, peak ${pulled.peak.toFixed(0)} N; pelvis ${lead(release).toFixed(3)} m `
        + `toward the target at the release (most ${most.toFixed(3)} m); ${fell}`);
      assert.ok(pulled.fellAt < 0 || pulled.fellAt > release, `${c.name}: fell during the pull`);
      assert.ok(lead(release) > PULL_MIN_SHIFT, `${c.name}: the pelvis moved ${lead(release).toFixed(3)} m toward the target (want > ${PULL_MIN_SHIFT})`);
      assert.ok(pulled.peak > 0 && pulled.peak <= DRAG.maxForce, `${c.name}: peak force ${pulled.peak}`);
      if (c.recovers) assert.equal(pulled.fellAt, -1, `${c.name}: a moderate pull should not topple it (${fell})`);
    }
    session.setKeys([]);
    session.reset();
  });

  test('drag: xfrc_applied holds the pull only during a drag; the force caps hold', async () => {
    session.reset();
    session.setKeys([]);
    const body = session.drag.geomBody[PELVIS];
    const p = geomPoint(session, PELVIS);
    assert.deepEqual(wrenches(session), [], 'nothing applied after a reset');
    assert.equal(session.setPerturb({ geom: PELVIS, local: [0, 0, 0], target: [p[0], p[1] + 0.3, p[2]], id: 7 }), 'start');
    let s = session.status().perturb;
    assert.equal(s.active, true);
    assert.equal(s.geom, PELVIS);
    assert.equal(s.body, body);
    assert.equal(s.id, 7);
    // The spring: k = vis.map.stiffness (100), m = the effective mass, counted at most maxMass.
    const m = session.drag.effective;
    assert.ok(m > 15 && m < 40, `effective mass at the pelvis ${m.toFixed(2)} kg`);
    assert.ok(Math.abs(session.drag.km - session.model.vis.map.stiffness * Math.min(m, DRAG.maxMass)) < 1e-9);
    for (let k = 0; k < 20; k++) {
      await session.step();
      const rows = wrenches(session);
      assert.equal(rows.length, 1, 'one body pulled');
      assert.equal(rows[0][0], body, 'the grabbed body');
      assert.ok(rows[0][1] > 0 && Math.abs(rows[0][1] - session.drag.force) < 1e-9, `force ${rows[0][1]} vs ${session.drag.force}`);
    }
    s = session.status().perturb;
    assert.ok(s.force > 0 && s.force <= s.maxForce && s.stretch > 0, JSON.stringify(s));
    assert.equal(session.clearPerturb(), true, 'released');
    assert.equal(session.status().perturb.active, false);
    await session.step();
    assert.deepEqual(wrenches(session), [], 'nothing applied after the release');

    // A wild drag, 5 m away: |F| stops at maxForce.
    const q = geomPoint(session, PELVIS);
    assert.equal(session.setPerturb({ geom: PELVIS, local: [0, 0, 0], target: [q[0], q[1] + 5, q[2]] }), 'start');
    for (let k = 0; k < 5; k++) {
      await session.step();
      assert.ok(wrenches(session)[0][1] <= DRAG.maxForce + 1e-9);
    }
    assert.equal(session.drag.peak, DRAG.maxForce, 'saturated at maxForce');
    assert.equal(session.status().perturb.force, DRAG.maxForce);
    // A hand, 3 m away: |F| stops at maxAccel times its effective mass.
    const r = geomPoint(session, HAND);
    assert.equal(session.setPerturb({ geom: HAND, local: [0, 0, 0], target: [r[0], r[1] + 3, r[2]] }), 'start', 'another geom: a new grab');
    const hand = session.drag.effective, limit = DRAG.maxAccel * hand;
    assert.ok(hand > 0.3 && hand < 2 && limit < DRAG.maxForce, `effective mass at the hand ${hand.toFixed(2)} kg`);
    assert.equal(session.status().perturb.maxForce, limit);
    for (let k = 0; k < 5; k++) {
      await session.step();
      const rows = wrenches(session);
      assert.deepEqual(rows.map((row) => row[0]), [session.drag.geomBody[HAND]]);
      assert.ok(rows[0][1] <= limit + 1e-9);
    }
    assert.ok(Math.abs(session.drag.peak - limit) < 1e-9, `hand peak ${session.drag.peak} vs ${limit}`);

    // A reset ends the drag and clears xfrc_applied.
    const events = [];
    const off = session.on('*', (e) => events.push(e));
    session.reset();
    off();
    assert.deepEqual(events.map((e) => [e.type, e.reason]), [['perturb-end', 'reset'], ['reset', 'user']]);
    assert.equal(events[0].geom, HAND);
    assert.equal(session.status().perturb.active, false);
    assert.deepEqual(wrenches(session), [], 'nothing applied after the reset');
    await session.step();
    assert.deepEqual(wrenches(session), []);
    session.reset();
  });

  test('drag: the pull goes on through a fall; the reset after it ends the drag with perturb-end', async () => {
    session.reset();
    session.setKeys([]);
    const events = [];
    const off = session.on('*', (e) => events.push(e));
    const body = session.drag.geomBody[PELVIS];
    const p = geomPoint(session, PELVIS);
    const grab = { geom: PELVIS, local: [0, 0, 0], target: [p[0] + 2, p[1], p[2]] }; // a big pull
    assert.equal(session.setPerturb(grab), 'start');
    for (let n = 0; n < 400 && !session.fall; n++) await session.step();
    assert.ok(session.fall, 'the pull toppled it');
    // While the body goes down (relaxed), the drag still pulls and still follows.
    assert.equal(session.status().perturb.active, true);
    assert.equal(session.setPerturb({ ...grab, target: [p[0] + 2, p[1], p[2] + 0.5] }), 'update');
    await session.step();
    assert.ok(session.fall, 'still falling');
    const rows = wrenches(session);
    assert.ok(rows.length === 1 && rows[0][0] === body && rows[0][1] > 0, 'pulling during the collapse');
    let steps = 0;
    while (session.fall && steps < 500) {
      await session.step();
      steps += 1;
    }
    off();
    const end = events.findIndex((e) => e.type === 'perturb-end');
    const reset = events.findIndex((e) => e.type === 'reset');
    assert.ok(end >= 0 && events[end].reason === 'fall-reset', 'perturb-end {reason: fall-reset}');
    assert.equal(events[end].geom, PELVIS);
    assert.equal(events[end].body, body);
    assert.ok(reset > end && events[reset].reason === 'fall', 'then the reset');
    assert.equal(session.status().perturb.active, false);
    assert.deepEqual(wrenches(session), [], 'nothing applied after the reset');
    // Updates for the ended drag, still on their way, are ignored...
    assert.equal(session.setPerturb(grab), 'stale');
    await session.step();
    assert.deepEqual(wrenches(session), []);
    // ...until the page releases (or grabs elsewhere).
    session.clearPerturb();
    assert.equal(session.setPerturb(grab), 'start');
    session.clearPerturb();
    session.reset();
  });

  test('drag: invalid requests are rejected, end any drag and never touch the physics', async () => {
    session.reset();
    session.setKeys([]);
    const ended = [];
    const off = session.on('perturb-end', (e) => ended.push(e));
    const p = geomPoint(session, PELVIS);
    const good = { geom: PELVIS, local: [0, 0, 0], target: p };
    const drawn = new Set(session.demo.visual.geom_ids);
    const hidden = [...Array(session.model.ngeom).keys()].find((g) => !drawn.has(g)); // e.g. the floor
    const bad = [
      null, 'perturb', {},
      { ...good, geom: hidden }, { ...good, geom: -1 }, { ...good, geom: session.model.ngeom },
      { ...good, geom: 1.5 }, { ...good, geom: String(PELVIS) }, { ...good, geom: undefined },
      { ...good, local: [0, NaN, 0] }, { ...good, local: [0, 0] }, { ...good, local: [1, 0, 0] }, { ...good, local: null },
      { ...good, target: [p[0], Infinity, p[2]] }, { ...good, target: undefined }, { ...good, target: [1e5, 0, 0] },
      { ...good, target: ['1', '2', '3'] },
    ];
    for (const spec of bad) assert.equal(session.setPerturb(spec), 'invalid', JSON.stringify(spec));
    assert.equal(ended.length, bad.length, 'each one tells the page to drop its drag');
    for (const e of ended) assert.ok(e.reason === 'error' && /^perturb: /.test(e.message), JSON.stringify(e));
    assert.equal(session.status().perturb.active, false);
    await session.step();
    assert.deepEqual(wrenches(session), [], 'nothing applied');
    // An invalid update ends a running drag.
    assert.equal(session.setPerturb(good), 'start');
    assert.equal(session.setPerturb({ ...good, target: [NaN, 0, 0] }), 'invalid');
    assert.equal(session.status().perturb.active, false);
    assert.equal(ended.at(-1).geom, PELVIS);
    await session.step();
    assert.deepEqual(wrenches(session), []);
    off();
    session.reset();
  });

  // Fail fast: a missing file rejects create() as soon as it fails, naming
  // it, and the other loads are aborted; before, create() waited for every
  // download (in a browser, the 25 MB of weights) before reporting anything.
  test('a missing asset fails create() at once and names it', async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'duet-e2e-'));
    try {
      for (const name of NEEDED) {
        if (name !== 'reference.bin.gz') fs.symlinkSync(fileURLToPath(new URL(name, ASSETS)), path.join(dir, name));
      }
      const stages = [];
      const t0 = performance.now();
      await assert.rejects(
        DemoSession.create({ assetsBase: pathToFileURL(`${dir}/`), backend: 'cpu', onProgress: (p) => stages.push(p.stage) }),
        (error) => /reference\.bin\.gz/.test(error.message),
      );
      const failMs = performance.now() - t0;
      t.diagnostic(`rejected after ${failMs.toFixed(0)} ms; a full create() took ${createMs.toFixed(0)} ms`);
      assert.ok(failMs < createMs / 2, `the failure took ${failMs.toFixed(0)} ms: it waited for the other loads`);
      assert.ok(!stages.includes('setup'), 'no setup stage after the failure');
    } finally {
      await new Promise((resolve) => setTimeout(resolve, 300)); // the aborted reads settle
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
