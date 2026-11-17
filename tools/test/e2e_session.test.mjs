// End-to-end check of demo/js/session.js in Node (node --test), on the real
// exported assets: DemoSession loads MuJoCo, physics.mjb.gz, the reference and
// both policies exactly as the browser worker does, then drives a scripted key
// schedule of 26 s of simulated time, stepped as fast as possible (no real-time
// pacing):
//   idle 1 s, W 4 s, W+A 3 s, W 2 s, D 2 s, S 3 s, idle 2 s,
//   Digit5 then W 3 s, Digit1 then W 2 s, A 2 s, idle 2 s.
// Checks: no fall; the root walks forward along its heading during W, at
// least 0.8x the commanded speed x time (a W that starts from standing is
// measured after its first 1.5 s: getting from a standstill to walking pace
// takes 1-1.5 s); it turns left during W+A and A and right during D and walks
// backward during S. A restart after a stop that misses the pace check is
// reported as an incident and at most one is tolerated: restarting after
// backing up occasionally hesitates for 2-3 s, a limit of the policy itself
// (fp32 too) that command shaping does not remove. The first walk after a
// reset is gated in DemoSession and must meet it. Prints per-step timings and
// the achievable real-time factor.
//
// Also: the real-time pacer (fake clock); command shaping (W while backing up
// stands first, the HUD says so, and the forward walk is served within
// SHAPING.maxHold); a fixed-seed key-masher (the human key model of the
// shaping benchmark: holds of 0.4-4 s, taps, steering while walking, direct
// reversals, stops, level changes and Shift) whose falls must stay within the
// measured bound; and the fall -> relaxed physics -> reset sequence. Loading:
// demo.json "versions" match the asset files (cache busting), and a missing
// asset fails create() at once, naming the file, instead of after the other
// downloads.
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
  DemoSession, Pacer, SHAPING, STARTING, STOPPING, clipLabel, speedText,
} from '../../demo/js/session.js';

const ASSETS = new URL('../../demo/assets/r2/', import.meta.url);
const NEEDED = ['demo.json', 'physics.mjb.gz', 'reference.bin.gz', 'human.bin.gz', 'osl.bin.gz', 'visual.bin.gz'];
const missing = NEEDED.filter((f) => !existsSync(fileURLToPath(new URL(f, ASSETS))));
const SKIP = missing.length ? `missing demo/assets/r2/{${missing.join(',')}}: run python -m duet_export` : false;

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
const STARTUP_SECONDS = 1.5; // not counted when a W starts from standing
const TURN_FRACTION = 0.5; // of commanded yaw rate x time, during W+A, A and D
const BACKWARD_FRACTION = 0.5; // of commanded speed x time, during S
const MAX_HESITANT_RESTARTS = 1; // restarts after a stop that miss the pace check

// Key-masher: the seeds and the falls allowed in them. The shaping benchmark
// (2026-11-16, WebGPU, 20 seeds x 45 s of this human model per candidate)
// measured 2.7 falls per 10 min without command shaping; at that rate 60 s
// expect 0.27 falls, and 3 or more would happen with probability 0.3 %.
const MASH_SEEDS = [0, 1];
const MASH_MAX_FALLS = 2;

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
      const fromRest = rows.length === 0 || rows[rows.length - 1].command.action === 'idle';
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
      const steady = r.forward && r.fromRest
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
        const want = BACKWARD_FRACTION * r.vx * r.seconds; // vx < 0
        assert.ok(r.progress <= want, `${label}: forward ${r.progress.toFixed(2)} m, want <= ${want.toFixed(2)} m`);
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
