// The sensor charts get every control step (node --test; no browser): the real
// demo/js/worker.js runs DemoSession on the CPU behind a stand-in `self`, and
// the page side is ui/sensors.js SensorCard, fed every posted frame on arrival
// as app.js does. The test hands the frame buffers back late, in bursts, so
// the worker runs out of buffers (MAX_IN_FLIGHT) and skips most steps' frames;
// it walks, is reset by a message, falls (a shove) and is reset after the fall.
// Checks, per epoch: every control step after the reset reaches the card once
// and in order (in a frame of its own or in a later frame's sensorRows), with
// the sensor frame the session had at that step; the reset state
// (frame.initial) is never charted; the card's history has a sample every
// 10 ms and no break in its lines. The rows are what most steps rely on.
//
//   cd tools && node --test test/sensor_stream.test.mjs
//
// Needs the asset folder (paths.mjs ASSETS). About 20-40 s of wall time (CPU
// inference, real-time pacing).

import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { after, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { loadJSON } from '../../demo/js/lib/container.js';
import { SensorCard } from '../../demo/js/ui/sensors.js';
import { ASSETS } from './paths.mjs';

const NEEDED = ['demo.json', 'physics.mjb.gz', 'reference.bin.gz', 'human.bin.gz', 'osl.bin.gz'];
const missing = NEEDED.filter((f) => !existsSync(fileURLToPath(new URL(f, ASSETS))));
const SKIP = missing.length ? `missing ${fileURLToPath(ASSETS)}{${missing.join(',')}}: run python -m duet_export` : false;

const RETURN_EVERY_MS = 250; // the page hands its frames back this rarely (bursts)
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The worker's world: `self` (its messages to the page land in `inbox`) and a
// MessageChannel that, unlike Node's, holds no handle open once idle.
const inbox = [];
let deliver = null; // set by the test: the page's handler for a worker message
globalThis.self = {
  postMessage(message) {
    inbox.push(message.type);
    if (deliver) deliver(message);
  },
  addEventListener() {},
  onmessage: null,
};
globalThis.MessageChannel = class {
  constructor() {
    this.port1 = { onmessage: null };
    this.port2 = { postMessage: () => setImmediate(() => this.port1.onmessage && this.port1.onmessage({ data: 0 })) };
  }
};
const send = (data) => self.onmessage({ data });

let session = null;
after(() => {
  send({ type: 'pause', paused: true });
  if (session) session.dispose();
});

test('every control step reaches the sensor card: skipped frames, resets, a fall', { skip: SKIP, timeout: 240000 }, async (t) => {
  const demo = await loadJSON(new URL('demo.json', ASSETS));
  // The session the worker creates, on the CPU, recording the sensor frame of
  // every step it runs (the truth the page must see).
  const { DemoSession } = await import('../../demo/js/session.js');
  const create = DemoSession.create;
  const truth = new Map(); // `${epoch}:${steps}` -> sensor values (null: none, the reset state)
  DemoSession.create = async (options) => {
    session = await create.call(DemoSession, { ...options, backend: 'cpu' });
    const step = session._step.bind(session);
    const row = new Float32Array(session.sensorCount);
    session._step = async () => {
      const info = await step();
      truth.set(`${session.resets}:${session.steps}`, session.sensorRow(row) ? Float32Array.from(row) : null);
      return info;
    };
    return session;
  };
  await import('../../demo/js/worker.js');

  // The page: the card, the frames held (returned in bursts), what arrived.
  const card = new SensorCard(null, demo);
  const dt = demo.sim.policy_dt;
  const held = [];
  const arrived = []; // {epoch, steps, via: 'frame' | 'row', values}
  const problems = [];
  let frames = 0, initial = 0, skippedRows = 0, ready = false;
  const check = (what, ok) => { if (!ok && problems.length < 20) problems.push(what); };
  deliver = (m) => {
    if (m.type === 'ready') ready = true;
    if (m.type === 'error') problems.push(`worker error: ${m.message}`);
    if (m.type !== 'frame') return;
    const f = m.frame;
    frames += 1;
    const rows = f.sensorRows;
    check('frames carry sensorRows (null or rows)', rows !== undefined);
    if (rows) {
      check('rows are whole', rows.values.length === rows.times.length * card.channelCount);
      for (let i = 0; i < rows.times.length; i++) {
        arrived.push({ epoch: f.epoch, steps: Math.round(rows.times[i] / dt), time: rows.times[i], via: 'row', values: rows.values.slice(i * card.channelCount, (i + 1) * card.channelCount) });
        skippedRows += 1;
      }
    }
    if (f.initial) initial += 1;
    else arrived.push({ epoch: f.epoch, steps: Math.round(f.time / dt), time: f.time, via: 'frame', values: Float32Array.from(f.sensors) });
    card.push(f); // as app.js onFrame: before the frame's buffer goes back
    const h = card.history;
    check(`no break in the history (epoch ${f.epoch}, t ${f.time.toFixed(2)})`, h.breaks() === 0);
    for (let i = 1; i < h.count; i++) check(`a sample every control step (epoch ${f.epoch})`, Math.abs(h.time(i) - h.time(i - 1) - dt) < 1e-9);
    if (!f.initial) check(`the newest sample is the frame's (t ${f.time})`, h.count > 0 && h.newest === f.time);
    held.push(f.buffer);
  };
  const giveBack = () => { for (const buffer of held.splice(0)) send({ type: 'recycle', buffer }); };

  send({ type: 'init', assetsBase: ASSETS.href, backend: 'cpu' });
  const t0 = Date.now();
  while (!ready && !problems.length && Date.now() - t0 < 120000) await sleep(50);
  assert.ok(ready, `the worker loaded (${problems.join('; ')})`);
  assert.ok(initial >= 1, 'the standing pose behind the loader is an initial frame');
  giveBack();

  // Run until `seconds` more of simulated time, handing frames back in bursts.
  const runFor = async (seconds) => {
    const until = session.steps + Math.round(seconds / dt);
    let last = Date.now();
    while (session.steps < until && !problems.length) {
      await sleep(20);
      if (Date.now() - last >= RETURN_EVERY_MS) { giveBack(); last = Date.now(); }
    }
  };
  send({ type: 'pause', paused: false });
  send({ type: 'keys', codes: ['KeyW'] });
  await runFor(2.5);
  send({ type: 'reset' });
  await runFor(1.5);
  // A shove: the walker falls, goes down relaxed for 1 s, and resets.
  const falls = session.falls, resets = session.resets;
  const v = session.core.views();
  v.qvel[0] = 6;
  v.qvel[1] = 6;
  const t1 = Date.now();
  while (session.resets === resets && !problems.length && Date.now() - t1 < 60000) {
    await sleep(20);
    if (held.length > 3) giveBack();
  }
  assert.ok(session.falls > falls && session.resets > resets, 'the shove felled it and it reset');
  await runFor(1.0);
  send({ type: 'keys', codes: [] });
  send({ type: 'pause', paused: true });
  await sleep(100);
  giveBack(); // the frame still owed goes out with the rows kept for it
  await sleep(100);
  giveBack();

  assert.deepEqual(problems, []);
  // Per epoch: the steps that arrived, in arrival order, against the truth.
  const epochs = [...new Set(arrived.map((a) => a.epoch))];
  let checked = 0;
  for (const epoch of epochs) {
    const got = arrived.filter((a) => a.epoch === epoch);
    for (let i = 1; i < got.length; i++) assert.ok(got[i].steps > got[i - 1].steps, `epoch ${epoch}: in order, none twice (${got[i - 1].steps} then ${got[i].steps})`);
    const want = [...truth.entries()].filter(([key, values]) => key.startsWith(`${epoch}:`) && values).map(([key]) => Number(key.split(':')[1]));
    const steps = got.map((a) => a.steps);
    if (want.length) {
      // Every step of the epoch the session ran after its reset state, up to
      // the last one posted (the run stopped there).
      const last = steps[steps.length - 1];
      const expected = want.filter((s) => s <= last);
      assert.deepEqual(steps, expected, `epoch ${epoch}: every control step arrived once`);
    }
    for (const a of got) {
      const values = truth.get(`${epoch}:${a.steps}`);
      assert.ok(values, `epoch ${epoch} step ${a.steps}: the session ran it and it was no reset state`);
      assert.deepEqual(Array.from(a.values), Array.from(values), `epoch ${epoch} step ${a.steps}: the values of that step (${a.via})`);
      checked += 1;
    }
  }
  const st = card.stats();
  t.diagnostic(`${frames} frames (${initial} initial), ${skippedRows} rows of skipped steps, ${checked} steps checked over ${epochs.length} epochs; ` +
    `history ${st.samples} samples over ${st.span.toFixed(2)} s, ${st.breaks} breaks, gap ${st.gap}`);
  assert.ok(epochs.length >= 3, `three epochs: the start, a reset, the reset after the fall (${epochs.length})`);
  assert.ok(skippedRows > checked / 3, `most steps came as rows of skipped frames (${skippedRows} of ${checked})`);
  assert.ok(st.streamed && Math.abs(st.gap - 1.5 * dt) < 1e-12, 'the card knows every step arrives');
  assert.ok(inbox.includes('frame') && !inbox.includes('error'));
});
