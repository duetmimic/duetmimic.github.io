// Demo worker (module): loads MuJoCo, the assets and both policies, and runs
// the simulation in real time (js/session.js). The main thread (js/app.js)
// renders the frames it posts.
//
// Main -> worker:
//   {type: 'init', assetsBase, backend}   once; assetsBase is an absolute URL
//   {type: 'keys', codes}                 held KeyboardEvent.code values
//   {type: 'level', code}                 a key press: Digit1-5, KeyQ, KeyE
//   {type: 'pause', paused}               the session starts paused
//   {type: 'reset'}
//   {type: 'recycle', buffer}             a frame buffer back (transferred)
//   {type: 'perturb', geom, local, target, id?}
//                                         start or update a drag (pull a body
//                                         part): geom a rendered robot geom
//                                         (demo.visual.geom_ids), local the
//                                         grabbed point in its frame (world =
//                                         geom_xpos + geom_xmat . local), target
//                                         the world point it is pulled toward
//                                         (MuJoCo world, z up, m); id optional,
//                                         echoed. On pointerdown, then at most
//                                         once per animation frame
//   {type: 'perturb', release: true}      the visitor let go
// Worker -> main:
//   {type: 'progress', loaded, total, stage, parts}
//   {type: 'ready', info}                 followed by a first frame and status
//   {type: 'frame', frame}                frame.buffer is transferred; frame.time
//                                         (s), frame.epoch, frame.initial and
//                                         frame.sensorRows, see below; the
//                                         extras frame.wrench (socket load on
//                                         the limb) and frame.sensors (the
//                                         OSL's sensor frame): render/frame.js
//   {type: 'status', status}              about 4 per second, and on changes;
//                                         status.perturb: {active, geom, body,
//                                         id, force (N), maxForce, stretch (m)}
//   {type: 'event', event}                fall, reset, level, pause, backend,
//                                         perturb-end (below)
//   {type: 'error', message, stack}       the worker stops after an error
//
// Drags (DemoSession.setPerturb, DragSpring in session.js): the spring pulls at
// every physics tick until a release. The worker ends a drag on its own with
// the event perturb-end {reason, message, geom, body, id}, after which the
// page drops its drag: reason 'reset' (R, the Reset button), 'fall-reset' (the
// reset after a fall; the drag keeps pulling through the fall itself) or
// 'error' (an invalid perturb message, e.g. a geom that is not drawn or a
// non-finite number, which never stops the worker; or the worker failing).
// Updates for a drag the worker ended, still on their way, are ignored; a
// release or a new grab (another geom or local point) starts over.
//
// Pacing: control steps run until simulated time catches up with wall time.
// More than 0.25 s behind, the debt is dropped and the real-time factor falls
// below 1 (Pacer in session.js). A paused clock does not run, so resuming
// causes no catch-up burst.
//
// Frames: the pose after every control step goes out, so the main thread can
// interpolate between consecutive poses by simulated time (render/scene.js).
// frame.time is the simulated time of the pose in seconds, counted in control
// steps since the session was created (falls included), so it never goes
// back; frame.epoch is the number of resets so far, and a new epoch means the
// pose jumped (a reset) and must not be blended with the ones before. At most
// MAX_IN_FLIGHT frames are out at once: the main thread holds the two around
// the time it draws plus newer ones, and posts each buffer back once it no
// longer reads it. A step whose frame cannot go out (all in flight) is
// skipped; the next free buffer carries the pose of that moment.
//
// frame.initial: true for the state a reset (or the load) starts from, before
// its first control step (SimCore's steps === 0): the reset pose stands the
// walker on the uncompressed socket piston with the foot pressed into the
// floor, so its extras read an unloaded socket (about 26 N) and a 3 g IMU,
// not a measurement of the walk. The page shows its pose, not its extras. A
// step a reset discards (its inference was in flight) leaves that state, so
// the frame after it is initial too.
//
// frame.sensorRows: the sensor charts need frame.sensors of every control
// step, also of the steps whose frame did not go out. The worker keeps the
// row of each skipped step (its frame.sensors values; ROW_CAPACITY at most,
// the oldest dropped) and sends the rows, oldest first, with the next frame
// of the same epoch: {times: Float64Array(n) (s, as frame.time), values:
// Float32Array(n * width)}, every row older than the frame itself (whose own
// row is frame.sensors), or null when there are none. So, with the frames,
// every step's row reaches the page once, in order: none twice (a row is
// dropped once a frame of its step or a later one went out), none of an older
// epoch (a reset drops them: the page starts over) and none of an initial
// state. The rows travel in the frame message rather than in one of their
// own, so they cannot arrive after a newer frame (the page's history keeps
// time order and drops an older row) and need no message of their own on the
// usual path, where every frame goes out. A page that predates them ignores
// both fields; frames without them (an older worker) are drawn as before.

import { DemoSession } from './session.js';

const STATUS_MS = 250;
// The main thread holds about three frames for interpolation (the one before
// the time it draws and those after); with 4 in flight, about 1 step in 5
// found no free buffer at 60 fps (2026-11-16); with 6, at 60 fps and without
// stalls, nearly every step's pose goes out. A page that draws less often
// returns buffers in bursts and steps are skipped (2026-11-19, RTX 4090:
// 0.1-5% of steps at 30-60 fps on a quiet machine, 11-25% at load 54-70, a
// third at 20 fps, half at 15 fps; software rendering at 4-8 fps 35-40%).
// The poses then come further apart (interpolated), and the skipped steps'
// sensor rows go out with the next frame (frame.sensorRows).
const MAX_IN_FLIGHT = 6;
const RECYCLE_TIMEOUT_MS = 1000; // a frame not returned by then is written off
// Sensor rows kept for frames that did not go out: 4 s of steps, more than the
// 3 s the charts show (frames go out again RECYCLE_TIMEOUT_MS after the page
// stopped returning them, so in practice they hold under 1 s).
const ROW_CAPACITY = 400;
const BATCH_BUDGET_MS = 30; // wall time per batch before yielding to messages
const PROGRESS_MS = 100;

let session = null;
let initializing = false;
let dead = false;
let wantPaused = true;
let keys = [];
const pendingLevels = [];

let busy = false;
let timer = 0;
let yieldQueued = false;
let inFlight = 0;
let lastReturnAt = 0;
let frameOwed = false;
let posted = { steps: -1, epoch: -1 }; // the pose last posted
let lastStatusAt = -Infinity;
let rows = null; // the sensor rows of skipped steps (keepRow), once loaded

const now = () => performance.now();
const post = (message, transfer) => self.postMessage(message, transfer || []);

function describeError(error) {
  if (error instanceof Error) return { message: error.message || String(error), stack: error.stack || '' };
  if (error && typeof error === 'object' && 'message' in error) return { message: String(error.message), stack: '' };
  return { message: String(error), stack: '' };
}

function fail(error) {
  if (dead) return;
  dead = true;
  clearTimeout(timer);
  timer = 0;
  try { // a drag in progress ends with the simulation
    if (session && session.drag.active) session.clearPerturb('error', 'the simulation stopped');
  } catch { /* the error below says it all */ }
  try {
    post({ type: 'error', ...describeError(error) });
  } catch {
    post({ type: 'error', message: 'The simulation stopped.' });
  }
}

self.addEventListener('error', (event) => {
  event.preventDefault();
  fail(event.error || event.message || 'worker error');
});
self.addEventListener('unhandledrejection', (event) => {
  event.preventDefault();
  fail(event.reason || 'unhandled rejection');
});

// ------------------------------------------------------------------ loading
async function init({ assetsBase, backend = 'auto' }) {
  if (session || initializing) return;
  initializing = true;
  let lastProgressAt = -Infinity;
  session = await DemoSession.create({
    assetsBase,
    backend: ['auto', 'webgpu', 'cpu'].includes(backend) ? backend : 'auto',
    paused: true,
    onProgress: (p) => {
      const t = now();
      if (p.loaded < p.total && t - lastProgressAt < PROGRESS_MS) return;
      lastProgressAt = t;
      post({ type: 'progress', loaded: p.loaded, total: p.total, stage: p.stage, parts: p.parts });
    },
  });
  initializing = false;
  if (dead) return;
  session.on('*', (event) => {
    post({ type: 'event', event });
    if (event.type === 'reset') postFrame(true);
    if (event.type !== 'pause') sendStatus();
  });
  const width = session.sensorCount;
  rows = width > 0 ? {
    width, epoch: -1, start: 0, count: 0,
    steps: new Float64Array(ROW_CAPACITY), values: new Float32Array(ROW_CAPACITY * width),
  } : null;
  session.setKeys(keys);
  for (const code of pendingLevels.splice(0)) session.handleKey(code);
  post({ type: 'ready', info: session.describe() });
  postFrame(true); // the standing pose, drawn behind the loader
  sendStatus();
  setPaused(wantPaused);
}

// ------------------------------------------------------------------ output
// The current pose, unless it already went out. force: a new epoch (reset) or
// the first frame, posted even with every buffer in flight.
function postFrame(force = false) {
  if (!session || dead) return false;
  const t = now();
  const steps = session.steps, epoch = session.resets;
  if (!force && steps === posted.steps && epoch === posted.epoch) {
    frameOwed = false; // nothing new since the last frame
    return false;
  }
  if (inFlight >= MAX_IN_FLIGHT && t - lastReturnAt > RECYCLE_TIMEOUT_MS) {
    inFlight = 0; // the main thread stopped returning frames (hidden, busy): start over
  }
  if (!force && inFlight >= MAX_IN_FLIGHT) {
    frameOwed = true;
    keepRow(steps, epoch); // its sensors go out with the next frame
    return false;
  }
  const out = session.extract();
  frameOwed = false;
  if (!out) return false; // the state cannot be drawn: the last frame stays up
  const frame = out.frame;
  frame.time = steps * session.dt;
  frame.epoch = epoch;
  frame.initial = session.initial;
  frame.sensorRows = takeRows(steps, epoch);
  posted = { steps, epoch };
  if (inFlight === 0) lastReturnAt = t;
  inFlight += 1;
  post({ type: 'frame', frame }, out.transfer);
  return true;
}

// The sensor row of the current state (step `steps` of `epoch`), whose frame
// did not go out: kept for the next frame (see the header). Once per step; an
// initial state or one that cannot be drawn has none.
function keepRow(steps, epoch) {
  const r = rows;
  if (!r) return;
  if (r.epoch !== epoch) { // a reset: the rows kept so far are of no use to the page
    r.epoch = epoch;
    r.start = 0;
    r.count = 0;
  }
  if (r.count && r.steps[(r.start + r.count - 1) % ROW_CAPACITY] === steps) return; // kept already
  const slot = r.count < ROW_CAPACITY ? (r.start + r.count) % ROW_CAPACITY : r.start;
  if (!session.sensorRow(r.values, slot * r.width)) return;
  r.steps[slot] = steps;
  if (r.count < ROW_CAPACITY) r.count += 1;
  else r.start = (r.start + 1) % ROW_CAPACITY; // full: the oldest row went
}

// frame.sensorRows for a frame of step `steps` of `epoch`: the rows kept for
// it, oldest first, copied (the ring is reused), or null. A row of that step
// itself is dropped (the frame's own sensors are it), as is everything of
// another epoch; the ring is empty afterwards.
function takeRows(steps, epoch) {
  const r = rows;
  if (!r || !r.count) return null;
  let n = 0;
  if (r.epoch === epoch) while (n < r.count && r.steps[(r.start + n) % ROW_CAPACITY] < steps) n += 1;
  let out = null;
  if (n) {
    const w = r.width, dt = session.dt;
    const times = new Float64Array(n), values = new Float32Array(n * w);
    for (let i = 0; i < n; i++) {
      const slot = (r.start + i) % ROW_CAPACITY;
      times[i] = r.steps[slot] * dt; // as frame.time
      for (let k = 0, from = slot * w, to = i * w; k < w; k++) values[to + k] = r.values[from + k];
    }
    out = { times, values };
  }
  r.start = 0;
  r.count = 0;
  return out;
}

function sendStatus() {
  if (!session || dead) return;
  lastStatusAt = now();
  post({ type: 'status', status: session.status() });
}

// ------------------------------------------------------------------ loop
// One batch: the steps owed now, one at a time with a frame after each,
// bounded by BATCH_BUDGET_MS; a status every STATUS_MS; then sleep until the
// next step is due, or yield to pending messages at once when behind.
const channel = new MessageChannel();
channel.port1.onmessage = () => {
  yieldQueued = false;
  loop();
};

function schedule() {
  if (dead || !session || session.paused || busy || timer || yieldQueued) return;
  const wait = session.msUntilNextStep();
  if (wait > 1) {
    timer = setTimeout(() => { timer = 0; loop(); }, Math.max(1, Math.floor(wait)));
  } else {
    yieldQueued = true; // a message, not setTimeout(0), which browsers clamp to 4 ms
    channel.port2.postMessage(0);
  }
}

async function loop() {
  if (dead || !session || session.paused || busy) return;
  busy = true;
  try {
    const started = now();
    while (!dead && !session.paused) {
      if (await session.advance({ budgetMs: BATCH_BUDGET_MS, maxSteps: 1 }) === 0) break;
      postFrame();
      if (now() - started >= BATCH_BUDGET_MS) break;
    }
    if (frameOwed) postFrame();
    if (now() - lastStatusAt >= STATUS_MS) sendStatus();
  } catch (error) {
    fail(error);
  } finally {
    busy = false;
  }
  schedule();
}

function setPaused(paused) {
  wantPaused = paused;
  if (!session) return;
  session.pause(paused);
  if (paused) {
    clearTimeout(timer);
    timer = 0;
  }
  sendStatus();
  schedule();
}

// A drag message. DemoSession.setPerturb checks it (a rendered robot geom, a
// grabbed point on it, finite numbers): an invalid one ends any drag with
// perturb-end {reason: 'error'} and leaves the simulation running.
function perturb(message) {
  if (dead) return;
  if (!session) { // not loaded yet: nothing can be dragged
    if (!message.release) {
      post({ type: 'event', event: { type: 'perturb-end', reason: 'error', message: 'perturb: the simulation is not ready', geom: null, body: null, id: null } });
    }
    return;
  }
  if (message.release) {
    if (session.clearPerturb('release')) sendStatus();
    return;
  }
  const result = session.setPerturb({ geom: message.geom, local: message.local, target: message.target, id: message.id });
  if (result === 'start') sendStatus();
}

// ------------------------------------------------------------------ messages
self.onmessage = (event) => {
  const message = event.data || {};
  try {
    switch (message.type) {
      case 'init':
        init(message).catch(fail);
        break;
      case 'keys':
        keys = Array.isArray(message.codes) ? message.codes.map(String) : [];
        if (session) {
          session.setKeys(keys);
          sendStatus();
        }
        break;
      case 'level':
        if (session) {
          if (!session.handleKey(String(message.code))) sendStatus();
        } else {
          pendingLevels.push(String(message.code));
        }
        break;
      case 'pause':
        setPaused(!!message.paused);
        break;
      case 'reset':
        if (session && !dead) session.reset('user'); // the reset event posts a frame and status
        break;
      case 'recycle':
        inFlight = Math.max(0, inFlight - 1);
        lastReturnAt = now();
        if (message.buffer && session) session.recycle(message.buffer);
        if (frameOwed && !busy) postFrame();
        break;
      case 'perturb':
        perturb(message);
        break;
      default:
        break;
    }
  } catch (error) {
    fail(error);
  }
};
