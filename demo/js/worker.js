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
// Worker -> main:
//   {type: 'progress', loaded, total, stage, parts}
//   {type: 'ready', info}                 followed by a first frame and status
//   {type: 'frame', frame}                frame.buffer is transferred; frame.time
//                                         (s) and frame.epoch, see below
//   {type: 'status', status}              about 4 per second, and on changes
//   {type: 'event', event}                fall, reset, level, pause, backend
//   {type: 'error', message, stack}       the worker stops after an error
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

import { DemoSession } from './session.js';

const STATUS_MS = 250;
// The main thread holds about three frames for interpolation (the one before
// the time it draws and those after); with 4 in flight, about 1 step in 5
// found no free buffer at 60 fps (2026-11-16), with 6 every step's pose goes out.
const MAX_IN_FLIGHT = 6;
const RECYCLE_TIMEOUT_MS = 1000; // a frame not returned by then is written off
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
    return false;
  }
  const out = session.extract();
  frameOwed = false;
  if (!out) return false; // the state cannot be drawn: the last frame stays up
  out.frame.time = steps * session.dt;
  out.frame.epoch = epoch;
  posted = { steps, epoch };
  if (inFlight === 0) lastReturnAt = t;
  inFlight += 1;
  post({ type: 'frame', frame: out.frame }, out.transfer);
  return true;
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
      default:
        break;
    }
  } catch (error) {
    fail(error);
  }
};
