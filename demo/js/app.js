// Demo page, main thread: the loader, the three.js view, keyboard and touch
// input and the HUD. Physics and both policies run in the module worker
// (js/worker.js); this file only draws the frames it posts and forwards input.
// DOM contract: tools/DESIGN.md and the comments in demo/index.html.
//
// Start flow: the download starts as soon as the page loads (a visitor of
// /demo/, or of the project page who pressed Start there, has opted in). When
// everything is ready, #start-btn is enabled; pressing it (or a movement key,
// or the touch pad) hides the loader and starts the simulation, which waits
// paused until then with the model standing behind the loader, seen from the
// front-right. Start switches the camera to 'chase', behind the walker's right
// shoulder. Embedded (?embed=1), the parent page focuses the iframe on its own
// Start click, so the demo starts by itself when it is ready and the iframe
// has keyboard focus; otherwise it shows Start, whose click also gives the
// iframe the keyboard.
//
// Embedded, the demo behaves like part of the page: a plain mouse wheel
// scrolls the parent page (Ctrl or ⌘ + wheel, or a pinch, zooms), the help
// starts as a one-line legend, and the simulation pauses while the frame is
// scrolled out of view.
//
// Touch: on a coarse pointer (html.touch, set by the page shell) or from the
// first touch, an on-screen pad (#touch-pad) holds the same KeyboardEvent.code
// values as the keyboard; help and hints switch to touch wording.
//
// Rendering: the walker is drawn between the worker's per-step poses
// (render/scene.js), and only when something changed, so a paused or
// off-screen demo costs no GPU work.
//
// Query: ?backend=cpu|webgpu (default auto), ?debug=1 (extra HUD rows).
// Test hook: window.__duet.state(), .stats(), .events().

import { loadContainer, loadJSON } from './lib/container.js';

const ASSETS = new URL('../assets/r2/', import.meta.url);
// Bytes not listed in demo.json, for the download note: vendor/mujoco/mujoco.wasm.gz
// plus three.js, mujoco.js and the demo's own scripts, styles and demo.json as
// GitHub Pages sends them (gzip). Measured 2026-11-16 (31.2 MB in all).
const UNLISTED_BYTES = 2475508 + 577000;
// Camera modes in the order C cycles them, and their names in the toast.
const CAMERA_MODES = ['chase', 'follow', 'free'];
const CAMERA_NAMES = {
  chase: 'behind the walker',
  follow: 'fixed angle (drag to orbit)',
  free: 'free (drag to orbit, right-drag to pan)',
};
const CAMERA_NAMES_TOUCH = { ...CAMERA_NAMES, free: 'free (drag to orbit, two fingers to pan)' };
const TOAST_MS = 1500;
const ZOOM_HINT_MS = 4000; // embedded: "Ctrl + scroll to zoom" at most this often
const CONTEXT_RESTORE_MS = 5000; // a lost WebGL context not back by then ends the demo
const ZOOM_KEY = /mac|iphone|ipad/i.test((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '') ? '⌘' : 'Ctrl';

const $ = (id) => document.getElementById(id);
const el = {
  root: document.documentElement,
  viewport: $('viewport'),
  loader: $('loader'),
  loaderStatus: $('loader-status'),
  loaderBar: $('loader-bar'),
  loaderSize: $('loader-size'),
  startBtn: $('start-btn'),
  hud: $('hud'),
  command: $('hud-command'),
  speed: $('hud-speed'),
  clip: $('hud-clip'),
  time: $('hud-time'),
  rtf: $('hud-rtf'),
  backend: $('hud-backend'),
  note: $('hud-note'),
  noteText: document.querySelector('#hud-note dd'),
  help: $('help'),
  toast: $('toast'),
  reset: $('btn-reset'),
  pause: $('btn-pause'),
  camera: $('btn-camera'),
  helpBtn: $('btn-help'),
  pad: $('touch-pad'),
  fast: $('pad-fast'),
  error: $('error'),
  errorText: $('error-text'),
};

const params = new URLSearchParams(location.search);
const embedded = el.root.classList.contains('embed');
const debug = params.has('debug') && !['0', 'false'].includes(params.get('debug'));
const requestedBackend = ['cpu', 'webgpu', 'auto'].includes(params.get('backend')) ? params.get('backend') : 'auto';
const reducedMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

const app = {
  phase: 'loading', // loading | ready | running | failed (also html[data-phase])
  view: null,
  worker: null,
  workerReady: false,
  info: null, // the worker's 'ready' info
  status: null, // the latest worker status
  level: null,
  root: null, // [x, y, z, heading] of the latest frame
  epoch: null, // reset count of the latest frame
  error: null,
  events: [],
  frames: { received: 0, returned: 0, rafs: 0, renders: 0 },
  fps: 0, // drawings per second
  cpuNoteShown: false,
  hasMoved: false, // until the first move the command pill says how to walk
  help: 'hidden', // full | compact | hidden
  touch: el.root.classList.contains('touch'),
  fast: false, // the touch pad's Fast toggle
  offscreen: false,
  contextLost: false,
};

function setPhase(phase) {
  app.phase = phase;
  el.root.dataset.phase = phase;
}

// ------------------------------------------------------------------ errors
function describe(error) {
  if (!error) return '';
  if (typeof error === 'string') return error;
  return error.message || String(error);
}

let contextTimer = 0;

function fail(message, detail) {
  if (app.phase === 'failed') return;
  setPhase('failed');
  clearTimeout(contextTimer);
  const text = detail ? `${message}\n\n${describe(detail)}` : message;
  app.error = text;
  if (detail && detail.stack) console.error(detail);
  el.errorText.textContent = text;
  el.error.hidden = false;
  el.loader.classList.add('is-failed');
  el.loaderStatus.textContent = message;
  el.startBtn.disabled = true;
  clearKeys();
  if (app.worker) {
    try { app.worker.terminate(); } catch { /* ignore */ }
  }
}

// Wrap an event handler so a bug shows the error overlay instead of leaving a
// half-working page.
const guard = (fn) => (...args) => {
  try {
    return fn(...args);
  } catch (error) {
    fail('The demo stopped because of an error.', error);
    return undefined;
  }
};

function unsupported() {
  if (typeof WebAssembly !== 'object') return 'This browser cannot run WebAssembly, which the physics engine needs. Try a recent Chrome, Edge, Firefox or Safari.';
  if (typeof Worker !== 'function') return 'This browser cannot run Web Workers, which the simulation needs. Try a recent Chrome, Edge, Firefox or Safari.';
  if (typeof DecompressionStream !== 'function') return 'This browser is too old for the demo. Try a recent Chrome, Edge, Firefox or Safari.';
  // The probe context is left to the garbage collector: losing it on purpose
  // makes Firefox warn "WebGL context was lost" on every load.
  let gl = null;
  try {
    gl = document.createElement('canvas').getContext('webgl2');
  } catch {
    gl = null;
  }
  if (!gl) {
    return 'This demo needs WebGL 2, which this browser or device does not provide. ' +
      'Check that hardware acceleration is turned on, or try a recent Chrome, Edge, Firefox or Safari.';
  }
  return null;
}

// The CPU-inference note, worded by why the human network is not on WebGPU
// (status().backendReason: the policy's info.reason or fallback, nn/index.js).
function cpuNote(reason = '') {
  if (reason.startsWith('CPU faster')) {
    return "This browser's WebGPU is too slow for per-step inference, so the networks run on the CPU in slow motion. A browser with fast WebGPU (recent Chrome, Edge or Safari) runs it in real time.";
  }
  if (/no navigator\.gpu|WebGPU is not available|no adapter/i.test(reason)) {
    return 'No WebGPU here, so the networks run on the CPU in slow motion. A browser with WebGPU (recent Chrome, Edge or Safari) runs it in real time.';
  }
  return 'The networks run on the CPU here, so the demo may run slower than real time.';
}

// ------------------------------------------------------------------ toast
let toastTimer = 0;

function pauseNotice() {
  return app.touch ? 'Paused — tap the play button to resume' : 'Paused — press Space to resume';
}

function toast(text, ms = TOAST_MS) {
  clearTimeout(toastTimer);
  el.toast.textContent = text;
  el.toast.classList.add('show');
  toastTimer = setTimeout(hideToast, ms);
}

// Back to the pause notice while the user's pause holds, else nothing.
function hideToast() {
  clearTimeout(toastTimer);
  if (pauseReasons.has('user')) {
    el.toast.textContent = pauseNotice();
    el.toast.classList.add('show');
  } else {
    el.toast.classList.remove('show');
  }
}

// ------------------------------------------------------------------ key labels
// Bindings use KeyboardEvent.code: a key's position on a US (QWERTY) layout.
// The labels show what the visitor's layout prints at those positions, from
// the Keyboard Map API where it exists, else learned from the first letter key
// whose e.key disagrees with its code (AZERTY: the W position types 'z').
// Elements with data-label="<code>" show the label of that code.
const LABELLED = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyR', 'KeyC', 'KeyH'];
// Common layouts that differ from QWERTY at those positions: one key from the
// fallback identifies the layout and so labels the others too.
const LAYOUTS = [
  { KeyW: 'z', KeyA: 'q', KeyS: 's', KeyD: 'd', KeyQ: 'a', KeyE: 'e', KeyR: 'r', KeyC: 'c', KeyH: 'h' }, // AZERTY
  { KeyW: ',', KeyA: 'a', KeyS: 'o', KeyD: 'e', KeyQ: "'", KeyE: '.', KeyR: 'p', KeyC: 'j', KeyH: 'd' }, // Dvorak
  { KeyW: 'w', KeyA: 'a', KeyS: 'r', KeyD: 's', KeyQ: 'q', KeyE: 'f', KeyR: 'p', KeyC: 'c', KeyH: 'h' }, // Colemak
];
const keyLabels = new Map(); // code -> label where it differs from the QWERTY letter
let layoutNoticed = false;

const keyLabel = (code) => keyLabels.get(code) || (/^Key[A-Z]$/.test(code) ? code.slice(3) : code);
const wasd = () => ['KeyW', 'KeyA', 'KeyS', 'KeyD'].map(keyLabel).join(' ');

// Label each code in LABELLED with lookup(code) (a printed character), where
// that is printable ASCII: non-Latin layouts keep the Latin letters, which
// their keycaps carry too. Returns whether a label changed.
function relabel(lookup) {
  let changed = false;
  for (const code of LABELLED) {
    const ch = lookup(code);
    if (typeof ch !== 'string' || !/^[!-~]$/.test(ch)) continue;
    const label = ch.toUpperCase();
    if (label === keyLabel(code)) continue;
    if (label === code.slice(3)) keyLabels.delete(code);
    else keyLabels.set(code, label);
    changed = true;
  }
  if (changed) refreshTexts();
  return changed;
}

function learnLayout(e) {
  const code = e.code;
  if (!/^Key[A-Z]$/.test(code) || typeof e.key !== 'string' || e.key.length !== 1) return;
  const ch = e.key.toLowerCase();
  if (ch === keyLabel(code).toLowerCase()) return;
  const layout = LAYOUTS.find((l) => l[code] === ch);
  if (relabel((c) => (c === code ? ch : layout && layout[c])) && !layoutNoticed) {
    layoutNoticed = true;
    toast(`Keys relabelled for your keyboard: ${wasd()} to walk and turn`, 4000);
  }
}

// ------------------------------------------------------------------ touch
// Touch mode: the pad, touch wording in the loader, pill, help and toasts.
function setTouch() {
  if (app.touch) return;
  app.touch = true;
  el.root.classList.add('touch');
  if (app.help === 'compact') setHelp('hidden'); // the pad replaces the keyboard legend
  refreshTexts();
}

// Texts that name keys or depend on touch mode.
function refreshTexts() {
  for (const node of document.querySelectorAll('[data-label]')) node.textContent = keyLabel(node.dataset.label);
  if (app.phase === 'ready') el.loaderStatus.textContent = readyText();
  if (app.status) showCommand(app.status);
  if (pauseReasons.has('user') && el.toast.classList.contains('show')) el.toast.textContent = pauseNotice();
}

// ------------------------------------------------------------------ loader
const progress = { worker: null, visual: { loaded: 0, total: 0 } };
const mb = (bytes) => (bytes / 1e6).toFixed(1);

function setBar(percent) {
  el.loaderBar.style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
}

// Combined progress once both totals are known (the worker's report and
// demo.json's visual size), so the total never jumps.
function updateLoader() {
  if (app.phase !== 'loading') return;
  const w = progress.worker;
  if (!w || !progress.visual.total) return; // the bar keeps its indeterminate sweep
  const loaded = w.loaded + progress.visual.loaded;
  const total = w.total + progress.visual.total;
  setBar(total > 0 ? (100 * loaded) / total : 0);
  el.loaderSize.hidden = true; // the status line shows the size from now on
  el.loaderStatus.textContent = loaded < total
    ? `Downloading… ${mb(loaded)} of ${mb(total)} MB`
    : 'Starting the simulation…';
}

function showDownloadSize(demo) {
  const p = demo.policies;
  const bytes = UNLISTED_BYTES + demo.physics.bytes + demo.reference.bytes + demo.visual.bytes + p.human.bytes + p.osl.bytes;
  el.loaderSize.textContent = `About ${Math.round(bytes / 1e6)} MB download`;
}

function readyText() {
  if (embedded) return app.touch ? 'Ready. Tap Start.' : 'Ready. Click Start to use the keyboard.';
  return app.touch ? 'Ready. Press Start, then hold ▲ to walk.' : `Ready. Press Start, then walk with ${wasd()}.`;
}

function maybeReady() {
  if (app.phase !== 'loading' || !app.workerReady || !app.view) return;
  setPhase('ready');
  setBar(100);
  el.startBtn.disabled = false;
  if (embedded && document.hasFocus()) {
    start();
    return;
  }
  el.loaderStatus.textContent = readyText();
  if (!embedded) el.startBtn.focus({ preventScroll: true });
}

function start() {
  if (app.phase !== 'ready') return;
  setPhase('running');
  el.startBtn.disabled = true;
  el.loaderStatus.textContent = 'Running';
  el.loader.classList.add('is-done');
  setTimeout(() => { el.loader.hidden = true; }, 500);
  el.hud.inert = false;
  setHelp(app.touch ? 'hidden' : embedded ? 'compact' : 'full');
  app.view.setCameraMode('chase');
  el.camera.dataset.mode = 'chase';
  const active = document.activeElement;
  if (active && active !== document.body && typeof active.blur === 'function') active.blur();
  setPause('start', false);
  maybeCpuNote();
}

// The stats panel carries the CPU note; where that panel is hidden (narrow
// windows, touch screens), say it once in a toast instead.
function maybeCpuNote() {
  const s = app.status;
  if (!s || s.backend !== 'CPU' || app.cpuNoteShown || app.phase !== 'running') return;
  app.cpuNoteShown = true;
  const stats = el.speed.closest('dl');
  if (!stats || getComputedStyle(stats).display === 'none') toast(cpuNote(s.backendReason), 6000);
}

// ------------------------------------------------------------------ pause
// Reasons: start (until Start), user (Space, the button), hidden (tab in the
// background), offscreen (embedded frame scrolled away), context (WebGL lost).
const pauseReasons = new Set(['start']);
let sentPaused = null;

function setPause(reason, on) {
  if (on) pauseReasons.add(reason);
  else pauseReasons.delete(reason);
  const paused = pauseReasons.size > 0;
  if (app.worker && paused !== sentPaused) {
    app.worker.postMessage({ type: 'pause', paused });
    sentPaused = paused;
  }
  const user = pauseReasons.has('user');
  el.pause.setAttribute('aria-pressed', String(user));
  el.pause.setAttribute('aria-label', user ? 'Resume' : 'Pause');
  el.pause.title = user ? 'Resume (Space)' : 'Pause (Space)';
}

function togglePause() {
  if (app.phase !== 'running') return;
  setPause('user', !pauseReasons.has('user'));
  hideToast(); // shows the pause notice while paused
}

// ------------------------------------------------------------------ input
const DEFAULT_BINDINGS = {
  forward: ['KeyW', 'ArrowUp'], backward: ['KeyS', 'ArrowDown'], left: ['KeyA', 'ArrowLeft'],
  right: ['KeyD', 'ArrowRight'], fast: ['ShiftLeft', 'ShiftRight'], slower: ['KeyQ'], faster: ['KeyE'],
  levels: ['Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5'],
};
let bindings = DEFAULT_BINDINGS;
let moveCodes = new Set();
let levelCodes = new Set();

function useBindings(b) {
  bindings = { ...DEFAULT_BINDINGS, ...(b || {}) };
  moveCodes = new Set([...bindings.forward, ...bindings.backward, ...bindings.left, ...bindings.right, ...bindings.fast]);
  levelCodes = new Set([...bindings.levels, ...bindings.slower, ...bindings.faster]);
}
useBindings(DEFAULT_BINDINGS);

const keysHeld = new Set(); // movement keys held on the keyboard
const padHeld = new Set(); // movement codes held on the touch pad
const down = new Set(); // every key held, for the kbd highlights
const kbds = Array.from(document.querySelectorAll('kbd[data-key]'), (k) => ({ el: k, codes: k.dataset.key.split(/\s+/) }));
const padButtons = el.pad ? Array.from(el.pad.querySelectorAll('button[data-code]')) : [];

// What the worker gets: keyboard and pad together, Shift while Fast is on.
function heldCodes() {
  const codes = new Set([...keysHeld, ...padHeld]);
  if (app.fast) codes.add('ShiftLeft');
  return [...codes];
}

function refreshKbds() {
  const levelCode = app.level === null ? null : bindings.levels[app.level];
  const lit = new Set([...down, ...heldCodes()]);
  for (const { el: k, codes } of kbds) {
    k.classList.toggle('pressed', codes.some((c) => lit.has(c) || c === levelCode));
  }
}

function sendKeys() {
  if (app.worker) app.worker.postMessage({ type: 'keys', codes: heldCodes() });
}

function clearKeys() {
  const had = keysHeld.size > 0 || padHeld.size > 0;
  keysHeld.clear();
  padHeld.clear();
  down.clear();
  for (const button of padButtons) button.classList.remove('pressed');
  refreshKbds();
  if (had && app.phase !== 'failed') sendKeys();
}

function isControl(target) {
  return !!(target && target.closest && target.closest('button, a[href], input, select, textarea, [contenteditable="true"]'));
}

function onKeyDown(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const code = e.code;
  if (!code) return;
  learnLayout(e);
  // A focused button or link handles its own Space and Enter.
  if ((code === 'Space' || code === 'Enter' || code === 'NumpadEnter') && isControl(e.target)) return;
  if (code === 'Space' || code.startsWith('Arrow')) e.preventDefault(); // no page (or parent page) scrolling
  if (app.phase === 'ready' && moveCodes.has(code)) start();
  if (app.phase !== 'running') return;
  if (!down.has(code)) {
    down.add(code);
    refreshKbds();
  }
  if (moveCodes.has(code)) {
    if (!keysHeld.has(code)) {
      keysHeld.add(code);
      sendKeys();
    }
    return;
  }
  if (e.repeat) return;
  if (levelCodes.has(code)) {
    app.worker.postMessage({ type: 'level', code });
    return;
  }
  if (e.key === '?') {
    toggleHelp();
    return;
  }
  switch (code) {
    case 'Space': togglePause(); break;
    case 'KeyR': resetSim(); break;
    case 'KeyC': cycleCamera(); break;
    case 'KeyH': toggleHelp(); break;
    default: break;
  }
}

function onKeyUp(e) {
  const code = e.code;
  if (!code) return;
  // A focused button clicks on Space keyup, so leave that alone; scrolling
  // is a keydown default, prevented there.
  if ((code === 'Space' || code.startsWith('Arrow')) && !isControl(e.target)) e.preventDefault();
  const wasDown = down.delete(code);
  const wasHeld = keysHeld.delete(code);
  if (wasDown || wasHeld) refreshKbds(); // after both: the highlights read the held set too
  if (wasHeld) sendKeys();
}

// Touch pad: each arrow holds its code while pressed (pointer capture, so a
// finger sliding off still lets go on release); Fast toggles Shift. The first
// press on the ready screen starts the demo.
function padPress(button, e) {
  e.preventDefault(); // no focus, no emulated mouse events or click
  if (e.pointerType === 'touch') setTouch();
  if (app.phase === 'ready') start();
  if (app.phase !== 'running') return;
  try { button.setPointerCapture(e.pointerId); } catch { /* the pointer is already gone */ }
  button.classList.add('pressed');
  const code = button.dataset.code;
  if (!padHeld.has(code)) {
    padHeld.add(code);
    refreshKbds();
    sendKeys();
  }
}

function padRelease(button) {
  button.classList.remove('pressed');
  if (padHeld.delete(button.dataset.code)) {
    refreshKbds();
    sendKeys();
  }
}

function toggleFast() {
  if (app.phase === 'ready') start();
  if (app.phase !== 'running') return;
  app.fast = !app.fast;
  el.fast.setAttribute('aria-pressed', String(app.fast));
  refreshKbds();
  sendKeys();
  toast(app.fast ? 'Fast: top speed while walking' : 'Fast: off');
}

function resetSim() {
  if (app.phase !== 'running' || !app.worker) return;
  app.worker.postMessage({ type: 'reset' });
  toast('Reset');
}

function cycleCamera() {
  if (!app.view) return;
  const next = CAMERA_MODES[(CAMERA_MODES.indexOf(app.view.cameraMode) + 1) % CAMERA_MODES.length];
  app.view.setCameraMode(next);
  el.camera.dataset.mode = next;
  toast(`Camera: ${(app.touch ? CAMERA_NAMES_TOUCH : CAMERA_NAMES)[next]}`);
}

// Help: 'full' (the panel), 'compact' (the one-line legend, embedded) or
// 'hidden'. It starts full on /demo/, compact when embedded and hidden on
// touch screens; H, ? and the ? button switch between full and that start.
function setHelp(mode) {
  app.help = mode;
  el.help.hidden = mode === 'hidden';
  el.help.dataset.mode = mode === 'compact' ? 'compact' : 'full';
  el.helpBtn.setAttribute('aria-expanded', String(mode === 'full'));
}

function toggleHelp() {
  setHelp(app.help === 'full' ? (embedded && !app.touch ? 'compact' : 'hidden') : 'full');
}

// A mouse click leaves the button focused; Space would then press it again
// instead of pausing, so give the keyboard back to the page.
function onButton(button, action) {
  button.addEventListener('click', guard((e) => {
    action();
    if (e.detail > 0) button.blur();
  }));
}

// ------------------------------------------------------------------ worker output
let heldFrame = null; // arrived before the renderer existed

// The renderer is done with a frame (onRelease): its buffer goes back.
function giveBack(frame) {
  if (!frame || !app.worker || app.phase === 'failed') return;
  const buffer = frame.buffer;
  if (!buffer || buffer.byteLength === 0) return;
  app.worker.postMessage({ type: 'recycle', buffer }, [buffer]);
  app.frames.returned += 1;
}

function onFrame(frame) {
  app.frames.received += 1;
  app.root = Array.from(frame.root);
  if (!app.view) {
    giveBack(heldFrame);
    heldFrame = frame;
    return;
  }
  // A new epoch is a reset: the camera goes back to its view with the frame
  // that shows it (the first frame gets the default view anyway).
  if (frame.epoch !== app.epoch) {
    if (app.epoch !== null) app.view.resetView();
    app.epoch = frame.epoch;
  }
  app.view.setFrame(frame);
}

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

// The command pill: how to start until the first move, then the command as
// the session shapes it ('Starting…', 'Stopping…' while it holds a walk back,
// with a ring for a dot).
function showCommand(s) {
  if (s.moving) app.hasMoved = true;
  let text = s.command;
  if (s.falling) text = 'Lost balance';
  else if (!app.hasMoved) text = app.touch ? 'Hold ▲ to walk' : `Hold ${keyLabel('KeyW')} to walk`;
  setText(el.command, text);
  el.command.dataset.state = s.moving && !s.falling && !s.paused ? 'moving' : 'idle';
  el.command.dataset.shaping = s.shaping && !s.falling ? s.shaping : '';
}

function onStatus(s) {
  app.status = s;
  showCommand(s);
  setText(el.speed, s.speed);
  setText(el.clip, s.clip);
  el.clip.title = s.clipName;
  setText(el.time, `${s.time.toFixed(1)} s`);
  setText(el.rtf, s.rtf === null || s.rtf === undefined ? '–' : `${s.rtf.toFixed(2)}×`);
  setText(el.backend, s.backend);
  el.backend.title = s.adapter ? `${s.backend} (${s.adapter})` : (s.backendReason || s.backend);
  if (el.note) {
    el.note.hidden = s.backend !== 'CPU';
    if (s.backend === 'CPU') setText(el.noteText, cpuNote(s.backendReason));
  }
  if (app.level !== s.level) {
    app.level = s.level;
    refreshKbds();
  }
  if (debug) updateDebug(s);
  maybeCpuNote();
}

function onEvent(event) {
  app.events.push(event);
  if (app.events.length > 100) app.events.shift();
  switch (event.type) {
    case 'fall':
      toast('Lost balance — resetting…', 1400);
      break;
    case 'reset':
      // The camera resets with the new epoch's frame (onFrame).
      if (event.reason === 'fall') {
        if (!reducedMotion.matches && el.viewport.animate) {
          el.viewport.animate([{ opacity: 0.2 }, { opacity: 1 }], { duration: 400, easing: 'ease-out' });
        }
        toast('Back on its feet');
      }
      break;
    case 'level':
      app.level = event.level;
      refreshKbds();
      toast(`Speed level ${event.name}: ${event.label}`);
      break;
    case 'backend':
      toast('WebGPU stopped working, so inference moved to the CPU', 4000);
      break;
    default:
      break;
  }
}

function onWorkerMessage(event) {
  const m = event.data || {};
  switch (m.type) {
    case 'progress':
      progress.worker = m;
      updateLoader();
      break;
    case 'ready':
      app.info = m.info;
      app.workerReady = true;
      updateLoader();
      maybeReady();
      break;
    case 'frame':
      onFrame(m.frame);
      break;
    case 'status':
      onStatus(m.status);
      break;
    case 'event':
      onEvent(m.event);
      break;
    case 'error': {
      const message = m.message || '';
      // A damaged file first: its message quotes the decoder, which may say "fetch".
      let headline = 'The simulation stopped because of an error.';
      if (/corrupt|truncated|Not a DUETBIN1/i.test(message)) headline = 'One of the demo files arrived damaged. Reload the page to try again.';
      else if (/HTTP \d|fetch|network|load failed/i.test(message)) headline = 'The demo could not download its files. Check your connection, then reload the page.';
      fail(headline, message);
      break;
    }
    default:
      break;
  }
}

function startWorker() {
  let worker;
  try {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  } catch (error) {
    fail('This browser cannot start the simulation (module workers are not supported). Try a recent Chrome, Edge, Firefox or Safari.', error);
    return;
  }
  app.worker = worker;
  worker.onmessage = guard(onWorkerMessage);
  worker.onerror = (e) => {
    if (e.preventDefault) e.preventDefault();
    fail('The simulation could not start.', e.message || 'The worker script failed to load.');
  };
  worker.onmessageerror = () => fail('The simulation sent a message this page could not read.');
  worker.postMessage({ type: 'init', assetsBase: ASSETS.href, backend: requestedBackend });
  sentPaused = true; // the worker starts paused
}

// ------------------------------------------------------------------ render loop
// Every animation frame: the renderer updates and draws only if the picture
// changed; nothing at all while the frame is off screen or without a context.
// While paused the clock stands still (dt 0), so the camera stops easing with
// the walker and drawing stops at once; orbiting and zooming still draw.
let fpsFrames = 0;
let fpsSince = 0;

function renderLoop(t) {
  if (app.phase === 'failed') return;
  requestAnimationFrame(guardedRender);
  app.frames.rafs += 1;
  if (!app.offscreen && !app.contextLost && app.view.tick(pauseReasons.size > 0 ? 0 : undefined)) {
    app.frames.renders += 1;
    fpsFrames += 1;
  }
  if (!fpsSince) fpsSince = t;
  if (t - fpsSince >= 1000) {
    app.fps = (1000 * fpsFrames) / (t - fpsSince);
    fpsFrames = 0;
    fpsSince = t;
  }
}

const guardedRender = (t) => {
  try {
    renderLoop(t);
  } catch (error) {
    fail('The 3D view stopped because of an error.', error);
  }
};

// A lost WebGL context pauses the simulation; one that does not come back
// within CONTEXT_RESTORE_MS ends the demo with a message. (three.js keeps the
// context restorable and rebuilds its state when it returns.)
function watchContext(canvas) {
  canvas.addEventListener('webglcontextlost', guard(() => {
    app.contextLost = true;
    clearKeys();
    setPause('context', true);
    toast('The graphics context was lost; trying to restore it…', CONTEXT_RESTORE_MS);
    clearTimeout(contextTimer);
    contextTimer = setTimeout(() => fail('The 3D view lost its graphics context. Reload the page to try again.'), CONTEXT_RESTORE_MS);
  }));
  canvas.addEventListener('webglcontextrestored', guard(() => {
    clearTimeout(contextTimer);
    app.contextLost = false;
    setPause('context', false);
    app.view.invalidate();
    hideToast();
  }));
}

// ------------------------------------------------------------------ debug rows
let debugRows = null;

function updateDebug(s) {
  if (!debugRows) {
    const stats = el.speed.closest('dl');
    debugRows = {};
    for (const [key, label] of [['fps', 'Render'], ['step', 'Step'], ['falls', 'Falls']]) {
      const row = document.createElement('div');
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = label;
      row.append(dt, dd);
      stats.insertBefore(row, el.note || null);
      debugRows[key] = dd;
    }
  }
  const t = s.timing;
  setText(debugRows.fps, `${app.fps.toFixed(0)} fps`);
  setText(debugRows.step, `nn ${t.policy.toFixed(1)} · phys ${(t.physics + t.observe).toFixed(1)} ms`);
  setText(debugRows.falls, `${s.falls} (dropped ${(s.dropped * 0.01).toFixed(1)} s)`);
}

// ------------------------------------------------------------------ test hook
window.__duet = Object.freeze({
  state() {
    const s = app.status;
    return {
      phase: app.phase,
      ready: app.phase === 'ready' || app.phase === 'running',
      running: app.phase === 'running',
      paused: pauseReasons.size > 0,
      pauseReasons: [...pauseReasons],
      backend: s ? s.backend : null,
      backendReason: s ? s.backendReason : null,
      adapter: s ? s.adapter : null,
      rtf: s ? s.rtf : null,
      simTime: s ? s.time : null,
      steps: s ? s.steps : 0,
      falls: s ? s.falls : 0,
      resets: s ? s.resets : 0,
      falling: s ? s.falling : false,
      pose: s ? s.pose : null,
      root: app.root,
      command: s ? s.command : null,
      action: s ? s.action : null,
      shaping: s ? s.shaping : null,
      clip: s ? s.clip : null,
      clipName: s ? s.clipName : null,
      level: s ? s.level : null,
      keys: heldCodes(),
      camera: app.view ? app.view.cameraMode : null,
      cameraDistance: app.view ? app.view.cameraDistance : null,
      help: app.help,
      touch: app.touch,
      offscreen: app.offscreen,
      fps: app.fps,
      frames: { ...app.frames },
      error: app.error,
      hud: {
        command: el.command.textContent,
        speed: el.speed.textContent,
        clip: el.clip.textContent,
        time: el.time.textContent,
        rtf: el.rtf.textContent,
        backend: el.backend.textContent,
        note: el.note ? !el.note.hidden : false,
        noteText: el.noteText ? el.noteText.textContent : '',
      },
    };
  },
  stats() {
    const s = app.status;
    return {
      fps: app.fps,
      frames: { ...app.frames },
      render: app.view ? app.view.stats : null,
      timing: s ? s.timing : null,
      rtf: s ? s.rtf : null,
      dropped: s ? s.dropped : 0,
      load: app.info ? app.info.loadTimes : null,
      backend: app.info ? { human: app.info.backend, osl: app.info.oslBackend, adapter: app.info.adapter, reason: app.info.reason } : null,
    };
  },
  events() { return app.events.slice(); },
});

// ------------------------------------------------------------------ main
// An asset's URL, with the exporter's content hash as a query (demo.json
// "versions") so a new deploy never meets a stale cached copy.
function assetURL(demo, file) {
  const version = demo.versions && demo.versions[file];
  return new URL(version ? `${file}?v=${encodeURIComponent(version)}` : file, ASSETS);
}

async function main() {
  const problem = unsupported();
  if (problem) {
    fail(problem);
    return;
  }
  // The worker first, so the large downloads start at once; demo.json (for
  // the download size and the progress total), three.js and the render meshes
  // load meanwhile. demo.json is revalidated every time: it names the files.
  startWorker();
  if (app.phase === 'failed') return;
  const demoPromise = loadJSON(new URL('demo.json', ASSETS), { cache: 'no-cache' });
  demoPromise.then(guard((demo) => {
    progress.visual.total = demo.visual.bytes;
    showDownloadSize(demo);
    updateLoader();
  }), () => { /* reported by the await below */ });

  window.addEventListener('keydown', guard(onKeyDown));
  window.addEventListener('keyup', guard(onKeyUp));
  window.addEventListener('blur', guard(clearKeys));
  window.addEventListener('touchstart', guard(setTouch), { capture: true, passive: true, once: true });
  document.addEventListener('visibilitychange', guard(() => {
    if (document.hidden) clearKeys();
    setPause('hidden', document.hidden);
  }));
  if (embedded) {
    // Clicking into the iframe once the demo is ready starts it too.
    window.addEventListener('focus', guard(() => { if (app.phase === 'ready') start(); }));
    // A plain wheel scrolls the parent page: stopped here, before the canvas,
    // OrbitControls never sees it (and never cancels the scroll). Ctrl or ⌘ +
    // wheel, and a trackpad pinch (which arrives with ctrlKey), zoom.
    let zoomHintAt = -Infinity;
    el.viewport.addEventListener('wheel', guard((e) => {
      if (e.ctrlKey || e.metaKey) return;
      e.stopPropagation();
      if (app.phase === 'running' && performance.now() - zoomHintAt > ZOOM_HINT_MS) {
        zoomHintAt = performance.now();
        toast(`${ZOOM_KEY} + scroll to zoom`);
      }
    }), { capture: true });
    // Scrolled out of view: pause (an iframe is never document.hidden).
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(guard((entries) => {
        const visible = entries[entries.length - 1].isIntersecting;
        app.offscreen = !visible;
        if (!visible) clearKeys();
        setPause('offscreen', !visible);
      })).observe(el.viewport);
    }
  }
  for (const node of document.querySelectorAll('[data-zoom-key]')) node.textContent = ZOOM_KEY;
  if (debug) for (const row of document.querySelectorAll('.stats [data-debug]')) row.hidden = false;
  if (navigator.keyboard && typeof navigator.keyboard.getLayoutMap === 'function') {
    navigator.keyboard.getLayoutMap().then(guard((map) => relabel((code) => map.get(code))), () => { /* keep the fallback */ });
  }
  onButton(el.startBtn, start);
  onButton(el.reset, resetSim);
  onButton(el.pause, togglePause);
  onButton(el.camera, cycleCamera);
  onButton(el.helpBtn, toggleHelp);
  if (el.fast) onButton(el.fast, toggleFast);
  for (const button of padButtons) {
    button.addEventListener('pointerdown', guard((e) => padPress(button, e)));
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      button.addEventListener(type, guard(() => padRelease(button)));
    }
    button.addEventListener('contextmenu', (e) => e.preventDefault()); // long press
  }
  window.addEventListener('unhandledrejection', (e) => fail('The demo stopped because of an error.', e.reason));

  const [demo, { DemoRenderer }] = await Promise.all([demoPromise, import('./render/scene.js')]);
  if (app.phase === 'failed') return;
  useBindings(demo.ui && demo.ui.bindings);
  if (app.level === null) app.level = demo.ui.default_level;
  refreshKbds();
  progress.visual.total = demo.visual.bytes;
  const visual = await loadContainer(assetURL(demo, demo.visual.file), (received) => {
    progress.visual.loaded = Math.min(received, demo.visual.bytes);
    updateLoader();
  });
  progress.visual.loaded = demo.visual.bytes;
  if (app.phase === 'failed') return;

  try {
    app.view = new DemoRenderer(el.viewport, demo.visual, visual, { onRelease: giveBack });
  } catch (error) {
    fail('The 3D view could not start. This demo needs WebGL 2 with hardware acceleration.', error);
    return;
  }
  watchContext(app.view.renderer.domElement);
  el.camera.dataset.mode = app.view.cameraMode;
  if (heldFrame) {
    const frame = heldFrame;
    heldFrame = null;
    onFrame(frame);
    app.frames.received -= 1;
  }
  requestAnimationFrame(guardedRender);
  updateLoader();
  maybeReady();
}

main().catch((error) => fail('The demo could not load. Check your connection, then reload the page.', error));
