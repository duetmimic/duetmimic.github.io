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
// front-right. Start switches the camera to 'chase': directly behind the
// walker and slightly above its head. Embedded (?embed=1), the parent page focuses the iframe on its own
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
// Pulling: a press on the body (left button, or one finger) grabs the part
// under it and drags it: the worker pulls that point toward the cursor with a
// damped spring (MuJoCo's mouse perturbation), and the walker has to catch
// itself. A press on the background orbits as before. See "pulling" below.
//
// Rendering: the walker is drawn between the worker's per-step poses
// (render/scene.js), and only when something changed, so a paused or
// off-screen demo costs no GPU work.
//
// X-ray view (X, or #btn-xray in #viewbar): the muscles hidden, the bones and
// the socket see-through, the rest of the prosthesis as it is, and arrows for
// the force and moment the prosthesis applies to the residual limb at the
// socket (view.setXray). #xray-legend names them and gives their size, from
// the wrench as drawn, at most every LEGEND_MS. Running or paused; it stays
// on through resets, camera changes and pauses, not across reloads.
//
// Sensor charts (G, or #btn-sensors): #sensor-card, strip charts of what the
// prosthesis itself measures over the last 3 s (ui/sensors.js: IMU, ankle
// load cell, foot contact, encoders), and a marker on the leg at each sensor
// in the same colours (view.setSensorMarkers). Every frame's sensors go into
// the card's history on arrival; the card draws in the render loop at the
// play time of the pose on screen. It sits in the right column under the
// view toggles (under the X-ray legend when both are on), full or compact as
// there is room (on a phone compact across the top), or not at all (a toast
// says so). Where it or the legend would cover the walker, the view frames
// the walker beside or below them (frameWalker, view.setFraming).
// Independent of the X-ray view; the same rules otherwise.
//
// Query: ?backend=cpu|webgpu (default auto), ?debug=1 (extra HUD rows),
// ?xray=1 (start in the X-ray view), ?sensors=1 (start with the sensor charts).
// Test hook: window.__duet.state(), .stats(), .events(), for pulling
// .pick(clientX, clientY), .project([x, y, z]), .geomPoint(geom, local?), for
// the X-ray view .xray() (the arrows as drawn), for the sensor charts
// .sensors() (the card and the markers as drawn), and .help(mode?) (ask for a
// help mode, then lay the panels out as a resize does; returns the mode shown).

import { loadContainer, loadJSON } from './lib/container.js';
import { SensorCard } from './ui/sensors.js';

const ASSETS = new URL('../assets/stage2/', import.meta.url);
// Bytes not listed in demo.json, for the download note: vendor/mujoco/mujoco.wasm.gz
// plus three.js, mujoco.js and the demo's own scripts, styles and demo.json as
// GitHub Pages sends them (gzip). Measured 2026-11-16 (31.2 MB in all with the
// R2 bundle's assets; 31.1 MB with bundle B's, 2026-11-18).
const UNLISTED_BYTES = 2475508 + 577000;
// Camera modes in the order C cycles them, and their names in the toast.
const CAMERA_MODES = ['chase', 'follow', 'free'];
const CAMERA_NAMES = {
  chase: 'behind the walker',
  follow: 'fixed angle (drag the background to orbit)',
  free: 'free (drag the background to orbit, right-drag to pan)',
};
const CAMERA_NAMES_TOUCH = { ...CAMERA_NAMES, free: 'free (drag the background to orbit, two fingers to pan)' };
const TOAST_MS = 1500;
const LEGEND_MS = 100; // the X-ray legend's numbers change at most this often
const SUMMARY_MS = 1000; // the sensor card's text for screen readers, at most this often (and on a pause)
const PANEL_GAP = 6; // px: the help yields to the X-ray legend and the sensor card closer than this
const COLUMN_GAP = 8; // px between the panels of the right column (view toggles, legend, card)
const TOAST_BESIDE_MIN = 240; // px: a toast narrower than this beside the right column goes under it
const NO_ROOM_CARD = 'No room for the sensor charts here';
const NO_ROOM_HELP = 'No room for the help here';
const NO_ROOM_FULL_HELP = 'No room for the full help here';
const TOAST_TALL = 52; // px: a two-line toast, for keeping it clear of the touch pad
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
  topbar: document.querySelector('#hud .topbar'),
  stats: document.querySelector('#hud .stats'),
  back: $('back-link'),
  viewbar: $('viewbar'),
  toast: $('toast'),
  reset: $('btn-reset'),
  pause: $('btn-pause'),
  camera: $('btn-camera'),
  helpBtn: $('btn-help'),
  xrayBtn: $('btn-xray'),
  xrayLegend: $('xray-legend'),
  xrayForce: $('xray-force'),
  xrayMoment: $('xray-moment'),
  sensorsBtn: $('btn-sensors'),
  sensorCard: $('sensor-card'),
  sensorSummary: $('sensor-summary'),
  pad: $('touch-pad'),
  fast: $('pad-fast'),
  error: $('error'),
  errorText: $('error-text'),
};

const params = new URLSearchParams(location.search);
const embedded = el.root.classList.contains('embed');
const flag = (name) => params.has(name) && !['0', 'false'].includes(params.get(name));
const debug = flag('debug');
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
  help: 'hidden', // full | compact | hidden: the mode asked for
  helpShown: 'hidden', // the mode shown (fitHelp: the X-ray legend may leave less room)
  xray: false, // the X-ray view
  sensors: false, // the sensor charts and markers
  card: null, // the SensorCard (once demo.json is in)
  cardMode: 'off', // off | full | compact | none (wanted, but no room: hidden)
  framing: null, // {scale, dx, dy, zone}: the walker moved clear of the right column (frameWalker), or null
  walker: null, // the walker's zone on screen (client px), framed or not
  walkerCovered: false, // a view panel covers the walker and there is no room to frame it clear
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
  try { endDrag(false); } catch { /* the view is gone too */ }
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
const LABELLED = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyR', 'KeyC', 'KeyH', 'KeyX', 'KeyG'];
// Common layouts that differ from QWERTY at those positions: one key from the
// fallback identifies the layout and so labels the others too.
const LAYOUTS = [
  { KeyW: 'z', KeyA: 'q', KeyS: 's', KeyD: 'd', KeyQ: 'a', KeyE: 'e', KeyR: 'r', KeyC: 'c', KeyH: 'h', KeyX: 'x', KeyG: 'g' }, // AZERTY
  { KeyW: ',', KeyA: 'a', KeyS: 'o', KeyD: 'e', KeyQ: "'", KeyE: '.', KeyR: 'p', KeyC: 'j', KeyH: 'd', KeyX: 'q', KeyG: 'i' }, // Dvorak
  { KeyW: 'w', KeyA: 'a', KeyS: 'r', KeyD: 's', KeyQ: 'q', KeyE: 'f', KeyR: 'p', KeyC: 'c', KeyH: 'h', KeyX: 'x', KeyG: 'd' }, // Colemak
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
  layoutHud();
}

// Texts that name keys or depend on touch mode: [data-label] elements and the
// buttons' titles ([data-title] + the label of [data-title-key]).
function refreshTexts() {
  for (const node of document.querySelectorAll('[data-label]')) node.textContent = keyLabel(node.dataset.label);
  for (const node of document.querySelectorAll('[data-title][data-title-key]')) node.title = `${node.dataset.title} (${keyLabel(node.dataset.titleKey)})`;
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
    case 'KeyX': toggleXray(); break;
    case 'KeyG': toggleSensors(); break;
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
  endDrag();
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
// touch screens. That is the mode asked for (app.help). A help panel that
// would overlap the X-ray legend, the sensor card, the view toggles or the
// top bar (short windows, the project page's iframe on a phone), or, on a
// keyboard, cover the walker where the view panels have moved it (narrow
// windows), shows less until there is room again (fitHelp; app.helpShown):
// with a keyboard the one-line legend, at the bottom right or left of the
// right column, wrapping there; on touch screens nothing. H, ? and the ?
// button toggle what is on screen: the full panel goes back to the start
// mode, anything less asks for the full panel. Where that changes nothing
// on screen, the one-line legend goes instead (no room for the panel), or a
// toast says there is no room for any help.
function setHelp(mode) {
  app.help = mode;
  fitHelp();
}

function toggleHelp() {
  const shown = app.helpShown;
  setHelp(shown === 'full' ? (embedded && !touchMode() ? 'compact' : 'hidden') : 'full');
  if (app.helpShown !== shown) return;
  if (shown !== 'hidden') { // the legend on screen is all that fits: H takes it away
    setHelp('hidden');
    toast(NO_ROOM_FULL_HELP);
  } else {
    toast(NO_ROOM_HELP);
  }
}

// The help as shown: 'full', 'compact' (the one-line legend), 'beside' (that
// legend left of the right column: class "beside", it wraps there) or
// 'hidden'. app.helpShown is 'compact' for both legend placements.
function showHelp(mode) {
  const compact = mode === 'compact' || mode === 'beside';
  app.helpShown = compact ? 'compact' : mode;
  el.help.hidden = mode === 'hidden';
  el.help.dataset.mode = compact ? 'compact' : 'full';
  el.help.classList.toggle('beside', mode === 'beside');
  el.helpBtn.setAttribute('aria-expanded', String(mode === 'full'));
}

const isShown = (node) => !!node && !node.hidden && getComputedStyle(node).display !== 'none';
const rectOf = (node) => (isShown(node) ? node.getBoundingClientRect() : null);
const crosses = (a, b, gap = 0) => a.left < b.right + gap && b.left < a.right + gap && a.top < b.bottom + gap && b.top < a.bottom + gap;
const touchMode = () => app.touch || el.root.classList.contains('touch');
// The page's edge margin (--edge, px): read once per layout pass.
const edgeOf = () => parseFloat(getComputedStyle(el.root).getPropertyValue('--edge')) || 16;

// The help mode shown for the one asked for (see setHelp): the first of the
// tries that crosses none of the panels around it (the X-ray legend, the
// sensor card, the view toggles, the top bar and, with a keyboard, the
// stats; demo.css keeps it clear of those by its width where both sit at
// the bottom) and, a
// keyboard's full panel, not the walker where the panels have framed it
// elsewhere (a narrow window with the card across its top would otherwise
// hide all but its head).
function fitHelp() {
  const touch = touchMode();
  const legend = touch ? [] : ['compact', 'beside']; // the one-line legend: keyboards only
  const tries = app.help === 'hidden' ? ['hidden']
    : app.help === 'full' ? ['full', ...legend, 'hidden']
      : app.help === 'compact' ? [...(touch ? ['compact'] : legend), 'hidden'] : [app.help, 'hidden'];
  const clear = [el.xrayLegend, el.sensorCard, el.viewbar, el.topbar, touch ? null : el.stats].map((node) => [rectOf(node), node]).filter(([r]) => r);
  const gap = (node) => (node === el.xrayLegend || node === el.sensorCard ? PANEL_GAP : 0);
  const column = clear.some(([, node]) => node === el.xrayLegend || node === el.sensorCard); // 'beside' needs one
  const plan = !touch && app.help === 'full' ? planFraming() : null;
  const walker = plan && plan.framing ? plan.framing.zone : null;
  for (const mode of tries) {
    if (mode === 'beside' && !column) continue;
    showHelp(mode);
    if (mode === 'hidden') return;
    const r = el.help.getBoundingClientRect();
    if (clear.some(([q, node]) => crosses(r, q, gap(node)))) continue;
    if (mode === 'full' && walker && crosses(r, walker)) continue;
    return;
  }
}

// The sensor card (html.sensors: the charts are on), in the right column:
// under the view toggles, and under the top bar where that spans the card's
// column and the X-ray legend when it is at the top. Its variants, in this
// order: full; compact across the width (portrait windows: the walker is
// framed below it, see frameWalker); compact. Each must fit above the bottom
// of the window, clear of the panels below it (the touch pad, the stats, the
// legend beside the pad, the back link), and leave the walker room
// (planFraming); the first that keeps the walker within FRAME_PREFER of the
// largest scale any of them allows wins (on a phone the card across the width,
// with plots twice as wide and the walker centred under it, rather than at
// the right with the walker squeezed into the strip beside it). Touch phones (the shorter side under 600 px) and
// narrow portrait windows get no full card. Hidden where none fits, with a
// toast when it was on screen until now (app.cardMode 'none'), and back when
// there is room again.
const FRAME_PREFER = 0.12; // of the walker's scale: an earlier variant may cost this much

function placeCard(edge = edgeOf()) {
  const card = el.sensorCard;
  if (!card) return;
  const before = app.cardMode;
  if (!el.root.classList.contains('sensors')) {
    card.hidden = true;
    app.cardMode = 'off';
    if (app.card) app.card.setVisible(false);
    updateSummary(0, true);
    return;
  }
  card.hidden = false;
  const wasWide = card.classList.contains('wide');
  const W = innerWidth, H = innerHeight;
  const legend = rectOf(el.xrayLegend);
  const legendTop = !!legend && legend.top < H / 2;
  const above = [el.viewbar, el.topbar].map(rectOf).filter(Boolean);
  if (legendTop) above.push(legend);
  const obstacles = [el.back, el.stats, el.pad, el.topbar, el.viewbar].map(rectOf).filter(Boolean);
  if (legend) obstacles.push(legend);
  const phone = (touchMode() && Math.min(W, H) < 600) || (W < 480 && H >= W);
  const variants = [phone ? null : ['full', false], H > W ? ['compact', true] : null, ['compact', false]].filter(Boolean);
  const put = ([mode, wide]) => {
    card.dataset.mode = mode;
    card.classList.toggle('wide', wide);
    card.style.top = '0px';
    const box = card.getBoundingClientRect();
    let top = 0;
    for (const r of above) {
      if (r.left < box.right && box.left < r.right) top = Math.max(top, r.bottom + COLUMN_GAP);
    }
    card.style.top = `${Math.ceil(top)}px`;
    const placed = card.getBoundingClientRect();
    return placed.left >= edge - 0.5 && placed.bottom <= H - edge + 0.5 && !obstacles.some((r) => crosses(placed, r, COLUMN_GAP - 2));
  };
  const fits = [];
  for (const v of variants) {
    if (!put(v)) continue;
    const plan = planFraming(edge);
    const scale = plan.covered ? 0 : plan.framing ? plan.framing.scale : 1;
    if (scale > 0) fits.push({ v, scale });
  }
  const largest = Math.max(0, ...fits.map((f) => f.scale));
  const best = fits.find((f) => f.scale >= largest - FRAME_PREFER) || null;
  const mode = best ? best.v[0] : 'none';
  if (best) put(best.v);
  app.cardMode = mode;
  card.hidden = mode === 'none';
  if (app.card) {
    if (mode !== 'none') app.card.setMode(mode);
    if (card.classList.contains('wide') !== wasWide) app.card.invalidate(); // another width, the same mode
    app.card.setVisible(app.sensors && mode !== 'none');
  }
  updateSummary(0, true);
  if (mode === 'none' && (before === 'full' || before === 'compact') && app.sensors && app.phase === 'running') toast(NO_ROOM_CARD);
}

// The toast keeps clear of the right column's panels at the top (the X-ray
// legend there, the sensor card): centred and narrower beside them
// (html.toast-beside) on windows wider than 800px, else left-aligned beside
// them (html.toast-left), each where that leaves it TOAST_BESIDE_MIN or more;
// otherwise under them (html.toast-below), and there right of the touch pad
// where the pad comes up to it (html.toast-pad, --toast-left: a phone-sized
// embed). --column-right is the column's distance from the window's right
// edge, --column-bottom its bottom (px).
function layoutToast(edge = edgeOf()) {
  const legend = rectOf(el.xrayLegend);
  const column = [legend && legend.top < innerHeight / 2 ? legend : null, rectOf(el.sensorCard)].filter(Boolean);
  el.root.classList.remove('toast-beside', 'toast-left', 'toast-below', 'toast-pad');
  if (!column.length) return;
  const right = Math.ceil(Math.max(...column.map((r) => innerWidth - r.left)));
  const bottom = Math.ceil(Math.max(...column.map((r) => r.bottom)));
  const centred = innerWidth - 2 * (right + COLUMN_GAP), left = innerWidth - right - COLUMN_GAP - edge;
  const mode = innerWidth > 800 && centred >= TOAST_BESIDE_MIN ? 'beside' : left >= TOAST_BESIDE_MIN ? 'left' : 'below';
  el.root.classList.add(`toast-${mode}`);
  el.root.style.setProperty('--column-right', `${right}px`);
  el.root.style.setProperty('--column-bottom', `${bottom}px`);
  const pad = mode === 'below' ? rectOf(el.pad) : null;
  const top = bottom + COLUMN_GAP;
  if (pad && pad.top < top + TOAST_TALL + COLUMN_GAP && pad.bottom > top) {
    el.root.classList.add('toast-pad');
    el.root.style.setProperty('--toast-left', `${Math.ceil(pad.right + COLUMN_GAP)}px`);
  }
}

// Framing the walker. With the default camera (render/scene.js: the chase
// camera at its default distance, and the hero view) the walker fills about
// WALKER_BAND of the window's height, centred, WALKER_HALF_WIDTH x the height
// to either side (measured walking, arms swinging; 2026-11-19). Where the
// view panels cross that zone (the sensor card or the X-ray legend in the
// right column at the top: portrait phones and tablets, narrow windows; the
// legend beside the touch pad on short phones), the view is moved and, if it
// must, shrunk (view.setFraming) into the free area that keeps it largest:
// left of the column or below it, or above the legend at the bottom; always
// under the top bar and above the touch pad. Elsewhere nothing changes.
const WALKER_BAND = [0.17, 0.88];
const WALKER_HALF_WIDTH = 0.16;
const FRAME_MIN_SCALE = 0.4; // smaller than this, the walker is left where it is

// The walker's zone on screen (client px) at scale s about (cx, cy).
function walkerZone(W, H, s = 1, cx = W / 2, cy = (H * (WALKER_BAND[0] + WALKER_BAND[1])) / 2) {
  const hw = WALKER_HALF_WIDTH * H * s, hh = ((WALKER_BAND[1] - WALKER_BAND[0]) * H * s) / 2;
  return { left: cx - hw, top: cy - hh, right: cx + hw, bottom: cy + hh };
}

// Where the walker goes with the panels as they are now (client px): {framing
// ({scale, dx, dy, zone} | null), covered (a panel over it and no room to
// frame it clear), natural (its zone unframed)}.
function planFraming(edge = edgeOf()) {
  const W = innerWidth, H = innerHeight;
  const natural = walkerZone(W, H);
  const legend = rectOf(el.xrayLegend);
  const legendTop = !!legend && legend.top < H / 2;
  const column = [rectOf(el.sensorCard), legendTop ? legend : null].filter(Boolean); // the right column, at the top
  const below = legend && !legendTop ? legend : null; // the legend beside the touch pad
  const over = column.some((r) => crosses(r, natural)) || (!!below && crosses(below, natural));
  let framing = null;
  if (over) {
    // An area starts under the top bar, the back link and the view toggles
    // above it, and ends over the touch pad and the legend at the bottom.
    const tops = [el.topbar, el.back, el.viewbar].map(rectOf).filter(Boolean);
    const ceiling = (left, right) => Math.max(0, ...tops.filter((r) => r.left < right && left < r.right).map((r) => r.bottom + COLUMN_GAP));
    let floor = H - edge;
    const pad = rectOf(el.pad);
    if (pad) floor = Math.min(floor, pad.top - COLUMN_GAP);
    if (below) floor = Math.min(floor, below.top - COLUMN_GAP);
    const columnLeft = column.length ? Math.min(...column.map((r) => r.left)) - COLUMN_GAP : W - edge;
    const areas = column.length ? [
      { left: edge, top: ceiling(edge, columnLeft), right: columnLeft, bottom: floor }, // left of the column
      { left: edge, top: Math.max(ceiling(edge, W - edge), ...column.map((r) => r.bottom + COLUMN_GAP)), right: W - edge, bottom: floor }, // below it
    ] : [{ left: edge, top: ceiling(edge, W - edge), right: W - edge, bottom: floor }]; // above the legend at the bottom
    const cx = (natural.left + natural.right) / 2, cy = (natural.top + natural.bottom) / 2;
    const hw = (natural.right - natural.left) / 2, hh = (natural.bottom - natural.top) / 2;
    const clamp = (v, lo, hi) => (lo > hi ? (lo + hi) / 2 : Math.min(hi, Math.max(lo, v)));
    let move = Infinity;
    for (const a of areas) {
      const s = Math.min(1, (a.right - a.left) / (2 * hw), (a.bottom - a.top) / (2 * hh));
      if (!(s >= FRAME_MIN_SCALE)) continue;
      const x = clamp(cx, a.left + s * hw, a.right - s * hw), y = clamp(cy, a.top + s * hh, a.bottom - s * hh);
      const d = Math.hypot(x - cx, y - cy);
      if (!framing || s > framing.scale + 1e-6 || (Math.abs(s - framing.scale) <= 1e-6 && d < move)) {
        // The zone's centre (cx, cy) goes to (x, y): c = s v + (1 - s) centre + d.
        framing = { scale: s, dx: x - s * cx - (1 - s) * (W / 2), dy: y - s * cy - (1 - s) * (H / 2), zone: walkerZone(W, H, s, x, y) };
        move = d;
      }
    }
  }
  return { framing, covered: over && !framing, natural };
}

function frameWalker(edge = edgeOf()) {
  const plan = planFraming(edge);
  app.framing = plan.framing;
  app.walker = plan.framing ? plan.framing.zone : plan.natural;
  app.walkerCovered = plan.covered; // the smallest embeds, under the X-ray legend
  if (app.view) app.view.setFraming(app.framing);
}

// The panels around the right column: the sensor card is placed under the
// view toggles and the X-ray legend, the toast keeps clear of both and the
// help fits beside them; the walker is framed clear of them. Run on every
// change of the legend, the card, the help, the touch mode and the window
// size.
function layoutHud() {
  const edge = edgeOf();
  placeCard(edge);
  layoutToast(edge);
  fitHelp();
  frameWalker(edge);
}

// ------------------------------------------------------------------ X-ray view
// On or off: the renderer (when there is one), the button, the legend and the
// panels around it; a toast unless quiet (?xray=1 at load).
function setXray(on, quiet = false) {
  app.xray = !!on;
  el.root.classList.toggle('xray', app.xray);
  if (el.xrayBtn) el.xrayBtn.setAttribute('aria-pressed', String(app.xray));
  if (el.xrayLegend) el.xrayLegend.hidden = !app.xray;
  if (app.view) app.view.setXray(app.xray);
  updateLegend(0, true);
  layoutHud();
  if (!quiet) toast(app.xray ? 'X-ray view on' : 'X-ray view off');
}

function toggleXray() {
  if (app.phase !== 'running' || !app.view) return;
  setXray(!app.xray);
}

// The legend's numbers: |F| (N) and |T| (N m) of the wrench as drawn, at most
// every LEGEND_MS (now: a requestAnimationFrame time), or at once with force.
let legendAt = -Infinity;

function updateLegend(now, force = false) {
  if (!app.xray || !el.xrayForce) return;
  if (!force && now - legendAt < LEGEND_MS) return;
  legendAt = force ? -Infinity : now;
  const w = app.view ? app.view.wrench : null;
  // No wrench (an older worker, the reset state before its first step) or a
  // non-finite one: a dash.
  const size = (a, b, c) => { const n = Math.hypot(a, b, c); return Number.isFinite(n) ? String(Math.round(n)) : '–'; };
  setText(el.xrayForce, w ? size(w[0], w[1], w[2]) : '–');
  setText(el.xrayMoment, w ? size(w[3], w[4], w[5]) : '–');
}

// ------------------------------------------------------------------ sensor charts
// On or off: the markers (when there is a renderer), the button, the card and
// the panels around it; a toast unless quiet (?sensors=1 at load), which says
// so where the card has no room.
function setSensors(on, quiet = false) {
  app.sensors = !!on;
  el.root.classList.toggle('sensors', app.sensors);
  if (el.sensorsBtn) el.sensorsBtn.setAttribute('aria-pressed', String(app.sensors));
  if (app.view) app.view.setSensorMarkers(app.sensors);
  layoutHud(); // places the card and shows it (app.card.setVisible) where it fits
  if (!quiet) toast(!app.sensors ? 'Sensor charts off' : app.cardMode === 'none' ? NO_ROOM_CARD : 'Sensor charts on');
}

function toggleSensors() {
  if (app.phase !== 'running' || !app.view) return;
  setSensors(!app.sensors);
}

// #sensor-summary: the card's latest values as text for screen readers (the
// charts are canvases), at most every SUMMARY_MS, at once with force (the
// card shown or hidden, a drawing of a paused pose); empty while the card is
// not on screen. Not a live region: read when the visitor gets there (or
// focuses #btn-sensors, which it describes).
let summaryAt = -Infinity;

function updateSummary(now, force = false) {
  if (!el.sensorSummary) return;
  if (!app.card || !app.card.visible) {
    summaryAt = -Infinity;
    setText(el.sensorSummary, '');
    return;
  }
  if (!force && now - summaryAt < SUMMARY_MS) return;
  summaryAt = force ? -Infinity : now;
  setText(el.sensorSummary, app.card.summary());
}

// A mouse click leaves the button focused; Space would then press it again
// instead of pausing, so give the keyboard back to the page.
function onButton(button, action) {
  button.addEventListener('click', guard((e) => {
    action();
    if (e.detail > 0) button.blur();
  }));
}

// ------------------------------------------------------------------ pulling
// Click (or touch) and drag a body part to pull it. The worker applies MuJoCo's
// mouse perturbation at every physics tick: a damped spring, scaled by the
// mass the grabbed point moves, from that point to a target (world
// coordinates). Messages: {type: 'perturb', geom, local, target, id} starts or
// moves the pull (id numbers the drags; the worker echoes it in status() and
// 'perturb-end'), {type: 'perturb', release: true} ends it.
//  - pointerdown (capture phase on #viewport, so a hit never reaches
//    OrbitControls on the canvas): view.pick(). On the body: capture the
//    pointer, switch OrbitControls off and hold the camera (view.holdCamera:
//    a camera following the pulled walker would carry the target along, so a
//    pull held still would never let up), post the start (target = the
//    grabbed point: no force yet). On the background: nothing, OrbitControls
//    orbits.
//  - Every animation frame while dragging, after the view's update: the target
//    is view.dragTarget() for the last pointer position with this frame's
//    camera (under the cursor, at the grabbed point's depth); posted when it
//    moved (at most once per frame) and drawn (view.setPerturb).
//  - Release on pointerup, pointercancel, lostpointercapture, a mouse move with
//    the left button up (a release the page never saw), window blur, a
//    hidden tab, the frame scrolled away, a lost WebGL context, a reset, or the
//    worker's 'perturb-end' event for this drag's id (the worker ended it: a
//    reset, a fall, an invalid request; one without an id is a request that
//    never started). The worker ignores target updates still on their way for
//    a drag it ended; the release posted then anyway costs nothing.
//  - Hover: at most one pick per animation frame; the cursor shows 'grab' over
//    the body and 'grabbing' during a drag.
// Mouse: the left button. Touch: a finger alone on the canvas. A finger on the
// touch pad does not count, so one thumb can walk while the other pulls (the
// pad finger is the primary pointer then, so isPrimary cannot decide); a
// second finger on the canvas makes a pinch, and does nothing during a drag.
// Only while running; allowed while paused, when the pull acts on resuming.
const TARGET_EPS = 1e-4; // m: smaller moves of the target are not posted
const PICK_SLOP = 4; // CSS px: a press this close to the body still grabs it (view.pick)
const PICK_SLOP_TOUCH = 12; // a fingertip
let grab = null; // the drag: {id, pointerId, pointerType, geom, body, local, x, y, target, sent}
let dragCount = 0; // drags so far: the next id
const canvasTouches = new Set(); // touch pointers down on the canvas (a drag, an orbit, a pinch)
const hover = { x: 0, y: 0, pending: false, cursor: '' };
const pulls = { started: 0, updates: 0, released: 0, ended: 0 }; // perturb messages posted, perturb-end events honoured (test hook)

const viewCanvas = () => (app.view ? app.view.renderer.domElement : null);

function setCursor(cursor) {
  const canvas = viewCanvas();
  if (!canvas || hover.cursor === cursor) return;
  hover.cursor = cursor;
  canvas.style.cursor = cursor;
}

function postPerturb(message) {
  if (app.worker && app.phase !== 'failed') app.worker.postMessage({ type: 'perturb', ...message });
}

function onPointerDown(e) {
  const canvas = viewCanvas();
  if (!canvas || e.target !== canvas) return;
  if (e.pointerType === 'touch') canvasTouches.add(e.pointerId);
  if (grab) return;
  if (app.phase !== 'running' || app.offscreen || app.contextLost) return;
  if (e.button !== 0 || (e.pointerType === 'touch' ? canvasTouches.size > 1 : !e.isPrimary)) return;
  const hit = app.view.pick(e.clientX, e.clientY, e.pointerType === 'touch' ? PICK_SLOP_TOUCH : PICK_SLOP);
  if (!hit) {
    setCursor('');
    return; // the background: OrbitControls orbits
  }
  e.stopPropagation(); // OrbitControls (on the canvas) never sees this press
  try { canvas.setPointerCapture(e.pointerId); } catch { /* the pointer is already gone */ }
  app.view.setControlsEnabled(false);
  app.view.holdCamera(true);
  dragCount += 1;
  grab = {
    id: dragCount, pointerId: e.pointerId, pointerType: e.pointerType, geom: hit.geom, body: hit.body, local: hit.local,
    x: e.clientX, y: e.clientY, target: hit.point, sent: hit.point,
  };
  postPerturb({ geom: hit.geom, local: hit.local, target: hit.point, id: grab.id });
  pulls.started += 1;
  app.view.setPerturb({ geom: hit.geom, local: hit.local, target: hit.point, force: 0 });
  setCursor('grabbing');
}

function onPointerMove(e) {
  if (grab) {
    if (e.pointerId === grab.pointerId) {
      // The left button is up, yet no pointerup came: it was released where
      // the page could not see it (outside the window in Playwright's
      // Firefox, for one). Let go now rather than pull on.
      if (e.pointerType === 'mouse' && (e.buttons & 1) === 0) {
        endDrag();
        return;
      }
      grab.x = e.clientX;
      grab.y = e.clientY;
    }
    return;
  }
  if (e.pointerType === 'touch' || e.buttons !== 0) return; // no hover; orbiting
  hover.x = e.clientX;
  hover.y = e.clientY;
  hover.pending = true;
}

function onPointerEnd(e) {
  if (grab && e.pointerId === grab.pointerId) endDrag();
}

// End the drag on this side; post the release unless the worker is gone.
function endDrag(notify = true) {
  if (!grab) return;
  const g = grab;
  grab = null; // first: releasing the capture below fires lostpointercapture
  const canvas = viewCanvas();
  try {
    if (canvas && canvas.hasPointerCapture(g.pointerId)) canvas.releasePointerCapture(g.pointerId);
  } catch { /* the pointer is gone */ }
  if (app.view) {
    app.view.setControlsEnabled(true);
    app.view.holdCamera(false);
    app.view.setPerturb(null);
  }
  if (notify) {
    postPerturb({ release: true });
    pulls.released += 1;
  }
  setCursor('');
  if (g.pointerType !== 'touch') { // still over the body? The next frame's hover pick says
    hover.x = g.x;
    hover.y = g.y;
    hover.pending = true;
  }
}

// Per animation frame, after view.update(): the drag's target, or the hover cursor.
function pointerFrame() {
  if (grab) {
    const target = app.view.dragTarget(grab.x, grab.y, grab);
    if (!target) {
      endDrag();
      return;
    }
    grab.target = target;
    const s = grab.sent;
    if (Math.abs(target[0] - s[0]) > TARGET_EPS || Math.abs(target[1] - s[1]) > TARGET_EPS || Math.abs(target[2] - s[2]) > TARGET_EPS) {
      postPerturb({ geom: grab.geom, local: grab.local, target, id: grab.id });
      grab.sent = target;
      pulls.updates += 1;
    }
    // The line's width: the force the worker last reported for this drag.
    const p = app.status && app.status.perturb;
    const mine = p && p.active && (p.id === undefined || p.id === null || p.id === grab.id);
    const force = mine && Number.isFinite(p.force) ? p.force : 0;
    app.view.setPerturb({ geom: grab.geom, local: grab.local, target, force, maxForce: p ? p.maxForce : undefined });
  } else if (hover.pending) {
    hover.pending = false;
    setCursor(app.phase === 'running' && !app.offscreen && app.view.pick(hover.x, hover.y, PICK_SLOP) ? 'grab' : '');
  }
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
  // The sensor history copies the frame's sensors now: setFrame may hand the
  // frame's buffer back to the worker at once. Kept whether the card is
  // shown or not, so it opens with the last seconds.
  if (app.card) app.card.push(frame);
  if (!app.view) {
    giveBack(heldFrame);
    heldFrame = frame;
    return;
  }
  // A new epoch is a reset: the camera goes back to its view with the frame
  // that shows it (the first frame gets the default view anyway), and a pull
  // on the old pose ends.
  if (frame.epoch !== app.epoch) {
    if (app.epoch !== null) {
      app.view.resetView();
      endDrag();
    }
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
    case 'perturb-end':
      // The worker let go (reset, fall, invalid request) of this drag, or of a
      // request that never started (no id); not of an older drag.
      if (grab && (event.id === undefined || event.id === null || event.id === grab.id)) {
        pulls.ended += 1;
        endDrag();
      }
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
// Every animation frame: the renderer updates, the drag's target follows the
// pointer with this frame's camera (or the hover cursor is checked), and the
// renderer draws only if the picture changed; nothing at all while the frame
// is off screen or without a context. While paused the clock stands still
// (dt 0), so the camera stops easing with the walker and drawing stops at
// once; orbiting and zooming still draw. The sensor card draws after the
// view, at the play time of the pose just drawn (when shown, at most 30
// times a second, only once that time has moved and new data has come).
let fpsFrames = 0;
let fpsSince = 0;

function renderLoop(t) {
  if (app.phase === 'failed') return;
  requestAnimationFrame(guardedRender);
  app.frames.rafs += 1;
  if (!app.offscreen && !app.contextLost) {
    app.view.update(pauseReasons.size > 0 ? 0 : undefined);
    pointerFrame();
    if (app.view.drawIfChanged()) {
      app.frames.renders += 1;
      fpsFrames += 1;
    }
    updateLegend(t);
    if (app.card && app.card.visible) {
      const settled = app.card.counts.settled;
      if (app.card.draw(app.view.playTime, t)) updateSummary(t, app.card.counts.settled !== settled);
    }
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
    endDrag();
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

// Pulling (see "pulling" above). The press is caught on #viewport in the
// capture phase, before the canvas and so before OrbitControls; the rest of
// the drag comes to the canvas, which holds the pointer capture.
function watchPointer(canvas) {
  el.viewport.addEventListener('pointerdown', guard(onPointerDown), { capture: true });
  for (const type of ['pointerup', 'pointercancel']) { // wherever the finger ends up
    window.addEventListener(type, (e) => { canvasTouches.delete(e.pointerId); }, { capture: true });
  }
  canvas.addEventListener('pointermove', guard(onPointerMove));
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(type, guard(onPointerEnd));
  }
  canvas.addEventListener('pointerleave', guard(() => {
    hover.pending = false;
    if (!grab) setCursor('');
  }));
  canvas.addEventListener('contextmenu', (e) => { if (grab) e.preventDefault(); });
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
      // Pulling: this side's drag, the perturb messages posted so far, and the
      // worker's view of it (status().perturb: {active, geom, body, id, force,
      // maxForce, stretch}).
      perturb: {
        active: grab !== null,
        id: grab ? grab.id : null,
        geom: grab ? grab.geom : null,
        body: grab ? grab.body : null,
        local: grab ? grab.local.slice() : null,
        target: grab ? grab.target.slice() : null,
        ...pulls,
        worker: s && s.perturb ? { ...s.perturb } : null,
      },
      cursor: hover.cursor,
      camera: app.view ? app.view.cameraMode : null,
      cameraDistance: app.view ? app.view.cameraDistance : null,
      cameraAzimuth: app.view ? app.view.cameraAzimuth : null,
      help: app.helpShown, // as shown; helpWanted: as asked for (H, the button, Start)
      helpWanted: app.help,
      xray: app.xray,
      sensors: app.sensors,
      sensorCard: app.cardMode, // off | full | compact | none (on, but no room)
      // The walker framed clear of the right column (portrait phones and
      // tablets): {scale, dx, dy, zone (client px)}, or null.
      framing: app.framing ? { ...app.framing, zone: { ...app.framing.zone } } : null,
      walker: app.walker ? { ...app.walker } : null, // the zone the layout keeps clear (framed or not)
      walkerCovered: app.walkerCovered, // no room to frame it clear of the panels over it
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
      // The sensor card: draws, static redraws, draws skipped by the rate
      // limit, settled draws (the play time stood still: a pause), draw time
      // (ms: last, mean, p95, max of the last 128), the history held
      // (samples, span s, epoch), frames with and without sensors.
      sensors: app.card ? app.card.stats() : null,
    };
  },
  events() { return app.events.slice(); },
  // Pulling helpers: what a press at a client point would grab, a world point
  // on screen (client coordinates), a drawn geom's point in the world.
  pick(clientX, clientY, slop = PICK_SLOP) { return app.view ? app.view.pick(clientX, clientY, slop) : null; },
  project(point) { return app.view ? app.view.projectToClient(point) : null; },
  geomPoint(geom, local) { return app.view ? app.view.geomPoint(geom, local) : null; },
  // The X-ray overlay as drawn: the anchor (world, and on screen), the wrench
  // as drawn (F, T: world axes; null for frames without one, and for the
  // state a reset starts from), each arrow's drawn length (m, 0 when hidden),
  // tip and direction, their scales and colours, and the legend's numbers.
  xray() {
    const v = app.view;
    if (!v) return null;
    const info = v.xrayInfo();
    const arrow = (a) => (a ? { shown: a.shown, length: a.length, direction: a.direction, tip: a.tip, origin: a.origin } : null);
    return {
      on: app.xray,
      anchor: info.anchor,
      anchorClient: info.anchor ? v.projectToClient(info.anchor) : null,
      F: info.wrench ? info.wrench.slice(0, 3) : null,
      T: info.wrench ? info.wrench.slice(3, 6) : null,
      forceLength: info.force ? info.force.length : 0,
      momentLength: info.moment ? info.moment.length : 0,
      force: arrow(info.force),
      moment: arrow(info.moment),
      scale: { force: v.options.xray.force.scale, moment: v.options.xray.moment.scale },
      colors: { force: v.options.xray.force.color, moment: v.options.xray.moment.color }, // as the renderer draws them
      legend: el.xrayLegend ? { shown: isShown(el.xrayLegend), force: el.xrayForce.textContent, moment: el.xrayMoment.textContent } : null,
    };
  },
  // The sensor charts as drawn: on, the card (shown, mode, box, its rows and
  // the values shown at the time `at`, empty, noData, stats), the play time
  // of the pose on screen, the sensor frame as drawn, and the markers
  // (view.sensorMarkersInfo() plus `client`, their position on screen).
  sensors() {
    const v = app.view, card = app.card;
    const box = el.sensorCard ? el.sensorCard.getBoundingClientRect().toJSON() : null;
    return {
      on: app.sensors,
      card: {
        shown: isShown(el.sensorCard), mode: app.cardMode, box,
        ...(card ? card.info() : {}),
        stats: card ? card.stats() : null,
      },
      playTime: v ? v.playTime : null,
      drawn: v ? v.sensors : null,
      markers: v ? v.sensorMarkersInfo().map((m) => ({ ...m, client: v.projectToClient(m.world) })) : [],
    };
  },
  // Ask for a help mode (optional), then lay out the panels around the X-ray
  // legend and the sensor card as a resize does; returns the help mode shown.
  help(mode) {
    if (mode !== undefined) app.help = mode;
    layoutHud();
    return app.helpShown;
  },
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
  window.addEventListener('blur', guard(() => {
    clearKeys();
    endDrag();
  }));
  window.addEventListener('touchstart', guard(setTouch), { capture: true, passive: true, once: true });
  document.addEventListener('visibilitychange', guard(() => {
    if (document.hidden) {
      clearKeys();
      endDrag();
      canvasTouches.clear(); // their pointerup or pointercancel may never come
    }
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
        if (!visible) {
          clearKeys();
          endDrag();
        }
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
  if (el.xrayBtn) onButton(el.xrayBtn, toggleXray);
  if (el.sensorsBtn) onButton(el.sensorsBtn, toggleSensors);
  if (el.fast) onButton(el.fast, toggleFast);
  window.addEventListener('resize', guard(layoutHud));
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(guard(() => {
      layoutHud();
      if (app.card) app.card.invalidate(); // the charts' text in the web fonts
    }), () => { /* keep the fallback font's layout */ });
  }
  if (flag('xray')) setXray(true, true);
  if (flag('sensors')) setSensors(true, true);
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
  // The sensor card (its history fills from now on, shown or not).
  if (el.sensorCard) {
    app.card = new SensorCard(el.sensorCard, demo);
    layoutHud(); // its mode and visibility (?sensors=1)
  }
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
    const socket = demo.sim && demo.sim.socket;
    app.view = new DemoRenderer(el.viewport, demo.visual, visual, { onRelease: giveBack, socketParent: socket ? socket.parent : null });
  } catch (error) {
    fail('The 3D view could not start. This demo needs WebGL 2 with hardware acceleration.', error);
    return;
  }
  // The legend's swatches in the arrows' colours as the renderer draws them.
  const xray = app.view.options.xray;
  el.root.style.setProperty('--xray-force', xray.force.color);
  el.root.style.setProperty('--xray-moment', xray.moment.color);
  watchContext(app.view.renderer.domElement);
  watchPointer(app.view.renderer.domElement);
  el.camera.dataset.mode = app.view.cameraMode;
  app.view.setXray(app.xray); // ?xray=1: the standing model behind the loader is in X-ray too
  app.view.setSensorMarkers(app.sensors); // ?sensors=1: and the markers on it
  app.view.setFraming(app.framing); // and framed clear of the card
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
