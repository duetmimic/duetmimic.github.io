// DemoSession: the simulation side of the demo, with no DOM. It runs inside the
// module worker (js/worker.js) and in Node (tools/test/e2e_session.test.mjs).
//
//   const session = await DemoSession.create({ assetsBase, backend, onProgress });
//   session.setKeys(['KeyW', 'KeyA']);   // held KeyboardEvent.code values
//   session.handleKey('Digit4');         // speed level: Digit1-5, Q/E
//   const info = await session.step();   // one 10 ms control step (SimCore info)
//   const steps = await session.advance(); // real-time pacing: the steps owed now
//   session.on('fall', (event) => ...);  // events, see below
//   session.status();                    // HUD fields
//   session.extract();                   // {frame, transfer} for the renderer, or null
//
// Loading (create): demo.json first (always revalidated: {cache: 'no-cache'}),
// then concurrently the MuJoCo wasm, the physics model, the reference clips and
// both policies. Each asset is fetched as `${file}?v=${demo.versions[file]}`
// (the exporter's content hashes), so a browser never mixes a new demo.json
// with cached old files after a deploy. onProgress receives {loaded, total,
// stage, parts}, weighted by bytes (sizes from demo.json; the wasm size from
// the server once it answers). The first failure rejects create() at once: the
// other downloads are aborted, and whatever still finishes is disposed.
//
// Command shaping (CommandShaper): between the keys (twistForKeys) and
// SimCore.requestTwist, two rules hold a walk request back briefly. While one
// holds, status() shows what is really commanded ('Starting…', 'Stopping…'),
// not the keys.
// - Start gate. When a gait starts out of the idle clip, the idle clip's frame
//   sets the stride's entry phase (GaitReference.switch). From the reset
//   stance, entries at phase 0.2-0.5 start fastest and straightest; phase 0
//   (double support, where a reset leaves the idle clock) is slow and veers
//   (at 1.0 m/s: 1.6-1.8 m in the first 2.5 s against 1.2 m; at 1.2 m/s,
//   2.2-2.5 m in 3 s in 8 of 8 gated starts). So the first walk after a reset
//   waits for the idle clock to reach that window: at most 0.35 s, 0.12 s on
//   average. Later starts are not gated: after a stop the feet stand wherever
//   the last stride left them, and gating made no difference (1.97 against
//   2.02 m in 3 s over 8 restarts). Turning in place is never gated.
// - Forward after backward. Walking forward out of a backward walk is the
//   policy's weakest transition: the body keeps moving backward under a
//   forward reference, then falls. A forward request while the reference
//   walks backward, or before the body has come to rest after it, stands
//   first: idle until the pelvis is slower than SHAPING.settleSpeed, at most
//   SHAPING.maxHold after the key press. Measured with a seeded key-masher and
//   paired transition trials (WebGPU, 2026-11-16): backing up at -0.7 m/s,
//   then W at 1.2 m/s, directly or after a 0.3 s stop, fell 3 times in 20
//   trials without the rule and 0 times in 32 with it; in the key-masher, 2 of
//   the 4 falls without the rule started from a backward walk, none with it.
//   The forward walk then starts 0.8-1.0 s after the key instead of 0.0-0.4 s.
//   The same guard for forward -> backward did not help (the falls moved into
//   the stop), so backing up is never held.
//
// Falls: when SimCore reports a failure, the session relaxes every actuator
// (muscle excitation 0, OSL torque 0) and keeps stepping the physics for
// `fallSeconds` so the body visibly goes down. If the state is not finite it
// holds the last drawable pose instead. Then it resets. Events, in order:
// 'fall' {failure, siteError, physics}, then 'reset' {reason: 'fall'}.
//
// Dragging (setPerturb / clearPerturb, DragSpring): the visitor grabs a point
// of a rendered body part and pulls it toward a target point with a damped
// spring, applied at every physics tick, also while a fall plays out. A reset
// ends the drag with 'perturb-end' {reason: 'reset' | 'fall-reset'} before
// the 'reset' event; an invalid request ends it with {reason: 'error'}.
//
// Events (on(type, fn); '*' receives all): fall, reset {reason: 'fall'|'user'},
// level {level, name, label}, pause {paused}, backend {backend, fallback},
// perturb-end {reason, message, geom, body, id}.
// Every event also carries {type, time (sim seconds since the reset), steps}.

import { fetchBytes, loadContainer, loadJSON } from './lib/container.js';
import { loadMujoco, modelFromBytes } from './lib/mujoco.js';
import { loadPolicy } from './nn/index.js';
import { FrameExtractor } from './render/frame.js';
import { SimCore, levelForKey, twistForKeys } from './sim/index.js';

const defaultClock = () => (globalThis.performance ? performance.now() : Date.now());

// Size of vendor/mujoco/mujoco.wasm.gz, used to weight progress until the
// server reports the real length.
export const WASM_BYTES_ESTIMATE = 2475508;

// Entry-phase window for the first walk after a reset (fraction of the idle clip).
export const START_WINDOW = [0.2, 0.5];
const IDLE_TWIST = Object.freeze([0, 0, 0]);

// Command shaping parameters (see the header); DemoSession's `shaping` option
// overrides them, e.g. {startWait: 0, maxHold: 0} turns both rules off.
export const SHAPING = Object.freeze({
  startWindow: START_WINDOW,
  startWait: 0.4, // s: the longest the start gate holds a walk back
  settleSpeed: 0.35, // m/s: pelvis speed (EMA) under which the body has come to rest
  settleTau: 0.15, // s: time constant of that EMA
  maxHold: 1.0, // s: the longest forward-after-backward holds a forward request back
});
export const STARTING = 'Starting…';
export const STOPPING = 'Stopping…';

function finite(a) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i];
  return Number.isFinite(sum);
}

const sign = (x) => (x > 1e-9 ? 1 : x < -1e-9 ? -1 : 0);

// ---------------------------------------------------------------------------
// CommandShaper: what to request for the keys' command (twistForKeys), given
// the reference and the body. Per control step DemoSession calls observe(),
// track(command), decide(command) and commit(decision) before
// SimCore.requestTwist(decision.twist), and sense() after the physics.
// observe() is idempotent and decide() is pure, so status() can preview the
// decision for the keys held right now.
// ---------------------------------------------------------------------------
export class CommandShaper {
  constructor(core, dt, config = SHAPING) {
    this.core = core;
    this.ref = core.reference;
    this.dt = dt;
    const c = { ...SHAPING, ...(config || {}) };
    const steps = (s) => Math.max(0, Math.round(s / dt));
    this.config = c;
    this.window = c.startWindow;
    this.maxWait = steps(c.startWait);
    this.maxHold = steps(c.maxHold);
    this.settleSpeed = c.settleSpeed;
    this.ema = Math.min(1, dt / c.settleTau);
    this.reset();
  }

  reset() {
    this.fresh = true; // no gait since the reset: the body is in the reset stance
    this.wait = 0; // steps the start gate has held
    this.dir = 0; // walking direction the reference serves (0: idle, turning in place)
    this.backing = false; // walking backward, or not yet at rest after it
    this.speed = 0; // pelvis horizontal speed, EMA (m/s)
    this.userDir = 0; // walking direction of the keys when last tracked
    this.userSince = 0; // control step that direction began
  }

  // Track the reference (it switches clips at the end of a step).
  observe() {
    const ref = this.ref;
    if (!ref.idle) this.fresh = false;
    this.dir = ref.idle ? 0 : sign(ref.twist[0]);
    if (this.dir < 0) this.backing = true;
    else if (this.dir > 0) this.backing = false;
  }

  // Once per control step, before decide(): when the keys' direction began.
  track(command) {
    const dir = command.action === 'idle' ? 0 : sign(command.twist[0]);
    if (dir !== this.userDir) {
      this.userDir = dir;
      this.userSince = this.core.steps;
    }
  }

  // Once per control step, after the physics: the pelvis speed (free-joint
  // linear velocity, world frame) and whether the body came to rest.
  sense(qvel, q) {
    this.speed += this.ema * (Math.hypot(qvel[q], qvel[q + 1]) - this.speed);
    if (this.backing && this.dir === 0 && !this.ref.blending && this.speed < this.settleSpeed) {
      this.backing = false;
    }
  }

  // The request for the keys' command now: {twist, reason, label, moving}.
  // reason: null (the command as is), 'start' (start gate) or 'reverse'
  // (forward after backward); label: the HUD command text.
  decide(command) {
    const ref = this.ref;
    const dir = command.action === 'idle' ? 0 : sign(command.twist[0]);
    if (dir > 0 && this.backing) {
      const held = dir === this.userDir ? this.core.steps - this.userSince : 0;
      if (held < this.maxHold) {
        return { twist: IDLE_TWIST, reason: 'reverse', label: ref.idle ? STARTING : STOPPING, moving: true };
      }
    }
    if (dir !== 0 && this.fresh && ref.idle && this.wait < this.maxWait) {
      const phase = ref.phase;
      if (!(phase >= this.window[0] && phase <= this.window[1])) {
        return { twist: IDLE_TWIST, reason: 'start', label: STARTING, moving: true };
      }
    }
    return { twist: command.twist, reason: null, label: command.label, moving: command.action !== 'idle' };
  }

  // Book-keeping for the decision this step requests.
  commit(decision) {
    this.wait = decision.reason === 'start' ? this.wait + 1 : 0;
  }
}

// ---------------------------------------------------------------------------
// Real-time pacing. Wall times are in ms (performance.now()), sim steps are
// `dt` seconds. due(now) tells how many control steps are owed. More than
// `maxLag` seconds behind, the debt is dropped (the clock re-anchors) and one
// step is due: the simulation then runs slower than real time instead of
// spiralling into ever longer catch-up bursts. restart() re-anchors without
// debt (start, resume after a pause). RTF is sim time over wall time across
// the last `window` seconds, null until half a second has been measured.
// ---------------------------------------------------------------------------
export class Pacer {
  constructor({ dt = 0.01, maxLag = 0.25, window = 2, clock = defaultClock } = {}) {
    this.dt = dt;
    this.dtMs = dt * 1000;
    this.maxLagSteps = Math.max(1, Math.round(maxLag / dt));
    this.windowMs = window * 1000;
    this.clock = clock;
    this.dropped = 0; // steps of debt dropped so far
    this.rtf = null;
    this.restart();
  }

  restart(now = this.clock()) {
    this.base = now;
    this.done = 0;
    this.samples = [{ t: now, steps: 0 }];
  }

  due(now = this.clock()) {
    const owed = Math.floor((now - this.base) / this.dtMs) - this.done;
    if (owed > this.maxLagSteps) {
      this.dropped += owed - 1;
      this.base = now - (this.done + 1) * this.dtMs;
      return 1;
    }
    return owed > 0 ? owed : 0;
  }

  tick(steps = 1) { this.done += steps; }

  // ms until the next step is due (<= 0: already due).
  msUntilNext(now = this.clock()) { return this.base + (this.done + 1) * this.dtMs - now; }

  sample(now = this.clock()) {
    const s = this.samples;
    s.push({ t: now, steps: this.done });
    while (s.length > 2 && now - s[1].t >= this.windowMs) s.shift();
    const span = now - s[0].t;
    if (span >= 500) this.rtf = ((this.done - s[0].steps) * this.dtMs) / span;
    return this.rtf;
  }
}

// ---------------------------------------------------------------------------
// HUD text helpers.
// ---------------------------------------------------------------------------

// A short clip label for the HUD (the full name goes in a tooltip):
// "idle", "fwd 1.00 m/s", "fwd 1.00, +0.50 rad/s", "bck 0.60 m/s", "turn -1.00 rad/s".
export function clipLabel(clip) {
  if (!clip || clip.idle) return 'idle';
  const [vx, , wz] = clip.twist;
  const w = `${wz < 0 ? '-' : '+'}${Math.abs(wz).toFixed(2)}`;
  if (Math.abs(vx) < 1e-6) return `turn ${w} rad/s`;
  const v = `${vx > 0 ? 'fwd' : 'bck'} ${Math.abs(vx).toFixed(2)}`;
  return Math.abs(wz) < 1e-6 ? `${v} m/s` : `${v}, ${w} rad/s`;
}

// The Speed readout: the commanded twist while moving, else the selected level.
export function speedText(command, ui, level) {
  if (command.action !== 'idle') return command.speed;
  const lv = ui.levels[level];
  return `level ${lv.name} · ${lv.label}`;
}

export const backendLabel = (backend) => (backend === 'webgpu' ? 'WebGPU' : 'CPU');

// ---------------------------------------------------------------------------
// DragSpring: pull a body part with the mouse (DemoSession.setPerturb). Our
// own take on the MuJoCo viewer's perturbation (engine_vis_interact.c of
// MuJoCo 3.11: mjv_initPerturb, mjv_applyPerturbForce): at every physics tick
// a damped spring pulls the grabbed point p, fixed in a rendered geom, toward
// the target point the visitor drags it to (world, z up):
//   F = k m (target - p) - 2 zeta sqrt(k) m (v_p - v_root),  |F| capped,
// applied at p: the geom's body gets F in xfrc_applied, with the torque
// (p - xipos) x F about its center of mass. As in the viewer, k is the model's
// vis.map.stiffness (100 1/s^2) and m is mjv_initPerturb's localmass, the
// effective mass at p when the grab starts: 3 / trace(J M^-1 J^T), J the
// point's translational Jacobian (it matches the native mjv_initPerturb to 6
// digits). Here m is about 0.9-1.3 kg at a hand, 4 kg at a forearm or upper
// arm, 10 kg at the head, 25 kg at the pelvis and 33 kg at the chest.
//
// Tuned in closed loop for the demo (DemoSession on WebGPU, 2026-11-16: a
// sideways pull ramped in over 0.1 s, held 1 s, released; a fall is the
// session's failure check). The departures from the viewer, in DRAG:
// - maxMass. With the viewer's m the trunk spring is 2500-3300 N/m: 0.35 m on
//   the standing pelvis peaked at 660 N and it fell; capped at 400 N it
//   recovered standing, but both 0.35 m pulls while walking fell. m counts at
//   most 3 kg, so every part heavier than that (forearm, upper arm, head,
//   legs, trunk) is pulled with 300 N/m and the force grows over the whole
//   drag. Standing, 0.35 m on the pelvis, chest or head (90-105 N) stumbles
//   and recovers, 0.6 m on the pelvis recovered once in two, 0.8-1 m (225-300
//   N) knocks it over. Walking at 1 m/s, 0.35 m on the pelvis recovered in 7
//   of 8 trials (4 gait phases, both sides), 0.6 m fell. Lighter parts keep
//   the viewer's spring: 0.4 m on a hand moves it 0.2 m (22 N).
// - dampFrame 'root'. The viewer damps the point's absolute velocity, which
//   brakes a walking body: held 0.1 m off a walking pelvis (the target moving
//   along), the pull peaked at 85 N, against 38 N with the damper acting on
//   the point's velocity relative to the walker's root, the point the camera
//   follows. zeta = 1, critical (the viewer's sqrt(k) m is 0.5).
// - maxAccel. A wild drag on a hand (3 m) pulled with 240 N, flung it at
//   3.6 m/s and toppled the walker; |F| <= 50 m/s^2 times the effective mass
//   holds a hand at about 45 N (1.5 m/s, no fall).
// - maxForce. Whatever the part, |F| stays at most 400 N, about half the
//   walker's weight: a 5 m drag on the pelvis stays finite and simply falls.
// ---------------------------------------------------------------------------
export const DRAG = Object.freeze({
  stiffness: null, // k (1/s^2); null: the model's vis.map.stiffness (100, as the viewer)
  damping: 1, // zeta, the damping ratio of the point on its spring
  dampFrame: 'root', // 'root': damp v_p - v_root; 'world': damp v_p (the viewer)
  maxMass: 3, // kg: the most m counts for, so k m <= 300 N/m
  maxAccel: 50, // m/s^2: |F| <= maxAccel * the effective mass
  maxForce: 400, // N: |F| <= maxForce
});

// Margin (m) by which a grabbed point may lie outside its geom's bounding
// sphere (render meshes are decimated copies of the physics geoms).
const GRAB_MARGIN = 0.05;
// Targets farther than this from the origin on any axis (m) are rejected.
const TARGET_LIMIT = 1e4;

const isVec3 = (x) => x != null && typeof x === 'object' && x.length === 3
  && Number.isFinite(x[0]) && Number.isFinite(x[1]) && Number.isFinite(x[2]);

export class DragSpring {
  // geomIds: the pickable geoms (demo.visual.geom_ids, the rendered robot
  // geoms); geoms of static bodies are never pickable. config: overrides of
  // DRAG, read when a grab starts.
  constructor({ mujoco, model, data, geomIds, config = null }) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.config = { ...DRAG, ...(config || {}) };
    this.ngeom = model.ngeom;
    this.geomBody = Int32Array.from(model.geom_bodyid);
    this.geomRadius = Float64Array.from(model.geom_rbound);
    this.bodyRoot = Int32Array.from(model.body_rootid);
    const weld = model.body_weldid;
    this.pickable = new Uint8Array(model.ngeom);
    for (const g of geomIds || []) {
      if (Number.isInteger(g) && g >= 0 && g < model.ngeom && weld[this.geomBody[g]] !== 0) this.pickable[g] = 1;
    }
    this.modelStiffness = model.vis.map.stiffness;
    // Scratch for the effective mass, allocated once: output arguments must
    // live in the wasm heap (DoubleBuffer), a plain typed array is not one.
    this.jac = new mujoco.DoubleBuffer(3 * model.nv);
    this.jacM2 = new mujoco.DoubleBuffer(3 * model.nv);
    this.sqrtInvD = new Float64Array(model.nv);
    this.point = new Float64Array(3); // p, world
    this.local = new Float64Array(3); // p, geom frame
    this.target = new Float64Array(3);
    this.active = false;
    this.geom = -1;
    this.body = -1;
    this.root = 0; // the body's tree root (the walker's free-joint body)
    this.id = null;
    this.effective = 0; // effective mass at p (kg)
    this.km = 0; // k m (N/m)
    this.cm = 0; // 2 zeta sqrt(k) m (N s/m)
    this.limit = 0; // |F| cap of this grab (N)
    this.relative = true; // damp relative to the root
    this.force = 0; // |F| of the latest tick (N)
    this.stretch = 0; // |target - p| of the latest tick (m)
    this.peak = 0; // largest |F| since the grab started (N)
    this.ended = null; // {geom, local, id} of the grab the session ended itself
    this.hook = (model_, data_, tick, v) => this.apply(v); // SimCore's disturb
  }

  // Why spec is no valid drag request, or null.
  problem(spec) {
    if (!spec || typeof spec !== 'object') return 'perturb: expected {geom, local, target}';
    const { geom, local, target } = spec;
    if (!Number.isInteger(geom) || geom < 0 || geom >= this.ngeom || !this.pickable[geom]) {
      return `perturb: geom ${String(geom)} is not a rendered robot geom`;
    }
    if (!isVec3(local)) return 'perturb: local must be three finite numbers';
    if (Math.hypot(local[0], local[1], local[2]) > this.geomRadius[geom] + GRAB_MARGIN) {
      return `perturb: local point lies outside geom ${geom}`;
    }
    if (!isVec3(target) || Math.max(Math.abs(target[0]), Math.abs(target[1]), Math.abs(target[2])) > TARGET_LIMIT) {
      return 'perturb: target must be three finite numbers (world, m)';
    }
    return null;
  }

  // Start a drag, or move the target of the current one (same geom, local and
  // id). spec must pass problem(); v: SimCore's views. Returns 'start',
  // 'update', or 'stale': an update for the grab the session ended itself,
  // ignored until a release (end()) or another grab.
  set(spec, v) {
    const { geom, local, target } = spec;
    const id = typeof spec.id === 'number' || typeof spec.id === 'string' ? spec.id : null;
    const l = this.local;
    let result = 'update';
    if (!this.active || geom !== this.geom || id !== this.id
        || local[0] !== l[0] || local[1] !== l[1] || local[2] !== l[2]) {
      const e = this.ended;
      if (!this.active && e && geom === e.geom && id === e.id
          && local[0] === e.local[0] && local[1] === e.local[1] && local[2] === e.local[2]) return 'stale';
      this._grab(geom, local, id, v);
      result = 'start';
    }
    this.target[0] = target[0];
    this.target[1] = target[1];
    this.target[2] = target[2];
    return result;
  }

  _grab(geom, local, id, v) {
    const c = this.config;
    this.geom = geom;
    this.body = this.geomBody[geom];
    this.root = this.bodyRoot[this.body];
    this.id = id;
    this.local[0] = local[0];
    this.local[1] = local[1];
    this.local[2] = local[2];
    const k = c.stiffness ?? this.modelStiffness;
    const effective = this.effectiveMass(this.body, this._worldPoint(v, this.point));
    const m = Math.min(effective, c.maxMass);
    this.effective = effective;
    this.km = k * m;
    this.cm = 2 * c.damping * Math.sqrt(k) * m;
    this.limit = Math.min(c.maxForce, c.maxAccel * effective);
    this.relative = c.dampFrame !== 'world';
    this.active = true;
    this.ended = null;
    this.force = 0;
    this.stretch = 0;
    this.peak = 0;
  }

  // mjv_initPerturb's localmass at a world point of `body`, from the current
  // Jacobian and factorized mass matrix (valid after mj_forward / mj_step).
  effectiveMass(body, point) {
    const { mujoco, model, data } = this;
    mujoco.mj_jac(model, data, this.jac, null, point, body);
    const invD = data.qLDiagInv, s = this.sqrtInvD;
    for (let i = 0; i < s.length; i++) s[i] = Math.sqrt(invD[i]);
    mujoco.mj_solveM2(model, data, this.jacM2, this.jac.GetView(), s);
    const x = this.jacM2.GetView();
    let invmass = 0;
    for (let i = 0; i < x.length; i++) invmass += x[i] * x[i];
    return invmass === 0 ? 1 : 3 / Math.max(invmass, 1e-15); // mjMINVAL
  }

  // The grabbed point in the world: geom_xpos + geom_xmat (row-major) . local.
  _worldPoint(v, out) {
    const gp = v.geom_xpos, gm = v.geom_xmat, l = this.local;
    const o = 3 * this.geom, m = 9 * this.geom;
    out[0] = gp[o] + gm[m] * l[0] + gm[m + 1] * l[1] + gm[m + 2] * l[2];
    out[1] = gp[o + 1] + gm[m + 3] * l[0] + gm[m + 4] * l[1] + gm[m + 5] * l[2];
    out[2] = gp[o + 2] + gm[m + 6] * l[0] + gm[m + 7] * l[1] + gm[m + 8] * l[2];
    return out;
  }

  // End the drag. byItself: the session ended it (a reset, an error), so the
  // updates for it still on their way are ignored (set() -> 'stale').
  end(byItself = false) {
    this.ended = byItself && this.active ? { geom: this.geom, local: Array.from(this.local), id: this.id } : null;
    this.active = false;
    this.force = 0;
    this.stretch = 0;
  }

  // SimCore's disturbance hook, before every mj_step: add the spring's wrench
  // to v.xfrc_applied. Reads the kinematics mj_step left. Allocation-free.
  apply(v) {
    if (!this.active) return;
    const p = this._worldPoint(v, this.point), t = this.target;
    const px = p[0], py = p[1], pz = p[2];
    // The velocity of p: the body's cvel ([angular, linear] at its tree
    // root's subtree COM) moved to p, as mj_objectVelocity does.
    const cv = v.cvel, com = v.subtree_com, c = 6 * this.body, r = 3 * this.root;
    const wx = cv[c], wy = cv[c + 1], wz = cv[c + 2];
    const dx = px - com[r], dy = py - com[r + 1], dz = pz - com[r + 2];
    let vx = cv[c + 3] + wy * dz - wz * dy;
    let vy = cv[c + 4] + wz * dx - wx * dz;
    let vz = cv[c + 5] + wx * dy - wy * dx;
    if (this.relative) { // minus the velocity of the root body's origin
      const q = 6 * this.root, xp = v.xpos;
      const sx = xp[r] - com[r], sy = xp[r + 1] - com[r + 1], sz = xp[r + 2] - com[r + 2];
      const ux = cv[q], uy = cv[q + 1], uz = cv[q + 2];
      vx -= cv[q + 3] + uy * sz - uz * sy;
      vy -= cv[q + 4] + uz * sx - ux * sz;
      vz -= cv[q + 5] + ux * sy - uy * sx;
    }
    const ex = t[0] - px, ey = t[1] - py, ez = t[2] - pz;
    let fx = this.km * ex - this.cm * vx;
    let fy = this.km * ey - this.cm * vy;
    let fz = this.km * ez - this.cm * vz;
    let f = Math.sqrt(fx * fx + fy * fy + fz * fz);
    if (!Number.isFinite(f)) { // a non-finite state: the fall handling takes over
      this.force = 0;
      return;
    }
    if (f > this.limit) {
      const s = this.limit / f;
      fx *= s; fy *= s; fz *= s;
      f = this.limit;
    }
    // F at p: force, and torque about the body's center of mass.
    const xi = v.xipos, b = 3 * this.body;
    const ax = px - xi[b], ay = py - xi[b + 1], az = pz - xi[b + 2];
    const xf = v.xfrc_applied, w = 6 * this.body;
    xf[w] += fx;
    xf[w + 1] += fy;
    xf[w + 2] += fz;
    xf[w + 3] += ay * fz - az * fy;
    xf[w + 4] += az * fx - ax * fz;
    xf[w + 5] += ax * fy - ay * fx;
    this.force = f;
    if (f > this.peak) this.peak = f;
    this.stretch = Math.sqrt(ex * ex + ey * ey + ez * ez);
  }

  // For status(): {active, geom, body, id, force, maxForce, stretch}.
  status() {
    const on = this.active;
    return {
      active: on,
      geom: on ? this.geom : null,
      body: on ? this.body : null,
      id: on ? this.id : null,
      force: this.force,
      maxForce: on ? this.limit : this.config.maxForce,
      stretch: this.stretch,
    };
  }

  dispose() {
    this.active = false;
    for (const buffer of [this.jac, this.jacM2]) {
      try { if (buffer) buffer.delete(); } catch { /* already gone */ }
    }
    this.jac = this.jacM2 = null;
  }
}

// ---------------------------------------------------------------------------
// DemoSession
// ---------------------------------------------------------------------------
export class DemoSession {
  // Download and build everything. assetsBase: URL (or string) of the folder
  // holding demo.json (an absolute URL; a file: URL under Node).
  // backend: 'auto' | 'webgpu' | 'cpu' for the human network (nn/loadPolicy);
  // the OSL network uses 'auto' unless 'cpu' is asked for. gpu: a GPU object
  // for WebGPU outside a browser (Node: the 'webgpu' package).
  static async create({ assetsBase, backend = 'auto', gpu, onProgress, clock = defaultClock, ...options } = {}) {
    if (!assetsBase) throw new TypeError('DemoSession.create needs assetsBase');
    const base = new URL(String(assetsBase).replace(/\/?$/, '/'));
    // One signal for every download, so the first failure can stop the rest.
    const controller = new AbortController();
    const init = { signal: controller.signal };
    const t0 = clock();
    const times = {};
    const demo = await loadJSON(new URL('demo.json', base), { ...init, cache: 'no-cache' });
    times.config = clock() - t0;
    // An asset's URL with its content hash (demo.json "versions", written by the
    // exporter): a new export changes the URL, so no cached copy can be stale.
    const versions = demo.versions || {};
    const asset = (file) => {
      const url = new URL(file, base);
      if (versions[file]) url.searchParams.set('v', versions[file]);
      return url;
    };

    const parts = {
      wasm: { loaded: 0, total: WASM_BYTES_ESTIMATE },
      physics: { loaded: 0, total: demo.physics.bytes },
      reference: { loaded: 0, total: demo.reference.bytes },
      human: { loaded: 0, total: demo.policies.human.bytes },
      osl: { loaded: 0, total: demo.policies.osl.bytes },
    };
    let failed = false; // no progress reports after the failure
    const report = (stage) => {
      if (!onProgress || failed) return;
      let loaded = 0, total = 0;
      for (const p of Object.values(parts)) { loaded += p.loaded; total += p.total; }
      onProgress({ loaded, total, stage: stage || (loaded >= total ? 'setup' : 'download'), parts });
    };
    const track = (name) => (received, total) => {
      const p = parts[name];
      if (total > 0) p.total = total;
      p.loaded = Math.min(received, p.total);
      report();
    };
    const finished = (name) => (value) => {
      const p = parts[name];
      p.loaded = p.total;
      times[name] = clock() - t0;
      report();
      return value;
    };
    report('download');

    const jobs = [
      loadMujoco(track('wasm'), init).then(finished('wasm')),
      fetchBytes(asset(demo.physics.file), track('physics'), init).then(finished('physics')),
      loadContainer(asset(demo.reference.file), track('reference'), init).then(finished('reference')),
      loadPolicy(asset(demo.policies.human.file), { backend, gpu, onProgress: track('human'), init }).then(finished('human')),
      // The OSL net is small (and fp32, which the WebGPU kernels do not take):
      // 'auto' keeps it on the CPU, where a GPU round trip would cost more.
      loadPolicy(asset(demo.policies.osl.file), {
        backend: backend === 'cpu' ? 'cpu' : 'auto', gpu, onProgress: track('osl'), init,
      }).then(finished('osl')),
    ];
    let loaded;
    try {
      loaded = await Promise.all(jobs);
    } catch (error) {
      // Fail fast: report the first failure now rather than after the 25 MB of
      // weights. Abort the other downloads; a policy that still finishes
      // loading (e.g. one past its download) is disposed (its GPU device).
      failed = true;
      controller.abort();
      for (const job of jobs.slice(3)) {
        job.then((policy) => { try { policy.dispose(); } catch { /* ignore */ } }, () => {});
      }
      throw error;
    }
    const [mujoco, physicsBytes, reference, human, osl] = loaded;
    report('setup');
    let model = null, data = null, session;
    try {
      const tModel = clock();
      ({ model, data } = modelFromBytes(mujoco, physicsBytes, demo.physics.file.replace(/\.gz$/, '')));
      times.model = clock() - tModel;
      session = new DemoSession({ demo, mujoco, model, data, reference, policies: { human, osl }, clock, gpu, ...options });
    } catch (error) {
      human.dispose();
      osl.dispose();
      if (data) data.delete();
      if (model) model.delete();
      throw error;
    }
    times.total = clock() - t0;
    session.loadTimes = times;
    return session;
  }

  // Build from loaded parts (create() does this). options: paused (false),
  // fallSeconds (1.0), clock, maxLag (0.25 s), rtfWindow (2 s), shaping
  // (overrides of SHAPING), drag (overrides of DRAG), gpu (kept referenced:
  // Dawn for Node hangs or aborts once its GPU object is collected).
  constructor({ demo, mujoco, model, data, reference, policies, clock = defaultClock, gpu = null,
    paused = false, fallSeconds = 1.0, maxLag = 0.25, rtfWindow = 2, shaping = null, drag = null }) {
    this.demo = demo;
    this.ui = demo.ui;
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.policies = policies;
    this.gpu = gpu;
    this.clock = clock;
    this.core = new SimCore({ mujoco, model, data, cfg: demo, reference, policies });
    this.drag = new DragSpring({ mujoco, model, data, geomIds: demo.visual.geom_ids, config: drag });
    this.core.disturb = this.drag.hook;
    this.shaper = new CommandShaper(this.core, demo.sim.policy_dt, shaping);
    this.shaped = null; // the last step's shaping decision
    this.extractor = new FrameExtractor(mujoco, model, demo.visual);
    this.dt = demo.sim.policy_dt;
    this.decimation = demo.sim.decimation;
    this.fallSteps = Math.max(1, Math.round(fallSeconds / this.dt));
    this.pacer = new Pacer({ dt: this.dt, maxLag, window: rtfWindow, clock });
    const root = (demo.visual && demo.visual.root) || {};
    this.rootBody = root.body ?? 1;
    this.forward = Float64Array.from(root.forward ?? [1, 0, 0]);
    this.clips = demo.reference.clips;

    this.keys = new Set();
    this.level = this.ui.default_level;
    this.paused = !!paused;
    this.steps = 0; // control steps since creation (falls included)
    this.falls = 0;
    this.resets = 0;
    this.fall = null; // {failure, siteError, physics, remaining} while going down
    this.drawable = true; // the MjData state is finite and can be drawn
    this.lastInfo = this.core.info();
    this.timing = { policy: 0, physics: 0, observe: 0, step: 0 }; // smoothed ms
    this._handlers = new Map();
    this._inflight = null;
    this._backend = policies.human.backend;
    this.loadTimes = null;
  }

  // ------------------------------------------------------------- events
  on(type, fn) {
    if (!this._handlers.has(type)) this._handlers.set(type, new Set());
    this._handlers.get(type).add(fn);
    return () => this._handlers.get(type).delete(fn);
  }

  _emit(type, detail = {}) {
    const event = { type, time: this.data.time, steps: this.steps, ...detail };
    for (const key of [type, '*']) {
      const set = this._handlers.get(key);
      if (set) for (const fn of [...set]) fn(event);
    }
    return event;
  }

  // ------------------------------------------------------------- input
  // The held keys (KeyboardEvent.code values); replaces the previous set.
  setKeys(codes) {
    this.keys = new Set(Array.from(codes || [], String));
  }

  // A key press that may change the speed level (Digit1-5, Q/E). Returns
  // whether the level changed.
  handleKey(code) {
    const level = levelForKey(String(code), this.ui, this.level);
    if (level === this.level) return false;
    this.level = level;
    const lv = this.ui.levels[level];
    this._emit('level', { level, name: lv.name, label: lv.label });
    return true;
  }

  // twistForKeys for the held keys and the selected level.
  command() { return twistForKeys(this.keys, this.ui, this.level); }

  pause(paused) {
    paused = !!paused;
    if (paused === this.paused) return;
    this.paused = paused;
    if (!paused) this.pacer.restart(); // the clock stopped while paused: no catch-up burst
    this._emit('pause', { paused });
  }

  // Drag a body part: start a drag, or move the target of the current one.
  // spec {geom, local, target, id?}: geom a rendered robot geom
  // (demo.visual.geom_ids), local the grabbed point in that geom's frame
  // (world = geom_xpos + geom_xmat . local), target the world point it is
  // pulled toward, id optional (echoed in status and perturb-end). Returns
  // 'start', 'update', 'stale' (an update for a drag the session ended itself,
  // e.g. at a reset: ignored) or 'invalid' (the drag, if any, ends with
  // 'perturb-end' {reason: 'error', message}).
  setPerturb(spec) {
    const problem = this.drag.problem(spec);
    if (problem) {
      this.clearPerturb('error', problem);
      return 'invalid';
    }
    return this.drag.set(spec, this.core.views());
  }

  // End the drag. reason 'release' (or none): the visitor let go, no event.
  // Any other reason ('reset', 'fall-reset', 'error') means the session ended
  // it on its own and emits 'perturb-end' {reason, message, geom, body, id};
  // 'error' does even without a drag, so the page drops its drag either way.
  // Returns whether a drag was active.
  clearPerturb(reason = 'release', message = null) {
    const drag = this.drag;
    const was = drag.active;
    const { geom, body, id } = drag.status();
    const own = reason !== 'release' && reason !== null;
    drag.end(own);
    if (own && (was || reason === 'error')) this._emit('perturb-end', { reason, message, geom, body, id });
    return was;
  }

  // Back to the idle start pose ('user': R / the Reset button; 'fall': after a fall).
  // A drag ends first ('perturb-end' {reason: 'reset' | 'fall-reset'}).
  reset(reason = 'user') {
    this.clearPerturb(reason === 'fall' ? 'fall-reset' : 'reset');
    this.core.reset();
    this.fall = null;
    this.drawable = true;
    this.shaper.reset();
    this.shaped = null;
    this.lastInfo = this.core.info();
    this.resets += 1;
    this._emit('reset', { reason });
  }

  // ------------------------------------------------------------- stepping
  // One control step (10 ms of simulated time): the policies and physics, or
  // while a fall plays out, physics with relaxed actuators. Calls made while
  // a step is in flight share it.
  step() {
    if (!this._inflight) {
      this._inflight = this._step().finally(() => { this._inflight = null; });
    }
    return this._inflight;
  }

  async _step() {
    if (this.fall) return this._fallStep();
    const shaper = this.shaper;
    const command = this.command();
    shaper.observe();
    shaper.track(command);
    const decision = shaper.decide(command);
    shaper.commit(decision);
    this.shaped = decision;
    this.core.requestTwist(decision.twist);
    const started = this.clock();
    const info = await this.core.step();
    if (!info.failed) shaper.sense(this.core.views().qvel, this.demo.sim.root_qvel);
    const elapsed = this.clock() - started;
    this.steps += 1;
    this.lastInfo = info;
    const t = this.timing, k = this.steps < 20 ? 1 / this.steps : 0.05;
    t.policy += k * (info.timing.policy - t.policy);
    t.physics += k * (info.timing.physics - t.physics);
    t.observe += k * (info.timing.observe - t.observe);
    t.step += k * (elapsed - t.step);
    const backend = this.policies.human.backend;
    if (backend !== this._backend) {
      this._backend = backend;
      const fallback = this.policies.human.info && this.policies.human.info.fallback;
      this._emit('backend', { backend: backendLabel(backend), fallback: fallback || null });
    }
    if (info.failed && !this.fall) this._beginFall(info);
    return info;
  }

  _stateFinite() {
    const v = this.core.views();
    return finite(v.qpos) && finite(v.qvel) && finite(v.act);
  }

  _beginFall(info) {
    const physics = this._stateFinite();
    this.drawable = physics;
    this.fall = { failure: info.failure, siteError: info.siteError, physics, remaining: this.fallSteps };
    this.falls += 1;
    this._emit('fall', { failure: info.failure, siteError: info.siteError, physics });
  }

  _fallStep() {
    const fall = this.fall;
    if (fall.physics) {
      const { mujoco, model, data, core } = this;
      for (let i = 0; i < this.decimation; i++) {
        const v = core.views();
        v.ctrl.fill(0); // muscles relax, the OSL motors go limp
        core.applyDisturbances(v); // a drag keeps pulling the falling body
        mujoco.mj_step(model, data);
        core.tick += 1;
      }
      if (!this._stateFinite()) {
        fall.physics = false;
        this.drawable = false;
      }
    }
    this.steps += 1;
    fall.remaining -= 1;
    this.lastInfo = this.core.info();
    const info = this.lastInfo;
    if (fall.remaining <= 0) this.reset('fall');
    return info;
  }

  // Real-time pacing: run the control steps owed now, until caught up, paused,
  // or `budgetMs` of wall time has gone into this call. Returns the number of
  // steps run. Call it again after msUntilNextStep() (0 when behind).
  async advance({ budgetMs = 30, maxSteps = 100 } = {}) {
    if (this.paused) return 0;
    const pacer = this.pacer;
    const started = this.clock();
    let steps = 0;
    while (!this.paused && steps < maxSteps && pacer.due(this.clock()) > 0) {
      await this.step();
      pacer.tick();
      steps += 1;
      if (this.clock() - started >= budgetMs) break;
    }
    if (!this.paused) pacer.sample(this.clock());
    return steps;
  }

  msUntilNextStep() { return this.paused ? Infinity : this.pacer.msUntilNext(this.clock()); }

  get rtf() { return this.pacer.rtf; }

  // ------------------------------------------------------------- output
  // The root body's position and heading (rad, CCW from +x, of demo.visual's
  // root.forward axis), as the renderer sees it.
  pose() {
    const v = this.core.views();
    const b = this.rootBody, f = this.forward, xp = v.xpos, xm = v.xmat, m = 9 * b;
    const fx = xm[m] * f[0] + xm[m + 1] * f[1] + xm[m + 2] * f[2];
    const fy = xm[m + 3] * f[0] + xm[m + 4] * f[1] + xm[m + 5] * f[2];
    return { x: xp[3 * b], y: xp[3 * b + 1], z: xp[3 * b + 2], heading: Math.atan2(fy, fx) };
  }

  // A render frame of the current state ({frame, transfer}), or null while the
  // state cannot be drawn (non-finite after a numerical failure).
  extract() {
    if (!this.drawable) return null;
    return this.extractor.extract(this.data);
  }

  recycle(buffer) { this.extractor.recycle(buffer); }

  get backend() { return backendLabel(this.policies.human.backend); }

  status() {
    const command = this.command();
    // What the next step requests for these keys (a shaping rule may hold
    // them back): the HUD shows that, e.g. 'Starting…' instead of 'Forward'.
    this.shaper.observe();
    const shaped = this.shaper.decide(command);
    const info = this.lastInfo;
    const lv = this.ui.levels[this.level];
    const human = this.policies.human;
    const clip = this.clips[info.clip];
    return {
      command: shaped.label,
      action: command.action,
      keys: command.keys,
      moving: shaped.moving,
      shaping: shaped.reason,
      fast: command.fast,
      level: this.level,
      levelName: lv.name,
      levelLabel: lv.label,
      speed: speedText(command, this.ui, this.level),
      clip: clipLabel(clip),
      clipName: info.clipName,
      blending: info.blending,
      time: info.time,
      steps: this.steps,
      rtf: this.pacer.rtf,
      dropped: this.pacer.dropped,
      backend: backendLabel(human.backend),
      oslBackend: backendLabel(this.policies.osl.backend),
      adapter: (human.info && human.info.adapter) || null,
      // Why the human network runs where it does (nn's info.reason: 'self-test
      // passed', 'CPU faster (...)', 'WebGPU unavailable: ...', and after a GPU
      // failure at run time 'WebGPU failed, now on the CPU: ...').
      backendReason: (human.info && human.info.reason) || '',
      cpuKernel: (human.info && human.info.cpuKernel) || null,
      falling: !!this.fall,
      failure: this.fall ? this.fall.failure : null,
      falls: this.falls,
      resets: this.resets,
      paused: this.paused,
      siteError: info.siteError,
      pose: this.pose(),
      timing: { ...this.timing },
      // The drag: {active, geom, body, id, force (N, the latest tick's |F|),
      // maxForce (N), stretch (m, |target - grabbed point|)}.
      perturb: this.drag.status(),
    };
  }

  // Static facts for the 'ready' message.
  describe() {
    const { human, osl } = this.policies;
    return {
      backend: backendLabel(human.backend),
      adapter: (human.info && human.info.adapter) || null,
      reason: (human.info && human.info.reason) || '',
      cpuKernel: (human.info && human.info.cpuKernel) || null,
      oslBackend: backendLabel(osl.backend),
      level: this.level,
      dt: this.dt,
      loadTimes: this.loadTimes,
      bytes: {
        physics: this.demo.physics.bytes,
        reference: this.demo.reference.bytes,
        human: this.demo.policies.human.bytes,
        osl: this.demo.policies.osl.bytes,
        visual: this.demo.visual.bytes,
      },
    };
  }

  dispose() {
    this.core.disturb = null;
    this.drag.dispose();
    for (const policy of Object.values(this.policies)) {
      try { policy.dispose(); } catch { /* ignore */ }
    }
    try { this.data.delete(); } catch { /* ignore */ }
    try { this.model.delete(); } catch { /* ignore */ }
    this.gpu = null;
  }
}
