// SimCore: the browser port of osl_deploy's synchronous runner.Simulation for
// the stage-2 split bundle at its default settings (lockstep controllers, no
// stress; actuator delay "train"; OSL sensor lag "validation"; "mean"
// inference; rebase every 5 s; reset to the idle clip, frame 0). Telemetry,
// metrics, async mode, stress, recorders and joysticks are left out.
//
// One control step (step()), in osl_deploy's order:
//   1. both controllers sample the same boundary observation (built at the end
//      of the previous step, or by reset()); the OSL input gets its previous-
//      command history (PreviousCommandPolicy) for control index k;
//   2. both policies run; their means map to ctrl = clip(a * scale + offset,
//      low, high) onto the actuator command bus, before tick 0;
//   3. `decimation` ticks: the delay line moves the bus into data.ctrl, mj_step;
//   4. the training failure check: mean distance of the 16 pelvis-relative
//      tracked bodies (xpos as mj_step left them) from reference(1) > 0.5 m;
//   5. mj_forward, consume the latched twist request (switch / rebase), advance
//      the reference, build both observations for the next step.
// After a failure step() does nothing and returns the failed info: reset().
//
// No DOM: runs in a module Worker and in Node.

import { CommandLatch, CommandSupport } from './commands.js';
import { ActuatorDelay } from './delay.js';
import { HumanObservation, OslObservation, PreviousCommand } from './observations.js';
import { GaitReference } from './reference.js';

const now = typeof performance !== 'undefined' ? () => performance.now() : () => Date.now();

const VIEWS = [
  'qpos', 'qvel', 'act', 'ctrl', 'actuator_length', 'actuator_velocity', 'actuator_force',
  'sensordata', 'xpos', 'xmat', 'cvel', 'subtree_com',
];

function finite(a) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i];
  return Number.isFinite(sum);
}

export class SimCore {
  // mujoco: the wasm module; model/data: MjModel/MjData of physics.mjb (or the
  // bundle's model.mjb, same ids); cfg: demo.json; reference: the
  // reference.bin.gz container ({meta, arrays}) or its arrays; policies:
  // {human, osl}, each {run(Float32Array) -> Promise<Float32Array>}.
  constructor({ mujoco, model, data, cfg, reference, policies }) {
    if (!mujoco || !model || !data || !cfg || !reference) {
      throw new TypeError('SimCore needs {mujoco, model, data, cfg, reference, policies}');
    }
    const sim = cfg.sim;
    if (!sim || !cfg.reference) throw new Error('demo.json lacks the sim/reference sections');
    const dims = sim.model;
    for (const key of ['nq', 'nv', 'nu', 'na', 'nbody', 'nsensordata']) {
      if (model[key] !== dims[key]) {
        throw new Error(`Model ${key} is ${model[key]}, demo.json expects ${dims[key]}`);
      }
    }
    if (Math.abs(model.opt.timestep - dims.timestep) > 1e-12) {
      throw new Error(`Model timestep ${model.opt.timestep} differs from ${dims.timestep}`);
    }
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.cfg = cfg;
    this.sim = sim;
    // osl_deploy's Bundle: keep a bad state for the failure check, no auto-reset.
    model.opt.disableflags = model.opt.disableflags | dims.disable_autoreset;
    const arrays = reference.arrays || reference;
    this.reference = new GaitReference(cfg.reference, arrays);
    this.support = new CommandSupport(cfg.reference, this.reference);
    this.latch = new CommandLatch(sim.rebase_steps);
    this.human = new HumanObservation(sim, cfg.reference);
    this.osl = new OslObservation(sim.osl.observation);
    this.previousCommand = new PreviousCommand(sim.osl.observation.previous_command);
    this.blocks = {
      human: sim.human.action.blocks.map(compileBlock),
      osl: sim.osl.action.blocks.map(compileBlock),
    };
    this.delay = new ActuatorDelay(model.nu, [...sim.human.action.blocks, ...sim.osl.action.blocks]);
    const covered = new Uint8Array(model.nu);
    for (const block of [...this.blocks.human, ...this.blocks.osl]) {
      for (const id of block.ids) covered[id] += 1;
    }
    if (covered.some((c) => c !== 1)) throw new Error('The controllers must drive every actuator exactly once');
    this.outputSize = { human: sim.human.action.size, osl: sim.osl.action.size };
    this.humanInput = new Float32Array(sim.human.observation.size);
    this.oslInput = new Float32Array(sim.osl.observation.size);
    this.lastMeans = {
      human: new Float32Array(this.outputSize.human),
      osl: new Float32Array(this.outputSize.osl),
    };
    this.resetJointPos = arrays.reset_joint_pos;
    this.resetJointVel = arrays.reset_joint_vel;
    this.hold = new Float64Array(model.nu);
    this.siteTarget = new Float64Array(3 * this.human.sites);
    this._root = { pos: new Float64Array(3), quat: new Float64Array(4), qvel: new Float64Array(6) };
    this.policies = policies || null;
    this.epoch = 0;
    this.v = null;
    this._inflight = null;
    this.reset();
  }

  // Cached MjData views; refreshed when wasm memory growth detached them.
  views() {
    const v = this.v;
    if (v && v.qpos.length) return v;
    const fresh = {};
    for (const key of VIEWS) fresh[key] = this.data[key];
    this.v = fresh;
    return fresh;
  }

  get time() { return this.data.time; }

  // Start over: idle clip, frame 0, standing still at the reference pose.
  reset() {
    const { mujoco, model, data, sim } = this;
    this.epoch += 1;
    this._inflight = null;
    const ref = this.reference;
    const start = this.cfg.reference.reset;
    ref.reset(start.clip, start.frame);
    this.latch.reset(ref.twist);
    mujoco.mj_resetData(model, data);
    const v = this.views();
    const root = this._root;
    ref.rootState(root.pos, root.quat, root.qvel);
    const q = sim.root_qpos, vq = sim.root_qvel;
    v.qpos.set(root.pos, q);
    v.qpos.set(root.quat, q + 3);
    v.qvel.set(root.qvel, vq);
    const qi = sim.qpos_ids, vi = sim.qvel_ids;
    for (let i = 0; i < qi.length; i++) {
      v.qpos[qi[i]] = this.resetJointPos[i];
      v.qvel[vi[i]] = this.resetJointVel[i];
    }
    mujoco.mj_forward(model, data); // mj_resetData cleared act, ctrl and warm start.
    const w = this.views();
    // A reset holds position servos at the start pose, everything else at 0.
    this.hold.fill(0);
    for (const [actuator, adr] of sim.hold_servos) this.hold[actuator] = w.qpos[adr];
    this.delay.reset(this.hold, w.ctrl);
    this.osl.reset(w.sensordata);
    this.osl.assemble(this.oslInput);
    this.human.build(w, ref, this.humanInput);
    this.previousCommand.reset();
    this.lastMeans.human.fill(0);
    this.lastMeans.osl.fill(0);
    this.tick = 0;
    this.steps = 0;
    this.failed = false;
    this.failure = null;
    this.siteError = 0;
    this.switched = false;
    this.timing = { policy: 0, physics: 0, observe: 0 };
    return this.info();
  }

  // Latch a twist [vx, vy, wz]; the next control boundary consumes it (after a
  // running 0.4 s blend ends). The latest request wins.
  requestTwist(twist) {
    this.latch.request(twist);
  }

  // One 10 ms control step. Concurrent calls share the step in flight; a
  // reset() during the policies' await discards that step.
  step() {
    if (!this._inflight) {
      this._inflight = this._step().finally(() => { this._inflight = null; });
    }
    return this._inflight;
  }

  async _step() {
    if (this.failed) return this.info();
    if (!this.policies) throw new Error('SimCore has no policies');
    const epoch = this.epoch;
    const k = this.steps; // control index since the reset (tick / decimation)
    this.previousCommand.write(k, this.oslInput);
    const started = now();
    const [human, osl] = await Promise.all([
      this.policies.human.run(this.humanInput),
      this.policies.osl.run(this.oslInput),
    ]);
    if (epoch !== this.epoch) return this.info(); // reset() ran meanwhile
    const policyDone = now();
    let failure = null;
    if (!this._apply('human', human) || !this._apply('osl', osl)) {
      failure = 'policy returned invalid actions';
    } else {
      this.previousCommand.update(osl);
    }

    const { mujoco, model, data, sim } = this;
    let v = this.views();
    if (failure === null) {
      for (let substep = 0; substep < sim.decimation; substep++) {
        this.delay.write(v.ctrl);
        mujoco.mj_step(model, data);
        this.tick += 1;
        v = this.views();
        if (!finite(v.qpos) || !finite(v.qvel) || !finite(v.act)) {
          failure = `numerical: non-finite state at substep ${substep + 1}/${sim.decimation}`;
          break;
        }
      }
    }
    const physicsDone = now();
    // Training's termination check on the sites of the last pre-integration tick.
    this.siteError = this._siteError(v);
    this.switched = false;
    if (failure === null) {
      if (!(this.siteError <= sim.failure_site_threshold)) {
        failure = `tracking: relative site error ${this.siteError.toFixed(3)} m > ${sim.failure_site_threshold} m`;
      }
      mujoco.mj_forward(model, data);
      v = this.views();
      this.switched = this.latch.consume(this.reference, this.support, v.qpos, sim.root_qpos);
      this.reference.advance();
      this.osl.observe(v.sensordata);
      const oslOk = this.osl.assemble(this.oslInput);
      const humanOk = this.human.build(v, this.reference, this.humanInput);
      if (failure === null && !(oslOk && humanOk)) failure = 'numerical: non-finite observation';
    }
    this.steps += 1;
    this.failed = failure !== null;
    this.failure = failure;
    const done = now();
    this.timing = {
      policy: policyDone - started,
      physics: physicsDone - policyDone,
      observe: done - physicsDone,
    };
    return this.info();
  }

  // Map one controller's mean onto the command bus; false if invalid.
  _apply(name, mean) {
    if (!mean || mean.length !== this.outputSize[name]) return false;
    if (!finite(mean)) return false;
    this.lastMeans[name].set(mean);
    const bus = this.delay.ctrl;
    for (const block of this.blocks[name]) {
      const { ids, scale, offset, low, high, start } = block;
      for (let i = 0; i < ids.length; i++) {
        const a = mean[start + i] * scale[i] + offset[i];
        bus[ids[i]] = Math.min(Math.max(a, low[i]), high[i]);
      }
    }
    return true;
  }

  // Mean distance of the tracked bodies, relative to the pelvis, from the
  // reference one step ahead (runner.mean_site_error on reference(1)).
  _siteError(v) {
    const target = this.reference.referenceRpos(1, this.siteTarget);
    const ids = this.human.bodyIds, xpos = v.xpos;
    const p = 3 * ids[0];
    let total = 0;
    for (let i = 1; i < ids.length; i++) {
      const b = 3 * ids[i], t = 3 * (i - 1);
      const d0 = xpos[b] - xpos[p] - target[t];
      const d1 = xpos[b + 1] - xpos[p + 1] - target[t + 1];
      const d2 = xpos[b + 2] - xpos[p + 2] - target[t + 2];
      total += Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    }
    return total / (ids.length - 1);
  }

  info() {
    const ref = this.reference;
    const t = ref.twist, r = this.latch.requested;
    return {
      time: this.data.time,
      step: this.steps,
      failed: this.failed,
      failure: this.failure,
      siteError: this.siteError,
      clip: ref.clip,
      clipName: ref.clipName,
      twist: [t[0], t[1], t[2]],
      requested: [r[0], r[1], r[2]],
      phase: ref.phase,
      blending: ref.blending,
      switched: this.switched,
      timing: this.timing,
    };
  }
}

function compileBlock(block) {
  return {
    name: block.name,
    start: block.start,
    ids: Int32Array.from(block.actuator_ids),
    scale: Float64Array.from(block.scale),
    offset: Float64Array.from(block.offset),
    low: Float64Array.from(block.low),
    high: Float64Array.from(block.high),
  };
}
