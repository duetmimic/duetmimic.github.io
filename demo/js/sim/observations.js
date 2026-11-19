// The two controllers' inputs, ported from osl_deploy:
// - HumanObservation: observations.FullBodyObservation (osl_mimic_gait_v1, 2338);
// - OslObservation: sensor_observation.SensorObservation (osl_split_sensors_v1:
//   the sensor terms demo.json lists, 500 inputs with the previous-command
//   history for bundle B) at default settings: per-bus lag "validation",
//   history, saturation, tare; no randomization, no noise;
// - PreviousCommand: policy.PreviousCommandPolicy's last_action history.
// Each writes into a caller-owned Float32Array (the policy input) and
// allocates nothing per call. `v` is SimCore's cache of MjData array views.

import { rotvecFromMatrix } from './math.js';

const TWO_PI = 2 * Math.PI;

function span(groups, name) {
  const g = groups[name];
  if (!g) throw new Error(`demo.json sim.human.observation lacks group ${name}`);
  return g;
}

export class HumanObservation {
  constructor(sim, refCfg) {
    const obs = sim.human.observation;
    this.size = obs.size;
    this.q = sim.root_qpos;
    this.v = sim.root_qvel;
    this.qposIds = Int32Array.from(sim.qpos_ids);
    this.qvelIds = Int32Array.from(sim.qvel_ids);
    this.bodyIds = Int32Array.from(sim.body_ids);
    this.bodyRoots = Int32Array.from(sim.body_rootids);
    this.muscleIds = Int32Array.from(sim.muscle_ids);
    this.activationIds = Int32Array.from(sim.activation_ids);
    this.touchAdr = Int32Array.from(sim.touch_adr);
    this.sites = this.bodyIds.length - 1;
    this.lookaheadSteps = refCfg.lookahead_steps;
    const g = obs.groups;
    this.at = {
      root: span(g, 'root_height')[0],
      joints: span(g, 'joint_position')[0],
      rootVel: span(g, 'root_linear_velocity')[0],
      jointVel: span(g, 'joint_velocity')[0],
      muscles: [
        span(g, 'muscle_length')[0], span(g, 'muscle_velocity')[0],
        span(g, 'muscle_force')[0], span(g, 'muscle_control')[0],
        span(g, 'muscle_activation')[0],
      ],
      touch: span(g, 'touch')[0],
      body: span(g, 'body_state')[0],
      reference: span(g, 'reference')[0],
      twist: span(g, 'twist')[0],
      clock: span(g, 'clock')[0],
    };
    // The layout this file writes must be the exporter's, slot for slot.
    const n = this.qposIds.length, m = this.muscleIds.length, s = this.sites;
    const expect = {
      joints: 5, rootVel: 5 + n, jointVel: 11 + n, touch: 11 + 2 * n + 5 * m,
      body: 15 + 2 * n + 5 * m,
      reference: 15 + 2 * n + 5 * m + 12 * s,
      twist: 15 + 2 * n + 5 * m + 12 * s + 3 * s + (this.lookaheadSteps - 1) * (9 + 3 * s) + 1,
    };
    for (const [key, value] of Object.entries(expect)) {
      if (this.at[key] !== value) throw new Error(`Human observation layout: ${key} at ${this.at[key]}, expected ${value}`);
    }
    if (this.at.clock + 2 !== this.size) throw new Error('Human observation layout: size mismatch');
    // Scratch.
    this.velocity = new Float64Array(6 * (s + 1));
    this.relative = new Float64Array(9);
    this.rpos = new Float64Array(3 * s);
  }

  // mj_objectVelocity(mjOBJ_XBODY, flg_local=0) of every tracked body: the
  // body's cvel moved from its tree root's subtree COM to the body origin,
  // [angular, linear] in world axes.
  _velocities(v) {
    const { cvel, xpos, subtree_com: com } = v;
    const out = this.velocity;
    for (let i = 0; i < this.bodyIds.length; i++) {
      const b = this.bodyIds[i], r = this.bodyRoots[i];
      const w0 = cvel[6 * b], w1 = cvel[6 * b + 1], w2 = cvel[6 * b + 2];
      const d0 = xpos[3 * b] - com[3 * r];
      const d1 = xpos[3 * b + 1] - com[3 * r + 1];
      const d2 = xpos[3 * b + 2] - com[3 * r + 2];
      // linear - dif x angular (mju_transformSpatial, motion vector)
      out[6 * i] = w0; out[6 * i + 1] = w1; out[6 * i + 2] = w2;
      out[6 * i + 3] = cvel[6 * b + 3] - (d1 * w2 - d2 * w1);
      out[6 * i + 4] = cvel[6 * b + 4] - (d2 * w0 - d0 * w2);
      out[6 * i + 5] = cvel[6 * b + 5] - (d0 * w1 - d1 * w0);
    }
    return out;
  }

  // The full observation at the current state (after mj_forward) and
  // reference, into out (Float32Array of `size`). Returns whether it is finite.
  build(v, reference, out) {
    const { qpos, qvel } = v;
    const q = this.q, vq = this.v, at = this.at;
    for (let i = 0; i < 5; i++) out[i] = qpos[q + 2 + i];
    const qi = this.qposIds, vi = this.qvelIds;
    for (let i = 0; i < qi.length; i++) out[at.joints + i] = qpos[qi[i]];
    for (let i = 0; i < 6; i++) out[at.rootVel + i] = qvel[vq + i];
    for (let i = 0; i < vi.length; i++) out[at.jointVel + i] = qvel[vi[i]];
    const mi = this.muscleIds, ai = this.activationIds, mo = at.muscles;
    const length = v.actuator_length, velocity = v.actuator_velocity, force = v.actuator_force;
    const ctrl = v.ctrl, act = v.act;
    for (let i = 0; i < mi.length; i++) {
      const id = mi[i];
      out[mo[0] + i] = length[id];
      out[mo[1] + i] = velocity[id];
      out[mo[2] + i] = force[id];
      out[mo[3] + i] = ctrl[id];
      out[mo[4] + i] = act[ai[i]];
    }
    const sensordata = v.sensordata;
    for (let i = 0; i < this.touchAdr.length; i++) out[at.touch + i] = sensordata[this.touchAdr[i]];

    // Goal, measured part: math.relative_quantities over the tracked bodies,
    // the anchor (pelvis) first. rpos (world), rotation vectors of R0^T Ri,
    // then per site [Ri^T R0 wi - w0, R0 (v0 - vi)].
    const { xpos, xmat } = v;
    const vel = this._velocities(v);
    const ids = this.bodyIds, s = this.sites, R = this.relative;
    const b0 = ids[0], p = 3 * b0, m0 = 9 * b0;
    const pos0 = at.body, rot0 = pos0 + 3 * s, vel0 = rot0 + 3 * s;
    const a00 = xmat[m0], a01 = xmat[m0 + 1], a02 = xmat[m0 + 2];
    const a10 = xmat[m0 + 3], a11 = xmat[m0 + 4], a12 = xmat[m0 + 5];
    const a20 = xmat[m0 + 6], a21 = xmat[m0 + 7], a22 = xmat[m0 + 8];
    const w00 = vel[0], w01 = vel[1], w02 = vel[2];
    const l00 = vel[3], l01 = vel[4], l02 = vel[5];
    for (let i = 0; i < s; i++) {
      const b = ids[i + 1], mb = 9 * b;
      out[pos0 + 3 * i] = xpos[3 * b] - xpos[p];
      out[pos0 + 3 * i + 1] = xpos[3 * b + 1] - xpos[p + 1];
      out[pos0 + 3 * i + 2] = xpos[3 * b + 2] - xpos[p + 2];
      // R = R0^T Rb (row-major)
      for (let r = 0; r < 3; r++) {
        const c0 = r === 0 ? a00 : r === 1 ? a01 : a02;
        const c1 = r === 0 ? a10 : r === 1 ? a11 : a12;
        const c2 = r === 0 ? a20 : r === 1 ? a21 : a22;
        for (let c = 0; c < 3; c++) {
          R[3 * r + c] = c0 * xmat[mb + c] + c1 * xmat[mb + 3 + c] + c2 * xmat[mb + 6 + c];
        }
      }
      rotvecFromMatrix(out, R, 0, rot0 + 3 * i);
      const k = 6 * (i + 1);
      const wi0 = vel[k], wi1 = vel[k + 1], wi2 = vel[k + 2];
      const o = vel0 + 6 * i;
      // R^T wi - w0
      out[o] = R[0] * wi0 + R[3] * wi1 + R[6] * wi2 - w00;
      out[o + 1] = R[1] * wi0 + R[4] * wi1 + R[7] * wi2 - w01;
      out[o + 2] = R[2] * wi0 + R[5] * wi1 + R[8] * wi2 - w02;
      // R0 (v0 - vi)
      const e0 = l00 - vel[k + 3], e1 = l01 - vel[k + 4], e2 = l02 - vel[k + 5];
      out[o + 3] = a00 * e0 + a01 * e1 + a02 * e2;
      out[o + 4] = a10 * e0 + a11 * e1 + a12 * e2;
      out[o + 5] = a20 * e0 + a21 * e1 + a22 * e2;
    }

    // Goal, reference part: site targets now, lookahead, phase; then the
    // served twist and the stride clock (zero while idle).
    reference.referenceRpos(0, this.rpos);
    out.set(this.rpos, at.reference);
    let o = reference.lookahead(out, at.reference + 3 * s);
    const phase = reference.phase;
    out[o] = phase;
    out[at.twist] = reference.twist[0];
    out[at.twist + 1] = reference.twist[1];
    out[at.twist + 2] = reference.twist[2];
    if (reference.idle) {
      out[at.clock] = 0; out[at.clock + 1] = 0;
    } else {
      out[at.clock] = Math.sin(TWO_PI * phase);
      out[at.clock + 1] = Math.cos(TWO_PI * phase);
    }
    let sum = 0;
    for (let i = 0; i < out.length; i++) sum += out[i];
    return Number.isFinite(sum);
  }
}

// One OSL sensor term's reads (demo.json sim.osl.observation.terms[*].reads),
// compiled for readSensorFrame: osl_deploy's SensorRead (sensordata address,
// selected components, frame slot, saturation, tare).
export function compileSensorReads(reads) {
  return reads.map((r) => ({
    adr: r.adr,
    components: r.components ? Int32Array.from(r.components) : null,
    dim: r.components ? r.components.length : r.size,
    frame: r.frame,
    saturation: r.saturation,
    tare: r.tare ? Float64Array.from(r.tare) : null,
  }));
}

// A sensor term's processed frame from the current sensordata, as osl_deploy's
// SensorObservation.read at default settings: per read, its selected
// components, clamped to +-saturation where one is declared (the IMU), minus
// the tare where one is declared (the load cells); value c of a read goes to
// out[base + read.frame + c]. The one implementation of that processing:
// OslObservation builds the policy input with it and render/frame.js the
// frames' `sensors`, so the display shows exactly what the OSL reads.
// Allocation-free.
export function readSensorFrame(reads, sensordata, out, base = 0) {
  for (let i = 0; i < reads.length; i++) {
    const r = reads[i];
    for (let c = 0; c < r.dim; c++) {
      let x = sensordata[r.adr + (r.components ? r.components[c] : c)];
      if (r.saturation != null) x = Math.min(Math.max(x, -r.saturation), r.saturation);
      if (r.tare) x -= r.tare[c];
      out[base + r.frame + c] = x;
    }
  }
}

// The OSL's own sensors: each bus term stacks `history` frames, oldest first
// (obs[offset + t * frame_dim + c]); with lag L its newest slot holds the frame
// read L control steps ago. A frame is the term's MJCF sensors read at the
// boundary, clamped to +-saturation (IMU) and minus the tare (load cells):
// readSensorFrame.
export class OslObservation {
  constructor(obsCfg) {
    this.size = obsCfg.size;
    this.terms = obsCfg.terms.map((t) => {
      const window = t.history + t.lag; // rows, oldest first
      return {
        name: t.name,
        offset: t.offset,
        frameDim: t.frame_dim,
        history: t.history,
        lag: t.lag,
        reads: compileSensorReads(t.reads),
        frameValues: new Float64Array(t.frame_dim),
        window: new Float64Array(window * t.frame_dim),
        rows: window,
      };
    });
    for (const t of this.terms) {
      const dim = t.reads.reduce((sum, r) => sum + r.dim, 0);
      if (dim !== t.frameDim) throw new Error(`OSL term ${t.name}: sensors add ${dim}, frame_dim ${t.frameDim}`);
    }
    // The sensor terms and the previous-command history tile the input exactly.
    const spans = this.terms.map((t) => [t.offset, t.offset + t.history * t.frameDim, t.name]);
    const pc = obsCfg.previous_command;
    if (pc) spans.push([pc.offset, pc.offset + pc.history * pc.width, 'previous_command']);
    spans.sort((a, b) => a[0] - b[0]);
    let end = 0;
    for (const [start, stop, name] of spans) {
      if (start !== end) throw new Error(`OSL observation: ${name} starts at ${start}, not ${end}`);
      end = stop;
    }
    if (end !== this.size) throw new Error(`OSL observation: its terms cover ${end} inputs, not ${this.size}`);
  }

  _read(sensordata) {
    for (const t of this.terms) readSensorFrame(t.reads, sensordata, t.frameValues, 0);
  }

  // A new episode: every slot of every window holds the first frame.
  reset(sensordata) {
    this._read(sensordata);
    for (const t of this.terms) {
      for (let row = 0; row < t.rows; row++) t.window.set(t.frameValues, row * t.frameDim);
    }
  }

  // The next control step: read and push one frame per bus.
  observe(sensordata) {
    this._read(sensordata);
    for (const t of this.terms) {
      const d = t.frameDim;
      t.window.copyWithin(0, d);
      t.window.set(t.frameValues, (t.rows - 1) * d);
    }
  }

  // The sensor terms into out (the previous-command term is PreviousCommand's).
  assemble(out) {
    let sum = 0;
    for (const t of this.terms) {
      // rows [0, history): the newest `lag` frames are not visible yet.
      const n = t.history * t.frameDim;
      for (let i = 0; i < n; i++) {
        out[t.offset + i] = t.window[i];
        sum += t.window[i];
      }
    }
    return Number.isFinite(sum);
  }
}

// policy.PreviousCommandPolicy: the OSL actor observes its own last `history`
// commands (raw policy units clamped to [-1, 1], never delayed, 0 after a
// reset), shifted in by control index so a gap repeats the held command.
export class PreviousCommand {
  constructor(cfg) {
    this.offset = cfg.offset;
    this.history = cfg.history;
    this.width = cfg.width;
    this.low = cfg.low;
    this.high = cfg.high;
    this.values = new Float32Array(this.history * this.width);
    this.command = new Float32Array(this.width);
    this.last = 0;
  }

  reset() {
    this.values.fill(0);
    this.command.fill(0);
    this.last = 0;
  }

  // Shift to control index k and write the history into the policy input.
  write(k, input) {
    const w = this.width;
    const gap = Math.min(Math.max(k - this.last, 0), this.history);
    if (gap) {
      this.values.copyWithin(0, gap * w);
      for (let row = this.history - gap; row < this.history; row++) {
        for (let c = 0; c < w; c++) {
          this.values[row * w + c] = Math.min(Math.max(this.command[c], this.low), this.high);
        }
      }
    }
    this.last = Math.max(this.last, k);
    input.set(this.values, this.offset);
  }

  // The policy's newest output (float32), the command the next step shifts in.
  update(mean) {
    for (let c = 0; c < this.width; c++) this.command[c] = mean[c];
  }
}
