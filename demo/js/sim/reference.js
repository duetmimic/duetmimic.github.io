// GaitReference: port of osl_deploy reference.GaitReference (mjlab's
// GaitLibraryCommand for one environment), restricted to what SimCore reads:
// the tracked-site positions relative to the pelvis (rpos) with the 0.4 s
// blend, the lookahead, the phase, clip switching with re-anchoring, and the
// per-step advance. Root position/velocity blending (only the reset frame and
// a reference ghost would read it) is left out.
//
// Clip indices are indices into cfg.clips (the clips the UI can reach), not
// the library's; cfg.clips[i].index is the library index.
//
// Numerics follow the Python dtypes, so the trace test can demand exactness:
// gathered frames are float32 values, as numpy keeps them (rotate_xy and the
// anchor shift round to float32), and the outgoing clip's interpolation and
// the rpos blend are float32 arithmetic, with the scalar first rounded to
// float32, exactly when Python's fraction is a Python float (NEP 50 weak
// scalar: after a switch into idle until the outgoing clip wraps), else float64.

import {
  compose, inverse, planar, pyMod, quatMul, rint, rotateInverse, yawQuat,
} from './math.js';

const f32 = Math.fround;
const ZERO3 = new Float64Array(3);
const BLENDED = ['rootPos', 'rpos']; // the outgoing fields SimCore reads

export class ReferenceFrame {
  constructor(sites) {
    this.rootPos = new Float64Array(3);
    this.rootQuat = new Float64Array(4);
    this.rootLinVel = new Float64Array(3);
    this.rootAngVelW = new Float64Array(3);
    this.rpos = new Float64Array(3 * sites);
  }
}

// Rotate the xy part of src[so..so+3] by (c, s) into out[oo..], rounding to
// float32 (numpy assigns float64 results into the float32 frame); z is copied.
function rotateXY(out, oo, src, so, c, s) {
  const x = src[so], y = src[so + 1];
  out[oo] = f32(c * x - s * y);
  out[oo + 1] = f32(s * x + c * y);
  out[oo + 2] = src[so + 2];
}

// Free-joint qvel of a frame: world linear, body angular (reference.root_qvel).
export function rootQvel(out, frame) {
  out[0] = frame.rootLinVel[0];
  out[1] = frame.rootLinVel[1];
  out[2] = frame.rootLinVel[2];
  rotateInverse(out, frame.rootQuat, frame.rootAngVelW, 0, 0, 3);
  return out;
}

export class GaitReference {
  // cfg: demo.json "reference"; arrays: reference.bin.gz container arrays.
  constructor(cfg, arrays) {
    const clips = cfg.clips;
    const n = clips.length;
    this.clips = clips;
    this.names = clips.map((c) => c.name);
    this.libraryIndex = Int32Array.from(clips, (c) => c.index);
    this.lengths = Int32Array.from(clips, (c) => c.num_frames);
    this.starts = Int32Array.from(clips, (c) => c.start);
    this.twists = new Float64Array(3 * n);
    this.strides = new Float64Array(3 * n);
    this.idleClips = new Uint8Array(n);
    clips.forEach((c, i) => {
      this.twists.set(c.twist, 3 * i);
      this.strides.set(c.stride, 3 * i);
      this.idleClips[i] = Math.hypot(c.twist[0], c.twist[1], c.twist[2]) < 1e-6 ? 1 : 0;
    });
    this.dt = cfg.policy_dt;
    this.blendSteps = cfg.blend_steps;
    this.lookaheadSteps = cfg.lookahead_steps;
    this.lookaheadStride = cfg.lookahead_stride;
    this.metric = Float64Array.from(cfg.metric);
    this.sites = cfg.sites;
    this.data = {
      rootPos: arrays.root_pos,
      rootQuat: arrays.root_quat,
      rootLinVel: arrays.root_lin_vel,
      rootAngVelW: arrays.root_ang_vel_w,
      rpos: arrays.rpos,
    };
    const frames = this.starts[n - 1] + this.lengths[n - 1];
    const widths = { rootPos: 3, rootQuat: 4, rootLinVel: 3, rootAngVelW: 3, rpos: 3 * this.sites };
    for (const [key, width] of Object.entries(widths)) {
      const a = this.data[key];
      if (!(a instanceof Float32Array) || a.length !== frames * width) {
        throw new Error(`reference.bin.gz: ${key} must be float32 [${frames}, ${width}]`);
      }
    }
    // State (GaitReference.state_keys).
    this.anchor = new Float64Array(3);
    this.previousAnchor = new Float64Array(3);
    this.catchup = new Float64Array(2);
    this.twist = new Float64Array(3);
    // Python-float bookkeeping of previous_frame / previous_rate (see header).
    this._pyFrame = true;
    this._pyRate = true;
    // Scratch.
    this._ga = new Float64Array(3);
    this._yq = new Float64Array(4);
    this._fa = new ReferenceFrame(this.sites);
    this._fb = new ReferenceFrame(this.sites);
    this._cur = new ReferenceFrame(this.sites);
    this._old = new ReferenceFrame(this.sites);
    this._now = new ReferenceFrame(this.sites);
    this._fut = new ReferenceFrame(this.sites);
    this._qa = new Float64Array(6);
    this._qb = new Float64Array(6);
    this._target = new Float64Array(3);
    this._p = new Float64Array(3);
    this._pa = new Float64Array(3);
    this._pb = new Float64Array(3);
    this._prev = new Float64Array(3);
    this.reset(cfg.reset ? cfg.reset.clip : 0, cfg.reset ? cfg.reset.frame : 0);
  }

  reset(clip, frame = 0) {
    if (!(clip >= 0 && clip < this.lengths.length)) throw new RangeError('Clip index out of range');
    if (!(frame >= 0 && frame < this.lengths[clip])) throw new RangeError('Frame index out of range');
    this.clip = this.previous = clip;
    this.loops = 0;
    this.frame = frame;
    this.previousFrame = frame;
    this.previousRate = 1.0;
    this._pyFrame = this._pyRate = true;
    const row = this.starts[clip] + frame;
    this.anchor[0] = -this.data.rootPos[3 * row];
    this.anchor[1] = -this.data.rootPos[3 * row + 1];
    this.anchor[2] = 0;
    this.previousAnchor.set(this.anchor);
    this.blendCount = this.blendSteps;
    this.catchup.fill(0);
    this.twist.set(this.twists.subarray(3 * clip, 3 * clip + 3));
  }

  get phase() { return this.frame / this.lengths[this.clip]; }
  get idle() { return this.idleClips[this.clip] === 1; }
  get blending() { return this.blendCount < this.blendSteps; }
  get clipName() { return this.names[this.clip]; }

  // Frame `frame` of `clip` (wrapping, each wrap composing the clip's stride
  // onto the anchor), moved by the SE(2) `anchor`, into `out`.
  gather(clip, frame, anchor, out) {
    const n = this.lengths[clip];
    let wraps = Math.floor(frame / n);
    const local = frame - wraps * n;
    const a = this._ga;
    a[0] = anchor[0]; a[1] = anchor[1]; a[2] = anchor[2];
    for (; wraps > 0; wraps--) compose(a, a, this.strides, 3 * clip);
    const row = this.starts[clip] + local;
    const c = Math.cos(a[2]), s = Math.sin(a[2]);
    const d = this.data;
    rotateXY(out.rootPos, 0, d.rootPos, 3 * row, c, s);
    out.rootPos[0] = f32(out.rootPos[0] + a[0]);
    out.rootPos[1] = f32(out.rootPos[1] + a[1]);
    rotateXY(out.rootLinVel, 0, d.rootLinVel, 3 * row, c, s);
    rotateXY(out.rootAngVelW, 0, d.rootAngVelW, 3 * row, c, s);
    const sites = this.sites, base = 3 * sites * row;
    for (let i = 0; i < sites; i++) rotateXY(out.rpos, 3 * i, d.rpos, base + 3 * i, c, s);
    quatMul(out.rootQuat, yawQuat(this._yq, a[2]), d.rootQuat, 0, 4 * row);
    return out;
  }

  // The outgoing clip during a blend (root_pos and rpos only).
  _outgoing(offset, out) {
    const x = this.previousFrame + offset * this.previousRate;
    const lo = Math.floor(x);
    const a = this.gather(this.previous, lo, this.previousAnchor, this._fa);
    const b = this.gather(this.previous, lo + 1, this.previousAnchor, this._fb);
    const fr = x - lo;
    // a + fraction * (b - a): b - a is float32 either way; the rest is float32
    // only when the fraction is a Python float, which numpy then casts to
    // float32 before multiplying (NEP 50 weak scalar).
    const single = this._pyFrame && this._pyRate;
    const fr32 = f32(fr);
    for (const key of BLENDED) {
      const A = a[key], B = b[key], O = out[key];
      if (single) {
        for (let i = 0; i < O.length; i++) O[i] = f32(A[i] + f32(fr32 * f32(B[i] - A[i])));
      } else {
        for (let i = 0; i < O.length; i++) O[i] = A[i] + fr * f32(B[i] - A[i]);
      }
    }
    return out;
  }

  // reference(offset)["rpos"]: the blended site targets, into out (length 3 * sites).
  referenceRpos(offset, out) {
    const cur = this.gather(this.clip, this.frame + offset, this.anchor, this._cur);
    const alpha = Math.min(1, (this.blendCount + offset) / this.blendSteps);
    if (alpha < 1) {
      const old = this._outgoing(offset, this._old).rpos;
      const C = cur.rpos;
      if (this._pyFrame && this._pyRate) {
        const alpha32 = f32(alpha); // float32 arrays: numpy casts the Python float
        for (let i = 0; i < out.length; i++) out[i] = f32(old[i] + f32(alpha32 * f32(C[i] - old[i])));
      } else {
        for (let i = 0; i < out.length; i++) out[i] = old[i] + alpha * (C[i] - old[i]);
      }
    } else {
      out.set(cur.rpos);
    }
    return out;
  }

  // The reset state (reference() at alpha 1): root position, wxyz quaternion
  // and free-joint qvel.
  rootState(pos, quat, qvel) {
    const cur = this.gather(this.clip, this.frame, this.anchor, this._cur);
    pos.set(cur.rootPos);
    quat.set(cur.rootQuat);
    rootQvel(qvel, cur);
  }

  // The goal lookahead through the current raw clip (no blend), written to
  // out[at..]: per step i = 1..lookahead_steps-1 the root position change (3),
  // root qvel change (6) and the future rpos (3 * sites). Returns the end index.
  lookahead(out, at) {
    const now = this.gather(this.clip, this.frame, this.anchor, this._now);
    const qn = rootQvel(this._qa, now);
    let o = at;
    for (let i = 1; i < this.lookaheadSteps; i++) {
      const fut = this.gather(this.clip, this.frame + i * this.lookaheadStride, this.anchor, this._fut);
      out[o++] = fut.rootPos[0] - now.rootPos[0];
      out[o++] = fut.rootPos[1] - now.rootPos[1];
      out[o++] = fut.rootPos[2] - now.rootPos[2];
      const qf = rootQvel(this._qb, fut);
      for (let j = 0; j < 6; j++) out[o++] = qf[j] - qn[j];
      const r = fut.rpos;
      for (let j = 0; j < r.length; j++) out[o++] = r[j];
    }
    return o;
  }

  // Training's snap: the nearest clip under the weighted L2 twist metric.
  select(twist) {
    const t = this.twists, m = this.metric;
    let best = -1, bestDistance = Infinity;
    for (let i = 0; i < this.lengths.length; i++) {
      const d0 = t[3 * i] - twist[0], d1 = t[3 * i + 1] - twist[1], d2 = t[3 * i + 2] - twist[2];
      const distance = d0 * d0 * m[0] + d1 * d1 * m[1] + d2 * d2 * m[2];
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    }
    return best;
  }

  // Phase-preserving switch to the clip `twist` selects, re-anchoring its xy and
  // yaw onto the robot's root pose (qpos[q..q+3] position, qpos[q+3..q+7] wxyz).
  switch(twist, qpos, q = 0) {
    const clip = this.select(twist);
    const target = planar(this._target, qpos, qpos, q, q + 3);
    const old = this.clip, oldFrame = this.frame;
    const same = clip === old;
    const L = this.lengths;
    const newFrame = same ? oldFrame : rint((oldFrame * L[clip]) / L[old]) % L[clip];
    const idle = this.idleClips[clip] === 1;
    const rate = idle ? 1.0 : L[old] / L[clip];
    const x = idle ? oldFrame : newFrame * rate;
    const fresh = this.gather(clip, newFrame, ZERO3, this._fa);
    const p = inverse(this._p, planar(this._p, fresh.rootPos, fresh.rootQuat));
    const anchor = compose(this._p, target, p); // the new clip's anchor
    const ax = anchor[0], ay = anchor[1], ayaw = anchor[2];
    const lo = Math.floor(x);
    const a = this.gather(old, lo, ZERO3, this._fa);
    const pa = planar(this._pa, a.rootPos, a.rootQuat);
    const b = this.gather(old, lo + 1, ZERO3, this._fb);
    const pb = planar(this._pb, b.rootPos, b.rootQuat);
    const turn = pyMod(pb[2] - pa[2] + Math.PI, 2 * Math.PI) - Math.PI;
    const fr = x - lo;
    const prev = this._prev;
    prev[0] = pa[0] + fr * (pb[0] - pa[0]);
    prev[1] = pa[1] + fr * (pb[1] - pa[1]);
    prev[2] = pa[2] + fr * turn;
    this.previous = same ? clip : old;
    if (same) {
      this.previousAnchor[0] = ax; this.previousAnchor[1] = ay; this.previousAnchor[2] = ayaw;
    } else {
      compose(this.previousAnchor, target, inverse(prev, prev));
    }
    this.previousFrame = same ? newFrame : x;
    this.previousRate = same ? 1.0 : rate;
    this._pyFrame = this._pyRate = same || idle;
    this.clip = clip;
    this.frame = newFrame;
    this.anchor[0] = ax; this.anchor[1] = ay; this.anchor[2] = ayaw;
    if (idle) this.twist.fill(0);
    else { this.twist[0] = twist[0]; this.twist[1] = twist[1]; this.twist[2] = twist[2]; }
    if (!same) this.blendCount = 0;
    this.catchup.fill(0);
  }

  // One policy step forward.
  advance() {
    const L = this.lengths;
    this.frame += 1;
    if (this.frame >= L[this.clip]) {
      this.loops += 1;
      this.frame -= L[this.clip];
      compose(this.anchor, this.anchor, this.strides, 3 * this.clip);
    }
    this.previousFrame += this.previousRate;
    this._pyFrame = this._pyFrame && this._pyRate;
    if (this.previousFrame >= L[this.previous]) {
      this.previousFrame -= L[this.previous];
      this._pyFrame = false; // Python float minus a numpy int: numpy float64.
      compose(this.previousAnchor, this.previousAnchor, this.strides, 3 * this.previous);
    }
    if (this.blendCount < this.blendSteps) {
      this.blendCount += 1;
      const cur = this.gather(this.clip, this.frame, this.anchor, this._cur).rootPos;
      const old = this._outgoing(0, this._old).rootPos;
      const single = this._pyFrame && this._pyRate;
      for (let j = 0; j < 2; j++) {
        this.catchup[j] += single
          ? f32(f32(cur[j] - old[j]) / this.blendSteps)
          : (cur[j] - old[j]) / this.blendSteps;
      }
      if (this.blendCount === this.blendSteps) {
        this.anchor[0] -= this.catchup[0];
        this.anchor[1] -= this.catchup[1];
        this.catchup.fill(0);
      }
    }
  }

  // GaitReference.state_keys, for tests and debugging (clip: subset index).
  state() {
    return {
      clip: this.clip, previous: this.previous, frame: this.frame,
      previous_frame: this.previousFrame, previous_rate: this.previousRate,
      anchor: Array.from(this.anchor), previous_anchor: Array.from(this.previousAnchor),
      blend_count: this.blendCount, catchup: Array.from(this.catchup),
      twist: Array.from(this.twist), loops: this.loops,
    };
  }
}
