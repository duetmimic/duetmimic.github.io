// Worker-side frame extraction for the three.js renderer (no three.js, no DOM).
//
//   const extractor = new FrameExtractor(mujoco, model, demo.visual, demo.sim);
//   const { frame, transfer } = extractor.extract(data);   // after mj_step/mj_forward
//   postMessage({ type: 'frame', frame }, transfer);
//   // optional: the main thread posts frame.buffer back -> extractor.recycle(buffer)
//
//   frameLayout(demo)   // the layout of those frames from demo.json alone, for
//                       // the main thread (no MuJoCo): extractor.layout
//   frameChannels(demo.sim)          // frame.sensors' channel names and units
//   extractor.readSensors(data, out) // frame.sensors of the current state alone
//
// frame (all views share one ArrayBuffer, `frame.buffer`, the only transfer):
//   geoms         Float32Array(G * 12): per drawn geom, in demo.visual.geom_ids
//                 order, geom_xpos (3) then geom_xmat (9, row-major, MuJoCo's)
//   segments      Float32Array(maxSegments * 8): per muscle segment p0 (3),
//                 p1 (3), radius (m), activation in [0, 1]; rows past
//                 segmentCount are stale
//   segmentCount  number of valid segment rows
//   root          Float32Array(4): root body position (3) and heading (rad,
//                 CCW from +x, of demo.visual.root.forward in the world)
// and, appended after root, the extras (only with the sim config, the 4th
// argument; older page code ignores them, and frames from an extractor without
// it, or from an older worker, lack them):
//   wrench        Float32Array(6), with demo.sim.socket: [Fx, Fy, Fz, Tx, Ty,
//                 Tz], N and N m, MuJoCo world axes (z up): the load the
//                 prosthesis applies to the residual limb at the socket body's
//                 origin (xpos of sim.socket.body, s; r = sim.socket.root):
//                   F = -cfrc_int[s][3:6]
//                   T = -(cfrc_int[s][0:3] + (subtree_com[r] - xpos[s]) x cfrc_int[s][3:6])
//                 cfrc_int[s] is the wrench the femur (the socket's parent)
//                 applies to the socket's subtree, [torque about
//                 subtree_com[r]; force]. MuJoCo computes it in every mj_forward
//                 and mj_step because the model has acceleration-stage sensors
//                 (sim.socket.cfrc_int_fresh); nothing here calls
//                 mj_rnePostConstraint, which after an mj_step would read the
//                 integrated velocities. In stance F points up; at a strike
//                 it lags the contact by a few physics ticks, so at a
//                 contact's first loaded control step it can still point
//                 down, as in swing. Hanging still it is the prosthesis's
//                 weight acting at its centre of mass (subtree_com[s]).
//   sensors       Float32Array(N), with demo.sim.osl.observation: the OSL's
//                 processed, undelayed sensor frame from the current
//                 sensordata, its bus terms in demo.json order (bundle B:
//                 actpack 10, loadcell 6, contact 2 = 18), each value exactly
//                 as the policy input gets it (sim/observations.js
//                 readSensorFrame: components, clamp to +-saturation, minus the
//                 tare); the policy sees it `lag` control steps later.
//                 Channel names and units: frameLayout(demo).sensors.channels.
//
// Segments follow mjv_addGeoms (MuJoCo 3.11): one capsule between consecutive
// wrap points of a spatial tendon, none when either point is a pulley
// (wrap_obj -2), radius tendon_width, or half of it when both points lie on a
// wrapping geom (wrap_obj >= 0). Activation is data.act of the muscle that
// drives the tendon (0 for a passive tendon). Coordinates are MuJoCo's: z up.
//
// Allocation. extract() makes no MjData getter calls (each allocates a fresh
// view; the heap views are cached and re-read only when the wasm heap grew,
// which detaches them, or `data` changed) and no scratch arrays: it writes
// into a pooled frame, and allocates only with an empty pool (a new buffer).
// A frame object and its views are built once per ArrayBuffer object
// (_prepare), but a buffer the main thread posts back arrives as a new
// ArrayBuffer object (structured clone), so in the worker recycle() builds
// them for every frame: about 0.68 KB of short-lived garbage per frame with
// the extras, 0.45 KB without them, which is what the extractor before the
// extras made in extract() (Node 22, 2026-11-18). Views cannot be moved to
// another buffer, so frames that carry views cost that much. Only a caller
// in the same realm that recycles the very ArrayBuffer it got (the Node
// tests) reuses them and allocates nothing.

import { compileSensorReads, readSensorFrame } from '../sim/observations.js';

const TRN_TENDON = 3;      // mjtTrn.mjTRN_TENDON
const WRAP_SPHERE = 4;     // mjtWrap.mjWRAP_SPHERE
const WRAP_CYLINDER = 5;   // mjtWrap.mjWRAP_CYLINDER
const GEOM_STRIDE = 12;
const SEGMENT_STRIDE = 8;
const ROOT_FLOATS = 4;
const WRENCH_FLOATS = 6;
const POOL_SIZE = 6; // as many as the worker keeps in flight (worker.js)

export const FRAME_LAYOUT_VERSION = 1;
const WRENCH_LABELS = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'];
const WRENCH_UNITS = ['N', 'N', 'N', 'N m', 'N m', 'N m'];
const AXES = 'xyz';

// The OSL bus terms of demo.json (sensor terms with reads), in frame order.
function busTerms(sim) {
  const terms = sim && sim.osl && sim.osl.observation && sim.osl.observation.terms;
  return Array.isArray(terms) ? terms.filter((t) => Array.isArray(t.reads) && t.reads.length) : [];
}

// A channel's name when demo.json predates terms[*].channels: its read's name,
// plus the component's axis for a vector sensor.
function fallbackChannel(term, c) {
  for (const r of term.reads) {
    const dim = r.components ? r.components.length : r.size;
    if (c >= r.frame && c < r.frame + dim) {
      if (dim === 1) return r.name;
      const k = r.components ? r.components[c - r.frame] : c - r.frame;
      return `${r.name}_${AXES[k] ?? k}`;
    }
  }
  return `${term.name}_${c}`;
}

// frame.sensors' channels from demo.sim alone: [{index, term, name, unit}],
// the bus terms' channels in frame order, named by terms[*].channels (else by
// fallbackChannel: the read's name, plus the axis of a vector sensor). The one
// naming rule for frameLayout and the sensor charts (ui/sensors.js).
export function frameChannels(sim) {
  const channels = [];
  for (const t of busTerms(sim)) {
    for (let c = 0; c < t.frame_dim; c++) {
      channels.push({
        index: channels.length,
        term: t.name,
        name: (t.channels && t.channels[c]) || fallbackChannel(t, c),
        unit: (t.units && t.units[c]) || '',
      });
    }
  }
  return channels;
}

// The layout of the frames a FrameExtractor writes, from demo.json alone (the
// main thread has no MuJoCo): {version, floats, bytes, geoms, segments, root,
// wrench, sensors}. Each section is {offset (in Float32 elements), byteOffset,
// length (floats), ...}; wrench and sensors are null when demo.json lacks
// what they need (an extractor built from it writes no such section).
//   geoms     + {stride: 12, count}            segments + {stride: 8, count}
//   wrench    + {labels: Fx..Tz, units, body}  (frame.wrench)
//   sensors   + {channels: [{index, term, name, unit}]} (frame.sensors)
// Prefer a frame's own views (frame.wrench, frame.sensors: absent in frames of
// an older worker) to views built from these offsets, and check
// frame.buffer.byteLength === layout.bytes before building any.
// `maxSegments` defaults to demo.visual.muscles.max_segments (the extractor
// passes the model's own bound, which the visual export wrote there).
export function frameLayout({ visual, sim } = {}, { maxSegments } = {}) {
  if (!visual || !Array.isArray(visual.geom_ids)) {
    throw new Error('frameLayout: demo.visual.geom_ids is missing (re-export the visual section)');
  }
  const segmentCount = maxSegments ?? (visual.muscles && visual.muscles.max_segments);
  if (!Number.isInteger(segmentCount) || segmentCount < 0) {
    throw new Error('frameLayout: demo.visual.muscles.max_segments is missing (re-export the visual section)');
  }
  let at = 0;
  const section = (length, extra) => {
    const s = { offset: at, byteOffset: 4 * at, length, ...extra };
    at += length;
    return s;
  };
  const geomCount = visual.geom_ids.length;
  const geoms = section(geomCount * GEOM_STRIDE, { stride: GEOM_STRIDE, count: geomCount });
  const segments = section(segmentCount * SEGMENT_STRIDE, { stride: SEGMENT_STRIDE, count: segmentCount });
  const root = section(ROOT_FLOATS);
  const socket = sim && sim.socket;
  const wrench = socket && Number.isInteger(socket.body) && socket.cfrc_int_fresh !== false
    ? section(WRENCH_FLOATS, { labels: WRENCH_LABELS.slice(), units: WRENCH_UNITS.slice(), body: socket.body })
    : null;
  const channels = frameChannels(sim);
  const sensors = channels.length ? section(channels.length, { channels }) : null;
  return { version: FRAME_LAYOUT_VERSION, floats: at, bytes: 4 * at, geoms, segments, root, wrench, sensors };
}

export class FrameExtractor {
  // sim: demo.sim (optional): with it, frames carry the extras (wrench,
  // sensors) its sections allow; without it they are as before.
  constructor(mujoco, model, visual, sim = null) {
    this.mujoco = mujoco;
    this.model = model;
    if (!visual || !Array.isArray(visual.geom_ids)) {
      throw new Error('FrameExtractor: demo.visual.geom_ids is missing (re-export the visual section)');
    }
    const ngeom = model.ngeom;
    this.geomIds = Int32Array.from(visual.geom_ids);
    for (const g of this.geomIds) {
      if (g < 0 || g >= ngeom) throw new Error(`FrameExtractor: geom id ${g} outside the model (ngeom ${ngeom})`);
    }

    // Muscle tables: tendon id, act address (-1: passive), width; the bound on
    // segments uses the model's wrap types (a wrapping geom adds two points).
    const muscles = visual.muscles || {};
    const tendons = muscles.tendon_ids ?? Array.from({ length: model.ntendon }, (_, t) => t);
    const actOf = new Int32Array(model.ntendon).fill(-1);
    const trntype = model.actuator_trntype, trnid = model.actuator_trnid, actadr = model.actuator_actadr;
    for (let a = 0; a < model.nu; a++) {
      if (trntype[a] === TRN_TENDON && actadr[a] >= 0) actOf[trnid[2 * a]] = actadr[a];
    }
    const width = model.tendon_width, tenAdr = model.tendon_adr, tenNum = model.tendon_num;
    const wrapType = model.wrap_type;
    const widthScale = muscles.width_scale ?? 1;
    this.tendons = Int32Array.from(tendons);
    this.tendonAct = new Int32Array(this.tendons.length);
    this.tendonWidth = new Float64Array(this.tendons.length);
    let bound = 0;
    this.tendons.forEach((t, k) => {
      if (t < 0 || t >= model.ntendon) throw new Error(`FrameExtractor: tendon id ${t} outside the model`);
      this.tendonAct[k] = actOf[t];
      this.tendonWidth[k] = width[t] * widthScale;
      let points = 0;
      for (let j = tenAdr[t]; j < tenAdr[t] + tenNum[t]; j++) {
        points += (wrapType[j] === WRAP_SPHERE || wrapType[j] === WRAP_CYLINDER) ? 2 : 1;
      }
      bound += Math.max(points - 1, 0);
    });
    this.maxSegments = bound;

    const root = visual.root || {};
    this.rootBody = root.body ?? 1;
    if (this.rootBody < 0 || this.rootBody >= model.nbody) throw new Error(`FrameExtractor: root body ${this.rootBody} outside the model`);
    const f = root.forward ?? [1, 0, 0];
    this.forward = Float64Array.from(f);

    // The layout, extras included (frameLayout, with the model's own bound).
    const layout = frameLayout({ visual, sim }, { maxSegments: bound });
    this.layout = layout;

    // Socket wrench: the body, its tree root (cfrc_int's reference point).
    this.socketBody = -1;
    this.socketRoot = -1;
    if (layout.wrench) {
      const socket = sim.socket;
      const body = socket.body;
      if (body <= 0 || body >= model.nbody) throw new Error(`FrameExtractor: socket body ${body} outside the model`);
      const rootId = model.body_rootid[body];
      if (socket.root !== undefined && socket.root !== rootId) {
        throw new Error(`FrameExtractor: socket body ${body} has tree root ${rootId}, demo.json says ${socket.root}`);
      }
      if (socket.parent !== undefined && socket.parent !== model.body_parentid[body]) {
        throw new Error(`FrameExtractor: socket body ${body} has parent ${model.body_parentid[body]}, demo.json says ${socket.parent}`);
      }
      this.socketBody = body;
      this.socketRoot = rootId;
    }

    // OSL sensor frame: each bus term's compiled reads and its start in it.
    this.sensorTerms = null;
    if (layout.sensors) {
      const nsensordata = model.nsensordata;
      let start = 0;
      this.sensorTerms = busTerms(sim).map((t) => {
        const reads = compileSensorReads(t.reads);
        for (const r of reads) {
          const last = r.adr + (r.components ? Math.max(...r.components) : r.dim - 1);
          if (r.adr < 0 || last >= nsensordata) {
            throw new Error(`FrameExtractor: OSL read at sensordata ${r.adr} outside the model (nsensordata ${nsensordata})`);
          }
        }
        const term = { start, reads };
        start += t.frame_dim;
        return term;
      });
    }

    this.geomCount = this.geomIds.length;
    this.geomOffset = layout.geoms.offset;
    this.segmentOffset = layout.segments.offset;
    this.rootOffset = layout.root.offset;
    this.wrenchOffset = layout.wrench ? layout.wrench.offset : -1;
    this.sensorsOffset = layout.sensors ? layout.sensors.offset : -1;
    this.sensorCount = layout.sensors ? layout.sensors.length : 0;
    this.floatCount = layout.floats;
    this.byteLength = layout.bytes;
    this._pool = [];            // ready {frame, transfer} of recycled buffers
    this._ready = new WeakMap(); // buffer -> its {frame, transfer}
    this._data = null;
    this._views = null;
  }

  // The {frame, transfer} of a buffer of this layout, built once per
  // ArrayBuffer object (a buffer back from the main thread is a new one).
  _prepare(buffer) {
    const L = this.layout;
    const frame = {
      geoms: new Float32Array(buffer, L.geoms.byteOffset, L.geoms.length),
      segments: new Float32Array(buffer, L.segments.byteOffset, L.segments.length),
      segmentCount: 0,
      root: new Float32Array(buffer, L.root.byteOffset, L.root.length),
      buffer,
      // No time/epoch: worker.js sets them; a frame without a finite time is
      // drawn untimed (render/scene.js setFrame).
    };
    if (L.wrench) frame.wrench = new Float32Array(buffer, L.wrench.byteOffset, L.wrench.length);
    if (L.sensors) frame.sensors = new Float32Array(buffer, L.sensors.byteOffset, L.sensors.length);
    const out = { frame, transfer: [buffer] };
    this._ready.set(buffer, out);
    return out;
  }

  // Live views into MjData (re-read when the wasm heap grew or data changed).
  _bind(data) {
    const v = this._views;
    if (v && this._data === data && v.geomXpos.length !== 0) return v;
    this._data = data;
    this._views = {
      geomXpos: data.geom_xpos,
      geomXmat: data.geom_xmat,
      wrapXpos: data.wrap_xpos,
      wrapObj: data.wrap_obj,
      wrapAdr: data.ten_wrapadr,
      wrapNum: data.ten_wrapnum,
      act: data.act,
      xpos: data.xpos,
      xmat: data.xmat,
      cfrcInt: this.socketBody >= 0 ? data.cfrc_int : null,
      subtreeCom: this.socketBody >= 0 ? data.subtree_com : null,
      sensordata: this.sensorTerms ? data.sensordata : null,
    };
    return this._views;
  }

  extract(data) {
    const v = this._bind(data);
    const out = this._pool.pop() || this._prepare(new ArrayBuffer(this.byteLength));
    const frame = out.frame;
    const geoms = frame.geoms, segments = frame.segments, root = frame.root;

    // Geoms: xpos + xmat.
    const gx = v.geomXpos, gm = v.geomXmat, ids = this.geomIds;
    for (let i = 0, o = 0; i < ids.length; i++, o += GEOM_STRIDE) {
      const g = ids[i], p = 3 * g, m = 9 * g;
      geoms[o] = gx[p]; geoms[o + 1] = gx[p + 1]; geoms[o + 2] = gx[p + 2];
      for (let k = 0; k < 9; k++) geoms[o + 3 + k] = gm[m + k];
    }

    // Muscle segments.
    const wx = v.wrapXpos, wo = v.wrapObj, wa = v.wrapAdr, wn = v.wrapNum, act = v.act;
    const tendons = this.tendons, tAct = this.tendonAct, tWidth = this.tendonWidth;
    const max = this.maxSegments;
    let n = 0;
    for (let k = 0; k < tendons.length; k++) {
      const t = tendons[k];
      const start = wa[t], end = start + wn[t] - 1;
      const w = tWidth[k];
      const a = tAct[k] >= 0 ? act[tAct[k]] : 0;
      const level = a < 0 ? 0 : (a > 1 ? 1 : a);
      for (let j = start; j < end; j++) {
        const o0 = wo[j], o1 = wo[j + 1];
        if (o0 === -2 || o1 === -2) continue;
        if (n === max) break;  // cannot happen with the model's bound; never overflow
        const o = n * SEGMENT_STRIDE, p = 3 * j;
        segments[o] = wx[p]; segments[o + 1] = wx[p + 1]; segments[o + 2] = wx[p + 2];
        segments[o + 3] = wx[p + 3]; segments[o + 4] = wx[p + 4]; segments[o + 5] = wx[p + 5];
        segments[o + 6] = (o0 >= 0 && o1 >= 0) ? 0.5 * w : w;
        segments[o + 7] = level;
        n++;
      }
    }
    frame.segmentCount = n;

    // Root position and heading.
    const b = this.rootBody, xp = v.xpos, xm = v.xmat, f = this.forward;
    root[0] = xp[3 * b]; root[1] = xp[3 * b + 1]; root[2] = xp[3 * b + 2];
    const m = 9 * b;
    const fx = xm[m] * f[0] + xm[m + 1] * f[1] + xm[m + 2] * f[2];
    const fy = xm[m + 3] * f[0] + xm[m + 4] * f[1] + xm[m + 5] * f[2];
    root[3] = Math.atan2(fy, fx);

    // Socket wrench: cfrc_int's torque moved from subtree_com[root] to the
    // socket origin (tau + d x f, d = subtree_com - xpos), both negated.
    if (this.socketBody >= 0) {
      const wrench = frame.wrench, c = v.cfrcInt, com = v.subtreeCom;
      const k = 6 * this.socketBody, s = 3 * this.socketBody, r = 3 * this.socketRoot;
      const f0 = c[k + 3], f1 = c[k + 4], f2 = c[k + 5];
      const d0 = com[r] - xp[s], d1 = com[r + 1] - xp[s + 1], d2 = com[r + 2] - xp[s + 2];
      wrench[0] = -f0; wrench[1] = -f1; wrench[2] = -f2;
      wrench[3] = -(c[k] + d1 * f2 - d2 * f1);
      wrench[4] = -(c[k + 1] + d2 * f0 - d0 * f2);
      wrench[5] = -(c[k + 2] + d0 * f1 - d1 * f0);
    }

    // OSL sensor frame.
    const terms = this.sensorTerms;
    if (terms) {
      const sensors = frame.sensors, sd = v.sensordata;
      for (let i = 0; i < terms.length; i++) readSensorFrame(terms[i].reads, sd, sensors, terms[i].start);
    }
    return out;
  }

  // frame.sensors of the current state alone, written into out[offset ..
  // offset + sensorCount) as extract() writes it (worker.js: the sensor rows
  // of the steps whose frame did not go out). Returns sensorCount (0: the
  // frames carry no sensors, nothing written). No allocation.
  readSensors(data, out, offset = 0) {
    const terms = this.sensorTerms;
    if (!terms) return 0;
    const sd = this._bind(data).sensordata;
    for (let i = 0; i < terms.length; i++) readSensorFrame(terms[i].reads, sd, out, offset + terms[i].start);
    return this.sensorCount;
  }

  // Give a frame's buffer back (after the main thread posted it home) to skip
  // a new buffer on a later extract(); a new ArrayBuffer object gets its frame
  // object and views here. Accepts a frame or its ArrayBuffer; ignores a
  // buffer of another size (another layout, or detached) and one already
  // pooled.
  recycle(frameOrBuffer) {
    const buffer = frameOrBuffer instanceof ArrayBuffer ? frameOrBuffer : frameOrBuffer?.buffer;
    if (!buffer || buffer.byteLength !== this.byteLength || this._pool.length >= POOL_SIZE) return;
    const out = this._ready.get(buffer) || this._prepare(buffer);
    if (this._pool.includes(out)) return;
    this._pool.push(out);
  }
}
