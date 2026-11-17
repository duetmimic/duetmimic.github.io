// Worker-side frame extraction for the three.js renderer (no three.js, no DOM).
//
//   const extractor = new FrameExtractor(mujoco, model, demo.visual);
//   const { frame, transfer } = extractor.extract(data);   // after mj_step/mj_forward
//   postMessage({ type: 'frame', frame }, transfer);
//   // optional: the main thread posts frame.buffer back -> extractor.recycle(buffer)
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
//
// Segments follow mjv_addGeoms (MuJoCo 3.11): one capsule between consecutive
// wrap points of a spatial tendon, none when either point is a pulley
// (wrap_obj -2), radius tendon_width, or half of it when both points lie on a
// wrapping geom (wrap_obj >= 0). Activation is data.act of the muscle that
// drives the tendon (0 for a passive tendon). Coordinates are MuJoCo's: z up.
//
// Nothing per call allocates except the outgoing buffer, and not even that
// when buffers come back through recycle(). Views into the wasm heap are
// cached and re-read only when the heap grows (old views detach) or `data`
// changes.

const TRN_TENDON = 3;      // mjtTrn.mjTRN_TENDON
const WRAP_SPHERE = 4;     // mjtWrap.mjWRAP_SPHERE
const WRAP_CYLINDER = 5;   // mjtWrap.mjWRAP_CYLINDER
const GEOM_STRIDE = 12;
const SEGMENT_STRIDE = 8;
const ROOT_FLOATS = 4;
const POOL_SIZE = 6; // as many as the worker keeps in flight (worker.js)

export class FrameExtractor {
  constructor(mujoco, model, visual) {
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

    this.geomCount = this.geomIds.length;
    this.geomOffset = 0;
    this.segmentOffset = this.geomCount * GEOM_STRIDE;
    this.rootOffset = this.segmentOffset + this.maxSegments * SEGMENT_STRIDE;
    this.floatCount = this.rootOffset + ROOT_FLOATS;
    this.byteLength = this.floatCount * 4;
    this._pool = [];
    this._data = null;
    this._views = null;
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
    };
    return this._views;
  }

  extract(data) {
    const v = this._bind(data);
    const buffer = this._pool.pop() || new ArrayBuffer(this.byteLength);
    const geoms = new Float32Array(buffer, this.geomOffset * 4, this.geomCount * GEOM_STRIDE);
    const segments = new Float32Array(buffer, this.segmentOffset * 4, this.maxSegments * SEGMENT_STRIDE);
    const root = new Float32Array(buffer, this.rootOffset * 4, ROOT_FLOATS);

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

    // Root position and heading.
    const b = this.rootBody, xp = v.xpos, xm = v.xmat, f = this.forward;
    root[0] = xp[3 * b]; root[1] = xp[3 * b + 1]; root[2] = xp[3 * b + 2];
    const m = 9 * b;
    const fx = xm[m] * f[0] + xm[m + 1] * f[1] + xm[m + 2] * f[2];
    const fy = xm[m + 3] * f[0] + xm[m + 4] * f[1] + xm[m + 5] * f[2];
    root[3] = Math.atan2(fy, fx);

    return { frame: { geoms, segments, segmentCount: n, root, buffer }, transfer: [buffer] };
  }

  // Give a frame's buffer back (after the main thread posted it home) to skip
  // the allocation on a later extract(). Accepts a frame or its ArrayBuffer.
  recycle(frameOrBuffer) {
    const buffer = frameOrBuffer instanceof ArrayBuffer ? frameOrBuffer : frameOrBuffer?.buffer;
    if (buffer && buffer.byteLength === this.byteLength && this._pool.length < POOL_SIZE) this._pool.push(buffer);
  }
}
