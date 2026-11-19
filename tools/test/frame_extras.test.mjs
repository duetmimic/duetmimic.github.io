// The data the X-ray and sensing views draw (node --test; no browser):
// render/frame.js's frame extras (frame.wrench, frame.sensors), its
// frameLayout(), and the demo.json contract behind them (sim.socket,
// sim.osl.sensor_set, the OSL terms' channels, units and read anchors,
// visual.overlays). tools/DESIGN.md, "Frame extras and overlays".
//
//   cd tools && node --test test/frame_extras.test.mjs
//
// Needs the asset folder (paths.mjs ASSETS: demo.json, physics.mjb.gz). The
// trace checks need tools/traces/trace.bin.gz recorded from the assets' bundle
// (python -m duet_export.trace; DUET_TRACE), else they skip. At every control
// boundary of the trace (osl_deploy's own Simulation), with the recorded
// state set and mj_forward:
//   - frame.wrench equals the native formula on the recorded rows
//     (socket_cfrc, root_subtree_com, socket_xpos) to float32 rounding;
//   - frame.sensors equals osl_deploy's undelayed OSL sensor frame (osl_frame);
//   - in stance (heel + toe contact > 200 N, also at the boundary before) the
//     force on the limb points up (world z > 0; at most 1% of these
//     boundaries may not) and follows the total contact force. A contact
//     onset's first loaded boundary is left out: the load lags the contact
//     there by a few physics ticks and can still point down, as in swing.
// Without the trace: the overlays against the physics model at several poses,
// the layout (extras on and off), buffer recycling, no MjData getter call per
// extract(), cfrc_int refreshed by mj_step, the wrench's sign and lever arm
// without the cfrc_int formula (the prosthesis hanging still: its weight at
// its centre of mass), and extract() timing.

import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { fetchBytes, loadContainer } from '../../demo/js/lib/container.js';
import { loadMujoco, modelFromBytes } from '../../demo/js/lib/mujoco.js';
import { FrameExtractor, frameLayout } from '../../demo/js/render/frame.js';
import { ASSETS, TRACE } from './paths.mjs';

const DEMO = new URL('demo.json', ASSETS);
const missing = ['demo.json', 'physics.mjb.gz'].filter((f) => !existsSync(fileURLToPath(new URL(f, ASSETS))));
const SKIP = missing.length ? `missing ${fileURLToPath(ASSETS)}{${missing.join(',')}}: run python -m duet_export` : false;
const demo = SKIP ? null : JSON.parse(readFileSync(fileURLToPath(DEMO), 'utf8'));
const CONTRACT_SKIP = SKIP || (!demo.sim.socket || !demo.visual.overlays
  ? 'demo.json lacks sim.socket or visual.overlays: re-run python -m duet_export --only visual,sim' : false);

const ROLES = ['bone', 'socket', 'hardware', 'cover'];
const KINDS = ['imu', 'load_cell', 'contact', 'encoder'];
const KIND_OF_SENSOR = { JOINTPOS: 'encoder', JOINTVEL: 'encoder', GYRO: 'imu', ACCELEROMETER: 'imu', FORCE: 'load_cell', TORQUE: 'load_cell', TOUCH: 'contact' };
const STANCE_N = 200; // heel + toe contact force that counts as stance
// Stance boundaries (loaded at this and the previous boundary) where the load
// on the limb may still point down. A pull or a fall can do that on correct
// physics (in a closed-loop run on B, 1 of 422 such boundaries, during a
// 400 N pull on the pelvis). The sign itself is pinned by the hanging check.
const STANCE_DOWN_FRACTION = 0.01;

const fmt = (x) => (x === 0 ? '0' : x.toExponential(2));
const row = (a, width, i) => a.subarray(width * i, width * (i + 1));

// The OSL bus terms of demo.json in frame order, with each frame channel's read.
function busChannels(sim) {
  const out = [];
  for (const term of sim.osl.observation.terms) {
    if (!term.reads || !term.reads.length) continue;
    const reads = new Array(term.frame_dim);
    for (const r of term.reads) {
      const dim = r.components ? r.components.length : r.size;
      for (let c = 0; c < dim; c++) reads[r.frame + c] = r;
    }
    for (let c = 0; c < term.frame_dim; c++) out.push({ term, read: reads[c], c });
  }
  return out;
}

// world = geom_xpos + geom_xmat . local (row-major), for MuJoCo geom id g.
function geomPoint(d, g, local) {
  const p = d.geom_xpos, m = d.geom_xmat, o = 3 * g, k = 9 * g;
  return [0, 1, 2].map((i) => p[o + i] + m[k + 3 * i] * local[0] + m[k + 3 * i + 1] * local[1] + m[k + 3 * i + 2] * local[2]);
}

// Row-major matrix of a unit quaternion (w x y z).
function quatMat([w, x, y, z]) {
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

const maxDiff = (a, b) => a.reduce((m, x, i) => Math.max(m, Math.abs(x - b[i])), 0);

// Where a touch sensor's box zone meets the foot (duet_export/visual.py
// contact_point): of the collision geoms moving with the site's body, the
// point of a geom's axis (a capsule's or cylinder's segment, else its centre)
// inside the box nearest the box's centre; the centre when none enters it.
// Returns [point, geom id or -1].
function contactPoint(m, d, site) {
  const c = Array.from(d.site_xpos.subarray(3 * site, 3 * site + 3));
  const r = d.site_xmat.subarray(9 * site, 9 * site + 9), half = m.site_size.subarray(3 * site, 3 * site + 3);
  const weld = m.body_weldid[m.site_bodyid[site]];
  const CAPSULE = 3, CYLINDER = 5; // mjtGeom
  let best = c, bestGeom = -1, bestDistance = Infinity;
  for (let g = 0; g < m.ngeom; g++) {
    if (m.body_weldid[m.geom_bodyid[g]] !== weld || !(m.geom_contype[g] || m.geom_conaffinity[g])) continue;
    const mid = Array.from(d.geom_xpos.subarray(3 * g, 3 * g + 3));
    let point = mid;
    if (m.geom_type[g] === CAPSULE || m.geom_type[g] === CYLINDER) {
      const h = m.geom_size[3 * g + 1], x = d.geom_xmat;
      const reach = [x[9 * g + 2] * h, x[9 * g + 5] * h, x[9 * g + 8] * h]; // the geom's z axis, scaled
      const a = mid.map((v, i) => v - reach[i]), ab = reach.map((v) => 2 * v);
      const t = Math.min(1, Math.max(0, ((c[0] - a[0]) * ab[0] + (c[1] - a[1]) * ab[1] + (c[2] - a[2]) * ab[2]) / (ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)));
      point = a.map((v, i) => v + t * ab[i]);
    }
    const dp = point.map((v, i) => v - c[i]);
    const inside = [0, 1, 2].every((j) => Math.abs(r[j] * dp[0] + r[3 + j] * dp[1] + r[6 + j] * dp[2]) <= half[j] + 1e-9);
    const distance = Math.hypot(...dp);
    if (inside && distance < bestDistance) { best = point; bestGeom = g; bestDistance = distance; }
  }
  return [best, bestGeom];
}

describe('frame extras and overlays', { skip: CONTRACT_SKIP }, () => {
  let mujoco, model, data, OBJ, sensorType;
  const sim = demo && demo.sim, visual = demo && demo.visual;

  before(async () => {
    mujoco = await loadMujoco();
    ({ model, data } = modelFromBytes(mujoco, await fetchBytes(new URL(demo.physics.file, ASSETS)), 'physics.mjb'));
    OBJ = mujoco.mjtObj;
    const names = Object.keys(KIND_OF_SENSOR);
    sensorType = new Map(names.map((n) => [mujoco.mjtSensor[`mjSENS_${n}`].value, n]));
  });

  after(() => {
    data?.delete();
    model?.delete();
  });

  const sensorAt = (adr) => {
    const hits = [];
    for (let s = 0; s < model.nsensor; s++) if (model.sensor_adr[s] === adr) hits.push(s);
    assert.equal(hits.length, 1, `one sensor at sensordata ${adr}`);
    return hits[0];
  };

  test('sim: socket, OSL channels and units, read anchors and sensor_set match the model', () => {
    const s = sim.socket;
    assert.equal(mujoco.mj_name2id(model, OBJ.mjOBJ_BODY.value, s.name), s.body, `body ${s.name}`);
    assert.equal(model.body_parentid[s.body], s.parent, 'socket parent');
    assert.equal(model.body_rootid[s.body], s.root, 'socket tree root');
    assert.ok(Math.abs(model.body_subtreemass[s.body] - s.subtree_mass) < 1e-6, 'socket subtree mass');
    assert.equal(s.cfrc_int_fresh, true, 'the model refreshes cfrc_int in mj_forward and mj_step');
    assert.ok(sim.osl.sensor_set && typeof sim.osl.sensor_set === 'object', 'sim.osl.sensor_set');
    for (const value of Object.values(sim.osl.sensor_set)) assert.equal(typeof value, 'boolean');
    let reads = 0;
    for (const term of sim.osl.observation.terms) {
      assert.equal(term.channels.length, term.frame_dim, `${term.name} channels`);
      assert.equal(term.units.length, term.frame_dim, `${term.name} units`);
      assert.equal(new Set(term.channels).size, term.frame_dim, `${term.name}: distinct channel names`);
      for (const r of term.reads) {
        const sensor = sensorAt(r.adr), a = r.anchor;
        const objtype = model.sensor_objtype[sensor], objid = model.sensor_objid[sensor];
        assert.equal(a.kind, objtype === OBJ.mjOBJ_SITE.value ? 'site' : objtype === OBJ.mjOBJ_JOINT.value ? 'joint' : '?', `${r.name} anchor kind`);
        assert.equal(a.id, objid, `${r.name} anchor id`);
        assert.equal(a.body, a.kind === 'site' ? model.site_bodyid[objid] : model.jnt_bodyid[objid], `${r.name} anchor body`);
        assert.equal(mujoco.mj_name2id(model, objtype, a.name), objid, `${r.name} anchor name ${a.name}`);
        reads += 1;
      }
    }
    console.log(`# socket ${s.name} (body ${s.body}, parent ${s.parent_name}, root ${s.root}): ${s.subtree_mass.toFixed(3)} kg below it; `
      + `${reads} OSL reads anchored; sensor_set ${JSON.stringify(sim.osl.sensor_set)}`);
  });

  test('visual.overlays: roles, socket, markers and zones are consistent', () => {
    const o = visual.overlays, ids = visual.geom_ids, G = ids.length;
    const socket = sim.socket.body;
    const inProsthesis = (b) => { while (b > 0 && b !== socket) b = model.body_parentid[b]; return b === socket; };
    // Roles: one per drawn geom, from its body.
    assert.equal(o.roles.length, G, 'one role per drawn geom');
    const counts = Object.fromEntries(ROLES.map((r) => [r, 0]));
    o.roles.forEach((role, i) => {
      assert.ok(ROLES.includes(role), `role ${role}`);
      counts[role] += 1;
      const body = model.geom_bodyid[ids[i]];
      if (role === 'socket') assert.equal(body, socket, `geom ${ids[i]}: socket role off the socket body`);
      else if (role === 'bone') assert.ok(!inProsthesis(body), `geom ${ids[i]}: a bone on the prosthesis`);
      else assert.ok(inProsthesis(body) && body !== socket, `geom ${ids[i]}: ${role} outside the prosthesis`);
    });
    for (let i = 0; i < G; i++) {
      if (model.geom_bodyid[ids[i]] === socket) assert.equal(o.roles[i], 'socket', `geom ${ids[i]} on the socket body`);
    }
    // Socket: a drawn geom on the socket body.
    const sk = o.socket;
    assert.ok(Number.isInteger(sk.geom) && sk.geom >= 0 && sk.geom < G, 'socket geom index');
    assert.equal(ids[sk.geom], sk.geom_id);
    assert.equal(model.geom_bodyid[sk.geom_id], socket, 'the socket geom is on the socket body');
    assert.equal(o.roles[sk.geom], 'socket');
    // Markers: channels tile the frame's sensors; each on a geom of its sensor's rigid body.
    const channels = busChannels(sim), N = channels.length;
    const seen = new Array(N).fill(0);
    const keys = new Set();
    for (const mk of o.markers) {
      assert.ok(!keys.has(mk.key), `marker key ${mk.key} repeats`);
      keys.add(mk.key);
      assert.ok(KINDS.includes(mk.kind), `${mk.key}: kind ${mk.kind}`);
      assert.ok(typeof mk.label === 'string' && mk.label.length, `${mk.key}: label`);
      assert.ok(Number.isInteger(mk.geom) && mk.geom >= 0 && mk.geom < G, `${mk.key}: geom index`);
      assert.equal(ids[mk.geom], mk.geom_id, `${mk.key}: geom_id`);
      const geomBody = model.geom_bodyid[mk.geom_id];
      assert.equal(model.body_weldid[geomBody], model.body_weldid[mk.anchor.body], `${mk.key}: geom ${mk.geom_id} does not move with body ${mk.anchor.body}`);
      assert.equal(geomBody, mk.anchor.body, `${mk.key}: on its sensor's own body (bundle B)`);
      assert.ok(mk.channels.length > 0, `${mk.key}: channels`);
      for (const c of mk.channels) {
        assert.ok(Number.isInteger(c) && c >= 0 && c < N, `${mk.key}: channel ${c}`);
        seen[c] += 1;
        const { read } = channels[c];
        assert.equal(read.anchor.kind, mk.anchor.kind, `${mk.key}: channel ${c} is read at a ${read.anchor.kind}`);
        assert.equal(read.anchor.id, mk.anchor.id, `${mk.key}: channel ${c} comes from ${read.name}`);
        assert.equal(KIND_OF_SENSOR[sensorType.get(model.sensor_type[sensorAt(read.adr)])], mk.kind, `${mk.key}: ${read.name} kind`);
      }
    }
    assert.deepEqual(seen, new Array(N).fill(1), 'every sensors channel feeds exactly one marker');
    // Zones: one per contact marker on a site, with its box.
    const contacts = o.markers.filter((mk) => mk.kind === 'contact' && mk.anchor.kind === 'site');
    assert.deepEqual(o.zones.map((z) => z.key), contacts.map((mk) => mk.key), 'a zone per touch sensor');
    for (const z of o.zones) {
      const mk = contacts.find((m) => m.key === z.key);
      assert.equal(z.geom, mk.geom);
      assert.equal(z.geom_id, mk.geom_id);
      const site = mk.anchor.id;
      assert.equal(z.type, 'box');
      assert.equal(model.site_type[site], mujoco.mjtGeom.mjGEOM_BOX.value, `${z.key}: a box site`);
      assert.ok(maxDiff(z.half, model.site_size.subarray(3 * site, 3 * site + 3)) < 1e-9, `${z.key}: half sizes`);
      assert.ok(Math.abs(Math.hypot(...z.quat) - 1) < 1e-8, `${z.key}: unit quaternion`);
    }
    console.log(`# overlays: roles ${JSON.stringify(counts)}; socket geom ${sk.geom_id} local [${sk.local}]; `
      + `markers ${o.markers.map((mk) => `${mk.key} (${mk.kind}, geom ${mk.geom_id}, channels ${mk.channels.join(',')})`).join('; ')}; `
      + `zones ${o.zones.map((z) => z.key).join(', ')}`);
  });

  test('visual.overlays: material categories agree with the roles (bundle B)', async () => {
    const { arrays } = await loadContainer(new URL(visual.file, ASSETS));
    const categoryRole = { bone: 'bone', socket: 'socket', hardware: 'hardware', osl_cover: 'cover' };
    visual.overlays.roles.forEach((role, i) => {
      const material = visual.materials[arrays.geom_material[i]];
      assert.equal(categoryRole[material.category], role, `geom ${visual.geom_ids[i]}: role ${role}, material ${material.category}`);
    });
  });

  test('visual.overlays: points and boxes ride their geoms at any pose', () => {
    const o = visual.overlays, s = sim.socket.body;
    const nq = model.nq;
    const qpos0 = Float64Array.from(model.qpos0);
    let rng = 12345;
    const random = () => { rng = (rng * 1103515245 + 12345) % 2147483648; return rng / 2147483648 - 0.5; };
    let worstPoint = 0, worstAxis = 0, worstBox = 0;
    const contactGeoms = new Set();
    for (let pose = 0; pose < 6; pose++) {
      mujoco.mj_resetData(model, data);
      data.qpos.set(qpos0);
      if (pose > 0) {
        // Every hinge and slide joint off its reference, the root moved and turned.
        for (let j = 0; j < model.njnt; j++) {
          const type = model.jnt_type[j], adr = model.jnt_qposadr[j];
          if (type === mujoco.mjtJoint.mjJNT_HINGE.value) data.qpos[adr] += 0.6 * random();
          else if (type === mujoco.mjtJoint.mjJNT_SLIDE.value) data.qpos[adr] += 0.02 * random();
          else if (type === mujoco.mjtJoint.mjJNT_FREE.value) {
            data.qpos[adr] += random(); data.qpos[adr + 1] += random();
            const h = random(), [w, x, y, z] = [Math.cos(h), 0.3 * random(), 0.3 * random(), Math.sin(h)];
            const n = Math.hypot(w, x, y, z);
            data.qpos.set([w / n, x / n, y / n, z / n], adr + 3);
          }
        }
      }
      assert.equal(data.qpos.length, nq);
      mujoco.mj_kinematics(model, data);
      const xpos = data.xpos, sitePos = data.site_xpos, siteMat = data.site_xmat, xanchor = data.xanchor, xaxis = data.xaxis;
      const sp = geomPoint(data, o.socket.geom_id, o.socket.local);
      worstPoint = Math.max(worstPoint, maxDiff(sp, xpos.subarray(3 * s, 3 * s + 3)));
      for (const mk of o.markers) {
        const p = geomPoint(data, mk.geom_id, mk.local), id = mk.anchor.id;
        // A touch zone's glyph sits where the zone meets the foot, every
        // other marker at its site or joint anchor.
        const zone = o.zones.find((z) => z.key === mk.key);
        let want = mk.anchor.kind === 'site' ? sitePos.subarray(3 * id, 3 * id + 3) : xanchor.subarray(3 * id, 3 * id + 3);
        if (zone) {
          const [point, geom] = contactPoint(model, data, id);
          assert.ok(geom >= 0, `${mk.key}: a collision geom enters the zone`);
          contactGeoms.add(`${mk.key} on geom ${geom}`);
          want = point;
          worstPoint = Math.max(worstPoint, maxDiff(geomPoint(data, zone.geom_id, zone.local), sitePos.subarray(3 * id, 3 * id + 3)));
        }
        worstPoint = Math.max(worstPoint, maxDiff(p, want));
        if (mk.axis) {
          const m = data.geom_xmat, k = 9 * mk.geom_id, a = mk.axis;
          const axis = [0, 1, 2].map((i) => m[k + 3 * i] * a[0] + m[k + 3 * i + 1] * a[1] + m[k + 3 * i + 2] * a[2]);
          worstAxis = Math.max(worstAxis, maxDiff(axis, xaxis.subarray(3 * id, 3 * id + 3)));
        }
      }
      for (const z of o.zones) {
        const mk = o.markers.find((m) => m.key === z.key), site = mk.anchor.id;
        const g = data.geom_xmat.subarray(9 * z.geom_id, 9 * z.geom_id + 9), q = quatMat(z.quat);
        const world = [];
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) world.push(g[3 * i] * q[j] + g[3 * i + 1] * q[3 + j] + g[3 * i + 2] * q[6 + j]);
        worstBox = Math.max(worstBox, maxDiff(world, siteMat.subarray(9 * site, 9 * site + 9)));
      }
    }
    console.log(`# overlays at 6 poses: points ${fmt(worstPoint)} m, joint axes ${fmt(worstAxis)}, box orientations ${fmt(worstBox)}; `
      + `contact glyphs ${[...contactGeoms].join(', ')}`);
    assert.ok(worstPoint < 1e-8, `overlay points off by ${worstPoint} m`);
    assert.ok(worstAxis < 1e-8, `joint axes off by ${worstAxis}`);
    assert.ok(worstBox < 1e-8, `zone boxes off by ${worstBox}`);
  });

  test('frameLayout: sections, channels, and the extractor writes exactly that layout', () => {
    const layout = frameLayout(demo);
    const ex = new FrameExtractor(mujoco, model, visual, sim);
    assert.deepEqual(ex.layout, layout, 'frameLayout(demo) is the extractor\'s layout');
    const channels = busChannels(sim), N = channels.length;
    const G = visual.geom_ids.length, S = visual.muscles.max_segments;
    assert.deepEqual([layout.geoms.offset, layout.geoms.length], [0, 12 * G]);
    assert.deepEqual([layout.segments.offset, layout.segments.length], [12 * G, 8 * S]);
    assert.deepEqual([layout.root.offset, layout.root.length], [12 * G + 8 * S, 4]);
    assert.deepEqual([layout.wrench.offset, layout.wrench.length], [12 * G + 8 * S + 4, 6]);
    assert.deepEqual([layout.sensors.offset, layout.sensors.length], [12 * G + 8 * S + 10, N]);
    assert.equal(layout.floats, 12 * G + 8 * S + 10 + N);
    assert.equal(layout.bytes, 4 * layout.floats);
    assert.equal(ex.byteLength, layout.bytes);
    for (const key of ['geoms', 'segments', 'root', 'wrench', 'sensors']) assert.equal(layout[key].byteOffset, 4 * layout[key].offset);
    assert.deepEqual(layout.wrench.labels, ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']);
    assert.deepEqual(layout.wrench.units, ['N', 'N', 'N', 'N m', 'N m', 'N m']);
    assert.equal(layout.wrench.body, sim.socket.body);
    assert.deepEqual(layout.sensors.channels.map((c) => c.index), channels.map((_, i) => i));
    assert.deepEqual(layout.sensors.channels.map((c) => [c.term, c.name, c.unit]),
      channels.map(({ term, c }) => [term.name, term.channels[c], term.units[c]]));
    // Without the channel names (an older demo.json): names from the reads.
    const bare = JSON.parse(JSON.stringify(sim));
    for (const t of bare.osl.observation.terms) { delete t.channels; delete t.units; }
    const fallback = frameLayout({ visual, sim: bare }).sensors.channels;
    assert.equal(fallback.length, N);
    assert.ok(fallback.every((c) => c.name && c.unit === ''), 'fallback channel names');
    console.log(`# frame layout: ${layout.bytes} bytes (${layout.floats} floats): geoms ${layout.geoms.offset}+${layout.geoms.length}, `
      + `segments ${layout.segments.offset}+${layout.segments.length}, root ${layout.root.offset}+4, wrench ${layout.wrench.offset}+6, `
      + `sensors ${layout.sensors.offset}+${N} [${layout.sensors.channels.map((c) => `${c.name} ${c.unit}`).join(', ')}]`);
  });

  test('extras off without the sim config; the other sections are unchanged either way', () => {
    const plain = new FrameExtractor(mujoco, model, visual);
    const ex = new FrameExtractor(mujoco, model, visual, sim);
    const G = visual.geom_ids.length;
    assert.equal(plain.layout.wrench, null);
    assert.equal(plain.layout.sensors, null);
    assert.equal(plain.byteLength, 4 * (12 * G + 8 * plain.maxSegments + 4), 'the layout before the extras');
    // A socket section without a fresh cfrc_int, or no sim.socket at all: no wrench.
    assert.equal(frameLayout({ visual, sim: { ...sim, socket: { ...sim.socket, cfrc_int_fresh: false } } }).wrench, null);
    assert.equal(frameLayout({ visual, sim: { ...sim, socket: undefined } }).wrench, null);
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    const a = plain.extract(data).frame, b = ex.extract(data).frame;
    assert.ok(!('wrench' in a) && !('sensors' in a), 'plain frames have no extras');
    assert.ok(b.wrench instanceof Float32Array && b.wrench.length === 6);
    assert.ok(b.sensors instanceof Float32Array && b.sensors.length === ex.layout.sensors.length);
    for (const key of ['geoms', 'segments', 'root']) {
      assert.deepEqual(Array.from(b[key]), Array.from(a[key]), `${key} unchanged by the extras`);
      assert.equal(b[key].buffer, b.buffer);
    }
    assert.equal(b.segmentCount, a.segmentCount);
    assert.equal(b.wrench.buffer, b.buffer);
    assert.equal(b.sensors.buffer, b.buffer);
    assert.equal(b.wrench.byteOffset, ex.layout.wrench.byteOffset);
    assert.equal(b.sensors.byteOffset, ex.layout.sensors.byteOffset);
  });

  test('recycle: pooled buffers are reused whole; foreign, detached and repeated buffers are not', () => {
    const ex = new FrameExtractor(mujoco, model, visual, sim);
    const plain = new FrameExtractor(mujoco, model, visual);
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    const first = ex.extract(data);
    ex.recycle(first.frame);
    const again = ex.extract(data);
    assert.equal(again, first, 'a recycled buffer comes back with its frame object');
    // Twice the same buffer: pooled once.
    ex.recycle(again.frame.buffer);
    ex.recycle(again.frame);
    const x = ex.extract(data), y = ex.extract(data);
    assert.equal(x, first);
    assert.notEqual(y.frame.buffer, x.frame.buffer, 'a buffer recycled twice is pooled once');
    // A buffer of another layout (an extractor without extras): ignored.
    const foreign = plain.extract(data).frame.buffer;
    ex.recycle(foreign);
    assert.notEqual(ex.extract(data).frame.buffer, foreign);
    // A detached buffer (it was transferred): ignored.
    const sent = ex.extract(data).frame.buffer;
    structuredClone(sent, { transfer: [sent] });
    assert.equal(sent.byteLength, 0);
    ex.recycle(sent);
    assert.notEqual(ex.extract(data).frame.buffer, sent);
    // A buffer that crossed threads arrives as a new object: it gets a new
    // frame object and views (in the worker, every recycled buffer does).
    const back = structuredClone(x.frame.buffer, { transfer: [x.frame.buffer] });
    ex.recycle(back);
    const out = ex.extract(data);
    assert.equal(out.frame.buffer, back);
    assert.equal(out.frame.sensors.buffer, back);
    assert.notEqual(out.frame, x.frame, 'a new ArrayBuffer object gets its own frame object');
  });

  test('extract() reads MjData through cached views only, also after the wasm heap grows', () => {
    const ex = new FrameExtractor(mujoco, model, visual, sim);
    let gets = 0;
    const counted = new Proxy(data, { get(target, key) { gets += 1; return Reflect.get(target, key); } });
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    let out = ex.extract(counted);
    const bound = gets;
    assert.ok(bound > 0, 'the first extract binds the views');
    gets = 0;
    for (let i = 0; i < 200; i++) {
      ex.recycle(out.frame);
      out = ex.extract(counted);
    }
    assert.equal(gets, 0, `${gets} MjData property reads in 200 extracts`);
    // Allocate in the wasm heap until it grows (its old views detach): the
    // extractor re-binds once and stays right.
    const before = { wrench: Array.from(out.frame.wrench), sensors: Array.from(out.frame.sensors) };
    const probe = data.qpos;
    const extra = [];
    for (let i = 0; i < 64 && probe.length !== 0; i++) extra.push(new mujoco.MjData(model));
    const grew = probe.length === 0;
    ex.recycle(out.frame);
    const grown = ex.extract(counted);
    if (grew) assert.ok(gets > 0 && gets <= bound, 'views re-read after the heap grew');
    else assert.equal(gets, 0);
    assert.deepEqual(Array.from(grown.frame.wrench), before.wrench);
    assert.deepEqual(Array.from(grown.frame.sensors), before.sensors);
    for (const d of extra) d.delete();
    console.log(`# MjData reads: ${bound} on the first extract, 0 in the next 200, `
      + `${grew ? `${gets} after the heap grew (${extra.length} MjData later)` : 'the heap did not grow'}`);
  });

  test('cfrc_int is refreshed by every mj_step (the socket wrench needs no extra MuJoCo call)', () => {
    const ex = new FrameExtractor(mujoco, model, visual, sim);
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    for (let i = 0; i < 3; i++) {
      data.cfrc_int.fill(NaN);
      mujoco.mj_step(model, data);
      const { frame } = ex.extract(data);
      assert.ok(frame.wrench.every(Number.isFinite), `wrench after mj_step ${i}: ${Array.from(frame.wrench)}`);
      ex.recycle(frame);
    }
    data.cfrc_int.fill(NaN);
    mujoco.mj_forward(model, data);
    const { frame } = ex.extract(data);
    assert.ok(frame.wrench.every(Number.isFinite), 'wrench after mj_forward');
    // Standing at the keyframe, the prosthesis hangs from (or stands on) the
    // limb: the vertical load is of the order of its weight or more.
    console.log(`# keyframe 0 after mj_forward: F [${Array.from(frame.wrench.subarray(0, 3), (x) => x.toFixed(1))}] N, `
      + `T [${Array.from(frame.wrench.subarray(3), (x) => x.toFixed(2))}] N m; prosthesis weight ${(sim.socket.subtree_mass * 9.81).toFixed(1)} N`);
  });

  test('frame.wrench: sign and lever arm without the cfrc_int formula (the prosthesis hanging still)', () => {
    // Lifted clear of everything and held still (qvel = qacc = 0, mj_inverse),
    // the limb carries the prosthesis's weight at its centre of mass: the load
    // on the limb is F = M g (down) and T = (subtree_com[s] - xpos[s]) x F
    // about the socket origin, M = body_subtreemass[s]. Nothing here uses the
    // cfrc_int formula, so a sign or lever-arm error that frame.js and the
    // trace check share shows up. The body keeps keyframe 0's joint angles (no
    // contact) and is turned rigidly, so the lever arm points many ways.
    const ex = new FrameExtractor(mujoco, model, visual, sim);
    const s = sim.socket.body, q = sim.root_qpos;
    const M = model.body_subtreemass[s];
    const F = Array.from(model.opt.gravity, (g) => M * g);
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let rng = 4242;
    const random = () => { rng = (rng * 1103515245 + 12345) % 2147483648; return rng / 2147483648 - 0.5; };
    const poses = 8;
    let worstF = 0, worstT = 0, maxT = 0;
    const unshifted = [Infinity, 0]; // range over the poses
    for (let pose = 0; pose < poses; pose++) {
      mujoco.mj_resetDataKeyframe(model, data, 0);
      data.qpos[q + 2] += 2;
      if (pose > 0) {
        const quat = [random(), random(), random(), random()], n = Math.hypot(...quat);
        data.qpos.set(quat.map((x) => x / n), q + 3);
      }
      data.qvel.fill(0);
      data.qacc.fill(0);
      mujoco.mj_inverse(model, data);
      // mj_inverse refreshes cfrc_int already (the model's acceleration-stage
      // sensors); called explicitly so the check does not depend on that.
      mujoco.mj_rnePostConstraint(model, data);
      assert.equal(data.ncon, 0, `pose ${pose}: no contact`);
      const { frame } = ex.extract(data);
      const r = [0, 1, 2].map((i) => data.subtree_com[3 * s + i] - data.xpos[3 * s + i]);
      const T = cross(r, F), scale = Math.hypot(...F);
      worstF = Math.max(worstF, maxDiff(frame.wrench.subarray(0, 3), F) / scale);
      worstT = Math.max(worstT, maxDiff(frame.wrench.subarray(3, 6), T) / (scale * Math.hypot(...r)));
      maxT = Math.max(maxT, Math.hypot(...T));
      // For scale: how far off a torque left about cfrc_int's own reference
      // point (the tree root's centre of mass, no lever-arm transfer) would be.
      const root = 3 * sim.socket.root;
      const off = maxDiff(cross([0, 1, 2].map((i) => data.subtree_com[3 * s + i] - data.subtree_com[root + i]), F), T);
      unshifted[0] = Math.min(unshifted[0], off);
      unshifted[1] = Math.max(unshifted[1], off);
      ex.recycle(frame);
    }
    console.log(`# hanging still at ${poses} orientations: F = M g = [${F.map((x) => x.toFixed(3))}] N to ${fmt(worstF)} of |F|, `
      + `T = (subtree_com[s] - xpos[s]) x F (up to ${maxT.toFixed(2)} N m) to ${fmt(worstT)} of |F||r|; `
      + `T left about the tree root's COM would be off by ${unshifted[0].toFixed(2)}-${unshifted[1].toFixed(2)} N m`);
    assert.ok(worstF < 1e-6, `hanging still, F differs from the prosthesis's weight by ${worstF} of |F|`);
    assert.ok(worstT < 1e-6, `hanging still, T differs from the weight's moment about the socket origin by ${worstT} of |F||r|`);
  });

  test('extract() timing with and without the extras (well under 2 ms)', () => {
    const ex = new FrameExtractor(mujoco, model, visual, sim);
    const plain = new FrameExtractor(mujoco, model, visual);
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    const time = (extractor) => {
      for (let i = 0; i < 300; i++) extractor.recycle(extractor.extract(data).frame);
      const N = 5000, t0 = performance.now();
      for (let i = 0; i < N; i++) extractor.recycle(extractor.extract(data).frame);
      return (performance.now() - t0) / N * 1000;
    };
    const runs = [];
    for (let k = 0; k < 3; k++) runs.push([time(plain), time(ex)]);
    const plainUs = Math.min(...runs.map((r) => r[0])), extrasUs = Math.min(...runs.map((r) => r[1]));
    console.log(`# extract(): ${plainUs.toFixed(2)} us without the extras (${plain.byteLength} bytes), `
      + `${extrasUs.toFixed(2)} us with them (${ex.byteLength} bytes), best of 3 x 5000 recycled`);
    assert.ok(extrasUs < 2000, `extract() takes ${extrasUs} us`);
  });
});

// ---------------------------------------------------------------- trace
const traceMissing = CONTRACT_SKIP || (!existsSync(fileURLToPath(TRACE))
  ? `no trace at ${fileURLToPath(TRACE)}: python -m duet_export.trace (DUET_TRACE)` : false);
const trace = traceMissing ? null : await loadContainer(TRACE);
function traceSkip() {
  if (traceMissing) return traceMissing;
  const meta = trace.meta;
  if (meta.manifest_sha256 !== demo.source.manifest_sha256) {
    return `stale trace: recorded from manifest ${String(meta.manifest_sha256).slice(0, 12)}, the assets from `
      + `${demo.source.manifest_sha256.slice(0, 12)}: regenerate with python -m duet_export.trace`;
  }
  for (const key of ['osl_frame', 'socket_cfrc', 'root_subtree_com', 'socket_xpos']) {
    if (!trace.arrays[key]) return `the trace has no ${key} rows (format ${meta.format}): regenerate it`;
  }
  return false;
}
const TRACE_SKIP = traceSkip();

describe('frame extras vs osl_deploy trace', { skip: TRACE_SKIP }, () => {
  let mujoco, model, data;

  before(async () => {
    mujoco = await loadMujoco();
    ({ model, data } = modelFromBytes(mujoco, await fetchBytes(new URL(demo.physics.file, ASSETS)), 'physics.mjb'));
  });

  after(() => {
    data?.delete();
    model?.delete();
  });

  test('frame.wrench and frame.sensors at every boundary; the load on the limb points up in stance', () => {
    const T = trace.arrays, meta = trace.meta, sim = demo.sim;
    const { nq, nv, na, nu } = sim.model;
    const ex = new FrameExtractor(mujoco, model, demo.visual, sim);
    const N = ex.layout.sensors.length;
    // The trace's frame is the same terms in the same order.
    const terms = meta.extras.osl_frame.terms;
    assert.deepEqual(terms.map((t) => [t.name, t.frame_dim]),
      sim.osl.observation.terms.map((t) => [t.name, t.frame_dim]), 'trace osl_frame terms vs demo.json');
    assert.equal(meta.extras.osl_frame.width, N);
    assert.equal(meta.extras.socket_cfrc.body_id, sim.socket.body);
    assert.equal(meta.extras.root_subtree_com.body_id, sim.socket.root);
    const heel = ex.layout.sensors.channels.find((c) => c.name === 'osl_heel_contact');
    const toe = ex.layout.sensors.channels.find((c) => c.name === 'osl_toe_contact');
    assert.ok(heel && toe, 'heel and toe contact channels');

    let fErr = 0, tErr = 0, sAbs = 0, sRel = 0, sBad = 0, checked = 0, worstF = null, worstS = null, maxF = 0;
    const stance = [];
    let upViolations = 0, onsets = 0, onsetsDown = 0, onsetMin = Infinity;
    let prevContact = NaN; // heel + toe at the boundary before (NaN: none, or a failed step's)
    const rows = T.time.length;
    for (let b = 0; b < rows; b++) {
      const py = row(T.osl_frame, N, b);
      if (!Number.isFinite(py[0])) { prevContact = NaN; continue; } // a failed step's boundary
      mujoco.mj_resetData(model, data);
      data.qpos.set(row(T.qpos, nq, b));
      data.qvel.set(row(T.qvel, nv, b));
      data.act.set(row(T.act, na, b));
      data.ctrl.set(row(T.ctrl, nu, b));
      data.qacc_warmstart.set(row(T.qacc_warmstart, nv, b));
      data.time = T.time[b];
      mujoco.mj_forward(model, data);
      const out = ex.extract(data), { wrench, sensors } = out.frame;
      // The native formula on the recorded rows (float64).
      const c = row(T.socket_cfrc, 6, b), com = row(T.root_subtree_com, 3, b), xp = row(T.socket_xpos, 3, b);
      const d = [com[0] - xp[0], com[1] - xp[1], com[2] - xp[2]];
      const F = [-c[3], -c[4], -c[5]];
      const Tq = [
        -(c[0] + d[1] * c[5] - d[2] * c[4]),
        -(c[1] + d[2] * c[3] - d[0] * c[5]),
        -(c[2] + d[0] * c[4] - d[1] * c[3]),
      ];
      const fMag = Math.hypot(...F), tMag = Math.hypot(...Tq);
      maxF = Math.max(maxF, fMag);
      const fe = maxDiff(Array.from(wrench.subarray(0, 3)), F) / Math.max(fMag, 1);
      const te = maxDiff(Array.from(wrench.subarray(3, 6)), Tq) / Math.max(tMag, 1);
      if (fe > fErr) { fErr = fe; worstF = { b, F, js: Array.from(wrench.subarray(0, 3)) }; }
      tErr = Math.max(tErr, te);
      for (let i = 0; i < N; i++) {
        const diff = Math.abs(sensors[i] - py[i]);
        const rel = diff / Math.max(Math.abs(py[i]), 1e-12);
        if (diff > sAbs) { sAbs = diff; worstS = { b, i, js: sensors[i], py: py[i] }; }
        if (Math.abs(py[i]) > 1) sRel = Math.max(sRel, rel);
        if (diff > 1e-3 && rel > 1e-5) sBad += 1;
      }
      const contact = py[heel.index] + py[toe.index];
      if (contact > STANCE_N && prevContact > STANCE_N) {
        stance.push([wrench[2], contact]);
        if (!(wrench[2] > 0)) upViolations += 1;
      } else if (contact > STANCE_N) { // a contact onset's first loaded boundary
        onsets += 1;
        onsetMin = Math.min(onsetMin, wrench[2]);
        if (!(wrench[2] > 0)) onsetsDown += 1;
      }
      prevContact = contact;
      ex.recycle(out.frame);
      checked += 1;
    }
    // Pearson correlation of the vertical load with the total contact force in stance.
    const n = stance.length;
    const mean = (k) => stance.reduce((s, p) => s + p[k], 0) / n;
    const mz = mean(0), mc = mean(1);
    let sxy = 0, sxx = 0, syy = 0;
    for (const [z, cf] of stance) { sxy += (z - mz) * (cf - mc); sxx += (z - mz) ** 2; syy += (cf - mc) ** 2; }
    const corr = sxy / Math.sqrt(sxx * syy);
    console.log(`# trace ${meta.bundle}: ${checked} boundaries; wrench vs native formula: force max rel err ${fmt(fErr)} `
      + `(|F| up to ${maxF.toFixed(0)} N), torque ${fmt(tErr)}; sensors vs osl_frame: max abs ${fmt(sAbs)}, `
      + `max rel (|value| > 1) ${fmt(sRel)}, ${sBad} outside abs 1e-3 / rel 1e-5`);
    console.log(`# stance (heel + toe > ${STANCE_N} N, also at the boundary before): ${n} boundaries, F_z > 0 at ${n - upViolations}, `
      + `mean F_z ${mz.toFixed(0)} N, mean contact ${mc.toFixed(0)} N, corr(F_z, contact) ${corr.toFixed(4)}; `
      + `${onsets} contact onsets left out (F_z from ${onsets ? onsetMin.toFixed(0) : '-'} N, down at ${onsetsDown})`);
    assert.ok(checked > 1000, `only ${checked} boundaries checked`);
    assert.ok(fErr < 1e-5, `force differs by ${fErr} of |F| at ${JSON.stringify(worstF)}`);
    assert.ok(tErr < 1e-5, `torque differs by ${tErr} of |T|`);
    assert.equal(sBad, 0, `sensor frame differs, worst ${JSON.stringify(worstS)}`);
    assert.ok(n > 100, `only ${n} stance boundaries`);
    assert.ok(upViolations <= Math.floor(STANCE_DOWN_FRACTION * n),
      `the load on the limb points down at ${upViolations} of ${n} stance boundaries (at most ${Math.floor(STANCE_DOWN_FRACTION * n)} may)`);
    assert.ok(corr > 0.9, `F_z follows the contact force with correlation ${corr}`);
  });
});
