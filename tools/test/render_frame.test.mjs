// render/frame.js against MuJoCo itself, plus the visual asset's integrity.
//
//   cd tools && node --test test/render_frame.test.mjs
//
// Needs demo/assets/stage2/{demo.json,visual.bin.gz} (or DUET_ASSETS; python -m
// duet_export --only visual) and, for the pose checks, traces/render_poses.json
// (python -m duet_export.visual poses traces/render_poses.json), which holds
// MuJoCo's own mjv_addGeoms tendon capsules and geom frames for two poses.
// The model is DUET_RENDER_MODEL, else the asset folder's physics.mjb.gz, else
// the original bundle model (paths.mjs BUNDLE; about 20 s to load in wasm).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { loadMujoco, modelFromBytes } from '../../demo/js/lib/mujoco.js';
import { fetchBytes, loadContainer } from '../../demo/js/lib/container.js';
import { FrameExtractor } from '../../demo/js/render/frame.js';
import { ASSETS, BUNDLE, TRACES } from './paths.mjs';

const FIXTURE = new URL('render_poses.json', TRACES);
const BUNDLE_MODEL = new URL('model.mjb', BUNDLE);

function modelUrl() {
  if (process.env.DUET_RENDER_MODEL) return pathToFileURL(process.env.DUET_RENDER_MODEL);
  const slim = new URL('physics.mjb.gz', ASSETS);
  return fs.existsSync(fileURLToPath(slim)) ? slim : BUNDLE_MODEL;
}

const demo = JSON.parse(fs.readFileSync(fileURLToPath(new URL('demo.json', ASSETS)), 'utf8'));
const visual = demo.visual;

test('visual.bin.gz matches demo.json and is well formed', async () => {
  assert.ok(visual, 'demo.json has no visual section');
  const { meta, arrays } = await loadContainer(new URL(visual.file, ASSETS));
  assert.equal(meta.format, visual.format);
  assert.deepEqual(Array.from(arrays.geom_id), visual.geom_ids);
  const nMesh = arrays.mesh_id.length;
  const nVert = arrays.positions.length / 3;
  for (let i = 0; i < nMesh; i++) {
    const vs = arrays.mesh_vertex_start[i], vc = arrays.mesh_vertex_count[i];
    const is = arrays.mesh_index_start[i], ic = arrays.mesh_index_count[i];
    assert.ok(vs + vc <= nVert && ic % 3 === 0 && vc <= 65536);
    for (let k = is; k < is + ic; k++) assert.ok(arrays.indices[k] < vc, `mesh ${i} index out of range`);
  }
  for (const m of arrays.geom_mesh) assert.ok(m < nMesh);
  for (const m of arrays.geom_material) assert.ok(m >= 0 && m < visual.materials.length);
  // Unit normals within int8 rounding.
  for (let k = 0; k < arrays.normals.length; k += 3) {
    const n = Math.hypot(arrays.normals[k], arrays.normals[k + 1], arrays.normals[k + 2]) / 127;
    assert.ok(Math.abs(n - 1) < 0.02, `normal ${k / 3} has length ${n}`);
  }
  const tris = arrays.indices.length / 3;
  assert.equal(tris, visual.stats.triangles);
  console.log(`# visual.bin.gz: ${visual.bytes} bytes, ${nMesh} meshes, ${tris} triangles, ${nVert} vertices`);
});

test('FrameExtractor reproduces MuJoCo geom frames and tendon capsules', async (t) => {
  const url = modelUrl();
  const mujoco = await loadMujoco();
  const t0 = performance.now();
  const { model, data } = modelFromBytes(mujoco, await fetchBytes(url));
  console.log(`# model ${url.pathname.split('/').slice(-2).join('/')} loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  try {
    const extractor = new FrameExtractor(mujoco, model, visual);
    assert.equal(extractor.maxSegments, visual.muscles.max_segments);

    if (!fs.existsSync(fileURLToPath(FIXTURE))) {
      t.diagnostic('no traces/render_poses.json: pose checks skipped');
    } else {
      const fixture = JSON.parse(fs.readFileSync(fileURLToPath(FIXTURE), 'utf8'));
      for (const [name, pose] of Object.entries(fixture.poses)) {
        data.qpos.set(pose.qpos);
        data.qvel.fill(0);
        data.act.set(pose.act);
        mujoco.mj_forward(model, data);
        const { frame, transfer } = extractor.extract(data);
        assert.deepEqual(transfer, [frame.buffer]);
        assert.equal(frame.buffer.byteLength, extractor.byteLength);

        // Geom frames.
        let geomErr = 0;
        pose.geom_xpos.forEach((p, i) => {
          for (let k = 0; k < 3; k++) geomErr = Math.max(geomErr, Math.abs(frame.geoms[12 * i + k] - p[k]));
          for (let k = 0; k < 9; k++) geomErr = Math.max(geomErr, Math.abs(frame.geoms[12 * i + 3 + k] - pose.geom_xmat[i][k]));
        });
        assert.ok(geomErr < 1e-5, `${name}: geom frames differ by ${geomErr}`);

        // Tendon capsules, in mjv_addGeoms order: [tendon, pos(3), axis(3), radius, half].
        const caps = pose.mjv_capsules;
        assert.equal(frame.segmentCount, caps.length, `${name}: segment count`);
        let posErr = 0, radErr = 0, axisErr = 0;
        caps.forEach((c, i) => {
          const s = frame.segments.subarray(8 * i, 8 * i + 8);
          const mid = [0, 1, 2].map((k) => (s[k] + s[k + 3]) / 2);
          const d = [0, 1, 2].map((k) => s[k + 3] - s[k]);
          const len = Math.hypot(...d);
          posErr = Math.max(posErr, ...mid.map((x, k) => Math.abs(x - c[1 + k])), Math.abs(len / 2 - c[8]));
          radErr = Math.max(radErr, Math.abs(s[6] - c[7]));
          if (len > 1e-6) axisErr = Math.max(axisErr, 1 - Math.abs((d[0] * c[4] + d[1] * c[5] + d[2] * c[6]) / len));
        });
        assert.ok(posErr < 1e-5 && radErr < 1e-7 && axisErr < 1e-5,
          `${name}: capsules differ (pos ${posErr}, radius ${radErr}, axis ${axisErr})`);

        // Activations: the act of each segment's muscle.
        const tendonOfSegment = caps.map((c) => c[0]);
        const actOf = new Map();
        for (let a = 0; a < model.nu; a++) {
          if (model.actuator_trntype[a] === 3) actOf.set(model.actuator_trnid[2 * a], model.actuator_actadr[a]);
        }
        tendonOfSegment.forEach((tn, i) => {
          const want = actOf.has(tn) ? Math.min(1, Math.max(0, pose.act[actOf.get(tn)])) : 0;
          assert.ok(Math.abs(frame.segments[8 * i + 7] - want) < 1e-6, `${name}: activation of segment ${i}`);
        });

        const rootBody = visual.root.body;
        for (let k = 0; k < 3; k++) assert.ok(Math.abs(frame.root[k] - data.xpos[3 * rootBody + k]) < 1e-5);
        console.log(`# ${name}: ${frame.segmentCount} segments, geom err ${geomErr.toExponential(1)}, ` +
          `capsule err ${posErr.toExponential(1)} m, heading ${(frame.root[3] * 180 / Math.PI).toFixed(1)} deg`);
        if (name === 'stride') {
          // The forward clip walks along +x: its heading is near 0.
          assert.ok(Math.abs(frame.root[3]) < 0.35, `stride heading ${frame.root[3]}`);
        }
        extractor.recycle(frame);
      }
    }

    // Timing: one extract per control step in the worker.
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    const N = 3000;
    for (let i = 0; i < 200; i++) extractor.recycle(extractor.extract(data).frame);
    let t1 = performance.now();
    for (let i = 0; i < N; i++) extractor.recycle(extractor.extract(data).frame);
    const pooled = (performance.now() - t1) / N * 1000;
    t1 = performance.now();
    for (let i = 0; i < N; i++) extractor.extract(data);
    const fresh = (performance.now() - t1) / N * 1000;
    console.log(`# extract(): ${pooled.toFixed(1)} us/call recycled, ${fresh.toFixed(1)} us/call with a new buffer ` +
      `(${extractor.byteLength} bytes, ${extractor.geomCount} geoms, <= ${extractor.maxSegments} segments)`);
    assert.ok(pooled < 2000, 'extract() should take well under 2 ms');
  } finally {
    data.delete();
    model.delete();
  }
});
