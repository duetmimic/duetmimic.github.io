// Static contract of an exported asset folder (node --test; no browser, no
// policy runs): its files must come from ONE bundle and agree with each other.
//
//   cd tools && node --test test/contract.test.mjs     (DUET_ASSETS: see paths.mjs)
//
// Why: two bundles' models can share every id, count and sensordata address
// (R2 and B differ only in three touch-zone boxes), so a folder that mixes
// them loads and runs without an error. With R2's physics.mjb.gz under B's
// policies, B's heel contact input reads 0 N. Checks:
//   - every demo.json section (physics, visual, policies, sim) records the same
//     bundle manifest sha256, source.manifest_sha256 (python -m duet_export
//     refuses a mixed --only export unless --allow-mixed);
//   - the observation layouts are self-consistent: the OSL sensor terms and the
//     previous-command history tile the OSL input, whose size is the OSL
//     network's (demo.json and the osl.bin.gz header); the human groups tile
//     the human input; the action blocks tile each action;
//   - every touch zone sim.touch_sites records (the manifest's, which osl_deploy
//     checked against the bundle's model.mjb) is the zone in physics.mjb.gz:
//     the same site, sensor and body, pose and box size to 1e-9; every touch
//     sensor a policy reads (sim.touch_adr, OSL reads of touch sensors) is one
//     of them;
//   - the bundle the checks that need it load (paths.mjs BUNDLE: sim.test.mjs,
//     render_frame.test.mjs) is the one the assets came from.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { fetchBytes, loadContainer } from '../../demo/js/lib/container.js';
import { loadMujoco, modelFromBytes } from '../../demo/js/lib/mujoco.js';
import { ASSETS, BUNDLE } from './paths.mjs';

const SECTIONS = { physics: ['physics'], visual: ['visual'], policies: ['policies'], sim: ['sim', 'reference', 'ui'] };
const TOLERANCE = 1e-9; // osl_deploy touch_sites.TOLERANCE: declared values are the compiled model's

const DEMO = new URL('demo.json', ASSETS);
const missing = ['demo.json', 'physics.mjb.gz', 'osl.bin.gz'].filter((f) => !existsSync(fileURLToPath(new URL(f, ASSETS))));
const SKIP = missing.length ? `missing ${fileURLToPath(ASSETS)}{${missing.join(',')}}: run python -m duet_export` : false;
const demo = SKIP ? null : JSON.parse(readFileSync(fileURLToPath(DEMO), 'utf8'));
const short = (sha) => (typeof sha === 'string' ? sha.slice(0, 12) : String(sha));

// [start, stop, name] spans must cover [0, size) exactly, each once.
function assertTiles(spans, size, what) {
  const sorted = [...spans].sort((a, b) => a[0] - b[0]);
  let end = 0;
  for (const [start, stop, name] of sorted) {
    assert.equal(start, end, `${what}: ${name} starts at ${start}, expected ${end}`);
    assert.ok(stop > start, `${what}: ${name} is empty`);
    end = stop;
  }
  assert.equal(end, size, `${what}: the spans cover ${end} of ${size}`);
}

describe('asset folder contract', { skip: SKIP }, () => {
  test('every section was exported from the same bundle as source names', () => {
    const sha = demo.source && demo.source.manifest_sha256;
    assert.match(String(sha), /^[0-9a-f]{64}$/, 'demo.json source.manifest_sha256');
    assert.ok(demo.sections, 'demo.json has no "sections" (an export before 2026-11-18): re-run python -m duet_export');
    for (const [name, keys] of Object.entries(SECTIONS)) {
      for (const key of keys) assert.ok(demo[key], `demo.json lacks "${key}" (section ${name})`);
      assert.equal(demo.sections[name], sha, `section ${name} comes from manifest ${short(demo.sections[name])}, `
        + `source ${demo.source.bundle} is ${short(sha)}: the folder mixes bundles (re-export every section)`);
    }
    assert.deepEqual(Object.keys(demo.sections).sort(), Object.keys(SECTIONS).sort());
  });

  test('observation and action layouts tile their vectors; sizes match the networks', async () => {
    const { sim, policies } = demo;
    // OSL: sensor terms (history x frame_dim each) + previous-command history.
    const obs = sim.osl.observation;
    const pc = obs.previous_command;
    assertTiles([
      ...obs.terms.map((t) => [t.offset, t.offset + t.history * t.frame_dim, t.name]),
      [pc.offset, pc.offset + pc.history * pc.width, 'previous_command'],
    ], obs.size, 'OSL observation');
    for (const t of obs.terms) {
      assertTiles(t.reads.map((r) => [r.frame, r.frame + (r.components ? r.components.length : r.size), r.name]),
        t.frame_dim, `OSL term ${t.name}`);
    }
    assert.equal(pc.width, sim.osl.action.size, 'the previous command is the OSL action');
    assert.equal(policies.osl.input_size, obs.size, 'osl network input vs the OSL observation');
    assert.equal(policies.osl.output_size, sim.osl.action.size);
    const header = (await loadContainer(new URL(policies.osl.file, ASSETS))).meta;
    assert.equal(header.input_size, obs.size, `${policies.osl.file} takes ${header.input_size} inputs`);
    assert.equal(header.output_size, sim.osl.action.size);
    // Human: observation groups, network sizes.
    const human = sim.human.observation;
    assertTiles(Object.entries(human.groups).map(([name, [start, stop]]) => [start, stop, name]), human.size, 'human observation');
    assert.equal(policies.human.input_size, human.size, 'human network input vs the human observation');
    assert.equal(policies.human.output_size, sim.human.action.size);
    // Actions: blocks tile each controller's output.
    for (const name of ['human', 'osl']) {
      const action = sim[name].action;
      assertTiles(action.blocks.map((b) => [b.start, b.start + b.size, b.name]), action.size, `${name} action`);
    }
  });

  describe('touch zones of physics.mjb.gz', () => {
    let mujoco, model, data;

    before(async () => {
      mujoco = await loadMujoco();
      ({ model, data } = modelFromBytes(mujoco, await fetchBytes(new URL(demo.physics.file, ASSETS)), 'physics.mjb'));
    });

    after(() => {
      data?.delete();
      model?.delete();
    });

    test('every recorded zone is the zone in the physics model; every touch read is a zone', () => {
      const zones = demo.sim.touch_sites;
      assert.ok(zones && zones.sites && zones.sites.length, 'demo.json lacks sim.touch_sites: re-run python -m duet_export');
      const OBJ = mujoco.mjtObj;
      const TOUCH = mujoco.mjtSensor.mjSENS_TOUCH.value, BOX = mujoco.mjtGeom.mjGEOM_BOX.value;
      const near = (actual, expected, what) => {
        assert.equal(actual.length, expected.length, what);
        for (let i = 0; i < expected.length; i++) {
          assert.ok(Math.abs(actual[i] - expected[i]) <= TOLERANCE,
            `${what} is [${Array.from(actual)}] in ${demo.physics.file}, the export recorded [${expected}] `
            + `(touch_sites ${zones.version}, ${demo.source.bundle}): the physics model comes from another bundle`);
        }
      };
      for (const z of zones.sites) {
        const site = mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, z.site);
        const sensor = mujoco.mj_name2id(model, OBJ.mjOBJ_SENSOR.value, z.sensor);
        assert.equal(site, z.site_id, `site ${z.site}`);
        assert.equal(sensor, z.sensor_id, `sensor ${z.sensor}`);
        assert.equal(mujoco.mj_name2id(model, OBJ.mjOBJ_BODY.value, z.body), z.body_id, `body ${z.body}`);
        assert.equal(model.site_bodyid[site], z.body_id, `${z.site} is on ${z.body}`);
        assert.equal(model.site_type[site], BOX, `${z.site} is a box`);
        assert.equal(model.sensor_type[sensor], TOUCH, `${z.sensor} is a touch sensor`);
        assert.equal(model.sensor_objtype[sensor], OBJ.mjOBJ_SITE.value, `${z.sensor} reads a site`);
        assert.equal(model.sensor_objid[sensor], site, `${z.sensor} reads ${z.site}`);
        assert.equal(model.sensor_adr[sensor], z.adr, `${z.sensor} sensordata address`);
        near(model.site_pos.subarray(3 * site, 3 * site + 3), z.pos, `${z.site} site_pos`);
        near(model.site_quat.subarray(4 * site, 4 * site + 4), z.quat, `${z.site} site_quat`);
        near(model.site_size.subarray(3 * site, 3 * site + 3), z.size, `${z.site} site_size`);
      }
      // The touch sensors the policies read: the human touch term and any OSL
      // read of a touch sensor (B: heel robot/r_foot, toe robot/r_toes).
      const zoneAdr = new Set(zones.sites.map((z) => z.adr));
      const sensorAt = (adr) => {
        for (let s = 0; s < model.nsensor; s++) if (model.sensor_adr[s] === adr) return s;
        return -1;
      };
      for (const adr of demo.sim.touch_adr) assert.ok(zoneAdr.has(adr), `human touch read at sensordata ${adr} is not a recorded zone`);
      let oslTouch = 0;
      for (const term of demo.sim.osl.observation.terms) {
        for (const read of term.reads) {
          const s = sensorAt(read.adr);
          assert.ok(s >= 0, `OSL read ${read.name}: no sensor starts at sensordata ${read.adr}`);
          if (model.sensor_type[s] !== TOUCH) continue;
          oslTouch += 1;
          assert.ok(zoneAdr.has(read.adr), `OSL read ${read.name} (sensordata ${read.adr}) is not a recorded zone`);
        }
      }
      console.log(`# touch zones ${zones.version} (${zones.declared ? 'declared in the manifest' : 'legacy table'}): `
        + `${zones.sites.length} zones match ${demo.physics.file}; ${demo.sim.touch_adr.length} human and ${oslTouch} OSL touch reads`);
    });
  });

  test('the bundle the trace and model checks load is the one the assets came from', (t) => {
    const manifest = new URL('manifest.json', BUNDLE);
    if (!existsSync(fileURLToPath(manifest))) {
      t.skip(`no bundle at ${fileURLToPath(BUNDLE)} (DUET_BUNDLE)`);
      return;
    }
    const sha = createHash('sha256').update(readFileSync(fileURLToPath(manifest))).digest('hex');
    assert.equal(sha, demo.source.manifest_sha256, `${fileURLToPath(BUNDLE)} (manifest ${short(sha)}) is not the bundle `
      + `the assets came from, ${demo.source.bundle} (${short(demo.source.manifest_sha256)}): set DUET_BUNDLE, or update `
      + 'tools/test/paths.mjs together with duet_export/common.py DEFAULT_BUNDLE');
  });
});
