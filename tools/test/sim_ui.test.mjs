// SimCore's command side and small parts (node --test; no MuJoCo, no model):
// the WASD table (every reachable twist is an exact clip label inside the
// trained support), twistForKeys / levelForKey, the reference container, the
// actuator delay line and the previous-command history.

import assert from 'node:assert/strict';
import { before, describe, test } from 'node:test';

import { loadContainer, loadJSON } from '../../demo/js/lib/container.js';
import {
  ActuatorDelay, CommandSupport, GaitReference, PreviousCommand, levelForKey, twistForKeys, uiTwists,
} from '../../demo/js/sim/index.js';
import { ASSETS } from './paths.mjs';

let demo, reference, ref, support;

before(async () => {
  demo = await loadJSON(new URL('demo.json', ASSETS));
  reference = await loadContainer(new URL('reference.bin.gz', ASSETS));
  ref = new GaitReference(demo.reference, reference.arrays);
  support = new CommandSupport(demo.reference, ref);
});

describe('UI twist table', () => {
  test('every UI twist is exactly a reachable clip label, served unchanged', () => {
    const clips = demo.reference.clips;
    const { low, high } = demo.reference.command;
    const served = new Float64Array(3);
    const used = new Set();
    for (const { level, action, twist } of uiTwists(demo.ui)) {
      const exact = clips.findIndex((c) => c.twist.every((x, i) => x === twist[i]));
      assert.ok(exact >= 0, `level ${level} ${action}: ${twist} is no clip label`);
      used.add(exact);
      assert.equal(ref.select(twist), exact, `${action}: nearest clip is not the labelled one`);
      assert.equal(clips[exact].idle, action === 'idle');
      support.serve(twist, served);
      assert.deepEqual(Array.from(served), action === 'idle' ? [0, 0, 0] : twist, `${action}: served twist differs`);
      for (let i = 0; i < 3; i++) assert.ok(twist[i] >= low[i] && twist[i] <= high[i]);
      if (action === 'idle') continue;
      const [vx, , wz] = twist;
      if (action.startsWith('forward')) assert.ok(vx > 0);
      if (action.startsWith('backward')) assert.ok(vx < 0);
      if (action.startsWith('turn')) assert.equal(vx, 0);
      // wz > 0 yaws the body left. Backward arcs steer like a car in reverse:
      // S+A curves the path to the walker's left, so the body yaws right.
      const yaw = action.startsWith('backward') ? -wz : wz;
      if (action.endsWith('left')) assert.ok(yaw > 0);
      if (action.endsWith('right')) assert.ok(yaw < 0);
      if (action === 'forward' || action === 'backward') assert.equal(wz, 0);
    }
    // The container holds exactly the clips the UI reaches.
    assert.equal(used.size, clips.length);
  });

  test('twistForKeys maps held keys to actions', () => {
    const ui = demo.ui;
    const keys = (...codes) => new Set(codes);
    const action = (...codes) => twistForKeys(keys(...codes), ui).action;
    assert.equal(action(), 'idle');
    assert.deepEqual(twistForKeys(keys(), ui).twist, [0, 0, 0]);
    assert.equal(action('KeyW'), 'forward');
    assert.equal(action('KeyW', 'KeyA'), 'forward_left');
    assert.equal(action('KeyW', 'KeyD'), 'forward_right');
    assert.equal(action('KeyA'), 'turn_left');
    assert.equal(action('KeyD'), 'turn_right');
    assert.equal(action('KeyS'), 'backward');
    assert.equal(action('KeyS', 'KeyA'), 'backward_left');
    assert.equal(action('KeyS', 'KeyD'), 'backward_right');
    assert.equal(action('KeyW', 'KeyS'), 'idle');
    assert.equal(action('KeyA', 'KeyD'), 'idle');
    assert.equal(action('KeyW', 'KeyS', 'KeyA'), 'turn_left');
    assert.equal(action('ArrowUp', 'ArrowLeft'), 'forward_left');
    assert.equal(action('w'), 'forward');
    assert.equal(action('W', 'd'), 'forward_right');
    const normal = twistForKeys(keys('KeyW'), ui);
    assert.equal(normal.level, ui.default_level);
    assert.equal(normal.twist[0], 1.0);
    const fast = twistForKeys(keys('KeyW', 'ShiftLeft'), ui);
    assert.equal(fast.level, ui.fast_level);
    assert.ok(fast.twist[0] > normal.twist[0]);
    assert.ok(twistForKeys(keys('KeyW', 'Shift'), ui).fast);
    assert.equal(twistForKeys(keys('KeyW'), ui, 0).twist[0], ui.levels[0].actions.forward.twist[0]);
    assert.equal(twistForKeys(keys('KeyW'), ui, 99).level, ui.levels.length - 1);
    // The returned twist is a copy.
    normal.twist[0] = 7;
    assert.equal(ui.levels[ui.default_level].actions.forward.twist[0], 1.0);
  });

  test('levelForKey: digits pick, Q/E step and clamp', () => {
    const ui = demo.ui;
    assert.equal(levelForKey('Digit1', ui, 2), 0);
    assert.equal(levelForKey('Digit5', ui, 2), 4);
    assert.equal(levelForKey('KeyQ', ui, 2), 1);
    assert.equal(levelForKey('KeyQ', ui, 0), 0);
    assert.equal(levelForKey('KeyE', ui, 2), 3);
    assert.equal(levelForKey('KeyE', ui, 4), 4);
    assert.equal(levelForKey('KeyW', ui, 3), 3);
  });
});

describe('reference container', () => {
  test('matches the demo.json clip table; the reset clip is idle', () => {
    const { meta, arrays } = reference;
    const clips = demo.reference.clips;
    assert.deepEqual(meta.clips, clips.map((c) => c.name));
    assert.deepEqual(meta.start, clips.map((c) => c.start));
    const frames = clips.at(-1).start + clips.at(-1).num_frames;
    assert.equal(arrays.root_pos.length, 3 * frames);
    assert.equal(arrays.rpos.length, 3 * demo.reference.sites * frames);
    assert.equal(arrays.reset_joint_pos.length, demo.sim.qpos_ids.length);
    const reset = clips[demo.reference.reset.clip];
    assert.equal(reset.name, 'idle');
    assert.ok(reset.idle);
    assert.equal(ref.clip, demo.reference.reset.clip);
    assert.deepEqual(Array.from(ref.anchor.subarray(0, 2)).map(Math.abs).every((x) => x < 1), true);
  });
});

describe('actuator delay', () => {
  test('each actuator reads the bus `lag` ticks back, history backfilled at reset', () => {
    const nu = 7;
    const blocks = [
      { actuator_ids: [0, 1, 2, 3, 4], delay_ticks: 15 },
      { actuator_ids: [5, 6], delay_ticks: 5 },
    ];
    const delay = new ActuatorDelay(nu, blocks);
    const hold = Float64Array.from({ length: nu }, (_, i) => 0.1 * i);
    const ctrl = new Float64Array(nu);
    delay.reset(hold, ctrl);
    assert.deepEqual(Array.from(ctrl), Array.from(hold));
    const bus = [];
    for (let tick = 0; tick < 60; tick++) {
      if (tick % 5 === 0) for (let i = 0; i < nu; i++) delay.ctrl[i] = Math.sin(tick + i);
      bus.push(Float64Array.from(delay.ctrl));
      delay.write(ctrl);
      for (let i = 0; i < nu; i++) {
        const lag = i < 5 ? 15 : 5;
        const expected = tick - lag >= 0 ? bus[tick - lag][i] : hold[i];
        assert.equal(ctrl[i], expected, `tick ${tick} actuator ${i}`);
      }
    }
  });
});

describe('previous command', () => {
  test('history of clamped commands, shifted by control index, zero at reset', () => {
    const cfg = { offset: 4, history: 3, width: 2, low: -1, high: 1 };
    const pc = new PreviousCommand(cfg);
    const input = new Float32Array(10);
    pc.reset();
    pc.write(0, input);
    assert.deepEqual(Array.from(input.subarray(4)), [0, 0, 0, 0, 0, 0]);
    pc.update(Float32Array.of(0.5, -2));
    pc.write(1, input);
    assert.deepEqual(Array.from(input.subarray(4)), [0, 0, 0, 0, 0.5, -1]);
    pc.update(Float32Array.of(0.25, 0.75));
    pc.write(3, input); // a gap of two steps repeats the held command
    assert.deepEqual(Array.from(input.subarray(4)), [0.5, -1, 0.25, 0.75, 0.25, 0.75]);
    pc.write(3, input); // same index: no shift
    assert.deepEqual(Array.from(input.subarray(4)), [0.5, -1, 0.25, 0.75, 0.25, 0.75]);
  });
});
