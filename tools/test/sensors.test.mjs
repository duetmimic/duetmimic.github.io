// The sensor charts' data side (demo/js/ui/sensors.js; node --test, no DOM):
// the rows against demo.json's OSL sensor channels (as render/frame.js lays
// out frame.sensors, also when demo.json lacks the channel names), the group
// colours against the scene's other colours (render/palette.js and the
// covers' colour in demo.json: the dataviz checks the module's header cites),
// the history ring (capacity, epochs, ordering, interpolation at a time,
// gaps, copying), and the card's intake without a page: the rows of skipped
// steps (frame.sensorRows), the reset state (frame.initial), a demo.json
// whose channels match no row.

import assert from 'node:assert/strict';
import { before, describe, test } from 'node:test';

import { loadJSON } from '../../demo/js/lib/container.js';
import { frameLayout } from '../../demo/js/render/frame.js';
import { ACCENT, XRAY_COLORS } from '../../demo/js/render/palette.js';
import {
  CAPACITY, GAP, MARKER_GROUPS, SENSOR_GROUPS, SENSOR_ROWS, SPAN, STREAM_GAP_STEPS, SensorCard, SensorHistory, resolveRows, sensorChannels,
} from '../../demo/js/ui/sensors.js';
import { ASSETS } from './paths.mjs';

let demo;
before(async () => {
  demo = await loadJSON(new URL('demo.json', ASSETS));
});

// OKLab distance x 100 (normal vision), WCAG contrast and OKLCH lightness:
// the measures of the dataviz skill's validator.
const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const rgb = (hex) => [0, 2, 4].map((i) => lin(parseInt(hex.slice(1 + i, 3 + i), 16) / 255));
function oklab(hex) {
  const [r, g, b] = rgb(hex);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
const oklabLinear = ([r, g, b]) => {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
};
const deltaE = (a, b) => { const p = oklab(a), q = oklab(b); return 100 * Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); };
// Under protanopia and deuteranopia (Machado, Oliveira and Fernandes 2009,
// severity 1, as the validator simulates them): the smaller distance.
const MACHADO = [
  [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
];
const deltaECvd = (a, b) => Math.min(...MACHADO.map((M) => {
  const sim = (hex) => oklabLinear(M.map((row) => Math.min(1, Math.max(0, row[0] * rgb(hex)[0] + row[1] * rgb(hex)[1] + row[2] * rgb(hex)[2]))));
  const p = sim(a), q = sim(b);
  return 100 * Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
}));
const luminance = (hex) => { const [r, g, b] = rgb(hex); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
const contrast = (a, b) => { const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };

describe('rows and channels', () => {
  test('the card reads frame.sensors as render/frame.js lays it out', () => {
    const layout = frameLayout(demo);
    assert.ok(layout.sensors, 'demo.json has the OSL sensor terms');
    const channels = sensorChannels(demo);
    assert.deepEqual(channels.map((c) => c.name), layout.sensors.channels.map((c) => c.name));
    assert.deepEqual(channels.map((c) => c.unit), layout.sensors.channels.map((c) => c.unit));
  });

  test('without channel names in demo.json both sides name the channels by their reads, and the rows still fill', () => {
    const bare = structuredClone(demo);
    for (const term of bare.sim.osl.observation.terms) { delete term.channels; delete term.units; }
    const names = sensorChannels(bare).map((c) => c.name);
    assert.deepEqual(names, frameLayout(bare).sensors.channels.map((c) => c.name), 'one naming rule');
    assert.deepEqual(names, sensorChannels(demo).map((c) => c.name), 'the read names give the exported names (bundle B)');
    assert.equal(resolveRows(bare).rows.length, SENSOR_ROWS.length);
  });

  test('every row is filled, in the order the user asked for, the compact rows a subset', () => {
    const { rows, width } = resolveRows(demo);
    assert.equal(width, frameLayout(demo).sensors.length);
    assert.deepEqual(rows.map((r) => r.key), ['gyro', 'accel', 'force', 'moment', 'contact', 'encoders']);
    for (const row of rows) {
      const want = SENSOR_ROWS.find((r) => r.key === row.key);
      assert.equal(row.lines.length, want.channels.length, `${row.key}: every channel is in demo.json`);
      const names = sensorChannels(demo);
      for (const line of row.lines) assert.equal(names[line.index].name, want.channels[line.step]);
      assert.ok(row.range[0] < row.range[1]);
      if (row.range[0] < 0) assert.ok(row.range[1] > 0, `${row.key}: the zero line lies inside the range`);
    }
    assert.deepEqual(rows.filter((r) => r.compact).map((r) => r.key), ['gyro', 'force', 'contact']);
    // The accelerometer's limit is its read's saturation in demo.json, inside the range.
    const accel = rows.find((r) => r.key === 'accel');
    const read = demo.sim.osl.observation.terms.flatMap((t) => t.reads || []).find((r) => r.name === 'osl_imu_acc');
    assert.equal(accel.limit, read.saturation);
    assert.ok(accel.limit < accel.range[1] && -accel.limit > accel.range[0]);
    // Encoders in degrees.
    assert.ok(Math.abs(rows.find((r) => r.key === 'encoders').scale - 180 / Math.PI) < 1e-12);
  });

  test('every marker kind of demo.json has a group, every group enough steps for its rows', () => {
    for (const m of demo.visual.overlays.markers) assert.ok(SENSOR_GROUPS[MARKER_GROUPS[m.kind]], `marker ${m.key} (${m.kind})`);
    for (const row of SENSOR_ROWS) {
      assert.ok(SENSOR_GROUPS[row.group].steps.length >= row.channels.length, `${row.key}: ${row.channels.length} lines`);
    }
  });
});

describe('colours (the dataviz checks)', () => {
  // The scene's own colours: the X-ray arrows and the slate bone ghost
  // (render/palette.js, which render/scene.js draws with), the accent, and
  // the orange covers as exported (demo.json visual.materials, sRGB).
  const arrows = { 'X-ray force': XRAY_COLORS.force, 'X-ray moment': XRAY_COLORS.moment };
  const toHex = (rgb) => `#${rgb.map((c) => Math.round(255 * c).toString(16).padStart(2, '0')).join('')}`;
  let scene;
  before(() => {
    const cover = demo.visual.materials.find((m) => m.category === 'osl_cover');
    assert.ok(cover, 'demo.json has the orange covers\' material');
    scene = { 'orange cover': toHex(cover.rgb), 'accent': ACCENT, 'slate ghost': XRAY_COLORS.bone };
  });
  test('each group hue is clear of the X-ray arrows, the orange and the slate ghost, and of the others', () => {
    const groups = Object.entries(SENSOR_GROUPS);
    for (const [key, g] of groups) {
      assert.match(g.color, /^#[0-9a-f]{6}$/);
      // Clearly apart from the arrows (both can be on): ΔE 20 or more (now
      // 20.0 sky against the violet, 20.0 jade against the green, rose 25).
      for (const [name, hex] of Object.entries(arrows)) {
        assert.ok(deltaE(g.color, hex) >= 19.5, `${key} ${g.color} vs ${name} ${hex}: ΔE ${deltaE(g.color, hex).toFixed(1)}`);
      }
      // The validator's normal-vision floor against the rest (gold 16 from
      // the orange, sky 18 from the slate ghost).
      for (const [name, hex] of Object.entries(scene)) {
        assert.ok(deltaE(g.color, hex) >= 15.5, `${key} ${g.color} vs ${name} ${hex}: ΔE ${deltaE(g.color, hex).toFixed(1)}`);
      }
    }
    // Any two markers can sit side by side on the leg: all pairs, normal
    // vision at least 15 (the validator's floor; 18.5 now) and simulated
    // colour-vision deficiency at least 8 (its target; 11.0 now).
    for (let i = 0; i < groups.length; i++) {
      for (let j = i + 1; j < groups.length; j++) {
        const [a, b] = [groups[i][1].color, groups[j][1].color];
        assert.ok(deltaE(a, b) >= 15, `${groups[i][0]} vs ${groups[j][0]}: ΔE ${deltaE(a, b).toFixed(1)}`);
        assert.ok(deltaECvd(a, b) >= 8, `${groups[i][0]} vs ${groups[j][0]}: CVD ΔE ${deltaECvd(a, b).toFixed(1)}`);
      }
    }
  });

  // The lines in each row are steps of its group's hue, darker than the
  // glyph, so they come nearer the arrows' green and violet (OKLab ΔE x 100:
  // jade lines 11.3-11.4 from the force green, sky 12.5-13.2 from the moment
  // violet, the dark gold 13.7 from the green). The validator's floor of 15
  // would take mid steps of chroma 0.085 at L 0.65 (grey-green, grey-blue)
  // or break the steps' hue and lightness ladder (an OKLCH search,
  // 2026-11-19), so these stay, named by their row's title and glyph; this
  // keeps them from drifting closer.
  test('each line colour stays clear of the X-ray arrows', () => {
    for (const [key, g] of Object.entries(SENSOR_GROUPS)) {
      for (const step of g.steps) {
        for (const [name, hex] of Object.entries(arrows)) {
          assert.ok(deltaE(step, hex) >= 11, `${key} line ${step} vs ${name} ${hex}: ΔE ${deltaE(step, hex).toFixed(1)}`);
        }
      }
    }
  });

  test('each group\'s steps read as steps of its hue on the white card', () => {
    const hue = (hex) => { const [, a, b] = oklab(hex); return (Math.atan2(b, a) * 180) / Math.PI; };
    const apart = (p, q) => Math.abs(((p - q + 540) % 360) - 180);
    for (const [key, g] of Object.entries(SENSOR_GROUPS)) {
      for (const step of g.steps) assert.ok(apart(hue(step), hue(g.color)) <= 4, `${key}: ${step} has the hue of ${g.color}`);
      const L = g.steps.map((hex) => oklab(hex)[0]);
      for (let k = 1; k < L.length; k++) assert.ok(L[k] - L[k - 1] >= 0.06, `${key}: darkest first, ΔL ${(L[k] - L[k - 1]).toFixed(3)}`);
      const lightest = g.steps[g.steps.length - 1];
      assert.ok(contrast(lightest, '#ffffff') >= 2, `${key}: ${lightest} at ${contrast(lightest, '#ffffff').toFixed(2)}:1 on white`);
      assert.ok(contrast(g.color, '#ffffff') >= 2, `${key}: the glyph ${g.color} at ${contrast(g.color, '#ffffff').toFixed(2)}:1 on white`);
    }
  });
});

describe('SensorHistory', () => {
  const frame = (t, v, n = 3) => Float32Array.from({ length: n }, (_, k) => v + k);

  test('keeps the newest CAPACITY samples, oldest first', () => {
    const h = new SensorHistory(3);
    assert.equal(h.capacity, CAPACITY);
    assert.ok(CAPACITY * 0.01 >= SPAN, 'the ring holds the span drawn at 100 Hz');
    for (let i = 0; i < CAPACITY + 25; i++) assert.ok(h.push(i * 0.01, 0, frame(i * 0.01, i)));
    assert.equal(h.count, CAPACITY);
    assert.ok(Math.abs(h.oldest - 0.25) < 1e-12 && Math.abs(h.newest - (CAPACITY + 24) * 0.01) < 1e-12);
    for (let i = 0; i < h.count; i++) {
      assert.equal(h.value(i, 0), 25 + i);
      assert.equal(h.value(i, 2), 27 + i);
      if (i) assert.ok(h.time(i) > h.time(i - 1));
    }
  });

  test('a new epoch clears it; an older time is dropped, the same time replaces', () => {
    const h = new SensorHistory(3);
    h.push(1.0, 0, frame(1, 1));
    h.push(1.01, 0, frame(1, 2));
    const v = h.version;
    assert.equal(h.push(1.0, 0, frame(1, 9)), false, 'older: dropped');
    assert.equal(h.version, v);
    assert.equal(h.push(1.01, 0, frame(1, 7)), true, 'the same time: replaced');
    assert.equal(h.count, 2);
    assert.equal(h.value(1, 0), 7);
    h.push(0.0, 1, frame(0, 5)); // a reset: time starts over
    assert.equal(h.count, 1);
    assert.equal(h.epoch, 1);
    assert.equal(h.value(0, 0), 5);
    assert.equal(h.push(NaN, 1, frame(0, 1)), false, 'no time');
    assert.equal(h.push(0.5, 1, new Float32Array(2)), false, 'too few channels');
    assert.equal(h.push(0.5, 1, null), false);
    assert.equal(h.count, 1);
  });

  test('copies the values: the frame\'s buffer may go back to the worker', () => {
    const h = new SensorHistory(3);
    const buffer = new ArrayBuffer(12);
    const view = new Float32Array(buffer);
    view.set([1, 2, 3]);
    h.push(0.5, 0, view);
    view.set([9, 9, 9]);
    structuredClone(buffer, { transfer: [buffer] }); // detaches it, as postMessage does
    assert.equal(view.length, 0);
    assert.deepEqual([0, 1, 2].map((c) => h.value(0, c)), [1, 2, 3]);
  });

  test('stores a row out of a larger array (frame.sensorRows)', () => {
    const h = new SensorHistory(2);
    const rows = Float32Array.from([1, 2, 3, 4, 5, 6]);
    assert.ok(h.push(0.01, 0, rows, 2));
    assert.ok(h.push(0.02, 0, rows, 4));
    assert.equal(h.push(0.03, 0, rows, 5), false, 'a row past the end');
    assert.deepEqual([h.value(0, 0), h.value(0, 1), h.value(1, 0), h.value(1, 1)], [3, 4, 5, 6]);
  });

  test('samples a time between steps, not across a gap', () => {
    const h = new SensorHistory(1);
    h.push(1.00, 0, [0]);
    h.push(1.01, 0, [10]);
    h.push(1.01 + 2 * GAP, 0, [100]); // frames missing for longer than GAP
    assert.equal(h.indexAt(0.99), -1);
    assert.equal(h.indexAt(1.005), 0);
    assert.ok(Number.isNaN(h.sample(0.99, 0)));
    assert.ok(Math.abs(h.sample(1.005, 0) - 5) < 1e-9);
    assert.equal(h.sample(1.05, 0), 10, 'across the gap: the last sample before it');
    assert.equal(h.sample(5, 0), 100);
  });
});

describe('SensorCard without a page', () => {
  test('collects frames, notes frames without sensors, never draws', () => {
    const card = new SensorCard(null, demo);
    const n = frameLayout(demo).sensors.length;
    card.push({ time: 0.01, epoch: 0, sensors: new Float32Array(n).fill(1) });
    card.push({ time: 0.02, epoch: 0, sensors: new Float32Array(n).fill(2) });
    assert.equal(card.history.count, 2);
    card.setVisible(true);
    assert.equal(card.draw(0.02, 1000), false, 'no canvases');
    assert.equal(card.info().noData, false);
    const old = new SensorCard(null, demo);
    old.push({ time: 0.01, epoch: 0 }); // an older worker's frame
    assert.equal(old.history.count, 0);
    assert.equal(old.info().noData, true);
    assert.deepEqual(old.stats().framesWithout, 1);
    // A reset still clears the history, even from a frame without sensors.
    card.push({ time: 0.0, epoch: 1 });
    assert.equal(card.history.count, 0);
  });

  test('takes the rows of skipped steps first, never charts the reset state, and then breaks the line only across a missing step', () => {
    const n = frameLayout(demo).sensors.length;
    const values = (v) => new Float32Array(n).fill(v);
    const rows = (times) => ({ times: Float64Array.from(times), values: Float32Array.from(times.flatMap((t) => Array(n).fill(t * 100))) });
    const card = new SensorCard(null, demo);
    assert.equal(card.history.gap, GAP, 'frames alone (an older worker): the 50 ms rule');
    card.push({ time: 1.0, epoch: 3, initial: true, sensors: values(9), sensorRows: null }); // the reset pose
    assert.equal(card.history.count, 0, 'the reset state is not charted');
    assert.equal(card.history.epoch, 3, 'but it starts the epoch');
    assert.ok(card.streamed && Math.abs(card.history.gap - STREAM_GAP_STEPS * demo.sim.policy_dt) < 1e-12, 'a worker that sends every step');
    card.push({ time: 1.01, epoch: 3, sensors: values(101), sensorRows: null });
    card.push({ time: 1.05, epoch: 3, sensors: values(105), sensorRows: rows([1.02, 1.03, 1.04]) });
    const h = card.history;
    assert.deepEqual(Array.from({ length: h.count }, (_, i) => +h.time(i).toFixed(2)), [1.01, 1.02, 1.03, 1.04, 1.05]);
    assert.deepEqual(Array.from({ length: h.count }, (_, i) => Math.round(h.value(i, 0))), [101, 102, 103, 104, 105]);
    assert.equal(h.breaks(), 0);
    card.push({ time: 1.08, epoch: 3, sensors: values(108), sensorRows: null }); // two steps really missing
    assert.equal(h.breaks(), 1, 'a missing step breaks the line');
    const st = card.stats();
    assert.deepEqual([st.rows, st.framesInitial, st.framesWith, st.streamed], [3, 1, 3, true]);
    // An older worker's frames: no rows, the 50 ms rule; a 20 ms step joins.
    const old = new SensorCard(null, demo);
    old.push({ time: 1.0, epoch: 0, sensors: values(1) });
    old.push({ time: 1.02, epoch: 0, sensors: values(2) });
    assert.ok(!old.streamed && old.history.gap === GAP && old.history.breaks() === 0);
  });

  test('a demo.json whose channels match no row says so', () => {
    const other = structuredClone(demo);
    for (const term of other.sim.osl.observation.terms) if (term.channels) term.channels = term.channels.map((c) => `other_${c}`);
    const card = new SensorCard(null, other);
    assert.equal(card.allRows.length, 0);
    card.push({ time: 0.01, epoch: 0, sensors: new Float32Array(frameLayout(other).sensors.length) });
    assert.equal(card.info().noRows, true);
    assert.match(card.summary(), /No charts/);
  });
});
