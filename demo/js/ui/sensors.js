// OSL onboard sensing (G, or #btn-sensors): the card of strip charts
// (#sensor-card) of what the prosthesis itself measures, over the last few
// seconds, and the history it draws from. render/scene.js draws the matching
// markers on the leg (setSensorMarkers) in the same group colours.
//
//   const card = new SensorCard(document.getElementById('sensor-card'), demo);
//   card.push(frame);                  // every worker frame, before view.setFrame
//   card.setVisible(true);             // shown: draws at the next draw() call
//   card.setMode('full' | 'compact');  // the rows (app.js picks it by the room)
//   card.draw(view.playTime, now);     // every animation frame; draws when due
//
// Data: frame.sensors (render/frame.js), the OSL's processed sensor frame,
// channels named as render/frame.js frameChannels names them. A frame's
// values are copied into a ring (SensorHistory, about 3.2 s at 100 Hz) on
// arrival, after the rows of the steps whose frame the worker could not post
// (frame.sensorRows, worker.js): the frame's buffer goes back to the worker
// once the renderer is done with it. With those rows every control step is
// there, so the line breaks only where steps are really missing (more than
// 1.5 control steps apart); frames alone (an older worker) carry only the
// steps that went out, and a step more than GAP s after the one before
// breaks the line. A new epoch (a reset) clears the ring. The state a reset
// starts from (frame.initial) is no measurement and is not stored. Frames
// without sensors (an older worker) leave the card empty with a note.
//
// Drawing: two stacked canvases at min(devicePixelRatio, 2): a static layer
// (titles, glyphs, grid, ranges, channel keys), redrawn only on a resize or a
// mode change, and a live layer (the traces, the latest values). draw() takes
// the renderer's play time, so the right edge of every chart is the pose on
// screen (the last sample is interpolated to it), and draws at most every
// 1 / MAX_RATE s, only while shown, once the time has moved and new data has
// arrived. When the play time stands still (a pause, a stall) for SETTLE_MS
// while the card shows an earlier time, it draws once more, at that time, so
// a paused card shows the paused pose; then nothing while paused, hidden or
// off screen (app.js does not call it). The latest values change at most
// every VALUE_MS (and with that last drawing). Fixed y ranges (from native
// rollouts of the bundle, see tools/DESIGN.md), each with a zero line; a
// trace beyond its range is clipped at the plot's edge.
//
// Colours: one hue per sensor group (the glyph before each row title and the
// markers on the leg), its channels in steps of that hue, darkest first, each
// named beside its value. See SENSOR_GROUPS.
//
// No DOM access at import: tools/test/sensors.test.mjs imports it in Node.

import { frameChannels } from '../render/frame.js';

export const SPAN = 3;           // s of history drawn
export const CAPACITY = 320;     // samples kept: 3.2 s at one per 10 ms control step
export const GAP = 0.05;         // s: frames alone (an older worker): a longer step between samples breaks the line
export const STREAM_GAP_STEPS = 1.5; // control steps: with every step's row, a longer step breaks the line
export const MAX_RATE = 30;      // draws per second at most
export const VALUE_MS = 100;     // the numbers change at most this often
export const SETTLE_MS = 50;     // a play time standing still this long is drawn (a pause)
const RATE_SLACK_MS = 3;         // requestAnimationFrame jitter allowed below 1 / MAX_RATE
const MAX_DPR = 2;
const TIMES_KEPT = 128;          // draw durations kept for stats()

// Sensor groups: label, the marker glyph (render/scene.js and the row glyph
// here), the hue (`color`, the markers and the row glyph) and its steps for
// the lines of a row (`steps`, darkest first: x y z, heel toe, knee ankle).
// Chosen with the dataviz method (tools/DESIGN.md, "Sensor charts"): four hue
// families, none of the scene's own (the X-ray arrows' green #008300 and
// violet #6250d6, the orange covers and accent, the slate ghost): IMU sky,
// load cell jade, foot contact gold, encoders rose. Each group colour is at
// least 20 (OKLab ΔE x 100, normal vision) from both arrows and 16 or more
// from the orange (gold, on the black foot; rose 19); all pairs pass the
// validator on the white card (normal vision 18.5 or more, colour-vision
// deficiency 11 or more). Sky, jade and gold are under 3:1 on white, so every
// row and line is named and every value printed. The steps of each hue pass
// the validator's ordinal checks on white (darkest first, ΔL >= 0.06, the
// lightest >= 2:1); the gold marker is lighter than its lines (it sits on the
// black foot; on white its lines need the contrast).
export const SENSOR_GROUPS = Object.freeze({
  imu: Object.freeze({ label: 'IMU', glyph: 'circle', color: '#2aa1fe', steps: Object.freeze(['#015e9d', '#007ed1', '#2aa1fe']) }),
  cell: Object.freeze({ label: 'Ankle load cell', glyph: 'diamond', color: '#18bd91', steps: Object.freeze(['#046e53', '#089772', '#18bd91']) }),
  contact: Object.freeze({ label: 'Foot contact', glyph: 'circle', color: '#d4b122', steps: Object.freeze(['#876f02', '#b59507']) }),
  encoder: Object.freeze({ label: 'Encoders', glyph: 'square', color: '#cf2b74', steps: Object.freeze(['#9c0352', '#cf2b74']) }),
});

// demo.visual.overlays.markers[*].kind -> sensor group.
export const MARKER_GROUPS = Object.freeze({ imu: 'imu', load_cell: 'cell', contact: 'contact', encoder: 'encoder' });

const DEG = 180 / Math.PI;

// The rows, in the order the card shows them; `compact` rows make the short
// card. channels: names in demo.sim.osl.observation.terms[*].channels; names:
// what the card calls them; range: the fixed y range (after `scale`); digits:
// decimals of the latest value; limit: lines at the sensor's own clamp (the
// saturation of its read in demo.json, else this).
export const SENSOR_ROWS = Object.freeze([
  { key: 'gyro', group: 'imu', title: 'IMU gyro', unit: 'rad/s', compact: true,
    channels: ['osl_imu_gyro_x', 'osl_imu_gyro_y', 'osl_imu_gyro_z'], names: ['x', 'y', 'z'], range: [-6, 6], digits: 1 },
  { key: 'accel', group: 'imu', title: 'IMU accel', unit: 'm/s²',
    channels: ['osl_imu_acc_x', 'osl_imu_acc_y', 'osl_imu_acc_z'], names: ['x', 'y', 'z'], range: [-48, 48], digits: 0,
    limit: { read: 'osl_imu_acc', value: 39.2266 } },
  { key: 'force', group: 'cell', title: 'Load cell force', unit: 'N', compact: true,
    channels: ['osl_ankle_load_force_x', 'osl_ankle_load_force_y', 'osl_ankle_load_force_z'], names: ['x', 'y', 'z'], range: [-1600, 600], digits: 0 },
  { key: 'moment', group: 'cell', title: 'Load cell moment', unit: 'N·m',
    channels: ['osl_ankle_load_torque_x', 'osl_ankle_load_torque_y', 'osl_ankle_load_torque_z'], names: ['x', 'y', 'z'], range: [-100, 50], digits: 0 },
  { key: 'contact', group: 'contact', title: 'Foot contact', unit: 'N', compact: true,
    channels: ['osl_heel_contact', 'osl_toe_contact'], names: ['heel', 'toe'], range: [0, 1500], digits: 0 },
  { key: 'encoders', group: 'encoder', title: 'Encoders', unit: 'deg',
    channels: ['osl_knee_pos', 'osl_ankle_pos'], names: ['knee', 'ankle'], range: [-30, 90], digits: 0, scale: DEG },
].map((row) => Object.freeze({ ...row, channels: Object.freeze(row.channels), names: Object.freeze(row.names), range: Object.freeze(row.range) })));

// The frame.sensors channel names of demo.json, in frame order:
// [{index, term, name, unit}], by the one rule render/frame.js frameLayout
// uses too (frameChannels; a demo.json without terms[*].channels names them
// by their reads).
export function sensorChannels(demo) {
  return frameChannels(demo && demo.sim);
}

// The rows this demo.json can fill: each with the frame indices of its
// channels (lines whose channel is missing are left out; a row without any
// is dropped) and its limit (the clamp of its read, if demo.json has one).
export function resolveRows(demo, rows = SENSOR_ROWS) {
  const channels = sensorChannels(demo);
  const byName = new Map(channels.map((c) => [c.name, c.index]));
  const reads = [];
  const terms = demo && demo.sim && demo.sim.osl && demo.sim.osl.observation && demo.sim.osl.observation.terms;
  if (Array.isArray(terms)) for (const t of terms) if (Array.isArray(t.reads)) reads.push(...t.reads);
  const out = [];
  for (const row of rows) {
    const lines = [];
    row.channels.forEach((name, k) => {
      if (byName.has(name)) lines.push({ index: byName.get(name), name: row.names[k], step: k });
    });
    if (!lines.length) continue;
    let limit = null;
    if (row.limit) {
      const read = reads.find((r) => r.name === row.limit.read);
      limit = read && Number.isFinite(read.saturation) ? read.saturation : row.limit.value;
    }
    out.push({ ...row, lines, limit });
  }
  return { rows: out, width: channels.length };
}

// ------------------------------------------------------------------ history
// The last CAPACITY sensor frames of one epoch, oldest first: times (s) and
// values (copied). O(1) per frame, no allocation after construction. `gap`:
// the longest step between two samples that the line (and sample()) still
// joins; GAP for frames alone, STREAM_GAP_STEPS control steps once every
// step's row arrives (SensorCard).
export class SensorHistory {
  constructor(width, capacity = CAPACITY) {
    this.width = width;
    this.capacity = capacity;
    this.times = new Float64Array(capacity);
    this.values = new Float32Array(capacity * width);
    this.start = 0;      // slot of the oldest sample
    this.count = 0;
    this.epoch = null;   // the epoch of the samples held
    this.version = 0;    // changes with every stored or cleared sample
    this.gap = GAP;
  }

  clear() {
    this.start = 0;
    this.count = 0;
    this.version += 1;
  }

  // Store the sensors of a frame at `time` (s) of `epoch` (values[offset ..
  // offset + width)): a new epoch clears the ring first; a time before the
  // newest is dropped, the same time replaces it. Returns whether it stored
  // the values.
  push(time, epoch, values, offset = 0) {
    if (epoch !== this.epoch) {
      this.clear();
      this.epoch = epoch;
    }
    if (!Number.isFinite(time) || !values || values.length < offset + this.width) return false;
    let slot;
    if (this.count) {
      const newest = this.times[this.slot(this.count - 1)];
      if (time < newest) return false;
      if (time === newest) slot = this.slot(this.count - 1);
    }
    if (slot === undefined) {
      if (this.count < this.capacity) {
        slot = this.slot(this.count);
        this.count += 1;
      } else {
        slot = this.start;
        this.start = (this.start + 1) % this.capacity;
      }
    }
    this.times[slot] = time;
    const w = this.width, o = slot * w;
    for (let k = 0; k < w; k++) this.values[o + k] = values[offset + k];
    this.version += 1;
    return true;
  }

  slot(i) { return (this.start + i) % this.capacity; }
  time(i) { return this.times[this.slot(i)]; }
  value(i, channel) { return this.values[this.slot(i) * this.width + channel]; }
  get newest() { return this.count ? this.time(this.count - 1) : null; }
  get oldest() { return this.count ? this.time(0) : null; }

  // The index of the last sample at or before t (-1: none).
  indexAt(t) {
    let lo = 0, hi = this.count - 1, found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.time(mid) <= t) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return found;
  }

  // A channel at time t: between the samples around it (no gap between
  // them), else the last sample at or before t; NaN before the first.
  sample(t, channel) {
    const i = this.indexAt(t);
    if (i < 0) return NaN;
    const a = this.value(i, channel);
    if (i + 1 >= this.count) return a;
    const ta = this.time(i), tb = this.time(i + 1);
    if (tb - ta > this.gap || tb <= ta) return a;
    return a + ((t - ta) / (tb - ta)) * (this.value(i + 1, channel) - a);
  }

  // Where the line breaks: steps between consecutive samples longer than gap.
  breaks() {
    let n = 0;
    for (let i = 1; i < this.count; i++) if (this.time(i) - this.time(i - 1) > this.gap) n += 1;
    return n;
  }
}

// ------------------------------------------------------------------ card
const MINUS = '−';
const NO_DATA = 'No sensor data from the simulation.';
const NO_ROWS = "No charts for this simulation's sensors.";
const formatValue = (v, digits) => {
  if (!Number.isFinite(v)) return '–';
  const text = v.toFixed(digits);
  if (/^-0(\.0+)?$/.test(text)) return text.slice(1); // no "−0"
  return text.startsWith('-') ? MINUS + text.slice(1) : text;
};
// A tick label: whole numbers in full, with a real minus.
const formatTick = (v) => (v < 0 ? MINUS + String(Math.abs(v)) : String(v));

// Layout of a row on the canvas (CSS px): the title line's height, the gap
// under the plot, the tick labels' gutter (at least this; wider when the
// widest label needs it, see _layout) and the values column. The row height
// is the plots box's (demo.css) over the rows shown.
const LAYOUT = {
  full: { title: 15, gap: 7, gutter: 28, values: 62 },
  compact: { title: 14, gap: 6, gutter: 28, values: 58 },
};
const TITLE_X = 16; // the row title's left edge, after its glyph
const TICK_FONT = '500 9.5px';
const TICK_PAD = 4; // CSS px between a tick label and its plot (right-aligned there)
const TICK_MARGIN = 1; // CSS px left of the widest label

// A row's range labels (right-aligned beside the plot): {text, value} at its
// limit lines when the sensor clamps inside the range, else at the range's
// ends; plus '0' when the zero line is far enough from both ends (decided
// when drawn).
function tickLabels(row) {
  const [lo, hi] = row.range;
  if (row.limit && row.limit < hi && -row.limit > lo) {
    return [{ text: formatTick(Math.round(row.limit)), value: row.limit }, { text: formatTick(-Math.round(row.limit)), value: -row.limit }];
  }
  return [{ text: formatTick(hi), value: hi }, { text: formatTick(lo), value: lo }];
}

export class SensorCard {
  // root: #sensor-card, holding .sc-plots (two canvases are made in it),
  // .sc-note; demo: demo.json. options.groups overrides SENSOR_GROUPS
  // (tests and tuning).
  constructor(root, demo, options = {}) {
    this.root = root;
    this.groups = { ...SENSOR_GROUPS, ...(options.groups || {}) };
    const { rows, width } = resolveRows(demo, options.rows || SENSOR_ROWS);
    this.allRows = rows;
    this.history = new SensorHistory(Math.max(1, width));
    this.channelCount = width;
    // The control step (s): with every step's row (frame.sensorRows) the line
    // breaks only across a missing step.
    const step = demo && demo.sim && demo.sim.policy_dt;
    this.step = Number.isFinite(step) && step > 0 ? step : 0.01;
    this.streamed = false;    // frames carry the rows of the steps not posted (worker.js)
    this.mode = root && root.dataset.mode === 'compact' ? 'compact' : 'full';
    this.rows = this._rowsFor(this.mode);
    this.visible = false;
    this.framesWithout = 0;   // frames that came without sensors
    this.framesWith = 0;
    this.framesInitial = 0;   // frames of the state a reset starts from (not stored)
    this.rowsIn = 0;          // rows of steps whose frame did not go out
    this._forced = true;      // draw at the next draw() whatever the clock
    this._staticDirty = true;
    this._lastT = NaN;
    this._lastVersion = -1;
    this._lastDraw = -Infinity;
    this._valuesAt = -Infinity;
    this._values = [];        // per row, per line: the text shown
    this._latest = new Float64Array(Math.max(1, width)).fill(NaN); // the values shown (as sampled)
    this._latestAt = NaN;     // the time they were sampled at
    this._times = new Float32Array(TIMES_KEPT);
    this._timeCount = 0;
    this.counts = { draws: 0, statics: 0, skipped: 0, settled: 0 };
    this._seenT = NaN;        // the play time draw() got last, and since when (ms)
    this._seenAt = -Infinity;
    this._geometry = null;    // CSS size and pixel ratio the canvases were sized for
    this._plots = root ? root.querySelector('.sc-plots') : null;
    this._note = root ? root.querySelector('.sc-note') : null;
    this._layers = null;
    if (this._plots) {
      const make = (cls) => {
        const canvas = document.createElement('canvas');
        canvas.className = cls;
        canvas.setAttribute('aria-hidden', 'true');
        this._plots.append(canvas);
        return { canvas, ctx: canvas.getContext('2d') };
      };
      this._layers = { static: make('sc-static'), live: make('sc-live') };
      this._box = null; // the plots box's size as the observer saw it last: draw() then reads no layout
      if (typeof ResizeObserver !== 'undefined') {
        // A box of another size than the canvases' (not the first sizing,
        // which draw() itself does when the card is shown).
        this._observer = new ResizeObserver(() => {
          const g = this._geometry, box = this._plots;
          const size = { w: box.clientWidth, h: box.clientHeight }; // laid out already: no extra layout
          if (g && (size.w !== g.w || size.h !== g.h)) this.invalidate();
          this._box = size;
        });
        this._observer.observe(this._plots);
      }
    }
  }

  _rowsFor(mode) {
    return mode === 'compact' ? this.allRows.filter((r) => r.compact) : this.allRows.slice();
  }

  // A worker frame: first the rows of the steps whose frame did not go out
  // (frame.sensorRows, oldest first, all older than the frame), then its own
  // sensors, into the history (copied now: the frame's buffer may go back to
  // the worker at once). A frame of the state a reset starts from
  // (frame.initial) only starts the new epoch. Frames without sensors count
  // toward the card's note.
  push(frame) {
    if (!frame) return;
    const h = this.history, w = this.channelCount;
    if (frame.sensorRows !== undefined && !this.streamed) {
      this.streamed = true; // a worker that sends every step: break the line only across a missing one
      h.gap = STREAM_GAP_STEPS * this.step;
    }
    const rows = frame.sensorRows;
    if (rows && rows.times && rows.values && w > 0 && rows.values.length === rows.times.length * w) {
      for (let i = 0; i < rows.times.length; i++) {
        if (h.push(rows.times[i], frame.epoch, rows.values, i * w)) this.rowsIn += 1;
      }
    }
    if (frame.initial) {
      this.framesInitial += 1;
      if (frame.epoch !== h.epoch) h.push(NaN, frame.epoch, null); // a reset: clears the ring
      return;
    }
    const values = frame.sensors;
    if (!values || values.length < w || w === 0) {
      this.framesWithout += 1;
      // A reset still clears what was there.
      if (frame.epoch !== h.epoch) h.push(NaN, frame.epoch, null);
      return;
    }
    this.framesWith += 1;
    h.push(frame.time, frame.epoch, values);
  }

  setVisible(on) {
    on = !!on;
    if (on === this.visible) return;
    this.visible = on;
    if (on) {
      this._forced = true;
      this._valuesAt = -Infinity;
      this._box = null; // read the box once: it may have changed while hidden
    }
  }

  // 'full' (every row) or 'compact' (the rows marked compact).
  setMode(mode) {
    mode = mode === 'compact' ? 'compact' : 'full';
    if (mode === this.mode) return;
    this.mode = mode;
    this.rows = this._rowsFor(mode);
    this._values = [];
    this.invalidate(); // a new height: read before the observer reports it
  }

  // The canvases' size, pixel ratio or fonts changed: the static layer is
  // redrawn and the live one with it at the next draw().
  invalidate() {
    this._staticDirty = true;
    this._forced = true;
    this._valuesAt = -Infinity;
    this._box = null; // read the box at the next draw
    this._layoutKey = null; // the tick labels' widths (a web font may have loaded)
  }

  // Draw if due (see the header): t is the renderer's play time (s, or null
  // before the first timed frame), now a requestAnimationFrame time (ms).
  // Returns whether it drew.
  draw(t, now = (typeof performance !== 'undefined' ? performance.now() : Date.now())) {
    if (!this.visible || !this._layers) return false;
    const h = this.history;
    if (!Number.isFinite(t)) t = h.newest;
    // A new pixel ratio (a browser zoom) resizes the canvases: draw at once.
    if (this._geometry && this._geometry.dpr !== Math.min(MAX_DPR, Math.max(1, globalThis.devicePixelRatio || 1))) this.invalidate();
    if (t !== this._seenT) {
      this._seenT = t;
      this._seenAt = now;
    }
    // The pose's time is not the one drawn (the traces, or the values, which
    // change less often).
    const stale = Number.isFinite(t) && (t !== this._lastT || t !== this._latestAt);
    const settled = stale && now - this._seenAt >= SETTLE_MS; // and the clock stopped there
    const fresh = h.version !== this._lastVersion;
    const full = this._forced || settled; // drawn now, the values with it
    if (!full) {
      if (!(stale && fresh)) return false;
      if (now - this._lastDraw < 1000 / MAX_RATE - RATE_SLACK_MS) {
        this.counts.skipped += 1;
        return false;
      }
    }
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (!this._fit()) return false;
    if (this._staticDirty) this._drawStatic();
    this._drawLive(t, now, full);
    if (settled) this.counts.settled += 1;
    this._lastT = t;
    this._lastVersion = h.version;
    this._lastDraw = now;
    this._forced = false;
    this.counts.draws += 1;
    const ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
    this._times[this._timeCount % TIMES_KEPT] = ms;
    this._timeCount += 1;
    return true;
  }

  // Draw counts and times (ms): the last, mean, 95th percentile and maximum
  // of the last TIMES_KEPT draws; the history held.
  stats() {
    const n = Math.min(this._timeCount, TIMES_KEPT);
    const times = Array.from(this._times.subarray(0, n)).sort((a, b) => a - b);
    const last = this._timeCount ? this._times[(this._timeCount - 1) % TIMES_KEPT] : null;
    const h = this.history;
    return {
      visible: this.visible,
      mode: this.mode,
      ...this.counts,
      ms: n ? {
        last, mean: times.reduce((s, v) => s + v, 0) / n,
        p95: times[Math.min(n - 1, Math.floor(0.95 * n))], max: times[n - 1], n,
      } : null,
      samples: h.count,
      span: h.count ? h.newest - h.oldest : 0,
      epoch: h.epoch,
      breaks: h.breaks(),
      gap: h.gap,
      streamed: this.streamed,
      rows: this.rowsIn,
      framesWith: this.framesWith,
      framesWithout: this.framesWithout,
      framesInitial: this.framesInitial,
    };
  }

  // The card as drawn last (tests): the play time of the traces' right edge
  // (drawnAt), the rows (each with its plot's box on the canvases, CSS px)
  // and their lines with the value shown, the time those were sampled at
  // (at), the canvases' pixel ratio.
  info() {
    const layout = this._geometry ? this._layout() : null;
    return {
      mode: this.mode,
      drawnAt: this._lastT,
      at: this._latestAt,
      dpr: this._geometry ? this._geometry.dpr : null,
      rows: this.rows.map((row, r) => ({
        key: row.key, group: row.group, title: row.title, unit: row.unit,
        plot: layout ? { ...layout[r].plot } : null,
        // The range labels as drawn: right-aligned at `right` (CSS px), `width` wide.
        ticks: this._ticks && this._ticks.has(row.key) ? this._ticks.get(row.key).map((t) => ({ ...t })) : [],
        lines: row.lines.map((line, k) => ({
          name: line.name, index: line.index, color: this._lineColor(row, line),
          value: this._latest[line.index] * (row.scale || 1), text: (this._values[r] && this._values[r][k]) || '',
        })),
      })),
      empty: this.history.count === 0,
      noData: this.framesWith === 0 && this.framesWithout > 0,
      noRows: this.allRows.length === 0,
    };
  }

  // The latest values as text, row by row ("IMU gyro, rad/s: x 0.1, y −0.4,
  // z 0.0. ..."), as the card shows them (app.js: a summary for screen
  // readers, which get no canvas). '' before the first drawing.
  summary() {
    if (!this.allRows.length) return NO_ROWS;
    if (this.framesWith === 0 && this.framesWithout > 0) return NO_DATA;
    const parts = [];
    this.rows.forEach((row, r) => {
      const texts = this._values[r];
      if (!texts || !texts.length) return;
      parts.push(`${row.title}, ${row.unit}: ${row.lines.map((line, k) => `${line.name} ${texts[k] || '–'}`).join(', ')}.`);
    });
    return parts.join(' ');
  }

  dispose() {
    if (this._observer) this._observer.disconnect();
    if (this._layers) for (const layer of Object.values(this._layers)) layer.canvas.remove();
    this._layers = null;
  }

  // ---------------------------------------------------------------- internals
  _lineColor(row, line) {
    const g = this.groups[row.group];
    const steps = g.steps && g.steps.length ? g.steps : [g.color];
    return steps[Math.min(line.step, steps.length - 1)];
  }

  // Size both canvases to the plots box at min(devicePixelRatio, 2); false
  // when the box has no size (hidden). The size comes from the resize
  // observer once it has seen the box shown: reading clientWidth in every
  // draw would force a layout after each HUD text change (the X-ray legend's
  // numbers, the status rows).
  _fit() {
    const box = this._plots;
    const seen = this._box && this._box.w > 0 && this._box.h > 0 && this._geometry;
    const w = seen ? this._box.w : box.clientWidth, h = seen ? this._box.h : box.clientHeight;
    if (!(w > 0 && h > 0)) return false;
    const dpr = Math.min(MAX_DPR, Math.max(1, globalThis.devicePixelRatio || 1));
    const g = this._geometry;
    if (!g || g.w !== w || g.h !== h || g.dpr !== dpr) {
      for (const { canvas } of Object.values(this._layers)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      this._geometry = { w, h, dpr };
      this._staticDirty = true;
    }
    return true;
  }

  // Where each row goes (CSS px): its title line, plot and values column;
  // kept until the canvases' size, the mode or the fonts change. The gutter
  // fits the widest range label of any row (both modes alike), so none is
  // cut at the canvas's left edge ('−1600' is 28-28.5 px wide in Inter).
  _layout() {
    const { w, h } = this._geometry;
    const key = `${this.mode} ${w} ${h}`;
    if (this._layoutKey === key) return this._layoutCache;
    const L = LAYOUT[this.mode];
    const gutter = Math.max(L.gutter, Math.ceil(this._tickWidth() + TICK_PAD + TICK_MARGIN));
    const rowHeight = h / Math.max(1, this.rows.length);
    this._layoutKey = key;
    this._layoutCache = this.rows.map((row, r) => {
      const top = r * rowHeight;
      const plot = { x: gutter, y: top + L.title, w: Math.max(10, w - gutter - L.values - 8), h: Math.max(8, rowHeight - L.title - L.gap) };
      return { row, top, titleY: top + L.title - 4, plot, values: { x: w - L.values, w: L.values } };
    });
    return this._layoutCache;
  }

  // The widest range label of all rows in the tick font (CSS px; 0 without a
  // canvas).
  _tickWidth() {
    const ctx = this._layers && this._layers.static.ctx;
    if (!ctx || typeof ctx.measureText !== 'function') return 0;
    const theme = this._themeCache || this._theme();
    ctx.save();
    ctx.font = `${TICK_FONT} ${theme.text}`;
    let width = 0;
    for (const row of this.allRows) {
      for (const label of [...tickLabels(row), { text: '0' }]) width = Math.max(width, ctx.measureText(label.text).width);
    }
    ctx.restore();
    return width;
  }

  _theme() {
    const css = typeof getComputedStyle === 'function' ? getComputedStyle(this.root) : null;
    const v = (name, fallback) => (css && css.getPropertyValue(name).trim()) || fallback;
    return {
      ink: v('--ink', '#1d1d1f'), ink2: v('--ink-2', '#3a3a3c'), muted: v('--muted', '#6e6e73'),
      text: v('--text', 'system-ui, sans-serif'), mono: v('--mono', 'ui-monospace, monospace'),
    };
  }

  _drawStatic() {
    const { canvas, ctx } = this._layers.static;
    const { dpr } = this._geometry;
    const theme = this._theme();
    this._themeCache = theme;
    this._ticks = new Map(); // row key -> its range labels as drawn (info())
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const compact = this.mode === 'compact';
    for (const { row, titleY, plot, values } of this._layout()) {
      const group = this.groups[row.group];
      // Title: the group's glyph (as on the leg: a faint dark edge, a white
      // ring, the colour), the name, the unit.
      drawGlyph(ctx, group.glyph, 6.75, titleY - 4, 3.6, group.color, 1.2);
      ctx.font = `600 ${compact ? 11.5 : 12}px ${theme.text}`;
      ctx.fillStyle = theme.ink;
      ctx.textBaseline = 'alphabetic';
      ctx.textAlign = 'left';
      ctx.fillText(row.title, TITLE_X, titleY);
      const tw = ctx.measureText(row.title).width;
      ctx.font = `500 ${compact ? 10.5 : 11}px ${theme.text}`;
      ctx.fillStyle = theme.muted;
      ctx.fillText(row.unit, TITLE_X + tw + 5, titleY);
      // Plot: a faint band, second guides, the zero line, the limits, the range.
      const [lo, hi] = row.range;
      const y = (v) => plot.y + ((hi - v) / (hi - lo)) * plot.h;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.028)';
      roundRect(ctx, plot.x, plot.y, plot.w, plot.h, 3);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.07)';
      ctx.beginPath();
      for (let s = 1; s < SPAN; s++) {
        const x = crisp(plot.x + (s / SPAN) * plot.w, dpr);
        ctx.moveTo(x, plot.y);
        ctx.lineTo(x, plot.y + plot.h);
      }
      ctx.stroke();
      if (lo <= 0 && hi > 0) { // the zero line (the plot's bottom edge for contact, 0..1500 N)
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.22)';
        ctx.beginPath();
        const yz = crisp(Math.min(y(0), plot.y + plot.h - 0.5), dpr);
        ctx.moveTo(plot.x, yz);
        ctx.lineTo(plot.x + plot.w, yz);
        ctx.stroke();
      }
      if (row.limit) {
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        for (const v of [row.limit, -row.limit]) {
          if (v <= lo || v >= hi) continue;
          const yl = crisp(y(v), dpr);
          ctx.moveTo(plot.x, yl);
          ctx.lineTo(plot.x + plot.w, yl);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // The range's ends, or, for a sensor that clamps, its limit (dashed).
      ctx.font = `${TICK_FONT} ${theme.text}`;
      ctx.fillStyle = theme.muted;
      ctx.textAlign = 'right';
      const ticks = [];
      const tick = (text, at, baseline) => {
        ctx.textBaseline = baseline;
        ctx.fillText(text, plot.x - TICK_PAD, at);
        ticks.push({ text, right: plot.x - TICK_PAD, width: ctx.measureText(text).width });
      };
      const [first, second] = tickLabels(row);
      if (row.limit && row.limit < hi && -row.limit > lo) {
        tick(first.text, y(first.value), 'middle');
        tick(second.text, y(second.value), 'middle');
      } else {
        tick(first.text, plot.y - 1, 'top');
        tick(second.text, plot.y + plot.h + 1, 'bottom');
      }
      if (lo < 0 && hi > 0) {
        const yz = y(0);
        if (yz - plot.y > 12 && plot.y + plot.h - yz > 12) tick('0', yz, 'middle');
      }
      this._ticks.set(row.key, ticks);
      // Channel keys: a short line in the channel's step and its name.
      const pitch = plot.h / Math.max(row.lines.length, 1);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.font = `500 ${compact ? 10 : 10.5}px ${theme.text}`;
      row.lines.forEach((line, k) => {
        const cy = plot.y + pitch * (k + 0.5);
        ctx.strokeStyle = this._lineColor(row, line);
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(values.x + 1, cy);
        ctx.lineTo(values.x + 9, cy);
        ctx.stroke();
        ctx.fillStyle = theme.ink2;
        ctx.fillText(line.name, values.x + 13, cy);
      });
    }
    this._staticDirty = false;
    this.counts.statics += 1;
    // The live layer is redrawn in full after this.
    this._valuesAt = -Infinity;
  }

  _drawLive(t, now, full = false) {
    const { canvas, ctx } = this._layers.live;
    const { dpr } = this._geometry;
    const h = this.history;
    const theme = this._themeCache || this._theme();
    const layout = this._layout();
    const showValues = full || now - this._valuesAt >= VALUE_MS;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (showValues) ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const t1 = Number.isFinite(t) ? t : 0, t0 = t1 - SPAN;
    // The samples in view: from the last one at or before t0 (the line enters
    // at the left edge) to the last at or before t1, then t1 interpolated.
    const last = Number.isFinite(t) ? h.indexAt(t1) : -1;
    let first = last >= 0 ? h.indexAt(t0) : -1;
    if (first < 0 && last >= 0) first = 0;
    const gap = h.gap;
    const next = last >= 0 && last + 1 < h.count && h.time(last + 1) - h.time(last) <= gap ? last + 1 : -1;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const { row, plot } of layout) {
      if (!showValues) ctx.clearRect(plot.x - 1, plot.y - 1, plot.w + 2, plot.h + 2);
      if (last < 0) continue;
      const [lo, hi] = row.range;
      const ky = plot.h / (hi - lo), kx = plot.w / SPAN, scale = row.scale || 1;
      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.x, plot.y - 1, plot.w, plot.h + 2);
      ctx.clip();
      // Lightest first, so the darkest step lies on top.
      for (let k = row.lines.length - 1; k >= 0; k--) {
        const line = row.lines[k], c = line.index;
        ctx.strokeStyle = this._lineColor(row, line);
        ctx.beginPath();
        let prev = -Infinity;
        for (let i = first; i <= last; i++) {
          const ti = h.time(i);
          const x = plot.x + (ti - t0) * kx;
          const yv = plot.y + (hi - h.value(i, c) * scale) * ky;
          if (ti - prev > gap) ctx.moveTo(x, yv);
          else ctx.lineTo(x, yv);
          prev = ti;
        }
        if (next >= 0 && h.time(last) < t1) {
          const ta = h.time(last), tb = h.time(next);
          const a = h.value(last, c), b = h.value(next, c);
          const v = a + ((t1 - ta) / (tb - ta)) * (b - a);
          ctx.lineTo(plot.x + plot.w, plot.y + (hi - v * scale) * ky);
        }
        ctx.stroke();
      }
      ctx.restore();
    }
    // The latest values: as sampled at t, at most every VALUE_MS.
    if (!showValues) return;
    this._valuesAt = now;
    this._latestAt = Number.isFinite(t) ? t : NaN;
    const compact = this.mode === 'compact';
    ctx.font = `500 ${compact ? 10 : 10.5}px ${theme.mono}`;
    ctx.fillStyle = theme.ink;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    layout.forEach(({ row, plot, values }, r) => {
      const pitch = plot.h / Math.max(row.lines.length, 1);
      const texts = this._values[r] || (this._values[r] = []);
      row.lines.forEach((line, k) => {
        const v = last >= 0 ? h.sample(t1, line.index) : NaN;
        this._latest[line.index] = v;
        const text = formatValue(v * (row.scale || 1), row.digits);
        texts[k] = text;
        ctx.fillText(text, values.x + values.w, plot.y + pitch * (k + 0.5));
      });
    });
    if (this._note) {
      // Frames without sensor data (an older simulation), or a demo.json whose
      // channels match no row: the card would stay empty without a word.
      const text = !this.allRows.length ? NO_ROWS : this.framesWith === 0 && this.framesWithout > 0 ? NO_DATA : '';
      if (text && this._note.textContent !== text) this._note.textContent = text;
      if (this._note.hidden === !!text) this._note.hidden = !text;
    }
  }
}

// A crisp 1-px line position at this pixel ratio.
function crisp(v, dpr) {
  return (Math.round(v * dpr - 0.5) + 0.5) / dpr;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') {
    ctx.roundRect(x, y, w, h, r);
    return;
  }
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// A group's glyph, centred at (x, y), its coloured core of radius r: as the
// markers on the leg, with `ring` px of white around the core and a faint
// dark edge (0.75 px) around that (ring 0: the core alone).
export function drawGlyph(ctx, glyph, x, y, r, color, ring = 0) {
  const shape = (rr) => {
    ctx.beginPath();
    if (glyph === 'diamond') {
      const d = rr * 1.18;
      ctx.moveTo(x, y - d);
      ctx.lineTo(x + d, y);
      ctx.lineTo(x, y + d);
      ctx.lineTo(x - d, y);
      ctx.closePath();
    } else if (glyph === 'square') {
      const s = rr * 0.9;
      roundRect(ctx, x - s, y - s, 2 * s, 2 * s, Math.min(1.2 + (rr - r), s));
    } else {
      ctx.arc(x, y, rr, 0, 2 * Math.PI);
    }
  };
  if (ring > 0) {
    shape(r + ring + 0.75);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fill();
    shape(r + ring);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }
  shape(r);
  ctx.fillStyle = color;
  ctx.fill();
}
