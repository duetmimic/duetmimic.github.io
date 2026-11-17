// Twist commands: training's command support (osl_deploy commands.CommandSupport),
// the request latch of runner.Simulation (request_twist / _consume_request) and
// the WASD mapping of demo.json "ui". No DOM; runs in a worker and in Node.

// CommandSupport: what the policy observes for a requested twist. Zero when
// the nearest clip is idle, else each axis clamped into the support of the
// whole library ([low, high]), zero again if that lands nearest to idle.
// Nearest-clip selection runs over the reachable clips (reference.select), so
// a request equal to a UI twist (an exact clip label) selects exactly that clip.
export class CommandSupport {
  constructor(refCfg, reference) {
    this.reference = reference;
    this.low = Float64Array.from(refCfg.command.low);
    this.high = Float64Array.from(refCfg.command.high);
  }

  select(twist) { return this.reference.select(twist); }

  // The served twist for `twist`, into out.
  serve(twist, out) {
    const ref = this.reference;
    if (ref.idleClips[ref.select(twist)]) return out.fill(0);
    for (let i = 0; i < 3; i++) out[i] = Math.min(Math.max(twist[i], this.low[i]), this.high[i]);
    if (ref.idleClips[ref.select(out)]) out.fill(0);
    return out;
  }
}

// runner.Simulation's request latch: the latest request wins and is consumed at
// the next control boundary, after physics and before the reference advances.
// No switch during a blend (the request waits for it to end); a periodic rebase
// re-anchors the reference onto the robot every `rebaseSteps` control steps.
export class CommandLatch {
  constructor(rebaseSteps) {
    this.rebaseSteps = rebaseSteps | 0;
    this.requested = new Float64Array(3);
    this._served = new Float64Array(3);
    this.pending = false;
    this.sinceAnchor = 0;
  }

  reset(twist) {
    this.requested.set(twist);
    this.pending = false;
    this.sinceAnchor = 0;
  }

  request(twist) {
    if (!twist || twist.length !== 3 || !Number.isFinite(twist[0])
        || !Number.isFinite(twist[1]) || !Number.isFinite(twist[2])) {
      throw new TypeError('A twist is three finite numbers [vx, vy, wz]');
    }
    this.requested[0] = twist[0]; this.requested[1] = twist[1]; this.requested[2] = twist[2];
    this.pending = true;
  }

  // The command boundary. Returns whether the reference switched clip or
  // re-anchored. qpos[q..q+7] is the robot's free-joint pose.
  consume(reference, support, qpos, q = 0) {
    this.sinceAnchor += 1;
    if (reference.blendCount < reference.blendSteps) return false;
    const rebase = this.rebaseSteps > 0 && this.rebaseSteps <= this.sinceAnchor;
    if (!(this.pending || rebase)) return false;
    const served = this._served;
    if (this.pending) support.serve(this.requested, served);
    else served.set(reference.twist); // A rebase keeps the current command.
    const switched = rebase || reference.select(served) !== reference.clip;
    if (switched) {
      reference.switch(served, qpos, q);
      this.sinceAnchor = 0;
    } else {
      reference.twist.set(served); // Same clip: no re-anchoring, no blend.
    }
    this.pending = false;
    return switched;
  }
}

// ---------------------------------------------------------------------------
// WASD. `keys` is a Set of held KeyboardEvent.code values ('KeyW', 'ArrowUp',
// 'ShiftLeft', ...); single letters ('w', 'W') and 'Shift' are accepted too.
// `level` is the selected speed level (0-based index into ui.levels; the app
// keeps it, e.g. Digit1-5 and Q/E, see levelForKey); Shift held uses
// ui.fast_level. Returns {twist, action, keys, label, speed, level, fast}:
// twist equals a library clip label exactly.
// ---------------------------------------------------------------------------
const ACTIONS = [
  // turn: right (-1), none, left (+1)
  ['backward_right', 'backward', 'backward_left'], // direction -1
  ['turn_right', 'idle', 'turn_left'], // direction 0
  ['forward_right', 'forward', 'forward_left'], // direction +1
];

function held(keys, codes) {
  for (const code of codes) {
    if (keys.has(code)) return true;
    if (code.startsWith('Key')) {
      const letter = code.slice(3);
      if (keys.has(letter.toLowerCase()) || keys.has(letter)) return true;
    } else if (code.startsWith('Shift') && keys.has('Shift')) {
      return true;
    }
  }
  return false;
}

export function actionForKeys(keys, ui) {
  const b = ui.bindings;
  const direction = (held(keys, b.forward) ? 1 : 0) - (held(keys, b.backward) ? 1 : 0);
  const turn = (held(keys, b.left) ? 1 : 0) - (held(keys, b.right) ? 1 : 0);
  return { action: ACTIONS[direction + 1][turn + 1], fast: held(keys, b.fast) };
}

export function twistForKeys(keys, ui, level = ui.default_level) {
  const { action, fast } = actionForKeys(keys, ui);
  const top = ui.levels.length - 1;
  const index = fast ? ui.fast_level : Math.min(Math.max(level | 0, 0), top);
  const entry = ui.levels[index].actions[action];
  return {
    twist: entry.twist.slice(),
    action,
    keys: entry.keys,
    label: entry.label,
    speed: entry.speed,
    level: index,
    fast,
  };
}

// The level a key press selects (Digit1-5 directly, Q/E one down/up), or the
// current one for any other key. `code` is a KeyboardEvent.code.
export function levelForKey(code, ui, level = ui.default_level) {
  const b = ui.bindings;
  const top = ui.levels.length - 1;
  const direct = b.levels.indexOf(code);
  if (direct >= 0) return Math.min(direct, top);
  if (b.slower.includes(code)) return Math.max(level - 1, 0);
  if (b.faster.includes(code)) return Math.min(level + 1, top);
  return level;
}

// Every twist the UI can request, as {level, action, twist}.
export function uiTwists(ui) {
  const out = [];
  ui.levels.forEach((lv, level) => {
    for (const [action, entry] of Object.entries(lv.actions)) out.push({ level, action, twist: entry.twist });
  });
  return out;
}
