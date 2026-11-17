"""Ground-truth traces of osl_deploy's own Simulation, for SimCore's Node checks.

    cd tools
    python -m duet_export.trace              # tools/traces/r2_trace.bin.gz
    python -m duet_export.trace --robust     # tools/traces/r2_trace_robust.json

Both run osl_deploy's synchronous runner.Simulation on the bundle at its
defaults (lockstep, no stress, actuator delay "train", OSL sensor lag
"validation", "mean" inference with the fp32 ONNX actors, rebase every 5 s,
start at the idle clip, frame 0) and request the UI's twists (simcfg.UI_LEVELS)
before every control step, as osl_deploy's interactive mode does.

Trace (DUETBIN1). Boundary b is the state the policies sample at control step b
(b = 0: the reset); step k runs from boundary k to boundary k + 1. N steps:

    boundary arrays, N + 1 rows (the state BEFORE that boundary's mj_forward;
    mj_forward does not touch qacc_warmstart, only mj_step saves it):
      qpos [84] qvel [83] act [339] ctrl [341] qacc_warmstart [83] time  float64
      ref_clip ref_previous ref_frame ref_blend_count ref_loops since_anchor int32
      ref_previous_frame ref_previous_rate float64, ref_anchor ref_previous_anchor
      [3], ref_catchup [2], ref_twist [3] float64   (GaitReference.state_keys;
      clip indices are LIBRARY indices)
    step arrays, N rows:
      requested [3] float64       twist requested just before step k
      human_obs [2338] float32    human input at boundary k
      osl_obs [600] float32       OSL input at step k as fed (last_action filled)
      human_mean [339], osl_mean [2] float32   the ONNX outputs
      bus [341] float64           command bus after step k's actions applied
      site_error float64, switched uint8, failed uint8

The robust mode drives a seeded pseudo-random key sequence over every UI
(level, action) state, switching every 1.5-4 s, counts falls (record
"failed"), resets after one and writes a JSON summary.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import DEFAULT_BUNDLE, load_bundle
from .container import write_container
from .simcfg import ACTIONS, DEFAULT_LEVEL, FAST_LEVEL, UI_LEVELS, clip_of, decimal_twist

TRACE_FORMAT = "duetmimic-trace/1"
TRACES = Path(__file__).resolve().parents[1] / "traces"
STATE_INT = ("clip", "previous", "frame", "blend_count", "loops")
STATE_FLOAT = ("previous_frame", "previous_rate")
STATE_VEC = {"anchor": 3, "previous_anchor": 3, "catchup": 2, "twist": 3}

# (seconds, action) at the default level; the long forward segment reaches the
# 5 s rebase (500 control steps after the last re-anchoring switch), and the
# final switch into idle blends with Python-float (float32) interpolation.
SCHEDULE = (
    (1.0, "idle"),
    (2.0, "forward"),
    (1.5, "forward_left"),
    (1.0, "turn_left"),
    (1.5, "backward"),
    (5.5, "forward"),
    (1.0, "idle"),
)


def ui_twist(twists, level, action):
    """The exact library label the UI requests for (level index, action)."""
    clip = clip_of(twists, decimal_twist(UI_LEVELS[level], action))
    return np.array([float(x) for x in twists[clip]])


def make_simulation(bundle):
    from osl_deploy.runner import Simulation

    return Simulation(bundle)  # every default: see the module docstring


def record_trace(bundle_dir, out_path, schedule=SCHEDULE, level=DEFAULT_LEVEL):
    import mujoco

    bundle = load_bundle(bundle_dir)
    twists = bundle.reference_arrays()["twists"]
    sim = make_simulation(bundle)
    m, d = sim.model, sim.data
    plan = []
    for seconds, action in schedule:
        plan += [ui_twist(twists, level, action)] * round(seconds / bundle.dt)
    n = len(plan)
    rows = n + 1
    out = {
        "qpos": np.zeros((rows, m.nq)),
        "qvel": np.zeros((rows, m.nv)),
        "act": np.zeros((rows, m.na)),
        "ctrl": np.zeros((rows, m.nu)),
        "qacc_warmstart": np.zeros((rows, m.nv)),
        "time": np.zeros(rows),
        "since_anchor": np.zeros(rows, np.int32),
        **{f"ref_{key}": np.zeros(rows, np.int32) for key in STATE_INT},
        **{f"ref_{key}": np.zeros(rows) for key in STATE_FLOAT},
        **{f"ref_{key}": np.zeros((rows, size)) for key, size in STATE_VEC.items()},
        "requested": np.zeros((n, 3)),
        "human_obs": np.zeros((n, 2338), np.float32),
        "osl_obs": np.zeros((n, 600), np.float32),
        "human_mean": np.zeros((n, 339), np.float32),
        "osl_mean": np.zeros((n, 2), np.float32),
        "bus": np.zeros((n, m.nu)),
        "site_error": np.zeros(n),
        "switched": np.zeros(n, np.uint8),
        "failed": np.zeros(n, np.uint8),
    }

    def boundary(b):
        out["qpos"][b], out["qvel"][b] = d.qpos, d.qvel
        out["act"][b], out["ctrl"][b] = d.act, d.ctrl
        out["qacc_warmstart"][b], out["time"][b] = d.qacc_warmstart, d.time
        out["since_anchor"][b] = sim._since_anchor
        ref = sim.reference
        for key in (*STATE_INT, *STATE_FLOAT, *STATE_VEC):
            out[f"ref_{key}"][b] = getattr(ref, key)

    boundary(0)
    started = time.perf_counter()
    steps = n
    for k, twist in enumerate(plan):
        sim.request_twist(twist)
        r = sim.step()
        osl = r["controllers"]["osl"]
        out["requested"][k] = twist
        out["human_obs"][k] = r["observation"]
        out["osl_obs"][k] = osl["policy_observation"]
        out["human_mean"][k] = r["mean"]
        out["osl_mean"][k] = osl["mean"]
        out["bus"][k] = sim.actuators.ctrl
        out["site_error"][k] = r["site_error"]
        out["switched"][k] = r["switched"]
        out["failed"][k] = r["failed"]
        boundary(k + 1)
        if r["failed"]:
            print(f"[trace] failed at step {k}: {sim.failure_reason}")
            steps = k + 1
            break
    elapsed = time.perf_counter() - started
    if steps < n:  # Keep what ran.
        for key, value in out.items():
            out[key] = value[: steps + (len(value) - n)]

    # Self-check: the recorded boundary state alone reproduces the native
    # boundary observation's state-derived part (everything before the
    # reference: joints, muscles, touch, tracked-body kinematics) after
    # mj_forward, so a replay needs nothing else.
    from osl_deploy.observations import FullBodyObservation

    observer = FullBodyObservation(bundle)
    measured = observer.layout.groups["reference"].start
    replay = mujoco.MjData(m)
    worst = 0.0
    for b in range(0, steps, max(1, steps // 40)):
        mujoco.mj_resetData(m, replay)
        replay.qpos[:], replay.qvel[:] = out["qpos"][b], out["qvel"][b]
        replay.act[:], replay.ctrl[:] = out["act"][b], out["ctrl"][b]
        replay.qacc_warmstart[:], replay.time = out["qacc_warmstart"][b], out["time"][b]
        mujoco.mj_forward(m, replay)
        state = observer(replay, sim.reference)[:measured]
        worst = max(worst, float(np.abs(state - out["human_obs"][b][:measured]).max()))
    meta = {
        "format": TRACE_FORMAT,
        "bundle": bundle.folder.name,
        "steps": int(steps),
        "dt": bundle.dt,
        "schedule": [
            {"seconds": s, "action": a, "twist": ui_twist(twists, level, a).tolist()}
            for s, a in schedule
        ],
        "level": level,
        "clip_names": bundle.manifest["reference"]["clip_names"],
        "settings": "osl_deploy runner.Simulation defaults (lockstep, delay train, "
        "sensor lag validation, mean inference, rebase 5 s, idle frame 0)",
        "native_replay_max_abs": worst,
        "seconds_wall": elapsed,
    }
    TRACES.mkdir(parents=True, exist_ok=True)
    stats = write_container(out_path, out, meta)
    failed = int(out["failed"].any())
    print(
        f"[trace] {steps} steps ({steps * bundle.dt:.2f} s sim, {elapsed:.1f} s wall), "
        f"failed={failed}, switches={int(out['switched'].sum())}, "
        f"native replay max |obs diff| {worst:.3g}, {out_path} "
        f"{stats['bytes'] / 1e6:.1f} MB"
    )
    return meta


def robust(bundle_dir, out_path, seconds=90.0, seed=0, dwell=(1.5, 4.0)):
    """Random UI key sequence; counts falls. Returns the JSON summary."""
    bundle = load_bundle(bundle_dir)
    twists = bundle.reference_arrays()["twists"]
    names = bundle.manifest["reference"]["clip_names"]
    rng = np.random.default_rng(seed)
    states = [(lv, a) for lv in range(len(UI_LEVELS)) for a in ACTIONS.values()]
    # Every state once (shuffled), then random draws: transitions are random.
    order = [states[i] for i in rng.permutation(len(states))]
    segments, total = [], 0.0
    while total < seconds:
        state = order.pop() if order else states[rng.integers(len(states))]
        length = float(rng.uniform(*dwell))
        segments.append((state, length))
        total += length
    sim = make_simulation(bundle)
    falls, worst, log = [], [], []
    started = time.perf_counter()
    t, previous = 0.0, ("reset", "idle")
    for (level, action), length in segments:
        twist = ui_twist(twists, level, action)
        steps = round(length / bundle.dt)
        peak = 0.0
        for _ in range(steps):
            sim.request_twist(twist)
            r = sim.step()
            peak = max(peak, r["site_error"])
            t += bundle.dt
            if r["failed"]:
                falls.append(
                    {
                        "time": round(t, 2),
                        "level": UI_LEVELS[level]["name"],
                        "action": action,
                        "twist": twist.tolist(),
                        "previous": previous,
                        "reason": sim.failure_reason,
                    }
                )
                print(f"[robust] FALL at {t:.2f} s: {falls[-1]}")
                sim.reset()
                break
        worst.append(peak)
        log.append(
            {
                "level": UI_LEVELS[level]["name"],
                "action": action,
                "clip": names[sim.reference.clip],
                "seconds": round(length, 2),
                "peak_site_error": round(peak, 4),
            }
        )
        previous = (UI_LEVELS[level]["name"], action)
    summary = {
        "format": "duetmimic-robust/1",
        "bundle": bundle.folder.name,
        "seed": seed,
        "sim_seconds": round(t, 2),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "segments": len(segments),
        "distinct_states": len({s for s, _ in segments}),
        "falls": falls,
        "max_peak_site_error": round(max(worst), 4),
        "threshold": 0.5,
        "levels": [dict(level) for level in UI_LEVELS],
        "fast_level": FAST_LEVEL,
        "log": log,
    }
    TRACES.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"[robust] {summary['sim_seconds']} s sim in {summary['wall_seconds']} s wall, "
        f"{len(segments)} segments, {summary['distinct_states']} states, "
        f"{len(falls)} falls, max peak site error {summary['max_peak_site_error']}"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="duet_export.trace", description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.robust:
        out = args.out or TRACES / "r2_trace_robust.json"
        robust(args.bundle, out, args.seconds, args.seed)
    else:
        record_trace(args.bundle, args.out or TRACES / "r2_trace.bin.gz")


if __name__ == "__main__":
    main()
