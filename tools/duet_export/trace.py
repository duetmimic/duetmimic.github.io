"""Ground-truth traces of osl_deploy's own Simulation, for SimCore's Node checks.

    cd tools
    python -m duet_export.trace              # tools/traces/trace.bin.gz + trace.json
    python -m duet_export.trace --robust     # tools/traces/trace_robust.json
                                             # (--seed N: trace_robust_seedN.json)

The bundle defaults to common.DEFAULT_BUNDLE, the one the demo's assets come
from; tools/test/sim.test.mjs skips a trace whose manifest_sha256 differs from
demo.json's source.manifest_sha256 ("stale trace").

Both run osl_deploy's synchronous runner.Simulation on the bundle at its
defaults (lockstep, no stress, actuator delay "train", OSL sensor lag
"validation", "mean" inference with the fp32 ONNX actors, rebase every 5 s,
start at the idle clip, frame 0) and request the UI's twists (simcfg.UI_LEVELS)
before every control step, as osl_deploy's interactive mode does.

Trace (DUETBIN1). Boundary b is the state the policies sample at control step b
(b = 0: the reset); step k runs from boundary k to boundary k + 1. N steps.
Sizes come from the bundle's manifest (H, O: the human and OSL observation
sizes; A, P: their action sizes; B: 2338, 500, 339, 2):

    boundary arrays, N + 1 rows (the state BEFORE that boundary's mj_forward;
    mj_forward does not touch qacc_warmstart, only mj_step saves it):
      qpos [nq] qvel [nv] act [na] ctrl [nu] qacc_warmstart [nv] time  float64
      ref_clip ref_previous ref_frame ref_blend_count ref_loops since_anchor int32
      ref_previous_frame ref_previous_rate float64, ref_anchor ref_previous_anchor
      [3], ref_catchup [2], ref_twist [3] float64   (GaitReference.state_keys;
      clip indices are LIBRARY indices)
    boundary extras, N + 1 rows, float64, taken where the runner reads the OSL
    sensors (after the boundary's mj_forward; b = 0: after the reset's):
      osl_frame [F]        the OSL observer's newest processed frame, before
                           any lag: its bus terms' frames in observation order
                           (meta extras.osl_frame.terms; B: actpack 10,
                           loadcell 6, contact 2 = 18). Saturated and tared,
                           exactly osl_deploy's SensorObservation.frame.
      socket_cfrc [6]      cfrc_int of body robot/OSL_KA_L1_osl_generic_socket:
                           MuJoCo's [torque, force] in world axes, the torque
                           about subtree_com of the body's tree root; the
                           wrench its parent (the femur) applies to the socket
                           subtree
      root_subtree_com [3] subtree_com of that tree root
      socket_xpos [3]      xpos of the socket body (its joints' anchor)
    (the boundary after a failed step has NaN extras. A numerical or watchdog
    failure stops the step before that boundary's mj_forward and sensor read;
    a tracking failure, the site error over its threshold, which is how falls
    show up, still runs both, but the trace marks every failed boundary the
    same way)
    step arrays, N rows:
      requested [3] float64       twist requested just before step k
      human_obs [H] float32       human input at boundary k
      osl_obs [O] float32         OSL input at step k as fed (last_action filled)
      human_mean [A], osl_mean [P] float32   the ONNX outputs
      bus [nu] float64            command bus after step k's actions applied
      site_error float64, switched uint8, failed uint8

meta (also written to trace.json beside the container): bundle, manifest_sha256,
sizes, schedule, extras (names, ids, frame layout), the replay self-checks.

The robust mode drives a seeded pseudo-random key sequence over every UI
(level, action) state, switching every 1.5-4 s, counts falls (record
"failed"), resets after one and writes a JSON summary.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import DEFAULT_BUNDLE, bundle_label, load_bundle, manifest_sha256
from .container import write_container
from .simcfg import ACTIONS, DEFAULT_LEVEL, FAST_LEVEL, UI_LEVELS, clip_of, decimal_twist

TRACE_FORMAT = "duetmimic-trace/2"
TRACES = Path(__file__).resolve().parents[1] / "traces"
TRACE_FILE = "trace.bin.gz"
STATE_INT = ("clip", "previous", "frame", "blend_count", "loops")
STATE_FLOAT = ("previous_frame", "previous_rate")
STATE_VEC = {"anchor": 3, "previous_anchor": 3, "catchup": 2, "twist": 3}
SOCKET_BODY = "robot/OSL_KA_L1_osl_generic_socket"
EXTRAS = ("osl_frame", "socket_cfrc", "root_subtree_com", "socket_xpos")

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


def sidecar(out_path):
    """trace.bin.gz -> trace.json (the meta, readable without the container)."""
    name = Path(out_path).name
    for suffix in (".bin.gz", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return Path(out_path).with_name(f"{name}.json")


def record_trace(bundle_dir, out_path, schedule=SCHEDULE, level=DEFAULT_LEVEL):
    import mujoco
    from osl_deploy.sensor_observation import SensorObservation

    bundle = load_bundle(bundle_dir)
    sha = manifest_sha256(bundle_dir)
    twists = bundle.reference_arrays()["twists"]
    sim = make_simulation(bundle)
    m, d = sim.model, sim.data
    # Input and output sizes from the manifest (R2: OSL 600 inputs, B: 500).
    sizes = {
        name: {
            "observation": int(bundle.controller(name).observation["size"]),
            "action": int(bundle.controller(name).action["size"]),
        }
        for name in ("human", "osl")
    }
    # The OSL controller's sensor provider, as the runner built it.
    observer = next(slot.observer for slot in sim.slots if slot.name == "osl")
    if not isinstance(observer, SensorObservation) or observer.randomization is not None:
        raise ValueError("Expected the OSL slot's SensorObservation without randomization")
    buses = list(observer.buses)
    frame_width = sum(int(t.frame_dim) for t in buses)
    socket = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SOCKET_BODY)
    if socket < 0:
        raise ValueError(f"The bundle model has no body {SOCKET_BODY}")
    root = int(m.body_rootid[socket])

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
        "osl_frame": np.zeros((rows, frame_width)),
        "socket_cfrc": np.zeros((rows, 6)),
        "root_subtree_com": np.zeros((rows, 3)),
        "socket_xpos": np.zeros((rows, 3)),
        "requested": np.zeros((n, 3)),
        "human_obs": np.zeros((n, sizes["human"]["observation"]), np.float32),
        "osl_obs": np.zeros((n, sizes["osl"]["observation"]), np.float32),
        "human_mean": np.zeros((n, sizes["human"]["action"]), np.float32),
        "osl_mean": np.zeros((n, sizes["osl"]["action"]), np.float32),
        "bus": np.zeros((n, m.nu)),
        "site_error": np.zeros(n),
        "switched": np.zeros(n, np.uint8),
        "failed": np.zeros(n, np.uint8),
    }

    def boundary(b, observed=True):
        out["qpos"][b], out["qvel"][b] = d.qpos, d.qvel
        out["act"][b], out["ctrl"][b] = d.act, d.ctrl
        out["qacc_warmstart"][b], out["time"][b] = d.qacc_warmstart, d.time
        out["since_anchor"][b] = sim._since_anchor
        ref = sim.reference
        for key in (*STATE_INT, *STATE_FLOAT, *STATE_VEC):
            out[f"ref_{key}"][b] = getattr(ref, key)
        if not observed:  # after a failed step (see the docstring): NaN, whatever the failure
            for key in EXTRAS:
                out[key][b] = np.nan
            return
        out["osl_frame"][b] = np.concatenate([observer.frame[t.name] for t in buses])
        out["socket_cfrc"][b] = d.cfrc_int[socket]
        out["root_subtree_com"][b] = d.subtree_com[root]
        out["socket_xpos"][b] = d.xpos[socket]

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
        boundary(k + 1, observed=not r["failed"])
        if r["failed"]:
            print(f"[trace] failed at step {k}: {sim.failure_reason}")
            steps = k + 1
            break
    elapsed = time.perf_counter() - started
    if steps < n:  # Keep what ran.
        for key, value in out.items():
            out[key] = value[: steps + (len(value) - n)]

    # Self-check: the recorded boundary state alone reproduces, after
    # mj_forward, the native boundary observation's state-derived part
    # (everything before the reference: joints, muscles, touch, tracked-body
    # kinematics) and every boundary extra, so a replay needs nothing else.
    from osl_deploy.observations import FullBodyObservation

    observer_replay = FullBodyObservation(bundle)
    measured = observer_replay.layout.groups["reference"].start
    frames = SensorObservation(bundle, bundle.controller("osl"))  # its read() only
    replay = mujoco.MjData(m)
    worst, worst_extra = 0.0, {key: 0.0 for key in EXTRAS}
    for b in range(0, steps, max(1, steps // 40)):
        mujoco.mj_resetData(m, replay)
        replay.qpos[:], replay.qvel[:] = out["qpos"][b], out["qvel"][b]
        replay.act[:], replay.ctrl[:] = out["act"][b], out["ctrl"][b]
        replay.qacc_warmstart[:], replay.time = out["qacc_warmstart"][b], out["time"][b]
        mujoco.mj_forward(m, replay)
        state = observer_replay(replay, sim.reference)[:measured]
        worst = max(worst, float(np.abs(state - out["human_obs"][b][:measured]).max()))
        frame = frames.read(replay)
        mine = {
            "osl_frame": np.concatenate([frame[t.name] for t in frames.buses]),
            "socket_cfrc": replay.cfrc_int[socket],
            "root_subtree_com": replay.subtree_com[root],
            "socket_xpos": replay.xpos[socket],
        }
        for key, value in mine.items():
            worst_extra[key] = max(worst_extra[key], float(np.abs(value - out[key][b]).max()))
    # The newest frame a term with lag L shows at step k is the one read at
    # boundary max(k - L, 0): the extras and the recorded inputs must agree.
    terms, at, lag_worst = [], 0, 0.0
    for t in buses:
        lag = int(observer.lags.get(t.name, 0))
        newest = int(t.span.start) + (int(t.history) - 1) * int(t.frame_dim)
        for k in range(steps):
            fed = out["osl_obs"][k][newest : newest + int(t.frame_dim)]
            read = out["osl_frame"][max(k - lag, 0)][at : at + int(t.frame_dim)].astype(np.float32)
            lag_worst = max(lag_worst, float(np.abs(fed - read).max()))
        terms.append(
            {
                "name": t.name,
                "frame_dim": int(t.frame_dim),
                "start": at,  # in osl_frame
                "offset": int(t.span.start),  # in the OSL observation
                "history": int(t.history),
                "lag": lag,
            }
        )
        at += int(t.frame_dim)
    if lag_worst != 0 or max(worst_extra.values()) > 1e-9:
        raise AssertionError(
            f"trace extras do not replay: lagged frames vs osl_obs {lag_worst}, {worst_extra}"
        )
    meta = {
        "format": TRACE_FORMAT,
        "bundle": bundle_label(bundle_dir, bundle.manifest),
        "manifest_sha256": sha,
        "steps": int(steps),
        "dt": bundle.dt,
        "sizes": sizes,
        "schedule": [
            {"seconds": s, "action": a, "twist": ui_twist(twists, level, a).tolist()}
            for s, a in schedule
        ],
        "level": level,
        "clip_names": bundle.manifest["reference"]["clip_names"],
        "settings": "osl_deploy runner.Simulation defaults (lockstep, delay train, "
        "sensor lag validation, mean inference, rebase 5 s, idle frame 0)",
        "extras": {
            "where": "each control boundary, where the runner reads the OSL sensors: after the "
            "boundary's mj_forward (b = 0: the reset's); NaN at a failed step's boundary",
            "osl_frame": {"width": frame_width, "terms": terms},
            "socket_cfrc": {
                "body": SOCKET_BODY,
                "body_id": int(socket),
                "convention": "MuJoCo cfrc_int [torque, force], world axes, torque about "
                "subtree_com[body_rootid]: the wrench the parent body applies to the subtree",
            },
            "root_subtree_com": {"body_id": root, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root)},
            "socket_xpos": {"body_id": int(socket)},
        },
        "native_replay_max_abs": worst,
        "extras_replay_max_abs": worst_extra,
        "lagged_frames_vs_osl_obs_max_abs": lag_worst,
        "seconds_wall": elapsed,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stats = write_container(out_path, out, meta)
    side = sidecar(out_path)
    side.write_text(json.dumps({"file": Path(out_path).name, "bytes": stats["bytes"], **meta}, indent=1) + "\n")
    failed = int(out["failed"].any())
    print(
        f"[trace] {meta['bundle']} (manifest {sha[:12]}): {steps} steps "
        f"({steps * bundle.dt:.2f} s sim, {elapsed:.1f} s wall), "
        f"failed={failed}, switches={int(out['switched'].sum())}, "
        f"native replay max |obs diff| {worst:.3g}, extras {max(worst_extra.values()):.3g}, "
        f"{out_path} {stats['bytes'] / 1e6:.1f} MB, {side.name}"
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
        "bundle": bundle_label(bundle_dir, bundle.manifest),
        "manifest_sha256": manifest_sha256(bundle_dir),
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"[robust] {summary['sim_seconds']} s sim in {summary['wall_seconds']} s wall, "
        f"{len(segments)} segments, {summary['distinct_states']} states, "
        f"{len(falls)} falls, max peak site error {summary['max_peak_site_error']}"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="duet_export.trace", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.robust:
        name = "trace_robust.json" if args.seed == 0 else f"trace_robust_seed{args.seed}.json"
        robust(args.bundle, args.out or TRACES / name, args.seconds, args.seed)
    else:
        record_trace(args.bundle, args.out or TRACES / TRACE_FILE)


if __name__ == "__main__":
    main()
