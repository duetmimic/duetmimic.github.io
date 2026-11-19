"""Export section `sim`: everything SimCore needs, plus reference.bin.gz.

    cd tools && python -m duet_export --out ../demo/assets/stage2 --only sim

SimCore (demo/js/sim/) is the browser port of osl_deploy's synchronous
runner.Simulation for a stage-2 bundle at its default settings: lockstep
controllers without stress, the training actuator delay ("train", the middle of
each block's range), the OSL sensor lag "validation", "mean" inference, a
reference rebase every 5 s, and an episode that starts at the idle clip, frame 0.
Every id, index and constant below is read from osl_deploy's own Bundle and the
manifest, never re-derived, so the JS side only has to follow them.

demo.json keys written here:

- `sim`: model sizes, the observation ids (root, joints, tracked bodies,
  muscles, foot touch sensors), each controller's observation layout and
  action blocks (actuator ids, affine map and clip, delay in physics ticks), the
  OSL sensor terms (sensordata addresses, saturation, tare, bus lag, history),
  the previous-command term, the failure threshold, the rebase interval, the
  reset clip, and the foot touch zones (`touch_sites`: the manifest's version
  and boxes, which osl_deploy's Bundle checked against model.mjb, with their
  site, sensor and body ids and sensordata addresses). SimCore does not read
  `touch_sites`; tools/test/contract.test.mjs checks physics.mjb.gz against it,
  because a touch sensor reads only contacts inside its zone and the zones are
  the one thing that tells two bundles' models apart (R2's heel zone misses
  the heel: r_foot reads 0 N there).
  For the page's sensing and X-ray views (render/frame.js extras; SimCore
  reads none of these): each OSL term's `channels` and `units` (the manifest's
  term_layout) and each read's `anchor` (the site the sensor reads, or its
  joint, with the body it moves with), `osl.sensor_set` (the manifest's), and
  `socket` (the prosthesis socket body, its parent and tree root, the mass
  below it, and whether MuJoCo refreshes cfrc_int in every mj_forward and
  mj_step: the frames' socket wrench reads it).
- `reference`: the gait-library configuration (blend, lookahead, twist metric,
  the command support of the WHOLE library) and the table of the clips the UI
  can reach ({name, index (library), twist, num_frames, stride, start, idle}),
  with `file` = reference.bin.gz.
- `ui`: the WASD design. Each speed level maps every key state to one twist
  that EXACTLY equals a library clip label (float32 values written as doubles),
  so SimCore's nearest-clip selection is exact.

reference.bin.gz (DUETBIN1) holds only the reachable clips, frames concatenated
clip after clip (row `start + frame`), and only the arrays SimCore reads:

    root_pos       float32 [F, 3]       world, clip coordinates
    root_quat      float32 [F, 4]       wxyz
    root_lin_vel   float32 [F, 3]       world
    root_ang_vel_w float32 [F, 3]       world
    rpos           float32 [F, 16, 3]   tracked sites minus the pelvis, world axes
    reset_joint_pos, reset_joint_vel float32 [77]  the reset frame's joints
"""

import os
from pathlib import Path

import numpy as np

from .common import load_bundle
from .container import write_container

FORMAT = "duetmimic-sim/1"
REFERENCE_FORMAT = "duetmimic-reference/1"
REFERENCE_FILE = "reference.bin.gz"
RESET_CLIP, RESET_FRAME = "idle", 0
FRAME_FIELDS = ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel_w", "rpos")
# The prosthesis socket body (the manifest names only its joints, robot/socket_*):
# the OSL hangs from it, and its cfrc_int is the whole load between the residual
# limb and the prosthesis (duet_export.trace records the same body).
SOCKET_BODY = "robot/OSL_KA_L1_osl_generic_socket"
WRENCH_CONVENTION = (
    "frame.wrench (demo/js/render/frame.js) = [Fx, Fy, Fz, Tx, Ty, Tz], N and N m, MuJoCo world "
    "axes: the load the prosthesis applies to the residual limb at this body's origin xpos[body]: "
    "F = -cfrc_int[body][3:6], T = -(cfrc_int[body][0:3] + (subtree_com[root] - xpos[body]) x "
    "cfrc_int[body][3:6]); cfrc_int[body] is the wrench the parent applies to the body's subtree, "
    "[torque about subtree_com[root]; force]"
)


def write_container_atomic(path, arrays, meta=None):
    """write_container through a temporary file and a rename: whoever reads the
    asset meanwhile (the demo, a benchmark) gets the old file or the new one,
    never a partial one."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        stats = write_container(tmp, arrays, meta)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return stats


def load_bundle_once(bundle_dir, ctx):
    """osl_deploy's Bundle for this export run, loaded once for every section that
    needs it (sim, visual) and kept in the run's ctx."""
    bundle = ctx.get("bundle") if ctx is not None else None
    if bundle is None:
        bundle = load_bundle(bundle_dir)
        if ctx is not None:
            ctx["bundle"] = bundle
    return bundle


# ---------------------------------------------------------------------------
# WASD design. Twist (vx m/s, vy m/s, wz rad/s), wz > 0 turns left (CCW seen
# from above). Walking forward or turning in place, A/D turn the body left/
# right. Walking backward, A/D steer like a car in reverse (user request,
# 2026-11-16; the first, tank-style table felt flipped): S+A backs up along a
# path that curves to the walker's LEFT, so the body yaws right (wz < 0); seen
# from the chase camera behind the walker it comes toward the camera and moves
# to the left of the screen. Opposite keys cancel (W+S: no translation, A+D:
# no turn).
#
# The values are decimal labels of library clips; ui_section() replaces each by
# the clip's exact float32 label. The forward grid has +-0.5 rad/s arcs at every
# speed (tighter ones exist only for 0.6-1.2 m/s); backward arcs stay gentle;
# turning in place uses 1.0 rad/s both ways (the R2 policy was weakest on
# tight right turns, toward the prosthesis side, from about 1.2 rad/s).
#
# Capped for robustness at a 1.2 m/s top speed, -0.7 m/s backward and
# +-0.25 rad/s backward arcs (the library goes to 1.5 / -0.8 / +-0.5). The caps
# and the findings in this comment were measured on the R2 bundle
# (stage2_R2_model_6898, 2026-11-16); the B bundle the demo now runs
# (2026-11-18) reaches the same 27 clips with identical labels and has not been
# re-measured. The caps were first set for the int8 human weights; the
# closed-loop quantization study (policy.py) later found int8 falls exactly as
# often as fp32, so the limit is the policy itself: fp32 falls on 1 of 8 entry
# phases reversing from -0.8 to 1.4 m/s, and walking forward out of a backward
# walk is the weakest transition in the 2026-11-16 key-masher benchmark
# (demo/js/session.js holds it back until the body is at rest). Restoring
# 1.4 / -0.8 / +-0.3 would make exactly that transition harsher, so the caps
# stay.
# Robustness: python -m duet_export.trace --robust [--seed N] (writes
# tools/traces/trace_robust[_seedN].json).
# ---------------------------------------------------------------------------
UI_LEVELS = (
    # forward vx, forward arc |wz|, backward vx, backward arc |wz|, turn |wz|
    {"name": "1", "forward": 0.6, "arc": 0.5, "backward": -0.5, "back_arc": 0.25, "turn": 1.0},
    {"name": "2", "forward": 0.8, "arc": 0.5, "backward": -0.6, "back_arc": 0.25, "turn": 1.0},
    {"name": "3", "forward": 1.0, "arc": 0.5, "backward": -0.6, "back_arc": 0.25, "turn": 1.0},
    {"name": "4", "forward": 1.1, "arc": 0.5, "backward": -0.7, "back_arc": 0.25, "turn": 1.0},
    {"name": "5", "forward": 1.2, "arc": 0.4, "backward": -0.7, "back_arc": 0.25, "turn": 1.0},
)
DEFAULT_LEVEL = 2  # 0-based index into UI_LEVELS: level "3", 1.0 m/s
FAST_LEVEL = 4  # Shift held: level "5"
# KeyboardEvent.code values (layout-independent physical keys).
BINDINGS = {
    "forward": ["KeyW", "ArrowUp"],
    "backward": ["KeyS", "ArrowDown"],
    "left": ["KeyA", "ArrowLeft"],
    "right": ["KeyD", "ArrowRight"],
    "fast": ["ShiftLeft", "ShiftRight"],
    "slower": ["KeyQ"],
    "faster": ["KeyE"],
    "levels": ["Digit1", "Digit2", "Digit3", "Digit4", "Digit5"],
}
# (direction, turn) -> action; direction +1 forward / -1 backward, turn +1 left.
ACTIONS = {
    (0, 0): "idle",
    (1, 0): "forward",
    (1, 1): "forward_left",
    (1, -1): "forward_right",
    (-1, 0): "backward",
    (-1, 1): "backward_left",
    (-1, -1): "backward_right",
    (0, 1): "turn_left",
    (0, -1): "turn_right",
}
ACTION_KEYS = {
    "idle": "",
    "forward": "W",
    "forward_left": "W+A",
    "forward_right": "W+D",
    "backward": "S",
    "backward_left": "S+A",
    "backward_right": "S+D",
    "turn_left": "A",
    "turn_right": "D",
}
ACTION_LABELS = {
    "idle": "Stand",
    "forward": "Forward",
    "forward_left": "Forward, left",
    "forward_right": "Forward, right",
    "backward": "Backward",
    "backward_left": "Backward, curving left",
    "backward_right": "Backward, curving right",
    "turn_left": "Turn left",
    "turn_right": "Turn right",
}


def decimal_twist(level, action):
    """The decimal (vx, vy, wz) a level gives an action."""
    if action == "idle":
        return (0.0, 0.0, 0.0)
    direction, turn = next(key for key, name in ACTIONS.items() if name == action)
    if direction > 0:
        return (level["forward"], 0.0, turn * level["arc"])
    if direction < 0:
        # Car in reverse: the path curves toward the key's side, the body yaws
        # the other way.
        return (level["backward"], 0.0, -turn * level["back_arc"])
    return (0.0, 0.0, turn * level["turn"])


def clip_of(twists, twist):
    """Index of the one library clip whose label is `twist` (decimal)."""
    hits = np.flatnonzero(np.all(np.abs(twists - np.asarray(twist)) < 1e-6, axis=1))
    if len(hits) != 1:
        raise ValueError(f"No unique library clip is labelled {twist}")
    return int(hits[0])


def twist_label(twist):
    vx, _, wz = twist
    parts = []
    if abs(vx) > 1e-9:
        parts.append(f"{vx:+.2f} m/s".replace("+", ""))
    if abs(wz) > 1e-9:
        parts.append(f"{wz:+.2f} rad/s".replace("+", ""))
    return ", ".join(parts) or "0"


def ui_section(twists):
    """(ui dict, [library clip indices the UI reaches]) for the library twists."""
    reach, levels = [], []
    for level in UI_LEVELS:
        actions = {}
        for action in ACTIONS.values():
            clip = clip_of(twists, decimal_twist(level, action))
            exact = [float(x) for x in twists[clip]]
            actions[action] = {
                "twist": exact,
                "keys": ACTION_KEYS[action],
                "label": ACTION_LABELS[action],
                "speed": twist_label(decimal_twist(level, action)),
            }
            if clip not in reach:
                reach.append(clip)
        levels.append(
            {
                "name": level["name"],
                "label": f"{level['forward']:.1f} m/s",
                "actions": actions,
            }
        )
    ui = {
        "format": "duetmimic-ui/1",
        "bindings": BINDINGS,
        "actions": {
            name: {"keys": ACTION_KEYS[name], "label": ACTION_LABELS[name]}
            for name in ACTIONS.values()
        },
        "levels": levels,
        "default_level": DEFAULT_LEVEL,
        "fast_level": FAST_LEVEL,
        "notes": (
            "twistForKeys(keys, ui, level) in demo/js/sim/commands.js: keys is a Set "
            "of held KeyboardEvent.code values (single letters such as 'w' also "
            "work). W/S walk forward/backward at the selected level's speed. A/D "
            "turn left/right: with W the body turns that way; with S they steer "
            "like a car in reverse, so S+A backs along a path curving to the "
            "walker's left while the body yaws right (wz < 0). A or D alone "
            "turns in place, opposite keys cancel, no key stands. Digit1-5 pick a "
            "level, Q/E step it down/up, Shift held uses the fast level. "
            "levels[*].actions[*].twist equals a library clip label exactly "
            "(twist = [vx m/s, vy m/s, wz rad/s], wz > 0 turns left)."
        ),
    }
    return ui, sorted(reach)


def reference_section(bundle, reach, out_dir):
    """(reference dict, container stats) for the reachable clips."""
    arrays = bundle.reference_arrays()
    m = bundle.manifest["reference"]
    names = m["clip_names"]
    twists, lengths = arrays["twists"], arrays["num_frames"]
    strides = arrays["strides"]
    from osl_deploy.commands import CommandSupport, idle_clips

    support = CommandSupport(twists, m["twist_metric_weights"], bundle.manifest.get("command"))
    idle = idle_clips(twists)
    reset = names.index(RESET_CLIP)
    if reset not in reach:
        reach = sorted([*reach, reset])
    clips, chunks, start = [], {key: [] for key in FRAME_FIELDS}, 0
    for index in reach:
        n = int(lengths[index])
        clips.append(
            {
                "name": names[index],
                "index": index,
                "twist": [float(x) for x in twists[index]],
                "num_frames": n,
                "stride": [float(x) for x in strides[index]],
                "start": start,
                "idle": bool(idle[index]),
            }
        )
        for key in FRAME_FIELDS:
            chunks[key].append(np.asarray(arrays[key][index, :n], dtype=np.float32))
        start += n
    data = {key: np.concatenate(chunks[key]) for key in FRAME_FIELDS}
    data["reset_joint_pos"] = np.asarray(arrays["joint_pos"][reset, RESET_FRAME], np.float32)
    data["reset_joint_vel"] = np.asarray(arrays["joint_vel"][reset, RESET_FRAME], np.float32)
    # Self-check: every row is the library's, bit for bit.
    for clip in clips:
        rows = slice(clip["start"], clip["start"] + clip["num_frames"])
        for key in FRAME_FIELDS:
            library = arrays[key][clip["index"], : clip["num_frames"]]
            if not np.array_equal(data[key][rows], library):
                raise AssertionError(f"reference subset differs for {clip['name']} {key}")
    meta = {
        "format": REFERENCE_FORMAT,
        "clips": [clip["name"] for clip in clips],
        "start": [clip["start"] for clip in clips],
        "num_frames": [clip["num_frames"] for clip in clips],
        "frames": int(start),
    }
    stats = write_container_atomic(out_dir / REFERENCE_FILE, data, meta)
    reset_subset = next(i for i, clip in enumerate(clips) if clip["index"] == reset)
    section = {
        "format": REFERENCE_FORMAT,
        "file": REFERENCE_FILE,
        "bytes": stats["bytes"],
        "raw_bytes": stats["raw_bytes"],
        "policy_dt": float(m["policy_dt"]),
        "blend_steps": max(1, round(m["blend_time_s"] / m["policy_dt"])),
        "lookahead_steps": int(m["lookahead_steps"]),
        "lookahead_stride": int(m["lookahead_stride"]),
        "metric": [float(x) for x in m["twist_metric_weights"]],
        # Training's command support over the WHOLE library (CommandSupport):
        # serve() clamps a request into [low, high] per axis.
        "command": {
            "low": [float(x) for x in support.low],
            "high": [float(x) for x in support.high],
        },
        "sites": int(arrays["rpos"].shape[2]),
        "library_clips": len(names),
        "clips": clips,
        "reset": {"clip": reset_subset, "name": RESET_CLIP, "frame": RESET_FRAME},
        "arrays": {key: list(value.shape) for key, value in data.items()},
    }
    return section, stats


def touch_sites_section(bundle, touch_adr, osl_terms):
    """The foot touch zones of model.mjb: the manifest's declaration (osl_deploy's
    Bundle refused the bundle unless model.mjb matches it to 1e-9; a manifest
    without `touch_sites` gets the "original" table, undeclared), with each
    zone's site, sensor and body ids and sensordata address in the model.

    Every touch sensor a policy reads (the human `touch` term, the OSL reads of
    mjSENS_TOUCH sensors) must be one of these zones.
    """
    import mujoco

    m, declared = bundle.model, bundle.touch_sites
    obj = mujoco.mjtObj
    sites = []
    for site in declared.sites:
        ids = {
            "site_id": mujoco.mj_name2id(m, obj.mjOBJ_SITE, site.name),
            "sensor_id": mujoco.mj_name2id(m, obj.mjOBJ_SENSOR, site.sensor),
            "body_id": mujoco.mj_name2id(m, obj.mjOBJ_BODY, site.body),
        }
        if min(ids.values()) < 0:
            raise ValueError(f"touch site {site.name}: missing from the model ({ids})")
        sensor = ids["sensor_id"]
        if (
            m.sensor_type[sensor] != mujoco.mjtSensor.mjSENS_TOUCH
            or m.sensor_objtype[sensor] != obj.mjOBJ_SITE
            or m.sensor_objid[sensor] != ids["site_id"]
            or m.site_bodyid[ids["site_id"]] != ids["body_id"]
        ):
            raise ValueError(f"touch site {site.name}: {site.sensor} is not its touch sensor")
        sites.append(
            {
                "site": site.name,
                "sensor": site.sensor,
                "body": site.body,
                **{key: int(value) for key, value in ids.items()},
                "adr": int(m.sensor_adr[sensor]),
                "type": site.type,
                "pos": [float(x) for x in site.pos],
                "quat": [float(x) for x in site.quat],
                "size": [float(x) for x in site.size],
            }
        )
    zones = {site["adr"] for site in sites}
    touch_reads = [
        read["adr"]
        for term in osl_terms
        for read in term["reads"]
        if m.sensor_type[int(np.flatnonzero(m.sensor_adr == read["adr"])[0])]
        == mujoco.mjtSensor.mjSENS_TOUCH
    ]
    stray = sorted((set(touch_adr) | set(touch_reads)) - zones)
    if stray:
        raise ValueError(f"touch reads at sensordata {stray} are not declared touch zones")
    return {
        "version": declared.version,
        "declared": bool(declared.declared),
        "frame": declared.frame,
        "sites": sites,
    }


def sensor_anchor(model, sensor):
    """Where an OSL sensor sits, for display: {kind: "site", id, body, name} for a
    sensor of a site (IMU, load cell, touch zone), {kind: "joint", ...} for one of
    a joint (an encoder; drawn at the joint's anchor, xanchor), with `body` the
    body the site or joint moves with."""
    import mujoco

    obj = mujoco.mjtObj
    objtype, objid = int(model.sensor_objtype[sensor]), int(model.sensor_objid[sensor])
    if objtype == obj.mjOBJ_SITE:
        kind, body = "site", int(model.site_bodyid[objid])
    elif objtype == obj.mjOBJ_JOINT:
        kind, body = "joint", int(model.jnt_bodyid[objid])
    else:
        name = mujoco.mj_id2name(model, obj.mjOBJ_SENSOR, sensor)
        raise ValueError(f"OSL sensor {name} reads a {obj(objtype).name}, not a site or a joint")
    return {"kind": kind, "id": objid, "body": body, "name": mujoco.mj_id2name(model, objtype, objid)}


def osl_bus_terms(bundle, sensors=None):
    """The OSL controller's bus terms (its sensor terms, in observation order) as
    demo.json lists them under sim.osl.observation.terms: offset, frame size,
    history and validation lag; the term's channel names and units (the
    manifest's term_layout, one per frame value); per read (osl_deploy's
    SensorRead) its sensordata slice, component selection, frame slot,
    saturation and tare, and its display `anchor` (sensor_anchor). The frames'
    `sensors` (render/frame.js) are these terms' frames, concatenated."""
    import mujoco
    from osl_deploy.sensor_observation import SensorObservation

    spec = bundle.controller("osl")
    if sensors is None:
        sensors = SensorObservation(bundle, spec)  # default: validation lag, no noise
    m = bundle.model
    layout = {entry["name"]: entry for entry in spec.observation.get("term_layout", [])}
    terms = []
    for term in sensors.terms:
        if not term.reads:
            continue
        entry = layout.get(term.name, {})
        labels = {key: entry.get(key) for key in ("channels", "units")}
        for key, values in labels.items():
            if values is not None and len(values) != int(term.frame_dim):
                raise ValueError(
                    f"OSL term {term.name}: {len(values)} {key} for a frame of {term.frame_dim} values"
                )
        reads = []
        for read in term.reads:
            sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, read.scene_name)
            if sensor < 0 or int(m.sensor_adr[sensor]) != int(read.data.start):
                raise ValueError(f"OSL read {read.name}: sensor {read.scene_name} is not at sensordata {read.data.start}")
            reads.append(
                {
                    "name": read.name,
                    "adr": int(read.data.start),
                    "size": int(read.size),
                    "components": None if read.components is None else [int(i) for i in read.components],
                    "frame": int(read.frame.start),
                    "saturation": read.saturation,
                    "tare": None if read.tare is None else [float(x) for x in read.tare],
                    "anchor": sensor_anchor(m, sensor),
                }
            )
        terms.append(
            {
                "name": term.name,
                "offset": int(term.offset),
                "frame_dim": int(term.frame_dim),
                "history": int(term.history),
                "lag": int(sensors.lags.get(term.name, 0)),
                "channels": None if labels["channels"] is None else [str(c) for c in labels["channels"]],
                "units": None if labels["units"] is None else [str(u) for u in labels["units"]],
                "reads": reads,
            }
        )
    return terms


def cfrc_int_fresh(model):
    """Whether MuJoCo recomputes every body's cfrc_int in each mj_forward and
    mj_step of this model: mj_sensorAcc runs mj_rnePostConstraint for an
    acceleration-stage sensor that needs it (force, torque, accelerometer), so
    the browser can read cfrc_int at any frame without calling it (which after
    an mj_step would use the integrated velocities)."""
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for step in (mujoco.mj_forward, mujoco.mj_step):
        data.cfrc_int[1:] = np.nan
        step(model, data)
        if not np.isfinite(data.cfrc_int[1:]).all():
            return False
    return True


def socket_section(model):
    """The prosthesis socket body: its id, parent (the residual femur) and kinematic
    tree root (cfrc_int's torque reference point is that root's subtree_com), the
    mass of its subtree (the whole prosthesis), whether cfrc_int is fresh in
    every frame, and the frames' wrench convention (render/frame.js)."""
    import mujoco

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SOCKET_BODY)
    if body < 0:
        raise ValueError(f"The bundle model has no body {SOCKET_BODY}")
    parent = int(model.body_parentid[body])
    return {
        "body": int(body),
        "name": SOCKET_BODY,
        "parent": parent,
        "parent_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
        "root": int(model.body_rootid[body]),
        "subtree_mass": round(float(model.body_subtreemass[body]), 9),
        "cfrc_int_fresh": cfrc_int_fresh(model),
        "wrench": WRENCH_CONVENTION,
    }


def sim_section(bundle):
    """SimCore's ids, maps and constants, from osl_deploy's own objects."""
    import mujoco
    from osl_deploy.actuator_delay import resolve_delays
    from osl_deploy.observations import observation_layout
    from osl_deploy.policy import ActionAdapter, check_coverage
    from osl_deploy.runner import (
        DEFAULT_REBASE_SECONDS,
        VALIDATION_SITE_THRESHOLD,
        rebase_steps,
    )
    from osl_deploy.sensor_observation import SensorObservation, previous_command

    b, m = bundle, bundle.model
    names = b.names
    if names != ["human", "osl"]:
        raise ValueError(f"Expected the controllers ['human', 'osl'], got {names}")
    timestep = float(m.opt.timestep)
    delays = resolve_delays("train", b.manifest, timestep)  # mid training range
    adapters = {name: ActionAdapter(b, None, controller=name) for name in names}
    check_coverage(m.nu, list(adapters.values()))

    def action(name):
        adapter = adapters[name]
        if adapter.mode != "mean":
            raise ValueError(f"{name}: default inference is {adapter.mode}, not mean")
        return {
            "size": int(adapter.size),
            "mode": adapter.mode,
            "blocks": [
                {
                    "name": block.name,
                    "start": int(block.span.start),
                    "size": int(block.span.stop - block.span.start),
                    "actuator_ids": [int(i) for i in block.ids],
                    "scale": [float(x) for x in block.scale],
                    "offset": [float(x) for x in block.offset],
                    "low": [float(x) for x in block.low],
                    "high": [float(x) for x in block.high],
                    "delay_ticks": int(delays[block.name]),
                }
                for block in adapter.blocks
            ],
        }

    human_layout = observation_layout(b)
    osl_spec = b.controller("osl")
    sensors = SensorObservation(b, osl_spec)  # default: validation lag, no noise
    span, history, width = previous_command(osl_spec)
    terms = osl_bus_terms(b, sensors)
    # A reset holds position servos at the start pose (actuator_delay.hold_ctrl);
    # every other actuator (muscles, torque motors) holds 0. None here.
    servos = []
    for i in range(m.nu):
        if (
            m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
            and m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
            and m.actuator_biasprm[i, 1] < 0
            and m.jnt_type[m.actuator_trnid[i, 0]]
            in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ):
            servos.append([i, int(m.jnt_qposadr[m.actuator_trnid[i, 0]])])
    return {
        "format": FORMAT,
        "model": {
            "nq": int(m.nq),
            "nv": int(m.nv),
            "nu": int(m.nu),
            "na": int(m.na),
            "nbody": int(m.nbody),
            "nsensordata": int(m.nsensordata),
            "timestep": timestep,
            # osl_deploy's Bundle disables MuJoCo's auto-reset on bad states.
            "disable_autoreset": int(mujoco.mjtDisableBit.mjDSBL_AUTORESET),
        },
        "decimation": int(b.decimation),
        "policy_dt": float(b.dt),
        "root_qpos": int(b.root_qpos),
        "root_qvel": int(b.root_qvel),
        "qpos_ids": [int(i) for i in b.qpos_ids],
        "qvel_ids": [int(i) for i in b.qvel_ids],
        # Tracked bodies, the anchor (pelvis) first, and their kinematic-tree
        # roots (mj_objectVelocity's reference point is the root's subtree COM).
        "body_ids": [int(i) for i in b.body_ids],
        "body_rootids": [int(m.body_rootid[i]) for i in b.body_ids],
        "muscle_ids": [int(i) for i in b.muscle_ids],
        "activation_ids": [int(i) for i in b.activation_ids],
        "touch_adr": [int(i) for i in b.touch_ids],
        "touch_sites": touch_sites_section(b, [int(i) for i in b.touch_ids], terms),
        "socket": socket_section(m),
        "human": {
            "observation": {
                "schema": b.full_body.schema,
                "size": int(human_layout.size),
                "groups": {
                    name: [int(s.start), int(s.stop)]
                    for name, s in human_layout.groups.items()
                },
            },
            "action": action("human"),
        },
        "osl": {
            "observation": {
                "schema": osl_spec.schema,
                "size": int(sensors.layout.size),
                "terms": terms,
                "previous_command": {
                    "offset": int(span.start),
                    "history": int(history),
                    "width": int(width),
                    "low": -1.0,
                    "high": 1.0,
                },
            },
            "action": action("osl"),
            # Which onboard sensors this bundle's OSL policy reads (the manifest's).
            "sensor_set": osl_spec.observation.get("sensor_set"),
        },
        "hold_servos": servos,
        "failure_site_threshold": float(VALIDATION_SITE_THRESHOLD),
        "rebase_steps": int(rebase_steps(DEFAULT_REBASE_SECONDS, b.dt)),
        "rebase_seconds": float(DEFAULT_REBASE_SECONDS),
        "settings": {
            "actuator_delay": "train",
            "actuator_delay_ticks": {k: int(v) for k, v in delays.items()},
            "sensor_lag": "validation",
            "inference": "mean",
            "controllers": "lockstep, no stress",
        },
    }


def export(bundle_dir, out_dir, ctx):
    bundle = load_bundle_once(bundle_dir, ctx)
    twists = bundle.reference_arrays()["twists"]
    ui, reach = ui_section(twists)
    reference, stats = reference_section(bundle, reach, out_dir)
    sim = sim_section(bundle)
    print(
        f"[duet_export] sim: {len(reference['clips'])} clips, "
        f"{reference['arrays']['root_pos'][0]} frames, {REFERENCE_FILE} "
        f"{stats['bytes'] / 1e6:.2f} MB ({stats['raw_bytes'] / 1e6:.2f} MB raw)"
    )
    return {"sim": sim, "reference": reference, "ui": ui}
