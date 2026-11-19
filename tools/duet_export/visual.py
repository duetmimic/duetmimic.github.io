"""Render assets for the browser demo: decimated meshes and the paper-figure look.

    cd tools
    python -m duet_export --out ../demo/assets/stage2 --only visual
    python -m duet_export.visual poses traces/render_poses.json   # test fixture

Writes `visual.bin.gz` (DUETBIN1) and returns demo.json's "visual" section.

What is drawn follows osl_deploy's figure tool (figure/scene.py, "paper" preset,
examples/figure/three_demos.toml): the robot geoms in MuJoCo's default visible
groups 0-2 (the floor plane is the page's own checker floor), the OSL knee and
ankle assembly meshes recoloured orange #FF6C0C, and every spatial tendon drawn
the way mjv_addGeoms draws it: a capsule between consecutive wrap points, none
across a pulley, radius tendon_width, half that between two points on a wrapping
geom. The figure replays kinematics, so its muscles all show tendon_rgba red;
here they are coloured by activation along a red ramp (muscles.colormap).

Meshes come from the ORIGINAL bundle model: the slim physics model keeps every
id but strips mesh data. Each drawn mesh is welded, simplified with
meshoptimizer (run in Node from tools/node_modules) to an absolute error, given
crease-angle normals, and stored as int16 positions (per-mesh box), int8
normals and per-mesh uint16 indices.

Coordinates stay MuJoCo's: z up, metres. A geom is drawn at data.geom_xpos /
geom_xmat (row-major); a mesh's vertices are in its geom frame, as MuJoCo's own
renderer uses them.

demo.json "visual.overlays" (computed from the model at qpos0, nothing in
visual.bin.gz) serves the page's X-ray and sensing views; every point is fixed
in the frame of a drawn geom on the same rigid body, so it rides the drawn,
interpolated pose (world = geom_xpos + geom_xmat . local). A `geom` is an index
into geom_ids (a row of the frame's geoms), `geom_id` the MuJoCo geom id:
  roles    per drawn geom: "socket" (on the prosthesis socket body,
           simcfg.SOCKET_BODY), "cover" (the OSL knee and ankle housings, the
           orange covers), "hardware" (the rest of the prosthesis: the socket
           body's subtree), "bone" (everything else: the human);
  socket   {geom, geom_id, body, local}: the socket body's origin, the point
           the frames' wrench acts at (render/frame.js);
  markers  one per OSL sensor (the reads of the bundle's OSL observation that
           share a site or joint and a kind): {key, label, kind (imu |
           load_cell | contact | encoder), geom, geom_id, local, channels
           (indices into the frame's sensors), anchor {kind: site | joint, id,
           name, body}; a joint's also `axis`, its axis in the geom frame}.
           `local` is the sensor's site, or its joint's anchor, except for a
           touch sensor: its zone is a box around the foot's collision
           capsules, whose centre can lie off the foot (B's toe zone runs
           2-3 cm past the toe), so its glyph goes where the zone meets the
           foot (contact_point: the point of a collision geom's axis inside
           the zone nearest its centre; B: the toe end of the middle capsule);
  zones    per touch sensor's zone: {key (its marker's), geom, geom_id, type
           (the site's geom type, "box"), local (centre), half (site_size),
           quat (orientation in the geom frame, w x y z)}.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from .common import DEFAULT_BUNDLE, bundle_label, manifest_sha256
from .simcfg import SOCKET_BODY, load_bundle_once, osl_bus_terms, write_container_atomic

FORMAT = "duetmimic-visual/1"
FILE = "visual.bin.gz"

OBJ = mujoco.mjtObj
GEOM = mujoco.mjtGeom
WRAP = mujoco.mjtWrap

# MjvOption's default geomgroup and tendongroup show groups 0-2.
VISIBLE_GROUPS = (0, 1, 2)
# figure/scene.py OSL_COVER_BODIES and three_demos.toml osl_cover_rgb.
OSL_COVER_BODIES = ("osl_knee_assembly", "osl_ankle_assembly")
OSL_COVER_RGB = (0xFF / 255, 0x6C / 255, 0x0C / 255)

# Simplification: absolute geometric error in metres. 0.3 mm is below a pixel
# at 1 m from a 720p, 45 degree camera; the demo camera sits about 3 m away.
SIMPLIFY_ERROR = 3.0e-4
# Crease angles for the normals: CAD hardware keeps its edges, bones are smooth.
CREASE_DEG = {"bone": 70.0, "hardware": 45.0, "osl_cover": 45.0, "socket": 45.0}

# The look (sRGB display values in [0, 1], as MuJoCo's rgba). The figure draws
# bones at geom rgba 0.8 grey on a white texture; the demo warms that slightly.
MATERIALS = {
    "bone": {"rgb": (0.86, 0.84, 0.80), "roughness": 0.62, "metalness": 0.0},
    "hardware": {"roughness": 0.45, "metalness": 0.25},  # rgb from the model
    "socket": {"roughness": 0.55, "metalness": 0.0},  # rgb, opacity from the model
    "osl_cover": {"rgb": OSL_COVER_RGB, "roughness": 0.50, "metalness": 0.0},
}
# Muscle colour by activation: piecewise linear in sRGB, from a pale pink at
# rest to dark red at 1. Walking activations have a median near 0.07 and a 90th
# percentile near 0.5 (osl_deploy figure runs, final_state/act), so most of the
# ramp's change sits below 0.35: a resting muscle stays pink, a working one is
# clearly red by 0.12-0.35 (the play-test's compromise ramp, 2026-11-16; the
# earlier ramp went light coral 0 -> red 0.3 -> deep red 1 and spent little
# contrast where most activations sit). Thin capsules are half in shade, so
# the stops sit a little brighter than the colour they read as.
MUSCLE_COLORMAP = (
    (0.0, (0.95, 0.72, 0.68)),
    (0.12, (0.96, 0.52, 0.47)),
    (0.35, (0.88, 0.18, 0.16)),
    (1.0, (0.62, 0.0, 0.04)),
)
FLOOR = {
    # figure/config.py Style.checker_rgb; the bundle's groundplane checker squares
    # are 0.125 m (300 px texture, texrepeat 4 per metre, 2 x 2 checks).
    "checker_rgb": [[0.88, 0.88, 0.88], [0.82, 0.82, 0.82]],
    "square": 0.125,
}
BACKGROUND_RGB = (0.955, 0.955, 0.95)
# Camera: MuJoCo's default fovy; azimuth_rel is the angle (degrees, CCW about z)
# from the walker's facing direction to the camera: -90 looks at its right side,
# where the OSL is. elevation < 0 looks down.
CAMERA = {"fovy": 45.0, "distance": 3.0, "elevation": -12.0, "azimuth_rel": -65.0,
          "target_offset": [0.0, 0.0, -0.05]}
# figure/config.py Style.light_dir: the direction the key light shines.
LIGHT_DIR = (0.25, 0.35, -1.0)

# Overlays: the role of each drawn geom, and a marker's kind by the type of its
# sensors (the types osl_deploy's sensor observation accepts).
ROLES = ("bone", "socket", "hardware", "cover")
SENSOR = mujoco.mjtSensor
MARKER_KIND = {
    int(SENSOR.mjSENS_JOINTPOS): "encoder",
    int(SENSOR.mjSENS_JOINTVEL): "encoder",
    int(SENSOR.mjSENS_GYRO): "imu",
    int(SENSOR.mjSENS_ACCELEROMETER): "imu",
    int(SENSOR.mjSENS_FORCE): "load_cell",
    int(SENSOR.mjSENS_TORQUE): "load_cell",
    int(SENSOR.mjSENS_TOUCH): "contact",
}
# A marker's key comes from its reads' names (B: osl_knee_pos + osl_knee_vel ->
# "knee_encoder", osl_imu_gyro + osl_imu_acc -> "imu", osl_ankle_load_force +
# _torque -> "ankle_cell", osl_heel_contact -> "heel"): their common words,
# without a leading "osl" or trailing words that name the quantity, then the
# kind's suffix.
MARKER_LEADING = {"osl"}
MARKER_TRAILING = {"pos", "vel", "gyro", "acc", "force", "torque", "load", "contact", "touch", "angle"}
MARKER_SUFFIX = {"encoder": ("_encoder", " encoder"), "load_cell": ("_cell", " load cell"), "contact": ("", " contact")}

NODE_SIMPLIFY = r"""
import fs from 'node:fs';
const [simplifierUrl, dir] = process.argv.slice(2);
const { MeshoptSimplifier } = await import(simplifierUrl);
await MeshoptSimplifier.ready;
const read = (name, Type) => {
  const bytes = new Uint8Array(fs.readFileSync(`${dir}/${name}`));  // own, aligned copy
  return new Type(bytes.buffer, 0, bytes.byteLength / Type.BYTES_PER_ELEMENT);
};
const jobs = JSON.parse(fs.readFileSync(`${dir}/jobs.json`, 'utf8'));
const positions = read('positions.f32', Float32Array);
const indices = read('indices.u32', Uint32Array);
const parts = [];
const results = [];
for (const job of jobs) {
  const pos = positions.subarray(job.vertex_start * 3, (job.vertex_start + job.vertex_count) * 3);
  const idx = indices.slice(job.index_start, job.index_start + job.index_count);
  const [out, error] = MeshoptSimplifier.simplify(
    idx, pos, 3, job.target_index_count, job.error, ['ErrorAbsolute', 'Prune']);
  parts.push(out);
  results.push({ index_count: out.length, error });
}
const total = parts.reduce((n, p) => n + p.length, 0);
const all = new Uint32Array(total);
let at = 0;
for (const p of parts) { all.set(p, at); at += p.length; }
fs.writeFileSync(`${dir}/out.u32`, Buffer.from(all.buffer));
fs.writeFileSync(`${dir}/out.json`, JSON.stringify(results));
"""


# --------------------------------------------------------------------------
# What is drawn


def load_model(bundle_dir):
    return mujoco.MjModel.from_binary_path(str(Path(bundle_dir) / "model.mjb"))


def drawn_geoms(model):
    """The robot geoms mjv_addGeoms draws with the default MjvOption (dynamic
    category: the static floor is left out), each with MuJoCo's evaluated look
    (material colour unless the geom rgba overrides it)."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 64)
    mujoco.mjv_addGeoms(
        model,
        data,
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
        scene,
    )
    looks = {}
    for g in scene.geoms[: scene.ngeom]:
        if g.objtype == int(OBJ.mjOBJ_GEOM):
            looks[int(g.objid)] = np.array(g.rgba, dtype=float)
    ids = sorted(looks)
    for i in ids:
        if model.geom_group[i] not in VISIBLE_GROUPS:
            raise ValueError(f"geom {i}: drawn but not in groups {VISIBLE_GROUPS}")
    return ids, looks


def osl_cover(model, geom):
    """An OSL knee or ankle assembly mesh: drawn as the figure's orange cover."""
    body = mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, model.geom_bodyid[geom]) or ""
    return bool(
        body.endswith(OSL_COVER_BODIES)
        and model.geom_type[geom] == GEOM.mjGEOM_MESH
        and model.geom_group[geom] == 0
    )


def category(model, geom, rgba):
    if osl_cover(model, geom):
        return "osl_cover"
    if rgba[3] < 0.999:
        return "socket"
    if float(np.mean(rgba[:3])) < 0.5:
        return "hardware"
    return "bone"


def materials_for(model, ids, looks):
    """(material specs for demo.json, material index of each drawn geom): one
    spec per distinct look; the category picks the palette entry above."""
    names, specs, index = [], [], []
    for g in ids:
        rgba = np.clip(looks[g], 0.0, 1.0)
        cat = category(model, g, rgba)
        spec = dict(MATERIALS[cat])
        spec.setdefault("rgb", tuple(float(c) for c in rgba[:3]))
        spec["opacity"] = float(rgba[3]) if cat == "socket" else 1.0
        spec = {
            "name": cat,
            "category": cat,
            "rgb": [round(float(c), 4) for c in spec["rgb"]],
            "opacity": round(spec["opacity"], 4),
            "roughness": spec["roughness"],
            "metalness": spec["metalness"],
        }
        key = json.dumps(spec, sort_keys=True)
        if key not in names:
            names.append(key)
            specs.append(spec)
        index.append(names.index(key))
    # Distinct names when one category ends up with several colours.
    seen = {}
    for spec in specs:
        n = seen.get(spec["name"], 0)
        seen[spec["name"]] = n + 1
        if n:
            spec["name"] = f"{spec['name']}_{n}"
    return specs, np.asarray(index, dtype=np.int32)


def drawn_tendons(model):
    """Spatial tendons in the visible tendon groups (fixed ones draw nothing)."""
    return [
        t
        for t in range(model.ntendon)
        if model.tendon_group[t] in VISIBLE_GROUPS and model.tendon_num[t] > 0
    ]


def max_segments(model, tendons):
    """An upper bound on the segments drawn: a wrapping geom adds two points,
    a site or pulley one; a tendon with n points has at most n - 1 segments."""
    total = 0
    for t in tendons:
        adr, num = model.tendon_adr[t], model.tendon_num[t]
        types = model.wrap_type[adr : adr + num]
        points = int(np.sum(np.where((types == WRAP.mjWRAP_SPHERE) | (types == WRAP.mjWRAP_CYLINDER), 2, 1)))
        total += max(points - 1, 0)
    return total


def tendon_segments(model, data, tendons):
    """Reference implementation of the muscle segments (what render/frame.js
    computes): (tendon, p0, p1, radius) per segment, mjv_addGeoms' rules."""
    xpos = data.wrap_xpos.reshape(-1, 3)
    obj = data.wrap_obj.reshape(-1)
    out = []
    for t in tendons:
        adr, num = int(data.ten_wrapadr[t]), int(data.ten_wrapnum[t])
        width = float(model.tendon_width[t])
        for j in range(adr, adr + num - 1):
            if obj[j] == -2 or obj[j + 1] == -2:
                continue
            radius = 0.5 * width if obj[j] >= 0 and obj[j + 1] >= 0 else width
            out.append((t, xpos[j].copy(), xpos[j + 1].copy(), radius))
    return out


def root_frame(bundle_dir, model, manifest):
    """(root body, forward axis in the root body's frame). Forward is the
    travel direction of the fastest straight forward clip at its first frame,
    snapped to a body axis when within 15 degrees."""
    joint = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, manifest["root_joint"])
    if joint < 0:
        raise ValueError(f"root joint {manifest['root_joint']!r} not in the model")
    body = int(model.jnt_bodyid[joint])
    forward = np.array([1.0, 0.0, 0.0])
    path = Path(bundle_dir) / "reference.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as ref:
            twists = ref["twists"]
            straight = (np.abs(twists[:, 1]) < 1e-6) & (np.abs(twists[:, 2]) < 1e-6)
            if np.any(straight & (twists[:, 0] > 0)):
                clip = int(np.argmax(np.where(straight, twists[:, 0], -np.inf)))
                n = int(ref["num_frames"][clip])
                pos = ref["root_pos"][clip, :n].astype(float)
                quat = ref["root_quat"][clip, 0].astype(float)
                travel = pos[-1] - pos[0]
                travel[2] = 0.0
                mat = np.zeros(9)
                mujoco.mju_quat2Mat(mat, quat)
                local = mat.reshape(3, 3).T @ (travel / np.linalg.norm(travel))
                axis = np.zeros(3)
                k = int(np.argmax(np.abs(local)))
                axis[k] = np.sign(local[k])
                forward = axis if local @ axis > np.cos(np.radians(15)) else local
    return body, [float(v) for v in forward]


# --------------------------------------------------------------------------
# Overlays (demo.json visual.overlays, see the module docstring)


def clean(values, digits=9):
    """Floats for demo.json: rounded (to a nanometre for lengths), no -0.0."""
    return [round(float(v), digits) + 0.0 for v in values]


def subtree_bodies(model, root):
    """The bodies of root's kinematic subtree, root included."""
    bodies = set()
    for b in range(model.nbody):
        a = b
        while a > 0 and a != root:
            a = int(model.body_parentid[a])
        if a == root:
            bodies.add(b)
    return bodies


def geom_roles(model, ids, socket):
    """Each drawn geom's role, by its body: on the socket body "socket"; on the
    rest of the socket's subtree (the prosthesis) "cover" for the OSL housings,
    else "hardware"; anywhere else (the human) "bone"."""
    prosthesis = subtree_bodies(model, socket)
    roles = []
    for g in ids:
        body = int(model.geom_bodyid[g])
        if body == socket:
            roles.append("socket")
        elif body in prosthesis:
            roles.append("cover" if osl_cover(model, g) else "hardware")
        else:
            roles.append("bone")
    return roles


def attached_geom(model, data, ids, body, point):
    """The drawn geom (index into ids) that moves rigidly with `body`: one on the
    body itself, else one on a body welded to it; the nearest to `point`."""
    candidates = [i for i, g in enumerate(ids) if model.geom_bodyid[g] == body]
    if not candidates:
        weld = model.body_weldid[body]
        candidates = [i for i, g in enumerate(ids) if model.body_weldid[model.geom_bodyid[g]] == weld]
    if not candidates:
        raise ValueError(f"no drawn geom moves with body {mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, body)}")
    return min(candidates, key=lambda i: float(np.linalg.norm(data.geom_xpos[ids[i]] - point)))


def in_geom_frame(data, geom, point):
    """A world point in a geom's frame: R_g^T (p - geom_xpos)."""
    rot = data.geom_xmat[geom].reshape(3, 3)
    return rot.T @ (np.asarray(point, dtype=float) - data.geom_xpos[geom])


def contact_point(model, data, site):
    """Where a touch sensor's zone (a box site) meets the foot: of the collision
    geoms (contype or conaffinity set) that move with the site's body, the
    point of a geom's axis (a capsule's or cylinder's segment, else its
    centre) inside the box nearest the box's centre; the box's centre when no
    axis enters it. A touch sensor sums the contacts inside its box, and those
    act on these geoms, so the glyph marks the part of the foot it feels."""
    centre = data.site_xpos[site].copy()
    rot = data.site_xmat[site].reshape(3, 3)
    half = model.site_size[site]
    weld = model.body_weldid[model.site_bodyid[site]]
    best, best_distance = centre, np.inf
    for g in range(model.ngeom):
        if model.body_weldid[model.geom_bodyid[g]] != weld or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        middle = data.geom_xpos[g]
        if model.geom_type[g] in (GEOM.mjGEOM_CAPSULE, GEOM.mjGEOM_CYLINDER):
            reach = data.geom_xmat[g].reshape(3, 3)[:, 2] * model.geom_size[g][1]
            a, b = middle - reach, middle + reach
            ab = b - a
            t = float(np.clip(np.dot(centre - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            point = a + t * ab
        else:
            point = middle.copy()
        if np.any(np.abs(rot.T @ (point - centre)) > half + 1e-9):
            continue
        distance = float(np.linalg.norm(point - centre))
        if distance < best_distance:
            best, best_distance = point, distance
    return best


def marker_name(read_names, kind):
    """(key, label) of a marker from its reads' names (MARKER_* above)."""
    words = [name.lower().split("_") for name in read_names]
    common = []
    for parts in zip(*words, strict=False):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    while common and common[0] in MARKER_LEADING:
        common.pop(0)
    while common and common[-1] in MARKER_TRAILING:
        common.pop()
    stem = "_".join(common)
    if kind == "imu":
        place = stem[: -len("imu")].rstrip("_") if stem.endswith("imu") else stem
        key = f"{place}_imu" if place else "imu"
        label = f"{place.replace('_', ' ')} IMU".strip()
    else:
        key_suffix, label_suffix = MARKER_SUFFIX[kind]
        key = f"{stem}{key_suffix}" if stem else kind
        label = f"{stem.replace('_', ' ')}{label_suffix}" if stem else kind.replace("_", " ")
    return key, label[:1].upper() + label[1:]


def overlay_section(model, ids, terms):
    """demo.json visual.overlays for the drawn geoms `ids` and the OSL bus terms
    as simcfg exports them (sim.osl.observation.terms): the frames' `sensors`
    are those terms' frames, concatenated, so a read's values are channels
    start + read.frame + c. Computed at qpos0 (every point is rigid in its
    geom's frame)."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    socket = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY, SOCKET_BODY)
    if socket < 0:
        raise ValueError(f"The bundle model has no body {SOCKET_BODY}")
    origin = data.xpos[socket].copy()
    s = attached_geom(model, data, ids, socket, origin)
    if model.geom_bodyid[ids[s]] != socket:
        raise ValueError(f"No drawn geom on the socket body {SOCKET_BODY}")
    overlays = {
        "roles": geom_roles(model, ids, socket),
        "socket": {
            "geom": s,
            "geom_id": int(ids[s]),
            "body": int(socket),
            "local": clean(in_geom_frame(data, ids[s], origin)),
        },
        "markers": [],
        "zones": [],
    }
    # Group the reads by (site or joint, kind), in frame order.
    groups, start = {}, 0
    for term in terms:
        for read in term["reads"]:
            sensor = int(np.flatnonzero(model.sensor_adr == read["adr"])[0])
            kind = MARKER_KIND.get(int(model.sensor_type[sensor]))
            if kind is None:
                raise ValueError(f"OSL read {read['name']}: no marker kind for sensor type {model.sensor_type[sensor]}")
            anchor = read["anchor"]
            group = groups.setdefault(
                (anchor["kind"], anchor["id"], kind), {"kind": kind, "anchor": anchor, "names": [], "channels": []}
            )
            dim = len(read["components"]) if read["components"] is not None else int(read["size"])
            group["names"].append(read["name"])
            group["channels"] += [start + int(read["frame"]) + c for c in range(dim)]
        start += int(term["frame_dim"])
    keys = set()
    for group in groups.values():
        anchor, kind = group["anchor"], group["kind"]
        key, label = marker_name(group["names"], kind)
        base, n = key, 1
        while key in keys:
            n += 1
            key = f"{base}_{n}"
        keys.add(key)
        point = data.site_xpos[anchor["id"]] if anchor["kind"] == "site" else data.xanchor[anchor["id"]]
        zone = kind == "contact" and anchor["kind"] == "site" and model.site_type[anchor["id"]] == GEOM.mjGEOM_BOX
        centre = point
        if zone:
            point = contact_point(model, data, anchor["id"])
        g = attached_geom(model, data, ids, anchor["body"], point)
        rot = data.geom_xmat[ids[g]].reshape(3, 3)
        marker = {
            "key": key,
            "label": label,
            "kind": kind,
            "geom": g,
            "geom_id": int(ids[g]),
            "local": clean(in_geom_frame(data, ids[g], point)),
            "channels": sorted(group["channels"]),
            "anchor": {
                "kind": anchor["kind"],
                "id": int(anchor["id"]),
                "name": anchor["name"],
                "body": int(anchor["body"]),
            },
        }
        if anchor["kind"] == "joint":
            marker["axis"] = clean(rot.T @ data.xaxis[anchor["id"]])
        overlays["markers"].append(marker)
        if zone:
            site = anchor["id"]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (rot.T @ data.site_xmat[site].reshape(3, 3)).reshape(-1))
            quat = -quat if quat[0] < 0 else quat
            overlays["zones"].append(
                {
                    "key": key,
                    "geom": g,
                    "geom_id": int(ids[g]),
                    "type": mujoco.mjtGeom(model.site_type[site]).name.removeprefix("mjGEOM_").lower(),
                    "local": clean(in_geom_frame(data, ids[g], centre)),
                    "half": clean(model.site_size[site]),
                    "quat": clean(quat),
                }
            )
    return overlays


# --------------------------------------------------------------------------
# Meshes


def weld(vertices, faces):
    """Merge exactly equal positions, drop degenerate faces."""
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    f = inverse.reshape(-1)[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return unique.astype(np.float32), f[keep].astype(np.uint32)


def node_simplify(meshes, error):
    """Simplify [(positions, faces)] with meshoptimizer in Node; returns the new
    faces of each (indices into the same positions) and the errors."""
    node = shutil.which("node")
    simplifier = Path(__file__).resolve().parents[1] / "node_modules/meshoptimizer/meshopt_simplifier.js"
    if node is None:
        raise RuntimeError("node is needed to simplify the meshes (Node 22)")
    if not simplifier.exists():
        raise RuntimeError(f"{simplifier} is missing: run `npm ci` in tools/")
    jobs, vstart, istart = [], 0, 0
    for positions, faces in meshes:
        jobs.append(
            {
                "vertex_start": vstart,
                "vertex_count": len(positions),
                "index_start": istart,
                "index_count": faces.size,
                "target_index_count": 0,
                "error": error,
            }
        )
        vstart += len(positions)
        istart += faces.size
    with tempfile.TemporaryDirectory(prefix="duet_visual_") as tmp:
        tmp = Path(tmp)
        np.concatenate([p for p, _ in meshes]).astype("<f4").tofile(tmp / "positions.f32")
        np.concatenate([f.reshape(-1) for _, f in meshes]).astype("<u4").tofile(tmp / "indices.u32")
        (tmp / "jobs.json").write_text(json.dumps(jobs))
        (tmp / "simplify.mjs").write_text(NODE_SIMPLIFY)
        run = subprocess.run(
            [node, str(tmp / "simplify.mjs"), simplifier.as_uri(), str(tmp)],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            raise RuntimeError(f"meshoptimizer failed:\n{run.stderr}")
        results = json.loads((tmp / "out.json").read_text())
        flat = np.fromfile(tmp / "out.u32", dtype="<u4")
    out, at = [], 0
    for result in results:
        n = result["index_count"]
        out.append((flat[at : at + n].reshape(-1, 3).copy(), float(result["error"])))
        at += n
    return out


def crease_normals(positions, faces, crease_deg):
    """Split vertices where faces meet at more than `crease_deg`; returns
    (positions, unit normals, faces). A corner's normal is the area-weighted sum
    of the face normals around its vertex that lie within the crease angle of
    its own face; corners with equal (vertex, int8 normal) share a vertex."""
    p = positions.astype(np.float64)
    f = faces.astype(np.int64)
    fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    keep = area > 1e-14
    f, fn, area = f[keep], fn[keep], area[keep]
    fu = fn / area[:, None]
    cv = f.reshape(-1)  # vertex of each corner
    cf = np.repeat(np.arange(len(f)), 3)  # face of each corner
    order = np.argsort(cv, kind="stable")
    sv, sf = cv[order], cf[order]
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    counts = np.diff(np.r_[starts, len(sv)])
    group = np.repeat(np.arange(len(starts)), counts)
    reps = counts[group]  # every corner pairs with each corner of its vertex
    i = np.repeat(np.arange(len(sv)), reps)
    offset = np.arange(len(i)) - np.repeat(np.cumsum(reps) - reps, reps)
    j = starts[group[i]] + offset
    fi, fj = sf[i], sf[j]
    near = np.einsum("ij,ij->i", fu[fi], fu[fj]) >= np.cos(np.radians(crease_deg))
    acc = np.stack(
        [np.bincount(i[near], weights=fn[fj[near], c], minlength=len(sv)) for c in range(3)],
        axis=1,
    )
    normals = np.empty_like(acc)
    normals[order] = acc
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-30)[:, None]
    q = np.clip(np.round(normals * 127.0), -127, 127).astype(np.int8)
    key = np.concatenate([cv[:, None], q.astype(np.int64)], axis=1)
    unique, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    out_pos = p[unique[:, 0]].astype(np.float32)
    out_nrm = normals[first]
    out_faces = inverse.reshape(-1, 3).astype(np.uint32)
    return out_pos, out_nrm, out_faces


def optimize_order(positions, normals, faces):
    """Vertices in first-use order (better locality for gzip and the GPU)."""
    flat = faces.reshape(-1)
    _, first = np.unique(flat, return_index=True)
    order = flat[np.sort(first)]
    remap = np.empty(len(positions), dtype=np.int64)
    remap[order] = np.arange(len(order))
    return positions[order], normals[order], remap[faces].astype(np.uint32)


def quantize_positions(positions):
    lo, hi = positions.min(0).astype(np.float64), positions.max(0).astype(np.float64)
    offset = (lo + hi) / 2
    scale = np.maximum((hi - lo) / 2 / 32767.0, 1e-9)
    q = np.clip(np.round((positions - offset) / scale), -32767, 32767).astype(np.int16)
    return q, offset.astype(np.float32), scale.astype(np.float32)


def build_meshes(model, mesh_ids, crease):
    """Simplified, normal-split, quantized meshes for MuJoCo mesh ids."""
    sources = []
    for mid in mesh_ids:
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        vertices = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float32)
        faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int64)
        sources.append(weld(vertices, faces))
    simplified = node_simplify(sources, SIMPLIFY_ERROR)
    meshes = []
    for mid, (positions, src_faces), (faces, error) in zip(mesh_ids, sources, simplified, strict=True):
        pos, nrm, tri = crease_normals(positions, faces, crease[mid])
        pos, nrm, tri = optimize_order(pos, nrm, tri)
        if len(pos) > 65535:
            raise ValueError(f"mesh {mid}: {len(pos)} vertices do not fit uint16 indices")
        q, offset, scale = quantize_positions(pos)
        meshes.append(
            {
                "mesh": int(mid),
                "name": mujoco.mj_id2name(model, OBJ.mjOBJ_MESH, mid),
                "source_triangles": int(len(src_faces)),
                "triangles": int(len(tri)),
                "vertices": int(len(pos)),
                "error": error,
                "positions": q,
                "normals": np.clip(np.round(nrm * 127.0), -127, 127).astype(np.int8),
                "indices": tri.reshape(-1).astype(np.uint16),
                "offset": offset,
                "scale": scale,
            }
        )
    return meshes


# --------------------------------------------------------------------------
# Export


def export(bundle_dir, out_dir, ctx):
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    manifest = ctx["manifest"]
    model = load_model(bundle_dir)
    ids, looks = drawn_geoms(model)
    specs, geom_material = materials_for(model, ids, looks)

    # A mesh shared by geoms of several categories takes the sharpest crease.
    mesh_ids, crease = [], {}
    for g, mat in zip(ids, geom_material, strict=True):
        if model.geom_type[g] == GEOM.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            angle = CREASE_DEG[specs[mat]["category"]]
            if mid not in crease:
                mesh_ids.append(mid)
            crease[mid] = min(crease.get(mid, angle), angle)
    meshes = build_meshes(model, mesh_ids, crease)
    mesh_index = {m["mesh"]: k for k, m in enumerate(meshes)}

    vstart = np.cumsum([0] + [m["vertices"] for m in meshes])[:-1]
    istart = np.cumsum([0] + [m["indices"].size for m in meshes])[:-1]
    arrays = {
        "positions": np.concatenate([m["positions"] for m in meshes]),
        "normals": np.concatenate([m["normals"] for m in meshes]),
        "indices": np.concatenate([m["indices"] for m in meshes]),
        "mesh_vertex_start": vstart.astype(np.int32),
        "mesh_vertex_count": np.array([m["vertices"] for m in meshes], dtype=np.int32),
        "mesh_index_start": istart.astype(np.int32),
        "mesh_index_count": np.array([m["indices"].size for m in meshes], dtype=np.int32),
        "mesh_offset": np.stack([m["offset"] for m in meshes]).astype(np.float32),
        "mesh_scale": np.stack([m["scale"] for m in meshes]).astype(np.float32),
        "mesh_id": np.array([m["mesh"] for m in meshes], dtype=np.int32),
        "geom_id": np.array(ids, dtype=np.int32),
        "geom_body": model.geom_bodyid[ids].astype(np.int32),
        "geom_type": model.geom_type[ids].astype(np.int32),
        "geom_size": model.geom_size[ids].astype(np.float32),
        "geom_mesh": np.array(
            [mesh_index.get(int(model.geom_dataid[g]), -1) if model.geom_type[g] == GEOM.mjGEOM_MESH else -1 for g in ids],
            dtype=np.int32,
        ),
        "geom_material": geom_material,
    }
    meta = {
        "format": FORMAT,
        "mesh_names": [m["name"] for m in meshes],
        "materials": [s["name"] for s in specs],
        "positions": "int16 per mesh: p = q * mesh_scale + mesh_offset (geom frame, metres)",
        "normals": "int8, n = q / 127",
        "indices": "uint16, local to each mesh (add nothing: one BufferGeometry per mesh)",
    }
    size = write_container_atomic(out_dir / FILE, arrays, meta)

    tendons = drawn_tendons(model)
    root_body, forward = root_frame(bundle_dir, model, manifest)
    triangles = sum(m["triangles"] for m in meshes)
    source = sum(m["source_triangles"] for m in meshes)
    print(
        f"[visual] {len(ids)} geoms, {len(meshes)} meshes: {source} -> {triangles} triangles, "
        f"{sum(m['vertices'] for m in meshes)} vertices; {FILE} {size['bytes'] / 1e6:.2f} MB "
        f"({size['raw_bytes'] / 1e6:.2f} MB raw); {len(tendons)} tendons, "
        f"<= {max_segments(model, tendons)} segments"
    )
    overlays = overlay_section(model, ids, osl_bus_terms(load_bundle_once(bundle_dir, ctx)))
    # The roles come from the bodies; the materials (by colour and opacity)
    # should agree with them.
    looks_as = [{"osl_cover": "cover"}.get(c, c) for c in (specs[k]["category"] for k in geom_material)]
    differ = [int(g) for g, role, look in zip(ids, overlays["roles"], looks_as, strict=True) if role != look]
    if differ:
        print(f"[visual] warning: geoms {differ} have a role other than their material's category")
    counts = {role: overlays["roles"].count(role) for role in ROLES}
    print(
        f"[visual] overlays: roles {counts}; socket geom {overlays['socket']['geom_id']} "
        f"local {overlays['socket']['local']}; markers "
        f"{[(mk['key'], mk['geom_id'], mk['channels']) for mk in overlays['markers']]}; "
        f"zones {[z['key'] for z in overlays['zones']]}"
    )
    return {
        "visual": {
            "format": FORMAT,
            "file": FILE,
            "bytes": size["bytes"],
            "raw_bytes": size["raw_bytes"],
            # Frame order of render/frame.js: geom_xpos + geom_xmat of these.
            "geom_ids": [int(g) for g in ids],
            "materials": specs,
            "muscles": {
                "tendon_ids": [int(t) for t in tendons],
                "max_segments": max_segments(model, tendons),
                "colormap": [[s, [round(c, 4) for c in rgb]] for s, rgb in MUSCLE_COLORMAP],
                "opacity": 1.0,
                "width_scale": 1.0,
            },
            "root": {
                "body": root_body,
                "name": mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, root_body),
                "forward": forward,
            },
            "floor": FLOOR,
            "background_rgb": list(BACKGROUND_RGB),
            "light_dir": list(LIGHT_DIR),
            "camera": CAMERA,
            "stats": {
                "meshes": len(meshes),
                "triangles": triangles,
                "source_triangles": source,
                "vertices": int(sum(m["vertices"] for m in meshes)),
                "simplify_error_m": SIMPLIFY_ERROR,
                "max_error_m": max(m["error"] for m in meshes),
            },
            # X-ray and sensing views (module docstring; tools/DESIGN.md).
            "overlays": overlays,
        }
    }


# --------------------------------------------------------------------------
# Test fixture: poses and MuJoCo's own tendon capsules (tools/test/render_*)


def reference_qpos(model, manifest, ref, clip, frame):
    qpos = model.qpos0.copy()
    joint = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, manifest["root_joint"])
    adr = int(model.jnt_qposadr[joint])
    qpos[adr : adr + 3] = ref["root_pos"][clip, frame]
    qpos[adr + 3 : adr + 7] = ref["root_quat"][clip, frame]
    ids = [mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, n) for n in manifest["joint_names"]]
    qpos[model.jnt_qposadr[ids]] = ref["joint_pos"][clip, frame]
    return qpos


def mjv_capsules(model, data):
    """MuJoCo's tendon capsules for the current state: (tendon, pos, axis, radius, half)."""
    scene = mujoco.MjvScene(model, maxgeom=20000)
    opt = mujoco.MjvOption()
    mujoco.mjv_updateScene(
        model, data, opt, None, mujoco.MjvCamera(), int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scene
    )
    out = []
    for g in scene.geoms[: scene.ngeom]:
        if g.objtype == int(OBJ.mjOBJ_TENDON):
            axis = np.array(g.mat).reshape(3, 3)[:, 2]
            out.append((int(g.objid), np.array(g.pos), axis, float(g.size[0]), float(g.size[2])))
    return out


def write_poses(path, bundle_dir=DEFAULT_BUNDLE):
    """Poses for the render tests: the idle clip's frame 0 and a forward-walk
    frame at the widest step, with skewed synthetic activations (median about
    0.15), plus MuJoCo's own capsules and geom frames to check frame.js."""
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    model = load_model(bundle_dir)
    data = mujoco.MjData(model)
    names = manifest["reference"]["clip_names"]
    ids, _ = drawn_geoms(model)
    tendons = drawn_tendons(model)
    rng = np.random.default_rng(7)
    out = {
        "bundle": bundle_label(bundle_dir, manifest),
        "manifest_sha256": manifest_sha256(bundle_dir),
        "poses": {},
    }
    with np.load(bundle_dir / "reference.npz", allow_pickle=False) as npz:
        ref = {k: npz[k] for k in ("root_pos", "root_quat", "joint_pos", "num_frames")}
    walk = names.index("walk_fwd_vx+01000_vy+00000_wz+00000")
    foot_l = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY, "robot/calcn_l")
    foot_r = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY, "robot/OSL_KA_L1_osl_foot_assembly")
    spread = []
    for k in range(int(ref["num_frames"][walk])):
        data.qpos[:] = reference_qpos(model, manifest, ref, walk, k)
        mujoco.mj_kinematics(model, data)
        spread.append(np.linalg.norm(data.xpos[foot_l, :2] - data.xpos[foot_r, :2]))
    stride = int(np.argmax(spread))
    for name, clip, frame in (("idle", names.index("idle"), 0), ("stride", walk, stride)):
        data.qpos[:] = reference_qpos(model, manifest, ref, clip, frame)
        data.qvel[:] = 0
        data.act[:] = rng.beta(0.7, 3.0, size=model.na)
        mujoco.mj_forward(model, data)
        capsules = mjv_capsules(model, data)
        mine = tendon_segments(model, data, tendons)
        if len(mine) != len(capsules):
            raise AssertionError(f"{name}: {len(mine)} segments, MuJoCo draws {len(capsules)}")
        # mjvGeom stores float32: compare at 1e-6 m.
        for (t, p0, p1, r), (t2, pos, _, radius, half) in zip(mine, capsules, strict=True):
            if (
                t != t2
                or abs(r - radius) > 1e-6
                or np.abs((p0 + p1) / 2 - pos).max() > 1e-6
                or abs(np.linalg.norm(p1 - p0) / 2 - half) > 1e-6
            ):
                raise AssertionError(f"{name}: tendon {t} segment differs from mjv_addGeoms")
        out["poses"][name] = {
            "clip": names[clip],
            "frame": frame,
            "qpos": data.qpos.tolist(),
            "act": data.act.tolist(),
            "mjv_capsules": [[t, *pos.tolist(), *axis.tolist(), r, h] for t, pos, axis, r, h in capsules],
            "geom_xpos": data.geom_xpos[ids].tolist(),
            "geom_xmat": data.geom_xmat[ids].tolist(),
        }
        print(f"[visual] pose {name}: clip {names[clip]!r} frame {frame}, {len(capsules)} capsules match")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    print(f"[visual] {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="duet_export.visual")
    sub = parser.add_subparsers(dest="command", required=True)
    poses = sub.add_parser("poses", help="write the render test fixture")
    poses.add_argument("out", type=Path)
    poses.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    if args.command == "poses":
        write_poses(args.out, args.bundle)


if __name__ == "__main__":
    main()
