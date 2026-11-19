"""physics.mjb.gz: the bundle's model.mjb without the mesh data physics never reads.

The bundle model is 103 MB, almost all of it render geometry: 1.1 M mesh BVH
nodes, 557 k triangles with normals and 275 k vertices. In MuJoCo 3.11.0 the
simulation pipeline reads mesh data only in collision (plus the tactile sensor,
which this bundle lacks), and there only through each mesh's convex hull:
mjc_initCCDObj, the hill-climbing support functions and mjc_PlaneConvex
(engine_collision_convex.c) read mesh_vert through the hull graph's
vert_globalid, mesh_graph, and the mesh_poly* and per-vertex polymap arrays
(multi-contact polygons, engine_collision_gjk.c). Faces, normals, texcoords,
mesh BVHs and octrees are read only by ray casting (mj_ray), SDF collisions,
the tactile sensor and the renderer.

Stripping rule (`strip`):
- A collision mesh is one used by a geom that can collide: contype or
  conaffinity nonzero, or named by an explicit <pair>.
- A collision mesh keeps its hull graph and polygons and, of its vertices, the
  ones the hull or a polygon references. They are renumbered in their original
  order, so every vertex-index comparison in GJK/EPA is unchanged. The hull
  graph decides the code path (hill-climbing needs a graph and at least
  mjMESH_HILLCLIMB_MIN vertices); a mesh whose path the renumbering would
  change keeps all of its vertices.
- Every other mesh keeps no vertices, hull or polygons.
- No mesh keeps faces, normals, texcoords, octree nodes or BVH nodes. Each mesh
  gets one childless BVH node holding its original root box, so mj_ray stays
  memory-safe: it reports no hit on a mesh geom instead of reading past the
  arrays.
- Everything else is byte-identical: every other size and id, names and paths,
  options, visual and statistic structs, body BVHs, textures and materials.

`check_supported` refuses models outside the rule (SDF geoms, flexes, plugins,
ray-cast or mesh-reading sensors). export() then proves the result before
writing it: the slim file loads and MuJoCo's own mj_saveModel reproduces it,
every other array is unchanged (`check_arrays`), every hull is intact
(`check_hulls`), and `verify_dynamics` steps the original and the slim model
side by side, bit for bit, with policy controls recorded from osl_deploy's
closed loop (walking), random and zero controls (falls), from penetrating
poses (also with multi-CCD on) and through every geom-distance query against
the collision meshes. The numbers go into demo.json; the native trace the WASM
test (tools/test/physics.test.mjs) compares against goes to
tools/test/physics_wasm_ref.bin.gz (tracked) for the demo's own asset folder,
and into the --out folder for any other (common.golden_path).
"""

import hashlib
import time
from pathlib import Path

import mujoco
import numpy as np

from . import mjb
from .common import golden_path, load_bundle
from .container import write_container, write_gzip

FILE = "physics.mjb.gz"
WASM_REF_NAME = "physics_wasm_ref.bin.gz"  # golden for physics.test.mjs; golden_path places it

HILLCLIMB_MIN = 10  # mjMESH_HILLCLIMB_MIN, engine_collision_convex.h (3.11.0)
MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
SDF = int(mujoco.mjtGeom.mjGEOM_SDF)
UNSUPPORTED_SENSORS = (
    mujoco.mjtSensor.mjSENS_RANGEFINDER,  # mj_ray: mesh faces and BVH
    mujoco.mjtSensor.mjSENS_TACTILE,  # mesh normals, SDF
    mujoco.mjtSensor.mjSENS_GEOMDIST,  # mj_geomDistance on any geom pair
    mujoco.mjtSensor.mjSENS_GEOMNORMAL,
    mujoco.mjtSensor.mjSENS_GEOMFROMTO,
    mujoco.mjtSensor.mjSENS_PLUGIN,
)
# Counts the demo relies on; all must survive stripping unchanged.
COUNTS = (
    "nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite", "ntendon", "nwrap",
    "nsensor", "nsensordata", "nmesh", "nmat", "ntex", "neq", "npair",
    "nexclude", "nkey", "nmocap", "ntree", "ncam", "nlight", "nnames",
)  # fmt: skip
# The only arrays strip() may change; every other one stays byte-identical.
STRIPPABLE = frozenset(
    [f"mesh_{k}" for k in (
        "vertadr", "vertnum", "faceadr", "facenum", "bvhadr", "bvhnum", "octadr",
        "octnum", "normaladr", "normalnum", "texcoordadr", "texcoordnum", "graphadr",
        "vert", "normal", "texcoord", "face", "facenormal", "facetexcoord", "graph",
        "polynum", "polyadr", "polynormal", "polyvertadr", "polyvertnum", "polyvert",
        "polymapadr", "polymapnum", "polymap",
    )]
    + ["bvh_depth", "bvh_child", "bvh_nodeid", "bvh_aabb", "body_bvhadr"]
    + ["oct_depth", "oct_child", "oct_aabb", "oct_coeff"]
)  # fmt: skip


# ----------------------------------------------------------------- stripping


def _col(a, name):
    return a[name][:, 0]


def check_supported(m):
    """Raise unless the stripping rule's proof covers this model (an Mjb)."""
    a, s = m.arrays, m.sizes
    problems = []
    if np.any(_col(a, "geom_type") == SDF):
        problems.append("SDF geoms (mjc_MeshSDF reads mesh faces and BVH)")
    for key in ("nflex", "nplugin", "nskin", "nbvhdynamic"):
        if s[key]:
            problems.append(f"{key} = {s[key]}")
    types = set(_col(a, "sensor_type").tolist())
    for kind in UNSUPPORTED_SENSORS:
        if int(kind) in types:
            problems.append(f"{kind.name} sensors")
    if problems:
        raise ValueError("physics.strip does not support: " + "; ".join(problems))


def collision_meshes(m):
    """Boolean mask over meshes: used by a geom that can take part in a contact."""
    a = m.arrays
    gtype, dataid = _col(a, "geom_type"), _col(a, "geom_dataid")
    can_collide = (_col(a, "geom_contype") | _col(a, "geom_conaffinity")) != 0
    can_collide[_col(a, "pair_geom1")] = True
    can_collide[_col(a, "pair_geom2")] = True
    keep = np.zeros(m.sizes["nmesh"], bool)
    used = (gtype == MESH) & (dataid >= 0) & can_collide
    keep[dataid[used]] = True
    return keep


def _graph_size(graph, adr):
    """Length of one mesh's hull graph: 2 + 3*numvert + 6*numface ints."""
    numvert, numface = int(graph[adr]), int(graph[adr + 1])
    return 2 + 3 * numvert + 6 * numface


def strip(src, hull_only=True):
    """(slim Mjb, info): `src` without the mesh data physics never reads."""
    check_supported(src)
    a, s = src.arrays, src.sizes
    out = src.copy()
    o = out.arrays
    nmesh = s["nmesh"]
    keep = collision_meshes(src)

    vertadr, vertnum = _col(a, "mesh_vertadr"), _col(a, "mesh_vertnum")
    graphadr, graph = _col(a, "mesh_graphadr"), _col(a, "mesh_graph")
    polyadr, polynum = _col(a, "mesh_polyadr"), _col(a, "mesh_polynum")
    pv_adr, pv_num = _col(a, "mesh_polyvertadr"), _col(a, "mesh_polyvertnum")
    polyvert, polymap = _col(a, "mesh_polyvert"), _col(a, "mesh_polymap")
    pm_adr, pm_num = _col(a, "mesh_polymapadr"), _col(a, "mesh_polymapnum")

    verts, pm_nums, pm_adrs, polymaps = [], [], [], []
    graphs, polynormals, polyvertnums, polyvertadrs, polyverts = [], [], [], [], []
    n_vert = n_polymap = n_graph = n_poly = n_polyvert = 0
    new = {key: np.zeros(nmesh, np.int32) for key in (
        "mesh_vertadr", "mesh_vertnum", "mesh_graphadr", "mesh_polyadr", "mesh_polynum",
    )}  # fmt: skip
    kept_vertices, hull_compacted = {}, []
    for i in range(nmesh):
        new["mesh_vertadr"][i] = n_vert
        new["mesh_polyadr"][i] = n_poly
        new["mesh_graphadr"][i] = -1
        if not keep[i]:
            continue
        v0, nv = int(vertadr[i]), int(vertnum[i])
        p0, npoly = int(polyadr[i]), int(polynum[i])
        g0 = int(graphadr[i])
        g = graph[g0 : g0 + _graph_size(graph, g0)].copy() if g0 >= 0 else None
        # Local vertices collision can reach: the hull's, polygons', polymap's.
        polys = range(p0, p0 + npoly)
        poly_locals = (
            np.concatenate([polyvert[pv_adr[p] : pv_adr[p] + pv_num[p]] for p in polys])
            if npoly
            else np.zeros(0, np.int32)
        )
        select = np.arange(nv)
        if hull_only and g is not None and nv >= HILLCLIMB_MIN:
            numvert = int(g[0])
            used = np.zeros(nv, bool)
            used[g[2 + numvert : 2 + 2 * numvert]] = True
            used[poly_locals] = True
            used[pm_num[v0 : v0 + nv] > 0] = True
            candidate = np.flatnonzero(used)
            if len(candidate) >= HILLCLIMB_MIN:  # keeps the hill-climbing path
                select = candidate
                hull_compacted.append(i)
        remap = np.full(nv, -1, np.int64)
        remap[select] = np.arange(len(select))
        if g is not None:
            numvert = int(g[0])
            ids = remap[g[2 + numvert : 2 + 2 * numvert]]
            assert ids.min() >= 0
            g[2 + numvert : 2 + 2 * numvert] = ids
            new["mesh_graphadr"][i] = n_graph
            graphs.append(g)
            n_graph += len(g)
        # Vertices and their per-vertex polygon maps (local polygon ids, unchanged).
        rows = v0 + select
        verts.append(a["mesh_vert"][rows])
        for r in rows:
            pm_adrs.append(n_polymap)
            pm_nums.append(pm_num[r])
            polymaps.append(polymap[pm_adr[r] : pm_adr[r] + pm_num[r]])
            n_polymap += int(pm_num[r])
        new["mesh_vertnum"][i] = len(select)
        n_vert += len(select)
        # Polygons: normals and vertex lists (local vertex ids, renumbered).
        for p in polys:
            seg = remap[polyvert[pv_adr[p] : pv_adr[p] + pv_num[p]]]
            assert seg.min(initial=0) >= 0
            polynormals.append(a["mesh_polynormal"][p])
            polyvertnums.append(pv_num[p])
            polyvertadrs.append(n_polyvert)
            polyverts.append(seg)
            n_polyvert += len(seg)
        new["mesh_polynum"][i] = npoly
        n_poly += npoly
        kept_vertices[i] = (nv, len(select))

    def col(values, dtype, width=1):
        array = np.asarray(values, dtype).reshape(-1, width)
        return array if len(values) else np.zeros((0, width), dtype)

    i32 = np.int32
    o["mesh_vert"] = col(np.concatenate(verts) if verts else [], np.float32, 3)
    o["mesh_polymapadr"] = col(pm_adrs, i32)
    o["mesh_polymapnum"] = col(pm_nums, i32)
    o["mesh_polymap"] = col(np.concatenate(polymaps) if polymaps else [], i32)
    o["mesh_graph"] = col(np.concatenate(graphs) if graphs else [], i32)
    o["mesh_polynormal"] = col(polynormals, np.float64, 3)
    o["mesh_polyvertadr"] = col(polyvertadrs, i32)
    o["mesh_polyvertnum"] = col(polyvertnums, i32)
    o["mesh_polyvert"] = col(np.concatenate(polyverts) if polyverts else [], i32)
    for key, values in new.items():
        o[key] = values.reshape(-1, 1)
    # No faces, normals, texcoords or octrees for any mesh: running offsets for
    # the arrays the compiler addresses that way, -1 where it marks "none".
    zeros = np.zeros((nmesh, 1), i32)
    o["mesh_normaladr"], o["mesh_normalnum"] = zeros.copy(), zeros.copy()
    o["mesh_faceadr"], o["mesh_facenum"] = zeros.copy(), zeros.copy()
    o["mesh_texcoordadr"], o["mesh_texcoordnum"] = zeros - 1, zeros.copy()
    o["mesh_octadr"], o["mesh_octnum"] = zeros - 1, zeros.copy()
    o["mesh_normal"] = np.zeros((0, 3), np.float32)
    o["mesh_texcoord"] = np.zeros((0, 2), np.float32)
    for key in ("mesh_face", "mesh_facenormal", "mesh_facetexcoord"):
        o[key] = np.zeros((0, 3), i32)
    for key, width in (("oct_depth", 1), ("oct_child", 8)):
        o[key] = np.zeros((0, width), i32)
    for key, width in (("oct_aabb", 6), ("oct_coeff", 8)):
        o[key] = np.zeros((0, width), np.float64)

    # BVH: every body's tree (children are tree-local, nodes hold geom ids), then
    # one childless node per mesh with its original root box.
    body_adr, body_num = _col(a, "body_bvhadr"), _col(a, "body_bvhnum")
    parts = {k: [] for k in ("bvh_depth", "bvh_child", "bvh_nodeid", "bvh_aabb")}
    new_body_adr, n_bvh = np.full(s["nbody"], -1, i32), 0
    for b in range(s["nbody"]):
        if body_num[b] == 0:
            continue
        lo, hi = int(body_adr[b]), int(body_adr[b] + body_num[b])
        for k in parts:
            parts[k].append(a[k][lo:hi])
        new_body_adr[b] = n_bvh
        n_bvh += hi - lo
    mesh_bvhadr = _col(a, "mesh_bvhadr")
    new_mesh_bvhadr = np.arange(n_bvh, n_bvh + nmesh, dtype=i32)
    parts["bvh_depth"].append(np.zeros((nmesh, 1), i32))
    parts["bvh_child"].append(np.full((nmesh, 2), -1, i32))
    parts["bvh_nodeid"].append(np.full((nmesh, 1), -1, i32))
    root_box = np.zeros((nmesh, 6))
    has_tree = mesh_bvhadr >= 0
    root_box[has_tree] = a["bvh_aabb"][mesh_bvhadr[has_tree]]
    parts["bvh_aabb"].append(root_box)
    for k, chunks in parts.items():
        o[k] = np.concatenate(chunks).astype(a[k].dtype)
    n_bvh += nmesh
    o["body_bvhadr"] = new_body_adr.reshape(-1, 1)
    o["mesh_bvhadr"] = new_mesh_bvhadr.reshape(-1, 1)
    o["mesh_bvhnum"] = np.ones((nmesh, 1), i32)

    sizes = out.sizes
    sizes.update(
        nmeshvert=n_vert,
        nmeshnormal=0,
        nmeshtexcoord=0,
        nmeshface=0,
        nmeshgraph=n_graph,
        nmeshpoly=n_poly,
        nmeshpolyvert=n_polyvert,
        nmeshpolymap=n_polymap,
        noct=0,
        nbvh=n_bvh,
        nbvhstatic=n_bvh,
    )
    info = {
        "rule": "keep the convex hull (graph, polygons, hull vertices) of meshes "
        "used by colliding geoms; drop every other mesh's data, all faces, "
        "normals, texcoords, octrees and mesh BVHs (one childless node per mesh)",
        "collision_meshes": [int(i) for i in np.flatnonzero(keep)],
        "hull_compacted": hull_compacted,
        "collision_mesh_vertices": {int(k): list(v) for k, v in kept_vertices.items()},
        "meshes_emptied": int(nmesh - keep.sum()),
        "sizes_before": {k: s[k] for k in _MESH_SIZES},
        "sizes_after": {k: sizes[k] for k in _MESH_SIZES},
    }
    return out, info


_MESH_SIZES = (
    "nmeshvert", "nmeshnormal", "nmeshtexcoord", "nmeshface", "nmeshgraph",
    "nmeshpoly", "nmeshpolyvert", "nmeshpolymap", "nbvh", "nbvhstatic", "noct",
)  # fmt: skip


# -------------------------------------------------------------- verification


def load(raw, name="model.mjb"):
    return mujoco.MjModel.from_binary_path(name, {name: bytes(raw)})


def check_arrays(src, slim):
    """(changed, unexpected): arrays and sizes that differ, and those of them
    outside STRIPPABLE / the mesh-and-BVH sizes (must be empty)."""
    lay = mjb.layout()
    changed = [
        f.name
        for f in lay.fields
        if src.arrays[f.name].tobytes() != slim.arrays[f.name].tobytes()
    ]
    unexpected = [name for name in changed if name not in STRIPPABLE]
    for key in lay.sizes:
        if src.sizes[key] != slim.sizes[key] and key not in (*_MESH_SIZES, "nbuffer"):
            unexpected.append(key)
    if src.structs != slim.structs or src.header != slim.header:
        unexpected.append("structs/header")
    return changed, unexpected


def check_hulls(orig, slim, meshes):
    """(max |diff|, hull vertices checked): what collision reads of each mesh.

    Walks the hull graph and the polygons of every collision mesh in both
    models and compares the vertex coordinates they lead to, the graph
    topology, the polygon normals and the polygon map around each hull vertex.
    Any structural mismatch returns inf.
    """

    def seg(values, adr, num):
        return values[int(adr) : int(adr) + int(num)]

    worst, checked = 0.0, 0
    models = (orig, slim)
    for i in meshes:
        vo, vs = (seg(m.mesh_vert, m.mesh_vertadr[i], m.mesh_vertnum[i]) for m in models)
        if orig.mesh_graphadr[i] >= 0:
            adr = orig.mesh_graphadr[i]
            n = int(orig.mesh_graph[adr])
            size = 2 + 3 * n + 6 * int(orig.mesh_graph[adr + 1])
            ga, gb = (seg(m.mesh_graph, m.mesh_graphadr[i], size).copy() for m in models)
            ids_a, ids_b = ga[2 + n : 2 + 2 * n].copy(), gb[2 + n : 2 + 2 * n].copy()
            ga[2 + n : 2 + 2 * n] = gb[2 + n : 2 + 2 * n] = 0  # topology must match
            if not np.array_equal(ga, gb):
                return float("inf"), checked
            worst = max(worst, float(np.abs(vo[ids_a] - vs[ids_b]).max()))
            checked += n
            for ja, jb in zip(ids_a, ids_b, strict=True):
                ra, rb = orig.mesh_vertadr[i] + ja, slim.mesh_vertadr[i] + jb
                ma = seg(orig.mesh_polymap, orig.mesh_polymapadr[ra], orig.mesh_polymapnum[ra])
                mb = seg(slim.mesh_polymap, slim.mesh_polymapadr[rb], slim.mesh_polymapnum[rb])
                if not np.array_equal(ma, mb):
                    return float("inf"), checked
        for p in range(orig.mesh_polynum[i]):
            pa, pb = orig.mesh_polyadr[i] + p, slim.mesh_polyadr[i] + p
            la = seg(orig.mesh_polyvert, orig.mesh_polyvertadr[pa], orig.mesh_polyvertnum[pa])
            lb = seg(slim.mesh_polyvert, slim.mesh_polyvertadr[pb], slim.mesh_polyvertnum[pb])
            same_normal = np.array_equal(orig.mesh_polynormal[pa], slim.mesh_polynormal[pb])
            if len(la) != len(lb) or not same_normal:
                return float("inf"), checked
            worst = max(worst, float(np.abs(vo[la] - vs[lb]).max(initial=0)))
    return worst, checked


def _diff(a, b):
    """Max |a - b|; 0 for bit-identical arrays, inf where only one is NaN."""
    if a.tobytes() == b.tobytes():
        return 0.0
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    d = np.abs(np.where(nan_a | nan_b, 0.0, a - b))
    if np.any(nan_a != nan_b):
        return float("inf")
    return float(d.max(initial=0.0))


def _state(d):
    return np.concatenate([d.qpos, d.qvel, d.act, d.sensordata])


class Pair:
    """The original and the slim model, stepped side by side and compared."""

    def __init__(self, orig, slim, mesh_geoms):
        self.models = (orig, slim)
        self.datas = (mujoco.MjData(orig), mujoco.MjData(slim))
        self.mesh_geoms = np.asarray(mesh_geoms)
        self.max_diff = 0.0
        self.steps = 0
        self.contacts = 0
        self.mesh_contacts = 0
        self.mesh_pairs = set()

    def start(self, qpos, qvel, act=None):
        for m, d in zip(self.models, self.datas, strict=True):
            mujoco.mj_resetData(m, d)
            d.qpos[:], d.qvel[:] = qpos, qvel
            if act is not None:
                d.act[:] = act
            mujoco.mj_forward(m, d)
        self._compare()

    def step(self, ctrl):
        for m, d in zip(self.models, self.datas, strict=True):
            d.ctrl[:] = ctrl
            d.xfrc_applied[:] = 0
            mujoco.mj_step(m, d)
        self.steps += 1
        self._compare()

    def forward(self):
        for m, d in zip(self.models, self.datas, strict=True):
            mujoco.mj_forward(m, d)
        self._compare()

    def _compare(self):
        a, b = self.datas
        diff = max(
            _diff(_state(a), _state(b)),
            _diff(a.qacc, b.qacc),
            _diff(a.qacc_warmstart, b.qacc_warmstart),
        )
        if a.ncon != b.ncon:
            diff = float("inf")
        else:
            n = a.ncon
            ca, cb = a.contact, b.contact
            for key in ("dist", "pos", "frame", "geom"):
                diff = max(diff, _diff(np.asarray(getattr(ca, key)[:n], float), np.asarray(getattr(cb, key)[:n], float)))
            geoms = np.asarray(ca.geom[:n])
            hits = np.isin(geoms, self.mesh_geoms).any(axis=1)
            self.contacts += n
            self.mesh_contacts += int(hits.sum())
            for g in geoms[hits]:
                self.mesh_pairs.add((int(g[0]), int(g[1])))
        self.max_diff = max(self.max_diff, diff)

    def report(self):
        return {
            "physics_steps": self.steps,
            "max_abs_diff": self.max_diff,
            "contacts": self.contacts,
            "mesh_contacts": self.mesh_contacts,
            "mesh_contact_pairs": len(self.mesh_pairs),
        }


def _frame_state(bundle, reference, clip, frame):
    """qpos, qvel of a reference frame, placed as osl_deploy's reset places it."""
    reference.reset(clip, frame)
    ref = reference.reference()
    qpos = bundle.model.qpos0.copy()
    qvel = np.zeros(bundle.model.nv)
    q, v = bundle.root_qpos, bundle.root_qvel
    qpos[q : q + 3], qpos[q + 3 : q + 7] = ref["root_pos"], ref["root_quat"]
    qvel[v : v + 6] = reference.root_qvel(ref)
    qpos[bundle.qpos_ids], qvel[bundle.qvel_ids] = ref["joint_pos"], ref["joint_vel"]
    return qpos, qvel


def _clip(reference, name):
    return list(reference.names).index(name)


def _policy_runs(bundle):
    """Closed-loop osl_deploy runs (both policies, training actuator delay).

    Returns [(label, qpos0, qvel0, [(substep_ctrl, qpos_after_step), ...])],
    with only the substeps a failing step completed. The first run starts
    standing in the idle clip and asks for 1 m/s after 0.5 s, so it covers gait
    initiation; the others start mid-stride.
    """
    from osl_deploy.runner import Simulation

    sim = Simulation(bundle)
    names = list(sim.reference.names)
    runs = []
    plans = [
        ("idle, then walk 1.0 m/s", 0, 0, None, 300, (50, [1.0, 0.0, 0.0])),
        ("walk 1.0 m/s turning 0.5 rad/s", None, 40, [1.0, 0.0, 0.5], 200, None),
        ("walk backward 0.5 m/s", None, 25, [-0.5, 0.0, 0.0], 200, None),
    ]
    for label, clip, frame, twist, steps, request in plans:
        if clip is None:
            clip = sim.support.select(sim.support.serve(twist))
        sim.reset(clip=clip, frame=frame, twist=twist)
        qpos0, qvel0 = sim.data.qpos.copy(), sim.data.qvel.copy()
        trace = []
        for k in range(steps):
            if request is not None and k == request[0]:
                sim.request_twist(request[1])
            record = sim.step()
            done = record["substeps_completed"]
            trace.append((record["substep_ctrl"][:done].copy(), record["qpos"].copy()))
            if record["failed"]:
                label += f", failed: {sim.failure_reason}"
                break
        q = bundle.root_qpos
        moved = np.linalg.norm(sim.data.qpos[q : q + 2] - qpos0[q : q + 2])
        runs.append((f"{label} [{names[clip]}], root moved {moved:.2f} m", qpos0, qvel0, trace))
    return runs


def _ctrl_limits(model):
    low = np.where(model.actuator_ctrllimited, model.actuator_ctrlrange[:, 0], -1.0)
    high = np.where(model.actuator_ctrllimited, model.actuator_ctrlrange[:, 1], 1.0)
    return low, high


def verify_dynamics(bundle_dir, orig_raw, slim_raw, mesh_geoms, seed=0):
    """Original vs slim, compared on every physics step; returns numbers."""
    bundle = load_bundle(bundle_dir)
    orig, slim = load(orig_raw), load(slim_raw)
    # osl_deploy's runtime setting (Bundle); same on both, never in the file.
    for m in (orig, slim):
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    results = {}
    started = time.perf_counter()

    # 1. Policy controls recorded from osl_deploy's closed loop, replayed open
    #    loop on both models (each control step: 5 ticks, then mj_forward, as
    #    runner.Simulation.step does). The original replay must reproduce the
    #    closed-loop run exactly, which proves the replay is that walk.
    walk = Pair(orig, slim, mesh_geoms)
    replay_error = 0.0
    labels = []
    for label, qpos0, qvel0, trace in _policy_runs(bundle):
        labels.append(f"{label}: {len(trace)} steps")
        walk.start(qpos0, qvel0)
        for ctrl, qpos_after in trace:
            for row in ctrl:
                walk.step(row)
            walk.forward()
            replay_error = max(replay_error, _diff(walk.datas[0].qpos, qpos_after))
    results["policy_walk"] = {
        **walk.report(),
        "runs": labels,
        "replay_vs_closed_loop_max_abs": replay_error,
    }

    # 2. Falls: random muscle/motor controls (held one control step) and all
    #    controls off, from mid-stride frames; the body hits the floor.
    from osl_deploy.reference import make_reference

    reference = make_reference(bundle)
    rng = np.random.default_rng(seed)
    low, high = _ctrl_limits(orig)
    falls = Pair(orig, slim, mesh_geoms)
    for clip_name, frame, mode in (
        ("walk_fwd_vx+01000_vy+00000_wz+00000", 10, "random"),
        ("walk_fwd_vx+01000_vy+00000_wz+00000", 30, "zero"),
        ("walk_bck_vx-00500_vy+00000_wz+00000", 15, "random"),
        ("idle", 0, "zero"),
    ):
        qpos, qvel = _frame_state(bundle, reference, _clip(reference, clip_name), frame)
        falls.start(qpos, qvel)
        ctrl = np.zeros(orig.nu)
        for tick in range(1000):  # 2 s
            if mode == "random" and tick % 5 == 0:
                ctrl = rng.uniform(low, high)
            falls.step(ctrl)
    results["falls"] = falls.report()

    # 3. Penetrating poses: reference frames with perturbed joints, the root
    #    lowered and tilted into the floor; one mj_forward each. Run once with
    #    the bundle's options and once with multi-CCD switched on (the bundle
    #    disables it), which also reads the renumbered polygon data.
    lengths = reference.lengths
    jnt_range = orig.jnt_range[bundle.joint_ids]
    limited = orig.jnt_limited[bundle.joint_ids].astype(bool)
    q = bundle.root_qpos
    pose_states = []
    for _ in range(600):
        clip = int(rng.integers(len(lengths)))
        qpos, qvel = _frame_state(bundle, reference, clip, int(rng.integers(lengths[clip])))
        joints = qpos[bundle.qpos_ids] + rng.normal(0, 0.35, len(bundle.qpos_ids))
        joints = np.where(limited, np.clip(joints, jnt_range[:, 0], jnt_range[:, 1]), joints)
        qpos[bundle.qpos_ids] = joints
        qpos[q + 2] -= rng.uniform(0.0, 0.8)
        tilt = rng.normal(0, 0.6, 3)
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, tilt / (np.linalg.norm(tilt) + 1e-12), np.linalg.norm(tilt))
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, quat, qpos[q + 3 : q + 7])
        qpos[q + 3 : q + 7] = out
        pose_states.append((qpos, qvel + rng.normal(0, 0.5, len(qvel))))
    multiccd = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    flags = orig.opt.disableflags
    for key, value in (("penetrating_poses", flags), ("penetrating_poses_multiccd", flags & ~multiccd)):
        for m in (orig, slim):
            m.opt.disableflags = value
        poses = Pair(orig, slim, mesh_geoms)
        for qpos, qvel in pose_states:
            poses.start(qpos, qvel)
        results[key] = {**poses.report(), "poses": len(pose_states)}
    for m in (orig, slim):
        m.opt.disableflags = flags

    # 4. Geom distance (GJK/EPA and the plane-mesh path, margin 10 m) between
    #    each collision mesh and every geom that can touch it, in 60 of those
    #    poses: distances and witness points must match bit for bit.
    da, db = mujoco.MjData(orig), mujoco.MjData(slim)
    can = (orig.geom_contype | orig.geom_conaffinity) != 0
    partners = np.flatnonzero(can)
    worst, queries = 0.0, 0
    fa, fb = np.zeros(6), np.zeros(6)
    for qpos, _ in pose_states[:60]:
        for m, d in ((orig, da), (slim, db)):
            d.qpos[:] = qpos
            mujoco.mj_kinematics(m, d)
        for g in mesh_geoms:
            for h in partners:
                if h == g or orig.geom_bodyid[h] == orig.geom_bodyid[g]:
                    continue
                ra = mujoco.mj_geomDistance(orig, da, int(g), int(h), 10.0, fa)
                rb = mujoco.mj_geomDistance(slim, db, int(g), int(h), 10.0, fb)
                worst = max(worst, _diff(np.array([ra]), np.array([rb])), _diff(fa, fb))
                queries += 1
    results["geom_distance"] = {"queries": queries, "max_abs_diff": worst}
    results["seconds"] = round(time.perf_counter() - started, 1)
    return results


def wasm_reference(bundle_dir, slim_raw, steps=200, every=25):
    """Native trace of the slim model the WASM test replays and compares.

    Starts mid-stride; ctrl[t][i] = ((7 i + 3 t) mod 11) / 10, exact in both
    numpy and JavaScript. Returns DUETBIN1 arrays.
    """
    bundle = load_bundle(bundle_dir)
    from osl_deploy.reference import make_reference

    reference = make_reference(bundle)
    clip = _clip(reference, "walk_fwd_vx+01000_vy+00000_wz+00000")
    qpos, qvel = _frame_state(bundle, reference, clip, 10)
    m = load(slim_raw)
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_forward(m, d)
    index = np.arange(m.nu)
    checkpoints = {"qpos": [], "qvel": [], "act": [], "ncon": []}
    for t in range(steps):
        d.ctrl[:] = ((7 * index + 3 * t) % 11) / 10
        mujoco.mj_step(m, d)
        if (t + 1) % every == 0:
            for key in ("qpos", "qvel", "act"):
                checkpoints[key].append(getattr(d, key).copy())
            checkpoints["ncon"].append(d.ncon)
    return {
        "qpos0": qpos,
        "qvel0": qvel,
        "qpos": np.array(checkpoints["qpos"]),
        "qvel": np.array(checkpoints["qvel"]),
        "act": np.array(checkpoints["act"]),
        "ncon": np.array(checkpoints["ncon"], np.int32),
    }, {"steps": steps, "every": every, "ctrl": "((7*i + 3*t) % 11) / 10"}


# -------------------------------------------------------------------- export


def _jsonable(value):
    """Plain Python containers and scalars (numpy scalars and tuples converted)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary(verified):
    lines = []
    for key, value in verified.items():
        if isinstance(value, dict) and "max_abs_diff" in value:
            extra = ", ".join(f"{k} {v}" for k, v in value.items() if k not in ("max_abs_diff", "runs"))
            lines.append(f"[physics] {key}: max |diff| {value['max_abs_diff']} ({extra})")
    return "\n".join(lines)


def export(bundle_dir, out_dir, ctx):
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    raw = (bundle_dir / "model.mjb").read_bytes()
    src = mjb.parse(raw)
    if mjb.serialize(src) != raw:
        raise RuntimeError("model.mjb does not round-trip through duet_export.mjb")
    slim, info = strip(src)
    slim_raw = mjb.serialize(slim)

    # Static checks: loads, nothing but mesh/BVH data changed, hulls intact.
    orig_model, slim_model = load(raw), load(slim_raw)
    changed, unexpected = check_arrays(src, slim)
    if unexpected:
        raise RuntimeError(f"stripping changed non-mesh data: {unexpected}")
    counts = {key: int(getattr(slim_model, key)) for key in COUNTS}
    timestep = float(slim_model.opt.timestep)
    for key in COUNTS:
        if getattr(orig_model, key) != counts[key]:
            raise RuntimeError(f"{key} changed")
    hull_diff, hull_vertices = check_hulls(orig_model, slim_model, info["collision_meshes"])
    if hull_diff != 0:
        raise RuntimeError(f"collision hulls differ ({hull_diff})")
    # MuJoCo's own writer must reproduce the file: it is a canonical 3.11.0 .mjb.
    resaved = np.zeros(mujoco.mj_sizeModel(slim_model), np.uint8)
    mujoco.mj_saveModel(slim_model, None, resaved)
    if resaved.tobytes() != slim_raw:
        raise RuntimeError("mj_saveModel does not reproduce the slim model")
    del orig_model, slim_model

    mesh_geoms = [
        int(g)
        for g in np.flatnonzero(_col(src.arrays, "geom_type") == MESH)
        if _col(src.arrays, "geom_dataid")[g] in info["collision_meshes"]
    ]
    dynamics = verify_dynamics(bundle_dir, raw, slim_raw, mesh_geoms)
    worst = max(
        dynamics[k]["max_abs_diff"]
        for k in (
            "policy_walk", "falls", "penetrating_poses", "penetrating_poses_multiccd",
            "geom_distance",
        )
    )  # fmt: skip
    if worst != 0 or dynamics["policy_walk"]["replay_vs_closed_loop_max_abs"] != 0:
        raise RuntimeError(f"slim model dynamics differ: {dynamics}")

    nbytes = write_gzip(out_dir / FILE, slim_raw)
    arrays, meta = wasm_reference(bundle_dir, slim_raw)
    meta["physics_sha256"] = hashlib.sha256(slim_raw).hexdigest()
    wasm_ref = golden_path(out_dir, WASM_REF_NAME)
    write_container(wasm_ref, arrays, meta)
    print(f"[physics] native trace for physics.test.mjs: {wasm_ref}")

    stripped = {k: [info["sizes_before"][k], info["sizes_after"][k]] for k in _MESH_SIZES}
    print(_summary(dynamics))
    print(
        f"[physics] {len(raw):,} -> {len(slim_raw):,} bytes raw, {nbytes:,} gzipped; "
        f"verification {dynamics['seconds']} s"
    )
    return _jsonable({
        "physics": {
            "file": FILE,
            "bytes": nbytes,
            "raw_bytes": len(slim_raw),
            "source_bytes": len(raw),
            "mujoco_version": mujoco.__version__,
            "timestep": timestep,
            "counts": counts,
            "collision_geoms": mesh_geoms,
            "stripped": {
                "rule": info["rule"],
                "collision_meshes": info["collision_meshes"],
                "collision_mesh_vertices": info["collision_mesh_vertices"],
                "meshes_emptied": info["meshes_emptied"],
                "sizes": stripped,
            },
            "verified": {
                "load": "MjModel.from_binary_path (native 3.11.0)",
                "round_trip_identical": True,
                "mj_saveModel_reproduces_file": True,
                "non_mesh_arrays_identical": True,
                "arrays_changed": changed,
                "collision_hull_vertices_checked": hull_vertices,
                "collision_hull_max_abs_diff": hull_diff,
                **dynamics,
            },
            "notes": [
                "mj_ray / mjv_select report no hit on mesh geoms (their faces are gone)",
                "osl_deploy disables mjDSBL_AUTORESET at runtime; the file keeps the "
                "bundle's options unchanged",
            ],
        }
    })
