"""Shared helpers for the exporter sections.

Run every exporter with the `osl-deploy` Conda environment (MuJoCo 3.11.0, the
version of the WASM build in demo/vendor/mujoco), from the tools/ folder:

    conda run -n osl-deploy python -m duet_export --bundle BUNDLE --out ../demo/assets/stage2

The exporter only *reads* osl_deploy (installed editable from
../osl_deploy) and the bundle; it never changes either.
"""

import hashlib
import json
from pathlib import Path

# The stage-2 bundle the demo runs (chosen 2026-11-18): mjlab run
# 2026-11-15_20-24-39_s2_r6s7_B_decr025, checkpoint model_6498, default sensing
# (OSL input 500: actpack 250, ankle load cell 150, heel/toe contact 50,
# last_action 50; touch zones feet_extended). An mjlab checkout beside
# 01_projpegleg/; tools/test/paths.mjs names the same folder.
DEFAULT_BUNDLE = (
    Path(__file__).resolve().parents[4]
    / "mjlab/logs/stage1_imports/wandb_exports/2026-11-15_20-24-39_s2_r6s7_B_decr025"
    / "deploy/model_6498"
)

TOOLS = Path(__file__).resolve().parents[1]
# The asset folder demo/js/app.js loads (ASSETS_DIR in tools/test/paths.mjs).
DEMO_ASSETS = TOOLS.parent / "demo" / "assets" / "stage2"
# Test goldens an export writes with the assets they describe: physics.py's
# native trace for tools/test/physics.test.mjs and policy.py's reference
# outputs for tools/test/nn.test.mjs. For DEMO_ASSETS they live in these
# folders (the first file is tracked by git, the second is git-ignored); for
# any other --out folder they go into that folder, so an export into a scratch
# folder never rewrites the repository's goldens. tools/test/paths.mjs finds
# them by the same rule (DUET_ASSETS).
GOLDEN_DIRS = {
    "physics_wasm_ref.bin.gz": TOOLS / "test",
    "nn_reference.bin.gz": TOOLS / "traces",
}


def golden_path(out_dir, name):
    """Where an export into out_dir writes the test golden `name` (see GOLDEN_DIRS)."""
    if Path(out_dir).resolve() == DEMO_ASSETS.resolve():
        return GOLDEN_DIRS[name] / name
    return Path(out_dir) / name


def load_bundle(bundle_dir):
    """osl_deploy's Bundle for a bundle folder (manifest, model, ids, controllers)."""
    from osl_deploy.bundle import Bundle

    return Bundle(Path(bundle_dir))


def manifest(bundle_dir):
    return json.loads((Path(bundle_dir) / "manifest.json").read_text())


def manifest_sha256(bundle_dir):
    """The bundle's identity: the sha256 of its manifest.json (which lists every file's hash)."""
    return sha256(Path(bundle_dir) / "manifest.json")


def bundle_label(bundle_dir, m=None):
    """'<training run>/<bundle folder>', e.g. '2026-11-15_20-24-39_s2_r6s7_B_decr025/model_6498'.

    The folder name alone can be as bare as 'model_6498'; without a recorded run
    it is all there is.
    """
    m = manifest(bundle_dir) if m is None else m
    folder = Path(bundle_dir).resolve().name
    run = (m.get("run") or {}).get("run")
    return f"{run}/{folder}" if run else folder


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
