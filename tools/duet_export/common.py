"""Shared helpers for the exporter sections.

Run every exporter with the `osl-deploy` Conda environment (MuJoCo 3.11.0, the
version of the WASM build in demo/vendor/mujoco), from the tools/ folder:

    conda run -n osl-deploy python -m duet_export --bundle BUNDLE --out ../demo/assets/r2

The exporter only *reads* osl_deploy (installed editable from
../osl_deploy); it never changes that repository.
"""

import hashlib
import json
from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).resolve().parents[3] / "osl_deploy/bundles/stage2_R2_model_6898"


def load_bundle(bundle_dir):
    """osl_deploy's Bundle for a bundle folder (manifest, model, ids, controllers)."""
    from osl_deploy.bundle import Bundle

    return Bundle(Path(bundle_dir))


def manifest(bundle_dir):
    return json.loads((Path(bundle_dir) / "manifest.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
