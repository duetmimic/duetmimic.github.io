"""Export one osl_deploy bundle into the browser demo's asset folder.

    cd tools
    conda run -n osl-deploy python -m duet_export \
        --bundle ../../osl_deploy/bundles/stage2_R2_model_6898 --out ../demo/assets/r2

Each section module exposes `export(bundle_dir, out_dir, ctx) -> dict`; the dict
is merged into the top level of `<out>/demo.json` (each section owns its own
keys, see tools/DESIGN.md). `--only a,b` re-runs some sections and keeps the
other keys of an existing demo.json.

Every run also rewrites demo.json's "versions": {file name: the first 12 hex
digits of its sha256} for every asset file demo.json references, whichever
sections ran. The demo fetches each asset as `<file>?v=<version>`, so after a
deploy a browser cannot pair the new demo.json with a cached old file.
"""

import argparse
import fcntl
import importlib
import json
import sys
from pathlib import Path

from .common import DEFAULT_BUNDLE, manifest, sha256

# section name -> module (in this package) with export(bundle_dir, out_dir, ctx)
SECTIONS = {
    "physics": "physics",  # physics.mjb.gz: the slim physics model
    "visual": "visual",  # visual.bin.gz: meshes + per-geom render data
    "policies": "policy",  # human.bin.gz, osl.bin.gz: MLP weights
    "sim": "simcfg",  # reference.bin.gz + everything SimCore needs
}
VERSION_DIGITS = 12


def referenced_files(demo):
    """Every "file" value in demo.json, at any depth: the asset files it names."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "file" and isinstance(value, str):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk({key: value for key, value in demo.items() if key != "versions"})
    return sorted(found)


def asset_versions(demo, out):
    """{file name: first VERSION_DIGITS hex of its sha256} for referenced files."""
    versions = {}
    for name in referenced_files(demo):
        path = out / name
        if path.is_file():
            versions[name] = sha256(path)[:VERSION_DIGITS]
        else:
            print(f"[duet_export] warning: demo.json names {name}, which is not in {out}", file=sys.stderr)
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(prog="duet_export", description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=",".join(SECTIONS))
    args = parser.parse_args(argv)
    bundle, out = args.bundle.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = set(wanted) - set(SECTIONS)
    if unknown:
        parser.error(f"unknown sections {sorted(unknown)}; choose from {list(SECTIONS)}")

    m = manifest(bundle)
    ctx = {"manifest": m}
    sections = {}
    for name in wanted:
        module = importlib.import_module(f".{SECTIONS[name]}", __package__)
        sections[name] = module.export(bundle, out, ctx)
        print(f"[duet_export] {name}: keys {sorted(sections[name])}")

    # Merge under an exclusive lock: several sections may be exported at once.
    path = out / "demo.json"
    with open(out / ".demo.json.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        demo = json.loads(path.read_text()) if path.exists() else {}
        demo["format"] = "duetmimic-demo/1"
        demo["source"] = {
            "bundle": bundle.name,
            "kind": m.get("kind"),
            "run": m.get("run", {}).get("run"),
            "checkpoint": m.get("run", {}).get("checkpoint"),
            "mujoco_version": m.get("mujoco_version"),
            "manifest_sha256": sha256(bundle / "manifest.json"),
        }
        for section in sections.values():
            demo.update(section)
        demo["versions"] = asset_versions(demo, out)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(demo, indent=1) + "\n")
        tmp.replace(path)
    print(f"[duet_export] versions: {demo['versions']}")
    print(f"[duet_export] {path}")


if __name__ == "__main__":
    main()
