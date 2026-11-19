"""Export one osl_deploy bundle into the browser demo's asset folder.

    cd tools
    conda run -n osl-deploy python -m duet_export --out ../demo/assets/stage2

The bundle defaults to common.DEFAULT_BUNDLE (the stage-2 bundle the demo runs);
`--bundle FOLDER` picks another. Each section module exposes
`export(bundle_dir, out_dir, ctx) -> dict`; the dict is merged into the top
level of `<out>/demo.json` (each section owns its own keys, see
tools/DESIGN.md). `--only a,b` re-runs some sections and keeps the other keys
of an existing demo.json.

demo.json "sections" records, per exported section, the sha256 of the bundle
manifest it was built from, and "source" names that bundle. An `--only` run
whose bundle differs from the sections it would keep is refused: R2's and B's
models share every id and count, so B's policies next to R2's physics.mjb.gz
load and run, but the OSL heel contact reads 0 N. `--allow-mixed` overrides the
refusal (for experiments only; tools/test/contract.test.mjs fails on such a
folder).

Every run also rewrites demo.json's "versions": {file name: the first 12 hex
digits of its sha256} for every asset file demo.json references, whichever
sections ran. The demo fetches each asset as `<file>?v=<version>`, so after a
deploy a browser cannot pair the new demo.json with a cached old file.

The physics and policies sections also write a test golden each
(physics_wasm_ref.bin.gz, nn_reference.bin.gz; common.golden_path): into
tools/test/ (tracked) and tools/traces/ (git-ignored) when --out is the demo's
own folder (common.DEMO_ASSETS), into the --out folder otherwise.
"""

import argparse
import fcntl
import importlib
import json
import sys
from pathlib import Path

from .common import DEFAULT_BUNDLE, bundle_label, manifest, manifest_sha256, sha256

# section name -> module (in this package) with export(bundle_dir, out_dir, ctx)
SECTIONS = {
    "physics": "physics",  # physics.mjb.gz: the slim physics model
    "visual": "visual",  # visual.bin.gz: meshes + per-geom render data
    "policies": "policy",  # human.bin.gz, osl.bin.gz: MLP weights
    "sim": "simcfg",  # reference.bin.gz + everything SimCore needs
}
# The demo.json keys each section writes.
SECTION_KEYS = {
    "physics": ("physics",),
    "visual": ("visual",),
    "policies": ("policies",),
    "sim": ("sim", "reference", "ui"),
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


def existing_sections(demo):
    """{section: manifest sha256 or None (unknown)} for the sections demo.json holds.

    A demo.json written before "sections" existed counts every section it holds
    as built from its source.manifest_sha256.
    """
    present = [name for name, keys in SECTION_KEYS.items() if all(k in demo for k in keys)]
    recorded = demo.get("sections")
    if isinstance(recorded, dict):
        return {name: recorded.get(name) for name in present}
    legacy = (demo.get("source") or {}).get("manifest_sha256")
    return {name: legacy for name in present}


def mixed_sections(demo, wanted, sha):
    """The sections this run keeps from demo.json that another bundle built."""
    return {
        name: other
        for name, other in existing_sections(demo).items()
        if name not in wanted and other != sha
    }


def mixing_message(mixed, demo, bundle, sha):
    def short(value):
        return value[:12] if value else "unknown"

    origin = (demo.get("source") or {}).get("bundle", "an unknown bundle")
    kept = ", ".join(f"{name} ({short(value)})" for name, value in sorted(mixed.items()))
    return (
        f"refusing to mix bundles in one asset folder: demo.json keeps the sections {kept} "
        f"from manifest(s) other than this run's bundle {bundle_label(bundle)} "
        f"(manifest {short(sha)}; demo.json's source is {origin}). The models of two bundles "
        "can share every id and count, so a mix loads and runs but is wrong without any "
        "error (R2's physics.mjb.gz under B's policies: the OSL heel contact reads 0 N). "
        "Export every section (omit --only), or pass --allow-mixed for an experiment."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="duet_export", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=",".join(SECTIONS))
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="keep sections exported from another bundle (refused by default)",
    )
    args = parser.parse_args(argv)
    bundle, out = args.bundle.resolve(), args.out.resolve()
    wanted = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = set(wanted) - set(SECTIONS)
    if unknown:
        parser.error(f"unknown sections {sorted(unknown)}; choose from {list(SECTIONS)}")

    m = manifest(bundle)
    sha = manifest_sha256(bundle)
    path = out / "demo.json"
    # Refuse a mix before any section writes its files.
    before = json.loads(path.read_text()) if path.exists() else {}
    mixed = mixed_sections(before, wanted, sha)
    if mixed and not args.allow_mixed:
        parser.error(mixing_message(mixed, before, bundle, sha))
    out.mkdir(parents=True, exist_ok=True)

    ctx = {"manifest": m}
    sections = {}
    for name in wanted:
        module = importlib.import_module(f".{SECTIONS[name]}", __package__)
        sections[name] = module.export(bundle, out, ctx)
        print(f"[duet_export] {name}: keys {sorted(sections[name])}")

    # Merge under an exclusive lock: several sections may be exported at once.
    with open(out / ".demo.json.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        demo = json.loads(path.read_text()) if path.exists() else {}
        mixed = mixed_sections(demo, wanted, sha)  # again: another run may have merged meanwhile
        if mixed and not args.allow_mixed:
            raise SystemExit(
                f"duet_export: {mixing_message(mixed, demo, bundle, sha)} The files of the "
                f"sections {', '.join(wanted)} were written to {out}, but demo.json was not updated."
            )
        recorded = existing_sections(demo)
        demo["format"] = "duetmimic-demo/1"
        demo["source"] = {
            "bundle": bundle_label(bundle, m),
            "kind": m.get("kind"),
            "run": m.get("run", {}).get("run"),
            "checkpoint": m.get("run", {}).get("checkpoint"),
            "mujoco_version": m.get("mujoco_version"),
            "manifest_sha256": sha,
        }
        for section in sections.values():
            demo.update(section)
        demo["sections"] = {
            name: sha if name in sections else recorded[name]
            for name in SECTIONS
            if name in sections or name in recorded
        }
        demo["versions"] = asset_versions(demo, out)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(demo, indent=1) + "\n")
        tmp.replace(path)
    if mixed:
        print(
            f"[duet_export] WARNING: --allow-mixed: sections {sorted(mixed)} come from another "
            "bundle than source names",
            file=sys.stderr,
        )
    print(f"[duet_export] source: {demo['source']['bundle']} (manifest {sha[:12]})")
    print(f"[duet_export] versions: {demo['versions']}")
    print(f"[duet_export] {path}")


if __name__ == "__main__":
    main()
