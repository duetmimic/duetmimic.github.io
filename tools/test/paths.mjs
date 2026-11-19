// Where the developer checks find the exported assets, the original bundle and
// the git-ignored traces: one place for every test in tools/test (not a test
// itself; `npm test` runs test/*.test.mjs only). The pages the browser tests
// serve (render_pose.html, nn_browser.html) take the folder as ?assets=.
//
//   DUET_ASSETS=<folder>  an exported asset folder (default demo/assets/stage2,
//                         the folder demo/js/app.js loads)
//   DUET_BUNDLE=<folder>  the original bundle the assets were exported from
//                         (default: duet_export/common.py DEFAULT_BUNDLE, the
//                         stage-2 bundle B in an mjlab checkout beside
//                         01_projpegleg/; keep the two in step)
//   DUET_TRACE=<file>     the osl_deploy trace for sim.test.mjs (default
//                         tools/traces/trace.bin.gz, python -m duet_export.trace)
//
// The goldens physics.test.mjs and nn.test.mjs compare against follow the
// asset folder, by the rule the exporter writes them with
// (duet_export/common.py golden_path): for the demo's own folder they are
// tools/test/physics_wasm_ref.bin.gz (tracked) and
// tools/traces/nn_reference.bin.gz (git-ignored); for any other folder
// (DUET_ASSETS) they are the files of those names inside it, written by
// `python -m duet_export --out <that folder>`.

import { realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const REPO = new URL('../../', import.meta.url);
export const TOOLS = new URL('../', import.meta.url);
export const TRACES = new URL('traces/', TOOLS);
// The demo's asset folder, relative to the repository root (the served site).
export const ASSETS_DIR = 'demo/assets/stage2/';
const DEFAULT_BUNDLE = '../../mjlab/logs/stage1_imports/wandb_exports/2026-11-15_20-24-39_s2_r6s7_B_decr025/deploy/model_6498/';

const folder = (dir) => pathToFileURL(`${path.resolve(dir)}${path.sep}`);
// A file URL as an absolute path with symbolic links resolved where it exists
// (Python's Path.resolve(), which the exporter compares with).
const real = (url) => {
  const file = fileURLToPath(url);
  try {
    return realpathSync(file);
  } catch {
    return path.resolve(file);
  }
};

export const ASSETS = process.env.DUET_ASSETS ? folder(process.env.DUET_ASSETS) : new URL(ASSETS_DIR, REPO);
export const BUNDLE = process.env.DUET_BUNDLE ? folder(process.env.DUET_BUNDLE) : new URL(DEFAULT_BUNDLE, REPO);
export const TRACE = process.env.DUET_TRACE ? pathToFileURL(path.resolve(process.env.DUET_TRACE)) : new URL('trace.bin.gz', TRACES);

// Whether ASSETS is the demo's own folder; its goldens then live in tools/.
export const IS_DEMO_ASSETS = real(ASSETS) === real(new URL(ASSETS_DIR, REPO));
export const PHYSICS_REF = IS_DEMO_ASSETS
  ? new URL('test/physics_wasm_ref.bin.gz', TOOLS)
  : new URL('physics_wasm_ref.bin.gz', ASSETS);
export const NN_REF = IS_DEMO_ASSETS ? new URL('nn_reference.bin.gz', TRACES) : new URL('nn_reference.bin.gz', ASSETS);

// A file of the asset folder (or of `base`) as a file-system path.
export const assetPath = (name, base = ASSETS) => fileURLToPath(new URL(name, base));

// A file or folder URL as a URL path on a server whose root is `root`
// (default: the repository), e.g. '/tools/test/physics_wasm_ref.bin.gz'
// (folders end in '/'); null when it lies outside the root.
export function served(url, root = REPO) {
  const rel = path.relative(fileURLToPath(root), fileURLToPath(url));
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null;
  const href = rel ? `/${rel.split(path.sep).join('/')}` : '';
  return url.href.endsWith('/') ? `${href}/` : href || '/';
}

// The asset folder as a URL path on a server whose root is `root` (default:
// the repository), e.g. '/demo/assets/stage2/'; null when DUET_ASSETS lies
// outside it.
export const servedAssets = (root = REPO) => served(ASSETS, root);
