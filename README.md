# DuetMimic project page

Project page for **DuetMimic: Control Guided Reinforcement Learning for
Musculoskeletal Human–Prosthesis Locomotion**, served by GitHub Pages from the
repository `duetmimic/duetmimic.github.io` at <https://duetmimic.github.io/>.

- `index.html`: the landing page: title, authors, links, the looping teaser
  video, the embedded interactive demo, the abstract and BibTeX. Placeholders
  still to fill are marked `<!-- TODO -->`; see [Before publishing](#before-publishing).
- `demo/`: the interactive demo. It runs the stage-2 split-network unicycle
  policy (osl_deploy bundle `stage2_R2_model_6898`) in the browser. A human
  policy drives the 339 muscles and an onboard policy drives the Open-Source
  Leg's knee and ankle. You steer it with W/A/S/D, or with the on-screen pad on
  a touch screen. Physics is the official MuJoCo 3.11.0 WebAssembly build.
  Rendering is three.js, and inference uses WebGPU where available, otherwise
  the CPU. The demo is deliberately stripped down: it keeps only what the two
  policies need to walk, not the full osl_deploy runtime.

There is no build step. Everything is static HTML, CSS and ES modules.
`tools/DESIGN.md` documents the layout, asset formats and module contracts.

## Local preview

Serve the repository root over HTTP. Opening the files directly (`file://`)
does not work, because the demo uses ES modules, a Web Worker and `fetch`.

```sh
cd duetmimic.github.io
python3 -m http.server 8000
```

Then open <http://localhost:8000>. `localhost` counts as a secure context, so
WebGPU is available there in browsers that support it.

## Regenerating the demo assets

The files in `demo/assets/r2/` are written by the exporter in
`tools/duet_export/`. It reads the osl_deploy bundle and never modifies it.
Run it in the `osl-deploy` Conda environment (MuJoCo 3.11.0, numpy, scipy,
onnxruntime):

```sh
source ~/miniconda3/etc/profile.d/conda.sh && conda activate osl-deploy
cd tools
python -m duet_export --out ../demo/assets/r2                 # every section
python -m duet_export --out ../demo/assets/r2 --only sim      # one section
```

The default bundle is `../osl_deploy/bundles/stage2_R2_model_6898`: an
osl_deploy checkout beside this repository. Pass `--bundle` to use another one. Sections
are `physics`, `visual`, `policies` and `sim`. Each one rewrites its own files
and its own keys of `demo.json`. Every run, whichever sections it exports,
also rewrites `demo.json`'s `versions`: a short content hash of every asset
file, which the demo uses for cache busting (see
[Caching](#size-bandwidth-and-caching)). See `tools/DESIGN.md` for what each
file holds.

## Developer checks

Developer checks live in `tools/test/`. Run `npm ci` once in `tools/`, then
`npm test` (it runs `node --test --test-concurrency=1 "test/*.test.mjs"`).
`tools/node_modules/` is not committed. The browser checks are opt-in; they
need Playwright's browsers (`npx playwright install chromium`, or
`chromium firefox webkit`) and `PLAYWRIGHT_BROWSERS_PATH` if they live
somewhere other than Playwright's default folder. All commands run from
`tools/`:

- `node --test test/e2e_session.test.mjs`: the demo's simulation side
  (DemoSession) in Node on the real assets: a scripted WASD schedule, the
  command shaping, a fixed-seed key-masher, falls, cache-busting versions and
  fail-fast loading. Inference uses WebGPU through Dawn when it finds a hardware
  adapter; `E2E_BACKEND=cpu|webgpu` forces one, `E2E_MASH_SECONDS` sets the
  masher's length.
- `DUET_BROWSER_TESTS=1 [E2E_GPU=1] [E2E_SHOTS_DIR=<folder>] [E2E_PORT=8741] node --test test/e2e_browser.test.mjs`
  drives the real demo page in headless Chromium for about a minute. Add
  `E2E_GPU=1` for a hardware GPU (Chromium flags
  `--use-angle=vulkan --enable-unsafe-webgpu --enable-features=Vulkan`).
- `DUET_COMPAT_TESTS=1 [COMPAT_ENGINES=chromium,firefox,webkit] [COMPAT_GPU=1] [COMPAT_WEBKIT_LIBS=<dir>] [COMPAT_SHOTS_DIR=<folder>] [COMPAT_PORT=8831] node --test test/compat_browsers.test.mjs`
  runs the project page and the demo in Chromium, Firefox and WebKit, each
  also in phone emulation. An engine that is not installed or cannot start
  is skipped with the reason. Playwright's WebKit needs system libraries
  that Ubuntu 24.04 lacks (`libavif16`, `libgav1-1`, `libyuv0`); install
  them once with `sudo npx playwright install-deps webkit`. Without sudo,
  `apt-get download` the three packages, extract them with `dpkg-deb -x`
  into one folder and pass its `usr/lib/x86_64-linux-gnu` as
  `COMPAT_WEBKIT_LIBS`. The test preloads them and skips Playwright's host
  check. The three engines take about two minutes in all.
- `[NN_TEST_HARDWARE_GPU=1] node --test test/nn_browser.test.mjs`: the MLP
  engine in a module worker in headless Chromium, WebGPU against the CPU.
  `tools/test/nn_browser.html` also runs by hand in any browser, for example
  Safari on a Mac, served from the repository root.
- `DUET_BROWSER_TESTS=1 [RENDER_SHOTS_DIR=<folder>] node --test test/render_shots.test.mjs`:
  renderer screenshots of fixture poses.
- The trace tests in `test/sim.test.mjs` skip until you regenerate their
  ground truth with `python -m duet_export.trace` (git-ignored, 22 MB).

### Page media

`static/media/` holds the landing-page media:

- `three_demos.mp4` (1920x1080 H.264, about 2.2 MB) was transcoded from the
  osl_deploy figure render
  `logs/figure/three_demos/video/three_demos.mp4` (3840x2160):

  ```sh
  nice -n 19 ffmpeg -threads 4 -i three_demos.mp4 -an -vf scale=1920:1080:flags=lanczos \
    -c:v libx264 -threads 4 -profile:v high -level:v 4.1 -pix_fmt yuv420p \
    -preset slow -crf 22 -movflags +faststart static/media/three_demos.mp4
  ```

- `three_demos_poster.jpg` is the video's first frame, so playback starts
  without a jump.
- `demo_poster.jpg` (1600x900) is a screenshot of the running demo at its
  default camera, taken with a hardware GPU in headless Chromium. It predates
  the current muscle colours (paler at rest, deeper red when active); reshoot
  it at the same path if the difference shows.
- `og.jpg` (1200x630, about 80 KB) is the social card that `og:image` and
  `twitter:image` point to. It was composed as an HTML page in headless
  Chromium and saved as a JPEG at quality 85. `demo_poster.jpg` is scaled to
  1280x720 and placed at (10, -62) px, so the walker stands right of centre. A
  left-to-right fade of the page band colour `#f5f5f7` runs from 96 % opacity
  through 88 % at 30 % of the width to 0 at 56 %. Over it, 72 px from the left
  and centred vertically: the kicker "Interactive demo" (Inter 600, 19 px,
  uppercase, `#c2410c`), "DuetMimic" (Inter Tight 700, 112 px, `#1d1d1f`),
  "Musculoskeletal human–prosthesis locomotion" (Inter Tight 500, 34 px,
  `#3a3a3c`, broken after "Musculoskeletal") and a 64x5 px orange rule. Remake
  it when the poster changes.

`static/img/` holds `favicon.svg` and `apple-touch-icon.png` (180x180): the
favicon rendered full-bleed in headless Chromium, with a square background
because iOS rounds the corners itself and shows transparency as black.

## Before publishing

- Fill every `<!-- TODO -->` placeholder in `index.html`: venue, authors and
  affiliations, the abstract and the BibTeX entry.
- Replace the placeholder links. Each of the four greyed-out link pills (Paper,
  arXiv, Code, Video) needs an `href`, `target="_blank" rel="noopener"`, and
  no `aria-disabled`, `is-soon` class or "(coming soon)" text. Update or delete
  the "coming soon" line under them.
- Confirm the licences of the assets the demo redistributes, then credit them
  in the footer (a commented-out line in `index.html` shows where):
  - the musculoskeletal model, MyoAssist's full-body model built with `myo_sim`;
  - the Open-Source Leg knee-ankle geometry (`OpenSourceLeg_KA_L1`).

  Both reach the demo through the bundle's `model.mjb`; the header of mjlab's
  `osl_fullbody.xml` records how they were combined.
- If the exported assets changed, re-measure the "About 31 MB download" note
  in `index.html` (the demo's own loader computes its figure from
  `demo.json`).
- Optionally reshoot `demo_poster.jpg` with the current muscle colours and
  remake `og.jpg` from it (see [Page media](#page-media)).

## Deploying to GitHub Pages

The site is the root site of the GitHub organization `duetmimic`: a repository
named exactly `duetmimic.github.io` in that organization is served at
<https://duetmimic.github.io/>. Every page and asset path is relative, so the
same files would also work as a project site under a sub-path. Only the
social-card metadata (`canonical`, `og:url`, `og:image`, `twitter:image` in
`index.html` and `demo/index.html`) holds absolute URLs; update them if the
address ever changes (for example to a custom domain).

1. Commit, then create the public repository and push `main` (the GitHub CLI
   sets `origin` for you; `--push` needs at least one commit):

   ```sh
   git add -A
   git commit -m "DuetMimic project page with in-browser WASD demo"
   gh repo create duetmimic/duetmimic.github.io --public --source=. --remote=origin --push
   ```

2. Turn on Pages from `main`, folder `/ (root)`: **Settings → Pages → Build and
   deployment → Deploy from a branch**, or from the command line:

   ```sh
   gh api -X POST repos/duetmimic/duetmimic.github.io/pages -f "source[branch]=main" -f "source[path]=/"
   ```

   GitHub may already have switched Pages on for a `<name>.github.io`
   repository; a 409 reply ("already exists") just means that.

3. The first build takes a few minutes (repository **Actions** tab, `pages build
   and deployment`). Later pushes to `main` redeploy automatically.

`.nojekyll` is present, so Pages serves the files as they are, with no Jekyll
processing.

The user makes every commit and push in this repository. Tools and agents
prepare the working tree and stop there.

## Size, bandwidth and caching

- GitHub rejects files over **100 MB** and warns above 50 MB. Every asset here
  must stay below that limit. The exporter writes gzip-compressed containers
  for this reason.
- **Git LFS does not work for the site.** Pages serves the LFS pointer file,
  not its content, so assets must be ordinary Git blobs.
- Pages sites should stay under 1 GB, and bandwidth has a **soft limit of about
  100 GB per month**. One demo load downloads about 31 MB, 25 MB of it the
  human policy's weights, which is roughly 3,000 loads per 100 GB. The landing
  page therefore fetches nothing from `demo/` until a visitor presses **Start
  interactive demo**. The landing page itself weighs about 2.5 MB: the teaser
  video, two posters, CSS and JS. `og.jpg` and `apple-touch-icon.png` are only
  fetched by link previews and home-screen shortcuts.
- Pages sends `Cache-Control: max-age=600`, so browsers reuse a file for up
  to 10 minutes without asking, then revalidate it. Unchanged files come back
  as `304 Not Modified` and are not downloaded again.
- **Cache busting.** Within those 10 minutes a returning visitor could combine
  a new `demo.json` with an old cached asset. The demo prevents that:
  `demo.json` is always revalidated (`cache: 'no-cache'`), and every asset
  is fetched as `<file>?v=<version>`, the version being the first 12 hex
  digits of the file's sha256 from `demo.json`'s `versions`. A new export
  therefore changes the URL of every file it changes. The JavaScript modules
  and `demo/vendor/` are not versioned, so code can lag a deploy by up to 10
  minutes.

## Licenses and credits

- `demo/vendor/mujoco/`: MuJoCo 3.11.0 WebAssembly bindings by Google DeepMind,
  from the npm package `@mujoco/mujoco@3.11.0`, under the **Apache License 2.0**.
  The npm package ships no license file, so `demo/vendor/mujoco/LICENSE` holds
  the license text copied from the MuJoCo repository at tag 3.11.0.
- `demo/vendor/three/`: three.js 0.187.0 under the **MIT License**. The license
  text is in `demo/vendor/three/LICENSE`.
- Fonts (Inter, Inter Tight, JetBrains Mono) are loaded from Google Fonts under
  the SIL Open Font License 1.1 and are not redistributed here. `og.jpg` has
  Inter and Inter Tight rendered into it, which the licence allows.
- The demo assets come from the policy bundle's `model.mjb`, which includes the
  musculoskeletal model and the Open-Source Leg geometry. Their licences are
  still to be confirmed and credited (see [Before publishing](#before-publishing)).
