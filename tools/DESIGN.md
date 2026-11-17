# DuetMimic web demo: design notes

A static GitHub Pages site, served from the repo `duetmimic/duetmimic.github.io` at https://duetmimic.github.io/. Its interactive demo runs the
stage-2 split-network unicycle policy (osl_deploy bundle
`stage2_R2_model_6898`: a 339-muscle human actor plus an onboard OSL actor) in
the browser. You steer it with WASD. **Scope: a stripped-down demo you can pilot.**
It is not a port of osl_deploy. It keeps only what the two policies need to
walk: their observations, the gait-library reference with blending and rebase,
the training actuator delay, and the OSL sensor history. It drops stress
tests, async mode, telemetry and exact parity suites. Physics may differ from
native MuJoCo in the last bits.

## Stack
- Physics: official MuJoCo **3.11.0** WebAssembly build, vendored in
  `demo/vendor/mujoco/`. It is the same version as the bundle's `model.mjb`;
  the `.mjb` format is version-locked. Loader: `demo/js/lib/mujoco.js`.
- Inference: a small custom MLP engine (`demo/js/nn/`, entry `nn/index.js`).
  The human net uses WebGPU when a hardware adapter passes a self-test: about
  0.3-2.5 ms per inference including readback. Otherwise it falls back to the
  CPU: plain JS, about 9-10 ms per inference in Chromium, Firefox and WebKit
  (two int8 kernels, the faster picked at load; see Findings), so a little
  below real time once physics is added. The OSL net always runs on the CPU
  (about 0.2 ms). Human weights are int8 per output row (0.0019 RMS action
  error against fp32); a 16 s closed-loop walk shows no fall and the same site
  error as fp32. The OSL weights stay fp32.
- Physics model: `physics.mjb.gz` is 463 KB, down from 103 MB. It keeps only
  the collision meshes' convex hulls and replaces the mesh BVHs with
  single-node stubs. Its dynamics are bit-identical to the bundle model. It
  loads in about 0.6 s in WASM; the full model takes 25 s. Rays (mj_ray,
  picking) miss mesh geoms.
- Rendering: three.js 0.187.0, vendored in `demo/vendor/three/`, on the main
  thread. Physics and inference run in a module Web Worker (`demo/js/worker.js`).
  Pages load three through an import map. From `demo/index.html` it is
  `{"imports": {"three": "./vendor/three/three.module.js", "three/addons/": "./vendor/three/addons/"}}`.
  Only `addons/controls/OrbitControls.js` is vendored so far; copy any other
  addon from `tools/node_modules/three/examples/jsm/`, keeping its sub-path.
- No build step: plain ES modules, served as static files.

## Repository layout
```
index.html, static/{css,js}/               landing page (title, looping video, embedded demo)
static/media/, static/img/                 teaser video and posters, social card og.jpg; favicon, apple-touch-icon
demo/index.html, demo/css/demo.css         full-window demo page (?embed=1 when iframed)
demo/js/app.js                             main thread: UI, keyboard and touch input, renderer, worker client
demo/js/worker.js                          worker: message protocol + real-time pacing around DemoSession
demo/js/session.js                         DemoSession (no DOM): loading, SimCore, command shaping, fall handling, HUD status
demo/js/lib/{container,mujoco}.js          shared loaders (DUETBIN1 containers, MuJoCo wasm)
demo/js/sim/                               SimCore: reference, observations, delay, step loop, commands
demo/js/nn/                                MLP engine (CPU + WebGPU)
demo/js/render/                            frame extraction (worker side) + three.js scene (main)
demo/vendor/{mujoco,three}/                pinned third-party builds
demo/assets/r2/                            exported assets (below)
tools/duet_export/                          Python exporter (osl-deploy Conda env), reads osl_deploy
tools/test/                                Node checks (node --test), browser smoke tests (Playwright)
```

## Assets (`demo/assets/r2/`, written by `python -m duet_export`)
| file | owner module | content |
| --- | --- | --- |
| `demo.json` | all sections | config; top-level keys owned per section (below) |
| `physics.mjb.gz` | `physics.py` | slim physics model: the bundle model with the mesh data that physics never reads stripped out. Every id (bodies, geoms, sites, tendons, actuators, sensors) is unchanged |
| `visual.bin.gz` | `visual.py` | render meshes (decimated), per-geom render info |
| `human.bin.gz`, `osl.bin.gz` | `policy.py` | MLP weights (DUETBIN1) |
| `reference.bin.gz` | `simcfg.py` | gait-library clips the UI can reach |

`demo.json` keys: `format`, `source` and `versions` (`__main__`), `physics`
(physics.py), `visual` (visual.py), `policies` (policy.py), `sim`, `reference`
and `ui` (simcfg.py). `versions` maps every file named by a `file` key to the
first 12 hex digits of its sha256; every export run rewrites it, whichever
sections ran.

Binary containers use the DUETBIN1 format: `tools/duet_export/container.py`
and `demo/js/lib/container.js`. Each is a gzip file holding a JSON header plus
named, 16-byte-aligned little-endian arrays. Do not invent other formats.

## JS module contracts
```js
// sim/ (SimCore): pure logic + MuJoCo calls, no DOM; runs in worker and Node
const core = new SimCore({ mujoco, model, data, cfg: demo, reference /*container arrays*/,
                           policies: { human, osl } /* {run(Float32Array) -> Promise<Float32Array>} */ });
core.reset();                      // idle clip, frame 0, standing
core.requestTwist([vx, vy, wz]);   // latched; consumed at the next control boundary (blend rules)
const info = await core.step();    // one 10 ms control step (5 physics ticks)
// info: {time, failed, failure, siteError, clip, clipName, twist, phase, blending}
twistForKeys(keys, demo.ui)        // sim/commands.js: Set of held keys -> {twist, label}

// nn/: const policy = await loadPolicy(url, { backend: 'auto' | 'webgpu' | 'cpu',
//        cpuKernel: 'auto' | 'packed' | 'int8', onProgress, init /* fetch init: {signal, ...} */ })
//      policy.backend, policy.inputSize, policy.outputSize, await policy.run(Float32Array)
//      policy.info: {reason, adapter, selfTest, fallback, cpuKernel, cpuMs, cpuRace}; reason
//      starts with 'self-test passed' | 'requested' | 'small network' | 'CPU only:' |
//      'CPU faster (' | 'WebGPU unavailable:' | 'WebGPU failed, now on the CPU:' (a GPU
//      failure at run time)
// lib/container.js: fetchBytes(url, onProgress, init), loadContainer(url, onProgress, init),
//      loadJSON(url, init); init goes to fetch(). Errors name the URL ("<url>: HTTP 404",
//      "<url>: corrupt or truncated gzip (...)").
// session.js: await DemoSession.create({ assetsBase, backend, gpu, onProgress })
//      fetches demo.json with {cache: 'no-cache'}, then each asset as <file>?v=<versions[file]>.
//      The first failure rejects at once, aborts the other downloads and disposes the
//      policies that still finish. session.status(): HUD fields, incl. command ('Starting…' /
//      'Stopping…' while command shaping holds a walk), shaping (null | 'start' | 'reverse'),
//      backend, backendReason (= the human policy's info.reason), cpuKernel.
// render/frame.js (worker, no three.js): new FrameExtractor(mujoco, model, demo.visual).extract(data)
//      -> {frame, transfer} where transfer lists ArrayBuffers for postMessage. worker.js posts
//      the pose after every control step (skipped while MAX_IN_FLIGHT = 6 frames are out) and adds
//      frame.time (simulated s, never decreasing) and frame.epoch (resets so far: a new epoch is a jump).
// render/scene.js (main): new DemoRenderer(hostElement, demo.visual, visualArrays,
//      { onRelease /* frame => {}: done with it, buffer back to the worker */, chaseAzimuth: 180, chaseElevation: -20, delay: 0.012 });
//      .setFrame(frame) (timed frames are queued and drawn interpolated, `delay` behind the newest);
//      .tick(dt?) once per animation frame: draws only if something changed, returns whether it drew;
//      .render(dt?); .invalidate(); .resize(); .resetView(); .setCameraMode('chase'|'follow'|'free');
//      .cameraMode; .cameraDistance; .stats; .dispose()
```

## Demo page DOM contract (`demo/index.html`, used by `demo/js/app.js`)
`#viewport` (renderer host, full window); `#loader` overlay with `#loader-title`,
`#loader-status`, `#loader-bar` (fill element, width in %), `#loader-size`,
`#start-btn`; `#hud` (inert until Start) with `#hud-command` (the command
label: `Starting…` or `Stopping…` while command shaping holds a walk back,
`data-shaping` says which rule), `#hud-speed`, `#hud-rtf`, `#hud-backend`,
`#hud-note` (shown while inference is on the CPU, worded from
`status().backendReason`) and the developer rows `#hud-clip`, `#hud-time`
(`data-debug`, shown with `?debug=1`); `#help` (controls, hidden until Start,
toggled by H; `data-mode` full or compact; `kbd[data-key=...]` get class
`pressed` while held; elements with `data-label="<code>"` show what the
visitor's keyboard layout prints for that code); `#touch-pad` (touch
steering: `button[data-code]` hold `KeyW`, `KeyA`, `KeyS`, `KeyD`,
`#pad-fast` toggles top speed); `#toast`; buttons
`#btn-reset`, `#btn-pause`, `#btn-camera` (`data-mode`), `#btn-help`; `#error`
overlay with `#error-text`; `#back-link` (hidden when `?embed=1`). The page
shell sets `html.embed`, `html.touch` (coarse pointer), `html.no-focus` and
`html[data-phase]`.

## Ground rules
- Read the bundle and osl_deploy source; never modify `../osl_deploy` or `../../mjlab`.
- Python: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate osl-deploy`.
  Install nothing into it. Run from `tools/` (`python -m duet_export ...`).
- Node 22; dev packages are already installed in `tools/node_modules`
  (playwright 1.64.1, meshoptimizer 1.3.0, three 0.187.0, webgpu 0.6.2 = Dawn
  for Node). Headless Chromium: `PLAYWRIGHT_BROWSERS_PATH=<scratchpad>/ms-playwright`.
  Launch it with `--enable-unsafe-webgpu --enable-features=Vulkan`. WebGPU then
  runs on SwiftShader, which is correct but slow. It needs a page served over
  http://127.0.0.1, not about:blank.
- The machine is shared with training jobs. Prefix heavy runs with `nice -n 19`,
  set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, run no
  parallel sweeps, and keep closed-loop checks to a few minutes.
- The user makes every git commit; the repository is initialised with none.

## Findings worth knowing (2026-11-16)
- **Falls come from the policy, not the port.** SimCore matches osl_deploy's
  Simulation to rounding on a 13.5 s trace: the reference state is exact, the
  human observation is within 3.5e-17, the OSL input is exact. A closed-loop
  study found int8 human weights fall exactly as often as fp32 on identical key
  sequences; fp16 diverges on the same borderline trials. The remaining falls
  are policy limits at some transitions: walk to turn in place, short taps from
  the reset stance, fast restarts after backing up. DemoSession relaxes the
  model for 1 s after a fall, then resets. The first walk after a reset waits
  up to 0.35 s for the idle clip's phase to reach 0.2-0.5.
- **Command shaping: forward after backward.** A seeded key-masher benchmark
  (casual and stress profiles, plus paired transition trials on WebGPU) found
  that every fall starts a new walk while the body still carries the last
  one's momentum. Walking forward out of a backward walk is the weakest
  transition: 3 falls in 20 trials without a rule, 0 in 32 with it. So W while
  backing up, or before the body has come to rest after it, first stands, and
  the forward walk starts once the pelvis is slower than 0.35 m/s, at most
  1.0 s after the key (`SHAPING` in session.js). The HUD says `Stopping…` then
  `Starting…` meanwhile. The same guard for forward to backward, a stand
  before walk to turn, a minimum gait dwell and a fast-start ramp showed no
  benefit (the falls moved into the stop, or none happened either way), so they
  were not adopted, and the capped speed table (1.2 m/s forward, -0.7 m/s
  backward) stays. Stopping out of a fast walk or arc still falls now and then.
- **Cache busting.** GitHub Pages sends `Cache-Control: max-age=600`, so for
  10 minutes after a deploy a returning visitor could pair a new `demo.json`
  with an old cached asset (a different observation layout fails with
  "expected N inputs"). The exporter writes `versions` (content hashes) into
  `demo.json`; the demo fetches `demo.json` with `cache: 'no-cache'` and every
  asset as `<file>?v=<version>`. JS modules and `demo/vendor/` are not
  versioned: code can lag a deploy by up to 10 minutes.
- **CPU kernel race.** JavaScriptCore runs the packed int8 kernel (4 weights
  per Int32 load, sign-extended with shifts) 2.7x slower than V8: 25.7 ms per
  human inference in WebKit against 9.5 ms in Chromium. A second kernel, one
  Int8Array load per weight with one multiply-add per statement, runs in
  9.1 ms in WebKit but 12.2 ms in Chromium and 14.8 ms in Firefox (packed:
  10.3). Both sum the same exact products in the same order, so their outputs
  are bit-identical. The CPU backend times both at load on the self-test input,
  in batches of at least 20 ms because `performance.now()` is coarsened to
  about 1 ms in WebKit and Firefox, and keeps the faster (about 0.13 s;
  `info.cpuKernel`, `info.cpuRace`). In the demo, WebKit went from RTF 0.33
  (27 ms of inference per step) to 0.85 (9.5 ms). JavaScriptCore is fussy about
  form: the same kernel written as one long expression per four columns ran at
  21.7 ms. Measured on x86-64 Linux (Playwright builds); Apple Silicon is
  untested.
- **Firefox WebGPU readback latency.** Firefox 157 (Linux, `dom.webgpu.enabled`)
  compiles and runs the WGSL correctly, but every `mapAsync` round trip takes
  about 100 ms, a fixed polling latency rather than compute (Chromium on the
  same GPU: 2-3 ms). So `auto` picks the CPU there ('CPU faster (GPU 100.0 ms,
  CPU 11.0 ms ...)'). The self-test times the CPU first and stops timing the
  GPU after the first run over 1.5x the CPU time: in Firefox one timed readback
  instead of five, about 0.4 s less load time (the human net then loads in
  0.73 s in a worker). The demo words its CPU note from `backendReason`.
  Windows and macOS Firefox, where WebGPU ships by default, are untested.
- **Touch steering.** On a coarse pointer, or from the first touch, the demo
  shows an on-screen pad (`#touch-pad`) whose buttons hold the same
  KeyboardEvent.code values as the keyboard (`KeyW`, `KeyA`, `KeyS`, `KeyD`)
  in the same held-key set; Fast toggles top speed. The landing page hides its
  key list on coarse pointers and says the pad steers. Do not rely on
  `navigator.maxTouchPoints`: Playwright's WebKit emulation reports 0 while
  `(pointer: coarse)` matches.
- **Default chase camera.** The ready screen keeps the front-right hero view;
  Start switches to `chase`, directly behind the walker and slightly above
  its head (azimuth 180°, elevation -20° at 3 m; set by the user on
  2026-11-16, replacing a right-rear -135° view), so W always walks into the screen and A/D turn the way they
  look. With the fixed-angle `follow` camera, the camera's angle to the
  walker's heading went from -149° to +64° over one U-turn in the play-test,
  after which W walked toward the camera and A and D looked reversed. C cycles
  chase, follow and free.
- **Render on demand and interpolation.** The worker posts the pose of every
  10 ms control step with its simulated time, and the main thread draws the
  walker between the two poses around its own clock. Before, poses went out
  at about 50 Hz on a 60 Hz display and 13-17% of frames repeated one. The
  renderer draws only when something changed (a new pose, the camera easing,
  orbiting or zooming): nothing while the embedded frame is scrolled out of
  view (an IntersectionObserver also pauses the simulation then), and while
  paused only when the visitor orbits or zooms.
- **WASM bindings: output arrays.** A plain Float64Array passed as an output
  argument (e.g. `mj_objectVelocity`'s `res`) is copied in and never written
  back. Use a heap view or `mujoco.DoubleBuffer`. SimCore computes body
  velocities in JS.
- **WASM bindings: loading.** `MjVFS.addBuffer` copies at about 5 MB/s.
  `lib/mujoco.js` writes to the Emscripten FS and loads through an empty
  MjVFS: 6 ms for the slim model.
- **nn backends.** The WebGPU kernels take int8_row networks only. Any other
  network, such as the fp32 OSL actor, runs on the CPU whatever backend is
  requested; `info.reason` says why.
- **Dawn for Node.** Keep a reference to the object `create([])` returns for as
  long as any policy uses it. Once it is garbage-collected, inference hangs.
- **Headless Chromium with a hardware GPU:** pass
  `--use-angle=vulkan --enable-unsafe-webgpu --enable-features=Vulkan`. On the
  RTX 4090 that gives 60 fps at real-time factor 1.00 with WebGPU inference.
  Without these flags you get SwiftShader, about 15 fps with CPU inference.
- **Playwright WebKit on Ubuntu 24.04** needs `libavif16`, `libgav1-1` and
  `libyuv0` (`sudo npx playwright install-deps webkit`, or `COMPAT_WEBKIT_LIBS`
  in `tools/test/compat_browsers.test.mjs`). A catch-all `page.route()` makes
  the demo's worker fail in WebKit with NotReadableError; route the fonts by
  URL pattern only.
- **Playwright request interception hides aborts in Firefox.** With any
  `page.route()` active, Playwright's Firefox keeps downloading a fetch the
  page aborted. Without a route, Firefox closes it within a few ms, as Chromium
  and WebKit do. Check download aborts (DemoSession's fail-fast loading)
  without routes.
