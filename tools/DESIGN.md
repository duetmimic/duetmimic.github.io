# DuetMimic web demo: design notes

A static GitHub Pages site, served from the repo `duetmimic/duetmimic.github.io` at https://duetmimic.github.io/. Its interactive demo runs the
stage-2 split-network unicycle policy in the browser: bundle B (mjlab run
`2026-11-15_20-24-39_s2_r6s7_B_decr025`, checkpoint `model_6498`), a
339-muscle human actor plus an onboard OSL actor whose 500 inputs are the
leg's encoders and shank IMU, its ankle load cell, heel and toe contact, and
its own last commands. Until 2026-11-18 it ran R2 (`stage2_R2_model_6898`: an
ankle and a socket load cell, no contact, 600 inputs). You steer it with WASD
(or the on-screen pad), and you can drag
a body part with the mouse or a finger to pull it off balance.
**Scope: a stripped-down demo you can pilot.**
It is not a port of osl_deploy. It keeps only what the two policies need to
walk: their observations, the gait-library reference with blending and rebase,
the training actuator delay, and the OSL sensor history. It drops stress
tests, async mode, telemetry and exact parity suites. Physics may differ from
native MuJoCo in the last bits.

Controls: W/S walk forward/backward, A/D turn left/right, A or D alone turns
in place, 1-5 and Shift set the speed. Walking backward, A/D steer like a car
in reverse (user request, 2026-11-16): S+A backs along a path that curves to
the walker's left while its body yaws right (wz < 0), so, seen from behind
it, it comes toward you and moves to your left; S+D mirrors it. The HUD says
"Backward, curving left/right". A press on the body (left button or one
finger) grabs the part under it and pulls it toward the pointer until the
release; a press on the background orbits as before. X (or #btn-xray) toggles
the X-ray view (user request, 2026-11-19): muscles hidden, bones and socket
see-through, the rest of the prosthesis opaque, and arrows for the force and
moment the prosthesis applies to the residual limb at the socket. G (or
#btn-sensors) toggles the sensor charts (user request, 2026-11-19: "a toggle
to show osl onboard sensing ... just imu + loadcell + force sensor + maybe
encoders"): a card of strip charts of what the prosthesis itself measures
over the last 3 s, and a marker on the leg at each sensor in the same colours.

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
  (about 0.2 ms). Human weights are int8 per output row (B: 0.0022 RMS action
  error against the fp32 ONNX, the export's own check; R2: 0.0019, and on R2 a
  16 s closed-loop walk showed no fall and the same site error as fp32,
  2026-11-16). The OSL weights stay fp32.
- Physics model: `physics.mjb.gz` is 463 KB, down from 103 MB. It keeps only
  the collision meshes' convex hulls and replaces the mesh BVHs with
  single-node stubs. Its dynamics are bit-identical to the bundle model. It
  loads in about 0.6 s in WASM; the full model takes 25 s. Rays (mj_ray,
  mjv_select) miss mesh geoms, so the drag's picking casts its rays on the
  render meshes on the main thread instead (render/scene.js `pick()`).
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
demo/js/app.js                             main thread: UI, keyboard, touch and drag input, renderer, worker client
demo/js/ui/sensors.js                      the sensor charts: history ring, canvas strip charts, sensor group colours (no DOM at import)
demo/js/worker.js                          worker: message protocol + real-time pacing around DemoSession
demo/js/session.js                         DemoSession (no DOM): loading, SimCore, command shaping, drag spring, fall handling, HUD status
demo/js/lib/{container,mujoco}.js          shared loaders (DUETBIN1 containers, MuJoCo wasm)
demo/js/sim/                               SimCore: reference, observations, delay, step loop, commands
demo/js/nn/                                MLP engine (CPU + WebGPU)
demo/js/render/                            frame extraction (worker side) + three.js scene, picking, pull overlay (main);
                                           palette.js: the scene's own colours (no DOM, no three.js)
demo/vendor/{mujoco,three}/                pinned third-party builds
demo/assets/stage2/                        exported assets (below)
tools/duet_export/                          Python exporter (osl-deploy Conda env), reads the bundle through osl_deploy
tools/test/                                Node checks (node --test), browser smoke tests (Playwright);
                                           paths.mjs: asset folder, bundle and trace locations (DUET_ASSETS, DUET_BUNDLE, DUET_TRACE)
```

## Assets (`demo/assets/stage2/`, written by `python -m duet_export`)
| file | owner module | content |
| --- | --- | --- |
| `demo.json` | all sections | config; top-level keys owned per section (below) |
| `physics.mjb.gz` | `physics.py` | slim physics model: the bundle model with the mesh data that physics never reads stripped out. Every id (bodies, geoms, sites, tendons, actuators, sensors) is unchanged |
| `visual.bin.gz` | `visual.py` | render meshes (decimated), per-geom render info |
| `human.bin.gz`, `osl.bin.gz` | `policy.py` | MLP weights (DUETBIN1) |
| `reference.bin.gz` | `simcfg.py` | gait-library clips the UI can reach |

`demo.json` keys: `format`, `source`, `sections` and `versions` (`__main__`),
`physics` (physics.py), `visual` (visual.py), `policies` (policy.py), `sim`,
`reference` and `ui` (simcfg.py). `versions` maps every file named by a `file`
key to the first 12 hex digits of its sha256; every export run rewrites it,
whichever sections ran. `source` names the bundle (`bundle` is
`<training run>/<bundle folder>`, plus its `manifest_sha256`), and `sections`
maps each exported section to the sha256 of the bundle manifest it was built
from. An `--only` run whose bundle differs from a section it would keep is
refused (`--allow-mixed` overrides): two bundles' models can share every id and
count (R2's and B's differ only in three touch-zone boxes), so a mixed folder
loads and runs but is silently wrong. `sim.touch_sites` records the bundle's
foot touch zones (the manifest's `touch_sites`, which osl_deploy checks against
`model.mjb`) with their site, sensor and body ids and sensordata addresses;
SimCore does not read it. `tools/test/contract.test.mjs` checks that every
section shares `source`'s manifest, that the observation layouts tile their
inputs and match the networks, and that the touch zones in `physics.mjb.gz`
are the recorded ones.

Test goldens: `physics.py` also writes the native MuJoCo trace that
`physics.test.mjs` replays (`physics_wasm_ref.bin.gz`, which records the slim
model's sha256), and `policy.py` writes reference outputs for `nn.test.mjs`
(`nn_reference.bin.gz`, which records each container's sha256). For
`demo/assets/stage2` they go to `tools/test/` (tracked) and `tools/traces/`
(git-ignored); for any other `--out`, into that folder (`golden_path` in
`common.py`; `tools/test/paths.mjs` resolves them by the same rule, so
`DUET_ASSETS` brings its own goldens and a scratch export never rewrites the
repository's).

Binary containers use the DUETBIN1 format: `tools/duet_export/container.py`
and `demo/js/lib/container.js`. Each is a gzip file holding a JSON header plus
named, 16-byte-aligned little-endian arrays. Do not invent other formats.

## JS module contracts
```js
// sim/ (SimCore): pure logic + MuJoCo calls, no DOM; runs in worker and Node
const core = new SimCore({ mujoco, model, data, cfg: demo, reference /*container arrays*/,
                           policies: { human, osl } /* {run(Float32Array) -> Promise<Float32Array>} */ });
core.reset();                      // idle clip, frame 0, standing (mj_resetData also clears xfrc_applied)
core.requestTwist([vx, vy, wz]);   // latched; consumed at the next control boundary (blend rules)
const info = await core.step();    // one 10 ms control step (5 physics ticks)
// info: {time, failed, failure, siteError, clip, clipName, twist, phase, blending}
core.disturb = (model, data, tick, views) => {};  // optional, null by default: called before every
                                   // mj_step on a cleared views.xfrc_applied; adds world-frame wrenches
                                   // (force, torque about xipos); reads the kinematics mj_step left (the
                                   // tick before, as MuJoCo's viewer); must not allocate
core.applyDisturbances(views?);    // clear xfrc_applied, call disturb: step() does it per tick; code
                                   // that steps outside step() (the session's relaxed fall) calls it too
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
//      session.initial: the state is the one a reset (or the load) starts from (SimCore has run
//      no control step since; worker.js marks its frames frame.initial); session.sensorCount
//      (frame.sensors' length, 0 without); session.sensorRow(out, offset?) -> values written
//      (frame.sensors of the current state alone; 0 for an initial or undrawable state).
//      The first failure rejects at once, aborts the other downloads and disposes the
//      policies that still finish. session.status(): HUD fields, incl. command ('Starting…' /
//      'Stopping…' while command shaping holds a walk), shaping (null | 'start' | 'reverse'),
//      backend, backendReason (= the human policy's info.reason), cpuKernel, perturb.
//      Pulling (DragSpring and DRAG in session.js, see Findings):
//      session.setPerturb({geom, local: [x, y, z], target: [x, y, z], id?}) starts a drag or moves
//      its target -> 'start' | 'update' | 'stale' (an update for a drag the session ended itself:
//      ignored until a release or another grab) | 'invalid' (any drag ends with perturb-end
//      {reason: 'error', message}). geom: a rendered robot geom (demo.visual.geom_ids, the same
//      ids as the physics model); local: the grabbed point in the geom's frame (world = geom_xpos +
//      geom_xmat · local, geom_xmat row-major), within geom_rbound + 5 cm; target: MuJoCo world, m.
//      session.clearPerturb(reason = 'release', message?) -> whether a drag was active; any reason
//      but 'release' ('reset', 'fall-reset', 'error') emits 'perturb-end' {reason, message, geom,
//      body, id}. reset() ends a drag first, so 'perturb-end' comes before 'reset'. The spring acts
//      at every physics tick, also through a fall's relaxed collapse. status().perturb: {active,
//      geom, body, id, force (N, the latest tick), maxForce (N, this grab's cap), stretch (m)}.
// worker.js, pulling: main -> worker {type: 'perturb', geom, local, target, id?} (on pointerdown,
//      then at most once per animation frame) and {type: 'perturb', release: true} (pointerup,
//      pointercancel, lostpointercapture, a mouse move with the left button up, blur, hidden,
//      off-screen, a lost context, a reset);
//      worker -> main {type: 'event', event: {type: 'perturb-end', reason, message, geom, body,
//      id}} when the worker ends a drag itself: 'reset', 'fall-reset' or 'error' (an invalid
//      message, one sent before loading finished, or a worker failure; never stops the worker).
//      The page drops its drag on a perturb-end for its own id (or for none).
// render/frame.js (worker, no three.js): new FrameExtractor(mujoco, model, demo.visual, demo.sim?)
//      .extract(data) -> {frame, transfer} where transfer lists ArrayBuffers for postMessage; with
//      demo.sim the frames carry the extras (frame.wrench, frame.sensors; see "Frame extras").
//      .readSensors(data, out, offset?) -> frame.sensors of the current state alone (no allocation);
//      frameLayout(demo) (main thread too); frameChannels(demo.sim) -> frame.sensors' channels
//      [{index, term, name, unit}] (the one naming rule: terms[*].channels, else the read names).
//      worker.js posts the pose after every control step (skipped while MAX_IN_FLIGHT = 6 frames
//      are out) and adds frame.time (simulated s, never decreasing), frame.epoch (resets so far: a
//      new epoch is a jump), frame.initial (the reset state before its first control step: its
//      extras are not a measurement, the page draws its pose only) and frame.sensorRows (the
//      sensor rows of the steps whose frame was skipped, {times, values} | null; see "Sensor
//      charts", Data).
// render/scene.js (main): new DemoRenderer(hostElement, demo.visual, visualArrays,
//      { onRelease /* frame => {}: done with it, buffer back to the worker */, chaseAzimuth: 180, chaseElevation: -20, delay: 0.012,
//        socketParent /* demo.sim.socket.parent: the residual limb's body, a stronger ghost in X-ray */,
//        xray /* overrides per key of DEFAULTS.xray: ghost looks, arrow colours, scales, caps, minimums */ });
//      .setFrame(frame) (timed frames are queued and drawn interpolated, `delay` behind the newest);
//      .tick(dt?) once per animation frame: draws only if something changed, returns whether it drew
//      (= .update(dt?) then .drawIfChanged(); app.js calls the two halves, moving the drag target
//      with this frame's camera in between);
//      .render(dt?); .invalidate(); .resize(); .resetView(); .setCameraMode('chase'|'follow'|'free');
//      .cameraMode; .cameraDistance; .cameraAzimuth (deg); .stats (incl. programs: shader
//      programs built so far); .dispose()
//      Pulling: .pick(clientX, clientY, slop?) -> {geom, body, local, point, distance} | null, the
//      robot part under the cursor at the pose on screen (bones and OSL parts by their render
//      triangles; a muscle grabs the nearest bone at the clicked point; slop in CSS px tries rings
//      of rays around a miss); .dragTarget(clientX, clientY, {geom, local}) -> [x, y, z], the cursor
//      ray met with the plane through the grabbed point, square to the view; .setPerturb({geom,
//      local, target, force?, maxForce?} | null) draws the pull (svg.pull-overlay, wider as force
//      nears maxForce) and lights the grabbed body; .setControlsEnabled(on) (OrbitControls);
//      .holdCamera(on) (no following during a drag, glides back after); .projectToClient([x, y, z])
//      -> [clientX, clientY] | null; .geomPoint(geom, local?) -> [x, y, z] as drawn.
//      X-ray view: .setXray(on) (drawn at the next tick: once while paused); .xray; .wrench ->
//      [Fx, Fy, Fz, Tx, Ty, Tz] as drawn (blended with the pose's alpha; world axes, MuJoCo's
//      = the scene's) | null (frames without it, initial frames); .xrayInfo() -> {on, wrench, anchor, force and
//      moment: {shown, origin, direction, length, tip}} read back from the arrow meshes (tests);
//      .roles (per drawn geom); .stats adds muscles (drawn or not), xray and markers. See
//      "X-ray view" below.
//      Sensor markers: .setSensorMarkers(on) (drawn at the next tick: once while paused);
//      .sensorMarkers; .sensors -> frame.sensors as drawn (blended with the pose's alpha) | null
//      (frames without it, initial frames); .sensorMarkersInfo() -> [{key, label, kind, group,
//      glyph, color, geom_id, local, shown, world (read back from the position attribute), size
//      (CSS px as drawn, the framing's scale included), fill, channel, layer (drawing order)}]
//      (tests); .playTime -> the simulated time of the pose on screen (the playback clock) | null.
//      options.sensors overrides DEFAULTS.sensors (glyph sizes, ring, contact growth, minScale,
//      colors per group). See "Sensor charts" below.
//      Framing: .setFraming({scale (<= 1), dx, dy} | null) draws the picture `scale` times its
//      size about the canvas centre, moved by (dx, dy) CSS px (camera.setViewOffset: the
//      camera, orbit and zoom unchanged; picking, projectToClient, the markers and the pull
//      overlay agree; the markers drawn min(1, max(minScale, scale)) times their size); kept
//      through resizes; .framing -> {scale, dx, dy} | null. app.js frames the walker clear of the
//      view panels with it (see "Sensor charts", Layout).
// render/palette.js (no DOM, no three.js): XRAY_COLORS {force, moment, bone, boneRim, socket,
//      socketRim, residual, residualRim, halo, dot}, ACCENT: the colours scene.js draws with and
//      the sensor colours are checked against (sensors.test.mjs).
// ui/sensors.js (main, no DOM at import): SENSOR_GROUPS {imu, cell, contact, encoder: {label,
//      glyph, color, steps}}, MARKER_GROUPS (overlays.markers kind -> group), SENSOR_ROWS, GAP,
//      STREAM_GAP_STEPS, sensorChannels(demo) (= frameChannels(demo.sim)), resolveRows(demo);
//      new SensorHistory(width, capacity?) .push(time, epoch, values, offset?) (copies; a new epoch
//      clears) .indexAt(t) .sample(t, channel) .breaks() .gap (s: the longest step the line joins);
//      new SensorCard(root /* #sensor-card */, demo, {groups?, rows?}) .push(frame) (before setFrame:
//      frame.sensorRows first, then the frame's own row unless frame.initial) .setVisible(on)
//      .setMode('full'|'compact') .draw(playTime, now) -> drew? .invalidate() .summary() (the
//      latest values as text) .stats() .info() .dispose().
// app.js test hook: window.__duet.state() (incl. perturb: {active, id, geom, body, local, target,
//      started, updates, released, ended, worker: status().perturb}, cursor, cameraAzimuth,
//      xray, sensors, sensorCard (off | full | compact | none: on, but no room), framing
//      ({scale, dx, dy, zone} | null: the walker framed clear of the view panels), walker (the
//      walker's zone on screen, client px, framed or not), walkerCovered (a panel covers the
//      walker and there is no room to frame it clear), help (shown: full | compact | hidden),
//      helpWanted (asked for)), .stats() (incl. render: view.stats; sensors: the card's {visible,
//      mode, draws, statics, skipped, settled, ms: {last, mean, p95, max, n}, samples, span,
//      epoch, breaks (where the line breaks), gap, streamed (the frames carry sensorRows), rows
//      (rows of skipped steps taken), framesWith, framesWithout, framesInitial}), .events(),
//      .pick(clientX, clientY, slop?), .project([x, y, z]), .geomPoint(geom, local?), .xray() ->
//      {on, anchor, anchorClient, F, T, forceLength, momentLength, force, moment, scale, colors
//      (the arrows' as drawn), legend: {shown, force, moment}}, .sensors() -> {on, card: {shown,
//      mode, box, drawnAt (play time of the traces' right edge), at (of the values), dpr, rows:
//      [{key, group, title, unit, plot (CSS px on the canvases), ticks: [{text, right, width}]
//      (the range labels as drawn, right-aligned at `right`), lines: [{name, index, color,
//      value, text}]}], empty, noData, noRows, stats}, playTime, drawn (the sensor frame as
//      drawn), markers: sensorMarkersInfo() + client ([x, y] on screen)}, .help(mode?) (ask for a
//      help mode, then lay the panels out as a resize does; returns the mode shown).
```

### Frame extras and overlays: the X-ray and sensing data contract (added 2026-11-18)
The data two views draw: the X-ray view (muscles hidden, the human bones and
the OSL socket transparent, the rest of the prosthesis opaque, arrows for the
total force and torque at the socket; built 2026-11-19 on `frame.wrench`,
`overlays.roles` and `overlays.socket`, see "X-ray view" below) and the sensor
charts (strip charts of the OSL's own sensors over 3 s, a marker on the leg at
each sensor; built 2026-11-19 on `frame.sensors`, the observation's channel
names and `overlays.markers`, see "Sensor charts" below; no UI uses `zones`
yet). Everything is derived from the bundle's manifest
and model by name; no id is hard-coded. Checked by
`tools/test/frame_extras.test.mjs`.

**demo.json `sim` additions** (`simcfg.py`; SimCore reads none of them):
- `sim.osl.observation.terms[*].channels`, `.units`: one name and unit per
  frame value (the manifest's `term_layout`), e.g. `osl_imu_acc_x`, `m/s^2`.
- `sim.osl.observation.terms[*].reads[*].anchor`: `{kind: 'site'|'joint', id,
  body, name}`, the site the sensor reads (IMU, load cell, touch zone) or its
  joint (an encoder, drawn at the joint's anchor), and the body it moves with.
- `sim.osl.sensor_set`: the manifest's (B: encoders, imu, ankle_load_cell,
  heel_contact, toe_contact true; socket_load_cell false).
- `sim.socket`: `{body, name: 'robot/OSL_KA_L1_osl_generic_socket', parent,
  parent_name, root (body_rootid), subtree_mass (kg below it: the
  prosthesis), cfrc_int_fresh, wrench (the convention, as text)}`. B: body 94,
  parent 93 (femur_r), root 2, 5.388 kg. `cfrc_int_fresh` is the exporter's
  check that every `mj_forward` and `mj_step` of the model recomputes
  `cfrc_int` (its acceleration-stage force, torque and accelerometer sensors
  make MuJoCo run `mj_rnePostConstraint`); without it frames get no wrench.

**demo.json `visual.overlays`** (`visual.py`; nothing in `visual.bin.gz`,
which stays byte-identical). Two numberings: a `geom` is an index into
`visual.geom_ids` (its row in `frame.geoms` and in `roles`), a `geom_id` the
MuJoCo geom id, which is what `DemoRenderer.geomPoint(geom_id, local)` and
the drag's `geom` field take. A `local` point is in that geom's frame:
world = geom_xpos + geom_xmat · local (row-major), so it rides the drawn,
interpolated pose. All are computed at qpos0 on a geom of the same rigid body
(the body itself, else one welded to it).
- `roles`: one per drawn geom, from the bodies, not the colours: `'socket'`
  (geoms on the socket body), `'cover'` (the OSL knee and ankle housings,
  drawn orange), `'hardware'` (the rest of the socket body's subtree),
  `'bone'` (everything else, the human). B: 96 bone (geoms 0-88 and
  96-102); 1 socket, geom 89 (geom_id 364); 4 hardware, geoms 90, 91, 93, 95
  (geom_ids 365 femoral pylon, 366 adapter, 368 tibial pylon, 370 foot);
  2 cover, geoms 92, 94 (geom_ids 367 knee, 369 ankle). The material
  categories agree.
- `socket`: `{geom, geom_id, body, local}`, the socket body's origin, where
  `frame.wrench` acts. B: geom 89 (geom_id 364), local [0, 0, 0.099171155].
- `markers`: one per OSL sensor, grouping the observation's reads by site or
  joint and kind, in frame order: `{key, label, kind: 'imu'|'load_cell'|
  'contact'|'encoder', geom, geom_id, local, channels (indices into
  frame.sensors), anchor {kind, id, name, body}}`, plus `axis` (the joint
  axis in the geom frame) for an encoder. Keys come from the read names
  (`osl_knee_pos` + `osl_knee_vel` -> `knee_encoder`). Every sensors channel
  belongs to exactly one marker. B, in array order (by first channel):
  knee_encoder [0, 2] on geom 92 (geom_id 367), ankle_encoder [1, 3] on
  geom 95 (geom_id 370), imu [4-9] on geom 92 (geom_id 367), ankle_cell
  [10-15] on geom 94 (geom_id 369), heel [16] and toe [17] on geom 95
  (geom_id 370). `local` is the read's site or its joint's anchor, except
  for a touch sensor on a box zone (2026-11-19): its glyph goes where the
  zone meets the foot (`contact_point` in visual.py): of the collision geoms
  moving with the zone's body, the point of a geom's axis (a capsule's or
  cylinder's segment, else its centre) inside the box nearest the box's
  centre, else the centre. A touch sensor sums the contacts inside its box,
  and they act on those geoms. B's toe zone runs to 2-3 cm past the drawn
  foot (its centre floated in front of the toe); its glyph now sits at the
  toe end of the middle capsule (geom 371), heel and toe both on geom 371's
  axis, inside the foot plate.
- `zones`: one per touch-sensor marker: `{key, geom, geom_id, type ('box'),
  local (box centre), half (site_size), quat (box orientation in the geom
  frame, w x y z)}`. B: heel (half [0.065, 0.03, 0.05]) and toe, both on
  geom 95 (geom_id 370).

**Frame extras** (`render/frame.js`). `new FrameExtractor(mujoco, model,
demo.visual, demo.sim)`: with the 4th argument, frames carry, after `root` in
the same pooled buffer (Float32):
- `frame.wrench` (6): `[Fx, Fy, Fz, Tx, Ty, Tz]`, N and N m, MuJoCo world axes
  (z up): the load the prosthesis applies to the residual limb at the socket
  body's origin. With s = `sim.socket.body`, r = `sim.socket.root`:
  F = -cfrc_int[s][3:6], T = -(cfrc_int[s][0:3] + (subtree_com[r] - xpos[s])
  × cfrc_int[s][3:6]). `cfrc_int[s]` is the wrench the femur applies to the
  socket's subtree, [torque about subtree_com[r]; force]. Nothing calls
  `mj_rnePostConstraint` (after an `mj_step` it would read the integrated
  velocities). F points up in stance. At a strike it follows the contact a
  few physics ticks late: the contact first decelerates the prosthesis's own
  mass, and `cfrc_int` holds the stiff piston damper's force at the tick's
  starting velocity, while the implicit integrator applies it at the end
  velocity. So at a contact's first loaded control step F can still point
  down, as in swing: in a closed-loop run on B a 215 N heel strike read
  F_z = -33 N, then +324 N one step later. Hanging still, F is the
  prosthesis's weight (M g down, M = `sim.socket.subtree_mass`) and
  T = (subtree_com[s] - xpos[s]) × F. B magnitudes: |F| up to 2.3 kN
  (turn-in-place heel slams), |T| up to about 130 N m: draw with a cap.
- `frame.sensors` (N, the sum of the bus terms' frame_dim; B: 18): the OSL's
  processed, undelayed sensor frame from the current sensordata, bus terms in
  demo.json order (B: actpack 10, loadcell 6, contact 2), through
  `readSensorFrame` in `sim/observations.js`, the one function that also
  builds the policy input (components, clamp to ±saturation, minus the tare).
  The policy sees a frame `lag` control steps later (B: 1). Live through
  falls (it reads sensordata, not SimCore's observer).
- `frameLayout(demo)` (also `extractor.layout`), for the main thread (no
  MuJoCo): `{version: 1, floats, bytes, geoms, segments, root, wrench,
  sensors}`, each section `{offset (floats), byteOffset, length, ...}`;
  `wrench` adds `labels` Fx..Tz, `units`, `body`; `sensors` adds `channels:
  [{index, term, name, unit}]` (`frameChannels(demo.sim)`, which the sensor
  card uses too: one naming rule; a demo.json without `terms[*].channels`
  names them by their reads). A section demo.json cannot supply is `null`.
- `extractor.readSensors(data, out, offset)`: frame.sensors of the current
  state alone, as `extract()` writes it (the worker's rows of skipped
  steps).
  B: 39008 bytes; geoms 0+1236, segments 1236+8488, root 9724+4, wrench
  9728+6, sensors 9734+18 (before the extras: 38912 bytes).
- Compatibility (JS modules are not cache-busted during a deploy): older page
  code ignores the extras (and `frame.initial`, `frame.sensorRows`); newer
  code must check `frame.wrench` / `frame.sensors` before use (frames from an
  older worker, or from an extractor built without the sim config, have
  neither) and prefer them to views built from `frameLayout` offsets (check
  `frame.buffer.byteLength === layout.bytes` first). `recycle` accepts only
  buffers of the extractor's own byteLength. A buffer detaches when the page
  posts it back, so copy the extras into the chart's rings on arrival, before
  `setFrame` (which may release a frame at once).
- The reset state (`frame.initial`, set by worker.js while SimCore has run no
  control step since the reset or the load, `session.initial`) is not a
  measurement (2026-11-19): the reset pose stands the walker on the socket
  piston at its rest length, with the foot pressed into the floor (heel
  132 N, toe 20 N on B), so the socket passes 26 N (|T| 0.8 N m, nearly
  horizontal) and the IMU reads about 3 g (acc y 29.6 m/s²); one step later
  |F| is 324 N, then 462, up to 567 N at step 7, and standing settles at
  280-370 N. These are the true values of that state (the OSL policy's whole
  history starts from it), but shown on the ready screen (`?xray=1`,
  `?sensors=1`), after R while paused, or as the first chart sample of every
  reset (a 30 m/s² spike for 3 s), they read as a measurement of standing.
  So the renderer draws its pose but no wrench and no sensor frame (the
  legend shows "–", the arrows go, the contact glyphs are open) and the
  card does not chart it. A step a reset discards (its inference was in
  flight) leaves that state; its frame is initial too.
- Allocation: `extract()` makes no MjData getter calls (each allocates a new
  view; the heap views are cached and re-read when the heap grows) and no
  scratch arrays. It writes into a pooled frame and allocates only when the
  pool is empty (a new buffer). A frame object and its views are built once
  per ArrayBuffer object, but in the worker every buffer the page posts back
  arrives as a new ArrayBuffer object (structured clone). So `recycle()`
  builds them for every frame: about 0.68 KB of short-lived garbage per frame
  with the extras and 0.45 KB without, which is what the extractor before the
  extras made in `extract()` (Node 22's V8; about 70 KB/s at 100 frames/s;
  browser engines size objects differently). Views cannot be moved to another
  buffer, so frames that carry views cost that much. Only a same-realm caller
  that recycles the very ArrayBuffer it got (the Node tests) allocates
  nothing per call.
- Wiring: `session.js` (`makeExtractor`) builds the extractor with
  `demo.visual` and `demo.sim`; if the sim config is unusable, it warns and
  builds a plain one, so the frames carry no extras.

### X-ray view (render/scene.js `setXray`, app.js; added 2026-11-19)
- **Parts.** Each drawn geom takes its material from one resolver
  (`_refreshMaterials`): its role (`overlays.roles`, else the material
  categories), X-ray on or off, and whether its body is grabbed. In X-ray,
  `bone` and `socket` meshes get a ghost and cast and receive no shadow;
  `hardware` and `cover` keep their base material, opacity 1 and shadows,
  exactly as without X-ray. Base materials are never changed. Ghosts and the
  grab's glowing variants are built by factories (`ghostMaterial`,
  `standardMaterial(spec, glow)`), never by `clone()`: a ShaderMaterial has
  no `emissive` (the old `_setGlow` cloned and set it, which would throw on a
  ghost) and a clone drops `onBeforeCompile`. Every ghost carries an
  `emissive` Color all the same (black, or the glow colour) so tests can ask
  any drawn material whether it glows.
- **Ghost.** One unlit ShaderMaterial: opacity and colour rise from `core` to
  `rim` as (1 - |N.V|)^power, then `tonemapping_fragment`,
  `colorspace_fragment` and `fog_fragment`, like the lit scene; transparent,
  no depth writes, depth-tested (the opaque prosthesis hides what is behind
  it). Looks (`DEFAULTS.xray`, sRGB): bones slate `#5d7394` (rim `#33435c`,
  core 0.07, rim 0.62); socket neutral charcoal `#45494f` (rim `#1d2024`,
  0.05 / 0.8, double-sided: three draws it back faces first); the residual
  limb's bones (`options.socketParent` = `demo.sim.socket.parent`, femur_r) a
  stronger slate (0.10 / 0.85). Blending order (renderOrder): bones 0, socket
  1, residual 2, so the femur's outline reads inside the cup. A grabbed
  ghost's rim turns the glow orange. Every program is built at construction
  (`_prewarm`): one drawing of the scene without X-ray (the robot with its
  base materials and the muscles) inside a 1-pixel scissor, which builds
  their programs and the shadow pass's depth programs (`renderer.compile()`
  builds no depth program), then `renderer.compile` of a scratch scene with
  the ghost, overlay and marker materials against the real one's lights and
  fog. So the first toggle compiles nothing either way, also the first X
  (off) of a session started with `?xray=1`, which used to build the muscle,
  its depth and the socket's programs then (a 26-43 ms stall on a quiet
  machine, up to 113 ms under load; render_shots counts `info.programs` for
  the on direction, e2e_browser for that off direction).
- **Muscles.** Hidden, and `_applyPose` skips their instance update. X-ray off
  recomputes them for the pose on screen from the two queued frames it was
  blended from (`_shown`), or, for an untimed frame (test pages), the frame
  kept for this (`_untimed`, released once off screen). If the queue moved on
  meanwhile, a newer frame waits and the next update poses them; they stay
  hidden until then. Picking skips hidden muscles; ghost bones stay pickable.
- **Arrows.** Both tails at the socket anchor (`overlays.socket.local` on its
  drawn geom, through the interpolated mesh matrix); the wrench is blended
  between the same two frames with the pose's alpha (`_applyPose`); MuJoCo's
  world is the scene's (z up), so no axis change. Force: along F, 0.5 mm per
  N, capped at 0.8 m, hidden below 3 N. Moment: along T (right-hand rule),
  double-headed, 4 mm per N m, capped at 0.6 m, hidden below 0.3 N m. Shaft
  radius 2% of the length within 3.5-6.5 mm, heads 20% within 20-50 mm (60%
  of a short arrow at most); each part over a white halo copy; a dark dot
  (`#1d1d1f`, 11 mm radius) marks the anchor. Flat MeshBasicMaterial, no tone
  mapping (the colours equal the legend's swatches), transparent pass,
  `depthTest` off, renderOrder 40-45 (moment, force, dot), no shadows, outside
  `geomMeshes` (never picked, never glowing). Frames without `wrench`, and
  the reset state (`frame.initial`), draw the dot only and the legend shows
  "–"; so does a non-finite wrench (an arrow is hidden unless its magnitude
  is finite and at least its minimum).
- **Colours.** Force `#008300` (green), moment `#6250d6` (violet): slots 6
  and 7 of the dataviz reference palette, validated as a pair on the scene's
  background `#f4f4f2` (`validate_palette.js`: CVD ΔE 28.8 deutan, normal
  35.5, contrast 4.5 and 5.2:1); ΔE ≥ 23 from the slate ghost as blended
  over the floor; not orange (covers `#ff6c0c`, accent, pull line). The
  sensing view should avoid these two, orange, the slate ghost and the
  charcoal socket. They live in `render/palette.js` (with the ghosts and the
  accent), which scene.js draws with and sensors.test.mjs measures against;
  app.js sets the CSS tokens `--xray-force`, `--xray-moment` (demo.css keeps
  the same values as defaults) from the renderer's options, so the legend's
  swatches are the drawn colours (e2e_browser compares them).
- **Cost** (stride pose, 1280x720): 122 draw calls and 165,192 triangles in
  X-ray against 209 and 350,776 without (no muscles, and ghosts cast no
  shadows); restoring gives exactly the baseline (render_shots).
- **UI** (app.js). `setXray(on, quiet)`: html.xray, `aria-pressed`, the
  legend, the renderer, a toast "X-ray view on/off"; X only while running
  (paused included), the button once Start has made the HUD live; `?xray=1`
  applies it at load (quietly, behind the loader too, where the reset pose
  shows the dot and dashes). Not stored. The legend's numbers (|F| in N, |T|
  in N m, rounded, `view.wrench`; "–" without one or when not finite) change
  at most every 100 ms, in the render loop, so a paused toggle costs one
  draw. They sit in no live region and are not aria-hidden: a screen reader
  reads them when it gets to the legend.
- **Layout** (demo.css; `layoutHud()` on every toggle, help change, touch
  change, resize and font load). `#viewbar` sits top right on row 1 (beside
  the back link below 800 px; embedded below 640 px it gets a row of its own
  under the top bar, and the toast moves below it); it is sized for two
  toggles. `#xray-legend` sits under it, right-aligned, 264 px wide (compact
  below 560 px wide or 460 px tall: "Socket load", no note line, 170 px;
  below 372 px the rows keep only swatches and numbers, 120 px); on touch
  phones 330-479 px wide and at least 600 px tall it sits bottom right
  beside the pad, where the floor is clear (at the top it covered the
  walker's head); where it still crosses the walker (at the top on narrower
  phones, beside the pad on touch windows 600-720 px tall), the walker is
  framed clear of it (see "Sensor charts", Framing the walker). The
  toast keeps clear of the right column (the legend at the top, the sensor
  card; see "Sensor charts", Layout). The help shows full, compact (keyboard:
  the one-line legend at the bottom right, else left of the right column,
  class `beside`, where it wraps) or not at all (touch, or a keyboard window
  with no room even for that) as there is room beside the legend, the sensor
  card, the view toggles and the top bar (`fitHelp`; asked-for mode
  `app.help`, shown `app.helpShown`); a keyboard's full help also gives way
  where the panels have framed the walker into its place (narrow windows,
  see "Sensor charts", Layout). H, ? and the ? button toggle what is shown:
  the full help goes back to the start mode, anything less asks for the full
  help; where that changes nothing on screen, a one-line legend still shown
  goes ("No room for the full help here") or, with nothing shown, a toast
  says "No room for the help here". So every press does something. At
  1280x720 the full help fits beside the full legend, at 1280x600 it shows
  compact; with both views on at 1280x720 the one-line legend sits left of
  the card (it used to vanish there, and the first H did nothing). The
  help's muscle-colour line hides in X-ray; its touch rows gain the X-ray and
  sensor-chart icons (left out below 460 px tall).
- Verified (2026-11-18, `frame_extras.test.mjs`): on all 1351 boundaries of
  `tools/traces/trace.bin.gz` (state set, `mj_forward`, extract),
  `frame.wrench` matches the native formula on the trace's `socket_cfrc`,
  `root_subtree_com` and `socket_xpos` to 5.6e-8 of |F| (|F| up to 2340 N)
  and 5.5e-8 of |T|; `frame.sensors` matches osl_deploy's `osl_frame` to
  7.8e-5 absolute (float32 at about 1 kN), relative 5.9e-8. In the 655
  stance boundaries (heel + toe > 200 N there and at the boundary before)
  F_z > 0 at all of them (the test tolerates 1% that are not), correlation
  with the contact force 0.992; the 15 contact onsets left out read F_z
  17 N and up. Without the formula, the prosthesis hanging still
  (`mj_inverse`, keyframe 0 lifted 2 m, 8 rigid orientations) gives F = M g
  to 4e-9 and T = (subtree_com[s] - xpos[s]) × F to 4e-8 of |F||r| (a
  torque left about the tree root's COM would be off by 5-21 N m). Overlay
  points reproduce the sites, joint anchors, the touch zones' centres and
  contact points (the rule above, recomputed in the test) and the socket
  origin to 7e-10 m at six random poses. `extract()` takes about 9-16 µs with or
  without the extras (0.5 µs more with them, noisy machine); 0 MjData reads
  per call after the first. The allocation figures above were measured
  separately (Node 22, all V8 heap spaces, no GC inside the measured loop):
  per `recycle()` + `extract()`, 0.68 KB with the extras and 0.45 KB without
  when every buffer comes back as a new ArrayBuffer object, under 10 bytes
  when the same ArrayBuffer comes back; the extractor before the extras made
  0.45 KB per frame either way.

### Sensor charts (ui/sensors.js, render/scene.js `setSensorMarkers`, app.js; added 2026-11-19)
- **What.** G or `#btn-sensors`: `#sensor-card`, strip charts of the OSL's
  own processed sensor frame (`frame.sensors`: what its policy gets one
  control step later) over the last 3 s, and a glyph on the leg at each sensor
  (`overlays.markers`). Rows, in the user's order: IMU gyro (rad/s; x y z),
  IMU accel (m/s²; x y z, dashed at the ±39.2 clamp, the read's `saturation`),
  load cell force (N), load cell moment (N·m), foot contact (N; heel, toe),
  encoders (deg; knee, ankle; positions only, the velocities are left out).
  The compact card (short windows, touch phones): gyro, force, contact. A
  footnote gives the shank frame's axes and the load cell's tare. Rows find
  their channels by name (`frameChannels`, the names and order `frameLayout`
  gives), so another bundle's layout only drops missing lines; where no row
  finds a channel the card says "No charts for this simulation's sensors."
- **Data.** `onFrame` pushes every frame into the card's `SensorHistory`
  before `setFrame` (the frame's buffer may go back to the worker at once):
  first the rows in `frame.sensorRows`, then the frame's own `sensors`; 320
  rows, 3.2 s at 100 Hz, times in a Float64Array, values copied into a
  Float32Array, O(1), no allocation. A new epoch (a reset) clears it, the
  same time replaces, an older one is dropped; an initial frame (the reset
  state) only starts its epoch. It fills whether the card is shown or not,
  so it opens with the last seconds. Frames without `sensors` (an older
  worker) are counted; with none ever, the card shows "No sensor data from
  the simulation."
- **The sensor stream (2026-11-19).** A step whose frame the worker cannot
  post (all `MAX_IN_FLIGHT` = 6 buffers out) used to be lost to the charts,
  whose lines broke into dashes wherever the page drew slowly: on the RTX
  4090 0.1-5% of the steps were skipped at 30-60 fps on a quiet machine,
  11-25% at load 54-70, a third at 20 fps and half at 15 fps (the lines
  broke 20-24 times per 8 s there), and software rendering at 2-8 fps
  skipped 35-80% (the traces covered 19-94% of their 3 s). Now the worker
  keeps the sensor row of each skipped step (`keepRow`: `session.sensorRow`,
  the same values `frame.sensors` would carry, 400 at most) and sends them,
  oldest first, with the next frame of the same epoch as `frame.sensorRows`
  (`{times: Float64Array, values: Float32Array}`, all older than that frame,
  or null). Every step's row then reaches the page once, in order, with no
  message of its own: a row cannot overtake a newer frame (the history
  drops an older time), a row of the frame's own step is dropped (the frame
  carries it), a reset drops what was kept (the page starts over), and
  initial states have none. A page that predates the field ignores it; a
  frame without it (an older worker) keeps the old rule below. Raising
  `MAX_IN_FLIGHT` instead would have helped the GPU down to about 15 fps but
  not software rendering. sensor_stream.test.mjs runs the real worker in
  Node, hands the frames back late (most steps skipped) through a reset and
  a fall, and finds every step in the card with its values.
- **Drawing.** Two stacked canvases at min(devicePixelRatio, 2): a static
  layer (titles, group glyphs, range labels, zero lines, 1 s guides, limit
  lines, channel keys), redrawn only on a resize, a mode change, a font load
  or a new pixel ratio, and a live layer (traces, latest values). The render
  loop calls `draw(view.playTime, t)` after `drawIfChanged`: the charts' right
  edge is the pose on screen (the last sample interpolated to it). It draws
  at most 30 times a second, only while shown, once the play time has moved
  and new data has arrived; when the play time stands still for 50 ms (a
  pause, a stall) at a time the card does not show, it draws once more, at
  that time (`settled`), so a paused card shows the paused pose (at 30 Hz the
  last drawing could be 33 ms and the values 100 ms behind, a heel strike's
  rise from 0 to 700 N). Then nothing while paused, hidden or off screen. The
  numbers change at most every 100 ms (mono, right-aligned). With the rows of
  skipped steps every sample comes one control step after the one before
  (simulated time stops while paused, and a reset clears the history), so
  the line breaks only where steps are really missing: more than
  `STREAM_GAP_STEPS` (1.5) control steps apart (in practice never: rows go
  only when a state cannot be drawn, and then until the reset). Frames of an
  older worker carry only the steps that went out, and keep the old rule: a
  step of more than `GAP` (50 ms) breaks the line. A trace beyond its range
  is clipped at the plot's edge. The range labels' gutter is as wide as the
  widest label needs (measured in the label font, at least 28 px): with a
  fixed 28 px, '−1600' (28-28.5 px in Inter) lost its minus sign at the
  canvas's left edge, so the force chart read 600 to 1600. The canvases are
  aria-hidden; for screen readers app.js writes the values shown as text
  into `#sensor-summary` (`card.summary()`: "IMU gyro, rad/s: x 0.1, y −0.4,
  z 0.0. ..."), about once a second and at a paused pose's drawing, empty
  while the card is off; `#btn-sensors` names it as its description (no
  live region, so nothing is announced as it changes).
- **Ranges** (fixed; from r3's native rollouts of B and the trace's walking
  percentiles, tightened from the suggested gyro ±8 and moment ±100 for
  readability): gyro ±6 rad/s (walking p1/p99 -4.3/5.5, turns reach 10:
  clipped); accel ±48 m/s² with the ±39.2 clamp dashed and labelled (the
  readings saturate there at every heel strike); force -1600..600 N (F_y down
  to -1375 walking, -2490 in a turn-in-place slam: clipped); moment
  -100..50 N·m (T_z down to -80); contact 0..1500 N (heel up to 1440 walking,
  2085 in a slam); encoders -30..90 deg (knee -4..81, ankle -15..34).
- **Colours** (`SENSOR_GROUPS`, dataviz method; `sensors.test.mjs` checks
  the distances). One hue per group, for the row glyph and the markers, and
  none of the scene's own families (the X-ray arrows' green and violet, the
  orange covers and accent, the slate ghost): IMU sky `#2aa1fe`, ankle load
  cell jade `#18bd91`, foot contact gold `#d4b122`, encoders rose `#cf2b74`
  (2026-11-19, second pass: the first set's plum encoders sat 15.7 (OKLab ΔE
  x 100) from the moment violet and its pink IMU read as the same purple
  family; its azure was 16.4 from the violet, its aqua 15.6 from the force
  green). Against the scene: from the moment violet sky 20.0, rose 25.2, jade
  34, gold 42; from the force green jade 19.9, gold 27.8, sky 32, rose 37;
  from the orange covers gold 16.5 (on the black foot, away from the covers'
  orange, with its own shape and fill), rose 20.3, jade 29.9, sky 36.1; from
  the slate ghost sky 18.0, the others 22 or more; rose 10.4 from a muscle's
  red (hidden in X-ray; the markers sit on the prosthesis). Only one hue fits
  between the green, the violet and the slate ghost at a usable lightness (a
  search over OKLCH: a sky and a teal there come at most 15 apart), so sky
  and jade sit at the two ends of that range and gold and rose fill the
  other two families. Validator, all pairs on the white card: lightness band
  and chroma pass, normal vision 18.5 or more (gold and jade), colour-vision
  deficiency 11.0 or more; sky 2.75:1, jade 2.41:1 and gold 2.08:1 on white
  need relief: every row and line is named, every value printed. A row's
  lines take steps of its hue, darkest first (x y z, heel toe, knee ankle):
  sky `#015e9d` `#007ed1` `#2aa1fe`, jade `#046e53` `#089772` `#18bd91`,
  gold `#876f02` `#b59507` (darker than the gold glyph, which on white is
  only a glyph), rose `#9c0352` `#cf2b74`; each set passes the validator's
  ordinal checks on white. Darker than the glyphs, the lines come nearer the
  arrows: jade lines 11.3-11.4 from the force green, sky 12.5-13.2 from the
  moment violet, the dark gold 13.7 from the green (the validator's floor is
  15). Lifting them to 15 needs mid steps of chroma about 0.085 at L 0.65
  (grey-green, grey-blue) or breaks the steps' hue and lightness ladder (an
  OKLCH search, 2026-11-19), so they stay: each row has its title and glyph,
  the legend and the card sit 60-280 px apart, and sensors.test.mjs keeps
  every line at least 11 from both arrows. The glyph shape is a second channel: circle (IMU),
  diamond (load cell), square (encoders), and a contact circle that fills;
  the card draws each row's glyph as the markers are drawn (a faint dark
  edge, a white ring, the colour), so the light gold reads on white.
  Adjacent on the leg: the IMU (sky) and the knee encoder (rose) on the knee
  cover, the load cell (jade) over the ankle encoder (rose) and the contact
  glyphs (gold) on the foot; sky and jade, the closest pair, never touch.
- **Markers.** One `THREE.Points` (one draw call; six sprites) with a
  ShaderMaterial: a constant size on screen (13.5 CSS px circle, 16 diamond,
  12.5 square, times the renderer's pixel ratio), a faint dark edge (0.75 px),
  a white ring (1.75 px), a core in the group colour, sRGB out, no tone
  mapping, no fog; no depth test, renderOrder 50 (over the X-ray overlay's
  40-45), outside `geomMeshes` (never picked, never aglow), no shadow;
  layered encoders, IMU, load cell, contact: the heel and toe glyphs, the
  only ones that change, on top (seen from behind, the ankle encoder lines
  up with the heel and the load cell with the toe; with the contact glyphs
  at the bottom a third of each was hidden on a desktop and 55-86% on a
  phone, 2026-11-19). In a framed view (a phone) they are drawn
  min(1, max(`minScale` 0.7, scale)) times their size, ring and edge
  included, as the walker is: at the iPhone's 0.48 the six fixed-size
  glyphs covered the knee and ankle units (at 0.6 a square's coloured core
  would be all antialiasing). Each pose writes their positions from the drawn geom's matrix and
  `local`; the heel and toe glyphs are open rings below 2 N, then fill from
  35% and grow by up to 7 px as their force (blended like the pose; a
  non-finite value counts as none) nears 800 N. Prewarmed with the X-ray
  programs. Each contact glyph sits where its zone meets the foot (on a
  collision capsule's axis: "Frame extras", markers); B's toe zone's own
  centre floats 2-3 cm in front of the foot.
- **Layout** (`placeCard`, `layoutToast`, `fitHelp`, `frameWalker` in
  app.js, on every toggle, help change, touch change, resize and font load).
  The card sits in the right column, under the view toggles, under the top
  bar where that spans its column (narrow windows) and under the X-ray legend
  when that is at the top. Variants, in this order: full (264 px wide, as the
  legend; 465 px tall); compact across the width (class `wide`, portrait
  windows only: on a 390 px phone its plots get 250 px instead of 120);
  compact (236 px, about 190 tall). Touch screens whose shorter side is under
  600 px and portrait windows under 480 px wide get no full card. A variant
  must fit above the window's bottom, clear of the back link, the stats, the
  touch pad, the top bar, the view toggles and the legend, and leave the
  walker room (see Framing the walker); of those, the first that keeps the
  walker within 0.12 of the largest scale any of them allows wins (a phone:
  across the width, the walker framed below it, even where compact at the
  right would leave it a little larger squeezed into the strip beside the
  card: 0.48 against 0.57 at 390x664; a 400x420 window: compact at the right,
  the walker moved left at full size). Hidden where none does
  (`sensorCard` 'none'; a toast "No room for the sensor charts here" when it
  is turned on there or shrinks away; back when there is room).
  Hidden at (the layout test's sizes): the three smallest landing iframes
  (382x286, 358x268, 328x246); touch windows 320 px wide up to 600 tall and
  400x360; one keyboard window, 320x420 (no room to frame the walker clear of
  it); with the X-ray legend also every window 360 px tall and windows 320
  px wide up to 540 tall (touch: 720), touch 400 px wide up to 480. The
  toast keeps clear of the column's panels at the top: centred and narrower
  beside them (`html.toast-beside`, over 800 px), left-aligned beside them
  (`html.toast-left`), else under them (`html.toast-below`), and there right
  of the touch pad where the pad comes up to it (`html.toast-pad`,
  `--toast-left`: the iPhone landing iframe, 358x269, put "X-ray view on" on
  the pad's Fast and right keys); `--column-right` and `--column-bottom` on
  the root. The toast is centred by its margins (not a transform), so a
  change of place does not animate. The help yields as to the legend: with
  the full card, the full help (510 px tall at 1280 px wide) fits only on
  windows about 1070 px tall or more; below, the one-line legend, left of the
  card where it would reach it. The X-ray legend sits beside the touch pad
  on touch phones from 330 px wide (it was 372 px: at 360x740 the 120 px
  legend stayed at the top, pushed the card down and left the walker framed
  at 0.42 instead of 0.58).
- **Framing the walker** (`frameWalker` in app.js, `setFraming` in
  render/scene.js; 2026-11-19). On a portrait phone there is no free corner
  for a 190 px card: at 390x844 the walker, seen from the chase camera,
  spans x 50-315 and y 172-714 (its geoms over 3 s of walking) and the pad
  y 722-832, so the card at the top right covered its head and torso (15%
  of its box; 7% of it, the right arm, under the full card on a 768x1024
  tablet), and beside the pad it would cover the prosthesis foot, where the
  contact glyphs are. So the view moves instead. With the default camera the
  walker fills about 0.17-0.88 of the window's height, centred, 0.16 of the
  height to either side; where the view panels cross that zone (the card or
  the legend in the right column at the top, the legend beside the pad), the
  view is moved, and shrunk if it must, into the free area that keeps it
  largest: left of the column (a tablet: 64 px to the left at full size),
  below it (a phone: under the card at 0.68 of its size at 390x844, 0.58 at
  360x740), or above a legend at the bottom (touch windows about 400 px wide
  and 600-720 tall: about 0.85), always under the top bar and above the touch
  pad; not below 0.4 of its size. `setFraming` is a view offset on the
  camera's projection (a 2-D zoom and pan), so the camera, the orbit and the
  zoom stay the visitor's, and picking, `projectToClient`, the markers and the
  pull overlay follow it (the muscles' pick margin scales with it). It
  changes at once, with the card. Desktop windows from 640x480 up: no
  change, or a shift of a few pixels. In a narrow keyboard window (390x844)
  the card across the top moved the walker down behind the full help (87 of
  its 103 geoms under it): where the walker is framed, a keyboard's full
  help that would cover it shows as the one-line legend instead. Left as
  they are: OrbitControls pans (free camera) by the unframed projection, so a
  framed scene follows a finger at `scale` of its travel; on a phone the
  toast sits under the card, over the framed walker's head (the pause notice
  too, while paused).
- **Checked** (2026-11-19). `sensors.test.mjs`: the rows against the asset
  folder's channels (also named by their reads), the colour distances (each
  group 19.5 or more from both arrows, 15.5 from the orange and the slate
  ghost; all pairs 15 normal, 8 CVD; each line 11 or more from both arrows;
  the steps' hue and order), the history ring, the card's intake (rows of
  skipped steps, initial frames, the line's break rule, no matching row).
  `sensor_stream.test.mjs`: every control step reaches the card (above).
  e2e_browser "sensor charts": G and the button, every plot inked, every
  range label inside the canvas, the screen-reader summary, contact in
  stance, a sample for every 10 ms and no break after 3 s of walking, at
  most 30 draws a second, a paused card at the paused pose and then still, a
  paused toggle drawing the view (and the card) once, a reset clearing the
  history, both views on, the markers within 1.5 px of
  `projectToClient(geomPoint(geom_id, local))` (0.000 px measured),
  `?sensors=1` (the reset pose behind the loader not charted, its contact
  glyphs open); its touch test (iPhone 13): the card across the top, the
  walker framed below it, none of its geoms under the card while walking,
  the markers between the card and the pad, no framing once off; its layout
  test, 6,384 window sizes with and without the legend and the card (the
  card full at 1,750, compact at 1,288, hidden at 154), no overlaps, and the
  walker's zone clear of the card everywhere and of the legend except in the
  328x246 landing iframe (no room to move it there). render_shots: each
  marker at its sensor in MuJoCo (site or joint anchor; a touch zone's where
  the zone meets the foot) to 0.0000 mm, the centre pixel of each filled
  glyph exactly its colour inside a white ring, the contact glyphs by force
  and on top, the sensor frame blended with the pose, a frame without
  sensors, an initial frame (open glyphs), non-finite values, no program
  compiled by the first toggle, one draw call and the baseline again when
  off, never picked, over the X-ray arrows, the glyph sizes under framing.
  compat_browsers: the card drawn and the markers on their geoms in
  Chromium, Firefox and WebKit, the phones' compact card on screen and clear
  of the other panels.

## Demo page DOM contract (`demo/index.html`, used by `demo/js/app.js`)
`#viewport` (renderer host, full window; app.js catches `pointerdown` on it in
the capture phase, so a press on the body never reaches OrbitControls on the
canvas. The renderer adds its canvas, whose `style.cursor` is `grab` over the
body and `grabbing` during a drag, and `svg.pull-overlay`: aria-hidden, no
pointer events, under the HUD panels, holding the pull's line, its halo, the
dot on the body and the target ring); `#loader` overlay with `#loader-title`,
`#loader-status`, `#loader-bar` (fill element, width in %), `#loader-size`,
`#start-btn`; `#hud` (inert until Start) with `#hud-command` (the command
label: `Starting…` or `Stopping…` while command shaping holds a walk back,
`data-shaping` says which rule), `#hud-speed`, `#hud-rtf`, `#hud-backend`,
`#hud-note` (shown while inference is on the CPU, worded from
`status().backendReason`) and the developer rows `#hud-clip`, `#hud-time`
(`data-debug`, shown with `?debug=1`); `#help` (controls, hidden until Start,
toggled by H; `data-mode` full or compact; `kbd[data-key=...]` get class
`pressed` while held; elements with `data-label="<code>"` show what the
visitor's keyboard layout prints for that code; a "Drag: Pull a body part" row,
`span.pad-key.drag-key`, in both the key grid and the touch grid, the note
"Pulled off balance, it has to catch itself." and "Drag body to pull" in the
one-line legend; an "X-ray view" row, `kbd[data-key="KeyX"]`, in the key grid
and "X X-ray" in the one-line legend; a "Sensor charts" row,
`kbd[data-key="KeyG"]`, in the key grid and "G sensors" in the one-line
legend; in the touch grid a `.xray-row` and a `.sensors-row` with the view
buttons' icons (`span.pad-key.icon-key`), left out below 460 px tall;
`.muscle-note`, hidden in X-ray); `#touch-pad` (touch
steering: `button[data-code]` hold `KeyW`, `KeyA`, `KeyS`, `KeyD`,
`#pad-fast` toggles top speed); `#toast`; buttons
`#btn-reset`, `#btn-pause`, `#btn-camera` (`data-mode`), `#btn-help`; in
`#hud`, `#viewbar` (a `role="group"` pill, top right) holding the view
toggles (`.tool`, `aria-pressed`, independent: both can be on) `#btn-xray`
(title "X-ray view (X)") and `#btn-sensors` (title "Sensor charts (G)",
`aria-describedby="sensor-summary"`); the buttons that name a key carry
`data-title` and `data-title-key` (a KeyboardEvent.code): app.js writes their
`title` as `<data-title> (<the key's label on this layout>)`;
`#xray-legend` (an `aside` with a fixed `aria-label`, `hidden` unless the
X-ray view is on, no pointer events): `.xl-title` ("Socket load" + `.xl-more`
" on the residual limb"), `dl.xl-rows` whose `dd`s hold `#xray-force` /
`#xray-moment` (`.xl-num`, app.js writes rounded |F|, |T|, "–" without a
wrench; no live region) and `.xl-unit`, the swatches
`svg.xl-swatch.xl-force|xl-moment` (colour `--xray-force`, `--xray-moment`,
which app.js sets from the renderer), row names `.xl-label`, and `.xl-note`;
`#sensor-card` (an `aside` with a fixed `aria-label`, `hidden` unless the
sensor charts are on and there is room, no pointer events; app.js sets its
`style.top`, `data-mode` full | compact and the class `wide`, compact across
the window's width): `.sc-title` ("OSL sensing" + `.sc-span` " · last 3 s",
shown compact), `.sc-sub`, `.sc-plots` (ui/sensors.js adds `canvas.sc-static`
and `canvas.sc-live`, aria-hidden), `.sc-note` (frames without sensor data,
or no chart for their channels; ui/sensors.js writes the text), `.sc-foot`
and `#sensor-summary` (`.sr-only`: the latest values as text, app.js about
once a second and on a pause, empty while the card is off; no live region);
`#error` overlay with `#error-text`; `#back-link` (hidden when `?embed=1`).
The page shell sets `html.embed`, `html.touch` (coarse pointer),
`html.no-focus` and `html[data-phase]`; app.js sets `html.xray` (X-ray on),
`html.sensors` (sensor charts on), one of `html.toast-beside`,
`html.toast-left`, `html.toast-below` (where the toast goes beside the right
column's panels at the top), `html.toast-pad` with `--toast-left` (px: under
the column it would reach the touch pad, so right of it) and
`--column-right`, `--column-bottom` (px) on the root, and the class `beside`
on `#help` (the one-line legend left of the right column).

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
- The user makes every git commit and push; tools and agents prepare the
  working tree and stop there.

## Findings worth knowing (2026-11-16 to 2026-11-19)
- **Sensor charts (2026-11-19; "Sensor charts" above).** The dataviz
  reference palette has no four free slots in this scene: green and violet
  are the X-ray arrows', orange the covers', and of the rest blue sits 10.5
  (OKLab ΔE x 100) from the violet and 11 from the slate ghost, magenta 14.7
  from the orange (the IMU's glyph sits on the orange knee cover), yellow and
  red 11-12 from it. The first set (azure, pink, aqua, plum) passed the
  validator, but with both views on its plum encoders (15.7 from the moment
  violet) and pink IMU read as the moment arrow's purple. A search over OKLCH
  (every hue, lightness 0.45-0.77) shows how little room there is: between
  the force green, the violet and the slate ghost only one blue-green fits at
  a usable lightness (a sky and a teal there come at most 15.1 apart), so the
  set takes the two ends of that range (sky 20.0 from the violet, jade 19.9
  from the green, 20.1 apart) and the two families left, gold and rose; the
  best set's worst distance is about 19-20 except gold to the orange (16.5),
  which therefore marks the foot, not a cover. The glyph shape stays a second
  channel. A card
  drawn at most 30 times a second lags the pose by up to 33 ms, and its
  numbers (changing at most every 100 ms) by up to 100 ms: paused at a heel
  strike (0 to 1000 N in about 18 ms), the card read 0 N under a loaded
  heel; hence the one drawing at a standing play time. A ResizeObserver
  fires when a hidden card shows, after the toggle's own drawing: it now
  redraws only for a new size (a paused toggle draws the card once). Pixel
  readbacks of the card's canvas (`getImageData`, in the tests) can move
  Chrome's 2D canvas off the GPU: the walking draws' p95 read 3.5 ms in a
  run that read pixels before, 0.6 ms without. The full help (510 px tall at
  1280 px wide) fits beside the full card only on windows about 1070 px tall
  or more; below, it shows compact. The toast, centred with a transform,
  slid across the screen when a resize moved it between beside, left and
  under the column (the layout test caught it over the legend mid-slide); it
  is now centred by its margins. Cost on the RTX 4090 (ANGLE on Vulkan,
  WebGPU inference, `nice -n 19` on a shared machine at load 40-60): walking
  with W held, alternating 4 s runs, nothing on 56.6 fps and RTF 0.86 (4
  runs; one at 49.4 fps and 0.46 under load), X-ray 53.3 and 0.84, sensor
  charts 59.9 and 0.98, both 60.0 and 0.97 (2 runs each); the card's draws
  0.32 ms mean, 1.20 ms p95, 3.5 ms max (642 draws over the runs); profiled
  alone, drawing the traces took 0.10 ms (median; p95 0.30) and the traces
  with the numbers 0.30 ms (p95 0.60). The markers are one draw call. A
  review screenshot with both views read RTF 0.68, so the cost was measured
  again (second pass, same GPU, load average 81-93, alternating 5 s runs of
  W held, 4 rounds): nothing on 44.3 ± 3.1 fps and RTF 0.38 ± 0.09, sensor
  charts 44.8 ± 2.6 and 0.43 ± 0.18, both 45.2 ± 6.3 and 0.46 ± 0.13, X-ray
  45.8 ± 4.3 and 0.39 ± 0.12, the card hidden (markers only) 44.8 and 0.39,
  the card without its backdrop blur 48.1 ± 4.5 and 0.40 (within the noise,
  so the blur stays). Within each mode the RTF swung 0.27-0.70 with the
  worker's inference time (4-28 ms a step, the shared CPU's load): the 0.68
  was the machine, not the card. The card's draws: p95 0.4-1.0 ms. One cost
  was real and is gone: the card read its plots box's size (`clientWidth`)
  at every draw, which forces a synchronous layout after any HUD text change
  (the legend's numbers, the status rows); it now keeps the size its
  ResizeObserver reports. The e2e GPU test compares the modes over 2 runs
  each: at load 90 and more it failed once (X-ray RTF 0.35 against 0.46 off,
  sensor charts 0.36, both 0.31) and passed on the rerun (X-ray 0.61, sensor
  charts 0.42, both 0.42, off 0.34; chart draws p95 0.40 ms).
  The charts used to get a sample only with a posted frame: a step whose
  frame the worker could not post (6 in flight) was lost to them. On the
  GPU, unloaded, samples came 10 ms apart, sometimes 20-40 ms (8 s of
  walking: 299, 15, 4 and 1 gaps of 10, 20, 30 and 40 ms), so lines never
  broke; a main thread starved for a moment (`nice -n 19` at load 50-60)
  lost a few steps in a row and the lines broke there; in software-rendered
  Chromium (1.5-8 fps) frames came in bursts 0.1-0.35 s apart and the traces
  were dashed. Since 2026-11-19 the worker sends the sensor rows of the
  skipped steps with the next frame (`frame.sensorRows`; "Sensor charts",
  The sensor stream), and the charts have every step.
  The reset state's frame (the first of every epoch, and the ready screen's)
  read as a measurement: an unloaded socket (26 N) and a 3 g IMU on a
  standing walker, frozen behind the loader with `?xray=1` or `?sensors=1`
  and after R while paused, and a 30 m/s² spike at the start of every
  reset's chart; it is now marked `frame.initial` and shown as no data
  ("Frame extras", the reset state).
  The e2e project-page test's "no steps off screen" check raced the batch
  the worker was running when the pause came (1080 against 1070 steps, as
  the X-ray step saw 1170 against 1160): it now counts from 600 ms after the
  pause.
- **X-ray view (2026-11-19; "X-ray view" above).** The grab glow used to
  `clone()` the mesh's material and set `emissive`: on a ghost ShaderMaterial
  that throws in the pointer handler (the error overlay ends the demo), and a
  clone drops `onBeforeCompile`. Hence one material resolver and factories.
  Without a prewarm, three would compile the ghost and overlay programs at
  the first X-ray drawing, inside an animation frame; with it the first
  toggle compiles none (render_shots counts `info.programs`). A session that
  started in the X-ray view (`?xray=1`) still built three programs at its
  first X (the muscles', their shadow depth program and the socket's base
  one: 26-43 ms on a quiet machine); `renderer.compile()` builds no depth
  program, so the prewarm now also draws the scene once without X-ray inside
  a 1-pixel scissor (2026-11-19). In stance the
  force acts along the residual femur, so from behind or beside the walker
  the force arrow covers the femur inside the cup: shafts stay thin (6.5 mm
  at most) and the femur's ghost is stronger and blended after the socket's,
  so its outline still reads. Cost on the RTX 4090 (ANGLE on Vulkan, WebGPU
  inference, `nice -n 19` on a shared machine at load 60): walking with W
  held, alternating runs of 4 s, X-ray on 59.7 fps and RTF 0.86 against 54.3
  fps and 0.80 off (4 runs each; run-to-run spread about 10 fps and 0.15
  RTF under that load); `render()` plus a 1-pixel readback of the stride pose
  at 1280x720, 3.3 ms on against 4.2 ms off.
  Testing the layout at the project page's iframe sizes (never tested below
  360 px tall) found the touch help, which sits above the pad, reaching the
  top bar at 382x286 and below; the help now gives way to the top bar and the
  view toggles as it does to the legend. The one-line help with "X X-ray"
  reached the stats panel at 861-1099 px (both bottom corners); it now wraps
  before them.
- **Bundle B replaces R2 (2026-11-18).** B's `model.mjb` differs from R2's
  only in `site_pos`/`site_size` of the three touch zones mjlab's
  `feet_extended` widens: site 1811 (the OSL heel zone, sensor `robot/r_foot`,
  now foot x -0.055 to 0.075) and the intact foot's 1944 and 1946 (8 mm
  lower). Every id, count and sensordata address is the same, and
  `visual.bin.gz` re-exports byte-identical. R2's heel zone ends short of the
  heel capsules, so `r_foot` reads 0 N whenever the OSL heel is down: a folder
  with R2's `physics.mjb.gz` under B's policies runs without an error while
  B's heel contact input reads 0 N. Hence `sections`, the exporter's refusal
  of mixed exports and `contract.test.mjs`. The trace
  (`tools/traces/trace.bin.gz`) also records, at every control boundary, the
  OSL observer's undelayed sensor frame (18 values for B), `cfrc_int` of the
  socket body, the subtree COM of its tree root and the socket's `xpos`.
  The `e2e_session` behaviour checks were tuned on R2, and on B two failed at
  first: S for 3 s backed 0.83 m (0.46x the commanded -0.6 m/s; the check
  wanted 0.5x), and the unpulled "walking, chest" drag run, W held from the
  reset, fell 1.96 s in. Command shaping was then re-tuned for B
  (2026-11-18; session.js header): B's start window is [0.3, 0.5] (R2's
  [0.2, 0.5] let 15 of 253 first walks fall on B), which fixed the drag run,
  and the S check counts after the first 1.5 s, since B, backing up out of
  the schedule's turn in place, hardly moves in its first second (0.62 m in
  3 s, 0.54-0.55 m of it after 1.5 s: 0.60x). e2e_session passes on B
  (WebGPU and CPU).
- **Falls come from the policy, not the port.** SimCore matches osl_deploy's
  Simulation to rounding on a 13.5 s trace: the reference state is exact, the
  human observation is within 3.5e-17 (R2, 2026-11-16; B, 2026-11-18: 1.5e-17),
  the OSL input is exact (on B including the contact term). A closed-loop
  study on R2 (2026-11-16) found int8 human weights fall exactly as often as
  fp32 on identical key sequences; fp16 diverges on the same borderline trials.
  The remaining falls on R2 were policy limits at some transitions: walk to
  turn in place, short taps from the reset stance, fast restarts after backing
  up. DemoSession relaxes the
  model for 1 s after a fall, then resets. The first walk after a reset waits
  for the idle clip's phase to reach B's start window: forward and turning
  arcs 0.3-0.5 (at most 0.39 s), backward 0.3-0.36 (at most 0.46 s; next
  item).
- **Command shaping: the first walk backward** (B, 2026-11-19). At -0.7 m/s
  (levels 4 and 5) B's first backward stride out of the reset stance falls
  when it starts at idle phase 0.40-0.68 (frames 20-34), in any idle cycle:
  with the start gate off, 74 of 215 starts there fell 1.7-3.4 s later (S+A
  53 of 81) and none of 470 elsewhere; at -0.5 and -0.6 m/s none of 753
  fell. The forward window [0.3, 0.5] takes in frames 20-25: with keys
  pressed every 10 ms over the first 2 s after a reset (S, S+A, S+D at levels
  1-5, 3000 first walks), 40 fell (1.3%, all at levels 4-5). Backward first
  walks therefore have a window of their own, `BACK_START_WINDOW` [0.3, 0.36]
  (frames 15-18, two short of the earliest fall), and `startWait` went from
  0.4 to 0.5 s: on the same 3000 presses none fell (the 40 that differ all
  fixed, none broken: exact sign test p 1.8e-12), and a backward first walk
  waits 0.22 s on average and 0.46 s at most (0.16 and 0.39 s before).
  Forward first walks are untouched (288 runs bit-identical), and 78 of 80
  paired key-masher seeds ran identically (the other two pressed S at frames
  19-25; no fall either way). Serving -0.6 m/s for the first 0.6 s of such a
  walk also fell 0 times, but adds a rule with state and a speed the HUD does
  not show; capping levels 4-5 backward at -0.6 m/s would drop that speed.
- **Command shaping: forward after backward** (R2, 2026-11-16; re-measured on
  B, 2026-11-18). On R2 a seeded key-masher benchmark (casual and stress
  profiles, plus paired transition trials on WebGPU) found that every fall
  starts a new walk while the body still carries the last one's momentum.
  Walking forward out of a backward walk was R2's weakest transition: 3 falls
  in 20 trials without a rule, 0 in 32 with it. So W while backing up, or
  before the body has come to rest after it, first stands, and the forward
  walk starts once the pelvis is slower than 0.35 m/s, at most 1.0 s after
  the key (`SHAPING` in session.js). The HUD says `Stopping…` then
  `Starting…` meanwhile. On B (186 paired sequences): of 130 direct
  reversals none fell with the rule and 4 without it, and without it the
  body drifts backward (under 0.3 m forward in 3 s: 33 against 12); after a
  stop it costs falls in one family (S 2.63 s at level 5, a 0.05-0.4 s stop,
  then W: 5 with it, none without; all 56 after a stop: 6 against 1); the
  masher falls as often either way (47 and 45 in 130 seeds x 30 s). It is
  kept. On R2 the same guard for forward to backward, a stand before walk to
  turn, a minimum gait dwell and a fast-start ramp showed no benefit (the
  falls moved into the stop, or none happened either way), so they were not
  adopted, and the capped speed table (1.2 m/s forward, -0.7 m/s backward)
  stays. B's key-masher rates with this shaping: 5.7 falls per 10 min
  (casual, 40 in 70 min) and 7.8 (stress, 51 in 65 min). Stopping out of a
  fast walk or arc still falls now and then.
- **Cache busting.** GitHub Pages sends `Cache-Control: max-age=600`, so for
  10 minutes after a deploy a returning visitor could pair a new `demo.json`
  with an old cached asset (a different observation layout fails with
  "expected N inputs"). The exporter writes `versions` (content hashes) into
  `demo.json`; the demo fetches `demo.json` with `cache: 'no-cache'` and every
  asset as `<file>?v=<version>`. JS modules and `demo/vendor/` are not
  versioned: code can lag a deploy by up to 10 minutes. So a renamed asset
  folder breaks a cached old `app.js` (HTTP 404 on the old `demo.json`, the
  error overlay until a reload). The swap to B (2026-11-18) renamed
  `demo/assets/r2/` to `demo/assets/stage2/` and removed R2 entirely (user
  decision: no fallback and no switching between controllers), accepting
  that window.
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
- **Backward steering: a car in reverse.** The first key table turned the body
  with A/D also while walking backward (tank-style: S+A yawed left, so the
  path curved right), which the user found flipped. Since 2026-11-16 S+A
  backs along a path that curves to the walker's left while the body yaws
  right (wz = -back_arc), like a car reversing with the wheel turned left;
  S+D mirrors it (`decimal_twist` in `tools/duet_export/simcfg.py`; the same
  clips are reached, so `reference.bin.gz` did not change). In the demo, with
  the camera held behind the walker, S+A comes toward the camera and leaves
  the screen on the left: 1.8 m back, 0.9 m to the left and -1.04 rad of yaw
  in 4 s at level 3 (R2, 2026-11-16); S+D leaves on the right. The help keeps "Turn left /
  right" for A/D, and the HUD says "Backward, curving left/right".
- **Pulling: a damped spring, MuJoCo's viewer perturbation retuned**
  (`DragSpring` and `DRAG` in session.js, our own code after
  `mjv_initPerturb` / `mjv_applyPerturbForce` in MuJoCo 3.11's
  `engine_vis_interact.c`, Apache-2.0). At every physics tick, also through
  a fall's 1 s collapse, the grabbed point p is pulled toward the target:
  F = k m (target - p) - 2 zeta sqrt(k) m (v_p - v_root), capped, applied at
  p (force plus torque (p - xipos) x F on the geom's body, in
  `xfrc_applied`). k is `vis.map.stiffness` (100 1/s²) and m the effective
  mass at p when the grab starts, the viewer's `localmass`
  3 / trace(J M^-1 J^T): equal to native MuJoCo to 6 digits, 25 kg at the
  pelvis, 33 kg at the chest, 10 kg at the head, 4 kg at a forearm, 1 kg at a
  hand. The viewer's own spring is far too strong for the trunk (0.35 m on
  the standing pelvis: 660 N, a fall), so m counts at most 3 kg (k m <= 300
  N/m); zeta = 1 acts on the point's velocity relative to the walker's root
  (the viewer damps the absolute velocity at a ratio of 0.5, which brakes a
  walking body: 85 N against 38 N holding a walking pelvis 0.1 m off);
  |F| <= 50 m/s² times the effective mass (a 3 m drag on a hand: 43 N, not
  240 N) and <= 400 N. The viewer's torsional damping is left out. On R2
  (2026-11-16): standing, 0.35 m on the pelvis, chest or head (90-105 N)
  stumbles and recovers, and 0.8-1 m knocks it over; walking at 1 m/s, 0.35 m
  on the pelvis recovered in 7 of 8 trials and 0.6 m fell. The damper leaves the whole body's motion
  alone by design, so a walker that steps toward the target can overshoot it
  by up to about 0.2 m before the spring pulls it back. Standing, a pull on
  the planted prosthesis foot hardly moves it (friction holds it). Without a
  drag the closed loop is bit-identical to before.
- **Pulling: the camera holds still during a drag.** The target is the cursor
  ray met with the plane through the grabbed point, square to the view,
  recomputed every animation frame so that it stays under the pointer. With
  the camera following the walker, the camera carried the target along and a
  held pull never let up: a 160 px pull held for 1.5 s dragged the walker
  1.2 m, reached the 400 N cap and felled it. With `holdCamera`, the same drag
  moved it 0.66 m while the force fell from 82 N to 8 N, and it caught itself
  (R2, 2026-11-16).
  The camera glides back after the release. While walking, a pointer held
  still pulls increasingly down and sideways as the walker walks away into
  the screen.
- **Pulling: picking.** `pick()` casts the cursor ray on the render meshes at
  the pose on screen: the slim physics model has no OSL mesh faces, so
  `mj_ray` would miss the prosthesis. Triangles are grouped in runs of 128
  with a box each: 0.04 ms per pick on the body, 0.3 ms for a miss (the 4 px
  mouse slop casts 25 rays). 31% of the grabbable pixels show a muscle with
  no bone behind it, so a muscle hit grabs the nearest bone at the clicked
  point (unless a bone lies within 8 cm behind it), within that bone's
  bounding radius + 3 cm; the worker accepts up to + 5 cm. `local` is in the
  geom's frame, as the render meshes are stored.
- **Pulling: touch.** A finger already on the touch pad is the primary
  pointer, so a finger on the body is not, and `isPrimary` cannot decide which
  finger may grab. app.js counts the touch pointers down on the canvas
  instead: a finger alone on the canvas grabs, so one thumb can hold ▲ while
  the other pulls; a second finger on the canvas makes a pinch, and does
  nothing during a drag. In Chromium a second finger's tap fires no click, so
  Reset and Pause do not respond while another finger drags.
- **Pulling: a release the page never sees.** Chromium and WebKit deliver the
  pointerup of the captured pointer when the button is released outside the
  window, or outside the iframe on the landing page. Playwright's Firefox does
  not; it sends a pointermove with no buttons instead. So app.js also ends a
  drag on a mouse move with the left button up, besides window blur, a hidden
  tab and the frame scrolling out of view. A drag the page still holds on to
  ends at the next press and release.
- **Render on demand and interpolation.** The worker posts the pose of every
  10 ms control step with its simulated time, and the main thread draws the
  walker between the two poses around its own clock. Before, poses went out
  at about 50 Hz on a 60 Hz display and 13-17% of frames repeated one. The
  renderer draws only when something changed (a new pose, the camera easing,
  orbiting or zooming): nothing while the embedded frame is scrolled out of
  view (an IntersectionObserver also pauses the simulation then), and while
  paused only when the visitor orbits or zooms.
- **WASM bindings: output arrays.** An output argument (a `val` parameter in
  the generated bindings, e.g. `mj_objectVelocity`'s `res`, `mj_jac`'s
  `jacp`) takes a TypedArray's `byteOffset` as an address in the wasm heap.
  A plain Float64Array's byteOffset is an offset into its own buffer (usually
  0), so MuJoCo writes at that heap address: memory is silently corrupted and
  the array never sees the result. Only views of the wasm heap (`data.qpos`,
  `buffer.GetView()`) or a `mujoco.DoubleBuffer` are safe outputs. Input
  arrays (`NumberArray` parameters such as `mj_jac`'s `point`) are copied, so
  plain arrays are fine there. SimCore computes body velocities in JS;
  DragSpring keeps its Jacobian scratch in DoubleBuffers.
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
