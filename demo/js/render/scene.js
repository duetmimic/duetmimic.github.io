// Main-thread three.js renderer for the DuetMimic demo, in the paper-figure look
// of osl_deploy's figure tool: light background, light-grey checker floor with
// soft shadows, warm bone, dark OSL hardware with orange covers, muscles as red
// capsules shaded by activation.
//
//   const view = new DemoRenderer(hostElement, demo.visual, visualContainer, options);
//   view.setFrame(frame);           // render/frame.js output, at any rate
//   view.tick();                    // once per requestAnimationFrame: draws only when something changed
//                                   // (= view.update(dt?) then view.drawIfChanged())
//   view.render();                  // update and always draw (or render(dt), seconds)
//   view.setCameraMode('follow');   // 'follow' | 'chase' | 'free'
//   view.resetView();               // put the camera back on the default view
//   view.dispose();
//
// Pulling a body part (app.js drives it; the spring itself runs in the worker):
//   view.pick(clientX, clientY, slop?)        // {geom, body, local, point, distance} | null
//   view.dragTarget(clientX, clientY, grab)   // [x, y, z]: where the pull points now
//   view.setPerturb({geom, local, target, force?, maxForce?} | null)  // draw the pull, light the body
//   view.setControlsEnabled(false)            // no orbiting during a drag
//   view.holdCamera(true)                     // nor following: a held pull stays put
//
// visualContainer is loadContainer(visual.bin.gz) ({meta, arrays}) or its arrays.
// World coordinates are MuJoCo's (z up, metres); the camera's up is +z.
// Camera modes: 'follow' keeps the user's orbit and zoom and translates with
// the pelvis (smoothed); 'chase' also swings the camera behind the walker as it
// turns; 'free' stops following and enables panning.
//
// Picking: pick() casts the ray under the cursor at the pose on screen (the
// interpolated one). Bones and prosthesis parts are hit on their triangles
// (split into runs of PICK_CHUNK triangles with a box each, so a ray tests a
// few hundred triangles, not the 25k of the OSL knee). Muscles count too, as
// their drawn capsules widened by PICK_MUSCLE_PX: a muscle hit grabs the
// nearest bone at that point, unless the ray meets a bone just behind it
// (within PICK_MUSCLE_DEPTH). `local` is the grabbed point in the geom's
// frame, MuJoCo's: world = geom_xpos + geom_xmat · local, as the render meshes
// are stored. dragTarget() meets the cursor ray with the plane through the
// grabbed point's current position, square to the view direction, so the
// target stays under the cursor (and at the grabbed point's depth). During a
// drag the camera holds still (holdCamera): following the pulled walker, it
// would carry the target along, and a pull held still would never let up (a
// 160 px pull held for 1.5 s dragged the walker 1.2 m and felled it). The pull is
// an SVG overlay over the canvas (no depth test, widths in CSS px): an
// orange line from the grabbed point to the target, with a dot at each end,
// wider the harder the worker reports it pulls; the grabbed body glows.
//
// Frames and interpolation. A frame without a time is shown as it is at the
// next update. A frame with frame.time (simulated seconds, increasing) and
// frame.epoch (the worker's reset count) is queued: updates play the queue back
// on a clock that trails the newest frame by `delay` (about one control step),
// so the two frames around the time drawn have always arrived, and pose the
// walker between them: geom positions, muscle end points and the root are
// blended linearly, geom rotations by normalised quaternion blending. The
// clock runs at the rate frames arrive (the simulation's real-time factor),
// eases toward its target, never runs backwards and never passes the newest
// frame. A frame from a new epoch (a reset) clears the queue: no blending
// across a jump. options.onRelease(frame) is called once the renderer no
// longer reads a frame, so its buffer can go back to the worker.
//
// Render on demand: tick() draws only when the pose, the camera, the light or
// the canvas size changed since the last drawing, so a paused walker with a
// settled camera costs no GPU work.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DEFAULTS = {
  maxPixelRatio: 2,
  antialias: true,
  shadows: true,
  shadowMapSize: 2048,
  shadowExtent: 2.6,        // half-width of the key light's shadow box (m)
  followRate: 6.0,          // 1/s: how fast the camera target catches the pelvis
  chaseRate: 1.6,           // 1/s: how fast 'chase' swings behind the walker
  chaseAzimuth: 180,        // deg from the walker's facing to the camera in 'chase':
                            // directly behind the walker
  chaseElevation: -20,      // deg of the 'chase' camera (< 0: above the target, looking
                            // down): slightly above the head at the default 3 m
  delay: 0.012,             // s: timed frames play back this far behind the newest
  onRelease: null,          // frame => {}: the renderer no longer reads this frame
  fog: [12, 60],            // near, far (m)
  floorSize: 200,           // m; the floor recentres under the walker
  hemiIntensity: 1.0,
  keyIntensity: 2.5,
  fillIntensity: 0.7,       // headlight, like MuJoCo's
  exposure: 1.0,
  preserveDrawingBuffer: false,
  controls: true,
  // The pull overlay (sRGB hex): line, its halo and the dot on the body; the
  // line widens from width[0] to width[1] CSS px as the force reaches full
  // (setPerturb's maxForce, else forceRef newtons). Then the grabbed body's glow.
  pull: { line: '#c2410c', halo: '#ffffff', grab: '#f26a1b', width: [2.25, 4.5], forceRef: 400 },
  glow: { color: '#f26a1b', intensity: 0.35 },
};

// Picking (see the header).
const PICK_CHUNK = 128;        // triangles per box
const PICK_MUSCLE_PX = 3;      // CSS px added to a muscle capsule's radius
const PICK_MUSCLE_DEPTH = 0.08; // m: a bone this close behind a muscle hit is taken instead
const PICK_REACH = 0.25;       // m: a muscle hit further than this from every bone grabs nothing
// m: a muscle hit grabs a geom only within this of the geom's bounding sphere
// about its origin (the worker refuses points beyond geom_rbound + 5 cm).
const PICK_RBOUND_MARGIN = 0.03;
const PULL_FORCE_TAU = 0.15;   // s: smoothing of the reported force (it arrives about 4 times a second)
const SVG_NS = 'http://www.w3.org/2000/svg';

const CAPSULE_EPS = 1e-3;   // half-height of the capsule's own cylinder (unit radius)
const LUT_SIZE = 256;

// Playback clock (timed frames): how fast it eases toward newest - delay, how
// far behind it may fall before it jumps (a stall), the window over which the
// arrival rate is measured, and the most frames it keeps.
const PLAY_TAU = 0.3;       // s
const PLAY_MAX_LAG = 0.25;  // s of simulated time
const RATE_WINDOW = 1.0;    // s of wall time
const RATE_GAP = 0.25;      // s: an arrival gap this long (a pause) restarts the rate window
const MAX_QUEUE = 8;
// A blended muscle segment whose end moved further than this between the two
// frames is a different segment (the wrap path changed): it is not blended.
const SEGMENT_JUMP2 = 0.2 * 0.2;
// Render on demand: changes below these are invisible (about 0.1 px at 3 m).
const DRAW_EPS_POS = 2e-4;  // m
const DRAW_EPS_ROT = 1e-8;  // 1 - |q.q'| (about 3e-4 rad)

const GEOM_SPHERE = 2, GEOM_CAPSULE = 3, GEOM_ELLIPSOID = 4, GEOM_CYLINDER = 5, GEOM_BOX = 6, GEOM_MESH = 7;

function srgb(rgb) {
  return new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
}

// Unit quaternion [x, y, z, w] of the rotation matrix m[o..o+8] (row-major, as
// MuJoCo's xmat), into q (Shepperd's method, as THREE.Quaternion uses).
function quatFromRows(m, o, q) {
  const m00 = m[o], m01 = m[o + 1], m02 = m[o + 2];
  const m10 = m[o + 3], m11 = m[o + 4], m12 = m[o + 5];
  const m20 = m[o + 6], m21 = m[o + 7], m22 = m[o + 8];
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    q[0] = (m21 - m12) * s; q[1] = (m02 - m20) * s; q[2] = (m10 - m01) * s; q[3] = 0.25 / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    q[0] = 0.25 * s; q[1] = (m01 + m10) / s; q[2] = (m02 + m20) / s; q[3] = (m21 - m12) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    q[0] = (m01 + m10) / s; q[1] = 0.25 * s; q[2] = (m12 + m21) / s; q[3] = (m02 - m20) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    q[0] = (m02 + m20) / s; q[1] = (m12 + m21) / s; q[2] = 0.25 * s; q[3] = (m10 - m01) / s;
  }
}

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// Muscle capsules: one unit-radius capsule whose two hemispheres are pushed
// apart by the per-instance half-length (in radii), so a uniform scale keeps
// round caps. Patched into both the lit and the shadow-depth shaders.
function stretchCapsule(shader) {
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', '#include <common>\nattribute float aHalf;')
    .replace('#include <begin_vertex>',
      `vec3 transformed = vec3( position );\ntransformed.y += sign( position.y ) * ( aHalf - ${CAPSULE_EPS.toFixed(6)} );`);
}

// The floor: a checker computed from the world position and box-filtered over
// each pixel's footprint (Quilez, "filterable procedurals"), so it fades to its
// mean grey at grazing angles instead of aliasing, with no texture or mipmaps.
function checkerFloorMaterial(colors, square) {
  const material = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const uniforms = {
    uCheckerA: { value: srgb(colors[0]) },
    uCheckerB: { value: srgb(colors[1]) },
    uInvSquare: { value: 1 / square },
  };
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uInvSquare;\nvarying vec2 vChecker;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvChecker = ( modelMatrix * vec4( transformed, 1.0 ) ).xy * uInvSquare;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
uniform vec3 uCheckerA;
uniform vec3 uCheckerB;
varying vec2 vChecker;
float filteredChecker( vec2 p ) {
  vec2 w = abs( dFdx( p ) ) + abs( dFdy( p ) ) + 1e-3;
  vec2 i = 2.0 * ( abs( fract( ( p - 0.5 * w ) * 0.5 ) - 0.5 ) - abs( fract( ( p + 0.5 * w ) * 0.5 ) - 0.5 ) ) / w;
  return 0.5 - 0.5 * i.x * i.y;
}`)
      .replace('#include <map_fragment>', 'diffuseColor.rgb *= mix( uCheckerA, uCheckerB, filteredChecker( vChecker ) );');
  };
  material.customProgramCacheKey = () => 'duet-checker-floor';
  return material;
}

function primitiveGeometry(type, size) {
  const [a, b, c] = size;
  let g;
  switch (type) {
    case GEOM_SPHERE: return new THREE.SphereGeometry(a, 24, 16);
    case GEOM_CAPSULE: g = new THREE.CapsuleGeometry(a, 2 * b, 6, 16); g.rotateX(Math.PI / 2); return g;
    case GEOM_CYLINDER: g = new THREE.CylinderGeometry(a, a, 2 * b, 24); g.rotateX(Math.PI / 2); return g;
    case GEOM_ELLIPSOID: g = new THREE.SphereGeometry(1, 24, 16); g.scale(a, b, c); g.computeVertexNormals(); return g;
    case GEOM_BOX: return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    default: return null;
  }
}

// A geometry's triangles for picking, in its own (the geom's) frame: positions,
// indices, and a box [min xyz, max xyz] per run of PICK_CHUNK triangles plus
// one around them all. null for an empty geometry.
function pickData(geometry) {
  const attr = geometry.attributes.position;
  if (!attr || attr.count === 0 || attr.itemSize !== 3 || attr.isInterleavedBufferAttribute) return null;
  const p = attr.array;
  const index = geometry.index ? geometry.index.array : Uint32Array.from({ length: attr.count }, (_, i) => i);
  const n = Math.floor(index.length / 3);
  if (n === 0) return null;
  const chunks = Math.ceil(n / PICK_CHUNK);
  const boxes = new Float32Array(6 * chunks);
  const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  let r2 = 0; // squared distance of the farthest vertex from the frame's origin
  for (let v = 0; v < p.length; v += 3) r2 = Math.max(r2, p[v] * p[v] + p[v + 1] * p[v + 1] + p[v + 2] * p[v + 2]);
  for (let c = 0; c < chunks; c++) {
    const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    const end = 3 * Math.min(n, (c + 1) * PICK_CHUNK);
    for (let k = 3 * c * PICK_CHUNK; k < end; k++) {
      const v = 3 * index[k];
      for (let j = 0; j < 3; j++) {
        const x = p[v + j];
        if (x < b[j]) b[j] = x;
        if (x > b[j + 3]) b[j + 3] = x;
      }
    }
    for (let j = 0; j < 3; j++) {
      boxes[6 * c + j] = b[j];
      boxes[6 * c + j + 3] = b[j + 3];
      box[j] = Math.min(box[j], b[j]);
      box[j + 3] = Math.max(box[j + 3], b[j + 3]);
    }
  }
  return { p, index, n, boxes, box, rbound: Math.sqrt(r2) };
}

// Entry distance of the ray o + t d (id = 1/d per axis) into box b at offset o6,
// or Infinity when it misses it or enters beyond `far`.
function rayBox(ox, oy, oz, ix, iy, iz, b, o6, far) {
  let t0 = (b[o6] - ox) * ix, t1 = (b[o6 + 3] - ox) * ix;
  let near = Math.min(t0, t1), exit = Math.max(t0, t1);
  t0 = (b[o6 + 1] - oy) * iy; t1 = (b[o6 + 4] - oy) * iy;
  near = Math.max(near, Math.min(t0, t1)); exit = Math.min(exit, Math.max(t0, t1));
  t0 = (b[o6 + 2] - oz) * iz; t1 = (b[o6 + 5] - oz) * iz;
  near = Math.max(near, Math.min(t0, t1)); exit = Math.min(exit, Math.max(t0, t1));
  return exit >= Math.max(near, 0) && near < far ? near : Infinity;
}

const safeInverse = (d) => 1 / (Math.abs(d) > 1e-12 ? d : (d < 0 ? -1e-12 : 1e-12));

// Nearest hit (t < far) of the ray o + t d on the triangles of pickData(),
// both faces (Moller-Trumbore), all in the geom frame; far when none.
function rayMesh(data, ox, oy, oz, dx, dy, dz, ix, iy, iz, far) {
  const { p, index, n, boxes } = data;
  let best = far;
  const chunks = boxes.length / 6;
  for (let c = 0; c < chunks; c++) {
    if (rayBox(ox, oy, oz, ix, iy, iz, boxes, 6 * c, best) === Infinity) continue;
    const end = 3 * Math.min(n, (c + 1) * PICK_CHUNK);
    for (let k = 3 * c * PICK_CHUNK; k < end; k += 3) {
      const a = 3 * index[k], b = 3 * index[k + 1], q = 3 * index[k + 2];
      const ax = p[a], ay = p[a + 1], az = p[a + 2];
      const e1x = p[b] - ax, e1y = p[b + 1] - ay, e1z = p[b + 2] - az;
      const e2x = p[q] - ax, e2y = p[q + 1] - ay, e2z = p[q + 2] - az;
      const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
      const det = e1x * hx + e1y * hy + e1z * hz;
      if (det === 0) continue;
      const inv = 1 / det;
      const sx = ox - ax, sy = oy - ay, sz = oz - az;
      const u = (sx * hx + sy * hy + sz * hz) * inv;
      if (u < 0 || u > 1) continue;
      const rx = sy * e1z - sz * e1y, ry = sz * e1x - sx * e1z, rz = sx * e1y - sy * e1x;
      const v = (dx * rx + dy * ry + dz * rz) * inv;
      if (v < 0 || u + v > 1) continue;
      const t = (e2x * rx + e2y * ry + e2z * rz) * inv;
      if (t > 1e-6 && t < best) best = t;
    }
  }
  return best;
}

// Entry distance (t > 0) of the ray o + t d (|d| = 1) into a sphere, or Infinity.
function raySphere(ox, oy, oz, dx, dy, dz, cx, cy, cz, r) {
  const px = ox - cx, py = oy - cy, pz = oz - cz;
  const b = dx * px + dy * py + dz * pz;
  const h = b * b - (px * px + py * py + pz * pz - r * r);
  if (h <= 0) return Infinity;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : Infinity;
}

// Entry distance (t > 0) of the ray o + t d (|d| = 1) into the capsule from a
// to b of radius r (Quilez, "ray-capsule intersection"), or Infinity.
function rayCapsule(ox, oy, oz, dx, dy, dz, ax, ay, az, bx, by, bz, r) {
  const bax = bx - ax, bay = by - ay, baz = bz - az;
  const oax = ox - ax, oay = oy - ay, oaz = oz - az;
  const baba = bax * bax + bay * bay + baz * baz;
  const bard = bax * dx + bay * dy + baz * dz;
  const baoa = bax * oax + bay * oay + baz * oaz;
  const a = baba - bard * bard;
  if (a > 1e-12) {
    const b = baba * (dx * oax + dy * oay + dz * oaz) - baoa * bard;
    const c = baba * (oax * oax + oay * oay + oaz * oaz) - baoa * baoa - r * r * baba;
    const h = b * b - a * c;
    if (h < 0) return Infinity; // misses the infinite cylinder, so the capsule too
    const t = (-b - Math.sqrt(h)) / a;
    const y = baoa + t * bard;
    if (y > 0 && y < baba) return t > 0 ? t : Infinity; // enters through the side
  }
  // Through an end cap (also for a ray along the axis or a zero-length segment).
  return Math.min(raySphere(ox, oy, oz, dx, dy, dz, ax, ay, az, r), raySphere(ox, oy, oz, dx, dy, dz, bx, by, bz, r));
}

// The pull overlay: an SVG over the canvas holding a haloed line and two dots.
function pullOverlay(host, colors) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('class', 'pull-overlay');
  svg.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;overflow:visible;pointer-events:none;display:none';
  const make = (tag, attrs) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    svg.appendChild(node);
    return node;
  };
  const halo = make('line', { stroke: colors.halo, 'stroke-width': 6, 'stroke-linecap': 'round', 'stroke-opacity': 0.85 });
  const line = make('line', { stroke: colors.line, 'stroke-width': 2.5, 'stroke-linecap': 'round' });
  const target = make('circle', { r: 4.5, fill: colors.halo, stroke: colors.line, 'stroke-width': 2.25 });
  const grab = make('circle', { r: 5.5, fill: colors.grab, stroke: colors.halo, 'stroke-width': 2 });
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
  host.appendChild(svg);
  return { svg, halo, line, target, grab, shown: false, xy: new Float64Array(4).fill(NaN), width: '' };
}

export class DemoRenderer {
  constructor(host, visual, visualContainer, options = {}) {
    if (!visual || !visualContainer) throw new Error('DemoRenderer needs demo.visual and the visual.bin.gz container');
    this.host = host;
    this.visual = visual;
    this.options = { ...DEFAULTS, ...options };
    const opts = this.options;
    const arrays = visualContainer.arrays ?? visualContainer;
    const meta = visualContainer.meta;
    if (meta && meta.format && meta.format !== visual.format) {
      throw new Error(`visual.bin.gz is ${meta.format}, demo.json expects ${visual.format}: re-export the visual section`);
    }
    const geomIds = Array.from(arrays.geom_id);
    if (geomIds.length !== visual.geom_ids.length || geomIds.some((g, i) => g !== visual.geom_ids[i])) {
      throw new Error('visual.bin.gz and demo.json list different geoms: re-export the visual section');
    }

    // Renderer.
    const renderer = this.renderer = new THREE.WebGLRenderer({
      antialias: opts.antialias,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: opts.preserveDrawingBuffer,
    });
    renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, opts.maxPixelRatio));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = opts.exposure;
    renderer.shadowMap.enabled = opts.shadows;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    const canvas = renderer.domElement;
    canvas.style.display = 'block';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    host.appendChild(canvas);

    // Scene, fog, camera.
    const scene = this.scene = new THREE.Scene();
    const background = srgb(visual.background_rgb || [0.955, 0.955, 0.95]);
    scene.background = background;
    scene.fog = new THREE.Fog(background, opts.fog[0], opts.fog[1]);
    const cam = this.cameraConfig = { fovy: 45, distance: 3.0, elevation: -12, azimuth_rel: -65, target_offset: [0, 0, -0.05], ...(visual.camera || {}) };
    const camera = this.camera = new THREE.PerspectiveCamera(cam.fovy, 1, 0.02, 400);
    camera.up.set(0, 0, 1);
    camera.position.set(3, -3, 1.5);
    scene.add(camera);

    this.controls = null;
    if (opts.controls) {
      const controls = this.controls = new OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      controls.minDistance = 0.6;
      controls.maxDistance = 30;
      controls.maxPolarAngle = Math.PI / 2 + 0.25;  // a little below the pelvis, never under the floor
      controls.enablePan = false;
      controls.target.set(0, 0, 0.9);
    }
    this._target = new THREE.Vector3(0, 0, 0.9);   // controls' target when controls are off

    // Lights: sky/ground fill, key light with shadows that follows the walker,
    // and a dim headlight on the camera (MuJoCo's headlight).
    const hemi = new THREE.HemisphereLight(0xffffff, srgb([0.62, 0.60, 0.57]), opts.hemiIntensity);
    hemi.position.set(0, 0, 1);
    scene.add(hemi);
    const key = this.keyLight = new THREE.DirectionalLight(0xffffff, opts.keyIntensity);
    const dir = new THREE.Vector3(...(visual.light_dir || [0.25, 0.35, -1])).normalize();
    this._lightOffset = dir.clone().multiplyScalar(-12);
    key.castShadow = opts.shadows;
    const s = key.shadow;
    s.mapSize.set(opts.shadowMapSize, opts.shadowMapSize);
    s.camera.left = -opts.shadowExtent; s.camera.right = opts.shadowExtent;
    s.camera.top = opts.shadowExtent; s.camera.bottom = -opts.shadowExtent;
    s.camera.near = 1; s.camera.far = 30;
    s.camera.updateProjectionMatrix();
    s.bias = -0.0003;
    s.normalBias = 0.012;
    s.radius = 2.5;
    scene.add(key, key.target);
    const fill = new THREE.DirectionalLight(0xffffff, opts.fillIntensity);
    fill.position.set(0, 0, 0);
    fill.target.position.set(0, 0, -1);
    camera.add(fill, fill.target);

    // Floor: a world-space checker on a plane that recentres under the walker.
    const floorCfg = visual.floor || {};
    this._square = floorCfg.square || 0.125;
    const floor = this.floor = new THREE.Mesh(
      new THREE.PlaneGeometry(opts.floorSize, opts.floorSize),
      checkerFloorMaterial(floorCfg.checker_rgb || [[0.88, 0.88, 0.88], [0.82, 0.82, 0.82]], this._square),
    );
    floor.receiveShadow = true;
    scene.add(floor);

    // Robot: one mesh per drawn geom, posed from the frame's geom_xpos/xmat.
    const materials = (visual.materials || []).map((m) => new THREE.MeshStandardMaterial({
      name: m.name,
      color: srgb(m.rgb),
      roughness: m.roughness ?? 0.6,
      metalness: m.metalness ?? 0,
      transparent: (m.opacity ?? 1) < 1,
      opacity: m.opacity ?? 1,
      depthWrite: (m.opacity ?? 1) >= 1,
    }));
    const meshGeoms = [];
    const nMesh = arrays.mesh_id.length;
    for (let i = 0; i < nMesh; i++) {
      const vs = arrays.mesh_vertex_start[i], vc = arrays.mesh_vertex_count[i];
      const is = arrays.mesh_index_start[i], ic = arrays.mesh_index_count[i];
      const q = arrays.positions.subarray(3 * vs, 3 * (vs + vc));
      const off = arrays.mesh_offset.subarray(3 * i, 3 * i + 3);
      const sc = arrays.mesh_scale.subarray(3 * i, 3 * i + 3);
      const pos = new Float32Array(3 * vc);
      for (let k = 0; k < vc; k++) {
        pos[3 * k] = q[3 * k] * sc[0] + off[0];
        pos[3 * k + 1] = q[3 * k + 1] * sc[1] + off[1];
        pos[3 * k + 2] = q[3 * k + 2] * sc[2] + off[2];
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('normal', new THREE.BufferAttribute(arrays.normals.slice(3 * vs, 3 * (vs + vc)), 3, true));
      g.setIndex(new THREE.BufferAttribute(arrays.indices.slice(is, is + ic), 1));
      g.computeBoundingSphere();
      meshGeoms.push(g);
    }
    const robot = this.robot = new THREE.Group();
    robot.visible = false;
    this.geomMeshes = [];
    for (let i = 0; i < geomIds.length; i++) {
      const m = arrays.geom_mesh[i];
      const geometry = m >= 0 ? meshGeoms[m]
        : primitiveGeometry(arrays.geom_type[i], arrays.geom_size.subarray(3 * i, 3 * i + 3));
      const material = materials[arrays.geom_material[i]] || materials[0];
      const mesh = new THREE.Mesh(geometry || new THREE.BufferGeometry(), material);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = true;
      mesh.receiveShadow = !material.transparent;
      robot.add(mesh);
      this.geomMeshes.push(mesh);
    }
    scene.add(robot);
    this.geomCount = geomIds.length;
    this._materials = materials;

    // Muscles: one InstancedMesh of stretched unit capsules.
    const mus = visual.muscles || {};
    const capacity = Math.max(1, mus.max_segments || 1024);
    const capsule = new THREE.CapsuleGeometry(1, 2 * CAPSULE_EPS, 2, 8);
    const half = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    half.setUsage(THREE.DynamicDrawUsage);
    capsule.setAttribute('aHalf', half);
    const muscleMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, metalness: 0, transparent: (mus.opacity ?? 1) < 1, opacity: mus.opacity ?? 1 });
    muscleMaterial.onBeforeCompile = stretchCapsule;
    muscleMaterial.customProgramCacheKey = () => 'duet-capsule';
    const depth = new THREE.MeshDepthMaterial();
    depth.onBeforeCompile = stretchCapsule;
    depth.customProgramCacheKey = () => 'duet-capsule-depth';
    const muscles = this.muscles = new THREE.InstancedMesh(capsule, muscleMaterial, capacity);
    muscles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    muscles.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(3 * capacity), 3);
    muscles.instanceColor.setUsage(THREE.DynamicDrawUsage);
    muscles.customDepthMaterial = depth;
    muscles.count = 0;
    muscles.frustumCulled = false;
    muscles.castShadow = true;
    muscles.receiveShadow = true;
    robot.add(muscles);
    this._depthMaterial = depth;
    this._capacity = capacity;
    this._halfAttr = half;
    this._lut = this._buildLut(mus.colormap || [[0, [0.98, 0.62, 0.58]], [0.3, [0.96, 0.33, 0.31]], [1, [0.8, 0.03, 0.04]]]);

    // Picking and pulling: each drawn geom's body and triangles (shared per
    // geometry), the muscle capsules as drawn, the pull overlay, the glow.
    this._geomIds = geomIds;
    this._geomBody = arrays.geom_body ? Array.from(arrays.geom_body) : geomIds.map(() => -1);
    this._geomIndex = new Map(geomIds.map((g, i) => [g, i]));
    const picks = new Map();
    this._pickData = this.geomMeshes.map((mesh) => {
      if (!picks.has(mesh.geometry)) picks.set(mesh.geometry, pickData(mesh.geometry));
      return picks.get(mesh.geometry);
    });
    this._segs = new Float32Array(7 * capacity); // per drawn muscle segment: p0, p1, radius
    this._segCount = 0;
    this._raycaster = new THREE.Raycaster();
    this._ndc = new THREE.Vector2();
    this._overlay = pullOverlay(host, opts.pull);
    this._pull = null;        // {index, local, target, force, forceRef} while a body part is pulled
    this._pullForce = 0;      // smoothed force (N) for the line width
    this._glow = { key: null, meshes: [], materials: new Map() };

    // State.
    this._frame = null;       // untimed frame not yet applied
    this._queue = [];         // timed frames of one epoch, oldest first
    this._play = null;        // playback time (simulated s) of the queue; null: jump to the next target
    this._shown = { a: null, b: null, alpha: -1 }; // the pose applied last
    this._rate = 1;           // simulated s per wall s at which frames arrive
    this._arrivals = { wall: [], time: [] };
    this._clamped = 0;        // times the clock ran into the newest frame (none newer had arrived)
    this._hasFrame = false;   // a frame has been applied (the walker is drawn)
    this._mode = 'follow';
    this._hold = false;       // holdCamera(): no following
    this._snap = true;
    this._root = new THREE.Vector3(0, 0, 0.9);
    this._heading = 0;
    this._smooth = new THREE.Vector3(0, 0, 0.9);
    this._lastTime = 0;
    this._tmp = new THREE.Vector3();
    this._tmpProject = new THREE.Vector3();
    this._qa = new Float64Array(4);
    this._qb = new Float64Array(4);
    this._redraw = true;      // something changed that the camera state does not show
    this._drawn = new Float64Array(10).fill(Infinity); // camera position, quaternion and light at the last drawing
    this._warned = false;

    this.resize();
    this._observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => this.resize()) : null;
    if (this._observer) this._observer.observe(host);
  }

  // A 256-entry activation -> linear RGB table from sRGB colormap stops.
  _buildLut(stops) {
    const lut = new Float32Array(3 * LUT_SIZE);
    const c = new THREE.Color();
    for (let i = 0; i < LUT_SIZE; i++) {
      const a = i / (LUT_SIZE - 1);
      let k = 0;
      while (k < stops.length - 2 && a > stops[k + 1][0]) k++;
      const [a0, c0] = stops[k], [a1, c1] = stops[Math.min(k + 1, stops.length - 1)];
      const t = a1 > a0 ? Math.min(1, Math.max(0, (a - a0) / (a1 - a0))) : 0;
      c.setRGB(c0[0] + t * (c1[0] - c0[0]), c0[1] + t * (c1[1] - c0[1]), c0[2] + t * (c1[2] - c0[2]), THREE.SRGBColorSpace);
      lut[3 * i] = c.r; lut[3 * i + 1] = c.g; lut[3 * i + 2] = c.b;
    }
    return lut;
  }

  // A frame to show (see the header): timed frames are queued for playback,
  // an untimed one replaces everything at the next update.
  setFrame(frame) {
    if (!frame || !frame.geoms || frame.geoms.length < 12 * this.geomCount) {
      throw new Error('DemoRenderer.setFrame: the frame does not match demo.visual.geom_ids');
    }
    const q = this._queue;
    if (!Number.isFinite(frame.time)) {
      this._releaseQueue(0);
      if (this._frame) this._release(this._frame);
      this._frame = frame;
      return;
    }
    if (this._frame) {
      this._release(this._frame);
      this._frame = null;
    }
    const last = q[q.length - 1];
    if (last && frame.epoch !== last.epoch) {
      this._releaseQueue(0); // a reset: start over at this frame
    } else if (last && frame.time <= last.time) {
      // The same pose again (or an older one): keep a single frame per time.
      if (frame.time < last.time) { this._release(frame); return; }
      q.pop();
      this._release(last);
    }
    if (!q.length) this._play = null;
    q.push(frame);
    this._noteArrival(frame.time);
    while (q.length > MAX_QUEUE) this._release(q.shift());
  }

  setCameraMode(mode) {
    if (!['follow', 'chase', 'free'].includes(mode)) throw new Error(`Unknown camera mode ${mode}`);
    // Leaving 'free': glide from wherever the user panned back to the walker.
    if (this._mode === 'free' && mode !== 'free') this._smooth.copy(this._targetVector());
    this._mode = mode;
    if (this.controls) this.controls.enablePan = mode === 'free';
  }

  get cameraMode() { return this._mode; }

  // Distance from the camera to its target (m).
  get cameraDistance() { return this.camera.position.distanceTo(this._targetVector()); }

  // Azimuth of the camera seen from its target (deg, CCW from +x, as setView).
  get cameraAzimuth() {
    const o = this._tmpProject.copy(this.camera.position).sub(this._targetVector());
    return THREE.MathUtils.radToDeg(Math.atan2(o.y, o.x));
  }

  // Put the camera on the default view of the walker at the next update: in
  // 'chase' behind the walker at chaseAzimuth, otherwise at demo.visual.camera.
  resetView() { this._snap = true; }

  // Draw again at the next tick() even if nothing seems to have changed
  // (e.g. after the WebGL context came back).
  invalidate() { this._redraw = true; }

  // Absolute view for tests and scripted shots: azimuth (deg, CCW from +x, of the
  // camera as seen from the target), elevation (deg, < 0 looks down), distance (m).
  setView({ azimuth, elevation, distance, target } = {}) {
    const t = target ? new THREE.Vector3(...target) : this._targetVector().clone();
    const offset = this.camera.position.clone().sub(this._targetVector());
    const r = distance ?? offset.length();
    const az = THREE.MathUtils.degToRad(azimuth ?? THREE.MathUtils.radToDeg(Math.atan2(offset.y, offset.x)));
    const el = THREE.MathUtils.degToRad(-(elevation ?? -THREE.MathUtils.radToDeg(Math.asin(offset.z / Math.max(r, 1e-9)))));
    this._targetVector().copy(t);
    this.camera.position.set(t.x + r * Math.cos(el) * Math.cos(az), t.y + r * Math.cos(el) * Math.sin(az), t.z + r * Math.sin(el));
    this.camera.lookAt(t);
    this._snap = false;
    if (this.controls) this.controls.update();
  }

  setMusclesVisible(on) {
    this.muscles.visible = !!on;
    this._redraw = true;
  }

  resize() {
    const w = Math.max(1, this.host.clientWidth || globalThis.innerWidth || 1);
    const h = Math.max(1, this.host.clientHeight || globalThis.innerHeight || 1);
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, this.options.maxPixelRatio));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._redraw = true;
  }

  // Pose the walker (the latest untimed frame, or the queue at the playback
  // time) and move the camera, light and floor. dt (s) is the time since the
  // last update; measured from the clock when omitted.
  update(dt) {
    const now = performance.now();
    if (typeof dt !== 'number') dt = this._lastTime ? (now - this._lastTime) / 1000 : 0;
    this._lastTime = now;
    dt = Math.max(0, dt);
    if (this._frame) {
      const f = this._frame;
      this._frame = null;
      this._applyPose(f, null, 0);
      this._shown.a = null; // nothing queued is on screen
      this._release(f);
    } else if (this._queue.length) {
      this._advance(dt);
    }
    this._updateCamera(Math.min(0.1, dt));
    if (this._pull) {
      this._pullForce += (this._pull.force - this._pullForce) * (1 - Math.exp(-dt / PULL_FORCE_TAU));
      this._updatePull();
    }
  }

  // update(dt), then draw if anything visible changed since the last drawing.
  // Returns whether it drew. For requestAnimationFrame loops.
  tick(dt) {
    this.update(dt);
    return this.drawIfChanged();
  }

  // Draw if anything visible changed since the last drawing (the second half
  // of tick(), for a caller with work between the update and the draw).
  drawIfChanged() {
    if (!this._needsDraw()) return false;
    this._draw();
    return true;
  }

  // ------------------------------------------------------------- pulling
  // The robot part under the cursor (client coordinates) as drawn now:
  // {geom (model geom id), body (its body id), local ([x, y, z] in the geom
  // frame), point ([x, y, z] world)}, or null for the background, the floor or
  // an empty canvas. See the header for how muscles count. slop (CSS px): when
  // the cursor itself misses, the nearest hit on rings of rays up to this far
  // from it (a fingertip, a thin part), the inner ring first.
  pick(clientX, clientY, slop = 0) {
    const hit = this._pickAt(clientX, clientY);
    if (hit || !(slop > 0)) return hit;
    for (const radius of [slop / 2, slop]) {
      let best = null;
      for (let k = 0; k < 12; k++) {
        const a = (k * Math.PI) / 6;
        const h = this._pickAt(clientX + radius * Math.cos(a), clientY + radius * Math.sin(a));
        if (h && (!best || h.distance < best.distance)) best = h;
      }
      if (best) return best;
    }
    return null;
  }

  _pickAt(clientX, clientY) {
    if (!this._hasFrame || !this.robot.visible) return null;
    const ray = this._cursorRay(clientX, clientY, true);
    if (!ray) return null;
    const bone = this._pickBones(ray);
    const muscle = this.muscles.visible ? this._pickMuscles(ray) : Infinity;
    let index = -1, t = Infinity;
    if (bone.t < Infinity && bone.t <= muscle + PICK_MUSCLE_DEPTH) {
      index = bone.index;
      t = bone.t;
    } else if (muscle < Infinity) {
      t = muscle;
      index = this._nearestGeom(ray.origin.x + t * ray.direction.x, ray.origin.y + t * ray.direction.y, ray.origin.z + t * ray.direction.z);
    }
    if (index < 0) return null;
    const o = ray.origin, d = ray.direction;
    const point = [o.x + t * d.x, o.y + t * d.y, o.z + t * d.z];
    return { geom: this._geomIds[index], body: this._geomBody[index], local: this._toLocal(index, point), point, distance: t };
  }

  // Where a pull on grab ({geom, local}) points for the cursor at (clientX,
  // clientY): the cursor ray met with the plane through the grabbed point as
  // drawn now, square to the view direction. Call it every animation frame
  // (after update()) while dragging: the target then stays under the cursor
  // while the camera moves. [x, y, z] world, or null for an unknown geom.
  dragTarget(clientX, clientY, grab) {
    const index = grab ? this._geomIndex.get(grab.geom) : undefined;
    if (index === undefined) return null;
    const p = this._toWorld(index, grab.local);
    const ray = this._cursorRay(clientX, clientY, false);
    if (!ray) return p;
    const n = this.camera.getWorldDirection(this._tmp);
    const o = ray.origin, d = ray.direction;
    const denom = n.x * d.x + n.y * d.y + n.z * d.z;
    const t = denom > 1e-6 ? (n.x * (p[0] - o.x) + n.y * (p[1] - o.y) + n.z * (p[2] - o.z)) / denom : NaN;
    if (!(t > 0)) return p; // the point is behind the camera: no pull
    return [o.x + t * d.x, o.y + t * d.y, o.z + t * d.z];
  }

  // Show a pull: a line from the grabbed point (geom + local, moving with the
  // body) to target (world), wider the nearer force (N, optional) is to
  // maxForce (N, optional), and the grabbed body aglow. null hides it.
  setPerturb(pull) {
    const index = pull ? this._geomIndex.get(pull.geom) : undefined;
    if (index === undefined) {
      this._pull = null;
      this._setGlow(null);
      this._showOverlay(false);
      return;
    }
    const force = Number.isFinite(pull.force) ? Math.max(0, pull.force) : 0;
    const forceRef = Number.isFinite(pull.maxForce) && pull.maxForce > 0 ? pull.maxForce : this.options.pull.forceRef;
    if (!this._pull || this._pull.index !== index) this._pullForce = force;
    this._pull = { index, local: Array.from(pull.local), target: Array.from(pull.target), force, forceRef };
    const body = this._geomBody[index];
    this._setGlow(body >= 0 ? `b${body}` : `g${index}`);
    this._updatePull();
  }

  // OrbitControls on or off (off while a body part is dragged).
  setControlsEnabled(on) {
    if (this.controls) this.controls.enabled = !!on;
  }

  // Hold the camera where it is (true) or let it follow the walker again
  // (false: it glides back, as when leaving 'free'). On during a drag: a
  // camera that follows the pulled walker would carry the target (under the
  // cursor) along with it, and a pull held still would never let up.
  holdCamera(on) {
    on = !!on;
    if (on === this._hold) return;
    this._hold = on;
    if (!on && this._mode !== 'free') this._smooth.copy(this._targetVector());
  }

  // A world point ([x, y, z]) on screen as drawn now: [clientX, clientY], or
  // null behind the camera. For tests and overlays.
  projectToClient(point) {
    const canvas = this.renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    const xy = this._project(point[0], point[1], point[2], rect.width, rect.height);
    return xy ? [rect.left + xy[0], rect.top + xy[1]] : null;
  }

  // The world position of a point given in a drawn geom's frame (default its
  // origin), as drawn now; null for a geom that is not drawn.
  geomPoint(geom, local = [0, 0, 0]) {
    const index = this._geomIndex.get(geom);
    return index === undefined ? null : this._toWorld(index, local);
  }

  // update(dt), then draw.
  render(dt) {
    this.update(dt);
    this._draw();
  }

  // Draw statistics from the last render (triangles, draw calls), and the
  // frame queue: frames held, playback time and lag behind the newest frame
  // (simulated s), arrival rate, and updates that ran out of frames.
  get stats() {
    const r = this.renderer.info.render;
    const q = this._queue, newest = q.length ? q[q.length - 1].time : null;
    return {
      triangles: r.triangles, calls: r.calls, segments: this.muscles.count,
      queued: q.length, play: this._play, lag: newest === null || this._play === null ? null : newest - this._play,
      rate: this._rate, clamped: this._clamped,
    };
  }

  _targetVector() { return this.controls ? this.controls.target : this._target; }

  _draw() {
    this.renderer.render(this.scene, this.camera);
    const c = this.camera.position, q = this.camera.quaternion, s = this._smooth, d = this._drawn;
    d[0] = c.x; d[1] = c.y; d[2] = c.z; d[3] = q.x; d[4] = q.y; d[5] = q.z; d[6] = q.w; d[7] = s.x; d[8] = s.y; d[9] = s.z;
    this._redraw = false;
  }

  // Whether the picture would differ from the last drawing: a new pose, a
  // resize, or the camera or the walker-following light moved visibly.
  _needsDraw() {
    if (this._redraw) return true;
    const c = this.camera.position, q = this.camera.quaternion, s = this._smooth, d = this._drawn;
    if (Math.abs(c.x - d[0]) > DRAW_EPS_POS || Math.abs(c.y - d[1]) > DRAW_EPS_POS || Math.abs(c.z - d[2]) > DRAW_EPS_POS) return true;
    if (Math.abs(s.x - d[7]) > DRAW_EPS_POS || Math.abs(s.y - d[8]) > DRAW_EPS_POS || Math.abs(s.z - d[9]) > DRAW_EPS_POS) return true;
    return 1 - Math.abs(q.x * d[3] + q.y * d[4] + q.z * d[5] + q.w * d[6]) > DRAW_EPS_ROT;
  }

  _release(frame) {
    if (frame && this.options.onRelease) this.options.onRelease(frame);
  }

  // Release the queued frames before index `keep` (all of them for 0).
  _releaseQueue(keep) {
    const q = this._queue;
    const n = keep > 0 ? keep : q.length;
    for (let i = 0; i < n; i++) this._release(q[i]);
    q.splice(0, n);
    if (!q.length) this._play = null;
  }

  // The rate at which simulated time arrives (the real-time factor), measured
  // over the last RATE_WINDOW s of arrivals; a gap (pause, stall) restarts it.
  _noteArrival(time) {
    const now = performance.now() / 1000;
    const { wall, time: sim } = this._arrivals;
    if (wall.length && (now - wall[wall.length - 1] > RATE_GAP || time < sim[sim.length - 1])) {
      wall.length = 0;
      sim.length = 0;
    }
    wall.push(now);
    sim.push(time);
    while (wall.length > 2 && now - wall[0] > RATE_WINDOW) { wall.shift(); sim.shift(); }
    const span = now - wall[0];
    if (span >= 0.3) this._rate = Math.min(2, Math.max(0.05, (time - sim[0]) / span));
  }

  // Move the playback time on by dt of wall time, release the frames it has
  // passed, and pose the walker between the two frames around it.
  _advance(dt) {
    const q = this._queue;
    const newest = q[q.length - 1].time;
    const target = newest - this.options.delay;
    let t = this._play;
    if (t === null || target - t > PLAY_MAX_LAG) {
      t = Math.max(q[0].time, target); // the first frame of an epoch, or far behind: jump
    } else {
      // Run at the arrival rate, then ease the result toward the target.
      const ahead = t + dt * this._rate;
      const next = Math.max(t, ahead + (target - ahead) * (1 - Math.exp(-dt / PLAY_TAU)));
      if (next > newest) {
        if (t < newest) this._clamped += 1; // ran out of frames (none newer had arrived)
        t = newest;
      } else {
        t = next;
      }
    }
    this._play = t;
    let i = 0;
    while (i + 1 < q.length && q[i + 1].time <= t) i += 1;
    if (i > 0) this._releaseQueue(i);
    const a = q[0], b = q.length > 1 ? q[1] : null;
    const alpha = b ? Math.min(1, Math.max(0, (t - a.time) / (b.time - a.time))) : 0;
    const shown = this._shown;
    if (shown.a === a && shown.b === b && shown.alpha === alpha) return; // the same pose
    this._applyPose(a, b, alpha);
    shown.a = a; shown.b = b; shown.alpha = alpha;
  }

  // Pose the walker at frame a, or between frames a and b (alpha in [0, 1]).
  _applyPose(a, b, alpha) {
    if (!b || alpha <= 0) { b = null; alpha = 0; }
    const ga = a.geoms, gb = b ? b.geoms : null;
    const qa = this._qa, qb = this._qb;
    for (let i = 0, o = 0; i < this.geomCount; i++, o += 12) {
      const mesh = this.geomMeshes[i];
      if (!gb) {
        mesh.matrix.set(
          ga[o + 3], ga[o + 4], ga[o + 5], ga[o],
          ga[o + 6], ga[o + 7], ga[o + 8], ga[o + 1],
          ga[o + 9], ga[o + 10], ga[o + 11], ga[o + 2],
          0, 0, 0, 1,
        );
      } else {
        quatFromRows(ga, o + 3, qa);
        quatFromRows(gb, o + 3, qb);
        const k = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3] < 0 ? -alpha : alpha;
        let x = qa[0] * (1 - alpha) + qb[0] * k, y = qa[1] * (1 - alpha) + qb[1] * k;
        let z = qa[2] * (1 - alpha) + qb[2] * k, w = qa[3] * (1 - alpha) + qb[3] * k;
        const norm = 1 / Math.hypot(x, y, z, w);
        x *= norm; y *= norm; z *= norm; w *= norm;
        mesh.matrix.set(
          1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), ga[o] + alpha * (gb[o] - ga[o]),
          2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), ga[o + 1] + alpha * (gb[o + 1] - ga[o + 1]),
          2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), ga[o + 2] + alpha * (gb[o + 2] - ga[o + 2]),
          0, 0, 0, 1,
        );
      }
      mesh.matrixWorldNeedsUpdate = true;
    }

    // Muscle instances: column-major matrices with a basis whose y axis runs
    // along the segment (Duff et al. 2017 orthonormal basis), uniform scale = radius.
    // Segments blend only between frames with the same segment rows; otherwise
    // the newer frame's are drawn.
    let sa = a.segments, sb = null, t = 0;
    let n = a.segmentCount | 0;
    if (b && (b.segmentCount | 0) === n) {
      sb = b.segments;
      t = alpha;
    } else if (b) {
      sa = b.segments;
      n = b.segmentCount | 0;
    }
    if (n > this._capacity) {
      if (!this._warned) console.warn(`DemoRenderer: ${n} muscle segments, capacity ${this._capacity}`);
      this._warned = true;
      n = this._capacity;
    }
    const m = this.muscles.instanceMatrix.array;
    const col = this.muscles.instanceColor.array, half = this._halfAttr.array, lut = this._lut;
    const segs = this._segs;
    this._segCount = n;
    for (let i = 0; i < n; i++) {
      const o = 8 * i;
      let x0 = sa[o], y0 = sa[o + 1], z0 = sa[o + 2], x1 = sa[o + 3], y1 = sa[o + 4], z1 = sa[o + 5];
      let r = sa[o + 6], level = sa[o + 7];
      if (sb) {
        const d0 = (sb[o] - x0) ** 2 + (sb[o + 1] - y0) ** 2 + (sb[o + 2] - z0) ** 2;
        const d1 = (sb[o + 3] - x1) ** 2 + (sb[o + 4] - y1) ** 2 + (sb[o + 5] - z1) ** 2;
        const u = d0 > SEGMENT_JUMP2 || d1 > SEGMENT_JUMP2 ? 1 : t;
        x0 += u * (sb[o] - x0); y0 += u * (sb[o + 1] - y0); z0 += u * (sb[o + 2] - z0);
        x1 += u * (sb[o + 3] - x1); y1 += u * (sb[o + 4] - y1); z1 += u * (sb[o + 5] - z1);
        r += u * (sb[o + 6] - r);
        level += u * (sb[o + 7] - level);
      }
      let dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
      const len = Math.hypot(dx, dy, dz);
      if (len > 1e-9) { dx /= len; dy /= len; dz /= len; } else { dx = 0; dy = 0; dz = 1; }
      const sign = dz >= 0 ? 1 : -1;
      const e = -1 / (sign + dz), f = dx * dy * e;
      // b1 = (1 + sign dx^2 e, sign f, -sign dx), b2 = (f, sign + dy^2 e, -dy); X = b2, Y = d, Z = b1.
      const s7 = 7 * i;
      segs[s7] = x0; segs[s7 + 1] = y0; segs[s7 + 2] = z0;
      segs[s7 + 3] = x1; segs[s7 + 4] = y1; segs[s7 + 5] = z1; segs[s7 + 6] = r;
      const j = 16 * i;
      m[j] = f * r; m[j + 1] = (sign + dy * dy * e) * r; m[j + 2] = -dy * r; m[j + 3] = 0;
      m[j + 4] = dx * r; m[j + 5] = dy * r; m[j + 6] = dz * r; m[j + 7] = 0;
      m[j + 8] = (1 + sign * dx * dx * e) * r; m[j + 9] = sign * f * r; m[j + 10] = -sign * dx * r; m[j + 11] = 0;
      m[j + 12] = x0 + 0.5 * len * dx; m[j + 13] = y0 + 0.5 * len * dy; m[j + 14] = z0 + 0.5 * len * dz; m[j + 15] = 1;
      half[i] = r > 0 ? 0.5 * len / r : 0;
      const k = 3 * Math.min(LUT_SIZE - 1, Math.max(0, Math.round(level * (LUT_SIZE - 1))));
      col[3 * i] = lut[k]; col[3 * i + 1] = lut[k + 1]; col[3 * i + 2] = lut[k + 2];
    }
    const mus = this.muscles;
    mus.count = n;
    mus.instanceMatrix.clearUpdateRanges(); mus.instanceMatrix.addUpdateRange(0, 16 * n); mus.instanceMatrix.needsUpdate = true;
    mus.instanceColor.clearUpdateRanges(); mus.instanceColor.addUpdateRange(0, 3 * n); mus.instanceColor.needsUpdate = true;
    this._halfAttr.clearUpdateRanges(); this._halfAttr.addUpdateRange(0, n); this._halfAttr.needsUpdate = true;

    // The root (camera target and chase heading), blended like the geoms.
    const ra = a.root, rb = b ? b.root : ra;
    const off = this.cameraConfig.target_offset || [0, 0, 0];
    this._root.set(
      ra[0] + alpha * (rb[0] - ra[0]) + off[0],
      ra[1] + alpha * (rb[1] - ra[1]) + off[1],
      ra[2] + alpha * (rb[2] - ra[2]) + off[2]);
    this._heading = ra[3] + alpha * wrapAngle(rb[3] - ra[3]);
    this.robot.visible = true;
    this._hasFrame = true;
    this._redraw = true;
  }

  _updateCamera(dt) {
    const target = this._targetVector();
    const camera = this.camera;
    if (this._snap && this._hasFrame) {
      // Default view: from the walker's facing, chaseAzimuth/chaseElevation in
      // 'chase' (no swing after a reset), else azimuth_rel/elevation; distance.
      const c = this.cameraConfig;
      const chase = this._mode === 'chase';
      const az = this._heading + THREE.MathUtils.degToRad(chase ? this.options.chaseAzimuth : c.azimuth_rel);
      const el = THREE.MathUtils.degToRad(-(chase ? this.options.chaseElevation : c.elevation));
      this._smooth.copy(this._root);
      target.copy(this._root);
      camera.position.set(
        target.x + c.distance * Math.cos(el) * Math.cos(az),
        target.y + c.distance * Math.cos(el) * Math.sin(az),
        target.z + c.distance * Math.sin(el));
      camera.lookAt(target);
      this._snap = false;
    } else if (this._hasFrame) {
      // The smoothed walker position drives the light in every mode and the
      // camera in 'follow' and 'chase' (unless held: holdCamera).
      this._smooth.lerp(this._root, 1 - Math.exp(-this.options.followRate * dt));
      if (this._mode !== 'free' && !this._hold) {
        const delta = this._tmp.copy(this._smooth).sub(target);
        camera.position.add(delta);
        target.add(delta);
      }
      if (this._mode === 'chase' && !this._hold) {
        // Swing behind the walker (azimuth) and up to the chase height
        // (elevation); the distance stays where the user zoomed it.
        const ox = camera.position.x - target.x, oy = camera.position.y - target.y;
        const oz = camera.position.z - target.z;
        const r = Math.hypot(ox, oy, oz) || this.cameraConfig.distance;
        const k = 1 - Math.exp(-this.options.chaseRate * dt);
        const azNow = Math.atan2(oy, ox);
        let dAz = this._heading + THREE.MathUtils.degToRad(this.options.chaseAzimuth) - azNow;
        dAz = Math.atan2(Math.sin(dAz), Math.cos(dAz));
        const az = azNow + k * dAz;
        const elNow = Math.asin(Math.min(1, Math.max(-1, oz / r)));
        const el = elNow + k * (THREE.MathUtils.degToRad(-this.options.chaseElevation) - elNow);
        camera.position.set(
          target.x + r * Math.cos(el) * Math.cos(az),
          target.y + r * Math.cos(el) * Math.sin(az),
          target.z + r * Math.sin(el));
      }
    }
    if (this.controls) this.controls.update(dt);
    else camera.lookAt(target);
    if (camera.position.z < 0.05) camera.position.z = 0.05;

    // Key light and its shadow box over the walker's feet; the floor under the
    // walker (under the camera target in 'free'), moved in whole periods.
    const p = this._hasFrame ? this._smooth : target;
    this.keyLight.target.position.set(p.x, p.y, 0);
    this.keyLight.position.set(p.x, p.y, 0).add(this._lightOffset);
    const q = this._mode === 'free' ? target : p;
    const period = 2 * this._square;
    this.floor.position.set(Math.round(q.x / period) * period, Math.round(q.y / period) * period, 0);
  }

  // ------------------------------------------------------------- picking internals
  // The camera ray through a client point (the raycaster's own ray: use it at
  // once). inside: null when the point is off the canvas.
  _cursorRay(clientX, clientY, inside) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return null;
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = 1 - ((clientY - rect.top) / rect.height) * 2;
    if (inside && (x < -1 || x > 1 || y < -1 || y > 1)) return null;
    this.camera.updateMatrixWorld();
    this._raycaster.setFromCamera(this._ndc.set(x, y), this.camera);
    return this._raycaster.ray;
  }

  // Nearest triangle hit over the drawn geoms: {index (into geom_ids), t}.
  _pickBones(ray) {
    const o = ray.origin, d = ray.direction;
    let best = Infinity, bestIndex = -1;
    for (let i = 0; i < this.geomCount; i++) {
      const data = this._pickData[i];
      if (!data) continue;
      // The ray in the geom frame: R^T (o - pos), R^T d (matrix = [R pos], column-major).
      const e = this.geomMeshes[i].matrix.elements;
      const px = o.x - e[12], py = o.y - e[13], pz = o.z - e[14];
      const ox = e[0] * px + e[1] * py + e[2] * pz, oy = e[4] * px + e[5] * py + e[6] * pz, oz = e[8] * px + e[9] * py + e[10] * pz;
      const dx = e[0] * d.x + e[1] * d.y + e[2] * d.z, dy = e[4] * d.x + e[5] * d.y + e[6] * d.z, dz = e[8] * d.x + e[9] * d.y + e[10] * d.z;
      const ix = safeInverse(dx), iy = safeInverse(dy), iz = safeInverse(dz);
      if (rayBox(ox, oy, oz, ix, iy, iz, data.box, 0, best) === Infinity) continue;
      const t = rayMesh(data, ox, oy, oz, dx, dy, dz, ix, iy, iz, best);
      if (t < best) {
        best = t;
        bestIndex = i;
      }
    }
    return { index: bestIndex, t: best };
  }

  // Nearest hit on the muscle capsules as drawn, each widened by PICK_MUSCLE_PX
  // at its depth: t along the ray, or Infinity.
  _pickMuscles(ray) {
    const n = this._segCount;
    if (!n) return Infinity;
    const o = ray.origin, d = ray.direction, segs = this._segs;
    const f = this.camera.getWorldDirection(this._tmp);
    const height = this.renderer.domElement.clientHeight || 1;
    const perMetre = (PICK_MUSCLE_PX * 2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2)) / height;
    let best = Infinity;
    for (let i = 0, s = 0; i < n; i++, s += 7) {
      const ax = segs[s], ay = segs[s + 1], az = segs[s + 2], bx = segs[s + 3], by = segs[s + 4], bz = segs[s + 5];
      const depth = (0.5 * (ax + bx) - o.x) * f.x + (0.5 * (ay + by) - o.y) * f.y + (0.5 * (az + bz) - o.z) * f.z;
      if (depth <= 0) continue;
      const t = rayCapsule(o.x, o.y, o.z, d.x, d.y, d.z, ax, ay, az, bx, by, bz, segs[s + 6] + perMetre * depth);
      if (t < best) best = t;
    }
    return best;
  }

  // The drawn geom whose box (in its own frame) is nearest to a world point,
  // within PICK_REACH, among those whose bounding sphere about their origin
  // (plus PICK_RBOUND_MARGIN) holds the point; -1 if none.
  _nearestGeom(x, y, z) {
    let best = PICK_REACH * PICK_REACH, bestIndex = -1;
    for (let i = 0; i < this.geomCount; i++) {
      const data = this._pickData[i];
      if (!data) continue;
      const [lx, ly, lz] = this._toLocal(i, [x, y, z]);
      if (Math.hypot(lx, ly, lz) > data.rbound + PICK_RBOUND_MARGIN) continue;
      const b = data.box;
      const dx = Math.max(b[0] - lx, 0, lx - b[3]), dy = Math.max(b[1] - ly, 0, ly - b[4]), dz = Math.max(b[2] - lz, 0, lz - b[5]);
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) {
        best = d2;
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  // World point -> the frame of drawn geom i as drawn now, and back.
  _toLocal(i, p) {
    const e = this.geomMeshes[i].matrix.elements;
    const x = p[0] - e[12], y = p[1] - e[13], z = p[2] - e[14];
    return [e[0] * x + e[1] * y + e[2] * z, e[4] * x + e[5] * y + e[6] * z, e[8] * x + e[9] * y + e[10] * z];
  }

  _toWorld(i, l) {
    const e = this.geomMeshes[i].matrix.elements;
    return [
      e[12] + e[0] * l[0] + e[4] * l[1] + e[8] * l[2],
      e[13] + e[1] * l[0] + e[5] * l[1] + e[9] * l[2],
      e[14] + e[2] * l[0] + e[6] * l[1] + e[10] * l[2],
    ];
  }

  // A world point in CSS pixels from the canvas's top-left corner, or null if
  // it is not in front of the camera.
  _project(x, y, z, width, height) {
    const cam = this.camera;
    cam.updateMatrixWorld();
    const v = this._tmpProject.set(x, y, z).applyMatrix4(cam.matrixWorldInverse);
    if (v.z > -cam.near) return null;
    v.applyMatrix4(cam.projectionMatrix);
    return [((v.x + 1) / 2) * width, ((1 - v.y) / 2) * height];
  }

  // Move the pull overlay to the grabbed point (as drawn now) and the target.
  _updatePull() {
    const pull = this._pull, ov = this._overlay;
    if (!pull) {
      this._showOverlay(false);
      return;
    }
    const canvas = this.renderer.domElement;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const g = this._toWorld(pull.index, pull.local);
    const a = this._project(g[0], g[1], g[2], w, h);
    const b = this._project(pull.target[0], pull.target[1], pull.target[2], w, h);
    if (!a || !b) {
      this._showOverlay(false);
      return;
    }
    const xy = ov.xy;
    if (!ov.shown || Math.abs(a[0] - xy[0]) > 0.05 || Math.abs(a[1] - xy[1]) > 0.05 || Math.abs(b[0] - xy[2]) > 0.05 || Math.abs(b[1] - xy[3]) > 0.05) {
      xy[0] = a[0]; xy[1] = a[1]; xy[2] = b[0]; xy[3] = b[1];
      const [x1, y1, x2, y2] = [a[0], a[1], b[0], b[1]].map((v) => v.toFixed(1));
      for (const node of [ov.halo, ov.line]) {
        node.setAttribute('x1', x1); node.setAttribute('y1', y1);
        node.setAttribute('x2', x2); node.setAttribute('y2', y2);
      }
      ov.grab.setAttribute('cx', x1); ov.grab.setAttribute('cy', y1);
      ov.target.setAttribute('cx', x2); ov.target.setAttribute('cy', y2);
    }
    const [w0, w1] = this.options.pull.width;
    const width = (w0 + (w1 - w0) * Math.min(1, this._pullForce / pull.forceRef)).toFixed(1);
    if (width !== ov.width) {
      ov.width = width;
      ov.line.setAttribute('stroke-width', width);
      ov.halo.setAttribute('stroke-width', (Number(width) + 3.5).toFixed(1));
    }
    this._showOverlay(true);
  }

  _showOverlay(on) {
    const ov = this._overlay;
    if (ov.shown === on) return;
    ov.shown = on;
    ov.svg.style.display = on ? 'block' : 'none';
  }

  // Light the meshes of one body ('b<body id>') or geom ('g<index>'), or none:
  // their materials are swapped for glowing copies, then back.
  _setGlow(key) {
    const glow = this._glow;
    if (glow.key === key) return;
    for (const [mesh, material] of glow.meshes) mesh.material = material;
    glow.meshes = [];
    glow.key = key;
    if (key !== null) {
      for (let i = 0; i < this.geomCount; i++) {
        const own = this._geomBody[i] >= 0 ? `b${this._geomBody[i]}` : `g${i}`;
        if (own !== key) continue;
        const mesh = this.geomMeshes[i];
        let lit = glow.materials.get(mesh.material);
        if (!lit) {
          lit = mesh.material.clone();
          lit.emissive.set(this.options.glow.color);
          lit.emissiveIntensity = this.options.glow.intensity;
          glow.materials.set(mesh.material, lit);
        }
        glow.meshes.push([mesh, mesh.material]);
        mesh.material = lit;
      }
    }
    this._redraw = true;
  }

  dispose() {
    if (this._observer) this._observer.disconnect();
    this._releaseQueue(0);
    if (this._frame) this._release(this._frame);
    this._frame = null;
    this._setGlow(null);
    for (const lit of this._glow.materials.values()) lit.dispose();
    this._overlay.svg.remove();
    if (this.controls) this.controls.dispose();
    const seen = new Set();
    this.scene.traverse((o) => {
      for (const res of [o.geometry, ...(Array.isArray(o.material) ? o.material : [o.material])]) {
        if (res && !seen.has(res)) {
          seen.add(res);
          if (res.map) res.map.dispose();
          res.dispose();
        }
      }
    });
    this._depthMaterial.dispose();
    this.keyLight.shadow.dispose?.();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
