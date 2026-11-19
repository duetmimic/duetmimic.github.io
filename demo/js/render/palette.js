// The scene's own colours (sRGB hex) that the views drawn over it are chosen
// against, in one module with no DOM and no three.js: render/scene.js draws
// with them (DEFAULTS), app.js gives the X-ray legend's swatches the arrow
// colours the renderer uses, and tools/test/sensors.test.mjs measures the
// sensor groups' colours against them. The orange OSL covers come from
// demo.json (visual.materials, category osl_cover). demo.css keeps copies as
// the tokens --xray-force, --xray-moment and --accent (CSS cannot import
// them; app.js overrides the first two at load).

// The X-ray view (render/scene.js setXray). Arrows: dataviz-validated as a
// pair on the scene's background (also against colour-vision deficiency), not
// orange, far from the slate ghosts. Ghosts: a surface's colour (and its
// silhouette's, rim).
export const XRAY_COLORS = Object.freeze({
  force: '#008300', // the force arrow, green
  moment: '#6250d6', // the moment arrow, violet
  bone: '#5d7394', // bone ghosts, slate
  boneRim: '#33435c',
  socket: '#45494f', // the socket's ghost, charcoal
  socketRim: '#1d2024',
  residual: '#4f6588', // the residual limb's bones, a stronger slate
  residualRim: '#22324a',
  halo: '#ffffff', // around each arrow part
  dot: '#1d1d1f', // the anchor
});

// The accent: a grabbed body's glow and the pull's dot on it (the pull line
// itself is the darker #c2410c).
export const ACCENT = '#f26a1b';
