three.js 0.187.0 (MIT, see LICENSE), copied unmodified from the npm package:
build/three.module.js, build/three.core.js, examples/jsm/controls/OrbitControls.js.
Pages map it with an import map: {"three": "<demo>/vendor/three/three.module.js",
"three/addons/": "<demo>/vendor/three/addons/"}. Copy further addons from the
same version into addons/ with their examples/jsm/ sub-path.
