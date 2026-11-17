// CPU backend: plain JavaScript, no SIMD, no threads. Works on the main thread,
// in a Web Worker and in Node.
//
// int8 rows have two kernels. Both sum the same exact products (int8 weight x
// float32 activation, exact in float64) in the same order, four columns at a
// time, so their outputs are bit-identical:
// - 'packed': each Int32 load carries 4 weights, sign-extended with shifts, and
//   one pass computes 8 rows so every activation load serves 8 rows.
// - 'int8': one Int8Array load per weight, 4 rows per pass.
// Human network (26.9M multiply-adds), ms per inference in a worker
// (Playwright builds on one x86-64 desktop, 2026-11-16):
//                      packed   int8
//   Chromium 156 (V8)    9.5    12.2
//   Firefox 157          10.3   14.8
//   WebKit 27.2 (JSC)    25.7    9.1
// createCpuBackend(net, {kernel: 'auto'}) times both on a fixed input when the
// network has int8 layers and keeps the faster ('auto' is the default; about
// 0.13 s at load); the backend reports the choice in .kernel and the timings
// in .race. Sums are JS numbers (float64); activations are stored as float32.

import { normalizeInto, selfTestInput } from './network.js';

export const CPU_KERNELS = Object.freeze(['packed', 'int8']);

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([0x04030201]).buffer)[0] === 0x01;
const now = () => (globalThis.performance ? performance.now() : Date.now());

// Int32 words of an int8 [rows, cols] matrix, each row padded with zero bytes
// to a whole number of words. A zero-copy view when the layout allows it.
export function packInt8Rows(weight, rows, cols) {
  const words = Math.ceil(cols / 4);
  if (LITTLE_ENDIAN && cols % 4 === 0 && weight.byteOffset % 4 === 0) {
    return { words, data: new Int32Array(weight.buffer, weight.byteOffset, rows * words) };
  }
  const data = new Int32Array(rows * words);
  if (LITTLE_ENDIAN) {
    const bytes = new Int8Array(data.buffer);
    for (let r = 0; r < rows; r++) bytes.set(weight.subarray(r * cols, (r + 1) * cols), r * words * 4);
  } else {
    for (let r = 0; r < rows; r++) {
      for (let k = 0; k < words; k++) {
        let word = 0;
        for (let b = 0; b < 4; b++) {
          const c = 4 * k + b;
          if (c < cols) word |= (weight[r * cols + c] & 0xff) << (8 * b);
        }
        data[r * words + k] = word;
      }
    }
  }
  return { words, data };
}

// The 4 signed bytes of w (byte b is weight 4k+b) dotted with x0..x3.
function dot4(w, x0, x1, x2, x3) {
  return ((w << 24) >> 24) * x0 + ((w << 16) >> 24) * x1 + ((w << 8) >> 24) * x2 + (w >> 24) * x3;
}

// 'packed': y[r] = scale[r] * (W[r] . x) + bias[r]; x needs 4 * words readable
// entries, zero past the input (a padded weight times a zero adds nothing).
function gemvInt8(W, words, rows, scale, bias, x, y) {
  let r = 0;
  for (; r + 8 <= rows; r += 8) {
    const o0 = r * words, o1 = o0 + words, o2 = o1 + words, o3 = o2 + words;
    const o4 = o3 + words, o5 = o4 + words, o6 = o5 + words, o7 = o6 + words;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (let k = 0, c = 0; k < words; k++, c += 4) {
      const x0 = x[c], x1 = x[c + 1], x2 = x[c + 2], x3 = x[c + 3];
      a0 += dot4(W[o0 + k], x0, x1, x2, x3);
      a1 += dot4(W[o1 + k], x0, x1, x2, x3);
      a2 += dot4(W[o2 + k], x0, x1, x2, x3);
      a3 += dot4(W[o3 + k], x0, x1, x2, x3);
      a4 += dot4(W[o4 + k], x0, x1, x2, x3);
      a5 += dot4(W[o5 + k], x0, x1, x2, x3);
      a6 += dot4(W[o6 + k], x0, x1, x2, x3);
      a7 += dot4(W[o7 + k], x0, x1, x2, x3);
    }
    y[r] = a0 * scale[r] + bias[r];
    y[r + 1] = a1 * scale[r + 1] + bias[r + 1];
    y[r + 2] = a2 * scale[r + 2] + bias[r + 2];
    y[r + 3] = a3 * scale[r + 3] + bias[r + 3];
    y[r + 4] = a4 * scale[r + 4] + bias[r + 4];
    y[r + 5] = a5 * scale[r + 5] + bias[r + 5];
    y[r + 6] = a6 * scale[r + 6] + bias[r + 6];
    y[r + 7] = a7 * scale[r + 7] + bias[r + 7];
  }
  for (; r < rows; r++) {
    const o = r * words;
    let a = 0;
    for (let k = 0, c = 0; k < words; k++, c += 4) a += dot4(W[o + k], x[c], x[c + 1], x[c + 2], x[c + 3]);
    y[r] = a * scale[r] + bias[r];
  }
}

// The last 1-3 columns of a row (cols not a multiple of 4), summed as dot4
// sums a zero-padded word: ((p0 + p1) + p2), then added to the row's sum.
function tail(W, o, c, cols, x) {
  let s = W[o + c] * x[c];
  if (c + 1 < cols) s += W[o + c + 1] * x[c + 1];
  if (c + 2 < cols) s += W[o + c + 2] * x[c + 2];
  return s;
}

// 'int8': the same result as gemvInt8 from the unpacked Int8Array weights
// (row-major [rows, cols]). Each 4-column group is summed as dot4 sums it,
// s = ((p0 + p1) + p2) + p3, then a += s. One multiply-add per statement:
// JavaScriptCore runs this form 2.5x faster than one long expression per
// group (0.78 against 1.95 ms for a 1024 x 2340 layer, WebKit 27.2).
function gemvI8(W, cols, rows, scale, bias, x, y) {
  const c4 = cols - (cols % 4);
  let r = 0;
  for (; r + 4 <= rows; r += 4) {
    const o0 = r * cols, o1 = o0 + cols, o2 = o1 + cols, o3 = o2 + cols;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (let c = 0; c < c4; c += 4) {
      let xc = x[c];
      let s0 = W[o0 + c] * xc, s1 = W[o1 + c] * xc, s2 = W[o2 + c] * xc, s3 = W[o3 + c] * xc;
      let k = c + 1;
      xc = x[k];
      s0 += W[o0 + k] * xc; s1 += W[o1 + k] * xc; s2 += W[o2 + k] * xc; s3 += W[o3 + k] * xc;
      k = c + 2;
      xc = x[k];
      s0 += W[o0 + k] * xc; s1 += W[o1 + k] * xc; s2 += W[o2 + k] * xc; s3 += W[o3 + k] * xc;
      k = c + 3;
      xc = x[k];
      s0 += W[o0 + k] * xc; s1 += W[o1 + k] * xc; s2 += W[o2 + k] * xc; s3 += W[o3 + k] * xc;
      a0 += s0; a1 += s1; a2 += s2; a3 += s3;
    }
    if (c4 < cols) {
      a0 += tail(W, o0, c4, cols, x);
      a1 += tail(W, o1, c4, cols, x);
      a2 += tail(W, o2, c4, cols, x);
      a3 += tail(W, o3, c4, cols, x);
    }
    y[r] = a0 * scale[r] + bias[r];
    y[r + 1] = a1 * scale[r + 1] + bias[r + 1];
    y[r + 2] = a2 * scale[r + 2] + bias[r + 2];
    y[r + 3] = a3 * scale[r + 3] + bias[r + 3];
  }
  for (; r < rows; r++) {
    const o = r * cols;
    let a = 0;
    for (let c = 0; c < c4; c += 4) {
      let s = W[o + c] * x[c];
      s += W[o + c + 1] * x[c + 1];
      s += W[o + c + 2] * x[c + 2];
      s += W[o + c + 3] * x[c + 3];
      a += s;
    }
    if (c4 < cols) a += tail(W, o, c4, cols, x);
    y[r] = a * scale[r] + bias[r];
  }
}

// y = W x + bias for a float32 [rows, cols] matrix, 4 rows per pass.
function gemvF32(W, rows, cols, bias, x, y) {
  let r = 0;
  for (; r + 4 <= rows; r += 4) {
    const o0 = r * cols, o1 = o0 + cols, o2 = o1 + cols, o3 = o2 + cols;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (let c = 0; c < cols; c++) {
      const xc = x[c];
      a0 += W[o0 + c] * xc;
      a1 += W[o1 + c] * xc;
      a2 += W[o2 + c] * xc;
      a3 += W[o3 + c] * xc;
    }
    y[r] = a0 + bias[r];
    y[r + 1] = a1 + bias[r + 1];
    y[r + 2] = a2 + bias[r + 2];
    y[r + 3] = a3 + bias[r + 3];
  }
  for (; r < rows; r++) {
    const o = r * cols;
    let a = 0;
    for (let c = 0; c < cols; c++) a += W[o + c] * x[c];
    y[r] = a + bias[r];
  }
}

// In place: optional layer norm (biased variance), then optional SiLU.
function post(h, n, norm, silu) {
  if (norm) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += h[i];
    const mean = sum / n;
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const d = h[i] - mean;
      sq += d * d;
    }
    const rstd = 1 / Math.sqrt(sq / n + norm.eps);
    const { gamma, beta } = norm;
    if (silu) {
      for (let i = 0; i < n; i++) {
        const v = (h[i] - mean) * rstd * gamma[i] + beta[i];
        h[i] = v / (1 + Math.exp(-v));
      }
    } else {
      for (let i = 0; i < n; i++) h[i] = (h[i] - mean) * rstd * gamma[i] + beta[i];
    }
  } else if (silu) {
    for (let i = 0; i < n; i++) {
      const v = h[i];
      h[i] = v / (1 + Math.exp(-v));
    }
  }
}

// ms per run of each kernel on input x. performance.now() is coarsened to
// about 1 ms in WebKit and Firefox, so each sample times a batch of runs lasting
// at least minMs. One untimed run per kernel first (JIT tiers), then `rounds`
// interleaved samples per kernel; the best sample counts.
function raceKernels(useKernel, runSync, x, { minMs = 20, rounds = 2, maxRuns = 200 } = {}) {
  const ms = {};
  for (const kernel of CPU_KERNELS) {
    useKernel(kernel);
    runSync(x);
    ms[kernel] = Infinity;
  }
  for (let round = 0; round < rounds; round++) {
    for (const kernel of CPU_KERNELS) {
      useKernel(kernel);
      const t0 = now();
      let runs = 0;
      let elapsed = 0;
      do {
        runSync(x);
        runs += 1;
        elapsed = now() - t0;
      } while (elapsed < minMs && runs < maxRuns);
      ms[kernel] = Math.min(ms[kernel], elapsed / runs);
    }
  }
  // A tie (or a clock that never moved) keeps the first kernel, 'packed'.
  const kernel = CPU_KERNELS.reduce((best, k) => (ms[k] < ms[best] ? k : best));
  return { kernel, ms };
}

// {backend: 'cpu', kernel, race, ms, runSync(x) -> Float32Array, run(x) -> Promise, dispose()}
//   kernel: the int8 kernel in use ('packed' | 'int8'), null without int8 layers
//   race: {packed, int8} ms per inference when 'auto' timed them, else null
//   ms: ms per inference of the kernel in use, when measured (else null)
// options.kernel: 'auto' (default) | 'packed' | 'int8'
export function createCpuBackend(net, options = {}) {
  const requested = options.kernel || 'auto';
  if (requested !== 'auto' && !CPU_KERNELS.includes(requested)) throw new Error(`Unknown CPU kernel ${requested}`);
  // Steps: normalize | linear | post (layernorm and/or silu, fused when adjacent).
  const steps = [];
  for (let i = 0; i < net.layers.length; i++) {
    const layer = net.layers[i];
    if (layer.kind === 'normalize') {
      steps.push({ kind: 'normalize', layer });
    } else if (layer.kind === 'linear') {
      const step = { kind: 'linear', rows: layer.out, cols: layer.in, bias: layer.bias };
      if (layer.quant === 'int8_row') {
        Object.assign(step, { int8: true, W8: layer.weight, scale: layer.scale, words: 0, W: null });
      } else {
        Object.assign(step, { int8: false, W: layer.weight });
      }
      steps.push(step);
    } else if (layer.kind === 'layernorm') {
      const silu = net.layers[i + 1] && net.layers[i + 1].kind === 'silu';
      steps.push({ kind: 'post', norm: layer, silu });
      if (silu) i++;
    } else if (layer.kind === 'silu') {
      steps.push({ kind: 'post', norm: null, silu: true });
    }
  }
  const int8Steps = steps.filter((s) => s.kind === 'linear' && s.int8);
  // Two ping-pong vectors, long enough for any padded row (zero padding stays zero:
  // a 'packed' step zeroes its input's tail before reading 4 * words entries).
  const length = Math.ceil(net.maxWidth / 4) * 4 + 4;
  const buffers = [new Float32Array(length), new Float32Array(length)];

  let packed = false;
  const useKernel = (kernel) => {
    packed = kernel === 'packed';
    if (!packed) return;
    for (const s of int8Steps) {
      if (s.W) continue;
      const { words, data } = packInt8Rows(s.W8, s.rows, s.cols);
      s.words = words;
      s.W = data;
    }
  };

  function runSync(input) {
    if (!input || input.length !== net.inputSize) {
      throw new Error(`${net.name}: expected ${net.inputSize} inputs, got ${input && input.length}`);
    }
    let x = buffers[0];
    let y = buffers[1];
    let size = net.inputSize;
    x.set(input);
    for (const step of steps) {
      if (step.kind === 'normalize') {
        normalizeInto(step.layer, x, x);
      } else if (step.kind === 'linear') {
        if (!step.int8) {
          gemvF32(step.W, step.rows, step.cols, step.bias, x, y);
        } else if (packed) {
          x.fill(0, size, step.words * 4);
          gemvInt8(step.W, step.words, step.rows, step.scale, step.bias, x, y);
        } else {
          gemvI8(step.W8, step.cols, step.rows, step.scale, step.bias, x, y);
        }
        [x, y] = [y, x];
        size = step.rows;
      } else {
        post(x, size, step.norm, step.silu);
      }
    }
    return x.slice(0, size);
  }

  let kernel = null;
  let race = null;
  let ms = null;
  if (int8Steps.length) {
    if (requested === 'auto') {
      ({ kernel, ms: race } = raceKernels(useKernel, runSync, selfTestInput(net), options.race));
      ms = race[kernel];
    } else {
      kernel = requested;
    }
    useKernel(kernel);
    if (kernel === 'int8') {
      // 'int8' reads the container's own bytes: drop the padded copies 'packed' made.
      for (const s of int8Steps) {
        s.W = null;
        s.words = 0;
      }
    }
  }

  return {
    backend: 'cpu',
    kernel,
    race,
    ms,
    runSync,
    run: async (input) => runSync(input),
    dispose() {},
  };
}
