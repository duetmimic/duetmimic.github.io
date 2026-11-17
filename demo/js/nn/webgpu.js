// WebGPU backend: WGSL compute kernels for an int8-row MLP at batch 1.
//
// - Weights: int8 packed 4 per u32 (the container's bytes, rows padded to 16),
//   read as vec4<u32> (16 weights) and sign-extended with shifts; no optional
//   WGSL feature (packed_4x8_integer_dot_product, subgroups, f16) is used.
// - GEMV: one 64-thread workgroup per 8 output rows; each thread walks the row
//   with stride 64 (one x load serves 8 rows), then a shared-memory tree
//   reduction (2 KB of workgroup storage). y = scale * (q . x) + bias.
// - Post: one 256-thread workgroup per vector: layer norm (two-pass mean and
//   biased variance) and/or SiLU, in place.
// - One command encoder per inference: input upload, every dispatch, one copy
//   of the output into a mappable buffer, one readback. Buffers and bind
//   groups are created once.
// - The observation normalizer runs on the CPU before the upload.
// Default device limits only (16 KB workgroup storage, 128 MB bindings).

import { normalizeInto } from './network.js';

// WebGPU enum values (fixed by the spec), so Node needs no injected globals.
const USAGE = { MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008, UNIFORM: 0x0040, STORAGE: 0x0080 };
const MAP_READ = 0x0001;

const WG = 64; // GEMV threads per workgroup
const ROWS = 8; // GEMV output rows per workgroup; divides PAD
const PAD = 16; // vector and weight-row padding: one vec4<u32> of int8
const POST_WG = 256;

const pad = (n) => Math.ceil(n / PAD) * PAD;

function gemvShader(rows) {
  const r = Array.from({ length: rows }, (_, i) => i);
  return /* wgsl */ `
struct Dims { rows: u32, quads: u32, rowsPad: u32, unused: u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> P: array<f32>;  // scale[rows], bias[rows]
@group(0) @binding(3) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

const WG = ${WG}u;
var<workgroup> partial: array<f32, ${WG * rows}>;

// The 4 signed bytes of w as floats (byte k is weight k).
fn bytes4(w: u32) -> vec4<f32> {
  return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w << 24u, w << 16u, w << 8u, w)) >> vec4<u32>(24u));
}

// 16 weights of one row (vec4<u32> q) dotted with 16 inputs.
fn quad(base: u32, q: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>) -> f32 {
  let w = W[base + q];
  return dot(bytes4(w.x), x0) + dot(bytes4(w.y), x1) + dot(bytes4(w.z), x2) + dot(bytes4(w.w), x3);
}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row0 = wid.x * ${rows}u;
  let last = dims.rows - 1u;
  ${r.map((i) => `let base${i} = min(row0 + ${i}u, last) * dims.quads;`).join('\n  ')}
  ${r.map((i) => `var acc${i} = 0.0;`).join('\n  ')}
  for (var q = t; q < dims.quads; q += WG) {
    let x0 = X[4u * q];
    let x1 = X[4u * q + 1u];
    let x2 = X[4u * q + 2u];
    let x3 = X[4u * q + 3u];
    ${r.map((i) => `acc${i} += quad(base${i}, q, x0, x1, x2, x3);`).join('\n    ')}
  }
  ${r.map((i) => `partial[${i}u * WG + t] = acc${i};`).join('\n  ')}
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (t < s) {
      ${r.map((i) => `partial[${i}u * WG + t] += partial[${i}u * WG + t + s];`).join('\n      ')}
    }
    workgroupBarrier();
  }
  if (t < ${rows}u) {
    let row = row0 + t;
    if (row < dims.rows) {
      Y[row] = partial[t * WG] * P[row] + P[dims.rows + row];
    } else if (row < dims.rowsPad) {
      Y[row] = 0.0;  // keep the padding the next layer reads at zero
    }
  }
}
`;
}

const POST_SHADER = /* wgsl */ `
struct Post { n: u32, eps: f32, flags: u32, unused: u32 }  // flags: 1 layer norm, 2 SiLU
@group(0) @binding(0) var<uniform> cfg: Post;
@group(0) @binding(1) var<storage, read> P: array<f32>;  // gamma[n], beta[n]
@group(0) @binding(2) var<storage, read_write> H: array<f32>;

const WG = ${POST_WG}u;
var<workgroup> red: array<f32, ${POST_WG}>;

fn total(t: u32, v: f32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (t < s) {
      red[t] += red[t + s];
    }
    workgroupBarrier();
  }
  let sum = red[0];
  workgroupBarrier();
  return sum;
}

@compute @workgroup_size(${POST_WG})
fn main(@builtin(local_invocation_index) t: u32) {
  let n = cfg.n;
  var s = 0.0;
  for (var i = t; i < n; i += WG) {
    s += H[i];
  }
  let mean = total(t, s) / f32(n);
  var v = 0.0;
  for (var i = t; i < n; i += WG) {
    let c = H[i] - mean;
    v += c * c;
  }
  let rstd = inverseSqrt(total(t, v) / f32(n) + cfg.eps);
  let norm = (cfg.flags & 1u) != 0u;
  let silu = (cfg.flags & 2u) != 0u;
  for (var i = t; i < n; i += WG) {
    var y = H[i];
    if (norm) {
      y = (y - mean) * rstd * P[i] + P[n + i];
    }
    if (silu) {
      let e = exp(-abs(y));  // sigmoid without overflow
      y = y * select(e, 1.0, y >= 0.0) / (1.0 + e);
    }
    H[i] = y;
  }
}
`;

// Which layers this backend runs: an optional leading normalize (on the CPU),
// int8_row linear layers, layer norms and SiLUs. Throws on anything else.
export function checkSupported(net) {
  net.layers.forEach((layer, index) => {
    if (layer.kind === 'normalize' && index !== 0) throw new Error('WebGPU backend: normalize must come first');
    if (layer.kind === 'linear' && layer.quant !== 'int8_row') {
      throw new Error(`WebGPU backend: ${layer.quant} linear layers are not supported (int8_row only)`);
    }
  });
  if (!net.layers.some((layer) => layer.kind === 'linear')) throw new Error('WebGPU backend: no linear layer');
}

function adapterLabel(adapter) {
  const info = adapter && adapter.info;
  if (!info) return 'unknown adapter';
  const parts = [info.vendor, info.architecture, info.device, info.description].filter(Boolean);
  const label = parts.join(' ') || 'unknown adapter';
  return info.isFallbackAdapter ? `${label} (fallback adapter)` : label;
}

async function pipeline(device, code, label) {
  const module = device.createShaderModule({ code, label });
  try {
    return await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
  } catch (error) {
    let details = '';
    try {
      const info = await module.getCompilationInfo();
      details = info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`).join('; ');
    } catch { /* no compilation info */ }
    throw new Error(`WebGPU ${label} pipeline: ${error.message}${details ? ` (${details})` : ''}`);
  }
}

// {backend: 'webgpu', adapter, run(x) -> Promise<Float32Array>, dispose()}.
// options.gpu: a GPU object (default navigator.gpu); options.device: an existing
// GPUDevice to share (then dispose() leaves it alive); options.software: false
// refuses a fallback (software) adapter such as SwiftShader.
export async function createWebGpuBackend(net, options = {}) {
  checkSupported(net);
  let device = options.device || null;
  const ownDevice = !device;
  let adapter = 'shared device';
  if (!device) {
    const gpu = options.gpu || (globalThis.navigator && globalThis.navigator.gpu);
    if (!gpu) throw new Error('WebGPU is not available (no navigator.gpu)');
    const found = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!found) throw new Error('WebGPU: no adapter');
    adapter = adapterLabel(found);
    const software = found.info ? found.info.isFallbackAdapter : found.isFallbackAdapter;
    if (software && options.software === false) throw new Error(`software adapter (${adapter})`);
    device = await found.requestDevice();
  }
  let failure = null; // set once the device is lost or reports an error
  device.lost.then((info) => {
    failure = failure || `device lost (${info.reason}): ${info.message}`;
  });
  const onError = (event) => {
    failure = failure || `GPU error: ${event.error && event.error.message}`;
  };
  if (typeof device.addEventListener === 'function') device.addEventListener('uncapturederror', onError);

  const buffers = [];
  const create = (size, usage) => {
    const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    buffers.push(buffer);
    return buffer;
  };
  const upload = (bytes, usage) => {
    const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(bytes.byteLength / 16) * 16), usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(bytes);
    buffer.unmap();
    buffers.push(buffer);
    return buffer;
  };
  const asBytes = (typed) => new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);

  async function build() {
    const [gemv, postPipe] = await Promise.all([
      pipeline(device, gemvShader(ROWS), 'gemv-int8'),
      pipeline(device, POST_SHADER, 'layernorm-silu'),
    ]);

    // Activation vectors: the padded input, then two ping-pong vectors.
    let maxPad = 0;
    for (const layer of net.layers) if (layer.kind === 'linear') maxPad = Math.max(maxPad, pad(layer.in), pad(layer.out));
    const inputPad = pad(net.inputSize);
    const host = new Float32Array(inputPad);
    const input = create(inputPad * 4, USAGE.STORAGE | USAGE.COPY_DST);
    const pingPong = [create(maxPad * 4, USAGE.STORAGE | USAGE.COPY_SRC), create(maxPad * 4, USAGE.STORAGE | USAGE.COPY_SRC)];
    const dummy = create(16, USAGE.STORAGE);

    const steps = [];
    let current = input;
    let size = net.inputSize;
    let flip = 0;
    let normalize = null;
    for (let i = 0; i < net.layers.length; i++) {
      const layer = net.layers[i];
      if (layer.kind === 'normalize') {
        normalize = layer;
      } else if (layer.kind === 'linear') {
        const inPad = pad(layer.in);
        const rowsPad = pad(layer.out);
        let bytes = asBytes(layer.weight);
        if (inPad !== layer.in) {
          bytes = new Uint8Array(layer.out * inPad);
          const source = asBytes(layer.weight);
          for (let r = 0; r < layer.out; r++) bytes.set(source.subarray(r * layer.in, (r + 1) * layer.in), r * inPad);
        }
        const params = new Float32Array(2 * layer.out);
        params.set(layer.scale, 0);
        params.set(layer.bias, layer.out);
        const dims = new Uint32Array([layer.out, inPad / 16, rowsPad, 0]);
        const output = pingPong[flip];
        flip ^= 1;
        const bindGroup = device.createBindGroup({
          layout: gemv.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: upload(asBytes(dims), USAGE.UNIFORM) } },
            { binding: 1, resource: { buffer: upload(bytes, USAGE.STORAGE) } },
            { binding: 2, resource: { buffer: upload(asBytes(params), USAGE.STORAGE) } },
            { binding: 3, resource: { buffer: current } },
            { binding: 4, resource: { buffer: output } },
          ],
        });
        steps.push({ pipeline: gemv, bindGroup, groups: rowsPad / ROWS });
        current = output;
        size = layer.out;
      } else {
        // layernorm (fused with a following silu) or a lone silu, in place
        const norm = layer.kind === 'layernorm' ? layer : null;
        const silu = layer.kind === 'silu' || (net.layers[i + 1] && net.layers[i + 1].kind === 'silu');
        if (norm && silu) i++;
        const uniform = new ArrayBuffer(16);
        new Uint32Array(uniform, 0, 1)[0] = size;
        new Float32Array(uniform, 4, 1)[0] = norm ? norm.eps : 0;
        new Uint32Array(uniform, 8, 1)[0] = (norm ? 1 : 0) | (silu ? 2 : 0);
        let params = dummy;
        if (norm) {
          const values = new Float32Array(2 * size);
          values.set(norm.gamma, 0);
          values.set(norm.beta, size);
          params = upload(asBytes(values), USAGE.STORAGE);
        }
        const bindGroup = device.createBindGroup({
          layout: postPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: upload(new Uint8Array(uniform), USAGE.UNIFORM) } },
            { binding: 1, resource: { buffer: params } },
            { binding: 2, resource: { buffer: current } },
          ],
        });
        steps.push({ pipeline: postPipe, bindGroup, groups: 1 });
      }
    }
    const outBytes = size * 4;
    const readBytes = Math.ceil(outBytes / 16) * 16;
    const read = create(readBytes, USAGE.MAP_READ | USAGE.COPY_DST);
    return { steps, host, input, output: current, read, outBytes, readBytes, normalize };
  }

  // Setup inside error scopes: a validation or out-of-memory error during it
  // fails the backend here instead of producing silent garbage later.
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let built = null;
  let setupError = null;
  try {
    built = await build();
  } catch (error) {
    setupError = error;
  }
  const validation = await device.popErrorScope().catch(() => null);
  const memory = await device.popErrorScope().catch(() => null);
  if (setupError || validation || memory) {
    if (typeof device.removeEventListener === 'function') device.removeEventListener('uncapturederror', onError);
    for (const buffer of buffers) {
      try { buffer.destroy(); } catch { /* ignore */ }
    }
    if (ownDevice) {
      try { device.destroy(); } catch { /* ignore */ }
    }
    throw setupError || new Error(`WebGPU setup: ${(validation || memory).message}`);
  }
  const { steps, host, input, output, read, outBytes, readBytes, normalize } = built;

  async function run(x) {
    if (failure) throw new Error(`WebGPU ${failure}`);
    if (!x || x.length !== net.inputSize) throw new Error(`${net.name}: expected ${net.inputSize} inputs, got ${x && x.length}`);
    if (normalize) normalizeInto(normalize, x, host);
    else host.set(x);
    device.queue.writeBuffer(input, 0, host);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const step of steps) {
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      pass.dispatchWorkgroups(step.groups);
    }
    pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, readBytes);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(MAP_READ, 0, readBytes);
    const result = new Float32Array(read.getMappedRange(0, readBytes).slice(0, outBytes));
    read.unmap();
    if (failure) throw new Error(`WebGPU ${failure}`);
    return result;
  }

  function dispose() {
    failure = failure || 'disposed';
    if (typeof device.removeEventListener === 'function') device.removeEventListener('uncapturederror', onError);
    for (const buffer of buffers) {
      try { buffer.destroy(); } catch { /* already gone */ }
    }
    buffers.length = 0;
    if (ownDevice) {
      try { device.destroy(); } catch { /* already gone */ }
    }
  }

  return { backend: 'webgpu', adapter, device, dispatches: steps.length, run, dispose };
}
