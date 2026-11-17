// MLP policy engine for the demo's two actor networks (no DOM; runs in a module
// Web Worker, on the main thread and in Node).
//
//   import { loadPolicy } from './nn/index.js';
//   const human = await loadPolicy(new URL('human.bin.gz', assets), { backend: 'auto' });
//   const mean = await human.run(observation);   // Float32Array(outputSize), a fresh array
//
// loadPolicy(source, options) -> Promise<policy>
//   source: URL or string (fetched; gzip handled by lib/container.js), container
//           bytes (Uint8Array/ArrayBuffer), or a parsed {meta, arrays} container.
//   options.backend: 'auto' (default) | 'webgpu' | 'cpu'
//     'auto'   WebGPU when navigator.gpu gives a hardware adapter and a device
//              AND the self-test passes (a fixed input agrees with the CPU
//              backend to 1e-3) AND the GPU is not clearly slower than the CPU;
//              a software adapter (SwiftShader, about 3x slower than the CPU
//              path) is skipped. Small networks (under 2M multiply-adds, e.g. the
//              OSL actor) always run on the CPU, where a GPU round trip would
//              cost more than the arithmetic.
//     'webgpu' WebGPU or an error (the self-test still runs and must pass).
//              A network the WebGPU kernels cannot run (they take int8_row
//              linear layers only, e.g. not the fp32 OSL actor) uses the CPU
//              under every backend option; info.reason says so.
//     'cpu'    plain JavaScript.
//   options.cpuKernel: 'auto' (default) | 'packed' | 'int8': the CPU backend's
//              int8 kernel; 'auto' times both at load and keeps the faster
//              (nn/cpu.js). Always built: it is also the WebGPU fallback.
//   options.onProgress(received, total): download progress (URL sources).
//   options.init: passed to fetch() for URL sources ({signal, cache, ...}). An
//              aborted signal also stops loadPolicy before the GPU setup.
//   options.gpu: a GPU object instead of navigator.gpu (Node: the 'webgpu' package).
//   options.device: share an existing GPUDevice (not destroyed by dispose()).
//
// policy: {backend, name, inputSize, outputSize, info, run(Float32Array) ->
//          Promise<Float32Array>, dispose()}
//   info: {requested, macs, reason, adapter, selfTest, fallback, cpuKernel,
//          cpuMs, cpuRace}. reason starts with one of 'requested', 'small
//          network', 'CPU only:', 'self-test passed', 'CPU faster (',
//          'WebGPU unavailable:' or, after a failure at run time,
//          'WebGPU failed, now on the CPU:'.
//   run() copies its input at call time and runs calls one after another. If the
//   GPU fails later (device lost, GPU error, an inference taking over 10 s), that
//   call and every later one run on the CPU instead: policy.backend becomes
//   'cpu', info.fallback holds the error and info.reason says so.
//   The output is the network's raw 'mean' (no clipping, no action scaling).

import { loadContainer, parseContainer } from '../lib/container.js';
import { parseNetwork, selfTestInput } from './network.js';
import { createCpuBackend } from './cpu.js';
import { checkSupported, createWebGpuBackend } from './webgpu.js';

export { parseNetwork, selfTestInput } from './network.js';
export { CPU_KERNELS } from './cpu.js';

export const SELF_TEST_TOLERANCE = 1e-3;
const GPU_MIN_MACS = 2e6; // 'auto': smaller networks stay on the CPU
export const GPU_SLOWER_FACTOR = 1.5; // 'auto': CPU wins only when the GPU is clearly slower
const GPU_TIMED_RUNS = 5;
const SELF_TEST_TIMEOUT_MS = 30000;
const RUN_TIMEOUT_MS = 10000; // a GPU inference this slow is treated as a failure

async function toContainer(source, onProgress, init) {
  if (source && source.meta && source.arrays) return source;
  if (source instanceof ArrayBuffer) return parseContainer(new Uint8Array(source));
  if (ArrayBuffer.isView(source)) return parseContainer(new Uint8Array(source.buffer, source.byteOffset, source.byteLength));
  return loadContainer(source, onProgress, init);
}

function throwIfAborted(signal) {
  if (signal && signal.aborted) throw signal.reason || new DOMException('The load was aborted.', 'AbortError');
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (!(d <= worst)) worst = d; // NaN wins
  }
  return worst;
}

const now = () => (globalThis.performance ? performance.now() : Date.now());

async function withTimeout(promise, ms, what) {
  promise.catch(() => {}); // a late rejection after the timeout is not unhandled
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

// ms per CPU inference: the kernel race's figure when it ran, else a batch of
// runs long enough for a clock coarsened to 1 ms (WebKit, Firefox).
function cpuMsPerRun(cpu, x) {
  if (Number.isFinite(cpu.ms)) return cpu.ms;
  cpu.runSync(x);
  let best = Infinity;
  for (let sample = 0; sample < 2; sample++) {
    const t0 = now();
    let runs = 0;
    let elapsed = 0;
    do {
      cpu.runSync(x);
      runs += 1;
      elapsed = now() - t0;
    } while (elapsed < 10 && runs < 100);
    best = Math.min(best, elapsed / runs);
  }
  return best;
}

// The WebGPU self-test on the fixed input: {maxDiff, gpuMs, cpuMs, gpuRuns}.
// The CPU is timed first. The GPU's first run finishes the pipeline warm-up and
// is checked against the CPU; then up to 5 timed runs (best of), stopping as
// soon as the best is over GPU_SLOWER_FACTOR x the CPU time: a GPU that slow
// loses anyway, and Firefox's readback takes about 100 ms per run.
export async function selfTest(net, cpu, gpu) {
  const x = selfTestInput(net);
  const cpuMs = cpuMsPerRun(cpu, x);
  const expected = cpu.runSync(x);
  const got = await gpu.run(x);
  const maxDiff = got.length === expected.length ? maxAbsDiff(got, expected) : Infinity;
  let gpuMs = Infinity;
  let gpuRuns = 0;
  if (maxDiff <= SELF_TEST_TOLERANCE) {
    while (gpuRuns < GPU_TIMED_RUNS) {
      const t = now();
      await gpu.run(x);
      gpuMs = Math.min(gpuMs, now() - t);
      gpuRuns += 1;
      if (gpuMs > GPU_SLOWER_FACTOR * cpuMs) break;
    }
  }
  return { maxDiff, gpuMs, cpuMs, gpuRuns };
}

// primary: the backend to use; cpu: the CPU backend to fall back to when the
// primary is WebGPU (null when the primary is the CPU itself).
function makePolicy(net, primary, cpu, info) {
  let active = primary;
  let chain = Promise.resolve();
  const policy = {
    backend: primary.backend,
    name: net.name,
    inputSize: net.inputSize,
    outputSize: net.outputSize,
    info,
    run(input) {
      if (!input || input.length !== net.inputSize) {
        return Promise.reject(new Error(`${net.name}: expected ${net.inputSize} inputs, got ${input && input.length}`));
      }
      const x = Float32Array.from(input);
      const job = chain.then(async () => {
        if (active.backend === 'cpu') return active.runSync(x);
        try {
          return await withTimeout(active.run(x), RUN_TIMEOUT_MS, 'WebGPU inference');
        } catch (error) {
          if (!cpu) throw error;
          // The GPU failed after it passed its self-test: continue on the CPU.
          info.fallback = `${error.message}`;
          info.reason = `WebGPU failed, now on the CPU: ${error.message}`;
          try { active.dispose(); } catch { /* ignore */ }
          active = cpu;
          policy.backend = 'cpu';
          if (typeof console !== 'undefined') console.warn(`[nn] ${net.name}: WebGPU failed, using the CPU: ${error.message}`);
          return cpu.runSync(x);
        }
      });
      chain = job.catch(() => {});
      return job;
    },
    dispose() {
      try { active.dispose(); } catch { /* ignore */ }
      if (cpu && cpu !== active) cpu.dispose();
    },
  };
  return policy;
}

export async function loadPolicy(source, options = {}) {
  const { backend = 'auto', cpuKernel = 'auto', onProgress, init, gpu, device } = options;
  if (!['auto', 'webgpu', 'cpu'].includes(backend)) throw new Error(`Unknown backend ${backend}`);
  const signal = init && init.signal;
  const net = parseNetwork(await toContainer(source, onProgress, init));
  throwIfAborted(signal);
  const cpu = createCpuBackend(net, { kernel: cpuKernel });
  const info = {
    requested: backend, macs: net.macs, reason: '', adapter: null, selfTest: null, fallback: null,
    cpuKernel: cpu.kernel, cpuMs: cpu.ms, cpuRace: cpu.race,
  };

  if (backend === 'cpu') return makePolicy(net, cpu, null, { ...info, reason: 'requested' });
  if (backend === 'auto' && net.macs < GPU_MIN_MACS) {
    return makePolicy(net, cpu, null, { ...info, reason: `small network (${(net.macs / 1e6).toFixed(2)}M multiply-adds)` });
  }
  try {
    checkSupported(net);
  } catch (error) {
    return makePolicy(net, cpu, null, { ...info, reason: `CPU only: ${error.message}` });
  }
  throwIfAborted(signal);

  let gpuBackend = null;
  try {
    gpuBackend = await createWebGpuBackend(net, { gpu, device, software: backend !== 'auto' });
    info.adapter = gpuBackend.adapter;
    const test = await withTimeout(selfTest(net, cpu, gpuBackend), SELF_TEST_TIMEOUT_MS, 'WebGPU self-test');
    info.selfTest = test;
    if (!(test.maxDiff <= SELF_TEST_TOLERANCE)) {
      throw new Error(`WebGPU self-test failed: max |GPU - CPU| = ${test.maxDiff}`);
    }
    if (backend === 'auto' && test.gpuMs > GPU_SLOWER_FACTOR * test.cpuMs) {
      gpuBackend.dispose();
      info.reason = `CPU faster (GPU ${test.gpuMs.toFixed(1)} ms, CPU ${test.cpuMs.toFixed(1)} ms on ${info.adapter})`;
      return makePolicy(net, cpu, null, info);
    }
    info.reason = backend === 'webgpu' ? 'requested' : 'self-test passed';
    return makePolicy(net, gpuBackend, cpu, info);
  } catch (error) {
    if (gpuBackend) gpuBackend.dispose();
    if (backend === 'webgpu') throw error;
    info.reason = `WebGPU unavailable: ${error.message}`;
    return makePolicy(net, cpu, null, info);
  }
}
