// MLP engine checks (demo/js/nn): node --test test/nn.test.mjs (from tools/).
//
// Needs the exported weights (demo/assets/stage2/{human,osl}.bin.gz, or
// DUET_ASSETS; see paths.mjs) with their demo.json, and the reference outputs
// that the exporter writes with them (NN_REF in paths.mjs:
// tools/traces/nn_reference.bin.gz for the demo's folder, the file of that
// name inside DUET_ASSETS otherwise):
//   python -m duet_export --out ../demo/assets/stage2 --only policies
// The WebGPU tests use Dawn through the 'webgpu' package and skip when it has
// no adapter. Timings are printed as test diagnostics.

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { loadContainer } from '../../demo/js/lib/container.js';
import {
  CPU_KERNELS, GPU_SLOWER_FACTOR, loadPolicy, parseNetwork, selfTest, selfTestInput, SELF_TEST_TOLERANCE,
} from '../../demo/js/nn/index.js';
import { createCpuBackend } from '../../demo/js/nn/cpu.js';
import { ASSETS, NN_REF } from './paths.mjs';

const REFERENCE = NN_REF;
const CPU_TOLERANCE = 1e-4; // JS CPU vs the float64 numpy forward of the same weights
const NETS = ['human', 'osl'];

const missing = [REFERENCE, new URL('human.bin.gz', ASSETS), new URL('osl.bin.gz', ASSETS), new URL('demo.json', ASSETS)]
  .filter((url) => !existsSync(fileURLToPath(url)))
  .map((url) => fileURLToPath(url));
const skipAll = missing.length
  ? `missing ${missing.join(', ')}: run python -m duet_export --out ${fileURLToPath(ASSETS)} --only policies`
  : false;

const reference = skipAll ? null : await loadContainer(REFERENCE);
// The network sizes the export recorded (demo.json "policies").
const demo = skipAll ? null : JSON.parse(readFileSync(new URL('demo.json', ASSETS), 'utf8'));
const rows = (name, key) => {
  const array = reference.arrays[`${name}.${key}`];
  const [count, width] = array.shape;
  return Array.from({ length: count }, (_, i) => array.subarray(i * width, (i + 1) * width));
};
const maxAbs = (a, b) => {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
};
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
async function timeRuns(policy, input, count) {
  const times = [];
  for (let i = 0; i < count; i++) {
    const t = performance.now();
    await policy.run(input);
    times.push(performance.now() - t);
  }
  return { median: median(times), min: Math.min(...times) };
}

// Dawn for Node; null when the package or an adapter is missing.
let dawn = null;
let dawnSkip = skipAll;
if (!skipAll) {
  try {
    const { create } = await import('webgpu');
    const gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) dawn = { gpu, software: !!(adapter.info && adapter.info.isFallbackAdapter) };
    else dawnSkip = 'Dawn found no WebGPU adapter';
  } catch (error) {
    dawnSkip = `no WebGPU in Node: ${error.message}`;
  }
}
after(() => {
  // dawn.node keeps the process alive while the GPU object is referenced.
  dawn = null;
});

test('reference outputs match the exported containers', { skip: skipAll }, () => {
  for (const name of NETS) {
    const net = reference.meta.nets[name];
    const bytes = readFileSync(new URL(net.file, ASSETS));
    const digest = createHash('sha256').update(bytes).digest('hex');
    assert.equal(digest, net.container_sha256, `${net.file} changed since the reference was written: re-run the exporter`);
  }
});

test('containers parse into the expected networks', { skip: skipAll }, async () => {
  const human = parseNetwork(await loadContainer(new URL('human.bin.gz', ASSETS)));
  assert.equal(human.inputSize, demo.policies.human.input_size);
  assert.equal(human.outputSize, demo.policies.human.output_size);
  assert.equal(human.inputSize, demo.sim.human.observation.size);
  assert.equal(human.outputSize, demo.sim.human.action.size);
  // The stage-2 human actor: the same architecture in R2 and B (different
  // weights), 2338 -> ... -> 339.
  assert.equal(human.inputSize, 2338);
  assert.equal(human.outputSize, 339);
  assert.equal(human.layers.filter((l) => l.kind === 'linear').length, 17);
  assert.ok(human.layers.filter((l) => l.kind === 'linear').every((l) => l.quant === 'int8_row'));
  assert.equal(human.macs, 26858496);
  const osl = parseNetwork(await loadContainer(new URL('osl.bin.gz', ASSETS)));
  assert.equal(osl.inputSize, demo.policies.osl.input_size);
  assert.equal(osl.inputSize, demo.sim.osl.observation.size);
  assert.equal(osl.outputSize, demo.policies.osl.output_size);
  assert.equal(osl.outputSize, demo.sim.osl.action.size);
  assert.equal(osl.outputSize, 2);
  assert.ok(osl.layers.filter((l) => l.kind === 'linear').every((l) => l.quant === 'f32'));
});

for (const name of NETS) {
  test(`cpu backend matches the float64 reference: ${name}`, { skip: skipAll }, async (t) => {
    const policy = await loadPolicy(new URL(`${name}.bin.gz`, ASSETS), { backend: 'cpu' });
    assert.equal(policy.backend, 'cpu');
    const inputs = rows(name, 'input');
    const expected = rows(name, 'expected');
    const onnx = reference.arrays[`${name}.onnx`] ? rows(name, 'onnx') : null;
    let worst = 0;
    let worstOnnx = 0;
    for (let i = 0; i < inputs.length; i++) {
      const out = await policy.run(inputs[i]);
      assert.equal(out.length, policy.outputSize);
      worst = Math.max(worst, maxAbs(out, expected[i]));
      if (onnx) worstOnnx = Math.max(worstOnnx, maxAbs(out, onnx[i]));
    }
    const timing = await timeRuns(policy, inputs[0], name === 'human' ? 20 : 200);
    t.diagnostic(`${name} cpu: max |js - fp64 ref| = ${worst.toExponential(2)} over ${inputs.length} inputs; ` +
      `max |js - onnx fp32| = ${worstOnnx.toExponential(2)}; ${timing.median.toFixed(2)} ms median, ${timing.min.toFixed(2)} ms min`);
    assert.ok(worst <= CPU_TOLERANCE, `max diff ${worst} > ${CPU_TOLERANCE}`);
    policy.dispose();
  });
}

// The int8 human weights were validated in closed loop at these errors against
// onnxruntime fp32 (tools/duet_export/policy.py, "Why int8 per row is enough"):
// 1.9e-3 RMS and 1.2e-2 max on the reference rows. A changed quantizer that
// leaves this envelope needs a new closed-loop check, not just a new reference.
const HUMAN_ENVELOPE = { rms: 3e-3, max: 2.5e-2 };
test('human weights stay inside the closed-loop-validated error envelope', { skip: skipAll }, async (t) => {
  if (!reference.arrays['human.onnx']) {
    t.skip('the reference has no onnxruntime outputs (exported without onnxruntime)');
    return;
  }
  const policy = await loadPolicy(new URL('human.bin.gz', ASSETS), { backend: 'cpu' });
  const inputs = rows('human', 'input');
  const onnx = rows('human', 'onnx');
  let sq = 0;
  let count = 0;
  let worst = 0;
  for (let i = 0; i < inputs.length; i++) {
    const out = await policy.run(inputs[i]);
    for (let j = 0; j < out.length; j++) {
      const d = out[j] - onnx[i][j];
      sq += d * d;
      count++;
      worst = Math.max(worst, Math.abs(d));
    }
  }
  const rms = Math.sqrt(sq / count);
  t.diagnostic(`human js vs onnxruntime fp32: rms ${rms.toExponential(2)}, max ${worst.toExponential(2)} ` +
    `over ${inputs.length} inputs (envelope rms ${HUMAN_ENVELOPE.rms}, max ${HUMAN_ENVELOPE.max})`);
  assert.ok(rms <= HUMAN_ENVELOPE.rms, `rms ${rms} > ${HUMAN_ENVELOPE.rms}`);
  assert.ok(worst <= HUMAN_ENVELOPE.max, `max ${worst} > ${HUMAN_ENVELOPE.max}`);
  policy.dispose();
});

// The two int8 kernels sum the same exact products in the same order: their
// float32 outputs must agree bit for bit, not just to a tolerance.
test('cpu int8 kernels (packed, int8) give bit-identical outputs', { skip: skipAll }, async (t) => {
  const net = parseNetwork(await loadContainer(new URL('human.bin.gz', ASSETS)));
  const backends = CPU_KERNELS.map((kernel) => createCpuBackend(net, { kernel }));
  assert.deepEqual(backends.map((b) => b.kernel), CPU_KERNELS);
  const inputs = [...rows('human', 'input'), selfTestInput(net)];
  // A 2-column tail (2338 inputs) and rows % 8 != 0 are covered by the real
  // network; this odd shape covers 1- and 3-column tails and leftover rows.
  let differing = 0;
  for (const x of inputs) {
    const [a, b] = backends.map((backend) => backend.runSync(x));
    const [ua, ub] = [new Uint32Array(a.buffer), new Uint32Array(b.buffer)];
    for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) differing += 1;
  }
  assert.equal(differing, 0, `${differing} outputs differ in their float32 bits`);
  for (const cols of [1, 3, 5, 7, 37]) {
    const out = 11;
    const weight = Int8Array.from({ length: out * cols }, (_, i) => ((i * 37 + 11) % 255) - 127);
    const tiny = parseNetwork({
      meta: { format: 'duetmimic-mlp/1', name: 'tiny', input_size: cols, output_size: out, layers: [
        { kind: 'linear', in: cols, out, quant: 'int8_row', weight: 'w', scale: 's', bias: 'b' }] },
      arrays: { w: weight, s: Float32Array.from({ length: out }, (_, i) => 0.01 * (i + 1)), b: new Float32Array(out).fill(0.25) },
    });
    const x = Float32Array.from({ length: cols }, (_, i) => Math.sin(1.7 * i) * (i % 3 ? 1 : 1e-3));
    const [a, b] = CPU_KERNELS.map((kernel) => createCpuBackend(tiny, { kernel }).runSync(x));
    assert.deepEqual(new Uint32Array(a.buffer), new Uint32Array(b.buffer), `${cols} columns`);
  }
  t.diagnostic(`${inputs.length} human inputs and 5 odd shapes: identical bits`);
});

test('the cpu backend races its int8 kernels and records the choice', { skip: skipAll }, async (t) => {
  const human = await loadPolicy(new URL('human.bin.gz', ASSETS), { backend: 'cpu' });
  const { cpuKernel, cpuMs, cpuRace } = human.info;
  t.diagnostic(`human: kernel ${cpuKernel}; ms per inference ${JSON.stringify(cpuRace, (k, v) => (typeof v === 'number' ? +v.toFixed(2) : v))}`);
  assert.ok(CPU_KERNELS.includes(cpuKernel), `cpuKernel ${cpuKernel}`);
  for (const kernel of CPU_KERNELS) assert.ok(cpuRace[kernel] > 0 && Number.isFinite(cpuRace[kernel]), `${kernel}: ${cpuRace[kernel]}`);
  assert.equal(cpuMs, cpuRace[cpuKernel]);
  assert.equal(cpuMs, Math.min(...CPU_KERNELS.map((k) => cpuRace[k])));
  human.dispose();
  const forced = await loadPolicy(new URL('human.bin.gz', ASSETS), { backend: 'cpu', cpuKernel: 'int8' });
  assert.equal(forced.info.cpuKernel, 'int8');
  assert.equal(forced.info.cpuRace, null);
  forced.dispose();
  const osl = await loadPolicy(new URL('osl.bin.gz', ASSETS), { backend: 'cpu' });
  assert.equal(osl.info.cpuKernel, null, 'an fp32 network has no int8 kernel to pick');
  osl.dispose();
});

// The self-test times the CPU first and stops timing a GPU as soon as its best
// run is over GPU_SLOWER_FACTOR x the CPU (Firefox: about 100 ms per readback).
test('selfTest stops timing a slow GPU after one run', { skip: skipAll }, async () => {
  const net = parseNetwork(await loadContainer(new URL('human.bin.gz', ASSETS)));
  const cpu = createCpuBackend(net);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const answer = cpu.runSync(selfTestInput(net)); // a fake GPU's results cost no CPU time
  const fake = (delayMs) => {
    const gpu = { calls: 0, run: async () => { gpu.calls += 1; if (delayMs) await sleep(delayMs); return answer.slice(); } };
    return gpu;
  };
  const slow = fake(Math.ceil(3 * GPU_SLOWER_FACTOR * cpu.ms) + 20);
  const result = await selfTest(net, cpu, slow);
  assert.equal(result.maxDiff, 0);
  assert.equal(result.cpuMs, cpu.ms, 'the CPU time comes from the kernel race');
  assert.equal(result.gpuRuns, 1);
  assert.equal(slow.calls, 2, 'one check run, one timed run');
  assert.ok(result.gpuMs > GPU_SLOWER_FACTOR * result.cpuMs);
  const fast = fake(0);
  const quick = await selfTest(net, cpu, fast);
  assert.equal(quick.gpuRuns, 5, 'a GPU that keeps up gets the best of 5');
  const wrong = { run: async () => answer.map((v) => v + 1) };
  const failed = await selfTest(net, cpu, wrong);
  assert.ok(failed.maxDiff > SELF_TEST_TOLERANCE);
  assert.equal(failed.gpuRuns, 0, 'a GPU with wrong results is not timed');
});

test('run() copies its input, keeps call order, rejects a wrong size', { skip: skipAll }, async () => {
  const policy = await loadPolicy(new URL('osl.bin.gz', ASSETS), { backend: 'cpu' });
  const [a, b] = rows('osl', 'input');
  const expectA = await policy.run(a);
  const expectB = await policy.run(b);
  const buffer = Float32Array.from(a);
  const first = policy.run(buffer);
  buffer.set(b); // mutating after the call must not change the first result
  const second = policy.run(buffer);
  assert.deepEqual(await first, expectA);
  assert.deepEqual(await second, expectB);
  const size = demo.policies.osl.input_size;
  await assert.rejects(policy.run(new Float32Array(size - 1)), new RegExp(`expected ${size} inputs`));
  policy.dispose();
});

test('parseNetwork rejects foreign or inconsistent containers', () => {
  assert.throws(() => parseNetwork({ meta: { format: 'other' }, arrays: {} }), /Not an MLP container/);
  const meta = { format: 'duetmimic-mlp/1', name: 'x', input_size: 2, output_size: 1, layers: [
    { kind: 'linear', in: 3, out: 1, quant: 'f32', weight: 'w', bias: 'b' }] };
  const arrays = { w: new Float32Array(3), b: new Float32Array(1) };
  assert.throws(() => parseNetwork({ meta, arrays }), /expects 3 inputs/);
  meta.layers[0] = { kind: 'gelu' };
  assert.throws(() => parseNetwork({ meta, arrays }), /unsupported layer kind/);
});

test('auto keeps the small OSL network on the CPU', { skip: skipAll }, async () => {
  const policy = await loadPolicy(new URL('osl.bin.gz', ASSETS), { backend: 'auto', gpu: dawn && dawn.gpu });
  assert.equal(policy.backend, 'cpu');
  assert.match(policy.info.reason, /small network/);
  policy.dispose();
});

test('auto without any WebGPU falls back to the CPU', { skip: skipAll }, async () => {
  const saved = globalThis.navigator && globalThis.navigator.gpu;
  assert.equal(saved, undefined, 'Node has no navigator.gpu');
  const policy = await loadPolicy(new URL('human.bin.gz', ASSETS), { backend: 'auto' });
  assert.equal(policy.backend, 'cpu');
  assert.match(policy.info.reason, /WebGPU unavailable/);
  await assert.rejects(loadPolicy(new URL('human.bin.gz', ASSETS), { backend: 'webgpu' }), /not available/);
  policy.dispose();
});

test('webgpu backend (Dawn) matches the CPU backend: human', { skip: dawnSkip || (!dawn && 'no adapter') }, async (t) => {
  const url = new URL('human.bin.gz', ASSETS);
  const cpu = await loadPolicy(url, { backend: 'cpu' });
  const gpu = await loadPolicy(url, { backend: 'webgpu', gpu: dawn.gpu });
  assert.equal(gpu.backend, 'webgpu');
  t.diagnostic(`adapter: ${gpu.info.adapter}; self-test max diff ${gpu.info.selfTest.maxDiff.toExponential(2)}, ` +
    `gpu ${gpu.info.selfTest.gpuMs.toFixed(2)} ms, cpu ${gpu.info.selfTest.cpuMs.toFixed(2)} ms`);
  const inputs = rows('human', 'input');
  const expected = rows('human', 'expected');
  let vsCpu = 0;
  let vsRef = 0;
  for (let i = 0; i < inputs.length; i++) {
    const [g, c] = [await gpu.run(inputs[i]), await cpu.run(inputs[i])];
    vsCpu = Math.max(vsCpu, maxAbs(g, c));
    vsRef = Math.max(vsRef, maxAbs(g, expected[i]));
  }
  const fixed = selfTestInput(parseNetwork(await loadContainer(url)));
  const selfTest = maxAbs(await gpu.run(fixed), await cpu.run(fixed));
  for (let i = 0; i < 5; i++) await gpu.run(inputs[0]);
  const timing = await timeRuns(gpu, inputs[0], 50);
  t.diagnostic(`human webgpu (Dawn): max |gpu - cpu| = ${vsCpu.toExponential(2)}, max |gpu - fp64 ref| = ` +
    `${vsRef.toExponential(2)}, self-test input ${selfTest.toExponential(2)}; ` +
    `${timing.median.toFixed(3)} ms median, ${timing.min.toFixed(3)} ms min per inference (incl. readback)`);
  assert.ok(vsCpu <= SELF_TEST_TOLERANCE, `gpu vs cpu ${vsCpu}`);
  assert.ok(vsRef <= SELF_TEST_TOLERANCE, `gpu vs reference ${vsRef}`);
  gpu.dispose();
  cpu.dispose();
});

const hardwareSkip = dawnSkip || (!dawn && 'no adapter') || (dawn.software && 'Dawn offers only a software adapter');
test('auto picks WebGPU for the human network when a real adapter exists', { skip: hardwareSkip }, async (t) => {
  const policy = await loadPolicy(new URL('human.bin.gz', ASSETS), { backend: 'auto', gpu: dawn.gpu });
  t.diagnostic(`auto -> ${policy.backend} (${policy.info.reason})`);
  assert.equal(policy.backend, 'webgpu');
  policy.dispose();
});

test('a lost GPU device falls back to the CPU mid-run', { skip: dawnSkip || (!dawn && 'no adapter') }, async (t) => {
  const url = new URL('human.bin.gz', ASSETS);
  const adapter = await dawn.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const device = await adapter.requestDevice();
  const policy = await loadPolicy(url, { backend: 'webgpu', device });
  const cpu = await loadPolicy(url, { backend: 'cpu' });
  const [input] = rows('human', 'input');
  assert.equal(policy.backend, 'webgpu');
  await policy.run(input);
  device.destroy(); // what a driver reset or a lost context looks like to the page
  await device.lost;
  const out = await policy.run(input);
  assert.equal(policy.backend, 'cpu');
  t.diagnostic(`fallback reason: ${policy.info.fallback}`);
  assert.match(policy.info.reason, /^WebGPU failed, now on the CPU: /);
  assert.deepEqual(out, await cpu.run(input));
  policy.dispose();
  cpu.dispose();
});
