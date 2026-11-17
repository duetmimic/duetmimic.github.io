// A validated MLP description from a DUETBIN1 container that
// tools/duet_export/policy.py wrote (format "duetmimic-mlp/1"; the layer list
// is documented in that file's docstring). Both backends build on this.

export const FORMAT = 'duetmimic-mlp/1';

function fail(name, message) {
  throw new Error(`MLP ${name}: ${message}`);
}

// {name, inputSize, outputSize, layers, macs, maxWidth} with each layer's
// arrays resolved to typed arrays and every size checked against its neighbours.
export function parseNetwork(container) {
  const meta = container && container.meta;
  const arrays = (container && container.arrays) || {};
  if (!meta || meta.format !== FORMAT) {
    throw new Error(`Not an MLP container (format ${meta && meta.format}, expected ${FORMAT})`);
  }
  const name = String(meta.name || 'policy');
  const array = (key, Type, length) => {
    const value = arrays[key];
    if (!(value instanceof Type)) fail(name, `array ${key} is missing or not ${Type.name}`);
    if (value.length !== length) fail(name, `array ${key} has ${value.length} values, expected ${length}`);
    return value;
  };
  const inputSize = meta.input_size;
  if (!Number.isInteger(inputSize) || inputSize < 1) fail(name, 'invalid input_size');
  if (!Array.isArray(meta.layers) || meta.layers.length === 0) fail(name, 'no layers');

  let size = inputSize;
  let macs = 0;
  let maxWidth = inputSize;
  const layers = meta.layers.map((entry, index) => {
    const where = `layer ${index} (${entry.kind})`;
    switch (entry.kind) {
      case 'normalize':
        if (entry.size !== size) fail(name, `${where}: size ${entry.size}, vector has ${size}`);
        return { kind: 'normalize', size, mean: array(entry.mean, Float32Array, size), div: array(entry.div, Float32Array, size) };
      case 'linear': {
        const { in: inSize, out } = entry;
        if (inSize !== size) fail(name, `${where}: expects ${inSize} inputs, vector has ${size}`);
        if (!Number.isInteger(out) || out < 1) fail(name, `${where}: invalid output size`);
        const layer = { kind: 'linear', in: inSize, out, quant: entry.quant, bias: array(entry.bias, Float32Array, out) };
        if (entry.quant === 'int8_row') {
          layer.weight = array(entry.weight, Int8Array, out * inSize);
          layer.scale = array(entry.scale, Float32Array, out);
        } else if (entry.quant === 'f32') {
          layer.weight = array(entry.weight, Float32Array, out * inSize);
        } else {
          fail(name, `${where}: unknown weight format ${entry.quant}`);
        }
        macs += inSize * out;
        size = out;
        maxWidth = Math.max(maxWidth, out);
        return layer;
      }
      case 'layernorm':
        if (entry.size !== size) fail(name, `${where}: size ${entry.size}, vector has ${size}`);
        if (!(entry.eps > 0) || !Number.isFinite(entry.eps)) fail(name, `${where}: invalid eps`);
        return { kind: 'layernorm', size, eps: entry.eps, gamma: array(entry.gamma, Float32Array, size), beta: array(entry.beta, Float32Array, size) };
      case 'silu':
        return { kind: 'silu' };
      default:
        return fail(name, `${where}: unsupported layer kind`);
    }
  });
  if (size !== meta.output_size) fail(name, `output has ${size} values, output_size is ${meta.output_size}`);
  return { name, inputSize, outputSize: size, layers, macs, maxWidth };
}

// out[i] = (x[i] - mean[i]) / div[i] for the layer's size (in place is fine).
export function normalizeInto(layer, x, out) {
  const { mean, div, size } = layer;
  for (let i = 0; i < size; i++) out[i] = (x[i] - mean[i]) / div[i];
}

// A deterministic input whose normalized values are uniform in [-1.5, 1.5]:
// the WebGPU self-test and the CPU kernel race run on it.
export function selfTestInput(net) {
  const x = new Float32Array(net.inputSize);
  const norm = net.layers[0].kind === 'normalize' ? net.layers[0] : null;
  let s = 0x2545f491;
  for (let i = 0; i < x.length; i++) {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    const z = ((s >>> 0) / 4294967296) * 3 - 1.5;
    x[i] = norm ? norm.mean[i] + norm.div[i] * z : z;
  }
  return x;
}
