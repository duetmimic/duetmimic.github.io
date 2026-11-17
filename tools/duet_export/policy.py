"""Export section `policies`: the two actor networks as DUETBIN1 MLP containers.

    cd tools && python -m duet_export --out ../demo/assets/r2 --only policies

Reads `human_actor.onnx` and `osl_actor.onnx` of the bundle (with onnx_lite, no
`onnx` package), turns each graph into a flat layer list, checks a float64
numpy forward pass against onnxruntime and against the bundle's `parity.npz`,
and writes

- `human.bin.gz`: Gemm weights int8, symmetric per output row
  (scale = max|row| / 127); everything else float32;
- `osl.bin.gz`: everything float32 (0.29M parameters, not worth quantizing);
- `tools/traces/nn_reference.bin.gz` (git-ignored): a few fixed inputs and the
  float64 outputs of the *exported* (quantized) layers, computed from the
  container bytes as written, for tools/test/nn.test.mjs.

Why int8 per row is enough (closed-loop study, 2026-11-16). Against fp32 the
human net's action error is 1.3e-3 RMS / 1.2e-2 max on 1350 trace observations
(1.3e-3 / 1.5e-2 on 2400 closed-loop observations from random key sequences),
about 2 % of the action noise the policy trained with (std 0.064-0.070). In
osl_deploy's own runner.Simulation on native MuJoCo, with the bundle's OSL
ONNX and a numpy human net, int8 fell exactly as often as fp32: idle -> 1.3 or
1.4 m/s from 30 entry phases, 0 vs 0; -0.8 m/s arc reversals, 0/20 vs 0/20;
40 hard transitions at 1.4 / -0.8 m/s (reversals, turn in place -> 1.4 m/s,
stop -> 1.4 m/s), 2 vs 2 on the same two trials; random key sequences over the
UI table and an extended one (1.4 / -0.8 m/s, +-0.3 rad/s back arcs), 2 vs 2.
Weights in fp16 (50 MB gz) deviate from fp32 on the same borderline trials, so
the remaining falls belong to the policy, not to quantization. Measured and
not adopted, since no closed-loop benefit showed: fp16 for the most sensitive
Gemms (first, second-to-last and last: 29.3 MB gz, 45 % lower RMS error),
int8 with per-64-column scales (27.4 MB, 35 % lower), fp16 everywhere.
Per-Gemm sensitivity (int8 on that Gemm only, RMS action error): last 8.3e-4
(rows with max/rms up to 17), first 6.2e-4 (normalized inputs reach |z| 38),
second-to-last 4.9e-4, the others 1.5-3.3e-4.

Container format (DUETBIN1, see container.py). `meta`:

    {
      "format": "duetmimic-mlp/1",
      "name": "human",
      "input_size": 2338, "output_size": 339,
      "input_name": "observation", "output_name": "mean",   # ONNX tensor names
      "layers": [ ...applied in order to a 1-D float vector... ],
      "params": 26921879, "macs": 26900000,
      "source": {"onnx": "human_actor.onnx", "sha256": "...", "opset": 18}
    }

Layer kinds (every array is named explicitly by the layer that uses it):

    {"kind": "normalize", "size": n, "mean": "0.mean", "div": "0.div"}
        y = (x - mean) / div                       mean, div: float32 [n]
    {"kind": "linear", "in": i, "out": o, "quant": "int8_row",
     "weight": "1.weight", "scale": "1.scale", "bias": "1.bias"}
        y[r] = scale[r] * sum_c weight[r, c] * x[c] + bias[r]
                                                   weight: int8 [o, i] row-major
                                                   scale, bias: float32 [o]
    {"kind": "linear", "in": i, "out": o, "quant": "f32",
     "weight": "1.weight", "bias": "1.bias"}
        y = weight @ x + bias                      weight: float32 [o, i]
    {"kind": "layernorm", "size": n, "eps": 1e-5, "gamma": "2.gamma", "beta": "2.beta"}
        y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
        (biased variance over the whole vector, as ONNX LayerNormalization)
    {"kind": "silu"}
        y = x * sigmoid(x)

Array names are "<layer index>.<parameter>". The JS engine (demo/js/nn/)
validates this list and rejects any other kind.
"""

import hashlib
from pathlib import Path

import numpy as np

from . import onnx_lite
from .container import read_container, write_container

FORMAT = "duetmimic-mlp/1"
REFERENCE_FORMAT = "duetmimic-nn-reference/1"
CONTROLLERS = {"human": "int8_row", "osl": "f32"}  # name -> linear weight format
ONNX_TOLERANCE = 1e-5  # numpy float64 forward vs onnxruntime / parity means
TRACES = Path(__file__).resolve().parents[1] / "traces"
SYNTHETIC_INPUTS = 2  # extra reference inputs: normalized values ~ N(0, 1)


# --- ONNX graph -> layer list ---------------------------------------------------


def _const(graph, name, what):
    if name not in graph.initializers:
        raise ValueError(f"{what}: {name} is not a constant initializer")
    return np.asarray(graph.initializers[name])


def graph_layers(model):
    """The float layer list of a chain-structured MLP graph.

    Follows the single graph input through the nodes, matching exactly the
    patterns an rsl_rl actor exports: Sub/Div (observation normalizer), Gemm,
    LayerNormalization, Sigmoid+Mul (SiLU). Anything else is an error, so a
    changed network fails here instead of producing a wrong demo.
    """
    graph = model.graph
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError("Expected one graph input and one output")
    consumers = {}
    for node in graph.nodes:
        for name in node.inputs:
            consumers.setdefault(name, []).append(node)
    current, final = graph.inputs[0].name, graph.outputs[0].name
    layers, used = [], 0

    def single(name, op):
        nodes = consumers.get(name, [])
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op else None

    while current != final:
        nodes = consumers.get(current, [])
        ops = sorted(node.op_type for node in nodes)
        if len(nodes) == 1 and nodes[0].op_type in ("Sub", "Div"):
            node = nodes[0]
            if node.inputs[0] != current:
                raise ValueError(f"{node.op_type}: the vector must be the first operand")
            value = _const(graph, node.inputs[1], node.op_type).astype(np.float32).ravel()
            if (
                not layers
                or layers[-1]["kind"] != "normalize"
                or (node.op_type == "Sub" and "div_set" in layers[-1])
            ):
                layers.append({"kind": "normalize", "mean": None, "div": None})
            entry = layers[-1]
            if node.op_type == "Sub":
                if entry["mean"] is not None:
                    raise ValueError("Two Sub nodes in one normalizer")
                entry["mean"] = value
            else:
                if entry["div"] is not None:
                    raise ValueError("Two Div nodes in one normalizer")
                entry["div"], entry["div_set"] = value, True
            current = node.outputs[0]
        elif len(nodes) == 1 and nodes[0].op_type == "Gemm":
            node = nodes[0]
            attrs = node.attrs
            if node.inputs[0] != current or attrs.get("transA", 0) != 0:
                raise ValueError("Gemm: unsupported transA or operand order")
            weight = _const(graph, node.inputs[1], "Gemm weight").astype(np.float32)
            if attrs.get("transB", 0) == 0:
                weight = weight.T  # stored [in, out]
            weight = np.ascontiguousarray(weight * np.float32(attrs.get("alpha", 1.0)))
            if len(node.inputs) > 2 and node.inputs[2]:
                bias = _const(graph, node.inputs[2], "Gemm bias").astype(np.float32).ravel()
                bias = bias * np.float32(attrs.get("beta", 1.0))
            else:
                bias = np.zeros(weight.shape[0], dtype=np.float32)
            if bias.shape != (weight.shape[0],):
                raise ValueError("Gemm: bias does not match the output size")
            layers.append({"kind": "linear", "weight": weight, "bias": bias})
            current = node.outputs[0]
        elif len(nodes) == 1 and nodes[0].op_type == "LayerNormalization":
            node = nodes[0]
            if node.attrs.get("axis", -1) not in (-1, 1):
                raise ValueError("LayerNormalization: only the last axis is supported")
            gamma = _const(graph, node.inputs[1], "LayerNorm scale").astype(np.float32).ravel()
            if len(node.inputs) > 2 and node.inputs[2]:
                beta = _const(graph, node.inputs[2], "LayerNorm bias").astype(np.float32).ravel()
            else:
                beta = np.zeros_like(gamma)
            eps = float(node.attrs.get("epsilon", 1e-5))
            if len(node.outputs) > 1 and any(consumers.get(o) for o in node.outputs[1:]):
                raise ValueError("LayerNormalization: mean/inv-std outputs are used")
            layers.append({"kind": "layernorm", "gamma": gamma, "beta": beta, "eps": eps})
            current = node.outputs[0]
        elif ops == ["Mul", "Sigmoid"]:
            sigmoid = next(node for node in nodes if node.op_type == "Sigmoid")
            mul = next(node for node in nodes if node.op_type == "Mul")
            if single(sigmoid.outputs[0], "Mul") is not mul or sorted(mul.inputs) != sorted(
                [current, sigmoid.outputs[0]]
            ):
                raise ValueError("Sigmoid/Mul is not a SiLU")
            layers.append({"kind": "silu"})
            current = mul.outputs[0]
            used += 1
        elif len(nodes) == 1 and nodes[0].op_type == "Identity":
            current = nodes[0].outputs[0]
        else:
            names = [f"{node.op_type}({node.name})" for node in nodes] or ["nothing"]
            raise ValueError(f"Unsupported graph structure after {current}: {names}")
        used += 1

    if used != len(graph.nodes):
        raise ValueError(f"Graph has {len(graph.nodes)} nodes, the chain uses {used}")
    size = int(graph.inputs[0].shape[-1])
    for layer in layers:
        if layer["kind"] == "normalize":
            layer.pop("div_set", None)
            layer["mean"] = _broadcast(layer["mean"], size, 0.0)
            layer["div"] = _broadcast(layer["div"], size, 1.0)
        elif layer["kind"] == "linear":
            if layer["weight"].shape[1] != size:
                raise ValueError(f"Linear layer expects {layer['weight'].shape[1]}, gets {size}")
            size = layer["weight"].shape[0]
        elif layer["kind"] == "layernorm" and layer["gamma"].shape != (size,):
            raise ValueError("LayerNormalization: scale does not match the vector size")
    return layers


def _broadcast(value, size, default):
    if value is None:
        return np.full(size, default, dtype=np.float32)
    return np.broadcast_to(value.astype(np.float32), (size,)).copy()


# --- quantization and forward pass ---------------------------------------------


def quantize_rows(weight):
    """(int8 [o, i], float32 scale [o]): symmetric, per output row."""
    weight = np.asarray(weight, dtype=np.float64)
    scale = (np.abs(weight).max(axis=1) / 127.0).astype(np.float32)
    safe = np.where(scale > 0, scale.astype(np.float64), 1.0)
    q = np.clip(np.rint(weight / safe[:, None]), -127, 127).astype(np.int8)
    return q, scale


def forward(layers, x):
    """float64 forward pass of a layer list (float or container layers).

    A linear layer with "q" (int8) and "scale" uses the dequantized weight
    q * scale, exactly what the JS engine multiplies by.
    """
    h = np.atleast_2d(np.asarray(x, dtype=np.float64))
    for layer in layers:
        kind = layer["kind"]
        if kind == "normalize":
            h = (h - layer["mean"].astype(np.float64)) / layer["div"].astype(np.float64)
        elif kind == "linear":
            if "q" in layer:
                weight = layer["q"].astype(np.float64) * layer["scale"].astype(np.float64)[:, None]
            else:
                weight = layer["weight"].astype(np.float64)
            h = h @ weight.T + layer["bias"].astype(np.float64)
        elif kind == "layernorm":
            mu = h.mean(axis=-1, keepdims=True)
            var = ((h - mu) ** 2).mean(axis=-1, keepdims=True)
            h = (h - mu) / np.sqrt(var + layer["eps"])
            h = h * layer["gamma"].astype(np.float64) + layer["beta"].astype(np.float64)
        elif kind == "silu":
            h = h * 0.5 * (1.0 + np.tanh(0.5 * h))  # x * sigmoid(x), overflow-free
        else:
            raise ValueError(f"Unknown layer kind {kind}")
    return h


# --- container ------------------------------------------------------------------


def pack_layers(layers, quant):
    """(meta layer list, arrays) for write_container; linear weights as `quant`."""
    entries, arrays = [], {}
    for index, layer in enumerate(layers):
        kind, key = layer["kind"], f"{index}."
        if kind == "normalize":
            arrays[key + "mean"], arrays[key + "div"] = layer["mean"], layer["div"]
            entries.append(
                {
                    "kind": kind,
                    "size": int(layer["mean"].size),
                    "mean": key + "mean",
                    "div": key + "div",
                }
            )
        elif kind == "linear":
            out_size, in_size = layer["weight"].shape
            entry = {"kind": kind, "in": int(in_size), "out": int(out_size), "quant": quant}
            if quant == "int8_row":
                q, scale = quantize_rows(layer["weight"])
                arrays[key + "weight"], arrays[key + "scale"] = q, scale
                entry.update(weight=key + "weight", scale=key + "scale")
            elif quant == "f32":
                arrays[key + "weight"] = layer["weight"].astype(np.float32)
                entry.update(weight=key + "weight")
            else:
                raise ValueError(f"Unknown weight format {quant}")
            arrays[key + "bias"] = layer["bias"].astype(np.float32)
            entry["bias"] = key + "bias"
            entries.append(entry)
        elif kind == "layernorm":
            arrays[key + "gamma"], arrays[key + "beta"] = layer["gamma"], layer["beta"]
            entries.append(
                {
                    "kind": kind,
                    "size": int(layer["gamma"].size),
                    "eps": float(layer["eps"]),
                    "gamma": key + "gamma",
                    "beta": key + "beta",
                }
            )
        elif kind == "silu":
            entries.append({"kind": kind})
    return entries, arrays


def container_layers(meta, arrays):
    """The forward()-ready layer list of a container read back from disk."""
    layers = []
    for entry in meta["layers"]:
        kind = entry["kind"]
        if kind == "normalize":
            layers.append(
                {"kind": kind, "mean": arrays[entry["mean"]], "div": arrays[entry["div"]]}
            )
        elif kind == "linear":
            weight = arrays[entry["weight"]].reshape(entry["out"], entry["in"])
            layer = {"kind": kind, "bias": arrays[entry["bias"]]}
            if entry["quant"] == "int8_row":
                layer.update(q=weight, scale=arrays[entry["scale"]])
            else:
                layer["weight"] = weight
            layers.append(layer)
        elif kind == "layernorm":
            layers.append(
                {
                    "kind": kind,
                    "gamma": arrays[entry["gamma"]],
                    "beta": arrays[entry["beta"]],
                    "eps": entry["eps"],
                }
            )
        else:
            layers.append({"kind": kind})
    return layers


def _onnx_outputs(path, input_name, x):
    """onnxruntime outputs for each row of x (None when onnxruntime is missing)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rows = [session.run(None, {input_name: row[None].astype(np.float32)})[0][0] for row in x]
    return np.asarray(rows, dtype=np.float32)


def _sha256_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _width_list(layers):
    widths = []
    for layer in layers:
        if layer["kind"] == "linear":
            if not widths:
                widths.append(int(layer["weight"].shape[1]))
            widths.append(int(layer["weight"].shape[0]))
    return widths


# --- section entry point -----------------------------------------------------------


def export(bundle_dir, out_dir, ctx):
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    m = ctx["manifest"]
    if "controllers" not in m:
        raise ValueError("policies: expected a two-controller (osl_split_dual) bundle")
    parity = np.load(bundle_dir / "parity.npz")
    result, references = {}, {}
    rng = np.random.default_rng(20261116)

    for name, quant in CONTROLLERS.items():
        spec = m["controllers"][name]
        io = spec["io"]
        onnx_path = bundle_dir / spec["onnx"]
        model = onnx_lite.load(onnx_path)
        graph = model.graph
        in_size, out_size = int(io["input_shape"][-1]), int(io["output_shape"][-1])
        if (graph.inputs[0].name, graph.outputs[0].name) != (io["input"], io["output"]):
            raise ValueError(f"{name}: ONNX tensor names differ from the manifest")
        if graph.inputs[0].shape[-1] != in_size or graph.outputs[0].shape[-1] != out_size:
            raise ValueError(f"{name}: ONNX shapes differ from the manifest")
        layers = graph_layers(model)
        widths = _width_list(layers)
        if widths[0] != in_size or widths[-1] != out_size:
            raise ValueError(f"{name}: layer widths {widths} do not match the manifest")

        # 1. The float layer list reproduces the network (fp64 vs onnxruntime/parity).
        observations = parity[f"{name}_observations"].astype(np.float32)
        recorded = parity[f"{name}_means"].astype(np.float64)
        exact = forward(layers, observations)
        onnx = _onnx_outputs(onnx_path, io["input"], observations)
        check = {"parity_rows": int(len(observations))}
        check["recorded_max_abs"] = float(np.abs(exact - recorded).max())
        if onnx is not None:
            check["onnx_max_abs"] = float(np.abs(exact - onnx.astype(np.float64)).max())
        worst = max(check.get("onnx_max_abs", 0.0), check["recorded_max_abs"])
        if worst > ONNX_TOLERANCE:
            raise ValueError(f"{name}: float forward differs from ONNX by {worst:.3g}")

        # 2. Write the container, read it back, measure what the demo will run.
        entries, arrays = pack_layers(layers, quant)
        meta = {
            "format": FORMAT,
            "name": name,
            "input_size": in_size,
            "output_size": out_size,
            "input_name": io["input"],
            "output_name": io["output"],
            "layers": entries,
            "params": int(sum(a.size for a in graph.initializers.values())),
            "macs": int(sum(a * b for a, b in zip(widths[:-1], widths[1:], strict=True))),
            "source": {
                "onnx": spec["onnx"],
                "sha256": _sha256_bytes(onnx_path),
                "opset": model.opset.get("", None),
            },
        }
        filename = f"{name}.bin.gz"
        sizes = write_container(out_dir / filename, arrays, meta)
        read_meta, read_arrays = read_container(out_dir / filename)
        exported = container_layers(read_meta, read_arrays)
        quantized = forward(exported, observations)
        baseline = onnx.astype(np.float64) if onnx is not None else recorded
        error = quantized - baseline
        check["export_max_abs"] = float(np.abs(error).max())
        check["export_rms"] = float(np.sqrt(np.mean(error**2)))
        check["action_rms"] = float(np.sqrt(np.mean(baseline**2)))

        # 3. Reference inputs/outputs for the JS engine test (tools/traces).
        z = rng.standard_normal((SYNTHETIC_INPUTS, in_size))
        normalize = layers[0] if layers[0]["kind"] == "normalize" else None
        synthetic = z if normalize is None else normalize["mean"] + normalize["div"] * z
        inputs = np.concatenate([observations, synthetic.astype(np.float32)])
        references[name] = {
            "input": inputs,
            "expected": forward(exported, inputs),
            "onnx": _onnx_outputs(onnx_path, io["input"], inputs),
            "sha256": _sha256_bytes(out_dir / filename),
        }

        result[name] = {
            "file": filename,
            "input_size": in_size,
            "output_size": out_size,
            "layers": len(entries),
            "bytes": sizes["bytes"],
            "raw_bytes": sizes["raw_bytes"],
            "quant": quant,
            "params": meta["params"],
            "macs": meta["macs"],
            "widths": widths,
            "check": {k: (round(v, 9) if isinstance(v, float) else v) for k, v in check.items()},
        }
        print(
            f"[policies] {name}: {filename} {sizes['bytes'] / 1e6:.2f} MB gz "
            f"({sizes['raw_bytes'] / 1e6:.2f} MB raw), {meta['params'] / 1e6:.2f}M params, "
            f"widths {widths}; fp64 vs ONNX {check.get('onnx_max_abs', float('nan')):.2e}, "
            f"vs recorded {check['recorded_max_abs']:.2e}; "
            f"{quant} vs ONNX max {check['export_max_abs']:.4g} rms {check['export_rms']:.4g} "
            f"(action rms {check['action_rms']:.3g})"
        )

    _write_references(references, out_dir)
    return {"policies": result}


def _write_references(references, out_dir):
    """tools/traces/nn_reference.bin.gz: fixed inputs -> float64 outputs per net."""
    TRACES.mkdir(parents=True, exist_ok=True)
    arrays, nets = {}, {}
    for name, ref in references.items():
        arrays[f"{name}.input"] = ref["input"].astype(np.float32)
        arrays[f"{name}.expected"] = ref["expected"].astype(np.float64)
        if ref["onnx"] is not None:
            arrays[f"{name}.onnx"] = ref["onnx"].astype(np.float32)
        nets[name] = {
            "file": f"{name}.bin.gz",
            "rows": int(len(ref["input"])),
            "parity_rows": int(len(ref["input"]) - SYNTHETIC_INPUTS),
            "container_sha256": ref["sha256"],
        }
    meta = {
        "format": REFERENCE_FORMAT,
        "assets": str(Path(out_dir).resolve()),
        "nets": nets,
        "note": "expected = float64 forward of the exported container (int8 weights "
        "dequantized as q * scale); onnx = onnxruntime float32 on the same inputs. "
        "The last rows are synthetic inputs whose normalized values are N(0, 1).",
    }
    path = TRACES / "nn_reference.bin.gz"
    write_container(path, arrays, meta)
    print(f"[policies] reference outputs: {path}")
