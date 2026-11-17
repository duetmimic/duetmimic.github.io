"""A minimal ONNX reader: protobuf wire format only, no `onnx` package needed.

The osl-deploy environment has onnxruntime but not onnx, and the exporter only
needs to *read* two small MLP graphs. This module decodes the subset of
`ModelProto` those graphs use:

- ModelProto: ir_version, producer, opset_import, graph
- GraphProto: node, initializer, input, output (value_info is skipped)
- NodeProto: op_type, name, domain, input, output, attribute
- AttributeProto: f, i, s, t, floats, ints, strings (graphs and type protos
  are rejected: an MLP has no control flow)
- TensorProto: dims, data_type and its data as raw_data or as the typed
  repeated fields (float_data, int32_data, int64_data, double_data,
  uint64_data); external data is rejected
- ValueInfoProto: name, element type, shape (dim_value or dim_param)

Field numbers follow onnx/onnx.proto (IR version 3 onwards). Repeated scalars
are accepted both packed and unpacked, as protobuf requires of a reader.

    model = load("actor.onnx")
    model.graph.nodes[0].op_type, model.graph.initializers["w"]  # numpy array
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Protobuf wire types.
VARINT, I64, LEN, I32 = 0, 1, 2, 5

# TensorProto.DataType -> numpy dtype (little-endian when stored as raw bytes).
DTYPES = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}

# AttributeProto.AttributeType values this reader decodes.
ATTR_FLOAT, ATTR_INT, ATTR_STRING, ATTR_TENSOR = 1, 2, 3, 4
ATTR_FLOATS, ATTR_INTS, ATTR_STRINGS = 6, 7, 8


@dataclass
class Tensor:
    name: str
    array: np.ndarray


@dataclass
class ValueInfo:
    name: str
    elem_type: int
    shape: list  # int for a fixed dim, str for a symbolic one, None if unknown


@dataclass
class Node:
    op_type: str
    name: str
    domain: str
    inputs: list
    outputs: list
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    name: str
    nodes: list
    initializers: dict  # name -> np.ndarray
    inputs: list  # ValueInfo, initializers excluded
    outputs: list  # ValueInfo


@dataclass
class Model:
    ir_version: int
    producer: str
    opset: dict  # domain ("" = ai.onnx) -> version
    graph: Graph


# --- wire format ------------------------------------------------------------


def _varint(buf, pos):
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("Varint longer than 10 bytes")


def _signed(value):
    """A 64-bit two's complement varint as a Python int (int32/int64 fields)."""
    return value - (1 << 64) if value >= 1 << 63 else value


def _fields(buf):
    """(field number, wire type, value) of each field of one message.

    The value is an int for a varint, a memoryview for a length-delimited,
    32-bit or 64-bit field. Nothing is copied.
    """
    buf = memoryview(buf)
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == VARINT:
            value, pos = _varint(buf, pos)
        elif wire == LEN:
            size, pos = _varint(buf, pos)
            value = buf[pos : pos + size]
            pos += size
        elif wire == I64:
            value = buf[pos : pos + 8]
            pos += 8
        elif wire == I32:
            value = buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire} (field {number})")
        if pos > end:
            raise ValueError(f"Truncated protobuf field {number}")
        yield number, wire, value


def _text(value):
    return bytes(value).decode("utf-8")


def _varints(wire, value):
    """A repeated varint field's values, packed (LEN) or one unpacked entry."""
    if wire == VARINT:
        return [_signed(value)]
    out, pos = [], 0
    while pos < len(value):
        item, pos = _varint(value, pos)
        out.append(_signed(item))
    return out


def _fixed(wire, value, dtype):
    """A repeated fixed-width field (float, double), packed or unpacked."""
    return np.frombuffer(bytes(value), dtype=np.dtype(dtype).newbyteorder("<"))


# --- messages ---------------------------------------------------------------


def parse_tensor(buf):
    dims, data_type, name, raw = [], 0, "", None
    typed = {"float": [], "int32": [], "int64": [], "double": [], "uint64": []}
    for number, wire, value in _fields(buf):
        if number == 1:
            dims.extend(_varints(wire, value))
        elif number == 2:
            data_type = value
        elif number == 4:
            typed["float"].append(_fixed(wire, value, "<f4"))
        elif number == 5:
            typed["int32"].extend(_varints(wire, value))
        elif number == 7:
            typed["int64"].extend(_varints(wire, value))
        elif number == 8:
            name = _text(value)
        elif number == 9:
            raw = value
        elif number == 10:
            typed["double"].append(_fixed(wire, value, "<f8"))
        elif number == 11:
            typed["uint64"].extend(v & ((1 << 64) - 1) for v in _varints(wire, value))
        elif number == 13 or (number == 14 and value == 1):
            raise ValueError(f"Tensor {name or '?'}: external data is not supported")
        elif number == 6:
            raise ValueError(f"Tensor {name or '?'}: string tensors are not supported")
    if data_type not in DTYPES:
        raise ValueError(f"Tensor {name}: unsupported data type {data_type}")
    dtype = np.dtype(DTYPES[data_type])
    count = int(np.prod(dims)) if dims else 1
    if raw is not None:
        array = np.frombuffer(bytes(raw), dtype=dtype.newbyteorder("<"))
    elif data_type == 1:
        array = np.concatenate(typed["float"]) if typed["float"] else np.zeros(0)
    elif data_type == 11:
        array = np.concatenate(typed["double"]) if typed["double"] else np.zeros(0)
    elif data_type == 7:
        array = np.asarray(typed["int64"], dtype=np.int64)
    elif data_type in (12, 13):
        array = np.asarray(typed["uint64"], dtype=np.uint64)
    elif data_type == 10:  # float16 bits travel in int32_data
        array = np.asarray(typed["int32"], dtype=np.uint16).view(np.float16)
    else:  # int8, uint8, int16, uint16, int32, bool travel in int32_data
        array = np.asarray(typed["int32"], dtype=np.int64)
    if array.size != count:
        raise ValueError(f"Tensor {name}: {array.size} values for dims {dims}")
    return Tensor(name, array.astype(dtype, copy=False).reshape(dims).copy())


def parse_attribute(buf):
    name, kind, values = "", 0, {}
    floats, ints, strings = [], [], []
    for number, wire, value in _fields(buf):
        if number == 1:
            name = _text(value)
        elif number == 20:
            kind = value
        elif number == 2:
            values[ATTR_FLOAT] = struct.unpack("<f", value)[0]
        elif number == 3:
            values[ATTR_INT] = _signed(value)
        elif number == 4:
            values[ATTR_STRING] = bytes(value)
        elif number == 5:
            values[ATTR_TENSOR] = parse_tensor(value).array
        elif number == 7:
            floats.extend(
                [struct.unpack("<f", value)[0]]
                if wire == I32
                else _fixed(wire, value, "<f4").tolist()
            )
        elif number == 8:
            ints.extend(_varints(wire, value))
        elif number == 9:
            strings.append(bytes(value))
        elif number in (6, 10, 11, 14, 15, 22, 23):
            raise ValueError(
                f"Attribute {name or '?'}: graph/tensor-list attributes are not supported"
            )
    values[ATTR_FLOATS], values[ATTR_INTS], values[ATTR_STRINGS] = floats, ints, strings
    if kind == 0:  # IR < 3 writers may omit the type: infer it from the content
        kind = next((k for k in (ATTR_FLOAT, ATTR_INT, ATTR_STRING, ATTR_TENSOR) if k in values), 0)
        kind = kind or next((k for k in (ATTR_FLOATS, ATTR_INTS, ATTR_STRINGS) if values[k]), 0)
    if kind not in values:
        raise ValueError(f"Attribute {name}: unsupported attribute type {kind}")
    return name, values[kind]


def parse_node(buf):
    node = Node(op_type="", name="", domain="", inputs=[], outputs=[])
    for number, _, value in _fields(buf):
        if number == 1:
            node.inputs.append(_text(value))
        elif number == 2:
            node.outputs.append(_text(value))
        elif number == 3:
            node.name = _text(value)
        elif number == 4:
            node.op_type = _text(value)
        elif number == 7:
            node.domain = _text(value)
        elif number == 5:
            key, attr = parse_attribute(value)
            node.attrs[key] = attr
    return node


def parse_value_info(buf):
    name, elem_type, shape = "", 0, []
    for number, _, value in _fields(buf):
        if number == 1:
            name = _text(value)
        elif number == 2:  # TypeProto
            for n_type, _, v_type in _fields(value):
                if n_type != 1:  # only tensor_type
                    continue
                for n_tensor, _, v_tensor in _fields(v_type):
                    if n_tensor == 1:
                        elem_type = v_tensor
                    elif n_tensor == 2:  # TensorShapeProto
                        for n_dim, _, v_dim in _fields(v_tensor):
                            if n_dim != 1:
                                continue
                            dim = None
                            for n_d, _, v_d in _fields(v_dim):
                                if n_d == 1:
                                    dim = _signed(v_d)
                                elif n_d == 2:
                                    dim = _text(v_d)
                            shape.append(dim)
    return ValueInfo(name, elem_type, shape)


def parse_graph(buf):
    name, nodes, initializers, inputs, outputs = "", [], {}, [], []
    for number, _, value in _fields(buf):
        if number == 1:
            nodes.append(parse_node(value))
        elif number == 2:
            name = _text(value)
        elif number == 5:
            tensor = parse_tensor(value)
            initializers[tensor.name] = tensor.array
        elif number == 11:
            inputs.append(parse_value_info(value))
        elif number == 12:
            outputs.append(parse_value_info(value))
        elif number == 15:
            raise ValueError("Sparse initializers are not supported")
    # IR < 4 lists initializers among the graph inputs; keep the real inputs only.
    inputs = [info for info in inputs if info.name not in initializers]
    return Graph(name, nodes, initializers, inputs, outputs)


def parse_model(buf):
    ir_version, producer, opset, graph = 0, "", {}, None
    for number, _, value in _fields(buf):
        if number == 1:
            ir_version = value
        elif number == 2:
            producer = _text(value)
        elif number == 7:
            graph = parse_graph(value)
        elif number == 8:
            domain, version = "", 1
            for n_op, _, v_op in _fields(value):
                if n_op == 1:
                    domain = _text(v_op)
                elif n_op == 2:
                    version = _signed(v_op)
            opset[domain] = version
    if graph is None:
        raise ValueError("Not an ONNX model: no graph")
    return Model(ir_version, producer, opset, graph)


def load(path):
    """The Model of an .onnx file."""
    return parse_model(Path(path).read_bytes())
