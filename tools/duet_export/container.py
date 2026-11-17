"""DUETBIN1: the one binary container every demo asset uses (Python side).

Layout of the *uncompressed* bytes (all little-endian):

    0   8   magic b"DUETBIN1"
    8   4   uint32 H: byte length of the UTF-8 JSON header
    12  H   header {"meta": {...}, "arrays": {name: {"dtype", "shape", "offset", "nbytes"}}}
    ..      zero padding so the data section starts at a multiple of 16
    D   ..  array data; every array starts at D + offset, offset a multiple of 16

The file on disk is that byte string gzip-compressed (mtime 0, so the output is
reproducible). The JavaScript reader is demo/js/lib/container.js; both sides
must stay in step.
"""

import gzip
import io
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"DUETBIN1"
ALIGN = 16
DTYPES = ("float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32")


def _pad(n, align=ALIGN):
    return (-n) % align


def pack(arrays, meta=None):
    """The uncompressed container bytes for {name: array} and a JSON-able meta."""
    index, chunks, offset = {}, [], 0
    for name, value in arrays.items():
        a = np.ascontiguousarray(value)
        if a.dtype.name not in DTYPES:
            raise ValueError(f"{name}: dtype {a.dtype} is not one of {DTYPES}")
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        index[name] = {
            "dtype": a.dtype.name,
            "shape": list(a.shape),
            "offset": offset,
            "nbytes": int(a.nbytes),
        }
        raw = a.tobytes()
        chunks.append(raw + b"\0" * _pad(len(raw)))
        offset += len(raw) + _pad(len(raw))
    header = json.dumps(
        {"meta": meta or {}, "arrays": index}, separators=(",", ":")
    ).encode()
    head = MAGIC + struct.pack("<I", len(header)) + header
    head += b"\0" * _pad(len(head))
    return head + b"".join(chunks)


def unpack(data):
    """(meta, {name: array}) from uncompressed container bytes."""
    if data[:8] != MAGIC:
        raise ValueError("Not a DUETBIN1 container")
    (length,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12 : 12 + length].decode())
    start = 12 + length
    start += _pad(start)
    arrays = {}
    for name, entry in header["arrays"].items():
        begin = start + entry["offset"]
        raw = data[begin : begin + entry["nbytes"]]
        arrays[name] = np.frombuffer(raw, dtype="<" + np.dtype(entry["dtype"]).str[1:])
        arrays[name] = arrays[name].reshape(entry["shape"]).copy()
    return header["meta"], arrays


def write_container(path, arrays, meta=None, level=9):
    """Write a gzip-compressed container; returns {"bytes", "raw_bytes"}."""
    raw = pack(arrays, meta)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=level, mtime=0) as f:
        f.write(raw)
    Path(path).write_bytes(buffer.getvalue())
    return {"bytes": len(buffer.getvalue()), "raw_bytes": len(raw)}


def read_container(path):
    """(meta, {name: array}) from a gzip-compressed (or plain) container file."""
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return unpack(data)


def write_gzip(path, raw, level=9):
    """Gzip arbitrary bytes reproducibly (mtime 0); returns the compressed size."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=level, mtime=0) as f:
        f.write(raw)
    Path(path).write_bytes(buffer.getvalue())
    return len(buffer.getvalue())
