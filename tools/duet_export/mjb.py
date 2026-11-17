"""Read and write MuJoCo binary models (.mjb) as plain numpy arrays.

The layout is MuJoCo's own (engine_io.c, mj_saveModel / mj_loadModelBuffer):

    5 int32      header {54321, sizeof(mjtNum) = 8, nsize, mj_version, npointers}
    nsize int64  the MJMODEL_SIZES fields, in order (nbuffer is the last one)
    raw structs  mjOption, mjVisual, mjStatistic, then the flg_gravcomp and
                 flg_surfacevel bytes (kept here as one opaque byte string)
    arrays       every MJMODEL_POINTERS field as raw bytes, sizeof(type) * nr * nc,
                 in order, with no padding

The loader rejects trailing bytes, requires the stored nbuffer to equal the one
mj_makeModel computes (every array 64-byte aligned from offset 0) and checks each
adr/num reference against its target size (mj_validateReferences).

Field order, element types and column counts are parsed from the mjxmacro.h that
ships with the installed mujoco wheel, so a file is read and written in the
layout of the MuJoCo version that will load it (3.11.0 for this demo; the .mjb
format is version-locked).
"""

import functools
import os
import re
from dataclasses import dataclass, field

import numpy as np

HEADER_ID = 54321
NHEADER = 5
ALIGN = 64  # mj_makeModel aligns every array of the model buffer to 64 bytes

# C element type of an X-macro field -> little-endian numpy dtype
DTYPES = {
    "int": np.dtype("<i4"),
    "mjtNum": np.dtype("<f8"),
    "float": np.dtype("<f4"),
    "mjtSize": np.dtype("<i8"),
    "mjtByte": np.dtype("u1"),
    "mjtBool": np.dtype("u1"),
    "char": np.dtype("u1"),
}


@dataclass(frozen=True)
class Field:
    ctype: str  # C element type ("int", "mjtNum", ...)
    name: str  # mjModel member name
    nr: str  # size member giving the number of rows
    nc: str  # column expression as written in mjxmacro.h ("3", "mjNREF", "MJ_M(nq)")

    @property
    def dtype(self):
        return DTYPES[self.ctype]


@dataclass(frozen=True)
class Layout:
    version: str  # mujoco.__version__ the layout was parsed for
    mj_version: int  # mj_version(), the header's 4th int
    sizes: tuple  # MJMODEL_SIZES member names, in file order
    fields: tuple  # MJMODEL_POINTERS as Field, in file order
    constants: dict = field(repr=False)  # mjN* constants used by column expressions

    def ncols(self, nc, sizes):
        """Evaluate a column expression ("3", "mjNREF", "MJ_M(nmocap)*3")."""
        total = 1
        for term in nc.replace(" ", "").split("*"):
            macro = re.fullmatch(r"MJ_M\((\w+)\)", term)
            if term.isdigit():
                total *= int(term)
            elif macro:
                total *= int(sizes[macro.group(1)])
            elif term in self.constants:
                total *= self.constants[term]
            elif term in sizes:
                total *= int(sizes[term])
            else:
                raise ValueError(f"Unknown term {term!r} in column expression {nc!r}")
        return total

    def shape(self, f, sizes):
        return (int(sizes[f.nr]), self.ncols(f.nc, sizes))

    def nbuffer(self, sizes):
        """mjModel.nbuffer as mj_makeModel computes it for these sizes."""
        offset = 0
        for f in self.fields:
            rows, cols = self.shape(f, sizes)
            offset += (-offset) % ALIGN + f.dtype.itemsize * rows * cols
        return offset

    def arrays_nbytes(self, sizes):
        total = 0
        for f in self.fields:
            rows, cols = self.shape(f, sizes)
            total += f.dtype.itemsize * rows * cols
        return total


def _macro_bodies(text):
    """{name: body} of every #define, line continuations joined."""
    bodies, lines, i = {}, text.split("\n"), 0
    while i < len(lines):
        match = re.match(r"\s*#define\s+(\w+)(\([^)]*\))?\s*(.*)$", lines[i])
        if match:
            parts = [match.group(3)]
            while parts[-1].rstrip().endswith("\\"):
                parts[-1] = parts[-1].rstrip()[:-1]
                i += 1
                parts.append(lines[i])
            bodies[match.group(1)] = " ".join(parts)
        i += 1
    return bodies


def _x_entries(bodies, name):
    """Argument lists of every X(...)/XNV(...) in a macro, nested macros expanded."""
    out, body, pos = [], re.sub(r"/\*.*?\*/", " ", bodies[name]), 0
    token = re.compile(r"\b(XNV|X|MJMODEL_\w+)\b")
    while True:
        match = token.search(body, pos)
        if not match:
            return out
        if match.group(1).startswith("MJMODEL_"):
            out += _x_entries(bodies, match.group(1))
            pos = match.end()
            continue
        start = body.index("(", match.end())
        depth, end = 0, start
        while True:
            depth += {"(": 1, ")": -1}.get(body[end], 0)
            if depth == 0:
                break
            end += 1
        out.append([arg.strip() for arg in body[start + 1 : end].split(",")])
        pos = end + 1


def _constant(name, defines):
    import mujoco

    if name in defines:
        return defines[name]
    value = getattr(mujoco, name, None)
    if value is not None:
        return int(value)
    for attr in dir(mujoco):  # enum members such as mjtTextureRole.mjNTEXROLE
        enum = getattr(mujoco, attr)
        if attr.startswith("mjt") and hasattr(enum, name):
            return int(getattr(enum, name))
    raise ValueError(f"Cannot resolve MuJoCo constant {name}")


@functools.lru_cache(maxsize=None)
def layout():
    """The .mjb layout of the installed mujoco wheel (parsed from its headers)."""
    import mujoco

    include = os.path.join(os.path.dirname(mujoco.__file__), "include", "mujoco")
    with open(os.path.join(include, "mjxmacro.h")) as f:
        bodies = _macro_bodies(f.read())
    defines = {}
    for header in ("mjmodel.h", "mjtype.h"):
        with open(os.path.join(include, header)) as f:
            for match in re.finditer(r"#define\s+(mjN\w+)\s+(\d+)", f.read()):
                defines[match.group(1)] = int(match.group(2))
    sizes = tuple(args[0] for args in _x_entries(bodies, "MJMODEL_SIZES"))
    fields = tuple(Field(*args) for args in _x_entries(bodies, "MJMODEL_POINTERS"))
    if sizes[-1] != "nbuffer":
        raise ValueError("MJMODEL_SIZES must end with nbuffer")
    for f in fields:
        if f.ctype not in DTYPES or f.nr not in sizes:
            raise ValueError(f"Unsupported field {f}")
    names = {f.nc for f in fields}
    constants = {}
    for nc in names:
        for term in nc.replace(" ", "").split("*"):
            if re.fullmatch(r"mjN\w+", term):
                constants[term] = _constant(term, defines)
    return Layout(mujoco.__version__, int(mujoco.mj_version()), sizes, fields, constants)


@dataclass
class Mjb:
    """A parsed .mjb: header ints, sizes, the opaque struct block and every array.

    `arrays[name]` has shape (nr, nc) and the field's C dtype; `sizes` is ordered
    as MJMODEL_SIZES. serialize() recomputes nbuffer, so code that edits sizes
    and arrays only has to keep them consistent with each other.
    """

    header: tuple
    sizes: dict
    structs: bytes
    arrays: dict

    def copy(self):
        return Mjb(
            self.header,
            dict(self.sizes),
            self.structs,
            {k: v.copy() for k, v in self.arrays.items()},
        )


def parse(data, lay=None):
    """An Mjb from .mjb bytes; checks the header against the installed MuJoCo."""
    lay = lay or layout()
    data = memoryview(data).cast("B")
    header = tuple(int(x) for x in np.frombuffer(data, "<i4", NHEADER))
    expected = (HEADER_ID, 8, len(lay.sizes), lay.mj_version, len(lay.fields))
    if header != expected:
        raise ValueError(f"mjb header {header} does not match this MuJoCo {expected}")
    offset = 4 * NHEADER
    values = np.frombuffer(data, "<i8", len(lay.sizes), offset)
    sizes = {name: int(v) for name, v in zip(lay.sizes, values, strict=True)}
    offset += 8 * len(lay.sizes)
    struct_bytes = len(data) - offset - lay.arrays_nbytes(sizes)
    if struct_bytes <= 0:
        raise ValueError("mjb file is truncated")
    structs = bytes(data[offset : offset + struct_bytes])
    offset += struct_bytes
    arrays = {}
    for f in lay.fields:
        rows, cols = lay.shape(f, sizes)
        a = np.frombuffer(data, f.dtype, rows * cols, offset)
        arrays[f.name] = a.reshape(rows, cols).copy()
        offset += a.nbytes
    if offset != len(data):
        raise ValueError("mjb arrays do not end at the end of the file")
    if lay.nbuffer(sizes) != sizes["nbuffer"]:
        raise ValueError("mjb nbuffer differs from the one mj_makeModel computes")
    return Mjb(header, sizes, structs, arrays)


def serialize(mjb, lay=None):
    """.mjb bytes of an Mjb; nbuffer is recomputed, every array shape checked."""
    lay = lay or layout()
    sizes = dict(mjb.sizes)
    if list(sizes) != list(lay.sizes):
        raise ValueError("sizes must list every MJMODEL_SIZES field in order")
    sizes["nbuffer"] = lay.nbuffer(sizes)
    parts = [
        np.asarray(mjb.header, "<i4").tobytes(),
        np.asarray([sizes[k] for k in lay.sizes], "<i8").tobytes(),
        mjb.structs,
    ]
    for f in lay.fields:
        a = np.asarray(mjb.arrays[f.name])
        shape = lay.shape(f, sizes)
        if a.dtype != f.dtype or a.size != shape[0] * shape[1]:
            raise ValueError(
                f"{f.name}: {a.dtype} {a.shape} does not match {f.dtype} {shape}"
            )
        parts.append(np.ascontiguousarray(a).tobytes())
    return b"".join(parts)


def read(path):
    with open(path, "rb") as f:
        return parse(f.read())
