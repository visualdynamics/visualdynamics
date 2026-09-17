"""STL triangle meshes, read and written with the standard library.

An STL file is already tessellated — the CAD program did the hard work
at export — so importing one is reading triangles and handing them to
the geometry as face elements. The format carries **no units** (the
object imports unit-less and the units pane asks, like any other
undeclared source) and, in its binary form, **no part names**: a
binary file arrives as one block named for the file. ASCII STL can
hold several ``solid <name>`` sections, and each becomes its own
named block — but CAD tools rarely write it, and 3MF (`threemf`) is
the format that actually preserves an assembly's parts. This importer
exists because *everything* exports STL.

Vertices repeat per triangle in the file; identical coordinates are
merged on import so shared edges share nodes.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np

from ..core.geometry import Geometry

#: UFF 2412 descriptor for a three-node face (see geometry.ELEMENT_TYPES)
TRI3 = 41


def sniff(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith('.stl')


def _is_binary(raw: bytes) -> bool:
    """Binary unless it honestly parses as ASCII STL.

    The 80-byte binary header is arbitrary and may open with the word
    'solid', so the word alone proves nothing — the declared triangle
    count against the file size is the test that cannot lie.
    """
    if len(raw) >= 84:
        count = struct.unpack('<I', raw[80:84])[0]
        if len(raw) == 84 + 50 * count:
            return True
    return raw.lstrip()[:5].lower() != b'solid'


def _read_binary(raw: bytes) -> list[tuple[str, np.ndarray]]:
    count = struct.unpack('<I', raw[80:84])[0]
    records = np.frombuffer(raw[84:84 + 50 * count],
                            dtype=np.dtype([('normal', '<f4', 3),
                                            ('corners', '<f4', (3, 3)),
                                            ('attr', '<u2')]))
    return [('', records['corners'].astype(np.float64))]


def _read_ascii(raw: bytes) -> list[tuple[str, np.ndarray]]:
    solids: list[tuple[str, np.ndarray]] = []
    name, corners = '', []
    for line in raw.decode('ascii', errors='replace').splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == 'solid':
            name, corners = ' '.join(words[1:]), []
        elif words[0] == 'vertex':
            corners.append([float(w) for w in words[1:4]])
        elif words[0] == 'endsolid':
            triangles = np.asarray(corners, dtype=np.float64)
            solids.append((name, triangles.reshape(-1, 3, 3)))
    if not solids and corners:
        # a file cut short of its endsolid still holds its triangles
        solids.append((name, np.asarray(corners).reshape(-1, 3, 3)))
    return solids


def mesh_geometry(parts: list[tuple[str, np.ndarray]],
                  length_unit: str | None = None) -> Geometry:
    """A Geometry from [(name, (n, 3, 3) corner arrays)] — one named
    block per part, vertices merged where coordinates are identical."""
    seen: dict[bytes, int] = {}
    xyz: list[np.ndarray] = []
    elem_conn, elem_block = [], []
    block_id, block_name = [], []
    for index, (name, corners) in enumerate(parts, start=1):
        block_id.append(index)
        block_name.append(name)
        for triangle in corners:
            conn = []
            for corner in triangle:
                key = corner.tobytes()
                node = seen.get(key)
                if node is None:
                    node = seen[key] = len(xyz) + 1
                    xyz.append(corner)
                conn.append(node)
            elem_conn.append(np.asarray(conn, dtype=np.int64))
            elem_block.append(index)
    count = len(elem_conn)
    return Geometry(
        node_id=np.arange(1, len(xyz) + 1),
        node_xyz=np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        elem_id=np.arange(1, count + 1),
        elem_type=np.full(count, TRI3, dtype=np.int64),
        elem_conn=elem_conn, elem_block=np.asarray(elem_block),
        block_id=np.asarray(block_id), block_name=block_name,
        length_unit=length_unit)


def load(path: str | os.PathLike,
         length_unit: str | None = None) -> Geometry:
    """Read an STL file as a geometry of triangle face elements.

    Parameters
    ----------
    path : str or path-like
        The ``.stl`` file, binary or ASCII.
    length_unit : str, optional
        What the file's numbers are in — STL cannot say. Undeclared,
        the geometry imports unit-less and units are defined later.

    Returns
    -------
    Geometry
        Triangles as face elements; one block per ASCII ``solid``
        (named), or a single block named for the file.
    """
    with open(path, 'rb') as handle:
        raw = handle.read()
    parts = (_read_binary(raw) if _is_binary(raw) else _read_ascii(raw))
    if not parts or not sum(len(corners) for _n, corners in parts):
        raise ValueError(f'{path} holds no triangles')
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    parts = [(name or stem, corners) for name, corners in parts]
    return mesh_geometry(parts, length_unit)


def faces_as_triangles(geometry: Geometry) -> list[tuple[int, np.ndarray]]:
    """[(block id, (n, 3, 3) corners)] from the face elements — quads
    split along a diagonal, higher-order faces read by their corner
    nodes, lines and volumes left out."""
    from ..core.geometry import ELEMENT_TYPES

    order = {int(i): position
             for position, i in enumerate(geometry.node_id)}
    by_block: dict[int, list[np.ndarray]] = {}
    for kind, conn, block in zip(geometry.elem_type, geometry.elem_conn,
                                 geometry.elem_block, strict=True):
        name, _count, render = ELEMENT_TYPES.get(
            int(kind), ('?', 0, 'point'))
        if render != 'face':
            continue
        corners = [geometry.node_xyz[order[int(node)]]
                   for node in conn[:4 if name.startswith('quad') else 3]]
        triangles = ([np.asarray(corners[:3])] if len(corners) == 3 else
                     [np.asarray([corners[0], corners[1], corners[2]]),
                      np.asarray([corners[0], corners[2], corners[3]])])
        by_block.setdefault(int(block), []).extend(triangles)
    return [(block, np.asarray(triangles))
            for block, triangles in by_block.items()]


def handles(obj: Any) -> bool:
    return isinstance(obj, Geometry)


def save(obj: Geometry, path: str | os.PathLike,
         unit_system: Any = None) -> None:
    """Write the geometry's face elements as binary STL.

    STL cannot name parts or declare units: the blocks flatten into
    one solid, and the numbers are meters (the stored SI) unless a
    `unit_system` says to write its own length unit instead. A
    geometry with no face elements is refused — there is nothing an
    STL can hold of it.
    """
    parts = faces_as_triangles(obj)
    triangles = (np.concatenate([corners for _block, corners in parts])
                 if parts else np.empty((0, 3, 3)))
    if not len(triangles):
        raise ValueError('this geometry has no face elements; '
                         'STL holds triangles and nothing else')
    if unit_system is not None:
        from ..units import convert
        triangles = convert(triangles, 'm',
                            unit_system.coherent.unit('length'))
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                        where=lengths > 0)
    records = np.zeros(len(triangles),
                       dtype=np.dtype([('normal', '<f4', 3),
                                       ('corners', '<f4', (3, 3)),
                                       ('attr', '<u2')]))
    records['normal'] = normals
    records['corners'] = triangles
    with open(path, 'wb') as out:
        out.write(b'visualdynamics'.ljust(80, b' '))
        out.write(struct.pack('<I', len(records)))
        out.write(records.tobytes())
