"""Read and write STL triangle meshes, with the standard library only.

An STL file is already tessellated by the CAD program that exported
it, so importing one is reading the triangles and giving them to the
geometry as face elements. The format has no units, so the object
imports unit-less and the units pane asks for them, as for any other
source without units. The binary form also has no part names, so a
binary file becomes one element group named for the file. ASCII STL
can hold several ``solid <name>`` sections, and each becomes its own
named group, but CAD programs rarely write it; 3MF (`threemf`) is the
format that keeps an assembly's parts. STL is read because nearly
every CAD program can export it.

Vertices are repeated for each triangle in the file; identical
coordinates are merged on import so that triangles with a shared edge
share its nodes.
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
    """A Geometry of triangle face elements from [(name, (n, 3, 3) corner
    arrays)]: one named element group per part, with vertices merged where
    their coordinates are identical.

    `length_unit` declares the unit of the coordinates; without it the
    geometry is unit-less.
    """
    seen: dict[bytes, int] = {}
    xyz: list[np.ndarray] = []
    elem_conn, elem_group = [], []
    group_id, group_name = [], []
    for index, (name, corners) in enumerate(parts, start=1):
        group_id.append(index)
        group_name.append(name)
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
            elem_group.append(index)
    count = len(elem_conn)
    return Geometry(
        node_id=np.arange(1, len(xyz) + 1),
        node_xyz=np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        elem_id=np.arange(1, count + 1),
        elem_type=np.full(count, TRI3, dtype=np.int64),
        elem_conn=elem_conn, elem_group=np.asarray(elem_group),
        group_id=np.asarray(group_id), group_name=group_name,
        length_unit=length_unit)


def load(path: str | os.PathLike,
         length_unit: str | None = None) -> Geometry:
    """Read an STL file as a geometry of triangle face elements.

    Parameters
    ----------
    path : str or path-like
        The ``.stl`` file, binary or ASCII.
    length_unit : str, optional
        The unit of the file's coordinates, which STL cannot state.
        Without it the geometry imports unit-less and the units can be
        defined later.

    Returns
    -------
    Geometry
        The triangles as face elements: one element group per ASCII
        ``solid``, named as the file names it, or one group named for
        the file. A ``solid`` with no name is also named for the file.

    Raises
    ------
    ValueError
        When the file holds no triangles.
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
    """[(element group id, (n, 3, 3) corners)] from the face elements.

    Quads are split along a diagonal, higher-order faces are taken by
    their corner nodes, and line and volume elements are left out.
    """
    from ..core.geometry import ELEMENT_TYPES

    order = {int(i): position
             for position, i in enumerate(geometry.node_id)}
    by_block: dict[int, list[np.ndarray]] = {}
    for kind, conn, block in zip(geometry.elem_type, geometry.elem_conn,
                                 geometry.elem_group, strict=True):
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
    """Write the geometry's face elements as binary STL (32-bit floats).

    STL cannot name parts or declare units, so every element group is
    written into one solid. Without `unit_system` the coordinates are
    written as stored, which is meters when the geometry's units are
    defined. With one, defined coordinates are converted from meters to
    its length unit; undefined ones are written as they stand.
    A geometry with no face elements is refused with ValueError, since
    STL holds triangles and nothing else.
    """
    parts = faces_as_triangles(obj)
    triangles = (np.concatenate([corners for _block, corners in parts])
                 if parts else np.empty((0, 3, 3)))
    if not len(triangles):
        raise ValueError('this geometry has no face elements; '
                         'STL holds triangles and nothing else')
    # undefined coordinates are not meters and are written as they stand,
    # the rule `exporters.geometry_values` keeps for every other writer;
    # they were scaled as if they were meters (2026-10-13, found by the
    # docstring pass)
    if unit_system is not None and obj.units_defined:
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
