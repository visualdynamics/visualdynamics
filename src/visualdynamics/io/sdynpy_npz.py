"""Importer for sdynpy native geometry files (.npz).

The format is a numpy .npz archive with four structured arrays. Field layout
(np.lib.format is self-describing; read with np.load):

- node: id, coordinate (3,), color, def_cs, disp_cs
- coordinate_system: id, name, color, cs_type, matrix (4, 3) — rows 0-2 are
  the rotation, row 3 the origin (length units)
- traceline: id, color, description, connectivity (object: node id array)
- element: id, type (UFF 2412 descriptor code), color, connectivity (object)

The file carries no units. `length_unit` may be given to declare them at
import; otherwise the geometry arrives unit-less.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..units import UnitSystem

import numpy as np

from ..core.geometry import Geometry, to_global
from ..units import si_factor
from .sniffing import npz_has


def sniff(path: str | os.PathLike) -> bool:
    # the two a geometry cannot be without. Tracelines and elements are
    # each optional — a point cloud has neither, and asking for all
    # four turned a geometry that happened to lack one into a file no
    # reader would claim (Brandon, 2026-09-20). Together these two
    # names belong to no other sdynpy save.
    return npz_has(path, {'node', 'coordinate_system'})


def load(path: str | os.PathLike, length_unit: str | None = None) -> Geometry:
    """sdynpy geometry files carry no units; without `length_unit` the
    geometry imports with its raw coordinates for the user to define later."""
    scale = 1.0 if length_unit is None else si_factor(length_unit, 'length')

    with np.load(path, allow_pickle=True) as d:
        node = d['node']
        cs = d['coordinate_system']
        tl = d['traceline'] if 'traceline' in d else np.empty(0, TRACELINE_DTYPE)
        el = d['element'] if 'element' in d else np.empty(0, ELEMENT_DTYPE)

    raw = cs['matrix'].astype(np.float64)
    matrix = raw.copy()
    matrix[:, 3, :] *= scale  # origin row carries length

    # a node is written in the frame it is placed in, and reading those
    # numbers as global puts it somewhere else entirely (Brandon,
    # 2026-09-20). Resolved in the file's own units, because an angle
    # is not a length and the scale must not touch it.
    return Geometry(
        node_id=node['id'],
        node_xyz=to_global(node['coordinate'].astype(np.float64),
                           node['def_cs'], cs['id'], cs['cs_type'],
                           raw) * scale,
        node_def_cs=node['def_cs'],
        node_disp_cs=node['disp_cs'],
        node_color=node['color'],
        cs_id=cs['id'],
        cs_name=[str(n) for n in cs['name']],
        cs_type=cs['cs_type'],
        cs_matrix=matrix,
        traceline_id=tl['id'],
        traceline_color=tl['color'],
        traceline_desc=[str(s) for s in tl['description']],
        traceline_conn=[np.asarray(c) for c in tl['connectivity']],
        elem_id=el['id'],
        elem_type=el['type'],
        elem_color=el['color'],
        elem_conn=[np.asarray(c) for c in el['connectivity']],
        length_unit=length_unit,
    )


def handles(obj: Any) -> bool:
    return isinstance(obj, Geometry)


NODE_DTYPE = [('id', '<u8'), ('coordinate', '<f8', (3,)), ('color', '<u2'),
              ('def_cs', '<u8'), ('disp_cs', '<u8')]
CS_DTYPE = [('id', '<u8'), ('name', '<U40'), ('color', '<u2'),
            ('cs_type', '<u2'), ('matrix', '<f8', (4, 3))]
TRACELINE_DTYPE = [('id', '<u8'), ('color', '<u2'), ('description', '<U40'),
                   ('connectivity', 'O')]
ELEMENT_DTYPE = [('id', '<u8'), ('type', 'u1'), ('color', '<u2'),
                 ('connectivity', 'O')]


def _connectivity(rows):
    """Object array of per-entity node-id arrays, as sdynpy stores them."""
    out = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        out[i] = np.asarray(row, dtype=np.uint64)
    return out


def save(geometry: Geometry, path: str | os.PathLike, unit_system: UnitSystem | None = None) -> None:
    """Write a geometry in sdynpy's own .npz layout.

    Coordinates go out in `unit_system`, or as stored without one. The
    format records no units either way, so whoever reads it has to be told
    — which is what "consistent units" means in this corner of the world.
    """
    from .exporters import geometry_values

    points, matrices = geometry_values(geometry, unit_system)
    node = np.zeros(geometry.num_nodes, dtype=NODE_DTYPE)
    node['id'] = geometry.node_id
    node['coordinate'] = points
    node['color'] = geometry.node_color
    node['def_cs'] = geometry.node_def_cs
    node['disp_cs'] = geometry.node_disp_cs

    cs = np.zeros(len(geometry.cs_id), dtype=CS_DTYPE)
    cs['id'] = geometry.cs_id
    cs['name'] = [str(n)[:40] for n in geometry.cs_name]
    cs['color'] = 1
    cs['cs_type'] = geometry.cs_type
    cs['matrix'] = matrices

    traceline = np.zeros(len(geometry.traceline_conn), dtype=TRACELINE_DTYPE)
    traceline['id'] = geometry.traceline_id
    traceline['color'] = geometry.traceline_color
    traceline['description'] = [str(s)[:40] for s in geometry.traceline_desc]
    traceline['connectivity'] = _connectivity(geometry.traceline_conn)

    element = np.zeros(len(geometry.elem_conn), dtype=ELEMENT_DTYPE)
    element['id'] = geometry.elem_id
    element['type'] = geometry.elem_type
    element['color'] = geometry.elem_color
    element['connectivity'] = _connectivity(geometry.elem_conn)

    np.savez(path, node=node, coordinate_system=cs, traceline=traceline,
             element=element)
