"""Read and write sdynpy geometry files (.npz).

The format is a NumPy .npz archive of structured arrays (self-describing;
read with `plain_numpy`, because the connectivity columns are pickled):

- node: id, coordinate (3,), color, def_cs, disp_cs;
- coordinate_system: id, name, color, cs_type, matrix (4, 3), where
  rows 0-2 are the rotation and row 3 is the origin (a length);
- traceline: id, color, description, connectivity (an object array of
  node id arrays, where a 0 lifts the pen, so one row can hold several
  separate lines);
- element: id, type (the universal file dataset 2412 descriptor code),
  color, connectivity (an object array).

'node' and 'coordinate_system' are required; 'traceline' and 'element'
may be absent, as in a point cloud. Each node's coordinates are read in
the coordinate system it is defined in and converted to global. The
file has no units: `length_unit` declares them at import, and without
it the geometry is unit-less.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..units import UnitSystem

import numpy as np

from ..core.geometry import Geometry, to_global, to_local
from ..units import si_factor
from .plain_numpy import PlainNpz
from .sniffing import npz_has


def sniff(path: str | os.PathLike) -> bool:
    # the two a geometry cannot be without. Tracelines and elements are
    # each optional — a point cloud has neither, and asking for all
    # four turned a geometry that happened to lack one into a file no
    # reader would claim (Brandon, 2026-09-20). Together these two
    # names belong to no other sdynpy save.
    return npz_has(path, {'node', 'coordinate_system'})


def load(path: str | os.PathLike, length_unit: str | None = None) -> Geometry:
    """Read an sdynpy geometry file.

    The file has no units; without `length_unit` the geometry imports
    with the file's coordinates and the units can be defined later. The
    elements are split into one element group per element family, and
    each traceline id becomes one drawn line.
    """
    scale = 1.0 if length_unit is None else si_factor(length_unit, 'length')

    with PlainNpz(path) as d:
        node = d['node']
        cs = d['coordinate_system']
        tl = (d.with_objects('traceline') if 'traceline' in d
              else np.empty(0, TRACELINE_DTYPE))
        el = (d.with_objects('element') if 'element' in d
              else np.empty(0, ELEMENT_DTYPE))

    raw = cs['matrix'].astype(np.float64)
    matrix = raw.copy()
    matrix[:, 3, :] *= scale  # origin row carries length

    # a node is written in the frame it is placed in, and reading those
    # numbers as global puts it somewhere else entirely (Brandon,
    # 2026-09-20). Resolved in the file's own units, because an angle
    # is not a length and the scale must not touch it.
    geometry = Geometry(
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
        elem_id=el['id'],
        elem_type=el['type'],
        elem_color=el['color'],
        elem_conn=[np.asarray(c) for c in el['connectivity']],
        length_unit=length_unit,
    )
    # the layout carries no blocks, so every element arrived in one; a
    # block holds one family (2026-09-30). A traceline is a block of
    # two-node line elements with no properties, one block per id.
    geometry.split_groups_by_family()
    by_id: dict[int, tuple[str, int, list]] = {}
    for tl_id, color, desc, conn in zip(tl['id'], tl['color'],
                                        tl['description'], tl['connectivity']):
        _n, _c, runs = by_id.setdefault(int(tl_id), (str(desc), int(color), []))
        runs.extend(_runs([int(n) for n in conn]))
    geometry.attach_drawn_lines([line for line in by_id.values() if line[2]])
    return geometry


def _runs(conn: list[int]) -> list[list[int]]:
    """A traceline's connectivity split at its zeros, the layout's pen-up.

    One connectivity can hold several separate lines. Leading, trailing
    and repeated zeros make no run, and a run of one node draws nothing.
    """
    # read as a node id, the 0 failed the import of a real sdynpy
    # geometry with "unknown nodes [0]" (Brandon, 2026-11-12)
    runs: list[list[int]] = [[]]
    for n in conn:
        if n == 0:
            runs.append([])
        else:
            runs[-1].append(n)
    return [run for run in runs if len(run) > 1]


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
    """Write a geometry in the sdynpy .npz layout.

    Coordinates are written in the coherent form of `unit_system` when
    the geometry's units are defined, and as stored otherwise. The format
    records no units either way, so the reader has to be told what they
    are. Each node's coordinates are written in the coordinate system it
    is defined in, its `def_cs`, as `load` reads them. Drawn lines
    are written as tracelines, one row per line with a 0 between its
    runs, and not as elements. A drawn line through a node numbered 0
    cannot be written, because the layout reads that 0 as a break.
    """
    from .exporters import geometry_values

    points, matrices = geometry_values(geometry, unit_system)
    node = np.zeros(geometry.num_nodes, dtype=NODE_DTYPE)
    node['id'] = geometry.node_id
    # written back into each node's own definition frame: the global
    # numbers under a local def_cs were resolved a second time by `load`
    # and the node moved (2026-10-13, found by the docstring pass)
    node['coordinate'] = to_local(points, geometry.node_def_cs,
                                  geometry.cs_id, geometry.cs_type, matrices)
    node['color'] = geometry.node_color
    node['def_cs'] = geometry.node_def_cs
    node['disp_cs'] = geometry.node_disp_cs

    cs = np.zeros(len(geometry.cs_id), dtype=CS_DTYPE)
    cs['id'] = geometry.cs_id
    cs['name'] = [str(n)[:40] for n in geometry.cs_name]
    cs['color'] = 1
    cs['cs_type'] = geometry.cs_type
    cs['matrix'] = matrices

    # a drawn line goes out as the layout's traceline under its block's
    # id, its runs joined by the layout's 0 pen-up into one row, and not
    # as elements as well. One row per run (until 2026-11-12) gave
    # sdynpy several tracelines sharing one id.
    drawn = geometry.drawn_lines()
    if any(0 in chain for line in drawn for chain in line['chains']):
        raise ValueError('a drawn line runs through node 0, which the sdynpy '
                         'layout reads as a break in the line')
    traceline = np.zeros(len(drawn), dtype=TRACELINE_DTYPE)
    traceline['id'] = [line['group'] for line in drawn]
    traceline['color'] = [line['color'] for line in drawn]
    traceline['description'] = [str(line['name'])[:40] for line in drawn]
    traceline['connectivity'] = _connectivity(
        [[n for i, chain in enumerate(line['chains'])
          for n in ([0] if i else []) + chain] for line in drawn])
    drawn_blocks = {line['group'] for line in drawn}
    kept = [i for i, b in enumerate(geometry.elem_group)
            if int(b) not in drawn_blocks]

    element = np.zeros(len(kept), dtype=ELEMENT_DTYPE)
    element['id'] = geometry.elem_id[kept]
    element['type'] = geometry.elem_type[kept]
    element['color'] = geometry.elem_color[kept]
    element['connectivity'] = _connectivity(
        [geometry.elem_conn[i] for i in kept])

    np.savez(path, node=node, coordinate_system=cs, traceline=traceline,
             element=element)
