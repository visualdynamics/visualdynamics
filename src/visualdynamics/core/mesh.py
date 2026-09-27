"""Plate meshes built from planes.

A structure made of flat plates — a box, a channel, a bracket — is
described most simply as the planes it is made of: each a rectangle at
its mid-thickness, meshed into rectangular plate elements and put in a
block of its own. `plane` makes one; `assemble` joins them into one
geometry and merges the nodes they share, so the plates are tied along
the lines where their mid-surfaces meet (Brandon, 2026-09-26, for the
BARC example). Where mid-surfaces do not meet — a bolted foot sitting on
a wall — nothing is shared, and the join is a rigid link
(`fem.RIGID`) between a node on each.

Each block is then given a material and a thickness (`fem.BlockProperties`,
or the Blocks table) and `fem.Model.from_geometry` builds the model.

Nodes on a shared line coincide because both planes divide it the
same way: give planes that meet the same element size.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Geometry

__all__ = ['assemble', 'plane']


def plane(corner: ArrayLike, edge_a: ArrayLike, edge_b: ArrayLike,
          size: float, name: str = '', *, unit: str | None = 'm') -> Geometry:
    """A rectangle meshed into four-node plate elements, in one block.

    Each edge is divided evenly into the whole number of elements that
    comes nearest `size`, so the elements come out close to square and
    close to that size (Brandon, 2026-09-26: even spacing, roughly square
    elements, and nothing to lay out by hand). Planes that meet along a
    line divide it the same way, since it is the same length in both, so
    `assemble` ties them there.

    Parameters
    ----------
    corner : array_like
        One corner, (x, y, z).
    edge_a, edge_b : array_like
        The two edges from that corner, as vectors: their lengths are the
        rectangle's sides. They must be perpendicular — the plate element
        is a rectangle.
    size : float
        The element size aimed at.
    name : str, optional
        The block's name — the part this plane is.
    unit : str or None, default 'm'
        The unit every length given here is in. The geometry holds them
        in SI, as any geometry with its units defined does, and remembers
        this one; None leaves them as given, units undefined.

    Returns
    -------
    Geometry
        Nodes numbered from 1, one block holding every element.
    """
    from ..units import si_transform

    scale = 1.0 if unit is None else si_transform(unit, 'length')[0]
    corner = np.asarray(corner, dtype=float)
    edge_a = np.asarray(edge_a, dtype=float)
    edge_b = np.asarray(edge_b, dtype=float)
    length_a, length_b = np.linalg.norm(edge_a), np.linalg.norm(edge_b)
    if not length_a > 0 or not length_b > 0:
        raise ValueError(f'{name or "a plane"}: an edge has no length')
    if not size or size <= 0:
        raise ValueError(f'{name or "a plane"}: the element size must be '
                         'positive')
    if abs(float(edge_a @ edge_b)) > 1e-9 * length_a * length_b:
        raise ValueError(f'{name or "a plane"}: the edges are not '
                         'perpendicular, and the plate element is a '
                         'rectangle')
    count_a = max(1, round(length_a / size)) + 1
    count_b = max(1, round(length_b / size)) + 1
    grid_a = np.linspace(0.0, length_a, count_a)
    grid_b = np.linspace(0.0, length_b, count_b)
    unit_a, unit_b = edge_a / length_a, edge_b / length_b
    xyz = np.array([corner + a * unit_a + b * unit_b
                    for b in grid_b for a in grid_a])
    node = np.arange(1, len(xyz) + 1)
    conn = []
    for j in range(count_b - 1):
        for i in range(count_a - 1):
            first = j * count_a + i
            conn.append(node[[first, first + 1, first + count_a + 1,
                              first + count_a]])
    return Geometry(node_id=node, node_xyz=xyz * scale, elem_conn=conn,
                    elem_type=[44] * len(conn), elem_block=[1] * len(conn),
                    block_id=[1], block_name=[name], length_unit=unit)


def assemble(*parts: Geometry, tolerance: float | None = None) -> Geometry:
    """One geometry from several, their nodes renumbered in turn, and
    coincident nodes merged — so planes meeting along a line are tied
    there. Blocks of the same name are one block: the five planes of a
    box, each named 'box', are the box — one part, given its material
    once. Unnamed blocks stay apart.

    Parameters
    ----------
    *parts : Geometry
        The pieces, in the same length unit.
    tolerance : float, optional
        How close nodes must be to be one, in meters (as the geometry
        holds them). Defaults to a millionth of the assembly's size.

    Returns
    -------
    Geometry
    """
    if not parts:
        raise ValueError('assembling takes at least one part')
    units = {part.length_unit for part in parts}
    if len(units) > 1:
        raise ValueError('the parts are in different length units: '
                         + ', '.join(sorted(str(u) for u in units)))
    node_ids, xyz, conn, types, blocks, block_ids, block_names = (
        [], [], [], [], [], [], [])
    properties = {}
    offset, next_block = 0, 1
    for part in parts:
        renumber = {int(n): offset + k + 1 for k, n in enumerate(part.node_id)}
        node_ids.extend(renumber.values())
        xyz.append(part.node_xyz)
        block_map = {}
        for k, block in enumerate(part.block_id):
            name = part.block_name[k]
            if name and name in block_names:
                block_map[int(block)] = block_ids[block_names.index(name)]
                continue
            block_map[int(block)] = next_block
            block_ids.append(next_block)
            block_names.append(name)
            if int(block) in part.block_properties:
                properties[next_block] = part.block_properties[int(block)]
            next_block += 1
        for k, element in enumerate(part.elem_conn):
            conn.append(np.asarray([renumber[int(n)] for n in element]))
            types.append(int(part.elem_type[k]))
            blocks.append(block_map[int(part.elem_block[k])])
        offset += len(part.node_id)
    whole = Geometry(node_id=node_ids, node_xyz=np.concatenate(xyz),
                     elem_conn=conn or None, elem_type=types or None,
                     elem_block=blocks or None, block_id=block_ids,
                     block_name=block_names, length_unit=parts[0].length_unit,
                     block_properties=properties)
    if tolerance is None:
        low, high = whole.node_xyz.min(axis=0), whole.node_xyz.max(axis=0)
        tolerance = 1e-6 * float(np.linalg.norm(high - low) or 1.0)
    whole.merge_coincident_nodes(tolerance)
    return whole
