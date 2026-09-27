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

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Geometry

__all__ = ['assemble', 'join', 'landing', 'plane', 'tie']


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


def landing(geometry: Geometry, part: Geometry,
            tolerance: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Which of `part`'s nodes fall on a node of `geometry`, and which one
    — what `join` does with them, asked without joining: Add Plane's
    reading says how many nodes a plane would share before it is added.

    Parameters
    ----------
    geometry, part : Geometry
        Where the part would go, and the part.
    tolerance : float, optional
        As `join`'s.

    Returns
    -------
    (ndarray of bool, ndarray of int)
        Per node of the part: whether it lands, and the row in
        `geometry` of the node it lands on (meaningful where it does).
    """
    from scipy.spatial import cKDTree

    if tolerance is None:
        both = np.vstack([geometry.node_xyz, part.node_xyz])
        tolerance = 1e-6 * float(np.linalg.norm(np.ptp(both, axis=0)) or 1.0)
    if not geometry.num_nodes:
        return (np.zeros(part.num_nodes, dtype=bool),
                np.zeros(part.num_nodes, dtype=int))
    distance, nearest = cKDTree(geometry.node_xyz).query(part.node_xyz)
    return distance <= tolerance, nearest


def join(geometry: Geometry, part: Geometry,
         tolerance: float | None = None) -> dict:
    """Put `part` into `geometry`, in place: each of the part's nodes that
    falls on a node already there becomes that node, the rest are added
    after the highest id, and the part's elements follow them. The nodes
    already there keep their ids — data linked to the geometry names
    them — which is what makes this the way a plane is added to a model
    under construction (Add Plane) as well as how `assemble` builds one.

    A block of the part named like one already in the geometry joins it:
    the five planes of a box, each named 'box', are one part, given its
    material once. An unnamed block is always a block of its own.

    Parameters
    ----------
    geometry : Geometry
        Where the part goes; changed in place.
    part : Geometry
        What goes in. Its coordinates must mean what the geometry's do:
        both SI (units defined) or both as given.
    tolerance : float, optional
        How close a node must be to one already there to be it, in
        meters once units are defined. Defaults to a millionth of the
        size of the two together.

    Returns
    -------
    dict
        'added', the nodes added; 'shared', the part's nodes that fell on
        nodes already there; 'elements', the elements added; 'blocks',
        the ids of the blocks they went into.
    """
    if (geometry.num_nodes and part.num_nodes
            and geometry.units_defined != part.units_defined):
        raise ValueError('one has its length unit defined and the other '
                         'does not, so their coordinates do not mean the '
                         'same thing')
    on, nearest = landing(geometry, part, tolerance)
    new_ids = geometry.add_nodes(part.node_xyz[~on])
    renumber = dict(zip(part.node_id[~on].tolist(), new_ids.tolist()))
    renumber.update(zip(part.node_id[on].tolist(),
                        geometry.node_id[nearest[on]].tolist()))
    names = list(geometry.block_name)
    block_map = {}
    for k, block in enumerate(part.block_id):
        name = part.block_name[k]
        if name and name in names:
            block_map[int(block)] = int(geometry.block_id[names.index(name)])
            continue
        new = geometry.add_block(name)
        names.append(name)
        block_map[int(block)] = new
        if int(block) in part.block_properties:
            geometry.block_properties[new] = part.block_properties[int(block)]
    geometry.add_elements(
        [[renumber[int(n)] for n in element] for element in part.elem_conn],
        part.elem_type, [block_map[int(b)] for b in part.elem_block])
    if geometry.length_unit is None:
        geometry.length_unit = part.length_unit
    return {'added': len(new_ids), 'shared': int(on.sum()),
            'elements': len(part.elem_conn),
            'blocks': sorted(set(block_map.values()))}


def assemble(*parts: Geometry, tolerance: float | None = None) -> Geometry:
    """One geometry from several, joined in turn (`join`): nodes numbered
    from 1 in the order they arrive, a node falling on one already there
    becoming it — so planes meeting along a line are tied there — and
    blocks of the same name one block. Unnamed blocks stay apart.

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
    if tolerance is None:
        both = np.vstack([part.node_xyz for part in parts])
        tolerance = 1e-6 * float(np.linalg.norm(np.ptp(both, axis=0)) or 1.0)
    whole = Geometry(node_id=[], node_xyz=np.empty((0, 3)),
                     length_unit=parts[0].length_unit)
    for part in parts:
        join(whole, part, tolerance)
    # a part's own coincident nodes, which joining it does not look for
    whole.merge_coincident_nodes(tolerance)
    return whole


#: the block ties go into when none is named and the geometry has no
#: rigid block yet
TIE_BLOCK = 'ties'


def _element_rows(geometry: Geometry, elements: Any) -> np.ndarray:
    """Rows of the given element ids, refusing any that is not there."""
    ids = np.asarray([int(e) for e in elements], dtype=np.int64)
    rows = np.flatnonzero(np.isin(geometry.elem_id, ids))
    if len(rows) != len(set(ids.tolist())):
        missing = sorted(set(ids.tolist()) - set(geometry.elem_id.tolist()))
        raise ValueError(f'no element {missing[0]}')
    return rows


def tie(geometry: Geometry, elements: Any, to: Any,
        block: str | None = None) -> dict:
    """Tie a patch of elements rigidly to what lies under it (Tie): every
    node of `elements` joined by a rigid, massless link to the nearest
    node of `to`, the links added to a rigid block. A bolted joint is the
    use (Brandon, 2026-09-26): select the elements under a washer, tie
    them to the part below — one step per bolt where two clicks per link
    had been the only way.

    A node the patch shares with the target is already joined and is
    left alone. Each link runs from the target's node to the patch's, so
    the patch follows the part it is bolted to.

    Parameters
    ----------
    geometry : Geometry
        Changed in place.
    elements : sequence of int
        The patch, by element id.
    to : str, int or sequence of int
        What to tie to: a block, by name or id — its nearest nodes — or a
        second patch, by element ids.
    block : str, optional
        The block the links go into, by name, made rigid if new. Defaults
        to the geometry's first rigid block, or a new one named 'ties'.

    Returns
    -------
    dict
        'links', how many were added; 'shared', the patch's nodes left
        alone because the target holds them already; 'block', the id of
        the block the links went into.
    """
    from scipy.spatial import cKDTree

    from .fem import RIGID, BlockProperties

    patch_rows = _element_rows(geometry, elements)
    if not len(patch_rows):
        raise ValueError('select the elements to tie')
    if isinstance(to, (str, int, np.integer)):
        target_ids = geometry.elements_in(to if isinstance(to, str) else int(to))
        if not target_ids:
            raise ValueError(f'block {to!r} holds no elements')
    else:
        target_ids = list(to)
    target_rows = _element_rows(geometry, target_ids)
    if not len(target_rows):
        raise ValueError('select the elements to tie to')
    patch = np.unique(np.concatenate([geometry.elem_conn[r] for r in patch_rows]))
    target = np.unique(np.concatenate([geometry.elem_conn[r]
                                       for r in target_rows]))
    followers = np.setdiff1d(patch, target)
    if not len(followers):
        raise ValueError('the patch lies on what it would be tied to — '
                         'every node is shared already')
    xyz = geometry.node_xyz
    _distance, nearest = cKDTree(xyz[geometry.node_index(target)]).query(
        xyz[geometry.node_index(followers)])
    rigid = [int(b) for b in geometry.block_id
             if getattr(geometry.block_properties.get(int(b)), 'material',
                        None) is not None
             and geometry.block_properties[int(b)].material.is_rigid]
    names = list(geometry.block_name)
    if block is None and rigid:
        block_id = rigid[0]
    else:
        name = TIE_BLOCK if block is None else block
        if name in names:
            block_id = int(geometry.block_id[names.index(name)])
            held = geometry.block_properties.get(block_id)
            if held is not None and not held.material.is_rigid:
                raise ValueError(f'block {name!r} is {held.material.name}, '
                                 'not rigid: name another for the ties')
        else:
            block_id = geometry.add_block(name)
        geometry.block_properties[block_id] = BlockProperties(RIGID)
    geometry.add_elements([[int(target[k]), int(node)]
                           for k, node in zip(nearest, followers)],
                          [21] * len(followers), [block_id] * len(followers))
    return {'links': len(followers), 'shared': len(patch) - len(followers),
            'block': block_id}
