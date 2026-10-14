"""Plate meshes built from planes, and solid meshes built from blocks.

A structure made of flat plates, such as a box, a channel or a bracket,
is described most simply as the planes it is made of: each one a
rectangle at its mid-thickness, meshed into rectangular plate elements
and put in an element group of its own. `plane` makes one, and `block`
makes a box of solid bricks the same way. `assemble` joins the pieces
into one geometry and merges the nodes they share, so the plates are
tied along the lines where their mid-surfaces meet. Where mid-surfaces
do not meet, as with a bolted foot sitting on a wall, no node is shared,
and the two are joined by rigid links (`fem.RIGID`) that `tie` adds.

Each element group is then given a material and a thickness
(`fem.GroupProperties`, or the Element Groups table), and
`fem.Model.from_geometry` builds the model.

Nodes on a shared line coincide only because both planes divide it the
same way, so planes that meet should be given the same element size.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Geometry

# Planes joined by assemble were written for the BARC example (Brandon,
# 2026-09-26); block followed for the Frame-Wing example (2026-09-30).

__all__ = ['assemble', 'block', 'join', 'landing', 'plane', 'tie']


def plane(corner: ArrayLike, edge_a: ArrayLike, edge_b: ArrayLike,
          size: float, name: str = '', *, unit: str | None = 'm') -> Geometry:
    """A rectangle meshed into four-node plate elements, in one element group.

    Each edge is divided evenly into the whole number of elements nearest
    `size`, so the elements come out close to square and close to that
    size. Planes that meet along a line divide it the same way, since the
    line has the same length in both, and `assemble` ties them there.

    Parameters
    ----------
    corner : array_like
        One corner, (x, y, z).
    edge_a, edge_b : array_like
        The two edges from that corner, as vectors; their lengths are the
        rectangle's sides. They must be perpendicular, because the plate
        element is a rectangle, and edges that are not are refused.
    size : float
        The element size aimed at, positive.
    name : str, optional
        The element group's name, which is the part this plane is.
    unit : str or None, default 'm'
        The unit every length given here is in. The geometry holds the
        lengths in SI, as any geometry with its units defined does, and
        remembers this unit; None leaves them as given, units undefined.

    Returns
    -------
    Geometry
        Nodes numbered from 1, one element group holding every element.
    """
    # even spacing, roughly square elements and nothing to lay out by
    # hand (Brandon, 2026-09-26)
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
                    elem_type=[44] * len(conn), elem_group=[1] * len(conn),
                    group_id=[1], group_name=[name], length_unit=unit)


def block(corner: ArrayLike, edge_a: ArrayLike, edge_b: ArrayLike,
          edge_c: ArrayLike, size: float, name: str = '', *,
          unit: str | None = 'm', holes: Sequence[Any] = (),
          hole_name: str | None = None) -> Geometry:
    """A rectangular block meshed into eight-node bricks, in one element
    group.

    This is `plane` in three dimensions: a corner and three perpendicular
    edges, each divided evenly into the whole number of elements nearest
    `size`, so the bricks come out close to cubes. Blocks that meet over a
    face divide it the same way, since it is the same rectangle in both,
    and `assemble` ties them there, so blocks that meet should be given
    the same size.

    A hole is a cylinder along one of the block's edges, and every brick
    whose center lies within the hole's radius of its axis is taken out
    of the block. A hole is given as (center, radius, axis), with the
    center a point on the hole's axis and `axis` 0, 1 or 2 for the edge it
    runs along (`edge_a`, `edge_b` or `edge_c`); as (center, radius, axis,
    depth) for a blind hole of that depth, measured from the end face
    nearer the center point; and with a fifth item naming the element
    group the hole's bricks go to instead of being removed (a threaded
    insert, given its own material in the Element Groups table), or None
    to remove them. A hole with no fifth item uses `hole_name`. Holes are
    applied in order, a later one over an earlier, so a through hole, then
    an insert of the same radius part way down, then a void the size of
    the insert's bore, is a threaded insert in a drilled hole. Every hole
    must run along the same edge, because the bricks are a grid swept
    along it; holes along different edges are refused.

    A hole is meshed round, not stair-stepped, by the same mesher Mesh
    Solids uses. Where there is room around the hole, the grid there is
    replaced by a ring mesh with its nodes evenly spaced on the circle,
    and a core of bricks where an insert fills it. Where there is not
    room, the grid's nodes nearest the rim are moved onto the circle,
    sliding along the block's face where the hole breaks out. A hole
    whose center lies on the block's edge is a round-over and keeps its
    steps, because its corner brick cannot follow an arc, and its area is
    already within a few percent of the fillet's.

    Parameters
    ----------
    corner : array_like
        One corner, (x, y, z).
    edge_a, edge_b, edge_c : array_like
        The three edges from that corner, as vectors; their lengths are
        the block's sides. They must be perpendicular, and edges that are
        not are refused.
    size : float
        The element size aimed at, positive.
    name : str, optional
        The element group's name, which is the part this block is.
    unit : str or None, default 'm'
        The unit every length given here is in, as for `plane`.
    holes : sequence of tuple, optional
        Cylindrical holes, as above, in the same unit and the same frame.
    hole_name : str, optional
        The element group the bricks of a hole that names none go to;
        None removes them.

    Returns
    -------
    Geometry
        Nodes numbered from 1; one element group, and one more for each
        name the holes put bricks in.

    Raises
    ------
    ValueError
        If an edge has no length, the edges are not perpendicular, `size`
        is not positive, the holes run along different edges, or the
        holes leave no bricks.
    """
    # A stair-stepped hole in a member six bricks wide was anywhere from
    # three to four bricks across as its center moved, and the frame's
    # modes changed by ten percent with it (2026-09-30), so a hole is
    # meshed round. Mesh Solids and this share one mesher,
    # `solids._mesh_grid` (2026-10-11).
    from ..units import si_transform
    from .solids import _mesh_grid

    scale = 1.0 if unit is None else si_transform(unit, 'length')[0]
    corner = np.asarray(corner, dtype=float)
    edges = [np.asarray(edge, dtype=float) for edge in (edge_a, edge_b, edge_c)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    if not all(length > 0 for length in lengths):
        raise ValueError(f'{name or "a block"}: an edge has no length')
    if not size or size <= 0:
        raise ValueError(f'{name or "a block"}: the element size must be '
                         'positive')
    for i, j in ((0, 1), (1, 2), (0, 2)):
        if abs(float(edges[i] @ edges[j])) > 1e-9 * lengths[i] * lengths[j]:
            raise ValueError(f'{name or "a block"}: the edges are not '
                             'perpendicular, and the brick is a box')
    axes = [edge / length for edge, length in zip(edges, lengths)]
    counts = [max(1, round(length / size)) + 1 for length in lengths]
    grids = [np.linspace(0.0, length, count)
             for length, count in zip(lengths, counts)]
    along_axis = {int(hole[2]) for hole in holes}
    if len(along_axis) > 1:
        raise ValueError(f'{name or "a block"}: every hole must run along '
                         'the same edge')
    w = along_axis.pop() if along_axis else 2
    order = [(w + 1) % 3, (w + 2) % 3, w]
    # the holes in the block's own frame, the corner at the origin, each
    # with the element group its bricks go to (0 for none)
    names = [name]
    cut = []
    for hole in holes:
        center = np.asarray(hole[0], dtype=float) - corner
        center = np.array([float(center @ unit_axis) for unit_axis in axes])
        depth = float(hole[3]) if len(hole) > 3 and hole[3] is not None else None
        target = hole[4] if len(hole) > 4 else hole_name
        if target is not None and target not in names:
            names.append(target)
        group = 0 if target is None else names.index(target) + 1
        if depth is None:
            low, high = 0.0, lengths[w]
        elif center[w] > lengths[w] / 2.0:
            # blind from the face the axis enters at: the corner's own
            # face when the center sits on it, the far face otherwise
            low, high = lengths[w] - depth, lengths[w]
        else:
            low, high = 0.0, depth
        cut.append((center, float(hole[1]), low, high, group))
    across = order[:2]

    def held(points):
        group = np.ones(len(points), dtype=int)
        for center, radius, low, high, to in cut:
            inside = ((np.linalg.norm(points[:, across] - center[across], axis=1)
                       <= radius) & (points[:, w] >= low) & (points[:, w] <= high))
            group[inside] = to
        return group

    # a hole centered on the block's edge or corner is a round-over and
    # keeps its steps: no circle for it
    circles = [(center, radius, low, high, True) for center, radius, low, high, _to in cut
               if all(0.0 < center[d] < lengths[d] for d in across)]
    local, conn, groups, _made = _mesh_grid([grids[k] for k in order], order, size,
                                            held, circles)
    if not len(conn):
        raise ValueError(f'{name or "a block"}: the holes leave nothing')
    xyz = corner + local @ np.array(axes)
    present = [k for k in range(1, len(names) + 1) if (groups == k).any()]
    return Geometry(node_id=np.arange(1, len(xyz) + 1), node_xyz=xyz * scale,
                    elem_conn=list(conn), elem_type=[115] * len(conn),
                    elem_group=[int(b) for b in groups],
                    group_id=present,
                    group_name=[names[k - 1] for k in present],
                    length_unit=unit)


def landing(geometry: Geometry, part: Geometry,
            tolerance: float | None = None,
            apart: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Which of `part`'s nodes fall on a node of `geometry`, and on which
    one.

    This is the matching `join` does, without the join, so Add Plane's
    reading can say how many nodes a plane would share before it is added.

    Parameters
    ----------
    geometry, part : Geometry
        Where the part would go, and the part.
    tolerance : float, optional
        As `join`'s.
    apart : bool, default False
        As `join`'s: only the nodes of the element groups the part joins
        are landed on.

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
    rows = (np.arange(geometry.num_nodes) if not apart
            else _own_group_rows(geometry, part))
    if not len(rows):
        return (np.zeros(part.num_nodes, dtype=bool),
                np.zeros(part.num_nodes, dtype=int))
    distance, nearest = cKDTree(geometry.node_xyz[rows]).query(part.node_xyz)
    return distance <= tolerance, rows[nearest]


def _own_group_rows(geometry: Geometry, part: Geometry) -> np.ndarray:
    """Rows of the geometry's nodes used by the element groups the part
    joins by name — what a part kept apart may land on. An unnamed
    group joins none, so a part of one lands on nothing."""
    names = {name for name in part.group_name if name}
    groups = [int(g) for g, name in zip(geometry.group_id, geometry.group_name)
              if name in names]
    rows = np.flatnonzero(np.isin(geometry.elem_group, groups))
    if not len(rows):
        return np.empty(0, dtype=int)
    used = np.unique(np.concatenate([geometry.elem_conn[k] for k in rows]))
    return np.sort(geometry.node_index(used))


def group_families(geometry: Geometry, group_id: int) -> set[str]:
    """The element families an element group of `geometry` holds, empty
    for an element group with no elements yet."""
    import numpy as np

    from .geometry import element_family

    mask = np.asarray(geometry.elem_group) == int(group_id)
    return {element_family(int(code)) for code in np.asarray(geometry.elem_type)[mask]}


def group_refusal(geometry: Geometry, part: Geometry) -> str | None:
    """Why `part` cannot join `geometry` by element group name, or None.

    An element group holds one element family (`Geometry.mixed_groups`),
    so a part whose element group has the name of one that already holds
    another family is refused. This is the one rule `join` refuses by, and
    the Add pane checks it before it offers Add.
    """
    # A plate added under the name of an element group of bricks made one
    # element group of two families, deleted whole from either family's
    # row in the tree (Brandon, 2026-10-02).
    from .geometry import FAMILY_LABELS

    names = list(geometry.group_name)
    for k, group in enumerate(part.group_id):
        name = part.group_name[k]
        if not name or name not in names:
            continue
        held = group_families(geometry, int(geometry.group_id[names.index(name)]))
        coming = group_families(part, int(group))
        if held and coming and held != coming:
            have = ', '.join(FAMILY_LABELS[f].lower() for f in sorted(held))
            want = ', '.join(FAMILY_LABELS[f].lower() for f in sorted(coming))
            return (f'element group {name!r} holds {have}, and {want} need an element group of '
                    'their own: give them another name')
    return None


def _solid_cells(geometry: Geometry) -> set[tuple]:
    """(type, sorted nodes) of every solid element: the cells already
    filled. Solids only, as `Geometry.duplicate_elements` has it."""
    from .geometry import SOLID_FAMILIES, element_family

    return {(int(t), tuple(sorted(int(n) for n in c)))
            for t, c in zip(geometry.elem_type, geometry.elem_conn)
            if element_family(int(t)) in SOLID_FAMILIES}


def already_there(geometry: Geometry, part: Geometry,
                  tolerance: float | None = None, apart: bool = False) -> int:
    """How many of `part`'s solid elements `join` would leave out because
    they are already there (every corner on a node already there and the
    cell already filled), so a reading can say it before the Add."""
    import numpy as np

    on, nearest = landing(geometry, part, tolerance, apart)
    if not on.any():
        return 0
    ids = {int(part.node_id[k]): int(geometry.node_id[nearest[k]])
           for k in np.flatnonzero(on)}
    there = _solid_cells(geometry)
    count = 0
    for element, code in zip(part.elem_conn, part.elem_type):
        renamed = [ids.get(int(n)) for n in element]
        if None not in renamed and (int(code), tuple(sorted(renamed))) in there:
            count += 1
    return count


def join(geometry: Geometry, part: Geometry,
         tolerance: float | None = None, apart: bool = False) -> dict:
    """Put `part` into `geometry`, in place.

    Each of the part's nodes that falls on a node already there becomes
    that node, the rest are added after the highest id, and the part's
    elements follow them. The nodes already there keep their ids, because
    data linked to the geometry refers to them by id, so this is how a
    plane is added to a model under construction (Add Plane) as well as
    how `assemble` builds one. A solid element of the part that would fill
    a cell already filled is left out, so the overlap of two crossing bars
    is counted once.

    An element group of the part with the same name as one already in the
    geometry joins it, so the five planes of a box, each named 'box', are
    one part and are given a material once. An unnamed element group is
    always an element group of its own. A named element group holding a
    different element family is refused (`group_refusal`), since an
    element group holds one family, and so is a part whose length unit is
    defined when the geometry's is not, or the other way round.

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
    apart : bool, default False
        Land only on the nodes of the element group the part joins by
        name, so a part meshed against another (a channel's foot on the
        box it is bolted to) keeps nodes of its own where the two touch,
        coincident and not joined, until Merge Coincident Nodes joins the
        places picked. A plane of the box still joins the box's other
        planes.

    Returns
    -------
    dict
        'added', the number of nodes added; 'shared', the number of the
        part's nodes that fell on nodes already there; 'elements', the
        number of elements added; 'duplicates', the number of the part's
        solid elements left out because they were already there;
        'groups', the ids of the element groups they went into.
    """
    # apart is for the BARC's bolts (Brandon, 2026-10-11).
    if (geometry.num_nodes and part.num_nodes
            and geometry.units_defined != part.units_defined):
        raise ValueError('one has its length unit defined and the other '
                         'does not, so their coordinates do not mean the '
                         'same thing')
    refusal = group_refusal(geometry, part)
    if refusal is not None:
        raise ValueError(refusal)
    on, nearest = landing(geometry, part, tolerance, apart)
    new_ids = geometry.add_nodes(part.node_xyz[~on])
    renumber = dict(zip(part.node_id[~on].tolist(), new_ids.tolist()))
    renumber.update(zip(part.node_id[on].tolist(),
                        geometry.node_id[nearest[on]].tolist()))
    names = list(geometry.group_name)
    group_map = {}
    for k, group in enumerate(part.group_id):
        name = part.group_name[k]
        if name and name in names:
            group_map[int(group)] = int(geometry.group_id[names.index(name)])
            continue
        new = geometry.add_group(name)
        names.append(name)
        group_map[int(group)] = new
        if int(group) in part.group_properties:
            geometry.group_properties[new] = part.group_properties[int(group)]
    # an element of the part that is an element already there — the
    # overlap of two crossing bars — fills that cell once, as the one
    # already there; adding it again counted the overlap twice
    # (2026-10-02)
    there = _solid_cells(geometry)
    conn, types, groups = [], [], []
    for element, code, group in zip(part.elem_conn, part.elem_type,
                                    part.elem_group):
        renamed = [renumber[int(n)] for n in element]
        if (int(code), tuple(sorted(renamed))) in there:
            continue
        conn.append(renamed)
        types.append(int(code))
        groups.append(group_map[int(group)])
    duplicates = len(part.elem_conn) - len(conn)
    geometry.add_elements(conn, types, groups)
    if geometry.length_unit is None:
        geometry.length_unit = part.length_unit
    return {'added': len(new_ids), 'shared': int(on.sum()),
            'elements': len(conn), 'duplicates': duplicates,
            'groups': sorted(set(group_map.values()))}


def assemble(*parts: Geometry, tolerance: float | None = None,
             apart: bool = False) -> Geometry:
    """One geometry from several, joined in turn by `join`.

    Nodes are numbered from 1 in the order they arrive, and a node that
    falls on one already there becomes it, so planes meeting along a line
    are tied there. Element groups of the same name become one element
    group; unnamed element groups stay separate. Coincident nodes within
    one part, which `join` does not look for, are merged as well (with
    `apart`, only within each element group).

    Parameters
    ----------
    *parts : Geometry
        The pieces, at least one, all in the same length unit; parts in
        different units are refused.
    tolerance : float, optional
        How close nodes must be to be one, in the units the geometry holds
        them in (meters when the length unit is defined). Defaults to a
        millionth of the assembly's size.
    apart : bool, default False
        Join each piece only to its own element group's nodes (`join`),
        so the element groups touch without sharing a node.

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
        join(whole, part, tolerance, apart)
    # a part's own coincident nodes, which joining it does not look for
    if apart:
        for group in whole.group_id:
            whole.merge_coincident_nodes(tolerance, groups=[int(group)])
    else:
        whole.merge_coincident_nodes(tolerance)
    return whole


#: the element group ties go into when none is named and the geometry has no
#: rigid element group yet
TIE_GROUP = 'ties'


def _element_rows(geometry: Geometry, elements: Any) -> np.ndarray:
    """Rows of the given element ids, refusing any that is not there."""
    ids = np.asarray([int(e) for e in elements], dtype=np.int64)
    rows = np.flatnonzero(np.isin(geometry.elem_id, ids))
    if len(rows) != len(set(ids.tolist())):
        missing = sorted(set(ids.tolist()) - set(geometry.elem_id.tolist()))
        raise ValueError(f'no element {missing[0]}')
    return rows


def tie(geometry: Geometry, elements: Any, to: Any,
        group: str | None = None) -> dict:
    """Tie a patch of elements rigidly to what lies under it (Tie).

    Every node of `elements` is joined by a rigid, massless link to the
    nearest node of `to`, and the links are added to a rigid element
    group. The use is a bolted joint: select the elements under a washer
    and tie them to the part below, one step per bolt.

    A node the patch shares with the target is already joined and is left
    alone, and a patch whose nodes are all shared with the target is
    refused. Each link runs from the target's node to the patch's, so the
    patch follows the part it is bolted to.

    Parameters
    ----------
    geometry : Geometry
        Changed in place.
    elements : sequence of int
        The patch, by element id.
    to : str, int or sequence of int
        What to tie to: an element group, by name or id (its nodes nearest
        the patch), or a second patch, by element ids.
    group : str, optional
        The element group the links go into, by name. One that is new or
        has no properties yet is made rigid; one with other properties is
        refused. Defaults to the geometry's first rigid element group, or
        else one named 'ties'.

    Returns
    -------
    dict
        'links', how many were added; 'shared', the patch's nodes left
        alone because the target holds them already; 'group', the id of
        the element group the links went into.
    """
    # A bolted joint is the use (Brandon, 2026-09-26): before Tie, two
    # clicks per link were the only way.
    from scipy.spatial import cKDTree

    from .fem import RIGID, GroupProperties

    patch_rows = _element_rows(geometry, elements)
    if not len(patch_rows):
        raise ValueError('select the elements to tie')
    if isinstance(to, (str, int, np.integer)):
        target_ids = geometry.elements_in(to if isinstance(to, str) else int(to))
        if not target_ids:
            raise ValueError(f'element group {to!r} holds no elements')
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
    rigid = [int(b) for b in geometry.group_id
             if getattr(geometry.group_properties.get(int(b)), 'material',
                        None) is not None
             and geometry.group_properties[int(b)].material.is_rigid]
    names = list(geometry.group_name)
    if group is None and rigid:
        group_id = rigid[0]
    else:
        name = TIE_GROUP if group is None else group
        if name in names:
            group_id = int(geometry.group_id[names.index(name)])
            held = geometry.group_properties.get(group_id)
            if held is not None and held.kind != 'rigid':
                made = {'mass': 'point masses', 'spring': 'springs',
                        'ground': 'ground'}.get(held.kind) or held.material.name
                raise ValueError(f'element group {name!r} is {made}, '
                                 'not rigid: name another for the ties')
        else:
            group_id = geometry.add_group(name)
        geometry.group_properties[group_id] = GroupProperties(RIGID)
    geometry.add_elements([[int(target[k]), int(node)]
                           for k, node in zip(nearest, followers)],
                          [21] * len(followers), [group_id] * len(followers))
    return {'links': len(followers), 'shared': len(patch) - len(followers),
            'group': group_id}
