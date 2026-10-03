"""Plate meshes built from planes, and solid meshes built from blocks.

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

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Geometry

__all__ = ['assemble', 'block', 'join', 'landing', 'plane', 'tie']


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


def block(corner: ArrayLike, edge_a: ArrayLike, edge_b: ArrayLike,
          edge_c: ArrayLike, size: float, name: str = '', *,
          unit: str | None = 'm', holes: Sequence[Any] = (),
          hole_name: str | None = None) -> Geometry:
    """A rectangular block meshed into eight-node bricks, in one block.

    `plane` one dimension up (2026-09-30, for the four-unit frame
    example): a corner and three perpendicular edges, each divided
    evenly into the whole number of elements nearest `size`, so the
    bricks come out close to cubes. Blocks that meet over a face divide
    it the same way, since it is the same rectangle in both, so
    `assemble` ties them there — give blocks that meet the same size.

    Holes are cut the way a structured mesh can cut them: every brick
    whose center lies within a hole's radius of its axis leaves the
    block. A hole is (center, radius, axis) with the center a point on
    the axis and the axis 0, 1 or 2 for the edge it runs along;
    (center, radius, axis, depth) for a blind hole that depth from the
    face the axis enters at; and a fifth item names the block its
    bricks go to instead of leaving — a threaded insert, given its own
    material in the Blocks table — or is None for a void. A hole with
    no fifth item takes `hole_name`. Holes apply in order, a later one
    over an earlier: a through hole, then an insert named to the same
    radius part way down, then a void of the insert's bore, is a
    threaded insert in a drilled hole. The nodes on a hole's rim — the
    ones its bricks share with the bricks beside them — are then moved
    radially onto the circle, a node on the block's own face sliding
    along that face to where the circle meets it, so a hole is round to
    within the polygon its rim nodes make rather than stair-stepped
    (a hole centered on the block's edge is a round-over and keeps its
    steps; its corner brick cannot follow an arc): a
    stair-stepped hole in a member six bricks wide was anywhere from
    three to four bricks across as its center fell, and the frame's
    modes wandered by ten percent with it (2026-09-30).

    Parameters
    ----------
    corner : array_like
        One corner, (x, y, z).
    edge_a, edge_b, edge_c : array_like
        The three edges from that corner, as vectors: their lengths are
        the block's sides. They must be perpendicular.
    size : float
        The element size aimed at.
    name : str, optional
        The block's name — the part this is.
    unit : str or None, default 'm'
        The unit every length given here is in, as for `plane`.
    holes : sequence of tuple, optional
        Cylindrical holes, as above, in the same unit and the same frame.
    hole_name : str, optional
        The block the bricks of a hole that names none go to; None
        removes them.

    Returns
    -------
    Geometry
        Nodes numbered from 1; one block, and one more for each name
        the holes put bricks in.
    """
    from ..units import si_transform

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
    na, nb, _nc = counts
    local = np.array([(a, b, c) for c in grids[2] for b in grids[1]
                      for a in grids[0]])
    xyz = corner + local @ np.array(axes)
    node = np.arange(1, len(xyz) + 1)
    conn, centers = [], []
    for k in range(counts[2] - 1):
        for j in range(nb - 1):
            for i in range(na - 1):
                first = k * na * nb + j * na + i
                cell = [first, first + 1, first + na + 1, first + na]
                conn.append(node[cell + [c + na * nb for c in cell]])
                centers.append(local[cell].mean(axis=0)
                               + 0.5 * (grids[2][k + 1] - grids[2][k])
                               * np.array([0.0, 0.0, 1.0]))
    centers = np.array(centers)
    # 1 is the block, 0 a brick removed, 2 and up the named blocks
    blocks = np.ones(len(conn), dtype=int)
    names = [name]
    rims: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    conn_array = np.array(conn) - 1 if conn else np.empty((0, 8), int)
    for hole in holes:
        center, radius, axis = hole[0], float(hole[1]), int(hole[2])
        depth = float(hole[3]) if len(hole) > 3 and hole[3] is not None else None
        target = hole[4] if len(hole) > 4 else hole_name
        along = np.asarray(center, dtype=float) - corner
        along = np.array([float(along @ unit_axis) for unit_axis in axes])
        across = [d for d in range(3) if d != axis]
        distance = np.linalg.norm(centers[:, across] - along[across], axis=1)
        inside = distance <= radius
        if depth is not None:
            # blind from the face the axis enters at: the corner's own
            # face when the center sits on it, the far face otherwise
            far = along[axis] > lengths[axis] / 2.0
            reach = (lengths[axis] - centers[:, axis] if far
                     else centers[:, axis])
            inside &= reach <= depth
        # a hole whose axis lies inside the block gets a round rim; one
        # centered on the block's own edge or corner is a round-over,
        # and its rim stays stair-stepped: the corner brick a fillet
        # keeps cannot be pulled onto the arc without folding, and its
        # area is within a few percent of the fillet's already
        within = all(0.0 < along[d] < lengths[d]
                     for d in range(3) if d != axis)
        beyond = distance > radius
        if within and inside.any() and beyond.any():
            # the rim: nodes the hole's bricks share with bricks beyond
            # its radius — not with the bricks under a blind hole's
            # floor, which lie inside the circle and must stay put
            rim = np.intersect1d(np.unique(conn_array[inside]),
                                 np.unique(conn_array[beyond]))
            rims.append((rim, along, radius, axis))
        if target is None:
            blocks[inside] = 0
        else:
            if target not in names:
                names.append(target)
            blocks[inside] = names.index(target) + 1
    for rim, along, radius, axis in rims:
        across = [d for d in range(3) if d != axis]
        for row in rim:
            offset = local[row, across] - along[across]
            distance = float(np.linalg.norm(offset))
            if distance == 0.0:
                continue
            target = along[across] + offset * (radius / distance)
            # a node on the block's face keeps to the face: it slides to
            # where the circle meets it, or stays if the circle does not
            on_face = [k for k, d in enumerate(across)
                       if local[row, d] <= 1e-12 * lengths[d]
                       or local[row, d] >= lengths[d] * (1 - 1e-12)]
            if len(on_face) == 2:
                continue
            if on_face:
                held, other = on_face[0], 1 - on_face[0]
                reach = radius ** 2 - offset[held] ** 2
                if reach < 0.0:
                    continue
                target = target.copy()
                target[held] = local[row, across[held]]
                target[other] = along[across[other]] + (
                    np.sign(offset[other]) or 1.0) * np.sqrt(reach)
            local[row, across] = target
        xyz = corner + local @ np.array(axes)
    if not (blocks > 0).all():
        keep = blocks > 0
        conn = [c for c, k in zip(conn, keep) if k]
        blocks = blocks[keep]
        used = np.unique(np.concatenate(conn)) if conn else np.array([], int)
        renumber = {int(old): new for new, old in enumerate(used, 1)}
        xyz = xyz[used - 1]
        conn = [np.array([renumber[int(n)] for n in c]) for c in conn]
        node = np.arange(1, len(xyz) + 1)
    if not conn:
        raise ValueError(f'{name or "a block"}: the holes leave nothing')
    present = [k for k in range(1, len(names) + 1) if (blocks == k).any()]
    return Geometry(node_id=node, node_xyz=xyz * scale, elem_conn=conn,
                    elem_type=[115] * len(conn),
                    elem_block=[int(b) for b in blocks],
                    block_id=present,
                    block_name=[names[k - 1] for k in present],
                    length_unit=unit)


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


def block_families(geometry: Geometry, block_id: int) -> set[str]:
    """The element families a block of `geometry` holds: empty for a
    block with no elements yet."""
    import numpy as np

    from .geometry import element_family

    mask = np.asarray(geometry.elem_block) == int(block_id)
    return {element_family(int(code)) for code in np.asarray(geometry.elem_type)[mask]}


def block_refusal(geometry: Geometry, part: Geometry) -> str | None:
    """Why `part` cannot join `geometry` by block name, or None.

    A block holds one element family (`Geometry.mixed_blocks`), and a
    part whose block is named like one already holding another family
    would make it two at once: a plate added under the name of a block
    of bricks was one block, deleted whole from either family's row in
    the tree (Brandon, 2026-10-02). The one rule `join` refuses by and
    the Add pane reads before it offers Add.
    """
    from .geometry import FAMILY_LABELS

    names = list(geometry.block_name)
    for k, block in enumerate(part.block_id):
        name = part.block_name[k]
        if not name or name not in names:
            continue
        held = block_families(geometry, int(geometry.block_id[names.index(name)]))
        coming = block_families(part, int(block))
        if held and coming and held != coming:
            have = ', '.join(FAMILY_LABELS[f].lower() for f in sorted(held))
            want = ', '.join(FAMILY_LABELS[f].lower() for f in sorted(coming))
            return (f'block {name!r} holds {have}, and {want} need a block of '
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
                  tolerance: float | None = None) -> int:
    """How many of `part`'s elements `join` would leave out for being
    elements already there — every corner on a node already there, the
    cell already filled — so a reading can say it before the Add."""
    import numpy as np

    on, nearest = landing(geometry, part, tolerance)
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
         tolerance: float | None = None) -> dict:
    """Put `part` into `geometry`, in place: each of the part's nodes that
    falls on a node already there becomes that node, the rest are added
    after the highest id, and the part's elements follow them. The nodes
    already there keep their ids — data linked to the geometry names
    them — which is what makes this the way a plane is added to a model
    under construction (Add Plane) as well as how `assemble` builds one.

    A block of the part named like one already in the geometry joins it:
    the five planes of a box, each named 'box', are one part, given its
    material once. An unnamed block is always a block of its own. A
    named block holding another element family is refused
    (`block_refusal`): a block holds one family.

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
        nodes already there; 'elements', the elements added; 'duplicates', the part's
        elements left out for being elements already there; 'blocks',
        the ids of the blocks they went into.
    """
    if (geometry.num_nodes and part.num_nodes
            and geometry.units_defined != part.units_defined):
        raise ValueError('one has its length unit defined and the other '
                         'does not, so their coordinates do not mean the '
                         'same thing')
    refusal = block_refusal(geometry, part)
    if refusal is not None:
        raise ValueError(refusal)
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
    # an element of the part that is an element already there — the
    # overlap of two crossing bars — fills that cell once, as the one
    # already there; adding it again counted the overlap twice
    # (2026-10-02)
    there = _solid_cells(geometry)
    conn, types, blocks = [], [], []
    for element, code, block in zip(part.elem_conn, part.elem_type,
                                    part.elem_block):
        renamed = [renumber[int(n)] for n in element]
        if (int(code), tuple(sorted(renamed))) in there:
            continue
        conn.append(renamed)
        types.append(int(code))
        blocks.append(block_map[int(block)])
    duplicates = len(part.elem_conn) - len(conn)
    geometry.add_elements(conn, types, blocks)
    if geometry.length_unit is None:
        geometry.length_unit = part.length_unit
    return {'added': len(new_ids), 'shared': int(on.sum()),
            'elements': len(conn), 'duplicates': duplicates,
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
