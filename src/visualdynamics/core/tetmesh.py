"""Tetrahedral meshing of a geometry's solids, from their recipe.

Mesh Solids meshes an extrusion into bricks (`solids.extrusion_mesh`).
A part that is not one, a hole drilled at an angle, cylinders along
three axes, is meshed here into tetrahedra, from the recipe's signed
distance (`solids.distance`) alone: nothing about the shape is assumed
beyond where its surface is and which way is out.

The method is the one of Persson and Strang (2004). Points are laid on
a body-centered lattice inside the solid, with points fixed along its
sharp edges and at its corners; every edge of their Delaunay
triangulation pushes its ends apart when shorter than the length aimed
at; a point pushed out of the solid is put back onto its surface along
the distance's gradient; and the triangulation is made again when the
points have moved enough to need it. The relaxation ends when the
points have settled, and the tetrahedra whose centroids fall outside
the solid are dropped.

Sharp edges are found from the recipe: each block gives six planes,
each cylinder a wall and two caps, and an edge is a curve where two of
them cross on the solid's surface, a corner a point where three do.
Points are placed along the curves a mesh size apart and held there,
so the mesh keeps every edge the recipe has and rounds nothing off.

The elements are ten-node tetrahedra (type 118): the corners, then the
midside nodes of edges 1-2, 2-3, 1-3, 1-4, 2-4, 3-4, those on a curved
wall moved onto it. Four-node ones (type 111) are made on request.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .geometry import Geometry
from .progress import Ticker
from .solids import (
    MESH_LIMIT,
    _basic_box,
    _box,
    _frame,
    _leaves,
    _owners,
    distance,
    solid,
    used_by,
)

#: edges aim at this many times the mean length, so the mesh is in
#: compression and fills its corners
SCALE = 1.1
#: the share of each step's push a point moves
STEP = 0.1
#: a point moved this share of the size since the last triangulation
#: calls for another. Persson and Strang use a tenth; a fifth gave the
#: same worst element and fewer poor ones on a 112,000-tetrahedron
#: mesh and a quarter fewer triangulations, which are most of the
#: time (2026-10-19)
RETRIANGULATE = 0.2
#: the relaxation ends when no interior point moved more than this
#: share of the size in a step
SETTLED = 0.01
#: and in any case after this many steps
MOST_STEPS = 200
#: a tetrahedron on the surface with a radius ratio below this is a
#: sliver, and is removed
SLIVER = 0.1
#: rounds of moving a sliver's corners and triangulating again
SLIVER_ROUNDS = 10
#: tetrahedra per unit of volume over the cube of the size, the
#: body-centered lattice relaxed: what `tet_count` estimates from
TETS_PER_CELL = 10.0
#: the element type by node count
TET_CODES = {4: 111, 10: 118}

#: the midside nodes of a ten-node tetrahedron, as (corner, corner)
#: pairs in the order they are numbered
TET10_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))


def tet_count(geometry: Any, ids, size: float) -> int:
    """About how many tetrahedra `tet_mesh` would make for these parts
    at `size`: their volume, read by sampling, over the cube of the
    size, times what a relaxed lattice packs into that. Within a
    quarter or so; the pane reads it to decide whether to preview.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id.
    size : float
        The element size, as the geometry holds its coordinates.

    Returns
    -------
    int
    """
    ids = [int(i) for i in ids]
    low, high = _extent(geometry, ids)
    rail = [np.linspace(a, b, 24) for a, b in zip(low, high)]
    at = np.stack(np.meshgrid(*rail, indexing='ij'), axis=-1).reshape(-1, 3)
    inside = (_distance(geometry, ids, at) < 0.0).mean()
    volume = float(inside * np.prod(high - low))
    return round(TETS_PER_CELL * volume / float(size) ** 3)


def tet_mesh(geometry: Any, ids, size: float, quadratic: bool = True,
             progress: Any = None) -> tuple[Any, dict]:
    """The parts as tetrahedra, ready to join to a geometry
    (`mesh.join`).

    Each part gets one element group, named for it (or for its kind and
    id when it has no name); an element goes to the part whose recipe
    holds its centroid, the later part where parts overlap. Parts that
    touch share their nodes there.

    Refused: a solid that is part of another, a size that is not
    positive, and a mesh of more than `MESH_LIMIT` tetrahedra by
    `tet_count`'s estimate.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id; each a solid nothing else uses.
    size : float
        The element size aimed at, as the geometry holds its
        coordinates: the edge length the relaxation settles to.
    quadratic : bool, default True
        Ten-node tetrahedra; False for four-node ones.
    progress : callable, optional
        Told ``(done, total)`` as the mesh goes: the edges, the points,
        each step of the relaxation, the elements.

    Returns
    -------
    (Geometry, dict)
        The tetrahedra, in the geometry's length unit; and what was
        made: 'tets' and 'nodes', the counts; 'steps', how many the
        relaxation took; 'quality', the least radius ratio (1 for a
        regular tetrahedron, 0 for a flat one); 'poor', how many are
        under 0.3; 'slivers', how many were removed from the surface;
        and 'shortest', the shortest edge.
    """
    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError('pick a solid to mesh')
    used = used_by(geometry)
    inside_another = [i for i in ids if i in used]
    if inside_another:
        raise ValueError(f'solid {inside_another[0]} is part of solid '
                         f'{used[inside_another[0]]}; mesh that one')
    size = float(size)
    if not size > 0.0:
        raise ValueError('an element size above zero')
    expected = tet_count(geometry, ids, size)
    if expected > MESH_LIMIT:
        raise ValueError(f'about {expected:,} tetrahedra at that size: use a larger one')
    ticker = Ticker(progress)
    ticker.add(MOST_STEPS + 3)

    def fd(points):
        return _distance(geometry, ids, points)

    low, high = _extent(geometry, ids)
    extent = float(np.linalg.norm(high - low))
    fixed = sharp_edges(geometry, ids, size)
    ticker.tick()
    points, nfixed = _seed(fd, fixed, low, high, size)
    ticker.tick()
    points, steps = _relax(fd, points, nfixed, size, extent, ticker)
    points, corners, quality, slivers = _tetrahedra(fd, points, nfixed, size, extent)
    if not len(corners):
        raise ValueError('the parts hold nothing to mesh')
    centroids = points[corners].mean(axis=1)
    groups = _owners(geometry, ids, centroids)
    corners, quality, groups = corners[groups > 0], quality[groups > 0], groups[groups > 0]
    if quadratic:
        points, conn = _quadratic(fd, _surfaces(geometry, ids, size), points, corners, size)
    else:
        conn = corners
    keep = np.unique(conn)
    renumber = np.zeros(len(points), dtype=int)
    renumber[keep] = np.arange(1, len(keep) + 1)
    edges = points[corners[:, [1, 2, 2, 3, 3, 3]]] - points[corners[:, [0, 0, 1, 0, 1, 2]]]
    shortest = float(np.linalg.norm(edges, axis=2).min()) if len(corners) else 0.0
    names, present = [], []
    for slot, part in enumerate(ids, start=1):
        if (groups == slot).any():
            item = solid(geometry, part)
            names.append(item.name.strip() or f'{item.kind.title()} {item.id}')
            present.append(slot)
    code = TET_CODES[conn.shape[1]]
    part = Geometry(node_id=np.arange(1, len(keep) + 1), node_xyz=points[keep],
                    elem_conn=list(renumber[conn]), elem_type=[code] * len(conn),
                    elem_group=[int(g) for g in groups], group_id=present,
                    group_name=names, length_unit=geometry.length_unit)
    ticker.tick()
    return part, {'tets': len(conn), 'nodes': len(keep), 'steps': steps,
                  'quality': float(quality.min()) if len(quality) else 0.0,
                  'poor': int((quality < 0.3).sum()), 'slivers': slivers,
                  'shortest': shortest}


def sharp_edges(geometry: Any, ids, size: float) -> np.ndarray:
    """The points `tet_mesh` holds fixed: the parts' corners, and points
    about `size` apart along every sharp edge of their surface.

    The surfaces are read from the recipe, each block's six planes and
    each cylinder's wall and caps, unbounded; a sharp edge is where two
    of them cross on the solid's surface, a corner where three do.
    Points seeded on one surface are moved onto the other and back
    until they sit on both, kept where the solid's distance is zero
    there, and thinned to the size.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id.
    size : float
        The mesh size: the spacing along an edge.

    Returns
    -------
    numpy.ndarray
        (n, 3), the corners first.
    """
    ids = [int(i) for i in ids]
    low, high = _extent(geometry, ids)
    extent = float(np.linalg.norm(high - low))
    tol = 1e-9 * extent

    def fd(points):
        return _distance(geometry, ids, points)

    surfaces = _surfaces(geometry, ids, size)
    fine = size / 3.0

    def crossing(which, seeds, rounds=30):
        """`seeds` moved onto every surface in `which` in turn until
        they lie on all, those on the solid's surface and within every
        surface's own extent kept."""
        q = seeds.copy()
        for _ in range(rounds):
            for k in which:
                q = _onto(surfaces[k][0], q, tol)
        on_all = np.all([np.abs(surfaces[k][0](q)) < tol for k in which], axis=0)
        on_all &= np.all([surfaces[k][4](q) for k in which], axis=0)
        on_all &= np.abs(fd(q)) < tol
        return q[on_all]

    curves, corners = [], []
    for i, (gi, seed_i, lo_i, hi_i, _within_i) in enumerate(surfaces):
        for j in range(i + 1, len(surfaces)):
            gj, _seed_j, lo_j, hi_j, _within_j = surfaces[j]
            lo, hi = np.maximum(lo_i, lo_j), np.minimum(hi_i, hi_j)
            if (hi < lo).any():
                continue
            seeds = seed_i(lo, hi, fine)
            if not len(seeds):
                continue
            seeds = seeds[np.abs(gj(seeds)) < 2.0 * size]
            if not len(seeds):
                continue
            found = crossing([i, j], seeds)
            if not len(found):
                continue
            curves.append(found)
            for k, (gk, _seed_k, lo_k, hi_k, _within_k) in enumerate(surfaces):
                if k in (i, j) or (hi_k < lo).any() or (lo_k > hi).any():
                    continue
                near = found[np.abs(gk(found)) < size]
                if len(near):
                    corners.append(crossing([i, j, k], near))
    kept: list[np.ndarray] = []
    if corners:
        for q in np.concatenate(corners):
            if not kept or cKDTree(np.array(kept)).query(q)[0] >= 0.25 * size:
                kept.append(q)
    for curve in curves:
        # along the curve, a point every 0.9 sizes of the way, and clear
        # of the corners and the curves before it by as much: one tree
        # over those, built once a curve
        others = cKDTree(np.array(kept)) if kept else None
        last = None
        for q in _along(curve):
            if last is not None and np.linalg.norm(q - last) < 0.9 * size:
                continue
            if others is not None and others.query(q)[0] < 0.9 * size:
                continue
            kept.append(q)
            last = q
    return np.array(kept).reshape(-1, 3)


# ---------------------------------------------------------------------------
# the recipe's surfaces

def _extent(geometry, ids):
    """The parts' bounding box, (low, high)."""
    boxes = [_box(geometry, i) for i in ids]
    return (np.min([lo for lo, _hi in boxes], axis=0),
            np.max([hi for _lo, hi in boxes], axis=0))


def _distance(geometry, ids, points):
    """The signed distance to the parts together: the least of theirs."""
    return np.min([distance(geometry, i, points) for i in ids], axis=0)


def _surfaces(geometry, ids, size):
    """(g, seed, low, high, within) for every surface the parts'
    recipes draw: `g(points)` its signed distance, unbounded; `seed(low,
    high, spacing)` points on it within a box; the box of the leaf it
    came from, grown by two sizes; and `within(points)`, which of them
    lie on the surface's own extent, a block face's rectangle, a
    cylinder's length or its cap's disk, a little beyond. A sharp
    edge is a crossing within both surfaces' extents: the extension of
    a hole's cap crossing a face is not one (it put a line of fixed
    points across a face, and a hole drilled at an angle gave the
    block's edges false corners). Coincident planes are one surface,
    within either's extent."""
    planes = []
    walls = []
    slack = 1e-6 * size
    for i in ids:
        for leaf in _leaves(geometry, i):
            center = np.asarray(leaf.center, dtype=float)
            frame = _frame(leaf)
            lo, hi = _basic_box(leaf)
            lo, hi = lo - 2.0 * size, hi + 2.0 * size
            if leaf.kind == 'block':
                half = np.asarray(leaf.widths, dtype=float) / 2.0
                for k in range(3):
                    others = [m for m in range(3) if m != k]
                    for sign in (-1.0, 1.0):
                        def within(q, center=center, frame=frame, others=others, half=half):
                            local = (q - center) @ frame.T
                            return np.all(np.abs(local[:, others]) <= half[others] + slack, axis=1)
                        planes.append((center + sign * frame[k] * half[k], sign * frame[k], lo, hi, within))
                if leaf.corner > 0.0:
                    wx, wy = half[0] - leaf.corner, half[1] - leaf.corner
                    for sx in (-1.0, 1.0):
                        for sy in (-1.0, 1.0):
                            def within(q, center=center, frame=frame, hz=half[2]):
                                return np.abs((q - center) @ frame[2]) <= hz + slack
                            walls.append((center + sx * wx * frame[0] + sy * wy * frame[1],
                                          frame[2], leaf.corner, lo, hi, within))
            else:
                def within(q, center=center, frame=frame, hl=leaf.length / 2.0):
                    return np.abs((q - center) @ frame[2]) <= hl + slack
                walls.append((center, frame[2], leaf.radius, lo, hi, within))
                for sign in (-1.0, 1.0):
                    def within(q, center=center, frame=frame, r=leaf.radius):
                        local = (q - center) @ frame.T
                        return np.hypot(local[:, 0], local[:, 1]) <= r + slack
                    planes.append((center + sign * frame[2] * leaf.length / 2.0,
                                   sign * frame[2], lo, hi, within))
    surfaces = []
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    for point, normal, lo, hi, within in planes:
        same = [k for k, (p2, n2) in enumerate(seen)
                if abs(abs(normal @ n2) - 1.0) < 1e-9 and abs((point - p2) @ n2) < 1e-9 * size]
        if same:
            # one plane, the box the union of both leaves' and the
            # extent either's
            g, seed, lo2, hi2, within2 = surfaces[same[0]]
            surfaces[same[0]] = (g, seed, np.minimum(lo, lo2), np.maximum(hi, hi2),
                                 lambda q, a=within, b=within2: a(q) | b(q))
            continue
        seen.append((point, normal))
        surfaces.append((*_plane(point, normal), lo, hi, within))
    for center, axis, radius, lo, hi, within in walls:
        surfaces.append((*_wall(center, axis, radius), lo, hi, within))
    return surfaces


def _across(axis):
    """Two unit vectors square to `axis` and to each other."""
    other = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, other)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def _plane(point, normal):
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    point = np.asarray(point, dtype=float)
    e1, e2 = _across(normal)

    def g(q):
        return (q - point) @ normal

    def seed(lo, hi, spacing):
        # the box's corners, in the plane's coordinates, bound the grid
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])]) - point
        u, v = corners @ e1, corners @ e2
        gu, gv = np.meshgrid(np.arange(u.min(), u.max() + spacing, spacing),
                             np.arange(v.min(), v.max() + spacing, spacing), indexing='ij')
        q = point + gu.ravel()[:, None] * e1 + gv.ravel()[:, None] * e2
        return q[np.all((q >= lo) & (q <= hi), axis=1)]
    return g, seed


def _wall(center, axis, radius):
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    e1, e2 = _across(axis)

    def g(q):
        rel = q - center
        along = rel @ axis
        return np.linalg.norm(rel - along[:, None] * axis, axis=1) - radius

    def seed(lo, hi, spacing):
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])]) - center
        along = corners @ axis
        t = np.arange(0.0, 2.0 * np.pi, spacing / radius)
        gt, gs = np.meshgrid(t, np.arange(along.min(), along.max() + spacing, spacing),
                             indexing='ij')
        q = (center + radius * (np.cos(gt.ravel())[:, None] * e1 + np.sin(gt.ravel())[:, None] * e2)
             + gs.ravel()[:, None] * axis)
        return q[np.all((q >= lo) & (q <= hi), axis=1)]
    return g, seed


def _gradient(g, q, step):
    return np.stack([(g(q + np.eye(3)[k] * step) - g(q)) / step for k in range(3)], axis=1)


def _onto(g, q, tol):
    """`q` moved onto the surface `g` = 0 along its gradient, once."""
    value = g(q)
    grad = _gradient(g, q, max(tol, 1e-12))
    return q - (value / np.maximum((grad ** 2).sum(axis=1), 1e-30))[:, None] * grad


def _along(curve):
    """A curve's points in order along it: each next the nearest not
    yet taken, so thinning to a spacing keeps them evenly spread."""
    if len(curve) < 3:
        return curve
    tree = cKDTree(curve)
    order, left = [0], set(range(1, len(curve)))
    while left:
        _d, near = tree.query(curve[order[-1]], k=min(len(curve), 12))
        following = next((int(k) for k in np.atleast_1d(near) if int(k) in left), None)
        if following is None:
            following = min(left, key=lambda k: float(np.linalg.norm(curve[k] - curve[order[-1]])))
        order.append(following)
        left.discard(following)
    return curve[order]


# ---------------------------------------------------------------------------
# the relaxation

def _seed(fd, fixed, low, high, size):
    """The starting points: the fixed ones, then a body-centered lattice
    inside the solid, cleared around the fixed ones."""
    rail = [np.arange(a - size, b + 2.0 * size, size) for a, b in zip(low, high)]
    lattice = np.stack(np.meshgrid(*rail, indexing='ij'), axis=-1).reshape(-1, 3)
    lattice = np.concatenate([lattice, lattice + size / 2.0])
    inside = lattice[fd(lattice) < -0.001 * size]
    if len(fixed):
        near = cKDTree(fixed).query(inside, distance_upper_bound=0.6 * size)[0] < np.inf
        inside = inside[~near]
    return np.concatenate([fixed.reshape(-1, 3), inside]), len(fixed)


def _bars(fd, points, size):
    """The Delaunay triangulation's tetrahedra inside the solid, and
    their edges, each once."""
    tets = Delaunay(points).simplices
    tets = tets[fd(points[tets].mean(axis=1)) < -0.001 * size]
    return tets, _edges(tets, len(points))[0]


def _edges(tets, count):
    """Each edge of the tetrahedra once, as (low, high) node pairs, and
    which edge each of the six of each tetrahedron is. Sorted as one
    integer a pair: a sort of the pairs by rows took forty percent of
    the meshing (2026-10-19)."""
    pairs = np.concatenate([tets[:, [a, b]] for a, b in TET10_EDGES])
    keys = (np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64) * count
            + np.maximum(pairs[:, 0], pairs[:, 1]))
    unique, inverse = np.unique(keys, return_inverse=True)
    return np.stack([unique // count, unique % count], axis=1), inverse


def _relax(fd, points, nfixed, size, extent, ticker):
    """Points pushed apart along short edges until they settle."""
    p = points.copy()
    last = np.full_like(p, np.inf)
    small = np.sqrt(np.finfo(float).eps) * size
    geps = 0.001 * size
    bars = None
    steps = 0
    for steps in range(1, MOST_STEPS + 1):
        if np.sqrt(((p - last) ** 2).sum(axis=1)).max() > RETRIANGULATE * size:
            last = p.copy()
            _tets, bars = _bars(fd, p, size)
        vector = p[bars[:, 0]] - p[bars[:, 1]]
        length = np.linalg.norm(vector, axis=1)
        aimed = SCALE * float(np.mean(length ** 3)) ** (1.0 / 3.0)
        push = np.maximum(aimed - length, 0.0) / np.maximum(length, 1e-30)
        force = np.zeros_like(p)
        np.add.at(force, bars[:, 0], push[:, None] * vector)
        np.add.at(force, bars[:, 1], -push[:, None] * vector)
        force[:nfixed] = 0.0
        p += STEP * force
        d = fd(p)
        out = d > 0.0
        if out.any():
            grad = _gradient(fd, p[out], small)
            p[out] -= (d[out] / np.maximum((grad ** 2).sum(axis=1), 1e-30))[:, None] * grad
        interior = d < -geps
        interior[:nfixed] = False
        moved = (np.linalg.norm(STEP * force[interior], axis=1).max() / size
                 if interior.any() else 0.0)
        ticker.tick()
        if moved < SETTLED:
            break
    ticker.tick(MOST_STEPS - steps)
    # one step per round leaves a point a little off the surface; the
    # ones near it are finished to it, so a wall is a wall to rounding
    near = np.abs(fd(p)) < 0.1 * size
    near[:nfixed] = False
    if near.any():
        q = p[near]
        for _ in range(4):
            q = _onto(fd, q, small)
        p[near] = q
    return p, steps


def _quality(points, tets):
    """The radius ratio of each tetrahedron: three times the inscribed
    radius over the circumscribed, 1 for a regular one, 0 for a flat
    one; and the signed volumes."""
    a, b, c, d = (points[tets[:, k]] for k in range(4))
    volume = np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a)) / 6.0
    size = np.abs(volume)

    def area(x, y, z):
        return 0.5 * np.linalg.norm(np.cross(y - x, z - x), axis=1)

    faces = area(a, b, c) + area(a, b, d) + area(a, c, d) + area(b, c, d)
    inner = 3.0 * size / np.maximum(faces, 1e-300)
    # the circumradius from the products of opposite edges
    pa = np.linalg.norm(b - a, axis=1) * np.linalg.norm(d - c, axis=1)
    pb = np.linalg.norm(c - a, axis=1) * np.linalg.norm(d - b, axis=1)
    pc = np.linalg.norm(d - a, axis=1) * np.linalg.norm(c - b, axis=1)
    s = (pa + pb + pc) / 2.0
    outer = np.sqrt(np.maximum(s * (s - pa) * (s - pb) * (s - pc), 0.0)) / np.maximum(6.0 * size, 1e-300)
    return np.where(size > 0.0, 3.0 * inner / np.maximum(outer, 1e-300), 0.0), volume


def _tetrahedra(fd, points, nfixed, size, extent):
    """The final tetrahedra, numbered the right way out, the slivers
    dealt with: (points, corners, quality, slivers removed).

    A Delaunay triangulation of well-spread points still holds slivers,
    flat tetrahedra whose four corners lie almost in a plane. Each
    round, every corner of a sliver that is free to move goes to the
    middle of its neighbors (back onto the surface when it was there),
    and the triangulation is made again; from the fourth round the
    corners are nudged a little at random instead, since a sliver's
    corners can sit in a plane with their neighbors' middles too. A
    sliver whose four corners all lie on the surface lies in it, and is
    dropped, which leaves the surface closed and the rest untouched."""
    p = points.copy()
    small = np.sqrt(np.finfo(float).eps) * size
    nudge = np.random.default_rng(0)
    for round_ in range(SLIVER_ROUNDS):
        tets, bars = _bars(fd, p, size)
        quality, _volume = _quality(p, tets)
        on_surface = np.abs(fd(p)) < 1e-6 * size
        slivers = tets[quality < SLIVER]
        loose = np.unique(slivers[~on_surface[slivers].all(axis=1)])
        loose = loose[loose >= nfixed]
        if not len(loose):
            break
        neighbors = np.zeros_like(p)
        count = np.zeros(len(p))
        np.add.at(neighbors, bars[:, 0], p[bars[:, 1]])
        np.add.at(neighbors, bars[:, 1], p[bars[:, 0]])
        np.add.at(count, bars[:, 0], 1.0)
        np.add.at(count, bars[:, 1], 1.0)
        was_on = on_surface[loose]
        if round_ < 3:
            p[loose] = neighbors[loose] / np.maximum(count[loose], 1.0)[:, None]
        else:
            p[loose] += nudge.normal(scale=0.15 * size, size=(len(loose), 3))
        d = fd(p[loose])
        back = was_on | (d > 0.0)
        q = p[loose[back]]
        for _ in range(3):
            q = _onto(fd, q, small)
        p[loose[back]] = q
    # no node outside the solid: one left a hair out of a wall where
    # two surfaces meet, where a Newton step converges slowly
    out = np.flatnonzero(fd(p) > 0.0)
    out = out[out >= nfixed]
    if len(out):
        q = p[out]
        for _ in range(8):
            q = _onto(fd, q, small)
        p[out] = q
    tets, _ = _bars(fd, p, size)
    quality, volume = _quality(p, tets)
    flipped = volume < 0.0
    tets[flipped] = tets[flipped][:, [0, 2, 1, 3]]
    # a sliver hugging the surface, every corner within a fifth of a
    # size of it, lies in the surface's skin: dropped, which leaves a
    # dimple no deeper than the sliver was. On a convex wall (a shaft's)
    # the cap over four surface points has its centroid inside, and
    # such caps were a seventh of the tetrahedra under 0.3 and the
    # worst at zero (2026-10-19)
    hugging = np.abs(fd(p)) < 0.2 * size
    sliver = (quality < SLIVER) & hugging[tets].all(axis=1)
    return p, tets[~sliver], quality[~sliver], int(sliver.sum())


def _quadratic(fd, surfaces, points, corners, size):
    """Ten-node connectivity: a node at the middle of every edge, the
    ones on a curved wall moved onto it. An edge is on a wall when both
    its ends lie on that surface of the recipe; the middle is moved
    onto every surface both ends share, and kept there when that
    leaves it on the solid's surface and moves it less than a third of
    the size. Across a sharp edge, or between two faces, the midside
    stays where it is."""
    unique, inverse = _edges(corners, len(points))
    middle = (points[unique[:, 0]] + points[unique[:, 1]]) / 2.0
    tol = 1e-6 * size
    on_surface = np.abs(fd(points)) < tol
    both = np.flatnonzero(on_surface[unique].all(axis=1))
    if len(both):
        ends = unique[both]
        shared = np.zeros((len(both), len(surfaces)), dtype=bool)
        for k, (g, _seed, _lo, _hi, _within) in enumerate(surfaces):
            shared[:, k] = ((np.abs(g(points[ends[:, 0]])) < tol)
                            & (np.abs(g(points[ends[:, 1]])) < tol))
        moved = middle[both].copy()
        for k, (g, _seed, _lo, _hi, _within) in enumerate(surfaces):
            which = np.flatnonzero(shared[:, k])
            if not len(which):
                continue
            q = moved[which]
            for _ in range(4):
                q = _onto(g, q, tol)
            moved[which] = q
        # a second surface shared (the rim where a wall meets a face):
        # back onto each in turn until on both
        twice = np.flatnonzero(shared.sum(axis=1) > 1)
        for _ in range(6):
            for k, (g, _seed, _lo, _hi, _within) in enumerate(surfaces):
                which = twice[shared[twice, k]]
                if len(which):
                    moved[which] = _onto(g, moved[which], tol)
        keep = ((np.abs(fd(moved)) < tol)
                & (np.linalg.norm(moved - middle[both], axis=1) < size / 3.0))
        middle[both[keep]] = moved[keep]
    # a chord from a wall node to one just inside dips out of the solid
    # where the wall is concave: its middle goes onto the surface
    out = np.flatnonzero(fd(middle) > tol)
    if len(out):
        q = middle[out]
        for _ in range(4):
            q = _onto(fd, q, tol)
        keep = ((np.abs(fd(q)) < tol)
                & (np.linalg.norm(q - middle[out], axis=1) < size / 3.0))
        middle[out[keep]] = q[keep]
    midside = len(points) + inverse.reshape(len(TET10_EDGES), len(corners)).T
    return np.concatenate([points, middle]), np.concatenate([corners, midside], axis=1)
