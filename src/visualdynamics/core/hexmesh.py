"""Brick meshing of any solid, over a grid fitted to its shape.

Mesh Solids meshes an extrusion into bricks on a grid with rings about
its holes (`solids.extrusion_mesh`). A part that is not an extrusion
is meshed here: a grid of cubes over it, the cubes whose centers are
inside kept, and the stair-stepped skin that leaves fitted to the
true surface. Nothing about the shape is assumed beyond its signed
distance (`solids.distance`) and the sharp edges read from its recipe
(`tetmesh.sharp_edges`).

The fitting is done in an order that keeps every brick the right way
out. The skin is pillowed first: every surface node gets a twin a
little way in, so each brick under the surface touches it by one face
only, and a layer of new bricks lies between. The corners and sharp
edges are captured by the nearest skin nodes, which are moved onto
them and held; the skin between is smoothed so it runs nearly parallel
to the surface it is about to meet, and then every skin node is
projected onto the surface along the distance's gradient, in two
stages, the inside smoothed after each. Last, every node takes the
position among a few candidates that raises the worst scaled Jacobian
of the bricks around it, surface nodes sliding on the surface, until
no node moves.

The result has round holes and sharp edges whatever the shape, with
bricks of lower quality than a ring mesh gives: the worst scaled
Jacobian is typically between 0.15 and 0.3, and the pane reads it.
Where the parts are an extrusion, `extrusion_mesh` is the better mesh
and is used instead (`brick_mesh`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .geometry import _CORNER_EDGES, Geometry
from .progress import Ticker
from .solids import (
    MESH_LIMIT,
    _cuts,
    _owners,
    solid,
    used_by,
)
from .tetmesh import _distance, _extent, _onto, _surfaces, sharp_edges

#: how far in a surface node's twin sits, as a share of the size
PILLOW_DEPTH = 0.3
#: how far a corner or an edge point may reach for its node, in sizes
CAPTURE = 1.0
#: Laplacian passes over the skin before it is projected
SKIN_PASSES = 10
#: the projection, in stages, the inside smoothed after each. Measured
#: on a 17,000-brick fit: one stage leaves the worst scaled Jacobian at
#: 0.16, two at 0.19 to 0.21, four at 0.19 for a third more time
STAGES = 2
STAGE_SWEEPS = 5
#: the final smoothing's most sweeps, and the share of the movable
#: nodes still moving below which it stops
FINAL_SWEEPS = 20
SETTLED = 0.002
#: a node whose worst scaled Jacobian is under this tries steps along
#: the axes too
TANGLED = 0.15

_EDGES8 = np.asarray(_CORNER_EDGES[8])
_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
_HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def overlay_mesh(geometry: Any, ids, size: float, progress: Any = None) -> tuple[Any, dict]:
    """The parts as eight-node bricks fitted to their shape, ready to
    join to a geometry (`mesh.join`).

    Each part gets one element group, named for it (or for its kind and
    id when it has no name); a brick goes to the part whose recipe holds
    its center, the later part where parts overlap. Parts that touch
    share their nodes there.

    Refused: a solid that is part of another, a size that is not
    positive, and a grid of more than `MESH_LIMIT` cells.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id; each a solid nothing else uses.
    size : float
        The element size, as the geometry holds its coordinates: the
        grid's spacing, the block faces falling on grid planes.
    progress : callable, optional
        Told ``(done, total)`` as the mesh goes.

    Returns
    -------
    (Geometry, dict)
        The bricks, in the geometry's length unit; and what was made:
        'bricks' and 'nodes', the counts; 'quality', the worst scaled
        Jacobian (1 for a cube, 0 for a flat brick); 'poor', how many
        bricks are under 0.3; 'uncaptured', how many points along the
        sharp edges found no node of their own; 'sweeps', the final
        smoothing's; and 'shortest', the shortest edge.
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
    ticker = Ticker(progress)
    ticker.add(3 + STAGES + FINAL_SWEEPS + 1)

    def fd(points):
        return _distance(geometry, ids, points)

    low, high = _extent(geometry, ids)
    tol = 1e-9 * float(np.linalg.norm(high - low))
    xyz, hexes, groups = _grid(geometry, ids, size)
    if not len(hexes):
        raise ValueError('the parts hold nothing to mesh')
    ticker.tick()
    xyz, hexes, groups, faces, face_hex, surface = _pillow(xyz, hexes, groups, size)
    ticker.tick()
    mesh = _Mesh(xyz, hexes, faces, face_hex, surface, fd, size, tol)
    uncaptured = mesh.capture(geometry, ids)
    mesh.smooth_skin()
    mesh.aim()
    ticker.tick()
    start = mesh.xyz.copy()
    for stage in range(1, STAGES + 1):
        mesh.xyz[surface] = start[surface] + stage / STAGES * (mesh.target[surface] - start[surface])
        mesh.smooth(STAGE_SWEEPS, slide=False)
        ticker.tick()
    sweeps = mesh.smooth(FINAL_SWEEPS, slide=True, ticker=ticker)
    ticker.tick(FINAL_SWEEPS - sweeps)
    mesh.finish()
    xyz = mesh.xyz
    quality = _scaled_jacobians(xyz, hexes).min(axis=1)
    folded = int((quality <= 0.0).sum())
    if folded:
        # a feature two bricks across or less leaves no room to fit
        # (the 1 in side hole at 0.5 in): refused, not handed over
        raise ValueError(f'{folded:,} bricks folded fitting the shape at that size: '
                         'use a smaller one, or tetrahedra')
    keep = np.unique(hexes)
    renumber = np.zeros(len(xyz), dtype=int)
    renumber[keep] = np.arange(1, len(keep) + 1)
    shortest = float(np.linalg.norm(xyz[hexes[:, [b for _a, b in _HEX_EDGES]]]
                                    - xyz[hexes[:, [a for a, _b in _HEX_EDGES]]], axis=2).min())
    names, present = [], []
    for slot, part in enumerate(ids, start=1):
        if (groups == slot).any():
            item = solid(geometry, part)
            names.append(item.name.strip() or f'{item.kind.title()} {item.id}')
            present.append(slot)
    part = Geometry(node_id=np.arange(1, len(keep) + 1), node_xyz=xyz[keep],
                    elem_conn=list(renumber[hexes]), elem_type=[115] * len(hexes),
                    elem_group=[int(g) for g in groups], group_id=present,
                    group_name=names, length_unit=geometry.length_unit)
    ticker.tick()
    return part, {'bricks': len(hexes), 'nodes': len(keep), 'quality': float(quality.min()),
                  'poor': int((quality < 0.3).sum()), 'uncaptured': uncaptured,
                  'sweeps': sweeps, 'shortest': shortest}


def grid_cells(geometry: Any, ids, size: float) -> int:
    """How many cells the grid `overlay_mesh` lays over these parts at
    `size` holds, voids included: the count before anything is asked
    of the shape, so a pane can decide whether to preview.

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
    return int(np.prod([len(line) - 1 for line in _lines(geometry, [int(i) for i in ids], float(size))]))


# ---------------------------------------------------------------------------
# the grid and the pillow

def _lines(geometry, ids, size):
    """The grid's lines along each axis: every block face on one, each
    span divided into the whole number of cells nearest `size`."""
    cuts = _cuts(geometry, ids)
    lines = []
    for k in range(3):
        edges = [cuts[k][0]]
        for a, b in zip(cuts[k][:-1], cuts[k][1:]):
            count = max(1, round((b - a) / size))
            edges.extend(np.linspace(a, b, count + 1)[1:])
        lines.append(np.asarray(edges))
    return lines


def _grid(geometry, ids, size):
    """The cubes over the parts whose centers a part holds: (nodes,
    bricks, each brick's part). The grid's lines fall on every block
    face, as the extrusion mesher's do."""
    lines = _lines(geometry, ids, size)
    nx, ny, nz = (len(line) for line in lines)
    if (nx - 1) * (ny - 1) * (nz - 1) > MESH_LIMIT:
        raise ValueError(f'{(nx - 1) * (ny - 1) * (nz - 1):,} bricks at that size: use a larger one')
    grid = np.stack(np.meshgrid(*lines, indexing='ij'), axis=-1).reshape(-1, 3)
    node = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    mids = [(line[:-1] + line[1:]) / 2.0 for line in lines]
    centers = np.stack(np.meshgrid(*mids, indexing='ij'), axis=-1).reshape(-1, 3)
    owner = _owners(geometry, ids, centers).reshape(nx - 1, ny - 1, nz - 1)
    i, j, k = np.nonzero(owner > 0)
    hexes = np.stack([node[i, j, k], node[i + 1, j, k], node[i + 1, j + 1, k], node[i, j + 1, k],
                      node[i, j, k + 1], node[i + 1, j, k + 1], node[i + 1, j + 1, k + 1],
                      node[i, j + 1, k + 1]], axis=1)
    used = np.unique(hexes)
    renumber = -np.ones(len(grid), dtype=int)
    renumber[used] = np.arange(len(used))
    return grid[used], renumber[hexes], owner[i, j, k]


def _boundary(hexes):
    """The faces no two bricks share, and the brick each belongs to."""
    faces = np.concatenate([hexes[:, list(f)] for f in _FACES])
    owner = np.concatenate([np.arange(len(hexes))] * 6)
    key = np.sort(faces, axis=1)
    _unique, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    pick = first[counts == 1]
    return faces[pick], owner[pick]


def _outward(xyz, hexes, faces, face_hex):
    """Each boundary face's unit normal, pointing out of its brick."""
    a, b, c, d = (xyz[faces[:, m]] for m in range(4))
    normal = np.cross(c - a, d - b)
    normal /= np.linalg.norm(normal, axis=1)[:, None] + 1e-30
    away = (a + b + c + d) / 4.0 - xyz[hexes[face_hex]].mean(axis=1)
    return normal * np.sign(np.einsum('ij,ij->i', normal, away))[:, None]


def _pillow(xyz, hexes, groups, size):
    """A layer of bricks under the skin: each surface node gets a twin
    `PILLOW_DEPTH` sizes in along the mean of its faces' inward
    normals, the bricks that used it use the twin, and a new brick
    joins each boundary face to its twins. Every brick is then
    numbered right-handed."""
    faces, face_hex = _boundary(hexes)
    surface = np.unique(faces)
    normal = _outward(xyz, hexes, faces, face_hex)
    mean = np.zeros_like(xyz)
    for m in range(4):
        np.add.at(mean, faces[:, m], normal)
    mean = mean[surface] / (np.linalg.norm(mean[surface], axis=1)[:, None] + 1e-30)
    twin_start = len(xyz)
    xyz = np.vstack([xyz, xyz[surface] - PILLOW_DEPTH * size * mean])
    remap = np.arange(len(xyz))
    remap[surface] = twin_start + np.arange(len(surface))
    pillow = np.concatenate([faces, remap[faces]], axis=1)
    hexes = np.concatenate([remap[hexes], pillow])
    groups = np.concatenate([groups, groups[face_hex]])
    volume = _corner_volumes(xyz, hexes).sum(axis=1)
    flipped = volume < 0.0
    hexes[flipped] = hexes[flipped][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    faces, face_hex = _boundary(hexes)
    return xyz, hexes, groups, faces, face_hex, surface


# ---------------------------------------------------------------------------
# quality

def _corner_volumes(xyz, hexes):
    points = xyz[hexes]
    legs = points[:, _EDGES8] - points[:, :, None, :]
    return np.einsum('mci,mci->mc', legs[:, :, 0], np.cross(legs[:, :, 1], legs[:, :, 2]))


def _scaled_jacobians(xyz, hexes):
    """Each corner's volume over the product of its three edge lengths:
    1 at a cube's corner, 0 where it is flat, negative inside out."""
    points = xyz[hexes]
    legs = points[:, _EDGES8] - points[:, :, None, :]
    volume = np.einsum('mci,mci->mc', legs[:, :, 0], np.cross(legs[:, :, 1], legs[:, :, 2]))
    return volume / np.maximum(np.linalg.norm(legs, axis=3).prod(axis=2), 1e-300)


# ---------------------------------------------------------------------------
# the fitting

class _Mesh:
    """The bricks being fitted: nodes, their kinds (0 inside, 1 on the
    surface, 2 on a sharp edge, 3 at a corner), the targets the surface
    nodes are moved to, and the smoothing."""

    def __init__(self, xyz, hexes, faces, face_hex, surface, fd, size, tol):
        self.xyz: np.ndarray = xyz
        self.hexes: np.ndarray = hexes
        self.faces: np.ndarray = faces
        self.face_hex: np.ndarray = face_hex
        self.surface: np.ndarray = surface
        self.fd: Any = fd
        self.size: float = size
        self.tol: float = tol
        self.kind: np.ndarray = np.zeros(len(xyz), dtype=int)
        self.kind[surface] = 1
        self.target: np.ndarray = xyz.copy()
        self.curve_of: dict[int, np.ndarray] = {}
        self.surfaces: list = []
        pairs = np.concatenate([hexes[:, [a, b]] for a, b in _HEX_EDGES])
        self.pairs: np.ndarray = np.unique(np.sort(pairs, axis=1), axis=0)
        skin = np.concatenate([faces[:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))])
        self.skin_pairs: np.ndarray = np.unique(np.sort(skin, axis=1), axis=0)
        self.hexes_of: list[np.ndarray] = [np.zeros(0, dtype=int)] * len(xyz)
        lists: list[list[int]] = [[] for _ in range(len(xyz))]
        for c, hx in enumerate(hexes):
            for n in hx:
                lists[n].append(c)
        self.hexes_of = [np.array(v, dtype=int) for v in lists]
        # a coloring: nodes of one brick take different colors, so a
        # color's nodes can move together and each be scored alone
        neighbors: list[set] = [set() for _ in range(len(xyz))]
        for hx in hexes:
            for n in hx:
                neighbors[n].update(int(m) for m in hx if m != n)
        color = -np.ones(len(xyz), dtype=int)
        for n in np.argsort(-np.array([len(v) for v in neighbors])):
            taken = {color[m] for m in neighbors[n] if color[m] >= 0}
            c = 0
            while c in taken:
                c += 1
            color[n] = c
        self.color: np.ndarray = color

    # -- targets -----------------------------------------------------------
    def project(self, points, steps=8):
        """Onto the surface along the gradient, each step at most half
        a size: a Newton step from a kink of the distance jumps far."""
        q = points.copy()
        for _ in range(steps):
            moved = _onto(self.fd, q, self.tol)
            step = moved - q
            length = np.linalg.norm(step, axis=1)
            far = length > 0.5 * self.size
            step[far] *= (0.5 * self.size / length[far])[:, None]
            q = q + step
        return q

    def onto_curves(self, nodes, points, rounds=6):
        """`points`, one per edge node in `nodes`, onto the two surfaces
        each lies on; the nodes on one pair of surfaces together."""
        q = points.copy()
        by_pair: dict[tuple, list[int]] = {}
        for idx, n in enumerate(nodes):
            by_pair.setdefault(tuple(self.curve_of[int(n)]), []).append(idx)
        for pair, members in by_pair.items():
            members = np.array(members)
            for _ in range(rounds):
                for s in pair:
                    q[members] = _onto(self.surfaces[s][0], q[members], self.tol)
        return q

    def capture(self, geometry, ids) -> int:
        """Corners and points along the sharp edges take the nearest
        free skin node each, within `CAPTURE` sizes, and hold it; how
        many points found none."""
        size = self.size
        points = sharp_edges(geometry, ids, size)
        self.surfaces = _surfaces(geometry, ids, size)
        on = np.stack([np.abs(g(points)) < 1e-6 * size for g, *_rest in self.surfaces], axis=1) \
            if len(points) else np.zeros((0, len(self.surfaces)), dtype=bool)
        count = on.sum(axis=1)
        tree = cKDTree(self.xyz[self.surface])
        claimed: set[int] = set()
        missed = 0
        for p, shared in zip(points[count >= 3], on[count >= 3]):
            for dist, idx in zip(*tree.query(p, k=min(12, len(self.surface)))):
                n = int(self.surface[idx])
                if n not in claimed and dist < 1.5 * size:
                    claimed.add(n)
                    self.kind[n] = 3
                    self.target[n] = self.xyz[n] = p
                    break
        for p, shared in zip(points[count == 2], on[count == 2]):
            for dist, idx in zip(*tree.query(p, k=min(12, len(self.surface)))):
                n = int(self.surface[idx])
                if n not in claimed and dist < CAPTURE * size:
                    claimed.add(n)
                    self.kind[n] = 2
                    self.target[n] = self.xyz[n] = p
                    self.curve_of[n] = np.flatnonzero(shared)
                    break
            else:
                missed += 1
        return missed

    def smooth_skin(self):
        """The skin smoothed between the captured edges, which hold, and
        the inside after it, so the skin runs nearly parallel to the
        surface before it is projected: projected from the stair steps,
        neighbors crossed."""
        xyz, pairs, skin_pairs = self.xyz, self.pairs, self.skin_pairs
        on_skin = self.kind >= 1
        held = self.kind >= 2
        for _ in range(SKIN_PASSES):
            along = _means(xyz, skin_pairs)
            inside = _means(xyz, pairs)
            new = xyz.copy()
            new[on_skin] = along[on_skin]
            new[~on_skin] = inside[~on_skin]
            new[held] = self.target[held]
            xyz[:] = new

    def aim(self):
        """Where each plain surface node goes: its projection onto the
        surface. A capture whose face would then turn into its brick is
        released, and its node goes to the surface like the rest."""
        plain = self.kind == 1
        self.target[plain] = self.project(self.xyz[plain])
        for _repair in range(5):
            twisted = self.twisted(self.target)
            culprits = np.unique(self.faces[twisted])
            culprits = culprits[self.kind[culprits] == 2]
            if not len(culprits):
                break
            self.kind[culprits] = 1
            self.target[culprits] = self.project(self.xyz[culprits])

    def twisted(self, points):
        """Boundary faces with a triangle facing into their brick."""
        a, b, c, d = (points[self.faces[:, m]] for m in range(4))
        away = (a + b + c + d) / 4.0 - points[self.hexes[self.face_hex]].mean(axis=1)
        turned = np.zeros(len(self.faces), dtype=bool)
        for x, y, z in ((a, b, c), (a, c, d), (a, b, d), (b, c, d)):
            turned |= np.einsum('ij,ij->i', np.cross(y - x, z - x), away) < 0.0
        return turned

    def finish(self):
        """Every surface node exactly on the surface, and every edge node
        on its edge: the smoothing's two-step projections leave a node
        a few thousandths of a size off. A move that would fold a
        brick is not made."""
        xyz, kind = self.xyz, self.kind
        plain, curve = np.flatnonzero(kind == 1), np.flatnonzero(kind == 2)
        new = xyz.copy()
        new[plain] = self.project(xyz[plain])
        if len(curve):
            new[curve] = self.onto_curves(curve, xyz[curve])
        before = _scaled_jacobians(xyz, self.hexes).min(axis=1)
        after = _scaled_jacobians(new, self.hexes).min(axis=1)
        folded = np.unique(self.hexes[(after <= 0.0) & (before > 0.0)])
        new[folded] = xyz[folded]
        xyz[:] = new

    # -- smoothing ---------------------------------------------------------
    def scores(self, points, nodes, cells):
        """Each of `nodes`' worst scaled Jacobian among its bricks,
        `cells` being every brick any of them is a corner of."""
        worst = _scaled_jacobians(points, self.hexes[cells]).min(axis=1)
        score = np.full(len(points), np.inf)
        np.minimum.at(score, self.hexes[cells].ravel(), np.repeat(worst, 8))
        return score[nodes]

    def smooth(self, sweeps, slide, ticker=None) -> int:
        """Each node takes, among a few candidate positions, the one
        that raises the worst scaled Jacobian of its bricks: toward the
        middle of its neighbors the whole way, half or a quarter, and
        for a tangled node small steps along the axes. With `slide`,
        surface nodes move too, along the surface; without, they hold
        where the stage put them. Sweeps until no node moves, or few."""
        xyz, kind, color = self.xyz, self.kind, self.color
        movable = np.flatnonzero(kind <= (2 if slide else 0))
        plain = np.flatnonzero(kind == 1)
        curve = np.flatnonzero(kind == 2)
        groups = [movable[color[movable] == c] for c in range(color.max() + 1)]
        cells = [np.unique(np.concatenate([self.hexes_of[n] for n in group]))
                 if len(group) else np.zeros(0, dtype=int) for group in groups]
        sweep = 0
        for sweep in range(1, sweeps + 1):
            want = xyz.copy()
            inside = kind == 0
            means = _means(xyz, self.pairs)
            want[inside] = means[inside]
            if slide:
                along = _means(xyz, self.skin_pairs)
                want[plain] = self.project(along[plain])
                want[curve] = self.onto_curves(curve, along[curve])
            changed = 0
            for group, attached in zip(groups, cells):
                if not len(group):
                    continue
                current = self.scores(xyz, group, attached)
                best, best_score = xyz[group].copy(), current.copy()
                candidates = [xyz[group] + share * (want[group] - xyz[group])
                              for share in (1.0, 0.5, 0.25)]
                tangled = current < TANGLED
                if tangled.any():
                    for axis in range(3):
                        for sign in (-1.0, 1.0):
                            step = np.zeros_like(candidates[0])
                            step[tangled, axis] = sign * 0.1 * self.size
                            candidates.append(xyz[group] + step)
                for moved in candidates:
                    moved = moved.copy()
                    if slide:
                        # a candidate is near the surface already: two
                        # steps put it back on
                        on_plain = kind[group] == 1
                        if on_plain.any():
                            moved[on_plain] = self.project(moved[on_plain], steps=2)
                        on_curve = np.flatnonzero(kind[group] == 2)
                        if len(on_curve):
                            moved[on_curve] = self.onto_curves(group[on_curve], moved[on_curve], rounds=2)
                    trial = xyz.copy()
                    trial[group] = moved
                    score = self.scores(trial, group, attached)
                    better = score > best_score + 1e-12
                    best[better], best_score[better] = moved[better], score[better]
                gained = best_score > current + 1e-12
                xyz[group[gained]] = best[gained]
                changed += int(gained.sum())
            if ticker is not None:
                ticker.tick()
            if changed <= SETTLED * len(movable):
                break
        return sweep


def _means(xyz, pairs):
    """Each node's neighbors' mean over `pairs`; its own position where
    it has none."""
    total = np.zeros_like(xyz)
    count = np.zeros(len(xyz))
    np.add.at(total, pairs[:, 0], xyz[pairs[:, 1]])
    np.add.at(total, pairs[:, 1], xyz[pairs[:, 0]])
    np.add.at(count, pairs[:, 0], 1.0)
    np.add.at(count, pairs[:, 1], 1.0)
    alone = count == 0
    total[alone] = xyz[alone]
    count[alone] = 1.0
    return total / count[:, None]
