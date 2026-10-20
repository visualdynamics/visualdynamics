"""Mated surfaces: where element groups' meshes lie on each other, and
how they are joined there (Mate Surfaces).

A mated set is a definition the geometry holds, not elements: two
element groups, the nodes on each side of the patch where they meet,
and how the sides are joined, rigidly or by springs with a stiffness
normal and shear to the surface. The links are made when a model is
built from the geometry (`fem.Model.from_geometry`), each node of the
first side joined to the nearest node of the second, so a set can be
edited or deleted at any time and leaves nothing behind.

`detect` finds the patches: the nodes of one group's outer faces that
lie on another's outer faces, within a tolerance, split into the
pieces their faces connect. The side with more nodes on the patch, the
finer mesh, is the first side and follows the second.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

# Brandon, 2026-10-20: Mate Surfaces acted on the whole geometry and
# made joint element groups of links. He asked for it on element groups
# picked in the tree, a patch at a time, its nodes picked by hand, and
# kept as a sub-object of the geometry that can be edited or deleted.
# Sharing nodes was dropped: Merge Coincident Nodes does that. The same
# morning: a normal and a shear setting, each rigid, a stiffness or
# free, and on curved surfaces, so each node is tied in a frame of its
# own (`fem.Model.add_tie`).

#: how each of the normal and the shear is held
MODES = ('rigid', 'stiffness', 'free')


@dataclass(frozen=True)
class MatedSet:
    """Two element groups joined where their surfaces mate.

    Each node of the first side is tied to the nearest node of the
    second, in a frame of its own: along the normal of the second side's
    surface where the node lies on it, and across it, in the surface's
    plane (shear). Each of the two is held rigidly, by a spring, or not
    at all.

    Attributes
    ----------
    id : int
        The set's number, unique in its geometry.
    name : str
        What the tree calls it.
    groups : (int, int)
        The element groups, by id: the first side's and the second's.
    nodes_a : tuple of int
        The first side's nodes, which follow the second.
    nodes_b : tuple of int
        The second side's nodes, which the first side's are tied to.
    normal_mode, shear_mode : {'rigid', 'stiffness', 'free'}
        How the normal and the shear are held.
    normal_stiffness, shear_stiffness : float
        Each tie's stiffness along the normal and in the plane, in N/m,
        where its mode is 'stiffness'.
    """

    id: int
    name: str
    groups: tuple[int, int]
    nodes_a: tuple[int, ...]
    nodes_b: tuple[int, ...]
    normal_mode: str = 'rigid'
    shear_mode: str = 'rigid'
    normal_stiffness: float = 0.0
    shear_stiffness: float = 0.0

    def __post_init__(self) -> None:
        for what, mode, value in (('normal', self.normal_mode, self.normal_stiffness),
                                  ('shear', self.shear_mode, self.shear_stiffness)):
            if mode not in MODES:
                raise ValueError(f"the {what} is held 'rigid', 'stiffness' or 'free', "
                                 f'not {mode!r}')
            if mode == 'stiffness' and not float(value) > 0.0:
                raise ValueError(f'give the {what} stiffness a value above zero')
        if self.normal_mode == self.shear_mode == 'free':
            raise ValueError('free in the normal and the shear, the set holds nothing')

    def describe(self) -> str:
        """How the set holds, in a few words: 'rigid', 'stiffness', or
        'normal rigid, shear free' where the two differ."""
        if self.normal_mode == self.shear_mode:
            return self.normal_mode
        return f'normal {self.normal_mode}, shear {self.shear_mode}'

    def as_record(self) -> dict:
        """The set as plain values, for a file."""
        return {'id': int(self.id), 'name': self.name,
                'groups': [int(g) for g in self.groups],
                'nodes_a': [int(n) for n in self.nodes_a],
                'nodes_b': [int(n) for n in self.nodes_b],
                'normal_mode': self.normal_mode, 'shear_mode': self.shear_mode,
                'normal_stiffness': float(self.normal_stiffness),
                'shear_stiffness': float(self.shear_stiffness)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> MatedSet:
        """A set from `as_record`'s values. A record written before the
        normal and the shear were set apart says 'how': rigid holds both
        rigidly, spring both by their stiffness."""
        how = record.get('how')
        mode = {'rigid': 'rigid', 'spring': 'stiffness'}.get(how, 'rigid')
        return cls(int(record['id']), str(record['name']),
                   tuple(int(g) for g in record['groups']),
                   tuple(int(n) for n in record['nodes_a']),
                   tuple(int(n) for n in record['nodes_b']),
                   str(record.get('normal_mode', mode)),
                   str(record.get('shear_mode', mode)),
                   float(record.get('normal_stiffness', 0.0)),
                   float(record.get('shear_stiffness', 0.0)))

    def followed(self, nodes: Mapping[int, int | None] | None = None,
                 groups: Mapping[int, int | None] | None = None) -> MatedSet | None:
        """The set after the geometry's ids changed: `nodes` and
        `groups` map an old id to its new one, or to None where it is
        gone. A node gone leaves its side; a side left empty, or a
        group gone, ends the set (None)."""
        nodes = nodes or {}
        groups = groups or {}
        kept_groups = tuple(groups.get(int(g), int(g)) for g in self.groups)
        if any(g is None for g in kept_groups):
            return None

        def side(held):
            out = []
            for n in held:
                to = nodes.get(int(n), int(n))
                if to is not None and to not in out:
                    out.append(int(to))
            return tuple(out)

        a, b = side(self.nodes_a), side(self.nodes_b)
        if not a or not b:
            return None
        return replace(self, groups=kept_groups, nodes_a=a, nodes_b=b)


def _skin_nodes(geometry: Any, group: int) -> list[int]:
    from .mesh import group_skin

    return sorted({int(n) for face in group_skin(geometry, group) for n in face})


def _two_sided(geometry: Any, group: int) -> bool:
    """Whether a group's faces have no outside: plates, whose normals
    point whichever way their corners were listed."""
    from .geometry import ELEMENT_TYPES

    rows = np.flatnonzero(geometry.elem_group == int(group))
    return any(ELEMENT_TYPES[int(geometry.elem_type[r])][2] == 'face' for r in rows)


def _rows(geometry: Any) -> dict[int, int]:
    """{node id: row}, once per call: looking rows up a face at a time
    sorted every node id each time, 54 s of a model's 68 (2026-10-20)."""
    return {int(n): k for k, n in enumerate(geometry.node_id)}


def _corners(geometry: Any, faces: list, rows: dict | None = None) -> np.ndarray:
    """Each face's corners' coordinates, (faces, 4, 3), a triangle's
    third corner repeated as its fourth."""
    rows = _rows(geometry) if rows is None else rows
    index = np.array([[rows[int(n)] for n in (f if len(f) == 4 else (*f, f[2]))]
                      for f in faces], dtype=np.int64).reshape(-1, 4)
    return geometry.node_xyz[index]


def _face_normals(geometry: Any, faces: list, rows: dict | None = None) -> np.ndarray:
    """Each face's unit normal, as its corners turn."""
    if not faces:
        return np.zeros((0, 3))
    xyz = _corners(geometry, faces, rows)
    # a triangle's second half is a corner repeated, and adds nothing
    n = (np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
         + np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]))
    length = np.linalg.norm(n, axis=1)
    return n / np.where(length > 0, length, 1.0)[:, None]


def _own_normals(geometry: Any, faces: list, nodes: np.ndarray,
                 rows: dict | None = None) -> np.ndarray:
    """The normals of each node's own faces, (nodes, 6, 3), zero where a
    node has fewer than six."""
    normals = _face_normals(geometry, faces, rows)
    held: dict[int, list[np.ndarray]] = {}
    for face, n in zip(faces, normals):
        for node in face:
            held.setdefault(int(node), []).append(n)
    out = np.zeros((len(nodes), 6, 3))
    for k, node in enumerate(nodes):
        for j, n in enumerate(held.get(int(node), [])[:6]):
            out[k, j] = n
    return out


def _nearest_faces(geometry: Any, nodes: np.ndarray, own: np.ndarray, faces: list,
                   tolerance: float, two_sided: bool,
                   reach: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each of `nodes`, the face of `faces` it lies on: of the faces
    as near as the nearest, the one facing most against the node's own
    faces' normals `own` (a node on an edge of the other
    surface lies on two faces of it, and the one it mates with faces it).
    Returns (the face, -1 where none is within reach; its distance;
    how squarely it faces the node, -1 head on to +1 the same way)."""
    from scipy.spatial import cKDTree

    from .mesh import _point_triangle_distance

    count = len(nodes)
    face = np.full(count, -1, dtype=np.int64)
    distance = np.full(count, np.inf)
    facing = np.ones(count)
    if not count or not faces:
        return face, distance, facing
    rows = _rows(geometry)
    normals = _face_normals(geometry, faces, rows)
    xyz = _corners(geometry, faces, rows)
    quad = np.array([len(f) == 4 for f in faces])
    triangles = np.concatenate([xyz[:, [0, 1, 2]], xyz[quad][:, [0, 2, 3]]])
    owner = np.concatenate([np.arange(len(faces)), np.flatnonzero(quad)])
    centers = triangles.mean(axis=1)
    radius = float(np.linalg.norm(triangles - centers[:, None, :], axis=2).max())
    points = geometry.node_xyz[geometry.node_index(nodes)]
    near = cKDTree(centers).query_ball_point(points, radius + (tolerance if reach is None
                                                               else reach))
    which_p = np.repeat(np.arange(count), [len(f) for f in near])
    which_t = np.fromiter((t for f in near for t in f), dtype=np.int64, count=len(which_p))
    if not len(which_t):
        return face, distance, facing
    d = _point_triangle_distance(points[which_p], triangles[which_t, 0],
                                 triangles[which_t, 1], triangles[which_t, 2])
    dots = np.einsum('pkj,pj->pk', own[which_p], normals[owner[which_t]])
    if two_sided:
        dots = -np.abs(dots)
    mine = np.linalg.norm(own[which_p], axis=2) > 0
    score = np.where(mine, dots, np.inf).min(axis=1)
    score[~np.isfinite(score)] = 0.0
    best_d = np.full(count, np.inf)
    np.minimum.at(best_d, which_p, d)
    # as near as the nearest, give or take a hundredth of the tolerance:
    # a node on an edge, at one distance from the faces either side of
    # it. The whole tolerance was too wide a window: at a pick distance
    # of 0.3 in a face 0.35 in off was taken over one 0.25 in off
    close = d <= best_d[which_p] + 0.01 * tolerance
    order = np.lexsort((d, score, ~close, which_p))
    first = np.ones(len(order), dtype=bool)
    first[1:] = which_p[order][1:] != which_p[order][:-1]
    pick = order[first]
    face[which_p[pick]] = owner[which_t[pick]]
    distance[which_p[pick]] = d[pick]
    facing[which_p[pick]] = score[pick]
    return face, distance, facing


#: the faces around a corner that count toward its normal face within
#: 45 degrees of the face a node lies on: across a sharper edge they are
#: another surface (a block's side beside its top)
SMOOTH = np.cos(np.radians(45.0))


def _smooth_normals(geometry: Any, faces: list, hit: np.ndarray,
                    points: np.ndarray, rows: dict | None = None) -> np.ndarray:
    """The surface's normal at each point, lying on face `hit` of
    `faces`: each of the face's corners' normals, averaged over the faces
    around the corner on the same surface (`SMOOTH`), blended by how
    near the point is to each corner. On a curved surface a node is tied
    along the surface's own normal there, not a facet's: a facet's is
    tilted at its corners by half the angle between facets, enough to
    hold a shaft in its bore against turning (2026-10-20)."""
    rows = _rows(geometry) if rows is None else rows
    normals = _face_normals(geometry, faces, rows)
    around: dict[int, list[int]] = {}
    for k, face in enumerate(faces):
        for node in face:
            around.setdefault(int(node), []).append(k)
    out = np.zeros((len(hit), 3))
    for i, (f, p) in enumerate(zip(hit, points)):
        own = normals[f]
        corners = [int(n) for n in faces[f]]
        xyz = geometry.node_xyz[[rows[n] for n in corners]]
        gaps = np.linalg.norm(xyz - p, axis=1)
        if gaps.min() <= 1e-12 * (gaps.max() or 1.0):
            weights = (gaps == gaps.min()).astype(float)
        else:
            weights = 1.0 / gaps
        blended = np.zeros(3)
        for corner, weight in zip(corners, weights):
            others = normals[around[corner]]
            dots = others @ own
            # a plate's faces point whichever way their corners turn: each
            # is counted the way the hit face points
            same = np.abs(dots) >= SMOOTH
            summed = (others[same] * np.sign(dots[same])[:, None]).sum(axis=0)
            blended += weight * summed / (np.linalg.norm(summed) or 1.0)
        out[i] = blended / (np.linalg.norm(blended) or 1.0)
    return out


def near_surface(geometry: Any, nodes: Any, faces: list, distance: float) -> np.ndarray:
    """Which of `nodes` lie within `distance` of the `faces`: the band a
    node may be picked in, around a mated surface."""
    nodes = np.asarray(list(nodes), dtype=np.int64)
    if not len(nodes) or not faces:
        return np.zeros(len(nodes), dtype=bool)
    own = np.zeros((len(nodes), 6, 3))
    _face, gap, _facing = _nearest_faces(geometry, nodes, own, faces, distance, False,
                                         reach=distance)
    return gap <= distance


def _patches(skin_a: list, nodes: np.ndarray) -> list[np.ndarray]:
    """`nodes` split into the pieces the faces of `skin_a` connect."""
    index = {int(n): k for k, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for face in skin_a:
        held = [index[int(n)] for n in face if int(n) in index]
        for k in held[1:]:
            parent[root(k)] = root(held[0])
    pieces: dict[int, list[int]] = {}
    for k in range(len(nodes)):
        pieces.setdefault(root(k), []).append(k)
    return [nodes[ks] for ks in pieces.values()]


def _solid_or_plate_groups(geometry: Any, groups) -> list[int]:
    from .geometry import ELEMENT_TYPES

    kept = []
    for g in groups:
        rows = np.flatnonzero(geometry.elem_group == int(g))
        if any(ELEMENT_TYPES[int(geometry.elem_type[r])][2] in ('face', 'volume') for r in rows):
            kept.append(int(g))
    return kept


def _on_triangle(p: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The point of the triangle `corners` nearest `p`, and its three
    barycentric weights (each 0 to 1, summing to one)."""
    a, b, c = corners
    normal = np.cross(b - a, c - a)
    area = float(normal @ normal)
    if area <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])
    q = p - ((p - a) @ normal) / area * normal
    w = np.array([np.cross(c - b, q - b) @ normal, np.cross(a - c, q - c) @ normal,
                  np.cross(b - a, q - a) @ normal]) / area
    if (w >= -1e-12).all():
        w = np.clip(w, 0.0, None)
        return q, w / w.sum()
    # outside: the nearest point of the nearest edge
    best = None
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = corners[j] - corners[i]
        t = float(np.clip((p - corners[i]) @ edge / (edge @ edge or 1.0), 0.0, 1.0))
        point = corners[i] + t * edge
        gap = float(np.linalg.norm(p - point))
        if best is None or gap < best[0]:
            weights = np.zeros(3)
            weights[i], weights[j] = 1.0 - t, t
            best = (gap, point, weights)
    return best[1], best[2]


def _support(geometry: Any, point: np.ndarray, face: tuple,
             rows: dict | None = None) -> tuple[tuple, tuple, np.ndarray]:
    """Where `point` lies on `face`: the corners whose motion carries it
    (a triangle's three, an edge's two, a corner alone), their weights,
    and the point on the face. A corner of no weight is left out, so a
    node on an edge between two faces names that edge's two corners and
    no corner of either face beyond it."""
    corners = [int(n) for n in face]
    rows = _rows(geometry) if rows is None else rows
    xyz = geometry.node_xyz[[rows[n] for n in corners]]
    best = None
    for fan in ([(0, 1, 2)] if len(corners) == 3 else [(0, 1, 2), (0, 2, 3)]):
        on, weights = _on_triangle(point, xyz[list(fan)])
        gap = float(np.linalg.norm(point - on))
        if best is None or gap < best[0]:
            best = (gap, [corners[i] for i in fan], weights, on)
    _gap, nodes, weights, on = best
    kept = [(n, w) for n, w in zip(nodes, weights) if w > 1e-9]
    total = sum(w for _n, w in kept)
    return tuple(n for n, _w in kept), tuple(w / total for _n, w in kept), on


#: how squarely a node's own faces must face the other surface for the
#: node to mate with it: a node at the other surface's edge that only
#: grazes it, its faces at right angles, does not
FACING = -0.5


def detect(geometry: Any, groups: Any = None, tolerance: float | None = None) -> list[dict]:
    """The patches where the given element groups' surfaces mate.

    For each pair of the groups, the nodes on one's outer faces that lie
    on the other's outer faces, within `tolerance`, facing them, and
    are not the other's own nodes already; split into the pieces their
    faces connect, so the four legs standing on a table top are four
    patches. The side with more nodes on the other is the first side,
    the finer mesh, and its nodes follow; the second side's nodes are
    the corners that carry the points the first side's nodes lie on
    (`_support`).

    Parameters
    ----------
    geometry : Geometry
        Whose elements.
    groups : sequence of int, optional
        The element groups, by id; every group of plates or solids when
        omitted.
    tolerance : float, optional
        How far a node may be from the other's faces, as the geometry
        holds its coordinates; a millionth of the geometry's size when
        omitted.

    Returns
    -------
    list of dict
        Each patch's 'groups' (first side's, second side's), 'nodes_a'
        and 'nodes_b' (tuples of node ids).
    """
    from .mesh import group_skin

    if tolerance is None:
        tolerance = 1e-6 * float(np.linalg.norm(np.ptp(geometry.node_xyz, axis=0)) or 1.0)
    chosen = (list(geometry.group_id) if groups is None else [int(g) for g in groups])
    chosen = _solid_or_plate_groups(geometry, chosen)
    skins = {g: group_skin(geometry, g) for g in chosen}
    rows = _rows(geometry)
    skin_nodes = {g: sorted({int(n) for face in skins[g] for n in face}) for g in chosen}
    sided = {g: _two_sided(geometry, g) for g in chosen}
    found = []
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            sides = {}
            for first, second in ((a, b), (b, a)):
                own = set(skin_nodes[second])
                candidates = np.array([n for n in skin_nodes[first] if n not in own],
                                      dtype=np.int64)
                normals = _own_normals(geometry, skins[first], candidates, rows)
                face, distance, facing = _nearest_faces(
                    geometry, candidates, normals, skins[second], tolerance,
                    sided[first] or sided[second])
                on = (face >= 0) & (distance <= tolerance) & (facing <= FACING)
                sides[(first, second)] = (candidates[on], face[on])
            if not len(sides[(a, b)][0]) and not len(sides[(b, a)][0]):
                continue
            first, second = (a, b) if len(sides[(a, b)][0]) >= len(sides[(b, a)][0]) else (b, a)
            nodes, hit = sides[(first, second)]
            where = {int(n): int(f) for n, f in zip(nodes, hit)}
            points = dict(zip((int(n) for n in nodes),
                              geometry.node_xyz[geometry.node_index(nodes)]))
            for piece in _patches(skins[first], nodes):
                corners = sorted({c for n in piece for c in _support(
                    geometry, points[int(n)], skins[second][where[int(n)]], rows)[0]})
                found.append({'groups': (first, second),
                              'nodes_a': tuple(sorted(int(n) for n in piece)),
                              'nodes_b': tuple(corners)})
    return found


def links(geometry: Any, mated: MatedSet) -> list[tuple]:
    """The ties a set makes, as (the second side's nodes the point moves
    with, their weights, the first side's node, the normal, the point).

    Each first-side node is tied to the point it lies on, of the second
    side's faces: the point moves as the face's corners around it do,
    weighted by where it sits in its triangle (`_support`), the corners
    taken off the second side carrying none, so a rigid motion of the two is carried exactly however
    their meshes differ (a tie to the nearest node instead held the
    whole model against turning, 2026-10-20). The normal is the second
    side's surface's there, the face facing the node as `detect` picks
    it, smoothed over that surface's faces around it, so a curved
    surface ties each node along its own normal. A node with no such
    face within reach is tied to the nearest second-side node, and its
    point is where it is. A node of the second group's own (`shared`)
    is one with it already and makes none.

    Raises ValueError, naming the set, when it names a node the
    geometry no longer has.
    """
    from scipy.spatial import cKDTree

    from .mesh import group_skin

    have = set(geometry.node_id.tolist())
    for n in (*mated.nodes_a, *mated.nodes_b):
        if int(n) not in have:
            raise ValueError(f'the mated surfaces {mated.name!r} name node {n}, which the '
                             'geometry no longer has: edit them or delete them')
    second_side = set(mated.nodes_b)
    first, second = mated.groups
    skin_a, skin_b = group_skin(geometry, first), group_skin(geometry, second)
    # a node of the second group's own surface is one with it already: a
    # tie would name it on both ends (`shared`)
    merged = second_side | {int(n) for f in skin_b for n in f}
    targets = np.asarray(mated.nodes_b, dtype=np.int64)
    followers = np.asarray([n for n in mated.nodes_a if n not in merged], dtype=np.int64)
    if not len(followers) or not len(targets):
        return []
    xyz = geometry.node_xyz
    _d, nearest = cKDTree(xyz[geometry.node_index(targets)]).query(
        xyz[geometry.node_index(followers)])
    allowed = [k for k, f in enumerate(skin_b) if any(int(n) in second_side for n in f)]
    rows = _rows(geometry)
    own = _own_normals(geometry, skin_a, followers, rows)
    spread = float(np.linalg.norm(np.ptp(xyz, axis=0)) or 1.0)
    face, _distance, _facing = _nearest_faces(
        geometry, followers, own, [skin_b[k] for k in allowed], 1e-6 * spread,
        _two_sided(geometry, first) or _two_sided(geometry, second), reach=0.05 * spread)
    points = xyz[geometry.node_index(followers)]
    on = face >= 0
    hit = np.array([allowed[f] for f in face[on]], dtype=np.int64)
    smooth = np.zeros((len(followers), 3))
    if on.any():
        smooth[on] = _smooth_normals(geometry, skin_b, hit, points[on], rows)
    whole = dict(zip(np.flatnonzero(on).tolist(), hit.tolist()))
    out = []
    for k, node in enumerate(followers):
        if k in whole:
            nodes, shares, point = _support(geometry, points[k], skin_b[whole[k]], rows)
            # a corner taken off the second side carries none of it
            held = [(n, w) for n, w in zip(nodes, shares) if n in second_side]
            if held and sum(w for _n, w in held) > 0.5:
                total = sum(w for _n, w in held)
                nodes = tuple(n for n, _w in held)
                shares = tuple(w / total for _n, w in held)
                point = sum(w * xyz[rows[n]] for n, w in zip(nodes, shares))
                out.append((nodes, shares, int(node), smooth[k], point))
                continue
        to = int(targets[nearest[k]])
        gap = points[k] - xyz[rows[to]]
        normal = gap / (np.linalg.norm(gap) or 1.0) if np.any(gap) else np.array([0, 0, 1.0])
        out.append(((to,), (1.0,), int(node), normal, points[k]))
    return out


def shared(geometry: Any, groups: tuple[int, int]) -> set[int]:
    """The nodes on both groups' outer faces: where the two meshes are
    one node already (a seam the mesher joined, or nodes merged), held
    tighter than any tie, and never tied."""
    from .mesh import group_skin

    first, second = (({int(n) for f in group_skin(geometry, int(g)) for n in f})
                     for g in groups)
    return first & second


def next_id(geometry: Any) -> int:
    """The id a new set takes: one past the highest."""
    return max((m.id for m in geometry.mates), default=0) + 1


def default_name(geometry: Any, groups: tuple[int, int]) -> str:
    """'post on base', or 'post on base 2' where that name is taken."""
    names = dict(zip((int(g) for g in geometry.group_id), geometry.group_name))
    first, second = (str(names.get(int(g), '')).strip() or f'element group {g}' for g in groups)
    base = f'{first} on {second}'
    taken = {m.name for m in geometry.mates}
    if base not in taken:
        return base
    k = 2
    while f'{base} {k}' in taken:
        k += 1
    return f'{base} {k}'


def add(geometry: Any, groups: tuple[int, int], nodes_a: Any, nodes_b: Any,
        normal_mode: str = 'rigid', shear_mode: str = 'rigid',
        normal_stiffness: float = 0.0, shear_stiffness: float = 0.0,
        name: str | None = None) -> MatedSet:
    """Add a mated set to the geometry, in place, and return it.

    Refused with ValueError when a side is empty, when a node or a group
    is not in the geometry, when both the normal and the shear are free,
    or when a stiffness asked for is not above zero.
    """
    groups = (int(groups[0]), int(groups[1]))
    known = {int(g) for g in geometry.group_id}
    for g in groups:
        if g not in known:
            raise ValueError(f'no element group {g}')
    nodes_a = tuple(dict.fromkeys(int(n) for n in nodes_a))
    nodes_b = tuple(dict.fromkeys(int(n) for n in nodes_b))
    if not nodes_a or not nodes_b:
        raise ValueError('each side of mated surfaces needs a node')
    have = set(geometry.node_id.tolist())
    for n in (*nodes_a, *nodes_b):
        if n not in have:
            raise ValueError(f'no node {n}')
    mated = MatedSet(next_id(geometry), name or default_name(geometry, groups), groups,
                     nodes_a, nodes_b, normal_mode, shear_mode, float(normal_stiffness),
                     float(shear_stiffness))
    geometry.mates.append(mated)
    return mated


def find(geometry: Any, mate_id: int) -> MatedSet:
    """The set `mate_id`, refused with ValueError when there is none."""
    for mated in geometry.mates:
        if mated.id == int(mate_id):
            return mated
    raise ValueError(f'no mated surfaces {mate_id}')


def change(geometry: Any, mate_id: int, **fields: Any) -> MatedSet:
    """Change a set's fields (`name`, `nodes_a`, `nodes_b`,
    `normal_mode`, `shear_mode`, `normal_stiffness`, `shear_stiffness`),
    in place; refused as `add` refuses."""
    old = find(geometry, mate_id)
    for key in ('nodes_a', 'nodes_b'):
        if key in fields:
            fields[key] = tuple(dict.fromkeys(int(n) for n in fields[key]))
            if not fields[key]:
                raise ValueError('each side of mated surfaces needs a node')
    new = replace(old, **fields)
    geometry.mates[geometry.mates.index(old)] = new
    return new


def delete(geometry: Any, mate_ids: Any) -> int:
    """Remove sets by id, in place; returns how many went."""
    wanted = {int(i) for i in np.atleast_1d(mate_ids)}
    before = len(geometry.mates)
    geometry.mates = [m for m in geometry.mates if m.id not in wanted]
    return before - len(geometry.mates)


def naming_groups(geometry: Any, groups: Any) -> list[int]:
    """The sets that join any of these element groups, by id."""
    wanted = {int(g) for g in groups}
    return [m.id for m in geometry.mates if wanted & set(m.groups)]
