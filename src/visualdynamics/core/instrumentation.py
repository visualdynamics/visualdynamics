"""Planning a test's instrumentation from a finite element model.

Where accelerometers can go, and which of those places best measure a
chosen set of modes, by Effective Independence (Kammer, "Sensor
placement for on-orbit modal identification and correlation of large
space structures", Journal of Guidance, Control, and Dynamics 14(2),
1991).

`skin` splits the model's reachable outside into surfaces, the pieces a
person would name when saying where a sensor may go: a block's six
faces, a cylinder's side, the inside of a box. A solid contributes only
its outer surface, less the faces where it touches another part; a
plate contributes all of it. Neighboring faces belong to one surface
while their normals are within `EDGE_ANGLE` of each other, so a surface
ends at a sharp edge.

`candidates` makes the experimental geometry of every place a sensor
can sit on the surfaces kept: one node per candidate, each measured in its own coordinate
system whose +Z is the outward normal of the surface it sits on and
whose +X lies in that surface along the global axis least aligned with
the normal, so it is in the plane of a system axis wherever the
surface allows. A node where surfaces meet at an edge or a corner is a
candidate on each of them, so that a sensor sits flat on one; within a
surface the faces' normals at the node are averaged. A plate's normal
points away from the middle of the model. Beams contribute every node,
measured in the global system. A place nearer than `edge_distance` to
its surface's edge is not a candidate, since a sensor's base needs flat
ground around the spot it is glued to; an edge here is wherever the
surface ends, at a sharp edge, a free edge, a hole or where parts touch.

`local_shapes` turns a shape set's translations into each candidate's
own directions, and `effective_independence` chooses the sensors:
tri-axial accelerometers first, as blocks of three directions at one
place, then uni-axial ones along single directions with the tri-axial
places fixed. Each pass removes, one candidate at a time, the one whose
removal shrinks the determinant of the Fisher information matrix the
least, and while far more candidates remain than are wanted it removes
the weakest tenth at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Brandon, 2026-10-21: plan instrumentation from the FE model, tri-axial
# accelerometers placed first, no sensor on a block's interior, and an
# experimental geometry whose local systems have +Z along the surface
# normal (the standard convention) and +X in the plane of a system axis
# where possible. PLAN.md, "Plan Instrumentation".

#: neighboring faces whose outward normals differ by more than this, in
#: degrees, are on separate surfaces: a block's edge is 90, a
#: cylinder's facets a few degrees each
# Brandon, 2026-10-22: 20 degrees, from 40
EDGE_ANGLE = 20.0

#: how far a sensor stays from its surface's edge, in meters: a quarter
#: of an inch (Brandon, 2026-10-22)
EDGE_DISTANCE = 0.25 * 0.0254

#: a surface of less than this share of the whole outside's area is
#: left out unless picked (`default_surfaces`)
SMALL_SURFACE = 0.01

#: removal in batches stops once this many times the wanted count
#: remain, and the rest is removed one at a time
BATCH_UNTIL = 4

#: the share of the remaining candidates removed per batch
BATCH_SHARE = 0.1

#: a DOF label as `local_shapes` reads it, one line each: the node, and
#: the axis and sign of a translation (a rotation, or anything else, gives
#: no node, and is passed over)
_TRANSLATION = re.compile(r'^(?:(\d+)([XYZxyz])([+-]?)|.*)$', re.MULTILINE)


@dataclass
class Surface:
    """One surface of the model a sensor can sit on.

    Attributes:
        id: Its number, from 1, in the order of the elements its first
            face belongs to: the same for the same geometry every time.
        group: The element group it belongs to.
        kind: 'solid' for a solid's outside, 'plate' for a plate.
        faces: Its faces, each as its corner nodes in order around it.
        normals: `(faces, 3)`, each face's outward unit normal.
        areas: Each face's area.
    """

    id: int
    group: int
    kind: str
    faces: list[tuple[int, ...]]
    normals: np.ndarray
    areas: np.ndarray

    @property
    def area(self) -> float:
        """The surface's whole area."""
        return float(self.areas.sum())


@dataclass
class Candidates:
    """The places a sensor can sit, as an experimental geometry.

    Attributes:
        geometry: One node per candidate, each measured in a coordinate
            system of its own (`node_disp_cs`), +Z the outward normal.
        source: The finite element node each candidate sits on, in the
            order of `geometry.node_id`. A node where faces meet at an
            edge gives more than one candidate, so a candidate's id is
            its source node's id only for the first of them.
        kind: Per candidate, 'solid', 'plate' or 'line': what it sits on.
        surface: Per candidate, the id of the `Surface` it sits on; 0
            for a beam's node.
    """

    geometry: Any
    source: np.ndarray
    kind: list[str] = field(default_factory=list)
    surface: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def frames(self) -> np.ndarray:
        """`(candidates, 3, 3)`: each candidate's axes, rows X, Y, Z, in
        global coordinates."""
        g = self.geometry
        index = {int(c): k for k, c in enumerate(g.cs_id)}
        return np.array([g.cs_matrix[index[int(c)], :3] for c in g.node_disp_cs]
                        ).reshape(-1, 3, 3)


def skin(geometry: Any, *, edge_angle: float = EDGE_ANGLE) -> list[Surface]:
    """The surfaces of `geometry` a sensor can sit on: the solids'
    outside less where parts touch, and the plates, each split at its
    sharp edges.

    Parameters
    ----------
    geometry : Geometry
        The finite element model's geometry.
    edge_angle : float, default EDGE_ANGLE
        Degrees between two neighboring faces' normals beyond which they
        are on separate surfaces.

    Returns
    -------
    list of Surface
        Numbered from 1, in the order of the elements.
    """
    from .geometry import ELEMENT_TYPES, face_corners

    xyz = geometry.node_xyz
    row_of = {int(n): k for k, n in enumerate(geometry.node_id)}
    middle = xyz.mean(axis=0) if len(xyz) else np.zeros(3)

    # nodes in the same place are one node here: a mesh whose coincident
    # nodes were never merged has the same inside and the same surfaces
    # as one that was (Brandon, 2026-10-21: imperfect meshes work)
    spacing = geometry.node_spacing()
    same = (geometry.coincident_nodes(
        0.25 * spacing, nodes=sorted({int(n) for conn in geometry.elem_conn for n in conn}))
            if spacing else {})
    # every face of every solid in the model, counted: a face two solids
    # share is inside, whichever groups they are in
    outer = _outer_faces(geometry, same)
    faces: list[tuple[tuple[int, ...], np.ndarray, float, str, int]] = []
    normals = np.zeros((len(outer), 3))
    areas = np.zeros(len(outer))
    # a pass per corner count: one face at a time was 2 s on a cone of
    # 137,584 bricks (Brandon, 2026-10-21)
    for size in sorted({len(f[0]) for f in outer}):
        which = np.array([k for k, f in enumerate(outer) if len(f[0]) == size])
        points = xyz[np.array([[row_of[n] for n in outer[k][0]] for k in which])]
        normal, area = _normals_and_areas(points)
        inward = np.array([outer[k][1] for k in which])
        flip = np.einsum('ij,ij->i', normal, points.mean(axis=1) - inward) < 0
        normal[flip] = -normal[flip]
        normals[which], areas[which] = normal, area
    for k, (corners, _inward_point, group) in enumerate(outer):
        faces.append((corners, normals[k], float(areas[k]), 'solid', group))
    # plates: every face, its normal away from the middle of the model
    # (along the positive axis it is nearest when the middle lies in it)
    for r, conn in enumerate(geometry.elem_conn):
        code = int(geometry.elem_type[r])
        if ELEMENT_TYPES.get(code, ('', 0, ''))[2] != 'face':
            continue
        corners = tuple(int(n) for n in conn[:face_corners(code)])
        points = xyz[[row_of[n] for n in corners]]
        normal, area = _normal_and_area(points)
        away = float(np.dot(normal, points.mean(axis=0) - middle))
        size = np.sqrt(area) if area > 0 else 1.0
        if away < -1e-9 * size or (abs(away) <= 1e-9 * size
                                   and normal[np.argmax(np.abs(normal))] < 0):
            normal = -normal
        faces.append((corners, normal, area, 'plate', int(geometry.elem_group[r])))

    faces = _without_contact(geometry, faces)

    # faces sharing an edge are one surface while their normals agree;
    # a surface stays within one element group and one kind. The angle
    # is between neighbors only, so a curved surface is one surface
    # however far round it turns, and an edge counts as shared when its
    # nodes are the same or in the same place (Brandon, 2026-10-21:
    # curved surfaces work, never a face at a time)
    cosine = np.cos(np.radians(edge_angle))
    parent = list(range(len(faces)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    on_edge: dict[tuple[int, int], list[int]] = {}
    for k, (corners, *_rest) in enumerate(faces):
        corners = tuple(same.get(n, n) for n in corners)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            on_edge.setdefault((min(a, b), max(a, b)), []).append(k)
    for sharing in on_edge.values():
        for i, a in enumerate(sharing):
            for b in sharing[i + 1:]:
                fa, fb = faces[a], faces[b]
                if (fa[3] == fb[3] and fa[4] == fb[4]
                        and np.dot(fa[1], fb[1]) >= cosine):
                    parent[root(a)] = root(b)
    members: dict[int, list[int]] = {}
    for k in range(len(faces)):
        members.setdefault(root(k), []).append(k)
    found = []
    for number, ks in enumerate(sorted(members.values(), key=min), start=1):
        found.append(Surface(
            number, faces[ks[0]][4], faces[ks[0]][3], [faces[k][0] for k in ks],
            np.array([faces[k][1] for k in ks]).reshape(-1, 3),
            np.array([faces[k][2] for k in ks], dtype=float)))
    return found


def default_surfaces(found: list[Surface]) -> list[int]:
    """The surfaces sensors go on unless told otherwise: every one of at
    least `SMALL_SURFACE` of the whole outside's area. Sensors go on
    the significant surfaces of a test article; a small one is most
    often a sliver along an edge, a fillet or a hole's bore. When no
    surface is that large, all of them.

    Parameters
    ----------
    found : list of Surface
        From `skin`.

    Returns
    -------
    list of int
        Their ids, in order.
    """
    # Brandon, 2026-10-22: at 20 degrees the cone split into 193
    # surfaces, most of them a single element; "any surfaces less than
    # 1% of the total surface area should be de-selected by default ...
    # we just mount to significant surfaces usually"
    total = sum(s.area for s in found)
    kept = [s.id for s in found if s.area >= SMALL_SURFACE * total]
    return kept or [s.id for s in found]


def candidates(geometry: Any, groups: Any = None, surfaces: Any = None, *,
               edge_angle: float = EDGE_ANGLE, found: Any = None,
               edge_distance: float = 0.0) -> Candidates:
    """Every place on `geometry` a sensor can be mounted, as an
    experimental geometry with a coordinate system per place.

    Parameters
    ----------
    geometry : Geometry
        The finite element model's geometry.
    groups : sequence of int, optional
        The element groups sensors may go on. All of them when omitted.
        A face shared with a group that is not picked is still inside.
    surfaces : sequence of int, optional
        The surfaces sensors may go on, by `Surface.id`. Those of
        `groups` that `default_surfaces` keeps when omitted: every one
        of at least `SMALL_SURFACE` of the outside's area.
    edge_angle : float, default EDGE_ANGLE
        Degrees between two neighboring faces' normals beyond which they
        are on separate surfaces.
    found : list of Surface, optional
        `skin(geometry, edge_angle=edge_angle)`, when already found.
    edge_distance : float, default 0
        How near a place may be to its surface's edge, in the
        geometry's coordinates (meters once its units are defined),
        measured in a straight line. Nothing is left out at 0. A
        surface with no edge, such as a whole sphere, keeps every place.

    Returns
    -------
    Candidates
    """
    from .geometry import ELEMENT_TYPES, Geometry

    picked = ({int(g) for g in geometry.group_id} if groups is None
              else {int(g) for g in groups})
    xyz = geometry.node_xyz
    row_of = {int(n): k for k, n in enumerate(geometry.node_id)}
    by_id = np.argsort(geometry.node_id)
    if found is None:
        found = skin(geometry, edge_angle=edge_angle)
    wanted = set(default_surfaces(found) if surfaces is None else (int(s) for s in surfaces))

    # a place per node per surface, the surface's faces at the node
    # averaged by area; the largest first, so a node's own id goes to it
    nodes, sums, kinds, owners = [], [], [], []
    for surface in found:
        if surface.group not in picked or (wanted is not None and surface.id not in wanted):
            continue
        corner_nodes = np.array([n for f in surface.faces for n in f], dtype=np.int64)
        per_face = np.array([len(f) for f in surface.faces])
        weight = np.where(surface.areas > 0, surface.areas, 1.0)[:, None] * surface.normals
        at, where = np.unique(corner_nodes, return_inverse=True)
        summed = np.zeros((len(at), 3))
        np.add.at(summed, where, np.repeat(weight, per_face, axis=0))
        if edge_distance > 0:
            rows = by_id[np.searchsorted(geometry.node_id, at, sorter=by_id)]
            away = _away_from_edge(surface.faces, at, xyz[rows], xyz, by_id,
                                   geometry.node_id, edge_distance)
            at, summed = at[away], summed[away]
        nodes.append(at)
        sums.append(summed)
        kinds.extend([surface.kind] * len(at))
        owners.append(np.full(len(at), surface.id))
    node = np.concatenate(nodes) if nodes else np.zeros(0, dtype=np.int64)
    summed = np.concatenate(sums) if sums else np.zeros((0, 3))
    owner = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)
    length = np.linalg.norm(summed, axis=1)
    keep = length > 0
    node, summed, owner, length = node[keep], summed[keep], owner[keep], length[keep]
    kinds = [k for k, on in zip(kinds, keep, strict=True) if on]
    normal = summed / length[:, None] if len(length) else summed
    # beams: every node no surface already gave, in the global system
    covered = set(node.tolist())
    beam: list[int] = []
    for r, conn in enumerate(geometry.elem_conn):
        code = int(geometry.elem_type[r])
        if (ELEMENT_TYPES.get(code, ('', 0, ''))[2] == 'line'
                and int(geometry.elem_group[r]) in picked):
            for n in conn:
                if int(n) not in covered:
                    covered.add(int(n))
                    beam.append(int(n))
    node = np.concatenate([node, np.array(beam, dtype=np.int64)])
    normal = np.vstack([normal, np.tile([0.0, 0.0, 1.0], (len(beam), 1))])
    owner = np.concatenate([owner, np.zeros(len(beam), dtype=np.int64)])
    length = np.concatenate([length, np.zeros(len(beam))])
    kinds = kinds + ['line'] * len(beam)
    order = np.lexsort((owner, -length, node))
    node, normal, owner = node[order], normal[order], owner[order]
    kinds = [kinds[k] for k in order]

    # ids: a node's own id for its first place, then fresh ids above
    # every node id for the rest, in node order
    count = len(node)
    first = np.ones(count, dtype=bool)
    first[1:] = node[1:] != node[:-1]
    fresh = int(geometry.node_id.max()) + 1 if len(geometry.node_id) else 1
    ids = np.where(first, node, fresh + np.cumsum(~first) - 1)
    line = np.array([k == 'line' for k in kinds], dtype=bool)
    frames = np.tile(np.eye(3), (count, 1, 1))
    if (~line).any():
        frames[~line] = _surface_frames(normal[~line])
    positions = xyz[[row_of[int(n)] for n in node]] if count else np.zeros((0, 3))
    cs_matrix = np.zeros((count + 1, 4, 3))
    cs_matrix[0, :3] = np.eye(3)
    cs_matrix[1:, :3] = frames
    cs_matrix[1:, 3] = positions
    systems = list(range(1, count + 2))
    experimental = Geometry(
        node_id=ids, node_xyz=positions, node_def_cs=np.ones(count, dtype=np.int64),
        node_disp_cs=np.array(systems[1:], dtype=np.int64), cs_id=systems,
        cs_name=[''] * (count + 1), cs_type=np.zeros(count + 1, dtype=np.int64),
        cs_matrix=cs_matrix, length_unit=geometry.length_unit)
    return Candidates(experimental, node.astype(np.int64), kinds, owner.astype(np.int64))


def surface_frame(normal: Any) -> np.ndarray:
    """The axes of a sensor on a surface with this outward normal, rows
    X, Y, Z: Z the normal, X the global axis least aligned with it laid
    into the surface, Y completing a right-handed set.

    Parameters
    ----------
    normal : array_like
        The outward normal; any length.

    Returns
    -------
    numpy.ndarray
        `(3, 3)`, orthonormal, determinant +1.
    """
    return _surface_frames(np.asarray(normal, dtype=float)[np.newaxis])[0]


def _surface_frames(normals: np.ndarray) -> np.ndarray:
    """`surface_frame` for `(n, 3)` normals at once, `(n, 3, 3)`."""
    z = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    axis = np.eye(3)[np.argmin(np.abs(z), axis=1)]
    x = axis - np.einsum('ij,ij->i', axis, z)[:, None] * z
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], axis=1)


def local_shapes(shapes: Any, modes: Any, plan: Candidates,
                 geometry: Any) -> tuple[np.ndarray, np.ndarray]:
    """The modes' translations at each candidate, in its own directions.

    Parameters
    ----------
    shapes : ShapeSet
        The finite element modes.
    modes : sequence of int
        Which of them, by index.
    plan : Candidates
        Where sensors can go.
    geometry : Geometry
        The finite element geometry the shapes are measured on, whose
        coordinate systems say which way each DOF points.

    Returns
    -------
    blocks : numpy.ndarray
        `(candidates, 3, modes)`: rows X, Y, Z of each candidate's own
        system. Real: a complex mode is taken by its real part.
    measured : numpy.ndarray
        Per candidate, whether the shapes cover all three translations
        at its node. A candidate they do not cover cannot be chosen.
    """
    modes = [int(m) for m in modes]
    matrix = np.real(np.asarray(shapes.shape_matrix))[modes]       # (m, dofs)
    wanted = {int(n) for n in plan.source}
    row_of = {int(n): k for k, n in enumerate(geometry.node_id)}
    system_of = {int(c): k for k, c in enumerate(geometry.cs_id)}
    # each translational DOF's node row, axis and sign, read once; a
    # vectorized pass then turns them all into global directions (a
    # lookup per DOF was 3 s on the Frame-Wing's 200,000 columns)
    # the labels read in one pass of a pattern, the same reading as
    # `parse_dof`: one call per label was 2.6 s on a cone's million DOFs
    found = _TRANSLATION.findall('\n'.join(str(dof).strip() for dof in shapes.coordinate))
    columns, rows, axes, signs = [], [], [], []
    for column, (node, axis, sign) in enumerate(found):
        if not node:
            continue
        node = int(node)
        if node not in wanted or node not in row_of:
            continue
        columns.append(column)
        rows.append(row_of[node])
        axes.append('XYZ'.index(axis.upper()))
        signs.append(-1.0 if sign == '-' else 1.0)
    rows = np.asarray(rows, dtype=np.int64)
    systems = np.array([system_of.get(int(c), -1) for c in geometry.node_disp_cs[rows]],
                       dtype=np.int64) if len(rows) else np.zeros(0, dtype=np.int64)
    # a node measured in a system the geometry lacks is measured in the
    # global one; a cylindrical or spherical one is not taken
    cartesian = np.array([s < 0 or int(geometry.cs_type[s]) == 0 for s in systems], dtype=bool)
    keep = cartesian
    columns = np.asarray(columns, dtype=np.int64)[keep]
    rows, axes = rows[keep], np.asarray(axes, dtype=np.int64)[keep]
    signs, systems = np.asarray(signs)[keep], systems[keep]
    directions = np.where((systems >= 0)[:, None],
                          geometry.cs_matrix[np.maximum(systems, 0), axes],
                          np.eye(3)[axes]) if len(rows) else np.zeros((0, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True) * signs[:, None]
    # every node's global translations: sum over its DOFs of direction x value
    node_index = {int(n): k for k, n in enumerate(sorted(wanted))}
    slot = np.array([node_index[int(geometry.node_id[r])] for r in rows], dtype=np.int64)
    global_at = np.zeros((len(node_index), 3, len(modes)))
    np.add.at(global_at, slot, directions[:, :, None] * matrix[:, columns].T[:, None, :])
    covered = np.zeros((len(node_index), 3), dtype=bool)
    covered[slot, np.argmax(np.abs(directions), axis=1)] = True
    frames = plan.frames
    at = np.array([node_index[int(n)] for n in plan.source], dtype=np.int64)
    blocks = np.einsum('kij,kjm->kim', frames, global_at[at])
    measured = covered[at].all(axis=1)
    return blocks, measured


@dataclass
class Placement:
    """The sensors Effective Independence chose.

    Attributes:
        triaxial: Candidate indices given a tri-axial accelerometer.
        uniaxial: `(candidate index, axis)` per uni-axial accelerometer,
            the axis 0, 1 or 2 for the candidate's X, Y or Z.
        information: The Fisher information matrix of the chosen
            directions over the modes planned for, `(planned, planned)`.
        independence: Each chosen sensor's share of the information:
            for a tri-axial accelerometer the sum over its three
            directions of the effective independence, for a uni-axial
            one its single value; tri-axial first, in the order above.
            They sum to the number of modes planned for.
        planned: The modes the sensors were chosen for, as positions in
            the modes given: all of them, or, when the sensors measure
            fewer directions than there are modes, the lowest that many.
    """

    triaxial: list[int]
    uniaxial: list[tuple[int, int]]
    information: np.ndarray
    independence: np.ndarray
    planned: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def condition(self) -> float:
        """The condition number of the information matrix: how nearly
        the chosen sensors fail to tell the modes apart."""
        return float(np.linalg.cond(self.information))


def effective_independence(blocks: Any, triaxial: int, uniaxial: int = 0, *,
                           axes: Any = (2,), allowed: Any = None,
                           sources: Any = None, lowest: Any = None) -> Placement:
    """Choose sensors by Effective Independence, tri-axial first.

    Parameters
    ----------
    blocks : array_like
        `(candidates, 3, modes)`, the modes at each candidate in its own
        directions (`local_shapes`).
    triaxial : int
        How many tri-axial accelerometers.
    uniaxial : int, default 0
        How many uni-axial accelerometers.
    axes : sequence of int, default (2,)
        Which of a candidate's directions a uni-axial accelerometer may
        measure: its Z, the surface normal, by default.
    allowed : array_like of bool, optional
        Per candidate, whether it may be chosen at all.
    sources : array_like of int, optional
        Per candidate, the node it sits on. A tri-axial accelerometer
        measures the same three translations whichever face of an edge
        it is given, so only one candidate per node is offered for them.
    lowest : array_like of int, optional
        The modes' positions, lowest frequency first: the order in which
        modes are planned for when the sensors cannot resolve them all.
        The order given when omitted.

    Returns
    -------
    Placement

    Raises
    ------
    ValueError
        When there are no sensors, fewer candidates than sensors, or
        the chosen directions cannot tell the modes planned for apart.

    Notes
    -----
    Sensors measuring D directions resolve at most D modes. With fewer
    directions than modes, the plan is made for the lowest D
    (`Placement.planned`). The tri-axial pass on its own is held to the
    same rule: it chooses for the lowest 3 x `triaxial` modes, and the
    uni-axial pass then chooses for all the modes planned for, with the
    tri-axial places fixed. Chosen for more modes than it can resolve,
    the tri-axial pass works on a singular information matrix, and the
    rounding of the arithmetic decides.
    """
    # Brandon, 2026-10-21: fewer directions than modes is planned for the
    # lowest modes, with a warning, rather than refused; the tri-axial
    # pass was found deciding by rounding with 30 tri-axials for 115
    # modes, and the same rule mends it
    blocks = np.asarray(blocks, dtype=float)
    count, _three, modes = blocks.shape
    triaxial, uniaxial = int(triaxial), int(uniaxial)
    if triaxial + uniaxial <= 0:
        raise ValueError('no sensors to place')
    order = np.arange(modes) if lowest is None else np.asarray(lowest, dtype=np.int64)
    planned = np.sort(order[:min(modes, 3 * triaxial + uniaxial)])
    for_tri = np.sort(order[:min(modes, 3 * triaxial)])
    allowed = (np.ones(count, dtype=bool) if allowed is None
               else np.asarray(allowed, dtype=bool))
    pool = np.flatnonzero(allowed)
    if sources is not None:
        sources = np.asarray(sources)
        _unique, first = np.unique(sources[pool], return_index=True)
        tri_pool = pool[np.sort(first)]
    else:
        tri_pool = pool
    if triaxial > len(tri_pool):
        raise ValueError(f'{triaxial} tri-axial accelerometers but {len(tri_pool)} places')
    scale = float(np.einsum('kim,kim->', blocks[pool], blocks[pool])) / max(modes, 1)

    # tri-axial: whole blocks, each removal costing log det(I - B A^-1 B').
    # Written as matrix products, which go to BLAS: as three-operand
    # einsums they ran as plain loops, 170 s on a cone of 83,811 places
    # and 115 modes (Brandon, 2026-10-21), 3 s this way
    rows = blocks[tri_pool][:, :, for_tri]                        # (k, 3, m)
    width = len(for_tri)
    ridge = 1e-12 * scale * np.eye(width)
    kept = np.arange(len(tri_pool) if triaxial else 0)
    while len(kept) > triaxial:
        active = rows[kept]
        flat = active.reshape(-1, width)
        inverse = np.linalg.inv(flat.T @ flat + ridge)
        inner = np.matmul((flat @ inverse).reshape(-1, 3, width), active.transpose(0, 2, 1))
        loss = -np.log(np.clip(np.linalg.det(np.eye(3) - inner), 1e-300, None))
        kept = _remove(kept, loss, triaxial)
    chosen_tri = [int(i) for i in tri_pool[kept]]

    # uni-axial: single directions at the places left, the tri-axial
    # ones fixed in the information; all the modes planned for
    width = len(planned)
    ridge = 1e-12 * scale * np.eye(width)
    reduced = blocks[:, :, planned]
    base = (np.einsum('kim,kin->mn', reduced[chosen_tri], reduced[chosen_tri])
            if chosen_tri else np.zeros((width, width)))
    taken = set(chosen_tri)
    options = [(int(k), int(a)) for k in pool if int(k) not in taken for a in axes]
    if uniaxial > len(options):
        raise ValueError(f'{uniaxial} uni-axial accelerometers but {len(options)} directions')
    vectors = (np.array([reduced[k, a] for k, a in options]) if options
               else np.zeros((0, width)))
    kept = np.arange(len(options))
    while len(kept) > uniaxial:
        active = vectors[kept]
        information = base + active.T @ active + ridge
        inverse = np.linalg.inv(information)
        independence = np.sum((active @ inverse) * active, axis=1)
        loss = -np.log(np.clip(1.0 - independence, 1e-300, None))
        kept = _remove(kept, loss, uniaxial)
    chosen_uni = [options[int(i)] for i in kept]

    rows = [reduced[k] for k in chosen_tri] + [reduced[k, a][np.newaxis] for k, a in chosen_uni]
    stacked = np.vstack(rows) if rows else np.zeros((0, width))
    information = stacked.T @ stacked
    if np.linalg.matrix_rank(information) < width:
        raise ValueError('the chosen directions cannot tell the modes apart: '
                         'add sensors or allow more places')
    inverse = np.linalg.inv(information)
    each = np.sum((stacked @ inverse) * stacked, axis=1)
    shares = np.concatenate([each[:3 * len(chosen_tri)].reshape(-1, 3).sum(axis=1)
                             if chosen_tri else np.zeros(0),
                             each[3 * len(chosen_tri):]])
    return Placement(chosen_tri, chosen_uni, information, shares, planned)


@dataclass
class Plan:
    """An instrumentation plan, as the three objects a test is set up
    from.

    Attributes:
        geometry: The experimental geometry of the chosen places only,
            each measured in its own system (+Z out of the surface).
        channels: The channel table: one row per measured direction,
            tri-axial accelerometers first, each's three rows sharing a
            node and named by its leg (`triax_dof`).
        shapes: The finite element modes at exactly those channels, in
            the places' own directions, for correlating the test.
        placement: What Effective Independence chose, with the
            information matrix and each sensor's share.
    """

    geometry: Any
    channels: Any
    shapes: Any
    placement: Placement


def plan(shapes: Any, geometry: Any, triaxial: int, uniaxial: int = 0, *,
         modes: Any = None, groups: Any = None, surfaces: Any = None,
         edge_angle: float | None = None, axes: Any = (2,),
         edge_distance: float | None = None) -> Plan:
    """Plan a test's accelerometers from a finite element model.

    Parameters
    ----------
    shapes : ShapeSet
        The finite element modes.
    geometry : Geometry
        The geometry they are on.
    triaxial : int
        How many tri-axial accelerometers.
    uniaxial : int, default 0
        How many uni-axial accelerometers.
    modes : sequence of int, optional
        The modes to measure, by index. The elastic modes (every one
        above 0 Hz) when omitted.
    groups : sequence of int, optional
        The element groups sensors may go on. All when omitted.
    surfaces : sequence of int, optional
        The surfaces sensors may go on, by `Surface.id`. Those of
        `groups` that `default_surfaces` keeps when omitted.
    edge_angle : float, optional
        Degrees between neighboring faces' normals beyond which they are
        on separate surfaces; `EDGE_ANGLE` when omitted.
    axes : sequence of int, default (2,)
        The directions a uni-axial accelerometer may measure: 0, 1, 2
        for a place's X, Y, Z. Z, the surface normal, by default.
    edge_distance : float, optional
        How near a sensor may be to its surface's edge, in the
        geometry's coordinates. `EDGE_DISTANCE`, a quarter of an inch,
        when omitted and the geometry's units are defined; 0 when they
        are not, since a distance means nothing against coordinates
        with no unit.

    Returns
    -------
    Plan

    Raises
    ------
    ValueError
        When no mode is chosen, or as `effective_independence` does.
    """
    if modes is None:
        modes = elastic(shapes)
    modes = [int(m) for m in modes]
    if not modes:
        raise ValueError('no modes to measure: every mode is a rigid-body mode at 0 Hz')
    if edge_distance is None:
        edge_distance = EDGE_DISTANCE if geometry.units_defined else 0.0
    places = candidates(geometry, groups, surfaces,
                        edge_angle=EDGE_ANGLE if edge_angle is None else float(edge_angle),
                        edge_distance=float(edge_distance))
    blocks, measured = local_shapes(shapes, modes, places, geometry)
    placement = effective_independence(blocks, triaxial, uniaxial, axes=axes,
                                       allowed=measured, sources=places.source,
                                       lowest=lowest_first(shapes, modes))
    return assemble(shapes, modes, places, blocks, placement)


class FewerDirectionsWarning(UserWarning):
    """The sensors measure fewer directions than there are modes: the
    plan is made for the lowest modes only, and the others cannot be
    told apart by it."""


def lowest_first(shapes: Any, modes: Any) -> np.ndarray:
    """The positions in `modes` ordered lowest frequency first: the
    order in which `effective_independence` plans for them."""
    return np.argsort(np.asarray(shapes.frequency)[[int(m) for m in modes]], kind='stable')


def shortfall(placement: Placement, modes: int) -> str:
    """What a plan for fewer modes than the `modes` asked for leaves out,
    in words; empty when it planned for them all."""
    planned = len(placement.planned)
    if planned >= modes:
        return ''
    return (f'{planned} directions for {modes} modes: planned for the lowest {planned}; '
            f'the other {modes - planned} cannot be told apart by these sensors')


def elastic(shapes: Any) -> list[int]:
    """The indices of a shape set's modes above 0 Hz: what a plan
    measures unless told otherwise."""
    return [k for k, f in enumerate(shapes.frequency) if f > 0]


def assemble(shapes: Any, modes: Any, places: Candidates, blocks: Any,
             placement: Placement) -> Plan:
    """The three objects of a plan from what Effective Independence
    chose: the step `plan` ends with, apart so that a preview can choose
    again without finding the places again.

    Parameters
    ----------
    shapes : ShapeSet
        The finite element modes.
    modes : sequence of int
        The modes measured, by index, as `blocks` was read for them.
    places : Candidates
        Where sensors could go.
    blocks : numpy.ndarray
        `(candidates, 3, modes)`, from `local_shapes`.
    placement : Placement
        What was chosen.

    Returns
    -------
    Plan
    """
    from .channel_table import ChannelTable
    from .geometry import Geometry
    from .shapes import ShapeSet

    modes = [int(m) for m in modes]
    ids = places.geometry.node_id
    channels: list[tuple[int, int, str, str]] = []      # place, axis, leg, comment
    for number, k in enumerate(placement.triaxial, start=1):
        channels.extend((k, axis, 'XYZ'[axis], f'tri-axial {number}') for axis in range(3))
    for number, (k, axis) in enumerate(placement.uniaxial, start=1):
        channels.append((k, axis, '', f'uni-axial {number}'))
    chosen = list(dict.fromkeys(k for k, _a, _l, _c in channels))
    experimental = places.geometry
    rows = {int(c): r for r, c in enumerate(experimental.cs_id)}
    systems = [rows[int(experimental.node_disp_cs[k])] for k in chosen]
    kept_geometry = Geometry(
        node_id=ids[chosen], node_xyz=experimental.node_xyz[chosen],
        node_def_cs=np.full(len(chosen), experimental.cs_id[0]),
        node_disp_cs=experimental.node_disp_cs[chosen],
        cs_id=np.concatenate([experimental.cs_id[:1], experimental.cs_id[systems]]),
        cs_name=[''] * (len(chosen) + 1), cs_type=np.zeros(len(chosen) + 1, dtype=np.int64),
        cs_matrix=np.concatenate([experimental.cs_matrix[:1], experimental.cs_matrix[systems]]),
        length_unit=experimental.length_unit)
    table = ChannelTable({
        'channel': np.arange(1, len(channels) + 1),
        'node': [int(ids[k]) for k, _a, _l, _c in channels],
        'direction': [f'{"XYZ"[a]}+' for _k, a, _l, _c in channels],
        'role': ['response'] * len(channels),
        'channel_type': ['acceleration'] * len(channels),
        'unit': ['g'] * len(channels),
        'triax_dof': [leg for _k, _a, leg, _c in channels],
        'comment': [comment for _k, _a, _l, comment in channels],
    })
    coordinate = [f'{int(ids[k])}{"XYZ"[a]}+' for k, a, _l, _c in channels]
    matrix = np.stack([blocks[k, a] for k, a, _l, _c in channels], axis=1)
    picked = np.asarray(modes)
    description = getattr(shapes, 'description', None)
    test_shapes = ShapeSet(
        shapes.frequency[picked], shapes.damping[picked], coordinate, matrix,
        modal_mass=np.asarray(shapes.modal_mass)[picked],
        mass_unit=shapes.mass_unit,
        description=[description[m] for m in modes] if description is not None else None)
    return Plan(kept_geometry, table, test_shapes, placement)


def _away_from_edge(faces: list, at: np.ndarray, points: np.ndarray, xyz: np.ndarray,
                    by_id: np.ndarray, node_id: np.ndarray, distance: float) -> np.ndarray:
    """Which of the nodes `at`, at `points`, are at least `distance`
    from the edge of the surface made of `faces`: the sides only one of
    its faces has."""
    from scipy.spatial import cKDTree

    sides = np.array([sorted((f[k], f[(k + 1) % len(f)])) for f in faces
                      for k in range(len(f))], dtype=np.int64).reshape(-1, 2)
    unique, count = np.unique(sides, axis=0, return_counts=True)
    edge = unique[count == 1]
    if not len(edge):
        return np.ones(len(at), dtype=bool)
    ends = xyz[by_id[np.searchsorted(node_id, edge.ravel(), sorter=by_id)]].reshape(-1, 2, 3)
    start, along = ends[:, 0], ends[:, 1] - ends[:, 0]
    middle = start + along / 2
    half = float(np.linalg.norm(along, axis=1).max()) / 2
    # every side whose middle is near enough that the side could be
    # nearer than `distance`, then its exact distance
    near = cKDTree(middle).query_ball_point(points, distance + half, return_sorted=False)
    which = np.repeat(np.arange(len(at)), [len(n) for n in near])
    side = np.fromiter((k for n in near for k in n), dtype=np.int64, count=len(which))
    if not len(side):
        return np.ones(len(at), dtype=bool)
    span = np.einsum('ij,ij->i', along[side], along[side])
    t = np.einsum('ij,ij->i', points[which] - start[side], along[side])
    t = np.clip(np.divide(t, span, out=np.zeros_like(t), where=span > 0), 0.0, 1.0)
    gap = np.linalg.norm(points[which] - start[side] - t[:, None] * along[side], axis=1)
    away = np.ones(len(at), dtype=bool)
    away[which[gap < distance]] = False
    return away


def _remove(kept: np.ndarray, loss: np.ndarray, target: int) -> np.ndarray:
    """`kept` less its weakest: a tenth of it while far more remain than
    are wanted, otherwise the single weakest."""
    if len(kept) > BATCH_UNTIL * max(target, 1):
        drop = min(max(1, int(BATCH_SHARE * len(kept))), len(kept) - BATCH_UNTIL * max(target, 1))
    else:
        drop = 1
    weakest = np.argsort(loss, kind='stable')[:drop]
    return np.delete(kept, weakest)


def _outer_faces(geometry: Any, same: dict | None = None
                 ) -> list[tuple[tuple[int, ...], np.ndarray, int]]:
    """The faces of the model's solids that no second solid shares, each
    with a point inside its solid and the solid's element group. `same`
    maps a node to the node in its place, so two faces in one place are
    shared whether or not their nodes were merged. In element order,
    each element's faces in order."""
    from .geometry import ELEMENT_TYPES, solid_faces

    xyz = geometry.node_xyz
    ids = np.asarray(geometry.node_id, dtype=np.int64)
    by_id = np.argsort(ids)

    def rows(nodes):
        return by_id[np.searchsorted(ids[by_id], nodes)]

    # the row a node counts as: its own, or the row of the node in its place
    counted = np.arange(len(ids))
    if same:
        counted[rows(np.fromiter(same, dtype=np.int64))] = rows(
            np.fromiter(same.values(), dtype=np.int64))
    by_count: dict[int, list[int]] = {}
    for r, conn in enumerate(geometry.elem_conn):
        if ELEMENT_TYPES.get(int(geometry.elem_type[r]), ('', 0, ''))[2] == 'volume':
            by_count.setdefault(len(conn), []).append(r)
    # every face of every solid, keyed by the rows it counts as, sorted
    # and padded to four; counted as arrays, since a dict of tuples one
    # face at a time was 4 s on the cone's 825,000 faces
    keys, order, corners, inside, group = [], [], [], [], []
    for count, elements in by_count.items():
        shape = solid_faces(count)
        if not shape:
            continue
        elements = np.array(elements)
        conn = np.array([geometry.elem_conn[r] for r in elements], dtype=np.int64)
        corner_count = 1 + max(max(face) for face in shape)
        middle = xyz[rows(conn[:, :corner_count])].mean(axis=1)
        for f, face in enumerate(shape):
            on = conn[:, list(face)]
            key = np.full((len(elements), 4), -1, dtype=np.int64)
            key[:, :len(face)] = np.sort(counted[rows(on)], axis=1)
            keys.append(np.sort(key, axis=1))
            order.append(elements * 16 + f)
            corners.extend(on.tolist())
            inside.append(middle)
            group.append(geometry.elem_group[elements])
    if not keys:
        return []
    keys = np.concatenate(keys)
    _unique, which, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    once = np.flatnonzero(counts[which.ravel()] == 1)
    once = once[np.argsort(np.concatenate(order)[once], kind='stable')]
    inside = np.concatenate(inside)
    group = np.concatenate(group)
    return [(tuple(corners[k]), inside[k], int(group[k])) for k in once]


def _without_contact(geometry: Any, faces: list) -> list:
    """`faces` less those lying on another element group's outer
    surface: where two parts touch, by nodes in the same place or by
    meshes that do not match (a washer on a frame)."""
    from .mates import near_surface

    spacing = geometry.node_spacing()
    if not spacing:
        return faces
    reach = 0.25 * spacing
    groups = sorted({f[4] for f in faces})
    if len(groups) < 2:
        return faces
    kept = []
    for group in groups:
        own = [f for f in faces if f[4] == group]
        others = [f[0] for f in faces if f[4] != group]
        nodes = sorted({n for f in own for n in f[0]})
        near = dict(zip(nodes, near_surface(geometry, nodes, others, reach), strict=True))
        kept.extend(f for f in own if not all(near[n] for n in f[0]))
    order = {id(f): k for k, f in enumerate(faces)}
    return sorted(kept, key=lambda f: order[id(f)])


def _normal_and_area(points: np.ndarray) -> tuple[np.ndarray, float]:
    """A polygon's unit normal (by its corners' order) and its area, by
    Newell's method, which a slightly warped quadrilateral does not
    upset."""
    normal, area = _normals_and_areas(points[np.newaxis])
    return normal[0], float(area[0])


def _normals_and_areas(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`_normal_and_area` for `(faces, corners, 3)` at once. A face of no
    area gets +Z."""
    following = np.roll(points, -1, axis=1)
    newell = np.stack([
        np.sum((points[..., 1] - following[..., 1]) * (points[..., 2] + following[..., 2]), axis=1),
        np.sum((points[..., 2] - following[..., 2]) * (points[..., 0] + following[..., 0]), axis=1),
        np.sum((points[..., 0] - following[..., 0]) * (points[..., 1] + following[..., 1]), axis=1),
    ], axis=1)
    length = np.linalg.norm(newell, axis=1)
    flat = length == 0
    normal = np.where(flat[:, None], [0.0, 0.0, 1.0], newell / np.where(flat, 1.0, length)[:, None])
    return normal, 0.5 * length

