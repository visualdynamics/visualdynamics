"""Shells: solids meshed into plates (Mesh Solids, Shells).

A flat solid, a plate or a disk with any holes subtracted from it, is
meshed on its own plane. A solid with a volume is meshed on its outer
surface. Either way the elements are four-node plates or three-node
triangles, in an element group named for the part, and their thickness
is set on that group in the Element Groups table.

A flat part is meshed by the brick mesher, so its holes come out round
as a brick mesh's do: the part is laid out in its own frame as a slab
one element thick (a plate a block, a disk a cylinder), each solid cut
from it run through the slab, and the bottom face of each brick, moved
onto the plane, is a plate. An outer surface is the faces of the part's
bricks, or of its tetrahedra for triangles, that no second element
shares; the tetrahedra keep their slivers for it, so the surface has no
dimple where one would have been removed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

# Brandon (2026-10-26): a plate and a disk on the Solids row, meshed
# with shell elements, and a solid meshed as shells taking its outer
# surface only; quads and triangles both offered

#: the element codes made: a four-node plate and a three-node triangle,
#: the ones `mesh.plane` writes
QUAD = 44
TRIANGLE = 41


def shell_mesh(geometry: Any, ids, size: float, triangles: bool = False,
               progress: Any = None) -> tuple[Any, dict]:
    """The parts as plates, ready to join to a geometry (`mesh.join`).

    A flat part (`solids.is_flat`) is meshed on its plane, and a part
    with a volume on its outer surface. Each part gets one element
    group, named for it. Parts that meet share their nodes where their
    meshes' nodes coincide.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id; each a solid nothing else uses.
    size : float
        The element size aimed at, as the geometry holds its
        coordinates.
    triangles : bool, default False
        Three-node triangles rather than four-node plates. A flat part's
        quads are each split on the shorter diagonal; a part with a
        volume takes the surface of a tetrahedral mesh.
    progress : callable, optional
        Told ``(done, total)`` as an outer surface's volume mesh goes.

    Returns
    -------
    (Geometry, dict)
        The plates, and what was made: 'method' ('shells'), 'elements',
        'nodes', 'flat' (how many parts were meshed on their plane) and
        'surfaces' (how many on their outer surface).
    """
    from .mesh import join
    from .solids import is_flat, refuse_parts_of_another

    ids = [int(i) for i in ids]
    refuse_parts_of_another(geometry, ids)
    if not float(size) > 0.0:
        raise ValueError('the element size must be positive')
    flat = [i for i in ids if is_flat(geometry, i)]
    solid = [i for i in ids if i not in flat]
    pieces = [_flat_part(geometry, i, float(size), triangles) for i in flat]
    if solid:
        pieces.append(_outer_surface(geometry, solid, float(size), triangles, progress))
    part = pieces[0]
    for piece in pieces[1:]:
        join(part, piece)
    return part, {'method': 'shells', 'elements': len(part.elem_conn),
                  'nodes': int(part.num_nodes), 'flat': len(flat),
                  'surfaces': len(solid)}


def mesh_size(geometry: Any, ids) -> float:
    """The element size the parts are meshed into shells at by default:
    `solids.mesh_size`, with a flat part measured across its plane, its
    own lack of thickness left out.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id.

    Returns
    -------
    float
    """
    from .solids import is_flat
    from .solids import mesh_size as solid_size

    ids = [int(i) for i in ids]
    sizes = []
    for i in ids:
        if is_flat(geometry, i):
            # a slab far thicker than the part is wide, so the thinnest
            # feature found is one across the plane
            scratch, _r, _o = flat_scratch(geometry, i, 1e3 * _reach(geometry, i))
            sizes.append(solid_size(scratch, [i]))
    rest = [i for i in ids if not is_flat(geometry, i)]
    if rest:
        sizes.append(solid_size(geometry, rest))
    return min(sizes)


def _reach(geometry: Any, part: int) -> float:
    """How far across a flat part's first plate or disk is."""
    from .solids import FLAT, solid

    first = next(solid(geometry, i) for i in _tree(geometry, part)
                 if solid(geometry, i).kind in FLAT)
    return max(max(first.widths), 2.0 * first.radius)


def _tree(geometry: Any, part: int) -> list[int]:
    """The part and every solid it is made of, by id."""
    from .solids import solid

    found = [int(part)]
    for operand in solid(geometry, part).operands:
        found += _tree(geometry, operand)
    return found


def flat_scratch(geometry: Any, part: int, thickness: float) -> tuple[Any, np.ndarray, np.ndarray]:
    """The flat part laid out in its own frame as a slab `thickness`
    through: a scratch geometry holding it, with the rotation (its axes
    as rows) and the origin that put it back.

    The frame is the first plate's or disk's. Every plate becomes a
    block and every disk a cylinder, `thickness` along Z about Z = 0,
    and every solid cut from them runs through the slab, so that the
    brick mesher sees an extrusion one element thick. Refused: a plate
    or a disk off the first one's plane, and a solid cut from it that is
    a cone or a sphere, or that crosses the plane at an angle, since the
    cut is then not the same through the slab.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    part : int
        The flat part, by id.
    thickness : float
        How thick the slab is, as the geometry holds its coordinates.

    Returns
    -------
    (Geometry, numpy.ndarray, numpy.ndarray)
    """
    from ..rotate import angles_of
    from .geometry import Geometry
    from .solids import (
        FLAT,
        HOLE,
        OPERATIONS,
        PATTERN,
        SECTIONS,
        _frame,
        _offsets,
        piece_name,
        solid,
    )

    tree = _tree(geometry, part)
    items = [solid(geometry, i) for i in tree]
    first = next(item for item in items if item.kind in FLAT)
    rotation = np.array(_frame(first))
    origin = np.asarray(first.center, dtype=float)
    reach = max(max(first.widths), first.radius)
    snug = 1e-9 * reach

    def local(item):
        return (np.array(_frame(item)) @ rotation.T,
                (np.asarray(item.center, dtype=float) - origin) @ rotation.T)

    def square(frame, item):
        if not np.isclose(abs(frame[2, 2]), 1.0, atol=1e-9):
            raise ValueError(f'{piece_name(geometry, item.id)} crosses the plane of '
                             f'{piece_name(geometry, part)} at an angle: a cut from a '
                             'plate or a disk runs square to it')

    thick = 3.0 * thickness
    out = []
    for item in items:
        if item.kind in OPERATIONS:
            out.append(item)
            continue
        if item.kind == PATTERN:
            frame = np.array(_frame(item)) @ rotation.T
            if np.abs(_offsets(item) @ rotation.T[:, 2]).max() > snug:
                raise ValueError(f'{piece_name(geometry, item.id)} puts copies off the '
                                 f'plane of {piece_name(geometry, part)}')
            out.append(replace(item, angles=angles_of(frame)))
            continue
        frame, center = local(item)
        at = (float(center[0]), float(center[1]), 0.0)
        if item.kind in FLAT:
            if not np.isclose(abs(frame[2, 2]), 1.0, atol=1e-9) or abs(center[2]) > snug:
                raise ValueError(f'{piece_name(geometry, item.id)} is not on the plane of '
                                 f'{piece_name(geometry, part)}: a flat part meshes on '
                                 'one plane')
            if item.kind == 'plate':
                out.append(replace(item, kind='block', center=at, angles=angles_of(frame),
                                   widths=(*item.widths[:2], thickness)))
            else:
                out.append(replace(item, kind='cylinder', center=at,
                                   angles=(0.0, 0.0, 0.0), length=thickness))
            continue
        if item.kind in ('cone', 'sphere'):
            raise ValueError(f'{piece_name(geometry, item.id)} is a {item.kind}: a plate '
                             'or a disk is cut by blocks, cylinders, sections and holes')
        square(frame, item)
        if item.kind == 'cylinder':
            out.append(replace(item, center=at, angles=(0.0, 0.0, 0.0), length=thick))
        elif item.kind == 'block':
            out.append(replace(item, center=at, angles=angles_of(frame),
                               widths=(*item.widths[:2], thick)))
        elif item.kind in SECTIONS:
            out.append(replace(item, center=at, angles=angles_of(frame), length=thick))
        elif item.kind == HOLE:
            # it runs along its own -Z from its center: started above the
            # slab, through it, the counterbore of no use in a plate
            side = float(np.sign(frame[2, 2]))
            out.append(replace(item, center=(at[0], at[1], side * thick / 2),
                               angles=angles_of(frame),
                               dimensions=(item.dimensions[0], 0.0, 0.0, 0.0)))
        else:
            out.append(item)
    scratch = Geometry(node_id=[], node_xyz=np.empty((0, 3)),
                       length_unit=geometry.length_unit)
    kept = set(tree)
    scratch.solids = out + [s for s in geometry.solids if s.id not in kept]
    return scratch, rotation, origin


def _flat_part(geometry: Any, part: int, size: float, triangles: bool) -> Any:
    """One flat part's plates: its slab meshed one brick through, each
    brick's bottom face moved onto the plane and put back where the part
    stands."""
    from .geometry import Geometry, solid_faces
    from .solids import brick_mesh

    scratch, rotation, origin = flat_scratch(geometry, part, size)
    bricks, _made = brick_mesh(scratch, [part], size)
    index = {int(n): k for k, n in enumerate(bricks.node_id)}
    xyz = np.asarray(bricks.node_xyz, dtype=float)
    faces, groups = [], []
    for conn, group in zip(bricks.elem_conn, bricks.elem_group):
        conn = [int(n) for n in conn]
        z = xyz[[index[n] for n in conn], 2]
        for face in solid_faces(len(conn)):
            if np.all(z[list(face)] < 0.0):
                corners = [conn[k] for k in face]
                points = xyz[[index[n] for n in corners]]
                normal = np.cross(points[2] - points[0], points[3] - points[1])
                # the plate's own Z is the face's normal, as a plate
                # drawn there would have it
                faces.append(corners if normal[2] > 0.0 else corners[::-1])
                groups.append(int(group))
                break
    used = sorted({n for face in faces for n in face})
    flat = xyz[[index[n] for n in used]]
    flat[:, 2] = 0.0
    placed = origin + flat @ rotation
    conn, code, owner = _split(faces, groups, dict(zip(used, placed)), triangles)
    return Geometry(node_id=used, node_xyz=placed, elem_conn=conn, elem_type=[code] * len(conn),
                    elem_group=owner, group_id=list(bricks.group_id),
                    group_name=list(bricks.group_name), length_unit=geometry.length_unit)


def _split(faces, groups, where, triangles: bool):
    """The faces as plates, or each split on its shorter diagonal into
    two triangles: (connectivity, element code, element groups)."""
    if not triangles:
        return faces, QUAD, groups
    conn, owner = [], []
    for face, group in zip(faces, groups):
        if len(face) == 3:
            conn.append(face)
            owner.append(group)
            continue
        a, b, c, d = face
        if (np.linalg.norm(where[a] - where[c]) <= np.linalg.norm(where[b] - where[d])):
            conn += [[a, b, c], [a, c, d]]
        else:
            conn += [[a, b, d], [b, c, d]]
        owner += [group, group]
    return conn, TRIANGLE, owner


def _outer_surface(geometry: Any, ids, size: float, triangles: bool, progress: Any) -> Any:
    """The parts' outer surface: the faces of their bricks (their
    four-node tetrahedra for triangles) that no second element shares,
    each in its element's group."""
    from .geometry import ELEMENT_TYPES, Geometry, solid_faces
    from .solids import brick_mesh
    from .tetmesh import tet_mesh

    if triangles:
        volume, _made = tet_mesh(geometry, ids, size, quadratic=False, progress=progress,
                                 keep_slivers=True)
    else:
        volume, _made = brick_mesh(geometry, ids, size, progress=progress)
    faces: dict[tuple[int, ...], tuple[list[int], int]] = {}
    for conn, code, group in zip(volume.elem_conn, volume.elem_type, volume.elem_group):
        if ELEMENT_TYPES[int(code)][2] != 'volume':
            continue                       # a seam's rigid links
        conn = [int(n) for n in conn]
        for face in solid_faces(len(conn)):
            corners = [conn[k] for k in face]
            key = tuple(sorted(corners))
            if key in faces:
                del faces[key]
            else:
                faces[key] = (corners, int(group))
    kept = list(faces.values())
    used = sorted({n for corners, _g in kept for n in corners})
    index = {int(n): k for k, n in enumerate(volume.node_id)}
    xyz = np.asarray(volume.node_xyz, dtype=float)[[index[n] for n in used]]
    conn = [corners for corners, _g in kept]
    owner = [g for _c, g in kept]
    code = TRIANGLE if triangles else QUAD
    return Geometry(node_id=used, node_xyz=xyz, elem_conn=conn, elem_type=[code] * len(conn),
                    elem_group=owner, group_id=list(volume.group_id),
                    group_name=list(volume.group_name), length_unit=geometry.length_unit)
