"""Test and model geometry: nodes, coordinate systems, elements and element groups.

Node coordinates are stored in the global cartesian frame, in meters once
their unit is known. A geometry read from a file that does not declare
units (an sdynpy .npz, an exodus file, a UNV file without dataset 164)
keeps the file's coordinates as they are, and `length_unit` is None
until `define_units()` is called. Element types are identified by the
FE descriptor codes of UFF dataset 2412.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..units import from_si, si_transform
from .entities import EntityView
from .validate import IdArray, Ids
from .validate import ids as _ids

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem
    from .rigid import MassProperties

CS_TYPES = {0: 'cartesian', 1: 'cylindrical', 2: 'spherical'}

# UFF dataset 2412 descriptor code -> (name, nodes per element, render class)
# render class: 'line', 'face', 'volume', 'point'
#
# A face element's *name* says how many corners it has — 'tri' three,
# anything else four — which is how the viewer draws one without a
# second table of node counts to keep in step (`face_corners`). Every
# face entry here is named accordingly.
ELEMENT_TYPES = {
    11: ('rod2', 2, 'line'),
    21: ('beam2', 2, 'line'),
    22: ('beam2_tapered', 2, 'line'),
    23: ('beam3', 3, 'line'),
    24: ('beam3_parabolic', 3, 'line'),
    31: ('pipe2', 2, 'line'),
    32: ('pipe3', 3, 'line'),
    # 2412's plane families, which differ from each other in what the
    # solver does with them and not at all in what is drawn: a
    # triangle is three corners and a quadrilateral four, whichever
    # family the descriptor names. A mesh of plane-strain quads
    # (54) was refused outright until they were all written down
    # (Brandon, 2026-09-20).
    41: ('tri3', 3, 'face'),           # plane stress
    42: ('tri6', 6, 'face'),
    43: ('tri9', 9, 'face'),
    44: ('quad4', 4, 'face'),
    45: ('quad8', 8, 'face'),
    46: ('quad12', 12, 'face'),
    51: ('tri3_strain', 3, 'face'),    # plane strain
    52: ('tri6_strain', 6, 'face'),
    53: ('tri9_strain', 9, 'face'),
    54: ('quad4_strain', 4, 'face'),
    55: ('quad8_strain', 8, 'face'),
    56: ('quad12_strain', 12, 'face'),
    61: ('tri3_plate', 3, 'face'),     # plate
    62: ('tri6_plate', 6, 'face'),
    63: ('tri9_plate', 9, 'face'),
    64: ('quad4_plate', 4, 'face'),
    65: ('quad8_plate', 8, 'face'),
    66: ('quad12_plate', 12, 'face'),
    71: ('tri3_membrane', 3, 'face'),  # membrane
    72: ('tri6_membrane', 6, 'face'),
    73: ('tri9_membrane', 9, 'face'),
    74: ('quad4_membrane', 4, 'face'),
    75: ('quad8_membrane', 8, 'face'),
    76: ('quad12_membrane', 12, 'face'),
    81: ('tri3_axisym', 3, 'face'),    # axisymmetric solid
    82: ('tri6_axisym', 6, 'face'),
    84: ('quad4_axisym', 4, 'face'),
    85: ('quad8_axisym', 8, 'face'),
    91: ('trishell3', 3, 'face'),      # thin shell
    92: ('trishell6', 6, 'face'),
    93: ('trishell9', 9, 'face'),
    94: ('quadshell4', 4, 'face'),
    95: ('quadshell8', 8, 'face'),
    96: ('quadshell12', 12, 'face'),
    # thick shell: a solid cell by another name, drawn as one
    101: ('wedge6_thick', 6, 'volume'),
    102: ('wedge15_thick', 15, 'volume'),
    103: ('wedge24_thick', 24, 'volume'),
    104: ('hex8_thick', 8, 'volume'),
    105: ('hex20_thick', 20, 'volume'),
    106: ('hex32_thick', 32, 'volume'),
    111: ('tet4', 4, 'volume'),
    112: ('wedge6', 6, 'volume'),
    113: ('wedge15', 15, 'volume'),
    114: ('wedge24', 24, 'volume'),
    115: ('hex8', 8, 'volume'),
    116: ('hex20', 20, 'volume'),
    117: ('hex27', 27, 'volume'),
    118: ('tet10', 10, 'volume'),
    # rigid links and the axisymmetric shells, all drawn as lines. A
    # rigid element joins however many nodes the file states; the two
    # it is drawn between are the two it names first.
    121: ('rigid_bar', 2, 'line'),
    122: ('rigid_element', 2, 'line'),
    171: ('shell_axisym2', 2, 'line'),
    172: ('shell_axisym3', 3, 'line'),
    # Springs, dampers and gaps come in two kinds: between two nodes,
    # which draws as the line it is, and from a node to ground, which
    # is one node and draws as a point. 136 was one node here until
    # 2026-10-08, to fit a Nastran reader that kept a CELAS at its first
    # grid; it is two, as the universal file has it, since the reader
    # keeps both grids and the stiffness.
    136: ('spring', 2, 'line'),
    137: ('spring_rotational', 2, 'line'),
    138: ('spring_ground', 1, 'point'),
    139: ('spring_ground_rotational', 1, 'point'),
    141: ('damper', 2, 'line'),
    142: ('damper_ground', 1, 'point'),
    151: ('gap', 2, 'line'),
    152: ('gap_ground', 1, 'point'),
    161: ('mass', 1, 'point'),
    # Codes from 200 up are visualdynamics's own, not UFF 2412's: the
    # universal file has no pyramid descriptor — I-DEAS meshed without
    # them — but exodus files carry them routinely in the transition
    # regions of hex-dominant meshes, and refusing the whole mesh for
    # its pyramids would refuse most of what a modern solver writes.
    # The UNV writer refuses these two by name instead.
    201: ('pyramid5', 5, 'volume'),
    202: ('pyramid13', 13, 'volume'),
}

#: the element families the tree lists under a geometry (Brandon,
#: 2026-09-30: one sub-item per element type, element groups under each), in
#: listing order, with how each is told apart: drawn shape, then the
#: corner count of the first-order element and its higher-order
#: siblings. Every 2412 code lands in one of them.
FAMILIES = ('beams', 'triangles', 'quads', 'tetras', 'wedges', 'hexes')
#: the two the vocabulary also holds and the tree lists only when
#: present: point elements (masses, grounded springs and dampers) and
#: pyramids, which no file here has carried yet
RARE_FAMILIES = ('points', 'pyramids')
#: the families whose elements fill a volume: two of them on one set of
#: nodes are one cell counted twice (`duplicate_elements`)
SOLID_FAMILIES = frozenset({'tetras', 'wedges', 'hexes', 'pyramids'})

FAMILY_LABELS = {'beams': 'Beams', 'triangles': 'Triangles',
                 'quads': 'Quads', 'tetras': 'Tetras', 'wedges': 'Wedges',
                 'hexes': 'Hexes', 'points': 'Points', 'pyramids': 'Pyramids'}
_FACE_FAMILIES = {3: 'triangles', 6: 'triangles', 9: 'triangles',
                  4: 'quads', 8: 'quads', 12: 'quads'}
_VOLUME_FAMILIES = {4: 'tetras', 10: 'tetras', 6: 'wedges', 15: 'wedges',
                    24: 'wedges', 8: 'hexes', 20: 'hexes', 27: 'hexes',
                    32: 'hexes', 5: 'pyramids', 13: 'pyramids'}


def element_family(code: int) -> str:
    """Which family a 2412 element code belongs to.

    Parameters
    ----------
    code : int
        The element type code.

    Returns
    -------
    str
        One of `FAMILIES`, or 'points' or 'pyramids' (`RARE_FAMILIES`).
    """
    _name, count, shape = ELEMENT_TYPES[int(code)]
    if shape == 'line':
        return 'beams'
    if shape == 'point':
        return 'points'
    if shape == 'face':
        return _FACE_FAMILIES[count]
    return _VOLUME_FAMILIES[count]



def placed(local: ArrayLike, kind: int, matrix: ArrayLike) -> np.ndarray:
    """A point given in a coordinate system, converted to global cartesian.

    `kind` is the system's `CS_TYPES` code and `matrix` its `(4, 3)`
    rows: three directions, then the origin. A cylindrical system gives
    (r, theta, z) and a spherical one (r, theta from +z, phi), with the
    angles in degrees in both, which is the convention of the universal
    file, Nastran's CORD2C and CORD2S, and sdynpy.
    """
    local = np.asarray(local, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if int(kind) == 1:
        r, theta, z = local
        theta = np.radians(theta)
        local = np.array([r * np.cos(theta), r * np.sin(theta), z])
    elif int(kind) == 2:
        r, theta, phi = local
        theta, phi = np.radians(theta), np.radians(phi)
        local = np.array([r * np.sin(theta) * np.cos(phi),
                          r * np.sin(theta) * np.sin(phi),
                          r * np.cos(theta)])
    return matrix[3] + local @ matrix[:3]


def written_in(point: ArrayLike, kind: int, matrix: ArrayLike) -> np.ndarray:
    """The inverse of `placed`: a global cartesian point in the
    coordinates of a system, so a file can be written with its nodes in
    the systems they were read in."""
    point = np.asarray(point, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    local = (point - matrix[3]) @ matrix[:3].T
    if int(kind) == 1:
        x, y, z = local
        return np.array([np.hypot(x, y), np.degrees(np.arctan2(y, x)), z])
    if int(kind) == 2:
        x, y, z = local
        r = float(np.sqrt(x * x + y * y + z * z))
        theta = np.degrees(np.arccos(z / r)) if r else 0.0
        return np.array([r, theta, np.degrees(np.arctan2(y, x))])
    return local


def _frames(cs_id, cs_type, cs_matrix):
    ids = [int(i) for i in np.asarray(cs_id).ravel()]
    kinds = np.asarray(cs_type).ravel()
    matrices = np.asarray(cs_matrix, dtype=float).reshape(-1, 4, 3)
    return {code: (int(kinds[k]), matrices[k]) for k, code in enumerate(ids)}


def to_global(node_xyz: ArrayLike, node_def_cs: ArrayLike, cs_id: Ids,
              cs_type: ArrayLike, cs_matrix: ArrayLike) -> np.ndarray:
    """Node coordinates given in each node's own placement system,
    converted to global cartesian, which is what `Geometry` stores.

    A file gives each node's position in the coordinate system the node
    is placed in, its `def_cs`. Reading those numbers as global would put
    the node in the wrong place. A node whose system is not among the
    given systems, or is the global system, keeps its coordinates as
    written.

    Parameters
    ----------
    node_xyz : array-like
        `(nodes, 3)` as the source wrote them.
    node_def_cs : array-like
        The frame each node is placed in.
    cs_id, cs_type, cs_matrix : array-like
        The frames themselves, as `Geometry` holds them.

    Returns
    -------
    numpy.ndarray
        `(nodes, 3)` in the global cartesian frame.
    """
    # A geometry with local placement systems drew with every node in
    # the wrong place while the file's numbers were read as global
    # (Brandon, 2026-09-20).
    xyz = np.asarray(node_xyz, dtype=float).reshape(-1, 3)
    frames = _frames(cs_id, cs_type, cs_matrix)
    out = xyz.copy()
    for k, code in enumerate(np.asarray(node_def_cs).ravel()[:len(xyz)]):
        found = frames.get(int(code))
        if found is not None:
            out[k] = placed(xyz[k], found[0], found[1])
    return out


def to_local(node_xyz: ArrayLike, node_def_cs: ArrayLike, cs_id: Ids,
             cs_type: ArrayLike, cs_matrix: ArrayLike) -> np.ndarray:
    """The inverse of `to_global`, for a writer that gives each node in
    the coordinate system it is placed in."""
    xyz = np.asarray(node_xyz, dtype=float).reshape(-1, 3)
    frames = _frames(cs_id, cs_type, cs_matrix)
    out = xyz.copy()
    for k, code in enumerate(np.asarray(node_def_cs).ravel()[:len(xyz)]):
        found = frames.get(int(code))
        if found is not None:
            out[k] = written_in(xyz[k], found[0], found[1])
    return out


def reconcile_type(code: int, node_count: int) -> int:
    """The descriptor that matches the number of nodes an element has,
    when the file's descriptor and node count disagree.

    A file gives both, and they can contradict each other. The node
    count is taken as correct, and a descriptor with the wrong count is
    replaced by the one in the same family with that many nodes: 41 (a
    plane-stress triangle) with four nodes becomes 44, the plane-stress
    quadrilateral. Dataset 2412 numbers each family within one decade,
    triangles before quadrilaterals, which is how the replacement is
    found. When no descriptor in the family fits, the original is kept,
    since an element that cannot be identified is better kept as written
    than renamed to a guess.

    Parameters
    ----------
    code : int
        The descriptor the file gave.
    node_count : int
        How many nodes the file actually listed for it.

    Returns
    -------
    int
        The descriptor to use.
    """
    # A file wrote a four-node element as descriptor 41, and it drew as a
    # triangle with the fourth node dropped (Brandon, 2026-09-20).
    code = int(code)
    known = ELEMENT_TYPES.get(code)
    if known is None or node_count < 1 or known[1] == node_count:
        return code
    family, render = code // 10, known[2]
    for other, (_name, count, kind) in ELEMENT_TYPES.items():
        if other // 10 == family and kind == render and count == node_count:
            return other
    return code


def face_corners(code: int) -> int:
    """How many corners a face element is drawn with: three for a
    triangle and four for a quadrilateral, whatever family its descriptor
    names and however many nodes it has.

    It is read from the name in `ELEMENT_TYPES` and not from the node
    count, because a nine-node cubic triangle and a nine-node
    quadrilateral have the same count.

    Parameters
    ----------
    code : int
        The element's descriptor.

    Returns
    -------
    int
        3 or 4.
    """
    # by name, not node count (Brandon, 2026-09-20)
    name = ELEMENT_TYPES[int(code)][0]
    return 3 if name.startswith('tri') else 4


#: the faces of a solid cell by its node count, as corner positions in
#: the cell's own node order: the hexahedron's six, the wedge's two
#: triangles and three quads, the tetrahedron's four. Bottom face then
#: top, the same way round, as UFF 2412 and Nastran number them.
_SOLID_FACES = {
    8: ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7)),
    6: ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    4: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
}


def solid_faces(node_count: int) -> tuple[tuple[int, ...], ...]:
    """The faces of a solid cell with this many nodes, each face as
    positions in the cell's node list.

    A higher-order cell is drawn by its corner nodes, which come first
    in its node list. An unknown node count gives an empty tuple.

    Parameters
    ----------
    node_count : int
        How many nodes the cell names.

    Returns
    -------
    tuple of tuple of int
    """
    if node_count in (8, 20, 27):
        return _SOLID_FACES[8]
    if node_count in (6, 15, 24):
        return _SOLID_FACES[6]
    if node_count in (4, 10):
        return _SOLID_FACES[4]
    if node_count in (5, 13):                 # a pyramid: a quad and four
        return ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
    return ()


#: per corner of a solid, three corners it shares an edge with: the
#: three edges whose triple product is that corner's share of the volume,
#: whose sign says whether the cell is still the right way out
_CORNER_EDGES = {
    8: ((1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7),
        (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)),
    6: ((1, 2, 3), (2, 0, 4), (0, 1, 5), (5, 4, 0), (3, 5, 1), (4, 3, 2)),
    4: ((1, 2, 3),),
    5: ((1, 3, 4), (2, 0, 4), (3, 1, 4), (0, 2, 4)),
}


def _turned(code: int, before: NDArray, after: NDArray) -> bool:
    """Whether moving an element's nodes from `before` to `after` turns
    it inside out: a solid's corner volume, or a face's normal at a
    corner, changing sign. A line has no inside: one drawn to nothing is
    a link across the gap the move closes — the Frame-Wing's screw ties,
    2026-10-11 — which the merge that follows removes
    (`merge_collapses`)."""
    family = element_family(code)
    if family in SOLID_FAMILIES:
        count = {'hexes': 8, 'wedges': 6, 'tetras': 4, 'pyramids': 5}[family]
        for corner, (a, b, c) in enumerate(_CORNER_EDGES[count]):
            then = np.linalg.det(before[[a, b, c]] - before[corner])
            now = np.linalg.det(after[[a, b, c]] - after[corner])
            if then * now <= 0.0:
                return True
        return False
    if family in ('quads', 'triangles'):
        corners = face_corners(code)
        for i in range(corners):
            j, k = (i + 1) % corners, (i - 1) % corners
            then = np.cross(before[j] - before[i], before[k] - before[i])
            now = np.cross(after[j] - after[i], after[k] - after[i])
            if float(np.dot(then, now)) <= 0.0:
                return True
        return False
    return False


class View:
    """The direction a geometry is viewed from in 3-D: where the eye is,
    seen from the model's center, and which way is up on screen.

    A view is only a direction. Every view is fitted to what it shows,
    so there is no distance or zoom to store. Set on a geometry, it is
    the view the app opens the geometry and everything drawn on it in,
    the view Reset View returns to, and the view of the report's scenes
    and exported figures. A model built with Y up then opens upright
    without its nodes being rotated. Raises ValueError for a zero or
    non-finite direction, or an `up` along the line of sight.

    Parameters
    ----------
    eye : sequence of 3 floats, default (1, 1, 1)
        The direction from the model toward the eye, in the global frame;
        any length.
    up : sequence of 3 floats, default (0, 0, 1)
        Which way is up on screen. Only its part across the line of sight
        counts, so it need not be exactly perpendicular to `eye`.

    Examples
    --------
    A model built with Y up, seen from the front, above and to the right:

    >>> geometry.view = View(eye=(1, 1, -1), up=(0, 1, 0))  # doctest: +SKIP
    """

    # One view for the app, Reset View, the report and exported figures
    # (Brandon, 2026-09-27).
    __slots__ = ('eye', 'up')

    def __init__(self, eye: ArrayLike = (1.0, 1.0, 1.0),
                 up: ArrayLike = (0.0, 0.0, 1.0)) -> None:
        # kept as written, so a view reads back the way it was stated;
        # every use goes through `_unit`
        self.eye: tuple[float, ...] = tuple(
            float(v) for v in np.asarray(eye, dtype=float).reshape(3))
        self.up: tuple[float, ...] = tuple(
            float(v) for v in np.asarray(up, dtype=float).reshape(3))
        eye_length = float(np.linalg.norm(self.eye))
        if not np.isfinite(eye_length) or eye_length == 0.0:
            raise ValueError('a view needs a direction to look from')
        up_length = float(np.linalg.norm(self.up))
        if not np.isfinite(up_length) or up_length == 0.0:
            raise ValueError('a view needs a direction for up')
        # the part of `up` across the line of sight is what the screen
        # shows as up; along it, there is none to show
        if np.linalg.norm(np.cross(self.eye, self.up)) < \
                1e-9 * eye_length * up_length:
            raise ValueError('up cannot lie along the line of sight')

    def _unit(self) -> tuple[np.ndarray, np.ndarray]:
        """(eye, up) as unit vectors, up made square to the eye."""
        eye = np.asarray(self.eye) / np.linalg.norm(self.eye)
        up = np.asarray(self.up) - np.dot(self.up, eye) * eye
        return eye, up / np.linalg.norm(up)

    def basis(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """``(right, up)`` on screen, as unit vectors in the global frame.

        Right is the line of sight crossed with up, and up is `up` made
        perpendicular to the line of sight.
        """
        eye, up = self._unit()
        return (tuple(float(v) for v in np.cross(up, eye)),
                tuple(float(v) for v in up))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, View):
            return NotImplemented
        return all(np.allclose(a, b) for a, b in zip(self._unit(),
                                                     other._unit()))

    def __hash__(self) -> int:
        eye, up = self._unit()
        return hash((tuple(np.round(eye, 9)), tuple(np.round(up, 9))))

    def __repr__(self) -> str:
        def short(vector):
            return '(' + ', '.join(repr(round(v, 4) + 0.0)
                                   for v in vector) + ')'
        return f'View(eye={short(self.eye)}, up={short(self.up)})'


#: how a geometry with no view of its own opens: from +X+Y+Z, Z up, the
#: isometric every 3-D view opened on before a geometry could say
#: otherwise
DEFAULT_VIEW = View()


class Geometry:
    """Nodes, coordinate systems, elements and element groups.

    The parameters are array-likes with the names of the attributes
    below; `elem_conn` holds one integer array of node ids per element.
    Coordinates are in the global cartesian frame, in meters once
    `length_unit` is declared. Without coordinate systems the geometry
    gets one, the global cartesian system with id 1. An element whose
    descriptor does not match its node count is corrected
    (`reconcile_type`), and an element group holding more than one
    element family is split (`split_groups_by_family`). Raises
    ValueError when the result is not consistent (`validate`).

    The arrays below hold the data; `nodes`, `coordinate_systems`,
    `elements` and `groups` are views of them, which is how the project
    tree lists a geometry and usually the easiest way for a script to
    use one. A view is not a copy, so writing through a row writes into
    the geometry.

    There are no separate tracelines. A line drawn through nodes is an
    element group of two-node line elements with no properties:
    `add_beams` creates one, and `drawn_lines` returns them for the file
    formats that store tracelines separately. The same element group
    becomes part of the structure once it is given a section.

    Attributes:
        node_id: Every node's id. Unique, because elements and placement
            refer to nodes by id.
        node_xyz: `(nodes, 3)` coordinates, in meters once
            `length_unit` is declared and the file's numbers before
            that.
        node_def_cs: The coordinate system each node is placed in.
        node_disp_cs: The coordinate system each node is measured in,
            which is the system a shape's values at that node are in.
        node_color: Palette index per node.
        cs_id: Coordinate system ids. Unique, for the same reason as
            node ids.
        cs_name: A name per system, often empty.
        cs_type: 0 cartesian, 1 cylindrical, 2 spherical (`CS_TYPES`).
        cs_matrix: `(systems, 4, 3)`: three direction rows, then the
            origin, so `cs_matrix[i, 3]` is the origin of system *i*.
        elem_id: Element ids. They are only labels; nothing refers to
            them.
        elem_type: UFF dataset 2412 descriptor code per element
            (`ELEMENT_TYPES` names them and says how each is drawn).
        elem_color: Palette index per element.
        elem_conn: One array of node ids per element.
        elem_group: The id of the element group each element belongs to.
        group_id: The declared element groups. Unique, since elements
            refer to them by id.
        group_name: A name per element group, such as `'wing'` or
            `'arm front left'`. This is where a mesh records that a
            region is a different part from its neighbor, and
            `fem.Model.from_geometry` reads a member's section from it.
        group_properties: `{element group id: fem.GroupProperties}`, the
            material and thickness or section of each element group that
            has them.
        length_unit: The unit of the coordinates, or `None` while it has
            not been declared, in which case they are the file's own
            numbers and nothing has been scaled.
        mass_properties: The reference point, mass and inertia for the
            rigid-body shapes (`core.rigid.MassProperties`), or None.
        view: The direction the geometry opens in 3-D (`View`), or None
            for `DEFAULT_VIEW`.
        solids: The solid shapes of the parts before they are meshed
            (`core.solids`).
    """

    # No tracelines (2026-09-30): one storage for one drawn idea, PLAN.md
    # "Geometry by element family".
    dimension = 'length'

    def __init__(self, node_id: Ids, node_xyz: ArrayLike,
                 node_def_cs: ArrayLike | None = None,
                 node_disp_cs: ArrayLike | None = None,
                 node_color: ArrayLike | None = None,
                 cs_id: Ids | None = None,
                 cs_name: Sequence[str] | None = None,
                 cs_type: ArrayLike | None = None,
                 cs_matrix: ArrayLike | None = None,
                 elem_id: Ids | None = None,
                 elem_type: ArrayLike | None = None,
                 elem_color: ArrayLike | None = None,
                 elem_conn: Sequence[ArrayLike] | None = None,
                 elem_group: Ids | None = None,
                 group_id: Ids | None = None,
                 group_name: Sequence[str] | None = None,
                 length_unit: str | None = None,
                 group_properties: Mapping[int, Any] | None = None) -> None:
        # coordinates are taken as given; length_unit records what they are
        # in (None = undefined, values are the file's raw numbers)
        self.length_unit: str | None = length_unit
        n = len(node_id)
        self.node_id: IdArray = _ids(node_id, 'node ids', unique=True)
        self.node_xyz: NDArray[np.float64] = np.asarray(
            node_xyz, dtype=np.float64).reshape(n, 3)
        self.node_def_cs: IdArray = self._default(node_def_cs, n, 1)
        self.node_disp_cs: IdArray = self._default(node_disp_cs, n, 1)
        self.node_color: NDArray[np.int64] = self._default(node_color, n, 1)

        if cs_id is None:
            cs_id, cs_name, cs_type = [1], [''], [0]
            cs_matrix = np.vstack([np.eye(3), np.zeros(3)])[np.newaxis]
        c = len(cs_id)
        self.cs_id: IdArray = _ids(cs_id, 'coordinate system ids', unique=True)
        self.cs_name: list[str] = (list(cs_name) if cs_name is not None
                                   else [''] * c)
        self.cs_type: NDArray[np.int64] = self._default(cs_type, c, 0)
        self.cs_matrix: NDArray[np.float64] = (
            np.asarray(cs_matrix, dtype=np.float64).reshape(c, 4, 3)
            if cs_matrix is not None
            else np.tile(np.vstack([np.eye(3), np.zeros(3)]), (c, 1, 1)))


        e = len(elem_conn) if elem_conn is not None else 0
        self.elem_id: IdArray = (
            _ids(elem_id, 'element ids') if elem_id is not None
            else self._default(None, e, None, arange=True))
        self.elem_type: NDArray[np.int64] = self._default(elem_type, e, 0)
        self.elem_color: NDArray[np.int64] = self._default(elem_color, e, 1)
        # Which element group each element belongs to, and what the element groups are
        # called. This is how a mesh says "these elements are the wing and
        # those are the tail": exodus calls them element blocks, and it is
        # the only place a file records that a region is made of something
        # different from its neighbor. Read without keeping it, a file of
        # two element groups comes back as one anonymous group on the way
        # out.
        self.elem_group: IdArray = (
            _ids(elem_group, 'element group ids') if elem_group is not None
            else self._default(None, e, 1))
        found = (np.unique(self.elem_group) if e
                 else np.array([], dtype=np.int64))
        self.group_id: IdArray = (
            _ids(group_id, 'element group ids', unique=True)
            if group_id is not None else found)
        self.group_name: list[str] = (list(group_name) if group_name is not None
                                      else [''] * len(self.group_id))
        # What each element group is made of, when the geometry is a model in
        # waiting: {element group id: fem.GroupProperties} — a material and a
        # thickness for an element group of plates, a material and a section for
        # an element group of beams. This is how every finite element format
        # states a structure, one property set per element group of one
        # element type, and it is what `fem.Model.from_geometry` builds
        # from (Brandon, 2026-09-25). Held here rather than in the
        # model so it is saved with the geometry and comes back. The
        # values are the FE module's own classes; this module only
        # carries them.
        self.group_properties: dict[int, Any] = (
            {int(k): v for k, v in group_properties.items()}
            if group_properties else {})
        self.elem_conn: list[IdArray] = [
            np.asarray(c, dtype=np.int64) for c in (elem_conn or [])]
        # a source states both what an element is and which nodes it
        # joins, and the two can contradict each other — a four-node
        # element carrying a triangle's descriptor drew as a triangle,
        # a node short (Brandon, 2026-09-20). The nodes are the
        # element; `reconcile_type` mends the label. Here rather than
        # in each reader, so every source is read the same way.
        if len(self.elem_type) == len(self.elem_conn):
            self.elem_type = np.array(
                [reconcile_type(int(code), len(conn))
                 for code, conn in zip(self.elem_type, self.elem_conn)],
                dtype=np.int64) if len(self.elem_conn) else self.elem_type
        #: the reference point, mass and inertia the rigid-body view
        #: sets, riding the geometry the way averaging rides a time
        #: history (`core.rigid`); None until set or adopted
        self.mass_properties: MassProperties | None = None
        #: how it opens in 3-D (`View`); None opens on `DEFAULT_VIEW`
        self.view: View | None = None
        #: the shapes its parts are made of before they are meshed — blocks,
        #: cylinders and booleans of them (`core.solids`, 2026-10-11); not
        #: elements, and read by nothing that solves or draws a mesh
        self.solids: list = []
        # one element group, one family (2026-09-30): a source that mixes them —
        # UNV 2412 and the sdynpy layout carry no element groups at all, so every
        # element arrives in one — is split on arrival rather than
        # refused, the same way a mislabeled type is mended above. Here
        # rather than in each reader, so every source is read the same way.
        self.split_groups_by_family()
        self.validate()

    @property
    def opening_view(self) -> View:
        """The view the geometry opens in 3-D: its own `view`, or
        `DEFAULT_VIEW` (from +X+Y+Z, Z up) when it has none."""
        return self.view if self.view is not None else DEFAULT_VIEW

    @staticmethod
    def _default(value, n, fill, arange=False):
        if value is not None:
            return np.asarray(value, dtype=np.int64)
        return np.arange(1, n + 1, dtype=np.int64) if arange else np.full(n, fill, np.int64)

    def validate(self) -> None:
        """Check that the geometry is consistent, and raise ValueError
        naming the first problem found.

        Node, coordinate system and element group ids must be unique,
        every element's nodes must exist, every element must name an
        element group the geometry has, every element type must be in
        `ELEMENT_TYPES`, and each element group must hold one element
        family.
        """
        # An id that other data points at has to mean one thing:
        # connectivity names nodes by id, and a node names the systems it
        # is placed and measured in. Element ids are labels — nothing
        # refers to them — so they are not held to this.
        # the same helper the constructor uses, so a duplicate introduced
        # by an edit is refused in the same words as one handed in
        for label, values in (('node ids', self.node_id),
                              ('coordinate system ids', self.cs_id)):
            _ids(values, label, unique=True)
        known = set(self.node_id.tolist())
        for i, conn in enumerate(self.elem_conn):
            missing = set(conn.tolist()) - known
            if missing:
                raise ValueError(
                    f"element {i} references unknown node ids {sorted(missing)}")
        if len(self.group_name) != len(self.group_id):
            raise ValueError(
                f'{len(self.group_name)} element group names for '
                f'{len(self.group_id)} element groups')
        _ids(self.group_id, 'element group ids', unique=True)
        stray = set(np.unique(self.elem_group).tolist()) - set(
            self.group_id.tolist())
        if stray:
            raise ValueError(
                'elements name element groups the geometry does not have: '
                + ', '.join(str(b) for b in sorted(stray)))
        for code in np.unique(self.elem_type) if len(self.elem_type) else []:
            if int(code) not in ELEMENT_TYPES:
                raise ValueError(f"Unknown element type code {int(code)}")
        # one element group, one family (2026-09-30): an element group's properties are
        # one kind, and the tree lists an element group under its family
        mixed = self.mixed_groups()
        if mixed:
            raise ValueError(
                'an element group holds one element family; '
                + ', '.join(f'element group {b} holds {" and ".join(fams)}'
                            for b, fams in mixed.items())
                + ' — split_groups_by_family() sorts them out')

    @property
    def num_nodes(self) -> int:
        """How many nodes the geometry defines."""
        return len(self.node_id)

    # ---- the four groups, as the project tree lists them ---------------------
    # Views, not copies: `geometry.nodes.xyz` is the geometry's own array,
    # and a row written through writes into the geometry. Built on each
    # call, which is cheap and leaves nothing to go stale.

    @property
    def nodes(self) -> EntityView:
        """Every node: `ids`, `xyz`, `colors`, `placement_systems` and
        `displacement_systems` (an `EntityView`)."""
        return EntityView(self, 'nodes')

    @property
    def coordinate_systems(self) -> EntityView:
        """Every coordinate system: `ids`, `names`, `types` and `matrices`
        (an `EntityView`)."""
        return EntityView(self, 'coordinate_systems')

    @property
    def elements(self) -> EntityView:
        """Every element: `ids`, `types`, `colors`, `groups` and `nodes`
        (an `EntityView`)."""
        return EntityView(self, 'elements')

    @property
    def groups(self) -> EntityView:
        """Every element group: `ids` and `names` (an `EntityView`).

        An element group does not store its elements. Which elements are
        in one is read from `elem_group` (`elements_in`), so moving an
        element to another element group is an edit to the element.
        """
        return EntityView(self, 'groups')

    def node_index(self, node_ids: Ids) -> np.ndarray:
        """Positions of the given node ids in the node arrays.

        Every id must be one the geometry has (`contains_nodes` checks);
        an id that is not present gives a wrong position or an
        IndexError, not a clear error.

        Parameters
        ----------
        node_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        numpy.ndarray
            Each identifier's row in the node arrays.
        """
        order = np.argsort(self.node_id)
        pos = np.searchsorted(self.node_id, node_ids, sorter=order)
        return order[pos]

    @property
    def extent(self) -> tuple[np.ndarray, np.ndarray]:
        """``(min_xyz, max_xyz)`` of the nodes, in meters once units are defined."""
        return self.node_xyz.min(axis=0), self.node_xyz.max(axis=0)

    def contains_nodes(self, node_ids: Ids) -> np.ndarray:
        """Boolean array: which of `node_ids` this geometry defines.

        Parameters
        ----------
        node_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        numpy.ndarray
            A boolean per identifier: whether the geometry has it.
        """
        return np.isin(np.asarray(node_ids), self.node_id)

    def missing_dofs(self, dofs: Sequence[str]) -> list[str]:
        """The DOF strings whose node this geometry does not define.

        Parameters
        ----------
        dofs : sequence of str
            The degrees of freedom to check, such as '101Z+'.

        Returns
        -------
        list of str
            The DOFs whose node the geometry does not have, including
            any DOF that names no node. A data object with such DOFs
            is not compatible with this geometry.
        """
        from .data import parse_dof

        nodes = [parse_dof(dof)[0] for dof in dofs]
        known = self.contains_nodes([-1 if n is None else n for n in nodes])
        return [dof for dof, present in zip(dofs, known) if not present]

    @property
    def units_defined(self) -> bool:
        """Whether the length unit of the coordinates has been declared.
        False until `define_units` is called."""
        return self.length_unit is not None

    def suggest_mass_properties(self) -> MassProperties:
        """Mass properties with the centroid of the nodes as the reference
        point and no mass, which give unit rigid-body shapes about the
        middle of the model.

        The rigid-body panel opens with these values, and
        `generate_rigid_body_modes` uses them when nothing has been set.
        They are the settings the virtual point transformation needs most
        often. Raises ValueError when the geometry has no nodes.

        Returns
        -------
        MassProperties
            The centroid, unscaled.
        """
        from .rigid import MassProperties

        if self.num_nodes == 0:
            raise ValueError('the geometry has no nodes to take the '
                             'centroid of')
        return MassProperties(tuple(self.node_xyz.mean(axis=0)))

    def define_units(self, length_unit: str) -> Geometry:
        """Declare the length unit of the coordinates, converting them to SI.

        The node coordinates, the coordinate system origins and the
        solids' lengths are converted. Declaring again reinterprets the
        original values instead of scaling twice, so a wrong unit can be
        corrected.

        Parameters
        ----------
        length_unit : str
            The unit the coordinates are in, such as 'm' or 'in'.

        Returns
        -------
        Geometry
            Self, converted to SI in place.
        """
        scale, _ = si_transform(length_unit, 'length')
        raw = (self.node_xyz if self.length_unit is None
               else from_si(self.node_xyz, self.length_unit, 'length'))
        self.node_xyz = raw * scale
        origins = self.cs_matrix[:, 3, :]
        raw_origins = (origins if self.length_unit is None
                       else from_si(origins, self.length_unit, 'length'))
        self.cs_matrix[:, 3, :] = raw_origins * scale
        # the solids' lengths follow the coordinates (their angles do not)
        raw_scale = (1.0 if self.length_unit is None
                     else 1.0 / si_transform(self.length_unit, 'length')[0])
        self.solids = [s.scaled(raw_scale * scale) for s in self.solids]
        self.length_unit = length_unit
        return self

    def undefine_units(self) -> Geometry:
        """Remove the length unit and restore the original coordinates; returns self."""
        if self.length_unit is None:
            return self
        self.node_xyz = from_si(self.node_xyz, self.length_unit, 'length')
        self.cs_matrix[:, 3, :] = from_si(self.cs_matrix[:, 3, :],
                                          self.length_unit, 'length')
        self.solids = [s.scaled(1.0 / si_transform(self.length_unit, 'length')[0])
                       for s in self.solids]
        self.length_unit = None
        return self

    # ---- adding -------------------------------------------------------------

    def _next_id(self, array):
        return int(array.max()) + 1 if len(array) else 1

    def dof_direction(self, dof: str) -> np.ndarray | None:
        """A DOF's direction in global coordinates, from the coordinate
        system its node is measured in.

        '101X+' at a node whose `node_disp_cs` is rotated is not global
        X. The report's table of control channels uses this to show which
        global axis each channel is nearest and how far it is from it. A
        rotational DOF points along its axis, as the DOF arrows draw it.
        A node whose system is not in the geometry is taken to be in the
        global system.

        Parameters
        ----------
        dof : str
            The DOF, such as '101X+' or '1313RZ-'.

        Returns
        -------
        numpy.ndarray or None
            A unit vector, or None when the DOF names no node this
            geometry has, has no axis (a node number alone), or is at a
            node measured in a cylindrical or spherical system, whose
            local axes depend on the node's position and are not
            computed here.
        """
        # The report's control-channel table reads which global axis a
        # channel is nearest and how far off it is (Brandon, 2026-09-19).
        from .data import parse_dof

        node, direction = parse_dof(dof)
        direction = str(direction).upper()
        letter, sign = direction.lstrip('R')[:1], direction.lstrip('R')[1:]
        axis = {'X': 0, 'Y': 1, 'Z': 2}.get(letter)
        if node is None or axis is None or sign not in ('', '+', '-'):
            return None
        rows = np.flatnonzero(self.node_id == int(node))
        if not rows.size:
            return None
        frames = np.flatnonzero(self.cs_id == self.node_disp_cs[rows[0]])
        if frames.size:
            if int(self.cs_type[frames[0]]) != 0:
                return None
            vector = np.asarray(self.cs_matrix[frames[0], axis], dtype=float)
        else:
            # a frame the table does not list is the global one: a
            # geometry with no systems at all measures every node in it
            vector = np.eye(3)[axis]
        norm = float(np.linalg.norm(vector))
        if not norm:
            return None
        return (-1.0 if sign == '-' else 1.0) * vector / norm

    def add_node(self, xyz: ArrayLike, node_id: int | None = None,
                 color: int = 1, def_cs: int | None = None,
                 disp_cs: int | None = None) -> int:
        """Add a node at `xyz` and return its id.

        Raises ValueError if a node with `node_id` already exists.

        Parameters
        ----------
        xyz : array_like
            The node's coordinates in the global cartesian frame, in
            meters once units are defined. They are stored as given.
        node_id : int, optional
            Its identifier. One more than the highest node id when
            omitted.
        color : int, default 1
            Its display color index.
        def_cs : int, optional
            The coordinate system the node is placed in, which a writer
            uses to give its position. `xyz` is not converted from it.
            The first coordinate system when omitted.
        disp_cs : int, optional
            The coordinate system displacements at the node are
            measured in. The first coordinate system when omitted.

        Returns
        -------
        int
            The node's identifier.
        """
        node_id = self._next_id(self.node_id) if node_id is None else int(node_id)
        if node_id in self.node_id:
            raise ValueError(f'node {node_id} already exists')
        default_cs = int(self.cs_id[0]) if len(self.cs_id) else 1
        self.node_id = np.append(self.node_id, node_id)
        self.node_xyz = np.vstack([self.node_xyz,
                                   np.asarray(xyz, dtype=np.float64)])
        self.node_color = np.append(self.node_color, int(color))
        self.node_def_cs = np.append(
            self.node_def_cs, default_cs if def_cs is None else int(def_cs))
        self.node_disp_cs = np.append(
            self.node_disp_cs, default_cs if disp_cs is None else int(disp_cs))
        return node_id

    def add_nodes(self, xyz: ArrayLike) -> np.ndarray:
        """Add nodes at `xyz`, numbered from one more than the highest id.

        This is `add_node` for many nodes at once, with the defaults of
        `add_node`. Each array is copied once instead of once per node,
        so adding the thousands of nodes of a meshed plane is fast.

        Parameters
        ----------
        xyz : array_like
            (n, 3) coordinates in the global cartesian frame, in meters
            once units are defined.

        Returns
        -------
        ndarray
            The new nodes' identifiers, in order.
        """
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        first = self._next_id(self.node_id)
        ids = np.arange(first, first + len(xyz), dtype=np.int64)
        default_cs = int(self.cs_id[0]) if len(self.cs_id) else 1
        self.node_id = np.concatenate([self.node_id, ids])
        self.node_xyz = np.vstack([self.node_xyz, xyz])
        self.node_color = np.concatenate(
            [self.node_color, np.ones(len(xyz), dtype=self.node_color.dtype)])
        self.node_def_cs = np.concatenate(
            [self.node_def_cs, np.full(len(xyz), default_cs,
                                       dtype=self.node_def_cs.dtype)])
        self.node_disp_cs = np.concatenate(
            [self.node_disp_cs, np.full(len(xyz), default_cs,
                                        dtype=self.node_disp_cs.dtype)])
        return ids

    def add_coordinate_system(self, origin: ArrayLike = (0.0, 0.0, 0.0),
                              rotation: ArrayLike | None = None,
                              cs_id: int | None = None, name: str = '',
                              cs_type: int = 0) -> int:
        """Add a coordinate system and return its id.

        Raises ValueError if a system with `cs_id` already exists.

        Parameters
        ----------
        origin : array_like, default (0, 0, 0)
            The system's origin, in meters once units are defined.
        rotation : array_like, optional
            A 3x3 matrix whose rows are the system's axes. Identity when
            omitted.
        cs_id : int, optional
            Its identifier. One more than the highest id when omitted.
        name : str, optional
            What to call it.
        cs_type : int, default 0
            0 cartesian, 1 cylindrical, 2 spherical.

        Returns
        -------
        int
            The coordinate system's identifier.
        """
        cs_id = self._next_id(self.cs_id) if cs_id is None else int(cs_id)
        if cs_id in self.cs_id:
            raise ValueError(f'coordinate system {cs_id} already exists')
        rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        matrix = np.vstack([rotation, np.asarray(origin, dtype=np.float64)])
        self.cs_id = np.append(self.cs_id, cs_id)
        self.cs_type = np.append(self.cs_type, int(cs_type))
        self.cs_name = [*self.cs_name, str(name)]
        self.cs_matrix = np.concatenate([self.cs_matrix, matrix[np.newaxis]])
        return cs_id

    def place_coordinate_system(self, cs_id: int, *,
                                origin: ArrayLike | None = None,
                                angles: ArrayLike | None = None,
                                rotation: ArrayLike | None = None) -> None:
        """Rotate a coordinate system, move it, or both, as the 3-D
        view's rings and arrows do.

        Raises KeyError for an unknown `cs_id`, and ValueError when both
        `angles` and `rotation` are given.

        Parameters
        ----------
        cs_id : int
            Which coordinate system.
        origin : array_like, optional
            The new origin, in the geometry's stored length unit (meters
            once its units are declared). Unchanged when omitted.
        angles : array_like, optional
            The rotation as angles in degrees about the fixed global X,
            then Y, then Z axes (`rotate.frame_from_angles`), the same
            angles the view's angle box and the box panel use.
            (0, 0, 0) removes any rotation. Unchanged when omitted.
        rotation : array_like, optional
            The rotation as a 3x3 matrix whose rows are the system's
            axes, instead of `angles`.

        Returns
        -------
        None
        """
        from ..rotate import frame_from_angles

        rows = np.flatnonzero(self.cs_id == int(cs_id))
        if not len(rows):
            raise KeyError(f'no coordinate system {int(cs_id)}')
        if angles is not None and rotation is not None:
            raise ValueError('give the turn as angles or as a rotation, '
                             'not both')
        row = int(rows[0])
        if angles is not None:
            self.cs_matrix[row, :3] = frame_from_angles(angles)[:3]
        elif rotation is not None:
            self.cs_matrix[row, :3] = np.asarray(rotation, dtype=np.float64)
        if origin is not None:
            self.cs_matrix[row, 3] = np.asarray(origin, dtype=np.float64)

    def add_beams(self, node_ids: Ids, group: int | None = None,
                  color: int = 1, elem_type: int = 21) -> int:
        """Add a chain of two-node line elements through the given nodes,
        in order, and return the element group's id.

        This is how a drawn line (a traceline in some file formats) is
        stored, and it becomes a run of beams once the element group is
        given a section. Raises ValueError for fewer than two nodes, an
        unknown node, or an `elem_type` that is not a two-node line.

        Parameters
        ----------
        node_ids : sequence of int
            The nodes the line runs through, in order; two at least.
        group : int, optional
            The element group the segments are added to. A new, unnamed
            element group when omitted; a drawn line is its own element
            group, named for the line.
        color : int, default 1
            The display color index of every segment.
        elem_type : int, default 21
            The two-node line code (`ELEMENT_TYPES`): a beam, unless the
            source says rod, pipe or rigid bar.

        Returns
        -------
        int
            The element group the segments are in.
        """
        # what a traceline was, before tracelines were stored as
        # element groups (2026-09-30)
        nodes = [int(n) for n in node_ids]
        if len(nodes) < 2:
            raise ValueError('a line needs at least two nodes')
        unknown = set(nodes) - set(self.node_id.tolist())
        if unknown:
            raise ValueError(f'unknown nodes {sorted(unknown)}')
        if ELEMENT_TYPES.get(int(elem_type), (None, 0, None))[1:] != (2, 'line'):
            raise ValueError(f'{elem_type} is not a two-node line element type')
        from itertools import pairwise

        group = self.add_group() if group is None else int(group)
        for a, b in pairwise(nodes):
            self.add_element([a, b], elem_type=int(elem_type), color=color,
                             group=group)
        return group

    def mixed_groups(self) -> dict[int, list[str]]:
        """``{element group id: families}`` for every element group that
        holds more than one element family.

        It is empty for any geometry that passes `validate`.

        Returns
        -------
        dict of int to list of str
        """
        found: dict[int, list[str]] = {}
        for code, group in zip(self.elem_type, self.elem_group):
            if int(code) not in ELEMENT_TYPES:
                continue                 # validate names the stray code
            family = element_family(int(code))
            families = found.setdefault(int(group), [])
            if family not in families:
                families.append(family)
        return {b: f for b, f in found.items() if len(f) > 1}

    def split_groups_by_family(self) -> dict[int, list[int]]:
        """Split every element group that holds more than one element family.

        The group keeps its first family, and each other family is moved
        into a new element group named with the old name and the
        family, such as 'wing quads'. A file without element groups (UNV
        dataset 2412, the sdynpy layout) puts every element in one group,
        so this is applied when such a file is read.

        Returns
        -------
        dict of int to list of int
            ``{old element group: [the new element groups made from it]}``,
            empty when no group was mixed.
        """
        # Split on import rather than show one element group under two
        # families and edit it twice (Brandon, 2026-09-30).
        made: dict[int, list[int]] = {}
        for group, families in self.mixed_groups().items():
            row = int(np.flatnonzero(self.group_id == group)[0])
            base = self.group_name[row]
            for family in families[1:]:
                label = FAMILY_LABELS[family].lower()
                new = self.add_group(f'{base} {label}'.strip())
                rows = [i for i, (code, b) in enumerate(
                    zip(self.elem_type, self.elem_group))
                    if int(b) == group and element_family(int(code)) == family]
                self.elem_group[rows] = new
                made.setdefault(group, []).append(new)
        return made

    def attach_drawn_lines(self, lines: Sequence[tuple[str, int, Sequence[Ids]]]
                           ) -> list[int]:
        """Add drawn lines read from a format that stores them separately
        from elements, each as its own element group of two-node line
        elements with no properties.

        This is the reading side of `drawn_lines`.

        Parameters
        ----------
        lines : sequence of (name, color, chains)
            One entry per line: its name, its color index, and its runs,
            each a sequence of node ids in drawing order (a UNV line that
            lifts the pen has several).

        Returns
        -------
        list of int
            The element groups made, one per line.
        """
        groups = []
        for name, color, chains in lines:
            group = self.add_group(str(name))
            for chain in chains:
                self.add_beams(chain, group=group, color=int(color))
            groups.append(group)
        return groups

    def is_drawn_line(self, group: int) -> bool:
        """Whether an element group is a drawn line.

        It is when every element in it is a two-node line element and the
        element group has no properties. The app and every file writer
        use this one test. An empty element group is not a drawn line.

        Parameters
        ----------
        group : int
            The element group's identifier.

        Returns
        -------
        bool
        """
        # the one test, on screen and in every file (PLAN.md "Geometry by
        # element family")
        group = int(group)
        if group in self.group_properties:
            return False
        rows = np.flatnonzero(self.elem_group == group)
        if not len(rows):
            return False
        return all(ELEMENT_TYPES.get(int(self.elem_type[r]),
                                     (None, 0, None))[1:] == (2, 'line')
                   for r in rows)

    def drawn_lines(self) -> list[dict[str, Any]]:
        """The drawn lines, in the form used by file formats that store
        tracelines separately from elements.

        One dict `{'group', 'name', 'color', 'chains'}` per drawn-line
        element group (`is_drawn_line`), in element group order. The
        chains are the group's segments taken in element order, each a
        list of node ids, with a new chain wherever a segment does not
        start where the last one ended. An element group read from one
        polyline gives that polyline back, and one read from a UNV line
        that lifts the pen gives its runs back under the one element
        group. The color is that of the group's first element.

        Returns
        -------
        list of dict
        """
        out = []
        for row, group in enumerate(self.group_id):
            group = int(group)
            if not self.is_drawn_line(group):
                continue
            rows = np.flatnonzero(self.elem_group == group)
            chains: list[list[int]] = []
            for r in rows:
                a, b = (int(n) for n in self.elem_conn[r][:2])
                if chains and chains[-1][-1] == a:
                    chains[-1].append(b)
                else:
                    chains.append([a, b])
            out.append({'group': group, 'name': self.group_name[row],
                        'color': int(self.elem_color[rows[0]]),
                        'chains': chains})
        return out

    def group_of(self, elem_id: int) -> str:
        """The name of the element group an element belongs to, or '' when
        that group is not declared.

        Parameters
        ----------
        elem_id : int
            Which element.

        Returns
        -------
        str
            The name of the element group it belongs to.
        """
        row = int(np.flatnonzero(self.elem_id == int(elem_id))[0])
        group = int(self.elem_group[row])
        where = np.flatnonzero(self.group_id == group)
        return self.group_name[int(where[0])] if len(where) else ''

    def elements_in(self, group: int | str) -> list[int]:
        """The ids of the elements in an element group, given by id or name.

        A name that no element group has gives an empty list; with two
        groups of one name, the first is used.

        Parameters
        ----------
        group : int or str
            An element group, by identifier or by name.

        Returns
        -------
        list of int
            The identifiers of the elements it holds.
        """
        if isinstance(group, str):
            where = [i for i, name in enumerate(self.group_name)
                     if name == group]
            if not where:
                return []
            group = int(self.group_id[where[0]])
        return [int(self.elem_id[i])
                for i in np.flatnonzero(self.elem_group == int(group))]

    def nodes_in(self, groups: Ids) -> list[int]:
        """The ids of the nodes used by the elements of these element
        groups, in ascending order. These are the nodes a merge limited to
        those groups may change.

        Parameters
        ----------
        groups : sequence of int
            Element groups, by identifier.

        Returns
        -------
        list of int
            The node identifiers.

        Raises
        ------
        ValueError
            When a group is not in the geometry.
        """
        groups = [int(g) for g in groups]
        missing = [g for g in groups if g not in self.group_id.tolist()]
        if missing:
            raise ValueError(f'no element group {missing[0]}')
        return sorted({int(n) for k in np.flatnonzero(
            np.isin(self.elem_group, groups)) for n in self.elem_conn[k]})

    def add_group(self, name: str = '', group_id: int | None = None) -> int:
        """Declare an element group and return its id.

        An empty element group is allowed: exodus files contain empty
        ones, and an element group has to exist before an element can be
        put in it. Raises ValueError if `group_id` is already used.

        Parameters
        ----------
        name : str, optional
            What to call the element group.
        group_id : int, optional
            Its identifier. One more than the highest id when omitted.

        Returns
        -------
        int
            The element group's identifier.
        """
        group_id = (self._next_id(self.group_id) if group_id is None
                    else int(group_id))
        if group_id in self.group_id:
            raise ValueError(f'element group {group_id} already exists')
        self.group_id = np.append(self.group_id, group_id)
        self.group_name.append(str(name))
        return group_id

    def add_element(self, node_ids: Ids, elem_type: int | None = None,
                    color: int = 1, group: int | None = None) -> int:
        """Add an element and return its row index.

        The element id is one more than the highest element id. Without
        `elem_type`, the type is chosen from the node count: 2 nodes a
        beam (21), 3 a triangle (41), 4 a quadrilateral (44); any other
        count needs `elem_type`. Raises ValueError for an unknown node or
        type, or for a group that holds a different element family.

        Parameters
        ----------
        node_ids : int or sequence of int
            The nodes the element connects, in order.
        elem_type : int, optional
            The element type code. Inferred from the node
            count when omitted.
        color : int, default 1
            Its display color index.
        group : int, optional
            The element group it belongs to, which is declared if it does
            not exist yet. When omitted, the first element group that
            holds this element's family (or is empty), otherwise a new
            one. An element group holds one family, so a beam added to a
            mesh of plates goes in an element group of its own.

        Returns
        -------
        int
            The element's row index in the element arrays (not its id).
        """
        # one element group, one family (2026-09-30)
        nodes = np.asarray([int(n) for n in node_ids], dtype=np.int64)
        unknown = set(nodes.tolist()) - set(self.node_id.tolist())
        if unknown:
            raise ValueError(f'unknown nodes {sorted(unknown)}')
        if elem_type is None:
            elem_type = {2: 21, 3: 41, 4: 44}.get(len(nodes))
            if elem_type is None:
                raise ValueError(
                    f'no default element type for {len(nodes)} nodes; '
                    'give elem_type')
        if int(elem_type) not in ELEMENT_TYPES:
            raise ValueError(f'unknown element type {elem_type}')
        family = element_family(int(elem_type))
        if group is None:
            held_by = {int(b): element_family(int(code))
                       for code, b in zip(self.elem_type, self.elem_group)}
            group = next((b for b in self.group_id.tolist()
                          if held_by.get(int(b), family) == family), None)
            if group is None:
                group = self.add_group()
        group = int(group)
        # one element group, one family: the element group's properties are one kind
        rows = np.flatnonzero(self.elem_group == group)
        if len(rows):
            held = element_family(int(self.elem_type[rows[0]]))
            if held != family:
                raise ValueError(
                    f'element group {group} holds {held}; a {family[:-1]} goes in '
                    'an element group of its own')
        self.elem_id = np.append(self.elem_id, self._next_id(self.elem_id))
        self.elem_type = np.append(self.elem_type, int(elem_type))
        self.elem_color = np.append(self.elem_color, int(color))
        if group not in self.group_id.tolist():
            self.group_id = np.append(self.group_id, group)
            self.group_name.append('')
        self.elem_group = np.append(self.elem_group, group)
        self.elem_conn.append(nodes)
        return len(self.elem_conn) - 1

    def add_elements(self, connectivity: Sequence[Ids],
                     elem_types: Ids, groups: Ids) -> None:
        """Add many elements at once, numbered from one more than the
        highest element id, with color 1.

        The nodes and types are checked once for all of them, and an
        element group may hold only one element family, as in
        `add_element`. If any check fails, ValueError is raised and
        nothing is added.

        Parameters
        ----------
        connectivity : sequence of sequence of int
            Each element's nodes, in order.
        elem_types : sequence of int
            Each element's type code.
        groups : sequence of int
            The element group each belongs to; an element group that does
            not exist yet is declared, with no name.

        Returns
        -------
        None
        """
        conn = [np.asarray([int(n) for n in element], dtype=np.int64)
                for element in connectivity]
        if not conn:
            return
        used = set(np.concatenate(conn).tolist())
        unknown = used - set(self.node_id.tolist())
        if unknown:
            raise ValueError(f'unknown nodes {sorted(unknown)[:10]}')
        types = [int(t) for t in elem_types]
        wrong = sorted(set(types) - set(ELEMENT_TYPES))
        if wrong:
            raise ValueError(f'unknown element type {wrong[0]}')
        groups = [int(b) for b in groups]
        # one element group, one family — the rule `add_element` keeps one element
        # at a time, kept here for many at once. Its absence let a plate
        # joined by name into an element group of bricks make one element group of both
        # (Brandon, 2026-10-02); checked before anything is appended, so
        # a refused call leaves the geometry as it was
        coming: dict[int, set[str]] = {}
        for group, code in zip(groups, types):
            coming.setdefault(group, set()).add(element_family(code))
        for group, families in coming.items():
            held = {element_family(int(code)) for code
                    in self.elem_type[self.elem_group == group]}
            if len(held | families) > 1:
                raise ValueError(
                    f'element group {group} would hold '
                    f'{" and ".join(sorted(held | families))}: an element group holds '
                    'one element family')
        for group in dict.fromkeys(groups):
            if group not in self.group_id.tolist():
                self.group_id = np.append(self.group_id, group)
                self.group_name.append('')
        first = self._next_id(self.elem_id)
        self.elem_id = np.concatenate(
            [self.elem_id, np.arange(first, first + len(conn))]).astype(np.int64)
        self.elem_type = np.concatenate([self.elem_type, types]).astype(np.int64)
        self.elem_color = np.concatenate(
            [self.elem_color, np.ones(len(conn))]).astype(np.int64)
        self.elem_group = np.concatenate([self.elem_group, groups]).astype(np.int64)
        self.elem_conn.extend(conn)

    # ---- deletion -----------------------------------------------------------

    def renumber_node(self, row: int, node_id: int) -> None:
        """Give a node a new id, and update the elements that use it.

        Elements refer to nodes by id, so the connectivity is updated
        too; otherwise every element at the node would lose it. Raises
        ValueError if another node already has the id.

        Parameters
        ----------
        row : int
            Which node, by row.
        node_id : int
            Its new identifier.

        Returns
        -------
        None
        """
        node_id = int(node_id)
        clash = np.flatnonzero(self.node_id == node_id)
        if len(clash) and clash[0] != row:
            raise ValueError(f'node {node_id} already exists')
        old = int(self.node_id[row])
        self.node_id[row] = node_id
        for conn in self.elem_conn:
            conn[conn == old] = node_id

    def renumber_group(self, row: int, group_id: int) -> None:
        """Give an element group a new id, and update its elements.

        An element refers to its element group by id, so the elements are
        updated too; otherwise they would refer to an element group that
        no longer exists. Raises ValueError if another element group
        already has the id.

        Parameters
        ----------
        row : int
            Which element group, by row.
        group_id : int
            Its new identifier.

        Returns
        -------
        None
        """
        group_id = int(group_id)
        clash = np.flatnonzero(self.group_id == group_id)
        if len(clash) and clash[0] != row:
            raise ValueError(f'element group {group_id} already exists')
        old = int(self.group_id[row])
        self.group_id[row] = group_id
        self.elem_group[self.elem_group == old] = group_id

    def renumber_coordinate_system(self, row: int, cs_id: int) -> None:
        """Give a coordinate system a new id, and update the nodes that use it.

        Both the placement and the displacement systems of the nodes are
        updated. Raises ValueError if another system already has the id.

        Parameters
        ----------
        row : int
            Which system, by row.
        cs_id : int
            Its new identifier.

        Returns
        -------
        None
        """
        cs_id = int(cs_id)
        clash = np.flatnonzero(self.cs_id == cs_id)
        if len(clash) and clash[0] != row:
            raise ValueError(f'coordinate system {cs_id} already exists')
        old = int(self.cs_id[row])
        self.cs_id[row] = cs_id
        for name in ('node_def_cs', 'node_disp_cs'):
            references = getattr(self, name)
            references[references == old] = cs_id

    def delete_nodes(self, node_ids: Ids) -> dict[str, int]:
        """Remove nodes, and the elements that use them.

        An element that uses a deleted node is deleted too. Ids that are
        not present are ignored.

        Parameters
        ----------
        node_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        dict of str to int
            'nodes' and 'elements', the number of each removed.
        """
        wanted = {int(node) for node in node_ids}
        keep = ~np.isin(self.node_id, list(wanted))
        removed_nodes = int((~keep).sum())
        if not removed_nodes:
            return {'nodes': 0, 'elements': 0}

        orphan_elements = [int(self.elem_id[i])
                           for i, conn in enumerate(self.elem_conn)
                           if wanted & {int(n) for n in conn}]
        self.delete_elements(orphan_elements)

        for name in ('node_id', 'node_def_cs', 'node_disp_cs', 'node_color'):
            setattr(self, name, getattr(self, name)[keep])
        self.node_xyz = self.node_xyz[keep]
        self.validate()
        return {'nodes': removed_nodes, 'elements': len(orphan_elements)}

    def coincident_nodes(self, tolerance: float,
                         nodes: Ids | None = None) -> dict[int, int]:
        """``{node: the node it coincides with}`` for every node within
        `tolerance` of another.

        Each node is mapped to the lowest id in its group, which is the
        node a merge keeps. Groups are chained: if a is within tolerance
        of b and b of c, all three are one point.

        Parameters
        ----------
        tolerance : float
            How close two nodes must be to be one point, in the stored
            length unit: meters once units are defined.
        nodes : sequence of int, optional
            Only these nodes are considered, by id; a node outside them
            is not merged and nothing is merged into it, however close.
            Every node when omitted.

        Returns
        -------
        dict of int to int
            Only the nodes that would be removed, each mapped to the node
            it becomes.
        """
        from scipy.spatial import cKDTree

        rows = (np.arange(len(self.node_id)) if nodes is None
                else np.flatnonzero(np.isin(self.node_id,
                                            [int(n) for n in nodes])))
        if len(rows) < 2:
            return {}
        pairs = cKDTree(self.node_xyz[rows]).query_pairs(float(tolerance))
        parent = list(range(len(rows)))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in pairs:
            a, b = root(i), root(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
        groups: dict[int, list[int]] = {}
        for i in range(len(rows)):
            groups.setdefault(root(i), []).append(i)
        out = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            ids = sorted(int(self.node_id[rows[i]]) for i in members)
            for other in ids[1:]:
                out[other] = ids[0]
        return out

    def duplicate_elements(self, groups: Ids | None = None) -> dict[int, int]:
        """``{element id: the element it duplicates}`` for every solid
        element of the same type and on the same nodes as an earlier
        element, such as two bricks filling one cell.

        The earlier element (in element order) is the one that stays.
        Only solids are checked: two beams on one line can be two
        members, and a plate on another plate's nodes is how a doubler or
        a layer is modeled, but two solids are never meant to fill one
        cell. Two element groups that overlap and share their nodes in
        the overlap (the bars of an X cross-section) put two elements in
        every cell there, which doubles the stiffness and mass of the
        region.

        Parameters
        ----------
        groups : sequence of int, optional
            Only the elements of these element groups, by id; every
            element when omitted.

        Returns
        -------
        dict of int to int
            Each element that fills a cell another element already
            fills, mapped to that element.
        """
        # The X cross-section's overlap drew every face of each pair as
        # interior, so the middle of the X disappeared from the view
        # (Brandon, 2026-10-02).
        wanted = None if groups is None else {int(g) for g in groups}
        first: dict[tuple, int] = {}
        out: dict[int, int] = {}
        for k, conn in enumerate(self.elem_conn):
            if element_family(int(self.elem_type[k])) not in SOLID_FAMILIES:
                continue
            if wanted is not None and int(self.elem_group[k]) not in wanted:
                continue
            key = (int(self.elem_type[k]), tuple(sorted(int(n) for n in conn)))
            eid = int(self.elem_id[k])
            if key in first:
                out[eid] = first[key]
            else:
                first[key] = eid
        return out

    def merge_duplicate_elements(self, groups: Ids | None = None) -> int:
        """Remove duplicate solid elements (`duplicate_elements`).

        The later element of each pair is removed, and the earlier one
        stays in its element group.

        Parameters
        ----------
        groups : sequence of int, optional
            Only the elements of these element groups, by id; every
            element when omitted.

        Returns
        -------
        int
            How many elements were removed.
        """
        going = self.duplicate_elements(groups)
        if going:
            self.delete_elements(list(going))
        return len(going)

    def _merge_candidates(self, groups: Ids | None,
                          exclude: Ids | None) -> list[int] | None:
        """The node ids a merge may touch: those of `groups`' elements
        (every node when None), less `exclude`; None for every node."""
        nodes = None if groups is None else self.nodes_in(groups)
        if not exclude:
            return nodes
        left_out = {int(n) for n in exclude}
        base = self.node_id.tolist() if nodes is None else nodes
        return [int(n) for n in base if int(n) not in left_out]

    def node_spacing(self, groups: Ids | None = None) -> float | None:
        """The mesh's finest spacing: the shortest nonzero distance between
        two nodes of one element, such as a brick's shortest edge or half
        an edge of a quad8.

        Merge Coincident Nodes sets its default tolerance to a quarter of
        this value. Nodes of parts meshed separately are usually much
        closer than that where the parts meet, and no two nodes of one
        element are.

        Parameters
        ----------
        groups : sequence of int, optional
            Only these element groups' elements, by id; every element
            when omitted.

        Returns
        -------
        float or None
            The spacing, in the stored length unit; None when no element
            has two nodes at different positions.
        """
        # the quarter is Brandon's, 2026-10-11
        rows = (range(len(self.elem_conn)) if groups is None
                else np.flatnonzero(np.isin(self.elem_group,
                                            [int(g) for g in groups])))
        return self._spacing(rows)

    def _spacing(self, rows) -> float | None:
        """`node_spacing` over these element rows."""
        by_count: dict[int, list[np.ndarray]] = {}
        for k in rows:
            conn = self.elem_conn[k]
            if len(conn) > 1:
                by_count.setdefault(len(conn), []).append(conn)
        finest = np.inf
        for count, conns in by_count.items():
            xyz = self.node_xyz[self.node_index(np.concatenate(conns))]
            xyz = xyz.reshape(len(conns), count, 3)
            first, second = np.triu_indices(count, 1)
            gaps = np.linalg.norm(xyz[:, first] - xyz[:, second], axis=-1)
            gaps = gaps[gaps > 0.0]
            if gaps.size:
                finest = min(finest, float(gaps.min()))
        return None if not np.isfinite(finest) else finest

    def coincident_spots(self, tolerance: float, groups: Ids | None = None,
                         exclude: Ids | None = None) -> list[list[int]]:
        """The places where a merge would make one node.

        Each place is a list of the node ids that coincide there, in
        ascending order, so the first is the node that stays. Merge
        Coincident Nodes marks these places in the view before anything
        is merged, so some of them can be left out.

        Parameters
        ----------
        tolerance : float
            How close two nodes must be to be one, in the stored length
            unit: meters once units are defined.
        groups : sequence of int, optional
            Only the nodes used by these element groups' elements, by id;
            every node when omitted.
        exclude : sequence of int, optional
            Nodes left out, by id: they are not merged and nothing is
            merged into them.

        Returns
        -------
        list of list of int
            One list per place, sorted by the node each keeps.
        """
        # marked a place at a time, for a person to leave some out
        # (Brandon, 2026-10-11)
        mapping = self.coincident_nodes(
            tolerance, self._merge_candidates(groups, exclude))
        spots: dict[int, list[int]] = {}
        for going, kept in mapping.items():
            spots.setdefault(int(kept), [int(kept)]).append(int(going))
        return [sorted(nodes) for _kept, nodes in sorted(spots.items())]

    def merge_folds(self, tolerance: float, groups: Ids | None = None,
                    exclude: Ids | None = None) -> str | None:
        """Why this merge would be refused, or None.

        Returns the message `merge_coincident_nodes` raises when the
        tolerance would fold an element onto itself, so a panel can show
        it before the merge is applied.

        Parameters
        ----------
        tolerance : float
            As `merge_coincident_nodes`.
        groups : sequence of int, optional
            As `merge_coincident_nodes`.
        exclude : sequence of int, optional
            As `merge_coincident_nodes`.

        Returns
        -------
        str or None
        """
        return self._folding(self.coincident_nodes(
            tolerance, self._merge_candidates(groups, exclude)), tolerance)

    def merge_collapses(self, tolerance: float, groups: Ids | None = None,
                        exclude: Ids | None = None) -> list[int]:
        """The ids of the line elements this merge would remove.

        These are the line elements (a rigid link, a spring, a beam)
        whose nodes would all become one node, so they no longer join
        anything. Merging a joint built with rigid links removes the
        links.

        Parameters
        ----------
        tolerance : float
            As `merge_coincident_nodes`.
        groups : sequence of int, optional
            As `merge_coincident_nodes`.
        exclude : sequence of int, optional
            As `merge_coincident_nodes`.

        Returns
        -------
        list of int
        """
        # The merge used to be refused for folding such links (Brandon,
        # 2026-10-11).
        return self._collapsed(self.coincident_nodes(
            tolerance, self._merge_candidates(groups, exclude)))

    def _collapsed(self, mapping: dict[int, int]) -> list[int]:
        if not mapping:
            return []
        out = []
        for i, conn in enumerate(self.elem_conn):
            if (len(conn) > 1 and ELEMENT_TYPES[int(self.elem_type[i])][2] == 'line'
                    and len({mapping.get(int(n), int(n)) for n in conn}) == 1):
                out.append(int(self.elem_id[i]))
        return out

    def _folding(self, mapping: dict[int, int], tolerance: float) -> str | None:
        gone = set(self._collapsed(mapping))
        for i, conn in enumerate(self.elem_conn):
            if int(self.elem_id[i]) in gone:
                continue
            renamed = [mapping.get(int(n), int(n)) for n in conn]
            if len(set(renamed)) < len(renamed):
                return (f'a tolerance of {tolerance:g} folds element '
                        f'{int(self.elem_id[i])} onto itself — two of its own '
                        'nodes are within it; use a smaller tolerance')
        return None

    def merge_coincident_nodes(self, tolerance: float,
                               groups: Ids | None = None,
                               exclude: Ids | None = None) -> dict[str, int]:
        """Merge nodes that are at one point into one node.

        Nodes within `tolerance` of each other (`coincident_nodes`)
        become the one with the lowest id: every element that uses one of
        them is changed to use that node, and the others are removed.
        Plates are only connected where they share nodes, so this is what
        connects a model built from separate planes at its edges.

        The merge is refused with ValueError, naming the element, when
        the tolerance would fold an element onto itself (two of its own
        nodes within the tolerance), since that means the tolerance is
        larger than the mesh. A line element whose nodes would all become
        one, such as a rigid link across the gap the merge closes, is
        removed instead (`merge_collapses`). Solid elements the merge
        leaves on exactly the same nodes fill one cell twice, and the
        duplicates are removed (`merge_duplicate_elements`).

        With `groups`, only those element groups are merged: the nodes
        their elements use are the candidates, so two groups picked
        together are joined to each other and to nothing else, and only
        their duplicate elements are removed. A merge over the whole
        geometry joins every part to every part it touches, which is
        usually more than intended; the app offers the merge on element
        groups only.

        Parameters
        ----------
        tolerance : float
            How close two nodes must be to be one, in the stored length
            unit: meters once units are defined.
        groups : sequence of int, optional
            The element groups whose nodes are merged, by id. Every node
            of the geometry when omitted.
        exclude : sequence of int, optional
            Nodes left out of the merge, by id: they are not merged and
            nothing is merged into them. In the app, these are the
            places that were unpicked (`coincident_spots`).

        Returns
        -------
        dict of str to int
            'merged', the number of nodes removed; 'into', the number of
            nodes they became; 'duplicates', the number of elements
            removed for filling a cell another already filled; and
            'links', the number of line elements removed because their
            nodes became one.
        """
        # The app offers the merge on element groups only (Brandon,
        # 2026-10-11).
        mapping = self.coincident_nodes(
            tolerance, self._merge_candidates(groups, exclude))
        if not mapping:
            duplicates = self.merge_duplicate_elements(groups)
            return {'merged': 0, 'into': 0, 'duplicates': duplicates,
                    'links': 0}
        refusal = self._folding(mapping, tolerance)
        if refusal is not None:
            raise ValueError(refusal)
        links = self._collapsed(mapping)
        self.delete_elements(links)
        self.elem_conn = [np.asarray([mapping.get(int(n), int(n))
                                      for n in conn], dtype=np.int64)
                          for conn in self.elem_conn]
        keep = ~np.isin(self.node_id, list(mapping))
        for name in ('node_id', 'node_def_cs', 'node_disp_cs', 'node_color'):
            setattr(self, name, getattr(self, name)[keep])
        self.node_xyz = self.node_xyz[keep]
        duplicates = self.merge_duplicate_elements(groups)
        self.validate()
        return {'merged': len(mapping), 'into': len(set(mapping.values())),
                'duplicates': duplicates, 'links': len(links)}

    # ---- Align Meshes (Brandon, 2026-10-11) -----------------------------

    def _parts(self, groups: Ids | None) -> list[np.ndarray]:
        """The element rows of `groups` (every element when None) as the
        pieces their shared nodes join them into: two parts meshed apart
        are two pieces whichever element groups they are in, and one
        element group holding two loose pieces is two."""
        rows = (np.arange(len(self.elem_conn)) if groups is None
                else np.flatnonzero(np.isin(self.elem_group,
                                            [int(g) for g in groups])))
        parent: dict[int, int] = {}

        def root(node: int) -> int:
            while parent.setdefault(node, node) != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for k in rows:
            nodes = [int(n) for n in self.elem_conn[k]]
            for node in nodes[1:]:
                parent[root(node)] = root(nodes[0])
        pieces: dict[int, list[int]] = {}
        for k in rows:
            pieces.setdefault(root(int(self.elem_conn[k][0])), []).append(int(k))
        return [np.asarray(piece) for piece in pieces.values()]

    def _skin(self, rows) -> set[int]:
        """The nodes of these element rows that lie on their outside:
        every node of a plate, a line or a point, and of a solid the
        nodes of the faces no second solid among them shares."""
        faces: dict[tuple[int, ...], tuple[int, ...]] = {}
        skin: set[int] = set()
        for k in rows:
            conn = [int(n) for n in self.elem_conn[k]]
            if element_family(int(self.elem_type[k])) not in SOLID_FAMILIES:
                skin.update(conn)
                continue
            for face in solid_faces(len(conn)):
                corners = tuple(conn[i] for i in face)
                key = tuple(sorted(corners))
                if key in faces:
                    del faces[key]
                else:
                    faces[key] = corners
        for corners in faces.values():
            skin.update(corners)
        return skin

    def _typical_spacing(self, rows) -> float | None:
        """A piece's mesh size, as Align Meshes reads it: the median over
        its elements of a brick's middle edge — its size across, for a
        layer one brick thin — and of any other element's shortest. Not
        the finest edge anywhere (`node_spacing`): the bricks bent onto a
        hole's circle have edges a fraction of the mesh's, and the
        Frame-Wing's wing, meshed at an eighth of an inch, read finer
        than its frame at a twelfth; and not a brick's shortest edge: the
        washers under the wing, 0.028 in thick on an eighth-inch grid,
        read finer than the frame and the frame moved onto them
        (2026-10-11)."""
        shortest = []
        by_count: dict[int, list[np.ndarray]] = {}
        for k in rows:
            conn = self.elem_conn[k]
            if len(conn) > 1:
                by_count.setdefault(len(conn), []).append(conn)
        for count, conns in by_count.items():
            xyz = self.node_xyz[self.node_index(np.concatenate(conns))]
            xyz = xyz.reshape(len(conns), count, 3)
            if count == 8:
                # a brick's three edges from its first corner, the middle
                edges = np.linalg.norm(xyz[:, [1, 3, 4]] - xyz[:, [0]], axis=-1)
                shortest.extend(np.sort(edges, axis=1)[:, 1].tolist())
                continue
            first, second = np.triu_indices(count, 1)
            gaps = np.linalg.norm(xyz[:, first] - xyz[:, second], axis=-1)
            gaps[gaps <= 0.0] = np.inf
            shortest.extend(gaps.min(axis=1).tolist())
        shortest = [g for g in shortest if np.isfinite(g) and g > 0.0]
        return float(np.median(shortest)) if shortest else None

    def align_reach(self, groups: Ids | None = None) -> float | None:
        """The default reach of Align Meshes: half the mesh size of the
        coarsest piece.

        A piece is a set of elements connected by shared nodes. With this
        reach no node moves more than half an element of its own mesh.
        Parts with a gap between them need a larger reach, by the size of
        the gap.

        Parameters
        ----------
        groups : sequence of int, optional
            The element groups, by id; every element when omitted.

        Returns
        -------
        float or None
            In the stored length unit; None when no element has a
            nonzero size.
        """
        spacings = [self._typical_spacing(rows) for rows in self._parts(groups)]
        spacings = [s for s in spacings if s]
        return 0.5 * max(spacings) if spacings else None

    def align_moves(self, reach: float, groups: Ids | None = None,
                    exclude: Ids | None = None) -> list[list[int]]:
        """The moves Align Meshes would make, each ``[node, onto]``: a node
        of a coarser piece and the node of a finer piece it would move
        onto, so the two are at one point and can be merged with Merge
        Coincident Nodes.

        A piece is a set of elements connected by shared nodes. Two
        pieces meshed at different sizes cannot be matched node for node,
        since their node counts differ, but each node of the coarser
        piece can be moved onto the nearest node of the finer one. No
        node moves more than `reach`. The rules:

        - Only nodes on the outside of the pieces move or are moved onto.
        - A node moves once, and onto a node no other node moves onto.
        - A node already at another node's position is a place to merge,
          not a move.
        - The nearest pairs are taken first. As a result, a washer's
          lower face moves onto the frame under it, and its upper face,
          whose nearest frame nodes are already taken, stays.
        - Of two pieces with the same mesh size, the later piece moves.
        - A move that would turn an element inside out is left out.

        Parameters
        ----------
        reach : float
            How far a node may move, in the stored length unit.
        groups : sequence of int, optional
            The element groups whose pieces are aligned, by id; every
            element when omitted.
        exclude : sequence of int, optional
            Nodes left out, by id: they are not moved and nothing is
            moved onto them. In the app, these are the moves that were
            unpicked.

        Returns
        -------
        list of [int, int]
            Sorted by the node moved.
        """
        # Brandon, 2026-10-11, for the Frame-Wing, whose wing is meshed at
        # an eighth of an inch on a frame meshed at a twelfth.
        from scipy.spatial import cKDTree

        parts = self._parts(groups)
        if len(parts) < 2:
            return []
        left_out = {int(n) for n in exclude or ()}
        skins = [sorted(self._skin(rows) - left_out) for rows in parts]
        spacing = [self._typical_spacing(rows) or 0.0 for rows in parts]
        low, high = self.extent if self.num_nodes else (np.zeros(3), np.zeros(3))
        # already one point: a place to merge, not a move
        touching = 1e-9 * float(np.linalg.norm(high - low) or 1.0)
        found = []
        for a, movers in enumerate(skins):
            for b, targets in enumerate(skins):
                finer = spacing[b] < spacing[a] or (spacing[b] == spacing[a]
                                                    and b < a)
                if a == b or not finer or not movers or not targets:
                    continue
                tree = cKDTree(self.node_xyz[self.node_index(targets)])
                distance, nearest = tree.query(
                    self.node_xyz[self.node_index(movers)],
                    distance_upper_bound=float(reach))
                for mover, d, j in zip(movers, distance, nearest):
                    if touching < d <= reach:
                        found.append((float(d), mover, targets[int(j)]))
        moves, moving, onto = [], set(), set()
        for _d, mover, target in sorted(found):
            if (mover in moving or mover in onto or target in onto
                    or target in moving):
                continue
            moves.append([mover, target])
            moving.add(mover)
            onto.add(target)
        return sorted(self._unfolded(moves))

    def _unfolded(self, moves: list[list[int]]) -> list[list[int]]:
        """The moves, nearest first, each tried against those before it
        and left out when it turns an element inside out — the rule Mesh
        Solids rounds a hole by. A washer's rim, rounded onto its circle,
        has bricks one move would fold (the Frame-Wing, 2026-10-11), and
        a refusal of the whole Align over one of forty moves left a
        person nothing to do but unpick it by hand."""
        if not moves:
            return moves
        movers = [a for a, _b in moves]
        touching: dict[int, list[int]] = {}
        wanted = set(movers)
        for k, conn in enumerate(self.elem_conn):
            for n in conn:
                if int(n) in wanted:
                    touching.setdefault(int(n), []).append(k)
        rows = dict(zip(movers, self.node_index(movers)))
        targets = dict(zip(movers, self.node_index([b for _a, b in moves])))
        grid = self.node_xyz.copy()
        after = self.node_xyz.copy()
        order = sorted(moves, key=lambda m: float(np.linalg.norm(
            grid[targets[m[0]]] - grid[rows[m[0]]])))
        kept = []
        for mover, target in order:
            after[rows[mover]] = grid[targets[mover]]
            elements = touching.get(mover, [])
            if any(_turned(int(self.elem_type[k]),
                           grid[self.node_index(self.elem_conn[k])],
                           after[self.node_index(self.elem_conn[k])])
                   for k in elements):
                after[rows[mover]] = grid[rows[mover]]
            else:
                kept.append([mover, target])
        return kept

    def align_folds(self, moves) -> str | None:
        """Why these moves would be refused, or None.

        Returns the message `align_meshes` raises when a move would turn
        an element inside out (a solid's corner volume or a plate's
        normal changing sign), so a panel can show it before the moves
        are applied.

        Parameters
        ----------
        moves : sequence of [int, int]
            As `align_moves` gives them.

        Returns
        -------
        str or None
        """
        if not moves:
            return None
        moved = {int(a): int(b) for a, b in moves}
        after = self.node_xyz.copy()
        rows = self.node_index(list(moved))
        after[rows] = self.node_xyz[self.node_index(list(moved.values()))]
        # the elements a move touches, found a node count at a time: a
        # loop over every element took a third of a second on the
        # Frame-Wing, on every click in the pane
        touched = []
        by_count: dict[int, list[int]] = {}
        for k, conn in enumerate(self.elem_conn):
            by_count.setdefault(len(conn), []).append(k)
        names = np.fromiter(moved, dtype=np.int64)
        for count, ks in by_count.items():
            conns = np.asarray([self.elem_conn[k] for k in ks]).reshape(len(ks), count)
            hit = np.isin(conns, names).any(axis=1)
            touched.extend(np.asarray(ks)[hit].tolist())
        touched.sort()
        # their rows in one lookup: `node_index` sorts every id each call
        flat = self.node_index(np.concatenate(
            [self.elem_conn[k] for k in touched])) if touched else []
        starts = np.cumsum([0] + [len(self.elem_conn[k]) for k in touched])
        for i, k in enumerate(touched):
            conn = self.elem_conn[k]
            index = flat[starts[i]:starts[i + 1]]
            if _turned(int(self.elem_type[k]), self.node_xyz[index],
                       after[index]):
                return (f'moving node {next(int(n) for n in conn if int(n) in moved)}'
                        f' turns element {int(self.elem_id[k])} inside out; '
                        'leave that move out or use a smaller reach')
        return None

    def align_meshes(self, reach: float, groups: Ids | None = None,
                     exclude: Ids | None = None) -> dict[str, float]:
        """Move the nodes of the coarser piece onto the nodes of the finer
        piece where they meet (`align_moves`), so Merge Coincident Nodes
        can join them.

        This is for parts meshed separately at sizes that do not match:
        it makes their nodes coincide at their mating faces or where they
        overlap. Nothing is merged and no node is removed. A move that
        would turn an element inside out is left out by `align_moves`;
        the call raises ValueError, naming the element, only when the
        nodes left out with `exclude` make one of the other moves do so
        (`align_folds`).

        Parameters
        ----------
        reach : float
            How far a node may move, in the stored length unit: meters
            once units are defined.
        groups : sequence of int, optional
            The element groups whose pieces are aligned, by id; every
            element of the geometry when omitted.
        exclude : sequence of int, optional
            Nodes left out, by id: neither moved nor moved onto.

        Returns
        -------
        dict
            'moved', how many nodes moved; 'farthest', the longest move,
            in the stored length unit.
        """
        moves = self.align_moves(reach, groups, exclude)
        if not moves:
            return {'moved': 0, 'farthest': 0.0}
        refusal = self.align_folds(moves)
        if refusal is not None:
            raise ValueError(refusal)
        rows = self.node_index([a for a, _b in moves])
        targets = self.node_xyz[self.node_index([b for _a, b in moves])]
        farthest = float(np.linalg.norm(targets - self.node_xyz[rows],
                                        axis=1).max())
        self.node_xyz[rows] = targets
        return {'moved': len(moves), 'farthest': farthest}

    def delete_coordinate_systems(self, cs_ids: Ids) -> dict[str, int]:
        """Remove coordinate systems, and move the nodes that used them to
        the first remaining system.

        Ids that are not present are ignored. Removing every coordinate
        system is refused with ValueError, since every node needs one.

        Parameters
        ----------
        cs_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        dict of str to int
            'coordinate_systems', the number removed, and
            'nodes_reassigned', the number of node placement and
            displacement references moved to the remaining system.
        """
        wanted = {int(cs) for cs in cs_ids}
        keep = ~np.isin(self.cs_id, list(wanted))
        if keep.sum() == 0:
            raise ValueError('a geometry needs at least one coordinate system')
        removed = int((~keep).sum())
        if not removed:
            return {'coordinate_systems': 0, 'nodes_reassigned': 0}

        fallback = int(self.cs_id[keep][0])
        reassigned = 0
        for name in ('node_def_cs', 'node_disp_cs'):
            array = getattr(self, name)
            stale = np.isin(array, list(wanted))
            reassigned += int(stale.sum())
            array[stale] = fallback
        self.cs_id = self.cs_id[keep]
        self.cs_type = self.cs_type[keep]
        self.cs_matrix = self.cs_matrix[keep]
        self.cs_name = [name for name, k in zip(self.cs_name, keep) if k]
        return {'coordinate_systems': removed, 'nodes_reassigned': reassigned}

    def _rows_for(self, ids, id_array):
        """Which rows the given ids sit in, highest first so a list can be
        deleted from without the later positions shifting. Ids that are
        not there are skipped, the way deleting a missing node is."""
        wanted = {int(i) for i in ids}
        return sorted((row for row, value in enumerate(id_array)
                       if int(value) in wanted), reverse=True)

    def delete_groups(self, group_ids: Ids, keep: Ids = ()) -> dict[str, int]:
        """Remove element groups with their elements, their properties, and
        the nodes no element outside them uses.

        A node that an element of another element group also uses stays,
        so a neighboring part is not cut along the nodes the two share.
        Ids that are not present are ignored. To combine element groups
        instead, use `merge_groups`.

        Parameters
        ----------
        group_ids : int or sequence of int
            The identifiers, one or many.
        keep : sequence of int, optional
            Nodes that stay whatever uses them (`delete_elements`).

        Returns
        -------
        dict of str to int
            'groups', 'elements' and 'nodes', the number of each removed.
        """
        # Deleting a part deletes the part (Brandon, 2026-09-27). It used
        # to move a deleted element group's elements into the first
        # element group left, which made a delete a merge.
        wanted = {int(group) for group in group_ids}
        staying = ~np.isin(self.group_id, list(wanted))
        removed = int((~staying).sum())
        if not removed:
            return {'groups': 0, 'elements': 0, 'nodes': 0}
        doomed = [int(e) for e, group in zip(self.elem_id, self.elem_group)
                  if int(group) in wanted]
        gone = self.delete_elements(doomed, nodes=True, keep=keep)
        self.group_name = [name for name, k in zip(self.group_name, staying) if k]
        for group in wanted:                 # a deleted element group's properties go with it
            self.group_properties.pop(group, None)
        self.group_id = self.group_id[staying]
        return {'groups': removed, 'elements': len(doomed),
                'nodes': gone['nodes']}

    def merge_refusal(self, group_ids: Ids) -> str | None:
        """Why these element groups cannot be merged, or None when they can.

        There must be at least two, all in the geometry, holding the same
        element types and with the same properties. A merged element group
        has one material and one thickness or section, so merging groups
        that differ would change the model without saying so.

        Parameters
        ----------
        group_ids : sequence of int
            The element groups, by identifier.

        Returns
        -------
        str or None
        """
        ids = list(dict.fromkeys(int(b) for b in group_ids))
        if len(ids) < 2:
            return 'select two element groups or more to merge'
        missing = [b for b in ids if b not in self.group_id.tolist()]
        if missing:
            return f'no element group {missing[0]}'
        kinds = {frozenset(int(t) for t in
                           self.elem_type[self.elem_group == group])
                 for group in ids}
        if len(kinds) > 1:
            return 'the element groups hold different element types'
        held = [self.group_properties.get(group) for group in ids]
        if any(other != held[0] for other in held[1:]):
            return ('the element groups differ in material, thickness or section')
        return None

    def merge_groups(self, group_ids: Ids) -> dict[str, int]:
        """Merge several element groups into the first one (Merge Element
        Groups).

        The elements of the others are moved into the first group, and
        the others are removed with their properties. Raises ValueError
        with the reason from `merge_refusal` when the groups cannot be
        merged.

        Parameters
        ----------
        group_ids : sequence of int
            The element groups; the first keeps its id, name and properties.

        Returns
        -------
        dict of str to int
            'into', the id of the element group kept; 'groups', how many
            were merged into it; 'elements', how many elements moved.
        """
        # Merge Element Groups (Brandon, 2026-09-27)
        reason = self.merge_refusal(group_ids)
        if reason is not None:
            raise ValueError(reason)
        ids = list(dict.fromkeys(int(b) for b in group_ids))
        into, others = ids[0], set(ids[1:])
        moved = np.isin(self.elem_group, list(others))
        self.elem_group[moved] = into
        keep = ~np.isin(self.group_id, list(others))
        self.group_name = [name for name, k in zip(self.group_name, keep) if k]
        for group in others:
            self.group_properties.pop(group, None)
        self.group_id = self.group_id[keep]
        return {'into': into, 'groups': len(others),
                'elements': int(moved.sum())}

    def delete_elements(self, elem_ids: Ids, nodes: bool = False,
                        keep: Ids = ()) -> dict[str, int]:
        """Remove elements by id. Ids that are not present are ignored.

        With `nodes`, the nodes used only by these elements are removed
        too, which is what deleting an element group (`delete_groups`)
        and deleting elements in the app do. A node another element uses
        stays, and so does a node in `keep`: the project keeps the nodes
        its data is measured or solved at (`Project.delete_elements`), so
        redrawing a test geometry's panels leaves its measurement points.
        `nodes` is off by default because the merge removes the links it
        collapses and then remaps their nodes.

        Parameters
        ----------
        elem_ids : int or sequence of int
            The identifiers, one or many.
        nodes : bool, default False
            Also remove the nodes only these elements used.
        keep : sequence of int, optional
            Nodes never removed that way.

        Returns
        -------
        dict of str to int
            'elements', the number removed, and with `nodes`, 'nodes',
            the number of nodes removed.
        """
        # Without `nodes`, a part's bricks deleted from the Hexes row left
        # every node of the part in place (Brandon, 2026-10-12).
        if nodes:
            doomed = {int(e) for e in np.atleast_1d(elem_ids)}
            inside, outside = set(), set()
            for element, conn in zip(self.elem_id, self.elem_conn):
                (inside if int(element) in doomed else outside).update(
                    int(n) for n in conn)
            report = self.delete_elements(sorted(doomed))
            spared = {int(n) for n in np.atleast_1d(keep)}
            gone = self.delete_nodes(sorted(inside - outside - spared))
            return {**report, 'nodes': gone['nodes']}
        rows = self._rows_for(elem_ids, self.elem_id)
        keep = np.ones(len(self.elem_id), dtype=bool)
        keep[rows] = False
        # rebuilt once: deleting row by row shifted every later row each
        # time, quadratic, and meshing a part again deletes its whole mesh
        self.elem_conn = [conn for conn, kept in zip(self.elem_conn, keep) if kept]
        self.elem_id = self.elem_id[keep]
        self.elem_type = self.elem_type[keep]
        self.elem_color = self.elem_color[keep]
        self.elem_group = self.elem_group[keep]
        # An element group whose last element has gone is still an element group: exodus
        # files carry empty ones, and forgetting the name would lose on a
        # round trip exactly what element groups were added to keep.
        return {'elements': len(rows)}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geometry):
            return NotImplemented
        scalar = all(np.array_equal(getattr(self, f), getattr(other, f)) for f in (
            'node_id', 'node_def_cs', 'node_disp_cs', 'node_color',
            'cs_id', 'cs_type',
            'elem_id', 'elem_type', 'elem_color', 'elem_group'))
        arrays = (np.allclose(self.node_xyz, other.node_xyz)
                  and np.allclose(self.cs_matrix, other.cs_matrix))
        ragged = (len(self.elem_conn) == len(other.elem_conn)
                  and all(np.array_equal(a, b) for a, b in
                          zip(self.elem_conn, other.elem_conn)))
        names = self.cs_name == other.cs_name
        return (scalar and arrays and ragged and names
                and self.length_unit == other.length_unit)

    def __repr__(self) -> str:
        units = self.length_unit if self.units_defined else 'units undefined'
        return (f"Geometry({self.num_nodes} nodes, {len(self.cs_id)} coordinate systems, "
                f"{len(self.elem_conn)} elements, {len(self.group_id)} element groups, "
                f"{units})")

    def save(self, path: str | os.PathLike) -> None:
        """Write the geometry to a file of its own.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write it.

        Returns
        -------
        None
        """
        from ..io import native
        native.save(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> Geometry:
        """Read a geometry written by `save`.

        Raises TypeError when the file holds something other than a
        geometry.
        """
        from ..io import native
        obj = native.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a geometry")
        return obj

    def plot(self, unit_system: UnitSystem | None = None,
             **kwargs: Any) -> Any:
        """Draw the geometry's nodes and elements.

        Parameters
        ----------
        unit_system : UnitSystem, optional
            Units to draw in.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..viz.geometry import plot_geometry
        return plot_geometry(self, unit_system=unit_system, **kwargs)

    def plot_dofs(self, source: Any, quantity: str,
                  unit_system: UnitSystem | None = None,
                  **kwargs: Any) -> Any:
        """Draw this geometry with a labeled arrow at every DOF where
        `source` has `quantity`, as the app's DOF arrows do.

        Parameters
        ----------
        source : DataArray
            The object whose degrees of freedom are drawn.
        quantity : str
            Which quantity's DOFs to show, such as 'acceleration'.
        unit_system : UnitSystem, optional
            Units to draw in.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plotter the scene is in.
        """
        from ..viz.geometry import plot_dofs
        return plot_dofs(self, source, quantity, unit_system=unit_system,
                         **kwargs)
