"""Test/model geometry: nodes, coordinate systems, tracelines, elements.

Node coordinates are stored in SI meters once their units are known, resolved
into the global cartesian frame. A geometry imported from a source that does
not declare units (sdynpy .npz, exodus, a UNV without a 164) keeps the file's
raw coordinates and reports `length_unit is None` until `define_units()` is
called. Element types use the UFF dataset 2412 FE descriptor codes as the
interchange vocabulary.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
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
    # is one node and draws as a point. 136 is the exception and stays
    # one node here on purpose — the Nastran reader writes a CELAS2 as
    # a 136 at a single grid, and exodus maps its SPRING to one.
    136: ('spring', 1, 'point'),
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


def placed(local: ArrayLike, kind: int, matrix: ArrayLike) -> np.ndarray:
    """One point written in a frame, as global cartesian.

    `kind` is the frame's `CS_TYPES` code and `matrix` its `(4, 3)`
    rows — three directions then the origin. A cylindrical frame
    writes (r, theta, z) and a spherical one (r, theta from +z, phi),
    both with the angles in **degrees**, which is what the universal
    file, a Nastran CORD2C/S and sdynpy all mean by them.
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
    """The inverse of `placed`: a global cartesian point as the frame
    would write it, so a file states its nodes the way it stated them
    before."""
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
    """Node coordinates, each written in the frame it is placed in, as
    global cartesian — what `Geometry` stores.

    A file states a node's position in whatever frame it was drawn in,
    and the frame is the node's `def_cs`. Reading those numbers as if
    they were global puts the node somewhere else entirely, which is
    what a geometry with local placement frames looked like (Brandon,
    2026-09-20). A node whose frame the file does not define, or which
    names the global frame, is taken as written.

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
    """The inverse of `to_global`, for a writer that states each node
    in the frame the geometry says it is placed in."""
    xyz = np.asarray(node_xyz, dtype=float).reshape(-1, 3)
    frames = _frames(cs_id, cs_type, cs_matrix)
    out = xyz.copy()
    for k, code in enumerate(np.asarray(node_def_cs).ravel()[:len(xyz)]):
        found = frames.get(int(code))
        if found is not None:
            out[k] = written_in(xyz[k], found[0], found[1])
    return out


def reconcile_type(code: int, node_count: int) -> int:
    """The descriptor that matches how many nodes an element really
    has, when the two disagree.

    A file states both, and they can contradict each other: one wrote
    a four-node element as descriptor 41, a plane-stress *triangle*,
    and it drew as a triangle with the fourth node dropped (Brandon,
    2026-09-20). The count is the element; the descriptor is a label
    on it. So a descriptor whose count is wrong is exchanged for the
    one in its own family that carries that many nodes — 41 with four
    nodes is 44, the plane-stress quadrilateral, keeping the family
    the file chose and mending only what it got wrong.

    2412 numbers a family in one decade, triangles before
    quadrilaterals, which is what makes the sibling findable. When no
    sibling fits, the descriptor stands: an element nobody can name is
    better carried as written than renamed to a guess.

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
    triangle, four for a quadrilateral, whichever family its
    descriptor names and however many nodes it carries.

    Read off the name in `ELEMENT_TYPES` rather than from the node
    count, which cannot tell a nine-node cubic triangle from a
    nine-node quadrilateral (Brandon, 2026-09-20).

    Parameters
    ----------
    code : int
        The element's descriptor.

    Returns
    -------
    int
        3 or 4.
    """
    name = ELEMENT_TYPES[int(code)][0]
    return 3 if name.startswith('tri') else 4


class Geometry:
    """Nodes, coordinate systems, tracelines, elements and blocks.

    Parameters are array-likes; connectivity lists contain one integer array
    of node ids per traceline/element. All coordinates in SI meters.

    The arrays below are the storage; `nodes`, `coordinate_systems`,
    `tracelines`, `elements` and `blocks` are **views** onto them, which
    is how the tree lists a geometry and how a script should usually
    reach one. A view is not a copy — writing through a row writes here.

    Attributes:
        node_id: Every node's id. Unique, because connectivity and
            placement refer to nodes by id.
        node_xyz: `(nodes, 3)` coordinates, in meters once
            `length_unit` is declared and the file's raw numbers before
            that.
        node_def_cs: The coordinate system each node is *placed* in.
        node_disp_cs: The system each node is *measured* in — the frame a
            shape's values at that node are expressed in.
        node_color: Palette index per node.
        cs_id: Coordinate system ids. Unique, for the same reason as
            nodes.
        cs_name: A name per system, often empty.
        cs_type: 0 cartesian, 1 cylindrical, 2 spherical (`CS_TYPES`).
        cs_matrix: `(systems, 4, 3)` — three direction rows then the
            origin, so `cs_matrix[i, 3]` is where system *i* sits.
        traceline_id: Ids of the display polylines. Not unique: a UNV
            trace line that lifts the pen arrives as several runs under
            one id, and deleting that id removes all of them.
        traceline_color: Palette index per line.
        traceline_desc: Free text per line.
        traceline_conn: One array of node ids per line, in drawing order.
        elem_id: Element ids. Labels — nothing refers to them.
        elem_type: UFF dataset 2412 descriptor code per element
            (`ELEMENT_TYPES` names them and says how each is drawn).
        elem_color: Palette index per element.
        elem_conn: One array of node ids per element.
        elem_block: Which block each element belongs to, by block id.
        block_id: The declared blocks. Unique, since elements name them.
        block_name: A name per block — `'wing'`, `'arm front left'`. This
            is where a mesh records that a region is a different part
            from its neighbor, and `fem.Model.from_geometry` reads a
            member's section from it.
        length_unit: What the coordinates are in, or `None` while that
            has not been declared — in which case they are the file's own
            numbers and nothing has been scaled.
    """

    dimension = 'length'

    def __init__(self, node_id: Ids, node_xyz: ArrayLike,
                 node_def_cs: ArrayLike | None = None,
                 node_disp_cs: ArrayLike | None = None,
                 node_color: ArrayLike | None = None,
                 cs_id: Ids | None = None,
                 cs_name: Sequence[str] | None = None,
                 cs_type: ArrayLike | None = None,
                 cs_matrix: ArrayLike | None = None,
                 traceline_id: Ids | None = None,
                 traceline_color: ArrayLike | None = None,
                 traceline_desc: Sequence[str] | None = None,
                 traceline_conn: Sequence[ArrayLike] | None = None,
                 elem_id: Ids | None = None,
                 elem_type: ArrayLike | None = None,
                 elem_color: ArrayLike | None = None,
                 elem_conn: Sequence[ArrayLike] | None = None,
                 elem_block: Ids | None = None,
                 block_id: Ids | None = None,
                 block_name: Sequence[str] | None = None,
                 length_unit: str | None = None) -> None:
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

        t = len(traceline_conn) if traceline_conn is not None else 0
        self.traceline_id: IdArray = (
            _ids(traceline_id, 'traceline ids') if traceline_id is not None
            else self._default(None, t, None, arange=True))
        self.traceline_color: NDArray[np.int64] = self._default(
            traceline_color, t, 1)
        self.traceline_desc: list[str] = (
            list(traceline_desc) if traceline_desc is not None else [''] * t)
        self.traceline_conn: list[IdArray] = [
            np.asarray(c, dtype=np.int64) for c in (traceline_conn or [])]

        e = len(elem_conn) if elem_conn is not None else 0
        self.elem_id: IdArray = (
            _ids(elem_id, 'element ids') if elem_id is not None
            else self._default(None, e, None, arange=True))
        self.elem_type: NDArray[np.int64] = self._default(elem_type, e, 0)
        self.elem_color: NDArray[np.int64] = self._default(elem_color, e, 1)
        # Which block each element belongs to, and what the blocks are
        # called. This is how a mesh says "these elements are the wing and
        # those are the tail": exodus calls them element blocks, and it is
        # the only place a file records that a region is made of something
        # different from its neighbor. Read without keeping it, a
        # two-block file comes back as one anonymous block on the way out.
        self.elem_block: IdArray = (
            _ids(elem_block, 'element block ids') if elem_block is not None
            else self._default(None, e, 1))
        found = (np.unique(self.elem_block) if e
                 else np.array([], dtype=np.int64))
        self.block_id: IdArray = (
            _ids(block_id, 'block ids', unique=True)
            if block_id is not None else found)
        self.block_name: list[str] = (list(block_name) if block_name is not None
                                      else [''] * len(self.block_id))
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

        self.validate()

    @staticmethod
    def _default(value, n, fill, arange=False):
        if value is not None:
            return np.asarray(value, dtype=np.int64)
        return np.arange(1, n + 1, dtype=np.int64) if arange else np.full(n, fill, np.int64)

    def validate(self) -> None:
        """Check the geometry hangs together — every element's nodes
        present, every identifier unique — and report what does not."""
        # An id that other data points at has to mean one thing:
        # connectivity names nodes by id, and a node names the systems it
        # is placed and measured in. Traceline and element ids are
        # labels — nothing refers to them — and a UNV trace line that
        # lifts the pen legitimately arrives as several polylines under
        # one id, so they are not held to this.
        # the same helper the constructor uses, so a duplicate introduced
        # by an edit is refused in the same words as one handed in
        for label, values in (('node ids', self.node_id),
                              ('coordinate system ids', self.cs_id)):
            _ids(values, label, unique=True)
        known = set(self.node_id.tolist())
        for kind, conns in (('traceline', self.traceline_conn),
                            ('element', self.elem_conn)):
            for i, conn in enumerate(conns):
                missing = set(conn.tolist()) - known
                if missing:
                    raise ValueError(
                        f"{kind} {i} references unknown node ids {sorted(missing)}")
        if len(self.block_name) != len(self.block_id):
            raise ValueError(
                f'{len(self.block_name)} block names for '
                f'{len(self.block_id)} blocks')
        _ids(self.block_id, 'block ids', unique=True)
        stray = set(np.unique(self.elem_block).tolist()) - set(
            self.block_id.tolist())
        if stray:
            raise ValueError(
                'elements name blocks the geometry does not have: '
                + ', '.join(str(b) for b in sorted(stray)))
        for code in np.unique(self.elem_type) if len(self.elem_type) else []:
            if int(code) not in ELEMENT_TYPES:
                raise ValueError(f"Unknown element type code {int(code)}")

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
        """Every node: `ids`, `xyz`, `colors`, and the two systems."""
        return EntityView(self, 'nodes')

    @property
    def coordinate_systems(self) -> EntityView:
        """Every coordinate system: `ids`, `names`, `types`, `matrices`."""
        return EntityView(self, 'coordinate_systems')

    @property
    def tracelines(self) -> EntityView:
        """Every traceline: `ids`, `descriptions`, `colors`, `nodes`."""
        return EntityView(self, 'tracelines')

    @property
    def elements(self) -> EntityView:
        """Every element: `ids`, `types`, `colors`, `nodes`."""
        return EntityView(self, 'elements')

    @property
    def blocks(self) -> EntityView:
        """Every element block: `ids` and `names`.

        A block groups elements rather than holding them — which elements
        are in one is read off `elem_block` (`elements_in`), so moving an
        element between blocks is an edit to the element.
        """
        return EntityView(self, 'blocks')

    def node_index(self, node_ids: Ids) -> np.ndarray:
        """Positions of the given node ids in the node arrays.

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
        """(min_xyz, max_xyz), in meters once units are defined."""
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
            Those the geometry has no node for — what makes a
            data object incompatible with it.
        """
        from .data import parse_dof

        nodes = [parse_dof(dof)[0] for dof in dofs]
        known = self.contains_nodes([-1 if n is None else n for n in nodes])
        return [dof for dof, present in zip(dofs, known) if not present]

    @property
    def units_defined(self) -> bool:
        """Whether the geometry knows what its coordinates mean.
        False until `define_units` names the length unit."""
        return self.length_unit is not None

    def suggest_mass_properties(self) -> MassProperties:
        """The centroid of the nodes as the reference point, and no
        mass — unit rigid-body shapes about the middle of the model.

        The seed the rigid-body pane opens with and what
        `generate_rigid_body_modes` adopts when nothing was set: unlike
        a whole-record truncation, this is a real answer, and the one
        the virtual-point transformation wants most often.

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
        """Declare what the coordinates are in, converting them to SI.

        Re-declaring reinterprets the original file values rather than
        scaling twice, so a wrong guess can simply be corrected.

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
        self.length_unit = length_unit
        return self

    def undefine_units(self) -> Geometry:
        """Take the declaration back, restoring the file's raw coordinates."""
        if self.length_unit is None:
            return self
        self.node_xyz = from_si(self.node_xyz, self.length_unit, 'length')
        self.cs_matrix[:, 3, :] = from_si(self.cs_matrix[:, 3, :],
                                          self.length_unit, 'length')
        self.length_unit = None
        return self

    # ---- adding -------------------------------------------------------------

    def _next_id(self, array):
        return int(array.max()) + 1 if len(array) else 1

    def dof_direction(self, dof: str) -> np.ndarray | None:
        """A DOF's direction in global coordinates, through the frame
        its node is measured in.

        '101X+' at a node whose `node_disp_cs` is rotated is not global
        X. The report's grid of control channels reads which global
        axis a channel is nearest and how far off it sits (Brandon,
        2026-09-19), and this is where that is answered. A rotational
        DOF points along its axis, as the DOF arrows draw it.

        Parameters
        ----------
        dof : str
            The DOF, such as '101X+' or '1313RZ-'.

        Returns
        -------
        numpy.ndarray or None
            A unit vector, or None when the DOF names no node this
            geometry has, no axis (a node number alone), or a node
            measured in a cylindrical or spherical frame — whose local
            axes turn with the node's position, a reading this does
            not attempt rather than half-answer.
        """
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
        """Append a node at `xyz` (SI). Returns its id.

        Parameters
        ----------
        xyz : array_like
            The node's coordinates.
        node_id : int, optional
            Its identifier. The next free one when omitted.
        color : int, default 1
            Its display color index.
        def_cs : int, optional
            The coordinate system the position is given in.
        disp_cs : int, optional
            The coordinate system displacements are measured in.

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

    def add_coordinate_system(self, origin: ArrayLike = (0.0, 0.0, 0.0),
                              rotation: ArrayLike | None = None,
                              cs_id: int | None = None, name: str = '',
                              cs_type: int = 0) -> int:
        """Append a coordinate system. Returns its id.

        Parameters
        ----------
        origin : array_like, default (0, 0, 0)
            The system's origin.
        rotation : array_like, optional
            A 3x3 rotation matrix. Identity when omitted.
        cs_id : int, optional
            Its identifier. The next free one when omitted.
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

    def add_traceline(self, node_ids: Ids, color: int = 1,
                      description: str = '') -> int:
        """Append a traceline through the given nodes. Returns its index.

        Parameters
        ----------
        node_ids : int or sequence of int
            The nodes the line passes through, in order.
        color : int, default 1
            Its display color index.
        description : str, optional
            A label for it.

        Returns
        -------
        int
            The traceline's identifier.
        """
        nodes = np.asarray([int(n) for n in node_ids], dtype=np.int64)
        unknown = set(nodes.tolist()) - set(self.node_id.tolist())
        if unknown:
            raise ValueError(f'unknown nodes {sorted(unknown)}')
        if len(nodes) < 2:
            raise ValueError('a traceline needs at least two nodes')
        self.traceline_id = np.append(self.traceline_id,
                                      self._next_id(self.traceline_id))
        self.traceline_color = np.append(self.traceline_color, int(color))
        self.traceline_desc.append(str(description))
        self.traceline_conn.append(nodes)
        return len(self.traceline_conn) - 1

    def block_of(self, elem_id: int) -> str:
        """The name of the block an element belongs to, or ''.

        Parameters
        ----------
        elem_id : int
            Which element.

        Returns
        -------
        str
            The name of the block it belongs to.
        """
        row = int(np.flatnonzero(self.elem_id == int(elem_id))[0])
        block = int(self.elem_block[row])
        where = np.flatnonzero(self.block_id == block)
        return self.block_name[int(where[0])] if len(where) else ''

    def elements_in(self, block: int | str) -> list[int]:
        """The ids of the elements in a block, named or numbered.

        Parameters
        ----------
        block : int or str
            A block, by identifier or by name.

        Returns
        -------
        list of int
            The identifiers of the elements it holds.
        """
        if isinstance(block, str):
            where = [i for i, name in enumerate(self.block_name)
                     if name == block]
            if not where:
                return []
            block = int(self.block_id[where[0]])
        return [int(self.elem_id[i])
                for i in np.flatnonzero(self.elem_block == int(block))]

    def add_block(self, name: str = '', block_id: int | None = None) -> int:
        """Declare an element block. Returns its id.

        A block with nothing in it is legitimate — exodus files carry
        empty ones, and a block has to exist before an element can be put
        in it.

        Parameters
        ----------
        name : str, optional
            What to call the block.
        block_id : int, optional
            Its identifier. The next free one when omitted.

        Returns
        -------
        int
            The block's identifier.
        """
        block_id = (self._next_id(self.block_id) if block_id is None
                    else int(block_id))
        if block_id in self.block_id:
            raise ValueError(f'block {block_id} already exists')
        self.block_id = np.append(self.block_id, block_id)
        self.block_name.append(str(name))
        return block_id

    def add_element(self, node_ids: Ids, elem_type: int | None = None,
                    color: int = 1, block: int | None = None) -> int:
        """Append an element. Type defaults to whatever fits the node count:
        2 nodes a beam, 3 a triangle, 4 a quadrilateral. Returns its index.

        Parameters
        ----------
        node_ids : int or sequence of int
            The nodes the element connects, in order.
        elem_type : int, optional
            The element type code. Inferred from the node
            count when omitted.
        color : int, default 1
            Its display color index.
        block : int, optional
            Which block it belongs to.

        Returns
        -------
        int
            The element's identifier.
        """
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
        self.elem_id = np.append(self.elem_id, self._next_id(self.elem_id))
        self.elem_type = np.append(self.elem_type, int(elem_type))
        self.elem_color = np.append(self.elem_color, int(color))
        if block is None:
            block = int(self.block_id[0]) if len(self.block_id) else 1
        block = int(block)
        if block not in self.block_id.tolist():
            self.block_id = np.append(self.block_id, block)
            self.block_name.append('')
        self.elem_block = np.append(self.elem_block, block)
        self.elem_conn.append(nodes)
        return len(self.elem_conn) - 1

    # ---- deletion -----------------------------------------------------------

    def renumber_node(self, row: int, node_id: int) -> None:
        """Give a node a new id, carrying its tracelines and elements over.

        Connectivity names nodes by id, so a rename that left it alone
        would orphan every line and face touching the node.

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
        for conn in (*self.traceline_conn, *self.elem_conn):
            conn[conn == old] = node_id

    def renumber_block(self, row: int, block_id: int) -> None:
        """Give a block a new id, carrying its elements over.

        An element names its block by id, so a renumber that left them
        alone would put every element of the block in a block that is no
        longer there — which `validate` refuses, after the damage.

        Parameters
        ----------
        row : int
            Which block, by row.
        block_id : int
            Its new identifier.

        Returns
        -------
        None
        """
        block_id = int(block_id)
        clash = np.flatnonzero(self.block_id == block_id)
        if len(clash) and clash[0] != row:
            raise ValueError(f'block {block_id} already exists')
        old = int(self.block_id[row])
        self.block_id[row] = block_id
        self.elem_block[self.elem_block == old] = block_id

    def renumber_coordinate_system(self, row: int, cs_id: int) -> None:
        """Give a coordinate system a new id, repointing the nodes using it.

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
        """Remove nodes, and anything that referenced them.

        A traceline or element naming a deleted node cannot survive, so it
        goes too. Returns what was removed, for reporting.

        Parameters
        ----------
        node_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        dict of str to int
            How many of each kind were removed, including the
            dependents that went with them.
        """
        wanted = {int(node) for node in node_ids}
        keep = ~np.isin(self.node_id, list(wanted))
        removed_nodes = int((~keep).sum())
        if not removed_nodes:
            return {'nodes': 0, 'tracelines': 0, 'elements': 0}

        orphan_lines = [int(self.traceline_id[i])
                        for i, conn in enumerate(self.traceline_conn)
                        if wanted & {int(n) for n in conn}]
        orphan_elements = [int(self.elem_id[i])
                           for i, conn in enumerate(self.elem_conn)
                           if wanted & {int(n) for n in conn}]
        self.delete_tracelines(orphan_lines)
        self.delete_elements(orphan_elements)

        for name in ('node_id', 'node_def_cs', 'node_disp_cs', 'node_color'):
            setattr(self, name, getattr(self, name)[keep])
        self.node_xyz = self.node_xyz[keep]
        self.validate()
        return {'nodes': removed_nodes, 'tracelines': len(orphan_lines),
                'elements': len(orphan_elements)}

    def delete_coordinate_systems(self, cs_ids: Ids) -> dict[str, int]:
        """Remove coordinate systems, reassigning any node that used them.

        The last coordinate system is never removed — nodes must reference
        something.

        Parameters
        ----------
        cs_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        dict of str to int
            How many of each kind were removed, including the
            dependents that went with them.
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

    def delete_tracelines(self, traceline_ids: Ids) -> dict[str, int]:
        """Remove tracelines by id. Ids that are not there are ignored.

        One id can name several polylines — a UNV trace line that lifts
        the pen arrives split into its drawn runs, all still that one
        trace line — and deleting it removes all of them, which is what
        deleting that trace line means.

        Parameters
        ----------
        traceline_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        dict of str to int
            How many of each kind were removed, including the
            dependents that went with them.
        """
        rows = self._rows_for(traceline_ids, self.traceline_id)
        for row in rows:
            del self.traceline_conn[row]
            del self.traceline_desc[row]
        keep = np.ones(len(self.traceline_id), dtype=bool)
        keep[rows] = False
        self.traceline_id = self.traceline_id[keep]
        self.traceline_color = self.traceline_color[keep]
        return {'tracelines': len(rows)}

    def delete_blocks(self, block_ids: Ids) -> dict[str, int]:
        """Remove blocks, moving anything in them into the first one left.

        Deleting the grouping must not delete what was grouped — an
        element is a piece of the mesh and a block is a label on it — so
        the elements move rather than go, the way a node whose coordinate
        system is deleted is reassigned. The last block cannot go while
        any element names one; with no elements at all there is nothing
        to hold and the geometry may have none.

        Parameters
        ----------
        block_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        dict of str to int
            How many of each kind were removed, including the
            dependents that went with them.
        """
        wanted = {int(block) for block in block_ids}
        keep = ~np.isin(self.block_id, list(wanted))
        removed = int((~keep).sum())
        if not removed:
            return {'blocks': 0, 'elements_reassigned': 0}
        if keep.sum() == 0 and len(self.elem_block):
            raise ValueError(
                'a geometry with elements needs a block to put them in')
        reassigned = 0
        if keep.sum():
            fallback = int(self.block_id[keep][0])
            stale = np.isin(self.elem_block, list(wanted))
            reassigned = int(stale.sum())
            self.elem_block[stale] = fallback
        self.block_name = [name for name, k in zip(self.block_name, keep) if k]
        self.block_id = self.block_id[keep]
        return {'blocks': removed, 'elements_reassigned': reassigned}

    def delete_elements(self, elem_ids: Ids) -> dict[str, int]:
        """Remove elements by id. Ids that are not there are ignored.

        Parameters
        ----------
        elem_ids : int or sequence of int
            The identifiers, one or many.

        Returns
        -------
        dict of str to int
            How many of each kind were removed, including the
            dependents that went with them.
        """
        rows = self._rows_for(elem_ids, self.elem_id)
        for row in rows:
            del self.elem_conn[row]
        keep = np.ones(len(self.elem_id), dtype=bool)
        keep[rows] = False
        self.elem_id = self.elem_id[keep]
        self.elem_type = self.elem_type[keep]
        self.elem_color = self.elem_color[keep]
        self.elem_block = self.elem_block[keep]
        # A block whose last element has gone is still a block: exodus
        # files carry empty ones, and forgetting the name would lose on a
        # round trip exactly what blocks were added to keep.
        return {'elements': len(rows)}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geometry):
            return NotImplemented
        scalar = all(np.array_equal(getattr(self, f), getattr(other, f)) for f in (
            'node_id', 'node_def_cs', 'node_disp_cs', 'node_color',
            'cs_id', 'cs_type', 'traceline_id', 'traceline_color',
            'elem_id', 'elem_type', 'elem_color', 'elem_block'))
        arrays = (np.allclose(self.node_xyz, other.node_xyz)
                  and np.allclose(self.cs_matrix, other.cs_matrix))
        ragged = (len(self.traceline_conn) == len(other.traceline_conn)
                  and all(np.array_equal(a, b) for a, b in
                          zip(self.traceline_conn, other.traceline_conn))
                  and len(self.elem_conn) == len(other.elem_conn)
                  and all(np.array_equal(a, b) for a, b in
                          zip(self.elem_conn, other.elem_conn)))
        names = (self.cs_name == other.cs_name
                 and self.traceline_desc == other.traceline_desc)
        return (scalar and arrays and ragged and names
                and self.length_unit == other.length_unit)

    def __repr__(self) -> str:
        units = self.length_unit if self.units_defined else 'units undefined'
        return (f"Geometry({self.num_nodes} nodes, {len(self.cs_id)} coordinate systems, "
                f"{len(self.traceline_conn)} tracelines, {len(self.elem_conn)} elements, "
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
        from ..io import native
        obj = native.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a geometry")
        return obj

    def plot(self, unit_system: UnitSystem | None = None,
             **kwargs: Any) -> Any:
        """Draw the geometry: nodes, elements and tracelines.

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
        """This geometry with labeled arrows at every DOF `source`
        measures as `quantity` — the GUI's DOF arrows.

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
