"""A beam finite element model, and the eigensolution it gives.

visualdynamics is an analysis toolset, so this is deliberately the smallest
modeling capability that produces something worth analyzing: three-dimensional
two-node beams and lumped masses, assembled into mass and stiffness matrices
and solved for real normal modes. It exists because a demonstration needs a
*truth* model — a dense analytical answer the measured one can be compared
against — and because building one should not require reaching for another
package.

There are two ways in. Build a structure member by member, as the example
below does; or draw a shape as a surface mesh and let `from_geometry` put a
member along every edge of it and share the mass over its nodes. The second
is the shorter road from any geometry — imported or built — to a set of
modes, and `visualdynamics.demo.drone` is built that way throughout.

What it is not: a general finite element code. There are no shells, no
solids, no constraints beyond fixing degrees of freedom, and no static
solution. A plate is modeled the way a frame is, as a grillage of beams,
which is a real modeling choice with a known cost rather than an
approximation hidden inside an element. Wiring a surface mesh's edges is the
same bargain: it answers what order of mode density and what mode families a
shape has, and it does not pretend to be shell theory.

Everything here is SI, because the mass and stiffness matrices are the one
place in visualdynamics where several dimensions have to be consistent with each
other at once — a length in millimeters beside a modulus in pascals is not
wrong in any single entry, it is wrong only in the answer. The objects that
come out (`Geometry`, `ShapeSet`) carry their units declared, so the
conversion happens once, at the boundary, as it does everywhere else.

    from visualdynamics import fem

    aluminum = fem.Material('aluminum', youngs_modulus=70e9, density=2700)
    tube = fem.Section.round_tube('16 mm tube', outer=0.016, wall=0.001)

    model = fem.Model('cantilever')
    for i in range(11):
        model.add_node(100 + i, i * 0.1, 0.0, 0.0)
    model.add_chain(range(100, 111), aluminum, tube)
    shapes = model.eigensolution(maximum_frequency=2000,
                                 fixed=['100'], damping=0.01)

The element formulation is Euler-Bernoulli with a consistent mass matrix:
no shear flexibility and no section rotary inertia, which is right while a
member is slender (length more than about ten times its depth) and
progressively optimistic when it is not. A drone arm at 180 mm long and
16 mm deep is comfortably inside that; a stubby mounting stub is not, and
frequencies there read high.

References
----------
The Euler-Bernoulli beam element and its consistent mass matrix are
textbook; these are where they are set out.

1. Przemieniecki, J. S. (1968). *Theory of Matrix Structural
   Analysis*. McGraw-Hill.
2. Cook, R. D., Malkus, D. S., Plesha, M. E., & Witt, R. J. (2002).
   *Concepts and Applications of Finite Element Analysis*, 4th ed.
   Wiley. The cubic Hermite shape functions the consistent mass
   follows from, and why consistent rather than lumped changes the
   frequencies returned.
3. Craig, R. R., & Kurdila, A. J. (2006). *Fundamentals of Structural
   Dynamics*, 2nd ed. Wiley. The generalized symmetric eigenproblem
   this solves, and normalization to unit modal mass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .data import direction_code
from .geometry import ELEMENT_TYPES, Geometry
from .shapes import ShapeSet

#: The six degrees of freedom every node carries, in order. A beam
#: transmits moments, so rotations are structural here rather than a
#: bookkeeping convenience — leaving them out would model pin joints.
DIRECTIONS = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')

#: When a mode's strain energy is called zero, so its frequency is set to
#: zero exactly rather than left at the 1e-4 Hz round-off gives it.
#: Downstream code distinguishes a rigid mode by `frequency == 0` — the FRF
#: synthesis cancels its 0/0 that way — so 'very small' would arrive as a
#: resonance a hair above DC instead.
#:
#: The measure is how much cancellation happened: phi^T K phi against the
#: same sum with every sign removed. A motion that strains nothing is a
#: total cancellation of large terms and lands at 1e-16; a real mode, even
#: the softest one a fine mesh has, keeps a part in a million of what it
#: adds up. Six orders of margin either side, and no knowledge of how stiff
#: the structure is.
#:
#: Projecting onto the analytically known rigid-body vectors was tried
#: first and is wrong: a cantilever's first bending mode is 99.04% rigid
#: translation in the mass metric, because that is what a cantilever mode
#: looks like. Being *shaped* like a rigid motion and *being* one are
#: different questions, and only the energy asks the second.
ZERO_ENERGY = 1e-12


@dataclass(frozen=True)
class Material:
    """An isotropic elastic material.

    Shear modulus is derived from the modulus and Poisson's ratio unless it
    is given: for carbon fiber laminates the isotropic relation is a poor
    guess and the torsional stiffness it implies can be out by a factor of
    two, so the door is left open to state it.
    """

    name: str
    youngs_modulus: float          #: Pa
    density: float                 #: kg/m^3
    poissons_ratio: float = 0.3
    modulus_of_rigidity: float | None = None    #: Pa, derived when None

    @property
    def shear_modulus(self) -> float:
        if self.modulus_of_rigidity is not None:
            return float(self.modulus_of_rigidity)
        return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))


@dataclass(frozen=True)
class Section:
    """A beam cross section, as the four numbers the element needs.

    `iy` and `iz` are second moments of area about the element's own local
    y and z axes, and `j` is the torsion constant — St Venant's, which
    equals the polar second moment only for a circular section. For
    anything else it is smaller, and using the polar value overstates
    torsional stiffness; the constructors below carry the right formula for
    the shapes they build.

    `polar` is the *inertia* term, always the true polar second moment
    (iy + iz), because rotary inertia about the axis is a property of where
    the material is and has nothing to do with warping.
    """

    name: str
    area: float                    #: m^2
    iy: float                      #: m^4, bending about local y
    iz: float                      #: m^4, bending about local z
    j: float                       #: m^4, St Venant torsion constant

    @property
    def polar(self) -> float:
        return self.iy + self.iz

    @classmethod
    def round_tube(cls, name: str, outer: float, wall: float) -> Section:
        """A circular tube, given its outside diameter and wall thickness."""
        ro, ri = outer / 2.0, outer / 2.0 - wall
        if ri < 0:
            raise ValueError(f'{name}: wall {wall} is thicker than the radius')
        area = math.pi * (ro ** 2 - ri ** 2)
        i = math.pi * (ro ** 4 - ri ** 4) / 4.0
        # a closed circular section is the one case where torsion is the
        # polar moment exactly: it does not warp
        return cls(name, area, i, i, 2.0 * i)

    @classmethod
    def rod(cls, name: str, diameter: float) -> Section:
        """A solid circular rod."""
        return cls.round_tube(name, diameter, diameter / 2.0)

    @classmethod
    def rectangle(cls, name: str, width: float, height: float) -> Section:
        """A solid rectangle, `width` along local y and `height` along z."""
        area = width * height
        iz = width ** 3 * height / 12.0     # bending in the local x-y plane
        iy = width * height ** 3 / 12.0     # bending in the local x-z plane
        a, b = max(width, height) / 2.0, min(width, height) / 2.0
        # St Venant's constant for a solid rectangle, to better than 0.1%
        # over the whole aspect range (Roark): the series in b/a truncated
        # where the next term is under a part in a thousand
        j = a * b ** 3 * (16 / 3 - 3.36 * (b / a) * (1 - b ** 4 / (12 * a ** 4)))
        return cls(name, area, iy, iz, j)

    @classmethod
    def square_tube(cls, name: str, width: float, wall: float) -> Section:
        """A square tube, given its outside width and wall thickness."""
        inner = width - 2.0 * wall
        if inner <= 0:
            raise ValueError(f'{name}: wall {wall} closes the section')
        area = width ** 2 - inner ** 2
        i = (width ** 4 - inner ** 4) / 12.0
        # Bredt's thin-wall formula: J = 4 A_m^2 t / s, with A_m the area
        # inside the wall centreline and s that centreline's length
        mean = width - wall
        j = 4.0 * (mean ** 2) ** 2 * wall / (4.0 * mean)
        return cls(name, area, i, i, j)


@dataclass
class Beam:
    """One two-node beam element."""

    node_a: int
    node_b: int
    material: Material
    section: Section
    #: a vector defining the local x-y plane, so the section's iy and iz
    #: mean something on a member that is not round. None picks a default.
    orientation: tuple[float, float, float] | None = None
    color: int = 1
    group: str = ''


@dataclass
class Plate:
    """One four-node rectangular plate-bending element.

    A flat shell: plane-stress membrane action in its own plane and
    Mindlin bending out of it, with the transverse shear tied at the
    edge midpoints (MITC4). The tying is not optional finesse — a
    plain bilinear Mindlin element locks in shear as the plate gets
    thin, and a 12x12 mesh of the locked element puts the first
    elastic mode of a thin free plate several times too high.

    Rectangles only, and refused otherwise rather than silently
    mis-integrated: the tying directions assume the natural axes align
    with the sides, which is exactly true for a rectangle and only
    approximately for anything else. The models this module exists to
    build mesh rectangular panels; a skewed general quad earns its
    place when something needs it, with the covariant transforms and
    the validation that come with it.

    Nodes run around the perimeter: 1-2 is the first edge, 1-4 the
    second, corner 3 opposite corner 1.
    """

    nodes: tuple[int, int, int, int]
    material: Material
    thickness: float               #: m
    color: int = 1
    group: str = ''


@dataclass
class LumpedMass:
    """A rigid item carried at a node: a motor, a battery, a camera.

    The inertias are about the global axes through the node. A point mass
    leaves them zero, which is honest for something small against the
    members carrying it and wrong for a battery the size of the structure —
    a mass with no inertia cannot rock, so a rocking mode simply will not
    appear.
    """

    node: int
    mass: float                                     #: kg
    inertia: tuple[float, float, float] = (0.0, 0.0, 0.0)   #: kg m^2
    name: str = ''


@dataclass
class Face:
    """A cosmetic face: shading, not stiffness.

    A grillage of beams reads as a wireframe, and a wireframe of a drone
    deck reads as nothing much. Faces spanning nodes that are already
    there give the renderer something to shade and the animation something
    to deform, while contributing nothing to the matrices — which is
    exactly the truth about them, and is why they are a separate kind
    rather than a zero-stiffness element.
    """

    nodes: tuple[int, ...]
    color: int = 1
    group: str = ''


def connected_pieces(neighbors: dict[int, set[int]]) -> list[list[int]]:
    """The connected components of an adjacency map, largest first.

    One piece is a structure; more than one is that many free bodies,
    each bringing its own six zero-frequency modes. The flood fill is
    shared by the model (asking over its beams and plates) and the
    drone's drawing (asking over its face edges, before a model
    exists), so the two cannot disagree about what "joined" means.
    """
    seen: set[int] = set()
    found: list[list[int]] = []
    for start in neighbors:
        if start in seen:
            continue
        stack, piece = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            piece.append(node)
            stack.extend(neighbors[node] - seen)
        found.append(sorted(piece))
    return sorted(found, key=len, reverse=True)


class Model:
    """Nodes, beams and lumped masses, and the modes they imply.

    Assembly is dense. The matrices are (6 x nodes) square, so a
    three-hundred-node model is a 1800 x 1800 pair — 26 MB and a couple of
    seconds to solve — and a thousand nodes is 290 MB and a minute or two.
    That is the working ceiling, and it is a deliberate one: a sparse
    assembly and a subspace solver would raise it by an order of magnitude
    and cost more code than the models this is here to build need. Measure
    before assuming it is the problem.

    Attributes:
        name: What the model is called; it becomes the geometry's name
            and the shape set's comment.
        length_unit: What the coordinates are in. Everything inside is
            SI; this is what the `Geometry` is told on the way out.
        beams: Every member, as `Beam` records naming two nodes, a
            material and a section.
        masses: Lumped masses, as `LumpedMass` records at a node.
        faces: Surfaces, as `Face` records naming three or four nodes.
            They carry no stiffness — a face is drawn, and its *edges*
            are what carry members.
    """

    def __init__(self, name: str = '', length_unit: str = 'm') -> None:
        self.name: str = name
        #: everything here is SI; this is what the Geometry is told
        self.length_unit: str = length_unit
        self._nodes: dict[int, np.ndarray] = {}
        self._node_group: dict[int, str] = {}
        self.beams: list[Beam] = []
        self.plates: list[Plate] = []
        self.masses: list[LumpedMass] = []
        self.faces: list[Face] = []

    # ---- building ---------------------------------------------------------

    def add_node(self, node_id: int, x: float, y: float, z: float,
                 group: str = '') -> int:
        node_id = int(node_id)
        if node_id in self._nodes:
            raise ValueError(f'node {node_id} is already in the model')
        self._nodes[node_id] = np.array([x, y, z], dtype=np.float64)
        self._node_group[node_id] = group
        return node_id

    def add_beam(self, node_a: int, node_b: int, material: Material,
                 section: Section,
                 orientation: Sequence[float] | None = None,
                 color: int = 1, group: str = '') -> Beam:
        for node in (node_a, node_b):
            if int(node) not in self._nodes:
                raise ValueError(f'beam names node {node}, which is not in the model')
        if int(node_a) == int(node_b):
            raise ValueError(f'beam from node {node_a} to itself has no length')
        beam = Beam(int(node_a), int(node_b), material, section,
                    None if orientation is None
                    else tuple(float(v) for v in orientation), color, group)
        self.beams.append(beam)
        return beam

    def add_chain(self, nodes: Sequence[int], material: Material,
                  section: Section,
                  orientation: Sequence[float] | None = None,
                  color: int = 1, group: str = '') -> list[Beam]:
        """A run of beams through consecutive nodes — a member, in one call."""
        nodes = [int(n) for n in nodes]
        return [self.add_beam(a, b, material, section, orientation, color, group)
                for a, b in pairwise(nodes)]

    def add_plate(self, nodes: Sequence[int], material: Material,
                  thickness: float, color: int = 1,
                  group: str = '') -> Plate:
        """One rectangular plate element over four existing nodes."""
        nodes = tuple(int(n) for n in nodes)
        if len(nodes) != 4 or len(set(nodes)) != 4:
            raise ValueError('a plate spans four distinct nodes')
        for node in nodes:
            if node not in self._nodes:
                raise ValueError(
                    f'plate names node {node}, which is not in the model')
        if float(thickness) <= 0.0:
            raise ValueError('a plate needs a positive thickness')
        # validate the rectangle at build time, not at solve time: the
        # person holding the bad corner coordinate is the one adding it
        _plate_frame(*[self._nodes[n] for n in nodes])
        plate = Plate(nodes, material, float(thickness), color, group)
        self.plates.append(plate)
        return plate

    def add_mass(self, node: int, mass: float,
                 inertia: Sequence[float] = (0.0, 0.0, 0.0),
                 name: str = '') -> LumpedMass:
        if int(node) not in self._nodes:
            raise ValueError(f'mass names node {node}, which is not in the model')
        item = LumpedMass(int(node), float(mass),
                          tuple(float(v) for v in inertia), name)
        self.masses.append(item)
        return item

    def add_face(self, nodes: Sequence[int], color: int = 1,
                 group: str = '') -> Face:
        nodes = tuple(int(n) for n in nodes)
        if len(nodes) not in (3, 4):
            raise ValueError('a face spans three or four nodes')
        for node in nodes:
            if node not in self._nodes:
                raise ValueError(f'face names node {node}, which is not in the model')
        face = Face(nodes, color, group)
        self.faces.append(face)
        return face

    @classmethod
    def from_geometry(cls, geometry: Geometry, material: Material,
                      section: Section,
                      total_mass: float | None = None,
                      tracelines: bool = False, name: str = '',
                      groups: dict[int, str] | None = None,
                      sections: dict[str, Section] | None = None) -> Model:
        """A structure from a drawn shape: members on its edges, mass at its
        nodes.

        The shortest route from *any* geometry to a set of modes. Every
        element contributes its own edges as beams — a face gives its
        perimeter, a line element gives its run — and the mass is shared
        equally over the nodes. What comes back is a model that can be
        solved like any other.

        This is a sanity-check tool, not a mesher. The members are a stand-in
        for whatever the real structure is: one section for the whole model,
        chosen to put the modes where they are wanted, and the answer scales
        as its square root. Shell bending, membrane action and any real
        thickness are simply not represented. What it *is* good for is
        looking at a shape and asking what order of mode density and what
        mode families it has — which is the question a display model usually
        raises first.

        `tracelines` wires the display polylines too, which is what a
        wireframe geometry with no elements needs to hold together at all.
        `groups` labels the nodes by the part they belong to; a Geometry
        does not carry that, and the first question asked of any result is
        which part of the structure a mode lives in. `sections` then gives
        one part a section of its own — {'prop': stiffer} — applied where
        both ends of an edge belong to it, which is how a part is made
        stiffer or softer than the rest without redrawing anything.
        """
        model = cls(name or getattr(geometry, 'name', '') or 'geometry',
                    length_unit=geometry.length_unit or 'm')
        labels = dict(groups or {})
        if not labels:
            # A geometry that carries element blocks says for itself which
            # part each node belongs to, so nothing has to be passed
            # alongside it. That matters because a side-channel does not
            # survive being saved: a geometry written to a file and read
            # back could not reproduce the model it came from.
            labels = _labels_from_blocks(geometry)
        for node, xyz in zip(geometry.node_id, geometry.node_xyz):
            model.add_node(int(node), *[float(v) for v in xyz],
                           group=labels.get(int(node), ''))

        # A member takes its section from the block of the element it came
        # from, not from labels on its end nodes. A node on a seam belongs
        # to two parts and can only answer for one, which left 24 of the
        # drone's 1248 blade members reading as ordinary frame; an element
        # belongs to exactly one block and is never ambiguous.
        parts = _block_labels(geometry)
        runs: list[tuple[list[int], str]] = []
        for index, (kind, conn) in enumerate(zip(geometry.elem_type,
                                                 geometry.elem_conn)):
            nodes = [int(n) for n in conn]
            label = parts[index] if index < len(parts) else ''
            shape = ELEMENT_TYPES.get(int(kind), (None, 0, 'line'))[2]
            if shape == 'line':
                runs.append((nodes, label))
            else:
                # a face closes on itself; three or four of them also
                # become a Face, so the shape can be drawn back
                runs.append((nodes + nodes[:1], label))
                if len(nodes) in (3, 4):
                    model.add_face(nodes, group=label)
        if tracelines:
            runs.extend(([int(n) for n in conn], '')
                        for conn in geometry.traceline_conn)

        seen: set[frozenset[int]] = set()
        for run, label in runs:
            chosen = section
            for part, alternative in (sections or {}).items():
                if label.startswith(part):
                    chosen = alternative
                    break
            for a, b in pairwise(run):
                edge = frozenset((a, b))
                if a == b or edge in seen:
                    continue
                seen.add(edge)
                model.add_beam(a, b, material, chosen, group=label)
        if not seen:
            raise ValueError(
                'the geometry has no elements or tracelines to make members '
                'from, so there is nothing to connect its nodes')

        loose = sorted(set(model.node_ids)
                       - {n for edge in seen for n in edge})
        if loose:
            raise ValueError(
                f'{len(loose)} nodes are connected to nothing, so they would '
                'carry mass with no stiffness and the solution would not '
                'factorize: ' + ', '.join(str(n) for n in loose[:8]))
        pieces = model.pieces()
        if len(pieces) > 1:
            sizes = ', '.join(str(len(p)) for p in pieces[:6])
            raise ValueError(
                f'the geometry is {len(pieces)} disconnected pieces ({sizes} '
                'nodes), which would solve as that many free bodies and give '
                f'{6 * len(pieces)} zero-frequency modes rather than 6. Its '
                'elements and tracelines do not join them: either they are '
                'meant to be separate structures, or the connectivity is '
                'incomplete')
        if total_mass is not None:
            model.distribute_mass(total_mass)
        return model

    def pieces(self) -> list[list[int]]:
        """The structure's disconnected parts, largest first.

        One piece is a structure; more than one is that many free bodies,
        each bringing its own six zero-frequency modes. It is the first
        thing to ask of any model that comes back too floppy, and the
        answer is almost never what was intended — the old airplane
        fixture, meshed and wired through its own tracelines, turned out
        to be three: the fuselage, a wing and the tail, none joined.
        """
        neighbors: dict[int, set[int]] = {n: set() for n in self.node_ids}
        for beam in self.beams:
            if beam.node_a in neighbors and beam.node_b in neighbors:
                neighbors[beam.node_a].add(beam.node_b)
                neighbors[beam.node_b].add(beam.node_a)
        for plate in self.plates:
            for k, node in enumerate(plate.nodes):
                other = plate.nodes[(k + 1) % 4]
                neighbors[node].add(other)
                neighbors[other].add(node)
        return connected_pieces(neighbors)

    def wire_faces(self, material: Material, section: Section,
                   color: int = 1, group: str = '') -> int:
        """Put a beam along every edge of every face, and return how many.

        This is the shortest road from a shape to a structure: draw the
        thing as a surface mesh, and let its own edges be its members. The
        geometry then *is* the model, with nothing derived, nothing tied,
        and no second set of nodes that only exist to be looked at.

        Beams, not axial springs. A spring on each edge leaves a quad free
        to shear and a flat sheet free to fold — the edges never change
        length, so nothing resists it — and the model comes back a
        mechanism with as many zero-frequency modes as it has panels. A
        beam carries moment, so a wireframe of them is a space frame and
        stands up. It is the same element the rest of this module uses.

        Shared edges are wired once. An edge that already has a beam is
        left alone, so explicit members (a truss strut, a standoff) can be
        placed first and keep their own section.
        """
        seen = {frozenset((beam.node_a, beam.node_b)) for beam in self.beams}
        added = 0
        for face in self.faces:
            nodes = face.nodes
            for k, node in enumerate(nodes):
                other = nodes[(k + 1) % len(nodes)]
                edge = frozenset((node, other))
                if edge in seen or node == other:
                    continue
                seen.add(edge)
                self.add_beam(node, other, material, section, color=color,
                              group=group or f'edge {len(self.beams)}')
                added += 1
        return added

    def distribute_mass(self, total: float, spin: float = 1.0 / 12.0
                        ) -> dict[int, float]:
        """Share `total` over the nodes by the members meeting at each.

        A node's share is proportional to the length of member it carries —
        half of every member that reaches it — so mass follows the material
        rather than the mesh. Returns {node: mass}.

        Sharing it *equally* is the obvious thing and it is a trap, because
        a node count is a statement about how finely something was drawn
        rather than about how much of it there is. On the demonstration
        airframe that put 42% of the mass into the propellers — they need
        the finest mesh to look right, so they collect the most nodes — and
        117 g on each rotor buried every airframe mode below 1.8 kHz under
        blade motion, while the battery, the heaviest real item on the
        aircraft, was left with 51 g.

        Each node also gets a rotary inertia, without which the three
        rotational degrees of freedom carry nothing, the mass matrix is
        singular and the Cholesky factorization fails outright. It is
        taken as `spin * m * L^2` with L the mean length of the members
        meeting there — the patch of structure the node stands for, spun
        about its own middle. The default 1/12 is a uniform rod's.
        """
        nodes = self.node_ids
        if not nodes:
            raise ValueError('there are no nodes to share the mass over')
        reach: dict[int, list[float]] = {node: [] for node in nodes}
        for beam in self.beams:
            length = self._length(beam)
            for node in (beam.node_a, beam.node_b):
                if node in reach:
                    reach[node].append(length)
        carried = {node: sum(lengths) / 2.0 for node, lengths in reach.items()}
        span = sum(carried.values())
        if span <= 0.0:
            raise ValueError(
                'no node carries any member, so there is nothing to share '
                'the mass out in proportion to')

        self.masses = [item for item in self.masses if item.name != 'shared']
        shares = {}
        for node in nodes:
            share = float(total) * carried[node] / span
            lengths = reach[node]
            scale = float(np.mean(lengths)) if lengths else 0.0
            inertia = spin * share * scale ** 2
            self.add_mass(node, share, (inertia, inertia, inertia),
                          name='shared')
            shares[node] = share
        return shares

    # ---- what is in it ----------------------------------------------------

    @property
    def node_ids(self) -> list[int]:
        """In insertion order: the matrices' row order follows this."""
        return list(self._nodes)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_dof(self) -> int:
        return 6 * len(self._nodes)

    def position(self, node: int) -> np.ndarray:
        return self._nodes[int(node)]

    def group(self, node: int) -> str:
        """What this node was added as part of — 'arm 2', 'top deck'.

        A label for the modeler's own use: nothing here reads it, but
        working out which part of a structure a mode lives in is the first
        question asked of any result, and reconstructing it from
        coordinates afterwards is guesswork.
        """
        return self._node_group[int(node)]

    def dof_strings(self) -> list[str]:
        """'101X+', '101Y+', … in the matrices' own order."""
        return [f'{node}{direction}'
                for node in self._nodes for direction in DIRECTIONS]

    @property
    def structural_mass(self) -> float:
        """What the members weigh, before anything is hung on them."""
        beams = float(sum(beam.section.area * beam.material.density
                          * self._length(beam) for beam in self.beams))
        plates = 0.0
        for plate in self.plates:
            _, a, b = _plate_frame(*[self._nodes[n] for n in plate.nodes])
            plates += plate.material.density * plate.thickness * a * b
        return beams + plates

    @property
    def total_mass(self) -> float:
        return self.structural_mass + float(sum(m.mass for m in self.masses))

    def _length(self, beam: Beam) -> float:
        return float(np.linalg.norm(self._nodes[beam.node_b]
                                    - self._nodes[beam.node_a]))

    # ---- the matrices -----------------------------------------------------

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the global mass and stiffness matrices.

        Rows and columns run structural node by structural node in
        insertion order, six per node in `DIRECTIONS` order, which is what
        `dof_strings()` spells out. Display nodes are absent: they carry
        nothing, so there is nothing of theirs to assemble.
        """
        n = self.num_dof
        mass = np.zeros((n, n), dtype=np.float64)
        stiffness = np.zeros((n, n), dtype=np.float64)
        index = {node: 6 * i for i, node in enumerate(self.node_ids)}

        for beam in self.beams:
            length = self._length(beam)
            if length == 0.0:
                raise ValueError(f'beam {beam.node_a}-{beam.node_b} has zero length')
            rotation = _element_axes(self._nodes[beam.node_a],
                                     self._nodes[beam.node_b], beam.orientation)
            transform = _block_diagonal(rotation, 4)
            # local -> global: k_g = T^T k_l T, with T mapping global
            # displacements onto local ones
            k = transform.T @ _beam_stiffness(beam.material, beam.section, length) @ transform
            m = transform.T @ _beam_mass(beam.material, beam.section, length) @ transform
            rows = np.r_[index[beam.node_a]:index[beam.node_a] + 6,
                         index[beam.node_b]:index[beam.node_b] + 6]
            grid = np.ix_(rows, rows)
            stiffness[grid] += k
            mass[grid] += m

        for plate in self.plates:
            corners = [self._nodes[n] for n in plate.nodes]
            rotation, a, b = _plate_frame(*corners)
            transform = _block_diagonal(rotation, 8)
            k_local, m_local = _plate_matrices(plate.material,
                                               plate.thickness, a, b)
            k = transform.T @ k_local @ transform
            m = transform.T @ m_local @ transform
            rows = np.concatenate([np.arange(index[n], index[n] + 6)
                                   for n in plate.nodes])
            grid = np.ix_(rows, rows)
            stiffness[grid] += k
            mass[grid] += m

        for item in self.masses:
            start = index[item.node]
            for offset in range(3):
                mass[start + offset, start + offset] += item.mass
            for offset, inertia in enumerate(item.inertia):
                mass[start + 3 + offset, start + 3 + offset] += inertia

        # assembly is symmetric by construction, but floating point addition
        # is not associative and the halves drift apart in the last bits;
        # eigh reads only one triangle, so an asymmetry here is silent
        return (mass + mass.T) / 2.0, (stiffness + stiffness.T) / 2.0

    def rigid_body_vectors(self) -> np.ndarray:
        """The six rigid-body motions, as columns over the model's DOFs.

        Written down from the node positions rather than found from the
        matrices: they are what the null space of an unconstrained
        stiffness matrix *is*, and knowing them in advance is what lets a
        rigid mode be recognized as rigid rather than as a very soft one.
        """
        n = self.num_dof
        solved = self.node_ids
        vectors = np.zeros((n, 6), dtype=np.float64)
        center = np.mean(np.array([self._nodes[k] for k in solved]), axis=0)
        for i, node in enumerate(solved):
            offset = self._nodes[node] - center
            for axis in range(3):
                vectors[6 * i + axis, axis] = 1.0            # translations
                # a small rotation about `axis` moves a point by the cross
                # product of the axis with its offset, and rotates it by
                # the axis itself
                spin = np.zeros(3)
                spin[axis] = 1.0
                vectors[6 * i:6 * i + 3, 3 + axis] = np.cross(spin, offset)
                vectors[6 * i + 3 + axis, 3 + axis] = 1.0
        return vectors

    # ---- the answer -------------------------------------------------------

    def eigensolution(self, maximum_frequency: float | None = None,
                      num_modes: int | None = None, damping: float = 0.0,
                      fixed: Sequence[str] = ()) -> ShapeSet:
        """Real normal modes, mass-normalized, as a ShapeSet.

        `fixed` names degrees of freedom to ground: '101X+' fixes one,
        '101' fixes all six of that node. The remaining problem is the
        symmetric generalized one, K phi = lambda M phi, solved by
        factoring M (Cholesky), reducing to a standard symmetric problem
        and transforming back — which is what makes the shapes come out
        mass-normalized to machine precision rather than normalized and
        then rescaled.

        `damping` is a fraction of critical, applied uniformly. A model
        has no damping of its own; it is stated so the modes can
        synthesize an FRF that looks like a measurement.
        """
        mass, stiffness = self.matrices()
        free = self._free_dofs(fixed)
        if not len(free):
            raise ValueError('every degree of freedom is fixed')
        reduced_m = mass[np.ix_(free, free)]
        reduced_k = stiffness[np.ix_(free, free)]

        try:
            factor = np.linalg.cholesky(reduced_m)
        except np.linalg.LinAlgError:
            starved = [self.dof_strings()[free[i]]
                       for i in np.flatnonzero(np.diag(reduced_m) <= 0.0)]
            raise ValueError(
                'the mass matrix is not positive definite, so these degrees '
                'of freedom carry no mass and no rotary inertia: '
                + (', '.join(starved[:6]) or 'none on the diagonal, so the '
                   'model is nearly a mechanism')) from None

        # M = L L^T, so K phi = lambda M phi becomes A y = lambda y with
        # A = L^-1 K L^-T and phi = L^-T y. y orthonormal then gives
        # phi^T M phi = y^T y = I exactly, which is the normalization we
        # want and not something applied afterwards.
        temporary = np.linalg.solve(factor, reduced_k)
        standard = np.linalg.solve(factor, temporary.T).T
        eigenvalues, vectors = np.linalg.eigh((standard + standard.T) / 2.0)
        shapes = np.linalg.solve(factor.T, vectors)

        full = np.zeros((self.num_dof, shapes.shape[1]), dtype=np.float64)
        full[free] = shapes
        # a rigid-body eigenvalue is zero plus round-off, and comes out
        # either side of it; the negative ones are not oscillations
        frequency = np.sqrt(np.clip(eigenvalues, 0.0, None)) / (2.0 * np.pi)
        frequency[_strains_nothing(full, stiffness)] = 0.0

        order = np.argsort(frequency, kind='stable')
        frequency, full = frequency[order], full[:, order]
        keep = np.ones(len(frequency), dtype=bool)
        if maximum_frequency is not None:
            keep &= frequency <= float(maximum_frequency)
        keep = np.flatnonzero(keep)
        if num_modes is not None:
            keep = keep[:int(num_modes)]

        return ShapeSet(frequency=frequency[keep],
                        damping=np.full(len(keep), float(damping)),
                        coordinate=self.dof_strings(),
                        shape_matrix=full[:, keep].T,
                        mass_unit='kg',
                        comment=self.name or '')

    def _free_dofs(self, fixed) -> np.ndarray:
        """Which rows survive after grounding what `fixed` names."""
        index = {node: 6 * i for i, node in enumerate(self.node_ids)}
        held = set()
        for item in fixed:
            text = str(item).strip()
            digits = 0
            while digits < len(text) and text[digits].isdigit():
                digits += 1
            node = int(text[:digits]) if digits else None
            if node not in index:
                raise ValueError(f'{item!r} does not name a node in the model')
            direction = text[digits:]
            if not direction:
                held.update(range(index[node], index[node] + 6))
            else:
                # a sign is meaningless when grounding — X- is the same
                # constraint as X+ — so the code's magnitude is the axis
                held.add(index[node] + abs(direction_code(direction)) - 1)
        return np.array([i for i in range(self.num_dof) if i not in held],
                        dtype=np.int64)

    # ---- what it looks like -----------------------------------------------

    def geometry(self, beams: bool = True) -> Geometry:
        """The model as a Geometry: nodes, beams as elements, faces as faces.

        Beams become element type 21 (beam2) rather than tracelines,
        because they *are* elements — a traceline is a line drawn through
        nodes to make a display readable, and confusing the two would make
        the model's own connectivity indistinguishable from a drawing aid
        the moment anything edited it.

        `beams=False` leaves them out, for a model whose members are all
        wrapped in surfaces: there the beams run *inside* the shells, so
        drawing them puts a wireframe over the thing it is the skeleton of.
        """
        node_ids = self.node_ids
        connectivity, types, colors = [], [], []
        for beam in self.beams if beams else ():
            connectivity.append([beam.node_a, beam.node_b])
            types.append(21)
            colors.append(beam.color)
        # plates are structural quads and appear as such — unlike faces,
        # which are drawings; both shade, only one carries stiffness
        for plate in self.plates:
            connectivity.append(list(plate.nodes))
            types.append(44)
            colors.append(plate.color)
        for face in self.faces:
            connectivity.append(list(face.nodes))
            types.append(44 if len(face.nodes) == 4 else 41)
            colors.append(face.color)
        # Elements go into the block of the part they belong to, so the
        # geometry states its own regions and a saved file can rebuild the
        # structure. Without this the blocks live only on the drawing, and
        # the model's own geometry — which is what gets saved — carries
        # nothing: the file comes back with every member the same section.
        # The part comes off the element, never off its first node: a node
        # on a seam belongs to two parts and answers for one, which put
        # five of the drone's blade members into the frame block.
        parts, blocks = {}, []
        for part in ([b.group for b in (self.beams if beams else ())]
                     + [p.group for p in self.plates]
                     + [f.group for f in self.faces]):
            blocks.append(parts.setdefault(part or 'body', len(parts) + 1))
        return Geometry(
            node_id=node_ids,
            node_xyz=np.array([self._nodes[node] for node in node_ids]),
            elem_conn=connectivity or None,
            elem_type=types or None,
            elem_color=colors or None,
            elem_block=blocks or None,
            block_id=list(parts.values()) or None,
            block_name=list(parts) or None,
            length_unit=self.length_unit)


def _block_labels(geometry) -> list[str]:
    """The block name of each element, in the geometry's own order."""
    names = list(getattr(geometry, 'block_name', []))
    ids = np.asarray(getattr(geometry, 'block_id', []), dtype=np.int64)
    named = {int(b): (names[i] if i < len(names) else '')
             for i, b in enumerate(ids)}
    block = np.asarray(getattr(geometry, 'elem_block', []), dtype=np.int64)
    return [named.get(int(b), '') for b in block]


def _labels_from_blocks(geometry) -> dict[int, str]:
    """{node: block name} from a geometry's own element blocks.

    A node on the seam between two blocks belongs to the one holding most
    of its elements, ties going to the block declared first. It cannot
    belong to both — a node has one part in every format that records the
    question — and giving it to whichever block happened to be numbered
    first emptied the parts that sit *between* others: the drone's canopy
    is ringed by waist, camera mount and four arms, and came back with a
    single node in it, which is one accelerometer where four were meant.

    Nothing structural rides on this. Sections come off the elements,
    which are never ambiguous; these labels name the part a node is in,
    for reading a mode and for placing sensors.
    """
    names = list(getattr(geometry, 'block_name', []))
    ids = np.asarray(getattr(geometry, 'block_id', []), dtype=np.int64)
    named = {int(b): (names[i] if i < len(names) else '')
             for i, b in enumerate(ids)}
    order = {name: i for i, name in enumerate(names)}
    block = np.asarray(getattr(geometry, 'elem_block', []), dtype=np.int64)
    votes: dict[int, dict[str, int]] = {}
    for i, conn in enumerate(geometry.elem_conn):
        label = named.get(int(block[i]), '') if i < len(block) else ''
        if not label:
            continue
        for node in conn:
            counted = votes.setdefault(int(node), {})
            counted[label] = counted.get(label, 0) + 1
    return {node: max(counted, key=lambda n: (counted[n], -order.get(n, 0)))
            for node, counted in votes.items()}


def _strains_nothing(shapes: np.ndarray, stiffness: np.ndarray) -> np.ndarray:
    """Which of these modes store no strain energy, and so sit at 0 Hz.

    Zero-energy is asked of the mode rather than of the structure, so a
    constrained model simply has none — a cantilever's modes all strain
    something — and a genuine mechanism (a node nothing connects to, a
    hinge built by accident) is found on the same footing as rigid-body
    motion. Both really are at zero frequency; the distinction between
    them is a modeling question, not a numerical one.
    """
    energy = np.sum(shapes * (stiffness @ shapes), axis=0)
    # the same sum with every cancellation removed: how big the terms were
    # before they annihilated each other
    magnitude = np.sum(np.abs(shapes) * (np.abs(stiffness) @ np.abs(shapes)),
                       axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(energy) / magnitude
    return np.nan_to_num(ratio, nan=0.0) < ZERO_ENERGY


# ---- element matrices -----------------------------------------------------


def _element_axes(start: np.ndarray, end: np.ndarray, orientation) -> np.ndarray:
    """The 3x3 rotation whose rows are the element's local axes.

    Local x runs from the first node to the second. The orientation vector
    fixes the roll about it by naming a direction the local x-y plane must
    contain — the same job Nastran's orientation vector does. Without one,
    global Z is used, falling back to global Y for a member that is nearly
    vertical and would otherwise have no plane at all.
    """
    axis = end - start
    length = np.linalg.norm(axis)
    axis = axis / length
    if orientation is None:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(axis @ reference)) > 0.99:
            reference = np.array([0.0, 1.0, 0.0])
    else:
        reference = np.asarray(orientation, dtype=np.float64)
        if np.linalg.norm(reference) == 0.0:
            raise ValueError('the orientation vector has no direction')
        reference = reference / np.linalg.norm(reference)
        if abs(float(axis @ reference)) > 0.9999:
            raise ValueError('the orientation vector lies along the beam, so '
                             'it names no plane')
    local_y = reference - float(axis @ reference) * axis
    local_y = local_y / np.linalg.norm(local_y)
    local_z = np.cross(axis, local_y)
    return np.array([axis, local_y, local_z])


def _block_diagonal(rotation: np.ndarray, times: int) -> np.ndarray:
    out = np.zeros((3 * times, 3 * times), dtype=np.float64)
    for i in range(times):
        out[3 * i:3 * i + 3, 3 * i:3 * i + 3] = rotation
    return out


def _plate_frame(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                 p4: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The element's own axes and side lengths, refusing non-rectangles.

    Local x runs along edge 1-2, local y along edge 1-4, local z is
    their normal. The checks are one relative tolerance: the corner
    angle must be right and corner 3 must sit where the other three
    say it does — which also catches a warped (non-planar) quad,
    because a flat rectangle is the only shape that passes both.
    """
    edge_x = np.asarray(p2, dtype=np.float64) - p1
    edge_y = np.asarray(p4, dtype=np.float64) - p1
    a, b = float(np.linalg.norm(edge_x)), float(np.linalg.norm(edge_y))
    if a == 0.0 or b == 0.0:
        raise ValueError('a plate edge has zero length')
    e1, e2 = edge_x / a, edge_y / b
    tolerance = 1e-6 * max(a, b)
    if abs(float(np.dot(edge_x, edge_y))) > tolerance * max(a, b):
        raise ValueError(
            'plate corners do not form a rectangle: the corner angle at '
            'node 1 is not square. Rectangular elements only — see Plate')
    if float(np.linalg.norm(p3 - (p1 + edge_x + edge_y))) > tolerance:
        raise ValueError(
            'plate corners do not form a rectangle: corner 3 is not '
            'where corners 1, 2 and 4 put it. Rectangular elements '
            'only — see Plate')
    return np.array([e1, e2, np.cross(e1, e2)]), a, b


#: transverse shear correction for a homogeneous section: the parabolic
#: shear distribution carries 5/6 of what a uniform one would
SHEAR_CORRECTION = 5.0 / 6.0

#: how much artificial drilling stiffness a plate carries, as a
#: fraction of G*t. Any small positive number serves: the penalty
#: exists so the in-plane rotation is not a mechanism. It acts on the
#: gap between the drilling rotation and the rotation the membrane
#: displacements themselves describe, (v,x - u,y)/2 — so a rigid
#: rotation, where the two agree exactly, strains it not at all and
#: the rigid-mode count stays six. Penalizing only the *differences*
#: between the four nodal drillings was tried first and leaves a
#: mesh-wide uniform spin with no displacement as a seventh
#: zero-frequency mode, which is how this constant earned its
#: sentence.
DRILLING_FRACTION = 1e-3


def _plate_matrices(material: Material, thickness: float, a: float,
                    b: float) -> tuple[np.ndarray, np.ndarray]:
    """Local stiffness and consistent mass for the rectangular shell.

    24x24, six DOFs per node in DIRECTIONS order, nodes at
    (0,0) (a,0) (a,b) (0,b) in the element's own plane. Membrane and
    bending integrate 2x2 Gauss (exact for the rectangle); the
    transverse shear is the MITC4 assumed field — gamma_xz tied at the
    midpoints of the two eta edges and interpolated linearly in eta,
    gamma_yz the mirror of that — which is what keeps the thin limit
    honest instead of shear-locked.

    Sign conventions, written down because they were derived once and
    should not be re-derived: the tilt of the normal toward +x is
    +theta_y and toward +y is -theta_x, so the curvatures are
    (theta_y,x | -theta_x,y | theta_y,y - theta_x,x) and the shears
    (w,x + theta_y | w,y - theta_x). Both vanish identically on every
    rigid motion, which the rigid-mode test holds to machine zero.
    """
    E, nu = material.youngs_modulus, material.poissons_ratio
    G, rho, t = material.shear_modulus, material.density, thickness
    plane = E / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0],
                                            [nu, 1.0, 0.0],
                                            [0.0, 0.0, (1.0 - nu) / 2.0]])
    d_membrane = t * plane
    d_bending = t ** 3 / 12.0 * plane
    d_shear = SHEAR_CORRECTION * G * t * np.eye(2)

    def basis(xi: float, eta: float):
        signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        n = np.array([(1 + xi * sx) * (1 + eta * sy) / 4.0
                      for sx, sy in signs])
        dx = np.array([sx * (1 + eta * sy) / 4.0 * 2.0 / a
                       for sx, sy in signs])
        dy = np.array([sy * (1 + xi * sx) / 4.0 * 2.0 / b
                       for sx, sy in signs])
        return n, dx, dy

    def shear_rows(xi: float, eta: float) -> np.ndarray:
        n, dx, dy = basis(xi, eta)
        rows = np.zeros((2, 24))
        for i in range(4):
            rows[0, 6 * i + 2] = dx[i]          # gamma_xz = w,x + theta_y
            rows[0, 6 * i + 4] = n[i]
            rows[1, 6 * i + 2] = dy[i]          # gamma_yz = w,y - theta_x
            rows[1, 6 * i + 3] = -n[i]
        return rows

    stiffness = np.zeros((24, 24))
    consistent = np.zeros((24, 24))
    gauss = 1.0 / math.sqrt(3.0)
    area_weight = a * b / 4.0                    # dA per unit natural area
    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            n, dx, dy = basis(xi, eta)
            b_membrane = np.zeros((3, 24))
            b_bending = np.zeros((3, 24))
            fields = np.zeros((6, 24))
            for i in range(4):
                col = 6 * i
                b_membrane[0, col] = dx[i]
                b_membrane[1, col + 1] = dy[i]
                b_membrane[2, col] = dy[i]
                b_membrane[2, col + 1] = dx[i]
                b_bending[0, col + 4] = dx[i]
                b_bending[1, col + 3] = -dy[i]
                b_bending[2, col + 3] = -dx[i]
                b_bending[2, col + 4] = dy[i]
                for field in range(6):
                    fields[field, col + field] = n[i]
            # the assumed shear: the raw rows evaluated at the tying
            # points, blended linearly across the element
            b_shear = np.zeros((2, 24))
            b_shear[0] = ((1 - eta) * shear_rows(0.0, -1.0)[0]
                          + (1 + eta) * shear_rows(0.0, 1.0)[0]) / 2.0
            b_shear[1] = ((1 - xi) * shear_rows(-1.0, 0.0)[1]
                          + (1 + xi) * shear_rows(1.0, 0.0)[1]) / 2.0

            # the drilling tie: theta_z minus the rotation the membrane
            # field describes, penalized softly (see DRILLING_FRACTION)
            b_drilling = np.zeros((1, 24))
            for i in range(4):
                col = 6 * i
                b_drilling[0, col] = dy[i] / 2.0
                b_drilling[0, col + 1] = -dx[i] / 2.0
                b_drilling[0, col + 5] = n[i]

            stiffness += area_weight * (
                b_membrane.T @ d_membrane @ b_membrane
                + b_bending.T @ d_bending @ b_bending
                + b_shear.T @ d_shear @ b_shear
                + DRILLING_FRACTION * G * t
                * (b_drilling.T @ b_drilling))
            # translations carry rho t, the bending rotations the
            # section's rotary inertia rho t^3/12. The drilling
            # rotation carries a thousandth of that — not zero, because
            # a massless DOF breaks the Cholesky factorization the
            # eigensolution stands on, and not the full value, because
            # inertia against penalty stiffness sets where the drilling
            # *artifact* modes land: at the full inertia a half-inch
            # plate grew a wall of a hundred and sixty fake modes at
            # 4.2 kHz, mid-band; at a thousandth they sit near 130 kHz,
            # far above anything the element is valid for. Rigid
            # rotation about the normal loses nothing measurable — its
            # inertia is the translations' r^2 terms, not this.
            spin = rho * t ** 3 / 12.0
            weights = np.diag([rho * t] * 3 + [spin, spin, spin * 1e-3])
            consistent += area_weight * (fields.T @ weights @ fields)

    return stiffness, consistent


def _beam_stiffness(material: Material, section: Section, length: float) -> np.ndarray:
    """The 12x12 local stiffness of an Euler-Bernoulli beam.

    Degrees of freedom run u, v, w, rx, ry, rz at the first node then the
    second. Bending in the local x-y plane pairs v with rz and uses iz;
    bending in x-z pairs w with ry and uses iy, with the signs of the
    coupling terms reversed because a positive rotation about y moves a
    point in the *negative* z direction.
    """
    e, g = material.youngs_modulus, material.shear_modulus
    a, iy, iz, j = section.area, section.iy, section.iz, section.j
    length2, length3 = length ** 2, length ** 3
    k = np.zeros((12, 12), dtype=np.float64)

    k[0, 0] = k[6, 6] = e * a / length
    k[0, 6] = -e * a / length

    k[3, 3] = k[9, 9] = g * j / length
    k[3, 9] = -g * j / length

    k[1, 1] = k[7, 7] = 12.0 * e * iz / length3
    k[1, 7] = -12.0 * e * iz / length3
    k[1, 5] = k[1, 11] = 6.0 * e * iz / length2
    k[5, 7] = k[7, 11] = -6.0 * e * iz / length2
    k[5, 5] = k[11, 11] = 4.0 * e * iz / length
    k[5, 11] = 2.0 * e * iz / length

    k[2, 2] = k[8, 8] = 12.0 * e * iy / length3
    k[2, 8] = -12.0 * e * iy / length3
    k[2, 4] = k[2, 10] = -6.0 * e * iy / length2
    k[4, 8] = k[8, 10] = 6.0 * e * iy / length2
    k[4, 4] = k[10, 10] = 4.0 * e * iy / length
    k[4, 10] = 2.0 * e * iy / length

    return k + np.triu(k, 1).T


def _beam_mass(material: Material, section: Section, length: float) -> np.ndarray:
    """The 12x12 consistent mass matrix of the same element.

    Consistent rather than lumped: the cubic shape functions that gave the
    stiffness give the mass too, so the pair is a proper Rayleigh-Ritz
    reduction and the frequencies converge from above. A lumped diagonal
    would converge from below and give the rotations no inertia at all,
    which would leave the mass matrix singular wherever nothing else does.
    """
    density, area = material.density, section.area
    unit = density * area * length / 420.0
    m = np.zeros((12, 12), dtype=np.float64)

    m[0, 0] = m[6, 6] = 140.0 * unit
    m[0, 6] = 70.0 * unit

    # torsion carries the polar second moment of area, not the torsion
    # constant: warping changes the stiffness, not where the material is
    rotary = density * section.polar * length / 6.0
    m[3, 3] = m[9, 9] = 2.0 * rotary
    m[3, 9] = rotary

    m[1, 1] = m[7, 7] = 156.0 * unit
    m[1, 7] = 54.0 * unit
    m[1, 5] = 22.0 * length * unit
    m[1, 11] = -13.0 * length * unit
    m[5, 7] = 13.0 * length * unit
    m[7, 11] = -22.0 * length * unit
    m[5, 5] = m[11, 11] = 4.0 * length ** 2 * unit
    m[5, 11] = -3.0 * length ** 2 * unit

    m[2, 2] = m[8, 8] = 156.0 * unit
    m[2, 8] = 54.0 * unit
    m[2, 4] = -22.0 * length * unit
    m[2, 10] = 13.0 * length * unit
    m[4, 8] = -13.0 * length * unit
    m[8, 10] = 22.0 * length * unit
    m[4, 4] = m[10, 10] = 4.0 * length ** 2 * unit
    m[4, 10] = -3.0 * length ** 2 * unit

    return m + np.triu(m, 1).T
