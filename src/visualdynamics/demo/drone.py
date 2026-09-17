"""The demonstration airframe: a quadcopter, built with visualdynamics.

This is the model every drone fixture comes from — the truth set, the reduced
test geometry, and the plant the Rattlesnake runs fly against. It is built
entirely with `visualdynamics.fem` — like every fixture model since the
sdynpy demo articles left the repository: a demonstration of the
toolset should be made by the toolset.

It builds a model and returns it; it writes no files. `testdata/`'s
`generate_drone.py`, in the visualdynamics-generators repository, is
what turns it into fixtures on disk.

    from visualdynamics.demo import drone

    model = drone.build()
    shapes = model.eigensolution(maximum_frequency=400, damping=0.01)

Run it directly to print what it weighs and where its modes are:

    python3 -m visualdynamics.demo.drone

**The geometry is the model.** Every surface is swept from a profile along a
path, every edge of every face becomes a beam, and the mass is shared equally
over the nodes. There is nothing derived and nothing tied: what you see is
what is solved, node for node.

That is the second design. The first was a beam skeleton with surfaces hung
off it on rigid ties — nodes that were drawn but not solved, their motion
extrapolated from the beam they rode. It worked, and it was cheap, and it was
a picture of a fidelity the model did not have: the arm walls looked meshed
and were carrying nothing. Letting the drawing be the structure is simpler to
write, simpler to explain, and honest about what it is.

**Beams on the edges, not springs.** A spring on each edge leaves a quad free
to shear and a flat panel free to fold, since neither changes any edge length,
and the model comes back a mechanism with a zero-frequency mode per panel. A
beam carries moment, so the same wireframe stands up. Proved on a cube before
anything else was built: six rigid-body modes, then 121.7 Hz.

**The members are massless and the nodes carry it all.** Equal shares of the
total, so the mass follows the model rather than a section table — which also
means it follows the *mesh*, and the mesh is kept roughly even for that
reason. Each node also gets a rotary inertia, without which the rotational
degrees of freedom carry nothing and the mass matrix will not factorize.

**The numbers are plausible, not surveyed.** One section size sets the whole
frame's stiffness, chosen to put the elastic modes in a band a shaker test
would use. This is a demonstration article, not a drone anyone has weighed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike

from visualdynamics import fem
from visualdynamics.core.geometry import Geometry

# ---- what it is made of ----------------------------------------------------

#: Carbon, and massless: every gram is at a node. The shear modulus is stated
#: rather than derived because a woven tube's is set by the matrix, not the
#: fibers, and Poisson's ratio would overstate torsion several times over.
FRAME = fem.Material('carbon frame', youngs_modulus=70e9, density=0.0,
                     modulus_of_rigidity=5e9)

#: The one number that sets how stiff the airframe is. Every edge in the
#: model is this section, so the whole frame moves together when it changes,
#: which is what makes it a single honest dial rather than a tuning surface.
#: The members are never drawn, so their size is a stiffness parameter and
#: nothing else.
#:
#: Sized for mode *density* rather than for any particular frequency: the
#: whole spectrum scales together, so what the section chooses is how many
#: modes land under a fixed ceiling. Bending dominates, so frequency goes
#: roughly as the diameter squared — 5.50 mm gave 42 modes under 2 kHz,
#: 4.00 gave 65, and this gives 94 with the first at 88 Hz. Relative
#: spacing is untouched by the change, so nothing became harder to fit.
MEMBER = fem.Section.round_tube('member', outer=0.0032, wall=0.00052)

#: The blades get a section of their own, and a much stiffer one. Shared
#: with the airframe they are long thin cantilevers whose bending modes are
#: simply the lowest things on the aircraft: modes 6 to 26 were all blade,
#: and the first airframe-led mode sat at 189 Hz with everything below it
#: buried. That is real — it is why a bench modal test is run props-off —
#: but it makes a demonstration about propellers rather than about an
#: airframe. Stiffened, the rotors behave as the lumps of mass on the motors
#: that they are at these frequencies, and the band belongs to the airframe.
BLADE = fem.Section.round_tube('blade', outer=0.013, wall=0.0021)

TOTAL_MASS = 1.100      #: kg, shared over the nodes by the members they carry

# ---- how big it is ---------------------------------------------------------

BODY_R = 0.058          #: the body's widest radius
BODY_TOP = 0.034        #: canopy apex above the datum
BODY_FLOOR = -0.030     #: belly pan below it
WAIST = 0.026           #: the open truss band between canopy and belly
ARM_R = 0.235           #: motor centers from the middle
ARM_THICK = 0.0064      #: a quarter inch of carbon, the slab the truss is cut from
ARM_DEEP = 0.042        #: the girder's depth at the body
ARM_SHALLOW = 0.022     #: and at the motor: a taper, as a real arm has
ARM_SHEAR = 0.000       #: how far the web leans, bay to bay
NACELLE_R = 0.021       #: motor can
NACELLE_Z = 0.030       #: how far the can stands above the arm
HUB_R = 0.011           #: the propeller hub on top of it
HUB_H = 0.013
PROP_R = 0.108          #: blade tip from the shaft — a nine inch propeller
PROP_ROOT = 0.026       #: chord at the root of a blade
PROP_TIP = 0.014        #: and at the tip
PROP_THICK = 0.0028     #: how thick the blade is at its fattest
PROP_TWIST = (26.0, 8.0)  #: pitch at root and tip, degrees
LEG_DROP = 0.105        #: feet below the datum

#: Front arms swept wide and rear arms longer: a deadcat, which is what a
#: camera drone does to keep the propellers out of shot — and, here, what
#: breaks the four-fold symmetry that would otherwise give repeated modes no
#: one-mode-at-a-time fit could separate.
ARMS = (
    ('front left', 58.0, ARM_R),
    ('rear left', 128.0, ARM_R * 1.07),
    ('rear right', 232.0, ARM_R * 1.07),
    ('front right', 302.0, ARM_R),
)

#: colors: 0 gray 1 blue 2 orange 3 green 4 red 5 purple 6 brown 7 pink
BODY_COLOR, ARM_COLOR, NACELLE_COLOR = 0, 1, 4
LEG_COLOR, BATTERY_COLOR, CAMERA_COLOR = 8, 2, 7
PROP_COLOR, HUB_COLOR = 5, 0


#: one cross-section: (across, up) offsets from the sweep's own frame
Profile = Sequence[tuple[float, float]]


# ---- drawing ---------------------------------------------------------------


def ring(radius: float, sides: int,
         phase: float = 0.0) -> list[tuple[float, float]]:
    """Points round a circle, in a sweep's own cross-section plane."""
    return [(radius * math.cos(phase + 2.0 * math.pi * s / sides),
             radius * math.sin(phase + 2.0 * math.pi * s / sides))
            for s in range(sides)]


def _frames(path):
    """A cross-section frame at every station of a polyline path.

    Tangents come from the neighbors, so a curved path gets a frame that
    turns with it — which is what lets an arched leg and a tapered arm be
    swept the same way as a straight tube. The reference direction is held
    fixed along the run so the section cannot spiral round it.
    """
    points = [np.asarray(p, dtype=float) for p in path]
    tangents = []
    for k in range(len(points)):
        if k == 0:
            step = points[1] - points[0]
        elif k == len(points) - 1:
            step = points[-1] - points[-2]
        else:
            step = points[k + 1] - points[k - 1]
        tangents.append(step / np.linalg.norm(step))
    overall = points[-1] - points[0]
    overall = overall / np.linalg.norm(overall)
    reference = (np.array([1.0, 0.0, 0.0]) if abs(overall[2]) > 0.9
                 else np.array([0.0, 0.0, 1.0]))
    out = []
    for point, tangent in zip(points, tangents):
        side = np.cross(tangent, reference)
        side = side / np.linalg.norm(side)
        out.append((point, side, np.cross(side, tangent)))
    return out


def sweep(shape: Shape, path: Sequence[ArrayLike],
          profiles: Profile | Sequence[Profile], group: str, color: int,
          cap_start: bool = False,
          cap_end: bool = False) -> list[list[int]]:
    """Sweep cross-sections along a path, as rings of nodes joined by quads.

    `profiles` is one profile per station, so a tube can taper, swell or
    change shape along its length; pass a single profile to keep it
    uniform. Returns the rings.
    """
    if not isinstance(profiles[0], (list, tuple)) or isinstance(
            profiles[0][0], (int, float)):
        stations: Sequence[Profile] = [profiles] * len(path)  # type: ignore[list-item]
    else:
        stations = profiles                                 # type: ignore[assignment]
    rings = []
    for (center, side, up), profile in zip(_frames(path), stations):
        row = []
        for a, b in profile:
            point = center + a * side + b * up
            row.append(shape.node(point, group))
        rings.append(row)
    for lower, upper in pairwise(rings):
        for s in range(len(lower)):
            t = (s + 1) % len(lower)
            shape.face([lower[s], lower[t], upper[t], upper[s]], color, group)
    for wanted, row, center in ((cap_start, rings[0], path[0]),
                                (cap_end, rings[-1], path[-1])):
        if not wanted:
            continue
        hub = shape.node(np.asarray(center, dtype=float), group)
        for s in range(len(row)):
            shape.face([hub, row[s], row[(s + 1) % len(row)]], color, group)
    return rings


def bar(shape: Shape, start: ArrayLike, end: ArrayLike, thick: float, depth: float,
        normal: ArrayLike, group: str,
        color: int) -> list[list[int]] | None:
    """A flat rectangular member between two points, in a stated plane.

    The frame is given rather than worked out, because a truss has members
    at every angle and letting each pick its own reference direction lets
    the sections twist relative to one another. `normal` is the plane the
    truss lies in, so every bar in one arm is the same slab of material
    seen from a different angle — which is what a plate truss is.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    tangent = end - start
    length = float(np.linalg.norm(tangent))
    if length < 1e-9:
        return None
    tangent = tangent / length
    across = np.asarray(normal, dtype=float)
    across = across / np.linalg.norm(across)
    within = np.cross(tangent, across)
    within = within / np.linalg.norm(within)
    rings = []
    for point in (start, end):
        rings.append([shape.node(point + a * across + b * within, group)
                      for a, b in ((-thick / 2, -depth / 2),
                                   (thick / 2, -depth / 2),
                                   (thick / 2, depth / 2),
                                   (-thick / 2, depth / 2))])
    for k in range(4):
        t = (k + 1) % 4
        shape.face([rings[0][k], rings[0][t], rings[1][t], rings[1][k]], color)
    # the ends are closed onto the joint itself, which is a node the
    # neighboring bars also close onto — so the joint is shared, not
    # stitched, and the truss is one surface rather than a pile of sticks
    for row, point in ((rings[0], start), (rings[1], end)):
        hub = shape.node(point, group)
        for k in range(4):
            shape.face([hub, row[k], row[(k + 1) % 4]], color)
    return rings


def blade_profile(chord: float, thick: float,
                  twist: float) -> list[tuple[float, float]]:
    """A blade section: a thin cambered shape, rotated to its pitch.

    Twist is applied to the profile rather than to the sweep, because the
    path is a straight radial line and its frame does not turn — a blade
    that is flat at the tip and coarse at the root is the whole reason a
    propeller looks like one rather than like a paddle.
    """
    points = ((-0.50, 0.00), (-0.30, -0.42), (0.10, -0.50), (0.42, -0.26),
              (0.50, 0.00), (0.30, 0.40), (-0.10, 0.46), (-0.38, 0.28))
    angle = math.radians(twist)
    cos, sin = math.cos(angle), math.sin(angle)
    out = []
    for a, b in points:
        u, v = a * chord, b * thick
        out.append((u * cos - v * sin, u * sin + v * cos))
    return out


def _propeller(shape, nlo, nhi, center, sides, name, blades=2, handed=1.0):
    """A hub on the motor, and blades radiating from it.

    The hub grows off the nacelle's own rim through a flat shoulder rather
    than standing a capped cylinder on a capped cylinder — two caps in the
    same place are two faces buried inside the solid, and they are exactly
    what makes a model look assembled rather than printed. The blade roots
    are left open inside the hub for the same reason: no face where nothing
    can be seen.
    """
    group = f'prop {name}'
    base = np.asarray(center, dtype=float)
    # the hub is one more box on the stack, closing the tower's open top
    rim = [nlo[-1][0], nlo[-1][1], nhi[-1][1], nhi[-1][0]]
    upper = [shape.node(np.asarray(shape.position(n))
                        + np.array([0.0, 0.0, HUB_H]), group) for n in rim]
    for k in range(4):
        t = (k + 1) % 4
        shape.face([rim[k], rim[t], upper[t], upper[k]], HUB_COLOR)
    # a crown node in the middle of the hub's top, so the blades have
    # something of the hub's own to close onto: an uncapped blade root
    # welds to nothing and every blade comes back a free body
    top = base + np.array([0.0, 0.0, HUB_H])
    crown = shape.node(top, group)
    for k in range(4):
        shape.face([crown, upper[k], upper[(k + 1) % 4]], HUB_COLOR)

    stations = 7
    for b in range(blades):
        azimuth = 2.0 * math.pi * b / blades
        out = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
        side = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0]) * handed
        path, profiles = [top], []
        for k in range(stations + 1):
            t = k / stations
            r = HUB_R * 0.4 + (PROP_R - HUB_R * 0.4) * t
            # a little sweep back and a little coning, as a real blade has
            point = (top + out * r + side * (0.020 * t ** 2)
                     + np.array([0.0, 0.0, 0.012 * t ** 1.5]))
            path.append(point)
            chord = PROP_ROOT + (PROP_TIP - PROP_ROOT) * math.sqrt(t)
            thick = PROP_THICK * (1.0 - 0.55 * t)
            twist = PROP_TWIST[0] + (PROP_TWIST[1] - PROP_TWIST[0]) * t
            profiles.append(blade_profile(chord, thick, twist * handed))
        profiles.insert(0, blade_profile(PROP_ROOT * 0.5, PROP_THICK,
                                         PROP_TWIST[0] * handed))
        sweep(shape, path, profiles, group, PROP_COLOR, cap_start=True,
              cap_end=True)


def bridge(shape: Shape, lower: Sequence[int], upper: Sequence[int],
           color: int) -> None:
    """Quads between two rings that already exist — a shared joint."""
    for s in range(len(lower)):
        t = (s + 1) % len(lower)
        shape.face([lower[s], lower[t], upper[t], upper[s]], color)


class Shape:
    """A drawing: nodes and faces, and nothing else.

    No members, no mass, no elements of any kind — the whole airframe is
    built into one of these, and `fem.Model.from_geometry` is what turns it
    into a structure afterwards. Keeping the two apart is the point: the
    geometry is authored once and the physics is derived from it, so there
    is no second description to disagree with the first.

    Anything landing on an existing node is welded to it. Parts are drawn
    independently — a strut between two ring nodes, a nacelle standing on
    an arm — and where two of them meet they put a node in the same place
    twice. Drawn, that looks joined; solved, it is not, and each loose
    piece brings six zero-frequency modes. The airframe came back with 210
    against the six it should have before this was here: 24 truss struts,
    4 nacelles, 4 legs and 2 camera mounts, every one a free body.
    """

    def __init__(self, first: int = 1000,
                 tolerance: float = 1e-6) -> None:
        self.tolerance: float = tolerance
        self.xyz: dict[int, np.ndarray] = {}
        self.group: dict[int, str] = {}
        self.faces: list[tuple[list[int], int]] = []
        self.face_group: list[str] = []
        self._at: dict[tuple, int] = {}
        self._next = first

    def node(self, point: ArrayLike, group: str = '') -> int:
        point = np.asarray(point, dtype=float)
        key = tuple(round(float(v) / self.tolerance) for v in point)
        if key in self._at:
            return self._at[key]
        node = self._next
        self.xyz[node] = point
        self.group[node] = group
        self._at[key] = node
        self._next += 1
        return node

    def face(self, nodes: Sequence[int], color: int = 1,
             group: str = '') -> None:
        # the part that drew it, not the part its first node happens to
        # belong to: a node on a seam belongs to two, a face to one
        self.faces.append(([int(n) for n in nodes], int(color)))
        self.face_group.append(group or self.group.get(int(nodes[0]), ''))

    def drop_face(self, nodes: Sequence[int]) -> bool:
        """Remove a face by the nodes it spans, if it is there."""
        wanted = set(nodes)
        for index, (existing, _color) in enumerate(self.faces):
            if set(existing) == wanted:
                self.faces.pop(index)
                return True
        return False

    def position(self, node: int) -> np.ndarray:
        return self.xyz[int(node)]

    def pieces(self) -> list[list[int]]:
        """The drawing's disconnected parts, largest first.

        Asked of the drawing rather than of a solved model, because that is
        when it can still be fixed cheaply. Two surfaces that touch are not
        joined unless they share nodes or something spans them, and the
        cost of finding out later is six zero-frequency modes per loose
        piece with nothing on screen to say which.
        """
        neighbors: dict[int, set[int]] = {n: set() for n in self.xyz}
        for nodes, _color in self.faces:
            for k, node in enumerate(nodes):
                other = nodes[(k + 1) % len(nodes)]
                neighbors[node].add(other)
                neighbors[other].add(node)
        return fem.connected_pieces(neighbors)

    def orient(self) -> int:
        """Wind every face the same way round, and turn the lot outward.

        Windings were being corrected girder by girder — reverse if the
        frame was flipped, reverse again if the rows ran the other way —
        and it never came right, because a part's winding depends on how
        it was *built* and the answer wanted is a property of the finished
        surface. So this asks the surface instead: walk face to face over
        shared edges, and where two neighbors traverse their shared edge
        the same way round, one of them is inside out. Then check the
        signed volume and turn everything over if the whole shell ended up
        pointing in.

        Faces whose normals point into the solid show as dark patches, and
        would color by displacement from the wrong side.
        """
        edges: dict[frozenset, list[int]] = {}
        for index, (nodes, _color) in enumerate(self.faces):
            for k, node in enumerate(nodes):
                edges.setdefault(
                    frozenset((node, nodes[(k + 1) % len(nodes)])),
                    []).append(index)

        def runs(index: int, a: int, b: int) -> bool:
            """Does face `index` traverse this edge from a to b?"""
            nodes = self.faces[index][0]
            for k, node in enumerate(nodes):
                if node == a and nodes[(k + 1) % len(nodes)] == b:
                    return True
            return False

        seen, flipped = set(), 0
        for start in range(len(self.faces)):
            if start in seen:
                continue
            seen.add(start)
            stack = [start]
            while stack:
                index = stack.pop()
                nodes = self.faces[index][0]
                for k, node in enumerate(nodes):
                    other = nodes[(k + 1) % len(nodes)]
                    for neighbor in edges[frozenset((node, other))]:
                        if neighbor in seen:
                            continue
                        seen.add(neighbor)
                        # neighbors agree when they cross the shared edge
                        # in opposite directions; agreeing means one is
                        # wound the wrong way round
                        if runs(neighbor, node, other):
                            face, color = self.faces[neighbor]
                            self.faces[neighbor] = (face[::-1], color)
                            flipped += 1
                        stack.append(neighbor)

        volume = 0.0
        for nodes, _color in self.faces:
            points = [self.xyz[n] for n in nodes]
            for k in range(1, len(points) - 1):
                volume += float(np.dot(points[0],
                                       np.cross(points[k], points[k + 1])))
        if volume < 0.0:
            self.faces = [(nodes[::-1], color) for nodes, color in self.faces]
            flipped += len(self.faces)
        return flipped

    def prune(self) -> int:
        """Drop nodes no face uses, and say how many there were.

        A window more than one cell tall leaves the nodes strictly inside
        it belonging to nothing: the edges between two hole cells get no
        wall, and there is no face on either side. Rather than forbid tall
        windows — which is what keeps a truss looking like a slotted plate
        — the drawing simply forgets the nodes nothing referred to.
        """
        used = {n for nodes, _color in self.faces for n in nodes}
        loose = [n for n in self.xyz if n not in used]
        for node in loose:
            key = tuple(round(float(v) / self.tolerance)
                        for v in self.xyz[node])
            self._at.pop(key, None)
            self.xyz.pop(node)
            self.group.pop(node, None)
        return len(loose)

    def geometry(self, length_unit: str = 'm') -> Geometry:
        """What was drawn, as a Geometry: quads and triangles, no lines.

        Each face goes into the block of the part that drew it, so the
        geometry says by itself which region is which — and a saved file
        can rebuild the structure without anything passed alongside it.

        Blocks are named for the whole part, sides and all ('arm front
        left', not 'arm'). Collapsing them to the part alone reads more
        tidily and loses the one thing the sensor set is picked by: which
        of the four arms a node is on.
        """
        ids = sorted(self.xyz)
        parts, blocks = {}, []
        for group in self.face_group:
            part = group or 'body'
            blocks.append(parts.setdefault(part, len(parts) + 1))
        return Geometry(
            node_id=ids,
            node_xyz=np.array([self.xyz[n] for n in ids]),
            elem_conn=[nodes for nodes, _ in self.faces],
            elem_type=[44 if len(nodes) == 4 else 41
                       for nodes, _ in self.faces],
            elem_color=[color for _, color in self.faces],
            elem_block=blocks,
            block_id=list(parts.values()),
            block_name=list(parts),
            length_unit=length_unit)


# ---- the airframe ----------------------------------------------------------


def _body(shape, sides, rings_up, rows):
    """The center: a curved canopy over a curved belly, on a waist band.

    The waist is a grid of nodes rather than a ring — `rows` of them up its
    height — because the arms grow straight out of it. An arm takes one
    column pair of that grid as its root cross-section and sweeps outward
    from there, so the join is nodes held in common and not a bridge over
    a gap. It is the difference between a printed part and an assembly.
    """
    waist = [[shape.node((BODY_R * math.cos(2 * math.pi * i / sides),
                          BODY_R * math.sin(2 * math.pi * i / sides),
                          -WAIST / 2 + WAIST * j / rows), 'waist')
              for j in range(rows + 1)] for i in range(sides)]

    # a canopy: a superellipsoid cap standing on the top of the waist
    top = [[waist[i][rows] for i in range(sides)]]
    for k in range(1, rings_up + 1):
        t = k / rings_up
        radius = BODY_R * math.cos(t * math.pi / 2) ** 0.65
        height = WAIST / 2 + BODY_TOP * math.sin(t * math.pi / 2)
        if radius < 1e-4:
            break
        top.append([shape.node((radius * math.cos(2 * math.pi * i / sides),
                                radius * math.sin(2 * math.pi * i / sides),
                                height), 'canopy')
                    for i in range(sides)])
    apex = shape.node((0.0, 0.0, WAIST / 2 + BODY_TOP), 'canopy')
    for lower, upper in pairwise(top):
        bridge(shape, lower, upper, BODY_COLOR)
    for i in range(sides):
        shape.face([apex, top[-1][(i + 1) % sides], top[-1][i]], BODY_COLOR)

    # and a belly pan under the other side of it
    bottom = [[waist[i][0] for i in range(sides)]]
    for k in range(1, rings_up + 1):
        t = k / rings_up
        radius = BODY_R * math.cos(t * math.pi / 2) ** 0.55
        height = -WAIST / 2 + BODY_FLOOR * math.sin(t * math.pi / 2)
        if radius < 1e-4:
            break
        bottom.append([shape.node((radius * math.cos(2 * math.pi * i / sides),
                                   radius * math.sin(2 * math.pi * i / sides),
                                   height), 'belly')
                       for i in range(sides)])
    floor = shape.node((0.0, 0.0, -WAIST / 2 + BODY_FLOOR), 'belly')
    for lower, upper in pairwise(bottom):
        bridge(shape, upper, lower, BODY_COLOR)
    for i in range(sides):
        shape.face([floor, bottom[-1][i], bottom[-1][(i + 1) % sides]],
                   BODY_COLOR)
    return waist, apex, floor, bottom


def _skin_waist(shape, waist, sides, rows, taken):
    """Close the waist band everywhere an arm does not grow out of it."""
    for i in range(sides):
        if i in taken:
            continue
        j_next = (i + 1) % sides
        for j in range(rows):
            shape.face([waist[i][j], waist[j_next][j],
                        waist[j_next][j + 1], waist[i][j + 1]], BODY_COLOR)


def _bulge(j, rows):
    """0 at the chords, 1 in the middle: how far a row leans when sheared."""
    return math.sin(math.pi * j / rows)


def girder(shape: Shape, left: Sequence[int], right: Sequence[int],
           path: Sequence[ArrayLike], half_height: float, half_thick: float,
           windows: Callable[[int, int], bool] | None, group: str,
           color: int, cap_end: bool = True, shear: float = 0.0,
           end_left: Sequence[int] | None = None,
           end_right: Sequence[int] | None = None
           ) -> tuple[list[list[int]], list[list[int]]]:
    """A hollow box swept along a path, with windows cut through it.

    One closed surface: two flat sides, a strip along the top and bottom,
    a cap at the far end, and a wall round every window joining one side to
    the other. Nothing overlaps anything and there is no face inside the
    solid — which is the whole point. Built as separate bars welded at
    their centers, as this was first, a truss is a heap of interpenetrating
    boxes with their end caps buried in the joints, and coloring it by
    displacement shows the insides through the skin.

    `left` and `right` are the rows of nodes the root starts from, so a
    girder can grow out of a surface that already exists rather than being
    parked against it; `end_left` and `end_right` do the same at the far
    end, so a member can grow *into* something as well as out of it.

    `shear` leans the interior node rows alternately along the run, which
    turns the openings from upright rectangles into a zigzag — the
    difference between a slotted plate and a truss.
    """
    rows = len(left) - 1
    lo = [list(left)]
    hi = [list(right)]
    # One reference for the whole run, chosen from where it ends up rather
    # than from each step. Picking it per station lets it swap axes the
    # moment a path tips past vertical, and the section spins a quarter
    # turn between one station and the next — which is what put a visible
    # twist in the legs, where they steepen towards the foot.
    overall = np.asarray(path[-1], dtype=float) - np.asarray(path[0],
                                                             dtype=float)
    overall = overall / np.linalg.norm(overall)
    # The cross-section direction comes from the panel this girder starts
    # on, not from a global axis. Taken from a fixed reference it is
    # whatever that axis gives, and any panel not lined up with it starts
    # rotated -- an arm at sixty degrees, a strap under a curved belly.
    # A sign flip cannot correct a rotation, which is why correcting the
    # handedness left the nacelle roots and the payload straps twisted.
    seam0 = (np.asarray(shape.position(right[0]), dtype=float)
             - np.asarray(shape.position(left[0]), dtype=float))
    seam0 = seam0 - float(np.dot(seam0, overall)) * overall
    if float(np.linalg.norm(seam0)) < 1e-9:
        seam0 = np.cross(overall, np.array([0.0, 0.0, 1.0]))
    seam0 = seam0 / np.linalg.norm(seam0)
    # Which way round the root panel already runs. The frame's own `across`
    # is whatever the cross product gives, and when it points from right to
    # left the first bay is built inside out: its quads cross over between
    # the panel and the next station, which is the twist that shows where
    # an arm meets the body. Ask the panel rather than assume.
    # And which way the rows run. Flipping `across` flips `up` with it, so
    # correcting left-for-right silently turns the rows upside down: the
    # girder counts j upward while the panel it started from counts it
    # downward, and every one of them is built inverted. That is what put
    # the legs on top of the arms -- row 0 of an arm is meant to be its
    # underside, because row 0 of the waist panel is.
    rise = (np.asarray(shape.position(left[-1]), dtype=float)
            - np.asarray(shape.position(left[0]), dtype=float))
    climbs = (1.0 if float(np.dot(np.cross(seam0, overall), rise)) >= 0.0
              else -1.0)
    for k in range(1, len(path)):
        center = np.asarray(path[k], dtype=float)
        tangent = np.asarray(path[k], dtype=float) - np.asarray(path[k - 1],
                                                                dtype=float)
        tangent = tangent / np.linalg.norm(tangent)
        across = seam0 - float(np.dot(seam0, tangent)) * tangent
        across = across / np.linalg.norm(across)
        up = np.cross(across, tangent)
        h, t = half_height[k], half_thick[k]
        if k == len(path) - 1 and end_left is not None:
            # Sort the panel it lands on into this girder's own frame
            # rather than trusting the order it was handed in. The two
            # panels a graft joins were built by different parts, for
            # different reasons, and their rows have no reason to run the
            # same way: taken as given, the last bay connects row j of one
            # to row j of the other and crosses over. That is the twist in
            # the camera's mounts.
            ends = list(end_left) + list(end_right)
            middle = np.mean([shape.position(n) for n in ends], axis=0)

            def placed(node: int, middle: np.ndarray = middle,
                       across: np.ndarray = across,
                       up: np.ndarray = up) -> tuple[float, float]:
                offset = np.asarray(shape.position(node),
                                    dtype=float) - middle
                return (float(np.dot(offset, across)),
                        float(np.dot(offset, up * climbs)))

            near = sorted(ends, key=lambda n: placed(n)[0])
            half = len(ends) // 2
            lo.append(sorted(near[:half], key=lambda n: placed(n)[1]))
            hi.append(sorted(near[half:], key=lambda n: placed(n)[1]))
            continue
        lean = shear * (1.0 if k % 2 else -1.0)
        lo.append([shape.node(center - across * t
                              + up * climbs * (-h + 2 * h * j / rows)
                              + tangent * lean * _bulge(j, rows), group)
                   for j in range(rows + 1)])
        hi.append([shape.node(center + across * t
                              + up * climbs * (-h + 2 * h * j / rows)
                              + tangent * lean * _bulge(j, rows), group)
                   for j in range(rows + 1)])

    # Flipping the frame reverses which way round a quad runs, so the
    # faces that follow it have to be reversed too or their normals point
    # into the solid — which renders as a dark band at the join and would
    # color by displacement from the inside.
    def face(nodes: Sequence[int], own_color: int | None = None) -> None:
        # Wound however it comes out; Shape.orient() settles the whole
        # surface afterwards, which is the only level at which the answer
        # is well defined.
        shape.face(nodes, color if own_color is None else own_color, group)

    holes = {(k, j) for k in range(len(path) - 1) for j in range(rows)
             if windows(k, j)}
    for k in range(len(path) - 1):
        for j in range(rows):
            if (k, j) in holes:
                continue
            face([lo[k][j], lo[k][j + 1], lo[k + 1][j + 1], lo[k + 1][j]])
            face([hi[k][j], hi[k + 1][j], hi[k + 1][j + 1], hi[k][j + 1]])
        # the narrow strips along the top and the bottom of the box
        face([lo[k][0], lo[k + 1][0], hi[k + 1][0], hi[k][0]])
        face([lo[k][rows], hi[k][rows], hi[k + 1][rows], lo[k + 1][rows]])
    # every window is walled from one side to the other, so the hole is a
    # hole through a solid rather than two gaps in two skins
    for k, j in holes:
        for a, b, neighbor in (((k, j), (k, j + 1), (k - 1, j)),
                                ((k + 1, j + 1), (k + 1, j), (k + 1, j)),
                                ((k + 1, j), (k, j), (k, j - 1)),
                                ((k, j + 1), (k + 1, j + 1), (k, j + 1))):
            if neighbor in holes:
                continue
            face([lo[a[0]][a[1]], lo[b[0]][b[1]],
                  hi[b[0]][b[1]], hi[a[0]][a[1]]])
    if cap_end:
        for j in range(rows):
            face([lo[-1][j], hi[-1][j], hi[-1][j + 1], lo[-1][j + 1]])
    return lo, hi


def _arm(shape, waist, sides, rows, index, radius, bays, name):
    """One arm: a hollow box girder growing straight out of the waist.

    Its root cross-section *is* one panel of the body's waist band — the
    same nodes, not a copy — so the arm and the body are one surface. It
    flares from the width of that panel down to a quarter-inch slab, deep
    at the root and shallow at the motor, with a window through every bay.
    """
    group = f'arm {name}'
    i, j_next = index, (index + 1) % sides
    theta = 2.0 * math.pi * (index + 0.5) / sides
    along = np.array([math.cos(theta), math.sin(theta), 0.0])

    root_thick = BODY_R * math.sin(math.pi / sides)
    path, half_h, half_t = [along * BODY_R * math.cos(math.pi / sides)], [], []
    half_h.append(WAIST / 2)
    half_t.append(root_thick)
    for k in range(1, bays + 1):
        t = k / bays
        r = BODY_R + (radius - BODY_R) * t
        rise = 0.006 * math.sin(t * math.pi) + 0.004 * t
        path.append(along * r + np.array([0.0, 0.0, rise]))
        # Blend out of the waist panel's own section rather than jumping to
        # the girder's in one bay. Jumping puts a steeply tilted facet
        # where the arm meets the body -- |nz| 0.37 against 0.12 along the
        # rest of the arm -- which reads as a fault in the mesh and is only
        # a section change too abrupt to shade like its neighbors.
        blend = min(1.0, t / 0.40)
        eased = blend * blend * (3.0 - 2.0 * blend)
        span = (ARM_DEEP + (ARM_SHALLOW - ARM_DEEP) * t) / 2
        half_h.append(WAIST / 2 * (1 - eased) + span * eased)
        half_t.append(root_thick * (1 - eased) + ARM_THICK / 2 * eased)

    def windows(k: int, j: int) -> bool:
        # Every other bay, so the openings stay separated by web rather
        # than running together into one long slot -- adjacent holes share
        # an edge, that edge gets no wall, and the two become one opening.
        # Tall, so the chords come out slender: the nodes stranded inside
        # a tall window are pruned afterwards rather than forbidden, which
        # is what kept this looking like a slotted plate.
        return k >= 1 and k % 2 == 1 and 1 <= j <= rows - 2

    lo, hi = girder(shape, waist[i], waist[j_next], path, half_h, half_t,
                    windows, group, ARM_COLOR, shear=ARM_SHEAR)

    # The motor tower rises from the last panel of the arm's top strip --
    # the same four nodes -- and the hub rises from the tower's own top.
    # Nothing is parked on anything: a cylinder standing on a capped box
    # end shares no node with it, which is four loose nacelles in the
    # check, and welding them by a center node only buries two caps in
    # each other.
    left = [lo[-2][rows], lo[-1][rows]]
    right = [hi[-2][rows], hi[-1][rows]]
    seat = (np.asarray(shape.position(left[0]))
            + np.asarray(shape.position(right[1]))) / 2.0
    tower = [seat, seat + np.array([0.0, 0.0, NACELLE_Z * 0.5]),
             seat + np.array([0.0, 0.0, NACELLE_Z])]
    span = float(np.linalg.norm(np.asarray(shape.position(left[0]))
                                - np.asarray(shape.position(left[1])))) / 2.0
    across = float(np.linalg.norm(np.asarray(shape.position(left[0]))
                                  - np.asarray(shape.position(right[0])))) / 2.0
    nlo, nhi = girder(shape, left, right, tower,
                      [span, span * 0.95, span * 0.9],
                      [across, NACELLE_R * 0.75, NACELLE_R * 0.85],
                      lambda k, j: False, f'nacelle {name}', NACELLE_COLOR,
                      cap_end=False)
    _propeller(shape, nlo, nhi, seat + np.array([0.0, 0.0, NACELLE_Z]),
               sides, name, handed=1.0 if 'left' in name else -1.0)
    return lo, hi, along


def _leg(shape, lo, hi, along, stations, drop, name):
    """A leg, grown out of the underside of the arm it hangs from.

    Its root panel is a quad the arm's bottom strip already owns — two
    stations of it — so the leg meets the airframe over an area at two
    points rather than being planted on it at one. It is the same hollow
    box the arm is, tapering and arching down to a foot.
    """
    group = f'leg {name}'
    a, b = 1, min(3, len(lo) - 1)
    left = [lo[a][0], lo[b][0]]
    right = [hi[a][0], hi[b][0]]
    root = (np.asarray(shape.position(left[0]))
            + np.asarray(shape.position(right[1]))) / 2.0
    foot = root + along * 0.070 + np.array([0.0, 0.0, -drop])

    path, half_h, half_t = [root], [0.0], [0.0]
    span = float(np.linalg.norm(
        np.asarray(shape.position(left[0]))
        - np.asarray(shape.position(left[1])))) / 2.0
    thick = float(np.linalg.norm(
        np.asarray(shape.position(left[0]))
        - np.asarray(shape.position(right[0])))) / 2.0
    for k in range(1, stations + 1):
        t = k / stations
        point = root + (foot - root) * t
        point = point + along * (0.022 * math.sin(t * math.pi))
        path.append(point)
        half_h.append(span * (0.60 - 0.28 * t))
        half_t.append(max(thick * (0.75 - 0.30 * t), 0.0035))
    return girder(shape, left, right, path, half_h, half_t,
                  lambda k, j: False, group, LEG_COLOR)


def _bridge_quads(shape, start, end, group, color, waist=0.85):
    """Join two four-node loops with a short tube that cannot cross itself.

    The two panels a graft joins were built by different parts, for
    different reasons, and there is no reason their corners come round in
    the same order. Rather than derive a frame and hope it lines up — which
    is what left the payload straps twisted through several attempts — this
    simply tries all eight ways of matching one loop to the other and takes
    the one that pairs each corner with the corner nearest it. A pairing
    that crosses is a longer pairing, so the shortest one does not.
    """
    here = [np.asarray(shape.position(n), dtype=float) for n in start]
    there = [np.asarray(shape.position(n), dtype=float) for n in end]

    def cost(order: Sequence[int]) -> float:
        return sum(float(np.dot(here[k] - there[order[k]],
                                here[k] - there[order[k]])) for k in range(4))

    orders = [tuple((r + k) % 4 for k in range(4)) for r in range(4)]
    orders += [tuple((r - k) % 4 for k in range(4)) for r in range(4)]
    best = min(orders, key=cost)
    matched = [end[i] for i in best]

    # A middle ring, necked in towards the line joining the two panels.
    # Measured from the corner's own offset rather than from that line it
    # lands outside the span altogether -- one corner came out 10 mm below
    # the lower panel, which folds the quad over on itself.
    # The two panels are usually tilted differently -- the pack's flank
    # slopes where the belly above it is nearly flat -- so one is rotated
    # relative to the other about the line joining them. Interpolating each
    # corner along a straight line then sweeps a ruled surface that folds
    # over on itself, and cutting it into more rings makes it worse rather
    # than better: three rings gave three folded quads where one gave two.
    #
    # Turning the section instead of dragging it is what fixes it. Each
    # corner is taken in polar coordinates about the axis, and the angle,
    # the radius and the distance along are interpolated separately, so the
    # rotation is spread evenly and no quad has to absorb it.
    begin = np.mean(here, axis=0)
    finish = np.mean([shape.position(n) for n in matched], axis=0)
    axis = finish - begin
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(e1)) < 1e-9:
        e1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    def polar(point: ArrayLike,
              origin: ArrayLike) -> tuple[float, float, float]:
        offset = np.asarray(point, dtype=float) - origin
        along = float(np.dot(offset, axis))
        flat = offset - along * axis
        return (math.atan2(float(np.dot(flat, e2)), float(np.dot(flat, e1))),
                float(np.linalg.norm(flat)), along)

    ends = [polar(here[k], begin) for k in range(4)]
    tips = [polar(shape.position(matched[k]), finish) for k in range(4)]

    rings, steps = [list(start)], 3
    for step in range(1, steps):
        f = step / steps
        row = []
        for k in range(4):
            a0, r0, s0 = ends[k]
            a1, r1, s1 = tips[k]
            turn = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            angle = a0 + turn * f
            radius = (r0 + (r1 - r0) * f) * waist
            along = s0 + (s1 - s0) * f
            center = begin + axis * (length * f)
            row.append(shape.node(center + radius * (math.cos(angle) * e1
                                                     + math.sin(angle) * e2)
                                  + along * axis, group))
        rings.append(row)
    rings.append(matched)

    for lower, upper in pairwise(rings):
        for k in range(4):
            t = (k + 1) % 4
            shape.face([lower[k], lower[t], upper[t], upper[k]], color)


def _graft(shape, belly, sides, left, right, group, color):
    """Grow a short member from one panel into the nearest belly panel."""
    here = np.mean([shape.position(n) for n in left + right], axis=0)

    def facing(loop: Sequence[int]) -> np.ndarray:
        pts = [np.asarray(shape.position(n), dtype=float) for n in loop]
        normal = np.cross(pts[1] - pts[0], pts[-1] - pts[0])
        length = float(np.linalg.norm(normal))
        return normal / length if length > 1e-12 else normal

    mine = facing([left[0], left[-1], right[-1], right[0]])
    best = None
    for r in range(len(belly) - 1):
        for a in range(sides):
            b = (a + 1) % sides
            quad = [belly[r][a], belly[r + 1][a], belly[r + 1][b], belly[r][b]]
            center = np.mean([shape.position(n) for n in quad], axis=0)
            span = float(np.linalg.norm(center - here))
            # Prefer a panel that faces this one, not merely the closest.
            # Two panels tilted differently cannot be joined by a tube that
            # does not twist somewhere, however the corners are paired --
            # the warp is a relative rotation, and shortening the run or
            # cutting it into more rings does not touch it.
            align = float(np.dot(mine, facing(quad)))
            score = span * (1.6 - 0.6 * abs(align))
            if best is None or score < best[0]:
                best = (score, [belly[r][a], belly[r + 1][a]],
                        [belly[r][b], belly[r + 1][b]])
    _span, end_left, end_right = best
    panel = [end_left[0], end_left[-1], end_right[-1], end_right[0]]
    start = [left[0], left[-1], right[-1], right[0]]

    # Cut a footprint the size of this strap into the belly panel and mesh
    # around it, rather than joining a 9 mm panel straight onto a 30 mm one
    # rotated ninety degrees from it. No pairing of corners rescues that --
    # the two are simply different shapes, and whatever tube spans them has
    # to fold somewhere. An inserted footprint is what a part printed in
    # one piece would have anyway.
    corners = [np.asarray(shape.position(n), dtype=float) for n in panel]
    middle = np.mean(corners, axis=0)
    seat = [shape.node(middle + (c - middle) * 0.42, group + ' seat')
            for c in corners]
    if shape.drop_face(panel):
        for k in range(4):
            t = (k + 1) % 4
            shape.face([panel[k], panel[t], seat[t], seat[k]], color)

    # order the seat to the strap it receives, so the bridge cannot cross
    def cost(turn: int) -> float:
        return sum(float(np.linalg.norm(
            np.asarray(shape.position(start[k]))
            - np.asarray(shape.position(seat[(turn + k) % 4]))))
            for k in range(4))

    turn = min(range(4), key=cost)
    _bridge_quads(shape, start, [seat[(turn + k) % 4] for k in range(4)],
                  group, color)


def _payload(shape, floor, belly, sides):
    """The battery under the belly and the camera on its mount."""
    pack = ((-0.052, -0.024), (-0.044, -0.030), (0.044, -0.030),
            (0.052, -0.024), (0.052, 0.024), (0.044, 0.030),
            (-0.044, 0.030), (-0.052, 0.024))
    # The first two stations are 3 mm apart and step the profile out from
    # 0.62 to 0.88, which leaves a nearly horizontal shoulder round the top
    # of the pack. That shoulder is what the straps leave from: a strap
    # rising out of the pack's vertical flank has to turn ninety degrees
    # immediately, and a tube leaving a panel parallel to its own direction
    # of travel folds over on itself however its corners are matched.
    top = BODY_FLOOR - WAIST / 2 - 0.004
    battery = sweep(
        shape,
        [(-0.014, 0.0, top), (-0.014, 0.0, top - 0.003),
         (-0.014, 0.0, top - 0.012), (-0.014, 0.0, top - 0.028),
         (-0.014, 0.0, top - 0.036)],
        [[(a * 0.62, b * 0.62) for a, b in pack],
         [(a * 0.88, b * 0.88) for a, b in pack],
         list(pack), list(pack),
         [(a * 0.88, b * 0.88) for a, b in pack]],
        'battery', BATTERY_COLOR, cap_start=True, cap_end=True)
    # Straps that grow out of the pack's own flank and into the belly, the
    # way the camera's mounts do. Struts run from a node to the middle of
    # the belly were separate welded prisms: joined, but pushed through
    # both surfaces rather than continuous with either.
    n_pack = len(battery[0])
    for k in (0, n_pack // 2):
        i, j = k % n_pack, (k + 1) % n_pack
        _graft(shape, belly, sides, [battery[0][i], battery[1][i]],
               [battery[0][j], battery[1][j]], 'battery', BATTERY_COLOR)

    # A camera: a squarish body running forward into a round lens barrel
    # and a hood at the end of it — one sweep, one surface. Built as a box
    # with a separate barrel pushed into its face, the two interpenetrate
    # and the barrel's back cap sits inside the body where it can only
    # ever be seen through the skin.
    nose = 0.062
    deck = BODY_FLOOR - WAIST / 2 - 0.028
    sides8 = max(8, (sides // 4) * 4)

    def squarish(half_w: float, half_h: float,
                 corner: float) -> list[tuple[float, float]]:
        out = []
        for k in range(sides8):
            a = 2.0 * math.pi * k / sides8
            c, s_ = math.cos(a), math.sin(a)
            # a superellipse: corner 1 is a circle, higher is squarer
            out.append((half_w * math.copysign(abs(c) ** (2 / corner), c),
                        half_h * math.copysign(abs(s_) ** (2 / corner), s_)))
        return out

    body = squarish(0.021, 0.019, 3.0)
    barrel = squarish(0.011, 0.011, 1.0)
    hood = squarish(0.014, 0.014, 1.0)
    camera = sweep(shape,
                   [(nose, 0.0, deck), (nose + 0.026, 0.0, deck),
                    (nose + 0.030, 0.0, deck), (nose + 0.044, 0.0, deck),
                    (nose + 0.050, 0.0, deck)],
                   [body, body, barrel, barrel, hood],
                   'camera', CAMERA_COLOR, cap_start=True, cap_end=True)
    # Each mount grows out of a panel of the camera's own upper flank and
    # into a panel the belly already has: four nodes shared at each end,
    # so it is continuous with both. A girder needs a *panel* to start
    # from, not an edge -- given one node a side it has no rows, builds no
    # faces at all, and the camera comes back a separate piece.
    n8 = len(camera[0])
    quarter = max(1, n8 // 4)
    for side_sign in (-1, 1):
        i = (quarter + side_sign * quarter) % n8
        j = (i + 1) % n8
        _graft(shape, belly, sides, [camera[0][i], camera[1][i]],
               [camera[0][j], camera[1][j]], 'camera mount', CAMERA_COLOR)
    return battery, camera


def build(sides: int = 12, arm_stations: int = 9, leg_stations: int = 5,
          body_rings: int = 3, total_mass: float = TOTAL_MASS) -> fem.Model:
    """The quadcopter, at whatever mesh density is asked for.

    `sides` is how many facets go round every swept tube and round the body,
    and is what most of the node count comes from. The defaults are about a
    thousand nodes, which the dense eigensolver clears in half a minute;
    tests build it coarser, since what they check is the airframe and not
    the mesh.
    """
    shape = draw(sides, arm_stations, leg_stations, body_rings)
    shape.prune()
    shape.orient()
    # The drawing becomes the structure here and only here: a member along
    # every edge of every face, and the mass shared over the nodes. There
    # is one description of this aircraft, and the physics is derived from
    # it rather than written beside it.
    # No groups passed: the geometry's own blocks say which part each face
    # belongs to, so a model built here and a model rebuilt from a saved
    # file are the same model. They were not while the parts arrived
    # alongside the drawing — 16 blade members read as frame.
    return fem.Model.from_geometry(
        shape.geometry(), FRAME, MEMBER, total_mass=total_mass,
        name='quadcopter', sections={'prop': BLADE})


def draw(sides: int = 12, arm_stations: int = 9, leg_stations: int = 5,
         body_rings: int = 3) -> Shape:
    """The airframe as a drawing — nodes and faces, no structure at all."""
    shape = Shape()
    rows = 5
    waist, _apex, floor, belly = _body(shape, sides, body_rings, rows)
    # each arm takes one panel of the waist band as its root, so the panel
    # itself is not skinned over -- the arm is what closes it
    taken = {round(sides * angle / 360.0) % sides for _n, angle, _r in ARMS}
    _skin_waist(shape, waist, sides, rows, taken)
    for (name, angle, radius), index in zip(ARMS, sorted(taken)):
        lo, hi, along = _arm(shape, waist, sides, rows, index, radius,
                             arm_stations, name)
        _leg(shape, lo, hi, along, leg_stations,
             LEG_DROP * (0.85 if 'front' in name else 1.0), name)
    _payload(shape, floor, belly, sides)
    return shape


#: Which drawn part each group belongs to, for reading a mode. The shell
#: pieces answer as one 'body'; everything else answers for itself.
#: Leaving a prefix out of this does not fail, it quietly lands in 'body' —
#: which is how 620 propeller nodes once reported themselves as body
#: motion, and made the first dozen modes look like a rigid airframe.
PARTS = {'canopy': 'body', 'belly': 'body', 'waist': 'body',
         'truss': 'body', 'arm': 'arm', 'leg': 'leg', 'nacelle': 'nacelle',
         'prop': 'prop', 'battery': 'battery', 'camera': 'camera'}


def part_of(model: fem.Model, node: int) -> str:
    """Which part of the airframe a node belongs to."""
    group = model.group(node)
    head = group.split()[0] if group else ''
    return PARTS.get(head, 'body')


def instrumented(model: fem.Model) -> dict[str, list[int]]:
    """The nodes a modal survey of this airframe would put sensors on.

    Found from the model rather than written down, because writing them
    down has been wrong twice: the numbering moves whenever the mesh does,
    and a generator holding stale ids produces a test of nowhere.
    """
    def nearest(candidates: Sequence[int], point: ArrayLike) -> int:
        point = np.asarray(point, dtype=float)
        return min(candidates, key=lambda n: float(np.linalg.norm(
            np.asarray(model.position(n))[:len(point)] - point)))

    groups: dict[str, list[int]] = {}
    for node in model.node_ids:
        groups.setdefault(model.group(node), []).append(node)

    found: dict[str, list[int]] = {'motors': [], 'arms': [], 'feet': []}
    for name, angle, radius in ARMS:
        theta = math.radians(angle)
        out = np.array([math.cos(theta), math.sin(theta)])
        nacelle = groups.get(f'nacelle {name}', [])
        if nacelle:
            found['motors'].append(max(nacelle,
                                       key=lambda n: model.position(n)[2]))
        arm = groups.get(f'arm {name}', [])
        if arm:
            found['arms'].append(nearest(arm, out * radius * 0.6))
        leg = groups.get(f'leg {name}', [])
        if leg:
            found['feet'].append(min(leg, key=lambda n: model.position(n)[2]))

    # The shell, not the canopy block: the canopy is the apex cap alone —
    # eight triangles — and the ring at 0.9 of the body radius is the
    # waist below it. Asking one block for both put all four body sensors
    # and the center one on the same node.
    body = [n for n in model.node_ids if part_of(model, n) == 'body']
    found['body'] = [nearest(body, (BODY_R * 0.9 * math.cos(math.radians(a)),
                                    BODY_R * 0.9 * math.sin(math.radians(a))))
                     for a in (0, 90, 180, 270)]
    found['center'] = [max(body, key=lambda n: model.position(n)[2])]
    found['payload'] = [
        min(groups.get('battery', [0]), key=lambda n: model.position(n)[2]),
        min(groups.get('camera', [0]), key=lambda n: model.position(n)[0]),
    ]
    return found


def describe(model: fem.Model, maximum_frequency: float = 500.0,
             damping: float = 0.01) -> None:
    """What it weighs and where its modes are, by what moves in each."""
    shapes = model.eigensolution(maximum_frequency=maximum_frequency,
                                 damping=damping)
    mass, _ = model.matrices()
    ids = model.node_ids
    order = ['body', 'arm', 'nacelle', 'leg', 'battery', 'camera']

    print(f'{model.num_nodes} nodes, {model.num_dof} DOF, '
          f'{len(model.beams)} members, {len(model.faces)} faces')
    print(f'{model.total_mass * 1000:.0f} g over '
          f'{len(ids)} nodes')
    print()
    print(f'{"f (Hz)":>9}  ' + '  '.join(f'{f:>8s}' for f in order))
    for frequency, shape in zip(shapes.frequency, shapes.shape_matrix):
        if frequency == 0.0:
            continue
        weighted = mass @ shape[:model.num_dof]
        share: dict[str, float] = {}
        for i, node in enumerate(ids):
            key = part_of(model, node)
            share[key] = share.get(key, 0.0) + float(
                np.dot(shape[6 * i:6 * i + 6], weighted[6 * i:6 * i + 6]))
        total = sum(share.values())
        print(f'{frequency:9.2f}  '
              + '  '.join(f'{100 * share.get(k, 0.0) / total:7.1f}%'
                          for k in order)
              + f'   {max(share, key=share.get)}')


if __name__ == '__main__':
    describe(build())
