"""The BARC, built from planes, and again from bricks.

The BARC — Box Assembly with Removable Component — is a small bolted
aluminum structure the structural dynamics community shares as a
common test article: a square box tube with its top wall slotted in two,
and on it the "Bench", two channels standing on the two halves and a
flat bar bolted across their tops. Its solid model, test data and
finite element models are shared on the SEM Dynamic Substructuring
Focus Group wiki, https://wiki.sem.org/wiki/BARC.

Here it is built the way a person would build a simple plate model with
this package (Brandon, 2026-09-26): each part as the planes it is made
of, meshed at mid-thickness (`visualdynamics.mesh`), each part an element group
given its material and thickness, and the bolts as rigid, massless
links. Every dimension below comes from the shared solid model (inches):

    from visualdynamics.demo import barc

    model = barc.build()
    shapes = model.eigensolution(maximum_frequency=2000)

**The joints are the model.** The parts meet through bolts, and their
mid-surfaces do not touch — a channel's foot sits 0.1875 in above the
box wall's mid-surface — so what joins them is a modeling choice, and it
decides the Bench's modes. The choices were checked against the finite
element models shared on the wiki while this was built: a single rigid
link at each bolt came out soft, and softer at every mesh refinement (a
point tie on a plate is a local singularity, not a joint), and each foot
tied over its whole area came out stiff. What is built ties each bolt
over the elements it clamps (`bolt_patch`), to the part below with
`mesh.tie` — what the app's Tie does with the same elements picked
(2026-09-26) — so the model a person builds in the app by the
documentation's steps is this one, node for node and link for link.
Four bolts join each channel to the box, each over the elements its
washer covers, and four join the beam to each channel, each over the
element it passes through (Brandon, 2026-10-11; the beam had one bolt
at each end).

**And from bricks** (2026-10-07): the same structure as solid
eight-node bricks (`mesh.block`), the higher-fidelity model, and the
removable component on its own — the two channels and the beam, the
part the BARC is named for (`solid_geometry`, `solid_project`). Bricks
touch where the parts do, so their bolts are built the other way
(Brandon, 2026-10-11): each part meshed apart (`mesh.assemble` with
`apart`), its nodes coincident with the next part's where the two
touch and joined to none of them, and each bolt made by merging the
places it clamps (`bolt_places`, Merge Coincident Nodes with the rest
left out) — at a foot the corners of the brick under the bolt and the
eight around it, the washer's area, and at the beam the four corners
of the brick the bolt passes through. The bolts' masses are point
masses, an element group of point elements given a mass
(`fem.GroupProperties(mass=...)`).

Run it directly to print the models and their first modes:

    python3 -m visualdynamics.demo.barc
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visualdynamics import View, fem, mesh

#: element size aimed at, inches: 0.125 converges the first ten modes to
#: about a percent (0.0625 moves them no more) and solves in a few
#: seconds — 5,500 nodes, the sparse solver
SIZE = 0.125

#: the box's wall mid-surface: a 6 in square tube, 0.25 in wall
BOX_MID = 2.875
BOX_DEPTH = 3.0
#: the top wall's slot: 0.5 in wide, centered, the full depth
SLOT_HALF = 0.25
#: the channels' flanges (feet and tops) and the beam, mid-surfaces
FOOT_Y, TOP_Y, BEAM_Y = 3.0625, 4.9375, 5.0625
THICKNESS = {'box': 0.25, 'right channel': 0.125, 'left channel': 0.125,
             'beam': 0.125}                  #: inches
MATERIAL = fem.material('6061-T6')

#: where the bolts pass through a part, (x, z): four at each channel's
#: foot and, above them, four through the beam into each channel's top
#: (Brandon, 2026-10-11: the beam was held by one bolt at each end)
BOLT_AXES = [(x, z) for x in (-2.1875, -1.8125, 1.8125, 2.1875)
             for z in (1.3125, 1.6875)]
#: the bolts: (x, z) on the part, the two mid-surfaces they join, and
#: the kind, a foot's or the beam's
BOLTS = ([(axis, BOX_MID, FOOT_Y, 'foot') for axis in BOLT_AXES]
         + [(axis, TOP_Y, BEAM_Y, 'top') for axis in BOLT_AXES])
#: how many elements across the patch a bolt clamps, centered on the
#: element it passes through: at a foot the three its #8 screw's 0.375
#: in washer covers, and at the beam the one element — its four corners,
#: as Brandon proposed (2026-10-11). Measured against the builds checked
#: against the wiki's models, which held the beam by one bolt at each
#: end: the planes within 0.2 % over the first ten elastic modes, MAC
#: 0.999 and up, where the washer's three at the beam too moved modes
#: six, seven, nine and ten (MAC 0.46 to 0.84); the bricks within 3.5 %,
#: MAC 0.97 and up, where four places at a foot read up to 10 % soft
#: and lost the ninth mode's shape (MAC 0.65).
CLAMPED = {'foot': 3, 'top': 1}

#: where the BARC is shared: its solid model, test data and models
WIKI = 'https://wiki.sem.org/wiki/BARC'
INCH = 0.0254


def _planes(size: float) -> list:
    x, y, z = np.eye(3)
    t = BOX_MID
    plate = (lambda corner, a, b, name:
             mesh.plane(corner, a, b, size, name, unit='in'))
    return [
        plate((-t, -t, 0), 2 * t * x, BOX_DEPTH * z, 'box'),        # bottom
        plate((t, -t, 0), 2 * t * y, BOX_DEPTH * z, 'box'),         # right
        plate((-t, -t, 0), 2 * t * y, BOX_DEPTH * z, 'box'),        # left
        plate((-t, t, 0), (t - SLOT_HALF) * x, BOX_DEPTH * z, 'box'),
        plate((SLOT_HALF, t, 0), (t - SLOT_HALF) * x, BOX_DEPTH * z, 'box'),
        # the right channel: web outboard at x = 2.4375, flanges inboard
        plate((1.5, FOOT_Y, 1), 0.9375 * x, z, 'right channel'),
        plate((1.5, TOP_Y, 1), 0.9375 * x, z, 'right channel'),
        plate((2.4375, FOOT_Y, 1), (TOP_Y - FOOT_Y) * y, z,
              'right channel'),
        # the left channel turned a quarter turn: web at z = 1.0625
        plate((-2.5, FOOT_Y, 1.0625), x, 0.9375 * z, 'left channel'),
        plate((-2.5, TOP_Y, 1.0625), x, 0.9375 * z, 'left channel'),
        plate((-2.5, FOOT_Y, 1.0625), x, (TOP_Y - FOOT_Y) * y,
              'left channel'),
        plate((-2.5, BEAM_Y, 1), 5 * x, z, 'beam'),
    ]


#: how the BARC opens in 3-D. It is built in its solid model's frame,
#: where y is up, and every 3-D view used to open z-up and draw it lying
#: on its side; the website figure and the downloads tile turned the
#: nodes to stand it up. A geometry says how it is seen now (2026-09-27),
#: so the model keeps its frame and opens upright everywhere: from the
#: front, above and to the right, y up.
VIEW = View(eye=(1.0, 1.0, -1.0), up=(0.0, 1.0, 0.0))


def geometry(size: float = SIZE) -> Any:
    """The BARC as a geometry of plates and rigid links, every element group
    given what it is made of — ready for `fem.Model.from_geometry`, or
    for the app's Solve Modes.

    Parameters
    ----------
    size : float
        The element size aimed at, in inches.

    Returns
    -------
    Geometry
        Element groups 'box', 'right channel', 'left channel', 'beam' (6061-T6
        plates) and 'bolts' (rigid links over the elements each bolt
        clamps, `bolt_patch`), opening on `VIEW`.
    """
    whole = mesh.assemble(*_planes(size))
    whole.view = VIEW
    whole.group_properties = {
        int(group): fem.GroupProperties(
            MATERIAL, THICKNESS[whole.group_name[i]] * INCH)
        for i, group in enumerate(whole.group_id)}
    for bolt in BOLTS:
        patch, below = bolt_patch(whole, bolt)
        mesh.tie(whole, patch, below, group='bolts')
    return whole


def bolt_patch(geometry: Any, bolt: tuple) -> tuple[list[int], str]:
    """The elements a bolt clamps, and the element group it bolts them to:
    the `CLAMPED` elements across of the part on top, centered on the
    element the bolt passes through — at a foot that element and the
    eight around it, the washer's area, and at the beam the element
    alone. Whole elements rather than the washer's circle because they
    are what a person picks: whole rows, countable in the view.

    Parameters
    ----------
    geometry : Geometry
        The BARC's planes, assembled.
    bolt : tuple
        One of `BOLTS`.

    Returns
    -------
    (list of int, str)
        The element ids, and the name of the element group below.
    """
    axis, _lower, upper, kind = bolt
    side = 'left channel' if axis[0] < 0 else 'right channel'
    top, below = ('beam', side) if kind == 'top' else (side, 'box')
    xyz = geometry.node_xyz / INCH
    on_face = {}
    for element in geometry.elements_in(top):
        row = int(np.flatnonzero(geometry.elem_id == element)[0])
        at = xyz[geometry.node_index(geometry.elem_conn[row])]
        if np.all(np.abs(at[:, 1] - upper) < 1e-6):
            on_face[element] = at
    return _clamped(on_face, axis, CLAMPED[kind]), below


def _clamped(corners: dict, axis: tuple, across: int) -> list[int]:
    """Of elements given as {id: their corners' (x, y, z), inches} on
    one face, the `across` × `across` centered on the one whose center
    is nearest the bolt's (x, z) — the rule both models' bolts are
    picked by."""
    centers = {e: at.mean(axis=0) for e, at in corners.items()}
    under = min(centers, key=lambda e: np.hypot(centers[e][0] - axis[0],
                                                centers[e][2] - axis[1]))
    middle = centers[under]
    # half a pitch past the last element counted, along each edge
    reach = (across // 2 + 0.5) * np.ptp(corners[under], axis=0)
    return [e for e, center in centers.items()
            if abs(center[0] - middle[0]) < reach[0]
            and abs(center[2] - middle[2]) < reach[2]]


def build(size: float = SIZE) -> fem.Model:
    """The BARC's finite element model (`geometry` built).

    Parameters
    ----------
    size : float
        The element size aimed at, in inches.

    Returns
    -------
    fem.Model
    """
    return fem.Model.from_geometry(geometry(size), name='BARC')


#: how far the solved example project solves, Hz
SOLVE_TO = 2000.0


def project(size: float = SIZE, solved: bool = False) -> Any:
    """A project to open in the app: the BARC's geometry, its element groups
    given their properties — Solve Modes on it gives its modes.

        barc.project().save('barc.vdyn')

    With `solved`, the modes are solved to `SOLVE_TO` already: the
    example project the downloads page offers.

    Parameters
    ----------
    size : float
        The element size aimed at, in inches.
    solved : bool, default False
        Solve the modes as well.

    Returns
    -------
    Project
    """
    from visualdynamics.project import Project

    out = Project('BARC')
    out.add('BARC', geometry(size))
    if solved:
        out.solve_modes('BARC', maximum_frequency=SOLVE_TO)
    return out


# ---- the BARC from bricks (2026-10-07) ---------------------------------

#: the models `solid_geometry` builds: the whole BARC, and the removable
#: component — the two channels and the beam — on its own
PARTS = ('BARC', 'removable component')
#: each bolt's mass, pounds, as the shared model lumps them: a #8 screw
#: at a foot, and at the beam the shared model's one 1/4 in screw per
#: channel, 0.0125 lb, shared by the four bolts here — the removable
#: component's modes read up to 6 % low with a #8 screw's mass at each
BOLT_MASS = {'foot': 0.00744, 'top': 0.0125 / 4}
#: where the parts touch, y in inches: a channel's foot on the box's top
#: wall, the beam on a channel's top
JOINT_Y = {'foot': 3.0, 'top': 5.0}
#: the face each bolt's head sits on, y in inches: a foot's top and the
#: beam's
HEAD_Y = {'foot': 3.125, 'top': 5.125}
#: the bolts: (x, z) and the kind, foot or top
SOLID_BOLTS = ([(axis, 'foot') for axis in BOLT_AXES]
               + [(axis, 'top') for axis in BOLT_AXES])
POUND = 0.45359237
#: how far the solved removable component solves, Hz: its seventh
#: elastic mode is just past `SOLVE_TO`
PART_SOLVE_TO = 2600.0


def _boxes(part: str) -> list:
    """The parts as boxes, inches: (corner, extent, element group)."""
    t = 0.125
    box = [
        ((-3, -3, 0), (6, 0.25, BOX_DEPTH)),                    # bottom
        ((2.75, -2.75, 0), (0.25, 5.5, BOX_DEPTH)),             # right
        ((-3, -2.75, 0), (0.25, 5.5, BOX_DEPTH)),               # left
        ((SLOT_HALF, 2.75, 0), (3 - SLOT_HALF, 0.25, BOX_DEPTH)),
        ((-3, 2.75, 0), (3 - SLOT_HALF, 0.25, BOX_DEPTH)),
    ]
    component = [
        # the right channel: foot and top flange inboard, web outboard
        ((1.5, 3.0, 1), (1, t, 1), 'right channel'),
        ((1.5, 4.875, 1), (1, t, 1), 'right channel'),
        ((2.375, 3.125, 1), (t, 1.75, 1), 'right channel'),
        # the left channel turned a quarter turn: its web at the back
        ((-2.5, 3.0, 1), (1, t, 1), 'left channel'),
        ((-2.5, 4.875, 1), (1, t, 1), 'left channel'),
        ((-2.5, 3.125, 1), (1, 1.75, t), 'left channel'),
        ((-2.5, 5.0, 1), (5, t, 1), 'beam'),
    ]
    whole = [(*b, 'box') for b in box] + component
    return whole if part == 'BARC' else component


def _bricks(size: float, part: str) -> list:
    """The parts as boxes meshed into bricks, each in the element group of
    its own name.

    Every box is cut at every other box's faces before it is meshed, so
    two boxes that meet share their face's nodes at any size. Meshed
    whole, a channel's 0.125 in web met its flange where the flange,
    divided at a quarter inch, had no line of nodes: the web hung from
    one edge, and the BARC read 76 Hz for 192 (2026-10-07). At an
    eighth of an inch every cut falls on a line of nodes already, and
    the mesh is the same as uncut.
    """
    import itertools

    boxes = _boxes(part)
    cuts = [sorted({round(corner[k] + d, 9) for corner, extent, _name in boxes
                    for d in (0.0, extent[k])}) for k in range(3)]
    out = []
    for corner, extent, name in boxes:
        spans = []
        for k in range(3):
            low, high = corner[k], corner[k] + extent[k]
            at = [low, *(v for v in cuts[k] if low + 1e-9 < v < high - 1e-9),
                  high]
            spans.append(list(itertools.pairwise(at)))
        for (x0, x1), (y0, y1), (z0, z1) in itertools.product(*spans):
            out.append(mesh.block((x0, y0, z0), (x1 - x0, 0, 0),
                                  (0, y1 - y0, 0), (0, 0, z1 - z0), size,
                                  name, unit='in'))
    return out


def solid_geometry(part: str = 'BARC', size: float = SIZE) -> Any:
    """The BARC, or its removable component, as a geometry of bricks
    and point masses, every element group given what it is made of — ready
    for `fem.Model.from_geometry`, or for the app's Solve Modes.

    Parameters
    ----------
    part : str
        One of `PARTS`: 'BARC', the whole assembly, or 'removable
        component', the two channels and the beam.
    size : float
        The element size aimed at, in inches.

    Returns
    -------
    Geometry
        Element groups 'box' (the assembly only), 'right channel', 'left
        channel', 'beam' (6061-T6 bricks, joined only at the bolts,
        `bolt_places`) and the bolts' masses, 'foot bolts' (the assembly
        only) and 'top bolts', opening on `VIEW`.
    """
    if part not in PARTS:
        raise ValueError(f'{part!r} is not one of {PARTS}')
    whole = mesh.assemble(*_bricks(size, part), apart=True)
    whole.view = VIEW
    whole.group_properties = {int(group): fem.GroupProperties(MATERIAL)
                              for group in whole.group_id}
    ids = dict(zip(whole.group_name, whole.group_id.tolist()))
    kinds = ('foot', 'top') if part == 'BARC' else ('top',)
    for kind in kinds:
        for side in ('right channel', 'left channel'):
            below, above = ('box', side) if kind == 'foot' else (side, 'beam')
            groups = [ids[below], ids[above]]
            places = {node for (bx, bz), k in SOLID_BOLTS if k == kind
                      and (bx > 0) == (side == 'right channel')
                      for node in bolt_places(whole, (bx, bz), kind)}
            # the pane's own tolerance, a quarter of the finest spacing;
            # every coincidence not under a bolt left out, as unpicking it
            tolerance = 0.25 * whole.node_spacing(groups)
            left_out = [node for spot in whole.coincident_spots(
                tolerance, groups) if places.isdisjoint(spot) for node in spot]
            whole.merge_coincident_nodes(tolerance, groups, left_out)
    # each bolt's mass at the node of its head face nearest its axis
    xyz = whole.node_xyz / INCH
    for kind in kinds:
        nodes = []
        for (bx, bz), k in SOLID_BOLTS:
            if k != kind:
                continue
            on_face = np.flatnonzero(np.abs(xyz[:, 1] - HEAD_Y[kind]) < 1e-6)
            nearest = on_face[np.argmin(np.hypot(xyz[on_face, 0] - bx,
                                                 xyz[on_face, 2] - bz))]
            nodes.append([int(whole.node_id[nearest])])
        group = whole.add_group(f'{kind} bolts')
        whole.add_elements(nodes, [161] * len(nodes), [group] * len(nodes))
        whole.group_properties[group] = fem.GroupProperties(
            mass=BOLT_MASS[kind] * POUND)
    return whole


def bolt_places(geometry: Any, axis: tuple, kind: str) -> set[int]:
    """The places a bolt clamps, as the ids of the nodes of the part on
    top that a merge joins there: the corners, on the face where the
    parts touch, of the bricks it clamps (`CLAMPED`, the rule the planes'
    bolts are picked by) — the washer's 4 × 4 places at a foot, the four
    corners of the brick the bolt passes through at the beam. What the
    app's merge pane marks, and a person clicks.

    Parameters
    ----------
    geometry : Geometry
        The BARC's bricks, assembled apart.
    axis : tuple
        The bolt's (x, z), inches.
    kind : str
        'foot' or 'top'.

    Returns
    -------
    set of int
        Node ids of the part on top.
    """
    side = 'right channel' if axis[0] > 0 else 'left channel'
    top = side if kind == 'foot' else 'beam'
    xyz = geometry.node_xyz / INCH
    faces = {}
    for element in geometry.elements_in(top):
        row = int(np.flatnonzero(geometry.elem_id == element)[0])
        conn = geometry.elem_conn[row]
        touching = np.abs(xyz[geometry.node_index(conn), 1]
                          - JOINT_Y[kind]) < 1e-6
        if touching.sum() == 4:
            faces[element] = conn[touching]
    patch = _clamped({e: xyz[geometry.node_index(nodes)]
                      for e, nodes in faces.items()}, axis, CLAMPED[kind])
    return {int(node) for e in patch for node in faces[e]}


def solid_build(part: str = 'BARC', size: float = SIZE) -> fem.Model:
    """The BARC's, or its removable component's, model of bricks
    (`solid_geometry` built).

    Parameters
    ----------
    part : str
        One of `PARTS`.
    size : float
        The element size aimed at, in inches.

    Returns
    -------
    fem.Model
    """
    return fem.Model.from_geometry(solid_geometry(part, size),
                                   name=PART_NAMES[part])


#: what each model is called in a project
PART_NAMES = {'BARC': 'BARC', 'removable component': 'Removable Component'}


def solid_project(size: float = SIZE, solved: bool = False) -> Any:
    """A project to open in the app: the BARC and its removable
    component, both of bricks — Solve Modes on either gives its modes.

        barc.solid_project().save('barc-bricks.vdyn')

    With `solved`, the BARC's modes are solved to `SOLVE_TO` and the
    removable component's to `PART_SOLVE_TO`: the example project the
    downloads page offers.

    Parameters
    ----------
    size : float
        The element size aimed at, in inches.
    solved : bool, default False
        Solve the modes as well.

    Returns
    -------
    Project
    """
    from visualdynamics.project import Project

    out = Project('BARC in Bricks')
    for part, top in zip(PARTS, (SOLVE_TO, PART_SOLVE_TO)):
        out.add(PART_NAMES[part], solid_geometry(part, size))
        if solved:
            out.solve_modes(PART_NAMES[part], maximum_frequency=top)
    return out


def describe(size: float = SIZE, modes: int = 10) -> None:
    """Print the models and their first elastic modes."""
    model = build(size)
    shapes = model.eigensolution(maximum_frequency=SOLVE_TO)
    print(f'BARC from planes: {model.num_nodes} nodes, {len(model.plates)} '
          f'plates, {len(model.rigid_links)} rigid links, '
          f'{model.structural_mass:.3f} kg')
    elastic = [f for f in shapes.frequency if f > 1.0][:modes]
    print('  elastic modes (Hz): ' + ', '.join(f'{f:.1f}' for f in elastic))
    for part, top in zip(PARTS, (SOLVE_TO, PART_SOLVE_TO)):
        model = solid_build(part, size)
        shapes = model.eigensolution(maximum_frequency=top)
        print(f'{PART_NAMES[part]} from bricks: {model.num_nodes} nodes, '
              f'{len(model.solids)} bricks, {model.total_mass:.3f} kg with '
              'the bolts')
        elastic = [f for f in shapes.frequency if f > 1.0][:modes]
        print('  elastic modes (Hz): '
              + ', '.join(f'{f:.1f}' for f in elastic))


if __name__ == '__main__':
    describe()
