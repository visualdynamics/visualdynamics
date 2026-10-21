"""The BARC, built from plates, and again from bricks.

The BARC (Box Assembly with Removable Component) is a small bolted
aluminum structure the structural dynamics community shares as a common
test article: a square box tube with its top wall slotted in two, and on
it the "Bench", two channels standing on the two halves and a flat bar,
the beam, bolted across their tops. Its solid model, test data and
finite element models are shared on the SEM Dynamic Substructuring
Focus Group wiki, https://wiki.sem.org/wiki/BARC.

Here it is built the way a person would build a simple plate model with
this package: each part as the planes it is made of, meshed at
mid-thickness (`visualdynamics.mesh`), each part an element group given
its material and thickness, and the bolts as rigid, massless links.
Every dimension below comes from the shared solid model, in inches:

    from visualdynamics.demo import barc

    model = barc.build()
    shapes = model.eigensolution(maximum_frequency=2000)

The joints decide the Bench's modes. The parts meet through bolts, and
their mid-surfaces do not touch (a channel's foot is 0.1875 in above the
box wall's mid-surface), so what joins them is a modeling choice. The
choices were checked against the finite element models shared on the
wiki: a single rigid link at each bolt came out soft, and softer at
every mesh refinement, because a point tie on a plate is a local
singularity and not a joint; each foot tied over its whole area came
out stiff. What is built ties each bolt over the elements it clamps
(`bolt_patch`) to the part below with `mesh.tie`, which is what the
app's Tie does with the same elements picked. So the model a person
builds in the app by the documentation's steps is this one, node for
node and link for link. Four bolts join each channel to the box, each
over the elements its washer covers, and four join the beam to each
channel, each over the element it passes through.

The same structure is also built from solid eight-node bricks, the
higher-fidelity model, along with the removable component on its own:
the two channels and the beam, the part the BARC is named for
(`solid_geometry`, `solid_project`). Its parts are solids, each made of
as few blocks as it can be (the box a block less its inside and its
slot, a channel a block less the inside of its C, the beam a block;
`BOX_SOLIDS`, `CHANNEL_SOLIDS`, `BEAM_SOLID`), and meshed by Mesh
Solids, following the documentation's steps. Bricks touch where the
parts do, so their bolts are made differently. The parts are meshed on
one grid, each with its own nodes (`parts_apart`), coincident with the
next part's nodes where the two touch and joined to none of them. Each
bolt is then a mated set, made by Mate Surfaces with the found patch
trimmed to the places the bolt clamps (`bolt_places`): at a foot, the
corners of the brick under the bolt and the eight around it, the
washer's area, and at the beam, the four corners of the brick the bolt
passes through. Each node there is tied rigidly to the face beneath it,
and the rest of the contact is left free. The
bolts' masses are point masses, an element group of point elements
given a mass (`fem.GroupProperties(mass=...)`).

Run it directly to print the models and their first modes:

    python3 -m visualdynamics.demo.barc
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visualdynamics import View, fem, mesh

# The plate model follows the steps a person takes in the app (Brandon,
# 2026-09-26), and the bolts are tied over patches the way the app's Tie
# does it (2026-09-26). The beam had one bolt at each end until Brandon,
# 2026-10-11, made it four per channel. The brick model was added on
# 2026-10-07; since 2026-10-11 its parts are solids, meshed by Mesh
# Solids, node for node the boxes they replaced. Its bolts were made by
# merging nodes from 2026-10-11 until Brandon, 2026-10-20, asked for the
# examples to be rebuilt with Mate Surfaces; the frequencies did not
# move, since a rigid tie on coincident nodes is the merge.

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
    given its material and thickness, ready for `fem.Model.from_geometry`
    or for the app's Solve Modes.

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
    """The elements a bolt clamps, and the element group it bolts them to.

    The clamped elements are a square `CLAMPED` elements across, on the
    part on top, centered on the element the bolt passes through: at a
    foot that element and the eight around it, the washer's area, and at
    the beam the element alone. Whole elements are used, not the washer's
    circle, because whole elements are what a person picks in the view,
    in rows that can be counted.

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
    """The BARC's finite element model of plates, built from `geometry`.

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
    """A project to open in the app: the BARC's plate geometry, its
    element groups given their properties, so Solve Modes on it gives its
    modes.

        barc.project().save('barc.vdyn')

    With `solved`, the modes are already solved to `SOLVE_TO` (2000 Hz).
    That is the example project the downloads page offers.

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


#: the BARC's parts as the solid blocks a person adds, inches: (center,
#: widths, name) — the fewest blocks and operations each part can be
#: (Brandon, 2026-10-11). The box is a block less its inside and the
#: slot in its top; a channel is a block less the inside of its C; the
#: beam is one. A cut reaches past the part's faces so it goes clean
#: through, its far faces on grid lines the mesh has anyway. These are
#: the parts' true outlines, not their mid-surfaces: a foot's underside
#: is the box's top face at y = 3, the beam's the channels' tops at y = 5.
BOX_SOLIDS = [((0.0, 0.0, 1.5), (6.0, 6.0, 3.0), 'box'),
              ((0.0, 0.0, 1.5), (5.5, 5.5, 4.0), ''),
              ((0.0, 2.8125, 1.5), (2 * SLOT_HALF, 0.625, 4.0), '')]
CHANNEL_SOLIDS = {
    # the right channel: its web outboard, open inboard
    'right channel': [((2.0, 4.0, 1.5), (1.0, 2.0, 1.0)),
                      ((1.8125, 4.0, 1.5), (1.125, 1.75, 1.5))],
    # the left channel turned a quarter turn: its web at the back, open
    # to the front
    'left channel': [((-2.0, 4.0, 1.5), (1.0, 2.0, 1.0)),
                     ((-2.0, 4.0, 1.6875), (1.5, 1.75, 1.125))]}
BEAM_SOLID = ((0.0, 5.0625, 1.5), (5.0, 0.125, 1.0))


def _parts(project: Any, name: str, part: str) -> list[int]:
    """The parts as solids, the way a person makes them in the app (Add
    Solid Block, then Subtract), in the order they are meshed: the box,
    the right channel, the left channel, the beam."""
    parts = []
    if part == 'BARC':
        box = [project.add_solid_block(name, center, widths, name=label, unit='in')
               for center, widths, label in BOX_SOLIDS]
        parts.append(project.combine_solids(name, 'subtract', box))
    for side, blocks in CHANNEL_SOLIDS.items():
        made = [project.add_solid_block(name, center, widths,
                                        name=side if k == 0 else '', unit='in')
                for k, (center, widths) in enumerate(blocks)]
        parts.append(project.combine_solids(name, 'subtract', made))
    parts.append(project.add_solid_block(name, *BEAM_SOLID, name='beam', unit='in'))
    return parts


def solid_geometry(part: str = 'BARC', size: float = SIZE) -> Any:
    """The BARC, or its removable component, as a geometry of bricks
    and point masses, every element group given its material or mass,
    ready for `fem.Model.from_geometry` or for the app's Solve Modes. A
    `part` not in `PARTS` raises ValueError.

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
    from visualdynamics.project import Project

    if part not in PARTS:
        raise ValueError(f'{part!r} is not one of {PARTS}')
    # the parts meshed together on one grid, cut at every part's faces,
    # each keeping its own nodes: where two touch their nodes lie on
    # each other's, joined to none, for the bolts to tie. Meshed one
    # at a time, each on its own grid, a channel's foot met the box's
    # top with no nodes in common at a quarter inch (2026-10-11)
    project = Project('BARC')
    name = project.new_geometry('BARC', unit='in')
    project.mesh_solids(name, _parts(project, name, part), size, unit='in',
                        parts_apart=True)
    whole = project[name]
    whole.view = VIEW
    whole.group_properties = {int(group): fem.GroupProperties(MATERIAL)
                              for group in whole.group_id}
    ids = dict(zip(whole.group_name, whole.group_id.tolist()))
    kinds = ('foot', 'top') if part == 'BARC' else ('top',)
    for kind in kinds:
        for side in ('right channel', 'left channel'):
            below, above = ('box', side) if kind == 'foot' else (side, 'beam')
            places = sorted({node for (bx, bz), k in SOLID_BOLTS if k == kind
                             and (bx > 0) == (side == 'right channel')
                             for node in bolt_places(whole, (bx, bz), kind)})
            # Mate Surfaces on the two parts, as the pane does it: the
            # patch where they touch, its first side kept at the places
            # the bolts clamp and clicked off everywhere else (a foot's
            # outer ring of 32; at the beam None, then the 16 under the
            # bolts), rigid, named for the joint
            groups = [ids[below], ids[above]]
            patch, = project.detect_mated_surfaces(
                name, groups, 0.25 * whole.node_spacing(groups))
            clamped = set(places) | set(_beneath(whole, ids[below], places))
            project.add_mated_surfaces(
                name, patch['groups'], [n for n in patch['nodes_a'] if n in clamped],
                patch['nodes_b'], name=f'{side.split()[0]} {kind} bolts')
    # each bolt's mass at the node of its head face nearest its axis
    xyz = whole.node_xyz / INCH
    for kind in kinds:
        nodes = []
        for (bx, bz), k in SOLID_BOLTS:
            if k != kind:
                continue
            on_face = np.flatnonzero(np.abs(xyz[:, 1] - HEAD_Y[kind]) < 1e-6)
            far = np.hypot(xyz[on_face, 0] - bx, xyz[on_face, 2] - bz)
            # the axis runs through a brick's middle, so its four corners
            # tie: the one at the lowest x, then z, whatever the nodes'
            # numbering (Mesh Solids numbers them otherwise than the old
            # boxes did, and moved the masses a brick, 2026-10-11)
            tied = on_face[far <= far.min() + 1e-9]
            nearest = tied[np.lexsort((xyz[tied, 2], xyz[tied, 0]))[0]]
            nodes.append([int(whole.node_id[nearest])])
        group = whole.add_group(f'{kind} bolts')
        whole.add_elements(nodes, [161] * len(nodes), [group] * len(nodes))
        whole.group_properties[group] = fem.GroupProperties(
            mass=BOLT_MASS[kind] * POUND)
    return whole


def _beneath(geometry: Any, group: int, places) -> list[int]:
    """The nodes of element group `group` at the places `places`: on
    one grid, each place's partner across the joint is the node of the
    part below at the same point."""
    from scipy.spatial import cKDTree

    rows = np.flatnonzero(geometry.elem_group == group)
    own = np.unique(np.concatenate([geometry.elem_conn[r] for r in rows]))
    tree = cKDTree(geometry.node_xyz[geometry.node_index(own)])
    gap, nearest = tree.query(geometry.node_xyz[geometry.node_index(places)])
    assert gap.max() < 1e-9, 'a place with no node beneath it'
    return sorted({int(own[k]) for k in nearest})


def bolt_places(geometry: Any, axis: tuple, kind: str) -> set[int]:
    """The places a bolt clamps, as the ids of the nodes of the part on
    top that a bolt's mated set ties there.

    They are the corners, on the face where the parts touch, of the
    bricks the bolt clamps, picked by the same `CLAMPED` rule as the
    plate model's bolts: the washer's 4 x 4 places at a foot, and the
    four corners of the brick the bolt passes through at the beam. These
    are the places a person keeps marked in the Mate Surfaces pane.

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
    """The finite element model of bricks of the BARC or of its removable
    component, built from `solid_geometry`.

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
    component, both of bricks, so Solve Modes on either gives its modes.

        barc.solid_project().save('barc-bricks.vdyn')

    With `solved`, the BARC's modes are already solved to `SOLVE_TO`
    (2000 Hz) and the removable component's to `PART_SOLVE_TO`
    (2600 Hz). That is the example project the downloads page offers.

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
    """Print the plate model, the brick BARC and the brick removable
    component, each with its size, its mass and its first `modes`
    elastic frequencies (those above 1 Hz)."""
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
