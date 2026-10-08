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
of, meshed at mid-thickness (`visualdynamics.mesh`), each part a block
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
over the elements its washer covers (`washer_patch`), to the part below
with `mesh.tie` — what the app's Tie does with the same elements picked
(2026-09-26) — so the model a person builds in the app by the
documentation's steps is this one, node for node and link for link.

**And from bricks** (2026-10-07): the same structure as solid
eight-node bricks (`mesh.block`), the higher-fidelity model, and the
removable component on its own — the two channels and the beam, the
part the BARC is named for (`solid_geometry`, `solid_project`). The
joints follow the finite element model shared on the wiki: the feet
are joined to the box over their whole footprint (the bricks share
nodes there), the beam rests 0.001 in over the channels so it joins
them only where it is bolted, tied over each washer with `mesh.tie`,
and the bolts are point masses, a block of point elements given a mass
(`fem.BlockProperties(mass=...)`). Without the bolt masses the
assembly reads 1 to 8 % stiffer. Checked against the wiki's models and
test: the assembly's first ten elastic modes 1 to 4 % above the
measured ones, its shapes matching the shared model's in order.

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

#: the bolts: (x, z) on the part, the two mid-surfaces they join, and
#: the washer's radius — #8 screws with 0.375 in washers at the feet,
#: 1/4 in screws with 0.5 in washers at the beam
WASHER_RADIUS = {'foot': 0.1875, 'top': 0.25}
BOLTS = ([((x, z), BOX_MID, FOOT_Y, WASHER_RADIUS['foot'])
          for x in (-2.1875, -1.8125, 1.8125, 2.1875) for z in (1.3125, 1.6875)]
         + [((x, 1.5), TOP_Y, BEAM_Y, WASHER_RADIUS['top'])
            for x in (-2.0, 2.0)])

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
    """The BARC as a geometry of plates and rigid links, every block
    given what it is made of — ready for `fem.Model.from_geometry`, or
    for the app's Solve Modes.

    Parameters
    ----------
    size : float
        The element size aimed at, in inches.

    Returns
    -------
    Geometry
        Blocks 'box', 'right channel', 'left channel', 'beam' (6061-T6
        plates) and 'bolts' (rigid links over the elements each washer
        covers, `washer_patch`), opening on `VIEW`.
    """
    whole = mesh.assemble(*_planes(size))
    whole.view = VIEW
    whole.block_properties = {
        int(block): fem.BlockProperties(
            MATERIAL, THICKNESS[whole.block_name[i]] * INCH)
        for i, block in enumerate(whole.block_id)}
    for bolt in BOLTS:
        patch, below = washer_patch(whole, bolt)
        mesh.tie(whole, patch, below, block='bolts')
    return whole


def washer_patch(geometry: Any, bolt: tuple) -> tuple[list[int], str]:
    """The elements a bolt's washer covers, and the block it bolts them to:
    the elements of the part on top whose centers lie within the washer's
    radius of the bolt along both edges — at `SIZE`, the element the bolt
    passes through and the eight around it under a foot, the 4 × 4 around
    the node it passes through on the beam. A square rather than the
    washer's circle because it is what a person picks: whole rows of
    elements, countable in the view.

    Parameters
    ----------
    geometry : Geometry
        The BARC's planes, assembled.
    bolt : tuple
        One of `BOLTS`.

    Returns
    -------
    (list of int, str)
        The element ids, and the name of the block below.
    """
    (bx, bz), _lower, upper, radius = bolt
    side = 'left channel' if bx < 0 else 'right channel'
    top, below = ('beam', side) if upper == BEAM_Y else (side, 'box')
    xyz = geometry.node_xyz / INCH
    patch = []
    for element in geometry.elements_in(top):
        row = int(np.flatnonzero(geometry.elem_id == element)[0])
        center = xyz[geometry.node_index(geometry.elem_conn[row])].mean(axis=0)
        if (abs(center[1] - upper) < 1e-6 and abs(center[0] - bx) <= radius
                and abs(center[2] - bz) <= radius):
            patch.append(element)
    return patch, below


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
    """A project to open in the app: the BARC's geometry, its blocks
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
#: how far the beam rests over the channels' tops, inches: far enough
#: that assembling does not fuse it to them, which made the beam a
#: flange of each channel and read stiff
BEAM_GAP = 0.001
#: how far around a beam bolt the beam is tied to its channel, inches:
#: the washer's radius, as `WASHER_RADIUS['top']`
TIE_RADIUS = 0.25
#: each bolt's mass, pounds, as the shared model lumps them: #8 screws
#: at the feet, 1/4 in at the beam
BOLT_MASS = {'foot': 0.00744, 'top': 0.0125}
#: the bolts: (x, z), the face its head sits on (y, inches), its kind
SOLID_BOLTS = ([((x, z), 3.125, 'foot')
                for x in (-2.1875, -1.8125, 1.8125, 2.1875)
                for z in (1.3125, 1.6875)]
               + [((x, 1.5), 5.125 + BEAM_GAP, 'top') for x in (-2.0, 2.0)])
POUND = 0.45359237
#: how far the solved removable component solves, Hz: its seventh
#: elastic mode is just past `SOLVE_TO`
PART_SOLVE_TO = 2600.0


def _boxes(part: str) -> list:
    """The parts as boxes, inches: (corner, extent, block)."""
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
        ((-2.5, 5.0 + BEAM_GAP, 1), (5, t, 1), 'beam'),
    ]
    whole = [(*b, 'box') for b in box] + component
    return whole if part == 'BARC' else component


def _bricks(size: float, part: str) -> list:
    """The parts as blocks of bricks, each a block of its own name.

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
    """The BARC, or its removable component, as a geometry of bricks,
    rigid ties and point masses, every block given what it is made of —
    ready for `fem.Model.from_geometry`, or for the app's Solve Modes.

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
        Blocks 'box' (the assembly only), 'right channel', 'left
        channel', 'beam' (6061-T6 bricks), 'bolt ties' (rigid links
        from the beam over each washer to its channel) and the bolts'
        masses, 'foot bolts' (the assembly only) and 'top bolts',
        opening on `VIEW`.
    """
    if part not in PARTS:
        raise ValueError(f'{part!r} is not one of {PARTS}')
    whole = mesh.assemble(*_bricks(size, part))
    whole.view = VIEW
    whole.block_properties = {int(block): fem.BlockProperties(MATERIAL)
                              for block in whole.block_id}
    xyz = whole.node_xyz / INCH
    for (bx, bz), _face, kind in SOLID_BOLTS:
        if kind != 'top':
            continue
        patch = []
        for element in whole.elements_in('beam'):
            row = int(np.flatnonzero(whole.elem_id == element)[0])
            center = xyz[whole.node_index(whole.elem_conn[row])].mean(axis=0)
            if (abs(center[0] - bx) <= TIE_RADIUS
                    and abs(center[2] - bz) <= TIE_RADIUS):
                patch.append(element)
        mesh.tie(whole, patch, 'left channel' if bx < 0 else 'right channel',
                 block='bolt ties')
    # each bolt's mass at the node of its head face nearest its axis
    xyz = whole.node_xyz / INCH
    for kind in ('foot', 'top'):
        heads = [(axis, face) for axis, face, k in SOLID_BOLTS if k == kind]
        if kind == 'foot' and part != 'BARC':
            continue
        nodes = []
        for (bx, bz), face in heads:
            on_face = np.flatnonzero(np.abs(xyz[:, 1] - face) < 1e-6)
            nearest = on_face[np.argmin(np.hypot(xyz[on_face, 0] - bx,
                                                 xyz[on_face, 2] - bz))]
            nodes.append([int(whole.node_id[nearest])])
        block = whole.add_block(f'{kind} bolts')
        whole.add_elements(nodes, [161] * len(nodes), [block] * len(nodes))
        whole.block_properties[block] = fem.BlockProperties(
            mass=BOLT_MASS[kind] * POUND)
    return whole


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
              f'{len(model.solids)} bricks, {len(model.rigid_links)} rigid '
              f'links, {model.total_mass:.3f} kg with the bolts')
        elastic = [f for f in shapes.frequency if f > 1.0][:modes]
        print('  elastic modes (Hz): '
              + ', '.join(f'{f:.1f}' for f in elastic))


if __name__ == '__main__':
    describe()
