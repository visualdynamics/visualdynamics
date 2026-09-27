"""The BARC, built from planes.

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

Run it directly to print the model and its first modes:

    python3 -m visualdynamics.demo.barc
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visualdynamics import fem, mesh

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
        covers, `washer_patch`).
    """
    whole = mesh.assemble(*_planes(size))
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


#: a quarter turn about x: the solid model's up (+y) to the 3-D scene's
#: (+z), its depth (+z) to -y — each DOF axis to the one it becomes, a
#: rotation with its translation, and the old axes that land negative
_UPRIGHT = {'X': 'X', 'Y': 'Z', 'Z': 'Y', 'RX': 'RX', 'RY': 'RZ', 'RZ': 'RY'}
_FLIPPED = {'Z', 'RZ'}


def upright(points: Any, shapes: Any = None) -> Any:
    """The BARC stood up for a picture. The model is built in the solid
    model's frame, where y is up; a 3-D scene treats z as up and draws it
    lying on its side. The website figure and the downloads tile turn it;
    the model and the docs keep the solid model's frame, so only
    pictures are turned.

    Parameters
    ----------
    points : Geometry
        The geometry to turn (a copy is returned).
    shapes : ShapeSet, optional
        Shapes on it, whose DOFs are renamed by the same turn.

    Returns
    -------
    Geometry, or (Geometry, ShapeSet) when `shapes` is given
    """
    import copy

    from visualdynamics.core.shapes import ShapeSet

    turned = copy.deepcopy(points)
    x, y, z = points.node_xyz.T
    turned.node_xyz = np.column_stack([x, -z, y])
    if shapes is None:
        return turned
    names = []
    for dof in shapes.coordinate:
        node = dof.rstrip('+-XYZR')
        axis, sign = dof[len(node):-1], dof[-1]
        if axis in _FLIPPED:
            sign = '-' if sign == '+' else '+'
        names.append(f'{node}{_UPRIGHT[axis]}{sign}')
    return turned, ShapeSet(shapes.frequency, shapes.damping, names,
                            shapes.shape_matrix, comment=shapes.comment)


def describe(size: float = SIZE, modes: int = 10) -> None:
    """Print the model and its first elastic modes."""
    model = build(size)
    shapes = model.eigensolution(maximum_frequency=SOLVE_TO)
    print(f'BARC from planes: {model.num_nodes} nodes, {len(model.plates)} '
          f'plates, {len(model.rigid_links)} rigid links, '
          f'{model.structural_mass:.3f} kg')
    elastic = [f for f in shapes.frequency if f > 1.0][:modes]
    print('  elastic modes (Hz): ' + ', '.join(f'{f:.1f}' for f in elastic))


if __name__ == '__main__':
    describe()
