"""The BARC, built from planes: a model checked against a published one.

The BARC — Box Assembly with Removable Component — is a small bolted
aluminum structure the structural dynamics community shares as a
common test article: a square box tube with its top wall slotted in two,
and on it the "Bench", two channels standing on the two halves and a
flat bar bolted across their tops. Its solid model, test data and a
finite element model's modes are shared on the SEM Dynamic
Substructuring Focus Group wiki, https://wiki.sem.org/wiki/BARC.

Here it is built the way a person would build a simple plate model with
this package (Brandon, 2026-09-26): each part as the planes it is made
of, meshed at mid-thickness (`visualdynamics.mesh`), each part a block
given its material and thickness, and the bolts as rigid, massless
links. Every dimension below comes from the shared solid model (inches):

    from visualdynamics.demo import barc

    geometry = barc.geometry()             # blocks already given properties
    model = barc.build()
    shapes = model.eigensolution(maximum_frequency=2000)
    for row in barc.compare(shapes, model):
        print(row)

**The joints are the model.** The parts meet through bolts, and their
mid-surfaces do not touch — a channel's foot sits 0.1875 in above the
box wall's mid-surface — so what joins them is a modeling choice, and it
decides the Bench's modes. Three were measured against the reference:

- each bolt a single rigid link between two nodes: 6 to 26% soft, and
  softer at every mesh refinement — a point tie on a plate is a local
  singularity, not a joint;
- each foot tied to the wall over its whole area: 5 to 20% stiff;
- each bolt tied over its washer's area (`WASHER_RADIUS`): converged
  (halving the mesh moves the frequencies about 1%), and at `SIZE` the
  first ten modes pair one-to-one with the reference, in order, at MAC
  0.83 to 0.996 and within −3 to +4% of its frequencies.

The last is what is built. What remains — the model a few percent stiff —
is left as it is rather than tuned away: the reference was a different
model (not plates), its material values are not published, and a check
that is tuned to agree says nothing.

Run it directly to print the comparison:

    python3 -m visualdynamics.demo.barc
"""

from __future__ import annotations

import pathlib
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

#: the reference model's frame is this model's moved by (0, −3, −1.5) in
REFERENCE_OFFSET = np.array([0.0, 3.0, 1.5])
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
        plates) and 'bolts' (rigid links over each washer's area).
    """
    whole = mesh.assemble(*_planes(size))
    xyz = whole.node_xyz / INCH

    def on(surface_y):
        return np.flatnonzero(np.abs(xyz[:, 1] - surface_y) < 1e-6)

    bolts = whole.add_block('bolts')
    for (bx, bz), lower, upper, radius in BOLTS:
        below = on(lower)
        for k in on(upper):
            if np.hypot(xyz[k, 0] - bx, xyz[k, 2] - bz) > radius + 1e-9:
                continue
            nearest = below[np.argmin(np.linalg.norm(
                xyz[below][:, [0, 2]] - xyz[k, [0, 2]], axis=1))]
            whole.add_element([int(whole.node_id[nearest]),
                               int(whole.node_id[k])], elem_type=21,
                              block=bolts)
    whole.block_properties = {
        int(block): (fem.BlockProperties(fem.RIGID)
                     if whole.block_name[i] == 'bolts'
                     else fem.BlockProperties(
                         MATERIAL, THICKNESS[whole.block_name[i]] * INCH))
        for i, block in enumerate(whole.block_id)}
    return whole


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


def reference(path: str | pathlib.Path | None = None) -> dict:
    """The published reference modes: the arrays of
    `testdata/barc/reference_modes.npz` (frozen from the wiki's shared
    data), their coordinates moved into this model's frame, in meters.

    Parameters
    ----------
    path : str or path, optional
        The file; the repository's own copy when omitted.

    Returns
    -------
    dict
        'frequency', 'dof', 'shape' (DOFs x modes) and 'coordinates'
        (m, this model's frame).
    """
    if path is None:
        path = (pathlib.Path(__file__).resolve().parents[3]
                / 'testdata' / 'barc' / 'reference_modes.npz')
    with np.load(path) as data:
        return {'frequency': data['frequency'],
                'dof': [str(d) for d in data['dof']],
                'shape': data['shape'],
                'coordinates': (data['coordinates'] + REFERENCE_OFFSET) * INCH}


def reference_shapes(path: str | pathlib.Path | None = None
                     ) -> tuple[Any, Any]:
    """The reference as the app holds it: its modes as a shape set and
    its 59 points as a geometry, in this model's frame and in meters — so
    the comparison screen (select both shape sets) projects the model's
    modes onto the reference's points and reads the MAC, and Match Modes
    commits the pairs.

    Parameters
    ----------
    path : str or path, optional
        As `reference` takes it.

    Returns
    -------
    tuple of (ShapeSet, Geometry)
    """
    from visualdynamics.core.geometry import Geometry
    from visualdynamics.core.shapes import ShapeSet

    ref = reference(path)
    points: dict[int, np.ndarray] = {}
    for dof, xyz in zip(ref['dof'], ref['coordinates'], strict=True):
        points.setdefault(int(dof[:-2]), xyz)
    geometry = Geometry(node_id=list(points),
                        node_xyz=np.array(list(points.values())),
                        length_unit='in')
    shapes = ShapeSet(ref['frequency'],
                      np.full(len(ref['frequency']), 0.01), ref['dof'],
                      np.asarray(ref['shape']).T,
                      comment='BARC reference modes (SEM wiki)')
    return shapes, geometry


#: how far the solved example project solves, Hz, and how many elastic
#: reference modes it matches: the ten that pair one-to-one and in order
#: (above them, where modes crowd, the pairing loosens below MAC 0.5)
SOLVE_TO = 2000.0
MATCHED = 10


def project(size: float = SIZE, solved: bool = False) -> Any:
    """A project to open in the app: the BARC's geometry, its blocks
    given their properties, and the reference's geometry and modes,
    linked. Solve Modes on 'BARC', then select 'Reference Modes' with the
    solved modes: the comparison screen reads the MAC, and Match Modes
    commits the pairs.

        barc.project().save('barc.vdyn')

    With `solved`, those steps are already taken — the modes solved to
    `SOLVE_TO` and the first `MATCHED` elastic reference modes matched to
    the model's by MAC — which is the example project the downloads page
    offers: it opens on the answer.

    Parameters
    ----------
    size : float
        The element size aimed at, in inches.
    solved : bool, default False
        Solve the modes and match them to the reference as well.

    Returns
    -------
    Project
    """
    from visualdynamics.project import Project

    out = Project('BARC')
    out.add('BARC', geometry(size))
    shapes, points = reference_shapes()
    out.add('Reference Geometry', points)
    out.add('Reference Modes', shapes)
    out.link('Reference Geometry', 'Reference Modes')
    if solved:
        modes = out.solve_modes('BARC', maximum_frequency=SOLVE_TO)
        mac = out.comparison_mac('Reference Modes', modes)
        # pairs chosen by hand, as a person would on the comparison
        # screen: the rigid-body modes left out (any rigid motion scores
        # against any other), then each elastic mode's best partner
        elastic = np.flatnonzero(shapes.frequency > 1.0)[:MATCHED]
        out.match_modes('Reference Modes', modes,
                        pairs=[(int(row), int(np.argmax(mac[row])))
                               for row in elastic])
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
    the model, its comparison with the reference and the docs keep the
    solid model's frame, so only pictures are turned.

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


def compare(shapes: Any, model: fem.Model,
            reference_modes: dict | None = None) -> list[dict]:
    """Each elastic reference mode beside the model's that matches it
    best by MAC, read at the reference's own points — the model's
    nearest node to each, in the reference's direction.

    Parameters
    ----------
    shapes : ShapeSet
        The model's modes (`model.eigensolution`).
    model : fem.Model
        The model they came from, for where its nodes are.
    reference_modes : dict, optional
        `reference()`; read when omitted.

    Returns
    -------
    list of dict
        One per elastic reference mode, in order: 'reference' and
        'model' (Hz), 'error' (percent), 'mac', and 'mode' (the model's
        elastic mode number, from 1).
    """
    ref = reference() if reference_modes is None else reference_modes
    nodes = model.node_ids
    xyz = np.array([model.position(n) for n in nodes])
    column = {dof: i for i, dof in enumerate(shapes.coordinate)}
    rows = []
    for dof, point in zip(ref['dof'], ref['coordinates'], strict=True):
        node = nodes[int(np.argmin(np.linalg.norm(xyz - point, axis=1)))]
        rows.append(column[f'{node}{dof[-2:]}'])
    ours = np.asarray(shapes.shape_matrix)[:, rows].T          # DOFs x modes
    theirs_all = np.asarray(ref['shape'])
    ref_elastic = np.flatnonzero(ref['frequency'] > 0.0)
    our_elastic = np.flatnonzero(np.asarray(shapes.frequency) > 0.0)
    theirs, mine = theirs_all[:, ref_elastic], ours[:, our_elastic]
    mac = ((theirs.T @ mine) ** 2
           / np.outer((theirs ** 2).sum(0), (mine ** 2).sum(0)))
    out = []
    for i, r in enumerate(ref_elastic):
        j = int(np.argmax(mac[i]))
        f_ref = float(ref['frequency'][r])
        f_model = float(shapes.frequency[our_elastic[j]])
        out.append({'reference': f_ref, 'model': f_model,
                    'error': 100.0 * (f_model / f_ref - 1.0),
                    'mac': float(mac[i, j]), 'mode': j + 1})
    return out


def describe(size: float = SIZE, modes: int = 10) -> None:
    """Print the model and its comparison with the reference."""
    model = build(size)
    shapes = model.eigensolution(maximum_frequency=2000.0)
    print(f'BARC from planes: {model.num_nodes} nodes, {len(model.plates)} '
          f'plates, {len(model.rigid_links)} rigid links, '
          f'{model.structural_mass:.3f} kg')
    print('  reference   model   error    MAC')
    for row in compare(shapes, model)[:modes]:
        print(f'  {row["reference"]:8.1f}  {row["model"]:7.1f}  '
              f'{row["error"]:+5.1f}%  {row["mac"]:.3f}')


if __name__ == '__main__':
    describe()
