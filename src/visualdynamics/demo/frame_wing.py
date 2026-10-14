"""The Frame-Wing, a frame and its wings, built from blocks of bricks.

The Frame-Wing is a small aluminum frame and two wings the structural
dynamics community shares as a common test article for substructuring:
a 16 by 6 in ladder cut from half-inch plate, two rails and five
uprights half an inch wide, with 41 threaded steel inserts for bolting
things to, and two rectangular 22 by 4.4 in wings, thin (1/8 in) and
thick (1/4 in), that screw across two of the frame's uprights. Its solid
models, finite element models and test data are shared on the SEM
Dynamic Substructuring Focus Group wiki,
https://wiki.sem.org/wiki/Round_Robin_Frame_Structure.

Here it is built the way a person builds it in the app, every step a
project verb the app records. Each part is made as solids: the frame a
plate less its four windows (with rounded corners, which leave the
fillets) and its 41 holes, the inserts a union of cylinders in the tops
of the holes, and each wing a plate less its eighteen holes. The holes
are copied in rows with Pattern, and the solids are meshed into bricks
with Mesh Solids. A wing is bolted on the way the BARC's bricks are, by
merging nodes: the wing and the washers under it are meshed apart from
the frame, Align Meshes puts the washers' undersides on the frame's
nodes, and Merge Coincident Nodes joins them. The screws are steel,
threaded into the four inserts they pass through (a steel cylinder over
each, made last so it takes those bricks), with the rest of each screw's
mass as a point mass on the wing's top. Every dimension below is read
from the shared finite element models, in inches:

    from visualdynamics.demo import frame_wing

    model = frame_wing.build()                    # the frame alone
    shapes = model.eigensolution(maximum_frequency=2000)
    both = frame_wing.build('thick wing')         # the frame with a wing on it

The model is made of solid elements. Beams describe a half-inch bar
spanning a window of about a hundred millimeters poorly, and plates
cannot describe it at all, so this example is what the eight-node brick
with incompatible modes was written for. The element was checked against
the mesh the wiki shares: solved here node for node, its first ten
elastic modes are within 0.1 % of the frequencies MSC Nastran gives for
the same mesh. The model built here is coarser than the shared mesh, a
brick a twelfth of an inch across and 45,087 nodes for the frame, and it
is within a few percent of the measured frames, which differ from each
other by as much.

The measured frequencies of the frames are on the wiki, and the wings'
measured frequencies were fitted from the shared test FRFs with
`Project.fit_modes`. The assemblies were checked against the finite
element models of the frame with each wing shared on the wiki (A.
Linderholt, Linnaeus University, 2024), tuned there to test: within
about 4 % over the first nine modes. None of the wiki's files ship with
this package.

Run it directly to print the models and their first modes:

    python3 -m visualdynamics.demo.frame_wing
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visualdynamics import View, fem

# Built with the app's verbs so that students can follow the
# documentation and make it themselves (Brandon, 2026-10-11).

#: element size aimed at, inches: six bricks across a half-inch member
#: and three across an insert hole, which converges the frame's first
#: nine modes to about a percent and solves in seconds
SIZE = 1.0 / 12.0
#: the wings are plates, and a plate's modes converge with the in-plane
#: mesh, not the layers through its thickness (one brick is enough):
#: an eighth of an inch in-plane keeps a wing under twenty thousand
#: nodes
WING_SIZE = 1.0 / 8.0

#: the frame: 16 by 6 in cut from half-inch plate, members half an inch
#: wide, in the shared models' frame — x along the rails, y along the
#: uprights, the top face at z = 0 and the plate below it
LENGTH, WIDTH, THICKNESS = 16.0, 6.0, 0.5
MEMBER = 0.5
#: the uprights' centers along the rails, and the rails' centers
UPRIGHTS = (-7.75, -3.875, 0.0, 3.875, 7.75)
RAILS = (-2.75, 2.75)
#: the inserts, as the shared model has them: a 0.29 in hole drilled
#: through, a threaded insert 5/16 in long in its top with a 0.209 in
#: bore — thirteen along each rail at a 1.2917 in pitch and three down
#: each upright, 41 in all. The insert's bore is not meshed: its
#: annulus is thinner than a brick, so the insert fills its hole with
#: the annulus's mass and stiffness spread over the circle (`INSERT`)
HOLE_RADIUS, INSERT_DEPTH, BORE_RADIUS = 0.1451, 0.3125, 0.1043
INSERT_PITCH = 15.5 / 12.0
INSERTS = ([(k * INSERT_PITCH, y) for y in RAILS for k in range(-6, 7)]
           + [(x, y) for x in UPRIGHTS for y in (-1.375, 0.0, 1.375)])
#: the fillet where an upright meets a rail, inside each window: 0.138
#: in, read off the shared mesh. It is what stiffens the frame's
#: in-plane modes — without it they come out 8 % low
FILLET = 0.138

#: the wings: 22 by 4.385 in, set across the uprights at x = 0 and
#: 3.875 with a 0.028 in washer under them, eighteen 0.209 in holes
WING_SPAN, WING_CHORD, WING_X = 22.0, 4.385, -0.255
WING_GAP = 0.0276
WING_THICKNESS = {'thin wing': 0.1243, 'thick wing': 0.2563}
WING_HOLE_RADIUS = 0.1043
WING_HOLES = ([(x, y) for x in (0.0, 3.875) for y in (-2.75, -1.375, 0.0, 1.375, 2.75)]
              + [(x, y) for x in (1.2917, 2.5833) for y in RAILS]
              + [(x, y) for x in (0.5626, 3.3126) for y in (-7.75, 7.75)])
#: the four screws, through the wing into the rail inserts under the
#: uprights it spans, and the washer each one presses through
SCREWS = [(x, y) for x in (0.0, 3.875) for y in RAILS]
WASHER_RADIUS = 0.196
#: a #10 screw with its two washers, steel, kg
SCREW_MASS = 0.0035

ALUMINUM = fem.material('6061-T6')
#: the threaded inserts as the shared model states them — a stiffness
#: tuned to the tested frames and a density that gives the frame its
#: measured mass (Linderholt, 2024) — spread from the insert's annulus
#: over the whole of its hole, by the annulus's share of the circle
_ANNULUS = 1.0 - (BORE_RADIUS / HOLE_RADIUS) ** 2
INSERT = fem.Material('threaded insert (as shared)', 1.6e10 * _ANNULUS,
                      7999.0 * _ANNULUS, 0.3)

#: where it is shared
WIKI = 'https://wiki.sem.org/wiki/Round_Robin_Frame_Structure'
INCH = 0.0254

#: how it opens in 3-D: from the front, above and to the right, z up —
#: the frame lies flat with its wing standing across it
VIEW = View(eye=(1.0, -1.0, 1.0), up=(0.0, 0.0, 1.0))

WINGS = tuple(WING_THICKNESS)
#: the steel the screws and washers are made of
STEEL = fem.material('1018 steel')
#: the insert pitch along a rail, inches
PITCH = INSERT_PITCH


def _holes(project: Any, name: str, radius: float, z: float, length: float,
           label: str = '') -> list[int]:
    """The 41 insert positions as two patterns of cylinders, the way a
    person makes them with Pattern, the fewest steps (Brandon,
    2026-10-11): one hole, 13 along X at the pitch and 2 along Y for the
    two rails; one hole, 5 along X and 3 along Y for the uprights."""
    first = project.add_solid_cylinder(name, (UPRIGHTS[0], RAILS[0], z), radius,
                                       length, name=label, unit='in')
    rails = project.pattern_solids(name, [first], (13, 2, 1),
                                   (PITCH, RAILS[1] - RAILS[0], 0), unit='in')
    upright = project.add_solid_cylinder(name, (UPRIGHTS[0], -1.375, z), radius,
                                         length, name=label, unit='in')
    uprights = project.pattern_solids(name, [upright], (len(UPRIGHTS), 3, 1),
                                      (UPRIGHTS[1] - UPRIGHTS[0], 1.375, 0), unit='in')
    return [rails, uprights]


def _frame(project: Any, name: str, screwed: bool, size: float) -> None:
    """The frame's solids, meshed: the plate less its windows and holes,
    the inserts, and — under a wing — the four screws threaded into the
    inserts they pass through, steel, made last so they take those
    inserts' bricks."""
    plate = project.add_solid_block(name, (0, 0, -THICKNESS / 2),
                                    (LENGTH, WIDTH, THICKNESS), name='frame',
                                    unit='in')
    span = UPRIGHTS[1] - UPRIGHTS[0]
    window = project.add_solid_block(
        name, ((UPRIGHTS[0] + UPRIGHTS[1]) / 2, 0, -THICKNESS / 2),
        (span - MEMBER, WIDTH - 2 * MEMBER, 2 * THICKNESS), corner=FILLET,
        unit='in')
    windows = project.pattern_solids(name, [window], (4, 1, 1), (span, 0, 0),
                                     unit='in')
    drilled = _holes(project, name, HOLE_RADIUS, -THICKNESS / 2, 2 * THICKNESS)
    body = project.combine_solids(name, 'subtract', [plate, windows, *drilled])
    inserts = project.combine_solids(
        name, 'union', _holes(project, name, HOLE_RADIUS, -INSERT_DEPTH / 2,
                              INSERT_DEPTH, 'inserts'))
    parts = [body, inserts]
    if screwed:
        parts.append(_four(project, name, HOLE_RADIUS, -INSERT_DEPTH / 2,
                           INSERT_DEPTH, 'screws'))
    project.mesh_solids(name, parts, size, unit='in')


def _four(project: Any, name: str, radius: float, z: float, length: float,
          label: str) -> int:
    """A cylinder at each of the four screws, a pattern: one, 2 along X
    between the uprights the wing spans and 2 along Y between the
    rails."""
    (x0, y0), (x1, _y), (_x, y1) = SCREWS[0], SCREWS[2], SCREWS[1]
    first = project.add_solid_cylinder(name, (x0, y0, z), radius, length,
                                       name=label, unit='in')
    return project.pattern_solids(name, [first], (2, 2, 1), (x1 - x0, y1 - y0, 0),
                                  unit='in')


def _wing(project: Any, name: str, wing: str, washers: bool,
          size: float) -> None:
    """A wing's solids, meshed: the plate less its eighteen holes and —
    on the frame — the washer under it at each screw, kept apart from
    the frame to be bolted to it."""
    thickness = WING_THICKNESS[wing]
    z = WING_GAP + thickness / 2
    plate = project.add_solid_block(
        name, (WING_X + WING_CHORD / 2, 0, z), (WING_CHORD, WING_SPAN, thickness),
        name=wing, unit='in')
    holes = []
    for (x, y), counts, spacing in (((0.0, -2.75), (2, 5, 1), (3.875, 1.375)),
                                     ((1.2917, RAILS[0]), (2, 2, 1),
                                      (PITCH, RAILS[1] - RAILS[0])),
                                     ((0.5626, -7.75), (2, 2, 1), (2.75, 15.5))):
        first = project.add_solid_cylinder(name, (x, y, z), WING_HOLE_RADIUS,
                                           1.0, unit='in')
        holes.append(project.pattern_solids(name, [first], counts,
                                            (*spacing, 0), unit='in'))
    parts = [project.combine_solids(name, 'subtract', [plate, *holes])]
    if washers:
        parts.append(_four(project, name, WASHER_RADIUS, WING_GAP / 2, WING_GAP,
                           'washers'))
    project.mesh_solids(name, parts, size, unit='in', apart=washers)


#: what each element group is made of, by name — the Element Groups table
MATERIALS = {'frame': ALUMINUM, 'inserts': INSERT, 'screws': STEEL,
             'washers': STEEL, **{name: ALUMINUM for name in WINGS}}


def _materials(geometry: Any) -> None:
    geometry.group_properties.update({
        int(group): fem.GroupProperties(MATERIALS[geometry.group_name[k]])
        for k, group in enumerate(geometry.group_id)
        if geometry.group_name[k] in MATERIALS})


def _bolt(project: Any, name: str, wing: str) -> None:
    """The wing bolted on: the frame, its inserts, its screws and the
    washers picked, Align Meshes and Merge Coincident Nodes on them, and
    each screw's grams on the node of the wing's top nearest it."""
    geometry = project[name]
    ids = dict(zip(geometry.group_name, geometry.group_id.tolist()))
    # a coarse mesh can leave a screw or an insert with no brick, and its
    # group with nothing in it: the ones there are picked
    picked = [ids[n] for n in ('frame', 'inserts', 'screws', 'washers') if n in ids]
    project.align_meshes(name, groups=picked)
    project.merge_coincident_nodes(name, 0.25 * geometry.node_spacing(picked),
                                   groups=picked)
    xyz = geometry.node_xyz / INCH
    top = np.flatnonzero(np.abs(xyz[:, 2] - (WING_GAP + WING_THICKNESS[wing])) < 1e-6)
    heads = [[int(geometry.node_id[top[np.argmin(
        np.hypot(xyz[top, 0] - x, xyz[top, 1] - y))]])] for x, y in SCREWS]
    group = geometry.add_group('screw heads')
    geometry.add_elements(heads, [161] * len(heads), [group] * len(heads))
    geometry.group_properties[group] = fem.GroupProperties(mass=SCREW_MASS)


def _model(project: Any, name: str, choice: str, size: float,
           wing_size: float) -> None:
    """One of the five models into `project` as geometry `name`."""
    alone = choice.endswith(' alone')
    part = choice.removesuffix(' alone')
    project.new_geometry(name, unit='in')
    if part == 'frame' or not alone:
        _frame(project, name, screwed=part != 'frame', size=size)
    if part != 'frame':
        _wing(project, name, part, washers=not alone, size=wing_size)
    _materials(project[name])
    if part != 'frame' and not alone:
        _bolt(project, name, part)
    project[name].view = VIEW


def geometry(wing: str | None = None, size: float = SIZE,
             wing_size: float = WING_SIZE) -> Any:
    """The frame, a wing, or the frame with a wing screwed on, as a
    geometry of bricks and point masses built from solids.

    Every element group is given its material or mass, so the geometry
    is ready for `fem.Model.from_geometry` or for the app's Solve Modes.
    The geometry keeps the solids it was meshed from, so it can be meshed
    again. Any other `wing` raises ValueError.

    Parameters
    ----------
    wing : {None, 'frame', 'thin wing', 'thick wing', 'thin wing alone', 'thick wing alone'}, optional
        None or 'frame' for the frame alone; a wing's name for the
        frame with that wing; 'thin wing alone' or 'thick wing alone'
        for the wing by itself.
    size : float
        The frame's target element size, in inches; `SIZE`, 1/12 in, by
        default.
    wing_size : float
        The wing's target element size, in inches; `WING_SIZE`, 1/8 in,
        by default.

    Returns
    -------
    Geometry
        Element groups 'frame' (6061-T6), 'inserts' (the insert material),
        the wing (6061-T6), and on an assembly 'screws' and 'washers'
        (steel) and 'screw heads' (point masses), opening on `VIEW`.
    """
    from visualdynamics.project import Project

    choice = (wing or 'frame').strip()
    if choice.removesuffix(' alone') not in ('frame', *WINGS):
        raise ValueError(f'{wing!r}: the frame, {" or ".join(WINGS)}, on '
                         'the frame or alone')
    project = Project('Frame-Wing')
    _model(project, 'Part', choice, size, wing_size)
    return project['Part']


def build(wing: str | None = None, size: float = SIZE,
          wing_size: float = WING_SIZE) -> fem.Model:
    """The finite element model, built from `geometry`.

    Parameters
    ----------
    wing, size, wing_size
        As for `geometry`.

    Returns
    -------
    fem.Model
    """
    name = (wing or 'frame').strip()
    return fem.Model.from_geometry(geometry(wing, size, wing_size),
                                   name=f'Frame-Wing: {name}')


#: how far the solved example projects solve, Hz
SOLVE_TO = 1500.0
#: the five models, as the example project names them
MODELS = {'Frame': 'frame', 'Thin Wing': 'thin wing alone',
          'Thick Wing': 'thick wing alone', 'Thin Wing on Frame': 'thin wing',
          'Thick Wing on Frame': 'thick wing'}


def project(solved: bool = False, size: float = SIZE,
            wing_size: float = WING_SIZE) -> Any:
    """A project to open in the app with five geometries: the frame, each
    wing alone, and the frame with each wing (`MODELS`).

    They are built by the verbs the app records, so the project's
    journal is the workflow, and their element groups are given their
    properties, so Solve Modes on any of them gives its modes.

        frame_wing.project().save('frame-wing.vdyn')

    With `solved`, every model's modes are already solved to `SOLVE_TO`
    (1500 Hz). That is the example project the downloads page offers.

    Parameters
    ----------
    solved : bool, default False
        Solve the modes as well.
    size, wing_size : float
        The target element sizes of the frame and the wings, in inches.

    Returns
    -------
    Project
    """
    from visualdynamics.project import Project

    out = Project('Frame-Wing')
    for label, choice in MODELS.items():
        _model(out, label, choice, size, wing_size)
        if solved:
            out.solve_modes(label, maximum_frequency=SOLVE_TO)
    return out


def describe(size: float = SIZE, modes: int = 9) -> None:
    """Print each of the five models with its size, its mass and its
    first `modes` elastic frequencies (those above 1 Hz)."""
    for name in MODELS.values():
        model = build(name, size)
        shapes = model.eigensolution(maximum_frequency=SOLVE_TO)
        print(f'{name}: {model.num_nodes} nodes, {len(model.solids)} bricks, '
              f'{model.total_mass * 1000:.1f} g')
        elastic = [f for f in shapes.frequency if f > 1.0][:modes]
        print('  elastic modes (Hz): ' + ', '.join(f'{f:.1f}' for f in elastic))


if __name__ == '__main__':
    describe()
