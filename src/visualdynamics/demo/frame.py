"""The four-unit frame and its wings, built from blocks of bricks.

The four-unit frame is a small aluminum ladder the structural dynamics
community shares as a common test article for substructuring: a 16 by
6 in frame cut from half-inch plate, two rails and five uprights half
an inch wide, with 41 threaded steel inserts for bolting things to, and
two rectangular 22 by 4.4 in wings, thin (1/8 in) and thick (1/4 in),
that screw across two of its uprights. Its solid models, finite element
models and test data are shared on the SEM Dynamic Substructuring
Focus Group wiki, https://wiki.sem.org/wiki/Round_Robin_Frame_Structure.

Here it is built the way a person would build a solid model with this
package (2026-09-30): each part as the rectangular blocks it is made
of, meshed into bricks (`visualdynamics.mesh.block`), the insert holes
cut from the blocks and their bricks given the insert's material, the
wing's holes cut through, and the screws as rigid links over the
bricks each washer covers, the way the BARC's bolts are. Every
dimension below is read off the shared finite element models, in
inches:

    from visualdynamics.demo import frame

    model = frame.build()                    # the frame alone
    shapes = model.eigensolution(maximum_frequency=2000)
    both = frame.build('thick wing')         # the frame with a wing on it

**The solid elements are the model.** Beams describe a half-inch bar
spanning a hundred-millimeter window poorly and plates not at all, so
this example is what the eight-node brick with incompatible modes was
written for. The element was checked against the mesh the wiki
shares: solved here node for node, its first ten elastic modes land
within 0.1 % of the frequencies MSC Nastran gives the same mesh. The
model built *here* is coarser — one brick in twelve of an inch, five
thousand nodes for the frame where the shared mesh has ninety
thousand — and its holes are stair-stepped, and it lands within a few
percent of the frames that were measured, which vary by as much among
themselves.

What the frames were measured to is on the wiki; the wings' and the
assembly's frequencies here were fitted from the shared test FRFs with
`Project.fit_modes`. None of the wiki's files ship with this package.

Run it directly to print the models and their first modes:

    python3 -m visualdynamics.demo.frame
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from visualdynamics import View, fem, mesh

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


def _frame_blocks(size: float) -> list:
    """The rails, the uprights and the fillets between them, each an
    element group named 'frame' with its holes cut from it, the inserts' bricks
    into 'inserts'; assembled, they share the faces where they meet.

    A rail is meshed as the segments between the uprights' edges and
    the fillets' ends, and an upright as the run between its fillets
    and the two short pieces beside them, so every face two blocks
    share is the same rectangle on both sides and divides the same
    way — a rail meshed whole at a twelfth of an inch puts nodes
    between the uprights' at 3.875 in, and the two never join. A fillet
    is a small square block with a void cut from its far corner: what
    is left is the material in the corner of the window."""
    x, y, z = np.eye(3)
    top = 0.0
    half = MEMBER / 2.0
    holes = []
    for cx, cy in INSERTS:
        holes.append(((cx, cy, top), HOLE_RADIUS, 2, None, None))
        holes.append(((cx, cy, top), HOLE_RADIUS, 2, INSERT_DEPTH, 'inserts'))
    parts = []
    # a fillet two bricks across keeps its corner brick; one brick across
    # is all void, so a mesh that coarse goes without fillets. Fillets
    # face the windows: the outer uprights' outer faces are the frame's
    # ends
    fillet = FILLET if round(FILLET / size) >= 2 else 0.0
    ends = (-LENGTH / 2, LENGTH / 2)

    def filleted(cx: float, side: int) -> bool:
        return ends[0] < cx + side * half < ends[1]

    edges = sorted(set(ends)
                   | {cx + side * (half + reach) for cx in UPRIGHTS
                      for side in (-1, 1)
                      for reach in ((0.0, fillet) if filleted(cx, side)
                                    else (0.0,))})
    for cy in RAILS:
        for start, stop in pairwise(edges):
            parts.append(mesh.block((start, cy - half, -THICKNESS),
                                    (stop - start) * x, MEMBER * y,
                                    THICKNESS * z, size, 'frame', unit='in',
                                    holes=holes))
    inner = WIDTH / 2 - MEMBER
    for cx in UPRIGHTS:
        for start, stop in ((-inner, -inner + fillet),
                            (-inner + fillet, inner - fillet),
                            (inner - fillet, inner)):
            if stop <= start:
                continue
            parts.append(mesh.block((cx - half, start, -THICKNESS),
                                    MEMBER * x, (stop - start) * y,
                                    THICKNESS * z, size, 'frame', unit='in',
                                    holes=holes))
        for side in (-1, 1) if fillet else ():
            if not filleted(cx, side):
                continue
            face = cx + side * half
            for rail_side in (-1, 1):
                rail_face = rail_side * inner
                # the fillet block fills the corner between the upright's
                # face and the rail's; its arc's center is the block's
                # far corner, and the void of the arc's radius about it
                # leaves the fillet
                corner = (min(face, face + side * FILLET),
                          min(rail_face, rail_face - rail_side * FILLET),
                          -THICKNESS)
                arc = (face + side * FILLET, rail_face - rail_side * FILLET,
                       top)
                parts.append(mesh.block(corner, FILLET * x, FILLET * y,
                                        THICKNESS * z, size, 'frame',
                                        unit='in',
                                        holes=[(arc, FILLET, 2, None, None)]))
    return parts


def _wing_block(wing: str, size: float) -> Any:
    x, y, z = np.eye(3)
    thickness = WING_THICKNESS[wing]
    holes = [((cx, cy, WING_GAP), WING_HOLE_RADIUS, 2, None, None)
             for cx, cy in WING_HOLES]
    return mesh.block((WING_X, -WING_SPAN / 2, WING_GAP), WING_CHORD * x,
                      WING_SPAN * y, thickness * z, size, wing, unit='in',
                      holes=holes)


def geometry(wing: str | None = None, size: float = SIZE,
             wing_size: float = WING_SIZE) -> Any:
    """The frame, a wing, or the frame with a wing screwed on, as a
    geometry of bricks and rigid links, every element group given what it is
    made of — ready for `fem.Model.from_geometry`, or for the app's
    Solve Modes.

    Parameters
    ----------
    wing : {None, 'thin wing', 'thick wing', 'frame'}, optional
        None or 'frame' for the frame alone; a wing's name for the
        frame with that wing; 'thin wing alone' or 'thick wing alone'
        for the wing by itself.
    size : float
        The frame's element size aimed at, in inches.
    wing_size : float
        The wing's, in inches.

    Returns
    -------
    Geometry
        Element groups 'frame' (6061-T6), 'inserts' (the insert material), the
        wing (6061-T6) and 'screws' (rigid links over the bricks each
        washer covers), opening on `VIEW`.
    """
    choice = (wing or 'frame').strip()
    alone = choice.endswith(' alone')
    part = choice.removesuffix(' alone')
    if part not in ('frame', *WINGS):
        raise ValueError(f'{wing!r}: the frame, {" or ".join(WINGS)}, on '
                         'the frame or alone')
    parts = []
    if part == 'frame' or not alone:
        parts.extend(_frame_blocks(size))
    if part != 'frame':
        parts.append(_wing_block(part, wing_size))
    whole = mesh.assemble(*parts)
    whole.view = VIEW
    materials = {'frame': ALUMINUM, 'inserts': INSERT,
                 **{name: ALUMINUM for name in WINGS}}
    whole.group_properties = {
        int(group): fem.GroupProperties(materials[whole.group_name[i]])
        for i, group in enumerate(whole.group_id)}
    if part != 'frame' and not alone:
        for screw in SCREWS:
            mesh.tie(whole, washer_patch(whole, part, screw), 'frame',
                     group='screws')
    return whole


def washer_patch(geometry: Any, wing: str, screw: tuple) -> list[int]:
    """The wing's bricks a screw's washer covers: those of the wing's
    element group whose centers lie within `WASHER_RADIUS` of the screw's axis,
    the full thickness of the wing — a rigid plug where the screw and
    its washers clamp the plate, tied to the frame under it.

    Parameters
    ----------
    geometry : Geometry
        The frame and wing, assembled.
    wing : str
        The wing's element group name.
    screw : tuple
        (x, y) of the screw, inches.

    Returns
    -------
    list of int
        The element ids.
    """
    xyz = geometry.node_xyz / INCH
    rows = geometry.node_index
    patch = []
    for element in geometry.elements_in(wing):
        row = int(np.flatnonzero(geometry.elem_id == element)[0])
        center = xyz[rows(geometry.elem_conn[row])].mean(axis=0)
        if np.hypot(center[0] - screw[0], center[1] - screw[1]) <= WASHER_RADIUS:
            patch.append(element)
    return patch


def build(wing: str | None = None, size: float = SIZE,
          wing_size: float = WING_SIZE) -> fem.Model:
    """The finite element model (`geometry` built), the screws' mass
    lumped at the frame under each.

    Parameters
    ----------
    wing, size, wing_size
        As for `geometry`.

    Returns
    -------
    fem.Model
    """
    part = geometry(wing, size, wing_size)
    name = (wing or 'frame').strip()
    model = fem.Model.from_geometry(part, name=f'Four-Unit Frame: {name}')
    if name in WINGS:
        xyz = part.node_xyz / INCH
        for screw in SCREWS:
            # the nearest frame node to the screw on the top face
            frame_nodes = np.unique(np.concatenate(
                [part.elem_conn[int(np.flatnonzero(part.elem_id == e)[0])]
                 for e in part.elements_in('inserts')]))
            at = xyz[part.node_index(frame_nodes)]
            nearest = int(frame_nodes[np.argmin(
                np.hypot(at[:, 0] - screw[0], at[:, 1] - screw[1]) + np.abs(at[:, 2]))])
            model.add_mass(nearest, SCREW_MASS, name='screw')
    return model


#: how far the solved example projects solve, Hz
SOLVE_TO = 1500.0


def project(solved: bool = False, size: float = SIZE,
            wing_size: float = WING_SIZE) -> Any:
    """A project to open in the app: the frame, each wing alone, and
    the frame with each wing, their element groups given their properties —
    Solve Modes on any of them gives its modes.

        frame.project().save('frame.vdyn')

    With `solved`, every model's modes are solved to `SOLVE_TO`
    already: the example project the downloads page offers.

    Parameters
    ----------
    solved : bool, default False
        Solve the modes as well.
    size, wing_size : float
        The element sizes aimed at, in inches.

    Returns
    -------
    Project
    """
    from visualdynamics.project import Project

    out = Project('Four-Unit Frame')
    for name in ('frame', *(f'{w} alone' for w in WINGS), *WINGS):
        label = {'frame': 'Frame'}.get(name, name.replace(' alone', '').title()
                                       + ('' if name.endswith('alone') else ' on Frame'))
        out.add(label, geometry(name, size, wing_size))
        if solved:
            out.solve_modes(label, maximum_frequency=SOLVE_TO)
    return out


def describe(size: float = SIZE, modes: int = 9) -> None:
    """Print each model and its first elastic modes."""
    for name in ('frame', 'thin wing alone', 'thick wing alone', *WINGS):
        model = build(name, size)
        shapes = model.eigensolution(maximum_frequency=SOLVE_TO)
        print(f'{name}: {model.num_nodes} nodes, {len(model.solids)} bricks, '
              f'{len(model.rigid_links)} rigid links, '
              f'{model.total_mass * 1000:.1f} g')
        elastic = [f for f in shapes.frequency if f > 1.0][:modes]
        print('  elastic modes (Hz): ' + ', '.join(f'{f:.1f}' for f in elastic))


if __name__ == '__main__':
    describe()
