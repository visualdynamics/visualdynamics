"""The demonstration airframe, pinned where it matters.

**Marked `slow`, and deselected by default.** This is the model the
website, the documentation and the examples are built from, and it is
checked when those are built or on demand (`pytest -m slow`) — not on
every edit. Even coarsened it is a 6732-degree-of-freedom eigenproblem,
and `np.linalg.eigh` on that is forty-six seconds no matter what
`maximum_frequency` is asked for, since the ceiling cuts the answer and
not the work. That one solve was most of the suite's wall clock.

What is left here is what is *about the drone*: that it is one connected
structure, that it weighs what it claims, that its sensor set is found
rather than written down, and that it has a rich set of modes in the band
a test would use. The rules that are about the toolset rather than this
model — that a member takes its section from its element's block, that a
node on a seam goes to the block holding most of it, that a saved
geometry alone rebuilds the model — moved to `test_fem.py`, where a
three-quad strip states them in a tenth of a second.

Loose bounds on purpose. These are not regression hashes; they are the
statements that would have to stay true for the demonstration to still be
the demonstration, and a tolerance tight enough to fail on a mesh change
would only teach people to update the number.

Built coarse. The shipped default is denser still, and none of these
claims are about the mesh density.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.demo import drone as drone_model

pytestmark = pytest.mark.slow

COARSE = {'sides': 8, 'arm_stations': 5, 'leg_stations': 3, 'body_rings': 2}


@pytest.fixture(scope='module')
def model():
    return drone_model.build(**COARSE)


@pytest.fixture(scope='module')
def shapes(model):
    return model.eigensolution(maximum_frequency=4000.0, damping=0.01)


def test_it_is_one_connected_structure(model):
    """The failure this model has actually had. Parts drawn touching are
    not joined until something joins them, and the first full build came
    back as 34 free bodies — 24 truss struts, 4 nacelles, 4 legs and 2
    camera mounts — which is 204 zero-frequency modes on top of the six
    real ones."""
    pieces = model.pieces()
    assert len(pieces) == 1, (
        f'{len(pieces)} disconnected pieces, sizes '
        f'{[len(p) for p in pieces][:8]}')


def test_it_is_free_free(shapes):
    """Hung on bungees, the way a modal test of one would be: six rigid
    modes at exactly zero, and nothing else at zero."""
    assert list(shapes.frequency[:6]) == [0.0] * 6
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    assert shapes.frequency[6] > 1.0


def test_it_weighs_what_it_says(model):
    """Every node carries an equal share, and they sum to the total."""
    assert model.total_mass == pytest.approx(drone_model.TOTAL_MASS)
    # the members are massless: the mass is at the nodes, all of it
    assert model.structural_mass == 0.0


def test_the_geometry_is_the_model(model):
    """No node is drawn that is not solved, and no node is solved that is
    not drawn. The first design had surfaces hung off a beam skeleton on
    rigid ties, which looked like a fidelity it did not have."""
    drawn = model.geometry(beams=False)
    assert len(drawn.node_id) == model.num_nodes
    assert model.num_dof == 6 * model.num_nodes
    assert len(drawn.elem_id) == len(model.faces)


def test_it_is_drawn_in_surfaces_only(model):
    """Quads and triangles, no lines: every member runs inside a surface,
    so drawing them would lay a wireframe over the thing they are in."""
    drawn = model.geometry(beams=False)
    assert {int(t) for t in drawn.elem_type} <= {41, 44}
    assert not drawn.drawn_lines()
    assert len(drawn.elem_id) > 100


def test_every_face_edge_carries_a_member(model):
    """The conversion that makes the drawing structural."""
    edges = set()
    for face in model.faces:
        for k, node in enumerate(face.nodes):
            edges.add(frozenset((node, face.nodes[(k + 1) % len(face.nodes)])))
    wired = {frozenset((b.node_a, b.node_b)) for b in model.beams}
    assert edges <= wired, f'{len(edges - wired)} face edges have no member'


def test_the_modes_are_dense_under_two_kilohertz(shapes):
    """What the airframe is sized for. The member section scales the whole
    spectrum together, so what it chooses is how many modes land under a
    fixed ceiling — not where any particular one sits.

    Read off the shared solve rather than asking for another:
    `maximum_frequency` cuts the answer after the eigensolution, not the
    work before it, so a second call at a lower ceiling costs the same
    forty-six seconds and returns a subset of what is already here.
    """
    elastic = [f for f in shapes.frequency if 0 < f <= 2000.0]
    assert len(elastic) > 15, f'only {len(elastic)} modes under 2 kHz'
    # and they are spread rather than piled into one corner of the band
    assert min(elastic) < 700.0
    assert max(elastic) > 1200.0


def test_the_airframe_has_the_parts_it_claims(model):
    """Each part is found by name, since the sensor set is located from
    them and a rename would quietly empty it."""
    parts = {drone_model.part_of(model, n) for n in model.node_ids}
    assert {'body', 'arm', 'nacelle', 'leg', 'battery', 'camera'} <= parts


def test_the_instrumented_nodes_are_found_not_written_down(model):
    """Node numbering moves whenever the mesh does, and a hard-coded list
    went stale twice — once silently, since a DOF that no longer exists is
    only noticed when something tries to synthesize it."""
    at = drone_model.instrumented(model)
    assert len(at['motors']) == 4
    assert len(at['arms']) == 4
    assert len(at['feet']) == 4
    assert len(at['payload']) == 2
    every = [n for group in at.values() for n in group]
    assert len(every) == len(set(every)), 'a node was instrumented twice'
    assert set(every) <= set(model.node_ids)


def test_the_motors_are_where_the_arms_end(model):
    """The sensor roles have to survive a change of mesh density, so they
    are checked against the geometry rather than against ids."""
    at = drone_model.instrumented(model)
    for node, (_name, angle, radius) in zip(at['motors'], drone_model.ARMS):
        position = np.asarray(model.position(node))
        assert np.linalg.norm(position[:2]) == pytest.approx(radius, rel=0.25)
        assert position[2] > 0.0, 'a motor sits above the arm'


def test_the_mesh_density_is_a_dial(model):
    """The truth model is dense and a plant does not have to be, so the
    same airframe has to be buildable at more than one refinement."""
    finer = drone_model.build(sides=10, arm_stations=6, leg_stations=3,
                              body_rings=2)
    assert finer.num_nodes > model.num_nodes
    assert finer.total_mass == pytest.approx(model.total_mass)
    assert len(finer.pieces()) == 1


def test_a_saved_geometry_alone_rebuilds_the_model(model, tmp_path):
    """The geometry is the whole description: sections come back off the
    element blocks, with nothing passed alongside the file.

    This is what element blocks are for, and it was wrong twice before it
    was right. Both times a part was read off an element's *first node*,
    and a node on a seam between two parts can only answer for one of
    them — 24 of the propeller's blade members came back as ordinary
    frame, moving every mode by a couple of percent. A face belongs to
    exactly one block, so the block is what is asked.
    """
    from visualdynamics.core import fem
    from visualdynamics.io import load, save

    path = tmp_path / 'drone.vdyn'
    save(model.geometry(beams=False), path)
    rebuilt = fem.Model.from_geometry(
        load(path), drone_model.FRAME, drone_model.MEMBER,
        total_mass=drone_model.TOTAL_MASS,
        sections={'prop': drone_model.BLADE})

    def sections(built):
        counted = {}
        for beam in built.beams:
            counted[beam.section.name] = counted.get(beam.section.name, 0) + 1
        return counted

    assert sections(rebuilt) == sections(model)
    assert sections(model).keys() == {'member', 'blade'}
    # The matrices, not the modes: they are what the modes come out of, so
    # equal ones say the same thing about every degree of freedom rather
    # than about the modes under some ceiling — and they cost a second
    # against the ninety-two two eigensolutions of this size cost. The
    # DOF order is checked first, since equal matrices only mean anything
    # while the rows are the same rows.
    assert rebuilt.dof_strings() == model.dof_strings(), (
        'the rebuilt model orders its degrees of freedom differently, so '
        'the matrices below would differ by a permutation and say nothing')
    for name, mine, theirs in zip(('mass', 'stiffness'),
                                  model.matrices(), rebuilt.matrices()):
        assert np.array_equal(mine, theirs), (
            f'the {name} matrix came back different: '
            f'{np.abs(mine - theirs).max():.3g} at worst')


def test_the_blocks_name_the_parts_by_side(model):
    """'arm front left', not 'arm'. The tidier name loses the one thing
    the sensor set is picked by — which of the four arms a node is on —
    and `instrumented` silently returns four empty groups."""
    drawn = model.geometry(beams=False)
    names = set(drawn.group_name)
    assert {'arm front left', 'arm rear right', 'leg front left',
            'nacelle rear left', 'canopy', 'battery', 'camera'} <= names
    assert len(drawn.group_id) == len(names)
