"""Discrete springs on `fem.Model` (2026-10-08, proposed for two-beam
substructuring cases): a spring between two degrees of freedom, or
from one to ground, written the way `fixed` writes a degree of
freedom, assembled with every other element."""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_fem import ALUMINUM

from visualdynamics.core.fem import Model, Section

ROD = Section.rod('rod', 0.01)
#: every direction but X, on the nodes of a spring-and-mass chain
NOT_X = ('Y+', 'Z+', 'RX+', 'RY+', 'RZ+')


def _held(*nodes):
    return [f'{node}{d}' for node in nodes for d in NOT_X]


def test_two_masses_on_a_spring_ring_at_the_textbook_frequency():
    """Two masses m on a spring k: one mode at 0 Hz and one at
    sqrt(2k/m)/2pi. Grounded instead, one mass rings at sqrt(k/m)/2pi."""
    m, k = 2.0, 8.0e4
    model = Model()
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 0.0, 0.0, 0.0)          # coincident: no length needed
    for node in (1, 2):
        model.add_mass(node, m)
    model.add_spring('1X+', '2X+', k)
    frequency = model.eigensolution(fixed=_held(1, 2)).frequency
    assert frequency == pytest.approx(
        [0.0, math.sqrt(2 * k / m) / (2 * math.pi)], abs=1e-6)
    alone = Model()
    alone.add_node(1, 0.0, 0.0, 0.0)
    alone.add_mass(1, m)
    alone.add_spring('1X+', None, k)
    assert alone.eigensolution(fixed=_held(1)).frequency == pytest.approx(
        [math.sqrt(k / m) / (2 * math.pi)])


def test_the_sign_turns_an_end_around():
    """The spring resists the difference of its ends, each along its own
    sign: X+ to X+ couples by -k, X+ to X- by +k."""
    for second, coupling in (('2X+', -1.0), ('2X-', 1.0)):
        model = Model()
        model.add_node(1, 0.0, 0.0, 0.0)
        model.add_node(2, 1.0, 0.0, 0.0)
        model.add_spring('1X+', second, 5.0)
        _mass, stiffness = model.matrices()
        assert stiffness[0, 0] == stiffness[6, 6] == 5.0
        assert stiffness[0, 6] == stiffness[6, 0] == coupling * 5.0
        assert np.count_nonzero(stiffness) == 4


def _two_beams(joints, stiffness=1e8, along=8):
    """A beam along x in two halves whose meeting nodes coincide, joined
    by a spring in each of `joints` — the two-beam cases' DUT and
    fixture, made one beam by springs stiff enough. 1e8 is five orders
    over the rod's EI/L and within 3e-5 of the whole beam; at 1e12 the
    dense solve lost 0.2 % to the conditioning, which is what a joint
    meant to be rigid is a rigid link for."""
    model = Model()
    xs = np.linspace(0.0, 1.0, along + 1)
    half = along // 2
    for i, x in enumerate(xs[:half + 1]):
        model.add_node(100 + i, x, 0.0, 0.0)
    for i, x in enumerate(xs[half:]):
        model.add_node(200 + i, x, 0.0, 0.0)
    for i in range(half):
        model.add_beam(100 + i, 101 + i, ALUMINUM, ROD)
        model.add_beam(200 + i, 201 + i, ALUMINUM, ROD)
    for direction in joints:
        model.add_spring(f'{100 + half}{direction}', f'200{direction}',
                         stiffness)
    return model


def _whole_beam(along=8):
    model = Model()
    for i, x in enumerate(np.linspace(0.0, 1.0, along + 1)):
        model.add_node(i + 1, x, 0.0, 0.0)
    for i in range(along):
        model.add_beam(i + 1, i + 2, ALUMINUM, ROD)
    return model


def test_stiff_springs_at_coincident_nodes_make_one_beam():
    """Six stiff springs at the joint are the continuous beam; leave out
    the rotation about y and the joint is a hinge in x-z bending. The
    beam's first bending mode in that plane, whose curvature peaks at
    the joint, becomes the hinge's mechanism at 0 Hz, and of the two
    first bending modes (x-y and x-z, 45 Hz each) one is left."""
    six = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')
    joined = _two_beams(six)
    assert len(joined.pieces()) == 1, 'springs join what they connect'
    whole = _whole_beam().eigensolution(num_modes=12).frequency
    sprung = joined.eigensolution(num_modes=12).frequency
    assert sprung == pytest.approx(whole, rel=1e-4, abs=1e-6)
    hinged = _two_beams([d for d in six if d != 'RY+']).eigensolution(
        num_modes=12).frequency
    assert int(np.sum(hinged == 0.0)) == 7, 'a hinge is one more mechanism'
    first = whole[6]
    assert int(np.sum(np.isclose(whole, first, rtol=0.01))) == 2
    assert int(np.sum(np.isclose(hinged, first, rtol=0.01))) == 1


def test_grounding_springs_at_the_ends_hold_the_beam_in_z():
    """A Z spring at each end of a free beam along x takes away its Z
    translation and its rotation about y: four rigid-body modes, not
    six, and two new ones on the springs."""
    model = _whole_beam()
    model.add_spring('1Z+', None, 1e4)
    model.add_spring('9Z+', None, 1e4)
    frequency = model.eigensolution(num_modes=12).frequency
    assert int(np.sum(frequency == 0.0)) == 4
    assert frequency[4] > 0.0


def test_dense_and_sparse_agree_on_springs():
    model = _two_beams(('X+', 'Y+', 'Z+', 'RX+', 'RZ+'), stiffness=1e5)
    model.add_spring('100Z+', None, 3e3)
    dense = model.eigensolution(num_modes=14, solver='dense').frequency
    sparse = model.eigensolution(num_modes=14, solver='sparse').frequency
    assert sparse == pytest.approx(dense, rel=1e-7, abs=1e-6)


def test_a_spring_is_refused_by_what_is_wrong_with_it():
    model = _whole_beam()
    for args, said in (
            (('99X+', None, 1.0), 'does not name a node'),
            (('1', None, 1.0), 'no direction'),
            (('1Q+', None, 1.0), 'not a direction'),
            (('1X+', '2RX+', 1.0), 'two translations or two rotations'),
            (('1X+', '1X-', 1.0), 'one degree of freedom'),
            (('1X+', '2X+', 0.0), 'stiffness is positive'),
            (('1X+', '2X+', -5.0), 'stiffness is positive')):
        with pytest.raises(ValueError, match=said):
            model.add_spring(*args)
    assert not model.springs, 'a refusal adds nothing'


def test_a_rotational_spring_on_a_node_only_solids_touch_is_refused():
    """A solid gives a rotation nothing to act on, so the solve grounds
    the rotations of a node only solids touch, and a spring on one would
    be grounded with it, silently."""
    from test_fem_solids import _bar_of_bricks

    model = _bar_of_bricks(along=2)
    model.add_spring('1RZ+', None, 1e3)
    with pytest.raises(ValueError, match='spring acts on a rotation of node 1'):
        model.eigensolution(num_modes=6)


def test_the_progress_bar_counts_the_springs():
    """The sparse assembly ticks once per element batch it scatters,
    springs among them; counted in the total too, the bar never runs
    past its end."""
    model = _two_beams(('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+'))
    told = []
    model.eigensolution(num_modes=8, solver='sparse',
                        progress=lambda done, total: told.append((done, total)))
    assert told and all(done <= total for done, total in told), told
