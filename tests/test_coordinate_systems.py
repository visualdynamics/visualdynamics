"""Drawing a coordinate system: named directions, angles as arcs."""

import numpy as np
import pytest
import pyvista as pv

from visualdynamics.viz.geometry import (
    ARC_RADIUS,
    ARROW_HEAD_LENGTH,
    ARROW_HEAD_RADIUS,
    ARROW_SHAFT_RADIUS,
    CS_ARCS,
    CS_AXIS_LABELS,
    _arc_points,
    add_coordinate_system,
)


def identity_frame(origin=(0.0, 0.0, 0.0)):
    matrix = np.zeros((4, 3))
    matrix[:3] = np.eye(3)
    matrix[3] = origin
    return matrix


def test_each_type_names_its_own_directions():
    assert CS_AXIS_LABELS[0] == ('$X$', '$Y$', '$Z$')
    assert CS_AXIS_LABELS[1][0] == '$R$' and CS_AXIS_LABELS[1][2] == '$Z$'
    assert 'theta' in CS_AXIS_LABELS[1][1]
    assert 'theta' in CS_AXIS_LABELS[2][1] and 'phi' in CS_AXIS_LABELS[2][2]


def test_only_the_angles_are_arcs():
    assert 0 not in CS_ARCS, 'a cartesian system is three straight arrows'
    assert set(CS_ARCS[1]) == {1}, 'cylindrical: theta only'
    assert set(CS_ARCS[2]) == {1, 2}, 'spherical: theta and phi; R is radial'


def test_a_cylindrical_theta_sweeps_from_r_towards_y():
    matrix = identity_frame()
    start, about = (matrix[i] for i in CS_ARCS[1][1])
    points = _arc_points(matrix[3], start, about, 1.0)
    assert np.allclose(points[0], [1, 0, 0]), 'starts along R'
    assert points[-1][1] > 0.5, 'turns towards +Y'
    assert np.allclose(points[:, 2], 0.0), 'stays in the R-theta plane'


def test_a_spherical_theta_sweeps_down_from_z():
    """The polar angle is measured from Z, so its arc starts there."""
    matrix = identity_frame()
    start, about = (matrix[i] for i in CS_ARCS[2][1])
    points = _arc_points(matrix[3], start, about, 1.0)
    assert np.allclose(points[0], [0, 0, 1]), 'starts along Z'
    assert points[-1][0] > 0.5, 'turns towards +X'
    assert np.allclose(points[:, 1], 0.0), 'stays in the Z-X plane'


def test_a_spherical_phi_sweeps_round_the_equator():
    matrix = identity_frame()
    start, about = (matrix[i] for i in CS_ARCS[2][2])
    points = _arc_points(matrix[3], start, about, 1.0)
    assert np.allclose(points[0], [1, 0, 0])
    assert points[-1][1] > 0.5
    assert np.allclose(points[:, 2], 0.0)


def test_arcs_are_drawn_inside_the_straight_arrows():
    """Or the spherical theta head lands on R's, and the labels collide."""
    matrix = identity_frame()
    start, about = (matrix[i] for i in CS_ARCS[2][1])
    points = _arc_points(matrix[3], start, about, ARC_RADIUS)
    assert np.linalg.norm(points[-1]) == pytest.approx(ARC_RADIUS)
    assert ARC_RADIUS < 1.0


def _label_actors(plotter):
    return [name for name in plotter.renderer.actors if 'label' in name.lower()]


def test_a_system_is_named_only_when_it_is_singled_out():
    plotter = pv.Plotter(off_screen=True)
    add_coordinate_system(plotter, identity_frame()[3], identity_frame(), 1,
                          1.0, label=None)
    assert not _label_actors(plotter), 'no text over an unselected system'

    add_coordinate_system(plotter, identity_frame()[3], identity_frame(), 1,
                          1.0, label=7)
    assert _label_actors(plotter), 'the selected one is named'
    plotter.close()


def test_an_unknown_type_falls_back_to_x_y_z():
    """A type this drawer has no names for is drawn with the cartesian
    ones rather than refused — the arrows are still where they are."""
    plotter = pv.Plotter(off_screen=True)
    written = []
    real = plotter.add_point_labels

    def record(points, labels, **kwargs):
        written.append(list(labels))
        return real(points, labels, **kwargs)

    plotter.add_point_labels = record
    add_coordinate_system(plotter, identity_frame()[3], identity_frame(), 99,
                          1.0, label=1)
    plotter.close()
    assert ['$X$', '$Y$', '$Z$'] in written, written


def test_arc_proportions_track_pyvistas_own_arrow():
    """An arc is drawn as a tube and a cone, sized like a pv.Arrow.

    As a plain polyline it came out visibly thinner than the straight
    arrows beside it, head included. If pyvista restyles its arrow, this
    fails rather than the two quietly drifting apart.
    """
    import inspect

    defaults = inspect.signature(pv.Arrow).parameters
    assert ARROW_SHAFT_RADIUS == defaults['shaft_radius'].default
    assert ARROW_HEAD_LENGTH == defaults['tip_length'].default
    assert ARROW_HEAD_RADIUS == defaults['tip_radius'].default


def test_a_curved_direction_is_solid_geometry():
    """A cylindrical system draws more than a cartesian one: its angle is a
    tube plus a cone where a straight direction is one arrow."""
    straight = pv.Plotter(off_screen=True)
    add_coordinate_system(straight, identity_frame()[3], identity_frame(),
                          0, 1.0)
    curved = pv.Plotter(off_screen=True)
    add_coordinate_system(curved, identity_frame()[3], identity_frame(),
                          1, 1.0)
    assert len(curved.renderer.actors) > len(straight.renderer.actors)
    straight.close()
    curved.close()
