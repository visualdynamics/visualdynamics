"""Turning a coordinate system: the geometry, without a window."""

import numpy as np
import pytest

from visualdynamics.rotate import (
    angle_in_plane,
    identity_frame,
    plane_hit,
    ring_points,
    ring_under_cursor,
    rotate_frame,
    rotation_about,
    wrapped,
)


def frame(origin=(0.0, 0.0, 0.0)):
    matrix = np.zeros((4, 3))
    matrix[:3] = np.eye(3)
    matrix[3] = origin
    return matrix


def test_a_quarter_turn_about_z_sends_x_to_y():
    turned = rotate_frame(frame(), 2, np.pi / 2)
    assert np.allclose(turned[0], [0, 1, 0], atol=1e-12)
    assert np.allclose(turned[1], [-1, 0, 0], atol=1e-12)
    assert np.allclose(turned[2], [0, 0, 1], atol=1e-12)


def test_rotation_leaves_the_origin_alone():
    turned = rotate_frame(frame((3.0, -1.0, 2.0)), 0, 0.7)
    assert np.allclose(turned[3], [3.0, -1.0, 2.0])


def test_the_frame_stays_orthonormal_after_many_turns():
    matrix = frame()
    for _ in range(200):
        matrix = rotate_frame(matrix, 1, 0.37)
    basis = matrix[:3]
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)


def test_the_axis_travels_with_the_frame():
    """Two half turns about the frame's own X is one full turn about it."""
    once = rotate_frame(rotate_frame(frame(), 0, np.pi / 4), 0, np.pi / 4)
    straight = rotate_frame(frame(), 0, np.pi / 2)
    assert np.allclose(once[:3], straight[:3], atol=1e-12)


def test_a_ring_circles_its_own_axis():
    points = ring_points(frame(), 2, radius=2.0)
    assert np.allclose(points[:, 2], 0.0), 'the Z ring lies in the X-Y plane'
    assert np.allclose(np.linalg.norm(points, axis=1), 2.0)
    assert len(points) > 8


def test_a_ring_follows_a_turned_frame():
    turned = rotate_frame(frame(), 0, np.pi / 2)   # Y and Z swap places
    points = ring_points(turned, 2, radius=1.0)
    # the ring about the frame's Z now lies in the world's X-Z plane
    assert np.allclose(points[:, 1], 0.0, atol=1e-12)


def test_the_angle_of_a_point_matches_where_the_ring_put_it():
    matrix = frame((1.0, 2.0, 3.0))
    for axis in (0, 1, 2):
        points = ring_points(matrix, axis, radius=1.5, steps=8)
        for i, point in enumerate(points):
            # compared as angles: half a turn reads as +pi or -pi, the same
            # place on the ring either way
            difference = wrapped(angle_in_plane(matrix, axis, point)
                                 - i * 2 * np.pi / 8)
            assert difference == pytest.approx(0.0, abs=1e-9)


def test_dragging_by_an_angle_turns_the_frame_by_that_angle():
    """What the drag measures and what the rotation applies must agree."""
    matrix = frame()
    start = ring_points(matrix, 2, radius=1.0, steps=8)[0]
    moved = ring_points(matrix, 2, radius=1.0, steps=8)[1]
    swept = wrapped(angle_in_plane(matrix, 2, moved)
                    - angle_in_plane(matrix, 2, start))
    turned = rotate_frame(matrix, 2, swept)
    assert angle_in_plane(turned, 2, moved) == pytest.approx(
        angle_in_plane(matrix, 2, start), abs=1e-9)


def test_wrapping_keeps_a_drag_past_the_seam_small():
    assert wrapped(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrapped(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert wrapped(0.3) == pytest.approx(0.3)


def test_the_nearest_ring_wins():
    rings = [(0, np.array([[100.0, 100.0]])), (2, np.array([[10.0, 10.0]]))]
    assert ring_under_cursor(rings, (12.0, 12.0)) == 2
    assert ring_under_cursor(rings, (102.0, 98.0)) == 0


def test_a_cursor_far_from_every_ring_picks_nothing():
    rings = [(0, np.array([[100.0, 100.0]]))]
    assert ring_under_cursor(rings, (300.0, 300.0)) is None


def test_a_ray_meets_the_plane_it_crosses():
    hit = plane_hit(origin=(0, 0, 0), normal=(0, 0, 1),
                    eye=(0, 0, 5), direction=(0, 0, -1))
    assert np.allclose(hit, [0, 0, 0])


def test_a_ray_along_the_plane_never_meets_it():
    assert plane_hit(origin=(0, 0, 0), normal=(0, 0, 1),
                     eye=(0, 0, 5), direction=(1, 0, 0)) is None


def test_resetting_keeps_the_origin_but_drops_the_turn():
    matrix = rotate_frame(frame((4.0, 5.0, 6.0)), 1, 1.1)
    reset = identity_frame(matrix)
    assert np.allclose(reset[:3], np.eye(3))
    assert np.allclose(reset[3], [4.0, 5.0, 6.0])


def test_rotation_about_is_a_proper_rotation():
    matrix = rotation_about((1.0, 2.0, 3.0), 0.9)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)


# ---- sliding (2026-10-02) ---------------------------------------------------


def _frame():
    from visualdynamics.rotate import rotate_frame

    plain = np.vstack([np.eye(3), [1.0, 2.0, 3.0]])
    return rotate_frame(plain, 2, np.radians(30.0))


def test_sliding_moves_the_origin_along_the_frames_own_axis():
    from visualdynamics.rotate import translate_frame

    frame = _frame()
    moved = translate_frame(frame, 0, 2.5)
    assert np.allclose(moved[:3], frame[:3]), 'the basis stays put'
    assert np.allclose(moved[3], frame[3] + 2.5 * frame[0])


def test_the_arrow_runs_from_the_origin_along_the_axis():
    from visualdynamics.rotate import arrow_points

    frame = _frame()
    points = arrow_points(frame, 1, 4.0)
    assert np.allclose(points[0], frame[3])
    assert np.allclose(points[-1], frame[3] + 4.0 * frame[1])
    assert len(points) == 24


def test_a_ray_crossing_the_axis_hits_where_it_crosses():
    from visualdynamics.rotate import axis_hit, distance_along

    frame = _frame()
    target = frame[3] + 3.0 * frame[0]
    eye = target + np.array([0.0, 0.0, 10.0])
    hit = axis_hit(frame[3], frame[0], eye, target - eye)
    assert hit is not None and np.allclose(hit, target)
    assert distance_along(frame, 0, hit) == pytest.approx(3.0)
    # a ray that misses the line still reads the nearest point on it
    offset = axis_hit(frame[3], frame[0], eye + 0.3 * frame[1], target - eye)
    assert np.allclose(offset, target)


def test_a_ray_parallel_to_the_axis_hits_nothing():
    from visualdynamics.rotate import axis_hit

    frame = _frame()
    assert axis_hit(frame[3], frame[0], frame[3] + frame[1], frame[0]) is None


def test_snapping_lands_on_the_grid_in_every_coordinate():
    from visualdynamics.rotate import snapped

    assert np.allclose(snapped([1.26, -0.04, 2.56], 0.1), [1.3, 0.0, 2.6])
    assert np.allclose(snapped([1.26, -0.04, 2.56], 0.0), [1.26, -0.04, 2.56])


def test_the_grid_is_a_tenth_of_an_inch_or_a_centimetre():
    from visualdynamics.rotate import grid_step

    assert grid_step('in') == pytest.approx(0.1)
    assert grid_step('ft') == pytest.approx(0.1 / 12.0)
    assert grid_step('mm') == pytest.approx(10.0)
    assert grid_step('cm') == pytest.approx(1.0)
    assert grid_step('m') == pytest.approx(0.01)


# ---- angles (2026-10-02) ----------------------------------------------------


def test_angles_turn_about_the_fixed_axes_x_then_y_then_z():
    from visualdynamics.rotate import frame_from_angles

    frame = frame_from_angles((90.0, 0.0, 0.0))
    assert np.allclose(frame[1], [0, 0, 1]) and np.allclose(frame[2], [0, -1, 0])
    # X then Z: the frame's X axis is carried round Z, its Y up first
    frame = frame_from_angles((90.0, 0.0, 90.0), origin=(1, 2, 3))
    assert np.allclose(frame[0], [0, 1, 0]) and np.allclose(frame[1], [0, 0, 1])
    assert np.allclose(frame[3], [1, 2, 3])


def test_the_angles_of_a_frame_undo_the_turn():
    from visualdynamics.rotate import angles_of, frame_from_angles

    for angles in [(30.0, -20.0, 75.0), (-170.0, 45.0, 10.0), (0.0, 0.0, -90.0),
                   (12.5, 0.0, 0.0)]:
        assert angles_of(frame_from_angles(angles)) == pytest.approx(angles)
    # a quarter turn about Y: X and Z are one turn, said as Z
    frame = frame_from_angles((25.0, 90.0, 40.0))
    back = angles_of(frame)
    assert back[0] == 0.0 and back[1] == pytest.approx(90.0)
    assert np.allclose(frame_from_angles(back), frame)
