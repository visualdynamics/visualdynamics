"""The gizmo's geometry, pinned by value where a mutation pass found it
unguarded (2026-10-11).

`tools/mutate.py` over rotate.py caught 97 of 111 against
test_rotate.py. The ray-plane and ray-axis meetings were checked
through drags that land within a tolerance; here they are checked at
points worked out by hand.
"""

from __future__ import annotations

import math

import numpy as np

from visualdynamics.rotate import (
    angles_of,
    axis_hit,
    frame_from_angles,
    plane_hit,
    ring_under_cursor,
)


def test_a_ray_meets_a_plane_where_it_crosses_it():
    hit = plane_hit((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (3.0, 0.0, 5.0), (0.0, 0.0, -2.0))
    assert hit.tolist() == [3.0, 0.0, 1.0]
    assert plane_hit((0, 0, 1), (0, 0, 1), (0, 0, 5), (1.0, 0.0, 0.0)) is None


def test_a_ray_comes_nearest_an_axis_at_its_foot():
    """The X axis through the origin, a ray straight down from (2, 5, 1):
    the closest approach is at x = 2 on the axis — however long the
    axis's direction vector is."""
    for length in (1.0, 1e-4):
        hit = axis_hit((0.0, 0.0, 0.0), (length, 0.0, 0.0), (2.0, 5.0, 1.0),
                       (0.0, -1.0, 0.0))
        assert np.allclose(hit, [2.0, 0.0, 0.0]), length
        # slanting along the axis too: from (0, 5, 1) heading (1, -1, 0)
        # the ray passes over the axis at x = 5
        slant = axis_hit((0.0, 0.0, 0.0), (length, 0.0, 0.0), (0.0, 5.0, 1.0),
                         (1.0, -1.0, 0.0))
        assert np.allclose(slant, [5.0, 0.0, 0.0]), length
    assert axis_hit((0, 0, 0), (1.0, 0, 0), (0, 5, 0), (3.0, 0, 0)) is None


def test_a_frame_is_four_rows_and_its_angles_read_back():
    frame = frame_from_angles((10.0, 20.0, 30.0), origin=(1.0, 2.0, 3.0))
    assert frame.shape == (4, 3) and frame[3].tolist() == [1.0, 2.0, 3.0]
    assert angles_of(frame) == (10.0, 20.0, 30.0)
    zero = angles_of(frame_from_angles((0.0, 0.0, 0.0)))
    assert zero == (0.0, 0.0, 0.0)
    assert all(math.copysign(1.0, a) == 1.0 for a in zero), 'never a negative zero'


def test_rings_equally_near_go_to_the_first():
    ring = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert ring_under_cursor([(0, ring), (1, ring)], (5.0, 3.0)) == 0
    assert ring_under_cursor([(0, ring + [0, 4.0]), (1, ring)], (5.0, 3.0)) == 0
    assert ring_under_cursor([(0, ring)], (5.0, 30.0)) is None
