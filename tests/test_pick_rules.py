"""Picking's geometry without a camera, pinned by value where a
mutation pass found it unguarded (2026-10-11).

`tools/mutate.py` over viz/pick.py caught 100 of 150 against
test_pick.py, whose picks go through a real VTK camera at the centers
of things. What a center never tests is an edge: a cursor exactly on a
face's boundary is inside it, whichever way the face winds on screen.
A projector that returns fixed pixels stands in for the camera, so the
pixels are chosen rather than found.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.viz.pick import EntityPicker, ScreenProjector


class _Fixed:
    """A projector whose nodes are already on screen where it says."""

    def __init__(self, screen, depth):
        self._screen = np.asarray(screen, dtype=float)
        self._depth = np.asarray(depth, dtype=float)

    def screen(self):
        return self._screen, self._depth


def _front_triangle_over_a_back_quad(clockwise):
    """A triangle at depth 0.1 in front of a quad at 0.9 that covers the
    whole screen: element 0 the quad, element 1 the triangle."""
    triangle = [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]]
    if clockwise:
        triangle = triangle[::-1]
    screen = [[-100, -100], [100, -100], [100, 100], [-100, 100]] + triangle
    geometry = visualdynamics.Geometry(np.arange(1, 8), np.zeros((7, 3)))
    geometry.add_elements([[1, 2, 3, 4], [5, 6, 7]], [44, 41], [1, 2])
    projector = _Fixed(screen, [0.9] * 4 + [0.1] * 3)
    return EntityPicker(geometry, 'elements', projector)


@pytest.mark.parametrize('clockwise', [False, True])
@pytest.mark.parametrize('cursor', [(10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)])
def test_a_cursor_on_a_faces_edge_is_inside_it(clockwise, cursor):
    picker = _front_triangle_over_a_back_quad(clockwise)
    assert picker.pick(*cursor, tolerance=0.0) == 1


def test_a_cursor_just_off_the_front_face_finds_the_back_one():
    picker = _front_triangle_over_a_back_quad(False)
    assert picker.pick(10.5, 10.5, tolerance=0.0) == 0
    assert picker.pick(10.0, -0.5, tolerance=0.0) == 0


def test_a_point_element_is_picked_where_its_node_is():
    geometry = visualdynamics.Geometry([1, 2, 3], np.zeros((3, 3)))
    geometry.add_elements([[1, 2], [3]], [21, 161], [1, 2])
    picker = EntityPicker(geometry, 'elements',
                          _Fixed([[0.0, 0.0], [10.0, 0.0], [50.0, 50.0]], [0.5] * 3))
    assert picker.pick(51.0, 50.0) == 1
    assert picker.pick(5.0, 1.0) == 0


def test_a_beam_is_picked_by_its_nearer_end_on_a_tie():
    """Two beams the cursor is equally near: the one whose midpoint is
    nearer the camera wins."""
    geometry = visualdynamics.Geometry([1, 2, 3, 4], np.zeros((4, 3)))
    geometry.add_elements([[1, 2], [3, 4]], [21, 21], [1, 1])
    screen = [[0.0, -5.0], [10.0, -5.0], [0.0, 5.0], [10.0, 5.0]]
    near_second = EntityPicker(geometry, 'elements', _Fixed(screen, [0.9, 0.9, 0.2, 0.2]))
    assert near_second.pick(5.0, 0.0) == 1
    near_first = EntityPicker(geometry, 'elements', _Fixed(screen, [0.2, 0.2, 0.9, 0.9]))
    assert near_first.pick(5.0, 0.0) == 0
    split = EntityPicker(geometry, 'elements', _Fixed(screen, [0.0, 0.8, 0.5, 0.5]))
    assert split.pick(5.0, 0.0) == 0, 'midpoint depth 0.4 against 0.5'


@pytest.mark.parametrize('offset', [0.0, 100.0])
def test_a_flat_model_has_its_plane_wherever_it_sits(offset):
    square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.01]]) + offset
    normal = ScreenProjector(None, square)._plane_normal()
    assert normal is not None and abs(normal[2]) > 0.99
    triangle = ScreenProjector(None, square[:3])._plane_normal()
    assert triangle is not None and abs(triangle[2]) == pytest.approx(1.0)
    assert ScreenProjector(None, square[:2])._plane_normal() is None


def test_a_model_thicker_than_a_fiftieth_of_its_width_has_no_plane():
    slab = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 0.1], [1, 0, 0.1], [1, 1, 0.1], [0, 1, 0.1]], float)
    assert ScreenProjector(None, slab)._plane_normal() is None
