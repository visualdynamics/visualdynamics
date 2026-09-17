"""The MAC as 3-D bars: one mesh, heights that are the values, 0..1.

Scene-level claims only; the window's toggle is `test_mac_bars_view`.
"""

import numpy as np

from visualdynamics.viz.mac_bars import (
    FOOTPRINT,
    HEIGHT,
    PLINTH,
    add_mac_bars,
    mac_bars_scene,
)
from visualdynamics.viz.waterfall import LABEL_LIMIT


def scene(matrix, frequencies=None, columns=None):
    import pyvista as pv

    matrix = np.atleast_2d(matrix)
    if frequencies is None:
        frequencies = np.linspace(10.0, 100.0, matrix.shape[0])
    plotter = pv.Plotter(off_screen=True)
    info = add_mac_bars(plotter, frequencies, matrix,
                        column_frequencies=columns)
    return plotter, info


def test_every_cell_is_a_box_on_one_mesh():
    matrix = np.random.default_rng(1).random((5, 3))
    plotter, info = scene(matrix, columns=np.linspace(5.0, 15.0, 3))
    mesh = plotter.renderer.actors['mac-bars'].mapper.dataset
    assert (info['rows'], info['columns']) == (5, 3)
    assert mesh.n_points == 5 * 3 * 8, 'eight corners per box'
    assert mesh.n_cells == 5 * 3 * 6, 'six faces per box'
    plotter.close()


def test_height_and_color_are_both_the_value():
    """The bar for 0.6 must stand at 0.6 of full height exactly, and
    carry 0.6 as its color scalar — the two readings of one number."""
    matrix = np.array([[1.0, 0.6], [0.0, 0.25]])
    plotter, _info = scene(matrix)
    mesh = plotter.renderer.actors['mac-bars'].mapper.dataset
    tall = HEIGHT * 2   # max(rows, columns) == 2
    points = np.asarray(mesh.points).reshape(4, 8, 3)
    tops = points[:, 4:, 2].max(axis=1)
    expected = np.maximum(matrix.ravel(), PLINTH) * tall
    assert np.allclose(tops, expected)
    values = mesh.cell_data['mac'].reshape(4, 6)
    assert np.allclose(values, matrix.ravel()[:, None])
    plotter.close()


def test_the_scale_is_pinned_zero_to_one():
    """Like the flat grid and the coherence map: a MAC is a bounded
    ratio, and a matrix topping out at 0.4 must not be recolored as
    if 0.4 were a match."""
    matrix = np.full((3, 3), 0.4)
    plotter, _info = scene(matrix)
    actor = plotter.renderer.actors['mac-bars']
    assert tuple(actor.mapper.scalar_range) == (0.0, 1.0)
    plotter.close()


def test_bars_leave_a_gap_between_neighbors():
    """A row of near-ones must read as bars, not a wall."""
    assert FOOTPRINT < 1.0
    matrix = np.ones((1, 2))
    plotter, _info = scene(matrix, columns=np.array([5.0, 9.0]))
    mesh = plotter.renderer.actors['mac-bars'].mapper.dataset
    x = np.asarray(mesh.points)[:, 0]
    first_right = x[:8].max()
    second_left = x[8:].min()
    assert second_left > first_right, 'neighboring boxes must not touch'
    plotter.close()


def test_frequency_labels_thin_past_the_limit():
    big = np.eye(3 * LABEL_LIMIT)
    plotter, info = scene(big,
                          frequencies=np.arange(3 * LABEL_LIMIT) + 10.0)
    assert len(info['named_rows']) <= LABEL_LIMIT
    assert 0 in info['named_rows'], 'the first mode anchors the axis'
    plotter.close()


def test_plot_mac_bars_renders_to_a_file(tmp_path):
    from visualdynamics.core.shapes import ShapeSet
    from visualdynamics.plot import plot_mac

    shapes = ShapeSet(np.array([10.0, 20.0, 30.0]),
                      np.array([0.01, 0.01, 0.01]),
                      ['1Z+', '2Z+', '3Z+'], np.eye(3))
    path = tmp_path / 'mac_bars.png'
    img = plot_mac(shapes, bars=True, screenshot=str(path))
    assert path.exists() and img.ndim == 3


def test_a_rectangular_cross_mac_keeps_its_shape():
    matrix = np.random.default_rng(2).random((6, 2))
    plotter, info = scene(matrix, columns=np.array([5.0, 9.0]))
    assert (info['rows'], info['columns']) == (6, 2)
    mesh = plotter.renderer.actors['mac-bars'].mapper.dataset
    xy = np.asarray(mesh.points)
    assert xy[:, 0].max() < 2.0 and xy[:, 1].max() < 6.0
    plotter.close()


def test_scene_camera_sees_row_zero_nearest():
    """The flat grid reads downward from the top-left; the bars put the
    same first mode in front. The camera sits at negative y, so row 0
    is the near row."""
    plotter = mac_bars_scene(np.array([10.0, 20.0]), np.eye(2),
                             off_screen=True)
    position = np.asarray(plotter.camera_position[0])
    assert position[1] < 0, 'the camera looks up the row axis'
    plotter.close()


def test_a_mismatched_frequency_list_is_refused_with_the_reason():
    import pytest

    with pytest.raises(ValueError, match='column frequencies'):
        scene(np.ones((2, 3)))   # rectangular, no column frequencies


def test_cell_to_pair_reads_the_construction_backwards():
    from visualdynamics.viz.mac_bars import cell_to_pair

    columns = 4
    assert cell_to_pair(0, columns) == (0, 0)
    assert cell_to_pair(5, columns) == (0, 0), 'six faces, one box'
    assert cell_to_pair(6, columns) == (0, 1)
    assert cell_to_pair(6 * 4, columns) == (1, 0)
    assert cell_to_pair(6 * 4 + 6 * 2 + 3, columns) == (1, 2)


def test_marks_trace_selected_active_and_matched():
    """The grid's marks, in the same red: outlines for the picked
    pairs, the boldest for the animated one, a checker over every face
    of a committed match — alternating, so half of each face stays the
    value's own color."""
    from visualdynamics.viz.mac_bars import CHECKER, MARK_LIFT

    matrix = np.eye(3)
    plotter, _info = scene(matrix, frequencies=np.array([10., 20., 30.]))
    import pyvista as pv

    plotter2 = pv.Plotter(off_screen=True)
    add_mac_bars(plotter2, np.array([10., 20., 30.]), matrix,
                 selected=[(0, 0), (1, 1)], active=(1, 1),
                 matched=[(2, 2)])
    actors = plotter2.renderer.actors
    assert actors['mac-bars-selected'].mapper.dataset.n_lines == 24, \
        'twelve edges per selected bar'
    assert actors['mac-bars-active'].mapper.dataset.n_lines == 12
    checker = actors['mac-bars-matched'].mapper.dataset
    assert checker.n_lines == 0, 'filled squares, not strokes'
    top = HEIGHT * 3 * 1.0 + MARK_LIFT   # a full match on a 3x3 grid
    z = np.asarray(checker.points)[:, 2]
    assert np.isclose(z.max(), top), 'the top face is checkered'
    assert np.isclose(z.min(), 0.0), 'and the sides run to the floor'
    centers = np.asarray(checker.cell_centers().points)
    on_top = np.isclose(centers[:, 2], top)
    assert on_top.sum() == CHECKER * CHECKER // 2, \
        'alternation: half the top face is red, half stays the value'
    assert (~on_top).sum() > 0, 'the sides are checkered too'
    plotter.close()
    plotter2.close()


def test_marks_yield_the_pick_to_the_bar_beneath():
    """Every mark stands in front of the face it marks, so a pickable
    mark swallows the click meant for the bar — selecting a checkered
    bar did exactly nothing. Only the bars answer the picker."""
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True)
    add_mac_bars(plotter, np.array([10., 20., 30.]), np.eye(3),
                 selected=[(0, 0)], active=(0, 0), matched=[(1, 1)])
    actors = plotter.renderer.actors
    assert actors['mac-bars'].GetPickable()
    for mark in ('mac-bars-selected', 'mac-bars-active',
                 'mac-bars-matched'):
        assert not actors[mark].GetPickable(), f'{mark} blocks the pick'
    plotter.close()


def test_picked_and_committed_wear_different_colors():
    """A blue outline on a red-checkered bar can be seen; the red one
    it used to be could not. Committed stays the grid's red."""
    import pyvista as pv

    from visualdynamics.viz.mac_bars import MARK_RED, PICK_BLUE

    plotter = pv.Plotter(off_screen=True)
    add_mac_bars(plotter, np.array([10., 20., 30.]), np.eye(3),
                 selected=[(0, 0)], active=(0, 0), matched=[(1, 1)])
    actors = plotter.renderer.actors
    blue = pv.Color(PICK_BLUE).float_rgb
    red = pv.Color(MARK_RED).float_rgb
    for mark in ('mac-bars-selected', 'mac-bars-active'):
        assert np.allclose(actors[mark].prop.color.float_rgb, blue)
    assert np.allclose(actors['mac-bars-matched'].prop.color.float_rgb,
                       red)
    plotter.close()


def test_marks_outside_the_grid_are_dropped():
    """A stale selection outliving a shrunk comparison must not draw
    outlines in empty space."""
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True)
    add_mac_bars(plotter, np.array([10., 20.]), np.eye(2),
                 selected=[(5, 5)], matched=[(7, 0)])
    actors = plotter.renderer.actors
    assert 'mac-bars-selected' not in actors
    assert 'mac-bars-matched' not in actors
    plotter.close()


def test_every_bar_wears_an_outline():
    """A field of near-1.0 MACs is a field of one color — viridis has
    nowhere to go above 0.95 — and without an outline the bars merge
    into a slab (Brandon, 2026-08-26). Twelve edges per bar, on every
    bar, not only the marked ones.
    """
    from visualdynamics.viz.mac_bars import EDGE_INK, EDGE_LIFT, MARK_LIFT

    plotter, _info = scene(np.full((4, 5), 0.97),
                           columns=np.linspace(10.0, 100.0, 5))
    edges = plotter.renderer.actors['mac-bars-edges']
    assert edges.mapper.dataset.n_lines == 12 * 4 * 5, \
        'twelve edges on each of the twenty bars'
    assert edges.prop.color.hex_rgb == EDGE_INK, 'drawn in the dark ink'
    assert not edges.mapper.dataset.n_faces, \
        'lines, not a second set of boxes over the first'
    assert EDGE_LIFT < MARK_LIFT, (
        'the plain outline stands inside the marks, so a selected '
        "bar's own frame is not fighting it for the same depth")
    plotter.close()


def test_the_outline_yields_to_the_bar_beneath():
    """Like every other mark: it stands in front of the face it traces,
    so a pickable one would swallow the click meant for the bar."""
    plotter, _info = scene(np.full((3, 3), 0.9))
    assert not plotter.renderer.actors['mac-bars-edges'].GetPickable()
    plotter.close()


#: a grid comfortably past OUTLINE_LIMIT, and fixed rather than
#: derived from it: a size computed from the constant under test
#: cannot falsify a change to that constant — raising the limit to
#: 10 000 asked this test for a hundred million bars and hung the
#: suite instead of failing it (2026-08-26)
DENSE = 60


def test_a_dense_grid_is_not_outlined():
    """Past the limit a bar is a pixel or two wide and the outlines
    close over the field into a dark mat — the opposite of the point.
    The bars themselves still draw."""
    from visualdynamics.viz.mac_bars import OUTLINE_LIMIT

    assert OUTLINE_LIMIT < DENSE, 'the dense case really is past the limit'
    plotter, info = scene(np.full((DENSE, DENSE), 0.97))
    assert 'mac-bars-edges' not in plotter.renderer.actors
    assert 'mac-bars' in plotter.renderer.actors, 'the bars are still there'
    assert info['rows'] == DENSE
    plotter.close()
    # and one cell under the limit still is
    plotter, _info = scene(np.full((OUTLINE_LIMIT, OUTLINE_LIMIT), 0.97))
    assert 'mac-bars-edges' in plotter.renderer.actors
    plotter.close()
