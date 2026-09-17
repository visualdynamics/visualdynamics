"""The Modal Assurance Criterion, as numbers and as the grid beside a
mode table.

MAC_ij = |phi_i^H psi_j|^2 / ((phi_i^H phi_i)(psi_j^H psi_j)): 1 for the
same shape in any scaling, near 0 for independent ones. The auto-MAC's
off-diagonals say how distinct a set's own modes are, which belongs
beside the list of them — when reading a shape set, and while fitting one.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.shapes import mac_matrix

pytestmark = pytest.mark.usefixtures('flat_grid')

def test_mac_is_one_for_the_same_shape_in_any_scaling():
    a = np.array([1.0, -2.0, 3.0, 0.5])
    matrix = mac_matrix(np.vstack([a, -2.5 * a]))
    assert np.allclose(matrix, 1.0)


def test_mac_is_zero_for_orthogonal_shapes():
    matrix = mac_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(matrix, np.eye(2))


def test_the_plate_auto_mac_is_nearly_identity():
    """Elastic modes only: the six rigid vectors are an arbitrary basis
    of the null space — the solver may hand back any mixture of them,
    and the MAC between two members of an arbitrary basis is not a
    statement about anything."""
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    elastic = np.flatnonzero(shapes.frequency > 0.0)
    matrix = shapes.auto_mac()[np.ix_(elastic, elastic)]
    assert np.allclose(np.diag(matrix), 1.0)
    off = matrix - np.diag(np.diag(matrix))
    assert off.max() < 0.6, 'mass-normalized modes are nearly independent'


# ---- where the grid shows ---------------------------------------------------

def _mac_labels(window):
    import pyqtgraph as pg

    plots = [i for i in window.mac_view.ci.items
             if hasattr(i, 'listDataItems')]
    return [i for p in plots for i in p.items
            if isinstance(i, pg.TextItem)]


def show_shapes(window, pump, shapes):
    window.add_object('Shapes', shapes)
    item = window._item_for_object('Shapes')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()


def test_selecting_a_shape_set_shows_its_auto_mac(window, pump):
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    show_shapes(window, pump, shapes)
    assert window.mac_view.isVisible()
    assert _mac_labels(window) == [], (
        'the color is the reading; per-cell numbers made it too busy')


def test_a_channel_table_shows_no_mac(window, pump):
    from visualdynamics import io

    table = io.load(fixture_path('plate', 'channel_table.vdyn'))
    window.add_object('Channels', table)
    item = window._item_for_object('Channels')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not window.mac_view.isVisible()


def test_the_fit_shows_the_mac_of_what_it_has_fitted(window, pump):
    import pyqtgraph as pg

    frfs = visualdynamics.import_file(fixture_path('plate', 'frfs.npz'))
    window.add_object('FRF', frfs)
    window.tree.setCurrentItem(window._item_for_object('FRF'))
    pump()
    window.start_modal_fit()
    pump()
    assert not window.mac_view.isVisible(), 'nothing fitted yet'
    window.confirm_mode_button.click()
    pump()
    assert window.mac_view.isVisible()
    window.confirm_mode_button.click()
    pump()
    images = [i for p in [i for i in window.mac_view.ci.items
                           if hasattr(i, 'listDataItems')]
              for i in p.items if isinstance(i, pg.ImageItem)]
    assert images and images[0].image.shape == (2, 2), 'a 2x2 grid'


def test_the_grid_is_square_and_zoom_stays_on_it(window, pump):
    """The widget stays square whatever the pane's shape, and zooming is
    allowed but clamped to the grid itself."""
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    show_shapes(window, pump, shapes)
    window.mac_frame.resize(320, 200)
    pump()
    geometry = window.mac_view.geometry()
    assert geometry.width() == geometry.height() == 200
    assert geometry.x() == (320 - 200) // 2, 'centered in the frame'
    plot = next(i for i in window.mac_view.ci.items
                if hasattr(i, 'getViewBox'))
    box = plot.getViewBox()
    assert box.state['mouseEnabled'] == [True, True]
    modes = len(shapes.frequency)
    box.setXRange(-50, 50, padding=0)
    box.setYRange(-50, 50, padding=0)
    pump()
    (x0, x1), (y0, y1) = box.viewRange()
    assert x0 >= 0 and x1 <= modes, 'x pinned to the grid'
    assert y0 >= 0 and y1 <= modes, 'y pinned to the grid'


# ---- the MAC between two selected sets --------------------------------------

def test_cross_mac_aligns_shared_dofs_whatever_their_order():
    """The sets need not cover the same DOFs or order them the same way;
    a positional comparison would scramble everything."""
    from visualdynamics.core.shapes import ShapeSet, cross_mac

    a = ShapeSet(frequency=[10.0, 20.0], damping=[0.01, 0.01],
                 coordinate=['1X+', '2X+', '3X+'],
                 shape_matrix=[[1.0, 2.0, 3.0], [3.0, -1.0, 0.5]])
    # the same two shapes, DOFs reversed and one extra the first lacks
    b = ShapeSet(frequency=[10.5, 19.5], damping=[0.01, 0.01],
                 coordinate=['3X+', '9X+', '2X+', '1X+'],
                 shape_matrix=[[3.0, 7.0, 2.0, 1.0],
                               [0.5, -4.0, -1.0, 3.0]])
    matrix = cross_mac(a, b)
    assert matrix.shape == (2, 2)
    assert np.allclose(np.diag(matrix), 1.0), 'same shapes, aligned'
    assert matrix[0, 1] < 0.5 and matrix[1, 0] < 0.5


def test_cross_mac_refuses_disjoint_sets():
    from visualdynamics.core.shapes import ShapeSet, cross_mac

    a = ShapeSet([10.0], [0.01], ['1X+'], [[1.0]])
    b = ShapeSet([10.0], [0.01], ['2X+'], [[1.0]])
    with pytest.raises(ValueError, match='share no DOFs'):
        cross_mac(a, b)


def test_two_selected_sets_show_their_cross_mac(window, pump):
    """Rows are the first set, columns the second, each axis ticked by
    its own frequencies — and the grid is as rectangular as they are."""
    import pyqtgraph as pg

    truth = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    from visualdynamics.core.shapes import ShapeSet
    few = ShapeSet(frequency=truth.frequency[:5], damping=truth.damping[:5],
                   coordinate=truth.coordinate,
                   shape_matrix=truth.shape_matrix[:5])
    window.add_object('Truth', truth)
    window.add_object('Fitted', few)
    # current first: setCurrentItem collapses the selection to itself
    window.tree.setCurrentItem(window._item_for_object('Truth'))
    window.tree.clearSelection()
    for name in ('Truth', 'Fitted'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.mac_frame.isVisible()
    plot = next(i for i in window.mac_view.ci.items
                if hasattr(i, 'getViewBox'))
    image = next(i for i in plot.items if isinstance(i, pg.ImageItem))
    assert image.image.shape == (5, truth.num_shapes), (
        'columns x rows: 64 rows for Truth, 5 columns for Fitted')
    bottom = plot.getAxis('bottom')
    step, positions = bottom.tickValues(0, 5, 400)[0]
    assert len(positions) == 5, 'five modes across, all named'
    assert bottom.tickStrings(positions, 1, step)[0] == (
        f'{float(few.frequency[0]):.1f}')
    left = plot.getAxis('left')
    step, positions = left.tickValues(0, truth.num_shapes, 4000)[0]
    assert left.tickStrings(positions, 1, step) == [
        f'{float(f):.1f}' for f in truth.frequency], (
        'with room for all 64, each row reads its own frequency')
    message = window.statusBar().currentMessage()
    assert 'MAC' in message and 'matched table' in message, (
        'two sets alone open the interactive pairing view')


def _cross_mac_of(window, pump, rows, columns):
    """A cross-MAC between a set of `rows` modes and one of `columns`,
    shown the way selecting both in the tree shows it."""
    from visualdynamics.core.shapes import ShapeSet

    truth = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))

    def cut(count):
        # real shapes, cycled when more are wanted than the fixture has;
        # the frequencies are spread out so each row reads as its own
        picks = np.arange(count) % truth.num_shapes
        return ShapeSet(frequency=np.linspace(10.0, 2000.0, count),
                        damping=truth.damping[picks],
                        coordinate=truth.coordinate,
                        shape_matrix=truth.shape_matrix[picks])

    window.add_object('Many', cut(rows))
    window.add_object('Few', cut(columns))
    window.tree.setCurrentItem(window._item_for_object('Many'))
    window.tree.clearSelection()
    for name in ('Many', 'Few'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    return next(i for i in window.mac_view.ci.items
                if hasattr(i, 'getViewBox'))


def test_a_lopsided_cross_mac_stays_wide_enough_to_read(window, pump):
    """139 modes against 8 held to its own shape is a 52-pixel sliver —
    narrower than the left axis alone — with the aspect lock inside it
    scrolling all but one column out of sight. The shape is clamped and
    the cells stretch instead, so every column stays on screen."""
    plot = _cross_mac_of(window, pump, 139, 8)
    window.mac_frame.resize(600, 900)
    pump()
    assert window.mac_frame.ratio == pytest.approx(1 / 3), 'clamped'
    assert window.mac_view.width() == 300, 'a third of its height, not 52'
    (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
    assert (x0, x1) == (0, 8), 'all eight columns, none scrolled away'
    assert (y0, y1) == (0, 139)


def test_a_squarish_cross_mac_still_has_square_cells(window, pump):
    """The clamp only bites past 3:1: inside it the frame is given the
    grid's own shape and the lock holds the cells square in it."""
    plot = _cross_mac_of(window, pump, 60, 30)
    assert window.mac_frame.ratio == pytest.approx(0.5)
    assert plot.getViewBox().state['aspectLocked']


def test_the_rendered_cross_mac_has_the_shape_it_has_on_screen(tmp_path):
    """Rendered to a file it is the same panel: 700 tall by the clamped
    shape, not stretched into a square nobody would recognize."""
    from PySide6.QtGui import QImage

    from visualdynamics.core.shapes import ShapeSet
    from visualdynamics.plot import plot_mac

    truth = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    few = ShapeSet(frequency=truth.frequency[:5], damping=truth.damping[:5],
                   coordinate=truth.coordinate,
                   shape_matrix=truth.shape_matrix[:5])
    path = plot_mac(truth, few, path=tmp_path / 'cross.png', show=False)
    image = QImage(str(path))
    assert image.width() / image.height() == pytest.approx(1 / 3, abs=0.02)

    square = QImage(str(plot_mac(truth, path=tmp_path / 'auto.png',
                                 show=False)))
    assert square.width() == square.height(), 'an auto-MAC is square'


def test_the_mode_labels_thin_to_the_ones_that_fit(window, pump):
    """One label per mode drew 139 frequencies on top of each other, an
    illegible smear down the axis: pyqtgraph drops a tick level that will
    not fit, never the labels inside one, so the axis thins them itself —
    and follows the zoom, so a cluster pulled in on is named in full."""
    plot = _cross_mac_of(window, pump, 139, 8)
    axis = plot.getAxis('left')
    step, positions = axis.tickValues(0, 139, 850)[0]
    labels = axis.tickStrings(positions, 1, step)
    assert len(labels) <= 850 // 18 + 1, 'no more than the axis has room for'
    assert len(labels) > 20, 'and not so few the axis says nothing'
    assert len(set(labels)) == len(labels), 'each names a different mode'

    zoomed_step, zoomed = axis.tickValues(40, 48, 850)[0]
    assert zoomed_step == 1, 'zoomed in, every mode in the range is named'
    assert len(zoomed) == 8


# ---- the mode-table toggle --------------------------------------------------

def test_shape_sets_offer_a_mode_table_toggle(window, pump):
    """Comparing shape sets is often about the MAC alone: the bar over the
    pane hides the table on the left, and the choice sticks."""
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    show_shapes(window, pump, shapes)
    assert window.table_bar.isVisible()
    assert window.table.isVisible(), 'shown until toggled off'
    window.mode_table_action.trigger()
    pump()
    assert not window.table.isVisible()
    assert window.mac_frame.isVisible(), 'the MAC keeps the pane'
    window.render_current()
    pump()
    assert not window.table.isVisible(), 'the choice survives a re-render'
    window.mode_table_action.trigger()
    pump()
    assert window.table.isVisible()


def test_other_tables_have_no_toggle_and_always_show(window, pump):
    """The mode-table and MAC toggles are the shape table's; a channel
    table's bar keeps only the specification sheet's toggle (2026-09-04)
    and the table itself always shows."""
    from visualdynamics import io

    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    show_shapes(window, pump, shapes)
    window.mode_table_action.trigger()   # hidden for shapes...
    pump()
    table = io.load(fixture_path('plate', 'channel_table.vdyn'))
    window.add_object('Channels', table)
    item = window._item_for_object('Channels')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not window.mode_table_action.isVisible()
    assert not window.mac_bars_action.isVisible()
    assert window.author_action.isVisible(), \
        'the bar stays up for the specification sheet alone'
    assert window.table.isVisible(), '...but a channel table always shows'


def test_the_bottom_tick_labels_stand_on_end(window, pump):
    """One frequency per mode overlaps horizontally by a dozen modes, so
    the bottom axis draws its labels rotated 90 degrees — and paints
    without error, which is the part a subclassed axis can get wrong."""
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    show_shapes(window, pump, shapes)
    plot = next(i for i in window.mac_view.ci.items
                if hasattr(i, 'getViewBox'))
    axis = plot.getAxis('bottom')
    assert type(axis).__name__ == 'UprightAxis'
    assert axis.height() == 48, 'room reserved for standing labels'
    window.mac_view.grab()   # a real paint through the rotated draw path


def test_the_table_and_the_mac_have_a_divider_to_drag(window, pump):
    """How much of the pane the mode list wants against how big the MAC
    should be is a judgment about the data in front of you. It used to
    be a fixed half-and-half row with nothing to grab."""
    from PySide6.QtCore import Qt

    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    show_shapes(window, pump, shapes)
    assert window.tables_row.orientation() == Qt.Orientation.Horizontal
    assert window.tables_row.count() == 2
    assert window.tables_row.handle(1) is not None, 'something to drag'

    window.tables_row.setSizes([700, 300])
    pump()
    table_width, mac_width = window.tables_row.sizes()
    assert table_width > mac_width, (table_width, mac_width)


def test_neither_side_can_be_dragged_away_entirely(window, pump):
    """Collapsing one to nothing leaves a pane that looks broken; the
    mode-table toggle is how the table goes away."""
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    show_shapes(window, pump, shapes)
    assert not window.tables_row.childrenCollapsible()
