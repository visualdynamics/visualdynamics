"""Two shape sets on one geometry: the cross-MAC drives an overlay.

Click a MAC cell and its two modes animate over each other on the
geometry, the second phase-aligned to the first so an arbitrary sign or
rotation cannot make matching shapes move against each other.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path
from PySide6.QtCore import QPointF, Qt

import visualdynamics
from visualdynamics.core.shapes import ShapeSet, aligned_mode

pytestmark = pytest.mark.usefixtures('flat_grid')

def test_a_flipped_shape_aligns_back(  ):
    a = ShapeSet([10.0], [0.01], ['1X+', '2X+', '3X+'],
                 [[1.0, -2.0, 0.5]])
    b = ShapeSet([10.2], [0.01], ['1X+', '2X+', '3X+'],
                 [[-1.1, 2.2, -0.55]])
    aligned = aligned_mode(a, 0, b, 0)
    assert np.isrealobj(aligned)
    assert np.vdot(a.shape_matrix[0], aligned) > 0, 'moving together now'
    assert np.allclose(aligned, -b.shape_matrix[0])


def test_a_rotated_complex_shape_aligns_back():
    base = np.array([1.0 + 0.2j, -0.5 + 0.1j, 0.3 - 0.4j])
    a = ShapeSet([10.0], [0.01], ['1X+', '2X+', '3X+'], [base])
    b = ShapeSet([10.0], [0.01], ['1X+', '2X+', '3X+'],
                 [base * np.exp(1j * 2.1)])
    aligned = aligned_mode(a, 0, b, 0)
    inner = np.vdot(a.shape_matrix[0], aligned)
    assert abs(np.angle(inner)) < 1e-9, 'the rotation is undone'


def test_alignment_factor_applies_to_a_third_vector():
    """Measured between a set and its projection, applied to the dense
    original: a sign flip in the projection flips the raw set too."""
    from visualdynamics.core.shapes import alignment_factor, apply_alignment

    a = ShapeSet([10.0], [0.01], ['1X+', '2X+'], [[1.0, 2.0]])
    projected = ShapeSet([10.0], [0.01], ['1X+', '2X+'],
                         [[-1.0, -2.0]])
    factor = alignment_factor(a, 0, projected, 0)
    dense = np.array([3.0, -1.0, 0.5])
    flipped = apply_alignment(dense, factor)
    assert np.isrealobj(flipped)
    assert np.allclose(flipped, -dense)


def test_disjoint_sets_are_left_alone():
    a = ShapeSet([10.0], [0.01], ['1X+'], [[1.0]])
    b = ShapeSet([10.0], [0.01], ['9X+'], [[-2.0]])
    assert np.allclose(aligned_mode(a, 0, b, 0), b.shape_matrix[0])


# ---- the comparison view ----------------------------------------------------

@pytest.fixture
def comparing(window, pump):
    geometry = visualdynamics.import_file(fixture_path('plate', 'geometry.unv'))
    truth = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    # a rigid baseline and elastic modes both, so the table has a 0 Hz
    # row with no percent error *and* rows that state one — modes 4..8
    # of 22 (two rigid, three elastic, the pair avoided)
    keep = [4, 5, 6, 7, 8]
    few = ShapeSet(truth.frequency[keep], truth.damping[keep],
                   truth.coordinate,
                   -truth.shape_matrix[keep])     # flipped on purpose
    window.add_object('Geometry', geometry)
    window.add_object('Truth', truth)
    window.add_object('Few', few)
    window.tree.setCurrentItem(window._item_for_object('Geometry'))
    window.tree.clearSelection()
    for name in ('Geometry', 'Truth', 'Few'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    return window


def test_the_mac_sits_left_and_the_overlay_animates(comparing, pump):
    from PySide6.QtCore import Qt

    from visualdynamics.viz.animate import PairedAnimator

    window = comparing
    assert window._compare is not None
    assert window.views.indexOf(window.table_pane) == 0, 'MAC pane left'
    assert window.views.orientation() == Qt.Orientation.Horizontal, (
        'side by side, not stacked')
    assert window.mac_frame.isVisible()
    assert isinstance(window.animator, PairedAnimator)
    assert 'MAC' in window.statusBar().currentMessage()


def test_the_default_cell_is_the_best_match_of_the_first_row(comparing):
    window = comparing
    row, column = window._compare['cell']
    assert row == 0
    truth = window.objects['Truth']
    from visualdynamics.core.shapes import cross_mac
    matrix = cross_mac(truth, window.objects['Few'])
    assert column == int(np.argmax(matrix[0]))


def test_clicking_a_cell_animates_that_pair(comparing, pump):
    window = comparing
    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    scene_point = plot.getViewBox().mapViewToScene(QPointF(2.5, 3.5))

    class Click:
        def scenePos(self):
            return scene_point

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

    window._mac_clicked(Click())
    pump()
    assert window._compare['cell'] == (3, 2), 'row is y, column is x'
    assert 'mode 3' in window.statusBar().currentMessage().lower() or \
        'Mode' in window.statusBar().currentMessage()


def test_the_flipped_copy_moves_with_the_original(comparing):
    """'Few' was built as minus the truth: without phase alignment the two
    copies would move exactly against each other."""
    window = comparing
    first = window.animator.first.deflection.offsets(0.0).copy()
    second = window.animator.second.deflection.offsets(0.0)
    direction = np.sum(first * second)
    assert direction > 0, 'aligned, not opposed'


def test_leaving_the_comparison_restores_the_pane_order(comparing, pump):
    window = comparing
    window.tree.clearSelection()
    item = window._item_for_object('Truth')
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    from PySide6.QtCore import Qt

    assert window.views.indexOf(window.table_pane) == 2
    assert window.views.orientation() == Qt.Orientation.Vertical
    assert window._compare is None


def test_a_click_keeps_the_animation_running(comparing, pump):
    """Stepping through MAC cells is how the comparison is browsed; the
    show must not stop at every step, nor the phase snap back."""
    from PySide6.QtCore import QPointF, Qt

    window = comparing
    window.set_playing(True)
    window.phase_slider.setValue(30)
    pump()
    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    scene_point = plot.getViewBox().mapViewToScene(QPointF(1.5, 1.5))

    class Click:
        def scenePos(self):
            return scene_point

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

    window._mac_clicked(Click())
    pump()
    assert window._playing, 'still running'
    assert window._timer.isActive(), 'and the frames are still coming'
    # No arithmetic on the phase here. It is a *cycle* — 120 steps every
    # two seconds, wrapping — so on a loaded machine a pump can carry it
    # right past the start, and `phase >= 30` called that a snap back at
    # 2. It failed one full run in about six, and reading it as a flake
    # would have been reading a wrapped cycle as a bug. What the click
    # must not do to the phase is checked without a running timer below,
    # where it is a fact rather than a race.
    assert window.animator.parameter == pytest.approx(
        window.phase_slider.value() / 120 * 2 * np.pi), (
        'the slider and the shape on screen say the same thing')
    window.set_playing(False)


def test_a_click_does_not_snap_the_phase_back(comparing, pump):
    """Clicking a MAC cell rebuilds the scene, and a rebuilt animator
    that started from zero would jerk the overlay back to its undeflected
    shape at every step of browsing the grid.

    Stopped, so this is the click's own doing and not a frame that landed
    with it.
    """
    from PySide6.QtCore import QPointF, Qt

    window = comparing
    window.set_playing(True)
    window.phase_slider.setValue(30)
    pump()
    window.set_playing(False)
    before = window.animator.parameter
    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    scene_point = plot.getViewBox().mapViewToScene(QPointF(1.5, 1.5))

    class Click:
        def scenePos(self):
            return scene_point

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

    window._mac_clicked(Click())
    assert window.phase_slider.value() == 30, 'the slider stayed put'
    assert window.animator.parameter == pytest.approx(before), (
        'and so did the shape it is showing')


def _matched_cell(window, row, column):
    model = window.table.model()
    return model.data(model.index(row, column))


def test_committed_matches_read_in_the_mode_table_format(comparing,
                                                         pump):
    """The + button commits the selected squares to a Matched Modes
    object; the table beside the MAC shows the committed matches in
    the mode table's own format — empty until something is added."""
    window = comparing
    model = window.table.model()
    assert window.table.isVisible()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    # the last column is the scaling the overlay normalizes away, and
    # is headed by the two sets so its direction reads off the header
    assert headers == ['Truth Mode', 'Frequency [Hz]', 'Damping [%]',
                       'Few Mode', 'Frequency [Hz]', 'Damping [%]',
                       'Δf [%]', 'MAC', 'Few/Truth']
    assert model.rowCount() == 0, 'nothing committed yet'
    assert window.add_matches_action.isVisible()
    window.add_matches()
    from visualdynamics.core.matches import MatchedModes

    assert isinstance(window.objects['Matched Modes'], MatchedModes)
    model = window.table.model()
    assert model.rowCount() == 1
    # the committed square wears the red checker: already added — one
    # filled path item over every committed cell, tagged to be found
    plot = next(i for i in window.mac_view.ci.items
                if hasattr(i, 'getViewBox'))
    checkers = [item for item in plot.items
                if getattr(item, 'mac_checker', False)]
    assert len(checkers) == 1, 'one path holds every committed cell'
    row, column = window._compare['cell']
    truth = window.objects['Truth']
    assert _matched_cell(window, 0, 0) == str(row + 1)
    assert _matched_cell(window, 0, 1) == \
        f'{float(truth.frequency[row]):.4f}'
    assert _matched_cell(window, 0, 2) == \
        f'{float(truth.damping[row]) * 100:.3f}'
    assert _matched_cell(window, 0, 3) == str(column + 1)
    # reselecting the two sets later finds the same committed table
    window.tree.setCurrentItem(window._item_for_object('Truth'))
    window.tree.clearSelection()
    for name in ('Truth', 'Few'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.table.model().rowCount() == 1
    # the table icon on the bar toggles it away and back
    window.mode_table_action.trigger()
    assert not window.table.isVisible()
    window.mode_table_action.trigger()
    assert window.table.isVisible()


def _click(window, pump, row, column, extend=False):
    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    scene_point = plot.getViewBox().mapViewToScene(
        QPointF(column + 0.5, row + 0.5))

    class Click:
        def scenePos(self):
            return scene_point

        def modifiers(self):
            return (Qt.KeyboardModifier.ShiftModifier if extend
                    else Qt.KeyboardModifier.NoModifier)

    window._mac_clicked(Click())
    pump()


def test_clicking_then_adding_lands_the_pair(comparing, pump):
    window = comparing
    _click(window, pump, 2, 1)
    window.add_matches()
    assert _matched_cell(window, 0, 0) == '3'
    assert _matched_cell(window, 0, 3) == '2'


def test_modifier_clicks_collect_and_the_plus_commits(comparing, pump):
    """The selection idiom on MAC cells: a plain click picks one pair,
    Shift extends, Shift on a selected pair removes it. The + button
    commits the selection to the Matched Modes object — dedupes on
    recommit — and deleting a table row removes the match."""
    window = comparing
    truth, few = window.objects['Truth'], window.objects['Few']
    _click(window, pump, 0, 0)
    assert window._compare['pairs'] == [(0, 0)]
    _click(window, pump, 1, 1, extend=True)
    assert window._compare['pairs'] == [(0, 0), (1, 1)]
    assert window._compare['cell'] == (1, 1), 'the new pair animates'
    _click(window, pump, 1, 1, extend=True)      # toggle one back off
    assert window._compare['pairs'] == [(0, 0)]
    _click(window, pump, 1, 1, extend=True)
    window.add_matches()
    assert window.table.model().rowCount() == 2
    assert _matched_cell(window, 1, 6) == '—', (
        'a 0 Hz rigid baseline has no percent error to state')
    flexible = next(i for i, f in enumerate(few.frequency)
                    if float(f) > 0)
    partner = int(np.argmin(np.abs(
        np.asarray(truth.frequency) - float(few.frequency[flexible]))))
    _click(window, pump, partner, flexible)
    window.add_matches()                          # extends the object
    assert window.table.model().rowCount() == 3
    expected = ((float(few.frequency[flexible])
                 - float(truth.frequency[partner]))
                / float(truth.frequency[partner]) * 100.0)
    assert _matched_cell(window, 2, 6) == f'{expected:+.2f}', (
        'frequency error, the first (basis) set as the baseline')
    assert float(_matched_cell(window, 2, 7)) == pytest.approx(
        1.0, abs=0.01), 'the same mode flipped still MACs to one'
    window.add_matches()                          # same square again
    assert window.table.model().rowCount() == 3, 'deduped, not doubled'
    matched = window.objects['Matched Modes']
    window._delete_table_rows('Matched Modes', 'match', [1])
    assert matched.pairs == [[0, 0], [partner, flexible]]
    assert window.table.model().rowCount() == 2, (
        'deleting a row removes the match')


def test_leaving_the_comparison_restores_the_mode_table(comparing, pump):
    window = comparing
    assert window.table.model().columnCount() == 9, 'the matched table'
    window.tree.clearSelection()
    item = window._item_for_object('Truth')
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    model = window.table.model()
    assert model.headerData(0, Qt.Orientation.Horizontal) == 'Mode', (
        'one set selected: the plain mode table again')


def test_the_grid_reads_tall_and_clicks_map_back(window, pump):
    """Whichever set has more modes makes the rows. With the small set
    leading the pair the grid transposes, and clicks still land on the
    right pair of modes."""
    truth = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    few = ShapeSet(truth.frequency[:5], truth.damping[:5],
                   truth.coordinate, -truth.shape_matrix[:5])
    window.add_object('Few', few)         # arrives first: leads the pair
    window.add_object('Truth', truth)
    window.tree.setCurrentItem(window._item_for_object('Few'))
    window.tree.clearSelection()
    for name in ('Few', 'Truth'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window._compare['pair'] == ('Few', 'Truth')
    assert window._compare['tall'] is True
    # taller than wide, but clamped: 5 modes against 64 held to their own
    # shape is a sliver (see test_mac's lopsided grid)
    assert window.mac_frame.ratio == pytest.approx(1 / 3)
    # display coordinates: x runs over Few's modes, y over Truth's
    _click(window, pump, 10, 2)
    assert window._compare['cell'] == (2, 10), (
        'the click maps back into (first set, second set) space')


def test_the_overlay_animates_from_the_linked_geometry(window, pump):
    """Selecting just the two sets animates them overlaid when a
    linked geometry can host it; the taskbar toggle turns it off."""
    from visualdynamics.viz.animate import PairedAnimator

    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.unv'))
    truth = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    few = ShapeSet(truth.frequency[:5], truth.damping[:5],
                   truth.coordinate, -truth.shape_matrix[:5])
    window.add_object('Geometry', geometry)
    window.add_object('Truth', truth)
    window.add_object('Few', few)
    window.tree.setCurrentItem(window._item_for_object('Geometry'))
    window.tree.clearSelection()
    for name in ('Geometry', 'Truth', 'Few'):
        window._item_for_object(name).setSelected(True)
    window.link_selected()
    window.tree.setCurrentItem(window._item_for_object('Truth'))
    window.tree.clearSelection()
    for name in ('Truth', 'Few'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.overlay_action.isVisible()
    assert isinstance(window.animator, PairedAnimator), (
        'the linked geometry hosts the overlay without being selected')
    assert window.scene.isVisible()
    # the corner table names what is moving: both sets' mode number,
    # frequency and damping
    # a CornerAnnotation holds one text per corner; 2 is upper-left
    caption = window.scene.plotter.actors['scene-caption'].GetText(2)
    assert 'Truth  mode' in caption and 'Few  mode' in caption
    assert 'Hz' in caption and '%' in caption
    sizes = window.views.sizes()
    assert sizes[window.views.indexOf(window.scene)] > 100, (
        'the animation gets real space, not a sliver')
    window.overlay_action.setChecked(False)
    window.render_current()
    pump()
    assert window.animator is None, 'toggled off: MAC and table alone'
    assert not window.scene.isVisible()
    window.overlay_action.setChecked(True)


def test_matches_need_no_report_and_the_template_binds_them(
        window, pump):
    """The user's route: highlight the two shape sets alone, pick MAC
    squares (Shift adds), press + — the Matched Modes object exists
    with no report anywhere. A modal report generated later binds it
    and renders the table."""
    import json

    from visualdynamics.core.report import modal_template
    from visualdynamics.report import render_html

    truth = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    few = ShapeSet(truth.frequency[:5], truth.damping[:5],
                   truth.coordinate, -truth.shape_matrix[:5])
    window.add_object('Truth', truth)
    window.add_object('Few', few)
    window.tree.setCurrentItem(window._item_for_object('Truth'))
    window.tree.clearSelection()
    for name in ('Truth', 'Few'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window._compare is not None, 'pairing needs no geometry'
    assert window.table.isVisible(), 'the matched table shows'
    _click(window, pump, 0, 0)
    _click(window, pump, 4, 4, extend=True)
    assert window.add_matches_action.isVisible()
    window.add_matches()
    matched = window.objects['Matched Modes']
    assert matched.first == 'Truth' and matched.second == 'Few'
    assert matched.pairs == [[0, 0], [4, 4]]
    assert window.linked_group('Matched Modes') is None, (
        'a comparison names a set on each side, so it joins neither')
    template = modal_template(window.objects)
    block = next(b for b in template.blocks if b.get('kind') == 'pairs')
    assert block['source'] == '@any:MatchedModes', (
        'symbolic, and unscoped: the matches are in no object group')
    payload = json.loads(
        render_html(template, window.objects).split(
            'type="application/json">')[1].split('</script>')[0])
    table = next(b for b in payload['blocks']
                 if b['kind'] == 'table' and 'MAC' in b['headers'])
    assert table['headers'][0] == 'Truth Mode'
    assert len(table['rows']) == 2
    mac = table['headers'].index('MAC')
    assert table['rows'][0][mac] == '1.000', 'the flipped copy MACs to one'
    # and the report carries the scaling the overlay figure cannot show
    assert table['headers'][-1] == 'Few/Truth'
    assert table['rows'][0][-1] == '1.00', 'a flipped copy is the same size'


def _committed(window, pump, cells):
    """Commit these MAC squares, so the table below has rows."""
    from PySide6.QtCore import QPointF, Qt

    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))

    class Click:
        def __init__(self, point, extend):
            self._point = point
            self._extend = extend

        def scenePos(self):
            return self._point

        def modifiers(self):
            return (Qt.KeyboardModifier.ShiftModifier if self._extend
                    else Qt.KeyboardModifier.NoModifier)

    for n, (row, column) in enumerate(cells):
        tall = window._compare.get('tall')
        x, y = ((row, column) if tall else (column, row))
        point = plot.getViewBox().mapViewToScene(
            QPointF(x + 0.5, y + 0.5))
        window._mac_clicked(Click(point, n > 0))
        pump()
    window.add_matches()
    pump()


def test_picking_a_row_of_the_matched_table_moves_the_comparison(comparing,
                                                                 pump):
    """The MAC and the table are two views of one list. Clicking a square
    already selects its row; this is the other direction, so stepping
    down the table walks the comparison."""
    window = comparing
    _committed(window, pump, [(0, 0), (2, 1)])
    assert window.table.model().rowCount() == 2

    window.table.selectRow(1)
    pump()
    assert tuple(window._compare['cell']) == (2, 1), (
        'the square follows the row')
    animated = window.animator
    window.table.selectRow(0)
    pump()
    assert tuple(window._compare['cell']) == (0, 0)
    assert window.animator is not animated, 'the scene was rebuilt for it'


def test_walking_the_table_does_not_stop_the_show(comparing, pump):
    window = comparing
    _committed(window, pump, [(0, 0), (2, 1)])
    window.set_playing(True)
    window.table.selectRow(1)
    pump()
    assert window._playing, 'still running'
    assert window._timer.isActive()
    window.set_playing(False)


def test_the_two_copies_are_drawn_one_through_the_other(comparing, pump):
    """A finite element model has a skin where a test set has a
    wireframe, and an opaque one hides the thing it is being compared
    against."""
    from visualdynamics.theme import OVERLAY_ALPHA

    window = comparing
    assert window._compare['alphas'] == (1.0, OVERLAY_ALPHA)
    assert window.animator.first.meshes, 'the basis is drawn'
    assert window.animator.second.meshes, 'and so is the comparison'
    opacities = {round(actor.GetProperty().GetOpacity(), 2)
                 for actor in window.scene.plotter.renderer.actors.values()
                 if hasattr(actor, 'GetProperty')}
    assert 1.0 in opacities and OVERLAY_ALPHA in opacities, opacities


def test_clicking_a_cell_keeps_the_mac_zoom(comparing, pump):
    """Picking a pair re-renders the whole comparison, and the rebuild
    was reframing the MAC — every click threw away the zoom the user
    had set to see the cells they were picking. Redrawing the same
    comparison keeps the view; a different comparison starts whole."""
    window = comparing
    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    plot.setRange(xRange=(1.0, 4.0), yRange=(1.0, 4.0), padding=0)
    pump()
    scene_point = plot.getViewBox().mapViewToScene(QPointF(2.5, 3.5))

    class Click:
        def scenePos(self):
            return scene_point

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

    window._mac_clicked(Click())
    pump()
    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    (x0, x1), (y0, y1) = plot.viewRange()
    assert (x0, x1) == pytest.approx((1.0, 4.0))
    assert (y0, y1) == pytest.approx((1.0, 4.0))

    # leaving for a lone shape set's auto-MAC and coming back is a new
    # comparison view: it frames itself whole again
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object('Truth'))
    window._item_for_object('Truth').setSelected(True)
    window.render_current()
    pump()
    for name in ('Geometry', 'Truth', 'Few'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    plot = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    (x0, x1), _ = plot.viewRange()
    assert (x1 - x0) > 4.5, 'reframed to the whole grid'
