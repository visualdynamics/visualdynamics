"""A specification on its own: one channel at a time, and its levels.

Six targets and their two dozen limit lines stacked on one axis are
unreadable, so a specification is read the way a comparison is — one
channel drawn, the rest offered in the bar and listed in the table
beneath, and the two pointing at each other.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

import visualdynamics
from visualdynamics.plot import data_curves, pair_label

pytestmark = pytest.mark.usefixtures('flat_reading')

@pytest.fixture
def showing(window, pump):
    """The survey run's specification, selected on its own."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'Specification')
    window.show_object(name)
    pump()
    return window, window.objects[name]


@pytest.fixture
def levels(showing, pump):
    """The same specification with its RMS reading up: the bars over
    the table of levels (Brandon, 2026-09-06 — the table is no longer
    the default beneath the spectra)."""
    window, spec = showing
    window.data_pane.rms_action.trigger()
    pump()
    return window, spec


def plot_of(window):
    return next(item for item in window.data_pane.graphics.ci.items
                if hasattr(item, 'listDataItems'))


def press(qt_app, widget, key):
    qt_app.sendEvent(widget, QKeyEvent(QEvent.Type.KeyPress, key,
                                       Qt.KeyboardModifier.NoModifier))
    qt_app.processEvents()


# ---- one channel at a time ----------------------------------------------


def test_only_one_channel_is_drawn(showing):
    window, spec = showing
    assert spec.num_records > 1, 'a specification worth cutting down'
    assert len(data_curves(plot_of(window))) == 1


def test_the_bar_offers_the_others(showing):
    """And the bar itself is up.

    A QAction's isVisible is its own flag and stays true inside a
    hidden toolbar, so checking the action alone agreed the drop-down
    was there while the screen showed nothing at all. isHidden is the
    question worth asking — it is about whether we hid it, not about
    whether the window happens to be on screen.
    """
    window, spec = showing
    assert window.data_pane.pair_action.isVisible()
    assert not window.data_pane.toolbar.isHidden(), 'the bar it lives in'
    assert window.data_pane.pair_box.count() == spec.num_records


def test_the_limits_are_still_shaded(showing):
    """A figure claiming a tolerance band has to show one."""
    import pyqtgraph as pg

    window, _spec = showing
    fills = [item for item in plot_of(window).items
             if isinstance(item, pg.FillBetweenItem)]
    assert len(fills) == 4, 'warning and abort, above and below'


def test_the_axis_does_not_reach_for_zero(showing):
    """A specification is written to zero outside its band, and zero on
    a log axis is minus infinity. The view has to stay on the data."""
    window, _spec = showing
    low, high = plot_of(window).getViewBox().viewRange()[1]
    assert low > -20.0, 'not reaching down to a decade count of -300'
    assert high - low < 10.0, 'and not spanning ten decades to get there'


# ---- the table beneath --------------------------------------------------


def test_a_row_per_channel_with_its_level(levels):
    window, spec = levels
    model = window.table.model()
    assert model.rowCount() == spec.num_records
    assert model.columnCount() == 2
    assert model.index(0, 0).data() == '101Z+'
    assert float(model.index(0, 1).data()) == pytest.approx(1.4217, abs=1e-3)


def test_the_level_is_the_specifications_own(levels):
    """Integrated from its own breakpoints over its own band, so it does
    not move with whatever is measured against it."""
    from visualdynamics.core.compliance import specification_rms

    window, spec = levels
    model = window.table.model()
    for row in range(model.rowCount()):
        assert float(model.index(row, 1).data()) == pytest.approx(
            specification_rms(spec, row), rel=1e-3)


def test_the_row_being_drawn_is_selected(levels):
    window, _spec = levels
    rows = {index.row() for index in window.table.selectedIndexes()}
    assert rows == {0}
    assert pair_label(window._plotted_pair) == '101Z+'


# The table and the spectra used to follow each other — arrow keys in
# the table changed the drawn channel, stepping the pair moved the
# selected row. They no longer share a screen: the table sits beneath
# the bars (2026-09-06), which draw every channel at once, so there is
# no drawn channel for a row to pick. The coupling itself still serves
# the comparison table and is tested there (test_pair_selector,
# test_replication).


def test_a_measurement_beside_it_gets_the_comparison_instead(window, pump):
    """The specification's own levels give way to the comparison.

    The table stays, but it is a different table: what each channel
    *asked for* is replaced by how far it came out from it, which is
    the question once there is something to answer it with. The three
    readings appear in the bar alongside.
    """
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    history = next(n for n, o in window.objects.items()
                   if type(o).__name__ == 'TimeHistory')
    window.tree.setCurrentItem(window._item_for_object(history))
    window.compute_psds()
    pump()
    from PySide6.QtCore import QItemSelectionModel

    spec = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'Specification')
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd' and n != spec)
    window.tree.clearSelection()
    for name in (spec, psd):
        item = window._item_for_object(name)
        item.setSelected(True)
        window.tree.setCurrentItem(
            item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
    window.render_current()
    pump()
    from PySide6.QtCore import Qt

    assert window.table.isVisible(), 'the comparison has its own table'
    model = window.table.model()
    assert [model.headerData(c, Qt.Orientation.Horizontal)
            for c in range(model.columnCount())] == [
        'Channel', 'RMS error [dB]', 'Outside abort [%]']
    assert all(action.isVisible() for action in
               window.data_pane.comparison_actions.values())



# ---- the same specification in a report ---------------------------------


def report_plot():
    from visualdynamics.report import _plot_block

    spec = visualdynamics.import_file(
        fixture_path('plate', 'random.nc4'))['Random_specification']
    return spec, _plot_block(
        {'kind': 'plot', 'source': 'S', 'mode': 'curves', 'caption': 'Spec'},
        spec, {}, visualdynamics.SI)


def test_the_report_plot_stays_on_the_data():
    """It used to plot a specification's zeros as log10(1e-300), which
    is -300 — an axis reaching three hundred decades down to reach
    numbers that were never there."""
    _spec, built = report_plot()
    values = [v for curve in built['curves'] for v in curve['y']
              if v is not None]
    assert min(values) > -20.0
    assert max(values) - min(values) < 10.0


def test_the_report_plot_shades_the_limits_it_claims():
    """Shaded, not drawn as four lines crowding the curve they bound —
    the same reading the app gives. A response is either inside the
    band or it is not, and a filled band says so without asking anyone
    to tell four dashed lines apart."""
    _spec, built = report_plot()
    zones = built['channels'][0]['zones']
    assert [z['severity'] for z in zones] == [
        'warning', 'abort', 'warning', 'abort']
    # the zones past abort have no far side: they run to the plot edge
    assert zones[1]['upper'] is None
    assert zones[3]['lower'] is None


def test_the_report_plot_draws_one_channel_and_offers_the_rest():
    spec, built = report_plot()
    assert spec.num_records > 1
    assert len(built['curves']) == 1, 'one target drawn'
    assert len(built['channels']) == spec.num_records, 'the rest a pick away'
    assert 'drop-down' in built['caption']


def test_every_array_of_a_channel_is_on_one_grid():
    """`_decimate` keeps the extremes of each bin, so which points it
    keeps depends on the values — two curves through it separately come
    back on two different frequency grids, and a band shaded between
    them would be shaded between points that are not above each other."""
    _spec, built = report_plot()
    width = len(built['x'])
    for channel in built['channels']:
        assert len(channel['y']) == width
        for zone in channel['zones']:
            for edge in (zone['lower'], zone['upper']):
                assert edge is None or len(edge) == width


def test_a_specification_with_no_limits_is_still_drawn():
    """Not every target carries bounds, and one that does not is a plain
    curve rather than nothing."""
    from visualdynamics.core.data import Specification
    from visualdynamics.report import _plot_block

    axis = np.linspace(1.0, 100.0, 64)
    bare = Specification(abscissa=axis,
                         ordinate=np.atleast_2d(np.full(64, 1e-3)),
                         response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'])
    built = _plot_block({'kind': 'plot', 'source': 'S', 'mode': 'curves'},
                        bare, {}, visualdynamics.SI)
    assert len(built['curves']) == 1


def test_a_lone_specification_takes_the_foreground_color(showing):
    """Beside a response it is gray, because it is the reference for
    something. Alone it *is* what is being looked at, and gray would be
    saying it is the reference for nothing."""
    from visualdynamics.theme import theme as resolve_theme

    window, _spec = showing
    curve = data_curves(plot_of(window))[0]
    wanted = resolve_theme(window.theme_name)['response_curve']
    assert curve.opts['pen'].color().name().lower() == wanted.lower()


def test_beside_a_response_it_goes_back_to_gray(window, pump):
    """The other half of the rule, so neither can drift."""
    from PySide6.QtCore import QItemSelectionModel

    from visualdynamics.theme import theme as resolve_theme

    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    history = next(n for n, o in window.objects.items()
                   if type(o).__name__ == 'TimeHistory')
    window.tree.setCurrentItem(window._item_for_object(history))
    window.compute_psds()
    pump()
    spec = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'Specification')
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd' and n != spec)
    window.tree.clearSelection()
    for name in (spec, psd):
        item = window._item_for_object(name)
        item.setSelected(True)
        window.tree.setCurrentItem(
            item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
    window.render_current()
    pump()
    colors = resolve_theme(window.theme_name)
    drawn = {c.opts['pen'].color().name().lower()
             for c in data_curves(plot_of(window))}
    assert colors['specification_curve'].lower() in drawn
    assert colors['response_curve'].lower() in drawn


def _spec_and_response_colors(limits):
    """The colors drawn for a specification and a response of one DOF.

    Built rather than imported: the case that broke is a specification
    carrying no limits, and the only fixture that has one is a transient
    run, which lives outside the repository. What matters here is the
    presence or absence of the bounds, so it is stated directly.
    """
    import pyqtgraph as pg

    from visualdynamics.core.data import Psd, Specification
    from visualdynamics.plot import build_plots
    from visualdynamics.theme import theme as resolve_theme

    axis = np.linspace(1.0, 100.0, 64)
    level = np.atleast_2d(np.full(64, 1e-3))
    spec = Specification(abscissa=axis, ordinate=level,
                         response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         **{bound: level * factor
                            for bound, factor in limits.items()})
    measured = Psd(abscissa=axis, ordinate=level * 1.1,
                   response_dof=['101Z+'], reference_dof=['101Z+'],
                   ordinate_dim=['acceleration**2/frequency'])
    layout = pg.GraphicsLayoutWidget()
    build_plots(layout, [('measured', measured, None), ('spec', spec, None)])
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    return ({c.opts['pen'].color().name().lower() for c in data_curves(plot)},
            resolve_theme(None))


def test_a_specification_with_no_limits_still_reads_as_the_reference(qt_app):
    """The colors say which curve is the measurement and which is the
    target, and that is true of a target with no bounds written on it —
    a transient run's, say, where a tolerance on a waveform is not a
    settled convention. Deciding it from the presence of limit curves
    made the same pair of objects draw two different ways depending on
    which environment produced them."""
    drawn, colors = _spec_and_response_colors({})
    assert colors['response_curve'].lower() in drawn
    assert colors['specification_curve'].lower() in drawn


def _spec_and_response_widths(limits, alone=False):
    """{color: pen width} for the same pair, or for the specification
    drawn on its own."""
    import pyqtgraph as pg

    from visualdynamics.core.data import Psd, Specification
    from visualdynamics.plot import build_plots

    axis = np.linspace(1.0, 100.0, 64)
    level = np.atleast_2d(np.full(64, 1e-3))
    spec = Specification(abscissa=axis, ordinate=level,
                         response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         **{bound: level * factor
                            for bound, factor in limits.items()})
    measured = Psd(abscissa=axis, ordinate=level * 1.1,
                   response_dof=['101Z+'], reference_dof=['101Z+'],
                   ordinate_dim=['acceleration**2/frequency'])
    layout = pg.GraphicsLayoutWidget()
    build_plots(layout, [('spec', spec, None)] if alone
                else [('measured', measured, None), ('spec', spec, None)])
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    return {c.opts['pen'].color().name().lower(): c.opts['pen'].widthF()
            for c in data_curves(plot)}


def test_the_reference_that_stands_back_is_twice_as_wide(qt_app):
    """At one pixel a gray requirement under a dense measurement
    vanished — Brandon could not find the specification under the sine
    levels (2026-09-25). Stood back, it is drawn at two, as the stage
    has always drawn its target; the measurement over it stays at one,
    and a specification alone, being what is looked at, stays at one."""
    from visualdynamics.theme import theme as resolve_theme

    colors = resolve_theme(None)
    widths = _spec_and_response_widths(
        {'warning_upper': 2.0, 'warning_lower': 0.5,
         'abort_upper': 4.0, 'abort_lower': 0.25})
    assert widths[colors['specification_curve'].lower()] == 2.0
    assert widths[colors['response_curve'].lower()] == 1.0
    alone = _spec_and_response_widths({}, alone=True)
    assert set(alone.values()) == {1.0}


def test_the_bounded_case_is_colored_the_same_way(qt_app):
    """The other half, so the two cannot drift apart — which is the whole
    complaint: a random run and a transient run drew the same selection
    differently."""
    bounded, colors = _spec_and_response_colors(
        {'warning_upper': 2.0, 'warning_lower': 0.5,
         'abort_upper': 4.0, 'abort_lower': 0.25})
    bare, _ = _spec_and_response_colors({})
    assert bounded == bare
    assert colors['response_curve'].lower() in bounded
    assert colors['specification_curve'].lower() in bounded
