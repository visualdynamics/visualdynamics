"""Drawing several selections at once.

Each distinct (abscissa, quantity) gets its own plot, and there is one
budget of curves across all of them so a large selection stays interactive.
How that budget is divided is the whole of this file: it used to be first
come, first served, which left later plots empty.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.plot import MAX_RECORDS, build_plots, curve_budget

# ---- dividing the budget ----------------------------------------------------

def test_two_plots_of_the_same_size_split_it_evenly():
    assert curve_budget(50, [45, 45]) == [25, 25]


def test_a_plot_wanting_less_than_its_share_gives_the_rest_away():
    """A two-channel plot beside a large one should cost it almost nothing."""
    assert curve_budget(50, [2, 100]) == [2, 48]


def test_everything_is_drawn_when_the_budget_is_not_reached():
    assert curve_budget(50, [10, 20]) == [10, 20]
    assert curve_budget(50, [45]) == [45]


def test_the_budget_is_never_exceeded():
    for needs in ([45, 45], [100, 1], [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                  [1000], [3, 3, 3]):
        assert sum(curve_budget(50, needs)) <= 50, needs


def test_no_plot_is_given_more_than_it_has():
    for needs in ([45, 45], [2, 100], [1, 1, 1], [60, 3]):
        for share, need in zip(curve_budget(50, needs), needs):
            assert share <= need, needs


def test_more_plots_than_curves_gives_one_each_until_it_runs_out():
    shares = curve_budget(3, [5, 5, 5, 5, 5])
    assert sum(shares) == 3
    assert shares == [1, 1, 1, 0, 0]


def test_nothing_to_draw_is_not_an_error():
    assert curve_budget(50, []) == []
    assert curve_budget(0, [10, 10]) == [0, 0]


# ---- through the real plotting ----------------------------------------------

@pytest.fixture
def both():
    return (visualdynamics.import_file(fixture_path('plate', 'time.npz')),
            visualdynamics.import_file(fixture_path('plate', 'spectrum.npz')))


def curves_by_axis(layout):
    import pyqtgraph as pg

    return {item.getAxis('bottom').labelText: len(item.listDataItems())
            for item in layout.ci.items if isinstance(item, pg.PlotItem)}


@pytest.mark.parametrize('spectrum_first', [False, True])
def test_a_time_history_and_a_spectrum_both_draw(qt_app, both, spectrum_first):
    """Selecting both used to leave one plot with axes and no lines: the
    first group spent the whole budget."""
    import pyqtgraph as pg

    time, spectrum = both
    series = [('spec', spectrum, None), ('time', time, None)]
    if not spectrum_first:
        series.reverse()
    layout = pg.GraphicsLayoutWidget()
    drawn, requested = build_plots(layout, series)

    assert requested == time.num_records + spectrum.num_records
    assert drawn == MAX_RECORDS
    counts = curves_by_axis(layout)
    assert set(counts) == {'time [s]', 'frequency [Hz]'}
    assert all(count > 0 for count in counts.values()), counts
    assert counts['time [s]'] == counts['frequency [Hz]'], \
        'the same two selections, so the same share either way round'


# ---- a specification's bounds --------------------------------------------

@pytest.fixture
def spec():
    return visualdynamics.import_file(
        fixture_path('plate', 'random.nc4'))['Random_specification']


def drawn_plot(qt_app, data):
    """The PlotItem, and the layout that owns it — which the caller must keep
    alive: dropping it lets Qt delete every item under it."""
    import pyqtgraph as pg

    layout = pg.GraphicsLayoutWidget()
    layout.resize(1200, 600)
    layout.show()
    build_plots(layout, [('spec', data, None)])
    for _ in range(6):
        qt_app.processEvents()
    plot = next(i for i in layout.ci.items if isinstance(i, pg.PlotItem))
    return plot, layout


def test_a_specification_draws_its_bounds_with_its_curve(qt_app, spec):
    from visualdynamics.plot import data_curves

    plot, _layout = drawn_plot(qt_app, spec)
    assert spec.num_records == 8
    # one curve each, and no more. The bounds are shading now: four more
    # lines per record crowded the curves being compared, and the zones
    # behind them say the same thing. data_curves leaves out the
    # invisible edges those zones are built from.
    assert len(data_curves(plot)) == 8


def test_the_bounds_are_shaded_rather_than_drawn(qt_app, spec):
    """Where a response may not go is a region, and a region reads as
    shading rather than as a pair of dashed lines."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot import data_curves

    plot, _layout = drawn_plot(qt_app, spec)
    zones = [item for item in plot.items
             if isinstance(item, pg.FillBetweenItem)]
    assert zones, 'the limits are there, as shading'
    assert all(item.opts['pen'].style() == Qt.PenStyle.SolidLine
               for item in data_curves(plot)), 'and nothing is dashed'


def test_the_view_still_reaches_the_limits(qt_app, spec):
    """They are not drawn, but they are still what the plot has to
    cover: a limit above everything measured must not be cropped off
    along with the zone that fills it."""
    import numpy as np

    from visualdynamics.units import DEFAULT_SYSTEM

    plot, _layout = drawn_plot(qt_app, spec)
    top = plot.getViewBox().state['limits']['yLimits'][1]
    highest = np.log10(np.nanmax(np.real(
        spec.display_limit('abort_upper', DEFAULT_SYSTEM))))
    assert top >= highest


def test_the_bounds_stay_out_of_the_legend(qt_app, spec):
    """Four more entries per record would bury the records themselves."""
    plot, _layout = drawn_plot(qt_app, spec)
    assert len(plot.legend.items) == spec.num_records


def test_the_legend_sits_below_the_plot_in_a_row(qt_app, spec):
    """Inside the view a legend lands on the data — a specification's
    bands, an FRF's peaks — and neither can be read (Brandon,
    2026-09-01). It goes below, horizontal, centered on the axes, and
    no wider than its entries; the plot keeps its whole area."""
    import pyqtgraph as pg

    plot, layout = drawn_plot(qt_app, spec)
    legend = plot.legend
    assert isinstance(legend, pg.LegendItem)
    assert legend in layout.ci.items, 'a layout row of its own, not a child of the view'
    [(legend_row, _)] = layout.ci.items[legend]
    [(plot_row, _)] = layout.ci.items[plot]
    assert legend_row == plot_row + 1
    view = plot.getViewBox().sceneBoundingRect()
    box = legend.sceneBoundingRect()
    assert box.top() >= view.bottom(), 'below the data, not over it'
    assert legend.columnCount > 1, 'horizontal'
    assert box.width() < 0.8 * view.width(), 'sized to its entries, not the strip'
    assert abs(box.center().x() - view.center().x()) < 1, (
        'centered on the axes — the strip also spans the left axis, so '
        'centering on that sat the names right of the data')


def test_the_legend_wraps_to_the_width_of_the_axes(qt_app, spec):
    """His second look: the names started at the center and ran off to
    the right. Two faults — the box froze at the first entry's width,
    and it was centered on the strip rather than the axes. Now the row
    holds as many entries as the axes are wide, wraps past that, and
    unwraps when the room comes back."""
    plot, layout = drawn_plot(qt_app, spec)
    legend = plot.legend
    entries = len(legend.items)
    assert entries >= 3, 'the fixture has enough records to wrap'
    wide = legend.sceneBoundingRect()
    assert legend.rowCount == 1 and legend.columnCount == entries

    layout.resize(220, 600)
    for _ in range(10):
        qt_app.processEvents()
    view = plot.getViewBox().sceneBoundingRect()
    box = legend.sceneBoundingRect()
    assert legend.columnCount < entries and legend.rowCount > 1, 'wrapped'
    assert box.width() <= view.width() + 1, 'never wider than the axes'
    assert box.height() > wide.height(), 'the rows stack'
    assert abs(box.center().x() - view.center().x()) < 1, 'still centered'
    assert box.top() >= view.bottom()

    layout.resize(1200, 600)
    for _ in range(10):
        qt_app.processEvents()
    assert legend.rowCount == 1 and legend.columnCount == entries, 'unwrapped'


def test_plain_data_draws_no_bounds(qt_app):
    """Only a specification has them; nothing else pays for the check."""
    data = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    plot, _layout = drawn_plot(qt_app, data)
    assert len(plot.listDataItems()) == data.num_records
