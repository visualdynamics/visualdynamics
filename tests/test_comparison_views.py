"""Reading a comparison: the spectra, or either of the bar charts.

A specification with a measurement against it is read three ways, one
at a time. The spectra are what the comparison looks like; the bar
charts are what it amounts to — how far each channel sits from the
level asked for, and how much of each channel fell outside.

There was a table here too. It is gone: six channels of it are read,
sixty are scanned, and the bars answer which channel is worst and how
many are out before anything is read at all.
"""

from __future__ import annotations

import pytest
from conftest import prepared_comparison as prepared
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from visualdynamics.theme import theme as resolve_theme


def select(window, *names):
    """Select several objects at once.

    setCurrentItem clears the selection unless it is told not to, which
    left only the last name selected and made every one of these pass
    for the wrong reason.
    """
    from PySide6.QtCore import QItemSelectionModel

    window.tree.clearSelection()
    for name in names:
        item = window._item_for_object(name)
        item.setSelected(True)
        window.tree.setCurrentItem(
            item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
    window.render_current()


def model(window):
    return window.table.model()


def column(window, title):
    head = model(window).headerData

    titles = [head(c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
              for c in range(model(window).columnCount())]
    return titles.index(title)


def cell(window, row, title):
    index = model(window).index(row, column(window, title))
    return model(window).data(index)




def comparing(window, pump):
    """A specification and its PSDs, both on the plot."""
    spec, psd = prepared(window, pump)
    select(window, spec, psd)
    pump()
    return spec, psd


# ---- three readings, one at a time --------------------------------------


def test_a_comparison_offers_all_three(window, pump):
    comparing(window, pump)
    offered = {k for k, a in window.data_pane.comparison_actions.items()
               if a.isVisible()}
    assert offered == {'curves', 'error', 'lines'}
    assert not window.data_pane.toolbar.isHidden()


def test_something_that_is_not_a_comparison_offers_none(window, pump,
                                                        survey):
    _shapes, frfs = survey
    window.add_object('FRF', frfs)
    window.show_object('FRF')
    pump()
    assert not any(a.isVisible() for a in
                   window.data_pane.comparison_actions.values())


def test_the_spectra_are_what_it_opens_on(window, pump):
    comparing(window, pump)
    assert window.data_pane.comparison_view == 'curves'
    assert window.bar_chart is None or not window.bar_chart.plot.isVisible()


def test_the_error_chart_is_a_bar_per_channel_in_dB(window, pump):
    from visualdynamics.core.compliance import ERROR_DB

    spec, _psd = comparing(window, pump)
    window.data_pane.comparison_actions['error'].trigger()
    pump()
    chart = window.bar_chart
    assert len(chart.rows) == window.objects[spec].num_records
    assert (chart.low, chart.high) == (-ERROR_DB, ERROR_DB)
    assert 'channels outside' in chart.summary.toPlainText()


def test_the_lines_chart_has_a_ceiling_and_no_floor(window, pump):
    """No amount of staying inside the abort limits is a fault."""
    from visualdynamics.core.compliance import LINES_PERCENT

    comparing(window, pump)
    window.data_pane.comparison_actions['lines'].trigger()
    pump()
    chart = window.bar_chart
    assert (chart.low, chart.high) == (LINES_PERCENT, None)
    assert all(v >= 0 for v in chart.values())
    assert chart.beyond(-5.0) is None, 'nothing is out below'


def test_a_bar_over_the_threshold_is_red_and_under_it_blue(window, pump):
    comparing(window, pump)
    window.data_pane.comparison_actions['error'].trigger()
    pump()
    chart = window.bar_chart
    assert chart.beyond(10.0) == 'over'
    assert chart.beyond(-10.0) == 'under'
    assert chart.beyond(0.0) is None
    colors = resolve_theme(window.theme_name)
    assert chart._brush(10.0).color().rgb() == \
        QColor(colors['exceed_over']).rgb()
    assert chart._brush(-10.0).color().rgb() == \
        QColor(colors['exceed_under']).rgb()


def test_the_chart_says_what_share_of_channels_is_out(window, pump):
    comparing(window, pump)
    window.data_pane.comparison_actions['lines'].trigger()
    pump()
    chart = window.bar_chart
    out = sum(1 for v in chart.values() if chart.beyond(v))
    assert chart.share() == pytest.approx(100.0 * out / len(chart.rows))
    assert f'{chart.share():.0f}%' in chart.summary.toPlainText()


# ---- the thresholds move -------------------------------------------------


def test_dragging_a_threshold_recounts(window, pump):
    comparing(window, pump)
    window.data_pane.comparison_actions['lines'].trigger()
    pump()
    chart = window.bar_chart
    was = chart.share()
    chart.set_threshold('low', 95.0)
    assert chart.share() < was, 'a higher bar to clear, fewer channels out'
    assert '95' in chart.summary.toPlainText()


def test_a_dragged_threshold_survives_a_redraw(window, pump):
    """A threshold that went back to its default on the next redraw
    would be no threshold at all."""
    comparing(window, pump)
    window.data_pane.comparison_actions['lines'].trigger()
    pump()
    window.bar_chart.set_threshold('low', 60.0)
    assert window.data_pane.lines_bound == pytest.approx(60.0)
    window.render_current()
    pump()
    assert window.bar_chart.low == pytest.approx(60.0)


def test_the_two_charts_remember_their_own_thresholds(window, pump):
    comparing(window, pump)
    pane = window.data_pane
    pane.comparison_actions['error'].trigger()
    pump()
    window.bar_chart.set_threshold('high', 1.5)
    pane.comparison_actions['lines'].trigger()
    pump()
    window.bar_chart.set_threshold('low', 40.0)
    pane.comparison_actions['error'].trigger()
    pump()
    assert window.bar_chart.high == pytest.approx(1.5), 'its own, not the other'


# ---- and the table really is gone ---------------------------------------


def test_the_comparison_has_a_table_of_its_own(window, pump):
    """One that answers the comparison's question rather than the
    specification's: how far each channel came out, and how much of its
    band is outside. The plot draws the channels that were picked and
    is never the whole list, so something has to be."""
    from PySide6.QtCore import Qt

    comparing(window, pump)
    assert window.table.isVisible()
    model = window.table.model()
    assert [model.headerData(c, Qt.Orientation.Horizontal)
            for c in range(model.columnCount())] == [
        'Channel', 'RMS error [dB]', 'Outside abort [%]']


def test_the_old_row_per_reading_model_is_gone_from_the_code():
    """The table that was removed listed a reading per row. What
    replaced it is a channel per row — `compliance_grid_model`."""
    import visualdynamics.gui.object_tables as tables

    assert not hasattr(tables, 'compliance_model')
    assert hasattr(tables, 'compliance_grid_model')


# ---- and they stay legible at seventy channels --------------------------


def many(count):
    """A chart of `count` channels, without needing a project of that size."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import error_chart

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    plot = layout.addPlot(row=0, col=0)
    rows = [(f'{100 + i}Z+', (i % 9) - 4.0, float(i)) for i in range(count)]
    return error_chart(plot, rows, resolve_theme('dark'))


_ALIVE: list = []


@pytest.mark.parametrize('count', [1, 21, 22, 70])
def test_every_channel_is_one_bar_on_one_axis(qt_app, count):
    """Split across sets of axes once, and un-split: with the names
    down the side there is nothing splitting buys that height does not,
    and both the app and the page scroll."""
    chart = many(count)
    assert len(chart.plot.items) > 0
    assert len(chart.bars.opts['width']) == count
    assert len(chart.rows) == count


def test_the_chart_fits_the_pane_however_many_channels(qt_app):
    """The reverse of what this test pinned until 2026-08-28: the chart
    used to *demand* ROW_HEIGHT per row, meant to make seventy names
    readable — but the app's plot surface has no scroll bars, so a
    260-row kurtosis reading was clipped off the bottom of the pane
    and no zoom could reach the missing bars. The chart takes the
    height it is given; crowded names are read by zooming in, which
    the y-axis allows. The report still grows with the count — it
    sizes its figure from ROW_HEIGHT itself, and a page scrolls."""
    small, large = many(6), many(70)
    assert large.plot.minimumHeight() == small.plot.minimumHeight(), \
        'no per-row height demand the pane cannot honor'


def test_dragging_the_threshold_recounts_every_channel(qt_app):
    chart = many(50)
    chart.set_threshold('low', -1.3)
    assert chart.low == pytest.approx(-1.3)
    assert '-1.3' in chart.summary.toPlainText()


def test_the_count_is_over_every_channel_not_just_the_first_axis(qt_app):
    chart = many(50)
    out = sum(1 for v in chart.values() if chart.beyond(v))
    assert f'of {len(chart.rows)} channels' in chart.summary.toPlainText()
    assert chart.share() == pytest.approx(100.0 * out / 50)


# ---- the thresholds are shaded ground, and they snap --------------------


def test_past_a_threshold_is_shaded_rather_than_lined(qt_app):
    """What is being said is "past here is out", and a filled region
    says it where a line leaves it to be inferred."""
    import pyqtgraph as pg

    chart = many(8)
    assert chart.lines, 'a threshold to shade'
    for _which, zone in chart.lines:
        assert isinstance(zone, pg.LinearRegionItem)


def test_over_is_red_and_under_is_blue(qt_app):
    """The same two colors, at the same weight, the specification plot
    shades its abort zones with."""
    from visualdynamics.plot.bars import ZONE_ALPHA

    colors = resolve_theme('dark')
    chart = many(8)
    zones = dict(chart.lines)
    assert zones['high'].brush.color().rgb() == \
        QColor(colors['exceed_over']).rgb()
    assert zones['low'].brush.color().rgb() == \
        QColor(colors['exceed_under']).rgb()
    assert zones['high'].brush.color().alpha() == ZONE_ALPHA


def test_a_one_sided_chart_shades_its_only_threshold_red(qt_app):
    """No amount of staying inside the abort limits is a fault, so its
    single threshold is the ceiling however it is named."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import lines_chart

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    rows = [(f'{100 + i}Z+', 0.0, float(i * 9)) for i in range(6)]
    chart = lines_chart(layout.addPlot(row=0, col=0), rows,
                        resolve_theme('dark'))
    assert len(chart.lines) == 1
    _which, zone = chart.lines[0]
    assert zone.brush.color().rgb() == \
        QColor(resolve_theme('dark')['exceed_over']).rgb()


def test_the_shading_reaches_past_anything_on_screen(qt_app):
    """It only has to be beyond what will ever be drawn — the view is
    set from the bars, so the shading gets no say in the scaling."""
    from visualdynamics.plot.bars import BEYOND

    chart = many(8)
    zones = dict(chart.lines)
    assert min(zones['low'].getRegion()) <= -BEYOND
    assert max(zones['high'].getRegion()) >= BEYOND


def test_each_threshold_shades_the_side_that_is_the_fault(qt_app):
    """Named for the threshold and shaded for the fault, which is where
    this went wrong: a one-sided chart's only threshold is called 'low'
    because it is the only one, and what is out is everything *above*
    it — so the shading belongs on the right."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import BEYOND, lines_chart

    zones = dict(many(8).lines)
    assert max(zones['low'].getRegion()) == pytest.approx(-3.0), 'blue below'
    assert min(zones['high'].getRegion()) == pytest.approx(3.0), 'red above'

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    rows = [(f'{100 + i}Z+', 0.0, float(i * 9)) for i in range(6)]
    one_sided = lines_chart(layout.addPlot(row=0, col=0), rows,
                            resolve_theme('dark'))
    _which, zone = one_sided.lines[0]
    low, high = sorted(zone.getRegion())
    assert low == pytest.approx(one_sided.low), 'starts at the threshold'
    assert high >= BEYOND, 'and runs to the right, not to the left'


def test_dragging_the_one_sided_zone_reads_its_inner_edge(qt_app):
    """The edge facing the data is the threshold; the far one is out at
    BEYOND and means nothing."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import lines_chart

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    rows = [(f'{100 + i}Z+', 0.0, float(i * 9)) for i in range(6)]
    chart = lines_chart(layout.addPlot(row=0, col=0), rows,
                        resolve_theme('dark'))
    _which, zone = chart.lines[0]
    zone.setRegion((25.0, 1e6))
    chart._dragged('low', zone)
    assert chart.low == pytest.approx(25.0)


@pytest.mark.parametrize('asked,lands', [
    (2.94736, 2.9), (-1.0499, -1.0), (10.06, 10.1), (-0.04, 0.0), (3.0, 3.0),
])
def test_a_threshold_snaps_to_a_tenth(qt_app, asked, lands):
    """A threshold that lands on -2.9473 dB is a number nobody wrote
    down."""
    chart = many(8)
    chart.set_threshold('high', asked)
    assert chart.high == pytest.approx(lands, abs=1e-9)


def test_the_shading_follows_the_threshold_it_was_given(qt_app):
    chart = many(8)
    chart.set_threshold('high', 1.5)
    assert min(dict(chart.lines)['high'].getRegion()) == pytest.approx(1.5)


def test_negative_bars_fit_inside_their_own_axis(qt_app):
    """Brandon, on the SRS deviation and the random RMS error alike:
    the axis did not show the full range of the data. The zoom fence
    offered the whole reach plus a margin on the positive side and
    only the margin below zero, so a chart of negative deviations
    clamped its own bars off the axis — four events sitting 13 dB low
    drew as slivers against the left edge."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import error_chart
    from visualdynamics.theme import theme as resolve_theme

    surface = pg.GraphicsLayoutWidget()
    plot = surface.addPlot()
    rows = [('1238X+ e1', -13.1, 0.0), ('1238X+ e2', -12.0, 0.0),
            ('1238X+ e3', -11.1, 0.0), ('1238X+ e4', -10.2, 0.0)]
    error_chart(plot, rows, resolve_theme('dark'), low=-3.0, high=3.0)
    view = plot.getViewBox()
    (x0, x1), _y = view.viewRange()
    assert x0 <= -13.1, f'the view opens at {x0:.1f}, cutting the bars'
    assert x1 >= 3.0, 'and still reaches the ceiling'
    limits = view.getState()['limits']['xLimits']
    assert limits[0] <= -13.1, 'the fence itself allows the bars'
    surface.close()
