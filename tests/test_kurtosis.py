"""How Gaussian a record is, per channel.

A spectrum says nothing about the shape of the distribution that
produced it: a smooth hiss and a train of rare hard peaks can have
identical PSDs and fatigue an article differently. These hold the
reading to scipy's own — Pearson, not Fisher — and hold the chart to
the band Brandon asked for (2026-08-24).
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.kurtosis import (
    HIGH,
    LOW,
    NOMINAL,
    channel_kurtosis,
    kurtosis,
)


def test_it_is_pearson_and_agrees_with_scipy():
    """Pearson's kurtosis is 3 for a Gaussian; Fisher's excess form is
    that minus three. The two differ by exactly the number a report
    would be claiming, so this pins which one we compute."""
    scipy_stats = pytest.importorskip('scipy.stats')

    rng = np.random.default_rng(4)
    for values in (rng.standard_normal(20000),
                   rng.uniform(-1, 1, 20000),
                   rng.standard_t(6, 20000),
                   rng.standard_normal(500) * 3.0 + 7.0):
        assert kurtosis(values) == pytest.approx(
            scipy_stats.kurtosis(values, fisher=False, bias=True))


def test_a_gaussian_record_reads_about_nominal():
    rng = np.random.default_rng(11)
    assert kurtosis(rng.standard_normal(200000)) == pytest.approx(
        NOMINAL, abs=0.05)
    assert LOW < NOMINAL < HIGH, 'the band brackets the nominal'


def test_the_shapes_that_matter_land_outside_the_band():
    """The two faults the reading exists to catch: rare hard peaks
    (heavy tails, high) and a clipped or non-random record (low)."""
    rng = np.random.default_rng(5)
    gauss = rng.standard_normal(100000)

    peaky = gauss.copy()
    peaky[::5000] *= 12.0            # a few rare, very hard hits
    assert kurtosis(peaky) > HIGH

    clipped = np.clip(gauss, -1.0, 1.0)
    assert kurtosis(clipped) < LOW

    sine = np.sin(np.linspace(0, 400 * np.pi, 100000))
    assert kurtosis(sine) == pytest.approx(1.5, abs=0.01), \
        'a pure sine is 1.5, the textbook value — and well under the band'


def test_a_dead_channel_has_no_shape_to_describe():
    """NaN, never zero: zero would draw as an impossibly flat
    distribution rather than as the absence of one."""
    assert np.isnan(kurtosis(np.zeros(1000)))
    assert np.isnan(kurtosis([1.0]))
    assert np.isnan(kurtosis([]))


def test_every_channel_gets_one_bar_whatever_it_measures():
    """Kurtosis is dimensionless, so accelerations and forces share an
    axis honestly — the one reading here where mixing quantities is
    not a lie (Brandon, 2026-08-24)."""
    rng = np.random.default_rng(7)
    t = np.arange(4096) / 1024.0
    data = visualdynamics.TimeHistory(
        t, rng.standard_normal((3, len(t))),
        response_dof=['9001X+', '101Z+', '104Z+'],
        ordinate_dim=['force', 'acceleration', 'acceleration'],
        ordinate_unit=['N', 'm/s**2', 'm/s**2'])
    rows = channel_kurtosis(data)
    assert len(rows) == 3, 'the force channel is on the chart too'
    assert [label for label, _v in rows] == [
        data.record_label(i) for i in range(3)]
    assert all(abs(value - NOMINAL) < 0.5 for _l, value in rows)
    # and a subset when the caller names one
    assert len(channel_kurtosis(data, [1])) == 1


# ---- the toolbar toggle -------------------------------------------------


def _record(window, pump, peaky=False):
    rng = np.random.default_rng(3)
    t = np.arange(8192) / 2048.0
    rows = rng.standard_normal((4, len(t)))
    if peaky:
        rows[1, ::400] *= 15.0          # one channel with rare hard hits
    history = visualdynamics.TimeHistory(
        t, rows, response_dof=['101Z+', '104Z+', '110Z+', '9001X+'],
        ordinate_dim=['acceleration'] * 3 + ['force'],
        ordinate_unit=['m/s**2'] * 3 + ['N'])
    window.add_object('Record', history)
    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    return history


def test_the_button_is_offered_for_a_record_and_not_a_spectrum(window,
                                                               pump):
    """A reading of a record: it appears with the averaging, because
    both are things you ask of time data."""
    _record(window, pump)
    assert window.data_pane.kurtosis_action.isVisible()
    psds = window.project.compute_psds('Record')
    window.show_object(psds)
    item = window._item_for_object(psds)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    assert not window.data_pane.kurtosis_action.isVisible(), \
        'a spectrum has no distribution to describe'


def test_toggling_it_draws_a_bar_per_channel(window, pump):
    """One bar per channel whatever it measures — the force channel is
    on the chart beside the accelerations, because kurtosis carries no
    units to disagree about."""
    from visualdynamics.core.kurtosis import HIGH, LOW

    _record(window, pump, peaky=True)
    pane = window.data_pane
    pane.kurtosis_action.setChecked(True)
    pane.kurtosis_wanted = True
    window.render_current()
    pump()
    chart = window.bar_chart
    assert chart is not None
    assert len(chart.rows) == 4, 'every channel, force included'
    assert (chart.low, chart.high) == (LOW, HIGH)
    labels = [label for label, _v in chart.rows]
    assert '9001X+' in ' '.join(labels)
    # the bars grow from the nominal, so a near-Gaussian channel is a
    # stub and the length of a bar is the departure (Brandon,
    # 2026-08-24). Drawn from zero, three units of agreement would
    # dwarf every reading on the chart.
    assert chart.baseline == NOMINAL
    drawn = chart.bars.opts
    assert list(drawn['x0']) == [NOMINAL] * len(chart.rows)
    widths = list(drawn['width'])
    assert widths[1] > 0, 'the peaky channel runs right of nominal'
    assert all(abs(w) < 1.0 for i, w in enumerate(widths) if i != 1)
    # the planted peaky channel is the one over the band
    beyond = [chart.beyond(value) for _l, value in chart.rows]
    assert beyond[1] == 'over'
    assert beyond[0] is None and beyond[2] is None

    # turning it off gives the trace back
    pane.kurtosis_action.setChecked(False)
    pane.kurtosis_wanted = False
    window.render_current()
    pump()
    assert window.data_pane.graphics.scene() is not None


def test_the_ordinary_channels_stand_back_in_gray(window, pump):
    """The chart is about the exceptions, so a channel inside the band
    is gray rather than ink — which is also what the report has always
    drawn there (Brandon, 2026-08-24)."""
    from visualdynamics.theme import theme as resolve_theme

    _record(window, pump, peaky=True)
    pane = window.data_pane
    pane.kurtosis_action.setChecked(True)
    pane.kurtosis_wanted = True
    window.render_current()
    pump()
    chart = window.bar_chart
    colors = resolve_theme(window.theme_name)
    inside = next(v for _l, v in chart.rows if chart.beyond(v) is None)
    over = next(v for _l, v in chart.rows if chart.beyond(v) == 'over')
    assert chart._brush(inside).color().name() == \
        colors['specification_curve']
    assert chart._brush(over).color().name() == colors['exceed_over']


def test_it_stands_the_mark_views_down(window, pump):
    """A different reading of the record, not another mark on it: the
    trace is gone, so there is nothing for the frames or the shock
    windows to be drawn on."""
    _record(window, pump)
    pane = window.data_pane
    pane.averaging_action.setChecked(True)
    pane._choose_averaging(True)
    pump()
    assert pane.averaging_wanted
    pane.kurtosis_action.setChecked(True)
    pane._choose_reading(True)
    pump()
    assert pane.kurtosis_wanted
    assert not pane.averaging_wanted, 'the frames stood down'
    assert not pane.averaging_action.isChecked()


# ---- headless, and in every report --------------------------------------


def test_the_chart_renders_without_a_window(tmp_path):
    """Whatever the window can draw, a script can render to a file."""
    from visualdynamics.plot import plot_kurtosis

    rng = np.random.default_rng(2)
    t = np.arange(4096) / 1024.0
    data = visualdynamics.TimeHistory(
        t, rng.standard_normal((5, len(t))),
        response_dof=[f'{101 + i}Z+' for i in range(5)],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')
    path = tmp_path / 'kurtosis.png'
    plot_kurtosis(data, path=str(path), show=False)
    assert path.stat().st_size > 1000

    dead = visualdynamics.TimeHistory(
        t, np.zeros((1, len(t))), response_dof=['101Z+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')
    with pytest.raises(ValueError, match='shape to describe'):
        plot_kurtosis(dead, path=str(tmp_path / 'no.png'), show=False)


def test_the_shock_and_modal_reports_do_not_claim_a_band_they_cannot_use():
    """A shock record is a transient in a long quiet stretch and never
    reads near three — the plate's own reads 16 to 27 over its
    windows. A band at 2 to 4 would mark every channel of every shock
    test in red and say nothing anyone could act on, so that report
    does not carry the figure (Brandon, 2026-08-24). Nor does the
    modal report (Brandon, 2026-09-08): a modal test is driven with
    burst random, whose kurtosis read whole is about three over its
    duty and says nothing about the article. The toolbar toggle still
    offers the reading; what a report asserts is the judgment."""
    from visualdynamics.core.report import modal_template, shock_template

    for template in (shock_template, modal_template):
        assert not [b for b in template({}).blocks
                    if b.get('mode') == 'kurtosis'], template.__name__


def test_every_other_report_carries_the_reading():
    """A spectrum cannot answer the question it asks, whatever kind of
    test produced the record — wherever the nominal means something."""
    from visualdynamics.core import report as templates

    for name in ('random', 'sine', 'sysid', 'transient'):
        blocks = getattr(templates, f'{name}_template')({}).blocks
        kurtosis = [b for b in blocks if b.get('mode') == 'kurtosis']
        assert len(kurtosis) == 1, name
        assert kurtosis[0]['source'] == '@basis:TimeHistory', name
        # a computed judgment, so it sits with the others rather than
        # beside the traces — the standing order
        assert blocks.index(kurtosis[0]) > max(
            (i for i, b in enumerate(blocks)
             if b.get('mode') == 'stage'), default=-1), name


def test_the_report_figure_bands_and_colors_the_bars():
    """Gray inside the band, red above, blue below, with the ground
    past each threshold shaded to match — the same reading the app
    gives and the same one every other bar chart here gives."""
    from visualdynamics.core.kurtosis import HIGH, LOW
    from visualdynamics.report import _build_block

    rng = np.random.default_rng(9)
    t = np.arange(8192) / 2048.0
    rows = rng.standard_normal((3, len(t)))
    rows[0, ::300] *= 18.0                 # peaky: over the band
    rows[2] = np.clip(rows[2], -0.9, 0.9)  # clipped: under it
    data = visualdynamics.TimeHistory(
        t, rows, response_dof=['101Z+', '104Z+', '110Z+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')
    built = _build_block({'kind': 'bars', 'mode': 'kurtosis',
                          'source': 'T', 'caption': 'Kurtosis'},
                         {'T': data}, visualdynamics.SI, [])
    assert built['kind'] == 'bars'
    assert (built['low'], built['high']) == (LOW, HIGH)
    # the bars grow from the nominal, so what a bar's length says is
    # the departure from Gaussian and its side says which way
    assert built['baseline'] == NOMINAL
    assert built['floor'] is None, 'the axis follows the bars'
    assert 'Pearson' in built['ylabel']
    assert built['values'][0] > HIGH, 'the peaky channel is over'
    assert built['values'][2] < LOW, 'the clipped one is under'
    assert LOW <= built['values'][1] <= HIGH, 'and the plain one is in'


# ---- which stretch of the record is read ---------------------------------


def _quiet_headed(rate=2048.0, seconds=8.0, live=(3.0, 6.0)):
    """A record like a real run: a Gaussian middle with silence before
    and after it, and the analysis pointed at the middle."""
    from visualdynamics.core.averaging import Averaging

    rng = np.random.default_rng(1)
    n = int(rate * seconds)
    row = rng.standard_normal(n) * 1e-4          # the silent lead-in
    first, last = int(live[0] * rate), int(live[1] * rate)
    row[first:last] = rng.standard_normal(last - first)
    history = visualdynamics.TimeHistory(
        np.arange(n) / rate, np.stack([row]), response_dof=['101Z+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')
    history.averaging = Averaging(
        frame_length=2048, frames=5, overlap=0.5, window='hann',
        start=live[0])
    return history


def test_the_reading_covers_what_the_spectra_cover():
    """A system-ID excitation read whole came out at 4.23 where every
    steady frame of it read 2.92 (Brandon, 2026-08-24). Nothing was
    wrong with the arithmetic: the record is only a quarter analyzed,
    and whole it is a mixture of a loud distribution and a quiet one,
    which genuinely is not Gaussian. A number that disagrees with the
    PSD printed beside it is worse than no number."""
    from visualdynamics.core.kurtosis import analyzed_span

    history = _quiet_headed()
    whole = kurtosis(np.real(history.ordinate[0]))
    assert whole > HIGH, 'read whole, the silence dominates the shape'

    spans, phrase = analyzed_span(history)
    assert len(spans) == 1 and spans[0][0] > 0
    assert 'averaged over' in phrase
    (_label, value), = channel_kurtosis(history)
    assert value == pytest.approx(NOMINAL, abs=0.2), \
        'over its own frames the record reads Gaussian, as it is'
    assert LOW < value < HIGH


def test_a_shock_record_is_read_over_its_windows():
    """All of them, pooled: the shape of a distribution does not care
    what order its samples arrived in, and the chart is one bar per
    channel (Brandon, 2026-08-24). The silence between events is not
    part of what was analyzed, and read whole it swamps everything."""
    from visualdynamics.core.kurtosis import analyzed_span
    from visualdynamics.core.shocks import Shock

    rng = np.random.default_rng(8)
    rate, n = 4096.0, 4096 * 4
    row = rng.standard_normal(n) * 1e-3
    for start in (0.5, 1.5, 2.5):
        at = int(start * rate)
        row[at:at + 400] = rng.standard_normal(400)
    history = visualdynamics.TimeHistory(
        np.arange(n) / rate, np.stack([row]), response_dof=['101Z+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')
    history.shocks = tuple(Shock(start, 400 / rate)
                           for start in (0.5, 1.5, 2.5))
    spans, phrase = analyzed_span(history)
    assert len(spans) == 3, 'every window, not just the first'
    assert 'shock windows' in phrase
    (_label, value), = channel_kurtosis(history)
    assert value == pytest.approx(NOMINAL, abs=0.5), \
        'the events themselves are Gaussian here; the silence is not'
    assert kurtosis(np.real(history.ordinate[0])) > 10, \
        'and read whole, the silence between them swamps it'


def test_a_record_with_neither_is_read_whole():
    """Nothing has said which part matters, so all of it does."""
    from visualdynamics.core.kurtosis import analyzed_span

    rng = np.random.default_rng(6)
    history = visualdynamics.TimeHistory(
        np.arange(2048) / 1024.0, rng.standard_normal((1, 2048)),
        response_dof=['101Z+'], ordinate_dim='acceleration',
        ordinate_unit='m/s**2')
    spans, phrase = analyzed_span(history)
    assert spans == [(0, 2048)] and phrase == 'over the record'


def test_the_axis_is_measured_from_the_baseline(qt_app):
    """Bars around three took their reach from the origin and fenced
    the axis at -5, crowding every bar into the right-hand quarter of
    a chart that was mostly empty (Brandon, 2026-08-24)."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import error_chart, kurtosis_chart
    from visualdynamics.theme import theme as resolve_theme

    colors = resolve_theme('light')
    widget = pg.GraphicsLayoutWidget()
    plot = widget.addPlot(row=0, col=0)
    kurtosis_chart(plot, [('101Z+', 4.25), ('104Z+', 4.30),
                          ('110Z+', 4.22)], colors)
    low, high = plot.viewRange()[0]
    assert low > 1.0, f'the axis does not reach down to zero: {low}'
    assert LOW - 1 < low < LOW, 'it brackets the lower threshold'
    assert HIGH < high < 5.0, 'and clears the bars past the upper one'

    # and a reading whose baseline *is* zero is untouched: symmetric
    # about the origin, exactly as it always was
    # held, or Qt collects the widget out from under the plot
    second = pg.GraphicsLayoutWidget()
    other = second.addPlot(row=0, col=0)
    error_chart(other, [('101Z+', -1.2, 4.0), ('104Z+', 2.5, 1.0)],
                colors)
    lo, hi = other.viewRange()[0]
    assert lo == pytest.approx(-hi), 'still centered on zero'


def test_the_system_id_reads_both_of_its_streams():
    """An ambient recording that is not Gaussian is a noisy room or a
    rattling fixture; a drive that is not is a different fault
    entirely. Both streams get a figure (Brandon, 2026-08-24), the
    quiet one first, as the time data reads."""
    import numpy as np

    from visualdynamics.core.report import sysid_template

    t = np.arange(4096) / 1024.0
    rng = np.random.default_rng(12)

    def stream(scale):
        return visualdynamics.TimeHistory(
            t, rng.standard_normal((2, len(t))) * scale,
            response_dof=['101Z+', '104Z+'],
            ordinate_dim='acceleration', ordinate_unit='m/s**2')

    objects = {'Quiet': stream(0.01), 'Loud': stream(1.0)}
    blocks = [b for b in sysid_template(objects).blocks
              if b.get('mode') == 'kurtosis']
    assert [b['source'] for b in blocks] == ['Quiet', 'Loud']
    assert 'ambient' in blocks[0]['caption']
    assert 'excitation' in blocks[1]['caption']
    # a spectral package has no streams at all, and keeps one slot
    package = [b for b in sysid_template({}).blocks
               if b.get('mode') == 'kurtosis']
    assert [b['source'] for b in package] == ['@basis:TimeHistory']


def test_a_channel_that_recorded_nothing_is_counted_out_loud():
    """The ambient stream's drive channels are exactly zero — the
    shakers were off — so they have no shape to describe. Eight bars
    from a twelve-channel record otherwise reads as a chart of
    everything."""
    import numpy as np

    from visualdynamics.report import _build_block

    t = np.arange(2048) / 1024.0
    rng = np.random.default_rng(13)
    rows = rng.standard_normal((3, len(t)))
    rows[2] = 0.0                      # a dead channel
    data = visualdynamics.TimeHistory(
        t, rows, response_dof=['101Z+', '104Z+', '9001X+'],
        ordinate_dim=['acceleration'] * 2 + ['force'],
        ordinate_unit=['m/s**2'] * 2 + ['N'])
    built = _build_block({'kind': 'bars', 'mode': 'kurtosis',
                          'source': 'T', 'caption': 'K'},
                         {'T': data}, visualdynamics.SI, [])
    assert len(built['values']) == 2, 'the dead channel has no bar'
    assert '1 channel recorded nothing' in built['caption']


def test_the_other_readings_stay_offered_while_this_one_is_up(window, pump):
    """Reversed on 2026-08-27, and worth saying why rather than just
    editing the assertions.

    The rule used to be the other way: kurtosis replaces the trace, so
    the frames and the shock windows had nothing to be drawn on and
    their buttons went away (Brandon, 2026-08-24). That reasoning was
    about *marks on a trace* and stopped holding when the five became a
    visible group of readings — the way out of a reading is to pick
    another one, and hiding the alternatives left the group showing
    only whichever button had been pressed, with no way across except
    unpicking it first. Their panels still go away; it is the buttons
    that stay.
    """
    _record(window, pump)
    pane = window.data_pane
    assert pane.averaging_action.isVisible()
    assert pane.shocks_action.isVisible()
    assert pane.kurtosis_action.isVisible()

    pane.kurtosis_action.setChecked(True)
    pane._choose_reading(True)
    window.render_current()
    pump()
    assert pane.kurtosis_action.isVisible(), 'the way back stays'
    assert pane.averaging_action.isVisible(), 'and the way across'
    assert pane.shocks_action.isVisible()
    assert not pane.averaging_panel.isVisible(), 'the panels do go'
    assert not pane.shock_panel.isVisible()

    pane.kurtosis_action.setChecked(False)
    pane._choose_reading(False)
    window.render_current()
    pump()
    assert pane.averaging_action.isVisible(), 'and they come back'
    assert pane.shocks_action.isVisible()
    assert not pane.averaging_action.isChecked(), 'unchecked, as left'


def test_a_chart_of_many_channels_fits_the_pane_it_is_given(qt_app):
    """260 rows — a 20-impact, 13-channel modal run — used to demand
    ROW_HEIGHT apiece, ~4700 px of minimum height in a pane that has
    no scroll bars: everything past the pane's bottom edge was simply
    clipped, and zoom operates on the data range, so no gesture could
    reach the missing bars (Brandon, 2026-08-28). The chart fits the
    height it is given; all the bars on screen is the opening shape.
    """
    import pyqtgraph as pg

    from visualdynamics.plot.bars import kurtosis_chart
    from visualdynamics.theme import theme as resolve_theme

    colors = resolve_theme('light')
    widget = pg.GraphicsLayoutWidget()
    widget.resize(400, 500)
    plot = widget.addPlot(row=0, col=0)
    rows = [(f'{100 + i}Z+', 3.0 + (i % 7) * 0.3) for i in range(260)]
    kurtosis_chart(plot, rows, colors)
    qt_app.processEvents()
    assert widget.ci.boundingRect().height() <= 510, \
        'the chart does not overflow the pane'
    # and the opening view really does hold every bar
    low, high = plot.viewRange()[1]
    assert low <= 0 and high >= len(rows) - 1
