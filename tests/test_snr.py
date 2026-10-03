"""Signal to noise, the subtracted form (`visualdynamics.core.snr`,
2026-10-03): the driven power less the ambient, over the ambient — line
by line on the curve, once per channel on the bars — and the bars in
the app, the headless plot and the system ID report."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import web_close, web_read, web_view

import visualdynamics
from visualdynamics.core.data import Psd
from visualdynamics.core.snr import (
    AT_FLOOR,
    THRESHOLD_DB,
    rms_signal_to_noise,
    signal_to_noise,
)

FREQ = np.arange(1.0, 101.0)
NOISE = 1e-4


def _pair(multiples=(1001.0, 3.0, 1.0), dofs=('1X+', '2X+', '3X+'),
          dims=None):
    """Driven densities planted at `multiples` of a flat ambient: 1001
    times is a signal 1000 times the noise, 30 dB exactly; 3 times is
    3 dB; 1 time is no signal at all."""
    dims = dims or ['acceleration**2/frequency'] * len(dofs)
    ambient = Psd(FREQ, np.full((len(dofs), len(FREQ)), NOISE),
                  response_dof=list(dofs), ordinate_dim=dims)
    driven = Psd(FREQ, NOISE * np.asarray(multiples)[:, None]
                 * np.ones((len(dofs), len(FREQ))),
                 response_dof=list(dofs), ordinate_dim=dims)
    return driven, ambient


def test_the_curve_takes_the_noise_off_before_dividing():
    """1001 times the floor is 30 dB of signal, not 30.004 dB of
    quotient; twice the floor plus one is 3 dB; equal to the floor is
    no signal, NaN, never 0 dB and never minus infinity."""
    driven, ambient = _pair()
    _x, rows, dofs, _dims = signal_to_noise(driven, ambient)
    assert dofs == ['1X+', '2X+', '3X+']
    db = 10 * np.log10(rows[:2])
    assert np.allclose(db[0], 30.0)
    assert np.allclose(db[1], 10 * np.log10(2.0))
    assert np.all(np.isnan(rows[2])), 'at the floor there is no signal'


def test_a_driven_line_below_its_floor_is_no_signal():
    driven, ambient = _pair(multiples=(0.5,), dofs=('1X+',))
    _x, rows, _dofs, _dims = signal_to_noise(driven, ambient)
    assert np.all(np.isnan(rows))


def test_the_rms_reading_is_one_number_per_channel():
    """The area under each density over the shared band: a flat 1001
    times the floor is 30 dB; the channel at its floor is NaN."""
    driven, ambient = _pair()
    rows, silent = rms_signal_to_noise(driven, ambient)
    assert silent == []
    assert [label for label, _v in rows] == ['1X+', '2X+', '3X+']
    assert rows[0][1] == pytest.approx(30.0)
    assert rows[1][1] == pytest.approx(10 * np.log10(2.0))
    assert np.isnan(rows[2][1])


def test_the_rms_reading_weighs_the_band_by_its_power():
    """Not a mean of line ratios: half the band at 1001 times the
    floor and half at the floor is a signal of 500 noise powers, 27 dB
    — where the median line would have said 0 or 30."""
    ambient = Psd(FREQ, np.full((1, len(FREQ)), NOISE), response_dof=['1X+'])
    values = np.full((1, len(FREQ)), NOISE)
    values[0, :50] *= 1001.0
    driven = Psd(FREQ, values, response_dof=['1X+'])
    ((_label, db),), _silent = rms_signal_to_noise(driven, ambient)
    assert db == pytest.approx(10 * np.log10(500.0))


def test_one_dof_with_two_quantities_names_both():
    """A drive point's accelerometer and load cell share a DOF; the
    bars must not label them alike."""
    driven, ambient = _pair(
        multiples=(1001.0, 101.0), dofs=('5Z+', '5Z+'),
        dims=['acceleration**2/frequency', 'force**2/frequency'])
    rows, _silent = rms_signal_to_noise(driven, ambient)
    labels = [label for label, _v in rows]
    assert labels == ['5Z+ (acceleration)', '5Z+ (force)'], labels


def test_the_selection_restricts_the_bars():
    driven, ambient = _pair()
    rows, _silent = rms_signal_to_noise(driven, ambient, records=[1])
    assert [label for label, _v in rows] == ['2X+']


def test_the_chart_judges_against_a_floor(qt_app):
    """One threshold, and too little is the fault: 3 dB and the
    channel at its floor are under 10 dB, 30 dB is not, and the count
    says so."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import snr_chart
    from visualdynamics.theme import theme

    driven, ambient = _pair()
    widget = pg.GraphicsLayoutWidget()
    chart = snr_chart(widget.addPlot(), rms_signal_to_noise(driven, ambient)[0],
                      theme('light'))
    assert chart.low == THRESHOLD_DB and chart.high is None
    assert [chart.beyond(v) for v in chart.values()] == \
        [None, 'under', 'under']
    assert chart.summary_text() == '2 of 3 channels under 10dB — 67%'
    assert chart.rows[2][0] == f'3X+ ({AT_FLOOR})'
    # the ground past the threshold is the floor's side: blue, below
    assert chart._span('low', THRESHOLD_DB)[1] == THRESHOLD_DB


def test_the_headless_plot_writes_a_file(tmp_path):
    from visualdynamics.plot import plot_snr

    driven, ambient = _pair()
    out = tmp_path / 'snr.png'
    plot_snr(driven, ambient, path=out, show=False)
    assert out.stat().st_size > 2000


def test_the_app_offers_the_bars_for_two_densities(window, pump):
    """Two PSDs selected: the third reading draws the bars, flat, and
    the console records the headless call that draws them."""
    driven, ambient = _pair()
    window.add_object('Excitation PSDs', driven)
    window.add_object('Noise PSDs', ambient)
    window._item_for_object('Excitation PSDs').setSelected(True)
    window._item_for_object('Noise PSDs').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane.spectra_actions['snr'].isVisible()
    pane.spectra_actions['snr'].trigger()
    pump()
    assert window.bar_chart.side == 'under'
    assert window.bar_chart.rows[0][0] == '1X+'
    assert not pane.waterfall_action.isVisible(), 'bars have no depth'
    assert '2 of 3 channels under 10dB' in \
        window.statusBar().currentMessage()
    assert any(line.startswith(
        "visualdynamics.plot.plot_snr(project['Excitation PSDs'], "
        "project['Noise PSDs']") for line in window.project.journal)


def test_the_report_block_carries_the_floor_and_the_gaps():
    from visualdynamics.report import _build_block

    driven, ambient = _pair()
    built = _build_block(
        {'kind': 'bars', 'mode': 'snr', 'source': 'Driven',
         'floor': 'Ambient', 'caption': 'RMS signal to noise'},
        {'Driven': driven, 'Ambient': ambient}, visualdynamics.SI, [])
    assert built['side'] == 'under' and built['low'] == THRESHOLD_DB
    assert built['values'][:2] == [30.0, round(10 * np.log10(2.0), 4)]
    assert built['values'][2] is None, 'no number, said as a gap'
    assert built['labels'][2] == f'3X+ ({AT_FLOOR})'


def test_the_system_id_report_carries_the_bars():
    from visualdynamics.core.report import sysid_template

    driven, ambient = _pair()
    report = sysid_template({'Excitation PSDs': driven,
                             'Noise PSDs': ambient})
    bars = [b for b in report.blocks
            if b.get('kind') == 'bars' and b.get('mode') == 'snr']
    assert len(bars) == 1
    assert bars[0]['source'] == 'Excitation PSDs'
    assert bars[0]['floor'] == 'Noise PSDs'


def test_the_page_draws_a_floor_chart_with_a_gap(tmp_path, qt_app):
    """The page script: a null value draws no bar, counts under the
    floor, and the drawing gets to the end of its rows."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl

    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    driven, ambient = _pair()
    report = Report('SNR', [{'kind': 'bars', 'mode': 'snr',
                             'source': 'Driven', 'floor': 'Ambient',
                             'caption': 'RMS signal to noise'}])
    path = tmp_path / 'snr.html'
    path.write_text(render_html(report, {'Driven': driven,
                                         'Ambient': ambient},
                                unit_system=visualdynamics.SI),
                    encoding='utf-8')
    view = web_view()
    view.resize(900, 600)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    try:
        drawn = web_read(
            view,
            "(() => { const c = document.querySelector('canvas.bars');"
            " return c && c.dataset.bars ? JSON.stringify("
            "{bars: +c.dataset.bars, out: +c.dataset.out}) : ''; })()")
    finally:
        web_close(view)
    assert drawn, 'the chart never finished drawing'
    got = json.loads(drawn)
    assert got['bars'] == 3
    assert got['out'] == 2, '3 dB and the channel at its floor are under 10 dB'


def test_a_silent_ambient_is_left_out_not_flagged():
    """A channel whose ambient recorded exactly nothing has no noise to
    divide by. Flagging it at the floor marked the plate's clean
    drive-point load cells as its worst channels (2026-10-03); it is
    left off the bars and the report says how many."""
    from visualdynamics.report import _build_block

    driven, ambient = _pair()
    quiet = np.array(ambient.ordinate, copy=True)
    quiet[0] = 0.0
    ambient = Psd(FREQ, quiet, response_dof=list(ambient.response_dof),
                  ordinate_dim=['acceleration**2/frequency'] * 3)
    rows, silent = rms_signal_to_noise(driven, ambient)
    assert silent == ['1X+']
    assert [label for label, _v in rows] == ['2X+', '3X+']
    built = _build_block(
        {'kind': 'bars', 'mode': 'snr', 'source': 'Driven',
         'floor': 'Ambient', 'caption': 'RMS signal to noise'},
        {'Driven': driven, 'Ambient': ambient}, visualdynamics.SI, [])
    assert len(built['values']) == 2
    assert '1 channel recorded no ambient noise' in built['caption']
