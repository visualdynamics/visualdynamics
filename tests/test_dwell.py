"""A dwell held as a segment of its own, and read against time.

The numbers are a one-mass resonator's, held to closed forms. Driven at
its own frequency from rest, a mode of quality factor Q reaches very
nearly 1 - exp(-pi n / Q) of its steady height after n cycles, so 95.7 %
after Q cycles and 99.8 % after 2Q whatever the frequency or the Q; and
exp(-pi n / Q) of it remains n cycles after the tone stops. The record is
a 1 g tone at 20 Hz sampled at 200 points a cycle, held for 3Q cycles and
then stopped, through Smallwood's ramp-invariant resonator
(`core.srs.ramp_invariant`, the filter the shock response spectrum
uses), with 2Q cycles of ring-down after it; at Q 30 that is 4.5 s on and
3 s off, and the tracked peak reads 28.72 g at 1.50 s.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.dwell import DWELL_SHARE, read_dwell
from visualdynamics.core.sine import (
    DWELL_SWEEP,
    LINEAR,
    LOG,
    SineSweepSpecification,
    SineTone,
)
from visualdynamics.core.sine_tracking import SineTracking, track_sine

G = 9.80665              # m/s**2 in one g
RATE = 4000.0            # 200 samples in each cycle of 20 Hz
FN = 20.0
CONTROL, RESPONSE = '1Z+', '101Z+'


def _resonator(x, q):
    """The absolute response of a resonator at `FN` and `q` on a base
    moving as `x`, from rest."""
    from scipy.signal import lfilter

    from visualdynamics.core.srs import ramp_invariant

    b, a = ramp_invariant([FN], RATE, 1.0 / (2.0 * q))
    return lfilter(b[0], a[0], x)


def _dwell_record(q=30.0, lead=0.0, on=None, off=None, tone=None):
    """A 1 g tone at `FN` held `on` cycles (3Q by default) after `lead`
    seconds of quiet, then `off` cycles (2Q) of nothing; the control
    and the resonator's response, in m/s**2, and a specification that
    holds the tone (`SineTone.dwell`, or `tone` when given)."""
    on = int(3 * q) if on is None else on
    off = int(2 * q) if off is None else off
    n = np.arange(round(lead * RATE) + (on + off) * 200)
    t = n / RATE
    local = t - lead
    x = np.sin(2.0 * np.pi * FN * local) * ((local >= 0.0)
                                            & (local < on / FN - 1e-12))
    y = _resonator(x, q)
    history = TimeHistory(t, np.vstack([x, y]) * G,
                          response_dof=[CONTROL, RESPONSE],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['m/s**2'] * 2)
    tone = SineTone.dwell('Dwell', FN, [G], on / FN) if tone is None else tone
    spec = SineSweepSpecification([tone], [CONTROL], ordinate_unit='m/s**2')
    return history, spec


# ---- the dwell as a segment ------------------------------------------------

#: the hair of a hertz a dwell at 20 Hz sweeps across, as `SineTone.dwell`
#: writes it
HAIR = FN * DWELL_SWEEP


def test_a_dwell_holds_its_frequency_for_its_seconds():
    """Written as a controller writes one: a linear segment across a
    millionth of its frequency, crossed in the time it holds."""
    tone = SineTone.dwell('Hold', 20.0, [2.0], 4.5, start_time=1.0)
    assert tone.segment_type.tolist() == [LINEAR]
    assert tone.frequency.tolist() == [20.0, 20.0 + HAIR]
    assert tone.duration() == pytest.approx(4.5)
    assert tone.span() == pytest.approx((1.0, 5.5))
    t = np.array([0.0, 1.0, 4.5])
    assert tone.frequency_at(t) == pytest.approx([20.0] * 3, rel=2e-6)
    assert tone.phase_at(t) == pytest.approx(2.0 * np.pi * 20.0 * t,
                                             rel=2e-6), \
        'the phase grows at the held frequency, 90 cycles in 4.5 s'
    assert tone.target([20.0, 21.0])[0, 0] == 2.0
    assert np.isnan(tone.target([20.0, 21.0])[1, 0]), \
        'over frequency a dwell is one point, and nothing else is specified'
    assert tone.is_dwell(0) and tone.dwell_segments() == [0]


def test_a_dwell_holds_for_some_time_and_a_zero_span_is_still_refused():
    with pytest.raises(ValueError, match='more than zero seconds'):
        SineTone.dwell('Bad', 20.0, [1.0], 0.0)
    with pytest.raises(ValueError, match='SineTone.dwell builds one'):
        SineTone('Bad', 0.0, [20.0, 20.0], [[1.0], [1.0]], [LINEAR], [1.0])


def test_a_sweep_and_a_dwell_share_one_tone():
    """10 to 20 Hz at 5 Hz/s (2 s), 20 Hz held 3 s, then 20 to 40 Hz at
    60 oct/min (1 s): the dwell is one more segment of the sweep law."""
    tone = SineTone('Mixed', 0.0, [10.0, 20.0, 20.0 + HAIR, 40.0],
                    [[1.0], [2.0], [2.0], [4.0]], [LINEAR, LINEAR, LOG],
                    [5.0, HAIR / 3.0, 60.0])
    # the log segment starts a hair above 20 Hz, so its octave is a
    # millionth short
    assert tone.segment_seconds() == pytest.approx([2.0, 3.0, 1.0], rel=2e-6)
    assert tone.frequency_at([1.0, 3.5, 5.5]) == pytest.approx(
        [15.0, 20.0, 20.0 * np.sqrt(2.0)], rel=2e-6)
    cycles = tone.phase_at([2.0, 5.0]) / (2.0 * np.pi)
    assert cycles[1] - cycles[0] == pytest.approx(60.0, rel=2e-6), \
        'the phase runs on through the dwell at 20 Hz'
    assert tone.dwell_segments() == [1]


def test_a_controllers_near_dwell_reads_as_a_dwell():
    """A controller cannot hold a zero span, so its file writes a dwell
    as a sweep across a ten-thousandth of a hertz; that is a dwell in
    all but name. A tenth of a hertz is a sweep."""
    near = SineTone('Near', 0.0, [20.0, 20.0001], [[1.0], [1.0]], [LINEAR],
                    [0.0001 / 4.5])
    assert near.is_dwell(0) and near.dwell_segments() == [0]
    swept = SineTone('Swept', 0.0, [20.0, 20.1], [[1.0], [1.0]], [LINEAR],
                     [0.1 / 4.5])
    assert not swept.is_dwell(0) and swept.dwell_segments() == []


def test_a_dwell_round_trips_through_native_as_an_ordinary_segment(tmp_path):
    spec = SineSweepSpecification(
        [SineTone.dwell('Hold', 20.0, [2.0], 4.5, warning_upper=[2.5])],
        [CONTROL], ordinate_unit='m/s**2')
    path = str(tmp_path / 'dwell.vdyn')
    visualdynamics.io.save(spec, path)
    back = visualdynamics.io.load(path)
    assert back == spec
    assert back.tones[0].segment_type.tolist() == [LINEAR]
    assert back.tones[0].dwell_segments() == [0]
    assert back.tones[0].duration() == pytest.approx(4.5)


# ---- the one-mass numbers ------------------------------------------------


@pytest.mark.parametrize('q', [20.0, 30.0, 50.0])
def test_q_cycles_reach_95_7_percent_and_2q_99_8(q):
    reading = read_dwell(*_dwell_record(q), onset=0.0)
    assert round(reading.share_at(q), 3) == 0.957, \
        '1 - exp(-pi) after Q cycles, whatever the Q'
    assert round(reading.share_at(2.0 * q), 3) == 0.998
    assert reading.settled and reading.notes == ()


@pytest.mark.parametrize('setting', [
    SineTracking(detector='peak'), SineTracking(detector='rms'),
    SineTracking(detector='mean')], ids=lambda s: s.describe())
def test_about_4_3_percent_remains_q_cycles_after_the_stop_at_q_30(setting):
    """exp(-pi) of the response remains Q cycles after the stop, for the
    peak detector as for the others. The peak over the thirtieth cycle
    after the stop, reported at its end, says 4.8 %: the detector's late
    report, a cycle behind the response, which the reading takes out."""
    reading = read_dwell(*_dwell_record(30.0), onset=0.0, setting=setting)
    assert round(reading.remaining_at(30.0), 3) == 0.043
    late = reading.level[~reading.held][29] / reading.steady
    assert setting.detector != 'peak' or round(late, 3) == 0.048


def test_the_tracked_peak_reads_28_72_g_at_1_50_s():
    history, spec = _dwell_record(30.0)
    reading = read_dwell(history, spec, onset=0.0)
    assert reading.seconds[29] == pytest.approx(1.50)
    assert round(reading.level[29] / G, 2) == 28.72
    assert reading.steady / G == pytest.approx(30.0, abs=0.05)
    # the very detector track_sine reads with, at the same instants
    level = track_sine(history, spec, SineTracking(detector='peak'),
                       channels=[RESPONSE], onset=0.0, lines=90)[0]
    assert np.max(np.abs(np.abs(level.ordinate[0]) - reading.level[:90])) \
        < 1e-12


@pytest.mark.parametrize('q', [20.0, 30.0, 50.0])
def test_the_build_up_and_the_ring_down_take_about_q_cycles(q):
    """To 95 % and down to 5 % both take Q ln 20 / pi cycles, 0.95 Q:
    28.6 at Q 30."""
    reading = read_dwell(*_dwell_record(q), onset=0.0)
    expected = q * np.log(20.0) / np.pi
    assert reading.built == pytest.approx(expected, abs=0.1)
    assert reading.built_seconds == pytest.approx(reading.built / FN)
    assert reading.rung == pytest.approx(expected, abs=0.1)
    assert reading.rung_seconds == pytest.approx(reading.rung / FN)


@pytest.mark.parametrize('setting', [
    SineTracking(detector='peak'), SineTracking(detector='peak', cycles=2.0),
    SineTracking(detector='rms'), SineTracking(detector='mean'),
    SineTracking(proportional=0.5)], ids=lambda s: s.describe())
def test_the_ring_down_reads_as_long_as_the_build_up(setting):
    """One Q, one count each way, whatever the detector. The peak over a
    falling cycle is its first crest, a whole span before the reading is
    reported; counted on the reported clock the ring-down read 29.6
    cycles against 28.6 for the build-up. Each reading is counted where
    it stands, so the two agree to a tenth of a cycle."""
    reading = read_dwell(*_dwell_record(30.0), onset=0.0, setting=setting)
    assert abs(reading.rung - reading.built) < 0.1
    reported = np.interp(
        0.05, (reading.level[~reading.held] / reading.steady)[::-1],
        reading.after[~reading.held][::-1])
    if setting.detector == 'peak' and setting.kind == 'unfiltered':
        assert reported - reading.rung == pytest.approx(setting.cycles,
                                                        abs=0.05), \
            'the lag taken out is the span the peak looks back over'
    assert np.all(reading.stands <= reading.seconds)


def test_the_tone_is_found_in_the_record():
    """Placed by matched filter on the control: within a quarter of a
    cycle of where it starts (a held tone's phase, not its start, sets
    the last quarter), which moves a share after Q cycles by a few
    tenths of a percent."""
    history, spec = _dwell_record(30.0, lead=0.37)
    reading = read_dwell(history, spec)
    assert abs(reading.onset - 0.37) <= 0.25 / FN + 1.0 / RATE
    assert reading.dof == RESPONSE, 'a response by default, not the control'
    assert reading.share_at(30.0) == pytest.approx(0.957, abs=0.003)


def test_a_band_reads_the_same_steady_level_and_lags_the_build_up():
    history, spec = _dwell_record(30.0)
    peak = read_dwell(history, spec, onset=0.0)
    band = read_dwell(history, spec, onset=0.0,
                      setting=SineTracking(proportional=0.5))
    assert band.steady == pytest.approx(peak.steady, rel=2e-3)
    assert band.built > peak.built, 'a band settles, and reads a rise late'


def test_the_control_can_be_read_and_barely_builds():
    reading = read_dwell(*_dwell_record(30.0), onset=0.0, channel=CONTROL)
    assert reading.dof == CONTROL
    assert reading.steady / G == pytest.approx(1.0, abs=1e-3)
    assert reading.built == pytest.approx(1.0)


def test_another_share_reads_by_the_same_rule():
    reading = read_dwell(*_dwell_record(30.0), onset=0.0, share=0.9)
    expected = 30.0 * np.log(10.0) / np.pi
    assert reading.built == pytest.approx(expected, abs=0.2)
    assert reading.rung == pytest.approx(expected, abs=0.2)


# ---- what the reading says rather than hides --------------------------------


def test_a_dwell_too_short_to_settle_says_so():
    reading = read_dwell(*_dwell_record(50.0, on=30), onset=0.0)
    assert not reading.settled
    assert any('had not settled' in note for note in reading.notes)


def test_a_dwell_a_sweep_leaves_reads_no_ring_down():
    tone = SineTone('Then', 0.0, [20.0, 20.0 + HAIR, 40.0], [[G], [G], [G]],
                    [LINEAR, LINEAR], [HAIR / 4.5, 20.0])
    history, spec = _dwell_record(30.0, tone=tone)
    reading = read_dwell(history, spec, onset=0.0)
    assert reading.rung is None and not reading.stops
    assert any('no ring-down' in note for note in reading.notes)
    assert reading.seconds[-1] == pytest.approx(4.5)
    with pytest.raises(ValueError, match='no ring-down'):
        reading.remaining_at(10.0)


def test_a_dwell_a_sweep_runs_into_is_not_from_rest():
    tone = SineTone('Into', 0.0, [10.0, 20.0, 20.0 + HAIR], [[G], [G], [G]],
                    [LINEAR, LINEAR], [20.0, HAIR / 4.0])
    history, spec = _dwell_record(30.0, tone=tone)
    reading = read_dwell(history, spec, onset=0.0)
    assert reading.segment == 1 and reading.start == pytest.approx(0.5)
    assert any('not from rest' in note for note in reading.notes)


def test_a_record_that_ends_first_reads_no_ring_down_count():
    reading = read_dwell(*_dwell_record(30.0), onset=0.0, after=0.5)
    assert reading.rung is None
    assert any('record ends' in note for note in reading.notes)
    # ten cycles read after the stop; the last stands for the ninth
    assert reading.remaining_at(9.0) == pytest.approx(
        np.exp(-np.pi * 9.0 / 30.0), rel=0.03)
    with pytest.raises(ValueError, match='past the record'):
        reading.remaining_at(10.0)


def test_what_cannot_be_read_is_refused():
    history, spec = _dwell_record(30.0)
    with pytest.raises(ValueError, match='between zero and one'):
        read_dwell(history, spec, share=1.0)
    with pytest.raises(ValueError, match='no record'):
        read_dwell(history, spec, channel='999X+')
    swept = SineSweepSpecification(
        [SineTone('Up', 0.0, [20.0, 40.0], [[G], [G]], [LINEAR], [10.0])],
        [CONTROL])
    with pytest.raises(ValueError, match='holds a dwell'):
        read_dwell(history, swept)
    reading = read_dwell(history, spec, onset=0.0)
    with pytest.raises(ValueError, match='outside the dwell'):
        reading.share_at(200.0)


def _remade(history, ordinate, *, abscissa=None, unit='m/s**2'):
    """`history` with another ordinate (SI), clock or declared unit."""
    return TimeHistory(history.abscissa if abscissa is None else abscissa,
                       ordinate, response_dof=list(history.response_dof),
                       ordinate_dim=['acceleration'] * 2,
                       ordinate_unit=[unit] * 2)


def test_the_line_names_the_si_unit_whatever_the_record_declared():
    """The numbers are SI; a record declared in g still reads 294.3
    m/s², not '294.3 g', and in g only through a unit system."""
    from visualdynamics.units import DEFAULT_SYSTEM

    history, spec = _dwell_record(30.0)
    reading = read_dwell(_remade(history, history.ordinate, unit='g'), spec,
                         onset=0.0)
    assert reading.unit == 'g'
    assert reading.describe().startswith('steady 294.3 m/s², 95 % in 28.6 ')
    assert reading.describe(DEFAULT_SYSTEM).startswith('steady 30.01 g, ')


def test_a_record_that_starts_after_the_dwell_built_reads_no_build_up():
    """Cut 2 s (40 cycles) into the dwell, the first reading is already
    over 95 %: no count, and a note, rather than '95 % in 41.0 cycles'.
    Cut half a second in, the crossing is still in the record and reads
    the 28.6 cycles of the whole one."""
    history, spec = _dwell_record(30.0)
    for cut, built in ((2.0, None), (0.5, 28.6)):
        k = round(cut * RATE)
        late = _remade(history, history.ordinate[:, k:],
                       abscissa=history.abscissa[:-k])
        reading = read_dwell(late, spec, onset=-cut)
        assert reading.cycles[0] == pytest.approx(cut * FN + 1.0)
        if built is None:
            assert reading.built is None
            assert reading.notes == (
                ('the record starts 40.0 cycles into the dwell, after the '
                 'level had reached 95 % of its steady value, so no build-up '
                 'is read'),)
            assert reading.rung == pytest.approx(28.6, abs=0.1)
        else:
            assert reading.built == pytest.approx(built, abs=0.1)
            assert reading.notes == ()


def test_a_dead_channel_or_a_gap_in_the_tail_is_refused():
    """A channel at zero would read 'steady 0, 95 % in 1.0 cycles,
    settled'; a NaN in the last quarter would make every number NaN."""
    history, spec = _dwell_record(30.0)
    dead = history.ordinate.copy()
    dead[1] = 0.0
    with pytest.raises(ValueError, match=r'101Z\+ reads no level .*steady 0\)'):
        read_dwell(_remade(history, dead), spec, onset=0.0)
    gap = history.ordinate.copy()
    gap[1, 85 * 200] = np.nan
    with pytest.raises(ValueError, match=r'reads no level .*steady nan\)'):
        read_dwell(_remade(history, gap), spec, onset=0.0)


def test_a_dwell_past_the_records_end_is_refused_as_such():
    history, spec = _dwell_record(30.0)
    with pytest.raises(ValueError, match='holds no whole cycle of the dwell'):
        read_dwell(history, spec, onset=100.0)


# ---- the plot -----------------------------------------------------------------


def test_the_plot_names_each_share_by_shape_and_words(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.dwell import BUILT_SYMBOL, RUNG_SYMBOL, build_dwell

    reading = read_dwell(*_dwell_record(30.0), onset=0.0)
    layout = pg.GraphicsLayoutWidget()
    plot = build_dwell(layout, reading)
    names = [label.text for _sample, label in plot.legend.items]
    assert '95 % after 28.6 cycles (1.43 s)' in names
    assert '5 % 28.6 cycles after the stop (1.43 s)' in names
    assert any(name.startswith('steady 30.01') for name in names)
    symbols = {item.opts['symbol'] for item in plot.listDataItems()
               if item.opts['symbol'] is not None}
    assert symbols == {BUILT_SYMBOL, RUNG_SYMBOL}
    styles = {item.opts['pen'].style() for item in plot.listDataItems()
              if item.opts['pen'] is not None}
    from PySide6.QtCore import Qt
    assert {Qt.DashLine, Qt.DotLine, Qt.SolidLine} <= styles, \
        'the shares differ by line style, not by hue'
    level, = [item for item in plot.listDataItems()
              if item.name() and item.name().startswith(RESPONSE)]
    assert np.array_equal(level.getData()[0], reading.stands), \
        'drawn where each reading stands, so the triangle sits on the curve'


def test_the_plot_draws_to_a_file_with_no_window(tmp_path):
    from visualdynamics.plot import plot_dwell

    history, spec = _dwell_record(30.0)
    path = tmp_path / 'dwell.png'
    plot_dwell(history, spec, onset=0.0, path=str(path), show=False)
    assert path.stat().st_size > 10_000


# ---- the readout in the window --------------------------------------------------


def _shown(window, pump, history, spec):
    window.add_object('Record', history)
    window.add_object('Sweep', spec)
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    pane = window.data_pane
    pane.sine_action.setChecked(True)
    pane._choose_sine(True)
    pump()
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    return pane.sine_panel.dwell


def test_the_readout_reads_a_held_tone_in_the_sine_view(window, pump):
    from visualdynamics.units import SI

    window.unit_system = SI
    panel = _shown(window, pump, *_dwell_record(30.0))
    assert panel.isVisible()
    assert panel.channel_box.currentText() == RESPONSE
    assert panel.derived['frequency'].text() == '20 Hz'
    assert panel.derived['built'].text() == '28.6 cycles, 1.43 s'
    assert panel.derived_names['built'].text() == 'To 95 %'
    assert panel.derived['settled'].text() == 'yes'
    assert panel.preview.items, 'the level against time is drawn'
    assert window.project.journal[-1] == (
        "visualdynamics.plot.plot_dwell(project['Record'], project['Sweep'], "
        "tone='Dwell', segment=0, channel='101Z+', share=0.95, "
        "path='dwell.png', show=False)")


def test_the_readout_stays_away_beside_a_sweep(window, pump):
    from test_extract_sine import _flat_sweep

    panel = _shown(window, pump, *_flat_sweep(noise=1.0))
    assert window.data_pane.sine_panel.isVisible()
    assert not panel.isVisible(), 'a sweep holds no dwell (principle 3)'


def test_a_share_or_a_channel_edit_reads_again(window, pump):
    panel = _shown(window, pump, *_dwell_record(30.0))
    panel.share_box.setValue(90.0)
    pump()
    assert panel.reading.share == pytest.approx(0.9)
    assert panel.derived_names['rung'].text() == 'Down to 10 %'
    assert 'share=0.9,' in window.project.journal[-1]
    panel.channel_box.setCurrentText(CONTROL)
    pump()
    assert panel.reading.dof == CONTROL
    assert "channel='1Z+'" in window.project.journal[-1]
    assert DWELL_SHARE == 0.95


def test_a_stray_cycle_neither_moves_the_steady_level_nor_unsettles_it():
    """The steady level is a median, and settling is read from halves'
    medians, so one cycle three times too tall changes neither."""
    history, spec = _dwell_record(30.0)
    clean = read_dwell(history, spec, onset=0.0)
    spiked = history.ordinate.copy()
    spiked[1, int(80.5 * 200)] *= 3.0           # one sample in cycle 81
    history = TimeHistory(history.abscissa, spiked,
                          response_dof=list(history.response_dof),
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['m/s**2'] * 2)
    reading = read_dwell(history, spec, onset=0.0)
    assert reading.level[80] > 2.0 * clean.level[80], 'the stray is read'
    assert reading.steady == pytest.approx(clean.steady, rel=1e-4)
    assert reading.settled


def test_a_dwell_specification_draws_on_its_own(window, pump):
    """Selected alone, a held tone stands on the specification's stage
    as a line along time at one frequency, drawn without a refusal."""
    _history, spec = _dwell_record(30.0)
    window.add_object('Sweep', spec)
    select_objects(window, pump, 'Sweep')
    window.render_current()
    pump()
    assert window.statusBar().currentMessage().startswith(
        'Dwell: the requirement at 1 DOF of Sweep on the stage')
