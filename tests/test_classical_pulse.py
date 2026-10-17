"""The classical pulse check: a measured pulse judged against its nominal
shape, envelope and velocity change.

Pinned against arithmetic that does not depend on the implementation:
the area under a half-sine is 2AT/pi, under a terminal-peak sawtooth
AT/2 plus half its fall times the peak, under a trapezoid A(T - e) for
edges of e; so a 100 g, 11 ms half-sine changes velocity by 270.4 in/s,
a 31.4 g one (pi/4 of 40 g) by 84.9, a 30 g trapezoid with 1 ms edges by
115.8, and a 40 g sawtooth whose fall takes a tenth of its duration by
a tenth more than its nominal 0.5AT. The envelope is pinned to its band
of a fifth of the peak, and the verdicts to pulses made wrong on
purpose: too high, too long, too slow to fall, ringing after.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.classical_pulse import (
    BAND,
    QUIET,
    VELOCITY_WINDOW,
    ClassicalPulse,
    check_pulse,
    judge_pulse,
    measure_pulse,
    pulse_channel,
    pulse_envelope,
    suggest_pulse,
)
from visualdynamics.core.data import TimeHistory

G = 9.80665
INCH = 0.0254
RATE = 102400.0
T = 0.011


def record(span=0.16, opens=0.06, rate=RATE):
    """Sample times over `span` seconds, and the pulse's start."""
    return np.arange(round(span * rate)) / rate, opens


def pulse(shape='half-sine', peak=50.0, duration=T, edge=None):
    return ClassicalPulse(shape, peak * G, duration, edge)


def judged(nominal, made=None, **tolerances):
    """`made` (the nominal by default) sampled from 60 ms into a record,
    judged against `nominal`."""
    t, opens = record()
    made = nominal if made is None else made
    return judge_pulse(t, made.at(t - opens), nominal, **tolerances)


def in_s(value):
    return value / INCH


# ---- the velocity change -----------------------------------------------------


@pytest.mark.parametrize('peak,change', [(100.0, 270.4), (10 * np.pi, 84.9)])
def test_a_half_sine_changes_velocity_by_2at_over_pi(peak, change):
    """The chapter's numbers: 270.4 in/s for 100 g, 11 ms, and 84.9 for
    the 31.4 g half-sine of equal velocity to a 40 g sawtooth. The
    nominal is the formula; the measured is the record integrated over
    the window, and the two agree."""
    nominal = pulse(peak=peak)
    check = judged(nominal)
    assert round(in_s(nominal.velocity_change), 1) == change
    assert round(in_s(check.velocity_change), 1) == change
    assert check.velocity_ratio == pytest.approx(1.0, abs=1e-4)


def test_a_sawtooth_falling_in_its_whole_allowance_is_a_tenth_over():
    """A 40 g, 11 ms sawtooth whose fall takes 1.1 ms changes velocity
    by 93.4 in/s against the nominal 0.5AT of 84.9: a tenth over,
    because the fall adds half its length times the peak. In tolerance
    under a fifth, so it passes; the fall is read at its 1.1 ms."""
    nominal = pulse('sawtooth', 40.0)
    check = judged(nominal, pulse('sawtooth', 40.0, edge=0.1 * T))
    assert round(in_s(nominal.velocity_change), 1) == 84.9
    # the formula leaves the fall out even where the nominal has one
    assert round(in_s(pulse('sawtooth', 40.0, edge=0.1 * T)
                      .velocity_change), 1) == 84.9
    assert round(in_s(check.velocity_change), 1) == 93.4
    assert check.velocity_ratio == pytest.approx(1.10, abs=1e-3)
    assert check.measured.fall == pytest.approx(0.1 * T, abs=1.0 / RATE)
    # the start and the end, the fall included, read off the record
    assert check.measured.start == pytest.approx(0.06, abs=1e-6)
    assert check.measured.end == pytest.approx(0.06 + 1.1 * T, abs=1e-6)
    # the rise is read to its last sample, the record's nearest to the
    # crest, which falls between two
    assert check.measured.duration == pytest.approx(
        np.floor((0.06 + T) * RATE) / RATE - 0.06, abs=1e-9)
    assert check.fall_ok and check.passed
    assert 'velocity change +10.0 % of nominal' in check.describe()


def test_a_trapezoids_change_counts_half_of_each_edge():
    nominal = pulse('trapezoid', 30.0, edge=0.001)
    check = judged(nominal)
    assert round(in_s(nominal.velocity_change), 1) == 115.8
    assert check.velocity_ratio == pytest.approx(1.0, abs=1e-4)
    assert check.measured.rise == pytest.approx(0.001, abs=1e-6)
    assert check.measured.fall == pytest.approx(0.001, abs=1e-6)
    # each edge extended to zero: the start, the end, and the span
    # between them the duration
    assert check.measured.start == pytest.approx(0.06, abs=1e-6)
    assert check.measured.end == pytest.approx(0.06 + T, abs=1e-6)
    assert check.measured.duration == pytest.approx(T, abs=1e-6)


def test_the_velocity_window_runs_from_0_4_before_to_0_1_after():
    """Placed on the measured pulse, in nominal durations: from 4.4 ms
    before an 11 ms pulse to 1.1 ms after it ends, and the velocity
    starts from rest at its first sample."""
    check = judged(pulse())
    before, after = VELOCITY_WINDOW
    assert (before, after) == (0.4, 0.1)
    assert check.window[0] == pytest.approx(0.06 - 0.4 * T, abs=1e-6)
    assert check.window[1] == pytest.approx(0.06 + T + 0.1 * T, abs=1e-6)
    assert check.velocity[0] == 0.0
    assert check.velocity_time[0] >= check.window[0]
    assert check.velocity_time[-1] <= check.window[1]


def test_the_velocity_is_integrated_raw():
    """A compensated pulse on a shaker: the pulse with a shallow pulse
    of the other sign either side, carrying its area between them, so
    the record as a whole changes velocity by nothing. The window keeps
    the main pulse's change; a mean taken out first would not."""
    t, opens = record(0.5, 0.25)
    nominal = pulse(peak=50.0)
    depth, length = 2.5 * G, 0.11
    compensation = ClassicalPulse('half-sine', -depth, length)
    made = (nominal.at(t - opens) + compensation.at(t - opens + length)
            + compensation.at(t - opens - T))
    assert abs(np.trapezoid(made, t)) < 1e-3 * abs(nominal.velocity_change)
    check = judge_pulse(t, made, nominal)
    assert check.passed
    assert check.velocity_ratio == pytest.approx(1.0, abs=0.01)


def test_the_velocity_goes_through_the_packages_raw_integration(
        monkeypatch):
    """One trapezoid rule in the package: the window is integrated by
    `filters.integrate` with no drift corner, as the shock check
    integrates, and its answer is the one the check reports."""
    from visualdynamics.core import filters

    calls = []
    real = filters.integrate

    def counted(data, drift_corner=filters.DRIFT_CORNER):
        calls.append(drift_corner)
        return real(data, drift_corner)

    monkeypatch.setattr(filters, 'integrate', counted)
    check = judged(pulse(peak=100.0))
    assert calls == [None]
    assert round(in_s(check.velocity_change), 1) == 270.4


def test_unevenly_spaced_samples_are_refused():
    """The integration steps by one sample time, so a record whose
    samples are not evenly spaced is refused rather than integrated as
    though they were."""
    t, opens = record()
    t = t.copy()
    t[100:] += 0.5 / RATE
    nominal = pulse()
    with pytest.raises(ValueError, match='A pulse check needs evenly '
                                         'spaced time samples'):
        judge_pulse(t, nominal.at(t - opens), nominal)


# ---- the envelope ------------------------------------------------------------


@pytest.mark.parametrize('nominal', [
    pulse(), pulse('sawtooth', 40.0), pulse('sawtooth', 20.0, edge=0.0005),
    pulse('trapezoid', 30.0, edge=0.001), pulse(peak=-50.0),
    pulse('sawtooth', -40.0)],
    ids=['half-sine', 'sawtooth', 'sawtooth-with-fall', 'trapezoid',
         'negative-half-sine', 'negative-sawtooth'])
def test_the_nominal_pulse_passes_its_own_envelope(nominal):
    check = judged(nominal)
    assert check.covered
    assert check.outside_count == 0
    assert check.passed, check.failures()
    assert check.describe().startswith('Passes: every point inside')


def test_the_band_is_a_fifth_of_the_peak_off_the_pulse_and_on_it():
    """Away from the pulse the envelope is plus or minus a fifth of the
    peak; at a half-sine's crest its top is 1.2 of the peak, and its
    bottom, the lowest of every admissible duration less a fifth, is
    sin(0.55 pi) - 0.2 = 0.788 of it at the highest."""
    nominal = pulse(peak=40.0)
    t = np.linspace(0.0, T, 2001)
    lower, upper = pulse_envelope(nominal, np.concatenate(
        [[-0.02, 0.05], t]))
    assert BAND == 0.2
    assert (lower[0], upper[0]) == pytest.approx((-8.0 * G, 8.0 * G))
    assert (lower[1], upper[1]) == pytest.approx((-8.0 * G, 8.0 * G))
    assert upper[2:].max() == pytest.approx(48.0 * G)
    assert lower[2:].max() == pytest.approx(
        (np.sin(0.55 * np.pi) - 0.2) * 40.0 * G, rel=1e-4)


def test_the_peak_is_held_where_the_envelopes_crest_dips():
    """So a pulse at 0.79 of its peak is inside the envelope at every
    point and still fails on its peak, which is held to 0.8 apart."""
    check = judged(pulse(), pulse(peak=0.79 * 50.0))
    assert check.envelope_ok
    assert not check.peak_ok
    assert check.failures()[0] == 'peak \u221221.0 % of nominal'


def test_the_envelope_admits_the_duration_tolerance():
    """Built from every admissible pulse, so a pulse a twentieth long is
    inside the envelope and inside its duration tolerance at once, and
    one a fifth long is outside both. At 1.09 the tail of the long pulse
    stands 0.26 of the peak above the nominal's zero, past the band: an
    envelope drawn round the nominal alone would fail what the duration
    tolerance passes."""
    nominal = pulse()
    assert np.sin(np.pi / 1.09) > BAND
    assert judged(nominal, pulse(duration=1.09 * T)).passed
    long = judged(nominal, pulse(duration=1.2 * T))
    assert not long.duration_ok and not long.envelope_ok
    assert long.measured.duration == pytest.approx(1.2 * T, rel=1e-3)
    assert 'duration +20.0 % of nominal' in long.describe()


def test_the_judged_span_is_the_quiet_either_side():
    """From 2.5 durations before the measured start to 2.5 after the
    latest admissible end."""
    check = judged(pulse())
    assert QUIET == 2.5
    assert check.span[0] == pytest.approx(0.06 - 2.5 * T, abs=1e-6)
    assert check.span[1] == pytest.approx(0.06 + 1.1 * T + 2.5 * T,
                                          abs=1e-6)
    assert check.time[0] >= check.span[0]
    assert check.time[-1] <= check.span[1]


# ---- the verdicts ------------------------------------------------------------


def test_a_pulse_a_quarter_high_fails_at_its_peak():
    nominal = pulse()
    check = judged(nominal, pulse(peak=62.5))
    assert not check.passed
    assert not check.peak_ok
    assert check.peak_ratio == pytest.approx(1.25, abs=1e-4)
    out = check.time[check.outside]
    # the samples out are around the crest, and only there
    assert out.size and np.all(np.abs(out - (0.06 + T / 2)) < T / 4)
    assert 'peak +25.0 % of nominal' in check.describe()


def test_a_pulse_15_percent_high_is_held_by_the_velocity_tolerance():
    """Inside a fifth's band at every point (1.15 of the peak is under
    1.2) and inside a fifth on velocity, so it passes; held to a tenth
    on the velocity change it fails, on that alone."""
    nominal = pulse()
    high = pulse(peak=57.5)
    check = judged(nominal, high)
    assert check.passed
    assert check.velocity_ratio == pytest.approx(1.15, abs=1e-3)
    tight = judged(nominal, high, velocity_tolerance=0.1)
    assert not tight.passed
    assert tight.failures() == ['velocity change +15.0 % of nominal']


def test_a_slow_fall_is_flagged():
    """A sawtooth that takes a fifth of its duration to fall is past
    the tenth allowed, and says so in milliseconds."""
    check = judged(pulse('sawtooth', 40.0), pulse('sawtooth', 40.0,
                                                  edge=0.2 * T))
    assert not check.fall_ok and not check.passed
    assert 'fall of 2.2 ms against 1.1 ms' in check.describe()


def test_a_slow_trapezoid_rise_is_flagged():
    check = judged(pulse('trapezoid', 30.0, edge=0.001),
                   pulse('trapezoid', 30.0, edge=0.002))
    assert not check.rise_ok and not check.fall_ok
    assert 'rise or fall of 2 ms against 1.1 ms' in check.describe()


def test_ringing_after_the_pulse_is_outside_the_band():
    """A fixture ringing at 400 Hz after the pulse, starting at a third
    of the peak: the pulse itself is in, the samples after it are out."""
    t, opens = record()
    nominal = pulse()
    after = np.clip(t - opens - T, 0.0, None)
    ring = np.where(t > opens + T, (50.0 * G / 3.0) * np.exp(-after / 0.004)
                    * np.sin(2 * np.pi * 400.0 * after), 0.0)
    check = judge_pulse(t, nominal.at(t - opens) + ring, nominal)
    assert check.peak_ok and check.duration_ok
    assert not check.envelope_ok
    assert np.all(check.time[check.outside] > opens + T)


def test_a_record_that_stops_short_is_not_covered():
    """The quiet after the pulse is part of what is judged; a record cut
    a duration after the pulse cannot show it, and is flagged for it."""
    t = np.arange(round(0.09 * RATE)) / RATE
    nominal = pulse()
    check = judge_pulse(t, nominal.at(t - 0.06), nominal)
    assert check.envelope_ok
    assert not check.covered and not check.passed
    assert 'does not reach across the quiet' in check.describe()


def test_noise_does_not_move_the_reading():
    """A noise floor of a fiftieth of the peak: the pulse is found from
    its peak outward, so noise either side is not taken for its edges."""
    t, opens = record()
    nominal = pulse()
    noise = G * np.random.default_rng(3).standard_normal(t.size)
    check = judge_pulse(t, nominal.at(t - opens) + noise, nominal)
    assert check.passed
    assert check.measured.start == pytest.approx(opens, abs=2e-4)
    assert check.measured.duration == pytest.approx(T, rel=0.03)


# ---- the measurement and the pulse itself ------------------------------------


def test_a_half_sine_is_timed_from_its_tenth_of_peak_crossings():
    t, opens = record()
    measured = measure_pulse(t, pulse().at(t - opens - 0.0001), 'half-sine')
    assert measured.start == pytest.approx(opens + 0.0001, abs=1e-6)
    assert measured.duration == pytest.approx(T, rel=1e-5)
    assert measured.rise is None and measured.fall is None


def test_a_pulse_that_runs_off_the_record_is_refused():
    t = np.arange(1000) / RATE
    with pytest.raises(ValueError, match='not come back down'):
        measure_pulse(t, pulse().at(t - 0.005), 'half-sine')
    with pytest.raises(ValueError, match='never moves'):
        measure_pulse(t, -pulse().at(t - 0.001), 'half-sine')


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_a_sample_that_is_not_a_number_is_refused_for_what_it_is(bad):
    """One NaN far from the pulse: refused by name, not read as a record
    that never moves in the pulse's direction."""
    t, opens = record()
    nominal = pulse()
    values = nominal.at(t - opens)
    values[17] = bad
    message = ('1 of the record\'s samples are not finite numbers '
               r'\(NaN or infinite\), the first at sample 17')
    with pytest.raises(ValueError, match=message):
        measure_pulse(t, values, 'half-sine')
    with pytest.raises(ValueError, match=message):
        judge_pulse(t, values, nominal)
    with pytest.raises(ValueError, match=message):
        suggest_pulse(t, values)


def test_the_crossings_are_interpolated_between_samples():
    """A sawtooth at 1 kHz, eleven samples up its rise, starting 0.3 of
    a sample after one: its rise is straight, so the tenth and nine
    tenths crossings interpolated between the samples either side of
    each are exact, and so is the start they extend to. The duration
    is read to the last sample on the rise, 10.7 ms, since the record
    has no sample nearer the instant fall."""
    rate = 1000.0
    t = np.arange(160) / rate
    opens = 0.0603
    measured = measure_pulse(t, pulse('sawtooth', 40.0).at(t - opens),
                             'sawtooth')
    assert measured.start == pytest.approx(opens, abs=1e-9)
    assert measured.duration == pytest.approx(0.0107, abs=1e-9)
    assert measured.peak_time == pytest.approx(0.071)


def test_the_suggestion_is_the_half_sine_the_record_resembles():
    t, opens = record()
    made = pulse(peak=-42.0, duration=0.009)
    suggested = suggest_pulse(t, made.at(t - opens))
    assert suggested.shape == 'half-sine'
    assert suggested.peak == pytest.approx(-42.0 * G, rel=1e-3)
    assert suggested.duration == pytest.approx(0.009, rel=1e-4)


@pytest.mark.parametrize('made', [
    pulse('sawtooth', 35.0, duration=0.008, edge=0.0006),
    pulse('trapezoid', -25.0, duration=0.010, edge=0.0015)],
    ids=['sawtooth', 'trapezoid'])
def test_the_suggestion_reads_the_edge_of_a_sawtooth_and_a_trapezoid(made):
    """The sawtooth's fall and the trapezoid's edges are read off the
    record with the duration, so the suggestion is the pulse made."""
    t, opens = record()
    suggested = suggest_pulse(t, made.at(t - opens), made.shape)
    assert suggested.shape == made.shape
    assert suggested.peak == pytest.approx(made.peak, rel=1e-3)
    # a sawtooth's rise is read to its last sample, the nearest to its
    # crest, which falls between two
    duration = (np.floor((opens + made.duration) * RATE) / RATE - opens
                if made.shape == 'sawtooth' else made.duration)
    assert suggested.duration == pytest.approx(duration, abs=1e-9)
    assert suggested.edge == pytest.approx(made.edge, abs=1e-6)


@pytest.mark.parametrize('args,match', [
    (('square', 1.0, T), 'is none of them'),
    (('half-sine', 0.0, T), 'other than zero'),
    (('half-sine', 1.0, 0.0), 'above zero'),
    (('half-sine', 1.0, T, 0.001), 'no edge'),
    (('trapezoid', 1.0, T), 'needs its edge'),
    (('trapezoid', 1.0, T, 0.006), 'still have a top'),
    (('sawtooth', 1.0, T, -0.001), 'cannot be negative')])
def test_a_pulse_that_is_not_one_is_refused(args, match):
    with pytest.raises(ValueError, match=match):
        ClassicalPulse(*args)


def test_the_pulse_describes_itself():
    assert pulse().describe('g', 1.0 / G) == '50 g, 11 ms half-sine'
    assert (pulse('sawtooth', 40.0, edge=0.0011).describe('g', 1.0 / G)
            == '40 g, 11 ms, 1.1 ms fall sawtooth')
    assert (pulse('trapezoid', 30.0, edge=0.001).describe('g', 1.0 / G)
            == '30 g, 11 ms, 1 ms edges trapezoid')
    assert pulse('sawtooth', 40.0).extent == T
    assert pulse('sawtooth', 40.0, edge=0.001).extent == pytest.approx(0.012)


@pytest.mark.parametrize('name,value', [
    ('band', 0.0), ('duration_tolerance', 1.0), ('velocity_tolerance', -0.1),
    ('edge_limit', 0.5), ('quiet', -1.0)])
def test_a_tolerance_out_of_range_is_refused(name, value):
    with pytest.raises(ValueError):
        judged(pulse(), **{name: value})


# ---- a record ----------------------------------------------------------------


def history(*rows, dims=None):
    t, _opens = record()
    rows = np.vstack(rows)
    return TimeHistory(t, rows,
                       response_dof=[f'{k + 1}Z+' for k in range(len(rows))],
                       ordinate_dim=dims or ['acceleration'] * len(rows))


def test_a_record_is_checked_on_its_first_acceleration_channel():
    t, opens = record()
    nominal = pulse()
    shot = history(np.zeros(t.size), nominal.at(t - opens),
                   dims=['force', 'acceleration'])
    assert pulse_channel(shot) == 1
    assert check_pulse(shot, nominal).passed
    assert check_pulse(shot, nominal, '2Z+').passed
    with pytest.raises(ValueError, match='not an acceleration'):
        check_pulse(shot, nominal, 0)
    with pytest.raises(ValueError, match='no channel 9Z'):
        check_pulse(shot, nominal, '9Z+')
    with pytest.raises(ValueError, match='holds none'):
        check_pulse(history(np.zeros(t.size), dims=['force']), nominal)


# ---- the view ----------------------------------------------------------------


def _plots(layout):
    import pyqtgraph as pg

    return [item for item in layout.ci.items if isinstance(item, pg.PlotItem)]


def test_the_view_marks_the_points_outside_and_shades_the_window(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.classical_pulse import build_pulse_check
    from visualdynamics.units import SI

    check = judged(pulse(), pulse(peak=62.5))
    layout = pg.GraphicsLayoutWidget()
    build_pulse_check(layout, check, unit_system=SI, name='1Z+')
    record_plot, velocity_plot = _plots(layout)
    assert 'Flagged' in record_plot.titleLabel.text
    assert record_plot.titleLabel.text.startswith(
        '1Z+: 490.3 m/s², 11 ms half-sine<br>Flagged: ')
    marks = [item for item in record_plot.listDataItems()
             if getattr(item, 'is_outside_mark', False)]
    assert len(marks) == 1
    assert marks[0].opts['symbol'] == 'o'
    assert len(marks[0].xData) == check.outside_count
    window = next(item for item in record_plot.items
                  if isinstance(item, pg.LinearRegionItem))
    assert window.getRegion() == pytest.approx(check.window)
    band = next(item for item in velocity_plot.items
                if isinstance(item, pg.LinearRegionItem))
    nominal = check.nominal_velocity_change
    assert band.getRegion() == pytest.approx((0.8 * nominal, 1.2 * nominal))
    assert 'velocity change' in velocity_plot.titleLabel.text
    layout.close()


def test_the_view_marks_nothing_on_a_good_pulse(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.classical_pulse import build_pulse_check

    layout = pg.GraphicsLayoutWidget()
    build_pulse_check(layout, judged(pulse()))
    record_plot, _velocity = _plots(layout)
    assert record_plot.titleLabel.text.split('<br>')[1].startswith('Passes')
    assert not [item for item in record_plot.listDataItems()
                if getattr(item, 'is_outside_mark', False)]
    layout.close()


def test_the_readout_is_in_the_windows_units():
    from visualdynamics.plot.classical_pulse import readout
    from visualdynamics.units import IN_LBF_S

    check = judged(pulse('sawtooth', 40.0), pulse('sawtooth', 40.0,
                                                  edge=0.1 * T))
    rows = readout(check, IN_LBF_S.with_units(acceleration='g'))
    assert list(rows) == ['peak', 'duration', 'fall', 'velocity', 'outside']
    assert rows['velocity'] == '93.43 in/s, +10.0 %'
    assert rows['fall'] == '1.1 of 1.1 ms'
    assert rows['outside'] == f'0 of {check.points}'
    slow = judged(pulse('sawtooth', 40.0), pulse('sawtooth', 40.0,
                                                 edge=0.2 * T))
    assert readout(slow)['fall'].endswith(', out')


def test_the_view_draws_to_a_file(qt_app, tmp_path):
    from visualdynamics.plot import plot_pulse_check

    t, opens = record()
    path = tmp_path / 'pulse.png'
    plot_pulse_check(history(pulse().at(t - opens)), 'half-sine', 50 * G, T,
                     path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


# ---- the reading in the window -----------------------------------------------


def _reading(window, pump, values, name='Shot'):
    t, _opens = record()
    window.add_object(name, TimeHistory(
        t, np.atleast_2d(values), response_dof=['101Z+'],
        ordinate_dim=['acceleration']))
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    window.data_pane.pulse_action.trigger()
    pump()
    return window.data_pane.pulse_panel


def test_the_pulse_check_is_a_reading_of_a_time_history(window, pump):
    """Offered on a record's bar, and choosing it puts the panel up on
    the half-sine the record resembles, the check drawn and its verdict
    in the panel and on the status line."""
    t, opens = record()
    panel = _reading(window, pump, pulse(peak=48.0).at(t - opens))
    pane = window.data_pane
    assert pane.pulse_action.isVisible()
    assert pane.pulse_action.text() == 'Pulse Check'
    assert pane.pulse_wanted and panel.isVisible()
    settings = panel.settings()
    assert settings['shape'] == 'half-sine'
    assert settings['peak'] == pytest.approx(48.0 * G, rel=1e-3)
    assert settings['duration'] == pytest.approx(T)
    assert panel.verdict.text().startswith('Passes')
    assert panel.readout.text().splitlines()[0].startswith('Peak: 48 g')
    assert 'Passes' in window._status_text
    plots = _plots(pane.graphics)
    assert len(plots) == 2


def test_the_panels_pulse_redraws_the_check(window, pump):
    """A 40 g sawtooth read as the half-sine it was opened on fails;
    told it is a sawtooth, it passes; told the velocity tolerance is a
    twelfth, its full-allowance fall fails it. The edge row shows for the
    sawtooth and not for the half-sine."""
    t, opens = record()
    panel = _reading(window, pump,
                     pulse('sawtooth', 40.0, edge=0.1 * T).at(t - opens))
    assert not panel.edge_box.isVisibleTo(panel)
    assert panel.verdict.text().startswith('Flagged')
    settings = panel.settings()
    panel.set_settings(dict(settings, shape='sawtooth', peak=40.0 * G,
                            duration=T, edge=0.0))
    pump()
    assert panel.edge_box.isVisibleTo(panel)
    assert panel.names['edge'].text() == 'Fall [ms]'
    assert panel.verdict.text() == ('Passes: every point inside the '
                                    'envelope, velocity change +10.0 % of '
                                    'nominal.')
    assert 'Fall: 1.1 of 1.1 ms' in panel.readout.text()
    panel.velocity_box.setValue(8.0)
    pump()
    assert panel.verdict.text() == 'Flagged: velocity change +10.0 % of nominal.'
    assert window.data_pane.pulse_settings['velocity_tolerance'] == 0.08


def test_a_record_with_no_acceleration_says_so(window, pump):
    t, opens = record()
    window.add_object('Force', TimeHistory(
        t, np.atleast_2d(pulse().at(t - opens)), response_dof=['101Z+'],
        ordinate_dim=['force']))
    window.show_object('Force')
    pump()
    window.data_pane.pulse_action.trigger()
    pump()
    panel = window.data_pane.pulse_panel
    assert 'holds none' in panel.verdict.text()
    assert panel.readout.text() == ''


def test_the_pulse_check_is_journaled_and_replays(window, pump, tmp_path,
                                                  monkeypatch):
    """Journaled as the headless call that draws it, settling in place,
    and the line run on its own draws the figure."""
    import visualdynamics
    import visualdynamics.plot

    t, opens = record()
    panel = _reading(window, pump, pulse().at(t - opens))
    panel.velocity_box.setValue(15.0)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_pulse_check(')]
    assert len(lines) == 1, 'the view settles to the reading that stands'
    assert "project['Shot'], 'half-sine'" in lines[0]
    assert "channel='101Z+'" in lines[0]
    assert 'velocity_tolerance=0.15' in lines[0]
    monkeypatch.chdir(tmp_path)
    room = {'visualdynamics': visualdynamics,
            'project': {'Shot': window.objects['Shot']}}
    exec(lines[0], room)                                 # noqa: S102
    assert (tmp_path / 'pulse_check.png').stat().st_size > 2000
