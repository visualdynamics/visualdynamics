"""Q read from the half-power width of a peak.

Held to the one-mass resonator, where the answer is known: an
accelerance and a transmissibility at a Q of 20 and of 50, sampled
finely, read their Q within one percent and their natural frequency
within a line. Sampled coarsely the same peaks read low, and say so.
A tracked sine level from a sweep through the same resonator reads its
Q at a slow sweep rate, and reads it low at a fast one, which is the
sweep-rate effect and not the reading. The readout follows the fitting
screen's cursor, drawn and in the status line.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from visualdynamics.core.data import Frf, TimeHistory
from visualdynamics.core.half_power import (
    MIN_LINES,
    HalfPowerWarning,
    read_half_power,
)
from visualdynamics.plot.half_power import label_text

FN = 180.0


def accelerance(f, fn, q):
    """|a/F| of one mass on a spring with damping ratio 1/(2q)."""
    zeta, w, wn = 0.5 / q, 2 * np.pi * f, 2 * np.pi * fn
    return -w ** 2 / (wn ** 2 - w ** 2 + 2j * zeta * wn * w)


def transmissibility(f, fn, q):
    """|response/base| of one mass on its spring riding a moving base."""
    zeta, r = 0.5 / q, f / fn
    return np.sqrt((1 + (2 * zeta * r) ** 2)
                   / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2))


def lines_across(fn, q, lines, offset=0.0):
    """A frequency axis with `lines` lines across the half-power width
    fn/q, shifted by `offset` of a line against the peak."""
    df = fn / q / lines
    return (np.arange(1, int(2 * fn / df)) + offset) * df


def one_mass_frf(q, lines=40, offset=0.37):
    f = lines_across(FN, q, lines, offset)
    return Frf(f, accelerance(f, FN, q)[None], response_dof=['1Z+'],
               reference_dof=['1Z+'], ordinate_dim='acceleration/force')


# ---- a finely sampled peak ----------------------------------------------


@pytest.mark.parametrize('q', [20.0, 50.0])
def test_a_finely_sampled_one_mass_peak_reads_its_q(q):
    frf = one_mass_frf(q)
    with warnings.catch_warnings():
        # a reading that should be clean must not warn
        warnings.simplefilter('error', HalfPowerWarning)
        reading = frf.half_power(FN * 1.004)
    spacing = frf.abscissa[1] - frf.abscissa[0]
    assert reading.q == pytest.approx(q, rel=0.01)
    assert abs(reading.frequency - FN) < spacing, 'within a line'
    assert reading.damping == pytest.approx(0.5 / q, rel=0.01)
    assert reading.width == pytest.approx(FN / q, rel=0.01)
    assert reading.lines == pytest.approx(40.0, rel=0.02)
    assert reading.resolved and reading.notes == ()
    # the top is placed between lines, not left on the tallest one
    assert reading.height == pytest.approx(abs(accelerance(FN, FN, q)),
                                           rel=1e-3)
    assert reading.level == pytest.approx(reading.height / np.sqrt(2))


@pytest.mark.parametrize('q', [20.0, 50.0])
def test_a_transmissibility_reads_its_q_too(q):
    f = lines_across(FN, q, 40, 0.61)
    reading = read_half_power(f, transmissibility(f, FN, q), FN)
    assert reading.q == pytest.approx(q, rel=0.01)
    assert reading.height == pytest.approx(q, rel=0.01), (
        'a transmissibility peak is Q high')


def test_a_complex_curve_is_read_by_its_magnitude():
    f = lines_across(FN, 20.0, 40)
    complex_curve = accelerance(f, FN, 20.0)
    assert read_half_power(f, complex_curve, FN) == read_half_power(
        f, np.abs(complex_curve), FN)


# ---- resolution ---------------------------------------------------------


def test_a_coarsely_sampled_peak_warns_and_reads_low():
    """One and a half lines across the width: the tallest line misses
    the top and the crossings are interpolated across straight segments
    of a curved flank, so the width reads wide and Q low."""
    f = lines_across(FN, 20.0, 1.5, 0.5)
    with pytest.warns(HalfPowerWarning, match=r'lines of .*Q is biased'):
        reading = read_half_power(f, accelerance(f, FN, 20.0), FN)
    assert reading.q < 0.95 * 20.0, 'biased low, as the warning says'
    assert reading.q > 0.5 * 20.0, 'but still a reading of this peak'
    assert reading.lines < MIN_LINES and not reading.resolved
    assert len(reading.notes) == 1 and 'lines' in reading.notes[0]


def test_the_warning_threshold_is_the_one_stated():
    """Four lines is where the reading stops warning: just under it
    warns, just over it does not."""
    assert MIN_LINES == 4
    for lines, warned in ((3.8, True), (4.2, False)):
        f = lines_across(FN, 20.0, lines, 0.25)
        reading = read_half_power(f, accelerance(f, FN, 20.0), FN,
                                  warn=False)
        assert (not reading.resolved) is warned, lines
        assert bool(reading.notes) is warned, lines


@pytest.mark.parametrize('lines,worst', [(4, -0.031), (2, -0.14)])
def test_the_bias_at_few_lines_is_what_the_threshold_says(lines, worst):
    """The measurement behind MIN_LINES: over twenty placements of the
    lines against the peak, Q at four lines is at most 3 % low and at
    two lines up to 14 % low, and never high."""
    errors = []
    for q in (20.0, 50.0):
        for offset in np.linspace(0.0, 1.0, 21)[:-1]:
            f = lines_across(FN, q, lines, offset)
            reading = read_half_power(f, accelerance(f, FN, q), FN,
                                      warn=False)
            errors.append(reading.q / q - 1.0)
    assert worst < min(errors) < worst / 2, min(errors)
    assert max(errors) < 0.0, 'the bias is low, never high'


# ---- which peak, and what is refused --------------------------------------


def test_a_frequency_on_a_flank_reads_the_peak_it_is_on():
    """Two modes; a frequency on the smaller one's flank reads the
    smaller one, not the tallest in the curve."""
    f = np.linspace(50.0, 400.0, 7001)
    curve = np.abs(accelerance(f, 100.0, 20.0)) \
        + 0.3 * np.abs(accelerance(f, 300.0, 50.0))
    reading = read_half_power(f, curve, 296.0)
    assert reading.frequency == pytest.approx(300.0, abs=0.1)
    # the first mode's tail lifts the flanks and widens the reading a
    # few percent: a curve, read faithfully, with a neighbor under it
    assert reading.q == pytest.approx(50.0, rel=0.08)
    tallest = read_half_power(f, curve, 296.0, within=(50.0, 400.0))
    assert tallest.frequency == pytest.approx(100.0, abs=0.1), (
        'within= reads the tallest line in the band instead')


def test_a_crossing_outside_the_curve_is_refused_not_guessed():
    f = np.linspace(FN - 2.0, FN + 40.0, 2000)
    with pytest.warns(HalfPowerWarning, match='ends at 178 Hz before '
                                              'falling to half power below'):
        reading = read_half_power(f, accelerance(f, FN, 20.0), FN)
    assert reading.low is None and reading.high is not None
    assert reading.width is None and reading.q is None
    assert reading.damping is None and not reading.resolved
    assert 'no half-power crossing below' in reading.describe()


def test_a_curve_that_stops_on_the_rise_reads_its_last_line():
    """Climbing from the middle of a curve that is still rising where it
    stops: the top is the last line, which has no neighbor above to place
    it between, and there is no crossing above it."""
    f = np.linspace(100.0, FN, 400)
    with pytest.warns(HalfPowerWarning, match='ends at 180 Hz before '
                                              'falling to half power above'):
        reading = read_half_power(f, accelerance(f, FN, 20.0), 150.0)
    assert reading.frequency == f[-1]
    assert reading.height == pytest.approx(abs(accelerance(FN, FN, 20.0)))
    assert reading.spacing == pytest.approx(f[1] - f[0])
    assert reading.low is not None and reading.high is None
    # and the same from the other end, falling all the way
    # an accelerance peaks a little above fn, at 180.1 Hz: start past it
    g = np.linspace(181.0, 260.0, 400)
    with pytest.warns(HalfPowerWarning, match='ends at 181 Hz'):
        first = read_half_power(g, accelerance(g, FN, 20.0), 230.0)
    assert first.frequency == g[0] and first.low is None
    assert first.high is not None


def test_a_frequency_above_the_peak_climbs_down_to_it():
    f = lines_across(FN, 20.0, 40, 0.37)
    curve = accelerance(f, FN, 20.0)
    assert read_half_power(f, curve, 186.0) == read_half_power(f, curve,
                                                                174.0)


def test_a_peak_narrower_than_its_lines_is_refused():
    """Lines a whole half-power width apart many times over: the top,
    placed between them, stands more than 3 dB above every line, so no
    crossing can be placed, and a width of zero is not reported as a Q
    of infinity."""
    f = FN + (np.arange(-1, 6) + 0.05) * 180.0
    with pytest.warns(HalfPowerWarning, match='narrower than its lines'):
        reading = read_half_power(f, accelerance(f, FN, 20.0), FN)
    assert reading.low is None and reading.high is None
    assert reading.q is None and len(reading.notes) == 1
    assert reading.describe().endswith('no half-power width can be read')


def test_a_line_at_zero_beside_the_top_leaves_it_on_its_line():
    """The top is placed through the logarithms of three lines, and a
    zero has none: the top stays where the line is."""
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reading = read_half_power(f, [0.0, 2.0, 1.0, 0.2, 0.1], 2.0, warn=False)
    assert (reading.frequency, reading.height) == (2.0, 2.0)
    assert reading.low == pytest.approx(2.0 - (2.0 - np.sqrt(2)) / 2.0)
    above = read_half_power(f, [0.1, 1.0, 2.0, 0.0, 0.1], 3.0, warn=False)
    assert (above.frequency, above.height) == (3.0, 2.0)


def test_an_uneven_axis_is_read_at_its_own_spacing():
    """A tracked level is read evenly in time, so on a log sweep its
    lines are evenly spaced in log frequency: the spacing at the peak is
    the mean of the two steps about it."""
    f = np.geomspace(100.0, 300.0, 2000)
    reading = read_half_power(f, accelerance(f, FN, 20.0), FN)
    i = int(np.argmax(np.abs(accelerance(f, FN, 20.0))))
    assert reading.spacing == pytest.approx((f[i + 1] - f[i - 1]) / 2.0)
    assert reading.q == pytest.approx(20.0, rel=0.01)


def test_a_band_holding_one_line_reads_the_peak_on_it():
    """The band's ends are inclusive: a band from the tallest line to
    itself holds that line."""
    f = lines_across(FN, 20.0, 40, 0.37)
    curve = accelerance(f, FN, 20.0)
    k = int(np.argmax(np.abs(curve)))
    reading = read_half_power(f, curve, 0.0, within=(f[k], f[k]))
    assert reading == read_half_power(f, curve, FN)


def test_the_warning_points_at_the_line_that_read():
    f = lines_across(FN, 20.0, 1.5, 0.5)
    with pytest.warns(HalfPowerWarning) as caught:
        read_half_power(f, accelerance(f, FN, 20.0), FN)
    assert caught[0].filename == __file__


def test_crossings_are_measured_down_from_a_line_not_up_to_one():
    from visualdynamics.core.half_power import crossings

    with pytest.raises(ValueError, match='above the line'):
        crossings([1.0, 2.0, 3.0], [1.0, 2.0, 1.0], 1, 2.5)
    assert crossings([1.0, 2.0, 3.0], [1.0, 2.0, 1.0], 1, 1.5) == (1.5, 2.5)


def test_a_reading_names_itself():
    f = lines_across(FN, 20.0, 40, 0.37)
    text = read_half_power(f, accelerance(f, FN, 20.0), FN).describe()
    # an accelerance peaks a hair above its natural frequency, at
    # fn / sqrt(1 - 2 zeta^2): 180.1 Hz here
    assert text == ('Q 19.9 at 180.1 Hz, half-power 175.8 to 184.8 Hz '
                    '(9.03 Hz, 40 lines)')


def test_a_reading_names_itself_without_exponents():
    """A peak at 10 kHz with a Q of 1500, read across 1500 lines: the
    numbers in four figures and three, the lines whole, and none of
    them in exponent form."""
    fn, q = 10240.0, 1500.0
    f = lines_across(fn, q, 1500, 0.37)
    f = f[(f > 0.98 * fn) & (f < 1.02 * fn)]
    reading = read_half_power(f, accelerance(f, fn, q), fn)
    text = reading.describe()
    assert 'e+' not in text and 'e-' not in text, text
    assert text == ('Q 1500 at 10240 Hz, half-power 10237 to 10243 Hz '
                    '(6.83 Hz, 1500 lines)')
    assert label_text(reading) == 'Q 1500'


def test_the_lines_a_reading_is_trusted_at_are_its_own():
    """A reading taken with a lower `min_lines` calls itself resolved by
    that count, not the module's, so it does not say it is unresolved
    after warning nothing."""
    f = lines_across(FN, 20.0, 3.0, 0.25)
    curve = accelerance(f, FN, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', HalfPowerWarning)
        reading = read_half_power(f, curve, FN, min_lines=2)
    assert 2 < reading.lines < MIN_LINES
    assert reading.resolved and reading.notes == ()
    assert reading.min_lines == 2
    default = read_half_power(f, curve, FN, warn=False)
    assert not default.resolved and len(default.notes) == 1


def _climb_a_line_at_a_time(m, index):
    """The climb as it was first written, a line at a time: the
    reference the array version is held to."""
    last = len(m) - 1
    while True:
        below = m[index - 1] if index > 0 else -np.inf
        above = m[index + 1] if index < last else -np.inf
        if below > m[index] and below >= above:
            index -= 1
        elif above > m[index]:
            index += 1
        else:
            return index


def _crossings_a_line_at_a_time(f, m, index, level):
    def side(step):
        i = index
        while 0 <= i + step < len(m):
            if m[i + step] < level:
                fraction = (m[i] - level) / (m[i] - m[i + step])
                return float(f[i] + fraction * (f[i + step] - f[i]))
            i += step
        return None

    return side(-1), side(+1)


def test_the_climb_and_the_crossings_are_the_rule_they_were():
    """The climb and the crossings run over the array at once, for a
    drag that reads on every tick; on rough curves, plateaus and peaks
    at the ends they land where the line-at-a-time rule landed."""
    from visualdynamics.core.half_power import _climb, crossings

    rng = np.random.default_rng(7)
    f = np.arange(60, dtype=float)
    curves = [rng.random(60), np.round(rng.random(60), 1),
              np.r_[np.arange(30.0), np.arange(30.0)[::-1]],
              np.arange(60.0), np.arange(60.0)[::-1], np.ones(60)]
    for m in curves:
        for start in range(60):
            top = _climb(m, start)
            assert top == _climb_a_line_at_a_time(m, start), (m, start)
            for share in (0.3, 0.7, 1.0):
                level = share * m[top]
                assert crossings(f, m, top, level) == \
                    _crossings_a_line_at_a_time(f, m, top, level)


@pytest.mark.parametrize('f,m,within,match', [
    ([1.0, 2.0], [1.0, 2.0], None, 'three lines'),
    ([1.0, 2.0, 3.0], [1.0, 2.0], None, '3 frequencies for 2'),
    ([3.0, 2.0, 1.0], [1.0, 2.0, 1.0], None, 'ascending'),
    ([1.0, 2.0, 2.0], [1.0, 2.0, 1.0], None, 'ascending'),
    ([1.0, 2.0, 3.0], [1.0, 2.0, 1.0], (5.0, 6.0), 'no line'),
])
def test_a_curve_that_cannot_be_read_is_refused(f, m, within, match):
    with pytest.raises(ValueError, match=match):
        read_half_power(f, m, 2.0, within=within)


# ---- a tracked sine level --------------------------------------------------


def _tracked(q, octaves_per_minute, fn=60.0, rate=2048.0):
    """A log sweep from 30 to 120 Hz through a one-mass resonator on a
    moving base, read by a 10 % proportional tracking filter: the
    tracked level of the response is the swept transmissibility."""
    from scipy import signal

    from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone
    from visualdynamics.core.sine_tracking import SineTracking, track_sine

    onset = 0.25
    tone = SineTone('Sine Tone 1', onset, [30.0, 120.0], [[1.0], [1.0]],
                    [LOG], [octaves_per_minute])
    t = np.arange(int((onset + tone.duration() + 0.25) * rate)) / rate
    local = np.clip(t - onset, 0.0, None)
    inside = (t >= onset) & (local <= tone.duration())
    drive = np.where(inside, np.sin(tone.phase_at(local)), 0.0)
    wn, zeta = 2 * np.pi * fn, 0.5 / q
    # first-order hold: exact for a drive straight between samples
    b, a, _dt = signal.cont2discrete(
        ([2 * zeta * wn, wn ** 2], [1, 2 * zeta * wn, wn ** 2]), 1 / rate,
        method='foh')
    response = signal.lfilter(np.ravel(b), a, drive)
    history = TimeHistory(t, np.stack([drive, response]),
                          response_dof=['1Z+', '2Z+'],
                          ordinate_dim=['acceleration'] * 2)
    spec = SineSweepSpecification([tone], ['1Z+'], ordinate_unit='m/s**2')
    return track_sine(history, spec, SineTracking(proportional=0.1),
                      onset=onset, channels=['2Z+'], lines=4000)[0]


def test_a_slow_sweeps_tracked_level_reads_the_resonators_q():
    """At 1 oct/min a 60 Hz mode with a Q of 20 has about thirteen times
    the cycles it needs to build inside its half-power band (87 fn / Q^2
    is 13 oct/min), and its level reads the Q within one percent."""
    level = _tracked(20.0, 1.0)
    reading = level.half_power(60.0)
    assert reading.q == pytest.approx(20.0, rel=0.01)
    assert reading.frequency == pytest.approx(60.0, rel=0.005)
    assert reading.resolved


def test_a_fast_sweeps_tracked_level_reads_its_q_low():
    """The same mode swept at 20 oct/min, past the rate it can build
    at: the peak is late and low and the width wide. The reading is
    faithful to the curve, so it is the curve that reads low."""
    reading = _tracked(20.0, 20.0).half_power(60.0)
    assert reading.q < 0.8 * 20.0
    assert reading.frequency > 61.0, 'the peak is carried up the sweep'


# ---- the modal fit's readout ------------------------------------------------


def test_the_fit_session_reads_the_peak_under_its_cursor():
    from visualdynamics.core.modal_fit import ModalFitSession
    from visualdynamics.units import IN_LBF_S

    session = ModalFitSession(one_mass_frf(20.0))
    session.move_to(FN + 1.0, damping=False)
    reading = session.half_power()
    assert reading.frequency == pytest.approx(FN, abs=0.25)
    assert reading.q == pytest.approx(20.0, rel=0.01)
    # the height in the units the CMIF is drawn in, the Q unmoved
    shown = session.half_power(unit_system=IN_LBF_S)
    expected = IN_LBF_S.from_si(reading.height, 'acceleration/force')
    assert shown.height == pytest.approx(expected, rel=1e-9)
    assert shown.height != pytest.approx(reading.height, rel=1e-3)
    assert shown.q == reading.q
    assert session.half_power(FN * 0.9).frequency == reading.frequency, (
        'a frequency on the flank reads the same peak')


def test_the_fit_seed_finds_its_crossings_where_the_readout_does():
    """One rule for where the half-power crossings are: the seed reads
    them at the cursor's line, the readout at the top between lines, and
    on a finely sampled peak the two widths agree."""
    from visualdynamics.core.modal_fit import ModalFitSession

    session = ModalFitSession(one_mass_frf(20.0, lines=200))
    reading = session.half_power(FN)
    assert session.estimate_damping(FN) == pytest.approx(reading.damping,
                                                         rel=0.005)
    # and both are the resonator's: agreeing on a wrong width would
    # pass the line above
    assert session.estimate_damping(FN) == pytest.approx(0.025, rel=0.01)


def test_the_fitting_screen_draws_the_reading_and_says_it(window, pump):
    window.add_object('FRF', one_mass_frf(20.0))
    window.tree.setCurrentItem(window._item_for_object('FRF'))
    pump()
    window.start_modal_fit()
    pump()
    overlay = window._fit_half_power
    assert overlay is not None and overlay.label.isVisible()
    assert overlay.label.toPlainText() == 'Q 19.9'
    low, high = overlay.crossings.getData()[0]
    assert (low, high) == pytest.approx((175.77, 184.80), abs=0.01)
    # the CMIF's ordinate is logarithmic, and the plot converts its data
    # items itself: what is drawn is the logarithm of the level
    level = overlay.level.getData()[1]
    assert 10 ** level[0] == pytest.approx(window._fit_reading.level,
                                           rel=1e-12)
    # a text item is placed in view coordinates, the logarithm on this
    # axis; placed at the level itself it would sit decades off screen
    assert overlay.label.pos().y() == pytest.approx(
        np.log10(window._fit_reading.level), rel=1e-9)
    assert overlay.label.pos().x() == pytest.approx(low, rel=1e-9)
    assert 'half-power Q 19.9 at 180.1 Hz' in window._status_text
    # a curve the reading refuses leaves no readout, rather than an
    # exception in the middle of a drag
    def refuse(**_kwargs):
        raise ValueError('the frequencies must be ascending')

    # a drag tick reads once, and a vertical drag (the damping alone)
    # not at all: the reading is of the residual at the cursor's line,
    # which neither the damping search nor a held damping moves
    reads = []
    session_read = window.fit.half_power

    def counted(**kwargs):
        reads.append(kwargs)
        return session_read(**kwargs)

    window.fit.half_power = counted
    window._fit_dear_cost = 0.0
    window._fit_cursor.setValue(FN + 2.0)
    pump()
    assert len(reads) == 1
    window._fit_cursor_dragged(
        window._fit_plot.getViewBox().sceneBoundingRect().center().y())
    assert len(reads) == 1
    window.fit.half_power = refuse
    window._move_fit_parabola()
    window._fit_status()
    assert not overlay.label.isVisible() and overlay.reading is None
    assert 'half-power' not in window._status_text
    window.stop_fitting()
    assert window._fit_half_power is None and window._fit_reading is None


# ---- drawn without a window --------------------------------------------------


def test_the_reading_draws_on_its_record_to_a_file(tmp_path):
    from visualdynamics.plot import plot_half_power

    path = tmp_path / 'half_power.png'
    assert plot_half_power(one_mass_frf(20.0), FN, path=path,
                           show=False) == str(path)
    assert path.exists() and path.stat().st_size > 2000


def test_a_peak_read_too_coarsely_says_so_on_the_plot():
    from visualdynamics.plot.half_power import label_text

    f = lines_across(FN, 20.0, 1.5, 0.5)
    reading = read_half_power(f, accelerance(f, FN, 20.0), FN, warn=False)
    assert label_text(reading) == (f'Q {reading.q:.3g}\n'
                                   f'{reading.lines:.2g} lines: too few')
    g = np.linspace(FN - 2.0, FN + 40.0, 2000)
    edge = read_half_power(g, accelerance(g, FN, 20.0), FN, warn=False)
    assert label_text(edge) == 'no half-power width'


def test_a_time_history_has_no_half_power_width():
    from visualdynamics.plot import plot_half_power

    t = np.linspace(0.0, 1.0, 100)
    history = TimeHistory(t, np.sin(t)[None], response_dof=['1Z+'])
    with pytest.raises(TypeError, match='against frequency'):
        plot_half_power(history, 0.5, show=False, path='never.png')


def test_a_power_quantity_is_refused_not_read_at_the_wrong_level(
        tmp_path):
    """A PSD, an SRS and a coherence are against frequency too, but none
    is a response's magnitude. A PSD is a power: its half-power level is
    half its peak, and read at 0.707 of it the Q 20 resonance reads
    Q 31."""
    from visualdynamics.core.data import Coherence, Psd, Srs
    from visualdynamics.plot import plot_half_power

    f = lines_across(FN, 20.0, 40, 0.37)
    power = np.abs(accelerance(f, FN, 20.0)) ** 2
    assert read_half_power(f, power, FN).q == pytest.approx(31.1, abs=0.1)
    for refused in (
            Psd(f, power[None], response_dof=['1Z+'],
                ordinate_dim='acceleration'),
            Srs(f, power[None], response_dof=['1Z+'],
                ordinate_dim='acceleration'),
            Coherence(f, np.clip(power / power.max(), 0, 1)[None],
                      response_dof=['1Z+'], reference_dof=['2Z+'])):
        with pytest.raises(TypeError, match=type(refused).__name__):
            plot_half_power(refused, FN, show=False,
                            path=tmp_path / 'never.png')


@pytest.mark.parametrize('how', ['frf', 'spectrum', 'session', 'plot'])
def test_every_way_in_warns_at_the_callers_line(how, tmp_path):
    """The warning names the line that asked for the reading, through
    each method that reads on the caller's behalf."""
    from visualdynamics.core.data import Spectrum
    from visualdynamics.core.modal_fit import ModalFitSession
    from visualdynamics.plot import plot_half_power

    frf = one_mass_frf(20.0, lines=1.5, offset=0.5)
    spectrum = Spectrum(frf.abscissa, np.abs(frf.ordinate),
                        response_dof=['1Z+'], ordinate_dim='acceleration')
    read = {
        'frf': lambda: frf.half_power(FN),
        'spectrum': lambda: spectrum.half_power(FN),
        'session': lambda: ModalFitSession(frf).half_power(FN),
        'plot': lambda: plot_half_power(frf, FN, show=False,
                                        path=tmp_path / 'q.png'),
    }[how]
    with pytest.warns(HalfPowerWarning) as caught:
        read()
    assert {w.filename for w in caught} == {__file__}


def test_the_drawn_reading_sits_on_the_curve_in_display_units():
    """The record is drawn in the display system and the reading made
    in SI: the top and the level must be converted by the same factor
    as the curve, or on in-lbf-s data they sit decades below it."""
    import pyqtgraph as pg

    from visualdynamics.plot import plot_half_power
    from visualdynamics.units import IN_LBF_S

    frf = one_mass_frf(20.0)
    reading = frf.half_power(FN)
    pane = plot_half_power(frf, FN, unit_system=IN_LBF_S, show=False)
    try:
        plot = next(item for item in pane.graphics.ci.items
                    if isinstance(item, pg.PlotItem))
        assert plot.listDataItems() and all(
            item.opts.get('symbol') is None for item in plot.listDataItems()
        ), 'the overlay is a reading of the data, not more data'
        top = next(item for item in plot.getViewBox().childGroup.childItems()
                   if isinstance(item, pg.PlotDataItem)
                   and item.opts.get('symbol') == 'd')
        shown = IN_LBF_S.from_si(reading.height, 'acceleration/force')
        assert shown != pytest.approx(reading.height, rel=1e-3)
        # the ordinate is logarithmic: what is drawn is the logarithm
        assert 10 ** top.getData()[1][0] == pytest.approx(shown, rel=1e-9)
    finally:
        pane.close()
        pane.deleteLater()


def test_a_line_with_no_value_on_a_flank_is_no_crossing():
    """Held as -inf, a NaN line made a crossing of its own at the
    missing line, and a Q was read off it (2026-10-28); dropped, the
    curve runs between its neighbors."""
    fn, q = 60.0, 20.0
    f = lines_across(fn, q, 12)
    m = accelerance(f, fn, q)
    whole = read_half_power(f, m, fn, warn=False)
    holed = np.abs(m).copy()
    k = int(np.argmax(holed)) + 3                # on the upper flank
    holed[k] = np.nan
    gapped = read_half_power(f, holed, fn, warn=False)
    assert gapped.q == pytest.approx(whole.q, rel=0.05)
    assert gapped.high != pytest.approx(f[k], abs=1e-9)
    # a NaN at the half-power line itself places no crossing there
    at_half = int(np.argmin(np.abs(np.abs(m) - whole.height / np.sqrt(2))))
    holed = np.abs(m).copy()
    holed[at_half] = np.nan
    assert read_half_power(f, holed, fn, warn=False).q == pytest.approx(
        whole.q, rel=0.05)
    with pytest.raises(ValueError, match='three lines'):
        read_half_power(f[:5], [1.0, np.nan, np.nan, np.nan, 1.0], f[2],
                        warn=False)
