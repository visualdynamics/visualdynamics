"""A sine sweep's rate checked against the modes it passes.

A mode needs about Q cycles to build, and a sweep gives it only the
time it spends in the mode's half-power band. These pin the arithmetic
of that time (86.6 / (Q R) seconds on a log sweep, fn / (Q r) on a
linear one), the rule it gives (87 fn / Q**2 oct/min, fn**2 / Q**2
Hz/s), and the simulated reading against hand values for one mode: at
22 Hz with a Q of 40, swept up at 1, 4 and 8 oct/min, the peak reads 95,
81 and 71 percent of its steady height and the Q read from its width
33.0, 21.6 and 16.6. Then the check against a specification, the Q read
off a tracked level judged against its own sweep, the plot, and the
reading in the window.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.sine import (
    LINEAR,
    LOG,
    SineSweepSpecification,
    SineTone,
)
from visualdynamics.core.sweep_rate import (
    RULE_CONSTANT,
    SAMPLES_PER_CYCLE,
    STEADY_RATIO,
    check_sweep_rate,
    half_power_band,
    read_swept_q,
    rule_rate,
    simulate_sweep,
    swept_q,
)

G = 9.80665


def tone(frequencies, rates, kinds=None, name='Sweep'):
    """A 1 g tone over these breakpoints at these rates, log unless
    told otherwise."""
    kinds = [LOG] * len(rates) if kinds is None else kinds
    return SineTone(name, 0.0, frequencies, [[G]] * len(frequencies),
                    kinds, rates)


def specification(*tones):
    return SineSweepSpecification(list(tones), ['1Z+'], ordinate_unit='m/s^2')


# ---- the arithmetic ---------------------------------------------------------


def test_the_rule_is_sixty_over_ln_2_times_fn_over_q_squared():
    assert RULE_CONSTANT == pytest.approx(86.56, abs=0.01)
    # the section's three examples: about 4, 1 and 44 oct/min
    assert rule_rate(20.0, 20.0) == pytest.approx(4.328, abs=0.001)
    assert rule_rate(10.0, 30.0) == pytest.approx(0.962, abs=0.001)
    assert rule_rate(200.0, 20.0) == pytest.approx(43.28, abs=0.01)
    assert rule_rate(22.0, 40.0) == pytest.approx(1.190, abs=0.001)


def test_a_linear_sweeps_rule_is_fn_over_q_squared_in_hz_per_second():
    assert rule_rate(20.0, 20.0, LINEAR) == pytest.approx(1.0)
    assert rule_rate(400.0, 20.0, LINEAR) == pytest.approx(400.0)


def test_the_band_is_fn_over_q_wide_and_centered():
    low, high = half_power_band(22.0, 40.0)
    assert (low, high) == pytest.approx((21.725, 22.275))
    assert high - low == pytest.approx(22.0 / 40.0)


@pytest.mark.parametrize('q, rate, seconds', [(40.0, 1.0, 2.16),
                                              (40.0, 4.0, 0.54),
                                              (40.0, 8.0, 0.27),
                                              (20.0, 4.0, 1.08)])
def test_a_log_sweep_spends_86_6_over_q_r_seconds_in_the_band(q, rate,
                                                              seconds):
    check, = check_sweep_rate(tone([10.0, 40.0], [rate]), [(22.0, q)])
    assert check.seconds == pytest.approx(86.6 / (q * rate), rel=0.002)
    assert round(check.seconds, 2) == seconds
    # and the cycles are the mode's frequency times that, near enough
    # (the drive's frequency runs across the band while it is in it)
    assert check.cycles == pytest.approx(22.0 * check.seconds, rel=1e-3)


def test_a_log_sweep_gives_every_mode_of_one_q_the_same_time():
    """The same time in every band of a given Q, whatever its frequency;
    so a 400 Hz mode gets twenty times the cycles of a 20 Hz one."""
    low, high = check_sweep_rate(tone([5.0, 2000.0], [4.0]),
                                 [(20.0, 20.0), (400.0, 20.0)])
    assert low.seconds == pytest.approx(high.seconds, rel=1e-9)
    assert high.cycles / low.cycles == pytest.approx(20.0, rel=1e-4)


def test_a_linear_sweep_lingers_on_the_high_modes():
    """1 Hz/s crosses a 20 Hz, Q 20 band in a second and a 400 Hz, Q 20
    band in twenty."""
    low, high = check_sweep_rate(tone([5.0, 2000.0], [1.0], [LINEAR]),
                                 [(20.0, 20.0), (400.0, 20.0)])
    assert low.seconds == pytest.approx(1.0)
    assert high.seconds == pytest.approx(20.0)
    assert low.start == pytest.approx(19.5 - 5.0)
    assert low.unit == 'Hz/s' and low.ratio == pytest.approx(1.0)


# ---- the simulated reading ---------------------------------------------------


def test_the_resonator_is_sampled_finely_enough_to_read_its_steady_height():
    """At `SAMPLES_PER_CYCLE` the filter's steady response at fn, read
    as the reading reads it (a cycle's rms as a sine's peak), is within
    0.1 % of Q sqrt(1 + 1/Q**2)."""
    from scipy.signal import lfilter

    from visualdynamics.core.srs import damping_for, ramp_invariant

    q = 40.0
    n = np.arange(400 * SAMPLES_PER_CYCLE)
    b, a = ramp_invariant([1.0], SAMPLES_PER_CYCLE, damping_for(q))
    y = lfilter(b[0], a[0], np.sin(2 * np.pi * n / SAMPLES_PER_CYCLE))
    steady = np.sqrt(2.0 * np.mean(y[-SAMPLES_PER_CYCLE:] ** 2))
    assert steady / (q * np.sqrt(1 + 1 / q ** 2)) == pytest.approx(1.0,
                                                                abs=1e-3)


@pytest.mark.parametrize('rate, percent, frequency, q_read',
                         [(1.0, 95, 22.22, 33.0),
                          (4.0, 81, 22.61, 21.6),
                          (8.0, 71, 22.97, 16.6)])
def test_the_22_hz_q_40_mode_reads_low_late_and_wide(rate, percent,
                                                     frequency, q_read):
    reading = simulate_sweep(22.0, 40.0, rate)
    assert reading.simulated
    assert round(100 * reading.peak_share) == percent
    assert reading.peak_frequency == pytest.approx(frequency, abs=0.02)
    assert reading.q_read == pytest.approx(q_read, abs=0.1)
    assert round(reading.q_read) == {1.0: 33, 4.0: 22, 8.0: 17}[rate]


def test_a_quarter_octave_per_minute_reads_the_mode_nearly_whole():
    reading = simulate_sweep(22.0, 40.0, 0.25)
    assert round(100 * reading.peak_share) == 99
    assert reading.peak_frequency == pytest.approx(22.06, abs=0.02)
    assert reading.q_read == pytest.approx(39.2, abs=0.1)


def test_a_sweep_down_reads_the_peak_below_the_mode():
    reading = simulate_sweep(22.0, 40.0, -4.0)
    assert reading.peak_frequency < 22.0 - 0.5
    assert round(100 * reading.peak_share) == 82
    assert reading.q_read == pytest.approx(22.0, abs=0.3)


@pytest.mark.parametrize('kind', [LOG, LINEAR])
@pytest.mark.parametrize('q', [20.0, 40.0, 100.0, 300.0])
@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_at_the_rule_a_mode_reads_about_94_percent_and_its_q_20_low(kind, q,
                                                                    sign):
    """What the docstrings and the guide say of the rule's own rate,
    measured: 93.0 to 94.8 % of the height for a Q of 20 up, the Q 18.5
    to 22.5 % low, on either law and in either direction."""
    reading = simulate_sweep(100.0, q, sign * rule_rate(100.0, q, kind), kind)
    assert 0.929 < reading.peak_share < 0.948
    assert -0.225 < reading.q_bias < -0.185


def test_a_rate_far_under_the_rule_is_taken_as_steady():
    """Just above `STEADY_RATIO` the simulation reads the steady peak to
    a fraction of a percent; below it, it is not run."""
    near = simulate_sweep(22.0, 40.0, 0.06 * rule_rate(22.0, 40.0))
    assert near.simulated
    assert near.peak_share == pytest.approx(1.0, abs=0.002)
    assert near.q_read == pytest.approx(40.0, rel=0.005)
    under = simulate_sweep(22.0, 40.0,
                           0.5 * STEADY_RATIO * rule_rate(22.0, 40.0))
    assert not under.simulated
    # the steady curve's own reading: a transmissibility peaks a hair
    # over Q sqrt(1 + 1/Q**2), a hair under fn, and a hair narrow
    assert under.peak_share == pytest.approx(1.0000780, abs=1e-6)
    assert under.peak_frequency == pytest.approx(21.99656, abs=1e-4)
    assert under.q_read == pytest.approx(39.9875, abs=1e-3)


def test_a_low_q_reads_the_same_either_side_of_the_steady_ratio():
    """At a Q of 3 the steady curve peaks 1.15 % over the height at fn
    and reads a Q of 2.85; the reading taken as steady says so, rather
    than the mode's own 1 and 3, so it does not jump where the
    simulation takes over."""
    q = 3.0
    zeta = 1 / (2 * q)
    r = np.linspace(0.5, 1.5, 200001)
    curve = np.sqrt((1 + (2 * zeta * r) ** 2)
                    / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2))
    expected = curve.max() / (q * np.sqrt(1 + 1 / q ** 2))
    under = simulate_sweep(10.0, q, 0.98 * STEADY_RATIO * rule_rate(10.0, q))
    over = simulate_sweep(10.0, q, 1.02 * STEADY_RATIO * rule_rate(10.0, q))
    assert not under.simulated and over.simulated
    assert under.peak_share == pytest.approx(expected, abs=1e-5)
    assert under.peak_frequency == pytest.approx(10.0 * r[curve.argmax()],
                                                 abs=1e-3)
    assert under.q_read == pytest.approx(2.853, abs=0.001)
    assert under.q_read == pytest.approx(over.q_read, rel=0.001)
    assert under.peak_share == pytest.approx(over.peak_share, abs=0.005)


def test_the_reading_depends_on_the_rate_over_the_frequency_only():
    """The same multiple of the rule reads the same at 22 Hz and at
    2.2 kHz: the simulation is made at 1 Hz and scaled."""
    a = simulate_sweep(22.0, 40.0, 4.0)
    b = simulate_sweep(2200.0, 40.0, 400.0)
    assert b.peak_share == pytest.approx(a.peak_share, rel=1e-9)
    assert b.peak_frequency == pytest.approx(100.0 * a.peak_frequency)


def test_the_steady_height_is_the_transmissibility_at_the_mode():
    """Q sqrt(1 + 1/Q**2), which a low Q tells apart from Q: a slow
    sweep of a Q 3 mode reads the curve's own maximum over that, and
    the maximum is worked out here from the one-mass transmissibility."""
    q = 3.0
    zeta = 1 / (2 * q)
    r = np.linspace(0.5, 1.5, 200001)
    curve = np.sqrt((1 + (2 * zeta * r) ** 2)
                    / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2))
    expected = curve.max() / (q * np.sqrt(1 + 1 / q ** 2))
    reading = simulate_sweep(10.0, q, 0.06 * rule_rate(10.0, q))
    assert reading.simulated
    # within half a percent: so broad a peak still lags the sweep a
    # little at this rate, which is what the low Q is here to show
    assert reading.peak_share == pytest.approx(expected, abs=0.006)
    assert expected > 1.01


def test_a_slow_linear_sweep_is_taken_as_steady_too():
    reading = simulate_sweep(100.0, 30.0,
                             0.5 * STEADY_RATIO * rule_rate(100.0, 30.0,
                                                            LINEAR), LINEAR)
    assert not reading.simulated


def test_the_reading_says_how_far_low_it_reads():
    reading = simulate_sweep(22.0, 40.0, 4.0)
    assert reading.under_read == pytest.approx(1 - reading.peak_share)
    assert reading.under_read == pytest.approx(0.19, abs=0.01)
    assert reading.q_bias == pytest.approx(21.6 / 40.0 - 1.0, abs=0.003)


def test_the_peak_is_placed_between_readings_and_its_width_between_lines():
    """On a Gaussian, whose logarithm is a parabola, the top is placed
    exactly however coarsely it is sampled, and the half-power width is
    2 s sqrt(ln 2); read the same in sweep order down as up."""
    from visualdynamics.core.sweep_rate import _read_peak

    f = np.arange(0.0, 2.0, 0.0005)
    s = 0.01
    level = 5.0 * np.exp(-((f - 1.000137) ** 2) / (2 * s ** 2))
    for curve in ((f, level), (f[::-1], level[::-1])):
        top, height, q = _read_peak(*curve, (0.0, 2.0))
        assert top == pytest.approx(1.000137, abs=1e-9)
        assert height == pytest.approx(5.0, rel=1e-9)
        assert q == pytest.approx(1.000137 / (2 * s * np.sqrt(np.log(2))),
                                  rel=1e-4)
    # a peak at the end of what was read has no crossings beyond it
    edge = level[:1999]
    assert _read_peak(f[:1999], edge, (0.0, 2.0))[2] is None


@pytest.mark.parametrize('args', [(0.0, 40.0, 4.0), (22.0, 0.5, 4.0),
                                  (22.0, 40.0, 0.0), (22.0, np.nan, 4.0)])
def test_a_reading_with_no_resonance_or_no_sweep_is_refused(args):
    with pytest.raises(ValueError):
        simulate_sweep(*args)


# ---- the check against a specification --------------------------------------


def test_the_chapter_sweep_is_too_fast_for_the_mode_it_searches():
    """4 oct/min on a 22 Hz, Q 40 mode: 3.4 times the rule's rate, 0.54 s
    and 12 cycles of the 40 it needs."""
    check, = check_sweep_rate(specification(tone([10.0, 40.0], [4.0])),
                              [(22.0, 40.0)])
    assert check.tone == 'Sweep' and check.direction == 'up'
    assert check.limit == pytest.approx(1.190, abs=0.001)
    assert check.ratio == pytest.approx(3.36, abs=0.01)
    assert check.cycles == pytest.approx(11.9, abs=0.05)
    assert check.cycles_needed == 40.0
    assert check.too_fast and check.verdict == 'too fast'
    assert round(100 * check.peak_share) == 81
    assert check.start == pytest.approx(60 * np.log2(21.725 / 10.0) / 4.0)
    assert check.describe() == (
        "22 Hz, Q 40, up at 4 oct/min: 0.541 s and 11.9 cycles in the "
        "band (40 to build), 3.4 times the rule's 1.19 oct/min; reads "
        "81 % of the peak and a Q of 21.6; too fast")


def test_a_mode_the_rate_lets_build_is_not_flagged():
    check, = check_sweep_rate(tone([10.0, 40.0], [4.0]), [(35.0, 10.0)])
    assert check.ratio == pytest.approx(4.0 / rule_rate(35.0, 10.0))
    assert not check.too_fast and check.verdict == 'builds'


def test_the_margin_moves_the_flag():
    check, = check_sweep_rate(tone([10.0, 40.0], [4.0]), [(22.0, 40.0)],
                              margin=4.0)
    assert not check.too_fast
    check, = check_sweep_rate(tone([10.0, 40.0], [1.0]), [(22.0, 40.0)],
                              margin=0.5)
    assert check.too_fast


def test_a_tone_says_when_a_segment_sweeps_through_a_frequency():
    """`SineTone.time_at`, the inverse of `frequency_at` on a segment,
    which the passes are timed by."""
    sweep = tone([10.0, 40.0, 20.0], [4.0, -2.0], [LOG, LINEAR])
    assert sweep.time_at(20.0, 0) == pytest.approx(15.0)
    assert sweep.time_at(30.0, 1) == pytest.approx(30.0 + 5.0)
    t = np.array([3.0, 17.0, 33.0, 39.0])
    for i, segment in enumerate([0, 0, 1, 1]):
        assert sweep.time_at(sweep.frequency_at(t[i]), segment) == \
            pytest.approx(t[i])


def test_a_mode_outside_the_sweep_has_no_row():
    checks = check_sweep_rate(tone([10.0, 40.0], [4.0]),
                              [(5.0, 20.0), (22.0, 40.0), (60.0, 20.0)])
    assert [check.frequency for check in checks] == [22.0]


def test_a_sweep_up_and_back_passes_each_mode_twice():
    checks = check_sweep_rate(tone([5.0, 100.0, 5.0], [2.0, -2.0]),
                              [(20.0, 20.0)])
    assert [check.direction for check in checks] == ['up', 'down']
    assert checks[0].seconds == pytest.approx(checks[1].seconds)
    assert checks[1].start > checks[0].start + checks[0].seconds


def test_a_band_across_a_breakpoint_is_one_pass_judged_at_the_faster_rate():
    """The rate changes at 22.1 Hz, inside a 22 Hz, Q 40 band: one pass,
    its time the two segments' together, judged at 4 oct/min, the
    faster, though the mode itself is swept at 1."""
    sweep = tone([10.0, 22.1, 40.0], [1.0, 4.0])
    check, = check_sweep_rate(sweep, [(22.0, 40.0)])
    assert check.rate == 4.0 and check.crossed and check.too_fast
    assert round(100 * check.peak_share) == 81
    expected = (60 * np.log2(22.1 / 21.725) / 1.0
                + 60 * np.log2(22.275 / 22.1) / 4.0)
    assert check.seconds == pytest.approx(expected)


def test_the_rate_checked_is_the_faster_of_the_segments_in_the_band():
    sweep = tone([10.0, 21.9, 40.0], [1.0, 4.0])
    check, = check_sweep_rate(sweep, [(22.0, 40.0)])
    assert check.rate == 4.0
    sweep = tone([10.0, 21.9, 40.0], [4.0, 1.0])
    check, = check_sweep_rate(sweep, [(22.0, 40.0)])
    assert check.rate == 4.0


def test_a_log_and_a_linear_segment_are_compared_as_multiples_of_the_rule():
    """1 Hz/s is 3.3 times a 22 Hz, Q 40 mode's linear rule
    (22**2 / 40**2 = 0.3025 Hz/s) and 2 oct/min 1.7 times its log rule
    (1.19 oct/min): the linear segment is the faster, though its rate is
    the smaller number."""
    sweep = tone([10.0, 22.1, 40.0], [2.0, 1.0], [LOG, LINEAR])
    check, = check_sweep_rate(sweep, [(22.0, 40.0)])
    assert check.segment_type == LINEAR and check.rate == 1.0
    assert check.ratio == pytest.approx(1.0 / 0.3025)


def test_a_sweep_that_reads_no_peak_says_so():
    """A sweep through two octaves in under a cycle leaves nothing to
    read, and says so in words rather than as a zero."""
    check, = check_sweep_rate(tone([5.0, 100.0], [20000.0]), [(22.0, 10.0)])
    assert np.isnan(check.peak_share) and check.q_read is None
    assert 'reads no peak and no Q; too fast' in check.describe()


def test_a_sweep_that_ends_inside_the_band_is_not_a_full_crossing():
    check, = check_sweep_rate(tone([10.0, 39.9], [4.0]), [(39.9, 20.0)])
    assert not check.crossed
    assert check.seconds == pytest.approx(60 * np.log2(1 / 0.975) / 4.0)
    assert 'not a full crossing' in check.describe()


def test_each_tone_of_a_specification_is_checked_unless_one_is_named():
    spec = specification(tone([10.0, 40.0], [4.0], name='Fast'),
                         tone([10.0, 40.0], [0.25], name='Slow'))
    checks = check_sweep_rate(spec, [(22.0, 40.0)])
    assert [(check.tone, check.too_fast) for check in checks] == \
        [('Fast', True), ('Slow', False)]
    only, = check_sweep_rate(spec, [(22.0, 40.0)], tone='Slow')
    assert only.tone == 'Slow'


def test_a_shape_sets_modes_are_checked_with_q_from_their_damping():
    from visualdynamics.core.shapes import ShapeSet

    # a rigid-body mode, with or without damping, an undamped one and
    # one at or past critical damping have no band, and are left out
    shapes = ShapeSet([0.0, 0.0, 22.0, 30.0, 35.0, 25.0, 28.0],
                      [0.0, 0.02, 0.0125, 0.0, 0.05, 1.0, 1.5], ['1Z+'],
                      np.ones((7, 1)))
    checks = check_sweep_rate(tone([10.0, 40.0], [4.0]), shapes)
    assert [(check.frequency, check.q) for check in checks] == \
        [(22.0, pytest.approx(40.0)), (35.0, pytest.approx(10.0))]


def test_anything_but_a_sine_specification_is_refused():
    with pytest.raises(TypeError):
        check_sweep_rate([1.0, 2.0], [(22.0, 40.0)])
    with pytest.raises(KeyError):
        check_sweep_rate(tone([10.0, 40.0], [4.0]), [(22.0, 40.0)],
                         tone='Other')
    assert check_sweep_rate(tone([10.0, 40.0], [4.0]), [(22.0, 40.0)],
                            tone='Sweep')


# ---- a Q read off a swept level ------------------------------------------------


@pytest.mark.parametrize('rate', [1.0, 4.0, 8.0])
def test_a_swept_q_is_traced_back_to_the_mode_that_reads_it(rate):
    """Each swept reading of the 22 Hz, Q 40 mode implies a Q of 40, and
    each is flagged: at 1 oct/min the rate is under the rule's, but the
    width has already spread the Q 17 percent low."""
    read = simulate_sweep(22.0, 40.0, rate).q_read
    judged = swept_q(22.0, read, rate)
    assert judged.q_true == pytest.approx(40.0, rel=0.02)
    assert judged.flagged
    assert 'reads low' in judged.describe()


def test_a_q_read_at_a_slow_sweep_stands():
    judged = swept_q(22.0, 39.5, 0.1)
    assert judged.q_true == pytest.approx(39.5, rel=0.01)
    assert not judged.flagged
    assert judged.describe().endswith('slow enough for it')


def test_a_q_wider_than_any_mode_can_read_is_said_to_be_unreadable():
    """At 8 oct/min the swept Q of a 22 Hz mode tops out near 19: a
    reading of 25 matches no mode, so no Q is inferred and it is
    flagged."""
    judged = swept_q(22.0, 25.0, 8.0)
    assert judged.q_true is None and judged.flagged
    assert judged.limit == pytest.approx(rule_rate(22.0, 25.0))
    assert 'too fast to read any Q' in judged.describe()


def _tracked(rate=4.0, frequencies=(10.0, 40.0), rates=None, lines=4000):
    """The chapter's record: a 1 g sweep from 10 to 40 Hz at `rate`
    (or over `frequencies` at `rates`) through a 22 Hz, Q 40 mode, and
    its response tracked with the rms detector `lines` times along the
    sweep."""
    from scipy.signal import lfilter

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.sine_tracking import SineTracking, track_sine
    from visualdynamics.core.srs import damping_for, ramp_invariant

    sweep = tone(list(frequencies), [rate] if rates is None else list(rates),
                 name=f'{rate:g} oct/min')
    spec = specification(sweep)
    fs = 2048.0
    t = np.arange(0.0, sweep.duration(), 1.0 / fs)
    x = np.sin(sweep.phase_at(t))
    b, a = ramp_invariant([22.0], fs, damping_for(40.0))
    y = lfilter(b[0], a[0], x)
    history = TimeHistory(t, np.vstack([x, y]) * G,
                          response_dof=['1Z+', '101Z+'],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['g'] * 2)
    level, = track_sine(history, spec, SineTracking(detector='rms'),
                        channels=['101Z+'], onset=0.0, lines=lines)
    return spec, level


def test_the_q_read_off_a_tracked_level_is_flagged_against_its_sweep():
    """The level the application tracks puts the peak at 22.61 Hz with
    a Q of 22.0; swept at 4 oct/min, that Q reads low, and the mode it
    implies is near the 40 that made it."""
    spec, level = _tracked()
    judged = read_swept_q(level, spec, 22.0)
    assert judged.frequency == pytest.approx(22.61, abs=0.02)
    assert judged.q_read == pytest.approx(22.0, abs=0.2)
    assert judged.height / G == pytest.approx(32.6, abs=0.1)
    assert judged.rate == 4.0 and judged.unit == 'oct/min'
    assert judged.flagged
    assert judged.q_true == pytest.approx(40.0, rel=0.06)
    assert judged.ratio == pytest.approx(
        4.0 / rule_rate(judged.frequency, judged.q_true))


def test_the_level_is_judged_against_its_own_tone():
    """A specification of two tones: the level's rate is its own tone's
    (4 oct/min), not the first tone's (1 oct/min)."""
    spec, level = _tracked()
    both = specification(tone([10.0, 40.0], [1.0], name='Other'),
                         spec.tones[0])
    assert read_swept_q(level, both, 22.0).rate == 4.0


def test_no_peak_in_the_window_gives_no_reading():
    """Nothing read where the level has no readings, nor where its
    tallest reading in the window is the skirt of the 22 Hz mode."""
    spec, level = _tracked()
    assert read_swept_q(level, spec, 100.0) is None
    assert read_swept_q(level, spec, 35.0) is None


def test_readings_under_the_noise_floor_are_not_read():
    """A level whose readings around the mode are all under their floor
    has no peak there to read, though their values stand."""
    spec, level = _tracked()
    level.below_floor = np.zeros(np.shape(level.ordinate), dtype=bool)
    level.below_floor[0, np.abs(level.abscissa - 22.0) < 4.4] = True
    assert read_swept_q(level, spec, 22.0) is None
    level.below_floor[:] = False
    assert read_swept_q(level, spec, 22.0).height / G == pytest.approx(
        32.6, abs=0.1)


def test_a_level_of_a_tone_the_specification_lacks_is_refused():
    spec, level = _tracked()
    other = specification(tone([10.0, 40.0], [1.0], name='Other'))
    with pytest.raises(KeyError, match="no tone named '4 oct/min'"):
        read_swept_q(level, other, 22.0)
    # given the tone itself, the caller has chosen it
    assert read_swept_q(level, spec.tones[0], 22.0).rate == 4.0


def test_a_sweep_up_and_back_is_read_on_the_pass_that_holds_the_peak():
    """Up at 4 oct/min and back at 1: the tracked level, sorted by
    frequency, holds the two passes interleaved. Read by its clock, the
    tallest peak is the slow pass's, below the mode, at 1 oct/min down,
    with the Q a 1 oct/min sweep reads (33, not 22)."""
    spec, level = _tracked(frequencies=(10.0, 40.0, 10.0), rates=(4.0, -1.0),
                           lines=12000)
    judged = read_swept_q(level, spec, 22.0)
    assert judged.rate == -1.0
    assert judged.frequency == pytest.approx(21.78, abs=0.03)
    assert judged.q_read == pytest.approx(33.0, abs=0.7)
    assert judged.q_true == pytest.approx(40.0, rel=0.06)
    # without its clock the two passes cannot be told apart
    level.seconds = None
    assert read_swept_q(level, spec, 22.0) is None


# ---- the plot ------------------------------------------------------------------


def test_the_plot_marks_each_crossing_by_its_verdict(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.sweep_rate import build_sweep_rate

    layout = pg.GraphicsLayoutWidget()
    checks = build_sweep_rate(layout, tone([5.0, 100.0, 5.0], [4.0, -4.0]),
                              [(22.0, 40.0), (9.0, 10.0)])
    plot = layout.getItem(0, 0)
    crossings = [item for item in plot.items
                 if getattr(item, 'is_crossing', False)]
    assert len(crossings) == len(checks) == 4
    for item in crossings:
        style = item.opts['pen'].style()
        dashed = style == pg.QtCore.Qt.PenStyle.DashLine
        assert dashed == item.too_fast, 'too fast is dashed, building solid'
    marks = [item.opts['symbol'] for item in plot.items
             if getattr(item, 'is_mode_mark', False)]
    assert sorted(marks) == ['o', 'o', 'x', 'x']
    labels = [item.toPlainText() for item in plot.items
              if getattr(item, 'is_check_label', False)]
    assert len(labels) == 2, 'one label per mode, not per pass'
    assert any(text.startswith('22 Hz, Q 40: 0.54 s') and 'too fast' in text
               for text in labels)
    assert any('builds' in text for text in labels)
    bands = [item for item in plot.items if getattr(item, 'is_band', False)]
    assert len(bands) == 2


def test_a_strip_is_tinted_for_its_modes_worst_pass(qt_app):
    """Up at a quarter octave a minute, which the 22 Hz, Q 40 mode
    builds under, and back at 4, which is too fast: the strip is the
    warning's tint, though the first pass builds."""
    import pyqtgraph as pg

    from visualdynamics.plot.sweep_rate import build_sweep_rate
    from visualdynamics.theme import theme

    layout = pg.GraphicsLayoutWidget()
    checks = build_sweep_rate(layout, tone([5.0, 100.0, 5.0], [0.25, -4.0]),
                              [(22.0, 40.0)], theme='light')
    assert [check.too_fast for check in checks] == [False, True]
    band, = [item for item in layout.getItem(0, 0).items
             if getattr(item, 'is_band', False)]
    assert band.too_fast
    warning = pg.mkColor(theme('light')['limit_warning'])
    assert band.brush.color().rgb() == warning.rgb()


def test_plot_sweep_rate_renders_headless(qt_app, tmp_path):
    from visualdynamics.plot import plot_sweep_rate

    path = plot_sweep_rate(specification(tone([10.0, 40.0], [4.0])),
                           [(22.0, 40.0)], theme='light',
                           path=tmp_path / 'sweep.png', show=False)
    assert (tmp_path / 'sweep.png').stat().st_size > 5000
    assert str(path).endswith('sweep.png')


# ---- the reading in the window ----------------------------------------------------


def _show_spec(window, pump, *names):
    spec = specification(tone([10.0, 40.0], [4.0], name='Search'))
    window.add_object('Sweep', spec)
    select_objects(window, pump, 'Sweep', *names)
    window.render_current()
    pump()
    return spec


def _titles(model):
    from PySide6.QtCore import Qt

    return [model.headerData(c, Qt.Orientation.Horizontal)
            for c in range(model.columnCount())]


def _sweep_rate(window, pump, modes='22 40, 35 10'):
    pane = window.data_pane
    pane.sweep_rate_action.setChecked(True)
    pane._choose_toggle('sweep_rate', True)
    pump()
    panel = pane.sweep_rate_panel
    panel.modes_box.setText(modes)
    panel.modes_box.editingFinished.emit()
    pump()
    return pane, panel


def test_the_sweep_rate_reading_is_offered_beside_a_sine_specification(window,
                                                                       pump):
    _show_spec(window, pump)
    assert window.data_pane.sweep_rate_action.isVisible()


def test_it_is_not_offered_for_other_data(window, pump):
    from test_extract_sine import _flat_sweep

    history, _spec = _flat_sweep(noise=6.0)
    window.add_object('Record', history)
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    assert not window.data_pane.sweep_rate_action.isVisible()


def test_the_modes_typed_are_checked_in_the_table_and_the_panel(window,
                                                                pump):
    _show_spec(window, pump)
    _pane, panel = _sweep_rate(window, pump)
    assert panel.isVisible()
    assert panel.derived['passes'].text() == '2'
    assert panel.derived['fast'].text() == '1'
    model = window.table.model()
    titles = _titles(model)
    assert titles[:3] == ['Tone', 'Mode [Hz]', 'Q']
    verdict = titles.index('Verdict')
    assert [model.index(r, verdict).data() for r in range(model.rowCount())] \
        == ['too fast', 'builds']
    assert model.index(0, titles.index('× Rule')).data() == '3.36'
    assert model.index(0, titles.index('Cycles')).data() == '11.9'
    assert 'Search: 2 passes through 2 modes, 1 too fast' in \
        window.statusBar().currentMessage()


def test_modes_that_do_not_read_are_refused_where_typed(window, pump):
    _show_spec(window, pump)
    _pane, panel = _sweep_rate(window, pump, '22 40')
    panel.modes_box.setText('22 forty')
    panel.modes_box.editingFinished.emit()
    pump()
    assert panel.modes_box.text() == '22 40'
    assert 'is not a frequency and a Q' in panel.readout.text()
    assert panel.settings()['modes'] == [(22.0, 40.0)]


def test_with_no_modes_the_panel_asks_for_them(window, pump):
    _show_spec(window, pump)
    _pane, panel = _sweep_rate(window, pump, '')
    assert panel.derived['passes'].text() == '—'
    assert 'Type the modes' in panel.readout.text()


def test_a_tracked_level_selected_with_it_has_its_q_flagged(window, pump):
    spec, level = _tracked()
    window.add_object('Sweep', spec)
    window.add_object('Level', level)
    select_objects(window, pump, 'Sweep', 'Level')
    window.render_current()
    pump()
    _pane, panel = _sweep_rate(window, pump, '22 40')
    model = window.table.model()
    titles = _titles(model)
    rows = [[model.index(r, c).data() for c in range(model.columnCount())]
            for r in range(model.rowCount())]
    level_row = next(row for row in rows
                     if row[titles.index('Pass')] == 'level at 101Z+')
    assert float(level_row[titles.index('Q read')]) == pytest.approx(22.0,
                                                                     abs=0.2)
    assert level_row[titles.index('Verdict')].startswith('reads low; Q about')
    assert 'reads low' in panel.readout.text()
    assert '1 Q read off the level flagged low' in \
        window.statusBar().currentMessage()


def test_the_reading_is_journaled_as_its_plot_call(window, pump):
    _show_spec(window, pump)
    _sweep_rate(window, pump, '22 40')
    assert window.project.journal[-1] == (
        "visualdynamics.plot.plot_sweep_rate(project['Sweep'], "
        "[(22.0, 40.0)], tone='Search', margin=1.0, "
        "path='sweep_rate.png', show=False)")


def test_the_qs_read_off_a_level_are_journaled_as_their_calls(window, pump):
    """A settling line per level beside the plot call, which a script
    replays to the same judgments, and which brings its import."""
    import visualdynamics
    import visualdynamics.core.sweep_rate

    spec, level = _tracked()
    window.add_object('Sweep', spec)
    window.add_object('Level', level)
    select_objects(window, pump, 'Sweep', 'Level')
    window.render_current()
    pump()
    _sweep_rate(window, pump, '22 40, 35 10')
    line = window.project.journal[-1]
    assert line == (
        "[visualdynamics.core.sweep_rate.read_swept_q(project['Level'], "
        "project['Sweep'], frequency, record=record, margin=1.0) "
        "for record in [0] for frequency in [22.0, 35.0]]")
    assert window.project.journal[-2].startswith(
        "visualdynamics.plot.plot_sweep_rate(project['Sweep'], "
        "[(22.0, 40.0), (35.0, 10.0)]")
    judged, skirt = eval(line, {'visualdynamics': visualdynamics,
                                'project': window.project})
    assert judged.q_read == pytest.approx(22.0, abs=0.2) and skirt is None
    assert 'import visualdynamics.core.sweep_rate' in \
        window.project.session_script()
    # a redraw settles the line rather than adding one
    _sweep_rate(window, pump, '22 40')
    assert window.project.journal[-1].endswith('for frequency in [22.0]]')
    assert sum('read_swept_q' in each for each in window.project.journal) == 1


def test_putting_it_away_brings_the_specification_back(window, pump):
    _show_spec(window, pump)
    pane, panel = _sweep_rate(window, pump)
    pane.sweep_rate_action.setChecked(False)
    pane._choose_toggle('sweep_rate', False)
    pump()
    assert not panel.isVisible()
    assert 'Search' in window.statusBar().currentMessage()
    assert 'passes through' not in window.statusBar().currentMessage()
