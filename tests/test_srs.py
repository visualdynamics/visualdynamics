"""Shock response spectra: the filter, the curve, and the target.

An SRS is easy to compute and easy to get subtly wrong, so these pin it
against things that are known independently of the implementation: the
gain of the filter at DC and at resonance, both closed form; the
high-frequency asymptote of a half-sine, which is a textbook result;
and the recursion itself against scipy running the same coefficients.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import srs
from visualdynamics.core.data import ShockSpecification, Srs, TimeHistory

RATE = 10000.0


def half_sine(peak=50.0, duration=0.011, span=0.5, rate=RATE):
    """The classic shock: a half-sine acceleration pulse, then silence."""
    n = int(duration * rate)
    values = np.zeros(int(span * rate))
    values[:n] = peak * np.sin(np.pi * np.arange(n) / n)
    return values


def history(values, dof='101Z+', rate=RATE):
    values = np.atleast_2d(values)
    t = np.arange(values.shape[1]) / rate
    return TimeHistory(t, values, response_dof=[dof] * values.shape[0],
                       ordinate_dim=['acceleration'] * values.shape[0])


# ---- the filter ---------------------------------------------------------


def test_the_filter_passes_dc_exactly():
    """An infinitely stiff oscillator goes wherever its base goes, so
    the gain at zero frequency is one. It is the cheapest statement that
    the coefficients are the right ones, and it is exact — not close."""
    frequencies = srs.octave_frequencies(5.0, 2000.0)
    b, a = srs.ramp_invariant(frequencies, RATE, 0.05)
    assert np.allclose(b.sum(axis=1) / a.sum(axis=1), 1.0, rtol=0, atol=1e-14)


@pytest.mark.parametrize('q', [5.0, 10.0, 50.0])
def test_the_gain_at_resonance_is_the_amplification(q):
    """Driven at its own frequency, an oscillator answers at Q — near
    enough. The closed form for absolute acceleration is
    sqrt(1 + (2 zeta)^2) / (2 zeta), which is Q for small damping."""
    fn = 100.0
    t = np.arange(int(4.0 * RATE)) / RATE
    highest, _lowest = srs.peaks(np.sin(2 * np.pi * fn * t),
                                 np.array([fn]), RATE, q=q)
    zeta = srs.damping_for(q)
    exact = np.sqrt(1.0 + 4.0 * zeta ** 2) / (2.0 * zeta)
    assert highest[0] == pytest.approx(exact, rel=0.02)


def test_the_recursion_matches_scipy_running_the_same_filter():
    """The recursion is written out by hand because scipy is a test
    dependency and not a runtime one. This is what says the hand-written
    one is the filter the coefficients describe."""
    lfilter = pytest.importorskip('scipy.signal').lfilter

    frequencies = srs.octave_frequencies(20.0, 1000.0)
    shock = half_sine()
    b, a = srs.ramp_invariant(frequencies, RATE, srs.damping_for())
    highest, lowest = srs.peaks(shock, frequencies, RATE)
    for k in range(len(frequencies)):
        reference = lfilter(b[k], a[k], shock)
        assert highest[k] == pytest.approx(reference.max(), rel=1e-10)
        assert lowest[k] == pytest.approx(reference.min(), rel=1e-10)


def test_damping_and_q_are_each_others_inverse():
    assert srs.damping_for(10.0) == 0.05
    assert srs.q_for(0.05) == 10.0
    assert srs.q_for(srs.damping_for(37.0)) == pytest.approx(37.0)


def test_critical_damping_is_refused():
    with pytest.raises(ValueError, match='under critical'):
        srs.ramp_invariant(np.array([100.0]), RATE, 1.0)
    with pytest.raises(ValueError, match='under critical'):
        srs.ramp_invariant(np.array([100.0]), RATE, 0.0)


def test_the_filter_is_exactly_ramp_invariant():
    """Ramp-invariant means exact for an input that is linear between
    samples: the filter's output then *is* the continuous oscillator's
    absolute acceleration, (2 zeta wn s + wn^2) / (s^2 + 2 zeta wn s +
    wn^2), and scipy's `lsim` with linear interpolation computes that
    response independently. Agreement to rounding on a random base
    acceleration sampled at only five times the natural frequency is
    the coefficients themselves checked — the scipy comparison above
    runs these same coefficients, and the DC gain is one whatever the
    damped frequency is, so a damped frequency of wn*sqrt(1 + zeta^2)
    or a sinc divided rather than multiplied passed every test before
    (mutation testing, 2026-10-10); they are 5 % and 82 % out here."""
    from scipy.signal import lfilter, lsim

    natural, zeta, rate = 100.0, 0.05, 500.0
    t = np.arange(200) / rate
    base = np.random.default_rng(3).normal(size=t.size)
    base[0] = 0.0
    b, a = srs.ramp_invariant(np.array([natural]), rate, zeta)
    wn = 2.0 * np.pi * natural
    _t, exact, _x = lsim(([2.0 * zeta * wn, wn ** 2],
                          [1.0, 2.0 * zeta * wn, wn ** 2]), base, t, interp=True)
    response = lfilter(b[0], a[0], base)
    assert np.abs(response - exact).max() < 1e-12 * np.abs(exact).max()


# ---- the curve ----------------------------------------------------------


def test_the_high_frequency_end_is_the_peak_of_the_shock():
    """The textbook result. A stiff enough oscillator has no time to
    respond and simply rides the base, so the SRS flattens out at the
    largest acceleration the base ever reached."""
    shock = half_sine(peak=50.0)
    frequencies = srs.octave_frequencies(10.0, 2000.0)
    spectrum = srs.maximax(shock, frequencies, RATE)
    assert spectrum[-1] == pytest.approx(np.abs(shock).max(), rel=1e-3)


def test_the_peak_of_a_half_sine_lands_where_it_should():
    """A half-sine of duration T peaks around f*T ~ 0.8, amplifying the
    input by about 1.7. Both are properties of the pulse, not of this
    code."""
    duration = 0.011
    shock = half_sine(peak=50.0, duration=duration)
    frequencies = srs.octave_frequencies(10.0, 2000.0)
    spectrum = srs.maximax(shock, frequencies, RATE)
    at = frequencies[np.argmax(spectrum)]
    assert at * duration == pytest.approx(0.8, rel=0.15)
    assert spectrum.max() / 50.0 == pytest.approx(1.7, rel=0.15)


def test_a_bigger_shock_is_bigger_everywhere():
    """The filter is linear, so twice the input is exactly twice the
    spectrum — at every frequency, not on average."""
    frequencies = srs.octave_frequencies(10.0, 2000.0)
    one = srs.maximax(half_sine(peak=50.0), frequencies, RATE)
    two = srs.maximax(half_sine(peak=100.0), frequencies, RATE)
    assert np.allclose(two, 2.0 * one, rtol=1e-12)


def test_more_damping_is_a_lower_spectrum():
    frequencies = srs.octave_frequencies(10.0, 2000.0)
    shock = half_sine()
    light = srs.maximax(shock, frequencies, RATE, q=50.0)
    heavy = srs.maximax(shock, frequencies, RATE, q=5.0)
    assert (heavy <= light + 1e-9).all()
    assert heavy.max() < light.max()


def test_the_octave_grid_is_geometric():
    frequencies = srs.octave_frequencies(10.0, 80.0, per_octave=6)
    assert frequencies[0] == pytest.approx(10.0)
    assert frequencies[-1] == pytest.approx(80.0)
    assert len(frequencies) == 19, 'three octaves at six a side, plus the end'
    ratios = frequencies[1:] / frequencies[:-1]
    assert np.allclose(ratios, 2.0 ** (1 / 6))


def test_a_band_that_is_not_a_band_is_refused():
    with pytest.raises(ValueError, match='not a band'):
        srs.octave_frequencies(100.0, 10.0)
    with pytest.raises(ValueError, match='not a band'):
        srs.octave_frequencies(10.0, 10.0)
    with pytest.raises(ValueError, match='not a band'):
        srs.octave_frequencies(0.0, 10.0)


# ---- the object ---------------------------------------------------------


def test_each_record_gets_its_own_spectrum():
    """A shock is one event. Several shocks in one recording are several
    records, and a test is judged on the worst of them, so they are not
    averaged into one curve."""
    spectra = history(np.vstack([half_sine(50.0), half_sine(30.0)])
                      ).compute_srs()
    assert isinstance(spectra, Srs)
    assert spectra.num_records == 2
    assert spectra.ordinate[0, -1] == pytest.approx(50.0, rel=1e-3)
    assert spectra.ordinate[1, -1] == pytest.approx(30.0, rel=1e-3)


def test_the_spectrum_keeps_the_quantity_it_was_measured_in():
    spectra = history(half_sine()).compute_srs()
    assert spectra.ordinate_dim == ['acceleration']
    assert spectra.ordinate_unit == ['m/s**2']
    assert spectra.abscissa_dim == 'frequency'


def test_the_amplification_belongs_to_the_curve():
    """An SRS at Q=10 and the same shock at Q=50 are different curves,
    so the Q is part of what the object is rather than something the
    caller has to remember."""
    spectra = history(half_sine()).compute_srs(q=25.0)
    assert spectra.q == 25.0
    assert spectra.damping == pytest.approx(0.02)
    assert 'Q=25' in repr(spectra)
    assert spectra != history(half_sine()).compute_srs(q=10.0)


def test_an_srs_round_trips_through_vibe(tmp_path):
    spectra = history(half_sine()).compute_srs(q=25.0, kind='positive')
    visualdynamics.save(spectra, tmp_path / 'shock')
    back = visualdynamics.load(tmp_path / 'shock.vdyn')
    assert isinstance(back, Srs)
    assert back.q == 25.0 and back.kind == 'positive'
    assert back == spectra


def test_positive_and_negative_are_read_separately():
    """A shock that pushes one way harder than the other says so, and
    maximax is the larger of the two."""
    shock = half_sine()          # a one-sided pulse
    frequencies = srs.octave_frequencies(10.0, 2000.0)
    highest, lowest = srs.peaks(shock, frequencies, RATE)
    assert (highest >= 0).all() and (lowest <= 0).all()
    assert np.allclose(srs.maximax(shock, frequencies, RATE),
                       np.maximum(highest, -lowest))


# ---- the target ---------------------------------------------------------


def target():
    """A shock specification: a required SRS, +6 dB and -3 dB around it."""
    frequencies = srs.octave_frequencies(20.0, 2000.0)
    required = 10.0 * (frequencies / 20.0) ** 0.5
    return ShockSpecification(
        abscissa=frequencies, ordinate=np.atleast_2d(required),
        response_dof=['101Z+'], ordinate_dim=['acceleration'],
        abort_upper=np.atleast_2d(required * 2.0),
        abort_lower=np.atleast_2d(required * 0.5))


def test_a_shock_target_is_an_srs_with_room_around_it():
    spec = target()
    assert isinstance(spec, Srs), 'it is an SRS in every other respect'
    assert spec.has_limits
    assert set(spec.limits) == {'abort_upper', 'abort_lower'}
    assert spec.function_type == Srs.function_type


def test_a_shock_targets_limit_comes_back_as_an_srs():
    """And at the target's own Q — a limit read at a different
    amplification is a different limit."""
    spec = target()
    upper = spec.limit('abort_upper')
    assert isinstance(upper, Srs) and not isinstance(upper, ShockSpecification)
    assert upper.q == spec.q and upper.kind == spec.kind
    assert np.allclose(upper.ordinate, spec.ordinate * 2.0)


def test_a_shock_target_round_trips_with_its_limits(tmp_path):
    spec = target()
    visualdynamics.save(spec, tmp_path / 'target')
    back = visualdynamics.load(tmp_path / 'target.vdyn')
    assert isinstance(back, ShockSpecification)
    assert back == spec
    assert set(back.limits) == set(spec.limits)


def test_the_limits_follow_the_ordinate_into_other_units():
    """The reason they are arrays beside the ordinate rather than
    objects of their own: one conversion moves all of them."""
    spec = target()
    shown = spec.display_ordinate(visualdynamics.IN_LBF_S)
    upper = spec.display_limit('abort_upper', visualdynamics.IN_LBF_S)
    assert np.allclose(upper, shown * 2.0)
    assert not np.allclose(shown, spec.ordinate), 'and it did convert'
