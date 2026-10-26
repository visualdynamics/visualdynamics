"""The fatigue damage and extreme response spectra, and time compression.

Pinned against hand values: a white density through a resonator, whose
response rms is Miles' equation and whose crossing rate is its natural
frequency, so the narrow-band damage and the expected largest peak have
closed forms; a resonator at the band's edge, which sees half its
resonance, and one far above the band, which follows its base; Rice's
peak distribution at its two ends; the trades of time for level of a
test document's worked examples. Against independent computations: the
response's mean square by adaptive quadrature over a coarse spectrum
and a breakpoint specification, rainflow against a sequential three-point
count written here, and the spectrum's damage against rainflow counts of
simulated records, through the spectrum as written and through one
computed from the record under the default Hann window. Then the
recursion's gain, what is refused with its cause named, a nonzero start
time, and the comparison of a test against a field environment.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import integrate

from visualdynamics.core import fatigue
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Psd, Specification, TimeHistory
from visualdynamics.core.fatigue import (
    CAP,
    Equivalence,
    ResponseSpectra,
    compared,
    compression_ratio,
    dirlik_damage,
    equivalence,
    exaggeration,
    fatigue_spectra,
    history_refusal,
    history_spectra,
    narrowband_damage,
    psd_spectra,
    pseudo_gain,
    rainflow,
    rainflow_damage,
    response_filter,
    rice_damage,
    rice_moment,
    spectrum_refusal,
    time_compression,
    turning_points,
)
from visualdynamics.core.progress import Cancelled
from visualdynamics.core.time_statistics import rice_rate

G = 9.80665
#: a flat 0.04 g^2/Hz, in SI
LEVEL = 0.04 * G ** 2


def _flat(low=1.0, high=10000.0, level=LEVEL, spacing=1.0, dof='1X+'):
    """A PSD of acceleration flat at `level` from `low` to `high`, zero
    elsewhere, on lines `spacing` apart from 0 Hz."""
    f = np.arange(0.0, 2.0 * high + spacing, spacing)
    values = np.where((f >= low) & (f <= high), level, 0.0)
    return Psd(f, values[None], response_dof=[dof], ordinate_dim=[
        fatigue.DENSITY])


def _miles(fn, q, level=LEVEL):
    return math.sqrt(math.pi / 2.0 * fn * q * level)


def _synth(seconds, rate, low, high, level=LEVEL, seed=3, start=0.0):
    """A record whose one-sided density is exactly `level` from `low` to
    `high`: every line at the same amplitude, random phases."""
    n = round(seconds * rate)
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / rate)
    amplitude = np.where((f >= low) & (f <= high),
                         math.sqrt(level * rate / n / 2.0) * n, 0.0)
    x = np.fft.irfft(amplitude * np.exp(2j * np.pi * rng.random(f.size)), n)
    t = start + np.arange(n) / rate
    return TimeHistory(t, x[None], response_dof=['1X+'],
                       ordinate_dim=['acceleration'], ordinate_unit=['g'])


# ---- a white density through a resonator ------------------------------

def test_miles_equation_gives_the_rms_and_the_rate_is_fn():
    """Over a band wide against the resonator, its rms is Miles' and it
    crosses zero upward fn times a second (both exact over all
    frequencies, since the integrals of |H|^2 and f^2 |H|^2 agree)."""
    result = psd_spectra(_flat(), b=4.0, duration=60.0,
                         frequencies=[100.0], q=10.0)
    assert result.rms[0] == pytest.approx(_miles(100.0, 10.0), rel=1e-3)
    assert result.crossings[0] == pytest.approx(100.0, rel=1e-3)


def test_the_mean_square_matches_quadrature_over_the_band():
    """Independent of Miles: the integral of |H|^2 G over the band itself,
    by adaptive quadrature, at two Q."""
    for q in (10.0, 50.0):
        result = psd_spectra(_flat(20.0, 2000.0), b=4.0, duration=1.0,
                             frequencies=[100.0], q=q)
        exact, _err = integrate.quad(
            lambda f, q=q: pseudo_gain(f, 100.0, q) * LEVEL, 19.5, 2000.5,
            points=[100.0], limit=400)
        assert result.rms[0] ** 2 == pytest.approx(exact, rel=1e-3)


def test_the_narrowband_damage_is_the_rayleigh_closed_form():
    """nu0 T (sqrt 2 sigma)^b Gamma(1 + b/2): at b = 2 that is
    2 fn T sigma^2, at b = 4 it is 8 fn T sigma^4."""
    fn, seconds = 100.0, 60.0
    for b, closed in ((2.0, lambda s: 2.0 * fn * seconds * s ** 2),
                      (4.0, lambda s: 8.0 * fn * seconds * s ** 4)):
        result = psd_spectra(_flat(), b=b, duration=seconds,
                             frequencies=[fn], q=10.0)
        assert result.fds[0] == pytest.approx(
            closed(_miles(fn, 10.0)), rel=5e-3)


def test_the_ers_is_sigma_times_the_expected_largest_peak():
    fn, seconds = 100.0, 3600.0
    result = psd_spectra(_flat(), b=4.0, duration=seconds,
                         frequencies=[fn], q=10.0)
    hand = _miles(fn, 10.0) * math.sqrt(2.0 * math.log(fn * seconds))
    assert result.ers[0] == pytest.approx(hand, rel=1e-3)


def test_the_damage_goes_as_the_duration_and_the_level_to_b_over_2():
    """Twice the time is twice the damage; four times the density is
    4^(b/2) times it."""
    one = psd_spectra(_flat(), b=6.0, duration=10.0, frequencies=[80.0])
    two = psd_spectra(_flat(), b=6.0, duration=20.0, frequencies=[80.0])
    four = psd_spectra(_flat(level=4 * LEVEL), b=6.0, duration=10.0,
                       frequencies=[80.0])
    assert two.fds[0] / one.fds[0] == pytest.approx(2.0, rel=1e-12)
    assert four.fds[0] / one.fds[0] == pytest.approx(4.0 ** 3, rel=1e-9)
    assert one.at_duration(20.0).fds[0] == pytest.approx(two.fds[0],
                                                         rel=1e-12)
    assert one.at_duration(20.0).ers[0] == pytest.approx(two.ers[0],
                                                         rel=1e-12)


# ---- spectra as they are read ----------------------------------------

def test_a_coarse_spectrum_is_integrated_as_drawn():
    """Lines 20 Hz apart under a resonator 1 Hz wide (fn 50, Q 50): the
    density is constant across each bin, and the response is the
    integral of |H|^2 under those steps, not |H|^2 sampled at the lines."""
    psd = _flat(30.0, 400.0, spacing=20.0)
    result = psd_spectra(psd, b=4.0, duration=1.0, frequencies=[50.0],
                         q=50.0)
    left, right = psd.bin_bounds()
    values = np.asarray(psd.ordinate[0])
    exact = sum(integrate.quad(lambda f, v=v: pseudo_gain(f, 50.0, 50.0) * v,
                               lo, hi, points=[50.0] if lo < 50 < hi
                               else None, limit=400)[0]
                for lo, hi, v in zip(left, right, values) if v > 0.0)
    assert result.rms[0] ** 2 == pytest.approx(exact, rel=1e-3)
    sampled = np.sum(values * pseudo_gain(psd.abscissa, 50.0, 50.0) * 20.0)
    assert abs(result.rms[0] ** 2 / sampled - 1.0) > 0.1


def test_a_breakpoint_specification_is_read_as_its_power_law():
    """Breakpoints rising at +3 dB/oct, flat, then falling at -6: each
    resonator's mean square against the power law between them."""
    f = np.array([20.0, 80.0, 500.0, 2000.0])
    g = np.array([0.01, 0.04, 0.04, 0.04 / 16.0]) * G ** 2
    spec = Specification(f, g[None], response_dof=['1X+'],
                         ordinate_dim=[fatigue.DENSITY])

    def density(x):
        return float(np.exp(np.interp(np.log(x), np.log(f), np.log(g))))

    natural = [40.0, 300.0, 1000.0]
    result = psd_spectra(spec, b=4.0, duration=1.0, frequencies=natural,
                         q=20.0)
    for fn, rms in zip(natural, result.rms):
        exact, _ = integrate.quad(
            lambda x, fn=fn: pseudo_gain(x, fn, 20.0) * density(x),
            20.0, 2000.0, points=[fn, 80.0, 500.0], limit=800)
        assert rms ** 2 == pytest.approx(exact, rel=1e-3)


def test_a_resonator_at_the_band_edge_sees_part_of_its_resonance():
    """A band that starts at fn, and one that stops there: each gives
    the resonator its own share of the response, the quadrature over
    that band's part of |H|^2, and the two together are Miles'."""
    shares = []
    # the two bands meet at the bin edge of 100.5 Hz, where fn sits
    for psd, lo, hi in ((_flat(101.0, 10000.0), 100.5, 10000.5),
                        (_flat(1.0, 100.0), 0.5, 100.5)):
        result = psd_spectra(psd, b=4.0, duration=1.0,
                             frequencies=[100.5], q=50.0)
        exact, _ = integrate.quad(
            lambda f: pseudo_gain(f, 100.5, 50.0) * LEVEL, lo, hi,
            points=[100.5], limit=800)
        assert result.rms[0] ** 2 == pytest.approx(exact, rel=1e-3)
        shares.append(result.rms[0] ** 2 / _miles(100.5, 50.0) ** 2)
    assert all(0.3 < share < 0.7 for share in shares)
    assert sum(shares) == pytest.approx(1.0, abs=0.02)


def test_a_resonator_far_above_the_band_follows_its_base():
    """Ten times above the band's top a resonator's gain is one to two
    percent, so its rms is the spectrum's and its rate Rice's rate of
    the spectrum, and the result notes it is off the spectrum."""
    psd = _flat(20.0, 200.0)
    result = psd_spectra(psd, b=4.0, duration=60.0,
                         frequencies=[2000.0], q=10.0)
    assert result.rms[0] == pytest.approx(math.sqrt(LEVEL * 180.0),
                                          rel=0.02)
    assert result.crossings[0] == pytest.approx(
        rice_rate(psd.abscissa, psd.ordinate[0]), rel=0.02)
    assert any('outside the spectrum' in note for note in result.notes)


def test_the_default_band_is_the_spectrum_extent():
    result = psd_spectra(_flat(20.0, 2000.0), b=4.0, duration=1.0)
    assert result.frequency[0] == pytest.approx(19.5)
    assert result.frequency[-1] == pytest.approx(2000.5, rel=0.03)
    assert np.allclose(np.diff(np.log2(result.frequency)), 1.0 / 12.0)


# ---- the peak distributions ------------------------------------------

def test_rice_moment_at_its_two_ends():
    """Rayleigh at alpha = 1, 2^(b/2) Gamma(1 + b/2); the positive half
    of a Gaussian at alpha = 0, 2^(b/2) Gamma((b+1)/2) / (2 sqrt pi)."""
    for b in (2.0, 4.0, 7.5):
        assert rice_moment(1.0, b)[0] == pytest.approx(
            2 ** (b / 2) * math.gamma(1 + b / 2), rel=1e-6)
        assert rice_moment(0.0, b)[0] == pytest.approx(
            2 ** (b / 2) * math.gamma((b + 1) / 2) / (2 * math.sqrt(math.pi)),
            rel=1e-6)


def test_rice_and_dirlik_are_the_narrowband_reading_for_a_tone():
    """With m4 = m2^2 / m0 and m1 = sqrt(m0 m2) the response is a
    single frequency: every reading is the narrow-band one."""
    m0, m2 = 3.0, 3.0 * 100.0 ** 2
    m4, m1 = m2 ** 2 / m0, math.sqrt(m0 * m2)
    narrow = narrowband_damage(m0, m2, 6.0, 10.0)
    assert rice_damage(m0, m2, m4, 6.0, 10.0) == pytest.approx(narrow,
                                                               rel=1e-6)
    assert dirlik_damage(m0, m1, m2, m4, 6.0, 10.0)[0] == pytest.approx(
        narrow, rel=1e-9)


def test_a_broad_response_counts_more_peaks_with_rice():
    """A resonator under a band reaching far above it carries small
    peaks: its irregularity is well under one, and counting every peak
    as a cycle gives more damage than the narrow-band reading."""
    wide = _flat(1.0, 10000.0)
    rice = psd_spectra(wide, b=4.0, duration=1.0, frequencies=[100.0],
                       method='rice')
    narrow = psd_spectra(wide, b=4.0, duration=1.0, frequencies=[100.0])
    assert rice.irregularity[0] < 0.5
    assert rice.fds[0] > 1.1 * narrow.fds[0]


# ---- rainflow ---------------------------------------------------------

def _three_point(values):
    """A sequential three-point rainflow count, written apart from the
    module's: a range closes when the next is at least as large, as a
    half cycle while it holds the starting point; the residue counts in
    halves."""
    stack, out = [], []
    for v in turning_points(values):
        stack.append(float(v))
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if len(stack) == 3:
                out.append((y, 0.5))
                stack.pop(0)
            else:
                out.append((y, 1.0))
                last = stack.pop()
                del stack[-2:]
                stack.append(last)
    out.extend((abs(b - a), 0.5) for a, b in itertools.pairwise(stack))
    return out


def test_turning_points_drop_runs_and_continuations():
    assert turning_points([0, 1, 2, 2, 1, -1, -1, 3]).tolist() == [0, 2, -1, 3]
    assert turning_points([]).size == 0
    assert turning_points([5, 5, 5]).tolist() == [5]


def test_rainflow_counts_a_hand_sequence():
    """-2, 1, -3, 5, -1, 3, -4, 4, -2: the inner 5/-1/3 closes -1 to 3
    (range 4); the rest is residue, eight ranges as half cycles."""
    ranges, counts = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    closed = ranges[counts == 1.0]
    assert sorted(closed.tolist()) == [4.0]
    assert sorted(ranges[counts == 0.5].tolist()) == [3, 4, 6, 8, 8, 9]


def test_rainflow_agrees_with_a_sequential_count():
    rng = np.random.default_rng(11)
    for trial in range(60):
        n = int(rng.integers(5, 400))
        x = (rng.standard_normal(n).cumsum() if trial % 2
             else rng.standard_normal(n))
        for b in (2.0, 5.0, 8.0):
            hand = sum(c * (r / 2.0) ** b for r, c in _three_point(x))
            assert rainflow_damage(x, b) == pytest.approx(hand, rel=1e-9)
        _ranges, counts = rainflow(x)
        assert counts.sum() == pytest.approx(
            sum(c for _r, c in _three_point(x)))


def test_rainflow_of_equal_ranges_closes_both():
    """Two neighboring ranges equal: both close, one a pass later, and
    the outer cycle they leave closes too."""
    values = [0, 4, 1, 3, 1, 3, 0, 4]
    ranges, counts = rainflow(values)
    assert sorted(ranges[counts == 1.0].tolist()) == [2.0, 2.0, 4.0]
    # the sequential count takes the outer range as three halves where
    # this takes it as a cycle and a half: the same count, and damage
    hand = _three_point(values)
    assert counts.sum() == pytest.approx(sum(c for _r, c in hand))
    assert rainflow_damage(values, 3.0) == pytest.approx(
        sum(c * (r / 2.0) ** 3 for r, c in hand))


# ---- the recursion ----------------------------------------------------

def test_the_recursion_has_unit_gain_at_zero_and_follows_h():
    from scipy.signal import freqz

    b, a = response_filter([10.0, 100.0, 250.0], 5000.0, 10.0)
    assert np.allclose(b.sum(axis=1) / a.sum(axis=1), 1.0, atol=1e-12)
    for k, fn in enumerate((10.0, 100.0, 250.0)):
        f = np.array([0.3, 1.0, 2.0]) * fn
        _w, h = freqz(b[k], a[k], worN=f, fs=5000.0)
        # a sampled tone joined by straight lines between samples keeps
        # sinc^2(f / fs) of its amplitude (the triangle's transform), so
        # the recursion's power gain is |H|^2 sinc^4; held flat between
        # samples instead, it would be sinc^2 (the images above fs / 2
        # add a few parts in ten thousand)
        assert np.allclose(np.abs(h) ** 2 / pseudo_gain(f, fn, 10.0),
                           np.sinc(f / 5000.0) ** 4, rtol=1e-3)


def test_natural_frequencies_stop_at_a_twentieth_of_the_rate():
    history = _synth(4.0, 2000.0, 10.0, 80.0)
    top = history_spectra(history, b=4.0, frequencies=[100.0])
    assert top.frequency[-1] == 100.0
    assert any('reads its input' in note for note in top.notes)
    with pytest.raises(ValueError, match='above a twentieth of the sample'):
        history_spectra(history, b=4.0, frequencies=[100.5])
    default = history_spectra(history, b=4.0)
    assert default.frequency[-1] <= 100.0
    assert default.frequency[0] == pytest.approx(10.0 / 4.0)


# ---- against simulated records ----------------------------------------

@pytest.fixture(scope='module')
def simulated():
    """Two records flat from 20 to 500 Hz, 100 s each at 10 kHz, read at
    five natural frequencies, b = 4, Q = 10."""
    natural = [30.0, 60.0, 100.0, 200.0, 400.0]
    records = [_synth(100.0, 10000.0, 20.0, 500.0, seed=s) for s in (3, 4)]
    counted = [history_spectra(r, b=4.0, frequencies=natural) for r in records]
    return natural, records, counted


def test_the_spectrum_s_damage_agrees_with_rainflow(simulated):
    """Dirlik's fit within 5 % of rainflow on average, the narrow-band
    reading above it by a few percent and never below Dirlik."""
    natural, _records, counted = simulated
    psd = _flat(20.0, 500.0, spacing=0.5)
    counts = np.mean([c.fds for c in counted], axis=0)
    dirlik = psd_spectra(psd, b=4.0, duration=100.0, frequencies=natural,
                         method='dirlik')
    narrow = psd_spectra(psd, b=4.0, duration=100.0, frequencies=natural)
    assert np.mean(dirlik.fds / counts) == pytest.approx(1.0, abs=0.05)
    assert 1.0 <= np.mean(narrow.fds / counts) <= 1.1
    assert np.all(narrow.fds >= dirlik.fds)


def test_the_rms_rate_and_peak_agree_with_the_record(simulated):
    natural, _records, counted = simulated
    psd = _flat(20.0, 500.0, spacing=0.5)
    read = psd_spectra(psd, b=4.0, duration=100.0, frequencies=natural)
    for c in counted:
        assert np.allclose(read.rms / c.rms, 1.0, atol=0.02)
        assert np.allclose(read.crossings / c.crossings, 1.0, atol=0.02)
        # the leading term of the expected largest peak reads a few
        # percent under what a run reaches
        assert 0.85 <= np.mean(read.ers / c.ers) <= 1.02


def test_under_the_default_window_the_measured_psd_reads_the_same(simulated):
    """The record's own PSD, averaged under the default (Hann) window,
    gives the narrow-band damage within 10 % of its rainflow count."""
    natural, records, counted = simulated
    assert Averaging(frame_length=8192).window == 'hann'
    record = records[0]
    measured = record.compute_psds(Averaging(frame_length=8192, overlap=0.5)
                                   .filled(len(record.abscissa), 10000.0))
    read = psd_spectra(measured, b=4.0, duration=100.0, frequencies=natural)
    assert np.mean(read.fds / counted[0].fds) == pytest.approx(1.0, abs=0.1)


def test_a_record_starting_late_reads_the_same():
    early = _synth(8.0, 4000.0, 20.0, 150.0, seed=8)
    late = _synth(8.0, 4000.0, 20.0, 150.0, seed=8, start=3.25)
    one = history_spectra(early, b=5.0, frequencies=[50.0, 100.0])
    two = history_spectra(late, b=5.0, frequencies=[50.0, 100.0])
    # the rate read off a later abscissa differs in its last digits
    assert np.allclose(one.fds, two.fds, rtol=1e-9, atol=0.0)
    assert np.allclose(one.ers, two.ers, rtol=1e-9, atol=0.0)


def test_a_record_is_leveled_first_and_scaled_by_a_duration():
    """An offset moves neither the FDS nor the ERS; a stated duration
    scales the FDS in proportion and leaves the ERS the record's."""
    history = _synth(8.0, 4000.0, 20.0, 150.0, seed=9)
    shifted = TimeHistory(history.abscissa, history.ordinate + 5.0,
                          response_dof=['1X+'], ordinate_dim=['acceleration'])
    one = history_spectra(history, b=5.0, frequencies=[60.0])
    two = history_spectra(shifted, b=5.0, frequencies=[60.0])
    assert two.fds[0] == pytest.approx(one.fds[0], rel=1e-9)
    assert two.ers[0] == pytest.approx(one.ers[0], rel=1e-9)
    longer = history_spectra(history, b=5.0, frequencies=[60.0],
                             duration=80.0)
    assert longer.fds[0] == pytest.approx(10.0 * one.fds[0], rel=1e-12)
    assert longer.ers[0] == one.ers[0]
    assert longer.ers_duration == pytest.approx(8.0)
    assert any('stationary sample' in note for note in longer.notes)


# ---- refusals ---------------------------------------------------------

def _velocity():
    f = np.arange(0.0, 100.0)
    return Psd(f, np.ones((1, f.size)), response_dof=['1X+'],
               ordinate_dim=['velocity**2/frequency'])


@pytest.mark.parametrize('data,said', [
    (Psd(np.arange(10.0), np.ones((1, 10)), response_dof=['1X+']),
     'not declared'),
    (_velocity(), 'density of velocity'),
    (Psd(np.arange(10.0), np.ones((2, 10)) * (1 + 1j),
         response_dof=['1X+', '2X+'], reference_dof=['2X+', '1X+'],
         ordinate_dim=['acceleration**2/frequency'] * 2), 'cross spectrum'),
])
def test_a_spectrum_that_cannot_be_read_one_way_is_refused(data, said):
    assert said in spectrum_refusal(data, 0)
    with pytest.raises(ValueError, match=said):
        psd_spectra(data, b=4.0, duration=1.0)


def test_a_bad_line_is_refused_with_its_frequency():
    psd = _flat(10.0, 100.0)
    psd.ordinate[0, 50] = np.nan
    with pytest.raises(ValueError, match='not finite, the first at 50 Hz'):
        psd_spectra(psd, b=4.0, duration=1.0)
    psd = _flat(10.0, 100.0)
    psd.ordinate[0, 40] = -1.0
    with pytest.raises(ValueError, match='below zero at 40 Hz'):
        psd_spectra(psd, b=4.0, duration=1.0)
    with pytest.raises(ValueError, match='no area'):
        psd_spectra(_flat(level=0.0), b=4.0, duration=1.0)
    with pytest.raises(ValueError, match='no record 1'):
        psd_spectra(_flat(), 1, b=4.0, duration=1.0)
    real_cross = Psd(np.arange(10.0), np.ones((1, 10)), response_dof=['1X+'],
                     reference_dof=['2X+'], ordinate_dim=[fatigue.DENSITY])
    assert 'cross spectrum' in spectrum_refusal(real_cross)


@pytest.mark.parametrize('settings,said', [
    ({'duration': 1.0}, 'exponent b is not given'),
    ({'b': 40.0, 'duration': 1.0}, 'outside 2 to 25'),
    ({'b': 1.0, 'duration': 1.0}, 'outside 2 to 25'),
    ({'b': 4.0, 'duration': 1.0, 'q': 0.5}, 'Q 0.5 is outside'),
    ({'b': 4.0, 'duration': 1.0, 'q': 1000.0}, 'Q 1000.0 is outside'),
    ({'b': 4.0}, 'duration is not given'),
    ({'b': 4.0, 'duration': 0.0}, 'not positive'),
    ({'b': 4.0, 'duration': float('nan')}, 'not positive'),
    ({'b': 4.0, 'duration': 1.0, 'method': 'peaks'}, 'not one of'),
    ({'b': 4.0, 'duration': 1.0, 'frequencies': [0.0, 10.0]},
     'positive and finite'),
    ({'b': 4.0, 'duration': 1.0, 'frequencies': []}, 'no natural'),
    ({'b': 4.0, 'duration': 1.0, 'low': 50.0, 'high': 20.0}, 'not a band'),
    ({'b': 4.0, 'duration': 1.0, 'low': 50.0, 'high': 50.0}, 'not a band'),
    ({'b': 4.0, 'duration': 1.0, 'low': 0.0, 'high': 50.0}, 'not a band'),
    ({'b': 4.0, 'duration': 1.0, 'low': 10.0, 'high': float('nan')},
     'not a band'),
    ({'b': 4.0, 'duration': 1.0, 'low': 10.0, 'high': float('inf')},
     'not a band'),
    ({'b': 4.0, 'duration': 1.0, 'low': float('-inf'), 'high': 50.0},
     'not a band'),
    ({'b': 4.0, 'duration': 'long'}, 'not a number'),
    ({'b': 'steep', 'duration': 1.0}, 'not a number'),
])
def test_settings_out_of_sense_are_refused(settings, said):
    with pytest.raises(ValueError, match=said):
        psd_spectra(_flat(10.0, 100.0), **settings)


def test_a_record_that_cannot_be_read_one_way_is_refused():
    history = _synth(2.0, 2000.0, 10.0, 50.0)
    nan = TimeHistory(history.abscissa, history.ordinate.copy(),
                      response_dof=['1X+'], ordinate_dim=['acceleration'])
    nan.ordinate[0, 1000] = np.nan
    assert 'not finite, the first at 0.5 s' in history_refusal(nan)
    dead = TimeHistory(history.abscissa, np.zeros_like(history.ordinate),
                       response_dof=['1X+'], ordinate_dim=['acceleration'])
    assert 'does not vary' in history_refusal(dead)
    constant = TimeHistory(history.abscissa, np.full_like(history.ordinate,
                                                          3.0),
                           response_dof=['1X+'], ordinate_dim=['acceleration'])
    assert 'every sample 3' in history_refusal(constant)
    bare = TimeHistory(history.abscissa, history.ordinate,
                       response_dof=['1X+'])
    assert 'not declared' in history_refusal(bare)
    force = TimeHistory(history.abscissa, history.ordinate,
                        response_dof=['1X+'], ordinate_dim=['force'])
    assert 'record of force' in history_refusal(force)
    assert 'no record 1' in history_refusal(history, 1)
    for data in (nan, dead, bare, force):
        with pytest.raises(ValueError):
            history_spectra(data, b=4.0)
    with pytest.raises(ValueError, match='counted by rainflow'):
        fatigue_spectra(history, b=4.0, method='dirlik')
    with pytest.raises(ValueError, match='PSD or a time history'):
        fatigue_spectra(object(), b=4.0)


def test_progress_is_told_and_a_cancel_stops_it():
    told = []
    psd_spectra(_flat(10.0, 100.0), b=4.0, duration=1.0,
                frequencies=np.linspace(10, 100, 40),
                progress=lambda done, total: told.append((done, total)))
    assert told[-1] == (40, 40)

    def cancel(_done, _total):
        raise Cancelled('stop')

    with pytest.raises(Cancelled):
        history_spectra(_synth(2.0, 2000.0, 10.0, 50.0), b=4.0,
                        progress=cancel)


# ---- time compression and equivalence ---------------------------------

def test_time_compression_matches_the_worked_examples():
    """Tenfold at 7.5 is 1.848 on the ASD, +2.67 dB, 1.36 on the rms;
    at 4 it is 3.16, +5.0 dB. At the cap of 4 on the ASD, 7.5 allows
    181 to 1 and 4 allows 16."""
    tenfold = time_compression(b=7.5, time_ratio=10.0)
    assert tenfold.asd_factor == pytest.approx(1.848, abs=5e-4)
    assert tenfold.decibels == pytest.approx(2.67, abs=5e-3)
    assert tenfold.rms_factor == pytest.approx(1.36, abs=5e-3)
    assert not tenfold.over_cap
    assert time_compression(b=4.0, time_ratio=10.0).decibels \
        == pytest.approx(5.0, abs=1e-9)
    assert tenfold.at_cap == pytest.approx(181.02, abs=0.01)
    assert time_compression(b=4.0, factor=4.0).time_ratio == pytest.approx(16)
    assert compression_ratio(2.0, 6.0, on='rms') == pytest.approx(64.0)
    assert exaggeration(64.0, 6.0, on='rms') == pytest.approx(2.0)
    over = time_compression(b=4.0, time_ratio=100.0)
    assert over.over_cap and 'over the cap of 4' in over.describe()
    assert 'within the cap' in tenfold.describe()
    longer = time_compression(b=4.0, factor=0.5)
    assert 'extends time 4 to 1' in longer.describe()
    assert CAP == {'asd': 4.0, 'rms': 2.0}


@pytest.mark.parametrize('kwargs,said', [
    ({'b': 4.0}, 'one of the two'),
    ({'b': 4.0, 'factor': 2.0, 'time_ratio': 2.0}, 'one of the two'),
    ({'b': 4.0, 'factor': 0.0}, 'not positive'),
    ({'b': 4.0, 'time_ratio': -1.0}, 'not positive'),
    ({'factor': 2.0}, 'exponent b is not given'),
    ({'b': 4.0, 'factor': 2.0, 'cap': 0.5}, 'below one'),
])
def test_a_trade_out_of_sense_is_refused(kwargs, said):
    with pytest.raises(ValueError, match=said):
        time_compression(**kwargs)
    with pytest.raises(ValueError, match="neither 'asd'"):
        compression_ratio(2.0, 4.0, on='peak')


def test_a_test_is_its_field_raised_and_shortened():
    """The field's spectrum raised by k on the ASD and run for a tenth
    of the time: its damage ratio is k^(b/2) / 10 at every resonator, its
    exaggeration k everywhere, and the factor it needs at its duration
    is (10 / k^(b/2))^(2/b)."""
    b, k = 7.5, 1.5
    field = psd_spectra(_flat(20.0, 2000.0), b=b, duration=600.0, q=10.0)
    test = psd_spectra(_flat(20.0, 2000.0, level=k * LEVEL), b=b,
                       duration=60.0, q=10.0)
    pair = equivalence(field, test)
    assert np.allclose(pair.damage_ratio, k ** (b / 2) / 10.0, rtol=1e-9)
    assert np.allclose(pair.exaggeration, k, rtol=1e-9)
    assert not pair.covers
    assert pair.factor_needed == pytest.approx(
        (10.0 / k ** (b / 2)) ** (2 / b), rel=1e-9)
    assert pair.duration_needed == pytest.approx(
        60.0 * 10.0 / k ** (b / 2), rel=1e-9)
    assert pair.compression().asd_factor == pytest.approx(k)
    assert pair.time_ratio == pytest.approx(10.0)
    assert np.allclose(pair.ers_db, 10.0 * math.log10(k)
                       + 20.0 * np.log10(test.ers / field.ers
                                         / math.sqrt(k)), atol=1e-9)
    assert 'falls short' in pair.describe()
    hard = equivalence(field, psd_spectra(
        _flat(20.0, 2000.0, level=8 * LEVEL), b=b, duration=60.0, q=10.0))
    assert hard.covers and hard.ers_covers
    assert hard.over_cap.size == hard.frequency.size
    assert 'covers' in hard.describe() and 'over the cap' in hard.describe()


def test_the_governing_frequency_is_where_the_test_is_thinnest():
    field = psd_spectra(_flat(20.0, 2000.0), b=4.0, duration=60.0)
    notched = _flat(20.0, 2000.0, level=2 * LEVEL)
    lines = (notched.abscissa > 300.0) & (notched.abscissa < 400.0)
    notched.ordinate[0, lines] = 0.2 * LEVEL
    pair = equivalence(field, psd_spectra(notched, b=4.0, duration=60.0))
    assert 300.0 < pair.governs < 400.0
    assert not pair.ers_covers


def test_two_results_read_apart_are_refused():
    one = psd_spectra(_flat(), b=4.0, duration=1.0, frequencies=[50.0])
    for other, said in (
            (psd_spectra(_flat(), b=5.0, duration=1.0, frequencies=[50.0]),
             'one Q and b'),
            (psd_spectra(_flat(), b=4.0, duration=1.0, frequencies=[50.0],
                         q=20.0), 'one Q and b'),
            (psd_spectra(_flat(), b=4.0, duration=1.0, frequencies=[60.0]),
             'different natural frequencies')):
        with pytest.raises(ValueError, match=said):
            equivalence(one, other)


def test_a_record_against_a_specification_on_one_set_of_frequencies():
    history = _synth(20.0, 4000.0, 20.0, 150.0, seed=5)
    spec = Specification([20.0, 150.0], [[LEVEL, LEVEL]],
                         response_dof=['1X+'],
                         ordinate_dim=[fatigue.DENSITY])
    pair = compared(history, spec, test_duration=20.0, b=4.0, q=10.0,
                    method='narrowband', low=30.0, high=120.0)
    assert isinstance(pair, Equivalence)
    assert np.array_equal(pair.field.frequency, pair.test.frequency)
    assert pair.field.source == 'time history'
    assert np.allclose(pair.damage_ratio, 1.0, atol=0.35)
    with pytest.raises(ValueError, match='duration is not given'):
        compared(history, spec, b=4.0)


def test_what_a_result_rests_on_is_said():
    result = psd_spectra(_flat(), b=7.5, duration=60.0, method='dirlik',
                         frequencies=[100.0])
    assert isinstance(result, ResponseSpectra)
    assert 'b = 7.5' in result.rests_on and 'Q = 10' in result.rests_on
    assert 'Dirlik' in result.rests_on
    assert 'dirlik' in result.describe() and '1 min' in result.describe()
    assert result.damping == pytest.approx(0.05)
    counted = history_spectra(_synth(4.0, 2000.0, 10.0, 80.0), b=4.0,
                              frequencies=[50.0])
    assert 'rainflow' in counted.rests_on and '4 s' in counted.rests_on
    assert fatigue.settings_of(result) == {
        'b': 7.5, 'q': 10.0, 'method': 'dirlik', 'duration': 60.0}
    assert 'method' not in fatigue.settings_of(counted)


# ---- each reading pinned where a change could hide ---------------------

def _rice_density(u, alpha):
    """Rice's peak density in standard deviations, written apart from
    the module's."""
    from scipy.stats import norm

    spread = math.sqrt(1.0 - alpha ** 2)
    return (spread * norm.pdf(u / spread) / spread * spread
            + alpha * u * math.exp(-u * u / 2.0)
            * norm.cdf(alpha * u / spread))


def test_rice_moment_between_its_ends_matches_quadrature():
    """At an irregularity between the ends, against adaptive quadrature
    of the density; and the density over the whole line is one."""
    for alpha in (0.3, 0.7):
        total, _ = integrate.quad(lambda u, a=alpha: _rice_density(u, a),
                                  -40.0, 40.0, limit=400)
        assert total == pytest.approx(1.0, abs=1e-8)
        for b in (3.0, 7.5):
            exact, _ = integrate.quad(
                lambda u, a=alpha, b=b: u ** b * _rice_density(u, a),
                0.0, 40.0, limit=400)
            assert rice_moment(alpha, b)[0] == pytest.approx(exact,
                                                             rel=1e-6)


def test_rice_damage_counts_at_the_rate_of_peaks():
    """m0 = 1, m2 = 1, m4 = 4: two peaks a second at irregularity one
    half, so the damage over 10 s is 20 times the moment there."""
    damage = rice_damage(1.0, 1.0, 4.0, 5.0, 10.0)
    assert damage == pytest.approx(20.0 * rice_moment(0.5, 5.0)[0],
                                   rel=1e-12)


def _dirlik_by_quadrature(m0, m1, m2, m4, b, seconds):
    """Dirlik's range density integrated numerically against
    (range / 2)^b, its weights written out here."""
    g = m2 / math.sqrt(m0 * m4)
    xm = m1 / m0 * math.sqrt(m2 / m4)
    d1 = 2 * (xm - g ** 2) / (1 + g ** 2)
    r = (g - xm - d1 ** 2) / (1 - g - d1 + d1 ** 2)
    d2 = (1 - g - d1 + d1 ** 2) / (1 - r)
    d3 = 1 - d1 - d2
    qd = 1.25 * (g - d3 - d2 * r) / d1
    sigma = math.sqrt(m0)

    def density(s):
        z = s / (2 * sigma)
        return (d1 / qd * math.exp(-z / qd)
                + d2 * z / r ** 2 * math.exp(-z * z / (2 * r * r))
                + d3 * z * math.exp(-z * z / 2)) / (2 * sigma)

    total, _ = integrate.quad(density, 0.0, 200 * sigma, limit=400)
    assert total == pytest.approx(1.0, abs=1e-6)
    mean, _ = integrate.quad(lambda s: (s / 2) ** b * density(s), 0.0,
                             200 * sigma, limit=400)
    return math.sqrt(m4 / m2) * seconds * mean


def test_dirlik_damage_matches_its_density_integrated():
    for moments in ((2.0, 25.0, 400.0, 2.0e5), (1.0, 90.0, 1.0e4, 2.0e8)):
        for b in (3.0, 6.0):
            assert dirlik_damage(*moments, b, 7.0)[0] == pytest.approx(
                _dirlik_by_quadrature(*moments, b, 7.0), rel=1e-6)


def test_the_peak_rate_and_irregularity_match_quadrature():
    psd = _flat(20.0, 2000.0)
    result = psd_spectra(psd, b=4.0, duration=1.0, frequencies=[150.0])

    def moment(k):
        return integrate.quad(
            lambda f: f ** k * pseudo_gain(f, 150.0, 10.0) * LEVEL, 19.5,
            2000.5, points=[150.0], limit=800)[0]

    assert result.peaks[0] == pytest.approx(math.sqrt(moment(4) / moment(2)),
                                            rel=1e-3)
    assert result.irregularity[0] == pytest.approx(
        moment(2) / math.sqrt(moment(0) * moment(4)), rel=1e-3)


def test_a_spectrum_from_zero_hertz_is_read_from_zero():
    """A level on the 0 Hz line: the grid starts at zero, the default
    band at the first line above it, and the mean square is the
    quadrature over the whole band."""
    psd = _flat(0.0, 400.0, spacing=2.0)
    result = psd_spectra(psd, b=4.0, duration=1.0, frequencies=[30.0])
    exact, _ = integrate.quad(lambda f: pseudo_gain(f, 30.0, 10.0) * LEVEL,
                              0.0, 401.0, points=[30.0], limit=800)
    assert result.rms[0] ** 2 == pytest.approx(exact, rel=1e-3)
    assert psd_spectra(psd, b=4.0, duration=1.0).frequency[0] == 2.0


def test_a_breakpoint_at_zero_hertz_holds_nothing():
    """A power law has no point at 0 Hz: a breakpoint there adds no
    area, and the spectrum reads as it does without it."""
    plain = Specification([20.0, 2000.0], [[LEVEL, LEVEL]],
                          response_dof=['1X+'], ordinate_dim=[fatigue.DENSITY])
    zero = Specification([0.0, 20.0, 2000.0], [[LEVEL, LEVEL, LEVEL]],
                         response_dof=['1X+'], ordinate_dim=[fatigue.DENSITY])
    one = psd_spectra(plain, b=4.0, duration=1.0)
    two = psd_spectra(zero, b=4.0, duration=1.0)
    assert np.array_equal(one.frequency, two.frequency)
    assert np.allclose(one.fds, two.fds, rtol=1e-12)


def test_the_ranges_own_ends_are_accepted():
    for q, b in ((1.0, 2.0), (500.0, 25.0)):
        result = psd_spectra(_flat(10.0, 100.0), b=b, q=q, duration=1.0,
                             frequencies=[50.0])
        assert result.q == q and result.b == b


def test_the_notes_say_where_and_how_much():
    psd = _flat(20.0, 200.0)
    inside = psd_spectra(psd, b=4.0, duration=60.0,
                         frequencies=[19.5, 100.0, 200.5])
    assert not any('outside' in note for note in inside.notes)
    one = psd_spectra(psd, b=4.0, duration=60.0, frequencies=[100.0, 400.0])
    assert ('1 natural frequency lies outside the spectrum (19.5 to 200.5 '
            'Hz)') in one.notes[0]
    two = psd_spectra(psd, b=4.0, duration=60.0, frequencies=[5.0, 400.0])
    assert '2 natural frequencies lie outside' in two.notes[0]
    quiet = psd_spectra(psd, b=4.0, duration=0.001, frequencies=[100.0])
    assert np.isnan(quiet.ers[0])
    assert 'cross zero once or less in 0.001 s' in quiet.notes[0]

    history = _synth(4.0, 2000.0, 10.0, 80.0)
    counted = history_spectra(history, b=4.0, frequencies=[24.0, 25.0,
                                                           100.0])
    assert ('1 resonator(s) below 25 Hz complete fewer than 100 cycles'
            in ' '.join(counted.notes))
    assert ('the top resonator, 100 Hz, reads its input 0.82 % low at '
            'resonance and its peaks up to 1.23 % low between samples'
            ) in counted.notes


def test_a_record_needs_four_samples():
    t = np.arange(4) / 100.0
    four = TimeHistory(t, [[0.0, 1.0, -1.0, 0.5]], response_dof=['1X+'],
                       ordinate_dim=['acceleration'])
    assert history_refusal(four) is None
    three = TimeHistory(t[:3], [[0.0, 1.0, -1.0]], response_dof=['1X+'],
                        ordinate_dim=['acceleration'])
    assert 'holds 3 samples' in history_refusal(three)
    uneven = TimeHistory([0.0, 0.01, 0.03, 0.04], [[0.0, 1.0, -1.0, 0.5]],
                         response_dof=['1X+'], ordinate_dim=['acceleration'])
    assert history_refusal(uneven) is not None


def test_a_count_with_no_closed_cycle_is_its_residue():
    ranges, counts = rainflow([0.0, 1.0, 3.0, -2.0])
    assert ranges.tolist() == [3.0, 5.0]
    assert counts.tolist() == [0.5, 0.5]
    ranges, counts = rainflow([2.0])
    assert ranges.size == 0 and counts.size == 0


def test_the_counted_rates_agree_with_the_spectrum(simulated):
    natural, _records, counted = simulated
    read = psd_spectra(_flat(20.0, 500.0, spacing=0.5), b=4.0,
                       duration=100.0, frequencies=natural)
    for c in counted:
        assert np.allclose(c.peaks / read.peaks, 1.0, atol=0.05)


def test_defaults_read_the_first_record():
    psd = _flat(10.0, 100.0)
    assert spectrum_refusal(psd) is None
    assert fatigue.held_band(psd) == (9.5, 100.5)
    by_default = fatigue_spectra(psd, b=4.0, duration=1.0, frequencies=[50.0])
    assert by_default.label == psd.record_label(0)


def test_words_of_a_result_and_a_trade():
    one = psd_spectra(_flat(), b=4.0, duration=3600.0, frequencies=[100.0])
    assert ', 100 Hz, ' in one.describe() and '1 h' in one.describe()
    many = psd_spectra(_flat(), b=4.0, duration=5400.0,
                       frequencies=[50.0, 100.0])
    assert '50 to 100 Hz' in many.describe() and '1.5 h' in many.describe()
    level = time_compression(b=4.0, factor=1.0)
    assert level.describe().startswith('compresses time 1 to 1')
    at_cap = time_compression(b=4.0, factor=4.0)
    assert not at_cap.over_cap
    assert time_compression(b=4.0, factor=2.0, cap=1.0).over_cap
    with pytest.raises(ValueError, match='not positive'):
        exaggeration(0.0, 4.0)


def _result(fds, ers, frequency=(50.0, 100.0), duration=60.0):
    return ResponseSpectra(
        frequency=np.array(frequency), fds=np.array(fds, dtype=float),
        ers=np.array(ers, dtype=float), rms=np.ones(len(frequency)),
        crossings=np.array(frequency), peaks=np.array(frequency), q=10.0,
        b=4.0, duration=duration, method='narrowband', source='spectrum',
        dimension='acceleration')


def test_the_comparison_at_its_edges():
    """Equal environments cover each other exactly; a test with no damage
    at a resonator where the field has some is judged there and needs an
    infinite factor; one with no field damage is not judged; the cap and
    the ERS are said, singular or plural."""
    same = equivalence(_result([1.0, 2.0], [3.0, 4.0]),
                       _result([1.0, 2.0], [3.0, 4.0]))
    assert same.covers and same.ers_covers
    assert np.allclose(same.damage_db, 0.0)
    assert 'least +0.00 dB' in same.describe()
    assert 'ERS' not in same.describe()

    dead = equivalence(_result([1.0, 2.0], [3.0, 4.0]),
                       _result([0.0, 2.0], [3.0, 4.0]))
    assert not dead.covers and dead.governs == 50.0
    assert dead.factor_needed == math.inf
    assert dead.duration_needed == math.inf
    assert 'does no damage at 50 Hz' in dead.describe()

    unjudged = equivalence(_result([0.0, 2.0], [3.0, 4.0]),
                           _result([1.0, 4.0], [3.0, 4.0]))
    assert unjudged.governs == 100.0
    assert unjudged.least_ratio == pytest.approx(2.0)

    hot = equivalence(_result([1.0, 1.0], [3.0, 4.0]),
                      _result([1.0, 100.0], [2.0, 4.0]))
    assert hot.over_cap.tolist() == [100.0]
    assert 'over the cap of 4 at 1 natural frequency;' in hot.describe()
    assert "its ERS falls below the field's" in hot.describe()
    assert hot.damage_db[1] == pytest.approx(20.0)
    with pytest.raises(ValueError, match='no natural frequency has'):
        equivalence(_result([0.0, 0.0], [1.0, 1.0]),
                    _result([1.0, 1.0], [1.0, 1.0]))


def test_the_shortfall_and_cover_are_said_in_decibels():
    b = 7.5
    field = psd_spectra(_flat(20.0, 2000.0), b=b, duration=600.0)
    short = equivalence(field, psd_spectra(
        _flat(20.0, 2000.0, level=1.5 * LEVEL), b=b, duration=60.0))
    shortfall = -10.0 * (b / 2 * math.log10(1.5) - 1.0)
    assert f'by {shortfall:.2f} dB' in short.describe()
    hard = equivalence(field, psd_spectra(
        _flat(20.0, 2000.0, level=8 * LEVEL), b=b, duration=60.0))
    margin = 10.0 * (b / 2 * math.log10(8.0) - 1.0)
    assert f'least {margin:+.2f} dB' in hard.describe()
    assert 'at 2 natural frequencies' not in hard.describe()


def test_progress_runs_across_both_records_compared():
    told = []
    compared(_flat(20.0, 200.0), _flat(20.0, 200.0, level=2 * LEVEL),
             field_duration=10.0, test_duration=10.0, b=4.0,
             frequencies=np.linspace(20.0, 200.0, 30),
             progress=lambda done, total: told.append((done, total)))
    assert told[0] == (0, 30) and told[-1] == (60, 60)
    dones = [done for done, _total in told]
    assert dones == sorted(dones)
    assert all(total >= 30 for _done, total in told)
