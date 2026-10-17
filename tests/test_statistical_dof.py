"""What an average is worth: the statistical degrees of freedom of an
averaged autospectrum, and the chi-square band they give.

The numbers are checked against hand values that do not come from the
code under test: two per frame with no overlap, Harris's overlap
correlations, the closed form Welch's factor takes for a Hann window at
half overlap, the chi-square quantiles at two degrees of freedom (an
exponential, so a logarithm) and the printed table at ten to a hundred.
Then a synthetic record is averaged and the band is held to its claim:
the true spectrum of white noise lies inside the 90 percent band nine
times in ten, and a band drawn from the nominal count, which ignores
the overlap, does not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.averaging import (
    CONFIDENCE_LEVELS,
    Averaging,
    chi_square_factors,
    confidence_bands,
    frame_correlation,
    statistical_dof,
    window_shape,
)
from visualdynamics.core.data import Psd, TimeHistory

# ---- the degrees of freedom ---------------------------------------------


@pytest.mark.parametrize('window', ['boxcar', 'hann', 'flattop'])
@pytest.mark.parametrize('frames', [1, 2, 7, 40])
def test_frames_that_do_not_overlap_give_two_each(window, frames):
    worth = Averaging(frame_length=512, window=window,
                      frames=frames).statistical_dof()
    assert worth.frames == frames
    assert worth.nominal == 2 * frames
    assert worth.equivalent == pytest.approx(2 * frames, abs=1e-12), (
        'independent frames are worth their full count, whatever the '
        'window')


@pytest.mark.parametrize('length', [64, 1000, 4096])
def test_hann_at_half_overlap_shares_a_sixth(length):
    """Periodic Hann at half overlap: the shared weight is N/16 over a
    total of 3N/8, exactly 1/6 (Harris's 16.7 percent)."""
    shape = window_shape('hann', length)
    assert frame_correlation(shape, length // 2) == pytest.approx(
        1.0 / 6.0, abs=1e-12)


@pytest.mark.parametrize('window, overlap, apart, expected', [
    # Harris (1978), table 1, "overlap correlation"
    ('boxcar', 0.5, 1, 0.500),
    ('boxcar', 0.75, 1, 0.750),
    ('hann', 0.75, 1, 0.659),
    ('hann', 0.75, 2, 0.167),
    ('hamming', 0.5, 1, 0.235),
    ('hamming', 0.75, 1, 0.707),
    ('blackmanharris', 0.5, 1, 0.038),
    ('blackmanharris', 0.75, 1, 0.460),
])
def test_the_correlations_are_harris_s(window, overlap, apart, expected):
    averaging = Averaging(frame_length=2048, overlap=overlap, window=window)
    found = frame_correlation(averaging.shape(), averaging.hop, apart)
    assert found == pytest.approx(expected, abs=0.002)


def test_frames_that_do_not_touch_share_nothing():
    shape = window_shape('hann', 256)
    assert frame_correlation(shape, 128, apart=2) == 0.0
    assert frame_correlation(shape, 300) == 0.0
    assert frame_correlation(shape, 128, apart=0) == pytest.approx(1.0)


@pytest.mark.parametrize('frames', [2, 3, 10, 100])
def test_hann_at_half_overlap_follows_welch_s_closed_form(frames):
    """Only neighbors overlap, so the variance factor is
    1 + 2 (1 - 1/n) (1/6)^2 and the count is 36 n^2 / (19 n - 1)."""
    worth = Averaging(frame_length=1024, overlap=0.5, window='hann',
                      frames=frames).statistical_dof()
    assert worth.nominal == 2 * frames
    assert worth.equivalent == pytest.approx(
        36.0 * frames ** 2 / (19.0 * frames - 1.0), rel=1e-12)


def test_many_half_overlapped_hann_frames_are_worth_eighteen_nineteenths():
    """The textbook ratio for Hann at half overlap, about 0.95, is the
    formula's limit and not a stored constant."""
    worth = Averaging(frame_length=1024, overlap=0.5, window='hann',
                      frames=100000).statistical_dof()
    assert worth.equivalent / worth.nominal == pytest.approx(18 / 19,
                                                             rel=1e-4)


def test_a_rectangle_at_half_overlap_is_discounted_more_than_hann():
    """The rectangle shares half its weight with its neighbor, Hann a
    sixth, so the rectangle's overlapped frames are worth less."""
    rectangle = Averaging(frame_length=1024, overlap=0.5, window='boxcar',
                          frames=20).statistical_dof()
    hann = Averaging(frame_length=1024, overlap=0.5, window='hann',
                     frames=20).statistical_dof()
    assert rectangle.equivalent == pytest.approx(
        40.0 / (1.0 + 2.0 * (19.0 / 20.0) * 0.25), rel=1e-12)
    assert rectangle.equivalent < hann.equivalent < 40.0


def test_every_lag_that_still_overlaps_is_counted():
    """A rectangle at three-quarter overlap shares 3/4, 1/2 and 1/4 with
    the frames one, two and three hops on, all exact, so Welch's factor
    over ten frames is 1 + 2(0.9(9/16) + 0.8(1/4) + 0.7(1/16)) = 2.5."""
    worth = Averaging(frame_length=1024, overlap=0.75, window='boxcar',
                      frames=10).statistical_dof()
    assert worth.equivalent == pytest.approx(20.0 / 2.5, rel=1e-12)


def test_only_lags_inside_the_average_are_counted():
    """Two frames have one pair between them, however far the window
    would reach: a lag past the last frame is not a pair."""
    worth = Averaging(frame_length=1024, overlap=0.75, window='boxcar',
                      frames=2).statistical_dof()
    assert worth.equivalent == pytest.approx(
        4.0 / (1.0 + 2.0 * 0.5 * 0.75 ** 2), rel=1e-12)


def test_a_hop_of_one_sample_is_an_average_too():
    shape = window_shape('hann', 64)
    worth = statistical_dof(3, shape, 1)
    assert worth.frames == 3
    assert 2.0 < worth.equivalent < 6.0


def test_separate_records_add_their_frames_in_full():
    averaging = Averaging(frame_length=1024, overlap=0.75, window='hann',
                          frames=12)
    one = averaging.statistical_dof()
    three = averaging.statistical_dof(records=3)
    assert three.frames == 36
    assert three.nominal == 3 * one.nominal
    assert three.equivalent == pytest.approx(3 * one.equivalent)


def test_the_function_and_the_method_agree():
    averaging = Averaging(frame_length=600, overlap=0.6, window='kaiser',
                          frames=9)
    assert averaging.statistical_dof(records=2) == statistical_dof(
        9, averaging.shape(), averaging.hop, records=2)
    # and both take one record when none is said
    single = statistical_dof(9, averaging.shape(), averaging.hop)
    assert single == averaging.statistical_dof()
    assert single.frames == 9


@pytest.mark.parametrize('frames, hop, records', [(0, 10, 1), (3, 0, 1),
                                                   (3, 10, 0)])
def test_an_average_of_nothing_is_refused(frames, hop, records):
    with pytest.raises(ValueError):
        statistical_dof(frames, window_shape('hann', 64), hop, records)


# ---- the chi-square band -------------------------------------------------


@pytest.mark.parametrize('level', [0.80, 0.90, 0.95, 0.99])
def test_two_degrees_of_freedom_are_an_exponential(level):
    """At two degrees of freedom chi-square is an exponential, whose
    quantile at p is -2 ln(1 - p), so the factors are logarithms."""
    tail = (1.0 - level) / 2.0
    lower, upper = chi_square_factors(2.0, level)
    assert lower == pytest.approx(-1.0 / math.log(tail), rel=1e-10)
    assert upper == pytest.approx(-1.0 / math.log(1.0 - tail), rel=1e-10)


#: chi-square quantiles as printed in the standard tables, at
#: (2.5, 5, 95, 97.5) percent
TABLE = {10: (3.247, 3.940, 18.307, 20.483),
         20: (9.591, 10.851, 31.410, 34.170),
         60: (40.482, 43.188, 79.082, 83.298),
         100: (74.222, 77.929, 124.342, 129.561)}


@pytest.mark.parametrize('dof', sorted(TABLE))
def test_the_factors_match_the_printed_table(dof):
    q025, q05, q95, q975 = TABLE[dof]
    assert chi_square_factors(dof, 0.90) == pytest.approx(
        (dof / q95, dof / q05), rel=2e-4)
    assert chi_square_factors(dof, 0.95) == pytest.approx(
        (dof / q975, dof / q025), rel=2e-4)


def test_a_band_reaches_further_up_than_down():
    """An average of few frames reads low more often than high, so the
    true value's band reaches further above the estimate than below it,
    in decibels as well as in ratio."""
    for dof in (2.0, 9.5, 53.0, 400.0):
        band = confidence_bands([1.0], dof, levels=[0.9])[0]
        assert band.upper_db > -band.lower_db > 0.0
    # and it narrows as the count grows
    widths = [confidence_bands([1.0], dof, levels=[0.9])[0].upper_db
              for dof in (4, 16, 64, 256)]
    assert widths == sorted(widths, reverse=True)


def test_the_band_is_the_estimate_times_its_factors():
    estimate = np.array([[1.0, 2.0, 4.0], [0.5, 0.25, 8.0]])
    bands = confidence_bands(estimate, 30.0)
    assert [band.level for band in bands] == list(CONFIDENCE_LEVELS)
    assert CONFIDENCE_LEVELS == (0.90, 0.95)
    for band in bands:
        low, high = chi_square_factors(30.0, band.level)
        assert (band.lower_factor, band.upper_factor) == (low, high)
        assert band.lower.shape == estimate.shape
        assert np.allclose(band.lower, estimate * low)
        assert np.allclose(band.upper, estimate * high)
        assert band.lower_db == pytest.approx(10 * math.log10(low))
    # the stricter band is the wider
    assert bands[1].lower_factor < bands[0].lower_factor
    assert bands[1].upper_factor > bands[0].upper_factor


@pytest.mark.parametrize('dof, level', [(0.0, 0.9), (-3.0, 0.9),
                                        (10.0, 0.0), (10.0, 1.0),
                                        (10.0, 90.0)])
def test_a_band_out_of_range_is_refused(dof, level):
    with pytest.raises(ValueError):
        chi_square_factors(dof, level)


def test_a_complex_estimate_has_no_chi_square_band():
    with pytest.raises(ValueError, match='complex'):
        confidence_bands(np.array([1.0 + 1.0j]), 10.0)


# ---- carried by the PSD --------------------------------------------------


def _white(frames, overlap, channels=4, length=256, rate=1024.0, seed=15):
    """A history of white noise with unit variance and its averaging."""
    averaging = Averaging(frame_length=length, overlap=overlap,
                          window='hann', frames=frames)
    rng = np.random.default_rng(seed)
    history = TimeHistory(np.arange(averaging.span) / rate,
                          rng.standard_normal((channels, averaging.span)),
                          response_dof=[f'{i + 1}X+' for i in
                                        range(channels)],
                          ordinate_dim='acceleration')
    history.averaging = averaging
    return history, averaging


def test_a_computed_psd_carries_its_count():
    history, averaging = _white(frames=10, overlap=0.5)
    psd = history.compute_psds()
    assert psd.statistical_dof.shape == (psd.num_records,)
    assert np.allclose(psd.statistical_dof,
                       averaging.statistical_dof().equivalent)
    # the averaging passed in, not the history's own, is the one counted
    other = Averaging(frame_length=128, frames=4, window='boxcar')
    assert np.allclose(history.compute_psds(other).statistical_dof, 8.0)


def test_a_capture_saved_one_record_per_average_pools_its_records():
    """Twenty records of one frame each, unwindowed: forty degrees of
    freedom, the same as the twenty averages the table counts."""
    history = visualdynamics.import_file(
        fixture_path('plate', 'modal_spectra.nc4'))['time_data']
    psd = history.compute_psds()
    per_channel = history.records_per_channel
    assert set(per_channel.values()) == {20}
    assert np.allclose(psd.statistical_dof, 40.0)


def test_the_psd_s_band_keeps_its_units_and_records():
    history, _averaging = _white(frames=10, overlap=0.5)
    history.define_units('g')
    psd = history.compute_psds()
    lower, upper = psd.confidence_band(0.95)
    low, high = chi_square_factors(psd.statistical_dof[0], 0.95)
    for edge, factor in ((lower, low), (upper, high)):
        assert type(edge) is Psd
        assert edge.ordinate_unit == psd.ordinate_unit
        assert edge.ordinate_dim == psd.ordinate_dim
        assert edge.response_dof == psd.response_dof
        assert edge.interpolation == 'bin'
        assert np.allclose(edge.ordinate, psd.ordinate * factor)
    # read in g, the band is the spectrum in g times the same factor
    from visualdynamics.units import IN_LBF_S

    shown = psd.display_ordinate(IN_LBF_S)
    assert np.allclose(upper.display_ordinate(IN_LBF_S),
                       shown * high)


def test_a_count_given_overrides_and_may_be_per_record():
    history, _averaging = _white(frames=10, overlap=0.5, channels=2)
    psd = history.compute_psds()
    lower, _upper = psd.confidence_band(0.9, dof=[2.0, 100.0])
    assert np.allclose(lower.ordinate[0],
                       psd.ordinate[0] * chi_square_factors(2.0)[0])
    assert np.allclose(lower.ordinate[1],
                       psd.ordinate[1] * chi_square_factors(100.0)[0])


def test_a_psd_that_does_not_say_how_it_was_averaged_has_no_band():
    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    assert psd.statistical_dof is None
    with pytest.raises(ValueError, match='does not say what it was '
                                         'averaged over'):
        psd.confidence_band()
    # an octave banding sums lines, and its count is not its source's
    history, _averaging = _white(frames=10, overlap=0.5)
    banded = history.compute_psds().to_octave(3)
    assert banded.statistical_dof is None


def test_cross_terms_have_no_chi_square_band():
    history, _averaging = _white(frames=6, overlap=0.5, channels=2)
    cpsd = history.compute_cpsds()
    cpsd.statistical_dof = np.full(cpsd.num_records, 12.0)
    with pytest.raises(ValueError, match='cross terms'):
        cpsd.confidence_band(dof=12.0)


def test_a_cross_term_at_a_drive_point_has_no_chi_square_band():
    """Force against acceleration at one DOF names 1X+ twice and is
    still a cross term: refused, not banded on its real part."""
    lines = np.linspace(10.0, 100.0, 4)
    cross = Psd(lines, np.full((1, 4), 2.0 + 1.0j), response_dof=['1X+'],
                reference_dof=['1X+'],
                ordinate_dim='acceleration*force/frequency')
    with pytest.raises(ValueError, match=r'cross terms \(1X\+/1X\+\)'):
        cross.confidence_band(dof=12.0)
    # the same, with no reference written down
    cross.reference_dof = None
    with pytest.raises(ValueError, match='cross terms'):
        cross.confidence_band(dof=12.0)
    # an autospectrum whose reference is written as its own DOF is one
    auto = Psd(lines, np.full((1, 4), 2.0), response_dof=['1X+'],
               reference_dof=['1X+'],
               ordinate_dim='acceleration**2/frequency')
    lower, _upper = auto.confidence_band(0.9, dof=12.0)
    assert np.allclose(lower.ordinate, 2.0 * chi_square_factors(12.0)[0])


def test_each_edge_says_its_level_and_side():
    history, _averaging = _white(frames=10, overlap=0.5, channels=1)
    lower, upper = history.compute_psds().confidence_band(0.95)
    assert lower.comment == ['95 % lower bound']
    assert upper.comment == ['95 % upper bound']


def test_merged_spectra_keep_their_counts():
    """Stacking leaves each record's average whole, so its count goes
    with it; a source without one leaves the merge without."""
    from visualdynamics.core.merge import merge

    first, _averaging = _white(frames=10, overlap=0.5, channels=2)
    second, _averaging = _white(frames=4, overlap=0.0, channels=1)
    second.response_dof = ['9Z+']
    a, b = first.compute_psds(), second.compute_psds()
    merged = merge([a, b])
    assert np.allclose(merged.statistical_dof,
                       [*a.statistical_dof, 8.0])
    lower, _upper = merged.confidence_band(0.9)
    assert np.allclose(lower.ordinate[2],
                       b.ordinate[0] * chi_square_factors(8.0)[0])
    b.statistical_dof = None
    assert merge([a, b]).statistical_dof is None


def test_deleting_records_keeps_each_count_with_its_channel():
    history, _averaging = _white(frames=10, overlap=0.5, channels=3)
    psd = history.compute_psds()
    psd.statistical_dof = np.array([10.0, 20.0, 30.0])
    psd.delete_records([1])
    assert list(psd.statistical_dof) == [10.0, 30.0]


def test_the_count_survives_a_save(tmp_path):
    history, _averaging = _white(frames=10, overlap=0.75)
    psd = history.compute_psds()
    path = tmp_path / 'psd.vdyn'
    visualdynamics.save(psd, path)
    back = visualdynamics.load(path)
    assert np.allclose(back.statistical_dof, psd.statistical_dof)


# ---- the band keeps its claim -------------------------------------------


def _coverage(level, dof_of, overlap, frames):
    """The share of lines whose true level lies inside the band, for
    white noise of unit variance: its one-sided density is 2/fs on every
    line between zero and Nyquist."""
    history, averaging = _white(frames=frames, overlap=overlap,
                                channels=64)
    psd = history.compute_psds()
    dof = dof_of(psd, averaging)
    lower, upper = psd.confidence_band(level, dof=dof)
    true = 2.0 / history.sample_rate
    inner = slice(1, -1)       # zero and Nyquist carry half the count
    inside = ((lower.ordinate[:, inner] <= true)
              & (true <= upper.ordinate[:, inner]))
    return float(inside.mean())


def _equivalent(psd, _averaging):
    return psd.statistical_dof


def _nominal(_psd, averaging):
    return float(averaging.statistical_dof().nominal)


@pytest.mark.parametrize('overlap, frames', [(0.0, 8), (0.5, 16),
                                             (0.75, 16)])
@pytest.mark.parametrize('level', [0.90, 0.95])
def test_the_true_spectrum_is_inside_the_band_as_often_as_it_says(
        overlap, frames, level):
    """Over some eight thousand lines of averaged white noise, the true
    density falls inside the band the stated share of the time. The
    lines are not all independent (a Hann window leaks into the next
    line), so the share is held to two points, not to its binomial
    spread."""
    share = _coverage(level, _equivalent, overlap, frames)
    assert share == pytest.approx(level, abs=0.02)


def test_the_nominal_count_overstates_an_overlapped_average():
    """At three-quarter overlap the frames share most of their samples:
    a band from 2n, which ignores that, is too narrow and holds the true
    level well under nine times in ten. The equivalent count is what
    makes the band honest."""
    nominal = _coverage(0.90, _nominal, 0.75, 16)
    equivalent = _coverage(0.90, _equivalent, 0.75, 16)
    assert nominal < 0.82
    assert equivalent == pytest.approx(0.90, abs=0.02)
