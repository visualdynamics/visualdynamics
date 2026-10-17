"""The time statistics: a record read in the time domain before its
spectrum.

Pinned against numbers that do not depend on the implementation: the
Gaussian's tails (0.27 % beyond three sigma), Rice's crossing rate of a
flat 20 to 2,000 Hz profile (the square root of (2000^3 - 20^3) over
3 times 1980, about 1,160 a second), the expected largest peak at 1,000
crossings a second (sqrt(2 ln 60,000) and sqrt(2 ln 3,600,000), about
4.7 and 5.5), and a worked example of a clipped record: 12 seconds of
the flat 0.04 g^2/Hz profile at 5,120 samples a second through a 25 g
range reads 8.83 grms, a crest factor of 2.83, a kurtosis of 2.86 and
287 samples, 0.47 %, on the rail. That record passes a kurtosis band of
2 to 4, and the rail count and the crest factor are what flag it. And
clean records that are not broadband or sampled near their band, a
single 2 % damped mode at 100 Hz and a 20 to 200 Hz band at 20,480
samples a second, read inside every band, seed after seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.kurtosis import HIGH, LOW, channel_kurtosis
from visualdynamics.core.time_statistics import (
    FRAME_PRODUCT,
    KURTOSIS_BAND,
    MIN_EXPECTED,
    STATISTICS,
    expected_largest_peak,
    gaussian_share,
    rice_rate,
    statistic_name,
    statistical_bandwidth,
    time_statistics,
)

G = 9.80665
RATE = 5120.0


def flat_random(seconds, rate, seed, low=20.0, high=2000.0, level=0.04):
    """A Gaussian record in g whose one-sided ASD is `level` g^2/Hz from
    `low` to `high`: white noise shaped line by line in frequency."""
    n = round(seconds * rate)
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    # white noise of unit variance has a one-sided ASD of 2/rate
    gain = np.where((f >= low) & (f <= high),
                    np.sqrt(level / (2.0 / rate)), 0.0)
    return np.fft.irfft(spectrum * gain, n)


def mode(seconds, rate, seed, frequency=100.0, damping=0.02):
    """The response of one mode to white noise: a narrow-band Gaussian
    record, its statistical bandwidth 2 pi zeta fn, 12.6 Hz."""
    n = round(seconds * rate)
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    r = np.fft.rfftfreq(n, 1.0 / rate) / frequency
    return np.fft.irfft(spectrum / (1.0 - r ** 2 + 2j * damping * r), n)


@pytest.fixture(scope='module')
def clean():
    """The worked example's record: 8.88 grms, its largest peak 37 g."""
    return flat_random(12.0, RATE, 1403)


@pytest.fixture(scope='module')
def minute():
    """A minute of the same profile sampled at 20,480 a second, fast
    enough that the crossings are not missed between samples."""
    return flat_random(60.0, 20480.0, 7)


# ---- the Gaussian's own numbers ---------------------------------------

def test_the_gaussian_puts_0_27_percent_beyond_three_sigma():
    assert 100 * gaussian_share(3) == pytest.approx(0.27, abs=0.0005)
    assert 100 * gaussian_share(4) == pytest.approx(0.0063, abs=0.00005)
    assert 100 * gaussian_share(5) == pytest.approx(0.000057, abs=5e-7)


def test_the_expected_largest_peak_is_4_7_in_a_minute_and_5_5_in_an_hour():
    assert expected_largest_peak(1000.0, 60.0) == pytest.approx(
        np.sqrt(2 * np.log(60000.0)))
    assert round(expected_largest_peak(1000.0, 60.0), 1) == 4.7
    assert round(expected_largest_peak(1000.0, 3600.0), 1) == 5.5


def test_a_run_of_one_crossing_has_no_expected_peak():
    assert np.isnan(expected_largest_peak(1.0, 1.0))
    assert np.isnan(expected_largest_peak(0.0, 60.0))


def test_a_flat_profile_crosses_about_1160_times_a_second():
    f = np.linspace(20.0, 2000.0, 100001)
    rate = rice_rate(f, np.full_like(f, 0.04))
    assert rate == pytest.approx(np.sqrt((2000.0**3 - 20.0**3)
                                         / (3 * 1980.0)), rel=1e-6)
    assert round(rate, -1) == 1160
    # the level cancels: the rate is a shape, not an amount
    assert rice_rate(f, np.full_like(f, 7.0)) == pytest.approx(rate)


def test_a_spectrum_with_no_area_has_no_rate():
    assert np.isnan(rice_rate([10.0, 20.0], [0.0, 0.0]))
    assert np.isnan(rice_rate([10.0], [1.0]))


# ---- a Gaussian record -------------------------------------------------

def test_a_gaussian_record_reads_as_the_gaussian_does(minute):
    stats = time_statistics(minute, 20480.0)
    beyond_3 = stats.tails[0]
    assert beyond_3.sigmas == 3.0 and beyond_3.judged
    assert 100 * beyond_3.share == pytest.approx(0.27, abs=0.02)
    assert not beyond_3.flagged
    assert stats.kurtosis == pytest.approx(3.0, abs=0.03)
    assert stats.skewness == pytest.approx(0.0, abs=0.03)
    assert stats.flagged == []
    assert stats.describe() == 'Every statistic is inside its band.'


def test_the_crossings_counted_agree_with_the_spectrum(minute):
    """Sampled at ten times its band, the record's counted crossings
    and its spectrum's agree with the profile's 1,160 to within a
    percent."""
    stats = time_statistics(minute, 20480.0)
    assert stats.spectral_crossings == pytest.approx(1160.5, rel=0.01)
    assert stats.crossings == pytest.approx(stats.spectral_crossings,
                                            rel=0.01)


def test_counting_misses_crossings_on_a_record_sampled_near_its_band(clean):
    """At 2.56 times the band two crossings can fall between samples,
    and the count runs about 8 % low; the spectrum's rate does not."""
    stats = time_statistics(clean, RATE)
    assert stats.spectral_crossings == pytest.approx(1160.5, rel=0.02)
    assert stats.crossings < 0.95 * stats.spectral_crossings


def test_the_expected_peak_is_for_the_duration_stated(minute):
    stats = time_statistics(minute, 20480.0, duration=3600.0)
    assert stats.duration == 3600.0
    assert stats.expected_peak == pytest.approx(
        expected_largest_peak(stats.spectral_crossings, 3600.0))
    assert round(stats.expected_peak, 1) == 5.5
    # and over the record's own minute, beside the largest it reached
    assert round(stats.expected_peak_here, 1) == 4.7
    assert 4.0 < stats.peak_sigmas < 6.0


def test_the_duration_is_the_records_own_length_by_default(clean):
    stats = time_statistics(clean, RATE)
    assert stats.duration == pytest.approx(12.0)
    assert stats.record_length == pytest.approx(12.0)


# ---- the worked example: clean, then through a 25 g range --------------

def test_the_clean_record_has_one_sample_at_its_peak(clean):
    stats = time_statistics(clean, RATE)
    assert round(stats.rms, 2) == 8.88
    assert round(stats.crest, 2) == 4.17
    assert round(stats.kurtosis, 2) == 2.99
    assert stats.at_peak == 1
    assert stats.flagged == []


def test_the_clipped_record_reads_the_worked_examples_numbers(clean):
    stats = time_statistics(np.clip(clean, -25.0, 25.0), RATE,
                            full_scale=25.0)
    assert round(stats.rms, 2) == 8.83
    assert abs(stats.peak) == 25.0
    assert round(stats.crest, 2) == 2.83
    assert round(stats.kurtosis, 2) == 2.86
    assert stats.rail == 287
    assert round(100 * stats.rail_share, 2) == 0.47
    assert stats.headroom == pytest.approx(1.0)


def test_the_clipped_record_is_flagged_by_the_rail_and_the_crest_factor(
        clean):
    """With the default bands its kurtosis passes, and it is still
    flagged: the two numbers that say clipped are the samples on the
    rail and a crest factor under three."""
    stats = time_statistics(np.clip(clean, -25.0, 25.0), RATE,
                            full_scale=25.0)
    flags = stats.flags()
    assert flags['kurtosis'] is False
    assert flags['rail'] is True
    assert flags['crest'] is True
    assert 'Samples on the rail' in stats.describe()
    assert 'Crest factor' in stats.describe()
    assert 'Kurtosis' not in stats.describe()


def test_the_kurtosis_band_is_the_bars_by_default_and_settable_tighter(
        clean):
    """2 to 4 by default, so a record at 2.86 is not flagged by its
    kurtosis; set to 2.9 to 3.1 it is, and the clean record is not."""
    assert KURTOSIS_BAND == (LOW, HIGH) == (2.0, 4.0)
    clipped = np.clip(clean, -25.0, 25.0)
    assert time_statistics(clipped, RATE).flags()['kurtosis'] is False
    tight = (2.9, 3.1)
    assert time_statistics(clipped, RATE,
                           kurtosis_band=tight).flags()['kurtosis'] is True
    assert time_statistics(clean, RATE,
                           kurtosis_band=tight).flags()['kurtosis'] is False


def test_a_clip_below_three_sigma_empties_the_three_sigma_tail(clean):
    stats = time_statistics(np.clip(clean, -25.0, 25.0), RATE)
    beyond_3 = stats.tails[0]
    assert beyond_3.count == 0 and beyond_3.judged and beyond_3.flagged


def test_without_a_range_the_rail_is_not_judged(clean):
    stats = time_statistics(np.clip(clean, -25.0, 25.0), RATE)
    assert stats.rail is None and stats.rail_share is None
    assert stats.headroom is None
    assert 'rail' not in stats.flags()
    # the excursions at its own peak still say it: 287 samples on 278
    # flat tops
    assert stats.at_peak == 278 and stats.flags()['at_peak']


def test_the_margin_counts_samples_a_code_inside_the_range(clean):
    clipped = np.clip(clean, -24.99, 24.99)
    assert time_statistics(clipped, RATE, full_scale=25.0,
                           margin=0.0).rail == 0
    assert time_statistics(clipped, RATE, full_scale=25.0,
                           margin=0.001).rail == 287


# ---- the other faults a look at the trace finds ------------------------

def test_a_dropout_stands_out_frame_by_frame(clean):
    """A tenth of a second at zero barely moves the record's kurtosis
    and stands far outside 2 to 4 in the frame it falls in."""
    spoiled = clean.copy()
    first = round(5.05 * RATE)
    spoiled[first:first + round(0.1 * RATE)] = 0.0
    stats = time_statistics(spoiled, RATE)
    assert stats.kurtosis == pytest.approx(3.0, abs=0.08)
    time, value = stats.worst_frame
    assert value > 5.0
    assert 5.0 <= time <= 5.2
    assert stats.frames_outside >= 1 and stats.flags()['frames']
    assert time_statistics(clean, RATE).frames_outside == 0


def test_a_dropout_of_whole_frames_is_outside_its_band(clean):
    """A second at zero, frame-aligned: five frames with no variance and
    so no kurtosis, each outside its band, while the record's kurtosis
    moves only to 3.27."""
    spoiled = clean.copy()
    spoiled[25 * 1024:30 * 1024] = 0.0
    stats = time_statistics(spoiled, RATE)
    assert stats.kurtosis == pytest.approx(3.27, abs=0.005)
    assert np.count_nonzero(np.isnan(stats.frame_kurtosis)) == 5
    assert stats.frames_outside == 5
    time, value = stats.worst_frame
    assert time == pytest.approx(5.1) and np.isnan(value)
    assert stats.flagged == ['frames']


def test_samples_that_are_not_finite_are_counted_and_flagged(clean):
    spoiled = clean.copy()
    spoiled[[100, 2000, 30000]] = [np.nan, np.inf, -np.inf]
    stats = time_statistics(spoiled, RATE)
    assert stats.nonfinite == 3 and stats.samples == clean.size - 3
    assert stats.kurtosis == pytest.approx(time_statistics(
        np.delete(clean, [100, 2000, 30000]), RATE).kurtosis)
    assert stats.flagged == ['nonfinite']
    assert 'Samples not finite' in stats.describe()
    assert time_statistics(clean, RATE).nonfinite == 0


def test_five_spikes_lift_the_crest_factor_past_six(clean):
    spiked = clean.copy()
    spiked[np.linspace(5000, 55000, 5).astype(int)] = 80.0
    stats = time_statistics(spiked, RATE)
    assert stats.crest == pytest.approx(80.0 / stats.rms)
    assert stats.flags()['crest'] and stats.crest > 6.0
    assert stats.at_peak == 5 and stats.flags()['at_peak']


def test_an_offset_is_read_by_the_mean_alone(clean):
    shifted = time_statistics(clean + 0.5, RATE)
    plain = time_statistics(clean, RATE)
    assert shifted.mean == pytest.approx(plain.mean + 0.5)
    assert shifted.std == pytest.approx(plain.std)
    assert shifted.kurtosis == pytest.approx(plain.kurtosis)
    assert shifted.rms / plain.rms - 1.0 < 0.002
    assert shifted.flagged == ['mean']


def test_the_peak_is_signed_and_placed_on_the_records_clock():
    x = np.zeros(1000)
    x[::7] = 1.0
    x[::7] *= np.where(np.arange(0, 1000, 7) % 2, 1.0, -1.0)
    x[640] = -9.0
    stats = time_statistics(x, 100.0, start=2.0)
    assert stats.peak == -9.0
    assert stats.peak_time == pytest.approx(2.0 + 6.40)


# ---- hand values ----------------------------------------------------------

def test_the_skewness_of_a_lopsided_record_is_its_third_moment():
    """[0, 0, 0, 1]: mean 1/4, second moment 3/16, third 3/32, so the
    skewness is (3/32) / (3/16)^1.5 = 2/sqrt(3)."""
    stats = time_statistics([0.0, 0.0, 0.0, 1.0], 4.0, frame=0.5)
    assert stats.skewness == pytest.approx(2.0 / np.sqrt(3.0))
    assert stats.kurtosis == pytest.approx((3 * (1 / 4) ** 4 + (3 / 4) ** 4)
                                           / 4 / (3 / 16) ** 2)


def sine(frequency=10.0, rate=1000.0, seconds=1.0, phase=0.3):
    t = np.arange(round(seconds * rate)) / rate
    return np.sin(2 * np.pi * frequency * t + phase)


def test_a_sine_crosses_its_mean_once_a_cycle_whatever_its_offset():
    plain = time_statistics(sine(), 1000.0)
    shifted = time_statistics(sine() + 5.0, 1000.0)
    assert plain.crossings == 10.0
    assert shifted.crossings == 10.0
    # and the spectrum's rate is the tone's, the offset taken out first
    assert plain.spectral_crossings == pytest.approx(10.0, rel=1e-6)
    assert shifted.spectral_crossings == pytest.approx(10.0, rel=1e-6)


def test_the_spectral_rate_weights_each_line_by_its_power():
    """Two tones, 100 Hz at amplitude 1 and 1,000 Hz at 0.1: power 1
    and 0.01, so the rate is sqrt((1e4 + 0.01 * 1e6) / 1.01)."""
    record = sine(100.0, 10000.0, 1.0, 0.0) + 0.1 * sine(1000.0, 10000.0,
                                                         1.0, 0.0)
    stats = time_statistics(record, 10000.0)
    assert stats.spectral_crossings == pytest.approx(
        np.sqrt((1e4 + 0.01 * 1e6) / 1.01), rel=1e-6)


def test_the_largest_excursion_is_counted_from_the_mean():
    """A unit sine sits 1 from its mean whatever the offset, and its
    standard deviation is 1/sqrt(2)."""
    stats = time_statistics(sine() + 5.0, 1000.0)
    # sampled, the top sits a hair under the crest
    assert stats.peak_sigmas == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert stats.crest == pytest.approx(abs(stats.peak) / stats.rms)


def test_a_peak_sampled_several_times_over_is_one_excursion():
    """A 20 to 200 Hz record at 20,480 a second puts three samples
    within the margin of its one largest peak; that is one excursion,
    and not flagged."""
    x = flat_random(30.0, 20480.0, 1, 20.0, 200.0)
    top = np.abs(x).max()
    assert np.count_nonzero(np.abs(x) >= 0.999 * top) == 3
    stats = time_statistics(x, 20480.0)
    assert stats.at_peak == 1 and stats.flags()['at_peak'] is False


def test_two_samples_at_the_peak_pass_and_three_are_flagged():
    x = 0.5 * sine(7.0, 1000.0, 1.0)
    x[[100, 300]] = 3.0
    assert time_statistics(x, 1000.0).flags()['at_peak'] is False
    x[500] = -3.0
    assert time_statistics(x, 1000.0).flags()['at_peak'] is True


def test_a_range_never_reached_is_not_flagged(clean):
    stats = time_statistics(clean, RATE, full_scale=50.0)
    assert stats.rail == 0 and stats.flags()['rail'] is False


def test_one_sample_on_the_rail_is_flagged(clean):
    stats = time_statistics(clean, RATE, margin=0.0,
                            full_scale=0.9999 * np.abs(clean).max())
    assert stats.rail == 1 and stats.flags()['rail'] is True


def test_a_sample_exactly_at_the_range_is_on_the_rail(clean):
    assert time_statistics(np.clip(clean, -25.0, 25.0), RATE,
                           full_scale=25.0, margin=0.0).rail >= 285


def test_a_small_offset_passes_and_a_large_one_is_flagged(clean):
    """The band is a share of the standard deviation, whatever the
    record's units."""
    scaled = 100.0 * (clean - clean.mean())
    std = scaled.std()
    assert time_statistics(scaled + 0.01 * std, RATE).flags()['mean'] \
        is False
    assert time_statistics(scaled + 0.03 * std, RATE).flags()['mean'] \
        is True


def test_the_worst_frame_is_the_furthest_from_three_either_way(clean):
    """A frame clipped hard reads well under 3, further from it than
    any clean frame reads above."""
    spoiled = clean.copy()
    first = round(7.0 * RATE)
    spoiled[first:first + 1024] = np.clip(spoiled[first:first + 1024],
                                          -4.0, 4.0)
    time, value = time_statistics(spoiled, RATE).worst_frame
    assert value < 2.2
    assert time == pytest.approx((first + 512) / RATE)


def test_a_dead_channel_is_not_gaussian():
    stats = time_statistics(np.zeros(4096), RATE)
    assert np.isnan(stats.kurtosis) and np.isnan(stats.crest)
    assert np.isnan(stats.peak_sigmas)
    assert np.isnan(stats.spectral_crossings)
    assert [tail.count for tail in stats.tails] == [0, 0, 0]
    flags = stats.flags()
    assert flags['kurtosis'] and flags['crest'] and not flags['mean']


def test_the_tails_are_counted_beyond_3_4_and_5_sigma(clean):
    stats = time_statistics(clean, RATE)
    assert [tail.sigmas for tail in stats.tails] == [3.0, 4.0, 5.0]
    assert [tail.expected for tail in stats.tails] == pytest.approx(
        [gaussian_share(3), gaussian_share(4), gaussian_share(5)])


def test_the_frame_is_kept_in_samples(clean):
    assert time_statistics(clean, RATE).frame_length == 1024
    assert time_statistics(clean, RATE, frame=0.5).frame_length == 2560
    # two samples is the shortest frame with a kurtosis
    assert time_statistics(clean, RATE, frame=2 / RATE).frame_length == 2


def test_two_samples_are_a_record():
    assert time_statistics([1.0, -1.0], 2.0, frame=1.0).samples == 2


def test_a_rate_from_two_lines_is_the_trapezoids():
    """Flat between 10 and 20 Hz on two lines: m0 = 10, m2 = 2500."""
    assert rice_rate([10.0, 20.0], [1.0, 1.0]) == pytest.approx(
        np.sqrt(250.0))


# ---- bands, spans and refusals -----------------------------------------

def test_a_tail_a_gaussian_barely_reaches_is_not_judged(clean):
    stats = time_statistics(clean, RATE)
    for tail in stats.tails:
        assert tail.independent == pytest.approx(
            tail.expected * 2.0 * stats.bandwidth * stats.record_length)
        assert tail.judged == (tail.independent >= MIN_EXPECTED)
    assert [round(tail.independent, 1) for tail in stats.tails] == [
        128.9, 3.0, 0.0]
    assert not stats.tails[2].judged and not stats.tails[2].flagged
    assert 'beyond_5' not in stats.flags()


def test_a_tail_is_judged_by_independent_values_not_samples():
    """A Gaussian puts 166 of a 12 s, 100 Hz mode's samples beyond
    three sigma, and they come from under one independent value: the
    tail is not judged."""
    stats = time_statistics(mode(12.0, RATE, 0), RATE)
    beyond_3 = stats.tails[0]
    assert round(beyond_3.expected * stats.samples) == 166
    assert beyond_3.independent == pytest.approx(0.75, abs=0.01)
    assert not beyond_3.judged and 'beyond_3' not in stats.flags()


def test_a_flat_band_is_its_own_width_and_a_mode_two_pi_zeta_fn():
    assert statistical_bandwidth([20.0, 2000.0], [1.0, 1.0]) == \
        pytest.approx(1980.0)
    f = np.linspace(0.0, 1000.0, 200001)
    r = f / 100.0
    square = 1.0 / np.abs(1.0 - r ** 2 + 0.04j * r) ** 2
    assert statistical_bandwidth(f, square) == pytest.approx(
        2 * np.pi * 0.02 * 100.0, rel=0.01)
    assert np.isnan(statistical_bandwidth([1.0, 2.0], [0.0, 0.0]))
    assert np.isnan(statistical_bandwidth([1.0], [1.0]))


def test_a_records_bandwidth_is_read_from_its_own_spectrum(clean):
    assert time_statistics(clean, RATE).bandwidth == pytest.approx(
        1980.0, rel=0.01)
    # the periodogram of two minutes of one mode puts about 1,500 lines
    # in its band, and the estimate scatters by about a tenth
    widths = np.array([
        time_statistics(mode(120.0, 2048.0, seed), 2048.0).bandwidth
        for seed in range(5)])
    assert np.all(np.abs(widths / 12.57 - 1.0) < 0.15)
    assert np.mean(widths) == pytest.approx(12.57, rel=0.05)


def test_the_frame_is_sized_from_the_bandwidth():
    """At least 0.2 s, and long enough to hold `FRAME_PRODUCT`
    bandwidth-seconds: about 1.9 s for a 20 to 200 Hz record, and
    longer than a 12 s record of one 100 Hz mode, which then has no
    frames to judge."""
    banded = time_statistics(flat_random(30.0, 20480.0, 0, 20.0, 200.0),
                             20480.0)
    assert banded.frame_length == round(FRAME_PRODUCT / banded.bandwidth
                                        * 20480.0)
    assert banded.frame_length / 20480.0 == pytest.approx(1.94, abs=0.01)
    narrow = time_statistics(mode(12.0, RATE, 0), RATE)
    assert narrow.frame_kurtosis.size == 0 and narrow.worst_frame is None
    assert 'frames' not in narrow.flags()
    # a frame given is the frame used
    assert time_statistics(mode(12.0, RATE, 0), RATE,
                           frame=0.5).frame_length == 2560


@pytest.mark.parametrize('seed', range(5))
def test_a_clean_narrow_band_record_reads_inside_every_band(seed):
    """A single mode, 100 Hz and 2 % damped, over 12 s and a minute,
    and a 20 to 200 Hz band sampled at 20,480 a second: Gaussian
    records whose neighboring samples are far from independent."""
    for values, rate in ((mode(12.0, RATE, seed), RATE),
                         (mode(60.0, 2048.0, seed), 2048.0),
                         (flat_random(30.0, 20480.0, seed, 20.0, 200.0),
                          20480.0)):
        stats = time_statistics(values, rate)
        assert stats.describe() == 'Every statistic is inside its band.'


def test_every_flag_has_a_name():
    assert set(STATISTICS) >= {'nonfinite', 'mean', 'skewness',
                               'kurtosis', 'crest', 'at_peak', 'rail',
                               'frames'}
    assert statistic_name('beyond_3.5') == 'Beyond 3.5σ'
    with pytest.raises(KeyError):
        statistic_name('nothing')


def test_spans_are_read_together_and_frames_tile_each(clean):
    spans = [(0, 20480), (40960, 61440)]
    stats = time_statistics(clean, RATE, spans=spans, start=1.0)
    together = time_statistics(np.concatenate([clean[:20480],
                                               clean[40960:]]), RATE)
    assert stats.samples == together.samples == 40960
    assert stats.kurtosis == pytest.approx(together.kurtosis)
    # 0.2 s frames of 1,024 samples: twenty in each four-second span,
    # each placed at its middle on the record's clock
    assert stats.frame_kurtosis.size == 40
    assert stats.frame_times[0] == pytest.approx(1.0 + 512 / RATE)
    assert stats.frame_times[20] == pytest.approx(1.0 + (40960 + 512)
                                                  / RATE)


@pytest.mark.parametrize('options,match', [
    ({'kurtosis_band': (3.1, 2.9)}, 'kurtosis band'),
    ({'crest_band': (6.0, 6.0)}, 'crest factor band'),
    ({'frame': 0.0001}, 'holds 1 sample'),
    ({'full_scale': 0.0}, 'range'),
    ({'margin': 1.0}, 'margin'),
])
def test_a_setting_that_says_nothing_is_refused(clean, options, match):
    with pytest.raises(ValueError, match=match):
        time_statistics(clean, RATE, **options)


def test_a_record_needs_two_samples_and_a_rate():
    with pytest.raises(ValueError, match='two finite'):
        time_statistics([1.0], 100.0)
    with pytest.raises(ValueError, match='sample rate'):
        time_statistics([1.0, 2.0], 0.0)


# ---- on a TimeHistory ---------------------------------------------------

def history(*rows):
    rows = np.vstack(rows) * G
    return TimeHistory(np.arange(rows.shape[1]) / RATE, rows,
                       response_dof=[f'{k + 1}Z+' for k in range(len(rows))],
                       ordinate_dim=['acceleration'] * len(rows),
                       ordinate_unit=['g'] * len(rows))


def test_a_records_kurtosis_is_the_kurtosis_bars(clean):
    record = history(clean, np.clip(clean, -25.0, 25.0))
    bars = dict(channel_kurtosis(record))
    for k, dof in enumerate(('1Z+', '2Z+')):
        assert record.time_statistics(k).kurtosis == bars[dof]


def test_a_records_values_and_range_are_in_si(clean):
    record = history(np.clip(clean, -25.0, 25.0))
    stats = record.time_statistics(0, full_scale=25.0 * G)
    assert stats.rms / G == pytest.approx(8.83, abs=0.005)
    assert stats.rail == 287


def test_a_record_with_frames_is_read_over_them(clean):
    from visualdynamics.core.averaging import Averaging

    record = history(clean)
    record.averaging = Averaging(frame_length=1024, frames=20)
    stats = record.time_statistics()
    assert stats.samples == 20 * 1024
    assert stats.kurtosis == dict(channel_kurtosis(record))['1Z+']


def test_a_record_is_read_on_its_own_clock_from_its_first_channel(clean):
    from visualdynamics.core.time_statistics import record_statistics

    record = history(clean, 2.0 * clean)
    late = TimeHistory(record.abscissa + 3.0, record.ordinate,
                       response_dof=['1Z+', '2Z+'],
                       ordinate_dim=['acceleration'] * 2)
    first = record_statistics(late)
    assert first.rms == pytest.approx(record_statistics(record, 0).rms)
    assert first.peak_time == pytest.approx(
        3.0 + np.argmax(np.abs(clean)) / RATE)


def test_a_channel_the_record_does_not_have_is_refused(clean):
    with pytest.raises(ValueError, match='no channel 2'):
        history(clean).time_statistics(2)
    with pytest.raises(ValueError, match='no channel 1'):
        history(clean).time_statistics(1)
    with pytest.raises(ValueError, match='no channel -1'):
        history(clean).time_statistics(-1)


# ---- the view -----------------------------------------------------------

def _titles(layout):
    import pyqtgraph as pg

    return [item.titleLabel.text for item in layout.ci.items
            if isinstance(item, pg.PlotItem)]


def _plots(layout):
    import pyqtgraph as pg

    return [item for item in layout.ci.items
            if isinstance(item, pg.PlotItem)]


def test_the_histogram_is_a_density():
    from visualdynamics.plot.time_statistics import histogram

    values = np.random.default_rng(3).standard_normal(20000)
    edges, density = histogram(values)
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)
    assert edges[0] == values.min() and edges[-1] == values.max()


def test_the_percentages_are_written_out():
    from visualdynamics.plot.time_statistics import percent

    assert percent(gaussian_share(3)) == '0.27 %'
    assert percent(gaussian_share(5)) == '0.000057 %'
    assert percent(0.0) == '0 %'


def test_the_view_states_the_verdict_and_marks_the_rail(qt_app, clean):
    import pyqtgraph as pg

    from visualdynamics.plot.time_statistics import build_time_statistics
    from visualdynamics.units import SI

    record = history(np.clip(clean, -25.0, 25.0))
    stats = record.time_statistics(0, full_scale=25.0 * G,
                                   kurtosis_band=(2.0, 4.0))
    layout = pg.GraphicsLayoutWidget()
    build_time_statistics(layout, record, stats, 0,
                          unit_system=SI.with_units(acceleration='g'))
    titles = _titles(layout)
    assert len(titles) == 3
    assert 'Samples on the rail' in titles[0]
    assert 'Crest factor' in titles[0]
    assert 'beyond 3σ 0 % (flagged)' in titles[1]
    assert '0 of 60 frames outside 2 to 4' in titles[2]
    trace, bell, _frames = _plots(layout)
    # every sample on the rail is a cross on the trace, in g
    crosses = [item for item in trace.listDataItems()
               if item.opts.get('symbol') == 'x']
    assert len(crosses) == 1
    _x, levels = crosses[0].getData()
    assert levels.size == 287
    # on the rail is within the margin of it, a tenth of a percent
    assert np.all(np.abs(levels) >= 0.999 * 25.0)
    assert np.all(np.abs(levels) <= 25.0)
    # the range across the trace, either way, in g
    rails = sorted(item.value() for item in trace.items
                   if isinstance(item, pg.InfiniteLine))
    assert rails == pytest.approx([-25.0, 25.0])
    # the Gaussian is dashed beside the solid histogram
    dashed = [item for item in bell.listDataItems()
              if item.opts['pen'].style() == pg.QtCore.Qt.PenStyle.DashLine]
    assert len(dashed) == 1
    marks = [item for item in bell.items if isinstance(item, pg.InfiniteLine)]
    assert len(marks) == 6
    layout.close()


def test_frames_outside_their_band_are_filled_triangles(qt_app, clean):
    import pyqtgraph as pg

    from visualdynamics.plot.time_statistics import build_time_statistics

    spoiled = clean.copy()
    spoiled[round(5.05 * RATE):round(5.15 * RATE)] = 0.0
    record = history(spoiled)
    stats = record.time_statistics()
    layout = pg.GraphicsLayoutWidget()
    build_time_statistics(layout, record, stats)
    frames = _plots(layout)[2]
    triangles = [item for item in frames.listDataItems()
                 if item.opts.get('symbol') == 't']
    assert len(triangles) == 1
    assert triangles[0].getData()[0].size == stats.frames_outside
    layout.close()


def test_a_frame_with_no_variance_is_marked(qt_app, clean):
    import pyqtgraph as pg

    from visualdynamics.plot.time_statistics import build_time_statistics

    spoiled = clean.copy()
    spoiled[25 * 1024:30 * 1024] = 0.0
    record = history(spoiled)
    stats = record.time_statistics()
    layout = pg.GraphicsLayoutWidget()
    build_time_statistics(layout, record, stats)
    frames = _plots(layout)[2]
    dead = [item for item in frames.listDataItems()
            if item.opts.get('symbol') == 't1']
    assert len(dead) == 1 and dead[0].name() == 'frame with no variance'
    times, levels = dead[0].getData()
    assert times == pytest.approx([5.1, 5.3, 5.5, 5.7, 5.9])
    assert np.all(levels == stats.frame_band[0])
    assert '5 of 60 frames outside' in _titles(layout)[2]
    assert 'no variance at 5.1 s' in _titles(layout)[2]
    layout.close()


def test_the_view_draws_times_in_the_unit_systems_units(qt_app, clean):
    import pyqtgraph as pg

    from visualdynamics.plot.time_statistics import build_time_statistics
    from visualdynamics.units import SI

    record = history(clean)
    stats = record.time_statistics()
    layout = pg.GraphicsLayoutWidget()
    build_time_statistics(layout, record, stats,
                          unit_system=SI.with_units(time='ms'))
    trace, _bell, frames = _plots(layout)
    times = trace.listDataItems()[0].xData
    assert times[-1] == pytest.approx(1000.0 * (clean.size - 1) / RATE)
    assert frames.listDataItems()[0].xData[0] == pytest.approx(100.0)
    layout.close()


def test_the_view_draws_to_a_file(qt_app, tmp_path, clean):
    from visualdynamics.plot import plot_time_statistics

    path = tmp_path / 'statistics.png'
    plot_time_statistics(history(clean), 0, full_scale=40.0 * G,
                         path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


# ---- in the app -----------------------------------------------------------

def _statistics(window, pump, record):
    name = window.add_object('Clipped', record)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    pane = window.data_pane
    if not pane.statistics_action.isChecked():
        pane.statistics_action.trigger()
    pump()
    return pane


def test_a_time_history_offers_the_statistics_reading(window, pump, clean):
    pane = _statistics(window, pump,
                       history(np.clip(clean, -25.0, 25.0)))
    assert pane.statistics_action.isVisible()
    assert pane.showing_statistics
    panel = pane.statistics_panel
    assert not panel.isHidden()
    assert panel.values['kurtosis'].text() == '2.860'
    assert panel.values['crest'].text() == '2.830'
    assert panel.values['rms'].text() == '8.835 g'
    # no range given: the rail is not read, the kurtosis passes the
    # default 2 to 4, and the crest factor says clipped
    assert panel.values['rail'].text() == '—'
    assert 'Kurtosis' not in panel.verdict.text()
    assert 'Crest factor' in panel.verdict.text()
    assert not panel.values['kurtosis'].font().bold()
    assert panel.values['crest'].font().bold()
    assert not panel.values['skewness'].font().bold()


def test_a_range_typed_in_the_panel_counts_the_rail(window, pump, clean):
    pane = _statistics(window, pump,
                       history(np.clip(clean, -25.0, 25.0)))
    panel = pane.statistics_panel
    panel.range_value.setValue(25.0)
    panel.range_box.setChecked(True)
    pump()
    assert panel.names['range_value'].text() == 'Range [g]'
    assert panel.values['rail'].text() == '287 (0.47 %)'
    assert panel.values['rail'].font().bold()
    assert panel.values['crest'].font().bold()
    assert not panel.values['kurtosis'].font().bold()
    # the record's share in the row, the Gaussian's in its tip
    assert panel.values['beyond_3'].text() == '0 %'
    assert '0.27 %' in panel.values['beyond_3'].toolTip()
    assert panel.verdict.text().startswith('Outside its band:')
    assert 'Samples on the rail' in panel.verdict.text()
    # the reading is journaled as the call that draws it, in SI
    line = window.project.journal[-1]
    assert line.startswith(
        "visualdynamics.plot.plot_time_statistics(project['Clipped'], 0")
    assert f'full_scale={25.0 * G!r}' in line
    assert 'kurtosis_band=(2.0, 4.0)' in line


def test_the_kurtosis_band_set_tighter_in_the_panel_flags_the_clip(
        window, pump, clean):
    pane = _statistics(window, pump,
                       history(np.clip(clean, -25.0, 25.0)))
    panel = pane.statistics_panel
    assert (panel.low_box.value(), panel.high_box.value()) == (2.0, 4.0)
    panel.low_box.setValue(2.9)
    panel.high_box.setValue(3.1)
    pump()
    assert panel.values['kurtosis'].font().bold()
    assert 'Kurtosis' in panel.verdict.text()
    assert 'kurtosis_band=(2.9, 3.1)' in window.project.journal[-1]


def test_the_settings_survive_a_redraw(window, pump, clean):
    pane = _statistics(window, pump, history(clean))
    pane.statistics_panel.frame_box.setValue(0.5)
    pump()
    window.render_current()
    pump()
    assert pane.statistics_panel.frame_box.value() == 0.5
    assert pane.statistics_panel.values['frames'].text() == '0 of 24'


def test_the_panel_counts_samples_not_finite(window, pump, clean):
    spoiled = clean.copy()
    spoiled[[100, 200]] = np.nan
    panel = _statistics(window, pump, history(spoiled)).statistics_panel
    assert panel.values['nonfinite'].text() == '2'
    assert panel.values['nonfinite'].font().bold()
    assert 'Samples not finite' in panel.verdict.text()


def test_the_panel_states_times_and_rates_in_the_unit_systems_units(
        qt_app, clean):
    from visualdynamics.gui.statistics_panel import StatisticsPanel

    stats = time_statistics(clean, RATE, start=1.0)
    panel = StatisticsPanel()
    panel.show_statistics(stats, (G, 0.0), 'g', (0.001, 'ms'))
    assert panel.values['peak_time'].text() == (
        f'{1000.0 * stats.peak_time:.4g} ms')
    assert panel.row_names['crossings'].text() == 'Crossings /ms'
    assert panel.values['spectral_crossings'].text() == (
        f'{stats.spectral_crossings / 1000.0:.4g}')
    assert panel.values['std'].text() == f'{stats.std / G:.4g} g'
    panel.close()


def test_statistics_is_offered_where_the_averaging_is(window):
    pane = window.data_pane
    controls = {'map_wanted': None, 'diagonal': None, 'cmif': False,
                'complex_data': False, 'pair': False}
    pane.show_controls(**controls, averaging=False, shocks=True)
    assert not pane.statistics_action.isVisible()
    assert pane.kurtosis_action.isVisible()
    pane.show_controls(**controls, averaging=True, shocks=True)
    assert pane.statistics_action.isVisible()


def test_the_frame_is_sized_from_the_band_until_one_is_typed(
        window, pump, clean):
    pane = _statistics(window, pump, history(clean))
    panel = pane.statistics_panel
    assert panel.frame_box.text() == 'from band'
    assert panel.settings()['frame'] is None
    assert 'frame=None' in window.project.journal[-1]


def test_the_statistics_reading_is_flat(window, pump, clean):
    pane = _statistics(window, pump, history(clean))
    assert not pane.waterfall_action.isVisible()
