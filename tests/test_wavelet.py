"""The scalogram: where a record's frequencies are, moment by moment.

A wavelet transform is easy to write plausibly and hard to write
correctly, and every way of getting it wrong produces a picture that
looks like a scalogram. So the tests here pin numbers a wrong
implementation could not produce: a tone lands on *its own* frequency
line, a chirp's ridge tracks the frequency it was built with, energy
compares across scales, and the cone of influence is where the edges
actually reach.

The frequency test is the one that earns its keep. The grid is 96 lines
per octave — 0.72% apart — and the tone sits exactly on one of them, so
the widely quoted `omega0 / 2*pi*s` approximation for the Morlet's
centre frequency would put the peak nearly two lines out. At a coarser
grid that bias hides inside the quantisation, which is exactly how it
survives in other people's code.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import wavelet as w

RATE = 2048.0


def seconds(count=8192):
    return np.arange(count) / RATE


def tone(frequency, count=8192, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * frequency * seconds(count))


def grid_about(centre, octaves=1.0, per_octave=96):
    """A frequency grid with `centre` exactly on a line."""
    reach = int(octaves * per_octave)
    return centre * 2.0 ** (np.arange(-reach, reach + 1) / per_octave)


# ---- the frequency a peak is reported at --------------------------------


def test_a_tone_lands_on_its_own_frequency_line():
    """The whole transform in one number."""
    frequencies = grid_about(100.0)
    middle = np.abs(w.scalogram(tone(100.0), RATE, frequencies))[:, 4096]
    assert frequencies[middle.argmax()] == pytest.approx(100.0, rel=1e-9)


@pytest.mark.parametrize('frequency', [40.0, 100.0, 250.0, 700.0])
def test_it_lands_there_wherever_the_tone_is(frequency):
    # a quarter octave either side, so the grid for the highest tone
    # stays under Nyquist — which the transform refuses, correctly
    frequencies = grid_about(frequency, octaves=0.25)
    middle = np.abs(w.scalogram(tone(frequency), RATE, frequencies))[:, 4096]
    assert frequencies[middle.argmax()] == pytest.approx(frequency, rel=1e-9)


def test_the_frequency_mapping_is_the_one_this_normalisation_needs():
    """The pairing, which is a single decision and not two.

    This module normalises for amplitude, whose exact relation is
    `2*pi/omega0`. Torrence and Compo's `4*pi/(omega0 + sqrt(2+omega0**2))`
    is exact for unit *energy* — it is the one quoted everywhere, and
    it was in this file's first draft alongside the amplitude
    normalisation, which is the mismatch the test below measures.
    """
    assert w.fourier_factor(6.0) == pytest.approx(2 * np.pi / 6.0, rel=1e-12)


def test_borrowing_the_other_conventions_mapping_would_bias_every_reading():
    """Why the pairing is tested rather than commented.

    Same transform, same record, T&C's scale relation instead of this
    one: a 100 Hz tone is reported at 101.4 Hz. Not scatter — every
    frequency, always high by the same 1.4%, and invisible on any grid
    coarser than about 1%. This is the failure the file's docstring is
    about, made to happen on purpose so the claim is not merely an
    assertion.
    """
    omega0 = 6.0
    energy_factor = 4.0 * np.pi / (omega0 + np.sqrt(2.0 + omega0 ** 2))
    frequencies = grid_about(100.0, octaves=0.25)

    # the wrong pairing, built here rather than in the module: scales
    # from the energy convention, amplitudes from this one
    mismatched = 1.0 / (frequencies * energy_factor)
    from scipy.fft import fft, ifft

    record = tone(100.0)
    spectrum = fft(record - record.mean())
    omega = 2.0 * np.pi * np.fft.fftfreq(record.size, 1.0 / RATE)
    daughters = 2.0 * np.exp(
        -0.5 * (mismatched[:, None] * omega[None, :] - omega0) ** 2) * (
            omega[None, :] > 0.0)
    magnitude = np.abs(ifft(daughters * spectrum[None, :], axis=1))[:, 4096]

    # the row labelled f would hold a *longer* wavelet than it should,
    # and a longer wavelet listens lower — so the tone is found on a row
    # labelled above where it really is
    reported = frequencies[magnitude.argmax()]
    assert reported == pytest.approx(100.0 * w.fourier_factor() / energy_factor,
                                     rel=0.005)
    assert reported > 101.0, 'reported high, and always high'

    # and the module itself, on the same record, is right
    correct = np.abs(w.scalogram(record, RATE, frequencies))[:, 4096]
    assert frequencies[correct.argmax()] == pytest.approx(100.0, rel=1e-9)


def test_scale_and_frequency_are_one_axis():
    """The correction that shaped the view: they are inverses, not two
    independent things to draw."""
    frequencies = np.array([10.0, 100.0, 1000.0])
    assert w.frequency_for(w.scale_for(frequencies)) == pytest.approx(
        frequencies)


# ---- when, which is what a spectrum cannot say --------------------------


def test_a_chirps_ridge_follows_the_frequency_it_was_built_with():
    """The reason the view exists. A PSD of this record says 50 to 400
    Hz and nothing about the order they arrive in."""
    t = seconds()
    duration = t[-1]
    low, high = 50.0, 400.0
    sweep = np.sin(2 * np.pi * (low * t + (high - low) / (2 * duration) * t ** 2))
    frequencies = w.log_frequencies(30.0, 600.0, per_octave=48)

    found = w.ridge(w.scalogram(sweep, RATE, frequencies), frequencies)
    expected = low + (high - low) * t / duration
    # away from the ends, where the cone of influence is
    inside = (t > 0.4) & (t < duration - 0.4)
    error = np.abs(found[inside] - expected[inside]) / expected[inside]
    assert np.median(error) < 0.02
    assert error.max() < 0.05


def test_a_burst_is_found_where_it_happened():
    """A tone present for a quarter of the record, silent either side.
    Its spectrum would say 200 Hz for the whole duration."""
    t = seconds()
    record = np.zeros_like(t)
    window = (t > 1.0) & (t < 2.0)
    record[window] = np.sin(2 * np.pi * 200.0 * t[window])
    frequencies = grid_about(200.0, octaves=0.5, per_octave=24)

    magnitude = np.abs(w.scalogram(record, RATE, frequencies))
    row = magnitude[np.argmin(np.abs(frequencies - 200.0))]
    assert row[window].mean() > 20 * row[t < 0.8].mean()


def test_two_bursts_at_different_times_and_frequencies_are_told_apart():
    """What a scalogram is for, and what no spectrum of this record
    could show: the 400 Hz event happens second."""
    t = seconds()
    record = np.zeros_like(t)
    first = (t > 0.5) & (t < 1.2)
    second = (t > 2.4) & (t < 3.1)
    record[first] = np.sin(2 * np.pi * 100.0 * t[first])
    record[second] = np.sin(2 * np.pi * 400.0 * t[second])
    frequencies = w.log_frequencies(50.0, 800.0, per_octave=24)

    magnitude = np.abs(w.scalogram(record, RATE, frequencies))
    low = magnitude[np.argmin(np.abs(frequencies - 100.0))]
    high = magnitude[np.argmin(np.abs(frequencies - 400.0))]
    assert low[first].mean() > 10 * low[second].mean()
    assert high[second].mean() > 10 * high[first].mean()


# ---- heights that mean the same thing at every scale --------------------


def test_the_magnitude_is_the_records_own_amplitude():
    """The normalisation choice, pinned as a number. A 2 g tone reads 2,
    so the colour bar can carry the record's units.

    Under the unit-energy convention this would read proportional to
    sqrt(scale) instead — the same tone four times taller at 50 Hz than
    at 800 — and nothing on the picture would say so."""
    for frequency in (50.0, 200.0, 800.0):
        frequencies = grid_about(frequency, octaves=0.25, per_octave=48)
        magnitude = np.abs(w.scalogram(tone(frequency, amplitude=2.0),
                                       RATE, frequencies))
        assert magnitude[:, 4096].max() == pytest.approx(2.0, rel=0.02), (
            f'{frequency} Hz')


def test_a_louder_record_reads_proportionally_louder():
    frequencies = grid_about(100.0, octaves=0.25, per_octave=24)
    quiet = np.abs(w.scalogram(tone(100.0), RATE, frequencies))[:, 4096].max()
    loud = np.abs(w.scalogram(tone(100.0, amplitude=3.0), RATE,
                              frequencies))[:, 4096].max()
    assert loud == pytest.approx(3.0 * quiet, rel=1e-6)


def test_a_constant_offset_is_not_drawn_as_signal():
    """The Morlet is not quite admissible, so a DC offset would paint a
    bright band along the bottom of every scalogram — a picture of the
    offset rather than of the record."""
    frequencies = w.log_frequencies(20.0, 500.0, per_octave=12)
    flat = np.abs(w.scalogram(np.full(8192, 5.0), RATE, frequencies))
    assert flat.max() < 1e-9


# ---- the edges, drawn rather than remembered ----------------------------


def test_the_cone_is_wide_at_the_bottom_and_narrow_at_the_top():
    """Low frequencies use long wavelets, so they hang further off the
    end of the record."""
    cone = w.cone_of_influence([20.0, 200.0], RATE)
    assert cone[0] > cone[1]
    assert cone[0] == pytest.approx(10.0 * cone[1], rel=1e-9)


def test_the_cone_is_where_the_edge_effect_actually_is():
    """Not a decorative curve: a step at the start of an otherwise
    steady record rings inside the cone and is gone outside it."""
    t = seconds()
    record = np.sin(2 * np.pi * 100.0 * t)
    record[:1] += 50.0                     # one bad sample at the very edge
    frequencies = grid_about(100.0, octaves=0.25, per_octave=24)
    magnitude = np.abs(w.scalogram(record, RATE, frequencies))
    row = magnitude[np.argmin(np.abs(frequencies - 100.0))]

    cone = w.cone_of_influence([100.0], RATE)[0]
    edge = t < cone
    beyond = (t > 2 * cone) & (t < 1.0)
    assert row[edge].max() > 2 * row[beyond].max(), 'the edge rings'
    assert row[beyond].std() / row[beyond].mean() < 0.05, 'and settles'


# ---- what it refuses ----------------------------------------------------


def test_it_refuses_frequencies_the_record_cannot_carry():
    with pytest.raises(ValueError, match='Nyquist'):
        w.scalogram(tone(100.0), RATE, [100.0, RATE / 2.0])


def test_it_refuses_a_complex_record():
    """A measurement is real; a complex array here is a mistake worth
    catching rather than transforming."""
    with pytest.raises(ValueError, match='complex'):
        w.scalogram(np.zeros(64, dtype=complex), RATE, [100.0])


def test_the_frequency_grid_refuses_a_backwards_range():
    with pytest.raises(ValueError, match='positive low'):
        w.log_frequencies(500.0, 50.0)


def test_the_grid_is_evenly_spaced_by_octave():
    frequencies = w.log_frequencies(10.0, 160.0, per_octave=6)
    assert frequencies[0] == pytest.approx(10.0)
    assert frequencies[-1] == pytest.approx(160.0)
    ratios = frequencies[1:] / frequencies[:-1]
    assert ratios == pytest.approx(np.full(len(ratios), 2.0 ** (1 / 6)))


# ---- the ends of the record do not meet ---------------------------------


def test_a_loud_start_is_not_drawn_into_a_silent_tail():
    """A transform the length of the record is a *circular* convolution,
    so the wavelet running off the end comes back on at the beginning.

    Measured before the fix (2026-08-28): a record loud for its first
    second and silent afterwards read 0.47 of full scale at its end —
    a signal that is not there, drawn from the other end of the record.
    Zero-padding is what stops the two ends meeting, and it is what
    Torrence and Compo do for the same reason.
    """
    t = seconds()
    record = np.where(t < 1.0, np.sin(2 * np.pi * 80.0 * t), 0.0)
    frequencies = grid_about(80.0, octaves=0.25, per_octave=24)

    magnitude = np.abs(w.scalogram(record, RATE, frequencies))
    row = magnitude[np.argmin(np.abs(frequencies - 80.0))]
    loud = row[t < 1.0].max()
    tail = row[t > t[-1] - 0.5].max()
    assert tail < 0.02 * loud, 'the silent end reads silent'


def test_the_padding_does_not_move_what_is_inside_the_record():
    """Scaffolding, not signal: the answer away from the ends is the
    answer, and the trim puts the columns back where they belong."""
    t = seconds()
    record = np.sin(2 * np.pi * 100.0 * t)
    frequencies = grid_about(100.0, octaves=0.25, per_octave=24)
    magnitude = np.abs(w.scalogram(record, RATE, frequencies))

    assert magnitude.shape == (len(frequencies), len(record))
    middle = magnitude[:, len(record) // 2]
    assert frequencies[middle.argmax()] == pytest.approx(100.0, rel=1e-9)
    # and the amplitude is still the record's own, away from the edges
    assert middle.max() == pytest.approx(1.0, rel=0.02)


def test_the_transform_is_the_same_whatever_length_the_record_is():
    """The pad rounds up to a length the FFT likes, and a reading must
    not depend on which one that turned out to be. An awkward record
    length is exactly where the rounding does most, so this is the case
    that would show it."""
    t = seconds(8192)
    awkward = seconds(8191)
    tone_a = np.sin(2 * np.pi * 100.0 * t)
    tone_b = np.sin(2 * np.pi * 100.0 * awkward)
    frequencies = grid_about(100.0, octaves=0.25, per_octave=24)

    a = np.abs(w.scalogram(tone_a, RATE, frequencies))[:, 4000]
    b = np.abs(w.scalogram(tone_b, RATE, frequencies))[:, 4000]
    assert a == pytest.approx(b, rel=1e-6)


# ---- a ridge that ripples, and whether that is the transform's fault ----


def test_one_tone_reads_perfectly_smooth_along_time():
    """The control for the question below, and a guard in its own right:
    a steady tone must give a steady ridge. Any ripple here would be the
    transform's, and there is none — measured at zero to five decimal
    places."""
    t = seconds(16384)
    frequencies = grid_about(400.0, octaves=0.25, per_octave=48)
    row = np.abs(w.scalogram(np.sin(2 * np.pi * 400.0 * t), RATE,
                             frequencies))[
        np.argmin(np.abs(frequencies - 400.0))]
    middle = row[2000:-2000]
    assert middle.std() / middle.mean() < 1e-4


@pytest.mark.parametrize('spacing', [1.0, 5.0, 40.0])
def test_two_tones_in_one_band_beat_at_their_difference(spacing):
    """Why a scalogram of a multi-tone run ripples along time, asked of
    the plate's sine sweep (Brandon, 2026-08-28).

    It is not the transform. A wavelet has a finite bandwidth — about
    `f/omega0` either side — and two tones inside it sum to an
    amplitude-modulated signal, exactly as they would through any
    analogue filter of the same width. The ripple is at their
    difference frequency, and it is in the record: band-passing that
    run at 380-420 Hz with no wavelet anywhere gives the same 50%
    modulation at the same 1.00 Hz.

    So the picture is right, and the ripple is the measurement. Raising
    `omega0` narrows the band and separates tones that are far enough
    apart; for tones a hertz apart it cannot, because separating 1 Hz
    at 400 Hz needs a wavelet hundreds of cycles long and seconds wide,
    at which point the reading has no time resolution left to lose.
    """
    t = seconds(16384)
    frequencies = grid_about(400.0, octaves=0.25, per_octave=48)
    pair = (np.sin(2 * np.pi * 400.0 * t)
            + np.sin(2 * np.pi * (400.0 + spacing) * t))
    row = np.abs(w.scalogram(pair, RATE, frequencies))[
        np.argmin(np.abs(frequencies - 400.0))]

    middle = row[2000:-2000]
    assert middle.std() / middle.mean() > 0.1, 'it ripples'
    centred = middle - middle.mean()
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(len(centred))))
    found = np.fft.rfftfreq(len(centred), 1.0 / RATE)[spectrum.argmax()]
    assert found == pytest.approx(spacing, rel=0.05), 'at their difference'


def test_a_wider_wavelet_separates_tones_far_enough_apart():
    """The knob the panel offers, and what it is for. 40 Hz apart at
    400 Hz is inside a six-cycle wavelet's band and outside a
    twenty-four-cycle one."""
    t = seconds(16384)
    frequencies = grid_about(400.0, octaves=0.25, per_octave=48)
    pair = np.sin(2 * np.pi * 400.0 * t) + np.sin(2 * np.pi * 440.0 * t)

    def wobble(omega0):
        row = np.abs(w.scalogram(pair, RATE, frequencies, omega0))[
            np.argmin(np.abs(frequencies - 400.0))]
        middle = row[3000:-3000]
        return middle.std() / middle.mean()

    assert wobble(6.0) > 0.3, 'both tones in the band'
    assert wobble(24.0) < 0.1, 'and a narrower wavelet tells them apart'


# ---- how near Nyquist the reading stays honest --------------------------


@pytest.mark.parametrize('fraction', [0.4, 0.6, 0.8, 0.95])
def test_the_reading_is_honest_close_to_nyquist(fraction):
    """The panel's default top of the axis used to be 0.4 of Nyquist,
    on the argument that a wavelet up there is a couple of samples long
    and resolves nothing. The argument was wrong and this is what
    measured it (Brandon asked why his sweep stopped at 800 Hz,
    2026-08-28).

    Amplitude, frequency and time localisation, because a defect could
    hide in any one of them: the Gaussian *is* truncated by the Nyquist
    edge up there, and what needed checking was whether the truncation
    takes anything but tail. It does not.
    """
    nyquist = RATE / 2.0
    frequency = fraction * nyquist
    t = seconds(16384)
    grid = frequency * 2.0 ** (np.arange(-48, 49) / 96)
    grid = grid[grid < nyquist * 0.999]

    steady = 2.0 * np.sin(2 * np.pi * frequency * t)
    magnitude = np.abs(w.scalogram(steady, RATE, grid))[:, 8192]
    assert magnitude.max() == pytest.approx(2.0, rel=0.01), 'amplitude'
    assert grid[magnitude.argmax()] == pytest.approx(frequency, rel=1e-3), \
        'frequency'

    at = 2.0
    burst = np.exp(-0.5 * ((t - at) / 0.02) ** 2) * np.sin(
        2 * np.pi * frequency * t)
    row = np.abs(w.scalogram(burst, RATE, grid))[
        np.argmin(np.abs(grid - frequency))]
    centre = (t * row).sum() / row.sum()
    assert centre == pytest.approx(at, abs=0.005), 'when it happened'


# ---- a long record: bounded memory, and a picture-sized reading ---------


def test_the_peak_reading_is_the_full_transform_where_it_fits():
    """Under the column budget the reading *is* the magnitude — a
    small record's amplitudes are drawn exactly, as they always were."""
    record = tone(100.0, count=4096)
    frequencies = grid_about(100.0, per_octave=12)
    times, held = w.scalogram_peaks(record, RATE, frequencies, columns=4096)
    full = np.abs(w.scalogram(record, RATE, frequencies))
    assert held.shape == full.shape
    np.testing.assert_allclose(held, full, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(times, seconds(4096))


def test_the_peak_reading_keeps_each_bins_largest_magnitude():
    """Past the budget, every column is the *largest* magnitude in its
    slice of time: a stride would land between the samples a burst
    lives in, and a mean would halve it."""
    record = np.zeros(8192)
    record[3000:3010] = 1.0   # ten samples: a stride of 32 mostly misses it
    frequencies = grid_about(200.0, octaves=0.5, per_octave=12)
    full = np.abs(w.scalogram(record, RATE, frequencies))
    times, held = w.scalogram_peaks(record, RATE, frequencies, columns=256)
    assert held.shape == (len(frequencies), 256)
    assert times.shape == (256,)
    expected = full.reshape(len(frequencies), 256, 32).max(axis=2)
    np.testing.assert_allclose(held, expected, rtol=1e-12, atol=1e-12)
    # each column's clock is the centre of its slice, so the first and
    # last columns sit half a slice inside the record's ends
    np.testing.assert_allclose(times[0], 15.5 / RATE)
    np.testing.assert_allclose(times[-1], (8192 - 16.5) / RATE)
    assert held.max() == pytest.approx(full.max()), 'the burst survives'


def test_a_ragged_last_slice_is_kept_not_dropped():
    record = np.zeros(1000)
    record[995] = 1.0    # in the short final slice
    frequencies = grid_about(200.0, octaves=0.5, per_octave=12)
    full = np.abs(w.scalogram(record, RATE, frequencies))
    _times, held = w.scalogram_peaks(record, RATE, frequencies, columns=300)
    assert held.shape[1] == 250, '1000 samples in slices of 4'
    np.testing.assert_allclose(held[:, -1], full[:, 996:].max(axis=1))


def test_the_transform_never_holds_every_scale_at_once(monkeypatch):
    """The whole point of the band budget: a five-minute record at
    16 kHz is 121 scales of 4.9M complex samples, three of them alive at
    once in the one-shot form — 27 GB, which is the crash Brandon saw.
    Pinned with tracemalloc, which sees numpy's allocations: the peak
    must stay under what one band plus the answer costs, not what every
    scale at once would."""
    import tracemalloc

    from scipy.fft import next_fast_len

    record = tone(100.0, count=32768)
    frequencies = w.log_frequencies(4.0, 900.0, 12)
    scales = w.scale_for(frequencies)
    margin = int(np.ceil(np.sqrt(2.0) * scales.max() * RATE))
    length = next_fast_len(record.size + 2 * margin)
    one_shot = 3 * len(frequencies) * length * 16   # daughters, product, ifft

    monkeypatch.setattr(w, 'BAND_BYTES', 2 * length * 16)   # two rows a band
    tracemalloc.start()
    try:
        held = w.scalogram_peaks(record, RATE, frequencies, columns=512)[1]
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert held.shape == (len(frequencies), 512)
    assert peak < one_shot / 4, (
        f'peak {peak / 2**20:.0f} MB against {one_shot / 2**20:.0f} MB '
        f'for the one-shot form')


def test_a_band_boundary_does_not_change_the_answer(monkeypatch):
    """The bands are scaffolding: however the scales are cut, the
    coefficients are the ones the one-shot form gave."""
    record = tone(100.0, count=4096) + 0.5 * tone(300.0, count=4096)
    frequencies = w.log_frequencies(50.0, 600.0, 12)
    whole = w.scalogram(record, RATE, frequencies)
    monkeypatch.setattr(w, 'BAND_BYTES', 1)   # one row a band
    banded = w.scalogram(record, RATE, frequencies)
    np.testing.assert_allclose(banded, whole, rtol=1e-12, atol=1e-12)
