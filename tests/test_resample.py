"""Resampling with an anti-alias filter, and the reading that shows it.

Pinned against arithmetic that does not depend on the implementation.
Content at F sampled at fs lands at |n fs - F|: a 4,600 Hz tone on a
channel sampled at 5,120 Hz lands at 520 Hz, and its samples are those
of a 520 Hz tone with its sign reversed. The alias-free span is the
rate over 2.56, so 5,120 Hz keeps 2,000 Hz, and a 1,024-sample frame
keeps 400 of its 512 lines and drops 112. The lowest frequency that
folds into the span is the rate less the span, 3,120 Hz, 1.56 spans.
The filter's corner is at 1.2 spans, 2,400 Hz, so the fold is 1.3
corners up: a Butterworth of design order N run forward and backward
is 20 log10(1 / (1 + 1.3^(2N))) there in the analog limit, 54.7 dB
down at twelfth order, 50.2 at eleventh and 45.6 at tenth, against
50 dB, and at the span, 1 / 1.2
corners, it is 0.11 dB down where a corner on the span would be 6.02.
The digital design is further down at the fold than the analog limit
at every factor, 55.8 dB at a factor of 10 (51,200 to 5,120 Hz). A 1 g
peak tone on
a line of a Hann-windowed autospectrum at 5 Hz lines stands at
0.5 / (1.5 x 5) = 0.0667 g^2/Hz. And a low-pass at the span run after
the decimation leaves the 520 Hz alias where it is.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import Filtering, filtered, response
from visualdynamics.core.resample import (
    CORNER,
    ORDER,
    SPAN_RATIO,
    AliasWarning,
    alias_free_span,
    alias_frequency,
    anti_alias_filter,
    decimation_factor,
    level_change,
    line_counts,
    ordinal,
    read_resample,
    resample,
)

G = 9.80665
#: the record's rate: sixteen times the 5,120 Hz channel, as the
#: voltage is described between the converter's samples
FINE = 81920.0
RATE = 5120.0


def record(tones=((4600.0, 1.0),), samples=16 * 4096, rate=FINE,
           noise=0.0, seed=0):
    """A record of tones, each (frequency, g peak), in m/s^2 shown in g,
    with a little white noise if asked."""
    t = np.arange(samples) / rate
    y = sum(peak * np.sin(2.0 * np.pi * f * t) for f, peak in tones)
    if noise:
        y = y + noise * np.random.default_rng(seed).standard_normal(samples)
    return TimeHistory(t, np.atleast_2d(y * G), response_dof=['1Z+'],
                       ordinate_dim=['acceleration'], ordinate_unit=['g'])


def zero_phase_butterworth_db(ratio, order):
    """The analog Butterworth run forward and backward, at `ratio`
    times its corner: the hand value the digital design approaches."""
    return 20.0 * np.log10(1.0 / (1.0 + ratio ** (2 * order)))


# ---- the arithmetic ---------------------------------------------------


def test_a_4600_hz_tone_sampled_at_5120_hz_lands_at_520_hz():
    assert alias_frequency(4600.0, 5120.0) == 520.0
    # the band 90 to 110 kHz folds into a 10 kHz band at 100 kS/s
    assert np.allclose(alias_frequency([90e3, 95e3, 105e3, 110e3], 100e3),
                       [10e3, 5e3, 5e3, 10e3])
    # below the Nyquist frequency a frequency is its own alias, and far
    # above it the nearest multiple of the rate is taken
    assert alias_frequency(1000.0, 5120.0) == 1000.0
    assert alias_frequency(3 * 5120.0 + 100.0, 5120.0) == pytest.approx(100.0)
    assert alias_frequency(2 * 5120.0 - 100.0, 5120.0) == pytest.approx(100.0)


def test_the_span_is_the_rate_over_2_56():
    assert SPAN_RATIO == 2.56
    assert alias_free_span(5120.0) == pytest.approx(2000.0)
    assert alias_free_span(51200.0) == pytest.approx(20000.0)
    with pytest.raises(ValueError, match='above 2'):
        alias_free_span(5120.0, 2.0)
    with pytest.raises(ValueError, match='above zero'):
        alias_free_span(0.0)


@pytest.mark.parametrize('frame,expected', [(1024, (512, 400, 112)),
                                            (2048, (1024, 800, 224)),
                                            (4096, (2048, 1600, 448)),
                                            (1000, (500, 390, 110))])
def test_a_frame_keeps_its_length_over_2_56_lines(frame, expected):
    assert line_counts(frame) == expected


def test_the_factor_is_a_whole_number():
    assert decimation_factor(51200.0, 5120.0) == 10
    assert decimation_factor(FINE, RATE) == 16
    # a rate read off an abscissa's step does not come out exact
    assert decimation_factor(51199.99999, 5120.0) == 10
    with pytest.raises(ValueError, match=r'5120 Hz \(every 10th sample\) '
                                         r'and 4654.55 Hz \(every 11th'):
        decimation_factor(51200.0, 5000.0)
    # the small factors are spelled as ordinals too, not '2th'
    with pytest.raises(ValueError, match='every 2nd sample') as refused:
        decimation_factor(5120.0, 2200.0)
    assert '1706.67 Hz (every 3rd sample)' in str(refused.value)
    with pytest.raises(ValueError, match='never raises'):
        decimation_factor(5120.0, 5120.0)
    with pytest.raises(ValueError, match='never raises'):
        decimation_factor(5120.0, 10240.0)
    with pytest.raises(ValueError, match='above zero'):
        decimation_factor(5120.0, 0.0)


def gain(filtering, rate, frequencies):
    """The zero-phase gain the record sees, dB, as `filters.response`
    reads it."""
    return response(filtering, rate, frequencies=frequencies)[1]


def test_the_span_is_flat_and_the_lowest_fold_55_db_down():
    """Brandon's numbers (2026-10-23), at 51,200 to 5,120 Hz: with the
    corner on the span the span was 1.46 dB down at 0.9 of it and
    6.02 dB at its top; with the twelfth-order corner at 1.2 spans it is
    0.10 dB down at its top and 55.8 dB down at the lowest fold."""
    span = alias_free_span(RATE)
    fold = RATE - span
    rate = 10 * RATE
    assert fold / span == pytest.approx(1.56)
    assert CORNER == 1.2 and ORDER == 12
    assert anti_alias_filter(RATE) == Filtering(high=2400.0, order=12)
    on_span = gain(Filtering(high=span, order=8), rate,
                   [0.9 * span, span, fold])
    assert on_span == pytest.approx([-1.46, -6.02, -62.82], abs=0.01)
    flat = gain(anti_alias_filter(RATE), rate, [0.9 * span, span, fold])
    assert flat == pytest.approx([-0.01, -0.10, -55.76], abs=0.01)
    # the analog limit, which the digital design is below at every factor
    for order, hand in ((12, -54.7), (11, -50.2), (10, -45.6)):
        assert zero_phase_butterworth_db(1.3, order) == pytest.approx(
            hand, abs=0.05)
    assert zero_phase_butterworth_db(1.0 / 1.2, 12) == pytest.approx(
        -0.11, abs=0.005)
    for factor in (2, 4, 10, 16, 100):
        at = gain(anti_alias_filter(RATE), factor * RATE, [span, fold])
        assert at[0] > -0.12
        assert at[1] < zero_phase_butterworth_db(1.3, 12) + 0.01
        assert gain(anti_alias_filter(RATE, 11), factor * RATE,
                    [fold])[0] < -50.0
    assert gain(anti_alias_filter(RATE, 10), 100 * RATE, [fold])[0] > -50.0
    # a span ratio that would put the corner past Nyquist is refused
    with pytest.raises(ValueError, match='Nyquist'):
        anti_alias_filter(RATE, ratio=2.3)


def test_the_words():
    assert level_change(-61.84) == '61.8 dB down'
    assert level_change(0.01) == 'unchanged'
    assert level_change(3.06) == '3.1 dB up'
    assert [ordinal(n) for n in (1, 2, 3, 4, 10, 11, 12, 13, 16, 21, 22,
                                 111, 123)] == [
        '1st', '2nd', '3rd', '4th', '10th', '11th', '12th', '13th', '16th',
        '21st', '22nd', '111th', '123rd']


# ---- resampling a record ------------------------------------------------


def test_with_no_filter_the_tone_is_a_520_hz_tone_with_its_sign_reversed():
    """Every 16th sample, bare: the samples are those of a 520 Hz tone,
    reversed, and nothing in them tells the two apart. It warns."""
    source = record()
    with pytest.warns(AliasWarning, match=r'5120 Hz with no anti-alias '
                                          r'filter: .* between 2560 and '
                                          r'40960 Hz'):
        bare = resample(source, RATE, anti_alias=False)
    t = bare.abscissa
    assert bare.sample_rate == pytest.approx(RATE)
    assert np.max(np.abs(bare.ordinate[0] / G
                         + np.sin(2.0 * np.pi * 520.0 * t))) < 1e-9


def test_the_warning_points_at_the_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        resample(record(), RATE, anti_alias=False)
    assert [w.category for w in caught] == [AliasWarning]
    assert caught[0].filename == __file__


def test_filtered_first_the_tone_is_gone_and_the_content_kept():
    """The 4,600 Hz tone is far down behind the default filter; a
    300 Hz tone, well inside the span, comes through at its height."""
    source = record(((4600.0, 1.0), (300.0, 0.1)))
    with warnings.catch_warnings():
        warnings.simplefilter('error', AliasWarning)
        slow = resample(source, RATE)
    t = slow.abscissa
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    expected = 0.1 * np.sin(2.0 * np.pi * 300.0 * t)
    assert np.max(np.abs(slow.ordinate[0, middle] / G
                         - expected[middle])) < 1e-4


def test_the_result_keeps_the_clock_the_records_and_the_shock_windows():
    """The windows are in seconds from the record's start, which is
    kept, so they are carried; the averaging counts samples and is
    not."""
    from visualdynamics.core.shocks import Shock

    source = record(samples=16 * 4096 + 5)
    source.averaging = Averaging(frame_length=8192, frames=2)
    source.shocks = (Shock(0.1, 0.2), Shock(0.5, 0.25))
    slow = resample(source, RATE)
    assert len(slow.abscissa) == 4097
    assert np.array_equal(slow.abscissa, source.abscissa[::16])
    assert slow.response_dof == ['1Z+']
    assert slow.ordinate_dim == ['acceleration']
    assert slow.ordinate_unit == ['g']
    assert slow.averaging is None
    assert slow.shocks == (Shock(0.1, 0.2), Shock(0.5, 0.25))
    # and land on the same samples of the slower clock
    assert [(round(w.start * RATE), round(w.duration * RATE))
            for w in slow.srs_windows()] == [(512, 1024), (2560, 1280)]
    assert not resample(record(), RATE).shocks


def test_a_record_that_starts_late_keeps_its_start():
    """A record whose clock starts at 3.25 s resamples and reads as the
    same record from zero does, its clock still starting at 3.25 s and
    its windows still on the same samples."""
    from visualdynamics.core.shocks import Shock

    early = record(((4600.0, 1.0), (300.0, 0.1)))
    late = TimeHistory(early.abscissa + 3.25, early.ordinate.copy(),
                       **early.record_fields())
    late.shocks = (Shock(0.1, 0.2),)
    slow = resample(late, RATE)
    assert slow.abscissa[0] == pytest.approx(3.25)
    assert slow.sample_rate == pytest.approx(RATE)
    assert np.allclose(slow.abscissa, late.abscissa[::16])
    assert np.array_equal(slow.ordinate, resample(early, RATE).ordinate)
    assert [(round(w.start * RATE), round(w.duration * RATE))
            for w in slow.srs_windows()] == [(512, 1024)]
    reading = read_resample(late, RATE, frequencies=[4600.0])
    same = read_resample(early, RATE, frequencies=[4600.0])
    assert reading.describe() == same.describe()
    (fold,) = reading.folds
    assert fold.bare / G ** 2 == pytest.approx(0.0667, abs=1e-4)
    for key in ('source', 'filtered', 'bare', 'after'):
        # to rounding: the clock read from 3.25 s on differs in its
        # last bits, and the filtered curve's floor is rounding itself
        g = same.spectra[key][1]
        assert np.allclose(reading.spectra[key][1], g, rtol=1e-6,
                           atol=1e-12 * np.max(g))


def test_filtering_a_row_at_a_time_matches_filtering_the_record():
    """The rows are filtered and decimated one at a time, on threads; the
    values are those of filtering the whole record and then keeping
    every 16th sample, to the bit."""
    from scipy.signal import sosfiltfilt

    from visualdynamics.core.filters import design

    rows = np.random.default_rng(3).standard_normal((8, 131072))
    source = TimeHistory(np.arange(131072) / FINE, rows,
                         response_dof=[f'{i}Z+' for i in range(8)],
                         ordinate_dim=['acceleration'] * 8)
    slow = resample(source, RATE)
    whole = sosfiltfilt(design(anti_alias_filter(RATE), FINE), rows)
    assert np.array_equal(slow.ordinate, whole[:, ::16])


def test_resampling_does_not_hold_the_filtered_record_whole():
    """Filtering the whole record and then decimating held twice the
    record at the peak (2.06 times, Brandon, 2026-10-23). A row at a
    time, the peak is under the record's own size."""
    import tracemalloc

    from visualdynamics.core.filters import _filter_pool

    rows = np.random.default_rng(4).standard_normal((8, 131072))
    source = TimeHistory(np.arange(131072) / FINE, rows,
                         response_dof=[f'{i}Z+' for i in range(8)],
                         ordinate_dim=['acceleration'] * 8)
    # the threads and the lazy imports made first, so the peak is the
    # resampling's own
    _filter_pool()
    resample(record(samples=4096), RATE)
    tracemalloc.start()
    try:
        resample(source, RATE)
        _now, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # 0.82 times here: two rows at a time, each taking three rows of
    # working space, against 2.01 filtering the record whole
    assert peak < 1.0 * rows.nbytes


def test_every_channel_is_resampled():
    one, two = record(((4600.0, 1.0),)), record(((300.0, 1.0),))
    both = TimeHistory(one.abscissa, np.vstack([one.ordinate, two.ordinate]),
                       response_dof=['1Z+', '2Z+'],
                       ordinate_dim=['acceleration', 'acceleration'])
    slow = resample(both, RATE)
    assert slow.ordinate.shape == (2, 4096)
    assert np.allclose(slow.ordinate[1], resample(two, RATE).ordinate[0])


def test_the_method_is_the_function():
    source = record()
    assert np.array_equal(source.resample(RATE).ordinate,
                          resample(source, RATE).ordinate)
    assert np.array_equal(source.resample(RATE, order=4).ordinate,
                          resample(source, RATE, order=4).ordinate)
    with pytest.warns(AliasWarning):
        source.resample(RATE, anti_alias=False)


def test_a_low_pass_after_decimating_leaves_the_alias_where_it_is():
    """Run on the bare 5,120 Hz record, the anti-alias low-pass finds
    the 520 Hz alias inside its pass band and leaves it."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AliasWarning)
        bare = resample(record(), RATE, anti_alias=False)
    after = filtered(bare, anti_alias_filter(RATE))
    middle = slice(1024, 3072)
    assert np.max(np.abs(after.ordinate[0, middle]
                         - bare.ordinate[0, middle])) / G < 1e-3


def test_an_unevenly_sampled_record_is_refused():
    source = record()
    source.abscissa[10] += 1e-6
    with pytest.raises(ValueError, match='evenly spaced'):
        resample(source, RATE)


# ---- the reading --------------------------------------------------------


def test_the_reading_of_the_worked_example():
    reading = read_resample(record(), RATE, frequencies=[4600.0])
    assert (reading.source_rate, reading.rate, reading.factor) == (
        pytest.approx(FINE), pytest.approx(RATE), 16)
    assert reading.nyquist == pytest.approx(2560.0)
    assert reading.span == pytest.approx(2000.0)
    assert reading.corner == pytest.approx(2400.0)
    assert (reading.frame_length, reading.lines, reading.kept,
            reading.dropped) == (1024, 512, 400, 112)
    assert reading.resolution == pytest.approx(5.0)
    assert reading.lowest_fold == pytest.approx(3120.0)
    assert reading.gain_at_span == pytest.approx(-0.11, abs=0.005)
    assert reading.gain_at_fold == pytest.approx(-55.1, abs=0.05)
    (fold,) = reading.folds
    assert (fold.frequency, fold.alias, fold.multiple, fold.in_span) == (
        4600.0, 520.0, 1, True)
    # the tone on a line: 0.5 g^2 over the Hann window's 1.5 lines of 5 Hz
    assert fold.source / G ** 2 == pytest.approx(0.0667, abs=1e-4)
    assert fold.bare / G ** 2 == pytest.approx(0.0667, abs=1e-4)
    assert fold.after_db == pytest.approx(0.0, abs=0.01)
    assert fold.filtered_db < -100.0
    assert reading.describe() == (
        '81920 Hz to 5120 Hz (every 16th sample) behind a low-pass at '
        '2400 Hz: Nyquist 2560 Hz, span 2000 Hz, 400 of 512 lines kept. '
        f'4600 Hz folds to 520 Hz, inside the span: '
        f'{level_change(fold.filtered_db)} filtered first, unchanged '
        'low-passed after decimating.')


def test_the_spectra_share_one_line_spacing():
    reading = read_resample(record(), RATE)
    spacings = {name: f[1] - f[0] for name, (f, _g) in reading.spectra.items()}
    assert set(spacings) == {'source', 'filtered', 'bare', 'after'}
    assert all(s == pytest.approx(5.0) for s in spacings.values())
    assert reading.spectra['source'][0][-1] == pytest.approx(FINE / 2.0)
    assert reading.spectra['bare'][0][-1] == pytest.approx(RATE / 2.0)
    assert reading.density_dim == 'acceleration**2/frequency'


def test_with_no_frequency_given_the_tallest_line_above_nyquist_is_folded():
    reading = read_resample(record(((4600.0, 1.0), (300.0, 2.0),
                                    (7000.0, 0.1))), RATE)
    assert [fold.frequency for fold in reading.folds] == [4600.0]


def test_a_frequency_that_folds_above_the_span_lands_among_the_dropped_lines():
    """2,900 Hz folds to 2,220 Hz, between the span and the Nyquist
    frequency: a line the analyzer drops. Filtered first it is taken
    down; the low-pass after the decimation, its corner at 2,400 Hz,
    leaves it about where it is, among the lines dropped anyway."""
    reading = read_resample(record(((2900.0, 1.0),)), RATE,
                            frequencies=[2900.0, 1000.0, 2560.0])
    (fold,) = reading.folds
    assert fold.alias == pytest.approx(2220.0)
    assert not fold.in_span
    assert fold.filtered_db < -30.0
    assert fold.after_db > -3.0
    assert 'among the dropped lines' in fold.describe()


def test_a_frequency_typed_that_does_not_fold_is_named_not_dropped():
    """1,000 Hz is under the new Nyquist frequency and 2,560 Hz on it:
    neither folds, and the reading says so rather than leaving them out
    without a word."""
    reading = read_resample(record(((2900.0, 1.0),)), RATE,
                            frequencies=[2900.0, 1000.0, 2560.0])
    assert [fold.frequency for fold in reading.folds] == [2900.0]
    assert reading.unfolded == (1000.0, 2560.0)
    assert reading.notes == (
        '1000 Hz is below the new Nyquist frequency and does not fold',
        '2560 Hz is at the new Nyquist frequency and does not fold')
    assert reading.describe().endswith(
        'low-passed after decimating; 1000 Hz is below the new Nyquist '
        'frequency and does not fold; 2560 Hz is at the new Nyquist '
        'frequency and does not fold.')
    assert read_resample(record(), RATE, frequencies=[4600.0]).notes == ()


def test_a_record_with_nothing_above_the_new_nyquist_folds_nothing():
    reading = read_resample(record(((300.0, 1.0),)), RATE)
    # a pure tone's leakage reaches above 2,560 Hz, so its tallest line
    # there is folded; the reading says where, at a level far down
    assert len(reading.folds) <= 1
    silent = TimeHistory(np.arange(16 * 4096) / FINE,
                         np.zeros((1, 16 * 4096)), response_dof=['1Z+'],
                         ordinate_dim=['acceleration'])
    assert read_resample(silent, RATE).folds == ()
    assert read_resample(silent, RATE).describe().endswith(
        'lines kept. 1Z+ is zero throughout, so no level is read where '
        'anything folds.')


def test_the_reading_reads_the_channel_asked_for():
    one, two = record(((4600.0, 1.0),)), record(((3600.0, 1.0),))
    both = TimeHistory(one.abscissa, np.vstack([one.ordinate, two.ordinate]),
                       response_dof=['1Z+', '2Z+'],
                       ordinate_dim=['acceleration', 'acceleration'])
    reading = read_resample(both, RATE, 1)
    assert reading.record == 1
    assert reading.label.startswith('2Z+')
    assert [fold.frequency for fold in reading.folds] == [3600.0]
    assert reading.folds[0].alias == pytest.approx(1520.0)


def test_the_frame_can_be_chosen_and_must_fit():
    source = record()
    reading = read_resample(source, RATE, frame_length=2048)
    assert (reading.lines, reading.kept, reading.dropped) == (1024, 800, 224)
    assert reading.resolution == pytest.approx(2.5)
    with pytest.raises(ValueError, match='longer than the record'):
        read_resample(source, RATE, frame_length=8192)
    with pytest.raises(ValueError, match='two samples'):
        read_resample(source, RATE, frame_length=1)
    # a short record takes the longest power of two it holds
    short = record(samples=16 * 600)
    assert read_resample(short, RATE).frame_length == 512


def test_the_reading_refuses_what_it_cannot_read():
    source = record()
    with pytest.raises(ValueError, match='no channel 3'):
        read_resample(source, RATE, 3)
    with pytest.raises(ValueError, match='too few for a spectrum'):
        read_resample(record(samples=200), FINE / 16)
    with pytest.raises(ValueError, match='whole factor'):
        read_resample(source, 5000.0)


def test_the_reading_does_not_warn():
    """The bare decimation is the comparison the reading shows, not a
    resampling anyone asked for."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', AliasWarning)
        read_resample(record(), RATE)


# ---- the view -----------------------------------------------------------


def _plots(layout):
    import pyqtgraph as pg

    return [item for item in layout.ci.items
            if isinstance(item, pg.PlotItem)]


def test_the_view_marks_the_span_the_nyquist_frequency_and_the_dropped_lines(
        qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.resample import build_resample

    reading = read_resample(record(noise=0.01), RATE, frequencies=[4600.0])
    layout = pg.GraphicsLayoutWidget()
    try:
        (plot,) = _plots(build_resample(layout, reading) and layout)
        lines = {round(item.value()): item for item in plot.items
                 if isinstance(item, pg.InfiniteLine)}
        assert set(lines) == {2000, 2560}
        assert lines[2560].pen.style() == Qt.PenStyle.DashLine
        assert lines[2000].pen.style() == Qt.PenStyle.SolidLine
        assert lines[2560].label.format == 'Nyquist 2560 Hz'
        assert lines[2000].label.format == 'span 2000 Hz'
        (region,) = [item for item in plot.items
                     if isinstance(item, pg.LinearRegionItem)]
        assert region.getRegion() == pytest.approx((2000.0, 2560.0))
        assert region.brush.style() == Qt.BrushStyle.BDiagPattern
        names = [label.text for _sample, label in plot.legend.items]
        assert names[:4] == [
            'as recorded, 81920 Hz',
            'filtered at 2400 Hz, then decimated to 5120 Hz',
            'decimated with no filter',
            'decimated, then low-passed at 2400 Hz']
        assert 'dropped lines, 112 of 512' in names
        assert 'above the new Nyquist frequency, as recorded' in names
        assert 'where it lands, decimated with no filter' in names
        # the spectra told apart by their line, not their color
        curves = [item for item in plot.listDataItems()
                  if item.opts['pen'] is not None and item.name()
                  in names[:4]]
        styles = [curve.opts['pen'].style() for curve in curves]
        assert styles == [Qt.PenStyle.SolidLine, Qt.PenStyle.SolidLine,
                          Qt.PenStyle.DashLine, Qt.PenStyle.DotLine]
        assert curves[1].opts['pen'].widthF() > curves[0].opts['pen'].widthF()
        assert plot.getViewBox().state['logMode'] == [False, True]
        assert 'every 16th sample, 400 of 512 lines kept' in \
            plot.titleLabel.text
        assert '4600 Hz folds to 520 Hz, inside the span' in \
            plot.titleLabel.text
        # the axis runs past the fold, so the content that folds is seen
        low, high = plot.getViewBox().viewRange()[0]
        assert low == 0.0 and high > 4600.0
    finally:
        layout.close()


def test_the_view_keeps_its_words_apart_and_inside_the_plot(qt_app):
    """At 1000 by 700 the span's and the Nyquist frequency's names do not
    touch (the Nyquist frequency's sits right of its line, the span's
    left of its own), and the words above a fold stay under the top of
    the plot rather than running under the title."""
    import pyqtgraph as pg

    from visualdynamics.plot.resample import build_resample

    reading = read_resample(record(noise=0.01), RATE, frequencies=[4600.0])
    layout = pg.GraphicsLayoutWidget()
    layout.resize(1000, 700)
    try:
        build_resample(layout, reading)
        layout.show()
        qt_app.processEvents()
        (plot,) = _plots(layout)
        names = {round(item.value()): item.label.sceneBoundingRect()
                 for item in plot.items if isinstance(item, pg.InfiniteLine)}
        assert names[2000].right() < names[2560].left()
        (words,) = [item for item in plot.items
                    if isinstance(item, pg.TextItem)]
        top = plot.getViewBox().sceneBoundingRect().top()
        assert words.sceneBoundingRect().top() >= top
    finally:
        layout.close()


def test_the_view_joins_each_fold_to_where_it_lands(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.resample import build_resample

    reading = read_resample(record(), RATE, frequencies=[4600.0])
    layout = pg.GraphicsLayoutWidget()
    try:
        build_resample(layout, reading)
        (plot,) = _plots(layout)
        marks = [item for item in plot.listDataItems()
                 if item.opts['symbol'] == 't']
        hollow = [m for m in marks if m.opts['symbolBrush'].color().alpha() == 0]
        filled = [m for m in marks if m.opts['symbolBrush'].color().alpha() > 0]
        assert [float(m.xData[0]) for m in hollow] == [4600.0]
        assert [float(m.xData[0]) for m in filled] == [520.0]
        (link,) = [item for item in plot.listDataItems()
                   if item.xData is not None and len(item.xData) == 2
                   and item.opts['symbol'] is None]
        assert list(link.xData) == [4600.0, 520.0]
        (words,) = [item for item in plot.items
                    if isinstance(item, pg.TextItem)]
        assert words.toPlainText().splitlines() == [
            '4600 Hz folds to 520 Hz', 'in the span',
            'low-passed after: unchanged',
            f'filtered first: {level_change(reading.folds[0].filtered_db)}']
    finally:
        layout.close()


def test_in_the_view_the_low_pass_after_lies_on_the_bare_curve_at_the_alias(
        qt_app):
    """The picture of a low-pass that came too late: at 520 Hz the
    dotted curve stands where the dashed one does, and the filtered
    curve is far below both."""
    import pyqtgraph as pg

    from visualdynamics.plot.resample import build_resample

    reading = read_resample(record(noise=0.01), RATE, frequencies=[4600.0])
    layout = pg.GraphicsLayoutWidget()
    try:
        build_resample(layout, reading)
        (plot,) = _plots(layout)
        by_name = {item.name(): item for item in plot.listDataItems()
                   if item.name()}

        def at(name, frequency):
            item = by_name[name]
            k = int(np.argmin(np.abs(item.xData - frequency)))
            return float(item.yData[k])

        bare = at('decimated with no filter', 520.0)
        after = at('decimated, then low-passed at 2400 Hz', 520.0)
        first = at('filtered at 2400 Hz, then decimated to 5120 Hz', 520.0)
        assert after == pytest.approx(bare, rel=1e-3)
        assert first < bare * 1e-3
        # drawn in the window's units: g^2/Hz in the default system
        assert bare == pytest.approx(reading.folds[0].bare / G ** 2,
                                     rel=1e-6)
    finally:
        layout.close()


def test_the_view_draws_to_a_file(qt_app, tmp_path):
    from visualdynamics.plot import plot_resample

    path = tmp_path / 'resample.png'
    assert plot_resample(record(), RATE, frequencies=[4600.0],
                         path=str(path), show=False) == str(path)
    assert path.exists() and path.stat().st_size > 2000


def test_the_summary():
    from visualdynamics.plot.resample import summary

    reading = read_resample(record(), RATE, frequencies=[4600.0])
    assert summary(reading).startswith('1Z+')
    assert summary(reading).endswith(reading.describe())


# ---- the window --------------------------------------------------------


def _resample_reading(window, pump, history=None, factor=16):
    """A record in the window, shown, on the Resample reading at
    `factor`."""
    name = window.add_object('Record', history or record(noise=0.001))
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    pane = window.data_pane
    assert pane.resample_action.isVisible()
    if not pane.resample_action.isChecked():
        pane.resample_action.trigger()
    pump()
    pane.resample_panel.factor_box.setValue(factor)
    pump()
    return pane


def test_a_time_history_offers_the_resample_reading(window, pump):
    """Choosing it puts the settings beside the spectrum and the
    reading in the panel and on the status line."""
    pane = _resample_reading(window, pump)
    panel = pane.resample_panel
    assert pane.resample_action.text() == 'Resample'
    assert panel.isVisible()
    assert len(_plots(pane.graphics)) == 1
    values = {key: label.text() for key, label in panel.values.items()}
    assert values == {
        'source': '81920 Hz', 'rate': '5120 Hz', 'nyquist': '2560 Hz',
        'span': '2000 Hz', 'lines': '400 of 512', 'dropped': '112',
        'resolution': '5 Hz', 'frame': '1024', 'at_span': '-0.1 dB',
        'lowest': '3120 Hz', 'at_lowest': '-55.1 dB'}
    assert panel.verdict.text().startswith(
        '4600 Hz folds to 520 Hz, inside the span: ')
    assert panel.verdict.text().endswith(
        'unchanged low-passed after decimating.')
    assert window._status_text.startswith(
        '1Z+: 81920 Hz to 5120 Hz (every 16th sample)')
    assert pane.resample_settings['factor'] == 16


def test_the_settings_redraw_the_reading_and_are_kept(window, pump):
    pane = _resample_reading(window, pump)
    panel = pane.resample_panel
    panel.order_box.setValue(6)
    panel.frame_box.setCurrentIndex(panel.frame_box.findData(2048))
    pump()
    assert panel.values['at_lowest'].text() == '-27.9 dB'
    assert panel.values['lines'].text() == '800 of 1024'
    assert pane.resample_settings == {'factor': 16, 'order': 6,
                                      'frame_length': 2048,
                                      'frequencies': None}
    # typed frequencies: one folding among the dropped lines
    panel.frequencies_edit.setText('2900, 4600')
    panel.frequencies_edit.editingFinished.emit()
    pump()
    assert pane.resample_settings['frequencies'] == (2900.0, 4600.0)
    lines = panel.verdict.text().splitlines()
    assert lines[0].startswith('2900 Hz folds to 2220 Hz, among the '
                               'dropped lines')
    assert lines[1].startswith('4600 Hz folds to 520 Hz, inside the span')
    # a typed frequency that does not fold is named under the folds
    panel.frequencies_edit.setText('1000, 4600')
    panel.frequencies_edit.editingFinished.emit()
    pump()
    lines = panel.verdict.text().splitlines()
    assert lines[0].startswith('4600 Hz folds to 520 Hz, inside the span')
    assert lines[1] == ('1000 Hz is below the new Nyquist frequency and '
                        'does not fold.')
    assert len(lines) == 2
    assert window._status_text.endswith(
        '1000 Hz is below the new Nyquist frequency and does not fold.')
    panel.frequencies_edit.setText('1000')
    panel.frequencies_edit.editingFinished.emit()
    pump()
    assert panel.verdict.text().splitlines() == [
        'Nothing above the new Nyquist frequency to fold.',
        '1000 Hz is below the new Nyquist frequency and does not fold.']
    panel.frequencies_edit.setText('2900, 4600')
    panel.frequencies_edit.editingFinished.emit()
    pump()
    # text that is not a list of frequencies is put back
    panel.frequencies_edit.setText('fast')
    panel.frequencies_edit.editingFinished.emit()
    pump()
    assert panel.frequencies_edit.text() == '2900, 4600'
    # and the settings survive clicking away and back
    other = window.add_object('Other', record(((300.0, 1.0),)))
    window.show_object(other)
    pump()
    window.show_object('Record')
    pump()
    assert panel.order_box.value() == 6
    assert panel.frequencies_edit.text() == '2900, 4600'


def test_the_resample_reading_is_journaled_as_its_plot_call(window, pump):
    pane = _resample_reading(window, pump)
    pane.resample_panel.frequencies_edit.setText('4600')
    pane.resample_panel.frequencies_edit.editingFinished.emit()
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_resample(')]
    assert lines == [
        ("visualdynamics.plot.plot_resample(project['Record'], 5120.0, 0, "
         "frequencies=(4600.0,), order=12, frame_length=1024, "
         "path='resample.png', show=False)")]


def test_a_factor_that_leaves_too_few_samples_says_so(window, pump):
    pane = _resample_reading(window, pump, record(samples=4096), factor=2)
    pane.resample_panel.factor_box.setValue(1024)
    pump()
    assert 'too few for a spectrum' in window._status_text
    assert pane.resample_panel.values['rate'].text() == '—'


def test_a_frame_longer_than_the_record_gives_way(window, pump):
    pane = _resample_reading(window, pump, record(samples=16 * 600))
    assert pane.resample_panel.values['frame'].text() == '512'
    assert 'frame_length' not in window.project.journal[-1]


def test_the_resample_reading_stands_the_trace_marks_down(window, pump):
    """It replaces the trace, so the averaging panel is put away and
    the 3-D toggle is not offered; another reading puts it away."""
    pane = _resample_reading(window, pump)
    assert not pane.averaging_panel.isVisibleTo(pane)
    assert not pane.waterfall_action.isVisible()
    pane.averaging_action.trigger()
    pump()
    assert not pane.resample_action.isChecked()
    assert not pane.resample_panel.isVisibleTo(pane)


def test_a_fold_reads_the_spike_within_a_line_of_the_frequency_given():
    """A tone at 4,604 Hz asked about as 4,600 Hz: its spike stands on
    the 4,605 Hz line, and the fold reads that spike, not the 4,600 Hz
    line on its shoulder."""
    reading = read_resample(record(((4604.0, 1.0),)), RATE,
                            frequencies=[4600.0])
    (fold,) = reading.folds
    f, g = reading.spectra['source']
    near = (f > 4590.0) & (f < 4615.0)
    assert fold.source == g[near].max()
    assert fold.source > g[np.argmin(np.abs(f - 4600.0))] * 2.0


# ---- the verb and its apply --------------------------------------------------

def test_the_project_resamples_as_the_record_does_and_records_how():
    """`Project.resample` is the one way the app and a script make the
    record (the reading had no apply and no verb, 2026-10-28): the same
    arrays as `TimeHistory.resample`, linked to the source, with a
    provenance a refresh can run again."""
    import visualdynamics

    project = visualdynamics.Project()
    project.add('Record', record())
    made = project.resample('Record', FINE / 4, order=8)
    assert made == f'Record at {FINE / 4:g} Hz'
    direct = project['Record'].resample(FINE / 4, order=8)
    np.testing.assert_array_equal(project[made].ordinate, direct.ordinate)
    assert project[made].sample_rate == pytest.approx(FINE / 4)
    assert project.provenance[made]['verb'] == 'resample'
    assert project.provenance[made]['params'] == {
        'rate': FINE / 4, 'anti_alias': True, 'order': 8}
    assert project.provenance[made]['source'] == 'Record'
    with pytest.raises(ValueError):
        project.resample('Record', FINE / 3.3)


def test_apply_resample_makes_the_record_at_the_panels_settings(window, pump):
    window.add_object('Record', record())
    window.show_object('Record')
    pump()
    pane = window.data_pane
    pane.resample_action.trigger()
    pump()
    pane.resample_panel.factor_box.setValue(8)
    pump()
    pane.resample_panel.apply_button.click()
    pump()
    added = [name for name in window.project if name.startswith('Record at')]
    assert added == [f'Record at {FINE / 8:g} Hz']
    assert window.project.provenance[added[0]]['verb'] == 'resample'
    assert window.project[added[0]].sample_rate == pytest.approx(FINE / 8)


# ---- dead, not finite, empty, and the edges -------------------------------


def test_a_sample_not_finite_is_refused_by_its_cause():
    """The filter runs forward and backward, so one sample that is not
    finite would make every sample of its channel so: refused, naming
    the channel and the count. Kept as it is when no filter runs."""
    noisy = record(noise=0.001)
    noisy.ordinate[0, 1000] = np.nan
    with pytest.raises(ValueError, match=r'1Z\+ holds 1 sample that is not '
                       'finite; the anti-alias filter would carry'):
        resample(noisy, RATE)
    with pytest.raises(ValueError, match='1 sample that is not finite'):
        read_resample(noisy, RATE)
    with pytest.warns(AliasWarning):
        bare = resample(noisy, RATE, anti_alias=False)
    assert int(np.sum(~np.isfinite(bare.ordinate))) == 0, \
        'sample 1000 is not one of every 16th'


def test_the_verb_refuses_a_sample_not_finite_and_adds_nothing():
    """`Project.resample` makes the record through `resample`, so the
    refusal is the verb's too, and no record is added beside it."""
    import visualdynamics

    noisy = record(noise=0.001)
    noisy.ordinate[0, 1000] = np.nan
    project = visualdynamics.Project()
    project.add('Record', noisy)
    with pytest.raises(ValueError, match='1 sample that is not finite'):
        project.resample('Record', RATE)
    assert list(project) == ['Record']


def test_a_dead_channel_reads_no_level_and_says_why():
    silent = TimeHistory(np.arange(16 * 4096) / FINE,
                         np.zeros((1, 16 * 4096)), response_dof=['1Z+'],
                         ordinate_dim=['acceleration'])
    reading = read_resample(silent, RATE, frequencies=[4600.0])
    fold, = reading.folds
    assert np.isnan(fold.filtered_db) and np.isnan(fold.after_db)
    assert reading.notes == (('1Z+ is zero throughout, so no level is read '
                              'where anything folds'),)


def test_a_record_with_no_line_above_the_new_nyquist_says_so():
    """A constant record holds nothing above 2,560 Hz but rounding, 300
    dB under its one line: nothing is folded, and the reading says so."""
    constant = TimeHistory(np.arange(16 * 4096) / FINE,
                           np.ones((1, 16 * 4096)), response_dof=['1Z+'],
                           ordinate_dim=['acceleration'])
    reading = read_resample(constant, RATE)
    assert reading.folds == ()
    assert reading.notes == (('the record holds no line above 2560 Hz, so '
                              'nothing folds'),)


def test_an_empty_record_is_refused():
    empty = TimeHistory(np.zeros(0), np.zeros((1, 0)), response_dof=['1Z+'])
    with pytest.raises(ValueError, match='two time samples'):
        resample(empty, RATE)


def test_a_fold_on_the_span_s_edge_is_inside_it():
    """5,120 + 2,000 Hz folds onto the span's top line, the last line
    an analyzer keeps: inside the span, at the edge."""
    reading = read_resample(record(((7120.0, 1.0),), noise=0.001), RATE,
                            frequencies=[7120.0])
    fold, = reading.folds
    assert fold.alias == pytest.approx(2000.0) and fold.in_span
    assert fold.filtered_db < -50.0


def test_a_long_channel_is_read_on_the_worker(window, pump, monkeypatch):
    """Past `RESAMPLE_INLINE_SAMPLES` the reading runs through
    `_run_long` with the bar up and is drawn as an inline one is."""
    from visualdynamics.gui import main_window as module

    ran = []
    original = window._run_long

    def spy(status, verb, *args, **kwargs):
        ran.append(status)
        return original(status, verb, *args, **kwargs)

    monkeypatch.setattr(window, '_run_long', spy)
    pane = _resample_reading(window, pump)
    assert ran == []
    monkeypatch.setattr(module, 'RESAMPLE_INLINE_SAMPLES', 1000)
    pane.resample_panel.factor_box.setValue(8)
    pump()
    assert ran and ran[-1] == 'Reading the resampling of 1Z+…'
    assert pane.resample_panel.values['rate'].text() == '10240 Hz'


# ---- the survivors of tools/mutate.py, each pinned -------------------------


def test_the_rules_hold_at_their_edges():
    assert decimation_factor(10240.0, 5120.0) == 2, 'every 2nd sample'
    # a rate read off an abscissa's step can be a few parts in ten
    # million off and is still a whole factor
    assert decimation_factor(FINE, RATE * (1.0 + 1e-7)) == 16
    with pytest.raises(ValueError, match='above zero'):
        decimation_factor(0.0, RATE)
    with pytest.raises(ValueError, match='above zero'):
        alias_frequency(100.0, 0.0)
    assert line_counts(2) == (1, 0, 1)
    with pytest.raises(ValueError, match='two samples'):
        line_counts(1)
    assert level_change(0.5) == '0.5 dB up'
    assert level_change(-0.5) == '0.5 dB down'
    assert ordinal(211) == '211th' and ordinal(201) == '201st'


def test_the_refusals_and_the_warning_name_the_nyquist_frequency():
    with pytest.raises(ValueError, match=r'above the Nyquist frequency '
                                         r'\(2560 Hz\)'):
        anti_alias_filter(RATE, ratio=2.1)
    with pytest.warns(AliasWarning, match=r'has folded into 0 to 2560 Hz'):
        resample(record(), RATE, anti_alias=False)


def test_a_record_of_odd_length_keeps_its_first_and_every_qth_sample():
    odd = record(samples=16 * 4096 + 1)
    slow = resample(odd, RATE)
    assert slow.ordinate.shape == (1, 4097)
    assert slow.abscissa[-1] == pytest.approx(odd.abscissa[-1])


def test_a_frequency_a_hair_under_the_new_nyquist_is_at_it():
    reading = read_resample(record(), RATE, frequencies=[2559.999, 2000.0])
    assert reading.notes == (
        '2560 Hz is at the new Nyquist frequency and does not fold',
        '2000 Hz is below the new Nyquist frequency and does not fold')


def test_with_nothing_asked_and_nothing_found_the_words_end_at_the_lines():
    reading = read_resample(record(), RATE, frequencies=[])
    assert reading.folds == () and reading.notes == ()
    assert reading.describe().endswith('400 of 512 lines kept.')


def test_the_first_of_several_channels_is_read_alone():
    one, two = record(((4600.0, 1.0),)), record(((3600.0, 1.0),))
    both = TimeHistory(one.abscissa, np.vstack([one.ordinate, two.ordinate]),
                       response_dof=['1Z+', '2Z+'],
                       ordinate_dim=['acceleration', 'acceleration'])
    reading = read_resample(both, RATE, 0)
    assert reading.label.startswith('1Z+')
    assert [fold.frequency for fold in reading.folds] == [4600.0]
    with pytest.raises(ValueError, match='no channel 2'):
        read_resample(both, RATE, 2)


def test_a_level_is_read_within_one_line_and_nowhere_else():
    from visualdynamics.core.resample import _level

    f = np.arange(0.0, 50.0) * 5.0
    g = np.ones(50)
    g[20] = 100.0
    assert _level((f, g), 100.0) == 100.0          # on the line
    assert _level((f, g), 95.0) == 100.0           # one line off
    assert _level((f, g), 90.0) == 1.0             # two lines off
    assert _level((f, g), 110.0) == 1.0
    g0 = np.ones(50)
    g0[0] = 7.0
    assert _level((f, g0), 0.0) == 7.0             # the first line
    assert _level((f, g), f[-1]) == 1.0            # the last line
    assert np.isnan(_level((f, g), f[-1] + 1.0))   # past the spectrum
    assert np.isnan(_level((f[:1], g[:1]), 0.0))   # one line, no spacing
    assert not np.isnan(_level((f[:2], g[:2]), 0.0))


def test_the_shortest_record_the_filter_allows_is_read():
    """Order 4: 32 decimated samples (`SHORTEST`) are enough, 31 are not.
    Order 30: the filter's padding asks for 94."""
    assert read_resample(record(samples=16 * 32), RATE, order=4).frame_length \
        == 32
    with pytest.raises(ValueError, match=r'keeps 31 samples, too few for a '
                                         r'spectrum \(at least 32\)'):
        read_resample(record(samples=16 * 31), RATE, order=4)
    with pytest.raises(ValueError, match=r'keeps 93 samples, too few for a '
                                         r'spectrum \(at least 94\)'):
        read_resample(record(samples=16 * 93), RATE, order=30)
    assert read_resample(record(samples=16 * 94), RATE,
                         order=30).frame_length == 64


def test_a_frame_the_length_of_the_record_fits_and_one_more_does_not():
    source = record(samples=16 * 1024)
    assert read_resample(source, RATE, frame_length=1024).frame_length == 1024
    with pytest.raises(ValueError, match=r'longer than the record \(1024 '
                                         r'samples there\)'):
        read_resample(source, RATE, frame_length=2048)
    assert read_resample(source, RATE, frame_length=2).frame_length == 2

