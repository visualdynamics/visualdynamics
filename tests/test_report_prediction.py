"""Responses against prediction: the analyst's predictions held in the
project, scaled to the level run, and each predicted peak read against
the measured one by the package's half-power reader.

Held to the one-mass resonator, where the answer is known: a response
ASD at minus 12 dB read against its full-level prediction scaled by
10^(-12/10), an amplitude by 10^(L/20), and a transmissibility unscaled;
the ratio named as an ASD ratio or an amplitude ratio and given in
decibels to match; and the bracket of the worked test, its Z mode found
at 410 Hz against 560 Hz predicted, which a band of plus or minus ten
percent does not reach and a band the analyst states does. What cannot
be read one way is refused with its cause, a channel that cannot be
compared says why, and the report draws nothing of it while the
feature is off.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.data import Frf, Psd, Specification, Spectrum
from visualdynamics.core.prediction import (
    Prediction,
    Predictions,
    compare,
    level_run,
    scale_factor,
)
from visualdynamics.core.report import Report, insert_options
from visualdynamics.io import native
from visualdynamics.project import Project
from visualdynamics.report import render_html

G = 9.80665
#: the input ASD of the worked examples, g²/Hz
INPUT = 0.04


def transmissibility(f, fn, q):
    """|response/base| of one mass on its spring riding a moving base."""
    zeta, r = 0.5 / q, np.asarray(f, dtype=float) / fn
    return np.sqrt((1 + (2 * zeta * r) ** 2)
                   / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2))


def response_psd(modes, level_db=-12.0, df=2.5, top=2000.0, dofs=None):
    """Response ASDs in SI over a flat input of `INPUT` g²/Hz at
    `level_db`, one record per (fn, q)."""
    f = np.arange(0.0, top, df)
    rows = [INPUT * G ** 2 * transmissibility(f, fn, q) ** 2
            * 10 ** (level_db / 10) for fn, q in modes]
    dofs = dofs or [f'{201 + k}Y+' for k in range(len(modes))]
    return Psd(f, np.vstack(rows), response_dof=dofs,
               ordinate_dim='acceleration**2/frequency',
               ordinate_unit='m/s**2')


def asd_prediction(channel='201Y+', fn=385.0, q=12.0, **kw):
    """A full-level ASD prediction of one mode, in g²/Hz."""
    peak = INPUT * float(transmissibility(fn, fn, q)) ** 2
    fields = {'unit': 'g', 'level_db': 0.0, 'frequency_uncertainty': 10.0,
              'source': 'Stress analysis SR-12 rev B', **kw}
    return Prediction(channel, 'asd', [(fn, peak, q)], **fields)


# ---- the scaling ------------------------------------------------------------


def test_the_scaling_is_ten_to_the_level_over_ten_on_an_asd_and_twenty_on_an_amplitude():
    # minus 12 dB: a sixteenth of the ASD (near enough), a quarter of the
    # amplitude, as the course works it
    assert scale_factor('asd', -12.0) == pytest.approx(0.0630957344, rel=1e-9)
    assert scale_factor('amplitude', -12.0) == pytest.approx(0.2511886432,
                                                            rel=1e-9)
    assert scale_factor('asd', -10.0) == pytest.approx(0.1, rel=1e-12)
    assert scale_factor('amplitude', -20.0) == pytest.approx(0.1, rel=1e-12)
    assert scale_factor('amplitude', 6.0) == pytest.approx(1.9952623, rel=1e-7)
    assert scale_factor('asd', 0.0) == 1.0
    with pytest.raises(ValueError, match='transmissibility is a ratio'):
        scale_factor('transmissibility', -12.0)
    with pytest.raises(ValueError, match='not a finite number'):
        scale_factor('asd', math.nan)
    with pytest.raises(ValueError, match="not 'psd'"):
        scale_factor('psd', 0.0)
    with pytest.raises(TypeError, match='a number'):
        scale_factor('asd', '-12')


def test_an_asd_run_at_minus_12_db_reads_against_its_scaled_prediction():
    measured = response_psd([(385.0, 12.0)])
    comparison = compare(Predictions([asd_prediction()]), measured, -12.0)
    (channel, row), = comparison.rows
    assert comparison.ratio_name == 'ASD ratio'
    assert channel.factor == pytest.approx(10 ** -1.2)
    full = INPUT * float(transmissibility(385.0, 385.0, 12.0)) ** 2
    assert row.predicted_level == pytest.approx(full * G ** 2 * 10 ** -1.2)
    # the measured peak: its frequency within half a line, its Q read on
    # the square root of the ASD, whose half power is half its height
    assert row.measured_frequency == pytest.approx(385.0, abs=1.25)
    assert row.measured_q == pytest.approx(12.0, rel=0.02)
    assert row.predicted_q == 12.0
    assert row.ratio == pytest.approx(1.0, abs=0.02)
    assert row.ratio_db == pytest.approx(10 * math.log10(row.ratio))
    assert comparison.scaling == ('scaled to the level run, -12 dB (stated), '
                                  'as 10^(L/10) on the ASD')


def test_the_q_of_an_asd_is_read_at_half_its_height_not_at_0707():
    """Read at 0.707 of the ASD's own height, the width is that of the
    amplitude's 0.84 points and Q comes out about 1.55 times high."""
    measured = response_psd([(385.0, 12.0)], df=0.5)
    (_channel, row), = compare([asd_prediction()], measured, -12.0).rows
    assert row.measured_q == pytest.approx(12.0, rel=0.01)


def test_a_prediction_stated_below_full_level_is_scaled_by_the_difference():
    measured = response_psd([(385.0, 12.0)], level_db=-6.0)
    stated_at = asd_prediction(level_db=-12.0)
    (channel, row), = compare([stated_at], measured, -6.0).rows
    assert channel.factor == pytest.approx(10 ** 0.6)
    # a prediction stated at -12 dB that holds the full-level number is
    # scaled up by 6 dB, and the run at -6 dB reads 12 dB under it
    assert row.ratio == pytest.approx(10 ** -1.2, rel=0.02)


def test_an_amplitude_is_scaled_by_twenty_and_named_an_amplitude_ratio():
    f = np.arange(20.0, 1000.0, 1.0)
    amplitude = 0.5 * G * transmissibility(f, 300.0, 20.0) * 10 ** (-6 / 20)
    measured = Spectrum(f, amplitude[None], response_dof=['5Z+'],
                        ordinate_dim='acceleration', ordinate_unit='m/s**2')
    prediction = Prediction('5Z+', 'amplitude', [(300.0, 10.0, 20.0)],
                            unit='g', level_db=0.0,
                            frequency_uncertainty=5.0)
    comparison = compare([prediction], measured, -6.0)
    (channel, row), = comparison.rows
    assert comparison.ratio_name == 'amplitude ratio'
    assert channel.factor == pytest.approx(10 ** (-6 / 20))
    assert row.predicted_level == pytest.approx(10.0 * G * 10 ** (-6 / 20))
    peak = 0.5 * float(np.max(transmissibility(
        np.linspace(290, 310, 200001), 300.0, 20.0)))
    assert row.measured_level == pytest.approx(peak * G * 10 ** (-6 / 20),
                                               rel=0.005)
    assert row.ratio_db == pytest.approx(20 * math.log10(row.ratio))
    assert 'as 10^(L/20) on the amplitude' in comparison.scaling


# ---- the worked test's bracket ------------------------------------------------


def bracket(q=14.8):
    """The bracket's transmissibility from the control point in Z: a mode
    at 410 Hz."""
    f = np.arange(2.5, 2000.0, 0.5)
    return Frf(f, transmissibility(f, 410.0, q)[None],
               response_dof=['301Z+'], reference_dof=['1Z+'],
               ordinate_dim='acceleration/acceleration',
               ordinate_unit='m/s**2', reference_unit='m/s**2')


def test_the_bracket_mode_far_from_its_prediction_is_not_matched_to_a_distant_peak():
    predicted = Prediction('301Z+', 'transmissibility', [(560.0, 8.0, 8.0)],
                           reference='1Z+', frequency_uncertainty=10.0)
    (_channel, row), = compare([predicted], bracket()).rows
    assert row.measured_frequency is None and row.ratio is None
    assert row.missing == 'no peak of the record between 504 and 616 Hz'
    assert row.band == pytest.approx((504.0, 616.0))


def test_the_bracket_mode_in_a_band_the_analyst_states_reads_its_ratio():
    """410 Hz and 14.8 against 560 Hz and 8: 26.8 percent low and about
    5.3 dB high, an amplitude ratio."""
    predicted = Prediction('301Z+', 'transmissibility',
                           [(560.0, 8.0, 8.0, (380.0, 600.0))],
                           reference='1Z+')
    comparison = compare([predicted], bracket())
    (channel, row), = comparison.rows
    assert comparison.ratio_name == 'amplitude ratio'
    assert comparison.level_db is None and channel.factor == 1.0
    fine = np.linspace(400.0, 420.0, 400001)
    curve = transmissibility(fine, 410.0, 14.8)
    assert row.measured_frequency == pytest.approx(
        fine[np.argmax(curve)], abs=0.05)
    assert row.measured_level == pytest.approx(curve.max(), rel=0.002)
    assert row.frequency_difference == pytest.approx(
        (row.measured_frequency - 560.0) / 560.0 * 100.0)
    assert row.frequency_difference == pytest.approx(-26.8, abs=0.1)
    assert row.ratio == pytest.approx(curve.max() / 8.0, rel=0.002)
    assert row.ratio_db == pytest.approx(20 * math.log10(curve.max() / 8.0),
                                         abs=0.02)
    assert row.ratio_db == pytest.approx(5.35, abs=0.05)
    assert 'not scaled' in comparison.scaling


def test_a_transmissibility_takes_no_level_and_no_unit():
    with pytest.raises(ValueError, match='not scaled; leave the level'):
        compare([Prediction('301Z+', 'transmissibility', [(410, 14.8)],
                            reference='1Z+', frequency_uncertainty=10)],
                bracket(), -12.0)
    with pytest.raises(ValueError, match='takes no unit'):
        Prediction('301Z+', 'transmissibility', [(410, 14.8)], unit='g',
                   reference='1Z+', frequency_uncertainty=10)
    with pytest.raises(ValueError, match='takes no input level'):
        Prediction('301Z+', 'transmissibility', [(410, 14.8)], level_db=0,
                   reference='1Z+', frequency_uncertainty=10)
    with pytest.raises(ValueError, match='names its input DOF'):
        Prediction('301Z+', 'transmissibility', [(410, 14.8)],
                   frequency_uncertainty=10)


def test_a_transmissibility_is_paired_by_its_input_dof_too():
    other_input = Prediction('301Z+', 'transmissibility', [(410, 14.8)],
                             reference='2Z+', frequency_uncertainty=10)
    comparison = compare([other_input], bracket())
    assert comparison.channels == () and comparison.absent == ('301Z+/2Z+',)
    assert comparison.lines()[-1] == 'Nothing was compared.'


# ---- what cannot be read one way ----------------------------------------------


@pytest.mark.parametrize('fields, message', [
    ({'unit': None}, 'states the unit'),
    ({'unit': 'blorps'}, 'cannot be read'),
    ({'level_db': None}, 'input level the prediction is stated at is not given'),
    ({'frequency_uncertainty': None}, 'no band to look for the peak at 385 Hz'),
    ({'frequency_uncertainty': 100.0}, 'leaves no band'),
    ({'frequency_uncertainty': -5.0}, 'must be above zero'),
    ({'level_uncertainty': 0.0}, 'must be above zero'),
    ({'reference': '1Z+'}, 'only a transmissibility'),
    ({'channel': '201Q+'}, 'names no direction'),
])
def test_a_prediction_that_cannot_be_read_one_way_is_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        asd_prediction(**fields)


def test_a_peak_needs_a_level_a_positive_frequency_and_a_band_holding_it():
    base = {'unit': 'g', 'level_db': 0.0, 'frequency_uncertainty': 10.0}
    with pytest.raises(ValueError, match='names no peak'):
        Prediction('201Y+', 'asd', [], **base)
    with pytest.raises(ValueError, match='gives no level'):
        Prediction('201Y+', 'asd', [385.0], **base)
    with pytest.raises(ValueError, match='peak frequency.*above zero'):
        Prediction('201Y+', 'asd', [(0.0, 0.5)], **base)
    with pytest.raises(ValueError, match='the level.*above zero'):
        Prediction('201Y+', 'asd', [(385.0, -0.5)], **base)
    with pytest.raises(ValueError, match='Q.*above zero'):
        Prediction('201Y+', 'asd', [(385.0, 0.5, 0.0)], **base)
    with pytest.raises(ValueError, match='does not hold the peak'):
        Prediction('201Y+', 'asd', [(385.0, 0.5, 12, (390.0, 420.0))], **base)
    with pytest.raises(ValueError, match='5 values'):
        Prediction('201Y+', 'asd', [(385.0, 0.5, 12, None, 1)], **base)
    with pytest.raises(ValueError, match="no field 'width'"):
        Prediction('201Y+', 'asd', [{'frequency': 385.0, 'width': 3}], **base)
    with pytest.raises(TypeError, match='not a str'):
        Prediction('201Y+', 'asd', '385', **base)
    # a band stated for the peak stands without the model's uncertainty
    held = Prediction('201Y+', 'asd', [(385.0, 0.5, 12, (370.0, 400.0))],
                      unit='g', level_db=0.0)
    assert held.bands == ((370.0, 400.0),)


def test_two_peaks_whose_bands_overlap_are_refused():
    base = {'unit': 'g', 'level_db': 0.0, 'frequency_uncertainty': 10.0}
    with pytest.raises(ValueError, match='385 and 420 Hz overlap'):
        Prediction('201Y+', 'asd', [(420.0, 0.3), (385.0, 0.5)], **base)
    # bands that only come close are two peaks
    held = Prediction('201Y+', 'asd', [(385.0, 0.5), (700.0, 0.3)], **base)
    assert len(held.bands) == 2


def test_a_curve_names_its_modes_and_its_levels_are_read_off_it():
    f = np.arange(10.0, 2000.0, 1.0)
    curve = INPUT * transmissibility(f, 385.0, 12.0) ** 2
    base = {'unit': 'g', 'level_db': 0.0, 'frequency_uncertainty': 10.0,
            'curve': (f, curve)}
    with pytest.raises(ValueError, match='read off the curve'):
        Prediction('201Y+', 'asd', [(385.0, 0.5)], **base)
    with pytest.raises(ValueError, match='at the mode named at 900 Hz, no '
                       'peak of the predicted curve between 810 and 990 Hz'):
        Prediction('201Y+', 'asd', [900.0], **base)
    with pytest.raises(ValueError, match='three lines'):
        Prediction('201Y+', 'asd', [385.0], **{**base, 'curve': ([1, 2],
                                                                 [1, 2])})
    with pytest.raises(ValueError, match='below zero'):
        Prediction('201Y+', 'asd', [385.0], **{**base,
                                               'curve': (f, -curve)})
    with pytest.raises(ValueError, match='ascending'):
        Prediction('201Y+', 'asd', [385.0], **{**base,
                                               'curve': (f[::-1], curve)})
    prediction = Prediction('201Y+', 'asd', [385.0], **base)
    measured = response_psd([(385.0, 12.0)])
    (_channel, row), = compare([prediction], measured, -12.0).rows
    assert row.predicted_frequency == pytest.approx(385.0, abs=1.0)
    assert row.predicted_q == pytest.approx(12.0, rel=0.01)
    assert row.predicted_level == pytest.approx(
        curve.max() * G ** 2 * 10 ** -1.2, rel=0.002)
    assert row.ratio == pytest.approx(1.0, abs=0.02)


def test_the_measured_object_must_be_a_measured_response():
    measured = response_psd([(385.0, 12.0)])
    spec = Specification(measured.abscissa, measured.ordinate,
                         response_dof=measured.response_dof,
                         ordinate_dim=measured.ordinate_dim)
    with pytest.raises(TypeError, match='requirement, not a measured'):
        compare([asd_prediction()], spec, -12.0)
    banded = response_psd([(385.0, 12.0)])
    banded.bandwidth = np.ones(len(banded.abscissa))
    with pytest.raises(ValueError, match='octave-band PSD'):
        compare([asd_prediction()], banded, -12.0)
    with pytest.raises(ValueError, match='no predictions'):
        compare(Predictions(), measured, -12.0)
    with pytest.raises(TypeError, match='a Prediction'):
        compare(['201Y+'], measured, -12.0)


def test_a_run_level_not_stated_is_refused_and_one_held_is_used():
    measured = response_psd([(385.0, 12.0)])
    with pytest.raises(ValueError, match='level the record was run at is '
                       'not stated'):
        compare([asd_prediction()], measured)
    assert level_run(measured) == (None, '')
    # the scaling a PSD holds for its specification: +12 to bring a run
    # at -12 dB up
    measured.scale_db = 12
    assert level_run(measured) == (-12.0, 'held on the record')
    comparison = compare([asd_prediction()], measured)
    assert comparison.level_db == -12.0
    assert '(held on the record)' in comparison.scaling
    assert compare([asd_prediction()], measured, -12.0).level_db == -12.0
    with pytest.raises(ValueError, match=r'-6 dB, disagrees .* \+12 dB'):
        compare([asd_prediction()], measured, -6.0)


# ---- a channel that cannot be compared says why ---------------------------


def test_each_channel_that_cannot_be_compared_says_why():
    measured = response_psd([(385.0, 12.0)] * 5,
                            dofs=['201Y+', '202Y+', '203Y+', '204Y+',
                                  '205Y+'])
    measured.ordinate[2] = 0.0
    measured.ordinate[3] = 0.25
    measured.ordinate_dim[4] = 'unknown'
    amplitude = Prediction('201Y+', 'amplitude', [(385.0, 2.0)], unit='g',
                           level_db=0.0, frequency_uncertainty=10.0)
    force = Prediction('202Y+', 'asd', [(385.0, 2.0)], unit='lbf',
                       level_db=0.0, frequency_uncertainty=10.0)
    comparison = compare([amplitude, force, asd_prediction('203Y+'),
                          asd_prediction('204Y+'), asd_prediction('205Y+'),
                          asd_prediction('299Y+')], measured, -12.0)
    notes = {channel.label: channel.note for channel in comparison.channels}
    assert notes == {
        '201Y+': 'the prediction is an amplitude and the record is an ASD',
        '202Y+': 'the record is acceleration**2/frequency and the '
                 'prediction force**2/frequency',
        '203Y+': 'the record is zero at every line',
        '204Y+': 'the record is constant at every line',
        '205Y+': "the record's units are not defined"}
    assert comparison.rows == []
    assert comparison.absent == ('299Y+',)
    lines = comparison.lines()
    assert lines[0] == ('201Y+: not compared, the prediction is an amplitude '
                        'and the record is an ASD')
    assert lines[-2:] == ['No record of 299Y+.', 'Nothing was compared.']


def test_two_records_of_one_channel_are_not_chosen_between():
    measured = response_psd([(385.0, 12.0), (390.0, 12.0)],
                            dofs=['201Y+', '201Y+'])
    (channel,) = compare([asd_prediction()], measured, -12.0).channels
    assert channel.note == '2 records of 201Y+; compare one'


def test_an_all_nan_record_and_a_cross_spectrum_are_not_read():
    measured = response_psd([(385.0, 12.0)])
    measured.ordinate[0] = np.nan
    (channel,) = compare([asd_prediction()], measured, -12.0).channels
    assert channel.note == 'no line of the record has a value'
    cross = response_psd([(385.0, 12.0)])
    cross.reference_dof = ['1Y+']
    assert compare([asd_prediction()], cross, -12.0).absent == ('201Y+',)


def test_one_nan_line_inside_the_band_is_named_and_one_outside_is_not_read():
    measured = response_psd([(385.0, 12.0)])
    f = measured.abscissa
    inside = int(np.argmin(np.abs(f - 400.0)))
    outside = int(np.argmin(np.abs(f - 1500.0)))
    measured.ordinate[0, outside] = np.nan
    (_channel, row), = compare([asd_prediction()], measured, -12.0).rows
    assert row.measured_frequency == pytest.approx(385.0, abs=1.25)
    measured.ordinate[0, inside] = np.nan
    (_channel, row), = compare([asd_prediction()], measured, -12.0).rows
    assert row.measured_frequency is None
    assert row.missing == ('the record has no value at 400 Hz, inside the '
                           'band 346.5 and 423.5 Hz')


def test_a_peak_at_the_first_line_or_beyond_the_record_has_no_counterpart():
    # the record starts on the mode: the band holds a falling flank only
    measured = response_psd([(385.0, 12.0)])
    measured.abscissa = measured.abscissa + 385.0
    (_channel, row), = compare([asd_prediction()], measured, -12.0).rows
    assert row.missing == 'no peak of the record between 346.5 and 423.5 Hz'
    # the band only partly covered: the record stops below the mode
    short = response_psd([(385.0, 12.0)], top=380.0)
    (_channel, row), = compare([asd_prediction()], short, -12.0).rows
    assert row.missing == 'no peak of the record between 346.5 and 423.5 Hz'
    # and wholly outside it
    low = response_psd([(385.0, 12.0)], top=300.0)
    (_channel, row), = compare([asd_prediction()], low, -12.0).rows
    assert row.missing == ('no line of the record lies between 346.5 and '
                           '423.5 Hz')


def test_a_peak_too_narrow_for_its_lines_reads_with_the_readers_warning():
    measured = response_psd([(385.0, 60.0)], df=5.0)
    (_channel, row), = compare([asd_prediction(q=60.0)], measured,
                               -12.0).rows
    assert row.measured_frequency is not None
    assert any('fewer than 4 lines' in note for note in row.notes)


def test_a_sine_level_below_its_floor_is_not_read():
    from visualdynamics.core.sine import SineLevel

    f = np.arange(100.0, 500.0, 1.0)
    level = SineLevel(f, (G * transmissibility(f, 300.0, 20.0))[None],
                      response_dof=['5Z+'], ordinate_dim='acceleration',
                      ordinate_unit='m/s**2',
                      below_floor=(np.abs(f - 300.0) < 2.0)[None])
    prediction = Prediction('5Z+', 'amplitude', [(300.0, 20.0)], unit='g',
                            level_db=0.0, frequency_uncertainty=5.0)
    (_channel, row), = compare([prediction], level, 0.0).rows
    assert row.measured_frequency is None
    assert row.missing.startswith('the record has no value at 299 Hz')


def test_simulated_noise_through_the_mode_reads_its_prediction():
    """An independent truth: white noise through one mode on a moving
    base, its response ASD estimated by the package under the default
    Hann window, read against the prediction."""
    from scipy import signal

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory

    fs, fn, q = 4096.0, 385.0, 12.0
    rng = np.random.default_rng(25)
    base = rng.standard_normal(int(fs * 240))
    # the natural frequency warped ahead, so the digital filter's mode
    # sits at fn
    wn, zeta = 2 * fs * np.tan(np.pi * fn / fs), 0.5 / q
    b, a = signal.bilinear([2 * zeta * wn, wn ** 2],
                           [1.0, 2 * zeta * wn, wn ** 2], fs)
    response = signal.lfilter(b, a, base)
    history = TimeHistory(np.arange(len(base)) / fs,
                          np.vstack([base, response]) * G,
                          response_dof=['1Y+', '201Y+'],
                          ordinate_dim='acceleration')
    frames = (len(base) - 4096) // 2048 + 1
    history.averaging = Averaging(4096, overlap=0.5, frames=frames)
    psds = history.compute_psds()
    density = float(np.median(psds.ordinate[0])) / G ** 2      # g²/Hz
    peak = density * float(transmissibility(fn, fn, q)) ** 2
    prediction = Prediction('201Y+', 'asd', [(fn, peak, q)], unit='g',
                            level_db=0.0, frequency_uncertainty=10.0)
    (_channel, row), = compare([prediction], psds, 0.0).rows
    # within the scatter of 479 Hann frames across a peak 32 lines wide
    assert row.measured_frequency == pytest.approx(fn, rel=0.01)
    assert row.measured_q == pytest.approx(q, rel=0.12)
    assert row.ratio == pytest.approx(1.0, abs=0.2)


# ---- the predictions object, saved ----------------------------------------


def test_predictions_hold_one_per_channel_and_quantity():
    first = asd_prediction()
    with pytest.raises(ValueError, match='two predictions for 201Y\\+'):
        Predictions([first, asd_prediction(fn=400.0)])
    held = Predictions([first])
    held.put(asd_prediction(fn=400.0))
    assert len(held) == 1 and held.predictions[0].peaks[0].frequency == 400.0
    held.put(asd_prediction('202Y+'))
    assert held.channels == ['201Y+', '202Y+']
    with pytest.raises(TypeError, match='holds Prediction'):
        held.put('201Y+')
    assert first.describe() == (
        '201Y+, ASD in g²/Hz at +0 dB: 385 Hz 5.8 (Q 12); from Stress '
        'analysis SR-12 rev B')


def test_a_project_with_predictions_saves_and_reads_back_stamped_schema_2(
        tmp_path):
    f = np.arange(10.0, 2000.0, 1.0)
    project = Project('bracket')
    project.add('Response PSD', response_psd([(385.0, 12.0)]))
    project.predict('Predictions', '201Y+', 'asd', [(385.0, 5.78, 12.0)],
                    unit='g', level_db=0.0, frequency_uncertainty=10.0,
                    level_uncertainty=3.0, source='SR-12 rev B',
                    customer=True)
    project.predict('Predictions', '202Y+', 'asd', [700.0], unit='g',
                    level_db=-3.0,
                    curve=(f, INPUT * transmissibility(f, 700.0, 7.0) ** 2),
                    frequency_uncertainty=5.0)
    project.predict('Predictions', '301Z+', 'transmissibility',
                    [(560.0, 8.0, 8.0, (380.0, 600.0))], reference='1Z+')
    path = tmp_path / 'bracket.vdyn'
    native.save_test(path, project.name, project)
    back = native.load(path)
    held, again = project['Predictions'], back['Predictions']
    assert [p.as_record() for p in again] == [p.as_record() for p in held]
    np.testing.assert_array_equal(again.predictions[1].curve[1],
                                  held.predictions[1].curve[1])
    assert again.predictions[2].bands == ((380.0, 600.0),)
    import h5py

    with h5py.File(path, 'r') as file:
        assert file.attrs['visualdynamics_schema'] == 2
    # a file with nothing version 2 added is stamped 1, as before
    del project['Predictions']
    plain = tmp_path / 'plain.vdyn'
    native.save_test(plain, project.name, project)
    with h5py.File(plain, 'r') as file:
        assert file.attrs['visualdynamics_schema'] == 1


def test_a_build_reading_only_schema_1_refuses_a_file_with_predictions(
        tmp_path, monkeypatch):
    held = Predictions([asd_prediction()])
    path = tmp_path / 'held.vdyn'
    native.save(held, path)
    assert len(native.load(path)) == 1
    monkeypatch.setattr(native, 'SCHEMA_VERSION', 1)
    with pytest.raises(ValueError, match='newer Visual Dynamics.*schema 2.*'
                       'Update to open it'):
        native.load(path)
    # and a report reading predictions, saved alone, is stamped 2 too
    report = Report('R', [{'kind': 'prediction', 'source': 'P',
                           'measured': 'M', 'level': -12.0}])
    assert native.schema_for(report) == 2
    assert native.schema_for(Report('R')) == 1


# ---- the project verbs ----------------------------------------------------


def test_the_verbs_are_journaled_replay_and_refuse_in_the_core():
    project = Project('t')
    project.add('Response PSD', response_psd([(385.0, 12.0)]))
    project.add('Report', Report('R'))
    assert project.predict('Predictions', '201Y+', 'asd', [(385, 5.78, 12)],
                           unit='g', level_db=0,
                           frequency_uncertainty=10) == 'Predictions'
    line = project.journal[-1]
    assert line == ("project.predict('Predictions', '201Y+', 'asd', "
                    "[(385, 5.78, 12)], unit='g', level_db=0, "
                    "frequency_uncertainty=10)")
    before = list(project.journal)
    with pytest.raises(ValueError, match='no band'):
        project.predict('Predictions', '202Y+', 'asd', [(385, 5.78)],
                        unit='g', level_db=0)
    with pytest.raises(ValueError, match='level the record was run at'):
        project.report_prediction('Report', 'Response PSD', 'Predictions')
    with pytest.raises(TypeError, match='not a predictions object'):
        project.predict('Report', '201Y+', 'asd', [(385, 5.78)], unit='g',
                        level_db=0, frequency_uncertainty=10)
    with pytest.raises(TypeError, match='not a predictions object'):
        project.report_prediction('Report', 'Response PSD', 'Response PSD',
                                  level=-12)
    assert project.journal == before, 'a refusal is not journaled'
    assert project['Report'].num_blocks == 0, 'and changes nothing'
    assert project.report_prediction('Report', 'Response PSD', 'Predictions',
                                     level=-12) == 0
    assert project['Report'].blocks[0] == {
        'kind': 'prediction', 'source': 'Predictions',
        'measured': 'Response PSD', 'level': -12.0, 'caption': ''}
    assert project.journal[-1] == ("project.report_prediction('Report', "
                                   "'Response PSD', 'Predictions', level=-12)")
    # the predict line replays into a project of its own
    replay = Project('t')
    exec(line, {'project': replay})                     # noqa: S102
    assert replay['Predictions'].predictions[0].as_record() == \
        project['Predictions'].predictions[0].as_record()
    assert project.predictions is project['Predictions']
    assert visualdynamics.project.describe(project['Predictions']) == \
        '1 channel'


def test_renaming_the_measurement_keeps_the_block_on_it():
    project = Project('t')
    project.add('Response PSD', response_psd([(385.0, 12.0)]))
    project.add('Report', Report('R'))
    project.predict('Predictions', '201Y+', 'asd', [(385, 5.78, 12)],
                    unit='g', level_db=0, frequency_uncertainty=10)
    project.report_prediction('Report', 'Response PSD', 'Predictions',
                              level=-12)
    project.rename('Response PSD', 'Run 3 PSD')
    project.rename('Predictions', 'Analyst')
    assert project['Report'].blocks[0]['measured'] == 'Run 3 PSD'
    assert project['Report'].blocks[0]['source'] == 'Analyst'


# ---- the report -----------------------------------------------------------


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


@pytest.fixture
def bracket_project():
    project = Project('t')
    project.add('Response PSD', response_psd([(385.0, 12.0), (410.0, 14.0)]))
    project.predict('Predictions', '201Y+', 'asd',
                    [(385.0, INPUT * 144.0, 12.0)], unit='g', level_db=0,
                    frequency_uncertainty=10, level_uncertainty=3,
                    source='Stress analysis SR-12 rev B', customer=True)
    project.predict('Predictions', '202Y+', 'asd',
                    [(560.0, INPUT * 64.0, 8.0)], unit='g', level_db=0,
                    frequency_uncertainty=10,
                    source='Stress analysis SR-12 rev B', customer=True)
    project.add('Report', Report('R', [{'kind': 'text', 'text': 'Run 3.'}]))
    project.report_prediction('Report', 'Response PSD', 'Predictions',
                              level=-12)
    return project


def _render(project, name='Report', **kw):
    return render_html(project[name], project, visualdynamics.SI,
                       object_groups=project.object_groups, **kw)


def test_off_the_block_draws_nothing_and_the_page_is_unchanged(
        bracket_project):
    report = bracket_project['Report']
    with_block = _render(bracket_project)
    without = Report('R', [{'kind': 'text', 'text': 'Run 3.'}])
    bracket_project.add('Plain', without)
    assert with_block == _render(bracket_project, 'Plain')
    assert [b['kind'] for b in _payload(with_block)['blocks']] == ['text']
    # nor on the edit page
    edit = _payload(_render(bracket_project, edit=True))
    assert [b['kind'] for b in edit['blocks']] == ['text']
    assert report.features == []


def test_on_a_figure_per_channel_then_the_table_and_its_note(bracket_project):
    bracket_project.report_features('Report', enable=['prediction'])
    blocks = _payload(_render(bracket_project))['blocks']
    assert [b['kind'] for b in blocks] == ['text', 'plot', 'plot', 'table',
                                           'text']
    figure = blocks[1]
    assert figure['label'] == 'Figure 1' and blocks[2]['label'] == 'Figure 2'
    assert figure['caption'] == ('201Y+: measured ASD over its prediction, '
                                 'scaled to -12 dB')
    assert figure['logx'] and figure['logy']
    measured, predicted = figure['curves']
    assert measured['ink'] and predicted['dash']
    assert predicted['label'] == 'prediction, scaled to -12 dB'
    # the peak alone is drawn across its band at its scaled level, in SI
    level = INPUT * 144.0 * G ** 2 * 10 ** -1.2
    assert predicted['x'] == pytest.approx([math.log10(346.5),
                                            math.log10(423.5)], rel=1e-5)
    assert predicted['y'] == pytest.approx([math.log10(level)] * 2, rel=1e-5)
    table = blocks[3]
    assert table['label'] == 'Table 1'
    assert table['caption'] == ('Responses against prediction: Response PSD, '
                                'scaled to the level run, -12 dB (stated), as '
                                '10^(L/10) on the ASD')
    assert table['headers'] == [
        'Channel', 'Predicted [Hz]', 'Predicted level [(m/s²)²/Hz]',
        'Predicted Q', 'Measured [Hz]', 'Measured level [(m/s²)²/Hz]',
        'Measured Q', 'Frequency difference [%]', 'ASD ratio (dB)',
        'Model uncertainty']
    first, second = table['rows']
    assert first[:4] == ['201Y+', '385', f'{level:.3g}', '12']
    assert first[6] == '12' and first[9] == '±10 %, ±3 dB'
    assert first[8].startswith('1.0') and first[8].endswith(' dB)')
    assert second[4:9] == ['—'] * 5 and second[9] == '±10 %'
    note = blocks[4]['html']
    assert ('Predictions for 201Y+, 202Y+: Stress analysis SR-12 rev B, '
            'supplied by the customer.') in note
    assert '202Y+ at 560 Hz: no peak of the record between 504 and 616 Hz.' \
        in note
    # the analyst's answer is not the program's
    page = _render(bracket_project).lower()
    for word in ('as predicted', 'explained', 'pass', 'not met'):
        assert word not in page.split('<script id="data"')[1].split(
            '</script>')[0], word


def test_the_table_reads_in_the_reports_units(bracket_project):
    bracket_project.report_features('Report', enable=['prediction'])
    html = render_html(bracket_project['Report'], bracket_project,
                       visualdynamics.DEFAULT_SYSTEM)
    table = next(b for b in _payload(html)['blocks'] if b['kind'] == 'table')
    assert 'g²/Hz' in table['headers'][2]
    assert table['rows'][0][2] == f'{INPUT * 144.0 * 10 ** -1.2:.3g}'


def test_a_refused_comparison_says_nothing_was_compared(bracket_project):
    bracket_project.report_features('Report', enable=['prediction'])
    bracket_project['Report'].blocks[1]['level'] = None
    blocks = _payload(_render(bracket_project))['blocks']
    assert [b['kind'] for b in blocks] == ['text', 'text']
    assert blocks[1]['html'] == (
        '<p><strong>Responses against prediction:</strong> nothing compared; '
        'the level the record was run at is not stated; an ASD prediction '
        'is scaled to it (dB relative to full level, 0 for a full-level '
        'run).</p>')
    bracket_project['Report'].blocks[1]['level'] = '-12'
    blocks = _payload(_render(bracket_project))['blocks']
    assert 'a number of dB' in blocks[1]['html']


def test_an_unbound_block_is_left_out_of_the_reading_page(bracket_project):
    bracket_project.report_features('Report', enable=['prediction'])
    bracket_project['Report'].blocks[1]['measured'] = 'Gone'
    blocks = _payload(_render(bracket_project))['blocks']
    assert [b['kind'] for b in blocks] == ['text']
    edit = _payload(_render(bracket_project, edit=True))['blocks']
    assert edit[1]['kind'] == 'unbound' and edit[1]['was'] == 'prediction'


def test_under_the_skeleton_it_falls_in_results(bracket_project):
    from visualdynamics.report.skeleton import check

    report = bracket_project['Report']
    report.features = ['skeleton_sections']
    owed = [o.key for o in check(report, bracket_project)]
    assert 'results' in owed, 'not drawn while its feature is off'
    bracket_project.report_features('Report', enable=['prediction'])
    assert 'results' not in [o.key for o in check(report, bracket_project)]
    blocks = _payload(_render(bracket_project))['blocks']
    heads = [k for k, b in enumerate(blocks) if b['kind'] == 'text'
             and 'class="skeleton"' in b.get('html', '')]
    results = next(k for k in heads if '8. Results' in blocks[k]['html'])
    following = heads[heads.index(results) + 1]
    assert [b['kind'] for b in blocks[results + 1:following]] == [
        'text', 'plot', 'plot', 'table', 'text']


def test_under_the_plot_conventions_it_is_an_asd_figure(bracket_project):
    from visualdynamics.report.plot_conventions import figure_kind

    block = bracket_project['Report'].blocks[1]
    assert figure_kind(block, bracket_project) == 'asd'
    assert figure_kind({**block, 'measured': 'Predictions'},
                       bracket_project) is None
    bracket_project.report_features(
        'Report', enable=['prediction', 'plot_conventions',
                          'shared_decades'])
    figures = [b for b in _payload(_render(bracket_project))['blocks']
               if b['kind'] == 'plot']
    assert figures[0]['home'] == figures[1]['home']


def test_the_insert_menu_offers_it_once_the_project_holds_predictions(
        bracket_project):
    assert ('prediction', 'Responses Against Prediction') in \
        insert_options(bracket_project)
    del bracket_project['Predictions']
    assert 'prediction' not in dict(insert_options(bracket_project))


# ---- the report editor ----------------------------------------------------


@pytest.fixture
def editing(window, pump, bracket_project):
    for name in ('Response PSD', 'Predictions'):
        window.add_object(name, bracket_project[name])
    window.add_object('Report', Report('R'))
    window.tree.setCurrentItem(window._item_for_object('Predictions'))
    pump()
    window.tree.setCurrentItem(window._item_for_object('Report'))
    window.render_current()
    pump()
    return window, window.report_editor, window.objects['Report']


def test_the_editor_inserts_the_block_with_the_feature_on_and_takes_its_level(
        editing):
    _window, editor, report = editing
    labels = [action.text() for action in editor.insert_menu.actions()]
    assert 'Responses Against Prediction' not in labels, \
        'not offered while its feature is off'
    editor._operate({'op': 'features', 'enable': ['prediction']})
    labels = [action.text() for action in editor.insert_menu.actions()]
    assert 'Responses Against Prediction' in labels
    editor.insert('prediction')
    assert report.blocks == [{'kind': 'prediction', 'source': 'Predictions',
                              'measured': 'Response PSD', 'level': None,
                              'caption': ''}]
    assert editor.field_boxes['measured'].currentData() == 'Response PSD'
    assert editor.level_edit is not None
    editor.level_edit.setText('-12')
    editor.level_edit.editingFinished.emit()
    assert report.blocks[0]['level'] == -12.0
    editor.level_edit.setText('minus twelve')
    editor.level_edit.editingFinished.emit()
    assert report.blocks[0]['level'] == -12.0, 'text not a number: unchanged'
    assert editor.level_edit.text() == '-12'
    editor.level_edit.setText('')
    editor.level_edit.editingFinished.emit()
    assert report.blocks[0]['level'] is None


def test_the_figures_and_table_draw_in_a_real_browser_engine(
        tmp_path, bracket_project):
    """The page's own script draws both figures on log-log axes, the
    prediction's segments among them, and the table with its rows."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from conftest import web_close, web_read, web_view
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication

    bracket_project.report_features('Report', enable=['prediction'])
    path = tmp_path / 'prediction.html'
    path.write_text(_render(bracket_project), encoding='utf-8')
    QApplication.instance() or QApplication(['x'])
    view = web_view()
    view.resize(1000, 800)
    view.load(QUrl.fromLocalFile(str(path)))
    try:
        js = web_read(
            view,
            "(() => { const all = document.querySelectorAll('canvas');"
            "if (all.length < 2 || !all[0].dataset.view"
            "    || !all[1].dataset.view) return null;"
            "const table = document.querySelector('table');"
            "return JSON.stringify({canvases: all.length,"
            " view: JSON.parse(all[0].dataset.view),"
            " headers: Array.from(table.querySelectorAll('th'),"
            "   th => th.textContent),"
            " rows: table.querySelectorAll('tbody tr').length}); })()")
    finally:
        web_close(view)
    built = json.loads(js)
    assert built['canvases'] == 2
    assert built['headers'][8] == 'ASD ratio (dB)'
    assert built['rows'] == 2
    # the window opens on decades of frequency that hold the mode
    assert built['view']['x0'] < math.log10(385.0) < built['view']['x1']


# ---- the edges the mutation run found ---------------------------------------


def test_the_lines_give_each_row_in_the_predictions_own_unit():
    measured = response_psd([(385.0, 12.0)])
    prediction = Prediction('201Y+', 'asd', [(385.27, 5.76, 12.0)], unit='g',
                            level_db=0.0, frequency_uncertainty=10.0)
    comparison = compare([prediction], measured, -12.0)
    (_channel, row), = comparison.rows
    predicted = row.predicted_level / G ** 2
    found = row.measured_level / G ** 2
    from visualdynamics.core.half_power import _figures

    assert predicted == pytest.approx(5.76 * 10 ** -1.2)
    assert comparison.lines() == [(
        f'201Y+: predicted 385.3 Hz, {_figures(predicted, 3)} g²/Hz; measured '
        f'{_figures(row.measured_frequency, 4)} Hz, {_figures(found, 3)} '
        f'g²/Hz; ASD ratio {_figures(row.ratio, 3)}, {row.ratio_db:+.1f} dB')]
    assert _figures(row.measured_frequency, 4) != \
        _figures(row.measured_frequency, 5)
    assert _figures(predicted, 3) != _figures(predicted, 4)


def test_messages_name_frequencies_to_four_figures():
    base = {'unit': 'g', 'level_db': 0.0}
    with pytest.raises(ValueError, match='no band to look for the peak at '
                       '385.3 Hz'):
        Prediction('201Y+', 'asd', [(385.27, 0.5)], **base)
    with pytest.raises(ValueError, match='the peak at 385.3 Hz: the band '
                       '390.3 to 420.8 Hz does not hold'):
        Prediction('201Y+', 'asd', [(385.27, 0.5, None, (390.27, 420.77))],
                   **base)
    with pytest.raises(ValueError, match='the peaks at 385.3 and 400.8 Hz '
                       'overlap'):
        Prediction('201Y+', 'asd', [(400.77, 0.3, None, (390.0, 420.0)),
                                    (385.27, 0.5, None, (370.0, 395.0))],
                   **base)
    f = np.arange(10.0, 2000.0, 1.0)
    with pytest.raises(ValueError, match='mode named at 900.3 Hz'):
        Prediction('201Y+', 'asd', [900.27], frequency_uncertainty=5.0,
                   curve=(f, transmissibility(f, 385.0, 12.0) ** 2), **base)
    measured = response_psd([(385.0, 12.0)])
    shifted = asd_prediction(fn=500.4, q=8.0)
    (_channel, row), = compare([shifted], measured, -12.0).rows
    assert row.missing == 'no peak of the record between 450.4 and 550.4 Hz'
    far = asd_prediction(fn=1000.4, q=8.0)
    measured.ordinate[0, 401] = np.nan                 # 1002.5 Hz
    (_channel, row), = compare([far], measured, -12.0).rows
    from visualdynamics.core.half_power import _figures

    assert _figures(1002.5, 4) != _figures(1002.5, 5)
    assert row.missing.startswith(
        f'the record has no value at {_figures(1002.5, 4)} Hz,')
    held = Prediction('201Y+', 'asd', [(385.27, 5.7812, 12.34)],
                      frequency_uncertainty=10.0, **base)
    assert held.describe() == ('201Y+, ASD in g²/Hz at +0 dB: 385.3 Hz 5.78 '
                               '(Q 12.3)')


def test_a_band_edge_on_a_line_is_inside_the_band():
    measured = response_psd([(385.0, 12.0)])
    f = measured.abscissa
    edged = Prediction('201Y+', 'asd', [(385.0, 5.8, 12.0, (380.0, 390.0))],
                       unit='g', level_db=0.0)
    for edge in (380.0, 390.0):
        broken = response_psd([(385.0, 12.0)])
        broken.ordinate[0, int(np.argmin(np.abs(f - edge)))] = np.nan
        (_channel, row), = compare([edged], broken, -12.0).rows
        assert row.missing.startswith(f'the record has no value at '
                                      f'{edge:g} Hz'), edge


def test_bands_that_touch_overlap_and_a_band_must_hold_its_peak_inside():
    base = {'unit': 'g', 'level_db': 0.0}
    with pytest.raises(ValueError, match='overlap'):
        Prediction('201Y+', 'asd', [(385.0, 0.5, None, (370.0, 400.0)),
                                    (415.0, 0.5, None, (400.0, 430.0))],
                   **base)
    with pytest.raises(ValueError, match='does not hold the peak'):
        Prediction('201Y+', 'asd', [(385.0, 0.5, None, (385.0, 420.0))],
                   **base)
    with pytest.raises(ValueError, match='does not hold the peak'):
        Prediction('201Y+', 'asd', [(385.0, 0.5, None, (350.0, 385.0))],
                   **base)


def test_a_curve_of_three_lines_from_0_hz_with_zeros_is_read():
    base = {'unit': 'g', 'level_db': 0.0, 'frequency_uncertainty': 50.0}
    held = Prediction('201Y+', 'asd', [(2.0,)],
                      curve=([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 0.0]),
                      **base)
    assert held.curve[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    # three lines and a peak between, from 0.5 Hz
    Prediction('201Y+', 'asd', [1.0], curve=([0.5, 1.0, 1.5],
                                              [0.5, 1.0, 0.5]), **base)
    with pytest.raises(ValueError, match='not a finite number'):
        Prediction('201Y+', 'asd', [1.0],
                   curve=([0.0, 1.0, 2.0], [0.5, 1.0, np.nan]), **base)
    with pytest.raises(ValueError, match='ascending and not below zero'):
        Prediction('201Y+', 'asd', [1.0],
                   curve=([-0.5, 1.0, 2.0], [0.5, 1.0, 0.5]), **base)
    with pytest.raises(ValueError, match="curve's frequencies must be "
                       'ascending'):
        Prediction('201Y+', 'asd', [1.0],
                   curve=([0.5, 1.0, 1.0, 2.0], [0.5, 1.0, 1.0, 0.5]),
                   **base)


def test_a_channel_that_is_not_a_string_is_refused_by_name():
    with pytest.raises(ValueError, match='names its channel'):
        asd_prediction(channel=201)
    with pytest.raises(ValueError, match='names its channel'):
        asd_prediction(channel='  ')


def test_a_zero_line_of_an_asd_is_a_value_and_an_frf_is_not_a_transmissibility():
    measured = response_psd([(385.0, 12.0)])
    measured.ordinate[0, int(np.argmin(np.abs(measured.abscissa - 350.0)))] \
        = 0.0
    (_channel, row), = compare([asd_prediction()], measured, -12.0).rows
    assert row.measured_frequency == pytest.approx(385.0, abs=1.25)
    accelerance = bracket()
    accelerance.ordinate_dim = ['acceleration/force']
    (channel,) = compare([Prediction('301Z+', 'transmissibility',
                                     [(410.0, 14.8)], reference='1Z+',
                                     frequency_uncertainty=10.0)],
                         accelerance).channels
    assert channel.note == ('the record is acceleration/force, not a '
                            'transmissibility (a quantity over itself)')


def test_a_psd_steps_across_its_bins_and_a_transmissibility_is_a_line():
    from visualdynamics.report.prediction import build

    measured = response_psd([(385.0, 12.0)])
    blocks = build({'kind': 'prediction', 'source': 'P', 'measured': 'M',
                    'level': -12.0},
                   {'P': Predictions([asd_prediction()]), 'M': measured},
                   visualdynamics.SI)
    assert blocks[0]['steps'] is True
    assert len(blocks[0]['edges']) == len(blocks[0]['x']) + 1
    frf = bracket()
    blocks = build({'kind': 'prediction', 'source': 'P', 'measured': 'M'},
                   {'P': Predictions([Prediction(
                       '301Z+', 'transmissibility', [(410.0, 14.8)],
                       reference='1Z+', frequency_uncertainty=10.0)]),
                    'M': frf}, visualdynamics.SI)
    assert 'steps' not in blocks[0] and 'edges' not in blocks[0]
    assert blocks[0]['caption'] == ('301Z+/1Z+: measured transmissibility '
                                    'over its prediction')


def test_a_predicted_curve_is_drawn_scaled_to_the_level_run():
    from visualdynamics.report.prediction import build

    f = np.arange(10.0, 2000.0, 1.0)
    curve = INPUT * transmissibility(f, 385.0, 12.0) ** 2
    prediction = Prediction('201Y+', 'asd', [385.0], unit='g', level_db=0.0,
                            frequency_uncertainty=10.0, curve=(f, curve))
    blocks = build({'kind': 'prediction', 'source': 'P', 'measured': 'M',
                    'level': -12.0},
                   {'P': Predictions([prediction]),
                    'M': response_psd([(385.0, 12.0)])}, visualdynamics.SI)
    drawn = blocks[0]['curves'][1]
    # the line of the drawn curve nearest the mode, against the curve
    # scaled by 10^(-12/10) there
    at = int(np.argmin(np.abs(np.asarray(drawn['x']) - math.log10(385.0))))
    nearest = int(np.argmin(np.abs(f - 10 ** drawn['x'][at])))
    expected = curve[nearest] * G ** 2 * 10 ** -1.2
    assert drawn['y'][at] == pytest.approx(math.log10(expected), abs=1e-4)


def test_the_table_gives_frequencies_to_four_figures_and_levels_to_three():
    from visualdynamics.core.half_power import _figures

    project = Project('t')
    project.add('Response PSD', response_psd([(385.0, 12.0), (410.0, 14.0)]))
    project.predict('P', '201Y+', 'asd', [(385.27, 5.7812, 12.34)], unit='g',
                    level_db=0, frequency_uncertainty=10)
    project.predict('P', '202Y+', 'asd', [(560.27, 2.56, 8)], unit='g',
                    level_db=0, frequency_uncertainty=10)
    project.add('Report', Report('R', features=['prediction']))
    project.report_prediction('Report', 'Response PSD', 'P', level=-12)
    blocks = _payload(_render(project))['blocks']
    first, _second = next(b for b in blocks if b['kind'] == 'table')['rows']
    level = 5.7812 * G ** 2 * 10 ** -1.2
    assert first[1:4] == ['385.3', _figures(level, 3), '12.3']
    assert _figures(level, 3) != _figures(level, 4)
    comparison = compare(project['P'], project['Response PSD'], -12.0)
    (_c, row), _second = comparison.rows
    assert first[4] == _figures(row.measured_frequency, 4)
    assert first[4] != _figures(row.measured_frequency, 5)
    assert first[8] == f'{_figures(row.ratio, 3)} ({row.ratio_db:+.1f} dB)'
    assert _figures(row.ratio, 3) != _figures(row.ratio, 4)
    note = blocks[-1]['html']
    assert '202Y+ at 560.3 Hz: no peak of the record between' in note
    assert 'Nothing was compared' not in note


def test_a_block_that_compares_nothing_says_so_in_its_note():
    project = Project('t')
    project.add('Response PSD', response_psd([(385.0, 12.0)]))
    project.predict('P', '299Y+', 'asd', [(385, 5.8, 12)], unit='g',
                    level_db=0, frequency_uncertainty=10)
    project.add('Report', Report('R', features=['prediction']))
    project.report_prediction('Report', 'Response PSD', 'P', level=-12)
    blocks = _payload(_render(project))['blocks']
    # no numbered table with no rows (2026-10-28): the note alone
    assert [b['kind'] for b in blocks] == ['text']
    assert blocks[0]['html'].startswith(
        '<p><strong>Responses against prediction:</strong> nothing compared.</p>')
    assert blocks[0]['html'].endswith(
        '<ul>\n<li>No record of 299Y+ in this measurement.</li>\n'
        '<li>Nothing was compared.</li>\n</ul>')


def test_a_drive_points_two_records_are_told_apart_by_the_predictions_quantity():
    """A drive point holds an accelerometer and a force gauge at one DOF:
    two records, one quantity each. Matched by DOF alone every drive
    point was refused as '2 records; compare one' (2026-10-28); the
    package's channel identity is (DOF, quantity), and the prediction
    names its quantity."""
    f = np.arange(0.0, 2000.0, 2.5)
    rows = np.vstack([INPUT * G ** 2 * transmissibility(f, 385.0, 12.0) ** 2,
                      np.full(f.size, 4.0)])
    measured = Psd(f, rows, response_dof=['201Y+', '201Y+'],
                   ordinate_dim=['acceleration**2/frequency',
                                 'force**2/frequency'])
    comparison = compare([asd_prediction()], measured, 0.0)
    channel, = comparison.channels
    assert channel.note == '' and channel.record == 0
    # the tallest line of the band, on a 2.5 Hz grid about the mode
    assert channel.rows[0].measured_level == pytest.approx(
        INPUT * G ** 2 * transmissibility(385.0, 385.0, 12.0) ** 2, rel=5e-3)
    # the same DOF twice in the prediction's own quantity is still two
    twice = Psd(f, rows[[0, 0]], response_dof=['201Y+', '201Y+'],
                ordinate_dim=['acceleration**2/frequency'] * 2)
    assert '2 records' in compare([asd_prediction()], twice, 0.0).channels[0].note


def test_a_block_that_compares_nothing_draws_a_note_and_no_empty_table():
    """A numbered table with no rows is a handle to nothing
    (2026-10-28): the note alone says what was not compared."""
    project = Project('t')
    project.add('Measured', response_psd([(385.0, 12.0)]))
    project.predict('Predictions', '999Z+', 'asd', [(385.0, 5.78, 12.0)],
                    unit='g', level_db=0.0, frequency_uncertainty=10.0)
    project.add('Report', Report('R', features=['prediction']))
    project.report_prediction('Report', 'Measured', 'Predictions', level=-12)
    blocks = _payload(_render(project))['blocks']
    assert [b['kind'] for b in blocks] == ['text']
    assert 'nothing compared' in blocks[0]['html']
    assert '999Z+' in blocks[0]['html']
