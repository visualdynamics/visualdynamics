"""The tracking detectors' span average and the drive's crossing,
pinned by value where a mutation pass found them reached only through
whole sweeps read to a tolerance (2026-10-11).

`tools/mutate.py` over sine_tracking.py against test_sine_tracking.py
caught 158 of 227, and 29 of the survivors were in `_span_average`:
the sweep tests read a tone's level to half a percent across hundreds
of lines, and a span read from the wrong edge or with its margin cut
short still lands inside that. Here a span is read once, against its
integral done by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sine_tracking import ONSET, RATE, _record
from test_sine_tracking import _tone as _sweep

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.sine import SineSweepSpecification
from visualdynamics.core.sine_tracking import (
    SineTracking,
    TrackedWaveform,
    _span_average,
    track_sine,
    track_waveform,
)

#: eight samples a cycle, so the span is upsampled eightfold
PER_CYCLE = 8


def _tone(cycles=12):
    phase = 2.0 * np.pi * np.arange(cycles * PER_CYCLE) / PER_CYCLE
    return np.sin(phase), phase


def _mean_sin_squared(a, b):
    """The mean of sin**2 over phase a to b, by hand."""
    return 0.5 - (np.sin(2.0 * b) - np.sin(2.0 * a)) / (4.0 * (b - a))


@pytest.mark.parametrize('detector, exact, within', [
    ('rms', 0.5, 2e-4),             # the mean of sin**2 over a cycle
    ('mean', 2.0 / np.pi, 1.5e-3),  # of |sin|: the trapezoid rounds its kinks
])
def test_a_cycle_between_samples_averages_to_its_integral(detector, exact, within):
    """From a phase between two samples to a sample one cycle on: the
    first point is interpolated at the phase asked for, and the margin
    either side carries the reconstruction."""
    signal, phase = _tone()
    end = 6 * PER_CYCLE + 3
    got = _span_average(signal, phase, end, phase[end] - 2.0 * np.pi, detector)
    assert got == pytest.approx(exact, rel=within)


def test_a_span_from_the_records_first_sample_starts_there():
    """Looser at the record's ends, where there is no margin to
    reconstruct from: 0.14 % at the start and 0.6 % at the end, read
    2026-10-11."""
    signal, phase = _tone()
    assert _span_average(signal, phase, PER_CYCLE, phase[0], 'rms') == \
        pytest.approx(0.5, rel=3e-3)


def test_a_span_to_the_records_last_sample_ends_there():
    signal, phase = _tone()
    end = len(phase) - 1
    assert _span_average(signal, phase, end, phase[end] - 2.0 * np.pi, 'rms') == \
        pytest.approx(0.5, rel=1.2e-2)


def test_a_short_span_is_averaged_and_an_empty_one_is_its_point():
    """Half a radian is still a span, not a point; a span of nothing is
    the waveform at its end."""
    signal, phase = _tone()
    end = 6 * PER_CYCLE + 3
    start = phase[end] - 0.5
    assert _span_average(signal, phase, end, start, 'rms') == pytest.approx(
        _mean_sin_squared(start, phase[end]), rel=2e-3)
    # the reconstructed waveform's, which is the sample's to 1e-4
    assert _span_average(signal, phase, end, phase[end], 'rms') == pytest.approx(
        signal[end] ** 2, rel=1e-4)


def _waveform(time, drive):
    time, drive = np.asarray(time, float), np.asarray(drive, float)
    return TrackedWaveform(setting=SineTracking(fixed=5.0), tone='Up', onset=0.0,
                           dof='1X+', dimension='acceleration', unit='m/s**2',
                           time=time, drive=drive, passed=np.zeros_like(time),
                           level=np.zeros_like(time))


def test_the_crossing_is_interpolated_between_samples_either_way():
    assert _waveform([0.0, 1.0, 2.0], [10.0, 20.0, 30.0]).instant(25.0) == 1.5
    assert _waveform([0.0, 1.0, 2.0], [30.0, 20.0, 10.0]).instant(25.0) == 0.5
    # landing on a sample from above interpolates to that sample
    assert _waveform([0.0, 1.0, 2.0], [30.0, 20.0, 10.0]).instant(20.0) == 1.0
    # a drive held at the frequency is there from its first sample
    assert _waveform([4.0, 5.0], [20.0, 20.0]).instant(20.0) == 4.0
    with pytest.raises(ValueError, match='it runs 10 to 30 Hz'):
        _waveform([0.0, 1.0, 2.0], [10.0, 20.0, 30.0]).instant(40.0)


def test_a_setting_is_refused_at_its_edges():
    with pytest.raises(ValueError, match='wider than zero'):
        SineTracking(fixed=0.0)
    assert SineTracking(fixed=5.0, order=1).order == 1
    with pytest.raises(ValueError, match='at least first order'):
        SineTracking(fixed=5.0, order=0)
    with pytest.raises(ValueError, match='more than zero cycles'):
        SineTracking(detector='rms', cycles=0.0)
    assert SineTracking(detector='rms', cycles=0.25).cycles == 0.25


def test_a_span_edge_is_interpolated_on_the_waveform():
    """A straight line, coarse enough that nothing is upsampled: the
    trapezoid is exact on it, so the mean over any span is the mean of
    its two ends — and the first end is interpolated, not the nearest
    sample's value."""
    step = 2.0 * np.pi / 100.0                     # 100 a cycle: no upsampling
    phase = step * np.arange(60)
    line = 1.0 + 0.01 * np.arange(60)
    end = 30
    for back in (0.3, 0.7, 4.4):                    # inside one step, and four
        start = phase[end] - back * step
        expected = (1.0 + 0.01 * (end - back) + line[end]) / 2.0
        assert _span_average(line, phase, end, start, 'mean') == pytest.approx(
            expected, rel=1e-12), back


@pytest.fixture(scope='module')
def clean():
    return _record('clean')


@pytest.fixture(scope='module')
def spec():
    return SineSweepSpecification([_sweep()], ['101Z+'], ordinate_unit='m/s**2')


def test_the_tone_is_laid_from_its_onset_through_its_last_whole_sample(clean, spec):
    waveform = track_waveform(clean, spec, SineTracking(fixed=10.0), onset=ONSET)
    dt = 1.0 / RATE
    steps = int(np.floor(_sweep().duration() / dt))
    assert waveform.time[0] == pytest.approx(ONSET)
    assert len(waveform.time) == steps + 1
    # a tone running past the recording is laid as far as the recording goes
    late = track_waveform(clean, spec, SineTracking(fixed=10.0), onset=ONSET + 1.0)
    assert late.time[-1] == clean.abscissa[-1]
    # two samples are the least the recording may hold of it
    end = len(clean.abscissa)
    assert len(track_waveform(clean, spec, SineTracking(fixed=10.0),
                              onset=(end - 2) * dt).time) == 2
    with pytest.raises(ValueError, match='the recording holds none of the tone'):
        track_waveform(clean, spec, SineTracking(fixed=10.0), onset=(end - 1) * dt)


def test_the_readings_start_once_the_widest_detector_has_its_cycles(clean, spec):
    """20 Hz rising an octave a second, f = 20 2**t: three cycles in,
    the phase 2 pi 20 (2**t - 1)/ln 2 is 6 pi, so f = 20 + 3 ln 2. The
    filtered setting's own cycles do not count; the last reading is the
    tone's last laid sample."""
    settings = [SineTracking(fixed=10.0), SineTracking(detector='rms', cycles=3.0)]
    levels = track_sine(clean, spec, settings, onset=ONSET)
    lines = np.asarray(levels[0].abscissa)
    assert lines[0] == pytest.approx(20.0 + 3.0 * np.log(2.0), abs=0.01)
    last = np.floor(_sweep().duration() * RATE) / RATE
    assert lines[-1] == pytest.approx(_sweep().frequency_at(np.array([last]))[0],
                                      rel=1e-9)
    with pytest.raises(ValueError, match='less than 1000 cycles'):
        track_sine(clean, spec, SineTracking(detector='rms', cycles=1000.0),
                   onset=ONSET)


def test_the_passed_waveform_is_the_record_in_its_own_phase(spec):
    """A tone laid as a sine rather than a cosine comes back through a
    band that passes it as that sine: moved back up to the drive by
    undoing the demodulation, not by repeating it, which would give
    the cosine record back and a sine record back negated."""
    tone = _sweep()
    t = np.arange(int((ONSET + tone.duration()) * RATE)) / RATE
    local = np.clip(t - ONSET, 0.0, None)
    x = np.where(t >= ONSET, np.sin(tone.phase_at(local)), 0.0)
    history = TimeHistory(t, x[None], response_dof=['101Z+'],
                          ordinate_dim=['acceleration'])
    waveform = track_waveform(history, spec, SineTracking(fixed=10.0), onset=ONSET)
    middle = (waveform.time > ONSET + 1.5) & (waveform.time < ONSET + 2.5)
    record = x[round(ONSET * RATE):][:len(waveform.time)]
    assert np.max(np.abs(waveform.passed[middle] - record[middle])) < 0.05
