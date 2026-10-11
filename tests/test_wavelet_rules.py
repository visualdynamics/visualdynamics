"""The scalogram's defaults and edges, pinned by value where a mutation
pass found them reached only through a drawn picture (2026-10-11).

`tools/mutate.py` over wavelet.py against test_wavelet.py caught 71 of
111. The transform itself was pinned well; what survived was the range
a record opens on, the reading the window and the plots share, and the
edges of what the grid and the transform accept. The rest of the
survivors are equivalent (an FFT's worker count, the analytic half's
zero bin, which the mean removal already empties) or tuning (the band
size, the column count).
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import wavelet as w
from visualdynamics.core.data import TimeHistory


@pytest.mark.parametrize('rate, duration, expected', [
    (1000.0, 10.0, (0.8, 490.0)),       # eight cycles of the lowest line fit
    (1000.0, 0.0, (4.9, 490.0)),        # no duration: two decades down
    (1000.0, 1000.0, (0.5, 490.0)),     # never below a thousandth of Nyquist
    (1000.0, 0.001, (245.0, 490.0)),    # never above an octave under the top
])
def test_the_default_range_is_what_the_record_can_carry(rate, duration, expected):
    assert w.default_range(rate, duration) == pytest.approx(expected)


def _history(start=3.0, rate=1024.0, samples=2048):
    # a binary rate, so the clock's steps are exact and Nyquist is too
    t = start + np.arange(samples) / rate
    tone = np.sin(2.0 * np.pi * 50.0 * t)
    return TimeHistory(t, np.array([tone, np.zeros(samples)]),
                       response_dof=['1X+', '2X+'], ordinate_dim='acceleration')


def test_a_reading_opens_on_the_default_range_at_twelve_lines_an_octave():
    clock, magnitude, frequencies, width = w.reading(_history())
    assert frequencies[0] == pytest.approx(4.0), 'eight cycles in two seconds'
    assert frequencies[-1] < 512.0
    assert frequencies[1:] / frequencies[:-1] == pytest.approx(
        np.full(frequencies.size - 1, 2.0 ** (1 / 12)))
    assert width == w.OMEGA0
    assert clock[0] == 3.0 and clock[-1] < 5.0, "on the record's own clock"
    assert magnitude.max() > 0.5, 'channel 0, the one with the tone'
    _clock, quiet, _f, _w = w.reading(_history(), channel=1)
    assert quiet.max() == 0.0


def test_a_reading_drops_the_lines_at_or_over_nyquist():
    # one line an octave: 128, 256, 512, 1024 — Nyquist itself is dropped
    _clock, _m, frequencies, _w = w.reading(_history(), low=128.0, high=600.0,
                                            per_octave=1)
    assert frequencies.tolist() == [128.0, 256.0]
    with pytest.raises(ValueError, match='it reaches 512 Hz'):
        w.reading(_history(), low=256.0, high=600.0, per_octave=1)


def test_the_grid_needs_a_range_and_a_line_an_octave():
    with pytest.raises(ValueError, match='positive low'):
        w.log_frequencies(50.0, 50.0)
    with pytest.raises(ValueError, match='positive low'):
        w.log_frequencies(0.0, 50.0)
    assert w.log_frequencies(10.0, 40.0, per_octave=1).tolist() == [10.0, 20.0, 40.0]
    with pytest.raises(ValueError, match='at least one line per octave'):
        w.log_frequencies(10.0, 40.0, per_octave=0)


def test_the_transform_takes_two_samples_and_nothing_at_zero_or_nyquist():
    assert w.scalogram(np.array([0.0, 1.0]), 100.0, [10.0]).shape == (1, 2)
    with pytest.raises(ValueError, match='at least two samples'):
        w.scalogram(np.array([1.0]), 100.0, [10.0])
    with pytest.raises(ValueError, match=r'below Nyquist \(50 Hz\); asked for 0 to 10 Hz'):
        w.scalogram(np.zeros(64), 100.0, [0.0, 10.0])
    with pytest.raises(ValueError, match='asked for 10 to 50 Hz'):
        w.scalogram(np.zeros(64), 100.0, [10.0, 50.0])
