"""Rules inside the sine extraction, pinned by value where a mutation
pass found them unguarded (2026-10-11).

The extraction is tested end to end — planted tones come back at their
amplitudes, within a tolerance — and a tolerance cannot tell a
constant from one a few percent off, or a clock correction from its
negative. `tools/mutate.py` over sine.py caught 499 of 783 mutants;
what survived in the pieces below is stated here as the rule each is.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_extract_sine import FS, _recording, _spec

from visualdynamics.core import sine


def test_the_noise_integral_is_the_integral_it_names():
    """The second-order filter's equivalent noise bandwidth in units of
    its corner — what the debiasing subtracts the noise power by."""
    from scipy.integrate import quad

    value, _error = quad(lambda x: 1.0 / (1.0 + x ** 4) ** 2, -np.inf, np.inf)
    assert sine._NOISE_INTEGRAL == pytest.approx(value, rel=1e-12)


def test_a_readings_scatter_is_the_floor_over_root_two_in_decibels():
    """Half the noise power lies along the tone: the amplitude's
    standard deviation is floor/sqrt(2), 20/ln(10) dB per unit ratio;
    infinite where nothing stood above the floor."""
    got = sine.scatter_db(np.array([1.0, 2.0, 0.0]),
                          np.array([0.1 * np.sqrt(2.0)] * 3))
    per_unit = 20.0 / np.log(10.0)
    assert got[0] == pytest.approx(0.1 * per_unit)
    assert got[1] == pytest.approx(0.05 * per_unit)
    assert np.isinf(got[2])


class _Span:
    """What `_clock_correction` reads of a laid tone: its length."""

    def __init__(self, n, dt):
        self.n, self.dt = n, dt


def test_the_clock_correction_is_the_line_through_the_slopes():
    """Slopes along a line a + b u (u seconds from the span's middle)
    give the phase correction c1 u + c2 u^2 whose slope it is: c1 = a,
    c2 = b / 2; the median across channels, so one channel's own phase
    cannot pull it."""
    span = _Span(1000, 0.01)                       # ten seconds, middle at 5
    times = np.array([1.0, 3.0, 6.0, 9.0])
    line = [(t, 0.3 + 0.08 * (t - 5.0), 1.0) for t in times]
    off = [(t, 7.0, 1.0) for t in times]           # a channel of its own
    c1, c2 = sine._clock_correction(span, [line, line, off])
    assert (c1, c2) == (pytest.approx(0.3), pytest.approx(0.04))
    # one point is a level and no curvature; no points, nothing
    assert sine._clock_correction(span, [[(2.0, 0.5, 1.0)]]) == (0.5, 0.0)
    assert sine._clock_correction(span, [[]]) == (0.0, 0.0)


def test_the_correction_is_added_about_the_middle_of_the_span():
    spec = _spec()
    tone = spec.tone('Up')
    laid = sine._Laid(tone, 1.0, int(FS), 4096, 1.0 / FS, 10.0)
    plain = laid.argument(0, 4096)
    laid.c1, laid.c2 = 2.0, 0.5
    u = laid.seconds(0, 4096) - 4096 / FS / 2.0
    assert np.allclose(laid.argument(0, 4096) - plain, 2.0 * u + 0.5 * u * u)


def test_the_smoothing_asked_for_follows_the_square_root_law(monkeypatch):
    """The scatter falls as the square root of the smoothing, so one
    measurement at BASE_CYCLES names the rung: 1.5 x BASE x
    (scatter / target)^2, up the ladder to the first rung at or above
    it — capped where a window at the tone's lowest frequency reaches
    a quarter of its span, and the top of the ladder for a tone not
    seen at all."""
    spec = _spec()
    up = spec.tone('Up')
    history = _recording(spec)

    def scatter(value):
        monkeypatch.setattr(sine, 'sample_levels', lambda *a, **k: [
            {'scatter_db': np.full((2, 8), value)}])
        return sine.suggest_cycles(history, spec, tones=['Up'])

    # 1.5 * 10 * (2.0)^2 = 60: exactly a rung
    assert scatter(2.0) == 60.0
    # 1.5 * 10 * (1.2)^2 = 21.6: the next rung up is 30
    assert scatter(1.2) == 30.0
    assert scatter(0.1) == sine.BASE_CYCLES, 'never under the base'
    cap = min(up.frequency) * up.duration() / 4.0
    assert scatter(np.inf) == pytest.approx(min(sine.CYCLES_LADDER[-1], cap))


def test_the_sampled_points_say_when_they_are_and_what_frequency_that_is():
    spec = _spec()
    history = _recording(spec)
    (reading,) = sine.sample_levels(history, spec, tones=['Up'], windows=5)
    up = spec.tone('Up')
    seconds = reading['seconds']
    assert len(seconds) == 5 and np.all(np.diff(seconds) > 0.0)
    assert up.start_time <= seconds[0] and seconds[-1] <= up.start_time + up.duration()
    assert np.allclose(reading['frequency'], up.frequency_at(seconds - up.start_time),
                       rtol=1e-3)


def test_a_tone_is_found_where_it_starts_without_a_window_too():
    spec = _spec()
    history = _recording(spec)
    records = np.asarray(history.ordinate)
    assert sine.find_tone(records, 1.0 / FS, spec.tone('Up')) == pytest.approx(
        1.0, abs=2e-3)
    with pytest.raises(ValueError, match='holds no onset candidates'):
        sine.find_tone(records, 1.0 / FS, spec.tone('Up'), search=(30.0, 40.0))
