"""The filter module's arithmetic pinned by value where a mutation pass
found it unguarded (2026-10-11): integration's drift control, the
edges of what a filter and an integration accept, and the range the
response is drawn over."""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import filters
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import Filtering

RATE = 1000.0


def _history(values, dim='acceleration'):
    t = np.arange(values.shape[-1]) / RATE
    return TimeHistory(t, np.atleast_2d(values), response_dof=['1X+'],
                       ordinate_dim=dim, ordinate_unit='m/s**2')


def test_a_bias_is_taken_out_before_it_can_integrate_to_a_ramp():
    """A 10 Hz acceleration riding on a 1 m/s^2 bias integrates to a
    velocity of amplitude 1/(2 pi 10) m/s about zero — the bias
    removed first, so it never becomes the 10 m/s ramp a high-pass
    would then be left to fight. Adding the mean instead of removing
    it passed every test before (mutation testing, 2026-10-11)."""
    t = np.arange(int(10 * RATE)) / RATE
    velocity = filters.integrate(_history(1.0 + np.sin(2 * np.pi * 10 * t)))
    middle = velocity.ordinate[0, 2000:8000]
    assert np.abs(middle).max() == pytest.approx(1.0 / (2 * np.pi * 10), rel=0.02)
    assert abs(middle.mean()) < 1e-3


def test_raw_integration_keeps_the_bias_as_the_ramp_it_is():
    t = np.arange(int(2 * RATE)) / RATE
    raw = filters.integrate(_history(np.ones(t.size)), drift_corner=None)
    assert raw.ordinate[0, -1] == pytest.approx(t[-1])


def test_the_drift_corner_sits_strictly_between_zero_and_nyquist():
    history = _history(np.ones(100))
    for corner in (0.0, RATE / 2.0):
        with pytest.raises(ValueError, match='must sit between zero and the Nyquist '
                                             r'frequency \(500 Hz\)'):
            filters.integrate(history, drift_corner=corner)
    filters.integrate(history, drift_corner=RATE / 2.0 - 1.0)


def test_a_filter_is_fourth_order_unless_told_and_a_band_runs_upward():
    assert Filtering(high=100.0).order == 4
    assert Filtering(high=100.0, order=1).order == 1
    with pytest.raises(ValueError, match='low \\(100 Hz\\) must sit below high'):
        Filtering(low=100.0, high=100.0)


def test_the_response_is_drawn_from_below_the_lowest_edge_to_nyquist():
    """Over the decades below Nyquist or half the lowest edge, whichever
    is lower; an edge past Nyquist mid-edit is drawn clamped, not
    refused."""
    f, db = filters.response(Filtering(low=10.0, high=100.0), RATE)
    nyquist = RATE / 2.0
    assert f[0] == pytest.approx(min(nyquist / 10.0 ** filters.RESPONSE_DECADES, 5.0))
    assert f[-1] == pytest.approx(nyquist)
    assert len(f) == 400
    f, db = filters.response(Filtering(high=2000.0), RATE)
    assert np.all(np.isfinite(db))
    # the zero-phase pair: -6.02 dB at a Butterworth's nominal corner
    f, db = filters.response(Filtering(high=100.0), RATE, lines=4001)
    assert np.interp(100.0, f, db) == pytest.approx(-6.02, abs=0.05)


def test_a_high_pass_edge_past_nyquist_is_drawn_clamped_but_designed_refused():
    _f, db = filters.response(Filtering(low=2000.0), RATE)
    assert np.all(np.isfinite(db)), 'clamped just under Nyquist to draw'
    with pytest.raises(ValueError, match=r'Nyquist frequency \(500 Hz\)'):
        filters.design(Filtering(low=2000.0), RATE)
