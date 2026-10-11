"""Kurtosis at its smallest and of nothing, and the span it reads,
pinned where a mutation pass found them unguarded (2026-10-11)."""

from __future__ import annotations

import math

import numpy as np

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.kurtosis import analyzed_span, kurtosis
from visualdynamics.core.shocks import Shock


def test_two_samples_are_enough_and_a_constant_has_no_shape():
    """Plus and minus one about their mean: the fourth moment over the
    second squared is one, the flattest a distribution can be."""
    assert kurtosis([1.0, 3.0]) == 1.0
    assert math.isnan(kurtosis([2.0]))
    assert math.isnan(kurtosis([4.0, 4.0, 4.0]))


def test_shock_windows_shorter_than_a_sample_are_passed_over():
    """At 100 samples a second, 3 ms from 0.501 s is sample 50 to 50."""
    t = np.arange(100) / 100.0
    history = TimeHistory(t, np.zeros((1, 100)), response_dof=['1X+'])
    history.shocks = [Shock(0.2, 0.1), Shock(0.501, 0.003)]
    assert analyzed_span(history) == ([(20, 30)], 'over the shock windows')
    history.shocks = [Shock(0.501, 0.003)]
    assert analyzed_span(history) == ([(0, 100)], 'over the record')
