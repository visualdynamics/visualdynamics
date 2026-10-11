"""The detector's helpers pinned by value where a mutation pass found
them reached only through whole-record detection (2026-10-11).

`tools/mutate.py` over detect.py caught 69 of 134 against its tests,
which plant a settled stretch and check that it is found. The pieces
that find it — the longest run, the scatter estimate, the trim, the
frame length, the plateau sweep — are stated here as the rules they
are, so a change to one is caught before it moves a detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import detect


@pytest.mark.parametrize('mask, run', [
    ([0, 1, 1, 0, 1, 1, 1, 0], (4, 7)),
    ([1, 1, 1, 0, 1, 1, 1], (0, 3)),       # a tie keeps the first
    ([0, 0, 1, 1], (2, 4)),                # a run to the end closes there
    ([0, 0, 0], (0, 0)),
])
def test_the_longest_run_of_true_last_exclusive(mask, run):
    assert detect._longest_run(np.array(mask, dtype=bool)) == run


def test_the_scatter_is_the_median_absolute_step_as_a_sigma():
    """Steps of plus and minus a: their median is zero, their median
    absolute deviation a, which is a/0.6745 as a normal's sigma, and a
    difference of two hops is sqrt(2) times as noisy as one."""
    level = np.array([0.0, 2.0] * 10 + [0.0])      # ten steps each way
    assert detect.observed_scatter(level) == pytest.approx(2.0 / 0.6745 / np.sqrt(2.0))
    # a single step in a long steady run barely moves it
    step = np.concatenate([np.full(50, 3.0), np.full(50, 9.0)])
    assert detect.observed_scatter(step) == 0.0
    assert detect.observed_scatter(np.array([1.0, 5.0])) == 0.0, 'too short to say'


def test_a_frame_is_the_power_of_two_nearest_a_second():
    assert detect.frame_length_for(1024.0) == 1024
    assert detect.frame_length_for(1000.0) == 1024
    assert detect.frame_length_for(1500.0) == 2048
    assert detect.frame_length_for(1.0) == 2, 'never under two samples'
    with pytest.raises(ValueError, match='must be positive'):
        detect.frame_length_for(0.0)


def test_the_trim_keeps_the_settled_middle():
    assert detect.trim(0, 100, (0.1, 0.9)) == (10, 90)
    assert detect.trim(20, 30, (0.25, 0.75)) == (22, 28)
    assert detect.trim(5, 5) == (5, 5), 'nothing to trim'
    assert detect.trim(0, 1, (0.5, 0.6)) == (0, 1), 'too short: whole'
    with pytest.raises(ValueError, match='low < high'):
        detect.trim(0, 10, (0.5, 0.5))
    detect.trim(0, 10, (0.0, 1.0))


def test_the_plateau_is_the_highest_level_held_long_enough():
    """Duration first, level second: a short overload above a long
    steady run loses to the run; with nothing held long enough, the
    highest level sustained at all is taken, briefly."""
    level = np.concatenate([np.zeros(20), np.full(10, 30.0), np.zeros(5),
                            np.full(60, 20.0), np.zeros(10)])
    assert detect.plateau(level, sigma=0.5, want_hops=40) == (35, 95), (
        'the long run at 20, not the overload at 30')
    assert detect.plateau(level, sigma=0.5, want_hops=200) == (20, 30), (
        'nothing held that long: the top level held at all, all of it')
    assert detect.plateau(np.zeros(0), sigma=1.0, want_hops=5) == (0, 0)
