"""Fractional-octave bands' edges and readings, pinned by value where a
mutation pass found them unguarded (2026-10-11).

`tools/mutate.py` over octave.py caught 48 of 65 against its tests;
what survived was reading a fraction back off band centers, the
default fraction, and the refusals and empty bands at the ends.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import octave


@pytest.mark.parametrize('fraction', [1, 3, 6, 24])
def test_a_fraction_is_read_back_off_its_band_centers(fraction):
    centers, _widths, _edges = octave.bands(10.0, 1000.0, fraction)
    assert octave.per_octave_of(centers) == fraction
    # a dead first line, at zero or not a number, is passed over
    assert octave.per_octave_of(np.concatenate([[0.0, np.nan], centers])) == fraction


def test_the_default_is_a_sixth_of_an_octave():
    centers, _widths, _edges = octave.bands(10.0, 1000.0)
    assert octave.per_octave_of(centers) == 6


def test_the_ends_are_refused_by_name():
    with pytest.raises(ValueError, match='at least two bands'):
        octave.per_octave_of([10.0, 0.0])
    assert octave.per_octave_of([1000.0, 1000.0 * 10 ** 0.1]) == 3, 'two are enough'
    with pytest.raises(ValueError, match='band range 0.0 to 10.0 is not a range'):
        octave.edges(0.0, 10.0)
    with pytest.raises(ValueError, match='band range 10.0 to 10.0 is not a range'):
        octave.edges(10.0, 10.0)
    with pytest.raises(ValueError, match='at least two lines'):
        octave.resample([10.0], [[1.0]], [8.0, 12.0])


def test_a_band_with_no_line_under_it_says_nothing():
    """Lines at 10 and 11 Hz stand for 9.5 to 11.5: a band from 11.5 up
    holds none of it — not a zero, which would be a reading."""
    got = octave.resample([10.0, 11.0], [[2.0, 2.0]], np.array([9.5, 11.5, 13.0]))
    assert got[0, 0] == pytest.approx(2.0)
    assert np.isnan(got[0, 1])
