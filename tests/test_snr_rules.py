"""Signal-to-noise's pick and its division, pinned where a mutation
pass found them unguarded (2026-10-11)."""

from __future__ import annotations

import numpy as np

from visualdynamics.core.data import Psd
from visualdynamics.core.snr import _subtracted, rms_signal_to_noise

FREQ = np.arange(1.0, 101.0)


def test_a_pick_names_driven_rows_whatever_order_the_ambient_is_in():
    driven = Psd(FREQ, np.full((2, 100), 4e-4), response_dof=['1X+', '2X+'],
                 ordinate_dim='acceleration**2/frequency')
    ambient = Psd(FREQ, np.full((2, 100), 1e-4), response_dof=['2X+', '1X+'],
                  ordinate_dim='acceleration**2/frequency')
    ((label, _db),), _silent = rms_signal_to_noise(driven, ambient, records=[0])
    assert label.startswith('1X+')


def test_the_ratio_is_over_an_ambient_that_recorded_something():
    got = _subtracted(np.array([3.0, 3.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    assert np.isnan(got[0]), 'nothing to divide by'
    assert got[1] == 2.0
    assert np.isnan(got[2]), 'no signal above the floor'
