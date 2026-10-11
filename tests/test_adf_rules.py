"""The extremes an ADF record header states (2026-10-11).

`tools/mutate.py` over adf.py found header fields the writer sets
and the reader never reads back. Held against the sample files, one
was wrong: the ordinate's max and min were clamped at zero, so an
all-positive record's minimum went out as 0. The header comparison
itself needs the samples and lives with them; this is the rule.
"""

from __future__ import annotations

import numpy as np

from visualdynamics.io import adf


def test_extremes_pass_over_nan_keep_infinity_and_default_to_zero():
    """What the reference files hold for records with NaN, infinities,
    only negatives, and no numbers at all."""
    assert adf._extremes(np.array([np.nan, 2.0, 1.0], '<f4')) == (2.0, 1.0)
    assert adf._extremes(np.array([-1.0, -8.0], '<f4')) == (-1.0, -8.0)
    assert adf._extremes(np.array([np.inf, -np.inf], '<f4')) == (np.inf, -np.inf)
    assert adf._extremes(np.array([np.nan], '<f4')) == (0.0, 0.0)
    assert adf._extremes(np.array([], '<f4')) == (0.0, 0.0)
