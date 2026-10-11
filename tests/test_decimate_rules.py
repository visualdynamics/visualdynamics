"""Peak-keeping decimation at its edges, by value (2026-10-11).

`tools/mutate.py` over decimate.py caught 35 of 46 against the report
and waterfall tests, which decimate large records and look at the
picture. What survived was where decimation starts, which sample an
empty slice keeps, the padding slice past the data, and the bin ends
an envelope is drawn at.
"""

from __future__ import annotations

import numpy as np

from visualdynamics.decimate import _extreme_indices, envelope_rows, peak_decimate


def test_a_record_at_its_budget_is_returned_whole_and_one_over_is_thinned():
    x = np.arange(8.0)
    y = np.array([0.0, 5.0, 1.0, 1.0, 1.0, 1.0, -3.0, 1.0])
    same_x, same_y = peak_decimate(x, y, 8)
    assert same_x is x and same_y is y
    thin_x, thin_y = peak_decimate(np.arange(9.0), np.append(y, 1.0), 8)
    assert len(thin_x) < 9 and {5.0, -3.0} <= set(thin_y.tolist())


def test_an_empty_slice_keeps_its_first_sample_and_the_padding_keeps_none():
    """Two slices of four: the first all gap keeps its start, the
    second its least and greatest. Six samples in four slices of two
    pad a fourth slice wholly past the data, which keeps nothing."""
    gaps = np.array([[np.nan] * 4 + [1.0, 5.0, 2.0, 3.0]])
    assert _extreme_indices(gaps, 2)[0].tolist() == [0, 4, 5]
    six = np.array([[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]])
    assert _extreme_indices(six, 4)[0].tolist() == [0, 1, 2, 3, 4, 5]


def test_an_envelope_is_drawn_at_each_bins_first_and_last_sample():
    """Ten samples on three bins of four: the bins end at samples 3, 7
    and 9 (the last cut short); each bin's two values go in the order
    they happened."""
    x = np.arange(10.0)
    rows = np.array([[0.0, 9.0, 1.0, 2.0, 8.0, 3.0, 0.5, 4.0, 7.0, 6.0]])
    axis, out = envelope_rows(x, rows, 3)
    assert axis.tolist() == [0.0, 3.0, 4.0, 7.0, 8.0, 9.0]
    assert out.tolist() == [[0.0, 9.0, 8.0, 0.5, 7.0, 6.0]]
    whole_axis, whole = envelope_rows(x[:6], rows[:, :6], 3)
    assert whole_axis.tolist() == x[:6].tolist() and whole.tolist() == rows[:, :6].tolist()
