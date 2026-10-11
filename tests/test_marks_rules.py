"""The stage marks' geometry by value, where a mutation pass found it
reached only through pictures (2026-10-11).

`tools/mutate.py` over viz/marks.py caught 74 of 235 against its three
test files. Most of what survived draws handles and previews into a
window; the pieces here are the plain geometry the app's stage and
the report's canvas share, and they can be held to numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.shocks import Shock
from visualdynamics.viz import marks
from visualdynamics.viz.waterfall import STAGE

SX, SY, SZ = STAGE


@pytest.mark.parametrize('low, high, labels', [
    (-1.0, 2.5, [(-1.0, '10⁻¹'), (0.0, '1'), (1.0, '10¹'), (2.0, '10²')]),
    (1.0 + 1e-12, 3.0 - 1e-12, [(1.0, '10¹'), (2.0, '10²'), (3.0, '10³')]),
    (0.2, 0.8, []),
])
def test_the_decades_inside_a_range_are_labeled_as_the_flat_plot_labels_them(
        low, high, labels):
    assert marks.decade_labels(low, high) == labels


def test_a_handle_leans_back_by_its_tilt_about_the_stage_x():
    s, c = np.sin(marks.TILT), np.cos(marks.TILT)
    leaned = marks._tilted(2.0, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0]],
                           seat=0.3, forward=-0.1)
    assert np.allclose(leaned, [[2.0, -0.1 + c, 0.3 + s], [2.0, -0.1 - s, 0.3 + c],
                                [2.5, -0.1, 0.3]], atol=1e-6)


def test_overlapping_frames_stack_on_the_rail_above_the_stage():
    """Four frames of a second at three-quarters overlap, on a stage
    from 0 to 10 s: each on its own row, a slot apart above the foot,
    each spanning its own second."""
    averaging = Averaging(frame_length=100, overlap=0.75, window='hann', frames=4,
                          start=2.0)
    got = marks.averaging_stage_geometry(averaging, 100.0, (0.0, 10.0, 0.0, 1.0))
    slot = min(marks.RAIL_SLOT, marks.RAIL_BAND / 4) * SZ
    foot = SZ * (1.0 + marks.RAIL_GAP)
    glyph = slot * marks.GLYPH_SHARE
    assert [f['baseline'] for f in got['frames']] == pytest.approx(
        [foot + level * slot for level in range(4)])
    starts = [2.0, 2.25, 2.5, 2.75]
    for frame, start in zip(got['frames'], starts):
        assert frame['caps'] == pytest.approx([start / 10 * SX, (start + 1) / 10 * SX])
        assert frame['xs'][0] == pytest.approx(start / 10 * SX)
        assert frame['xs'][-1] == pytest.approx((start + 0.99) / 10 * SX)
        assert max(frame['glyph']) - frame['baseline'] == pytest.approx(
            glyph * max(frame['weights']))
    assert got['span'] == pytest.approx([0.2 * SX, 0.375 * SX])
    assert got['cap'] == pytest.approx(glyph * marks.CAP_SHARE)
    assert got['wall'] == pytest.approx(SY * marks.WALL_SET_BACK)


def test_a_shock_window_lands_on_the_stage_by_the_records_clock():
    got = marks.shock_stage_geometry([Shock(1.0, 0.5), Shock(4.0, 2.0)],
                                     (10.0, 20.0, 0.0, 1.0), origin=10.0)
    assert got == {'windows': [
        {'span': pytest.approx([0.1 * SX, 0.15 * SX]), 'label': '1'},
        {'span': pytest.approx([0.4 * SX, 0.6 * SX]), 'label': '2'}]}
