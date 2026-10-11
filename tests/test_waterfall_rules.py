"""The waterfall's stage and axis by value, where a mutation pass found
them reached only through pictures (2026-10-11).

`tools/mutate.py` over viz/waterfall.py caught 108 of 189 against its
two test files. `stage_curves` is the normalization the app's scene
and the report's canvas share; here three curves with a gap in one
are placed by hand, and the axis widening is held at its budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.viz.waterfall import STAGE, nice_axis, stage_curves

SX, SY, SZ = STAGE
X = np.linspace(10.0, 90.0, 9)


def _arrays(log_scaled=False):
    gap = np.full(9, 3.0)
    gap[4] = np.nan
    return {'curves': [(X, np.full(9, 1.0)), (X, np.linspace(1.0, 5.0, 9)), (X, gap)],
            'log_scaled': log_scaled, 'labels': ['a', 'b', 'c'],
            'xlabel': 'x', 'zlabel': 'z'}


def test_curves_stand_at_their_depth_and_height_and_break_at_a_gap():
    """10 to 90 Hz across (widening it to round ticks would waste a
    quarter of the stage), levels 1 to 5 up, the third curve two runs
    around its missing line."""
    got = stage_curves(_arrays())
    assert got['extents'] == (10.0, 90.0, 1.0, 5.0)
    runs = got['runs']
    assert [(r['record'], r['first'], len(r['points'])) for r in runs] == [
        (0, 0, 9), (1, 0, 9), (2, 0, 4), (2, 5, 4)]
    assert [r['station'] for r in runs] == [0.0, SY / 2, SY, SY]
    assert np.allclose(runs[0]['points'][:, 0], (X - 10.0) / 80.0 * SX)
    assert np.allclose(runs[0]['points'][:, 2], 0.0)
    assert np.allclose(runs[1]['points'][:, 2], np.linspace(0.0, SZ, 9))
    assert np.allclose(runs[3]['points'][:, 0], (X[5:] - 10.0) / 80.0 * SX)
    assert np.allclose(runs[3]['points'][:, 2], 0.5 * SZ)
    assert got['xn'](50.0) == pytest.approx(0.5 * SX)
    assert got['zn'](4.0) == pytest.approx(0.75 * SZ)


def test_an_object_that_pins_its_axis_pins_the_stage_unless_logged():
    pinned = stage_curves(_arrays(), ordinate_limits=(0.0, 10.0))
    assert pinned['extents'][2:] == (0.0, 10.0)
    assert np.allclose(pinned['runs'][0]['points'][:, 2], 0.1 * SZ)
    logged = stage_curves(_arrays(log_scaled=True), ordinate_limits=(0.0, 10.0))
    assert logged['extents'][2:] == (1.0, 5.0)


def test_one_curve_stands_at_the_front():
    one = stage_curves({**_arrays(), 'curves': _arrays()['curves'][:1]})
    assert one['runs'][0]['station'] == 0.0


@pytest.mark.parametrize('low, high, expected', [
    (0.0, 0.4998, (0.0, 0.5, 6)),          # widened: 0.0002 is under 5 % of the span
    (10.0, 90.0, (10.0, 90.0, 5)),         # 0 to 100 would waste a quarter
    (0.0, 1024.0, (0.0, 1024.0, 5)),       # 1200 would waste a sixth: kept exact
    (3.0, 3.0, (3.0, 3.0, 5)),             # no span: as given
])
def test_an_axis_widens_to_round_ticks_only_when_nearly_free(low, high, expected):
    assert nice_axis(low, high) == pytest.approx(expected)
