"""Which way a stage's limit zones point.

`viz.sinespec.limit_zones` is what the sine stage and the banded stage
both shade from (2026-10-10): yellow between warning and abort, and
past abort the over color upward to the stage's ceiling and the under
color down to its floor. Swapping the two passed the whole suite
before this, though a red stripe below the lower limit says the
opposite of what happened.
"""

from __future__ import annotations

import numpy as np

from visualdynamics.viz.sinespec import limit_zones


def test_past_abort_is_over_upward_and_under_downward():
    curves = {'warning_upper': np.array([3.0, 3.0]),
              'abort_upper': np.array([4.0, 4.0]),
              'warning_lower': np.array([1.5, 1.5]),
              'abort_lower': np.array([1.0, 1.0])}
    zones = limit_zones(curves.get, 2, 10.0)
    keys = [key for _inner, _outer, key in zones]
    assert keys == ['limit_warning', 'exceed_over', 'limit_warning',
                    'exceed_under']
    (w_up, a_up, _), (a_up2, ceiling, _), (w_lo, a_lo, _), (a_lo2, floor, _) = zones
    assert np.allclose(ceiling, 10.0) and np.allclose(floor, 0.0)
    assert np.allclose(a_up, a_up2) and np.allclose(a_lo, a_lo2)
    assert np.allclose(w_up, 3.0) and np.allclose(a_up, 4.0)
    assert np.allclose(w_lo, 1.5) and np.allclose(a_lo, 1.0)


def test_a_warning_alone_runs_to_the_edge():
    zones = limit_zones({'warning_upper': np.array([3.0])}.get, 1, 10.0)
    assert len(zones) == 1
    inner, outer, key = zones[0]
    assert key == 'limit_warning' and np.allclose(outer, 10.0)
    assert np.allclose(inner, 3.0)
