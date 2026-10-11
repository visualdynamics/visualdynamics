"""Where the sine stage puts a specification, read off its meshes
(2026-10-11).

`tools/mutate.py` over viz/sinespec.py caught 65 of 168 against its
tests, 72 of the survivors in `add_sine_stage`: frequency along,
time in depth, amplitude up, each onto `STAGE`, and a mutant that
moved one of them drew a stage the tests still counted right. A
plotter that keeps what it is given stands in for the window.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.sine import SineSweepSpecification, SineTone
from visualdynamics.units import SI
from visualdynamics.viz.sinespec import add_sine_stage
from visualdynamics.viz.waterfall import STAGE

SX, SY, SZ = STAGE


class _Kept:
    def __init__(self):
        self.meshes, self.labels, self.bounds = {}, None, None

    def add_mesh(self, mesh, name=None, **_style):
        self.meshes[name] = mesh

    def add_point_labels(self, points, names, **_style):
        self.labels = (np.asarray(points), list(names))

    def show_bounds(self, **style):
        self.bounds = style['axes_ranges']


def _spec():
    """Two tones on one channel: 100 to 400 Hz from 1 s at 2 m/s^2 and
    400 down to 200 Hz from 4 s at 1 m/s^2, the first with an upper
    abort at 4."""
    up = SineTone('Up', 1.0, [100.0, 400.0], [[2.0], [2.0]], [0], [120.0])
    down = SineTone('Down', 4.0, [400.0, 200.0], [[1.0], [1.0]], [0], [-120.0])
    up.limits['abort_upper'] = np.array([[4.0], [4.0]])
    return SineSweepSpecification([up, down], ['1X+'], ordinate_unit='m/s**2')


def test_the_tones_cross_the_stage_by_frequency_and_time_at_their_height():
    spec = _spec()
    plotter = _Kept()
    arrays = add_sine_stage(plotter, specification=spec, unit_system=SI)
    up, down = spec.tones
    t0, t1 = 1.0, down.start_time + down.duration()
    assert plotter.bounds == (100.0, 400.0, t0, pytest.approx(t1), 0.0, 4.0), (
        'the floor is zero: height is level')
    first = np.asarray(plotter.meshes['sine-0-target'].points)
    assert first[0] == pytest.approx([0.0, 0.0, 0.5 * SZ], abs=1e-6)
    assert first[-1] == pytest.approx([SX, (1.0 + up.duration() - t0) / (t1 - t0) * SY,
                                       0.5 * SZ], abs=1e-5)
    second = np.asarray(plotter.meshes['sine-1-target'].points)
    assert second[0] == pytest.approx([SX, (4.0 - t0) / (t1 - t0) * SY, 0.25 * SZ],
                                      abs=1e-5)
    assert second[-1] == pytest.approx([SX / 3.0, SY, 0.25 * SZ], abs=1e-5)
    abort = np.asarray(plotter.meshes['sine-0-abort_upper'].points)
    assert np.allclose(abort[:, 2], SZ)
    assert arrays['points'] == len(first) + len(second) + len(abort)


def test_each_tone_is_labeled_just_before_and_above_its_start():
    plotter = _Kept()
    add_sine_stage(plotter, specification=_spec(), unit_system=SI)
    points, names = plotter.labels
    assert names == ['Up', 'Down']
    first = np.asarray(plotter.meshes['sine-1-target'].points)[0]
    assert points[0] == pytest.approx([0.0, -0.03 * SY, 0.5 * SZ + 0.04 * SZ], abs=1e-6)
    assert points[1] == pytest.approx([first[0], first[1] - 0.03 * SY,
                                       0.25 * SZ + 0.04 * SZ], abs=1e-5)


def test_nothing_to_stage_draws_nothing():
    plotter = _Kept()
    assert add_sine_stage(plotter)['points'] == 0
    assert plotter.meshes == {} and plotter.bounds is None
