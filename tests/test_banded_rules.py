"""Where the banded stage puts things, read off its meshes
(2026-10-11).

`tools/mutate.py` over viz/banded.py caught 60 of 109 against the
stage's tests, which count stations and pixels. The stage is a
normalization — frequency along, channel in depth, level up, each onto
`STAGE` — and 35 survivors moved something along one of those axes.
A plotter that keeps what it is given stands in for the window.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_banded_stage import FREQ, _measured, _specification

from visualdynamics.units import SI
from visualdynamics.viz.banded import add_banded_stage
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

    def remove_actor(self, name):
        self.meshes.pop(name, None)


def _staged(*args, **kwargs):
    plotter = _Kept()
    # in SI, so the levels are the decades the fixtures are written in
    arrays = add_banded_stage(plotter, *args, unit_system=SI, **kwargs)
    return plotter, arrays


def test_each_channel_stands_at_its_depth_across_the_whole_stage():
    plotter, arrays = _staged(_specification())
    x0, x1, z0, z1 = arrays['extents']
    assert (x0, x1) == (FREQ[0], FREQ[-1])
    # flat at 1e-2, read in decades, between aborts 6 dB either side
    assert (z0, z1) == (pytest.approx(-2.6), pytest.approx(-1.4))
    for k in range(3):
        points = np.asarray(plotter.meshes[f'banded-{k}-target'].points)
        assert points[0, 0] == pytest.approx(0.0) and points[-1, 0] == pytest.approx(SX)
        assert np.allclose(points[:, 1], k / 2 * SY)
        assert np.allclose(points[:, 2], 0.5 * SZ, atol=1e-6), 'midway between the aborts'
    assert plotter.bounds == (x0, x1, 0, 2, z0, z1)


def test_the_labels_sit_just_before_and_above_each_channels_start():
    plotter, _arrays = _staged(_specification())
    points, names = plotter.labels
    assert names == ['101Z+', '104Z+', '110Z+']
    assert np.allclose(points, [[-0.03 * SX, k / 2 * SY, 0.5 * SZ + 0.04 * SZ]
                                for k in range(3)], atol=1e-6)


def test_one_channel_stands_at_the_front_and_its_axis_still_has_depth():
    plotter, _arrays = _staged(_specification(channels=('101Z+',)))
    assert np.allclose(np.asarray(plotter.meshes['banded-0-target'].points)[:, 1], 0.0)
    assert plotter.bounds[2:4] == (0, 1)


def test_an_exceedance_is_boxed_from_its_limit_to_the_top_over_its_bins():
    """Three lines 9 dB over a flat target, past the 6 dB abort: the
    box stands on the abort and reaches the top of the stage."""
    plotter, arrays = _staged(_specification(), _measured(loud_lines=3))
    x0, x1, z0, z1 = arrays['extents']
    (name,) = [n for n in plotter.meshes if n.endswith('exceed_over')]
    corners = np.asarray(plotter.meshes[name].points)
    assert np.allclose(corners[:, 1], 0.0), 'on the first channel'
    abort = (-2.0 + 0.6 - z0) / (z1 - z0) * SZ
    assert sorted(set(np.round(corners[:, 2], 5))) == pytest.approx([abort, SZ], abs=1e-5)
    spacing = FREQ[1] - FREQ[0]
    low = (FREQ[40] - spacing / 2 - x0) / (x1 - x0) * SX
    high = (FREQ[42] + spacing / 2 - x0) / (x1 - x0) * SX
    assert corners[:, 0].min() == pytest.approx(low, abs=1e-5)
    assert corners[:, 0].max() == pytest.approx(high, abs=1e-5)
