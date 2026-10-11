"""Where the paired stage puts a pair, by value (2026-10-11).

`tools/mutate.py` over viz/paired.py caught 23 of 84 against
test_paired_stage.py; 36 survivors were in `paired_stage_curves`, the
normalization the app's stage and the report's figure share. Flat
densities put every run at a height worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Psd
from visualdynamics.units import SI
from visualdynamics.viz.paired import add_paired_stage, paired_stage_curves
from visualdynamics.viz.waterfall import STAGE

SX, SY, SZ = STAGE
FREQ = np.linspace(2.0, 1000.0, 50)
FLOOR = 1e-6
#: an overlay spans the lines' own bins, half a line past each end
HALF = (FREQ[1] - FREQ[0]) / 2.0
X0, X1 = FREQ[0] - HALF, FREQ[-1] + HALF


def _density(levels, channels=('1X+', '2X+', '3X+')):
    return Psd(FREQ, np.asarray(levels, dtype=float)[:, None] * np.ones(len(FREQ)),
               response_dof=list(channels),
               ordinate_dim=['acceleration**2/frequency'] * len(channels))


def test_an_overlay_puts_the_louder_on_top_and_the_quieter_on_the_floor():
    """Flat at 1e-3 over flat at 1e-6, read in decades: the loud runs
    at the stage's top, the quiet at its floor, every channel at its
    depth, every run across the whole span."""
    loud, quiet = _density([1e-3] * 3), _density([FLOOR] * 3)
    arrays = paired_stage_curves(loud, quiet, unit_system=SI)
    assert arrays['extents'] == (pytest.approx(X0), pytest.approx(X1),
                                 pytest.approx(-6.0), pytest.approx(-3.0))
    assert len(arrays['runs']) == 6 and arrays['drawn'] == 3
    for run in arrays['runs']:
        k = ['1X+', '2X+', '3X+'].index(run['label'])
        points = run['points']
        assert np.allclose(points[:, 1], k / 2 * SY) and run['station'] == k / 2 * SY
        # drawn as steps over the bins, so from edge to edge
        assert points[0, 0] == pytest.approx(0.0) and points[-1, 0] == pytest.approx(SX)
        assert np.allclose(points[:, 2], 0.0 if run['quiet'] else SZ)


def test_a_ratio_is_the_signal_over_the_noise_in_decibels_on_the_stage():
    """1001, 101 and 11 times the floor are 30, 20 and 10 dB of signal:
    top, middle and floor of the stage."""
    loud, quiet = _density(np.array([1001.0, 101.0, 11.0]) * FLOOR), _density([FLOOR] * 3)
    arrays = paired_stage_curves(loud, quiet, mode='ratio', unit_system=SI)
    assert arrays['extents'] == (FREQ[0], FREQ[-1], pytest.approx(10.0), pytest.approx(30.0))
    heights = [float(run['points'][0, 2]) for run in arrays['runs']]
    assert heights == pytest.approx([SZ, SZ / 2, 0.0])
    assert [run['station'] for run in arrays['runs']] == [0.0, SY / 2, SY]
    assert np.allclose(arrays['runs'][0]['levels'], 30.0)


class _Kept:
    def __init__(self):
        self.meshes, self.labels, self.bounds = {}, None, None

    def add_mesh(self, mesh, name=None, **_style):
        self.meshes[name] = mesh

    def add_point_labels(self, points, names, **_style):
        self.labels = (np.asarray(points), list(names))

    def show_bounds(self, **style):
        self.bounds = style['axes_ranges']


def test_the_stage_labels_each_loud_run_just_before_its_start():
    plotter = _Kept()
    loud, quiet = _density([1e-3] * 3), _density([FLOOR] * 3)
    arrays = add_paired_stage(plotter, loud, quiet, unit_system=SI)
    points, names = plotter.labels
    assert names == ['1X+', '2X+', '3X+']
    assert np.allclose(points, [[-0.03 * SX, k / 2 * SY, SZ] for k in range(3)])
    assert plotter.bounds == (pytest.approx(X0), pytest.approx(X1), 0, 2,
                              pytest.approx(-6.0), pytest.approx(-3.0))
    # a step over each line's bin is two points a line
    assert arrays['points'] == 6 * 2 * len(FREQ)
    assert plotter.meshes['paired-loud'].n_points == 3 * 2 * len(FREQ)
    assert plotter.meshes['paired-quiet'].n_points == 3 * 2 * len(FREQ)
