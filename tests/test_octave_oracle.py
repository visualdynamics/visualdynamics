"""Octave banding of a PSD, against sdynpy's own.

Left out of the first oracle on purpose: sdynpy's banding lives in
`PowerSpectralDensityArray.bandwidth_average`, and its conventions had
not been checked against ours. Read side by side (2026-09-23) they are
the same rule — each line owns half a line spacing either side of it, a
line straddling a band edge is split by the fraction inside, and the
band's content is divided by its *full* width — so a band with data
under it must agree. Frozen by `generate_octave_oracle.py` in the
private generators repository and read here as numbers, never imported
(`AGENTS.md` hard rule 1).

The same edges go into both, sdynpy's own `nth_octave_freqs`, so what is
compared is the banding alone; the edges themselves are checked against
sdynpy's in `tests/test_octave.py`. Bands are matched by their edges,
not their labels: sdynpy names a band by the arithmetic mean of its
edges and Visual Dynamics by the geometric mean, which is where a
proportional band's center actually is.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core import octave
from visualdynamics.core.data import Psd

ORACLE = fixture_path('sdynpy_oracle', 'octave.npz')
ORDERS = (1, 3, 6, 12)


@pytest.fixture(scope='module')
def oracle():
    with np.load(ORACLE) as frozen:
        return {key: frozen[key] for key in frozen.files}


def _bounds(oracle, order):
    lower, upper = oracle[f'lower_{order}'], oracle[f'upper_{order}']
    assert np.allclose(upper[:-1], lower[1:], rtol=1e-12), 'contiguous bands'
    return np.append(lower, upper[-1])


@pytest.mark.parametrize('order', ORDERS)
def test_banding_agrees_with_sdynpys(oracle, order):
    """The banding itself, on sdynpy's own edges: two real PSDs and a
    complex CPSD cross term, which bands the same way."""
    ours = octave.resample(oracle['lines'], oracle['ordinate'],
                           _bounds(oracle, order))
    theirs = oracle[f'banded_{order}']
    assert ours.shape == theirs.shape
    assert np.isfinite(ours).all(), 'every one of these bands has data'
    scale = np.abs(theirs).max(axis=1, keepdims=True)
    worst = (np.abs(ours - theirs) / scale).max()
    assert worst < 1e-12, f'{order}/octave departs by {worst:.2e} of peak'


@pytest.mark.parametrize('order', ORDERS)
def test_a_psd_banded_the_usual_way_agrees_too(oracle, order):
    """The call a user actually makes, `Psd.to_octave`, choosing its own
    edges. Every band sdynpy produces has a counterpart here with both
    edges the same, and the same value. This package bands further down
    than sdynpy does, which is a difference of range, not of values."""
    psd = Psd(oracle['lines'], oracle['ordinate'][:2].real,
              response_dof=['1X+', '2X+'],
              ordinate_dim='acceleration**2/frequency')
    banded = psd.to_octave(order)
    lower, upper = banded.bin_bounds()
    their_lower = oracle[f'lower_{order}']
    their_upper = oracle[f'upper_{order}']
    ours, theirs = [], []
    for j, edge in enumerate(their_lower):
        i = int(np.argmin(np.abs(lower - edge)))
        assert np.isclose(lower[i], edge, rtol=1e-9), (
            f'{order}/octave: no band here starts at sdynpy\'s {edge:.6g} Hz'
        )
        assert np.isclose(upper[i], their_upper[j], rtol=1e-9), (
            f'{order}/octave: the band at {edge:.6g} Hz ends elsewhere'
        )
        ours.append(np.asarray(banded.ordinate)[:, i])
        theirs.append(oracle[f'banded_{order}'][:2, j].real)
    ours, theirs = np.array(ours).T, np.array(theirs).T
    worst = (np.abs(ours - theirs) / np.abs(theirs).max()).max()
    assert worst < 1e-12, f'{order}/octave departs by {worst:.2e} of peak'


def test_the_frozen_case_exercises_the_split_at_the_edges(oracle):
    """So the agreement means something: band edges fall inside lines,
    not on their boundaries, so the fraction-of-a-line split is what is
    being compared — and a complex record is there."""
    half = (oracle['lines'][1] - oracle['lines'][0]) / 2.0
    edges = oracle['lower_3']
    on_boundary = np.isclose(np.mod(edges, half), 0.0, atol=1e-9)
    assert (~on_boundary).mean() > 0.9, 'most edges split a line'
    assert np.iscomplexobj(oracle['ordinate']) and np.abs(
        oracle['ordinate'][2].imag).max() > 0, 'a complex cross term'


@pytest.mark.parametrize('order', ORDERS)
def test_a_partly_filled_band_is_divided_by_its_full_width_in_both(
        oracle, order):
    """The top band at every density runs past the end of the data — 2%
    to 48% covered — and both tools divide its content by the band's
    *whole* width, not the part with data under it. So it reads low in
    both, and agrees. That is the rule decided twice on 2026-09-19 (a
    band is a defined frequency band, not trimmed to the data), and
    sdynpy does the same; dividing by the covered width instead fails
    `test_banding_agrees_with_sdynpys` on exactly these bands."""
    step = oracle['lines'][1] - oracle['lines'][0]
    top = oracle['lines'][-1] + step / 2.0
    lower, upper = oracle[f'lower_{order}'][-1], oracle[f'upper_{order}'][-1]
    covered = (min(upper, top) - lower) / (upper - lower)
    assert 0.0 < covered < 1.0, f'the top band is partly covered: {covered:.0%}'
