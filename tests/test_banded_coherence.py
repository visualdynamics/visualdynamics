"""Coherence with bands, and the coherence of a CPSD.

A coherence computed from an octave-banded CPSD — |<S_pq>|² over
<S_pp><S_qq>, each band's averages — is a band's value, and needs the
bands the CPSD has. The band-average paper found `Coherence` had none:
`build_plots` stopped at `bin_bounds` (2026-09-26). The bins are now one
implementation shared with `Psd` (`data._Bands`), and `Psd.coherence`
computes the ordinary coherence of every cross term, banded or not.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Coherence, Psd

F = np.linspace(0.0, 2000.0, 2001)


def cpsd(pp=1.0, qq=4.0, pq=1.0 + 0.0j):
    """Two channels' CPSD with a flat coherence of |pq|² / (pp qq)."""
    ones = np.ones_like(F)
    return Psd(F, np.array([pp * ones, pq * ones, np.conj(pq) * ones,
                            qq * ones]),
               response_dof=['1X+', '1X+', '2X+', '2X+'],
               reference_dof=['1X+', '2X+', '1X+', '2X+'],
               ordinate_dim=['acceleration**2/frequency'] * 4)


def test_the_coherence_of_a_cpsd_is_the_ratio(qt_app):
    coherence = cpsd(pq=1.0 + 1.0j).coherence()
    assert isinstance(coherence, Coherence)
    assert list(zip(coherence.response_dof, coherence.reference_dof)) == [
        ('1X+', '2X+'), ('2X+', '1X+')]
    assert np.allclose(coherence.ordinate, 2.0 / 4.0)
    assert coherence.bandwidth is None


def test_a_banded_cpsds_coherence_is_the_bands_and_keeps_them(qt_app):
    """Each band's averaged cross spectrum against its averaged
    autospectra — not the narrowband coherence averaged, which weights a
    band's lines alike whatever their power."""
    values = cpsd()
    # the cross spectrum strong at low lines, weak at high: a band's
    # coherence is of its averages
    values.ordinate[1] = values.ordinate[2] = np.where(F < 1000.0, 1.8, 0.2)
    banded = values.to_octave(3)
    coherence = banded.coherence()
    assert coherence.bandwidth is not None
    assert np.array_equal(coherence.bandwidth, banded.bandwidth)
    expected = (np.abs(banded.ordinate[1]) ** 2
                / (banded.ordinate[0].real * banded.ordinate[3].real))
    assert np.allclose(coherence.ordinate[0], np.clip(expected, 0, 1))
    assert np.allclose(coherence.bin_edges(), banded.bin_edges())


def test_a_coherence_takes_bands_and_refuses_the_wrong_shape():
    ok = Coherence(F[:3], np.ones((1, 3)), response_dof=['1X+'],
                   reference_dof=['2X+'], bandwidth=[1.0, 2.0, 3.0])
    left, right = ok.bin_bounds()
    assert np.allclose(right - left, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='bandwidth has shape'):
        Coherence(F[:3], np.ones((1, 3)), response_dof=['1X+'],
                  reference_dof=['2X+'], bandwidth=[1.0, 2.0])


def test_a_banded_coherence_saves_and_draws(qt_app, tmp_path):
    """What can be read can be written: the bands survive a project
    file, and the plot that stopped at `bin_bounds` draws."""
    import pyqtgraph as pg

    import visualdynamics
    from visualdynamics.plot import build_plots

    coherence = cpsd().to_octave(3).coherence()
    project = visualdynamics.Project('p')
    project.add('C', coherence)
    project.save(tmp_path / 'c.vdyn')
    back = visualdynamics.Project.open(tmp_path / 'c.vdyn')['C']
    assert np.array_equal(back.bandwidth, coherence.bandwidth)
    layout = pg.GraphicsLayoutWidget()
    drawn, _requested = build_plots(layout, [('C', back, None)])
    assert drawn == back.num_records


def test_a_banded_coherence_draws_flat_across_its_bands(qt_app):
    """A band's coherence is one number for the band, and draws flat
    across it the way a banded PSD does (Brandon, 2026-09-27); a
    narrowband coherence is a value at each line and stays a line."""
    from visualdynamics.plot import drawing_shape

    banded = cpsd().to_octave(3).coherence()
    assert drawing_shape(banded) == 'steps'
    assert drawing_shape(cpsd().coherence()) == 'line'
    assert drawing_shape(banded, tagged=True) == 'line'
