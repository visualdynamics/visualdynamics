"""The pole search: honest on its own model, and honest about the rest.

A day of measurement behind this file (2026-08-28), and the story it
pins is not the one the day started with.

The **frequency walk**: the search\'s fixed two-pass grid stopped *on*
its edge when the cost minimum lay just outside, and answered the edge
as though converged — on the plate\'s own recording the 823 Hz mode
came back 0.29 Hz low from an unpadded FRF and right from a padded
one. Zero padding appeared to help mode fitting and was only ever
shortening that leash. The search now re-centers while its winner sits
on the frequency edge, bounded by the neighboring line\'s territory
and by confirmed modes\' half-power claims; on the plate the fit no
longer depends on the grid the FRF was computed on — which is what let
the zero-padding feature be removed entirely the same day.

The **damping stays leashed**: a walking damping was measured to
inflate cluster fits into blends, since a fatter mode always explains
more of a cluster\'s band.

The **residue constraint stays real**: complex residues were tried —
they cure the pole\'s sensitivity to broadband phase the model cannot
express — and were measured to pull the plate\'s true real modes ~0.25
Hz low by chasing leakage\'s phase instead. The real constraint is the
model this fitter declares, real chains cancel common phase in the FRF
ratio, and the residual limitation is pinned below with its size.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Frf
from visualdynamics.core.modal_fit import ModalFitSession

FS = 2048.0
F_TRUE, Z_TRUE = 180.0, 0.02


def continuous(freq):
    omega = 2 * np.pi * freq
    wr = 2 * np.pi * F_TRUE
    return 1.0 / (wr ** 2 - omega ** 2 + 2j * Z_TRUE * wr * omega)


def discrete(freq):
    """An exact discrete-time resonator: matched-z, so the pole\'s
    (frequency, damping) are exactly the continuous pair — while the
    response\'s *shape* is genuinely not a continuous SDOF\'s, which is
    the mismatch the limitation test measures."""
    wn = 2 * np.pi * F_TRUE
    pole = np.exp((-Z_TRUE * wn + 1j * wn * np.sqrt(1 - Z_TRUE ** 2)) / FS)
    a = np.poly([pole, np.conj(pole)]).real
    z = np.exp(1j * 2 * np.pi * freq / FS)
    return 1.0 / np.polyval(a[::-1], 1.0 / z)


def session(values, freq):
    frf = Frf(freq, values[None, :], response_dof=['101Z+'],
              reference_dof=['901X+'], ordinate_dim=['length/force'])
    return ModalFitSession(frf)


@pytest.fixture
def grid():
    return np.arange(2.0, 1024.0, 2.0)      # a deliberately coarse 2 Hz


def test_the_exact_model_is_recovered_exactly(grid):
    """On data the model fits, a 2 Hz grid costs nothing: the pole
    comes back to hundredths of a line."""
    s = session(continuous(grid), grid)
    index = int(np.argmin(np.abs(grid - F_TRUE)))
    frequency, damping = s.search(index)
    assert frequency == pytest.approx(F_TRUE, abs=0.02)
    assert damping == pytest.approx(Z_TRUE, abs=0.0005)


@pytest.mark.parametrize('offset', [-1, 1])
def test_the_search_walks_home_from_a_neighboring_line(grid, offset):
    """Started one line from the peak, the truth is a full line spacing
    away — beyond the old fixed schedule\'s reach, which stopped on its
    grid edge and answered it as though converged. The walk re-centers
    and arrives."""
    s = session(continuous(grid), grid)
    index = int(np.argmin(np.abs(grid - F_TRUE))) + offset
    frequency, _damping = s.search(index)
    assert frequency == pytest.approx(F_TRUE, abs=0.03)


def test_the_plate_fit_is_grid_stable():
    """The claim the frequency walk exists for, on the canonical
    recording: the modal fit does not depend on which grid the FRF
    happened to be computed on. Pinned by fitting at two frame lengths
    whose grids differ and checking the modes land together — the
    823 Hz mode is the one that used to rail 0.29 Hz low on the coarse
    grid. (This test once compared padded against unpadded FRFs; the
    zero-padding feature was removed the same day the walk made it
    unnecessary, which was the plan all along.)"""
    from dataclasses import replace

    from conftest import fixture_path

    import visualdynamics

    found = {}
    for frames, frame_length in ((10, 2048), (21, 1024)):
        project = visualdynamics.Project('grid')
        project.import_file(fixture_path('plate', 'modal.nc4'))
        history = project.time_history
        history.averaging = replace(history.averaging,
                                    frame_length=frame_length,
                                    frames=frames)
        frfs = project.compute_frfs(history, 'H1')
        project.fit_modes(frfs, bounds=(300.0, 1300.0), limit=5,
                          name='Modes')
        found[frame_length] = np.sort(project['Modes'].frequency)
    # different grids, one structure: the same five modes to within a
    # tenth of the coarse grid's spacing
    assert np.allclose(found[2048], found[1024], atol=0.4), found


# ---- the documented limitation, pinned with its size --------------------


def test_uncanceled_phase_still_moves_the_pole_boundedly(grid):
    """Data whose broadband phase the real-residue model cannot express
    — here an IIR plant whose shape is genuinely not a continuous
    SDOF\'s — biases the pole, and the walk\'s cap is what bounds the
    damage: the cost\'s own minimum is at 182.1 Hz on this data, and
    the search follows it no further than the cap allows. Pinned with
    its size so a future complex-mode fitter has a number to beat.
    (Complex residues cure this case, and were measured to pull the
    plate\'s real modes ~0.25 Hz low through leakage\'s phase — which is
    why they were tried and not kept.)"""
    s = session(discrete(grid), grid)
    index = int(np.argmin(np.abs(grid - F_TRUE)))
    frequency, damping = s.search(index)
    assert 1.5 < frequency - F_TRUE < 3.0, (
        'the bias moved: if it shrank, the cost or the constraint '
        'changed — update the pin and the story; if it grew, something '
        'broke')
    assert damping == pytest.approx(Z_TRUE, abs=0.002), \
        'the damping stays honest even where the frequency cannot'
