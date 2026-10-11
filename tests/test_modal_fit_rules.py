"""The joint residue refit and the coherence weights, pinned by value
where a mutation pass found them checked only to a MAC (2026-10-11).

`tools/mutate.py` over modal_fit.py caught 332 of 511 against its two
test files. The refit's test asked for both shapes back to a MAC of
0.9999, and a basis column built wrong, or a weight applied to one
side of the least squares and not the other, still lands there. On
data synthesized exactly from the poles the user confirmed, the joint
solve is exact — so it is held to exact here, with weights and
without. A mode far from the rest is a cluster of one and keeps its
sequential fit, and a cluster beside it is solved on what that mode
leaves (fixed 2026-10-11: it was solved on the measurement whole).

The ridge rungs of the refit's ladder are not pinned: on exact data
the unregularized rung wins, and on noisy data which rung wins is the
heuristic the ladder exists to make.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from visualdynamics.core.data import Frf
from visualdynamics.core.modal_fit import ModalFitSession

RESPONSES = ['1Z+', '2Z+', '3Z+']


def _frf(truth):
    """Accelerances synthesized from `truth`, driven at 1Z+."""
    frequencies = np.linspace(20.0, 80.0, 1201)
    omega = 2.0 * np.pi * frequencies
    rows = np.zeros((3, len(frequencies)), dtype=complex)
    for f_r, zeta, phi in truth:
        omega_r = 2.0 * np.pi * f_r
        q = ((1j * omega) ** 2
             / (omega_r ** 2 - omega ** 2 + 2j * zeta * omega_r * omega))
        rows += np.outer(phi * phi[0], q)
    return Frf(frequencies, rows, response_dof=RESPONSES,
               reference_dof=['1Z+'] * 3)


PAIR = [(50.0, 0.02, np.array([1.0, 0.5, -0.5])),
        (53.0, 0.02, np.array([0.2, 1.0, 0.6]))]
FAR = (75.0, 0.02, np.array([0.7, -0.4, 1.0]))


def _coherence(frf):
    """Varying across the lines and between the channels, so every
    line of every response's solve carries a different weight."""
    f = np.asarray(frf.abscissa)
    return SimpleNamespace(response_dof=RESPONSES, abscissa=f,
                           ordinate=np.array([0.5 + 0.4 * np.sin(f / 3.0 + k)
                                              for k in range(3)]))


def _fitted(truth, weighted=False):
    frf = _frf(truth)
    session = ModalFitSession(frf, coherence=_coherence(frf) if weighted else None)
    assert (session.weights is not None) == weighted
    for f_r, zeta, _phi in truth:
        session.confirm(frequency=f_r, damping=zeta)
    return session


@pytest.mark.parametrize('weighted', [False, True])
def test_the_joint_refit_of_exact_data_is_exact(weighted):
    session = _fitted(PAIR, weighted)
    assert not np.allclose(session.modes[0]['shape'][:3], PAIR[0][2], atol=1e-3), (
        'the pair borrows from each other before the refit')
    assert session.refine_residues() == 2
    for mode, (_f, _z, phi) in zip(session.modes, PAIR):
        assert np.allclose(mode['shape'][:3], phi, atol=1e-9)
        assert mode['scaled']


@pytest.mark.parametrize('weighted', [False, True])
def test_a_cluster_is_refit_on_what_the_other_modes_leave(weighted):
    """The pair beside a 75 Hz mode: solved on the measurement as it
    stood, the pair's residues took in the third mode's tail and came
    back 1.2 % and 4.0 % off (2026-10-11). Solved on the measurement
    less that mode at its sequential residues, the pair comes back to
    what the third mode's own fit allows — it is off by 1e-4 itself —
    and the third, a cluster of one, keeps its sequential fit."""
    session = _fitted(PAIR + [FAR], weighted)
    alone = session.modes[2]['shape'].copy()
    assert session.refine_residues() == 3
    for mode, (_f, _z, phi) in zip(session.modes, PAIR):
        assert np.allclose(mode['shape'][:3], phi, atol=1e-3)
    assert np.array_equal(session.modes[2]['shape'], alone)


def test_coherence_weighs_each_channel_by_its_inverse_variance():
    """g2 / (1 - g2), normalized to the largest: 0.75 is three and 0.5
    one, a channel the coherence does not name weighs as one, and a
    claim of perfect coherence is held at 0.999."""
    f = np.linspace(20.0, 80.0, 1201)
    frf = _frf(PAIR)

    def weights(curves):
        dofs = list(curves)
        return ModalFitSession(frf, coherence=SimpleNamespace(
            response_dof=dofs, abscissa=f,
            ordinate=np.array([np.full(f.size, curves[d]) for d in dofs]))).weights

    got = weights({'1Z+': 0.75, '2Z+': 0.5})
    assert np.allclose(got, [1.0, 1.0 / 3.0, 1.0 / 3.0])
    got = weights({'1Z+': 1.0, '2Z+': 0.5})
    assert np.allclose(got, [1.0, 1.0 / 999.0, 1.0 / 999.0])
    assert weights({'9Z+': 0.5}) is None, 'no channel of ours named: no weights'
    assert weights({'1Z+': 0.0, '2Z+': 0.0, '3Z+': 0.0}) is None, 'nothing to trust'


@pytest.mark.parametrize('low, middle, high, vertex', [
    (1.0, 0.0, 1.0, 0.0),            # symmetric: the middle
    (3.0, 1.0, 2.0, 1.0 / 6.0),      # leaning toward the lower high side
    (1.0, 1.0, 1.0, 0.0),            # flat: no vertex to find
    (0.0, 0.5, 1.0, 0.0),            # a straight rise: none either
    (0.0, 1.0, 0.0, 0.0),            # a peak, not a valley
    (10.0, 0.0, -9.0, 0.5),          # held within half a step
])
def test_three_costs_put_the_minimum_at_the_parabolas_vertex(low, middle, high, vertex):
    assert ModalFitSession._vertex(low, middle, high) == pytest.approx(vertex)


def _sdof(truth, zeta, df=0.5, lines=400):
    f = np.arange(1, lines + 1) * df
    w, wn = 2 * np.pi * f, 2 * np.pi * truth
    h = (1j * w) ** 2 / (wn ** 2 - w ** 2 + 2j * zeta * wn * w)
    return Frf(f, np.array([h, 0.7 * h, -0.3 * h]), response_dof=RESPONSES,
               reference_dof=['1Z+'] * 3, ordinate_dim='acceleration/force')


def test_the_cost_of_the_true_pole_is_nothing_and_of_any_other_something():
    """Exact data: one SDOF term at the true pole explains all of the
    projected residual. A pole on a line with no damping has no finite
    term there, and costs as much as explaining nothing."""
    session = ModalFitSession(_sdof(63.0, 0.02))
    index = int(np.argmin(np.abs(session.frequencies - 63.0)))
    cost = session._residual_cost(index, 0.08)
    assert cost(63.0, 0.02) == pytest.approx(0.0, abs=1e-20)
    assert 0.0 < cost(63.2, 0.02) < 1.0 and 0.0 < cost(63.0, 0.03) < 1.0
    assert cost(63.0, 0.0) == 1.0


@pytest.mark.parametrize(('truth', 'zeta'), [
    (63.13, 0.02), (63.00, 0.02), (120.37, 0.005), (40.62, 0.05)])
def test_the_search_lands_on_the_pole_of_exact_data(truth, zeta):
    """The cost's minimum is the true pole, so the search's error is its
    own. With two passes the damping read 0.46 % high in all four of
    these (2026-10-11); with four, within 2e-5 of the damping and 5e-5
    of a line."""
    session = ModalFitSession(_sdof(truth, zeta))
    found, damping = session.suggest()
    assert abs(found - truth) < 1e-3 * 0.5
    assert damping == pytest.approx(zeta, rel=1e-4)
    assert session.search_damping(truth) == pytest.approx(zeta, rel=1e-4)
