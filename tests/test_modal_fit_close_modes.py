"""Fitting modes whose half-power widths overlap, and fits whose
references are not all measured as responses.

The plant is synthetic: seven modes at 1 % damping, three of them close
enough that their half-power widths overlap, over twelve response-only
DOFs and three drive points. The FRFs are its exact accelerance, so the
fit is judged against the modal model that made them.

Ported from Hertzforge (its 8fe67687), where the bias was found on a
simulated plate test: a pole searched alone beside a close neighbor came
out at half its damping.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.data import Frf
from visualdynamics.core.modal_fit import ModalFitSession
from visualdynamics.core.shapes import ShapeSet, cross_mac

#: 150 and 2800 Hz lie outside the fitted band and leave their tails in it
FREQUENCIES = (150.0, 418.1, 1125.4, 1190.7, 1216.0, 1950.5, 2800.0)
CLOSE = (1125.4, 1190.7, 1216.0)
ALONE = (418.1, 1950.5)
RESPONSE_ONLY = [f'{100 + n}Z+' for n in range(1, 13)]
DRIVES = ['201Z+', '202Z+', '203Z+']
BOUNDS = (300.0, 2400.0)


def _plant():
    """(truth, FRF): every DOF against every drive, at 1 Hz."""
    rng = np.random.default_rng(3)
    dofs = RESPONSE_ONLY + DRIVES
    truth = ShapeSet(FREQUENCIES, np.full(len(FREQUENCIES), 0.01), dofs,
                     rng.standard_normal((len(FREQUENCIES), len(dofs))))
    lines = np.arange(200.0, 3001.0, 1.0)
    responses = [dof for dof in dofs for _ in DRIVES]
    references = [ref for _ in dofs for ref in DRIVES]
    rows = truth.synthesize_frf(lines, responses, references, power=2)
    return truth, Frf(lines, rows, response_dof=responses,
                      reference_dof=references)


def _against(fit, truth, frequency):
    """(fitted mode, truth's amplitude over the fit's, MAC) for the
    fitted mode nearest `frequency`, on the DOFs both sets hold."""
    i = int(np.argmin(np.abs(fit.frequency - frequency)))
    j = int(np.argmin(np.abs(truth.frequency - frequency)))
    shared = [dof for dof in fit.coordinate if dof in truth.coordinate]
    a = fit.shape_matrix[i, [fit.coordinate.index(dof) for dof in shared]]
    b = truth.shape_matrix[j, [truth.coordinate.index(dof) for dof in shared]]
    return (i, abs(float(b @ a)) / float(a @ a),
            float(a @ b) ** 2 / float((a @ a) * (b @ b)))


def _project():
    truth, frf = _plant()
    project = visualdynamics.Project('close modes')
    project.add('FRF', frf)
    project.add('Truth', truth)
    return project, truth


def test_refine_finds_the_poles_of_overlapping_modes():
    """Each mode's pole is searched alone on the residual, so the poles
    of modes whose widths overlap come out biased, and their amplitudes
    with them. Refine searches the poles of overlapping modes together,
    and the cluster comes back on the truth while the modes that stand
    alone are left as they were."""
    # five modes, the five in the band: asked for more, the automatic
    # loop goes on to confirm wide modes of what is left, and one of
    # those reaches the 1950.5 Hz mode, which is then no longer alone
    project, truth = _project()
    sequential = project[project.fit_modes('FRF', bounds=BOUNDS, limit=5,
                                           name='Sequential')]
    worst = max(abs(sequential.damping[_against(sequential, truth, f)[0]]
                    - 0.01) for f in CLOSE)
    assert worst > 0.002, ('the one-at-a-time poles are visibly off, or '
                           'this checks nothing')

    fit = project[project.fit_modes('FRF', bounds=BOUNDS, limit=5,
                                    refine=3)]
    assert len(fit.frequency) == 5
    for frequency in CLOSE:
        i, amplitude, mac = _against(fit, truth, frequency)
        assert fit.frequency[i] == pytest.approx(frequency, abs=0.1)
        assert fit.damping[i] == pytest.approx(0.01, abs=0.001), \
            f'{frequency} Hz'
        assert amplitude == pytest.approx(1.0, abs=0.05), f'{frequency} Hz'
        assert mac > 0.999
    for frequency in ALONE:
        i, amplitude, _mac = _against(fit, truth, frequency)
        k, before, _ = _against(sequential, truth, frequency)
        assert fit.frequency[i] == sequential.frequency[k], \
            'a mode alone keeps its pole'
        assert fit.damping[i] == sequential.damping[k]
        assert abs(amplitude - 1.0) <= abs(before - 1.0) + 1e-9


def test_the_joint_pole_search_holds_what_it_cannot_move():
    """One channel buried in noise, which its coherence says to distrust:
    the weighted loop confirms the close modes and, around them, modes
    that are only noise. Those are held at their poles while the real
    modes of their cluster are searched, so the noise does not cost the
    cluster its refinement."""
    from visualdynamics.core.data import MultipleCoherence

    truth, frf = _plant()
    rng = np.random.default_rng(11)
    noisy = frf.ordinate.copy()
    bad = [i for i, dof in enumerate(frf.response_dof) if dof == '104Z+']
    level = np.median(np.abs(noisy[bad]))
    noisy[bad] += level * (rng.standard_normal(noisy[bad].shape)
                           + 1j * rng.standard_normal(noisy[bad].shape))
    frf = Frf(frf.abscissa, noisy, response_dof=frf.response_dof,
              reference_dof=frf.reference_dof)
    dofs = RESPONSE_ONLY + DRIVES
    gamma = np.full((len(dofs), len(frf.abscissa)), 0.98)
    gamma[dofs.index('104Z+')] = 0.05
    coherence = MultipleCoherence(frf.abscissa, gamma, response_dof=dofs)
    project = visualdynamics.Project('weighted')
    project.add('FRF', frf)
    fit = project[project.fit_modes('FRF', bounds=BOUNDS, limit=10, refine=3,
                                    coherence=coherence)]
    for frequency in CLOSE:
        i, _amplitude, _mac = _against(fit, truth, frequency)
        assert fit.frequency[i] == pytest.approx(frequency, abs=0.2)
        assert fit.damping[i] == pytest.approx(0.01, abs=0.001), \
            f'{frequency} Hz'


def test_a_refined_mode_keeps_the_pole_it_was_confirmed_at():
    _truth, frf = _plant()
    session = ModalFitSession(frf)
    session.suggest(BOUNDS)
    for count in range(5):
        session.confirm(description=f'mode {count + 1}')
        session.suggest(BOUNDS)
    picks = list(session.confirmed)
    assert [session.pick(k) for k in range(5)] == sorted(picks)
    session.refine_residues()
    moved = [k for k, mode in enumerate(session.modes)
             if (mode['frequency'], mode['damping']) not in picks]
    assert moved, 'Refine moved a pole, or this checks nothing'
    assert session.confirmed == picks, 'the replay list is as confirmed'
    assert sorted(session.pick(k) for k in range(5)) == sorted(picks)
    assert [mode['frequency'] for mode in session.modes] == sorted(
        mode['frequency'] for mode in session.modes)
    # a set opened for editing has no confirm order: each mode's pick is
    # its pole
    again = ModalFitSession(frf)
    again.adopt(session.shape_set())
    assert [again.pick(k) for k in range(5)] == [
        (mode['frequency'], mode['damping']) for mode in session.modes]


def test_a_refined_pole_keeps_its_place_in_the_replay(window, pump):
    """Refine moves the poles of overlapping modes, and the replay list
    holds the poles they were confirmed at: taking a refined mode back,
    typing its damping and journaling the fit must still find its
    pick."""
    from conftest import select_objects
    from PySide6.QtCore import Qt

    _truth, frf = _plant()
    window.add_object('FRF', frf)
    pump()
    select_objects(window, pump, 'FRF')
    window.start_modal_fit()
    pump()
    session = window.fit
    session.suggest(BOUNDS)
    for count in range(5):
        session.confirm(description=f'mode {count + 1}')
        session.suggest(BOUNDS)
    window._publish_fit()
    picks = list(session.confirmed)
    window.refine_all_modes()
    pump()
    moved = [k for k, mode in enumerate(session.modes)
             if (mode['frequency'], mode['damping']) not in picks]
    assert moved, 'Refine moved a pole, or this checks nothing'
    assert f'{len(moved)} mode' in window._status_text
    assert 'moved in frequency or damping' in window._status_text
    k = moved[0]
    line = window.project.journal[-1]
    assert line.startswith("project.fit_modes('FRF', at=")
    assert all(f"'mode {n}'" in line for n in range(1, 6)), \
        'every description rides its pick'
    assert 'refine=1' in line
    assert all(repr(pick[0]) in line for pick in picks), \
        'the picks are the poles as confirmed'
    # the table shows the pole as it is now
    model = window._fit_model
    assert float(model.data(model.index(k, 1), Qt.ItemDataRole.EditRole)) \
        == pytest.approx(session.modes[k]['frequency'], abs=1e-3)

    assert model.setData(model.index(k, 2), '2', Qt.ItemDataRole.EditRole)
    pump()
    assert session.modes[k]['damping'] == 0.02
    assert len(session.confirmed) == 5
    assert sum(pick[1] == 0.02 for pick in session.confirmed) == 1

    window._delete_fit_modes([k])
    pump()
    assert len(session.confirmed) == 4, \
        'the refined mode left the replay list too'
    assert all(pick[1] != 0.02 for pick in session.confirmed)


def test_reference_only_dofs_stay_out_of_an_unscaled_mac():
    """Fitted without a drive point, a set's reference DOFs hold the
    references' participation, on a scale of its own: the products of
    response and reference values are measured, their split is not. A
    MAC over the response DOFs and those together reads a good fit as a
    poor one, so the MAC leaves them out of an unscaled set."""
    project, truth = _project()
    frf = project['FRF']
    rows = [i for i, dof in enumerate(frf.response_dof)
            if dof in RESPONSE_ONLY]
    name = project.fit_modes('FRF', bounds=(300.0, 1000.0), limit=1,
                             records=rows)
    fit = project[name]
    assert fit.unscaled
    assert fit.reference_only == DRIVES
    assert set(fit.coordinate) == set(RESPONSE_ONLY + DRIVES), (
        'the participation is kept, for synthesis and Edit Fit')

    mac = project.comparison_mac(name, 'Truth')
    assert mac[0, FREQUENCIES.index(418.1)] > 0.999
    together = ShapeSet(fit.frequency, fit.damping, fit.coordinate,
                        fit.shape_matrix)
    assert cross_mac(together, truth)[0, 1] < 0.95, (
        'with the participation in, the MAC is visibly wrong, or this '
        'checks nothing')
    assert cross_mac(truth, fit)[FREQUENCIES.index(418.1), 0] > 0.999, \
        'whichever of the two is the unscaled one'


def test_a_scaled_fit_keeps_its_reference_dofs_in_the_mac():
    """With a drive point the scale is measured, and the value at a
    reference-only DOF is the mode shape there."""
    project, truth = _project()
    frf = project['FRF']
    rows = [i for i, dof in enumerate(frf.response_dof)
            if dof in [*RESPONSE_ONLY, '201Z+']]
    fit = project[project.fit_modes('FRF', bounds=(300.0, 1000.0), limit=1,
                                    records=rows)]
    assert not fit.unscaled
    assert fit.reference_only == ['202Z+', '203Z+']
    j = FREQUENCIES.index(418.1)
    sign = np.sign(fit.shape_matrix[0, 0] * truth.shape_matrix[j, 0])
    for dof in fit.reference_only:
        assert sign * fit.shape_matrix[0, fit.coordinate.index(dof)] == \
            pytest.approx(truth.shape_matrix[j, truth.coordinate.index(dof)],
                          rel=0.01)
    assert cross_mac(fit, truth)[0, j] > 0.9999


def test_a_mac_on_nothing_but_participation_is_refused():
    fit = ShapeSet([10.0], [0.01], ['1Z+', '2Z+'], [[1.0, 2.0]],
                   unscaled=True, reference_only=['2Z+'])
    other = ShapeSet([10.0], [0.01], ['2Z+', '3Z+'], [[1.0, 1.0]])
    with pytest.raises(ValueError, match='reference'):
        cross_mac(fit, other)


def test_reference_only_is_kept_in_coordinate_order_and_checked():
    held = ShapeSet([10.0], [0.01], ['1Z+', '2Z+', '3Z+'], [[1.0, 2.0, 3.0]],
                    reference_only=['3Z+', '2Z+'])
    assert held.reference_only == ['2Z+', '3Z+']
    assert ShapeSet([10.0], [0.01], ['1Z+'], [[1.0]]).reference_only == []
    with pytest.raises(ValueError, match='reference_only names 9Z'):
        ShapeSet([10.0], [0.01], ['1Z+'], [[1.0]], reference_only=['9Z+'])


def test_merged_sets_keep_their_reference_only_dofs():
    from visualdynamics.core.merge import merge

    first = ShapeSet([10.0], [0.01], ['1Z+', '2Z+'], [[1.0, 2.0]],
                     unscaled=True, reference_only=['2Z+'])
    second = ShapeSet([20.0], [0.01], ['1Z+', '2Z+'], [[2.0, 1.0]])
    assert merge([first, second]).reference_only == ['2Z+']


def test_reference_only_rides_the_project_file(tmp_path):
    project = visualdynamics.Project('file')
    project.add('Fit', ShapeSet([10.0], [0.01], ['1Z+', '2Z-'], [[1.0, 2.0]],
                                unscaled=True, reference_only=['2Z-']))
    project.add('Plain', ShapeSet([10.0], [0.01], ['1Z+'], [[1.0]]))
    project.save(tmp_path / 'fit.vdyn')
    back = visualdynamics.Project.open(tmp_path / 'fit.vdyn')
    assert back['Fit'].reference_only == ['2Z-']
    assert back['Plain'].reference_only == []


# ---- the joint pole search, on its own ----------------------------------------

#: the close modes as the one-at-a-time search leaves them
BIASED = [(1125.4, 0.01), (1191.62, 0.0174), (1216.62, 0.0075)]


def _searched(session, poles):
    """The joint search over the band the poles span, on the measurement
    as it stands: nothing taken out, so the modes outside the band are
    in it as their tails."""
    low = min(f * (1 - 3 * z) for f, z in poles)
    high = max(f * (1 + 3 * z) for f, z in poles)
    lines = np.arange(
        max(int(np.searchsorted(session.frequencies, low)) - 2, 0),
        min(int(np.searchsorted(session.frequencies, high)) + 3,
            len(session.frequencies)))
    return session._joint_poles(lines, session.matrix[lines], poles)


def _on_the_truth(found, tolerance=0.005):
    for (frequency, damping), truth in zip(found, CLOSE):
        assert frequency == pytest.approx(truth, abs=tolerance)
        assert damping == pytest.approx(0.01, abs=2e-5)


def test_the_search_allows_for_the_modes_outside_its_band():
    """Four modes outside the band leave their tails in it. A mass line
    and a stiffness line per record stand for them, and the poles come
    out on the truth with nothing taken out of the measurement."""
    _truth, frf = _plant()
    _on_the_truth(_searched(ModalFitSession(frf), BIASED))


def test_a_pole_on_its_limit_is_held_and_the_rest_are_searched():
    """A stray mode at half of critical among three real ones cannot
    move to anything better. It keeps its pole to the digit, and the
    real ones are searched with it held."""
    _truth, frf = _plant()
    stray = (1168.88, 0.5)
    found = _searched(ModalFitSession(frf), [*BIASED, stray])
    assert found[3] == stray
    for (frequency, damping), truth in zip(found[:3], CLOSE):
        assert frequency == pytest.approx(truth, abs=0.01)
        assert damping == pytest.approx(0.01, abs=1e-4)
    weak = (1150.0, 1e-4)
    found = _searched(ModalFitSession(frf), [*BIASED, weak])
    assert found[3] == weak, 'and one at the least damping a fit allows'
    _on_the_truth(found[:3])


def test_two_poles_on_one_mode_are_both_held():
    """With free residues one pole can stand for two modes, so two
    poles that end within a line of each other are one mode counted
    twice: both keep their poles and the others are searched."""
    _truth, frf = _plant()
    twice = [(1125.4, 0.01), (1190.2, 0.01), (1191.2, 0.012), (1216.0, 0.01)]
    found = _searched(ModalFitSession(frf), twice)
    assert found[1] == twice[1] and found[2] == twice[2]
    assert found[0][0] == pytest.approx(1125.4, abs=0.01)


def test_a_search_with_nothing_to_move_gives_nothing():
    _truth, frf = _plant()
    session = ModalFitSession(frf)
    assert _searched(session, [(1168.88, 0.5)]) is None, 'every pole held'
    lines = np.arange(900, 903)
    assert session._joint_poles(lines, session.matrix[lines],
                                [(1125.4, 0.01)]) is None, 'too few lines'
    assert session._joint_poles(lines, session.matrix[lines],
                                [(0.0, 0.01)]) is None, 'a rigid mode'


def test_the_search_counts_each_response_by_its_coherence():
    """One channel under twenty times its own level of noise, which its
    coherence says to distrust: weighted, the poles are on the truth;
    unweighted, the noise moves them."""
    from visualdynamics.core.data import MultipleCoherence

    _truth, frf = _plant()
    rng = np.random.default_rng(11)
    noisy = frf.ordinate.copy()
    bad = [i for i, dof in enumerate(frf.response_dof) if dof == '104Z+']
    level = 20.0 * np.median(np.abs(noisy[bad]))
    noisy[bad] += level * (rng.standard_normal(noisy[bad].shape)
                           + 1j * rng.standard_normal(noisy[bad].shape))
    frf = Frf(frf.abscissa, noisy, response_dof=frf.response_dof,
              reference_dof=frf.reference_dof)
    dofs = RESPONSE_ONLY + DRIVES
    gamma = np.full((len(dofs), len(frf.abscissa)), 0.98)
    gamma[dofs.index('104Z+')] = 0.05
    coherence = MultipleCoherence(frf.abscissa, gamma, response_dof=dofs)
    _on_the_truth(_searched(ModalFitSession(frf, coherence=coherence), BIASED))
    unweighted = _searched(ModalFitSession(frf), BIASED)
    assert abs(unweighted[0][0] - 1125.4) > 0.02, \
        'the noise is enough to matter, or this checks nothing'


def test_a_scaled_sets_reference_dofs_count_in_the_mac():
    """`reference_only` leaves a DOF out only where the set is unscaled:
    with a drive point the value there is the shape's."""
    fit = ShapeSet([10.0], [0.01], ['1Z+', '2Z+'], [[1.0, 2.0]],
                   reference_only=['2Z+'])
    other = ShapeSet([10.0], [0.01], ['1Z+', '2Z+'], [[1.0, -2.0]])
    assert cross_mac(fit, other)[0, 0] == pytest.approx(9.0 / 25.0)
    loose = ShapeSet([10.0], [0.01], ['1Z+', '2Z+'], [[1.0, 2.0]],
                     unscaled=True, reference_only=['2Z+'])
    assert cross_mac(loose, other)[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize('damping, held', [(0.45, False), (0.7, True)])
def test_a_damping_past_what_a_fit_allows_is_not_taken(damping, held):
    """The search may end inside its own bounds at a damping no fit here
    can have (`DAMPING_LIMITS`). That pole is held, and with it the only
    one, the search gives nothing; just inside the limit it is taken."""
    from visualdynamics.core.modal_fit import DAMPING_LIMITS

    assert DAMPING_LIMITS == (1e-4, 0.5)
    dofs = ['1Z+', '2Z+', '3Z+']
    truth = ShapeSet([1000.0], [damping], dofs, [[1.0, -0.6, 0.3]])
    lines = np.arange(200.0, 3001.0, 1.0)
    frf = Frf(lines, truth.synthesize_frf(lines, dofs, ['1Z+'] * 3, power=2),
              response_dof=dofs, reference_dof=['1Z+'] * 3)
    session = ModalFitSession(frf)
    band = np.arange(300, 1300)
    found = session._joint_poles(band, session.matrix[band], [(1000.0, 0.3)])
    if held:
        assert found is None
    else:
        assert found[0][0] == pytest.approx(1000.0, abs=0.5)
        assert found[0][1] == pytest.approx(damping, abs=1e-3)
