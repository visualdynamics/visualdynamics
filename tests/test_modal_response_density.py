"""Modal responses to a force CPSD: each mode answering the forces' cross
spectral density, as the modal response CPSD with every cross term.

Brandon, 2026-10-30: Compute Modal Responses is offered on a force CPSD
and a shape set, as it is on a force time history, and Expand takes the
result to the set's DOFs, so the CPSD chain uses the same buttons as the
time-history chain. Expanded, it must be the synthesized FRFs driven by
the forces, conj(H) S_ff Hᵀ, cross terms and phase included. A modal
force CPSD was offered Expand and refused it when pressed; it is not
offered now.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import bar_acts, select_objects
from scipy.linalg import eigh

import visualdynamics
from visualdynamics.core import modal_response as mr
from visualdynamics.core.data import Psd
from visualdynamics.core.drive_prediction import through
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.core.transform import expands, to_modal, to_physical

#: a free-free chain of four masses: one rigid mode and three flexible
M = np.diag([2.0, 1.0, 1.5, 1.0])
K = 1e5 * np.array([[1.0, -1.0, 0.0, 0.0], [-1.0, 3.0, -2.0, 0.0],
                    [0.0, -2.0, 3.0, -1.0], [0.0, 0.0, -1.0, 1.0]])
DOFS = ['1X+', '2X+', '3X+', '4X+']
DRIVES = ['1X+', '3X+']
LINES = np.linspace(0.0, 200.0, 401)


def _model():
    lam, phi = eigh(K, M)                       # mass-normalized
    frequency = np.sqrt(np.clip(lam, 0.0, None)) / (2 * np.pi)
    frequency[0] = 0.0                          # the rigid mode, exactly
    return ShapeSet(frequency, np.array([0.0, 0.02, 0.03, 0.01]), DOFS,
                    phi.T, mass_unit='kg')


def _forces(correlated=True):
    """(lines, 2, 2) Hermitian force density, positive definite, its
    cross term carrying a phase that turns with frequency."""
    a = 1.0 + 0.5 * np.sin(LINES / 30.0)
    b = 2.0 + np.cos(LINES / 50.0)
    c = (0.7 * np.sqrt(a * b) * np.exp(1j * LINES / 40.0) if correlated
         else np.zeros_like(LINES, dtype=complex))
    s = np.empty((LINES.size, 2, 2), dtype=complex)
    s[:, 0, 0], s[:, 1, 1] = a, b
    s[:, 0, 1], s[:, 1, 0] = c, np.conj(c)
    return s


def _density(s, autos_only=False):
    """The force density as a CPSD object: a record per (row, column)
    pair, each stored as conj(F_a)·F_b, as compute_cpsds stores it."""
    if autos_only:
        return Psd(LINES, np.array([s[:, a, a] for a in range(2)]),
                   response_dof=DRIVES, ordinate_dim='force**2/frequency')
    rows = [s[:, a, b] for a in range(2) for b in range(2)]
    return Psd(LINES, np.array(rows),
               response_dof=[DRIVES[a] for a in range(2) for _b in range(2)],
               reference_dof=[DRIVES[b] for _a in range(2) for b in range(2)],
               ordinate_dim=['force**2/frequency' if a == b else
                             'force*force/frequency'
                             for a in range(2) for b in range(2)])


def _matrix(psd, dofs):
    """(lines, n, n) from a density's records."""
    at = {(r, c): i for i, (r, c) in
          enumerate(zip(psd.response_dof, psd.reference_dof))}
    return np.stack([np.stack([psd.ordinate[at[(r, c)]] for c in dofs], -1)
                     for r in dofs], 1)


def _synthesized(shapes, s, power):
    """conj(H) S_ff Hᵀ with H from synthesize_frf, at every DOF."""
    pairs = [(r, d) for r in DOFS for d in DRIVES]
    frf = shapes.synthesize_frf(LINES, [r for r, _ in pairs],
                                [d for _, d in pairs], power=power)
    h = frf.T.reshape(LINES.size, len(DOFS), len(DRIVES))
    return through(h, s)


# ---- exact, against the synthesized FRFs ------------------------------------

@pytest.mark.parametrize('quantity, power', [('acceleration', 2),
                                             ('velocity', 1),
                                             ('displacement', 0)])
def test_expanded_it_is_the_synthesized_frfs_driving_the_forces(quantity,
                                                                 power):
    shapes, s = _model(), _forces()
    modal = mr.modal_response_density(_density(s), shapes, quantity=quantity)
    assert modal.num_records == 16
    assert set(modal.ordinate_dim) == {
        {'acceleration': 'modal_acceleration', 'velocity': 'modal_velocity',
         'displacement': 'modal_length'}[quantity] + '**2/frequency'}
    physical, _report = to_physical(modal, shapes)
    got = _matrix(physical, DOFS)
    want = _synthesized(shapes, s, power)
    lines = slice(None) if power == 2 else slice(1, None)
    scale = np.abs(want[lines]).max()
    assert np.abs(got[lines] - want[lines]).max() < 1e-12 * scale
    # the cross terms carry phase: a real comparison would pass a
    # conjugated answer
    assert np.abs(np.angle(want[lines, 0, 2])).max() > 0.5


def test_the_rigid_mode_cancels_at_0_hz_in_acceleration():
    """At 0 Hz the rigid mode's (iω)² against its −ω² is exactly 1, so
    the modal acceleration there is the modal force over the mass."""
    shapes, s = _model(), _forces()
    modal = mr.modal_response_density(_density(s), shapes)
    first = modal.response_dof.index('M1')
    assert np.isfinite(modal.ordinate[:, 0]).all()
    phi = np.real(shapes.shape_matrix[0])
    drive = [DOFS.index(d) for d in DRIVES]
    expected = phi[drive] @ s[0] @ phi[drive]
    assert modal.ordinate[first, 0] == pytest.approx(expected, rel=1e-12)


def test_zero_cross_terms_are_uncorrelated_drives_and_go_through():
    shapes, s = _model(), _forces(correlated=False)
    modal = mr.modal_response_density(_density(s), shapes)
    physical, _report = to_physical(modal, shapes)
    want = _synthesized(shapes, s, 2)
    assert np.abs(_matrix(physical, DOFS) - want).max() < \
        1e-12 * np.abs(want).max()


def test_a_nan_line_does_not_take_the_phase_off_a_density():
    """A rigid mode's displacement at 0 Hz is nan; Psd kept the real
    part alone whenever any line was nan, because the check for a phase
    compared two nan maxima."""
    ordinate = np.array([[np.nan, 1.0 + 1.0j, 2.0 - 0.5j]])
    psd = Psd(LINES[:3], ordinate, response_dof=['1X+'],
              reference_dof=['2X+'], ordinate_dim='force*force/frequency')
    assert np.iscomplexobj(psd.ordinate)
    assert psd.ordinate[0, 1] == 1.0 + 1.0j


def test_autospectra_alone_at_several_drives_are_refused():
    with pytest.raises(ValueError, match='autospectra only'):
        mr.modal_response_density(_density(_forces(), autos_only=True),
                                  _model())


def test_a_modal_force_density_is_used_as_it_is():
    shapes, s = _model(), _forces()
    pp, _report = to_modal(_density(s), shapes)
    assert set(pp.ordinate_dim) == {'modal_force**2/frequency'}
    direct = mr.modal_response_density(_density(s), shapes)
    from_modal = mr.modal_response_density(pp, shapes)
    assert np.abs(from_modal.ordinate - direct.ordinate).max() < \
        1e-12 * np.abs(direct.ordinate).max()


def test_modal_force_records_picked_beside_others_are_the_forces():
    """A density holding modal forces and modal responses side by side:
    the forces picked are taken alone, every attribute of theirs."""
    shapes, s = _model(), _forces()
    pp, _report = to_modal(_density(s), shapes)
    qq = mr.modal_response_density(_density(s), shapes)
    # the responses in the reverse order, so a force record found by
    # the wrong list's DOFs would be a different pair
    both = Psd(LINES, np.vstack([qq.ordinate[::-1], pp.ordinate]),
               response_dof=qq.response_dof[::-1] + pp.response_dof,
               reference_dof=qq.reference_dof[::-1] + pp.reference_dof,
               ordinate_dim=qq.ordinate_dim[::-1] + pp.ordinate_dim)
    forces = list(range(qq.num_records, both.num_records))
    again = mr.modal_response_density(both, shapes, records=forces)
    assert again.response_dof == qq.response_dof
    np.testing.assert_allclose(again.ordinate, qq.ordinate, rtol=1e-13)


def test_only_the_modes_picked_and_autospectra_for_a_preview():
    shapes, s = _model(), _forces()
    full = mr.modal_response_density(_density(s), shapes)
    some = mr.modal_response_density(_density(s), shapes, modes=[1, 3])
    assert some.response_dof == ['M2', 'M2', 'M4', 'M4']
    assert some.reference_dof == ['M2', 'M4', 'M2', 'M4']
    for i, (r, c) in enumerate(zip(some.response_dof, some.reference_dof)):
        j = list(zip(full.response_dof, full.reference_dof)).index((r, c))
        np.testing.assert_allclose(some.ordinate[i], full.ordinate[j],
                                   rtol=1e-14)
    autos = mr.modal_response_density(_density(s), shapes, cross=False)
    assert autos.response_dof == autos.reference_dof == ['M1', 'M2', 'M3',
                                                         'M4']
    for i, dof in enumerate(autos.response_dof):
        j = list(zip(full.response_dof, full.reference_dof)).index((dof, dof))
        np.testing.assert_allclose(autos.ordinate[i], full.ordinate[j],
                                   rtol=1e-14)


def test_what_is_refused():
    shapes, density = _model(), _density(_forces())
    with pytest.raises(ValueError, match='not a response'):
        mr.modal_response_density(density, shapes, quantity='jerk')
    with pytest.raises(ValueError, match='no mode 5'):
        mr.modal_response_density(density, shapes, modes=[4])
    one = ShapeSet([10.0], [0.02], DOFS, np.ones((1, 4)), mass_unit='kg')
    assert mr.modal_response_density(density, one).num_records == 1
    empty = ShapeSet(np.zeros(0), np.zeros(0), DOFS, np.zeros((0, 4)),
                     mass_unit='kg')
    with pytest.raises(ValueError, match='no modes'):
        mr.modal_force_density(density, empty)
    motion = Psd(LINES, np.ones((1, LINES.size)), response_dof=['1X+'],
                 reference_dof=['1X+'],
                 ordinate_dim='acceleration**2/frequency')
    with pytest.raises(ValueError, match='no force density'):
        mr.modal_response_density(motion, shapes)
    banded = density.to_octave(3)
    with pytest.raises(ValueError, match='proportional bands'):
        mr.modal_response_density(banded, shapes)


def test_a_density_at_breakpoints_is_refused_and_on_lines_goes_through():
    """A specification at breakpoints converted to a CPSD kept its
    breakpoints, and the modal response was computed at those four
    frequencies alone, every mode between them missed (Brandon,
    2026-10-30). Refused, with the way to lines named."""
    shapes, s = _model(), _forces(correlated=False)
    at = [0, 400]                                  # 0 and 200 Hz
    breakpoints = Psd(LINES[at], _density(s).ordinate[:, at],
                      response_dof=_density(s).response_dof,
                      reference_dof=_density(s).reference_dof,
                      ordinate_dim=_density(s).ordinate_dim)
    breakpoints.interpolation = 'log_log'
    with pytest.raises(ValueError, match='breakpoints.*onto lines'):
        mr.modal_response_density(breakpoints, shapes)
    breakpoints.interpolation = 'bin'
    assert mr.modal_response_density(breakpoints, shapes).num_records == 16


def test_a_modal_force_density_does_not_expand_and_a_response_does():
    """Expand was offered on Transform's modal force CPSD and refused
    when pressed: many force distributions give one modal force."""
    shapes, density = _model(), _density(_forces())
    pp, _report = to_modal(density, shapes)
    assert not expands(pp, shapes)
    assert expands(mr.modal_response_density(density, shapes), shapes)


# ---- the verb and the bar ---------------------------------------------------

def _project():
    project = visualdynamics.Project()
    project['Shapes'] = _model()
    project['Forces'] = _density(_forces())
    return project


def test_the_verb_is_offered_on_the_pair_and_makes_an_object():
    project = _project()
    assert 'compute_modal_responses' in dict(
        project.selection_verbs('Forces', 'Shapes'))
    added = project.compute_modal_responses('Forces', 'Shapes')
    assert added == 'Forces Modal Responses'
    assert isinstance(project[added], Psd)
    assert project.journal[-1] == \
        "project.compute_modal_responses('Forces', 'Shapes')"
    verbs = dict(project.selection_verbs(added, 'Shapes'))
    assert 'expand' in verbs
    expanded = project.expand(added, 'Shapes')
    want = _synthesized(project['Shapes'], _forces(), 2)
    got = _matrix(project[expanded], DOFS)
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()


def test_expand_is_not_offered_on_a_modal_force_density():
    project = _project()
    modal = project.transform('Forces', 'Shapes')
    assert 'expand' not in dict(project.selection_verbs(modal, 'Shapes'))


def test_the_pair_offers_the_act_and_the_pane_previews(window, pump):
    window.add_object('Shapes', _model())
    window.add_object('Forces', _density(_forces()))
    pump()
    select_objects(window, pump, 'Forces', 'Shapes')
    assert 'Compute Modal Responses' in bar_acts(window.data_pane)
    window.modal_response_act()
    pump()
    panel = window.data_pane.modal_response_panel
    assert panel.isVisible()
    assert panel._derived['modes'][1].text() == '4 of 4'
    assert panel._derived['forces'][1].text() == '2'
    assert panel._derived['samples'][0].text() == 'Lines'
    assert panel._derived['samples'][1].text() == f'{LINES.size:,}'
    assert ('modal acceleration density of 4 modes from 2 force channels'
            in window.statusBar().currentMessage())
    panel.apply_button.click()
    pump()
    made = window.objects['Forces Modal Responses']
    assert isinstance(made, Psd) and made.num_records == 16
    assert window._item_for_object('Forces Modal Responses').isSelected()
