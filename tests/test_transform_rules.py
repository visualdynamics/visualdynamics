"""The transform's report and refusals, pinned by value where a mutation
pass found them read only for their presence (2026-10-11).

`tools/mutate.py` over transform.py against its two test files caught
137 of 213. The transforms themselves were pinned to round-off; what
survived was the arithmetic of what the report says about them — the
residual fraction, the counts of what was left out, the pairs a refusal
names — which the tests checked for being there rather than for being
right.

The set here is two modes over three DOFs, Φ = [[1, 0], [0, 1], [1, 1]],
whose one direction out of reach is (1, 1, -1).
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.transform import (
    TransformReport,
    carried_modes,
    to_modal,
    to_physical,
)

T = np.arange(256) / 256.0
DOFS = ['1X+', '2X+', '3X+']
PHI = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _shapes(count=2):
    matrix = PHI.T if count == 2 else np.eye(count, 3)
    return visualdynamics.ShapeSet(frequency=np.arange(1, count + 1) * 10.0,
                                   damping=[0.01] * count, coordinate=DOFS,
                                   shape_matrix=matrix)


def test_the_residual_is_the_share_of_each_dofs_rms_the_modes_miss():
    """u = Φ q + r with r along (1, 1, -1): the fit takes Φ q exactly
    and leaves r, so each DOF's residual is rms(r) / rms(u) — sqrt(0.2)
    where a mode is also moving, all of it where none is. A fourth
    channel at rest has nothing to miss."""
    s, c = np.sin(2 * np.pi * 4 * T), np.cos(2 * np.pi * 4 * T)
    u = np.outer(PHI[:, 0], s) + 0.5 * np.outer([1.0, 1.0, -1.0], c)
    shapes = visualdynamics.ShapeSet([10.0, 20.0], [0.01] * 2,
                                     DOFS + ['4X+'],
                                     np.column_stack([PHI.T, [0.0, 0.0]]))
    history = visualdynamics.TimeHistory(T, np.vstack([u, np.zeros(len(T))]),
                                         response_dof=DOFS + ['4X+'],
                                         ordinate_dim='acceleration')
    _modal, report = to_modal(history, shapes)
    assert report.residual == {'1X+': pytest.approx(np.sqrt(0.2)),
                               '2X+': pytest.approx(1.0),
                               '3X+': pytest.approx(np.sqrt(0.2)),
                               '4X+': 0.0}
    assert report.describe() == '4 DOFs shared, residual 100.0% at worst'


def test_a_report_says_one_dof_in_the_singular():
    report = TransformReport(shared={'acceleration': ['1X+']})
    assert report.describe() == '1 DOF shared'


def test_a_pick_names_records_that_exist():
    history = visualdynamics.TimeHistory(T, np.outer(PHI[:, 0], np.ones(len(T))),
                                         response_dof=DOFS,
                                         ordinate_dim='acceleration')
    to_modal(history, _shapes(), records=[0, 1, 2])
    with pytest.raises(ValueError, match='no record 3: the object has 3'):
        to_modal(history, _shapes(), records=[1, 3])
    with pytest.raises(ValueError, match='no record -1: the object has 3'):
        to_modal(history, _shapes(), records=[-1])


def _frf(pairs, dims='acceleration/force', lines=8):
    f = np.linspace(1.0, 50.0, lines)
    rows = np.ones((len(pairs), lines), dtype=complex)
    return visualdynamics.Frf(f, rows, response_dof=[u for u, _f in pairs],
                              reference_dof=[g for _u, g in pairs],
                              ordinate_dim=dims)


def test_a_response_missing_a_reference_is_refused_with_the_count():
    one = _frf([('1X+', '1X+'), ('2X+', '1X+'), ('3X+', '1X+'),
                ('1X+', '2X+'), ('2X+', '2X+')])
    with pytest.raises(ValueError, match=r'no FRF between 3X\+ and 2X\+: the responses'):
        to_modal(one, _shapes())
    three = _frf([('1X+', '1X+'), ('2X+', '1X+'), ('3X+', '1X+'),
                  ('1X+', '2X+')])
    with pytest.raises(ValueError,
                       match=r'no FRF between 2X\+ and 2X\+ and 1 more: the'):
        to_modal(three, _shapes())


def test_the_frf_residual_is_the_share_of_each_response_the_modes_miss():
    """Each reference's column a fit of its own; the residual is over
    every reference and line at once."""
    f = np.linspace(1.0, 50.0, 8)
    modal = np.ones((8, 2, 1), dtype=complex)
    outside = 0.5 * np.array([1.0, 1.0, -1.0])[:, None]
    matrix = PHI @ modal + outside                     # 8 × 3 × 1
    frf = visualdynamics.Frf(f, matrix[:, :, 0].T, response_dof=DOFS,
                             reference_dof=['9X+'] * 3,
                             ordinate_dim='acceleration/force')
    _modal, report = to_modal(frf, _shapes())
    # |u| = 1.5, 1.5, 1.5 (1+0.5, 1+0.5, 2-0.5); |r| = 0.5 at each
    assert report.residual == {dof: pytest.approx(1.0 / 3.0) for dof in DOFS}


@pytest.mark.parametrize('drives, said', [
    (['7X+', '8X+', '9X+'], '7X+, 8X+, 9X+ not in the shapes'),
    (['6X+', '7X+', '8X+', '9X+'], '6X+, 7X+, 8X+ and 1 more not in the shapes'),
])
def test_drives_outside_the_set_are_named_three_at_most(drives, said):
    frf = _frf([(u, g) for g in drives for u in DOFS])
    _modal, report = to_modal(frf, _shapes())
    assert report.notes == ['references kept physical: ' + said]


def test_what_is_left_out_is_counted_record_by_record():
    temperatures = _frf([(u, '1X+') for u in DOFS] + [('1X+', '2X+')],
                        dims=['acceleration/force'] * 3 + ['temperature/force'])
    _modal, report = to_modal(temperatures, _shapes(), records=[0, 1, 2, 3])
    assert report.skipped == {'temperature/force': 1}
    modal = _frf([('M1', '9X+'), ('M2', '9X+'), ('M1', '8X+'), ('M2', '8X+')],
                 dims=['acceleration/force'] * 2 + ['force/force'] * 2)
    _physical, report = to_physical(modal, _shapes())
    assert report.skipped == {'force/force': 2}


def test_a_matrix_missing_pairs_counts_them():
    f = np.linspace(1.0, 50.0, 8)

    def psd(pairs):
        return visualdynamics.Psd(f, np.ones((len(pairs), 8), dtype=complex),
                                  response_dof=[a for a, _b in pairs],
                                  reference_dof=[b for _a, b in pairs],
                                  ordinate_dim='acceleration**2/frequency')

    autos = [(d, d) for d in DOFS]
    with pytest.raises(ValueError, match=r'between 1X\+ and 2X\+ and 1 more pair: a'):
        to_modal(psd(autos + [('1X+', '3X+'), ('2X+', '3X+')]), _shapes())
    with pytest.raises(ValueError, match=r'between 1X\+ and 2X\+ and 3 more pairs: a'):
        to_modal(psd(autos + [('2X+', '3X+')]), _shapes())


def test_the_modes_carried_are_listed_in_mode_order_not_spelling():
    """M2 before M10, which a string sort would put first; empty when
    every mode went."""
    shapes = visualdynamics.ShapeSet(np.arange(1, 12) * 10.0, [0.01] * 11,
                                     ['1X+'], np.ones((11, 1)))
    report = TransformReport(shared={'acceleration': ['M10', 'M2', '1X+']})
    assert carried_modes(report, shapes) == ['M2', 'M10']
    every = TransformReport(shared={'acceleration': [f'M{k}' for k in range(1, 12)]})
    assert carried_modes(every, shapes) == []


def test_an_expansion_names_the_sets_modal_dofs_when_refused():
    physical = visualdynamics.TimeHistory(T, np.zeros((1, len(T))),
                                          response_dof=['5Z+'],
                                          ordinate_dim='acceleration')
    with pytest.raises(ValueError, match=r'of this set \(M1 … M2\)'):
        to_physical(physical, _shapes())


def _psd(pairs, values=None, dims='acceleration**2/frequency', lines=4):
    f = np.linspace(1.0, 50.0, lines)
    rows = (np.ones((len(pairs), lines), dtype=complex) if values is None
            else np.repeat(np.asarray(values, dtype=complex)[:, None], lines, axis=1))
    return visualdynamics.Psd(f, rows, response_dof=[a for a, _b in pairs],
                              reference_dof=[b for _a, b in pairs],
                              ordinate_dim=dims)


def test_the_density_residual_is_the_share_of_each_autos_power_missed():
    """S = Φ Φᵀ + 0.25 v vᵀ, v = (1, 1, -1) out of reach: the autos are
    1.25, 1.25, 2.25 and the modes explain all but 0.25 of each. A
    fourth DOF at rest has no power to miss."""
    v = np.array([1.0, 1.0, -1.0, 0.0])
    phi = np.vstack([PHI, [0.0, 0.0]])
    matrix = phi @ phi.T + 0.25 * np.outer(v, v)
    names = DOFS + ['4X+']
    pairs = [(a, b) for a in names for b in names]
    shapes = visualdynamics.ShapeSet([10.0, 20.0], [0.01] * 2, names, phi.T)
    _modal, report = to_modal(_psd(pairs, [matrix[names.index(a), names.index(b)]
                                           for a, b in pairs]), shapes)
    assert report.residual == {'1X+': pytest.approx(0.2), '2X+': pytest.approx(0.2),
                               '3X+': pytest.approx(1.0 / 9.0), '4X+': 0.0}
    assert report.notes == [], 'no cross-quantity terms to leave out'


def test_a_density_residual_moving_with_a_mode_is_still_read_per_channel():
    """One coherent motion, u = Φ₁ + 0.5 v with v = (1, 1, -1) out of
    reach, so S = u uᵀ and the residual is 0.5 v on every channel. Each
    channel's share of power is then the square of the rows path's rms
    share on the same motion: 1/9, 1 and 1. Subtracting the explained
    power instead keeps the cross terms and reads 5/9, 1 and 0."""
    u = PHI[:, 0] + 0.5 * np.array([1.0, 1.0, -1.0])
    matrix = np.outer(u, u)
    pairs = [(a, b) for a in DOFS for b in DOFS]
    _modal, report = to_modal(_psd(pairs, [matrix[DOFS.index(a), DOFS.index(b)]
                                           for a, b in pairs]), _shapes())
    assert report.residual == {'1X+': pytest.approx(1.0 / 9.0),
                               '2X+': pytest.approx(1.0),
                               '3X+': pytest.approx(1.0)}
    history = visualdynamics.TimeHistory(
        T, np.outer(u, np.sin(2 * np.pi * 4 * T)), response_dof=DOFS,
        ordinate_dim='acceleration')
    _rows, rows = to_modal(history, _shapes())
    assert {dof: share ** 2 for dof, share in rows.residual.items()} == \
        pytest.approx(report.residual)


@pytest.mark.parametrize('data, said', [
    (_psd([(a, b) for a in DOFS for b in DOFS], dims='temperature**2/frequency'),
     (r'nothing to transform: no motion or force density of this object is on '
     r'DOFs the shapes cover \(not transformed: 0 DOFs shared, not transformed: '
     r'9 temperature\)')),
    (_frf([(u, '1X+') for u in DOFS], dims='temperature/force'),
     (r'nothing to transform: no motion response of this FRF is on a DOF the '
     r'shapes cover \(not transformed:')),
])
def test_nothing_to_transform_says_what_was_left_out(data, said):
    with pytest.raises(ValueError, match=said):
        to_modal(data, _shapes())


def test_modal_forces_are_said_not_to_expand():
    forces = visualdynamics.TimeHistory(T, np.zeros((2, len(T))),
                                        response_dof=['M1', 'M2'],
                                        ordinate_dim='force')
    with pytest.raises(ValueError, match=r"modal DOFs \(modal forces do not expand"):
        to_physical(forces, _shapes())


def test_a_modal_frf_expands_the_modes_it_holds_and_needs_their_references():
    one = _frf([('M2', '9X+')])
    one.ordinate[0] = 2.0
    physical, report = to_physical(one, _shapes())
    assert np.allclose(physical.ordinate[:, 0], 2.0 * PHI[:, 1])
    assert report.shared == {'acceleration': ['M2']}
    holed = _frf([('M1', '9X+'), ('M2', '9X+'), ('M1', '8X+')])
    with pytest.raises(ValueError, match='no FRF between M2 and 8X[+]: the modes expand'):
        to_physical(holed, _shapes())


def test_a_modal_record_with_its_units_undefined_expands_with_them_undefined():
    frf = _frf([('M1', '9X+'), ('M2', '9X+')], dims='unknown')
    frf.dimension_hint = ['acceleration/force'] * 2
    physical, _report = to_physical(frf, _shapes())
    assert set(physical.ordinate_dim) == {'unknown'}
    assert set(physical.dimension_hint) == {'acceleration/force'}
    defined, _report = to_physical(_frf([('M1', '9X+'), ('M2', '9X+')]), _shapes())
    assert set(defined.ordinate_dim) == {'acceleration/force'}
    assert set(defined.dimension_hint) == {None}


def test_modal_force_densities_are_counted_out_record_by_record():
    modal = [(f'M{k}', f'M{l}') for k in (1, 2) for l in (1, 2)]
    psd = _psd(modal + modal, dims=['acceleration**2/frequency'] * 4
               + ['force**2/frequency'] * 4)
    _physical, report = to_physical(psd, _shapes())
    assert report.skipped == {'force**2/frequency': 4}
    undefined = _psd(modal, dims='unknown')
    undefined.dimension_hint = ['acceleration**2/frequency'] * 4
    physical, _report = to_physical(undefined, _shapes())
    assert set(physical.ordinate_dim) == {'unknown'}
