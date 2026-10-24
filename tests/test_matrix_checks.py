"""The two matrix checks of a multi-input test: the conditioning of an
FRF or CSD matrix, and the realizability of a cross-spectral matrix,
with its repair and the specification sheet's refusal.

Checked against matrices whose singular values and eigenvalues are
worked by hand: a diagonal plant, a rank-one plant, two modes whose
columns part by the square of their frequency ratio at DC, two drives
correlated by a known coherence, and three channels asked to be in
step, in step and opposed, whose coherence matrix has eigenvalues -1,
2 and 2 and whose nearest realizable neighbor has every coherence a
quarter.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.author import SpecificationDraft
from visualdynamics.core.data import Frf, Psd, Specification
from visualdynamics.core.matrix_checks import (
    CONDITION_LIMIT,
    check_spectral_matrix,
    condition_matrix,
    entries_refusal,
    frf_matrix,
    read_conditioning,
    read_realizability,
    repair_coherence,
)

F = np.array([10.0, 20.0, 30.0, 40.0])

#: three channels in step, in step, and opposed: every coherence one,
#: phases 0, 0 and 180 degrees. Eigenvalues -1, 2, 2 (trace 3,
#: determinant -4), the negative one along (1, -1, 1)
OPPOSED = np.array([[1, 1, -1], [1, 1, 1], [-1, 1, 1]], dtype=complex)


def _frf(matrix, frequency=None, dims=None, responses=('101Z+', '102Z+'),
         references=('1Z+', '2Z+')):
    """An FRF object holding `matrix` (lines x responses x references),
    response-major as compute_frfs lays it out."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim == 2:
        matrix = np.repeat(matrix[None], len(F), axis=0)
    frequency = F if frequency is None else frequency
    rows, resp, refs = [], [], []
    for i, r in enumerate(responses[:matrix.shape[1]]):
        for j, d in enumerate(references[:matrix.shape[2]]):
            rows.append(matrix[:, i, j])
            resp.append(r)
            refs.append(d)
    return Frf(frequency, np.asarray(rows), response_dof=resp,
               reference_dof=refs,
               ordinate_dim=dims or ['acceleration/force'] * len(rows))


def _csd(matrix, channels=('1Z+', '2Z+', '3Z+'), dims=None,
         cls=Psd, **kwargs):
    """A CSD object holding `matrix` (lines x n x n), every term."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim == 2:
        matrix = np.repeat(matrix[None], len(F), axis=0)
    n = matrix.shape[1]
    dims = dims or ['acceleration'] * n
    rows, resp, refs, record_dims = [], [], [], []
    for i in range(n):
        for j in range(n):
            rows.append(matrix[:, i, j])
            resp.append(channels[i])
            refs.append(channels[j])
            record_dims.append(f'{dims[i]}**2/frequency' if dims[i] == dims[j]
                               else f'{dims[i]}*{dims[j]}/frequency')
    return cls(F[:matrix.shape[0]], np.asarray(rows), response_dof=resp,
               reference_dof=refs, ordinate_dim=record_dims, **kwargs)


# ---- conditioning, by hand --------------------------------------------------

def test_a_diagonal_plant_has_its_entries_for_singular_values():
    reading = _frf(np.diag([3.0, 1.0])).conditioning()
    np.testing.assert_allclose(reading.singular, [[3.0] * 4, [1.0] * 4])
    np.testing.assert_allclose(reading.condition, 3.0)
    assert (reading.rank == 2).all() and not reading.ill.any()
    assert reading.limit == CONDITION_LIMIT
    # the limit is the reading's: under three, every line is flagged
    # and the effective rank drops to one
    tight = _frf(np.diag([3.0, 1.0])).conditioning(limit=2.0)
    assert tight.ill.all() and (tight.rank == 1).all()
    assert tight.bands() == [(10.0, 40.0)]
    assert 'over 2 at 4 of 4 lines, 10 to 40 Hz' in tight.describe()


def test_a_rank_one_plant_is_singular():
    """[[1, 2], [2, 4]] is one column twice over: singular values 5 and
    0, an infinite condition number, rank one."""
    reading = _frf([[1.0, 2.0], [2.0, 4.0]]).conditioning()
    np.testing.assert_allclose(reading.singular[0], 5.0)
    np.testing.assert_allclose(reading.singular[1], 0.0, atol=1e-12)
    assert reading.ill.all() and (reading.rank == 1).all()
    point = reading.at(25.0)
    assert point.frequency in (20.0, 30.0) and point.ill
    assert 'rank 1 of 2' in point.describe()


def test_two_modes_part_by_their_frequency_ratio_squared_at_dc():
    """Two modes, the first moving both drives together and the second
    against each other, at 100 and 400 Hz. At DC the plant is
    phi1 phi1^T / w1^2 + phi2 phi2^T / w2^2, whose singular values are
    2 / w1^2 and 2 / w2^2: a condition number of (400/100)^2 = 16, and
    at the first mode, where it alone answers, the columns are one."""
    f = np.array([0.0, 50.0, 100.0, 250.0])
    w = 2 * np.pi * f
    shapes = [(100.0, np.array([1.0, 1.0])), (400.0, np.array([1.0, -1.0]))]
    h = np.zeros((len(f), 2, 2), dtype=complex)
    for fn, phi in shapes:
        wn = 2 * np.pi * fn
        h += np.outer(phi, phi)[None] / (
            wn ** 2 - w[:, None, None] ** 2
            + 2j * 0.01 * wn * w[:, None, None])
    reading = _frf(h, frequency=f).conditioning()
    np.testing.assert_allclose(reading.condition[0], 16.0, rtol=1e-12)
    w1, w2 = 2 * np.pi * 100.0, 2 * np.pi * 400.0
    np.testing.assert_allclose(reading.singular[:, 0],
                               [2 / w1 ** 2, 2 / w2 ** 2], rtol=1e-12)
    assert reading.ill[2] and not reading.ill[0] and not reading.ill[3]


def test_a_tall_plant_and_a_wide_one():
    tall = condition_matrix(F[:1], np.array([[[3, 0], [0, 4], [0, 0]]]),
                            ['a', 'b', 'c'], ['d', 'e'])
    np.testing.assert_allclose(tall.singular[:, 0], [4.0, 3.0])
    np.testing.assert_allclose(tall.condition, 4.0 / 3.0)
    # a response nothing reaches is named, though a tall matrix keeps
    # its full rank without it
    assert tall.notes == (('c is reached by no reference at any line: a '
                           'dead channel gives the matrix a zero row'),)
    wide = condition_matrix(F[:1], np.array([[[3, 0, 0], [0, 4, 0]]]),
                            ['a', 'b'], ['c', 'd', 'e'])
    assert wide.size == 2
    assert any('3 references drive 2 responses' in n for n in wide.notes)


def test_correlated_drives_make_a_nearly_singular_gff():
    """Two drives of equal power with coherence rho squared: Gff is
    [[1, rho], [rho, 1]], eigenvalues 1 +- rho, condition number
    (1 + rho) / (1 - rho). At rho 0.98, 99, within 100; at 0.99, 199."""
    rho = np.array([0.98, 0.99, 0.5, 0.0])
    gff = np.ones((4, 2, 2), dtype=complex)
    gff[:, 0, 1] = gff[:, 1, 0] = rho
    reading = _csd(gff, channels=('1Z+', '2Z+'),
                   dims=['force', 'force']).conditioning()
    np.testing.assert_allclose(reading.condition,
                               (1 + rho) / (1 - rho), rtol=1e-12)
    assert reading.kind == 'CSD' and reading.rows == ('1Z+', '2Z+')
    assert list(reading.ill) == [False, True, False, False]


def test_conditioning_refuses_what_it_cannot_read():
    frf = _frf(np.diag([3.0, 1.0]))
    with pytest.raises(ValueError, match='no records are selected'):
        frf.conditioning(records=[])
    with pytest.raises(ValueError, match='compares references'):
        frf.conditioning(records=[0, 2])
    with pytest.raises(ValueError, match=r'incomplete: 102Z\+ has no FRF '
                                         r'to 2Z\+'):
        frf.conditioning(records=[0, 1, 2])
    mixed = _frf(np.diag([3.0, 1.0]), dims=['acceleration/force'] * 3
                 + ['acceleration/voltage'])
    with pytest.raises(ValueError, match='different units'):
        mixed.conditioning()
    with pytest.raises(ValueError, match='more than 1'):
        frf.conditioning(limit=1.0)
    with pytest.raises(TypeError):
        read_conditioning(visualdynamics.TimeHistory(
            np.arange(4.0), np.ones((1, 4)), response_dof=['1Z+']))
    spec = _csd(np.eye(2), channels=('1Z+', '2Z+'),
                dims=['acceleration', 'force'])
    with pytest.raises(ValueError, match='different quantities'):
        spec.conditioning()


def test_undeclared_units_are_read_and_said():
    reading = _frf(np.diag([3.0, 1.0]), dims=['unknown'] * 4).conditioning()
    np.testing.assert_allclose(reading.condition, 3.0)
    assert any('not declared' in n for n in reading.notes)


def test_a_nan_line_a_zero_line_and_a_dead_drive():
    h = np.repeat(np.diag([3.0, 1.0]).astype(complex)[None], 4, axis=0)
    h[1, 0, 0] = np.nan
    h[2] = 0.0
    reading = condition_matrix(F, h, ['a', 'b'], ['c', 'd'])
    assert list(reading.judged) == [True, False, False, True]
    assert np.isnan(reading.condition[1]) and np.isnan(reading.condition[2])
    assert list(reading.rank) == [2, 0, 0, 2]
    assert ('1 line holds a value that is not a finite number, in a '
            'against c, and is not judged') in reading.notes
    assert any('1 lines are zero throughout' in n for n in reading.notes)
    assert 'not judged' in reading.at(20.0).describe()
    assert not reading.at(20.0).ill, 'a line not judged is not flagged'
    dead = h.copy()
    dead[:, :, 1] = 0.0
    dead[1, 0, 0] = 3.0
    dead[2, 0, 0] = 3.0
    reading = condition_matrix(F, dead, ['a', 'b'], ['c', 'd'])
    assert any('d reaches no row' in n for n in reading.notes)
    assert np.isinf(reading.condition).all() and reading.ill.all()
    empty = condition_matrix(F, np.full((4, 2, 2), np.nan), ['a', 'b'],
                             ['c', 'd'])
    assert 'no line could be judged' in empty.describe()


def test_the_cmif_assembles_its_matrix_with_the_reading():
    """The CMIF's singular values are the SVD of the matrix the records
    make, read in the display units; one helper assembles both."""
    from visualdynamics.plot import cmif_curves
    from visualdynamics.units import SI

    rng = np.random.default_rng(7)
    h = rng.standard_normal((4, 3, 2)) + 1j * rng.standard_normal((4, 3, 2))
    frf = _frf(h, responses=('101Z+', '102Z+', '103Z+'))
    singular, _f = cmif_curves(frf, unit_system=SI)
    np.testing.assert_allclose(singular,
                               np.linalg.svd(h, compute_uv=False).T)
    np.testing.assert_allclose(frf.conditioning().singular, singular)
    rows, columns, matrix, filled = frf_matrix(frf, [0, 1, 3])
    assert rows == [('101Z+', 'acceleration'), ('102Z+', 'acceleration')]
    assert columns == [('1Z+', 'force'), ('2Z+', 'force')]
    assert filled.tolist() == [[True, True], [False, True]]
    assert matrix[0, 1, 0] == 0.0


# ---- realizability, by hand -------------------------------------------------

def test_three_channels_in_step_in_step_and_opposed():
    reading = check_spectral_matrix(F[:1], OPPOSED[None], ['a', 'b', 'c'])
    np.testing.assert_allclose(reading.eigenvalues[:, 0], [-1.0, 2.0, 2.0],
                               atol=1e-12)
    assert reading.failing.all() and not reading.realizable
    # the nearest unit-diagonal matrix that holds has every off-diagonal
    # a half: the form 1 + a, 1 + a, 1 - 2a is realizable to a = 1/2
    np.testing.assert_allclose(reading.repaired[0].real,
                               [[1, 0.5, -0.5], [0.5, 1, 0.5],
                                [-0.5, 0.5, 1]], atol=1e-9)
    coherence, phase = reading.changes()
    assert coherence == pytest.approx(0.75, abs=1e-9)
    assert phase == pytest.approx(0.0, abs=1e-6)
    before, after, _p, _q = reading.pair(0, 2)
    np.testing.assert_allclose([before[0], after[0]], [1.0, 0.25],
                               atol=1e-9)


def test_without_the_diagonal_the_negative_eigenvalue_is_set_to_zero():
    """C + v v^H, v = (1, -1, 1) / sqrt(3) the eigenvector of -1: the
    diagonal rises to 4/3 and the off-diagonals fall to 2/3."""
    reading = check_spectral_matrix(F[:1], OPPOSED[None], ['a', 'b', 'c'],
                                    keep_diagonal=False)
    third = 1.0 / 3.0
    np.testing.assert_allclose(reading.repaired[0].real,
                               [[4 * third, 2 * third, -2 * third],
                                [2 * third, 4 * third, 2 * third],
                                [-2 * third, 2 * third, 4 * third]],
                               atol=1e-12)
    assert any('moves the autospectra' in n for n in reading.notes)


def test_the_check_is_free_of_units_and_levels():
    """A positive diagonal scaling does not change the signs of the
    eigenvalues, so the verdict and the coherence-matrix eigenvalues are
    the same whatever the levels and units of the channels."""
    scale = np.array([1e-6, 1.0, 1e3])
    s = OPPOSED * np.sqrt(np.outer(scale, scale))
    spec = _csd(s, dims=['acceleration', 'force', 'angular_acceleration'])
    reading = spec.realizability()
    np.testing.assert_allclose(reading.eigenvalues[:, 0], [-1, 2, 2],
                               atol=1e-9)
    assert reading.channels == ('1Z+', '2Z+', '3Z+')
    np.testing.assert_allclose(reading.matrix()[0], s, rtol=1e-12)


def test_a_realizable_matrix_is_left_bit_for_bit():
    spec = SpecificationDraft.at_dofs(
        ['1Z+', '2Z+', '3Z+'], 'acceleration').with_all_pairs(0.9).make()
    reading = spec.realizability()
    assert reading.realizable and not reading.failing.any()
    assert reading.worst() is None and reading.changes() == (0.0, 0.0)
    assert np.array_equal(reading.repaired, reading.coherence)
    assert (reading.rank == 3).all()
    assert 'realizable at all 2 lines' in reading.describe()
    coherent = SpecificationDraft.at_dofs(
        ['1Z+', '2Z+', '3Z+'], 'acceleration').with_all_pairs(1.0).make()
    assert (coherent.realizability().rank == 1).all(), \
        'fully coherent channels are one motion'
    repaired = spec.made_realizable()
    assert np.array_equal(repaired.ordinate, spec.ordinate)


def test_a_cross_term_that_is_not_its_mirrors_conjugate():
    s = np.repeat(np.eye(2, dtype=complex)[None], 4, axis=0)
    s[:, 0, 1] = 0.5
    s[:, 1, 0] = 0.3
    reading = _csd(s, channels=('1Z+', '2Z+')).realizability()
    np.testing.assert_allclose(reading.asymmetry, 0.1)
    assert reading.failing.all()
    assert any('not Hermitian' in n for n in reading.notes)
    np.testing.assert_allclose(reading.repaired[:, 0, 1], 0.4)


def test_a_negative_autospectrum_and_a_dead_channel_with_a_cross_term():
    s = np.repeat(np.eye(2, dtype=complex)[None], 4, axis=0)
    s[0, 1, 1] = -0.5
    # line 1: channel 2 has no power, and yet a cross term with channel 1
    s[1, 1, 1] = 0.0
    s[1, 0, 1] = s[1, 1, 0] = 0.3
    reading = _csd(s, channels=('1Z+', '2Z+')).realizability()
    assert list(reading.failing) == [True, True, False, False]
    assert any('below zero' in n for n in reading.notes)
    # repaired: the dead channel's cross term goes, its zero stays
    np.testing.assert_allclose(reading.matrix(True)[1],
                               [[1.0, 0.0], [0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(reading.matrix(True)[0, 1, 1], 0.0)


def test_nan_lines_are_not_judged_and_nothing_judged_is_no_pass():
    s = np.repeat(OPPOSED[None], 4, axis=0)
    s[2, 0, 1] = np.nan
    reading = check_spectral_matrix(F, s, ['a', 'b', 'c'])
    assert list(reading.judged) == [True, True, False, True]
    assert not reading.failing[2]
    assert any('not a finite number' in n for n in reading.notes)
    assert reading.bands() == [(10.0, 20.0), (40.0, 40.0)]
    nothing = check_spectral_matrix(F, np.full((4, 3, 3), np.nan),
                                    ['a', 'b', 'c'])
    assert not nothing.realizable
    assert 'no line could be judged' in nothing.describe()


def test_an_incomplete_matrix_is_refused_and_a_mirror_completes_one():
    spec = SpecificationDraft.at_dofs(['1Z+', '2Z+', '3Z+'], 'acceleration')
    partial = spec._copy(pairs={(0, 1): (0.5, 0.0), (1, 2): (0.5, 0.0),
                                (0, 2): None}).make()
    with pytest.raises(ValueError, match=r'1Z\+ and 3Z\+ have no cross term'):
        partial.realizability()
    autos = Psd(F, np.ones((2, 4)), response_dof=['1Z+', '2Z+'],
                ordinate_dim=['acceleration**2/frequency'] * 2)
    with pytest.raises(ValueError, match='autospectrum set'):
        autos.realizability()
    with pytest.raises(ValueError, match='one channel'):
        _csd(np.eye(2), channels=('1Z+', '2Z+')).realizability(records=[0])
    with pytest.raises(ValueError, match='no records are selected'):
        partial.realizability(records=[])
    full = _csd(OPPOSED)
    upper = [i for i in range(full.num_records)
             if full.response_dof[i] <= full.reference_dof[i]]
    reading = read_realizability(full, upper)
    assert any('taken as the conjugate' in n for n in reading.notes)
    np.testing.assert_allclose(reading.eigenvalues[:, 0], [-1, 2, 2],
                               atol=1e-12)
    with pytest.raises(TypeError):
        read_realizability(_frf(np.eye(2)))


def test_a_phase_that_wanders_fails_only_where_it_cannot_hold():
    """Coherence 0.81 between every pair, in step between 1-2 and 2-3,
    and 1-3's phase swept from 0 to 180 degrees. Each line's verdict is
    the sign of the smallest eigenvalue of that line's matrix, computed
    directly; the repair is confined to the failing lines."""
    phases = np.linspace(0.0, 180.0, 4)
    a = 0.9
    s = np.empty((4, 3, 3), dtype=complex)
    for k, phi in enumerate(phases):
        e = a * np.exp(1j * np.radians(phi))
        s[k] = [[1, a, e], [a, 1, a], [np.conj(e), a, 1]]
    reading = check_spectral_matrix(F, s * 2.5, ['a', 'b', 'c'])
    direct = np.linalg.eigvalsh(s)[:, 0]
    assert list(reading.failing) == list(direct < -1e-6 * np.linalg.eigvalsh(
        s)[:, -1])
    assert list(reading.failing) == [False, True, True, True]
    assert np.array_equal(reading.repaired[:1], reading.coherence[:1])
    assert reading.bands() == [(20.0, 40.0)]


def test_made_realizable_moves_only_the_cross_terms_at_failing_lines():
    s = np.repeat(OPPOSED[None], 4, axis=0) * 0.04
    s[0] = np.eye(3) * 0.04
    limits = np.full((9, 4), np.nan)
    autos = [0, 4, 8]
    limits[autos] = 0.08
    spec = _csd(s, cls=Specification, abort_upper=limits)
    repaired = spec.made_realizable()
    assert isinstance(repaired, Specification)
    assert repaired is not spec and repaired.ordinate is not spec.ordinate
    np.testing.assert_array_equal(np.real(repaired.ordinate[autos]),
                                  np.real(spec.ordinate[autos]))
    assert np.array_equal(repaired.ordinate[:, 0], spec.ordinate[:, 0])
    assert repaired.realizability().realizable
    np.testing.assert_allclose(repaired.ordinate[1, 1:], 0.02)
    np.testing.assert_array_equal(repaired.limits['abort_upper'],
                                  spec.limits['abort_upper'])
    # without the diagonal the autospectra rise by 4/3, and the limit
    # follows each one, so the band stays where it was in decibels
    loose = spec.made_realizable(keep_diagonal=False)
    np.testing.assert_allclose(np.real(loose.ordinate[0, 1:]), 0.04 * 4 / 3)
    np.testing.assert_allclose(loose.limits['abort_upper'][0, 1:],
                               0.08 * 4 / 3)
    np.testing.assert_array_equal(spec.limits['abort_upper'], limits)


def test_the_repair_converges_on_a_larger_matrix():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    c = (a + a.conj().T) / 2
    d = np.sqrt(np.abs(np.diag(c)).real) + 1.0
    c = c / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    assert np.linalg.eigvalsh(c)[0] < -0.1
    fixed = repair_coherence(c[None])[0]
    np.testing.assert_allclose(np.diag(fixed).real, 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(fixed)[0] > -1e-9
    # and nearer than the clip-and-rescale a quick repair would give
    clipped = repair_coherence(c[None], keep_diagonal=False)[0]
    e = 1 / np.sqrt(np.diag(clipped).real)
    rescaled = clipped * np.outer(e, e)
    assert np.linalg.norm(fixed - c) <= np.linalg.norm(rescaled - c) + 1e-9


def test_a_tolerance_out_of_range_is_refused():
    with pytest.raises(ValueError, match='between 0 and 1'):
        check_spectral_matrix(F[:1], OPPOSED[None], 'abc', tolerance=0.0)


# ---- the specification sheet ------------------------------------------------

def test_the_sheet_refuses_entries_that_cannot_hold_and_says_the_fix():
    draft = SpecificationDraft.at_dofs(['1Z+', '2Z+', '3Z+'], 'acceleration')
    with pytest.raises(ValueError) as refused:
        draft._copy(pairs={(0, 1): (1.0, 0.0), (1, 2): (1.0, 0.0),
                           (0, 2): (1.0, 180.0)})
    words = str(refused.value)
    assert 'the cross terms of 1Z+, 2Z+ and 3Z+ cannot hold together' in words
    assert 'eigenvalue of -1 against a largest of 2' in words
    assert 'every line from 20 to 2000 Hz' in words
    assert '1Z+–3Z+ coherence 0.25 at 180°' in words
    assert '1Z+–2Z+ coherence 0.25 at 0°' in words
    with pytest.raises(ValueError, match='cannot hold together'):
        draft.with_all_pairs(1.0, 90.0)
    two = SpecificationDraft.at_dofs(['1Z+', '2Z+'], 'acceleration')
    assert two.with_all_pairs(1.0, 90.0).pairs[(0, 1)] == (1.0, 90.0)


def test_only_the_groups_whose_pairs_are_all_entered_are_judged():
    channels = ['1Z+', '2Z+', '3Z+', '4Z+']
    bad = {(0, 1): (1.0, 0.0), (1, 2): (1.0, 0.0), (0, 2): (1.0, 180.0)}
    assert entries_refusal(channels, bad, [20.0, 2000.0]) is not None
    # the bad triangle with one pair unentered is not judged: what the
    # missing pair would be is not said
    open_triangle = dict(bad)
    open_triangle[(0, 2)] = None
    open_triangle.update({(0, 3): (0.5, 0.0), (1, 3): (0.5, 0.0)})
    assert entries_refusal(channels, open_triangle, [20.0, 2000.0]) is None
    draft = SpecificationDraft.at_dofs(channels, 'acceleration')
    assert draft._copy(pairs=open_triangle).unset_pairs() == [(0, 2), (2, 3)]


def test_a_sheet_opened_on_a_specification_that_cannot_hold_is_refused():
    spec = _csd(np.repeat(OPPOSED[None], 4, axis=0) * 0.01,
                cls=Specification)
    with pytest.raises(ValueError, match='cannot hold together'):
        SpecificationDraft.from_specification(spec)
    repaired = spec.made_realizable()
    opened = SpecificationDraft.from_specification(repaired)
    assert opened.pairs[(0, 2)] == pytest.approx((0.25, 180.0))


# ---- the project verb -------------------------------------------------------

def test_make_realizable_adds_the_repair_and_refreshes_to_it():
    project = visualdynamics.Project('t')
    project.add('Spec', _csd(np.repeat(OPPOSED[None], 4, axis=0) * 0.01,
                             cls=Specification))
    name = project.make_realizable('Spec')
    assert name == 'Spec Realizable'
    assert project[name].realizability().realizable
    assert project.provenance[name]['verb'] == 'make_realizable'
    assert project.journal[-1] == "project.make_realizable('Spec')"
    assert 'make_realizable' in dict(project.verbs('Spec'))
    before = project[name].ordinate.copy()
    project.refresh(name)
    np.testing.assert_array_equal(project[name].ordinate, before)
    with pytest.raises(ValueError, match='nothing to repair'):
        project.make_realizable(name)
    project.add('Autos', Psd(F, np.ones((2, 4)), response_dof=['1Z+', '2Z+'],
                             ordinate_dim=['acceleration**2/frequency'] * 2))
    assert 'make_realizable' not in dict(project.verbs('Autos'))


# ---- the edges the first mutation pass found unwatched ----------------------

def test_a_condition_number_at_the_limit_is_within_it():
    """kappa equal to the limit is not over it, and the smallest
    singular value then counts toward the effective rank."""
    reading = _frf(np.diag([100.0, 1.0])).conditioning()
    assert not reading.ill.any() and (reading.rank == 2).all()
    assert not reading.at(10.0).ill
    assert reading.notes == (), 'a square plant needs no note'
    small = _frf(np.diag([3e-3, 1e-3])).conditioning()
    assert small.judged.all(), 'a small plant is not a zero one'
    assert small.notes == ()
    point = small.at(31.0)
    assert point.frequency == 30.0
    assert point.describe('g/lbf', 2.0) == (
        '30 Hz: condition number 3, within 100; rank 2 of 2; singular '
        'values 0.006, 0.002 g/lbf')


def test_a_csd_reading_is_titled_as_one_and_a_bad_matrix_is_refused():
    gff = np.repeat(np.eye(2, dtype=complex)[None], 4, axis=0)
    reading = _csd(gff, channels=('1Z+', '2Z+'),
                   dims=['force', 'force']).conditioning()
    assert reading.title == 'CSD of 2 channels'
    with pytest.raises(ValueError, match='one rows-by-columns matrix'):
        condition_matrix(F, np.eye(2), ['a', 'b'], ['c', 'd'])
    with pytest.raises(ValueError, match='empty'):
        condition_matrix(F, np.zeros((4, 0, 2)), [], ['c', 'd'])
    with pytest.raises(ValueError, match='one square matrix per line'):
        check_spectral_matrix(F, np.zeros((4, 2, 3)), ['a', 'b'])
    with pytest.raises(ValueError, match='empty'):
        check_spectral_matrix(F, np.zeros((4, 0, 0)), [])
    with pytest.raises(ValueError, match='between 0 and 1'):
        check_spectral_matrix(F[:1], OPPOSED[None], 'abc', tolerance=1.0)


def test_the_messages_count_what_they_name():
    spec = _csd(np.eye(3) * 0.01)
    lone = [i for i in range(spec.num_records)
            if spec.response_dof[i] == spec.reference_dof[i]]
    lone.append(1)          # one cross term: two pairs still missing
    with pytest.raises(ValueError, match=r'\(1 more missing\)'):
        spec.realizability(records=lone)
    full = _csd(OPPOSED)
    upper = [i for i in range(full.num_records)
             if full.response_dof[i] <= full.reference_dof[i]]
    reading = read_realizability(full, upper)
    assert reading.notes[0] == ('3 terms held only by their mirrors taken '
                                'as the conjugate')
    one = [i for i in range(full.num_records) if i != 3]
    assert read_realizability(full, one).notes[0] == (
        '1 term held only by its mirror taken as the conjugate')
    h = np.repeat(np.diag([3.0, 1.0])[None], 9, axis=0).astype(complex)
    h[1::2] = [[1.0, 2.0], [2.0, 4.0]]
    flags = condition_matrix(np.arange(9.0) + 1.0, h, ['a', 'b'], ['c', 'd'])
    assert flags.describe().endswith('2 Hz, 4 Hz, 6 Hz and 1 more band')


def test_a_point_says_where_it_stands():
    reading = check_spectral_matrix(F, np.repeat(OPPOSED[None], 4, axis=0),
                                    ['a', 'b', 'c'])
    point = reading.at(20.0)
    assert not point.realizable and point.rank == 2
    assert point.describe() == ('20 Hz: not realizable; smallest eigenvalue '
                                '-1 against largest 2, rank 2 of 3')
    nan = np.full((4, 3, 3), np.nan, dtype=complex)
    assert check_spectral_matrix(F, nan, 'abc').at(20.0).describe() == \
        '20 Hz: not judged'
    assert check_spectral_matrix(F, nan, 'abc').at(20.0).realizable is False
    assert 'the repair moves' in reading.describe()


def test_two_channels_off_hermitian_name_their_pair_and_its_change():
    s = np.repeat(np.eye(2, dtype=complex)[None], 4, axis=0)
    s[:, 0, 1] = 0.5j
    s[:, 1, 0] = 0.3j
    reading = _csd(s, channels=('1Z+', '2Z+')).realizability()
    assert reading.worst() == (0, 1, 0)
    coherence, phase = reading.changes()
    # the mean of 0.5j and the conjugate of 0.3j is 0.1j: coherence
    # 0.25 to 0.01, the phase from 90 to 90
    assert coherence == pytest.approx(0.24)
    assert phase == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(check_spectral_matrix(
        F, np.where(np.arange(4)[:, None, None] == 2, np.nan, s),
        ['a', 'b']).coherence[2]).all()


def test_a_phase_moved_by_the_repair_is_reported_and_kept_complex():
    phases = np.linspace(0.0, 180.0, 4)
    a = 0.6
    s = np.empty((4, 3, 3), dtype=complex)
    for k, phi in enumerate(phases):
        e = a * np.exp(1j * np.radians(phi))
        s[k] = [[1, a, e], [a, 1, a], [np.conj(e), a, 1]]
    spec = _csd(s * 0.01)
    reading = spec.realizability()
    assert list(reading.failing) == [False, False, True, True]
    _coherence, phase = reading.changes()
    assert 0.5 < phase < 10.0
    repaired = spec.made_realizable()
    k = [i for i in range(spec.num_records)
         if (spec.response_dof[i], spec.reference_dof[i]) == ('1Z+', '3Z+')]
    np.testing.assert_allclose(repaired.ordinate[k[0]],
                               reading.matrix(True)[:, 0, 2])
    assert np.abs(np.imag(repaired.ordinate[k[0], 2])) > 1e-4
    real = _csd(np.repeat(OPPOSED[None], 4, axis=0))
    assert not np.iscomplexobj(real.ordinate)
    assert not np.iscomplexobj(real.made_realizable().ordinate)


def test_a_silent_channel_with_no_cross_term_is_realizable():
    s = np.repeat(np.eye(2, dtype=complex)[None], 4, axis=0) * 4.0
    s[0, 1, 1] = 0.0
    reading = _csd(s, channels=('1Z+', '2Z+')).realizability()
    assert reading.realizable and not reading.negative.any()
    assert list(reading.rank) == [1, 2, 2, 2]
    quiet = check_spectral_matrix(F, s, ['a', 'b'], keep_diagonal=False)
    assert quiet.notes == ()
    # line 1: a channel at 0.25 with a silent partner and a cross term
    # of 0.06. The silent one is scaled by the root of the line's
    # largest autospectrum, 0.5, so the term reads 0.06 / 0.25 = 0.24
    # and the line fails, as no signals give a silent channel a cross
    # term
    s[1, 0, 0] = 0.25
    s[1, 1, 1] = 0.0
    s[1, 0, 1] = s[1, 1, 0] = 0.06
    reading = check_spectral_matrix(F, s, ['a', 'b'])
    np.testing.assert_allclose(reading.scale[1], [0.5, 0.5])
    np.testing.assert_allclose(reading.coherence[1, 0, 1], 0.24)
    assert list(reading.failing) == [False, True, False, False]


def test_the_refusal_names_the_channels_in_the_conflict_alone():
    channels = ['1Z+', '2Z+', '3Z+', '4Z+']
    pairs = {(0, 1): (1.0, 0.0), (1, 2): (1.0, 0.0), (0, 2): (1.0, 180.0),
             (0, 3): (0.0, 0.0), (1, 3): (0.0, 0.0), (2, 3): (0.0, 0.0)}
    words = entries_refusal(channels, pairs, [20.0, 2000.0])
    assert words.startswith('the cross terms of 1Z+, 2Z+ and 3Z+ cannot')
    # the three it moves, largest first, and not the three it leaves
    shown = words.split('nearest entries that hold are ')[1]
    assert shown.count(' coherence 0.25 at ') == 3
    assert shown.endswith(' and 3 more')
    assert '4Z+' not in shown
    assert entries_refusal(channels, pairs, [20.0, 2000.0],
                           tolerance=0.9) is None


def test_the_quantity_picks_its_block():
    from visualdynamics.core.matrix_checks import spectral_matrix

    spec = _csd(np.eye(3), dims=['acceleration', 'acceleration', 'force'])
    channels, _matrix, filled = spectral_matrix(spec,
                                                quantity='acceleration')
    assert channels == [('1Z+', 'acceleration'), ('2Z+', 'acceleration')]
    assert filled.all()


# ---- dead, not-a-number and empty inputs ------------------------------------

def _history(signals, fs=1024.0, start=0.0, dofs=None, dims=None):
    """A time history of `signals` (channels x samples) at `fs`, starting
    at `start` seconds, with Hann frames of 256 samples overlapped by half
    filled in: the averaging an analyst gets by default."""
    from visualdynamics.core.averaging import Averaging

    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    n = signals.shape[1]
    dofs = dofs or [f'{k + 1}Z+' for k in range(len(signals))]
    history = visualdynamics.TimeHistory(
        start + np.arange(n) / fs, signals, response_dof=dofs,
        ordinate_dim=dims or ['acceleration'] * len(signals))
    history.averaging = Averaging(frame_length=256, overlap=0.5).filled(n, fs)
    assert history.averaging.window == 'hann'
    return history


def test_a_dead_channel_is_named_by_both_checks():
    rng = np.random.default_rng(11)
    signals = rng.standard_normal((3, 8192))
    signals[1] = 0.0
    csd = _history(signals).compute_cpsds()
    realizability = csd.realizability()
    assert realizability.realizable
    assert ('2Z+ has no power and no cross term at any line judged: a dead '
            'channel is realizable, and says nothing about how the others '
            'move with it') in realizability.notes
    conditioning = csd.conditioning()
    assert conditioning.ill[conditioning.judged].all()
    assert any(n.startswith('2Z+ reaches no row') for n in conditioning.notes)
    plant = np.repeat(np.diag([3.0, 1.0]).astype(complex)[None], 4, axis=0)
    plant[:, 1, :] = 0.0
    reading = condition_matrix(F, plant, ['a', 'b'], ['c', 'd'])
    assert any(n.startswith('b is reached by no reference') for n in
               reading.notes)


def test_a_record_with_one_nan_sample_is_named_and_judges_nothing():
    rng = np.random.default_rng(12)
    signals = rng.standard_normal((2, 4096))
    signals[1, 100] = np.nan
    csd = _history(signals).compute_cpsds()
    realizability = csd.realizability()
    assert not realizability.realizable, 'nothing judged is no pass'
    assert realizability.describe() == '2 channels: no line could be judged'
    assert any(n.endswith('a value that is not a finite number, in 1Z+ '
                          'against 2Z+ and 2 more terms, and are not judged')
               for n in realizability.notes)
    conditioning = csd.conditioning()
    assert 'no line could be judged' in conditioning.describe()
    with pytest.raises(ValueError, match='no line could be judged.*1Z\\+ '
                                         'against 2Z\\+'):
        project = visualdynamics.Project()
        project.add('Measured', csd)
        project.make_realizable('Measured')


def test_a_matrix_of_zeros_is_judged_by_nothing():
    zero = check_spectral_matrix(F, np.zeros((4, 2, 2)), ['a', 'b'])
    assert not zero.realizable and not zero.judged.any()
    assert zero.describe() == '2 channels: no line could be judged'
    assert zero.notes == ('4 lines are zero throughout and are not judged',)
    one = np.repeat(np.eye(2, dtype=complex)[None], 4, axis=0)
    one[2] = 0.0
    reading = check_spectral_matrix(F, one, ['a', 'b'])
    assert list(reading.judged) == [True, True, False, True]
    assert reading.realizable
    assert reading.notes == ('1 line is zero throughout and is not judged',)


def test_an_empty_selection_is_refused_by_both_checks():
    csd = _csd(np.eye(3))
    with pytest.raises(ValueError, match='no records are selected'):
        csd.realizability(records=[])
    with pytest.raises(ValueError, match='no records are selected'):
        csd.conditioning(records=[])
    with pytest.raises(ValueError, match='no records are selected'):
        csd.made_realizable(records=[])


def test_terms_declared_and_undeclared_are_not_one_matrix():
    dims = (['acceleration**2/frequency'] * 3 + ['unknown']
            + ['acceleration**2/frequency'] * 5)
    csd = _csd(np.eye(3))
    mixed = Psd(csd.abscissa, csd.ordinate, response_dof=csd.response_dof,
                reference_dof=csd.reference_dof, ordinate_dim=dims)
    with pytest.raises(ValueError, match=r'2Z\+ against 1Z\+ has no declared '
                                         'units and other records do'):
        mixed.realizability()
    with pytest.raises(ValueError, match='different units'):
        mixed.conditioning()
    none = Psd(csd.abscissa, csd.ordinate, response_dof=csd.response_dof,
               reference_dof=csd.reference_dof, ordinate_dim=['unknown'] * 9)
    assert none.realizability().realizable


# ---- measured matrices: the edges, the default window, white noise ----------

def test_a_measured_csd_is_realizable_at_every_line_and_at_the_edges():
    """A cross-spectral matrix computed from records under the default
    Hann window is realizable at every line, whatever the records hold:
    a tone on the first line and one at the edge of the span, a nonzero
    start time, noise on every channel, two tones that cross, and fewer
    frames than channels (a matrix of lower rank, still realizable)."""
    fs, n = 1024.0, 16384
    t = np.arange(n) / fs
    rng = np.random.default_rng(13)
    df = fs / 256
    chirp_up = np.sin(2 * np.pi * (50.0 * t + 10.0 * t ** 2))
    chirp_down = np.sin(2 * np.pi * (300.0 * t - 10.0 * t ** 2))
    signals = np.vstack([
        np.sin(2 * np.pi * df * t) + 0.1 * rng.standard_normal(n),
        np.sin(2 * np.pi * (fs / 2 - df) * t) + 0.1 * rng.standard_normal(n),
        chirp_up + chirp_down + 0.1 * rng.standard_normal(n),
        0.5 * chirp_up + rng.standard_normal(n)])
    for start in (0.0, 12.5):
        reading = _history(signals, fs, start).compute_cpsds().realizability()
        assert reading.judged.all() and reading.realizable, reading.describe()
    short = _history(signals[:, :384], fs)
    assert short.averaging.frames == 2
    reading = short.compute_cpsds().realizability()
    assert reading.realizable and (reading.rank[1:] <= 2).all()


def test_white_noise_trips_neither_check_by_chance():
    """The realizability tolerance and the condition-number limit,
    checked against what chance gives: independent white-noise drives
    averaged under the default Hann window, twenty records of 64 frames.
    A measured matrix fails realizability only by arithmetic, and a pair
    of independent drives reaches a condition number of 100 only with a
    coherence estimate near 0.96, which 64 frames make vanishingly
    rare."""
    fs = 1024.0
    realizable_trips = condition_trips = lines = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        csd = _history(rng.standard_normal((2, 256 * 33)), fs,
                       dims=['force', 'force']).compute_cpsds()
        realizable_trips += int(csd.realizability().failing.sum())
        conditioning = csd.conditioning()
        judged = conditioning.judged
        condition_trips += int(conditioning.ill[judged].sum())
        lines += int(judged.sum())
    assert lines == 20 * 129
    assert realizable_trips == 0
    assert condition_trips == 0


def test_progress_is_told_and_a_raise_in_it_stops_the_repair():
    told = []
    s = np.repeat(OPPOSED[None], 4, axis=0)
    reading = _csd(s).realizability(progress=lambda d, t: told.append((d, t)))
    assert not reading.realizable
    assert told[-1][0] == told[-1][1] > 0, 'the bar ends full'

    class Stop(Exception):
        pass

    def stop(_done, _total):
        raise Stop

    with pytest.raises(Stop):
        _csd(s).made_realizable(progress=stop)
    from visualdynamics.core.progress import BUSY

    told.clear()
    _frf(np.diag([3.0, 1.0])).conditioning(
        progress=lambda d, t: told.append((d, t)))
    assert told == [(0, BUSY)]


# ---- what a mutation run found untested ------------------------------------

def test_the_counts_in_the_words_are_exact():
    from visualdynamics.core.matrix_checks import _span_text

    assert _span_text([(10.0, 20.0), (40.0, 40.0)]) == '10 to 20 Hz and 40 Hz'
    three = [(10.0, 20.0), (40.0, 40.0), (60.0, 70.0)]
    assert _span_text(three) == '10 to 20 Hz, 40 Hz and 60 to 70 Hz'
    spec = SpecificationDraft.at_dofs(['1Z+', '2Z+', '3Z+'], 'acceleration')
    one_missing = spec._copy(pairs={(0, 1): (0.5, 0.0), (1, 2): (0.5, 0.0),
                                    (0, 2): None}).make()
    with pytest.raises(ValueError) as refused:
        one_missing.realizability()
    assert str(refused.value).startswith(
        'the matrix is incomplete: 1Z+ and 3Z+ have no cross term. It is')
    three_units = _frf(np.diag([3.0, 1.0]),
                       dims=['acceleration/force', 'velocity/force',
                             'acceleration/voltage', 'acceleration/force'])
    with pytest.raises(ValueError, match=r'different units \(acceleration/'
                                         r'force and velocity/force\): '):
        three_units.conditioning()
    csd = _csd(np.eye(3), dims=['acceleration', 'force', 'voltage'])
    with pytest.raises(ValueError, match=r'quantities \(acceleration and '
                                         r'force\): '):
        csd.conditioning()


def test_a_matrix_of_the_wrong_length_and_a_rank_one_line():
    with pytest.raises(ValueError, match='one rows-by-columns matrix per'):
        condition_matrix(F, np.ones((3, 2, 2)), ['a', 'b'], ['c', 'd'])
    reading = condition_matrix(F, np.ones((4, 2, 2)), ['a', 'b'], ['c', 'd'])
    assert reading.ill.all() and reading.notes == ()


def test_a_line_with_no_power_on_any_channel_and_a_cross_term_fails():
    s = np.repeat(np.eye(2, dtype=complex)[None], 4, axis=0)
    s[1] = [[0.0, 0.3], [0.3, 0.0]]
    reading = check_spectral_matrix(F, s, ['a', 'b'])
    np.testing.assert_allclose(reading.scale[1], [1.0, 1.0])
    assert list(reading.failing) == [False, True, False, False]
    assert np.isfinite(reading.eigenvalues).all()


def test_the_tolerance_and_the_rank_scale_with_the_largest_eigenvalue():
    # two channels, eigenvalues 1 - |c| and 1 + |c|: the smallest is
    # judged against the tolerance times the largest, about 2
    over = np.array([[1.0, 1.0 + 1.5e-6], [1.0 + 1.5e-6, 1.0]])
    reading = check_spectral_matrix(F[:1], over[None], ['a', 'b'])
    assert reading.realizable, '-1.5e-6 is within 1e-6 times 2'
    near = np.array([[1.0, 1.0 - 1e-6], [1.0 - 1e-6, 1.0]])
    reading = check_spectral_matrix(F[:1], near[None], ['a', 'b'])
    assert list(reading.rank) == [1], '1e-6 is under 1e-6 times 2'
    s = np.repeat(OPPOSED[None], 4, axis=0)
    s[2, 0, 1] = np.nan
    reading = check_spectral_matrix(F, s, ['a', 'b', 'c'])
    assert list(reading.rank) == [2, 2, 0, 2]


def test_the_largest_eigenvalue_is_the_one_named():
    pairs = {(0, 1): (1.0, 0.0), (1, 2): (0.64, 0.0), (0, 2): (0.81, 180.0)}
    m = np.eye(3, dtype=complex)
    for (i, j), (coherence, phase) in pairs.items():
        m[i, j] = np.sqrt(coherence) * np.exp(1j * np.radians(phase))
        m[j, i] = np.conj(m[i, j])
    reading = _csd(m).realizability()
    assert reading.at(20.0).describe() == (
        '20 Hz: not realizable; smallest eigenvalue -0.8025 against largest '
        '2.017, rank 2 of 3')
    words = entries_refusal(['1Z+', '2Z+', '3Z+'], pairs, [20.0, 2000.0])
    assert '(the matrix they make has an eigenvalue of -0.802 against a ' \
           'largest of 2.02)' in words


def test_the_repair_reports_the_term_it_moved_most():
    """Four channels: three that cannot hold together, and a fourth
    coherent with two of them. The fourth's terms are the largest and
    move least; the conflicting three move most, and are the ones
    reported and offered."""
    a = np.sqrt(0.81)
    m = np.array([[1, 1, -1, a], [1, 1, 1, a], [-1, 1, 1, 0], [a, a, 0, 1]],
                 dtype=complex)
    reading = check_spectral_matrix(F, np.repeat(m[None], 4, 0),
                                    ['1Z+', '2Z+', '3Z+', '4Z+'])
    assert reading.worst()[:2] == (0, 2)
    pairs = {(0, 1): (1.0, 0.0), (1, 2): (1.0, 0.0), (0, 2): (1.0, 180.0),
             (0, 3): (0.81, 0.0), (1, 3): (0.81, 0.0), (2, 3): (0.0, 0.0)}
    words = entries_refusal(['1Z+', '2Z+', '3Z+', '4Z+'], pairs,
                            [20.0, 2000.0])
    shown = words.split('nearest entries that hold are ')[1]
    assert shown.startswith('1Z+–3Z+ coherence ')
    assert '4Z+' not in shown.split(' and 3 more')[0]


def test_terms_moved_equally_report_the_first_whatever_the_rounding():
    """In the matrix above, 1Z+–3Z+ and 2Z+–3Z+ are moved by the same
    amount, differing only in the last digit, and which is larger
    depends on the platform's rounding. The first is reported, so the
    reading is the same everywhere, even with the second a rounding
    above."""
    a = np.sqrt(0.81)
    m = np.array([[1, 1, -1, a], [1, 1, 1, a], [-1, 1, 1, 0], [a, a, 0, 1]],
                 dtype=complex)
    reading = check_spectral_matrix(F, np.repeat(m[None], 4, 0),
                                    ['1Z+', '2Z+', '3Z+', '4Z+'])
    change = np.abs(reading.repaired - reading.coherence)[0]
    assert change[0, 2] == pytest.approx(change[1, 2], rel=1e-12)
    nudged = reading.repaired.copy()
    nudged[:, 1, 2] = (reading.coherence[:, 1, 2]
                       + (reading.repaired[:, 1, 2]
                          - reading.coherence[:, 1, 2]) * (1.0 + 1e-12))
    assert dataclasses.replace(reading, repaired=nudged).worst()[:2] == (0, 2)


def test_a_phase_where_the_coherence_was_zero_is_not_a_change():
    """A pair written at no coherence has no phase to change: the repair
    gives it a coherence at 120 degrees, and that is not counted as a
    turn of 120 degrees."""
    turn = np.exp(1j * np.radians(60.0))
    m = np.array([[1, turn, 0], [np.conj(turn), 1, turn],
                  [0, np.conj(turn), 1]])
    reading = check_spectral_matrix(F, np.repeat(m[None], 4, 0),
                                    ['a', 'b', 'c'])
    coherence, phase = reading.changes()
    assert coherence > 0.4
    assert phase == pytest.approx(0.0, abs=1e-6)


def test_the_refusal_names_the_channels_that_carry_the_conflict():
    names = ['1Z+', '2Z+', '3Z+']
    # two nearly one channel, the third coupled weakly: the conflict is
    # carried by the first two, the third's share of it under the line
    two = {(0, 1): (0.99, 0.0), (0, 2): (0.1, 0.0), (1, 2): (0.1, 180.0)}
    words = entries_refusal(names, two, [20.0, 2000.0])
    assert words.startswith('the cross terms of 1Z+ and 2Z+ cannot hold')
    # a phase the repair keeps at zero reads 0, not -0, and the three
    # terms of three channels have no 'more'; the two moved equally keep
    # the channels' order
    assert words.endswith('The nearest entries that hold are 1Z+–2Z+ '
                          'coherence 0.759 at 0°; 1Z+–3Z+ coherence 0.0643 '
                          'at 0°; 2Z+–3Z+ coherence 0.0643 at 180°')
    # every channel's share over the line, the smallest near it
    three = {(0, 1): (0.99, 0.0), (0, 2): (0.5, 0.0), (1, 2): (0.5, 180.0)}
    assert entries_refusal(names, three, [20.0, 2000.0]).startswith(
        'the cross terms of 1Z+, 2Z+ and 3Z+ cannot hold')
    # a fourth channel with a small part in the conflict is not named
    small = {(0, 1): (1.0, 0.0), (1, 2): (1.0, 0.0), (0, 2): (1.0, 180.0),
             (0, 3): (0.1, 0.0), (1, 3): (0.0, 0.0), (2, 3): (0.0, 0.0)}
    assert entries_refusal(names + ['4Z+'], small, [20.0, 2000.0]).startswith(
        'the cross terms of 1Z+, 2Z+ and 3Z+ cannot hold')
    # phases to a hundredth of a degree
    odd = {(0, 1): (1.0, 0.0), (1, 2): (0.64, 47.0), (0, 2): (0.81, 203.0)}
    assert ('1Z+–2Z+ coherence 0.348 at 4.71°; '
            in entries_refusal(names, odd, [20.0, 2000.0]))
