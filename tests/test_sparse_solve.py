"""The sparse symmetric factorization behind the modal solve: Apple's
Accelerate on a Mac, SuperLU in symmetric mode everywhere, and the two
giving one answer.

The Frame-Wing (270,522 degrees of freedom) solved to 2 kHz in 28.7 s
through SuperLU as it was, 12.9 s through SuperLU told the matrix is
symmetric, and 8.0 s through Accelerate, every frequency the same to
1e-12. The suite's models are small, so these tests pin the rules and
the agreement, not the seconds.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import pytest
import scipy.sparse as sp

from visualdynamics.core import sparse_solve
from visualdynamics.core.sparse_solve import (
    BACKEND_VARIABLE,
    AccelerateFactor,
    SuperLUFactor,
    factor_symmetric,
)

on_mac = pytest.mark.skipif(sys.platform != 'darwin',
                            reason='Accelerate is the macOS framework')


def _laplacian(n=400, shift=0.0):
    """A symmetric tridiagonal system: positive definite at shift 0,
    indefinite once shifted past its lowest eigenvalue."""
    main = np.full(n, 2.0) - shift
    off = np.full(n - 1, -1.0)
    return sp.diags([off, main, off], [-1, 0, 1], format='csc')


def test_the_struct_layouts_are_the_headers():
    """The ctypes layouts are copied from Accelerate's Sparse/Solve.h;
    a field out of place would hand the framework garbage without a
    word. The sizes are what those definitions come to on LP64."""
    if ctypes.sizeof(ctypes.c_long) != 8:
        pytest.skip('LP64 sizes')
    sizes = {'SparseMatrixStructure': sparse_solve._MatrixStructure,
             'SparseMatrix_Double': sparse_solve._SparseMatrix,
             'DenseMatrix_Double': sparse_solve._DenseMatrix,
             'SparseOpaqueSymbolicFactorization': sparse_solve._Symbolic,
             'SparseOpaqueFactorization_Double': sparse_solve._Factorization,
             'SparseSymbolicFactorOptions': sparse_solve._SymbolicOptions,
             'SparseNumericFactorOptions': sparse_solve._NumericOptions}
    assert {name: ctypes.sizeof(kind) for name, kind in sizes.items()} \
        == sparse_solve.STRUCT_SIZES


@pytest.mark.parametrize('rhs_shape', [(400,), (400, 3)])
def test_superlu_solves_a_symmetric_system(rhs_shape):
    a = _laplacian()
    b = np.random.default_rng(1).standard_normal(rhs_shape)
    x = SuperLUFactor(a).solve(b)
    assert x.shape == b.shape
    assert np.linalg.norm(a @ x - b) <= 1e-10 * np.linalg.norm(b)


def test_superlu_is_told_the_matrix_is_symmetric(monkeypatch):
    """The general ordering factored a 135,000-DOF model in 14 s with
    164 M nonzeros in its factors; the symmetric ordering in 3.8 s with
    87 M. The options are what make the difference."""
    import scipy.sparse.linalg as sla

    seen = {}
    real = sla.splu

    def spy(matrix, **options):
        seen.update(options)
        return real(matrix, **options)

    monkeypatch.setattr(sla, 'splu', spy)
    SuperLUFactor(_laplacian())
    assert seen['permc_spec'] == 'MMD_AT_PLUS_A'
    assert seen['options'] == {'SymmetricMode': True}


@on_mac
@pytest.mark.parametrize('rhs_shape', [(400,), (400, 3)])
def test_accelerate_solves_a_positive_definite_system_by_cholesky(rhs_shape):
    a = _laplacian()
    b = np.random.default_rng(2).standard_normal(rhs_shape)
    factor = AccelerateFactor(a)
    assert factor.kind == 'Cholesky'
    x = factor.solve(b)
    assert x.shape == b.shape
    assert np.linalg.norm(a @ x - b) <= 1e-10 * np.linalg.norm(b)


@on_mac
def test_accelerate_takes_an_indefinite_matrix_by_ldlt():
    """A shift past the lowest eigenvalue leaves the matrix indefinite:
    Cholesky is refused and LDLT takes it, with the same answer."""
    a = _laplacian(shift=1.0)
    b = np.random.default_rng(3).standard_normal(400)
    factor = AccelerateFactor(a)
    assert factor.kind == 'LDLT'
    x = factor.solve(b)
    assert np.linalg.norm(a @ x - b) <= 1e-9 * np.linalg.norm(b)
    assert np.allclose(x, SuperLUFactor(a).solve(b))


@on_mac
@pytest.mark.parametrize('singular', [
    sp.csc_matrix((400, 400)),
    sp.csc_matrix(np.diag([1.0, 2.0, 0.0, 3.0])),
    sp.csc_matrix(np.ones((6, 6))),
])
def test_a_singular_matrix_is_refused_not_answered_with_nan(singular):
    """Accelerate's LDLT reports success on an exactly singular matrix
    and its solves are NaN; the factor probes once and refuses, so the
    matrix goes on to SuperLU and the modal solve's own message."""
    with pytest.raises(ValueError, match='singular'):
        AccelerateFactor(singular)
    with pytest.raises(RuntimeError, match='singular'):
        factor_symmetric(singular)


def test_the_backend_is_accelerate_on_a_mac_and_superlu_elsewhere(monkeypatch):
    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)
    factor = factor_symmetric(_laplacian())
    assert factor.backend == ('accelerate' if sys.platform == 'darwin'
                              else 'superlu')
    monkeypatch.setenv(BACKEND_VARIABLE, 'superlu')
    assert factor_symmetric(_laplacian()).backend == 'superlu'


def test_a_refusal_falls_back_to_superlu(monkeypatch):
    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)

    def refuse(self, matrix):
        raise ValueError('Accelerate refused the matrix: a test')

    monkeypatch.setattr(AccelerateFactor, '__init__', refuse)
    assert factor_symmetric(_laplacian()).backend == 'superlu'


@on_mac
def test_the_modes_are_the_same_through_either_factorization(monkeypatch):
    """The sparse solver and its polish both solve with the factor:
    the two backends must give one set of modes."""
    from test_fem import straight_beam

    from visualdynamics.core import fem

    model = straight_beam(1.0, fem.Section.rectangle('bar', 0.02, 0.01), 30,
                          axis=(0.3, 1.0, 0.2))
    found = {}
    for backend in ('accelerate', 'superlu'):
        monkeypatch.setenv(BACKEND_VARIABLE, backend)
        found[backend] = model.eigensolution(num_modes=12, fixed=['100'],
                                             solver='sparse').frequency
    assert np.allclose(found['accelerate'], found['superlu'], rtol=1e-9,
                       atol=1e-9)
    assert found['superlu'][0] > 1.0, 'a cantilever, no rigid modes'


def test_the_modal_solve_and_its_polish_take_their_factor_here(monkeypatch):
    """`_sparse_modes` and `polish_modes` both factor (K - sigma M)
    through `factor_symmetric`, so a machine's best solver carries the
    whole solve, not one half of it."""
    from test_fem import straight_beam

    from visualdynamics.core import fem

    asked = []
    real = sparse_solve.factor_symmetric

    def spy(matrix):
        asked.append(matrix.shape)
        return real(matrix)

    monkeypatch.setattr(sparse_solve, 'factor_symmetric', spy)
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 8)
    model.eigensolution(num_modes=4, fixed=['100'], solver='sparse')
    assert len(asked) == 1, 'one factorization for the Lanczos and the polish'
    k, m = model.matrices()
    asked.clear()
    fem.polish_modes(sp.csc_matrix(k), sp.csc_matrix(m), -1.0,
                     np.random.default_rng(4).standard_normal((k.shape[0], 3)),
                     2)
    assert asked == [k.shape], 'the polish on its own factors here too'
