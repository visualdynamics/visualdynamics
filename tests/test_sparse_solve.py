"""The sparse symmetric factorization behind the modal solve: Apple's
Accelerate on a Mac, PARDISO on x86-64 Windows and Linux, SuperLU in
symmetric mode everywhere, and each giving one answer.

The Frame-Wing (270,522 degrees of freedom) solved to 2 kHz in 28.7 s
through SuperLU as it was, 12.9 s through SuperLU told the matrix is
symmetric, and 8.0 s through Accelerate, every frequency the same to
1e-12. The suite's models are small, so these tests pin the rules and
the agreement, not the seconds.
"""

from __future__ import annotations

import ctypes
import re
import sys

import numpy as np
import pytest
import scipy.sparse as sp

from visualdynamics.core import sparse_solve
from visualdynamics.core.sparse_solve import (
    BACKEND_VARIABLE,
    AccelerateFactor,
    PardisoFactor,
    SuperLUFactor,
    factor_symmetric,
)

on_mac = pytest.mark.skipif(sys.platform != 'darwin',
                            reason='Accelerate is the macOS framework')
on_pardiso = pytest.mark.skipif(sparse_solve._pypardiso() is None,
                                reason='oneMKL is built for x86-64 Windows and Linux')


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


@on_pardiso
@pytest.mark.parametrize('rhs_shape', [(400,), (400, 3)])
def test_pardiso_solves_a_positive_definite_system_by_cholesky(rhs_shape):
    a = _laplacian()
    b = np.random.default_rng(5).standard_normal(rhs_shape)
    factor = PardisoFactor(a)
    assert factor.kind == 'Cholesky'
    x = factor.solve(b)
    assert x.shape == b.shape
    assert np.linalg.norm(a @ x - b) <= 1e-10 * np.linalg.norm(b)


@on_pardiso
def test_pardiso_takes_an_indefinite_matrix_by_ldlt():
    """Cholesky is refused past the lowest eigenvalue and LDLT takes
    the matrix, with SuperLU's answer."""
    a = _laplacian(shift=1.0)
    b = np.random.default_rng(6).standard_normal(400)
    factor = PardisoFactor(a)
    assert factor.kind == 'LDLT'
    x = factor.solve(b)
    assert np.linalg.norm(a @ x - b) <= 1e-9 * np.linalg.norm(b)
    assert np.allclose(x, SuperLUFactor(a).solve(b))


@on_pardiso
@pytest.mark.parametrize('singular', [
    sp.csc_matrix((400, 400)),
    sp.csc_matrix(np.diag([1.0, 2.0, 0.0, 3.0])),
    sp.csc_matrix(np.ones((6, 6))),
])
def test_pardiso_refuses_a_singular_matrix(singular):
    """PARDISO's LDLT perturbs a zero pivot and reports success; the
    probe refuses the factor, and the matrix goes on to SuperLU and the
    modal solve's own message."""
    with pytest.raises(ValueError, match='PARDISO'):
        PardisoFactor(singular)
    with pytest.raises(RuntimeError, match='singular'):
        factor_symmetric(singular)


@on_pardiso
def test_a_dropped_pardiso_factor_gives_back_its_memory():
    """oneMKL's factor lives outside Python, so a factor dropped
    without releasing it held its memory until the program ended: a
    gigabyte a solve on a large model. The release is a finalizer, so
    it also runs at exit while the modules it calls are still there."""
    import gc

    factor = PardisoFactor(_laplacian())
    finalizer = factor._finalizer
    # the function and its arguments only: `peek` also hands back the
    # factor, and holding it would keep it alive
    release, arguments = finalizer.peek()[1:3]
    assert release == factor._solver.free_memory and arguments == (True,)
    del factor
    gc.collect()
    assert not finalizer.alive


def test_the_fast_backends_solves_are_symmetric_to_round_off(monkeypatch):
    """The block Lanczos reads the computed operator's asymmetry as part
    of each pair's residual (`fem.block_lanczos_modes`). oneMKL refines
    every solve twice by default, and the refined operator was 1.3e-8
    off symmetric on the rigid-link model, against 7e-16 for its plain
    Cholesky: a free plate's 60th mode came out 1.2e-5 off."""
    from test_sparse_solver import _rigid_link_model

    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)
    stiffness, mass, _t, _s, sigma, _k = _rigid_link_model()[0].scaled_system()
    factor = factor_symmetric((stiffness - sigma * mass).tocsc())
    if factor.backend == 'superlu':
        pytest.skip('SuperLU is the only backend here; its LU is not symmetric')
    x, y = np.random.default_rng(7).standard_normal((2, factor.shape[0]))
    forward, backward = y @ factor.solve(x), x @ factor.solve(y)
    assert abs(forward - backward) <= 1e-13 * abs(forward)


def test_pardiso_is_given_every_diagonal_entry():
    """PARDISO requires a symmetric matrix's diagonal stored, zeros
    included; a sum of scipy matrices drops them, and a zero diagonal
    is the singular matrix the probe must see refused rather than a
    structure PARDISO misreads."""
    a = sp.csc_matrix(np.array([[0.0, 1.0, 0.0],
                                [1.0, 2.0, 0.0],
                                [0.0, 0.0, 0.0]]))
    upper = sparse_solve._upper_triangle(a)
    assert upper.format == 'csr'
    # the stored structure, read from the index arrays: `nonzero` skips
    # a stored zero
    rows = np.repeat(np.arange(3), np.diff(upper.indptr))
    assert set(zip(rows.tolist(), upper.indices.tolist())) \
        == {(0, 0), (0, 1), (1, 1), (2, 2)}
    assert np.array_equal(upper.toarray(), np.triu(a.toarray()))


def test_the_backend_is_the_fastest_the_machine_has(monkeypatch):
    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)
    factor = factor_symmetric(_laplacian())
    assert factor.backend == (
        'accelerate' if sys.platform == 'darwin'
        else 'pardiso' if sparse_solve._pypardiso() is not None
        else 'superlu')
    monkeypatch.setenv(BACKEND_VARIABLE, 'superlu')
    assert factor_symmetric(_laplacian()).backend == 'superlu'


def test_a_refusal_falls_back_to_superlu(monkeypatch):
    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)

    def refuse(self, matrix):
        raise ValueError('refused the matrix: a test')

    monkeypatch.setattr(AccelerateFactor, '__init__', refuse)
    monkeypatch.setattr(PardisoFactor, '__init__', refuse)
    assert factor_symmetric(_laplacian()).backend == 'superlu'


def test_the_modes_are_the_same_through_either_factorization(monkeypatch):
    """The sparse solver and its polish both solve with the factor:
    the machine's fast backend and SuperLU must give one set of modes."""
    from test_fem import straight_beam

    from visualdynamics.core import fem

    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)
    if factor_symmetric(_laplacian()).backend == 'superlu':
        pytest.skip('SuperLU is the only backend here')
    model = straight_beam(1.0, fem.Section.rectangle('bar', 0.02, 0.01), 30,
                          axis=(0.3, 1.0, 0.2))
    found = {}
    for backend in ('', 'superlu'):
        monkeypatch.setenv(BACKEND_VARIABLE, backend)
        found[backend] = model.eigensolution(num_modes=12, fixed=['100'],
                                             solver='sparse').frequency
    assert np.allclose(found[''], found['superlu'], rtol=1e-9, atol=1e-9)
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


def test_the_solver_check_passes_on_the_backend_the_machine_should_have(monkeypatch):
    """`--check-solver` is how a packaged build shows it carries its
    fast solver: 0 here, from source, with the backend named."""
    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)
    code, message = sparse_solve.check()
    assert code == 0, message
    # PARDISO's names the library it loaded and how it is threaded
    assert re.match(rf'{sparse_solve.expected_backend()}( \(.+\))?: residual', message), message
    assert '230 of 1000 modes below 1.0' in message


def test_the_solver_check_fails_a_superlu_fallback(monkeypatch):
    """The failure it exists for: a bundle without oneMKL still solves,
    through SuperLU, and only this says so."""
    if sparse_solve.expected_backend() == 'superlu':
        pytest.skip('SuperLU is what this machine should have')
    monkeypatch.setenv(BACKEND_VARIABLE, 'superlu')
    code, message = sparse_solve.check()
    assert code == 1
    assert message.startswith(f'expected {sparse_solve.expected_backend()}, got superlu')


def test_the_solver_check_says_why_pardiso_did_not_load(monkeypatch):
    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)
    monkeypatch.setattr(sparse_solve, 'expected_backend', lambda: 'pardiso')
    monkeypatch.setattr(sparse_solve, '_pardiso',
                        {'module': None, 'why': 'ImportError: no mkl_rt here'})

    def refuse(self, matrix):
        raise ValueError('Accelerate is not available here')

    monkeypatch.setattr(AccelerateFactor, '__init__', refuse)
    code, message = sparse_solve.check()
    assert code == 1
    assert message.startswith('expected pardiso, got superlu')
    assert message.endswith('ImportError: no mkl_rt here')


def test_the_solver_check_fails_wrong_answers(monkeypatch):
    monkeypatch.delenv(BACKEND_VARIABLE, raising=False)
    monkeypatch.setattr(sparse_solve, 'count_below', lambda *_args: 229)
    code, message = sparse_solve.check()
    assert code == 1 and message.startswith('wrong answers from')
    assert message.endswith('230 are below 1.0')


def test_the_app_runs_the_solver_check_before_anything_qt(monkeypatch):
    from visualdynamics import gui

    monkeypatch.setattr(sparse_solve, 'check', lambda: (1, 'a test'))
    assert gui.main(['--check-solver']) == 1
    monkeypatch.setattr(sparse_solve, 'check', lambda: (0, 'a test'))
    assert gui.main(['--check-solver']) == 0


def test_the_expected_backend_follows_the_dependency_marker(monkeypatch):
    """pypardiso is declared for x86-64 Windows and Linux; the check
    expects PARDISO exactly where pip installs it."""
    import platform

    for system, machine, expected in [('linux', 'x86_64', 'pardiso'),
                                      ('win32', 'AMD64', 'pardiso'),
                                      ('darwin', 'x86_64', 'accelerate'),
                                      ('darwin', 'arm64', 'accelerate'),
                                      ('win32', 'ARM64', 'superlu'),
                                      ('linux', 'aarch64', 'superlu')]:
        monkeypatch.setattr(sparse_solve.sys, 'platform', system)
        monkeypatch.setattr(platform, 'machine', lambda machine=machine: machine)
        assert sparse_solve.expected_backend() == expected, (system, machine)


@on_pardiso
def test_pardiso_is_threaded_through_tbb():
    """The packaged builds carry TBB's threading layer and not the
    OpenMP runtime (under Intel's developer-tools EULA), so oneMKL is
    told TBB before it loads, and a pip install runs as a build does."""
    from threadpoolctl import threadpool_info

    PardisoFactor(_laplacian())
    layers = {pool.get('threading_layer') for pool in threadpool_info()
              if pool['internal_api'] == 'mkl'}
    assert layers == {'tbb'}
    code, message = sparse_solve.check()
    assert code == 0 and 'threaded by tbb' in message, message


def _benchmark():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parent.parent / 'tools' / 'solver_benchmark.py'
    spec = importlib.util.spec_from_file_location('solver_benchmark', path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    return benchmark


def test_the_solver_benchmark_times_both_factorizations(capsys):
    """`tools/solver_benchmark.py`, what the Windows runner is timed
    with: a block solved through the machine's own factorization and,
    under its SuperLU limit, through SuperLU, to the same frequencies."""
    benchmark = _benchmark()
    assert benchmark.main(['--modes', '10', '--superlu-up-to', '24', '4x3x2']) == 0
    rows = [line for line in capsys.readouterr().out.splitlines()
            if line.startswith('| 24 |')]
    own, superlu = rows[-2:]
    assert f'| {sparse_solve.expected_backend()} |' in own
    assert '| superlu |' in superlu
    assert own.split('|')[-2] == superlu.split('|')[-2], 'the same modes'
    assert own.split('|')[-2].strip().endswith(' Hz'), 'ten modes reach the elastic ones'
    benchmark.main(['--modes', '4', '--superlu-up-to', '0', '4x3x2'])
    assert 'every mode asked for is rigid' in capsys.readouterr().out


def test_a_benchmark_case_that_crashes_says_how(capsys, monkeypatch):
    """A case once died on a Linux runner with nothing on stderr and was
    reported as "failed: no output". The signal, and the crash's place
    from faulthandler, are what tell a native crash from an exception."""
    benchmark = _benchmark()
    monkeypatch.setattr(benchmark, 'CASE', 'import faulthandler; faulthandler._sigsegv()')
    outcome = benchmark.run_case((1, 1, 1), 1, False, 60.0)['outcome']
    if sys.platform == 'win32':
        assert outcome.startswith('failed: exit 0xC0000005'), outcome
    else:
        assert outcome.startswith('failed: SIGSEGV'), outcome
    assert 'Fatal Python error: Segmentation fault' in capsys.readouterr().err


def test_a_benchmark_failure_names_the_way_out():
    failure = _benchmark().failure
    assert failure(-9, '', '') == 'failed: SIGKILL, nothing on stderr'
    assert failure(-11, '{"dof": 180}\n', '') == \
        'failed: SIGSEGV after printing its result, nothing on stderr'
    assert failure(1, '', 'Traceback\nValueError: no') == 'failed: exit 1, ValueError: no'
    assert failure(0xC0000005, '', '') == 'failed: exit 0xC0000005, nothing on stderr'
    assert failure(2, 'Intel oneMKL FATAL ERROR: Cannot load libmkl_core.so.3.\n', '') == \
        'failed: exit 2, Intel oneMKL FATAL ERROR: Cannot load libmkl_core.so.3.', \
        "oneMKL's own errors go to stdout"
    assert failure(-11, '{"dof": 180}\n', 'Fatal Python error: Segmentation fault') == \
        'failed: SIGSEGV after printing its result, Fatal Python error: Segmentation fault'
