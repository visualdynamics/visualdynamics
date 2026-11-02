"""A sparse symmetric matrix factored once and solved many times, by the
fastest solver the machine has.

The modal solve of a large model factors (K - sigma M) once and solves
with it a few hundred times: every Lanczos step and every polish step
is a solve. On macOS the factorization is Apple's, in the Accelerate
framework every Mac carries (Sparse Solvers: a supernodal Cholesky, or
LDLT when the matrix is not positive definite), reached through ctypes
from the layouts in its public header. On Windows and Linux on Intel
and AMD processors it is PARDISO, the multithreaded supernodal solver
in Intel's oneMKL, through pypardiso (Cholesky, or LDLT with
Bunch-Kaufman pivoting). Everywhere else, and wherever those refuse a
matrix, it is SuperLU's LU through scipy, told the matrix is symmetric
so it orders and pivots for that.
"""

from __future__ import annotations

import ctypes as C
import math
import os
import sys
import weakref
from typing import Any

import numpy as np
import scipy.sparse as sp

#: set to 'superlu' to keep Accelerate and PARDISO out, which is how the
#: tests run both paths on one machine
BACKEND_VARIABLE = 'VISUALDYNAMICS_SPARSE_SOLVER'


def _superlu_only() -> bool:
    return os.environ.get(BACKEND_VARIABLE, '').lower() == 'superlu'


#: the backward error a probe solve may have before the factor is
#: refused as that of a matrix singular to working precision. A
#: factorization reports success on such a matrix and hands back a
#: factor whose solves are garbage: a tied model with a rotation
#: nothing held had solves with a backward error of 11 under SuperLU
#: and finite garbage under Accelerate's LDLT (2026-10-19), and two
#: eigensolvers ran on them. A sound factor of an ill-conditioned
#: matrix measures 1e-8 to 2e-6 on the models here (the springs' the
#: worst), a garbage one 0.25, 11 and 6e8.
PROBE_LIMIT = 1e-3


def probe_backward_error(matrix: Any, solve: Any) -> float:
    """|A x - b| / |b| for one solve with a fixed right-hand side, the
    cosines of the row numbers; infinite when the answer is not."""
    probe = np.cos(np.arange(matrix.shape[0], dtype=np.float64))
    answer = solve(probe)
    if not np.isfinite(answer).all():
        return math.inf
    return float(np.linalg.norm(matrix @ answer - probe) / np.linalg.norm(probe))


def count_below(stiffness: Any, mass: Any, shift: float) -> int:
    """How many eigenvalues of K phi = lambda M phi lie below `shift`,
    read off the signs of one factorization of K - shift M.

    By Sylvester's law of inertia, a symmetric matrix factored as
    L D Lᵀ has as many negative entries in D as it has negative
    eigenvalues, and the negative eigenvalues of K - shift M are the
    modes below the shift (M positive definite). Where PARDISO is
    available its LDLT reports the count itself; Bunch-Kaufman's 2 x 2
    pivots keep the count. Elsewhere SuperLU in symmetric mode with no
    pivoting off the diagonal is that factorization, its U's diagonal
    being D, which is why the threshold is zero here and not the
    factor the solves use. A pivot within round-off of zero,
    from a shift on top of an eigenvalue, has a sign that is round-off,
    so a caller allows the count one either way.

    Parameters
    ----------
    stiffness, mass : scipy.sparse matrix
        K and M, symmetric, M positive definite.
    shift : float
        The eigenvalue to count below, (rad/s)².

    Returns
    -------
    int
        The count, rigid-body modes included when the shift is above
        zero.

    Raises
    ------
    RuntimeError
        If the factorization fails, had to pivot off the diagonal or
        perturbed a pivot, so its signs are not the inertia.

    References
    ----------
    A standard method, used here to size the search up to a frequency:

    1. J. J. Sylvester, *A demonstration of the theorem that every
       homogeneous quadratic polynomial is reducible by real orthogonal
       substitutions to the form of a sum of positive and negative
       squares*, Philosophical Magazine, 1852.
    """
    from scipy.sparse.linalg import splu

    shifted = sp.csc_matrix(stiffness - float(shift) * mass)
    # 1.5 s where SuperLU took 29 on a 135,000-DOF frame, 5.2 s to 41 on
    # a 106,000-DOF block of bricks, the same counts (2026-11-02)
    if not _superlu_only() and _pypardiso() is not None:
        return _pardiso_negatives(shifted)
    factor = splu(shifted,
                  permc_spec='MMD_AT_PLUS_A',
                  options={'SymmetricMode': True, 'DiagPivotThresh': 0.0})
    if not np.array_equal(factor.perm_r, factor.perm_c):
        raise RuntimeError('the factorization pivoted off the diagonal')
    return int(np.count_nonzero(factor.U.diagonal() < 0.0))


class SuperLUFactor:
    """SuperLU's factorization of a symmetric matrix, through scipy."""

    backend = 'superlu'

    def __init__(self, matrix: Any) -> None:
        from scipy.sparse.linalg import splu

        # Told the matrix is symmetric (2026-10-18): the default ordering
        # treats it as general, and on a 135,000-DOF model factored in
        # 14 s with 164 M nonzeros in the factors where this takes 3.8 s
        # and 87 M; every solve is half the work as well
        self._factor = splu(sp.csc_matrix(matrix), permc_spec='MMD_AT_PLUS_A',
                            options={'SymmetricMode': True})
        self.shape: tuple[int, int] = self._factor.shape
        residual = probe_backward_error(matrix, self.solve)
        if not residual < PROBE_LIMIT:
            raise RuntimeError('the matrix is singular to working precision: '
                               f'a probe solve has a backward error of {residual:.1e}')

    def solve(self, rhs: Any) -> np.ndarray:
        """X with A X = `rhs`, for one right-hand side or a block of
        them, in the shape `rhs` came in."""
        return self._factor.solve(np.asarray(rhs, dtype=np.float64))


# ---- Accelerate's Sparse Solvers, from <Accelerate/Sparse/Solve.h> --------
#
# The public functions in that header are inline wrappers over the
# `_Sparse*_Double` entry points below, and the layouts are copied from
# its struct definitions; `tests/test_sparse_solve.py` pins their sizes.

class _MatrixStructure(C.Structure):
    _fields_ = [('rowCount', C.c_int), ('columnCount', C.c_int),
                ('columnStarts', C.POINTER(C.c_long)),
                ('rowIndices', C.POINTER(C.c_int)),
                ('attributes', C.c_uint32), ('blockSize', C.c_uint8)]


class _SparseMatrix(C.Structure):
    _fields_ = [('structure', _MatrixStructure),
                ('data', C.POINTER(C.c_double))]


class _DenseMatrix(C.Structure):
    _fields_ = [('rowCount', C.c_int), ('columnCount', C.c_int),
                ('columnStride', C.c_int), ('attributes', C.c_uint32),
                ('data', C.POINTER(C.c_double))]


class _Symbolic(C.Structure):
    _fields_ = [('status', C.c_int32), ('rowCount', C.c_int),
                ('columnCount', C.c_int), ('attributes', C.c_uint32),
                ('blockSize', C.c_uint8), ('type', C.c_uint8),
                ('factorization', C.c_void_p),
                ('workspaceSize_Float', C.c_size_t),
                ('workspaceSize_Double', C.c_size_t),
                ('factorSize_Float', C.c_size_t),
                ('factorSize_Double', C.c_size_t)]


class _Factorization(C.Structure):
    _fields_ = [('status', C.c_int32), ('attributes', C.c_uint32),
                ('symbolic', _Symbolic), ('userFactorStorage', C.c_bool),
                ('numeric', C.c_void_p),
                ('solveWorkspaceRequiredStatic', C.c_size_t),
                ('solveWorkspaceRequiredPerRHS', C.c_size_t)]


_MALLOC = C.CFUNCTYPE(C.c_void_p, C.c_size_t)
_FREE = C.CFUNCTYPE(None, C.c_void_p)
_REPORT = C.CFUNCTYPE(None, C.c_char_p)


class _SymbolicOptions(C.Structure):
    _fields_ = [('control', C.c_uint32), ('orderMethod', C.c_uint8),
                ('order', C.POINTER(C.c_int)),
                ('ignoreRowsAndColumns', C.POINTER(C.c_int)),
                ('malloc', _MALLOC), ('free', _FREE), ('reportError', _REPORT)]


class _NumericOptions(C.Structure):
    _fields_ = [('control', C.c_uint32), ('scalingMethod', C.c_uint8),
                ('scaling', C.c_void_p), ('pivotTolerance', C.c_double),
                ('zeroTolerance', C.c_double)]


#: the sizes the header's layouts come to on a 64-bit machine
STRUCT_SIZES = {'SparseMatrixStructure': 32, 'SparseMatrix_Double': 40,
                'DenseMatrix_Double': 24, 'SparseOpaqueSymbolicFactorization': 64,
                'SparseOpaqueFactorization_Double': 104,
                'SparseSymbolicFactorOptions': 48, 'SparseNumericFactorOptions': 32}

#: SparseAttributes_t: transpose (1 bit), triangle (1 bit, lower = 1),
#: kind (2 bits, symmetric = 3) — the lower triangle of a symmetric matrix
_LOWER_SYMMETRIC = (1 << 1) | (3 << 2)
#: SparseFactorization_t
_CHOLESKY, _LDLT = 0, 1
_STATUS = {0: 'ok', -1: 'the factorization failed', -2: 'the matrix is singular',
           -3: 'an internal error', -4: 'a parameter error'}

_accelerate: dict[str, Any] = {}


def _library() -> dict[str, Any] | None:
    """The entry points, bound once; None where Accelerate is not."""
    if _accelerate:
        return _accelerate if _accelerate.get('ok') else None
    _accelerate['ok'] = False
    if sys.platform != 'darwin':
        return None
    try:
        lib = C.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
        libc = C.CDLL(None)
        factor = lib._SparseFactorSymmetric_Double
        solve = lib._SparseSolveOpaque_Double
        destroy = lib._SparseDestroyOpaqueNumeric_Double
    except (OSError, AttributeError):
        return None
    factor.restype = _Factorization
    factor.argtypes = [C.c_uint8, C.POINTER(_SparseMatrix),
                       C.POINTER(_SymbolicOptions), C.POINTER(_NumericOptions)]
    solve.restype = None
    solve.argtypes = [C.POINTER(_Factorization), C.POINTER(_DenseMatrix),
                      C.POINTER(_DenseMatrix), C.c_void_p]
    destroy.restype = None
    destroy.argtypes = [C.POINTER(_Factorization)]
    libc.malloc.restype = C.c_void_p
    libc.malloc.argtypes = [C.c_size_t]
    libc.free.restype = None
    libc.free.argtypes = [C.c_void_p]
    messages: list[str] = []
    report = _REPORT(lambda text: messages.append(text.decode(errors='replace')))
    _accelerate.update(
        ok=True, factor=factor, solve=solve, destroy=destroy, messages=messages,
        # Accelerate's defaults, with its errors kept rather than printed
        symbolic=_SymbolicOptions(0, 0, None, None, _MALLOC(('malloc', libc)),
                                  _FREE(('free', libc)), report),
        numeric=_NumericOptions(0, 0, None, 0.1, 0.0), report=report)
    return _accelerate


class AccelerateFactor:
    """Apple's supernodal factorization of a symmetric matrix: Cholesky,
    or LDLT when the matrix is not positive definite.

    Raises `ValueError` when Accelerate refuses the matrix, naming its
    status, so the caller can fall back.
    """

    backend = 'accelerate'

    def __init__(self, matrix: Any) -> None:
        lib = _library()
        if lib is None:
            raise ValueError('Accelerate is not available here')
        lower = sp.tril(matrix, format='csc')
        lower.sort_indices()
        # kept for the factorization's life: Accelerate reads them in
        # place and the factor may refer back to the structure
        self._starts = np.ascontiguousarray(lower.indptr, dtype=np.int64)
        self._rows = np.ascontiguousarray(lower.indices, dtype=np.int32)
        self._values = np.ascontiguousarray(lower.data, dtype=np.float64)
        n = lower.shape[0]
        self.shape: tuple[int, int] = (n, n)
        self._matrix = _SparseMatrix(
            _MatrixStructure(n, n, self._starts.ctypes.data_as(C.POINTER(C.c_long)),
                             self._rows.ctypes.data_as(C.POINTER(C.c_int)),
                             _LOWER_SYMMETRIC, 1),
            self._values.ctypes.data_as(C.POINTER(C.c_double)))
        self._factor = None
        for kind in (_CHOLESKY, _LDLT):
            del lib['messages'][:]
            factored = lib['factor'](kind, C.byref(self._matrix),
                                     C.byref(lib['symbolic']), C.byref(lib['numeric']))
            if factored.status == 0:
                self._factor = factored
                self.kind: str = 'Cholesky' if kind == _CHOLESKY else 'LDLT'
                # LDLT reports success on an exactly singular matrix and
                # its solves come back NaN (measured 2026-10-18: an empty
                # matrix, a zero row, a rank-one block all 'ok'), or
                # finite garbage on one singular to working precision
                # (2026-10-19, a backward error of 0.25), where Cholesky
                # succeeding proves the matrix definite. One probe solve
                # tells, and a refusal here sends the matrix on to
                # SuperLU, whose own probe the modal solve turns into
                # its message
                if kind == _LDLT and not probe_backward_error(
                        matrix, self.solve) < PROBE_LIMIT:
                    lib['destroy'](C.byref(factored))
                    self._factor = None
                    raise ValueError('Accelerate found the matrix singular')
                return
            lib['destroy'](C.byref(factored))
        why = _STATUS.get(factored.status, f'status {factored.status}')
        said = '; '.join(m.strip() for m in lib['messages'] if m.strip())
        raise ValueError(f'Accelerate refused the matrix: {why}'
                         + (f' ({said})' if said else ''))

    def solve(self, rhs: Any) -> np.ndarray:
        """X with A X = `rhs`, for one right-hand side or a block of
        them, in the shape `rhs` came in."""
        lib = _library()
        given = np.asarray(rhs, dtype=np.float64)
        b = np.asfortranarray(given.reshape(given.shape[0], -1))
        n, count = b.shape
        x = np.zeros((n, count), dtype=np.float64, order='F')
        right = _DenseMatrix(n, count, n, 0, b.ctypes.data_as(C.POINTER(C.c_double)))
        out = _DenseMatrix(n, count, n, 0, x.ctypes.data_as(C.POINTER(C.c_double)))
        needed = (self._factor.solveWorkspaceRequiredStatic
                  + count * self._factor.solveWorkspaceRequiredPerRHS)
        workspace = C.create_string_buffer(int(needed) + 64)
        lib['solve'](C.byref(self._factor), C.byref(right), C.byref(out),
                     C.cast(workspace, C.c_void_p))
        return x.reshape(given.shape) if given.ndim == 1 else x

    def __del__(self) -> None:
        lib = _accelerate if _accelerate.get('ok') else None
        if lib is not None and getattr(self, '_factor', None) is not None:
            lib['destroy'](C.byref(self._factor))
            self._factor = None


# ---- PARDISO, in Intel's oneMKL, through pypardiso -------------------------
#
# A dependency on x86-64 Windows and Linux only, where oneMKL is built
# (Brandon, 2026-11-02). A Windows machine's 193,627-brick modal solve
# took 70 minutes through SuperLU where a Mac's Accelerate took two;
# measured on a two-core Linux laptop, PARDISO factored a 106,000-DOF
# block of bricks in 6.1 s on one thread and 3.6 s on four, against
# SuperLU's 239 s, and the 608,000-DOF block of 192,000 bricks in 106 s.
# Its own nested-dissection ordering (METIS) is most of that: the
# factor had half SuperLU's nonzeros on the block, and fewer on a thin
# frame as well. PLAN.md, "PARDISO on Windows and Linux".

_pardiso: dict[str, Any] = {}


def _pypardiso() -> Any:
    """The pypardiso module, imported once; None where it or the oneMKL
    library it loads is not."""
    if 'module' not in _pardiso:
        # threaded through TBB, before oneMKL loads (Brandon, 2026-11-02):
        # the packaged builds carry TBB and not the OpenMP runtime, which
        # is under Intel's developer-tools EULA, and TBB factored as fast
        # on the measured machine. A default, so a user's own choice
        # stands where the library for it is installed
        os.environ.setdefault('MKL_THREADING_LAYER', 'TBB')
        try:
            import pypardiso
        except (ImportError, OSError) as missing:
            pypardiso = None
            _pardiso['why'] = f'{type(missing).__name__}: {missing}'
        _pardiso['module'] = pypardiso
    return _pardiso['module']


def expected_backend() -> str:
    """The backend this machine should factor with: 'accelerate' on a
    Mac, 'pardiso' on x86-64 Windows and Linux (where `pypardiso` is a
    dependency), 'superlu' anywhere else."""
    import platform

    if sys.platform == 'darwin':
        return 'accelerate'
    if platform.machine() in ('x86_64', 'AMD64'):
        return 'pardiso'
    return 'superlu'


def check() -> tuple[int, str]:
    """Check that this installation factors with the solver it should.

    `visualdynamics --check-solver` runs this and exits with the
    returned code. The packaged builds are tested with it, because a
    bundle that leaves oneMKL out still solves, through SuperLU, many
    times slower on a large model.

    Returns
    -------
    tuple of (int, str)
        0 and the backend, with the library it loaded, when a small
        model's factorization took `expected_backend` and its solve
        and mode count are right; 1 and what happened otherwise.
    """
    n = 1000
    matrix = sp.diags([np.full(n - 1, -1.0), np.full(n, 2.5), np.full(n - 1, -1.0)],
                      [-1, 0, 1], format='csc')
    expected = expected_backend()
    try:
        factor = factor_symmetric(matrix)
        rhs = np.cos(np.arange(n, dtype=np.float64))
        residual = float(np.linalg.norm(matrix @ factor.solve(rhs) - rhs)
                         / np.linalg.norm(rhs))
        below = count_below(matrix, sp.identity(n, format='csc'), 1.0)
    except Exception as failure:  # noqa: BLE001 — whatever the packaged app would hit
        return 1, f'the factorization failed: {type(failure).__name__}: {failure}'
    # the eigenvalues are 2.5 - 2 cos(k pi / (n + 1)), k = 1 to n
    eigenvalues = 2.5 - 2.0 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))
    exact = int(np.count_nonzero(eigenvalues < 1.0))
    where = ''
    if factor.backend == 'pardiso':
        from threadpoolctl import threadpool_info

        layers = {pool.get('threading_layer') for pool in threadpool_info()
                  if pool['internal_api'] == 'mkl'}
        where = f' ({factor._solver.libmkl._name}, threaded by {", ".join(sorted(map(str, layers)))})'
    said = f'{factor.backend}{where}: residual {residual:.1e}, {below} of {n} modes below 1.0'
    if factor.backend != expected:
        why = _pardiso.get('why', '') if expected == 'pardiso' else ''
        return 1, f'expected {expected}, got {said}' + (f'; {why}' if why else '')
    if not (residual < 1e-12 and below == exact):
        return 1, f'wrong answers from {said}; {exact} are below 1.0'
    return 0, said


def _upper_triangle(matrix: Any) -> Any:
    """The upper triangle, CSR, every diagonal entry stored even where
    it is zero, as PARDISO requires of a symmetric matrix."""
    strict = sp.triu(matrix, k=1, format='coo')
    index = np.arange(matrix.shape[0])
    # built from triplets: a sum of matrices would drop the zeros
    upper = sp.csr_matrix(
        (np.concatenate([np.asarray(matrix.diagonal(), dtype=np.float64),
                         strict.data.astype(np.float64)]),
         (np.concatenate([index, strict.row]), np.concatenate([index, strict.col]))),
        shape=matrix.shape)
    upper.sort_indices()
    return upper


#: PARDISO's iparm, 1-based as its manual numbers them: the settings
#: given rather than oneMKL's defaults (1), nested dissection by
#: parallel METIS (2), pivots perturbed at 1e-8 in LDLT (10), no
#: scaling (11) and no matching (13), which the matrix's own symmetric
#: scaling makes needless, Bunch-Kaufman pivots (21), and the two-level
#: factorization that spreads over many cores (24)
PARDISO_SETTINGS = {1: 1, 2: 3, 10: 8, 11: 0, 13: 0, 21: 1, 24: 1}

#: iterative refinement steps on each solve (iparm 8), by matrix type.
#: None for Cholesky: its factor is exact, and a refined answer is
#: corrected by its own residual, so the operator is no longer
#: symmetric. oneMKL's default of two steps left it 1.3e-8 off
#: symmetric on the rigid-link test model, against 7e-16 unrefined, and
#: the block Lanczos, which reads that asymmetry as part of every
#: pair's residual, left a free plate's 60th mode 1.2e-5 off where
#: SuperLU's solves gave 1.7e-12. Two for LDLT: its perturbed pivots
#: factor a nearby matrix, and refinement brings the solve back, a
#: shifted Laplacian's backward error 1.5e-8 to 8e-16 (2026-11-02)
PARDISO_REFINEMENT = {2: 0, -2: 2}


def _pardiso_solver(mtype: int) -> Any:
    """A pypardiso solver for matrix type `mtype` (2 positive definite,
    -2 symmetric indefinite), with `PARDISO_SETTINGS` and the type's
    `PARDISO_REFINEMENT`."""
    # the matrix is kept, not hashed, to tell the factored one at each
    # solve: a hash of a large model's matrix took longer than the solve
    # it guarded
    solver = _pypardiso().PyPardisoSolver(mtype=mtype, size_limit_storage=math.inf)
    for index, value in PARDISO_SETTINGS.items():
        solver.set_iparm(index, value)
    solver.set_iparm(8, PARDISO_REFINEMENT[mtype])
    return solver


def _pardiso_negatives(matrix: Any) -> int:
    """The negative eigenvalue count of a symmetric matrix, from
    PARDISO's LDLT (`count_below`)."""
    from pypardiso.pardiso_wrapper import PyPardisoError

    solver = _pardiso_solver(-2)
    try:
        solver.factorize(_upper_triangle(matrix))
        perturbed = int(solver.get_iparm(14))
        negatives = int(solver.get_iparm(23))
    except (PyPardisoError, ValueError) as failure:
        raise RuntimeError(f'PARDISO could not factor the matrix: {failure}') from None
    finally:
        solver.free_memory(everything=True)
    if perturbed:
        raise RuntimeError(f'PARDISO perturbed {perturbed} pivots')
    return negatives


class PardisoFactor:
    """PARDISO's supernodal factorization of a symmetric matrix:
    Cholesky, or LDLT when the matrix is not positive definite.

    Raises `ValueError` when PARDISO is not available or refuses the
    matrix, so the caller can fall back.
    """

    backend = 'pardiso'

    def __init__(self, matrix: Any) -> None:
        pypardiso = _pypardiso()
        if pypardiso is None:
            raise ValueError('PARDISO is not available here')
        from pypardiso.pardiso_wrapper import PyPardisoError

        self._upper = _upper_triangle(matrix)
        n = self._upper.shape[0]
        self.shape: tuple[int, int] = (n, n)
        self._solver = None
        failure: Exception | None = None
        for mtype, kind in ((2, 'Cholesky'), (-2, 'LDLT')):
            solver = _pardiso_solver(mtype)
            try:
                solver.factorize(self._upper)
            except (PyPardisoError, ValueError) as refused:
                solver.free_memory(everything=True)
                failure = refused
                continue
            self._solver, self.kind = solver, kind
            # oneMKL's own memory, which Python does not see, freed when
            # the factor is dropped: without it a factor held its
            # gigabytes until exit. A finalizer rather than __del__,
            # which ran during interpreter shutdown, after the modules
            # pypardiso's release calls on were gone
            self._finalizer = weakref.finalize(self, solver.free_memory, True)
            # LDLT perturbs a pivot within round-off of zero and reports
            # success on a singular matrix, as Accelerate's LDLT does;
            # one probe solve tells, and is cheap beside the factor
            if not probe_backward_error(matrix, self.solve) < PROBE_LIMIT:
                self._release()
                raise ValueError('PARDISO found the matrix singular')
            return
        raise ValueError(f'PARDISO refused the matrix: {failure}')

    def solve(self, rhs: Any) -> np.ndarray:
        """X with A X = `rhs`, for one right-hand side or a block of
        them, in the shape `rhs` came in."""
        return self._solver.solve(self._upper, np.asarray(rhs, dtype=np.float64))

    def _release(self) -> None:
        self._finalizer()
        self._solver = None


def factor_symmetric(matrix: Any) -> AccelerateFactor | PardisoFactor | SuperLUFactor:
    """The matrix factored by the fastest solver here that takes it:
    Accelerate on a Mac, PARDISO on x86-64 Windows and Linux, else
    SuperLU. Each answers `solve(rhs)` and `shape`.

    Parameters
    ----------
    matrix : sparse matrix
        Square and symmetric.

    Returns
    -------
    AccelerateFactor, PardisoFactor or SuperLUFactor
    """
    if not _superlu_only():
        for fast in (AccelerateFactor, PardisoFactor):
            try:
                return fast(matrix)
            except ValueError:
                pass
    return SuperLUFactor(matrix)
