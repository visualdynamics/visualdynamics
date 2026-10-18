"""A sparse symmetric matrix factored once and solved many times, by the
fastest solver the machine has.

The modal solve of a large model factors (K - sigma M) once and solves
with it a few hundred times: every Lanczos step and every polish step
is a solve. On macOS the factorization is Apple's, in the Accelerate
framework every Mac carries (Sparse Solvers: a supernodal Cholesky, or
LDLT when the matrix is not positive definite), reached through ctypes
from the layouts in its public header. Everywhere else, and wherever
Accelerate refuses a matrix, it is SuperLU's LU through scipy, told the
matrix is symmetric so it orders and pivots for that.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp

#: set to 'superlu' to keep Accelerate out, which is how the tests run
#: both paths on one machine
BACKEND_VARIABLE = 'VISUALDYNAMICS_SPARSE_SOLVER'


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
                # matrix, a zero row, a rank-one block all 'ok'), where
                # Cholesky succeeding proves the matrix definite. One
                # probe solve tells, and a refusal here sends the matrix
                # on to SuperLU, whose 'exactly singular' the modal solve
                # already turns into its own message
                if kind == _LDLT and not np.isfinite(
                        self.solve(np.ones(n))).all():
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


def factor_symmetric(matrix: Any) -> AccelerateFactor | SuperLUFactor:
    """The matrix factored, by Accelerate where it is and takes the
    matrix, else by SuperLU; both answer `solve(rhs)` and `shape`.

    Parameters
    ----------
    matrix : sparse matrix
        Square and symmetric.

    Returns
    -------
    AccelerateFactor or SuperLUFactor
    """
    if os.environ.get(BACKEND_VARIABLE, '').lower() != 'superlu':
        try:
            return AccelerateFactor(matrix)
        except ValueError:
            pass
    return SuperLUFactor(matrix)
