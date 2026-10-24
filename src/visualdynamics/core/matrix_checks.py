"""Two checks on the matrices a multi-input test is built from, made at
every frequency line before anything is inverted or played.

A multi-input test stands on two matrices per line. The plant matrix
`H`, one row per control channel and one column per drive, is what the
controller inverts to turn the control error into a drive correction.
The cross-spectral density matrix `S`, one row and one column per
channel, is what a specification asks the control channels to hold:
their autospectra on the diagonal and, off it, how each pair moves
together. Each can be wrong in a way no single record shows.

Conditioning
------------
The singular values of `H` at a line are its gains in the combinations
of drives it responds to most and least, largest first. Their ratio is
the condition number,

    kappa = sigma_max / sigma_min,

dimensionless. It bounds how much a relative error in the measured
plant can grow in its inverse: an estimate good to one percent, at a
condition number of 100, can give a drive wrong by up to a hundred
percent. Two drives that reach the control channels in nearly the same
proportions, such as two shakers under one stiff fixture below its first
flexible mode, make two nearly equal columns, a small `sigma_min` and a
large condition number, and the inverse that separates them is made of
noise. `CONDITION_LIMIT` is the threshold a line is flagged over, a
choice and not a law; a controller holds a setting of its own for it.

The effective rank at that threshold is the number of singular values
at least `sigma_max / limit`, so a line is flagged exactly where the
effective rank falls under the full rank. The same reading is made of
the cross-spectral matrix of the drives, `Gff`, whose singular values
are its eigenvalues: drives that are nearly copies of one another make
it nearly singular, and an FRF estimate from them inverts it.

`read_conditioning` reads an FRF (several references) or a CSD and
returns a `Conditioning`. The matrix is in one unit or the reading is
refused: a condition number of a matrix whose rows are in different
units changes when one row's unit does, so it would describe the units
and not the plant.

Realizability
-------------
The cross-spectral matrix of any set of signals is Hermitian (the term
for channel j against i is the conjugate of i against j) and positive
semi-definite (no combination of the channels has negative power) at
every line. A specification that is not cannot be produced by any
drive, and a controller asked for one fails at those lines, however
long it runs. Each entry can look right on its own: three channels with
every coherence 1 and phases of 0, 0 and 180 degrees ask for 1 in step
with 2, 2 in step with 3, and 1 opposite 3, which no motion does.

The check is made on the coherence matrix, the cross-spectral matrix
scaled by the root of its diagonal,

    C_ij = S_ij / sqrt(S_ii S_jj),

which has ones on its diagonal, is free of units, and is positive
semi-definite exactly where `S` is (a scaling by a positive diagonal
does not change the signs of the eigenvalues). Its eigenvalues sum to
the number of channels. A line is realizable where the smallest is no
lower than minus `tolerance` times the largest and the matrix is
Hermitian to the same tolerance. The default `REALIZABLE_TOLERANCE`
passes a matrix written to six significant figures and refuses any
departure larger than that. The rank at a line counts the eigenvalues
over `tolerance` times the largest: one for fully coherent channels,
the number of channels for independent ones.

A matrix that fails is repaired to the nearest one that is realizable,
in the Frobenius norm of the coherence matrix. Keeping the diagonal
(the default), the autospectra stay exactly as written and only the
cross terms move: the nearest unit-diagonal positive semi-definite
matrix, found by alternating projections with Dykstra's correction.
Not keeping it, the negative eigenvalues are set to zero and the matrix
rebuilt, which also moves the autospectra. Lines that pass are left as
they are, bit for bit. The reading reports the largest change in
coherence and in phase the repair made, and where.

`read_realizability` reads a CSD and returns a `Realizability`;
`SpecificationDraft` refuses a sheet whose coherence and phase entries
cannot hold together, by the same check.

References
----------
The singular value decomposition and the condition number are Golub
and Van Loan's; the nearest positive semi-definite matrix is Higham's,
in two papers; the properties of a cross-spectral density matrix are
Bendat and Piersol's.

1. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*, 4th
   ed. Johns Hopkins University Press. The singular value
   decomposition, the condition number, and the bound it puts on the
   error of a solution.
2. Higham, N. J. (1988). "Computing a nearest symmetric positive
   semidefinite matrix." *Linear Algebra and its Applications*, 103,
   103-118.
   [doi:10.1016/0024-3795(88)90223-6](https://doi.org/10.1016/0024-3795(88)90223-6)
   The nearest positive semi-definite matrix in the Frobenius norm, by
   setting the negative eigenvalues to zero.
3. Higham, N. J. (2002). "Computing the nearest correlation matrix: a
   problem from finance." *IMA Journal of Numerical Analysis*, 22(3),
   329-343.
   [doi:10.1093/imanum/22.3.329](https://doi.org/10.1093/imanum/22.3.329)
   The alternating projections, with Dykstra's correction, that find
   the nearest unit-diagonal positive semi-definite matrix.
4. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The cross-spectral density
   matrix of several records, Hermitian and non-negative definite, and
   the ordinary and multiple coherence read from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ..units import UNKNOWN
from .progress import Ticker

#: the condition number a line is flagged over. An FRF estimate good to
#: one percent, inverted at a condition number of 100, can give a drive
#: wrong by up to a hundred percent (the error bound the condition
#: number is), so past it the inverse says more about the noise than
#: the plant. A choice, set per reading
CONDITION_LIMIT = 100.0

#: how far below zero the smallest eigenvalue of a coherence matrix may
#: lie, as a fraction of the largest, and how far from Hermitian it may
#: be, before a line is unrealizable; also the threshold the rank is
#: counted over. A matrix written to six significant figures passes
REALIZABLE_TOLERANCE = 1e-6

#: the alternating projections of the repair: how many at most, and the
#: change between two passes, in coherence-matrix entries, a line stops
#: at: far finer than a specification is written to, and the scaling
#: that ends the repair makes it exactly realizable whatever is left
REPAIR_ITERATIONS = 2000
REPAIR_CONVERGED = 1e-9


# ---- the matrices, assembled from records ---------------------------------

def _labels(keys: list[tuple[str, str]]) -> list[str]:
    """A label per (DOF, quantity) channel: the DOF, with the quantity
    added only where a DOF carries two (an accelerometer and a force
    gauge at one drive point)."""
    dofs = [dof for dof, _quantity in keys]
    return [dof if dofs.count(dof) == 1 else f'{dof} {quantity}'
            for dof, quantity in keys]


def frf_matrix(frf: Any, records: Any = None, values: Any = None
               ) -> tuple[list[tuple[str, str]], list[tuple[str, str]],
                          np.ndarray, np.ndarray]:
    """The records of an FRF assembled into the response-by-reference
    matrix they belong to, at every line.

    A row is a response channel and a column a reference channel, each
    named (DOF, quantity) from the record's dimension, so a drive
    point's accelerometer and force gauge are not one channel. `records`
    picks the records (all of them when omitted) and `values`, when
    given, are their values in that order (the display units, for a
    plot); otherwise the stored SI values are used. Returns (rows,
    columns, matrix of lines x rows x columns, filled), where `filled`
    says which entries a record supplied; an entry no record supplied
    is zero in the matrix. The CMIF and the conditioning reading both
    assemble their matrix here.
    """
    from .data import channel_quantities

    wanted = (list(range(frf.num_records)) if records is None
              else [int(i) for i in records])
    values = (np.asarray(frf.ordinate)[wanted] if values is None
              else np.asarray(values))
    rows: list[tuple[str, str]] = []
    columns: list[tuple[str, str]] = []
    keys = []
    for i in wanted:
        out, into = channel_quantities(frf.known_dim(i))
        row = (str(frf.response_dof[i]), out)
        column = (str(frf.reference_dof[i]), into)
        if row not in rows:
            rows.append(row)
        if column not in columns:
            columns.append(column)
        keys.append((rows.index(row), columns.index(column)))
    matrix = np.zeros((len(frf.abscissa), len(rows), len(columns)),
                      dtype=np.complex128)
    filled = np.zeros((len(rows), len(columns)), dtype=bool)
    for position, (r, c) in enumerate(keys):
        matrix[:, r, c] = values[position]
        filled[r, c] = True
    return rows, columns, matrix, filled


def spectral_matrix(psd: Any, records: Any = None,
                    quantity: str | None = None
                    ) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """The records of a CSD assembled into its square matrix, at every
    line.

    A channel is (DOF, quantity), read from each record's dimension: a
    cross term 'acceleration*force/frequency' runs from an acceleration
    channel to a force channel. `records` picks the records (all when
    omitted) and `quantity` keeps only the records whose two channels
    are both of it. Returns (channels, matrix of lines x n x n, filled),
    with an entry no record supplied zero and not filled. The principal
    shapes and the realizability and conditioning readings all assemble
    their matrix here.
    """
    from .data import channel_quantities

    wanted = (list(range(psd.num_records)) if records is None
              else [int(i) for i in records])
    references = (psd.reference_dof if psd.reference_dof is not None
                  else psd.response_dof)
    entries = []
    for i in wanted:
        out, into = channel_quantities(psd.known_dim(i))
        if quantity is not None and (out, into) != (quantity, quantity):
            continue
        entries.append((i, (str(psd.response_dof[i]), out),
                        (str(references[i]), into)))
    channels: list[tuple[str, str]] = []
    for _i, row, _column in entries:
        if row not in channels:
            channels.append(row)
    for _i, _row, column in entries:
        if column not in channels:
            channels.append(column)
    index = {key: k for k, key in enumerate(channels)}
    n = len(channels)
    ordinate = np.asarray(psd.ordinate)
    matrix = np.zeros((len(psd.abscissa), n, n), dtype=np.complex128)
    filled = np.zeros((n, n), dtype=bool)
    for i, row, column in entries:
        matrix[:, index[row], index[column]] = ordinate[i]
        filled[index[row], index[column]] = True
    return channels, matrix, filled


def _completed(matrix: np.ndarray, filled: np.ndarray, labels: list[str]
               ) -> tuple[np.ndarray, list[str]]:
    """A square cross-spectral matrix with each entry that only its
    mirror supplied filled in as the mirror's conjugate. Refused, naming
    a missing pair, when neither half of a pair or a diagonal is there.
    Returns the matrix and a note when anything was filled in."""
    n = filled.shape[0]
    missing = [(i, j) for i in range(n) for j in range(i, n)
               if not filled[i, j] and not filled[j, i]]
    if missing:
        i, j = missing[0]
        what = (f'{labels[i]} has no autospectrum' if i == j else
                f'{labels[i]} and {labels[j]} have no cross term')
        more = f' ({len(missing) - 1} more missing)' if len(missing) > 1 else ''
        raise ValueError(
            f'the matrix is incomplete: {what}{more}. It is judged whole; '
            'select the records of channels whose terms are all there')
    mirrored = ~filled & filled.T
    notes = []
    if mirrored.any():
        out = matrix.copy()
        rows, cols = np.nonzero(mirrored)
        out[:, rows, cols] = np.conj(matrix[:, cols, rows])
        count = int(mirrored.sum())
        notes.append(f'{count} term{"s" * (count != 1)} held only by '
                     f'{"their mirrors" if count != 1 else "its mirror"} '
                     'taken as the conjugate')
        return out, notes
    return matrix, notes


def _not_finite_note(matrix: np.ndarray, finite: np.ndarray, rows: Any,
                     columns: Any) -> str:
    """The note for lines holding a value that is not a finite number,
    naming the first entry that holds one, so the reader knows which
    record to look at rather than only that some line was passed over."""
    bad = ~np.isfinite(matrix[~finite]).all(axis=0)
    r, c = (int(k) for k in np.argwhere(bad)[0])
    where = (str(rows[r]) if r == c and list(rows) == list(columns)
             else f'{rows[r]} against {columns[c]}')
    others = int(bad.sum()) - 1
    more = f' and {others} more term{"s" * (others != 1)}' if others else ''
    count = int((~finite).sum())
    return (f'{count} line{"s" * (count != 1)} hold{"s" * (count == 1)} a '
            f'value that is not a finite number, in {where}{more}, and '
            f'{"are" if count != 1 else "is"} not judged')


def flagged_runs(frequency: np.ndarray, flags: np.ndarray
                 ) -> list[tuple[float, float]]:
    """(first, last) frequency of each run of consecutive flagged
    lines."""
    out = []
    k = 0
    flags = np.asarray(flags, dtype=bool)
    while k < len(flags):
        if flags[k]:
            start = k
            while k + 1 < len(flags) and flags[k + 1]:
                k += 1
            out.append((float(frequency[start]), float(frequency[k])))
        k += 1
    return out


def _span_text(runs: list[tuple[float, float]], most: int = 3) -> str:
    """Runs of lines as words: '20 to 85 Hz, 400 Hz and 2 more bands'."""
    parts = [f'{low:.4g} Hz' if low == high else
             f'{low:.4g} to {high:.4g} Hz' for low, high in runs[:most]]
    if len(runs) > most:
        parts.append(f'{len(runs) - most} more band'
                     f'{"s" * (len(runs) - most != 1)}')
    return (', '.join(parts[:-1]) + ' and ' + parts[-1]
            if len(parts) > 1 else parts[0])


# ---- conditioning -----------------------------------------------------------

@dataclass(frozen=True)
class ConditioningPoint:
    """The conditioning of the matrix at one line: what
    `Conditioning.at` returns.

    `singular` holds the singular values, largest first, in the
    reading's units (SI); `condition` is the largest over the smallest,
    infinite where the smallest is zero and not a number where the line
    was not judged; `rank` is the effective rank at `limit`, of a full
    rank of `size`.
    """

    frequency: float
    singular: tuple[float, ...]
    condition: float
    rank: int
    size: int
    limit: float

    @property
    def ill(self) -> bool:
        """Whether the condition number is over the limit, or the
        smallest singular value is zero."""
        return bool(not np.isnan(self.condition)
                    and not self.condition <= self.limit)

    def describe(self, unit: str = '', scale: float = 1.0) -> str:
        """One line for a status bar: the condition number against the
        limit, the effective rank, and the singular values in `unit`
        (`scale` converts the reading's SI into it)."""
        if np.isnan(self.condition):
            return f'{self.frequency:.4g} Hz: not judged'
        unit = f' {unit}' if unit else ''
        shown = ', '.join(f'{value * scale:.4g}' for value in self.singular)
        verdict = 'over' if self.ill else 'within'
        return (f'{self.frequency:.4g} Hz: condition number '
                f'{self.condition:.4g}, {verdict} {self.limit:g}; rank '
                f'{self.rank} of {self.size}; singular values {shown}{unit}')


@dataclass(frozen=True, eq=False)
class Conditioning:
    """The singular values and the condition number of a matrix at every
    line: what `read_conditioning` returns.

    `frequency` holds the lines in Hz and `singular` the singular values
    (one row per value, largest first; not a number on a line that was
    not judged), in the matrix's units (SI, `dimension`). `rows` and
    `columns` name the matrix's channels, responses and references for
    an FRF, the channels twice for a CSD; `kind` is 'FRF' or 'CSD'.
    `limit` is the condition number lines are flagged over. `notes` says
    what the reading should be read with.
    """

    frequency: np.ndarray
    singular: np.ndarray
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    limit: float = CONDITION_LIMIT
    dimension: str = UNKNOWN
    kind: str = 'FRF'
    notes: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        """The full rank, the smaller of the rows and the columns."""
        return int(self.singular.shape[0])

    @property
    def judged(self) -> np.ndarray:
        """The lines that were judged: every entry a finite number and
        the matrix not all zero."""
        largest = self.singular[0]
        return np.isfinite(largest) & (largest > 0.0)

    @property
    def condition(self) -> np.ndarray:
        """The condition number at every line, the largest singular value
        over the smallest: infinite where the smallest is zero, not a
        number where the line was not judged."""
        largest, smallest = self.singular[0], self.singular[-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(smallest > 0.0, largest / smallest, np.inf)
        return np.where(self.judged, ratio, np.nan)

    @property
    def rank(self) -> np.ndarray:
        """The effective rank at every line: how many singular values are
        at least the largest over `limit`. Zero on a line not judged."""
        largest = self.singular[0]
        with np.errstate(invalid='ignore'):
            counted = self.singular >= largest / self.limit
        return np.where(self.judged, counted.sum(axis=0), 0)

    @property
    def ill(self) -> np.ndarray:
        """The lines flagged: judged, with a condition number over the
        limit (where the effective rank is under the full rank)."""
        return self.judged & ~(self.condition <= self.limit)

    def bands(self) -> list[tuple[float, float]]:
        """The runs of flagged lines, as (first, last) frequencies."""
        return flagged_runs(self.frequency, self.ill)

    def line(self, frequency: float) -> int:
        """The index of the line nearest `frequency`."""
        return int(np.argmin(np.abs(np.asarray(self.frequency, dtype=float)
                                    - float(frequency))))

    def at(self, frequency: float) -> ConditioningPoint:
        """The reading at the line nearest `frequency`."""
        k = self.line(frequency)
        return ConditioningPoint(
            frequency=float(self.frequency[k]),
            singular=tuple(float(v) for v in self.singular[:, k]),
            condition=float(self.condition[k]), rank=int(self.rank[k]),
            size=self.size, limit=float(self.limit))

    @property
    def title(self) -> str:
        """What the matrix is, in words."""
        if self.kind == 'CSD':
            return f'CSD of {len(self.rows)} channels'
        return (f'FRF from {len(self.columns)} references to '
                f'{len(self.rows)} responses')

    def describe(self) -> str:
        """The verdict in a sentence: where the condition number is over
        the limit, or that it is nowhere, or that nothing was judged."""
        judged = int(self.judged.sum())
        if not judged:
            return f'{self.title}: no line could be judged'
        runs = self.bands()
        worst = float(np.nanmax(np.where(self.judged, self.condition,
                                         np.nan)))
        if not runs:
            return (f'{self.title}: condition number within '
                    f'{self.limit:g} at all {judged} lines judged '
                    f'(largest {worst:.4g})')
        flagged = int(self.ill.sum())
        return (f'{self.title}: condition number over {self.limit:g} at '
                f'{flagged} of {judged} lines, {_span_text(runs)}')


def condition_matrix(frequency: Any, matrix: Any, rows: Any, columns: Any,
                     *, limit: float = CONDITION_LIMIT,
                     dimension: str = UNKNOWN, kind: str = 'FRF',
                     notes: Any = (),
                     progress: Callable[[int, int], None] | None = None
                     ) -> Conditioning:
    """The singular values of a matrix at every line, as a
    `Conditioning`.

    `matrix` is lines x rows x columns, complex, one unit throughout.
    A line holding any value that is not a finite number is not judged
    (its singular values are not a number) and is noted with the first
    entry that holds one; so is a line where the matrix is all zero, and
    a column (or, for an FRF, a row) that is zero at every line is named.
    `limit` is the condition number lines are flagged over, more than
    one. `progress`, when given, is told ``(0, core.progress.BUSY)``
    before the decomposition, which cannot count itself.
    """
    limit = float(limit)
    if not limit > 1.0:
        raise ValueError('a condition-number limit is more than 1; every '
                         'matrix has a condition number of at least 1')
    f = np.asarray(frequency, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 3 or matrix.shape[0] != len(f):
        raise ValueError('the matrix is one rows-by-columns matrix per '
                         'line')
    size = min(matrix.shape[1], matrix.shape[2])
    if size == 0:
        raise ValueError('the matrix is empty; there is nothing to read')
    finite = np.isfinite(matrix).all(axis=(1, 2))
    singular = np.full((size, len(f)), np.nan)
    if finite.any():
        Ticker(progress).busy()
        singular[:, finite] = np.linalg.svd(matrix[finite],
                                            compute_uv=False).T
    notes = list(notes)
    if not finite.all():
        notes.append(_not_finite_note(matrix, finite, rows, columns))
    zero = finite & ~(singular[0] > 0.0)
    if zero.any():
        notes.append(f'{int(zero.sum())} lines are zero throughout and '
                     'are not judged')
    dead = [str(columns[c]) for c in range(matrix.shape[2])
            if finite.any() and not np.any(matrix[finite][:, :, c])]
    if dead and finite.any():
        notes.append(f'{", ".join(dead)} reach{"es" * (len(dead) == 1)} '
                     'no row at any line: the matrix is singular wherever '
                     'it is judged')
    # a response that nothing reaches is a dead channel too; in a CSD it
    # is its own column, named above
    silent = [str(rows[r]) for r in range(matrix.shape[1])
              if kind == 'FRF' and finite.any()
              and not np.any(matrix[finite][:, r, :])]
    if silent:
        notes.append(f'{", ".join(silent)} '
                     f'{"is" if len(silent) == 1 else "are"} reached by no '
                     'reference at any line: a dead channel gives the '
                     'matrix a zero row')
    if kind == 'FRF' and matrix.shape[2] > matrix.shape[1]:
        notes.append(f'{matrix.shape[2]} references drive '
                     f'{matrix.shape[1]} responses: the references can '
                     'make one motion in more than one way, and the '
                     f'condition number is of the {size} gains there are')
    return Conditioning(frequency=f, singular=singular,
                        rows=tuple(str(r) for r in rows),
                        columns=tuple(str(c) for c in columns),
                        limit=limit, dimension=str(dimension), kind=kind,
                        notes=tuple(notes))


def _one_unit(data: Any, wanted: list[int]) -> tuple[str, list[str]]:
    """The one dimension the picked records share, and a note when it is
    undeclared. Refused when they are in different units, or some are
    declared and some are not."""
    dims = list(dict.fromkeys(str(data.ordinate_dim[i]) for i in wanted))
    if len(dims) > 1:
        shown = ' and '.join(dims[:2])
        raise ValueError(
            f'the records are in different units ({shown}): the singular '
            'values of a matrix whose rows are in different units change '
            'when one unit does, so they would describe the units and not '
            'the plant. Select records of one quantity')
    if dims[0] == UNKNOWN:
        return UNKNOWN, [('the units are not declared: the singular '
                          'values are in whatever units the records hold, '
                          'taken to be one unit throughout')]
    return dims[0], []


def read_conditioning(data: Any, records: Any = None, *,
                      limit: float = CONDITION_LIMIT,
                      progress: Callable[[int, int], None] | None = None
                      ) -> Conditioning:
    """The singular values and condition number of an FRF matrix, or of
    a cross-spectral matrix, at every line.

    An `Frf` gives the plant matrix, responses by references; it needs
    two references or more, since a condition number compares drives.
    A `Psd` with cross terms gives the square cross-spectral matrix of
    its channels, such as the drives' `Gff`. `records` picks the records
    (all of them when omitted): the matrix they make must be complete
    (a CSD term held only by its mirror is taken as the mirror's
    conjugate) and in one unit. `limit` is the condition number lines
    are flagged over. `progress` is passed to `condition_matrix`.

    Raises TypeError for an object of neither kind, and ValueError for
    an empty selection, records in different units, an incomplete
    matrix, an FRF with one reference, or a CSD of one channel.
    """
    from .data import Frf, Psd

    wanted = (list(range(data.num_records)) if records is None
              else [int(i) for i in records])
    if not wanted:
        raise ValueError('no records are selected; there is no matrix to '
                         'read')
    if isinstance(data, Frf):
        if data.reference_dof is None:
            raise ValueError('the FRF names no references')
        dimension, notes = _one_unit(data, wanted)
        rows, columns, matrix, filled = frf_matrix(data, wanted)
        if len(columns) < 2:
            raise ValueError(
                'conditioning compares references, and these records have '
                f'one ({_labels(columns)[0]}); select an FRF from two '
                'references or more')
        if not filled.all():
            r, c = np.argwhere(~filled)[0]
            raise ValueError(
                f'the matrix is incomplete: {_labels(rows)[r]} has no FRF '
                f'to {_labels(columns)[c]}. Select every response against '
                'every reference')
        return condition_matrix(data.abscissa, matrix, _labels(rows),
                                _labels(columns), limit=limit,
                                dimension=dimension, kind='FRF',
                                notes=notes, progress=progress)
    if isinstance(data, Psd):
        if data.reference_dof is None:
            raise ValueError('an autospectrum set has no cross terms; the '
                             'matrix needs them')
        channels, matrix, filled = spectral_matrix(data, wanted)
        quantities = list(dict.fromkeys(q for _dof, q in channels))
        if len(quantities) > 1:
            raise ValueError(
                f'the channels measure different quantities '
                f'({" and ".join(quantities[:2])}): the singular values of '
                'a matrix in different units change when one unit does. '
                'Select the records of one quantity')
        if len(channels) < 2:
            raise ValueError('conditioning compares channels, and these '
                             'records have one')
        labels = _labels(channels)
        matrix, notes = _completed(matrix, filled, labels)
        dimension, unit_notes = _one_unit(data, wanted)
        return condition_matrix(data.abscissa, matrix, labels, labels,
                                limit=limit, dimension=dimension,
                                kind='CSD', notes=unit_notes + notes,
                                progress=progress)
    raise TypeError(f'a {type(data).__name__} is neither an FRF nor a CSD; '
                    'there is no matrix to condition')


# ---- realizability ----------------------------------------------------------

def _normalized(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coherence matrix, scale): the cross-spectral matrix divided by
    the root of its diagonal on both sides, and that root.

    A channel with no power at a line (or a negative autospectrum) is
    scaled by the root of the line's largest autospectrum instead, so
    its cross terms stay in view at the line's own scale: a zero
    autospectrum with a cross term that is not zero then gives a
    negative eigenvalue, as it should, where dividing by zero would
    hide it.
    """
    diagonal = np.real(np.diagonal(matrix, axis1=1, axis2=2))
    largest = np.max(np.where(np.isfinite(diagonal), diagonal, 0.0), axis=1)
    fallback = np.where(largest > 0.0, np.sqrt(np.maximum(largest, 0.0)),
                        1.0)
    with np.errstate(invalid='ignore'):
        scale = np.where(diagonal > 0.0, np.sqrt(np.abs(diagonal)),
                         fallback[:, None])
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return matrix / (scale[:, :, None] * scale[:, None, :]), scale


def _psd_part(matrix: np.ndarray) -> np.ndarray:
    """Each Hermitian matrix with its negative eigenvalues set to zero:
    the nearest positive semi-definite matrix in the Frobenius norm
    (Higham, 1988)."""
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 0.0)[:, None, :]) @ np.conj(
        np.swapaxes(vectors, -2, -1))


def repair_coherence(matrix: Any, keep_diagonal: bool = True,
                     progress: Callable[[int, int], None] | None = None
                     ) -> np.ndarray:
    """The nearest realizable matrix to each Hermitian coherence matrix,
    in the Frobenius norm.

    `matrix` is lines x n x n. With `keep_diagonal`, the diagonal is
    held where it is (zero stays zero, a negative entry goes to zero)
    and the nearest positive semi-definite matrix with that diagonal is
    found by alternating projections with Dykstra's correction (Higham,
    2002), then scaled onto the diagonal exactly, which keeps it positive
    semi-definite. A line still moving after `REPAIR_ITERATIONS` passes
    is realizable all the same, if not quite the nearest. Without it,
    the negative eigenvalues are set to zero (Higham, 1988) and the
    diagonal moves too. `progress`, when given, is told ``(done,
    total)`` pass by pass, out of `REPAIR_ITERATIONS`; whatever it
    raises stops the repair.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    hermitian = 0.5 * (matrix + np.conj(np.swapaxes(matrix, -2, -1)))
    if not keep_diagonal:
        return _psd_part(hermitian)
    target = np.maximum(np.real(np.diagonal(hermitian, axis1=1, axis2=2)),
                        0.0)
    n = hermitian.shape[-1]
    at = np.arange(n)
    y = hermitian.copy()
    y[:, at, at] = target
    correction = np.zeros_like(y)
    x = y.copy()
    # each line stops when it has converged, so a target of thousands
    # of lines pays for its slowest lines alone
    active = np.arange(len(y))
    ticker = Ticker(progress)
    ticker.add(REPAIR_ITERATIONS)
    for _ in range(REPAIR_ITERATIONS):
        if not len(active):
            break
        ticker.tick()
        r = y[active] - correction[active]
        projected = _psd_part(r)
        correction[active] = projected - r
        x[active] = projected
        moved = projected.copy()
        moved[:, at, at] = target[active]
        change = np.max(np.abs(moved - y[active]), axis=(1, 2))
        y[active] = moved
        active = active[change >= REPAIR_CONVERGED]
    ticker.tick(ticker.total - ticker.done)
    # x is positive semi-definite and its diagonal is within the
    # convergence of the target: a scaling by a positive diagonal puts
    # it on the target exactly and keeps it positive semi-definite. A
    # channel held at zero has its row and column zero
    diagonal = np.real(np.diagonal(x, axis1=1, axis2=2))
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where((target > 0.0) & (diagonal > 0.0),
                          np.sqrt(target / diagonal), 0.0)
    out = x * factor[:, :, None] * factor[:, None, :]
    out[:, at, at] = target
    return out


@dataclass(frozen=True)
class RealizabilityPoint:
    """The realizability of the matrix at one line: what
    `Realizability.at` returns. `eigenvalues` are those of the coherence
    matrix, smallest first; `asymmetry` is its largest departure from
    Hermitian; `realizable` says whether the line passes."""

    frequency: float
    eigenvalues: tuple[float, ...]
    asymmetry: float
    rank: int
    realizable: bool
    tolerance: float

    def describe(self) -> str:
        """One line for a status bar."""
        if not self.eigenvalues or np.isnan(self.eigenvalues[0]):
            return f'{self.frequency:.4g} Hz: not judged'
        verdict = 'realizable' if self.realizable else 'not realizable'
        return (f'{self.frequency:.4g} Hz: {verdict}; smallest eigenvalue '
                f'{self.eigenvalues[0]:.4g} against largest '
                f'{self.eigenvalues[-1]:.4g}, rank {self.rank} of '
                f'{len(self.eigenvalues)}')


@dataclass(frozen=True, eq=False)
class Realizability:
    """Whether a cross-spectral matrix can be produced by any set of
    signals, line by line, and the nearest one that can: what
    `read_realizability` returns.

    `frequency` holds the lines in Hz. `coherence` is the coherence
    matrix at every line (lines x n x n, the cross-spectral matrix
    scaled by the root of its diagonal) and `scale` that root (lines x
    n), so the cross-spectral matrix is recovered as `scale_i scale_j
    coherence_ij`. `eigenvalues` (n x lines, smallest first) are those of
    the coherence matrix's Hermitian part; `asymmetry` is its largest
    departure from Hermitian at each line; `negative` marks lines where
    an autospectrum is below zero. `repaired` is the coherence matrix
    after the repair, the same as `coherence` on every line that passes.
    `channels` names the rows; `tolerance` is the check's.
    """

    frequency: np.ndarray
    coherence: np.ndarray
    scale: np.ndarray
    eigenvalues: np.ndarray
    asymmetry: np.ndarray
    negative: np.ndarray
    repaired: np.ndarray
    channels: tuple[str, ...]
    tolerance: float = REALIZABLE_TOLERANCE
    keep_diagonal: bool = True
    notes: tuple[str, ...] = ()

    @property
    def judged(self) -> np.ndarray:
        """The lines judged: every entry a finite number."""
        return np.isfinite(self.eigenvalues).all(axis=0)

    @property
    def smallest(self) -> np.ndarray:
        """The smallest eigenvalue at every line."""
        return self.eigenvalues[0]

    @property
    def largest(self) -> np.ndarray:
        """The largest eigenvalue at every line."""
        return self.eigenvalues[-1]

    @property
    def failing(self) -> np.ndarray:
        """The lines judged and not realizable: an eigenvalue below minus
        the tolerance times the largest, a departure from Hermitian over
        the tolerance, or a negative autospectrum."""
        with np.errstate(invalid='ignore'):
            indefinite = self.smallest < -self.tolerance * np.maximum(
                self.largest, 0.0)
            return self.judged & (indefinite | (self.asymmetry
                                                > self.tolerance)
                                  | self.negative)

    @property
    def realizable(self) -> bool:
        """Whether every line was judged realizable. False when no line
        could be judged: a check that judged nothing has not passed."""
        return bool(self.judged.any() and not self.failing.any())

    @property
    def rank(self) -> np.ndarray:
        """The rank at every line: the eigenvalues over the tolerance
        times the largest. Zero on a line not judged."""
        with np.errstate(invalid='ignore'):
            counted = self.eigenvalues > self.tolerance * self.largest
        return np.where(self.judged, counted.sum(axis=0), 0)

    def bands(self) -> list[tuple[float, float]]:
        """The runs of lines that are not realizable, as (first, last)
        frequencies."""
        return flagged_runs(self.frequency, self.failing)

    def matrix(self, repaired: bool = False) -> np.ndarray:
        """The cross-spectral matrix at every line (lines x n x n), as
        read or, with `repaired`, after the repair."""
        c = self.repaired if repaired else self.coherence
        return c * self.scale[:, :, None] * self.scale[:, None, :]

    def pair(self, i: int, j: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(coherence before, after, phase before, after) of channels `i`
        and `j` at every line, the phase in degrees: what the repair did
        to that pair. Coherence is `data.ordinary_coherence` of the
        matrix before and after."""
        from .data import ordinary_coherence

        out = []
        for c in (self.coherence, self.repaired):
            out.append(ordinary_coherence(c[:, i, i], c[:, j, j], c[:, i, j]))
        before, after = self.coherence[:, i, j], self.repaired[:, i, j]
        return (out[0], out[1], np.degrees(np.angle(before)),
                np.degrees(np.angle(after)))

    def worst(self) -> tuple[int, int, int] | None:
        """(i, j, line) of the cross term the repair moved most, as a
        complex coherence entry, or None when nothing moved."""
        return self._worst

    # worked out once and kept: the reading is read again at every move
    # of the window's cursor, and on a large target these are a tenth
    # of a second of the window's time each
    @cached_property
    def _worst(self) -> tuple[int, int, int] | None:
        n = len(self.channels)
        if n < 2 or not self.failing.any():
            return None
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        change = np.abs(self.repaired - self.coherence)
        change = np.where(upper[None] & self.failing[:, None, None],
                          change, -1.0)
        line, i, j = np.unravel_index(int(np.argmax(change)), change.shape)
        if change[line, i, j] <= 0.0:
            return None
        return int(i), int(j), int(line)

    def changes(self) -> tuple[float, float]:
        """(coherence, phase in degrees): the largest change the repair
        made to any pair's coherence and to any phase, over lines where
        the pair's coherence is not zero after (a phase is undefined
        without it)."""
        return self._changes

    @cached_property
    def _changes(self) -> tuple[float, float]:
        n = len(self.channels)
        largest_coherence = largest_phase = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                before, after, phase_before, phase_after = self.pair(i, j)
                moved = self.failing & np.isfinite(before) & np.isfinite(after)
                if not moved.any():
                    continue
                largest_coherence = max(largest_coherence, float(
                    np.max(np.abs(after - before)[moved])))
                held = moved & (after > 0.0) & (before > 0.0)
                if held.any():
                    turn = (phase_after - phase_before + 180.0) % 360.0 - 180.0
                    largest_phase = max(largest_phase,
                                        float(np.max(np.abs(turn[held]))))
        return largest_coherence, largest_phase

    def line(self, frequency: float) -> int:
        """The index of the line nearest `frequency`."""
        return int(np.argmin(np.abs(np.asarray(self.frequency, dtype=float)
                                    - float(frequency))))

    def at(self, frequency: float) -> RealizabilityPoint:
        """The reading at the line nearest `frequency`."""
        k = self.line(frequency)
        return RealizabilityPoint(
            frequency=float(self.frequency[k]),
            eigenvalues=tuple(float(v) for v in self.eigenvalues[:, k]),
            asymmetry=float(self.asymmetry[k]), rank=int(self.rank[k]),
            realizable=bool(self.judged[k] and not self.failing[k]),
            tolerance=float(self.tolerance))

    def describe(self) -> str:
        """The verdict in a sentence: realizable, or the bands that are
        not and the pair the repair moves most, or that nothing was
        judged."""
        judged = int(self.judged.sum())
        n = len(self.channels)
        if not judged:
            return f'{n} channels: no line could be judged'
        if self.realizable:
            return f'{n} channels: realizable at all {judged} lines'
        flagged = int(self.failing.sum())
        text = (f'{n} channels: not realizable at {flagged} of {judged} '
                f'lines, {_span_text(self.bands())}')
        worst = self.worst()
        if worst is not None:
            i, j, line = worst
            text += (f'; the repair moves {self.channels[i]} and '
                     f'{self.channels[j]} most, at '
                     f'{self.frequency[line]:.4g} Hz')
        return text


def check_spectral_matrix(frequency: Any, matrix: Any, channels: Any, *,
                          tolerance: float = REALIZABLE_TOLERANCE,
                          keep_diagonal: bool = True,
                          notes: Any = (),
                          progress: Callable[[int, int], None] | None = None
                          ) -> Realizability:
    """Whether a cross-spectral matrix is realizable at every line, and
    the nearest that is, as a `Realizability`.

    `matrix` is lines x n x n, complex, any units (the check is on the
    coherence matrix). A line holding a value that is not a finite
    number is not judged, and is noted with the first entry that holds
    one; so is a line where the matrix is zero throughout, since a
    matrix of nothing says nothing about how channels move together. A
    channel with no power and no cross term at any line judged is named.
    Lines that fail are repaired (`repair_coherence`, which `progress`
    is passed to); the rest are left exactly as they are.
    """
    tolerance = float(tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError('a realizability tolerance lies between 0 and 1')
    f = np.asarray(frequency, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.complex128)
    if (matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]
            or matrix.shape[0] != len(f)):
        raise ValueError('the cross-spectral matrix is one square matrix '
                         'per line')
    n = matrix.shape[1]
    if n == 0:
        raise ValueError('the matrix is empty; there is nothing to judge')
    finite = np.isfinite(matrix).all(axis=(1, 2))
    safe = np.where(finite[:, None, None], matrix, 0.0)
    # a line of zeros is judged by nothing: its eigenvalues are zero and
    # would read as realizable, a pass on no evidence
    zero = finite & ~np.any(safe, axis=(1, 2))
    judged = finite & ~zero
    coherence, scale = _normalized(safe)
    conjugate = np.conj(np.swapaxes(coherence, -2, -1))
    asymmetry = 0.5 * np.max(np.abs(coherence - conjugate), axis=(1, 2))
    diagonal = np.real(np.diagonal(safe, axis1=1, axis2=2))
    negative = finite & (diagonal < 0.0).any(axis=1)
    eigenvalues = np.full((n, len(f)), np.nan)
    if judged.any():
        eigenvalues[:, judged] = np.linalg.eigvalsh(
            0.5 * (coherence[judged] + conjugate[judged])).T
    coherence = np.where(finite[:, None, None], coherence, np.nan + 0j)
    reading = Realizability(
        frequency=f, coherence=coherence, scale=scale,
        eigenvalues=eigenvalues,
        asymmetry=np.where(finite, asymmetry, np.nan), negative=negative,
        repaired=coherence, channels=tuple(str(c) for c in channels),
        tolerance=tolerance, keep_diagonal=bool(keep_diagonal), notes=())
    failing = reading.failing
    repaired = coherence.copy()
    if failing.any():
        repaired[failing] = repair_coherence(coherence[failing],
                                             keep_diagonal, progress)
    notes = list(notes)
    labels = [str(c) for c in channels]
    if not finite.all():
        notes.append(_not_finite_note(matrix, finite, labels, labels))
    if zero.any():
        count = int(zero.sum())
        notes.append(f'{count} line{"s" * (count != 1)} '
                     f'{"are" if count != 1 else "is"} zero throughout and '
                     f'{"are" if count != 1 else "is"} not judged')
    if judged.any():
        live = np.any(safe[judged] != 0.0, axis=(0, 2))
        dead = [labels[k] for k in range(n) if not live[k]]
        if dead:
            notes.append(f'{", ".join(dead)} '
                         f'{"has" if len(dead) == 1 else "have"} no power '
                         'and no cross term at any line judged: a dead '
                         'channel is realizable, and says nothing about how '
                         'the others move with it')
    if negative.any():
        notes.append(f'{int(negative.sum())} lines have an autospectrum '
                     'below zero, which no signal has; the repair takes '
                     'it to zero')
    unsymmetric = finite & (asymmetry > tolerance)
    if unsymmetric.any():
        notes.append(f'{int(unsymmetric.sum())} lines are not Hermitian: '
                     'a cross term is not the conjugate of its mirror; '
                     'the repair takes their mean')
    if failing.any() and not keep_diagonal:
        notes.append('the repair was not held to the diagonal, so it '
                     'moves the autospectra too')
    return Realizability(
        frequency=f, coherence=coherence, scale=scale,
        eigenvalues=eigenvalues, asymmetry=reading.asymmetry,
        negative=negative, repaired=repaired, channels=reading.channels,
        tolerance=tolerance, keep_diagonal=bool(keep_diagonal),
        notes=tuple(notes))


def read_realizability(psd: Any, records: Any = None, *,
                       tolerance: float = REALIZABLE_TOLERANCE,
                       keep_diagonal: bool = True,
                       progress: Callable[[int, int], None] | None = None
                       ) -> Realizability:
    """Whether a CSD (a specification, or a measured cross-spectral
    matrix) is realizable at every line, and the nearest that is.

    `records` picks the records (all of them when omitted); the matrix
    they make must be complete, a term held only by its mirror being
    taken as the mirror's conjugate. The channels may be of different
    quantities, since the check is on the coherence matrix, which has
    no units; but the records are declared in units throughout or not
    at all, since a coherence of a declared autospectrum against a cross
    term in whatever units a file held is not a coherence. A
    specification written at breakpoints is judged at its breakpoints.
    `progress` is passed to the repair (`repair_coherence`).

    Raises TypeError for an object that is not a `Psd`, and ValueError
    for an empty selection, an object with no cross terms, records some
    declared in units and some not, fewer than two channels, or an
    incomplete matrix.
    """
    from .data import Psd

    if not isinstance(psd, Psd):
        raise TypeError(f'a {type(psd).__name__} is not a cross-spectral '
                        'density; there is no matrix to judge')
    wanted = (list(range(psd.num_records)) if records is None
              else [int(i) for i in records])
    if not wanted:
        raise ValueError('no records are selected; there is no matrix to '
                         'judge')
    if psd.reference_dof is None:
        raise ValueError('an autospectrum set has no cross terms, and '
                         'realizability is a statement about them')
    undeclared = [i for i in wanted if str(psd.ordinate_dim[i]) == UNKNOWN]
    if undeclared and len(undeclared) < len(wanted):
        i = undeclared[0]
        raise ValueError(
            f'{psd.response_dof[i]} against {psd.reference_dof[i]} has no '
            'declared units and other records do: a coherence of terms in '
            'SI against terms in whatever units a file held is not a '
            'coherence. Declare the units of every record, or of none')
    channels, matrix, filled = spectral_matrix(psd, wanted)
    if len(channels) < 2:
        raise ValueError('realizability is about how channels move '
                         'together, and these records have one channel')
    labels = _labels(channels)
    matrix, notes = _completed(matrix, filled, labels)
    return check_spectral_matrix(psd.abscissa, matrix, labels,
                                 tolerance=tolerance,
                                 keep_diagonal=keep_diagonal, notes=notes,
                                 progress=progress)


def made_realizable(psd: Any, records: Any = None, *,
                    tolerance: float = REALIZABLE_TOLERANCE,
                    keep_diagonal: bool = True,
                    progress: Callable[[int, int], None] | None = None
                    ) -> tuple[Any, Realizability]:
    """A copy of a CSD with the matrix the picked records make repaired
    at every line that is not realizable, and the reading that says what
    moved.

    Every record keeps its place, its DOFs and its units; only the
    values at failing lines change, the cross terms and, without
    `keep_diagonal`, the autospectra. A specification's limits follow an
    autospectrum that moved, so its bands stay where they were in
    decibels about it. `progress` is passed to the repair.
    """
    from .data import _recast, channel_quantities

    reading = read_realizability(psd, records, tolerance=tolerance,
                                 keep_diagonal=keep_diagonal,
                                 progress=progress)
    # every field copied, as reading a PSD as a specification does, so
    # nothing passes through a constructor that might drop one
    out = _recast(psd, psd.__class__)
    ordinate = np.array(psd.ordinate, dtype=np.complex128)
    failing = reading.failing
    if failing.any():
        before = reading.matrix(False)
        after = reading.matrix(True)
        wanted = (list(range(psd.num_records)) if records is None
                  else [int(i) for i in records])
        references = (psd.reference_dof if psd.reference_dof is not None
                      else psd.response_dof)
        # the reading's rows are the channels in `spectral_matrix`'s order
        channels, _matrix, _filled = spectral_matrix(psd, wanted)
        index = {key: k for k, key in enumerate(channels)}
        limits = getattr(out, 'limits', None)
        moved_limits = ({name: np.array(values, dtype=float)
                         for name, values in limits.items()}
                        if limits else None)
        for i in wanted:
            out_q, into_q = channel_quantities(psd.known_dim(i))
            r = index[(str(psd.response_dof[i]), out_q)]
            c = index[(str(references[i]), into_q)]
            values = after[failing, r, c]
            # an autospectrum is real; the repair's rounding is not
            ordinate[i, failing] = np.real(values) if r == c else values
            if r == c and moved_limits is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.real(after[failing, r, r]) / np.real(
                        before[failing, r, r])
                ratio = np.where(np.isfinite(ratio), ratio, 1.0)
                for values in moved_limits.values():
                    values[i, failing] = values[i, failing] * ratio
        if moved_limits is not None:
            out.limits = moved_limits
    if not np.iscomplexobj(psd.ordinate):
        ordinate = np.real(ordinate)
    out.ordinate = ordinate
    return out, reading


# ---- a specification sheet's entries ----------------------------------------

def _complete_groups(n: int, pairs: set[tuple[int, int]]) -> list[list[int]]:
    """The largest groups of channels in which every pair is in `pairs`
    (the maximal cliques, by Bron and Kerbosch's recursion with a pivot),
    each in channel order."""
    neighbors = {k: set() for k in range(n)}
    for i, j in pairs:
        neighbors[i].add(j)
        neighbors[j].add(i)
    found: list[list[int]] = []

    def grow(chosen: set[int], candidates: set[int], done: set[int]) -> None:
        if not candidates and not done:
            found.append(sorted(chosen))
            return
        pivot = max(candidates | done, key=lambda k: len(neighbors[k]
                                                         & candidates))
        for k in list(candidates - neighbors[pivot]):
            grow(chosen | {k}, candidates & neighbors[k],
                 done & neighbors[k])
            candidates = candidates - {k}
            done = done | {k}

    grow(set(), set(range(n)), set())
    return found


def entries_refusal(channels: Any, pairs: Any, frequencies: Any,
                    tolerance: float = REALIZABLE_TOLERANCE) -> str | None:
    """Why a specification sheet's coherence and phase entries cannot
    hold together, or None when they can.

    `pairs` is `{(i, j): (coherence, phase_degrees) or None}` for `i <
    j`, as `SpecificationDraft` holds it. The entries do not change with
    frequency and a scaling by the autospectra does not change the
    verdict, so the coherence matrix they make is judged once, for every
    line from the first breakpoint to the last. A pair not entered is
    not judged: each largest group of channels whose pairs are all
    entered is judged on its own. The reason names the channels the
    failure is in, the eigenvalue, the band, and the nearest entries
    that would hold.
    """
    channels = [str(c) for c in channels]
    stated = {(int(i), int(j)): value for (i, j), value in pairs.items()
              if value is not None}
    for group in _complete_groups(len(channels), set(stated)):
        if len(group) < 3:
            # two channels with a coherence in [0, 1] always hold
            continue
        m = len(group)
        c = np.eye(m, dtype=np.complex128)
        for a in range(m):
            for b in range(a + 1, m):
                coherence, phase = stated[(group[a], group[b])]
                c[a, b] = np.sqrt(coherence) * np.exp(1j * np.radians(phase))
                c[b, a] = np.conj(c[a, b])
        values, vectors = np.linalg.eigh(c)
        if not values[0] < -tolerance * values[-1]:
            continue
        weight = np.abs(vectors[:, 0]) ** 2
        named = [channels[group[k]] for k in range(m)
                 if weight[k] >= 0.5 / m]
        repaired = repair_coherence(c[None])[0]
        moves = []
        for a in range(m):
            for b in range(a + 1, m):
                moves.append((abs(repaired[a, b] - c[a, b]), a, b))
        moves.sort(key=lambda move: -move[0])
        shown = []
        for _size, a, b in moves[:3]:
            entry = repaired[a, b]
            # to a hundredth of a degree, so a phase the repair kept at
            # zero does not read as 1.7e-14
            phase = (round(float(np.degrees(np.angle(entry))), 2) + 0.0
                     if abs(entry) > 0.0 else 0.0)
            shown.append(f'{channels[group[a]]}–{channels[group[b]]} '
                         f'coherence {abs(entry) ** 2:.3g} at {phase:g}°')
        more = (f' and {len(moves) - 3} more' if len(moves) > 3 else '')
        who = (', '.join(named[:-1]) + ' and ' + named[-1]
               if len(named) > 1 else named[0])
        return (f'the cross terms of {who} cannot hold together: no set of '
                'signals has those coherences and phases (the matrix they '
                f'make has an eigenvalue of {values[0]:.3g} against a '
                f'largest of {values[-1]:.3g}), at every line from '
                f'{float(frequencies[0]):g} to {float(frequencies[-1]):g} '
                'Hz, since the entries do not change with frequency. The '
                f'nearest entries that hold are {"; ".join(shown)}{more}')
    return None
