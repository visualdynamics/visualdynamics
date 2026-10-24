"""Partial, reference and virtual coherence of a multiple-input record.

In a test with several drives, the multiple coherence says how much of
a response the drives account for together. It does not say which drive
does, and it does not say whether the drives were independent enough
for the identification to separate them. Three readings of the same
cross-spectral matrix answer those questions:

- The **reference coherence** is the ordinary coherence between every
  pair of references. Two drives coherent near one cannot be told apart
  at that line: the reference matrix `G_rr` is nearly singular there,
  and an FRF computed from it is whichever of many answers the
  pseudo-inverse picks.
- The **partial coherence** of a response with one reference is the
  ordinary coherence between the two after the linear effect of every
  other reference has been removed from both: the coherence of the
  *conditioned* spectra. With references `x_i` and the rest `r`,

      G_ab·r = G_ab − G_ar G_rr⁻¹ G_rb
      γ²_iy·r = |G_iy·r|² / (G_ii·r G_yy·r)

  It says how much of what the others leave of the response this
  reference explains. With one reference it is the ordinary coherence.
  For two, `1 − γ²_y:x = (1 − γ²_1y)(1 − γ²_2y·1)`: the multiple
  coherence is built from an ordinary and a partial coherence.
- The **virtual coherence** is the coherence of a response with each
  principal reference: the uncorrelated combinations of the references
  given by the eigenvectors of `G_rr`, `G_rr = U Λ Uᴴ`. The principal
  references have autospectra `Λ` and cross spectra `Uᴴ G_ry` with the
  response, and their coherences with it sum to the multiple
  coherence. A principal reference whose eigenvalue is small next to
  the first is a direction the drives hardly excited, so the count of
  large eigenvalues is how many independent drives the test really
  had.

The spectra follow the convention of the rest of the package,
`G_ab = E[A* B]`, so `G_ba` is the conjugate of `G_ab` and every matrix
is Hermitian.

A conditioned autospectrum that is nothing next to the one it came
from is a channel the other references explain completely, and its
partial coherence is undefined, not a number; so are the coherences of
a channel that recorded nothing and of a line holding a value that is
not a finite number. A dead channel or a line nobody drove is not a
coherence of zero.

References
----------
The partial and multiple coherence functions and the conditioned
spectra they are read from are standard; this is where they are set
out.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. Multiple-input systems:
   conditioned spectral density functions, the partial coherence
   function of an output with one input when the other inputs are
   removed, and the multiple coherence function written as a product
   of ordinary and partial coherences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .data import ordinary_coherence

#: a conditioned autospectrum at or under this share of the autospectrum
#: it was conditioned from is a channel the other references explain
#: completely: what is left is round-off, and a coherence of round-off
#: with anything is not a number anyone should read
CONDITIONED_FLOOR = 1e-10

#: a principal reference whose eigenvalue is at or under this share of
#: the first is a direction the drives did not excite at that line
PRINCIPAL_FLOOR = 1e-10

#: the reference coherence over which two drives are said, in the notes,
#: to be hard to tell apart
COUPLED = 0.9

# P7 of the course's audit of the toolset: beside the ordinary and
# multiple coherence already here, which drive explains a response
# (partial), and whether the drives were independent (reference and
# virtual).


def multiple_coherence(grr: np.ndarray, gry: np.ndarray,
                       gyy: np.ndarray) -> np.ndarray:
    """The multiple coherence of each response with a set of references,
    `γ² = g_ryᴴ G_rr⁺ g_ry / G_yy`, at every line.

    The one statement of the multiple coherence:
    `TimeHistory.compute_multiple_coherence` and `read_partial_coherence`
    both read it here. A pseudo-inverse is used rather than a solve,
    because two drives can be nearly the same and `G_rr` is then close
    to singular. Limited to 0 to 1; not a number where the response's
    autospectrum is not above zero or a value is not finite.

    Parameters
    ----------
    grr : numpy.ndarray
        `(lines, q, q)`, the references' cross-spectral matrix.
    gry : numpy.ndarray
        `(lines, q, p)`, the cross spectra from each reference to each
        response, `E[R* Y]`.
    gyy : numpy.ndarray
        `(p, lines)`, the responses' autospectra.

    Returns
    -------
    numpy.ndarray
        `(p, lines)`.
    """
    grr = np.asarray(grr, dtype=np.complex128)
    gry = np.asarray(gry, dtype=np.complex128)
    gyy = np.real(np.asarray(gyy)).astype(np.float64)
    finite = (np.all(np.isfinite(grr), axis=(1, 2))[np.newaxis, :]
              & np.all(np.isfinite(gry), axis=1).T & np.isfinite(gyy))
    safe_rr = np.where(np.isfinite(grr), grr, 0.0)
    safe_ry = np.where(np.isfinite(gry), gry, 0.0)
    inverse = np.linalg.pinv(safe_rr)
    explained = np.real(np.einsum('lip,lij,ljp->pl', np.conj(safe_ry),
                                  inverse, safe_ry))
    with np.errstate(divide='ignore', invalid='ignore'):
        value = explained / gyy
    return np.where(finite & (gyy > 0.0), np.clip(value, 0.0, 1.0), np.nan)


def conditioned(matrix: np.ndarray, keep: Sequence[int],
                given: Sequence[int]) -> np.ndarray:
    """The conditioned cross-spectral matrix of the channels `keep` with
    the linear effect of the channels `given` removed from them,
    `G_kk·g = G_kk − G_kg G_gg⁺ G_gk`, at every line.

    Parameters
    ----------
    matrix : numpy.ndarray
        `(lines, n, n)`, Hermitian.
    keep, given : sequence of int
        Channel indices; `given` may be empty, and then the block of
        `keep` comes back as it is.

    Returns
    -------
    numpy.ndarray
        `(lines, len(keep), len(keep))`.
    """
    keep, given = list(keep), list(given)
    block = matrix[:, keep][:, :, keep]
    if not given:
        return block.copy()
    gkg = matrix[:, keep][:, :, given]
    ggg = matrix[:, given][:, :, given]
    return block - gkg @ np.linalg.pinv(ggg, hermitian=True) @ \
        np.conj(np.swapaxes(gkg, 1, 2))


def _floored(conditioned_auto: np.ndarray, original: np.ndarray
             ) -> np.ndarray:
    """A conditioned autospectrum with what is only round-off of the
    original set to zero, so the coherence read from it is undefined."""
    conditioned_auto = np.real(conditioned_auto)
    original = np.real(original)
    return np.where(conditioned_auto > CONDITIONED_FLOOR * original,
                    conditioned_auto, 0.0)


@dataclass
class PartialCoherencePoint:
    """The coherences of one response at one line.

    Attributes
    ----------
    frequency : float
        The line, Hz.
    multiple : float
        The multiple coherence with every reference.
    ordinary, partial, virtual : numpy.ndarray
        One value per reference (per principal reference for
        `virtual`).
    principal : numpy.ndarray
        The principal references' autospectra, largest first, as shares
        of the first.
    """

    frequency: float
    multiple: float
    ordinary: np.ndarray
    partial: np.ndarray
    virtual: np.ndarray
    principal: np.ndarray


@dataclass
class PartialCoherence:
    """The partial, reference and virtual coherence of a multiple-input
    record, at every line.

    Attributes
    ----------
    frequency : numpy.ndarray
        The lines, Hz.
    references, responses : list of str
        The channels, by DOF (with the quantity where a DOF carries
        two).
    multiple : numpy.ndarray
        `(responses, lines)`.
    ordinary, partial, virtual : numpy.ndarray
        `(responses, references, lines)`: each response with each
        reference alone, with each conditioned on the others, and with
        each principal reference.
    reference : numpy.ndarray
        `(references, references, lines)`: the ordinary coherence
        between every pair of references; one on the diagonal of a
        live channel.
    principal : numpy.ndarray
        `(references, lines)`: the eigenvalues of `G_rr`, largest first,
        in the references' squared unit per hertz when they share one.
    averages : int
        The frames the spectra were averaged over.
    notes : list of str
        What the numbers should be read with.
    """

    frequency: np.ndarray
    references: list[str]
    responses: list[str]
    multiple: np.ndarray
    ordinary: np.ndarray
    partial: np.ndarray
    virtual: np.ndarray
    reference: np.ndarray
    principal: np.ndarray
    averages: int = 0
    notes: list[str] = field(default_factory=list)

    def index(self, response: int | str) -> int:
        """A response's row, by position or by name."""
        if isinstance(response, str):
            if response not in self.responses:
                raise ValueError(f'{response!r} is not a response of this '
                                 'reading: ' + ', '.join(self.responses))
            return self.responses.index(response)
        response = int(response)
        if not 0 <= response < len(self.responses):
            raise ValueError(f'no response {response}: the reading has '
                             f'{len(self.responses)}')
        return response

    def at(self, frequency: float, response: int | str = 0
           ) -> PartialCoherencePoint:
        """The coherences of one response at the line nearest
        `frequency`."""
        k = self.index(response)
        line = int(np.argmin(np.abs(self.frequency - float(frequency))))
        first = self.principal[0, line]
        with np.errstate(divide='ignore', invalid='ignore'):
            shares = (self.principal[:, line] / first if first > 0.0
                      else np.full(len(self.references), np.nan))
        return PartialCoherencePoint(
            float(self.frequency[line]), float(self.multiple[k, line]),
            self.ordinary[k, :, line].copy(), self.partial[k, :, line].copy(),
            self.virtual[k, :, line].copy(), shares)

    def coupled(self, limit: float = COUPLED
                ) -> list[tuple[str, str, float, float]]:
        """The pairs of references whose coherence passes `limit` at any
        line, as (one, other, share of the lines past it, the largest),
        the most coupled first."""
        found = []
        q = len(self.references)
        for i in range(q):
            for j in range(i + 1, q):
                values = self.reference[i, j]
                defined = np.isfinite(values)
                if not defined.any():
                    continue
                past = values[defined] > limit
                if past.any():
                    found.append((self.references[i], self.references[j],
                                  float(np.mean(past)),
                                  float(np.max(values[defined]))))
        return sorted(found, key=lambda entry: -entry[2])

    def effective_references(self, floor: float = 0.01) -> np.ndarray:
        """How many principal references carry at least `floor` of the
        first one's power, at every line: the number of drives the test
        had in effect."""
        first = self.principal[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            shares = self.principal / first
        return np.where(first > 0.0, np.sum(shares >= floor, axis=0), 0)

    def describe(self, response: int | str = 0) -> str:
        """One line about a response: which reference explains most of
        it once the others are removed, and the drives' worst
        coupling."""
        k = self.index(response)
        with np.errstate(all='ignore'):
            medians = [np.nanmedian(row) if np.isfinite(row).any()
                       else np.nan for row in self.partial[k]]
        parts = [(f'{self.responses[k]} against {len(self.references)} '
                  f'reference{"s" * (len(self.references) != 1)}')]
        if np.isfinite(medians).any():
            best = int(np.nanargmax(medians))
            parts.append(f'most explained by {self.references[best]} '
                         f'(median partial coherence {medians[best]:.2f})')
        else:
            parts.append('no partial coherence defined')
        coupled = self.coupled()
        if coupled:
            one, other, share, _top = coupled[0]
            parts.append(f'{one} and {other} coherent over {COUPLED:g} on '
                         f'{share:.0%} of the lines')
        return ', '.join(parts)


def _labels(keys: Sequence[tuple[str, str]]) -> list[str]:
    """A channel's DOF, with its quantity only where a DOF carries two."""
    dofs = [dof for dof, _quantity in keys]
    return [dof if dofs.count(dof) == 1 else f'{dof} {quantity}'
            for dof, quantity in keys]


def read_partial_coherence(frequency: np.ndarray, matrix: np.ndarray,
                           references: Sequence[str],
                           responses: Sequence[str], *,
                           averages: int = 0) -> PartialCoherence:
    """Read the partial, reference and virtual coherence from a
    cross-spectral matrix whose first channels are the references.

    Parameters
    ----------
    frequency : numpy.ndarray
        The lines, Hz.
    matrix : numpy.ndarray
        `(lines, q + p, q + p)`, Hermitian: the `q` references, then
        the `p` responses, `G_ab = E[A* B]`.
    references, responses : sequence of str
        The channels' names, in the matrix's order.
    averages : int, optional
        The frames averaged, carried for the notes.

    Returns
    -------
    PartialCoherence

    Raises
    ------
    ValueError
        When there is no reference or no response, or the matrix's
        shape does not match the names.
    """
    references, responses = list(references), list(responses)
    q, p = len(references), len(responses)
    matrix = np.asarray(matrix, dtype=np.complex128)
    frequency = np.asarray(frequency, dtype=np.float64)
    if not q or not p:
        raise ValueError('partial coherence needs at least one reference '
                         'and one response')
    if matrix.shape != (len(frequency), q + p, q + p):
        raise ValueError(f'the matrix is {matrix.shape}, not '
                         f'({len(frequency)}, {q + p}, {q + p})')
    lines = len(frequency)
    finite = np.all(np.isfinite(matrix), axis=(1, 2))
    safe = np.where(finite[:, np.newaxis, np.newaxis], matrix, 0.0)
    auto = np.real(np.diagonal(safe, axis1=1, axis2=2)).T      # (n, lines)
    auto = np.where(finite[np.newaxis, :], auto, np.nan)

    reference = np.full((q, q, lines), np.nan)
    for i in range(q):
        for j in range(q):
            reference[i, j] = ordinary_coherence(auto[i], auto[j],
                                                 safe[:, i, j])
    ordinary = np.full((p, q, lines), np.nan)
    for y in range(p):
        for i in range(q):
            ordinary[y, i] = ordinary_coherence(auto[i], auto[q + y],
                                                safe[:, i, q + y])

    multiple = multiple_coherence(safe[:, :q, :q], safe[:, :q, q:],
                                  auto[q:])

    # each reference with the responses, the other references removed
    # from all of them; the coherence of what is left
    partial = np.full((p, q, lines), np.nan)
    for i in range(q):
        others = [j for j in range(q) if j != i]
        keep = [i, *range(q, q + p)]
        block = conditioned(safe, keep, others)
        drive = _floored(block[:, 0, 0], auto[i])
        for y in range(p):
            response = _floored(block[:, 1 + y, 1 + y], auto[q + y])
            partial[y, i] = ordinary_coherence(
                np.where(finite, drive, np.nan), response, block[:, 0, 1 + y])

    # the principal references: G_rr = U Λ Uᴴ, largest first; their
    # cross spectra with a response are Uᴴ g_ry
    values, vectors = np.linalg.eigh(safe[:, :q, :q])
    values, vectors = values[:, ::-1], vectors[:, :, ::-1]
    first = values[:, :1]
    principal_auto = np.where(values > PRINCIPAL_FLOOR * first, values, 0.0)
    cross = np.conj(np.swapaxes(vectors, 1, 2)) @ safe[:, :q, q:]
    virtual = np.full((p, q, lines), np.nan)
    for y in range(p):
        for k in range(q):
            virtual[y, k] = ordinary_coherence(
                np.where(finite, principal_auto[:, k], np.nan), auto[q + y],
                cross[:, k, y])
    principal = np.where(finite[np.newaxis, :], np.maximum(values.T, 0.0),
                         np.nan)

    notes = _notes(references, responses, auto, finite, partial, q)
    return PartialCoherence(frequency, references, responses, multiple,
                            ordinary, partial, virtual, reference, principal,
                            int(averages), notes)


def _notes(references, responses, auto, finite, partial, q):
    notes = []
    names = [*references, *responses]
    dead = [name for name, row in zip(names, auto)
            if not np.any(np.nan_to_num(row) > 0.0)]
    if dead:
        several = len(dead) > 1
        notes.append(f'{", ".join(dead)} recorded nothing, so '
                     + ('their coherences are' if several
                        else 'its coherence is') + ' undefined')
    bad = int(np.sum(~finite))
    if bad:
        notes.append(f'{bad} line{"s" * (bad != 1)} holding a value that is '
                     'not a finite number, where nothing is read')
    if q == 1:
        notes.append('one reference: the partial coherence is the ordinary '
                     'coherence, and both are the multiple coherence')
    else:
        explained = [references[i] for i in range(q)
                     if not np.any(np.isfinite(partial[:, i]))
                     and references[i] not in dead]
        if explained:
            one = len(explained) == 1
            notes.append(f'{", ".join(explained)} '
                         f'{"is" if one else "are each"} explained '
                         'completely by the other references at every '
                         f'line, so {"its" if one else "their"} partial '
                         'coherence is undefined: the drives were not '
                         'independent')
    return notes
