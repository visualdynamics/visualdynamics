"""Mode shapes.

A ShapeSet holds a set of modes over a common set of DOFs: the modal
frequency and damping of each mode, and a shape coefficient at each DOF.
Shapes may be real (normal modes) or complex (modes of a damped
structure, often from a test).

Units follow the same rule as the rest of `visualdynamics`. Frequency
is in Hz and damping is a fraction of critical, and both are always
known. Shape coefficients are mass-normalized (phi^T M phi = I), so
their unit is 1/sqrt(mass), and a shape normalized against a mass matrix
in kg is a different number from one normalized against a mass matrix in
slinch. Files do not record which unit was used, so the mass unit starts
undefined and is declared with `ShapeSet.define_units`; the values are
then stored in SI, 1/sqrt(kg).

Because of the half power, this unit cannot be written in the
integer-power dimensions of `visualdynamics.units`, so a ShapeSet stores
its mass unit directly instead of a dimension expression.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..units import si_factor
from .validate import damping as _damping
from .validate import dofs as _dofs
from .validate import frequency as _frequency

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem
    from .geometry import Geometry


def mac_matrix(shapes: ArrayLike,
               others: ArrayLike | None = None) -> np.ndarray:
    """The Modal Assurance Criterion between two shape matrices.

    MAC_ij = |phi_i^H psi_j|^2 / ((phi_i^H phi_i)(psi_j^H psi_j)), which
    is 1 for the same shape at any scaling and near 0 for independent
    shapes. Each matrix has one shape per row. With one argument it is
    the auto-MAC, whose off-diagonal terms show how distinct the set's
    own modes are.
    """
    shapes = np.atleast_2d(shapes)
    others = shapes if others is None else np.atleast_2d(others)
    cross = shapes.conj() @ others.T
    own = np.real(np.sum(shapes.conj() * shapes, axis=1))
    theirs = np.real(np.sum(others.conj() * others, axis=1))
    return np.abs(cross) ** 2 / np.outer(own, theirs)


def cross_mac(a: ShapeSet, b: ShapeSet) -> np.ndarray:
    """MAC between two shape sets, on the DOFs they share.

    The sets do not have to cover the same DOFs or list them in the same
    order; each shape is limited to the shared DOFs before the
    comparison. Rows are `a`'s modes and columns are `b`'s. Raises
    ValueError when the sets share no DOFs.
    """
    positions = {dof: i for i, dof in enumerate(b.coordinate)}
    shared = [dof for dof in a.coordinate if dof in positions]
    if not shared:
        raise ValueError('the shape sets share no DOFs')
    ours = [a.coordinate.index(dof) for dof in shared]
    theirs = [positions[dof] for dof in shared]
    return mac_matrix(a.shape_matrix[:, ours], b.shape_matrix[:, theirs])


def alignment_factor(a: ShapeSet, i: int, b: ShapeSet,
                     j: int) -> complex:
    """The unit factor that phase-aligns b's mode j with a's mode i.

    A real normal mode is only defined up to its sign, and a complex mode
    up to a rotation. The factor that removes the difference is computed
    from the phase of the inner product on the shared DOFs (the same
    inner product as in the MAC numerator). It is 1.0 when the sets share
    no DOFs or the inner product is zero. It is useful by itself when the
    factor is measured between two comparable sets (a set and a
    projection) and has to be applied to a third (the dense set the
    projection was made from).
    """
    positions = {dof: k for k, dof in enumerate(b.coordinate)}
    shared = [dof for dof in a.coordinate if dof in positions]
    if not shared:
        return 1.0
    ours = a.shape_matrix[i, [a.coordinate.index(dof) for dof in shared]]
    theirs = b.shape_matrix[j, [positions[dof] for dof in shared]]
    inner = np.vdot(ours, theirs)
    if inner == 0:
        return 1.0
    return np.exp(-1j * np.angle(inner))


def apply_alignment(shape: np.ndarray, factor: complex) -> np.ndarray:
    """`shape` times the alignment factor, kept real if `shape` was real
    and the factor has no imaginary part."""
    aligned = shape * factor
    if not np.iscomplexobj(shape) and abs(np.imag(factor)) < 1e-9:
        aligned = np.real(aligned)
    return aligned


def aligned_mode(a: ShapeSet, i: int, b: ShapeSet, j: int) -> np.ndarray:
    """b's mode j, phase-aligned to move with a's mode i.

    Returns the full shape vector over b's coordinates
    (`alignment_factor` applied to it).
    """
    return apply_alignment(b.shape_matrix[j], alignment_factor(a, i, b, j))


def paired_mode(a: ShapeSet, i: int, b: ShapeSet, j: int,
                projected: ShapeSet | None = None) -> np.ndarray:
    """b's mode j over b's own coordinates, phase-aligned to move with
    a's mode i when the two are overlaid.

    Sets on one geometry share DOFs and are aligned directly. Sets on two
    geometries share none, so the alignment is measured against
    `projected` (b projected onto a's DOFs by `correlate.project_shapes`)
    and applied to b's original shape. The window's overlay, the
    report's overlay and `ShapeSet.animate_pair` all use this function.
    """
    if projected is None:
        return aligned_mode(a, i, b, j)
    return apply_alignment(b.shape_matrix[j],
                           alignment_factor(a, i, projected, j))


def pair_scale_note(a_name: str, a: ShapeSet, b_name: str, b: ShapeSet,
                    i: int, j: int) -> str | None:
    """One line giving the size of this pair's second shape relative to
    its first, as in ``'Test/Model = 1.02'``, or None when it cannot be
    computed.

    An overlay scales each shape to its own peak, which makes a sparse
    test set and a dense model comparable but also hides a set that is
    thirty times the size of the other. This line gives the number the
    overlay cannot show.

    It is for the pair on screen, not for the whole sets. A constant
    factor across every mode is a unit or a normalization convention,
    but a factor that is 1.00 everywhere and 3.4 on one mode means that
    mode was fitted poorly, and a figure averaged over the set would hide
    that. `ScaleComparison.message` describes the whole set and is what
    the report uses.

    The sets are named by `a_name` and `b_name` rather than as first and
    second, so a reader looking at two animations can tell which one is
    larger. A set that is `unscaled` is named in parentheses after the
    ratio, since its size has no physical meaning.
    """
    try:
        ratios = compare_scaling(a, b, [(i, j)]).ratios
    except (ValueError, IndexError):
        return None
    if not ratios.size or not np.isfinite(ratios[0]):
        return None
    # an unscaled fit has no drive point pinning its size, so the ratio
    # is arithmetic rather than physics — said in a word, because the
    # number is about to be read as if it meant one
    unscaled = [name for name, shapes in ((a_name, a), (b_name, b))
                if getattr(shapes, 'unscaled', False)]
    note = f'{b_name}/{a_name} = {float(ratios[0]):.2f}'
    return note + (f'  ({", ".join(unscaled)} unscaled)'
                   if unscaled else '')


def pair_caption(a_name: str, a: ShapeSet, b_name: str, b: ShapeSet,
                 i: int, j: int) -> str:
    """The caption in the corner of an overlay.

    One line for each mode (set name, mode number, description,
    frequency in Hz and damping in percent), then the scale note from
    `pair_scale_note` when there is one.
    """
    def described(name: str, shape_set: ShapeSet, mode: int) -> str:
        description = str(shape_set.description[mode]).strip()
        return (f'{name}  mode {mode + 1}'
                + (f'  {description}' if description else '')
                + f'  {float(shape_set.frequency[mode]):.4f} Hz'
                f'  {float(shape_set.damping[mode]) * 100:.3f} %')

    caption = described(a_name, a, i) + '\n' + described(b_name, b, j)
    scaling = pair_scale_note(a_name, a, b_name, b, i, j)
    return caption + ('\n' + scaling if scaling else '')


#: how far the two sets' levels may differ before it is worth saying so,
#: as a fraction. Two properly mass-normalized sets of one structure in
#: one unit system land far closer than this; a tenth is loose enough
#: not to fire on ordinary test-analysis scatter in modal mass.
SCALE_TOLERANCE = 0.1

#: how much the per-mode ratio may vary across the matched pairs before
#: the two sets are not normalized the same way at all. A constant
#: factor is a unit or a scaling convention; a factor that changes mode
#: by mode is a different normalization, and the two want different
#: sentences.
SCALE_SPREAD = 1.5


class ScaleComparison:
    """Whether two shape sets are scaled the same way, and what it means
    if they are not.

    An overlay of two mode shapes scales each to its own peak, so a
    sparse test set and a dense model are drawn at a comparable size and
    the comparison is of shape only. As a result a reader cannot see
    that one set is thirty times the size of the other. `message` gives
    that information as a sentence.

    There are two cases, because the two kinds of difference look
    different:

    - A constant factor. Every matched pair differing by the same amount
      is a unit or a normalization convention, not a difference in the
      structure. Mode shapes scale as one over the square root of mass,
      so sets normalized in grams and in kilograms differ by a factor of
      about 31.6.
    - A factor that changes from mode to mode. This is two different
      normalizations, most often unit-normalized shapes (each mode's
      peak set to 1) beside mass-normalized ones.

    `ratios` is `second / first` for each matched pair, measured as the
    norm over the DOFs the two sets share, so a sparse set and a dense
    one are compared over the same DOFs (`scale_ratios`). `first` and
    `second` are the two shape sets, used by `message` to check their
    mass units and whether either is unscaled.
    """

    def __init__(self, ratios: ArrayLike, first: str | None = None,
                 second: str | None = None) -> None:
        self.ratios: np.ndarray = np.asarray(ratios, dtype=float)
        self.first, self.second = first, second

    def __repr__(self) -> str:
        if not self.ratios.size:
            return 'ScaleComparison(nothing to compare)'
        return (f'ScaleComparison(factor={self.factor:.4g}, '
                f'spread={self.spread:.3g}, equal={self.equal})')

    @property
    def factor(self) -> float:
        """The median ratio of the second set to the first, or NaN when
        there are no pairs."""
        return (float(np.median(self.ratios)) if self.ratios.size
                else float('nan'))

    @property
    def spread(self) -> float:
        """The largest ratio over the smallest; 1.0 when every pair has
        the same ratio, or when fewer than two ratios are usable."""
        usable = self.ratios[np.isfinite(self.ratios) & (self.ratios > 0)]
        if usable.size < 2:
            return 1.0
        return float(usable.max() / usable.min())

    @property
    def consistent(self) -> bool:
        """Whether the two sets have one scaling factor, whatever it is
        (`spread` at most 1.5)."""
        return self.spread <= SCALE_SPREAD

    @property
    def equal(self) -> bool:
        """Whether the sets are scaled the same: one factor (`consistent`)
        and that factor within 10% of 1."""
        factor = self.factor
        return bool(self.consistent and np.isfinite(factor)
                    and abs(factor - 1.0) <= SCALE_TOLERANCE)

    def message(self, first_name: str = 'the first set',
                second_name: str = 'the second set') -> str | None:
        """A sentence describing the scale difference, or None when the
        sets are scaled the same.

        The message describes the difference and does not call it an
        error, because a scale difference is sometimes intended and the
        reader knows whether it is. The overlay cannot show it, because
        it scales each shape to its own peak. An unscaled set, or mass
        units that differ or are declared on only one set, are reported
        before the ratios.
        """
        if not self.ratios.size:
            return None
        for shapes, name in ((self.first, first_name),
                             (self.second, second_name)):
            if getattr(shapes, 'unscaled', False):
                return (f'{name} is unscaled — no drive point was '
                        'measured, so its shapes carry an arbitrary '
                        'scale and their size means nothing. The '
                        'overlay compares shape, which is unaffected.')
        units = self._units_note(first_name, second_name)
        if units:
            return units
        if self.equal:
            return None
        if not self.consistent:
            low, high = float(self.ratios.min()), float(self.ratios.max())
            return (f'{second_name} is between {low:.4g} and {high:.4g} '
                    f'times {first_name}, mode by mode — so the two are '
                    'not normalized the same way. Unit-normalized shapes '
                    'beside mass-normalized ones look like this. The '
                    'overlay scales each to its own peak, so the shapes '
                    'still compare.')
        return (f'{second_name} is {self.factor:.4g} times {first_name} '
                'across every matched mode. A constant factor is a unit '
                'or a normalization convention rather than a difference '
                'in the structure — shapes go as one over the square root '
                'of mass, so a mass unit out by a thousand shows up as '
                'about 31.6. The overlay scales each to its own peak, so '
                'the shapes still compare.')

    def _units_note(self, first_name, second_name):
        """The declared mass units, when they are the plain explanation.

        Checked before the numbers because it *is* the answer when it
        applies: two sets normalized against different mass units differ
        by exactly the square root of that ratio, and naming the units
        beats reporting the factor they imply.
        """
        first = getattr(self.first, 'mass_unit', None)
        second = getattr(self.second, 'mass_unit', None)
        if first is not None and second is not None and first != second:
            return (f'{first_name} is normalized in {first} and '
                    f'{second_name} in {second}. Shapes go as one over '
                    'the square root of mass, so their coefficients are '
                    'not comparable until both are declared in the same '
                    'unit.')
        if (first is None) != (second is None):
            named, bare = ((first_name, second_name) if second is None
                           else (second_name, first_name))
            return (f'{named} declares the mass unit it was normalized '
                    f"against and {bare} does not, so the two sets' "
                    'coefficients cannot be compared. Define the units '
                    f'on {bare}.')
        return None


def scale_ratios(a: ShapeSet, b: ShapeSet,
                 pairs: Sequence[Sequence[int]] | None = None
                 ) -> list[float | None]:
    """The size of b's mode relative to a's, one value per pair.

    Each ratio is the norm of b's mode over the norm of a's, both limited
    to the DOFs the two sets share; otherwise a sparse test set and a
    dense model would differ by the square root of the ratio of their DOF
    counts.

    `pairs` is a list of (a's mode, b's mode). Without it, the modes are
    paired in order up to the length of the shorter set.

    The list has one entry per pair, in order, with None where there is
    no value: a mode index past the end of its set, a mode that is zero
    over the shared DOFs, or two sets that share no DOFs. The entries are
    not filtered out, because a table puts them in rows beside the pairs
    they belong to, and a shorter list would put values in the wrong
    rows.
    """
    if pairs is None:
        pairs = [(i, i) for i in range(min(a.num_shapes, b.num_shapes))]
    positions = {dof: i for i, dof in enumerate(b.coordinate)}
    shared = [dof for dof in a.coordinate if dof in positions]
    if not shared:
        return [None] * len(list(pairs))
    ours = a.shape_matrix[:, [a.coordinate.index(dof) for dof in shared]]
    theirs = b.shape_matrix[:, [positions[dof] for dof in shared]]
    out: list[float | None] = []
    for row, column in pairs:
        if not (0 <= row < len(ours) and 0 <= column < len(theirs)):
            out.append(None)
            continue
        mine = float(np.linalg.norm(ours[row]))
        yours = float(np.linalg.norm(theirs[column]))
        out.append(yours / mine if mine > 0.0 and yours > 0.0 else None)
    return out


def compare_scaling(a: ShapeSet, b: ShapeSet,
                    pairs: Sequence[Sequence[int]] | None = None) -> ScaleComparison:
    """`ScaleComparison` of two shape sets over their matched pairs.

    The pairs that have a ratio in `scale_ratios` are summarized as one
    factor and a spread. See `scale_ratios` for how a ratio is computed
    and when there is none.
    """
    return ScaleComparison(
        [r for r in scale_ratios(a, b, pairs) if r is not None], a, b)


def _split_sign(dof):
    """('101X-') -> ('101X', -1.0); a bare '101X' reads as positive."""
    if dof and dof[-1] in '+-':
        return dof[:-1], -1.0 if dof[-1] == '-' else 1.0
    return dof, 1.0


class ShapeSet:
    """Mode shapes over a shared set of DOFs.

    `shape_matrix` is (modes, dofs). `coordinate` lists the DOF strings
    that the columns correspond to ('101X+'). The modes are always stored
    in ascending frequency, so rigid-body modes come first.

    A fitted set is also the record of the fit: reopening one in the app
    (Edit Fit) restores the fitting session that produced it, which is
    why the descriptions and the scaling flag are stored with the
    numbers.

    Attributes:
        frequency: Hz per mode. A rigid-body mode is exactly 0.0, because
            the FRF synthesis tests for exactly 0.0 to handle its 0/0; a
            very small frequency is not treated the same way.
        damping: Fraction of critical per mode, so 2% is 0.02.
        coordinate: The DOF string of each column of `shape_matrix`.
        shape_matrix: `(modes, dofs)`. Complex for complex modes; the
            overlay and the MAC handle either.
        modal_mass: Per mode. 1.0 for every mode of a mass-normalized
            set, which is what an eigensolution here produces. For
            complex shapes it is modal A, the first-order scaling, and
            `synthesize_frf` uses it that way. It is complex when an
            imported file had it complex; the values are kept as read and
            never converted to real.
        modal_damping: Complex modal damping per mode where a file had
            one (I-DEAS ADFs do), or None. It is different from
            `damping`, the viscous fraction of critical: it is the
            complex-mode estimate as the identification software
            reported it.
        mass_unit: The mass unit the shapes were normalized against, or
            `None` when it has not been declared.
        description: Free text per mode describing the shape, for
            example 'first torsion'.
        comment: One line per mode; a single string given to the
            constructor is used for every mode.
        unscaled: True when the fit had no drive point to set the
            mass-normalized scale. Shapes and MACs are not affected, but
            the modal masses are only a convention, and comparisons do
            not report a scale factor from them.
    """

    def __init__(self, frequency: ArrayLike, damping: ArrayLike,
                 coordinate: Sequence[str], shape_matrix: ArrayLike,
                 modal_mass: ArrayLike | None = None,
                 comment: str | Sequence[str] | None = None,
                 mass_unit: str | None = None,
                 description: Sequence[str] | None = None,
                 unscaled: bool = False,
                 modal_damping: ArrayLike | None = None) -> None:
        # True when the fit had no drive point to pin the
        # mass-normalized scale: shapes and MACs are fine, modal
        # masses are a convention, not physics
        self.unscaled: bool = bool(unscaled)
        self.frequency: np.ndarray = _frequency(frequency)
        self.damping: np.ndarray = _damping(damping)
        matrix = np.atleast_2d(np.asarray(shape_matrix))
        self.shape_matrix: np.ndarray = matrix
        n = matrix.shape[0]
        if len(self.frequency) != n or len(self.damping) != n:
            raise ValueError(
                f'shape_matrix has {n} modes but frequency/damping have '
                f'{len(self.frequency)}/{len(self.damping)}')
        self.coordinate: list[str] = _dofs(coordinate, 'coordinate')
        if matrix.shape[1] != len(self.coordinate):
            raise ValueError(
                f'shape_matrix has {matrix.shape[1]} columns but '
                f'{len(self.coordinate)} coordinates')
        # complex where a source says so — an imported complex-mode set
        # carries complex modal parameters as measured, and squeezing
        # them real would silently discard half the estimate. The fit
        # here never produces them; imports may.
        mass = (np.ones(n) if modal_mass is None
                else np.atleast_1d(np.asarray(modal_mass)))
        if np.iscomplexobj(mass) and not np.any(mass.imag):
            mass = mass.real
        self.modal_mass: np.ndarray = mass.astype(
            np.complex128 if np.iscomplexobj(mass) else np.float64)
        #: complex modal damping per mode where a source carried one
        #: (I-DEAS ADFs do); None for sets that never had it
        self.modal_damping: np.ndarray | None = (
            None if modal_damping is None
            else np.atleast_1d(np.asarray(modal_damping)))
        self.comment: list[str]
        self.description: list[str]
        if comment is None:
            self.comment = [''] * n
        elif isinstance(comment, str):
            self.comment = [comment] * n
        else:
            self.comment = [str(c) for c in comment]
            if len(self.comment) != n:
                raise ValueError(f'comment has {len(self.comment)} entries, '
                                 f'expected {n}')
        # free text the user can write against each mode
        if description is None:
            self.description = [''] * n
        else:
            self.description = [str(d) for d in description]
            if len(self.description) != n:
                raise ValueError(
                    f'description has {len(self.description)} entries, '
                    f'expected {n}')
        self.mass_unit: str | None = mass_unit or None
        if self.mass_unit is not None:
            si_factor(self.mass_unit, 'mass')  # validates
        self._sort_by_frequency()

    def _sort_by_frequency(self) -> None:
        """Modes in ascending frequency, always.

        A set is read as a list — mode 1, mode 2 — and everything that
        steps through one (the mode box, the MAC's axes, a matched
        table, a report) reads it in the order it is stored. Fitting is
        the case that breaks it: peaks are confirmed in whatever order
        they are found, and the largest is rarely the lowest, so a set
        fitted from the top down came out backwards and every reading of
        it was backwards too.

        Stable, so coincident frequencies keep the order they arrived in
        — a pair of repeated roots is not two modes anyone has ordered.
        Rigid-body modes are exactly 0.0 and therefore first, which is
        the convention every solver uses.
        """
        order = np.argsort(self.frequency, kind='stable')
        if np.array_equal(order, np.arange(len(order))):
            return                      # already in order; touch nothing
        self.frequency = self.frequency[order]
        self.damping = self.damping[order]
        self.shape_matrix = self.shape_matrix[order]
        self.modal_mass = self.modal_mass[order]
        self.comment = [self.comment[i] for i in order]
        self.description = [self.description[i] for i in order]

    def _dof_lookup(self):
        """{base DOF: (column, sign)} — '101X-' is minus the '101X+' column."""
        return {base: (i, sign) for i, (base, sign) in
                enumerate(_split_sign(dof) for dof in self.coordinate)}

    def auto_mac(self) -> np.ndarray:
        """The MAC of every mode against every other mode; the diagonal is 1."""
        return mac_matrix(self.shape_matrix)

    def covers(self, dof: str) -> bool:
        """Whether the shape set has a coefficient at this DOF.

        The sign is ignored, so '101Z-' is covered when '101Z+' is.

        Parameters
        ----------
        dof : str
            A degree of freedom, such as '101Z+'.

        Returns
        -------
        bool
            Whether the shapes include it.
        """
        return _split_sign(str(dof))[0] in self._dof_lookup()

    def synthesize_frf(self, frequencies: ArrayLike,
                       response_dof: Sequence[str],
                       reference_dof: Sequence[str],
                       modes: Sequence[int] | None = None,
                       power: int = 0) -> np.ndarray:
        """FRFs from the modal model, one row per DOF pair.

        For real shapes, the second-order residue form

            H_jk(f) = sum_r (iw)^power phi_jr phi_kr
                            / (m_r (w_r^2 - w^2 + 2i z_r w_r w))

        with w = 2*pi*f. It is exact for mass-normalized shapes, and
        `modal_mass` holds any other scaling. For complex shapes, the
        first-order form, with a pole and its conjugate:

            H_jk(f) = sum_r (iw)^power [ psi_jr psi_kr / (A_r (iw - l_r))
                          + conj(psi_jr psi_kr / A_r) / (iw - conj(l_r)) ]

        with the pole l_r = -z_r w_r + i w_r sqrt(1 - z_r^2) and
        `modal_mass` used as modal A, the scaling of a complex mode (1.0
        when a set has none). The two forms give the same FRF for a real
        mode whose modal A is 2i w_d m_r, where w_d is the damped natural
        frequency.

        The form is selected by the shapes alone: complex shapes use the
        first-order form and real shapes the second-order form. The
        second-order form cannot represent a complex mode, because it
        gives the conjugate pole the residue -R where the structure has
        conj(R), whatever `modal_mass` is set to. Modal A is conjugated
        in the second term, which only matters when modal A is complex.

        `power` selects the response quantity: 0 displacement per force,
        1 velocity, 2 acceleration. It is applied inside the sum because
        a real rigid-body mode's denominator is exactly -w^2: at w = 0
        its accelerance reduces to the finite residue, while multiplying
        afterward would give 0/0. Its displacement and velocity at w = 0
        are unbounded and are returned as nan.

        Both forms are checked to rounding error against the direct
        inverse of the dynamic stiffness (proportionally damped for real
        shapes, non-proportionally damped for complex ones) and against
        the output of sdynpy's synthesis saved as test data
        (`tests/test_frf_synthesis_exact.py`,
        `tests/test_modal_frf_oracle.py`).

        `modes` limits the sum to some modes; a synthesis from a subset
        of modes plotted beside the measurement shows which modes the
        measurement contains. Raises ValueError for a DOF the shapes do
        not cover, which `covers` can check in advance.

        Parameters
        ----------
        frequencies : array_like
            The lines to synthesize at, in Hz.
        response_dof : sequence of str
            The response degrees of freedom.
        reference_dof : sequence of str
            The drive degrees of freedom.
        modes : sequence of int, optional
            Which modes to include. All of them when omitted.
        power : int, default 0
            0 receptance, 1 mobility, 2 accelerance.

        Returns
        -------
        numpy.ndarray
            The synthesized FRFs, one row per response and drive pair.

        References
        ----------
        1. Ewins, D. J. (2000). *Modal Testing: Theory, Practice and
           Application*, 2nd ed. Research Studies Press. The residue
           form for real modes, and why non-proportional damping makes
           the modes complex.
        2. Maia, N. M. M., & Silva, J. M. M. (1997). *Theoretical and
           Experimental Modal Analysis*. Research Studies Press. The
           first-order (state-space) form, its conjugate pole pairs,
           and modal A as the scaling that goes with them.
        """
        # The form follows the shapes, complex or real, the same rule
        # sdynpy uses. Until 2026-09-23 complex sets went through the
        # second-order form, and with modal A in `modal_mass` every
        # imported complex set synthesized an answer rotated 90 degrees
        # and 2 w_d times too small. Modal A is conjugated in the second
        # term; sdynpy's is not, and the two agree whenever modal A is
        # real. On the complex modal A of the test system the
        # unconjugated form is 2.5% of peak off the exact FRF (measured
        # 2026-09-24).
        frequencies = np.asarray(frequencies, dtype=np.float64)
        picked = (np.arange(self.num_shapes) if modes is None
                  else np.asarray(sorted({int(m) for m in modes})))
        lookup = self._dof_lookup()

        def shape_at(dof: str) -> np.ndarray:
            base, sign = _split_sign(str(dof))
            if base not in lookup:
                raise ValueError(f'shapes have no coefficient at DOF {dof!r}')
            column, stored_sign = lookup[base]
            return self.shape_matrix[picked, column] * (sign * stored_sign)

        omega = 2.0 * np.pi * frequencies
        omega_r = 2.0 * np.pi * self.frequency[picked]
        zeta = self.damping[picked]
        numerator = (1j * omega) ** power
        out = np.empty((len(response_dof), len(frequencies)),
                       dtype=np.complex128)
        if self.is_complex:
            # complex so an overdamped entry gives a real pole pair
            # rather than nan
            pole = (-zeta * omega_r
                    + 1j * omega_r * np.sqrt((1.0 - zeta ** 2).astype(complex)))
            # an undamped mode's own resonance line divides by zero; the
            # infinity is the right answer there, not worth a warning
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = numerator[None, :] / (1j * omega[None, :] - pole[:, None])
                mirror = numerator[None, :] / (1j * omega[None, :]
                                               - np.conj(pole)[:, None])
            for row, (response, reference) in enumerate(zip(response_dof,
                                                            reference_dof)):
                residues = (shape_at(response) * shape_at(reference)
                            / self.modal_mass[picked])
                out[row] = residues @ ratio + np.conj(residues) @ mirror
            return out
        denominator = ((omega_r ** 2)[:, None] - (omega ** 2)[None, :]
                       + 2j * (zeta * omega_r)[:, None] * omega[None, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = numerator[None, :] / denominator
        rigid, still = omega_r == 0.0, omega == 0.0
        if rigid.any() and still.any():
            ratio[np.ix_(rigid, still)] = 1.0 if power == 2 else np.nan
        for row, (response, reference) in enumerate(zip(response_dof,
                                                        reference_dof)):
            residues = (shape_at(response) * shape_at(reference)
                        / self.modal_mass[picked])
            out[row] = residues @ ratio
        return out

    def delete_modes(self, indices: Sequence[int]) -> None:
        """Remove the given modes in place.

        Raises IndexError for an index out of range, and ValueError if
        every mode would be removed (delete the object instead).

        Parameters
        ----------
        indices : sequence of int
            Which modes to remove, by zero-based index.

        Returns
        -------
        None
        """
        doomed = {int(i) for i in indices}
        bad = [i for i in doomed if not 0 <= i < self.num_shapes]
        if bad:
            raise IndexError(f'no such modes: {sorted(bad)}')
        keep = [i for i in range(self.num_shapes) if i not in doomed]
        if not keep:
            raise ValueError('cannot delete every mode; delete the '
                             'object instead')
        self.frequency = self.frequency[keep]
        self.damping = self.damping[keep]
        self.shape_matrix = self.shape_matrix[keep]
        self.modal_mass = self.modal_mass[keep]
        self.comment = [self.comment[i] for i in keep]
        self.description = [self.description[i] for i in keep]

    @property
    def num_shapes(self) -> int:
        """How many mode shapes the set holds."""
        return self.shape_matrix.shape[0]

    @property
    def num_dofs(self) -> int:
        """How many degrees of freedom each shape covers."""
        return self.shape_matrix.shape[1]

    @property
    def is_complex(self) -> bool:
        """Whether these are complex modes.

        Real normal modes move every DOF in or out of phase; complex
        modes do not, which is the result of non-proportional damping.
        A set is complex when its shape matrix has a complex dtype.
        """
        return np.iscomplexobj(self.shape_matrix)

    # ---- units --------------------------------------------------------------

    @property
    def units_defined(self) -> bool:
        """Whether a mass unit has been declared. Without one, a modal
        mass is a number with no unit."""
        return self.mass_unit is not None

    @staticmethod
    def _scale_to_si(mass_unit):
        """Factor taking coefficients in 1/sqrt(mass_unit) to 1/sqrt(kg)."""
        return 1.0 / math.sqrt(si_factor(mass_unit, 'mass'))

    def define_units(self, mass_unit: str) -> ShapeSet:
        """Declare the mass unit the shapes were normalized against.

        The coefficients are converted to SI, 1/sqrt(kg). Declaring again
        reinterprets the original values instead of scaling twice, so a
        wrong unit can be corrected.

        Parameters
        ----------
        mass_unit : str
            The mass unit of the mass matrix the shapes were normalized
            against, such as 'kg'.

        Returns
        -------
        ShapeSet
            Self, converted to SI in place.
        """
        scale = self._scale_to_si(mass_unit)
        raw = (self.shape_matrix if self.mass_unit is None
               else self.shape_matrix / self._scale_to_si(self.mass_unit))
        self.shape_matrix = raw * scale
        self.mass_unit = mass_unit
        return self

    def undefine_units(self) -> ShapeSet:
        """Remove the mass unit and restore the original coefficients; returns self."""
        if self.mass_unit is None:
            return self
        self.shape_matrix = self.shape_matrix / self._scale_to_si(self.mass_unit)
        self.mass_unit = None
        return self

    def display_shapes(self, unit_system: UnitSystem) -> np.ndarray:
        """The coefficients in the display system's 1/sqrt(mass).

        A set without a declared mass unit is returned unchanged.

        Parameters
        ----------
        unit_system : UnitSystem
            The units to present in.

        Returns
        -------
        numpy.ndarray
            The shape matrix in display units.
        """
        if not self.units_defined:
            return self.shape_matrix
        return self.shape_matrix * math.sqrt(
            si_factor(unit_system.unit('mass'), 'mass'))

    def unit_label(self, unit_system: UnitSystem | None = None) -> str:
        """The unit label of the coefficients, such as '1/√kg'.

        Parameters
        ----------
        unit_system : UnitSystem, optional
            The units to label in. The declared mass unit when omitted.

        Returns
        -------
        str
            The label, or '' when no mass unit has been declared.
        """
        if not self.units_defined:
            return ''
        unit = (self.mass_unit if unit_system is None
                else unit_system.unit('mass'))
        return f'1/√{unit}'

    def mode_label(self, i: int) -> str:
        """A short label for one mode, such as 'Mode 3 — 12.4 Hz, 2% damping'.

        Parameters
        ----------
        i : int
            Which mode, by zero-based index.

        Returns
        -------
        str
            A short label: its frequency, and its damping when known.
        """
        label = f'Mode {i + 1} — {self.frequency[i]:.4g} Hz'
        if self.damping[i]:
            label += f', {self.damping[i] * 100:.3g}% damping'
        return label

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShapeSet):
            return NotImplemented
        return (np.allclose(self.frequency, other.frequency)
                and np.allclose(self.damping, other.damping)
                and self.coordinate == other.coordinate
                and self.shape_matrix.shape == other.shape_matrix.shape
                and np.allclose(self.shape_matrix, other.shape_matrix)
                and np.allclose(self.modal_mass, other.modal_mass)
                and self.comment == other.comment
                and self.description == other.description
                and self.mass_unit == other.mass_unit)

    def __repr__(self) -> str:
        kind = 'complex' if self.is_complex else 'real'
        span = (f'{self.frequency.min():.4g}-{self.frequency.max():.4g} Hz'
                if self.num_shapes else 'no modes')
        units = self.unit_label() if self.units_defined else 'units undefined'
        return (f'ShapeSet({self.num_shapes} {kind} modes x {self.num_dofs} '
                f'DOFs, {span}, {units})')

    def save(self, path: str | os.PathLike) -> None:
        """Write the shape set to a file of its own.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write it.

        Returns
        -------
        None
        """
        from ..io import native
        native.save(self, path)

    def plot_mac(self, other: ShapeSet | None = None, **kwargs: Any) -> Any:
        """Plot the MAC: this set against itself, or against `other`.

        Parameters
        ----------
        other : ShapeSet, optional
            The set to compare against. When omitted, the plot is the
            auto-MAC, which shows repeated or similar modes.
        **kwargs
            Passed through to the plotting layer.

        Returns
        -------
        object
            The plot widget.
        """
        from ..plot import plot_mac
        return plot_mac(self, other, **kwargs)

    def animate(self, geometry: Geometry, mode: int = 0,
                **kwargs: Any) -> Any:
        """Animate one mode on a geometry, as the app does.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        mode : int, default 0
            Which mode, by index.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..viz.animate import animate_shape
        return animate_shape(geometry, self, mode, **kwargs)

    def animate_pair(self, geometry: Geometry, mode: int,
                     other: ShapeSet, other_mode: int,
                     other_geometry: Geometry | None = None, *,
                     on_basis: bool = False,
                     projected: ShapeSet | None = None,
                     names: tuple[str, str] = ('first', 'second'),
                     **kwargs: Any) -> Any:
        """Animate this mode and a mode of another set overlaid, as the
        window does for a picked cell of the two sets' cross-MAC.

        The other mode is phase-aligned to this one (`paired_mode`) and
        each is scaled to its own peak, so the animation compares shape.
        The caption names both modes and gives the ratio of their sizes
        (`pair_caption`). When the sets are on two geometries, the other
        set is projected onto this one's DOFs to measure the alignment
        (`correlate.project_shapes`) and drawn on its own geometry, or,
        with `on_basis`, the projection is drawn on this geometry.
        Raises ValueError when either set has no DOFs on its geometry.

        Parameters
        ----------
        geometry : Geometry
            This set's geometry.
        mode : int
            This set's mode, by index.
        other : ShapeSet
            The set compared against.
        other_mode : int
            Its mode, by index.
        other_geometry : Geometry, optional
            The other set's geometry; this one's when omitted.
        on_basis : bool, default False
            Draw the other set's projection on this geometry instead.
        projected : ShapeSet, optional
            The other set already projected onto this set's DOFs
            (`correlate.project_shapes`), for a script that animates
            many pairs of the same two sets. Computed here when omitted
            and the geometries differ.
        names : tuple of str
            The names the caption uses for the two sets, by default
            ('first', 'second').
        **kwargs
            Passed through to `viz.animate.animate_pair`: `colors`,
            `alphas`, `scale`, `screenshot=`, `movie=`.

        Returns
        -------
        object
            The plotter, the still, or the movie's path.
        """
        from ..viz.animate import animate_pair, pair_deflections

        other_geometry = other_geometry or geometry
        if other_geometry is not geometry and projected is None:
            from .correlate import project_shapes
            projected, _report = project_shapes(other, other_geometry,
                                                self, geometry)
        first, second = pair_deflections(self, mode, other, other_mode,
                                         geometry, other_geometry,
                                         projected, on_basis)
        if not len(first[1].rows) or not len(second[1].rows):
            raise ValueError('a set of the pair has no DOFs on its '
                             'geometry')
        kwargs.setdefault('caption', pair_caption(
            names[0], self, names[1], other, mode, other_mode))
        return animate_pair(first, second, **kwargs)

    def plot(self, geometry: Geometry | None = None, mode: int = 0,
             **kwargs: Any) -> Any:
        """Plot the set: its auto-MAC, or one mode animated when a
        geometry is given.

        Parameters
        ----------
        geometry : Geometry, optional
            The geometry to draw on. Without it the auto-MAC is plotted.
        mode : int, default 0
            Which mode, when a geometry is given.
        **kwargs
            Passed through to `plot_mac` or `animate`.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        if geometry is None:
            return self.plot_mac(**kwargs)
        return self.animate(geometry, mode, **kwargs)


def synthesize_overlay(shapes: ShapeSet, frf: Any,
                       records: Sequence[int] | None = None,
                       modes: Sequence[int] | None = None) -> Any:
    """The modal model's prediction of `frf`, as an Frf to plot over it.

    The result has the same DOF pairs and frequencies as `frf`, computed
    with `ShapeSet.synthesize_frf` over `modes` (all of them by default),
    for the records in `records` (all by default). Each record copies its
    original's dimension and units, so both are plotted on one axis, and
    is computed for the original's quantity by powers of iω; accelerance
    is used when the measurement's quantity is not known, as is usual in
    a modal test. Records at DOFs the shapes do not cover are left out,
    and the result is None when that is every record or `frf` has no
    reference DOFs.

    The result has `synthesized` set to True (it is drawn dashed) and
    `mode_frequencies` (the frequencies marked on the fitting screen), so
    a plot can tell it from a measurement.
    """
    import numpy as np

    from .data import Frf

    rows = (list(records) if records is not None
            else list(range(frf.num_records)))
    usable = [i for i in rows
              if frf.reference_dof is not None
              and shapes.covers(frf.response_dof[i])
              and shapes.covers(frf.reference_dof[i])]
    if not usable:
        return None
    # grouped by quantity, because the (iω) power applies inside the
    # modal sum — see synthesize_frf on rigid-body modes at 0 Hz
    powers = {'length': 0, 'velocity': 1, 'acceleration': 2}
    by_power: dict[int, list[int]] = {}
    for j, i in enumerate(usable):
        power = powers.get(frf.known_dim(i).partition('/')[0], 2)
        by_power.setdefault(power, []).append(j)
    values = np.empty((len(usable), len(frf.abscissa)), dtype=np.complex128)
    for power, positions in by_power.items():
        values[positions] = shapes.synthesize_frf(
            frf.abscissa,
            [frf.response_dof[usable[j]] for j in positions],
            [frf.reference_dof[usable[j]] for j in positions],
            modes=modes, power=power)
    overlay = Frf(
        frf.abscissa, values,
        response_dof=[frf.response_dof[i] for i in usable],
        reference_dof=[frf.reference_dof[i] for i in usable],
        ordinate_dim=[frf.ordinate_dim[i] for i in usable],
        ordinate_unit=[frf.ordinate_unit[i] for i in usable],
        reference_unit=[frf.reference_unit[i] for i in usable],
        dimension_hint=[frf.dimension_hint[i] for i in usable],
        comment='synthesized')
    overlay.synthesized = True
    used = modes if modes is not None else range(shapes.num_shapes)
    overlay.mode_frequencies = [float(shapes.frequency[m]) for m in used]
    return overlay
