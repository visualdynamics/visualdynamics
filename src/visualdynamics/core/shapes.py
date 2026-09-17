"""Mode shapes.

A ShapeSet holds a set of modes over a common set of DOFs: modal frequency
and damping per mode, plus a shape coefficient per DOF. Shapes may be real
(normal modes) or complex (damped/experimental modes).

Units follow the same rule as the rest of visualdynamics. Frequency is Hz and damping
is a fraction of critical, both always known. Shape coefficients are
mass-normalized (phi^T M phi = I), so they carry units of 1/sqrt(mass) — a
shape normalized against a kg mass matrix is not the same number as one
normalized against slinch. Files do not record which was used, so the mass
unit starts undefined and the user declares it; values are then stored in
SI, 1/sqrt(kg).

The half power puts this outside the integer-power dimension algebra in
visualdynamics.units, so a ShapeSet carries its mass unit directly rather than a
dimension expression.
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

    MAC_ij = |phi_i^H psi_j|^2 / ((phi_i^H phi_i)(psi_j^H psi_j)) — 1 for
    the same shape in any scaling, near 0 for independent ones. With one
    argument it is the auto-MAC, whose off-diagonals say how distinct the
    set's own modes are.
    """
    shapes = np.atleast_2d(shapes)
    others = shapes if others is None else np.atleast_2d(others)
    cross = shapes.conj() @ others.T
    own = np.real(np.sum(shapes.conj() * shapes, axis=1))
    theirs = np.real(np.sum(others.conj() * others, axis=1))
    return np.abs(cross) ** 2 / np.outer(own, theirs)


def cross_mac(a: ShapeSet, b: ShapeSet) -> np.ndarray:
    """MAC between two shape sets, on the DOFs they share.

    The sets need not cover the same DOFs or list them in the same order;
    each shape is restricted to the shared set before comparing. Rows are
    `a`'s modes, columns are `b`'s. Raises when nothing is shared —
    a MAC over no DOFs is not a number.
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

    A real normal mode is defined up to sign and a complex one up to a
    rotation; the factor undoing that is the phase of the inner product
    on the shared DOFs — the MAC numerator's own. 1.0 when the sets
    share nothing (there is no alignment to speak of). Useful on its
    own when the factor, measured between comparable sets (a set and a
    projection), must be applied to a third (the projection's dense
    original)."""
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
    """`shape` times the alignment factor, kept real when it was."""
    aligned = shape * factor
    if not np.iscomplexobj(shape) and abs(np.imag(factor)) < 1e-9:
        aligned = np.real(aligned)
    return aligned


def aligned_mode(a: ShapeSet, i: int, b: ShapeSet, j: int) -> np.ndarray:
    """b's mode j, phase-aligned to move with a's mode i.

    Returns the full shape vector over b's coordinates.
    """
    return apply_alignment(b.shape_matrix[j], alignment_factor(a, i, b, j))


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
    """Whether two shape sets are scaled alike, and what it means if not.

    Overlaying two mode shapes scales each to its own peak, so that a
    sparse test set and a dense model swing comparably and the
    comparison is one of *shape*. That is what overlaying is for — and
    it also means a reader cannot see that one set is thirty times the
    other, which is exactly the thing they would want to know. This is
    the sentence that says so.

    Two readings, because the two faults look different:

    - **a constant factor.** Every matched pair off by the same amount
      is a unit or a normalization convention, not a difference in the
      structure. Mode shapes go as one over the square root of mass, so
      a set normalized in grams sits about 31.6 times one normalized in
      kilograms.
    - **a factor that changes mode by mode.** That is not one scaling
      applied wrongly, it is two different normalizations — unit-
      normalized shapes (each mode's peak set to 1) beside
      mass-normalized ones being the usual pair.

    `ratios` is `second / first` per matched pair, measured as the norm
    over the DOFs the two sets share so that a sparse set and a dense
    one are compared on the same footing.
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
        """How much bigger the second set is, typically. NaN with no
        pairs to measure."""
        return (float(np.median(self.ratios)) if self.ratios.size
                else float('nan'))

    @property
    def spread(self) -> float:
        """Largest ratio over smallest: 1.0 is perfectly consistent."""
        usable = self.ratios[np.isfinite(self.ratios) & (self.ratios > 0)]
        if usable.size < 2:
            return 1.0
        return float(usable.max() / usable.min())

    @property
    def consistent(self) -> bool:
        """Do the two sets carry one scaling, whatever it is?"""
        return self.spread <= SCALE_SPREAD

    @property
    def equal(self) -> bool:
        """Are they scaled alike — same convention *and* same size?"""
        factor = self.factor
        return bool(self.consistent and np.isfinite(factor)
                    and abs(factor - 1.0) <= SCALE_TOLERANCE)

    def message(self, first_name: str = 'the first set',
                second_name: str = 'the second set') -> str | None:
        """What to tell the reader, or None when there is nothing to say.

        Deliberately a description and not an accusation: a scale
        difference is sometimes exactly what was intended, and the
        person reading knows which case they are in. What they cannot do
        is see it, because the overlay has normalized it away.
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
    """How much bigger b's mode is than a's, one answer per pair.

    Each ratio is the norm of b's mode over the norm of a's, both
    restricted to the DOFs the two sets share — a sparse test set and a
    dense model would otherwise differ by the square root of the DOF
    count and nothing else.

    `pairs` is [(a's mode, b's mode)]; without it the modes are taken in
    order, as far as the shorter set goes.

    **One entry per pair given, in order**, `None` where there is no
    answer: a mode index past the end of its set, a mode that is zero
    over the shared DOFs (which is a mode that set does not see there),
    or two sets sharing no DOFs at all. Aligned rather than filtered
    because a table puts these in rows beside the pairs that produced
    them, and a shorter list would silently slide up the column.
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

    The set-wide reading of `scale_ratios`: the pairs that have an
    answer, summarized into one factor and a spread. See that function
    for what a ratio is and when there is none.
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

    `shape_matrix` is (modes, dofs). `coordinate` lists the DOF strings the
    columns correspond to ('101X+').

    A fitted set is also the *record of the fit*: reopening one in the
    app (Edit Fit) reconstructs the session that produced it, which is
    why the description and the scaling flag ride along with the numbers.

    Attributes:
        frequency: Hz per mode. A rigid-body mode is exactly 0.0 — the
            FRF synthesis cancels its 0/0 by testing for that, so 'very
            small' is not the same thing.
        damping: Fraction of critical per mode, so 2% is 0.02.
        coordinate: The DOF string of each *column* of `shape_matrix`.
        shape_matrix: `(modes, dofs)`. Complex for a complex mode; the
            overlay and MAC machinery handles either.
        modal_mass: Per mode. 1.0 throughout for a mass-normalized set,
            which is what an eigensolution here produces. Complex when
            an imported source carried complex modal mass — kept as
            measured, never squeezed real.
        modal_damping: Complex modal damping per mode where a source
            carried one (I-DEAS ADFs do), or None. Distinct from
            `damping`, the viscous fraction of critical: this is the
            complex-mode estimate as the identifying tool reported it.
        mass_unit: What `modal_mass` is in, or `None` when undeclared.
        description: Free text per mode — what the shape *is*, filled in
            while reading the table ('first torsion').
        comment: One line about the set as a whole.
        unscaled: True when the fit had no drive point to pin the
            mass-normalized scale. Shapes and MACs are unaffected;
            modal masses are then a convention rather than physics, and
            comparisons refuse to read a scale factor out of them.
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
        """MAC of every mode against every other; the diagonal is 1."""
        return mac_matrix(self.shape_matrix)

    def covers(self, dof: str) -> bool:
        """Does the shape set have a coefficient at this DOF?

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

        H_jk(f) = sum_r (iw)^power phi_jr phi_kr
                        / (m_r (w_r^2 - w^2 + 2i z_r w_r w))
        with w = 2*pi*f — the residue form for mass-normalized shapes, with
        `modal_mass` carrying any other scaling. `power` picks the response
        quantity: 0 displacement per force, 1 velocity, 2 acceleration. It
        applies inside the sum because a rigid-body mode's denominator is
        exactly -w^2: at w = 0 its accelerance cancels to the finite
        residue, where an after-the-fact multiply is 0/0 and a screenful
        of warnings. Its displacement and velocity there are genuinely
        unbounded and come back as nan.

        `modes` restricts the sum; a truncated synthesis beside the
        measurement is what shows which modes the measurement actually
        contains. Raises ValueError for a DOF the shapes do not cover;
        `covers` says so in advance.

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
        """
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
        denominator = ((omega_r ** 2)[:, None] - (omega ** 2)[None, :]
                       + 2j * (self.damping[picked] * omega_r)[:, None]
                       * omega[None, :])
        numerator = (1j * omega) ** power
        # an undamped mode's own resonance line divides by zero too; the
        # infinity is the right answer there, not worth a warning
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = numerator[None, :] / denominator
        rigid, still = omega_r == 0.0, omega == 0.0
        if rigid.any() and still.any():
            ratio[np.ix_(rigid, still)] = 1.0 if power == 2 else np.nan
        out = np.empty((len(response_dof), len(frequencies)),
                       dtype=np.complex128)
        for row, (response, reference) in enumerate(zip(response_dof,
                                                        reference_dof)):
            residues = (shape_at(response) * shape_at(reference)
                        / self.modal_mass[picked])
            out[row] = residues @ ratio
        return out

    def delete_modes(self, indices: Sequence[int]) -> None:
        """Remove the given modes in place; the last one is refused.

        Parameters
        ----------
        indices : sequence of int
            Which modes to remove.

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
        """Whether these are complex modes. Real normal modes move
        every DOF in phase; complex ones do not, which is what a
        damped or non-proportionally damped structure produces."""
        return np.iscomplexobj(self.shape_matrix)

    # ---- units --------------------------------------------------------------

    @property
    def units_defined(self) -> bool:
        """Whether the shapes carry a mass unit, without which a
        modal mass is a number with no scale behind it."""
        return self.mass_unit is not None

    @staticmethod
    def _scale_to_si(mass_unit):
        """Factor taking coefficients in 1/sqrt(mass_unit) to 1/sqrt(kg)."""
        return 1.0 / math.sqrt(si_factor(mass_unit, 'mass'))

    def define_units(self, mass_unit: str) -> ShapeSet:
        """Declare the mass unit the shapes were normalized against.

        Re-declaring reinterprets the file's values rather than scaling
        twice, so a wrong guess can be corrected.

        Parameters
        ----------
        mass_unit : str
            The unit modal mass is in.

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
        """Take the declaration back, restoring the file's raw coefficients."""
        if self.mass_unit is None:
            return self
        self.shape_matrix = self.shape_matrix / self._scale_to_si(self.mass_unit)
        self.mass_unit = None
        return self

    def display_shapes(self, unit_system: UnitSystem) -> np.ndarray:
        """Coefficients in the display system's 1/sqrt(mass); undefined pass
        through unchanged.

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
        """'1/√kg' for the stored unit, or the display system's.

        Parameters
        ----------
        unit_system : UnitSystem, optional
            Units to label in.

        Returns
        -------
        str
            How the shapes' own unit reads.
        """
        if not self.units_defined:
            return ''
        unit = (self.mass_unit if unit_system is None
                else unit_system.unit('mass'))
        return f'1/√{unit}'

    def mode_label(self, i: int) -> str:
        """'Mode 3 — 12.4 Hz, 2.0% damping'.

        Parameters
        ----------
        i : int
            Which mode.

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
        """The MAC grid: this set against itself, or against `other`.

        Parameters
        ----------
        other : ShapeSet, optional
            The set to compare against. This set against itself when
            omitted, which is how repeated modes show up.
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
        """This mode moving on a geometry, as the GUI animates it.

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

    def plot(self, geometry: Geometry | None = None, mode: int = 0,
             **kwargs: Any) -> Any:
        """The set's own reading: its auto-MAC, or one mode animated
        when a geometry says where to put it.

        Parameters
        ----------
        geometry : Geometry, optional
            The geometry to draw on.
        mode : int, default 0
            Which mode.
        **kwargs
            Passed through to the scene.

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
    """The modal model's prediction of `frf`, as an Frf to draw over it.

    Same DOF pairs, same frequencies, residue form over `modes` (all of
    them by default). Each record copies its original's dimension and
    units, so the pair lands on one axis, and is scaled to the
    original's quantity by powers of iω — accelerance assumed when
    nothing said what the measurement is, the modal-test norm. Records
    at DOFs the shapes do not cover are left out; None when that is all
    of them.

    The result carries `synthesized` (drawn dashed) and
    `mode_frequencies` (the fitting screen's bookmarks), so a plot can
    tell it from the measurement wherever it is handed one.
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
