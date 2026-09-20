"""Data arrays: time histories, spectra, FRFs, PSDs.

Shared structure: one common abscissa (time or frequency, SI), an ordinate
matrix of shape (records, samples), a response DOF string per record
('101X+'), an optional reference DOF per record (FRFs, cross spectra), and
per-record unit information.

Units are per record, because one measurement set routinely mixes
quantities — a test's time data may hold accelerations, forces, voltages,
strains and temperatures side by side. Each record carries:

- `ordinate_unit[i]`: the unit its values were declared in, or None when the
  source did not say. `reference_unit[i]` holds an FRF's denominator unit.
- `ordinate_dim[i]`: the dimension tag driving display ('acceleration',
  'acceleration/force', ..., or 'unknown').
- `dimension_hint[i]`: what kind of quantity the source said this was, kept
  only for records whose units are undefined.

A record whose unit is unknown keeps the file's raw numbers untouched;
`define_units()` converts it to SI once and remembers the unit, so a wrong
guess can be corrected later by reinterpreting rather than reimporting.

Knowing what a quantity *is* and knowing what *scale* its numbers are on are
separate things, and a file can tell us one without the other: a UNV with no
dataset 164 still says a channel is an acceleration, but leaves no way to
tell g from m/s**2. Such a record imports undefined — raw values, dimension
'unknown', nothing converted — while `dimension_hint` remembers the claim.
It is advisory only: nothing scales by it. It narrows the choices offered
when units are defined by hand, labels a plot axis, and survives a round
trip back out to a format that can name a quantity without sizing it.

Function type codes follow UFF dataset 58 (also used by sdynpy):
1 time response, 4 FRF, 6 coherence, 9 PSD, 12 spectrum,
24 shock response spectrum, 26 multiple coherence. A specification is
a PSD and a shock specification is an SRS — each is the ordinary thing
with limits on it, and rides a file as its own type.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..units import UnitSystem
    from .averaging import Averaging

from ..units import (
    SI,
    UNKNOWN,
    UnitError,
    dimension_of,
    parse_dimension,
    si_factor,
    si_transform,
)

#: how much of an object's ordinate a display conversion holds at
#: once, in bytes (`DataArray.display_blocks`): a quarter gigabyte is
#: a few hundred records of a long run, and a block's transients —
#: the gather, the scaling, the decimator's padding — are bounded by
#: it whatever the object weighs
DISPLAY_BLOCK_BYTES = 256 * 2 ** 20
from .validate import DIRECTION_CODES
from .validate import dofs as _dofs

# direction code (UFF convention) <-> string. Built from the one list in
# `validate`, so the vocabulary is stated once: a thirteenth direction
# would be a change in one place, and the two cannot disagree.
_DIRECTIONS = {0: '', **{code: text
                         for text, code in DIRECTION_CODES.items()}}


def dof_string(node: int | str, direction_code: int) -> str:
    return f'{node}{_DIRECTIONS.get(int(direction_code), "")}'


_CODES = {text: code for code, text in _DIRECTIONS.items()}


def direction_code(direction: str) -> int:
    """'Z+' -> 3. The inverse of the table `dof_string` reads.

    An unsigned direction means the positive one, which is how DOFs are
    usually written by hand.
    """
    text = str(direction).strip().upper()
    if text in _CODES:
        return _CODES[text]
    if text + '+' in _CODES:
        return _CODES[text + '+']
    raise ValueError(f'unknown direction {direction!r}')


def parse_dof(text: str) -> tuple[int | None, str]:
    """'101RX+' -> (101, 'RX+'). The inverse of dof_string().

    Returns (None, direction) when the leading node number is missing or
    unparseable, so callers can report it rather than crash.
    """
    text = str(text).strip()
    digits = 0
    while digits < len(text) and text[digits].isdigit():
        digits += 1
    if not digits:
        return None, text
    return int(text[:digits]), text[digits:]


#: how much imaginary part, against the real, is a measurement rather
#: than the residue of computing one. A measured autospectrum comes back
#: with an imaginary part about 1e-17 of its real one — the arithmetic,
#: not a phase — where a cross term of the same file is at 5e-2.
REAL_FLOOR = 1e-9


def _even_steps(abscissa, what: str) -> float:
    """The sample step, or a refusal that says which way it is wrong.

    An abscissa is stored as it arrived — uneven spacing and out-of-order
    samples are both real, and the storage refuses neither (see
    `DataArray.abscissa`). What cannot be done on such a record is
    anything with an FFT in it, so the refusal lives here, at the
    operation, and says which of the two problems it found: 'unevenly
    spaced' sent someone looking for a dropped sample once when the
    record was simply in the wrong order.
    """
    steps = np.diff(abscissa)
    if (steps <= 0).any():
        first = int(np.flatnonzero(steps <= 0)[0])
        raise ValueError(
            f'{what} need time samples in order; sample {first + 1} is at '
            f'{float(abscissa[first + 1]):g} s, after {float(abscissa[first]):g}')
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        widest = float(np.abs(steps - steps[0]).max())
        raise ValueError(
            f'{what} need evenly spaced time samples; the step varies by '
            f'{widest:g} s across the record')
    return float(steps[0])


def has_phase(values: ArrayLike) -> bool:
    """Is there imaginary content here, or only the dust of computing it?"""
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        return False
    real = np.abs(values.real).max(initial=0.0)
    return bool(np.abs(values.imag).max(initial=0.0) > REAL_FLOOR * real)


def channel_quantities(dim: str) -> tuple[str, str]:
    """(response, reference) quantities a record's dimension names.

    A record's dimension composes its two channels' quantities — an
    FRF's 'acceleration/force', a CPSD cross term's
    'acceleration*force/frequency', a diagonal's
    'acceleration**2/frequency' — and channel *identity* needs the
    factors, not the compound: keyed by the whole dimension, one
    accelerometer's records split into a row per thing it was measured
    against, which is how a 13-channel CPSD came up 26 rows.
    """
    numerator = dim.partition('/')[0]
    if numerator.endswith('**2'):
        base = numerator.removesuffix('**2')
        return base, base
    response, star, reference = numerator.partition('*')
    if star:
        return response, reference
    denominator = dim.partition('/')[2]
    return numerator, denominator or numerator


#: the viewer's choice of frequency axis for every plot, stage and
#: report figure: True for decades, False for hertz, None to leave each
#: kind of data to its own convention (an SRS in decades, the rest
#: linear). Some readers of a specification want log frequency and some
#: want linear (Brandon, 2026-09-05); this is one switch for all of
#: them, a habit of the viewer rather than a property of the data, so
#: it is not saved with a project and every reader consults it through
#: `DataArray.log_abscissa` rather than being handed it.
_FREQUENCY_AXIS: bool | None = None


def frequency_axis(mode: str | bool | None = ...) -> str:
    """Read or set how frequency axes are drawn: 'log', 'linear', or
    'default' (each kind of data by its own convention).

    Parameters
    ----------
    mode : {'log', 'linear', 'default'} or bool or None, optional
        The choice to make. Omitted, the current one is read back.
        True and False stand for 'log' and 'linear'; None for
        'default'.

    Returns
    -------
    str
        The choice in force after the call.
    """
    global _FREQUENCY_AXIS
    if mode is not ...:
        if mode in ('default', None):
            _FREQUENCY_AXIS = None
        elif mode in ('log', True):
            _FREQUENCY_AXIS = True
        elif mode in ('linear', False):
            _FREQUENCY_AXIS = False
        else:
            raise ValueError(f"frequency axis is 'log', 'linear' or "
                             f"'default', not {mode!r}")
    return {None: 'default', True: 'log', False: 'linear'}[_FREQUENCY_AXIS]


class _LogAbscissa:
    """`log_abscissa` on a class or an instance: the viewer's choice
    when one is made and the data is over frequency, else the class's
    own default. A descriptor rather than a property so `Srs.log_abscissa`
    still reads on the class, as the tests and the stage always have."""

    def __get__(self, obj, cls=None):
        cls = cls if cls is not None else type(obj)
        if _FREQUENCY_AXIS is not None and cls.abscissa_dim == 'frequency':
            return _FREQUENCY_AXIS
        return cls._log_abscissa_default


class DataArray:
    """Base class; use a concrete subclass (TimeHistory, Spectrum, Frf, Psd).

    One object holds **many records** — 36 accelerometer channels, or the
    2 592 FRFs of a 36-by-72 matrix — sharing one abscissa. Everything
    that varies between records is a list of that length, so record *i*
    is `ordinate[i]` measured at `response_dof[i]`, and there is no
    per-record object to go stale.

    Values are stored in **SI** once their units are known. A record
    whose units were never declared keeps the file's raw numbers and
    reports `ordinate_dim == 'unknown'`; what the file *said* it was,
    without saying its scale, is kept beside it in `dimension_hint`.

    Attributes:
        abscissa: The x axis, shared by every record — seconds for a time
            history, hertz for anything in the frequency domain. Stored
            as it arrived: **uneven spacing and out-of-order samples are
            both allowed**, because both are real and refusing them at
            the door would refuse real data. What needs an even step —
            anything with an FFT under it — asks for one at the point of
            use and says so when it cannot have it.
        ordinate: `(records, len(abscissa))`. Complex where the subclass
            says so (`complex_ordinate`), real otherwise.
        response_dof: What each record was measured at, as a DOF string
            (`'101X+'`). One per record.
        reference_dof: What each record was measured *against*, for the
            types that need one — the shaker on an FRF, the other channel
            of a cross spectrum. `None` where the type has no reference.
        block: Which repeat of the same measurement each record is: an
            average, a run, a shock. A short label, not a time.
        ordinate_dim: The quantity each record measures
            (`'acceleration'`, `'force'`), or `'unknown'` while its units
            are undeclared.
        ordinate_unit: The unit its values are in — always the SI one
            while the dimension is known, since that is how they are
            stored. `None` means undeclared.
        reference_unit: The same for the reference of a ratio, so an FRF
            record knows both halves of m/s²/N.
        dimension_hint: What the file claimed a record measures without
            saying at what scale. Nothing is ever scaled by a hint: it
            narrows the units offered, labels an axis, and survives
            export.
        comment: Free text per record, as the source file carried it.
    """

    function_type = 0
    abscissa_dim = 'time'
    complex_ordinate = False
    needs_reference = False
    # a frequency-domain magnitude spans decades and reads on a log axis; a
    # bounded ratio does not, which is what these two are for
    log_ordinate = None        # None: logarithmic iff the abscissa is frequency
    #: whether the *abscissa* reads on a log axis, by default. Most
    #: frequency data here is drawn linear in frequency (a PSD's bins,
    #: an FRF's lines); an SRS is the exception — its abscissa is
    #: oscillator natural frequency laid out in octaves by convention,
    #: and every reader of one expects decades (Brandon, 2026-08-25, on
    #: reviewer feedback). `log_abscissa` is what the plot, the stage
    #: and the report read; it answers with this default unless the
    #: viewer has chosen otherwise (`frequency_axis`).
    _log_abscissa_default = False
    log_abscissa = _LogAbscissa()
    ordinate_limits = None     # (low, high) to pin the axis to

    def __init__(self, abscissa: ArrayLike, ordinate: ArrayLike,
                 response_dof: str | Sequence[str],
                 reference_dof: str | Sequence[str] | None = None,
                 ordinate_dim: str | Sequence[str] | None = None,
                 comment: str | Sequence[str] | None = None,
                 ordinate_unit: str | Sequence[str | None] | None = None,
                 reference_unit: str | Sequence[str | None] | None = None,
                 dimension_hint: str | Sequence[str | None] | None = None,
                 block: str | Sequence[str] | None = None) -> None:
        self.abscissa: np.ndarray = np.asarray(abscissa, dtype=np.float64)
        ordinate = np.atleast_2d(np.asarray(
            ordinate, dtype=np.complex128 if self.complex_ordinate else np.float64))
        self.ordinate: np.ndarray = ordinate
        n = ordinate.shape[0]
        if self.abscissa.ndim != 1 or ordinate.shape[1] != len(self.abscissa):
            raise ValueError(
                f'ordinate shape {ordinate.shape} does not match '
                f'abscissa length {len(self.abscissa)}')
        # a channel whose node or direction was never recorded imports
        # as a DOF of '' — see `validate.dofs`
        self.response_dof: list[str] = _dofs(
            self._str_list(response_dof, n, 'response_dof'), 'response DOF',
            allow_unknown=True)
        self.reference_dof: list[str] | None
        if reference_dof is None:
            if self.needs_reference:
                raise ValueError(f'{type(self).__name__} requires reference_dof')
            self.reference_dof = None
        else:
            self.reference_dof = _dofs(
                self._str_list(reference_dof, n, 'reference_dof'),
                'reference DOF', allow_unknown=True)
        # A record is its response plus, sometimes, one more thing. For a
        # measurement against something else that is the reference DOF; for
        # the same measurement repeated it is which repeat — an average, a
        # run, a temperature. `block` holds the short label of that second
        # kind, and nothing about it is time-specific.
        self.block: list[str] | None = (None if block is None
                                        else self._str_list(block, n, 'block'))
        self.ordinate_dim: list[str] = self._str_list(
            UNKNOWN if ordinate_dim is None else ordinate_dim, n, 'ordinate_dim')
        for dim in set(self.ordinate_dim):
            parse_dimension(dim)  # validates
        self.ordinate_unit: list[str | None] = self._opt_list(
            ordinate_unit, n, 'ordinate_unit')
        self.reference_unit: list[str | None] = self._opt_list(
            reference_unit, n, 'reference_unit')
        self.comment: list[str] = self._str_list(
            comment if comment is not None else '', n, 'comment')
        self.dimension_hint: list[str | None] = self._opt_list(
            dimension_hint, n, 'dimension_hint')
        for hint in set(self.dimension_hint):
            if hint is not None:
                parse_dimension(hint)  # validates
        self._fill_si_units()
        self._drop_stale_hints()

    def _fill_si_units(self):
        """Name the unit of any record whose dimension is known.

        Values are stored in SI, so a record with a known dimension and no
        recorded unit is in that dimension's SI unit. Filling it keeps the
        invariant 'no unit recorded' == 'units undefined', which the GUI
        relies on for its labels and badges.
        """
        for i, dim in enumerate(self.ordinate_dim):
            if dim == UNKNOWN or self.ordinate_unit[i] is not None:
                continue
            unit, reference = self._si_units_for(dim)
            self.ordinate_unit[i] = unit
            if self.reference_unit[i] is None:
                self.reference_unit[i] = reference

    def _drop_stale_hints(self):
        """Forget a source's claim that is redundant or says nothing.

        A hint is what we fall back on; keeping one beside a defined record
        would leave two answers to the same question, free to disagree. A
        hint of 'unknown' is not a claim at all — a file saying it does not
        know is the same as a file that never said.
        """
        for i, dim in enumerate(self.ordinate_dim):
            if dim != UNKNOWN or self.dimension_hint[i] == UNKNOWN:
                self.dimension_hint[i] = None

    def known_dim(self, i: int) -> str:
        """What quantity record `i` holds, whether or not its unit is known.

        The dimension when the units are defined, otherwise the source's
        claim, otherwise 'unknown'. Never use this to scale anything — a
        hinted record's values are raw, and the scale is exactly what is
        missing.

        Parameters
        ----------
        i : int
            Which record.

        Returns
        -------
        str
            The quantity the record holds, falling back to its
            dimension hint when the unit is undeclared.
        """
        if self.ordinate_dim[i] != UNKNOWN:
            return self.ordinate_dim[i]
        return self.dimension_hint[i] or UNKNOWN

    @classmethod
    def _si_units_for(cls, dimension):
        """(ordinate unit, reference unit) in SI for a known dimension."""
        return SI.unit(dimension), None

    @staticmethod
    def _str_list(value, n, name):
        if isinstance(value, str):
            return [value] * n
        out = [str(v) for v in value]
        if len(out) != n:
            raise ValueError(f'{name} has {len(out)} entries, expected {n}')
        return out

    @staticmethod
    def _opt_list(value, n, name):
        """List of optional strings; None means 'not declared'."""
        if value is None:
            return [None] * n
        if isinstance(value, str):
            return [value] * n
        out = [None if v is None or v == '' else str(v) for v in value]
        if len(out) != n:
            raise ValueError(f'{name} has {len(out)} entries, expected {n}')
        return out

    def rename_dof(self, old: str, new: str,
                   quantity: str | None = None) -> int:
        """Give a channel's coordinate a new name, in place.

        A channel is a coordinate *and* a quantity — a drive point
        carries a load cell and an accelerometer at one DOF — and the
        rename is the channel's: a force labeled at the wrong node
        moves without taking the accelerometer at that node with it,
        and the other is changed explicitly if it should be (Brandon,
        2026-09-06). The channel moves wherever a record wears it, as
        a response and as a reference alike: a CPSD's accelerometer is
        on both sides of its cross terms and is one sensor. A rename
        that would give two records one identity — two accelerometers
        at one point — is refused, where it would have made two rows of
        the grid into one and hidden a record.

        Parameters
        ----------
        old : str
            The coordinate as it is, '101Z+'.
        new : str
            The coordinate to give it; normalized the way every DOF is
            ('101Z' is '101Z+'), and refused when it is not one.
        quantity : str, optional
            Which channel at `old`: 'acceleration', 'force' … as
            `channel_quantities` names a record's factors. Every
            quantity at the coordinate when omitted — the point rather
            than the channel, said explicitly.

        Returns
        -------
        int
            How many records changed, counting a response and a
            reference on one record separately.
        """
        old = str(old).strip()
        (new,) = _dofs([str(new)], 'DOF')
        if new == old:
            return 0
        factors = [channel_quantities(self.known_dim(i))
                   for i in range(self.num_records)]
        rows = [i for i, dof in enumerate(self.response_dof)
                if dof == old and quantity in (None, factors[i][0])]
        columns = [i for i, dof in enumerate(self.reference_dof or [])
                   if dof == old and quantity in (None, factors[i][1])]
        if not rows and not columns:
            raise ValueError(f'no {quantity + " " if quantity else ""}record '
                             f'at {old!r}')
        moving = {factors[i][0] for i in rows}
        for i, dof in enumerate(self.response_dof):
            if dof == new and factors[i][0] in moving:
                raise ValueError(
                    f'{new} already has a {factors[i][0]} record — two '
                    'channels cannot share a coordinate and a quantity')
        for i in rows:
            self.response_dof[i] = new
        for i in columns:
            self.reference_dof[i] = new
        return len(rows) + len(columns)

    def delete_records(self, indices: Sequence[int] | None = None, *,
                       dof: str | Sequence[str] | None = None,
                       dim: str | Sequence[str] | None = None,
                       reference: str | Sequence[str] | None = None,
                       capture: int | Sequence[int] | None = None) -> None:
        """Remove records in place — by index, by DOF, or by capture.

        Everything a record owns goes with it: its row of the ordinate, its
        DOFs, units, comment, hint, block — and, on a specification, its
        limit curves, which would otherwise silently belong to the wrong
        channels. Removing the last record is refused: an empty data array
        is not a state anything else here can show.

        `dof` and `capture` are the selectors a person means — "drop
        channel 101Z+", "drop the third run" — where indices are the
        machine's (Brandon, 2026-08-30, reading thirteen indices in the
        journal where one capture number would have said it). They
        combine as an intersection, and either combines with explicit
        indices as a union.

        Parameters
        ----------
        indices : sequence of int, optional
            Which records to remove, by position.
        dof : str or sequence of str, optional
            Remove every record at these response DOFs. A drive point
            carries two records at one DOF — a force and an
            acceleration — and the DOF alone takes both; `dim` is how
            the finer thing is said.
        dim : str or sequence of str, optional
            Restrict to these quantities ('force', 'acceleration', …)
            — the other half of a channel's identity.
        reference : str or sequence of str, optional
            Remove every record at these reference DOFs — a column of
            an FRF matrix, where `dof` is a row.
        capture : int or sequence of int, optional
            Remove these captures — each channel's n-th playing, the
            numbering `capture_indices` gives. Time histories only.

        Returns
        -------
        None
        """
        doomed = {int(i) for i in indices} if indices is not None else set()
        if dof is not None or dim is not None or reference is not None \
                or capture is not None:
            wanted_dofs = ({dof} if isinstance(dof, str)
                           else None if dof is None else {str(d) for d in dof})
            wanted_dims = ({dim} if isinstance(dim, str)
                           else None if dim is None else {str(d) for d in dim})
            wanted_references = None
            if reference is not None:
                if self.reference_dof is None:
                    raise ValueError(
                        'references are a matrix\'s; this '
                        f'{type(self).__name__} has none')
                wanted_references = ({reference}
                                     if isinstance(reference, str)
                                     else {str(r) for r in reference})
            wanted_captures = None
            if capture is not None:
                ordinals = getattr(self, 'capture_indices', None)
                if ordinals is None:
                    raise ValueError(
                        'captures are a time history\'s; this is a '
                        f'{type(self).__name__}')
                ordinals = ordinals()
                wanted_captures = ({int(capture)}
                                   if isinstance(capture, (int, np.integer))
                                   else {int(c) for c in capture})
            chosen = []
            for i in range(self.num_records):
                if wanted_dofs is not None \
                        and self.response_dof[i] not in wanted_dofs:
                    continue
                if wanted_dims is not None \
                        and self.ordinate_dim[i] not in wanted_dims:
                    continue
                if wanted_references is not None \
                        and self.reference_dof[i] not in wanted_references:
                    continue
                if wanted_captures is not None \
                        and ordinals[i] not in wanted_captures:
                    continue
                chosen.append(i)
            if not chosen:
                raise ValueError('nothing matches that selection')
            doomed |= set(chosen)
        bad = [i for i in doomed if not 0 <= i < self.num_records]
        if bad:
            raise IndexError(f'no such records: {sorted(bad)}')
        keep = [i for i in range(self.num_records) if i not in doomed]
        if not keep:
            raise ValueError('cannot delete every record; delete the '
                             'object instead')
        self.ordinate = self.ordinate[keep]
        for name in ('response_dof', 'reference_dof', 'block',
                     'ordinate_dim', 'ordinate_unit', 'reference_unit',
                     'comment', 'dimension_hint'):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, [values[i] for i in keep])
        for name, values in getattr(self, 'limits', {}).items():
            self.limits[name] = values[keep]
        if self.block is not None:
            # repeated-measurement labels renumber to contiguous —
            # 'avg 1, avg 3' after deleting the second capture is a
            # gap the tree's columns would faithfully show (Brandon,
            # 2026-08-30) — by the merge rule, which is the one
            # implementation of what a block label is: bookkeeping,
            # not identity. Named blocks (an exodus 'KE') survive.
            from .merge import _merged_blocks

            self.block = _merged_blocks([self])

    @property
    def num_records(self) -> int:
        """How many records this object holds. One measurement per
        record, all sharing the object's single abscissa."""
        return self.ordinate.shape[0]

    # ---- units --------------------------------------------------------------

    @property
    def units_defined(self) -> bool:
        """Whether every record knows what it measures. False while
        any record is still in the file's own unconverted numbers."""
        return all(dim != UNKNOWN for dim in self.ordinate_dim)

    @property
    def undefined_records(self) -> list[int]:
        """Which records still have no declared dimension — the ones
        holding the file's raw numbers, awaiting `define_units`."""
        return [i for i, dim in enumerate(self.ordinate_dim) if dim == UNKNOWN]

    def _conversion(self, unit, reference_unit=None):
        """(scale, offset, dimension) taking declared values to SI."""
        dim = dimension_of(unit)
        if dim is None:
            raise UnitError(f"Cannot determine the physical dimension of {unit!r}")
        scale, offset = si_transform(unit)
        return scale, offset, dim

    def define_units(self, units: str | Sequence[str | None],
                     reference_units: str | Sequence[str | None] | None
                     = None) -> DataArray:
        """Declare what the ordinate values are in, converting them to SI.

        `units` is a single unit applied to every record, a sequence with one
        entry per record, or a {record index: unit} mapping to set only some.
        Entries of None leave a record's units undefined. Records that already
        have units are reinterpreted, not re-scaled twice.

        Parameters
        ----------
        units : str or sequence of str
            The unit each record's values are in; one string applies to
            every record.
        reference_units : str or sequence of str, optional
            The denominator unit, for records that have one.

        Returns
        -------
        DataArray
            Self, converted to SI in place.
        """
        specs = self._expand(units, 'units')
        refs = self._expand(reference_units, 'reference_units')
        for i, unit in enumerate(specs):
            if unit is None:
                continue
            reference = refs[i]
            scale, offset, dim = self._conversion(unit, reference)
            old_scale, old_offset = self._current_transform(i)
            for values in self._value_arrays():
                raw = (values[i] - old_offset) / old_scale
                values[i] = raw * scale + offset
            self.ordinate_dim[i] = dim
            self.ordinate_unit[i] = unit
            self.reference_unit[i] = reference
        self._drop_stale_hints()
        return self

    def undefine_units(self, records: Sequence[int] | None = None) -> DataArray:
        """Take a declaration back, restoring the file's raw values.

        The inverse of `define_units`. A wrong guess should be correctable
        without reimporting, and that means being able to withdraw one, not
        only to replace it — there is no unit string meaning 'I no longer
        know'.

        Parameters
        ----------
        records : sequence of int, optional
            Which records to revert. All of them when omitted.

        Returns
        -------
        DataArray
            Self, with the file's raw values restored.
        """
        for i in (range(self.num_records) if records is None
                  else [int(r) for r in records]):
            if self.ordinate_unit[i] is None:
                continue
            old_scale, old_offset = self._current_transform(i)
            for values in self._value_arrays():
                values[i] = (values[i] - old_offset) / old_scale
            self.ordinate_dim[i] = UNKNOWN
            self.ordinate_unit[i] = None
            self.reference_unit[i] = None
        return self

    def _value_arrays(self):
        """Every array held in the ordinate's units, row for row.

        A subclass storing more curves of the same quantity — a
        specification's limit lines — lists them here, and every conversion
        the ordinate gets they get too. Nothing else has to know.
        """
        return (self.ordinate,)

    def _plain_kwargs(self):
        """What else a copy of this record has to be told about itself.

        Empty for most kinds: an FRF is an FRF. An SRS is not an SRS
        without the Q it was worked out at, so it lists that here and
        anything rebuilding one of its curves carries it along.
        """
        return {}

    def _current_transform(self, i):
        """(scale, offset) already applied to record `i`, or (1, 0) if raw."""
        if self.ordinate_unit[i] is None:
            return 1.0, 0.0
        scale, offset, _ = self._conversion(self.ordinate_unit[i],
                                            self.reference_unit[i])
        return scale, offset

    def _expand(self, value, name):
        n = self.num_records
        if value is None:
            return [None] * n
        if isinstance(value, str):
            return [value] * n
        if isinstance(value, dict):
            out = [None] * n
            for index, unit in value.items():
                out[int(index)] = unit
            return out
        out = list(value)
        if len(out) != n:
            raise ValueError(f'{name} has {len(out)} entries, expected {n}')
        return out

    def _raw_record(self, i):
        """Record `i` back in the units it was declared in (raw file values)."""
        if self.ordinate_unit[i] is None:
            return self.ordinate[i]
        scale, offset, _ = self._conversion(self.ordinate_unit[i],
                                            self.reference_unit[i])
        return (self.ordinate[i] - offset) / scale

    def column_keys(self) -> list[str] | None:
        """What tells one record from another besides its response.

        The reference DOF when records are a matrix of measurements, the
        block when they are the same measurement repeated, None when the
        response alone is the whole identity. This is what decides whether
        an object expands into a grid.
        """
        return self.reference_dof if self.reference_dof is not None else self.block

    def record_label(self, i: int) -> str:
        """A short label for one record, for a legend or an axis.

        Parameters
        ----------
        i : int
            Which record.

        Returns
        -------
        str
            The record's DOF, plus whatever tells it from its neighbors.
        """
        if self.reference_dof is not None:
            return f'{self.response_dof[i]}/{self.reference_dof[i]}'
        if self.block is not None:
            # a space, not a slash: this is not a ratio of two DOFs —
            # stripped, because either half may be empty: an exodus
            # global variable is all block and no DOF, and ' KE' with a
            # ghost space is not a label anyone typed
            return f'{self.response_dof[i]} {self.block[i]}'.strip()
        return self.response_dof[i]

    def record_pair(self, i: int) -> tuple[str, str]:
        """The (response, reference) record `i` is between.

        A record with no reference is an autospectrum — a channel
        against itself — so it pairs with the diagonal, which is what
        a specification bounds. The one reading of that rule: the
        plot, the table beside it and the report all ask here, so
        their labels cannot disagree.

        Parameters
        ----------
        i : int
            The record.

        Returns
        -------
        tuple of str
            Response DOF, reference DOF.
        """
        response = self.response_dof[i]
        reference = (self.reference_dof[i]
                     if self.reference_dof is not None else response)
        return response, reference

    def log_scaled(self) -> bool:
        """Whether this object's magnitude reads on a log axis.

        Logarithmic for frequency-domain data unless the class pins it
        (`log_ordinate` — a coherence is a 0..1 ratio and says nothing on
        a log axis). The object answers so the 2-D plot and the 3-D
        waterfall read one rule and cannot disagree about its axis.
        """
        return (self.abscissa_dim == 'frequency' if self.log_ordinate is None
                else bool(self.log_ordinate))

    def display_abscissa(self, unit_system: UnitSystem) -> np.ndarray:
        """The abscissa converted into a unit system's own units.

        Parameters
        ----------
        unit_system : UnitSystem
            The units to present in.

        Returns
        -------
        numpy.ndarray
            The abscissa in display units.
        """
        return unit_system.from_si(self.abscissa, self.abscissa_dim)

    def display_ordinate(self, unit_system: UnitSystem,
                         records: Iterable[int] | None = None
                         ) -> np.ndarray:
        """Ordinate in display units; undefined records pass through as-is.

        `records` limits the work to the ones asked for. A 1356-record FRF
        has one or two distinct dimensions in it, so the conversion is
        gathered per dimension and applied to a whole block at once —
        converting row by row meant a unit lookup per record, which is how
        drawing a single curve came to cost 2713 trips through pint.

        Parameters
        ----------
        unit_system : UnitSystem
            The units to present the values in.
        records : iterable of int, optional
            Which records to convert. All of them when omitted.

        Returns
        -------
        numpy.ndarray
            The values in display units. Records with undefined
            units pass through untouched.
        """
        rows = range(self.num_records) if records is None else list(records)
        out = np.empty((len(rows), self.ordinate.shape[1]),
                       dtype=self.ordinate.dtype)
        # converted a block of records at a time into slices of `out`:
        # gathering every asked-for record and scaling the gather were
        # two whole copies beside the answer, and on a 1.2 GB history
        # the conversion alone peaked at three times its output
        # (measured 2026-09-18, on the way to a 22 GB import that took
        # the application down at 67 GB)
        for start, stop, block in self.display_blocks(unit_system, rows):
            out[start:stop] = block
        return out

    def display_blocks(self, unit_system: UnitSystem,
                       records: Iterable[int] | None = None,
                       block_bytes: int | None = None):
        """`display_ordinate` a block of records at a time.

        Yields `(start, stop, values)`: positions within `records` and
        the converted rows for them, each block at most `block_bytes`
        (`DISPLAY_BLOCK_BYTES` by default) — so a reading that thins
        every record before it draws it, the 3-D stage above all,
        never holds a converted copy of the whole object. A record
        larger than the block is still one block: a row is not split.

        Parameters
        ----------
        unit_system : UnitSystem
            The units to present the values in.
        records : iterable of int, optional
            Which records to convert. All of them when omitted.
        block_bytes : int, optional
            The most a block may hold, in bytes.

        Yields
        ------
        tuple of (int, int, numpy.ndarray)
            The block's positions among `records`, and its values in
            display units.
        """
        rows = range(self.num_records) if records is None else list(records)
        limit = DISPLAY_BLOCK_BYTES if block_bytes is None else int(block_bytes)
        row_bytes = max(1, self.ordinate.shape[1] * self.ordinate.dtype.itemsize)
        per_block = max(1, limit // row_bytes)
        for start in range(0, len(rows), per_block):
            chunk = rows[start:start + per_block]
            block = np.empty((len(chunk), self.ordinate.shape[1]),
                             dtype=self.ordinate.dtype)
            by_dimension = {}
            for position, record in enumerate(chunk):
                by_dimension.setdefault(self.ordinate_dim[record], []).append(
                    (position, record))
            for dim, pairs in by_dimension.items():
                positions = [position for position, _ in pairs]
                source = self.ordinate[[record for _, record in pairs]]
                block[positions] = (source if dim == UNKNOWN
                                    else unit_system.from_si(source, dim))
            yield start, start + len(chunk), block

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        refs = ((self.reference_dof is None and other.reference_dof is None)
                or (self.reference_dof is not None
                    and other.reference_dof is not None
                    and self.reference_dof == other.reference_dof))
        return (np.allclose(self.abscissa, other.abscissa)
                and self.ordinate.shape == other.ordinate.shape
                and np.allclose(self.ordinate, other.ordinate)
                and self.response_dof == other.response_dof
                and self.block == other.block
                and refs
                and self.ordinate_dim == other.ordinate_dim
                and self.ordinate_unit == other.ordinate_unit
                and self.reference_unit == other.reference_unit
                and self.dimension_hint == other.dimension_hint
                and self.comment == other.comment)

    def __repr__(self) -> str:
        dims = sorted(set(self.ordinate_dim))
        return (f'{type(self).__name__}({self.num_records} records x '
                f'{len(self.abscissa)} samples, {", ".join(dims)})')

    def save(self, path: str | os.PathLike) -> None:
        """Write this object to a `.vdyn` file of its own.

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

    def plot(self, unit_system: UnitSystem | None = None,
             **kwargs: Any) -> Any:
        """Draw every record on one set of axes.

        Parameters
        ----------
        unit_system : UnitSystem, optional
            Units to draw in.
        **kwargs
            Passed through to the plotting layer.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..plot import plot_data
        return plot_data(self, unit_system=unit_system, **kwargs)

    def save_plot(self, path: str | os.PathLike,
                  unit_system: UnitSystem | None = None,
                  **kwargs: Any) -> Any:
        """Draw the records and write the figure to `path`.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write the image.
        unit_system : UnitSystem, optional
            Units to draw in.
        **kwargs
            Passed through to the plotting layer.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..plot import save_plot
        return save_plot(self, path, unit_system=unit_system, **kwargs)

    def plot_waterfall(self, records: Sequence[int] | None = None,
                       **kwargs: Any) -> Any:
        """The records spread along a depth axis, colored by level —
        the plot bar's 3-D reading, scripted. `screenshot=` renders
        headless to a file; without it a window of the app's own 3-D
        pane opens.

        Parameters
        ----------
        records : sequence of int, optional
            Which records to stage. All of them when omitted.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..viz.waterfall import plot_waterfall
        return plot_waterfall(self, records, **kwargs)


class TimeHistory(DataArray):
    """A measurement against time: the record as it was acquired.

    Everything else in a random or shock test is derived from one of
    these, and the deriving is here — spectra, PSDs, the full CPSD
    matrix, multiple coherence, shock response spectra. Two things ride
    along that say *how* to read it: `averaging`, the frames a spectrum
    is averaged over, and `shocks`, the events an SRS is computed from.
    Both are the app's two views of a trace, and both are stored on the
    history rather than passed at the call, so a PSD and the coherence
    beside it cannot describe different measurements.
    """

    function_type = 1
    abscissa_dim = 'time'
    complex_ordinate = False

    def compute_spectra(self) -> Spectrum:
        """The averaged spectrum of every channel — sdynpy's convention.

        A channel's frames — its records across the averages — are each
        FFT'd single-sided with no amplitude scaling (numpy's rfft,
        norm='backward': a sine of amplitude A on a bin reads A*N/2),
        rectangular window, and averaged as the *complex mean*, exactly
        what sdynpy's `TimeHistoryArray.fft` does with frames. Phase is
        preserved; content whose phase is random frame to frame — a
        burst random excitation's response — averages toward zero,
        which is that convention's documented behavior. Returns a
        Spectrum with one record per channel, carrying the channel's
        own quantity and units.
        """
        if len(self.abscissa) < 2:
            raise ValueError('Spectra need at least two time samples')
        step = _even_steps(self.abscissa, 'Spectra')
        groups = {}
        for i, dof in enumerate(self.response_dof):
            key = (dof, self.ordinate_dim[i], self.ordinate_unit[i],
                   self.dimension_hint[i])
            groups.setdefault(key, []).append(i)
        frequencies = np.fft.rfftfreq(len(self.abscissa), step)
        rows, dofs, dims, units, hints = [], [], [], [], []
        for (dof, dim, unit, hint), indices in groups.items():
            frames = np.fft.rfft(self.ordinate[indices], axis=1)
            rows.append(np.mean(frames, axis=0))
            dofs.append(dof)
            dims.append(dim)
            units.append(unit)
            hints.append(hint)
        return Spectrum(frequencies,
                        np.asarray(rows, dtype=np.complex128),
                        response_dof=dofs, ordinate_dim=dims,
                        ordinate_unit=units, dimension_hint=hints,
                        comment='averaged spectrum')

    @property
    def sample_rate(self) -> float:
        """Samples per second, from the abscissa — which must be even."""
        if len(self.abscissa) < 2:
            raise ValueError('a sample rate needs at least two time samples')
        return 1.0 / _even_steps(self.abscissa, 'a sample rate')

    def channel_key(self, i: int) -> tuple[str, str, str | None]:
        """What makes record `i` the same channel as another.

        The DOF and what is measured there, never the DOF alone: a drive
        point carries a force record and an acceleration record at the
        same DOF, and those two are not each other's averages. This is
        the key the framing groups on, so counting channels and
        averaging them cannot disagree about what a channel is.

        Parameters
        ----------
        i : int
            Which record.

        Returns
        -------
        tuple of (str, str, str or None)
            The DOF, quantity and unit that identify the channel —
            records sharing this key are the same channel.
        """
        return (self.response_dof[i], self.ordinate_dim[i],
                self.dimension_hint[i])

    @property
    def records_per_channel(self) -> dict[tuple[str, str, str | None], int]:
        """{channel key: how many records carry it}.

        Usually one. A capture a controller saved frame by frame holds
        one record per average.
        """
        counts = {}
        for i in range(len(self.response_dof)):
            key = self.channel_key(i)
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def split_into_frames(self) -> bool:
        """Whether the records are already the averages.

        A controller that saves its spectral captures writes each frame
        as its own record. There is then nothing to slice and nothing to
        overlap: the frame length and the count are settled by the file,
        and the only parameter left to choose is the window.
        """
        averaging = self.averaging
        return (averaging is not None
                and averaging.frames == 1
                and averaging.frame_length == len(self.abscissa)
                and max(self.records_per_channel.values(), default=0) > 1)

    @property
    def average_counts(self) -> tuple[int, int]:
        """(fewest, most) frames any one channel will be averaged over.

        The two agree for anything a controller wrote, which saves every
        channel the same number of times. They part only for a history
        assembled by hand out of unequal captures, and then the table
        has to say so rather than quote a number that is true of some
        channels and not others.
        """
        counts = self.records_per_channel
        if not counts:
            return (0, 0)
        per_record = 1 if self.averaging is None else self.averaging.frames
        return (min(counts.values()) * per_record,
                max(counts.values()) * per_record)

    def suggest_averaging(self, **kwargs: Any) -> Averaging:
        """Averaging parameters worked out from the record itself.

        A run holds more than the test — the shaker coming up, a
        reduced-level check, whatever was still recording afterwards —
        and this finds the settled stretch worth averaging and as many
        frames as it will carry. See `visualdynamics.core.detect`.

        A record that already carries an averaging — a controller's own
        recipe from an import, or one set by hand — keeps its frame
        length, window, overlap and detrend: only the start and the
        count are worked out (Brandon, 2026-09-19: Detect must not
        override the frame length, window and overlap the controller
        used). A bare record gets the detector's own recipe.

        Parameters
        ----------
        **kwargs
            Overrides for individual parameters, such as `window`.

        Returns
        -------
        Averaging
            Parameters worked out from the record itself.
        """
        from dataclasses import replace

        from .detect import suggest

        own = self.averaging
        if own is not None:
            kwargs.setdefault('frame_length', own.frame_length)
            kwargs.setdefault('window', own.window)
            kwargs.setdefault('overlap', own.overlap)
        found = suggest(self, **kwargs)
        if own is not None:
            found = replace(found, detrend=own.detrend,
                            window_parameter=own.window_parameter)
        return found

    def _spectral_frame(self, averaging=None):
        """(frequencies, one-sided scale, channel -> windowed frames).

        The framing a PSD and a CPSD share: the same rfft grid, the same
        DC-and-Nyquist-unhalved scaling, and the same cutting of a
        channel into frames, so the two cannot drift apart in what they
        mean by an average.

        Without an `averaging` each record is its own frame, whole and
        unwindowed — a burst-random run saved as 20 captures is already
        20 averages, and always was. With one, every record is cut into
        `frames` frames from `start`, each windowed, and all of them are
        pooled: 20 captures cut three ways average 60 frames.

        The window comes into the scaling as the sum of its squares,
        which is `n` for a rectangle — so the two paths agree exactly
        where they overlap.
        """
        from .averaging import Averaging

        if averaging is None:
            averaging = self.averaging
        if len(self.abscissa) < 2:
            raise ValueError('PSDs need at least two time samples')
        step = _even_steps(self.abscissa, 'PSDs')
        fs = 1.0 / step
        samples = len(self.abscissa)
        if averaging is None:
            averaging = Averaging.for_records(samples)
        if not averaging.fits(samples, fs):
            most = averaging.most_frames(samples, fs)
            raise ValueError(
                f'{averaging.frames} frames of {averaging.frame_length} '
                f'samples from {averaging.start:g} s do not fit in '
                f'{samples} samples — room for {most}')
        rows = {}
        for i in range(len(self.response_dof)):
            rows.setdefault(self.channel_key(i), []).append(i)
        n = averaging.frame_length
        shape = averaging.shape()
        first = averaging.start_sample(fs)
        groups = {}
        for key, indices in rows.items():
            frames = []
            for i in indices:
                for f in range(averaging.frames):
                    at = first + f * averaging.hop
                    frame = self.ordinate[i][at:at + n]
                    # leveled before it is windowed, exactly where
                    # scipy's welch does it; 'none' is the default and
                    # the sdynpy convention, which the oracle pins
                    if averaging.detrend == 'mean':
                        frame = frame - frame.mean()
                    elif averaging.detrend == 'linear':
                        from scipy.signal import detrend as _detrend

                        frame = _detrend(frame, type='linear')
                    frames.append(frame * shape)
            groups[key] = np.asarray(frames)
        frequencies = np.fft.rfftfreq(n, step)
        # Welch's scaling with a window: the sum of its squares stands in
        # for the frame length, and is the frame length for a rectangle
        power = float(np.sum(shape ** 2))
        scale = np.full(len(frequencies), 2.0 / (fs * power))
        scale[0] = 1.0 / (fs * power)
        if n % 2 == 0:
            scale[-1] = 1.0 / (fs * power)
        return frequencies, scale, groups

    def capture_indices(self) -> list[int]:
        """Which playing each record is: 0 for a channel's first
        record, 1 for its second, and so on — in exactly the order
        `_spectral_frame` pools them, so the playing the averaging
        view pages to is one of the playings the average adds up. A
        channel is a `channel_key` group, the same grouping the
        pooling uses.
        """
        seen: dict = {}
        out = []
        for i in range(len(self.response_dof)):
            key = self.channel_key(i)
            out.append(seen.get(key, 0))
            seen[key] = seen.get(key, 0) + 1
        return out

    #: how this history should be cut into frames, when something knows.
    #: Set from the file on import and edited in the averaging view; its
    #: Compute buttons use whatever is here at the time.
    averaging = None

    #: which stretches of this history are the shocks, when something
    #: knows. Set by the detector or by the shock view, and used by
    #: Compute SRS the way `averaging` is — the parameters live on the
    #: object, not in the call.
    shocks = None

    #: the stretch this history would be cut to, when something
    #: knows. Set in the truncate view; Truncate Data cuts with
    #: whatever is here at the time.
    truncation = None

    def suggest_truncation(self) -> Any:
        """The whole record — the only neutral span.

        A starting point for the truncate view's handles, never a
        default the act adopts: keeping everything is not an
        act, so Truncate Data refuses until a real span is set.
        """
        from .truncate import Truncation

        abscissa = np.asarray(self.abscissa, dtype=float)
        return Truncation(float(abscissa[0]), float(abscissa[-1]))

    def truncate(self, truncation: Any = None) -> TimeHistory:
        """This record cut to a span (`core.truncate.truncate`), using
        the history's own `truncation` unless one is passed.

        Parameters
        ----------
        truncation : Truncation, optional
            The start and stop, in seconds on the record's own clock.
            Defaults to the record's own `truncation`.

        Returns
        -------
        TimeHistory
            The samples inside the span, every channel, the clock
            kept.
        """
        from .truncate import truncate

        chosen = self.truncation if truncation is None else truncation
        if chosen is None:
            raise ValueError('no span to keep: set `truncation` first, '
                             'or pass one')
        return truncate(self, chosen)

    #: the filter this history is read through, when something knows.
    #: Set in the filter view; Filter Data computes with whatever is
    #: here at the time, the way the Compute buttons read `averaging`.
    filtering = None

    def suggest_filtering(self) -> Any:
        """A starting low-pass: a tenth of the sample rate, order 4.

        A low-pass rather than any other kind, because cutting noise
        above the content is the reach-for-first case; the filter view
        offers high- and band-pass beside it.

        A judgment, not a detection — nothing in the record says where
        its content stops being signal. A tenth of the rate is where
        the integrate/differentiate round trip was measured at ~2% RMS
        (core.filters), and it sits below the mounted-resonance range a
        shock accelerometer pollutes. The filter view exists precisely
        so this number gets looked at rather than trusted.
        """
        from .filters import Filtering

        return Filtering(high=self.sample_rate / 10.0)

    def filter(self, filtering: Any = None) -> TimeHistory:
        """This record through its filter (`core.filters.filtered`) —
        low-, high- or band-pass, whichever the filtering describes —
        using the history's own `filtering` unless one is passed.

        Parameters
        ----------
        filtering : Filtering, optional
            The pass-band edges and order. Defaults to the record's
            own `filtering`.

        Returns
        -------
        TimeHistory
            Every channel through the filter, zero phase.
        """
        from .filters import filtered

        chosen = self.filtering if filtering is None else filtering
        if chosen is None:
            raise ValueError('no filter to apply: set `filtering` first, '
                             'or pass one')
        return filtered(self, chosen)

    def integrate(self, drift_corner: Any = ...) -> TimeHistory:
        """One integration — acceleration to velocity, velocity to
        displacement (`core.filters.integrate`). ``...`` takes the
        default drift corner; ``None`` integrates raw, drift and all.

        Parameters
        ----------
        drift_corner : float or None, optional
            High-pass corner in Hz applied after integrating, so a sensor
            bias cannot become a ramp. `...` takes the default; `None`
            integrates raw.

        Returns
        -------
        TimeHistory
            Acceleration becomes velocity, velocity becomes
            displacement. Other quantities are left out.
        """
        from .filters import DRIFT_CORNER, integrate

        return integrate(self, DRIFT_CORNER if drift_corner is ...
                         else drift_corner)

    def differentiate(self) -> TimeHistory:
        """One differentiation — displacement to velocity, velocity to
        acceleration (`core.filters.differentiate`)."""
        from .filters import differentiate

        return differentiate(self)

    def srs_windows(self, shocks: Sequence[Any] | None = None
                    ) -> list[Any]:
        """The stretches an SRS of this record would read, settled.

        The fallback chain `compute_srs` has always used, extracted so
        the shock panel's derived rows and the spectrum itself cannot
        disagree about it (one implementation): the shocks the record
        carries; failing those, the averaging frames when the record
        is being read as frames; failing everything, the whole record
        as one window. Clipped to the record either way.

        Parameters
        ----------
        shocks : sequence of Shock, optional
            Override windows; the record's own when omitted.

        Returns
        -------
        list of Shock
            The settled windows, clipped to the record.
        """
        from .shocks import Shock

        rate = self.sample_rate
        samples = self.ordinate.shape[1]
        windows = self.shocks if shocks is None else shocks
        if not windows and self.averaging is not None:
            # the frames, when the record is being read as frames. A
            # transient run's playings are its averaging and nothing
            # else — the detector, asked at the same record, hunts for
            # events it was never meant to find here and answers with
            # its own idea of how many there are. Two readings of one
            # record is exactly what the two buttons refuse to show at
            # once, and the numbers should not disagree either.
            windows = tuple(Shock(start, stop) for start, stop
                            in self.averaging.frame_bounds(rate))
        if not windows:
            windows = (Shock(0.0, samples / rate),)
        return [w.clipped(samples, rate) for w in windows]

    def srs_band(self, shocks: Sequence[Any] | None = None
                 ) -> tuple[float, float]:
        """(low, high) in Hz: the band these windows can support.

        From a frequency low enough that the *shortest* window still
        holds a cycle of it — a curve is one grid across every event,
        so the shortest is what the grid has to fit — up to a fifth
        of the sample rate, above which the ramp-invariant filter is
        being asked about frequencies the record cannot resolve.
        What `compute_srs` uses when no band is given, and what the
        shock panel states beside its settings.

        Parameters
        ----------
        shocks : sequence of Shock, optional
            Override windows; the record's own when omitted.

        Returns
        -------
        tuple of float
            (low, high) in Hz.
        """
        rate = self.sample_rate
        windows = self.srs_windows(shocks)
        shortest = min(w.samples(rate) for w in windows) / rate
        return max(1.0 / shortest, 1.0), rate / 5.0

    def compute_srs(self, shocks: Sequence[int] | None = None,
                    low: float | None = None, high: float | None = None,
                    per_octave: int | None = None, q: float | None = None,
                    kind: str = 'maximax') -> Srs:
        """A shock response spectrum for every channel of every shock.

        The parameters live on the history, the way averaging does: pass
        `shocks` to override, or leave it and the windows already on the
        object are used. With none anywhere, the whole record is one
        window, which is what a history holding a single trimmed
        transient is.

        The windows are the shocks the record carries, or — when it is
        being read as frames rather than events — the frames. A
        specification carries neither and is one window: it is a single
        playing of a waveform, whole.

        One curve per channel *per event*, never averaged across events.
        A shock test is judged on the worst shock, and the mean of four
        of them describes none of them. Which event a curve came from is
        in `block`, exactly as which average a frame came from is.

        The band defaults to what the windows can support: from a
        frequency low enough that the *shortest* window still holds a
        cycle of it — a curve is one grid across every event, so the
        shortest is what the grid has to fit — up to a fifth of the
        sample rate, above which the ramp-invariant filter is being
        asked about frequencies the record cannot resolve.

        Parameters
        ----------
        shocks : sequence of int, optional
            Which shock windows to use. All of them when omitted.
        low, high : float, optional
            The natural-frequency band, in Hz.
        per_octave : int, optional
            Frequency lines per octave.
        q : float, optional
            The oscillator amplification.
        kind : str, default 'maximax'
            Which peak to keep: 'maximax' (largest magnitude of either
            sign), 'positive' or 'negative'.

        Returns
        -------
        Srs
            One curve per channel per shock.
        """
        from .srs import DEFAULT_Q, PER_OCTAVE, maximax, octave_frequencies
        from .srs import peaks as srs_peaks

        rate = self.sample_rate
        windows = self.srs_windows(shocks)
        named = len(windows) > 1
        band_low, band_high = self.srs_band(shocks)
        low = band_low if low is None else float(low)
        high = band_high if high is None else float(high)
        frequencies = octave_frequencies(
            low, high, PER_OCTAVE if per_octave is None else per_octave)
        q = DEFAULT_Q if q is None else float(q)

        rows, dofs, dims, hints, blocks = [], [], [], [], []
        for i, record in enumerate(self.ordinate):
            for k, window in enumerate(windows):
                first, last = window.bounds(rate)
                cut = record.real[first:last]
                if kind == 'maximax':
                    rows.append(maximax(cut, frequencies, rate, q=q))
                else:
                    highest, lowest = srs_peaks(cut, frequencies, rate, q=q)
                    rows.append(highest if kind == 'positive' else -lowest)
                dofs.append(self.response_dof[i])
                dims.append(self.ordinate_dim[i])
                hints.append(self.dimension_hint[i])
                blocks.append(f'shock {k + 1}' if named
                              else (None if self.block is None
                                    else self.block[i]))
        return self.srs_type()(
            frequencies, np.asarray(rows, dtype=np.float64),
            response_dof=dofs, ordinate_dim=dims,
            dimension_hint=hints,
            block=blocks if named or self.block is not None else None,
            q=q, kind=kind,
            comment=f'{kind} SRS at Q={q:g}')

    def compute_psds(self, averaging: Averaging | None = None) -> Psd:
        """One-sided auto-power spectral density per channel, averaged
        across the frames.

        Welch's method where each average is already its own frame:
        rectangular window, no overlap, Gxx = 2|X|²/(fs·N) with DC and
        Nyquist unhalved, the frames' powers averaged. Power is
        phase-insensitive, so burst random's random phase costs
        nothing here. Values land in (SI unit)²/Hz with the Psd
        dimension convention ('acceleration**2/frequency'); a channel
        with undefined units stays undefined, its hint squared along.

        Parameters
        ----------
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.

        Returns
        -------
        Psd
            One auto-power spectral density per channel.
        """
        frequencies, scale, groups = self._spectral_frame(averaging)
        rows, dofs, dims, hints = [], [], [], []
        for (dof, dim, hint), windowed in groups.items():
            frames = np.fft.rfft(windowed, axis=1)
            rows.append(np.mean(np.abs(frames) ** 2, axis=0) * scale)
            dofs.append(dof)
            dims.append(f'{dim}**2/frequency' if dim != UNKNOWN
                        else UNKNOWN)
            hints.append(f'{hint}**2/frequency' if hint else None)
        out = self.psd_type()(
            frequencies, np.asarray(rows, dtype=np.float64),
            response_dof=dofs, ordinate_dim=dims,
            dimension_hint=hints, comment='averaged PSD')
        # computed from a record, so a density per bin whatever class
        # it came back as — the PSD of a target is a Specification, and
        # a Specification is otherwise a power law through breakpoints
        out.interpolation = 'bin'
        return out

    def psd_type(self) -> type[Psd]:
        """What a PSD of this history is.

        A plain record's spectra are plain spectra. A **target's** are
        still a target: the PSD of a waveform the article was required
        to see is the spectrum it was required to see, and losing that
        on the way through an FFT would leave two objects of the same
        class with nothing but a name to say which was the requirement.
        Overridden in `TransientSpecification` rather than decided by
        the caller, so a script and the app cannot disagree.
        """
        return Psd

    def srs_type(self) -> type[Srs]:
        """What an SRS of this history is — see `psd_type`."""
        return Srs

    def compute_cpsds(self, averaging: Averaging | None = None) -> Psd:
        """The full cross-spectral density matrix, averaged across the
        frames — every channel against every channel, not just each
        against itself.

        Gxy = 2·conj(X)·Y/(fs·N) with DC and Nyquist unhalved, which is
        `compute_psds` on the diagonal, where conj(X)·X is |X|². The
        cross terms are how two channels move together, which is most of
        what a CPSD is for, and they are what a PSD throws away.

        Laid out as the importer lays an imported matrix out: one record
        per (response, reference) pair, row by row, so an n-channel
        history gives n² records that read as a grid. A cross term's
        dimension is the product of the two, `a*b/frequency`, against
        `a**2/frequency` down the diagonal.

        Parameters
        ----------
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.

        Returns
        -------
        Psd
            The full cross-spectral matrix, every channel against
            every channel.
        """
        frequencies, scale, groups = self._spectral_frame(averaging)
        keys = list(groups)
        frames = {key: np.fft.rfft(windowed, axis=1)
                  for key, windowed in groups.items()}
        records, responses, references = [], [], []
        dims, hints = [], []
        for row, (dof_i, dim_i, hint_i) in enumerate(keys):
            for column, (dof_j, dim_j, hint_j) in enumerate(keys):
                cross = np.mean(np.conj(frames[keys[row]])
                                * frames[keys[column]], axis=0) * scale
                records.append(cross)
                responses.append(dof_i)
                references.append(dof_j)
                known = UNKNOWN not in (dim_i, dim_j)
                dims.append(
                    (f'{dim_i}**2/frequency' if dim_i == dim_j
                     else f'{dim_i}*{dim_j}/frequency') if known
                    else UNKNOWN)
                hints.append(
                    (f'{hint_i}**2/frequency' if hint_i == hint_j
                     else f'{hint_i}*{hint_j}/frequency')
                    if hint_i and hint_j else None)
        return Psd(frequencies, np.asarray(records, dtype=np.complex128),
                   response_dof=responses, reference_dof=references,
                   ordinate_dim=dims, dimension_hint=hints,
                   comment='averaged CPSD')

    #: quantities a channel is measured in when it is driving rather
    #: than responding. What `compute_multiple_coherence` takes for its
    #: references when it is not told: a shaker is instrumented by its
    #: force, a shock machine by the volts commanding it.
    EXCITATION_DIMS = ('force', 'voltage')

    def drive_dofs(self) -> list[str]:
        """The DOFs this history looks like it was driven at.

        A guess from the quantities alone, and it is only ever a
        default. What actually makes a channel a drive is that the
        controller had a feedback device on it, which the channel table
        records and a time history does not.
        """
        return list(dict.fromkeys(
            dof for dof, dim in zip(self.response_dof, self.ordinate_dim)
            if dim in self.EXCITATION_DIMS))

    def _cross_spectral_frame(self, what, references=None, averaging=None):
        """The pieces every cross-spectral estimate is built from.

        Returns (frequencies, keys, drives, responses, cross, averages),
        where `drives` and `responses` index into `keys` and `cross(i,
        j)` is the averaged cross spectrum between two channels, one
        value per line.

        An FRF and a multiple coherence are two readings of exactly the
        same frames and the same reference set — H1 is the best linear
        prediction of a response from the references, and the coherence
        is how much of the response that prediction accounts for. Worked
        out separately they could disagree about which channel is a
        reference or how the record was framed, and then the coherence
        beside an FRF would not be that FRF's coherence.

        `references` names the drives **by DOF**; without it they are
        guessed from the quantities (`drive_dofs`). A DOF is not a
        channel: a drive point carries a force record *and* an
        acceleration record at the same DOF, and the reference is the
        force. Naming one picks the excitation channel there, and the
        accelerometer at the same DOF stays a response — which is what
        a controller computes against.
        """
        frequencies, scale, groups = self._spectral_frame(averaging)
        keys = list(groups)
        wanted = list(references) if references is not None \
            else self.drive_dofs()
        drives = []
        for dof in wanted:
            # the excitation channel at that DOF, and only if there is
            # none does the DOF alone decide — a reference named by a
            # caller who has no force there is still a reference
            at = [i for i, key in enumerate(keys) if key[0] == dof]
            driven = [i for i in at if keys[i][1] in self.EXCITATION_DIMS]
            chosen = (driven or at)
            if chosen and chosen[0] not in drives:
                drives.append(chosen[0])
        if not drives:
            raise ValueError(
                f'{what} needs reference channels; none of '
                f'{wanted or "the excitation quantities"} is in this history')
        responses = [i for i in range(len(keys)) if i not in drives]
        if not responses:
            raise ValueError('every channel is a reference; there is '
                             'nothing left to explain')

        spectra = np.asarray([np.fft.rfft(groups[key], axis=1)
                              for key in keys])
        averages = spectra.shape[1]
        if averages <= len(drives):
            # Not a poor estimate — no estimate. Fitting a response to
            # `n` references from `n` or fewer averages is an exact fit
            # at every line: the reference matrix is singular, the
            # coherence comes back 1.0 everywhere, and the FRF is
            # whichever of infinitely many answers the pseudo-inverse
            # picks. Both look like results and are statements about the
            # arithmetic instead.
            raise ValueError(
                f'{what} needs more averages than references: '
                f'{averages} average{"s" * (averages != 1)} against '
                f'{len(drives)} reference{"s" * (len(drives) != 1)} fits '
                f'exactly. Set the averaging on this history first — the '
                f'same frames a PSD uses.')

        def cross(i: int, j: int) -> np.ndarray:
            return np.mean(np.conj(spectra[i]) * spectra[j], axis=0) * scale

        return frequencies, keys, drives, responses, cross, averages

    #: the estimators, in the order they are offered. Which one is right
    #: is a statement about where the noise is, and only the person who
    #: ran the test knows that — so it is a choice and not a default
    #: dressed up as one.
    FRF_METHODS = ('Hv', 'H1', 'H2')

    def compute_frfs(self, references: Sequence[str] | None = None,
                     averaging: Averaging | None = None,
                     method: str = 'Hv') -> Frf:
        """The frequency response functions, one per response/drive pair.

        The three estimators differ in one assumption — where the noise
        is — and agree wherever there is little of it. They part company
        exactly where a measurement is worst, which is why the choice
        matters and why it is a choice.

        **H1** assumes the noise is on the *response*. It biases low at
        resonance, where the response is large and the force small, and
        it is what a controller computes and a modal fit expects.

            Gfx = Gff H,   so   H = Gff^-1 Gfx

        With one reference that is the textbook `Gfx / Gff`. With
        several it is the MIMO estimate, and the matrix inverse is the
        whole point: two shakers driving one article are correlated, and
        dividing each response by each drive separately would credit
        both with the same motion.

        **H2** assumes the noise is on the *reference*, and biases high
        at anti-resonance for the mirror-image reason. With one
        reference it is `Gxx / Gxf`, one response at a time. With
        several references it needs as many equations as unknowns, and
        there are exactly enough only when the system is *square* — as
        many responses as references — where it becomes the classical
        coupled form `Gxx * Gfx^-1` (Rocklin, Crowley and Vold, 1985),
        computed here exactly as sdynpy computes it (matched by
        decision, Brandon 2026-08-28, and pinned against its numbers).
        The coupling is worth knowing about: every response feeds one
        matrix inverse, so a channel's H2 depends on which other
        channels are in the set — measured at 0.2% on the oracle
        signals, growing with noise — where H1 and Hv rows never do. A
        non-square multi-reference set is refused; Hv answers the same
        noise-on-both question per response, uncoupled.

        **Hv** (the default) assumes noise on both and asks for neither:
        it is the total-least-squares fit, the null direction of

            [[Gff, Gfx], [Gxf, Gxx]]

        taken as the eigenvector of its smallest eigenvalue, per response
        and per line. It falls between H1 and H2 — strictly between,
        wherever the coherence is under one — and needs no claim about
        which instrument is the better one. It is the default because
        that claim is the one a test least often gets to make honestly:
        an accelerometer out on a structure and a force cell in the load
        path are both imperfect, in different places.

        Note what a total-least-squares fit means with units in play: it
        weighs a unit of error on the force against a unit of error on
        the acceleration, and those are not the same thing. That is
        baked into the estimator and is the received formulation; it is
        why Hv is a middle reading and not a better one.

        A pseudo-inverse rather than a solve for H1, because two shakers
        can be very nearly the same drive and `Gff` is then close to
        singular — where a solve raises or returns nonsense, a
        pseudo-inverse gives the least-squares answer the estimate is
        asking for anyway.

        Built on the same frames `compute_psds` averages and the same
        references `compute_multiple_coherence` uses, so the coherence
        beside an FRF is that FRF's coherence.

        Parameters
        ----------
        references : sequence of str, optional
            The drive DOFs. Detected from the record when omitted.
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.
        method : str, default 'Hv'
            The estimator: 'Hv', 'H1' or 'H2'.

        Returns
        -------
        Frf
            One record per response and drive pair.
        """
        if method not in self.FRF_METHODS:
            raise ValueError(
                f'{method!r} is not an FRF estimator: '
                + ', '.join(self.FRF_METHODS))
        frequencies, keys, drives, responses, cross, averages = \
            self._cross_spectral_frame('FRFs', references, averaging)
        # Multi-reference H2 exists only where the system is square —
        # as many responses as references — and there it is the
        # classical coupled form, Gxx * Gfx^-1 (Rocklin, Crowley and
        # Vold, 1985), adopted to match sdynpy (Brandon, 2026-08-28;
        # this file refused it for a day first, and the reasoning both
        # ways is worth keeping). The coupling is real: every response
        # feeds the one inverse, so a channel's H2 depends on which
        # other channels are in the set — measured at 0.2% on the
        # oracle signals with 5% noise, growing with noise, where H1
        # is bit-identical under the same swap. That property argued
        # for refusing; matching the tool the audience already trusts
        # argued for computing; Brandon chose compatibility, and the
        # property is documented instead of avoided. Non-square
        # multi-reference stays refused — there the equations genuinely
        # cannot balance — and Hv answers per response, uncoupled.
        coupled = None
        if method == 'H2' and len(drives) > 1:
            if len(responses) != len(drives):
                raise ValueError(
                    f'H2 with {len(drives)} references needs exactly '
                    f'{len(drives)} responses (the coupled square form) '
                    f'and this history has {len(responses)}. '
                    f'Hv is the estimator that answers the same question '
                    f'for any shape.')
            cxx = np.stack([np.stack([cross(i, j) for j in responses],
                                     axis=-1)
                            for i in responses], axis=-2)
            cfx = np.stack([np.stack([cross(i, j) for j in responses],
                                     axis=-1)
                            for i in drives], axis=-2)
            # H solves H * Gfx = Gxx, taken as solve(Gfx^T, Gxx^T)^T per
            # line. The outer conj translates conventions: this file's
            # cross() is Bendat & Piersol's conj(X)*Y, the classical
            # form is written in X*conj(Y), and the two matrices are
            # elementwise conjugates — verified against sdynpy's own
            # square H2 at machine precision rather than trusted.
            coupled = np.conj(np.swapaxes(
                np.linalg.solve(np.swapaxes(cfx, -2, -1),
                                np.swapaxes(cxx, -2, -1)), -2, -1))

        estimator = getattr(self, f'_frf_{method.lower()}')
        gff = np.stack([np.stack([cross(i, j) for j in drives], axis=-1)
                        for i in drives], axis=-2)

        rows, response_dof, reference_dof, dims, hints = [], [], [], [], []
        for position, x in enumerate(responses):
            if coupled is not None:
                estimate = coupled[:, position, :]
            else:
                gfx = np.stack([cross(i, x) for i in drives], axis=-1)
                estimate = estimator(gff, gfx, np.real(cross(x, x)))
            # response-major, so a pair reads the way it is named:
            # every drive for one response, then the next response
            for column, i in enumerate(drives):
                rows.append(estimate[:, column])
                response_dof.append(keys[x][0])
                reference_dof.append(keys[i][0])
                dims.append(self._ratio(keys[x][1], keys[i][1]))
                hints.append(self._ratio(keys[x][2], keys[i][2])
                             if keys[x][2] and keys[i][2] else None)
        return Frf(frequencies, np.asarray(rows), response_dof=response_dof,
                   reference_dof=reference_dof, ordinate_dim=dims,
                   dimension_hint=hints,
                   comment=f'{method} from {len(drives)} '
                           f'reference{"s" * (len(drives) != 1)}, '
                           f'{averages} averages')

    @staticmethod
    def _frf_h1(gff, gfx, _gxx):
        """Noise on the response: the least-squares fit of x to f."""
        return (np.linalg.pinv(gff) @ gfx[..., None])[..., 0]

    @staticmethod
    def _frf_h2(_gff, gfx, gxx):
        """Noise on the reference: the least-squares fit of f to x,
        inverted. Single reference only, which the caller has checked —
        `Gxf` is the conjugate of `Gfx`, and `H2 = Gxx / Gxf`.

        `gxx` is one number per line and `gfx` is one per line per
        reference, so the response's own power needs the reference axis
        put back on it before the division — without it numpy broadcasts
        lines against lines and hands back a square matrix that indexes
        without complaint.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return gxx[..., None] / np.conj(gfx)

    @staticmethod
    def _frf_hv(gff, gfx, gxx):
        """Noise on both: the total-least-squares fit.

        The model says `[H, -1]` annihilates `[F; X]`, so the estimate
        is the null direction of that vector's own cross-spectral
        matrix. With noise there is no exact null, and the direction
        that comes closest is the eigenvector of the smallest
        eigenvalue — the matrix is Hermitian, so `eigh` gives them in
        ascending order and the first column is the one.

        Scaled by the response's own component, since the eigenvector is
        only defined up to a factor and the model fixes that factor by
        the -1.
        """
        references = gfx.shape[-1]
        block = np.empty(gff.shape[:-2] + (references + 1, references + 1),
                         dtype=complex)
        block[..., :references, :references] = gff
        block[..., :references, references] = gfx
        block[..., references, :references] = np.conj(gfx)
        block[..., references, references] = gxx
        _values, vectors = np.linalg.eigh(block)
        smallest = vectors[..., 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            return -smallest[..., :references] / smallest[..., references:]

    @staticmethod
    def _ratio(response, reference):
        """'acceleration' over 'force' is 'acceleration/force'; anything
        over an unknown quantity is unknown, because a ratio is only as
        defined as its worse half."""
        if response == UNKNOWN or reference == UNKNOWN:
            return UNKNOWN
        return f'{response}/{reference}'

    def compute_multiple_coherence(
            self, references: Sequence[str] | None = None,
            averaging: Averaging | None = None) -> MultipleCoherence:
        """How much of each response the drives together account for.

        Ordinary coherence asks what one reference explains. Multiple
        coherence asks what a whole set of them explains at once, which
        is the only useful question in a MIMO test: two shakers driving
        one article are correlated with each other, so a response can
        look poorly coherent with either one alone while being fully
        accounted for by the pair.

        For a response x and references r,

            gamma^2 = (Grx^H Grr^-1 Grx) / Gxx

        — the power of the best linear prediction of x from all the
        references at once, over the power actually measured. With one
        reference it collapses to the ordinary coherence, which is the
        cheapest check that the algebra is right.

        Built on the same frames `compute_psds` averages, so it covers
        the stretch of record the averaging view has set and no other:
        a coherence worked out over the whole file would describe a
        different measurement from the PSD beside it.

        `references` and the framing are `_cross_spectral_frame`'s, the
        same ones `compute_frfs` uses — so the coherence beside an FRF
        is that FRF's coherence and not a differently-framed one.

        A pseudo-inverse rather than a solve, because two shakers
        driving one article can be very nearly the same drive and the
        reference matrix is then close to singular — where a solve
        raises or returns nonsense, a pseudo-inverse gives the
        least-squares answer the estimate is asking for anyway.

        Parameters
        ----------
        references : sequence of str, optional
            The drive DOFs. Detected from the record when omitted.
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.

        Returns
        -------
        MultipleCoherence
            One curve per response channel.
        """
        frequencies, keys, drives, responses, cross, averages = \
            self._cross_spectral_frame('multiple coherence', references,
                                       averaging)

        # (lines, drives, drives), Hermitian and positive semi-definite
        grr = np.stack([np.stack([cross(i, j) for j in drives], axis=-1)
                        for i in drives], axis=-2)
        inverse = np.linalg.pinv(grr)

        rows = []
        for i in responses:
            grx = np.stack([cross(k, i) for k in drives], axis=-1)[..., None]
            explained = np.real(
                np.conj(np.swapaxes(grx, -1, -2)) @ inverse @ grx)[..., 0, 0]
            measured = np.real(cross(i, i))
            with np.errstate(divide='ignore', invalid='ignore'):
                value = np.where(measured > 0.0, explained / measured, 0.0)
            # a ratio of powers cannot exceed one; a few parts in 1e12
            # over is the arithmetic, and a real overshoot means too few
            # averages for the number of references, which is a property
            # of the analysis and not of the article
            rows.append(np.clip(value, 0.0, 1.0))
        return MultipleCoherence(
            frequencies, np.asarray(rows, dtype=np.float64),
            response_dof=[keys[i][0] for i in responses],
            comment=f'multiple coherence against {len(drives)} references, '
                    f'{averages} averages')

    def to_sep005(self, name: str | None = None) -> list[dict[str, Any]]:
        """This record as SEP 005 timeseries — the sdypy ecosystem's
        interchange form (`io.sep005` reads them back).

            timeseries = history.to_sep005('run 4')

        Returns the standard's **list** form: usually one dict, and one
        per unit where channels mix. The sdypy validator holds a
        series' ``unit_str`` to a single string, so accelerometers
        beside a force gauge cannot be one compliant series — the list
        of series is exactly what the standard provides for that, and a
        split series wears the unit in its name so two of them stay
        distinguishable.

        Values go exactly as they are held: SI where units are defined,
        with ``unit_str`` naming the SI unit, and the file's raw
        numbers where they are not, with ``unit_str`` empty — the
        standard allows an empty unit, and inventing one would claim a
        scale nobody declared. ``fs`` says the sampling when it is
        even; an uneven record sends its ``time`` vector, which the
        standard equally accepts. ``quantity`` rides where the
        standard has a letter for what a series measures.

        `name` defaults to the comment when the record carries one,
        because a SEP 005 series must be named and the comment is the
        nearest thing to a name an object holds — the project knows
        what it called this record, the record does not.

        Parameters
        ----------
        name : str, optional
            A name for the series.

        Returns
        -------
        list of dict
            One SEP 005 timeseries mapping per channel.
        """
        from ..io.sep005 import DIMENSION_TO_QUANTITY
        from ..units import SI

        name = str(name or next((c for c in self.comment if c),
                                'time history'))
        groups: dict[tuple[str, str | None], list[int]] = {}
        for i in range(self.num_records):
            unit = (SI.label(self.ordinate_dim[i])
                    if self.ordinate_unit[i] is not None else '')
            letter = DIMENSION_TO_QUANTITY.get(self.known_dim(i))
            groups.setdefault((unit, letter), []).append(i)
        data = np.asarray(np.real(self.ordinate), dtype=float)
        out = []
        for (unit, letter), rows in groups.items():
            series: dict[str, Any] = {
                # (n,) for a lone channel — the standard's preferred
                # single-channel shape, and the one the sdypy validator
                # measures a `time` vector against correctly
                'data': data[rows] if len(rows) > 1 else data[rows[0]],
                'name': (name if len(groups) == 1
                         else f'{name} [{unit or letter or "raw"}]'),
                'channel_name': [str(self.response_dof[i]) for i in rows],
                'unit_str': unit,
            }
            try:
                series['fs'] = float(self.sample_rate)
            except ValueError:
                series['time'] = np.asarray(self.abscissa, dtype=float)
            if letter is not None:
                series['quantity'] = letter
            out.append(series)
        return out


class Spectrum(DataArray):
    """A linear spectrum: amplitude and phase at each frequency line.

    The complex average of a record's frames, not a power average — so
    content whose phase is random frame to frame averages toward zero,
    which is what makes this the wrong reading for burst random and the
    right one for a deterministic signal. A `Psd` is the power average
    and does not have that property.
    """

    function_type = 12
    abscissa_dim = 'frequency'
    complex_ordinate = True

    def animate(self, geometry: Any, frequency: float | None = None,
                **kwargs: Any) -> Any:
        """The operating deflection shape at one frequency line, moving
        on a geometry as the GUI animates it. Defaults to the strongest
        line; `frequency` picks another.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        frequency : float, optional
            Which frequency line. The strongest when omitted.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..viz.animate import animate_ods
        return animate_ods(geometry, self, frequency=frequency, **kwargs)


class Frf(DataArray):
    """Frequency response function: response per unit reference."""

    function_type = 4
    abscissa_dim = 'frequency'
    complex_ordinate = True
    needs_reference = True

    @classmethod
    def _si_units_for(cls, dimension):
        response, _, reference = dimension.partition('/')
        return SI.unit(response), SI.unit(reference)

    def _conversion(self, unit, reference_unit=None):
        if reference_unit is None:
            raise UnitError(
                'An FRF needs both a response unit and a reference unit')
        response_dim = dimension_of(unit)
        reference_dim = dimension_of(reference_unit)
        if response_dim is None or reference_dim is None:
            raise UnitError(
                f"Cannot determine dimensions of {unit!r} / {reference_unit!r}")
        scale = si_factor(unit) / si_factor(reference_unit)
        return scale, 0.0, f'{response_dim}/{reference_dim}'

    def plot_cmif(self, shapes: Any = None, **kwargs: Any) -> Any:
        """The CMIF the fitting screen draws; with `shapes` the modal
        model's synthesis is drawn dashed over the measurement.

        Parameters
        ----------
        shapes : ShapeSet, optional
            A modal fit, drawn as the synthesized CMIF over the measured one.
        **kwargs
            Passed through to the plotting layer.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..plot import plot_cmif
        return plot_cmif(self, shapes, **kwargs)

    def animate(self, geometry: Any, frequency: float | None = None,
                **kwargs: Any) -> Any:
        """The operating deflection shape at one frequency line, moving
        on a geometry as the GUI animates it. Defaults to the strongest
        line; `frequency` picks another.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        frequency : float, optional
            Which frequency line. The strongest when omitted.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        from ..viz.animate import animate_ods
        return animate_ods(geometry, self, frequency=frequency, **kwargs)


class _CoherenceBase(DataArray):
    """How much of a response its references account for, from 0 to 1.

    A ratio of spectra, so dimensionless by construction — which makes it
    the one kind of data array that never needs a unit declared. It arrives
    fully defined and the Define Units step never offers it.
    """

    abscissa_dim = 'frequency'
    complex_ordinate = False
    log_ordinate = False        # a 0..1 ratio says nothing on a log axis
    ordinate_limits = (0.0, 1.05)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # whatever a caller passes, a coherence is dimensionless
        kwargs['ordinate_dim'] = 'dimensionless'
        kwargs.setdefault('ordinate_unit', 'dimensionless')
        super().__init__(*args, **kwargs)

    def plot_map(self, **kwargs: Any) -> Any:
        """The coherence map: frequency across, channel down, 0..1."""
        from ..plot import plot_coherence_map
        return plot_coherence_map(self, **kwargs)


class Coherence(_CoherenceBase):
    """Ordinary coherence: one response against one reference.

    UFF dataset 58 calls this function type 6, and gives *multiple*
    coherence a code of its own — see `MultipleCoherence`. They are not the
    same object with an optional reference: one is a matrix of pairs and the
    other is one curve per response, and a file that says 6 is promising a
    reference DOF per record.
    """

    function_type = 6
    needs_reference = True


class MultipleCoherence(_CoherenceBase):
    """How much of a response *all* the references together account for.

    One curve per response channel, with no reference of its own — the
    references are summed over, which is the whole point of it. This is what
    a Rattlesnake modal survey reports, and what its random environment
    reports against its control channels.

    UFF dataset 58 function type 26. visualdynamics wrote 6 here for a while, which
    claimed ordinary coherence and a reference DOF that was never there.
    """

    function_type = 26


def paired_channels(signal: Any, floor: Any) -> list[tuple[int, int, str, str]]:
    """The channels two densities share, as (signal row, floor row,
    DOF, quantity).

    The pairing rule under every two-density reading (Brandon,
    2026-08-25): matched by **(DOF, quantity)**, never by index — a
    drive point carries an accelerometer and a load cell at one DOF,
    and keyed by DOF alone the dict collapsed them, silently losing
    half the drive-point channels from the first ratio drawn.
    Channels only one side measured are left out; sharing none
    refuses. Both sides must stand on the same frequency lines —
    neither a ratio nor an overlay means anything across mismatched
    bins, so that refuses rather than interpolating an answer nobody
    measured.

    Here rather than inside `density_ratio` because the overlay of
    the two densities and the ratio of them are the same pair read
    two ways: the report draws them one above the other, and a figure
    that paired its channels differently from the figure below it
    would be two claims about one measurement.
    """
    if len(signal.abscissa) != len(floor.abscissa) or not np.allclose(
            signal.abscissa, floor.abscissa, rtol=1e-9, atol=1e-12):
        raise ValueError(
            'the two densities stand on different frequency lines; '
            'compute both with the same framing and the ratio exists')

    def keys(density):
        dims = np.atleast_1d(density.ordinate_dim)
        return [(str(dof), str(dims[min(i, len(dims) - 1)]))
                for i, dof in enumerate(density.response_dof)]

    quiet = {key: j for j, key in enumerate(keys(floor))}
    shared = [(i, quiet[key], key[0], key[1])
              for i, key in enumerate(keys(signal)) if key in quiet]
    if not shared:
        raise ValueError('the two densities share no channel; there '
                         'is no ratio to take')
    return shared


def density_ratio(signal: Any, floor: Any):
    """One density over another, channel by channel, line by line.

    Channels pair by `paired_channels`; a zero floor line answers
    NaN, never infinity — on real hardware zero means below
    resolution, in a simulation it means the quiet was exactly
    silent, and neither is an infinite signal-to-noise.

    Returns ``(abscissa, rows, dofs, dims)`` — the ratio as plain
    arrays plus each row's quantity, for whoever is reading it: the
    plot's ratio view, the stage, the report's ratio figure — every
    ratio divides here, and the quantity rides along so a mixed
    object's rows can be told apart.
    """
    rows, dofs, dims = [], [], []
    for i, j, dof, dim in paired_channels(signal, floor):
        top = np.real(np.asarray(signal.ordinate[i]))
        bottom = np.real(np.asarray(floor.ordinate[j]))
        with np.errstate(divide='ignore', invalid='ignore'):
            rows.append(np.where(bottom > 0.0, top / bottom, np.nan))
        dofs.append(dof)
        dims.append(dim)
    return np.asarray(signal.abscissa), np.asarray(rows), dofs, dims


class Psd(DataArray):
    #: the width of the band each line stands for, when the lines are
    #: not evenly spaced. `None` for a narrowband spectrum, where the
    #: bins are the midpoints between neighbors and inferring them is
    #: exact. An octave-band spectrum sets it: its bins are geometric,
    #: its edges are a standard's and not its neighbors', and reading
    #: them off the centers would be a hair out at every band and wrong
    #: at the two ends.
    """Power spectral density: the declared unit is the engineering unit
    whose square-per-Hz the values are in (declare 'g' for g^2/Hz).

    Held complex because a *cross* spectrum is complex — the phase
    between two channels is most of what a CPSD is for. An autospectrum
    is not: a channel against itself is a magnitude squared, real by
    construction. So the type allows complex and the object stores what
    it actually has, which for a specification or a set of ASDs is a
    real array of half the size.
    """

    #: How a value is read *between* the points, which is the one fact
    #: that settles both the integral and the picture:
    #:
    #: 'bin'     — constant across its own bin, whose width is
    #:             `bandwidth` or the midpoints either side. The area is
    #:             the sum of value times width, and it draws as steps.
    #: 'log_log' — a power law to the next point, which is the straight
    #:             line two breakpoints make on log axes. The area has a
    #:             closed form, and it draws as that curve.
    #:
    #: One field and not two, because a spectrum drawn one way and
    #: integrated the other is a picture of a number nobody computed.
    #: They used to be chosen separately — the drawing by class, the
    #: integral by whichever function the caller reached for — and the
    #: PSD of a transient target was drawn as a power law through four
    #: thousand density lines while its level was worked out as their
    #: area.
    #:
    #: Per *instance* rather than per class, because a specification is
    #: either: one written at a dozen breakpoints is a curve, and one
    #: computed from a record is a density like any other.
    interpolation = 'bin'

    #: Decibels added to this spectrum *when it is compared against a
    #: specification* — nowhere else. A run captured at -6 dB is
    #: compared by scaling it up to the 0 dB requirement, which is
    #: standard practice; the data itself is never touched, and the
    #: spectrum drawn or integrated alone is exactly what was measured.
    #: `None` means detect it from the data (`compliance.detect_scale_db`,
    #: the whole-dB offset that best lays the spectrum on the
    #: specification); a number is a user holding it, including 0 for
    #: "do not scale". Every comparison the app, the scripts and the
    #: report make resolves it through `compliance.comparison_scale_db`,
    #: so the curves, the error bars and the table cannot disagree
    #: about what was compared.
    scale_db: int | None = None

    def principal_shapes(self, quantity: str | None = None
                         ) -> tuple[list[str], np.ndarray, str]:
        """The dominant shape of the cross-spectral matrix at each line,
        as (dofs, shapes (channels x lines), quantity).

        The channels of one quantity form a square Hermitian matrix per
        line; its largest eigenvalue's eigenvector, scaled by the square
        root of that eigenvalue, is the principal operating deflection
        shape — the direction of the output spectra's own CMIF, with
        each channel's phase relative to the others and no reference to
        choose. `quantity` picks which channels (the commonest when not
        told). Refuses a set with no cross records — an autospectrum set
        has no phase and its reading is the envelope — and an incomplete
        block, whose eigenvectors would be shapes of holes.

        Parameters
        ----------
        quantity : str, optional
            Which quantity, for a mixed object.

        Returns
        -------
        tuple of (list of str, numpy.ndarray, str)
            The DOF labels, the dominant shape at each line, and the
            quantity they are in.
        """
        if self.reference_dof is None:
            raise ValueError('no cross records — an autospectrum set '
                             'has no phase')
        factors = [channel_quantities(self.known_dim(i))
                   for i in range(self.num_records)]
        kinds = [response for response, _reference in factors]
        if quantity is None:
            kinds_present = [k for k in kinds if k != UNKNOWN]
            if not kinds_present:
                raise ValueError('no record names a quantity')
            quantity = max(set(kinds_present), key=kinds_present.count)
        channels: list[str] = []
        for i, dof in enumerate(self.response_dof):
            if kinds[i] == quantity and dof not in channels:
                channels.append(dof)
        n = len(channels)
        if n < 2:
            raise ValueError(f'fewer than two {quantity} channels')
        index = {dof: k for k, dof in enumerate(channels)}
        matrix = np.zeros((len(self.abscissa), n, n), dtype=np.complex128)
        filled = np.zeros((n, n), dtype=bool)
        for i in range(self.num_records):
            row = index.get(self.response_dof[i])
            column = index.get(self.reference_dof[i])
            if (row is None or column is None
                    or (kinds[i], factors[i][1]) != (quantity, quantity)):
                continue
            matrix[:, row, column] = self.ordinate[i]
            filled[row, column] = True
        if not filled.all():
            raise ValueError(f'the {quantity} cross-spectral block is '
                             'incomplete')
        values, vectors = np.linalg.eigh(matrix)
        principal = (vectors[:, :, -1]
                     * np.sqrt(np.maximum(values[:, -1:], 0.0)))
        return channels, principal.T.copy(), quantity

    def animate(self, geometry: Any, frequency: float | None = None,
                quantity: str | None = None, **kwargs: Any) -> Any:
        """This set on a geometry, as the GUI shows it.

        A CPSD — cross records present — animates its principal
        operating deflection shape: the dominant eigenvector of the
        cross-spectral matrix per line, each channel's phase relative
        to the others. An autospectrum set has no phase, so it shows
        the envelope instead: two copies deflected ±sqrt(PSD), color
        reading dB below the loudest node at any line. Defaults to the
        strongest line; `frequency` picks another, `quantity` which
        measurement deflects.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        frequency : float, optional
            Which frequency line.
        quantity : str, optional
            Which quantity, for a mixed object.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plot widget or plotter.
        """
        try:
            dofs, shapes, _used = self.principal_shapes(quantity)
        except ValueError:
            from ..viz.animate import animate_envelope
            return animate_envelope(geometry, self, frequency=frequency,
                                    quantity=quantity, **kwargs)
        return Spectrum(self.abscissa, shapes,
                        response_dof=dofs).animate(
            geometry, frequency, **kwargs)

    def area(self, record: int = 0, low: float | None = None,
             high: float | None = None) -> float:
        """The area under one record, over a band or over all of it.

        The one integral. Whichever way this spectrum is read, it is
        read the same way here as it is drawn — that is what the field
        above is for, and why nothing outside this method chooses.

        Units are the ordinate's times frequency, so the square root of
        it is an RMS for a PSD.

        Parameters
        ----------
        record : int, default 0
            Which record.
        low, high : float, optional
            The band to integrate over.

        Returns
        -------
        float
            The area beneath the curve — the mean square, whose root
            is the RMS.
        """
        return self._area_of(np.asarray(self.ordinate[record]), low, high)

    def _area_of(self, values: ArrayLike, low: float | None,
                 high: float | None) -> float:
        """The area under one curve on this spectrum's axis, read the
        way this spectrum is — the target, or one of a specification's
        limits, which is written the same way."""
        from .compliance import log_log_area

        values = np.real(np.asarray(values, dtype=float))
        if self.interpolation == 'log_log':
            return log_log_area(self.abscissa, values, low, high)
        left, right = self.bin_bounds()
        if low is not None:
            left, right = np.maximum(left, low), np.maximum(right, low)
        if high is not None:
            left, right = np.minimum(left, high), np.minimum(right, high)
        width = np.maximum(right - left, 0.0)
        good = np.isfinite(values) & (values >= 0.0) & np.isfinite(width)
        if not good.any():
            return float('nan')
        return float(np.sum(values[good] * width[good]))

    def areas(self, record: int, lows: ArrayLike, highs: ArrayLike
              ) -> np.ndarray:
        """`area` over several bands at once: the comparison is
        judged cell by cell, and a thousand cells must not cost a
        thousand passes over the lines.

        Parameters
        ----------
        record : int
            Which record.
        lows, highs : array-like
            The bands' edges, paired.

        Returns
        -------
        numpy.ndarray
            One area per band; NaN where nothing is written.
        """
        return self._areas_of(np.asarray(self.ordinate[record]), lows, highs)

    def _areas_of(self, values: ArrayLike, lows: ArrayLike,
                  highs: ArrayLike) -> np.ndarray:
        from .compliance import log_log_areas

        values = np.real(np.asarray(values, dtype=float))
        lows = np.atleast_1d(np.asarray(lows, dtype=float))
        highs = np.atleast_1d(np.asarray(highs, dtype=float))
        if self.interpolation == 'log_log':
            return log_log_areas(self.abscissa, values, lows, highs)
        # the running total of the density, knot by knot, and a band's
        # area as the difference of two readings of it: a bin's edges
        # are knots, its area accrues linearly between them, and a gap
        # between bins accrues nothing
        left, right = self.bin_bounds()
        good = np.isfinite(values) & (values >= 0.0) & (right > left)
        # only the bins the bands can reach: a running total that
        # started far below the first band would carry every line
        # outside it into the subtraction, and a band's area must not
        # move in the twelfth decimal with what lies outside it
        if lows.size:
            good &= (right > np.nanmin(lows)) & (left < np.nanmax(highs))
        if not good.any():
            return np.zeros(lows.shape) if lows.size else np.full(lows.shape, np.nan)
        left, right, held = left[good], right[good], values[good]
        order = np.argsort(left)
        left, right, held = left[order], right[order], held[order]
        gained = held * (right - left)
        before = np.concatenate([[0.0], np.cumsum(gained)[:-1]])
        knots = np.column_stack([left, right]).ravel()
        totals = np.column_stack([before, before + gained]).ravel()
        total = np.interp(np.clip(highs, knots[0], knots[-1]), knots, totals) \
            - np.interp(np.clip(lows, knots[0], knots[-1]), knots, totals)
        return np.where(highs > lows, np.maximum(total, 0.0), 0.0)

    def written(self, record: int | None = None) -> np.ndarray:
        """Which lines say something: finite, and for a requirement
        positive — a controller writes zero or NaN on every line it did
        not control, and those lines are not a requirement of nothing.
        Over one record, or any record when none is named.

        Parameters
        ----------
        record : int, optional
            The record; every record when omitted.

        Returns
        -------
        numpy.ndarray
            A boolean per line.
        """
        values = np.real(np.atleast_2d(np.asarray(self.ordinate)))
        if record is not None:
            values = values[[int(record)]]
        return np.isfinite(values).any(axis=0)

    def extent(self, record: int | None = None) -> tuple[float, float] | None:
        """(low, high) this spectrum speaks for, in hertz.

        For a density per bin, the outer edges of the written bins — a
        line stands for its whole bin. For a curve between breakpoints,
        the first and last written points. What octave banding clips
        its end bands to, and what a comparison is judged over
        (Brandon, 2026-09-19): a controller's target is stored on
        every FFT line to Nyquist and speaks only where it is written.

        Parameters
        ----------
        record : int, optional
            The record; the union over every record when omitted.

        Returns
        -------
        tuple of float, or None
            None when nothing is written above zero hertz.
        """
        frequencies = np.asarray(self.abscissa, dtype=float)
        said = self.written(record) & np.isfinite(frequencies)
        if self.interpolation == 'log_log':
            said &= frequencies > 0.0
            if not said.any():
                return None
            return float(frequencies[said].min()), float(frequencies[said].max())
        left, right = self.bin_bounds()
        said &= right > 0.0
        if not said.any():
            return None
        return float(left[said].min()), float(right[said].max())

    def bin_edges(self) -> np.ndarray:
        """The edges a step plot of this spectrum lands on: one more
        than there are lines, `bin_bounds` tiled."""
        left, right = self.bin_bounds()
        return np.concatenate([left, right[-1:]])


    function_type = 9
    abscissa_dim = 'frequency'
    complex_ordinate = True

    #: the width of the band each line stands for, when the lines are
    #: not evenly spaced. None for a narrowband spectrum, whose bins
    #: are the midpoints between neighbors and exactly inferable. An
    #: octave-band spectrum sets it: its bins are geometric, its edges
    #: are a standard's rather than its neighbors', and reading them
    #: off the centers is a hair out at every band and plainly wrong at
    #: the two ends.
    bandwidth: np.ndarray | None = None


    def __init__(self, *args: Any, bandwidth: ArrayLike | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not has_phase(self.ordinate):
            self.ordinate: np.ndarray = np.ascontiguousarray(self.ordinate.real)
        if bandwidth is not None:
            bandwidth = np.asarray(bandwidth, dtype=np.float64)
            if bandwidth.shape != self.abscissa.shape:
                raise ValueError(
                    f'bandwidth has shape {bandwidth.shape}, expected '
                    f'{self.abscissa.shape} to match the abscissa')
            self.bandwidth = bandwidth

    def to_octave(self, per_octave: int | None = None,
                  low: float | None = None,
                  high: float | None = None) -> Psd:
        """This spectrum integrated onto proportional bands.

        An integration, not a resampling: each band takes the
        mean-square content that falls in it, divided by its own width,
        so the area under the spectrum — and therefore the RMS it
        carries — is unchanged. Reading the narrowband curve at each
        band center would throw away everything between the centers.

        The band grid is absolute (see `visualdynamics.core.octave`), so this
        needs no specification to be told about: the frequency range
        only chooses which bands of the one fixed grid come back, and
        two runs banded the same way land on the same bands whatever
        their ranges were.

        Cross terms come through complex, which is what makes this work
        for a CPSD as well as a PSD.

        Parameters
        ----------
        per_octave : int, optional
            Bands per octave.
        low, high : float, optional
            The band to cover.

        Returns
        -------
        Psd
            The same power, arranged on proportional bands.
        """
        from .octave import resample

        frequencies = np.asarray(self.abscissa, dtype=float)
        per_octave, centers, widths, bounds = self._octave_grid(
            per_octave, low, high)
        banded = resample(frequencies, self.ordinate, bounds, widths,
                          source=self.bin_bounds())
        out = Psd(centers, banded,
                  response_dof=list(self.response_dof),
                  reference_dof=(None if self.reference_dof is None
                                 else list(self.reference_dof)),
                  ordinate_dim=list(self.ordinate_dim),
                  ordinate_unit=list(self.ordinate_unit),
                  reference_unit=list(self.reference_unit),
                  dimension_hint=list(self.dimension_hint),
                  block=None if self.block is None else list(self.block),
                  bandwidth=widths,
                  comment=f'1/{per_octave} octave bands')
        # banding conserves the area and changes nothing about what the
        # object is — a held comparison scale included, so the report's
        # own banding compares what the narrowband comparison compared
        out.scale_db = self.scale_db
        return out

    def _octave_grid(self, per_octave, low, high):
        """(per_octave, centers, widths, bounds): the bands this spectrum
        bands onto — one grid, shared by the ordinate and, for a
        specification, its limits. The standard's own bands, whole:
        they are defined frequency bands and are not cut to the data,
        nor read or drawn over part of themselves (Brandon,
        2026-09-19, twice). A band the data only partly fills holds
        that content over its whole width."""
        from .octave import PER_OCTAVE, bands

        frequencies = np.asarray(self.abscissa, dtype=float)
        positive = frequencies[frequencies > 0.0]
        if positive.size < 2:
            raise ValueError(
                'octave bands need a spectrum with frequency lines above '
                'zero; this one has ' + str(positive.size))
        # the range the bands have to span is the range the *bins*
        # cover, not the range the centers do. A line at 0.5 Hz stands
        # for a bin reaching down to 0.25, and a grid that started at
        # the line would leave that half outside every band — which is
        # a small loss of area and an unnecessary one. And only the
        # written bins: a target stored to Nyquist and written to
        # 2000 Hz speaks to 2000 Hz.
        span = self.extent()
        reach, _beyond = self.bin_bounds()
        inside = reach[reach > 0.0]
        floor = float(inside.min()) if inside.size else float(positive.min())
        if span is None:
            span = (floor, float(positive.max()))
        low = max(span[0], floor) if low is None else float(low)
        high = span[1] if high is None else float(high)
        per_octave = PER_OCTAVE if per_octave is None else int(per_octave)
        return (per_octave, *bands(low, high, per_octave))

    def bin_widths(self) -> np.ndarray:
        """The width of every line's own bin.

        Its own when it has one, the midpoints between neighbors when
        it does not — so a caller integrating a spectrum never has to
        ask which kind it is holding.
        """
        if self.bandwidth is not None:
            return self.bandwidth
        return np.gradient(np.asarray(self.abscissa, dtype=float))

    def bin_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(left, right) of every line's own bin — see `octave.bin_bounds`."""
        from .octave import bin_bounds

        return bin_bounds(np.asarray(self.abscissa, dtype=float),
                          self.bandwidth)

    @classmethod
    def _si_units_for(cls, dimension):
        numerator = dimension.split('/')[0]
        if numerator.endswith('**2'):
            return SI.unit(numerator[:-3]), None
        base, _, cross = numerator.partition('*')
        return SI.unit(base), (SI.unit(cross) if cross else None)

    def _conversion(self, unit, reference_unit=None):
        base_dim = dimension_of(unit)
        if base_dim is None:
            raise UnitError(f"Cannot determine the physical dimension of {unit!r}")
        scale = si_factor(unit)
        if reference_unit is None or reference_unit == unit:
            return scale ** 2, 0.0, f'{base_dim}**2/frequency'
        cross_dim = dimension_of(reference_unit)
        if cross_dim is None:
            raise UnitError(
                f"Cannot determine the physical dimension of {reference_unit!r}")
        return (scale * si_factor(reference_unit), 0.0,
                f'{base_dim}*{cross_dim}/frequency')


class Srs(DataArray):
    """A shock response spectrum: the peak an oscillator reached.

    Not a spectrum of the shock. Every point is the largest response a
    single-degree-of-freedom oscillator of that natural frequency ever
    reached while its base was shaken by the measured transient — one
    number out of one whole run of one filter. Two shocks with the same
    SRS can look nothing alike, and an SRS cannot be turned back into a
    time history, because the phase that produced each peak is gone.

    The abscissa is the oscillator's natural frequency, not a frequency
    in the shock — laid out in decades (`log_abscissa`): an SRS is
    specified at octave-spaced natural frequencies, and drawn linear
    the bottom five octaves crush into the left margin. The ordinate is in the quantity the base was measured
    in — an acceleration transient gives an acceleration SRS.

    `q` and `kind` change what the curve means, so they belong to the
    object rather than to whoever happened to compute it: an SRS at
    Q = 10 and the same shock at Q = 50 are different curves, and a
    maximax reading is not a positive one. See `visualdynamics.core.srs`.
    """

    function_type = 24          # dataset 58's code for a shock response
    abscissa_dim = 'frequency'
    _log_abscissa_default = True  # decades, by every SRS reader's convention

    def __init__(self, *args: Any, q: float | None = None,
                 kind: str = 'maximax', **kwargs: Any) -> None:
        from .srs import DEFAULT_Q

        super().__init__(*args, **kwargs)
        self.q: float = DEFAULT_Q if q is None else float(q)
        self.kind: str = str(kind)

    @property
    def damping(self) -> float:
        """The damping ratio the amplification factor means."""
        from .srs import damping_for

        return damping_for(self.q)

    def _plain_kwargs(self):
        return {'q': self.q, 'kind': self.kind}

    def __eq__(self, other: object) -> bool:
        if super().__eq__(other) is not True:
            return NotImplemented if not isinstance(other, Srs) else False
        return (self.q == other.q) and (self.kind == other.kind)

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f', {self.kind} at Q={self.q:g})'


class Bounded:
    """Limit curves held beside an ordinate, in the same units as it.

    What a test was controlled to is a target and the room around it:
    how far the response may stray before the controller warns, and how
    far before it stops. That is true of a random vibration
    specification and of a shock one alike, and it is the same four
    curves either way, so it is written once here and mixed into both.

    The limits are arrays beside the ordinate rather than four more data
    objects. They share the abscissa, the DOFs, the dimensions and the
    units by construction, which is the point: nothing can drift out of
    step, and `_value_arrays` carries them through every unit conversion
    the ordinate sees. `limit()` hands one back as a plain object of the
    underlying kind when a caller wants one, to plot or export as such.

    Limits are per control channel, so a record that is not one has
    none: those rows are NaN, which plots as a gap rather than as a line
    at zero.

    Attributes:
        warning_lower: The curve below which the controller warns, per
            record, in the ordinate's own units. NaN where a record is
            not a control channel, or where no such limit was written.
        warning_upper: The same, above the target.
        abort_lower: The curve below which the controller stops the test.
        abort_upper: The same, above the target.
    """

    LIMITS = ('warning_lower', 'warning_upper', 'abort_lower', 'abort_upper')

    #: what one of these curves is when it is taken on its own
    PLAIN: type = DataArray

    def __init__(self, *args: Any, warning_lower: ArrayLike | None = None,
                 warning_upper: ArrayLike | None = None,
                 abort_lower: ArrayLike | None = None,
                 abort_upper: ArrayLike | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.limits: dict[str, np.ndarray] = {}
        given = {'warning_lower': warning_lower,
                 'warning_upper': warning_upper,
                 'abort_lower': abort_lower,
                 'abort_upper': abort_upper}
        for name, values in given.items():
            if values is None:
                continue
            values = np.atleast_2d(np.asarray(values, dtype=np.float64))
            if values.shape != self.ordinate.shape:
                raise ValueError(
                    f'{name} has shape {values.shape}, expected '
                    f'{self.ordinate.shape} to match the ordinate')
            self.limits[name] = values

    def _value_arrays(self):
        return (self.ordinate, *self.limits.values())

    @property
    def has_limits(self) -> bool:
        return bool(self.limits)

    def limit_area(self, name: str, record: int = 0,
                   low: float | None = None,
                   high: float | None = None) -> float:
        """The area under one limit curve, read the way the
        specification is — a limit is written the way its owner is.

        Parameters
        ----------
        name : str
            One of `LIMITS`.
        record : int, default 0
            Which record.
        low, high : float, optional
            The band to integrate over.

        Returns
        -------
        float
            The area beneath the limit; NaN when it was never written.
        """
        values = self.limits.get(name)
        if values is None:
            return float('nan')
        return self._area_of(np.atleast_2d(values)[record], low, high)

    def limit_areas(self, name: str, record: int, lows: ArrayLike,
                    highs: ArrayLike) -> np.ndarray:
        """`limit_area` over several bands at once — see `areas`."""
        values = self.limits.get(name)
        if values is None:
            return np.full(np.shape(np.atleast_1d(lows)), np.nan)
        return self._areas_of(np.atleast_2d(values)[record], lows, highs)

    def limit(self, name: str) -> DataArray | None:
        """One limit in its own right, or None if it has none."""
        if name not in self.limits:
            return None
        out = self.PLAIN(
            abscissa=self.abscissa, ordinate=self.limits[name],
            response_dof=self.response_dof,
            reference_dof=self.reference_dof, block=self.block,
            ordinate_dim=self.ordinate_dim,
            ordinate_unit=self.ordinate_unit,
            reference_unit=self.reference_unit,
            comment=[f'{c} {name}'.strip() for c in self.comment],
            **self._plain_kwargs())
        # a limit is written the way its specification is, so it is read
        # and integrated the same way. Left to its class default, the
        # limits of a computed specification would come back claiming to
        # be breakpoints of a power law.
        reading = getattr(self, 'interpolation', None)
        if reading is not None and hasattr(out, 'interpolation'):
            out.interpolation = reading
        return out

    def display_limit(self, name: str,
                      unit_system: UnitSystem) -> np.ndarray | None:
        """A limit in display units; undefined records pass through as-is."""
        values = self.limits.get(name)
        if values is None:
            return None
        out = np.empty_like(values)
        for i, dim in enumerate(self.ordinate_dim):
            out[i] = (values[i] if dim == UNKNOWN
                      else unit_system.from_si(values[i], dim))
        return out

    def __eq__(self, other: object) -> bool:
        if super().__eq__(other) is not True:
            return NotImplemented if not isinstance(other, self.PLAIN) else False
        if set(self.limits) != set(getattr(other, 'limits', {})):
            return False
        return all(np.allclose(values, other.limits[name], equal_nan=True)
                   for name, values in self.limits.items())

    def __repr__(self) -> str:
        band = (f', {len(self.limits)} limits' if self.limits
                else ', no limits')
        return super().__repr__()[:-1] + band + ')'


class Specification(Bounded, Psd):
    """What a random vibration test was controlled to: a PSD, and its band.

    A specification is a PSD in every respect — same abscissa, same
    quantity, same conversions — with the four limit curves `Bounded`
    carries. So it inherits both, rather than reimplementing either.
    """

    PLAIN = Psd

    #: a written specification's points are breakpoints of a power
    #: law. `compute_psds` says otherwise for one it computed, and an
    #: importer says otherwise for a target on a controller's lines
    #: (`reading_of`).
    interpolation = 'log_log'

    #: how many lines an even grid needs before it reads as lines
    #: rather than as breakpoints that happen to be evenly spaced
    LINES_AT_LEAST = 8

    @staticmethod
    def reading_of(frequencies: ArrayLike) -> str:
        """How a specification written at these frequencies reads:
        'bin', a density per line, for many lines on an even grid — a
        controller's target on its FFT lines — and 'log_log', breakpoints
        of a power law, for a few unevenly spaced points (Brandon,
        2026-09-19: "breakpoint specifications have very few frequency
        lines and I would expect an uneven spacing of them"). Both
        octave-band and narrowband specifications step; only a
        breakpoint curve is drawn as the law between its points.

        Parameters
        ----------
        frequencies : array_like
            The specification's frequency lines, ascending.

        Returns
        -------
        str
            'bin' or 'log_log'.
        """
        f = np.asarray(frequencies, dtype=float)
        if f.size < Specification.LINES_AT_LEAST:
            return 'log_log'
        steps = np.diff(f)
        even = np.all(np.abs(steps - np.median(steps))
                      <= 1e-6 * max(abs(float(np.median(steps))), 1e-12))
        return 'bin' if even else 'log_log'

    #: how finely a breakpoint curve's cross terms are read when a band
    #: integrates them: points per band on a log grid
    _CROSS_POINTS = 64

    def written(self, record: int | None = None) -> np.ndarray:
        """Which lines the requirement is written on: finite and
        positive. A controller writes zero on the lines it did not
        control, and zero is not a requirement of silence.

        Parameters
        ----------
        record : int, optional
            The record; every record when omitted.

        Returns
        -------
        numpy.ndarray
            A boolean per line.
        """
        values = np.atleast_2d(np.asarray(self.ordinate))
        if record is not None:
            values = values[[int(record)]]
        magnitude = np.abs(values) if np.iscomplexobj(values) else np.real(values)
        with np.errstate(invalid='ignore'):
            return (np.isfinite(magnitude) & (magnitude > 0.0)).any(axis=0)

    def to_octave(self, per_octave: int | None = None,
                  low: float | None = None,
                  high: float | None = None) -> Specification:
        """This specification integrated onto proportional bands, its
        warning and abort limits with it.

        The same area rule as `Psd.to_octave`, read the way the object
        is: a specification computed on lines is a density and each
        line is a bin; one written at breakpoints is a power law
        between them, and each band takes the exact area under that law
        (`compliance.log_log_area`) over the part of the band the
        specification covers, divided by the band's width. The limits
        are curves written the same way and go through the same rule,
        so a band's warning line stands in the same relation to its
        target as the breakpoints did. Cross terms of a breakpoint
        specification are not power laws (a phase is not), so they are
        read onto a fine log grid the way the authoring sheet reads
        them — magnitude log–log, phase straight — and integrated there.

        The result is a density per band (`interpolation` 'bin', the
        bands' widths carried), which is what a banded measurement is
        compared against band for band. Brandon, 2026-09-18: the
        specification and the measurement it judges should be
        convertible alike, so a compliance can be read on bands at both
        ends — reversing the earlier reasoning (PLAN.md, "Octave
        bands") that a written curve needed no banding because the
        comparison integrates it exactly; it does, and a banded
        specification is still the object a person asks to see and
        hand on.

        Parameters
        ----------
        per_octave : int, optional
            Bands per octave.
        low, high : float, optional
            The band to cover.

        Returns
        -------
        Specification
            The same power and the same limits, arranged on
            proportional bands.
        """
        from .compliance import log_log_area, written_band
        from .octave import resample

        frequencies = np.asarray(self.abscissa, dtype=float)
        rows = np.atleast_2d(self.ordinate)
        if self.interpolation == 'log_log':
            # the grid spans what the curve is written over — its first
            # breakpoint to its last — not bins inferred between
            # breakpoints, which put the first band at the midpoint of
            # the first segment and lost everything below it
            spans = [written_band(frequencies, np.abs(row)) for row in rows]
            spans = [span for span in spans if span is not None]
            if not spans:
                raise ValueError('octave bands need a specification written '
                                 'over a range of frequencies above zero')
            from .octave import PER_OCTAVE, bands

            per_octave = PER_OCTAVE if per_octave is None else int(per_octave)
            low = min(span[0] for span in spans) if low is None else float(low)
            high = (max(span[1] for span in spans) if high is None
                    else float(high))
            centers, widths, bounds = bands(low, high, per_octave)

            def band_real(row):
                out = np.full(len(centers), np.nan)
                band = written_band(frequencies, row)
                if band is None:
                    return out
                for k in range(len(centers)):
                    lo = max(bounds[k], band[0])
                    hi = min(bounds[k + 1], band[1])
                    if hi > lo:
                        out[k] = log_log_area(frequencies, row, lo, hi) \
                            / widths[k]
                return out

            def band_complex(row):
                from .author import read_at_lines

                out = np.full(len(centers), np.nan, dtype=complex)
                band = written_band(frequencies, np.abs(row))
                if band is None:
                    return out
                for k in range(len(centers)):
                    lo = max(bounds[k], band[0])
                    hi = min(bounds[k + 1], band[1])
                    if hi > lo:
                        grid = np.geomspace(lo, hi, self._CROSS_POINTS)
                        values = read_at_lines(frequencies, row, grid)
                        out[k] = np.trapezoid(values, grid) / widths[k]
                return out

            banded = np.array([
                band_complex(row) if np.iscomplexobj(row) and np.any(row.imag)
                else band_real(np.real(row)) for row in rows])
            limits = {name: np.array([band_real(row) for row in values])
                      for name, values in self.limits.items()}
        else:
            per_octave, centers, widths, bounds = self._octave_grid(
                per_octave, low, high)
            source = self.bin_bounds()
            banded = resample(frequencies, rows, bounds, widths, source=source)
            limits = {}
            for name, values in self.limits.items():
                values = np.atleast_2d(values)
                # a limit row that is all NaN is a cross term's, which
                # has no limit of its own; the integration would read
                # its NaN as nothing and hand back zeros
                unwritten = ~np.isfinite(values).any(axis=1)
                out = resample(frequencies, values, bounds, widths,
                               source=source).astype(float)
                out[unwritten] = np.nan
                limits[name] = out
        out = Specification(
            centers, banded,
            response_dof=list(self.response_dof),
            reference_dof=(None if self.reference_dof is None
                           else list(self.reference_dof)),
            ordinate_dim=list(self.ordinate_dim),
            ordinate_unit=list(self.ordinate_unit),
            reference_unit=list(self.reference_unit),
            dimension_hint=list(self.dimension_hint),
            block=None if self.block is None else list(self.block),
            bandwidth=widths,
            comment=f'1/{per_octave} octave bands',
            **limits)
        out.interpolation = 'bin'
        out.scale_db = self.scale_db
        constraints = getattr(self, 'band_constraints', None)
        if constraints:
            out.band_constraints = dict(constraints)
        return out


class ShockSpecification(Bounded, Srs):
    """What a shock test was controlled to: an SRS, and its band.

    The same relationship a `Specification` has to a `Psd`. A shock
    target is written as a required SRS with tolerance either side of
    it — conventionally +6 dB and -3 dB, which is a factor of 2 up and
    0.707 down — and the limits are those curves.

    Written at one Q, and read at that Q: a target quoted at Q = 10 says
    nothing about what the same shock does to a Q = 50 oscillator, so
    the amplification travels with the target the way it travels with a
    measurement.
    """

    PLAIN = Srs


class TransientSpecification(TimeHistory):
    """What a transient test was controlled to: a target time history.

    A different thing from a `ShockSpecification`, and the difference is
    worth keeping straight because the two get called by each other's
    names. A **shock** specification is an SRS: a required response
    spectrum, and a controller meets it by producing *some* transient
    whose spectrum lands inside the band. A **transient** specification
    is a waveform: this acceleration, sample by sample, and the
    controller inverts the structure's transfer function to reproduce
    it. Rattlesnake can run the second today; the first it cannot.

    So this is a time history that happens to be a target, and the
    comparison it invites is against another time history — what the
    article actually did — rather than against a band. It carries no
    limits for that reason: a tolerance on a waveform is not a settled
    idea the way a tolerance on a spectrum is, and inventing one here
    would be inventing a convention rather than reading one.

    Its derived spectra stay targets. A PSD of this is a
    `Specification` and an SRS of it is a `ShockSpecification` — both
    without limits, for the reason above — because the spectrum of a
    waveform the article was required to see is the spectrum it was
    required to see. Left as plain objects they would be
    indistinguishable from the response's own spectra but for a name,
    and the comparison between them would have to be made by hand
    instead of by type, which is the one thing this whole arrangement
    exists to avoid.
    """

    def psd_type(self) -> type[Psd]:
        """What a PSD of this record is: still a specification.

        The spectrum of a required waveform is itself a requirement,
        so it comes back as `Specification` rather than a plain `Psd`.
        """
        return Specification

    def srs_type(self) -> type[Srs]:
        """What an SRS of this record is: a shock specification, for
        the same reason `psd_type` gives."""
        return ShockSpecification


# Both kinds of coherence, for the places that care that a thing is a bounded
# ratio rather than which kind it is
COHERENCE_TYPES = (Coherence, MultipleCoherence)

DATA_CLASSES = {cls.function_type: cls
                for cls in (TimeHistory, Spectrum, Frf, Psd, Srs,
                            Coherence, MultipleCoherence)}

# A Specification shares the PSD function type, a ShockSpecification the
# SRS one and a TransientSpecification the time-response one — dataset 58
# has no code for a control target — so the native format names the class
# outright
NAMED_CLASSES = {cls.__name__: cls
                 for cls in (TimeHistory, Spectrum, Frf, Psd, Srs,
                             Coherence, MultipleCoherence,
                             Specification, ShockSpecification,
                             TransientSpecification)}


def class_for_function_type(code: int) -> type[DataArray]:
    try:
        return DATA_CLASSES[int(code)]
    except KeyError:
        raise ValueError(
            f'Unsupported function type {code}; '
            f'supported: {sorted(DATA_CLASSES)}')
