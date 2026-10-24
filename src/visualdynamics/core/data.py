"""Data arrays: time histories, spectra, FRFs, PSDs.

Every kind has the same structure: one abscissa shared by all records
(time or frequency, in SI), an ordinate matrix of shape (records,
samples), a response DOF string per record ('101X+'), an optional
reference DOF per record (FRFs, cross spectra), and unit information per
record.

Units are kept per record, because one measurement set often mixes
quantities: a test's time data may hold accelerations, forces, voltages,
strains and temperatures together. Each record has:

- `ordinate_unit[i]`: the unit its values were declared in, or None when
  the source did not say. `reference_unit[i]` holds an FRF's denominator
  unit, or a cross spectrum's second unit.
- `ordinate_dim[i]`: the dimension tag used for display ('acceleration',
  'acceleration/force', ..., or 'unknown').
- `dimension_hint[i]`: the kind of quantity the source said this was,
  kept only for records whose units are undefined.

A record whose unit is unknown keeps the file's raw numbers unchanged.
`define_units()` converts it to SI once and records the unit, so a wrong
choice can be corrected later by redefining the units instead of
importing again.

What a quantity is and what scale its numbers are on are separate facts,
and a file can give one without the other. A UNV with no dataset 164
still says a channel is an acceleration, but gives no way to tell g from
m/s**2. Such a record is imported undefined (raw values, dimension
'unknown', nothing converted), and `dimension_hint` keeps the claim.
Nothing is ever scaled by a hint. It limits the units offered when units
are defined by hand, labels a plot axis, and is written back out to a
format that can name a quantity without giving its unit.

Function type codes follow UFF dataset 58 (also used by sdynpy):
1 time response, 4 FRF, 6 coherence, 9 PSD, 12 spectrum,
24 shock response spectrum, 26 multiple coherence. A specification is a
PSD with limits, and a shock specification is an SRS with limits; each
is saved to a file as its own type.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.fft as scipy_fft
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..units import UnitSystem
    from .amplitude_spectrum import AmplitudeSpectrum
    from .averaging import Averaging, ConfidenceBand
    from .half_power import HalfPower
    from .matrix_checks import Conditioning, Realizability
    from .rms_check import RmsCheck
    from .time_statistics import TimeStatistics
    from .tone_readout import ToneReading
    from .transmissibility import TransmissibilityReading

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
    """(101, 3) -> '101Z+': a node and a UFF direction code as a DOF string.

    Parameters
    ----------
    node : int or str
        The node id.
    direction_code : int
        The UFF direction code (1 to 6 for X+ to RZ+, negative for the
        minus directions, 0 for none).

    Returns
    -------
    str
        The DOF string; just the node when the code is 0 or unknown.
    """
    return f'{node}{_DIRECTIONS.get(int(direction_code), "")}'


_CODES = {text: code for code, text in _DIRECTIONS.items()}


def direction_code(direction: str) -> int:
    """'Z+' -> 3: a direction string as its UFF code; the inverse of the
    table `dof_string` reads.

    An unsigned direction means the positive one, which is how DOFs are
    usually written by hand. Case and surrounding spaces are ignored.

    Parameters
    ----------
    direction : str
        The direction, such as 'Z+', 'z' or 'RX-'.

    Returns
    -------
    int
        The direction code.

    Raises
    ------
    ValueError
        If the direction is not one of the twelve.
    """
    text = str(direction).strip().upper()
    if text in _CODES:
        return _CODES[text]
    if text + '+' in _CODES:
        return _CODES[text + '+']
    raise ValueError(f'unknown direction {direction!r}')


def parse_dof(text: str) -> tuple[int | None, str]:
    """'101RX+' -> (101, 'RX+'); the inverse of `dof_string`.

    Parameters
    ----------
    text : str
        The DOF string.

    Returns
    -------
    tuple of (int or None, str)
        The node number and the rest of the string. The node is None
        when the string does not start with a digit, so a caller can
        report the problem instead of failing.
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

    `what` names the act that needs the step, as the subject of the
    refusal: 'Computing spectra', 'A sample rate'. Every reader of a
    time history's step comes here, the sine extraction included —
    it kept a copy of its own that checked the spacing but not the
    order (2026-10-10).

    An abscissa is stored as it arrived — uneven spacing and out-of-order
    samples are both real, and the storage refuses neither (see
    `DataArray.abscissa`). What cannot be done on such a record is
    anything with an FFT in it, so the refusal lives here, at the
    operation, and says which of the two problems it found: 'unevenly
    spaced' sent someone looking for a dropped sample once when the
    record was simply in the wrong order.
    """
    steps = np.diff(abscissa)
    if not len(steps):
        raise ValueError(f'{what} needs at least two time samples')
    if (steps <= 0).any():
        first = int(np.flatnonzero(steps <= 0)[0])
        raise ValueError(
            f'{what} needs time samples in order; sample {first + 1} is at '
            f'{float(abscissa[first + 1]):g} s, after {float(abscissa[first]):g}')
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        widest = float(np.abs(steps - steps[0]).max())
        raise ValueError(
            f'{what} needs evenly spaced time samples; the step varies by '
            f'{widest:g} s across the record')
    return float(steps[0])


def has_phase(values: ArrayLike) -> bool:
    """Whether complex values have an imaginary part that was measured,
    not only round-off from computing them.

    The imaginary part counts when its largest magnitude is more than
    `REAL_FLOOR` times the largest real magnitude.

    Parameters
    ----------
    values : array-like
        The values.

    Returns
    -------
    bool
        False for real values or an imaginary part below the floor.
    """
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        return False
    real = np.abs(values.real).max(initial=0.0)
    return bool(np.abs(values.imag).max(initial=0.0) > REAL_FLOOR * real)


def channel_quantities(dim: str) -> tuple[str, str]:
    """The (response, reference) quantities a record's dimension names.

    A record's dimension combines its two channels' quantities: an FRF's
    'acceleration/force' gives ('acceleration', 'force'), a CPSD cross
    term's 'acceleration*force/frequency' the same, and a diagonal's
    'acceleration**2/frequency' gives ('acceleration', 'acceleration').
    A single quantity is paired with itself.

    Parameters
    ----------
    dim : str
        The record's dimension.

    Returns
    -------
    tuple of (str, str)
        The response quantity and the reference quantity.
    """
    # A channel is identified by these factors, not by the compound: keyed
    # by the whole dimension, one accelerometer's records split into a row
    # per thing it was measured against, which is how a 13-channel CPSD
    # came up 26 rows.
    numerator = dim.partition('/')[0]
    if numerator.endswith('**2'):
        base = numerator.removesuffix('**2')
        return base, base
    response, star, reference = numerator.partition('*')
    if star:
        return response, reference
    denominator = dim.partition('/')[2]
    return numerator, denominator or numerator


def ordinary_coherence(gpp: ArrayLike, gqq: ArrayLike,
                       gpq: ArrayLike) -> np.ndarray:
    """The ordinary coherence of one pair of channels at every line,
    |Gpq|² / (Gpp Gqq), from their cross spectrum and their two
    autospectra.

    One statement of the ratio for everything that reads it
    (`Cpsd.coherence`, `core.transmissibility`), so the coherence beside
    a CPSD and the one a transmissibility reading is judged by cannot
    differ in how they treat an edge. A ratio of powers cannot exceed
    one, so a value a few parts in 1e12 over is the arithmetic and is
    held at one, and nothing reads under zero. Where either autospectrum
    is not above zero, or any of the three is not a finite number, the
    coherence is undefined and is not a number: a line no one drove, a
    channel that recorded nothing and a sample that is not a number are
    not a coherence of zero.

    Parameters
    ----------
    gpp, gqq : array-like
        The two autospectra, one value per line (the real part is read).
    gpq : array-like
        The cross spectrum between them, the same shape.

    Returns
    -------
    numpy.ndarray
        The coherence, 0 to 1, or not a number where it is undefined.
    """
    gpp = np.real(np.asarray(gpp)).astype(np.float64)
    gqq = np.real(np.asarray(gqq)).astype(np.float64)
    gpq = np.asarray(gpq, dtype=np.complex128)
    defined = (np.isfinite(gpp) & np.isfinite(gqq) & np.isfinite(gpq)
               & (gpp > 0.0) & (gqq > 0.0))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        value = np.abs(gpq) ** 2 / (gpp * gqq)
    return np.where(defined, np.clip(value, 0.0, 1.0), np.nan)


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
    """Read or set how frequency axes are drawn in every plot, 3-D view
    and report figure: 'log', 'linear', or 'default' (each kind of data
    by its own convention: an SRS on a log axis, the rest linear).

    The setting belongs to the session, not to a project, and is not
    saved.

    Parameters
    ----------
    mode : {'log', 'linear', 'default'} or bool or None, optional
        The choice to make. When omitted, the current one is returned.
        True and False stand for 'log' and 'linear', None for
        'default'.

    Returns
    -------
    str
        The setting after the call.

    Raises
    ------
    ValueError
        If `mode` is none of these.
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


def _recast(obj: Any, cls: type) -> Any:
    """`obj` as an instance of `cls`, every field copied — what reading
    a PSD as a specification, or back, changes is the kind and nothing
    else, so nothing is rebuilt through a constructor that might not
    know a field (a declared-later unit, a dimension hint, a band)."""
    import copy

    out = cls.__new__(cls)
    out.__dict__.update(copy.deepcopy(obj.__dict__))
    return out


class DataArray:
    """Base class; use a concrete subclass (TimeHistory, Spectrum, Frf, Psd).

    One object holds many records, such as 36 accelerometer channels or
    the 2 592 FRFs of a 36-by-72 matrix, all sharing one abscissa.
    Everything that varies between records is a list with one entry per
    record, so record *i* is `ordinate[i]` measured at `response_dof[i]`.

    Values are stored in SI once their units are known. A record whose
    units were never declared keeps the file's raw numbers and has
    `ordinate_dim == 'unknown'`; what the file said it was, without giving
    its unit, is kept in `dimension_hint`.

    Parameters
    ----------
    abscissa : array-like
        The x axis, in seconds or hertz.
    ordinate : array-like
        The values, shape (records, len(abscissa)); a single record may
        be one-dimensional.
    response_dof : str or sequence of str
        One DOF string per record, or one string for every record.
    reference_dof : str or sequence of str, optional
        One per record. Required by the kinds that need a reference
        (`Frf`).
    ordinate_dim : str or sequence of str, optional
        The dimension of each record; 'unknown' when omitted.
    comment : str or sequence of str, optional
        Free text per record.
    ordinate_unit, reference_unit : str or sequence of str, optional
        The unit each record was declared in. Filled with the SI unit
        where the dimension is known and no unit is given.
    dimension_hint : str or sequence of str, optional
        The quantity the source named for each record, kept only where
        the dimension is unknown.
    block : str or sequence of str, optional
        A label per record for repeats of the same measurement.

    Raises
    ------
    ValueError
        If the ordinate's shape does not match the abscissa, a list has
        the wrong length, a DOF cannot be read, or a needed reference is
        missing.

    Attributes
    ----------
    abscissa : numpy.ndarray
        The x axis, shared by every record: seconds for a time history,
        hertz for anything in the frequency domain. It is stored as it
        arrived, and uneven spacing and out-of-order samples are both
        allowed, since both occur in real data. Anything that needs an
        even step (anything with an FFT in it) checks for one when it
        runs and raises a ValueError saying what is wrong.
    ordinate : numpy.ndarray
        Shape `(records, len(abscissa))`. Complex where the subclass says
        so (`complex_ordinate`), real otherwise.
    response_dof : list of str
        What each record was measured at, as a DOF string (`'101X+'`).
    reference_dof : list of str or None
        What each record was measured against, for the kinds that have
        one: the shaker on an FRF, the other channel of a cross spectrum.
        None where the kind has no reference.
    block : list of str or None
        Which repeat of the same measurement each record is: an average,
        a run, a shock. A short label, not a time.
    ordinate_dim : list of str
        The quantity each record measures (`'acceleration'`, `'force'`),
        or `'unknown'` while its units are undeclared.
    ordinate_unit : list of str or None
        The unit each record was declared in, such as 'g'. The values are
        stored in SI whatever the declared unit; a record made with a
        known dimension and no unit is given the SI unit. None means
        undeclared.
    reference_unit : list of str or None
        The same for the reference of a ratio, so an FRF record has both
        halves of (m/s²)/N.
    dimension_hint : list of str or None
        What the file said a record measures, without giving its unit.
        Nothing is ever scaled by a hint: it limits the units offered,
        labels an axis, and is written back out on export.
    comment : list of str
        Free text per record, as the source file had it.
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
            unit, reference = self.units_for(dim)
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

        The dimension when the units are defined, otherwise the dimension
        hint, otherwise 'unknown'. Do not scale values by this: a hinted
        record's values are raw, and its scale is what is not known.

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
    def units_for(cls, dimension, system=SI):
        """The (ordinate unit, reference unit) a unit system gives for a
        known dimension.

        With SI, the default, this is the pair a record is labeled with
        when it is made with a known dimension and no unit. With another
        system it is the pair a record is labeled with when its values
        are written in that system, as the `.mat` writer does in the
        display units. The base class returns no reference unit; `Frf`
        and `Psd` split the dimension into its two halves.

        Parameters
        ----------
        dimension : str
            An `ordinate_dim`, such as 'acceleration'.
        system : UnitSystem, optional
            The unit system; SI when omitted.

        Returns
        -------
        tuple of (str, str or None)
            The ordinate unit and the reference unit, None where the
            record has no reference.
        """
        # Generalized from the SI pair on 2026-10-13, for a `.mat` written
        # in the display units.
        return system.unit(dimension), None

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

    def record_fields(self) -> dict[str, Any]:
        """Each record's description (DOFs, quantity, units, hint, comment
        and block) as the keyword arguments for making a derived object
        with the same records. Every list is a copy.

        A filter, a truncation and an octave banding keep their source's
        records and change only the values. Override `comment` where the
        derived object should say how it was made.

        Returns
        -------
        dict
            Keyword arguments for a `DataArray` constructor, without the
            abscissa and ordinate.
        """
        # The one statement of what "the same records" carries, where it
        # was written out at each derivation before (2026-10-10).
        return {
            'response_dof': list(self.response_dof),
            'reference_dof': (None if self.reference_dof is None
                              else list(self.reference_dof)),
            'ordinate_dim': list(self.ordinate_dim),
            'ordinate_unit': list(self.ordinate_unit),
            'reference_unit': list(self.reference_unit),
            'dimension_hint': list(self.dimension_hint),
            'comment': list(self.comment),
            'block': None if self.block is None else list(self.block)}

    def rename_dof(self, old: str, new: str,
                   quantity: str | None = None) -> int:
        """Give a channel's coordinate a new name, in place.

        A channel is a coordinate and a quantity: a drive point has a
        load cell and an accelerometer at one DOF. With `quantity` given,
        only that channel is renamed, so a force labeled at the wrong
        node can be moved without moving the accelerometer at the same
        node. The channel is renamed wherever it appears, as a response
        and as a reference, since a CPSD's accelerometer is on both sides
        of its cross terms and is one sensor. A rename that would give
        two records the same coordinate and quantity (two accelerometers
        at one point) is refused.

        Parameters
        ----------
        old : str
            The coordinate as it is, '101Z+'.
        new : str
            The coordinate to give it; normalized the way every DOF is
            ('101Z' is '101Z+'), and refused when it is not one.
        quantity : str, optional
            Which channel at `old`: 'acceleration', 'force' and so on, as
            `channel_quantities` names a record's factors. When omitted,
            every quantity at the coordinate is renamed.

        Returns
        -------
        int
            How many DOFs changed, counting a response and a reference
            on one record separately; 0 when `new` equals `old`.

        Raises
        ------
        ValueError
            If `new` is not a DOF, no record matches, or the rename would
            put two channels of one quantity at one coordinate.
        """
        # The rename is the channel's, not the point's: a force labeled
        # at the wrong node moves without the accelerometer beside it,
        # and the other is changed explicitly if it should be (Brandon,
        # 2026-09-06). Two records given one identity would have made two
        # rows of the grid into one and hidden a record, so that is
        # refused.
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
        """Remove records in place, by index, by DOF, or by capture.

        Everything a record has is removed with it: its row of the
        ordinate, its DOFs, units, comment, hint and block, and on a
        specification its limit curves. Removing every record is refused,
        since an empty data array cannot be shown. Numbered block labels
        are renumbered without gaps afterward; named ones are kept.

        `dof`, `dim`, `reference` and `capture` select records by what
        they are ("channel 101Z+", "the third run"). Together they select
        the records that match all of them, and that selection is added
        to any `indices` given.

        Parameters
        ----------
        indices : sequence of int, optional
            Which records to remove, by position.
        dof : str or sequence of str, optional
            Remove the records at these response DOFs. A drive point has
            two records at one DOF, a force and an acceleration, and the
            DOF alone selects both; add `dim` to select one.
        dim : str or sequence of str, optional
            Limit the selection to records whose `ordinate_dim` is one of
            these ('force', 'acceleration', ...).
        reference : str or sequence of str, optional
            Remove the records at these reference DOFs: a column of an
            FRF matrix, where `dof` is a row.
        capture : int or sequence of int, optional
            Remove these captures, numbered as `capture_indices` numbers
            them (each channel's n-th recording). Time histories only.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the selection matches nothing, would remove every record,
            or uses `reference` or `capture` on a kind that has none.
        IndexError
            If an index is out of range.
        """
        # `dof` and `capture` are the selectors a person means, where
        # indices are the machine's (Brandon, 2026-08-30, reading
        # thirteen indices in the journal where one capture number would
        # have said it). The limit curves go too, because left behind they
        # would belong to the wrong channels.
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
        # a PSD's degrees of freedom are per record, and left whole they
        # would describe the wrong channels' averages
        counts = getattr(self, 'statistical_dof', None)
        if counts is not None:
            # typed here as well as on Psd: the annotations test reads the
            # first assignment it meets, which for a DataArray is this one
            self.statistical_dof: np.ndarray | None = np.asarray(counts)[keep]
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
        """How many records this object holds, all sharing its one
        abscissa."""
        return self.ordinate.shape[0]

    # ---- units --------------------------------------------------------------

    @property
    def units_defined(self) -> bool:
        """Whether every record has its units defined. False while any
        record still holds the file's unconverted numbers."""
        return all(dim != UNKNOWN for dim in self.ordinate_dim)

    @property
    def undefined_records(self) -> list[int]:
        """The indices of the records with no declared dimension, which
        hold the file's raw numbers until `define_units` is called."""
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

        `units` is one unit applied to every record, a sequence with one
        entry per record, or a {record index: unit} mapping to set only
        some. An entry of None leaves that record as it is. A record that
        already has units is converted from its original values, so it is
        never scaled twice. The record's dimension is set from the unit.

        Parameters
        ----------
        units : str, sequence of str, or dict
            The unit each record's values are in.
        reference_units : str, sequence of str, or dict, optional
            The reference unit, in the same forms: an FRF's denominator
            (required for an FRF), or a cross spectrum's second channel.
            Kinds with no reference ignore it.

        Returns
        -------
        DataArray
            Self, converted to SI in place.

        Raises
        ------
        UnitError
            If a unit cannot be read or names no tracked dimension.
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
        """Withdraw a units declaration, restoring the file's raw values.

        The inverse of `define_units`. The records return to dimension
        'unknown' with no unit. There is no unit string that means "not
        known", so withdrawing a declaration has its own call. Records
        that have no units are left as they are.

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
        """What tells one record from another besides its response DOF.

        The reference DOFs when the records are a matrix of measurements,
        the block labels when they are the same measurement repeated, and
        None when the response DOF alone identifies a record. An object
        with column keys is shown as a grid.
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
        """The (response, reference) DOFs of record `i`.

        A record with no reference is an autospectrum, a channel against
        itself, so its reference is its own response DOF (the diagonal,
        which is what a specification bounds).

        Parameters
        ----------
        i : int
            The record.

        Returns
        -------
        tuple of str
            Response DOF, reference DOF.
        """
        # The plot, the table beside it and the report all ask here, so
        # their labels cannot disagree.
        response = self.response_dof[i]
        reference = (self.reference_dof[i]
                     if self.reference_dof is not None else response)
        return response, reference

    def log_scaled(self) -> bool:
        """Whether this object's magnitude is drawn on a log axis.

        Logarithmic for frequency-domain data unless the class sets
        `log_ordinate` (a coherence is a ratio from 0 to 1 and is drawn
        linear).

        Returns
        -------
        bool
            True for a log ordinate axis.
        """
        # Asked of the object so the 2-D plot and the 3-D waterfall read
        # one rule and cannot disagree about the axis.
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
        """The ordinate in display units; records with undefined units are
        returned unchanged.

        Only the records in `records` are converted, in that order.

        Parameters
        ----------
        unit_system : UnitSystem
            The units to present the values in.
        records : iterable of int, optional
            Which records to convert. All of them when omitted.

        Returns
        -------
        numpy.ndarray
            The values in display units, one row per record asked
            for. Records with undefined units are unchanged.
        """
        # The conversion is gathered per dimension and applied to a block
        # of records at once: a 1356-record FRF has one or two distinct
        # dimensions, and converting row by row meant a unit lookup per
        # record, which is how drawing a single curve came to cost 2713
        # trips through pint.
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
        """`display_ordinate`, a block of records at a time.

        Yields `(start, stop, values)`: positions within `records` and the
        converted rows for them, each block at most `block_bytes`
        (`DISPLAY_BLOCK_BYTES` by default). A caller that reduces each
        record before drawing it, such as the 3-D view, then never holds
        a converted copy of the whole object. A record larger than the
        limit is one block on its own; a row is never split.

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
            Where to write it; `.vdyn` is added when the suffix is not
            there.

        Returns
        -------
        None
        """
        from ..io import native
        native.save(self, path)

    def plot(self, unit_system: UnitSystem | None = None,
             **kwargs: Any) -> Any:
        """Draw every record on one set of axes, in a window.

        Parameters
        ----------
        unit_system : UnitSystem, optional
            Units to draw in.
        **kwargs
            Passed to `visualdynamics.plot.plot_data` (`path`, `show`,
            `marks`, `theme` and others).

        Returns
        -------
        object
            The plot pane, or the path when `path` is given.
        """
        from ..plot import plot_data
        return plot_data(self, unit_system=unit_system, **kwargs)

    def save_plot(self, path: str | os.PathLike,
                  unit_system: UnitSystem | None = None,
                  **kwargs: Any) -> Any:
        """Draw the records and write the figure to `path`, with no window.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write the image, a .png or .svg.
        unit_system : UnitSystem, optional
            Units to draw in.
        **kwargs
            Passed to `visualdynamics.plot.save_plot`.

        Returns
        -------
        str
            The path written.
        """
        from ..plot import save_plot
        return save_plot(self, path, unit_system=unit_system, **kwargs)

    def plot_waterfall(self, records: Sequence[int] | None = None,
                       **kwargs: Any) -> Any:
        """Draw the records in 3-D, spread along a depth axis and colored by
        level, as the plot bar's 3-D view does.

        With `screenshot=` (a path) it renders to that file with no
        window; without it, a window with the app's 3-D pane opens.

        Parameters
        ----------
        records : sequence of int, optional
            Which records to draw. All of them when omitted.
        **kwargs
            Passed to `visualdynamics.viz.waterfall.plot_waterfall`
            (`screenshot`, `unit_system`, `component`, `marks` and
            others).

        Returns
        -------
        object
            The pane (its `.plotter` is the PyVista plotter), or the
            image array when rendering to a file.
        """
        from ..viz.waterfall import plot_waterfall
        return plot_waterfall(self, records, **kwargs)


class TimeHistory(DataArray):
    """A measurement against time, as it was acquired.

    Everything else in a random or shock test is computed from a time
    history, and the methods that compute it are here: spectra, PSDs, the
    full CPSD matrix, FRFs, multiple coherence and shock response
    spectra. Settings that say how to process the record are stored on
    it, not passed to each call: `averaging` (the frames a spectrum is
    averaged over), `shocks` (the events an SRS is computed from),
    `truncation`, `filtering`, `sine_extraction` and `roles`. A PSD and
    the coherence computed beside it therefore describe the same frames.
    """

    function_type = 1
    abscissa_dim = 'time'
    complex_ordinate = False

    def animate(self, geometry: Any, records: Any = None,
                **kwargs: Any) -> Any:
        """Animate the records on a geometry, sample by sample, as the app
        plays a time history selected with a geometry.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        records : sequence of int, optional
            Which records. One record per DOF is used either way
            (`animation_records`).
        **kwargs
            Passed to `visualdynamics.viz.animate.animate_time`:
            `quantity=` picks which measurement moves the geometry in a
            mixed record (the most common one by default),
            `screenshot=` saves a still at the sample with the largest
            deflection, `movie=` saves the record once through as an MP4,
            `seconds=` sets how long that takes.

        Returns
        -------
        object
            The plotter, or the path written.
        """
        from ..viz.animate import animate_time
        return animate_time(geometry, self, records, **kwargs)

    def compute_spectra(self) -> Spectrum:
        """The averaged spectrum of every channel.

        Each of a channel's records (one per average) is transformed with
        a one-sided FFT, a rectangular window and no amplitude scaling
        (numpy's rfft with norm='backward', so a sine of amplitude A on a
        bin reads A*N/2), and the transforms are averaged as a complex
        mean. This is the same convention as sdynpy's
        `TimeHistoryArray.fft`. Phase is kept, so content whose phase is
        random from frame to frame, such as the response to burst random
        excitation, averages toward zero. For lines scaled so a sine
        reads its amplitude under a chosen window, with the frames'
        magnitudes averaged, see `amplitude_spectrum`.

        Returns
        -------
        Spectrum
            One record per channel, with the channel's quantity and
            units.

        Raises
        ------
        ValueError
            If the time samples are not evenly spaced and in order.
        """
        step = _even_steps(self.abscissa, 'Computing spectra')
        groups = {}
        for i, dof in enumerate(self.response_dof):
            key = (dof, self.ordinate_dim[i], self.ordinate_unit[i],
                   self.dimension_hint[i])
            groups.setdefault(key, []).append(i)
        frequencies = np.fft.rfftfreq(len(self.abscissa), step)
        rows, dofs, dims, units, hints = [], [], [], [], []
        for (dof, dim, unit, hint), indices in groups.items():
            frames = scipy_fft.rfft(workers=-1, x=self.ordinate[indices], axis=1)
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

    def amplitude_spectrum(self, averaging: Averaging | None = None, *,
                           window: str = 'hann',
                           parameter: float | None = None
                           ) -> AmplitudeSpectrum:
        """Every channel's amplitude spectrum: each line scaled so a sine
        of amplitude A on it reads A, in peak units, under `window`, the
        frames' magnitudes averaged.

        Beside `compute_spectra`, which is left as it is: that one is
        unscaled, rectangular and a complex mean. Here the window's
        amplitude correction, N/Σw (×2.000 for Hann), is applied and
        stated on the result, and a tone keeps its amplitude in the
        average whatever its phase in each frame.
        `core.amplitude_spectrum.amplitude_spectrum` does the reading
        and says how; `visualdynamics.plot.plot_amplitude_spectrum`
        draws it.

        Parameters
        ----------
        averaging : Averaging, optional
            How to cut the record into frames: the record's own
            `averaging` by default, or one whole frame per record. Its
            window is not used.
        window : str, default 'hann'
            The window, by any name `core.averaging.normalize_window`
            takes: 'hann' for a general reading, 'flattop' for a tone's
            amplitude wherever it falls between lines.
        parameter : float, optional
            The Tukey window's alpha or the Kaiser window's beta.

        Returns
        -------
        AmplitudeSpectrum
            One row per channel, in SI peak units as the record is
            stored, with the window, its coherent gain and correction,
            the line spacing and the frames averaged.

        Raises
        ------
        ValueError
            If the samples are not evenly spaced, or the frames do not
            fit in the record.
        """
        from .amplitude_spectrum import amplitude_spectrum
        return amplitude_spectrum(self, averaging, window=window,
                                  parameter=parameter)

    @property
    def sample_rate(self) -> float:
        """Samples per second, from the abscissa. Raises a ValueError if
        the samples are not evenly spaced and in order."""
        return 1.0 / _even_steps(self.abscissa, 'A sample rate')

    def channel_key(self, i: int) -> tuple[str, str, str | None]:
        """What makes record `i` the same channel as another.

        The DOF and what is measured there, never the DOF alone: a drive
        point has a force record and an acceleration record at the same
        DOF, and those two are not averages of each other. Spectra, PSDs
        and the channel counts all group records by this key.

        Parameters
        ----------
        i : int
            Which record.

        Returns
        -------
        tuple of (str, str, str or None)
            The response DOF, the dimension and the dimension hint.
            Records with the same key are the same channel.
        """
        return (self.response_dof[i], self.ordinate_dim[i],
                self.dimension_hint[i])

    @property
    def records_per_channel(self) -> dict[tuple[str, str, str | None], int]:
        """{channel key: number of records with that key}.

        Usually one. A capture that a controller saved frame by frame has
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
        as its own record. True when the averaging takes each record
        whole as one frame and some channel has more than one record.
        Then the frame length and count are fixed by the file, and only
        the window is left to choose.
        """
        averaging = self.averaging
        return (averaging is not None
                and averaging.frames == 1
                and averaging.frame_length == len(self.abscissa)
                and max(self.records_per_channel.values(), default=0) > 1)

    @property
    def average_counts(self) -> tuple[int, int]:
        """(fewest, most) frames any one channel will be averaged over.

        The two are equal for anything a controller wrote, since it saves
        every channel the same number of times. They differ only for a
        history put together by hand from unequal captures, and the
        table then shows both instead of a number true of only some
        channels.
        """
        counts = self.records_per_channel
        if not counts:
            return (0, 0)
        per_record = 1 if self.averaging is None else self.averaging.frames
        return (min(counts.values()) * per_record,
                max(counts.values()) * per_record)

    def suggest_averaging(self, **kwargs: Any) -> Averaging:
        """Averaging parameters worked out from the record itself.

        A run holds more than the test (the shaker coming up, a
        reduced-level check, whatever was still recording afterward), and
        this finds the steady stretch to average and as many frames as fit
        in it. See `visualdynamics.core.detect`.

        A record that already has an averaging, from an import or set by
        hand, keeps its frame length, window, overlap and detrend; only
        the start and the number of frames are worked out. A record with
        none gets the detector's own settings.

        Parameters
        ----------
        **kwargs
            Overrides for individual parameters, such as `window`.

        Returns
        -------
        Averaging
            The suggested averaging.

        Raises
        ------
        ValueError
            If the detector cannot frame the record.
        """
        # The record's own frame length, window and overlap are kept
        # (Brandon, 2026-09-19: Detect must not override the frame
        # length, window and overlap the controller used).
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

    def default_averaging(self) -> Averaging:
        """The averaging a view of this record opens with.

        The record's own averaging when its start was set (by an import,
        a drag or a typed value). Otherwise the result of
        `suggest_averaging`, unless that has fewer frames than the
        record's own averaging asks for, in which case the record's own is
        used. A record with no averaging that the detector cannot frame
        gets each record as one frame.

        Returns
        -------
        Averaging
            The averaging to show.
        """
        # A burst record has no steady stretch to find, and the detector
        # once returned four of the twenty averages a modal survey asked
        # for (2026-09-19), hence the fewer-frames rule. The view opens
        # where Detect would put it, not at t = 0 (Brandon, 2026-09-30).
        from .averaging import Averaging

        own = self.averaging
        if own is not None and own.start_set:
            return own
        try:
            found = self.suggest_averaging()
        except ValueError:
            return (own if own is not None
                    else Averaging.for_records(len(self.abscissa)))
        if own is not None and found.frames < own.frames:
            return own
        return found

    def _resolved_averaging(self, averaging=None):
        """The averaging a computation given `averaging` actually uses:
        that one, else the record's own, else one whole frame per record.

        One statement of the fallback, read by the framing and by the
        degrees of freedom a PSD is stamped with, so the count cannot
        describe frames other than the ones averaged.
        """
        from .averaging import Averaging

        if averaging is None:
            averaging = self.averaging
        if averaging is None:
            averaging = Averaging.for_records(len(self.abscissa))
        return averaging

    def _spectral_frame(self, averaging=None):
        """(frequencies, one-sided scale, channel -> windowed frames).

        The framing a PSD and a CPSD share: the same rfft grid, the same
        one-sided scaling (DC and Nyquist not doubled), and the same
        cutting of a channel into frames, so the two cannot drift apart
        in what they mean by an average.

        Without an `averaging` each record is its own frame, whole and
        unwindowed — a burst-random run saved as 20 captures is already
        20 averages, and always was. With one, every record is cut into
        `frames` frames from `start`, each windowed, and all of them are
        pooled: 20 captures cut three ways average 60 frames.

        The window comes into the scaling as the sum of its squares,
        which is `n` for a rectangle — so the two paths agree exactly
        where they overlap.
        """
        step = _even_steps(self.abscissa, 'Computing PSDs')
        fs = 1.0 / step
        samples = len(self.abscissa)
        averaging = self._resolved_averaging(averaging)
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
        """Which capture each record is: 0 for a channel's first record,
        1 for its second, and so on.

        A channel is a `channel_key` group. The numbering is the order in
        which the PSD and CPSD computations combine the records, so the
        capture the averaging view shows is one of those averaged.

        Returns
        -------
        list of int
            One capture number per record.
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

    #: which stretches of this record are which kind of motion, for
    #: judging a replication of it segment by segment: marks as
    #: `replication_segments.replication_segments` takes them, on this
    #: record's own clock. Set by the segments reading, kept with the
    #: record so a reference is marked once.
    segments = None

    #: the stretch this history would be cut to, when something
    #: knows. Set in the truncate view; Truncate Data cuts with
    #: whatever is here at the time.
    truncation = None

    #: what each channel is to an FRF — {(DOF, quantity): role}, with
    #: role 'reference' (it drives), 'response' (it answers) or
    #: 'monitor' (recorded, judged as neither): the channel table's own
    #: three words (Brandon, 2026-09-25: *time data should be assigned
    #: reference, response, or monitor; FRFs computed between reference
    #: and response*). A channel not named here takes the default for its
    #: quantity (`default_role`) — so None is the default throughout. An importer that reads a
    #: channel table with the data seeds this from the table; the grid's
    #: Role column changes it. A role is a channel's, not a point's: a
    #: drive point's load cell and the accelerometer beside it are two
    #: channels, and either may be a reference — an accelerometer as a
    #: reference gives acceleration over acceleration, a
    #: transmissibility.
    roles = None

    def suggest_truncation(self) -> Any:
        """A truncation that spans the whole record.

        This is the starting position of the truncate view's handles,
        not a default that `truncate` uses: keeping everything changes
        nothing, so Truncate Data is refused until a span is set.

        Returns
        -------
        Truncation
            From the first sample's time to the last's.
        """
        from .truncate import Truncation

        abscissa = np.asarray(self.abscissa, dtype=float)
        return Truncation(float(abscissa[0]), float(abscissa[-1]))

    def truncate(self, truncation: Any = None) -> TimeHistory:
        """This record cut to a span (`core.truncate.truncate`), using the
        history's own `truncation` unless one is passed.

        Parameters
        ----------
        truncation : Truncation, optional
            The start and stop, in seconds on the record's own clock.
            Defaults to the record's own `truncation`.

        Returns
        -------
        TimeHistory
            The samples inside the span, every channel, with their
            original times.

        Raises
        ------
        ValueError
            If no truncation is passed or set.
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

    #: how a sweep's levels are read out of this record (a
    #: `core.sine.SineExtraction`), when something knows. Set in the
    #: sine view; Extract Sine Levels reads whatever is here, and with
    #: nothing here adopts the automatic setting (2026-09-30).
    sine_extraction = None

    def suggest_sine_extraction(self, specification: Any = None) -> Any:
        """The sine extraction setting a record starts from: automatic
        smoothing, chosen from the recording and the specification when
        the levels are read, with the clock refinement on.

        Parameters
        ----------
        specification : SineSweepSpecification, optional
            The sweep the levels would be read against. When given, the
            smoothing the automatic setting would choose is worked out
            now and stored in the setting as `chosen`, so the sine view
            can show it before anything is extracted.

        Returns
        -------
        SineExtraction
            The default setting.
        """
        from .sine import SineExtraction, suggest_cycles

        setting = SineExtraction()
        if specification is not None:
            from dataclasses import replace

            setting = replace(setting, chosen=suggest_cycles(
                self, specification, setting.target_db))
        return setting

    def suggest_filtering(self) -> Any:
        """A starting low-pass filter at a tenth of the sample rate,
        order 4.

        A low-pass, because removing noise above the content is the most
        common case; the filter view also offers high-pass and band-pass.
        The corner is a choice, not a detection, since nothing in the
        record says where its content stops being signal. At a tenth of
        the sample rate the integrate and differentiate round trip was
        measured at about 2 % RMS error (`core.filters`), and the corner
        is below the mounted-resonance range of a shock accelerometer.
        Check it in the filter view before using it.

        Returns
        -------
        Filtering
            The suggested filter.
        """
        from .filters import Filtering

        return Filtering(high=self.sample_rate / 10.0)

    def filter(self, filtering: Any = None) -> TimeHistory:
        """This record filtered (`core.filters.filtered`): low-pass,
        high-pass or band-pass, as the filtering describes, using the
        history's own `filtering` unless one is passed.

        Parameters
        ----------
        filtering : Filtering, optional
            The pass-band edges and order. Defaults to the record's
            own `filtering`.

        Returns
        -------
        TimeHistory
            Every channel filtered, with zero phase shift.

        Raises
        ------
        ValueError
            If no filtering is passed or set.
        """
        from .filters import filtered

        chosen = self.filtering if filtering is None else filtering
        if chosen is None:
            raise ValueError('no filter to apply: set `filtering` first, '
                             'or pass one')
        return filtered(self, chosen)

    def resample(self, rate: float, *, anti_alias: bool = True,
                 order: int | None = None) -> TimeHistory:
        """This record at the lower sample rate `rate`
        (`core.resample.resample`): every channel through a zero-phase
        low-pass at the new rate over 2.56, then every q-th sample kept.

        `core.resample.read_resample` reads what the resampling does on
        a channel's spectrum, and `visualdynamics.plot.plot_resample`
        draws it.

        Parameters
        ----------
        rate : float
            The new sample rate, in Hz, a whole division of this
            record's.
        anti_alias : bool, default True
            Filter before decimating. Without the filter, anything the
            record holds above the new Nyquist frequency folds into the
            new band, and an `core.resample.AliasWarning` says so.
        order : int, optional
            The anti-alias filter's design order,
            `core.resample.ORDER` when omitted.

        Returns
        -------
        TimeHistory
            The record at `rate`.

        Raises
        ------
        ValueError
            If the record is not evenly sampled, or `rate` is not a
            whole division of its sample rate.
        """
        from .resample import ORDER, resample

        return resample(self, rate, anti_alias=anti_alias,
                        order=ORDER if order is None else order)

    def integrate(self, drift_corner: Any = ...) -> TimeHistory:
        """One integration: acceleration to velocity, velocity to
        displacement (`core.filters.integrate`).

        Parameters
        ----------
        drift_corner : float or None, optional
            High-pass corner in Hz applied after integrating, so a sensor
            bias does not become a ramp. Omitted (`...`), the default
            corner is used; `None` integrates with no high-pass.

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
        """One differentiation: displacement to velocity, velocity to
        acceleration (`core.filters.differentiate`).

        Returns
        -------
        TimeHistory
            The differentiated channels.
        """
        from .filters import differentiate

        return differentiate(self)

    def time_statistics(self, record: int = 0, **options: Any
                        ) -> TimeStatistics:
        """Read one channel's statistics in the time domain: the
        moments, the crest factor and its peak, the excursions at the
        peak, the samples on the rail and those not finite, the tails
        beyond 3, 4 and 5 sigma against the Gaussian's, the crossing
        rate counted and from the spectrum, the expected largest peak
        and the kurtosis frame by frame, each judged against its band.

        `core.time_statistics.record_statistics` does the reading, over
        the span the kurtosis bars read, and says how;
        `visualdynamics.plot.plot_time_statistics` draws it.

        Parameters
        ----------
        record : int, default 0
            Which channel.
        **options
            `full_scale` (the channel's range, in SI), `margin`,
            `frame`, `duration` and the bands, as
            `core.time_statistics.time_statistics` takes them.

        Returns
        -------
        TimeStatistics
            Every number, in SI as the record is stored, the bands and
            the flags.
        """
        from .time_statistics import record_statistics
        return record_statistics(self, record, **options)

    def srs_windows(self, shocks: Sequence[Any] | None = None
                    ) -> list[Any]:
        """The windows an SRS of this record would be computed over.

        The record's shocks; if it has none, its averaging frames; if it
        has neither, the whole record as one window. Every window is
        clipped to the record. `compute_srs` and the shock panel both use
        this.

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
            # a window's second field is its duration: the frame's stop
            # went in there, and every frame but the first ran long by its
            # own start (2026-10-13, found by the docstring pass)
            windows = tuple(Shock(start, stop - start) for start, stop
                            in self.averaging.frame_bounds(rate))
        if not windows:
            windows = (Shock(0.0, samples / rate),)
        return [w.clipped(samples, rate) for w in windows]

    def srs_band(self, shocks: Sequence[Any] | None = None
                 ) -> tuple[float, float]:
        """(low, high) in Hz: the band the SRS windows can support.

        The low end is the frequency whose period is the shortest window
        (at least 1 Hz), since every curve uses one frequency grid and
        the shortest window has to hold a cycle of the lowest frequency.
        The high end is a fifth of the sample rate; above it the
        ramp-invariant filter would be asked about frequencies the record
        cannot resolve. `compute_srs` uses this band when none is given,
        and the shock panel shows it.

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

        The windows are stored on the history, as the averaging is: pass
        `shocks` to override them. Otherwise the record's shocks are
        used, or its averaging frames when it has no shocks, or the whole
        record as one window when it has neither (`srs_windows`). A
        specification has neither, and is one window.

        There is one curve per channel per event, never an average across
        events, because a shock test is judged on the worst shock and the
        mean of four shocks describes none of them. The event a curve came
        from is in its `block` label ('shock 2').

        The band defaults to `srs_band`: from the lowest frequency the
        shortest window holds a cycle of, up to a fifth of the sample
        rate.

        Parameters
        ----------
        shocks : sequence of Shock, optional
            The windows to use in place of the record's own.
        low, high : float, optional
            The natural-frequency band, in Hz.
        per_octave : int, optional
            Frequency lines per octave.
        q : float, optional
            The oscillator amplification; `srs.DEFAULT_Q` when omitted.
        kind : str, default 'maximax'
            Which peak to keep: 'maximax' (largest magnitude of either
            sign), 'positive' or 'negative' (reported as a positive
            magnitude).

        Returns
        -------
        Srs
            One curve per channel per shock (a `ShockSpecification` for
            a target, see `srs_type`).

        Raises
        ------
        ValueError
            If `kind` is not one of `srs.KINDS`.
        """
        from .srs import KINDS

        if kind not in KINDS:
            raise ValueError(f'an SRS keeps one of {", ".join(KINDS)}; '
                             f'{kind!r} is none of them')
        return self.compute_srs_polarities(shocks, low, high, per_octave,
                                           q)[kind]

    def compute_srs_polarities(self, shocks: Sequence[Any] | None = None,
                               low: float | None = None,
                               high: float | None = None,
                               per_octave: int | None = None,
                               q: float | None = None) -> dict[str, Srs]:
        """The positive, negative and maximax spectra together, from one
        pass of the filter.

        The windows, the band and the curves are those of `compute_srs`,
        which is this call keeping one of the three. Each oscillator's
        highest and lowest response come out of the same pass
        (`srs.peaks`), so the three spectra cost what one does, and they
        are worth reading together: a shock whose response is nearly all
        in one direction shows it as a gap between the positive and the
        negative spectrum that the maximax alone hides. The negative
        spectrum is reported as a positive magnitude, so the two are
        drawn on one log axis.

        Parameters
        ----------
        shocks : sequence of Shock, optional
            The windows to use in place of the record's own.
        low, high : float, optional
            The natural-frequency band, in Hz; `srs_band` when omitted.
        per_octave : int, optional
            Frequency lines per octave; `srs.PER_OCTAVE` (12) when
            omitted.
        q : float, optional
            The oscillator amplification; `srs.DEFAULT_Q` when omitted.

        Returns
        -------
        dict of str to Srs
            'positive', 'negative' and 'maximax', in that order, each one
            curve per channel per shock and each carrying its own `kind`.
        """
        # Added with the shock check (2026-10-16), which draws the
        # positive and negative spectra together: one `compute_srs` per
        # polarity would run the filter three times over the same record
        # for numbers the first pass already holds. `compute_srs` keeps
        # one of these, so the windows, band and labels have one
        # statement.
        from .srs import DEFAULT_Q, KINDS, PER_OCTAVE, octave_frequencies
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

        rows = {kind: [] for kind in KINDS}
        dofs, dims, hints, blocks = [], [], [], []
        for i, record in enumerate(self.ordinate):
            for k, window in enumerate(windows):
                first, last = window.bounds(rate)
                highest, lowest = srs_peaks(record.real[first:last],
                                            frequencies, rate, q=q)
                rows['positive'].append(highest)
                rows['negative'].append(-lowest)
                rows['maximax'].append(np.maximum(highest, -lowest))
                dofs.append(self.response_dof[i])
                dims.append(self.ordinate_dim[i])
                hints.append(self.dimension_hint[i])
                blocks.append(f'shock {k + 1}' if named
                              else (None if self.block is None
                                    else self.block[i]))
        return {kind: self.srs_type()(
            frequencies, np.asarray(rows[kind], dtype=np.float64),
            response_dof=list(dofs), ordinate_dim=list(dims),
            dimension_hint=list(hints),
            block=(list(blocks) if named or self.block is not None
                   else None),
            q=q, kind=kind,
            comment=f'{kind} SRS at Q={q:g}') for kind in KINDS}

    def compute_psds(self, averaging: Averaging | None = None) -> Psd:
        """One-sided auto-power spectral density per channel, averaged
        across the frames.

        Welch's method: each frame is windowed and transformed, and
        Gxx = 2|X|²/(fs·S) with S the sum of the window's squares (N for
        a rectangular window). DC and Nyquist have no mirror image to fold
        in and are |X|²/(fs·S), as scipy's welch computes them, so the
        spectrum sums to the record's mean square. The frames' powers are
        averaged. Power does not depend on phase, so the random phase of
        burst random excitation does not affect the result. Values are in
        (SI unit)²/Hz with dimensions such as
        'acceleration**2/frequency'; a channel with undefined units stays
        undefined, with its hint squared the same way.

        Each channel's statistical degrees of freedom, discounted for
        overlap (`Averaging.statistical_dof`), are stored on the result
        as `Psd.statistical_dof`, so the PSD can say what its average
        is worth (`Psd.confidence_band`) without the history beside it,
        and how the window ties neighboring lines together
        (`Averaging.line_correlation`) as `Psd.line_correlation`, so a
        sum of adjacent lines can say what it is worth too.

        Parameters
        ----------
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.

        Returns
        -------
        Psd
            One auto-power spectral density per channel (a
            `Specification` for a target, see `psd_type`).

        Raises
        ------
        ValueError
            If the samples are not evenly spaced, or the frames do not
            fit in the record.
        """
        frequencies, scale, groups = self._spectral_frame(averaging)
        used = self._resolved_averaging(averaging)
        rows, dofs, dims, hints, counts = [], [], [], [], []
        for (dof, dim, hint), windowed in groups.items():
            frames = scipy_fft.rfft(workers=-1, x=windowed, axis=1)
            rows.append(np.mean(np.abs(frames) ** 2, axis=0) * scale)
            dofs.append(dof)
            dims.append(f'{dim}**2/frequency' if dim != UNKNOWN
                        else UNKNOWN)
            hints.append(f'{hint}**2/frequency' if hint else None)
            # each of the channel's records was cut into the same
            # frames and pooled, so the count is per channel: a history
            # put together from unequal captures has unequal counts
            records = len(windowed) // used.frames
            counts.append(used.statistical_dof(records=records).equivalent)
        out = self.psd_type()(
            frequencies, np.asarray(rows, dtype=np.float64),
            response_dof=dofs, ordinate_dim=dims,
            dimension_hint=hints, comment='averaged PSD')
        # computed from a record, so a density per bin whatever class
        # it came back as — the PSD of a target is a Specification, and
        # a Specification is otherwise a power law through breakpoints
        out.interpolation = 'bin'
        out.statistical_dof = np.asarray(counts, dtype=np.float64)
        out.line_correlation = used.line_correlation()
        return out

    def psd_type(self) -> type[Psd]:
        """The class of a PSD computed from this history.

        `Psd` for an ordinary record. A target's PSD is still a target:
        the PSD of a waveform the test article was required to see is the
        spectrum it was required to see, so `TransientSpecification`
        overrides this to return `Specification`. Otherwise a requirement
        and a measurement would be two objects of the same class, told
        apart only by name.

        Returns
        -------
        type
            A `Psd` subclass.
        """
        # Overridden in the class rather than decided by the caller, so a
        # script and the app cannot disagree.
        return Psd

    def srs_type(self) -> type[Srs]:
        """The class of an SRS computed from this history: `Srs`, or
        `ShockSpecification` for a target (see `psd_type`).

        Returns
        -------
        type
            An `Srs` subclass.
        """
        return Srs

    def compute_cpsds(self, averaging: Averaging | None = None) -> Psd:
        """The full cross-spectral density matrix, averaged across the
        frames: every channel against every channel.

        Gxy = 2·conj(X)·Y/(fs·S), not doubled at DC and Nyquist, which
        equals `compute_psds` on the diagonal, where conj(X)·X is |X|².
        The cross terms describe how two channels move together, which a
        PSD does not.

        The records are laid out as an imported matrix is: one record per
        (response, reference) pair, row by row, so an n-channel history
        gives n² records shown as a grid. A cross term's dimension is the
        product of the two, `a*b/frequency`, and a diagonal term's is
        `a**2/frequency`.

        Parameters
        ----------
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.

        Returns
        -------
        Psd
            The full cross-spectral matrix, complex, with reference DOFs.

        Raises
        ------
        ValueError
            If the samples are not evenly spaced, or the frames do not
            fit in the record.
        """
        frequencies, scale, groups = self._spectral_frame(averaging)
        keys = list(groups)
        frames = {key: scipy_fft.rfft(workers=-1, x=windowed, axis=1)
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
        """The DOFs this history appears to have been driven at: those
        with a force or voltage record (`EXCITATION_DIMS`).

        This is a guess from the quantities alone and only a default.
        What makes a channel a drive is that the controller had a
        feedback device on it, which the channel table records and a time
        history does not.

        Returns
        -------
        list of str
            The DOFs, each once, in record order.
        """
        return list(dict.fromkeys(
            dof for dof, dim in zip(self.response_dof, self.ordinate_dim)
            if dim in self.EXCITATION_DIMS))

    #: the three things a channel can be to an FRF, as the channel
    #: table names them
    ROLES = ('reference', 'response', 'monitor')

    #: the default when nothing has said (Brandon, 2026-09-25): a force
    #: is a reference, a motion is a response, and anything else —
    #: volts, pressure, strain, temperature, a quantity not declared —
    #: is a monitor. A moment is a force about an axis and a rotation a
    #: motion about one, so the rotational quantities follow their
    #: translational ones.
    REFERENCE_DIMS = ('force', 'moment')
    RESPONSE_DIMS = ('acceleration', 'velocity', 'length',
                     'angular_acceleration', 'angular_velocity', 'angle')

    @classmethod
    def default_role(cls, quantity: str) -> str:
        """The role a channel measuring `quantity` has when nothing
        has said otherwise.

        Parameters
        ----------
        quantity : str
            What the channel measures, as `ordinate_dim` names it.

        Returns
        -------
        str
            'reference', 'response' or 'monitor'.
        """
        if quantity in cls.REFERENCE_DIMS:
            return 'reference'
        if quantity in cls.RESPONSE_DIMS:
            return 'response'
        return 'monitor'

    def channel_identities(self) -> list[tuple[str, str]]:
        """Every channel as (DOF, quantity), in order of first record.
        `roles` names channels by this pair.

        Returns
        -------
        list of (str, str)
            One entry per channel.
        """
        return list(dict.fromkeys(zip(self.response_dof, self.ordinate_dim)))

    def channel_roles(self, stated: bool = True
                      ) -> dict[tuple[str, str], str]:
        """Every channel's role, {(DOF, quantity): role}.

        The role `roles` gives, or for a channel it does not name, the
        default for the quantity (`default_role`): a force is a
        reference, a motion a response, anything else a monitor. FRFs,
        multiple coherence and the grid's Role column all read this.

        Parameters
        ----------
        stated : bool, default True
            False gives the defaults alone, ignoring `roles`. The
            provenance fingerprint compares against these, so stating
            the roles the defaults would give does not mark derived
            objects out of date.

        Returns
        -------
        dict of (str, str) to str
            Every channel's role, in first-record order.
        """
        stated = {(str(dof), str(quantity)): str(role) for (dof, quantity),
                  role in ((self.roles or {}) if stated else {}).items()}
        return {identity: stated.get(identity,
                                     self.default_role(identity[1]))
                for identity in self.channel_identities()}

    def reference_channels(self, references: Sequence[Any] | None = None
                           ) -> list[tuple[str, str]]:
        """The channels an FRF or a coherence takes as references, as
        (DOF, quantity) pairs.

        When `references` is omitted, the channels whose role is
        'reference' (`channel_roles`). A call may name them instead, as
        (DOF, quantity) pairs or by DOF alone. A DOF alone picks the
        excitation channel (force or voltage) at that point, the force at
        a drive point and not the accelerometer beside it, which is what
        a controller computes against; if the point has no excitation
        channel, its first channel is used.

        Parameters
        ----------
        references : sequence of str or (str, str), optional
            References to use in place of the roles.

        Returns
        -------
        list of (str, str)
            The reference channels, each once; a name that matches no
            channel is left out, and the caller says so.
        """
        identities = self.channel_identities()
        if references is None:
            roles = self.channel_roles()
            return [identity for identity in identities
                    if roles[identity] == 'reference']
        chosen: list[tuple[str, str]] = []
        for item in references:
            if isinstance(item, str):
                at = [identity for identity in identities if identity[0] == item]
                driven = [identity for identity in at
                          if identity[1] in self.EXCITATION_DIMS]
                picked = (driven or at)[:1]
            else:
                dof, quantity = item
                picked = [identity for identity in identities
                          if identity == (str(dof), str(quantity))]
            for identity in picked:
                if identity not in chosen:
                    chosen.append(identity)
        return chosen

    def response_channels(self, references: Sequence[Any] | None = None
                          ) -> list[tuple[str, str]]:
        """The channels an FRF or a coherence explains, as (DOF, quantity)
        pairs: every channel that is neither a reference nor a monitor.

        With `references` named by the call, a channel whose role is
        'reference' but that the call left out is a response, so a script
        asking for one drive of two gets the other as a response. A
        monitor is left out either way.

        Parameters
        ----------
        references : sequence of str or (str, str), optional
            As `reference_channels` takes them.

        Returns
        -------
        list of (str, str)
            The response channels, in order of first record.
        """
        drives = set(self.reference_channels(references))
        roles = self.channel_roles()
        return [identity for identity in self.channel_identities()
                if identity not in drives and roles[identity] != 'monitor']

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

        The references and the responses are the channels' roles
        (`channel_roles`): FRFs run from references to responses, and a
        monitor is on neither side. `references` overrides the
        references for one call, by channel — (DOF, quantity) — or by
        bare DOF.
        """
        frequencies, scale, groups = self._spectral_frame(averaging)
        keys = list(groups)
        def indices(identities):
            # a channel key is (DOF, quantity, unit hint); the identity
            # is its first two, and a channel is one key
            found = []
            for identity in identities:
                at = [i for i, key in enumerate(keys) if key[:2] == identity]
                if at and at[0] not in found:
                    found.append(at[0])
            return found

        drives = indices(self.reference_channels(references))
        if not drives:
            wanted = (f'none of {list(references)} is in this history'
                      if references is not None else
                      'no channel has the role reference')
            raise ValueError(f'{what} needs reference channels; {wanted}')
        responses = indices(self.response_channels(references))
        if not responses:
            raise ValueError(
                f'{what} needs response channels; every channel is a '
                'reference or a monitor')

        spectra = np.asarray([scipy_fft.rfft(workers=-1, x=groups[key], axis=1)
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
        """The frequency response functions, one per response and reference
        pair.

        The three estimators differ in one assumption, where the noise
        is, and agree wherever there is little noise. They differ most
        where a measurement is worst, so the estimator is a choice.

        H1 assumes the noise is on the response. It is biased low at
        resonance, where the response is large and the force small, and
        it is what a controller computes and a modal fit expects.

            Gfx = Gff H,   so   H = Gff^-1 Gfx

        With one reference that is `Gfx / Gff`. With several it is the
        MIMO estimate, and the matrix inverse is required: two shakers
        driving one test article are correlated, and dividing each
        response by each drive separately would credit both with the
        same motion. A pseudoinverse is used instead of a solve, because
        two shakers can be very nearly the same drive and `Gff` is then
        close to singular; the pseudoinverse gives the least-squares
        answer.

        H2 assumes the noise is on the reference, and is biased high at
        anti-resonance for the same reason. With one reference it is
        `Gxx / Gxf`, one response at a time. With several references it
        needs as many equations as unknowns, which holds only when the
        system is square (as many responses as references). It is then
        the coupled form `Gxx * Gfx^-1` (Rocklin, Crowley and Vold,
        1985), computed the same way as sdynpy computes it. Because every
        response enters one matrix inverse, a channel's H2 depends on
        which other channels are in the set (measured at about 0.2 % on
        test signals with 5 % noise, more with more noise); H1 and Hv
        rows do not. A multi-reference set that is not square is refused;
        Hv answers the same question per response for any shape.

        Hv, the default, assumes noise on both. It is the total least
        squares fit: the null direction of

            [[Gff, Gfx], [Gxf, Gxx]]

        taken as the eigenvector of its smallest eigenvalue, per response
        and per line. It falls between H1 and H2 (strictly between
        wherever the coherence is less than one) and needs no claim about
        which instrument is better. It is the default because a test can
        rarely make that claim: an accelerometer on a structure and a
        force cell in the load path both have errors, in different
        places. A total least squares fit weighs a unit of error on the
        force against a unit of error on the acceleration, which are not
        the same thing; that is part of the standard formulation, and it
        is why Hv is a middle estimate and not a better one.

        The FRFs use the same frames `compute_psds` averages and the same
        references `compute_multiple_coherence` uses, so the coherence
        beside an FRF is that FRF's coherence.

        Parameters
        ----------
        references : sequence of str or (str, str), optional
            The reference channels, by DOF or as (DOF, quantity) pairs
            (see `reference_channels`). When omitted, the channels whose
            role is 'reference'.
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.
        method : str, default 'Hv'
            The estimator: 'Hv', 'H1' or 'H2'.

        Returns
        -------
        Frf
            One record per response and reference pair, every reference
            for one response before the next response.

        Raises
        ------
        ValueError
            If `method` is unknown, there are no reference or no response
            channels, there are no more averages than references, or H2
            is asked of a multi-reference set that is not square.

        References
        ----------
        The coupled multi-reference H2 form is Rocklin, Crowley and
        Vold's; the rest set out the three estimators and their
        errors.

        1. Rocklin, G. T., Crowley, J., & Vold, H. (1985). "A
           comparison of H1, H2 and Hv frequency response functions."
           *Proceedings of the International Modal Analysis Conference
           (IMAC)*.
        2. Ewins, D. J. (2000). *Modal Testing: Theory, Practice and
           Application*, 2nd ed. Research Studies Press. Which
           estimator biases which way, and why the bias is worst
           exactly where the measurement matters.
        3. Bendat, J. S., & Piersol, A. G. (2010). *Random Data:
           Analysis and Measurement Procedures*, 4th ed. Wiley. The
           single- and multiple-input frequency response estimates and
           their errors.
        4. Van Huffel, S., & Vandewalle, J. (1991). *The Total Least
           Squares Problem: Computational Aspects and Analysis*. SIAM.
           The fit Hv performs, and what it means to weigh error in
           one measured quantity against error in another.
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
        """How much of each response the references together account for.

        Ordinary coherence measures what one reference explains; multiple
        coherence measures what the whole set of references explains at
        once. In a MIMO test this is the useful measure: two shakers
        driving one test article are correlated with each other, so a
        response can have poor coherence with either one alone and still
        be fully accounted for by the pair.

        For a response x and references r,

            gamma^2 = (Grx^H Grr^-1 Grx) / Gxx

        the power of the best linear prediction of x from all the
        references at once, divided by the measured power. With one
        reference it reduces to the ordinary coherence. Values are
        limited to 0 to 1.

        The coherence uses the same frames `compute_psds` averages, so it
        covers the stretch of record the averaging view has set, and the
        same references and framing as `compute_frfs`, so the coherence
        beside an FRF is that FRF's coherence. A pseudoinverse is used
        instead of a solve, because two shakers can be very nearly the
        same drive and the reference matrix is then close to singular.

        Parameters
        ----------
        references : sequence of str or (str, str), optional
            The reference channels, by DOF or as (DOF, quantity) pairs
            (see `reference_channels`). When omitted, the channels whose
            role is 'reference'.
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.

        Returns
        -------
        MultipleCoherence
            One curve per response channel.

        Raises
        ------
        ValueError
            If there are no reference or no response channels, or no more
            averages than references.

        References
        ----------
        The multiple coherence function is standard; these are where
        it and the bias of an estimate of it are set out.

        1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data:
           Analysis and Measurement Procedures*, 4th ed. Wiley.
           Multiple-input systems: the multiple coherence function,
           read as the fraction of a response's power the whole set of
           references accounts for.
        2. Carter, G. C., Knapp, C. H., & Nuttall, A. H. (1973).
           "Estimation of the magnitude-squared coherence function via
           overlapped fast Fourier transform processing." *IEEE
           Transactions on Audio and Electroacoustics*, 21(4),
           337-344.
           [doi:10.1109/TAU.1973.1162496](https://doi.org/10.1109/TAU.1973.1162496)
           The bias of a coherence estimate against the number of
           averages, which is why a coherence from too few frames
           reads high.
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

    def read_transmissibility(self, response: Any,
                              reference: Any = None,
                              averaging: Averaging | None = None
                              ) -> TransmissibilityReading:
        """The ASD-ratio transmissibility of one response over one
        reference, beside H1, with H1's coherence and random error.

        The ratio is the root of the response's autospectrum over the
        reference's; H1 is their cross-spectrum over the reference's
        autospectrum. The two are tied by the coherence: the ratio is
        |H1| over the root of the coherence at every line, so it never
        reads under H1, and the gap between them says the coherence is
        under one without saying why. Noise on the response lifts the
        ratio and leaves H1; noise on the reference pulls H1 down by the
        coherence and the ratio by its root, so there the ratio is the
        nearer of the two. The error of |H1| follows from the coherence
        and the averages (`core.transmissibility` sets out both). The
        three spectra come
        from the frames `compute_psds` averages and `compute_frfs`
        estimates from, so this H1 is `compute_frfs(method='H1')`'s for
        the pair and this coherence the ordinary one
        `compute_multiple_coherence` gives against the one reference.

        The error is read at the averages the frames are worth, the
        equivalent count discounted for overlap that the PSD's
        confidence band uses (`Averaging.statistical_dof`, half its
        equivalent degrees of freedom), not the frames cut; the reading
        carries both.

        Parameters
        ----------
        response : str or (str, str)
            The response channel, by DOF or as a (DOF, quantity) pair.
        reference : str or (str, str), optional
            The input channel, the same way (a DOF alone picks the
            excitation channel there, as `reference_channels` does).
            When omitted, the one channel whose role is 'reference'.
        averaging : Averaging, optional
            How to cut the record into frames. Defaults to the record's
            own `averaging`, or one frame per record.

        Returns
        -------
        TransmissibilityReading
            H1, the ASD ratio and the coherence at every line, in SI as
            the record is stored, with the frames and the averages.

        Raises
        ------
        ValueError
            If there is not exactly one reference, the response is not
            a response channel of this history, a DOF given alone holds
            more than one quantity, there are fewer than
            two averages, or either channel holds a sample that is not a
            finite number in the frames read.
        """
        from .transmissibility import read_transmissibility

        named = None if reference is None else [reference]
        drives = self.reference_channels(named)
        if len(drives) != 1:
            wanted = ('no channel matches the reference given'
                      if reference is not None and not drives else
                      f'{len(drives)} channels have the role reference')
            raise ValueError(
                'a transmissibility is one response over one reference; '
                f'{wanted}. Name the reference.')
        frequencies, keys, drive, responses, cross, frames = \
            self._cross_spectral_frame('a transmissibility', drives,
                                       averaging)
        if isinstance(response, str):
            found = [i for i in responses if keys[i][0] == response]
        else:
            dof, quantity = response
            found = [i for i in responses
                     if keys[i][:2] == (str(dof), str(quantity))]
        if len({keys[i][1] for i in found}) > 1:
            # two quantities at one DOF (an accelerometer and a velocity
            # probe): which one a bare DOF meant depends on the row order,
            # so say so rather than read whichever came first
            raise ValueError(
                f'{response} holds more than one quantity ('
                + ', '.join(sorted({str(keys[i][1]) for i in found}))
                + '); name the response as a (DOF, quantity) pair')
        if not found:
            raise ValueError(
                f'{response!r} is not a response of this history against '
                f'{drives[0][0]}: the responses are '
                + ', '.join(sorted({keys[i][0] for i in responses})))
        x, y = drive[0], found[0]
        gxx, gyy = np.real(cross(x, x)), np.real(cross(y, y))
        for role, channel, power in (('reference', x, gxx),
                                     ('response', y, gyy)):
            # one sample that is not a number spreads through its frame's
            # transform to every line, and every line of the average
            # with it: read on, it would say the input has no power or
            # the coherence is nil, which is a statement about the
            # record's storage and not about the article
            if not np.isfinite(power).all():
                raise ValueError(
                    f'the {role} channel {keys[channel][0]} holds samples '
                    'that are not finite numbers in the frames read, so '
                    'its spectrum is not a number at any line; trim the '
                    'record or move the frames past them')
        used = self._resolved_averaging(averaging)
        # the frames of every record were pooled, each record cut the
        # same way, so the count is the averaging's times the records
        worth = used.statistical_dof(records=max(frames // used.frames, 1))
        return read_transmissibility(
            frequencies, gxx, gyy, cross(x, y), frames,
            worth.equivalent / 2.0,
            response=keys[y][0], reference=keys[x][0],
            dimension=self._ratio(keys[y][1], keys[x][1]))

    def to_sep005(self, name: str | None = None) -> list[dict[str, Any]]:
        """This record as SEP 005 timeseries, the sdypy interchange format
        (`io.sep005` reads them back).

            timeseries = history.to_sep005('run 4')

        Returns the standard's list form: usually one dict, and one per
        unit and quantity where the channels mix. The sdypy validator
        requires a series' ``unit_str`` to be a single string, so
        accelerometers and a force gauge cannot be one valid series; the
        standard's list of series is for that case. A split series has
        its unit in its name, so the series can be told apart.

        Values are written as they are held: SI where units are defined,
        with ``unit_str`` naming the SI unit, and the file's raw numbers
        where they are not, with ``unit_str`` empty (the standard allows
        an empty unit). ``fs`` gives the sample rate when the samples are
        evenly spaced; otherwise the ``time`` vector is written, which
        the standard also accepts. ``quantity`` is set where the standard
        has a letter for what a series measures.

        Parameters
        ----------
        name : str, optional
            A name for the series. Defaults to the first non-empty
            comment, or 'time history', since a SEP 005 series must be
            named.

        Returns
        -------
        list of dict
            One SEP 005 timeseries mapping per unit and quantity, each
            holding every channel of that unit.
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


def _read_half_power(data: Any, near: float, record: int,
                     within: tuple[float, float] | None,
                     warn: bool) -> HalfPower:
    """`Spectrum.half_power` and `Frf.half_power`: one reading of one
    record's magnitude, warned at the line that called the method."""
    from .half_power import read_half_power
    return read_half_power(data.abscissa, data.ordinate[record], near,
                           within=within, warn=warn, stacklevel=3)


class Spectrum(DataArray):
    """A linear spectrum: amplitude and phase at each frequency line.

    It is the complex average of a record's frames, not a power average,
    so content whose phase is random from frame to frame averages toward
    zero. That makes it suited to a deterministic signal and not to burst
    random. A `Psd` is the power average and does not have that property.
    """

    function_type = 12
    abscissa_dim = 'frequency'
    complex_ordinate = True

    def animate(self, geometry: Any, frequency: float | None = None,
                **kwargs: Any) -> Any:
        """Animate the operating deflection shape at one frequency line on
        a geometry, as the app does. The strongest line is used unless
        `frequency` picks another.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        frequency : float, optional
            Which frequency line. The strongest when omitted.
        **kwargs
            Passed to `visualdynamics.viz.animate.animate_ods`
            (`screenshot`, `movie`, `scale` and others).

        Returns
        -------
        object
            The plotter, or the path written.
        """
        from ..viz.animate import animate_ods
        return animate_ods(geometry, self, frequency=frequency, **kwargs)

    def half_power(self, near: float, record: int = 0, *,
                   within: tuple[float, float] | None = None,
                   warn: bool = True) -> HalfPower:
        """Read Q from the half-power width of the peak nearest `near`
        on one record's magnitude.

        `core.half_power.read_half_power` does the reading and says how;
        `visualdynamics.plot.plot_half_power` draws it on the record.

        Parameters
        ----------
        near : float
            A frequency on the peak to read, in Hz.
        record : int, default 0
            Which record.
        within : tuple of float, optional
            A (low, high) band in Hz whose tallest line is the peak.
        warn : bool, default True
            Warn a missing crossing or too few lines across the peak as
            a `HalfPowerWarning`; the notes are on the reading either
            way.

        Returns
        -------
        HalfPower
            The peak's frequency and height (in SI, as the record is
            stored), its half-power crossings, Q and damping ratio.
        """
        return _read_half_power(self, near, record, within, warn)


class Frf(DataArray):
    """Frequency response function: response per unit reference.

    Every record has a reference DOF. Units are declared as a response
    unit and a reference unit (`define_units('g', 'lbf')`), and the
    dimension is their ratio, such as 'acceleration/force'.
    """

    function_type = 4
    abscissa_dim = 'frequency'
    complex_ordinate = True
    needs_reference = True

    @classmethod
    def units_for(cls, dimension, system=SI):
        """The (response unit, reference unit) a unit system gives for an
        FRF dimension: 'acceleration/force' in SI is ('m/s**2', 'N').
        See `DataArray.units_for`.

        Parameters
        ----------
        dimension : str
            An `ordinate_dim`, such as 'acceleration/force'.
        system : UnitSystem, optional
            The unit system; SI when omitted.

        Returns
        -------
        tuple of (str, str)
            The response unit and the reference unit.
        """
        response, _, reference = dimension.partition('/')
        return system.unit(response), system.unit(reference)

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
        """Plot the CMIF, as the fitting screen draws it. With `shapes`,
        the modal model's synthesized CMIF is drawn dashed over the
        measured one.

        Parameters
        ----------
        shapes : ShapeSet, optional
            A modal fit, drawn as the synthesized CMIF over the measured one.
        **kwargs
            Passed to `visualdynamics.plot.plot_cmif`.

        Returns
        -------
        object
            The plot pane, or the path when a file is written.
        """
        from ..plot import plot_cmif
        return plot_cmif(self, shapes, **kwargs)

    def conditioning(self, records: Sequence[int] | None = None, *,
                     limit: float | None = None,
                     progress: Callable[[int, int], None] | None = None
                     ) -> Conditioning:
        """The singular values and condition number of this FRF matrix at
        every line, and the lines where it is ill-conditioned
        (`core.matrix_checks.read_conditioning`).

        The records make the matrix of responses by references; two
        references or more, every response against every reference, one
        unit. Where the condition number is over `limit`, the drives
        reach the responses in nearly the same proportions and an
        inverse of this matrix, which a controller makes to drive them,
        amplifies the error in it by up to that much.

        Parameters
        ----------
        records : sequence of int, optional
            The records that make the matrix; all of them when omitted.
        limit : float, optional
            The condition number lines are flagged over;
            `core.matrix_checks.CONDITION_LIMIT` when omitted.
        progress : callable, optional
            Told ``(0, core.progress.BUSY)`` before the decomposition;
            the app's progress bar uses it.

        Returns
        -------
        Conditioning
            The singular values at every line, with the condition
            number, the effective rank and the flagged bands.

        Raises
        ------
        ValueError
            For an empty selection, one reference, records in different
            units, or an incomplete matrix.
        """
        from .matrix_checks import CONDITION_LIMIT, read_conditioning

        return read_conditioning(
            self, records, limit=CONDITION_LIMIT if limit is None else limit,
            progress=progress)

    def half_power(self, near: float, record: int = 0, *,
                   within: tuple[float, float] | None = None,
                   warn: bool = True) -> HalfPower:
        """Read Q from the half-power width of the peak nearest `near`
        on one record's magnitude.

        `core.half_power.read_half_power` does the reading and says how;
        `visualdynamics.plot.plot_half_power` draws it on the record.

        Parameters
        ----------
        near : float
            A frequency on the peak to read, in Hz.
        record : int, default 0
            Which record.
        within : tuple of float, optional
            A (low, high) band in Hz whose tallest line is the peak.
        warn : bool, default True
            Warn a missing crossing or too few lines across the peak as
            a `HalfPowerWarning`; the notes are on the reading either
            way.

        Returns
        -------
        HalfPower
            The peak's frequency and height (in SI, as the record is
            stored), its half-power crossings, Q and damping ratio.
        """
        return _read_half_power(self, near, record, within, warn)

    def animate(self, geometry: Any, frequency: float | None = None,
                **kwargs: Any) -> Any:
        """Animate the operating deflection shape at one frequency line on
        a geometry, as the app does. The strongest line is used unless
        `frequency` picks another.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        frequency : float, optional
            Which frequency line. The strongest when omitted.
        **kwargs
            Passed to `visualdynamics.viz.animate.animate_ods`
            (`screenshot`, `movie`, `scale` and others).

        Returns
        -------
        object
            The plotter, or the path written.
        """
        from ..viz.animate import animate_ods
        return animate_ods(geometry, self, frequency=frequency, **kwargs)


class _Bands:
    """The bins a frequency-domain array's lines stand for — shared by a
    spectral density and a coherence, one rule for both.

    `bandwidth` is the width of the band each line stands for when the
    lines are not evenly spaced: None for a narrowband array, whose bins
    are the midpoints between neighbors and exactly inferable; set by an
    octave-banded one, whose bins are geometric and whose edges are a
    standard's rather than its neighbors' — read off the centers they
    are a hair out at every band and plainly wrong at the two ends. A
    coherence computed from a banded CPSD carries the CPSD's bands and
    draws flat across them the way it does (2026-09-26).
    """

    bandwidth: np.ndarray | None = None

    def _take_bandwidth(self, bandwidth: ArrayLike | None) -> None:
        if bandwidth is None:
            return
        bandwidth = np.asarray(bandwidth, dtype=np.float64)
        if bandwidth.shape != self.abscissa.shape:
            raise ValueError(
                f'bandwidth has shape {bandwidth.shape}, expected '
                f'{self.abscissa.shape} to match the abscissa')
        self.bandwidth = bandwidth

    def bin_widths(self) -> np.ndarray:
        """The width of every line's bin, in hertz.

        `bandwidth` when the object has one (octave bands); otherwise the
        distance between the midpoints to the neighboring lines.

        Returns
        -------
        numpy.ndarray
            One width per line.
        """
        if self.bandwidth is not None:
            return self.bandwidth
        return np.gradient(np.asarray(self.abscissa, dtype=float))

    def bin_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(left, right) edges of every line's bin, in hertz; see
        `octave.bin_bounds`.

        Returns
        -------
        tuple of numpy.ndarray
            The left edges and the right edges, one per line.
        """
        from .octave import bin_bounds

        return bin_bounds(np.asarray(self.abscissa, dtype=float),
                          self.bandwidth)

    def bin_edges(self) -> np.ndarray:
        """The edges for a step plot: every bin's left edge, then the last
        bin's right edge, so one more than there are lines.

        Returns
        -------
        numpy.ndarray
            The edges, in hertz.
        """
        left, right = self.bin_bounds()
        return np.concatenate([left, right[-1:]])


class _CoherenceBase(_Bands, DataArray):
    """How much of a response its references account for, from 0 to 1.

    A coherence is a ratio of spectra and so is always dimensionless,
    whatever dimension a caller passes. It never needs units declared,
    and the Define Units step does not offer it.
    """

    abscissa_dim = 'frequency'
    complex_ordinate = False
    log_ordinate = False        # a 0..1 ratio says nothing on a log axis
    ordinate_limits = (0.0, 1.05)

    def __init__(self, *args: Any, bandwidth: ArrayLike | None = None,
                 **kwargs: Any) -> None:
        # whatever a caller passes, a coherence is dimensionless
        kwargs['ordinate_dim'] = 'dimensionless'
        kwargs.setdefault('ordinate_unit', 'dimensionless')
        super().__init__(*args, **kwargs)
        self._take_bandwidth(bandwidth)

    def plot_map(self, **kwargs: Any) -> Any:
        """Plot the coherence map: frequency across, channel down, values
        from 0 to 1.

        Parameters
        ----------
        **kwargs
            Passed to `visualdynamics.plot.plot_coherence_map`.

        Returns
        -------
        object
            The plot pane, or the path when a file is written.
        """
        from ..plot import plot_coherence_map
        return plot_coherence_map(self, **kwargs)


class Coherence(_CoherenceBase):
    """Ordinary coherence: one response against one reference.

    UFF dataset 58 function type 6, and multiple coherence has a code of
    its own (see `MultipleCoherence`). The two are separate classes, not
    one class with an optional reference: ordinary coherence is a matrix
    of pairs and has a reference DOF on every record, while multiple
    coherence is one curve per response.
    """

    function_type = 6
    needs_reference = True


class MultipleCoherence(_CoherenceBase):
    """How much of a response all the references together account for.

    One curve per response channel, with no reference DOF, since it is
    computed over the whole set of references. A Rattlesnake modal survey
    reports this, and so does its random environment against its control
    channels. UFF dataset 58 function type 26.
    """
    # visualdynamics wrote function type 6 here for a while, which claimed
    # ordinary coherence and a reference DOF that was never there.

    function_type = 26


def paired_channels(signal: Any, floor: Any) -> list[tuple[int, int, str, str]]:
    """The channels two densities share, as (signal row, floor row, DOF,
    quantity).

    Channels are matched by (DOF, quantity), never by index or by DOF
    alone, since a drive point has an accelerometer and a load cell at
    one DOF. Channels only one side has are left out. Both densities must
    be on the same frequency lines; neither a ratio nor an overlay means
    anything across different bins, so that is refused instead of
    interpolated. The signal-to-noise ratio and the overlay of the two
    densities both pair channels here.

    Parameters
    ----------
    signal, floor : Psd
        The two densities, such as a run and its noise floor.

    Returns
    -------
    list of (int, int, str, str)
        One entry per shared channel.

    Raises
    ------
    ValueError
        If the frequency lines differ or no channel is shared.
    """
    # The pairing rule under every two-density reading (Brandon,
    # 2026-08-25). Keyed by DOF alone, the dict collapsed a drive point's
    # two channels and lost half the drive-point channels from the first
    # ratio drawn. Here rather than inside `snr.signal_to_noise` because
    # the report draws the overlay above the ratio, and a figure that
    # paired its channels differently from the figure below it would be
    # two claims about one measurement.
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


class Psd(_Bands, DataArray):
    #: the width of the band each line stands for, when the lines are
    #: not evenly spaced. `None` for a narrowband spectrum, where the
    #: bins are the midpoints between neighbors and inferring them is
    #: exact. An octave-band spectrum sets it: its bins are geometric,
    #: its edges are a standard's and not its neighbors', and reading
    #: them off the centers would be a hair out at every band and wrong
    #: at the two ends.
    """Power spectral density. The declared unit is the engineering unit
    whose square per hertz the values are in (declare 'g' for g²/Hz).

    The class allows complex values because a cross spectrum is complex:
    the phase between two channels is most of what a CPSD is for. An
    autospectrum is a magnitude squared and real, so an object with no
    imaginary content (`has_phase`) stores a real array, half the size.

    `interpolation` says how values are read between the lines: 'bin'
    (constant across each line's bin, drawn as steps) or 'log_log' (a
    power law between breakpoints). `area`, the plots and octave banding
    all follow it. `scale_db` is the decibels added to this spectrum when
    it is compared against a specification, and nowhere else; None means
    it is detected from the data.

    Parameters
    ----------
    *args, **kwargs
        As for `DataArray`.
    bandwidth : array-like, optional
        The width of each line's band, for lines that are not evenly
        spaced (octave bands). Omitted for a narrowband spectrum.
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

    #: The statistical degrees of freedom of each record's average,
    #: discounted for overlap (`core.averaging.statistical_dof`): one
    #: number per record, or None when nothing says how the spectrum
    #: was averaged. `TimeHistory.compute_psds` sets it. Merging
    #: spectra keeps their counts when every one has them, since each
    #: record's average is unchanged. A spectrum read from a file or
    #: banded onto octaves has none, because its count is not the one
    #: its source frames had: banding sums lines and raises it, and a
    #: guess would draw a band the data does not have. These are
    #: statistical degrees of freedom, not the measurement DOFs in
    #: `response_dof`.
    statistical_dof: np.ndarray | None = None

    #: How the averaged power at lines 0, 1, 2, ... apart varies
    #: together, from the window, hop and frame count it was averaged
    #: with (`core.averaging.line_correlation`): 1 first, then the
    #: correlation at each separation, the same for every record since
    #: every record was cut into the same frames. None when nothing
    #: says how the spectrum was averaged, on the same terms as
    #: `statistical_dof`. A sum of adjacent lines is worth less than
    #: their counts added by as much as these say (`core.chance`).
    line_correlation: np.ndarray | None = None

    def confidence_band(self, level: float = 0.90,
                        dof: float | ArrayLike | None = None
                        ) -> tuple[Psd, Psd]:
        """The chi-square confidence band on this averaged spectrum: two
        PSDs, the band's lower and upper edges, in this spectrum's units.

        Each record's edges are its values times the factors
        `core.averaging.chi_square_factors` gives at its degrees of
        freedom, so the band follows the spectrum line for line, and on
        log axes it is the spectrum moved down and up by a fixed number
        of decibels. The true spectrum lies inside it at the stated
        confidence, line by line.

        Parameters
        ----------
        level : float, optional
            The confidence, a fraction: 0.90 by default, 0.95 for the
            stricter band.
        dof : float or array-like, optional
            The statistical degrees of freedom, one number for every
            record or one per record. Defaults to `statistical_dof`.

        Returns
        -------
        tuple of (Psd, Psd)
            The lower edge and the upper edge, each with this
            spectrum's records, units and lines.

        Raises
        ------
        ValueError
            If the spectrum carries no degrees of freedom and none are
            given, if it holds cross terms (whose error is not
            chi-square), if it is a `Specification` (a target, not an
            average: `confidence_refusal`), or if a count or the level
            is out of range.
        """
        from .averaging import chi_square_factors

        counts = self._band_counts(dof)
        values = np.real(self.ordinate)
        factors = np.array([chi_square_factors(n, level) for n in counts])
        edges = []
        for column, word in ((0, 'lower'), (1, 'upper')):
            fields = self.record_fields()
            fields['comment'] = ([f'{round(level * 100)} % {word} bound']
                                 * self.num_records)
            edge = Psd(self.abscissa.copy(),
                       values * factors[:, column][:, None],
                       bandwidth=(None if self.bandwidth is None
                                  else np.array(self.bandwidth)),
                       **fields)
            edge.interpolation = self.interpolation
            edges.append(edge)
        return edges[0], edges[1]

    def confidence_reading(self, level: float = 0.90,
                           dof: float | ArrayLike | None = None
                           ) -> list[ConfidenceBand]:
        """The chi-square band on each record, as the reading states it:
        one `core.averaging.ConfidenceBand` per record, holding the
        confidence, the degrees of freedom it was worked out at, the two
        factors and their decibels (`ConfidenceBand.describe` puts them
        in words), and the edges in this spectrum's units.

        The same band `confidence_band` returns as two PSDs, from the
        same counts by the same rule, so a number read here is the edge
        drawn on the plot.

        Parameters
        ----------
        level : float, optional
            The confidence, a fraction: 0.90 by default, 0.95 for the
            stricter band.
        dof : float or array-like, optional
            The statistical degrees of freedom, one number for every
            record or one per record. Defaults to `statistical_dof`.

        Returns
        -------
        list of ConfidenceBand
            One per record, in record order.

        Raises
        ------
        ValueError
            As `confidence_band` raises: no degrees of freedom known or
            given, a cross term among the records, a specification, or
            a count or the level out of range.
        """
        from .averaging import confidence_bands

        counts = self._band_counts(dof)
        values = np.real(self.ordinate)
        return [confidence_bands(values[i], counts[i], levels=[level])[0]
                for i in range(self.num_records)]

    def is_autospectrum(self, record: int | None = None) -> bool:
        """Whether a record is an autospectrum, a channel against itself:
        one DOF and one quantity on both sides. Only such a record has a
        chi-square band (`confidence_band`), a tone to read
        (`tone_refusal`) or a chance of reading outside a band
        (`core.chance`).

        Parameters
        ----------
        record : int, optional
            Which record; every record must be one when omitted.

        Returns
        -------
        bool
        """
        if record is not None:
            return self._is_auto(int(record))
        return self._cross_term() is None

    def _is_auto(self, record: int) -> bool:
        """Whether record `record` names one DOF and one quantity twice."""
        # A record is an autospectrum only when its two DOFs are one
        # and its two quantities are one: force against acceleration
        # at a drive point names 1X+ twice and is still a cross term,
        # whose real part alone the band would otherwise have been
        # drawn on (2026-10-16).
        response, reference = self.record_pair(record)
        first, second = channel_quantities(self.ordinate_dim[record])
        return response == reference and first == second

    def _cross_term(self) -> int | None:
        """The first record that is a cross term, or None."""
        return next((i for i in range(self.num_records)
                     if not self._is_auto(i)), None)

    def confidence_refusal(self) -> str | None:
        """Why this spectrum has no chi-square band, or None when it has
        one: `confidence_band` and `confidence_reading` raise this as
        their ValueError, and the application offers the band only where
        it is None.

        Two things are refused, whatever count is given. A cross term,
        whose magnitude has a different error law, so a band drawn on it
        as if chi-square would be a claim nobody computed. And a
        `Specification`, which is a target and not an average, even the
        spectrum of a transient specification, which is averaged from a
        waveform and carries the count of that averaging: the band says
        where the true spectrum of a measurement lies, and a requirement
        has no true spectrum other than itself. A missing count is not
        refused here, since it can be given (`confidence_band`'s `dof`).

        Returns
        -------
        str or None
            The reason, in words, or None.
        """
        if isinstance(self, Specification):
            return ('a specification is a target, not an average, and '
                    'has no chi-square band')
        cross = self._cross_term()
        if cross is not None:
            return ('a chi-square band is on an autospectrum; this PSD '
                    f'holds cross terms ({self.record_label(cross)})')
        return None

    def _band_counts(self, dof: float | ArrayLike | None) -> np.ndarray:
        """Each record's degrees of freedom for its band: `dof` spread
        over the records, or the spectrum's own. One statement of the
        refusals (`confidence_refusal`, then the count), read by
        `confidence_band` and `confidence_reading`."""
        # Refused rather than skipped, so the caller knows.
        refusal = self.confidence_refusal()
        if refusal is not None:
            raise ValueError(refusal)
        counts = self.statistical_dof if dof is None else dof
        if counts is None:
            raise ValueError(
                'this PSD does not say what it was averaged over; pass '
                'dof, or compute it from a time history')
        return np.broadcast_to(np.asarray(counts, dtype=float),
                               (self.num_records,))

    def principal_shapes(self, quantity: str | None = None
                         ) -> tuple[list[str], np.ndarray, str]:
        """The dominant shape of the cross-spectral matrix at each line, as
        (dofs, shapes (channels x lines), quantity).

        The channels of one quantity form a square Hermitian matrix at
        each line. The eigenvector of its largest eigenvalue, scaled by
        the square root of that eigenvalue, is the principal operating
        deflection shape: each channel's phase relative to the others,
        with no reference to choose. `quantity` picks the channels (the
        most common quantity when omitted). A set with no cross records
        is refused, since an autospectrum set has no phase (its display
        is the envelope), and so is a matrix with missing terms.

        Parameters
        ----------
        quantity : str, optional
            Which quantity, for a mixed object.

        Returns
        -------
        tuple of (list of str, numpy.ndarray, str)
            The DOF labels, the dominant shape at each line, and the
            quantity they are in.

        Raises
        ------
        ValueError
            If there are no cross records, no record names a quantity,
            fewer than two channels have the quantity, or the matrix is
            incomplete.
        """
        from .matrix_checks import spectral_matrix

        if self.reference_dof is None:
            raise ValueError('no cross records — an autospectrum set '
                             'has no phase')
        if quantity is None:
            kinds_present = [channel_quantities(self.known_dim(i))[0]
                             for i in range(self.num_records)]
            kinds_present = [k for k in kinds_present if k != UNKNOWN]
            if not kinds_present:
                raise ValueError('no record names a quantity')
            quantity = max(set(kinds_present), key=kinds_present.count)
        # the matrix assembled where the realizability and conditioning
        # readings assemble theirs, one channel per (DOF, quantity)
        keys, matrix, filled = spectral_matrix(self, quantity=quantity)
        channels = [dof for dof, _quantity in keys]
        n = len(channels)
        if n < 2:
            raise ValueError(f'fewer than two {quantity} channels')
        if not filled.all():
            raise ValueError(f'the {quantity} cross-spectral block is '
                             'incomplete')
        values, vectors = np.linalg.eigh(matrix)
        principal = (vectors[:, :, -1]
                     * np.sqrt(np.maximum(values[:, -1:], 0.0)))
        return channels, principal.T.copy(), quantity

    def animate(self, geometry: Any, frequency: float | None = None,
                quantity: str | None = None, **kwargs: Any) -> Any:
        """Animate this set on a geometry, as the app shows it.

        A CPSD (with cross records) animates its principal operating
        deflection shape (`principal_shapes`). An autospectrum set has no
        phase, so it shows the envelope instead: two copies deflected
        ±sqrt(PSD), colored by dB below the largest node at that line.
        The strongest line is used unless `frequency` picks another;
        `quantity` picks which measurement deflects the geometry.

        Parameters
        ----------
        geometry : Geometry
            The geometry to move.
        frequency : float, optional
            Which frequency line.
        quantity : str, optional
            Which quantity, for a mixed object.
        **kwargs
            Passed to `visualdynamics.viz.animate.animate_ods` or
            `animate_envelope` (`screenshot`, `movie` and others).

        Returns
        -------
        object
            The plotter, or the path written.
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

    def tone_refusal(self, record: int = 0) -> str | None:
        """Why a tone cannot be read off record `record`, or None when
        it can: `read_tone` raises this as its ValueError, and the
        application offers the reading only where it is None.

        A tone is read off an autospectrum (a record whose response and
        reference are one channel, with no phase) on evenly spaced
        lines; an octave-band spectrum, one written at breakpoints and
        a cross spectrum are refused.

        Parameters
        ----------
        record : int, default 0
            Which record.

        Returns
        -------
        str or None
            The reason, in words, or None.
        """
        from .tone_readout import line_spacing

        if self.bandwidth is not None or self.interpolation != 'bin':
            return ('a tone is read off the evenly spaced lines of an '
                    'averaged spectrum; this one is banded or written at '
                    'breakpoints')
        response, reference = self.record_pair(record)
        # a cross spectrum's real part is not a density of one signal,
        # and its imaginary part is the phase between two; one DOF named
        # twice over two quantities is a cross term too
        # (`is_autospectrum`)
        if (not self.is_autospectrum(record)
                or has_phase(self.ordinate[record])):
            return (f'record {record} ({response}/{reference}) is a cross '
                    'spectrum; a tone is read off an autospectrum')
        try:
            line_spacing(self.abscissa)
        except ValueError as refusal:
            return str(refusal)
        # five: the rectangle's lobe of a line either side and a line of
        # floor beyond each, the fewest any window reads a tone from
        if len(self.abscissa) < 5:
            return f'{len(self.abscissa)} lines are too few to read a tone'
        return None

    def can_read_tone(self, record: int = 0) -> bool:
        """Whether a tone can be read off record `record`
        (`tone_refusal` says why not).

        Parameters
        ----------
        record : int, default 0
            Which record.

        Returns
        -------
        bool
        """
        return self.tone_refusal(record) is None

    def read_tone(self, frequency: float, record: int = 0, *,
                  window: Any = 'hann', parameter: float | None = None,
                  snap: bool = True, warn: bool = True) -> ToneReading:
        """Read the tone near `frequency` off one record of this
        autospectrum, as an amplitude beside the spike it stands at.

        The lines the tone leaks into, less the floor beside them, times
        the line spacing, are its mean square whatever the spacing and
        window; `core.tone_readout.read_tone` does the reading and says
        how, and `visualdynamics.plot.plot_tone_readout` draws it.

        Parameters
        ----------
        frequency : float
            A frequency at or near the tone, in Hz.
        record : int, default 0
            Which record.
        window : str or Averaging, default 'hann'
            The window this spectrum was averaged with, or the
            `Averaging` itself; a project gives the one a computed PSD
            was made with as `Project.averaging_of`.
        parameter : float, optional
            The Tukey window's alpha or the Kaiser window's beta, when
            `window` is a name; an `Averaging` carries its own.
        snap : bool, default True
            Find the tone's own spike near `frequency`; False reads the
            lines at `frequency` exactly, as a tone table states it.
        warn : bool, default True
            Warn a tone between lines, an uneven floor or no tone as a
            `ToneWarning`; the notes are on the reading either way.

        Returns
        -------
        ToneReading
            The tone's amplitude in the square root of this spectrum's
            units (SI, as the spectrum is stored), the spike, the floor
            and the notes, whose densities are in SI too
            (`ToneReading.notes_in` writes them in others).

        Raises
        ------
        ValueError
            For a record `tone_refusal` refuses: a spectrum on uneven
            lines (octave bands) or written at breakpoints, which has no
            line spacing to undo, or a cross spectrum.
        """
        from .tone_readout import read_tone

        refusal = self.tone_refusal(record)
        if refusal is not None:
            raise ValueError(refusal)
        dimension = str(self.ordinate_dim[record])
        unit = ('' if dimension == UNKNOWN
                else SI.label_text(channel_quantities(dimension)[0]))
        return read_tone(self.abscissa, self.ordinate[record], frequency,
                         window=window, parameter=parameter, snap=snap,
                         unit=unit, warn=warn)

    def rms_check(self, history: TimeHistory, record: int = 0, *,
                  averaging: Averaging | None = None,
                  threshold_db: float | None = None) -> RmsCheck:
        """Check one record of this autospectrum against the time history
        it came from: the rms from the area under it beside the record's
        own rms, and the gap between them in dB.

        The area under a density is the mean square of its record, so a
        correctly scaled spectrum gives back the record's rms;
        `core.rms_check.check_record` does the check and says how, and
        `visualdynamics.plot.plot_rms_check` draws it.

        Parameters
        ----------
        history : TimeHistory
            The record this spectrum was computed from; the channel with
            this record's DOF and quantity is the one read, every
            capture of it pooled.
        record : int, default 0
            Which record of this spectrum.
        averaging : Averaging, optional
            How the spectrum was averaged, which sets the span the
            record's rms is read over and the frames' own mean square;
            `history`'s own when omitted. A project gives a computed
            spectrum's as `Project.averaging_of`.
        threshold_db : float, optional
            The gap, in dB either way, beyond which the two disagree;
            `core.rms_check.GAP_DB` when omitted.

        Returns
        -------
        RmsCheck
            Both rms in the record's unit (SI, as stored), the gap, the
            frames' own rms when the averaging is known, and the
            verdict.

        Raises
        ------
        ValueError
            For a banded spectrum, one written at breakpoints or a cross
            spectrum, when `history` holds no such channel, or when the
            frames do not fit in it.
        """
        from .rms_check import GAP_DB, check_record

        return check_record(self, history, record, averaging=averaging,
                            threshold_db=(GAP_DB if threshold_db is None
                                          else threshold_db))

    def area(self, record: int = 0, low: float | None = None,
             high: float | None = None) -> float:
        """The area under one record, over a band or over all of it.

        The record is integrated the way it is drawn, as set by
        `interpolation`: the sum of value times bin width for 'bin', the
        exact power-law area for 'log_log'. The area is in the ordinate's
        units times hertz, so for a PSD its square root is the RMS.

        Parameters
        ----------
        record : int, default 0
            Which record.
        low, high : float, optional
            The band to integrate over.

        Returns
        -------
        float
            The area under the curve (the mean square, whose root is
            the RMS); NaN when no line is written.
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
        """`area` over several bands at once, in one pass over the lines.

        Parameters
        ----------
        record : int
            Which record.
        lows, highs : array-like
            The bands' edges, paired.

        Returns
        -------
        numpy.ndarray
            One area per band; 0 for a band outside the written lines
            (for a breakpoint curve with fewer than two usable points,
            NaN).
        """
        # Vectorized because the comparison is judged cell by cell, and a
        # thousand cells must not cost a thousand passes over the lines.
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
        """Which lines hold a value: the finite ones, in one record, or in
        any record when none is named. (`Specification.written` also
        requires a positive value.)

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
        """(low, high), in hertz, of the range this spectrum is written
        over.

        For a density per bin, the outer edges of the written bins, since
        a line covers its whole bin. For a curve between breakpoints, the
        first and last written points. Octave banding and comparisons use
        this range, because a controller's target is stored on every FFT
        line to Nyquist but is only written over part of them.

        Parameters
        ----------
        record : int, optional
            The record; the union over every record when omitted.

        Returns
        -------
        tuple of float, or None
            None when nothing is written above zero hertz.
        """
        # What octave banding clips its end bands to, and what a
        # comparison is judged over (Brandon, 2026-09-19).
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

    function_type = 9
    abscissa_dim = 'frequency'
    complex_ordinate = True

    def __init__(self, *args: Any, bandwidth: ArrayLike | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not has_phase(self.ordinate):
            self.ordinate: np.ndarray = np.ascontiguousarray(self.ordinate.real)
        self._take_bandwidth(bandwidth)

    def as_specification(self) -> Specification:
        """This PSD as a specification: the same values, records, units and
        bands, as the requirement a measurement is judged against, with no
        limits.

        Some formats have no way to mark a PSD as a requirement (ESCDF's
        data types do not), so a target written by another program
        arrives as a PSD, and the user marks it with this. A banded PSD is
        read per bin, as a banded specification is; otherwise its
        frequencies decide whether it is read as a controller's lines or
        as breakpoints (`Specification.reading_of`). Any `scale_db` is
        dropped, since a requirement is not scaled for comparison.

        Returns
        -------
        Specification
            A copy; this PSD is unchanged.
        """
        # The person who knows says so (Brandon, 2026-10-09).
        spec = _recast(self, Specification)
        spec.limits = {}
        # a requirement is not scaled to be compared: the test level
        # belongs to the measurement
        spec.__dict__.pop('scale_db', None)
        spec.interpolation = ('bin' if self.bandwidth is not None
                              else Specification.reading_of(self.abscissa))
        return spec

    def to_octave(self, per_octave: int | None = None,
                  low: float | None = None,
                  high: float | None = None) -> Psd:
        """This spectrum integrated onto proportional (octave) bands.

        Each band takes the mean-square content that falls in it, divided
        by its own width, so the area under the spectrum, and so the RMS,
        is unchanged. Reading the narrowband curve at each band center
        would lose everything between the centers.

        The band grid is fixed (see `visualdynamics.core.octave`): the
        frequency range only chooses which bands of that grid are
        returned, so two runs banded the same way are on the same bands
        whatever their ranges. Bands are whole, so a band the data only
        partly covers holds that content averaged over its whole width.
        Cross terms stay complex, so a CPSD can be banded as well as a
        PSD. `scale_db` is kept.

        Parameters
        ----------
        per_octave : int, optional
            Bands per octave; `octave.PER_OCTAVE` when omitted.
        low, high : float, optional
            The frequency range to cover, in hertz. By default, the
            range the written bins cover (`extent`).

        Returns
        -------
        Psd
            The same power on proportional bands, with `bandwidth` set.

        Raises
        ------
        ValueError
            If the spectrum has fewer than two lines above zero hertz.
        """
        from .octave import resample

        frequencies = np.asarray(self.abscissa, dtype=float)
        per_octave, centers, widths, bounds = self._octave_grid(
            per_octave, low, high)
        banded = resample(frequencies, self.ordinate, bounds, widths,
                          source=self.bin_bounds())
        out = Psd(centers, banded,
                  **{**self.record_fields(),
                     'comment': f'1/{per_octave} octave bands'},
                  bandwidth=widths)
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

    def coherence(self) -> Coherence:
        """The ordinary coherence of every cross term this CPSD holds:
        |S_pq|² / (S_pp S_qq), from the cross spectrum and the two
        autospectra.

        For an octave-banded CPSD the result is the banded quantity (each
        band's averaged cross spectrum against its averaged autospectra),
        and it keeps the bands, so it is drawn flat across each band as
        the CPSD is. That is not the average of a narrowband coherence
        over the band, which would weight a band's lines equally whatever
        their power. Values are limited to 0 to 1. A cross term without
        both of its autospectra is left out.

        Returns
        -------
        Coherence
            One record per cross term, response against reference.

        Raises
        ------
        ValueError
            If the object has no reference DOFs, or no cross term has
            both its autospectra.
        """
        # Banded coherence added 2026-09-26, for the band-average paper.
        if self.reference_dof is None:
            raise ValueError('coherence needs a CPSD: records with a '
                             'reference as well as a response')
        autos = {}
        for i, (p, q) in enumerate(zip(self.response_dof, self.reference_dof,
                                       strict=True)):
            if p == q:
                autos.setdefault(str(p), i)
        rows, responses, references = [], [], []
        ordinate = np.asarray(self.ordinate)
        for i, (p, q) in enumerate(zip(self.response_dof, self.reference_dof,
                                       strict=True)):
            if p == q:
                continue
            if str(p) not in autos or str(q) not in autos:
                continue
            # an undefined line reads zero in a Coherence record, as it
            # always has here
            value = ordinary_coherence(ordinate[autos[str(p)]],
                                       ordinate[autos[str(q)]], ordinate[i])
            rows.append(np.nan_to_num(value, nan=0.0))
            responses.append(p)
            references.append(q)
        if not rows:
            raise ValueError('no cross term has both its autospectra here')
        return Coherence(np.asarray(self.abscissa), np.asarray(rows),
                         response_dof=responses, reference_dof=references,
                         bandwidth=self.bandwidth,
                         comment=(f'coherence of {self.comment}'.strip()
                                  if self.comment else 'coherence'))

    def conditioning(self, records: Sequence[int] | None = None, *,
                     limit: float | None = None,
                     progress: Callable[[int, int], None] | None = None
                     ) -> Conditioning:
        """The singular values and condition number of this cross-spectral
        matrix at every line (`core.matrix_checks.read_conditioning`).

        Read of the drives' CSD, `Gff`, it says where the drives are
        nearly copies of one another, so that an FRF estimated from them
        inverts a matrix that is nearly singular.

        Parameters
        ----------
        records : sequence of int, optional
            The records that make the matrix; all of them when omitted.
            The channels must be of one quantity.
        limit : float, optional
            The condition number lines are flagged over;
            `core.matrix_checks.CONDITION_LIMIT` when omitted.
        progress : callable, optional
            Told ``(0, core.progress.BUSY)`` before the decomposition;
            the app's progress bar uses it.

        Returns
        -------
        Conditioning
            The singular values at every line, with the condition
            number, the effective rank and the flagged bands.

        Raises
        ------
        ValueError
            For an empty selection, no cross terms, channels of
            different quantities, or an incomplete matrix.
        """
        from .matrix_checks import CONDITION_LIMIT, read_conditioning

        return read_conditioning(
            self, records, limit=CONDITION_LIMIT if limit is None else limit,
            progress=progress)

    def realizability(self, records: Sequence[int] | None = None, *,
                      tolerance: float | None = None,
                      keep_diagonal: bool = True,
                      progress: Callable[[int, int], None] | None = None
                      ) -> Realizability:
        """Whether this cross-spectral matrix is Hermitian and positive
        semi-definite at every line, so that some set of signals has it,
        and the nearest matrix that is
        (`core.matrix_checks.read_realizability`).

        A specification that is not cannot be played: no drive gives
        the control channels those coherences and phases together.

        Parameters
        ----------
        records : sequence of int, optional
            The records that make the matrix; all of them when omitted.
        tolerance : float, optional
            How far below zero the smallest eigenvalue of the coherence
            matrix may lie, as a fraction of the largest;
            `core.matrix_checks.REALIZABLE_TOLERANCE` when omitted.
        keep_diagonal : bool, default True
            Whether the repair holds the autospectra where they are.
        progress : callable, optional
            Told ``(done, total)`` as the repair goes on; what it raises
            stops it. The app's progress bar uses it.

        Returns
        -------
        Realizability
            The eigenvalues at every line, the verdict, the failing
            bands, and the repaired matrix.

        Raises
        ------
        ValueError
            For an empty selection, no cross terms, records some declared
            in units and some not, one channel, or an incomplete matrix.
        """
        from .matrix_checks import REALIZABLE_TOLERANCE, read_realizability

        return read_realizability(
            self, records, keep_diagonal=keep_diagonal,
            tolerance=REALIZABLE_TOLERANCE if tolerance is None
            else tolerance, progress=progress)

    def made_realizable(self, records: Sequence[int] | None = None, *,
                        tolerance: float | None = None,
                        keep_diagonal: bool = True,
                        progress: Callable[[int, int], None] | None = None
                        ) -> Psd:
        """A copy with the cross-spectral matrix repaired to the nearest
        realizable one at every line that is not
        (`core.matrix_checks.made_realizable`); the lines that are, and
        every record's place and units, are unchanged.

        Parameters
        ----------
        records : sequence of int, optional
            The records that make the matrix; all of them when omitted.
        tolerance : float, optional
            As for `realizability`.
        keep_diagonal : bool, default True
            Hold the autospectra where they are and move only the cross
            terms. Without it a specification's limits follow each
            autospectrum that moved.
        progress : callable, optional
            As for `realizability`.

        Returns
        -------
        Psd
            The repaired copy, of this object's class. `realizability`
            on this object says what moved.
        """
        from .matrix_checks import REALIZABLE_TOLERANCE, made_realizable

        repaired, _reading = made_realizable(
            self, records, keep_diagonal=keep_diagonal,
            tolerance=REALIZABLE_TOLERANCE if tolerance is None
            else tolerance, progress=progress)
        return repaired

    @classmethod
    def units_for(cls, dimension, system=SI):
        """The (unit, reference unit) a unit system gives for a PSD
        dimension: 'acceleration**2/frequency' in SI is ('m/s**2', None),
        and a cross term's 'acceleration*force/frequency' is ('m/s**2',
        'N'). See `DataArray.units_for`.

        Parameters
        ----------
        dimension : str
            An `ordinate_dim`, such as 'acceleration**2/frequency'.
        system : UnitSystem, optional
            The unit system; SI when omitted.

        Returns
        -------
        tuple of (str, str or None)
            The engineering unit and, for a cross term with two
            quantities, the second unit.
        """
        numerator = dimension.split('/')[0]
        if numerator.endswith('**2'):
            return system.unit(numerator[:-3]), None
        base, _, cross = numerator.partition('*')
        return system.unit(base), (system.unit(cross) if cross else None)

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
    """A shock response spectrum: the peak response of an oscillator at
    each natural frequency.

    It is not a spectrum of the shock. Every point is the largest
    response that a single-degree-of-freedom oscillator of that natural
    frequency reached while its base was driven by the measured
    transient: one number from the whole response of one filter. Two
    shocks with the same SRS can look very different, and an SRS cannot
    be turned back into a time history, because the phase that produced
    each peak is not kept.

    The abscissa is the oscillator's natural frequency, not a frequency
    in the shock, and is drawn on a log axis (`log_abscissa`): an SRS is
    specified at octave-spaced natural frequencies, and on a linear axis
    the lowest five octaves would be squeezed against the left edge. The
    ordinate is in the quantity the base was measured in, so an
    acceleration transient gives an acceleration SRS.

    `q` and `kind` change what the curve means, so they are stored on
    the object: an SRS at Q = 10 and the same shock at Q = 50 are
    different curves, and a maximax SRS is not a positive one. See
    `visualdynamics.core.srs`.

    Parameters
    ----------
    *args, **kwargs
        As for `DataArray`.
    q : float, optional
        The amplification factor; `srs.DEFAULT_Q` when omitted.
    kind : str, default 'maximax'
        Which peak each point is: 'maximax', 'positive' or 'negative'.
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
        """The damping ratio for the amplification factor, 1/(2Q)."""
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
    """Limit curves stored with an ordinate, in the same units.

    A test is controlled to a target and the tolerance around it: how
    far the response may go before the controller warns, and how far
    before it stops. A random vibration specification and a shock
    specification both have the same four curves, so they are defined
    once here and mixed into both classes.

    The limits are arrays stored with the ordinate, not four more data
    objects, so they share the abscissa, the DOFs, the dimensions and
    the units, and every unit conversion of the ordinate is applied to
    them too. `limit()` returns one as an object of the underlying kind,
    to plot or export.

    Limits are per control channel, so a record that is not a control
    channel has none: its rows are NaN, which plots as a gap and not as
    a line at zero.

    Parameters
    ----------
    *args, **kwargs
        As for the underlying class (`Psd` or `Srs`).
    warning_lower, warning_upper, abort_lower, abort_upper : array-like, optional
        The limit curves, each the shape of the ordinate.

    Raises
    ------
    ValueError
        If a limit's shape differs from the ordinate's.

    Attributes
    ----------
    limits : dict of str to numpy.ndarray
        The limits given, by name (`LIMITS`). Each is in SI like the
        ordinate, NaN where a record is not a control channel or no
        limit was written: `warning_lower` and `warning_upper` are where
        the controller warns, below and above the target, and
        `abort_lower` and `abort_upper` where it stops the test.
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
        """Whether any limit curve is stored."""
        return bool(self.limits)

    def limit_area(self, name: str, record: int = 0,
                   low: float | None = None,
                   high: float | None = None) -> float:
        """The area under one limit curve, integrated the same way as the
        specification (its `interpolation`).

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
        """`limit_area` over several bands at once; see `areas`.

        Parameters
        ----------
        name : str
            One of `LIMITS`.
        record : int
            Which record.
        lows, highs : array-like
            The bands' edges, paired.

        Returns
        -------
        numpy.ndarray
            One area per band; all NaN when the limit was never written.
        """
        values = self.limits.get(name)
        if values is None:
            return np.full(np.shape(np.atleast_1d(lows)), np.nan)
        return self._areas_of(np.atleast_2d(values)[record], lows, highs)

    def limit(self, name: str) -> DataArray | None:
        """One limit as an object of the underlying kind (`PLAIN`), or None
        if that limit is not stored.

        The object has the specification's records and units, and its
        `interpolation` where it has one.

        Parameters
        ----------
        name : str
            One of `LIMITS`.

        Returns
        -------
        DataArray or None
            The limit curve.
        """
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
        """A limit in display units; records with undefined units are
        returned unchanged.

        Parameters
        ----------
        name : str
            One of `LIMITS`.
        unit_system : UnitSystem
            The units to convert to.

        Returns
        -------
        numpy.ndarray or None
            The limit's values, or None if that limit is not stored.
        """
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
    """What a random vibration test was controlled to: a PSD and its
    limits.

    A specification is a PSD in every respect (the same abscissa,
    quantity and unit conversions) with the four limit curves of
    `Bounded`, and it inherits from both. By default its points are read
    as breakpoints of a power law (`interpolation` 'log_log'); one
    computed from a record, or a target on a controller's lines, is read
    per bin.
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
        """How a specification written at these frequencies is read.

        'bin', a density per line, for `LINES_AT_LEAST` or more lines on
        an even grid (a controller's target on its FFT lines), and
        'log_log', breakpoints of a power law, otherwise. Octave-band and
        narrowband specifications are drawn as steps; only a breakpoint
        curve is drawn as the power law between its points.

        Parameters
        ----------
        frequencies : array_like
            The specification's frequency lines, ascending.

        Returns
        -------
        str
            'bin' or 'log_log'.
        """
        # Brandon, 2026-09-19: "breakpoint specifications have very few
        # frequency lines and I would expect an uneven spacing of them".
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
        """Which lines the requirement is written on: finite and positive,
        in one record, or in any record when none is named. A controller
        writes zero on the lines it did not control, and those lines are
        not a requirement of zero.

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

    def as_psd(self) -> Psd:
        """This specification as a plain PSD; the inverse of
        `Psd.as_specification`, for a PSD marked as a requirement by
        mistake. Its `interpolation` is kept.

        Returns
        -------
        Psd
            A copy; this specification is unchanged.

        Raises
        ------
        ValueError
            If it has limits: a PSD cannot hold them, and dropping them
            without being asked would lose what the file said.
            `limit(name)` returns one as a PSD of its own.
        """
        if self.limits:
            raise ValueError(
                'it carries warning or abort limits, which a PSD cannot '
                'hold; take one out with limit(name) instead')
        plain = _recast(self, Psd)
        plain.__dict__.pop('limits', None)
        return plain

    def to_octave(self, per_octave: int | None = None,
                  low: float | None = None,
                  high: float | None = None) -> Specification:
        """This specification integrated onto proportional bands, with its
        warning and abort limits.

        The same area rule as `Psd.to_octave`, applied the way the object
        is read. A specification on lines is a density, and each line is
        a bin. One written at breakpoints is a power law between them,
        and each band takes the exact area under that law
        (`compliance.log_log_area`) over the part of the band the
        specification covers, divided by the band's width. The limits go
        through the same rule, so a band's warning line has the same
        relation to its target as at the breakpoints. Cross terms of a
        breakpoint specification are not power laws (the phase is not),
        so they are read onto a fine log grid as the authoring sheet
        reads them (magnitude log-log, phase linear) and integrated
        there.

        The result is a density per band (`interpolation` 'bin', with the
        bands' widths), which a banded measurement is compared against
        band by band. `scale_db` and any band constraints are kept.

        Parameters
        ----------
        per_octave : int, optional
            Bands per octave.
        low, high : float, optional
            The band to cover.

        Returns
        -------
        Specification
            The same power and the same limits on proportional bands.

        Raises
        ------
        ValueError
            If the specification is not written over a range of
            frequencies above zero.
        """
        # Brandon, 2026-09-18: the specification and the measurement it
        # judges should be convertible alike, so a compliance can be read
        # on bands at both ends. This reverses the earlier reasoning
        # (PLAN.md, "Octave bands") that a written curve needed no banding
        # because the comparison integrates it exactly; it does, and a
        # banded specification is still the object a person asks to see
        # and pass on.
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
            **{**self.record_fields(),
               'comment': f'1/{per_octave} octave bands'},
            bandwidth=widths, **limits)
        out.interpolation = 'bin'
        out.scale_db = self.scale_db
        constraints = getattr(self, 'band_constraints', None)
        if constraints:
            out.band_constraints = dict(constraints)
        return out


class ShockSpecification(Bounded, Srs):
    """What a shock test was controlled to: an SRS and its limits.

    It is to `Srs` what `Specification` is to `Psd`. A shock target is
    written as a required SRS with a tolerance on each side, usually
    +6 dB and -3 dB (a factor of 2 up and 0.707 down), and the limits are
    those curves.

    A target is written at one Q and read at that Q: a target at Q = 10
    says nothing about what the same shock does to a Q = 50 oscillator,
    so the Q is stored with the target as it is with a measurement.
    """

    PLAIN = Srs

    @classmethod
    def from_breakpoints(cls, frequencies: ArrayLike, levels: ArrayLike,
                         response_dof: str | Sequence[str], *,
                         per_octave: int | None = None,
                         q: float | None = None, kind: str = 'maximax',
                         ordinate_dim: str = 'acceleration',
                         ordinate_unit: str | None = None,
                         warning: tuple[float, float] | None = None,
                         abort: tuple[float, float] | None = None
                         ) -> ShockSpecification:
        """A requirement written as breakpoints, read onto the points of
        its spacing, with its bands as limits.

            spec = ShockSpecification.from_breakpoints(
                [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '101Z+',
                ordinate_unit='g', warning=(-1.5, 3.0), abort=(-3.0, 6.0))

        A shock requirement is usually written as a few breakpoints
        joined by straight lines on log axes (a rising slope in dB per
        octave, a knee and a plateau), with its Q, its spacing and its
        type stated beside it. The points are the spacing's, from the
        first breakpoint up, `per_octave` to an octave, the last one at
        or below the last breakpoint, so every point lies on the curve
        the breakpoints draw (`compliance.log_interpolate`). That
        spacing is the one a tolerance is counted at, so a measured
        spectrum computed at the same `per_octave` from the same lowest
        frequency lands on the requirement's points. Its last point can
        fall past the last breakpoint, where nothing is required, and is
        not judged.

        Each band, given as dB below and above the requirement, becomes
        a pair of limit curves: `warning` the warning limits and `abort`
        the abort limits. A tolerance written as an inner and an outer
        band is the two of them, inner as the warning, outer as abort.

        Parameters
        ----------
        frequencies : array-like
            The breakpoints' frequencies in Hz, increasing.
        levels : array-like
            The level at each breakpoint, one row per channel, in
            `ordinate_unit`. A single row (which may be one-dimensional)
            is the requirement at every channel named.
        response_dof : str or sequence of str
            The DOF each row is required at.
        per_octave : int, optional
            Points to an octave; `srs.PER_OCTAVE` when omitted.
        q : float, optional
            The amplification the requirement is written at;
            `srs.DEFAULT_Q` when omitted.
        kind : str, default 'maximax'
            The spectrum's type: 'maximax', 'positive' or 'negative'.
        ordinate_dim : str, default 'acceleration'
            The quantity the levels are of.
        ordinate_unit : str, optional
            The unit the levels are written in; SI when omitted.
        warning, abort : (float, float), optional
            A band in dB below and above the requirement, kept as that
            pair of limits.

        Returns
        -------
        ShockSpecification
            The requirement on its points, with the limits given.

        Raises
        ------
        ValueError
            For fewer than two breakpoints, frequencies that are not
            positive and increasing, levels that are not positive, a
            number of level rows other than one or one per DOF, a
            spacing that is not a whole number of points to the octave,
            or a band whose lower edge is not below its upper.
        """
        from .compliance import _band_db, log_interpolate
        from .srs import DEFAULT_Q, PER_OCTAVE

        corners = np.asarray(frequencies, dtype=float)
        rows = np.atleast_2d(np.asarray(levels, dtype=float))
        if corners.ndim != 1 or corners.size < 2:
            raise ValueError('a requirement needs at least two breakpoints')
        if not (np.all(corners > 0.0) and np.all(np.diff(corners) > 0.0)):
            raise ValueError('the breakpoints\' frequencies must be positive '
                             'and increasing')
        if rows.shape[1] != corners.size:
            raise ValueError(f'{rows.shape[1]} levels for {corners.size} '
                             'breakpoints')
        if not np.all(rows > 0.0):
            raise ValueError('a required level must be positive')
        if ordinate_unit is not None:
            scale, offset = si_transform(ordinate_unit, ordinate_dim)
            rows = rows * scale + offset
        # refused rather than truncated (2026-10-16): int(6.5) is 6,
        # and the requirement would be read onto a spacing nobody asked
        # for without a word
        step = PER_OCTAVE if per_octave is None else per_octave
        if (isinstance(step, bool) or float(step) != int(step)
                or int(step) < 1):
            raise ValueError(f'{per_octave!r} is not a whole number of '
                             'points to the octave')
        step = int(step)
        # whole steps only, so the last point stays on the curve: the
        # nearest whole number of steps, as `octave_frequencies` takes,
        # can land past the last breakpoint, where nothing is written
        steps = int(np.floor(np.log2(corners[-1] / corners[0]) * step + 1e-9))
        points = corners[0] * 2.0 ** (np.arange(steps + 1) / float(step))
        # a band a whole number of octaves wide ends on its breakpoint,
        # and 2**k can round a hair past it, where the curve is not
        # written and the last point would read NaN
        points = np.minimum(points, corners[-1])
        ordinate = np.vstack([log_interpolate(points, corners, row)
                              for row in rows])
        dofs = ([response_dof] if isinstance(response_dof, str)
                else list(response_dof))
        if ordinate.shape[0] not in (1, len(dofs)):
            # two rows at one DOF used to make two records at that DOF,
            # and every reading takes the first a DOF names, so the
            # second was never read (2026-10-16)
            raise ValueError(f'{ordinate.shape[0]} rows of levels for '
                             f'{len(dofs)} DOF{"s" * (len(dofs) != 1)}; '
                             'give one row, or one per DOF')
        if ordinate.shape[0] == 1 and len(dofs) > 1:
            # one requirement at several channels, as a shock is often
            # required alike on every axis
            ordinate = np.repeat(ordinate, len(dofs), axis=0)
        count = ordinate.shape[0]
        limits = {}
        for name, band in (('warning', warning), ('abort', abort)):
            if band is None:
                continue
            below, above = _band_db(name, band)
            limits[f'{name}_lower'] = ordinate * 10.0 ** (below / 20.0)
            limits[f'{name}_upper'] = ordinate * 10.0 ** (above / 20.0)
        q = DEFAULT_Q if q is None else float(q)
        return cls(points, ordinate, response_dof=dofs,
                   ordinate_dim=[ordinate_dim] * count,
                   ordinate_unit=(None if ordinate_unit is None
                                  else [ordinate_unit] * count),
                   comment=f'{kind} SRS at Q={q:g}, {step} to the octave',
                   q=q, kind=kind, **limits)


class TransientSpecification(TimeHistory):
    """What a transient test was controlled to: a target time history.

    This is different from a `ShockSpecification`, although the two are
    often called by each other's names. A shock specification is an SRS,
    a required response spectrum, and a controller meets it with any
    transient whose SRS falls inside the limits. A transient
    specification is a waveform, this acceleration sample by sample, and
    the controller inverts the structure's transfer function to
    reproduce it.

    It is a time history that is a target, and it is compared against
    another time history (what the test article did), not against
    limits. It has no limits, since there is no settled convention for a
    tolerance on a waveform as there is for one on a spectrum.

    Its computed spectra are targets too. Its PSD is a `Specification`
    and its SRS a `ShockSpecification`, both without limits, because the
    spectrum of a waveform the test article was required to see is the
    spectrum it was required to see. As plain objects they could not be
    told from the response's own spectra except by name, and the
    comparison would have to be set up by hand instead of by type.
    """
    # Rattlesnake can run a transient specification; it cannot run a
    # shock (SRS) specification.

    def psd_type(self) -> type[Psd]:
        """The class of a PSD of this record: `Specification`, since the
        spectrum of a required waveform is itself a requirement.

        Returns
        -------
        type
            `Specification`.
        """
        return Specification

    def srs_type(self) -> type[Srs]:
        """The class of an SRS of this record: `ShockSpecification`, for
        the same reason `psd_type` gives.

        Returns
        -------
        type
            `ShockSpecification`.
        """
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
    """The data class for a UFF dataset 58 function type code.

    Parameters
    ----------
    code : int
        The function type: 1, 4, 6, 9, 12, 24 or 26.

    Returns
    -------
    type
        The `DataArray` subclass, such as `Frf` for 4.

    Raises
    ------
    ValueError
        If the code is not one of those.
    """
    try:
        return DATA_CLASSES[int(code)]
    except KeyError:
        raise ValueError(
            f'Unsupported function type {code}; '
            f'supported: {sorted(DATA_CLASSES)}')
