"""Field data to specification statistics: a set of measured spectra
turned into the limit a specification is written from, and the
straight-line envelope written over that limit.

A random or shock specification derived from measurements starts as a
set of spectra: one PSD or SRS per flight, per run, per event or per
segment of a mission. Every one of them differs from the next, so the
specification is written over a statistical limit of the set, a level
at each frequency that the set's next member is not expected to
exceed, and the limit is then covered by a few straight lines on
log-log axes, which is the breakpoint table a laboratory receives.
This module makes both steps reproducible, and says on each result
what it rests on.

**The set.** `gather` reads the records onto common frequency lines:
the records of one PSD or SRS, or records from several, as long as
they can be read one way. Records of different quantities, or with
units declared on one side and not the other, are refused
(`compliance.units_check`), as are a PSD mixed with an SRS, two SRS at
different Q or of different kinds, a cross spectrum, and frequency
ranges that do not overlap. A NaN, or a line at zero or below, inside
the band read is refused with its record and frequency named, since a
level of zero has no logarithm and a dead channel is not a quiet
flight. Records on different lines are read onto the first record's
lines over the range they share, log-log between their own lines, and
the result says so.

**The limits.** The levels are taken in decibels, 10 log10 of a
density for a PSD and 20 log10 of a peak for an SRS, so a level spread
by a factor reads as a spread in decibels whatever its size. At each
line, over the n records,

- `envelope`: the largest of the records (the maximum envelope). It
  assumes nothing about how the levels are distributed. Whatever that
  distribution is, the largest of n independent records stands above a
  share `coverage` of the population with probability
  1 - coverage**n, which the result reports beside it.
- `distribution_free`: the same largest level, taken as a limit at a
  stated `coverage` and `confidence`, which it can be only when
  1 - coverage**n reaches the confidence. A set smaller than
  ln(1 - confidence) / ln(coverage) records is refused, with the count
  it needs.
- `mean_k`: the mean of the levels in decibels plus `k` standard
  deviations of them, the standard deviation estimated from the
  records with n - 1 in its denominator.
- `tolerance`: the normal tolerance limit, the mean plus a factor
  times the standard deviation, the factor chosen so that the limit
  stands above a share `coverage` of the population with probability
  `confidence`, on the assumption that the levels in decibels are
  normally distributed across the records (that the levels are
  log-normal). With the standard deviation estimated from n records the
  factor is

      k = t'(confidence; n - 1, z sqrt(n)) / sqrt(n)

  where z is the standard normal quantile at `coverage` and t' the
  quantile of the noncentral t distribution with n - 1 degrees of
  freedom and noncentrality z sqrt(n). That is exact: the sample mean
  is normal and the sample variance chi-square, and the event that the
  limit covers the population's quantile is the event that a
  noncentral t variable stays below k sqrt(n). Ten records at 95
  percent coverage and 95 percent confidence give k = 2.911, three give
  7.656, and at 50 percent confidence the factor falls toward z =
  1.645 as the set grows. Fewer than two records leave no standard
  deviation to estimate and are refused, unless one is stated.

A standard deviation can instead be stated, `sigma_db`, when the set
is too small to estimate one. It is then taken as known, and the
tolerance factor is z + z_c / sqrt(n), with z_c the normal quantile
at `confidence`: one record with 3 dB stated gives a limit 1.645 x 3 =
4.9 dB above it at 95 percent coverage and 50 percent confidence, and
(2.326 + 1.282) x 3 = 10.8 dB at 99 percent and 90 percent. The result
says the scatter was stated and not measured.

**The envelope written over the limit.** `fit_envelope` covers the
limit with straight lines on log-log axes, by stated rules. The lines
are at or above the limit at every line. The breakpoints are at lines
of the limit and their levels are free: for a given set of breakpoint
frequencies the levels are those of the cover that stands least above
the limit on average, in decibels over log frequency, which is a
linear program. The fit starts with one line across the band. While
the lines stand more than `max_overshoot_db` above the limit at some
line, a breakpoint is added in the span with the most area between its
line and the limit, at the line that span stands highest above, and
the levels are fitted again. It stops there, or at `max_breakpoints`,
in which case the allowance is not met and the result says by how much
and where. Once the allowance is met, any breakpoint whose removal
still meets it is removed. The area the envelope adds over the limit
(a PSD's mean square, read log-log between lines for both) is reported
in decibels; two straight runs that agree at neighboring lines agree
between them, so a cover at every line is a cover everywhere. A
narrowband limit is ragged from line to line, and lines that stay
within a few decibels of every valley need a breakpoint at almost
every other line; the limit of spectra banded first (`Psd.to_octave`)
is the one a few lines can follow.

A cover always raises the rms. Where the rms is to be kept instead
(`rms='conserve'`, a PSD only, since an SRS has no area), the covering
lines are lowered by the decibels they added, so the envelope's area
equals the limit's; the lines where it then falls below the limit are
counted and the deepest named.

The margin a document asks for is added when the specification is
written (`EnvelopeFit.specification`), in decibels on the same
convention, as its own step and shown as such.

References
----------
The normal distribution, the sampling distributions of a sample's mean
and variance and the Student t are standard results of statistics; the
tolerance factor follows from them, as above, and the limits are those
a test document names for summarizing a set of measured spectra.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The normal distribution,
   and the sampling distributions of a sample mean and variance
   (normal and chi-square) and of the Student t, on which the
   tolerance factor rests.
2. MIL-STD-810H (2019), *Environmental Engineering Considerations and
   Laboratory Tests*, Method 514.8 (Vibration), Annex F, Appendix B.
   The upper limits a set of measured spectra is summarized by: the
   normal tolerance limit, the envelope, the distribution-free
   tolerance limit and the empirical tolerance limit.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .compliance import log_interpolate, log_log_area, units_check
from .progress import Ticker

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import ShockSpecification, Specification

#: the limits a set can be summarized by (see the module's text)
STATISTICS = ('envelope', 'distribution_free', 'mean_k', 'tolerance')

#: the share of the population a limit is to stand above, unless told
#: otherwise
COVERAGE = 0.95

#: the probability that it does, unless told otherwise: 50 percent, the
#: estimate that is as likely to be high as low
CONFIDENCE = 0.50

#: how far the straight-line envelope may stand above the limit at any
#: line, in decibels. The gap a reader accepts between an envelope and
#: the limit under it is a judgment; this is a starting point, and the
#: result states the allowance it was fitted with.
OVERSHOOT_DB = 3.0

#: what the envelope does to the rms: 'exceed', the cover, which
#: always raises it, or 'conserve', the cover lowered until its area is
#: the limit's
RMS_RULES = ('exceed', 'conserve')

#: the most breakpoints the envelope is given before it stops meeting
#: the overshoot allowance: a written table is read and typed by a
#: person, and a narrowband limit can need a breakpoint at every other
#: line to stay within a few decibels
MAX_BREAKPOINTS = 12

#: a level read a hair above the allowance is within it: the
#: overshoot of a straight line through two points of the limit is
#: worked out in floating point
_SLACK_DB = 1e-9


def _count(n: int, word: str) -> str:
    """'1 record', '2 records': a count and its noun."""
    return f'{n} {word}' + ('s' if n != 1 else '')


def _checked_share(name: str, value: float) -> float:
    """`value` as a float, refused unless it is a share strictly between
    0 and 1."""
    value = float(value)
    if not (np.isfinite(value) and 0.0 < value < 1.0):
        raise ValueError(f'a {name} is a share strictly between 0 and 1, '
                         f'not {value:g}')
    return value


def _db_scale(kind: str) -> float:
    """Decibels per decade of level: 10 for a density, which is a
    square, and 20 for an SRS, which is a peak."""
    return 10.0 if kind == 'psd' else 20.0


@dataclass
class FieldSet:
    """Records read onto common frequency lines, ready to summarize.

    Attributes:
        frequencies: The lines, in Hz, increasing and positive.
        levels: One row per record, the level at each line in SI (a
            density for a PSD, a peak for an SRS), every value finite
            and positive.
        labels: How each record is named in a message: its object, its
            record number and its DOF.
        dofs: The DOF each record was measured at.
        kind: 'psd' or 'srs'.
        dim: The records' dimension, as `ordinate_dim` spells it.
        q: The SRS's amplification, or None for a PSD.
        srs_kind: The SRS's kind ('maximax', 'positive', 'negative'), or
            None for a PSD.
        notes: What was done to read the records one way: lines read
            from another grid, a 0 Hz line left out, raw units.
    """

    frequencies: np.ndarray
    levels: np.ndarray
    labels: list[str]
    dofs: list[str]
    kind: str
    dim: str
    q: float | None = None
    srs_kind: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """How many records the set holds."""
        return int(self.levels.shape[0])

    @property
    def db_scale(self) -> float:
        """10 for a PSD and 20 for an SRS: the decibels of a factor of
        ten in level."""
        return _db_scale(self.kind)

    @property
    def levels_db(self) -> np.ndarray:
        """The levels in decibels re one SI unit, on the record's own
        convention (`db_scale`)."""
        return self.db_scale * np.log10(self.levels)


def _members(spectra: Any, records: Sequence[int] | None
             ) -> list[tuple[Any, int]]:
    """(object, record) for every record the caller named."""
    from .data import DataArray

    if isinstance(spectra, DataArray):
        chosen = (range(len(spectra.response_dof)) if records is None
                  else records)
        return [(spectra, int(r)) for r in chosen]
    if records is not None:
        raise ValueError('records are chosen per object when a set spans '
                         'several; pass (object, record) pairs instead')
    out: list[tuple[Any, int]] = []
    for item in spectra:
        if isinstance(item, DataArray):
            out.extend((item, r) for r in range(len(item.response_dof)))
        else:
            data, record = item
            out.append((data, int(record)))
    return out


def _label(data: Any, record: int) -> str:
    return (f'{type(data).__name__} record {record + 1} '
            f'({data.response_dof[record] or "no DOF"})')


def gather(spectra: Any, records: Sequence[int] | None = None, *,
           band: tuple[float, float] | None = None) -> FieldSet:
    """Read a set of PSDs or SRSs onto common lines, refusing what
    cannot be read one way.

    Parameters
    ----------
    spectra : Psd, Srs, or sequence
        One PSD or SRS (its records are the set), or a sequence of
        them (every record of each), or a sequence of
        `(object, record)` pairs.
    records : sequence of int, optional
        With a single object, the records that make the set; every
        record when omitted.
    band : (float, float), optional
        The frequencies to read, in Hz; the range every record covers
        when omitted. A band reaching past that range is refused.

    Returns
    -------
    FieldSet
        The records' levels on the first record's lines inside the
        band, with notes on what was done to read them one way.

    Raises
    ------
    TypeError
        For an object that is not a plain PSD or SRS (a specification
        is a requirement, not a measurement).
    ValueError
        For an empty set; a PSD mixed with an SRS; SRSs at different
        Q or of different kinds; a cross spectrum; records of
        different quantities, or with
        units declared on one side only; ranges with no frequency in
        common, or a band outside the common range; fewer than two
        lines in the band; or a NaN, or a level at or below zero, at a
        line inside it, named by record and frequency.
    """
    from .data import Bounded, Psd, Srs, autospectrum

    members = _members(spectra, records)
    if not members:
        raise ValueError('no records chosen: a limit is taken over a set')
    kinds = set()
    for data, record in members:
        if isinstance(data, Bounded):
            raise TypeError(
                f'{_label(data, record)} is a specification, a requirement '
                'already; a limit is taken over measured spectra')
        if isinstance(data, Psd):
            kinds.add('psd')
        elif isinstance(data, Srs):
            kinds.add('srs')
        else:
            raise TypeError(
                f'{_label(data, record)} is not a PSD or an SRS; a field '
                'limit is taken over autospectra or shock response spectra')
        if not 0 <= record < len(data.response_dof):
            raise ValueError(f'{type(data).__name__} has '
                             f'{len(data.response_dof)} records; no record '
                             f'{record + 1}')
    if len(kinds) > 1:
        raise ValueError('the set mixes PSDs and SRSs, which are different '
                         'quantities; take a limit over one kind')
    kind = kinds.pop()
    first, first_record = members[0]
    notes: list[str] = []
    q = srs_kind = None
    if kind == 'srs':
        q, srs_kind = float(first.q), str(first.kind)
        for data, record in members[1:]:
            if float(data.q) != q or str(data.kind) != srs_kind:
                raise ValueError(
                    f'{_label(data, record)} is a {data.kind} SRS at '
                    f'Q = {data.q:g} and {_label(first, first_record)} a '
                    f'{srs_kind} SRS at Q = {q:g}; spectra at different Q '
                    'or of different kinds are different curves')
    for data, record in members:
        # `autospectrum` is the one rule: one DOF and one quantity on
        # both sides, so a drive point's acceleration against its force
        # is a cross spectrum though it names one DOF twice
        if kind == 'psd' and not autospectrum(data, record):
            raise ValueError(
                f'{_label(data, record)} is a cross spectrum; a limit '
                'is taken over autospectra, which have a level')
    dim = str(first.ordinate_dim[first_record])
    for data, record in members[1:]:
        refused, _note = units_check(
            dim, str(data.ordinate_dim[record]),
            wanted_name=_label(first, first_record),
            held_name=_label(data, record))
        if refused:
            raise ValueError(refused)
    _refused, raw = units_check(dim, dim)
    if raw:
        notes.append('the units are not declared: the levels are taken as '
                     'the files gave them, in one unit')

    # the range every record covers, on its own positive lines
    spans = []
    for data, record in members:
        f = np.asarray(data.abscissa, dtype=float)
        positive = f[np.isfinite(f) & (f > 0.0)]
        if positive.size < 2:
            raise ValueError(f'{_label(data, record)} has fewer than two '
                             'lines above 0 Hz')
        spans.append((float(positive.min()), float(positive.max())))
    low = max(s[0] for s in spans)
    high = min(s[1] for s in spans)
    if not low < high:
        listed = '; '.join(f'{_label(d, r)} {a:g} to {b:g} Hz'
                           for (d, r), (a, b) in zip(members, spans))
        raise ValueError(f'the records share no frequency range: {listed}')
    if band is not None:
        lo, hi = float(band[0]), float(band[1])
        if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
            raise ValueError('a band is two positive frequencies, the lower '
                             'first')
        outside = []
        if lo < low * (1 - 1e-9):
            outside.append(f'{lo:g} to {low:g} Hz')
        if hi > high * (1 + 1e-9):
            outside.append(f'{high:g} to {hi:g} Hz')
        if outside:
            raise ValueError(
                f'the band reaches past the range every record covers '
                f'({low:g} to {high:g} Hz): no data from '
                + ' or '.join(outside))
        low, high = max(lo, low), min(hi, high)

    f0 = np.asarray(first.abscissa, dtype=float)
    if np.any(f0 == 0.0):
        notes.append('the 0 Hz line is left out: a level on log axes needs '
                     'a positive frequency')
    # low is a positive line or a band's edge, so these exclude 0 Hz
    keep = (f0 >= low) & (f0 <= high)
    lines = f0[keep]
    order = np.argsort(lines)
    lines = lines[order]
    if lines.size < 2:
        raise ValueError(f'fewer than two lines of '
                         f'{_label(first, first_record)} fall between '
                         f'{low:g} and {high:g} Hz')

    rows = []
    regridded = False
    for data, record in members:
        f = np.asarray(data.abscissa, dtype=float)
        own = np.real(np.asarray(data.ordinate[record]))
        inside = (f >= low) & (f <= high)
        # a fault is named at the record's own line, before anything is
        # read onto another grid, so the frequency in the message is one
        # the reader can find in the record
        for at, value in zip(f[inside], own[inside]):
            if not np.isfinite(value):
                raise ValueError(f'{_label(data, record)} has no level at '
                                 f'{at:g} Hz (NaN)')
        dead = inside & (own <= 0.0)
        if inside.any() and dead[inside].all():
            raise ValueError(f'{_label(data, record)} is zero at every line '
                             f'from {low:g} to {high:g} Hz: a dead channel '
                             'is not a measurement')
        if dead.any():
            at = float(f[dead][0])
            raise ValueError(f'{_label(data, record)} has a level of '
                             f'{own[dead][0]:g} at {at:g} Hz, which has no '
                             'logarithm; a dead line is not a quiet one')
        if f.shape == f0.shape and np.array_equal(f, f0):
            rows.append(own[keep][order])
            continue
        regridded = True
        # read from all of the record's own lines, those just outside
        # the band included, so a line of the first record that falls
        # between the band's edge and the record's next line is read;
        # `log_interpolate` drops the 0 Hz line and any NaN itself
        read = log_interpolate(lines, f, own)
        if not np.all(np.isfinite(read)):
            at = float(lines[~np.isfinite(read)][0])
            raise ValueError(f'{_label(data, record)} cannot be read at '
                             f'{at:g} Hz, a line of '
                             f'{_label(first, first_record)}: its own lines '
                             'there hold no level')
        rows.append(read)
    levels = np.asarray(rows, dtype=float)
    if regridded:
        notes.append(f'records on other lines were read onto the '
                     f'{lines.size} lines of {_label(first, first_record)} '
                     f'from {lines[0]:g} to {lines[-1]:g} Hz, log-log '
                     'between their own')
    dofs = [str(data.response_dof[record]) for data, record in members]
    if len(set(dofs)) > 1:
        notes.append(f'the set spans {len(set(dofs))} DOFs; the limit treats '
                     'them as one population')
    return FieldSet(frequencies=lines, levels=levels,
                    labels=[_label(d, r) for d, r in members], dofs=dofs,
                    kind=kind, dim=dim, q=q, srs_kind=srs_kind, notes=notes)


def tolerance_factor(count: int, coverage: float = COVERAGE,
                     confidence: float = CONFIDENCE, *,
                     sigma_known: bool = False) -> float:
    """The one-sided normal tolerance factor: how many standard
    deviations above the mean a limit stands so that it is above a
    share `coverage` of a normal population with probability
    `confidence`, from `count` samples.

    With the standard deviation estimated from the samples it is the
    noncentral t quantile over sqrt(count) (see the module's text); with
    it known it is z + z_c / sqrt(count).

    Parameters
    ----------
    count : int
        The number of samples (records).
    coverage, confidence : float
        Each strictly between 0 and 1.
    sigma_known : bool, default False
        Whether the standard deviation is known rather than estimated.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        For a coverage or confidence outside (0, 1), fewer than one
        sample, or fewer than two with the standard deviation to be
        estimated.
    """
    from scipy import stats

    count = int(count)
    coverage = _checked_share('coverage', coverage)
    confidence = _checked_share('confidence', confidence)
    if count < 1:
        raise ValueError('a tolerance limit needs at least one record')
    z = float(stats.norm.ppf(coverage))
    if sigma_known:
        return z + float(stats.norm.ppf(confidence)) / np.sqrt(count)
    if count < 2:
        raise ValueError('one record has no scatter to estimate; state a '
                         'standard deviation (sigma_db), or take the envelope')
    root = np.sqrt(count)
    return float(stats.nct.ppf(confidence, count - 1, z * root) / root)


def records_needed(coverage: float = COVERAGE,
                   confidence: float = CONFIDENCE) -> int:
    """The fewest records whose largest stands above `coverage` of the
    population with probability at least `confidence`, whatever the
    distribution: the smallest n with 1 - coverage**n >= confidence.

    Parameters
    ----------
    coverage, confidence : float
        Each strictly between 0 and 1.

    Returns
    -------
    int
    """
    coverage = _checked_share('coverage', coverage)
    confidence = _checked_share('confidence', confidence)
    n = int(np.ceil(np.log1p(-confidence) / np.log(coverage) - 1e-12))
    return max(n, 1)


@dataclass
class FieldLimit:
    """A statistical limit over a set of spectra, and what it rests on.

    Attributes:
        frequencies: The lines, in Hz.
        levels: The limit at each line, in SI.
        statistic: Which limit (`STATISTICS`).
        fieldset: The set it was taken over.
        coverage: The share of the population the limit is to stand
            above (for `envelope`, the share its distribution-free
            confidence is read at).
        confidence: The probability it does, for `distribution_free`
            and `tolerance`; for `envelope`, 1 - coverage**n, read off
            the set; None for `mean_k`.
        factor: The standard deviations above the mean, for `mean_k`
            and `tolerance`; None otherwise.
        sigma_db: The standard deviation in dB when it was stated
            rather than estimated; None otherwise.
        mean_db: The mean of the levels in dB at each line.
        scatter_db: The standard deviation of the levels in dB at each
            line, estimated from the records (n - 1 in its
            denominator); None for a single record.
        notes: The set's notes, and any the limit adds.
    """

    frequencies: np.ndarray
    levels: np.ndarray
    statistic: str
    fieldset: FieldSet
    coverage: float
    confidence: float | None
    factor: float | None
    sigma_db: float | None
    mean_db: np.ndarray
    scatter_db: np.ndarray | None
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """How many records the limit was taken over."""
        return self.fieldset.count

    @property
    def kind(self) -> str:
        """'psd' or 'srs'."""
        return self.fieldset.kind

    @property
    def levels_db(self) -> np.ndarray:
        """The limit in decibels re one SI unit."""
        return self.fieldset.db_scale * np.log10(self.levels)

    @property
    def rests_on(self) -> str:
        """In words: how many records the limit was taken over, and the
        assumption it needs."""
        n = self.count
        records = _count(n, 'record')
        if self.statistic == 'envelope':
            return (f'{records}; no distribution assumed. The largest of '
                    f'{n} stands above {self.coverage:.0%} of records like '
                    f'them with {self.confidence:.0%} confidence')
        if self.statistic == 'distribution_free':
            return (f'{records}; no distribution assumed, records taken as '
                    f'independent ({records_needed(self.coverage, self.confidence)} '
                    'needed for this coverage and confidence)')
        scatter = (f'a {self.sigma_db:g} dB standard deviation stated, not '
                   'measured' if self.sigma_db is not None
                   else 'the standard deviation estimated from them')
        return (f'{records}; the levels taken as log-normal (normal in dB) '
                f'across records, {scatter}')

    def describe(self) -> str:
        """One line: the limit, its count and its factor."""
        names = {'envelope': 'maximum envelope',
                 'distribution_free': 'distribution-free limit',
                 'mean_k': 'mean plus k standard deviations',
                 'tolerance': 'normal tolerance limit'}
        said = names[self.statistic]
        if self.statistic in ('distribution_free', 'tolerance'):
            said += (f' P{self.coverage * 100:g}/'
                     f'{self.confidence * 100:g}')
        if self.factor is not None:
            said += f', k = {self.factor:.3f}'
        return f'{said} over {_count(self.count, "record")}'


def field_limit(fieldset: FieldSet, statistic: str = 'tolerance', *,
                coverage: float = COVERAGE, confidence: float = CONFIDENCE,
                k: float | None = None,
                sigma_db: float | None = None) -> FieldLimit:
    """The statistical limit over a set, at every line.

    Parameters
    ----------
    fieldset : FieldSet
        The set (`gather`).
    statistic : str, default 'tolerance'
        One of `STATISTICS`.
    coverage : float, default `COVERAGE`
        The share of the population the limit stands above.
    confidence : float, default `CONFIDENCE`
        The probability it does, for 'distribution_free' and
        'tolerance'.
    k : float, optional
        The number of standard deviations, for 'mean_k' (where it is
        required) and nowhere else.
    sigma_db : float, optional
        A standard deviation in dB to take as known instead of
        estimating one, for 'mean_k' and 'tolerance'. It is what lets a
        single record give either.

    Returns
    -------
    FieldLimit

    Raises
    ------
    ValueError
        For an unknown statistic; a coverage or confidence outside
        (0, 1); `k` missing for 'mean_k' or given for another
        statistic; `sigma_db` that is not positive, or given to a limit
        that does not use one; a single record with no `sigma_db`
        where a standard deviation is needed; or a set too small for
        the distribution-free limit asked, with the count it needs.
    """
    if statistic not in STATISTICS:
        raise ValueError(f'unknown statistic {statistic!r}; one of '
                         + ', '.join(STATISTICS))
    coverage = _checked_share('coverage', coverage)
    confidence = _checked_share('confidence', confidence)
    if k is not None and statistic != 'mean_k':
        raise ValueError(f'k is the factor of the mean_k limit; the '
                         f'{statistic} limit sets its own')
    if sigma_db is not None:
        sigma_db = float(sigma_db)
        if statistic in ('envelope', 'distribution_free'):
            raise ValueError(f'the {statistic} limit takes no standard '
                             'deviation; it is the largest record')
        if not (np.isfinite(sigma_db) and sigma_db > 0.0):
            raise ValueError('a stated standard deviation is a positive '
                             'number of decibels')
    y = fieldset.levels_db
    n = fieldset.count
    mean = y.mean(axis=0)
    scatter = y.std(axis=0, ddof=1) if n >= 2 else None
    notes = list(fieldset.notes)
    factor = None
    stated_confidence: float | None = confidence
    if statistic in ('envelope', 'distribution_free'):
        if statistic == 'distribution_free':
            needed = records_needed(coverage, confidence)
            if n < needed:
                raise ValueError(
                    f'the largest of {_count(n, "record")} stands above '
                    f'{coverage:.0%} of the population with only '
                    f'{1 - coverage ** n:.0%} confidence; a distribution-free '
                    f'limit at {coverage:.0%} and {confidence:.0%} needs at '
                    f'least {needed} records')
        else:
            stated_confidence = 1.0 - coverage ** n
        top = y.max(axis=0)
    else:
        if statistic == 'mean_k':
            if k is None:
                raise ValueError('the mean_k limit needs k, the number of '
                                 'standard deviations above the mean')
            factor = float(k)
            if not np.isfinite(factor):
                raise ValueError('k is a finite number of standard '
                                 'deviations')
            stated_confidence = None
        else:
            factor = tolerance_factor(n, coverage, confidence,
                                      sigma_known=sigma_db is not None)
        if sigma_db is not None:
            spread = np.full_like(mean, sigma_db)
        elif scatter is None:
            raise ValueError('one record has no scatter to estimate; state a '
                             'standard deviation (sigma_db), or take the '
                             'envelope')
        else:
            spread = scatter
        top = mean + factor * spread
        if sigma_db is not None and scatter is not None:
            notes.append(f'the records scatter {float(np.median(scatter)):.2f} '
                         f'dB at the median line; {sigma_db:g} dB was stated')
    return FieldLimit(
        frequencies=fieldset.frequencies.copy(),
        levels=10.0 ** (top / fieldset.db_scale), statistic=statistic,
        fieldset=fieldset, coverage=coverage, confidence=stated_confidence,
        factor=factor, sigma_db=sigma_db, mean_db=mean, scatter_db=scatter,
        notes=notes)


# ---- the straight-line envelope ------------------------------------------


def _least_cover(x: np.ndarray, y: np.ndarray, points: list[int]
                 ) -> np.ndarray:
    """The levels at breakpoints `points` (indices into `x`) of the
    straight lines that are at or above `y` at every point and stand
    least above it on average over `x`: a linear program, since the
    lines are linear in the levels and so is their area."""
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    at = np.asarray(points)
    count = len(at)
    # each point is held under the line of the span it falls in; a
    # breakpoint holds itself (a weight of one on its own level). Two
    # weights a row, so the matrix is kept sparse: a narrowband limit
    # has thousands of rows and a solve is made for every breakpoint
    # tried
    span = np.clip(np.searchsorted(x[at], x, side='right') - 1, 0, count - 2)
    left, right = x[at][span], x[at][span + 1]
    weight = (x - left) / (right - left)
    index = np.arange(len(x))
    rows = csr_matrix((np.concatenate([1.0 - weight, weight]),
                       (np.concatenate([index, index]),
                        np.concatenate([span, span + 1]))),
                      shape=(len(x), count))
    # the area under the lines over x is the trapezoid sum of the
    # levels: each breakpoint weighted by half the spans either side
    gaps = np.diff(x[at])
    area = np.zeros(count)
    area[:-1] += 0.5 * gaps
    area[1:] += 0.5 * gaps
    solved = linprog(area, A_ub=-rows, b_ub=-y, bounds=[(None, None)] * count,
                     method='highs')
    if not solved.success:                           # pragma: no cover
        raise ValueError(f'the envelope could not be fitted: {solved.message}')
    levels = np.asarray(solved.x, dtype=float)
    # the solver meets each bound to its own tolerance; a line a
    # hair under the limit is lifted so the cover holds, read the way
    # every caller reads the lines (`_lines_at`)
    short = float(np.max(y - _lines_at(x, points, levels)))
    return levels + max(short, 0.0)


def _lines_at(x: np.ndarray, points: list[int],
              levels: np.ndarray) -> np.ndarray:
    return np.interp(x, x[np.asarray(points)], levels)


def cover(x: ArrayLike, y: ArrayLike, max_overshoot: float,
          max_points: int | None = None) -> tuple[list[int], np.ndarray]:
    """Straight lines over a curve: at or above it at every point, and,
    where `max_points` allows, no more than `max_overshoot` above it at
    any.

    `x` and `y` are the axes the lines are straight on (log frequency
    and decibels for a spectrum). The breakpoints are at points of `x`
    and their levels are free: for a given set of breakpoints the
    levels are those of the cover that stands least above the curve on
    average over `x` (`_least_cover`). It starts with one line from the
    first point to the last. While some point is more than the
    allowance under the lines, a breakpoint is added in the span with
    the most area between its line and the curve, at the point that
    span stands highest above, and the levels are fitted again; it
    stops at `max_points`, the allowance then not met. When the
    allowance is met, any breakpoint whose removal still meets it is
    removed, in passes from the low end until a pass removes none.

    Parameters
    ----------
    x, y : array-like
        The points, `x` strictly increasing.
    max_overshoot : float
        The allowance, in the units of `y`; zero or more.
    max_points : int, optional
        The most breakpoints, two or more; no limit when omitted.

    Returns
    -------
    (list of int, numpy.ndarray)
        The breakpoints as indices into `x`, and the lines' level at
        each, in the units of `y`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    points = [0, len(x) - 1]
    levels = _least_cover(x, y, points)
    while True:
        over = _lines_at(x, points, levels) - y
        if over.max() <= max_overshoot + _SLACK_DB:
            break
        if max_points is not None and len(points) >= max_points:
            return points, levels
        best, where = -1.0, None
        for p, q in itertools.pairwise(points):
            if q - p < 2:
                continue
            inside = over[p:q + 1]
            # twice the trapezoid area; only the order of the spans
            # is read from it
            excess = float(np.sum((inside[1:] + inside[:-1])
                                  * np.diff(x[p:q + 1])))
            if excess > best:
                # the split goes where the curve turns: the point
                # farthest from the chord through the curve at the
                # span's ends. The point the lines stand highest above
                # was tried first and spent its breakpoints trimming
                # the tails of a hump one line at a time, never
                # reaching its knees
                chord = y[p] + (y[q] - y[p]) * (x[p:q + 1] - x[p]) / (
                    x[q] - x[p])
                turn = np.abs(y[p:q + 1] - chord)
                best, where = excess, p + 1 + int(np.argmax(turn[1:-1]))
        if where is None:
            # every span is two neighboring points and still the lines
            # stand above: cannot happen, since a line through two
            # points of the curve covers them exactly
            break                                    # pragma: no cover
        points = sorted({*points, where})
        levels = _least_cover(x, y, points)
    # a breakpoint the others do not need. Passes from the low end
    # until one removes nothing, since a removal can free a breakpoint
    # that an earlier test kept
    removed = True
    while removed:
        removed, i = False, 1
        while i < len(points) - 1:
            trial = points[:i] + points[i + 1:]
            fitted = _least_cover(x, y, trial)
            if (np.max(_lines_at(x, trial, fitted) - y)
                    <= max_overshoot + _SLACK_DB):
                points, levels, removed = trial, fitted, True
            else:
                i += 1
    return points, levels


@dataclass
class EnvelopeFit:
    """A limit covered by straight lines on log-log axes.

    Attributes:
        frequencies: The breakpoints, in Hz.
        levels: The envelope's level at each breakpoint, in SI: at or
            above the limit, and lowered by `added_db` under 'conserve'.
        limit: The limit covered.
        max_overshoot_db: The allowance it was fitted with.
        overshoot_db: The most the envelope stands above the limit at
            any line, in dB.
        rule: 'exceed' or 'conserve' (`RMS_RULES`).
        max_breakpoints: The most breakpoints it was allowed, or None.
        met: Whether the envelope is within the allowance at every
            line; False when the breakpoints allowed ran out first.
        limit_area: The area under the limit over its lines, read
            log-log between them (a PSD's mean square in SI); NaN for an
            SRS.
        envelope_area: The same for the covering lines, before any
            lowering; NaN for an SRS.
        added_db: The area the cover adds over the limit, in dB
            (10 log10 of the ratio); NaN for an SRS.
        below: Under 'conserve', how many of the limit's lines the
            lowered envelope falls below; 0 under 'exceed'.
        deepest_db: Under 'conserve', the most it falls below the limit
            at any line, in dB (a positive number); 0 otherwise.
        notes: The limit's notes, and any the fit adds.
    """

    frequencies: np.ndarray
    levels: np.ndarray
    limit: FieldLimit
    max_overshoot_db: float
    overshoot_db: float
    rule: str
    limit_area: float
    envelope_area: float
    added_db: float
    below: int
    deepest_db: float
    max_breakpoints: int | None = MAX_BREAKPOINTS
    met: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def breakpoints(self) -> int:
        """How many breakpoints the envelope has."""
        return len(self.frequencies)

    @property
    def limit_rms(self) -> float:
        """The rms under the limit, the square root of its area; NaN for
        an SRS."""
        return float(np.sqrt(self.limit_area))

    @property
    def envelope_rms(self) -> float:
        """The rms under the envelope as it stands (lowered under
        'conserve'); NaN for an SRS."""
        if self.rule == 'conserve':
            return self.limit_rms
        return float(np.sqrt(self.envelope_area))

    def at(self, frequencies: ArrayLike) -> np.ndarray:
        """The envelope's level at any frequencies inside its band,
        straight between breakpoints on log-log axes; NaN outside."""
        return log_interpolate(frequencies, self.frequencies, self.levels)

    def describe(self) -> str:
        """One line: the breakpoints, the overshoot and the area added."""
        said = (f'{self.breakpoints} breakpoints, at most '
                f'{self.overshoot_db:.2f} dB over the limit'
                + ('' if self.met else
                   f' (past the {self.max_overshoot_db:g} dB asked)'))
        if self.limit.kind == 'psd':
            if self.rule == 'conserve':
                said += (f', lowered {self.added_db:.2f} dB to keep the rms: '
                         f'{_count(self.below, "line")} under '
                         f'the limit, by up to {self.deepest_db:.2f} dB')
            else:
                said += f', {self.added_db:+.2f} dB of area added'
        return said

    def dof(self, dof: str | None = None) -> str:
        """The DOF a specification from this envelope is written at:
        `dof` when given, or the set's one DOF.

        Raises
        ------
        ValueError
            When the set spans several DOFs and none is named.
        """
        if dof is not None:
            return str(dof)
        dofs = sorted(set(self.limit.fieldset.dofs))
        if len(dofs) != 1:
            raise ValueError(f'the set spans {", ".join(dofs)}; name the DOF '
                             'the specification is written at')
        return dofs[0]

    def margined(self, margin_db: float = 0.0) -> np.ndarray:
        """The breakpoint levels with `margin_db` added, on the set's
        decibel convention (10 log10 for a PSD, 20 log10 for an SRS)."""
        margin_db = float(margin_db)
        if not np.isfinite(margin_db):
            raise ValueError('a margin is a finite number of decibels')
        return self.levels * 10.0 ** (margin_db / self.limit.fieldset.db_scale)

    def specification(self, margin_db: float = 0.0, dof: str | None = None,
                      comment: str = '') -> Specification | ShockSpecification:
        """The specification this envelope writes, with `margin_db`
        added: a `Specification` at the breakpoints for a PSD, through
        `core.author.SpecificationDraft` (with that sheet's starter
        bands), or a `ShockSpecification` on the SRS's own spacing,
        Q and kind for an SRS, with no bands.

        Parameters
        ----------
        margin_db : float, default 0
            Decibels added to every breakpoint.
        dof : str, optional
            The DOF to write it at; the set's one DOF when omitted.
        comment : str, optional
            For the records; a line saying how it was derived when
            omitted.

        Returns
        -------
        Specification or ShockSpecification

        Raises
        ------
        ValueError
            For undeclared units, a set spanning several DOFs with none
            named, or a margin that is not finite.
        """
        from ..units import UNKNOWN
        from .data import ShockSpecification, channel_quantities

        fieldset = self.limit.fieldset
        if fieldset.dim == UNKNOWN:
            raise ValueError('the units are not declared; a specification is '
                             'written in a declared quantity, so set the '
                             'units first')
        at = self.dof(dof)
        levels = self.margined(margin_db)
        said = comment or (f'derived from {fieldset.count} records: '
                           f'{self.limit.describe()}; {self.describe()}'
                           + (f'; {margin_db:+g} dB margin' if margin_db else ''))
        if fieldset.kind == 'srs':
            from .octave import per_octave_of

            try:
                per_octave = per_octave_of(fieldset.frequencies)
            except ValueError:
                per_octave = None
            spec = ShockSpecification.from_breakpoints(
                self.frequencies, levels, at,
                per_octave=per_octave or None,
                q=fieldset.q, kind=fieldset.srs_kind,
                ordinate_dim=fieldset.dim)
            spec.comment = [said] * len(spec.comment)
            return spec
        from .author import SpecificationDraft

        quantity, _other = channel_quantities(fieldset.dim)
        draft = SpecificationDraft(channels=[at], dims=[quantity],
                                   frequencies=list(self.frequencies),
                                   levels=[list(levels)])
        return draft.make(said)


def fit_envelope(limit: FieldLimit, max_overshoot_db: float = OVERSHOOT_DB,
                 rms: str = 'exceed', *,
                 max_breakpoints: int | None = MAX_BREAKPOINTS,
                 progress: Callable[[int, int], None] | None = None
                 ) -> EnvelopeFit:
    """Cover a limit with straight lines on log-log axes, by the rules
    in the module's text.

    Parameters
    ----------
    limit : FieldLimit
        The limit (`field_limit`).
    max_overshoot_db : float, default `OVERSHOOT_DB`
        The most the lines may stand above the limit at any line, in
        dB on the set's convention.
    rms : str, default 'exceed'
        'exceed', the cover itself, or 'conserve', the cover lowered to
        the limit's area (a PSD only).
    max_breakpoints : int or None, default `MAX_BREAKPOINTS`
        The most breakpoints, two or more; None for no limit. Where it
        stops the fit, the allowance is not met, and the result says so
        (`met`).
    progress : callable, optional
        Called with ``(done, total)``; the fit cannot count itself, so
        it is announced as busy and ticked once when done.

    Returns
    -------
    EnvelopeFit

    Raises
    ------
    ValueError
        For an allowance that is negative or not finite, fewer than two
        breakpoints allowed, an unknown rule, or 'conserve' asked of an
        SRS.
    """
    max_overshoot_db = float(max_overshoot_db)
    if not (np.isfinite(max_overshoot_db) and max_overshoot_db >= 0.0):
        raise ValueError('the overshoot allowed is zero or more decibels')
    if max_breakpoints is not None:
        max_breakpoints = int(max_breakpoints)
        if max_breakpoints < 2:
            raise ValueError('an envelope has at least two breakpoints, its '
                             'two ends')
    if rms not in RMS_RULES:
        raise ValueError(f'unknown rms rule {rms!r}; one of '
                         + ', '.join(RMS_RULES))
    if rms == 'conserve' and limit.kind == 'srs':
        raise ValueError('an SRS has no area to conserve: each point is a '
                         'peak, not a share of a mean square')
    ticker = Ticker(progress)
    ticker.add(1)
    ticker.busy()
    f = limit.frequencies
    x = np.log10(f)
    y = limit.levels_db
    points, fitted_db = cover(x, y, max_overshoot_db, max_breakpoints)
    ticker.tick()
    breakpoints = f[points]
    levels = 10.0 ** (fitted_db / limit.fieldset.db_scale)
    line_db = _lines_at(x, points, fitted_db)
    overshoot = float(np.max(line_db - y))
    met = overshoot <= max_overshoot_db + _SLACK_DB
    notes = list(limit.notes)
    if not met:
        worst = float(f[int(np.argmax(line_db - y))])
        notes.append(f'{len(points)} breakpoints, the most allowed, leave the '
                     f'envelope {overshoot:.2f} dB over the limit at '
                     f'{worst:g} Hz, past the {max_overshoot_db:g} dB asked')
    area = fit_area = added = float('nan')
    below, deepest = 0, 0.0
    if limit.kind == 'psd':
        low, high = float(f[0]), float(f[-1])
        area = log_log_area(f, limit.levels, low, high)
        fit_area = log_log_area(breakpoints, levels, low, high)
        added = float(10.0 * np.log10(fit_area / area))
        if rms == 'conserve':
            levels = levels * 10.0 ** (-added / 10.0)
            short = (y - (line_db - added))
            below = int(np.sum(short > _SLACK_DB))
            deepest = float(max(np.max(short), 0.0))
    return EnvelopeFit(
        frequencies=np.asarray(breakpoints, dtype=float),
        levels=np.asarray(levels, dtype=float), limit=limit,
        max_overshoot_db=max_overshoot_db, overshoot_db=max(overshoot, 0.0),
        rule=rms, limit_area=area, envelope_area=fit_area, added_db=added,
        max_breakpoints=max_breakpoints, met=met, below=below,
        deepest_db=deepest, notes=notes)


def derive(spectra: Any, records: Sequence[int] | None = None, *,
           statistic: str = 'tolerance', coverage: float = COVERAGE,
           confidence: float = CONFIDENCE, k: float | None = None,
           sigma_db: float | None = None,
           band: tuple[float, float] | None = None,
           max_overshoot_db: float = OVERSHOOT_DB, rms: str = 'exceed',
           max_breakpoints: int | None = MAX_BREAKPOINTS,
           progress: Callable[[int, int], None] | None = None
           ) -> EnvelopeFit:
    """The whole chain from a set of spectra: `gather`, `field_limit`
    and `fit_envelope`, with the arguments each takes.

        fit = derive(flights, statistic='tolerance', coverage=0.95,
                     confidence=0.5)
        spec = fit.specification(margin_db=3.0)

    Returns
    -------
    EnvelopeFit
        The envelope; its `limit` is the limit and `limit.fieldset` the
        set.
    """
    fieldset = gather(spectra, records, band=band)
    limit = field_limit(fieldset, statistic, coverage=coverage,
                        confidence=confidence, k=k, sigma_db=sigma_db)
    return fit_envelope(limit, max_overshoot_db, rms,
                        max_breakpoints=max_breakpoints, progress=progress)


def settings_of(fit: EnvelopeFit) -> dict[str, Any]:
    """The arguments `derive` was called with, as far as the result
    records them: what a script needs to make it again."""
    limit = fit.limit
    out: dict[str, Any] = {'statistic': limit.statistic,
                           'coverage': limit.coverage}
    if limit.statistic in ('distribution_free', 'tolerance'):
        out['confidence'] = limit.confidence
    if limit.statistic == 'mean_k':
        out['k'] = limit.factor
    if limit.sigma_db is not None:
        out['sigma_db'] = limit.sigma_db
    out['max_overshoot_db'] = fit.max_overshoot_db
    out['max_breakpoints'] = fit.max_breakpoints
    out['rms'] = fit.rule
    return out

