"""Multi-point control: several control channels combined into one, and
the two judgements a test run that way is accepted on.

A random test on one shaker is often controlled at more than one point.
The controller combines the control channels' spectra, line by line,
into one composite and holds that to the reference; how they are
combined is the strategy:

- 'average', the mean of the channels' densities, each counted alike;
- 'weighted', the mean with weights chosen for the channels;
- 'maximum', the highest channel at each line, so nothing sensed
  exceeds the reference and every other point is under it;
- 'minimum', the lowest channel at each line, so every point sensed
  meets the reference and every other point is over it.

The average is taken on the densities themselves, not on their
decibels, because that is the arithmetic a controller does. With two
channels, one at four times the reference and one at a quarter, the
average is just over twice the reference: the hot spot shows, far
smaller than it is.

So a run controlled this way has two results, and a report carries
both: the composite against the composite band, and each control
channel against its own band, which is wider. The second exists
because the first can pass with a channel far out. A channel reading
near zero inside an average (a dead or loose sensor) leaves the
composite in tolerance while the controller drives the live channels
up to make up for it, and nothing on the composite says so. The
reading here lists, for each channel, the lines where it is outside its
own band while the composite is inside its band, which is the fact the
composite hides.

Every judgement is `compliance.compare` and `compliance.exceedances`,
the comparison every other random reading makes: the composite and
each channel are compared cell by cell, as areas, against a band
around the reference, at the comparison's one scale. Nothing here
judges a line by its own rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import Psd, Specification

#: the strategies a composite is formed by, in the order a controller's
#: settings usually list them
STRATEGIES = ('average', 'weighted', 'maximum', 'minimum')

#: the band each control channel is judged against when none is given,
#: in dB below and above the reference (a density, so ten log ten). A
#: starting point to edit, not a standard: the test plan names the
#: individual band, and it is usually wider than the composite's
INDIVIDUAL_DB = (-6.0, 6.0)

#: the composite band offered when the specification carries no limits
#: of its own, in dB below and above the reference
COMPOSITE_DB = (-3.0, 3.0)

#: how far below the other channels, in dB at the median line, a channel
#: reads before the reading says it reads as a dead sensor does. A
#: transmissibility across a fixture moves a channel a few dB, a mode
#: ten or so over a narrow band; twenty across the whole band is a
#: sensor that is not reporting the motion
DEAD_DB = 20.0


@dataclass(frozen=True)
class ChannelReading:
    """One control channel of a multi-point reading, against its own
    band."""

    #: the channel's label, its response DOF
    label: str
    #: which record of the measurement it is
    record: int
    #: its share of the composite: the normalized weight under
    #: 'average' and 'weighted', NaN under 'maximum' and 'minimum',
    #: where no channel has a fixed share
    weight: float
    #: `compliance.compare` against the individual band, with
    #: 'outside_lines' and 'outside_percent' for the cells outside it
    result: dict[str, Any]
    #: per line of the measurement, above and below the individual band
    over: np.ndarray
    under: np.ndarray
    #: per line, outside the individual band where the composite is
    #: inside the composite band
    hidden: np.ndarray
    #: under 'maximum' and 'minimum', the lines at which this channel
    #: is the one the composite takes; None under the averages
    held_lines: int | None = None

    @property
    def outside(self) -> np.ndarray:
        """Per line, outside the individual band either way."""
        return self.over | self.under

    @property
    def outside_lines(self) -> int:
        """How many lines are outside the individual band."""
        return int(self.outside.sum())

    @property
    def hidden_lines(self) -> int:
        """How many lines are outside the individual band while the
        composite is inside its own."""
        return int(self.hidden.sum())


@dataclass(frozen=True)
class MultiPointReading:
    """What `multi_point` read: the composite, its judgement, and each
    channel's."""

    #: one of `STRATEGIES`
    strategy: str
    #: the composite, a PSD of one record on the measurement's lines
    composite: Psd
    #: the specification record the composite is held to
    spec_record: int
    #: the composite band and the individual band as specifications of
    #: one record each, the band held as its abort limits; the plot
    #: draws these, so the curves drawn are the curves judged against
    composite_band: Specification
    individual_band: Specification
    #: each band in words: '−3/+3 dB', or 'the warning limits'
    composite_text: str
    individual_text: str
    #: `compliance.compare` of the composite against the composite
    #: band, with 'outside_lines' and 'outside_percent'
    result: dict[str, Any]
    #: per line, the composite above and below the composite band
    over: np.ndarray
    under: np.ndarray
    #: the control channels, in the order given
    channels: tuple[ChannelReading, ...]
    #: the dB added to the measurement for the comparison
    scale_db: float
    #: what the reader should know about the data: lines not formed,
    #: a channel reading as dead, a zero weight, a scale detected,
    #: units undeclared
    notes: tuple[str, ...] = field(default=())

    @property
    def outside_lines(self) -> int:
        """How many lines of the composite are outside its band."""
        return int((self.over | self.under).sum())

    @property
    def hiding(self) -> tuple[ChannelReading, ...]:
        """The channels outside their own band at a line where the
        composite is inside its band."""
        return tuple(c for c in self.channels if c.hidden_lines)


def composite(values: ArrayLike, strategy: str = 'average',
              weights: ArrayLike | None = None) -> np.ndarray:
    """The composite of several channels' densities, line by line.

    `values` is (channels, lines), real densities on one set of lines.
    'average' and 'weighted' take the weighted mean of the densities
    (weights normalized to sum to one, equal under 'average'),
    'maximum' and 'minimum' the highest and lowest channel at each line.

    A line where any channel is not a finite, non-negative number has no
    composite and is NaN: the composite of these channels was not formed
    there, and averaging the rest would be a composite of other
    channels. A channel reading zero is a number and is counted, as a
    controller counts it, which is what lets a dead channel be found.

    Parameters
    ----------
    values : array_like
        The densities, one row per channel.
    strategy : str, default 'average'
        One of `STRATEGIES`.
    weights : array_like, optional
        One per channel, finite and not negative, under 'weighted'
        only; normalized here.

    Returns
    -------
    numpy.ndarray
        One value per line.

    Raises
    ------
    ValueError
        For a strategy not in `STRATEGIES`, weights under a strategy
        that takes none or none under 'weighted', or weights that are
        not one finite, non-negative number per channel with a
        positive sum.
    """
    values = np.atleast_2d(np.asarray(np.real(values), dtype=float))
    share = _weights(strategy, weights, values.shape[0])
    with np.errstate(invalid='ignore'):
        usable = np.isfinite(values) & (values >= 0.0)
    formed = usable.all(axis=0)
    out = np.full(values.shape[1], np.nan)
    if not formed.any():
        return out
    held = values[:, formed]
    if strategy == 'maximum':
        out[formed] = held.max(axis=0)
    elif strategy == 'minimum':
        out[formed] = held.min(axis=0)
    else:
        out[formed] = share @ held
    return out


def _weights(strategy: str, weights: ArrayLike | None,
             count: int) -> np.ndarray | None:
    """The normalized weights a strategy averages with, None for the
    extremal strategies; refused as `composite` says."""
    if strategy not in STRATEGIES:
        raise ValueError(f'the strategy is one of {", ".join(STRATEGIES)}, '
                         f'not {strategy!r}')
    if strategy != 'weighted':
        if weights is not None:
            raise ValueError(
                f'weights are for the weighted strategy; {strategy} takes '
                'none')
        return (np.full(count, 1.0 / count)
                if strategy == 'average' else None)
    if weights is None:
        raise ValueError('the weighted strategy needs a weight per channel')
    share = np.asarray(weights, dtype=float).ravel()
    if share.size != count:
        raise ValueError(f'{share.size} weights for {count} channels; give '
                         'one per channel')
    if not np.isfinite(share).all() or (share < 0.0).any():
        raise ValueError('a weight is a finite number, zero or more')
    if not share.sum() > 0.0:
        raise ValueError('the weights add up to zero, so nothing is '
                         'averaged')
    return share / share.sum()


def multi_point(specification: Specification, measured: Psd,
                channels: Sequence[int | str] | None = None,
                strategy: str = 'average',
                weights: ArrayLike | None = None,
                composite_band: str | tuple[float, float] = 'warning',
                individual_band: str | tuple[float, float] = INDIVIDUAL_DB,
                spec_record: int | None = None,
                scale_db: float | None = None) -> MultiPointReading:
    """Read a multi-point control run: the composite of its control
    channels against the composite band, and each channel against its
    own.

    The composite is formed line by line (`composite`) from the control
    channels' densities and compared with the reference by
    `compliance.compare`; each channel is compared the same way against
    the individual band. A band is a pair of the specification's own
    limits ('warning' or 'abort'), or a pair of dB below and above the
    reference, ten log ten on a density (an infinite edge is no edge on
    that side). For each channel the lines outside its own band are
    found, and among them those where the composite is inside its band:
    the lines the composite hides.

    The reference is one record of the specification: `spec_record`
    when given, the only record when there is one, and otherwise the
    record each channel pairs with (`compliance.matched_records`),
    which must all ask for the same thing, since a composite is held to
    one reference. The scale is resolved once for every channel, as
    `compliance.compare_all` resolves it.

    Parameters
    ----------
    specification : Specification
        The reference and, where its limits are used, the bands.
    measured : Psd
        The run, holding the control channels' autospectra.
    channels : sequence of int or str, optional
        The control channels, by record or by response DOF; by default
        the autospectra that pair with the specification's records.
        Two or more.
    strategy : str, default 'average'
        One of `STRATEGIES`.
    weights : array_like, optional
        One per channel, under 'weighted' only.
    composite_band, individual_band : str or tuple of float
        'warning' or 'abort' for that pair of the specification's
        limits, or (below, above) in dB.
    spec_record : int, optional
        The reference record, chosen as above when omitted.
    scale_db : float, optional
        The comparison's scale; resolved through
        `compliance.comparison_scale_db` when omitted.

    Returns
    -------
    MultiPointReading
        The composite and its judgement, each channel's, and the notes.

    Raises
    ------
    ValueError
        With the reason, for anything the reading cannot be made from:
        a strategy or weights `composite` refuses, fewer than two
        channels, a channel named twice, not found, ambiguous or a
        cross term, channels of different quantities, channels whose
        specification records differ, a band the specification does not
        carry or edges on the wrong side of the reference, or a pair
        `compliance.compare` refuses (bands of another fraction,
        another quantity, units declared on one side only).
    """
    from .compliance import (
        comparison_scale_db,
        units_check,
    )
    from .data import Psd, Specification

    if not isinstance(specification, Specification):
        raise TypeError('the reference is a random specification')
    if not isinstance(measured, Psd) or isinstance(measured, Specification):
        raise TypeError('the control channels are a measured PSD')
    records = control_channels(specification, measured, channels)
    labels = [str(measured.response_dof[i]) for i in records]
    _weights(strategy, weights, len(records))       # refused before work
    # one quantity across the channels, by the rule every comparison
    # uses: an average of an acceleration and a velocity is no quantity
    first = records[0]
    raw_note = None
    for i, label in zip(records[1:], labels[1:], strict=True):
        refused, note = units_check(measured.ordinate_dim[first],
                                    measured.ordinate_dim[i],
                                    wanted_name=f'channel {labels[0]}',
                                    held_name=f'channel {label}')
        if refused is not None:
            raise ValueError(refused.replace('cannot be compared',
                                             'cannot be combined'))
        raw_note = raw_note or note
    j = _reference(specification, measured, records, spec_record)
    composite_spec, composite_text = _band(specification, j, composite_band,
                                           'composite')
    individual_spec, individual_text = _band(specification, j,
                                             individual_band, 'individual')
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured, None,
                                       records, labels)
        detected = (scale_db
                    and getattr(measured, 'scale_db', None) is None)
    else:
        detected = False
    scale_db = float(scale_db)

    values = np.real(np.asarray(measured.ordinate))[records]
    formed = composite(values, strategy, weights)
    summed = Psd(measured.abscissa, formed[None],
                 response_dof=[''],
                 ordinate_dim=[measured.ordinate_dim[first]],
                 ordinate_unit=[measured.ordinate_unit[first]],
                 dimension_hint=[measured.dimension_hint[first]],
                 comment=[f'{strategy} of {", ".join(labels)}'],
                 bandwidth=measured.bandwidth)
    summed.interpolation = measured.interpolation

    result, over, under = _judged(composite_spec, summed, 0, scale_db)
    composite_in = ~(over | under) & np.isfinite(formed)
    share = _weights(strategy, weights, len(records))
    held = _held(values, formed, strategy)
    readings = []
    for k, (i, label) in enumerate(zip(records, labels, strict=True)):
        found, c_over, c_under = _judged(individual_spec, measured, i,
                                         scale_db)
        readings.append(ChannelReading(
            label=label, record=int(i),
            weight=float('nan') if share is None else float(share[k]),
            result=found, over=c_over, under=c_under,
            hidden=(c_over | c_under) & composite_in,
            held_lines=None if held is None else int(held[k])))

    notes = _notes(values, labels, share, formed)
    if detected:
        notes.append(f'the measurement was scaled by {scale_db:+g} dB, '
                     f'detected from the data as a run at {-scale_db:+g} '
                     'dB')
    if raw_note is not None or result.get('note'):
        notes.append(raw_note or result['note'])
    return MultiPointReading(
        strategy=strategy, composite=summed, spec_record=j,
        composite_band=composite_spec, individual_band=individual_spec,
        composite_text=composite_text, individual_text=individual_text,
        result=result, over=over, under=under, channels=tuple(readings),
        scale_db=scale_db, notes=tuple(notes))


def control_channels(specification: Specification, measured: Psd,
                     channels: Sequence[int | str] | None = None
                     ) -> list[int]:
    """The measured records a multi-point reading combines, by the rule
    `multi_point` uses: the records named, by index or by response
    DOF, or by default the autospectra that pair with the
    specification's records (`compliance.matched_records`). The window
    asks this before it offers the reading, so the reading is offered
    exactly where it can be made.

    Parameters
    ----------
    specification : Specification
        The reference the channels pair with.
    measured : Psd
        The run.
    channels : sequence of int or str, optional
        The channels named; the paired autospectra when omitted.

    Returns
    -------
    list of int
        Two or more records of `measured`, in the order named.

    Raises
    ------
    ValueError
        With the reason, for fewer than two channels, or a channel
        named twice, not found, ambiguous or a cross term.
    """
    from .compliance import matched_records

    autos = [i for i in range(measured.num_records)
             if measured.reference_dof is None
             or str(measured.reference_dof[i]) == str(measured.response_dof[i])]
    if channels is None:
        found = []
        for _label, _j, i in matched_records(specification, measured):
            if i in autos and i not in found:
                found.append(i)
    else:
        found = []
        for name in channels:
            if isinstance(name, (int, np.integer)):
                i = int(name)
                if not 0 <= i < measured.num_records:
                    raise ValueError(f'there is no record {i}; the '
                                     f'measurement has '
                                     f'{measured.num_records}')
            else:
                at = [k for k in range(measured.num_records)
                      if str(measured.response_dof[k]) == str(name)
                      and k in autos]
                if not at:
                    raise ValueError(f'the measurement has no autospectrum '
                                     f'at {name}')
                if len(at) > 1:
                    raise ValueError(f'the measurement has {len(at)} '
                                     f'autospectra at {name}; name the one '
                                     'meant by its record')
                i = at[0]
            if i not in autos:
                raise ValueError(
                    f'record {i} is a cross term, '
                    f'{measured.response_dof[i]}/{measured.reference_dof[i]}; '
                    'a composite is of autospectra')
            if i in found:
                raise ValueError(f'{measured.response_dof[i]} is named '
                                 'twice; each channel counts once')
            found.append(i)
    if len(found) < 2:
        raise ValueError(
            'a composite needs two or more control channels; '
            + ('one is a single-point comparison, which the comparison '
               'itself reads' if found else
               'none of the measurement\'s autospectra pairs with the '
               'specification, so name them'))
    return found


def _reference(specification: Specification, measured: Psd,
               records: list[int], spec_record: int | None) -> int:
    """The one specification record the composite is held to."""
    from .compliance import matched_records

    if spec_record is not None:
        if not 0 <= int(spec_record) < specification.num_records:
            raise ValueError(f'the specification has no record '
                             f'{spec_record}')
        return int(spec_record)
    if specification.num_records == 1:
        return 0
    paired = [j for _label, j, i in matched_records(
        specification, measured, None, records) if i in records]
    if not paired:
        raise ValueError('the specification names none of these channels; '
                         'say which of its records is the reference')
    first = np.real(np.asarray(specification.ordinate[paired[0]]))
    for j in paired[1:]:
        other = np.real(np.asarray(specification.ordinate[j]))
        if not np.allclose(first, other, rtol=1e-9, atol=0.0,
                           equal_nan=True):
            raise ValueError(
                f'the specification asks {specification.response_dof[j]} '
                f'for another level than {specification.response_dof[paired[0]]}; '
                'a composite is held to one reference, so say which '
                'record it is')
    return paired[0]


def _db_text(below: float, above: float) -> str:
    """'−3/+3 dB', an open edge written as 'up' or 'down'."""
    low = 'down' if np.isinf(below) else f'{below:+g}'
    high = 'up' if np.isinf(above) else f'{above:+g}'
    return f'{low}/{high} dB'.replace('-', '−')


def _band(specification: Specification, j: int,
          band: str | tuple[float, float],
          name: str) -> tuple[Specification, str]:
    """(the band as a specification of one record holding it as its
    abort limits, the band in words)."""
    from .data import Specification

    target = np.real(np.asarray(specification.ordinate[j], dtype=float))
    if isinstance(band, str):
        if band not in ('warning', 'abort'):
            raise ValueError(f'the {name} band is \'warning\' or \'abort\' '
                             f'or a pair in dB, not {band!r}')
        edges = {edge: specification.limits.get(f'{band}_{edge}')
                 for edge in ('lower', 'upper')}
        edges = {edge: np.atleast_2d(values)[j]
                 for edge, values in edges.items() if values is not None}
        if not edges:
            raise ValueError(
                f'the specification carries no {band} limits to judge the '
                f'{name} against; give the band in dB')
        text = f'the {band} limits'
    else:
        try:
            below, above = (float(v) for v in band)
        except (TypeError, ValueError):
            raise ValueError(f'the {name} band is a pair of dB, below and '
                             f'above the reference, not {band!r}') from None
        if np.isnan(below) or np.isnan(above) or below > 0.0 or above < 0.0:
            raise ValueError(
                f'the {name} band\'s lower edge is at or below the reference '
                f'and its upper edge at or above it; {_db_text(below, above)} '
                'is not')
        if np.isinf(below) and np.isinf(above):
            raise ValueError(f'the {name} band has no edge on either side')
        edges = {edge: target * 10.0 ** (db / 10.0)
                 for edge, db in (('lower', below), ('upper', above))
                 if np.isfinite(db)}
        text = _db_text(below, above)
    held = Specification(
        specification.abscissa, target[None],
        response_dof=[specification.response_dof[j]],
        ordinate_dim=[specification.ordinate_dim[j]],
        ordinate_unit=[specification.ordinate_unit[j]],
        dimension_hint=[specification.dimension_hint[j]],
        bandwidth=specification.bandwidth,
        **{f'abort_{edge}': values[None] for edge, values in edges.items()})
    held.interpolation = specification.interpolation
    return held, text


def _judged(band: Specification, measured: Psd, record: int,
            scale_db: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """(compare's result with the band's counts renamed, lines over,
    lines under) for one record against one band, refused with
    compare's reason."""
    from .compliance import compare, exceedances

    found = compare(band, measured, 0, record, scale_db=scale_db)
    if found.get('refused'):
        raise ValueError(found['refused'])
    # the band is held as the abort pair, so that is what compare
    # counted against; renamed so the result says what it is
    found = {key: value for key, value in found.items()
             if not key.startswith(('warning_', 'abort_'))} | {
        'outside_lines': int(found.get('abort_lines', 0)),
        'outside_percent': float(found.get('abort_percent', 0.0))}
    over, under = exceedances(band, measured, 0, record, 'abort', scale_db)
    return found, over, under


def _held(values: np.ndarray, formed: np.ndarray,
          strategy: str) -> np.ndarray | None:
    """Under an extremal strategy, how many lines each channel is the one
    taken (the first, where two read the same); None under the
    averages."""
    if strategy not in ('maximum', 'minimum'):
        return None
    good = np.isfinite(formed)
    pick = (np.argmax if strategy == 'maximum' else np.argmin)(
        values[:, good], axis=0)
    return np.bincount(pick, minlength=values.shape[0])


def _notes(values: np.ndarray, labels: list[str],
           share: np.ndarray | None, formed: np.ndarray) -> list[str]:
    """What a reader of the composite should know about its channels."""
    notes = []
    with np.errstate(invalid='ignore'):
        unusable = ~(np.isfinite(values) & (values >= 0.0))
    for label, bad in zip(labels, unusable, strict=True):
        if bad.all():
            notes.append(f'{label} has no finite, non-negative line, so no '
                         'composite was formed')
        elif bad.any():
            notes.append(f'{label} has {int(bad.sum())} line'
                         f'{"s" * (int(bad.sum()) != 1)} that are not a '
                         'finite, non-negative density; the composite is '
                         'not formed or judged there')
    if share is not None:
        for label, weight in zip(labels, share, strict=True):
            if weight == 0.0:
                notes.append(f'{label} has weight zero and is not in the '
                             'composite; it is still judged against its own '
                             'band')
    # a channel far below the rest at its median line is a sensor not
    # reporting the motion: zero everywhere, or a few lines of noise
    good = np.isfinite(formed)
    if good.sum() and values.shape[0] > 1:
        for k, label in enumerate(labels):
            mine = values[k, good]
            if not (mine > 0.0).any():
                notes.append(f'{label} reads zero at every line, as a dead '
                             'or disconnected sensor does; the composite '
                             'counts it as it reads')
                continue
            others = np.median(np.delete(values[:, good], k, axis=0), axis=0)
            usable = (mine > 0.0) & (others > 0.0)
            if not usable.any():
                continue
            below = float(np.median(10.0 * np.log10(mine[usable]
                                                    / others[usable])))
            if below < -DEAD_DB:
                notes.append(
                    f'{label} reads {-below:.0f} dB below the other channels '
                    'at its median line, as a dead or disconnected sensor '
                    'does; the composite counts it as it reads')
    return notes
