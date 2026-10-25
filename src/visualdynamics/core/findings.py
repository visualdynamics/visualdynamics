"""Findings: deviations, anomalies and nonconformances, each in four parts.

A test report writes every departure from the plan (a deviation), every
unexpected observation (an anomaly) and every requirement not met (a
nonconformance) in the same four parts, in this order: the **fact**
(what happened, on which run, channel and band, by how much, and the
record that shows it), the **cause** as far as it is known, the
**effect** on what the test proves, and the **approval** of a deviation
or the **disposition** of an anomaly or nonconformance, with who and
under which authority. Each carries a number of its kind (D-1, A-1,
NC-1) that the summary, the result table and the record refer to.

This module holds the rules a finding is checked against, wherever it
is entered: the kinds and their numbers (`KINDS`, `number`,
`next_number`), what a finding holds (`clean`, which refuses what
cannot be read one way), which of its parts are still owed (`owed`),
the sentences by which an author says a kind has none
(`NONE_SENTENCES`, `declared`), and the one part a program can offer:
a fact read off a measurement (`excursions`), the band, side and size
by which a channel left one pair of limits.

Only the fact is ever offered, and only as a draft the author accepts
or edits. Cause, effect and approval are the author's words, and so is
"none": an empty register is not a statement that nothing happened.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray, Specification

#: the kinds of finding, in the order a report lists them
KINDS = ('deviation', 'anomaly', 'nonconformance')

#: each kind's number prefix: D-1, A-1, NC-1
PREFIX = {'deviation': 'D', 'anomaly': 'A', 'nonconformance': 'NC'}

#: each kind named in the plural, for a summary line ("Deviations: D-1")
PLURAL = {'deviation': 'Deviations', 'anomaly': 'Anomalies',
          'nonconformance': 'Nonconformances'}

#: what the fourth part is called: a deviation is approved before it is
#: run; an anomaly or a nonconformance is disposed of after it is found
FOURTH = {'deviation': 'Approval', 'anomaly': 'Disposition',
          'nonconformance': 'Disposition'}

#: the four parts, in the order they are written
PARTS = ('fact', 'cause', 'effect', 'approval')

#: the sentence that says a kind has none. Together, in order, they are
#: the deviations section's one-step None
#: (`report.skeleton.SECTIONS`), so the two cannot say different things.
NONE_SENTENCES = {'deviation': 'No deviations.',
                  'anomaly': 'No anomalies.',
                  'nonconformance': 'No nonconformances.'}

#: what a finding's status may be: not stated, open or closed
STATUSES = ('', 'open', 'closed')

#: the limit pairs a fact can be read against
PAIRS = ('warning', 'abort')

_NUMBER = re.compile(r'^(D|A|NC)-([1-9][0-9]*)$')
_KIND_OF = {prefix: kind for kind, prefix in PREFIX.items()}


def kind_of(value: Any) -> str:
    """`value` as one of `KINDS`; `ValueError` naming the kinds for
    anything else, `TypeError` for a value that is not a string."""
    if not isinstance(value, str):
        raise TypeError(f'a finding\'s kind is a str, not '
                        f'{type(value).__name__}')
    if value not in KINDS:
        raise ValueError(f'unknown kind of finding {value!r}; the kinds '
                         f'are: {", ".join(KINDS)}')
    return value


def number(text: Any) -> tuple[str, int]:
    """(kind, n) for a finding's number such as 'D-2' or 'NC-1'.

    Refused with `ValueError`: anything not a kind's prefix, a hyphen
    and a whole number from 1 ('D-0', 'd-1', 'D1', 'D-01'). `TypeError`
    for a value that is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f'a finding\'s number is a str, not '
                        f'{type(text).__name__}')
    found = _NUMBER.match(text)
    if found is None:
        raise ValueError(f'{text!r} is not a finding number: D-1, A-1 or '
                         'NC-1, the kind\'s letters, a hyphen and a whole '
                         'number from 1')
    return _KIND_OF[found.group(1)], int(found.group(2))


def next_number(entries: Iterable[Mapping[str, Any]], kind: str,
                issued: Mapping[str, int] | None = None) -> str:
    """The number the next finding of `kind` takes: one past the
    highest of that kind entered, or ever issued (`issued`, the highest
    number by prefix, as `issued_of` reads it), so a number once given
    is never given again to another finding, even after the first is
    removed."""
    kind_of(kind)
    taken = [number(entry['number'])[1] for entry in entries
             if number(entry['number'])[0] == kind]
    highest = max([*taken, int((issued or {}).get(PREFIX[kind], 0))])
    return f'{PREFIX[kind]}-{highest + 1}'


def issued_of(value: Any) -> dict[str, int]:
    """The highest number issued of each kind, by prefix ({'D': 3}), as
    a register keeps it; empty for None. Refuses anything else with
    `ValueError`."""
    if value is None:
        return {}
    if (not isinstance(value, Mapping)
            or not all(key in _KIND_OF for key in value)
            or not all(isinstance(n, int) and not isinstance(n, bool)
                       and n >= 1 for n in value.values())):
        raise ValueError(f'the numbers issued are a whole number from 1 by '
                         f'prefix (D, A, NC), not {value!r}')
    return dict(value)


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f'{what} is a str, not {type(value).__name__}')
    return value.strip()


def band_of(value: Any, what: str = 'band') -> list[float] | None:
    """A band as [low, high] in Hz, or None for none.

    Refused with `ValueError`: a count other than two, a number that is
    not finite, a negative frequency, or a low edge above the high one;
    with `TypeError`, anything but a sequence of numbers. Equal edges
    are one frequency.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f'{what} is two frequencies in Hz, low and high, '
                        f'not {value!r}')
    if len(value) != 2:
        raise ValueError(f'{what} is two frequencies in Hz, low and high; '
                         f'{len(value)} were given')
    edges = []
    for edge in value:
        if isinstance(edge, bool) or not isinstance(edge, (int, float,
                                                           np.number)):
            raise TypeError(f'{what}: {edge!r} is not a frequency')
        edge = float(edge)
        if not math.isfinite(edge):
            raise ValueError(f'{what}: {edge} Hz is not a frequency')
        if edge < 0.0:
            raise ValueError(f'{what}: {edge:g} Hz is negative')
        edges.append(edge)
    if edges[0] > edges[1]:
        raise ValueError(f'{what}: the low edge {edges[0]:g} Hz is above '
                         f'the high edge {edges[1]:g} Hz')
    return edges


def clean_draft(value: Any) -> dict[str, Any] | None:
    """A fact draft as stored on a finding: 'text' (required, not
    blank), 'source', 'run', 'channel' and 'band'. None for none.
    `TypeError` for anything but a mapping, `ValueError` for a field
    it does not hold or blank text."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f'a fact draft is a mapping, not {value!r}')
    unknown = sorted(set(value) - {'text', 'source', 'run', 'channel',
                                   'band'})
    if unknown:
        raise ValueError(f'a fact draft holds text, source, run, channel '
                         f'and band, not {", ".join(unknown)}')
    text = _text(value.get('text', ''), "a fact draft's text")
    if not text:
        raise ValueError('a fact draft with no text offers nothing')
    return {'text': text,
            'source': _text(value.get('source', ''), "a draft's source"),
            'run': _text(value.get('run', ''), "a draft's run"),
            'channel': _text(value.get('channel', ''), "a draft's channel"),
            'band': band_of(value.get('band'), "a draft's band")}


#: the fields a finding holds, other than its number and kind
FIELDS = ('run', 'channel', 'band', *PARTS, 'status', 'draft')


def clean(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A finding as it is stored, every field present and checked.

    `entry` holds 'number' (from which the kind is read) and any of
    'kind' (which must agree with the number), 'run', 'channel', 'band',
    'fact', 'cause', 'effect', 'approval', 'status' and 'draft'. Text is
    stripped; a missing field is empty.

    Raises
    ------
    ValueError
        For a malformed number, a kind that disagrees with it, a band
        that is not two ordered, finite, non-negative frequencies, a
        status outside `STATUSES`, a malformed draft, or a field this
        module does not know. The message names the finding.
    TypeError
        When a text field is not a string, or a band or draft is not of
        its type.
    """
    if not isinstance(entry, Mapping):
        raise TypeError(f'a finding is a mapping, not '
                        f'{type(entry).__name__}')
    kind, _n = number(entry.get('number'))
    label = entry['number']
    unknown = sorted(set(entry) - {'number', 'kind', *FIELDS})
    if unknown:
        raise ValueError(f'{label}: a finding holds no '
                         f'{", ".join(unknown)}')
    if 'kind' in entry and entry['kind'] != kind:
        raise ValueError(f'{label} is a {kind} by its number, not a '
                         f'{entry["kind"]}')
    out: dict[str, Any] = {'number': label, 'kind': kind}
    for field in ('run', 'channel', *PARTS):
        out[field] = _text(entry.get(field, ''), f'{label}: {field}')
    out['band'] = band_of(entry.get('band'), f'{label}: band')
    status = _text(entry.get('status', ''), f'{label}: status')
    if status not in STATUSES:
        raise ValueError(f'{label}: status {status!r} is not open or closed')
    out['status'] = status
    out['draft'] = clean_draft(entry.get('draft'))
    return out


def read(entries: Any) -> list[dict[str, Any]]:
    """A register as stored (a list of findings), each `clean`, in the
    order a report lists them: by kind, then by number. Refuses a
    number given twice (`ValueError`) and anything that is not a list
    (`TypeError`)."""
    if entries is None:
        return []
    if isinstance(entries, (str, bytes, Mapping)) or not isinstance(
            entries, Sequence):
        raise TypeError(f'a findings register is a list of findings, not '
                        f'{type(entries).__name__}')
    out = [clean(entry) for entry in entries]
    seen: set[str] = set()
    for entry in out:
        if entry['number'] in seen:
            raise ValueError(f'{entry["number"]} is given to two findings')
        seen.add(entry['number'])
    return sorted(out, key=lambda entry: (KINDS.index(entry['kind']),
                                          number(entry['number'])[1]))


def owed(entry: Mapping[str, Any]) -> list[str]:
    """The parts of a finding still owed, in order: each of fact, cause,
    effect and approval that is blank. A part is never filled for the
    author; "not determined" is a cause, and the author writes it."""
    return [part for part in PARTS if not str(entry.get(part, '')).strip()]


def declared(text: str) -> set[str]:
    """The kinds an author's words say there are none of: each kind
    whose `NONE_SENTENCES` sentence the words hold."""
    return {kind for kind, sentence in NONE_SENTENCES.items()
            if sentence in (text or '')}


def withdraw(text: str, kinds: Iterable[str]) -> str:
    """`text` with the none sentence of each kind in `kinds` taken out,
    and the white space it leaves tidied."""
    out = text or ''
    for kind in kinds:
        out = out.replace(NONE_SENTENCES[kind_of(kind)], '')
    lines = [re.sub(r'[ \t]{2,}', ' ', line).strip()
             for line in out.splitlines()]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def declaration(kinds: Iterable[str]) -> str:
    """The none sentences of `kinds`, in the order of `KINDS`."""
    wanted = {kind_of(kind) for kind in kinds}
    return ' '.join(NONE_SENTENCES[kind] for kind in KINDS if kind in wanted)


# ---- a fact read off a measurement ---------------------------------------


@dataclass(frozen=True)
class Excursion:
    """One stretch of a channel outside one limit.

    `side` is 'above' (over the upper limit) or 'below' (under the
    lower). `low` and `high` are the first and last measured lines in
    the stretch, in Hz; `lines` how many lines it holds. `worst_db` is
    how far beyond the limit it went at most, as a ratio of density in
    dB (ten log ten), positive; infinite where a line below a lower
    limit holds no power at all.
    """

    side: str
    low: float
    high: float
    lines: int
    worst_db: float


@dataclass(frozen=True)
class ExcursionReading:
    """What `excursions` read: the stretches outside the limits, in
    frequency order; how many cells were judged and over what range
    (`judged`, `low`, `high`); how many measured lines with no value
    (NaN) were passed over where the specification is written and in
    the band asked (`unjudged`); and the scale the measurement was
    compared at (`scale_db`)."""

    excursions: tuple[Excursion, ...]
    judged: int
    low: float
    high: float
    unjudged: int
    scale_db: float


def pair_records(specification: Specification, measured: DataArray,
                 channel: str) -> tuple[int, int]:
    """(specification record, measured record) for `channel`, paired the
    way every comparison pairs them (`compliance.matched_records`, by
    DOF and quantity).

    Refused with `ValueError`: a channel the two do not have in common
    (naming the channels they do), and a label that names two pairs (a
    DOF compared as two quantities), naming the quantities.
    """
    from .compliance import matched_records

    if not isinstance(channel, str) or not channel.strip():
        raise ValueError('name the channel to read, such as 2Y+')
    rows = matched_records(specification, measured)
    mine = [(spec, held) for label, spec, held in rows if label == channel]
    if not mine:
        common = ', '.join(label for label, _s, _m in rows) or 'none'
        raise ValueError(f'{channel} is not a channel the specification and '
                         f'the measurement have in common; they share: '
                         f'{common}')
    if len(mine) > 1:
        quantities = ', '.join(measured.ordinate_dim[held]
                               for _s, held in mine)
        raise ValueError(f'{channel} is compared as {len(mine)} quantities '
                         f'({quantities}); one finding reads one record')
    return mine[0]


def _missing(specification: Any, measured: Any, spec_record: int,
             measured_record: int, band: Sequence[float] | None) -> int:
    """How many measured lines with no value (NaN or infinite) fall
    where the specification is written and inside `band`: lines the
    comparison passes over as gaps, which a reading has to say. Read
    after a cell was judged, so the specification is written somewhere."""
    from .compliance import band_of as written_over

    reach = written_over(specification, spec_record)
    lines = np.asarray(measured.abscissa, dtype=float)
    values = np.asarray(np.real(measured.ordinate[measured_record]),
                        dtype=float)
    low, high = reach if band is None else (max(reach[0], band[0]),
                                            min(reach[1], band[1]))
    return int(np.sum(~np.isfinite(values) & (lines >= low)
                      & (lines <= high)))


def excursions(specification: Specification, measured: DataArray,
               spec_record: int = 0, measured_record: int = 0, *,
               pair: str = 'abort',
               band: Sequence[float] | None = None,
               scale_db: float | None = None) -> ExcursionReading:
    """Where one measured record went outside one pair of limits.

    Every cell of the comparison (`compliance.judge`, the same cells
    and the same scale the comparison figure and its readouts use) is
    judged against the pair's upper and lower limit. Neighboring cells
    out on the same side make one stretch, reported by its first and
    last measured line, its line count and its worst size: the cell's
    density against the limit's, in dB. With `band` (low, high in Hz),
    only cells reaching into it are read.

    Parameters
    ----------
    specification, measured : Specification, DataArray
        The pair, as a comparison figure draws it.
    spec_record, measured_record : int
        Which record of each (`pair_records` finds them by channel).
    pair : str, default 'abort'
        'abort' or 'warning'.
    band : (float, float), optional
        The frequencies to read, in Hz.
    scale_db : float, optional
        The comparison's scale; resolved for the whole measurement
        (`compliance.comparison_scale_db`) when omitted, as
        `compliance.compare_all` resolves it.

    Returns
    -------
    ExcursionReading
        The stretches, possibly none. No stretch is not a statement
        that the channel was inside its limits anywhere it was not
        judged: `judged`, `low`, `high` and `unjudged` say what was.

    Raises
    ------
    ValueError
        A pair other than warning or abort, or one the specification
        does not carry; a pair the comparison refuses (different
        quantities, units declared on one side only, a banded
        measurement against a breakpoint curve), with its reason; a
        band that is not two ordered frequencies; nothing judged in the
        band (no data, no limit, or a band outside both); and a
        channel that holds no power where it was judged.
    """
    from .compliance import compare, comparison_scale_db, judge

    if pair not in PAIRS:
        raise ValueError(f'limits are read as warning or abort, not {pair!r}')
    edges = [(f'{pair}_{edge}', over) for edge, over
             in (('upper', True), ('lower', False))
             if specification.limits.get(f'{pair}_{edge}') is not None]
    if not edges:
        raise ValueError(f'the specification carries no {pair} limits')
    band = band_of(band)
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)
    refused = compare(specification, measured, spec_record, measured_record,
                      scale_db=scale_db).get('refused')
    if refused:
        raise ValueError(refused)
    sides = {}
    for edge, over in edges:
        sides[edge] = judge(specification, measured, spec_record,
                            measured_record, edge, over, scale_db)
    first = next(iter(sides.values()))
    cells = first['cells']
    lows = np.array([cell['low'] for cell in cells], dtype=float)
    highs = np.array([cell['high'] for cell in cells], dtype=float)
    if band is None:
        inside = np.ones(len(cells), dtype=bool)
    else:
        # a cell reaching into the band, by any width; for a band of one
        # frequency, the cell holding it
        inside = (highs > band[0]) & (lows < band[1]) if band[0] < band[1] \
            else (lows <= band[0]) & (highs >= band[0])
    judged = inside & np.logical_and.reduce(
        [result['judged'] for result in sides.values()])
    where = (f' from {band[0]:g} to {band[1]:g} Hz' if band is not None
             else '')
    if not judged.any():
        raise ValueError(f'nothing judged{where}: no cell there has both '
                         'data and a limit')
    held = first['held']
    if not np.any(held[judged] > 0.0):
        raise ValueError(f'the channel holds no power{where} (an all-zero '
                         'record); there is no excursion to read')
    lines = np.asarray(measured.abscissa, dtype=float)
    found: list[Excursion] = []
    out_side = np.full(len(cells), '', dtype=object)
    size = np.full(len(cells), np.nan)
    for edge, over in edges:
        result = sides[edge]
        side = 'above' if over else 'below'
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (result['held'] / result['asked'] if over
                     else result['asked'] / result['held'])
            db = 10.0 * np.log10(ratio)
        mark = result['out'] & judged
        out_side[mark] = side
        size[mark] = db[mark]
    k = 0
    while k < len(cells):
        if not out_side[k]:
            k += 1
            continue
        start = k
        while (k + 1 < len(cells) and out_side[k + 1] == out_side[start]
               and lows[k + 1] <= highs[k] * (1.0 + 1e-12)):
            k += 1
        reach = (lines >= lows[start] * (1.0 - 1e-12)) \
            & (lines <= highs[k] * (1.0 + 1e-12))
        centers = lines[reach]
        found.append(Excursion(
            side=str(out_side[start]),
            low=float(centers[0] if centers.size else lows[start]),
            high=float(centers[-1] if centers.size else highs[k]),
            lines=int(centers.size),
            worst_db=float(np.max(size[start:k + 1]))))
        k += 1
    return ExcursionReading(
        excursions=tuple(found), judged=int(judged.sum()),
        low=float(lows[judged][0]), high=float(highs[judged][-1]),
        unjudged=_missing(specification, measured, spec_record,
                          measured_record, band),
        scale_db=float(scale_db))
