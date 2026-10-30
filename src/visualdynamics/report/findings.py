"""Findings register: deviations, anomalies and nonconformances in four
parts (report feature `findings`).

With the feature on, the author enters each finding in a register:
its number and kind (D-1, A-1, NC-1), the run, channel and band it
concerns, and its four parts, fact, cause, effect, and approval or
disposition (`visualdynamics.core.findings`). The page draws the
register in section 9 of the skeleton, each finding as its four
labeled parts, and lists the findings by number under the Summary. A
part left blank is drawn as owed, and the export's check lists it; it
is never filled in and never left out.

**The register is the author's section.** The skeleton keeps the
deviations section as the author's to write, and the register lives
in that section's own block (the text block carrying `section:
'deviations'`, under `'findings'`), written only by what the author
does: entering a finding, editing it, accepting a draft. Nothing is
generated into it.

- The **fact** may be offered as a draft, read off a measurement the
  project holds (`draft_from_comparison`: the channel, the band, the
  side and size of the excursion, the record). A draft is held beside
  the finding, and its words are not drawn; the author accepts it,
  edits it or discards it (`accept`, `discard`), and once accepted it
  is the author's fact. Until then the fact stays owed, on the page
  and in the check, marked as having a drafted fact waiting. Every
  source of a draft feeds it through one function, `offer`.
- A **run log** brought into the project offers each deviation and
  anomaly it holds as a draft (`offer_log`): its kind, its fact as
  typed at the console and the log's number, proposed, citing the log,
  test and entry. The drafts are kept beside the register, never in
  it: the register and section 9 do not change, the section stays
  owed, and the editing page shows the drafts marked as drafted from
  the run log. Accepting one (`accept_log`) is what enters the finding,
  under the log's number when it is free, with its fact through
  `offer`; its cause, effect and approval or disposition stay owed. A
  fact the author has written is never offered over, and the log's
  approvals and dispositions are never offered. A log with no deviation
  or anomaly offers nothing, never "none".
- **Cause, effect and approval** are only ever typed.
- **None** is the skeleton's one-step None (`none`), and it says only
  what the author asks it to: with nothing entered it writes the
  section's own None wording, exactly as the skeleton does; with
  findings entered, it writes the "none" sentence of each kind that
  has none, such as "No anomalies.". An empty register is never read
  as "none": until the author says so, a kind with nothing entered is
  owed.

The feature needs Skeleton sections, whose sections it fills.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core import findings as core
from . import skeleton

#: the feature's key in `report.features`
FEATURE = 'findings'

#: the skeleton section the register fills
SECTION = 'deviations'

#: the fields an author enters on a finding
EDITABLE = ('run', 'channel', 'band', *core.PARTS, 'status')

#: what a blank fact says while a draft of it waits: the draft is
#: offered, marked as drafted, and fills nothing until the author
#: accepts or edits it
DRAFTED = 'a drafted fact waits for the author to accept or edit'

#: the most stretches on one side a fact draft lists one by one; past
#: this, the side is said once, with its count, its reach and its worst
#: stretch
LISTED = 3


@dataclass(frozen=True)
class FactDraft:
    """A fact offered for the author to accept, edit or discard.

    `text` is the draft. `source` says where it came from, shown beside
    it in the editor. `run`, `channel` and `band` are what the draft
    concerns; on acceptance they fill the finding's own fields where
    the author has left them blank.
    """

    text: str
    source: str
    run: str = ''
    channel: str = ''
    band: tuple[float, float] | None = field(default=None)


def _require(report: Any) -> None:
    from .features import enabled

    if FEATURE not in enabled(report):
        raise ValueError(f'report {getattr(report, "title", "")!r} does not '
                         'have the Findings register enabled')


def _block(report: Any) -> dict[str, Any] | None:
    held = skeleton.blocks_of(report, SECTION)
    return report.blocks[held[0]] if held else None


def register(report: Any) -> list[dict[str, Any]]:
    """The findings entered on `report`, each checked
    (`core.findings.clean`), by kind and then number. Empty when none
    are."""
    block = _block(report)
    return core.read(block.get('findings') if block is not None else None)


#: where the section's block keeps the highest number issued of each
#: kind, so a removed finding's number is not given again
ISSUED = 'findings_issued'


def _issued(report: Any) -> dict[str, int]:
    block = _block(report)
    return core.issued_of(block.get(ISSUED) if block is not None else None)


def _store(report: Any, entries: Sequence[Mapping[str, Any]]) -> None:
    block = report.blocks[skeleton.write(report, SECTION)]
    entries = core.read(entries)
    issued = _issued(report)
    for entry in entries:
        kind, n = core.number(entry['number'])
        issued[core.PREFIX[kind]] = max(issued.get(core.PREFIX[kind], 0), n)
    if issued:
        block[ISSUED] = issued
    if entries:
        block['findings'] = [dict(entry) for entry in entries]
    else:
        block.pop('findings', None)


def find(report: Any, number: str) -> dict[str, Any]:
    """The finding numbered `number`; `ValueError` naming the numbers
    entered when there is none."""
    core.number(number)
    for entry in register(report):
        if entry['number'] == number:
            return entry
    taken = ', '.join(entry['number'] for entry in register(report))
    raise ValueError(f'no finding {number}; entered: {taken or "none"}')


def _fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """The fields given, checked by name: None leaves a field alone, and
    an empty band clears the band."""
    unknown = sorted(set(fields) - set(EDITABLE))
    if unknown:
        raise ValueError(f'a finding is entered with {", ".join(EDITABLE)}, '
                         f'not {", ".join(unknown)}')
    out = {}
    for key, value in fields.items():
        if value is None:
            continue
        if (key == 'band' and isinstance(value, Sequence)
                and not isinstance(value, (str, bytes)) and not len(value)):
            value = None
        out[key] = value
    return out


def _words(report: Any) -> str:
    block = _block(report)
    return str(block.get('text', '')) if block is not None else ''


def add(report: Any, kind: str, *, number: str | None = None,
        **fields: Any) -> str:
    """Enter a finding of `kind` and return its number.

    The number is the next of its kind (`core.findings.next_number`)
    unless `number` is given, which must be of that kind and not yet
    taken. `fields` are any of `EDITABLE`. Entering a finding of a
    kind the author had said there were none of takes that kind's
    none sentence out of the section's words: the finding is the later
    statement.

    Raises
    ------
    ValueError
        When the report does not have the feature enabled, for an
        unknown kind, a number of another kind or already taken, an
        unknown field, or a field `core.findings.clean` refuses.
        Nothing is changed.
    TypeError
        When a text field is not a string.
    """
    _require(report)
    core.kind_of(kind)
    entries = register(report)
    if number is None:
        number = core.next_number(entries, kind, _issued(report))
    else:
        given, _n = core.number(number)
        if given != kind:
            raise ValueError(f'{number} is a {given} number, not a {kind}')
        if any(entry['number'] == number for entry in entries):
            raise ValueError(f'{number} is already entered')
        if _n <= _issued(report).get(core.PREFIX[kind], 0):
            raise ValueError(f'{number} was issued before and removed; a '
                             'number is not given to a second finding')
    entry = core.clean({'number': number, 'kind': kind, **_fields(fields)})
    words = _words(report)
    _store(report, [*entries, entry])
    if kind in core.declared(words):
        skeleton.write(report, SECTION, core.withdraw(words, [kind]))
    return number


def edit(report: Any, number: str, **fields: Any) -> None:
    """Change the fields given of finding `number` (any of `EDITABLE`;
    a field given as None is left alone, and an empty band clears the
    band). Refused as `add` refuses, before anything changes; a
    finding's kind is its number's and is not edited (enter the
    finding again under the other kind, as a report does when an
    anomaly is found to be a nonconformance)."""
    _require(report)
    entry = {**find(report, number), **_fields(fields)}
    entry = core.clean(entry)
    _store(report, [entry if other['number'] == number else other
                    for other in register(report)])


def remove(report: Any, number: str) -> None:
    """Take finding `number` out of the register. Its number is not
    given again (`core.findings.next_number`)."""
    _require(report)
    find(report, number)
    _store(report, [entry for entry in register(report)
                    if entry['number'] != number])


def none(report: Any) -> str:
    """The deviations section's one-step None, and the words it wrote.

    With nothing entered, it is the skeleton's None exactly: the
    section says "No deviations. No anomalies. No nonconformances.".
    With findings entered, it says "none" only of each kind that has
    nothing entered, keeping the rest of the section's words.

    Raises `ValueError` when the feature is off, and when every kind
    has a finding entered, so there is no kind to say "none" of.
    """
    _require(report)
    entries = register(report)
    if not entries:
        index = skeleton.write(report, SECTION, none=True)
        return str(report.blocks[index]['text'])
    entered = {entry['kind'] for entry in entries}
    empty = [kind for kind in core.KINDS if kind not in entered]
    if not empty:
        raise ValueError(f'{skeleton.BY_KEY[SECTION].heading}: every kind has '
                         'a finding entered, so there is none to set to None')
    kept = core.withdraw(_words(report), core.KINDS)
    text = (f'{kept}\n\n' if kept else '') + core.declaration(empty)
    skeleton.write(report, SECTION, text)
    return text


# ---- drafts of the fact ----------------------------------------------------


def offer(report: Any, number: str, draft: FactDraft) -> None:
    """Offer `draft` as the fact of finding `number`, for the author to
    accept, edit or discard; it replaces any draft already offered.

    This is the one way a fact draft reaches a finding, whatever made
    it: `draft_from_comparison` reads one off a measurement, and any
    later source (a run log's entry, say) builds a `FactDraft` and
    offers it here. A draft is not drawn on the page and fills nothing
    until it is accepted.
    """
    _require(report)
    if not isinstance(draft, FactDraft):
        raise TypeError(f'a fact draft is a FactDraft, not '
                        f'{type(draft).__name__}')
    stored = asdict(draft)
    stored['band'] = list(draft.band) if draft.band is not None else None
    entry = core.clean({**find(report, number), 'draft': stored})
    _store(report, [entry if other['number'] == number else other
                    for other in register(report)])


def accept(report: Any, number: str, text: str | None = None) -> None:
    """Make the draft offered on finding `number` its fact: as offered,
    or as the author edited it (`text`). The draft's run, channel and
    band fill the finding's own where the author left them blank, and
    never replace what the author entered. Refused when no draft is
    waiting, or when the text accepted is blank."""
    _require(report)
    entry = find(report, number)
    draft = entry['draft']
    if draft is None:
        raise ValueError(f'{number} has no fact draft waiting to be accepted')
    fact = draft['text'] if text is None else core.clean(
        {'number': number, 'fact': text})['fact']
    if not fact:
        raise ValueError(f'{number}: a blank fact is not accepted; discard '
                         'the draft instead')
    entry = {**entry, 'fact': fact, 'draft': None}
    for key in ('run', 'channel', 'band'):
        if not entry[key] and draft[key]:
            entry[key] = draft[key]
    _store(report, [core.clean(entry) if other['number'] == number else other
                    for other in register(report)])


def discard(report: Any, number: str) -> None:
    """Drop the draft offered on finding `number`; refused when none is
    waiting."""
    _require(report)
    entry = find(report, number)
    if entry['draft'] is None:
        raise ValueError(f'{number} has no fact draft waiting to be discarded')
    _store(report, [{**entry, 'draft': None} if other['number'] == number
                    else other for other in register(report)])


#: what findings offered from a run log are marked with on the editing
#: page until the author accepts them
FROM_LOG = 'DRAFTED FROM THE RUN LOG, NOT YET ACCEPTED'

#: where the deviations section's block keeps the findings a run log
#: offers, apart from the register: nothing here is a finding until the
#: author accepts it
LOG_DRAFTS = 'log_drafts'


def log_drafts(report: Any) -> list[dict[str, Any]]:
    """The findings a run log offers `report` and the author has not
    accepted or discarded, each {'number', 'kind', 'fact', 'source',
    'run'}, in the order offered. The number is the log's, proposed;
    none of them is in the register."""
    block = _block(report)
    held = block.get(LOG_DRAFTS) if block is not None else None
    return [dict(draft) for draft in held or ()]


def _keep_drafts(report: Any, drafts: Sequence[Mapping[str, Any]]) -> None:
    held = skeleton.blocks_of(report, SECTION)
    if not held and not drafts:
        return
    block = report.blocks[held[0] if held else skeleton.write(report,
                                                              SECTION)]
    if drafts:
        block[LOG_DRAFTS] = [dict(draft) for draft in drafts]
    else:
        block.pop(LOG_DRAFTS, None)


def offer_log(report: Any, extract: Any) -> list[str]:
    """Offer the deviations and anomalies a run log extract holds as
    drafts, and return a line on each.

    Each is offered with the log's number, proposed, its kind and its
    fact as typed at the console, its source citing the log, test and
    entry (with the extract's badge when the log has newer entries). A
    draft is kept beside the register, never in it: the register and
    the deviations section do not change, the section stays owed, and
    the editing page shows the drafts marked `FROM_LOG` until the author
    accepts one (`accept_log`, which enters the finding) or discards it
    (`discard_log`). A new offer replaces the drafts waiting. Not
    offered, each with its line: a finding whose number the register
    holds with its fact written, and a finding with no fact logged. The
    cause, the effect and the approval or disposition are never offered,
    whatever the log holds.

    A log holding no deviation or anomaly offers nothing; its line says
    so, and leaves setting the section to None (`none`) to the author.

    Raises `ValueError` when the feature is off.
    """
    _require(report)
    found = extract.findings()
    if not found:
        _keep_drafts(report, [])
        return [(f'The run log holds no deviation or anomaly of test '
                 f'{extract.test} by entry #{extract.upto}; nothing is '
                 "offered, and whether there were none is the author's to "
                 'say (set the section to None).')]
    cited = extract.cited()
    written = {entry['number'] for entry in register(report) if entry['fact']}
    drafts, lines = [], []
    for finding in found:
        number = finding.number
        fact = (finding.fact or '').strip()
        if number in written:
            lines.append(f"{number}: the register holds it with its fact "
                         "written; the log's is not offered over it.")
            continue
        if not fact:
            lines.append(f'{number}: the log holds no fact for it, so '
                         'nothing is offered; enter it in the register.')
            continue
        run = finding.entry.run
        drafts.append({'number': number, 'kind': finding.kind, 'fact': fact,
                       'source': f'{cited}, entry #{finding.entry.id}',
                       'run': str(run) if run is not None else ''})
        lines.append(f'{number}: offered as a draft from the run log, not '
                     'in the register until accepted.')
    _keep_drafts(report, drafts)
    return lines


def _log_draft(report: Any, number: str) -> dict[str, Any]:
    for draft in log_drafts(report):
        if draft['number'] == number:
            return draft
    waiting = ', '.join(draft['number'] for draft in log_drafts(report))
    raise ValueError(f'no finding {number} offered from the run log; '
                     f'waiting: {waiting or "none"}')


def accept_log(report: Any, number: str, text: str | None = None) -> str:
    """Accept the finding the run log offered as `number`, and return the
    number it is entered under.

    The finding is entered in the register then, and not before: under
    the log's number when the register has not given that number, and
    otherwise under the next number of its kind (`add`); a finding the
    register already holds under that number with its fact blank takes
    the fact instead. Its fact, as offered or as the author edited it
    (`text`), reaches it through the register's one entry point for
    drafts (`offer`) and is accepted (`accept`); its run fills the
    finding's where blank. Its cause, effect and approval or disposition
    stay owed. The draft is no longer offered afterwards.

    Raises `ValueError` when the feature is off, no draft is offered as
    `number`, or the fact accepted is blank; nothing is changed.
    """
    _require(report)
    draft = _log_draft(report, number)
    kind = draft['kind']
    if text is not None and not core.clean({'number': number,
                                            'fact': text})['fact']:
        raise ValueError(f'{number}: a blank fact is not accepted; discard '
                         'the draft instead')
    held = next((entry for entry in register(report)
                 if entry['number'] == number), None)
    _n = core.number(number)[1]
    if held is not None and not held['fact']:
        entered = number
    elif held is None and _n > _issued(report).get(core.PREFIX[kind], 0):
        entered = add(report, kind, number=number)
    else:
        entered = add(report, kind)
    offer(report, entered, FactDraft(text=draft['fact'],
                                     source=draft['source'],
                                     run=draft['run']))
    accept(report, entered, text)
    _keep_drafts(report, [other for other in log_drafts(report)
                          if other['number'] != number])
    return entered


def discard_log(report: Any, number: str) -> None:
    """Drop the finding the run log offered as `number`; nothing enters
    the register. Refused when no draft is offered as `number`."""
    _require(report)
    _log_draft(report, number)
    _keep_drafts(report, [other for other in log_drafts(report)
                          if other['number'] != number])


def _hz(value: float) -> str:
    """A frequency with as few decimals as say it (one at least)."""
    for places in range(1, 5):
        if abs(round(value, places) - value) <= 1e-6 * max(1.0, abs(value)):
            return f'{value:.{places}f}'
    return f'{value:.4f}'


def _lines(count: int) -> str:
    return f'{count} line{"s" * (count != 1)}'


def _size(db: float) -> str:
    return (f'by up to {db:.2f} dB' if math.isfinite(db)
            else 'with no power on its lowest line')


def _reach(low: float, high: float) -> str:
    return (f'at {_hz(low)} Hz' if low == high
            else f'from {_hz(low)} to {_hz(high)} Hz')


def _stretch(excursion: core.Excursion, pair: str) -> str:
    return (f'{excursion.side} its {pair} limit '
            f'{_reach(excursion.low, excursion.high)} '
            f'{_size(excursion.worst_db)} ({_lines(excursion.lines)})')


def fact_text(channel: str, reading: core.ExcursionReading, pair: str,
              record: str, band: Sequence[float] | None = None) -> str:
    """The fact a reading supports, in words: the channel, each stretch
    above the limit and then each below (a side with more than `LISTED`
    said once, with its count, its reach and its worst stretch), the
    record, and what was not judged. Refused when the reading holds no stretch: a
    reading inside its limits is not a fact about a finding."""
    found = reading.excursions
    if not found:
        raise ValueError(f'{channel} is inside its {pair} limits wherever it '
                         f'was judged, {_hz(reading.low)} to '
                         f'{_hz(reading.high)} Hz; there is no excursion to '
                         'draft a fact from')
    parts = []
    for side in ('above', 'below'):
        mine = [excursion for excursion in found if excursion.side == side]
        if len(mine) <= LISTED:
            parts.extend(_stretch(excursion, pair) for excursion in mine)
            continue
        worst = max(mine, key=lambda excursion: excursion.worst_db)
        parts.append(
            f'{side} its {pair} limit in {len(mine)} stretches from '
            f'{_hz(mine[0].low)} to {_hz(mine[-1].high)} Hz, '
            f'{_lines(sum(e.lines for e in mine))} in all, '
            f'{_size(worst.worst_db)} ({_reach(worst.low, worst.high)})')
    text = f'{channel} ' + '; '.join(parts) + f'; {record}'
    if reading.scale_db:
        text += f'; compared at {reading.scale_db:+g} dB'
    if band is not None and (reading.low > band[0] or reading.high < band[1]):
        text += (f'; judged from {_hz(reading.low)} to {_hz(reading.high)} Hz '
                 f'of the {_hz(band[0])} to {_hz(band[1])} Hz asked')
    if reading.unjudged:
        text += (f'; {_lines(reading.unjudged)} with no value, not '
                 'judged')
    return text + '.'


def fact_from_comparison(objects: Mapping[str, Any], measured: str,
                         specification: str, channel: str, *,
                         band: Sequence[float] | None = None,
                         pair: str = 'abort',
                         provenance: Mapping[str, Mapping[str, Any]]
                         | None = None) -> FactDraft:
    """A fact draft read off a comparison the project holds: `measured`
    (a spectrum, by name) against `specification` (by name), on
    `channel`, over `band` if given, against the `pair` of limits.

    The reading is `core.findings.excursions`, on the same cells and at
    the same scale as the comparison figure. The record is named, and
    the time history it was computed from where `provenance` (the
    project's) says; that time history is offered as the run.

    Raises
    ------
    ValueError
        A name that is not an object of the project or not of the
        right type, a channel the two do not share, anything
        `excursions` refuses, and a channel inside its limits wherever
        it was judged, since that is no fact about a finding.
    """
    from ..core.data import DataArray, Specification

    def named(name, wanted, what):
        obj = objects.get(name) if isinstance(name, str) else None
        if obj is None:
            raise ValueError(f'{name!r} is not an object of the project')
        if not isinstance(obj, wanted) or (wanted is DataArray and
                                           isinstance(obj, Specification)):
            raise ValueError(f'{name!r} is a {type(obj).__name__}, not '
                             f'{what}')
        return obj

    spec = named(specification, Specification, 'a specification')
    held = named(measured, DataArray, 'a measured spectrum')
    band = core.band_of(band)
    spec_record, held_record = core.pair_records(spec, held, channel)
    reading = core.excursions(spec, held, spec_record, held_record,
                              pair=pair, band=band)
    source = ((provenance or {}).get(measured) or {}).get('source')
    kind = 'PSD' if type(held).__name__ == 'Psd' else type(held).__name__
    record = (f'{kind} {measured!r}, from {source!r}' if source
              else f'{kind} {measured!r}')
    text = fact_text(channel, reading, pair, record, band)
    stretches = reading.excursions
    return FactDraft(
        text=text,
        source=(f'{measured} against {specification}, {channel}, '
                f'{pair} limits'),
        run=str(source or measured), channel=channel,
        band=(stretches[0].low, stretches[-1].high))


def draft_from_comparison(report: Any, number: str,
                          objects: Mapping[str, Any], measured: str,
                          specification: str, channel: str | None = None,
                          *, band: Sequence[float] | None = None,
                          pair: str = 'abort',
                          provenance: Mapping[str, Mapping[str, Any]]
                          | None = None) -> FactDraft:
    """Read a fact draft off a comparison (`fact_from_comparison`) and
    `offer` it on finding `number`. The channel and band default to the
    finding's own. Returns the draft. Refused, changing nothing, as
    those two refuse, and when no channel is given or entered."""
    _require(report)
    entry = find(report, number)
    channel = channel if channel is not None else entry['channel']
    if not channel:
        raise ValueError(f'{number}: name the channel to read the fact off')
    if band is None:
        band = entry['band']
    draft = fact_from_comparison(objects, measured, specification, channel,
                                 band=band, pair=pair, provenance=provenance)
    offer(report, number, draft)
    return draft


# ---- the check before export ------------------------------------------------


def check(report: Any, owed: Sequence[skeleton.Owed]) -> tuple[skeleton.Owed,
                                                               ...]:
    """The skeleton's `owed` list with section 9 read off the register.

    With the feature off, `owed` as it is. With it on, the section is
    owed, as the skeleton says, while nothing is entered and nothing
    written; otherwise each of these is listed under it: a kind with
    nothing entered that the author has not said is none, a kind both
    entered and said to be none, and each finding's blank parts.
    """
    from .features import enabled

    if FEATURE not in enabled(report):
        return tuple(owed)
    found = skeleton.BY_KEY[SECTION]
    entries = register(report)
    words = skeleton.words(report, SECTION)
    waiting = log_drafts(report)
    # drafts from a run log are not findings: the section is owed as it
    # was, and the check says what waits to be accepted
    numbers = ', '.join(draft['number'] for draft in waiting)
    extra = ([skeleton.Owed(
        SECTION, found.number, found.title,
        f'{numbers} offered from the run log, not yet accepted')]
        if waiting else [])
    if not entries and not words:
        return (tuple(sorted([*owed, *extra], key=lambda item: item.number))
                if extra else tuple(owed))
    said = core.declared(words)
    reasons = []
    for kind in core.KINDS:
        mine = [entry for entry in entries if entry['kind'] == kind]
        plural = core.PLURAL[kind].lower()
        if not mine and kind not in said:
            reasons.append(f'{plural} neither entered nor set to None')
        elif mine and kind in said:
            reasons.append(f'{plural} entered and also said to be none')
    for entry in entries:
        missing = [core.FOURTH[entry['kind']].lower() if part == 'approval'
                   else part for part in core.owed(entry)]
        if missing:
            reasons.append(f'{entry["number"]}: {", ".join(missing)} not yet '
                           'written' + (f' ({DRAFTED})' if 'fact' in missing
                                        and entry['draft'] is not None
                                        else ''))
    rest = [item for item in owed if item.key != SECTION]
    mine = [skeleton.Owed(SECTION, found.number, found.title, reason)
            for reason in reasons]
    return tuple(sorted([*rest, *mine, *extra],
                        key=lambda item: item.number))


# ---- the page ---------------------------------------------------------------


@dataclass
class Heads:
    """The skeleton heads the register draws into (`prepare`) and what
    it adds to each once they are drawn (`draw`)."""

    summary: dict[str, Any] | None
    deviations: dict[str, Any] | None
    summary_html: str
    deviations_html: str
    #: the findings a run log offers, drawn on the editing page only
    offered_html: str = ''


def _escape(text: str) -> str:
    # the page keeps a text block's white space, so a line break in a
    # part is drawn as a break and nothing else adds one
    return '<br>'.join(html.escape(line) for line in text.splitlines())


def _owed_part() -> str:
    return f'<span class="owed">{skeleton.OWED}</span>'


def _where(entry: Mapping[str, Any]) -> str:
    said = []
    if entry['run']:
        said.append(f'run {entry["run"]}')
    if entry['channel']:
        said.append(entry['channel'])
    if entry['band'] is not None:
        low, high = entry['band']
        said.append(f'{_hz(low)} Hz' if low == high
                    else f'{_hz(low)} to {_hz(high)} Hz')
    if entry['status']:
        said.append(entry['status'])
    return '; '.join(said)


def finding_html(entry: Mapping[str, Any]) -> str:
    """One finding as the page draws it: its number and what it
    concerns, then its four labeled parts, each blank one owed."""
    where = _where(entry)
    head = (f'<p class="finding-head"><strong>'
            f'{html.escape(entry["number"])}.</strong> '
            f'{entry["kind"].capitalize()}'
            + (f' <span class="finding-where">({html.escape(where)})</span>'
               if where else '') + '</p>')
    parts = []
    for part in core.PARTS:
        label = (core.FOURTH[entry['kind']] if part == 'approval'
                 else part.capitalize())
        said = entry[part]
        shown = _escape(said) if said else _owed_part()
        if part == 'fact' and not said and entry['draft'] is not None:
            # a draft is never the fact until accepted: the part stays
            # owed, and says a drafted fact is waiting
            shown += f' <span class="finding-drafted">{DRAFTED}</span>'
        parts.append(f'<p class="finding-part"><strong>{label}:</strong> '
                     f'{shown}</p>')
    return f'<div class="finding">{head}{"".join(parts)}</div>'


def register_html(entries: Sequence[Mapping[str, Any]], said: set[str]) -> str:
    """The register as section 9 draws it: each kind in turn, its
    findings in number order, or its none sentence where the author
    said so, or an owed line where neither."""
    out = []
    for kind in core.KINDS:
        mine = [entry for entry in entries if entry['kind'] == kind]
        if mine:
            out.extend(finding_html(entry) for entry in mine)
        elif kind in said:
            out.append(f'<p class="finding-none">'
                       f'{html.escape(core.NONE_SENTENCES[kind])}</p>')
        else:
            out.append(f'<p class="finding-none">{_owed_part()} '
                       f'{core.PLURAL[kind]}: enter each in its four parts, '
                       'or set None.</p>')
    return ''.join(out)


def summary_html(entries: Sequence[Mapping[str, Any]], said: set[str]) -> str:
    """The Summary's line of findings by number: "Deviations: D-1, D-2.
    Anomalies: A-1 (closed). Nonconformances: none." A kind the author
    has neither entered nor said is none is owed."""
    out = []
    for kind in core.KINDS:
        mine = [entry for entry in entries if entry['kind'] == kind]
        if mine:
            listed = ', '.join(
                html.escape(entry['number'])
                + (f' ({entry["status"]})' if entry['status'] else '')
                for entry in mine)
        elif kind in said:
            listed = 'none'
        else:
            listed = '<span class="owed">not yet written</span>'
        out.append(f'{core.PLURAL[kind]}: {listed}.')
    return f'<p class="findings-summary">{" ".join(out)}</p>'


def offered_html(drafts: Sequence[Mapping[str, Any]]) -> str:
    """The findings a run log offers, as the editing page draws them under
    section 9: marked `FROM_LOG`, each with its proposed number, its kind,
    where it came from and its fact. '' when none is offered."""
    if not drafts:
        return ''
    out = [(f'<p class="finding-head"><span class="finding-drafted">'
            f'{FROM_LOG}</span></p>')]
    for draft in drafts:
        out.append(f'<p class="finding-part"><strong>Proposed '
                   f'{html.escape(draft["number"])}.</strong> '
                   f'{draft["kind"].capitalize()} <span class="finding-where">'
                   f'({html.escape(draft["source"])})</span>: '
                   f'{_escape(draft["fact"])}</p>')
    return f'<div class="finding">{"".join(out)}</div>'


def prepare(blocks: Sequence[dict[str, Any]], report: Any,
            edit: bool = False) -> Heads:
    """Ready the page `skeleton.arrange` laid out for the register.

    Section 9's head keeps the skeleton's owed card while nothing is
    entered and nothing written. Otherwise the register fills the
    section: its head shows the heading and the author's other words,
    the none sentences being drawn in their kinds' places by `draw`.
    Call before the text is drawn as HTML, and `draw` after
    `skeleton.finish`. With `edit`, the findings a run log offers
    (`log_drafts`) are drawn under section 9 as well, marked as drafted;
    they are never on the reading page.
    """
    def head(key):
        return next((block for block in blocks
                     if (block.get('skeleton') or {}).get('key') == key), None)

    entries = register(report)
    summary, deviations = head('summary'), head(SECTION)
    words = str(deviations.get('text', '')) if deviations is not None else ''
    said = core.declared(words)
    entered = {entry['kind'] for entry in entries}
    filled = ''
    if deviations is not None and (entries or words.strip()):
        # a none sentence of a kind with findings stays where the author
        # put it, so the contradiction is seen; the others are drawn in
        # their kinds' places
        deviations['text'] = core.withdraw(words, said - entered)
        deviations['skeleton']['state'] = 'written'
        filled = register_html(entries, said)
    return Heads(summary, deviations, summary_html(entries, said), filled,
                 offered_html(log_drafts(report)) if edit else '')


def draw(heads: Heads) -> None:
    """Add the register to section 9's head and the line of findings to
    the Summary's, once `skeleton.finish` has drawn them."""
    if heads.summary is not None:
        heads.summary['html'] = heads.summary.get('html', '') \
            + heads.summary_html
    if heads.deviations is not None and heads.deviations_html:
        heads.deviations['html'] = heads.deviations.get('html', '') \
            + heads.deviations_html
    if heads.deviations is not None and heads.offered_html:
        heads.deviations['html'] = heads.deviations.get('html', '') \
            + heads.offered_html
