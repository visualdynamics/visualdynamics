"""The test requirement and the result against it, in a report (report
feature `test_requirement`).

A requirement block carries the requirement the plan states
(`core.requirement.TestRequirement`), the specification it is held to
and the run it reads, and is saved with the report and its template.
With the feature on it draws two things:

- the requirement in words: the governing document and its revision,
  the tolerance clause, the control strategy, each band and tolerance,
  the decision rule, and where each number came from, with "supplied by
  the customer; the laboratory has not verified its derivation" for a
  number the customer supplied. With skeleton sections on, it falls
  under the Requirement section, where the author still owes the plan,
  the failure definition and the completion criteria.
- beneath it, with the feature `control_uncertainty` on as well, the
  control channels' measurement uncertainty against one third of the
  tolerance (`uncertainty`): a row per control channel with
  its contributions, the combined uncertainty in percent on the
  acceleration, on the ASD and in dB, the third it is held against, and
  met, not met or not judged; one line saying it was not judged when
  nothing is entered.
- the result table: a row per criterion with what was required, what
  was measured (each miss with its band, its worst size in dB and its
  line count) and the result, worded met, not met or not judged, and a
  note beneath with what the rows cannot say. Its cross-axis rows read
  each cross-axis channel against its in-axis channel, and its
  limit-channel rows give each notch a limit channel cut, the as-run
  notch, against the notches the plan names. A row with a notch the
  plan does not name, and a cross-axis row over its limit, offers its
  fact to the findings register (`core.requirement.Criterion.finding`):
  `draft_finding` builds the fact draft, naming the record, and hands
  it to the register's one entry point for drafts
  (`report.findings.offer`), as a comparison's draft is handed. With
  the feature `findings` on, a Finding column gives each such row the
  number of the finding its fact went to, or says none is entered. With
  skeleton sections on, the table falls under Results.

The pass/fail box reads from a requirement bound to the same
specification and run: MET, NOT MET or NOT JUDGED, how many criteria
were met against which document, the criteria not met and not judged,
and the decision rule. A box with no requirement bound to its pair
keeps its reading and is labeled a screening by the Visual Dynamics
default rule, so it is never read as a statement of conformity. Both
only while the feature is on; with it off a requirement block is not
drawn and the box is the box it was.

Nothing here is the author's text: the requirement is the author's own
numbers in words, and the table states what the program measured. The
statement of conformity in the conclusion stays the author's.

Where the requirement's run takes its duration at level is one
function, `run_duration`, so a later source of run identity feeds it
there and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# P17 of the report-features plan (Kevin Cross, 2026-10-24): the test
# requirement in words and a met or not met row per criterion, the
# pass/fail box driven by it or labeled a screening.

#: the feature's key in `report.features`
FEATURE = 'test_requirement'

#: the table's caption when the block gives none
CAPTION = 'Results against the requirement'

#: the feature that reads the cross-axis and limit channels
ROLES = 'cross_axis'

#: what a role's row says while that feature is off
ROLES_OFF = ('not judged: read with the report feature Cross-axis and limit '
             'channels on')

#: the uncertainty table's caption
UNCERTAINTY = 'Control-channel uncertainty against one third of the tolerance'

#: what a box with no requirement bound is labeled
SCREENING = 'Screening (Visual Dynamics default rule)'

#: the Finding cell of a row whose offered fact no finding holds yet
UNENTERED = 'none entered'

#: the result cell's mark, by state, for the page's style
MARKS = {'not met': 'notmet', 'not judged': 'notjudged'}


@dataclass(frozen=True)
class Made:
    """The page blocks one requirement block draws: its words, for the
    Requirement section, and its result table and note, for Results."""

    words: list[dict[str, Any]]
    results: list[dict[str, Any]]


def add(report: Any, requirement: Any, *, source: str, measured: str,
        duration_run: float | None = None, caption: str = '',
        at: int | None = None) -> int:
    """Give `report` a requirement block and return its index.

    The block is a text block carrying the requirement's values
    (`TestRequirement.as_record`), the specification (`source`) and the
    run (`measured`) it reads, the run's duration at level when stated,
    and the table's caption. It is drawn only while the feature is on,
    and is saved with the report and its template. It is put at `at`,
    or after the report's first pass/fail box, or at the end; without
    the skeleton it draws where it stands.

    Raises `TypeError` when `requirement` is not a `TestRequirement`,
    and `ValueError` for a duration that is not a positive number or a
    binding left empty. Nothing is changed when it raises.
    """
    from ..core.requirement import TestRequirement, _number

    if not isinstance(requirement, TestRequirement):
        raise TypeError('the requirement is a TestRequirement')
    if not source or not measured:
        raise ValueError('a requirement block names its specification and '
                         'its run')
    if duration_run is not None:
        duration_run = _number(duration_run, 'the duration at level')
        if not duration_run > 0.0:
            raise ValueError('the duration at level is a number of seconds '
                             'above 0')
    block: dict[str, Any] = {
        'kind': 'text', 'text': '', 'requirement': requirement.as_record(),
        'source': str(source), 'measured': str(measured),
        'caption': str(caption)}
    if duration_run is not None:
        block['duration_run'] = duration_run
    if at is None:
        boxes = [k for k, held in enumerate(report.blocks)
                 if held.get('kind') == 'verdict']
        at = boxes[0] + 1 if boxes else len(report.blocks)
    report.blocks.insert(int(at), block)
    return int(at)


def run_duration(block: Mapping[str, Any], objects: Mapping[str, Any],
                 object_groups: Sequence[Mapping[str, Any]] | None = None
                 ) -> float | None:
    """How long the block's run was at level, seconds, or None when
    nothing says.

    Today it is the controller's counter, typed on the block
    (`duration_run`). This is the one place the result table takes it
    from, so a later source (a run's own identity) is read here.
    """
    value = block.get('duration_run')
    return None if value is None else float(value)


def inputs(block: Mapping[str, Any], objects: Mapping[str, Any],
           object_groups: Sequence[Mapping[str, Any]] | None = None
           ) -> tuple[Any, Any, Any, str, str] | None:
    """(requirement, specification, run, the specification's name, the
    run's name), or None while either binding does not resolve.

    Raises `ValueError` for stored values the requirement refuses.
    """
    from ..core.report import resolve_binding
    from ..core.requirement import TestRequirement

    source = resolve_binding(block.get('source'), objects, object_groups)
    name = resolve_binding(block.get('measured'), objects, object_groups)
    if source not in objects or name not in objects:
        return None
    requirement = TestRequirement.from_record(block['requirement'])
    return requirement, objects[source], objects[name], str(source), str(name)


def channel_table(objects: Mapping[str, Any]) -> Any:
    """The project's channel table, whose Control flags name the control
    channels, or None."""
    from ..core.channel_table import ChannelTable

    tables = [obj for obj in objects.values() if isinstance(obj, ChannelTable)]
    return tables[0] if len(tables) == 1 else None


#: what a reading canceled before it was made is kept as, until the
#: report is drawn again
CANCELED = 'the reading was canceled; it is made when the report is drawn again'


def _key(block: Mapping[str, Any], read_from: Mapping[str, Any],
         duration: float | None, roles: bool = True) -> tuple:
    """What identifies one reading: the requirement's values, the
    duration, whether the roles are read, and each object read as
    `check_blocks` identifies it."""
    import json

    from .check_blocks import _state

    return ('requirement', json.dumps(block.get('requirement'),
                                      sort_keys=True, default=repr),
            duration, bool(roles),
            tuple((name, _state(obj)) for name, obj in read_from.items()))


def roles_on(report: Any) -> bool:
    """Whether `report` reads the cross-axis and limit channels (the
    feature `cross_axis`)."""
    from .features import enabled

    return ROLES in enabled(report)


def _unread_roles(requirement: Any, results: Any) -> Any:
    """`results` of the requirement read without its roles, with a row
    for each role saying it was not read: each cross-axis limit's row,
    and one per limit channel, so a result with a role stated is never
    met unread."""
    from dataclasses import replace

    from ..core.requirement import NOT_JUDGED, Criterion

    rows = [replace(row, measured=ROLES_OFF, state=NOT_JUDGED)
            if row.key == 'cross_axis' and requirement.cross_axis_channels
            else row for row in results.rows]
    rows += [Criterion('limit', f'Limit channel {channel.channel}',
                       channel.text, ROLES_OFF, NOT_JUDGED,
                       channel=channel.channel)
             for channel in requirement.limit_channels]
    return replace(results, rows=tuple(rows))


def _objects(block: Mapping[str, Any], objects: Mapping[str, Any],
             object_groups: Sequence[Mapping[str, Any]] | None
             ) -> dict[str, Any] | None:
    """The objects a block reads, by role, or None while a binding does
    not resolve."""
    from ..core.report import resolve_binding

    out = {}
    for key in ('source', 'measured'):
        name = resolve_binding(block.get(key), objects, object_groups)
        if name not in objects:
            return None
        out[key] = objects[name]
    table = channel_table(objects)
    if table is not None:
        out['table'] = table
    # the objects the limit channels name for their profiles, kept by
    # the name the requirement gives them
    for name in _profile_names(block):
        held = resolve_binding(name, objects, object_groups)
        if held in objects:
            out[f'profile:{name}'] = objects[held]
    return out


def _profile_names(block: Mapping[str, Any]) -> list[str]:
    """The profile names of the block's limit channels, as stored; none
    for values the requirement would refuse, which it says when read."""
    stored = (block.get('requirement') or {}).get('limit_channels') or []
    names = []
    for item in stored if isinstance(stored, list) else []:
        if isinstance(item, (list, tuple)) and len(item) > 1 \
                and isinstance(item[1], str) and item[1]:
            names.append(item[1])
    return names


def read(block: Mapping[str, Any], objects: Mapping[str, Any],
         object_groups: Sequence[Mapping[str, Any]] | None = None, *,
         roles: bool = True) -> Any:
    """The block's `core.requirement.Results`, or None while a binding
    does not resolve. Refusals raise `ValueError` as
    `core.requirement.evaluate` raises them. With `roles` False (the
    report's feature `cross_axis` off) the cross-axis and limit channels
    are not read, and their rows say so.

    Made once and kept while the requirement, the duration and the
    objects it read stay the same (in `check_blocks`'s store of
    readings), since a reading of several control channels takes longer
    than the GUI thread may wait; the app makes it on its worker first
    (`prepare`)."""
    from dataclasses import replace

    from ..core.requirement import evaluate
    from .check_blocks import _keep, _kept

    read_from = _objects(block, objects, object_groups)
    if read_from is None:
        return None
    duration = run_duration(block, objects, object_groups)
    key = _key(block, read_from, duration, roles)
    kept = _kept(key, read_from)
    if kept is None:
        try:
            requirement, specification, measured, _s, _n = inputs(
                block, objects, object_groups)
            held = requirement if roles else replace(
                requirement, cross_axis_channels={}, limit_channels=())
            kept = evaluate(held, specification, measured,
                            table=read_from.get('table'),
                            duration_run=duration,
                            profiles={name.removeprefix('profile:'): obj
                                      for name, obj in read_from.items()
                                      if name.startswith('profile:')})
            if not roles:
                kept = _unread_roles(requirement, kept)
        except (TypeError, ValueError) as refusal:
            kept = type(refusal)(str(refusal))
        _keep(key, read_from, kept)
    if isinstance(kept, Exception):
        raise type(kept)(str(kept))
    return kept


def pending(report: Any, objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None
            ) -> list[Mapping[str, Any]]:
    """The requirement blocks of `report` with no kept reading, or whose
    reading was canceled: what `prepare` would read. Empty when the
    feature is off."""
    from .check_blocks import _kept
    from .features import enabled

    if FEATURE not in enabled(report):
        return []
    out = []
    for block in report.blocks:
        if 'requirement' not in block:
            continue
        read_from = _objects(block, objects, object_groups)
        if read_from is None:
            continue
        kept = _kept(_key(block, read_from, run_duration(
            block, objects, object_groups), roles_on(report)), read_from)
        if kept is None or str(kept) == CANCELED:
            out.append(block)
    return out


def prepare(report: Any, objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None, *,
            progress: Any = None) -> int:
    """Read every requirement block of `report` with no kept reading, so
    the page is drawn without reading any. `progress(done, total)` is
    told before each and at the end; what it raises (a cancel) stops
    there, and each block not yet read is kept as canceled, drawn as
    nothing judged and offered again by `pending`. Returns how many were
    read."""
    from .check_blocks import _KEPT, _keep

    waiting = pending(report, objects, object_groups)

    def key_of(block):
        read_from = _objects(block, objects, object_groups)
        return read_from, _key(block, read_from, run_duration(
            block, objects, object_groups), roles_on(report))

    for done, block in enumerate(waiting):
        try:
            if progress is not None:
                progress(done, len(waiting))
        except BaseException:
            for rest in waiting[done:]:
                read_from, key = key_of(rest)
                _keep(key, read_from, ValueError(CANCELED))
            raise
        _KEPT.pop(key_of(block)[1], None)
        try:
            read(block, objects, object_groups, roles=roles_on(report))
        except (TypeError, ValueError):
            pass        # kept, and said on the page
    if progress is not None:
        progress(len(waiting), len(waiting))
    return len(waiting)


def build(block: Mapping[str, Any], objects: Mapping[str, Any],
          object_groups: Sequence[Mapping[str, Any]] | None = None, *,
          uncertainty_on: bool = False, roles_on: bool = False,
          register: Sequence[Mapping[str, Any]] | None = None
          ) -> Made | None:
    """The page blocks of one requirement block, or None while a binding
    does not resolve. A requirement that cannot be read against its run
    at all draws its words and, for the results, one paragraph saying
    why, since a reading that judged nothing says so. `uncertainty_on`
    (the feature `control_uncertainty`) adds the control channels'
    uncertainty under the words, `roles_on` (the feature
    `cross_axis`) reads the cross-axis and limit channels, and
    `register` (the findings entered, with the feature `findings` on)
    gives a row that offers a finding its Finding cell."""
    from ..core.requirement import TestRequirement

    try:
        requirement = TestRequirement.from_record(block['requirement'])
    except (TypeError, ValueError) as refusal:
        return Made([], [{'kind': 'text', 'text':
                          f'**{CAPTION}:** the requirement cannot be read; '
                          f'{refusal}.'}])
    words = [{'kind': 'text', 'text': _words(requirement)}]
    try:
        found = read(block, objects, object_groups, roles=roles_on)
    except (TypeError, ValueError) as refusal:
        if uncertainty_on:
            words += uncertainty(requirement)
        return Made(words, [{'kind': 'text', 'text':
                             f'**{CAPTION}:** nothing judged; {refusal}.'}])
    if found is None:
        return None
    if uncertainty_on:
        words += uncertainty(requirement, found.channels)
    name = str(block.get('measured') or '')
    return Made(words, [table(found, name, str(block.get('caption') or ''),
                              register),
                        {'kind': 'text', 'text': _note(found)}])


def place(blocks: Sequence[dict[str, Any]], made: Sequence[Made]
          ) -> list[dict[str, Any]]:
    """The skeleton's page blocks with each requirement's words at the
    end of the Requirement section and its result table and note at the
    head of Results, before the figures, since the control result
    against the tolerance is what that section opens with."""
    from .skeleton import under

    out = under(blocks, 'requirement',
                [built for each in made for built in each.words])
    head = next(at for at, block in enumerate(out)
                if block.get('skeleton', {}).get('key') == 'results')
    return (out[:head + 1] + [built for each in made for built in each.results]
            + out[head + 1:])


def requirement_lines(block: Mapping[str, Any]) -> list[str]:
    """The block's requirement in words, a line each
    (`TestRequirement.describe`). Raises `ValueError` for stored values
    the requirement refuses."""
    from ..core.requirement import TestRequirement

    return TestRequirement.from_record(block['requirement']).describe()


def uncertainty(requirement: Any, channels: Sequence[str] | None = None
                ) -> list[dict[str, Any]]:
    """The control channels' measurement uncertainty against one third
    of the tolerance (`core.requirement.uncertainty_budget`): a table
    with a row per control channel and a note, or, with nothing entered
    for any, one line saying it was not judged."""
    from ..core.requirement import COVERAGE, uncertainty_budget

    if not requirement.uncertainty:
        return [{'kind': 'text', 'text':
                 f'**{UNCERTAINTY}:** not judged; no contributions are '
                 'entered for the control channels.'}]
    budgets = uncertainty_budget(requirement, channels)
    rows, marks = [], []
    for budget in budgets:
        if budget.contributions:
            numbers = [f'{budget.acceleration_percent:.2g} %',
                       f'{budget.asd_percent:.2g} %', f'{budget.db:.2g} dB']
        else:
            numbers = ['—'] * 3
        third = f'{budget.band_limit_db:.2g} dB (±{budget.band_db:g} dB band)'
        if budget.grms_limit_percent is not None:
            third += (f'; {budget.grms_limit_percent:.2g} % '
                      f'(±{requirement.grms_percent:g} % rms)')
        # a reading not judged says so in its own words
        result = (budget.words if budget.state == 'not judged' else
                  budget.state + (f': {budget.words}' if budget.words else ''))
        rows.append([budget.channel,
                     '; '.join(c.text for c in budget.contributions) or '—',
                     *numbers, third, result])
        marks.append([''] * 6 + [MARKS.get(budget.state, '')])
    built: dict[str, Any] = {
        'kind': 'table', 'caption': UNCERTAINTY,
        'headers': ['Channel', 'Contributions', 'Combined, acceleration',
                    'On the ASD', 'In dB', 'One third of the tolerance',
                    'Result'],
        'rows': rows}
    if any(any(mark) for mark in marks):
        built['marks'] = marks
    lines = [(f'- Each contribution is divided by its coverage factor, the '
             f'results combined as the root of the sum of their squares, '
             f'and the total expanded at k = {COVERAGE:g}; in dB it is '
             'taken on the side it is larger.')]
    found = {budget.channel for budget in budgets}
    others = [channel for channel in requirement.uncertainty
              if channel not in found]
    if others:
        lines.append(f'- Contributions are entered for {", ".join(others)}, '
                     f'which {"is" if len(others) == 1 else "are"} not a '
                     'control channel of the run.')
    return [built, {'kind': 'text', 'text': '\n'.join(lines)}]


def _words(requirement: Any) -> str:
    """The requirement as Markdown: a bold lead and a line per part."""
    return '**Test requirement.**\n\n' + '\n'.join(
        f'- {line}' for line in requirement.describe())


def table(results: Any, name: str = '', caption: str = '',
          register: Sequence[Mapping[str, Any]] | None = None
          ) -> dict[str, Any]:
    """The result table: a row per criterion, its result cell marked by
    state. With a findings `register` (the feature `findings` on), a
    Finding column is added when any row offers a finding
    (`Criterion.finding`): the numbers of the findings its fact went
    to, or `UNENTERED`; blank on a row that offers none."""
    findings = register is not None and any(
        row.finding is not None for row in results.rows)
    headers = ['Criterion', 'Requirement', 'Measured', 'Result']
    if findings:
        headers.append('Finding')
    rows, marks = [], []
    for row in results.rows:
        cells = [row.name, row.required, row.measured, row.state]
        mark = ['', '', '', MARKS.get(row.state, '')]
        if findings:
            cells.append(_finding_cell(row.finding, register))
            mark.append('')
        rows.append(cells)
        marks.append(mark)
    said = caption or CAPTION
    where = ', '.join(part for part in (
        name, results.requirement.document_text) if part)
    built: dict[str, Any] = {'kind': 'table',
                             'caption': f'{said}: {where}' if where else said,
                             'headers': headers, 'rows': rows}
    if any(any(mark) for mark in marks):
        built['marks'] = marks
    return built


def _finding_cell(offer: Any, register: Sequence[Mapping[str, Any]]) -> str:
    """The Finding cell of a row: blank for a row that offers no
    finding, else the numbers of the findings its fact went to (an
    accepted fact keeps the draft's channel and band, a waiting draft
    carries them), else `UNENTERED`."""
    if offer is None:
        return ''
    held = (offer.channel, list(offer.band))
    numbers = [entry['number'] for entry in register
               if (entry['channel'], entry['band']) == held
               or (entry['draft'] is not None
                   and (entry['draft']['channel'],
                        entry['draft']['band']) == held)]
    return ', '.join(numbers) or UNENTERED


def fact_draft(row: Any, results: Any, measured: str, specification: str,
               provenance: Mapping[str, Mapping[str, Any]] | None = None
               ) -> Any:
    """The `report.findings.FactDraft` a row of the result table offers:
    its fact (`Criterion.finding`), the record it was read off and the
    time history that record came from where `provenance` (the
    project's) says, and the scale the run was compared at. The time
    history, or else the record, is the run; the channel and band are
    the offer's.

    Raises `ValueError` for a row that offers no finding."""
    from .findings import FactDraft

    offer = row.finding
    if offer is None:
        raise ValueError(f'{row.name} offers no finding: only a notch the '
                         'plan does not name and a cross-axis channel over '
                         'its limit do')
    source = ((provenance or {}).get(measured) or {}).get('source')
    record = (f'PSD {measured!r}, from {source!r}' if source
              else f'PSD {measured!r}')
    text = f'{offer.text}; {record}'
    if results.scale_db:
        text += f'; compared at {results.scale_db:+g} dB'
    return FactDraft(
        text=text + '.',
        source=(f'{measured} against {specification}, '
                f'{results.requirement.document_text}, {row.name}'),
        run=str(source or measured), channel=offer.channel,
        band=tuple(offer.band))


def draft_finding(report: Any, number: str, objects: Mapping[str, Any],
                  measured: str, specification: str, criterion: int | str,
                  object_groups: Sequence[Mapping[str, Any]] | None = None,
                  *, provenance: Mapping[str, Mapping[str, Any]]
                  | None = None) -> Any:
    """Offer finding `number` the fact a row of a result table offers,
    and return the draft (`fact_draft`).

    The row is of the requirement bound to `specification` and
    `measured`, read as the page reads it, and is named by `criterion`:
    its number in the table from 1, or its name. The draft goes to the
    finding through the register's one entry point for drafts
    (`report.findings.offer`), so it fills nothing until the author
    accepts it.

    Raises
    ------
    ValueError
        When the report does not have the features `test_requirement`
        and `findings` enabled; for an unknown finding, no requirement
        bound to the pair, one that cannot be read, a row number out of
        range, a name no row has, a name two offering rows share, and a
        row that offers no finding. Nothing is changed.
    """
    from ..core.report import resolve_binding
    from . import findings
    from .features import enabled

    if FEATURE not in enabled(report):
        raise ValueError(f'report {getattr(report, "title", "")!r} does not '
                         'have Requirement and results enabled')
    findings.find(report, number)
    pair = (resolve_binding(specification, objects, object_groups),
            resolve_binding(measured, objects, object_groups))
    bound = [block for block in report.blocks if 'requirement' in block
             and tuple(resolve_binding(block.get(key), objects,
                                       object_groups)
                       for key in ('source', 'measured')) == pair]
    if not bound:
        raise ValueError(f'no requirement is bound to {specification!r} '
                         f'and {measured!r}')
    results = read(bound[0], objects, object_groups, roles=roles_on(report))
    if results is None:
        raise ValueError(f'the requirement bound to {specification!r} and '
                         f'{measured!r} cannot be read: a binding does not '
                         'resolve')
    if isinstance(criterion, int) and not isinstance(criterion, bool):
        if not 1 <= criterion <= len(results.rows):
            raise ValueError(f'the result table has rows 1 to '
                             f'{len(results.rows)}, not {criterion}')
        row = results.rows[criterion - 1]
    else:
        named = [row for row in results.rows if row.name == criterion]
        if not named:
            raise ValueError(f'the result table has no row {criterion!r}')
        offering = [row for row in named if row.finding is not None]
        if len(offering) > 1:
            raise ValueError(f'{len(offering)} rows {criterion!r} offer a '
                             'finding; name the row by its number')
        row = offering[0] if offering else named[0]
    draft = fact_draft(row, results, pair[1], pair[0], provenance)
    findings.offer(report, number, draft)
    return draft


def _note(results: Any) -> str:
    """What the rows cannot say, as Markdown: the result, the scale, and
    the reading's notes."""
    count = len(results.rows)
    lines = [(f'- {results.met} of {count} '
              f'criteri{"on" if count == 1 else "a"} met; the result is '
              f'{results.state}.')]
    if results.scale_db:
        lines.append(f'- Every comparison is of the measurement scaled by '
                     f'{results.scale_db:+g} dB.')
    lines.extend(f'- {note[:1].upper()}{note[1:]}.' for note in results.notes)
    return '\n'.join(lines)


def matching(report: Any, verdict: Mapping[str, Any],
             objects: Mapping[str, Any],
             object_groups: Sequence[Mapping[str, Any]] | None = None
             ) -> Mapping[str, Any] | None:
    """The requirement block the pass/fail block `verdict` reads from:
    the first bound to the same specification and run, else the one
    requirement of a report that holds one (the box of a generated
    report judges the octave-band pair, the requirement the narrowband
    run of the same test), else None."""
    from ..core.report import resolve_binding

    pair = tuple(resolve_binding(verdict.get(key), objects, object_groups)
                 for key in ('source', 'measured'))
    held = [block for block in report.blocks if 'requirement' in block]
    for block in held:
        if tuple(resolve_binding(block.get(key), objects, object_groups)
                 for key in ('source', 'measured')) == pair:
            return block
    return held[0] if len(held) == 1 else None


def verdict(built: dict[str, Any] | None, block: Mapping[str, Any],
            report: Any, objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None
            ) -> dict[str, Any] | None:
    """The pass/fail box `built` from `block`, as the feature draws it.

    Read from a requirement bound to the same pair when the report has
    one: 'requirement' carries the result (its state, how many criteria
    were met of how many, the document, the criteria not met and not
    judged, and the rule). Otherwise the box keeps its reading and is
    marked 'screening'. None stays None.
    """
    if built is None:
        return None
    bound = matching(report, block, objects, object_groups)
    if bound is not None:
        try:
            found = read(bound, objects, object_groups,
                         roles=roles_on(report))
        except (TypeError, ValueError) as refusal:
            return {**built, 'requirement': {
                'state': 'not judged', 'met': 0, 'rows': 0,
                'document': '', 'missed': [], 'unjudged': [str(refusal)],
                'rule': ''}}
        if found is not None:
            requirement = found.requirement
            return {**built, 'requirement': {
                'state': found.state, 'met': found.met,
                'rows': len(found.rows),
                'document': requirement.document_text,
                'missed': [f'{row.name}, {row.required}' for row in found.rows
                           if row.state == 'not met'],
                'unjudged': [f'{row.name}, {row.required}'
                             for row in found.rows
                             if row.state == 'not judged'],
                'rule': requirement.rule_text}}
    return {**built, 'screening': SCREENING}
