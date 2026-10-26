"""Runs and chronology (report feature `run_sections`).

A real test is a dozen runs on three axes, and a report bound to one
time history cannot be the report of the test. With the feature on:

- every time history in the project is a run, with its identity
  (`identity`): what its file's name said on import
  (`core.runs.read_name`), what was entered (`set_identity`), and the
  level the comparison with a specification detects
  (`compliance.read_scale_db`) when nothing else gives it. Anything
  read two ways is refused with the record named, and stays unknown
  until it is entered. A history cut or filtered from another is the
  same run as the one it came from (`root_of`), not a run of its own;
- the Results section of the skeleton is grouped by run
  (`group_results`): each generated section whose figures all come from
  one run goes under that run's heading, in the order the runs were
  run, and a section that compares runs, or comes from none, stays
  first. `repeat` adds the Results sections a report has for one run
  again for each other run, bound to that run's own objects;
- the chronology of the runs (`chronology`): each run in order with its
  axis, level, start, the time it was at level and the interruptions
  found in its record (`core.runs.read_levels`), offered as draft text
  for the skeleton's Chronology section (`draft`). The chronology is the
  author's: the draft is placed only when the author accepts it
  (`skeleton.write`), and the outcome of each run and the explanation of
  each interruption are left to the author.

A run log, when there is one, is a second source of outcomes and
explanations. It feeds `outcomes`, and nothing else here changes.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..core.runs import (
    DROP_DB,
    ENTERED,
    FIELD_NAMES,
    MIN_S,
    WINDOW_S,
    RecordReading,
    RunIdentity,
    entered,
    level_text,
    order,
    read_levels,
)

# P27 of the report-features plan (Kevin Cross, 2026-10-24): run
# identity on each record, the Results section per run, and the
# chronology offered as draft text the author accepts or edits. Built
# from the records alone; the run log is a later source (`outcomes`).

#: the feature's key in `report.features`
FEATURE = 'run_sections'

#: the binding fields that tie a figure to the data it draws, and so to
#: a run; a specification is the requirement, the same for every run
BOUND = ('source', 'measured', 'dofs_source', 'floor')

#: project types whose records are bursts or events, not one continuous
#: run at level
EVENTS = ('Shock', 'Transient', 'Modal Test', 'System ID')


def _provenance(objects: Mapping[str, Any]) -> Mapping[str, Mapping]:
    return getattr(objects, 'provenance', None) or {}


def _groups(objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None
            ) -> Sequence[Mapping[str, Any]]:
    if object_groups is None:
        object_groups = getattr(objects, 'object_groups', None)
    return object_groups or []


def _linked(name: str, object_groups: Sequence[Mapping[str, Any]]
            ) -> list[str]:
    for group in object_groups:
        if name in group.get('members', ()):
            return list(group['members'])
    return []


def root_of(objects: Mapping[str, Any], name: str | None) -> str | None:
    """The run `name` belongs to: the time history it was computed from,
    through any number of steps, back to the first. A history derived
    from another (cut, filtered) answers with the one it came from.
    None when `name` comes from no time history."""
    from ..core.data import TimeHistory

    provenance = _provenance(objects)
    found, seen = None, set()
    while isinstance(name, str) and name not in seen:
        seen.add(name)
        if isinstance(objects.get(name), TimeHistory):
            found = name
        name = provenance.get(name, {}).get('source')
    return found


def histories(objects: Mapping[str, Any]) -> list[str]:
    """The runs of the project: each time history not computed from
    another, in the project's order."""
    from ..core.data import TimeHistory

    return [name for name, obj in objects.items()
            if isinstance(obj, TimeHistory) and root_of(objects, name) == name]


def _run_history(objects: Mapping[str, Any], name: str) -> Any:
    from ..core.data import TimeHistory

    if name not in objects:
        raise ValueError(f'no object called {name!r}')
    history = objects[name]
    if not isinstance(history, TimeHistory):
        raise TypeError(f'{name!r} is a {type(history).__name__}, not a time '
                         'history; a run identity belongs to a time history')
    root = root_of(objects, name)
    if root != name:
        raise ValueError(f'{name!r} was computed from {root!r} and is part '
                         f'of that run; its identity is {root!r}\'s')
    return history


def _comparison_level(objects: Mapping[str, Any], root: str,
                      object_groups: Sequence[Mapping[str, Any]]
                      ) -> tuple[float | None, str, str]:
    """(level, where it came from, '') read off the comparisons of the
    run's PSDs with their specification; (None, '', why) when they give
    it two ways; (None, '', '') when there is no comparison to read."""
    from ..core.compliance import read_scale_db
    from ..core.data import Psd, Specification
    from ..core.report import is_banded

    psds = [name for name, obj in objects.items()
            if isinstance(obj, Psd) and not isinstance(obj, Specification)
            and not is_banded(obj) and root_of(objects, name) == root]
    specifications = [name for name, obj in objects.items()
                      if isinstance(obj, Specification) and not is_banded(obj)]
    readings: dict[float, str] = {}
    for psd in psds:
        linked = _linked(psd, object_groups)
        against = ([name for name in specifications if name in linked]
                   or specifications)
        if not against:
            continue
        if len(against) > 1:
            return None, '', (
                f'{len(against)} specifications could give the level of '
                f'{psd!r} ({", ".join(against)})')
        family = (objects.scale_family(psd) if hasattr(objects, 'scale_family')
                  else [psd])
        held = [objects[name].scale_db for name in family
                if getattr(objects[name], 'scale_db', None) is not None]
        if held:
            readings.setdefault(-float(held[0]) + 0.0,
                                f'the comparison scale set on {psd!r}')
            continue
        scale = read_scale_db(objects[against[0]], objects[psd])
        if scale is not None:
            readings.setdefault(-float(scale) + 0.0,
                                f'the comparison of {psd!r} with '
                                f'{against[0]!r}')
    if len(readings) > 1:
        told = '; '.join(f'{level_text(level)} from {where}'
                         for level, where in readings.items())
        ways = {2: 'two', 3: 'three'}.get(len(readings), str(len(readings)))
        return None, '', f'the level is read {ways} ways ({told})'
    if readings:
        level, where = next(iter(readings.items()))
        return level, where, ''
    return None, '', ''


def identity(objects: Mapping[str, Any], name: str,
             object_groups: Sequence[Mapping[str, Any]] | None = None
             ) -> RunIdentity:
    """The run identity of the time history `name`, as the report reads it.

    What the history carries (read from its file's name on import, or
    entered), and the level from the comparison with its specification
    when nothing else gives one. A level read from a name and a
    different one read from the comparison is refused, named; an
    entered level stands over both. An object computed from a history
    answers with that run's identity.

    Raises `ValueError` for a name not in the project, and `TypeError`
    for an object that is not a time history and comes from none.
    """
    root = root_of(objects, name) if name in objects else None
    if root is None:
        _run_history(objects, name)
    stored = getattr(objects[root], 'run_identity', None) or RunIdentity()
    if stored.reason('level_db') or stored.source('level_db') == ENTERED:
        return stored
    groups = _groups(objects, object_groups)
    level, where, why = _comparison_level(objects, root, groups)
    if stored.level_db is None:
        if level is not None:
            return stored.with_values({'level_db': level}, where)
        if why:
            return replace(stored, refused=stored.refused
                           + (('level_db', why),))
        return stored
    if level is not None and level != stored.level_db:
        told = (f'the level is read two ways ({level_text(stored.level_db)} '
                f'from the {stored.source("level_db")}, {level_text(level)} '
                f'from {where})')
        cleared = stored.with_values({'level_db': None}, '')
        return replace(cleared, refused=cleared.refused
                       + (('level_db', told),))
    return stored


def set_identity(objects: Mapping[str, Any], name: str,
                 **fields: Any) -> RunIdentity:
    """Enter fields of the run identity of the time history `name`, and
    return what it carries afterwards.

    An entered field stands over what was read; an empty string clears
    it, back to what the record shows. Refused, before anything
    changes: a history computed from another (its run is that one's),
    with `ValueError`; an object that is not a time history, with
    `TypeError`; and every refusal of `core.runs.entered`, each naming
    the record.
    """
    history = _run_history(objects, name)
    values = entered(name, **fields)
    stored = getattr(history, 'run_identity', None) or RunIdentity()
    changed = stored.with_values(values, ENTERED)
    # a cleared field may have been read from the file's name: what was
    # read is not kept apart, so clearing it leaves it unknown, which
    # the source column says
    history.run_identity = changed if changed.to_json() else None
    return changed


# ---- the record ---------------------------------------------------------

def _channels(objects: Mapping[str, Any], name: str,
              object_groups: Sequence[Mapping[str, Any]]
              ) -> tuple[list[int], str]:
    """The records of history `name` its level is read on, and how they
    were chosen: its control channels (the channel table's Control
    flag), else its response channels; one quantity, the most common
    among them, and never a reference (a drive)."""
    from ..core.channel_table import ChannelTable

    history = objects[name]
    roles = history.channel_roles()
    flagged: set[str] = set()
    for member in _linked(name, object_groups):
        table = objects.get(member)
        if isinstance(table, ChannelTable):
            dofs = table.dof_strings()
            flagged |= {dofs[i] for i in table.chosen('control')}
    pairs = list(zip(history.response_dof, history.ordinate_dim, strict=True))
    if flagged:
        chosen = [i for i, (dof, quantity) in enumerate(pairs)
                  if dof in flagged
                  and roles.get((dof, quantity)) != 'reference']
        how = 'the control channels'
    else:
        chosen = []
    if not chosen:
        chosen = [i for i, pair in enumerate(pairs)
                  if roles.get(pair) == 'response']
        how = 'the response channels'
    if not chosen:
        chosen = list(range(len(pairs)))
        how = 'every channel'
    if not chosen:
        return [], how
    quantities = [pairs[i][1] for i in chosen]
    common = max(dict.fromkeys(quantities), key=quantities.count)
    return [i for i in chosen if pairs[i][1] == common], how


def read_record(objects: Mapping[str, Any], name: str,
                object_groups: Sequence[Mapping[str, Any]] | None = None,
                window_s: float = WINDOW_S, drop_db: float = DROP_DB,
                min_s: float = MIN_S) -> RecordReading:
    """What the record of run `name` shows: its time at level and the
    interruptions in it (`core.runs.read_levels`), on its control
    channels. A record of bursts or events (a shock, a transient, a
    modal or system-ID run, a stack of captures) is not one run at
    level, and is not read: `not_read` says so."""
    import numpy as np

    from ..core.data import _even_steps

    history = objects[name]
    abscissa = np.asarray(history.abscissa, dtype=float)
    start = float(abscissa[0]) if len(abscissa) else 0.0
    # a record of n samples at step dt lasts n dt, as `read_levels`
    # counts it; the span between the first and last instants is the
    # nearest reading while the step is not known
    seconds = (float(abscissa[-1] - abscissa[0]) * len(abscissa)
               / (len(abscissa) - 1) if len(abscissa) > 1 else 0.0)

    def refused(why: str) -> RecordReading:
        return RecordReading(start, seconds, not_read=why,
                             settings={'window_s': float(window_s),
                                       'drop_db': float(drop_db),
                                       'min_s': float(min_s)})

    kind = getattr(objects, 'project_type', None)
    if kind in EVENTS:
        return refused(f'a {kind.lower()} record is not one run at level; '
                       'its time at level and interruptions are not read')
    if getattr(history, 'shocks', None):
        return refused('a record of shocks is not one run at level; its '
                       'time at level and interruptions are not read')
    if history.block is not None and len(set(history.block)) > 1:
        return refused(f'a stack of {len(set(history.block))} captures is not '
                       'one continuous record; its time at level and '
                       'interruptions are not read')
    if len(abscissa) < 2:
        return refused('the record has fewer than two samples')
    try:
        step = _even_steps(abscissa, 'Reading the time at level')
    except ValueError as uneven:
        return refused(str(uneven))
    rows, how = _channels(objects, name, _groups(objects, object_groups))
    labels = [history.response_dof[i] or f'record {i + 1}' for i in rows]
    reading = read_levels(history.ordinate[rows], 1.0 / step, labels,
                          start, window_s, drop_db, min_s)
    if reading.not_read:
        return replace(reading, not_read=f'{reading.not_read} (on {how})')
    return reading


# ---- what a later source says ---------------------------------------------

@dataclass(frozen=True)
class Outcome:
    """What a source other than the record says of a run: its outcome
    ('completed', 'aborted by the limit channel') and an explanation for
    each interruption found, in order."""

    text: str = ''
    explanations: tuple[str, ...] = ()


def outcomes(objects: Mapping[str, Any]) -> dict[str, Outcome]:
    """The outcome of each run, by record name, from a source other than
    the record.

    This is the one place such a source is read. A run log kept beside
    the test is the planned one; until there is a source, no run has an
    outcome here, and the chronology leaves the Outcome column and the
    explanation of each interruption to the author.
    """
    del objects
    return {}


# ---- the chronology -----------------------------------------------------

@dataclass(frozen=True)
class RunRow:
    """One run in the chronology: its record, identity, what its record
    shows, and what a later source says of it."""

    name: str
    identity: RunIdentity
    reading: RecordReading
    outcome: Outcome = Outcome()


@dataclass(frozen=True)
class Chronology:
    """The runs in the order they were run (`chronology`). `ordered_by`
    is 'start' or 'run number', '' for a lone run; `notes` are what was
    refused or left out, each naming its record."""

    rows: tuple[RunRow, ...]
    ordered_by: str
    notes: tuple[str, ...] = ()


def chronology(objects: Mapping[str, Any],
               object_groups: Sequence[Mapping[str, Any]] | None = None,
               window_s: float = WINDOW_S, drop_db: float = DROP_DB,
               min_s: float = MIN_S,
               progress: Callable[[int, int], None] | None = None
               ) -> Chronology:
    """The runs of the project in the order they were run, each with its
    identity and what its record shows.

    Raises `ValueError` when the project holds no time history, when
    the order cannot be read from the identities (naming the records
    with neither a start nor a run number), and for a window, drop or
    duration `core.runs.read_levels` refuses.
    """
    from ..core.runs import check_settings

    check_settings(window_s, drop_db, min_s)
    names = histories(objects)
    if not names:
        raise ValueError('the project holds no time history, so there are '
                         'no runs to list')
    groups = _groups(objects, object_groups)
    identities = [(name, identity(objects, name, groups)) for name in names]
    ordered = order(identities)
    by_name = dict(identities)
    told = outcomes(objects)
    rows, notes = [], []
    for done, name in enumerate(ordered, start=1):
        found = by_name[name]
        reading = read_record(objects, name, groups, window_s, drop_db, min_s)
        rows.append(RunRow(name, found, reading, told.get(name, Outcome())))
        for field_name, why in found.refused:
            notes.append(f'{name}: {why}, so the {FIELD_NAMES[field_name]} '
                         'is not read; enter it.')
        if reading.not_read:
            notes.append(f'{name}: {reading.not_read}.')
        for left in reading.left_out:
            notes.append(f'{name}: {left}, left out of the level.')
        if progress is not None:
            progress(done, len(ordered))
    ordered_by = ('' if len(identities) < 2
                  else 'start' if all(found.start for _n, found in identities)
                  else 'run number')
    return Chronology(tuple(rows), ordered_by, tuple(notes))


def _cell(text: str) -> str:
    return str(text).replace('|', '\\|').replace('\n', ' ')


def draft(found: Chronology) -> str:
    """The chronology as Markdown for the skeleton's Chronology section:
    a table of the runs in order, then how the time at level and the
    interruptions were read, and what is the author's to add.

    Draft text only: nothing places it in a report until the author
    accepts it (`Project.report_section`)."""
    commented = any(row.identity.comment for row in found.rows)
    head = ['Run', 'Axis', 'Level', 'Start', 'Record', 'At level',
            'Interruptions found', 'Outcome']
    if commented:
        head.append('Comment')
    lines = ['| ' + ' | '.join(head) + ' |',
             '|' + '|'.join('---:' if title == 'At level' else '---'
                            for title in head) + '|']
    for row in found.rows:
        who = row.identity
        reading = row.reading
        if reading.not_read:
            at = 'not read'
            breaks = 'not read'
        else:
            at = f'{reading.at_level_s:.1f} s'
            parts = [str(stop) for stop in reading.interruptions]
            for k, why in enumerate(row.outcome.explanations[:len(parts)]):
                if why:
                    parts[k] += f': {why}'
            breaks = '; '.join(parts) if parts else 'none found'
        cells = [who.run, who.axis,
                 '' if who.level_db is None else level_text(who.level_db),
                 who.start.replace('T', ' '), row.name, at, breaks,
                 row.outcome.text]
        if commented:
            cells.append(who.comment)
        lines.append('| ' + ' | '.join(_cell(cell) for cell in cells) + ' |')
    settings = found.rows[0].reading.settings if found.rows else {}
    window = settings.get('window_s', WINDOW_S)
    drop = settings.get('drop_db', DROP_DB)
    least = settings.get('min_s', MIN_S)
    ordered = (f'The runs are in the order of their {found.ordered_by}. '
               if found.ordered_by else '')
    lines += [
        '',
        (f'{ordered}The time at level is read from the running RMS of each '
         f"record's control channels over {window:g} s, within 3 dB of the "
         f"run's level. An interruption is a drop of at least {drop:g} dB "
         f'below that level for at least {least:g} s, between reaching level '
         'and leaving it, timed on the record. The outcome of each run and '
         'the cause of each interruption are for the author to enter.')]
    if found.notes:
        lines += [''] + [f'- {note}' for note in found.notes]
    return '\n'.join(lines)


# ---- the Results section, run by run --------------------------------------

def runs_of(block: Mapping[str, Any], objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None
            ) -> set[str]:
    """The runs a report block draws from: the run of each object its
    data fields bind. Empty for text, and for a figure of a
    specification alone."""
    from ..core.report import resolve_binding

    out = set()
    for key in BOUND:
        if block.get(key):
            root = root_of(objects, resolve_binding(block[key], objects,
                                                    object_groups))
            if root is not None:
                out.add(root)
    return out


def _sections(origins: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Positions split where a text block opens a '## ' heading, as the
    skeleton reads a generated report's sections."""
    from .skeleton import _heading

    out: list[list[int]] = []
    for position, origin in enumerate(origins):
        if _heading(origin) or not out:
            out.append([])
        out[-1].append(position)
    return out


def run_order(objects: Mapping[str, Any], names: Sequence[str],
              object_groups: Sequence[Mapping[str, Any]]
              ) -> tuple[list[str], dict[str, RunIdentity]]:
    """The runs `names` in the order they were run, and each one's
    identity. When the identities cannot say the order, the project's
    order: a page keeps showing the runs, and the chronology, which
    refuses, names what to enter."""
    every = histories(objects)
    names = [name for name in every if name in set(names)]
    found = {name: identity(objects, name, object_groups) for name in names}
    try:
        return order([(name, found[name]) for name in names]), found
    except ValueError:
        # the page keeps the project's order; the chronology names
        # what is missing
        return names, found


def heading(name: str, found: RunIdentity) -> str:
    """A run's heading over its results: 'Run 7, Z axis, -6 dB,
    2026-10-11 11:11:05 (Time History 2)', or the record's name alone
    when nothing identifies the run."""
    label = found.label()
    if not label:
        return name
    return f'{label[0].upper()}{label[1:]} ({name})'


def group_results(blocks: list[dict[str, Any]],
                  origin_of: Mapping[int, Mapping[str, Any]],
                  objects: Mapping[str, Any],
                  object_groups: Sequence[Mapping[str, Any]] | None = None
                  ) -> list[dict[str, Any]]:
    """The page laid out by `skeleton.arrange`, with its Results section
    grouped by run.

    `origin_of` maps each built block (by `id`) to the report block it
    was built from. The blocks under the Results head are read in their
    generated sections; a section whose figures all come from one run
    goes under that run's heading (a text block carrying `run_head`, for
    `finish`), runs in the order they were run; a section that draws on
    no run, or on several, stays first. The checks (feature
    `check_blocks`, their blocks still carrying 'check_head') stay where
    that feature puts them, closing the section. Nothing is grouped when
    no section comes from a run, or when the only run is one nothing
    identifies.
    """
    groups = _groups(objects, object_groups)
    start = next((k for k, block in enumerate(blocks)
                  if block.get('skeleton', {}).get('key') == 'results'), None)
    if start is None:
        return blocks
    end = next((k for k in range(start + 1, len(blocks))
                if 'skeleton' in blocks[k]), len(blocks))
    under = blocks[start + 1:end]
    origins = [origin_of.get(id(block), {}) for block in under]
    tagged: list[tuple[list[int], str | None]] = []
    closing: list[int] = []
    for section in _sections(origins):
        if any('check_head' in under[position] for position in section):
            # the checks close the Results whatever the runs
            closing.extend(section)
            continue
        found: set[str] = set()
        for position in section:
            found |= runs_of(origins[position], objects, groups)
        tagged.append((section, next(iter(found)) if len(found) == 1
                       else None))
    present = [run for _section, run in tagged if run is not None]
    if not present:
        return blocks
    ordered, found = run_order(objects, present, groups)
    if len(ordered) == 1 and not found[ordered[0]].known:
        return blocks
    laid: list[dict[str, Any]] = []
    for section, run in tagged:
        if run is None:
            laid.extend(under[position] for position in section)
    for run in ordered:
        laid.append({'kind': 'text', 'text': '',
                     'run_head': heading(run, found[run])})
        for section, owner in tagged:
            if owner == run:
                laid.extend(under[position] for position in section)
    laid.extend(under[position] for position in closing)
    return blocks[:start + 1] + laid + blocks[end:]


def finish(blocks: Sequence[dict[str, Any]]) -> None:
    """Draw each run heading `group_results` made, and the drafted
    chronology `mark_drafted` put on the Chronology head, in place."""
    from .markdown import to_html

    for block in blocks:
        title = block.pop('run_head', None)
        if title is not None:
            block['html'] = f'<h3 class="run">{html.escape(title)}</h3>'
        offered = block.pop('drafted', None)
        if offered is not None:
            block['html'] = (block.get('html', '')
                             + f'<div class="drafted"><strong>{DRAFTED}'
                             f'</strong>{to_html(offered, tables=True)}</div>')


# ---- the drafted chronology ---------------------------------------------
#
# The chronology is one of the sections only the author writes. A draft
# made from the records is offered beside the section's words, marked as
# drafted, and the section stays owed until the author accepts it or
# writes it in their own words: it never fills the section by itself.

#: what a drafted chronology is marked with on the editing page
DRAFTED = 'DRAFTED FROM THE RECORDS, NOT YET ACCEPTED'

#: the section the draft is offered to
SECTION = 'chronology'


def offer(report: Any, text: str) -> int:
    """Offer `text` as the drafted chronology of `report`, and return the
    index of the Chronology section's block.

    The draft is kept on the section's block as 'draft', apart from the
    section's words, which do not change: the section stays owed, and
    the editing page shows the draft under the owed card, marked as
    drafted. A new draft replaces the one on offer.
    """
    from .skeleton import write

    if not isinstance(text, str):
        raise TypeError(f'a drafted chronology is text, not '
                        f'{type(text).__name__}')
    index = write(report, SECTION)
    report.blocks[index]['draft'] = text
    return index


def drafted(report: Any) -> str:
    """The chronology on offer, '' when there is none."""
    from .skeleton import blocks_of

    return next((str(report.blocks[index].get('draft', ''))
                 for index in blocks_of(report, SECTION)
                 if report.blocks[index].get('draft')), '')


def accept(report: Any, text: str | None = None) -> int:
    """Make the chronology on offer the section's words, or `text`, the
    draft as the author edited it, and return the block's index. The
    draft is no longer on offer afterwards.

    Raises `ValueError` when there is no draft on offer and no `text`.
    """
    from .skeleton import blocks_of, write

    offered = drafted(report)
    if text is None and not offered:
        raise ValueError('no drafted chronology is on offer to accept; draft '
                         'one from the runs first')
    index = write(report, SECTION, offered if text is None else text)
    for held in blocks_of(report, SECTION):
        report.blocks[held].pop('draft', None)
    return index


def mark_drafted(blocks: Sequence[dict[str, Any]], report: Any) -> None:
    """Put the chronology on offer on the Chronology head the skeleton
    laid out, while that section is owed, for `finish` to draw marked
    as drafted. Once the author has written the section, a draft left
    on its block is not shown."""
    offered = drafted(report)
    if not offered:
        return
    for block in blocks:
        meta = block.get('skeleton', {})
        if meta.get('key') == SECTION and meta.get('state') == 'owed':
            block['drafted'] = offered


# ---- the Results section, repeated for each run ---------------------------

def _lineage(objects: Mapping[str, Any], name: str) -> list[tuple[str, Any]]:
    """How `name` was computed from its run: the (verb, parameters) of
    each step from the run's history to it, first step first."""
    provenance = _provenance(objects)
    steps, seen = [], set()
    root = root_of(objects, name)
    while name != root and name not in seen and name in provenance:
        seen.add(name)
        record = provenance[name]
        steps.append((record.get('verb'), record.get('params')))
        name = record.get('source')
    return steps[::-1]


def counterpart(objects: Mapping[str, Any], name: str, run: str
                ) -> tuple[str | None, str]:
    """The object computed from run `run` the way `name` was computed from
    its own run: same kind, same steps with the same parameters. (name,
    '') when there is one; (None, why) when there is none, or more than
    one."""
    if root_of(objects, name) == name:
        return run, ''
    kind = type(objects[name])
    steps = _lineage(objects, name)
    found = [other for other, obj in objects.items()
             if type(obj) is kind and root_of(objects, other) == run
             and _lineage(objects, other) == steps]
    if len(found) == 1:
        return found[0], ''
    if not found:
        return None, (f'nothing computed from {run!r} stands for {name!r}; '
                      'compute it, then repeat')
    return None, (f'{len(found)} objects computed from {run!r} could stand '
                  f'for {name!r} ({", ".join(found)})')


def repeat(report: Any, objects: Mapping[str, Any],
           object_groups: Sequence[Mapping[str, Any]] | None = None
           ) -> list[str]:
    """Add, for each run the report does not draw on yet, the report's
    Results sections for its first run again, bound to that run's own
    objects. Returns a line per run on what was added or why not.

    A section is a generated section of the report (its '## ' heading
    and what follows) that the skeleton places under Results and whose
    figures all come from one run, the first run the report draws on in
    time order. Each copy binds by name the
    object computed from the other run the same way (`counterpart`),
    and its captions name that run; a section with a figure that has no
    counterpart is not copied, and the line says what to compute. The
    copies are placed after the first run's last section, so the
    report's closing text stays last. A check of its own is not copied:
    it is drawn with the checks, not in a section. A run already drawn
    on is left alone, so a second call adds nothing.

    Raises `ValueError` when the report draws on no run, or has no
    Results section drawn from one: there is then nothing to repeat.
    """
    from ..core.report import resolve_binding

    groups = _groups(objects, object_groups)
    blocks = report.blocks
    # a skeleton section's words and a check of its own
    # (`check_blocks.add`) are not part of a generated section
    where = [k for k, block in enumerate(blocks)
             if 'section' not in block and 'check' not in block]
    origins = [blocks[k] for k in where]
    # where the skeleton puts each block: only what falls under Results
    # is repeated (a pass/fail box heads the Summary, a channel table
    # the Instrumentation)
    from .skeleton import _figure, instrumentation, place

    placed = place([(block, _figure(block.get('kind')),
                     _figure(block.get('kind'))
                     and instrumentation(block, objects, groups))
                    for block in origins])
    sections = []
    for section in _sections(origins):
        found: set[str] = set()
        for position in section:
            found |= runs_of(origins[position], objects, groups)
        sections.append(([where[p] for p in section],
                         next(iter(found)) if len(found) == 1 else None,
                         found, placed[section[0]]))
    drawn = set().union(*(found for _s, _r, found, _p in sections)) \
        if sections else set()
    if not drawn:
        raise ValueError(f'report {getattr(report, "title", "")!r} draws on '
                         'no run, so there is nothing to repeat')
    ordered, identities = run_order(objects, histories(objects), groups)
    first = next(run for run in ordered if run in drawn)
    pattern = [indices for indices, run, _f, where_placed in sections
               if run == first and where_placed == 'results']
    if not pattern:
        raise ValueError(f'report {getattr(report, "title", "")!r} has no '
                         'Results section drawn from one run to repeat')
    lines = []
    added: list[dict[str, Any]] = []
    for run in ordered:
        if run in drawn:
            continue
        label = identities[run].label() or run
        made, missing = [], []
        for indices in pattern:
            copies, why = [], ''
            for index in indices:
                block = dict(blocks[index])
                for key in BOUND:
                    if not block.get(key):
                        continue
                    bound = resolve_binding(block[key], objects, groups)
                    if root_of(objects, bound) != first:
                        continue
                    other, why = counterpart(objects, bound, run)
                    if other is None:
                        break
                    block[key] = other
                if why:
                    break
                if block.get('kind') != 'text' and block.get('caption'):
                    block['caption'] = f'{block["caption"]} ({label})'
                copies.append(block)
            if why:
                missing.append(why)
            else:
                made.extend(copies)
        added.extend(made)
        count = sum(1 for block in made if block.get('kind') != 'text')
        line = f'{run}: {count} figure{"s" * (count != 1)} added'
        if missing:
            line += '; not added: ' + '; '.join(dict.fromkeys(missing))
        lines.append(line)
    if added:
        at = max(index for indices in pattern for index in indices) + 1
        for offset, block in enumerate(added):
            report.add(block, at + offset)
    if not lines:
        lines.append('every run is drawn on already; nothing added')
    return lines
