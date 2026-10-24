"""Skeleton sections: every section a test report needs, in order
(report feature `skeleton_sections`).

A test report is read by people who look for each part of it in the
same place every time, so a section that is missing tells them nothing:
they cannot tell "nothing happened" from "nobody wrote it down". With
the feature on, the report is laid out on the twelve sections of
`SECTIONS` and closed with the end mark, whatever the project holds.

What Visual Dynamics generates falls under the sections it answers
(`place`): the pass/fail box and the opening description under the
Summary, the geometry, the photographs and the channel table under
Instrumentation, the closing reading guide under the Conclusion, and
every other figure and its text under Results.

What only the author can write (the article, the setup and method, the
chronology, the deviations, the discussion, the conclusion's wording)
is never generated. Each such section holds the author's own words, a
text block marked with the section's key (`{'kind': 'text', 'text':
..., 'section': key}`), and until that block says something the page
shows the section as owed: NOT YET WRITTEN, with what belongs there.
A section with nothing to report says so in one step (`write(...,
none=True)`): "No deviations. No anomalies. No nonconformances." The
discussion carries who interpreted (`by`), and the conclusion carries
the sentence that the results relate only to the item tested.

`check` lists what is still owed, for the export to ask about and for
a script (`Project.report_check`). `write` is the one place a
section's words change; the editor's pane and `Project.report_section`
both call it.

The arrangement is made on the page being rendered (`arrange`,
`finish`), not on the report: the report's own blocks keep their order,
a section block is drawn only while the feature is on, and turning the
feature off gives back the page it was.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# P18 of the report-features plan (Kevin Cross, 2026-10-24): the
# thirteen-section skeleton, owed sections marked rather than invented,
# "None" written rather than a section left out, and a check before
# export. The skeleton's order is the order readers look in.

#: the feature's key in `report.features`
FEATURE = 'skeleton_sections'

#: what an owed section says on the page, in capitals so it is read in
#: a monochrome copy as well as by its frame
OWED = 'NOT YET WRITTEN'

#: the conclusion's fixed sentence: a test on one article says what that
#: article did; what it means for the design is the program's argument
TESTED = 'The results relate only to the item tested.'

#: the last line of the report, so a reader holding a stack of pages
#: knows whether all of it is there
END = 'End of report.'

#: why a section is listed by `check`
NOT_WRITTEN = 'not yet written'
EMPTY = 'nothing in it'
UNATTRIBUTED = 'interpretation not attributed (say by whom)'


@dataclass(frozen=True)
class Section:
    """One section of the skeleton.

    `required` sections are the author's to write and are owed until
    they say something. The others (Instrumentation, Results) are
    filled by what the project generates and are owed only when nothing
    falls under them. `none` is what one step writes when the section
    has nothing to report, or '' for a section that always has
    something (every report describes the article it tested).
    """

    key: str
    number: int
    title: str
    description: str
    required: bool = True
    none: str = ''

    @property
    def heading(self) -> str:
        """'9. Deviations, Anomalies and Nonconformances'."""
        return f'{self.number}. {self.title}'


#: the sections, in the order they appear
SECTIONS: tuple[Section, ...] = (
    Section('identification', 1, 'Identification',
            'The title, report number and revision, page n of N and the '
            "laboratory; on the first page the laboratory's address, where "
            'the work was done, the customer and contact, the dates of '
            'receipt, test and issue, and who authorized the report.'),
    Section('summary', 2, 'Summary',
            'What was tested and why, the degree of success against each '
            'objective, the result against each requirement as met or not '
            'met, the deviations by number, and the open anomalies.'),
    Section('requirement', 3, 'Requirement',
            'The governing document and its revision, the plan and '
            'procedure with their revisions, the tolerance clause, the '
            'failure definition, the completion criteria, and any data the '
            "customer supplied, identified as the customer's."),
    Section('article', 4, 'Test Article',
            'Description, part and serial identification, configuration, '
            'condition on receipt, and any adjustment the plan allowed to '
            'the article or its fixture.'),
    Section('setup', 5, 'Setup and Method',
            'The facility and shaker system, the fixture and its surveys, '
            'the control strategy, the settings, and the test equipment '
            "with each item's calibration status."),
    Section('instrumentation', 6, 'Instrumentation',
            'The channel table, with any sensor averaging stated.',
            required=False),
    Section('chronology', 7, 'Chronology',
            'The runs in order with their levels, durations and outcomes, '
            'every interruption and its recovery, and every failure, from '
            'the log.'),
    Section('results', 8, 'Results',
            'The control result against the tolerance by channel, the '
            'responses against prediction, the functional data, the '
            'surveys, the inspections and the test conditions.',
            required=False),
    Section('deviations', 9, 'Deviations, Anomalies and Nonconformances',
            'Each one with its fact, cause, effect, and approval or '
            'disposition.',
            none='No deviations. No anomalies. No nonconformances.'),
    Section('discussion', 10, 'Discussion and Interpretation',
            'What the results mean against the objectives, marked as '
            'interpretation, and by whom.',
            none='No interpretation is offered beyond the results.'),
    Section('conclusion', 11, 'Conclusion',
            'The statement of conformity, the requirement it covers, and '
            'the rule by which it was decided.'),
    Section('record', 12, 'Record and Data Package',
            'What is attached, what is referenced, and where the data are '
            'kept.'),
)

#: the sections by key
BY_KEY: dict[str, Section] = {section.key: section for section in SECTIONS}

#: the one section that carries who interpreted
ATTRIBUTED = 'discussion'


@dataclass(frozen=True)
class Owed:
    """A section `check` found still owed: which, and why."""

    key: str
    number: int
    title: str
    reason: str

    def __str__(self) -> str:
        return f'{self.number}. {self.title}: {self.reason}'


def section(key: Any) -> Section:
    """The section named `key`.

    Raises `ValueError` for a key that is not a section, naming the
    sections, and `TypeError` for one that is not a string.
    """
    if not isinstance(key, str):
        raise TypeError(f'a skeleton section is named by its key, a str, '
                        f'not {type(key).__name__}')
    found = BY_KEY.get(key)
    if found is None:
        raise ValueError(f'unknown skeleton section {key!r}; the sections '
                         f'are: {", ".join(BY_KEY)}')
    return found


def blocks_of(report: Any, key: str) -> list[int]:
    """The indices of `report`'s blocks holding section `key`'s words,
    in order. Normally one; none until the section is first written."""
    return [index for index, block in enumerate(report.blocks)
            if block.get('section') == key]


def words(report: Any, key: str) -> str:
    """What the author has written in section `key`, stripped; '' while
    it is owed."""
    texts = (str(report.blocks[index].get('text', '')).strip()
             for index in blocks_of(report, key))
    return '\n\n'.join(text for text in texts if text)


def write(report: Any, key: str, text: str | None = None, *,
          none: bool = False, by: str | None = None) -> int:
    """Set what section `key` says, and return the index of its block.

    The section's block is added at the end of the report the first
    time (where it sits among the blocks does not matter: the page lays
    the sections out in order). `text` replaces the section's words.
    `none` writes the section's own "nothing to report" wording, as one
    step. `by` is who made the interpretation, for the discussion only;
    an empty `by` clears it. With none of them, the block is made if it
    is missing and nothing else changes.

    Refused, before anything changes: an unknown section; `text` and
    `none` together; `none` for a section that always has something to
    say; `by` for a section other than the discussion.

    Raises
    ------
    ValueError
        For a refusal; the message names the section and the reason.
    TypeError
        When `key`, `text` or `by` is not a string.
    """
    found = section(key)
    if text is not None and not isinstance(text, str):
        raise TypeError(f'{found.heading}: the text is a str, not '
                        f'{type(text).__name__}')
    if by is not None and not isinstance(by, str):
        raise TypeError(f'{found.heading}: by is a str, not '
                        f'{type(by).__name__}')
    if none and text is not None:
        raise ValueError(f'{found.heading}: give text or none, not both')
    if none and not found.none:
        raise ValueError(f'{found.heading} cannot be set to None: every '
                         'report has something to say here')
    if by is not None and key != ATTRIBUTED:
        raise ValueError(f'{found.heading} has no interpretation to '
                         f'attribute; by belongs to '
                         f'{BY_KEY[ATTRIBUTED].heading}')
    held = blocks_of(report, key)
    if held:
        index = held[0]
    else:
        report.add({'kind': 'text', 'text': '', 'section': key})
        index = report.num_blocks - 1
    block = report.blocks[index]
    if none:
        block['text'] = found.none
    elif text is not None:
        block['text'] = text
    if by is not None:
        if by.strip():
            block['by'] = by.strip()
        else:
            block.pop('by', None)
    return index


def state(found: Section, text: str, by: str = '',
          generated: bool = False) -> str:
    """'written', 'owed' or 'quiet' (a filled section with nothing of
    the author's in it, which shows its heading only).

    `text` is what the author wrote, `by` who interpreted, and
    `generated` whether anything the project generates falls under the
    section. The discussion, written without saying by whom, is
    'unattributed'.
    """
    if text:
        if found.key == ATTRIBUTED and text != found.none and not by:
            return 'unattributed'
        return 'written'
    if found.required or not generated:
        return 'owed'
    return 'quiet'


def _figure(kind: Any) -> bool:
    # what reads as a figure for laying sections out: not text, not an
    # unbound card (which the reading page leaves out), and not the
    # pass/fail box, which heads the report rather than illustrating it
    return kind not in ('text', 'unbound', 'verdict')


def _heading(block: Mapping[str, Any]) -> bool:
    # the same sections `core.report.prune_unbound` reads: a text block
    # opening with a level-two heading starts one
    return (block.get('kind') == 'text'
            and str(block.get('text', '')).lstrip().startswith('## '))


def instrumentation(block: Mapping[str, Any], objects: Mapping[str, Any],
                    object_groups: Sequence[Mapping[str, Any]] | None = None
                    ) -> bool:
    """Whether a block shows what was tested and how it was instrumented:
    a photograph, the geometry or a DOF scene (a scene with no shapes),
    or a channel table."""
    from ..core.channel_table import ChannelTable
    from ..core.report import resolve_binding

    kind = block.get('kind')
    if kind == 'photo':
        return True
    if kind == 'scene':
        return not block.get('shapes')
    if kind == 'table':
        name = resolve_binding(block.get('source'), objects, object_groups)
        return isinstance(objects.get(name or ''), ChannelTable)
    return False


def place(entries: Sequence[tuple[Mapping[str, Any], bool, bool]]
          ) -> list[str]:
    """The section each generated block falls under.

    `entries` is the report's blocks in their order, other than the
    section blocks, each as (block, is a drawn figure, shows the
    instrumentation). The blocks are read in the sections their `## `
    headings make. A section holding the pass/fail box goes under the
    Summary, and so does a section of text only that comes before every
    figure (the description that opens a generated report). A section
    with a drawn photograph, geometry, DOF scene or channel table goes
    under Instrumentation. A section of text only after every figure
    (the reading guide that closes it) goes under the Conclusion.
    Everything else is Results.
    """
    groups: list[list[int]] = []
    for position, (block, _figure, _shows) in enumerate(entries):
        if _heading(block) or not groups:
            groups.append([])
        groups[-1].append(position)
    figures = [position for position, (_b, figure, _s) in enumerate(entries)
               if figure]
    first = figures[0] if figures else len(entries)
    last = figures[-1] if figures else -1
    out = [''] * len(entries)
    for group in groups:
        members = [entries[position] for position in group]
        if any(block.get('kind') == 'verdict' for block, _f, _s in members):
            where = 'summary'
        elif any(figure and shows for _b, figure, shows in members):
            where = 'instrumentation'
        elif not any(figure for _b, figure, _s in members) \
                and group[-1] < first:
            where = 'summary'
        elif not any(figure for _b, figure, _s in members) \
                and group[0] > last:
            where = 'conclusion'
        else:
            where = 'results'
        for position in group:
            out[position] = where
    return out


def check(report: Any, objects: Mapping[str, Any],
          object_groups: Sequence[Mapping[str, Any]] | None = None
          ) -> tuple[Owed, ...]:
    """The sections of `report` still owed, in order: each section the
    author has not written, a filled section with nothing in it, and a
    discussion that does not say by whom. Empty when nothing is owed.

    A generated block counts as filling its section when its bindings
    resolve in `objects` (a text block always does), which is what the
    reading page draws.

    Raises `ValueError` when the report does not have the skeleton
    sections enabled: there is then no skeleton to check it against,
    and an empty list would read as a report complete.
    """
    from ..core.report import resolve_binding
    from .features import enabled

    if FEATURE not in enabled(report):
        raise ValueError(f'report {getattr(report, "title", "")!r} does not '
                         'have Skeleton sections enabled, so there is no '
                         'skeleton to check it against')
    keys = ('source', 'geometry', 'dofs_source', 'shapes', 'measured',
            'specification')

    def drawn(block):
        needed = [block.get(key) for key in keys if block.get(key)]
        return all(resolve_binding(name, objects, object_groups) in objects
                   for name in needed)

    entries = []
    for block in report.blocks:
        if 'section' in block or not drawn(block):
            continue
        figure = _figure(block.get('kind'))
        entries.append((block, figure, figure and instrumentation(
            block, objects, object_groups)))
    filled = set(place(entries))
    owed = []
    for found in SECTIONS:
        held = blocks_of(report, found.key)
        by = str(report.blocks[held[0]].get('by', '')) if held else ''
        now = state(found, words(report, found.key), by,
                    found.key in filled)
        if now in ('owed', 'unattributed'):
            reason = (UNATTRIBUTED if now == 'unattributed' else
                      NOT_WRITTEN if found.required else EMPTY)
            owed.append(Owed(found.key, found.number, found.title, reason))
    return tuple(owed)


def arrange(built: Sequence[dict[str, Any]],
            origins: Sequence[Mapping[str, Any]],
            held: Mapping[str, Sequence[tuple[int, Mapping[str, Any],
                                              dict[str, Any]]]],
            objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None,
            edit: bool = False) -> list[dict[str, Any]]:
    """The page's blocks laid out on the skeleton.

    `built` is the page's blocks as built, in the report's order, and
    `origins` the report block each was built from. `held` is each
    section's own blocks, by key, as (index in the report, the block,
    the block built: its text with live references resolved). Returns the blocks in skeleton order: for each section, a
    head (its heading and the author's words, or the owed card) and
    then what falls under it; and last the end mark. A head carries
    `skeleton` for `finish` to draw once the text is rendered. On the
    edit page, a head carries the index of the section's block, or
    'section:<key>' while it has none, so a click on it opens it.
    """
    entries = []
    for block, origin in zip(built, origins, strict=True):
        figure = _figure(block.get('kind'))
        entries.append((origin, figure, figure and instrumentation(
            origin, objects, object_groups)))
    where = place(entries)
    out: list[dict[str, Any]] = []
    for found in SECTIONS:
        mine = [block for block, at in zip(built, where, strict=True)
                if at == found.key]
        own = held.get(found.key, ())
        texts = [str(built_text.get('text', '')).strip()
                 for _index, _block, built_text in own]
        text = '\n\n'.join(text for text in texts if text)
        by = str(own[0][1].get('by', '')) if own else ''
        generated = any(block.get('kind') != 'unbound' for block in mine)
        head: dict[str, Any] = {
            'kind': 'text', 'text': text,
            'skeleton': {'key': found.key, 'by': by,
                         'state': state(found, text, by, generated),
                         # the discussion names who interpreted, unless
                         # it says there is no interpretation
                         'attribute': (found.key == ATTRIBUTED and bool(text)
                                       and text != found.none)}}
        if edit:
            head['index'] = own[0][0] if own else f'section:{found.key}'
        out.append(head)
        out.extend(mine)
    out.append({'kind': 'text', 'text': '', 'skeleton': {'key': 'end'}})
    return out


def finish(blocks: Sequence[dict[str, Any]]) -> None:
    """Draw each head `arrange` made, in place, once its words are HTML:
    the heading, then the author's words, the owed card or nothing;
    the discussion's attribution and the conclusion's sentence."""
    for block in blocks:
        meta = block.pop('skeleton', None)
        if meta is None:
            continue
        if meta['key'] == 'end':
            block['html'] = f'<p class="endmark">{html.escape(END)}</p>'
            continue
        found = BY_KEY[meta['key']]
        parts = [f'<h2 class="skeleton">{html.escape(found.heading)}</h2>']
        written = block.get('html', '')
        if meta['state'] == 'owed':
            parts.append(f'<div class="owed"><strong>{OWED}</strong> '
                         f'{html.escape(found.description)}</div>')
        elif meta['state'] in ('written', 'unattributed'):
            if meta['attribute']:
                who = html.escape(meta['by']) if meta['by'] else \
                    '<span class="owed">not stated</span>'
                parts.append(f'<p class="interpretation">Interpretation, '
                             f'by {who}.</p>')
            parts.append(written)
        if found.key == 'conclusion':
            parts.append(f'<p class="tested">{html.escape(TESTED)}</p>')
        # joined with nothing between: a text block keeps its white
        # space (`.text`), and a line between the heading and the card
        # would draw as a gap
        block['html'] = ''.join(part for part in parts if part)
