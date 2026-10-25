"""Calibration in date: each sensor against the test date (report
feature `calibration_in_date`).

The first thing a quality signer checks is that every sensor was in
calibration on the test date. The channel table already holds each
channel's expiration; this feature compares it with the dates of the
test (`core.calibration`) and puts the answer where the report's reader
looks for it.

- **Instrumentation.** The channel table gains a Calibration column:
  in date, expired so many days before the test, lapses during the
  test, or no date given. An expired or lapsing channel's Expiration
  and Calibration cells are shaded in the exceedance red. Under the
  table, a paragraph says what was found: how many channels are in
  date, each one out of date by name and by how much, and the channels
  with no date, which were not checked. With no test date entered, or a
  date or cell that cannot be read, it says calibration was not checked
  and why.
- **Setup and Method.** That section is the author's and is never
  generated. `draft` offers its equipment list with each sensor's
  calibration status as Markdown, a table the author may put in the
  section (the editor's Insert Calibration Status button, or
  `Project.report_section` with `Project.report_calibration_text`).

The test dates are kept on the channel-table blocks of the report
(`START`, `END`), written by `set_dates` (`Project.report_test_date`, and
the dates in the editor's pane on the channel table). A template keeps
no dates (`io.report_template`), so a house template never judges the
next test against the last one's date.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..core.calibration import (
    EXPIRED,
    LAPSES,
    CalibrationCheck,
    DateSpan,
    check,
    span_of,
)

#: the feature's key in `report.features`
FEATURE = 'calibration_in_date'

#: where a channel-table block keeps the test's first and last day
START = 'test_date'
END = 'test_end'

#: the column the channel table gains
COLUMN = 'Calibration'

#: the shading of an out-of-date channel's cells: the exceedance red
#: the page already draws a marked cell in
MARK = 'over'


def channel_table_of(block: Mapping[str, Any], objects: Mapping[str, Any],
           object_groups: Sequence[Mapping[str, Any]] | None = None):
    """The channel table a block shows, or None."""
    from ..core.channel_table import ChannelTable
    from ..core.report import resolve_binding

    if block.get('kind') != 'table':
        return None
    name = resolve_binding(block.get('source'), objects, object_groups)
    found = objects.get(name or '')
    return found if isinstance(found, ChannelTable) else None


def table_blocks(report: Any, objects: Mapping[str, Any],
                 object_groups: Sequence[Mapping[str, Any]] | None = None
                 ) -> list[int]:
    """The indices of `report`'s blocks showing a channel table."""
    return [index for index, block in enumerate(report.blocks)
            if channel_table_of(block, objects, object_groups) is not None]


def span_for(block: Mapping[str, Any]) -> DateSpan | None:
    """The test dates a channel-table block keeps, or None.

    Raises `ValueError` when a kept date cannot be read one way (a file
    edited by hand), naming it.
    """
    return span_of(block.get(START), block.get(END))


def _require(report: Any) -> None:
    from .features import enabled

    if FEATURE not in enabled(report):
        raise ValueError(f'report {getattr(report, "title", "")!r} does not '
                         'have Calibration in date enabled')


def _tables_of(report, objects, object_groups) -> list[int]:
    held = table_blocks(report, objects, object_groups)
    if not held:
        raise ValueError(f'report {getattr(report, "title", "")!r} has no '
                         'channel table block, and calibration is read from '
                         'the channel table')
    return held


def set_dates(report: Any, objects: Mapping[str, Any],
              object_groups: Sequence[Mapping[str, Any]] | None = None,
              start: Any = None, end: Any = None) -> DateSpan | None:
    """Set the test's dates on every channel-table block of `report`,
    and return them.

    `start` is the first test day and `end` the last, for a test over
    several days; each a date or a YYYY-MM-DD string. A `start` of None
    or '' clears the dates.

    Refused, before anything changes: a report without the feature
    enabled (the dates would do nothing), a report with no channel
    table block, and dates that cannot be read one way or run
    backwards.

    Raises
    ------
    ValueError
        For a refusal, naming the reason.
    TypeError
        For a date that is neither a date nor a string.
    """
    _require(report)
    held = _tables_of(report, objects, object_groups)
    span = span_of(start, end)
    for index in held:
        block = report.blocks[index]
        if span is None:
            block.pop(START, None)
            block.pop(END, None)
            continue
        block[START] = span.start.isoformat()
        if span.end != span.start:
            block[END] = span.end.isoformat()
        else:
            block.pop(END, None)
    return span


def checked(report: Any, objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None
            ) -> tuple[CalibrationCheck, Any]:
    """(the check, the channel table) for `report`'s first channel-table
    block.

    Raises `ValueError` without the feature enabled, without a channel
    table block, without a test date (there is then nothing to judge
    against, and no answer would read as in date), and for a date or an
    Expiration cell that cannot be read one way.
    """
    _require(report)
    index = _tables_of(report, objects, object_groups)[0]
    block = report.blocks[index]
    span = span_for(block)
    if span is None:
        raise ValueError(f'report {getattr(report, "title", "")!r} has no '
                         'test date, so calibration has nothing to be '
                         'checked against: enter the test date')
    table = channel_table_of(block, objects, object_groups)
    return check(table, span), table


def apply(built: dict[str, Any], block: Mapping[str, Any],
          objects: Mapping[str, Any],
          object_groups: Sequence[Mapping[str, Any]] | None = None
          ) -> dict[str, Any] | list[dict[str, Any]]:
    """A built table block with the calibration added, when it shows a
    channel table: the block with its Calibration column and marks, and
    the paragraph that says what was found. Any other block is returned
    as it was."""
    table = channel_table_of(block, objects, object_groups)
    if table is None:
        return built
    try:
        span = span_for(block)
        found = check(table, span) if span is not None else None
    except ValueError as refused:
        return [built, _paragraph([f'Calibration not checked: {refused}.'])]
    if found is None:
        return [built, _paragraph([('Calibration not checked: no test date '
                                   'is entered.')])]
    expiration = built['headers'].index('Expiration') \
        if 'Expiration' in built['headers'] else None
    built['headers'] = [*built['headers'], COLUMN]
    built['rows'] = [[*row, channel.words]
                     for row, channel in zip(built['rows'], found.channels,
                                             strict=True)]
    width = len(built['headers'])
    marks: list[list[str | None]] = []
    for channel in found.channels:
        row: list[str | None] = [None] * width
        if channel.status in (EXPIRED, LAPSES):
            row[-1] = MARK
            if expiration is not None:
                row[expiration] = MARK
        marks.append(row)
    if found.lapsed:
        built['marks'] = marks
    return [built, _paragraph(found.summary())]


def _paragraph(lines: Sequence[str]) -> dict[str, Any]:
    # the headline as a paragraph, and each further line an item under it
    text = f'**{lines[0]}**'
    if len(lines) > 1:
        text += '\n\n' + '\n'.join(f'- {line}' for line in lines[1:])
    return {'kind': 'text', 'text': text}


def _cell(value: Any) -> str:
    # a pipe inside a cell is escaped, and a line break cannot be in one
    return ' '.join(str(value).split()).replace('|', '\\|')


def draft(report: Any, objects: Mapping[str, Any],
          object_groups: Sequence[Mapping[str, Any]] | None = None) -> str:
    """The Setup and Method section's equipment list, with each sensor's
    calibration status on the test date, as Markdown for the author to
    put in the section: a sentence, a table of the sensors (channel,
    DOF, quantity, make, model, serial number, expiration and status),
    and the lines of the check.

    It is offered, never written in: the section stays the author's.
    Refused as `checked` refuses.
    """
    found, table = checked(report, objects, object_groups)
    if not found.channels:
        return ' '.join(found.summary())
    head = ['Channel', 'DOF', 'Quantity', 'Make', 'Model', 'Serial Number',
            'Expiration', 'Calibration']
    lines = [(f'Sensors and their calibration status on the test '
             f'date{"s" * (found.span.days != 1)}, {found.span}, from the '
             'channel table:'), '',
             '| ' + ' | '.join(head) + ' |',
             '|' + '|'.join('---' for _ in head) + '|']
    for channel in found.channels:
        row = channel.row
        quantity = channel.key[1] if channel.key[1] != 'unknown' else ''
        cells = [channel.channel, channel.key[0], quantity,
                 table['make'][row], table['model'][row],
                 table['serial_number'][row],
                 channel.expiration.isoformat() if channel.expiration
                 else '', channel.words]
        lines.append('| ' + ' | '.join(_cell(cell) for cell in cells) + ' |')
    lines.append('')
    lines.extend(found.summary())
    return '\n'.join(lines)
