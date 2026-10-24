"""Data index: the files a report rests on, in the report (report
feature `data_index`).

A report is the argument and the record is the evidence, so the report
says exactly where the evidence is. With the feature on, the report
carries the project's data index (`Project.data_index`): a table of
every file by name, what it holds, its format, its size in bytes, its
SHA-256 checksum, whether it is attached or referenced, and what the
last check found. Under it go a line saying when the files were last
checked and how many matched, a line naming the program's version and
the project file's own checksum, the analysis recipe (each derived
object, the verb that computed it and what from, read off the
project's provenance), and each attached text file printed whole: a
CSV as a table, anything else as it was written.

What stays the author's: where the archive is, who keeps it, and what
the contract says travels with the report. With the skeleton sections
on, the index is placed under section 12, Record and Data Package,
after the author's words, and the section is owed until the author
writes them; the program never writes them. Without the skeleton the
index closes the report under its own heading.

Everything here is read from the index as it stands; nothing is hashed
while a page is drawn. The check (`Project.verify_data_index`) is run
by Export before the page is written, off the event loop in the app.
"""

from __future__ import annotations

import csv
import html
import io
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..core.data_index import CHANGED, MATCHES, NOT_FOUND, DataIndex

#: the feature's key in `report.features`
FEATURE = 'data_index'

#: the index table's columns
HEADERS = ('File', 'Holds', 'Format', 'Bytes', 'SHA-256', 'Role', 'Check')

#: the index table's caption
CAPTION = ('Data index: every file this report rests on, with what it '
           'holds, its size and its SHA-256 checksum.')

#: the analysis recipe's columns and caption
RECIPE_HEADERS = ('Object', 'Computed by', 'From')
RECIPE_CAPTION = ('Analysis recipe: each derived object, the verb that '
                  'computed it and what from. The settings each used are '
                  'recorded in the project file.')

#: what the Check column says. A failed check is in capitals, so it
#: reads in a monochrome copy as well as by its color.
CHECK_WORDS = {'': 'not checked', MATCHES: 'matches', CHANGED: 'CHANGED',
               NOT_FOUND: 'NOT FOUND'}

#: what the page says when nothing is indexed: an empty table would
#: read as a complete one
EMPTY = ('The data index is empty: no file has been imported into this '
         'project or added to its index.')

#: what the page says when it is drawn from objects alone, without the
#: project that holds the index
UNKNOWN = ('This page was drawn from the objects alone, without the '
           "project's data index.")

#: the heading the index goes under when the skeleton is off
HEADING = 'Record and Data Package'


def _when(stamp: str) -> str:
    """'2026-10-24 14:02 UTC' from an index's ISO time; the stamp as it
    is when it cannot be read."""
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    return moment.strftime('%Y-%m-%d %H:%M UTC')


def _check(entry: Any) -> str:
    word = CHECK_WORDS[entry.status]
    if entry.found and entry.status in (MATCHES, CHANGED):
        word += ', found by name in another folder'
    return word


def _mark(status: str) -> str:
    return {CHANGED: 'changed', NOT_FOUND: 'changed',
            '': 'unchecked'}.get(status, '')


def table(index: DataIndex) -> dict[str, Any]:
    """The index as a page table block, one row per file in the order
    indexed. The SHA-256 cells are marked to be set in a fixed-width
    face, and a check that did not match is marked as a failure."""
    rows, marks = [], []
    for entry in index:
        rows.append([entry.name, entry.holds, entry.format or '',
                     f'{entry.size:,}', entry.sha256, entry.role,
                     _check(entry)])
        marks.append(['', '', '', '', 'hash', '', _mark(entry.status)])
    return {'kind': 'table', 'caption': CAPTION, 'headers': list(HEADERS),
            'rows': rows, 'marks': marks}


def checked(index: DataIndex) -> str:
    """One sentence on the last check: when, and how many files matched,
    changed or were not found; or that the files have not been checked
    since they were indexed."""
    entries = list(index)
    done = [entry for entry in entries if entry.status]
    if not done:
        return ('Not checked: each file was hashed when it was indexed and '
                'has not been checked against its checksum since.')
    when = max(entry.checked for entry in done)
    counts = {status: sum(entry.status == status for entry in done)
              for status in (MATCHES, CHANGED, NOT_FOUND)}
    one = len(entries) == 1
    parts = [(f'{counts[MATCHES]} of {len(entries)} '
              f'file{"" if one else "s"} match{"es" if one else ""} '
              'the checksum indexed')]
    if counts[CHANGED]:
        parts.append(f'{counts[CHANGED]} changed')
    if counts[NOT_FOUND]:
        parts.append(f'{counts[NOT_FOUND]} not found')
    unchecked = len(entries) - len(done)
    if unchecked:
        parts.append(f'{unchecked} indexed since, not checked')
    return f'Checked {_when(when)}: {"; ".join(parts)}.'


def analysis(index: DataIndex, project_file_known: bool) -> str:
    """One sentence on the analysis: the program's version, and the
    project file's name, size and checksum as the last check found it,
    or that the project is not in a file yet."""
    from .. import __version__

    said = f'Analysis: Visual Dynamics {__version__}.'
    found = index.project_file
    if found is not None:
        name, size, sha, _when_hashed = found
        return (f'{said} Project file: {name}, {size:,} bytes, SHA-256 '
                f'{sha}, as it stood at the check; changes made since it was '
                'saved are not in it.')
    if project_file_known:
        return f'{said} The project file has not been checked.'
    return (f'{said} The project has not been saved to a file, so the '
            'analysis is not yet part of the record.')


def recipe(provenance: Mapping[str, Mapping[str, Any]],
           objects: Mapping[str, Any]) -> dict[str, Any] | None:
    """The analysis recipe as a page table block: each derived object
    still in the project, in the project's order, with the verb that
    computed it and its source. None when nothing is derived."""
    rows = [[name, str(provenance[name].get('verb', '')),
             str(provenance[name].get('source') or '')]
            for name in objects if name in provenance]
    if not rows:
        return None
    return {'kind': 'table', 'caption': RECIPE_CAPTION,
            'headers': list(RECIPE_HEADERS), 'rows': rows}


def _delimiter(name: str) -> str | None:
    suffix = os.path.splitext(name)[1].lower()
    return {'.csv': ',', '.tsv': '\t'}.get(suffix)


def attachment(entry: Any) -> dict[str, Any] | None:
    """An attached text file's contents as a page block: a CSV or TSV as
    a table, its first row the header, when every row has the header's
    width; anything else, a ragged CSV included, as preformatted text,
    as it was written, since a short row cannot be placed in its
    columns one way. None for an entry with no kept text."""
    if entry.role != 'attached' or not entry.text:
        return None
    caption = f'Attachment: {entry.name}, {entry.holds}.'
    delimiter = _delimiter(entry.name)
    if delimiter is not None:
        rows = [row for row in csv.reader(io.StringIO(entry.text),
                                          delimiter=delimiter) if row]
        if rows and all(len(row) == len(rows[0]) for row in rows):
            return {'kind': 'table', 'caption': caption,
                    'headers': rows[0], 'rows': rows[1:]}
    return _html(f'<p class="attachment">{html.escape(caption)}</p>'
                 f'<pre class="attachment">{html.escape(entry.text)}</pre>')


def _html(markup: str) -> dict[str, Any]:
    # a text block already drawn: the renderer leaves its HTML alone
    return {'kind': 'text', 'html': markup}


def page_blocks(objects: Mapping[str, Any], heading: bool = False,
                index: DataIndex | None = None) -> list[dict[str, Any]]:
    """The page blocks the feature adds, in order.

    `objects` is what the report was rendered from: a `Project`, whose
    `data_index`, `provenance` and `file` are read, or a plain mapping,
    for which `index` gives the index; without one, the page says it
    was drawn without the index rather than show it empty. With
    `heading`, the blocks open with the section's own heading, for a
    report without the skeleton.
    """
    if index is None:
        index = getattr(objects, 'data_index', None)
    out: list[dict[str, Any]] = []
    if heading:
        out.append(_html(f'<h2>{html.escape(HEADING)}</h2>'))
    if index is None:
        # an empty index would read as nothing imported, which this
        # page cannot know
        out.append(_html(f'<p class="index-empty">{html.escape(UNKNOWN)}'
                         '</p>'))
        return out
    if not len(index):
        out.append(_html(f'<p class="index-empty">{html.escape(EMPTY)}</p>'))
    else:
        out.append(table(index))
        out.append(_html(f'<p class="index-check">'
                         f'{html.escape(checked(index))}</p>'))
    out.append(_html(f'<p class="index-analysis">{html.escape(analysis(
        index, bool(getattr(objects, "file", None))))}</p>'))
    made = recipe(getattr(objects, 'provenance', None) or {}, objects)
    if made is not None:
        out.append(made)
    for entry in index:
        printed = attachment(entry)
        if printed is not None:
            out.append(printed)
    return out
