"""The subset of Markdown a report's text blocks are written in.

It is written here, as the viewers are, so that no third-party code is
carried inside a report. The subset is what a test report uses:
headings with # to ####, **bold**, *italic*, `code`, bullet lists (- or
*), numbered lists (1. or 1)), and paragraphs separated by blank lines.
The text is HTML-escaped first, so HTML typed into a text block is shown
as text, not interpreted.

Pipe tables are read when asked for (`to_html(..., tables=True)`, which
the report feature `skeleton_sections` does, so a section's author can
type the table a section asks for): a header row, a delimiter row of
dashes with optional colons for alignment, and body rows, each line
opening and closing with `|`. A backslash before a pipe makes it part
of a cell.
"""

from __future__ import annotations

import html
import re

_BOLD = re.compile(r'\*\*(.+?)\*\*')
_ITALIC = re.compile(r'(?<!\*)\*([^*]+)\*(?!\*)')
_CODE = re.compile(r'`([^`]+)`')
_HEADING = re.compile(r'^(#{1,4})\s+(.*)$')
_BULLET = re.compile(r'^[-*]\s+(.*)$')
_NUMBERED = re.compile(r'^\d+[.)]\s+(.*)$')
_ROW = re.compile(r'^\|.*\|$')
_DELIMITER = re.compile(r'^:?-+:?$')
#: what a delimiter cell's colons say
_ALIGN = {(True, False): 'left', (True, True): 'center',
          (False, True): 'right'}


def _inline(text):
    text = html.escape(text, quote=False)
    text = _CODE.sub(r'<code>\1</code>', text)
    text = _BOLD.sub(r'<strong>\1</strong>', text)
    return _ITALIC.sub(r'<em>\1</em>', text)


def _cells(line: str) -> list[str]:
    """A table row's cells, stripped: split on every pipe not escaped,
    the outer pipes dropped."""
    inner = line.strip()[1:-1]
    cells, cell, escaped = [], [], False
    for char in inner:
        if escaped:
            # only a pipe is escaped; any other backslash is kept
            cell.append(char if char == '|' else '\\' + char)
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '|':
            cells.append(''.join(cell).strip())
            cell = []
        else:
            cell.append(char)
    if escaped:
        cell.append('\\')
    cells.append(''.join(cell).strip())
    return cells


def _alignments(line: str) -> list[str] | None:
    """The alignment of each column ('' for none) when `line` is a
    delimiter row, else None."""
    if not _ROW.match(line):
        return None
    cells = _cells(line)
    if not all(_DELIMITER.match(cell) for cell in cells):
        return None
    return [_ALIGN.get((cell.startswith(':'), cell.endswith(':')), '')
            for cell in cells]


def _table(header: list[str], aligns: list[str],
           rows: list[list[str]]) -> str:
    """A pipe table as HTML. A row is cut or padded to the header's
    width, as the header sets the columns."""
    def cell(tag, text, align):
        style = f' style="text-align: {align}"' if align else ''
        return f'<{tag}{style}>{_inline(text)}</{tag}>'

    width = len(header)
    aligns = (aligns + [''] * width)[:width]
    head = ''.join(cell('th', text, align)
                   for text, align in zip(header, aligns, strict=True))
    body = ''.join(
        '<tr>' + ''.join(cell('td', text, align) for text, align in zip(
            (row + [''] * width)[:width], aligns, strict=True)) + '</tr>'
        for row in rows)
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def to_html(text: str, tables: bool = False) -> str:
    """The Markdown subset in `text` as an HTML string.

    A heading becomes h2 to h5, since h1 is the report's title. Lines of
    one paragraph are kept as lines, joined with <br>. With `tables`,
    pipe tables are read; without, a pipe is text like any other.
    """
    out, paragraph, listing = [], [], None

    def flush_paragraph() -> None:
        if paragraph:
            out.append('<p>' + '<br>'.join(paragraph) + '</p>')
            paragraph.clear()

    def flush_list() -> None:
        nonlocal listing
        if listing:
            out.append(f'</{listing}>')
            listing = None

    lines = (text or '').splitlines()
    at = 0
    while at < len(lines):
        line = lines[at]
        at += 1
        stripped = line.strip()
        aligns = (_alignments(lines[at].strip())
                  if tables and at < len(lines) and _ROW.match(stripped)
                  else None)
        if aligns is not None:
            # a header row with its delimiter under it: the table runs
            # while the lines are rows
            flush_paragraph()
            flush_list()
            rows = []
            at += 1
            while at < len(lines) and _ROW.match(lines[at].strip()):
                rows.append(_cells(lines[at]))
                at += 1
            out.append(_table(_cells(stripped), aligns, rows))
            continue
        heading = _HEADING.match(stripped)
        bullet = _BULLET.match(stripped)
        numbered = _NUMBERED.match(stripped)
        if not stripped:
            flush_paragraph()
            flush_list()
        elif heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1)) + 1   # h1 is the report title
            out.append(f'<h{level}>{_inline(heading.group(2))}</h{level}>')
        elif bullet or numbered:
            flush_paragraph()
            wanted = 'ul' if bullet else 'ol'
            if listing != wanted:
                flush_list()
                out.append(f'<{wanted}>')
                listing = wanted
            out.append(f'<li>{_inline((bullet or numbered).group(1))}</li>')
        else:
            flush_list()
            paragraph.append(_inline(stripped))
    flush_paragraph()
    flush_list()
    return '\n'.join(out)
