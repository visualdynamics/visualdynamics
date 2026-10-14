"""The subset of Markdown a report's text blocks are written in.

It is written here, as the viewers are, so that no third-party code is
carried inside a report. The subset is what a test report uses:
headings with # to ####, **bold**, *italic*, `code`, bullet lists (- or
*), numbered lists (1. or 1)), and paragraphs separated by blank lines.
The text is HTML-escaped first, so HTML typed into a text block is shown
as text, not interpreted.
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


def _inline(text):
    text = html.escape(text, quote=False)
    text = _CODE.sub(r'<code>\1</code>', text)
    text = _BOLD.sub(r'<strong>\1</strong>', text)
    return _ITALIC.sub(r'<em>\1</em>', text)


def to_html(text: str) -> str:
    """The Markdown subset in `text` as an HTML string.

    A heading becomes h2 to h5, since h1 is the report's title. Lines of
    one paragraph are kept as lines, joined with <br>.
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

    for line in (text or '').splitlines():
        stripped = line.strip()
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
