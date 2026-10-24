"""A report plot's caption, drawn under the plot as a block of rows.

A caption saved inside the picture travels with it: a plot pasted into
a document or attached to an email keeps the lines that let a reader
check it (which channel, which run, the line spacing, the window, the
overlap, the degrees of freedom, the requirement). The rows are a
`core.report_caption.RecordCaption`'s, one per line, each led by its
label in bold, so nothing is carried by color.
"""

from __future__ import annotations

import html
import textwrap
from typing import Any


def caption_html(caption: Any, width: int = 110) -> str:
    """The caption's rows as HTML lines for a label: the label in bold,
    then what it says, one row a line, a row longer than `width`
    characters carried on to the next line indented.

    Parameters
    ----------
    caption : RecordCaption
        The caption.
    width : int, default 110
        The characters a line holds before it is carried on.

    Returns
    -------
    str
    """
    # wrapped here rather than by the label: a LabelItem sizes itself
    # to its longest line, and a requirement row with its rms values
    # ran off the right of a 1000-pixel picture
    lines = []
    for label, said in caption.rows():
        pieces = textwrap.wrap(f'{label}: {said}', width) or [f'{label}:']
        head = html.escape(pieces[0])
        bold = html.escape(label) + ':'
        lines.append(f'<b>{bold}</b>{head[len(bold):]}')
        lines.extend('&nbsp;&nbsp;&nbsp;&nbsp;' + html.escape(piece)
                     for piece in pieces[1:])
    return '<br>'.join(lines)


def add_caption(layout: Any, caption: Any, color: Any = None,
                size: str = '9pt') -> Any:
    """Put the caption block under everything already in `layout`, the
    full width of it.

    Parameters
    ----------
    layout : pyqtgraph.GraphicsLayout or GraphicsLayoutWidget
        What the plot was built into.
    caption : RecordCaption
        The caption.
    color : color, optional
        The text color; the theme's foreground is what callers pass.
    size : str, default '9pt'
        The font size.

    Returns
    -------
    pyqtgraph.LabelItem
        The block, marked `is_report_caption` and holding the caption it
        draws as `caption`.
    """
    import pyqtgraph as pg

    grid = getattr(layout, 'ci', layout)
    rows = grid.layout.rowCount()
    columns = max(grid.layout.columnCount(), 1)
    block = pg.LabelItem(justify='left', color=color, size=size)
    block.setText(caption_html(caption))
    block.is_report_caption = True
    block.caption = caption
    grid.addItem(block, row=rows, col=0, colspan=columns)
    return block
