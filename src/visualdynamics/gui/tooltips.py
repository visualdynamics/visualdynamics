"""Long tooltips wrap.

Qt draws a plain-text tooltip on one line however long it is, and the
bar's tooltips (what each icon does, in a sentence) ran across most of
the screen. A tooltip given as rich text wraps, so a plain tooltip wider
than `WRAP_CHARACTERS` is stored as a paragraph of that width when it is
set. Short tooltips, and the rich-text ones the app writes itself, are
stored as they are.

The wrapping happens in `setToolTip` itself, on widgets, actions and the
items of trees, tables and lists (`install`), because every tooltip the
app has is set through one of those. A bar button takes its action's
tooltip, so it wraps with it.
"""

from __future__ import annotations

import html
from typing import Any

from PySide6.QtGui import QAction, QFontMetrics
from PySide6.QtGui import Qt as QtGui
from PySide6.QtWidgets import (
    QApplication,
    QListWidgetItem,
    QTableWidgetItem,
    QToolTip,
    QTreeWidgetItem,
    QWidget,
)

# A filter on the application that rewrapped each tooltip as it was
# about to show was tried first (2026-10-17): it ran in Python for every
# event of every object, PySide had to wrap objects half torn down (the
# report view's web engine widgets among them), and the suite's workers
# crashed with segmentation faults. Wrapping at the setter costs nothing
# while the app runs.

#: the width a tooltip wraps at, in average characters of the tooltip
#: font: about the measure of a line in a book, so a sentence-long
#: tooltip takes two or three lines rather than the screen's width
WRAP_CHARACTERS = 60

#: the classes whose `setToolTip` wraps, once `install` has run
WRAPPED = (QWidget, QAction, QTreeWidgetItem, QTableWidgetItem, QListWidgetItem)


def wrapped(text: str, width: int) -> str:
    """`text` as rich text that wraps at `width` pixels, its own line
    breaks kept."""
    body = html.escape(text).replace('\n', '<br>')
    return f'<table width="{width}"><tr><td>{body}</td></tr></table>'


def plain(text: str) -> str:
    """A tooltip's words, whether or not it was wrapped: what a reader
    sees, for the code and tests that read tooltips back."""
    head = '<table width="'
    if text.startswith(head) and text.endswith('</td></tr></table>'):
        body = text[text.index('<td>') + 4:-len('</td></tr></table>')]
        return html.unescape(body.replace('<br>', '\n'))
    return text


def wrap_if_long(text: Any) -> Any:
    """`text` as a wrapping paragraph when it is plain text wider than
    `WRAP_CHARACTERS`; otherwise `text` unchanged."""
    if not isinstance(text, str) or not text or QtGui.mightBeRichText(text):
        return text
    if QApplication.instance() is None:
        return text
    metrics = QFontMetrics(QToolTip.font())
    width = WRAP_CHARACTERS * metrics.averageCharWidth()
    if max(metrics.horizontalAdvance(line) for line in text.split('\n')) <= width:
        return text
    return wrapped(text, width)


def _wrapping(original):
    def set_tooltip(self, text, *args):
        original(self, wrap_if_long(text), *args)
    set_tooltip.wraps_long_tooltips = True
    return set_tooltip


def install() -> None:
    """Make every long tooltip set from now on wrap; once only."""
    for cls in WRAPPED:
        if not getattr(cls.setToolTip, 'wraps_long_tooltips', False):
            cls.setToolTip = _wrapping(cls.setToolTip)
