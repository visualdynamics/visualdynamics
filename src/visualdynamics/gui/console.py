"""The console: the session as Python, along the bottom of the window.

A tab, not a menu (Brandon, 2026-08-30): the tab announces that the
console exists, sitting collapsed at the bottom edge, and clicking it
is the whole gesture — expand to read, click again to put it away.
It shows `project.journal` — every verb and settings write of this
sitting as the line that replays it — selectable and copyable, so the
Python-comfortable half of the audience can lift a session straight
into a script while everyone else never has to see it.

Read-only by decision: an input line was designed and dropped
(Brandon, 2026-08-30) — the journal's value is copy-out. PLAN.md,
"The console".
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .drawer_tab import DrawerTab

#: how tall the scrollback opens — enough to read a working stretch of
#: a session without claiming the plot's space
CONSOLE_HEIGHT = 180


class ConsolePanel(QWidget):
    """The scrollback, and the tab that floats over the views' edge.

    The panel holds only the scrollback, so collapsed it is zero rows
    tall — the first cut reserved a full-width strip for the tab and
    the empty strip read as a gray bar across the window (Brandon,
    2026-08-31). The tab is not in any layout: `float_over` overlays
    it on the bottom edge of whatever the console expands under, the
    way a drawer handle sits on the drawer. Nothing built in does
    this — Qt's tab widgets are document tabs and the native macOS
    drawer is long deprecated — so the handle is placed by hand and
    follows its host's resizes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self._host: QWidget | None = None

        # A tab, drawn as one: a flared handle centered on the edge with
        # a chevron either side of the word, to the design Brandon had
        # made (2026-09-03) — the first cut was a styled button with a
        # font triangle, and before that a full-width bar that read as
        # a divider (2026-08-31).
        self.tab: DrawerTab = DrawerTab('Console')
        self.tab.setToolTip(
            'This session as Python — every act as the line that '
            'replays it, copyable into a script')
        self.tab.toggled.connect(self._toggled)

        self.view: QPlainTextEdit = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.view.setFixedHeight(CONSOLE_HEIGHT)
        self.view.hide()
        column.addWidget(self.view)

        #: what the scrollback currently shows, compared by value: the
        #: journal edits its own last line when a drag settles, so the
        #: refresh rewrites on any difference rather than appending
        self._shown: tuple[str, ...] = ()
        self._toggled(False)

    def float_over(self, host: QWidget) -> None:
        """Overlay the tab on `host`'s bottom edge, centered, and keep
        it there through resizes.

        Parameters
        ----------
        host : QWidget
            A plain container holding the views *and* this panel — the
            tab rides the drawer's top edge like a handle. Never a
            QSplitter: a splitter adopts any child widget as a new
            pane, and the first cut's tab became a third panel parked
            at the right edge.
        """
        self._host = host
        self.tab.setParent(host)
        host.installEventFilter(self)
        self.tab.show()
        self._place_tab()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._host and event.type() in (
                QEvent.Type.Resize, QEvent.Type.Show):
            self._place_tab()
        return False

    def _place_tab(self) -> None:
        if self._host is None:
            return
        size = self.tab.sizeHint()
        self.tab.resize(size)
        # just above the drawer: at the host's bottom edge collapsed,
        # riding up on the scrollback when it opens
        drawer = CONSOLE_HEIGHT if self.tab.isChecked() else 0
        self.tab.move((self._host.width() - size.width()) // 2,
                      self._host.height() - drawer - size.height())
        self.tab.raise_()

    def _toggled(self, open_: bool) -> None:
        self.view.setVisible(open_)
        # pinned, not left to the layout: an empty Preferred-policy
        # widget soaks up the splitter's spare space and the console
        # "collapsed" to most of the window
        self.setFixedHeight(CONSOLE_HEIGHT if open_ else 0)
        self._place_tab()

    @property
    def showing(self) -> bool:
        """Whether the scrollback is expanded."""
        return self.view.isVisible()

    def refresh(self, journal: list[str]) -> None:
        """Bring the scrollback up to date with the session's journal.

        Cheap when nothing changed — one tuple comparison — and a full
        rewrite when something did, because the journal is allowed to
        edit its own tail (a drag session settles to one line) and an
        append-only view would keep the superseded write.

        Parameters
        ----------
        journal : list of str
            The project's session journal, shown verbatim.
        """
        lines = tuple(journal)
        if lines == self._shown:
            return
        self._shown = lines
        # a line is shown whole up to a screenful and elided past it,
        # with its length said: a live edit of a controller's target
        # journals a draft of sixty thousand band pairs, megabytes of
        # text, and rendering that twice per drag was the drag's whole
        # cost (2026-09-06). The journal itself keeps the full line;
        # the session script is where it is read whole
        self.view.setPlainText('\n'.join(elided(line) for line in lines))
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())


#: how much of one journal line the console shows before eliding it
SHOWN_CHARACTERS = 2000


def elided(line: str) -> str:
    """A journal line as the console shows it: whole up to
    `SHOWN_CHARACTERS`, else its head and how long it really is."""
    if len(line) <= SHOWN_CHARACTERS:
        return line
    return (f'{line[:SHOWN_CHARACTERS]} … [{len(line):,} characters; the '
            'session script has the whole line]')

