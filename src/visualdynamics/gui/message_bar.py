"""The window's status bar, its message text that can be selected and
copied.

Qt's status bar paints a message itself, as a picture of the text, so a
refusal or a reading could be read but not copied into a note or a
report. Here the message is a label that can be selected with the mouse
and copied with Cmd+C (Ctrl+C elsewhere) or its right-click menu, and
`showMessage`, `currentMessage` and `clearMessage` keep their meanings,
so nothing that writes or reads the bar changes. A message no longer
hides the bar's left-hand widgets either, which Qt's own does.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QStatusBar, QWidget

# Brandon, 2026-10-20: "is it possible to make it so I can highlight the
# text in the bar at the bottom so I can copy it with command + c?"


class MessageBar(QStatusBar):
    """A status bar whose message is selectable text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._message = ''
        self.label: QLabel = QLabel()
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                           | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        # a long message is cut at the bar's edge rather than widening the
        # window; the whole of it is in the tooltip and in what is copied
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.label.setMinimumWidth(0)
        self.addWidget(self.label, 1)

    def showMessage(self, text: str, timeout: int = 0) -> None:
        """Show `text` until the next message; `timeout` is not used."""
        self._message = str(text)
        self.label.setText(self._message)
        self.label.setToolTip(self._message)
        self.messageChanged.emit(self._message)

    def currentMessage(self) -> str:
        """The message shown."""
        return self._message

    def clearMessage(self) -> None:
        """Show no message."""
        self.showMessage('')
