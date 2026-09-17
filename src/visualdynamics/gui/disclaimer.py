"""The alpha notice: read and acknowledged before the window takes input.

Brandon, 2026-09-11: this is an alpha release, and a user should have to
acknowledge, every time the application loads, that its results are to
be checked. So the launcher shows this dialog after the window is built
and before the event loop takes anything else. One enabled button, Quit,
and a checkbox that enables Continue; nothing in the window is reachable
until Continue is pressed. Every launch, by decision — a few seconds,
and the point stays in front of the user for as long as the release is
an alpha. Not in `MainWindow`, so the API and the tests' window fixture
never meet it; the launcher's `--no-disclaimer` flag is for a script
that opens the window and has already read this.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__

TITLE = 'Visual Dynamics — alpha release'

TEXT = (
    f'<p><b>Visual Dynamics {__version__} is an alpha release.</b></p>'
    '<p>It has been checked against known cases, not yet against a broad '
    'set of tests. Numbers it produces — modal parameters, spectra, '
    'levels, comparisons against a specification — are to be verified by '
    'independent means before anything depends on them.</p>'
    '<p>The project file, <code>.vdyn</code>, is provisional. The next '
    'alpha will save projects in the Engineering Sciences Common Data '
    'Format (<code>.escdf</code>), a public standard, and will still '
    'open <code>.vdyn</code> files; the first release that is not an '
    'alpha will not. Open each <code>.vdyn</code> once in that alpha and '
    'save it again.</p>'
    '<p>Anything that looks wrong is worth reporting: '
    '<a href="mailto:contact@visualdynamics.org">contact@visualdynamics.org</a>.</p>'
)

ACKNOWLEDGMENT = 'I understand that results are to be checked'


class Disclaimer(QDialog):
    """The notice as a modal dialog. `exec()` answers Accepted only
    after the checkbox is ticked and Continue pressed; closing the
    dialog any other way is Quit."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.notice: QLabel = QLabel(TEXT)
        self.notice.setWordWrap(True)
        self.notice.setOpenExternalLinks(True)
        self.notice.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction)
        self.notice.setMaximumWidth(520)
        layout.addWidget(self.notice)
        self.acknowledged: QCheckBox = QCheckBox(ACKNOWLEDGMENT)
        layout.addWidget(self.acknowledged)
        self.buttons: QDialogButtonBox = QDialogButtonBox()
        self.continue_button: QPushButton = self.buttons.addButton(
            'Continue', QDialogButtonBox.ButtonRole.AcceptRole)
        self.quit_button: QPushButton = self.buttons.addButton(
            'Quit', QDialogButtonBox.ButtonRole.RejectRole)
        self.continue_button.setEnabled(False)
        self.continue_button.setDefault(True)
        self.acknowledged.toggled.connect(self.continue_button.setEnabled)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def accept(self) -> None:
        # the keyboard can reach Continue while it is disabled (Enter on
        # the default button); the acknowledgment is what admits
        if not self.acknowledged.isChecked():
            return
        super().accept()


def acknowledged(parent: QWidget | None = None) -> bool:
    """Show the notice; True when the user read it and continued."""
    return Disclaimer(parent).exec() == QDialog.DialogCode.Accepted
