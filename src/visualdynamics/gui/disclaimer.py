"""The alpha notice, acknowledged before the window takes input.

The launcher shows this dialog the first time each version is
launched, after the window is built and before the event loop handles
anything else. The version acknowledged is kept in the preference store
(`gui.preferences.settings`), so a later launch of the same version opens
straight to the window, and the next version asks again. Quit is enabled
from the start; Continue is enabled only once the checkbox is ticked,
and nothing in the window can be reached until Continue is pressed.

It is not part of `MainWindow`, so the API and the tests' window fixture
never show it. The launcher's `--no-disclaimer` flag skips it, for a
script that opens the window and has already read the notice.
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
from .preferences import settings

# Brandon, 2026-09-11: this is an alpha release, and a user should have
# to acknowledge that its results are to be checked. It was shown on
# every launch until Brandon, 2026-10-21, found that tiresome: now the
# first launch of each version, which still puts it in front of anyone
# running a release they have not acknowledged.

#: the version whose notice was last acknowledged, in the preference store
KEY = 'acknowledged_version'

TITLE = 'Visual Dynamics — alpha release'

TEXT = (
    f'<p><b>Visual Dynamics {__version__} is an alpha release.</b></p>'
    '<p>It has been checked against known cases, not yet against a broad '
    'set of tests. Numbers it produces — modal parameters, spectra, '
    'levels, comparisons against a specification — are to be verified by '
    'independent means before anything depends on them.</p>'
    '<p>Projects are saved as <code>.vdyn</code> files. While this is '
    'an alpha the <code>.vdyn</code> format may still change; the '
    'promise that a later version opens every earlier file starts with '
    'the first release that is not an alpha.</p>'
    '<p>Anything that looks wrong is worth reporting: '
    '<a href="mailto:contact@visualdynamics.org">contact@visualdynamics.org</a>.</p>'
)

ACKNOWLEDGMENT = 'I understand that results are to be checked'


class Disclaimer(QDialog):
    """The notice as a modal dialog. `exec()` returns Accepted only
    after the checkbox is ticked and Continue is pressed; closing the
    dialog any other way counts as Quit."""

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


def needed() -> bool:
    """Whether this version's notice is still to be acknowledged: True
    on the first launch of a version, False once it was."""
    return str(settings().value(KEY, '')) != __version__


def remember() -> None:
    """Record that this version's notice was acknowledged."""
    store = settings()
    store.setValue(KEY, __version__)
    store.sync()


def acknowledged(parent: QWidget | None = None) -> bool:
    """Show the notice. Returns True when the user ticked the checkbox
    and pressed Continue, and False otherwise."""
    return Disclaimer(parent).exec() == QDialog.DialogCode.Accepted
