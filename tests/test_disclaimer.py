"""The alpha notice, acknowledged before the window takes input.

Brandon, 2026-09-11: an alpha release should make the user acknowledge,
every time the application loads, that its results are to be checked.
The dialog has one enabled button, Quit, and a checkbox that enables
Continue; the launcher shows it after the window is built and exits if
it is refused; the API and the tests' window fixture never meet it.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog

from visualdynamics import __version__
from visualdynamics.gui.disclaimer import ACKNOWLEDGMENT, TEXT, Disclaimer


def test_the_version_says_alpha():
    assert 'a' in __version__ and __version__.startswith('0.1.0'), \
        'the stage is in the version itself, so every reader agrees'
    assert __version__ in TEXT and 'alpha release' in TEXT
    assert 'verified by independent means' in TEXT
    assert 'contact@visualdynamics.org' in TEXT


def test_the_notice_says_the_project_file_is_provisional():
    """Brandon, 2026-09-13: release today, with `.vdyn` explicitly
    alpha-only and a dated end — the next alpha writes `.escdf` and
    still reads `.vdyn`, the first non-alpha does not. Said where a
    user reads: here, the downloads page, the README, the guide."""
    assert '.vdyn' in TEXT and 'provisional' in TEXT
    assert '.escdf' in TEXT and 'save it again' in TEXT


def test_continue_needs_the_acknowledgment(qt_app):
    dialog = Disclaimer()
    assert dialog.acknowledged.text() == ACKNOWLEDGMENT
    assert not dialog.continue_button.isEnabled()
    assert dialog.quit_button.isEnabled(), 'the one way out without reading'
    # Enter reaches the default button whether or not it is enabled:
    # the acknowledgment is what admits, not the button's state
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.acknowledged.setChecked(True)
    assert dialog.continue_button.isEnabled()
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_quit_is_a_refusal(qt_app):
    dialog = Disclaimer()
    dialog.acknowledged.setChecked(True)
    dialog.reject()
    assert dialog.result() == QDialog.DialogCode.Rejected


def test_the_launcher_exits_when_the_notice_is_refused(qt_app, monkeypatch):
    """`main` shows the notice after the window is built and returns
    without running the event loop when it is refused; `--no-disclaimer`
    skips it for a script that has read this."""
    from PySide6.QtWidgets import QWidget

    from visualdynamics import gui
    from visualdynamics.gui import disclaimer, main_window

    class Window(QWidget):
        """What `main` builds, without a 3-D view: the real one under
        the offscreen platform is the fixture's `offscreen_3d=True`
        window, and `main` builds the real one."""

        def import_path(self, _path):
            pass

    monkeypatch.setattr(main_window, 'MainWindow', Window)
    shown = []
    monkeypatch.setattr(disclaimer, 'acknowledged',
                        lambda parent=None: (shown.append(parent), False)[1])
    monkeypatch.setattr(qt_app, 'exec', lambda: 99)
    assert gui.main([]) == 0, 'refused: the application exits'
    assert len(shown) == 1 and shown[0] is not None, 'shown over the window'
    monkeypatch.setattr(disclaimer, 'acknowledged',
                        lambda parent=None: (shown.append(parent), True)[1])
    assert gui.main([]) == 99, 'acknowledged: the event loop runs'
    assert gui.main(['--no-disclaimer']) == 99
    assert len(shown) == 2, 'the flag skips the notice'


def test_a_pre_release_sorts_below_its_release():
    from visualdynamics.update import newer, parse_version

    assert parse_version('0.1.0a1') < parse_version('0.1.0')
    assert parse_version('0.1.0a1') < parse_version('0.1.0a2') \
        < parse_version('0.1.0b1') < parse_version('0.1.0rc1') \
        < parse_version('0.1.0') < parse_version('0.1.1a1')
    assert newer('0.1.0a10', '0.1.0a9'), (
        'the stage number is a number: a text compare puts a10 below a9, '
        'and 0.1.0a10 is the first release where that could bite')
    assert newer('0.1.0', '0.1.0a1'), 'the first non-alpha is the upgrade'
    assert not newer('0.1.0a1', '0.1.0')
    assert parse_version('malformed') == parse_version('nonsense'), \
        "a letter that is not a stage tag is not one ('a' in 'malformed')"
