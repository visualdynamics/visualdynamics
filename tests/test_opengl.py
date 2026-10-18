"""The launcher's OpenGL check: a machine that cannot draw the 3-D view
is told so before the window is built, instead of dying in VTK.

The probe itself is Windows calls and runs only there (the release smoke
test runs it with and without Mesa beside the exe). What can be pinned
everywhere is the judgment of what the probe reads, and that the
launcher stops on it before building a window.
"""

from __future__ import annotations

import pytest

from visualdynamics.gui import opengl


@pytest.mark.parametrize('version,renderer', [
    ('4.6.0 NVIDIA 551.23', 'NVIDIA GeForce RTX 3060/PCIe/SSE2'),
    ('4.5 (Compatibility Profile) Mesa 26.2.0', 'llvmpipe (LLVM 19.1.7)'),
    ('3.2.0 Build 9.17.10.4459', 'Intel(R) HD Graphics'),
])
def test_opengl_from_3_2_up_draws_the_view(version, renderer):
    assert opengl.judge(version, renderer) is None


def test_windows_own_opengl_1_1_is_named_with_what_to_do():
    reason = opengl.judge('1.1.0', 'GDI Generic')
    assert 'OpenGL 3.2 or later' in reason
    assert 'this machine offers OpenGL 1.1.0 (GDI Generic)' in reason
    assert 'Install or update the graphics driver' in reason
    assert "Mesa's opengl32.dll" in reason


def test_just_under_the_need_is_refused():
    assert opengl.judge('3.1.0', 'Mesa') is not None


def test_no_pixel_format_at_all_is_said():
    assert 'Windows offered no OpenGL here' in opengl.judge(None, None)


def test_only_windows_is_asked(monkeypatch):
    """macOS always has OpenGL and the Linux AppImage carries Mesa; the
    Windows calls are made on Windows alone."""
    def refuse():
        raise AssertionError('the Windows probe ran off Windows')

    monkeypatch.setattr(opengl, 'windows_opengl', refuse)
    monkeypatch.setattr(opengl.sys, 'platform', 'darwin')
    assert opengl.missing() is None
    monkeypatch.setattr(opengl, 'windows_opengl',
                        lambda: ('1.1.0', 'GDI Generic'))
    monkeypatch.setattr(opengl.sys, 'platform', 'win32')
    assert 'OpenGL 1.1.0 (GDI Generic)' in opengl.missing()


def test_the_launcher_says_what_is_missing_before_building_a_window(
        qt_app, monkeypatch):
    """The window's 3-D view is where a driverless machine dies, so the
    reason is shown and the launcher returns 2 without building one."""
    import signal

    from PySide6.QtWidgets import QMessageBox

    from visualdynamics import gui
    from visualdynamics.gui import main_window

    shown = []
    monkeypatch.setattr(opengl, 'missing', lambda: 'no OpenGL here')
    monkeypatch.setattr(QMessageBox, 'critical',
                        lambda _parent, title, text: shown.append((title,
                                                                   text)))

    def built(*_args, **_kwargs):
        raise AssertionError('a window was built on a machine that '
                             'cannot draw it')

    monkeypatch.setattr(main_window, 'MainWindow', built)
    # what main() sets for the process, kept from leaking into the suite
    monkeypatch.setattr(signal, 'signal', lambda *_args: None)
    monkeypatch.setenv('QT_LOGGING_RULES', 'qt.core.qobject.connect=false')
    assert gui.main(['--no-disclaimer']) == 2
    assert shown == [('Visual Dynamics', 'no OpenGL here')]


def test_check_gl_prints_and_returns_the_checks_answer(qt_app, monkeypatch,
                                                       capsys):
    import signal

    from visualdynamics import gui
    from visualdynamics.gui import main_window

    def built(*_args, **_kwargs):
        raise AssertionError('--check-gl went on to build a window')

    # a self-check that fell through would open the window and wait on
    # the alpha notice; refusing the window fails it instead
    monkeypatch.setattr(main_window, 'MainWindow', built)
    monkeypatch.setattr(opengl, 'check', lambda: (2, 'missing: OpenGL 1.1'))
    monkeypatch.setattr(signal, 'signal', lambda *_args: None)
    monkeypatch.setenv('QT_LOGGING_RULES', 'qt.core.qobject.connect=false')
    assert gui.main(['--check-gl']) == 2
    assert capsys.readouterr().out.strip() == 'missing: OpenGL 1.1'
