"""The launcher's OpenGL check: a machine that cannot draw the 3-D view
is told so before the window is built, instead of dying in VTK.

The probe itself is Windows calls and runs only there (the release smoke
test runs it with and without Mesa beside the exe). What can be pinned
everywhere is the judgment of what the probe reads, and that the
launcher stops on it before building a window.
"""

from __future__ import annotations

import os

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


def _no_qt(monkeypatch):
    """Fail the test the moment the launcher touches Qt: the binding
    check, a QApplication, or the window. A machine with no OpenGL must
    not build a QApplication at all, since that is what crashed on the
    way out behind Windows' crash dialog (the 0.1.0a56 release run)."""
    import PySide6.QtWidgets as widgets

    from visualdynamics import gui
    from visualdynamics.gui import main_window

    def refuse(*_args, **_kwargs):
        raise AssertionError('Qt was touched before the OpenGL check')

    monkeypatch.setattr(gui, 'check_qt_binding', refuse)
    monkeypatch.setattr(widgets, 'QApplication', refuse)
    monkeypatch.setattr(main_window, 'MainWindow', refuse)


def test_the_launcher_says_what_is_missing_before_touching_qt(monkeypatch):
    """The reason is said through `opengl.say` and the launcher returns 2
    with no QApplication, no binding check and no window."""
    _no_qt(monkeypatch)
    from visualdynamics import gui

    said = []
    monkeypatch.setattr(opengl, 'missing', lambda: 'no OpenGL here')
    monkeypatch.setattr(opengl, 'say', said.append)
    assert gui.main(['--no-disclaimer']) == 2
    assert said == ['no OpenGL here']


def test_a_probe_that_cannot_run_does_not_refuse_the_launch(monkeypatch,
                                                            capsys):
    """A missing Win32 name raised inside the probe on the release
    runner. The check reports it; the launch goes on, since a check
    that could not run is no finding about the machine."""
    def broken():
        raise AttributeError("module 'ctypes.wintypes' has no attribute 'X'")

    monkeypatch.setattr(opengl, 'windows_opengl', broken)
    monkeypatch.setattr(opengl.sys, 'platform', 'win32')
    assert opengl.missing() is None
    assert 'could not run' in capsys.readouterr().err
    code, message = opengl.check()
    assert code == 1 and 'could not run' in message and "'X'" in message


def test_the_reason_is_said_through_win32_on_windows(monkeypatch):
    """No Qt on the refusal path on Windows: the box is user32's."""
    import types

    boxes = []
    user32 = types.SimpleNamespace(
        MessageBoxW=lambda owner, text, title, flags: boxes.append(
            (owner, text, title, flags)))
    monkeypatch.setattr(opengl.ctypes, 'windll',
                        types.SimpleNamespace(user32=user32), raising=False)
    monkeypatch.setattr(opengl.sys, 'platform', 'win32')
    opengl.say('no OpenGL here')
    assert boxes == [(None, 'no OpenGL here', 'Visual Dynamics', 0x10)]


def test_the_win32_probe_types_build_everywhere():
    """The pixel format and the handle types are defined at import, so a
    name that ctypes.wintypes does not have (HGLRC was not) fails the
    suite on every platform, not the first Windows machine."""
    import ctypes

    # PIXELFORMATDESCRIPTOR is 40 bytes where it is used; ctypes'
    # DWORD is c_ulong, eight bytes on LP64, which pads it to 64 here
    expected = 40 if ctypes.sizeof(ctypes.c_ulong) == 4 else 64
    assert ctypes.sizeof(opengl._PixelFormat) == expected
    assert [name for name, _kind in opengl._PixelFormat._fields_][:3] \
        == ['nSize', 'nVersion', 'dwFlags']
    assert len(opengl._PixelFormat._fields_) == 26
    assert opengl._HGLRC is opengl.wintypes.HANDLE


def test_check_gl_prints_and_returns_the_checks_answer_before_any_qt(
        monkeypatch, capsys):
    _no_qt(monkeypatch)
    from visualdynamics import gui

    monkeypatch.setattr(opengl, 'check', lambda: (2, 'missing: OpenGL 1.1'))
    assert gui.main(['--check-gl']) == 2
    assert capsys.readouterr().out.strip() == 'missing: OpenGL 1.1'


def test_qt_quick_draws_with_opengl_as_the_3d_views_do(monkeypatch, qt_app):
    """Qt composes a window with one graphics system. The 3-D views are
    OpenGL widgets and the report's web view is Qt Quick, Metal by
    default on a Mac; together, every 3-D view composed blank (Brandon,
    2026-10-21). The launcher and the window itself both ask Qt Quick for
    OpenGL before any web view exists, and leave a choice already made
    in the environment alone. The blank itself shows only on a real
    screen, so what is pinned here is the asking."""
    from visualdynamics.gui import before_application
    from visualdynamics.gui.main_window import MainWindow

    monkeypatch.delenv(opengl.QUICK_BACKEND, raising=False)
    before_application()
    assert os.environ[opengl.QUICK_BACKEND] == 'opengl', 'the launcher'
    monkeypatch.delenv(opengl.QUICK_BACKEND, raising=False)
    window = MainWindow(offscreen_3d=True)
    try:
        assert os.environ[opengl.QUICK_BACKEND] == 'opengl', (
            'a window a script builds without the launcher')
    finally:
        window.close()
        window.deleteLater()
    monkeypatch.setenv(opengl.QUICK_BACKEND, 'metal')
    opengl.one_graphics_system()
    assert os.environ[opengl.QUICK_BACKEND] == 'metal', 'a choice made stands'
