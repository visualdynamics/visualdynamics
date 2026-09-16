
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics import theme as theme_module

PLATE = fixture_path('plate', 'geometry.exo')


def test_theme_names():
    assert set(theme_module.THEMES) == {'light', 'dark'}
    assert theme_module.theme('dark')['name'] == 'dark'
    assert theme_module.theme()['name'] == theme_module.DEFAULT


def test_theme_accepts_colors_dict():
    colors = theme_module.DARK
    assert theme_module.theme(colors) is colors


def test_unknown_theme_raises():
    with pytest.raises(ValueError):
        theme_module.theme('solarized')


def test_system_scheme_without_qt_app():
    """Scripting use with no QApplication falls back rather than failing."""
    assert theme_module.system_scheme() in ('light', 'dark')


def test_system_scheme_reads_the_apps_own_hint(qt_app):
    """With an application up — the startup path since apply_theme runs
    at launch — the answer comes from Qt's scheme, not the default."""
    from PySide6.QtCore import Qt

    got = theme_module.system_scheme(qt_app)
    scheme = qt_app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        assert got == 'dark'
    elif scheme == Qt.ColorScheme.Light:
        assert got == 'light'
    else:
        assert got in ('light', 'dark'), 'judged by palette lightness'


def test_system_scheme_judges_by_palette_when_qt_cannot_say():
    """Qt reports Unknown on platforms without a scheme signal; the
    window's background lightness is the honest fallback."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette

    class Hints:
        def colorScheme(self):
            return Qt.ColorScheme.Unknown

    class FakeApp:
        def __init__(self, window_color):
            self._palette = QPalette()
            self._palette.setColor(QPalette.ColorRole.Window,
                                   QColor(window_color))
        def styleHints(self):
            return Hints()
        def palette(self):
            return self._palette

    assert theme_module.system_scheme(FakeApp('#101010')) == 'dark'
    assert theme_module.system_scheme(FakeApp('#f0f0f0')) == 'light'


def test_scene_background_follows_theme():
    import pyvista as pv

    from visualdynamics.viz.geometry import geometry_scene

    geometry = visualdynamics.import_file(PLATE, length_unit='m')
    for name in ('light', 'dark'):
        plotter = pv.Plotter(off_screen=True)
        geometry_scene(geometry, plotter=plotter, theme=name)
        expected = pv.Color(theme_module.THEMES[name]['scene_background'])
        assert plotter.background_color == expected
        plotter.close()


def test_an_environment_override_beats_every_guess(monkeypatch, qt_app):
    """`VISUALDYNAMICS_THEME=dark` is the user's own statement — what
    the launcher's --theme sets — and it wins over Qt's answer."""
    monkeypatch.setenv(theme_module.OVERRIDE, 'dark')
    assert theme_module.system_scheme(qt_app) == 'dark'
    monkeypatch.setenv(theme_module.OVERRIDE, 'Light ')
    assert theme_module.system_scheme(qt_app) == 'light'
    monkeypatch.setenv(theme_module.OVERRIDE, 'sepia')
    assert theme_module.system_scheme(qt_app) in ('light', 'dark'), \
        'an unknown word is ignored, not obeyed'


def test_the_linux_desktop_is_asked_when_qt_cannot_say(monkeypatch, tmp_path):
    """A packaged Linux build carried no platform-theme plugin, so Qt
    answered Unknown and the light palette won on a dark desktop
    (Brandon's friend, 2026-09-14). The desktop is asked directly:
    the settings portal, then GNOME's key, then KDE's file; a door
    that is not there is silence, not an error."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette

    monkeypatch.delenv(theme_module.OVERRIDE, raising=False)

    class Hints:
        def colorScheme(self):
            return Qt.ColorScheme.Unknown

    class FakeApp:
        def styleHints(self):
            return Hints()

        def palette(self):
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.Window, QColor('#f0f0f0'))
            return palette

    answers = {}

    def run(cmd):
        if cmd[0] not in answers:
            raise FileNotFoundError(cmd[0])
        return answers[cmd[0]]

    # the portal says dark, in the shape gdbus prints
    answers['gdbus'] = '(<<uint32 1>>,)\n'
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) == 'dark'
    # ...or light, which is an answer and not a fall-through
    answers['gdbus'] = '(<<uint32 2>>,)\n'
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) == 'light'
    # "no preference" (0) is no answer: the next door is asked, and
    # what it says wins — a portal that answers 0 must not read as light
    answers['gdbus'] = '(<<uint32 0>>,)\n'
    answers['gsettings'] = "'prefer-dark'\n"
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) == 'dark'
    # no portal: GNOME's own key
    answers.clear()
    answers['gsettings'] = "'prefer-dark'\n"
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) == 'dark'
    answers['gsettings'] = "'prefer-light'\n"
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) == 'light'
    answers['gsettings'] = "'default'\n"
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) is None
    # neither tool: KDE's file
    (tmp_path / '.config').mkdir()
    (tmp_path / '.config' / 'kdeglobals').write_text(
        '[General]\nColorScheme=BreezeDark\n')
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) == 'dark'
    (tmp_path / '.config' / 'kdeglobals').write_text(
        '[General]\nColorScheme=BreezeLight\n')
    assert theme_module.desktop_scheme('linux', run, str(tmp_path)) == 'light'
    # not Linux: never asked
    assert theme_module.desktop_scheme('darwin', run, str(tmp_path)) is None

    # and system_scheme consults it before the palette
    monkeypatch.setattr(theme_module, 'desktop_scheme', lambda: 'dark')
    assert theme_module.system_scheme(FakeApp()) == 'dark'
    monkeypatch.setattr(theme_module, 'desktop_scheme', lambda: None)
    assert theme_module.system_scheme(FakeApp()) == 'light', 'palette last'


def test_the_launcher_takes_a_theme_flag():
    import pytest

    from visualdynamics.gui import theme_flag

    assert theme_flag(['a.vdyn', '--theme', 'dark']) == (['a.vdyn'], 'dark')
    assert theme_flag(['--theme=Light', 'b.nc4']) == (['b.nc4'], 'light')
    assert theme_flag(['b.nc4']) == (['b.nc4'], None)
    with pytest.raises(SystemExit):
        theme_flag(['--theme', 'sepia'])
