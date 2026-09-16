"""File → Appearance: light, dark, or the platform's, remembered.

A friend of Brandon's on a Linux desktop Qt could not read got a light
window and no way to change it (2026-09-14). The menu is the user's
toggle; the choice is stored and worn on the next launch; System means
follow the platform again, which is the default. This launch's own
statement (`--theme`, the environment) beats the remembered choice.
"""

from __future__ import annotations

import pytest

from visualdynamics.gui import preferences
from visualdynamics.theme import OVERRIDE


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    monkeypatch.delenv(OVERRIDE, raising=False)
    preferences.remember_appearance('system')
    yield
    preferences.remember_appearance('system')


def _appearance_actions(window):
    file_menu = next(m for m in window.menuBar().findChildren(type(window.menuBar().actions()[0].menu()))
                     if m.title() == '&File')
    submenu = next(a.menu() for a in file_menu.actions()
                   if a.menu() is not None and a.text() == '&Appearance')
    return {a.text(): a for a in submenu.actions()}


def test_the_menu_offers_three_exclusive_choices_with_system_checked(window):
    actions = _appearance_actions(window)
    assert list(actions) == ['&System', '&Light', '&Dark']
    assert all(a.isCheckable() for a in actions.values())
    assert actions['&System'].isChecked()
    assert actions['&Light'].actionGroup() is actions['&Dark'].actionGroup()
    assert actions['&Light'].actionGroup().isExclusive()


def test_a_choice_is_worn_now_and_remembered_for_the_next_window(
        window, pump, window_factory):
    actions = _appearance_actions(window)
    actions['&Dark'].trigger()
    pump()
    assert window.theme_name == 'dark'
    assert actions['&Dark'].isChecked() and not actions['&System'].isChecked()
    assert preferences.remembered_appearance() == 'dark'
    # the next launch: a fresh window reads the store
    later = window_factory()
    assert later.theme_name == 'dark'
    assert _appearance_actions(later)['&Dark'].isChecked()
    actions['&Light'].trigger()
    pump()
    assert window.theme_name == 'light'
    assert preferences.remembered_appearance() == 'light'


def test_system_follows_the_platform_again(window, pump, monkeypatch):
    actions = _appearance_actions(window)
    actions['&Dark'].trigger()
    pump()
    monkeypatch.setattr(preferences, 'system_scheme', lambda: 'light')
    actions['&System'].trigger()
    pump()
    assert window.theme_name == 'light'
    assert preferences.remembered_appearance() == 'system'
    # and a platform switch is followed only while System is chosen
    monkeypatch.setattr(preferences, 'system_scheme', lambda: 'dark')
    window._scheme_changed(None)
    assert window.theme_name == 'dark'
    actions['&Light'].trigger()
    pump()
    window._scheme_changed(None)
    assert window.theme_name == 'light', 'a chosen appearance ignores the platform'


def test_this_launchs_statement_beats_the_remembered_choice(monkeypatch):
    preferences.remember_appearance('dark')
    assert preferences.chosen_scheme() == 'dark'
    monkeypatch.setenv(OVERRIDE, 'light')
    assert preferences.chosen_scheme() == 'light'


def test_the_store_refuses_a_word_that_is_not_an_appearance():
    with pytest.raises(ValueError):
        preferences.remember_appearance('sepia')


def test_the_tests_never_touch_the_real_store():
    """The store the tests write is the session's own folder, set at
    conftest import — a fixture-scoped redirect left one test writing
    into the real preferences file (2026-09-14)."""
    from conftest import SETTINGS_STORE

    assert preferences.settings().fileName().startswith(SETTINGS_STORE), \
        preferences.settings().fileName()


def test_the_whole_application_wears_the_choice(window, pump, monkeypatch):
    """The first cut changed the plots and the scene and left the
    window chrome as the OS had it: Brandon chose Light and saw nothing
    change (2026-09-14). The choice goes to the application's style
    hints (`QStyleHints.setColorScheme`, Qt 6.8+), which every native
    widget follows; System hands the decision back to the platform.
    The offscreen platform the tests run on ignores the request, so
    the call is what is pinned here — the effect was watched on macOS:
    the widget palette's window colour went from lightness 50 to 236
    and back."""
    from visualdynamics.gui import main_window

    worn = []
    monkeypatch.setattr(main_window, 'wear_appearance',
                        lambda app=None, choice=None: worn.append(choice))
    actions = _appearance_actions(window)
    actions['&Light'].trigger()
    pump()
    actions['&Dark'].trigger()
    pump()
    actions['&System'].trigger()
    pump()
    assert worn == ['light', 'dark', 'system']
    assert window.theme_name in ('light', 'dark')


def test_a_fresh_window_wears_the_remembered_choice_first(window_factory,
                                                         monkeypatch):
    """Worn before the panes are built, so chrome and drawn parts start
    on the same scheme."""
    from visualdynamics.gui import main_window

    worn = []
    monkeypatch.setattr(main_window, 'wear_appearance',
                        lambda app=None, choice=None: worn.append('called'))
    preferences.remember_appearance('light')
    later = window_factory()
    assert worn == ['called']
    assert later.theme_name == 'light'


def test_wear_appearance_maps_the_choice_onto_qt(qt_app):
    """The mapping itself: a recorder in place of the hints."""
    from PySide6.QtCore import Qt

    class Hints:
        def __init__(self):
            self.set = []

        def setColorScheme(self, scheme):
            self.set.append(scheme)

    class App:
        def __init__(self):
            self.hints = Hints()

        def styleHints(self):
            return self.hints

    app = App()
    preferences.wear_appearance(app, 'light')
    preferences.wear_appearance(app, 'dark')
    preferences.wear_appearance(app, 'system')
    assert app.hints.set == [Qt.ColorScheme.Light, Qt.ColorScheme.Dark,
                             Qt.ColorScheme.Unknown]
    preferences.remember_appearance('dark')
    preferences.wear_appearance(app)          # the default: what is chosen
    assert app.hints.set[-1] == Qt.ColorScheme.Dark

    # and every widget is made to re-read the palette the scheme set,
    # which they do not do on their own (macOS, 2026-09-14)
    refreshed = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(preferences, 'refresh_palettes',
                      lambda app=None: refreshed.append(app))
        preferences.wear_appearance(app, 'light')
    assert refreshed == [app]


def test_a_platform_switch_refreshes_the_palettes_too(window, monkeypatch):
    """The OS-driven switch, not only the menu: without the refresh the
    status bar and the panes stayed light after the Mac went dark
    (Brandon's screenshot, 2026-09-14)."""
    from visualdynamics.gui import main_window

    calls = []
    monkeypatch.setattr(main_window, 'refresh_palettes',
                        lambda app=None: calls.append(app))
    window._scheme_changed(None)
    assert len(calls) == 1


def test_the_scene_is_told_the_theme_not_just_to_repaint(window, pump):
    """Told only to repaint, the 3-D view repainted in the theme it was
    built with: a window opened light stayed white after Dark
    (Brandon's screenshot, 2026-09-14)."""
    actions = _appearance_actions(window)
    actions['&Light'].trigger()
    pump()
    assert window.scene.theme_name == 'light'
    actions['&Dark'].trigger()
    pump()
    assert window.scene.theme_name == 'dark'
    assert window.data_pane.theme_name == 'dark'


def test_a_refresh_leaves_the_themed_widgets_their_own_palette(window, pump):
    """`refresh_palettes` makes widgets re-read the application palette
    (macOS widgets keep a stale one after a scheme switch) and must not
    take the tree's and the tables' own theme colours away with it."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette

    from visualdynamics.gui.main_window import resolve_theme

    base_before = window.tree.palette().color(QPalette.ColorRole.Base).name()
    assert window.tree.testAttribute(Qt.WidgetAttribute.WA_SetPalette)
    preferences.refresh_palettes()
    pump()
    assert window.tree.testAttribute(Qt.WidgetAttribute.WA_SetPalette)
    assert window.tree.palette().color(QPalette.ColorRole.Base).name() == base_before
    assert base_before == resolve_theme(window.theme_name)['scene_background'].lower()
    assert not window.statusBar().testAttribute(Qt.WidgetAttribute.WA_SetPalette), \
        'a refreshed widget does not come to own a palette'
