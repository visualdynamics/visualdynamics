"""What the application remembers between launches: today, the
appearance.

One `QSettings` under one name, so a preference set on one launch is
read on the next; the tests point the store at a temporary folder
through `QSettings.setPath`, so nothing a test chooses reaches the
user's own file.

The appearance a window wears is decided in this order, and the order
is the point: an explicit statement for this launch
(`VISUALDYNAMICS_THEME`, which `--theme` sets) beats the remembered
choice, and the remembered choice beats what the platform reports —
`'system'` means follow the platform, which is the default and what
`theme.system_scheme` answers.
"""

from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import QSettings

from ..theme import OVERRIDE, THEMES, system_scheme

#: the remembered choices, in menu order
APPEARANCES = ('system', 'light', 'dark')
KEY = 'appearance'


#: a file to keep the preferences in instead of the platform's store —
#: what the tests set, so nothing a test chooses reaches the user's own
STORE = 'VISUALDYNAMICS_SETTINGS'


def settings() -> QSettings:
    """The application's own store: the platform's (a plist on macOS,
    the registry on Windows, a file under ~/.config on Linux), or the
    file `VISUALDYNAMICS_SETTINGS` names."""
    path = os.environ.get(STORE)
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings('visualdynamics', 'Visual Dynamics')


def remembered_appearance() -> str:
    """'system', 'light' or 'dark' — 'system' when nothing was chosen
    or the stored word is not one of ours."""
    said = str(settings().value(KEY, 'system')).strip().lower()
    return said if said in APPEARANCES else 'system'


def remember_appearance(choice: str) -> None:
    """Store the choice; 'system' clears it rather than storing a word
    that means "nothing chosen"."""
    if choice not in APPEARANCES:
        raise ValueError(f'appearance is one of {APPEARANCES}, not {choice!r}')
    store = settings()
    if choice == 'system':
        store.remove(KEY)
    else:
        store.setValue(KEY, choice)
    store.sync()


def chosen_scheme() -> str:
    """The theme to wear now: this launch's statement, else the
    remembered choice, else the platform's."""
    said = os.environ.get(OVERRIDE, '').strip().lower()
    if said in THEMES:
        return said
    remembered = remembered_appearance()
    if remembered in THEMES:
        return remembered
    return system_scheme()


def wear_appearance(app: Any = None, choice: str | None = None) -> None:
    """Make the whole application wear the chosen appearance — the
    window chrome, the menus, every native widget — not only the parts
    this program draws itself.

    Qt 6.8 gave `QStyleHints.setColorScheme`: Light or Dark forces the
    application's appearance (on macOS, the NSApp appearance), Unknown
    follows the platform again. Without it the first cut of the menu
    changed the plots and the scene and left the chrome as the OS had
    it, and Brandon chose Light and saw nothing change (2026-09-14).
    The platform then reports the forced scheme, which is what
    `theme.system_scheme` and the `colorSchemeChanged` slot read, so
    every later repaint agrees. `choice` defaults to what
    `chosen_scheme` would answer: this launch's statement, else the
    remembered choice, else the platform's.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = app or QApplication.instance()
    if app is None:
        return
    hints = app.styleHints()
    if not hasattr(hints, 'setColorScheme'):
        return                      # an older Qt: the drawn parts only
    if choice is None:
        said = os.environ.get(OVERRIDE, '').strip().lower()
        choice = said if said in THEMES else remembered_appearance()
    hints.setColorScheme({'light': Qt.ColorScheme.Light,
                          'dark': Qt.ColorScheme.Dark}.get(
                              choice, Qt.ColorScheme.Unknown))
    refresh_palettes(app)


def refresh_palettes(app: Any = None) -> None:
    """Make every widget re-read the application palette.

    On macOS with Qt 6.11 the application palette follows a color
    scheme change at once, but widgets already on screen keep the
    palette they resolved before it — the status bar and the panes
    stayed light after Brandon switched back to Dark (his screenshot,
    2026-09-14), while a fresh window came up right. Setting an
    *empty* palette on a widget makes it resolve from its parent and
    the application again, and leaves `WA_SetPalette` clear, so the
    widgets that own a palette on purpose (the tree and the tables,
    colored by the theme) are skipped and keep theirs.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication, QWidget

    app = app or QApplication.instance()
    if app is None or not hasattr(app, 'topLevelWidgets'):
        return
    for top in app.topLevelWidgets():
        for widget in (top, *top.findChildren(QWidget)):
            if not widget.testAttribute(Qt.WidgetAttribute.WA_SetPalette):
                widget.setPalette(QPalette())
