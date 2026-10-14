"""What the application remembers between launches: the appearance,
whether the plots open in 3-D, and the creator name written into ESCDF
files.

The preferences are kept in one `QSettings` store under one name, so a
preference set on one launch is read on the next. When
`VISUALDYNAMICS_SETTINGS` names a file, the store is that INI file
instead. The tests set it, so nothing a test chooses reaches the user's
own preferences.

The appearance a window uses is decided in this order: an explicit
statement for this launch (`VISUALDYNAMICS_THEME`, which `--theme`
sets), then the remembered choice, then what the platform reports.
`'system'` means follow the platform. It is the default, and
`theme.system_scheme` reads the platform's answer.
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
    """The application's preference store: the platform's (a plist on
    macOS, the registry on Windows, a file under ~/.config on Linux), or
    the INI file `VISUALDYNAMICS_SETTINGS` names."""
    path = os.environ.get(STORE)
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings('visualdynamics', 'Visual Dynamics')


def remembered_appearance() -> str:
    """The remembered appearance, 'system', 'light' or 'dark'. It is
    'system' when nothing was chosen or the stored word is not one of
    these."""
    said = str(settings().value(KEY, 'system')).strip().lower()
    return said if said in APPEARANCES else 'system'


def remember_appearance(choice: str) -> None:
    """Store the appearance choice, 'system', 'light' or 'dark'.

    'system' removes the stored choice instead of storing it. Anything
    else raises ValueError.
    """
    if choice not in APPEARANCES:
        raise ValueError(f'appearance is one of {APPEARANCES}, not {choice!r}')
    store = settings()
    if choice == 'system':
        store.remove(KEY)
    else:
        store.setValue(KEY, choice)
    store.sync()


#: the 2D/3D reading, one choice for the data plots and the MAC alike
THREE_D_KEY = 'reading_3d'


def remembered_3d() -> bool:
    """Whether the plots open in 3-D: True by default, and False once
    the 2-D/3-D toggle was last left on 2-D. The one choice covers the
    data plots and the MAC."""
    # 3-D by default is Brandon's call. The choice persists across the
    # MAC and across launches (Brandon, 2026-09-28).
    said = settings().value(THREE_D_KEY, True)
    return str(said).strip().lower() not in ('false', '0', 'no')


def remember_3d(on: bool) -> None:
    """Store the 2D/3D choice."""
    store = settings()
    store.setValue(THREE_D_KEY, bool(on))
    store.sync()


CREATOR_KEY = 'creator'


def remembered_creator() -> str:
    """The name an exported ESCDF file records as its creator
    (File → Creator Name…), or '' when none was set. The writer then
    records the login name, because a file format library must never
    prompt for one."""
    # The login-name fallback, and never a prompt: 2026-09-30.
    return str(settings().value(CREATOR_KEY, '') or '').strip()


def remember_creator(name: str) -> None:
    """Store the creator name."""
    store = settings()
    store.setValue(CREATOR_KEY, str(name).strip())
    store.sync()


def chosen_scheme() -> str:
    """The theme to use now: this launch's statement, else the
    remembered choice, else the platform's."""
    said = os.environ.get(OVERRIDE, '').strip().lower()
    if said in THEMES:
        return said
    remembered = remembered_appearance()
    if remembered in THEMES:
        return remembered
    return system_scheme()


def wear_appearance(app: Any = None, choice: str | None = None,
                    refresh: bool = True) -> None:
    """Make the whole application use the chosen appearance: the window
    frame, the menus and every native widget, as well as the parts this
    program draws itself.

    This is `QStyleHints.setColorScheme`, from Qt 6.8: Light or Dark
    forces the application's appearance (on macOS, the NSApp
    appearance), and Unknown follows the platform again. On an older Qt
    nothing is done here, and only the parts this program draws change.
    The platform then reports the forced scheme, which is what
    `theme.system_scheme` and the `colorSchemeChanged` slot read, so
    every later repaint agrees.

    `app` defaults to the running `QApplication`, and with none nothing
    happens. `choice` is 'system', 'light' or 'dark', and defaults to
    this launch's statement, else the remembered choice ('system'
    follows the platform). `refresh` also makes every widget already
    built re-read the palette (`refresh_palettes`), which a change of
    appearance needs. A window that is being built passes False, because
    its widgets read the palette as they are made.
    """
    # Without setColorScheme the first cut of the menu changed the plots
    # and the scene and left the window frame as the OS had it: Brandon
    # chose Light and saw nothing change (2026-09-14).
    # refresh=False for a window being built: walking every other
    # top-level widget from its constructor reached windows and web views
    # that were closed and waiting to be deleted, a segmentation fault in
    # refresh_palettes on a CI runner (2026-09-27, the Python 3.12 job
    # after public #62).
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
    if refresh:
        refresh_palettes(app)


def refresh_palettes(app: Any = None) -> None:
    """Make every widget re-read the application palette.

    On macOS with Qt 6.11 the application palette follows a change of
    color scheme at once, but widgets already on screen keep the palette
    they resolved before it, while a new window is drawn correctly.
    Setting an empty palette on a widget makes it resolve from its
    parent and the application again, and leaves its `WA_SetPalette`
    attribute clear. Widgets with that attribute set own a palette on
    purpose (the tree and the tables, colored by the theme), so they are
    skipped and keep theirs.
    """
    # Seen in Brandon's screenshot (2026-09-14): the status bar and the
    # panes stayed light after he switched back to Dark.
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
