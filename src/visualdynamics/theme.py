"""Light/dark theme colors shared by the 3D view, 2D plots, and the GUI —
and the one answer to "which is it right now".

The colours are for the things visualdynamics draws itself (the VTK
scene and the pyqtgraph plots); Qt's own widgets take theirs from the
application palette. Which scheme applies is decided here too, in
`system_scheme`: a statement for this launch (`VISUALDYNAMICS_THEME`,
which the `--theme` flag sets), else what Qt reports, else — on Linux,
where a packaged build without a platform-theme plugin made Qt answer
Unknown on a dark desktop (2026-09-14) — the desktop asked directly
through `desktop_scheme` (the settings portal, then GNOME's key, then
KDE's file), else the palette's own lightness. The user's remembered
choice, and telling Qt to *wear* a scheme rather than follow the OS,
live in `gui/preferences.py`, which is Qt's side of this.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LIGHT = {
    'name': 'light',
    'scene_background': '#ffffff',
    # flat, equal to the base: the gradient read as depth for a while
    # and then read as clutter (Brandon, 2026-08-23) — pure white and
    # pure black let the data carry the scene
    'scene_background_top': '#ffffff',
    'scene_text': '#1c1c1e',
    'plot_background': '#ffffff',
    'plot_background_top': '#ffffff',
    'plot_foreground': '#1c1c1e',
    'scene_muted': '#b8b8bd',       # context geometry behind a selection
    'scene_highlight': '#ff8c2b',   # the selected entity, or the one hovered
    'scene_picked': '#1f9d55',      # nodes gathered so far for a new element
    'row_error': '#c02a1f',         # an object that does not fit the geometry
    # the zones a random-vibration response must not be in: between the
    # warning and abort limits, and beyond abort. Filled translucent, so
    # these are the colours before alpha.
    'limit_warning': '#e6b800',
    'limit_abort': '#d13b2e',
    # the frames a PSD will be averaged over, shaded on the time
    # history they are cut from. The band is translucent, so that is the
    # colour before alpha; the window drawn over each frame is warm on
    # purpose — a blue mark over blue traces reads as another channel
    # a measured line that went outside an abort limit, shaded over its
    # own bin: red where it went over, blue where it fell under. Two
    # colours because which way it went is the first thing to know,
    # and stronger than the zone shading they sit inside
    # a specification and the response it bounds, drawn as a pair: the
    # response in the foreground colour because it is what is being
    # looked at, the specification in grey behind it because it is the
    # reference. Neither takes a palette colour — there is only ever
    # one pair on the plot, so there is nothing to tell apart.
    'response_curve': '#1c1c1e',
    'specification_curve': '#8a8a8f',
    'exceed_over': '#d13b2e',
    'exceed_under': '#2f6fd0',
    'averaging_band': '#3b7dd8',
    'averaging_window': '#c2410c',
    # the filtered trace previewed over the raw one, flat and on the
    # stage alike. **Magenta, measured** (Brandon, 2026-08-25): green
    # was the first choice against the 2-D palette, and it disappeared
    # on the stage — viridis runs purple, teal, *green*, yellow, so a
    # green twin sat inside the very colormap it had to stand out
    # from (49 units from the nearest stop, against magenta's 203).
    # Magenta is in neither the colormap nor the mark colours.
    'filter_preview': '#b5179e',
}

DARK = {
    'name': 'dark',
    'scene_background': '#000000',
    # flat, equal to the base — see the light theme's note
    'scene_background_top': '#000000',
    'scene_text': '#e8e8ea',
    'plot_background': '#000000',
    'plot_background_top': '#000000',
    'plot_foreground': '#e8e8ea',
    'scene_muted': '#4a4a55',       # context geometry behind a selection
    'scene_highlight': '#ffb020',   # the selected entity, or the one hovered
    'scene_picked': '#3ddc84',      # nodes gathered so far for a new element
    'row_error': '#ff6b61',         # an object that does not fit the geometry
    # the zones a random-vibration response must not be in: between the
    # warning and abort limits, and beyond abort. Filled translucent, so
    # these are the colours before alpha.
    'limit_warning': '#ffcc33',
    'limit_abort': '#ff6b61',
    # the frames a PSD will be averaged over, shaded on the time
    # history they are cut from. The band is translucent, so that is the
    # colour before alpha; the window drawn over each frame is warm on
    # purpose — a blue mark over blue traces reads as another channel
    # a measured line that went outside an abort limit, shaded over its
    # own bin: red where it went over, blue where it fell under. Two
    # colours because which way it went is the first thing to know,
    # and stronger than the zone shading they sit inside
    # a specification and the response it bounds, drawn as a pair: the
    # response in the foreground colour because it is what is being
    # looked at, the specification in grey behind it because it is the
    # reference. Neither takes a palette colour — there is only ever
    # one pair on the plot, so there is nothing to tell apart.
    'response_curve': '#ffffff',
    'specification_curve': '#9a9aa2',
    'exceed_over': '#ff6b61',
    'exceed_under': '#5fa8ff',
    'averaging_band': '#5fa8ff',
    'averaging_window': '#fb923c',
    'filter_preview': '#ff5ae0',
}

#: How solid the *other* set is when two mode shapes are overlaid. The
#: basis stays opaque and the comparison is drawn through it: a finite
#: element model has a skin where a test set has a wireframe, and an
#: opaque FEM simply hides the thing it is being compared against.
#:
#: Here rather than in the window because the report draws the same
#: overlay in a canvas and must draw it the same way. It did not, and
#: the difference was invisible until the two were put side by side.
OVERLAY_ALPHA = 0.25

THEMES = {'light': LIGHT, 'dark': DARK}

DEFAULT = 'light'


def theme(name: str | Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve a theme name (or a colors dict) to a colors dict."""
    if isinstance(name, dict):
        return name
    if name is None:
        return THEMES[DEFAULT]
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(f"Unknown theme {name!r}; choose from {sorted(THEMES)}")


#: viridis, as the nine stops everything interpolates between. One
#: colour scale, one set of numbers: the app paints a moving model by
#: displacement on it, pyqtgraph draws its MAC grids and coherence
#: maps on it, the app icon is drawn with it, and the report is handed
#: these very stops rather than carrying a second copy in JavaScript —
#: so a colour means one thing wherever it appears.
VIRIDIS = ((68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142),
           (33, 144, 141), (39, 173, 129), (92, 200, 99),
           (170, 220, 50), (253, 231, 37))


def colormap(values: Any) -> Any:
    """`values` in 0..1 as an (N, 3) array of RGB, on `VIRIDIS`."""
    import numpy as np

    stops = np.asarray(VIRIDIS, dtype=float)
    t = np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * (len(stops) - 1)
    low = np.clip(t.astype(int), 0, len(stops) - 2)
    f = (t - low)[..., None]
    return stops[low] * (1.0 - f) + stops[low + 1] * f


#: an explicit answer beats every guess: `VISUALDYNAMICS_THEME=dark`
#: (or light) in the environment, which the launcher's `--theme` sets
OVERRIDE = 'VISUALDYNAMICS_THEME'


def system_scheme(app: Any = None) -> str:
    """'dark' or 'light': the environment's say, else the OS appearance
    via Qt, else the desktop asked directly, else the palette.

    Falls back to the default theme when Qt or an application instance is
    unavailable (e.g. plain scripting use).
    """
    import os

    said = os.environ.get(OVERRIDE, '').strip().lower()
    if said in THEMES:
        return said
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return DEFAULT
    app = app or QApplication.instance()
    if app is None:
        return DEFAULT
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return 'dark'
    if scheme == Qt.ColorScheme.Light:
        return 'light'
    # Qt could not tell. On Linux that is the usual case: Qt learns the
    # scheme through a platform-theme plugin, and a packaged build
    # shipped none until 2026-09-14 — a friend of Brandon's on a dark
    # desktop got a light window. The desktop is asked directly before
    # the palette is judged, since the palette is Qt's own light one
    # whenever the plugin is missing.
    asked = desktop_scheme()
    if asked is not None:
        return asked
    palette = app.palette()  # judge by palette lightness
    return ('dark' if palette.color(palette.ColorRole.Window).lightness() < 128
            else 'light')


def desktop_scheme(platform: str | None = None,
                   run: Any = None, home: str | None = None) -> str | None:
    """The Linux desktop's own answer, or None where there is none.

    Three doors, in the order a modern desktop answers them: the XDG
    settings portal's `color-scheme` (GNOME, KDE and the rest, over
    D-Bus through `gdbus`, which ships with GLib), GNOME's
    `gsettings` key of the same name, and KDE's `kdeglobals`. Each is
    a subprocess or a file read with a short timeout, and any failure
    is "no answer" — never an exception in the way of a window.
    `platform`, `run` and `home` are for the tests.
    """
    import os
    import subprocess
    import sys

    platform = platform or sys.platform
    if not platform.startswith('linux'):
        return None
    run = run or (lambda cmd: subprocess.run(
        cmd, capture_output=True, text=True, timeout=2, check=False).stdout)
    for command, dark, light in (
            (['gdbus', 'call', '--session',
              '--dest', 'org.freedesktop.portal.Desktop',
              '--object-path', '/org/freedesktop/portal/desktop',
              '--method', 'org.freedesktop.portal.Settings.ReadOne',
              'org.freedesktop.appearance', 'color-scheme'],
             'uint32 1', 'uint32 2'),
            (['gsettings', 'get', 'org.gnome.desktop.interface',
              'color-scheme'],
             'prefer-dark', 'prefer-light')):
        try:
            out = run(command) or ''
        except Exception:  # noqa: BLE001, S112 — absent tool, timeout: no answer from this door
            continue
        if dark in out:
            return 'dark'
        if light in out:
            return 'light'
    try:
        kdeglobals = os.path.join(home or os.path.expanduser('~'),
                                  '.config', 'kdeglobals')
        with open(kdeglobals, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip().lower().startswith('colorscheme='):
                    return ('dark' if 'dark' in line.lower() else 'light')
    except OSError:
        pass
    return None
