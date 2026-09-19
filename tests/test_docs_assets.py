"""The site wears the application's own mark and colors.

Documentation that looks like a different product than the one it
documents is a small thing that reads as carelessness, and the way it
happens is that the app's palette moves and the site's does not. So the
site's stylesheet is *generated* from `visualdynamics.theme` — the same
dictionaries the 3D scene and the plots are drawn with — and its logo
from `draw_app_icon`, the function that draws the icon in the Dock.

This holds the committed assets to what those produce now. They are
committed rather than built with the site because rendering the icon
wants Qt, and a documentation build should not.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))


def test_the_assets_match_the_app(qt_app):
    import make_docs_assets

    assert make_docs_assets.main(check=True) == 0, (
        "the site's logo or palette no longer matches the app — run "
        'python tools/make_docs_assets.py')


def test_the_stylesheet_carries_the_apps_own_colors():
    """Not a copy of them: the values in the file are the values in
    `theme.py`, so there is no second palette to keep in step."""
    import make_docs_assets

    from visualdynamics.theme import DARK, LIGHT

    css = (ROOT / 'docs' / 'assets' / 'theme.css').read_text(encoding='utf-8')
    assert LIGHT['plot_background'] in css
    assert LIGHT['scene_highlight'] in css, 'the highlight the app uses'
    assert DARK['scene_background'] in css
    assert DARK['plot_foreground'] in css
    assert css == make_docs_assets.stylesheet()


def test_the_logo_is_the_application_icon(qt_app):
    """Rendered by the same function, so the tab, the Dock and the page
    header cannot end up wearing three different marks."""
    import make_docs_assets
    from PySide6.QtGui import QImage

    from visualdynamics.gui.icons import draw_app_icon

    logo = QImage()
    assert logo.loadFromData(
        (ROOT / 'docs' / 'assets' / 'logo.png').read_bytes())
    drawn = draw_app_icon(make_docs_assets.LOGO)
    assert (logo.width(), logo.height()) == (drawn.width(), drawn.height())
    # the same mark, allowing the rasterizer its rounding: the file was
    # drawn on a Mac and Windows antialiases a little differently (a
    # bug report, 2026-09-19); a different mark is off by far more
    center = (logo.width() // 2, logo.height() // 2)
    a, b = logo.pixelColor(*center), drawn.pixelColor(*center)
    assert max(abs(a.red() - b.red()), abs(a.green() - b.green()),
               abs(a.blue() - b.blue()), abs(a.alpha() - b.alpha())) <= 8
