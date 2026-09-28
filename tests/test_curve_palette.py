"""The curve palette takes a shade for its background (2026-09-27).

On white the dark-background palette stood at 1.8:1 (yellow) to 3.7:1,
and a one-pixel line that faint was hard to see. Wider lines would have
helped and cost ten times the repaint (PLAN.md, "Curves against a
specification"); a darker shade of the same hue costs nothing.
"""

from __future__ import annotations

import colorsys


def _luminance(color):
    channels = [int(color.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b):
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _hue(color):
    rgb = [int(color.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return colorsys.rgb_to_hls(*rgb)[0] * 360


def test_the_light_shade_reads_on_white_and_keeps_its_hue():
    """4.5:1, the floor for readable text, on the light plot background;
    and each the same hue as its dark twin, so a record keeps its color
    from one theme to the other."""
    from visualdynamics.plot import CURVE_COLORS, CURVE_COLORS_LIGHT
    from visualdynamics.theme import theme

    white = theme('light')['plot_background']
    black = theme('dark')['plot_background']
    assert len(CURVE_COLORS_LIGHT) == len(CURVE_COLORS)
    for light, dark in zip(CURVE_COLORS_LIGHT, CURVE_COLORS):
        assert _contrast(light, white) >= 4.5, light
        assert _contrast(dark, black) >= 4.5, dark
        turn = abs(_hue(light) - _hue(dark)) % 360
        assert min(turn, 360 - turn) < 6, (light, dark)


def test_the_palette_follows_the_plot_background():
    from visualdynamics.plot import (
        CURVE_COLORS,
        CURVE_COLORS_LIGHT,
        curve_color,
    )
    from visualdynamics.theme import theme

    assert curve_color(1, theme('light')) == CURVE_COLORS_LIGHT[1]
    assert curve_color(1, theme('dark')) == CURVE_COLORS[1]
    assert curve_color(11, theme('light')) == CURVE_COLORS_LIGHT[1]


def test_a_plot_draws_the_shade_of_its_theme(qt_app):
    import numpy as np
    import pyqtgraph as pg

    from visualdynamics.core.data import Psd
    from visualdynamics.plot import CURVE_COLORS, CURVE_COLORS_LIGHT, build_plots

    f = np.linspace(1.0, 100.0, 64)
    psd = Psd(f, np.full((3, 64), 1e-3), response_dof=['1X+', '2X+', '3X+'],
              reference_dof=['1X+', '2X+', '3X+'],
              ordinate_dim=['acceleration**2/frequency'] * 3)
    drawn = {}
    for name in ('light', 'dark'):
        layout = pg.GraphicsLayoutWidget()
        build_plots(layout, [('p', psd, None)], theme=name)
        plot = next(item for item in layout.ci.items
                    if hasattr(item, 'listDataItems'))
        drawn[name] = [c.opts['pen'].color().name()
                       for c in plot.listDataItems() if c.name()]
    assert drawn['light'] == CURVE_COLORS_LIGHT[:3]
    assert drawn['dark'] == CURVE_COLORS[:3]


def test_the_report_page_wears_the_shade_of_its_theme(qt_app, tmp_path):
    """The page is handed both shades and picks by its own theme, the
    legend's swatches with the curves."""
    import pytest

    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from conftest import web_close, web_read
    from PySide6.QtCore import QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from test_export_html import _pair

    import visualdynamics
    from visualdynamics.plot import CURVE_COLORS, CURVE_COLORS_LIGHT

    measured, spec = _pair()
    probe = ("(() => { const spans = document.querySelectorAll('.legend span');"
             "if (spans.length < 2) return null;"
             "return spans[1].style.getPropertyValue('--swatch'); })()")
    for name, palette in (('light', CURVE_COLORS_LIGHT), ('dark', CURVE_COLORS)):
        path = visualdynamics.export_html(tmp_path / f'{name}.html', measured,
                                          specification=spec,
                                          channel='101Z+', theme=name)
        view = QWebEngineView()
        view.resize(700, 400)
        view.load(QUrl.fromLocalFile(str(path)))
        view.show()
        try:
            swatch = web_read(view, probe)
        finally:
            web_close(view)
        assert swatch.strip().lower() == palette[0], (name, swatch)
