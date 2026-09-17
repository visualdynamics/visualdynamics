"""The packaged glyphs: read from their SVG, drawn by QPainter."""

import re
from importlib import resources
from pathlib import Path

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor

from visualdynamics.gui.glyphs import drawer_glyph, glyph, polygon_of

ROOT = Path(__file__).resolve().parent.parent


def _image(icon, size=16):
    return icon.pixmap(QSize(size, size)).toImage()


def _points(icon, size=64):
    """Which way a chevron points. Its rows carry about the same ink
    from tip to base, so the count says nothing; what marks the tip is
    ink gathered on the centreline where the base's is spread wide."""
    image = _image(icon, size)
    inked = [(x, y) for y in range(image.height())
             for x in range(image.width())
             if image.pixelColor(x, y).alpha() > 128]
    assert inked, 'nothing drawn'
    xs, ys = [x for x, _ in inked], [y for _, y in inked]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def spread(pixels, center):
        return sum(abs(p - center) for p in pixels) / len(pixels)

    if span_x > span_y:                 # wider than tall: up or down
        top = [x for x, y in inked if y < min(ys) + span_y / 4]
        bottom = [x for x, y in inked if y > max(ys) - span_y / 4]
        return 'up' if spread(top, cx) < spread(bottom, cx) else 'down'
    left = [y for x, y in inked if x < min(xs) + span_x / 4]
    right = [y for x, y in inked if x > max(xs) - span_x / 4]
    return 'left' if spread(left, cy) < spread(right, cy) else 'right'


def test_the_shipped_chevron_is_one_closed_polygon():
    svg = (resources.files('visualdynamics.gui') / 'icons'
           / 'drawer-tab.svg').read_text()
    vertices, side = polygon_of(svg)
    assert side == 16 and len(vertices) == 6
    assert '<metadata' not in svg, 'the design tool’s manifest was stripped'


@pytest.mark.parametrize('data', [
    'M0 0 C 1 1 2 2 3 3 Z',            # a curve
    'M0 0 l 1 1 2 2 Z',                # relative
    'M0 0 L 1 1',                      # two vertices
])
def test_paths_outside_the_dialect_are_refused(data):
    with pytest.raises(ValueError):
        polygon_of(f'<svg viewBox="0 0 16 16"><path d="{data}"/></svg>')


def test_two_paths_or_an_oblong_box_are_refused(qt_app):
    with pytest.raises(ValueError):
        polygon_of('<svg viewBox="0 0 16 16"><path d="M0 0 L1 0 L1 1 Z"/>'
                   '<path d="M0 0 L1 0 L1 1 Z"/></svg>')
    with pytest.raises(ValueError):
        polygon_of('<svg viewBox="0 0 16 8"><path d="M0 0 L1 0 L1 1 Z"/></svg>')


@pytest.mark.parametrize('edge, closed_towards', [
    ('bottom', 'up'), ('top', 'down'), ('left', 'right'), ('right', 'left')])
def test_a_closed_drawer_points_away_from_its_edge(qt_app, edge,
                                                   closed_towards):
    """And an open one points at it: where a click will move the edge."""
    def direction(open_):
        return _points(drawer_glyph(edge, open_, QColor('black'), 64))

    opposite = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    assert direction(False) == closed_towards
    assert direction(True) == opposite[closed_towards]


def test_the_glyph_is_drawn_in_the_color_asked_for(qt_app):
    # drawn large: at 16 px the stroke is about a pixel and a half
    # wide and no pixel is fully covered, and a part-covered pixel's
    # color comes back through premultiplication rounding
    image = _image(glyph('drawer-tab', QColor('#c02020'), 64), 64)
    covered = {image.pixelColor(x, y).name()
               for y in range(image.height()) for x in range(image.width())
               if image.pixelColor(x, y).alpha() == 255}
    assert covered == {'#c02020'}


def test_the_glyph_offers_a_retina_pixmap(qt_app):
    icon = glyph('drawer-tab', QColor('black'), 16)
    assert {s.width() for s in icon.availableSizes()} == {16, 32}


def test_the_glyph_files_ship_in_every_package():
    """Package data for the wheel, a datas entry for the bundles: a
    glyph that renders from the checkout and not from the installed
    app is the miss this pins."""
    pyproject = (ROOT / 'pyproject.toml').read_text()
    assert re.search(r'visualdynamics = \[.*"gui/icons/\*\.svg".*\]',
                     pyproject)
    spec = (ROOT / 'packaging' / 'visualdynamics.spec').read_text()
    assert "'visualdynamics/gui/icons'" in spec
