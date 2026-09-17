"""Glyphs drawn from the package's own SVG files, without Qt's SVG module.

The sanctioned Qt modules stop short of QtSvg (AGENTS.md rule 3), and
the one glyph so far — the drawer tab's chevron, drawn in Claude Design
(Brandon, 2026-09-03) — is a six-vertex outline that needs no renderer:
its vertices are read straight out of the path and painted as a
polygon, in whatever color the palette says, at whatever pixel ratio
the screen has. A file replaces a file when the design changes.

The parser takes exactly the dialect the files are asked for: absolute
M/L/Z, straight edges, one path. Anything else is refused rather than
drawn wrong.
"""

from __future__ import annotations

import re
from importlib import resources

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF

_TOKENS = re.compile(r'[A-Za-z]|-?(?:\d+\.?\d*|\.\d+)')

#: the way the drawer-tab chevron points when the drawer is *closed*,
#: as a clockwise rotation of the file's up-pointing glyph — away
#: from the edge the drawer lives on, toward where the edge will move
_CLOSED_TURN = {'bottom': 0, 'left': 90, 'top': 180, 'right': 270}


def polygon_of(svg: str) -> tuple[list[tuple[float, float]], float]:
    """The vertices of the one straight-edged path an SVG holds, with
    the side of its square viewBox.

    Raises ValueError for anything outside the dialect: curves,
    relative commands, several paths, a non-square viewBox.
    """
    box = re.search(r'viewBox="0 0 (\S+) (\S+)"', svg)
    if box is None or box.group(1) != box.group(2):
        raise ValueError('the glyph needs a square viewBox at the origin')
    paths = re.findall(r'<path\b[^>]*\bd="([^"]*)"', svg)
    if len(paths) != 1:
        raise ValueError(f'one path expected, {len(paths)} found')
    tokens = _TOKENS.findall(paths[0])
    vertices: list[tuple[float, float]] = []
    command = None
    while tokens:
        token = tokens.pop(0)
        if token.isalpha():
            if token not in 'MLZ':
                raise ValueError(f'path command {token!r} is not drawn: '
                                 'straight absolute edges only')
            command = token
            continue
        if command not in 'ML' or not tokens:
            raise ValueError('malformed path data')
        vertices.append((float(token), float(tokens.pop(0))))
    if len(vertices) < 3:
        raise ValueError('a polygon needs at least three vertices')
    return vertices, float(box.group(1))


def _load(name: str) -> tuple[list[tuple[float, float]], float]:
    file = resources.files('visualdynamics.gui') / 'icons' / f'{name}.svg'
    return polygon_of(file.read_text(encoding='utf-8'))


def glyph(name: str, color: QColor, size: int = 16,
          turn: float = 0) -> QIcon:
    """An icon of a packaged glyph, filled with `color`, turned
    `turn` degrees clockwise about its center, `size` pixels a side.

    Rendered at one and two device pixels per point so a Retina
    screen gets a sharp edge; the icon hands out whichever fits.
    """
    vertices, side = _load(name)
    polygon = QPolygonF([QPointF(x, y) for x, y in vertices])
    icon = QIcon()
    for ratio in (1, 2):
        pixmap = QPixmap(size * ratio, size * ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.scale(size * ratio / side, size * ratio / side)
        painter.translate(side / 2, side / 2)
        painter.rotate(turn)
        painter.translate(-side / 2, -side / 2)
        painter.drawPolygon(polygon)
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        icon.addPixmap(pixmap)
    return icon


def drawer_glyph(edge: str, open_: bool, color: QColor,
                 size: int = 16) -> QIcon:
    """The chevron a drawer tab wears: pointing away from `edge`
    ('bottom', 'top', 'left' or 'right') while the drawer is closed,
    toward it once open — where a click will move the edge."""
    return glyph('drawer-tab', color, size,
                 _CLOSED_TURN[edge] + (180 if open_ else 0))
