"""The application's icon: four FRFs on the waterfall stage.

The 3-D reading of data is what this application looks like — the
waterfall stage, channels receding, level-colored — so that is what
the tile shows (Brandon, 2026-08-23; it was a plate carrying a mode
before). The curves are real: accelerances synthesized from the
demonstration plate's own eigensolution, noiseless on purpose, drawn
from the exact viewpoint `viz.waterfall.place_camera` gives the stage.

Drawn with QPainter and numpy alone. An icon that needed a GL context
to exist could not be built on a machine without one, and under Qt's
offscreen platform VTK's render widget takes the process down with it
— which is exactly the situation an icon build runs in. The synthesis
is a fifth of a second and the drawing is cached, so the app pays for
its icon once.

Two deliberate departures from the honest plot, both because a tile
is not a plot: the vertical is stretched to fill the tile (the
stage's view projects wide and short), and the color/height range is
clipped to the top `DECADES` decades — unclipped, one deep
antiresonance eats the whole range and every peak reads flat green.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from PySide6.QtGui import QIcon, QImage

from functools import cache

import numpy as np

#: the synthesis: a drive at the plate's corner, responses spread
#: across it, the band the plate's modes live in
RESPONSES = ('101Z+', '110Z+', '1304Z+', '1313Z+')
DRIVE = '101Z+'
BAND = (60.0, 1500.0)
POINTS = 480
DAMPING = 0.02
#: color and height read the top decades only — see the module note
DECADES = 2.4

#: the tile's inset from the icon square — macOS insets its own
TILE_INSET = 0.06
#: the even border the curves keep clear on every side, as a share
#: of the *tile* — about what Apple's own marks leave (Brandon,
#: 2026-08-23; the fill crept too far). Measured from the tile's
#: edge to the stroke's outside edge, not the polyline — the round
#: caps would eat the border — and the first cut measured from the
#: icon square instead, which equals the tile inset exactly: ink
#: started at the tile's own edge and the border read as none.
MARGIN = 0.05
#: amplitude exaggeration over the stage's own proportion, for a tile's
#: worth of drama
AMPLITUDE = 1.35
#: curve thickness as a share of the tile size
THICKNESS = 0.020
#: below this many pixels three curves read; four are a smear
SMALL = 96


def colormap(values: ArrayLike) -> np.ndarray:
    """`values` in 0..1 as an (N, 3) array of RGB — `theme.VIRIDIS`,
    which everything that paints by level reads from."""
    from ..theme import colormap as ramp

    return ramp(values)


@cache
def synthesized_levels() -> np.ndarray:
    """The four curves, as levels in 0..1: color and height alike.

    Noiseless accelerances from the demonstration plate's own modes
    (`ShapeSet.synthesize_frf`), log magnitude, clipped to the top
    `DECADES` decades of the set and normalized. Real data, no
    measurement — the icon shows what the instrument shows, clean.
    """
    from ..demo import plate

    shapes = plate.build().eigensolution(
        maximum_frequency=2000.0, damping=DAMPING)
    freq = np.linspace(BAND[0], BAND[1], POINTS)
    magnitudes = np.stack([
        np.abs(shapes.synthesize_frf(freq, [dof], [DRIVE], power=2)[0])
        for dof in RESPONSES])
    rows = np.log10(magnitudes)
    return np.clip((rows - (rows.max() - DECADES)) / DECADES, 0.0, 1.0)


@cache
def stage_basis() -> tuple[np.ndarray, np.ndarray]:
    """(right, up) — the stage's home view. See
    `viz.waterfall.stage_basis`, which owns it: the icon's geometry
    *is* the app's default 3-D view of data, and so is the report's
    3-D figure."""
    from ..viz.waterfall import stage_basis as basis

    return basis()


def curve_count(size: int) -> int:
    """How many curves a tile this size can carry legibly."""
    return 3 if size < SMALL else 4


def screen_curves(size: int) -> list[np.ndarray]:
    """The curves as (N, 2) pixel paths, front first.

    Projected through the stage basis, then fitted per axis into the
    tile — the stage's view projects wide and short, and a uniform fit
    left the tile empty top and bottom. An icon may stretch where a
    plot may not.
    """
    from ..viz.waterfall import STAGE

    sx, sy, sz = STAGE
    level = synthesized_levels()
    n = curve_count(size)
    right, up = stage_basis()
    u = np.linspace(0.0, 1.0, level.shape[1])
    screens = []
    for k in range(n):
        station = (k / max(n - 1, 1)) * sy
        world = np.stack([u * sx, np.full_like(u, station),
                          level[k] * sz * AMPLITUDE], axis=-1)
        screens.append(np.stack([world @ right, -(world @ up)], axis=-1))
    stacked = np.vstack(screens)
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    tile = size * (1.0 - 2.0 * TILE_INSET)
    margin = (size * TILE_INSET + tile * MARGIN
              + size * THICKNESS / 2.0)
    scale = np.array([(size - 2 * margin) / (high[0] - low[0]),
                      (size - 2 * margin) / (high[1] - low[1])])
    center = (low + high) / 2.0
    return [(screen - center) * scale + size / 2.0 for screen in screens]


@cache
def draw_app_icon(size: int) -> QImage:
    """The icon at `size` pixels square, as a QImage.

    Cached: the synthesis is a fifth of a second, the drawing less,
    and every caller asks for the same handful of sizes.
    """
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QImage,
        QPainter,
        QPainterPath,
        QPen,
    )

    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # the tile: macOS insets its icons, and one that fills its square
    # sits visibly larger than everything beside it in the Dock. Pure
    # black, no gradient — the curves are the mark, and black is what
    # makes viridis carry (Brandon, 2026-08-23)
    inset = size * TILE_INSET
    tile = QRectF(inset, inset, size - 2 * inset, size - 2 * inset)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(0, 0, 0)))
    radius = size * 0.225
    painter.drawRoundedRect(tile, radius, radius)
    # a curve may reach the tile's edge but never past its corners
    clip = QPainterPath()
    clip.addRoundedRect(tile, radius, radius)
    painter.setClipPath(clip)

    level = synthesized_levels()
    curves = screen_curves(size)
    pen = QPen()
    pen.setWidthF(max(size * THICKNESS, 1.5))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    for k in range(len(curves) - 1, -1, -1):      # back to front
        points = curves[k]
        rgb = colormap(level[k])
        for i in range(len(points) - 1):
            pen.setColor(QColor(int(rgb[i, 0]), int(rgb[i, 1]),
                                int(rgb[i, 2])))
            painter.setPen(pen)
            painter.drawLine(QPointF(*points[i]), QPointF(*points[i + 1]))
    painter.end()
    return image


@cache
def app_icon() -> QIcon:
    """The icon at the sizes a window manager asks for.

    Several rather than one scaled: below about a hundred pixels the
    fourth curve goes and the rest thicken, because four ridges over
    thirty pixels is a smear. `tools/make_app.py` renders the same
    function into a Mac `.icns`, so the Dock and the window wear one
    mark.
    """
    from PySide6.QtGui import QIcon, QPixmap

    icon = QIcon()
    for size in (16, 32, 64, 128, 256, 512):
        icon.addPixmap(QPixmap.fromImage(draw_app_icon(size)))
    return icon
