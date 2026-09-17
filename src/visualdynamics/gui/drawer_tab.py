"""The handle a collapsible drawer wears on its edge.

Drawn to the design Brandon had made in Claude Design (2026-09-03,
`icons/drawer-tab-shape.svg` is the reference): a flat tab with
rounded shoulders and feet that flare out into the edge it sits on, the
word in the middle and a chevron either side of it pointing where a
click will move the edge. The reference is one fixed width, so the
shape is drawn here from its radii at whatever width the label needs,
and the colors come from the palette so light and dark both work.

Painted by hand rather than styled: a stylesheet gives a box, and the
flared feet are what make it read as a handle on an edge rather than a
button floating near one.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QAbstractButton, QWidget

from .glyphs import drawer_glyph


class DrawerTab(QAbstractButton):
    """A checkable handle for a drawer on the bottom edge: checked is
    open. The edge is fixed for now; a drawer on another edge would
    turn the shape and the text, and the glyph already knows how."""

    #: the reference's radii: the rounded top corners and the concave
    #: feet, in pixels
    SHOULDER = 8
    FOOT = 6
    #: the reference's height, with the body starting 1.5 px down so
    #: the stroke sits on whole pixels
    HEIGHT = 26
    TOP = 1.5
    #: the chevron a side, and its distances from foot and word
    GLYPH = 12
    INSET = 4
    GAP = 7

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        # The tab floats over the 3-D view, and the VTK widget takes a
        # native window (`winId()`), which makes every sibling up the
        # tree native too — this one included, and it has to be, or it
        # could never draw over the view. A native child has a surface
        # of its own, and it needs an alpha channel or everything
        # outside the flared outline shows as a gray box on the black
        # scene (Brandon, 2026-09-04). Declared translucent here, before
        # the surface exists, so it is made with one. Not enough on its
        # own — see `paintEvent` for the rest of that story.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def sizeHint(self) -> QSize:
        word = self.fontMetrics().horizontalAdvance(self.text())
        flank = self.FOOT + self.INSET + self.GLYPH + self.GAP
        return QSize(2 * flank + word, self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def outline(self) -> QPainterPath:
        """The tab's edge at its current size, open along the base so
        no line is drawn across the edge it sits on."""
        w, h = self.width(), self.height()
        top, base = self.TOP, h - 0.5
        foot, shoulder = self.FOOT, self.SHOULDER
        path = QPainterPath(QPointF(0, base))
        # a concave foot: a quarter circle centered on the edge, from
        # straight below its center round to its right
        path.arcTo(QRectF(-foot, base - 2 * foot, 2 * foot, 2 * foot),
                   270, 90)
        path.lineTo(foot, top + shoulder)
        path.arcTo(QRectF(foot, top, 2 * shoulder, 2 * shoulder), 180, -90)
        path.lineTo(w - foot - shoulder, top)
        path.arcTo(QRectF(w - foot - 2 * shoulder, top,
                          2 * shoulder, 2 * shoulder), 90, -90)
        path.lineTo(w - foot, base - foot)
        path.arcTo(QRectF(w - foot, base - 2 * foot, 2 * foot, 2 * foot),
                   180, 90)
        return path

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        # Erase to nothing first. A native child does not paint on its
        # own surface: it paints into the top-level's backing image, at
        # its place, and Qt copies that patch to the child's surface on
        # flush. Translucent widgets get no background erase, so the
        # patch keeps whatever was last painted there — clear while the
        # image is fresh, the window color once a relayout of the host
        # (the console opening or closing) has painted right through
        # it. That was the gray box, and why it came and went (Brandon,
        # 2026-09-04 and 2026-09-12): the tab's paint alone drew over
        # the fill and never removed it. Source mode writes the
        # transparent pixels rather than blending them, which would do
        # nothing.
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        role = QPalette.ColorRole.Button
        if self.isDown():
            role = QPalette.ColorRole.Midlight
        elif self.underMouse():
            role = QPalette.ColorRole.Light
        outline = self.outline()
        painter.fillPath(outline, palette.color(role))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))
        painter.drawPath(outline)

        color = palette.color(QPalette.ColorRole.ButtonText)
        glyph = drawer_glyph('bottom', self.isChecked(), color, self.GLYPH)
        pixmap = glyph.pixmap(QSize(self.GLYPH, self.GLYPH),
                              self.devicePixelRatioF())
        body = QRect(0, int(self.TOP), self.width(),
                     self.height() - int(self.TOP))
        y = body.top() + (body.height() - self.GLYPH) // 2
        painter.drawPixmap(self.FOOT + self.INSET, y, pixmap)
        painter.drawPixmap(self.width() - self.FOOT - self.INSET - self.GLYPH,
                           y, pixmap)
        painter.setPen(color)
        painter.drawText(body, Qt.AlignmentFlag.AlignCenter, self.text())

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.update()
