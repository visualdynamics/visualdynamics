"""Type icons for the project tree.

Icons are drawn at runtime with QPainter rather than shipped as assets, so
they stay sharp at any DPI and need no files. Each type gets a distinct
shape as well as a distinct color — shape carries the meaning, so the icons
are still tellable apart at 16 px and without relying on color vision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import DataArray

import math
from functools import cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)

SIZE = 64

COLORS = {
    'Geometry': '#1f77b4',      # blue
    'TimeHistory': '#2ca02c',   # green
    'Frf': '#ff7f0e',           # orange
    'Psd': '#9467bd',           # purple
    # the same purple: a banded PSD is the same measurement on
    # different bins, and says which it is by its shape
    'OctavePsd': '#9467bd',
    'Spectrum': '#17becf',      # teal
    'Specification': '#d62728',  # red: the thing being aimed at
    'OctaveSpecification': '#d62728',  # the same requirement, on bands
    'Srs': '#00a878',           # jade
    # a shock target is aimed at just as a random one is, so it takes the
    # same red and says which kind it is by its shape
    'ShockSpecification': '#d62728',
    # a transient target is aimed at just as the other two are, and says
    # which kind it is by its shape
    'TransientSpecification': '#d62728',
    # a sine sweep target is aimed at too, and says which kind it is by
    # its shape: a chirp climbing in frequency
    'SineSweepSpecification': '#d62728',
    # the measured level a sweep actually reached: the same chirp in
    # the measured spectra's teal rather than the targets' red
    'SineLevel': '#17becf',
    # the extraction, whole: one object holding every tone's level
    'SineLevelSet': '#17becf',
    # a tone's requirement as a curve: derived from the specification
    # on demand, wearing the targets' red like its parent
    'SineTarget': '#d62728',
    'Coherence': '#bcbd22',     # olive
    'MultipleCoherence': '#bcbd22',
    # signal over noise: the driven line riding high over the ambient
    # one — a ratio, so it borrows the coherences' olive
    'ShapeSet': '#d94f8a',      # magenta
    # sky. Not gray: gray is what an *unfilled* slot wears
    # (`EMPTY_COLOR`), so a real channel table drawn in it read as a
    # placeholder for one. Lighter and brighter than Geometry's blue,
    # which is the only other blue and sits a row or two away.
    'ChannelTable': '#4bb3fd',
    'Report': '#d2c14e',        # manila, like the folder it replaces
    'Photos': '#b07d62',        # sienna, a print's warm border
    'MatchedModes': '#3fb950',  # green: agreement found
    'Test': '#6e7681',          # slate: the container itself
}


# What a record *is*, as opposed to what object holds it. A time history can
# carry accelerations, forces, voltages and temperatures side by side, so the
# quantity belongs on the record's own icon rather than in its label.
#
# Shape carries the meaning here as everywhere else: displacement, velocity
# and acceleration share one trace and are told apart by the dot notation a
# dynamicist already reads — x, x-dot, x-double-dot — while the other three
# get shapes of their own.
QUANTITY_COLORS = {
    # keyed by the *dimension* the units engine reports, which is what a
    # record actually carries. 'displacement' is the word a person uses
    # and 'length' is the dimension pint knows; keying the icon by the
    # word meant `quantity_of('length')` answered None and every
    # displacement channel ever measured fell back to its object's icon
    # — the blue trace below was unreachable for real data.
    'length': '#1f77b4',         # blue — displacement
    'strain': '#8c564b',         # brown
    'pressure': '#7f7f7f',       # gray
    'velocity': '#17becf',       # cyan
    'acceleration': '#9467bd',   # purple
    'force': '#2ca02c',          # green
    'temperature': '#d62728',    # red
    'voltage': '#e8a33d',        # gold
    # the rotations at a virtual point (core.transform) wear the
    # color of their linear counterpart — the color says the kind
    # of quantity, the shape says it turns: an arc where the linear
    # one is a trace, with the same dot notation over it, and a twist
    # about an axis where the force is an arrow into a surface
    'angle': '#1f77b4',
    'angular_velocity': '#17becf',
    'angular_acceleration': '#9467bd',
    'moment': '#2ca02c',
    # modal responses through mass-normalized shapes (units.py's modal
    # family): the linear glyph, since a modal acceleration is still
    # an acceleration to the eye, drawn smaller with an M in the corner
    'modal_length': '#1f77b4',
    'modal_velocity': '#17becf',
    'modal_acceleration': '#9467bd',
    'modal_force': '#2ca02c',
}


def _pen(color, width=6):
    pen = QPen(QColor(color))
    pen.setWidth(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _curve(points):
    path = QPainterPath(QPointF(*points[0]))
    for x, y in points[1:]:
        path.lineTo(x, y)
    return path


def _draw_geometry(painter, color):
    """Nodes joined by tracelines."""
    points = [(12, 50), (52, 48), (32, 14)]
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPolygon([QPointF(*p) for p in points])
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    for x, y in points:
        painter.drawEllipse(QPointF(x, y), 6.5, 6.5)


#: A pulse and its ringdown. The shape a time history wears, and the
#: shape its *target* wears — a transient specification is a waveform
#: wanted sample by sample, so the two say the same thing and differ by
#: color and by the tolerance band around the target.
_RINGDOWN = [(8, 32), (17, 32), (21, 12), (26, 48), (31, 25), (36, 38),
             (41, 29), (46, 34), (51, 32), (56, 32)]

#: how far a specification's tolerance curves sit either side of it, and
#: how faint. One rule for all three, so a target reads as a target
#: whatever kind of data it is a target for.
_TOLERANCE = 9
_TOLERANCE_ALPHA = 120


def _tolerance_pen(color):
    faint = QColor(color)
    faint.setAlpha(_TOLERANCE_ALPHA)
    return _pen(faint, 4)


def _draw_time_history(painter, color):
    """A pulse and its ringdown, on the zero line.

    A measured waveform rather than a tidy sine: the sine said
    "oscillation" and this says "a record of something that happened",
    which is what a time history holds. Its specification wears the same
    shape (see `_draw_transient_specification`) — same measurement,
    one measured and one asked for.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, 3))
    painter.drawLine(QPointF(8, 32), QPointF(56, 32))
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve(_RINGDOWN))


def _draw_frf(painter, color):
    """A resonant peak."""
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (8, 47), (18, 45), (25, 41), (29, 15), (33, 41), (39, 43),
        (45, 31), (49, 43), (56, 45)]))


#: the density both a PSD and its target are drawn from
_PSD_SPINE = [(9, 46), (17, 39), (23, 25), (31, 35), (39, 19), (47, 33),
              (55, 42)]


def _psd_body(painter, color):
    """The filled density itself — the PSD icon, and the middle curve of
    a random specification."""
    area = _curve(_PSD_SPINE)
    area.lineTo(55, 55)
    area.lineTo(9, 55)
    area.closeSubpath()
    fill = QColor(color)
    fill.setAlpha(90)
    painter.fillPath(area, QBrush(fill))
    painter.setPen(_pen(color, 5))
    painter.drawPath(_curve(_PSD_SPINE))


def _draw_psd(painter, color):
    """A filled spectral density."""
    _psd_body(painter, color)


def _draw_octave_psd(painter, color):
    """A density on proportional bands: bars that widen to the right.

    The same purple as the PSD it came from, because it *is* that PSD —
    same quantity, same units, the same area under it. What differs is
    the bins, so the bins are what the icon draws: contiguous bars, each
    wider than the last, where the narrowband PSD is one smooth filled
    curve. Told apart by shape, the way a specification is told from the
    measurement it bounds.
    """
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    # Four bands, each half again as wide as the one before, drawn edge
    # to edge. The widening is the whole glyph: evenly spaced bars are
    # what `Spectrum` already draws, and a banded PSD that looked like
    # one would be told apart only by its color.
    left = 9.0
    for width, top in ((5.8, 17), (8.7, 30), (13.0, 40), (19.5, 24)):
        painter.drawRect(QRectF(left, 52 - top, width - 1.2, top))
        left += width


def _draw_spectrum(painter, color):
    """Discrete spectral lines."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    for x, height in [(13, 20), (23, 36), (33, 27), (43, 45), (53, 15)]:
        painter.drawRoundedRect(QRectF(x - 4, 52 - height, 8, height), 2, 2)


def _draw_specification(painter, color):
    """A random target: the required density, and the room either side.

    The PSD it is a target for, in red with tolerance curves — the rule
    every specification follows. It used to be concentric rings, which
    said "target" plainly enough but said nothing about *what kind*, and
    left the three specifications looking like three unrelated things
    rather than three targets.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for drop in (-_TOLERANCE, _TOLERANCE):
        painter.setPen(_tolerance_pen(color))
        painter.drawPath(_curve([(x, y + drop) for x, y in _PSD_SPINE]))
    _psd_body(painter, color)


def _draw_srs(painter, color):
    """The shape every shock response spectrum has.

    It climbs with frequency, turns over, and flattens onto the peak of
    the shock itself — a stiff enough oscillator just rides the base. No
    fill, so it does not read as the PSD it sits beside.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([(10, 52), (20, 44), (28, 26), (36, 18),
                             (46, 20), (55, 20)]))


def _draw_shock_specification(painter, color):
    """A shock target: the required spectrum, and the room either side.

    Three curves of the SRS shape rather than the concentric rings a
    random specification wears — both are targets, and this says which
    kind without needing the color to.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # the SRS shape, lifted a little to leave the lower tolerance curve
    # room inside the tile
    spine = [(10, 46), (20, 38), (28, 22), (36, 15), (46, 17), (55, 17)]
    for drop in (-_TOLERANCE, _TOLERANCE):
        painter.setPen(_tolerance_pen(color))
        painter.drawPath(_curve([(x, y + drop) for x, y in spine]))
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve(spine))


def _draw_sine_sweep_specification(painter, color):
    """A sine sweep target: a chirp, the cycles tightening rightward.

    The other targets wear this red too; the accelerating wave is what
    says *swept sine* without needing the color to.
    """
    import math

    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, 6))
    points = []
    for i in range(33):
        x = 8 + 48 * i / 32
        # phase quadratic in x: frequency rising left to right
        phase = 2.6 * math.pi * (i / 32) ** 2 * 3
        points.append((x, 32 - 16 * math.sin(phase)))
    painter.drawPath(_curve(points))


def _draw_transient_specification(painter, color):
    """A transient target: the waveform itself, wanted sample by sample.

    A pulse and its ringdown, which is what a transient specification is
    — where a shock specification is a spectrum. The two are different
    tests and read differently on the tree: a curve in *time* against a
    curve in frequency.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for drop in (-_TOLERANCE, _TOLERANCE):
        painter.setPen(_tolerance_pen(color))
        painter.drawPath(_curve([(x, y + drop) for x, y in _RINGDOWN]))
    painter.setPen(_pen(color, 3))
    painter.drawLine(QPointF(8, 32), QPointF(56, 32))
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve(_RINGDOWN))


def _draw_coherence(painter, color):
    """A trace riding just under unity, dipping where the answer is poor.

    The dashed ceiling is what makes it read as a bounded ratio rather than
    one more spectrum: coherence is only ever interesting relative to one.
    """
    ceiling = QColor(color)
    ceiling.setAlpha(130)
    painter.setPen(QPen(ceiling, 4))
    painter.drawLine(QPointF(8, 15), QPointF(56, 15))
    # one deep notch rather than several shallow ones: at 16 px a row of
    # small dips is just a wave, and there is already a wave in the set
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (8, 24), (16, 23), (22, 25), (27, 38), (32, 50), (37, 38),
        (42, 25), (48, 23), (56, 24)]))


def _draw_multiple_coherence(painter, color):
    """The same ceiling and notch, over several traces summed into one.

    Ordinary coherence is one response against one reference; this is one
    response against all of them at once. The stacked strokes under the
    curve are the references it accounts for — enough to tell the two apart
    in a tree at 16 px, which is the only place it has to work.
    """
    ceiling = QColor(color)
    ceiling.setAlpha(130)
    painter.setPen(QPen(ceiling, 4))
    painter.drawLine(QPointF(8, 13), QPointF(56, 13))
    stack = QColor(color)
    stack.setAlpha(90)
    painter.setPen(QPen(stack, 4))
    for y in (46, 53):
        painter.drawLine(QPointF(14, y), QPointF(50, y))
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (8, 22), (18, 21), (25, 24), (32, 36), (39, 24), (46, 21), (56, 22)]))


def _draw_log_axis(painter, color):
    """Decade ticks closing up to the right over a baseline: the
    frequency axis in logarithms, which is what the button switches."""
    painter.setPen(_pen(color, 5))
    painter.drawLine(QPointF(6, 46), QPointF(58, 46))
    faint = QColor(color)
    faint.setAlpha(150)
    for k, x in enumerate((8, 22, 30, 36, 40, 43, 46, 48, 50, 52, 54, 56)):
        tall = k in (0, 1, 6)
        painter.setPen(_pen(color if tall else faint, 5 if tall else 3))
        painter.drawLine(QPointF(x, 46), QPointF(x, 22 if tall else 34))


def _draw_octave(painter, color):
    """Wide flat bands over a narrowband wiggle.

    What the button previews on the plot, in miniature: the banded
    steps standing over the fine-grained spectrum they integrate.
    """
    import math as _math

    faint = QColor(color)
    faint.setAlpha(90)
    painter.setPen(_pen(faint, 3))
    painter.drawPath(_curve([
        (x, 40 - 10 * _math.sin(x / 3.0) - 8 * _math.sin(x / 7.3))
        for x in range(6, 59)]))
    painter.setPen(_pen(color, 5))
    for left, right, top in ((6, 22, 38), (22, 40, 26), (40, 58, 33)):
        painter.drawLine(QPointF(left, top), QPointF(right, top))
        painter.drawLine(QPointF(right, top),
                         QPointF(right, 26 if top == 38 else 33))


def _draw_truncate(painter, color):
    """A trace with its ends grayed away behind two cut lines.

    What the button puts on the plot, in miniature: the kept stretch
    at full strength between two edges, the discarded lead-in and tail
    faded — the fading is the point, or the icon reads as a plain
    time history with brackets.
    """
    faint = QColor(color)
    faint.setAlpha(70)
    trace = [(x, 32 - 14 * math.sin(x / 6.5)) for x in range(6, 59)]
    painter.setPen(_pen(faint, 4))
    painter.drawPath(_curve([point for point in trace if point[0] <= 20]))
    painter.drawPath(_curve([point for point in trace if point[0] >= 44]))
    painter.setPen(_pen(color, 4))
    painter.drawPath(_curve([point for point in trace
                             if 18 <= point[0] <= 46]))
    painter.setPen(_pen(color, 4))
    for x in (20, 44):
        painter.drawLine(QPointF(x, 8), QPointF(x, 56))


def _draw_shocks(painter, color):
    """Two transients on a baseline, each bracketed by its window.

    What the button puts on the plot, in miniature: the events, and the
    spans each spectrum would be computed over. The brackets are the
    point — the trace alone would read as a time history icon.
    """
    faint = QColor(color)
    faint.setAlpha(70)
    for first, last in ((8, 30), (34, 56)):
        painter.fillRect(QRectF(first, 12, last - first, 40), QBrush(faint))
    painter.setPen(_pen(color, 4))
    painter.drawLine(QPointF(6, 44), QPointF(58, 44))
    for at in (15, 41):
        # the pulse, and one bounce of the ringdown after it
        painter.drawPath(_curve([(at - 3, 44), (at, 18), (at + 3, 44),
                                 (at + 6, 36), (at + 9, 44)]))


def _draw_bars(painter, color):
    """A bar chart with one bar over a threshold.

    The RMS-error reading: how far each channel sits from what was
    asked for, and the line it must not cross.
    """
    painter.setPen(Qt.PenStyle.NoPen)
    faint = QColor(color)
    faint.setAlpha(110)
    for x, height, hot in ((12, 22, False), (23, 34, True),
                           (34, 16, False), (45, 28, True)):
        painter.setBrush(QBrush(color if hot else faint))
        painter.drawRect(QRectF(x, 48 - height, 8, height))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    pen = QPen(QColor(color), 3)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawLine(QPointF(8, 22), QPointF(58, 22))


def _draw_filter(painter, color):
    """A low-pass magnitude response: flat, then rolling off.

    The corner is marked the way the kurtosis glyph marks its bounds —
    a faint dashed vertical — because the corner is the one number the
    view exists to set.
    """
    faint = QColor(color)
    faint.setAlpha(110)
    pen = QPen(faint, 3)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLine(QPointF(34, 10), QPointF(34, 54))
    path = QPainterPath()
    path.moveTo(6, 16)
    path.lineTo(26, 16)
    # the knee at the corner, then the roll-off to the floor
    path.cubicTo(38, 16, 36, 48, 58, 50)
    painter.setPen(QPen(QColor(color), 4))
    painter.drawPath(path)


def _draw_kurtosis(painter, color):
    """A distribution with heavy tails, over the band it should be in.

    Not a bar chart glyph: every reading on this bar that *is* a bar
    chart already wears one, and what tells this button apart is what
    it measures — the shape of the distribution rather than a level.
    A bell with its tails lifted, between two bounds.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    faint = QColor(color)
    faint.setAlpha(110)
    pen = QPen(faint, 3)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    for x in (14, 50):
        painter.drawLine(QPointF(x, 10), QPointF(x, 52))
    path = QPainterPath()
    path.moveTo(6, 50)
    # the tails ride high on the way in and out — the peaky record
    # this reading exists to catch
    path.cubicTo(20, 48, 24, 14, 32, 14)
    path.cubicTo(40, 14, 44, 48, 58, 50)
    painter.setPen(QPen(QColor(color), 4))
    painter.drawPath(path)


def _draw_wavelet(painter, color):
    """A wave packet: a burst that starts, peaks and dies away.

    Not a spectrum glyph and not a trace glyph, because the reading is
    neither — it is what the record contains *and when*. The Morlet's
    own shape says that in one mark: a sinusoid under a Gaussian, so
    the eye reads a burst localized in time rather than a tone running
    the width of the button.
    """
    painter.setBrush(Qt.BrushStyle.NoBrush)
    faint = QColor(color)
    faint.setAlpha(110)
    pen = QPen(faint, 3)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    # the envelope, faint, so the packet reads as windowed rather than
    # as a wave that happens to fade
    envelope = QPainterPath()
    envelope.moveTo(6, 32)
    envelope.cubicTo(20, 32, 22, 8, 32, 8)
    envelope.cubicTo(42, 8, 44, 32, 58, 32)
    painter.drawPath(envelope)

    painter.setPen(QPen(QColor(color), 4))
    packet = QPainterPath()
    packet.moveTo(6, 32)
    import math

    for step in range(1, 53):
        x = 6 + step
        t = (x - 32) / 11.0
        y = 32 - 24 * math.exp(-0.5 * t * t) * math.cos(2.6 * t)
        packet.lineTo(x, y)
    painter.drawPath(packet)


def _draw_percent_bars(painter, color):
    """The same chart read the other way: how much of each channel's
    band fell outside, which is a share and so has a floor at zero."""
    painter.setPen(Qt.PenStyle.NoPen)
    faint = QColor(color)
    faint.setAlpha(110)
    painter.setBrush(QBrush(faint))
    painter.drawRect(QRectF(10, 20, 44, 30))
    for x, height, hot in ((13, 10, False), (24, 26, True),
                           (35, 8, False), (46, 20, False)):
        painter.setBrush(QBrush(color if hot else faint))
        painter.drawRect(QRectF(x, 50 - height, 7, height))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    pen = QPen(QColor(color), 3)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawLine(QPointF(8, 32), QPointF(58, 32))


def _draw_report(painter, color):
    """A page with a chart on it: what the button makes.

    A calculator sat here before (the tree's calculator column, gone
    2026-09-04), which said "this computes something" — true of every
    row it sat on, and no help at all in saying what this one produces.
    """
    page = QColor(color)
    page.setAlpha(60)
    painter.setPen(_pen(color, 4))
    painter.setBrush(QBrush(page))
    painter.drawRoundedRect(QRectF(14, 8, 36, 48), 4, 4)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # a line of text, then the figure under it
    painter.setPen(_pen(color, 3))
    for y in (18, 25):
        painter.drawLine(QPointF(21, y), QPointF(43, y))
    painter.setPen(_pen(color, 4))
    painter.drawPath(_curve([(21, 46), (27, 38), (33, 42), (39, 32),
                             (43, 35)]))


def _draw_drive_point(painter, color):
    """A matrix with its diagonal filled: the drive-point records.

    A drive-point FRF is the diagonal of the response x reference grid —
    excitation and response at one DOF — and that is exactly the set this
    button selects, so the icon shows the grid it acts on.
    """
    faint = QColor(color)
    faint.setAlpha(70)
    for row in range(3):
        for column in range(3):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(color) if row == column else faint))
            painter.drawRect(QRectF(12 + column * 14, 12 + row * 14, 12, 12))


def _draw_residual(painter, color):
    """What is left when the confirmed modes are taken out: the
    explained peak faint, the remainder solid — the curve the toggle
    puts under the fit."""
    faint = QColor(color)
    faint.setAlpha(70)
    painter.setPen(QPen(faint, 5))
    path = QPainterPath(QPointF(8, 54))
    path.cubicTo(QPointF(22, 54), QPointF(24, 10), QPointF(32, 10))
    path.cubicTo(QPointF(40, 10), QPointF(42, 54), QPointF(56, 54))
    painter.drawPath(path)
    painter.setPen(QPen(QColor(color), 5))
    leftover = QPainterPath(QPointF(8, 50))
    leftover.cubicTo(QPointF(18, 50), QPointF(20, 38), QPointF(26, 38))
    leftover.cubicTo(QPointF(32, 38), QPointF(32, 50), QPointF(40, 50))
    leftover.cubicTo(QPointF(48, 50), QPointF(50, 44), QPointF(56, 44))
    painter.drawPath(leftover)


def _draw_strain(painter, color):
    """A bar stretched by arrows at both ends — dL over L.

    A serpentine gauge foil is what a strain gauge looks like and is
    unreadable at 16 px; what strain *is* survives the size.
    """
    painter.setPen(_pen(color, 7))
    painter.drawLine(QPointF(22, 32), QPointF(42, 32))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(6, 32), QPointF(20, 24), QPointF(20, 40)])
    painter.drawPolygon([QPointF(58, 32), QPointF(44, 24), QPointF(44, 40)])
    ends = QColor(color)
    ends.setAlpha(150)
    painter.setPen(QPen(ends, 5))
    painter.drawLine(QPointF(22, 18), QPointF(22, 46))
    painter.drawLine(QPointF(42, 18), QPointF(42, 46))


def _draw_pressure(painter, color):
    """A dial with a needle: the one round silhouette in the set, so it
    tells apart from every trace and arrow at a glance."""
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPointF(32, 32), 22, 22)
    painter.setPen(_pen(color, 6))
    painter.drawLine(QPointF(32, 32), QPointF(46, 20))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawEllipse(QPointF(32, 32), 5, 5)
    ticks = QColor(color)
    ticks.setAlpha(150)
    painter.setPen(QPen(ticks, 4))
    for x, y in ((14, 22), (32, 8), (50, 22)):
        painter.drawPoint(QPointF(x, y))


def _draw_averaging(painter, color):
    """Two overlapping frames on a trace, with a window over each.

    What the button puts on the plot, so the icon is the thing itself:
    the shaded bands a PSD is averaged over, offset by half a frame
    because that is the usual overlap, and the bell each is multiplied
    by drawn over them.
    """
    faint = QColor(color)
    faint.setAlpha(60)
    painter.setPen(Qt.PenStyle.NoPen)
    for left in (8, 28):
        painter.setBrush(QBrush(faint))
        painter.drawRect(QRectF(left, 10, 28, 44))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(_pen(color, 5))
    for left in (8, 28):
        painter.drawPath(_curve([
            (left + i, 50 - 30 * (0.5 - 0.5 * math.cos(i / 28 * 2 * math.pi)))
            for i in range(29)]))


def _draw_cmif(painter, color):
    """Stacked singular-value traces peaking together: the CMIF."""
    def peaks(base: float, amplitude: float,
              phase: float) -> list[tuple[float, float]]:
        return [(10 + i, base - amplitude
                 * abs(math.sin(i / 44 * 2 * math.pi + phase)))
                for i in range(45)]
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve(peaks(26, 14, 0.0)))
    faint = QColor(color)
    faint.setAlpha(120)
    painter.setPen(QPen(faint, 6))
    painter.drawPath(_curve(peaks(48, 9, 0.35)))


def _draw_waterfall(painter, color):
    """Three peaked traces receding along a depth axis: the 3-D reading.

    Drawn back to front, faintest first, so the near trace overlaps the
    far ones the way the scene does.
    """
    for step, alpha in ((2, 80), (1, 150), (0, 255)):
        trace = QColor(color)
        trace.setAlpha(alpha)
        painter.setPen(QPen(trace, 5))
        painter.drawPath(_curve([
            (8 + 7 * step + i,
             46 - 9 * step - 22 * math.exp(-((i - 20) / 7) ** 2))
            for i in range(41)]))


def _draw_curves(painter, color):
    """Two traces overlaid: coherence plotted a channel at a time."""
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (10 + i, 26 - 12 * math.sin(i / 44 * 2 * math.pi)) for i in range(45)]))
    faint = QColor(color)
    faint.setAlpha(120)
    painter.setPen(QPen(faint, 6))
    painter.drawPath(_curve([
        (10 + i, 42 - 10 * math.sin(i / 44 * 2 * math.pi + 1.2))
        for i in range(45)]))


def _draw_map(painter, color):
    """Banded rows: every channel at once, coherence as color."""
    painter.setPen(Qt.PenStyle.NoPen)
    for row, alphas in enumerate((
            (255, 170, 90, 200), (120, 255, 200, 60), (200, 70, 255, 150))):
        for column, alpha in enumerate(alphas):
            band = QColor(color)
            band.setAlpha(alpha)
            painter.setBrush(QBrush(band))
            painter.drawRect(QRectF(11 + column * 11, 16 + row * 11, 11, 11))


def _draw_channel_table(painter, color):
    """A table with a header row."""
    header = QColor(color)
    header.setAlpha(110)
    painter.fillRect(QRectF(11, 15, 42, 10), QBrush(header))
    painter.setPen(_pen(color, 4))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(11, 15, 42, 34), 3, 3)
    for y in (25, 37):
        painter.drawLine(QPointF(11, y), QPointF(53, y))
    painter.drawLine(QPointF(32, 15), QPointF(32, 49))


def _draw_shape_set(painter, color):
    """A deformed mode shape with nodes riding on it."""
    points = [(10 + i, 32 - 16 * math.sin(i / 44 * 2 * math.pi))
              for i in range(45)]
    painter.setPen(_pen(color, 5))
    painter.drawPath(_curve(points))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    for index in (0, 11, 22, 33, 44):
        x, y = points[index]
        painter.drawEllipse(QPointF(x, y), 5, 5)


def _draw_mode(painter, color):
    """One mode out of a set: a single deformed curve."""
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (10 + i, 32 - 14 * math.sin(i / 44 * math.pi)) for i in range(45)]))


def _draw_project(painter, color):
    """A dense mesh sampled down onto sparse points: the projection."""
    faded = QColor(color)
    faded.setAlpha(110)
    painter.setPen(_pen(faded, 4))
    # the dense model: a small hatched patch, upper left
    for i in range(3):
        painter.drawLine(10, 12 + i * 8, 30, 12 + i * 8)
        painter.drawLine(10 + i * 10, 10, 10 + i * 10, 30)
    # the arrow down onto the sparse test points
    painter.setPen(_pen(color, 5))
    painter.drawLine(32, 32, 44, 44)
    painter.drawLine(44, 44, 44, 34)
    painter.drawLine(44, 44, 34, 44)
    painter.setBrush(QBrush(QColor(color)))
    for x, y in ((14, 50), (32, 54), (50, 50)):
        painter.drawEllipse(x - 4, y - 4, 8, 8)


def _draw_overlay(painter, color):
    """Two mode curves over each other: the comparison animation."""
    faded = QColor(color)
    faded.setAlpha(120)
    painter.setPen(_pen(faded, 6))
    painter.drawPath(_curve([
        (10 + i, 36 - 10 * math.sin(i / 44 * math.pi)) for i in range(45)]))
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (10 + i, 32 - 16 * math.sin(i / 44 * math.pi)) for i in range(45)]))


def _draw_ratio(painter, color):
    """One curve over another with a fraction bar: the dB ratio."""
    faded = QColor(color)
    faded.setAlpha(120)
    painter.setPen(_pen(color, 5))
    painter.drawPath(_curve([
        (12 + i, 18 - 6 * math.sin(i / 40 * math.pi)) for i in range(41)]))
    painter.setPen(_pen(color, 5))
    painter.drawLine(10, 32, 54, 32)
    painter.setPen(_pen(faded, 5))
    painter.drawPath(_curve([
        (12 + i, 50 - 6 * math.sin(i / 40 * math.pi)) for i in range(41)]))


def _draw_refresh(painter, color):
    """A circular arrow: this object no longer matches its source's
    settings, click to recompute it."""
    from PySide6.QtCore import QRectF

    pen = _pen(color, 6)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    rect = QRectF(14, 14, 36, 36)
    painter.drawArc(rect, 40 * 16, 260 * 16)
    # the arrowhead where the arc ends (at ~40 degrees)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    head = QPainterPath()
    head.moveTo(52, 18)
    head.lineTo(44, 34)
    head.lineTo(56, 33)
    head.closeSubpath()
    painter.drawPath(head)


def _draw_matched_modes(painter, color):
    """A little MAC grid with two agreed cells picked out."""
    faded = QColor(color)
    faded.setAlpha(70)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(faded))
    for i in range(3):
        for j in range(3):
            painter.drawRect(12 + j * 14, 12 + i * 14, 11, 11)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawRect(12, 12, 11, 11)
    painter.drawRect(40, 40, 11, 11)


def _draw_test(painter, color):
    """Stacked layers: a test holding geometry, data and metadata."""
    fill = QColor(color)
    fill.setAlpha(90)
    for i, y in enumerate((40, 28, 16)):
        painter.setBrush(QBrush(fill if i else QColor(color)))
        painter.setPen(_pen(color, 4))
        painter.drawRoundedRect(QRectF(12, y, 40, 14), 3, 3)


def _draw_play(painter, color):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(20, 14), QPointF(20, 50), QPointF(50, 32)])


def _draw_up(painter, color):
    """Play, turned to point up: the report bar's Move Up."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(14, 44), QPointF(50, 44), QPointF(32, 16)])


def _draw_down(painter, color):
    """And down: Move Down."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(14, 20), QPointF(50, 20), QPointF(32, 48)])


def _draw_faster(painter, color):
    """Two triangles pointing the way play does: play, but more so."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    for shift in (0, 18):
        painter.drawPolygon([QPointF(14 + shift, 16), QPointF(14 + shift, 48),
                             QPointF(32 + shift, 32)])


def _draw_slower(painter, color):
    """The same pair, mirrored. Not a rewind — nothing plays backwards
    here — but the shape everyone reads as 'less of that'."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    for shift in (0, 18):
        painter.drawPolygon([QPointF(50 - shift, 16), QPointF(50 - shift, 48),
                             QPointF(32 - shift, 32)])


def _draw_pause(painter, color):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawRoundedRect(QRectF(19, 14, 9, 36), 2, 2)
    painter.drawRoundedRect(QRectF(36, 14, 9, 36), 2, 2)


def _draw_add(painter, color):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawRoundedRect(QRectF(28, 14, 8, 36), 2, 2)
    painter.drawRoundedRect(QRectF(14, 28, 36, 8), 2, 2)


def _element_shape(painter, color, points):
    """An element outline with a dot on every node it takes.

    Sized so the silhouette — line, triangle, square — still reads at the
    18 px the toolbar draws it at, where the corner dots merge into the
    outline and only the overall shape survives.
    """
    painter.setPen(_pen(color, 7))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    path = _curve(points)
    if len(points) > 2:
        path.closeSubpath()
    painter.drawPath(path)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    for x, y in points:
        painter.drawEllipse(QPointF(x, y), 7, 7)


def _draw_beam(painter, color):
    _element_shape(painter, color, [(12, 50), (52, 14)])


def _draw_tri(painter, color):
    _element_shape(painter, color, [(32, 10), (54, 52), (10, 52)])


def _draw_quad(painter, color):
    _element_shape(painter, color, [(11, 11), (53, 11), (53, 53), (11, 53)])


def _draw_edit(painter, color):
    """A pencil. Filled rather than outlined: at 18 px an outline of this
    shape thins out to an ambiguous diagonal stroke."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(11, 53), QPointF(15, 38), QPointF(40, 13),
                         QPointF(51, 24), QPointF(26, 49)])
    painter.setPen(_pen(color, 3))
    painter.drawLine(QPointF(15, 38), QPointF(26, 49))


def _draw_trash(painter, color):
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLine(QPointF(12, 20), QPointF(52, 20))     # lid
    painter.drawLine(QPointF(26, 20), QPointF(26, 13))     # handle
    painter.drawLine(QPointF(38, 20), QPointF(38, 13))
    painter.drawLine(QPointF(26, 13), QPointF(38, 13))
    path = QPainterPath(QPointF(17, 25))                   # body
    path.lineTo(QPointF(20, 52))
    path.lineTo(QPointF(44, 52))
    path.lineTo(QPointF(47, 25))
    painter.drawPath(path)


def _draw_rotate(painter, color):
    """A ring with an arrowhead: turning a coordinate system."""
    painter.setPen(_pen(color, 6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawArc(QRectF(14, 14, 36, 36), 30 * 16, 280 * 16)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(44, 6), QPointF(54, 22), QPointF(36, 20)])


def _draw_integrate(painter, color):
    """An integral sign: one integration of the record."""
    painter.setPen(_pen(color, 6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_curve([(38, 10), (34, 8), (30, 12), (30, 52),
                             (26, 56), (22, 52)]))
    painter.setPen(_pen(color, 4))
    painter.drawLine(QPointF(38, 30), QPointF(48, 30))
    painter.drawLine(QPointF(38, 40), QPointF(48, 40))


def _draw_differentiate(painter, color):
    """A tangent on a curve: the record's rate of change."""
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_curve([(8 + i, 50 - 30 * (i / 48) ** 2)
                             for i in range(49)]))
    faded = QColor(color)
    faded.setAlpha(150)
    painter.setPen(QPen(faded, 5))
    painter.drawLine(QPointF(22, 54), QPointF(56, 22))


def _draw_transform(painter, color):
    """Many traces becoming few through an arrow: a record through
    a shape set to modal responses."""
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for y in (16, 32, 48):
        painter.drawPath(_curve([(6, y), (12, y - 6), (18, y + 6), (24, y)]))
    painter.drawLine(QPointF(30, 32), QPointF(42, 32))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(40, 24), QPointF(50, 32), QPointF(40, 40)])
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_curve([(50, 32), (53, 22), (57, 42), (60, 32)]))


def _draw_expand(painter, color):
    """The transform read the other way: few traces becoming many."""
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_curve([(4, 32), (7, 22), (11, 42), (14, 32)]))
    painter.drawLine(QPointF(22, 32), QPointF(34, 32))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(32, 24), QPointF(42, 32), QPointF(32, 40)])
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for y in (16, 32, 48):
        painter.drawPath(_curve([(42, y), (48, y - 6), (54, y + 6), (60, y)]))


def _draw_merge(painter, color):
    """Two lines joining into one: objects merged into one."""
    painter.setPen(_pen(color, 6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_curve([(8, 14), (26, 14), (36, 32), (56, 32)]))
    painter.drawPath(_curve([(8, 50), (26, 50), (36, 32)]))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(50, 24), QPointF(60, 32), QPointF(50, 40)])


def _draw_sine(painter, color):
    """A sweep: a sine whose frequency rises across the icon."""
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_curve([
        (6 + i, 32 - 16 * math.sin(2 * math.pi * (i / 52) ** 2 * 3.0))
        for i in range(53)]))


def _draw_rigid(painter, color):
    """A body turning about a marked point: the rigid-body modes."""
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(19, 25, 26, 16), 4, 4)
    painter.drawArc(QRectF(9, 9, 46, 46), 40 * 16, 230 * 16)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawEllipse(QPointF(32, 33), 4, 4)
    painter.drawPolygon([QPointF(18, 56), QPointF(32, 48),
                         QPointF(30, 62)])


def _draw_reset(painter, color):
    """An arrow curving back on itself: undo the turn."""
    painter.setPen(_pen(color, 6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawArc(QRectF(14, 14, 36, 36), 150 * 16, -280 * 16)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(20, 6), QPointF(28, 22), QPointF(10, 20)])


def _draw_colormap(painter, color):
    """A viridis strip: the control colors the model by displacement.

    The map itself, not a copy of five of its stops — the swatch and the
    thing it stands for have to be the same colors."""
    painter.setPen(Qt.PenStyle.NoPen)
    steps = 22
    width = 44 / steps
    for i in range(steps):
        painter.setBrush(QBrush(viridis(i / (steps - 1))))
        painter.drawRect(QRectF(10 + i * width, 20, width + 0.5, 24))
    painter.setPen(_pen(color, 3))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(10, 20, 44, 24))


def _draw_default(painter, color):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawEllipse(QPointF(32, 32), 16, 16)


def _draw_link(painter, color):
    """Two chain links joined."""
    painter.setPen(_pen(color, 6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(8, 26, 26, 14), 7, 7)
    painter.drawRoundedRect(QRectF(30, 26, 26, 14), 7, 7)


def _draw_unlink(painter, color):
    """Two chain links parted."""
    painter.setPen(_pen(color, 6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(4, 26, 22, 14), 7, 7)
    painter.drawRoundedRect(QRectF(38, 26, 22, 14), 7, 7)
    painter.drawLine(QPointF(30, 24), QPointF(34, 42))


def _draw_dofs(painter, color):
    """An arrow leaving a node: the DOF marker."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawEllipse(QPointF(16, 48), 7, 7)
    painter.setPen(_pen(color, 6))
    painter.drawLine(QPointF(20, 44), QPointF(44, 20))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawPolygon([QPointF(52, 12), QPointF(48, 32),
                         QPointF(32, 16)])


def _draw_photos(painter, color):
    """A framed photo: sun over mountains."""
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(8, 12, 48, 40), 6, 6)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawEllipse(QPointF(21, 25), 5.5, 5.5)
    painter.drawPolygon([QPointF(13, 47), QPointF(28, 30),
                         QPointF(37, 40), QPointF(45, 28),
                         QPointF(51, 47)])


_DRAW = {
    'Geometry': _draw_geometry,
    'TimeHistory': _draw_time_history,
    'Frf': _draw_frf,
    'Psd': _draw_psd,
    'OctavePsd': _draw_octave_psd,
    'Spectrum': _draw_spectrum,
    'Specification': _draw_specification,
    'OctaveSpecification': _draw_octave_psd,
    'Srs': _draw_srs,
    'ShockSpecification': _draw_shock_specification,
    'SineSweepSpecification': _draw_sine_sweep_specification,
    'SineLevel': _draw_sine_sweep_specification,
    'SineLevelSet': _draw_sine_sweep_specification,
    'SineTarget': _draw_sine_sweep_specification,
    'TransientSpecification': _draw_transient_specification,
    'Coherence': _draw_coherence,
    'MultipleCoherence': _draw_multiple_coherence,
    'ChannelTable': _draw_channel_table,
    # the same page-with-a-chart the button beside the project root
    # wears — what that button makes is exactly this object, so the
    # glyph the user pressed is the glyph they get back. It also stops
    # a report reading as a channel table, which it did while the two
    # shared a grid and were told apart by color alone.
    'Report': _draw_report,
    'Photos': _draw_photos,
    'ShapeSet': _draw_shape_set,
    'MatchedModes': _draw_matched_modes,
    'Test': _draw_test,
}


def _draw_nodes(painter, color):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    for x, y in [(18, 22), (44, 18), (32, 46)]:
        painter.drawEllipse(QPointF(x, y), 7, 7)


def _draw_coordinate_systems(painter, color):
    """An axis triad."""
    origin = QPointF(20, 46)
    for dx, dy in [(26, 0), (0, -26), (16, -14)]:
        painter.setPen(_pen(color, 5))
        painter.drawLine(origin, QPointF(origin.x() + dx, origin.y() + dy))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawEllipse(origin, 5, 5)


def _draw_tracelines(painter, color):
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([(11, 44), (24, 20), (38, 44), (53, 20)]))


def _draw_elements(painter, color):
    painter.setPen(_pen(color, 5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(13, 16, 38, 32))
    painter.drawLine(QPointF(32, 16), QPointF(32, 48))
    painter.drawLine(QPointF(13, 32), QPointF(51, 32))


def _draw_blocks(painter, color):
    """Two groups of cells, one filled: elements sorted into blocks.

    The elements icon is one mesh divided into cells; this is that mesh
    divided into *parts*, which is what a block is — so the difference
    between them has to be the grouping, not the cells.
    """
    painter.setPen(_pen(color, 5))
    painter.setBrush(QBrush(QColor(color)))
    painter.drawRect(QRectF(11, 18, 18, 28))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(35, 18, 18, 28))
    painter.drawLine(QPointF(35, 32), QPointF(53, 32))


def _draw_bounds(painter, color):
    """A boxed plot frame with ticks: the axes drawn around the geometry."""
    painter.setPen(_pen(color, 4))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(14, 14, 36, 36))
    for offset in (23, 32, 41):
        painter.drawLine(QPointF(offset, 50), QPointF(offset, 44))
        painter.drawLine(QPointF(14, offset), QPointF(20, offset))


def _draw_record(painter, color):
    """A single trace: one channel out of many."""
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (10 + i, 32 - 14 * math.sin(i / 44 * 2 * math.pi)) for i in range(45)]))


def _trace(painter, color, top=32, amplitude=12):
    """The sine wave a record has always been drawn as."""
    painter.setPen(_pen(color, 6))
    painter.drawPath(_curve([
        (10 + i, top - amplitude * math.sin(i / 44 * 2 * math.pi))
        for i in range(45)]))


def _dots(painter, color, count):
    """Newton's notation, over the trace: one dot a derivative.

    Generously sized. At the first attempt they were 4.5 units across, which
    is a single pixel once the icon is drawn at 16 — the distinction between
    a velocity and an acceleration disappeared at exactly the size the tree
    uses it.
    """
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    spacing = 19
    for i in range(count):
        x = 32 + (i - (count - 1) / 2) * spacing
        painter.drawEllipse(QPointF(x, 13), 7, 7)


def _draw_displacement(painter, color):
    """x: the trace itself, undifferentiated."""
    _trace(painter, color, top=34, amplitude=15)


def _draw_velocity(painter, color):
    """x with one dot."""
    _trace(painter, color, top=42, amplitude=11)
    _dots(painter, color, 1)


def _draw_acceleration(painter, color):
    """x with two dots."""
    _trace(painter, color, top=42, amplitude=11)
    _dots(painter, color, 2)


def _arc_arrow(painter, color, center, radius, start, sweep, width=6):
    """A ring segment with an arrowhead at its end, turning
    clockwise on screen from `start` degrees through `sweep`."""
    cx, cy = center
    painter.setPen(_pen(color, width))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawArc(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius),
                    int(start * 16), int(-sweep * 16))
    # the arrowhead sits on the tangent at the arc's end — Qt's angles
    # run counter-clockwise on a y-down screen, so the point at angle
    # `a` is (cx + r cos a, cy − r sin a) and clockwise travel heads
    # along (sin a, cos a)
    end = math.radians(start - sweep)
    px, py = cx + radius * math.cos(end), cy - radius * math.sin(end)
    tx, ty = math.sin(end), math.cos(end)
    nx, ny = ty, -tx
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(px + 10 * tx, py + 10 * ty),
                         QPointF(px - 3 * tx + 7 * nx, py - 3 * ty + 7 * ny),
                         QPointF(px - 3 * tx - 7 * nx, py - 3 * ty - 7 * ny)])


def _draw_angle(painter, color):
    """θ: a turn, undifferentiated — the rotation's own trace."""
    _arc_arrow(painter, color, (32, 34), 17, 200, 290)


def _draw_angular_velocity(painter, color):
    """θ with one dot."""
    _arc_arrow(painter, color, (32, 40), 14, 200, 290)
    _dots(painter, color, 1)


def _draw_angular_acceleration(painter, color):
    """θ with two dots."""
    _arc_arrow(painter, color, (32, 40), 14, 200, 290)
    _dots(painter, color, 2)


def _draw_moment(painter, color):
    """A twist about an axis: the torque a force at an offset makes."""
    shaft = QColor(color)
    shaft.setAlpha(150)
    painter.setPen(QPen(shaft, 6))
    painter.drawLine(QPointF(32, 6), QPointF(32, 58))
    _arc_arrow(painter, color, (32, 32), 17, 150, 250, width=7)


def _modal(draw):
    """The linear glyph at three quarters, with an M in the corner: a
    modal coordinate of that quantity."""
    def drawer(painter, color):
        painter.save()
        painter.scale(0.75, 0.75)
        draw(painter, color)
        painter.restore()
        font = QFont()
        font.setBold(True)
        font.setPixelSize(26)
        painter.setFont(font)
        painter.setPen(QPen(QColor(color)))
        painter.drawText(QRectF(38, 36, 26, 28), Qt.AlignmentFlag.AlignCenter,
                         'M')
    drawer.__name__ = f'_draw_modal_{draw.__name__.removeprefix("_draw_")}'
    return drawer


def _draw_force(painter, color):
    """An arrow pressing into a fixed surface."""
    painter.setPen(_pen(color, 7))
    painter.drawLine(QPointF(32, 8), QPointF(32, 38))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(32, 48), QPointF(22, 32), QPointF(42, 32)])
    ground = QColor(color)
    ground.setAlpha(150)
    painter.setPen(QPen(ground, 6))
    painter.drawLine(QPointF(12, 55), QPointF(52, 55))


def _draw_temperature(painter, color):
    """A thermometer: bulb and stem, with its scale beside it."""
    painter.setPen(_pen(color, 6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLine(QPointF(26, 14), QPointF(26, 40))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawEllipse(QPointF(26, 47), 10, 10)
    ticks = QColor(color)
    ticks.setAlpha(150)
    painter.setPen(QPen(ticks, 5))
    for y in (18, 27, 36):
        painter.drawLine(QPointF(38, y), QPointF(50, y))


def _draw_voltage(painter, color):
    """A bolt."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color)))
    painter.drawPolygon([QPointF(p[0], p[1]) for p in (
        (38, 8), (18, 34), (29, 34), (24, 56), (46, 28), (34, 28), (40, 8))])


_QUANTITY_DRAW = {
    'length': _draw_displacement,        # the dimension; 'displacement'
                                         # is the word for it
    'strain': _draw_strain,
    'pressure': _draw_pressure,
    'velocity': _draw_velocity,
    'acceleration': _draw_acceleration,
    'force': _draw_force,
    'temperature': _draw_temperature,
    'voltage': _draw_voltage,
    'angle': _draw_angle,
    'angular_velocity': _draw_angular_velocity,
    'angular_acceleration': _draw_angular_acceleration,
    'moment': _draw_moment,
    'modal_length': _modal(_draw_displacement),
    'modal_velocity': _modal(_draw_velocity),
    'modal_acceleration': _modal(_draw_acceleration),
    'modal_force': _modal(_draw_force),
}


def _draw_channel(painter, color):
    """One row of a channel table."""
    fill = QColor(color)
    fill.setAlpha(120)
    painter.fillRect(QRectF(11, 26, 42, 12), QBrush(fill))
    painter.setPen(_pen(color, 4))
    painter.drawLine(QPointF(11, 18), QPointF(53, 18))
    painter.drawLine(QPointF(11, 46), QPointF(53, 46))


CONTROL_COLOR = '#8b949e'


@cache
@cache
def color_swatch(rgb: tuple[float, float, float]) -> QIcon:
    """A filled chip for a palette color, for drop-downs and menus."""
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(_pen('#00000066', 3))
    painter.setBrush(QBrush(QColor.fromRgbF(*rgb)))
    painter.drawRoundedRect(QRectF(9, 9, 46, 46), 8, 8)
    painter.end()
    return QIcon(pixmap)


def control_icon(name: str) -> QIcon:
    """Toolbar glyphs: play, pause, add, and the element types."""
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    {'play': _draw_play, 'pause': _draw_pause,
     'faster': _draw_faster, 'slower': _draw_slower,
     'up': _draw_up, 'down': _draw_down,
     'add': _draw_add,
     'beam': _draw_beam, 'tri': _draw_tri, 'quad': _draw_quad,
     'edit': _draw_edit, 'trash': _draw_trash,
     'colormap': _draw_colormap,
     'curves': _draw_curves, 'map': _draw_map,
     'waterfall': _draw_waterfall,
     'drive_point': _draw_drive_point, 'cmif': _draw_cmif,
     'averaging': _draw_averaging, 'shocks': _draw_shocks,
     'residual': _draw_residual,
     'report': _draw_report,
     'bars': _draw_bars, 'percent_bars': _draw_percent_bars,
     'kurtosis': _draw_kurtosis, 'filter': _draw_filter,
     'truncate': _draw_truncate, 'octave': _draw_octave,
     'log_axis': _draw_log_axis,
     'wavelet': _draw_wavelet,
     'dofs': _draw_dofs, 'link': _draw_link, 'unlink': _draw_unlink,
     'table': _draw_channel_table, 'overlay': _draw_overlay,
     'ratio': _draw_ratio,
     'project': _draw_project,
     'rotate': _draw_rotate,
     'reset': _draw_reset,
     'rigid': _draw_rigid,
     'integrate': _draw_integrate, 'differentiate': _draw_differentiate,
     'transform': _draw_transform, 'expand': _draw_expand,
     'merge': _draw_merge, 'sine': _draw_sine,
     'refresh': _draw_refresh}.get(name, _draw_default)(painter, CONTROL_COLOR)
    painter.end()
    return QIcon(pixmap)


_CHILD_DRAW = {
    'nodes': _draw_nodes,
    'coordinate_systems': _draw_coordinate_systems,
    'tracelines': _draw_tracelines,
    'elements': _draw_elements,
    'blocks': _draw_blocks,
    'bounds': _draw_bounds,
    'mode': _draw_mode,
    'record': _draw_record,
    'channel': _draw_channel,
}


BADGE_COLOR = '#e6a23c'  # amber: units not yet defined


def _draw_units_badge(painter):
    """A small '?' badge marking an object whose units are undefined."""
    center = QPointF(SIZE - 15, SIZE - 15)
    painter.setPen(QPen(QColor('#ffffff'), 3))
    painter.setBrush(QBrush(QColor(BADGE_COLOR)))
    painter.drawEllipse(center, 14, 14)
    font = painter.font()
    font.setPixelSize(22)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QPen(QColor('#ffffff')))
    painter.drawText(QRectF(center.x() - 14, center.y() - 14, 28, 28),
                     Qt.AlignmentFlag.AlignCenter, '?')


EMPTY_COLOR = '#6a6a72'      # a category with nothing in it


@cache
def child_icon(kind: str, parent_type: str = 'Geometry',
               units_defined: bool = True, empty: bool = False) -> QIcon:
    """Icon for a sub-item, drawn in its parent object's color.

    Carries the same undefined-units badge as a top-level icon, so a single
    channel with no units declared is visible without expanding anything.
    """
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw = _CHILD_DRAW.get(kind, _draw_default)
    draw(painter, EMPTY_COLOR if empty else COLORS.get(parent_type, '#7f7f7f'))
    if not units_defined:
        _draw_units_badge(painter)
    painter.end()
    return QIcon(pixmap)


def quantity_of(dimension: str) -> str | None:
    """The quantity a *simple* dimension names, if it is one we draw.

    Compounds return None; `record_icon` splits those into their channel
    factors itself, and the other callers — a channel table's unit, a
    DOF-arrow quantity — only ever hold simple dimensions.
    """
    dimension = (dimension or '').strip()
    return dimension if dimension in _QUANTITY_DRAW else None


@cache
def quantity_icon(quantity: str, units_defined: bool = True) -> QIcon:
    """Icon for what a record measures, rather than for what holds it."""
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    _QUANTITY_DRAW[quantity](painter, QUANTITY_COLORS[quantity])
    if not units_defined:
        _draw_units_badge(painter)
    painter.end()
    return QIcon(pixmap)


def record_icon(data: DataArray, index: int) -> QIcon:
    """The icon for one record: what that record measures.

    A density or a square resolves to its base quantity — a PSD record
    is accelerations, squared per hertz — and a ratio or a cross term
    draws *both* quantities as a fraction, response over reference: an
    FRF record wears acceleration-over-force, a CPSD cross term the
    same. Identical factors collapse to the single glyph, and a record
    whose quantities we cannot draw — a coherence, an undeclared
    channel — keeps its object's type icon. Undefined units keep their
    badge either way, and a `dimension_hint` is enough to choose: a
    source that said "force" without sizing it has still said what the
    record is.
    """
    from ..core.data import channel_quantities

    defined = data.ordinate_dim[index] != 'unknown'
    response, reference = channel_quantities(data.known_dim(index))
    if (response in _QUANTITY_DRAW and reference in _QUANTITY_DRAW
            and response != reference):
        return ratio_icon(response, reference, defined)
    if response in _QUANTITY_DRAW:
        return quantity_icon(response, defined)
    return type_icon(type(data).__name__, defined)


@cache
def ratio_icon(response: str, reference: str,
               units_defined: bool = True) -> QIcon:
    """Two quantities as a fraction, the way the dimension reads:
    response upper-left, reference lower-right, a slash between."""
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    scale = 0.58
    painter.save()
    painter.scale(scale, scale)
    _QUANTITY_DRAW[response](painter, QUANTITY_COLORS[response])
    painter.restore()
    painter.save()
    painter.translate(SIZE * (1 - scale), SIZE * (1 - scale))
    painter.scale(scale, scale)
    _QUANTITY_DRAW[reference](painter, QUANTITY_COLORS[reference])
    painter.restore()
    painter.setPen(QPen(QColor('#8e8e93'), 3.0))
    painter.drawLine(QPointF(SIZE * 0.30, SIZE * 0.92),
                     QPointF(SIZE * 0.70, SIZE * 0.08))
    if not units_defined:
        _draw_units_badge(painter)
    painter.end()
    return QIcon(pixmap)


@cache
def type_icon(type_name: str, units_defined: bool = True) -> QIcon:
    """Icon for a visualdynamics type, by class name. Requires a running QApplication."""
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    _DRAW.get(type_name, _draw_default)(painter, COLORS.get(type_name, '#7f7f7f'))
    if not units_defined:
        _draw_units_badge(painter)
    painter.end()
    return QIcon(pixmap)


def object_icon(obj: Any, units_defined: bool = True) -> QIcon:
    """The icon for an object, which is not always its class's.

    A PSD integrated onto proportional bands is still a `Psd` — same
    units, same dimension, the same area under it — and deliberately so:
    everything that reads a spectrum works on either without asking. But
    the two do not *look* alike and must not read alike in the tree, so
    the icon asks the object a question its class name cannot answer.

    The same rule as `record_icon`: what a thing measures, not what
    holds it.
    """
    from ..core.data import Psd, Specification

    name = type(obj).__name__
    # `type(obj) is Psd`, not isinstance: Specification subclasses Psd
    # and has an icon of its own to keep
    if type(obj) is Psd and getattr(obj, 'bandwidth', None) is not None:
        name = 'OctavePsd'
    # and the requirement on bands, the same way (2026-09-18)
    if isinstance(obj, Specification) \
            and getattr(obj, 'bandwidth', None) is not None:
        name = 'OctaveSpecification'
    return type_icon(name, units_defined)


@cache
def placeholder_icon(type_name: str) -> QIcon:
    """The type's glyph in gray: expected by the project's type, not
    yet in the project."""
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    _DRAW.get(type_name, _draw_default)(painter, EMPTY_COLOR)
    painter.end()
    return QIcon(pixmap)


# The application's own icon is rendered rather than painted — a plate
# carrying a mode, through a small z-buffer — so it lives next door.
# Re-exported because this is where icons are looked for.
from ..theme import VIRIDIS  # noqa: F401  (the one color scale)
from .app_icon import (  # noqa: F401
    app_icon,
    colormap,
    draw_app_icon,
)


def viridis(t: float) -> QColor:
    """The color at `t` in 0..1 on viridis, clamped."""
    return QColor(*[round(v) for v in colormap(float(t))])
