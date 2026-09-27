"""A phase axis shows its ±180° labels.

Pinned at ±185°, the ±180° tick labels sat a few pixels from the axis
ends, and pyqtgraph, which draws no label that does not fit, dropped
them: the axis read -90, 0 and 90 alone (Brandon, 2026-09-27, through
the band-average paper's cross-term figure). The axis runs ±200° with
ticks at every quarter turn, in the app and in the report alike.
"""

from __future__ import annotations

import numpy as np

from visualdynamics.core.data import Psd

_ALIVE = []


def _cross(banded=False):
    f = np.linspace(0.0, 2000.0, 2001)
    turning = np.exp(1j * np.linspace(-np.pi, np.pi, f.size) * 3)
    ones = np.ones_like(f)
    values = Psd(f, np.array([ones, turning, np.conj(turning), ones]),
                 response_dof=['1X+', '1X+', '2X+', '2X+'],
                 reference_dof=['1X+', '2X+', '1X+', '2X+'],
                 ordinate_dim=['acceleration**2/frequency'] * 4)
    return values.to_octave(6, 20, 1800) if banded else values


def _drawn_labels(qt_app, data, size=(420, 220)):
    """The tick labels the phase axis actually draws at this size."""
    import pyqtgraph as pg
    from PySide6.QtGui import QImage, QPainter

    from visualdynamics.plot import build_plots

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    layout.resize(*size)
    build_plots(layout, [('C', data, [1])], component='phase')
    layout.show()
    for _ in range(10):
        qt_app.processEvents()
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    axis = plot.getAxis('left')
    image = QImage(10, 10, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        _axes, _ticks, texts = axis.generateDrawSpecs(painter)
    finally:
        painter.end()
    return [text for _rect, _flags, text in texts]


def test_the_half_turns_are_labeled(qt_app):
    for banded in (False, True):
        labels = _drawn_labels(qt_app, _cross(banded))
        assert labels == ['-180', '-90', '0', '90', '180'], (banded, labels)


def test_the_report_draws_the_same_axis():
    import json
    import re

    from visualdynamics.core.report import Report
    from visualdynamics.plot import PHASE_LIMITS, PHASE_TICKS
    from visualdynamics.report import render_html

    text = render_html(Report('R', [{'kind': 'plot', 'source': 'C',
                                     'component': 'phase'}]),
                       {'C': _cross()})
    data = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        text, re.DOTALL).group(1))
    [block] = data['blocks']
    assert block['yrange'] == list(PHASE_LIMITS)
    assert block['yticks'] == list(PHASE_TICKS)
