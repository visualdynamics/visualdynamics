"""The filtered trace, previewed over the raw one.

A corner frequency is a number; what it *costs* is a shape. The
preview draws each visible curve again through the filter being set,
so the trade — noise gone, content kept — is read on the data it will
be made from, before Apply Filter makes anything.

The preview filters the y-data already on the plot. That is the whole
record in the units on screen, and filtering commutes with a unit
scale, so this is exactly what `core.filters.filtered` will produce —
no second implementation of the rule, just scipy called on what is
drawn. One color for every preview rather than each channel's own:
the question this view answers is "what does the filter do", and a
preview dressed as another channel answers a different one. The
color is `theme`'s `filter_preview`, shared with the stage's twin.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class FilterOverlay:
    """Filtered twins of a plot's curves. Owns every item it adds and
    takes them all away again on `remove`, so a redraw cannot leave a
    preview behind."""

    def __init__(self, plot: Any, filtering: Any, sample_rate: float,
                 colors: Mapping[str, str]) -> None:
        self.plot: Any = plot
        self.sample_rate: float = float(sample_rate)
        self.colors: Mapping[str, str] = colors
        #: the curves being previewed, snapshotted at construction so
        #: the previews themselves are never re-previewed
        self._sources: list[Any] = list(plot.listDataItems())
        #: each source's own pen, to put back when the view closes
        self._pens: list[Any] = [curve.opts.get('pen')
                                 for curve in self._sources]
        self._previews: list[Any] = []
        #: the settings the previews were last drawn with
        self.filtering: Any = filtering
        self.set_filtering(filtering)

    def set_filtering(self, filtering: Any) -> None:
        """Restate the previews for a new corner or order."""
        import pyqtgraph as pg
        from scipy.signal import sosfiltfilt

        from ..core.filters import design
        from . import curve_color

        self.filtering = filtering
        self._take_down()
        try:
            sos = design(filtering, self.sample_rate)
        except ValueError:
            return                    # out of range mid-edit: draw nothing
        # The filtered record takes the ink and the raw one stands
        # back in gray behind it (Brandon, 2026-08-25). The first pass
        # had it the other way round — raw in the channel colors, the
        # twin in one flat color — which asked the reader to judge
        # the filtered record, the thing actually being decided, from
        # the drabber of the two lines. Gray-for-the-reference is what
        # a specification and its response already do here.
        stood_back = pg.mkPen(self.colors['specification_curve'], width=1)
        for index, curve in enumerate(self._sources):
            x, y = curve.getData()
            if y is None or len(y) < 12:
                continue              # too short for filtfilt's padding
            curve.setPen(stood_back)
            preview = pg.PlotDataItem(
                x, sosfiltfilt(sos, np.asarray(y, dtype=float)),
                pen=pg.mkPen(curve_color(index), width=1))
            # over the raw trace: the preview is what is being decided
            preview.setZValue(15)
            self.plot.addItem(preview)
            self._previews.append(preview)

    def _take_down(self) -> None:
        for preview in self._previews:
            self.plot.removeItem(preview)
        self._previews = []

    def remove(self) -> None:
        """Take the previews off and give the sources their pens back."""
        self._take_down()
        for curve, pen in zip(self._sources, self._pens):
            if pen is not None:
                curve.setPen(pen)
