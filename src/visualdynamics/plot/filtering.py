"""The filtered time history, previewed over the unfiltered one.

A corner frequency alone does not show what a filter will remove. The
preview draws each curve on the plot again through the filter being
set, so the noise removed and the content kept can be judged on the
data itself before Apply Filter creates anything.

The preview filters the y-data already on the plot. That is the whole
record in the units on screen, and a filter is linear, so the result is
the same as what `core.filters.filtered` will produce: both use the
same `core.filters.design` and scipy's `sosfiltfilt`. Each preview is
drawn in its channel's curve color, and the unfiltered curve under it
is drawn gray, the color a specification is drawn in behind its
response.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class FilterOverlay:
    """Filtered copies of a plot's curves, drawn over the originals.

    The overlay owns every item it adds and removes all of them in
    `remove`, which also gives each original curve its pen back, so a
    redraw cannot leave a preview behind.
    """

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
        """Redraw the previews for new filter settings (a corner or an order).

        Settings that cannot be designed at this sample rate, as happens
        while a number is being typed, draw no preview.
        """
        import pyqtgraph as pg
        from scipy.signal import sosfiltfilt

        from ..core.filters import design
        from . import CURVE_WIDTH, curve_color

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
        stood_back = pg.mkPen(self.colors['specification_curve'],
                              width=CURVE_WIDTH)
        for index, curve in enumerate(self._sources):
            x, y = curve.getData()
            if y is None or len(y) < 12:
                continue              # too short for filtfilt's padding
            curve.setPen(stood_back)
            preview = pg.PlotDataItem(
                x, sosfiltfilt(sos, np.asarray(y, dtype=float)),
                pen=pg.mkPen(curve_color(index, self.colors),
                             width=CURVE_WIDTH))
            # over the raw trace: the preview is what is being decided
            preview.setZValue(15)
            self.plot.addItem(preview)
            self._previews.append(preview)

    def _take_down(self) -> None:
        for preview in self._previews:
            self.plot.removeItem(preview)
        self._previews = []

    def remove(self) -> None:
        """Remove the previews and give the original curves their pens back."""
        self._take_down()
        for curve, pen in zip(self._sources, self._pens):
            if pen is not None:
                curve.setPen(pen)
