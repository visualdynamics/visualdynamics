"""The pieces every settings panel is built from.

The settings panels (averaging, filter, sine, truncate, wavelet, rigid
body, octave, and the geometry panes) share one frame: the grid, the
fixed width and the bold title, the editable rows, the rule under them
and the derived readings below it. Those are built here once, and each
panel keeps what is its own: which rows, which readings, and its
button.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Seven panels once carried their own copies of the frame and of the
# loops that lay out the rows and readings (2026-09-12; the rows and the
# rule, 2026-10-10).

#: every settings panel's width: wide enough for the widest label and a
#: spin box beside it, narrow enough to leave the plot the room
PANEL_WIDTH = 210


def panel_grid(panel: QWidget, title: QLabel) -> QGridLayout:
    """A grid layout on `panel` with `title`, set in bold, across its
    first row. The panel's width is fixed at `PANEL_WIDTH`."""
    grid = QGridLayout(panel)
    grid.setContentsMargins(8, 8, 8, 8)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(4)
    panel.setSizePolicy(QSizePolicy.Policy.Fixed,
                        QSizePolicy.Policy.Preferred)
    panel.setFixedWidth(PANEL_WIDTH)
    font = title.font()
    font.setBold(True)
    title.setFont(font)
    grid.addWidget(title, 0, 0, 1, 2)
    return grid


def add_settings(grid: QGridLayout, rows, first_row: int = 1) -> list[QLabel]:
    """Add the editable rows: a name and its editor for each
    `(label, editor)`, from `first_row` down. Returns the name labels in
    order, for a panel that renames or hides a row as its settings
    change."""
    names = []
    for offset, (label, editor) in enumerate(rows):
        name = QLabel(label)
        grid.addWidget(name, first_row + offset, 0)
        grid.addWidget(editor, first_row + offset, 1)
        names.append(name)
    return names


def add_keyed_settings(grid: QGridLayout, rows, first_row: int = 1
                       ) -> tuple[dict[str, QLabel], dict[str, QWidget]]:
    """`add_settings` for rows a panel looks up by key, each a
    `(key, label, editor)`, so one can be renamed or hidden as the
    settings change. Returns two dicts: key → name label and key →
    editor."""
    names = add_settings(grid, [(label, editor) for _key, label, editor in rows],
                         first_row)
    return ({key: name for (key, _label, _editor), name in zip(rows, names)},
            {key: editor for key, _label, editor in rows})


def add_rule(grid: QGridLayout, row: int) -> None:
    """Add the sunken line across the panel, at `row`, between the
    editable rows and the readings that follow from them."""
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.HLine)
    rule.setFrameShadow(QFrame.Shadow.Sunken)
    grid.addWidget(rule, row, 0, 1, 2)


def add_derived(grid: QGridLayout, derived, first_row: int
                ) -> dict[str, tuple[QLabel, QLabel]]:
    """Add the read-only rows under the editable ones: a grayed name and
    a right-aligned value (a dash until it is set) for each
    `(key, label)`, from `first_row` down. Returns key → (name label,
    value label)."""
    made = {}
    for offset, (key, label) in enumerate(derived):
        name = QLabel(label)
        name.setEnabled(False)
        value = QLabel('—')
        value.setAlignment(Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(name, first_row + offset, 0)
        grid.addWidget(value, first_row + offset, 1)
        made[key] = (name, value)
    return made


class PreviewPlot(QWidget):
    """The small plot a panel draws its preview in, such as the filter's
    response or the sine extraction's sampled readings. Frequency is on
    a log axis, the mouse and the menu are turned off, and the axes are
    in the theme's foreground color. A subclass adds its own curves and
    sets only how it differs, through `HEIGHT`, `LOG_Y` and
    `AXIS_WIDTH`."""

    # The filter and sine panels built this plot line for line apart
    # until 2026-10-10.

    HEIGHT = 150
    LOG_Y = True
    AXIS_WIDTH = 34

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        from ..theme import theme as resolve_theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.colors: Mapping[str, str] = resolve_theme(None)
        self.widget: Any = pg.PlotWidget()
        self.widget.setFixedHeight(self.HEIGHT)
        self.plot: Any = self.widget.getPlotItem()
        self.plot.setLogMode(x=True, y=self.LOG_Y)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        for edge in ('left', 'bottom'):
            axis = self.plot.getAxis(edge)
            axis.setStyle(tickTextOffset=2, tickLength=-3)
            axis.setTextPen(self.colors['plot_foreground'])
            axis.setPen(self.colors['plot_foreground'])
        self.plot.getAxis('left').setWidth(self.AXIS_WIDTH)
        layout.addWidget(self.widget)

    def apply_theme(self, colors: Mapping[str, str]) -> None:
        """Redraw the background and the axes in a new theme's colors."""
        self.colors = colors
        self.widget.setBackground(colors['plot_background'])
        for edge in ('left', 'bottom'):
            axis = self.plot.getAxis(edge)
            axis.setTextPen(colors['plot_foreground'])
            axis.setPen(colors['plot_foreground'])
