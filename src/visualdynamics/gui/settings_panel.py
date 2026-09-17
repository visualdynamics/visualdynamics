"""What every settings panel is built from.

Six panels — averaging, filter, truncate, wavelet, rigid body, octave
— wore the same chrome (the grid, the fixed width, the bold title) and
listed their derived readings with the same loop, each with its own
copy (2026-09-12). The copies live here once; each panel keeps what
is its own: the rows, the readings, the button.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

#: every settings panel's width: wide enough for the widest label and a
#: spin box beside it, narrow enough to leave the plot the room
PANEL_WIDTH = 210


def panel_grid(panel: QWidget, title: QLabel) -> QGridLayout:
    """The panel's grid with the title across its first row, the panel
    fixed at `PANEL_WIDTH` and never stretched taller than it asks."""
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


def add_derived(grid: QGridLayout, derived, first_row: int
                ) -> dict[str, tuple[QLabel, QLabel]]:
    """The read-only rows under the editable ones: a grayed name and a
    right-aligned value per `(key, label)`, from `first_row` down.
    Returns key → (name label, value label)."""
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
