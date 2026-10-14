"""The octave-band settings, as a table beside the spectrum.

One setting is made here, the bands per octave, and the banded spectrum
it would make is previewed over the narrowband data it is made from.
The band steps are drawn on the same axes, so the trade between
legibility and resolution is judged on the spectrum being banded. These
are principle 13's four parts: the reading on the bar, these settings,
that preview, and the Apply button below them.

Nothing here computes the result. The button runs the same
`compute_octave` verb a script calls, with whatever spacing is set at
that moment.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.octave import CHOICES, PER_OCTAVE
from .settings_panel import add_derived, add_rule, add_settings, panel_grid


class OctavePanel(QWidget):
    """The settings table. Edits are emitted as bands per octave."""

    #: the spacing moved; here is bands-per-octave now
    changed = Signal(int)

    #: the user asked for the banded spectrum this panel describes
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Octave Bands')
        grid = panel_grid(self, self.title)

        self.per_box: QComboBox = QComboBox()
        for n in CHOICES:
            self.per_box.addItem(
                'full octave' if n == 1 else f'1/{n} octave', n)
        self.per_box.setCurrentIndex(CHOICES.index(PER_OCTAVE))
        self.per_box.setToolTip(
            'Band spacing. A sixth of an octave is the usual reading; '
            'a third and a whole octave are both ordinary. Each band '
            'takes the mean-square content that falls in it, so the '
            'RMS carried is unchanged whatever the spacing')
        add_settings(grid, [('Spacing', self.per_box)])

        add_rule(grid, 2)

        # what follows from the one, shown so the trade is visible
        # while it is being made
        self.derived: dict[str, QLabel] = {}
        derived = (('bands', 'Bands'), ('grid', 'Grid'))
        for key, (_name, value) in add_derived(grid, derived,
                                               3).items():
            self.derived[key] = value
        # the grid is absolute and base ten, and saying so here is what
        # makes "two runs land on the same bands" checkable at a glance
        self.derived['grid'].setText('ANSI S1.11, base 10')

        self.apply_button: QPushButton = QPushButton('Apply Octave Bands')
        self.apply_button.setToolTip(
            'Make the banded spectrum — exactly the steps previewed '
            'over the narrowband — beside this one')
        self.apply_button.clicked.connect(self.apply_asked.emit)
        grid.addWidget(self.apply_button, 5, 0, 1, 2)
        grid.setRowStretch(6, 1)

        self.per_box.currentIndexChanged.connect(
            lambda _index: self.changed.emit(self.per_octave()))

    def per_octave(self) -> int:
        """The bands per octave currently chosen."""
        return int(self.per_box.currentData())

    def show_bands(self, count: int | None) -> None:
        """Show how many bands the preview has, or a dash when `count`
        is None because the spectrum is refused (too narrow for even one
        band)."""
        self.derived['bands'].setText('—' if count is None
                                      else str(count))
