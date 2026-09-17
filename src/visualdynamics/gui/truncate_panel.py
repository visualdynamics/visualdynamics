"""The truncation parameters, as a table beside the time history.

Two instants are set here — where the kept stretch starts and where it
stops — and the record on the plot shows the cut as it is made: the
discarded ends grayed over, the kept span between two draggable edges
(`plot.truncation`), the same span-editing grammar as the averaging
region because it is the same kind of gesture.

The panel says what the settings are, reports when they move, and
its Apply Truncation button makes the truncated record with whatever
is set at that moment — the project verb a script calls, the same
division of labor as the averaging and filter panels, because it is
the same kind of thing: parameters that ride the record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.truncate import Truncation
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import TimeHistory


class TruncatePanel(QWidget):
    """The parameter table. Edits arrive as a whole `Truncation`."""

    #: a parameter moved; here is the truncation that describes it now
    changed = Signal(object)

    #: the user asked for the cut record this panel describes
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: the record the parameters describe, for counting samples
        self._history = None
        #: the record's own first and last instants — the walls the
        #: span cannot leave
        self._first = 0.0
        self._last = 1.0
        #: the least daylight between start and stop: one sample
        #: period, or the boxes' own resolution where that is coarser
        self._gap = 1e-4
        #: set while the panel is writing to its own editors, so that
        #: restating a clamped value does not read as a fresh edit
        self._loading = False

        self.title: QLabel = QLabel('Truncate')
        grid = panel_grid(self, self.title)

        self.start_box: DoubleSpinBox = DoubleSpinBox()
        self.start_box.setToolTip(
            'Where the kept stretch starts, on the record’s own clock. '
            'Drag the left edge of the span on the plot to set it by '
            'hand')
        self.stop_box: DoubleSpinBox = DoubleSpinBox()
        self.stop_box.setToolTip(
            'Where the kept stretch stops. Drag the right edge of the '
            'span on the plot to set it by hand')
        for box in (self.start_box, self.stop_box):
            box.setDecimals(4)
            box.setSuffix(' s')

        rows = (('Start', self.start_box), ('Stop', self.stop_box))
        for row, (label, editor) in enumerate(rows, start=1):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(editor, row, 1)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(rule, len(rows) + 1, 0, 1, 2)

        # what follows from the two, shown so the cost of the cut is
        # visible while it is being made rather than after the record
        # comes out
        self.derived: dict[str, QLabel] = {}
        derived = (('record', 'Record'), ('kept', 'Kept'),
                   ('samples', 'Samples'))
        for key, (_name, value) in add_derived(grid, derived,
                                               len(rows) + 2).items():
            self.derived[key] = value

        # the act itself, where its settings are set (Brandon,
        # 2026-08-28). Disabled while the span covers the whole
        # record — the cut of everything is a copy, and a button that
        # makes one is a button that does nothing. There is no
        # whole-record reset beside it (Brandon, 2026-08-28): the
        # boxes and the drag reach every span, this one included.
        self.apply_button: QPushButton = QPushButton('Apply Truncation')
        self.apply_button.clicked.connect(self.apply_asked.emit)
        grid.addWidget(self.apply_button, len(rows) + 2 + len(derived),
                       0, 1, 2)
        grid.setRowStretch(len(rows) + 3 + len(derived), 1)

        commit_on_enter(self.start_box, self.stop_box)
        self.start_box.valueChanged.connect(self._edited)
        self.stop_box.valueChanged.connect(self._edited)

    # ---- what the panel is describing -------------------------------------

    def show_history(self, history: TimeHistory,
                     truncation: Truncation) -> None:
        """Point the panel at a time history and the span set on it.

        The walls come from the history rather than being guessed from
        the numbers already in the boxes.
        """
        import numpy as np

        self._history = history
        abscissa = np.asarray(history.abscissa, dtype=float)
        self._first = float(abscissa[0])
        self._last = float(abscissa[-1])
        if len(abscissa) > 1:
            self._gap = max(float(abscissa[1] - abscissa[0]), 1e-4)
        self.set_truncation(truncation)

    def set_truncation(self, truncation: Truncation) -> None:
        """Restate the panel from a truncation, without re-emitting."""
        self._loading = True
        try:
            # the walls first, for the same reason every panel here
            # sets its limits first: a spin box clamps to the range it
            # has at the moment it is written
            self.start_box.setRange(self._first, self._last - self._gap)
            self.stop_box.setRange(self._first + self._gap, self._last)
            for box in (self.start_box, self.stop_box):
                box.setSingleStep(
                    max((self._last - self._first) / 100.0, self._gap))
            self.start_box.setValue(truncation.start)
            self.stop_box.setValue(truncation.stop)
            self._constrain()
        finally:
            self._loading = False
        self._restate(self.truncation())

    def truncation(self) -> Truncation:
        """The truncation the editors currently describe."""
        return Truncation(self.start_box.value(), self.stop_box.value())

    # ---- editing ----------------------------------------------------------

    def _constrain(self) -> None:
        """Keep the stop after the start, by the boxes' own ranges.

        The ranges are the enforcement — a value typed past the other
        edge clamps one sample short of it, the filter panel's rule —
        so a backwards span cannot be stated, rather than being caught
        after. Only ever called with `_loading` held, because a clamp
        writes the box and the write must not read as a fresh edit.
        """
        self.start_box.setRange(
            self._first, max(self.stop_box.value() - self._gap,
                             self._first))
        self.stop_box.setRange(
            min(self.start_box.value() + self._gap, self._last),
            self._last)

    def _edited(self, *_args):
        if self._loading:
            return
        self._loading = True
        try:
            self._constrain()
        finally:
            self._loading = False
        settled = self.truncation()
        self._restate(settled)
        self.changed.emit(settled)

    # ---- what follows from the two ----------------------------------------

    def _restate(self, truncation: Truncation) -> None:
        """Read the cut's cost off the record it will be made from."""
        import numpy as np

        self.derived['record'].setText(
            f'{self._last - self._first:.4g} s')
        kept = truncation.stop - truncation.start
        share = kept / max(self._last - self._first, 1e-12)
        self.derived['kept'].setText(f'{kept:.4g} s ({share:.0%})')
        if self._history is not None:
            abscissa = np.asarray(self._history.abscissa, dtype=float)
            inside = int(((abscissa >= truncation.start)
                          & (abscissa <= truncation.stop)).sum())
            self.derived['samples'].setText(
                f'{inside} of {len(abscissa)}')
        whole = (truncation.start <= self._first + self._gap / 2.0
                 and truncation.stop >= self._last - self._gap / 2.0)
        self.apply_button.setEnabled(not whole)
        self.apply_button.setToolTip(
            'Nothing to cut: the span still covers the whole record'
            if whole else
            'Make the record holding only this span, beside this one')
