"""The two matrix checks of a multi-input test, read at a cursor.

The Conditioning toggle on the bar of an FRF from several references,
or of a CSD, draws the matrix's singular values and its condition
number against frequency (`plot.matrix_checks`) and puts a cursor on
the plot; `ConditioningPanel` says, at the cursor's line, the condition
number against its limit, the effective rank and the singular values,
with the whole matrix's verdict and notes beneath. Nothing is set and
nothing is made, so it is read-only.

The Realizability toggle on the bar of a CSD or a specification draws
the eigenvalues of its coherence matrix, and `RealizabilityPanel` reads
them at the cursor: the smallest against the largest, the rank, the
departure from Hermitian, and the verdict at that line and over the
whole matrix. Its one button, Make Realizable, is the apply of the
reading: it runs `Project.make_realizable`, which adds the repaired
matrix beside the one read, and it is offered only while there is
something to repair.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QWidget

from .settings_panel import add_derived, panel_grid

#: the conditioning readings, in order: key and row name
CONDITIONING_ROWS = (('matrix', 'Matrix'), ('frequency', 'At'),
                     ('condition', 'Condition number'),
                     ('rank', 'Effective rank'),
                     ('singular', 'Singular values'))

#: the realizability readings, in order: key and row name
REALIZABILITY_ROWS = (('channels', 'Channels'), ('frequency', 'At'),
                      ('line', 'This line'),
                      ('smallest', 'Smallest eigenvalue'),
                      ('largest', 'Largest eigenvalue'),
                      ('rank', 'Rank'), ('asymmetry', 'Off Hermitian'),
                      ('changes', 'Repair moves'))


def _sentences(notes: list[str]) -> str:
    return '\n'.join(note[0].upper() + note[1:] + '.' for note in notes
                     if note)


class _ReadingPanel(QWidget):
    """Rows of readings at a line, then the verdict and the notes."""

    ROWS: tuple[tuple[str, str], ...] = ()
    TITLE = ''

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel(self.TITLE)
        self.grid: QGridLayout = panel_grid(self, self.title)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(self.grid, self.ROWS, 1).items()}
        #: the whole matrix's verdict, in a sentence
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        self.grid.addWidget(self.verdict, 1 + len(self.ROWS), 0, 1, 2)
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        self.grid.addWidget(self.notes, 2 + len(self.ROWS), 0, 1, 2)

    def _write(self, values: dict[str, str | None], verdict: str,
               notes: list[str]) -> None:
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        self.verdict.setText(verdict)
        self.notes.setText(_sentences(notes))


class ConditioningPanel(_ReadingPanel):
    """The conditioning reading at the cursor's line."""

    ROWS = CONDITIONING_ROWS
    TITLE = 'Conditioning'

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid.setRowStretch(3 + len(self.ROWS), 1)

    def show_reading(self, reading: Any, point: Any = None, *,
                     scale: float = 1.0, unit: str = '',
                     refusal: str = '') -> None:
        """Write `point`, a `ConditioningPoint` of `reading`, into the
        rows, the singular values converted by `scale` into `unit`, with
        the reading's verdict and notes beneath. None clears the rows,
        and `refusal` says why there is no reading."""
        values: dict[str, str | None] = dict.fromkeys(
            key for key, _name in self.ROWS)
        if reading is None or point is None:
            self._write(values, refusal, [])
            return
        unit_text = f' {unit}' if unit else ''
        judged = not np.isnan(point.condition)
        values.update(
            matrix=reading.title,
            frequency=f'{point.frequency:.4g} Hz',
            condition=(f'{point.condition:.4g} '
                       f'({"over" if point.ill else "within"} '
                       f'{point.limit:g})' if judged else 'not judged'),
            rank=f'{point.rank} of {point.size}' if judged else None,
            singular=(', '.join(f'{value * scale:.3g}'
                                for value in point.singular) + unit_text
                      if judged else None))
        notes = list(reading.notes)
        if point.ill:
            notes.insert(0, 'at this line an inverse of the matrix '
                            'amplifies an error in it by up to the '
                            'condition number: the references reach the '
                            'rows in nearly the same proportions')
        self._write(values, reading.describe(), notes)


class RealizabilityPanel(_ReadingPanel):
    """The realizability reading at the cursor's line, and the button
    that repairs the matrix."""

    ROWS = REALIZABILITY_ROWS
    TITLE = 'Realizability'
    #: Make Realizable was pressed
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.apply_button: QPushButton = QPushButton('Make Realizable')
        self.apply_button.setToolTip(
            'Add a copy of this matrix repaired to the nearest realizable '
            'one at every line that is not, the autospectra held where '
            'they are and only the cross terms moved')
        self.apply_button.clicked.connect(self.apply_asked.emit)
        self.apply_button.setEnabled(False)
        self.grid.addWidget(self.apply_button, 3 + len(self.ROWS), 0, 1, 2)
        self.grid.setRowStretch(4 + len(self.ROWS), 1)

    def show_reading(self, reading: Any, point: Any = None, *,
                     refusal: str = '') -> None:
        """Write `point`, a `RealizabilityPoint` of `reading`, into the
        rows, with the reading's verdict and notes beneath, and offer
        the repair only where the matrix needs one. None clears the
        rows, and `refusal` says why there is no reading."""
        values: dict[str, str | None] = dict.fromkeys(
            key for key, _name in self.ROWS)
        if reading is None or point is None:
            self.apply_button.setEnabled(False)
            self._write(values, refusal, [])
            return
        judged = bool(point.eigenvalues) and not np.isnan(point.eigenvalues[0])
        coherence, phase = reading.changes()
        values.update(
            channels=f'{len(reading.channels)}',
            frequency=f'{point.frequency:.4g} Hz',
            line=(('realizable' if point.realizable else 'not realizable')
                  if judged else 'not judged'),
            smallest=f'{point.eigenvalues[0]:.4g}' if judged else None,
            largest=f'{point.eigenvalues[-1]:.4g}' if judged else None,
            rank=(f'{point.rank} of {len(point.eigenvalues)}'
                  if judged else None),
            asymmetry=f'{point.asymmetry:.2g}' if judged else None,
            changes=(f'coherence {coherence:.3g}, phase {phase:.3g}°'
                     if not reading.realizable and reading.judged.any()
                     else 'nothing'))
        notes = list(reading.notes)
        if judged and not point.realizable:
            notes.insert(0, 'at this line no set of signals has this '
                            'matrix: a combination of the channels is '
                            'asked for negative power')
        self.apply_button.setEnabled(bool(reading.judged.any()
                                          and not reading.realizable))
        self._write(values, reading.describe(), notes)
