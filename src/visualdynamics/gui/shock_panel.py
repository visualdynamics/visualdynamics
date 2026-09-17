"""The shocks found in a time history, as a table beside it.

One row per event: when its window opens and how long it runs. Both are
editable, because detection is a good guess and not an oracle — and the
same two numbers are what dragging a region on the plot changes, so the
table and the trace are two views of one list.

The panel says where the events are, reports when that changes, and
its Compute SRS button turns the history into spectra with whatever is
set at that moment — the project verb a script calls. That is exactly
how the averaging panel stands to a PSD.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.shocks import Shock, held_apart, same_length, uniform

#: the panel's natural width — wide enough for two times in
#: milliseconds and a Detect button, narrow enough to leave the plot the
#: room it needs
WIDTH = 250

#: how many decimals a time is shown and edited to. Milliseconds:
#: shock windows are tens to hundreds of them, and a microsecond of
#: precision in a draggable edge is a pretense.
PLACES = 4


class ShockPanel(QWidget):
    """The events, and what can be done to them.

    `changed` carries the whole list rather than which row moved: the
    list is short, the receiver rewrites the overlay from it either way,
    and a diff nobody uses is a thing that can be wrong.
    """

    changed = Signal(object)
    detect_asked = Signal()
    #: the user asked for the SRS these windows parameterize
    srs_asked = Signal()
    #: the Add button: a window the detector missed; the owner places it
    add_asked = Signal()
    #: the "same length" box was ticked or cleared, carrying its state
    length_mode_changed = Signal(bool)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed,
                           QSizePolicy.Policy.Expanding)
        self._shocks = ()
        self._writing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.heading: QLabel = QLabel('Shocks')
        self.heading.setStyleSheet('font-weight: 600;')
        layout.addWidget(self.heading)

        self.table: QTableWidget = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(['Start [s]', 'Length [s]'])
        self.table.verticalHeader().setVisible(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.itemChanged.connect(self._edited)
        layout.addWidget(self.table, 1)

        self.same_length: QCheckBox = QCheckBox('Same length for all')
        self.same_length.setToolTip(
            'Analyze every event in a window of one length, so what '
            'differs between their spectra is the events.\n'
            'Dragging any edge then sets the length for all of them.')
        self.same_length.toggled.connect(self._length_mode)
        layout.addWidget(self.same_length)

        self.summary: QLabel = QLabel('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        buttons = QHBoxLayout()
        self.detect_button: QPushButton = QPushButton('Detect')
        self.detect_button.setToolTip(
            'Find the shocks in this record and replace the list')
        self.detect_button.clicked.connect(self.detect_asked.emit)
        buttons.addWidget(self.detect_button)
        self.add_button: QPushButton = QPushButton('Add')
        self.add_button.setToolTip(
            'Add a window the detector missed, in the largest open '
            'stretch — drag or type it into place')
        self.add_button.clicked.connect(self.add_asked.emit)
        buttons.addWidget(self.add_button)
        self.remove_button: QPushButton = QPushButton('Remove')
        self.remove_button.setToolTip('Drop the selected shock')
        self.remove_button.clicked.connect(self._remove)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)
        # the analysis choices the SRS is computed with (Brandon,
        # 2026-08-29): parameters of the act, recorded in its recipe —
        # a refresh after the windows move keeps them. There is no
        # mass setting because the standard SRS has none: the SDOF's
        # mass cancels for base-input response, so Q, spacing and
        # which peak are the whole parameter space.
        from ..core.srs import DEFAULT_Q, PER_OCTAVE
        from .editors import DoubleSpinBox, SpinBox, commit_on_enter

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.q_box: DoubleSpinBox = DoubleSpinBox()
        self.q_box.setDecimals(1)
        self.q_box.setRange(1.0, 500.0)
        self.q_box.setValue(DEFAULT_Q)
        self.q_box.setToolTip(
            'The oscillator amplification, Q = 1/(2ζ): 10 is 5% '
            'damping, the shock-test convention')
        self.kind_box: QComboBox = QComboBox()
        for key, label in (('maximax', 'Maximax'),
                           ('positive', 'Positive'),
                           ('negative', 'Negative')):
            self.kind_box.addItem(label, key)
        self.kind_box.setToolTip(
            'Which peak each oscillator reports: the largest magnitude '
            'of either sign (maximax), or the positive or negative '
            'excursion alone')
        self.per_octave_box: SpinBox = SpinBox()
        self.per_octave_box.setRange(1, 96)
        self.per_octave_box.setValue(PER_OCTAVE)
        self.per_octave_box.setToolTip(
            'Natural-frequency lines per octave')
        for row, (label, editor) in enumerate((
                ('Q', self.q_box), ('Peak', self.kind_box),
                ('Per octave', self.per_octave_box))):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(editor, row, 1)
        band_name = QLabel('Band')
        band_name.setEnabled(False)
        self.band_label: QLabel = QLabel('—')
        self.band_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                     | Qt.AlignmentFlag.AlignVCenter)
        self.band_label.setToolTip(
            'What the windows can support: low enough that the '
            'shortest window still holds a cycle, up to a fifth of '
            'the sample rate')
        grid.addWidget(band_name, 3, 0)
        grid.addWidget(self.band_label, 3, 1)
        layout.addLayout(grid)
        commit_on_enter(self.q_box, self.per_octave_box)

        # the act these windows parameterize, right where they are set
        self.srs_button: QPushButton = QPushButton('Compute SRS')
        self.srs_button.setToolTip(
            'A shock response spectrum over each of these windows, '
            'every channel')
        self.srs_button.clicked.connect(self.srs_asked.emit)
        layout.addWidget(self.srs_button)

    # ---- what it is showing ----------------------------------------------

    def shocks(self) -> tuple[Shock, ...]:
        return self._shocks

    def set_shocks(self, shocks: Sequence[Shock],
                   locked: bool = False) -> None:
        """Show these, without reporting them back as an edit."""
        shocks = tuple(shocks)
        self._shocks = shocks
        self._writing = True
        try:
            self.table.setRowCount(len(shocks))
            for row, shock in enumerate(shocks):
                for column, value in ((0, shock.start), (1, shock.duration)):
                    item = QTableWidgetItem(f'{value:.{PLACES}f}')
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
                    if locked:
                        item.setFlags(item.flags()
                                      & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, column, item)
        finally:
            self._writing = False
        self.table.setEnabled(not locked)
        self.detect_button.setEnabled(not locked)
        # unlike Remove, Add serves the empty list too — that is the
        # list most in need of a first window
        self.add_button.setEnabled(not locked)
        self.remove_button.setEnabled(not locked and bool(shocks))
        # derived, never stored: whether the series is at one length is
        # already written in the lengths, and a second copy of it in
        # the project file is a copy that can disagree with them
        self._writing = True
        try:
            self.same_length.setChecked(uniform(shocks))
        finally:
            self._writing = False
        self.same_length.setEnabled(not locked and len(shocks) > 1)
        self._summarize(shocks, locked)

    def _summarize(self, shocks, locked):
        if locked:
            self.summary.setText('The file already cut this record into '
                                 'captures, so the events are settled.')
        elif not shocks:
            self.summary.setText('No shocks found. Detect looks for events '
                                 'that rise clear of the record’s own quiet.')
        else:
            covered = sum(s.duration for s in shocks)
            self.summary.setText(
                f'{len(shocks)} shock{"s" * (len(shocks) != 1)}, '
                f'{covered:.3f} s of record analyzed. One spectrum per '
                f'channel per shock.')

    # ---- edits ------------------------------------------------------------

    def _length_mode(self, on):
        """The box was ticked or cleared.

        Ticking it is an edit — it makes the windows agree — so it goes
        out as one, sized from the longest of them. Clearing it changes
        nothing about the list, only what the next drag will do, so
        there is nothing to report but the state itself.
        """
        if self._writing:
            return
        if on and len(self._shocks) > 1:
            settled = same_length(
                self._shocks, max(s.duration for s in self._shocks))
            if settled != self._shocks:
                self._shocks = settled
                self.set_shocks(settled)
                self.changed.emit(settled)
        self.length_mode_changed.emit(bool(on))

    def _edited(self, item):
        """A cell was typed into: rebuild that row and report the list.

        A value that will not parse, or that makes no window, puts the
        old one back rather than raising — the table is a control, and a
        control that throws at a typo is a trap.
        """
        if self._writing or item.row() >= len(self._shocks):
            return
        row, column = item.row(), item.column()
        was = self._shocks[row]
        try:
            value = float(item.text())
        except ValueError:
            self.set_shocks(self._shocks)
            return
        try:
            moved = (Shock(value, was.duration) if column == 0
                     else Shock(was.start, value))
        except ValueError:
            self.set_shocks(self._shocks)
            return
        edited = tuple(moved if k == row else shock
                       for k, shock in enumerate(self._shocks))
        # the same two rules the plot's drag goes through, because this
        # is the same edit arriving by another route — and the column
        # is this route's version of resize-or-move: a start is that
        # event's own, a length can be the series'
        self._shocks = (
            same_length(edited, moved.duration)
            if column == 1 and self.same_length.isChecked() and len(edited) > 1
            else held_apart(edited, moved=row))
        self.set_shocks(self._shocks)
        self.changed.emit(self._shocks)

    def _remove(self):
        """Drop the selected event.

        A shock that was never a shock — a switch transient, a dropped
        cable — is removed here rather than by moving a threshold until
        the detector agrees.
        """
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows:
            return
        kept = tuple(shock for k, shock in enumerate(self._shocks)
                     if k not in rows)
        self.set_shocks(kept)
        self.changed.emit(kept)

    def srs_settings(self) -> dict:
        """The analysis choices as `compute_srs` keyword arguments."""
        return {'q': float(self.q_box.value()),
                'kind': str(self.kind_box.currentData()),
                'per_octave': int(self.per_octave_box.value())}

    def show_band(self, low: float, high: float) -> None:
        """State the band the current windows support."""
        self.band_label.setText(f'{low:.4g}–{high:.4g} Hz')
