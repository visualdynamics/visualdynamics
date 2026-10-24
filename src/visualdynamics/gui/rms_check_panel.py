"""The rms check, beside the autospectrum it checks.

The RMS Check reading of a plain autospectrum compares the rms from the
area under it with the rms of the time history it came from
(`core.rms_check`), and this panel says what was found: the two rms in
the display unit, the gap between them in dB, the frames' own rms and
the window's share of the gap when the averaging is known, the lines
summed and the span the record was read over, then the verdict in
words and anything the reader should know under it. The plot carries
the judgment and the gap in its corner (`plot.rms_check`).

Two settings are made here. The time history defaults to the one the
spectrum was computed from (`Project.rms_source`), and any other record
in the project holding the same channel can be chosen instead, which is
how an imported spectrum, a controller's, is checked against its run.
The threshold is the gap beyond which the two are said to disagree, a
few hundredths of a decibel by default. Nothing is made, so there is no
Apply button: the reading is of a spectrum that already exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from ..core.rms_check import GAP_DB
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: the readings under the rule, in order: key and row name
ROWS = (('channel', 'Channel'), ('spectrum', 'From the spectrum'),
        ('time', 'From the time history'), ('gap', 'Gap'),
        ('framed', 'From the frames'), ('share', "Window's share"),
        ('lines', 'Lines'), ('span', 'Span'), ('window', 'Window'))


class RmsCheckPanel(QWidget):
    """The time history and the threshold, then the check. Edits are
    emitted as {'history': name, 'threshold_db': dB}."""

    #: the time history or the threshold was changed; the settings now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('RMS Check')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.history_box: QComboBox = QComboBox()
        self.history_box.setToolTip(
            'The time history the spectrum is checked against: the one '
            'it was computed from, or another record holding the same '
            'channel, such as the run an imported spectrum came from')
        self.threshold_box: DoubleSpinBox = DoubleSpinBox()
        self.threshold_box.setDecimals(3)
        self.threshold_box.setRange(0.0, 20.0)
        self.threshold_box.setSingleStep(0.01)
        self.threshold_box.setSuffix(' dB')
        self.threshold_box.setValue(GAP_DB)
        self.threshold_box.setToolTip(
            'The gap beyond which the spectrum and the record disagree. '
            'A windowed average of a long random record lands within a '
            "few hundredths of a decibel; a window's amplitude correction "
            'applied to a density reads 1.76 dB high on Hann')
        commit_on_enter(self.threshold_box)
        add_settings(grid, [('Time history', self.history_box),
                            ('Threshold', self.threshold_box)])
        add_rule(grid, 3)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 4).items()}
        # the verdict in words, then the notes, under the numbers
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        font = self.verdict.font()
        font.setBold(True)
        self.verdict.setFont(font)
        grid.addWidget(self.verdict, 4 + len(ROWS), 0, 1, 2)
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 5 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(6 + len(ROWS), 1)
        self.history_box.currentIndexChanged.connect(self._edited)
        self.threshold_box.valueChanged.connect(self._edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """{'history': the time history's name or None, 'threshold_db':
        the threshold in dB}."""
        data = self.history_box.currentData()
        return {'history': None if data is None else str(data),
                'threshold_db': float(self.threshold_box.value())}

    def set_choices(self, histories: Sequence[str], chosen: str | None,
                    source: str | None, threshold_db: float) -> None:
        """Offer `histories` by name, `chosen` selected and `source` (the
        one the spectrum was computed from) marked as such, and show
        `threshold_db`, without emitting."""
        self._loading = True
        try:
            self.history_box.clear()
            for name in histories:
                said = f'{name} (computed from)' if name == source else name
                self.history_box.addItem(said, name)
            index = self.history_box.findData(chosen)
            self.history_box.setCurrentIndex(max(index, 0))
            self.threshold_box.setValue(float(threshold_db))
        finally:
            self._loading = False

    def show_check(self, check: Any, *, unit: str = '',
                   scale: float = 1.0, refusal: str | None = None) -> None:
        """Write `check`, an `RmsCheck`, into the rows, its rms converted
        by `scale` into `unit`; None clears them, with `refusal` said in
        its place when there is one."""
        def rms(value: float | None) -> str | None:
            if value is None:
                return None
            return f'{value * scale:.4g} {unit} rms'.replace('  ', ' ')

        def db(value: float | None, digits: int = 3) -> str | None:
            if value is None or not np.isfinite(value):
                return None
            return f'{value:+.{digits}f} dB'

        values = dict.fromkeys(key for key, _name in ROWS)
        if check is not None:
            span = check.span
            values.update(
                channel=check.dof or None,
                spectrum=rms(check.spectrum_rms),
                time=rms(check.time_rms),
                gap=db(check.gap_db),
                framed=rms(check.framed_rms),
                share=db(check.window_gap_db),
                lines=f'{check.lines}, {check.spacing:.4g} Hz apart',
                span=(None if span is None else
                      f'{span[0]:.4g} to {span[1]:.4g} s'),
                window=('not known' if check.window is None else
                        f'{check.window}, {check.noise_bandwidth:.3g} lines'))
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        if check is None:
            self.verdict.setText('' if refusal is None
                                 else refusal[0].upper() + refusal[1:] + '.')
            self.notes.setText('')
            return
        verdict = check.verdict
        self.verdict.setText(verdict[0].upper() + verdict[1:] + '.')
        self.notes.setText('\n'.join(note[0].upper() + note[1:] + '.'
                                     for note in check.notes))
