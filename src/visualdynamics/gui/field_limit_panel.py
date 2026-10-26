"""The field limit's settings and what it found, beside the set drawn
with its limit and envelope.

The Field Limit reading of a set of measured PSDs or SRSs (the records
of one, or several selected together) takes a statistical limit over
the set and writes a straight-line envelope over it
(`core.field_limit`); the plot draws the set, the limit and the
envelope (`plot.field_limit`). The settings are the limit's (which
statistic, the coverage and confidence, k, and a standard deviation to
state instead of estimating one), the envelope's (the overshoot
allowed, the most breakpoints, and whether the rms is exceeded or
kept) and the margin to add. Rows that a statistic does not use are
hidden. Under the rule the panel reads out what the limit rests on,
the factor, the breakpoints, the overshoot, the rms and the area
added, and the notes; a refusal is said in their place.

Make Specification is the apply button: it writes the specification
the reading shows into the project (`Project.derive_specification`),
with the margin added.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QWidget

from ..core import field_limit as fl
from .editors import DoubleSpinBox, SpinBox, commit_on_enter
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the statistics offered, in order: key and what the box says
CHOICES = (('tolerance', 'Normal tolerance limit'),
           ('mean_k', 'Mean + k std. dev.'),
           ('envelope', 'Maximum envelope'),
           ('distribution_free', 'Distribution-free limit'))

#: the readings under the rule: key and row name
ROWS = (('records', 'Records'), ('factor', 'Factor k'),
        ('scatter', 'Scatter (median)'), ('breakpoints', 'Breakpoints'),
        ('overshoot', 'Most over limit'), ('limit_rms', 'Limit rms'),
        ('envelope_rms', 'Envelope rms'), ('added', 'Area added'))


class FieldLimitPanel(QWidget):
    """The limit's and the envelope's settings, then the reading.
    Edits are emitted as the dict `settings` returns."""

    #: a setting was changed; the settings now
    changed = Signal(dict)
    #: Make Specification was pressed
    make_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Field Limit')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.statistic_box: QComboBox = QComboBox()
        for key, said in CHOICES:
            self.statistic_box.addItem(said, key)
        self.statistic_box.setToolTip(
            'The limit taken over the set at each line: the normal '
            'tolerance limit (the levels log-normal across records), the '
            'mean plus k standard deviations in dB, the largest record, '
            'or the largest record as a distribution-free limit')
        self.coverage_box: DoubleSpinBox = self._percent(
            fl.COVERAGE, 'The share of the population the limit stands '
            'above')
        self.confidence_box: DoubleSpinBox = self._percent(
            fl.CONFIDENCE, 'The probability that it does: 50 % is the '
            'estimate as likely high as low')
        self.k_box: DoubleSpinBox = DoubleSpinBox()
        self.k_box.setRange(-10.0, 20.0)
        self.k_box.setDecimals(3)
        self.k_box.setSingleStep(0.1)
        self.k_box.setValue(2.0)
        self.k_box.setToolTip('Standard deviations above the mean, in dB')
        self.stated_box: QCheckBox = QCheckBox('State it')
        self.stated_box.setToolTip(
            'Take the standard deviation as known instead of estimating '
            'it from the records, which a single record needs')
        self.sigma_box: DoubleSpinBox = DoubleSpinBox()
        self.sigma_box.setRange(0.01, 30.0)
        self.sigma_box.setDecimals(2)
        self.sigma_box.setSuffix(' dB')
        self.sigma_box.setValue(3.0)
        self.sigma_box.setToolTip('The standard deviation stated, in dB')
        self.overshoot_box: DoubleSpinBox = DoubleSpinBox()
        self.overshoot_box.setRange(0.0, 40.0)
        self.overshoot_box.setDecimals(2)
        self.overshoot_box.setSingleStep(0.5)
        self.overshoot_box.setSuffix(' dB')
        self.overshoot_box.setValue(fl.OVERSHOOT_DB)
        self.overshoot_box.setToolTip(
            'The most the envelope may stand above the limit at any line')
        self.breakpoints_box: SpinBox = SpinBox()
        self.breakpoints_box.setRange(2, 200)
        self.breakpoints_box.setValue(fl.MAX_BREAKPOINTS)
        self.breakpoints_box.setToolTip(
            'The most breakpoints the envelope may have; where they run '
            'out first, the overshoot allowed is not met and the reading '
            'says so')
        self.rms_box: QComboBox = QComboBox()
        self.rms_box.addItem('Exceed', 'exceed')
        self.rms_box.addItem('Conserve', 'conserve')
        self.rms_box.setToolTip(
            'Exceed: the envelope covers the limit and raises the rms. '
            'Conserve: the envelope lowered until its rms is the '
            "limit's, falling under it at some lines (a PSD only)")
        self.margin_box: DoubleSpinBox = DoubleSpinBox()
        self.margin_box.setRange(-20.0, 40.0)
        self.margin_box.setDecimals(2)
        self.margin_box.setSingleStep(0.5)
        self.margin_box.setSuffix(' dB')
        self.margin_box.setValue(0.0)
        self.margin_box.setToolTip(
            'Decibels added to the envelope when the specification is '
            'made: 10 log10 on a PSD, 20 log10 on an SRS')
        commit_on_enter(self.coverage_box, self.confidence_box, self.k_box,
                        self.sigma_box, self.overshoot_box,
                        self.breakpoints_box, self.margin_box)
        rows = [('statistic', 'Limit', self.statistic_box),
                ('coverage', 'Coverage', self.coverage_box),
                ('confidence', 'Confidence', self.confidence_box),
                ('k', 'k', self.k_box),
                ('stated', 'Std. dev.', self.stated_box),
                ('sigma', 'Stated', self.sigma_box),
                ('overshoot', 'Overshoot', self.overshoot_box),
                ('breakpoints', 'Breakpoints', self.breakpoints_box),
                ('rms', 'RMS', self.rms_box),
                ('margin', 'Margin', self.margin_box)]
        self.names, self.editors = add_keyed_settings(grid, rows)
        first = 1 + len(rows)
        add_rule(grid, first)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, first + 1).items()}
        below = first + 1 + len(ROWS)
        self.rests_on: QLabel = QLabel('')
        self.rests_on.setWordWrap(True)
        grid.addWidget(self.rests_on, below, 0, 1, 2)
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, below + 1, 0, 1, 2)
        self.make_button: QPushButton = QPushButton('Make Specification')
        self.make_button.setToolTip(
            'Add the specification this envelope writes to the project, '
            'with the margin added')
        self.make_button.clicked.connect(self.make_asked.emit)
        grid.addWidget(self.make_button, below + 2, 0, 1, 2)
        grid.setRowStretch(below + 3, 1)
        for box in (self.statistic_box, self.rms_box):
            box.currentIndexChanged.connect(self._edited)
        for box in (self.coverage_box, self.confidence_box, self.k_box,
                    self.sigma_box, self.overshoot_box, self.breakpoints_box,
                    self.margin_box):
            box.valueChanged.connect(self._edited)
        self.stated_box.toggled.connect(self._edited)
        self._show_rows()

    @staticmethod
    def _percent(value: float, tip: str) -> DoubleSpinBox:
        box = DoubleSpinBox()
        box.setRange(0.1, 99.9)
        box.setDecimals(1)
        box.setSuffix(' %')
        box.setValue(100.0 * value)
        box.setToolTip(tip)
        return box

    def _show_rows(self) -> None:
        """Show only the rows the statistic uses."""
        statistic = self.statistic_box.currentData()
        uses = {
            'coverage': True,
            'confidence': statistic in ('tolerance', 'distribution_free'),
            'k': statistic == 'mean_k',
            'stated': statistic in ('tolerance', 'mean_k'),
            'sigma': (statistic in ('tolerance', 'mean_k')
                      and self.stated_box.isChecked()),
        }
        for key, shown in uses.items():
            self.names[key].setVisible(shown)
            self.editors[key].setVisible(shown)

    def _edited(self, *_args: Any) -> None:
        self._show_rows()
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """The settings as `Project.field_limit` takes them, with
        `margin_db` beside them; only what the statistic uses is set."""
        statistic = str(self.statistic_box.currentData())
        out: dict[str, Any] = {
            'statistic': statistic,
            'coverage': round(self.coverage_box.value() / 100.0, 6),
            'max_overshoot_db': float(self.overshoot_box.value()),
            'max_breakpoints': int(self.breakpoints_box.value()),
            'rms': str(self.rms_box.currentData()),
        }
        if statistic in ('tolerance', 'distribution_free'):
            out['confidence'] = round(self.confidence_box.value() / 100.0, 6)
        if statistic == 'mean_k':
            out['k'] = float(self.k_box.value())
        if statistic in ('tolerance', 'mean_k') and self.stated_box.isChecked():
            out['sigma_db'] = float(self.sigma_box.value())
        out['margin_db'] = float(self.margin_box.value())
        return out

    def set_settings(self, settings: dict[str, Any]) -> None:
        """Show `settings` (as `settings` returns them) without
        emitting."""
        self._loading = True
        try:
            index = self.statistic_box.findData(
                settings.get('statistic', 'tolerance'))
            self.statistic_box.setCurrentIndex(max(index, 0))
            self.coverage_box.setValue(
                100.0 * settings.get('coverage', fl.COVERAGE))
            self.confidence_box.setValue(
                100.0 * settings.get('confidence', fl.CONFIDENCE))
            if 'k' in settings:
                self.k_box.setValue(float(settings['k']))
            self.stated_box.setChecked('sigma_db' in settings)
            if 'sigma_db' in settings:
                self.sigma_box.setValue(float(settings['sigma_db']))
            self.overshoot_box.setValue(
                settings.get('max_overshoot_db', fl.OVERSHOOT_DB))
            self.breakpoints_box.setValue(
                settings.get('max_breakpoints', fl.MAX_BREAKPOINTS))
            self.rms_box.setCurrentIndex(
                max(self.rms_box.findData(settings.get('rms', 'exceed')), 0))
            self.margin_box.setValue(settings.get('margin_db', 0.0))
        finally:
            self._loading = False
        self._show_rows()

    def show_fit(self, fit: Any, *, unit: str = '', scale: float = 1.0,
                 refusal: str | None = None) -> None:
        """Write `fit`, an `EnvelopeFit`, into the rows, its rms
        converted by `scale` into `unit`; None clears them, with
        `refusal` said in their place. Make Specification is offered
        only when there is a fit to make one from."""
        values = dict.fromkeys(key for key, _name in ROWS)
        if fit is not None:
            limit = fit.limit
            values['records'] = str(limit.count)
            if limit.factor is not None:
                values['factor'] = f'{limit.factor:.3f}'
            if limit.sigma_db is not None:
                values['scatter'] = f'{limit.sigma_db:g} dB stated'
            elif limit.scatter_db is not None:
                values['scatter'] = f'{float(np.median(limit.scatter_db)):.2f} dB'
            values['breakpoints'] = (f'{fit.breakpoints}'
                                     + ('' if fit.met else ', the most'))
            values['overshoot'] = f'{fit.overshoot_db:.2f} dB'
            if limit.kind == 'psd':
                values['limit_rms'] = f'{fit.limit_rms * scale:.4g} {unit}'.strip()
                values['envelope_rms'] = (f'{fit.envelope_rms * scale:.4g} '
                                          f'{unit}').strip()
                values['added'] = (f'{fit.added_db:+.2f} dB' if fit.rule == 'exceed'
                                   else f'kept; {fit.below} lines under')
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        self.make_button.setEnabled(fit is not None)
        if fit is None:
            self.rests_on.setText('' if refusal is None
                                  else refusal[0].upper() + refusal[1:] + '.')
            self.notes.setText('')
            return
        said = fit.limit.rests_on
        self.rests_on.setText(said[0].upper() + said[1:] + '.')
        self.notes.setText('\n'.join(note[0].upper() + note[1:] + '.'
                                     for note in fit.notes))
