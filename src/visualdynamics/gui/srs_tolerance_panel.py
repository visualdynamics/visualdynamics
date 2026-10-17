"""The points rule's settings, beside a shock spectrum judged against its
requirement.

The Points reading of a shock-spectrum comparison draws each selected
spectrum against the requirement's tolerance band, its points outside
marked (`plot.srs_tolerance`). This panel holds what that judgement is
made under (the inner band, the outer band if any, the rule and the
share it asks for, or the requirement's own limits as the bands when
it has them) and, under the rule, what came of it: for each
polarity the count and share of its points inside and the verdict, and
its worst point with its distance from the requirement in dB.

Every edit redraws the plot at once, so the effect of a band or a rule
is seen on the spectrum while it is set. Nothing is made: the
judgement is a reading of two objects that already exist, as the RMS
deviation bars are, so there is no Apply button to press. The same
call a script makes, `core.compliance.srs_tolerance`, gives every
number shown here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QWidget,
)

from ..core.compliance import SRS_INNER_DB, SRS_POINTS_PERCENT
from .editors import DoubleSpinBox
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the outer band offered when it is first ticked, dB below and above
OUTER_DB = (-3.0, 6.0)

#: the rules as the box names them, in `SRS_RULES` order
RULE_NAMES = (('share', 'Share inside'), ('above_lower', 'All above lower'))


def _spin(low: float, high: float, value: float, tip: str) -> DoubleSpinBox:
    """A dB (or percent) field in half-decibel steps, reporting only a
    finished edit, since each report redraws the judgement."""
    box = DoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(2)
    box.setSingleStep(0.5)
    box.setValue(value)
    box.setToolTip(tip)
    return box


class SrsTolerancePanel(QWidget):
    """The settings and the readout. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Points Rule')
        grid = panel_grid(self, self.title)
        self._loading = False
        #: which of the requirement's limits can stand as the bands,
        #: (warning, abort), and the requirement they were offered for
        self._has_limits: tuple[bool, bool] = (False, False)
        self._offered_for: Any = None

        self.rule_box: QComboBox = QComboBox()
        for key, name in RULE_NAMES:
            self.rule_box.addItem(name, key)
        self.rule_box.setToolTip(
            'Share inside: at least the share below of the points lie '
            'inside the inner band. All above lower: every point is at '
            'or above the inner band\'s lower edge, with no ceiling. '
            'The test plan says which applies')
        self.percent_box: DoubleSpinBox = _spin(
            0.0, 100.0, SRS_POINTS_PERCENT,
            'The share of the points, in percent, that must lie inside '
            'the inner band')
        self.percent_box.setSingleStep(5.0)
        self.inner_low: DoubleSpinBox = _spin(
            -40.0, 0.0, SRS_INNER_DB[0],
            'The inner band\'s lower edge, dB below the requirement')
        self.inner_high: DoubleSpinBox = _spin(
            0.0, 40.0, SRS_INNER_DB[1],
            'The inner band\'s upper edge, dB above the requirement')
        self.limits_box: QCheckBox = QCheckBox()
        self.limits_box.setToolTip(
            'Judge against the requirement\'s own warning and abort '
            'limits, the curves the Spectra reading shades')
        self.outer_box: QCheckBox = QCheckBox()
        self.outer_box.setToolTip(
            'An outer band every point must lie inside, whatever the '
            'rule says')
        self.outer_low: DoubleSpinBox = _spin(
            -40.0, 0.0, OUTER_DB[0], 'The outer band\'s lower edge, dB')
        self.outer_high: DoubleSpinBox = _spin(
            0.0, 40.0, OUTER_DB[1], 'The outer band\'s upper edge, dB')
        self.names, self.editors = add_keyed_settings(grid, [
            ('rule', 'Rule', self.rule_box),
            ('percent', 'Share [%]', self.percent_box),
            ('limits', 'Spec limits', self.limits_box),
            ('inner_low', 'Inner below [dB]', self.inner_low),
            ('inner_high', 'Inner above [dB]', self.inner_high),
            ('outer', 'Outer band', self.outer_box),
            ('outer_low', 'Outer below [dB]', self.outer_low),
            ('outer_high', 'Outer above [dB]', self.outer_high),
        ])
        add_rule(grid, 9)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, (('points', 'Points'),), 10).items()}
        # one line per polarity, as many as are selected, so a label
        # that wraps rather than a fixed set of rows
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 11, 0, 1, 2)
        grid.setRowStretch(12, 1)

        self.rule_box.currentIndexChanged.connect(self._edited)
        self.limits_box.toggled.connect(self._edited)
        self.outer_box.toggled.connect(self._edited)
        for box in (self.percent_box, self.inner_low, self.inner_high,
                    self.outer_low, self.outer_high):
            box.valueChanged.connect(self._edited)
        self._shown_rows()

    def _from_limits(self) -> tuple[bool, bool]:
        """Whether the inner and the outer band are the requirement's
        limits rather than the fields."""
        use = self.limits_box.isChecked()
        return use and self._has_limits[0], use and self._has_limits[1]

    def _shown_rows(self) -> None:
        """Show only the rows that apply (principle 3): the share under
        the share rule, the limits choice where the requirement has
        limits, a band's fields where its limits do not stand for it,
        the inner band's upper edge where the rule has one, the outer
        band's edges while it is ticked."""
        share = self.rule() == 'share'
        inner, outer_limits = self._from_limits()
        outer = self.outer_box.isChecked() and not outer_limits
        for key, shown in (('percent', share),
                           ('limits', any(self._has_limits)),
                           ('inner_low', not inner),
                           ('inner_high', share and not inner),
                           ('outer', not outer_limits),
                           ('outer_low', outer), ('outer_high', outer)):
            self.names[key].setVisible(shown)
            self.editors[key].setVisible(shown)

    def _edited(self, *_args: Any) -> None:
        self._shown_rows()
        if not self._loading:
            self.changed.emit(self.settings())

    def rule(self) -> str:
        """The rule chosen, one of `SRS_RULES`."""
        return str(self.rule_box.currentData())

    def settings(self) -> dict[str, Any]:
        """The judgement's arguments as `srs_tolerance` takes them:
        'inner', 'outer' (None when unticked, or when the requirement's
        limits stand for the band), 'rule', 'percent' and 'limits'."""
        inner, outer = self._from_limits()
        return {'inner': (None if inner else
                          (self.inner_low.value(), self.inner_high.value())),
                'outer': ((self.outer_low.value(), self.outer_high.value())
                          if self.outer_box.isChecked() and not outer
                          else None),
                'rule': self.rule(), 'percent': self.percent_box.value(),
                'limits': inner or outer}

    def offer_limits(self, key: Any, inner: Any, outer: Any) -> None:
        """Offer a requirement's own limits as the bands, without a
        redraw: `key` names the requirement and record shown, and
        `inner` and `outer` are its warning and abort bands as
        `compliance.srs_limit_bands` gives them, None where it has
        none. A requirement new to the panel ticks the choice, so it is
        judged against the curves its comparison shades, and a band
        that is a fixed number of dB seeds its fields, so unticking
        starts from the requirement's numbers. The same key offered
        again keeps what was set."""
        if key == self._offered_for:
            return
        self._offered_for = key
        self._has_limits = (inner is not None, outer is not None)
        self._loading = True
        try:
            self.limits_box.setChecked(any(self._has_limits))
            for band, low, high in ((inner, self.inner_low, self.inner_high),
                                    (outer, self.outer_low, self.outer_high)):
                if band is not None and not any(np.ndim(v) for v in band):
                    low.setValue(band[0])
                    high.setValue(band[1])
            if outer is not None:
                self.outer_box.setChecked(True)
        finally:
            self._loading = False
        self._shown_rows()

    def set_settings(self, inner: tuple[float, float] | None = None,
                     outer: tuple[float, float] | None = None,
                     rule: str | None = None,
                     percent: float | None = None,
                     limits: bool | None = None) -> None:
        """Set the fields without a redraw per field, then emit once."""
        self._loading = True
        try:
            if limits is not None:
                self.limits_box.setChecked(limits)
            if rule is not None:
                self.rule_box.setCurrentIndex(
                    [key for key, _name in RULE_NAMES].index(rule))
            if percent is not None:
                self.percent_box.setValue(percent)
            if inner is not None:
                self.inner_low.setValue(inner[0])
                self.inner_high.setValue(inner[1])
            self.outer_box.setChecked(outer is not None)
            if outer is not None:
                self.outer_low.setValue(outer[0])
                self.outer_high.setValue(outer[1])
        finally:
            self._loading = False
        self._edited()

    def show_results(self, results: Sequence[dict[str, Any]],
                     names: Sequence[str] | None = None) -> None:
        """Write each judged spectrum's line: the count and share inside
        and the verdict (`plot.srs_tolerance.describe`), then its worst
        point. `names` labels each spectrum when its polarity alone
        would not tell two apart. Empty `results` clears the readout."""
        from ..plot.srs_tolerance import describe

        if not results:
            self.derived['points'].setText('—')
            self.readout.setText('')
            return
        self.derived['points'].setText(str(results[0]['points']))
        lines = []
        for k, result in enumerate(results):
            line = describe(result)
            if names is not None:
                line = f'{names[k]} {line}'
            if result['points']:
                line += (f'\nworst {result["worst_db"]:+.1f} dB at '
                         f'{result["worst_frequency"]:.4g} Hz'
                         ).replace('-', '−')
            lines.append(line)
        self.readout.setText('\n'.join(lines))
