"""The modes a sine sweep is checked against, beside the sweep drawn
through them.

The Sweep Rate reading of a sine specification draws one tone's sweep
as frequency against time with each mode's half-power band across it
(`plot.sweep_rate`), and lists every pass in the table under the plot
(`core.sweep_rate.check_sweep_rate`). This panel holds what the check
is made against: the modes, each a natural frequency and a Q, typed as
pairs, and the multiple of the rule's rate at which a pass is flagged.
Under them are the counts that follow, and, where a tracked level of
the tone is selected with the specification, the judgment of the Q
read off that level at each mode (`core.sweep_rate.read_swept_q`).

Every finished edit redraws the plot and the table. Nothing is made:
the check is a reading of a specification that already exists, as the
points rule is a reading of a shock spectrum, so there is no Apply
button. A script gets the same numbers from `check_sweep_rate`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QLineEdit, QWidget

from ..core.sweep_rate import MARGIN
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the modes the panel opens with: none, so nothing is judged until a
#: mode is given; a guessed mode would be a verdict on an article that
#: does not exist
MODES: tuple[tuple[float, float], ...] = ()


def parse_modes(text: str) -> list[tuple[float, float]]:
    """Read '22 40, 180 20' as [(22.0, 40.0), (180.0, 20.0)]: pairs of
    natural frequency in Hz and Q, the pairs separated by commas or
    semicolons and the two numbers of a pair by spaces or a slash.

    Raises ValueError, naming the pair, for a pair that is not two
    numbers, a frequency that is not positive or a Q at or under 0.5
    (critical damping, where there is no band to sweep through)."""
    modes = []
    for piece in re.split(r'[;,]', text):
        piece = piece.strip()
        if not piece:
            continue
        numbers = [part for part in re.split(r'[\s/]+', piece) if part]
        try:
            frequency, q = (float(number) for number in numbers)
        except ValueError:
            raise ValueError(f'{piece!r} is not a frequency and a Q, '
                             'such as 22 40') from None
        if not frequency > 0.0:
            raise ValueError(f'{piece!r}: the frequency must be positive')
        if not q > 0.5:
            raise ValueError(f'{piece!r}: a Q of 0.5 or less is critically '
                             'damped and has no band to sweep through')
        modes.append((frequency, q))
    return modes


def modes_text(modes: Sequence[tuple[float, float]]) -> str:
    """The modes as the field writes them: '22 40, 180 20'."""
    return ', '.join(f'{frequency:g} {q:g}' for frequency, q in modes)


class SweepRatePanel(QWidget):
    """The modes and the margin, and the counts. Edits are emitted as
    `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Sweep Rate')
        grid = panel_grid(self, self.title)
        self._loading = False
        self._modes: list[tuple[float, float]] = list(MODES)

        self.modes_box: QLineEdit = QLineEdit(modes_text(self._modes))
        self.modes_box.setPlaceholderText('22 40, 180 20')
        self.modes_box.setToolTip(
            'The modes to check the sweep against: each a natural '
            'frequency in Hz and its Q, as pairs separated by commas, '
            'such as 22 40, 180 20')
        self.margin_box: DoubleSpinBox = DoubleSpinBox()
        self.margin_box.setRange(0.05, 100.0)
        self.margin_box.setDecimals(2)
        self.margin_box.setSingleStep(0.25)
        self.margin_box.setValue(MARGIN)
        self.margin_box.setToolTip(
            'Flag a pass whose rate is above this multiple of the '
            'rule\'s rate, 87 fn / Q² oct/min on a log sweep or '
            'fn² / Q² Hz/s on a linear one. At 1 a mode reads about '
            '94 percent of its steady height')
        commit_on_enter(self.margin_box)
        self.names, self.editors = add_keyed_settings(grid, [
            ('modes', 'Modes [Hz Q]', self.modes_box),
            ('margin', 'Flag above [× rule]', self.margin_box),
        ])
        add_rule(grid, 3)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, (('passes', 'Passes'), ('fast', 'Too fast')),
                4).items()}
        # one line per pass, and per Q read off a level, as many as there
        # are, so a label that wraps rather than a fixed set of rows
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 6, 0, 1, 2)
        grid.setRowStretch(7, 1)

        self.modes_box.editingFinished.connect(self._modes_edited)
        self.margin_box.valueChanged.connect(self._edited)

    def _modes_edited(self) -> None:
        """The field was left: take the modes if they read, or put the
        last good ones back and say why (principle 3: refused when
        entered, not reported afterwards)."""
        try:
            modes = parse_modes(self.modes_box.text())
        except ValueError as error:
            self.modes_box.setText(modes_text(self._modes))
            self.readout.setText(str(error))
            return
        if modes == self._modes:
            return
        self._modes = modes
        self._edited()

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """The check's arguments as `check_sweep_rate` takes them:
        'modes', a list of (frequency, Q), and 'margin'."""
        return {'modes': list(self._modes),
                'margin': self.margin_box.value()}

    def set_settings(self, modes: Sequence[tuple[float, float]] | None = None,
                     margin: float | None = None) -> None:
        """Set the fields without a redraw per field, then emit once."""
        self._loading = True
        try:
            if modes is not None:
                self._modes = [(float(f), float(q)) for f, q in modes]
                self.modes_box.setText(modes_text(self._modes))
            if margin is not None:
                self.margin_box.setValue(margin)
        finally:
            self._loading = False
        self._edited()

    def show_results(self, checks: Sequence[Any],
                     swept: Sequence[Any] = ()) -> None:
        """Write the counts, a short line per pass, and a line per Q
        read off a level (`SweptQ.describe`). With no modes given, say
        how to give them."""
        if not self._modes:
            self.derived['passes'].setText('—')
            self.derived['fast'].setText('—')
            self.readout.setText('Type the modes above, a frequency in '
                                 'Hz and a Q for each')
            return
        self.derived['passes'].setText(str(len(checks)))
        self.derived['fast'].setText(
            str(sum(check.too_fast for check in checks)))
        lines = [f'{check.frequency:.4g} Hz, Q {check.q:.3g}, '
                 f'{check.direction}: {check.ratio:.2g} × rule, '
                 f'{check.verdict}' for check in checks]
        if not checks:
            lines.append('No mode lies inside this tone\'s sweep')
        lines += [reading.describe() for reading in swept]
        self.readout.setText('\n'.join(lines))
