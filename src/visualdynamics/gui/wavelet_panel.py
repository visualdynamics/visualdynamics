"""The settings the scalogram is computed with, as a table beside it.

Four values are set here: the bottom and top of the frequency range,
the lines per octave, and the number of cycles under the wavelet, which
sets how it trades time resolution against frequency resolution. Three
more are shown that follow from them: the width of the cone of
influence at the bottom of the range, the frequency resolution there,
and the number of lines.

The record is not chosen here. The scalogram uses the record selected
in the project tree, as every other view does, and this panel holds
only the settings the tree cannot.

The derived values are shown because the scalogram is the one reading
in this toolset where the settings decide whether most of the picture
is artifact. Ask for 1 Hz on a two-second record and the bottom of the
axis is all cone of influence, and the picture still looks plausible
while it says nothing about the measurement. So the panel says so before
the transform runs, with a note under the table when the cone at the
lowest frequency covers a third of the record or more.

Nothing here computes the transform. The panel shows the settings and
reports when they change, as the averaging panel does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
)

from ..core import wavelet
from .editors import DoubleSpinBox, SpinBox, commit_on_enter
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import TimeHistory

# The record was briefly chosen here, in a combo box, which put the same
# choice in two places. The tree is the one selection everywhere
# (Brandon, 2026-08-29).

#: The default frequency range lives in `core.wavelet.default_range`
#: now — one implementation for this panel, `plot_scalogram` and the
#: report, after the three drifted within a day (the panel's top moved
#: from 0.4 of Nyquist to 0.98 when the 0.4 was measured to be
#: superstition, and the scripting plot kept the old number). The
#: measurement and the reasoning ride with the function.


class WaveletPanel(QWidget):
    """The scalogram's settings. Each edit is reported on `changed` as
    the whole settings dict: 'low' and 'high' in Hz, 'per_octave' and
    'omega0'."""

    #: a parameter moved; here is what the scalogram should be now
    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sample_rate: float = 1.0
        self.samples: int = 0
        self.duration: float = 0.0
        #: set while the panel writes to its own editors, so restating a
        #: clamped value does not read as a fresh edit
        self._loading = False

        self.title: QLabel = QLabel('Wavelet')
        grid = panel_grid(self, self.title)

        self.low_box: DoubleSpinBox = DoubleSpinBox()
        self.low_box.setDecimals(2)
        self.low_box.setSuffix(' Hz')
        self.low_box.setToolTip(
            'The bottom of the frequency axis. Low frequencies use long '
            'wavelets, so this is what decides how much of the picture '
            'is cone of influence')

        self.high_box: DoubleSpinBox = DoubleSpinBox()
        self.high_box.setDecimals(2)
        self.high_box.setSuffix(' Hz')
        self.high_box.setToolTip(
            'The top of the frequency axis. Defaults to everything the '
            'record carries; the reading stays true to a fraction of a '
            'percent right up against Nyquist')

        self.per_octave_box: SpinBox = SpinBox()
        self.per_octave_box.setRange(1, 96)
        self.per_octave_box.setToolTip(
            'Lines per octave. The axis is logarithmic because a '
            "wavelet's bandwidth is a constant fraction of its "
            'frequency — evenly spaced lines would crowd at the top')

        self.omega_box: DoubleSpinBox = DoubleSpinBox()
        self.omega_box.setDecimals(1)
        self.omega_box.setRange(3.0, 30.0)
        self.omega_box.setSingleStep(1.0)
        self.omega_box.setToolTip(
            'Cycles under the wavelet, which is the trade itself: low '
            'resolves when and blurs what, high resolves what and '
            'blurs when. Six is conventional, and near the lowest value '
            'for which the transform is admissible at all')

        rows = (('From', self.low_box),
                ('To', self.high_box),
                ('Per octave', self.per_octave_box),
                ('Cycles', self.omega_box))
        add_settings(grid, rows)

        add_rule(grid, len(rows) + 1)

        self.derived: dict[str, QLabel] = {}
        derived = (('cone', 'Cone at bottom'), ('resolves', 'Resolves'),
                   ('transforms', 'Lines'))
        for key, (_name, value) in add_derived(grid, derived,
                                               len(rows) + 2).items():
            self.derived[key] = value

        self.note: QLabel = QLabel()
        self.note.setWordWrap(True)
        self.note.setEnabled(False)
        grid.addWidget(self.note, len(rows) + 2 + len(derived), 0, 1, 2)
        grid.setRowStretch(len(rows) + 3 + len(derived), 1)

        commit_on_enter(self.low_box, self.high_box, self.per_octave_box,
                        self.omega_box)
        for editor in (self.low_box, self.high_box, self.per_octave_box,
                       self.omega_box):
            editor.valueChanged.connect(self._edited)

    # ---- what the panel is describing -------------------------------------

    def show_history(self, history: TimeHistory,
                     settings: dict[str, Any] | None = None) -> None:
        """Show the settings for a time history.

        `settings` is the scalogram's settings on that history, and when
        it is None the panel opens on `suggested`. The limits every value
        is clamped to come from this history's sample rate, not from the
        values the boxes held for the last record, and the range is
        clamped as a pair, so it keeps at least an octave of width.
        """
        self.sample_rate = history.sample_rate
        self.samples = len(history.abscissa)
        self.duration = self.samples / self.sample_rate if self.sample_rate \
            else 0.0
        settled = dict(self.suggested()) if settings is None else dict(settings)
        # through the clamp, not straight into the boxes: a range tuned
        # on one record and carried to a slower one arrives above this
        # Nyquist, and each box clamping on its own would settle both
        # ends onto the same number — a range of no width, which is not
        # a range the derived rows can be computed from
        settled = self._clamped_against(settled, history.sample_rate)

        self._loading = True
        try:
            self._apply_limits()
            self.low_box.setValue(settled['low'])
            self.high_box.setValue(settled['high'])
            self.per_octave_box.setValue(settled['per_octave'])
            self.omega_box.setValue(settled['omega0'])
        finally:
            self._loading = False
        self._restate(self.settings())

    def suggested(self) -> dict[str, Any]:
        """The default settings for this record.

        The range is `core.wavelet.default_range`: the top at 0.98 of
        Nyquist, and the bottom at eight cycles over the record's
        duration, so the longest wavelets still fit inside the record and
        the bottom of the range is not mostly cone of influence. Lines per
        octave and cycles are the defaults `core.wavelet.PER_OCTAVE` and
        `core.wavelet.OMEGA0`.
        """
        low, high = wavelet.default_range(self.sample_rate or 1.0,
                                          self.duration)
        return {'low': low, 'high': high,
                'per_octave': wavelet.PER_OCTAVE, 'omega0': wavelet.OMEGA0}

    def settings(self) -> dict[str, Any]:
        """The settings the editors show now, as a dict: 'low' and 'high'
        in Hz, 'per_octave' and 'omega0'."""
        return {'low': self.low_box.value(),
                'high': self.high_box.value(),
                'per_octave': self.per_octave_box.value(),
                'omega0': self.omega_box.value()}

    def _apply_limits(self) -> None:
        """Keep the boxes inside what the record can carry."""
        nyquist = (self.sample_rate or 1.0) / 2.0
        floor = nyquist / 10000.0
        self.low_box.setRange(floor, nyquist * 0.98)
        self.high_box.setRange(floor, nyquist * 0.98)
        self.low_box.setSingleStep(max(nyquist / 200.0, 0.01))
        self.high_box.setSingleStep(max(nyquist / 200.0, 0.01))

    # ---- editing ----------------------------------------------------------

    def _edited(self, *_args) -> None:
        if self._loading:
            return
        settled = self._clamped(self.settings())
        if settled != self.settings():
            self._loading = True
            try:
                self.low_box.setValue(settled['low'])
                self.high_box.setValue(settled['high'])
            finally:
                self._loading = False
        self._restate(settled)
        self.changed.emit(settled)

    def _clamped(self, settings: dict[str, Any]) -> dict[str, Any]:
        """The nearest settings the record can actually be read with.

        A range that runs backwards is the one a person makes by typing
        the top into the bottom box, and refusing the edit would leave
        them stuck with a number they cannot correct in either order.
        The top gives way, because it is the one with room.
        """
        return self._clamped_against(settings, self.sample_rate or 1.0)

    def _clamped_against(self, settings: dict[str, Any],
                         sample_rate: float) -> dict[str, Any]:
        """The same, against a rate given rather than the one held —
        which is what `show_history` needs, before the rate is its own.

        The range keeps a width: an octave at least, because a bottom
        clamped onto its own top is a range nothing can be computed
        from, and the derived rows raised rather than reading blank the
        first time that happened.
        """
        settled = dict(settings)
        nyquist = (sample_rate or 1.0) / 2.0
        settled['high'] = min(max(settled['high'], nyquist / 1000.0),
                              nyquist * 0.98)
        settled['low'] = min(max(settled['low'], nyquist / 10000.0),
                             settled['high'] / 2.0)
        return settled

    def _restate(self, settings: dict[str, Any]) -> None:
        """What follows from the four, including what will be artifact."""
        rate = self.sample_rate or 1.0
        low = settings['low']
        cone = float(wavelet.cone_of_influence([low], rate,
                                               settings['omega0'])[0])
        self.derived['cone'].setText(f'{cone:.3g} s')
        self.derived['resolves'].setText(
            f'{low / settings["omega0"] * 2:.3g} Hz')
        lines = len(wavelet.log_frequencies(low, settings['high'],
                                            settings['per_octave']))
        self.derived['transforms'].setText(str(lines))

        # the warning that earns the panel its place: a range whose
        # cone eats the record is a picture of where the record was
        # cut, and it looks exactly like a picture of the record
        if self.duration and 2 * cone >= self.duration:
            self.note.setText(
                f'The cone at {low:.3g} Hz covers the whole record: '
                'nothing at the bottom of this range is measurement. '
                'Raise the bottom, or read a longer stretch.')
        elif self.duration and 2 * cone >= self.duration / 3.0:
            self.note.setText(
                f'The bottom {2 * cone / self.duration:.0%} of this '
                'record is inside the cone at the lowest frequency.')
        else:
            self.note.setText('')
