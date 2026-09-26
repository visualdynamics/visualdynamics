"""The averaging parameters, as a table beside the time history.

Five numbers are set here — start, frame length, overlap, window, frame
count — and four more are shown that follow from them: where the
analysis ends, how fine the frequency resolution will be, how long each
frame lasts, and how many averages the PSD will actually be built from.
The derived four are never edited, because a value that can be typed in
two places disagrees with itself by the second edit.

The panel says what the parameters are, reports when they move, and
computes with its own buttons (Compute PSDs, CPSDs, FRFs, Multiple
Coherence, Spectra) — each calling the project verb a script calls,
with whatever is set at that moment (principle 13).

A capture the controller already cut into frames arrives with its frame
length and count settled by the file. The panel shows them, grayed, and
leaves only the window live — there is no second frame inside a record
to overlap with, and nowhere for the start to go.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.averaging import DETRENDS, WINDOW_PARAMETERS, WINDOWS, Averaging
from .editors import DoubleSpinBox, SpinBox, commit_on_enter
from .settings_panel import add_derived, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import TimeHistory


class AveragingPanel(QWidget):
    """The parameter table. Edits arrive as a whole `Averaging`."""

    #: a parameter moved; here is the averaging that describes it now
    changed = Signal(object)

    #: the user asked for the spectra these frames parameterize —
    #: the act lives where its settings are set (Brandon, 2026-08-28)
    psds_asked = Signal()
    cpsds_asked = Signal()
    spectra_asked = Signal()
    frfs_asked = Signal()
    coherence_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sample_rate: float = 1.0
        self.samples: int = 0
        #: the record's first instant, what the start box counts from
        self.origin: float = 0.0
        self.records: int = 1
        #: the file already cut this record into one frame per
        #: average, so there is no stream to search for the test —
        #: every other parameter is still the user's to choose
        self.precut: bool = False
        #: the record the parameters describe, for working them out again
        self._history = None
        #: which window the parameter row is currently fitted to
        self._was_window: str | None = None
        #: set while the panel is writing to its own editors, so that
        #: restating a clamped value does not read as a fresh edit
        self._loading = False

        self.title: QLabel = QLabel('Averaging')
        grid = panel_grid(self, self.title)

        self.start_box: DoubleSpinBox = DoubleSpinBox()
        self.start_box.setDecimals(4)
        self.start_box.setSuffix(' s')
        self.start_box.setToolTip(
            'Where the analysis begins, on the record’s own clock')

        self.length_box: SpinBox = SpinBox()
        self.length_box.setRange(2, 2 ** 24)
        self.length_box.setToolTip(
            'Samples in each frame. The frequency resolution follows '
            'from it: a longer frame is a finer one')

        self.overlap_box: DoubleSpinBox = DoubleSpinBox()
        self.overlap_box.setRange(0.0, 99.0)
        self.overlap_box.setDecimals(1)
        self.overlap_box.setSuffix(' %')
        self.overlap_box.setToolTip(
            'How much of a frame the next one repeats')

        self.window_box: QComboBox = QComboBox()
        for name in WINDOWS:
            self.window_box.addItem(name.capitalize(), name)
        self.window_box.setToolTip(
            'The shape each frame is multiplied by before its FFT')

        # the one box a parameterized window brings with it — tukey's
        # alpha, kaiser's beta — shown only while such a window is
        # chosen (principle 3), its label restated to the parameter's
        # own name so the row never needs a second glance
        self.parameter_box: DoubleSpinBox = DoubleSpinBox()
        self.parameter_box.setDecimals(2)
        self.parameter_box.setToolTip(
            'The window’s own number. Tukey’s alpha is the tapered '
            'fraction — 0 is a boxcar, 1 is a hann; Kaiser’s beta '
            'trades main-lobe width for sidelobes, 14 sitting in the '
            'blackman-harris class')

        self.detrend_box: QComboBox = QComboBox()
        for key in DETRENDS:
            self.detrend_box.addItem(key.capitalize(), key)
        self.detrend_box.setToolTip(
            'How each frame is leveled before its window: nothing '
            '(the convention here and in sdynpy), its own mean '
            'removed (what scipy’s welch does by default), or a '
            'least-squares line taken out — for a record whose drift '
            'the frames should not carry into the low bins')

        self.frames_box: SpinBox = SpinBox()
        self.frames_box.setRange(1, 2 ** 20)
        self.frames_box.setToolTip(
            'Frames averaged per record. Drag the edge of the shaded '
            'region on the plot to change it there')

        rows = (('Start', self.start_box),
                ('Frame length', self.length_box),
                ('Overlap', self.overlap_box),
                ('Window', self.window_box),
                ('Alpha', self.parameter_box),
                ('Detrend', self.detrend_box),
                ('Frames', self.frames_box))
        for row, (label, editor) in enumerate(rows, start=1):
            name = QLabel(label)
            grid.addWidget(name, row, 0)
            grid.addWidget(editor, row, 1)
            if editor is self.parameter_box:
                self._parameter_label = name

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(rule, len(rows) + 1, 0, 1, 2)

        # what follows from the five, shown so the trade is visible while
        # it is being made rather than after the PSD comes out
        self.derived: dict[str, QLabel] = {}
        derived = (('averages', 'Averages'), ('resolution', 'Δf'),
                   ('duration', 'Frame'), ('stop', 'Ends'))
        for key, (_name, value) in add_derived(grid, derived,
                                               len(rows) + 2).items():
            self.derived[key] = value

        self.detect_button: QPushButton = QPushButton('Detect')
        self.detect_button.setToolTip(
            'Work the parameters out from the record: find the settled '
            'stretch at the highest level it holds for long enough, and '
            'take as many frames as it will carry')
        grid.addWidget(self.detect_button, len(rows) + 2 + len(derived),
                       0, 1, 2)

        # the acts these frames parameterize, right where they are
        # set — deciding the framing and computing from it are one
        # intent, not a trip through the tree (Brandon, 2026-08-28)
        self.psds_button: QPushButton = QPushButton('Compute PSDs')
        self.psds_button.setToolTip(
            'Auto-power spectral density per channel, averaged over '
            'exactly these frames')
        self.psds_button.clicked.connect(self.psds_asked.emit)
        grid.addWidget(self.psds_button, len(rows) + 3 + len(derived),
                       0, 1, 2)
        self.cpsds_button: QPushButton = QPushButton('Compute CPSDs')
        self.cpsds_button.setToolTip(
            'The full cross-power matrix, phase kept, over these '
            'same frames')
        self.cpsds_button.clicked.connect(self.cpsds_asked.emit)
        grid.addWidget(self.cpsds_button, len(rows) + 4 + len(derived),
                       0, 1, 2)
        self.frfs_button: QPushButton = QPushButton('Compute FRFs…')
        self.frfs_button.setToolTip(
            'The transfer functions from the excitation channels, over '
            'these frames — the ellipsis asks which estimator first. '
            'Offered when the history holds excitation channels')
        self.frfs_button.clicked.connect(self.frfs_asked.emit)
        grid.addWidget(self.frfs_button, len(rows) + 5 + len(derived),
                       0, 1, 2)
        self.coherence_button: QPushButton = QPushButton(
            'Multiple Coherence')
        self.coherence_button.setToolTip(
            'How much of each response all the excitation channels '
            'together account for, line by line, over these same '
            'frames. Offered when the history holds excitation '
            'channels')
        self.coherence_button.clicked.connect(self.coherence_asked.emit)
        grid.addWidget(self.coherence_button,
                       len(rows) + 6 + len(derived), 0, 1, 2)
        self.spectra_button: QPushButton = QPushButton('Compute Spectra')
        self.spectra_button.setToolTip(
            'Averaged linear spectra per channel, over these frames')
        self.spectra_button.clicked.connect(self.spectra_asked.emit)
        grid.addWidget(self.spectra_button,
                       len(rows) + 7 + len(derived), 0, 1, 2)

        self.note: QLabel = QLabel()
        self.note.setWordWrap(True)
        self.note.setEnabled(False)
        grid.addWidget(self.note, len(rows) + 8 + len(derived), 0, 1, 2)
        grid.setRowStretch(len(rows) + 9 + len(derived), 1)

        commit_on_enter(self.start_box, self.overlap_box,
                        self.length_box, self.frames_box,
                        self.parameter_box)
        for editor in (self.start_box, self.overlap_box):
            editor.valueChanged.connect(self._edited)
        for editor in (self.length_box, self.frames_box):
            editor.valueChanged.connect(self._edited)
        self.window_box.currentIndexChanged.connect(self._edited)
        self.detrend_box.currentIndexChanged.connect(self._edited)
        self.parameter_box.valueChanged.connect(self._edited)
        self.detect_button.clicked.connect(self._detect)

    # ---- worked out from the record ---------------------------------------

    def _detect(self):
        """Take the parameters the record itself implies.

        Reported like any other edit, so it lands on the object and the
        shading follows — the button fills the table in, it does not
        compute anything.
        """
        if self._history is None:
            return
        try:
            found = self._history.suggest_averaging(
                window=self.chosen_window(), overlap=self.overlap())
        except ValueError:
            # unevenly sampled, or too short to frame at all
            return
        self.set_averaging(found)
        self.changed.emit(found)

    def chosen_window(self) -> str:
        """Which window shapes each frame.

        Not `window()`: that is Qt's own verb on every widget, for the
        top-level window this one sits in, and taking the name would
        hand a string to the next caller that meant Qt's.
        """
        return self.window_box.currentData()

    def overlap(self) -> float:
        return self.overlap_box.value() / 100.0

    # ---- what the panel is describing -------------------------------------

    def show_history(self, history: TimeHistory,
                     averaging: Averaging) -> None:
        """Point the panel at a time history and the averaging set on it.

        The record's length and rate are what every clamp is against, so
        they come from the history rather than being guessed from the
        numbers already in the boxes.
        """
        self._history = history
        self.sample_rate = history.sample_rate
        self.samples = len(history.abscissa)
        # the box speaks the record's clock (a truncation keeps its
        # instants; an import window reads the run's); the averaging
        # counts from the record's beginning — `Averaging.stop`
        self.origin = float(history.abscissa[0]) if self.samples else 0.0
        self.precut = history.split_into_frames
        self.records = max(history.records_per_channel.values(), default=1)
        # only what applies (principle 3): the reference-driven
        # estimates need excitation channels, and a button that can
        # only refuse is a question, not a control
        driven = (bool(history.reference_channels())
                  and bool(history.response_channels()))
        self.frfs_button.setVisible(driven)
        self.coherence_button.setVisible(driven)
        self.set_averaging(averaging)

    def set_averaging(self, averaging: Averaging) -> None:
        """Restate the panel from an averaging — what a drag arrives as."""
        self._loading = True
        try:
            # the limits first: a spin box clamps to the range it has at
            # the moment it is written, so setting the values against the
            # *previous* averaging's limits silently truncates them — a
            # detected 57 frames came out as the 22 the last frame length
            # allowed
            self._apply_limits(averaging)
            self.start_box.setValue(self.origin + averaging.start)
            self.length_box.setValue(averaging.frame_length)
            self.overlap_box.setValue(averaging.overlap * 100.0)
            self.frames_box.setValue(averaging.frames)
            at = self.detrend_box.findData(averaging.detrend)
            if at >= 0:
                self.detrend_box.setCurrentIndex(at)
            index = self.window_box.findData(averaging.window)
            if index >= 0:
                self.window_box.setCurrentIndex(index)
            self._restate_parameter()
            self._was_window = averaging.window
            if averaging.window_parameter is not None:
                self.parameter_box.setValue(averaging.window_parameter)
        finally:
            self._loading = False
        self._restate(averaging)

    def averaging(self) -> Averaging:
        """The averaging the editors currently describe, unclamped."""
        offered = self.window_box.currentData() in WINDOW_PARAMETERS
        return Averaging(frame_length=self.length_box.value(),
                         overlap=self.overlap_box.value() / 100.0,
                         window=self.window_box.currentData(),
                         window_parameter=(self.parameter_box.value()
                                           if offered else None),
                         detrend=self.detrend_box.currentData(),
                         frames=self.frames_box.value(),
                         start=max(self.start_box.value() - self.origin, 0.0))

    # ---- editing ----------------------------------------------------------

    def _restate_parameter(self, adopt_default: bool = False) -> None:
        """Fit the parameter row to the chosen window: the label takes
        the parameter's own name, the range its documented span, and
        the row is absent entirely for the windows that take none
        (principle 3). `adopt_default` seeds the documented value —
        for the moment a parameterized window is *chosen*, never for a
        restate of one already set."""
        settings = WINDOW_PARAMETERS.get(self.window_box.currentData())
        offered = settings is not None
        self._parameter_label.setVisible(offered)
        self.parameter_box.setVisible(offered)
        if not offered:
            return
        what, default, (lowest, highest) = settings
        self._parameter_label.setText(what.capitalize())
        was = self._loading
        self._loading = True
        try:
            self.parameter_box.setRange(lowest, highest)
            self.parameter_box.setSingleStep((highest - lowest) / 20.0)
            if adopt_default:
                self.parameter_box.setValue(default)
        finally:
            self._loading = was

    def _edited(self, *_args):
        """A parameter moved: clamp it to the record and report it."""
        if self._loading:
            return
        # a window switch re-fits its parameter row first, so the
        # averaging built below carries the right number
        switched = self.window_box.currentData() != self._was_window
        if switched:
            self._restate_parameter(adopt_default=True)
            self._was_window = self.window_box.currentData()
        settled = self._clamped(self.averaging())
        if settled != self.averaging():
            self.set_averaging(settled)
        else:
            self._restate(settled)
        self.changed.emit(settled)

    def _clamped(self, averaging):
        """The nearest averaging that fits inside the record.

        Lengthening a frame with the count already at what the record
        held asks for more record than there is; rather than refuse the
        edit, the count comes down to what the new frame length allows.
        """
        if not self.samples:
            return averaging
        rate = self.sample_rate
        last_start = max(self.samples - averaging.frame_length, 0) / rate
        settled = replace(averaging, start=min(averaging.start, last_start))
        room = settled.most_frames(self.samples, rate)
        return replace(settled, frames=max(min(settled.frames, room), 1))

    def _apply_limits(self, averaging):
        """Keep the spin boxes inside what the record can carry, so a
        held arrow button stops at the end rather than at a refusal."""
        rate = self.sample_rate or 1.0
        self.length_box.setMaximum(max(self.samples, 2))
        last_start = max(self.samples - averaging.frame_length, 0) / rate
        self.start_box.setRange(self.origin, self.origin + last_start)
        self.start_box.setSingleStep(averaging.frame_length / rate / 4.0)
        self.frames_box.setMaximum(
            max(averaging.most_frames(self.samples, rate), 1))


        # Every parameter stays the user's, on a pre-cut capture as
        # much as on a stream. The frames the controller wrote are what
        # it averaged, and they are the right default — but they are not
        # the only reading of the record. A burst-random capture is half
        # excitation and half ringdown, and analyzing the burst alone,
        # or putting an exponential window on the decay, is an ordinary
        # thing to want and used to be impossible here.
        #
        # Only detection goes: it searches a stream for the stretch that
        # is the test, and inside one already-cut frame there is no such
        # search to run.
        self.detect_button.setEnabled(not self.precut)

    # ---- what follows from the five ---------------------------------------

    def _restate(self, averaging):
        rate = self.sample_rate or 1.0
        self.derived['resolution'].setText(
            f'{rate / averaging.frame_length:.4g} Hz')
        # the arithmetic said, not just the product: how many
        # playings, cut how many ways, is the whole answer to "what
        # am I averaging" (Brandon, 2026-08-28)
        averages = averaging.frames * self.records
        self.derived['averages'].setText(
            f'{self.records} × {averaging.frames} = {averages}'
            if self.records > 1 else str(averages))
        self.derived['duration'].setText(
            f'{averaging.frame_length / rate:.4g} s')
        self.derived['stop'].setText(
            f'{averaging.stop(rate, self.origin):.4g} s')
        if self.precut:
            whole = averaging.frame_length == self.samples and not averaging.start
            self.note.setText(
                f'The file saved each average as its own record: '
                f'{self.records} records of {self.samples} samples'
                + ('. Shorten the frame or move its start to analyze '
                   'part of each one.' if whole else
                   f', read {averaging.frame_length} samples at a time.'))
        elif self.records > 1:
            self.note.setText(
                f'{averaging.frames} frames from each of {self.records} '
                f'records, pooled.')
        else:
            self.note.setText('')
