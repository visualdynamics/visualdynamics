"""The specification being edited, as tables beside the plot.

The panel lays out what an author states. One grid holds the
breakpoints, with a level per channel, and another holds each pair's
coherence and phase, with a row under it that sets every pair at once.
Above them are the form (Breakpoints, or Interpolated onto frequency
lines at a spacing), the two band constraints, Symmetric and Uniform,
and how the levels are entered: Levels, a level at every breakpoint,
or Slopes, a level at the first breakpoint and a slope in decibels per
octave into each one after it, with the level the slope reaches shown
beside it. Under Slopes an edit holds every slope it does not change,
as the draft's `with_slope` and `with_level` do. Areas and running rms
adds the reading of the breakpoint table under the grid (each
segment's slope, area, share and the rms accumulated to its end, for
the channel on the plot), and the window writes the areas on the plot
and draws the running rms under it. The bands themselves are dragged on the plot beside the panel,
which draws the autospectra with their bands. The same panel is used
for every way in: a shape set's modal coordinates, a channel table's
control channels, or a specification opened to edit.

The two grids are the application's ordinary tables,
`CopyPasteTableView` over `TableModel` columns. So a column header
selects the column, a row header the row, Cmd-click adds a cell, and
copy, paste, Delete to clear, Batch Edit and the fill handle come from
the one implementation every table uses. A frequency typed between two
others re-sorts the breakpoints instead of being refused.

Nothing here computes anything. The panel hands back a
`SpecificationDraft` in SI and reports when it changes, and the window
applies every edit to the specification through
`Project.author_specification`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ..core.author import SpecificationDraft
from ..core.breakpoints import (
    levels_from_slopes,
    levels_holding_slopes,
    levels_with_slope,
    segment_slopes,
)
from .editors import DoubleSpinBox
from .object_tables import _elsewhere
from .tables import Column, CopyPasteTableView, TableModel

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem

# One sheet for every way in (Brandon, 2026-09-04: a generic editor, not
# a modal one). The grids are the ordinary tables (PLAN.md, "How tables
# behave"; Brandon, 2026-09-06: every editable table should be the same
# table).

#: wider than the reading panels: a grid with a column per channel
#: needs the room, and scrolls sideways past six
PANEL_WIDTH = 440


class Sheet:
    """The draft's fields in SI, as the two grids edit them.

    The table models write here one cell at a time, and the panel reads
    a whole draft back. The display unit is applied when a cell is shown
    and undone when one is typed, so a cell nobody edited keeps its
    exact SI value.
    """

    def __init__(self, draft: SpecificationDraft,
                 unit_system: UnitSystem) -> None:
        self.channels: list[str] = list(draft.channels)
        self.dims: list[str] = list(draft.dims)
        self.densities: list[str] = [f'{dim}**2/frequency'
                                     for dim in draft.dims]
        self.frequencies: list[float] = list(draft.frequencies)
        self.levels: list[list[float]] = [list(row) for row in draft.levels]
        self.pairs: dict[tuple[int, int], tuple[float, float] | None] = dict(
            draft.pairs)
        self.pair_keys: list[tuple[int, int]] = draft.pairable()
        self.sources: list[str] = list(draft.sources)
        self.spanning: bool = len(set(draft.sources)) > 1
        self.bands: list[dict] = [dict(entry) for entry in draft.bands]
        self.notes: list[str] = list(draft.notes)
        self.form: str = draft.form
        self.spacing: float | None = draft.spacing
        #: how the levels are typed, the draft's own; slopes are offered
        #: on the breakpoints form only (`by_slope`)
        self.entry: str = draft.entry
        self.unit_system: UnitSystem = unit_system
        #: set by a frequency edit: the rows have moved and the view
        #: has to be restated rather than repainted
        self.reordered: bool = False
        #: set by an edit under slope entry that moved the levels of
        #: other rows (every slope after it held): the view is restated
        self.spread: bool = False

    @property
    def by_slope(self) -> bool:
        """Whether the levels are typed by slope here: the draft's entry
        is 'slopes' and the sheet is on its breakpoints. The lines form
        is a controller's reading of the breakpoints, and a slope into
        each of four thousand lines is nothing anyone types."""
        return self.entry == 'slopes' and self.form == 'breakpoints'

    def draft(self) -> SpecificationDraft:
        """The sheet as a `SpecificationDraft`. The draft's own checks
        raise ValueError for anything no specification could hold."""
        return SpecificationDraft(
            list(self.channels), list(self.dims), list(self.frequencies),
            [list(row) for row in self.levels], dict(self.pairs),
            self.bands, list(self.notes),
            list(self.sources), self.form, self.spacing, self.entry)

    def sort(self) -> None:
        """Put the breakpoints in frequency order, with their levels, so a
        frequency typed between two others moves to its place."""
        # Brandon, 2026-09-06: typed out of order, the sheet stalled.
        order = np.argsort(self.frequencies, kind='stable')
        self.frequencies = [self.frequencies[k] for k in order]
        self.levels = [[row[k] for k in order] for row in self.levels]

    def level_channel(self, column: int) -> int | None:
        """The channel whose level the breakpoints grid shows in
        `column`, or None for the frequency column and, by slope, a
        slope column (the grid then holds a level and a slope a
        channel)."""
        if column < 1:
            return None
        if not self.by_slope:
            return column - 1
        k, slope = divmod(column - 1, 2)
        return None if slope else k


def _set_frequency(sheet: Sheet, row: int, text: str) -> None:
    value = float(text)
    if not value > 0:
        raise ValueError('a breakpoint frequency is positive')
    others = [f for i, f in enumerate(sheet.frequencies) if i != row]
    if value in others:
        raise ValueError(f'there is already a breakpoint at {value:g} Hz')
    held = ([segment_slopes(sheet.frequencies, row_levels)
             for row_levels in sheet.levels] if sheet.by_slope else None)
    sheet.frequencies[row] = value
    before = list(sheet.frequencies)
    sheet.sort()
    sheet.reordered = sheet.frequencies != before
    if held is not None and not sheet.reordered:
        # by slope, a frequency moved within its neighbors keeps every
        # slope and the levels follow from the first; moved past a
        # neighbor, the segments themselves change and the levels go
        # with their rows, as they do when typed by level
        sheet.levels = [[float(v) for v in levels_from_slopes(
            sheet.frequencies, row_levels[0], slopes)]
            for row_levels, slopes in zip(sheet.levels, held)]
        sheet.spread = True


def _level_getter(k: int):
    def getter(sheet: Sheet, row: int) -> float:
        return float(sheet.unit_system.from_si(sheet.levels[k][row],
                                               sheet.densities[k]))
    return getter


def _level_setter(k: int):
    def setter(sheet: Sheet, row: int, text: str) -> None:
        value = float(text)
        if not value > 0:
            raise ValueError(f'{sheet.channels[k]} needs a positive level at '
                             f'{sheet.frequencies[row]:g} Hz')
        level = float(sheet.unit_system.to_si(value, sheet.densities[k]))
        if sheet.by_slope:
            # the slopes after it held, as `SpecificationDraft.with_level`
            # holds them under slope entry
            sheet.levels[k] = [float(v) for v in levels_holding_slopes(
                sheet.frequencies, sheet.levels[k], row, level)]
            sheet.spread = True
        else:
            sheet.levels[k][row] = level
    return setter


#: why the first breakpoint's slope cell takes nothing
NO_SLOPE_IN = ('the first breakpoint has no slope into it; type its level, '
               'and a slope into each breakpoint after it')


def _slope_getter(k: int):
    def getter(sheet: Sheet, row: int) -> float | None:
        if row == 0:
            return None
        return float(segment_slopes(sheet.frequencies[row - 1:row + 1],
                                    sheet.levels[k][row - 1:row + 1])[0])
    return getter


def _slope_setter(k: int):
    def setter(sheet: Sheet, row: int, text: str) -> None:
        if row == 0:
            raise ValueError(NO_SLOPE_IN)
        value = float(text)
        if not np.isfinite(value):
            raise ValueError('a slope is a finite number of decibels per '
                             'octave')
        # the level the slope reaches, every slope after it held, as
        # `SpecificationDraft.with_slope` computes it
        sheet.levels[k] = [float(v) for v in levels_with_slope(
            sheet.frequencies, sheet.levels[k], row - 1, value)]
        sheet.spread = True
    return setter


def _slope_text(value: Any) -> str:
    return '' if value is None else f'{value:.4g}'


def _pair_getter(which: int):
    def getter(sheet: Sheet, row: int) -> float | None:
        value = sheet.pairs.get(sheet.pair_keys[row])
        return None if value is None else value[which]
    return getter


def _pair_setter(which: int):
    def setter(sheet: Sheet, row: int, text: str) -> None:
        key = sheet.pair_keys[row]
        if not text.strip():
            sheet.pairs[key] = None        # unstated again
            return
        value = float(text)
        if which == 0 and not 0.0 <= value <= 1.0:
            raise ValueError(f'a coherence lies in [0, 1], not {value:g}')
        current = sheet.pairs.get(key) or (0.0, 0.0)
        sheet.pairs[key] = ((value, current[1]) if which == 0
                            else (current[0], value))
    return setter


def _pair_label(sheet: Sheet, row: int) -> str:
    i, j = sheet.pair_keys[row]
    return ((f'{sheet.sources[i]}: ' if sheet.spanning else '')
            + f'{sheet.channels[i]}–{sheet.channels[j]}')


def _number_text(value: Any) -> str:
    return '' if value is None else f'{value:g}'


def sheet_models(sheet: Sheet, parent: Any = None
                 ) -> tuple[TableModel, TableModel]:
    """(breakpoints, pairs): the sheet's two grids as ordinary table
    models, which the table conventions test holds to the same contract
    as every other table. The window records each edit in the journal
    through the verb it calls, so each editable column is declared as
    journaled elsewhere."""
    unit_system = sheet.unit_system
    points = [Column('Hz', lambda s, r: s.frequencies[r], set=_set_frequency,
                     format=lambda v: f'{v:g}', journal=_elsewhere)]
    for k, (name, density) in enumerate(zip(sheet.channels, sheet.densities)):
        source = f'{sheet.sources[k]}\n' if sheet.spanning else ''
        title = f'{source}{name}\n{unit_system.label_text(density)}'
        points.append(Column(title, _level_getter(k), set=_level_setter(k),
                             format=lambda v: f'{v:.6g}', journal=_elsewhere,
                             affects_row=sheet.by_slope))
        if sheet.by_slope:
            # by slope, each channel's level is shown beside the slope
            # into it: the level the slope reaches, which is what a
            # written table leaves to the reader
            points.append(Column(
                f'{source}{name}\ndB/oct', _slope_getter(k),
                set=_slope_setter(k), format=_slope_text,
                journal=_elsewhere, affects_row=True,
                does_not_apply=lambda s, r: NO_SLOPE_IN if r == 0 else ''))
    pairs = [Column('Pair', _pair_label),
             Column('Coherence', _pair_getter(0), set=_pair_setter(0),
                    format=_number_text, journal=_elsewhere),
             Column('Phase °', _pair_getter(1), set=_pair_setter(1),
                    format=_number_text, journal=_elsewhere)]
    return (TableModel(sheet, points, lambda s: len(s.frequencies), parent),
            TableModel(sheet, pairs, lambda s: len(s.pair_keys), parent))


class AuthorPanel(QWidget):
    """The specification editor: the breakpoints and pairs grids, the
    form buttons and the band constraints.

    Each edit is reported on `changed` as a whole `SpecificationDraft`.
    The form buttons report on `form_asked` and the constraints on
    `constraint_asked`, both for the whole specification. There is no
    apply button, because the window applies every edit to the
    specification as it is made.
    """

    #: the draft moved; here is what it says now
    changed = Signal(object)
    #: the form buttons: ('breakpoints', None) or ('lines', spacing),
    #: for the whole specification the sheet is open on
    form_asked = Signal(str, object)
    #: the constraints, ('symmetric' | 'uniform', on), for every channel
    #: of the whole specification — as the form is (Brandon, 2026-09-06)
    constraint_asked = Signal(str, bool)
    #: how the levels are typed, 'levels' or 'slopes', for the sheet's
    #: draft: nothing in the requirement changes, so nothing is written
    entry_asked = Signal(str)
    #: the reading of segment areas and running rms, on or off
    areas_asked = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._unit_system = None
        self._sheet: Sheet | None = None
        self._loading = False
        # a cell's edit is handed on a tick later, so a paste, a fill or
        # a Batch Edit is handed on once, after its last cell. Handed on
        # at once, the first cell's edit re-rendered the window, which
        # put fresh models on the grids, and the rest of a pasted block
        # went into the model just taken down: two cells pasted, one
        # landed (Brandon, 2026-10-30)
        self._settle: QTimer = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(0)
        self._settle.timeout.connect(self._edits_settled)

        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        # a minimum, not a fixed width: the sheet sits in a splitter
        # beside the plot and the user drags it as wide as the grids
        # need (Brandon, 2026-09-06)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(PANEL_WIDTH)

        self.title: QLabel = QLabel('Specification')
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        # the two forms of one requirement (Brandon, 2026-09-06): the
        # breakpoints it is written from, or the same read onto a
        # controller's frequency lines. Editing is done on the few
        # points; the lines are for the controller's file. Which is
        # showing is the button that is down; the other converts
        head = QHBoxLayout()
        head.addWidget(self.title)
        head.addStretch(1)
        self.form_group: QButtonGroup = QButtonGroup(self)
        self.form_group.setExclusive(True)
        self.breakpoints_button: QPushButton = QPushButton('Breakpoints')
        self.breakpoints_button.setCheckable(True)
        self.breakpoints_button.setToolTip(
            'The few points the requirement is written from — the lines '
            'where its power laws bend')
        self.breakpoints_button.clicked.connect(self._to_breakpoints)
        self.interpolated_button: QPushButton = QPushButton('Interpolated')
        self.interpolated_button.setCheckable(True)
        self.interpolated_button.setToolTip(
            'The same requirement read onto evenly spaced frequency '
            'lines at the spacing beside, as a controller writes its target')
        self.interpolated_button.clicked.connect(self._to_interpolated)
        for button in (self.breakpoints_button, self.interpolated_button):
            self.form_group.addButton(button)
            head.addWidget(button)
        self.spacing_box: DoubleSpinBox = DoubleSpinBox()
        self.spacing_box.setRange(0.0, 100000.0)
        self.spacing_box.setDecimals(4)
        self.spacing_box.setSuffix(' Hz')
        self.spacing_box.setSpecialValueText('spacing?')
        self.spacing_box.setToolTip(
            'The frequency spacing of the interpolated lines — the '
            'specification\'s own, or a time history\'s averaging, '
            'or typed here')
        head.addWidget(self.spacing_box)
        grid.addLayout(head, 0, 0, 1, 4)
        # the bands' two constraints, for every channel of the object:
        # the bands themselves are on the plot, and these say how a
        # drag there moves them
        constraints = QHBoxLayout()
        constraints.addWidget(QLabel('Bands:'))
        self.symmetric_box: QCheckBox = QCheckBox('Symmetric')
        self.symmetric_box.setToolTip(
            'The band above the target is the band below, mirrored; a '
            'drag on one edge moves both')
        self.symmetric_box.toggled.connect(
            lambda on: self._constraint_toggled('symmetric', on))
        self.uniform_box: QCheckBox = QCheckBox('Uniform')
        self.uniform_box.setToolTip(
            'One band over the whole frequency range; a drag anywhere '
            'moves it everywhere')
        self.uniform_box.toggled.connect(
            lambda on: self._constraint_toggled('uniform', on))
        constraints.addWidget(self.symmetric_box)
        constraints.addWidget(self.uniform_box)
        constraints.addStretch(1)
        grid.addLayout(constraints, 1, 0, 1, 4)
        # how the levels are typed: a level at every breakpoint, or a
        # level at the first and a slope into each one after it, as
        # many written tables give them. The two are the same
        # requirement; which one is down decides what an edit holds
        entry = QHBoxLayout()
        self.entry_label: QLabel = QLabel('Enter:')
        entry.addWidget(self.entry_label)
        self.entry_group: QButtonGroup = QButtonGroup(self)
        self.entry_group.setExclusive(True)
        self.levels_button: QPushButton = QPushButton('Levels')
        self.levels_button.setCheckable(True)
        self.levels_button.setToolTip(
            'Type a level at every breakpoint; the slopes between them '
            'follow')
        self.levels_button.clicked.connect(lambda: self._entry_clicked('levels'))
        self.slopes_button: QPushButton = QPushButton('Slopes')
        self.slopes_button.setCheckable(True)
        self.slopes_button.setToolTip(
            'Type a level at the first breakpoint and a slope in dB per '
            'octave into each one after it; the level the slope reaches is '
            'shown beside it, and an edit holds every slope it does not '
            'change')
        self.slopes_button.clicked.connect(lambda: self._entry_clicked('slopes'))
        for button in (self.levels_button, self.slopes_button):
            self.entry_group.addButton(button)
            entry.addWidget(button)
        entry.addStretch(1)
        self.areas_box: QCheckBox = QCheckBox('Areas and running rms')
        self.areas_box.setToolTip(
            "Each segment's slope and area, its share of the whole, and the "
            'rms accumulated from the lowest breakpoint, for the channel on '
            'the plot: written on the plot, the running rms drawn dashed '
            'under it, and read out under the breakpoints')
        self.areas_box.toggled.connect(self._areas_toggled)
        entry.addWidget(self.areas_box)
        grid.addLayout(entry, 2, 0, 1, 4)
        # where the sheet came from, and what a door could not carry
        self.origin: QLabel = QLabel('')
        self.origin.setWordWrap(True)
        self.origin.setEnabled(False)
        grid.addWidget(self.origin, 3, 0, 1, 4)

        # the breakpoints: a row per frequency, a column per channel —
        # the ordinary table, with everything that brings
        self.points: CopyPasteTableView = CopyPasteTableView()
        self.points.setToolTip(
            'The target at each breakpoint, a power law between them — '
            'type a level per channel in the display units shown. Click '
            'a column or row header to select it, Cmd-click to add cells; '
            'copy, paste, Delete to clear, right-click for Batch Edit, '
            'drag the corner handle to fill down')
        self.points.setMinimumHeight(120)
        self.points.edits_applied.connect(self._edits_applied)
        grid.addWidget(self.points, 4, 0, 1, 4)
        buttons = QHBoxLayout()
        self.add_button: QPushButton = QPushButton('Add breakpoint')
        self.add_button.setToolTip('A new breakpoint after the selected one, '
                                   'or at the end')
        self.add_button.clicked.connect(self._add_point)
        self.remove_button: QPushButton = QPushButton('Remove breakpoint')
        self.remove_button.setToolTip('Take the selected breakpoints out')
        self.remove_button.clicked.connect(self._remove_point)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        self.scale_box: DoubleSpinBox = DoubleSpinBox()
        self.scale_box.setRange(-60.0, 60.0)
        self.scale_box.setDecimals(2)
        self.scale_box.setSuffix(' dB')
        self.scale_box.setToolTip('Raise or lower the selected levels by '
                                  'this much — select a column header for '
                                  'a channel, a row header for a breakpoint')
        self.scale_button: QPushButton = QPushButton('Scale selected')
        self.scale_button.setToolTip('Raise or lower the selected levels by '
                                     'the decibels in the box beside')
        self.scale_button.clicked.connect(self._scale_selected)
        buttons.addWidget(self.scale_box)
        buttons.addWidget(self.scale_button)
        grid.addLayout(buttons, 5, 0, 1, 4)
        # the breakpoint table read, for the channel on the plot: shown
        # while Areas and running rms is ticked (`show_reading`)
        self.reading: QLabel = QLabel('')
        self.reading.setWordWrap(True)
        self.reading.setTextFormat(Qt.TextFormat.RichText)
        self.reading.setVisible(False)
        self._reading_channel: str | None = None
        grid.addWidget(self.reading, 6, 0, 1, 4)

        # the pairs: coherence and phase between every two channels,
        # unstated until the author says — independent is a statement
        pairs_label = QLabel('Cross terms')
        pairs_label.setEnabled(False)
        grid.addWidget(pairs_label, 7, 0, 1, 4)
        self.pairs: CopyPasteTableView = CopyPasteTableView()
        self.pairs.setToolTip(
            'The cross term of each pair from its coherence (0 '
            'independent, 1 fully coherent) and phase. A pair left blank '
            'is absent from the specification; Delete clears one')
        self.pairs.setMinimumHeight(110)
        self.pairs.edits_applied.connect(self._edits_applied)
        grid.addWidget(self.pairs, 8, 0, 1, 4)
        all_row = QHBoxLayout()
        all_row.addWidget(QLabel('All pairs:'))
        self.all_coherence: DoubleSpinBox = DoubleSpinBox()
        self.all_coherence.setRange(0.0, 1.0)
        self.all_coherence.setDecimals(3)
        self.all_coherence.setSingleStep(0.1)
        self.all_coherence.setToolTip('Coherence to state for every pair')
        self.all_phase: DoubleSpinBox = DoubleSpinBox()
        self.all_phase.setRange(-180.0, 180.0)
        self.all_phase.setDecimals(1)
        self.all_phase.setSuffix('°')
        self.all_phase.setToolTip('Phase to state for every pair')
        self.all_button: QPushButton = QPushButton('Set')
        self.all_button.setToolTip('State this coherence and phase for '
                                   'every pair at once')
        self.all_button.clicked.connect(self._set_all_pairs)
        for widget in (self.all_coherence, self.all_phase, self.all_button):
            all_row.addWidget(widget)
        all_row.addStretch(1)
        grid.addLayout(all_row, 9, 0, 1, 4)

        # the bands are not here: they are dragged on the plot, per
        # section, for every selected channel at once (Brandon,
        # 2026-09-06 — the only place the bands are shown is the plot)
        self.problem: QLabel = QLabel('')
        self.problem.setWordWrap(True)
        grid.addWidget(self.problem, 10, 0, 1, 4)

        # no button: every edit lands on the specification the sheet
        # is open on as it is made (Brandon, 2026-09-06), so switching
        # to another object never leaves an edit behind
        grid.setRowStretch(11, 1)

    # ---- what the panel is describing -------------------------------------

    def show_draft(self, draft: SpecificationDraft, unit_system: UnitSystem,
                   origin: str = '',
                   spacing_hint: float | None = None) -> None:
        """Show a draft, with each channel's levels in the display units
        of its density. `origin` says where the sheet came from and is
        shown with the draft's notes. `spacing_hint` is the line spacing
        to offer when the draft has none, such as a time history's
        averaging."""
        self._unit_system = unit_system
        self._loading = True
        try:
            (self.interpolated_button if draft.form == 'lines'
             else self.breakpoints_button).setChecked(True)
            # slopes are typed on the breakpoints; on the lines form
            # the choice does not apply and is not shown
            (self.slopes_button if draft.entry == 'slopes'
             else self.levels_button).setChecked(True)
            # and so are the areas, which read a breakpoint table; a
            # controller's lines are a density per line, integrated as
            # bins (`SpecificationDraft.reading`)
            for widget in (self.entry_label, self.levels_button,
                           self.slopes_button, self.areas_box):
                widget.setVisible(draft.form == 'breakpoints')
            # the constraints read off the channels: on when every
            # channel of the sheet has them
            self.symmetric_box.setChecked(
                all(entry['symmetric'] for entry in draft.bands))
            self.uniform_box.setChecked(
                all(entry['uniform'] for entry in draft.bands))
            spacing = draft.spacing if draft.spacing is not None else spacing_hint
            self.spacing_box.setValue(spacing if spacing is not None else 0.0)
            said = origin
            if draft.notes:
                said = (said + ' — ' if said else '') + '; '.join(draft.notes)
            self.origin.setText(said)
            self.origin.setVisible(bool(said))
            self._sheet = Sheet(draft, unit_system)
            self._install_models()
        finally:
            self._loading = False
        self._restate()

    def _install_models(self) -> None:
        """Fresh models over the sheet, on both views."""
        points, pairs = sheet_models(self._sheet, self)
        for view, model in ((self.points, points), (self.pairs, pairs)):
            old = view.model()
            view.setModel(model)
            if old is not None:
                old.deleteLater()
            model.edit_rejected.connect(self._refused)
            model.dataChanged.connect(self._model_changed)
        self.points.resizeColumnsToContents()
        self.pairs.resizeColumnsToContents()
        # asked outright: a model swapped under a view that is on screen
        # is a repaint the view schedules itself, but Brandon saw the
        # old rows stand until the selection changed (2026-09-06), so
        # the viewports are told rather than trusted
        for view in (self.points, self.pairs):
            view.viewport().update()
            view.update()

    def draft(self) -> SpecificationDraft:
        """The draft the sheet holds, in SI. Raises ValueError when no
        sheet is open, or for anything no specification could hold."""
        if self._sheet is None:
            raise ValueError('no sheet is open')
        return self._sheet.draft()

    def _reload(self) -> None:
        """The grids restated from the sheet after a change of shape —
        a breakpoint added, removed or re-sorted — with the selection
        let go, since the rows it named have moved."""
        self._loading = True
        try:
            self._install_models()
        finally:
            self._loading = False

    def _model_changed(self, *_args) -> None:
        if not self._loading:
            self._settle.start()

    def _edits_settled(self) -> None:
        if self._sheet is not None and (self._sheet.reordered
                                        or self._sheet.spread):
            # a frequency typed between two others moved its row, or an
            # edit by slope moved the levels of the rows after it: the
            # grids are restated from the sheet
            self._sheet.reordered = self._sheet.spread = False
            self._reload()
        self._settled()

    def _settled(self) -> None:
        """Say where the sheet stands and hand it on."""
        draft = self._restate()
        if draft is not None:
            self.changed.emit(draft)

    def _refused(self, reason: str) -> None:
        self.problem.setText(reason)

    def _edits_applied(self, applied: int, rejected: int, reason: str) -> None:
        if rejected and reason:
            self.problem.setText(f'{rejected} cell{"s" * (rejected != 1)} '
                                 f'refused: {reason}')

    def _selected_rows(self) -> list[int]:
        selection = self.points.selectionModel()
        if selection is None:
            return []
        rows = sorted({index.row() for index in selection.selectedIndexes()})
        if not rows and self.points.currentIndex().isValid():
            rows = [self.points.currentIndex().row()]
        return rows

    def _add_point(self) -> None:
        """A breakpoint after the selected one — halfway to the next
        in log frequency, its levels the neighbors' — or past the end."""
        sheet = self._sheet
        if sheet is None:
            return
        rows = self._selected_rows()
        after = rows[-1] if rows else len(sheet.frequencies) - 1
        low = sheet.frequencies[after]
        if after + 1 < len(sheet.frequencies):
            frequency = (low * sheet.frequencies[after + 1]) ** 0.5
        else:
            frequency = low * 2.0
        was = sheet.draft()
        sheet.frequencies.insert(after + 1, frequency)
        for row in sheet.levels:
            row.insert(after + 1, row[after])
        sheet.bands = was._bands_on(sheet.frequencies)   # the split inherits
        self._reload()
        self._settled()

    def _remove_point(self) -> None:
        sheet = self._sheet
        if sheet is None:
            return
        rows = self._selected_rows()
        if not rows or len(sheet.frequencies) - len(rows) < 2:
            self.problem.setText('a specification keeps at least two '
                                 'breakpoints')
            return
        was = sheet.draft()
        keep = [i for i in range(len(sheet.frequencies)) if i not in rows]
        sheet.frequencies = [sheet.frequencies[i] for i in keep]
        sheet.levels = [[row[i] for i in keep] for row in sheet.levels]
        sheet.bands = was._bands_on(sheet.frequencies)   # the merge keeps its start's
        self._reload()
        self._settled()

    def _entry_clicked(self, entry: str) -> None:
        """Levels or Slopes: the sheet's draft typed the other way. The
        window keeps the draft and redraws; the object is not written,
        since the requirement is the same."""
        if self._loading:
            return
        self.entry_asked.emit(entry)

    @property
    def areas_wanted(self) -> bool:
        """Whether the segment areas and the running rms are asked for,
        and the sheet is on its breakpoints, where they are offered."""
        return (self.areas_box.isChecked() and self._sheet is not None
                and self._sheet.form == 'breakpoints')

    def _areas_toggled(self, on: bool) -> None:
        self.show_reading(self._reading_channel)
        if not self._loading:
            self.areas_asked.emit(bool(on))

    def show_reading(self, channel: str | None = None) -> None:
        """Read the sheet's breakpoint table for `channel` (the one on
        the plot; the first when it is not on the sheet) under the
        grid: each segment's slope, area, share of the whole and the
        rms accumulated to its end, then the total, in the display
        units. Hidden while Areas and running rms is off, and on the
        lines form, where it is not offered. A table of more segments
        than are labeled on the plot is read as its total alone."""
        from ..plot.breakpoints import MOST_LABELS, readout

        self._reading_channel = channel
        if not self.areas_wanted or self._sheet is None:
            self.reading.setVisible(False)
            return
        try:
            draft = self.draft()
        except ValueError:
            self.reading.setVisible(False)
            return
        k = draft.channels.index(channel) if channel in draft.channels else 0
        rows = readout(draft.reading(k), draft.dims[k], self._unit_system)
        shown = rows if len(rows) - 1 <= MOST_LABELS else rows[-1:]
        right = ' align="right"'
        cells = ''.join(
            '<tr>' + ''.join(f'<td{align}>{row[key]}</td>' for key, align in (
                ('segment', ''), ('slope', ''), ('area', right),
                ('share', right), ('running', right))) + '</tr>'
            for row in shown)
        head = ('<tr><td><i>segment</i></td><td><i>slope</i></td>'
                f'<td{right}><i>area</i></td><td{right}><i>share</i></td>'
                f'<td{right}><i>rms to its end</i></td></tr>')
        said = ('' if shown is rows else
                f' ({len(rows) - 1} segments, too many to list: the total '
                'alone)')
        self.reading.setText(
            f'<b>{draft.channels[k]}</b>, the channel on the plot{said}'
            f'<table cellspacing="0" cellpadding="2">{head}{cells}</table>')
        self.reading.setVisible(True)

    def _constraint_toggled(self, which: str, on: bool) -> None:
        if self._loading:
            return
        self.constraint_asked.emit(which, bool(on))

    def _to_breakpoints(self) -> None:
        """The few points the power laws bend at — asked of the whole
        specification, not the channels the sheet happens to hold
        (Brandon, 2026-09-06: the form is the object's)."""
        self.form_asked.emit('breakpoints', None)

    def _to_interpolated(self) -> None:
        """Read onto lines at the spacing in the box — the whole
        specification again."""
        spacing = self.spacing_box.value()
        if not spacing > 0:
            self.breakpoints_button.setChecked(True)
            self.problem.setText('give the spacing of the frequency lines '
                                 'to interpolate onto — there is no time '
                                 'history to take it from')
            return
        self.form_asked.emit('lines', spacing)

    def _scale_selected(self) -> None:
        """The selected levels by the box's decibels — a column header
        for a channel, a row header for a breakpoint, Cmd-click for a
        few cells (Brandon, 2026-09-06: scale selected, not scale all)."""
        sheet = self._sheet
        # by slope, a slope cell in the selection is not a level and is
        # passed over; each selected level is scaled as it stands, and
        # the slopes into and out of it follow from the levels
        cells = ([] if sheet is None else
                 [(index.row(), sheet.level_channel(index.column()), index.column())
                  for index in self.points.editable_selection()
                  if sheet.level_channel(index.column()) is not None])
        if sheet is None or not cells:
            self.problem.setText('select the levels to scale — a column '
                                 'header selects a channel, a row header a '
                                 'breakpoint')
            return
        factor = 10 ** (self.scale_box.value() / 10.0)
        for row, k, _column in cells:
            sheet.levels[k][row] *= factor
        model = self.points.model()
        rows = [r for r, _k, _c in cells]
        columns = [c for _r, _k, c in cells]
        model.dataChanged.emit(model.index(min(rows), min(columns)),
                               model.index(max(rows), model.columnCount() - 1
                                           if sheet.by_slope else max(columns)))

    def _set_all_pairs(self) -> None:
        sheet = self._sheet
        if sheet is None:
            return
        for key in sheet.pair_keys:
            sheet.pairs[key] = (self.all_coherence.value(),
                                self.all_phase.value())
        model = self.pairs.model()
        if model.rowCount():
            model.dataChanged.emit(model.index(0, 1),
                                   model.index(model.rowCount() - 1, 2))
        else:
            self._settled()

    def _restate(self):
        """Say what the sheet cannot hold, or what it leaves unstated —
        an unstated pair is absent from the specification, not a
        refusal."""
        try:
            draft = self.draft()
        except ValueError as refusal:
            self.problem.setText(str(refusal))
            return None
        unset = draft.unset_pairs()
        self.problem.setText(
            f'{len(unset)} pair{"s" * (len(unset) != 1)} unstated — absent '
            'from the specification until a coherence and phase are typed, '
            'or set for all at once' if unset else '')
        return draft
