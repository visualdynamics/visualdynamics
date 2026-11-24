"""Settings and preview for time-history scaling and paired TRAC."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.data import TimeHistory
from ..core.time_compare import compare, response_records, scores, sides
from .editors import DoubleSpinBox
from .object_tables import _elsewhere
from .tables import Column, CopyPasteTableView, TableModel


def pair_model(pairs: list[list[int]], labels: list[str],
               parent: Any = None) -> TableModel:
    """The pair editor's table: one row a pair, the first and the second
    response each picked from `labels`. `pairs` holds indices into
    `labels` and is edited in place; the comparison the pairs are given
    to is what the journal records, so the columns journal elsewhere."""

    def getter(side: int):
        return lambda obj, row: labels[obj[row][side]]

    def setter(side: int):
        def put(obj, row, text):
            text = str(text).strip()
            if text not in labels:
                raise ValueError(f'{text!r} is not one of the picked responses')
            obj[row][side] = labels.index(text)
        return put

    columns = [Column(title, getter(side), set=setter(side), choices=list(labels),
                      journal=_elsewhere,
                      alignment=int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
               for side, title in enumerate(('First (colored)', 'Second (gray)'))]
    return TableModel(pairs, columns, len, parent)


class TimeTools:
    """Display time tools beside the plot; the project owns the computations."""

    def __init__(self, window: Any) -> None:
        self.window: Any = window
        pane = window.data_pane
        self.group: QActionGroup = QActionGroup(pane)
        self.group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        self.compare_action: QAction = self._action(
            'TRAC / Overlay', 'Overlay paired responses and choose the TRAC time window')
        self.panel: QWidget = QWidget(pane)
        self.panel.setMinimumWidth(245)
        layout = QVBoxLayout(self.panel)
        self.note: QLabel = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.settings: QWidget = QWidget()
        form = QFormLayout(self.settings)
        self.start: DoubleSpinBox = self._seconds()
        self.stop: DoubleSpinBox = self._seconds()
        self.start.setToolTip('First time in seconds included in the TRAC score')
        self.stop.setToolTip('Last time in seconds included in the TRAC score')
        form.addRow('TRAC start (s)', self.start)
        form.addRow('TRAC end (s)', self.stop)
        self.manual: QPushButton = QPushButton('Edit pairs…')
        self.manual.setToolTip('Choose the two source records for each comparison station')
        form.addRow(self.manual)
        self.table: QTableWidget = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['DOF', 'TRAC', 'sTRAC'])
        self.table.horizontalHeaderItem(2).setToolTip(
            'Signed gain multiplying the second trace to fit the first. Ideal: 1.')
        self.map_mode: QComboBox = QComboBox()
        self.map_mode.addItems(['Off', 'TRAC points', 'sTRAC points',
                                'TRAC contour', 'sTRAC contour',
                                'TRAC smooth contour', 'sTRAC smooth contour'])
        self.map_mode.currentIndexChanged.connect(self._edited)
        self.map_mode.setToolTip('Color the linked geometry in the 3-D view by each '
                                 'station\'s score, as points or a contour')
        self.mesh_choice: QComboBox = QComboBox()
        self.mesh_choice.currentIndexChanged.connect(self._edited)
        self.mesh_choice.setToolTip('A finite element geometry to draw the contour on, under the measured stations')
        self.figures: QSplitter = QSplitter()
        self.traces: QSplitter = QSplitter(Qt.Orientation.Vertical)
        self.figures.addWidget(self.traces)
        self.trace_visible: QCheckBox = QCheckBox('Show time traces')
        self.trace_visible.setChecked(True)
        self.trace_visible.setToolTip('Show the selected pair\'s time traces')
        self.trace_visible.toggled.connect(self._edited)
        self.trace_frame: QWidget = self._figure_frame(
            'Time traces', lambda: self.trace_visible.setChecked(False))
        self.waterfall_frame: QWidget = self._figure_frame('Waterfall', self._close_waterfall)
        self.traces.addWidget(self.trace_frame)
        self.traces.addWidget(self.waterfall_frame)
        pane._time_tools_layout.insertWidget(0, self.figures, 1)
        self.figures.hide()
        form.addRow('Geometry correlation', self.map_mode)
        form.addRow('FEM overlay', self.mesh_choice)
        form.addRow(self.trace_visible)
        self.correlation_key: tuple | None = None
        self.correlation_count: int = 0
        self.mesh_source: tuple | None = None
        form.addRow(self.table)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.settings, 1)
        self.apply: QPushButton = QPushButton('Save comparison')
        self.apply.setToolTip('Keep the previewed result and its settings in the project')
        layout.addWidget(self.apply)
        pane._time_tools_layout.addWidget(self.panel)
        self.start.valueChanged.connect(self._edited)
        self.stop.valueChanged.connect(self._edited)
        self.table.currentCellChanged.connect(self._edited)
        self.manual.clicked.connect(self._manual_pairs)
        self.apply.clicked.connect(self._apply)
        self.key: tuple | None = None
        self.series: list = []
        self.explicit: list | None = None
        self.draft: TimeHistory | None = None
        self.saved: str | None = None
        self.hide()

    def _action(self, name: str, tip: str) -> QAction:
        action = self.window.data_pane.toolbar.addAction(name)
        action.setCheckable(True)
        action.setToolTip(tip)
        self.group.addAction(action)
        action.triggered.connect(self._start_reading)
        return action

    @staticmethod
    def _seconds() -> DoubleSpinBox:
        box = DoubleSpinBox()
        box.setDecimals(12)
        box.setRange(-1e12, 1e12)
        box.setSingleStep(0.01)
        box.setKeyboardTracking(False)
        return box

    def hide(self) -> None:
        """Put away the settings while a different view is being drawn."""
        pane = self.window.data_pane
        if not self.figures.isHidden():
            pane.author_split.insertWidget(0, pane.graphics)
            if pane._waterfall_page is not None:
                pane.author_split.insertWidget(1, pane._waterfall_page)
            pane.author_split.show()
            self.figures.hide()
        self.panel.hide()
        self.compare_action.setVisible(False)

    def _start_reading(self, _checked: bool) -> None:
        self.window.data_pane.end_readings()
        self.window.render_current()

    def _edited(self, *_args: Any) -> None:
        # Redrawing the whole window hides this pane before putting it back,
        # taking keyboard focus away from the value being entered.
        if self.panel.isVisible() and self.series:
            self.window._show_status(self._preview())

    def add_results(self, names: list[str]) -> None:
        """Show all outputs as separate rows and select them together."""
        if not names:
            return
        window = self.window
        with QSignalBlocker(window.tree):
            for name in names:
                window.show_object(name, select=False)
            window.tree.setCurrentItem(window._item_for_object(names[0]))
            window.tree.clearSelection()
            for name in names:
                window._item_for_object(name).setSelected(True)
        window._selection_changed()

    def _input(self) -> tuple[list[str], dict]:
        return ([n for n, _d, _r in self.series],
                {n: list(r) for n, _d, r in self.series if r is not None})

    def _manual_pairs(self) -> None:
        names, records = self._input()
        project = self.window.project
        choices = [(name, i) for name in names
                   for i in records.get(name, response_records(project[name]))]
        if len(choices) < 2:
            return
        labels = [f'{n}: {project[n].record_label(i)}' for n, i in choices]
        if self.draft is not None:
            pairs = [[choices.index(tuple(pair[:2])), choices.index(tuple(pair[2:]))]
                     for pair in self.draft.time_comparison['pairs']]
        else:
            pairs = [[0, 1]]
        dialog = QDialog(self.window)
        dialog.setWindowTitle('Pair time responses')
        dialog.resize(780, 440)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('Choose one first and one second response per pair.'))
        model = pair_model(pairs, labels, dialog)
        table = CopyPasteTableView(dialog)
        table.setModel(model)
        table.rows_deletable = True
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 350)
        layout.addWidget(table)
        refused = QLabel()
        refused.setWordWrap(True)
        model.edit_rejected.connect(refused.setText)
        table.edits_applied.connect(
            lambda _written, _refused, why: refused.setText(why))

        def change(edit) -> None:
            model.beginResetModel()
            edit()
            model.endResetModel()

        def remove(rows) -> None:
            keep = [pair for k, pair in enumerate(pairs) if k not in set(rows)]
            change(lambda: pairs.__setitem__(slice(None), keep))

        table.rows_deleted.connect(remove)
        row = QHBoxLayout()
        add_button = QPushButton('Add pair')
        add_button.setToolTip('Add a pair below the others')
        remove_button = QPushButton('Remove pair')
        remove_button.setToolTip('Remove the pair holding the current cell')
        add_button.clicked.connect(lambda: change(lambda: pairs.append([0, 1])))
        remove_button.clicked.connect(
            lambda: remove([table.currentIndex().row()]))
        row.addWidget(add_button)
        row.addWidget(remove_button)
        layout.addLayout(row)
        layout.addWidget(refused)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted and pairs:
            self.explicit = [[*choices[first], *choices[second]]
                             for first, second in pairs]
            self.draft = None
            self.window.render_current()

    def _apply(self) -> None:
        project = self.window.project
        names, records = self._input()
        try:
            if self.saved is not None:
                project.set_time_comparison_window(self.saved, *self._window())
                self.window.render_current()
            elif self.compare_action.isChecked():
                name = project.compare_time_histories(
                    *names, pairs=self.explicit, records=records,
                    start=self._window()[0], stop=self._window()[1])
                self.add_results([name])
        except ValueError as error:
            self.note.setText(str(error))

    def render(self, series: list, *, compare_sections: bool = False) -> str | None:
        """Offer the tools for time selections and draw a requested preview."""
        self.hide()
        if not series or not all(isinstance(d, TimeHistory) for _n, d, _r in series):
            self.key = None
            return None
        key = tuple((n, id(d), tuple(r) if r is not None else None) for n, d, r in series)
        if key != self.key:
            if compare_sections:
                self.window.data_pane.waterfall_action.setChecked(False)
            self.compare_action.setChecked(compare_sections)
            self.explicit = self.draft = None
            self.key = key
        self.series = series
        name, data, _ = series[0]
        self.saved = name if len(series) == 1 and getattr(data, 'time_comparison', None) else None
        comparable = (len(series) > 1 or self.saved is not None
                      or (series[0][2] is not None and len(series[0][2]) > 1))
        self.compare_action.setVisible(comparable)
        if not (self.saved or self.compare_action.isChecked()):
            return None
        if self.saved:
            self.compare_action.setChecked(True)
        pane = self.window.data_pane
        pane.show_controls(map_wanted=None, diagonal=None, cmif=False,
                           complex_data=False, pair=False)
        pane.toolbar.show()
        self.panel.show()
        return self._preview()

    def _preview(self) -> str:
        self.apply.setEnabled(True)
        try:
            return self._compare_preview(self.series[0][1])
        except ValueError as error:
            self.note.setText(str(error))
            self.apply.setEnabled(False)
            pane = self.window.data_pane
            pane.show_waterfall(False)
            pane.graphics.clear()
            with QSignalBlocker(self.table):
                self.table.setRowCount(0)
            return str(error)

    def _window(self) -> tuple[float, float]:
        return (max(self.start.value(), float(self.draft.abscissa[0])),
                min(self.stop.value(), float(self.draft.abscissa[-1])))

    def _compare_preview(self, data: TimeHistory) -> str:
        from ..plot import build_plots
        from ..viz.paired import add_paired_stage
        from ..viz.waterfall import place_camera, waterfall_groups

        self.manual.setVisible(self.saved is None)
        self.apply.setText('Save TRAC window' if self.saved else 'Save comparison')
        if self.draft is None:
            names, records = self._input()
            self.draft = (data if self.saved else compare(
                self.window.project, names, pairs=self.explicit, records=records))
            settings = self.draft.time_comparison
            with QSignalBlocker(self.start), QSignalBlocker(self.stop):
                low, high = float(self.draft.abscissa[0]), float(self.draft.abscissa[-1])
                self.start.setRange(low, high)
                self.stop.setRange(low, high)
                self.start.setValue(settings['start'])
                self.stop.setValue(settings['stop'])
        draft = self.draft
        start, stop = self._window()
        rows = scores(draft, start, stop)
        left, right = sides(draft, (start, stop))
        pane = self.window.data_pane
        pane.offer_waterfall(True)
        current = max(0, min(self.table.currentRow(), len(rows)-1))
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                values = [row['dof'], *[f'{row[k]:.4g}' if row[k] is not None else '—'
                                         for k in ('trac', 'strac')]]
                for j, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    a, _r, b, _s = row['pair']
                    item.setToolTip(row['reason'] or f'{a} / {b}\n{row["samples"]} samples')
                    self.table.setItem(i, j, item)
            self.table.setCurrentCell(current, 0)
        if pane.showing_waterfall:
            pane.show_waterfall(True)
        if pane.showing_waterfall and pane.waterfall_plotter is not None:
            plotter = pane.waterfall_plotter
            plotter.clear()
            from ..theme import theme

            plotter.set_background(theme(self.window.theme_name)['scene_background'])
            groups = waterfall_groups(left)
            pane.show_quantities([(key[0], key) for key, _r in groups] if len(groups) > 1 else [])
            labels = {left.record_label(i): ('TRAC —' if row['trac'] is None
                                              else f'TRAC {row["trac"]:.3f}')
                      for i, row in enumerate(rows)}
            info = add_paired_stage(plotter, left, right, unit_system=self.window.unit_system,
                             theme=self.window.theme_name, station_notes=labels,
                             quantity=pane.chosen_quantity() if len(groups) > 1 else None,
                             page=pane.chosen_page())
            pane.show_pages(info['page'], info['pages'])
            if self.window._waterfall_object != self.key:
                place_camera(plotter)
                self.window._waterfall_object = self.key
            plotter.render()
        else:
            pane.show_waterfall(False)
        row = rows[current]
        a, _i, b, _j = row['pair']
        build_plots(pane.graphics, [(a, left, [current]), (b, right, [current])],
                    unit_system=self.window.unit_system, theme=self.window.theme_name)
        score = '—' if row['trac'] is None else f"{row['trac']:.3f}"
        from pyqtgraph import PlotItem

        for plot in pane.graphics.ci.items:
            if isinstance(plot, PlotItem):
                plot.setTitle(f"{row['dof']} · TRAC {score}")
        map_note = self._correlation_map(rows, start, stop)
        if pane.graphics.parentWidget() is not self.trace_frame:
            self.trace_frame.layout().addWidget(pane.graphics, 1)
        pane.graphics.show()
        self.trace_frame.setVisible(self.trace_visible.isChecked())
        waterfall = pane.showing_waterfall and pane._waterfall_page is not None
        if waterfall:
            page = pane._waterfall_page
            if page.parentWidget() is not self.waterfall_frame:
                self.waterfall_frame.layout().addWidget(page, 1)
            page.show()
        self.waterfall_frame.setVisible(waterfall)
        pane.author_split.hide()
        self.figures.show()
        count = sum(row['trac'] is not None for row in rows)
        note = (f'{count} of {len(rows)} pairs scored, {start:.6g}–{stop:.6g} s.\n'
                'sTRAC multiplies the second trace to fit the first. Ideal: 1.')
        if count:
            mean = sum(row['trac'] for row in rows if row['trac'] is not None) / count
            note += f'\nMean TRAC: {mean:.3f}'
        if map_note:
            note += '\n' + map_note
        self.note.setText(note)
        return note

    def _correlation_map(self, rows: list, start: float, stop: float) -> str:
        from ..core.geometry import Geometry

        project = self.window.project
        linked = project.object_group_of(self.saved) if self.saved else None
        geometries = [project[name] for name in linked or ()
                      if isinstance(project[name], Geometry)]
        choices = [name for name, obj in project.items()
                   if isinstance(obj, Geometry) and all(obj is not geometry for geometry in geometries)] if geometries else []
        with QSignalBlocker(self.mesh_choice):
            selected = self.mesh_choice.currentData() if self.mesh_source == self.key else None
            self.mesh_source = self.key
            self.mesh_choice.clear()
            self.mesh_choice.addItem('None', None)
            for name in choices:
                self.mesh_choice.addItem(name, name)
            self.mesh_choice.setCurrentIndex(max(0, self.mesh_choice.findData(selected)))
        self.mesh_choice.setEnabled(bool(choices))
        mesh = self.mesh_choice.currentData()
        if mesh is not None:
            geometries.append(project[mesh])
        self.map_mode.setEnabled(bool(geometries))
        mode = self.map_mode.currentIndex()
        if not mode or not geometries:
            self.correlation_key = None
            return 'Link a saved comparison to geometry to map its scores.' if mode else ''
        # drawn in the window's own 3-D view, with its view cube and
        # camera, rather than in a second view of its own
        window = self.window
        window.reveal_scene()
        plotter = window.scene.plotter
        if plotter is None:
            return '3D geometry rendering is unavailable.'
        metric = 'strac' if mode in (2, 4, 6) else 'trac'
        from ..theme import theme

        colors = theme(window.theme_name)
        key = (self.key, tuple(map(id, geometries)), mode, start, stop,
               window.unit_system, window.theme_name)
        if key != self.correlation_key:
            from ..viz.geometry import annotate_scene, held_camera, restore_camera
            from ..viz.time_correlation import draw_time_correlation

            # framed when the geometries change, as any scene is; a new
            # window or contour keeps the camera where it was put
            showing = ('time correlation', self.key, key[1])
            reframe = window._framed != showing
            window._framed = showing
            camera = held_camera(plotter)
            plotter.clear()
            plotter.set_background(colors['scene_background'],
                                   top=colors['scene_background_top'])
            self.correlation_count, axis_unit = draw_time_correlation(
                plotter, geometries, rows, metric=metric, contour=mode >= 3,
                unit_system=window.unit_system, text_color=colors['scene_text'],
                smooth=mode >= 5)
            window.scene.axis_unit = axis_unit
            annotate_scene(plotter, axis_unit, colors, window.scene.bounds_visible,
                           window.scene.orientation_visible, cube=True)
            if reframe:
                window.scene.frame()
            else:
                restore_camera(plotter, camera)
            self.correlation_key = key
            plotter.render()
        message = f'{self.correlation_count} scored stations on the 3-D view.'
        if mode >= 3 and not any(len(geometry.elem_type) for geometry in geometries):
            message += ' Select a FEM overlay to show a surface contour.'
        return message

    def _close_waterfall(self) -> None:
        self.window.data_pane.waterfall_action.setChecked(False)
        self._edited()

    @staticmethod
    def _figure_frame(title, close_action):
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel(title))
        header.addStretch()
        close = QPushButton('×')
        close.setToolTip(f'Hide {title.lower()}')
        close.clicked.connect(close_action)
        header.addWidget(close)
        layout.addLayout(header)
        return frame
