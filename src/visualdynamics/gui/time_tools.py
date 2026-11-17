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
from ..core.time_compare import compare, response_records, scaled, scores, sides
from .editors import DoubleSpinBox


class TimeTools:
    """Display time tools beside the plot; the project owns the computations."""

    def __init__(self, window: Any) -> None:
        self.window: Any = window
        pane = window.data_pane
        self.group: QActionGroup = QActionGroup(pane)
        self.group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        self.compare_action: QAction = self._action(
            'TRAC / Overlay', 'Overlay paired responses and choose the TRAC time window')
        self.scale_action: QAction = self._action(
            'Scale', 'Preview a multiplier and apply it to copies of selected time histories')
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
        self.map_mode.addItems(['Off', 'TRAC points', 'sTRAC points', 'TRAC contour', 'sTRAC contour'])
        self.map_mode.currentIndexChanged.connect(self._edited)
        self.mesh_choice: QComboBox = QComboBox()
        self.mesh_choice.currentIndexChanged.connect(self._edited)
        self.layout_choice: QComboBox = QComboBox()
        self.layout_choice.addItems(['Stacked', 'Geometry on right', 'Geometry on left'])
        self.layout_choice.currentIndexChanged.connect(self._arrange_views)
        self.figures: QSplitter = QSplitter()
        self.traces: QSplitter = QSplitter(Qt.Orientation.Vertical)
        self.figures.addWidget(self.traces)
        self.waterfall_frame: QWidget = QWidget()
        waterfall_layout = QVBoxLayout(self.waterfall_frame)
        waterfall_layout.setContentsMargins(0, 0, 0, 0)
        close_row = QHBoxLayout()
        close_row.addWidget(QLabel('Waterfall'))
        close_row.addStretch()
        close = QPushButton('×')
        close.setToolTip('Hide waterfall')
        close.clicked.connect(self._close_waterfall)
        close_row.addWidget(close)
        waterfall_layout.addLayout(close_row)
        self.traces.addWidget(self.waterfall_frame)
        pane._time_tools_layout.insertWidget(0, self.figures, 1)
        self.figures.hide()
        form.addRow('Geometry correlation', self.map_mode)
        form.addRow('FEM overlay', self.mesh_choice)
        form.addRow('Layout', self.layout_choice)
        self.correlation_view: Any = None
        self.correlation_key: tuple | None = None
        self.correlation_count: int = 0
        self.mesh_source: tuple | None = None
        form.addRow(self.table)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.settings, 1)
        self.scale_settings: QWidget = QWidget()
        scale_form = QFormLayout(self.scale_settings)
        self.factor: DoubleSpinBox = DoubleSpinBox()
        self.factor.setToolTip('Multiply response samples by this factor; units stay the same')
        self.factor.setRange(-1e12, 1e12)
        self.factor.setDecimals(9)
        self.factor.setKeyboardTracking(False)
        self.factor.setValue(1)
        scale_form.addRow('Multiplier', self.factor)
        self.references: QCheckBox = QCheckBox('Include reference channels')
        self.references.setToolTip('Also scale the reference force or drive channels')
        scale_form.addRow(self.references)
        layout.addWidget(self.scale_settings)
        self.apply: QPushButton = QPushButton('Save comparison')
        self.apply.setToolTip('Keep the previewed result and its settings in the project')
        layout.addWidget(self.apply)
        pane._time_tools_layout.addWidget(self.panel)
        self.start.valueChanged.connect(self._edited)
        self.stop.valueChanged.connect(self._edited)
        self.factor.valueChanged.connect(self._edited)
        self.references.toggled.connect(self._edited)
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
        if self.correlation_view is not None:
            self.correlation_view.hide()
        pane = self.window.data_pane
        if not self.figures.isHidden():
            pane.author_split.insertWidget(0, pane.graphics)
            if pane._waterfall_page is not None:
                pane.author_split.insertWidget(1, pane._waterfall_page)
            pane.author_split.show()
            self.figures.hide()
        self.panel.hide()
        self.compare_action.setVisible(False)
        self.scale_action.setVisible(False)

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
        dialog = QDialog(self.window)
        dialog.setWindowTitle('Pair time responses')
        dialog.resize(780, 440)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('Choose one first and one second response per pair.'))
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(['First (colored)', 'Second (gray)'])
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 350)
        layout.addWidget(table)

        def add(pair=None):
            row = table.rowCount()
            table.insertRow(row)
            for col in range(2):
                box = QComboBox()
                for n, i in choices:
                    box.addItem(f'{n}: {project[n].record_label(i)}', (n, i))
                if pair is not None:
                    box.setCurrentIndex(choices.index(tuple(pair[col*2:col*2+2])))
                else:
                    box.setCurrentIndex(col)
                table.setCellWidget(row, col, box)

        if self.draft is not None:
            for pair in self.draft.time_comparison['pairs']:
                add(pair)
        else:
            add()
        row = QHBoxLayout()
        add_button = QPushButton('Add pair')
        remove_button = QPushButton('Remove pair')
        add_button.clicked.connect(lambda: add())
        remove_button.clicked.connect(lambda: table.removeRow(table.currentRow()))
        row.addWidget(add_button)
        row.addWidget(remove_button)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.explicit = [[*table.cellWidget(i, 0).currentData(),
                              *table.cellWidget(i, 1).currentData()]
                             for i in range(table.rowCount())]
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
            elif self.scale_action.isChecked():
                self.add_results(project.scale_time_histories(
                    *names, factor=self.factor.value(), records=records,
                    include_references=self.references.isChecked()))
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
            self.scale_action.setChecked(False)
            self.explicit = self.draft = None
            self.key = key
        self.series = series
        name, data, _ = series[0]
        self.saved = name if len(series) == 1 and getattr(data, 'time_comparison', None) else None
        comparable = (len(series) > 1 or self.saved is not None
                      or (series[0][2] is not None and len(series[0][2]) > 1))
        self.compare_action.setVisible(comparable)
        self.scale_action.setVisible(self.saved is None)
        if not (self.saved or self.compare_action.isChecked() or self.scale_action.isChecked()):
            return None
        if self.saved:
            self.compare_action.setChecked(True)
        pane = self.window.data_pane
        pane.show_controls(map_wanted=None, diagonal=None, cmif=False,
                           complex_data=False, pair=False)
        pane.toolbar.show()
        self.panel.show()
        comparison = self.compare_action.isChecked()
        self.settings.setVisible(comparison)
        self.scale_settings.setVisible(not comparison)
        return self._preview()

    def _preview(self) -> str:
        self.apply.setEnabled(True)
        try:
            if self.compare_action.isChecked():
                return self._compare_preview(self.series[0][1])
            return self._scale_preview()
        except ValueError as error:
            self.note.setText(str(error))
            self.apply.setEnabled(False)
            pane = self.window.data_pane
            pane.show_waterfall(False)
            pane.graphics.clear()
            with QSignalBlocker(self.table):
                self.table.setRowCount(0)
            return str(error)

    def _scale_preview(self) -> str:
        from ..plot import build_plots

        name, data, records = self.series[0]
        result = scaled(data, self.factor.value(), records, self.references.isChecked())
        pane = self.window.data_pane
        pane.show_waterfall(False)
        build_plots(pane.graphics, [(name, data, records), ('Scaled', result, records)],
                    unit_system=self.window.unit_system, theme=self.window.theme_name)
        self.apply.setText('Apply scale')
        note = f'Preview: {name}\nApply to {len(self.series)} selected histories.'
        self.note.setText(note)
        return note

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
        map_note = self._correlation_map(rows, start, stop)
        if pane.graphics.parentWidget() is not self.traces:
            self.traces.insertWidget(0, pane.graphics)
        pane.graphics.show()
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
        from .opengl import three_d_view

        project = self.window.project
        linked = project.object_group_of(self.saved) if self.saved else None
        geometries = [project[name] for name in linked or ()
                      if isinstance(project[name], Geometry)]
        from ..viz.time_correlation import matching_surfaces

        surfaces = matching_surfaces(geometries[0], project.values()) if geometries else []
        choices = [project.name_of(surface) for surface in surfaces]
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
            if self.correlation_view is not None:
                self.correlation_view.hide()
            return 'Link a saved comparison to geometry to map its scores.' if mode else ''
        if self.correlation_view is None:
            if self.window.data_pane.offscreen:
                return ''
            view = three_d_view(self.window.data_pane)
            if view is None:
                return '3D geometry rendering is unavailable.'
            self.correlation_view = view
            self._arrange_views()
            view.enable_mesh_picking(
                callback=self._pick_correlation_point, left_clicking=True,
                show=False, show_message=False, picker='point', use_actor=False)
        view = self.correlation_view
        metric = 'strac' if mode in (2, 4) else 'trac'
        from ..theme import theme

        colors = theme(self.window.theme_name)
        key = (self.key, tuple(map(id, geometries)), mode, start, stop,
               self.window.unit_system, self.window.theme_name)
        if key != self.correlation_key:
            from ..viz.time_correlation import draw_time_correlation

            reset = self.correlation_key is None or self.correlation_key[:2] != key[:2]
            self.correlation_key = key
            view.clear()
            view.set_background(colors['scene_background'])
            count = draw_time_correlation(view, geometries, rows, metric=metric,
                                          contour=mode >= 3, unit_system=self.window.unit_system,
                                          text_color=colors['scene_text'])
            if reset:
                view.reset_camera()
            self.correlation_count = count
        view.show()
        message = f'{self.correlation_count} scored stations. Click a station to select its traces.'
        if mode >= 3 and not any(len(geometry.elem_type) for geometry in geometries):
            message += ' Select a FEM overlay to show a surface contour.'
        return message

    def _pick_correlation_point(self, mesh: Any) -> None:
        if mesh is not None and 'comparison_row' in mesh.field_data:
            rows = list(mesh.field_data['comparison_row'])
            current = self.table.currentRow()
            row = rows[(rows.index(current) + 1) % len(rows)] if current in rows else rows[0]
            self.table.setCurrentCell(int(row), 0)
            if row == current:
                self._edited()

    def _close_waterfall(self) -> None:
        self.window.data_pane.waterfall_action.setChecked(False)
        self._edited()

    def _arrange_views(self) -> None:
        view = self.correlation_view
        if view is None:
            return
        layout = self.layout_choice.currentIndex()
        self.figures.setOrientation(Qt.Orientation.Horizontal if layout else Qt.Orientation.Vertical)
        self.figures.insertWidget(1 if layout == 1 else 0, view)
        self.figures.setSizes([1, 2] if layout != 1 else [2, 1])
