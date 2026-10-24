"""Every screenshot in the workflow guides, generated from the app.

The real MainWindow, offscreen, walking the six workflows over the
plate demonstration data step by step and grabbing the window at each
one — so the pictures in the documentation are the application working,
regenerate with the fixtures, and cannot drift from what a user sees.

    QT_QPA_PLATFORM=offscreen ./.venv/bin/python tools/make_guide_screenshots.py

Needs stressdata/plate/transient.nc4 for the transient walkthrough
(generate_plate_runs.py --transient, in the visualdynamics-generators
repository) and writes docs/guide/images/.
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PLATE = ROOT / 'testdata' / 'plate'
STRESS = ROOT / 'stressdata' / 'plate'
OUT = ROOT / 'docs' / 'guide' / 'images'

#: a size that reads in the docs: wide enough for the three panes,
#: short enough that text in the grabs stays legible when scaled
WINDOW = (1440, 900)


def pump(times=12):
    from PySide6.QtWidgets import QApplication

    for _ in range(times):
        QApplication.processEvents()


def fresh_window():
    from visualdynamics.gui.main_window import MainWindow

    window = MainWindow(offscreen_3d=True)
    window.resize(*WINDOW)
    # the display units are part of the picture: pinned, not whatever
    # QSettings remembered from the last interactive session
    window.unit_combo.setCurrentText('in-slinch-lbf-s (g)')
    # the flat readings, deliberately: the 3-D stage became the default
    # after these pages were first shot, and under the offscreen
    # platform the VTK widget cannot paint into a grab — the window
    # shows its '3D view disabled' label instead, which is what every
    # regenerated picture would hold. The flat reading photographs,
    # and it is the reading the pages narrate.
    window.data_pane.waterfall_action.setChecked(False)
    window.show()
    pump()
    return window


def close(window):
    from PySide6.QtWidgets import QApplication

    window.close()
    window.deleteLater()
    for _ in range(10):
        QApplication.processEvents()


def select(window, *names, current=None):
    # current first: Qt's setCurrentItem clears the selection, so
    # setting it last quietly reduced every pair to a single object
    window.tree.setCurrentItem(
        window._item_for_object(current or names[0]))
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)
    pump()


#: the color of a "click here" mark: an orange that reads against the
#: light theme's panes and the plot's white alike, and is no data color
MARK = '#e8590c'


def act(window, label):
    """The visible bar button labeled `label` (an act on the plot's or
    the 3-D view's bar, or a reading), for a mark."""
    from PySide6.QtWidgets import QToolBar

    for bar in window.findChildren(QToolBar):
        for action in bar.actions():
            if action.text() == label:
                button = bar.widgetForAction(action)
                if button is not None and button.isVisible():
                    return button
    raise ValueError(f'no visible button is labeled {label!r}')


def _marked_rect(window, target):
    """Where `target` sits in the window: an object's name (its tree
    row), the project row (`window.test_item`), a QAction, a widget, or
    `(widget, rect)` for a place the widget paints rather than a widget
    of its own (a table cell, a link bracket), `rect` in the widget's
    coordinates."""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QTreeWidgetItem

    if isinstance(target, tuple):
        widget, rect = target
        return QRect(widget.mapTo(window, rect.topLeft()), rect.size())
    if isinstance(target, str):
        target = window._item_for_object(target)
    if isinstance(target, QTreeWidgetItem):
        tree = window.tree
        row = tree.visualItemRect(target)
        return QRect(tree.viewport().mapTo(window, row.topLeft()), row.size())
    if isinstance(target, QAction):
        target = act(window, target.text())
    if not target.isVisible():
        raise ValueError(f'{target!r} is not on screen to be marked')
    return QRect(target.mapTo(window, QPoint(0, 0)), target.size())


def shoot(window, name, marks=()):
    """Grab the window as `name`.png. Each of `marks` (in the order the
    user clicks them) is boxed in `MARK` with its number beside it, so
    the picture says where to click; the boxes come from the widgets'
    own geometry, so a regenerated picture cannot point at the wrong
    place."""
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPen

    pump(20)
    path = OUT / f'{name}.png'
    picture = window.grab()
    if marks:
        painter = QPainter(picture)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setBold(True)
        font.setPixelSize(13)
        painter.setFont(font)
        for number, target in enumerate(marks, start=1):
            box = _marked_rect(window, target).adjusted(-4, -4, 4, 4)
            painter.setPen(QPen(QColor(MARK), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box, 6, 6)
            badge = QRect(max(box.left() - 12, 2), max(box.top() - 12, 2),
                          24, 24)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(MARK))
            painter.drawEllipse(badge)
            painter.setPen(QColor('white'))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(number))
        painter.end()
    picture.save(str(path))
    print(f'  {name}.png', flush=True)


CHROME = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')


def report_shot(window, name, report=None):
    """The exported report (`report`, or the project's first),
    screenshotted by headless Chrome.

    The report editor is a QtWebEngine view, and web engine content
    does not paint into an offscreen window grab — the app shot came
    back blank. The exported HTML is the same report the reader would
    hand out, and Chrome renders it headlessly and reproducibly."""
    import subprocess
    import tempfile

    if not os.path.exists(CHROME):
        print(f'  {name}.png skipped — no Chrome for the report render',
              flush=True)
        return
    if report is None:
        report = by_type(window, 'Report')
    with tempfile.TemporaryDirectory() as scratch:
        html = os.path.join(scratch, 'report.html')
        window.project.export_report(report, html)
        subprocess.run([CHROME, '--headless', '--disable-gpu',
                        f'--screenshot={OUT / name}.png',
                        '--window-size=1280,1600',
                        '--hide-scrollbars', f'file://{html}'],
                       check=True, capture_output=True, timeout=120)
    print(f'  {name}.png', flush=True)


def shoot_report(window, kind):
    """Where to click for the report, and the report it makes from the
    `kind` template; every walkthrough ends on these two pictures."""
    window.tree.setCurrentItem(window.test_item)
    pump(20)
    shoot(window, f'{kind}-report-click',
          marks=[window.test_item, act(window, 'Generate Report')])
    window.generate_report(kind)
    pump(30)
    report_shot(window, f'{kind}-report')


def add_geometry_and_photos(window, which):
    """The two objects a controller file cannot bring: the article's
    geometry, and the setup photographs — taken here from the demo
    project so the report renders complete."""
    import visualdynamics

    window.import_paths([str(PLATE / 'geometry.npz')])
    pump(20)
    geometry = by_type(window, 'Geometry')
    window.objects[geometry].define_units('m')
    demo = ROOT / 'stressdata' / 'plate_projects' / f'{which}.vdyn'
    if demo.exists():
        photos = visualdynamics.load(str(demo))['Photos']
        window.add_object('Photos', photos)
    names = list(window.objects)
    select(window, *names)
    window.link_selected()
    pump()
    window.set_link_role(geometry, 'Basis')
    pump()


def by_type(window, kind):
    return next(name for name, obj in window.objects.items()
                if type(obj).__name__ == kind)


# ---- modal ----------------------------------------------------------------

def modal():
    from PySide6.QtCore import QRect, QRectF, Qt

    print('modal:', flush=True)
    window = fresh_window()
    # where the survey goes in: dropped on the project tree
    shoot(window, 'modal-drop', marks=[window.tree])
    window.import_paths([str(PLATE / 'modal_spectra.nc4'),
                         str(PLATE / 'test_geometry.npz')])
    pump(30)
    geometry = by_type(window, 'Geometry')
    frf = by_type(window, 'Frf')
    select(window, frf)
    shoot(window, 'modal-import')

    # declare the geometry's units in the Imported Units pane
    select(window, geometry)
    window.define_units()
    pump(20)
    shoot(window, 'modal-units')
    table = window.units_table
    model = table.model()
    unit = next(column for column in range(model.columnCount())
                if model.headerData(column, Qt.Orientation.Horizontal)
                == 'Unit')
    cell = (table.viewport(), table.visualRect(model.index(0, unit)))
    shoot(window, 'modal-units-click', marks=[geometry, cell])
    window.close_units_panel()
    window.objects[geometry].define_units('m')
    pump()

    # link the measured side and declare it the basis
    names = list(window.objects)
    select(window, *names)
    # the rows as one mark: a box and a number on each of five
    # neighbors buried the names they pointed at
    tree = window.tree
    rows = [tree.visualItemRect(window._item_for_object(name))
            for name in names]
    picked = (tree.viewport(), QRect(
        rows[0].left(), min(row.top() for row in rows),
        rows[0].width(),
        max(row.bottom() for row in rows) - min(row.top() for row in rows)
        + 1))
    shoot(window, 'modal-link-click', marks=[picked, window.link_action])
    window.link_selected()
    pump()
    span = next(items for _color, items, _bold, _name in tree.link_spans
                if window._item_for_object(frf) in items)
    rows = [tree.visualItemRect(item) for item in span if item is not None]
    rows = [row for row in rows if row.height() > 0]
    top = min(row.center().y() for row in rows)
    bottom = max(row.center().y() for row in rows)
    bracket = (tree.viewport(), QRect(0, top, 12, bottom - top))
    shoot(window, 'modal-basis-click', marks=[bracket])
    window.set_link_role(frf, 'Basis')
    pump()
    select(window, frf)
    shoot(window, 'modal-link')

    # the channel table's own page wants a picture of the table: every
    # column typed, the drop-downs and the check box visible
    table = by_type(window, 'ChannelTable')
    select(window, table)
    shoot(window, 'channel-table')

    history = by_type(window, 'TimeHistory')
    select(window, history)
    pane = window.data_pane
    pane.averaging_action.trigger()
    pump(20)
    shoot(window, 'modal-psds-click',
          marks=[history, pane.averaging_action,
                 pane.averaging_panel.psds_button])
    window.compute_psds()
    pump(20)
    shoot(window, 'modal-psds')

    select(window, frf)
    pump(20)
    shoot(window, 'modal-fit-click',
          marks=[frf, act(window, 'Fit Modal Model')])
    window.start_modal_fit()
    pump(20)
    shoot(window, 'modal-fit-open')
    for _ in range(2):
        window.confirm_fit_mode()
        pump(10)
    shoot(window, 'modal-fit-buttons-click',
          marks=[window.find_mode_button, window.confirm_mode_button,
                 window.refine_modes_button])
    window.find_next_mode()
    pump(10)
    shoot(window, 'modal-fit')
    # three more confirms: five modes, the repeated pair's second
    # tooth included
    for _ in range(3):
        window.confirm_fit_mode()
        pump(10)

    fitted = by_type(window, 'ShapeSet')
    select(window, fitted, frf)
    pump(20)
    shoot(window, 'modal-edit-fit-click',
          marks=[fitted, frf, act(window, 'Edit Fit')])
    # the deflected shape comes from the same scene the app animates,
    # rendered by the headless API: the offscreen window cannot host
    # VTK (see conftest for why), and this is the one figure that is
    # a scene rather than a plot or the window itself
    window.project.animate(fitted, mode=0,
                           screenshot=str(OUT / 'modal-mode.png'))
    print('  modal-mode.png', flush=True)

    # the FEM pair, for correlation
    window.import_paths([str(PLATE / 'geometry.npz'),
                         str(PLATE / 'shapes.npy')])
    pump(30)
    fem_geometry = next(n for n, o in window.objects.items()
                        if type(o).__name__ == 'Geometry'
                        and n != geometry)
    fem_shapes = next(n for n, o in window.objects.items()
                      if type(o).__name__ == 'ShapeSet' and n != fitted)
    window.objects[fem_geometry].define_units('m')
    window.objects[fem_shapes].define_units('kg')
    select(window, fem_geometry, fem_shapes)
    window.link_selected()
    pump()
    select(window, fitted, fem_shapes)
    pump(20)
    shoot(window, 'modal-mac')
    # the picked square, where the grid drew it: a tall grid puts the
    # first set across, so its mode is the x of the cell
    grid = next(item for item in window.mac_view.ci.items
                if hasattr(item, 'getViewBox'))
    row, column = window._compare['cell']
    x, y = (row, column) if window._compare.get('tall') else (column, row)
    on_scene = grid.getViewBox().mapViewToScene(
        QRectF(x, y, 1, 1)).boundingRect()
    square = (window.mac_view.viewport(),
                  window.mac_view.mapFromScene(on_scene).boundingRect())
    shoot(window, 'modal-mac-click',
          marks=[fitted, fem_shapes, square, window.add_matches_action])

    matched = window.project.match_modes(fitted, fem_shapes,
                                         threshold=0.7)
    window.show_object(matched)
    demo = ROOT / 'stressdata' / 'plate_projects' / 'modal.vdyn'
    if demo.exists():
        import visualdynamics

        window.add_object('Photos',
                          visualdynamics.load(str(demo))['Photos'])
        select(window, 'Photos', frf)
        window.link_selected()
        pump()
    shoot_report(window, 'modal')
    close(window)


# ---- random ---------------------------------------------------------------

def random():
    print('random:', flush=True)
    window = fresh_window()
    # where a run goes in: dropped on the project tree
    shoot(window, 'random-drop', marks=[window.tree])
    window.import_paths([str(PLATE / 'random.nc4')])
    pump(30)
    history = by_type(window, 'TimeHistory')
    select(window, history)
    shoot(window, 'random-import')

    pane = window.data_pane
    pane.statistics_action.trigger()
    pump(20)
    shoot(window, 'random-statistics-click',
          marks=[history, pane.statistics_action])

    pane.averaging_action.trigger()
    pump(20)
    shoot(window, 'random-psds-click',
          marks=[history, pane.averaging_action,
                 pane.averaging_panel.psds_button])
    window.compute_psds()
    pump(20)
    shoot(window, 'random-psds')

    psd = by_type(window, 'Psd')
    select(window, psd)
    pane.octave_action.trigger()
    pump(20)
    shoot(window, 'random-octave-click',
          marks=[psd, pane.octave_action, pane.octave_panel.apply_button])
    # Apply Octave Bands calls this verb; the pane's default spacing is
    # the same sixth-octave
    banded = window.project.compute_octave(psd)
    window.show_object(banded)
    pump(20)
    shoot(window, 'random-octave')

    select(window, history)
    if not pane.averaging_action.isChecked():
        pane.averaging_action.trigger()
        pump(20)
    shoot(window, 'random-coherence-click',
          marks=[history, pane.averaging_panel.coherence_button])
    window.compute_multiple_coherence()
    pump(20)
    shoot(window, 'random-coherence')

    spec = by_type(window, 'Specification')
    select(window, psd, spec)
    pump(20)
    shoot(window, 'random-comparison', marks=[psd, spec])

    add_geometry_and_photos(window, 'random')
    shoot_report(window, 'random')
    close(window)


# ---- transient ------------------------------------------------------------

def transient():
    run = STRESS / 'transient.nc4'
    if not run.exists():
        print('transient: skipped — run generate_plate_runs.py '
              '--transient from visualdynamics-generators')
        return
    print('transient:', flush=True)
    window = fresh_window()
    # where a run goes in: dropped on the project tree
    shoot(window, 'transient-drop', marks=[window.tree])
    window.import_paths([str(run)])
    pump(30)
    history = by_type(window, 'TimeHistory')
    spec = by_type(window, 'TransientSpecification')
    select(window, history)
    shoot(window, 'transient-import')

    pane = window.data_pane
    select(window, spec)
    pump(20)
    shoot(window, 'transient-spec', marks=[spec])

    # the shaker check is a reading of the target alone; put away
    # again so the pair below opens on its replication views
    pane.fit_action.trigger()
    pump(20)
    shoot(window, 'transient-fit-click', marks=[spec, pane.fit_action])
    pane.fit_action.trigger()
    pump(20)

    select(window, history, spec, current=history)
    pump(20)
    shoot(window, 'transient-overlay', marks=[history, spec])

    # the waveform-error reading of the same pair
    pane.replication_actions['waveform'].trigger()
    pump(20)
    # the bars follow the table's picked cells, and the first cell alone
    # is picked as it opens: the whole first event, every channel
    window.table.selectColumn(1)
    pump(20)
    shoot(window, 'transient-error',
          marks=[pane.replication_actions['waveform']])

    pane.replication_actions['segments'].trigger()
    pump(20)
    shoot(window, 'transient-segments-click',
          marks=[pane.replication_actions['segments'],
                 pane.segments_panel.add_button])
    pane.replication_actions['overlay'].trigger()
    pump(20)

    select(window, spec)
    if not pane.averaging_action.isChecked():
        pane.averaging_action.trigger()
        pump(20)
    shoot(window, 'transient-psds-click',
          marks=[spec, pane.averaging_action,
                 pane.averaging_panel.psds_button])
    window.compute_psds()
    pump()
    select(window, history)
    window.compute_psds()
    pump(20)
    measured = by_type(window, 'Psd')
    target = next(n for n, o in window.objects.items()
                  if type(o).__name__ == 'Specification'
                  and 'PSD' in n)
    select(window, measured, target)
    pump(20)
    # top row first: the tree lists the target's PSDs above the
    # measured ones, and a lower row's badge drawn first is covered
    shoot(window, 'transient-level', marks=[target, measured])

    add_geometry_and_photos(window, 'transient')
    shoot_report(window, 'transient')
    close(window)


# ---- shock ----------------------------------------------------------------

def shock():
    print('shock:', flush=True)
    window = fresh_window()
    window.import_paths([str(PLATE / 'shock.nc4')])
    pump(30)
    # the file cannot say it was a shock test; the engineer does, from
    # the project row's right-click menu — a popup a window grab does
    # not hold, so the row is what is marked
    shoot(window, 'shock-type-click', marks=[window.test_item])
    window.set_project_type('Shock')
    pump()
    history = by_type(window, 'TimeHistory')
    select(window, history)
    pane = window.data_pane
    pane.shocks_action.trigger()
    pump(20)
    shoot(window, 'shock-detect-click',
          marks=[history, pane.shocks_action,
                 pane.shock_panel.detect_button])
    window._detect_shocks()
    pump(20)
    shoot(window, 'shock-import')

    shoot(window, 'shock-srs-click',
          marks=[history, pane.shock_panel.srs_button])
    window.compute_srs()
    pump(20)
    shoot(window, 'shock-srs')

    # the requirement band, authored the way the demo project authors it
    import numpy as np

    from visualdynamics.core.data import ShockSpecification

    srs_name = by_type(window, 'Srs')
    srs = window.objects[srs_name]
    frequencies = np.asarray(srs.abscissa, dtype=float)
    top = np.asarray(srs.ordinate, dtype=float).max(axis=0)
    level = 10.0 ** np.interp(
        np.log10(frequencies),
        np.log10([frequencies[0], 400.0, 2000.0, frequencies[-1]]),
        np.log10([top.max() * 0.05, top.max() * 0.9,
                  top.max() * 1.1, top.max() * 1.1]))
    window.add_object('Shock Specification', ShockSpecification(
        abscissa=frequencies, ordinate=level[None, :],
        response_dof=[srs.response_dof[0]],
        ordinate_dim='acceleration', ordinate_unit='m/s**2',
        q=srs.q, kind=srs.kind,
        abort_upper=level[None, :] * 10.0 ** 0.3,
        abort_lower=level[None, :] * 10.0 ** -0.3))
    pump()
    select(window, srs_name, 'Shock Specification')
    pump(20)
    shoot(window, 'shock-spec')

    shoot(window, 'shock-error-click',
          marks=[srs_name, 'Shock Specification',
                 pane.srs_actions['error']])
    pane.srs_actions['error'].trigger()
    pump(20)
    shoot(window, 'shock-points-click',
          marks=[srs_name, 'Shock Specification',
                 pane.srs_actions['points']])
    pane.srs_actions['curves'].trigger()
    pump(20)

    select(window, history)
    shoot(window, 'shock-pulse-click', marks=[history, pane.pulse_action])

    add_geometry_and_photos(window, 'shock')
    shoot_report(window, 'shock')
    close(window)

    # the optional filter, integration and stroke, in a window of their
    # own: applied in the one above, they would change the spectra and
    # the report the other pictures show
    window = fresh_window()
    window.import_paths([str(PLATE / 'shock.nc4')])
    pump(30)
    window.set_project_type('Shock')
    pump()
    history = by_type(window, 'TimeHistory')
    select(window, history)
    pane = window.data_pane
    pane.filter_action.trigger()
    pump(20)
    shoot(window, 'shock-filter-click',
          marks=[history, pane.filter_action,
                 pane.filter_panel.apply_button])
    before = set(window.objects)
    window.filter_data()
    pump(20)
    (filtered,) = set(window.objects) - before
    select(window, filtered)
    shoot(window, 'shock-integrate-click',
          marks=[filtered, act(window, 'Integrate')])

    select(window, history)
    shoot(window, 'shock-stroke-click', marks=[history, pane.stroke_action])
    close(window)


def sine():
    run = STRESS / 'sine.nc4'
    if not run.exists():
        print('sine: skipped — stressdata/plate/sine.nc4 absent', flush=True)
        return
    print('sine:', flush=True)
    window = fresh_window()
    shoot(window, 'sine-drop', marks=[window.tree])
    window.import_paths([str(run)])
    pump(30)
    # the file says what it is: a sine run arrives as a Sine Sweep
    # project, specification and all
    spec = by_type(window, 'SineSweepSpecification')
    select(window, spec)
    pane = window.data_pane
    shoot(window, 'sine-spec-click', marks=[spec, pane.waterfall_action])
    shoot(window, 'sine-spec')

    # the Sweep Rate reading of the same specification, which the
    # judgment section points at
    shoot(window, 'sine-sweep-rate-click',
          marks=[spec, pane.sweep_rate_action])

    history = by_type(window, 'TimeHistory')
    select(window, history)
    pump(10)
    shoot(window, 'sine-wavelet-click', marks=[history, pane.wavelet_action])
    # the wavelet reading of the sweep: the four tones' trajectories,
    # drawn where they actually are in time
    pane.wavelet_action.trigger()
    pump(40)
    shoot(window, 'sine-wavelet')
    pane.wavelet_action.trigger()
    pump(10)

    shoot(window, 'sine-extract-click',
          marks=[history, act(window, 'Extract Sine Levels')])
    pane.sine_action.trigger()
    pump(40)
    panel = pane.sine_panel
    shoot(window, 'sine-smoothing-click',
          marks=[history, pane.sine_action, panel.mode_box,
                 panel.extract_button])
    pane.sine_action.trigger()
    pump(10)

    window.extract_sine_levels()
    pump(30)
    levels = by_type(window, 'SineLevelSet')
    # the specification first: its row sits just above the levels', and
    # marked the other way round the second box is drawn over the first
    # badge
    select(window, spec)
    pump(20)
    shoot(window, 'sine-levels-click', marks=[spec, levels])
    select(window, levels, spec)
    pump(20)
    shoot(window, 'sine-levels')

    add_geometry_and_photos(window, 'sine')
    shoot_report(window, 'sine')
    close(window)


def sysid():
    stream = STRESS / 'sysid_stream.nc4'
    if not stream.exists():
        print('sysid: skipped — stressdata/plate/sysid_stream.nc4 absent',
              flush=True)
        return
    print('sysid:', flush=True)
    window = fresh_window()
    # declared first, so the import names the two streams itself; a
    # person importing into an untyped project is asked instead — the
    # dialog the page describes
    # where the stream goes in: dropped on the project tree, shot
    # before the type is declared, as a person dropping it sees it
    shoot(window, 'sysid-drop', marks=[window.tree])
    window.set_project_type('System ID')
    window.import_paths([str(stream)])
    pump(30)
    excitation = 'Excitation Time History'
    select(window, excitation)
    shoot(window, 'sysid-import')

    pane = window.data_pane
    panel = pane.averaging_panel

    def averaging_on():
        if not pane.averaging_action.isChecked():
            pane.averaging_action.trigger()
            pump(20)

    averaging_on()
    shoot(window, 'sysid-frfs-click',
          marks=[excitation, pane.averaging_action, panel.frfs_button])
    # Compute FRFs... asks for the estimator in a dialog; the project
    # verb takes the answer, H1, directly
    frfs = window.project.compute_frfs(excitation, 'H1')
    window.show_object(frfs)
    pump(30)
    shoot(window, 'sysid-frfs')

    select(window, excitation)
    averaging_on()
    shoot(window, 'sysid-coherence-click',
          marks=[excitation, panel.coherence_button])
    coherence = window.project.compute_multiple_coherence(excitation)
    window.show_object(coherence)
    pump(20)
    shoot(window, 'sysid-coherence')

    select(window, excitation)
    averaging_on()
    shoot(window, 'sysid-psds-click', marks=[excitation, panel.psds_button])
    driven = window.project.compute_psds(excitation)
    window.project['Noise Time History'].averaging = \
        window.project[excitation].averaging
    quiet = window.project.compute_psds('Noise Time History')
    # the project verbs put the results in the project; the window
    # catches up by being shown them, and only then can they be picked
    window.show_object(driven)
    window.show_object(quiet)
    pump(10)
    select(window, driven, quiet)
    pump(20)
    ratio = pane.spectra_actions['ratio']
    shoot(window, 'sysid-snr-click', marks=[driven, quiet, ratio])
    ratio.trigger()
    pump(20)
    shoot(window, 'sysid-snr')

    # add_geometry_and_photos, opened up so the group can be shot
    # selected before Link is pressed
    window.import_paths([str(PLATE / 'geometry.npz')])
    pump(20)
    geometry = by_type(window, 'Geometry')
    window.objects[geometry].define_units('m')
    demo = ROOT / 'stressdata' / 'plate_projects' / 'sysid.vdyn'
    if demo.exists():
        import visualdynamics

        window.add_object('Photos', visualdynamics.load(str(demo))['Photos'])
    names = list(window.objects)
    select(window, *names)
    rows = sorted(names, key=lambda name: window.tree.visualItemRect(
        window._item_for_object(name)).top())
    shoot(window, 'sysid-link-click',
          marks=[rows[0], rows[-1], window.link_action])
    window.link_selected()
    pump()
    window.set_link_role(geometry, 'Basis')
    pump()

    shoot_report(window, 'sysid')
    close(window)


def mixed():
    run = STRESS / 'mixed.nc4'
    if not run.exists():
        print('mixed: skipped — stressdata/plate/mixed.nc4 absent',
              flush=True)
        return
    print('mixed:', flush=True)
    window = fresh_window()
    shoot(window, 'mixed-drop', marks=[window.tree])
    window.import_paths([str(run)])
    pump(30)
    history = by_type(window, 'TimeHistory')
    select(window, history)
    shoot(window, 'mixed-import')

    # the random half, as the random walkthrough does it
    pane = window.data_pane
    pane.averaging_action.trigger()
    pump(20)
    shoot(window, 'mixed-psds-click',
          marks=[history, pane.averaging_action,
                 pane.averaging_panel.psds_button])
    window.compute_psds()
    pump(20)
    shoot(window, 'mixed-psds')

    psd = by_type(window, 'Psd')
    select(window, psd)
    pane.octave_action.trigger()
    pump(20)
    shoot(window, 'mixed-octave-click',
          marks=[psd, pane.octave_action, pane.octave_panel.apply_button])
    banded = window.project.compute_octave(psd)
    window.show_object(banded)
    pump(20)
    shoot(window, 'mixed-octave')

    select(window, history)
    if not pane.averaging_action.isChecked():
        pane.averaging_action.trigger()
        pump(20)
    shoot(window, 'mixed-coherence-click',
          marks=[history, pane.averaging_panel.coherence_button])
    window.compute_multiple_coherence()
    pump(20)
    shoot(window, 'mixed-coherence')

    # the specification banded too, as the random guide does it, so
    # the random report's octave section has its banded requirement
    spec = by_type(window, 'Specification')
    window.show_object(window.project.compute_octave(spec))
    pump(20)

    # the specification's row sits above the PSD's, so it is marked
    # first: the second badge would otherwise cover the first
    select(window, psd, spec)
    pump(20)
    shoot(window, 'mixed-comparison', marks=[spec, psd])

    # the sine half: the act on the time history's bar
    select(window, history)
    if pane.averaging_action.isChecked():
        pane.averaging_action.trigger()
        pump(20)
    shoot(window, 'mixed-sine-click',
          marks=[history, act(window, 'Extract Sine Levels')])
    window.extract_sine_levels()
    pump(30)
    levels = by_type(window, 'SineLevelSet')
    select(window, levels, by_type(window, 'SineSweepSpecification'))
    pump(20)
    shoot(window, 'mixed-sine-levels')

    # the tone read off the random's PSD, the cursor where it starts
    select(window, psd)
    pane.tone_action.trigger()
    pump(20)
    shoot(window, 'mixed-tones-click', marks=[psd, pane.tone_action])
    pane.tone_action.trigger()
    pump(20)

    add_geometry_and_photos(window, 'random')
    window.tree.setCurrentItem(window.test_item)
    pump(20)
    shoot(window, 'mixed-report-click',
          marks=[window.test_item, act(window, 'Generate Report')])
    # what the button does for this type, one template at a time so
    # each report's name is in hand for its own picture
    reports = {template: window.generate_report(template)
               for template in ('random', 'sine')}
    pump(30)
    for template, report in reports.items():
        report_shot(window, f'mixed-report-{template}', report)
    close(window)


# ---- the project tree page -------------------------------------------------


def tree():
    """The readings the project-tree page shows: one object at a time,
    and every pairing that means something. The flat readings are
    window grabs; the readings that live in the 3-D scene are rendered
    by the same headless entry points the API offers, since the
    offscreen window cannot host VTK (see fresh_window)."""
    import numpy as np

    from visualdynamics.deform import TimeDeflection, animation_records
    from visualdynamics.theme import theme as resolve_theme
    from visualdynamics.viz.animate import (
        GeometryAnimator,
        animate_envelope,
        animate_ods,
    )
    from visualdynamics.viz.geometry import annotate_scene

    print('tree:', flush=True)
    window = fresh_window()
    window.import_paths([str(PLATE / 'modal_spectra.nc4'),
                         str(PLATE / 'test_geometry.npz')])
    pump(30)
    geometry = by_type(window, 'Geometry')
    frf = by_type(window, 'Frf')
    history = by_type(window, 'TimeHistory')
    window.objects[geometry].define_units('m')
    names = list(window.objects)
    select(window, *names)
    window.link_selected()
    pump()

    def one(name, act):
        try:
            act()
        except Exception:  # noqa: BLE001 — one picture must not cost the rest
            import traceback
            traceback.print_exc()
            print(f'  {name}.png FAILED', flush=True)

    # the project row: the panes clear, Generate Report on the bar
    def project_row():
        from PySide6.QtWidgets import QLabel

        window.tree.clearSelection()
        window.tree.setCurrentItem(window.test_item)
        window.test_item.setSelected(True)
        pump(20)
        # the offscreen window says '3D view disabled' where the real
        # one holds an empty scene; the picture is of the empty pane
        for label in window.scene.findChildren(QLabel):
            if 'offscreen' in label.text():
                label.hide()
        pump(5)
        shoot(window, 'tree-project')
    one('tree-project', project_row)

    # one object at a time
    def frf_alone():
        select(window, frf)
        pump(10)
        # a checkable action toggles on trigger, as the tests drive it
        window.data_pane.drive_point_action.trigger()
        pump(20)
        shoot(window, 'tree-frf')
        window.data_pane.drive_point_action.trigger()
        pump(10)
    one('tree-frf', frf_alone)

    select(window, history)
    window.compute_psds()
    pump(20)
    psd = by_type(window, 'Psd')

    def psd_alone():
        select(window, psd)
        pump(20)
        shoot(window, 'tree-psd')
    one('tree-psd', psd_alone)

    geo = window.objects[geometry]
    us = window.unit_system
    colors = resolve_theme(window.theme_name)

    # data on the geometry, rendered headlessly
    def time_on_geometry():
        data = window.objects[history]
        indices, _notes = animation_records(data, None)
        dofs = [data.response_dof[i] for i in indices]
        ordinate = np.asarray(data.ordinate)[list(indices)]
        deflection = TimeDeflection(geo, dofs, ordinate)
        import pyvista as pv
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(colors['scene_background'])
        animator = GeometryAnimator(plotter, geo, deflection, unit_system=us,
                                    colormap=True)
        animator.set_scale(1.0)
        # the sample where the record is loudest: a still of a
        # deflection has to be taken where there is one
        loudest = int(np.argmax(np.abs(ordinate).max(axis=0)))
        animator.set_parameter(loudest)
        annotate_scene(plotter, animator.axis_unit, colors)
        plotter.screenshot(str(OUT / 'tree-geometry-time.png'))
        plotter.close()
        print('  tree-geometry-time.png', flush=True)
    one('tree-geometry-time', time_on_geometry)

    def ods_on_geometry():
        animate_ods(geo, window.objects[frf], unit_system=us,
                    theme=window.theme_name,
                    screenshot=str(OUT / 'tree-geometry-ods.png'))
        print('  tree-geometry-ods.png', flush=True)
    one('tree-geometry-ods', ods_on_geometry)

    def envelope_on_geometry():
        animate_envelope(geo, window.objects[psd], unit_system=us,
                         theme=window.theme_name,
                         screenshot=str(OUT / 'tree-geometry-envelope.png'))
        print('  tree-geometry-envelope.png', flush=True)
    one('tree-geometry-envelope', envelope_on_geometry)

    # an FRF and the modes fitted to it: the resynthesis under the measurement
    def synthesis():
        fitted = window.project.fit_modes(frf, limit=8)
        window.show_object(fitted)       # the verb added it; the tree learns of it here
        pump(20)
        window.pair_mode = 'overlay'
        select(window, frf, fitted, current=frf)
        pump(30)
        shoot(window, 'tree-frf-synthesis')
    one('tree-frf-synthesis', synthesis)
    close(window)


# ---- the finite element pages ---------------------------------------------


def show_scene(window, look=None):
    """Put the 3-D view's picture where the offscreen window holds its
    '3D view disabled' label, so a grab shows the model being built.

    The offscreen window still keeps the scene in an off-screen plotter
    (the panes' previews, the picked elements, the merge's marks); only
    the embedded widget is missing. Rendered at the label's size and
    set on the label, it photographs with the rest of the window.
    `look`, an actor's name and a zoom, centers the view on that actor
    (found by what is drawn there, so no coordinate has to be converted
    to the display unit the scene is drawn in) and zooms in from the
    framing, as a pan and a scroll do."""
    import numpy as np
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QSizePolicy

    from visualdynamics.gui.panes import image_of

    pump(10)
    label = window.scene._page
    # a picture held at its own size would push the panes about
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    plotter = window.scene.plotter
    plotter.window_size = [label.width(), label.height()]
    # fitted to the pane's width now: a table opened beside the view
    # narrows it, which the embedded widget's own resize would refit
    plotter.reset_camera()
    if look is not None:
        actor, zoom = look
        bounds = np.reshape(plotter.renderer.actors[actor].GetBounds(), (3, 2))
        camera = plotter.camera
        shift = bounds.mean(axis=1) - np.asarray(camera.focal_point)
        camera.focal_point = tuple(np.asarray(camera.focal_point) + shift)
        camera.position = tuple(np.asarray(camera.position) + shift)
        camera.zoom(zoom)
        # a camera moved by hand is not drawn until asked: the
        # screenshot alone returns the framing's picture again
        plotter.render()
    pixels = plotter.screenshot(return_img=True)
    label.setPixmap(QPixmap.fromImage(image_of(pixels)))


def upright(window, name, view):
    """Give geometry `name` the view it is built to be seen from and
    turn the camera to it, as Reset View does."""
    window.objects[name].view = view
    window.render_current()
    pump(10)
    window.scene.reset_view()


def table_width(window, width):
    """Give the table beside the 3-D view `width` pixels of the window,
    as dragging the divider does."""
    views = window.views
    total = sum(views.sizes())
    sizes = [0] * views.count()
    sizes[views.indexOf(window.scene)] = total - width
    sizes[views.indexOf(window.table_pane)] = width
    views.setSizes(sizes)
    pump(10)


def tree_rows(window, kind, name=None):
    """The tree rows whose reference is of `kind` ('family', 'group',
    'solids', 'solid'), of geometry `name` when given, in tree order."""
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    found = []

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            reference = child.data(0, ROLE_REFERENCE)
            if (reference is not None and reference[0] == kind
                    and name in (None, reference[1])):
                found.append(child)
            walk(child)

    walk(window.tree.invisibleRootItem())
    return found


def family_row(window, name, label):
    """Geometry `name`'s row for a family ('Quads', 'Hexes', 'Points')
    or its Solids, expanded into view."""
    item = window._item_for_object(name)
    item.setExpanded(True)
    return next(item.child(i) for i in range(item.childCount())
                if item.child(i).text(0).startswith(label))


def group_row(window, name, group):
    """Geometry `name`'s row for the element group named `group`."""
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    geometry = window.objects[name]
    gid = int(geometry.group_id[list(geometry.group_name).index(group)])
    row = next(r for r in tree_rows(window, 'group', name)
               if r.data(0, ROLE_REFERENCE)[2] == gid)
    row.parent().setExpanded(True)
    return row


def top_solids(window, name):
    """Geometry `name`'s parts' rows, in tree order: the solids directly
    under its Solids row, not those inside one."""
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    return [row for row in tree_rows(window, 'solid', name)
            if row.parent().data(0, ROLE_REFERENCE)[0] == 'solids']


def solid_row(window, name, label):
    """Geometry `name`'s part reading `label`."""
    return next(row for row in top_solids(window, name)
                if row.text(0) == label)


def pick(window, rows):
    """Select tree rows the way a click and Cmd-clicks do: the first
    current, the rest added."""
    window.tree.clearSelection()
    window.tree.setCurrentItem(rows[0])
    for row in rows[1:]:
        row.setSelected(True)
    pump(20)


def bar_act(window, verb):
    """Run the bar's act `verb` for the current selection."""
    act = next(a for a in window.acts_for() if a[0] == verb)
    act[3]()
    pump(20)


def open_modes(window):
    """The Modes reading on the selected geometry, its pane up."""
    window.scene.modes_action.setChecked(True)
    window.scene._choose_modes(True)
    pump(20)
    return window.scene.modes_panel


def solve(window, top):
    """Solve Modes on the selected geometry from the Modes pane, its
    highest frequency typed as the page says."""
    panel = open_modes(window)
    panel.top_box.setValue(top)
    panel.solve_button.click()
    pump(30)


def set_group(window, name, group, **fields):
    """Type into the Element Groups table, opened on `group` by the
    pencil on its row: each of `fields` a property's title without its
    unit ('Material', 'Thickness', 'E', ...) and the text typed into it.
    The table is one column per group picked, a property per row."""
    from PySide6.QtCore import Qt

    row = group_row(window, name, group)

    def shown():
        if window.editing is None or window.editing[1] != 'groups':
            return []
        model = window.table.model()
        return [str(model.headerData(c, Qt.Orientation.Horizontal))
                for c in range(model.item_count())]

    if group not in shown():
        pick(window, [row])
        window._tree_item_clicked(row, 1)
        pump(20)
    model = window.table.model()
    at = shown().index(group)
    for header, text in fields.items():
        prop = next(k for k, c in enumerate(model.columns)
                    if c.title.split(' [')[0] == header)
        model.setData(model.cell(at, prop), text, Qt.ItemDataRole.EditRole)
    pump(10)
    return row


#: the BARC's planes as the page's table types them: element group,
#: center and widths, inches
BARC_PLANES = [
    ('box', (0, -2.875, 1.5), (5.75, 0, 3)),
    ('box', (2.875, 0, 1.5), (0, 5.75, 3)),
    ('box', (-2.875, 0, 1.5), (0, 5.75, 3)),
    ('box', (-1.5625, 2.875, 1.5), (2.625, 0, 3)),
    ('box', (1.5625, 2.875, 1.5), (2.625, 0, 3)),
    ('right channel', (1.96875, 3.0625, 1.5), (0.9375, 0, 1)),
    ('right channel', (1.96875, 4.9375, 1.5), (0.9375, 0, 1)),
    ('right channel', (2.4375, 4, 1.5), (0, 1.875, 1)),
    ('left channel', (-2, 3.0625, 1.53125), (1, 0, 0.9375)),
    ('left channel', (-2, 4.9375, 1.53125), (1, 0, 0.9375)),
    ('left channel', (-2, 4, 1.0625), (1, 1.875, 0)),
    ('beam', (0, 5.0625, 1.5), (5, 0, 1)),
]


def fem():
    """The BARC page: a plate model typed plane by plane, tied at its
    bolts, given its properties and solved; then the same structure
    from solid blocks, meshed apart and bolted with Mate Surfaces."""
    import numpy as np
    from PySide6.QtCore import QItemSelectionModel, QPoint, Qt
    from PySide6.QtWidgets import QToolBar

    from visualdynamics.demo import barc
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    print('fem:', flush=True)
    window = fresh_window()
    show_scene(window)
    window.tree.clearSelection()
    window.tree.setCurrentItem(window.test_item)
    window.test_item.setSelected(True)
    pump(20)
    shoot(window, 'fem-new-geometry-click',
          marks=[window.test_item, act(window, 'New Geometry')])
    window.new_geometry_act()
    pump(20)
    name = 'Geometry'
    # upright, as the BARC's own model opens (y up); a new geometry
    # opens z up, and a person turns it
    upright(window, name, barc.VIEW)

    quads = family_row(window, name, 'Quads')
    pick(window, [quads])
    window.add_plane_act()
    pump(20)
    panel = window.scene.mesh_panel
    group, center, widths = BARC_PLANES[0]
    panel.set_values(group=group, center=center, widths=widths, size=0.125)
    show_scene(window)
    shoot(window, 'fem-add-plane-click',
          marks=[quads, act(window, 'Add Plane'), panel.add_button])
    panel.add_button.click()
    pump(5)
    window.close_mesh_pane()
    # the rest by the verb each Add records: the pane's fields hold four
    # decimals, and 1.96875 typed there is 1.9688, which leaves the
    # channels' flanges off their webs
    for group, center, widths in BARC_PLANES[1:]:
        across = [np.eye(3)[i] * w for i, w in enumerate(widths) if w]
        corner = np.subtract(center, np.sum(across, axis=0) / 2)
        window.project.add_plane(name, tuple(corner), tuple(across[0]),
                                 tuple(across[1]), 0.125, group, unit='in')
    window._refresh_item(window._item_for_object(name), window.objects[name])
    window.render_current()
    pick(window, [window._item_for_object(name)])
    show_scene(window)
    shoot(window, 'fem-planes')

    # one bolt's patch picked in the Quads table, Tie pressed, the part
    # below chosen from its menu; the other fifteen tied the same way
    geometry = window.objects[name]
    quads = family_row(window, name, 'Quads')
    window._tree_item_clicked(quads, 1)
    pump(20)
    bolts = barc.BOLTS
    patch, below = barc.bolt_patch(geometry, bolts[0])
    model = window.table.model()
    window.table.clearSelection()
    for element in patch:
        row = int(np.flatnonzero(geometry.elem_id == element)[0])
        window.table.selectionModel().select(
            model.index(row, 0),
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows)
    pump(20)
    tie_button = next(bar.widgetForAction(window.tie_action)
                      for bar in window.findChildren(QToolBar)
                      if window.tie_action in bar.actions())
    menus = []
    window._pop_menu = menus.append
    window.tie_action.trigger()
    pump(10)
    menu = menus[0]
    # the menu as it pops up under the button, a child of the window so
    # the grab holds it
    menu.setParent(window, Qt.WindowType.Widget)
    menu.adjustSize()
    menu.move(tie_button.mapTo(window, QPoint(tie_button.width() + 24,
                                              tie_button.height() + 16)))
    menu.show()
    menu.setActiveAction(next(a for a in menu.actions() if a.text() == below))
    pump(10)
    table_width(window, 460)
    show_scene(window)
    shoot(window, 'fem-tie-click', marks=[quads, tie_button, menu])
    menu.hide()
    menu.deleteLater()
    del window._pop_menu
    for bolt in bolts:
        patch, below = barc.bolt_patch(geometry, bolt)
        window._tie(name, patch, below, below)
    pump(20)
    window.stop_editing()
    pump(10)

    # the properties, typed into the Element Groups table
    for part, thickness in (('box', 0.25), ('right channel', 0.125),
                            ('left channel', 0.125), ('beam', 0.125)):
        set_group(window, name, part, Material='6061-T6')
        set_group(window, name, part, Thickness=f'{thickness}')
    window.stop_editing()
    row = group_row(window, name, 'box')
    window._tree_item_clicked(row, 1)
    pump(20)
    # the row unselected, so its cells read as typed rather than as
    # the selection's highlight
    window.table.clearSelection()
    table_width(window, 460)
    show_scene(window)
    shoot(window, 'fem-groups', marks=[row])
    window.stop_editing()
    pump(10)

    item = window._item_for_object(name)
    pick(window, [item])
    show_scene(window)
    panel = open_modes(window)
    shoot(window, 'fem-solve-click',
          marks=[item, act(window, 'Modes'), panel.solve_button])
    solve(window, 2000.0)
    shapes = by_type(window, 'ShapeSet')
    window.project.animate(shapes, mode=6,
                           screenshot=str(OUT / 'fem-mode.png'))
    print('  fem-mode.png', flush=True)
    close(window)

    # ---- the BARC from bricks
    window = fresh_window()
    window.tree.clearSelection()
    window.tree.setCurrentItem(window.test_item)
    window.test_item.setSelected(True)
    pump(10)
    window.new_geometry_act()
    pump(20)
    upright(window, name, barc.VIEW)
    blocks = [*barc.BOX_SOLIDS,
              *[(c, w, side if k == 0 else '')
                for side, pair in barc.CHANNEL_SOLIDS.items()
                for k, (c, w) in enumerate(pair)],
              (*barc.BEAM_SOLID, 'beam')]
    for k, (center, widths, label) in enumerate(blocks):
        pick(window, [family_row(window, name, 'Solids')])
        window.add_solid_block_act()
        pump(10)
        panel = window.scene.mesh_panel
        panel.set_values(center=center, widths=widths, group=label)
        pump(10)
        if k == 0:
            show_scene(window)
            shoot(window, 'fem-bricks-block-click',
                  marks=[family_row(window, name, 'Solids'),
                         act(window, 'Add Solid Block'), panel.add_button])
        panel.add_button.click()
        pump(10)
    window.close_mesh_pane()
    pump(10)
    # the box's three blocks, then each channel's two, subtracted
    rows = tree_rows(window, 'solid', name)
    pick(window, rows[:3])
    show_scene(window)
    # a click on the first, a Shift-click on the last
    shoot(window, 'fem-bricks-subtract-click',
          marks=[rows[0], rows[2], act(window, 'Subtract')])
    bar_act(window, 'subtract_solids')
    for side in ('right channel', 'left channel'):
        first = solid_row(window, name, side)
        following = tree_rows(window, 'solid', name)
        after = following[following.index(first) + 1]
        pick(window, [first, after])
        bar_act(window, 'subtract_solids')
    for row in tree_rows(window, 'solid', name):
        row.setExpanded(False)
    parts = top_solids(window, name)
    pick(window, parts)
    window.mesh_solids_act()
    pump(20)
    meshing = window.scene.mesh_solids_panel
    meshing.size.setValue(0.125)
    meshing.parts_apart.setChecked(True)
    pump(20)
    show_scene(window)
    shoot(window, 'fem-bricks-mesh-click',
          marks=[parts[0], parts[-1], act(window, 'Mesh Solids'),
                 meshing.parts_apart,
                 meshing.mesh_button])
    meshing.mesh_button.click()
    pump(30)

    # the right foot's joint: the patch where the foot rests on the box
    # found, the foot's outer ring clicked out of the box's side, the 49
    # nodes under the washers kept rigid (the demonstration's own set)
    geometry = window.objects[name]
    rows = [group_row(window, name, 'box'),
            group_row(window, name, 'right channel')]
    pick(window, rows)
    bar_act(window, 'mate_surfaces')
    mating = window.scene.mate_panel
    places = sorted({node for (x, z), kind in barc.SOLID_BOLTS
                     if kind == 'foot' and x > 0
                     for node in barc.bolt_places(geometry, (x, z), 'foot')})
    box = int(geometry.group_id[list(geometry.group_name).index('box')])
    clamped = set(barc._beneath(geometry, box, places))
    assert mating.side == 'a', 'the box is not the side picked first'
    for node in list(window._mating['patches'][0]['nodes_a']):
        if node not in clamped:
            window.toggle_mate_node(node)
    assert len(window._mating['patches'][0]['nodes_a']) == 49
    mating.name.setText('right foot bolts')
    pump(20)
    show_scene(window, look=('mate-a', 2.5))
    shoot(window, 'fem-bricks-mate-click',
          marks=[*rows, act(window, 'Mate Surfaces'), mating.name,
                 mating.keep_button])
    mating.keep_button.click()
    pump(20)
    window.close_mate_pane()
    pump(10)
    kept = next(m.id for m in geometry.mates if m.name == 'right foot bolts')
    joint = next(r for r in tree_rows(window, 'mate', name)
                 if r.data(0, ROLE_REFERENCE)[2] == kept)
    joint.parent().setExpanded(True)
    pick(window, [joint])
    show_scene(window, look=('mate-a', 2.5))
    shoot(window, 'fem-bricks-mated')

    # the joint opened again by its pencil, made a stiffness along the
    # normal; shot before Save, which the page leaves to the reader
    window.edit_mated_surfaces_act(name, kept)
    pump(20)
    mating.normal_mode.setCurrentIndex(mating.normal_mode.findData('stiffness'))
    mating.normal.setValue(50000.0)
    pump(20)
    show_scene(window, look=('mate-a', 2.5))
    shoot(window, 'fem-bricks-joint-click',
          marks=[joint, mating.normal_mode, mating.normal, mating.keep_button])
    window.close_mate_pane()
    pump(10)
    close(window)


def solid_by_id(window, name, solid):
    """Geometry `name`'s row for solid `solid`, its id."""
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    return next(row for row in tree_rows(window, 'solid', name)
                if row.data(0, ROLE_REFERENCE)[2] == solid)


def add_solid(window, name, verb, **values):
    """Add one solid through its pane, opened from the Solids row:
    `verb` 'add_solid_block' or 'add_solid_cylinder', `values` the
    fields typed. Returns the new solid's id and the open pane."""
    pick(window, [family_row(window, name, 'Solids')])
    bar_act(window, verb)
    panel = window.scene.mesh_panel
    panel.set_values(**values)
    pump(10)
    return panel


def pattern(window, name, solid, counts, spacing):
    """Pattern solid `solid` through the Pattern pane; returns the
    pattern's id and the open pane, before Pattern is pressed."""
    pick(window, [solid_by_id(window, name, solid)])
    bar_act(window, 'pattern_solids')
    panel = window.scene.pattern_panel
    panel.set_values(counts=counts, spacing=spacing)
    pump(10)
    return panel


def fem_solids():
    """The Frame-Wing page: the frame drawn as solids, patterned, cut,
    meshed, given its materials and solved; then a wing bolted on by
    mating the meshes where they meet."""
    from visualdynamics.demo import frame_wing

    print('fem_solids:', flush=True)
    window = fresh_window()
    window.tree.clearSelection()
    window.tree.setCurrentItem(window.test_item)
    window.test_item.setSelected(True)
    pump(10)
    window.new_geometry_act()
    pump(20)
    name = window.rename_object('Geometry', 'Frame')
    pump(10)
    upright(window, name, frame_wing.VIEW)
    geometry = window.objects[name]

    def added(panel):
        panel.add_button.click()
        pump(10)
        return geometry.solids[-1].id

    plate = added(add_solid(window, name, 'add_solid_block',
                            center=(0, 0, -0.25), widths=(16, 6, 0.5),
                            group='frame'))
    panel = add_solid(window, name, 'add_solid_block',
                      center=(-5.8125, 0, -0.25), widths=(3.375, 5, 1),
                      corner=0.138, group='')
    show_scene(window)
    shoot(window, 'fem-solids-block-click',
          marks=[family_row(window, name, 'Solids'),
                 act(window, 'Add Solid Block'), panel.add_button])
    opening = added(panel)
    panel = add_solid(window, name, 'add_solid_cylinder',
                      center=(-7.75, -2.75, -0.25), radius=0.1451, length=1.0,
                      group='')
    show_scene(window)
    shoot(window, 'fem-solids-cylinder-click',
          marks=[family_row(window, name, 'Solids'),
                 act(window, 'Add Cylinder'), panel.add_button])
    rail = added(panel)
    upright_hole = added(add_solid(window, name, 'add_solid_cylinder',
                                   center=(-7.75, -1.375, -0.25),
                                   radius=0.1451, length=1.0, group=''))
    window.close_mesh_pane()
    pump(10)

    patterns = []
    for solid, counts, spacing in ((opening, (4, 1, 1), (3.875, 0, 0)),
                                   (rail, (13, 2, 1), (1.291667, 5.5, 0)),
                                   (upright_hole, (5, 3, 1), (3.875, 1.375, 0))):
        panel = pattern(window, name, solid, counts, spacing)
        if solid == rail:
            show_scene(window)
            shoot(window, 'fem-solids-pattern-click',
                  marks=[solid_by_id(window, name, solid),
                         act(window, 'Pattern'), panel.pattern_button])
        panel.pattern_button.click()
        pump(10)
        patterns.append(geometry.solids[-1].id)
    for row in tree_rows(window, 'solid', name):
        row.setExpanded(False)
    rows = [solid_by_id(window, name, s) for s in (plate, *patterns)]
    pick(window, rows)
    show_scene(window)
    # a click on the plate, a Shift-click on the last pattern
    shoot(window, 'fem-solids-subtract-click',
          marks=[rows[0], rows[-1], act(window, 'Subtract')])
    bar_act(window, 'subtract_solids')
    frame = geometry.solids[-1].id

    # the inserts: the same two holes again, shorter, patterned and joined
    # by the verb each Add records: the pane's fields hold four
    # decimals, and -0.15625 typed there is -0.1563, which puts the
    # inserts' faces off the frame's grid and adds a layer of bricks
    inserts = [window.project.add_solid_cylinder(
        name, (x, y, -0.15625), 0.1451, 0.3125, name='inserts', unit='in')
        for x, y in ((-7.75, -2.75), (-7.75, -1.375))]
    window._solids_changed(name)
    joined = []
    for solid, counts, spacing in ((inserts[0], (13, 2, 1), (1.291667, 5.5, 0)),
                                   (inserts[1], (5, 3, 1), (3.875, 1.375, 0))):
        pattern(window, name, solid, counts, spacing).pattern_button.click()
        pump(10)
        joined.append(geometry.solids[-1].id)
    pick(window, [solid_by_id(window, name, s) for s in joined])
    bar_act(window, 'union_solids')
    insert_part = geometry.solids[-1].id
    for row in tree_rows(window, 'solid', name):
        row.setExpanded(False)
    pick(window, [family_row(window, name, 'Solids')])
    show_scene(window)
    shoot(window, 'fem-solids-frame')

    rows = [solid_by_id(window, name, s) for s in (frame, insert_part)]
    pick(window, rows)
    bar_act(window, 'mesh_solids')
    meshing = window.scene.mesh_solids_panel
    meshing.size.setValue(0.083334)
    pump(20)
    show_scene(window)
    shoot(window, 'fem-solids-mesh-click',
          marks=[*rows, act(window, 'Mesh Solids'), meshing.mesh_button])
    meshing.mesh_button.click()
    pump(30)
    pick(window, [window._item_for_object(name)])
    family_row(window, name, 'Hexes').setExpanded(True)
    show_scene(window)
    shoot(window, 'fem-solids-mesh')

    set_group(window, name, 'frame', Material='6061-T6')
    window.stop_editing()
    set_group(window, name, 'inserts', E='1.1216e6')
    set_group(window, name, 'inserts', ρ='3.6175e-4', ν='0.3')
    window.stop_editing()
    row = group_row(window, name, 'inserts')
    window._tree_item_clicked(row, 1)
    pump(20)
    window.table.clearSelection()
    table_width(window, 460)
    show_scene(window)
    shoot(window, 'fem-solids-groups', marks=[row])
    window.stop_editing()
    pump(10)

    pick(window, [window._item_for_object(name)])
    solve(window, 1500.0)
    shapes = by_type(window, 'ShapeSet')
    window.project.animate(shapes, mode=6,
                           screenshot=str(OUT / 'fem-solids-mode.png'))
    print('  fem-solids-mode.png', flush=True)

    # the thick wing on the frame, built by the demonstration's own
    # steps up to the bolting, which is done here in the window
    wing = 'Thick Wing on Frame'
    window.project.new_geometry(wing, unit='in')
    frame_wing._frame(window.project, wing, screwed=True, size=frame_wing.SIZE)
    frame_wing._wing(window.project, wing, 'thick wing', washers=True,
                     size=frame_wing.WING_SIZE)
    frame_wing._materials(window.project[wing])
    window.project[wing].view = frame_wing.VIEW
    window.show_object(wing)
    pump(20)
    window._item_for_object(name).setExpanded(False)
    rows = [group_row(window, wing, g)
            for g in ('frame', 'inserts', 'screws', 'washers')]
    pick(window, rows)
    bar_act(window, 'mate_surfaces')
    mating = window.scene.mate_panel
    listed = mating.patches
    assert listed.count() == 8, f'{listed.count()} patches, not eight'
    listed.selectAll()
    pump(20)
    show_scene(window)
    first, last = listed.item(0), listed.item(listed.count() - 1)
    shoot(window, 'fem-solids-mate-click',
          marks=[*rows, act(window, 'Mate Surfaces'),
                 (listed.viewport(), listed.visualItemRect(first)),
                 (listed.viewport(), listed.visualItemRect(last)),
                 mating.keep_button])
    mating.keep_button.click()
    pump(20)
    window.close_mate_pane()
    pump(10)
    geometry = window.objects[wing]
    assert len(geometry.mates) == 8
    joint = tree_rows(window, 'mate', wing)[0]
    joint.parent().setExpanded(True)
    pick(window, [joint])
    show_scene(window, look=('mate-a', 16.0))
    shoot(window, 'fem-solids-mated')
    close(window)


def plan():
    """The Plan Instrumentation page: the BARC's solved bricks, the act on
    the shape set's bar, then the pane with the box's thin edges left
    out and the planned sensors on the rest."""
    import tempfile

    from visualdynamics.demo import barc

    print('plan:', flush=True)
    window = fresh_window()
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, 'BARC.vdyn')
        barc.solid_project(solved=True).save(path)
        window.import_paths([path])
        pump(60)
    shapes = next(name for name, obj in window.objects.items()
                  if type(obj).__name__ == 'ShapeSet' and 'Removable' not in name)
    pick(window, [window._item_for_object(shapes)])
    # the auto-MAC flat: the offscreen window cannot draw its 3-D bars
    if window.mac_bars_action.isChecked():
        window.mac_bars_action.trigger()
        pump(10)
    shoot(window, 'plan-click', marks=[shapes, act(window, 'Plan Instrumentation')])
    bar_act(window, 'plan_instrumentation')
    panel = window.scene.instrumentation_panel
    panel.triaxial.setValue(6)
    panel.uniaxial.setValue(4)
    # the box's thin edges, too narrow for an accelerometer: the rims of
    # its open ends and its two thin sides, each a quarter inch wide.
    # Narrow is read as area over the outline's half-perimeter, since a
    # rim is a ring whose outline spans the whole box
    import numpy as np

    from visualdynamics.core.instrumentation import surface_frame

    geometry = window.objects['BARC']
    names = dict(zip((int(g) for g in geometry.group_id), geometry.group_name))
    rows = {int(n): k for k, n in enumerate(geometry.node_id)}
    _angle, found = window._planning_skin()
    for surface in found:
        if str(names[surface.group]).strip() != 'box':
            continue
        points = geometry.node_xyz[[rows[n] for f in surface.faces for n in f]]
        across = points @ surface_frame(surface.normals.mean(axis=0))[:2].T
        narrow = surface.area / np.ptp(across, axis=0).sum() < 0.6 * 0.0254
        # the smallest of them start out left out already
        if narrow and surface.id not in window._planning['off']:
            window.toggle_plan_surface(surface.id)
    window._plan_now()
    pump(20)
    show_scene(window)
    shoot(window, 'plan-pane', marks=[panel.keep_button])
    close(window)


def report_features():
    """The report editor with the Test report preset chosen: the Report
    Features panel in the pane, and a control figure on the page drawn
    under the plot conventions.

    The page is a web view, and web content paints into an offscreen
    grab only when Chromium draws without a GPU: run with
    QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu set before the application
    starts. The view is nudged by a pixel so it paints again before the
    grab."""
    import time

    print('report_features:', flush=True)
    window = fresh_window()
    window.resize(WINDOW[0], 1100)
    window.import_paths([str(PLATE / 'random.nc4')])
    pump(30)
    window.work_up()
    pump(30)
    report = by_type(window, 'Report')
    select(window, report)
    window.render_current()
    pump(60)
    editor = window.report_editor
    panel = editor.features_panel
    panel.preset_box.setCurrentIndex(panel.preset_box.findData('test_report'))
    # room for the panel's reasons beside the page
    editor.split.setSizes([760, 440])
    deadline = time.monotonic() + 30.0
    while editor.page.isLoading() and time.monotonic() < deadline:
        pump(5)
        time.sleep(0.05)
    pump(60)
    # the page opens at its top; the first control figure is the one
    # the conventions change, so it is brought into view
    editor.page.runJavaScript(
        "const grid = document.querySelector('.grid');"
        "if (grid) grid.closest('section').scrollIntoView();")
    for _ in range(3):
        size = editor.view.size()
        editor.view.resize(size.width() + 1, size.height())
        pump(5)
        editor.view.resize(size)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            pump(5)
            time.sleep(0.05)
    editor.grab().save(str(OUT / 'report-features.png'))
    print('  report-features.png', flush=True)
    close(window)


def main(which=('modal', 'random', 'transient', 'shock', 'sine', 'sysid',
                'mixed', 'tree', 'fem', 'fem_solids', 'plan',
                'report_features')):
    import traceback

    from PySide6.QtWidgets import QApplication

    import visualdynamics  # noqa: F401 — pins the Qt binding first

    QApplication.instance() or QApplication(['visualdynamics'])
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (modal, random, transient, shock, sine, sysid, mixed, tree,
                  fem, fem_solids, plan, report_features):
        if build.__name__ not in which:
            continue
        try:
            build()
        except Exception:  # noqa: BLE001 — one broken walkthrough
            # must not cost the other three their screenshots
            traceback.print_exc()
            print(f'{build.__name__}: FAILED', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    # every workflow unless some are named: the default once stopped
    # at four and the sine and system-identification pages quietly
    # kept their old pictures (2026-09-03)
    picked = tuple(a.lstrip('-') for a in sys.argv[1:])
    main(picked or ('modal', 'random', 'transient', 'shock', 'sine', 'sysid',
                    'mixed', 'tree', 'fem', 'fem_solids', 'plan',
                    'report_features'))
