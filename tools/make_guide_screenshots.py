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


def shoot(window, name):
    pump(20)
    path = OUT / f'{name}.png'
    window.grab().save(str(path))
    print(f'  {name}.png', flush=True)


CHROME = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')


def report_shot(window, name):
    """The exported report, screenshotted by headless Chrome.

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


def add_geometry_and_photos(window, which):
    """The two objects a controller file cannot bring: the article's
    geometry, and the setup photographs — taken here from the demo
    project so the report renders complete."""
    import visualdynamics

    window.import_paths([str(PLATE / 'geometry.npz')])
    pump(20)
    geometry = next(n for n, o in window.objects.items()
                    if type(o).__name__ == 'Geometry')
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
    print('modal:', flush=True)
    window = fresh_window()
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
    window.close_units_panel()
    window.objects[geometry].define_units('m')
    pump()

    # link the measured side and declare it the basis
    names = list(window.objects)
    select(window, *names)
    window.link_selected()
    pump()
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
    window.compute_psds()
    pump(20)
    shoot(window, 'modal-psds')

    select(window, frf)
    window.start_modal_fit()
    pump(20)
    shoot(window, 'modal-fit-open')
    for _ in range(2):
        window.confirm_fit_mode()
        pump(10)
    window.find_next_mode()
    pump(10)
    shoot(window, 'modal-fit')
    # three more confirms: five modes, the repeated pair's second
    # tooth included
    for _ in range(3):
        window.confirm_fit_mode()
        pump(10)

    fitted = by_type(window, 'ShapeSet')
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
    window.generate_report('modal')
    pump(30)
    report_shot(window, 'modal-report')
    close(window)


# ---- random ---------------------------------------------------------------

def random():
    print('random:', flush=True)
    window = fresh_window()
    window.import_paths([str(PLATE / 'random.nc4')])
    pump(30)
    history = by_type(window, 'TimeHistory')
    select(window, history)
    shoot(window, 'random-import')

    window.compute_psds()
    pump(20)
    shoot(window, 'random-psds')

    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd')
    select(window, psd)
    # the app sets the band spacing in its Octave Bands pane and Apply
    # Octave Bands calls this verb; its default is the same sixth-octave
    banded = window.project.compute_octave(psd)
    window.show_object(banded)
    pump(20)
    shoot(window, 'random-octave')

    select(window, history)
    window.compute_multiple_coherence()
    pump(20)
    shoot(window, 'random-coherence')

    spec = by_type(window, 'Specification')
    select(window, psd, spec)
    pump(20)
    shoot(window, 'random-comparison')

    add_geometry_and_photos(window, 'random')
    window.generate_report('random')
    pump(30)
    report_shot(window, 'random-report')
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
    window.import_paths([str(run)])
    pump(30)
    history = by_type(window, 'TimeHistory')
    spec = by_type(window, 'TransientSpecification')
    select(window, history)
    shoot(window, 'transient-import')

    select(window, spec)
    pump(20)
    shoot(window, 'transient-spec')

    select(window, history, spec, current=history)
    pump(20)
    shoot(window, 'transient-overlay')

    # the waveform-error reading of the same pair
    window.data_pane.replication_actions['waveform'].trigger()
    pump(20)
    shoot(window, 'transient-error')

    select(window, spec)
    window.compute_psds()
    pump()
    select(window, history)
    window.compute_psds()
    pump(20)
    measured = next(n for n, o in window.objects.items()
                    if type(o).__name__ == 'Psd')
    target = next(n for n, o in window.objects.items()
                  if type(o).__name__ == 'Specification'
                  and 'PSD' in n)
    select(window, measured, target)
    pump(20)
    shoot(window, 'transient-level')

    add_geometry_and_photos(window, 'transient')
    window.generate_report('transient')
    pump(30)
    report_shot(window, 'transient-report')
    close(window)


# ---- shock ----------------------------------------------------------------

def shock():
    print('shock:', flush=True)
    window = fresh_window()
    window.import_paths([str(PLATE / 'shock.nc4')])
    pump(30)
    # the file cannot say it was a shock test; the engineer does
    window.set_project_type('Shock')
    pump()
    history = by_type(window, 'TimeHistory')
    select(window, history)
    shoot(window, 'shock-import')

    window.compute_srs()
    pump(20)
    shoot(window, 'shock-srs')

    # the requirement band, authored the way the demo project authors it
    import numpy as np

    from visualdynamics.core.data import ShockSpecification

    srs = window.objects[by_type(window, 'Srs')]
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
    select(window, by_type(window, 'Srs'), 'Shock Specification')
    pump(20)
    shoot(window, 'shock-spec')

    add_geometry_and_photos(window, 'shock')
    window.generate_report('shock')
    pump(30)
    report_shot(window, 'shock-report')
    close(window)


def sine():
    run = STRESS / 'sine.nc4'
    if not run.exists():
        print('sine: skipped — stressdata/plate/sine.nc4 absent', flush=True)
        return
    print('sine:', flush=True)
    window = fresh_window()
    window.import_paths([str(run)])
    pump(30)
    # the file says what it is: a sine run arrives as a Sine Sweep
    # project, specification and all
    spec = by_type(window, 'SineSweepSpecification')
    select(window, spec)
    shoot(window, 'sine-spec')

    history = by_type(window, 'TimeHistory')
    select(window, history)
    pump(10)
    # the wavelet reading of the sweep: the four tones' trajectories,
    # drawn where they actually are in time
    window.data_pane.wavelet_action.trigger()
    pump(40)
    shoot(window, 'sine-wavelet')
    window.data_pane.wavelet_action.trigger()
    pump(10)

    window.extract_sine_levels()
    pump(30)
    levels = by_type(window, 'SineLevelSet')
    select(window, levels, spec)
    pump(20)
    shoot(window, 'sine-levels')

    add_geometry_and_photos(window, 'sine')
    window.generate_report('sine')
    pump(30)
    report_shot(window, 'sine-report')
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
    window.set_project_type('System ID')
    window.import_paths([str(stream)])
    pump(30)
    select(window, 'Excitation Time History')
    shoot(window, 'sysid-import')

    frfs = window.project.compute_frfs('Excitation Time History', 'H1')
    window.show_object(frfs)
    pump(30)
    shoot(window, 'sysid-frfs')

    coherence = window.project.compute_multiple_coherence(
        'Excitation Time History')
    window.show_object(coherence)
    pump(20)
    shoot(window, 'sysid-coherence')

    driven = window.project.compute_psds('Excitation Time History')
    window.project['Noise Time History'].averaging = \
        window.project['Excitation Time History'].averaging
    quiet = window.project.compute_psds('Noise Time History')
    # the project verbs put the results in the project; the window
    # catches up by being shown them, and only then can they be picked
    window.show_object(driven)
    window.show_object(quiet)
    pump(10)
    select(window, driven, quiet)
    pump(20)
    window.data_pane.spectra_actions['ratio'].trigger()
    pump(20)
    shoot(window, 'sysid-snr')

    add_geometry_and_photos(window, 'sysid')
    window.generate_report('sysid')
    pump(30)
    report_shot(window, 'sysid-report')
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


def main(which=('modal', 'random', 'transient', 'shock', 'sine', 'sysid',
                'tree')):
    import traceback

    from PySide6.QtWidgets import QApplication

    import visualdynamics  # noqa: F401 — pins the Qt binding first

    QApplication.instance() or QApplication(['visualdynamics'])
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (modal, random, transient, shock, sine, sysid, tree):
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
                    'tree'))
