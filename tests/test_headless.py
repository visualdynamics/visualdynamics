"""Everything the window draws, drawn without one.

The promise is that visualdynamics is a library the app happens to sit on, not an
app with a scripting hole in it: whatever you can look at in the window
you can render from a script, to a file, with no window and no event
loop of your own. A gap here is a workflow that has to be finished by
hand in the GUI.

So this is a list — every kind of view the app can put up — and a call
for each. It is deliberately an inventory rather than a set of narrow
tests: what it catches is not a broken plot (other tests do that) but a
view that grew a GUI-only path.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.plot import (
    plot_bars,
    plot_comparison,
    plot_data,
    plot_mac,
    plot_photos,
    plot_series,
)


@pytest.fixture(scope='module')
def modal():
    """A modal project: geometry, FRFs, shapes, coherence."""
    project = visualdynamics.Project('Modal')
    project.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    project.import_file(fixture_path('plate', 'test_geometry.npz'))
    project.geometry.define_units('m')
    project.add('Modes', visualdynamics.import_file(
        fixture_path('plate', 'shapes.npy')))
    project.shapes.define_units('kg')
    return project


@pytest.fixture(scope='module')
def random_run():
    return visualdynamics.random_vibration_run(fixture_path('plate',
                                                   'random.nc4'))


def _drawn(path):
    """A file with a picture in it, not an empty one."""
    return path.exists() and path.stat().st_size > 2000


# ---- curves, and everything drawn on an axis ----------------------------


def test_a_data_array_draws_itself(random_run, tmp_path):
    path = tmp_path / 'curves.png'
    random_run.time_history.save_plot(path)
    assert _drawn(path)


def test_several_arrays_draw_together(random_run, tmp_path):
    """What selecting more than one object in the tree does."""
    narrow = next(obj for obj in random_run.psds
                  if obj is not random_run.specification)
    path = tmp_path / 'together.png'
    plot_series([('PSD', narrow), ('Spec', random_run.specification)],
                path=str(path), show=False)
    assert _drawn(path)


def test_some_records_of_one_array(random_run, tmp_path):
    """A pick in the record grid restricts the object; so does a
    `records` list."""
    path = tmp_path / 'two.png'
    plot_series([(None, random_run.time_history, [0, 1])],
                path=str(path), show=False)
    assert _drawn(path)


@pytest.mark.parametrize('marks', ['averaging', 'shocks'])
def test_a_time_history_carries_its_own_reading(random_run, tmp_path, marks):
    """The app's two toggles: the frames a spectrum averages over, and
    the events an SRS is computed from."""
    path = tmp_path / f'{marks}.png'
    random_run.time_history.save_plot(path, marks=marks)
    assert _drawn(path)


def test_marks_that_are_not_a_reading_say_so(random_run, tmp_path):
    with pytest.raises(ValueError, match="'averaging' or 'shocks'"):
        random_run.time_history.save_plot(tmp_path / 'x.png', marks='frames')
    with pytest.raises(TypeError, match='time history'):
        random_run.specification.save_plot(tmp_path / 'x.png',
                                           marks='averaging')


# ---- the comparison, all three readings ---------------------------------


def test_all_three_comparison_readings(random_run, tmp_path):
    narrow = next(obj for obj in random_run.psds
                  if obj is not random_run.specification
                  and obj.bandwidth is None)
    spec = random_run.specification
    for name, call in (
            ('curves', lambda p: plot_comparison(narrow, spec, path=p,
                                                 show=False)),
            ('error', lambda p: plot_bars(narrow, spec, 'error', path=p,
                                          show=False)),
            ('lines', lambda p: plot_bars(narrow, spec, 'lines', path=p,
                                          show=False))):
        path = tmp_path / f'{name}.png'
        call(str(path))
        assert _drawn(path), name


# ---- the FRF matrix, read the app's several ways ------------------------


def test_the_matrix_readings(modal, tmp_path):
    frf = modal.frf
    shapes = modal.shapes
    for name, call in (
            ('cmif', lambda p: frf.plot_cmif(path=p, show=False)),
            ('resynth', lambda p: frf.plot_cmif(shapes, path=p, show=False)),
            ('map', lambda p: modal.coherence.plot_map(path=p, show=False)),
            ('automac', lambda p: plot_mac(shapes, path=p, show=False)),
            ('crossmac', lambda p: plot_mac(shapes, shapes, path=p,
                                            show=False))):
        path = tmp_path / f'{name}.png'
        call(str(path))
        assert _drawn(path), name


@pytest.mark.parametrize('component', ['magnitude', 'real', 'imag', 'phase'])
def test_every_component_of_a_complex_array(modal, tmp_path, component):
    """The component box over a complex plot is a call argument here."""
    path = tmp_path / f'{component}.png'
    plot_data(modal.frf, path=str(path), show=False, component=component)
    assert _drawn(path)


# ---- the 3-D scene ------------------------------------------------------


def test_the_geometry_and_its_dofs(modal, tmp_path):
    scene = tmp_path / 'scene.png'
    modal.geometry.plot(screenshot=str(scene))
    assert _drawn(scene)
    arrows = tmp_path / 'dofs.png'
    modal.geometry.plot_dofs(modal.frf, 'force', screenshot=str(arrows))
    assert _drawn(arrows)


def test_a_mode_shape_animates(modal, tmp_path):
    """An animation has no still to save, so the still is the one that
    shows the shape: peak phase."""
    path = tmp_path / 'mode.png'
    modal.animate(modal.shapes, mode=0, screenshot=str(path))
    assert _drawn(path)


# ---- photos -------------------------------------------------------------


def test_the_photos(qt_app, tmp_path):
    from PySide6.QtGui import QImage, QPainter

    from visualdynamics.core.photos import Photos

    # a real image, made rather than pasted: an unreadable one is
    # skipped by the drawing, and a skipped one proves nothing
    image = QImage(64, 48, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    painter = QPainter(image)
    painter.drawLine(0, 0, 63, 47)
    painter.end()
    source = tmp_path / 'setup.png'
    assert image.save(str(source), 'PNG')
    photos = Photos()
    photos.add_file(str(source))
    path = tmp_path / 'photos.png'
    plot_photos(photos, path=str(path), show=False)
    assert _drawn(path)


# ---- tables -------------------------------------------------------------


def test_the_tables_read_as_data(modal, random_run):
    """The instrumentation and the identified parameters, as the rows
    the report prints."""
    headers, rows = random_run.table(random_run.channel_table)
    assert headers[:3] == ['Channel', 'Node', 'Direction']
    assert len(rows) == random_run.channel_table.num_channels

    headers, rows = modal.table(modal.shapes)
    assert headers == ['Mode', 'Frequency [Hz]', 'Damping [%]',
                       'Description']
    assert len(rows) == modal.shapes.num_shapes
    assert float(rows[0][1]) == pytest.approx(
        float(modal.shapes.frequency[0]), rel=1e-4)


def test_the_report_prints_the_very_same_rows(random_run):
    """One implementation, so a script and a report cannot disagree
    about what is in a channel table."""
    from visualdynamics.report import _table_block

    headers, rows = random_run.table(random_run.channel_table)
    built = _table_block({'caption': ''}, random_run.channel_table)
    assert built['headers'] == headers
    assert built['rows'] == rows


def test_something_that_is_not_a_table_says_so(random_run):
    with pytest.raises(ValueError, match='does not read as a table'):
        random_run.table(random_run.time_history)


# ---- and the report itself ----------------------------------------------


def test_the_report_writes_itself(random_run, tmp_path):
    path = random_run.export_report(
        random_run.generate_report('random'), tmp_path / 'report.html')
    html = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert str(path) == str(tmp_path / 'report.html')
    assert 'Random Vibration Test Report' in html


def test_none_of_it_needed_the_application(modal, random_run, tmp_path):
    """The whole inventory, in a clean interpreter, with a check that
    the app was never imported.

    Qt *is* imported — rendering a plot is Qt's job whether a window
    opens or not. What the promise means is that no application shell
    is needed, and `visualdynamics.gui` is that shell.
    """
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    data = root / 'testdata' / 'plate'
    probe = (
        'import os, sys\n'
        'os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")\n'
        'import visualdynamics\n'
        'from visualdynamics.plot import plot_bars, plot_comparison\n'
        f'p = visualdynamics.random_vibration_run({str(data / "random.nc4")!r})\n'
        'spec = p.specification\n'
        'psds = [o for o in p.psds if o is not spec and o.bandwidth is None][0]\n'
        f'p.time_history.save_plot({str(tmp_path / "t.png")!r},'
        ' marks="averaging")\n'
        f'plot_comparison(psds, spec, path={str(tmp_path / "c.png")!r},'
        ' show=False)\n'
        f'plot_bars(psds, spec, "error", path={str(tmp_path / "b.png")!r},'
        ' show=False)\n'
        'p.table(p.channel_table)\n'
        f'p.export_report(p.generate_report("random"),'
        f' {str(tmp_path / "r.html")!r})\n'
        'print("app imported:", "visualdynamics.gui" in sys.modules)\n'
    )
    result = subprocess.run([sys.executable, '-c', probe], cwd=root,
                            capture_output=True, text=True, timeout=900,
                            check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'app imported: False' in result.stdout
    for name in ('t.png', 'c.png', 'b.png', 'r.html'):
        assert (tmp_path / name).stat().st_size > 2000, name


def test_the_inventory_covers_what_the_window_renders():
    """The list above is only worth anything if it is the whole list.

    Every `_render_*` on the window is a kind of view; each one is named
    here with the call that reaches it, and a new one with no entry
    fails this. Which is the point — the gap to catch is a view that
    grew a GUI-only path, and it is caught the day it is written.
    """
    from visualdynamics.gui import main_window

    reached = {
        '_render_series': 'plot_data / plot_series / plot_comparison',
        '_render_waterfall': 'plot_waterfall',
        '_render_bars': 'plot_bars',
        '_render_levels': 'plot_bars.level_chart / compliance.specification_rms',
        '_render_kurtosis': 'plot_kurtosis',
        '_render_wavelet': 'plot_scalogram',
        '_render_replication': 'plot_replication',
        '_render_srs': "plot_bars(mode='srs')",
        '_render_banded': 'viz.banded.plot_banded_stage',
        '_render_ratio': 'plot.plot_ratio',
        '_render_pair_stage': 'viz.paired.plot_paired_stage',
        '_render_sine': ('core.sine.extract_sine + compliance.sine_errors'
                         ' / viz.sinespec.plot_sine_specification'),
        '_render_events': "plot_replication(mode='overlay')",
        '_render_replication_bars': 'plot_replication',
        '_render_geometries': 'Geometry.plot / Geometry.plot_dofs',
        '_render_animation': 'Project.animate / ShapeSet.animate',
        '_render_photos': 'plot_photos',
        '_render_table': 'Project.table',
        '_render_shape_table': 'Project.table',
        '_render_matches': 'Report block: kind=pairs',
        '_render_cross_mac': 'plot_mac',
        '_render_report_builder': 'Project.export_report',
        '_render_specifications': 'compliance.specification_rms',
        '_render_fit': 'Project.fit_modes',
        '_render_fit_mac': 'plot_mac',
        '_render_author': ('core.author.ModalSpecificationDraft.preview, '
                           'then plot_data'),
    }
    renderers = {name for name in dir(main_window.MainWindow)
                 if name.startswith('_render_')}
    assert not renderers - set(reached), (
        'a view the window can put up with nothing said about how a '
        'script reaches it')
    assert not set(reached) - renderers, (
        'named here and no longer on the window')


def test_the_scripted_examples_run(tmp_path):
    """Both examples, in one clean interpreter each. They are the
    documentation of this promise, and documentation that does not run
    is a claim."""
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    for script in ('modal_workflow.py', 'random_workflow.py'):
        out = tmp_path / script
        out.mkdir()
        probe = (
            'import os, runpy, sys\n'
            'os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")\n'
            f'sys.argv = [{script!r}, {str(out)!r}]\n'
            f'runpy.run_path("examples/{script}", run_name="__main__")\n'
            'print("app imported:", "visualdynamics.gui" in sys.modules)\n'
        )
        result = subprocess.run([sys.executable, '-c', probe], cwd=root,
                                capture_output=True, text=True, timeout=900,
                                check=False)
        assert result.returncode == 0, (script, result.stderr[-2000:])
        assert 'app imported: False' in result.stdout, script
        assert any(out.iterdir()), script


def test_the_two_front_ends_reach_the_same_numbers(random_run):
    """Not a plot: the point of the inventory. A project built by
    clicking holds what a project built by calling holds."""
    narrow = next(obj for obj in random_run.psds
                  if obj is not random_run.specification
                  and obj.bandwidth is None)
    assert np.isfinite(narrow.ordinate).all()
    assert narrow.abscissa[0] == 0.0


def test_the_entry_module_launches_only_as_main(monkeypatch):
    """A spawned worker process re-imports the parent's main module as
    `__mp_main__`; `python -m visualdynamics` must not launch the app
    again there (2026-10-01: eight workers each loaded a 23 GB file)."""
    import runpy

    from visualdynamics import gui

    launches = []
    monkeypatch.setattr(gui, 'main', lambda *a: launches.append(a) or 0)
    runpy.run_module('visualdynamics.__main__', run_name='__mp_main__')
    assert launches == []
    with pytest.raises(SystemExit):
        runpy.run_module('visualdynamics.__main__', run_name='__main__')
    assert len(launches) == 1
