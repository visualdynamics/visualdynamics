"""The OpenGL check and VTK wait for the first 3-D view.

The launch imports no VTK and runs no OpenGL check: the main window
comes up without either, and a window that never shows a 3-D view never
pays for them. The first 3-D view, wherever it is asked for (a project
opened after launch, the stage, the MAC bars, a script's window), runs
the check once, before anything of VTK is built (`gui.opengl.ready`,
through `gui.opengl.three_d_view`). A machine that fails it is told why,
once, and every 3-D view is refused while the rest of the window works.

What the launch imports is read in a fresh interpreter, since the
suite's own process has imported VTK long before any test here runs.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.gui import opengl

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: the reason a machine with only Windows' own OpenGL is given
REFUSED = opengl.judge('1.1.0', 'GDI Generic')

# The real launcher (`gui.main`) in a fresh interpreter, with the check
# and its message box replaced by recorders and the event loop by a
# function that reads the window and, given a project, opens it the way
# File > Open does after launch.
_LAUNCH = '''
import json, os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from visualdynamics import gui
from visualdynamics.gui import opengl

ROOTS = {'vtk', 'vtkmodules', 'pyvista', 'pyvistaqt'}


def vtk_loaded():
    return sorted(m for m in sys.modules if m.split('.')[0] in ROOTS)


reason, path = json.loads(sys.argv[1])
asked, said, result = [], [], {}


def missing():
    asked.append(bool(vtk_loaded()))     # was VTK in before the check?
    return reason


opengl.missing = missing
opengl.say = said.append


def run(app):
    from visualdynamics.gui.main_window import MainWindow

    window = next(w for w in app.topLevelWidgets()
                  if isinstance(w, MainWindow))
    app.processEvents()
    result['shown'] = window.isVisible()
    result['vtk_at_launch'] = vtk_loaded()
    result['asked_at_launch'] = len(asked)
    if path:
        window.import_path(path)
        for _ in range(5):
            app.processEvents()
        plotter = window.scene.built_plotter
        result['view'] = None if plotter is None else type(plotter).__name__
        result['actors'] = (0 if plotter is None
                            else len(plotter.renderer.actors))
        result['page'] = '' if plotter is not None else window.scene._page.text()
        result['vtk_after'] = bool(vtk_loaded())
    result['asked'] = asked
    result['said'] = said
    window.close()
    return 0


QApplication.exec = run
code = gui.main(['--no-disclaimer'])
print('RESULT ' + json.dumps(result))
sys.exit(code)
'''


def _launch(tmp_path, reason=None, path=None):
    """What the launched window read, from a fresh interpreter."""
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen',
               PYVISTA_OFF_SCREEN='true',
               VISUALDYNAMICS_SETTINGS=str(tmp_path / 'preferences.ini'))
    result = subprocess.run(
        [sys.executable, '-c', _LAUNCH, json.dumps([reason, path])],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
        check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(line for line in result.stdout.splitlines()
                if line.startswith('RESULT '))
    return json.loads(line[len('RESULT '):])


@pytest.fixture
def plate(tmp_path):
    """A project holding one geometry, saved as a .vdyn: opening it
    selects the geometry, which draws it in the scene."""
    project = visualdynamics.Project('Plate')
    project.import_file(fixture_path('plate', 'test_geometry.npz'))
    project.geometry.define_units('m')
    return project.save(tmp_path / 'plate.vdyn')


@pytest.fixture
def fresh_check(monkeypatch):
    """The check not yet run in this process, its asks counted and its
    message recorded instead of shown; the test sets the answer."""
    asked, said = [], []
    answer = {'reason': None}

    def missing():
        asked.append(True)
        return answer['reason']

    monkeypatch.setattr(opengl, '_found', [])
    monkeypatch.setattr(opengl, 'missing', missing)
    monkeypatch.setattr(opengl, 'say', said.append)
    return answer, asked, said


# ---- the launch -----------------------------------------------------------


def test_the_launched_window_imports_no_vtk_and_runs_no_check(tmp_path):
    """The window is up, and nothing of VTK, PyVista or pyvistaqt is in
    the process, and the check has not been asked."""
    seen = _launch(tmp_path)
    assert seen['shown']
    assert seen['vtk_at_launch'] == [], 'VTK imported at launch'
    assert seen['asked_at_launch'] == 0, 'the check ran at launch'
    assert seen['asked'] == []


def test_a_project_opened_after_launch_is_checked_before_its_3d_view(
        tmp_path, plate):
    """The path a window that started without a 3-D view takes into one:
    the check runs once, before VTK is imported, and the scene is then
    built and drawn."""
    seen = _launch(tmp_path, path=plate)
    assert seen['vtk_at_launch'] == [] and seen['asked_at_launch'] == 0
    assert seen['asked'] == [False], 'asked once, before any VTK'
    assert seen['view'] == 'QtInteractor'
    assert seen['actors'] > 0, 'the geometry was drawn'
    assert seen['said'] == []


def test_a_failing_check_on_opening_says_why_and_builds_nothing(
        tmp_path, plate):
    """The machine is told the check's own reason, once; no VTK is
    imported, and the scene's page says why there is no 3-D view."""
    seen = _launch(tmp_path, reason=REFUSED, path=plate)
    assert seen['asked'] == [False]
    assert seen['said'] == [REFUSED]
    assert seen['view'] is None
    assert seen['vtk_after'] is False, 'VTK imported on a refused machine'
    assert seen['page'] == REFUSED


# ---- every 3-D view in a session -----------------------------------------


def _add_run(window):
    """A time history, the kind of selection the stage draws in 3-D."""
    t = np.arange(2048) / 1024.0
    window.add_object('Run', visualdynamics.TimeHistory(
        t, np.random.default_rng(1).standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration'))


def _select(window, pump, name):
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()


def test_the_check_runs_once_before_the_first_of_several_3d_views(
        window, pump, fresh_check):
    """No 3-D view and no ask while nothing 3-D is shown; the scene asks
    once; the stage and the MAC bars built after it do not ask again."""
    _answer, asked, said = fresh_check
    window.render_current()
    pump()
    assert window.scene.built_plotter is None, 'built with nothing to draw'
    assert asked == []
    window.import_path(fixture_path('plate', 'test_geometry.npz'))
    pump()
    assert asked == [True], 'the first 3-D view asks'
    scene = window.scene.built_plotter
    assert scene is not None and len(scene.renderer.actors) > 0
    assert window.data_pane.create_waterfall_plotter() is not None
    assert window._mac_bars_plotter() is not None
    assert asked == [True], 'a later 3-D view asked again'
    assert said == []


def test_a_failing_check_blocks_every_3d_view_and_the_window_goes_on(
        window, pump, fresh_check):
    """Refused once, with the check's message; the scene keeps its page
    saying why, the stage's toggle is not offered and the data draws
    flat, the MAC bars are not built, and no slot raised on the way
    (the `window` fixture fails the test if one did)."""
    answer, asked, said = fresh_check
    answer['reason'] = REFUSED
    window.import_path(fixture_path('plate', 'test_geometry.npz'))
    pump()
    assert said == [REFUSED]
    assert window.scene.built_plotter is None
    assert window.scene._page.text() == REFUSED
    _add_run(window)
    _select(window, pump, 'Run')
    assert window.data_pane.waterfall_plotter is None, 'the stage was built'
    assert not window.data_pane.showing_waterfall
    assert not window.data_pane.waterfall_action.isVisible()
    assert window.data_pane.graphics.isVisible(), 'the flat plot stands'
    assert window._mac_bars_plotter() is None
    assert not window._mac_bars_wanted()
    assert asked == [True], 'asked again'
    assert said == [REFUSED], 'said again'


def test_with_the_check_passing_the_stage_is_built(window, pump, fresh_check):
    """The other side of the refusal: the same selection, with the check
    passing, builds the stage, so the test above is not passing on a
    selection that never asked for one."""
    _answer, asked, _said = fresh_check
    _add_run(window)
    _select(window, pump, 'Run')
    assert window.data_pane.waterfall_plotter is not None
    assert asked == [True]


# ---- a script's 3-D windows ----------------------------------------------


def test_a_scripts_3d_window_on_a_refused_machine_raises_the_reason(
        qt_app, fresh_check, monkeypatch):
    """`geometry.plot()` and the animation windows have no window to
    fall back on: the reason comes back as the error, said once, and no
    pane or VTK window is built."""
    from visualdynamics.gui import windows
    from visualdynamics.viz import animate

    answer, _asked, said = fresh_check
    answer['reason'] = REFUSED

    def built(*_args, **_kwargs):
        raise AssertionError('a 3-D window was built on a refused machine')

    monkeypatch.setattr(windows, 'ScenePane', built)
    monkeypatch.setattr('pyvistaqt.BackgroundPlotter', built)
    with pytest.raises(RuntimeError) as raised:
        windows.scene_window(lambda plotter: None, show=False)
    assert str(raised.value) == REFUSED
    with pytest.raises(RuntimeError) as raised:
        animate._play(None, None, colors={}, title='', show=False)
    assert str(raised.value) == REFUSED
    assert said == [REFUSED]


def test_the_movie_self_check_reports_a_refused_machine(fresh_check,
                                                       monkeypatch):
    """`--check-movie` renders off screen, so it asks too, and answers 2
    with the reason rather than building a VTK window."""
    from visualdynamics.gui import movie

    answer, _asked, said = fresh_check
    answer['reason'] = REFUSED
    monkeypatch.setattr(movie, 'unavailable_reason', lambda: None)
    assert movie.check('never-written.mp4') == (2, REFUSED)
    assert said == [REFUSED]
