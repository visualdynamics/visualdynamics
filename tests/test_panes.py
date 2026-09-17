"""The view panes on their own.

The point of a pane is that it does not need the window: it can be built,
shown and driven with nothing else alive, which is what lets the same
widget sit in the main window, in a dock torn off it, or in a window a
script opened. These tests never construct a MainWindow — if the pane
ever reaches back for the project, the tree or the selection, they stop
importing.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from visualdynamics.core.data import Frf
from visualdynamics.gui.panes import DataPane, ScenePane
from visualdynamics.plot import build_plots


@pytest.fixture
def pane(qt_app):
    pane = DataPane('light')
    pane.resize(600, 400)
    pane.show()
    qt_app.processEvents()
    yield pane
    pane.close()
    pane.deleteLater()
    qt_app.processEvents()


def test_the_pane_stands_up_with_no_window(pane):
    """Nothing but a QApplication: no MainWindow, no project."""
    assert pane.graphics is not None
    assert pane.plot_mode is None, 'the data chooses until someone clicks'


def test_the_pane_does_not_drag_the_window_in_behind_it():
    """In a clean interpreter, building a pane must not import the
    window — a back-import would make 'standalone' a fiction and pull
    the whole application in for a single plot."""
    probe = ('import os, sys\n'
             "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
             'from PySide6.QtWidgets import QApplication\n'
             'app = QApplication([])\n'
             'from visualdynamics.gui.panes import DataPane, ScenePane\n'
             "DataPane('light').show()\n"
             "ScenePane('light', offscreen=True).show()\n"
             "print('window imported:', "
             "'visualdynamics.gui.main_window' in sys.modules)\n")
    result = subprocess.run([sys.executable, '-c', probe],
                            capture_output=True, text=True, timeout=300,
                            check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'window imported: False' in result.stdout, result.stdout


def test_it_draws_without_anyone_holding_it(pane, qt_app):
    """A plot built straight into the pane's own layout."""
    frf = Frf(abscissa=np.linspace(0.0, 100.0, 64),
              ordinate=np.ones((2, 64), dtype=complex),
              response_dof=['101X+', '102X+'], reference_dof=['101X+'] * 2)
    drawn, _requested = build_plots(pane.graphics, [('FRF', frf, None)])
    qt_app.processEvents()
    assert drawn == 2
    assert any(hasattr(item, 'viewRange') for item in pane.graphics.ci.items)


def test_only_the_controls_the_data_can_use_are_shown(pane):
    pane.show_controls(map_wanted=None, diagonal=None, cmif=True,
                       complex_data=True, pair=False)
    assert pane.toolbar.isVisible()
    assert pane.cmif_action.isVisible()
    assert pane.component_action.isVisible()
    assert not pane.curves_action.isVisible(), 'no map choice for an FRF'
    assert not pane.drive_point_action.isVisible()
    assert not pane.pair_edit_action.isVisible()


def test_the_bar_goes_away_when_nothing_applies(pane):
    pane.show_controls(map_wanted=None, diagonal=None, cmif=False,
                       complex_data=False, pair=False)
    assert not pane.toolbar.isVisible(), (
        'a control that is not a live option is absent, not grayed')


def test_the_cmif_takes_the_component_box_with_it(pane):
    """Singular values have no real part to pick."""
    pane.cmif_action.setChecked(True)
    pane.show_controls(map_wanted=None, diagonal=None, cmif=True,
                       complex_data=True, pair=False)
    assert not pane.component_action.isVisible()


def test_the_diagonal_button_is_named_for_what_the_diagonal_is(pane):
    pane.show_controls(map_wanted=None, diagonal=('Autospectra', 'the ASDs',
                                                  True),
                       cmif=False, complex_data=False, pair=False)
    assert pane.drive_point_action.text() == 'Autospectra'
    assert pane.drive_point_action.isChecked(), 'derived from the selection'


def test_choosing_a_reading_asks_for_a_redraw_rather_than_drawing(pane):
    """The pane keeps the choice; what it means is someone else's."""
    seen = []
    pane.reread.connect(lambda: seen.append(pane.plot_mode))
    pane.map_action.trigger()
    assert seen == ['map']
    assert pane.plot_mode == 'map'
    pane.reset_plot_mode()
    assert pane.plot_mode is None


def test_the_diagonal_filter_is_reported_not_applied(pane):
    """Filtering moves the tree selection, which a plot cannot reach."""
    seen = []
    pane.drive_points_toggled.connect(seen.append)
    pane.drive_point_action.setChecked(False)
    pane.drive_point_action.trigger()
    assert seen == [True]


def test_the_pair_choice_is_reported_not_acted_on(pane):
    """Edit Fit opens a fitting session, which is not the pane's to open."""
    seen = []
    pane.pair_mode_chosen.connect(seen.append)
    pane.pair_edit_action.trigger()
    pane.pair_synthesis_action.trigger()
    assert seen == ['fit', 'overlay']


def test_the_fitting_screen_leaves_only_the_pair_choice(pane):
    pane.show_pair_controls(True)
    assert pane.toolbar.isVisible()
    assert pane.pair_edit_action.isVisible()
    assert pane.pair_edit_action.isChecked()
    for action in (pane.curves_action, pane.map_action,
                   pane.drive_point_action, pane.cmif_action):
        assert not action.isVisible()


# ---- the 3-D view -------------------------------------------------------


@pytest.fixture
def scene(qt_app):
    """Offscreen, so no VTK widget is ever created: what makes a pane
    safe to build and tear down mid-suite."""
    scene = ScenePane('light', offscreen=True)
    scene.resize(600, 400)
    scene.show()
    qt_app.processEvents()
    yield scene
    scene.close()
    scene.deleteLater()
    qt_app.processEvents()


def test_the_scene_stands_up_with_no_window(scene):
    assert scene.plotter is not None, 'offscreen builds its plotter at once'
    assert not scene.bounds_visible, 'the labeled box is clutter until asked'
    assert scene.orientation_visible, 'the triad is useful at a glance'


def test_the_annotations_are_the_panes_to_toggle(scene):
    """Both read the scene alone — nothing about the project decides
    whether a triad is drawn."""
    scene.bounds_action.setChecked(True)
    assert scene.bounds_visible
    scene.orientation_action.setChecked(False)
    assert not scene.orientation_visible


def test_a_theme_reaches_the_scene_background(scene):
    """An empty 3D view must match the theme, not VTK's white default."""
    scene.apply_theme('dark')
    dark = scene.plotter.background_color
    scene.apply_theme('light')
    assert scene.plotter.background_color != dark


def test_anyone_placing_it_can_add_to_its_bar(scene):
    """The window hangs its own project-reading actions here; so could
    a script."""
    from PySide6.QtGui import QAction

    before = len(scene.toolbar.actions())
    scene.toolbar.addAction(QAction('Mine', scene))
    assert len(scene.toolbar.actions()) == before + 1
    assert scene.bounds_action in scene.toolbar.actions(), 'its own survive'


def test_clearing_leaves_a_themed_empty_view(scene):
    scene.clear()
    assert scene.plotter is not None


# ---- a pane in a window of its own --------------------------------------


def test_a_scripted_plot_comes_up_in_the_apps_own_pane(qt_app):
    """`data.plot()` is the same widget the window uses, so a plot opened
    from a script wears the same bar."""
    from visualdynamics.gui.panes import DataPane

    frf = Frf(abscissa=np.linspace(0.0, 100.0, 64),
              ordinate=np.ones((2, 64), dtype=complex),
              response_dof=['101X+', '102X+'], reference_dof=['101X+'] * 2)
    pane = frf.plot(show=False)
    qt_app.processEvents()
    try:
        assert isinstance(pane, DataPane)
        assert any(hasattr(item, 'viewRange')
                   for item in pane.graphics.ci.items)
        assert pane.component_action.isVisible(), 'an FRF is complex'
    finally:
        pane.close()
        pane.deleteLater()
        qt_app.processEvents()


def test_real_data_is_not_offered_a_component_to_pick(qt_app):
    from visualdynamics.core.data import TimeHistory

    history = TimeHistory(abscissa=np.linspace(0.0, 1.0, 32),
                          ordinate=np.zeros((1, 32)), response_dof=['101X+'])
    pane = history.plot(show=False)
    qt_app.processEvents()
    try:
        assert not pane.component_action.isVisible()
        assert not pane.toolbar.isVisible(), 'nothing applies, so no bar'
    finally:
        pane.close()
        pane.deleteLater()
        qt_app.processEvents()


def test_choosing_a_component_redraws_from_the_pane(qt_app):
    """The pane owns the choice, so the redraw reads it back rather than
    being told once at the call."""
    frf = Frf(abscissa=np.linspace(1.0, 100.0, 64),
              ordinate=np.full((1, 64), 1 + 2j),
              response_dof=['101X+'], reference_dof=['101X+'])
    pane = frf.plot(show=False)
    qt_app.processEvents()
    try:
        def drawn_y():
            plot = next(item for item in pane.graphics.ci.items
                        if hasattr(item, 'viewRange'))
            return plot.listDataItems()[0].getData()[1]

        magnitude = drawn_y().copy()
        pane.component_box.setCurrentIndex(2)          # imaginary
        qt_app.processEvents()
        assert not np.allclose(drawn_y(), magnitude), (
            'the curve has to change when the component does')
    finally:
        pane.close()
        pane.deleteLater()
        qt_app.processEvents()


def test_rendering_to_a_file_never_builds_a_pane(tmp_path, qt_app):
    """A headless run must not go near the app's widgets."""
    frf = Frf(abscissa=np.linspace(0.0, 100.0, 32),
              ordinate=np.ones((1, 32), dtype=complex),
              response_dof=['101X+'], reference_dof=['101X+'])
    out = tmp_path / 'frf.png'
    result = frf.plot(path=out)
    assert str(result) == str(out) and out.exists()
