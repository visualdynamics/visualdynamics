"""A geometry's default view (Brandon, 2026-09-27).

Every 3-D view of a geometry opened from +X+Y+Z with Z up — the app by
PyVista's default camera, the report by an isometric of its own — and
nothing let a geometry say otherwise, so a model built Y-up drew lying
on its side and its pictures turned its nodes to stand it up. A geometry
carries a `View` now: the app opens it and everything drawn on it on
that view, Reset View returns to it, and the report's scenes and the
exported figures are drawn from it.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics import View


def _camera_basis(plotter):
    """(right, up) on screen, from a PyVista camera."""
    position, focus, up = (np.asarray(v, dtype=float)
                           for v in plotter.camera_position)
    sight = focus - position
    sight /= np.linalg.norm(sight)
    right = np.cross(sight, up)
    right /= np.linalg.norm(right)
    return right, np.cross(right, sight)


def _plate():
    from visualdynamics import mesh

    return mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.25)


# ---- the view itself ------------------------------------------------------


def test_a_view_is_a_direction_and_an_up():
    view = View(eye=(2, 2, -2), up=(0, 3, 0))
    assert view == View(eye=(1, 1, -1), up=(0, 1, 0)), 'lengths do not count'
    assert view.eye == (2.0, 2.0, -2.0), 'kept as written'
    right, up = view.basis()
    # a camera's own basis: right-handed, the eye on the near side
    assert np.allclose(np.cross(right, up), np.array([1, 1, -1]) / np.sqrt(3))
    assert abs(np.dot(up, [1, 1, -1])) < 1e-12
    assert eval(repr(view), {'View': View}) == view
    with pytest.raises(ValueError, match='line of sight'):
        View(eye=(0, 0, 1), up=(0, 0, 2))
    with pytest.raises(ValueError, match='direction'):
        View(eye=(0, 0, 0))


def test_a_geometry_without_one_opens_on_the_old_isometric():
    geometry = _plate()
    assert geometry.view is None
    assert geometry.opening_view == View(eye=(1, 1, 1), up=(0, 0, 1))


def test_the_report_and_the_app_see_the_same_view():
    """The page draws a scene from `View.basis`, the app from a PyVista
    camera `place_view` turns; for the default and a Y-up view alike
    the two agree, right and up both. The page's own isometric used to
    have right pointing the other way — the model seen from beneath."""
    import pyvista as pv

    from visualdynamics.viz.geometry import add_geometry, place_view

    geometry = _plate()
    for view in (View(), View(eye=(1, 1, -1), up=(0, 1, 0)),
                 View(eye=(0, -1, 0.2), up=(0, 0, 1))):
        plotter = pv.Plotter(off_screen=True)
        try:
            add_geometry(plotter, geometry)
            place_view(plotter, view, render=False)
            right, up = _camera_basis(plotter)
        finally:
            plotter.close()
        page_right, page_up = view.basis()
        assert np.allclose(right, page_right, atol=1e-9), view
        assert np.allclose(up, page_up, atol=1e-9), view


# ---- kept, and carried ----------------------------------------------------


def test_a_view_survives_the_project_file(tmp_path):
    geometry = _plate()
    geometry.view = View(eye=(0, -1, 0.5), up=(0, 0, 1))
    bare = _plate()
    project = visualdynamics.Project('views')
    project.add('Turned', geometry)
    project.add('Bare', bare)
    project.save(tmp_path / 'views.vdyn')
    back = visualdynamics.load(tmp_path / 'views.vdyn')
    assert back['Turned'].view == geometry.view
    assert back['Bare'].view is None, 'absent stays absent'


def test_a_merged_geometry_keeps_the_view():
    from visualdynamics import mesh

    first = _plate()
    second = mesh.plane((0, 0, 1), (1, 0, 0), (0, 1, 0), 0.25)
    second.node_id = second.node_id + 1000
    second.elem_conn = [conn + 1000 for conn in second.elem_conn]
    second.view = View(eye=(1, 0, 0), up=(0, 0, 1))
    project = visualdynamics.Project('merge')
    project.add('A', first)
    project.add('B', second)
    merged = project[project.merge('A', 'B')]
    assert merged.view == second.view


def test_set_view_is_a_journaled_verb():
    project = visualdynamics.Project('journal')
    project.add('Plate', _plate())
    view = View(eye=(1, 0, 1), up=(0, 0, 1))
    project.set_view('Plate', view)
    assert project['Plate'].view == view
    script = project.session_script()
    assert 'from visualdynamics import View' in script
    assert ("project.set_view('Plate', View(eye=(1.0, 0.0, 1.0), "
            "up=(0.0, 0.0, 1.0)))") in script
    project.set_view('Plate', None)
    assert project['Plate'].view is None
    assert 'set_view' in dict(project.selection_verbs('Plate'))
    with pytest.raises(TypeError):
        project.set_view('Plate', ((1, 0, 0), (0, 0, 1)))


# ---- drawn from it --------------------------------------------------------


def test_a_report_scene_opens_on_the_geometry_view():
    from visualdynamics.report import _scene_block

    geometry = _plate()
    geometry.view = View(eye=(1, 1, -1), up=(0, 1, 0))
    built = _scene_block({'kind': 'scene'}, geometry, None,
                         visualdynamics.SI, {})
    assert np.allclose(built['home'], geometry.view.basis(), atol=1e-6)
    geometry.view = None
    built = _scene_block({'kind': 'scene'}, geometry, None,
                         visualdynamics.SI, {})
    assert np.allclose(built['home'], View().basis(), atol=1e-6)


def test_an_exported_figure_is_drawn_from_the_view(tmp_path, monkeypatch):
    from visualdynamics.viz import geometry as scene

    geometry = _plate()
    geometry.view = View(eye=(0, -1, 0), up=(0, 0, 1))
    seen = []
    real = scene.place_view

    def spy(plotter, view=None, render=True):
        seen.append(view)
        return real(plotter, view, render=render)

    monkeypatch.setattr(scene, 'place_view', spy)
    scene.plot_geometry(geometry, screenshot=str(tmp_path / 'g.png'))
    scene.plot_geometry(geometry, screenshot=str(tmp_path / 'p.png'),
                        size_in=(2.0, 1.5), dpi=150)
    assert seen == [geometry.view, geometry.view]


# ---- the app --------------------------------------------------------------


def _show(window, pump, name):
    item = window._item_for_object(name)
    window.tree.clearSelection()
    window.tree.setCurrentItem(item)
    item.setSelected(True)
    pump()


@pytest.fixture
def plate(window, pump):
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    name = next(n for n, o in window.objects.items()
                if isinstance(o, visualdynamics.Geometry))
    return name


def test_set_default_view_keeps_the_view_on_screen(plate, window, pump):
    """Turn the model, Set Default View: the geometry opens that way from
    then on, and Reset View comes back to it after another turn."""
    plotter = window.scene.plotter
    assert window.scene.reset_view_action.isVisible()
    plotter.camera.Azimuth(70)
    plotter.camera.Elevation(-30)
    turned = _camera_basis(plotter)
    labels = [label for _verb, label, *_rest in window.acts_for([plate])]
    assert 'Set Default View' in labels
    window.set_view_act()
    view = window.objects[plate].view
    assert view is not None
    assert np.allclose(view.basis(), turned, atol=2e-3)
    plotter.camera.Azimuth(-120)
    assert not np.allclose(_camera_basis(plotter), turned, atol=2e-3)
    window.scene.reset_view_action.trigger()
    assert np.allclose(_camera_basis(plotter), turned, atol=2e-3)
    assert "set_view(" in window.project.session_script()


def test_a_geometry_opens_on_its_view(window, pump):
    """Shown for the first time, a geometry is turned to its own view."""
    view = View(eye=(0, -1, 0.3), up=(0, 0, 1))
    geometry = _plate()
    geometry.view = view
    window.add_object('Turned', geometry)
    _show(window, pump, 'Turned')
    assert np.allclose(_camera_basis(window.scene.plotter), view.basis(),
                       atol=1e-6)
    assert window.scene.home_view == view


def test_a_turn_survives_stepping_to_the_same_geometry_again(window, pump):
    """Framing is for new content: a second showing of the same geometry
    — its shapes, say — is refitted, not turned back, so the user's turn
    stands; another geometry opens on its own view."""
    first, second = _plate(), _plate()
    second.view = View(eye=(1, 0, 0), up=(0, 0, 1))
    window.add_object('First', first)
    window.add_object('Second', second)
    _show(window, pump, 'First')
    plotter = window.scene.plotter
    plotter.camera.Azimuth(50)
    turned = _camera_basis(plotter)
    window.scene.set_home_view(first.opening_view, first)
    window.scene.frame()
    assert np.allclose(_camera_basis(plotter), turned, atol=1e-6)
    _show(window, pump, 'Second')
    assert np.allclose(_camera_basis(plotter), second.view.basis(),
                       atol=1e-6)


def test_the_page_opens_a_scene_on_the_geometry_view(qt_app, tmp_path):
    """The page draws from the payload's home, not an isometric of its
    own: opened, a scene's published view is the geometry's basis."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    import json

    from conftest import web_close, web_read
    from PySide6.QtCore import QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView

    geometry = _plate()
    geometry.view = View(eye=(1, 1, -1), up=(0, 1, 0))
    path = visualdynamics.export_html(tmp_path / 'g.html',
                                      geometry=geometry, theme='light')
    view = QWebEngineView()
    view.resize(600, 400)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    probe = ("(() => { const c = document.querySelector('canvas');"
             "return c && c.dataset.view ? c.dataset.view : null; })()")
    try:
        opened = json.loads(web_read(view, probe))
    finally:
        web_close(view)
    assert np.allclose(opened['basis'], geometry.view.basis(), atol=1e-4)
