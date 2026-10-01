"""The view belongs to the user.

`reset_camera()` runs when the *displayed object* changes and at no other
time. Everything else — a redraw, an edit, adding an element, leaving
edit mode, switching display units — has to leave the camera exactly
where it was put, because a view someone set up is work and throwing it
away mid-edit is the most irritating thing a 3D application can do.

It reads as one rule and is enforced in a dozen places (`_framed`,
`camera_position` saved and restored around every `plotter.clear()`), so
it is the kind of rule that comes apart one caller at a time. It was
covered only inside the legacy end-to-end script.
"""

from __future__ import annotations

import pytest
from conftest import edit_category as edit
from conftest import fixture_path


def camera(window):
    return tuple(window.scene.plotter.camera.position)


@pytest.fixture
def zoomed(window, pump):
    """A plate on screen with the camera moved off its framing."""
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    framed = camera(window)
    window.scene.plotter.camera.Dolly(2.0)          # the user zooms in
    pump()
    assert camera(window) != framed, 'the fixture did not move the camera'
    return camera(window)


def test_redrawing_what_is_already_there_keeps_the_view(zoomed, window, pump):
    window.render_current()
    pump()
    assert camera(window) == zoomed


def test_an_edit_redraws_the_scene_without_reframing_it(zoomed, window, pump):
    edit(window, pump, 'Nodes')
    assert camera(window) == zoomed
    window.table.selectRow(3)
    pump()
    window._draw_edit_selection()
    assert camera(window) == zoomed, 'selecting a row is not new content'


def test_creating_an_element_does_not_zoom_back_out(zoomed, window, pump):
    edit(window, pump, 'Triangles')
    window.set_add_mode(True)
    screen = window._projector.screen()[0]
    for n, row in enumerate((0, 1, 6)):
        window.hover_at(*screen[row])
        window._add_at(*screen[row], extend=n > 0)
    assert camera(window) == zoomed, 'the scene is rebuilt; the view is not'
    window.set_add_mode(False)
    window.stop_editing()
    pump()
    assert camera(window) == zoomed, 'nor does leaving edit mode'


def test_switching_display_units_keeps_the_view(zoomed, window, pump):
    """The model is redrawn in different numbers, but it is the same
    model — re-framing here would move the view on a units change."""
    window.unit_combo.setCurrentText('m-kg-N-s')
    pump()
    assert camera(window) == zoomed


def test_a_geometry_shown_for_the_first_time_is_framed(zoomed, window, pump):
    """The one time it must: new content nobody has looked at."""
    window.import_paths([fixture_path('plate', 'geometry.npz')])
    pump()
    item = window._item_for_object('Geometry (2)')
    window.tree.clearSelection()
    window.tree.setCurrentItem(item)
    item.setSelected(True)
    pump()
    assert camera(window) != zoomed, 'a geometry never seen is framed'
