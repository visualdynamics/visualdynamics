"""Averaging frames and shock windows drawn on the waterfall stage.

The marks read the same numbers the flat overlays read —
`Averaging.frame_bounds`, `window_shape`, the shocks' own windows —
so the assertions pin positions against those numbers, not against
the drawing."""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.shocks import Shock
from visualdynamics.viz.waterfall import STAGE

# a nonzero origin on purpose: a record cut to start at half a second
# must have its marks shifted with it, and extents starting at zero
# would let a dropped origin term pass
EXTENTS = (0.5, 8.5, -1.0, 1.0)


def _averaging(frames=5, length=1024, rate=1024.0):
    return Averaging(frame_length=length, frames=frames, overlap=0.5,
                     window='hann'), rate


def test_the_frames_land_where_the_averaging_says():
    import pyvista as pv

    from visualdynamics.viz.marks import add_averaging_marks

    averaging, rate = _averaging()
    plotter = pv.Plotter(off_screen=True)
    counts = add_averaging_marks(plotter, averaging, rate, EXTENTS)
    assert counts == {'frames': 5, 'edges': 2}
    names = list(plotter.actors)
    assert 'marks-averaging-windows' in names
    assert 'marks-averaging-bands' in names
    # the analysis span is one filled slab, the same reading as a
    # shock's window, rimmed at both ends
    span = plotter.actors['marks-averaging-span'].mapper.dataset
    seconds = averaging.stop(rate)
    expected = (seconds - EXTENTS[0]) / (EXTENTS[1] - EXTENTS[0]) \
        * STAGE[0]
    assert np.asarray(span.points)[:, 0].max() == pytest.approx(expected)
    stop = plotter.actors['marks-averaging-edge-stop'].mapper.dataset
    assert np.asarray(stop.points)[:, 0] == pytest.approx(expected)
    # one window glyph polyline per frame
    glyphs = plotter.actors['marks-averaging-windows'].mapper.dataset
    assert glyphs.n_cells == 5
    # glyphs ride above the stage ceiling, on the back wall
    points = np.asarray(glyphs.points)
    assert points[:, 2].min() >= STAGE[2]
    assert points[:, 1].max() > STAGE[1]
    # a tick at each end of every window — the 2-D rail's caps, which
    # are what say where a tapering frame stopped (Brandon, 2026-08-24)
    caps = plotter.actors['marks-averaging-caps'].mapper.dataset
    assert caps.n_cells == 10, 'two ticks per frame'
    tick_x = np.unique(np.asarray(caps.points, dtype=np.float64)[:, 0])
    for low, _high in averaging.frame_bounds(rate):
        expected = (low - EXTENTS[0]) / 8.0 * STAGE[0]
        assert np.isclose(tick_x, expected, atol=1e-5).any(), \
            'a tick under every frame start'
    plotter.close()


def test_the_band_shading_carries_the_windows_own_weight():
    """The 2-D bands fade with the window — the shading is the weight
    each moment carries — and the stage bands say the same thing: the
    band mesh holds the window's value per point, hann's zero at the
    frame edges and its one at the center."""
    import pyvista as pv

    from visualdynamics.viz.marks import add_averaging_marks

    averaging, rate = _averaging()
    plotter = pv.Plotter(off_screen=True)
    add_averaging_marks(plotter, averaging, rate, EXTENTS)
    bands = plotter.actors['marks-averaging-bands'].mapper.dataset
    assert bands.n_points > 0, \
        'the mapper owns the mesh — the transfer-function path lost it'
    weight = np.asarray(bands.point_data['weight'])
    assert weight.min() == pytest.approx(0.0, abs=1e-6), \
        "hann opens and closes at zero, and so must the band's shading"
    assert weight.max() == pytest.approx(1.0, abs=0.01)
    # within one frame the edges are lighter than the middle
    per_frame = weight.reshape(5, -1)
    assert (per_frame[:, 0] < per_frame[:, per_frame.shape[1] // 2]).all()
    # and the shading is translucency: the alpha channel carries the
    # weight, the color stays one color
    fade = np.asarray(bands.point_data['fade'])
    assert (np.unique(fade[:, :3], axis=0).shape[0] == 1), 'one color'
    assert np.corrcoef(fade[:, 3], weight)[0, 1] > 0.999, \
        'the alpha rides the window'
    plotter.close()


def test_a_flattops_negative_shoulders_shade_by_magnitude():
    """A flattop dips below zero at its shoulders. Cast signed, the
    negative alpha wrapped through uint8 and the band went solid
    where the window nearly vanishes; the shading is |w| — a
    negatively weighted moment still carries its (tiny) weight."""
    import pyvista as pv

    from visualdynamics.viz.marks import FADE_OPACITY, add_averaging_marks

    averaging = Averaging(frame_length=1024, frames=3, overlap=0.5,
                          window='flattop')
    plotter = pv.Plotter(off_screen=True)
    add_averaging_marks(plotter, averaging, 1024.0, EXTENTS)
    bands = plotter.actors['marks-averaging-bands'].mapper.dataset
    weight = np.asarray(bands.point_data['weight'])
    assert weight.min() < 0, 'the fixture really has negative shoulders'
    fade = np.asarray(bands.point_data['fade'])
    dipped = fade[weight < 0, 3].astype(float)
    expected = np.round(np.abs(weight[weight < 0])
                        * FADE_OPACITY * 255)
    assert np.array_equal(dipped, expected), \
        'the shoulders shade at the few counts their magnitude earns'
    # flattop's deepest dip is ~6.9% of its peak — alpha 8 of 255
    assert dipped.max() <= 10, 'faint, never a wraparound to solid'
    plotter.close()


def test_each_shock_window_is_a_numbered_slab():
    import pyvista as pv

    from visualdynamics.viz.marks import add_shock_marks

    shocks = [Shock(start=1.0, duration=0.5),
              Shock(start=4.0, duration=0.5)]
    plotter = pv.Plotter(off_screen=True)
    counts = add_shock_marks(plotter, shocks, EXTENTS)
    assert counts == {'windows': 2}
    slab = plotter.actors['marks-shock-1'].mapper.dataset
    x = np.asarray(slab.points)[:, 0]
    assert x.min() == pytest.approx((4.0 - 0.5) / 8.0 * STAGE[0])
    assert x.max() == pytest.approx((4.5 - 0.5) / 8.0 * STAGE[0])
    assert 'marks-shock-numbers' in ' '.join(plotter.actors)
    plotter.close()


def test_no_shocks_marks_nothing():
    import pyvista as pv

    from visualdynamics.viz.marks import add_shock_marks

    plotter = pv.Plotter(off_screen=True)
    assert add_shock_marks(plotter, (), EXTENTS) == {'windows': 0}
    assert not [n for n in plotter.actors if 'marks-shock' in n]
    plotter.close()


def _drawn_x_labels(pane):
    """The x labels as the screen draws them — GetXAxisRange can read
    the truth while the rendered strings are stale, which is exactly
    the failure this measures."""
    axes = pane.waterfall_plotter.renderer.cube_axes_actor
    labels = axes.GetAxisLabels(0)
    out = []
    for k in range(labels.GetNumberOfValues()):
        try:
            out.append(float(labels.GetValue(k)))
        except ValueError:
            pass
    return out


def test_the_marks_do_not_stretch_the_axes(window, pump):
    """The rail rides above the ceiling and the wall stands behind the
    stations; counted in the scene's bounds they stretched the cube
    axes' box while its labeled ranges stayed put — the time axis
    read wrong values the moment the averaging view opened (Brandon,
    2026-08-24)."""
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(3)
    fs = 1024.0
    t = np.arange(int(fs * 8)) / fs
    window.add_object('Long', TimeHistory(
        t, rng.standard_normal((3, len(t))),
        response_dof=['101Z+', '104Z+', '110Z+']))
    item = window._item_for_object('Long')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    pane = window.data_pane
    axes = pane.waterfall_plotter.renderer.cube_axes_actor
    before = tuple(axes.GetBounds())
    pane.averaging_action.trigger()
    pump()
    assert pane._waterfall_page.isVisible()
    assert any('marks-averaging' in n
               for n in pane.waterfall_plotter.actors)
    after = tuple(pane.waterfall_plotter.renderer
                  .cube_axes_actor.GetBounds())
    assert after == pytest.approx(before, abs=1e-6), \
        'the marks joined the stage without resizing its axes'
    # and the labels still read seconds, not stage units: pyvista's
    # SetBounds rewrites the labeled ranges, which the first fix
    # tripped over — the bounds held while the time axis went 0..1.6
    x_range = tuple(pane.waterfall_plotter.renderer
                    .cube_axes_actor.GetXAxisRange())
    assert x_range == pytest.approx((0.0, 8.0), abs=1e-3), \
        'the time axis reads the record, eight seconds of it'
    assert _drawn_x_labels(pane) and max(_drawn_x_labels(pane)) > 4.0, \
        'and the drawn strings say so too — the range read true while ' \
        'the screen showed stage units'


def test_the_shock_projects_time_axis_reads_seconds(window, pump):
    """The real file Brandon caught it on: open the shock project,
    shocks view, 3-D reading — the time axis must span the record's
    own seconds, not the stage's unit box."""
    import os

    path = os.path.join(os.path.dirname(__file__), '..', 'stressdata',
                        'plate_projects', 'shock.vdyn')
    if not os.path.exists(path):
        pytest.skip('plate stressdata not on this machine')
    from visualdynamics.core.data import TimeHistory

    window.import_paths([path])
    pump()
    name = next(n for n, o in window.objects.items()
                if isinstance(o, TimeHistory))
    history = window.objects[name]
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    pane = window.data_pane
    pane.shocks_action.setChecked(True)
    pane.shocks_wanted = True
    window.render_current()
    pump()
    assert any(n.startswith('marks-shock-')
               for n in pane.waterfall_plotter.actors)
    x_range = tuple(pane.waterfall_plotter.renderer
                    .cube_axes_actor.GetXAxisRange())
    assert x_range == pytest.approx(
        (float(history.abscissa[0]), float(history.abscissa[-1])),
        abs=1e-3), 'the axis spans the record, in its own seconds'
    drawn = _drawn_x_labels(pane)
    assert drawn and max(drawn) == pytest.approx(
        float(history.abscissa[-1]), rel=0.2), \
        'the drawn strings read seconds, not stage units'


def test_a_panel_edit_reshapes_the_stage_marks(window, pump):
    """Change the window type while the 3-D reading is up and the
    stage's glyphs follow — the flat overlays restate themselves, the
    stage rebuilds with the scene (Brandon, 2026-08-24: the type
    change did nothing in 3-D)."""
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(9)
    fs = 1024.0
    t = np.arange(int(fs * 8)) / fs
    history = TimeHistory(t, rng.standard_normal((3, len(t))),
                          response_dof=['101Z+', '104Z+', '110Z+'])
    window.add_object('Long', history)
    item = window._item_for_object('Long')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    pane = window.data_pane
    pane.averaging_action.trigger()
    pump()
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the marks ride the stage'
    panel = pane.averaging_panel
    panel.window_box.setCurrentIndex(panel.window_box.findData('hann'))
    pump()
    bands = pane.waterfall_plotter.actors[
        'marks-averaging-bands'].mapper.dataset
    assert np.asarray(bands.point_data['weight']).min() == \
        pytest.approx(0.0, abs=1e-6), 'hann tapers to zero'
    panel.window_box.setCurrentIndex(
        panel.window_box.findData('rectangle'))
    pump()
    bands = pane.waterfall_plotter.actors[
        'marks-averaging-bands'].mapper.dataset
    assert np.asarray(bands.point_data['weight']).min() == \
        pytest.approx(1.0), 'boxcar: the glyphs reshaped in place'
    assert history.averaging.window == 'boxcar', 'and the object heard'


def test_the_shock_view_marks_the_stage(window, pump):
    """The GUI path: a history with detected shocks, shocks view up,
    3-D reading — the slabs draw and the panel edits."""
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(5)
    fs, seconds = 2048.0, 8.0
    t = np.arange(int(fs * seconds)) / fs
    quiet = rng.standard_normal((3, len(t))) * 1e-4
    for start in (1.0, 4.0):
        lump = slice(int(start * fs), int((start + 0.1) * fs))
        quiet[:, lump] += np.sin(
            2 * np.pi * 400.0 * t[lump]) * 50.0
    history = TimeHistory(t, quiet,
                          response_dof=['101Z+', '104Z+', '110Z+'])
    window.add_object('Bang', history)
    item = window._item_for_object('Bang')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    pane = window.data_pane
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the 3-D reading is the default'
    window._detect_shocks()
    pane.shocks_action.setChecked(True)
    pane.shocks_wanted = True
    window.render_current()
    pump()
    assert pane._waterfall_page.isVisible(), 'the stage holds'
    assert pane.shock_panel.isVisible()
    names = [n for n in pane.waterfall_plotter.actors
             if n.startswith('marks-shock-')]
    assert names, 'the windows drew as slabs'


# ---- the handles, and dragging them --------------------------------------

def test_every_slab_wears_three_handles():
    import pyvista as pv

    from visualdynamics.viz.marks import add_averaging_marks, add_shock_marks

    averaging, rate = _averaging()
    plotter = pv.Plotter(off_screen=True)
    add_averaging_marks(plotter, averaging, rate, EXTENTS)
    names = set(plotter.actors)
    assert {'marks-averaging-handle-start', 'marks-averaging-handle-move',
            'marks-averaging-handle-stop'} <= names
    plotter.close()

    plotter = pv.Plotter(off_screen=True)
    add_shock_marks(plotter, [Shock(1.0, 0.5), Shock(4.0, 0.5)], EXTENTS)
    names = set(plotter.actors)
    assert {'marks-shock-0-handle-open', 'marks-shock-1-handle-move',
            'marks-shock-1-handle-close'} <= names
    plotter.close()


def test_a_locked_record_offers_no_shock_handles():
    """A record the controller already cut into frames is not the
    user's to re-window — the 2-D regions go immovable, the stage
    simply offers nothing to grab."""
    import pyvista as pv

    from visualdynamics.viz.marks import add_shock_marks

    plotter = pv.Plotter(off_screen=True)
    add_shock_marks(plotter, [Shock(1.0, 0.5)], EXTENTS, locked=True)
    assert not [n for n in plotter.actors if 'handle' in n]
    assert 'marks-shock-0' in plotter.actors, 'the slab still shows'
    plotter.close()


def _stage_with_averaging(window, pump):
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(9)
    fs = 1024.0
    t = np.arange(int(fs * 8)) / fs
    history = TimeHistory(t, rng.standard_normal((3, len(t))),
                          response_dof=['101Z+', '104Z+', '110Z+'])
    # a real multi-frame analysis, not the whole-record default —
    # a drag needs frames to take away and room to slide in
    history.averaging = Averaging(frame_length=1024, frames=5,
                                  overlap=0.0, window='hann', start=1.0)
    window.add_object('Long', history)
    item = window._item_for_object('Long')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.data_pane.averaging_action.trigger()
    pump()
    assert window.data_pane._waterfall_page.isVisible()
    return history


def test_dragging_the_stop_handle_takes_frames_away(window, pump):
    """The same reading as pulling the 2-D region's right edge in:
    the span ends sooner, so it holds fewer whole frames."""
    history = _stage_with_averaging(window, pump)
    before = history.averaging
    dragger = window._stage_dragger
    assert dragger is not None and dragger._context is not None
    assert dragger.begin('marks-averaging-handle-stop')
    dragger.drag_to(4.0)
    dragger.finish(4.0)
    pump()
    after = history.averaging
    assert before.frames == 5 and after.frames == 3, \
        'the span to 4 s holds three whole frames, not five'
    assert after.start == before.start, 'the start edge never moved'
    assert after.stop(1024.0) == pytest.approx(4.0), \
        'the analysis ends where the handle let go'
    # and the panel followed — one control shown twice
    assert window.data_pane.averaging_panel.frames_box.value() == 3


def test_dragging_the_move_handle_slides_the_span(window, pump):
    history = _stage_with_averaging(window, pump)
    before = history.averaging
    dragger = window._stage_dragger
    assert dragger.begin('marks-averaging-handle-move')
    middle = (before.start + before.stop(1024.0)) / 2.0
    dragger.finish(middle + 1.0)
    pump()
    after = history.averaging
    assert after.frames == before.frames, 'a move never resizes'
    assert after.start == pytest.approx(before.start + 1.0)


def test_dragging_a_shock_edge_commits_through_the_shared_rule(window,
                                                               pump):
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.shocks import Shock as CoreShock

    rng = np.random.default_rng(5)
    fs = 2048.0
    t = np.arange(int(fs * 8)) / fs
    history = TimeHistory(t, rng.standard_normal((2, len(t))) * 1e-3,
                          response_dof=['101Z+', '104Z+'])
    history.shocks = (CoreShock(1.0, 0.5), CoreShock(4.0, 0.5))
    window.add_object('Bang', history)
    item = window._item_for_object('Bang')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    pane = window.data_pane
    pane.shocks_action.setChecked(True)
    pane.shocks_wanted = True
    window.render_current()
    pump()
    dragger = window._stage_dragger
    assert dragger is not None and dragger._context['kind'] == 'shocks'
    # the windows share a length, so resizing the first regrows both
    assert dragger.begin('marks-shock-0-handle-close')
    dragger.finish(1.8)
    pump()
    first, second = history.shocks
    assert first.duration == pytest.approx(0.8, abs=1e-6)
    assert second.duration == pytest.approx(0.8, abs=1e-6), \
        'the shared length reached the window nobody dragged'
    assert second.start == pytest.approx(4.0), \
        'a resize moves no starts'
    # and a move drags one window only
    assert dragger.begin('marks-shock-0-handle-move')
    anchor = (history.shocks[0].start + history.shocks[0].stop) / 2.0
    dragger.finish(anchor + 0.5)
    pump()
    first, second = history.shocks
    assert first.start == pytest.approx(1.5, abs=1e-6)
    assert second.start == pytest.approx(4.0)


def test_a_second_drag_keeps_the_firsts_edit(window, pump):
    """Drag the stop in, then drag the start: the stop must stay where
    the first drag put it. The commit used to skip the stage
    re-render, so the dragger stayed armed with the old span and the
    second drag quietly committed the original stop back (Brandon,
    2026-08-24)."""
    history = _stage_with_averaging(window, pump)
    dragger = window._stage_dragger
    assert dragger.begin('marks-averaging-handle-stop')
    dragger.finish(4.0)
    pump()
    moved = history.averaging
    assert moved.stop(1024.0) == pytest.approx(4.0)
    # now the other edge — the dragger must anchor on the new span
    assert dragger.begin('marks-averaging-handle-start')
    dragger.finish(2.0)
    pump()
    after = history.averaging
    assert after.start == pytest.approx(2.0)
    assert after.stop(1024.0) == pytest.approx(4.0), \
        "the stop stays where the first drag put it"


def test_a_drag_preview_renders_one_frame(window, pump, monkeypatch):
    """Every actor the preview adds used to render its own frame —
    several of them with the axes momentarily inflated, so the drag
    visibly stuttered before each corrected frame (Brandon,
    2026-08-24). One preview, one render."""
    _stage_with_averaging(window, pump)
    plotter = window.data_pane.waterfall_plotter
    frames = []
    real = plotter.render
    monkeypatch.setattr(plotter, 'render',
                        lambda *a, **k: (frames.append(1),
                                         real(*a, **k))[1])
    dragger = window._stage_dragger
    assert dragger.begin('marks-averaging-handle-stop')
    dragger.drag_to(4.0)
    assert len(frames) == 1, \
        f'{len(frames)} frames for one preview; the stutter is back'


def test_the_handles_are_engraved_disks():
    """Gray disks, not spheres: an arrow on each edge handle pointing
    the way that edge extends, a three-line grip on the center — and
    the engravings are not pickable, so a click on one grabs the
    disk under it (Brandon, 2026-08-24)."""
    import pyvista as pv

    from visualdynamics.viz.marks import add_averaging_marks

    averaging, rate = _averaging()
    plotter = pv.Plotter(off_screen=True)
    add_averaging_marks(plotter, averaging, rate, EXTENTS)
    start = plotter.actors['marks-averaging-handle-start']
    disk = start.mapper.dataset
    points = np.asarray(disk.points, dtype=np.float64)
    spread = points - points.mean(axis=0)
    thinnest = np.linalg.svd(spread, compute_uv=False)[-1]
    assert thinnest < 1e-6, 'a flat disk, not a sphere'
    assert np.ptp(points[:, 2]) > 0.01, \
        'and tilted toward the viewer, not lying in the floor'
    assert points[:, 1].max() <= 1e-6, \
        'wholly in front of the stage — half a disk buried in the ' \
        'slab was half a disk that could not be clicked'
    assert start.GetPickable(), 'the disk is the drag target'
    slab = plotter.actors['marks-averaging-span']
    assert not slab.GetPickable(), \
        'the slab yields every click to the handles'
    def points_toward(icon_actor):
        """The arrow's direction: its apex is the lone vertex, on the
        other side of the centroid from the two base corners —
        comparing extremes against the center let a swapped arrow
        pass on its base corners, which is exactly what happened."""
        triangle = np.asarray(icon_actor.mapper.dataset.points)[:, 0]
        centroid = triangle.mean()
        apex = triangle[np.argmax(np.abs(triangle - centroid))]
        return 'left' if apex < centroid else 'right'

    left = plotter.actors['marks-averaging-handle-start-icon']
    assert not left.GetPickable(), 'the engraving yields to its disk'
    assert points_toward(left) == 'left', 'the start arrow points left'
    right = plotter.actors['marks-averaging-handle-stop-icon']
    assert points_toward(right) == 'right', 'the stop arrow points right'
    grip = plotter.actors['marks-averaging-handle-move-icon']
    assert grip.mapper.dataset.n_cells == 3, 'three grip lines'
    plotter.close()


# ---- the pointer's own arithmetic ---------------------------------------
#
# The VTK plumbing is split from the semantics because a synthetic mouse
# cannot reach an offscreen interactor — and half of that is literal:
# vtkPropPicker.Pick segfaults against the offscreen render window, so
# _handle_under and _pressed stay untested here and are exercised only
# by a person's click. What *can* be pinned headless is pinned:
# the display-ray projection onto the time axis, and the release path's
# duty to hand the camera style back.


def test_the_pointer_ray_reads_the_time_axis(window, pump):
    """A display point projected onto the stage's front edge must read
    back as the seconds that stage x means — the whole drag rests on
    this arithmetic, and nothing else checks it."""
    from visualdynamics.viz.waterfall import STAGE

    _stage_with_averaging(window, pump)
    dragger = window._stage_dragger
    plotter = window.data_pane.waterfall_plotter
    plotter.render()
    renderer = plotter.renderer
    x0, x1, *_rest = dragger._context['extents']
    for s in (0.0, STAGE[0] * 0.35, STAGE[0]):
        renderer.SetWorldPoint(s, 0.0, 0.0, 1.0)
        renderer.WorldToDisplay()
        dx, dy, _depth = renderer.GetDisplayPoint()
        dragger._iren.SetEventPosition(round(dx), round(dy))
        expected = x0 + s / STAGE[0] * (x1 - x0)
        assert dragger._pointer_seconds() == pytest.approx(
            expected, abs=0.02), \
            'the ray lands within a display pixel of the stage x'


def test_a_release_hands_the_camera_style_back(window, pump):
    """The press stands the interactor style down so the camera cannot
    orbit under the handle; the release must put it back, or every
    later orbit gesture is dead — driven through the event handlers
    the interactor itself would call."""
    from visualdynamics.viz.waterfall import STAGE

    history = _stage_with_averaging(window, pump)
    before = history.averaging
    dragger = window._stage_dragger
    iren = dragger._iren
    plotter = window.data_pane.waterfall_plotter
    plotter.render()
    x0, x1, *_rest = dragger._context['extents']

    def position_at(seconds):
        s = (seconds - x0) / (x1 - x0) * STAGE[0]
        renderer = plotter.renderer
        renderer.SetWorldPoint(s, 0.0, 0.0, 1.0)
        renderer.WorldToDisplay()
        dx, dy, _depth = renderer.GetDisplayPoint()
        iren.SetEventPosition(round(dx), round(dy))

    anchor = (before.start + before.stop(1024.0)) / 2.0
    assert dragger.begin('marks-averaging-handle-move', anchor)
    # what _pressed does after the pick this test cannot make
    style = iren.GetInteractorStyle()
    dragger._style = style
    iren.SetInteractorStyle(None)

    position_at(anchor + 1.0)
    dragger._moved(None, None)
    dragger._released(None, None)
    pump()
    after = history.averaging
    assert after.start == pytest.approx(before.start + 1.0, abs=0.02), \
        'the release committed the slide the pointer asked for'
    assert iren.GetInteractorStyle() is style, \
        'and the camera style came back'
    assert dragger._style is None
