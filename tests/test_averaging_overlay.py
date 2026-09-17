"""The averaging drawn on the record it will be cut from.

Five numbers in a table do not say whether the analysis sits where you
meant it to. The marks do, so what these hold is that they are where the
frames are, that a drag lands on whole frames, and that the overlay
takes every one of its items away again.

The rail above the trace carries one window per frame; the band under
each falls to the foot of the plot from that frame's own slot, so the
height says which level a frame is on and an overlap has a shape rather
than only a shade.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.theme import theme as resolve_theme

RATE = 1024.0
SAMPLES = 8192

# a GraphicsLayoutWidget that goes out of scope takes its items' C++
# objects with it, and touching one afterwards raises "already deleted"
_ALIVE = []


def plot_of(samples=SAMPLES, offset=0.0, theme='light'):
    """A plot of a plain record, and its view fixed so the marks — which
    are sized from the visible height — have something settled to sit in.
    """
    import pyqtgraph as pg

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    plot = layout.addPlot()
    t = np.arange(samples) / RATE
    plot.plot(t, offset + np.sin(2 * np.pi * 20 * t))
    plot.getViewBox().setYRange(offset - 1.0, offset + 1.0, padding=0)
    return plot


def overlay(qt_app, averaging, samples=SAMPLES, locked=False, theme='light'):
    from visualdynamics.plot.averaging import AveragingOverlay

    plot = plot_of(samples)
    return AveragingOverlay(plot, averaging, RATE, samples,
                            resolve_theme(theme), locked=locked), plot


def fill_color(item):
    return item.opts['fillBrush'].color()


def band_gradient(band):
    """([0..1 across the frame], alpha at each) for a band's shading."""
    stops = band.opts['fillBrush'].gradient().stops()
    return [at for at, _color in stops], [c.alpha() for _at, c in stops]


def bands_of(overlaid):
    """[(low, high)] each band covers."""
    return [(item.getData()[0][0], item.getData()[0][-1])
            for item in overlaid.bands]


def band_tops(overlaid):
    """The height each band rises to — its frame's slot on the rail."""
    return [float(item.getData()[1][0]) for item in overlaid.bands]


def glyphs_of(overlaid):
    """[(low, high, baseline, peak)] for every window on the rail."""
    out = []
    for item in overlaid.glyphs:
        x, y = item.getData()
        out.append((float(x.min()), float(x.max()),
                    float(item.opts['fillLevel']), float(y.max())))
    return out


def test_a_band_sits_on_every_frame(qt_app):
    """The whole point: what is shaded is what will be averaged."""
    averaging = Averaging(frame_length=1024, overlap=0.5, frames=4)
    overlaid, _plot = overlay(qt_app, averaging)
    assert len(overlaid.bands) == 4
    assert bands_of(overlaid) == pytest.approx(averaging.frame_bounds(RATE))


def test_overlapping_frames_overlap_on_the_plot(qt_app):
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.5, frames=3))
    bands = bands_of(overlaid)
    assert bands[1][0] < bands[0][1], 'the second starts inside the first'
    assert bands[0][1] - bands[1][0] == pytest.approx(0.5, abs=1e-9)


# ---- the height is what says which frame is which -----------------------


def test_a_band_rises_to_its_own_slot(qt_app):
    """Frames on different levels reach different heights, so where two
    share record you see two heights and the step between them. Shade
    alone said almost nothing at half overlap: the whole span came out
    one wash of blue but for the two ends."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.5, frames=4))
    tops = band_tops(overlaid)
    levels, _count = overlaid.averaging.levels(overlaid.sample_rate)
    assert levels == [0, 1, 0, 1], 'half overlap alternates levels'
    assert tops[0] == pytest.approx(tops[2]), 'one level, one height'
    assert tops[1] == pytest.approx(tops[3])
    assert tops[0] > tops[1], 'and the deeper level is the shorter'


def test_bands_on_one_level_are_all_the_same_height(qt_app):
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=5))
    assert len({round(top, 9) for top in band_tops(overlaid)}) == 1


def test_a_band_falls_to_the_foot_of_the_plot(qt_app):
    """It is a column under its window, not a floating bar."""
    overlaid, plot = overlay(
        qt_app, Averaging(frame_length=1024, frames=2))
    foot = plot.getViewBox().viewRange()[1][0]
    assert all(band.opts['fillLevel'] == pytest.approx(foot)
               for band in overlaid.bands)


def test_a_band_fades_across_with_its_window(qt_app):
    """The shading is the weight each moment of the record carries. A
    Hann frame counts for almost nothing at its ends and for everything
    in the middle, and drawn flat the band said the opposite."""
    from visualdynamics.core.averaging import window_shape

    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, window='hann', frames=1))
    across, alphas = band_gradient(overlaid.bands[0])
    wanted = window_shape('hann', 1024)
    at = (np.asarray(across) * (len(wanted) - 1)).round().astype(int)
    peak = max(alphas)
    assert peak > 0
    assert np.allclose(np.asarray(alphas) / peak, wanted[at], atol=0.02)


def test_a_rectangle_window_shades_flat(qt_app):
    """It weights every sample the same, so anything but a flat band
    would be a picture of an analysis nobody asked for."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, window='rectangle', frames=1))
    _across, alphas = band_gradient(overlaid.bands[0])
    assert len(set(alphas)) == 1


def test_the_gradient_runs_from_one_end_of_the_frame_to_the_other(qt_app):
    """The stops are fractions along it, so a gradient anchored anywhere
    else paints the same numbers over the wrong stretch of record."""
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=3, start=1.0)
    overlaid, _plot = overlay(qt_app, averaging)
    for band, (low, high) in zip(overlaid.bands,
                                 averaging.frame_bounds(RATE)):
        gradient = band.opts['fillBrush'].gradient()
        assert gradient.start().x() == pytest.approx(low)
        assert gradient.finalStop().x() == pytest.approx(high)


def test_a_flat_tops_shoulders_come_out_clear(qt_app):
    """It weights them slightly negatively, which is a fact about the
    window rather than a drawable alpha. Too little to round to one, so
    they shade as nothing — and the peak is what the gradient is
    normalized by, a flat top not reaching one."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, window='flattop', frames=1))
    _across, alphas = band_gradient(overlaid.bands[0])
    assert min(alphas) == 0
    assert max(alphas) / 255 == pytest.approx(0.25, abs=0.01), (
        'the peak still reaches the full shade')


def test_every_frame_is_shaded_the_same_color(qt_app):
    """The shape carries the frame now; a second color would be one
    encoding too many."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=256, overlap=0.0, frames=20))
    colors = set()
    for band in overlaid.bands:
        for _at, stop in band.opts['fillBrush'].gradient().stops():
            colors.add(stop.rgb())
    assert len(colors) == 1


def test_the_bands_are_translucent_enough_to_read_through(qt_app):
    """They are a wash over the trace, not a thing in their own right —
    a quarter opaque at the very peak of the window and less everywhere
    else."""
    overlaid, _plot = overlay(qt_app, Averaging(frame_length=1024, frames=3))
    for band in overlaid.bands:
        _across, alphas = band_gradient(band)
        assert max(alphas) / 255 == pytest.approx(0.25, abs=0.01)
        assert min(alphas) >= 0, 'a negative weight is not a drawable alpha'


# ---- the window, in the rail --------------------------------------------


def test_the_window_is_drawn_in_the_rail_and_not_over_the_data(qt_app):
    """Laid over the trace it fought with it for the same ink. In the
    rail it is read where it is set."""
    overlaid, plot = overlay(
        qt_app, Averaging(frame_length=1024, window='hann', frames=2))
    foot, top = plot.getViewBox().viewRange()[1]
    for _low, _high, baseline, peak in glyphs_of(overlaid):
        assert baseline > foot + 0.5 * (top - foot), 'up in the rail'
        assert peak <= top, 'and inside the view'


def test_a_window_covers_the_whole_of_its_frame(qt_app):
    """It is the frame's own shape, so anything less would misreport
    where the frame starts and stops."""
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=3)
    overlaid, _plot = overlay(qt_app, averaging)
    for (low, high), (left, right, _b, _p) in zip(
            averaging.frame_bounds(RATE), glyphs_of(overlaid)):
        assert left == pytest.approx(low)
        assert right == pytest.approx(high - 1.0 / RATE), (
            'the last sample of the frame, not the first of the next')


def test_the_window_shape_is_the_window_that_will_be_applied(qt_app):
    """Drawing a Hann over frames that will be multiplied by a flat top
    would be a picture of the wrong analysis."""
    from visualdynamics.core.averaging import window_shape

    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=128, window='flattop', frames=1))
    _x, y = overlaid.glyphs[0].getData()
    lifted = np.asarray(y) - overlaid.glyphs[0].opts['fillLevel']
    wanted = window_shape('flattop', 128)
    assert np.allclose(lifted / lifted.max(), wanted / wanted.max())


def test_the_area_under_the_window_is_filled(qt_app):
    """An outline alone is thin at this size; the fill is what makes a
    frame's window read at a glance."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, window='hann', frames=2))
    for glyph in overlaid.glyphs:
        color = fill_color(glyph)
        assert 0 < color.alpha() < 255, 'filled, and not opaque'
        assert glyph.opts['fillLevel'] == pytest.approx(
            float(glyph.getData()[1].min())), 'filled down to its baseline'


def test_every_window_in_the_rail_is_the_same_shade(qt_app):
    """One color throughout. What separates one window from the next is
    its outline and the ticks under its ends, not a change of shade."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=512, overlap=0.0, frames=6))
    assert len({fill_color(glyph).rgba() for glyph in overlaid.glyphs}) == 1


def test_a_tick_marks_where_each_frame_starts_and_stops(qt_app):
    """A tapering window comes back to its baseline at both ends and
    would otherwise say nothing about where it stopped."""
    averaging = Averaging(frame_length=1024, window='hann', frames=3)
    overlaid, _plot = overlay(qt_app, averaging)
    marked = {round(float(at), 6) for tick in overlaid.caps
              for at in np.asarray(tick.getData()[0])}
    for low, high in averaging.frame_bounds(RATE):
        assert round(low, 6) in marked
        assert round(high, 6) in marked


def test_a_tick_crosses_its_baseline(qt_app):
    """Above and below, so it reads against a filled window."""
    overlaid, _plot = overlay(qt_app, Averaging(frame_length=1024, frames=1))
    ticks = np.concatenate([np.asarray(tick.getData()[1])
                            for tick in overlaid.caps])
    baseline = overlaid.glyphs[0].opts['fillLevel']
    assert ticks.min() < baseline < ticks.max()


# ---- the levels ---------------------------------------------------------


def test_frames_that_do_not_overlap_share_one_level(qt_app):
    """Stacking them would waste the height and imply a difference that
    is not there."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=5))
    assert len({round(b, 9) for _l, _r, b, _p in glyphs_of(overlaid)}) == 1


def test_overlapping_frames_are_stacked_onto_levels(qt_app):
    """Two windows in one slot would run through each other exactly
    where the frames share record, which is what the rail is for."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.5, frames=4))
    rail = glyphs_of(overlaid)
    assert len({round(b, 9) for _l, _r, b, _p in rail}) == 2
    assert all(abs(first[2] - second[2]) > 1e-9
               for first, second in itertools.pairwise(rail))


def test_a_deeper_overlap_takes_more_levels(qt_app):
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.75, frames=8))
    assert len({round(b, 9) for _l, _r, b, _p in glyphs_of(overlaid)}) == 4


def test_the_levels_never_touch_each_other(qt_app):
    """A window fills its slot but not the gap above it, or the peak of
    one level would sit in the trough of the next."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.5, frames=4))
    by_level = {}
    for _l, _r, baseline, peak in glyphs_of(overlaid):
        by_level[round(baseline, 9)] = peak
    baselines = sorted(by_level)
    for lower, upper in itertools.pairwise(baselines):
        assert by_level[lower] < upper, 'a level reached into the one above'


def test_the_rail_stays_out_of_the_data(qt_app):
    """However many levels there are. At a fixed step four of them would
    walk down into the trace; they share a fixed slice of the height."""
    from visualdynamics.core.averaging import RAIL_BAND, RAIL_TOP

    for overlap in (0.0, 0.5, 0.75, 0.9):
        overlaid, plot = overlay(
            qt_app, Averaging(frame_length=512, overlap=overlap, frames=12))
        foot, top = plot.getViewBox().viewRange()[1]
        floor = foot + (top - foot) * (RAIL_TOP - RAIL_BAND)
        lowest = min(baseline for _l, _r, baseline, _p in glyphs_of(overlaid))
        assert lowest >= floor - 1e-9, f'overlap {overlap} reached the data'


def test_the_rail_moves_with_the_view(qt_app):
    """It is pinned to the top of what is visible, so zooming must not
    leave it stranded off screen. Inside the plot's own limits: a view
    is locked to its data's extents, plus whatever room the rail asked
    for, and a range outside that is clamped back."""
    overlaid, plot = overlay(qt_app, Averaging(frame_length=1024, frames=3))
    before = [b for _l, _r, b, _p in glyphs_of(overlaid)]
    low, high = plot.getViewBox().viewRange()[1]
    middle = (low + high) / 2.0
    plot.getViewBox().setYRange(middle, high, padding=0)
    after = [b for _l, _r, b, _p in glyphs_of(overlaid)]
    assert all(middle <= baseline <= high for baseline in after), after
    assert after != before


def test_the_marks_keep_their_proportions_when_zoomed(qt_app):
    """A height in data units would shrink to a smear on a taller view."""
    from visualdynamics.core.averaging import GLYPH_SHARE, RAIL_SLOT

    overlaid, plot = overlay(qt_app, Averaging(frame_length=1024, frames=2))
    for low, high in ((-1.0, 1.0), (-10.0, 10.0)):
        plot.getViewBox().setYRange(low, high, padding=0)
        _l, _r, baseline, peak = glyphs_of(overlaid)[0]
        assert (peak - baseline) / (high - low) == pytest.approx(
            RAIL_SLOT * GLYPH_SHARE, rel=1e-6)


# ---- the marks are marks, not data --------------------------------------


def test_the_marks_do_not_move_the_plot(qt_app):
    """They describe the record; letting them vote on the extents would
    rescale the view every time one was drawn, and a taller view draws
    them taller still."""
    from visualdynamics.plot.averaging import AveragingOverlay

    plot = plot_of()
    view = plot.getViewBox()
    before = view.childrenBounds()
    overlaid = AveragingOverlay(plot, Averaging(frame_length=1024, frames=4),
                                RATE, SAMPLES, resolve_theme('light'))
    assert overlaid.bands, 'there were marks to ignore'
    assert view.childrenBounds() == before, 'the marks changed the extents'


def test_the_room_made_for_the_rail_does_not_run_away(qt_app):
    """The rail is given room above the trace, and that is the only
    thing that may move the view. The marks are excluded from the
    extents because they are drawn *from* the extents: counted in, a
    taller view draws them taller, which makes the view taller again.
    Held on a trace at -5, where the rail is above everything.
    """
    from visualdynamics.plot.averaging import AveragingOverlay

    plot = plot_of(offset=-5.0)
    view = plot.getViewBox()
    view.enableAutoRange(True)
    view.updateAutoRange()
    before = view.viewRange()[1]
    assert before[1] < 0.0, 'the trace really is all below zero'

    overlaid = AveragingOverlay(plot, Averaging(frame_length=1024, frames=3),
                                RATE, SAMPLES, resolve_theme('light'))
    made = view.viewRange()[1]
    floor = view.childrenBounds()[1][0]      # the trace's own minimum
    assert made[1] > before[1], 'room was made above the trace'
    assert made[0] <= floor, 'and none taken from under the trace'

    # and it settles there: redrawing the marks against the new range
    # must not ask for more room again
    for _ in range(3):
        overlaid._rescale()
    assert view.viewRange()[1] == pytest.approx(made), 'the view ran away'


def test_the_marks_are_not_counted_as_data(qt_app):
    """Anything reading the curves off a plot — an export, a legend —
    must not find the windows among them."""
    from visualdynamics.plot import data_curves

    overlaid, plot = overlay(qt_app, Averaging(frame_length=1024, frames=2))
    assert len(data_curves(plot)) == 1, 'the trace, not the marks'
    assert overlaid.glyphs, 'and there were marks to skip'


# ---- dragging -----------------------------------------------------------


def drag(overlaid, low, high):
    """The gesture, start to finish.

    setRegion emits both sigRegionChanged and sigRegionChangeFinished,
    which is what a drag and its release do.
    """
    overlaid.region.setRegion((low, high))
    return overlaid.averaging


def test_the_region_spans_the_whole_analysis(qt_app):
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=3, start=1.0)
    overlaid, _plot = overlay(qt_app, averaging)
    assert overlaid.region.getRegion() == pytest.approx((1.0, 4.0))


def test_dragging_the_middle_moves_the_analysis(qt_app):
    """The gesture the whole thing is for: the same frames, elsewhere."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=2))
    moved = drag(overlaid, 2.0, 4.0)
    assert moved.start == pytest.approx(2.0)
    assert moved.frames == 2, 'the same analysis, further along'
    assert bands_of(overlaid) == pytest.approx(moved.frame_bounds(RATE))


def test_dragging_an_edge_out_adds_whole_frames(qt_app):
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=2))
    assert drag(overlaid, 0.0, 4.0).frames == 4


def test_half_a_frame_of_room_adds_no_frame(qt_app):
    """What the snap is for. Dragged 2.5 frames out, it takes two and
    puts the edge back where the second one ends."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=1))
    assert drag(overlaid, 0.0, 2.5).frames == 2
    assert overlaid.region.getRegion() == pytest.approx((0.0, 2.0)), (
        'the edge snapped back onto the last whole frame')


def test_a_drag_cannot_run_off_the_end_of_the_record(qt_app):
    """There is no record there to average."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=2))
    dragged = drag(overlaid, 7.5, 20.0)
    assert dragged.fits(SAMPLES, RATE)
    assert dragged.start <= (SAMPLES - 1024) / RATE
    assert dragged.frames >= 1, 'and there is still an analysis'


def test_a_drag_cannot_start_before_the_record(qt_app):
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=2))
    assert drag(overlaid, -3.0, 2.0).start == 0.0


def test_dragging_shut_leaves_one_frame(qt_app):
    """An analysis of no frames is not an analysis."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=4))
    assert drag(overlaid, 1.0, 1.0).frames == 1


def test_the_overlap_survives_a_drag(qt_app):
    """A drag says where and how many, never how much the frames share
    — that is the table's, and re-deriving it would silently retune the
    analysis every time the region was nudged."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.75, window='hamming',
                          frames=2))
    dragged = drag(overlaid, 1.0, 3.0)
    assert dragged.overlap == 0.75
    assert dragged.window == 'hamming'
    assert dragged.frame_length == 1024


def test_a_drag_is_reported_once(qt_app):
    """The overlay draws and tells; storing the result is not its job.

    Once, though: the release snaps the region onto whole frames, and
    moving a region is itself a finished drag, so the report goes out
    twice unless the snap says it is the one snapping. Dragged to 3.7 s
    so that the snap really moves — landed exactly, it does nothing and
    the bug hides.
    """
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=1))
    seen = []
    overlaid.changed.connect(seen.append)
    drag(overlaid, 1.0, 3.7)
    assert len(seen) == 1, [entry.frames for entry in seen]
    assert seen[0].frames == 2 and seen[0].start == pytest.approx(1.0)
    assert overlaid.region.getRegion() == pytest.approx((1.0, 3.0))


# ---- a capture that is already cut into frames --------------------------


def test_a_locked_overlay_cannot_be_dragged(qt_app):
    """The controller settled the frames; there is nothing here to
    choose, and a region that moved would promise otherwise."""
    overlaid, _plot = overlay(
        qt_app, Averaging.for_records(1024), samples=1024, locked=True)
    assert not overlaid.region.movable
    assert bands_of(overlaid) == pytest.approx([(0.0, 1.0)])


# ---- taking it away -----------------------------------------------------


def test_removing_takes_every_mark_off(qt_app):
    """A redraw that left a mark behind would stack them up until the
    plot was a solid block."""
    overlaid, plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.5, frames=5))
    before = len(plot.items)
    overlaid.remove()
    assert overlaid.bands == [] and overlaid.glyphs == [] and overlaid.caps == []
    assert len(plot.items) == before - 21, (
        '5 bands, 5 windows, a tick at each end of each, the region')
    assert len(plot.items) == 1, 'the trace, and nothing else'


def test_redrawing_does_not_stack_marks_up(qt_app):
    """Every zoom, every drag and every edit redraws; leaking one item
    a time would fill the plot within a session."""
    overlaid, plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.5, frames=3))
    settled = len(plot.items)
    for _ in range(5):
        overlaid.set_averaging(Averaging(frame_length=1024, overlap=0.5,
                                         frames=3))
    assert len(plot.items) == settled


def test_setting_an_averaging_redraws_the_marks(qt_app):
    """What an edit in the table arrives as."""
    overlaid, _plot = overlay(
        qt_app, Averaging(frame_length=1024, overlap=0.0, frames=2))
    wider = Averaging(frame_length=2048, overlap=0.0, window='blackman',
                      frames=3, start=0.5)
    overlaid.set_averaging(wider)
    assert bands_of(overlaid) == pytest.approx(wider.frame_bounds(RATE))
    assert overlaid.region.getRegion() == pytest.approx(
        (0.5, wider.stop(RATE)))


def test_the_rail_is_given_room_above_the_trace(qt_app):
    """Drawn into the height the trace already fills, the windows and
    their ticks land on the data they describe and neither reads. The
    room has to be asked for: the marks are kept out of the extents, so
    they cannot make it by being there."""
    import pyqtgraph as pg

    from visualdynamics.plot.averaging import AveragingOverlay

    plot = plot_of()
    view = plot.getViewBox()
    top = view.childrenBounds()[1][1]
    overlaid = AveragingOverlay(plot, Averaging(frame_length=1024, frames=3),
                                RATE, SAMPLES, resolve_theme('light'))
    foot = min(baseline for _l, _r, baseline, _p in glyphs_of(overlaid))
    assert foot > top, 'the rail still sits on the trace'
    assert isinstance(plot, pg.PlotItem)


def test_more_levels_take_more_room(qt_app):
    """Four lines need more clear height than one, and the trace should
    lose no more of the view than the rail actually uses."""
    def headroom(averaging):
        plot = plot_of()
        top = plot.getViewBox().childrenBounds()[1][1]
        overlay_ = __import__('visualdynamics.plot.averaging', fromlist=['x'])
        marks = overlay_.AveragingOverlay(plot, averaging, RATE, SAMPLES,
                                          resolve_theme('light'))
        _low, high = plot.getViewBox().viewRange()[1]
        assert marks.glyphs
        return high - top

    shallow = headroom(Averaging(frame_length=1024, overlap=0.0, frames=3))
    deep = headroom(Averaging(frame_length=1024, overlap=0.75, frames=8))
    assert deep > shallow * 1.5, (shallow, deep)


def test_the_plots_own_ceiling_comes_back(qt_app):
    """A view is locked to its data so zoom and pan cannot wander into
    blank space. The rail lifts that ceiling to make its room and has to
    put it back, or the plot stays scrollable into nothing."""
    from visualdynamics.plot.averaging import AveragingOverlay

    plot = plot_of()
    view = plot.getViewBox()
    # what build_plots does: locked to the data's own extents
    low, high = view.childrenBounds()[1]
    view.setLimits(yMin=low, yMax=high)
    before = view.state['limits']['yLimits'][:]
    overlaid = AveragingOverlay(plot, Averaging(frame_length=1024, frames=3),
                                RATE, SAMPLES, resolve_theme('light'))
    assert view.state['limits']['yLimits'][1] > before[1], 'room was made'
    overlaid.remove()
    assert view.state['limits']['yLimits'] == before


def test_a_ceiling_already_high_enough_is_left_alone(qt_app):
    """Outward only. Brought down to fit the rail, the lock would take
    away room the user had to pan in."""
    from visualdynamics.plot.averaging import AveragingOverlay

    plot = plot_of()
    view = plot.getViewBox()
    view.setLimits(yMin=-50.0, yMax=50.0)
    AveragingOverlay(plot, Averaging(frame_length=1024, frames=3),
                     RATE, SAMPLES, resolve_theme('light'))
    assert view.state['limits']['yLimits'] == [-50.0, 50.0]
