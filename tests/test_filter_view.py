"""The filter view: the toggle, the panel, the preview, the verbs.

The GUI half of the motion chain (Brandon, 2026-08-24): a taskbar
toggle like the averaging's, a parameter panel beside the plot, and
the filtered trace previewed in green over the raw one — with the
calculator entries that actually make the derived records.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics


def _record(window, pump, dims=None):
    rng = np.random.default_rng(7)
    t = np.arange(8192) / 2048.0
    rows = rng.standard_normal((3, len(t)))
    dims = dims or ['acceleration'] * 2 + ['force']
    history = visualdynamics.TimeHistory(
        t, rows, response_dof=['101Z+', '104Z+', '9001X+'],
        ordinate_dim=dims)
    window.add_object('Record', history)
    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    return history


def _toggle_filter(window, pump):
    pane = window.data_pane
    pane.filter_action.setChecked(True)
    pane._choose_filter(True)
    pump()
    return pane


def _flat(window, pump):
    """Stand the 3-D reading down: the stage is the default for a
    record, so a test of the *flat* preview has to say so."""
    pane = window.data_pane
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    return pane


# ---- the toggle ----------------------------------------------------------


def test_the_button_is_offered_for_a_record_and_not_a_spectrum(window,
                                                               pump):
    _record(window, pump)
    assert window.data_pane.filter_action.isVisible()
    psds = window.project.compute_psds('Record')
    window.show_object(psds)
    item = window._item_for_object(psds)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    assert not window.data_pane.filter_action.isVisible(), \
        'a spectrum is not filtered here — the chain starts at a record'


def test_toggling_it_shows_the_panel_and_the_preview(window, pump):
    history = _record(window, pump)
    pane = _toggle_filter(window, pump)
    _flat(window, pump)
    assert pane.filter_panel.isVisible()
    assert window.filter_overlays, 'a preview overlay is up'
    drawn = sum(len(overlay._previews) for overlay in window.filter_overlays)
    assert drawn == history.num_records, 'one preview per channel'
    corner = pane.filter_panel.corner_box.value()
    assert corner == pytest.approx(history.sample_rate / 10.0), \
        'the panel opens on the suggestion'


def test_the_preview_actually_filters(window, pump):
    """The preview is the low-pass, not a copy: above the corner the
    drawn twin must carry almost nothing of what the raw trace has."""
    history = _record(window, pump)
    _toggle_filter(window, pump)
    _flat(window, pump)
    overlay = window.filter_overlays[0]
    _raw_x, raw_y = overlay._sources[0].getData()
    _x, filtered_y = overlay._previews[0].getData()
    rate = history.sample_rate
    corner = rate / 10.0
    lines = np.fft.rfftfreq(len(raw_y), 1.0 / rate)
    above = lines > 4 * corner
    raw_power = float(np.sum(np.abs(np.fft.rfft(raw_y))[above] ** 2))
    filtered_power = float(
        np.sum(np.abs(np.fft.rfft(filtered_y))[above] ** 2))
    assert filtered_power < raw_power * 1e-4


def test_the_three_marks_are_exclusive_every_way(window, pump):
    """One reading of the trace at a time — all three pairs, both
    directions. Shocks up with the filter still on was exactly the
    both-at-once state Brandon caught (2026-08-24): the first pass
    only excluded the filter against averaging and kurtosis."""
    _record(window, pump)
    pane = window.data_pane
    toggles = {
        'averaging': (pane.averaging_action, pane._choose_averaging,
                      'averaging_wanted'),
        'shocks': (pane.shocks_action, pane._choose_shocks,
                   'shocks_wanted'),
        'filter': (pane.filter_action, pane._choose_filter,
                   'filter_wanted'),
    }
    for first, second in [(a, b) for a in toggles for b in toggles
                          if a != b]:
        action, choose, _wanted = toggles[first]
        action.setChecked(True)
        choose(True)
        pump()
        action2, choose2, _wanted2 = toggles[second]
        action2.setChecked(True)
        choose2(True)
        pump()
        for name, (act, _c, wanted) in toggles.items():
            expected = name == second
            assert getattr(pane, wanted) == expected, \
                f'{first} then {second}: {name} wanted'
            assert act.isChecked() == expected, \
                f'{first} then {second}: {name} checked'
        action2.setChecked(False)
        choose2(False)
        pump()


def test_kurtosis_stands_the_filter_down_but_leaves_its_button(window, pump):
    """The exclusion still holds — one reading at a time — while the
    button stays, because it is how a reader gets back to the filter
    (reversed 2026-08-27; see test_kurtosis for the reasoning)."""
    _record(window, pump)
    pane = _toggle_filter(window, pump)
    pane.kurtosis_action.setChecked(True)
    pane._choose_reading(True)
    pump()
    assert not pane.filter_wanted, 'the reading stood down'
    assert not pane.filter_panel.isVisible(), 'and its panel with it'
    assert pane.filter_action.isVisible(), 'the way back stays offered'


# ---- editing -------------------------------------------------------------


def test_an_edit_lands_on_the_history_and_raises_the_badge(window, pump):
    """The panel stores what it says, and the stored value is what the
    provenance fingerprints — so the edit is the moment a filtered
    record's refresh badge appears."""
    from visualdynamics.core.filters import Filtering

    history = _record(window, pump)
    filtered = window.project.filter_data('Record')
    assert not window.project.stale()
    _toggle_filter(window, pump)
    panel = window.data_pane.filter_panel
    panel.corner_box.setValue(50.0)
    pump()
    assert history.filtering == Filtering(high=50.0, order=4)
    assert filtered in window.project.stale()
    assert window._stale.get(filtered), 'the badge state re-read itself'


def test_the_preview_restates_on_an_edit(window, pump):
    _record(window, pump)
    _toggle_filter(window, pump)
    _flat(window, pump)
    overlay = window.filter_overlays[0]
    _x, before = overlay._previews[0].getData()
    window.data_pane.filter_panel.corner_box.setValue(20.0)
    pump()
    _x, after = overlay._previews[0].getData()
    assert not np.allclose(before, after), \
        'a tighter corner draws a visibly different preview'


# ---- the calculator ------------------------------------------------------


def test_the_bar_offers_the_chain_where_it_applies(window, pump):
    history = _record(window, pump)
    labels = [label for _v, label, *_rest in window.acts_for(['Record'])]
    assert 'Filter Data' not in labels, \
        'the act lives on the panel now (Brandon, 2026-08-28)'
    assert 'Integrate' in labels, 'accelerations integrate'
    assert 'Differentiate' not in labels, \
        'nothing here differentiates — no velocity or displacement'
    window.add_object('Velocity', history.integrate())
    labels = [label for _v, label, *_rest in window.acts_for(['Velocity'])]
    assert 'Integrate' in labels and 'Differentiate' in labels


def test_the_verbs_run_from_the_tree_selection(window, pump):
    _record(window, pump)
    window.filter_data()
    pump()
    assert 'Record Filtered' in window.objects
    item = window._item_for_object('Record Filtered')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.integrate_history()
    pump()
    assert 'Record Filtered Velocity' in window.objects
    velocity = window.objects['Record Filtered Velocity']
    assert velocity.num_records == 2, 'the force channel stayed behind'
    assert set(velocity.ordinate_dim) == {'velocity'}


# ---- the 3-D reading -----------------------------------------------------


def _stage_meshes(window, prefix):
    plotter = window.data_pane.waterfall_plotter
    return {name: actor.mapper.dataset
            for name, actor in plotter.actors.items()
            if name.startswith(prefix) and hasattr(actor, 'mapper')}


def test_the_stage_keeps_its_3d_reading_under_the_filter(window, pump):
    """Brandon selected the filter with 3-D up and the view dropped to
    2-D (2026-08-25). That was my override forcing the drawing flat —
    a judgment about occlusion made without drawing it, and a view
    choice silently overruling another view choice, which this
    interface does not do. Both readings stand now."""
    _record(window, pump)
    pane = window.data_pane
    assert pane.showing_waterfall, 'the stage is the default for a record'
    _toggle_filter(window, pump)
    assert pane.showing_waterfall, 'and it survives the filter view'
    assert pane.filter_panel.isVisible(), 'with the panel beside it'
    assert _stage_meshes(window, 'marks-filter'), \
        'the filtered twins are on the stage'


def test_the_twin_is_normalized_against_the_raw_stage(window, pump):
    """The whole of the correctness here: the twin comes from a second
    `waterfall_arrays` call, and scaled to its own range it would draw
    stretched to the raw ribbon's height — a quieter record looking
    exactly as loud, and the comparison saying nothing. The banded
    stage taught this one (Brandon, 2026-08-25)."""
    from visualdynamics.viz.waterfall import STAGE

    _record(window, pump)
    pane = window.data_pane
    _toggle_filter(window, pump)
    panel = pane.filter_panel
    panel.corner_box.setValue(20.0)     # well under the content
    pump()
    twins = _stage_meshes(window, 'marks-filter')
    assert twins
    _sx, _sy, sz = STAGE
    tallest = max(mesh.bounds[5] - mesh.bounds[4] for mesh in twins.values())
    assert tallest < sz * 0.9, \
        'a heavily filtered record draws shorter than the stage it is on'


def test_the_stage_twin_restates_on_an_edit(window, pump):
    _record(window, pump)
    pane = _toggle_filter(window, pump)
    before = {name: mesh.bounds
              for name, mesh in _stage_meshes(window, 'marks-filter').items()}
    assert before
    pane.filter_panel.corner_box.setValue(15.0)
    pump()
    after = {name: mesh.bounds
             for name, mesh in _stage_meshes(window, 'marks-filter').items()}
    assert after
    assert before != after, 'a tighter corner redraws the twins'


def test_the_filtered_data_takes_the_ink_and_the_raw_stands_back(window,
                                                                pump):
    """Brandon, 2026-08-25: the filtered record is what is being
    decided, so it carries the level colormap on the stage and the
    channel colors flat, while the raw record — the reference —
    stands back in gray. The first pass had it inverted."""
    from visualdynamics.theme import theme as resolve_theme

    _record(window, pump)
    pane = window.data_pane
    gray = resolve_theme(window.theme_name)['specification_curve']

    # on the stage: the raw ribbon is flat gray, the twin is scalared
    _toggle_filter(window, pump)
    plotter = pane.waterfall_plotter
    twin = plotter.actors['marks-filter'].mapper.dataset
    assert 'level' in twin.point_data, \
        'the twin is drawn on the level colormap'
    raw = plotter.actors['waterfall'].mapper
    assert raw.scalar_visibility in (0, False), \
        'the raw ribbon stands back in one color, off the colormap'

    # and flat: the sources go gray, the previews take the palette
    _flat(window, pump)
    overlay = window.filter_overlays[0]
    assert overlay._sources[0].opts['pen'].color().name() == gray
    preview_color = overlay._previews[0].opts['pen'].color().name()
    assert preview_color != gray, 'the filtered curve keeps its color'


def test_putting_the_filter_away_takes_the_twins_with_it(window, pump):
    _record(window, pump)
    pane = _toggle_filter(window, pump)
    assert _stage_meshes(window, 'marks-filter')
    pane.filter_action.setChecked(False)
    pane._choose_filter(False)
    pump()
    assert not _stage_meshes(window, 'marks-filter'), \
        'a preview cannot outlive the view that asked for it'


def test_the_preview_color_is_outside_the_colormap_it_overlays():
    """Measured, not judged by eye (Brandon, 2026-08-25): the first
    preview color was green, chosen against the 2-D palette, and on
    the stage it disappeared — viridis runs purple, teal, *green*,
    yellow, so the twin sat inside the very colormap it had to stand
    out from. Pinned as a distance so a future palette edit cannot
    quietly walk it back in."""
    from visualdynamics.plot import curve_color
    from visualdynamics.theme import VIRIDIS
    from visualdynamics.theme import theme as resolve_theme

    def rgb(text):
        return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))

    def nearest(text, others):
        r, g, b = rgb(text)
        return min(((r - c[0]) ** 2 + (g - c[1]) ** 2
                    + (b - c[2]) ** 2) ** 0.5 for c in others)

    palette = [rgb(curve_color(i)) for i in range(8)]
    for name in ('light', 'dark'):
        preview = resolve_theme(name)['filter_preview']
        assert nearest(preview, VIRIDIS) > 100, \
            f'{name}: the twin would vanish into the stage ribbons'
        assert nearest(preview, palette) > 35, \
            f'{name}: the twin would read as another channel flat'


# ---- the filter's own shape ----------------------------------------------


def test_the_panel_draws_the_filter_it_is_making(window, pump):
    """Brandon, 2026-08-25: the shape should be visible where the
    numbers are set, not inferred from two of them."""
    import numpy as np

    _record(window, pump)
    pane = _toggle_filter(window, pump)
    panel = pane.filter_panel
    curve = panel.response.curve
    hertz, y = curve.xData, curve.yData
    assert hertz is not None and len(hertz) > 100, 'a curve is drawn'
    corner = panel.corner_box.value()
    # Both ends of pyqtgraph's transform, pinned once. The curve is
    # *given* hertz, because a log-mode plot transforms its own data
    # items; what it then *displays* is log10, and the corner line —
    # an InfiniteLine, positioned in view coordinates — has to be
    # given that log10 itself. Handing the curve log10 as well drew
    # the decades twice over (Brandon, 2026-08-25).
    assert hertz[-1] == pytest.approx(panel.sample_rate / 2.0, rel=0.01)
    shown, _shown_y = curve.getData()
    assert shown[-1] == pytest.approx(np.log10(hertz[-1]), abs=1e-6), \
        'displayed once-logged, never twice'
    assert panel.response.lines['high'].value() == pytest.approx(
        np.log10(corner), abs=1e-6), 'the corner is marked where it is'
    assert not panel.response.lines['low'].isVisible(), \
        'a low-pass has no lower edge to mark'
    passband = float(np.interp(corner / 8.0, hertz, y))
    assert passband == pytest.approx(0.0, abs=0.2), 'flat below the corner'
    assert float(np.interp(corner, hertz, y)) < -5.0, \
        'and down at the corner'

    panel.corner_box.setValue(corner / 4.0)
    pump()
    moved = panel.response.curve.xData
    assert panel.response.lines['high'].value() == pytest.approx(
        np.log10(corner / 4.0), abs=1e-6), 'the mark follows the setting'
    assert np.array_equal(moved, hertz), \
        'against a still axis — a scale that follows the corner would ' \
        'make every setting look identical'


def test_the_response_is_what_the_data_sees_not_the_filters_own():
    """`filtered` runs the filter forward and backward, so the record
    is multiplied by |H| twice. Plotting |H| would draw a filter
    nothing here applies: −3 dB at the corner where the real pair is
    at −6.02 (Brandon, 2026-08-25)."""
    import numpy as np
    from scipy.signal import butter, sosfreqz

    from visualdynamics.core.filters import Filtering, response

    rate, corner, order = 8192.0, 800.0, 4
    frequencies, magnitude = response(Filtering(high=corner,
                                                order=order), rate)
    assert float(np.interp(corner, frequencies, magnitude)) == \
        pytest.approx(-6.02, abs=0.05)

    sos = butter(order, corner, 'low', fs=rate, output='sos')
    _w, h = sosfreqz(sos, worN=frequencies, fs=rate)
    single = 20 * np.log10(np.abs(h))
    assert float(np.interp(corner, frequencies, single)) == \
        pytest.approx(-3.01, abs=0.05), 'the one-pass filter, for contrast'
    # above the floor the curve is exactly twice the one-pass dB;
    # below it `response` clamps, on purpose
    alive = single > -100.0
    assert np.allclose(magnitude[alive], 2 * single[alive], atol=1e-6), \
        'the drawn curve is exactly the squared magnitude'


def test_dragging_the_corner_line_sets_the_corner(window, pump):
    """Brandon, 2026-08-25: the line *is* the corner, so dragging it
    is the control — better than a slider beside it, which would be a
    second thing to keep in step with the first."""
    import numpy as np

    _record(window, pump)
    pane = _toggle_filter(window, pump)
    panel = pane.filter_panel
    line = panel.response.lines['high']
    assert line.movable, 'the line takes the mouse'

    heard: list = []
    panel.changed.connect(heard.append)
    wanted = panel.corner_box.value() / 3.0
    line.setValue(np.log10(wanted))
    line.sigDragged.emit(line)          # what the mouse would raise
    pump()

    assert panel.corner_box.value() == pytest.approx(round(wanted, 1)), \
        'the box follows the line'
    # the curve moved with it
    assert float(np.interp(wanted, panel.response.curve.xData,
                           panel.response.curve.yData)) < -5.0
    # but the edit has not been said yet: every listener refilters
    # every visible curve, and per-move that froze the whole panel on
    # a 260-record history (Brandon, 2026-08-28)
    assert not heard, 'mid-drag, the expensive work waits'
    line.sigPositionChangeFinished.emit(line)   # the mouse lets go
    pump()
    assert len(heard) == 1, 'the release says it, once'
    assert heard[-1].high == pytest.approx(round(wanted, 1))


def test_the_line_cannot_be_dragged_past_what_the_filter_allows(window,
                                                                pump):
    """`filtered` refuses a corner at or above Nyquist, so the drag is
    bounded rather than left to fail afterwards — refuse invalid state
    at entry, which is this interface's standing rule."""
    import numpy as np

    _record(window, pump)
    pane = _toggle_filter(window, pump)
    panel = pane.filter_panel
    low, high = panel.response.lines['high'].maxRange
    assert 10.0 ** high < panel.sample_rate / 2.0, 'stops below Nyquist'
    assert 10.0 ** low == pytest.approx(
        panel.response.curve.xData[0], rel=0.01), 'and at the axis edge'
    # and the bound is enforced, not merely declared
    panel.response.lines['high'].setValue(np.log10(panel.sample_rate))
    assert panel.response.lines['high'].value() <= high


# ---- what a drag is allowed to cost --------------------------------------


def test_a_drag_does_not_rebuild_the_report_at_every_step(window, pump):
    """Brandon, 2026-08-25: dragging the corner was slow, and the
    filtering was not why. A report rebuild costs 254 ms on
    shock.vdyn against 3 ms to filter the record — so a drag spent
    almost all of itself redrawing a document nobody could read yet.
    It settles instead: nothing during the gesture, once when the
    hand stops."""
    import numpy as np

    _record(window, pump)
    window.add_object('Report', window.project.generate_report('random'))
    window.show_object('Report')
    pump()
    assert window.report_editor is not None, 'the editor is open'

    rebuilds: list = []
    original = window.report_editor.rebuild
    window.report_editor.rebuild = lambda *a, **k: (rebuilds.append(1),
                                                    original(*a, **k))[1]

    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    pane = _toggle_filter(window, pump)
    line = pane.filter_panel.response.lines['high']
    rebuilds.clear()

    for hz in (300.0, 200.0, 120.0, 80.0, 50.0):
        line.setValue(np.log10(hz))
        line.sigDragged.emit(line)
        pump()
    assert not rebuilds, \
        f'the drag rebuilt the report {len(rebuilds)} times'
    # the release is what carries the edit out of the panel — and the
    # report is off screen while the record is selected, so the edit
    # marks the page stale rather than arming a rebuild for a
    # document nobody is looking at (2026-09-03)
    line.sigPositionChangeFinished.emit(line)
    pump()
    assert not window._report_settle.isActive(), 'nothing pending off screen'
    assert window.report_editor.stale, 'but the page knows it is behind'
    assert not rebuilds

    # and it lands when the report is looked at again: exactly once
    window.show_object('Report')
    pump()
    assert len(rebuilds) == 1, 'exactly once, on show'
    assert not window.report_editor.stale


def test_a_discrete_change_still_rebuilds_at_once(window, pump):
    """Only a gesture defers. Refreshing an object is an act with an
    end, and its report must be right the moment it finishes."""
    _record(window, pump)
    window.add_object('Report', window.project.generate_report('random'))
    window.show_object('Report')
    pump()
    rebuilds: list = []
    original = window.report_editor.rebuild
    window.report_editor.rebuild = lambda *a, **k: (rebuilds.append(1),
                                                    original(*a, **k))[1]
    window._report_content_changed()
    assert len(rebuilds) == 1
    assert not window._report_settle.isActive()


# ---- the other kinds -----------------------------------------------------


def test_the_kind_choice_shows_only_what_applies(window, pump):
    """Principle 3: a low-pass shows one corner, a band-pass its two
    edges — never a grayed pair of controls that cannot be used."""
    _record(window, pump)
    panel = _toggle_filter(window, pump).filter_panel
    assert panel.kind_box.currentText() == 'Low-pass'
    assert panel.corner_box.isVisibleTo(panel)
    assert not panel.from_box.isVisibleTo(panel)
    assert not panel.to_box.isVisibleTo(panel)

    panel.kind_box.setCurrentIndex(2)          # band-pass
    pump()
    assert not panel.corner_box.isVisibleTo(panel)
    assert panel.from_box.isVisibleTo(panel)
    assert panel.to_box.isVisibleTo(panel)


def test_switching_kind_carries_the_corner_and_proposes_a_band(window,
                                                               pump):
    """The corner travels with its meaning: toward a band-pass it
    becomes the top edge, with the bottom proposed a decade under;
    leaving again, the box takes the edge nearest in meaning."""
    from visualdynamics.core.filters import Filtering

    history = _record(window, pump)
    panel = _toggle_filter(window, pump).filter_panel
    panel.corner_box.setValue(400.0)
    pump()
    proposed = panel.from_box.value()          # held from the suggestion

    panel.kind_box.setCurrentIndex(2)          # band-pass
    pump()
    assert history.filtering == Filtering(low=proposed, high=400.0,
                                          order=4), \
        'the corner becomes the top edge; the bottom keeps what it held'

    panel.kind_box.setCurrentIndex(1)          # high-pass
    pump()
    assert history.filtering == Filtering(low=proposed, order=4), \
        'leaving a band-pass toward a high-pass takes the bottom edge'


def test_a_band_pass_draws_two_lines_and_each_writes_its_box(window,
                                                             pump):
    import numpy as np

    _record(window, pump)
    panel = _toggle_filter(window, pump).filter_panel
    panel.kind_box.setCurrentIndex(2)
    pump()
    low, high = panel.response.lines['low'], panel.response.lines['high']
    assert low.isVisible() and high.isVisible()

    low.setValue(np.log10(60.0))
    low.sigDragged.emit(low)
    low.sigPositionChangeFinished.emit(low)
    pump()
    assert panel.from_box.value() == pytest.approx(60.0)

    high.setValue(np.log10(500.0))
    high.sigDragged.emit(high)
    high.sigPositionChangeFinished.emit(high)
    pump()
    assert panel.to_box.value() == pytest.approx(500.0)


def test_a_band_pass_cannot_state_crossed_edges(window, pump):
    """The boxes' ranges are the enforcement: a value typed past the
    other edge clamps one step short of it, rather than raising after
    the fact — refuse invalid state at entry."""
    history = _record(window, pump)
    panel = _toggle_filter(window, pump).filter_panel
    panel.kind_box.setCurrentIndex(2)
    pump()
    panel.to_box.setValue(300.0)
    pump()
    panel.from_box.setValue(900.0)             # past the top edge
    pump()
    assert panel.from_box.value() == pytest.approx(299.9), \
        'clamped one step under the top edge'
    assert history.filtering.kind == 'band-pass'
    assert history.filtering.low < history.filtering.high


def test_the_panel_applies_the_filter_it_is_drawing(window, pump):
    """The act where its settings are set (Brandon, 2026-08-28): the
    button replaced a note apologizing that the record was made
    elsewhere, and makes exactly the filter on show."""
    _record(window, pump)
    panel = _toggle_filter(window, pump).filter_panel
    panel.corner_box.setValue(200.0)
    pump()
    panel.apply_button.click()
    pump()
    cut = window.objects['Record Filtered']
    assert cut is not None
    provenance = window.project.provenance['Record Filtered']
    assert provenance['verb'] == 'filter_data'
    assert window.project['Record'].filtering.high == pytest.approx(200.0)
