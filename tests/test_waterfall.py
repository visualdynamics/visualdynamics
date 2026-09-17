"""The 3-D waterfall: the numbers, the mesh, and the size ceiling.

The scene claims live in `waterfall_arrays`' dict — log mapping, group
choice, labels — so nearly everything here holds numbers, not pixels.
The mesh tests pin the two decisions that carry large models: one
PolyData whatever the record count, and no record bringing more points
than the budget.
"""

import numpy as np
import pytest

from visualdynamics.core.data import Coherence, Psd, TimeHistory
from visualdynamics.decimate import peak_decimate
from visualdynamics.units import DEFAULT_SYSTEM
from visualdynamics.viz.waterfall import (
    LABEL_LIMIT,
    POINT_BUDGET,
    _finite_runs,
    add_waterfall,
    plot_waterfall,
    waterfall_arrays,
    waterfall_group,
)

PSD_DIM = 'acceleration**2/frequency'
PSD_UNIT = '(m/s**2)**2/Hz'


def psd(rows=4, samples=513):
    rng = np.random.default_rng(11)
    f = np.linspace(0.0, 2000.0, samples)
    values = 1e-4 * (1 + rng.random((rows, samples)))
    return Psd(f, values, response_dof=[f'{100 + i}Z+' for i in range(rows)],
               ordinate_dim=[PSD_DIM] * rows, ordinate_unit=[PSD_UNIT] * rows)


# ---- the shared decimator ------------------------------------------------


def test_peak_decimation_keeps_every_extreme():
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, 100_000)
    y = rng.standard_normal(100_000)
    y[31_337] = 40.0   # the one spike that is the reason to look
    y[77_777] = -40.0
    cx, cy = peak_decimate(x, y, 4096)
    assert len(cx) <= 4096
    assert cy.max() == 40.0 and cy.min() == -40.0
    assert np.all(np.diff(cx) >= 0), 'decimation must not reorder x'


def test_short_input_passes_through_untouched():
    x = np.arange(10.0)
    y = x ** 2
    cx, cy = peak_decimate(x, y, 4096)
    assert cx is x and cy is y


def test_a_slice_of_all_gap_keeps_the_gap():
    """nanargmin raises on an all-NaN slice; the gap must survive
    instead, so a specification's out-of-band NaN still reads as a
    hole after decimation."""
    x = np.arange(10_000.0)
    y = np.full(10_000, np.nan)
    y[:100] = 1.0
    cx, cy = peak_decimate(x, y, 100)
    assert len(cx) <= 100
    assert np.isnan(cy).any(), 'the gap thinned away entirely'


# ---- how an object says its axis reads -----------------------------------


def test_the_object_answers_for_its_own_axis():
    assert psd().log_scaled() is True
    history = TimeHistory(np.linspace(0, 1, 64), np.ones((2, 64)),
                          response_dof=['1Z+', '2Z+'])
    assert history.log_scaled() is False
    coherence = Coherence(np.linspace(0, 100, 64), np.ones((1, 64)),
                          response_dof=['1Z+'], reference_dof=['2Z+'])
    assert coherence.log_scaled() is False, 'a 0..1 ratio never reads log'


# ---- which records draw together -----------------------------------------


def test_mixed_quantities_draw_the_largest_group():
    """One floor cannot hold accelerations and volts; the bigger group
    draws and the rest are named for the status bar."""
    f = np.linspace(0.0, 100.0, 65)
    data = Psd(f, np.ones((5, 65)),
               response_dof=[f'{i}Z+' for i in range(1, 6)],
               ordinate_dim=[PSD_DIM] * 3 + ['voltage**2/frequency'] * 2,
               ordinate_unit=[PSD_UNIT] * 3 + ['V**2/Hz'] * 2)
    drawn, left_out = waterfall_group(data)
    assert drawn == [0, 1, 2]
    assert left_out == [3, 4]


def test_a_record_pick_narrows_the_group():
    drawn, left_out = waterfall_group(psd(rows=6), records=[5, 1, 3])
    assert drawn == [5, 1, 3], 'the pick is taken as given, in its order'
    assert left_out == []


# ---- the arrays ----------------------------------------------------------


def test_log_data_maps_to_log10_exactly():
    data = psd(rows=2, samples=257)   # under budget: decimation is identity
    arrays = waterfall_arrays(data)
    assert arrays['log_scaled'] is True
    _cx, cz = arrays['curves'][0]
    # a PSD draws flat across each bin, so every value appears on both
    # of its bin's edges — the log mapping rides the step outline
    expected = np.repeat(
        np.log10(data.display_ordinate(DEFAULT_SYSTEM, [0])[0]), 2)
    assert np.allclose(cz, expected)


def test_zero_is_a_gap_not_a_value():
    """A log axis fed a zero once dragged a report 300 decades down;
    here it must become NaN and then a split line, never a point."""
    f = np.linspace(0.0, 100.0, 129)
    values = np.ones((1, 129))
    values[0, 40:60] = 0.0
    data = Psd(f, values, response_dof=['1Z+'],
               ordinate_dim=[PSD_DIM], ordinate_unit=[PSD_UNIT])
    arrays = waterfall_arrays(data)
    _cx, cz = arrays['curves'][0]
    assert np.isnan(cz[80:120]).all(), (
        'the gap rides the step outline, two points per bin')


def test_time_data_stays_linear():
    t = np.linspace(0.0, 1.0, 200)
    history = TimeHistory(t, np.vstack([np.sin(9 * t), np.cos(9 * t)]),
                          response_dof=['1Z+', '2Z+'],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['m/s**2'] * 2)
    arrays = waterfall_arrays(history)
    assert arrays['log_scaled'] is False
    assert arrays['xlabel'] == 'Time (s)'


def test_labels_are_the_record_labels():
    data = psd(rows=3)
    arrays = waterfall_arrays(data)
    assert arrays['labels'] == [data.record_label(i) for i in range(3)]


def test_axis_titles_are_ascii():
    """VTK's axis titles silently drop unicode superscripts —
    (in/s²)²/Hz read (in/s)/Hz, wrong by two squarings — so the titles
    must carry caret exponents that every font has."""
    from visualdynamics.units import IN_LBF_S

    # the coherent system named explicitly: the default displays
    # acceleration in g, whose label has no superscript to convert,
    # and this test is about the conversion
    arrays = waterfall_arrays(psd(), unit_system=IN_LBF_S)
    assert arrays['zlabel'] == 'log10 (in/s^2)^2/Hz'
    assert arrays['xlabel'] == 'Frequency (Hz)'
    assert arrays['zlabel'].isascii() and arrays['xlabel'].isascii()


# ---- gaps become connectivity, never NaN points --------------------------


def test_finite_runs_split_on_gaps():
    z = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0])
    assert _finite_runs(z) == [(0, 2), (4, 7)]


def test_a_lone_point_between_gaps_draws_nothing():
    z = np.array([np.nan, 1.0, np.nan])
    assert _finite_runs(z) == []


def test_no_finite_values_no_runs():
    assert _finite_runs(np.full(5, np.nan)) == []


# ---- the mesh ------------------------------------------------------------


def scene(data, **kwargs):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True)
    info = add_waterfall(plotter, data, **kwargs)
    return plotter, info


def test_the_whole_set_is_one_mesh_under_budget():
    """The two large-model decisions, pinned: one PolyData whatever the
    record count, and no record past the point budget.

    Driven by a *time history*, deliberately. This used to use a PSD,
    which quietly made it a test of two rules that have since parted
    company: a stepped spectrum is not budgeted at all, because
    thinning its outline destroys the treads and the area with them.
    The budget is for records, which is what it was always for.
    """
    t = np.linspace(0.0, 10.0, 20_000)
    rng = np.random.default_rng(9)
    data = TimeHistory(t, rng.standard_normal((30, 20_000)),
                       response_dof=[f'{100 + i}Z+' for i in range(30)],
                       ordinate_dim=['acceleration'] * 30,
                       ordinate_unit=['m/s**2'] * 30)
    plotter, info = scene(data, budget=512)
    mesh = plotter.renderer.actors['waterfall'].mapper.dataset
    assert info['points'] == mesh.n_points
    assert mesh.n_points <= 30 * 512
    assert np.isfinite(mesh.points).all(), 'gaps must be connectivity, not NaN points'
    assert mesh.n_points < 30 * 20_000, 'the budget did not bite'
    plotter.close()


def test_a_stepped_spectrum_is_still_one_mesh():
    """The other half of the pair, kept for steps: however many bins
    and records, the scene is one PolyData under one actor."""
    plotter, info = scene(psd(rows=12, samples=3_000))
    mesh = plotter.renderer.actors['waterfall'].mapper.dataset
    assert info['points'] == mesh.n_points == 12 * 2 * 3_000
    assert len(plotter.renderer.actors) >= 1
    plotter.close()


def test_one_color_scale_for_the_whole_scene():
    """Color is the level itself, point for point on one shared scale —
    never re-scaled per channel, which would paint every channel as if
    it peaked alike and hide exactly the comparison a waterfall is for.
    Point-for-point, not endpoints: a per-channel rescaling that kept
    the global extremes would pass a min/max check and still lie."""
    data = psd(rows=3, samples=257)   # under budget: order is record order
    plotter, _info = scene(data)
    mesh = plotter.renderer.actors['waterfall'].mapper.dataset
    levels = mesh.point_data['level']
    expected = np.repeat(
        np.log10(data.display_ordinate(DEFAULT_SYSTEM, [0, 1, 2])), 2,
        axis=1)
    assert np.allclose(levels, expected.ravel())
    # and the mapper reads that one range, so a quiet channel four
    # decades down draws four decades darker
    low, high = plotter.renderer.actors['waterfall'].mapper.scalar_range
    assert np.isclose(low, np.nanmin(expected))
    assert np.isclose(high, np.nanmax(expected))
    plotter.close()


def test_every_channel_is_named_until_naming_is_a_smear():
    few, few_info = scene(psd(rows=12))
    assert few_info['named'] == list(range(12))
    few.close()
    many, many_info = scene(psd(rows=3 * LABEL_LIMIT))
    assert len(many_info['named']) <= LABEL_LIMIT
    assert 0 in many_info['named'], 'the first record anchors the axis'
    many.close()


def test_a_gapped_record_splits_into_segments():
    f = np.linspace(0.0, 100.0, 129)
    values = np.ones((1, 129))
    values[0, 60:70] = 0.0    # zero -> NaN on a log axis -> a gap
    data = Psd(f, values, response_dof=['1Z+'],
               ordinate_dim=[PSD_DIM], ordinate_unit=[PSD_UNIT])
    plotter, _info = scene(data)
    mesh = plotter.renderer.actors['waterfall'].mapper.dataset
    assert mesh.n_lines == 2, 'the gap must split the line, not bridge it'
    plotter.close()


# ---- headless ------------------------------------------------------------


def test_plot_waterfall_renders_to_a_file(tmp_path):
    path = tmp_path / 'waterfall.png'
    img = plot_waterfall(psd(), screenshot=str(path))
    assert path.exists() and path.stat().st_size > 0
    assert img.ndim == 3


def test_point_budget_is_the_module_default():
    """`POINT_BUDGET` is what the GUI will pass nothing to get; a
    change to it is a change to every scene and should trip a test."""
    assert POINT_BUDGET == 4096


def test_the_object_method_is_the_same_renderer(tmp_path):
    """`psd.plot_waterfall(...)` is the guide's promise; it must reach
    the one implementation rather than growing a second."""
    path = tmp_path / 'from_method.png'
    img = psd().plot_waterfall(screenshot=str(path))
    assert path.exists() and img.ndim == 3


def test_one_record_still_stands_on_the_stage():
    """A single record is a legitimate waterfall — one line at station
    zero, colored by level — since the reading is the default now and
    a lone channel must not be the case that breaks it."""
    plotter, info = scene(psd(rows=1))
    mesh = plotter.renderer.actors['waterfall'].mapper.dataset
    assert mesh.n_points > 0
    assert info['named'] == [0]
    assert np.isfinite(mesh.points).all()
    plotter.close()


def test_a_pinned_axis_pins_the_stage():
    """A coherence's wall of ones must read as the wall it is — too few
    averages — not as mountains. The object's own `ordinate_limits`
    pin the stage and the color scale exactly as they pin the 2-D
    axis, so a 0.94..0.96 band sits high and flat, not full-height."""
    from visualdynamics.core.data import MultipleCoherence
    from visualdynamics.viz.waterfall import STAGE

    f = np.linspace(0.0, 100.0, 129)
    rng = np.random.default_rng(2)
    values = 0.95 + 0.01 * rng.standard_normal((3, 129))
    coherence = MultipleCoherence(f, values,
                                  response_dof=['1Z+', '2Z+', '3Z+'])
    plotter, _info = scene(coherence)
    actor = plotter.renderer.actors['waterfall']
    assert tuple(actor.mapper.scalar_range) == (0.0, 1.05)
    z = np.asarray(actor.mapper.dataset.points)[:, 2]
    sz = STAGE[2]
    assert z.max() < 0.95 * sz, 'the wall must not fill the stage'
    assert z.min() > 0.8 * sz, 'and it sits high, where its values are'
    plotter.close()


# ---- the hard rule: a density draws as the energy beneath it -------------


def test_the_area_under_the_drawn_psd_is_its_own_integral():
    """The rule, stated as arithmetic: flat across each bin means the
    trapezoid under the trace on the stage equals sum(G * df) — the
    RMS-squared is literally the area drawn, in 3-D as in 2-D."""
    from visualdynamics.plot import bin_edges

    data = psd(rows=1, samples=65)     # under budget: nothing decimated
    arrays = waterfall_arrays(data)
    cx, cz = arrays['curves'][0]
    drawn_area = np.trapezoid(10.0 ** cz, cx)
    values = data.display_ordinate(DEFAULT_SYSTEM, [0])[0]
    widths = np.diff(bin_edges(data.abscissa))
    assert np.isclose(drawn_area, float(np.sum(values * widths)),
                      rtol=1e-12), 'the picture and the integral disagree'


def test_an_octave_psd_steps_on_its_own_band_edges():
    """A banded spectrum carries its own bin widths, and the trace must
    land on those exact edges — midpoints between centers are wrong for
    bands whose centers are geometric means."""
    from visualdynamics.core.octave import bin_bounds

    centers = np.array([10.0, 20.0, 40.0])
    widths = np.array([7.0, 14.0, 28.0])
    data = Psd(centers, np.ones((1, 3)), response_dof=['1Z+'],
               ordinate_dim=[PSD_DIM], ordinate_unit=[PSD_UNIT],
               bandwidth=widths)
    arrays = waterfall_arrays(data)
    cx, _cz = arrays['curves'][0]
    lower, upper = bin_bounds(centers, widths)
    expected = np.repeat(np.concatenate([lower, [upper[-1]]]), 2)[1:-1]
    assert np.allclose(cx, expected), 'the bands, not the midpoints'


def test_a_specification_follows_its_power_law():
    """log_log breakpoints mean the power law between them: halfway in
    log frequency the drawn value is the log-interpolated one, not the
    straight-line midpoint a polyline would take."""
    from visualdynamics.core.data import Specification

    spec = Specification(np.array([10.0, 100.0]),
                         np.array([[1e-4, 1e-2]]),
                         response_dof=['1Z+'],
                         ordinate_dim=[PSD_DIM], ordinate_unit=[PSD_UNIT])
    assert spec.interpolation == 'log_log'
    arrays = waterfall_arrays(spec)
    cx, cz = arrays['curves'][0]
    assert len(cx) > 2, 'filled in, not two breakpoints joined straight'
    at = np.argmin(np.abs(cx - np.sqrt(10.0 * 100.0)))
    display = spec.display_ordinate(DEFAULT_SYSTEM, [0])[0]
    law_mid = np.sqrt(display[0] * display[1])
    assert np.isclose(10.0 ** cz[at], law_mid, rtol=1e-2), \
        'halfway in log frequency, the geometric mean of the breakpoints'


def test_a_signed_component_is_never_a_density():
    """The real part of a CPSD is signed; drawing it as steps would
    claim an area for a quantity that has none."""
    from visualdynamics.plot import drawing_shape

    data = psd(rows=2, samples=65)
    assert drawing_shape(data, tagged=False) == 'steps'
    assert drawing_shape(data, tagged=True) == 'line'


def test_a_stepped_spectrum_keeps_every_tread_past_the_budget():
    """The bug Brandon found in transient.vdyn: the step outline is two
    points per bin, so it crosses the point budget at half the line
    count a plain curve would — and peak decimation keeps each slice's
    *extremes*, not its bin edges, so the treads were exactly what got
    thinned away. The staircase came back a polyline, and the area
    under it stopped being sum(G*df).

    A spectrum is thousands of lines where a record is hundreds of
    thousands, so a step curve is not decimated at all — the same call
    the 2-D plot makes, for the same stated reason.
    """
    from visualdynamics.plot import bin_edges

    # comfortably past POINT_BUDGET once doubled into an outline
    lines = POINT_BUDGET // 2 + 500
    data = psd(rows=2, samples=lines)
    arrays = waterfall_arrays(data)
    cx, cz = arrays['curves'][0]
    assert len(cx) == 2 * lines, 'every bin keeps both of its edges'

    # and the rule the thinning had broken: the area drawn is the
    # area summed, exactly
    drawn = np.trapezoid(10.0 ** cz, cx)
    values = data.display_ordinate(DEFAULT_SYSTEM, [0])[0]
    widths = np.diff(bin_edges(data.display_abscissa(DEFAULT_SYSTEM)))
    assert np.isclose(drawn, float(np.sum(values * widths)), rtol=1e-12)


def test_a_long_record_is_still_decimated():
    """The budget is for time histories, which is what it was always
    for — dropping it for steps must not drop it for everything."""
    t = np.linspace(0.0, 10.0, 200_000)
    rng = np.random.default_rng(4)
    history = TimeHistory(t, rng.standard_normal((2, 200_000)),
                          response_dof=['1Z+', '2Z+'],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['m/s**2'] * 2)
    arrays = waterfall_arrays(history)
    cx, _cz = arrays['curves'][0]
    assert len(cx) <= POINT_BUDGET, 'a record still thins to the budget'


# ---- paging, when one scene cannot hold every record --------------------


def test_a_scene_holds_what_it_can_draw_exactly_and_pages_the_rest():
    """The budget is spent as a record cap, never as more thinning:
    what is drawn is drawn exactly, and the overflow is a page away.
    A full CPSD from a 30-channel survey is 900 records — an ordinary
    modal survey, and 29.5M vertices at 16k lines."""
    from visualdynamics.viz.waterfall import SCENE_BUDGET, records_per_page

    lines = 4001
    data = psd(rows=40, samples=lines)
    budget = 20 * 2 * lines            # room for exactly twenty records
    arrays = waterfall_arrays(data, scene_budget=budget)
    assert arrays['per_page'] == 20
    assert arrays['pages'] == 2
    assert arrays['page'] == 0
    assert len(arrays['drawn']) == 20
    assert arrays['of_quantity'] == 40
    # and every record that *is* drawn keeps every tread
    cx, _cz = arrays['curves'][0]
    assert len(cx) == 2 * lines

    later = waterfall_arrays(data, scene_budget=budget, page=1)
    assert later['page'] == 1
    assert later['drawn'] == arrays['drawn'][:0] + list(range(20, 40))
    assert not set(later['drawn']) & set(arrays['drawn']), 'no overlap'

    # the real default holds a 900-record CPSD in three pages
    assert records_per_page(data, 900, 16385, 'steps', SCENE_BUDGET) == 305


def test_a_page_past_the_end_lands_on_the_last_one():
    data = psd(rows=10, samples=1001)
    budget = 3 * 2 * 1001
    arrays = waterfall_arrays(data, scene_budget=budget, page=99)
    assert arrays['page'] == arrays['pages'] - 1
    assert arrays['drawn'], 'a clamped page still draws'


def test_one_record_draws_even_when_it_alone_is_over_budget():
    """A page showing nothing is not a page."""
    from visualdynamics.viz.waterfall import records_per_page

    assert records_per_page(None, 5, 1_000_000, 'steps', budget=10) == 1


def test_a_small_object_has_one_page_and_says_so():
    arrays = waterfall_arrays(psd(rows=4))
    assert arrays['pages'] == 1 and arrays['page'] == 0


def test_finish_scene_is_how_every_stage_ends(tmp_path):
    """One tail for banded, sine and paired: `screenshot` renders the
    file and hands back the image with the plotter closed; otherwise
    the plotter comes back for the caller to drive. The paired stage
    used to return None from a screenshot while its siblings returned
    the image — the one function is what keeps the three agreeing."""
    import pyvista as pv

    from visualdynamics.viz.waterfall import finish_scene

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    path = tmp_path / 'stage.png'
    image = finish_scene(plotter, str(path), show=True)
    assert path.stat().st_size > 1000
    assert image.ndim == 3, 'the rendered frame comes back'

    plotter = pv.Plotter(off_screen=True)
    kept = finish_scene(plotter, None, show=False)
    assert kept is plotter, 'no file asked: the scene is the caller\'s'
    plotter.close()


# ---- the axis tells the truth --------------------------------------------


def test_nice_axis_widens_only_when_nearly_free():
    """A 0.4998 s record's five even divisions put ticks at 0.125 and
    0.375, and the one-decimal default *printed* them as 0.1 and 0.4 —
    numbers that do not sit where they claim. Brandon read a 0.5 s
    record as 0.6 s against the truncation boxes' honest limits
    (2026-08-28). Widened to 0..0.5 with six labels, every tick is
    round and true; a range that cannot be widened cheaply keeps its
    exact ends instead."""
    from visualdynamics.viz.waterfall import nice_axis

    assert nice_axis(0.0, 0.499755859375) == (0.0, 0.5, 6)
    assert nice_axis(-0.1, 0.4) == pytest.approx((-0.1, 0.4, 6))
    # a 1024 Hz Nyquist would widen to 1200 — a fifth of the stage
    # holding no data — so it keeps its exact range
    assert nice_axis(0.0, 1024.0) == (0.0, 1024.0, 5)
    # degenerate spans change nothing
    assert nice_axis(0.0, 0.0) == (0.0, 0.0, 5)


def test_the_drawn_time_labels_stand_where_they_claim(window, pump):
    """The whole point, read off the rendered strings: each label,
    parsed back, equals the position it is drawn at. The nice case
    comes out round; the exact-range case comes out at the true even
    divisions in the house %.4g format — never rounded to a place a
    tick does not sit."""
    from visualdynamics.core.data import TimeHistory

    def labels_of(pane):
        axes = pane.waterfall_plotter.renderer.cube_axes_actor
        drawn = axes.GetAxisLabels(0)
        return (axes.GetXAxisRange(),
                [drawn.GetValue(i)
                 for i in range(drawn.GetNumberOfValues())])

    def show(name, seconds, rate=2048.0):
        t = np.arange(int(rate * seconds)) / rate
        window.add_object(name, TimeHistory(
            t, np.random.default_rng(1).standard_normal((2, len(t))),
            response_dof=['101Z+', '104Z+']))
        item = window._item_for_object(name)
        window.tree.clearSelection()
        item.setSelected(True)
        window.tree.setCurrentItem(item)
        window.render_current()
        pump()

    show('Half', 0.5)                    # 0..0.49951: widens to 0.5
    (low, high), drawn = labels_of(window.data_pane)
    assert (low, high) == (0.0, 0.5)
    assert drawn == ['0', '0.1', '0.2', '0.3', '0.4', '0.5']

    show('Odd', 1.7)                     # widening would cost ~18%
    (low, high), drawn = labels_of(window.data_pane)
    assert high == pytest.approx(1.7, abs=1e-3), 'the exact range held'
    positions = np.linspace(low, high, len(drawn))
    for text, at in zip(drawn, positions):
        assert float(text) == pytest.approx(at, rel=5e-4), \
            f'label {text!r} is drawn at {at:.6g} and must say so'
