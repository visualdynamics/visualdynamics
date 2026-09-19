"""Every banded spectrum on the stage: channels receding, bands along.

The sine stage's reading, generalized (Brandon, 2026-08-22): the
depth axis is the channel, a specification alone shows every
channel's requirement with its zones at once, and a comparison shows
every channel's measurement with the lines outside an abort limit
boxed over their own bins — the 2-D shading, in depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Psd, Specification

FREQ = np.linspace(20.0, 2000.0, 100)


def _specification(channels=('101Z+', '104Z+', '110Z+')):
    level = np.full((len(channels), len(FREQ)), 1e-2)
    return Specification(
        abscissa=FREQ, ordinate=level,
        response_dof=list(channels),
        ordinate_dim=['acceleration**2/frequency'] * len(channels),
        warning_lower=level * 10 ** (-3 / 10),
        warning_upper=level * 10 ** (3 / 10),
        abort_lower=level * 10 ** (-6 / 10),
        abort_upper=level * 10 ** (6 / 10))


def _measured(loud_lines=0, channels=('101Z+', '104Z+', '110Z+')):
    """Flat at the target, except `loud_lines` lines on the first
    channel pushed past the upper abort."""
    values = np.full((len(channels), len(FREQ)), 1e-2)
    if loud_lines:
        values[0, 40:40 + loud_lines] = 1e-2 * 10 ** (9 / 10)
    return Psd(abscissa=FREQ, ordinate=values,
               response_dof=list(channels),
               ordinate_dim=['acceleration**2/frequency'] * len(channels))


def test_the_spec_alone_stations_every_channel():
    from visualdynamics.viz.banded import banded_stage_arrays

    arrays = banded_stage_arrays(_specification())
    assert [s['dof'] for s in arrays['stations']] == \
        ['101Z+', '104Z+', '110Z+']
    first = arrays['stations'][0]
    assert sorted(first['limits']) == ['abort_lower', 'abort_upper',
                                      'warning_lower', 'warning_upper']
    assert not first['exceed'], 'nothing measured, nothing exceeded'


def test_a_target_with_cross_terms_stations_each_channels_auto():
    """A virtual point's target states every pair (zeros, no
    coherence), so its records are a full matrix. The first record
    wearing a channel is then a cross term, and taking it as the
    station's target drew one channel of three (Brandon, 2026-09-07):
    a station is its channel's autospectrum wherever that record sits.
    """
    from visualdynamics.viz.banded import banded_stage_arrays

    channels = ['M1', 'M2', 'M3']
    autos = np.array([1e-2, 2e-2, 3e-2])
    rows, resp, refs, limits = [], [], [], {n: [] for n in (
        'warning_lower', 'warning_upper', 'abort_lower', 'abort_upper')}
    for a, da in enumerate(channels):
        for b, db in enumerate(channels):
            rows.append(np.full(len(FREQ), autos[a] if a == b else 0.0,
                                dtype=complex))
            resp.append(da)
            refs.append(db)
            for name, db_ in (('warning_lower', -3), ('warning_upper', 3),
                              ('abort_lower', -6), ('abort_upper', 6)):
                limits[name].append(np.full(len(FREQ), autos[a] * 10 ** (db_ / 10))
                                    if a == b else np.full(len(FREQ), np.nan))
    spec = Specification(
        abscissa=FREQ, ordinate=np.asarray(rows), response_dof=resp,
        reference_dof=refs, ordinate_dim=['acceleration**2/frequency'] * 9,
        **{name: np.asarray(v) for name, v in limits.items()})
    arrays = banded_stage_arrays(spec)
    assert [s['dof'] for s in arrays['stations']] == channels
    # judged against each other, so the display system's unit is no
    # part of the claim: every target finite, each a decade ratio
    # above the first, its bands a fixed step off it
    first = arrays['stations'][0]['target']
    assert np.isfinite(first).all()
    for station, level in zip(arrays['stations'], autos):
        assert np.isfinite(station['target']).all(), station['dof']
        assert np.allclose(station['target'] - first,
                           np.log10(level / autos[0])), station['dof']
        assert sorted(station['limits']) == ['abort_lower', 'abort_upper',
                                             'warning_lower', 'warning_upper']
        assert np.allclose(station['limits']['abort_upper'] - station['target'],
                           0.6)


def test_exceedances_are_judged_where_the_table_judges():
    from visualdynamics.viz.banded import banded_stage_arrays

    arrays = banded_stage_arrays(_specification(), _measured(loud_lines=5))
    exceed = {s['dof']: s['exceed'] for s in arrays['stations']}
    assert len(exceed['101Z+']) == 1, 'the loud channel, one bound'
    assert exceed['101Z+'][0]['out'].sum() == 5, 'the five planted lines'
    assert not exceed['104Z+'] and not exceed['110Z+']


def test_the_stage_boxes_each_line_over_its_own_bin():
    import pyvista as pv

    from visualdynamics.viz.banded import add_banded_stage

    plotter = pv.Plotter(off_screen=True)
    arrays = add_banded_stage(plotter, _specification(),
                              _measured(loud_lines=5))
    boxes = [name for name in plotter.actors
             if 'exceed_over' in name and name.startswith('banded-')]
    assert len(boxes) == 1
    mesh = plotter.actors[boxes[0]].mapper.dataset
    assert mesh.n_cells == 5, 'one rectangle per exceeding line'
    for c in range(mesh.n_cells):
        heights = np.round(np.asarray(mesh.get_cell(c).points)[:, 2], 6)
        values, counts = np.unique(heights, return_counts=True)
        assert len(values) == 2 and list(counts) == [2, 2], \
            'rectangles, two corners on each height'
    zones = [name for name in plotter.actors if 'zone' in name]
    assert len(zones) == 4 * 3, 'four zone ribbons per channel'
    assert arrays['points'] > 0
    plotter.close()


def test_a_lone_spec_wears_the_flat_plots_own_pen():
    """The 2-D reading draws a specification in response_curve; the
    stage drew it tab10 blue, which read as a different object."""
    import pyvista as pv

    from visualdynamics.theme import theme
    from visualdynamics.viz.banded import add_banded_stage

    plotter = pv.Plotter(off_screen=True)
    add_banded_stage(plotter, _specification())
    targets = [name for name in plotter.actors if 'target' in name]
    assert targets
    expected = pv.Color(theme(None)['response_curve']).float_rgb
    for name in targets:
        drawn = tuple(plotter.actors[name].prop.color.float_rgb)
        assert drawn == pytest.approx(expected, abs=0.01), name
    plotter.close()


def test_picked_spec_records_restrict_the_stations():
    from visualdynamics.viz.banded import banded_stage_arrays

    arrays = banded_stage_arrays(_specification(),
                                 specification_records=[1])
    assert [s['dof'] for s in arrays['stations']] == ['104Z+']
    assert not arrays['stations'][0]['cross']


def _cross_term_target(coherence=0.5):
    """Three channels with every pair stated at `coherence`, in phase."""
    channels = ['M1', 'M2', 'M3']
    autos = np.array([1e-2, 2e-2, 3e-2])
    rows, resp, refs, limits = [], [], [], {n: [] for n in (
        'warning_lower', 'warning_upper', 'abort_lower', 'abort_upper')}
    for a, da in enumerate(channels):
        for b, db in enumerate(channels):
            level = autos[a] if a == b else np.sqrt(coherence * autos[a] * autos[b])
            rows.append(np.full(len(FREQ), level, dtype=complex))
            resp.append(da)
            refs.append(db)
            for name, db_ in (('warning_lower', -3), ('warning_upper', 3),
                              ('abort_lower', -6), ('abort_upper', 6)):
                limits[name].append(np.full(len(FREQ), autos[a] * 10 ** (db_ / 10))
                                    if a == b else np.full(len(FREQ), np.nan))
    return Specification(
        abscissa=FREQ, ordinate=np.asarray(rows), response_dof=resp,
        reference_dof=refs, ordinate_dim=['acceleration**2/frequency'] * 9,
        **{name: np.asarray(v) for name, v in limits.items()}), autos


def test_picked_cross_terms_are_stationed_as_their_magnitudes():
    """Brandon, 2026-09-07: nine records of a virtual point's target
    picked, and the stage showed three, "never a cross-term label". A
    picked cross term is its own station — the magnitude, no bands,
    labeled by its pair — beside the autos with theirs."""
    from visualdynamics.viz.banded import banded_stage_arrays

    spec, _autos = _cross_term_target()
    arrays = banded_stage_arrays(spec, specification_records=[0, 1, 4, 5])
    stations = arrays['stations']
    assert [s['dof'] for s in stations] == ['M1', 'M1/M2', 'M2', 'M2/M3']
    assert [s['cross'] for s in stations] == [False, True, False, True]
    assert all(np.isfinite(s['target']).all() for s in stations)
    m1, cross, m2 = stations[0], stations[1], stations[2]
    assert cross['limits'] == {}, 'a cross term has no bands'
    assert sorted(m1['limits']) == ['abort_lower', 'abort_upper',
                                    'warning_lower', 'warning_upper']
    # |M1/M2| = sqrt(0.5 · M1 · M2): half a decade's worth below the
    # autos' geometric mean, in the same display unit as they are
    assert np.allclose(cross['target'] - (m1['target'] + m2['target']) / 2,
                       np.log10(np.sqrt(0.5)))
    # the object picked whole stays the channels' reading
    whole = banded_stage_arrays(spec)
    assert [s['dof'] for s in whole['stations']] == ['M1', 'M2', 'M3']
    # a pair stated as zero coherence is zero at every line: stationed,
    # labeled, and nothing to draw on a log axis
    zero, _autos = _cross_term_target(coherence=0.0)
    blank = banded_stage_arrays(zero, specification_records=[1])['stations'][0]
    assert blank['dof'] == 'M1/M2' and blank['cross']
    assert not np.isfinite(blank['target']).any()


def test_the_window_says_how_many_cross_terms_are_on_the_stage(window, pump):
    spec, _autos = _cross_term_target(coherence=0.0)
    window.add_object('Target', spec)
    pump()
    window.tree.setCurrentItem(window._item_for_object('Target'))
    window.tree.clearSelection()
    grid = window.record_grids['Target']
    for row, column in ((0, 0), (0, 1), (1, 1), (2, 2)):
        grid.item(row, column).setSelected(True)
    pump()
    window.data_pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    said = window.statusBar().currentMessage()
    assert '3 channels with their bands and 1 cross term' in said, said
    assert '1 of them zero at every line' in said


def test_the_comparison_defaults_to_the_stage(window, pump):
    window.add_object('Specification', _specification())
    window.add_object('PSD', _measured(loud_lines=5))
    window._item_for_object('Specification').setSelected(True)
    window._item_for_object('PSD').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible()
    actors = list(pane.waterfall_plotter.actors)
    assert any('measured' in name for name in actors)
    assert any('exceed_over' in name for name in actors)
    # the toggle stands down to the flat paged comparison
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert not pane._waterfall_page.isVisible()


def test_the_spec_alone_defaults_to_the_stage(window, pump):
    window.add_object('Specification', _specification())
    window._item_for_object('Specification').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible()
    names = [name for name in pane.waterfall_plotter.actors
             if name.startswith('banded-')]
    assert sum('target' in name for name in names) == 3, \
        'every channel of the requirement, on one stage'
    assert any('zone' in name for name in names)


def test_the_toggle_survives_standing_down(window, pump):
    """Going to 2D must not hide the button that returns to 3D —
    it did, because the plain-curves pass recomputed the offer."""
    window.add_object('Specification', _specification())
    window._item_for_object('Specification').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane.waterfall_action.isVisible()
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert not pane._waterfall_page.isVisible(), 'flat, as asked'
    assert pane.waterfall_action.isVisible(), \
        'and the way back to the stage is still on the bar'
    # the comparison path had the same hole
    window.add_object('PSD', _measured())
    window.tree.clearSelection()
    window._item_for_object('Specification').setSelected(True)
    window._item_for_object('PSD').setSelected(True)
    pump()
    assert pane.waterfall_action.isVisible()


def test_a_stage_is_one_drawing_not_one_render_per_actor():
    """A specification with sixteen channels drew itself in 160
    renders — one per add_mesh, pyvista's default — and its bands
    arrived on screen one at a time (Brandon, 2026-09-01). A drawer
    now runs with rendering suppressed and hands the plotter back the
    way it found it; the caller renders once."""
    import pyvista as pv

    from visualdynamics.viz.banded import add_banded_stage
    from visualdynamics.viz.waterfall import add_waterfall

    plotter = pv.Plotter(off_screen=True)
    # shown once first, as the app's stage always is: before show(),
    # pyvista's render() is a no-op whatever the suppression says, and
    # the test could not tell a guarded drawer from an unguarded one
    plotter.show(auto_close=False)
    # count real renders, not calls to render(): the suppression lives
    # inside render(), and pyvista's own path reaches the window through
    # this hook only when it actually draws
    renders = []
    plotter.renderers.on_plotter_render = lambda *a, **k: renders.append(1)
    add_banded_stage(plotter, _specification())
    assert not renders, f'{len(renders)} renders inside one drawing'
    assert plotter.suppress_rendering is False, 'handed back as found'
    assert any(name.startswith('banded-') for name in plotter.actors), (
        'and the stage was actually drawn')

    renders.clear()
    add_waterfall(plotter, _specification())
    assert not renders, 'the waterfall drawer holds to the same rule'
    assert plotter.suppress_rendering is False


def test_a_banded_specification_stands_as_steps():
    """An octave-band specification is a density per bin and draws
    flat across each, the 2-D plot's stepMode and the waterfall's
    outline; the banded stage drew it as the line through its bin
    centers (Brandon, 2026-09-18). Two points per bin on the bin's
    edges now, for the target and every limit; the exceedances are
    still judged on the specification's own lines."""
    import numpy as np

    from visualdynamics.core.data import Specification
    from visualdynamics.plot import step_outline
    from visualdynamics.viz.banded import banded_stage_arrays

    freq = np.linspace(10.0, 2000.0, 200)
    level = np.full((1, len(freq)), 1e-2)
    spec = Specification(abscissa=freq, ordinate=level, response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         ordinate_unit=['m/s**2'],
                         warning_upper=level * 2, abort_upper=level * 4)
    plain = banded_stage_arrays(spec)
    assert plain['stations'][0]['target'].shape == (200,)
    assert np.array_equal(plain['spec_x_drawn'], plain['spec_x'])
    banded = spec.to_octave(3)
    arrays = banded_stage_arrays(banded)
    bins = len(banded.abscissa)
    station = arrays['stations'][0]
    assert station['target'].shape == (2 * bins,)
    assert station['limits']['abort_upper'].shape == (2 * bins,)
    edges, _rows = step_outline(banded.abscissa, banded.ordinate.real,
                                banded.bin_widths())
    assert np.array_equal(arrays['spec_x_drawn'], edges)
    assert np.array_equal(arrays['spec_x'], banded.abscissa), \
        'the lines themselves, for judging'
    assert np.array_equal(station['target'][0::2], station['target'][1::2]), \
        'flat across each bin'


def test_a_specification_on_lines_stands_as_steps_too():
    """The controller's target on its lines is a density per line (its
    reading is 'bin' with no band widths) and steps on its stage by
    the one drawing rule, not only when it carries band widths
    (Brandon, 2026-09-19)."""
    import numpy as np

    from visualdynamics.core.data import Specification
    from visualdynamics.viz.banded import banded_stage_arrays

    freq = np.arange(10.0, 210.0)
    level = np.full((1, len(freq)), 1e-2)
    spec = Specification(freq, level, response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         ordinate_unit=['m/s**2'], abort_upper=level * 4)
    spec.interpolation = Specification.reading_of(freq)
    assert spec.interpolation == 'bin' and spec.bandwidth is None
    arrays = banded_stage_arrays(spec)
    assert arrays['stations'][0]['target'].shape == (2 * len(freq),)
    assert arrays['stations'][0]['limits']['abort_upper'].shape == (2 * len(freq),)
