"""Two densities on the stage: overlaid per shared channel, or divided.

The 3-D reading of two selected PSDs (Brandon, 2026-08-25): matched
stations recede along the depth axis, the louder wears the level
coloring, the quieter the compared gray — and the Ratio reading
divides through `density_ratio`, in decibels, so the stage and the
flat plot cannot disagree."""

from __future__ import annotations

import numpy as np
from conftest import overlay_selection as _overlay_selection

from visualdynamics.core.data import Psd

FREQ = np.linspace(2.0, 1000.0, 400)


def _pair(level=1e-3, floor=1e-6, channels=('101Z+', '104Z+', '110Z+')):
    def density(value):
        return Psd(abscissa=FREQ,
                   ordinate=np.full((len(channels), len(FREQ)), value),
                   response_dof=list(channels),
                   ordinate_dim=['acceleration**2/frequency']
                   * len(channels))
    return density(level), density(floor)


def test_stations_pair_by_shared_label():
    from visualdynamics.viz.paired import paired_arrays

    loud, quiet = _pair()
    arrays = paired_arrays(loud, quiet)
    assert [s['label'] for s in arrays['stations']] == \
        ['101Z+', '104Z+', '110Z+']
    assert arrays['unmatched'] == 0
    lonely, _ = _pair(channels=('101Z+', '999X+'))
    arrays = paired_arrays(lonely, quiet)
    assert [s['label'] for s in arrays['stations']] == ['101Z+']
    assert arrays['unmatched'] == 3, 'each side counted its own strays'


def test_the_overlay_stage_draws_both_sides():
    import pyvista as pv

    from visualdynamics.viz.paired import add_paired_stage

    loud, quiet = _pair()
    plotter = pv.Plotter(off_screen=True)
    info = add_paired_stage(plotter, loud, quiet)
    assert info['drawn'] == 3
    assert 'paired-loud' in plotter.actors
    assert 'paired-quiet' in plotter.actors
    assert plotter.actors['paired-quiet'].mapper.dataset.n_points > 0
    plotter.close()


def test_the_ratio_stage_divides_in_decibels():
    import pyvista as pv

    from visualdynamics.viz.paired import add_paired_stage

    loud, quiet = _pair(level=1e-3, floor=1e-6)
    plotter = pv.Plotter(off_screen=True)
    info = add_paired_stage(plotter, loud, quiet, mode='ratio')
    assert info['drawn'] == 3
    mesh = plotter.actors['paired-ratio'].mapper.dataset
    assert np.allclose(np.asarray(mesh.point_data['level']), 30.0), \
        'a thousandfold power ratio reads 30 dB, per point'
    plotter.close()


def test_two_selected_densities_default_to_the_stage(window, pump):
    loud, quiet = _pair()
    window.add_object('Excitation PSDs', loud)
    window.add_object('Noise PSDs', quiet)
    window._item_for_object('Excitation PSDs').setSelected(True)
    window._item_for_object('Noise PSDs').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the pair comes up in 3-D'
    assert 'paired-loud' in pane.waterfall_plotter.actors
    assert pane.spectra_actions['overlay'].isVisible()
    # the Ratio reading swaps the stage's drawing in place
    pane.spectra_actions['ratio'].trigger()
    pump()
    assert pane._waterfall_page.isVisible(), 'still the stage'
    assert 'paired-ratio' in pane.waterfall_plotter.actors
    # and the toggle stands down to the flat ratio
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert not pane._waterfall_page.isVisible()
    import pyqtgraph as pg
    plot = next(item for item in pane.graphics.ci.items
                if isinstance(item, pg.PlotItem))
    assert plot.listDataItems(), 'the flat dB curves took over'
    assert pane.waterfall_action.isVisible(), 'the way back stays'


def test_a_pair_with_nothing_shared_still_reads_on_the_stage(window, pump):
    """Nothing pairs station by station, but the picks still stack
    onto one stage (Brandon, 2026-08-30) — this used to fall back to
    the flat plot, which remains a toggle away."""
    loud, _ = _pair(channels=('101Z+',))
    _, quiet = _pair(channels=('999X+',))
    window.add_object('A', loud)
    window.add_object('B', quiet)
    window._item_for_object('A').setSelected(True)
    window._item_for_object('B').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the stacked stage stands'
    assert len(window._waterfall_drawn) == 2, 'both lone channels drew'
    pane.waterfall_action.trigger()
    pump()
    window.render_current()
    pump()
    import pyqtgraph as pg
    plots = [item for item in pane.graphics.ci.items
             if isinstance(item, pg.PlotItem)]
    assert plots and any(p.listDataItems() for p in plots), \
        'and the flat overlay is a toggle away — never a blank pane'


def _mixed_pair():
    """Eight accelerations and four forces on each side — the sysid
    stream's own shape."""
    channels = [f'{n}Z+' for n in (101, 104, 110, 113)] + \
               [f'{n}Z+' for n in (101, 110)]
    dims = ['acceleration**2/frequency'] * 4 + ['force**2/frequency'] * 2

    def density(value):
        return Psd(abscissa=FREQ,
                   ordinate=np.full((6, len(FREQ)), value),
                   response_dof=channels, ordinate_dim=dims)
    return density(1e-3), density(1e-6)


def test_a_mixed_pair_offers_the_quantity_box(window, pump):
    """One floor, one quantity — the paired stage too. The first cut
    drew only the largest group in overlay while the ratio divided
    everything: a silent cap and a disagreement in one."""
    loud, quiet = _mixed_pair()
    window.add_object('Excitation PSDs', loud)
    window.add_object('Noise PSDs', quiet)
    window._item_for_object('Excitation PSDs').setSelected(True)
    window._item_for_object('Noise PSDs').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane._waterfall_page.isVisible()
    assert pane.quantity_action.isVisible(), \
        'two quantities on stage is a choice, and the box offers it'
    # the default floor is the larger group: the four accelerations
    assert pane.waterfall_plotter.actors[
        'paired-loud'].mapper.dataset.n_points > 0
    info_labels = [pane.quantity_box.itemText(k)
                   for k in range(pane.quantity_box.count())]
    assert any('acceleration' in text for text in info_labels)
    assert any('force' in text for text in info_labels)
    # the ratio honors the same choice rather than dividing everything
    pane.spectra_actions['ratio'].trigger()
    pump()
    mesh = pane.waterfall_plotter.actors['paired-ratio'].mapper.dataset
    stations = np.unique(np.round(np.asarray(mesh.points)[:, 1], 6))
    assert len(stations) == 4, \
        'the chosen quantity, all four channels of it — never a mix'


# ---- the resynthesis overlay reads on the same stage ---------------------


def test_the_resynthesis_overlay_comes_up_on_the_stage(window, pump, survey):
    """A measured set and its resynthesis are two objects holding the
    same channels, which is what the paired stage is for — so the
    overlay reads in 3-D like every other pair, station by station,
    rather than as ninety-six curves interfering on one axis (Brandon,
    2026-08-26)."""
    shapes, frfs = survey
    window.add_object('Shapes', shapes)
    window.add_object('FRF', frfs)
    _overlay_selection(window, pump)

    pane = window.data_pane
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the overlay comes up in 3-D'
    actors = pane.waterfall_plotter.actors
    assert 'paired-loud' in actors, 'the measurement, level-colored'
    assert 'paired-quiet' in actors, 'the synthesis, stood back'
    assert 'on the stage' in window.statusBar().currentMessage()


def _levels(pane, name):
    """The levels actually drawn into one of the stage's two meshes."""
    return np.asarray(
        pane.waterfall_plotter.actors[name].mapper.dataset
        .point_data['level'])


def test_the_stage_stands_the_synthesis_back_not_the_measurement(
        window, pump, survey):
    """Which of the two wears the level coloring is the whole reading,
    and it must match the flat plot: measurement solid, synthesis drawn
    over it. Truncated to one mode on purpose — the fixtures were born
    from this very model, so a full-mode synthesis *is* the measurement
    to machine precision, and a test using one cannot tell the two
    apart whichever way round they are drawn (it did not, first try).
    """
    shapes, frfs = survey
    window.add_object('Shapes', shapes)
    window.add_object('FRF', frfs)
    _overlay_selection(window, pump, modes=[0])

    pane = window.data_pane
    measured, synthesized = _levels(pane, 'paired-loud'), \
        _levels(pane, 'paired-quiet')
    assert measured.max() > synthesized.max() + 1.0, (
        'one mode of twenty-two cannot reach the measurement\'s peaks, '
        f'so the level-colored mesh is the measurement: '
        f'{measured.max():.3f} against {synthesized.max():.3f}')
    assert measured.mean() > synthesized.mean() + 1.0, \
        'and it stands above it almost everywhere, not just at a peak'


def test_the_resynthesis_stage_stands_down_to_the_flat_overlay(
        window, pump, survey):
    """The toggle is a toggle: 2-D brings back the dashed flat overlay,
    and the way back stays on the bar."""
    shapes, frfs = survey
    window.add_object('Shapes', shapes)
    window.add_object('FRF', frfs)
    _overlay_selection(window, pump)
    pane = window.data_pane
    assert pane.waterfall_action.isVisible(), 'the toggle is offered'

    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert not pane._waterfall_page.isVisible(), 'the flat plot took over'
    import pyqtgraph as pg
    plot = next(item for item in pane.graphics.ci.items
                if isinstance(item, pg.PlotItem))
    assert plot.listDataItems(), 'and it drew something'
    assert 'synthesized from 22 modes' in \
        window.statusBar().currentMessage(), 'the flat reading is unchanged'
    assert pane.waterfall_action.isVisible(), 'the way back stays'
