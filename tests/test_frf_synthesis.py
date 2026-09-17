"""FRFs re-synthesized from a modal model, overlaid on the measurement.

Selecting a shape set (or picked modes in its grid) beside measured FRFs
asks the modal model to predict them: the residue form
H_jk = sum_r phi_jr phi_kr / (m_r (w_r^2 - w^2 + 2i z_r w_r w)),
scaled to each measurement's quantity by powers of iw. The survey
fixtures were generated from exactly this modal model, so the full-mode
synthesis must reproduce the measured FRFs to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path, overlay_selection

import visualdynamics

pytestmark = pytest.mark.usefixtures('flat_reading')

def test_full_synthesis_reproduces_the_fixture_frfs(survey):
    shapes, frfs = survey
    receptance = shapes.synthesize_frf(
        frfs.abscissa, frfs.response_dof[:5], frfs.reference_dof[:5])
    accelerance = -(2.0 * np.pi * frfs.abscissa) ** 2 * receptance
    for row in range(5):
        measured = frfs.ordinate[row]
        scale = np.abs(measured).max()
        assert np.abs(accelerance[row] - measured).max() < 1e-9 * scale


def test_a_signed_dof_flips_the_shape(survey):
    shapes, frfs = survey
    plus = shapes.synthesize_frf(frfs.abscissa, ['101X+'], ['101Z+'])
    minus = shapes.synthesize_frf(frfs.abscissa, ['101X-'], ['101Z+'])
    assert np.allclose(minus, -plus)


def test_a_mode_subset_synthesizes_only_those_modes(survey):
    shapes, frfs = survey
    # an out-of-plane pair: the plate's in-plane DOFs answer at the
    # level of round-off, where any truncation is invisible
    full = shapes.synthesize_frf(frfs.abscissa, ['107Z+'], ['101Z+'])
    truncated = shapes.synthesize_frf(frfs.abscissa, ['107Z+'], ['101Z+'],
                                      modes=range(10))
    assert not np.allclose(truncated, full), 'truncation must show'
    rest = shapes.synthesize_frf(frfs.abscissa, ['107Z+'], ['101Z+'],
                                 modes=range(10, shapes.num_shapes))
    assert np.allclose(truncated + rest, full), 'modes sum independently'


def test_rigid_body_modes_at_zero_hertz_raise_no_warnings(survey):
    """The user's data starts at 0 Hz and the model has rigid-body modes:
    that bin was 0/0, and every render printed a screenful of
    RuntimeWarnings. The power applies inside the sum now, where the
    rigid accelerance at DC cancels exactly to the residue."""
    import warnings

    shapes, frfs = survey
    freq = np.concatenate([[0.0], frfs.abscissa])
    assert (shapes.frequency == 0.0).any(), 'the fixture has rigid modes'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        accelerance = shapes.synthesize_frf(freq, ['101X+'], ['101Z+'],
                                            power=2)
        receptance = shapes.synthesize_frf(freq, ['101X+'], ['101Z+'])
    rigid = shapes.frequency == 0.0
    lookup = {c: i for i, c in enumerate(shapes.coordinate)}
    expected_dc = np.sum(
        shapes.shape_matrix[rigid, lookup['101X+']]
        * shapes.shape_matrix[rigid, lookup['101Z+']]
        / shapes.modal_mass[rigid])
    assert np.isfinite(accelerance[0, 0])
    assert accelerance[0, 0] == pytest.approx(expected_dc)
    assert not np.isfinite(receptance[0, 0]), (
        'a free body has no static deflection to report')


def test_the_power_matches_the_external_multiply_off_dc(survey):
    shapes, frfs = survey
    receptance = shapes.synthesize_frf(frfs.abscissa, ['101X+'], ['101Z+'])
    accelerance = shapes.synthesize_frf(frfs.abscissa, ['101X+'], ['101Z+'],
                                        power=2)
    outside = -(2.0 * np.pi * frfs.abscissa) ** 2 * receptance
    assert np.allclose(accelerance, outside)


def test_a_dof_outside_the_shapes_is_refused(survey):
    shapes, frfs = survey
    assert not shapes.covers('99999X+')
    with pytest.raises(ValueError, match='99999X'):
        shapes.synthesize_frf(frfs.abscissa, ['99999X+'], ['101Z+'])


# ---- the GUI overlay --------------------------------------------------------

@pytest.fixture
def shown(window, pump, survey):
    shapes, frfs = survey
    window.add_object('Shapes', shapes)
    window.add_object('FRF', frfs)
    return window, shapes, frfs


def test_selecting_shapes_beside_frfs_overlays_the_synthesis(shown, pump):
    window, _shapes, _frfs = shown
    overlay_selection(window, pump)
    assert 'synthesized from 22 modes' in window.statusBar().currentMessage()


def test_selecting_both_whole_objects_opens_the_fit_screen(shown, pump):
    """The user's ask, twice over: FRF plus shape set is a fit session,
    seeded from the set and publishing back to it."""
    window, shapes, _frfs = shown
    window.tree.setCurrentItem(window._item_for_object('FRF'))
    window.tree.clearSelection()
    for name in ('FRF', 'Shapes'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.fit is not None
    assert len(window.fit.modes) == shapes.num_shapes
    assert window.fit_object_name == 'Shapes'
    assert window.fit_bar.isVisible()


def test_the_overlay_is_dashed_in_its_pairs_color(shown, pump):
    from PySide6.QtCore import Qt

    window, shapes, _frfs = shown
    window.tree.clearSelection()
    shapes_item = window._item_for_object('Shapes')
    shapes_item.setExpanded(True)
    pump()
    window.record_grids['Shapes'].select_records(
        list(range(shapes.num_shapes)))
    shapes_item.setSelected(True)
    # one record, so the pairing is unambiguous to read back
    item = window._item_for_object('FRF')
    item.setExpanded(True)
    item.setSelected(True)
    window.record_grids['FRF'].select_records([0])
    window.render_current()
    pump()
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    curves = [c for plot in plots for c in plot.listDataItems()]
    assert len(curves) == 2, 'the measurement and its prediction'
    pens = [c.opts['pen'] for c in curves]
    styles = {pen.style() for pen in pens}
    assert styles == {Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine}
    assert pens[0].color().name() == pens[1].color().name(), 'same pair, same color'


def test_picking_modes_keeps_the_frfs_selected(shown, pump):
    """Mode picks predict beside data — they must not evict it.

    The eviction branch only runs when the grid's owner is not yet
    selected, so that is the setup: FRF selected, Shapes not.
    """
    window, _shapes, _frfs = shown
    window.tree.clearSelection()
    frf_item = window._item_for_object('FRF')
    frf_item.setSelected(True)
    window.tree.setCurrentItem(frf_item)
    item = window._item_for_object('Shapes')
    item.setExpanded(True)
    pump()
    assert not item.isSelected(), 'the setup the eviction branch needs'
    grid = window.record_grids['Shapes']
    grid.item(0, 0).setSelected(True)      # a real click, signals live
    pump()
    assert frf_item.isSelected(), 'FRF survived the mode pick'


def test_picked_modes_truncate_the_synthesis(shown, pump):
    window, _shapes, _frfs = shown
    overlay_selection(window, pump, modes=[0, 1, 2])
    assert 'synthesized from 3 modes' in window.statusBar().currentMessage()


def test_the_overlay_carries_mode_markers_in_the_theme_ink(shown, pump):
    """The fitting screen's bookmarks follow the synthesis to the FRF
    overlay — one dashed line per synthesized mode, in the theme's
    foreground (white on dark, black on light) so they read apart from
    the gray grid."""
    import pyqtgraph as pg

    from visualdynamics.plot import resolve_theme

    window, shapes, _frfs = shown
    # three distinct frequencies: two rigid modes both mark 0.0 Hz and
    # coincident markers draw as one line
    overlay_selection(window, pump, modes=[0, 6, 8])
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    markers = [item for plot in plots for item in plot.items
               if isinstance(item, pg.InfiniteLine)]
    assert [m.value() for m in markers] == [
        float(shapes.frequency[m]) for m in (0, 6, 8)]
    ink = pg.mkColor(resolve_theme(window.theme_name)['plot_foreground'])
    color = markers[0].pen.color()
    assert (color.red(), color.green(), color.blue()) == (
        ink.red(), ink.green(), ink.blue())


# ---- specifications follow their measurement's color ------------------------

def test_a_specification_and_its_response_are_a_pair(window, pump):
    """One comparison is drawn at a time, so there is nothing to tell
    apart by color: the response takes the foreground because it is
    what is being looked at, the specification gray behind it because it
    is the reference. They used to share a palette color, which is what
    a plot of six comparisons at once needed and this is not."""
    from PySide6.QtCore import Qt

    from visualdynamics.theme import theme as resolve_theme

    out = visualdynamics.import_file(fixture_path('plate', 'random_spectra.nc4'))
    window.add_object('CPSD', out['Random_response_cpsd'])
    window.add_object('Specification', out['Random_specification'])
    window.tree.clearSelection()
    for i in range(window.test_item.childCount()):
        window.test_item.child(i).setSelected(True)
    window.render_current()
    pump()
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    curves = [c for plot in plots for c in plot.listDataItems()
              if c.opts.get('name')]
    assert len(curves) == 2, 'one comparison, two curves'
    colors = {c.opts['pen'].color().name() for c in curves}
    palette = resolve_theme(window.theme_name)
    assert colors == {palette['response_curve'],
                      palette['specification_curve']}
    assert {c.opts['pen'].style() for c in curves} == {Qt.PenStyle.SolidLine}


def test_the_pair_toggle_switches_between_edit_and_resynthesis(shown, pump):
    """One selection, two readings: Edit Fit is the fitting screen,
    Resynthesis the overlay — a sticky choice on the plot bar."""
    window, shapes, _frfs = shown
    window.tree.setCurrentItem(window._item_for_object('FRF'))
    window.tree.clearSelection()
    for name in ('FRF', 'Shapes'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.fit is not None, 'Edit Fit is the default'
    assert window.data_pane.pair_edit_action.isVisible()
    assert window.data_pane.pair_edit_action.isChecked()

    window.data_pane.pair_synthesis_action.trigger()
    pump()
    assert window.fit is None, 'resynthesis is the overlay, not the fit'
    assert 'synthesized from 22 modes' in window.statusBar().currentMessage()
    assert window.data_pane.pair_synthesis_action.isChecked()

    window.data_pane.pair_edit_action.trigger()
    pump()
    assert window.fit is not None, 'and back'
    assert len(window.fit.modes) == shapes.num_shapes, 'seeded again'


def test_the_toggle_hides_without_the_pair(shown, pump):
    window, _shapes, _frfs = shown
    window.tree.clearSelection()
    item = window._item_for_object('FRF')
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not window.data_pane.pair_edit_action.isVisible()
    assert not window.data_pane.pair_synthesis_action.isVisible()


def test_mode_picks_keep_the_overlay_but_offer_edit(shown, pump):
    """Browsing a truncated synthesis must not snap into the fit — but
    the Edit button stays one click away, and Edit means Edit."""
    window, shapes, _frfs = shown
    overlay_selection(window, pump, modes=[6, 7, 8])
    assert window.fit is None, 'picks mean the overlay'
    assert 'synthesized from 3 modes' in window.statusBar().currentMessage()
    assert window.data_pane.pair_edit_action.isVisible(), 'Edit offered anyway'
    assert window.data_pane.pair_synthesis_action.isChecked()
    window.data_pane.pair_edit_action.trigger()
    pump()
    assert window.fit is not None
    assert len(window.fit.modes) == shapes.num_shapes, (
        'Edit opens the whole set')
    assert window.fit_object_name == 'Shapes'
