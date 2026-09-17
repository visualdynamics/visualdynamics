"""The Complex Mode Indicator Function: the FRF matrix as singular values.

At every frequency line the records assemble into the response x reference
matrix they are and get an SVD; the singular values plot over frequency.
The largest peaks at every mode, and the second peaking too is how a
repeated root shows itself. With a modal model selected beside the FRFs,
the synthesis's CMIF draws dashed over the measurement's, color for color.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path
from PySide6.QtCore import Qt

import visualdynamics
from visualdynamics.plot import cmif_curves


def test_the_cmif_is_the_singular_values_of_the_matrix(survey):
    """Checked against a hand-assembled matrix and numpy's own SVD — not
    through cmif_curves' plumbing, so the complex matrix reaching the SVD
    is part of what is checked (an |H| matrix is a plausible wrong CMIF
    that agrees with itself)."""
    _shapes, frfs = survey
    singular, _x = cmif_curves(frfs)
    references = sorted(set(frfs.reference_dof))
    assert singular.shape == (len(references), len(frfs.abscissa))
    assert (np.diff(singular, axis=0) <= 1e-12).all(), 'sorted descending'
    responses = sorted(set(frfs.response_dof))
    lines = [10, 100, 200]
    for line in lines:
        matrix = np.zeros((len(responses), len(references)),
                          dtype=np.complex128)
        for i in range(frfs.num_records):
            matrix[responses.index(frfs.response_dof[i]),
                   references.index(frfs.reference_dof[i])] = (
                frfs.ordinate[i, line])
        expected = np.linalg.svd(matrix, compute_uv=False)
        assert np.allclose(sorted(singular[:, line], reverse=True),
                           expected)


def test_a_truncated_synthesis_drops_its_zero_singular_values(survey):
    """Three modes against four references is rank three; the fourth
    singular value is numerical dust and must not be drawn twenty decades
    down."""
    import pyqtgraph as pg

    from visualdynamics.core.data import Frf
    from visualdynamics.plot import build_cmif

    shapes, frfs = survey
    synthesized = shapes.synthesize_frf(
        frfs.abscissa, frfs.response_dof, frfs.reference_dof,
        modes=range(6, 9), power=2)
    truncated = Frf(frfs.abscissa, synthesized,
                    response_dof=frfs.response_dof,
                    reference_dof=frfs.reference_dof)
    layout = pg.GraphicsLayoutWidget()
    drawn = build_cmif(layout, [('T', truncated, None)])
    assert drawn == 3, 'rank three, three curves'


# ---- the toggle on the plot bar ---------------------------------------------

@pytest.fixture
def shown(window, pump, survey):
    shapes, frfs = survey
    window.add_object('Shapes', shapes)
    window.add_object('FRF', frfs)
    return window, shapes, frfs


def frf_only(window, pump):
    # these tests read the flat plot's curves and axes; the waterfall
    # is the default reading now, so the flat plot is asked for
    window.data_pane.waterfall_action.setChecked(False)
    window.tree.clearSelection()
    item = window._item_for_object('FRF')
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()


def plotted_curves(window):
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    return [c for plot in plots for c in plot.listDataItems()]


def test_the_toggle_shows_for_frfs_and_swaps_the_plot(shown, pump):
    window, _shapes, _frfs = shown
    frf_only(window, pump)
    assert window.data_pane.cmif_action.isVisible()
    window.data_pane.cmif_action.trigger()
    pump()
    assert 'CMIF: 4 singular value curves' in window.statusBar().currentMessage()
    assert len(plotted_curves(window)) == 4
    window.data_pane.cmif_action.trigger()
    pump()
    assert 'CMIF' not in window.statusBar().currentMessage()


def test_a_modal_model_overlays_its_cmif_dashed(shown, pump):
    window, shapes, _frfs = shown
    window.pair_mode = 'overlay'      # the toggle's resynthesis side
    window.tree.clearSelection()
    for i in range(window.test_item.childCount()):
        window.test_item.child(i).setSelected(True)
    window.data_pane.cmif_action.setChecked(True)
    window.render_current()
    pump()
    assert 'CMIF: 8 singular value curves' in window.statusBar().currentMessage()
    curves = plotted_curves(window)
    solid = [c for c in curves
             if c.opts['pen'].style() == Qt.PenStyle.SolidLine]
    dashed = [c for c in curves
              if c.opts['pen'].style() == Qt.PenStyle.DashLine]
    assert len(solid) == 4 and len(dashed) == 4
    assert ([p.opts['pen'].color().name() for p in solid]
            == [p.opts['pen'].color().name() for p in dashed]), (
        'singular value for singular value, same color')
    # the fitting screen's gray bookmarks ride along with the synthesis
    import pyqtgraph as pg
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    markers = [item for plot in plots for item in plot.items
               if isinstance(item, pg.InfiniteLine)]
    # coincident frequencies collapse to one line — identical marks
    # overprinted would just darken each other
    unique = list(dict.fromkeys(float(f) for f in shapes.frequency))
    assert [m.value() for m in markers] == unique
    assert len(markers) < shapes.num_shapes, (
        'the fixture has coincident modes, so the dedupe must bite')


def test_the_component_dropdown_reads_the_frf_four_ways(shown, pump):
    """Magnitude stays the log default; Real, Imaginary and Phase plot
    signed values on a linear axis, exactly what the drop-down says."""
    import numpy as np

    window, _shapes, frfs = shown
    frf_only(window, pump)
    assert window.data_pane.component_action.isVisible()
    values = frfs.display_ordinate(window.unit_system, [0])[0]
    window.data_pane.component_box.setCurrentIndex(2)          # Imaginary
    pump()
    assert np.allclose(plotted_curves(window)[0].yData, values.imag)
    window.data_pane.component_box.setCurrentIndex(3)          # Phase
    pump()
    assert np.allclose(plotted_curves(window)[0].yData,
                       np.degrees(np.angle(values)))
    window.data_pane.component_box.setCurrentIndex(0)          # back to Magnitude
    pump()
    assert np.allclose(plotted_curves(window)[0].yData, np.abs(values))
    # the CMIF is singular values — no component to pick there
    window.data_pane.cmif_action.setChecked(True)
    window.render_current()
    pump()
    assert not window.data_pane.component_action.isVisible()
    window.data_pane.cmif_action.setChecked(False)


def test_zoom_cannot_leave_the_data(shown, pump):
    """The opening view is also the outer limit: however far the user
    zooms or pans, the axes never leave what the data covers."""
    window, _shapes, frfs = shown
    frf_only(window, pump)
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    box = plots[0].getViewBox()
    assert box.state['limits']['xLimits'][0] is not None, 'limits armed'
    box.setXRange(-1e9, 1e9, padding=0)
    box.setYRange(-1e9, 1e9, padding=0)
    pump()
    (x0, x1), (y0, y1) = box.viewRange()
    xs = frfs.display_abscissa(window.unit_system)
    span = float(xs.max() - xs.min())
    assert x0 >= float(xs.min()) - 0.03 * span
    assert x1 <= float(xs.max()) + 0.03 * span
    magnitudes = np.abs(frfs.display_ordinate(window.unit_system))
    tops = np.log10(magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)])
    assert y0 >= tops.min() - 1 and y1 <= tops.max() + 1
    # the CMIF plot locks the same way
    window.data_pane.cmif_action.setChecked(True)
    window.render_current()
    pump()
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    box = plots[0].getViewBox()
    box.setXRange(-1e9, 1e9, padding=0)
    pump()
    x0, x1 = box.viewRange()[0]
    assert x0 >= float(xs.min()) - 0.03 * span
    assert x1 <= float(xs.max()) + 0.03 * span
    window.data_pane.cmif_action.setChecked(False)


def test_the_toggle_hides_for_non_frf_data(window, pump):
    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    window.add_object('PSD', psd)
    item = window.test_item.child(0)
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not window.data_pane.cmif_action.isVisible()
