"""The matrix comparison drawn as a grid, and read in the window beside a
specification and a cross-spectral matrix selected together."""

from __future__ import annotations

import numpy as np
from test_matrix_comparison import SPEC, Specification, matrix, measured, term

# ---- the view ------------------------------------------------------------------


def test_the_view_draws_headless(tmp_path):
    from visualdynamics.plot import plot_matrix_comparison

    path = tmp_path / 'matrix.png'
    plot_matrix_comparison(SPEC, measured(coherence=0.49), scale_db=0,
                           path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


def test_the_grid_is_the_matrix_each_cell_with_its_verdict(qt_app):
    """Levels on the diagonal, the coherence above and the phase below;
    the cell that is out says OUT and wears a heavy solid frame, the
    lines out are marked, and the summary strips and table follow."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.matrix_comparison import build_matrix_comparison

    layout = pg.GraphicsLayoutWidget()
    try:
        result = build_matrix_comparison(layout, SPEC,
                                         measured(coherence=0.49),
                                         scale_db=0)
        cells = {(r, c): layout.getItem(r, c) for r in range(2)
                 for c in range(2)}
        assert all(plot.matrix_cell == key for key, plot in cells.items())
        titles = {key: plot.titleLabel.text for key, plot in cells.items()}
        assert 'level in +0.0 dB' in titles[(0, 0)]
        assert 'coherence OUT 100 % of lines' in titles[(0, 1)]
        assert 'phase in 0 % of lines' in titles[(1, 0)]
        assert '2X+/1X+' in titles[(1, 0)]
        assert cells[(0, 1)].cell_state == 'out'
        frame = cells[(0, 1)].getViewBox().border
        assert frame.width() == 3.0 and frame.style() == Qt.PenStyle.SolidLine
        assert cells[(1, 0)].getViewBox().border.width() == 1.0
        marks = [item for item in cells[(0, 1)].listDataItems()
                 if getattr(item, 'is_out_mark', False)]
        assert len(marks) == 1 and marks[0].opts['symbol'] == 'x'
        assert len(marks[0].xData) == 40
        assert not [item for item in cells[(1, 0)].listDataItems()
                    if getattr(item, 'is_out_mark', False)]
        assert layout.getItem(2, 0).is_level_summary
        assert layout.getItem(2, 1).is_cross_summary
        table = layout.getItem(0, 2)
        assert table.is_matrix_table
        assert result.describe() in table.text
        assert '100.0 % out' in table.text
    finally:
        layout.close()


def test_a_pair_not_specified_is_framed_dotted_and_says_so(qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.matrix_comparison import build_matrix_comparison

    spec = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)},
                  Specification)
    run = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0),
                                   (0, 2): term(0.5, 0.0, 1.0, 2.0),
                                   (1, 2): term(0.5, 0.0, 4.0, 2.0)})
    layout = pg.GraphicsLayoutWidget()
    try:
        build_matrix_comparison(layout, spec, run, scale_db=0, table=False)
        plot = layout.getItem(0, 2)
        assert 'coherence not specified' in plot.titleLabel.text
        assert plot.getViewBox().border.style() == Qt.PenStyle.DotLine
        assert plot.listDataItems() == []
        assert layout.getItem(0, 3) is None, 'no table when asked for none'
    finally:
        layout.close()


# ---- the reading in the window ----------------------------------------------


def _matrix(window, pump, run=None, spec=SPEC):
    """A specification and a CPSD in the window, selected, on the Matrix
    reading."""
    from conftest import select_objects

    window.add_object('Spec', spec)
    window.add_object('Run', measured() if run is None else run)
    select_objects(window, pump, 'Spec', 'Run')
    pane = window.data_pane
    assert pane.matrix_action.isVisible()
    pane.matrix_action.trigger()
    pump()
    return pane.matrix_panel


def test_the_matrix_is_a_reading_of_a_specification_and_a_cpsd(window, pump):
    panel = _matrix(window, pump, measured(phase=45.0))
    assert panel.isVisible()
    assert window.data_pane.matrix_action.text() == 'Matrix'
    assert panel.derived['verdict'].text() == 'fails'
    lines = panel.readout.text().splitlines()
    assert lines[2:] == [
        '1X+/2X+ coherence in: 0 of 40 lines past ±0.10',
        '1X+/2X+ phase out: 100.0 % of 40 lines past ±10°']
    assert window._status_text == (
        'Run against Spec: fails: levels pass on 2 channels; 1 of 1 cross '
        'term out (coherence 0, phase 1)')
    plot = window.data_pane.graphics.getItem(1, 0)
    assert plot.matrix_cell == (1, 0) and plot.cell_state == 'out'


def test_the_panels_tolerances_redraw_the_grid(window, pump):
    panel = _matrix(window, pump, measured(phase=45.0))
    panel.phase_box.setValue(20.0)
    pump()
    assert panel.derived['verdict'].text() == 'passes'
    assert window.data_pane.graphics.getItem(1, 0).cell_state == 'in'
    panel.set_settings(coherence_floor=0.7)
    pump()
    assert panel.readout.text().splitlines()[3] == (
        '1X+/2X+ phase not judged: specified coherence under 0.70')


def test_the_matrix_reading_is_journaled_as_its_plot_call(window, pump):
    panel = _matrix(window, pump)
    panel.set_settings(coherence_tolerance=0.15)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_matrix_comparison(')]
    assert lines == [(
        "visualdynamics.plot.plot_matrix_comparison(project['Spec'], "
        "project['Run'], coherence_tolerance=0.15, phase_tolerance=10.0, "
        "coherence_floor=0.5, scale_db=0, path='matrix.png', show=False)")]


def test_a_pair_that_cannot_be_judged_as_a_matrix_says_why(window, pump):
    run = matrix([1.0, 4.0], {(0, 1): term(0.5, 0.0, 1.0, 4.0)})
    run.response_dof = ['1X+', '9X+', '1X+', '9X+']
    run.reference_dof = ['1X+', '1X+', '9X+', '9X+']
    panel = _matrix(window, pump, run)
    assert panel.derived['verdict'].text() == '—'
    assert window._status_text.startswith(
        'Matrix: a matrix comparison needs at least two control channels')


def test_the_matrix_is_not_offered_for_other_pairs(window, pump):
    """A specification beside a set of autospectra, or two CPSDs, is not
    a matrix against its specification."""
    from conftest import select_objects

    from visualdynamics.core.data import Psd

    window.add_object('Spec', SPEC)
    window.add_object('Autos', Psd(np.arange(20.0, 420.0, 10.0),
                                   np.ones((2, 40)), response_dof=['1X+', '2X+'],
                                   ordinate_dim=['acceleration**2/frequency']
                                   * 2))
    window.add_object('Run', measured())
    window.add_object('Run 2', measured())
    pane = window.data_pane
    for pair in (('Spec', 'Autos'), ('Run', 'Run 2')):
        select_objects(window, pump, *pair)
        assert not pane.matrix_action.isVisible()
        assert not pane.matrix_panel.isVisibleTo(pane)


def test_unchecking_the_matrix_goes_back_to_the_channels(window, pump):
    panel = _matrix(window, pump)
    window.data_pane.matrix_action.trigger()
    pump()
    assert not panel.isVisibleTo(window.data_pane)
    # the channels' comparison, on its own default reading
    assert window._status_text == 'Run against Spec: 2 channels on the stage'


def test_a_tolerance_moved_judges_again_without_reading_again(window, pump,
                                                              monkeypatch):
    """The levels and readings do not depend on the tolerances, so moving
    one does not compare the matrix again; another pair selected does."""
    from conftest import select_objects

    from visualdynamics.core import matrix_comparison as module

    panel = _matrix(window, pump, measured(phase=45.0))
    reads = []
    real = module.compare_matrix
    monkeypatch.setattr(module, 'compare_matrix',
                        lambda *a, **k: reads.append(1) or real(*a, **k))
    panel.phase_box.setValue(20.0)
    pump()
    assert reads == []
    assert panel.derived['verdict'].text() == 'passes'
    window.add_object('Run 2', measured(phase=60.0))
    select_objects(window, pump, 'Spec', 'Run 2')
    assert reads == [1]
    assert panel.derived['verdict'].text() == 'fails'
