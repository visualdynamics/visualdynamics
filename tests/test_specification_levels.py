"""A specification's own levels: a toggle, bars over the table.

The table of RMS levels used to sit beneath the spectra by default.
Brandon (2026-09-06): take it out of the main plot and make it a
toggle on the bar — on, the upper pane is a bar per channel and the
lower pane the table, the comparison's RMS reading without the ±3 dB
coloring; off, the spectra alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core.compliance import specification_rms


@pytest.fixture
def showing(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object('Specification')
    pump()
    return window, window.objects['Specification']


def _curves(window):
    plot = next(item for item in window.data_pane.graphics.ci.items
                if hasattr(item, 'listDataItems'))
    return plot.listDataItems()


def test_the_spectra_alone_are_the_default(showing):
    window, _spec = showing
    assert window.data_pane.rms_action.isVisible()
    assert not window.data_pane.rms_action.isChecked()
    assert not window.table.isVisibleTo(window), 'no table beneath by default'
    assert _curves(window), 'the spectrum is drawn'
    assert window.bar_chart is None


def test_the_toggle_puts_the_bars_over_the_table(showing, pump):
    window, spec = showing
    window.data_pane.rms_action.trigger()
    pump()
    chart = window.bar_chart
    assert chart is not None and chart.low is None and chart.high is None, \
        'no thresholds: nothing to be out of'
    assert chart.lines == [], 'and no shaded zones'
    assert [label for label, _v in chart.rows] == list(spec.response_dof)
    # read in the display unit, as the heading names it
    shown = [float(window.unit_system.from_si(
        specification_rms(spec, i),
        spec.ordinate_dim[i].replace('**2/frequency', '')))
        for i in range(spec.num_records)]
    assert np.allclose(chart.values(), shown)
    assert all(chart.beyond(v) is None for v in chart.values())
    assert chart.plot.getAxis('bottom').labelText.startswith('RMS level')
    summary = chart.summary.toPlainText()
    assert summary.startswith('8 channels — ') and 'highest at' in summary
    assert window.statusBar().currentMessage().startswith(summary), \
        'the bars say it first; the table adds its count'
    assert window.table.isVisibleTo(window)
    model = window.table.model()
    assert model.rowCount() == spec.num_records
    assert float(model.index(0, 1).data()) == pytest.approx(
        shown[0], rel=1e-3)
    # and back
    window.data_pane.rms_action.trigger()
    pump()
    assert not window.table.isVisibleTo(window)
    assert _curves(window)


def test_picked_records_restrict_the_bars(showing, pump):
    window, spec = showing
    window.tree.clearSelection()
    grid = window.record_grids['Specification']
    grid.item(0, 0).setSelected(True)
    grid.item(2, 0).setSelected(True)
    pump()
    window.data_pane.rms_action.trigger()
    pump()
    assert [label for label, _v in window.bar_chart.rows] == [
        spec.response_dof[0], spec.response_dof[2]]


def test_a_comparison_keeps_its_own_readings(showing, pump):
    """A specification with a measurement beside it has the three
    comparison readings, not this toggle."""
    from PySide6.QtCore import QItemSelectionModel

    window, _spec = showing
    history = next(n for n, o in window.objects.items()
                   if type(o).__name__ == 'TimeHistory')
    window.tree.setCurrentItem(window._item_for_object(history))
    window.compute_psds()
    pump()
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd' and n != 'Specification')
    window.tree.clearSelection()
    for name in ('Specification', psd):
        item = window._item_for_object(name)
        item.setSelected(True)
        window.tree.setCurrentItem(
            item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
    window.render_current()
    pump()
    assert not window.data_pane.rms_action.isVisible()
    assert window.data_pane.comparison_actions['error'].isVisible()


def test_a_target_holding_its_cross_terms_reads_one_level_per_channel(window,
                                                                      pump):
    """A virtual point's target states every pair (2026-09-07), so its
    records are a full matrix; the levels reading listed nine rows for
    three channels, six of them NaN, and counted nine. A level is a
    channel's: the autos, and nothing for a cross term."""
    from visualdynamics.core.data import Specification

    f = np.linspace(20.0, 2000.0, 50)
    channels = ['M1', 'M2', 'M3']
    rows, resp, refs = [], [], []
    for a, da in enumerate(channels):
        for b, db in enumerate(channels):
            rows.append(np.full(len(f), 1e-2 * (a + 1) if a == b else 0.0,
                                dtype=complex))
            resp.append(da)
            refs.append(db)
    spec = Specification(abscissa=f, ordinate=np.asarray(rows),
                         response_dof=resp, reference_dof=refs,
                         ordinate_dim=['acceleration**2/frequency'] * 9)
    window.add_object('Target', spec)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object('Target')
    pump()
    window.data_pane.rms_action.trigger()
    pump()
    assert window.statusBar().currentMessage().startswith('3 channels — M3 highest')
    model = window.table.model()
    from PySide6.QtCore import Qt
    labels = [model.data(model.index(r, 0), Qt.ItemDataRole.DisplayRole)
              for r in range(model.rowCount())]
    assert labels == channels, 'the autos, and nothing for a cross term'
    levels = [model.data(model.index(r, 1), Qt.ItemDataRole.DisplayRole)
              for r in range(model.rowCount())]
    assert all('nan' not in str(v) for v in levels)


def test_the_levels_are_read_in_the_display_units(window, pump):
    """A specification is stored in SI, and its levels were written in
    SI under the display unit's heading: the workmanship profile (20 Hz
    0.01, 80 to 500 Hz 0.04, 2000 Hz 0.01 g²/Hz, 6.78 grms) read 66.53
    in a g system, its rms in m/s². The bars and the table read it in
    the unit their heading names."""
    from PySide6.QtCore import Qt

    from visualdynamics.core.data import Specification
    from visualdynamics.units import SI, SYSTEMS

    g = 9.80665
    profile = np.array([0.01, 0.04, 0.04, 0.01]) * g**2
    spec = Specification(abscissa=np.array([20.0, 80.0, 500.0, 2000.0]),
                         ordinate=profile[np.newaxis].astype(complex),
                         response_dof=['1Z+'],
                         ordinate_dim=['acceleration**2/frequency'])
    assert specification_rms(spec) == pytest.approx(6.784 * g, rel=1e-3), \
        'stored in SI'
    window.unit_system = SYSTEMS['in-slinch-lbf-s (g)']
    window.add_object('Workmanship', spec)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object('Workmanship')
    pump()
    window.data_pane.rms_action.trigger()
    pump()
    assert window.bar_chart.values() == pytest.approx([6.784], rel=1e-3)
    model = window.table.model()
    heading = model.headerData(1, Qt.Orientation.Horizontal,
                               Qt.ItemDataRole.DisplayRole)
    assert heading == 'Specification RMS [g]'
    assert model.data(model.index(0, 1),
                      Qt.ItemDataRole.DisplayRole) == '6.784'
    # and in SI, the stored value under m/s²
    window._units_changed(SI.name)
    pump()
    assert window.bar_chart.values() == pytest.approx([6.784 * g], rel=1e-3)


def test_the_level_is_read_in_the_quantity_the_dimension_names():
    """The unit a specification's rms is shown in comes from the
    package's one reading of a dimension (`channel_quantities`), so a
    cross term's or an undeclared dimension cannot be converted as if
    it named a quantity it does not."""
    from visualdynamics.core.data import channel_quantities
    from visualdynamics.units import DEFAULT_SYSTEM, UNKNOWN

    assert channel_quantities('acceleration**2/frequency') == (
        'acceleration', 'acceleration')
    assert channel_quantities('force*acceleration/frequency')[0] == 'force'
    assert channel_quantities(UNKNOWN) == (UNKNOWN, UNKNOWN)
    assert DEFAULT_SYSTEM.from_si(2.5, channel_quantities(UNKNOWN)[0]) == 2.5
