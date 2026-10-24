"""The confidence band in the window: the chi-square band shaded on an
averaged autospectrum, the confidence and the count in the pane beside
it.

A statement about a spectrum that exists, so a toggle on the bar with
no act (as the tone readout is), offered for an autospectrum and
nowhere else. The count is the one the spectrum was stamped with when
it was computed, or one typed in the pane for a spectrum that carries
none.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.averaging import (
    Averaging,
    chi_square_factors,
    dof_text,
)
from visualdynamics.core.data import Psd, TimeHistory

FS = 1024.0


def _record(seconds=8.0, channels=2):
    """White noise in g on two channels, averaged over Hann frames at
    half overlap."""
    rng = np.random.default_rng(1506)
    n = int(seconds * FS)
    history = TimeHistory(np.arange(n) / FS,
                          rng.standard_normal((channels, n)),
                          response_dof=[f'{i + 1}Z+' for i in
                                        range(channels)],
                          ordinate_dim=['acceleration'] * channels,
                          ordinate_unit=['g'] * channels)
    history.averaging = Averaging(frame_length=512, overlap=0.5,
                                  window='hann').filled(n, FS)
    return history


def _flat(window, pump, name):
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    return pane


def _asd(window, pump):
    window.add_object('Record', _record())
    name = window.project.compute_psds('Record')
    pane = _flat(window, pump, name)
    return name, window.objects[name], pane


def _band_on(window, pump):
    name, asd, pane = _asd(window, pump)
    pane.confidence_action.trigger()
    pump()
    return name, asd, pane


def _edges(window):
    import pyqtgraph as pg

    return [item for _plot, item in window.confidence_previews
            if isinstance(item, pg.PlotDataItem)]


def test_the_band_is_offered_on_an_autospectrum_alone(window, pump):
    _name, _shown, pane = _asd(window, pump)
    assert pane.confidence_action.isVisible()
    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not pane.confidence_action.isVisible(), \
        'a time history is not an average'
    f = np.arange(1.0, 65.0)
    cross = Psd(f, np.ones((1, 64)), response_dof=['1Z+'],
                reference_dof=['2Z+'],
                ordinate_dim='acceleration*acceleration/frequency')
    window.add_object('Cross', cross)
    _flat(window, pump, 'Cross')
    assert not pane.confidence_action.isVisible(), \
        'a cross term has no chi-square band'
    from visualdynamics.core.data import Specification

    target = Specification(f, np.full((1, 64), 0.04), response_dof=['1Z+'],
                           ordinate_dim='acceleration**2/frequency')
    target.statistical_dof = np.array([40.0])
    window.add_object('Target', target)
    _flat(window, pump, 'Target')
    assert not pane.confidence_action.isVisible(), \
        'a specification is a target, even one carrying a count'


def test_the_toggle_shades_the_band_and_reads_it(window, pump):
    _name, asd, pane = _band_on(window, pump)
    panel = pane.confidence_panel
    assert panel.isVisible()
    edges = _edges(window)
    assert len(edges) == 2 * asd.num_records, 'one band per record drawn'
    count = asd.statistical_dof[0]
    low, high = chi_square_factors(count, 0.9)
    derived = panel.derived
    assert derived['source'].text() == 'the averaging'
    assert derived['channel'].text() == '1Z+'
    assert derived['dof'].text() == dof_text(count)
    assert derived['below'].text() == f'{10 * np.log10(low):+.1f} dB'
    assert derived['above'].text() == f'{10 * np.log10(high):+.1f} dB'
    assert panel.level_box.currentText() == '90 %'
    assert panel.dof_box.text() == 'as averaged'
    band = asd.confidence_reading(0.9)[0]
    assert window.statusBar().currentMessage() == '1Z+: ' + band.describe()
    assert '90 lines in 100' in panel.notes.text()
    # the plot says it in words beside the shade
    plot = window.confidence_previews[0][0]
    names = [label.text for _sample, label in plot.legend.items]
    assert f'90 % confidence, {dof_text(count)} DOF' in names


def test_ninety_five_is_chosen_in_the_pane_and_kept(window, pump):
    _name, asd, pane = _band_on(window, pump)
    before = _edges(window)[1].getData()[1].copy()
    box = pane.confidence_panel.level_box
    box.setCurrentIndex(box.findData(0.95))
    pump()
    after = _edges(window)[1].getData()[1]
    _low, high90 = chi_square_factors(asd.statistical_dof[0], 0.90)
    _low, high95 = chi_square_factors(asd.statistical_dof[0], 0.95)
    assert np.allclose(after - before, np.log10(high95 / high90)), \
        'the upper edge moves up by the ratio of the two factors'
    assert pane.confidence_panel.derived['above'].text() == \
        f'{10 * np.log10(high95):+.1f} dB'
    # the confidence is one choice, kept for the next spectrum drawn
    window.render_current()
    pump()
    assert window._confidence_level == 0.95
    assert box.currentText() == '95 %'


def test_an_imported_spectrum_asks_for_its_count(window, pump):
    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    window.add_object('Imported', psd)
    pane = _flat(window, pump, 'Imported')
    assert pane.confidence_action.isVisible(), \
        'offered: the count can be typed'
    pane.confidence_action.trigger()
    pump()
    panel = pane.confidence_panel
    assert window.confidence_previews == [], 'no count, no band guessed'
    assert panel.derived['source'].text() == 'not known'
    assert 'Type the degrees of freedom' in panel.notes.text()
    assert window.statusBar().currentMessage().startswith(
        'No confidence band: this PSD does not say')
    panel.dof_box.setValue(40.0)
    pump()
    assert panel.derived['source'].text() == 'typed'
    assert panel.derived['dof'].text() == '40'
    assert len(_edges(window)) >= 2
    assert window._confidence_dofs == {'Imported': 40.0}
    # and back to the spectrum's own, which it does not have
    panel.dof_box.setValue(0.0)
    pump()
    assert window.confidence_previews == []
    assert 'Imported' not in window._confidence_dofs


def test_a_typed_count_overrides_the_averaging_s(window, pump):
    """Typed over a count the spectrum carries: the rows, the edge
    drawn and the legend all say 16, not the stamped count."""
    name, asd, pane = _band_on(window, pump)
    stamped = asd.statistical_dof[0]
    assert dof_text(stamped) != '16'
    pane.confidence_panel.dof_box.setValue(16.0)
    pump()
    derived = pane.confidence_panel.derived
    assert derived['source'].text() == 'typed'
    assert derived['dof'].text() == '16'
    assert derived['below'].text() == '-2.2 dB'
    assert derived['above'].text() == '+3.0 dB'
    assert window._confidence_dofs[name] == 16.0
    # the edge on the plot is the spectrum times the factor at 16
    low16, high16 = chi_square_factors(16.0, 0.9)
    low_n, high_n = chi_square_factors(stamped, 0.9)
    edges = _edges(window)
    lower, upper = (edge.getData()[1] for edge in edges[:2])
    plot = window.confidence_previews[0][0]
    log_y = plot.ctrl.logYCheck.isChecked()
    shown = np.real(asd.ordinate[0])
    seen = shown[np.isfinite(shown) & (shown > 0)]
    drawn_ratio = (10.0 ** (upper - lower) if log_y else upper / lower)
    assert np.allclose(drawn_ratio[np.isfinite(drawn_ratio)],
                       high16 / low16), \
        'the band is as wide as 16 DOF make it'
    assert not np.isclose(high16 / low16, high_n / low_n)
    assert seen.size
    names = [label.text for _sample, label in plot.legend.items]
    assert '90 % confidence, 16 DOF' in names
    assert not any(dof_text(stamped) in label for label in names)


def test_a_dead_first_channel_is_passed_over_not_read(window, pump):
    """A channel with no finite line has no band drawn, so the rows
    read the next channel and the note names the one passed over."""
    record = _record()
    record.ordinate[0] = np.nan
    window.add_object('Record', record)
    name = window.project.compute_psds('Record')
    pane = _flat(window, pump, name)
    pane.confidence_action.trigger()
    pump()
    derived = pane.confidence_panel.derived
    assert derived['channel'].text() == '2Z+'
    assert 'No band on 1Z+' in pane.confidence_panel.notes.text()
    assert window.statusBar().currentMessage().startswith('2Z+: ')


def test_no_channel_with_a_line_reads_no_band(window, pump):
    record = _record()
    record.ordinate[:] = np.nan
    window.add_object('Record', record)
    name = window.project.compute_psds('Record')
    pane = _flat(window, pump, name)
    pane.confidence_action.trigger()
    pump()
    panel = pane.confidence_panel
    assert panel.derived['channel'].text() == '—'
    assert panel.derived['dof'].text() == '—'
    assert 'no band to shade' in panel.notes.text()
    assert window.statusBar().currentMessage().startswith(
        'No confidence band: no line drawn')


def test_records_averaged_differently_are_said_to_differ(window, pump):
    _name, asd, pane = _asd(window, pump)
    asd.statistical_dof = np.array([30.0, 60.0])
    pane.confidence_action.trigger()
    pump()
    notes = pane.confidence_panel.notes.text()
    assert 'averaged to 30 to 60 DOF' in notes
    assert pane.confidence_panel.derived['dof'].text() == '30'


def test_the_stage_stands_down_while_the_band_is_shaded(window, pump):
    _name, _shown, pane = _band_on(window, pump)
    pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert window.confidence_previews, 'the band is on the flat plot'
    assert not (pane._waterfall_page is not None
                and pane._waterfall_page.isVisible())


def test_putting_the_band_away_takes_it_off(window, pump):
    _name, _shown, pane = _band_on(window, pump)
    pane.confidence_action.trigger()
    pump()
    assert window.confidence_previews == []
    assert not pane.confidence_panel.isVisible()


def test_the_band_is_journaled_as_its_plot_call(window, pump):
    name, _asd, pane = _band_on(window, pump)
    prefix = f"project[{name!r}].plot(marks='confidence'"
    lines = [line for line in window.project.journal
             if line.startswith(prefix)]
    assert lines == [(f"{prefix}, levels=[0.9], path='confidence.png', "
                     "show=False)")]
    # a change settles the same line rather than adding one
    pane.confidence_panel.dof_box.setValue(120.0)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith(prefix)]
    assert lines == [(f"{prefix}, levels=[0.9], dof=120.0, "
                     "path='confidence.png', show=False)")]


def test_the_journaled_call_names_the_records_drawn(window, pump):
    """One channel picked on the plot: the replay shades that channel,
    not every record, and the pane reads the channel picked."""
    name, _asd, pane = _band_on(window, pump)
    window._select_records(name, [1])
    window.render_current()
    pump()
    assert pane.confidence_panel.derived['channel'].text() == '2Z+'
    prefix = f"project[{name!r}].plot(marks='confidence'"
    lines = [line for line in window.project.journal
             if line.startswith(prefix)]
    assert lines == [(f"{prefix}, levels=[0.9], records=[1], "
                     "path='confidence.png', show=False)")]


@pytest.mark.parametrize('level', [0.9, 0.95])
def test_the_journaled_call_draws_the_same_band(window, pump, tmp_path,
                                                level):
    """What the console writes runs headless and draws."""
    _name, asd, pane = _band_on(window, pump)
    box = pane.confidence_panel.level_box
    box.setCurrentIndex(box.findData(level))
    pump()
    path = tmp_path / 'confidence.png'
    asd.plot(marks='confidence', levels=[level], path=str(path),
             show=False)
    assert path.stat().st_size > 2000
