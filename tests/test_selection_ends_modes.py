"""A mode belongs to the object it was entered on.

An edit, or a reading whose settings pane sits on the right of the view,
is something being done to one object. Moving to another ends it: the
button comes up, the pane goes and the preview goes with it. The numbers
typed into a pane are kept for the next time it is opened; the mode is
not.

Each test enters one mode and moves to a twin of the object it was on,
which is the case that matters: a twin can take the same reading, so a
mode left on shows up on it. It then comes back, because a flag left set
under a hidden button only shows on the way back.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from conftest import edit_category, fixture_path, select_objects
from test_drive_prediction_view import _selected as _prediction_pair
from test_fatigue_view import _history as _fatigue_record
from test_field_limit_view import _flights
from test_limit_prediction_view import _selected as _notch_pair
from test_matrix_checks_view import _plant
from test_matrix_comparison import SPEC, measured
from test_overload import table as _channel_table
from test_partial_coherence_view import _coherence
from test_rms_check_view import _asd as _rms_spectrum
from test_signature import AFTER, BEFORE, signature
from test_sine_profile import _controller
from test_sine_profile import _shown as _sine_specification
from test_sine_view import _sweep
from test_tone_view import _asd
from test_transformation_residual_view import _transformed
from test_transmissibility_view import _frf

from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.data import Psd, Specification, TimeHistory
from visualdynamics.core.fem import MATERIALS, GroupProperties


def _add_aside(window, name, obj):
    """Add `obj` leaving the selection where it is: adding an object
    selects it, which is a move of its own."""
    selected = list(window.tree.selectedItems())
    current = window.tree.currentItem()
    window.tree.blockSignals(True)
    added = window.add_object(name, obj)
    window.tree.setCurrentItem(current)
    window.tree.clearSelection()
    for item in selected:
        item.setSelected(True)
    window.tree.blockSignals(False)
    return added


def _twin(window, *names):
    """An exact copy of each object, unselected, under a new name."""
    return [_add_aside(window, f'{name} twin',
                       copy.deepcopy(window.objects[name])) for name in names]


def _away_and_back(window, pump, names, check):
    """Move to twins of `names`, `check`, come back, `check` again."""
    twins = _twin(window, *names)
    select_objects(window, pump, *twins)
    check()
    select_objects(window, pump, *names)
    check()


def _off(window, pump, names, action, panel):
    """Moving from `names` to their twins and back leaves `action`
    unchecked and `panel` hidden, both ways."""
    assert action.isChecked(), 'the mode was on to begin with'
    if panel is not None:
        assert panel.isVisible(), 'and its pane was up'

    def check():
        assert not action.isChecked()
        if panel is not None:
            assert not panel.isVisible()

    _away_and_back(window, pump, names, check)


def _gray_slot(window):
    """A row standing for a step of the project not yet filled."""
    from PySide6.QtWidgets import QTreeWidgetItemIterator

    from visualdynamics.gui.project_tree import ROLE_REFERENCE

    found = QTreeWidgetItemIterator(window.tree)
    while found.value() is not None:
        reference = found.value().data(0, ROLE_REFERENCE)
        if reference is not None and reference[0] == 'placeholder':
            return found.value()
        found += 1
    raise AssertionError('the project type holds no gray slot')


def _click_row(window, pump, name):
    """A real click on the row: `select_objects` passes through an empty
    selection on its way, and a click does not."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    tree = window.tree
    rect = tree.visualItemRect(window._item_for_object(name))
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                     pos=rect.center())
    pump()
    assert tree.currentItem() is window._item_for_object(name)


def _pick_in_grid(window, pump, name, row=0):
    """A record of `name` picked in its grid, a plain click, which reaches
    the window by the grid's signal and not the tree's."""
    window._item_for_object(name).setExpanded(True)
    pump()
    grid = window.record_grids[name]
    grid.clearSelection()
    grid.item(row, 0).setSelected(True)
    pump()


def _record(duration=4.0, rate=1024.0):
    t = np.arange(int(duration * rate)) / rate
    return TimeHistory(
        t, np.vstack([np.sin(2 * np.pi * 40.0 * t),
                      0.5 * np.cos(2 * np.pi * 90.0 * t)]),
        response_dof=['101Z+', '102Z+'],
        ordinate_dim=['acceleration', 'acceleration'])


def _on_a_record(window, pump):
    window.add_object('Record', _record())
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object('Record')
    pump()
    return window.data_pane


# ---- the readings of a record -------------------------------------------


def test_averaging_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.averaging_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.averaging_action, pane.averaging_panel)


def test_filter_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.filter_action, pane.filter_panel)


def test_sine_levels_end_on_another_object(window, pump):
    _sweep(window, pump)
    pane = window.data_pane
    pane.sine_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.sine_action, pane.sine_panel)


def test_truncate_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.truncate_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.truncate_action, pane.truncate_panel)


def test_kurtosis_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.kurtosis_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.kurtosis_action, None)


def test_statistics_end_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.statistics_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.statistics_action,
         pane.statistics_panel)


def test_amplitude_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.amplitude_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.amplitude_action,
         pane.amplitude_panel)


def test_overload_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    window.add_object('Channels', _channel_table([101, 102], [10.0, 10.0],
                                                 [100.0, 100.0]))
    window.project.link('Record', 'Channels')
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    pane.overload_action.trigger()
    pump()
    assert pane.overload_action.isChecked()
    assert pane.overload_panel.isVisible()
    # linked to the same table, or the twin is not offered the reading
    twin, = _twin(window, 'Record')
    window.project.link(twin, 'Channels')
    select_objects(window, pump, twin)
    window.render_current()
    pump()
    assert pane.overload_action.isVisible(), 'the twin can take it'
    assert not pane.overload_action.isChecked()
    assert not pane.overload_panel.isVisible()
    select_objects(window, pump, 'Record')
    assert not pane.overload_action.isChecked()
    assert not pane.overload_panel.isVisible()


def test_shocks_end_on_another_object(window, pump):
    window.import_paths([fixture_path('plate', 'shock.nc4')])
    pump()
    window.set_project_type('Shock')
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    select_objects(window, pump, name)
    pane = window.data_pane
    pane.shocks_action.trigger()
    pump()
    _off(window, pump, [name], pane.shocks_action, pane.shock_panel)


def test_wavelet_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.wavelet_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.wavelet_action, pane.wavelet_panel)


def test_pulse_check_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.pulse_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.pulse_action, pane.pulse_panel)


def test_replication_fit_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.fit_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.fit_action,
         pane.replication_fit_panel)


def test_stroke_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.stroke_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.stroke_action, pane.stroke_panel)


def test_resample_ends_on_another_object(window, pump):
    pane = _on_a_record(window, pump)
    pane.resample_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.resample_action, pane.resample_panel)


def test_fatigue_ends_on_another_object(window, pump):
    window.add_object('Record', _fatigue_record())
    window.show_object('Record')
    pump()
    pane = window.data_pane
    pane.fatigue_action.trigger()
    pump()
    _off(window, pump, ['Record'], pane.fatigue_action, pane.fatigue_panel)


# ---- the readings of a spectrum, an FRF and a matrix --------------------


def test_octave_bands_end_on_another_object(window, pump):
    name, _asd_object, pane = _asd(window, pump)
    pane.octave_action.trigger()
    pump()
    _off(window, pump, [name], pane.octave_action, pane.octave_panel)


def test_tones_end_on_another_object(window, pump):
    name, _asd_object, pane = _asd(window, pump)
    pane.tone_action.trigger()
    pump()
    _off(window, pump, [name], pane.tone_action, pane.tone_panel)


def test_confidence_ends_on_another_object(window, pump):
    name, _asd_object, pane = _asd(window, pump)
    pane.confidence_action.trigger()
    pump()
    _off(window, pump, [name], pane.confidence_action, pane.confidence_panel)


def test_rms_check_ends_on_another_object(window, pump):
    name, _asd_object, pane = _rms_spectrum(window, pump)
    pane.rms_check_action.trigger()
    pump()
    _off(window, pump, [name], pane.rms_check_action, pane.rms_check_panel)


def test_transmissibility_ends_on_another_object(window, pump):
    name, pane = _frf(window, pump)
    pane.transmissibility_action.trigger()
    pump()
    _off(window, pump, [name], pane.transmissibility_action,
         pane.transmissibility_panel)


def test_conditioning_ends_on_another_object(window, pump):
    window.add_object('Plant', _plant())
    window.show_object('Plant')
    pump()
    pane = window.data_pane
    pane.conditioning_action.trigger()
    pump()
    _off(window, pump, ['Plant'], pane.conditioning_action,
         pane.conditioning_panel)


def test_realizability_ends_on_another_object(window, pump):
    window.add_object('Run', measured())
    window.show_object('Run')
    pump()
    pane = window.data_pane
    pane.realizability_action.trigger()
    pump()
    _off(window, pump, ['Run'], pane.realizability_action,
         pane.realizability_panel)


def test_drive_prediction_ends_on_another_object(window, pump):
    pane = _prediction_pair(window, pump)
    pane.prediction_action.trigger()
    pump()
    _off(window, pump, ['Plant', 'Profile'], pane.prediction_action,
         pane.prediction_panel)


def test_notch_prediction_ends_on_another_object(window, pump):
    pane = _notch_pair(window, pump)
    pane.notch_action.trigger()
    pump()
    _off(window, pump, ['Low level H1', 'Profile'], pane.notch_action,
         pane.notch_panel)


def test_partial_coherence_ends_on_another_object(window, pump):
    name, pane = _coherence(window, pump)
    pane.partial_coherence_action.trigger()
    pump()
    _off(window, pump, [name], pane.partial_coherence_action,
         pane.partial_coherence_panel)


def test_rigid_body_residual_ends_on_another_object(window, pump):
    name, pane = _transformed(window, pump)
    pane.transformation_action.trigger()
    pump()
    _off(window, pump, [name], pane.transformation_action,
         pane.transformation_panel)


def test_field_limit_ends_on_another_object(window, pump):
    window.add_object('Flights', _flights())
    window.show_object('Flights')
    pump()
    pane = window.data_pane
    pane.field_limit_action.trigger()
    pump()
    _off(window, pump, ['Flights'], pane.field_limit_action,
         pane.field_limit_panel)


def _sine_tone():
    return _controller([0.26, 2.0, 2.0, 5.0, 5.0],
                       [5.0, 13.8, 28.0, 80.0, 500.0])


def test_sine_profile_ends_on_another_object(window, pump):
    pane, panel = _sine_specification(window, pump, _sine_tone())
    _off(window, pump, ['Sweep'], pane.sine_profile_action, panel)


def test_sweep_rate_ends_on_another_object(window, pump):
    window.add_object('Sweep', _sine_tone())
    select_objects(window, pump, 'Sweep')
    window.render_current()
    pump()
    pane = window.data_pane
    pane.sweep_rate_action.trigger()
    pump()
    _off(window, pump, ['Sweep'], pane.sweep_rate_action,
         pane.sweep_rate_panel)


def test_signature_ends_on_another_object(window, pump):
    window.add_object('Before', signature(BEFORE))
    window.add_object('After', signature(AFTER))
    select_objects(window, pump, 'Before', 'After')
    pane = window.data_pane
    pane.signature_action.trigger()
    pump()
    _off(window, pump, ['Before', 'After'], pane.signature_action,
         pane.signature_panel)


def test_matrix_comparison_ends_on_another_object(window, pump):
    window.add_object('Spec', SPEC)
    window.add_object('Run', measured())
    select_objects(window, pump, 'Spec', 'Run')
    pane = window.data_pane
    pane.matrix_action.trigger()
    pump()
    _off(window, pump, ['Spec', 'Run'], pane.matrix_action, pane.matrix_panel)


# ---- the readings of a comparison that hold a pane ----------------------


def _spec_and_run(window, pump):
    f = np.linspace(20.0, 2000.0, 400)
    dofs = ['101Z+', '102Z+']
    window.add_object('Spec', Specification(
        f, np.full((2, f.size), 1e-2), response_dof=dofs,
        reference_dof=dofs, ordinate_dim=['acceleration**2/frequency'] * 2))
    window.add_object('Run', Psd(
        f, np.full((2, f.size), 1.1e-2), response_dof=dofs,
        reference_dof=dofs, ordinate_dim=['acceleration**2/frequency'] * 2))
    select_objects(window, pump, 'Spec', 'Run')
    window.render_current()
    pump()
    return window.data_pane


def test_chance_ends_on_another_object(window, pump):
    pane = _spec_and_run(window, pump)
    pane.comparison_actions['chance'].trigger()
    pump()
    assert pane.chance_panel.isVisible()

    def check():
        assert pane.comparison_view == 'curves'
        assert not pane.chance_panel.isVisible()

    _away_and_back(window, pump, ['Spec', 'Run'], check)


def test_composite_ends_on_another_object(window, pump):
    from test_multi_point import _composite_pair

    _composite_pair(window, pump, 1.0, 1.1)
    names = ['Reference', 'Run']
    pane = window.data_pane
    assert pane.multi_point_panel.isVisible()

    def check():
        assert pane.comparison_view == 'curves'
        assert not pane.multi_point_panel.isVisible()

    _away_and_back(window, pump, names, check)


def test_segments_end_on_another_object(window, pump):
    from test_replication_segments import _segments_reading

    _segments_reading(window, pump)
    pane = window.data_pane
    assert pane.segments_panel.isVisible()

    def check():
        assert pane.replication_view == 'overlay'
        assert not pane.segments_panel.isVisible()

    _away_and_back(window, pump, ['Run', 'Reference'], check)


def test_points_rule_ends_on_another_object(window, pump):
    from test_srs_tolerance import _shock_pair

    _shock_pair(window, pump, (0.0, 'maximax'))
    names = ['Requirement', 'SRS 1']
    pane = window.data_pane
    assert pane.srs_tolerance_panel.isVisible()

    def check():
        assert pane.srs_view == 'curves'
        assert not pane.srs_tolerance_panel.isVisible()

    _away_and_back(window, pump, names, check)


# ---- the readings with nothing to set -----------------------------------


def _plain_again(window, pump, names, pane, view, plain, actions, choice):
    """Choice `choice` of group `view` on `names`; moving to their twins
    and back puts the group on `plain`, its button with it."""
    assert actions[choice].isVisible(), 'offered to begin with'
    actions[choice].trigger()
    pump()
    assert getattr(pane, view) == choice

    def check():
        assert getattr(pane, view) == plain
        assert not actions[choice].isChecked()

    _away_and_back(window, pump, names, check)


def test_rms_ends_on_another_object(window, pump):
    from test_matrix_comparison import SPEC as spec

    window.add_object('Spec', spec)
    select_objects(window, pump, 'Spec')
    pane = window.data_pane
    assert pane.rms_action.isVisible()
    pane.rms_action.trigger()
    pump()
    _off(window, pump, ['Spec'], pane.rms_action, None)


def test_cmif_ends_on_another_object(window, pump, survey):
    _truth, frfs = survey
    window.add_object('FRFs', frfs)
    select_objects(window, pump, 'FRFs')
    pane = window.data_pane
    assert pane.cmif_action.isVisible()
    pane.cmif_action.trigger()
    pump()
    _off(window, pump, ['FRFs'], pane.cmif_action, None)


def test_rms_error_ends_on_another_object(window, pump):
    pane = _spec_and_run(window, pump)
    _plain_again(window, pump, ['Spec', 'Run'], pane, 'comparison_view',
                 'curves', pane.comparison_actions, 'error')


def test_lines_out_ends_on_another_object(window, pump):
    pane = _spec_and_run(window, pump)
    _plain_again(window, pump, ['Spec', 'Run'], pane, 'comparison_view',
                 'curves', pane.comparison_actions, 'lines')


def test_waveform_error_ends_on_another_object(window, pump):
    from test_replication_segments import _small_pair

    run, reference = _small_pair()
    window.add_object('Run', run)
    window.add_object('Reference', reference)
    select_objects(window, pump, 'Run', 'Reference')
    pane = window.data_pane
    _plain_again(window, pump, ['Run', 'Reference'], pane, 'replication_view',
                 'overlay', pane.replication_actions, 'waveform')


def test_srs_error_bars_end_on_another_object(window, pump):
    from test_srs_tolerance import _shock_pair

    _shock_pair(window, pump, (0.0, 'maximax'))
    pane = window.data_pane
    pane.srs_actions['curves'].trigger()
    pump()
    _plain_again(window, pump, ['Requirement', 'SRS 1'], pane, 'srs_view',
                 'curves', pane.srs_actions, 'error')


def _two_densities(window, pump):
    f = np.linspace(20.0, 2000.0, 400)
    dofs = ['101Z+', '102Z+']
    for name, level in (('Loud', 1e-2), ('Quiet', 1e-4)):
        window.add_object(name, Psd(
            f, np.full((2, f.size), level), response_dof=dofs,
            reference_dof=dofs,
            ordinate_dim=['acceleration**2/frequency'] * 2))
    select_objects(window, pump, 'Loud', 'Quiet')
    window.render_current()
    pump()
    return window.data_pane


def test_signal_to_noise_ends_on_another_object(window, pump):
    pane = _two_densities(window, pump)
    _plain_again(window, pump, ['Loud', 'Quiet'], pane, 'spectra_view',
                 'overlay', pane.spectra_actions, 'ratio')


def test_rms_signal_to_noise_ends_on_another_object(window, pump):
    pane = _two_densities(window, pump)
    _plain_again(window, pump, ['Loud', 'Quiet'], pane, 'spectra_view',
                 'overlay', pane.spectra_actions, 'snr')


def test_resynthesis_ends_on_another_object(window, pump, survey):
    """Ended, an FRF and a shape set read as a pair does until told
    otherwise, and that is the fit."""
    truth, frfs = survey
    window.add_object('FRF', frfs)
    window.add_object('Shapes', truth)
    window.add_object('Elsewhere', _record())
    select_objects(window, pump, 'FRF', 'Shapes')
    pane = window.data_pane
    assert window.fit is not None, 'the fit is the default'
    pane.pair_synthesis_action.trigger()
    pump()
    assert window.fit is None and window.pair_mode == 'overlay'
    select_objects(window, pump, 'Elsewhere')
    assert window.pair_mode == 'fit'
    assert not pane.pair_synthesis_action.isVisible()
    select_objects(window, pump, 'FRF', 'Shapes')
    assert window.fit is not None
    assert pane.pair_edit_action.isChecked()


def test_resynthesis_stays_while_its_modes_are_picked(window, pump, survey):
    """The modes of the set being read are the same objects."""
    truth, frfs = survey
    window.add_object('FRF', frfs)
    window.add_object('Shapes', truth)
    select_objects(window, pump, 'FRF', 'Shapes')
    pane = window.data_pane
    pane.pair_synthesis_action.trigger()
    pump()
    _pick_in_grid(window, pump, 'Shapes')
    assert window.pair_mode == 'overlay'
    assert window.fit is None


def test_drive_points_end_on_another_object(window, pump, survey):
    """Not a mode with state of its own: the button is down exactly when
    the picks are the diagonal, and the picks go with the move."""
    _truth, frfs = survey
    window.add_object('FRFs', frfs)
    select_objects(window, pump, 'FRFs')
    pane = window.data_pane
    assert pane.drive_point_action.isVisible()
    pane.drive_point_action.trigger()
    pump()
    assert window.record_grids['FRFs'].selected_records()
    _off(window, pump, ['FRFs'], pane.drive_point_action, None)
    assert not window.record_grids['FRFs'].selected_records()


# ---- the sheet and the acts' panes --------------------------------------


def test_the_specification_sheet_ends_on_another_object(window, pump):
    from test_author_view import _open, _populate

    _populate(window, pump)
    _open(window, pump)
    action = window.author_action
    panel = window.data_pane.author_panel
    name = next(n for n, o in window.objects.items()
                if isinstance(o, ChannelTable))
    assert panel.isVisible()
    window.add_object('Elsewhere', _record())
    select_objects(window, pump, 'Elsewhere')
    assert not panel.isVisible()
    assert not action.isChecked() and not window.author_data_action.isChecked()
    select_objects(window, pump, name)
    assert not panel.isVisible()
    assert not action.isChecked()


def test_the_sheet_ends_on_a_part_of_another_object(window, pump):
    """A row inside another object is another object as much as the
    object's own row is, and the sheet has no door there."""
    from test_author_view import _open, _populate

    _populate(window, pump)
    _open(window, pump)
    panel = window.data_pane.author_panel
    spec = next(n for n, o in window.objects.items()
                if isinstance(o, Specification))
    _add_aside(window, 'Elsewhere', _record())
    assert panel.isVisible()
    _pick_in_grid(window, pump, 'Elsewhere')
    assert not panel.isVisible()
    select_objects(window, pump, spec)
    assert not panel.isVisible()
    assert not window.author_data_action.isChecked()


def test_generate_signal_ends_on_another_object(window, pump):
    from test_generate_signal import _open as _open_signal
    from test_generate_signal import _target

    window.add_object('Target', _target())
    window.add_object('Elsewhere', _record())
    panel = _open_signal(window, pump, 'Target')
    assert panel.isVisible()
    select_objects(window, pump, 'Elsewhere')
    assert not panel.isVisible() and window._signaling is None


def test_convert_to_psd_ends_on_another_object(window, pump):
    from test_convert_breakpoints import _open

    panel = _open(window, pump)
    assert panel.isVisible()
    window.add_object('Elsewhere', _record())
    select_objects(window, pump, 'Elsewhere')
    assert not panel.isVisible() and window._lining is None


def test_modal_responses_end_on_another_object(window, pump):
    from test_modal_response import _populate

    _populate(window, pump)
    select_objects(window, pump, 'Run', 'Shapes')
    window.modal_response_act()
    pump()
    panel = window.data_pane.modal_response_panel
    assert panel.isVisible()
    select_objects(window, pump, 'Run')
    assert not panel.isVisible() and window._modaling is None


def test_state_space_export_ends_on_another_object(window, pump, monkeypatch):
    from test_state_space import _open

    panel = _open(window, pump, monkeypatch)
    assert panel.isVisible()
    select_objects(window, pump, 'Modes')
    assert not panel.isVisible() and window._staging is None


def test_imported_units_end_on_another_object(window, pump):
    window.add_object('Record', _record())
    window.add_object('Elsewhere', _record())
    select_objects(window, pump, 'Record')
    window.define_units()
    pump()
    assert window.units_panel.isVisible()
    select_objects(window, pump, 'Elsewhere')
    assert not window.units_panel.isVisible()
    assert window.units_target is None


# ---- the fit ------------------------------------------------------------


def test_the_modal_fit_ends_on_another_object(window, pump, survey):
    _truth, frfs = survey
    window.add_object('FRFs', frfs)
    window.add_object('Elsewhere', _record())
    select_objects(window, pump, 'FRFs')
    window.start_modal_fit()
    pump()
    assert window.fit is not None and window.fit_bar.isVisible()
    select_objects(window, pump, 'Elsewhere')
    assert window.fit is None and not window.fit_bar.isVisible()


# ---- the report ---------------------------------------------------------


@pytest.mark.parametrize('elsewhere', ['Run', 'Channels'])
def test_the_report_editor_ends_on_another_object(window, pump, elsewhere):
    """To a record, which takes the view from the table pane, and to a
    channel table, which shares that pane with the report."""
    window.add_object('Run', _record())
    window.add_object('Channels', _channel_table([101, 102], [10.0, 10.0],
                                                 [100.0, 100.0]))
    name = window.generate_report('random')
    select_objects(window, pump, name)
    window.render_current()
    pump()
    assert window.report_editor.isVisible()
    _click_row(window, pump, elsewhere)
    assert not window.report_editor.isVisible()


# ---- the 3-D view: its reading, editing and its panes -------------------


@pytest.fixture
def bar(window, pump):
    """A meshed, free steel bar with a material, and a second part."""
    project = window.project
    for name in ('Bar', 'Other'):
        made = project.new_geometry(name, unit='in')
        solid = project.add_solid_block(made, (0, 0, 0), (8, 2, 1),
                                        name='block', unit='in')
        project.mesh_solids(made, [solid], 1.0, unit='in')
        geometry = project[made]
        for group in geometry.group_id:
            geometry.group_properties[int(group)] = GroupProperties(
                MATERIALS['1018 steel'])
        window.show_object(made)
    select_objects(window, pump, 'Bar')
    return project['Bar']


def test_rigid_body_modes_end_on_another_object(bar, window, pump):
    scene = window.scene
    scene.modes_action.trigger()
    pump()
    assert scene.modes_action.isChecked() and scene.modes_panel.isVisible()
    select_objects(window, pump, 'Other')
    assert not scene.modes_action.isChecked()
    assert not scene.modes_panel.isVisible()
    assert window._rigid_preview is None
    select_objects(window, pump, 'Bar')
    assert not scene.modes_action.isChecked()
    assert not scene.modes_panel.isVisible()


def _edit(window, pump, label='Nodes'):
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    window.add_object('Other', copy.deepcopy(window.objects['Geometry']))
    return edit_category(window, pump, label)


def test_editing_ends_on_another_object(window, pump):
    _edit(window, pump)
    assert window.editing is not None
    assert window.edit_toggle_action.isChecked()
    select_objects(window, pump, 'Other')
    assert window.editing is None
    assert not window.edit_toggle_action.isChecked()


def test_editing_ends_on_a_record_in_another_objects_grid(window, pump):
    """A record picked in another object's grid reaches the window by the
    grid's own signal and not the tree's, and it is the same move."""
    window.add_object('Record', _record())
    _edit(window, pump)
    assert window.editing is not None
    _pick_in_grid(window, pump, 'Record')
    assert window.editing is None
    assert not window.edit_toggle_action.isChecked()


def test_add_mode_ends_on_another_object(window, pump):
    _edit(window, pump)
    window.add_action.setChecked(True)
    pump()
    assert window.add_mode
    select_objects(window, pump, 'Other')
    assert not window.add_mode
    assert not window.add_action.isChecked()
    assert not window.add_action.isVisible()


def test_rotate_ends_on_another_object(window, pump):
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    window.add_object('Other', copy.deepcopy(window.objects['Geometry']))
    window.objects['Geometry'].add_coordinate_system(origin=(0.1, 0.0, 0.0))
    edit_category(window, pump, 'Coordinate')
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    assert window.rotate_action.isChecked()
    select_objects(window, pump, 'Other')
    assert window.editing is None
    assert not window.rotate_action.isChecked()
    assert not window.rotate_action.isVisible()
    assert window._angle_actions[0].isVisible() is False


def test_tie_ends_on_another_object(window, pump):
    _edit(window, pump, 'Quads')
    window.table.selectRow(0)
    pump()
    window._await_second_patch([window.table.model().index(0, 0).row()])
    pump()
    assert window.tie_action.isChecked()
    select_objects(window, pump, 'Other')
    assert window._tie_patch is None
    assert not window.tie_action.isChecked()
    assert not window.tie_action.isVisible()


def test_add_solid_pane_ends_on_another_object(bar, window, pump):
    window.add_solid_block_act()
    pump()
    assert window.scene.mesh_panel.isVisible()
    select_objects(window, pump, 'Other')
    assert window._mesh is None
    assert not window.scene.mesh_panel.isVisible()


def test_add_solid_pane_ends_on_a_record_in_another_objects_grid(
        bar, window, pump):
    window.add_object('Record', _record())
    select_objects(window, pump, 'Bar')
    window.add_solid_block_act()
    pump()
    assert window.scene.mesh_panel.isVisible()
    _pick_in_grid(window, pump, 'Record')
    assert window._mesh is None
    assert not window.scene.mesh_panel.isVisible()


def _solid_rows(window, pump, name='Bar'):
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    found = []

    def walk(item):
        item.setExpanded(True)
        for k in range(item.childCount()):
            child = item.child(k)
            if (child.data(0, ROLE_REFERENCE) or (None,))[0] == 'solid':
                found.append(child)
            walk(child)

    walk(window._item_for_object(name))
    pump()
    return found


def test_mesh_solids_pane_ends_on_another_object(bar, window, pump):
    row = _solid_rows(window, pump)[0]
    window.tree.clearSelection()
    window.tree.setCurrentItem(row)
    row.setSelected(True)
    pump()
    window.mesh_solids_act()
    pump()
    assert window.scene.mesh_solids_panel.isVisible()
    select_objects(window, pump, 'Other')
    assert window._meshing is None
    assert not window.scene.mesh_solids_panel.isVisible()


def test_pattern_pane_ends_on_another_object(bar, window, pump):
    row = _solid_rows(window, pump)[0]
    window.tree.clearSelection()
    window.tree.setCurrentItem(row)
    row.setSelected(True)
    pump()
    window.pattern_solids_act()
    pump()
    assert window.scene.pattern_panel.isVisible()
    select_objects(window, pump, 'Other')
    assert window._patterning is None
    assert not window.scene.pattern_panel.isVisible()


def _two_groups(window, pump):
    from test_align_meshes import _wing_on_frame, group_row

    window.add_object('Geometry', _wing_on_frame())
    window.add_object('Other', _wing_on_frame())
    window.tree.clearSelection()
    window.tree.setCurrentItem(group_row(window, 1))
    group_row(window, 2).setSelected(True)
    pump()
    return {act[0]: act for act in window.acts_for()}


def test_merge_pane_ends_on_another_object(window, pump):
    _two_groups(window, pump)['merge_coincident_nodes'][3]()
    pump()
    assert window.scene.merge_panel.isVisible()
    select_objects(window, pump, 'Other')
    assert window._merging is None
    assert not window.scene.merge_panel.isVisible()


def test_mate_pane_ends_on_another_object(window, pump):
    from test_mate import _group, _two_blocks

    _project, _name, geometry = _two_blocks()
    window.add_object('Geometry', geometry)
    window.add_object('Other', copy.deepcopy(geometry))
    window.mate_surfaces_act('Geometry', [_group(geometry, 'post'),
                                          _group(geometry, 'base')])
    pump()
    assert window.scene.mate_panel.isVisible()
    select_objects(window, pump, 'Other')
    assert window.__dict__.get('_mating') is None
    assert not window.scene.mate_panel.isVisible()


def test_plan_instrumentation_ends_on_another_object(bar, window, pump):
    modes = window.project.solve_modes('Bar', num_modes=8)
    window.show_object(modes)
    pump()
    {act[0]: act for act in window.acts_for()}['plan_instrumentation'][3]()
    pump()
    assert window.scene.instrumentation_panel.isVisible()
    select_objects(window, pump, 'Other')
    assert window.__dict__.get('_planning') is None
    assert not window.scene.instrumentation_panel.isVisible()


# ---- how the same object is told from another ---------------------------


def test_picking_a_records_own_channels_keeps_the_reading(window, pump):
    """Channels of the record a reading is on are the same object."""
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    _pick_in_grid(window, pump, 'Record', row=1)
    assert window.record_grids['Record'].selected_records()
    assert pane.filter_action.isChecked()
    assert pane.filter_panel.isVisible()
    _pick_in_grid(window, pump, 'Record', row=0)
    assert pane.filter_action.isChecked()
    assert pane.filter_panel.isVisible()


def test_adding_an_object_to_the_selection_keeps_the_reading(window, pump):
    """The object it was on is still selected."""
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    twin, = _twin(window, 'Record')
    select_objects(window, pump, 'Record', twin)
    select_objects(window, pump, 'Record')
    assert pane.filter_action.isChecked()
    assert pane.filter_panel.isVisible()


def test_a_reading_does_not_follow_a_pair_on_to_its_other_half(window, pump):
    """Filter on A, B added beside it, then B alone: the filter was never
    on B."""
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    twin, = _twin(window, 'Record')
    select_objects(window, pump, 'Record', twin)
    select_objects(window, pump, twin)
    assert not pane.filter_action.isChecked()
    assert not pane.filter_panel.isVisible()


def test_a_redraw_keeps_the_reading(window, pump):
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    window.render_current()
    pump()
    assert pane.filter_action.isChecked()
    assert pane.filter_panel.isVisible()


def test_renaming_the_object_keeps_the_reading(window, pump):
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    window._item_for_object('Record').setText(0, 'Renamed')
    pump()
    assert 'Renamed' in window.objects
    select_objects(window, pump, 'Renamed')
    assert pane.filter_action.isChecked()
    assert pane.filter_panel.isVisible()


def test_a_reading_ends_on_a_click_on_another_objects_row(window, pump):
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    twin, = _twin(window, 'Record')
    _click_row(window, pump, twin)
    assert not pane.filter_action.isChecked()
    assert not pane.filter_panel.isVisible()


def test_a_reading_ends_on_a_record_in_another_objects_grid(window, pump):
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    twin, = _twin(window, 'Record')
    _pick_in_grid(window, pump, twin)
    assert not pane.filter_action.isChecked()
    assert not pane.filter_panel.isVisible()


def test_a_reading_ends_on_the_project_row(window, pump):
    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    window.tree.setCurrentItem(window.test_item)
    pump()
    assert not pane.filter_panel.isVisible()
    select_objects(window, pump, 'Record')
    assert not pane.filter_action.isChecked()
    assert not pane.filter_panel.isVisible()


def test_nothing_selected_puts_every_pane_away(window, pump):
    """With nothing selected there is nothing for a pane to edit."""
    from test_generate_signal import _open as _open_signal
    from test_generate_signal import _target

    pane = _on_a_record(window, pump)
    pane.filter_action.trigger()
    pump()
    window.tree.clearSelection()
    window.tree.setCurrentItem(None)
    pump()
    assert not pane.filter_panel.isVisible()
    assert not pane.filter_action.isVisible(), 'and its button off the bar'
    window.add_object('Target', _target())
    panel = _open_signal(window, pump, 'Target')
    assert panel.isVisible()
    window.tree.clearSelection()
    window.tree.setCurrentItem(None)
    pump()
    assert not panel.isVisible()


def test_a_gray_slot_is_nothing_selected(window, pump):
    """A gray slot cannot be selected, so a click on one selects nothing:
    the pane goes, and the reading is there again on the same object and
    gone on another."""
    pane = _on_a_record(window, pump)
    window.set_project_type('Random Vibration')
    pump()
    select_objects(window, pump, 'Record')
    pane.filter_action.trigger()
    pump()
    window.tree.setCurrentItem(_gray_slot(window))
    pump()
    assert not window.tree.selectedItems()
    assert not pane.filter_panel.isVisible()
    select_objects(window, pump, 'Record')
    assert pane.filter_action.isChecked()
    assert pane.filter_panel.isVisible()
    window.tree.setCurrentItem(_gray_slot(window))
    pump()
    twin, = _twin(window, 'Record')
    select_objects(window, pump, twin)
    assert not pane.filter_action.isChecked()
    assert not pane.filter_panel.isVisible()


# ---- what the pane remembers, and what deliberately stays ---------------


def test_the_settings_are_kept_for_next_time(window, pump):
    """The mode ends; what was typed into its pane does not."""
    pane = _on_a_record(window, pump)
    pane.wavelet_action.trigger()
    pump()
    pane.wavelet_panel.low_box.setValue(45.0)
    pump()
    twin, = _twin(window, 'Record')
    select_objects(window, pump, twin)
    assert not pane.wavelet_action.isChecked()
    pane.wavelet_action.trigger()
    pump()
    assert pane.wavelet_settings['low'] == pytest.approx(45.0)


def test_how_things_are_drawn_stays_across_a_move(window, pump):
    """The 2-D/3-D toggle says how any object is drawn. It belongs to no
    one object, and it holds no pane."""
    pane = _on_a_record(window, pump)
    assert not pane.waterfall_action.isChecked()
    pane.filter_action.trigger()
    pump()
    twin, = _twin(window, 'Record')
    select_objects(window, pump, twin)
    assert not pane.filter_action.isChecked(), 'the move was a move'
    assert not pane.waterfall_action.isChecked()


def test_how_things_are_drawn_on_the_window_stays_across_a_move(window, pump):
    """The same for the scene's and the table bar's toggles: each set
    against its default, and each still so after a reading ended."""
    pane = _on_a_record(window, pump)
    actions = (window.colormap_action, window.sensors_action,
               window.mode_table_action, window.mac_action,
               window.mac_bars_action, window.overlay_action)
    for action in actions:
        assert action.isChecked(), 'on by default'
        action.setChecked(False)
    pane.filter_action.trigger()
    pump()
    twin, = _twin(window, 'Record')
    select_objects(window, pump, twin)
    assert not pane.filter_action.isChecked(), 'the move was a move'
    assert not any(action.isChecked() for action in actions)


def test_a_comparisons_bars_stay_on_the_same_object(window, pump):
    """The channels of a run picked, the run still being read: a reading
    with no pane keeps to the same-object rule the others do."""
    pane = _spec_and_run(window, pump)
    pane.comparison_actions['error'].trigger()
    pump()
    _pick_in_grid(window, pump, 'Run')
    assert pane.comparison_view == 'error'
    # the run current first: the specification alone would be a move off
    # the run the pick left it on
    select_objects(window, pump, 'Run', 'Spec')
    assert pane.comparison_view == 'error'
    assert pane.comparison_actions['error'].isChecked()
