"""A shape set and a channel table as a modal state-space model.

Brandon, 2026-10-31: to run Rattlesnake synthetically on the RC-on-plate
model, a shape set and its channel table are written as the A, B, C and
D matrices Rattlesnake's state-space virtual hardware integrates. The
model is checked against the set's own synthesized FRF, the one the
rest of the package already trusts, by evaluating C (iωI − A)⁻¹ B + D
directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.state_space import state_space_model
from visualdynamics.units import SI, SYSTEMS, si_factor

LINES = np.array([0.5, 3.0, 47.0, 50.0, 300.0, 900.0])
POWER = {'length': 0, 'velocity': 1, 'acceleration': 2}


def _shapes(**kwargs):
    rng = np.random.default_rng(7)
    options = {'mass_unit': 'kg'}
    options.update(kwargs)
    return visualdynamics.ShapeSet(
        frequency=[0.0, 50.0, 120.0, 5000.0],
        damping=[0.02, 0.01, 0.03, 0.02],
        coordinate=['1X+', '2X+', '3X+', '4Y+'],
        shape_matrix=rng.standard_normal((4, 4)),
        modal_mass=[1.0, 1.0, 2.0, 1.0], **options)


def _table(**changes):
    columns = {
        'channel': [1, 2, 3, 4, 5],
        'node': [1, 2, 3, 4, 3],
        'direction': ['X+', 'X+', 'X-', 'Y+', 'X+'],
        'channel_type': ['force', 'acceleration', 'velocity', 'force',
                         'length'],
        'unit': ['lbf', 'g', 'in/s', 'N', 'mm'],
        'role': ['reference', 'response', 'response', 'reference',
                 'monitor']}
    columns.update(changes)
    return ChannelTable(columns)


def _frf(model, lines):
    """C (iωI − A)⁻¹ B + D at each line, (lines, outputs, inputs),
    written out here rather than taken from `model.frf`, which the
    preview draws and this test checks."""
    eye = np.eye(model.A.shape[0])
    return np.array([model.C @ np.linalg.solve(2j * np.pi * f * eye - model.A,
                                               model.B) + model.D
                     for f in lines])


def test_the_models_own_frf_is_the_matrices():
    model = state_space_model(_shapes(), _table(), maximum_frequency=1000.0)
    assert np.array_equal(model.frf(LINES), _frf(model, LINES))
    assert model.frf(LINES).shape == (6, 5, 2)


def test_the_model_is_the_sets_synthesized_frf_in_the_tables_units():
    shapes, table = _shapes(), _table()
    model = state_space_model(shapes, table, maximum_frequency=1000.0)
    # the set as the model holds it: the rigid mode lifted to 1 Hz, the
    # 5000 Hz mode dropped
    kept = visualdynamics.ShapeSet(
        frequency=[1.0, 50.0, 120.0], damping=shapes.damping[:3],
        coordinate=shapes.coordinate, shape_matrix=shapes.shape_matrix[:3],
        modal_mass=shapes.modal_mass[:3], mass_unit='kg')
    got = _frf(model, LINES)
    inputs = [(0, '1X+', 'lbf'), (3, '4Y+', 'N')]
    for row, (dof, kind, unit) in enumerate(zip(
            table.dof_strings(), table.types(), list(table.frame['unit']))):
        for column, (_r, reference, force_unit) in enumerate(inputs):
            if kind == 'force':
                continue
            want = kept.synthesize_frf(LINES, [dof], [reference],
                                       power=POWER[kind])[0]
            want = want * si_factor(force_unit, 'force') / si_factor(unit, kind)
            assert np.allclose(got[:, row, column], want,
                               rtol=1e-9, atol=1e-12 * np.abs(want).max()), \
                (row, column)
    assert model.A.shape == (6, 6) and model.B.shape == (6, 2)
    assert model.C.shape == (5, 6) and model.D.shape == (5, 2), \
        'a row per channel, a column per reference, as Rattlesnake checks'
    assert model.lifted == 1 and model.dropped == 1
    assert list(model.mode_frequency) == [1.0, 50.0, 120.0]


def test_a_reference_row_reads_its_own_input():
    model = state_space_model(_shapes(), _table(), maximum_frequency=1000.0)
    assert not model.C[[0, 3]].any()
    assert np.array_equal(model.D[[0, 3]], np.eye(2)), \
        'the force gauge reads the force applied, in its own unit'


def test_the_units_scale_rows_and_columns():
    """A row in g is 1/9.80665 of the same row in m/s²; a column in lbf
    is 4.448 times the same column in N."""
    base = state_space_model(_shapes(), _table(unit=['N', 'm/s^2', 'in/s',
                                                     'N', 'mm']))
    scaled = state_space_model(_shapes(), _table())
    assert np.allclose(scaled.C[1], base.C[1] / 9.80665, rtol=1e-12)
    assert np.allclose(scaled.D[1, 0], base.D[1, 0] / 9.80665 * 4.4482216152605,
                       rtol=1e-12)
    assert np.allclose(scaled.B[:, 0], base.B[:, 0] * 4.4482216152605,
                       rtol=1e-12)
    assert np.array_equal(scaled.B[:, 1], base.B[:, 1])
    assert scaled.channel_unit == ['lbf', 'g', 'in/s', 'N', 'mm']


def test_a_blank_unit_is_the_systems():
    table = _table(unit=['', '', '', 'N', 'mm'])
    model = state_space_model(_shapes(), table,
                              unit_system=SYSTEMS['in-slinch-lbf-s (g)'])
    assert model.channel_unit == ['lbf', 'g', 'in/s', 'N', 'mm']
    named = state_space_model(_shapes(), _table())
    assert np.allclose(model.C, named.C) and np.allclose(model.B, named.B)
    si = state_space_model(_shapes(), table)
    assert si.channel_unit == ['N', SI.unit('acceleration'), 'm/s', 'N',
                               'mm'], 'SI with no system named'


def test_every_mode_is_kept_with_no_limit_and_a_soft_mode_is_lifted():
    model = state_space_model(_shapes(), _table(), rigid_frequency=2.5)
    assert list(model.mode_frequency) == [2.5, 50.0, 120.0, 5000.0]
    assert model.dropped == 0 and model.lifted == 1
    soft = visualdynamics.ShapeSet(
        frequency=[0.4, 50.0], damping=[0.01, 0.01],
        coordinate=['1X+', '2X+', '3X+', '4Y+'],
        shape_matrix=np.ones((2, 4)), mass_unit='kg')
    nearly = state_space_model(soft, _table())
    assert list(nearly.mode_frequency) == [1.0, 50.0], \
        'a mode under the rigid frequency is lifted to it'
    assert nearly.lifted == 1
    at = state_space_model(soft, _table(), rigid_frequency=50.0)
    assert list(at.mode_frequency) == [50.0, 50.0] and at.lifted == 1, \
        'a mode already at the rigid frequency is not counted as lifted'


def test_the_summary_says_what_was_changed():
    model = state_space_model(_shapes(), _table(), maximum_frequency=1000.0)
    assert model.summary() == (
        '3 modes, 2 inputs, 5 channels; 1 mode moved up to 1 Hz; 1 mode '
        'above the maximum frequency left out; written in lbf, g, in/s, N, mm')
    table = _table(node=[1, 2, 3, 4, 4], direction=['X+', 'X+', 'X-', 'Y+',
                                                     'Y+'])
    soft = visualdynamics.ShapeSet(
        frequency=[0.0, 0.2, 50.0, 2000.0, 3000.0],
        damping=[0.01] * 5, coordinate=['1X+', '2X+', '3X+', '4Y+'],
        shape_matrix=np.ones((5, 4)), mass_unit='kg')
    model = state_space_model(soft, table, maximum_frequency=1000.0)
    assert model.summary() == (
        '3 modes, 2 inputs, 5 channels; 2 modes moved up to 1 Hz; 2 modes '
        'above the maximum frequency left out; written in lbf, g, in/s, N, mm')
    assert state_space_model(soft, table, rigid_frequency=0.1).summary() == (
        '5 modes, 2 inputs, 5 channels; 1 mode moved up to 0.1 Hz; '
        'written in lbf, g, in/s, N, mm')


def test_the_damping_is_kept_on_a_lifted_mode():
    model = state_space_model(_shapes(), _table(), maximum_frequency=1000.0)
    block = model.A[:2, :2]
    omega = 2 * np.pi * 1.0
    assert np.allclose(block, [[0.0, 1.0], [-omega ** 2, -2 * 0.02 * omega]])


@pytest.mark.parametrize('change, match', [
    ({'node': [1, 2, 3, 4, 9]}, 'no coefficient at 9X'),
    ({'node': [1, 2, 3, 4, '']}, 'channel 5 has no node or direction'),
    ({'role': ['response'] * 5}, 'no reference rows'),
    ({'channel_type': ['force', 'strain', 'velocity', 'force', 'length']},
     'channel 2 is a strain'),
    ({'channel_type': ['acceleration', 'acceleration', 'velocity', 'force',
                       'length'], 'unit': ['g', 'g', 'in/s', 'N', 'mm']},
     'channel 1 is a reference but measures acceleration'),
])
def test_a_table_the_model_cannot_answer_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        state_space_model(_shapes(), _table(**change))


def test_a_set_with_no_physical_scale_is_refused():
    with pytest.raises(ValueError, match='mass unit'):
        state_space_model(_shapes(mass_unit=None), _table())
    with pytest.raises(ValueError, match='no drive point'):
        state_space_model(_shapes(unscaled=True), _table())
    complex_set = _shapes()
    complex_set.shape_matrix = complex_set.shape_matrix * (1 + 0.1j)
    with pytest.raises(ValueError, match='complex'):
        state_space_model(complex_set, _table())
    with pytest.raises(ValueError, match='drops every mode'):
        state_space_model(_shapes(), _table(), maximum_frequency=0.0)


def test_the_file_is_what_rattlesnake_reads(tmp_path):
    from visualdynamics.io.rattlesnake_state_space import save

    model = state_space_model(_shapes(), _table(), maximum_frequency=1000.0)
    path = tmp_path / 'plant.npz'
    save(model, path)
    with np.load(path, allow_pickle=False) as data:
        for key in 'ABCD':
            assert np.array_equal(data[key], getattr(model, key))
            assert data[key].dtype == np.float64
        assert list(data['channel_dof']) == ['1X+', '2X+', '3X-', '4Y+', '3X+']
        assert list(data['input_dof']) == ['1X+', '4Y+']
        assert list(data['channel_unit']) == model.channel_unit
        assert list(data['mode_frequency']) == [1.0, 50.0, 120.0]


def test_the_project_writes_it_and_journals_it(tmp_path):
    project = visualdynamics.Project('Plant')
    project.add('Modes', _shapes())
    project.add('Channels', _table())
    path = tmp_path / 'plant.npz'
    model = project.export_state_space('Modes', 'Channels', path,
                                       maximum_frequency=1000.0)
    assert path.exists() and model.dropped == 1
    assert any('export_state_space(' in line and "'Modes'" in line
               and 'maximum_frequency=1000.0' in line
               for line in project.journal)
    with pytest.raises(TypeError, match='not a shape set'):
        project.export_state_space('Channels', 'Modes', path)
    with pytest.raises(TypeError, match='not a channel table'):
        project.export_state_space('Modes', 'Modes', path)


def test_a_system_by_name_is_the_system_and_replays(tmp_path):
    project = visualdynamics.Project('Plant')
    project.add('Modes', _shapes())
    project.add('Channels', _table(unit=['', '', '', 'N', 'mm']))
    model = project.export_state_space('Modes', 'Channels', tmp_path / 'p.npz',
                                       unit_system='in-slinch-lbf-s (g)')
    assert model.channel_unit == ['lbf', 'g', 'in/s', 'N', 'mm']
    assert "unit_system='in-slinch-lbf-s (g)'" in project.journal[-1]
    with pytest.raises(ValueError, match="no unit system is named 'cgs'"):
        project.export_state_space('Modes', 'Channels', tmp_path / 'p.npz',
                                   unit_system='cgs')


# ---- the pane ----------------------------------------------------------------

def _open(window, pump, monkeypatch, shapes=None, table=None):
    """The pair added and selected, the act pressed; the series the
    preview drew are kept on the window as `drawn`."""
    from conftest import bar_acts, select_objects

    window.add_object('Modes', _shapes() if shapes is None else shapes)
    window.add_object('Channels', _table() if table is None else table)
    pump()
    select_objects(window, pump, 'Modes', 'Channels')
    assert 'Export State-Space Model' in bar_acts(window.data_pane) \
        or 'Export State-Space Model' in [act[1] for act in window.acts_for()]
    drawn = []
    real = window._render_series

    def spy(series, *args, **kwargs):
        drawn.append(series)
        return real(series, *args, **kwargs)

    monkeypatch.setattr(window, '_render_series', spy)
    window.drawn = drawn
    window.export_state_space_act()
    pump()
    return window.data_pane.state_space_panel


def test_the_act_is_offered_for_a_real_set_and_a_table_only(window, pump):
    window.add_object('Modes', _shapes())
    window.add_object('Channels', _table())
    complex_set = _shapes()
    complex_set.shape_matrix = complex_set.shape_matrix * (1 + 0.1j)
    window.add_object('Complex', complex_set)
    pump()

    def offered(*names):
        return 'Export State-Space Model' in [
            act[1] for act in window.acts_for(list(names))]

    assert offered('Modes', 'Channels')
    assert not offered('Modes') and not offered('Channels')
    assert not offered('Complex', 'Channels'), 'show only what applies'


def test_the_pane_opens_with_no_maximum_guessed_and_previews(window, pump,
                                                             monkeypatch):
    panel = _open(window, pump, monkeypatch)
    assert panel.isVisible() and panel.frequencies() == (0.0, 1.0)
    assert panel.maximum_box.text() == '—'
    reading = {key: value.text() for key, (_name, value) in panel._derived.items()}
    assert reading == {'modes': '4 of 4', 'lifted': '1', 'dropped': '0',
                       'inputs': '2', 'channels': '5',
                       'units': 'lbf, g, in/s, N, mm'}
    assert panel.note.text().startswith('Every mode kept, the highest at 5000 Hz')
    assert panel.apply_button.isEnabled()
    assert ('Modes at Channels: 4 modes, 2 inputs, 5 channels'
            in window.statusBar().currentMessage())
    # no response shares an input's DOF, so the first response is drawn
    # from each input, the model solid over the whole set dashed
    (model, every), = [s for s in window.drawn if len(s) == 2]
    assert model[0] == 'Modes model' and every[0] == 'Modes, every mode'
    assert every[1].synthesized and not getattr(model[1], 'synthesized', False)
    assert list(model[1].response_dof) == ['2X+', '2X+']
    assert list(model[1].reference_dof) == ['1X+', '4Y+']
    # in SI, as a stored FRF is: the model is the set with its rigid mode
    # lifted, and the two agree well above 1 Hz
    f = model[1].abscissa
    band = f > 20.0
    scale = np.abs(every[1].ordinate[:, band]).max()
    assert np.allclose(model[1].ordinate[:, band], every[1].ordinate[:, band],
                       rtol=1e-3, atol=1e-4 * scale)
    assert f[-1] == pytest.approx(1.25 * 5000.0)


def test_a_drive_point_is_drawn_alone(window, pump, monkeypatch):
    _open(window, pump, monkeypatch, table=_table(node=[1, 1, 3, 4, 3]))
    (model, _every), = [s for s in window.drawn if len(s) == 2]
    assert list(model[1].response_dof) == ['1X+']
    assert list(model[1].reference_dof) == ['1X+']


def test_the_maximum_drops_modes_and_export_writes_it(window, pump,
                                                      monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    panel = _open(window, pump, monkeypatch)
    panel.maximum_box.setValue(1000.0)
    pump()
    assert panel._derived['modes'][1].text() == '3 of 4'
    assert panel._derived['dropped'][1].text() == '1'
    assert not panel.note.isVisible()
    (model, _every), = [s for s in window.drawn[-1:]]
    panel.maximum_box.setValue(0.0)
    pump()
    assert panel._derived['modes'][1].text() == '4 of 4', \
        'a blank maximum keeps every mode, not none'
    panel.maximum_box.setValue(1000.0)
    pump()
    assert model[1].abscissa[-1] == pytest.approx(1250.0)
    panel.rigid_box.setValue(2.5)
    pump()
    asked = []

    def fake(parent, title, start, filters):
        asked.append((title, start, filters))
        return str(tmp_path / 'plant'), filters

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(fake))
    panel.apply_button.click()
    pump()
    assert asked == [('Export State-Space Model', 'Modes.npz',
                      'State-space model (*.npz)')]
    with np.load(tmp_path / 'plant.npz') as data:
        assert list(data['mode_frequency']) == [2.5, 50.0, 120.0]
    assert window.project.journal[-1] == (
        f"project.export_state_space('Modes', 'Channels', "
        f"{str(tmp_path / 'plant.npz')!r}, maximum_frequency=1000.0, "
        "rigid_frequency=2.5)")
    assert not panel.isVisible() and window._staging is None
    assert window.statusBar().currentMessage().startswith(
        'Wrote plant.npz: 3 modes, 2 inputs, 5 channels; 1 mode moved up to 2.5 Hz')


def test_a_blank_unit_takes_the_display_system_in_the_file_too(
        window, pump, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from visualdynamics.units import SYSTEMS

    window.unit_system = SYSTEMS['in-slinch-lbf-s (g)']
    panel = _open(window, pump, monkeypatch,
                  table=_table(unit=['', '', '', 'N', 'mm']))
    assert panel._derived['units'][1].text() == 'lbf, g, in/s, N, mm'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(
        lambda *args: (str(tmp_path / 'plant.npz'), '')))
    panel.apply_button.click()
    pump()
    with np.load(tmp_path / 'plant.npz') as data:
        assert list(data['channel_unit']) == ['lbf', 'g', 'in/s', 'N', 'mm']
    assert window.project.journal[-1].endswith(
        "unit_system='in-slinch-lbf-s (g)')")


def test_a_refusal_is_said_in_the_pane(window, pump, monkeypatch):
    panel = _open(window, pump, monkeypatch, shapes=_shapes(mass_unit=None))
    assert panel.note.text().startswith('Cannot export: the shape set has no '
                                        'mass unit')
    assert not panel.apply_button.isEnabled()


def test_the_act_again_or_another_selection_puts_it_away(window, pump,
                                                         monkeypatch):
    from conftest import select_objects

    panel = _open(window, pump, monkeypatch)
    window.export_state_space_act()
    pump()
    assert not panel.isVisible() and window._staging is None
    window.export_state_space_act()
    pump()
    assert panel.isVisible()
    select_objects(window, pump, 'Modes')
    assert not panel.isVisible() and window._staging is None
