"""Modal FRFs synthesized from a shape set: each mode's own response to a
unit modal force, on a test frame's lines.

Brandon, 2026-11-11: synthesize modal FRFs directly from mode shapes, on
the whole set or on modes picked in it, and check that the units work
for displacement and velocity FRFs as well as acceleration. Checked
against the direct inverse of the dynamic stiffness, the package's own
physical synthesis, and the rule the units must keep: a shape squared
is one over a mass, so the modal FRF over a mass is the physical one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

import visualdynamics
from visualdynamics.core.modal_frf import (
    above_nyquist,
    frequency_lines,
    frfs_at_channels,
    modal_matrix,
    synthesize_modal_frfs,
)
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.core.transform import to_modal, to_physical
from visualdynamics.units import SYSTEMS, UNKNOWN, si_factor

M = np.diag([2.0, 1.0])
K = np.array([[3e5, -1e5], [-1e5, 1e5]])
ZETA = np.array([0.02, 0.03])
POWER = {'displacement': 0, 'velocity': 1, 'acceleration': 2}
PHYSICAL = {'displacement': 'length', 'velocity': 'velocity',
            'acceleration': 'acceleration'}


def _model():
    lam, phi = eigh(K, M)                       # mass-normalized
    w = np.sqrt(lam)
    shapes = ShapeSet(w / (2 * np.pi), ZETA, ['1X+', '2X+'], phi.T,
                      mass_unit='kg')
    return shapes, phi, w


def _direct(f, phi, w, power):
    """(K - w^2 M + i w C)^-1 times (i w)^p, with the proportional
    damping the modes carry: the physical truth, from the matrices."""
    damping = M @ phi @ np.diag(2 * ZETA * w) @ phi.T @ M
    out = []
    for omega in 2 * np.pi * f:
        h = np.linalg.inv(K - omega ** 2 * M + 1j * omega * damping)
        out.append((1j * omega) ** power * h)
    return np.array(out)                         # lines x 2 x 2


# ---- the lines ----------------------------------------------------------------


def test_the_lines_are_a_test_frames():
    lines = frequency_lines(1024.0, 2048)
    assert lines[0] == 0.0 and lines[1] == 0.5 and lines[-1] == 512.0
    assert lines.size == 1025
    odd = frequency_lines(1000.0, 5)
    assert np.allclose(odd, [0.0, 200.0, 400.0]), 'an odd frame stops short'
    assert np.allclose(frequency_lines(100.0, 2), [0.0, 50.0]), 'the shortest frame'
    for rate, length in ((0.0, 8), (-1.0, 8), (np.nan, 8), (100.0, 1),
                         (100.0, 2.5)):
        with pytest.raises(ValueError):
            frequency_lines(rate, length)


# ---- the values ---------------------------------------------------------------


@pytest.mark.parametrize('quantity', ['displacement', 'velocity',
                                      'acceleration'])
def test_through_the_shapes_they_are_the_physical_frfs(quantity):
    """phi H_r phi summed over the modes is the direct inverse of the
    dynamic stiffness, for each response quantity."""
    shapes, phi, w = _model()
    modal = synthesize_modal_frfs(shapes, 2048.0, 1024, quantity)
    f = modal.abscissa[1:]
    h = modal.ordinate[:, 1:]                    # modes x lines
    physical = np.einsum('jr,rl,kr->ljk', phi, h, phi)
    direct = _direct(f, phi, w, POWER[quantity])
    assert np.allclose(physical, direct, rtol=1e-10, atol=0)
    sums = shapes.synthesize_frf(f, ['1X+'], ['2X+'], power=POWER[quantity])
    assert np.allclose(physical[:, 0, 1], sums[0], rtol=1e-12, atol=0)


def test_each_record_is_its_mode_against_itself():
    shapes, _, w = _model()
    modal = synthesize_modal_frfs(shapes, 2048.0, 1024, modes=[1],
                                  label='Modes')
    assert modal.response_dof == ['M2'] and modal.reference_dof == ['M2']
    assert modal.comment[0].startswith(
        'Modes: modal acceleration per modal force, mode 2 at')
    omega = 2 * np.pi * modal.abscissa
    expected = -omega ** 2 / (w[1] ** 2 - omega ** 2
                              + 2j * ZETA[1] * w[1] * omega)
    assert np.allclose(modal.ordinate[0], expected, rtol=1e-12, atol=0)


def test_modes_are_picked_sorted_once_and_checked():
    shapes, _, _ = _model()
    both = synthesize_modal_frfs(shapes, 2048.0, 64, modes=[1, 0, 1])
    assert both.response_dof == ['M1', 'M2']
    with pytest.raises(ValueError, match='mode 3 is not'):
        synthesize_modal_frfs(shapes, 2048.0, 64, modes=[2])
    with pytest.raises(ValueError, match='no modes'):
        synthesize_modal_frfs(shapes, 2048.0, 64, modes=[])
    with pytest.raises(ValueError, match='one of'):
        synthesize_modal_frfs(shapes, 2048.0, 64, quantity='jerk')


def test_a_rigid_mode_at_zero_hertz():
    """Its accelerance at 0 Hz is 1/m_r; its displacement and velocity
    there are unbounded, and nan, not a number that looks finite."""
    shapes = ShapeSet([0.0, 40.0], [0.0, 0.02], ['1X+', '2X+'],
                      np.array([[0.5, 0.5], [0.5, -0.5]]),
                      modal_mass=[2.0, 1.0], mass_unit='kg')
    for quantity, at_zero in (('acceleration', 0.5), ('velocity', np.nan),
                              ('displacement', np.nan)):
        modal = synthesize_modal_frfs(shapes, 512.0, 512, quantity)
        assert modal.abscissa[0] == 0.0
        if np.isnan(at_zero):
            assert np.isnan(modal.ordinate[0, 0])
        else:
            assert modal.ordinate[0, 0] == pytest.approx(at_zero)
        omega = 2 * np.pi * modal.abscissa[1]
        assert modal.ordinate[0, 1] == pytest.approx(
            (1j * omega) ** POWER[quantity] / (2.0 * -omega ** 2))


def test_a_complex_set_is_refused():
    shapes = ShapeSet([10.0], [0.02], ['1X+'], np.array([[1.0 + 0.5j]]))
    with pytest.raises(ValueError, match='complex'):
        synthesize_modal_frfs(shapes, 100.0, 16)


def test_the_modes_above_nyquist_are_counted():
    shapes, _, _ = _model()
    highest = float(shapes.frequency.max())
    assert above_nyquist(shapes, None, 2 * highest + 1.0) == 0
    assert above_nyquist(shapes, None, 2 * highest - 1.0) == 1
    assert above_nyquist(shapes, None, 2 * highest) == 0, 'on the last line'
    assert above_nyquist(shapes, [0], 2 * highest - 1.0) == 0


# ---- the units ------------------------------------------------------------------


@pytest.mark.parametrize('quantity', ['displacement', 'velocity',
                                      'acceleration'])
def test_the_dimension_is_the_modal_response_per_modal_force(quantity):
    shapes, _, _ = _model()
    modal = synthesize_modal_frfs(shapes, 2048.0, 64, quantity)
    assert modal.ordinate_dim == [
        f'modal_{PHYSICAL[quantity]}/modal_force'] * 2
    assert modal.units_defined
    bare = ShapeSet(shapes.frequency, shapes.damping, shapes.coordinate,
                    shapes.shape_matrix)
    undefined = synthesize_modal_frfs(bare, 2048.0, 64, quantity)
    assert undefined.ordinate_dim == [UNKNOWN] * 2
    assert undefined.dimension_hint == [
        f'modal_{PHYSICAL[quantity]}/modal_force'] * 2


@pytest.mark.parametrize('system', sorted(SYSTEMS))
@pytest.mark.parametrize('quantity', ['displacement', 'velocity',
                                      'acceleration'])
def test_the_units_keep_the_physics_in_every_system(system, quantity):
    """A mass-normalized shape is in one over the square root of a mass,
    so a modal FRF divided by a mass is a physical FRF, in any system's
    units. The modal force was once force times root mass, and in
    in-slinch-lbf-s this read 175 times over."""
    from visualdynamics import Frf

    units = SYSTEMS[system]
    response, reference = Frf.units_for(
        f'modal_{PHYSICAL[quantity]}/modal_force', units)
    modal = si_factor(response) / si_factor(reference)
    physical_response, force = Frf.units_for(f'{PHYSICAL[quantity]}/force',
                                             units.coherent)
    physical = si_factor(physical_response) / si_factor(force)
    mass = si_factor(units.coherent.unit('mass'))
    assert modal / mass == pytest.approx(physical, rel=1e-12)


@pytest.mark.parametrize('system', ['m-kg-N-s', 'in-slinch-lbf-s',
                                    'ft-slug-lbf-s'])
def test_in_a_coherent_system_they_read_as_in_si(system):
    """s², s and 1: a coherent system's time unit is the second, so the
    numbers do not change with the system."""
    from visualdynamics import Frf

    for quantity in POWER:
        response, reference = Frf.units_for(
            f'modal_{PHYSICAL[quantity]}/modal_force', SYSTEMS[system])
        assert si_factor(response) / si_factor(reference) == pytest.approx(
            1.0, rel=1e-12), quantity


# ---- the project verb ---------------------------------------------------------


def _project():
    shapes, _, _ = _model()
    project = visualdynamics.Project()
    project.add('Modes', shapes)
    return project, shapes


def test_the_verb_makes_an_object_beside_the_set_and_journals():
    project, shapes = _project()
    assert 'synthesize_modal_frfs' in dict(project.selection_verbs('Modes'))
    made = project.synthesize_modal_frfs('Modes', sample_rate=1024.0,
                                         frame_length=2048, modes=[1])
    assert made == 'Modes Modal FRFs'
    modal = project[made]
    assert modal.response_dof == ['M2']
    assert np.array_equal(modal.ordinate, synthesize_modal_frfs(
        shapes, 1024.0, 2048, modes=[1]).ordinate)
    assert set(project.object_group_of(made)) == {'Modes', made}
    assert ("project.synthesize_modal_frfs('Modes', sample_rate=1024.0, "
            "frame_length=2048, modes=[1])") in project.journal
    velocity = project.synthesize_modal_frfs(
        'Modes', sample_rate=1024.0, frame_length=2048, quantity='velocity')
    assert velocity == 'Modes Modal FRFs (2)'
    assert project[velocity].ordinate_dim == ['modal_velocity/modal_force'] * 2


def test_it_is_not_offered_on_a_complex_set():
    project, _ = _project()
    project.add('Complex', ShapeSet([10.0], [0.02], ['1X+'],
                                    np.array([[1.0 + 0.5j]])))
    assert 'synthesize_modal_frfs' not in dict(
        project.selection_verbs('Complex'))
    project.add('Lines', visualdynamics.Frf(
        [1.0, 2.0], np.ones((1, 2)), response_dof='1X+',
        reference_dof='1X+', ordinate_dim='acceleration/force'))
    with pytest.raises(TypeError, match='not a shape set'):
        project.synthesize_modal_frfs('Lines', sample_rate=8.0,
                                      frame_length=8)


def test_a_new_damping_marks_them_stale_and_a_new_shape_does_not():
    """The modal FRFs read the frequencies, damping and modal masses and
    never the shapes, so only a change to those is a change to them."""
    project, shapes = _project()
    made = project.synthesize_modal_frfs('Modes', sample_rate=1024.0,
                                         frame_length=256)
    reshaped = ShapeSet(shapes.frequency, shapes.damping, shapes.coordinate,
                        -shapes.shape_matrix, mass_unit='kg')
    project['Modes'] = reshaped
    assert project.stale() == {}
    damped = ShapeSet(shapes.frequency, 2 * shapes.damping, shapes.coordinate,
                      shapes.shape_matrix, mass_unit='kg')
    project['Modes'] = damped
    assert made in project.stale()
    project.refresh(made)
    assert project.stale() == {}
    assert np.array_equal(project[made].ordinate, synthesize_modal_frfs(
        damped, 1024.0, 256).ordinate)


def test_they_are_saved_and_read_back_with_their_units(tmp_path):
    project, _ = _project()
    made = project.synthesize_modal_frfs('Modes', sample_rate=1024.0,
                                         frame_length=256,
                                         quantity='displacement')
    path = project.save(tmp_path / 'modal.vdyn')
    again = visualdynamics.Project.open(path)
    assert again[made].ordinate_dim == ['modal_length/modal_force'] * 2
    assert np.array_equal(again[made].ordinate, project[made].ordinate,
                          equal_nan=True)
    assert again.stale() == {}


# ---- the pane ---------------------------------------------------------------


def _offered(window, label):
    """The act by this label on whichever bar is up: a set selected
    alone shows its mode table, with the plot and the 3-D view down."""
    bars = (window.data_pane.toolbar, window.scene.toolbar, window.table_bar)
    found = [a for b in bars if b.isVisibleTo(window) for a in b.actions()
             if a.isVisible() and a.text() == label]
    return found[0] if len(found) == 1 else None


def _populate(window, pump, **sets):
    shapes, _, _ = _model()
    window.add_object('Shapes', shapes)
    for name, other in sets.items():
        window.add_object(name, other)
    pump()


def test_a_set_offers_the_act_and_the_pane_previews(window, pump):
    from conftest import select_objects

    _populate(window, pump)
    select_objects(window, pump, 'Shapes')
    _offered(window, 'Synthesize Modal FRFs').trigger()
    pump()
    panel = window.data_pane.modal_frf_panel
    assert panel.isVisible()
    assert window.data_pane.flat_only, 'the preview is a plot; the 3-D view stands down'
    derived = {key: panel._derived[key][1].text() for key in panel._derived}
    assert derived == {'modes': '2 of 2', 'spacing': '1 Hz',
                       'lines': '4,097', 'nyquist': '4096 Hz',
                       'highest': derived['highest']}
    assert derived['highest'].endswith(' Hz')
    assert ('Shapes: modal acceleration FRFs of 2 modes, 4,097 lines to '
            '4096 Hz') in window.statusBar().currentMessage()
    panel.quantity_box.setCurrentIndex(panel.quantity_box.findData('velocity'))
    pump()
    assert 'modal velocity FRFs' in window.statusBar().currentMessage()
    panel.frame_box.setValue(2048)
    pump()
    assert panel._derived['spacing'][1].text() == '4 Hz'
    panel.apply_button.click()
    pump()
    made = window.objects['Shapes Modal FRFs']
    assert made.response_dof == ['M1', 'M2']
    assert made.ordinate_dim == ['modal_velocity/modal_force'] * 2
    assert made.abscissa[1] == 4.0
    assert window._item_for_object('Shapes Modal FRFs').isSelected(), \
        'done: the FRFs are shown'
    assert not panel.isVisible()
    assert window.statusBar().currentMessage().startswith(
        'Shapes Modal FRFs: modal velocity FRFs of 2 modes from Shapes')
    assert any("synthesize_modal_frfs('Shapes', sample_rate=8192.0, "
               "frame_length=2048, quantity='velocity')" in line
               for line in window.project.journal)


def test_modes_picked_in_the_grid_are_the_modes_synthesized(window, pump):
    _populate(window, pump)
    item = window._item_for_object('Shapes')
    window.tree.clearSelection()
    item.setExpanded(True)
    pump()
    # the row first, then the mode in its grid: selecting the row
    # releases its grid picks, as a click on it does
    window.tree.setCurrentItem(item)
    pump()
    window.record_grids['Shapes'].select_records([1])
    window.record_grids['Shapes'].selection_changed.emit()
    pump()
    assert window._modal_frf_candidates() == ('Shapes', (1,))
    _offered(window, 'Synthesize Modal FRFs').trigger()
    pump()
    panel = window.data_pane.modal_frf_panel
    assert panel._derived['modes'][1].text() == '1 of 2'
    panel.apply_button.click()
    pump()
    assert window.objects['Shapes Modal FRFs'].response_dof == ['M2']
    assert any("modes=[1]" in line and 'synthesize_modal_frfs' in line
               for line in window.project.journal)


def test_the_settings_are_kept_and_the_pane_stands_down(window, pump):
    from conftest import select_objects

    bare = ShapeSet([5.0], [0.02], ['1X+'], np.array([[1.0]]))
    _populate(window, pump, Bare=bare)
    select_objects(window, pump, 'Shapes')
    window.synthesize_modal_frfs_act()
    pump()
    panel = window.data_pane.modal_frf_panel
    panel.rate_box.setValue(100.0)
    pump()
    assert 'above the Nyquist frequency, kept' in panel.note.text()
    window.synthesize_modal_frfs_act()
    pump()
    assert not panel.isVisible() and window._modal_frfing is None, \
        'the act again puts it away'
    select_objects(window, pump, 'Bare')
    window.synthesize_modal_frfs_act()
    pump()
    assert panel.isVisible()
    assert panel.rate_box.value() == 100.0, 'the settings are kept'
    assert 'no mass unit' in panel.note.text()
    select_objects(window, pump, 'Shapes')
    assert not panel.isVisible() and window._modal_frfing is None


def test_a_complex_set_is_not_offered_the_act(window, pump):
    from conftest import select_objects

    _populate(window, pump, Complex=ShapeSet([10.0], [0.02], ['1X+'],
                                             np.array([[1.0 + 0.5j]])))
    select_objects(window, pump, 'Shapes')
    assert _offered(window, 'Synthesize Modal FRFs') is not None
    select_objects(window, pump, 'Complex')
    assert _offered(window, 'Synthesize Modal FRFs') is None


# ---- back to physical FRFs, at a channel table's channels ---------------------


@pytest.mark.parametrize('synthesized', ['displacement', 'velocity',
                                         'acceleration'])
@pytest.mark.parametrize('measured', ['length', 'velocity', 'acceleration'])
def test_at_the_channels_they_are_the_sets_own_synthesis(synthesized, measured):
    """Whatever the modal FRFs answer in, each row answers in what its
    channel measures, and a channel read negative is the negative."""
    shapes, _, _ = _model()
    modal = synthesize_modal_frfs(shapes, 1024.0, 512, synthesized)
    out, report = frfs_at_channels(modal, shapes, ['1X+'],
                                   [('2X-', measured), ('1X+', measured)])
    assert out.response_dof == ['2X-', '1X+']
    assert out.reference_dof == ['1X+', '1X+']
    assert out.ordinate_dim == [f'{measured}/force'] * 2
    assert report.dropped == [] and report.shared[PHYSICAL[synthesized]] == [
        'M1', 'M2']
    power = {'length': 0, 'velocity': 1, 'acceleration': 2}[measured]
    f = modal.abscissa[1:]
    exact = shapes.synthesize_frf(f, ['2X-', '1X+'], ['1X+', '1X+'],
                                  power=power)
    assert np.allclose(out.ordinate[:, 1:], exact, rtol=1e-12, atol=0)
    assert np.allclose(exact[0], -shapes.synthesize_frf(
        f, ['2X+'], ['1X+'], power=power)[0], rtol=1e-14, atol=0)
    lowered = power < POWER[synthesized]
    assert np.isnan(out.ordinate[0, 0]) == lowered, (
        'turned down to a lower quantity, 0 Hz divides by zero: nan, '
        'never a number that looks right')


def test_picked_records_sum_those_modes_alone():
    shapes, _, _ = _model()
    modal = synthesize_modal_frfs(shapes, 1024.0, 512)
    out, report = frfs_at_channels(modal, shapes, ['1X+'],
                                   [('2X+', 'acceleration')], records=[1])
    assert report.shared['acceleration'] == ['M2']
    f = modal.abscissa
    assert np.allclose(out.ordinate[0], shapes.synthesize_frf(
        f, ['2X+'], ['1X+'], modes=[1], power=2)[0], rtol=1e-12, atol=0)


def test_a_transformed_frf_is_the_whole_matrix_and_expands_back():
    """A measured FRF transformed to modal coordinates holds every mode
    against every mode; at the channels it is the FRF again."""
    shapes, _, _ = _model()
    f = np.arange(1.0, 200.0)
    dofs = ['1X+', '2X+']
    physical = visualdynamics.Frf(
        f, shapes.synthesize_frf(f, [r for r in dofs for _c in dofs],
                                 [c for _r in dofs for c in dofs], power=2),
        response_dof=[r for r in dofs for _c in dofs],
        reference_dof=[c for _r in dofs for c in dofs],
        ordinate_dim=['acceleration/force'] * 4)
    modal, _ = to_modal(physical, shapes)
    modes, matrix, quantity, known = modal_matrix(modal)
    assert modes == [0, 1] and quantity == 'acceleration' and known
    assert np.abs(matrix[:, 0, 1]).max() < 1e-9 * np.abs(matrix[:, 0, 0]).max(), \
        'proportional damping: no modal cross term'
    out, _ = frfs_at_channels(modal, shapes, ['2X+'], [('1X+', 'acceleration')])
    assert np.allclose(out.ordinate[0], physical.ordinate[1], rtol=1e-9, atol=0)
    partial = visualdynamics.Frf(
        f, modal.ordinate[:3], response_dof=modal.response_dof[:3],
        reference_dof=modal.reference_dof[:3], ordinate_dim=modal.ordinate_dim[:3])
    with pytest.raises(ValueError, match='no FRF between M2 and M2'):
        modal_matrix(partial)


def test_what_cannot_be_expanded_is_refused_or_reported():
    shapes, _, _ = _model()
    modal = synthesize_modal_frfs(shapes, 1024.0, 64)
    out, report = frfs_at_channels(modal, shapes, ['1X+', '9Z+'],
                                   [('2X+', 'acceleration'), ('8Y+', 'velocity')])
    assert out.response_dof == ['2X+'] and out.reference_dof == ['1X+']
    assert report.dropped == ['9Z+', '8Y+']
    with pytest.raises(ValueError, match='no reference'):
        frfs_at_channels(modal, shapes, ['9Z+'], [('2X+', 'acceleration')])
    with pytest.raises(ValueError, match='no response'):
        frfs_at_channels(modal, shapes, ['1X+'], [('9Z+', 'acceleration')])
    with pytest.raises(ValueError, match='not a motion'):
        frfs_at_channels(modal, shapes, ['1X+'], [('2X+', 'temperature')])
    physical, _ = frfs_at_channels(modal, shapes, ['1X+'],
                                   [('2X+', 'acceleration')])
    with pytest.raises(ValueError, match='not a modal FRF'):
        modal_matrix(physical)
    half = visualdynamics.Frf(
        modal.abscissa, modal.ordinate[:1], response_dof=['M1'],
        reference_dof=['1X+'], ordinate_dim=['modal_acceleration/force'])
    with pytest.raises(ValueError, match='not a modal FRF'):
        modal_matrix(half), 'a reference kept physical'
    per_motion = visualdynamics.Frf(
        modal.abscissa, modal.ordinate[:1], response_dof=['M1'],
        reference_dof=['M1'],
        ordinate_dim=['modal_acceleration/modal_velocity'])
    with pytest.raises(ValueError, match='not a motion per modal force'):
        modal_matrix(per_motion)
    both = visualdynamics.Frf(
        modal.abscissa, np.vstack([modal.ordinate, modal.ordinate]),
        response_dof=['M1', 'M2'] * 2, reference_dof=['M1', 'M2'] * 2,
        ordinate_dim=['modal_acceleration/modal_force'] * 2
        + ['modal_velocity/modal_force'] * 2)
    with pytest.raises(ValueError, match='one response at a time'):
        modal_matrix(both)
    small = ShapeSet([10.0], [0.02], ['1X+'], np.array([[1.0]]), mass_unit='kg')
    with pytest.raises(ValueError, match='reach M2'):
        frfs_at_channels(modal, small, ['1X+'], [('1X+', 'acceleration')])


def test_units_follow_the_modal_frfs():
    shapes, _, _ = _model()
    bare = ShapeSet(shapes.frequency, shapes.damping, shapes.coordinate,
                    shapes.shape_matrix)
    modal = synthesize_modal_frfs(bare, 1024.0, 64, 'velocity')
    out, _ = frfs_at_channels(modal, bare, ['1X+'], [('2X+', 'length')])
    assert out.ordinate_dim == [UNKNOWN]
    assert out.dimension_hint == ['length/force']


def test_each_mode_against_itself_expands_through_the_set_too():
    """The diagonal modal FRFs Synthesize Modal FRFs writes are a whole
    matrix with zero cross terms, so the set takes them to every DOF
    pair."""
    shapes, _, _ = _model()
    modal = synthesize_modal_frfs(shapes, 1024.0, 512, 'velocity')
    out, _ = to_physical(modal, shapes)
    pairs = list(zip(out.response_dof, out.reference_dof))
    assert pairs == [('1X+', '1X+'), ('1X+', '2X+'), ('2X+', '1X+'),
                     ('2X+', '2X+')]
    f = modal.abscissa
    for i, (row, column) in enumerate(pairs):
        assert np.allclose(out.ordinate[i], shapes.synthesize_frf(
            f, [row], [column], power=1)[0], rtol=1e-12, atol=0,
            equal_nan=True)


# ---- the verb, with a channel table -------------------------------------------


def _table(**changes):
    columns = {'channel': [1, 2, 3, 4],
               'node': ['1', '2', '1', '2'],
               'direction': ['X+', 'X-', 'X+', 'X+'],
               'role': ['reference', 'response', 'monitor', 'monitor'],
               'unit': ['N', 'g', 'in/s', 'degC'],
               'channel_type': ['force', 'acceleration', 'velocity',
                                'temperature']}
    columns.update(changes)
    return visualdynamics.ChannelTable(columns)


def test_the_verb_expands_modal_frfs_at_a_tables_channels():
    project, shapes = _project()
    project.add('Channels', _table())
    modal = project.synthesize_modal_frfs('Modes', sample_rate=1024.0,
                                          frame_length=512)
    made = project.expand(modal, 'Channels')
    assert made == 'Modes Modal FRFs @ Channels'
    frfs = project[made]
    assert frfs.response_dof == ['2X-', '1X+']
    assert frfs.reference_dof == ['1X+', '1X+']
    assert frfs.ordinate_dim == ['acceleration/force', 'velocity/force']
    report = frfs.transform_report
    assert report.basis == 'Modes' and report.projection == {}
    assert report.skipped == {'temperature': 1}
    f = frfs.abscissa[1:]
    assert np.allclose(frfs.ordinate[1, 1:], shapes.synthesize_frf(
        f, ['1X+'], ['1X+'], power=1)[0], rtol=1e-12, atol=0)
    assert made in project.object_group_of('Channels')
    assert "project.expand('Modes Modal FRFs', 'Channels')" in project.journal
    assert project.stale() == {}
    damped = ShapeSet(shapes.frequency, 2 * shapes.damping, shapes.coordinate,
                      shapes.shape_matrix, mass_unit='kg')
    project['Modes'] = damped
    assert set(project.stale()) == {modal}
    project.refresh(modal)
    assert made in project.stale(), 'its modal FRFs changed under it'
    project.refresh(made)
    assert np.allclose(project[made].ordinate[1, 1:], damped.synthesize_frf(
        f, ['1X+'], ['1X+'], power=1)[0], rtol=1e-12, atol=0)
    project['Channels'] = _table(direction=['X+', 'X+', 'X+', 'X+'])
    assert made in project.stale(), 'its channels changed'


def test_the_verb_refuses_what_it_cannot_expand():
    project, _ = _project()
    project.add('Channels', _table())
    project.add('No drive', _table(role=['response'] * 4))
    modal = project.synthesize_modal_frfs('Modes', sample_rate=1024.0,
                                          frame_length=64)
    with pytest.raises(ValueError, match='no reference channel'):
        project.expand(modal, 'No drive')
    project.add('Lines', visualdynamics.Frf(
        [1.0, 2.0], np.ones((1, 2)), response_dof='1X+',
        reference_dof='1X+', ordinate_dim='acceleration/force'))
    with pytest.raises(ValueError, match='nothing says which shape set'):
        project.expand('Lines', 'Channels')
    assert 'expand' in dict(project.selection_verbs(modal, 'Channels'))
    orphan = project[modal]
    project.add('Orphan', visualdynamics.Frf(
        orphan.abscissa, orphan.ordinate, response_dof=orphan.response_dof,
        reference_dof=orphan.reference_dof, ordinate_dim=orphan.ordinate_dim))
    assert 'expand' not in dict(project.selection_verbs('Orphan', 'Channels')), \
        'modal FRFs of no known set are not offered it'
    assert 'expand' not in dict(project.selection_verbs(modal, 'No drive')), \
        'nor a table with no drive'


def test_at_a_test_geometry_the_shapes_are_taken_at_its_channels():
    """FE modes on their own geometry, a channel table on the test's:
    the FRFs between the test's channels, from the FE modes projected
    there as Project onto Basis DOFs projects them."""
    from test_correlate import _channels, _fem, _test_model

    fem_geometry, fem_shapes = _fem()
    fem_shapes.define_units('kg')
    test_geometry, _shapes = _test_model()
    project = visualdynamics.Project()
    project.add('FEM Geometry', fem_geometry)
    project.add('FEM Shapes', fem_shapes)
    project.add('Test Geometry', test_geometry)
    project.add('Channels', _channels())
    project.link('FEM Geometry', 'FEM Shapes')
    project.link('Test Geometry', 'Channels')
    modal = project.synthesize_modal_frfs('FEM Shapes', sample_rate=128.0,
                                          frame_length=128)
    made = project.expand(modal, 'Channels')
    frfs = project[made]
    assert frfs.response_dof == ['101X+'] and frfs.reference_dof == ['102X-']
    assert frfs.transform_report.projection['matched'] == 2
    projected = project[project.project_onto_basis('FEM Shapes',
                                                   onto='Channels')]
    f = frfs.abscissa
    assert np.allclose(frfs.ordinate[0], projected.synthesize_frf(
        f, ['101X+'], ['102X-'], power=2)[0], rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match='within the match tolerance'):
        project.expand(modal, 'Channels', tolerance=1e-6)
    looser = project.expand(modal, 'Channels', tolerance=0.05)
    assert project.provenance[looser]['params']['tolerance'] == 0.05
    assert any("tolerance=0.05" in line for line in project.journal)


def test_a_channel_too_far_from_the_model_is_dropped_and_said():
    from test_correlate import _fem, _test_model

    fem_geometry, fem_shapes = _fem()
    fem_shapes.define_units('kg')
    test_geometry, _shapes = _test_model(extra_node=True)
    project = visualdynamics.Project()
    project.add('FEM Geometry', fem_geometry)
    project.add('FEM Shapes', fem_shapes)
    project.add('Test Geometry', test_geometry)
    project.add('Channels', visualdynamics.ChannelTable({
        'channel': [1, 2, 3], 'node': ['101', '102', '103'],
        'direction': ['X+', 'X-', 'X+'],
        'role': ['response', 'reference', 'response'],
        'unit': ['g', 'N', 'g'],
        'channel_type': ['acceleration', 'force', 'acceleration']}))
    project.link('FEM Geometry', 'FEM Shapes')
    project.link('Test Geometry', 'Channels')
    modal = project.synthesize_modal_frfs('FEM Shapes', sample_rate=128.0,
                                          frame_length=128)
    frfs = project[project.expand(modal, 'Channels')]
    assert frfs.response_dof == ['101X+']
    assert frfs.transform_report.dropped == ['103X+']
    assert '1 not in the shapes' in frfs.transform_report.describe()


# ---- the pane, for modal FRFs and a channel table -----------------------------


def _expand_window(window, pump):
    from conftest import select_objects

    shapes, _, _ = _model()
    window.add_object('Modes', shapes)
    window.add_object('Channels', _table())
    pump()
    modal = window.project.synthesize_modal_frfs('Modes', sample_rate=1024.0,
                                                 frame_length=512)
    window.show_object(modal)
    pump()
    select_objects(window, pump, modal, 'Channels')
    return modal


def test_modal_frfs_and_a_table_open_the_pane_and_expand(window, pump):
    modal = _expand_window(window, pump)
    _offered(window, 'Expand to Physical Responses').trigger()
    pump()
    panel = window.data_pane.channel_frf_panel
    assert panel.isVisible() and window.data_pane.flat_only
    derived = {key: panel._derived[key][1].text() for key in panel._derived}
    assert derived == {'modes': '2', 'references': '1', 'responses': '2',
                       'pairs': '2', 'shapes': 'Modes, at its own DOFs',
                       'dropped': 'none'}
    assert not panel.tolerance_box.isEnabled(), 'no projection, no tolerance'
    assert 'Not expanded: 1 temperature channel' in panel.note.text()
    assert ('The velocity FRFs have no value at 0 Hz: from modal acceleration '
            'FRFs that divides by zero frequency') in panel.note.text()
    assert ('Modes Modal FRFs at Channels: 2 FRFs from 2 modes | 1 record'
            in window.statusBar().currentMessage()), 'the drive point alone'
    panel.apply_button.click()
    pump()
    made = f'{modal} @ Channels'
    assert window.objects[made].response_dof == ['2X-', '1X+']
    assert window._item_for_object(made).isSelected()
    assert not panel.isVisible()
    assert any(line.endswith(f"expand('{modal}', 'Channels')")
               for line in window.project.journal), 'no defaults written'


def test_picked_records_expand_those_modes(window, pump):
    from conftest import select_objects

    modal = _expand_window(window, pump)
    window._item_for_object(modal).setExpanded(True)
    pump()
    # the record picked in the FRFs' grid, then the table Cmd-clicked
    # beside it: selecting the owner row would release the pick
    select_objects(window, pump, modal)
    window.record_grids[modal].select_records([1])
    window.record_grids[modal].selection_changed.emit()
    pump()
    # a Cmd-click: the table added to the selection and made current
    from PySide6.QtCore import QItemSelectionModel
    window.tree.setCurrentItem(window._item_for_object('Channels'), 0,
                               QItemSelectionModel.SelectionFlag.Select)
    pump()
    assert window._channel_frf_candidates() == (modal, 'Channels', (1,))
    _offered(window, 'Expand to Physical Responses').trigger()
    pump()
    panel = window.data_pane.channel_frf_panel
    assert panel._derived['modes'][1].text() == '1'
    panel.apply_button.click()
    pump()
    assert any('records=[1]' in line and 'expand' in line
               for line in window.project.journal)
    select_objects(window, pump, 'Channels')
    assert window._channel_expanding is None


def test_on_another_geometry_the_tolerance_applies(window, pump):
    from conftest import select_objects
    from test_correlate import _channels, _fem, _test_model

    fem_geometry, fem_shapes = _fem()
    fem_shapes.define_units('kg')
    test_geometry, _shapes = _test_model()
    for name, obj in (('FEM Geometry', fem_geometry), ('FEM Shapes', fem_shapes),
                      ('Test Geometry', test_geometry), ('Channels', _channels())):
        window.add_object(name, obj)
    window.project.link('FEM Geometry', 'FEM Shapes')
    window.project.link('Test Geometry', 'Channels')
    modal = window.project.synthesize_modal_frfs('FEM Shapes', sample_rate=128.0,
                                                 frame_length=128)
    window.show_object(modal)
    pump()
    select_objects(window, pump, modal, 'Channels')
    window.expand_act()
    pump()
    panel = window.data_pane.channel_frf_panel
    assert panel.tolerance_box.isEnabled()
    assert panel._derived['shapes'][1].text().startswith(
        'FEM Shapes, projected: 2 of 2 nodes, worst')
    panel.tolerance_box.setValue(0.1)
    pump()
    assert panel.note.text().startswith('Cannot expand:')
    assert not panel.apply_button.isEnabled()
    assert panel.tolerance_box.isEnabled(), 'still there to loosen'
    panel.tolerance_box.setValue(5.0)
    pump()
    assert panel.apply_button.isEnabled()
    panel.apply_button.click()
    pump()
    assert any('tolerance=0.05' in line and 'expand' in line
               for line in window.project.journal)
    select_objects(window, pump, modal, 'Channels')
    window.expand_act()
    pump()
    assert panel.isVisible()
    window.expand_act()
    pump()
    assert not panel.isVisible(), 'the act again puts it away'
