"""The transform in the window: an act on the bar that makes the object.

A transform has no settings, so it is an act and not a reading
(Brandon, 2026-09-04): with one time history and one shape set
selected together, the plot's bar offers Transform to Modal Responses
— or Expand to Physical Responses, when the record reads as modal —
and pressing it runs the project verb, shows what it made and puts
the account on the status line. No toggle, no pane, no preview, and
nothing in the right-click menu: every act lives on the bar.
"""

from __future__ import annotations

import numpy as np
from conftest import menu_entries
from conftest import select_objects as _select

import visualdynamics
from visualdynamics.core.rigid import MassProperties, rigid_body_shapes

T = np.arange(1024) / 1024.0


def _fixtures():
    rng = np.random.default_rng(1)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    q = np.zeros((6, len(T)))
    q[0] = np.sin(2 * np.pi * 5 * T)
    q[5] = 0.3 * np.cos(2 * np.pi * 3 * T)
    run = visualdynamics.TimeHistory(T, rigid.shape_matrix.T @ q,
                                     response_dof=list(rigid.coordinate),
                                     ordinate_dim='acceleration')
    return geometry, rigid, run


def _populate(window, pump):
    geometry, rigid, run = _fixtures()
    window.add_object('Plate', geometry)
    window.add_object('Modes', rigid)
    window.project.link('Plate', 'Modes')
    window.add_object('Run', run)
    pump()
    return rigid


def _bar_acts(window):
    return [label for _v, label, *_rest in window.acts_for()]


def test_the_bar_offers_the_transform_to_the_pair(window, pump, qt_app):
    _populate(window, pump)
    _select(window, pump, 'Run', 'Modes')
    assert 'Transform to Modal Responses' in _bar_acts(window)
    for name in ('Run', 'Modes'):
        entries = [e.replace('&', '') for e in menu_entries(
            window, qt_app, window._item_for_object(name))]
        assert not any('Transform' in e for e in entries), \
            'the right-click carries no acts'
    _select(window, pump, 'Run', 'Modes', 'Plate')
    assert 'Transform to Modal Responses' in _bar_acts(window), \
        'the geometry may come along'
    _select(window, pump, 'Run')
    assert not any('Transform' in a for a in _bar_acts(window))
    _select(window, pump, 'Modes')
    assert _bar_acts(window) == []


def test_the_entry_makes_the_object_and_shows_it(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Run', 'Modes')
    window.transform_selection()
    pump()
    assert 'Run Modal Responses' in window.project
    made = window.project['Run Modal Responses']
    assert made.response_dof == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    assert made.ordinate_dim[3] == 'angular_acceleration'
    assert window.project.journal[-1] == \
        "project.transform('Run', 'Modes')"
    assert window.statusBar().currentMessage().startswith(
        'Run Modal Responses: 15 DOFs shared')
    assert 'through Modes' in window.statusBar().currentMessage()
    assert window.current_object() is made, 'the act shows what it made'
    assert not hasattr(window.data_pane, 'transform_action'), \
        'no toggle and no pane: a transform has no settings'


def test_a_refusal_lands_on_the_status_line(window, pump):
    _geometry, rigid, run = _fixtures()
    zs = [i for i, dof in enumerate(rigid.coordinate) if dof.endswith('Z+')]
    flat = visualdynamics.TimeHistory(
        T, run.ordinate[zs], response_dof=[rigid.coordinate[i] for i in zs],
        ordinate_dim='acceleration')
    window.add_object('Modes', rigid)
    window.add_object('Zs', flat)
    _select(window, pump, 'Zs', 'Modes')
    window.transform_selection()
    pump()
    assert 'resolve only 3 of 6 modes' in window.statusBar().currentMessage()
    assert 'Zs Modal Responses' not in window.project


def test_the_entry_turns_round_for_a_modal_record(window, pump):
    _populate(window, pump)
    window.project.transform('Run', 'Modes')
    window.show_object('Run Modal Responses')
    _select(window, pump, 'Run Modal Responses', 'Modes')
    assert 'Expand to Physical Responses' in _bar_acts(window)
    window.transform_selection()
    pump()
    assert 'Run Physical Responses' in window.project
    assert 'Run Physical Responses' in next(
        group['members'] for group in window.project.object_groups
        if 'Plate' in group['members']), \
        'the expansion lands with the geometry it animates on'
    assert window.project.journal[-1] == \
        "project.expand('Run Modal Responses', 'Modes')"
    assert np.allclose(window.project['Run Physical Responses'].ordinate,
                       window.project['Run'].ordinate, atol=1e-10)
    assert window.statusBar().currentMessage().startswith(
        'Run Physical Responses: 6 DOFs shared')


def test_without_the_pair_the_act_says_what_to_select(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Run')
    window.transform_selection()
    assert 'Select a data object and a shape set' in \
        window.statusBar().currentMessage()


def _pick(window, name, records):
    """Click the grid cells of `records`, as a user does — with the
    signals on, so the tree learns the object is selected too."""
    grid = window.record_grids[name]
    wanted = set(records)
    for cell, record in grid._records.items():
        if record in wanted:
            grid.item(*cell).setSelected(True)


def test_a_picked_mode_expands_its_contribution_alone(window, pump):
    """One record of the modal responses picked in its grid, beside the
    set: the entry says one mode, and the object it makes is that
    mode's contribution, named for it (Brandon, 2026-09-04)."""
    _populate(window, pump)
    window.project.transform('Run', 'Modes')
    window.show_object('Run Modal Responses')
    pump()
    window.tree.clearSelection()
    window._item_for_object('Modes').setSelected(True)
    item = window._item_for_object('Run Modal Responses')
    item.setExpanded(True)
    pump()
    _pick(window, 'Run Modal Responses', [5])
    pump()
    picked = [(kind, detail) for kind, _name, _obj, detail
              in window.selected_references() if kind == 'record']
    assert picked == [('record', 5)]
    assert 'Expand to Physical Responses' in _bar_acts(window), \
        'a record pick is its object, for the bar'
    window.transform_selection()
    pump()
    assert 'Run Physical Responses (M6)' in window.project
    made = window.project['Run Physical Responses (M6)']
    rigid = window.project['Modes']
    q6 = window.project['Run Modal Responses'].ordinate[5]
    assert np.allclose(made.ordinate, np.outer(rigid.shape_matrix[5], q6),
                       atol=1e-10)
    assert made.comment[0] == 'acceleration from modes M6'
    assert window.project.journal[-1] == \
        "project.expand('Run Modal Responses', 'Modes', records=[5])"
    assert 'Run Physical Responses (M6)' in next(
        group['members'] for group in window.project.object_groups
        if 'Plate' in group['members'])


def test_picked_channels_transform_alone(window, pump):
    _populate(window, pump)
    window.tree.clearSelection()
    window._item_for_object('Modes').setSelected(True)
    item = window._item_for_object('Run')
    item.setExpanded(True)
    pump()
    _pick(window, 'Run', range(1, 15))
    pump()
    assert 'Transform to Modal Responses' in _bar_acts(window)
    window.transform_selection()
    pump()
    made = window.project['Run Modal Responses']
    assert len(made.transform_report.shared['acceleration']) == 14
    assert window.project.journal[-1] == (
        "project.transform('Run', 'Modes', records=[1, 2, 3, 4, 5, 6, 7, "
        "8, 9, 10, 11, 12, 13, 14])")
