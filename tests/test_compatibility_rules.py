"""Modal objects against their shape sets, and the link rule, pinned by
value where a mutation pass found them unguarded (2026-10-11).

`tools/mutate.py` over compatibility.py caught 22 of 43 against
test_compatibility.py, which is about geometries; the modal half was
reached only through the transform's tests, and not for its message.
"""

from __future__ import annotations

import numpy as np

import visualdynamics
from visualdynamics.compatibility import (
    MAX_LISTED,
    Issue,
    _listed,
    blocks_a_link,
    check_compatibility,
    check_modal,
    check_object,
)

T = np.linspace(0.0, 1.0, 4)


def _modal(count):
    return visualdynamics.TimeHistory(T, np.zeros((count, 4)),
                                      response_dof=[f'M{k}' for k in range(1, count + 1)],
                                      ordinate_dim='acceleration')


def _set(count):
    return visualdynamics.ShapeSet(np.arange(1, count + 1) * 10.0, [0.0] * count,
                                   ['1X+'], np.ones((count, 1)))


def test_a_set_with_every_mode_named_answers_for_the_object():
    assert check_modal('q', _modal(4), {'A': _set(2), 'B': _set(4)}) is None


def test_the_largest_set_is_named_and_the_modes_beyond_it_listed():
    issue = check_modal('q', _modal(4), {'A': _set(2), 'B': _set(3), 'q': _set(9)})
    assert issue.kind == 'modes-not-in-set'
    assert issue.message == "modal coordinates up to M4, but shape set 'B' has 3 modes"
    assert issue.missing_dofs == ['M4'] and issue.sub_items == [3]


def test_with_no_set_at_all_every_mode_is_beyond():
    issue = check_modal('q', _modal(2), {'q': _modal(2)})
    assert issue.message.startswith('modal coordinates up to M2, and no shape set')
    assert issue.missing_dofs == ['M1', 'M2']


def test_only_a_modal_issue_blocks_a_link():
    assert not blocks_a_link(None)
    assert not blocks_a_link(Issue(name='x', kind='dofs-not-in-geometry', message=''))
    assert blocks_a_link(Issue(name='x', kind='modes-not-in-set', message=''))


def test_a_list_is_summarized_past_its_length():
    six = [f'{k}X+' for k in range(MAX_LISTED)]
    assert _listed(six) == ', '.join(six)
    assert _listed(six + ['9X+', '10X+']).endswith(' and 2 more')


def test_a_linked_modal_object_answers_to_its_own_groups_sets():
    objects = {'q': _modal(3), 'small': _set(2), 'big': _set(3)}
    linked = [{'members': ['q', 'small']}]
    report = check_compatibility(objects, object_groups=linked)
    assert report.incompatible_names == ['q']
    assert check_compatibility(objects).incompatible_names == [], (
        'unlinked, any set in the project answers')


# ---- forces with modal responses appended -----------------------------------

def _mixed(modes, force_at='1X+'):
    """A force channel, then modal coordinates M1 … M`modes`."""
    return visualdynamics.TimeHistory(
        T, np.zeros((1 + modes, 4)),
        response_dof=[force_at] + [f'M{k}' for k in range(1, modes + 1)],
        ordinate_dim=['force'] + ['acceleration'] * modes)


def _geometry():
    return visualdynamics.Geometry([1, 2], np.zeros((2, 3)))


def test_each_kind_of_record_answers_to_its_own_rule():
    """Forces at nodes the geometry has, and modal records a set in the
    group covers: nothing to report. Before, every M record read as a
    node the geometry lacked."""
    assert check_object('Run', _mixed(3), _geometry(), 'Plate',
                        companions={'Modes': _set(3)}) is None


def test_a_force_off_the_geometry_is_named_alone():
    issue = check_object('Run', _mixed(3, force_at='9X+'), _geometry(),
                         'Plate', companions={'Modes': _set(3)})
    assert issue.kind == 'dofs-not-in-geometry'
    assert issue.message == ("1 of 4 records reference DOFs not in "
                             "geometry 'Plate': 9X+")
    assert issue.sub_items == [0] and issue.missing_dofs == ['9X+']


def test_modes_beyond_the_set_are_named_and_do_not_block_a_link():
    issue = check_object('Run', _mixed(4), _geometry(), 'Plate',
                         companions={'Modes': _set(3)})
    assert issue.kind == 'modes-beyond-set'
    assert issue.message == ("modal coordinates up to M4, but shape set "
                             "'Modes' has 3 modes")
    assert issue.sub_items == [4] and issue.missing_dofs == ['M4']
    assert not blocks_a_link(issue), 'the forces are drawn all the same'


def test_both_shortfalls_are_one_issue():
    issue = check_object('Run', _mixed(4, force_at='9X+'), _geometry(),
                         'Plate', companions={'Modes': _set(3)})
    assert issue.kind == 'dofs-not-in-geometry'
    assert issue.message == (
        "1 of 5 records reference DOFs not in geometry 'Plate': 9X+; "
        "modal coordinates up to M4, but shape set 'Modes' has 3 modes")
    assert issue.sub_items == [0, 4]


def test_the_project_check_reads_the_group_and_runs_without_a_geometry():
    groups = [{'members': ['Plate', 'Modes', 'Run'], 'role': None}]
    objects = {'Plate': _geometry(), 'Modes': _set(3), 'Run': _mixed(3)}
    assert check_compatibility(objects, 'Plate', groups).issues == {}
    objects['Run'] = _mixed(4)
    assert check_compatibility(objects, 'Plate', groups).issues[
        'Run'].sub_items == [4]
    alone = check_compatibility({'Modes': _set(3), 'Run': _mixed(4)})
    assert alone.issues['Run'].kind == 'modes-beyond-set', \
        'no geometry: the modal records are still checked'


def test_a_mixed_history_links_to_a_geometry_without_its_set():
    project = visualdynamics.Project()
    project.add('Plate', _geometry())
    project.add('Run', _mixed(3))
    project.link('Plate', 'Run')
    assert any({'Plate', 'Run'} <= set(group['members'])
               for group in project.object_groups)


def test_the_tree_does_not_paint_a_mixed_history_red(window, pump):
    from PySide6.QtCore import Qt

    window.add_object('Plate', _geometry())
    window.add_object('Modes', _set(3))
    window.add_object('Run', _mixed(3))
    window._link_names(['Plate', 'Modes', 'Run'], announce=False)
    pump()
    assert window.report.issue_for('Run') is None
    row = window._item_for_object('Run')
    assert row.data(0, Qt.ItemDataRole.ForegroundRole) is None
