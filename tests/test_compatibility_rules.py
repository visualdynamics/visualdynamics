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
