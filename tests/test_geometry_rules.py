"""Geometry's tables, defaults and refusals, pinned by value where a
mutation pass found them unguarded (2026-10-11).

`tools/mutate.py` over geometry.py against its five core test files
caught 359 of 504. Thirty of the survivors were the element table's
node counts and the families they map to — a number off by one in a
row no fixture uses changes nothing any test reads — so the table is
checked against itself: every name says its node count and its kind,
and the two have to agree.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.geometry import (
    CS_TYPES,
    ELEMENT_TYPES,
    View,
    element_family,
)


def _family_by_name(name, shape):
    if shape == 'point':
        return 'points'
    if shape == 'line':
        return 'beams'
    for prefix, family in (('trishell', 'triangles'), ('quadshell', 'quads'),
                           ('tri', 'triangles'), ('quad', 'quads'),
                           ('tet', 'tetras'), ('wedge', 'wedges'),
                           ('hex', 'hexes'), ('pyramid', 'pyramids')):
        if name.startswith(prefix):
            return family
    raise AssertionError(name)


@pytest.mark.parametrize('code', sorted(ELEMENT_TYPES))
def test_every_element_type_says_its_node_count_and_family_by_name(code):
    name, count, shape = ELEMENT_TYPES[code]
    digits = re.search(r'\d+', name)
    if digits:
        assert int(digits.group()) == count, name
    else:
        assert count == (1 if shape == 'point' else 2), name
    assert element_family(code) == _family_by_name(name, shape)


def test_coordinate_system_types_are_the_universal_files_codes():
    assert CS_TYPES == {0: 'cartesian', 1: 'cylindrical', 2: 'spherical'}


def test_a_view_needs_two_directions_that_are_not_one_line():
    View((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match='a direction to look from'):
        View((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match='a direction for up'):
        View((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='along the line of sight'):
        View((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    # a little across the line of sight is enough, at any scale
    View((1000.0, 0.0, 0.0), (0.0, 1e-6, 1.0e-3))


def test_a_geometry_fills_what_it_was_not_told():
    """Nodes defined and displaced in system 1 and drawn in color 1; a
    system with no name is unnamed, cartesian, unturned and at the
    origin."""
    g = visualdynamics.Geometry([1, 2], np.zeros((2, 3)), cs_id=[1, 7])
    assert g.node_def_cs.tolist() == [1, 1] and g.node_disp_cs.tolist() == [1, 1]
    assert g.node_color.tolist() == [1, 1]
    assert list(g.cs_name) == ['', ''] and g.cs_type.tolist() == [0, 0]
    assert np.array_equal(g.cs_matrix, np.tile(np.vstack([np.eye(3), np.zeros(3)]),
                                               (2, 1, 1)))


def _systems():
    return visualdynamics.Geometry([1], np.zeros((1, 3)), cs_id=[1, 7])


def test_a_system_is_placed_by_its_id_and_turned_one_way_only():
    g = _systems()
    g.place_coordinate_system(7, origin=(1.0, 2.0, 3.0), angles=(0.0, 0.0, 90.0))
    assert np.allclose(g.cs_matrix[1, :3], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert g.cs_matrix[1, 3].tolist() == [1.0, 2.0, 3.0]
    assert np.array_equal(g.cs_matrix[0], np.vstack([np.eye(3), np.zeros(3)])), (
        'the other system is untouched')
    turn = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    g.place_coordinate_system(7, rotation=turn)
    assert np.array_equal(g.cs_matrix[1, :3], turn)
    assert g.cs_matrix[1, 3].tolist() == [1.0, 2.0, 3.0], 'the origin kept'
    with pytest.raises(ValueError, match='as angles or as a rotation, not both'):
        g.place_coordinate_system(7, angles=(0, 0, 0), rotation=np.eye(3))
    with pytest.raises(KeyError, match='no coordinate system 3'):
        g.place_coordinate_system(3, origin=(0, 0, 0))


def test_a_system_cannot_take_another_systems_id_but_may_keep_its_own():
    g = _systems()
    g.renumber_coordinate_system(1, 7)
    with pytest.raises(ValueError, match='coordinate system 1 already exists'):
        g.renumber_coordinate_system(1, 1)
    g.renumber_coordinate_system(1, 8)
    assert g.cs_id.tolist() == [1, 8]


def _chain(n=4):
    return visualdynamics.Geometry(np.arange(1, n + 1),
                                   np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]))


def test_many_elements_are_checked_before_any_is_added():
    g = _chain()
    with pytest.raises(ValueError, match=r'unknown nodes \[5, 6, 7, 8, 9, 10, 11, 12, 13, 14\]$'):
        g.add_elements([[1, n] for n in range(20, 4, -1)], [21] * 16, [1] * 16)
    with pytest.raises(ValueError, match='unknown element type 7$'):
        g.add_elements([[1, 2], [2, 3]], [21, 7], [1, 1])
    g.add_elements([[1, 2]], [21], [3])
    with pytest.raises(ValueError, match='element group 3 would hold beams and triangles'):
        g.add_elements([[2, 3], [1, 2, 3]], [21, 41], [3, 3])
    assert len(g.elem_conn) == 1, 'a refusal adds nothing'
    g.add_elements([[2, 3], [3, 4]], [21, 21], [3, 5])
    assert g.elem_id.tolist() == [1, 2, 3]
    assert g.group_id.tolist() == [3, 5] and list(g.group_name) == ['', '']
    g.add_elements([], [], [])
    assert len(g.elem_conn) == 3


def test_one_element_is_refused_from_a_group_of_another_family():
    g = _chain()
    assert g.add_element([1, 2], 21, group=4) == 0
    assert g.add_element([2, 3], 21, group=4) == 1, 'its row'
    with pytest.raises(ValueError, match='element group 4 holds beams; a triangle goes in'):
        g.add_element([1, 2, 3], 41, group=4)


def test_coincident_nodes_join_through_chains_to_the_lowest_id():
    """5 within reach of 3, and 3 of 9, makes all three one — named 3,
    the lowest, whichever row it sits in."""
    g = visualdynamics.Geometry([5, 3, 9, 4], [[0.0, 0, 0], [0.8, 0, 0], [1.6, 0, 0],
                                               [5.0, 0, 0]])
    assert g.coincident_nodes(1.0) == {5: 3, 9: 3}
    assert visualdynamics.Geometry([1], np.zeros((1, 3))).coincident_nodes(1.0) == {}


def test_a_merge_with_nothing_to_merge_says_so():
    g = _chain()
    g.add_elements([[1, 2]], [21], [1])
    assert g.merge_coincident_nodes(0.1) == {'merged': 0, 'into': 0, 'duplicates': 0,
                                                'links': 0}
    # a plate two of whose corners would become one is refused; a beam
    # whose two nodes become one joins nothing and goes (2026-10-11)
    g.add_elements([[1, 2, 3, 4]], [94], [2])
    with pytest.raises(ValueError, match='folds element 2 onto itself'):
        g.merge_coincident_nodes(1.5)
    g.delete_elements([2])
    assert g.merge_coincident_nodes(1.5)['links'] == 1
    assert g.elem_id.tolist() == []


def test_deleting_what_is_not_there_deletes_nothing():
    g = _chain()
    assert g.delete_groups([42]) == {'groups': 0, 'elements': 0, 'nodes': 0}
    assert g.delete_nodes([42]) == {'nodes': 0, 'elements': 0}
    assert g.delete_coordinate_systems([42]) == {'coordinate_systems': 0,
                                                 'nodes_reassigned': 0}


def test_a_merge_of_groups_names_the_group_that_is_not_there():
    g = _chain()
    g.add_elements([[1, 2], [3, 4]], [21, 21], [1, 2])
    assert g.merge_refusal([1, 2]) is None
    assert g.merge_refusal([1, 9, 8]) == 'no element group 9'


def test_a_drawn_line_is_its_chains_and_its_first_elements_color():
    g = _chain(5)
    g.add_elements([[1, 2], [2, 3], [4, 5]], [21] * 3, [1] * 3)
    g.elem_color[:] = [6, 2, 2]
    (line,) = g.drawn_lines()
    assert line['chains'] == [[1, 2, 3], [4, 5]] and line['color'] == 6


def test_only_solids_on_one_set_of_nodes_are_one_cell():
    g = visualdynamics.Geometry(np.arange(1, 9), np.eye(8, 3))
    g.add_elements([list(range(1, 9)), list(range(8, 0, -1))], [115, 115], [1, 1])
    g.add_elements([[1, 2], [2, 1]], [21, 21], [2, 2])
    assert g.duplicate_elements() == {2: 1}
