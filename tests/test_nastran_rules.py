"""Nastran's card tables and the edges of its cards, pinned by value
where a mutation pass found them unguarded (2026-10-11).

`tools/mutate.py` over nastran.py caught 275 of 429 against
test_nastran.py and test_celas_spc.py. Forty-two survivors were the
two card tables, which a round trip reads and writes through alike;
they are checked here against the element vocabulary instead — a card
must name elements of its own shape and node count. The rest were the
boundaries of the cards: a spring's third component against its
fourth, a mass carried off its grid, the columns after 72.
"""

from __future__ import annotations

import pytest

import visualdynamics
from visualdynamics.core.fem import GroupProperties
from visualdynamics.core.geometry import ELEMENT_TYPES, element_family
from visualdynamics.io import nastran

FAMILY_OF_CARD = {'CROD': 'beams', 'CBAR': 'beams', 'CBEAM': 'beams',
                  'CTRIA3': 'triangles', 'CTRIA6': 'triangles',
                  'CQUAD4': 'quads', 'CQUAD8': 'quads', 'CTETRA': 'tetras',
                  'CPENTA': 'wedges', 'CHEXA': 'hexes', 'CPYRAM': 'pyramids',
                  'CELAS2': 'beams', 'CONM2': 'points'}


def test_a_card_reads_as_elements_of_its_own_shape_and_node_count():
    for card, by_count in nastran._ELEMENT_CARDS.items():
        for count, code in by_count.items():
            assert ELEMENT_TYPES[code][1] == count, (card, count)
            assert element_family(code) == FAMILY_OF_CARD[card], (card, code)


def test_an_element_is_written_as_a_card_of_its_own_shape():
    for code, card in nastran._CARD_NAMES.items():
        family = element_family(code)
        assert family == FAMILY_OF_CARD[card] or (card == 'CELAS2'
                                                  and family == 'points'), (code, card)
    assert sorted(nastran._CARD_NAMES) == [
        11, 21, 22, 23, 24, 41, 42, 44, 45, 91, 92, 94, 95, 111, 112, 113,
        115, 116, 117, 118, 136, 137, 138, 139, 161, 201, 202]


def _load(tmp_path, *cards):
    deck = tmp_path / 'deck.bdf'
    deck.write_text('BEGIN BULK\n' + ''.join(f'{c}\n' for c in cards) + 'ENDDATA\n')
    return nastran.load(str(deck))


GRIDS = ('GRID,1,,0.,0.,0.', 'GRID,2,,1.,0.,0.')


@pytest.mark.parametrize('component, between, alone', [
    (3, 136, 138), (4, 137, 139)])
def test_a_springs_third_component_translates_and_its_fourth_turns(
        tmp_path, component, between, alone):
    geometry = _load(tmp_path, *GRIDS, f'CELAS2,21,1.+3,1,{component},2,{component}',
                     f'CELAS2,22,1.+3,1,{component}')
    kinds = dict(zip(geometry.elem_id.tolist(), geometry.elem_type.tolist()))
    assert (kinds[21], kinds[22]) == (between, alone)


def test_cylindrical_and_spherical_systems_keep_their_kind(tmp_path):
    geometry = _load(tmp_path, *GRIDS,
                     'CORD2R,5,0, 0.,0.,0., 0.,0.,1., 1.,0.,0.',
                     'CORD2C,6,0, 0.,0.,0., 0.,0.,1., 1.,0.,0.',
                     'CORD2S,7,0, 0.,0.,0., 0.,0.,1., 1.,0.,0.')
    assert geometry.cs_id.tolist() == [0, 5, 6, 7]
    assert geometry.cs_type.tolist() == [0, 0, 1, 2]


def test_a_mass_off_its_grid_or_of_nothing_is_left_for_the_analyst(tmp_path):
    """Plain point masses get a group per distinct mass, numbered after
    every property id; a mass with an offset, or none, gets no mass."""
    geometry = _load(tmp_path, *GRIDS, 'GRID,3,,2.,0.,0.', 'GRID,4,,3.,0.,0.',
                     'CBAR,1,7,1,2',
                     'CONM2,11,1,0,2.0', 'CONM2,12,2,0,0.5', 'CONM2,13,3,0,2.0',
                     'CONM2,14,4,0,2.0,0.1', 'CONM2,15,4,0,0.0')
    group = dict(zip(geometry.elem_id.tolist(), geometry.elem_group.tolist()))
    assert (group[12], group[11], group[13]) == (8, 9, 9), 'by mass, after property 7'
    names = dict(zip(geometry.group_id.tolist(), geometry.group_name))
    assert (names[8], names[9]) == ('CONM2 0.5', 'CONM2 2')
    assert group[14] == group[15] == 0
    assert geometry.group_properties[9].mass == 2.0


def test_the_columns_after_72_are_not_fields(tmp_path):
    """A small-field card's continuation marker sits in columns 73-80;
    reading it as a ninth field would turn a node's output system into
    text."""
    card = 'GRID           1       0     1.0     2.0     3.0       0' + ' ' * 16 + '+CONT'
    assert len(card) > 72
    geometry = _load(tmp_path, card)
    assert geometry.node_xyz.tolist() == [[1.0, 2.0, 3.0]]


def test_a_spring_stiff_in_two_directions_is_two_cards_with_fresh_ids(tmp_path):
    geometry = visualdynamics.Geometry([1, 2], [[0.0, 0, 0], [1.0, 0, 0]])
    geometry.add_elements([[1, 2], [1, 2]], [136, 21], [1, 2])
    geometry.group_properties[1] = GroupProperties(
        stiffness=(5.0, None, 7.0, None, None, None))
    path = tmp_path / 'out.bdf'
    nastran.save(geometry, str(path))
    springs = [line for line in path.read_text().splitlines()
               if line.startswith('CELAS2')]
    fields = [nastran._fields(line) for line in springs]
    assert [(f[1], f[2], f[4]) for f in fields] == [(1, 5.0, 1), (3, 7.0, 3)], (
        'the first keeps its id; the second takes one past every element')
    back = nastran.load(str(path))
    assert len(set(back.elem_id.tolist())) == len(back.elem_id)


def test_a_spring_in_no_spring_group_is_written_where_it_is(tmp_path):
    """No stiffness to say: zero, from its first grid in X, to its
    second in X when it has one."""
    from visualdynamics.core.fem import material

    geometry = visualdynamics.Geometry([1, 2], [[0.0, 0, 0], [1.0, 0, 0]])
    geometry.add_elements([[1, 2], [1]], [136, 138], [1, 2])
    geometry.group_properties[1] = GroupProperties(material('1018 steel'))
    path = tmp_path / 'out.bdf'
    nastran.save(geometry, str(path))
    fields = [nastran._fields(line) for line in path.read_text().splitlines()
              if line.startswith('CELAS2')]
    assert [f[1:] for f in fields] == [[1, 0.0, 1, 1, 2, 1], [2, 0.0, 1, 1]]
