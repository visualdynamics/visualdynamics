"""Merge's refusals and labels, pinned by value where a mutation pass
found them read only for a word or two (2026-10-11).

`tools/mutate.py` over merge.py caught 65 of 98 against test_merge.py.
The sine specifications' merge was not in that file at all; the time
refusals were checked for a word and a number, not for which channel
each number belongs to; and a merge of two objects alike cannot tell
the first from the second, so the cases here are three objects, or two
that differ where it matters.
"""

from __future__ import annotations

import numpy as np

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.merge import merge, mergeable
from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone

T = np.linspace(0.0, 1.0, 8)


def _spec(name, dofs=('1Z+',), unit='m/s**2', comment=''):
    tone = SineTone(name, 0.0, [20.0, 40.0], [[1.0] * len(dofs)] * 2, [LOG], [60.0])
    return SineSweepSpecification([tone], list(dofs), ordinate_unit=unit,
                                  comment=comment)


def test_sine_specifications_join_their_tones_under_one_set_of_channels():
    merged = merge([_spec('A', comment='first'), _spec('B'), _spec('C', comment='third')])
    assert [tone.name for tone in merged.tones] == ['A', 'B', 'C']
    assert merged.comment == 'first; third'


def test_sine_specifications_are_refused_by_what_differs_wherever_it_is():
    """The odd one out third in line as well as second: every other
    specification is checked against the first."""
    for odd in (1, 2):
        specs = [_spec('A'), _spec('B'), _spec('C')]
        specs[odd] = _spec('BC'[odd - 1], dofs=('2Z+',))
        assert mergeable(specs) == ("the specifications name different "
                                    "control channels: ['1Z+'] and ['2Z+']"), odd
    said = mergeable([_spec('A'), _spec('B'), _spec('C', unit='g')])
    assert said == 'the specifications are in different units'
    said = mergeable([_spec('A'), _spec('B'), _spec('A'), _spec('B'), _spec('C')])
    assert said == "tone names repeat across the specifications: ['A', 'B']"
    assert mergeable([_spec('A'), _spec('B')]) is None


def test_a_half_shared_merge_names_a_shared_channel_and_an_unshared_one():
    a = TimeHistory(T, np.ones((2, 8)), response_dof=['1X+', '2X+'],
                    ordinate_dim='acceleration')
    b = TimeHistory(T, np.ones((2, 8)), response_dof=['1X+', '3X+'],
                    ordinate_dim='acceleration')
    said = mergeable([a, b])
    assert said.startswith('the histories share 1X+ but not ')
    assert said.split(' but not ')[1].split(':')[0] in ('2X+', '3X+')


def test_unequal_captures_name_each_count_with_its_channel():
    a = TimeHistory(T, np.ones((20, 8)), response_dof=['1X+'] * 20)
    b = TimeHistory(T, np.ones((5, 8)), response_dof=['2X+'] * 5)
    assert '1X+ would hold 20 and 2X+ 5, and' in mergeable([a, b])


def test_an_unrepeated_channel_keeps_its_own_block_label():
    """Disjoint channels merge without averaging columns, but a label a
    record already had rides along; one with none is the first average."""
    a = TimeHistory(T, np.ones((1, 8)), response_dof=['1X+'], block=['run 7'])
    b = TimeHistory(T, np.ones((1, 8)), response_dof=['2X+'])
    assert merge([a, b]).block == ['run 7', 'avg 1']


def test_specifications_must_carry_the_same_limit_curves():
    f = np.linspace(10.0, 100.0, 5)

    def spec(dof='1X+', **limits):
        return visualdynamics.Specification(f, np.ones((1, 5)), response_dof=[dof],
                                            ordinate_dim='acceleration**2/frequency',
                                            **limits)

    upper = np.full((1, 5), 2.0)
    said = mergeable([spec(abort_upper=upper), spec('2X+'),
                      spec('3X+', abort_upper=upper)])
    assert said == 'the specifications carry different limit curves'
    assert mergeable([spec(abort_upper=upper), spec('2X+', abort_upper=upper)]) is None


def test_merged_geometries_keep_every_coordinate_system_once_and_the_first_view():
    a = visualdynamics.Geometry([1], [[0.0, 0.0, 0.0]], length_unit='m')
    b = visualdynamics.Geometry([2], [[1.0, 0.0, 0.0]], cs_id=[1, 5],
                                cs_name=['global', 'tilted'],
                                cs_type=[0, 0], cs_matrix=[np.eye(4, 3)] * 2,
                                length_unit='m')
    b.view = {'azimuth': 30.0}
    merged = merge([a, b])
    assert merged.cs_id.tolist() == [1, 5]
    assert list(merged.cs_name) == ['', 'tilted'], 'the first to name a system names it'
    assert merged.view == {'azimuth': 30.0}


def test_a_channel_in_two_histories_becomes_two_averages():
    a = TimeHistory(T, np.ones((1, 8)), response_dof=['1X+'])
    b = TimeHistory(T, np.ones((1, 8)), response_dof=['1X+'])
    assert merge([a, b]).block == ['avg 1', 'avg 2']


def test_shape_sets_merge_with_their_units_all_defined_or_all_not():
    def modes(frequency, unit):
        return visualdynamics.ShapeSet([frequency], [0.01], ['1X+'], [[1.0]],
                                       mass_unit=unit)

    assert mergeable([modes(10.0, 'kg'), modes(20.0, 'kg')]) is None
    assert mergeable([modes(10.0, None), modes(20.0, None)]) is None
    assert mergeable([modes(10.0, 'kg'), modes(20.0, None)]) == (
        'some have units defined and some do not')
