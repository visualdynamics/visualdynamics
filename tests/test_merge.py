"""Merging several objects of one type into one.

The option appears only when the merge is legitimate — same type, and
none of the refusals below — which is `mergeable`'s answer; `merge`
builds the combination. The rules are Brandon's: geometries refuse
colliding node numbers, FRFs refuse duplicate response/reference pairs,
repeated time-history DOFs become averaging columns, channel tables
never merge.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Frf, TimeHistory
from visualdynamics.core.merge import merge, mergeable
from visualdynamics.core.shapes import ShapeSet


def frf_by_reference(frfs):
    """The survey FRFs split into one set per reference — how a modal
    test often arrives."""
    rows = {}
    for i in range(frfs.num_records):
        rows.setdefault(frfs.reference_dof[i], []).append(i)
    return [Frf(frfs.abscissa, frfs.ordinate[picked],
                response_dof=[frfs.response_dof[i] for i in picked],
                reference_dof=[reference] * len(picked))
            for reference, picked in rows.items()]


def test_per_reference_frf_sets_merge_back_whole(survey):
    _shapes, frfs = survey
    parts = frf_by_reference(frfs)
    assert len(parts) == 4 and mergeable(parts) is None
    whole = merge(parts)
    assert whole.num_records == frfs.num_records
    assert set(zip(whole.response_dof, whole.reference_dof)) == \
        set(zip(frfs.response_dof, frfs.reference_dof))
    row = list(zip(whole.response_dof, whole.reference_dof)).index(
        (frfs.response_dof[7], frfs.reference_dof[7]))
    assert np.allclose(whole.ordinate[row], frfs.ordinate[7])


def test_duplicate_pairs_refuse(survey):
    _shapes, frfs = survey
    parts = frf_by_reference(frfs)
    assert 'appears more than once' in mergeable([parts[0], parts[0]])


def test_different_abscissas_refuse(survey):
    _shapes, frfs = survey
    parts = frf_by_reference(frfs)
    other = Frf(parts[1].abscissa * 2.0, parts[1].ordinate,
                response_dof=parts[1].response_dof,
                reference_dof=parts[1].reference_dof)
    assert 'abscissas differ' in mergeable([parts[0], other])


def test_repeated_time_dofs_become_averaging_columns():
    hits = [TimeHistory(np.linspace(0, 1, 8), value * np.ones((2, 8)),
                        response_dof=['1X+', '2X+'])
            for value in (1.0, 2.0, 3.0)]
    averaged = merge(hits)
    assert averaged.num_records == 6
    assert averaged.block == ['avg 1'] * 2 + ['avg 2'] * 2 + ['avg 3'] * 2
    from visualdynamics.gui.record_grid import grid_axes
    rows, columns = grid_axes(averaged)
    assert len(rows) == 2 and len(columns) == 3, 'DOF rows, average columns'


def test_disjoint_time_dofs_merge_without_blocks():
    a = TimeHistory(np.linspace(0, 1, 8), np.ones((1, 8)),
                    response_dof=['1X+'])
    b = TimeHistory(np.linspace(0, 1, 8), np.ones((1, 8)),
                    response_dof=['2X+'])
    assert merge([a, b]).block is None


def test_a_mixture_of_shared_and_unshared_channels_refuses():
    """The roving-survey case (Brandon, 2026-08-29): passes share their
    references and nothing else, and merging them builds an object
    whose force/response pairing survives only as row order — a record
    that could never have been measured."""
    t = np.linspace(0, 1, 8)
    passes = [TimeHistory(t, np.ones((2, 8)),
                          response_dof=[dof, '9001X+'],
                          ordinate_dim=['acceleration', 'force'])
              for dof in ('1Z+', '2Z+')]
    said = mergeable(passes)
    assert 'neither' in said and '9001X+' in said, said


def test_disjoint_channels_with_unequal_captures_refuse():
    """No shared channel, but one side holds twenty captures and the
    other five: merging claims they played at the same time, and the
    counts say they did not."""
    t = np.linspace(0, 1, 8)
    a = TimeHistory(t, np.ones((20, 8)), response_dof=['1X+'] * 20)
    b = TimeHistory(t, np.ones((5, 8)), response_dof=['2X+'] * 5)
    said = mergeable([a, b])
    assert 'same number of captures' in said and '20' in said, said


def test_repeat_runs_of_unequal_length_still_merge():
    """Twenty captures of a setup plus five more of the same setup is
    twenty-five averages per channel — uniform, honest, allowed."""
    t = np.linspace(0, 1, 8)
    a = TimeHistory(t, np.ones((4, 8)), response_dof=['1X+', '2X+'] * 2)
    b = TimeHistory(t, np.ones((2, 8)), response_dof=['1X+', '2X+'])
    assert mergeable([a, b]) is None
    assert merge([a, b]).num_records == 6


def test_mixed_types_refuse(survey):
    shapes, frfs = survey
    assert 'same type' in mergeable([shapes, frfs])


# ---- channel tables ------------------------------------------------------
#
# The rule used to be that channel tables never merge. The multi-pass
# survey changed it (Brandon, 2026-08-20): six roving passes carry six
# tables whose reference rows repeat verbatim and whose response rows
# are disjoint, and a project wants one table, not six.


def _pass_table(nodes, first_channel=1):
    """A pass's table: response rows for `nodes`, then the shared
    reference rows every pass carries verbatim."""
    rows = [{'channel': first_channel + k, 'node': node,
             'direction': 'Z+', 'role': 'response', 'unit': 'm/s**2',
             'sensitivity': '10.0'} for k, node in enumerate(nodes)]
    rows += [{'channel': first_channel + len(nodes) + k, 'node': node,
              'direction': 'Z+', 'role': 'reference', 'unit': 'N',
              'sensitivity': '2.0'} for k, node in enumerate((901, 902))]
    from visualdynamics.core.channel_table import ChannelTable

    return ChannelTable({key: [row[key] for row in rows]
                         for key in rows[0]})


def test_pass_tables_merge_collapsing_the_shared_references():
    """One table per roving pass: disjoint responses, identical
    reference rows. The merge keeps every response once, the
    references once, and renumbers — channel numbers are the wiring
    of one acquisition, not the identity of a point."""
    a, b = _pass_table([101, 102]), _pass_table([103, 104])
    assert mergeable([a, b]) is None
    merged = merge([a, b])
    frame = merged.frame
    assert len(frame) == 6, 'four responses and the two shared references'
    assert list(frame['channel']) == [1, 2, 3, 4, 5, 6], 'renumbered'
    responses = frame[frame['role'] == 'response']['node']
    assert sorted(int(n) for n in responses) == [101, 102, 103, 104]
    references = frame[frame['role'] == 'reference']['node']
    assert [int(n) for n in references] == [901, 902]


def test_identical_tables_merge_to_one_of_themselves():
    a, b = _pass_table([101, 102]), _pass_table([101, 102])
    merged = merge([a, b])
    assert len(merged.frame) == len(a.frame)


def test_tables_that_disagree_about_a_point_refuse():
    """The same point in the same role with two descriptions is a
    disagreement to resolve, not a table to build."""
    a, b = _pass_table([101, 102]), _pass_table([101, 102])
    b.frame.loc[0, 'sensitivity'] = '99.0'
    reason = mergeable([a, b])
    assert reason is not None and '101Z+' in reason


def test_geometries_with_colliding_nodes_refuse(survey):
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    assert 'node numbers collide' in mergeable([geometry, geometry])


def test_disjoint_geometries_merge_whole():
    from visualdynamics.core.geometry import Geometry

    left = Geometry(node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]])
    left.add_beams([1, 2])
    right = Geometry(node_id=[11, 12], node_xyz=[[0, 1, 0], [1, 1, 0]])
    right.add_beams([11, 12])
    both = merge([left, right])
    assert sorted(both.node_id.tolist()) == [1, 2, 11, 12]
    assert len(both.elem_conn) == 2
    both.validate()


def test_mixed_unit_states_refuse():
    from visualdynamics.core.geometry import Geometry

    defined = Geometry(node_id=[1], node_xyz=[[0, 0, 0]])
    defined.define_units('m')
    raw = Geometry(node_id=[2], node_xyz=[[1, 0, 0]])
    assert 'units defined' in mergeable([defined, raw])


def test_shape_sets_merge_sorted_by_frequency(survey):
    truth, _frfs = survey
    low = ShapeSet(truth.frequency[:5], truth.damping[:5], truth.coordinate,
                   truth.shape_matrix[:5])
    high = ShapeSet(truth.frequency[5:9], truth.damping[5:9],
                    truth.coordinate, truth.shape_matrix[5:9])
    both = merge([high, low])       # deliberately out of order
    assert both.num_shapes == 9
    assert (np.diff(both.frequency) >= 0).all()
    lookup = truth.coordinate.index('101X+')
    row = int(np.argmin(np.abs(both.frequency - truth.frequency[6])))
    assert both.shape_matrix[row, both.coordinate.index('101X+')] == \
        pytest.approx(float(truth.shape_matrix[6, lookup]))


def test_shape_sets_over_different_dofs_refuse(survey):
    truth, _frfs = survey
    partial = ShapeSet(truth.frequency[:3], truth.damping[:3],
                       truth.coordinate[:10], truth.shape_matrix[:3, :10])
    assert 'different DOFs' in mergeable([truth, partial])


# ---- the menu ---------------------------------------------------------------

def select_objects(window, names):
    window.tree.setCurrentItem(window._item_for_object(names[0]))
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)


def test_the_menu_offers_merge_only_when_it_would_work(window, pump,
                                                       survey):
    _shapes, frfs = survey
    parts = frf_by_reference(frfs)
    window.add_object('Ref 1', parts[0])
    window.add_object('Ref 2', parts[1])
    window.add_object('Twin', parts[0])
    select_objects(window, ['Ref 1', 'Ref 2'])
    assert window._merge_candidates() is not None
    select_objects(window, ['Ref 1', 'Twin'])
    assert window._merge_candidates() is None, 'duplicate pairs hide it'
    select_objects(window, ['Ref 1'])
    assert window._merge_candidates() is None, 'one object is no merge'


def test_merging_replaces_the_selection_with_one_object(window, pump,
                                                        survey):
    _shapes, frfs = survey
    parts = frf_by_reference(frfs)
    window.add_object('Ref 1', parts[0])
    window.add_object('Ref 2', parts[1])
    select_objects(window, ['Ref 1', 'Ref 2'])
    window.merge_selected()
    pump()
    assert 'Ref 1' not in window.objects and 'Ref 2' not in window.objects
    assert 'FRF' in window.objects
    assert window.objects['FRF'].num_records == \
        parts[0].num_records + parts[1].num_records
    assert 'Merged Ref 1, Ref 2 into FRF' in \
        window.statusBar().currentMessage()
