"""Rules in `core.shapes` pinned by value where a mutation pass found
them reached only through pictures and whole-set comparisons
(2026-10-11).

`tools/mutate.py` over shapes.py against its shape, scaling and
synthesis tests caught 152 of 199. What survived: which quantity an
overlay synthesizes for each record, which modes a delete may name,
the per-pair scale ratio's edges, a DOF's sign, and the damping a
label prints. Stated here as the rules they are.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Frf
from visualdynamics.core.shapes import (
    ScaleComparison,
    ShapeSet,
    _split_sign,
    pair_caption,
    scale_ratios,
    synthesize_overlay,
)


def _modes():
    return ShapeSet(frequency=[10.0, 25.0, 40.5], damping=[0.02, 0.03, 0.0],
                    coordinate=['1X+', '2X+'],
                    shape_matrix=[[1.0, 0.5], [0.3, -1.0], [0.8, 0.8]],
                    description=['first', 'second', 'third'])


def test_an_overlay_synthesizes_each_record_as_the_quantity_it_measures():
    """A displacement per force is a receptance, a velocity per force a
    mobility, an acceleration per force an accelerance — and a record
    that does not say is taken for an accelerometer's, the usual case."""
    shapes = _modes()
    f = np.linspace(0.0, 50.0, 51)
    frf = Frf(f, np.ones((4, f.size), dtype=complex),
              response_dof=['1X+', '2X+', '1X+', '2X+'],
              reference_dof=['1X+'] * 4,
              ordinate_dim=['length/force', 'velocity/force',
                            'acceleration/force', 'unknown'])
    overlay = synthesize_overlay(shapes, frf)
    assert overlay.synthesized and overlay.comment == ['synthesized'] * 4
    for row, power in enumerate([0, 1, 2, 2]):
        expected = shapes.synthesize_frf(f, [frf.response_dof[row]], ['1X+'],
                                         power=power)[0]
        assert np.allclose(overlay.ordinate[row], expected), row
    assert overlay.mode_frequencies == [10.0, 25.0, 40.5]
    picked = synthesize_overlay(shapes, frf, records=[1], modes=[1])
    assert picked.response_dof == ['2X+']
    assert picked.mode_frequencies == [25.0]


def test_an_overlay_needs_a_record_the_shapes_cover_at_both_ends():
    f = np.linspace(0.0, 50.0, 6)
    off = Frf(f, np.ones((1, 6), dtype=complex), response_dof=['7X+'],
              reference_dof=['1X+'], ordinate_dim=['acceleration/force'])
    assert synthesize_overlay(_modes(), off) is None


def test_a_delete_names_modes_that_exist_and_leaves_one():
    shapes = _modes()
    with pytest.raises(IndexError, match=r'no such modes: \[3\]'):
        shapes.delete_modes([3])
    with pytest.raises(IndexError, match=r'no such modes: \[-1\]'):
        shapes.delete_modes([-1])
    with pytest.raises(ValueError, match='cannot delete every mode'):
        shapes.delete_modes([0, 1, 2])
    assert shapes.num_shapes == 3, 'a refusal changes nothing'
    shapes.delete_modes([0])
    assert shapes.frequency.tolist() == [25.0, 40.5]
    assert shapes.description == ['second', 'third']
    assert shapes.shape_matrix.tolist() == [[0.3, -1.0], [0.8, 0.8]]


def test_a_scale_ratio_is_one_per_pair_and_none_where_there_is_no_answer():
    a = _modes()
    b = ShapeSet(frequency=[10.0, 25.0, 40.0], damping=[0.0] * 3,
                 coordinate=['2X+', '1X+', '3X+'],
                 shape_matrix=[[1.5, 3.0, 9.0], [0.0, 0.0, 9.0], [1.6, 1.6, 0.0]])
    zeroed = ShapeSet(frequency=[10.0], damping=[0.0], coordinate=['1X+', '2X+'],
                      shape_matrix=[[0.0, 0.0]])
    # norms on the shared 1X+ and 2X+ only: b's mode 0 is three times a's
    got = scale_ratios(a, b, [(0, 0), (2, 2), (0, 1), (3, 0), (0, 3)])
    assert got[:2] == [pytest.approx(3.0), pytest.approx(2.0)]
    assert got[2:] == [None, None, None], 'a zero mode, and two past the end'
    assert scale_ratios(zeroed, a, [(0, 0)]) == [None]
    assert scale_ratios(a, b) == [pytest.approx(3.0), None, pytest.approx(2.0)]
    stranger = ShapeSet([5.0], [0.0], ['9Z+'], [[1.0]])
    assert scale_ratios(a, stranger, [(0, 0), (1, 0)]) == [None, None]


def test_the_spread_of_ratios_is_largest_over_smallest_of_the_usable():
    assert ScaleComparison([0.0, 2.0, 4.0, np.nan]).spread == 2.0
    assert ScaleComparison([3.0]).spread == 1.0, 'one ratio cannot spread'
    assert ScaleComparison([2.0, 3.0]).spread == 1.5
    assert ScaleComparison([2.0, 3.0]).consistent, 'the limit itself is consistent'


@pytest.mark.parametrize('dof, split', [
    ('101X-', ('101X', -1.0)),
    ('101X+', ('101X', 1.0)),
    ('101X', ('101X', 1.0)),
    ('', ('', 1.0)),
])
def test_a_dofs_sign_is_its_last_character(dof, split):
    assert _split_sign(dof) == split


def test_damping_is_printed_as_a_percentage():
    shapes = _modes()
    assert shapes.mode_label(0) == 'Mode 1 — 10 Hz, 2% damping'
    assert shapes.mode_label(2) == 'Mode 3 — 40.5 Hz', 'undamped says nothing'
    caption = pair_caption('Test', shapes, 'FEM', shapes, 1, 0)
    assert caption.splitlines()[:2] == [
        'Test  mode 2  second  25.0000 Hz  3.000 %',
        'FEM  mode 1  first  10.0000 Hz  2.000 %']


def test_a_matrix_that_does_not_fit_its_coordinates_is_refused_with_both_counts():
    with pytest.raises(ValueError, match='shape_matrix has 3 columns but 2 coordinates'):
        ShapeSet([10.0], [0.0], ['1X+', '2X+'], [[1.0, 2.0, 3.0]])
