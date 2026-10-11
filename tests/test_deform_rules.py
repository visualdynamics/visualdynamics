"""Deflections' yardsticks and line handling, pinned by value where a
mutation pass found them unguarded (2026-10-11).

`tools/mutate.py` over deform.py caught 98 of 173 against test_deform
and test_complex_modes. Most of what survived is the chunk size the
peak walks the lines in, which changes the memory used and not the
answer; what is pinned here changes the answer.
"""

from __future__ import annotations

import numpy as np

import visualdynamics
from visualdynamics.deform import EnvelopeDeflection, OdsDeflection, _passes, model_size


def _line(offset=(0.0, 0.0, 0.0)):
    return visualdynamics.Geometry([1, 2, 3], np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
                                   + np.asarray(offset), length_unit='m')


def test_the_yardstick_is_the_bounding_boxs_diagonal_wherever_the_box_is():
    box = visualdynamics.Geometry([1, 2], [[10.0, 10.0, 10.0], [13.0, 14.0, 10.0]])
    assert model_size(box) == 5.0
    assert model_size(visualdynamics.Geometry([1], [[3.0, 3.0, 3.0]])) == 1.0, (
        'a single point is measured against one')


def test_a_line_past_either_end_is_held_at_the_end():
    ods = OdsDeflection(_line(), ['1Z+', '2Z+'], np.ones((2, 5), dtype=complex))
    ods.line = 99
    assert ods.line == 4
    ods.line = -3
    assert ods.line == 0


def test_an_envelope_reaches_furthest_where_two_axes_of_a_node_add():
    """Node 1 measured in X and Y at equal level: its diagonal is the
    root of two of the line's top amplitude. The other line's top is
    larger and is normalized away; a silent line is not a division by
    zero."""
    psd = np.array([[1.0, 4.0, 0.0],      # 1X+
                    [1.0, 0.0, 0.0],      # 1Y+
                    [0.25, 1.0, 0.0]])    # 2Z+
    envelope = EnvelopeDeflection(_line(), ['1X+', '1Y+', '2Z+'], psd)
    assert envelope.peak_magnitude == np.sqrt(2.0)


def test_repeated_targets_are_split_into_passes_that_never_repeat_one():
    assert [group.tolist() for group in _passes([5, 5, 7, 5])] == [[0, 2], [1], [3]]
    assert _passes([]) == []


def test_an_envelope_counts_what_it_leaves_out_by_reason():
    """Two autos and two cross records of accelerations, and a force
    auto: the crosses go first, then the force — the commoner kind
    answers."""
    from visualdynamics.core.data import Psd
    from visualdynamics.deform import envelope_records

    f = np.linspace(1.0, 10.0, 4)
    psd = Psd(f, np.ones((5, 4)), response_dof=['1X+', '1X+', '2X+', '2X+', '3X+'],
              reference_dof=['1X+', '2X+', '1X+', '2X+', '3X+'],
              ordinate_dim=['acceleration**2/frequency'] * 4 + ['force**2/frequency'])
    assert envelope_records(psd) == ([0, 3], 'acceleration', 2, 1)
