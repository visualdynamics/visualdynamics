"""What a derived record keeps of its source.

A filter, a truncation and an octave banding change the numbers and
nothing about which records they are: the DOFs, the quantity, the
units, the comment and the capture each record came from all carry.
`DataArray.record_fields` is the one statement of that (2026-10-10);
before it the list was written out at each, and nothing here noticed
when a copy dropped a field — a falsification that replaced the
comments, or forgot the blocks, passed the whole suite.
"""

from __future__ import annotations

import numpy as np

from visualdynamics.core.data import Psd, TimeHistory
from visualdynamics.core.filters import Filtering, filtered
from visualdynamics.core.truncate import Truncation, truncate


def _history():
    t = np.arange(4096) / 1024.0
    return TimeHistory(
        t, np.vstack([np.sin(2 * np.pi * 50 * t), np.cos(2 * np.pi * 80 * t)]),
        response_dof=['101X+', '102Y+'], ordinate_dim='acceleration',
        ordinate_unit='m/s^2', comment=['left', 'right'],
        block=['shock 1', 'shock 2'])


def _kept(source, derived):
    assert derived.response_dof == source.response_dof
    assert derived.ordinate_dim == source.ordinate_dim
    assert derived.ordinate_unit == source.ordinate_unit
    assert derived.block == source.block
    assert derived.block is not source.block, 'a copy, not the same list'


def test_a_filtered_record_is_the_same_records():
    source = _history()
    out = filtered(source, Filtering(low=None, high=200.0))
    _kept(source, out)
    assert out.comment == ['left', 'right']


def test_a_truncated_record_is_the_same_records():
    source = _history()
    out = truncate(source, Truncation(start=1.0, stop=2.0))
    _kept(source, out)
    assert out.comment == ['left', 'right']


def test_octave_bands_are_the_same_records_and_say_what_they_are():
    f = np.linspace(0.0, 2000.0, 2001)
    source = Psd(f, np.ones((2, len(f))), response_dof=['101X+', '102Y+'],
                 reference_dof=['101X+', '102Y+'], ordinate_dim='acceleration',
                 block=['run 1', 'run 2'])
    out = source.to_octave(3)
    _kept(source, out)
    assert out.reference_dof == source.reference_dof
    assert set(out.comment) == {'1/3 octave bands'}
