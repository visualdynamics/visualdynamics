"""The native ADF readers and writers, round-tripped.

`.afu`, `.ati` and `.ash` — the I-DEAS-era Associated Data Files —
written by visualdynamics and read back by it, values, DOFs, units and
shapes intact. These tests need no fixture beyond what they write
themselves, which is what makes them the ones that travel: they hold
the readers to their own output rather than to a sample set.

The tests that hold them to *other* tooling's output live beside this
file where that sample set does.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import visualdynamics


def all_records(imported):
    """Flatten an import result to one records list, whatever grouped."""
    objects = (imported.values() if isinstance(imported, dict)
               else [imported])
    rows = []
    for obj in objects:
        for k in range(obj.num_records):
            rows.append((obj, k))
    return rows


def _same_functions(a, b):
    A, B = all_records(a), all_records(b)
    assert len(A) == len(B)
    for (obj, k) in A:
        dof = obj.response_dof[k]
        match = next((o, j) for o, j in B if o.response_dof[j] == dof
                     and np.allclose(np.asarray(o.ordinate[j]),
                                     np.asarray(obj.ordinate[k]),
                                     rtol=1e-6, atol=1e-30))
        other, j = match
        assert np.allclose(other.abscissa, obj.abscissa, rtol=1e-6)
        assert other.ordinate_dim[j] == obj.ordinate_dim[k]


def test_a_truncated_file_is_refused():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.afu', delete=False) as f:
        f.write(b'\xff\xff\xff\xff' + b'\x00' * 100)
        path = f.name
    try:
        with pytest.raises(ValueError, match='is not an Associated Data File'):
            visualdynamics.import_file(path, format='adf')
    finally:
        os.unlink(path)


def test_leg1_functions_round_trip(tmp_path):
    from visualdynamics.core.data import Frf

    up = np.arange(1, 33, dtype=float)
    frf = Frf(np.arange(32) * 4.0,
              np.array([up * 0.01 + 1j * up[::-1] * 0.02]),
              response_dof=['9X+'], reference_dof=['1Z+'],
              ordinate_dim='acceleration/force',
              ordinate_unit='m/s**2', reference_unit='N',
              comment='leg one')
    visualdynamics.export_file(frf, str(tmp_path / 'l1.afu'),
                               format='adf_functions')
    back = visualdynamics.import_file(str(tmp_path / 'l1.afu'))
    _same_functions(frf, back)
    assert back.comment[0] == 'leg one'


def test_leg1_time_round_trip(tmp_path):
    t = np.arange(128) / 256.0
    hist = visualdynamics.TimeHistory(
        t, np.array([np.cos(2 * np.pi * 7 * t), t * 0.5]),
        response_dof=['3Z+', '4Z-'], ordinate_dim='acceleration',
        ordinate_unit='m/s**2')
    visualdynamics.export_file(hist, str(tmp_path / 'l1.ati'),
                               format='adf_time')
    back = visualdynamics.import_file(str(tmp_path / 'l1.ati'))
    _same_functions(hist, back)


def test_leg1_shapes_round_trip_with_the_modal_quad(tmp_path):
    from visualdynamics.core.shapes import ShapeSet

    s = ShapeSet([12.5, 20.0], [0.02, 0.03],
                 ['1X+', '1Y+', '2X+', '2Y+'],
                 [[1.0, 2.0, 3.0, 4.0], [0.5 + 0.5j, 2j, -1.0, 0.25]],
                 modal_mass=[1.5 + 0.25j, 2.0],
                 modal_damping=[0.033 + 0.044j, 0.05],
                 comment=['one', 'two'])
    visualdynamics.export_file(s, str(tmp_path / 'l1.ash'),
                               format='adf_shapes')
    back = visualdynamics.import_file(str(tmp_path / 'l1.ash'))
    assert np.allclose(back.shape_matrix, s.shape_matrix, rtol=1e-6)
    assert np.allclose(back.frequency, s.frequency, rtol=1e-6)
    assert np.allclose(back.damping, s.damping, rtol=1e-5)
    assert np.allclose(back.modal_mass, s.modal_mass, rtol=1e-6), (
        'complex modal mass survives whole')
    assert np.allclose(back.modal_damping, s.modal_damping, rtol=1e-5)
    assert back.comment == ['one', 'two']
