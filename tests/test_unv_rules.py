"""The universal file read as its layout says, not only as our writer
writes it (2026-10-11).

`tools/mutate.py` over unv.py caught 252 of 546 against test_unv.py
and test_unv_complex_modes.py. A round trip cannot see a column read
one place over when the writer writes it one place over too, nor a
code in a table both sides share; and our writer writes every
function unevenly spaced, so the even-spacing form other tools write
was never read at all. The datasets here are typed by hand from the
record layouts, the code tables are stated from the format's own
lists, and the exponents dataset 58 converts by are derived from the
quantities' dimensions rather than restated.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.io import unv
from visualdynamics.units import unit_registry

END = '    -1\n'

#: dataset 58's specific data types, each with the SI unit of the
#: quantity it names
SI_OF = {'pressure': 'Pa', 'length': 'm', 'force': 'N', 'velocity': 'm/s',
         'acceleration': 'm/s**2', 'mass': 'kg', 'time': 's', 'frequency': 'Hz'}


def test_the_58_exponents_are_each_quantitys_length_and_force_powers():
    """In UFF terms a quantity is length^a force^b (time never rescales
    between its unit systems): mass is force s^2 / length, so b is the
    mass exponent and a the length exponent less it."""
    ureg = unit_registry()
    for code, (dim, lexp, fexp) in unv._58_DATA_TYPES.items():
        if dim not in SI_OF:
            assert (lexp, fexp) == (0, 0), code
            continue
        powers = ureg.Quantity(1.0, SI_OF[dim]).dimensionality
        assert (lexp, fexp) == (powers['[length]'] - powers['[mass]'],
                                powers['[mass]']), (code, dim)


def test_the_code_tables_are_the_formats_own():
    assert {code: dim for code, (dim, _l, _f) in unv._58_DATA_TYPES.items()} == {
        0: 'unknown', 1: None, 2: 'pressure', 3: 'dimensionless', 8: 'length',
        9: 'force', 11: 'velocity', 12: 'acceleration', 13: 'force',
        15: 'pressure', 16: 'mass', 17: 'time', 18: 'frequency'}
    assert unv._58_NAMES == {1: 'time', 4: 'frf', 6: 'coherence', 9: 'psd',
                             12: 'spectrum', 24: 'srs', 26: 'multiple_coherence'}
    # bytes per stored number: real and complex single, real and complex double
    assert unv._58_PRECISION == {2: 4, 4: 8, 5: 4, 6: 8}
    assert (unv.LITTLE_ENDIAN, unv.BIG_ENDIAN) == (1, 2)
    assert unv._FP_FORMATS == {1: 'DEC VMS', 2: 'IEEE 754', 3: 'IBM 5/370'}
    assert {name: code for name, (code, _d) in unv._164_SYSTEMS.items()} == {
        'm-kg-N-s': 1, 'mm-kg-N-s': 10, 'in-slinch-lbf-s': 7, 'ft-slug-lbf-s': 2}
    assert unv._BEAM_DESCRIPTORS == {11, 21, 22, 23, 24}


def _record_6(function, node, direction, reference=0, reference_direction=0):
    # I5,I10,I5,I10,1X,A10,I10,I4,1X,A10,I10,I4
    return (f'{function:5d}{1:10d}{0:5d}{0:10d} {"NONE":<10}{node:10d}{direction:4d}'
            f' {"NONE":<10}{reference:10d}{reference_direction:4d}\n')


def _axis(code):
    # I10,3I5,2(1X,10A1)
    return f'{code:10d}{0:5d}{0:5d}{0:5d} {"":<10} {"":<10}\n'


def _58(function, record_6, form, numerator, denominator, values, abscissa=17):
    ids = ''.join(f'{text}\n' for text in ('a record', 'NONE', 'NONE', 'NONE', 'NONE'))
    body = ''.join(f'{v:13.5E}' for v in values) + '\n'
    return (END + '    58\n' + ids + record_6 + form + _axis(abscissa)
            + _axis(numerator) + _axis(denominator) + _axis(0) + body + END)


def test_an_evenly_spaced_function_starts_at_its_minimum(tmp_path):
    """Record 7: real single, four values, even, from 5 s by 0.5 s — and
    a node number that fills its whole ten-column field."""
    form = f'{2:10d}{4:10d}{1:10d}{5.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'
    path = tmp_path / 'even.unv'
    path.write_text(_58(1, _record_6(1, 1234567890, 3), form, 12, 0,
                        [1.0, 2.0, 3.0, 4.0]))
    history = visualdynamics.import_file(path)
    assert np.allclose(history.abscissa, [5.0, 5.5, 6.0, 6.5])
    assert np.allclose(history.ordinate[0], [1.0, 2.0, 3.0, 4.0])
    assert history.response_dof == ['1234567890Z+']
    assert history.known_dim(0) == 'acceleration'


def test_an_frf_reads_its_reference_from_its_own_columns(tmp_path):
    """Complex single, even, an acceleration over a force between two
    ten-digit nodes, the reference's direction negative."""
    form = f'{5:10d}{2:10d}{1:10d}{1.0:13.5E}{1.0:13.5E}{0.0:13.5E}\n'
    path = tmp_path / 'frf.unv'
    path.write_text(_58(4, _record_6(4, 1000000001, 1, 2000000002, -2), form, 12, 13,
                        [1.0, 2.0, 3.0, -4.0], abscissa=18))
    frf = visualdynamics.import_file(path)
    assert frf.response_dof == ['1000000001X+'] and frf.reference_dof == ['2000000002Y-']
    assert np.allclose(frf.ordinate[0], [1 + 2j, 3 - 4j])
    assert frf.known_dim(0) == 'acceleration/force'


def test_an_frf_in_inch_units_converts_by_its_quotients_exponents(tmp_path):
    """(in/s^2)/lbf is length^1 force^-1 in the file's units, so one of
    them is 0.0254 / 4.4482216 SI."""
    form = f'{2:10d}{1:10d}{1:10d}{1.0:13.5E}{1.0:13.5E}{0.0:13.5E}\n'
    units = unv._units_dataset(visualdynamics.units.SYSTEMS['in-slinch-lbf-s'])
    path = tmp_path / 'inch.unv'
    path.write_text(units + '\n' + _58(4, _record_6(4, 1, 3, 2, 3), form, 12, 13,
                                       [1.0], abscissa=18))
    frf = visualdynamics.import_file(path)
    assert frf.ordinate_dim == ['acceleration/force']
    assert frf.ordinate[0, 0].real == pytest.approx(0.0254 / 4.4482216152605)


def _55(definition, record_7, record_8, nodes):
    ids = ''.join(f'{text}\n' for text in ('a mode', 'NONE', 'NONE', 'NONE', 'NONE'))
    body = ''.join(f'{node:10d}\n' + ''.join(f'{v:13.5E}' for v in values) + '\n'
                   for node, values in nodes)
    return END + '    55\n' + ids + definition + record_7 + record_8 + body + END


def test_a_normal_mode_with_a_short_parameter_record_takes_the_defaults(tmp_path):
    """Record 8 holding the frequency alone: unit modal mass, no damping.
    Characteristic 2 is a three-component vector whatever the count
    field says."""
    definition = f'{1:10d}{2:10d}{2:10d}{8:10d}{2:10d}{1:10d}\n'
    record_7 = f'{2:10d}{4:10d}{1:10d}{1:10d}\n'
    path = tmp_path / 'mode.unv'
    path.write_text(_55(definition, record_7, f'{12.5:13.5E}\n',
                        [(101, [1.0, 2.0, 3.0])]))
    shapes = visualdynamics.import_file(path)
    assert shapes.frequency.tolist() == [12.5] and shapes.damping.tolist() == [0.0]
    assert np.real(shapes.modal_mass).tolist() == [1.0]
    assert sorted(shapes.coordinate) == ['101X+', '101Y+', '101Z+']
    assert [shapes.shape_matrix[0, shapes.coordinate.index(dof)]
            for dof in ('101X+', '101Y+', '101Z+')] == [1.0, 2.0, 3.0]


def test_a_complex_mode_reads_its_pole_and_a_missing_modal_a_as_one(tmp_path):
    """Analysis type 3: an eigenvalue of magnitude 2 pi 10 whose real
    part is -0.06 of it is 10 Hz at 6 % of critical; a modal A written
    as zero is read as one, what the synthesis assumes of a set with
    no scaling."""
    omega = 2.0 * np.pi * 10.0
    eigenvalue = complex(-0.06 * omega, omega * np.sqrt(1 - 0.06 ** 2))
    definition = f'{1:10d}{3:10d}{2:10d}{8:10d}{5:10d}{3:10d}\n'
    record_7 = f'{2:10d}{6:10d}{1:10d}{1:10d}\n'
    record_8 = ''.join(f'{v:13.5E}' for v in (eigenvalue.real, eigenvalue.imag,
                                              0.0, 0.0, 0.0, 0.0)) + '\n'
    path = tmp_path / 'complex.unv'
    path.write_text(_55(definition, record_7, record_8,
                        [(7, [1.0, 0.5, 0.0, 0.0, 0.0, -1.0])]))
    shapes = visualdynamics.import_file(path)
    assert shapes.frequency[0] == pytest.approx(10.0, rel=1e-5)
    assert shapes.damping[0] == pytest.approx(0.06, rel=1e-4)
    assert shapes.modal_mass[0] == 1.0
    column = shapes.coordinate.index('7X+')
    assert shapes.shape_matrix[0, column] == pytest.approx(1.0 + 0.5j)


@pytest.mark.parametrize('complex_set', [False, True])
def test_a_written_mode_states_its_kind_in_the_formats_fields(tmp_path, complex_set):
    """Record 6: model type 1; analysis type 2, a normal mode, or 3, a
    complex eigenvalue; characteristic 2, a three-component vector;
    specific data type 8, displacement; data type 2 real or 5 complex;
    three values a node. Record 7: two integers and four reals (six for
    a complex mode), load case 1, the mode's number."""
    shape = np.array([[1.0, 0.5], [0.2, 1.0]])
    shapes = visualdynamics.ShapeSet([10.0, 20.0], [0.01, 0.02], ['1Z+', '2Z+'],
                                     shape * (1 + 0.1j) if complex_set else shape)
    path = tmp_path / 'modes.unv'
    unv.save(shapes, path)
    blocks = [records for number, records, _payload
              in unv.iter_datasets(path.read_bytes()) if number == 55]
    assert len(blocks) == 2
    for mode, records in enumerate(blocks):
        assert [int(v) for v in records[5].split()] == (
            [1, 3, 2, 8, 5, 3] if complex_set else [1, 2, 2, 8, 2, 3])
        assert [int(v) for v in records[6].split()] == (
            [2, 6, 1, mode + 1] if complex_set else [2, 4, 1, mode + 1])


def test_coordinate_systems_read_as_the_layout_lays_them_out(tmp_path):
    """Dataset 2420 as the reader takes it: a part uid and name, then
    per system its label, type and color, its name, and four rows —
    three axes and an origin. Dataset 2411 names each node's
    definition and displacement systems."""
    rows = {1: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
            7: [[0, 1, 0], [-1, 0, 0], [0, 0, 1], [1, 2, 3]]}
    body = f'{1:10d}\nPART\n'
    for label, kind, color, name in ((1, 0, 1, 'Global'), (7, 1, 5, 'Cylinder')):
        body += f'{label:10d}{kind:10d}{color:10d}\n{name}\n'
        body += ''.join(''.join(f'{v:25.16E}' for v in row) + '\n'
                        for row in rows[label])
    nodes = (f'{1:10d}{1:10d}{7:10d}{11:10d}\n'
             + ''.join(f'{v:25.16E}' for v in (1.0, 2.0, 3.0)) + '\n')
    path = tmp_path / 'systems.unv'
    path.write_text(END + '  2420\n' + body + END + END + '  2411\n' + nodes + END)
    geometry = visualdynamics.import_file(path)
    assert geometry.cs_id.tolist() == [1, 7]
    assert geometry.cs_type.tolist() == [0, 1]
    assert list(geometry.cs_name) == ['Global', 'Cylinder']
    assert np.array_equal(geometry.cs_matrix[1], rows[7])
    assert (geometry.node_def_cs.tolist(), geometry.node_disp_cs.tolist()) == ([1], [7])
    assert geometry.node_xyz.tolist() == [[1.0, 2.0, 3.0]]
