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

import struct

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
        0: 'unknown', 1: None, 2: 'pressure', 3: 'dimensionless',
        5: 'temperature', 8: 'length',
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


# ---- binary 58b as the format lays it out, and temperature ---------------

def _58b(function, form, numerator, blob):
    """A binary dataset 58: typed by hand, little endian and IEEE 754."""
    ids = ''.join(f'{text}\n' for text in ('a record', 'NONE', 'NONE', 'NONE',
                                           'NONE'))
    marker = f'{58:6d}b{1:6d}{2:6d}{11:12d}{len(blob):12d}{0:6d}{0:6d}{0:12d}{0:12d}\n'
    header = (ids + _record_6(function, 1, 3) + form + _axis(17)
              + _axis(numerator) + _axis(0) + _axis(0))
    return (END + marker + header).encode() + blob + b'\n' + END.encode()


def _uneven(ordinate_type, count):
    return f'{ordinate_type:10d}{count:10d}{0:10d}{0.0:13.5E}{0.0:13.5E}{0.0:13.5E}\n'


T = np.array([0.0, 0.5, 1.25, 3.0])
Y = np.array([20.0, 21.5, 19.25, 22.75])


def test_binary_double_uneven_reads_the_formats_single_precision_abscissa(
        tmp_path):
    """Case 6: "The abcissa is always real single precision", so each
    point is a 4-byte X and an 8-byte Y. ASCII hides this; read as all
    doubles the binary form refused to import."""
    blob = b''.join(struct.pack('<fd', x, y) for x, y in zip(T, Y))
    path = tmp_path / 'mixed.uff'
    path.write_bytes(_58b(1, _uneven(4, 4), 12, blob))
    history = visualdynamics.import_file(path)
    assert np.array_equal(history.abscissa, T)
    assert np.array_equal(history.ordinate[0], Y)


def test_binary_complex_double_uneven_reads_the_formats_layout(tmp_path):
    """Case 8: X, then the real and imaginary parts, as doubles."""
    blob = b''.join(struct.pack('<fdd', x, y, -y) for x, y in zip(T, Y))
    path = tmp_path / 'mixed.uff'
    path.write_bytes(_58b(12, _uneven(6, 4), 12, blob))
    spectrum = visualdynamics.import_file(path)
    assert np.array_equal(spectrum.abscissa, T)
    assert np.array_equal(spectrum.ordinate[0], Y - 1j * Y)


def test_binary_double_uneven_still_reads_an_all_double_abscissa(tmp_path):
    """Not the format's layout, but what this module's `save` and other
    writers produce: the byte count tells the two apart."""
    blob = np.column_stack([T, Y]).astype('<f8').tobytes()
    path = tmp_path / 'doubles.uff'
    path.write_bytes(_58b(1, _uneven(4, 4), 12, blob))
    history = visualdynamics.import_file(path)
    assert np.array_equal(history.abscissa, T)
    assert np.array_equal(history.ordinate[0], Y)


def test_binary_double_even_is_doubles_with_no_abscissa_at_all(tmp_path):
    """Even spacing stores no X, so the single-precision X of the uneven
    cases must not be read into it, though its byte count is just as
    short of the all-double uneven size."""
    form = f'{4:10d}{4:10d}{1:10d}{0.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'
    path = tmp_path / 'even.uff'
    path.write_bytes(_58b(1, form, 12, Y.astype('<f8').tobytes()))
    history = visualdynamics.import_file(path)
    assert np.allclose(history.abscissa, [0.0, 0.5, 1.0, 1.5])
    assert np.array_equal(history.ordinate[0], Y)


def _164(temperature, offset):
    fields = (1.0, 1.0, temperature)
    return (END + '   164\n' + f'{1:10d}{"SI":<20}{2:10d}\n'
            + ''.join(f'{v:25.17E}'.replace('E', 'D') for v in fields) + '\n'
            + f'{offset:25.17E}'.replace('E', 'D') + '\n' + END)


@pytest.mark.parametrize('binary', [False, True], ids=['ascii', 'binary'])
def test_a_temperature_with_no_units_stated_is_named_and_left_raw(tmp_path,
                                                                 binary):
    """Without a 164 the scale is unstated: as kelvin, 20 degC would be
    20 K, and as dimensionless the file's word for it would be lost."""
    path = tmp_path / 'temperature.uff'
    if binary:
        path.write_bytes(_58b(1, _uneven(4, 4), 5,
                              np.column_stack([T, Y]).astype('<f8').tobytes()))
    else:
        form = f'{2:10d}{4:10d}{1:10d}{0.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'
        path.write_text(_58(1, _record_6(1, 1, 3), form, 5, 0, Y))
    history = visualdynamics.import_file(path)
    assert np.allclose(history.ordinate[0], Y)
    assert history.ordinate_dim[0] == 'unknown'
    assert history.known_dim(0) == 'temperature'


def test_a_temperature_converts_by_the_164s_factor_and_offset(tmp_path):
    """Degrees Celsius: factor 1, offset 273.15. An offset is the reason
    the length and force exponents cannot say this."""
    form = f'{2:10d}{4:10d}{1:10d}{0.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'
    path = tmp_path / 'celsius.uff'
    path.write_text(_164(1.0, 273.15) + _58(1, _record_6(1, 1, 3), form, 5, 0, Y))
    history = visualdynamics.import_file(path)
    assert history.ordinate_dim[0] == 'temperature'
    assert np.allclose(history.ordinate[0], Y + 273.15)


def test_a_temperature_in_degrees_fahrenheit_converts_by_the_factor_too(
        tmp_path):
    """K = (file + offset) / factor: 32 degF is 273.15 K."""
    form = f'{2:10d}{1:10d}{1:10d}{0.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'
    path = tmp_path / 'fahrenheit.uff'
    path.write_text(_164(1.8, 459.67) + _58(1, _record_6(1, 1, 3), form, 5, 0,
                                            [32.0]))
    history = visualdynamics.import_file(path)
    assert history.ordinate[0, 0] == pytest.approx(273.15)


def test_a_declared_unit_sizes_a_temperature_with_its_offset(tmp_path):
    form = f'{2:10d}{1:10d}{1:10d}{0.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'
    path = tmp_path / 'declared.uff'
    path.write_text(_58(1, _record_6(1, 1, 3), form, 5, 0, [20.0]))
    history = unv.load(path, ordinate_unit='degC')
    assert history.ordinate_dim[0] == 'temperature'
    assert history.ordinate[0, 0] == pytest.approx(293.15)


def test_a_temperature_is_written_with_its_code_and_comes_back(tmp_path):
    """Through this module's own writer, ASCII and binary, by the 164."""
    from visualdynamics.core.data import TimeHistory

    history = TimeHistory(T, np.array([Y + 273.15]), response_dof=['1Z+'],
                          ordinate_dim='temperature')
    for name, binary in (('a.unv', False), ('b.unv', True)):
        path = tmp_path / name
        unv.save(history, path, binary=binary)
        back = unv.load(path)
        assert back.ordinate_dim[0] == 'temperature'
        assert np.allclose(back.ordinate[0], history.ordinate[0])
        records = next(r for n, r, _p in unv.iter_datasets(path.read_bytes())
                       if n == 58)
        assert int(records[8][:10]) == 5


def test_the_refusal_says_which_function_type_it_passed_over(tmp_path):
    form = f'{2:10d}{1:10d}{1:10d}{0.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'
    path = tmp_path / 'general.uff'
    path.write_text(_58(0, _record_6(0, 1, 3), form, 5, 0, [20.0]))
    with pytest.raises(ValueError, match=r'function type 0 \(read: 1, 4, '):
        unv.load(path)


def test_the_refusal_quotes_a_dataset_line_it_could_not_read(tmp_path):
    """A binary marker this module does not recognize is dropped whole,
    which from outside looked like a file with nothing in it."""
    path = tmp_path / 'marker.uff'
    path.write_bytes(END.encode() + b'    58B\nnot read\n' + END.encode())
    with pytest.raises(ValueError, match=r"first being '58B'"):
        unv.load(path)


def _bare_58b(blob, form, numerator=5, end=b'\n', node=8001, function=1):
    """A 58b whose marker stops at the `b`, laid out as a Data Physics
    file's is: the ID lines, then record 6 with a node and direction 0."""
    ids = ('Data Physics Corporation\n\nNONE\n'
           f'{node}\nSensitivity = 10.307 mV/EU : \n')
    record_6 = (f'{function:5d}{0:10d}{1:5d}{0:10d} {"":<10}{node:10d}{0:4d}'
                f' {"NONE":<10}{0:10d}{0:4d}\n')
    header = ids + record_6 + form + _axis(17) + _axis(numerator) + _axis(0) + _axis(0)
    return (END + '    58b\n' + header).encode() + blob + end + END.encode()


def _even(ordinate_type, count):
    return f'{ordinate_type:10d}{count:10d}{1:10d}{0.0:13.5E}{0.5:13.5E}{0.0:13.5E}\n'


@pytest.mark.parametrize('ordinate_type, dtype', [(2, '<f4'), (4, '<f8')],
                         ids=['single', 'double'])
def test_a_bare_marker_reads_by_record_7(tmp_path, ordinate_type, dtype):
    """Several datasets in a row, since only the size says where one
    stops, with values holding the bytes of a newline and of a -1 line."""
    awkward = np.frombuffer(b'\n    -1\n' + bytes(24), dtype='<f8')[:4]
    first = Y if ordinate_type == 2 else awkward
    second = Y * 2
    path = tmp_path / 'bare.uff'
    path.write_bytes(
        _bare_58b(np.asarray(first, dtype).tobytes(), _even(ordinate_type, 4))
        + _bare_58b(np.asarray(second, dtype).tobytes(), _even(ordinate_type, 4),
                    node=8002))
    history = visualdynamics.import_file(path)
    assert history.response_dof == ['8001', '8002']
    assert np.allclose(history.abscissa, [0.0, 0.5, 1.0, 1.5])
    assert np.array_equal(history.ordinate[0], np.asarray(first, dtype))
    assert np.array_equal(history.ordinate[1], np.asarray(second, dtype))
    assert history.known_dim(0) == 'temperature'


def test_a_bare_marker_does_not_need_a_newline_after_its_values(tmp_path):
    path = tmp_path / 'bare.uff'
    path.write_bytes(_bare_58b(Y.astype('<f8').tobytes(), _even(4, 4), end=b''))
    assert np.array_equal(visualdynamics.import_file(path).ordinate[0], Y)


def test_a_bare_marker_counts_two_floats_for_a_complex_value(tmp_path):
    values = np.column_stack([Y, -Y]).astype('<f4').tobytes()
    path = tmp_path / 'bare.uff'
    path.write_bytes(_bare_58b(values, _even(5, 4), numerator=12, function=12))
    assert np.array_equal(visualdynamics.import_file(path).ordinate[0], Y - 1j * Y)


def test_a_bare_marker_keeps_the_newline_after_its_values_out_of_the_records():
    data = _bare_58b(Y.astype('<f8').tobytes(), _even(4, 4))
    (number, records, payload), = unv.iter_datasets(data)
    assert number == 58
    assert len(records) == 11
    assert payload.order == unv.LITTLE_ENDIAN
    assert payload.fp_format == unv.IEEE_754


def test_a_bare_marker_ends_at_the_end_of_a_file_as_well(tmp_path):
    whole = _bare_58b(Y.astype('<f8').tobytes(), _even(4, 4))
    path = tmp_path / 'bare.uff'
    path.write_bytes(whole[:-len(END)])
    assert np.array_equal(visualdynamics.import_file(path).ordinate[0], Y)


def test_a_truncated_file_with_a_bare_marker_is_refused(tmp_path):
    whole = _bare_58b(Y.astype('<f8').tobytes(), _even(4, 4))
    path = tmp_path / 'bare.uff'
    path.write_bytes(whole[:-(len(END) + 9)])
    with pytest.raises(ValueError, match='gives no byte count'):
        unv.load(path)


@pytest.mark.parametrize('blob', [
    b''.join(struct.pack('<fd', x, y) for x, y in zip(T, Y)),
    np.column_stack([T, Y]).astype('<f8').tobytes()],
    ids=['single-precision X', 'double-precision X'])
def test_a_bare_marker_finds_the_x_layout_by_the_delimiter(tmp_path, blob):
    path = tmp_path / 'bare.uff'
    path.write_bytes(_bare_58b(blob, _uneven(4, 4)))
    history = visualdynamics.import_file(path)
    assert np.array_equal(history.abscissa, T)
    assert np.array_equal(history.ordinate[0], Y)


def test_a_bare_marker_with_the_wrong_count_is_refused_rather_than_misread(
        tmp_path):
    path = tmp_path / 'bare.uff'
    path.write_bytes(_bare_58b(Y.astype('<f8').tobytes(), _even(4, 7)))
    with pytest.raises(ValueError, match='gives no byte count'):
        unv.load(path)


def test_a_bare_marker_of_another_dataset_is_still_unread(tmp_path):
    """Only dataset 58 has a known header to size it by."""
    path = tmp_path / 'other.uff'
    path.write_bytes(END.encode() + b'  2414b\nnot read\n' + END.encode())
    with pytest.raises(ValueError, match=r"first being '2414b'"):
        unv.load(path)


def test_a_file_with_nothing_in_it_says_only_that(tmp_path):
    path = tmp_path / 'empty.uff'
    path.write_text(END + '  9999\nnot a dataset read here\n' + END)
    with pytest.raises(ValueError, match=r'empty\.uff$'):
        unv.load(path)
