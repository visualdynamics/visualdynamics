import os
import pathlib

import numpy as np
from conftest import fixture_path

import visualdynamics

AIRPLANE_UNV = fixture_path('plate', 'geometry.unv')
AIRPLANE_NPZ = fixture_path('plate', 'geometry.npz')
FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'units_and_elements.unv')
FRF_UNV = fixture_path('plate', 'frfs.unv')


def test_unv_without_164_is_unitless():
    """No units dataset: import raw and let the user define units later."""
    geo = visualdynamics.import_file(AIRPLANE_UNV)
    assert not geo.units_defined
    geo.define_units('m')
    declared = visualdynamics.import_file(AIRPLANE_UNV, length_unit='m')
    assert np.allclose(geo.node_xyz, declared.node_xyz)


def test_unv_matches_npz():
    """sdynpy's UNV export of the survey must agree with its .npz export."""
    a = visualdynamics.import_file(AIRPLANE_UNV, length_unit='m')
    b = visualdynamics.import_file(AIRPLANE_NPZ, length_unit='m')
    assert np.array_equal(np.sort(a.node_id), np.sort(b.node_id))
    order_a, order_b = np.argsort(a.node_id), np.argsort(b.node_id)
    assert np.allclose(a.node_xyz[order_a], b.node_xyz[order_b])
    ours = [line['chains'] for line in a.drawn_lines()]
    theirs = [line['chains'] for line in b.drawn_lines()]
    assert ours == theirs


def test_unv_units_from_dataset_164():
    geo = visualdynamics.import_file(FIXTURE)  # no length_unit needed: file has a 164
    lo, hi = geo.extent
    # 10 inch cube footprint -> 0.254 m in SI storage
    assert np.allclose(hi - lo, [0.254, 0.254, 0.254])


def test_unv_beam_element_orientation_record_skipped():
    geo = visualdynamics.import_file(FIXTURE)
    # the 2412 quad and beam, then the 82 line's two segments
    assert len(geo.elem_conn) == 4
    quad, beam = geo.elem_conn[:2]
    assert np.array_equal(quad, [1, 2, 3, 4])
    assert np.array_equal(beam, [5, 6])
    assert geo.elem_type.tolist() == [94, 21, 21, 21]
    # and the 2412 elements landed in blocks of their own families
    assert int(geo.elem_block[0]) != int(geo.elem_block[1])


def test_unv_traceline_pen_up_is_one_block_of_two_runs():
    """An 82 line that lifts the pen is one block of two-node line
    elements holding both runs (2026-09-30), read back as the runs."""
    geo = visualdynamics.import_file(FIXTURE)
    lines = geo.drawn_lines()
    # the 2412 beam carries no properties either, so it is a drawn
    # line too; the 82 line is the block holding two runs
    runs = [line for line in lines if len(line['chains']) == 2]
    assert len(runs) == 1
    assert runs[0]['chains'] == [[1, 2], [3, 4]]


def test_unv_explicit_unit_overridden_by_164():
    """A file-carried 164 wins over nothing; without it declared units apply."""
    geo = visualdynamics.import_file(FIXTURE, length_unit='m')
    lo, hi = geo.extent
    assert np.allclose(hi - lo, [0.254, 0.254, 0.254])


def _without_164(source, tmp_path):
    """A copy of a UNV with its units dataset removed.

    Plenty of real files have no 164 — it is optional — and the point of a
    dimension hint is what such a file can still tell us.
    """
    import re

    text = pathlib.Path(source).read_text(encoding='utf-8')
    stripped = re.sub(r'    -1\n   164\n.*?    -1\n', '', text, flags=re.DOTALL)
    assert stripped != text, 'fixture was supposed to have a 164'
    path = tmp_path / 'no_164.unv'
    path.write_text(stripped)
    return str(path)


def test_a_58_names_its_quantity_even_with_no_164(tmp_path):
    """Data type 12 says 'acceleration' whether or not anything sizes it."""
    frf = visualdynamics.import_file(_without_164(FRF_UNV, tmp_path))
    assert not frf.units_defined, 'nothing sized it, so nothing is defined'
    assert frf.ordinate_dim == ['unknown'] * frf.num_records
    assert frf.dimension_hint == ['acceleration/force'] * frf.num_records
    assert frf.known_dim(0) == 'acceleration/force'


def test_a_hint_scales_nothing(tmp_path):
    """The quantity is known and the scale is not, so the values stay raw."""
    path = _without_164(FRF_UNV, tmp_path)
    hinted = visualdynamics.import_file(path)
    declared = visualdynamics.import_file(path, ordinate_unit='g')
    assert np.allclose(hinted.ordinate * 9.80665, declared.ordinate)


def test_the_quantity_survives_a_unv_round_trip(tmp_path):
    """A 58 can name a quantity without a 164 to size it, so writing one
    back loses nothing we knew — before, it went out as data type 0."""
    frf = visualdynamics.import_file(_without_164(FRF_UNV, tmp_path))
    out = str(tmp_path / 'out.unv')
    visualdynamics.export_file(frf, out, format='unv')
    assert '\n   164\n' not in pathlib.Path(out).read_text(encoding='utf-8'), 'nothing to size'
    assert visualdynamics.import_file(out).dimension_hint == frf.dimension_hint


# ---- the parabolic solids, and the element UFF never had -----------------


def _solid_mesh(with_pyramid=False):
    """Thirteen nodes: a tet10 and a wedge15 sharing them loosely, and
    optionally a pyramid5 — the element exodus carries and UFF 2412
    never defined."""
    nodes = list(range(1, 16))
    geometry = visualdynamics.Geometry(
        node_id=nodes, node_xyz=np.random.default_rng(5).normal(size=(15, 3)),
        elem_id=[1, 2] + ([3] if with_pyramid else []),
        elem_type=[118, 113] + ([201] if with_pyramid else []),
        elem_color=[1, 2] + ([3] if with_pyramid else []),
        elem_conn=[np.arange(1, 11), np.arange(1, 16)]
                  + ([np.arange(1, 6)] if with_pyramid else []))
    geometry.define_units('m')
    return geometry


def test_tet10_and_wedge15_round_trip_the_universal_file(tmp_path):
    """The most common solid element in modern FEA (caught missing
    2026-08-21 while auditing element coverage): a tet-meshed model
    refused outright. UFF 2412 codes 118 and 113."""
    from visualdynamics.io import unv

    geometry = _solid_mesh()
    unv.save(geometry, str(tmp_path / 'solids.unv'))
    back = visualdynamics.import_file(str(tmp_path / 'solids.unv'),
                                      length_unit='m')
    assert list(back.elem_type) == [118, 113]
    assert [len(c) for c in back.elem_conn] == [10, 15]
    assert np.array_equal(back.elem_conn[0], np.arange(1, 11))


def test_a_pyramid_cannot_pretend_to_be_a_universal_element(tmp_path):
    """UFF 2412 has no pyramid descriptor — I-DEAS meshed without
    them — so the writer refuses by name instead of emitting a code
    no other reader could interpret."""
    import pytest

    from visualdynamics.io import unv

    geometry = _solid_mesh(with_pyramid=True)
    with pytest.raises(ValueError, match='pyramid'):
        unv.save(geometry, str(tmp_path / 'refused.unv'))


# ---- dataset 58b, the binary form ---------------------------------------


def _functions():
    """Four complex FRFs, values that five decimal places cannot hold."""
    from visualdynamics.core.data import Frf

    rng = np.random.default_rng(20260825)
    frequency = np.linspace(0.0, 500.0, 257)
    ordinate = (rng.normal(size=(4, 257)) + 1j * rng.normal(size=(4, 257)))
    return Frf(frequency, ordinate,
               response_dof=[f'{101 + i}Z+' for i in range(4)],
               reference_dof=['1X+'] * 4,
               ordinate_dim='acceleration/force',
               ordinate_unit='m/s**2', reference_unit='N')


def test_the_binary_form_round_trips_exactly(tmp_path):
    """58b writes the values as IEEE 754 doubles, so what comes back is
    the float that went in. The ASCII form cannot say that: 13.5E is
    five decimal places, and this is the difference the binary form
    exists for."""
    from visualdynamics.io import unv

    frf = _functions()
    unv.save(frf, tmp_path / 'ascii.unv')
    unv.save(frf, tmp_path / 'binary.unv', binary=True)

    exact = unv.load(tmp_path / 'binary.unv')
    rounded = unv.load(tmp_path / 'ascii.unv')
    assert np.array_equal(exact.ordinate, frf.ordinate), \
        'the binary form is the values themselves, not a rendering of them'
    assert np.array_equal(exact.abscissa, frf.abscissa)
    assert not np.array_equal(rounded.ordinate, frf.ordinate), \
        'and the ASCII form is not — if it were, this proves nothing'
    assert np.allclose(rounded.ordinate, frf.ordinate, atol=1e-4)
    # everything the header carries is untouched by the change of form
    assert list(exact.response_dof) == list(frf.response_dof)
    assert list(exact.reference_dof) == list(frf.reference_dof)
    assert list(exact.ordinate_unit) == list(frf.ordinate_unit)
    assert (tmp_path / 'binary.unv').stat().st_size \
        < (tmp_path / 'ascii.unv').stat().st_size, 'and it is smaller'


def test_a_binary_payload_holding_a_delimiter_is_not_cut_in_half(tmp_path):
    """The bytes of a float can spell a '-1' delimiter line, and a
    reader that split the file into lines before understanding it would
    end a dataset in the middle of the data. The byte count on the
    marker line is what the payload is taken by, and it is never
    scanned."""
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.io import unv

    # \n    -1\n as float64 bytes, embedded in an otherwise ordinary
    # record: b'\n    -1\n' is exactly eight bytes, which is one double
    trap = np.frombuffer(b'\n    -1\n', dtype='<f8')[0]
    ordinate = np.array([1.0, trap, 2.0, trap, 3.0])
    history = TimeHistory(np.arange(5.0), [ordinate],
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration',
                          ordinate_unit='m/s**2')
    unv.save(history, tmp_path / 'trap.unv', binary=True)
    assert b'\n    -1\n' in (tmp_path / 'trap.unv').read_bytes(), \
        'the file really does contain the trap this test is about'

    back = unv.load(tmp_path / 'trap.unv')
    assert np.array_equal(np.real(back.ordinate[0]), ordinate)


def _rebuilt_58b(path, order, fp_format, swap=False):
    """The 58b dataset in `path`, written out again declaring `order`
    and `fp_format` — and byte-swapped to match when asked.

    Built through `iter_datasets` rather than by slicing the file at
    fixed columns: the point is to exercise the reader against a file
    it did not write, not to test the test's own arithmetic (which is
    what broke first).
    """
    from visualdynamics.io import unv

    number, records, payload = next(
        (n, r, p) for n, r, p in unv.iter_datasets(path.read_bytes())
        if p is not None)
    assert number == 58, 'the binary dataset is a 58'
    blob = payload.data
    if swap:
        blob = np.frombuffer(blob, dtype='<f8').astype('>f8').tobytes()
        assert blob != payload.data, 'the swap actually changed the bytes'
    marker = (f'{58:6d}b{order:6d}{fp_format:6d}'
              f'{len(records):12d}{len(blob):12d}'
              f'{0:6d}{0:6d}{0:12d}{0:12d}\n')
    return (('    -1\n' + marker + '\n'.join(records) + '\n').encode()
            + blob + b'\n    -1\n')


def test_a_big_endian_binary_file_reads(tmp_path):
    """The marker line declares the byte order and the reader obeys it.
    Written on a big-endian machine, or by a tool that chose to, the
    same file must read the same numbers."""
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.io import unv

    values = np.array([1.5, -2.25, 3.125, 4.0])
    history = TimeHistory(np.arange(4.0), [values],
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration',
                          ordinate_unit='m/s**2')
    unv.save(history, tmp_path / 'little.unv', binary=True)
    (tmp_path / 'big.unv').write_bytes(
        _rebuilt_58b(tmp_path / 'little.unv', unv.BIG_ENDIAN, unv.IEEE_754,
                     swap=True))

    back = unv.load(tmp_path / 'big.unv')
    assert np.array_equal(np.real(back.ordinate[0]), values)


def test_a_float_format_we_cannot_decode_is_refused_by_name(tmp_path):
    """DEC VMS and IBM 5/370 floats are different representations, not
    different byte orders. Read as IEEE they would come back as
    plausible wrong numbers, so the reader says which format it found
    and stops."""
    import pytest

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.io import unv

    history = TimeHistory(np.arange(4.0), [np.arange(4.0)],
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration',
                          ordinate_unit='m/s**2')
    unv.save(history, tmp_path / 'ieee.unv', binary=True)
    (tmp_path / 'ibm.unv').write_bytes(
        _rebuilt_58b(tmp_path / 'ieee.unv', unv.LITTLE_ENDIAN, 3))

    with pytest.raises(ValueError, match='IBM 5/370'):
        unv.load(tmp_path / 'ibm.unv')


def test_binary_functions_beside_ascii_geometry_are_not_dropped(tmp_path):
    """The shape a real export takes: geometry in ASCII, functions in
    58b, one file. Before the binary form was understood this loaded
    the geometry and threw the functions away without a word — the
    silent half of the loss, and the reason this is a test rather than
    a note (Brandon, 2026-08-25)."""
    from visualdynamics.core.geometry import Geometry
    from visualdynamics.io import unv

    geometry = Geometry([1, 2], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    frf = _functions()
    unv.save(geometry, tmp_path / 'geo.unv')
    unv.save(frf, tmp_path / 'fn.unv', binary=True)
    (tmp_path / 'both.unv').write_bytes((tmp_path / 'geo.unv').read_bytes()
                                        + (tmp_path / 'fn.unv').read_bytes())

    back = unv.load(tmp_path / 'both.unv')
    assert isinstance(back, dict) and set(back) == {'geometry', 'frf'}, \
        f'both objects came back, not just the ASCII one: {back}'
    assert back['geometry'].num_nodes == 2
    assert np.array_equal(back['frf'].ordinate, frf.ordinate)


def test_the_binary_form_is_offered_as_its_own_export(tmp_path):
    """Anything the API can do the Export menu can do (PRINCIPLES.md,
    4). Choosing by suffix still gets the ASCII form, which is the one
    every reader takes."""
    import visualdynamics as vd
    from visualdynamics.io import exporters

    frf = _functions()
    names = [e.name for e in exporters(frf)]
    assert 'unv' in names and 'unv_binary' in names

    vd.io.export_file(frf, str(tmp_path / 'by_name.unv'), format='unv_binary')
    vd.io.export_file(frf, str(tmp_path / 'by_suffix.unv'))
    assert b'58b' in (tmp_path / 'by_name.unv').read_bytes()
    assert b'58b' not in (tmp_path / 'by_suffix.unv').read_bytes()
    assert np.array_equal(
        vd.import_file(str(tmp_path / 'by_name.unv')).ordinate, frf.ordinate)
