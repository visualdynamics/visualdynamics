
import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics

FRF_NPZ = fixture_path('plate', 'frfs.npz')
FRF_UNV = fixture_path('plate', 'frfs.unv')
TIME_NPZ = fixture_path('plate', 'time.npz')
PSD_NPZ = fixture_path('plate', 'psd.npz')


def test_sdynpy_data_without_units_is_unitless():
    for path in (FRF_NPZ, TIME_NPZ):
        data = visualdynamics.import_file(path)
        assert not data.units_defined
        assert data.undefined_records == list(range(data.num_records))


def test_define_units_per_record():
    """One object can mix units — accel on some channels, force on others."""
    time = visualdynamics.import_file(TIME_NPZ)
    # a channel table can mix units: accelerometers, a load cell, a voltage
    units = ['m/s**2', 'g', 'N', 'V']
    units += ['m/s**2'] * (time.num_records - len(units))
    time.define_units(units)
    assert time.ordinate_dim[:4] == ['acceleration', 'acceleration', 'force',
                                     'voltage']
    assert time.units_defined
    raw = visualdynamics.import_file(TIME_NPZ)
    assert np.allclose(time.ordinate[1], raw.ordinate[1] * 9.80665)
    assert np.allclose(time.ordinate[0], raw.ordinate[0])


def test_define_units_partial_leaves_rest_unknown():
    time = visualdynamics.import_file(TIME_NPZ)
    time.define_units({0: 'g'})
    assert time.ordinate_dim[0] == 'acceleration'
    assert time.undefined_records == list(range(1, time.num_records))
    assert not time.units_defined


def test_define_units_on_frf_needs_both():
    frf = visualdynamics.import_file(FRF_NPZ)
    frf.define_units('m/s**2', 'N')
    assert frf.ordinate_dim == ['acceleration/force'] * frf.num_records
    declared = visualdynamics.import_file(FRF_NPZ, response_unit='m/s**2',
                                reference_unit='N')
    assert np.allclose(frf.ordinate, declared.ordinate)


def test_temperature_units_are_affine():
    """degC/degF need offsets, not just scale factors."""
    time = visualdynamics.import_file(TIME_NPZ)
    raw = visualdynamics.import_file(TIME_NPZ).ordinate[0].copy()
    time.define_units({0: 'degC'})
    assert time.ordinate_dim[0] == 'temperature'
    assert np.allclose(time.ordinate[0], raw + 273.15)
    # and displaying in degF round-trips through the offset correctly
    shown = time.display_ordinate(visualdynamics.IN_LBF_S)[0]
    assert np.allclose(shown, raw * 9 / 5 + 32)


def test_import_sdynpy_frf():
    frf = visualdynamics.import_file(FRF_NPZ, response_unit='m/s**2', reference_unit='N')
    assert isinstance(frf, visualdynamics.Frf)
    # a modal test: every response against each of four drive points
    references = sorted(set(frf.reference_dof))
    responses = sorted(set(frf.response_dof))
    assert references == ['101Z+', '110Z+', '1304Z+', '1310Z+'], references
    assert len(responses) == 25, 'one Z per survey grid node'
    assert frf.num_records == len(responses) * len(references)
    assert frf.ordinate_dim == ['acceleration/force'] * frf.num_records
    assert np.iscomplexobj(frf.ordinate)
    assert frf.abscissa[0] == 2.0 and frf.abscissa[-1] == 2500.0


def test_import_sdynpy_time():
    time = visualdynamics.import_file(TIME_NPZ, ordinate_unit='m/s**2')
    assert isinstance(time, visualdynamics.TimeHistory)
    assert time.num_records == 25, 'the impulse response at every sensor'
    assert time.ordinate_dim == ['acceleration'] * time.num_records
    assert not np.iscomplexobj(time.ordinate)


def test_import_sdynpy_psd():
    psd = visualdynamics.import_file(PSD_NPZ, ordinate_unit='m/s**2')
    assert isinstance(psd, visualdynamics.Psd)
    assert psd.ordinate_dim == ['acceleration**2/frequency'] * psd.num_records
    assert psd.reference_dof == psd.response_dof


def test_unv_58_units_automatic():
    """The UNV file carries a 164 (SI), so no units declaration is needed."""
    frf = visualdynamics.import_file(FRF_UNV)
    assert isinstance(frf, visualdynamics.Frf)
    assert frf.ordinate_dim == ['acceleration/force'] * frf.num_records


def test_unv_58_matches_npz():
    """Same FRFs through two formats must agree exactly, record by
    record — both carry the full four-reference set."""
    a = visualdynamics.import_file(FRF_UNV)
    b = visualdynamics.import_file(FRF_NPZ, response_unit='m/s**2', reference_unit='N')
    assert a.num_records == b.num_records
    pairs = {(res, ref): i for i, (res, ref)
             in enumerate(zip(b.response_dof, b.reference_dof))}
    order = [pairs[pair]
             for pair in zip(a.response_dof, a.reference_dof)]
    assert np.allclose(a.abscissa, b.abscissa)
    # the UNV is ASCII at six significant figures; the tolerance is
    # the format's own, not the reader's
    assert np.allclose(a.ordinate, np.asarray(b.ordinate)[order], rtol=1e-5)


def test_unit_conversion_on_import():
    si = visualdynamics.import_file(FRF_NPZ, response_unit='m/s**2', reference_unit='N')
    gs = visualdynamics.import_file(FRF_NPZ, response_unit='g', reference_unit='lbf')
    assert np.allclose(gs.ordinate, si.ordinate * 9.80665 / 4.4482216152605)


def test_display_ordinate():
    frf = visualdynamics.import_file(FRF_NPZ, response_unit='m/s**2', reference_unit='N')
    disp = frf.display_ordinate(visualdynamics.IN_LBF_S)
    factor = 0.0254 / 4.4482216152605  # (in/s^2)/lbf in SI
    assert np.allclose(disp, frf.ordinate / factor)
    assert visualdynamics.IN_LBF_S.label('acceleration/force') == '(in/s**2)/lbf'


def test_data_save_load_round_trip(tmp_path):
    for obj in [
        visualdynamics.import_file(FRF_NPZ, response_unit='m/s**2', reference_unit='N'),
        visualdynamics.import_file(TIME_NPZ, ordinate_unit='m/s**2'),
        visualdynamics.import_file(PSD_NPZ, ordinate_unit='m/s**2'),
    ]:
        path = str(tmp_path / f'{type(obj).__name__}.vdyn')
        obj.save(path)
        assert visualdynamics.load(path) == obj


def test_frf_requires_reference():
    with pytest.raises(ValueError):
        visualdynamics.Frf(abscissa=[1.0, 2.0], ordinate=[[1j, 2j]],
                 response_dof=['1X+'])


def test_channel_table_round_trip(tmp_path):
    table = visualdynamics.ChannelTable({
        'channel': [1, 2], 'node': ['101', '102'],
        'direction': ['X+', 'Z-'], 'unit': ['m/s^2', 'N'],
        'serial_number': ['A1', 'B2']})
    assert table.dof_strings() == ['101X+', '102Z-']
    path = str(tmp_path / 'table.vdyn')
    table.save(path)
    assert visualdynamics.load(path) == table


def test_known_dimension_always_names_its_unit():
    """No record may claim a dimension but no unit — the GUI's labels and
    badges both key off that state, so it must not disagree with itself."""
    for path, kwargs in [
            (FRF_UNV, {}),
            (FRF_NPZ, {'response_unit': 'm/s**2', 'reference_unit': 'N'}),
            (TIME_NPZ, {'ordinate_unit': 'g'}),
            (PSD_NPZ, {'ordinate_unit': 'g'})]:
        data = visualdynamics.import_file(path, **kwargs)
        for i, dim in enumerate(data.ordinate_dim):
            named = data.ordinate_unit[i] is not None
            assert named == (dim != 'unknown'), (path, i, dim,
                                                 data.ordinate_unit[i])


def test_unv_frf_units_are_si():
    """A UNV carrying a 164 arrives fully unit-aware."""
    frf = visualdynamics.import_file(FRF_UNV)
    assert frf.units_defined
    assert frf.ordinate_unit == ['m/s**2'] * frf.num_records
    assert frf.reference_unit == ['N'] * frf.num_records


def test_filled_units_round_trip_through_raw_values():
    """Redefining after an auto-filled unit reinterprets, not double-scales."""
    frf = visualdynamics.import_file(FRF_UNV)
    original = frf.ordinate.copy()
    frf.define_units('g', 'lbf')          # same data, different declaration
    frf.define_units('m/s**2', 'N')        # back again
    assert np.allclose(frf.ordinate, original)


def test_defining_units_forgets_the_hint():
    """Once the real dimension is known the hint is a second answer to the
    same question, free to drift out of agreement with the first."""
    data = visualdynamics.TimeHistory([0.0, 1.0], [[1.0, 2.0]], ['101X+'],
                            dimension_hint='acceleration')
    assert data.known_dim(0) == 'acceleration'
    data.define_units('N')
    assert data.dimension_hint == [None]
    assert data.known_dim(0) == 'force'


def test_a_hint_of_unknown_is_not_a_hint():
    """A file saying it does not know is the same as one that never said."""
    data = visualdynamics.TimeHistory([0.0, 1.0], [[1.0, 2.0]], ['101X+'],
                            dimension_hint='unknown')
    assert data.dimension_hint == [None]


def test_a_defined_record_never_carries_a_hint():
    data = visualdynamics.TimeHistory([0.0, 1.0], [[1.0, 2.0]], ['101X+'],
                            ordinate_dim='force', dimension_hint='acceleration')
    assert data.dimension_hint == [None], 'the dimension is the answer'


def test_the_hint_survives_a_native_round_trip(tmp_path):
    data = visualdynamics.TimeHistory([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]],
                            ['101X+', '102X+'],
                            dimension_hint=['acceleration', None])
    path = str(tmp_path / 'hint.vdyn')
    data.save(path)
    assert visualdynamics.load(path) == data
    assert visualdynamics.load(path).dimension_hint == ['acceleration', None]


def test_a_hinted_record_is_offered_only_units_of_its_quantity():
    from visualdynamics.core.unit_choices import ALL_ORDINATE_UNITS, units_for

    assert units_for('acceleration') == ['g', 'in/s**2', 'm/s**2', 'mm/s**2']
    assert units_for('acceleration/force') == units_for('acceleration'), \
        "an FRF's ordinate is named by its numerator"
    assert units_for('acceleration**2/frequency') == units_for('acceleration'), \
        'a PSD is declared by the engineering unit whose square it holds'
    assert units_for('unknown') == ALL_ORDINATE_UNITS, 'no claim, no narrowing'
    assert units_for('') == ALL_ORDINATE_UNITS


def test_a_declaration_can_be_withdrawn():
    """A wrong guess must be correctable without reimporting, and there is
    no unit string meaning 'I no longer know'."""
    data = visualdynamics.import_file(TIME_NPZ)
    raw = data.ordinate.copy()
    data.define_units('g')
    assert data.units_defined
    data.undefine_units()
    assert not data.units_defined
    assert data.ordinate_unit == [None] * data.num_records
    assert np.allclose(data.ordinate, raw), 'the file values are back'


def test_withdrawing_takes_only_the_records_asked_for():
    data = visualdynamics.import_file(TIME_NPZ)
    data.define_units('g')
    data.undefine_units([1, 3])
    assert data.undefined_records == [1, 3]


def test_withdrawing_an_undeclared_record_is_harmless():
    data = visualdynamics.import_file(TIME_NPZ)
    raw = data.ordinate.copy()
    data.undefine_units()
    assert np.allclose(data.ordinate, raw)


# ---- what a DOF may be ------------------------------------------------------

def test_a_dof_that_is_wrong_is_refused_where_one_that_is_missing_is_not():
    """The distinction the objects draw: an *omission* imports so the
    user can fix it, a *mistake* does not, because no amount of fixing
    the channel table makes Q an axis or 'not a dof' a measurement."""
    import numpy as np
    import pytest

    from visualdynamics.core.data import TimeHistory

    t = np.arange(4) / 4.0
    values = [[1.0, 2.0, 3.0, 4.0]]
    with pytest.raises(ValueError, match='names no direction'):
        TimeHistory(t, values, response_dof=['101Q+'])
    with pytest.raises(ValueError, match='names no direction'):
        TimeHistory(t, values, response_dof=['not a dof'])
    # and the omissions, which import
    assert TimeHistory(t, values, response_dof=['X+']).response_dof == ['X+']
    assert TimeHistory(t, values, response_dof=['101']).response_dof == ['101']


def test_a_dof_written_by_hand_is_normalized():
    """An unsigned direction is the positive one, which is how people
    write them — and it comes back normalized so everything downstream
    compares like with like."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory

    t = np.arange(4) / 4.0
    history = TimeHistory(t, [[1.0, 2.0, 3.0, 4.0]], response_dof=['7z'])
    assert history.response_dof == ['7Z+']


def test_a_reference_dof_is_held_to_the_same_rule():
    import numpy as np
    import pytest

    from visualdynamics.core.data import Frf

    f = np.arange(4) * 1.0
    with pytest.raises(ValueError, match='reference DOF'):
        Frf(f, [[1 + 0j] * 4], response_dof=['1X+'], reference_dof=['sledge'])


# ---- an abscissa is stored as it arrived ------------------------------------

def test_an_uneven_or_shuffled_abscissa_is_stored_without_complaint():
    """Both are real. A record resampled onto its own event times is
    unevenly spaced; one assembled from segments can arrive out of
    order. Refusing either at the door would be refusing real data, and
    re-sorting silently would be inventing a record nobody measured."""
    from visualdynamics.core.data import TimeHistory

    uneven = TimeHistory([0.0, 0.1, 0.35, 0.4], [[1.0, 2.0, 3.0, 4.0]],
                         response_dof=['1X+'])
    assert list(uneven.abscissa) == [0.0, 0.1, 0.35, 0.4]
    shuffled = TimeHistory([0.0, 0.2, 0.1, 0.3], [[1.0, 2.0, 3.0, 4.0]],
                           response_dof=['1X+'])
    assert list(shuffled.abscissa) == [0.0, 0.2, 0.1, 0.3]


def test_what_needs_an_even_step_asks_at_the_point_of_use():
    """And says which of the two problems it found — 'unevenly spaced'
    sent someone looking for a dropped sample when the record was simply
    in the wrong order."""
    import numpy as np
    import pytest

    from visualdynamics.core.data import TimeHistory

    uneven = TimeHistory([0.0, 0.1, 0.35, 0.4], [[1.0, 2.0, 3.0, 4.0]],
                         response_dof=['1X+'])
    with pytest.raises(ValueError, match='evenly spaced'):
        _ = uneven.sample_rate
    with pytest.raises(ValueError, match='evenly spaced'):
        uneven.compute_spectra()

    shuffled = TimeHistory([0.0, 0.2, 0.1, 0.3], [[1.0, 2.0, 3.0, 4.0]],
                           response_dof=['1X+'])
    with pytest.raises(ValueError, match='in order') as refused:
        _ = shuffled.sample_rate
    assert 'sample 2' in str(refused.value), 'and points at where it turns'

    even = TimeHistory(np.arange(4) / 4.0, [[1.0, 2.0, 3.0, 4.0]],
                       response_dof=['1X+'])
    assert even.sample_rate == pytest.approx(4.0)


def test_a_channel_missing_its_node_or_direction_still_imports():
    """People leave fields out. A DOF string is the node and the
    direction concatenated, so whatever was recorded is what comes out —
    and all four shapes import, because refusing the file would leave
    the user with nothing to fix. The record is then visibly
    incompatible with any geometry, and the channel table is right there
    to correct it in."""
    import numpy as np

    from visualdynamics.core.channel_table import ChannelTable
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.geometry import Geometry

    table = ChannelTable({'channel': [1, 2, 3, 4],
                          'node': ['101', '102', '', ''],
                          'direction': ['Z+', '', 'Y+', ''],
                          'unit': ['g'] * 4})
    assert table.dof_strings() == ['101Z+', '102', 'Y+', '']

    t = np.arange(4) / 4.0
    history = TimeHistory(t, np.ones((4, 4)), response_dof=table.dof_strings())
    assert history.response_dof == ['101Z+', '102', 'Y+', '']

    geometry = Geometry(node_id=[101, 102], node_xyz=[[0, 0, 0], [1, 0, 0]])
    missing = geometry.missing_dofs(history.response_dof)
    assert missing == ['Y+', ''], 'the incomplete ones do not land on it'


def test_a_direction_that_is_not_a_direction_is_still_a_typo():
    """No amount of fixing the channel table makes Q an axis."""
    import numpy as np
    import pytest

    from visualdynamics.core.data import TimeHistory

    with pytest.raises(ValueError, match='names no direction'):
        TimeHistory(np.arange(4) / 4.0, [[1.0, 2.0, 3.0, 4.0]],
                    response_dof=['101Q+'])
