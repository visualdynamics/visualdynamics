
import numpy as np
import pytest
from conftest import fixture_path, plate_geometry_and_shapes

import visualdynamics
from visualdynamics.deform import (
    EnvelopeDeflection,
    OdsDeflection,
    ShapeDeflection,
    TimeDeflection,
    auto_scale,
    dof_directions,
    model_size,
    node_displacements,
)


def simple_geometry(**kwargs):
    """Three nodes in a line, one meter apart."""
    return visualdynamics.Geometry(node_id=[1, 2, 3],
                         node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                         length_unit='m', **kwargs)


def test_signed_directions():
    geometry = simple_geometry()
    displacement = node_displacements(
        geometry, ['1X+', '2Y-', '3Z+'], [2.0, 3.0, 4.0])
    assert np.allclose(displacement, [[2, 0, 0], [0, -3, 0], [0, 0, 4]])


def test_rotational_dofs_do_not_move_nodes():
    geometry = simple_geometry()
    displacement = node_displacements(
        geometry, ['1RX+', '1RY+', '1RZ+'], [5.0, 5.0, 5.0])
    assert np.allclose(displacement, 0)


def test_nodes_without_data_hold_still():
    geometry = simple_geometry()
    displacement = node_displacements(geometry, ['2Z+'], [1.0])
    assert np.allclose(displacement[0], 0)
    assert np.allclose(displacement[2], 0)
    assert np.allclose(displacement[1], [0, 0, 1])


def test_dofs_for_absent_nodes_are_dropped():
    geometry = simple_geometry()
    rows, _directions, used = dof_directions(geometry, ["1X+", "99X+"])
    assert list(used) == [True, False]
    assert len(rows) == 1


def test_several_dofs_on_one_node_combine():
    geometry = simple_geometry()
    displacement = node_displacements(
        geometry, ['2X+', '2Y+', '2Z+'], [1.0, 2.0, 3.0])
    assert np.allclose(displacement[1], [1, 2, 3])


def test_displacement_coordinate_system_is_applied():
    """A node whose displacement CS is rotated 90 degrees about Z moves its
    local X along global Y."""
    rotation = np.array([[0.0, 1.0, 0.0],      # local X -> global Y
                         [-1.0, 0.0, 0.0],     # local Y -> global -X
                         [0.0, 0.0, 1.0]])
    matrix = np.vstack([rotation, np.zeros(3)])
    geometry = visualdynamics.Geometry(
        node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]],
        node_disp_cs=[7, 1], cs_id=[1, 7], cs_name=['', 'rotated'],
        cs_type=[0, 0],
        cs_matrix=[np.vstack([np.eye(3), np.zeros(3)]), matrix],
        length_unit='m')
    displacement = node_displacements(geometry, ['1X+', '2X+'], [1.0, 1.0])
    assert np.allclose(displacement[0], [0, 1, 0]), 'rotated node'
    assert np.allclose(displacement[1], [1, 0, 0]), 'unrotated node'


def test_auto_scale_puts_peak_at_a_fraction_of_the_model():
    geometry = simple_geometry()
    scale = auto_scale(geometry, peak=0.5, fraction=0.1)
    assert scale * 0.5 == pytest.approx(0.1 * model_size(geometry))


def test_auto_scale_survives_a_zero_shape():
    assert auto_scale(simple_geometry(), peak=0.0) == 1.0


class TestShapeDeflection:
    def geometry_and_shapes(self):
        return plate_geometry_and_shapes()

    def test_real_mode_sweeps_a_cosine(self):
        geometry, shapes = self.geometry_and_shapes()
        deflection = ShapeDeflection(geometry, shapes.coordinate,
                                     shapes.shape_matrix[8])
        at_zero = deflection.offsets(0.0).copy()
        assert np.allclose(deflection.offsets(np.pi).copy(), -at_zero)
        assert np.allclose(deflection.offsets(2 * np.pi).copy(), at_zero)
        assert np.allclose(deflection.offsets(np.pi / 2).copy(), 0)

    def test_complex_mode_travels(self):
        geometry, shapes = self.geometry_and_shapes()
        shape = shapes.shape_matrix[8].astype(complex)
        shape[:50] *= 1j                       # give part of it a phase
        deflection = ShapeDeflection(geometry, shapes.coordinate, shape)
        assert not np.allclose(deflection.offsets(np.pi / 2).copy(), 0)

    def test_one_row_per_moving_node(self):
        geometry, shapes = self.geometry_and_shapes()
        deflection = ShapeDeflection(geometry, shapes.coordinate,
                                     shapes.shape_matrix[8])
        assert len(deflection.rows) == geometry.num_nodes
        assert len(np.unique(deflection.rows)) == len(deflection.rows)
        assert deflection.offsets(0.0).shape == (len(deflection.rows), 3)

    def test_offsets_reuse_one_buffer(self):
        """Documented behavior: a frame allocates nothing."""
        geometry, shapes = self.geometry_and_shapes()
        deflection = ShapeDeflection(geometry, shapes.coordinate,
                                     shapes.shape_matrix[8])
        assert deflection.offsets(0.0) is deflection.offsets(1.0)


class TestTimeDeflection:
    def test_frames_follow_the_samples(self):
        geometry = simple_geometry()
        ordinate = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        deflection = TimeDeflection(geometry, ['1Z+', '3Z+'], ordinate)
        assert deflection.num_frames == 3
        assert np.allclose(deflection.offsets(0), [[0, 0, 1], [0, 0, 10]])
        assert np.allclose(deflection.offsets(2), [[0, 0, 3], [0, 0, 30]])

    def test_repeated_nodes_accumulate_per_frame(self):
        """X and Y channels on one node must sum, not overwrite."""
        geometry = simple_geometry()
        ordinate = np.array([[1.0, 2.0], [5.0, 6.0]])
        deflection = TimeDeflection(geometry, ['2X+', '2Y+'], ordinate)
        assert len(deflection.rows) == 1
        assert np.allclose(deflection.offsets(0), [[1, 5, 0]])
        assert np.allclose(deflection.offsets(1), [[2, 6, 0]])

    def test_a_measurement_moves_every_measured_node(self):
        geometry = visualdynamics.import_file(
            fixture_path('plate', 'geometry.npz'),
            length_unit='m')
        time = visualdynamics.import_file(
            fixture_path('plate', 'time.npz'))
        deflection = TimeDeflection(geometry, time.response_dof, time.ordinate)
        assert len(deflection.rows) == len(time.response_dof), (
            'every measured node moves; the unmeasured mesh rides along')
        assert deflection.num_frames == len(time.abscissa)

    def test_only_measured_nodes_are_listed(self):
        """A partial measurement moves only what it covers."""
        geometry = visualdynamics.import_file(
            fixture_path('plate', 'geometry.npz'),
            length_unit='m')
        measured = [f'{int(node)}Z+' for node in geometry.node_id[:10]]
        deflection = TimeDeflection(geometry, measured,
                                    np.ones((len(measured), 8)))
        assert len(deflection.rows) == 10 < geometry.num_nodes

    def test_peak_drives_the_scale(self):
        geometry = simple_geometry()
        deflection = TimeDeflection(geometry, ['1Z+'],
                                    np.array([[0.0, -4.0, 2.0]]))
        assert deflection.peak == 4.0


def _sweep_peak(deflection, parameters):
    """The furthest any node gets, found the slow honest way."""
    return max(float(np.linalg.norm(deflection.offsets(p), axis=1).max())
               for p in parameters)


def _scatter_geometry(n=25, seed=4):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    geo = visualdynamics.Geometry(node_id=ids, node_xyz=rng.random((n, 3)))
    geo.define_units('m')
    return geo, [f'{i}{d}+' for i in ids for d in 'XYZ'], rng


def test_peak_magnitude_of_a_real_mode_matches_a_phase_sweep():
    geo, dofs, rng = _scatter_geometry()
    deflection = ShapeDeflection(geo, dofs, rng.normal(size=len(dofs)))
    sweep = _sweep_peak(deflection, np.linspace(0, 2 * np.pi, 2001))
    assert deflection.peak_magnitude == pytest.approx(sweep, rel=1e-6)


def test_peak_magnitude_of_a_complex_mode_matches_a_phase_sweep():
    geo, dofs, rng = _scatter_geometry()
    shape = rng.normal(size=len(dofs)) + 1j * rng.normal(size=len(dofs))
    deflection = ShapeDeflection(geo, dofs, shape)
    sweep = _sweep_peak(deflection, np.linspace(0, 2 * np.pi, 4001))
    assert deflection.peak_magnitude == pytest.approx(sweep, rel=1e-5)


def test_peak_magnitude_exceeds_the_largest_single_dof():
    """A node moving in X, Y and Z travels further than any one of them."""
    geo, dofs, rng = _scatter_geometry()
    deflection = ShapeDeflection(geo, dofs, rng.normal(size=len(dofs)))
    assert deflection.peak_magnitude > deflection.peak


def test_peak_magnitude_of_time_data_matches_every_sample():
    geo, dofs, rng = _scatter_geometry()
    deflection = TimeDeflection(geo, dofs, rng.normal(size=(len(dofs), 300)))
    every = _sweep_peak(deflection, range(deflection.num_frames))
    assert deflection.peak_magnitude == pytest.approx(every, rel=1e-9)


def test_peak_magnitude_spans_chunk_boundaries():
    """Enough nodes and samples that the walk takes several chunks."""
    geo, dofs, rng = _scatter_geometry(n=400)
    ordinate = rng.normal(size=(len(dofs), 900))
    deflection = TimeDeflection(geo, dofs, ordinate)
    every = _sweep_peak(deflection, range(deflection.num_frames))
    assert deflection.peak_magnitude == pytest.approx(every, rel=1e-9)


# --- the operating deflection shape ------------------------------------------


def test_ods_offsets_are_the_complex_pattern_at_the_line():
    """At one line the records' complex values sweep like a complex mode:
    offsets(phase) = Re(H(line) * e^{i*phase}) — up to the line's own
    scale and a global phase. A steady state has no time origin, so
    each line is rotated to put its fullest deflection at phase zero:
    a resonance line is nearly pure imaginary, and without the rotation
    every still and every first frame showed the pattern's null, a
    flat model."""
    geometry = simple_geometry()
    ordinate = np.array([[1 + 0j, 2j],
                         [3 + 0j, -1j]])
    ods = OdsDeflection(geometry, ['1Z+', '2Z+'], ordinate)
    assert ods.line == 0
    assert np.allclose(ods.offsets(0.0)[:, 2], [1 / 3, 1])
    ods.line = 1
    # [2j, -1j] normalized and rotated real: the shape shows at phase
    # 0, nulls at pi/2
    assert np.allclose(ods.offsets(0.0)[:, 2], [1, -0.5])
    assert np.allclose(ods.offsets(np.pi / 2)[:, 2], [0, 0])


def test_ods_every_line_deflects_at_full_scale():
    """The cursor starts where a record is largest, but every line shows
    its shape at the same size: keeping the true relative amplitude was
    tried first and read as broken — away from resonance the model
    barely moved. How much a frequency responds is the plot's curve;
    the scene answers what the shape there is."""
    geometry = simple_geometry()
    ordinate = np.array([[1 + 0j, 5j],
                         [2 + 0j, 1j]])
    ods = OdsDeflection(geometry, ['1Z+', '2Z+'], ordinate)
    assert ods.strongest_line == 1
    ods.line = 0
    weak = float(np.abs(ods.offsets(0.0)).max())
    ods.line = 1
    assert float(np.abs(ods.offsets(0.0)).max()) == weak == 1.0, (
        'the strongest record of any line deflects the same')
    assert ods.peak == 1.0, 'the yardstick the animator scales against'


def test_ods_peak_magnitude_matches_a_line_and_phase_sweep():
    geo, dofs, rng = _scatter_geometry()
    ordinate = (rng.normal(size=(len(dofs), 7))
                + 1j * rng.normal(size=(len(dofs), 7)))
    ods = OdsDeflection(geo, dofs, ordinate)
    brute = 0.0
    for line in range(7):
        ods.line = line
        for phase in np.linspace(0, 2 * np.pi, 721):
            brute = max(brute, float(np.linalg.norm(
                ods.offsets(phase), axis=1).max()))
    assert ods.peak_magnitude == pytest.approx(brute, rel=1e-4)


# --- the PSD envelope --------------------------------------------------------


def test_envelope_offsets_are_sqrt_psd_at_full_scale():
    """Amplitude, not power — a 6 dB drop halves the picture — and each
    line normalized to its strongest record, the ODS's convention."""
    geometry = simple_geometry()
    ordinate = np.array([[4.0, 1.0],
                         [16.0, 1.0]])
    envelope = EnvelopeDeflection(geometry, ['1Z+', '2Z+'], ordinate)
    assert envelope.strongest_line == 0
    assert np.allclose(envelope.offsets(0.0)[:, 2], [0.5, 1.0])
    envelope.line = 1
    assert np.allclose(envelope.offsets(0.0)[:, 2], [1.0, 1.0])


def test_envelope_color_is_absolute():
    """Length answers where the energy sits at this line; color answers
    how loud this line is anywhere: dB below the loudest node at any
    line, floored at -40."""
    geometry = simple_geometry()
    ordinate = np.array([[100.0, 1.0],
                         [1.0, 0.0]])
    envelope = EnvelopeDeflection(geometry, ['1Z+', '2Z+'], ordinate)
    assert np.allclose(envelope.node_decibels(), [0.0, -20.0])
    envelope.line = 1
    # full length (per-line normalization), quiet color (absolute)
    assert np.allclose(envelope.offsets(0.0)[:, 2], [1.0, 0.0])
    assert np.allclose(envelope.node_decibels(), [-20.0, -40.0]), (
        'a silent channel sits on the floor, not at -infinity')


def test_envelope_combines_axes_on_a_node():
    """Two axes on one node reach the in-phase diagonal — two copies
    cannot show four corners, and the guide says so."""
    geometry = simple_geometry()
    ordinate = np.array([[9.0], [9.0]])
    envelope = EnvelopeDeflection(geometry, ['2X+', '2Z+'], ordinate)
    assert np.allclose(envelope.offsets(0.0), [[1.0, 0.0, 1.0]])
