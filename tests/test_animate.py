
import numpy as np
import pytest
from conftest import fixture_path, plate_geometry_and_shapes

import visualdynamics
from visualdynamics.deform import ShapeDeflection, TimeDeflection


@pytest.fixture(scope='module')
def model():
    return plate_geometry_and_shapes()


def animator(geometry, deflection, unit_system=None, **kwargs):
    import pyvista as pv

    from visualdynamics.viz.animate import GeometryAnimator

    plotter = pv.Plotter(off_screen=True)
    return GeometryAnimator(plotter, geometry, deflection,
                            unit_system=unit_system or visualdynamics.SI, **kwargs)


def test_every_mesh_shares_one_point_buffer(model):
    """The whole scene moves with a single write."""
    geometry, shapes = model
    anim = animator(geometry, ShapeDeflection(geometry, shapes.coordinate,
                                              shapes.shape_matrix[8]))
    assert len(anim.meshes) > 1, 'the model draws several meshes'
    anim.set_parameter(0.0)
    for mesh in anim.meshes:
        assert np.shares_memory(np.asarray(mesh.points), anim.view)
    anim.plotter.close()


def test_deflection_scales_to_a_tenth_of_the_model(model):
    geometry, shapes = model
    deflection = ShapeDeflection(geometry, shapes.coordinate,
                                 shapes.shape_matrix[8])
    for system in (visualdynamics.SI, visualdynamics.IN_LBF_S):
        anim = animator(geometry, deflection, system)
        anim.set_parameter(0.0)
        moved = float(np.abs(anim.view - anim.base).max())
        assert moved == pytest.approx(0.1 * anim.model_size, rel=1e-6)
        anim.plotter.close()


def test_user_scale_multiplies_the_motion(model):
    geometry, shapes = model
    anim = animator(geometry, ShapeDeflection(geometry, shapes.coordinate,
                                              shapes.shape_matrix[8]))
    anim.set_parameter(0.0)
    at_one = float(np.abs(anim.view - anim.base).max())
    anim.set_scale(3.0)
    assert float(np.abs(anim.view - anim.base).max()) == pytest.approx(
        3 * at_one, rel=1e-9)
    anim.plotter.close()


def test_reset_restores_the_model(model):
    geometry, shapes = model
    anim = animator(geometry, ShapeDeflection(geometry, shapes.coordinate,
                                              shapes.shape_matrix[8]))
    anim.set_parameter(0.7)
    assert np.abs(anim.view - anim.base).max() > 0
    anim.reset()
    assert np.array_equal(anim.view, anim.base)
    anim.plotter.close()


def test_frames_do_not_rebuild_the_scene(model):
    """Actors are built once; a frame only moves points."""
    geometry, shapes = model
    anim = animator(geometry, ShapeDeflection(geometry, shapes.coordinate,
                                              shapes.shape_matrix[8]))
    before = len(anim.plotter.renderer.actors)
    meshes = list(anim.meshes)
    for step in range(10):
        anim.set_parameter(step / 10 * 2 * np.pi)
    assert len(anim.plotter.renderer.actors) == before
    assert anim.meshes == meshes
    anim.plotter.close()


def test_time_playback_steps_samples():
    geometry = visualdynamics.Geometry(node_id=[1, 2],
                             node_xyz=[[0, 0, 0], [1, 0, 0]],
                             length_unit='m')
    ordinate = np.array([[0.0, 1.0, -1.0]])
    anim = animator(geometry, TimeDeflection(geometry, ['2Z+'], ordinate))
    anim.set_parameter(0)
    assert np.allclose(anim.view, anim.base), 'first sample is zero'
    anim.set_parameter(1)
    peak = anim.view[1, 2] - anim.base[1, 2]
    assert peak > 0
    anim.set_parameter(2)
    assert anim.view[1, 2] - anim.base[1, 2] == pytest.approx(-peak)
    anim.plotter.close()


def test_unmeasured_nodes_never_move():
    geometry = visualdynamics.Geometry(node_id=[1, 2, 3],
                             node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                             length_unit='m')
    anim = animator(geometry, TimeDeflection(geometry, ['2Z+'],
                                             np.array([[1.0, 2.0]])))
    anim.set_parameter(1)
    assert np.array_equal(anim.view[0], anim.base[0])
    assert np.array_equal(anim.view[2], anim.base[2])
    assert not np.array_equal(anim.view[1], anim.base[1])
    anim.plotter.close()


def _mode(model, index=3):
    geometry, shapes = model
    return geometry, ShapeDeflection(geometry, shapes.coordinate,
                                     shapes.shape_matrix[index])


def test_colormap_shares_one_scalar_array_across_every_mesh(model):
    geometry, deflection = _mode(model)
    moving = animator(geometry, deflection, colormap=True)
    assert moving.magnitude is not None
    assert len(moving.meshes) > 1
    assert all(mesh.GetPointData().GetScalars() is moving.scalars_source
               for mesh in moving.meshes), 'one write must recolor them all'


def test_colors_follow_the_phase_and_go_dark_at_the_crossing(model):
    geometry, deflection = _mode(model)
    moving = animator(geometry, deflection, colormap=True)

    moving.set_parameter(0.0)
    at_peak = float(moving.magnitude.max())
    moving.set_parameter(np.pi / 2)
    at_crossing = float(moving.magnitude.max())
    assert at_peak > 0
    assert at_crossing < at_peak * 1e-6, (
        'a real mode is motionless a quarter turn on, so nothing is lit')

    moving.set_parameter(np.pi / 3)
    part_way = float(moving.magnitude.max())
    assert 0 < part_way < at_peak, 'the colors pulse with the animation'


def test_the_color_scale_is_fixed_to_the_furthest_the_model_ever_moves(model):
    geometry, deflection = _mode(model)
    moving = animator(geometry, deflection, colormap=True)
    assert deflection.peak_magnitude > 0
    moving.set_parameter(0.0)
    assert float(moving.magnitude.max()) == pytest.approx(
        deflection.peak_magnitude, rel=1e-9), (
        'the top of the scale is reached, and only at the peak')
    for phase in (0.4, 1.0, 2.0, 3.0):
        moving.set_parameter(phase)
        assert float(moving.magnitude.max()) <= deflection.peak_magnitude


def test_without_the_colormap_nothing_is_computed_per_frame(model):
    geometry, deflection = _mode(model)
    moving = animator(geometry, deflection)
    assert moving.magnitude is None and moving.scalars_source is None
    moving.set_parameter(1.0)      # must not raise


# --- the scripting API the guide documents ---------------------------------
#
# `frf.animate(geometry)` and `psd.animate(geometry)` are in plotting.md
# with worked calls, and were verified by hand while they were built and
# by nothing afterwards: 101 lines of animate.py, the whole of both
# entry points, ran in no test. The interactive branches still cannot —
# they open a window — but the `screenshot=` path is the one a script
# takes, and it is the one documented.


@pytest.fixture(scope='module')
def measured():
    geometry = visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz'), length_unit='m')
    frf = visualdynamics.import_file(
        fixture_path('plate', 'frfs.npz'))
    psd = visualdynamics.import_file(
        fixture_path('plate', 'psd.npz'))
    return geometry, frf, psd


def test_an_frf_animates_its_operating_deflection_shape(measured, tmp_path):
    geometry, frf, _psd = measured
    path = tmp_path / 'ods.png'
    image = frf.animate(geometry, screenshot=str(path))
    assert path.stat().st_size > 1000
    assert image.ndim == 3, 'the rendered frame comes back too'


def test_the_frequency_argument_picks_the_line(measured, tmp_path):
    """Without one it starts on the strongest line; with one it starts
    where it was told, and the two are different pictures."""
    from visualdynamics.deform import OdsDeflection, animation_records

    geometry, frf, _psd = measured
    indices, _notes = animation_records(frf, None)
    dofs = [frf.response_dof[i] for i in indices]
    ods = OdsDeflection(geometry, dofs, np.asarray(frf.ordinate)[indices])
    strongest = ods.strongest_line
    ods.line = strongest
    at_peak = ods.offsets(0.0).copy()
    ods.line = int(np.argmin(np.abs(np.asarray(frf.abscissa) - 1000.0)))
    assert not np.allclose(ods.offsets(0.0), at_peak)
    # and the entry point takes the frequency at all
    path = tmp_path / 'ods-1000.png'
    frf.animate(geometry, frequency=1000.0, screenshot=str(path))
    assert path.stat().st_size > 1000


def test_a_psd_animates_its_envelope(measured, tmp_path):
    geometry, _frf, psd = measured
    path = tmp_path / 'envelope.png'
    psd.animate(geometry, screenshot=str(path))
    assert path.stat().st_size > 1000


def test_a_cpsd_animates_its_principal_shape_not_its_envelope(tmp_path):
    """A whole CPSD carries phase, so `animate` routes to the principal
    operating deflection rather than the phaseless envelope."""
    from visualdynamics.core.data import Psd

    geometry = visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz'), length_unit='m')
    channels = ['101Z+', '707Z+']
    shape = np.array([1.0, 2.0j])
    rows, rdof, fdof = [], [], []
    for i, ri in enumerate(channels):
        for j, rj in enumerate(channels):
            rows.append(np.full(8, shape[i] * np.conj(shape[j])))
            rdof.append(ri)
            fdof.append(rj)
    cpsd = Psd(np.arange(1.0, 9.0), np.array(rows), response_dof=rdof,
               reference_dof=fdof,
               ordinate_dim='acceleration**2/frequency',
               ordinate_unit='m/s**2')
    dofs, shapes, used = cpsd.principal_shapes()
    assert used == 'acceleration' and shapes.shape[0] == 2
    assert dofs == channels, 'the principal shape covers both channels'
    path = tmp_path / 'principal.png'
    cpsd.animate(geometry, screenshot=str(path))
    assert path.stat().st_size > 1000


def test_an_operating_shape_needs_phase_to_show(measured, tmp_path):
    """Real data has none, and drawing it as though it did is the one
    thing the envelope exists to avoid."""
    from visualdynamics.viz.animate import animate_ods

    geometry, _frf, psd = measured
    with pytest.raises(ValueError, match='needs phase'):
        animate_ods(geometry, psd, screenshot=str(tmp_path / 'no.png'))


def test_an_envelope_refuses_a_quantity_nothing_measures(measured, tmp_path):
    from visualdynamics.viz.animate import animate_envelope

    geometry, _frf, psd = measured
    with pytest.raises(ValueError, match='no auto records measure'):
        animate_envelope(geometry, psd, quantity='temperature',
                         screenshot=str(tmp_path / 'no.png'))


def test_animating_data_that_lands_nowhere_says_so(tmp_path):
    from visualdynamics.core.data import Frf
    from visualdynamics.viz.animate import animate_ods

    geometry = visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz'), length_unit='m')
    stranger = Frf(abscissa=np.arange(4.0),
                   ordinate=np.ones((1, 4), dtype=complex),
                   response_dof=['99999Z+'], reference_dof=['99999Z+'],
                   ordinate_dim='acceleration/force',
                   ordinate_unit='m/s**2', reference_unit='N')
    with pytest.raises(ValueError, match='land on this geometry'):
        animate_ods(geometry, stranger, screenshot=str(tmp_path / 'no.png'))


def test_envelope_records_is_the_one_filtering_rule():
    """The app's envelope view and the headless call share this rule
    (they each carried a copy once): cross records go first, then one
    quantity — both before the one-record-per-DOF dedupe, or a drive
    point's cross row or force PSD listed first would shadow its own
    DOF's genuine accelerometer auto."""
    import numpy as np

    from visualdynamics.core.data import Psd
    from visualdynamics.deform import envelope_records

    f = np.arange(8.0)
    rows = np.ones((4, 8))
    data = Psd(f, rows,
               response_dof=['101Z+', '101Z+', '101Z+', '102Z+'],
               reference_dof=['9001X+', '', '101Z+', ''],
               ordinate_dim=['acceleration**2/frequency',
                             'force**2/frequency',
                             'acceleration**2/frequency',
                             'acceleration**2/frequency'])
    indices, quantity, crossed, left_out = envelope_records(data)
    assert crossed == 1, 'the true cross record (its reference elsewhere)'
    assert quantity == 'acceleration', 'the commonest kind answers'
    assert indices == [2, 3], \
        'the drive point auto survives its own cross row and force row'
    assert left_out == 1, 'the force record waits for the quantity box'

    indices, quantity, _crossed, left_out = envelope_records(
        data, quantity='force')
    assert quantity == 'force' and indices == [1]
    assert left_out == 2, 'now the accelerations wait'

    asked, _q, _c, _l = envelope_records(data, records=[0, 3])
    assert asked == [3], 'an explicit record list is filtered, not obeyed'
