"""A project to and from the Engineering Sciences Common Data Format
(`visualdynamics.io.escdf_objects`, 2026-09-30).

Two round trips, and the reference in the middle. A project written
here reads back whole, every object from the attachment that holds it
in its own layout, its object groups as named activities and back. The
same file with every attachment stripped — what a foreign reader would
keep — reads back from the standard fields alone: nodes, lines,
elements and per-node axes; records, units and abscissa; frequencies
and shapes. And the reference implementation loads and validates every
dataset the export writes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Frf, Psd, Srs, TimeHistory
from visualdynamics.core.photos import Photos
from visualdynamics.core.report import Report
from visualdynamics.io import escdf, escdf_objects
from visualdynamics.io.notes import ImportNote

reference = pytest.importorskip('escdf')


@pytest.fixture(autouse=True)
def _no_prompt(monkeypatch):
    monkeypatch.setattr(reference.ESCDF, 'get_or_prompt_attribution_name',
                        staticmethod(lambda **kwargs: 'test'))


def _project():
    """A project of one of everything the format has a type for, and
    of what it has not."""
    from visualdynamics.demo import plate

    project = visualdynamics.Project('Plate modal test')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    model = plate.build()
    geometry = model.geometry()
    geometry.define_units('m')
    project.add('Plate FEM', geometry)
    project.add('FEM Modes', model.eigensolution(num_modes=8))
    f = np.linspace(0, 100, 51)
    project.add('FRF', Frf(f, np.ones((2, 51)) * (1 + 1j), ['1X+', '2X+'],
                           reference_dof=['3Z+', '3Z+'], ordinate_unit='m/s**2',
                           reference_unit='N'))
    psd = Psd(f, np.ones((2, 51)), ['1X+', '2X+'], reference_dof=['1X+', '2X+'])
    psd.define_units('m/s**2')               # a density declares its base quantity
    project.add('PSD', psd)
    project.add('SRS', Srs(np.geomspace(10, 1000, 25), np.ones((1, 25)), ['1X+'],
                           ordinate_unit='m/s**2', q=10.0))
    project.link('Plate FEM', 'FEM Modes', 'FRF', 'PSD', 'SRS', name='Analysis')
    project.add('Photos', Photos(['first'], ['png'], [b'\x89PNG\r\n\x1a\n' + b'x' * 20]))
    project.add('Report', Report('Plate report', [{'kind': 'text', 'text': 'Hello.'}]))
    return project


def _names(project):
    return [name for name in project]


def test_a_project_round_trips_whole(tmp_path):
    project = _project()
    path = tmp_path / 'plate.escdf'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        project.save(path)
        back = visualdynamics.Project.open(path)
    assert _names(back) == _names(project), 'the order it held them'
    assert back.name == project.name and back.project_type == project.project_type
    assert back.object_groups == project.object_groups, 'a named group is an activity is a named group'
    assert back.provenance == project.provenance
    for name, obj in project.items():
        again = back[name]
        assert type(again) is type(obj), name
    np.testing.assert_allclose(back['FRF'].ordinate, project['FRF'].ordinate)
    assert back['FRF'].reference_unit == project['FRF'].reference_unit
    assert back['SRS'].q == project['SRS'].q
    assert list(back['Plate FEM'].group_name) == list(project['Plate FEM'].group_name)
    np.testing.assert_allclose(back['FEM Modes'].shape_matrix,
                               project['FEM Modes'].shape_matrix)
    assert back['Photos'].images[0] == project['Photos'].images[0]
    assert back['Report'].blocks == project['Report'].blocks


def test_the_reference_loads_and_validates_every_dataset(tmp_path):
    project = _project()
    path = tmp_path / 'plate.escdf'
    project.save(path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        theirs = reference.ESCDF.load(str(path))
    assert 'Analysis' in theirs.activities.names
    names = {d.name: d for d in theirs.get_activity_data('Analysis')}
    assert {'FEM_Modes', 'FRF', 'PSD', 'SRS'} <= set(names)
    assert names['SRS'].dataset_type == 'response_spectrum'
    assert names['FRF'].ordinate_unit[...] == '(m/s^2)/N'
    assert str(names['PSD'].ordinate_unit[...]) == '(m/s^2)^2/Hz'
    for dataset in list(theirs.metadata) + [
            d for activity in theirs.activities for d in activity.data]:
        assert dataset.validate(), dataset.name
    assert 'Plate_FEM' in theirs.activities['Analysis'].metadata_links
    report = theirs.metadata['Report']
    labels = list(report.attachment_names[...])
    assert 'Report.html' in labels, 'the report rendered, for anyone'
    html = bytes(report.attachments[...][labels.index('Report.html')]).decode('utf-8')
    assert html.startswith('<!DOCTYPE html>') and 'Hello.' in html
    assert theirs.metadata['Photos'].attachment_names[...][0] == 'first.png'


def _stripped(path, out):
    """The same file with every attachment and note removed: what a
    reader that keeps only the standard fields would pass along."""
    file = escdf.read(path)
    for dataset in list(file.metadata.values()) + [
            d for a in file.activities.values() for d in a.data.values()]:
        for key in ('attachments', 'attachment_names', 'notes'):
            dataset.values.pop(key, None)
    del file.metadata[escdf_objects.PROJECT_RECORD]
    escdf.write(file, out)


def test_the_standard_fields_alone_rebuild_the_objects(tmp_path):
    project = _project()
    path = tmp_path / 'plate.escdf'
    project.save(path)
    bare = tmp_path / 'bare.escdf'
    _stripped(path, bare)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        back = visualdynamics.Project.open(bare)
    # the report and the photos were attachments; the rest is the standard's
    assert set(back) == set(project) - {'Report', 'Photos'}
    geometry, original = back['Plate FEM'], project['Plate FEM']
    assert geometry.num_nodes == original.num_nodes
    np.testing.assert_allclose(geometry.node_xyz, original.node_xyz)
    assert len(geometry.elem_conn) == len(original.elem_conn)
    assert geometry.length_unit == 'm'
    frf = back['FRF']
    assert frf.units_defined
    assert frf.ordinate_unit == ['m/s**2', 'm/s**2'] and frf.reference_unit == ['N', 'N']
    assert frf.reference_dof == ['3Z+', '3Z+']
    assert back['SRS'].q == pytest.approx(10.0)
    np.testing.assert_allclose(back['FEM Modes'].frequency, project['FEM Modes'].frequency)
    np.testing.assert_allclose(back['FEM Modes'].shape_matrix,
                               project['FEM Modes'].shape_matrix)
    channel_table = back['Channel Table']
    assert list(channel_table.frame['node']) == list(project['Channel Table'].frame['node'])
    # the activity is the object group, named for it
    group = next(g for g in back.object_groups if g.get('name') == 'Analysis')
    assert set(group['members']) == {'Plate FEM', 'FEM Modes', 'FRF', 'PSD', 'SRS'}


def test_units_are_declared_and_converted_on_import(tmp_path):
    """A foreign file's units are declared, not merely named: values in
    g arrive in m/s**2 and show as g, a PSD in g^2/Hz likewise, and a
    time base in milliseconds becomes seconds (the first cut named the
    units and the app showed them undefined — Brandon, 2026-09-30)."""
    file = escdf.File(created_by='x')
    time = escdf.Dataset('t', 'data', 'Time in g')
    time.values = {'data_type': 'time response', 'channel': np.array([['1X+']]),
                   'ordinate_unit': 'g', 'abscissa_unit': 'ms',
                   'ordinate': np.ones((1, 4)), 'abscissa_start': 0.0,
                   'abscissa_step': 10.0}
    psd = escdf.Dataset('p', 'data', 'Spec CPSD')
    psd.values = {'data_type': 'power spectral density',
                  'channel': np.array([['1X+', '1X+']]),
                  'ordinate_unit': np.array(['g^2/Hz'], dtype=object),
                  'abscissa_unit': 'Hz', 'ordinate': np.ones((1, 3)),
                  'abscissa': np.array([10., 20., 40.])}
    file.activities['a'] = escdf.Activity('a', 'A', data={'t': time, 'p': psd})
    path = tmp_path / 'units.escdf'
    escdf.write(file, path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        project = escdf_objects.load(path)
    history = project['Time in g']
    assert history.units_defined and history.ordinate_unit == ['g']
    assert history.ordinate[0, 0] == pytest.approx(9.80665)
    np.testing.assert_allclose(history.abscissa, [0.0, 0.01, 0.02, 0.03])
    spec = project['Spec CPSD']
    assert spec.units_defined and spec.ordinate_unit == ['g'], 'the base quantity'
    assert spec.ordinate[0, 0] == pytest.approx(9.80665 ** 2)
    assert spec.ordinate_dim[0] == 'acceleration**2/frequency'


def test_a_cross_density_declares_both_quantities(tmp_path):
    file = escdf.File(created_by='x')
    cross = escdf.Dataset('c', 'data', 'Cross')
    cross.values = {'data_type': 'power spectral density',
                    'channel': np.array([['1X+', '2Z+']]),
                    'ordinate_unit': 'g*N/Hz', 'abscissa_unit': 'Hz',
                    'ordinate': np.ones((1, 3)) + 0j,
                    'abscissa': np.array([10., 20., 40.])}
    file.activities['a'] = escdf.Activity('a', 'A', data={'c': cross})
    path = tmp_path / 'cross.escdf'
    escdf.write(file, path)
    project = escdf_objects.load(path)
    density = project['Cross']
    assert density.ordinate_unit == ['g'] and density.reference_unit == ['N']
    assert density.ordinate_dim[0] == 'acceleration*force/frequency'
    assert density.ordinate[0, 0] == pytest.approx(9.80665)
    # and back out in the file's spelling
    again = tmp_path / 'again.escdf'
    project.save(again)
    written = escdf.read(again)
    dataset = next(d for a in written.activities.values() for d in a.data.values()
                   if d.descriptive_name == 'Cross')
    assert dataset.values['ordinate_unit'] == 'g*N/Hz'


def test_per_node_axes_become_coordinate_systems(tmp_path):
    """A foreign geometry whose nodes have local axes: each distinct
    frame becomes a coordinate system the node is displaced in, so a
    channel measured along a node's own x is drawn along it."""
    file = escdf.File(created_by='x')
    geometry = escdf.Dataset('g', 'geometry', 'turned')
    turned = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.]])
    geometry.values = {
        'node_id': np.array([1, 2], dtype=np.uint64),
        'node_position': np.array([[0, 0, 0], [1, 0, 0.]]),
        'node_x_direction': np.array([[1, 0, 0], turned[0]]),
        'node_y_direction': np.array([[0, 1, 0], turned[1]]),
        'node_z_direction': np.array([[0, 0, 1], turned[2]]),
        'position_units': 'mm',
    }
    file.metadata['g'] = geometry
    file.activities['a'] = escdf.Activity('a', 'A', links=['g'])
    path = tmp_path / 'axes.escdf'
    escdf.write(file, path)
    project = escdf_objects.load(path)
    g = project['turned']
    assert g.length_unit == 'mm'
    np.testing.assert_allclose(g.dof_direction('1X+'), [1, 0, 0])
    np.testing.assert_allclose(g.dof_direction('2X+'), [0, 1, 0])
    np.testing.assert_allclose(g.dof_direction('2Y+'), [-1, 0, 0])


def test_a_geometry_of_bare_nodes_says_so(tmp_path):
    file = escdf.File(created_by='x')
    geometry = escdf.Dataset('g', 'geometry', 'points')
    geometry.values = {
        'node_id': np.array([1, 2], dtype=np.uint64),
        'node_position': np.array([[0, 0, 0], [1, 0, 0.]]),
        'node_x_direction': np.tile([1., 0, 0], (2, 1)),
        'node_y_direction': np.tile([0., 1, 0], (2, 1)),
        'node_z_direction': np.tile([0., 0, 1], (2, 1)),
        'position_units': 'm',
    }
    file.metadata['g'] = geometry
    file.activities['a'] = escdf.Activity('a', 'A', links=['g'])
    path = tmp_path / 'points.escdf'
    escdf.write(file, path)
    with pytest.warns(ImportNote, match='2 nodes and no line_connection'):
        project = escdf_objects.load(path)
    assert project['points'].num_nodes == 2


def test_what_the_specifications_do_not_define_is_left_out_with_a_note(tmp_path):
    file = escdf.File(created_by='x')
    file.metadata['m'] = escdf.Dataset('m', 'mystery', 'a mystery')
    file.activities['a'] = escdf.Activity('a', 'A', links=['m'])
    path = tmp_path / 'mystery.escdf'
    escdf.write(file, path)
    with pytest.warns(ImportNote, match="'mystery' is not one the specifications define"):
        project = escdf_objects.load(path)
    assert not list(project)


def test_the_window_saves_the_project_as_the_format_with_the_creator_name(
        window, pump, tmp_path, monkeypatch):
    """Save Project offers the format, and the file names the creator
    the preference holds (File → Creator Name…), or the login name."""
    from PySide6.QtWidgets import QFileDialog

    from visualdynamics.gui import preferences

    window.import_paths([fixture_path('plate', 'modal.nc4')])
    pump()
    # named without a suffix: the dialog adds `.h5`, the default
    path = tmp_path / 'saved'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (
                            str(path), 'Engineering Sciences Common Data Format (*.h5)')))
    preferences.remember_creator('Test Person')
    try:
        window.save_test()
    finally:
        preferences.remember_creator('')
    file = escdf.read(tmp_path / 'saved.h5')
    assert file.created_by == 'Test Person'
    assert file.problems() == []
    assert 'Saved' in window.statusBar().currentMessage()


def test_one_object_exports_as_a_project_of_one(tmp_path):
    t = np.linspace(0, 1, 11)
    history = TimeHistory(t, np.zeros((1, 11)), ['1X+'], ordinate_unit='g')
    path = tmp_path / 'one'
    visualdynamics.io.export_file(history, str(path), format='escdf')
    file = escdf.read(tmp_path / 'one.h5')
    assert file.problems() == []
    activity = next(iter(file.activities.values()))
    data = next(iter(activity.data.values()))
    assert data.values['data_type'] == 'time response'
    assert data.values['abscissa_step'] == pytest.approx(0.1)
    assert data.values['ordinate_unit'] == 'g'


def test_a_file_is_written_as_h5_unless_it_names_another_suffix(tmp_path):
    """`.h5` is the default (Brandon, 2026-10-03): the format's files
    are named for their container where it is used, and one written
    here should sit beside them. A name already ending in `.hdf5` or
    the old `.escdf` keeps it, through the export and through
    `Project.save`, and every one of them opens again."""
    project = _project()
    project.save(tmp_path / 'plain.h5')
    visualdynamics.io.export_file(project, str(tmp_path / 'bare'), format='escdf')
    project.save(tmp_path / 'long.hdf5')
    project.save(tmp_path / 'old.escdf')
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'bare.h5', 'long.hdf5', 'old.escdf', 'plain.h5']
    for path in tmp_path.iterdir():
        assert escdf.read(path).problems() == []
        assert set(visualdynamics.Project.open(path)) == set(project)
