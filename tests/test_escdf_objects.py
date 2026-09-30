"""A project to and from the Engineering Sciences Common Data Format
(`visualdynamics.io.escdf_objects`, 2026-09-30).

Two round trips, and the reference in the middle. A project written
here reads back whole, every object from the attachment that holds it
in its own layout, its link groups as named activities and back. The
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
    project.add('PSD', Psd(f, np.ones((2, 51)), ['1X+', '2X+'], reference_dof=['1X+', '2X+'],
                           ordinate_unit='(m/s**2)**2/Hz'))
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
    assert back.links == project.links, 'a named group is an activity is a named group'
    assert back.provenance == project.provenance
    for name, obj in project.items():
        again = back[name]
        assert type(again) is type(obj), name
    np.testing.assert_allclose(back['FRF'].ordinate, project['FRF'].ordinate)
    assert back['FRF'].reference_unit == project['FRF'].reference_unit
    assert back['SRS'].q == project['SRS'].q
    assert list(back['Plate FEM'].block_name) == list(project['Plate FEM'].block_name)
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
    assert names['FRF'].ordinate_unit[...] == '(m/s**2)/(N)'
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
    assert frf.ordinate_unit == ['m/s**2', 'm/s**2'] and frf.reference_unit == ['N', 'N']
    assert frf.reference_dof == ['3Z+', '3Z+']
    assert back['SRS'].q == pytest.approx(10.0)
    np.testing.assert_allclose(back['FEM Modes'].frequency, project['FEM Modes'].frequency)
    np.testing.assert_allclose(back['FEM Modes'].shape_matrix,
                               project['FEM Modes'].shape_matrix)
    channel_table = back['Channel Table']
    assert list(channel_table.frame['node']) == list(project['Channel Table'].frame['node'])
    # the activity is the link group, named for it
    group = next(g for g in back.links if g.get('name') == 'Analysis')
    assert set(group['members']) == {'Plate FEM', 'FEM Modes', 'FRF', 'PSD', 'SRS'}


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
    path = tmp_path / 'saved.escdf'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (
                            str(path), 'Engineering Sciences Common Data Format (*.escdf)')))
    preferences.remember_creator('Test Person')
    try:
        window.save_test()
    finally:
        preferences.remember_creator('')
    file = escdf.read(path)
    assert file.created_by == 'Test Person'
    assert file.problems() == []
    assert 'Saved' in window.statusBar().currentMessage()


def test_one_object_exports_as_a_project_of_one(tmp_path):
    t = np.linspace(0, 1, 11)
    history = TimeHistory(t, np.zeros((1, 11)), ['1X+'], ordinate_unit='g')
    path = tmp_path / 'one'
    visualdynamics.io.export_file(history, str(path), format='escdf')
    file = escdf.read(tmp_path / 'one.escdf')
    assert file.problems() == []
    activity = next(iter(file.activities.values()))
    data = next(iter(activity.data.values()))
    assert data.values['data_type'] == 'time response'
    assert data.values['abscissa_step'] == pytest.approx(0.1)
    assert data.values['ordinate_unit'] == 'g'
