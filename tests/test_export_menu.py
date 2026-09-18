"""Saving an object is one verb with a file-type list, not two verbs.

*Save As* and *Export* were the same act under two names — pick an
object, pick a file, write it — and which formats you were offered
depended on which of the two you had found. A user who wanted a UNV had
to know the second entry existed. Now the format is the save dialog's
file-type list, which is where a save dialog has always asked.

Neither is in the File menu: that saves the whole project. Export always
wrote exactly one object, and 'Export Selected As...' read as a batch
operation that never existed.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path
from PySide6.QtWidgets import QFileDialog

import visualdynamics


@pytest.fixture
def geometry(window, pump):
    geometry = visualdynamics.import_file(fixture_path('plate',
                                                       'geometry.unv'))
    window.add_object('Geometry', geometry)
    window.tree.setCurrentItem(window.test_item.child(0))
    pump()
    return geometry


def offered(window, monkeypatch, answer=('', '')):
    """The dialog's file-type list, without showing a dialog."""
    seen = {}

    def fake(parent, title, start, filters):
        seen['title'] = title
        seen['filters'] = filters.split(';;')
        return answer

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(fake))
    window.save_selected()
    return seen


def test_the_file_menu_saves_the_project_and_nothing_else(window):
    # read each menu inside the loop: an abandoned generator's cleanup
    # tears the shiboken wrappers down mid-test
    labels = []
    bar = window.menuBar()
    for action in bar.actions():
        if action.text() == '&File':
            labels = [a.text() for a in action.menu().actions()]
    assert not any('Export' in text for text in labels)
    assert any('Save &Project' in text for text in labels)
    assert not any('Delete' in text for text in labels)


def test_the_context_menu_offers_one_save(window, geometry):
    assert window.save_action.text() == '&Save As...'
    assert not hasattr(window, 'export_action'), 'Export was the second name'
    # the action still acts on current_object(), the right-clicked one
    assert window.current_object() is geometry


def test_the_formats_are_the_dialogs_file_types(window, geometry, monkeypatch):
    """`.vdyn` first, because it is the only form that comes back
    whole; then whatever this object can be written as."""
    seen = offered(window, monkeypatch)
    assert seen['filters'][0] == 'Visual Dynamics object (*.vdyn)'
    assert any('unv' in f or 'UNV' in f for f in seen['filters'][1:]), \
        seen['filters']


def test_only_the_formats_this_object_has(window, pump, monkeypatch):
    """Said in the list rather than in an error afterwards: a report
    has one foreign form, the template it can be saved as (2026-09-08),
    plus the project layout in MATLAB's container (2026-09-18), so
    those and `.vdyn` are offered and nothing else."""
    from visualdynamics.core.report import Report

    window.add_object('Report', Report())
    window.tree.setCurrentItem(window._item_for_object('Report'))
    pump()
    seen = offered(window, monkeypatch)
    assert seen['filters'] == ['Visual Dynamics object (*.vdyn)',
                               'Report template (.vdreport) (*.vdreport)',
                               'MATLAB file (.mat) (*.mat)']


def test_choosing_a_foreign_type_writes_that_format(window, geometry,
                                                    monkeypatch, tmp_path):
    """And choosing the native one writes a `.vdyn` — one dialog, two
    outcomes, decided by the file type and nothing else."""
    unv = next(f for f in offered(window, monkeypatch)['filters']
               if '.unv' in f)
    out = tmp_path / 'geometry.unv'
    offered(window, monkeypatch, answer=(str(out), unv))
    assert out.exists()
    assert isinstance(visualdynamics.import_file(str(out)),
                      visualdynamics.Geometry)

    native = tmp_path / 'geometry.vdyn'
    offered(window, monkeypatch,
            answer=(str(native), 'Visual Dynamics object (*.vdyn)'))
    assert native.exists()


def test_a_suffix_is_added_when_the_dialog_gives_none(window, geometry,
                                                      monkeypatch, tmp_path):
    bare = tmp_path / 'geometry'
    offered(window, monkeypatch,
            answer=(str(bare), 'Visual Dynamics object (*.vdyn)'))
    assert (tmp_path / 'geometry.vdyn').exists()


def test_a_specification_is_offered_as_the_controllers_target(window, pump,
                                                              monkeypatch):
    """The format Rattlesnake loads before a random test is in the
    list — said there, beside sdynpy's array, rather than found."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.tree.setCurrentItem(window._item_for_object('Specification'))
    pump()
    seen = offered(window, monkeypatch)
    assert 'Rattlesnake random specification (.npz) (*.npz)' in seen['filters']
    assert 'sdynpy data array (.npz) (*.npz)' in seen['filters'], \
        'two formats share a suffix; the list tells them apart'

