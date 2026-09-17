"""What the window says when something is wrong with what it was given.

Two messages that only ever appeared in the end-to-end script, both of
them the only feedback the user gets in their situation:

- a file that will not import at all, dropped among files that will. The
  ones that work still arrive; the ones that do not are named together in
  one dialog rather than silently missing.
- a row painted incompatible. The color says *that* something is wrong;
  hovering it is where the reason lives, and moving off has to put back
  whatever the status bar was saying before.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.shapes import ShapeSet


def test_the_files_that_import_still_do_and_the_rest_are_named(window, pump,
                                                               tmp_path,
                                                               monkeypatch):
    from visualdynamics.gui import main_window as window_module

    broken = tmp_path / 'not-really.unv'
    broken.write_text('this is not a universal file\n')
    shown = []
    monkeypatch.setattr(window_module.QMessageBox, 'warning',
                        lambda parent, title, text: shown.append((title, text)))

    imported = window.import_paths([fixture_path('plate', 'geometry.exo'),
                                    str(broken)])
    pump()
    assert len(imported) == 1, 'the good one arrived'
    assert 'Geometry' in window.objects
    assert shown, 'and the bad one was reported rather than dropped'
    title, text = shown[0]
    assert 'could not be imported' in title
    assert 'not-really.unv' in text, 'named, so it can be looked at'


def test_hovering_an_incompatible_row_explains_it(window, pump):
    """And moving off puts back what the status bar was saying."""
    geometry = visualdynamics.import_file(fixture_path('plate',
                                                       'geometry.npz'))
    stranger = ShapeSet(frequency=[10.0], damping=[0.01],
                        coordinate=['90001X+'],
                        shape_matrix=np.array([[1.0]]))
    window.add_object('Geometry', geometry)
    window.add_object('Elsewhere', stranger)
    window.refresh_compatibility()
    pump()
    assert window.report is not None
    assert not window.report.is_compatible('Elsewhere'), 'the fixture is the case'

    window.tree.clearSelection()
    item = window._item_for_object('Geometry')
    window.tree.setCurrentItem(item)
    item.setSelected(True)
    pump()
    resting = window.statusBar().currentMessage()

    odd = window._item_for_object('Elsewhere')
    window._hover_item(odd)
    explained = window.statusBar().currentMessage()
    assert explained != resting and explained, 'the reason, while hovering it'
    assert explained == window.report.issue_for('Elsewhere').message

    window._hover_item(item)
    assert window.statusBar().currentMessage() == resting, (
        'and off it again, what was there before')


def test_hovering_says_nothing_before_anything_is_checked(window):
    """No project, no report: hovering must not reach through a None."""
    window.report = None
    window._hover_item(None)
    assert True, 'it returned rather than raising'


def test_a_malformed_file_is_reported_rather_than_thrown(window, pump,
                                                          tmp_path,
                                                          monkeypatch):
    """A file refused *by the objects* raises ValueError, but one that is
    simply malformed raises whatever its reader trips over first — an
    IndexError off the end of a short UNV record, a KeyError for a
    missing netCDF variable. Those used to escape into the Qt event
    loop, where a traceback goes nowhere and the import appears to do
    nothing at all."""
    from visualdynamics.gui import main_window as window_module

    broken = tmp_path / 'broken.unv'
    broken.write_text('    -1\n    58\nnot a real dataset\n    -1\n')
    shown = []
    monkeypatch.setattr(window_module.QMessageBox, 'warning',
                        lambda parent, title, text: shown.append((title, text)))

    imported = window.import_paths([fixture_path('plate', 'geometry.exo'),
                                    str(broken)])
    pump()
    assert len(imported) == 1, 'the good file still arrived'
    assert shown, 'and the bad one was named'
    _title, text = shown[0]
    assert 'broken.unv' in text
    assert 'IndexError' in text, (
        'the kind of failure, since the message alone reads as nonsense')


def test_a_message_box_under_a_headless_test_fails_by_name(window, pump,
                                                          tmp_path):
    """The fixture's guard: a failed import wants to raise a warning
    box, and with nobody to click it a headless run would hang there
    — CI did, for 25 minutes, on a project file absent from the runner
    (2026-09-13). The box raises instead, naming itself."""
    missing = tmp_path / 'nowhere.vdyn'
    with pytest.raises(AssertionError, match='QMessageBox.warning'):
        window.import_paths([str(missing)])
