"""An exodus modal file whose rigid-body modes came out slightly negative.

An eigensolver reports a rigid-body eigenvalue as a number near zero of
either sign, and the file writes the frequency as it came out: -3.2e-7
Hz. A ShapeSet refuses a negative frequency, so such a file did not
import at all (Brandon, 2026-09-25). It imports now: the mode is a
rigid-body mode and reads as 0 Hz, the file's own value is kept on the
mode's description, and the import says so — as a warning a script's
log shows, and in a dialog at the window, which had no channel for a
reader's caveats before.
"""

from __future__ import annotations

import warnings

import netCDF4
import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.geometry import Geometry
from visualdynamics.core.shapes import ShapeSet


def _modal_file(tmp_path, frequencies):
    geometry = Geometry(node_id=[1, 2, 3],
                        node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                        length_unit='m')
    shapes = ShapeSet([0.0, 0.0, 12.5], [0.0, 0.0, 0.01],
                      ['1Z+', '2Z+', '3Z+'],
                      [[1, 1, 1], [1, 0, -1], [1, -2, 1]])
    path = tmp_path / 'modes.exo'
    visualdynamics.export_file(shapes, str(path), format='exodus',
                               geometry=geometry)
    # the writer nudges coincident frequencies apart; the file under test
    # carries exactly the solver's numbers
    with netCDF4.Dataset(path, 'r+') as ds:
        ds.variables['time_whole'][:] = np.asarray(frequencies)
    return str(path)


def test_negative_frequencies_import_as_rigid_body_modes_and_say_so(tmp_path):
    path = _modal_file(tmp_path, [-3.2e-7, -1.1e-8, 12.5])
    from visualdynamics.io import ImportNote

    with pytest.warns(ImportNote, match=r'2 modes with a negative frequency '
                      r'\(-3\.2e-07, -1\.1e-08 Hz\) read as 0 Hz'):
        back = visualdynamics.import_file(path)
    shapes = back['shapes']
    assert list(shapes.frequency) == [0.0, 0.0, 12.5], (
        'the negative ones are zero, the real one untouched')
    assert shapes.description[0] == (
        'frequency -3.2e-07 Hz in the file, read as 0 (a rigid-body mode)')
    assert shapes.description[1].startswith('frequency -1.1e-08 Hz')
    assert shapes.description[2] == '', 'nothing said about a good mode'


def test_a_file_with_no_negative_frequency_is_read_in_silence(tmp_path):
    path = _modal_file(tmp_path, [0.0, 1e-9, 12.5])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        back = visualdynamics.import_file(path)
    assert not [w for w in caught if 'negative frequency' in str(w.message)]
    assert back['shapes'].description == ['', '', '']


def _answer_dialogs(window_module, monkeypatch, shown):
    """The window asks how an exodus file's steps read; a person picks
    mode shapes. The report dialogs are recorded rather than shown."""
    monkeypatch.setattr(
        window_module.QInputDialog, 'getItem',
        lambda parent, title, label, items, *rest: (items[0], True))
    for kind in ('information', 'warning'):
        monkeypatch.setattr(
            window_module.QMessageBox, kind,
            lambda parent, title, text: shown.append((title, text)))


def test_the_window_shows_the_note_and_keeps_the_shapes(window, pump,
                                                       tmp_path, monkeypatch):
    """A warning to stderr is a warning nobody at the window sees: the
    reader's note lands in a dialog, and the shapes land in the tree."""
    from visualdynamics.gui import main_window as window_module

    shown = []
    _answer_dialogs(window_module, monkeypatch, shown)
    path = _modal_file(tmp_path, [-3.2e-7, 0.0, 12.5])
    window.import_paths([path])
    pump()
    assert any(isinstance(o, ShapeSet) for o in window.objects.values()), \
        'the file imported'
    assert len(shown) == 1
    title, text = shown[0]
    assert title == 'Imported, with a note'
    assert '1 mode with a negative frequency (-3.2e-07 Hz) read as 0 Hz' in text


def test_a_librarys_warning_is_not_a_note(window, pump, tmp_path, monkeypatch):
    """Only a reader's own ImportNote reaches the dialog. The first day
    an unclosed-socket ResourceWarning from somewhere under an import
    went into "Imported, with a note", which is not what it was."""
    from visualdynamics.gui import main_window as window_module

    shown = []
    _answer_dialogs(window_module, monkeypatch, shown)
    real = window_module.io.import_file

    def noisy(path, **options):
        warnings.warn('unclosed <socket.socket fd=51>', ResourceWarning,
                      stacklevel=2)
        warnings.warn('something old', DeprecationWarning, stacklevel=2)
        return real(path, **options)

    monkeypatch.setattr(window_module.io, 'import_file', noisy)
    path = _modal_file(tmp_path, [0.0, 0.0, 12.5])
    window.import_paths([path])
    pump()
    assert any(isinstance(o, ShapeSet) for o in window.objects.values())
    assert shown == [], 'no dialog for a warning that is not about the file'


def test_a_failure_and_a_note_share_the_one_dialog(window, pump, tmp_path,
                                                   monkeypatch):
    from visualdynamics.gui import main_window as window_module

    shown = []
    _answer_dialogs(window_module, monkeypatch, shown)
    broken = tmp_path / 'broken.unv'
    broken.write_text('not a universal file\n')
    good = _modal_file(tmp_path, [-3.2e-7, 0.0, 12.5])
    window.import_paths([good, str(broken)])
    pump()
    assert len(shown) == 1, 'one dialog, not one per kind of message'
    title, text = shown[0]
    assert title == 'Some files could not be imported'
    assert 'broken.unv' in text and 'negative frequency' in text
