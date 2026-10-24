"""Importing delimited text in the window: the layout found, shown on a
preview of the channels, set where it is wrong.

The dialog is driven headless here, its boxes set the way a person
sets them, and each test pins what the import would then be told
(`options()`, the keywords a script passes), what the preview read, and
whether Import is offered. The window's own import is driven with the
dialog's answer stubbed, as the stream-window tests do.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_delimited_import import G, _csv, _read, _signal, _write

import visualdynamics
from visualdynamics.core.data import TimeHistory

# ---- the picture -----------------------------------------------------------------


def test_the_preview_draws_a_row_per_channel_and_names_what_it_found(
        qt_app, tmp_path):
    import pyqtgraph as pg

    from visualdynamics.plot.text_import import build_import_preview

    t, a, _f = _signal(64)
    rows = [f'{x:.17g},{y:.17g},0,{"" if 10 <= k < 13 else y:.17}'
            for k, (x, y) in enumerate(zip(t, a, strict=True))]
    path = _write(tmp_path, 'Time (s),A (g),B (g),C (g)\n' + '\n'.join(rows))
    history, _notes = _read(path)
    widget = pg.GraphicsLayoutWidget()
    plots = build_import_preview(widget, history, theme='light')
    assert [p.titleLabel.text for p in plots] == [
        'A', 'B: constant at 0: a dead or unconnected channel',
        'C: 3 missing (crosses), the first at 0.001953 s']
    [mark] = [item for item in plots[2].listDataItems()
              if item.opts.get('symbol') == 'x']
    x, y = mark.getData()
    assert np.allclose(x, t[10:13]) and not y.any(), 'at its time, on zero'
    assert 'g' in plots[0].getAxis('left').labelText
    from PySide6.QtCore import Qt

    styles = [p.listDataItems()[0].opts['pen'].style() for p in plots[:2]]
    assert styles == [Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine], (
        'a dead channel dashed, not told by its color')
    few = build_import_preview(pg.GraphicsLayoutWidget(), history, rows=2)
    assert few[-1].titleLabel.text.endswith('(and 1 more channel not drawn)')


def test_the_preview_draws_from_a_script_to_a_file(qt_app, tmp_path):
    path, _ = _csv(tmp_path)
    target = tmp_path / 'preview.png'
    visualdynamics.plot.plot_import_preview(path, path=str(target))
    assert target.exists() and target.stat().st_size > 1000


# ---- the dialog ------------------------------------------------------------------

def _dialog(window, path):
    from visualdynamics.gui.text_import import TextImportDialog
    from visualdynamics.theme import theme as resolve_theme

    return TextImportDialog(path, resolve_theme(window.theme_name))


def test_the_dialog_opens_on_what_was_found(window, tmp_path):
    path, _ = _csv(tmp_path, samples=300)
    dialog = _dialog(window, path)
    assert dialog.header.text().startswith(
        'run.csv: 3 columns separated by comma, names on line 1, data from line 2')
    assert dialog.time_box.currentText() == 'Time'
    assert [dialog.table.item(r, 0).text() for r in range(2)] == ['Accel', 'Force']
    assert [dialog.table.item(r, 2).text() for r in range(2)] == ['g', 'lbf']
    assert dialog.options() == {}, 'untouched: the plain import'
    assert dialog.history.ordinate.shape == (2, 300)
    assert dialog.import_button.isEnabled()
    assert dialog.message.text() == '2 channels at 5120 samples per second.'


def test_no_time_column_needs_a_rate_before_it_imports(window, tmp_path):
    path, _ = _csv(tmp_path)
    dialog = _dialog(window, path)
    dialog.time_box.setCurrentIndex(0)
    assert not dialog.import_button.isEnabled()
    assert 'neither a time column nor a stated sample rate' in dialog.message.text()
    assert dialog.table.rowCount() == 3, 'the old time column is a channel now'
    dialog.rate_box.setValue(2048.0)
    assert dialog.import_button.isEnabled()
    assert dialog.options() == {'time_column': False, 'sample_rate': 2048.0}
    assert dialog.history.sample_rate == 2048.0


def test_a_rate_against_the_time_column_is_refused_in_the_dialog(window, tmp_path):
    path, _ = _csv(tmp_path)
    dialog = _dialog(window, path)
    dialog.rate_box.setValue(5000.0)
    assert not dialog.import_button.isEnabled()
    assert dialog.message.text().startswith('Cannot import: ')
    assert 'one of them is wrong' in dialog.message.text()


def test_columns_names_and_units_set_in_the_table(window, tmp_path):
    from PySide6.QtCore import Qt

    path, _ = _csv(tmp_path, header='Time (s),Accel (Gs),Force (lbf)')
    dialog = _dialog(window, path)
    assert "'Gs' in the header reads as a time" in dialog.message.text()
    dialog.table.item(0, 1).setText('101Z')
    dialog.table.item(0, 2).setText('g')
    assert dialog.options() == {'names': ['101Z', 'Force'], 'units': {'Accel': 'g'}}
    assert dialog.history.response_dof == ['101Z+', '']
    assert dialog.history.ordinate_unit == ['g', 'lbf']
    assert dialog.message.text() == '2 channels at 5120 samples per second.'
    dialog.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert dialog.options()['columns'] == ['Accel']
    assert dialog.options()['names'] == ['101Z']
    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not dialog.import_button.isEnabled()
    assert 'no column is checked' in dialog.message.text()


def test_a_wrong_delimiter_is_refused_and_the_right_one_restores(window, tmp_path):
    path, _ = _csv(tmp_path)
    dialog = _dialog(window, path)
    dialog.delimiter_box.setCurrentIndex(dialog.delimiter_box.findData('\t'))
    assert not dialog.import_button.isEnabled()
    assert 'not read as delimited text' in dialog.message.text()
    dialog.delimiter_box.setCurrentIndex(dialog.delimiter_box.findData(','))
    assert dialog.import_button.isEnabled() and dialog.options() == {}


def test_a_stated_delimiter_rides_the_answer(window, tmp_path):
    t, a, f = _signal(40)
    rows = [f'{x:.9f};{y:.6f};{z:.6f}'.replace('.', ',')
            for x, y, z in zip(t, a, f, strict=True)]
    path = _write(tmp_path, 'Time;A (g);B\n' + '\n'.join(rows))
    dialog = _dialog(window, path)
    assert dialog.decimal_box.currentData() == ','
    dialog.decimal_box.setCurrentIndex(dialog.decimal_box.findData('.'))
    assert not dialog.import_button.isEnabled(), 'the numbers no longer read'
    dialog.decimal_box.setCurrentIndex(dialog.decimal_box.findData(','))
    assert dialog.import_button.isEnabled()
    assert np.allclose(dialog.history.ordinate[0], a * G, atol=1e-5)


def test_the_preview_reads_the_head_of_the_file(window, tmp_path, monkeypatch):
    from visualdynamics.gui import text_import

    monkeypatch.setattr(text_import, 'PREVIEW_ROWS', 100)
    path, _ = _csv(tmp_path, samples=1000)
    dialog = _dialog(window, path)
    assert dialog.history.ordinate.shape == (2, 100)
    assert 'up to 100 rows' in dialog.header.text()


# ---- the window ------------------------------------------------------------------

def _histories(window):
    return [obj for obj in window.project.values() if isinstance(obj, TimeHistory)]


def test_the_window_imports_what_the_dialog_chose(window, pump, tmp_path,
                                                  monkeypatch):
    from visualdynamics.gui import text_import

    path, (_t, a, _f) = _csv(tmp_path)
    asked = []

    def choose(parent, asked_path, colors):
        asked.append(asked_path)
        return {'columns': ['Accel'], 'names': ['101Z+']}

    monkeypatch.setattr(text_import, 'ask_text_import', choose)
    window.import_paths([path])
    pump()
    assert asked == [path]
    [history] = _histories(window)
    assert history.response_dof == ['101Z+']
    assert np.allclose(history.ordinate[0], a * G)
    assert (f"project.import_file({path!r}, columns=['Accel'], "
            "names=['101Z+'])") in window.project.journal


def test_cancelling_the_dialog_skips_the_file(window, pump, tmp_path, monkeypatch):
    from visualdynamics.gui import text_import

    path, _ = _csv(tmp_path)
    monkeypatch.setattr(text_import, 'ask_text_import', lambda *a, **k: None)
    window.import_paths([path])
    pump()
    assert _histories(window) == []


def test_a_file_that_is_not_text_is_not_asked_about(window, pump, monkeypatch):
    from conftest import fixture_path

    from visualdynamics.gui import text_import

    monkeypatch.setattr(text_import, 'ask_text_import',
                        lambda *a, **k: pytest.fail('asked about a UNV'))
    window.import_paths([fixture_path('plate', 'geometry.unv')])
    pump()


def test_the_import_notes_reach_the_window(window, pump, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from visualdynamics.gui import text_import

    t, a, _f = _signal(32)
    path = _write(tmp_path, 'Time (s),A (g),B (g)\n' + '\n'.join(
        f'{x:.17g},{y:.17g},0' for x, y in zip(t, a, strict=True)))
    monkeypatch.setattr(text_import, 'ask_text_import', lambda *a, **k: {})
    told = []
    monkeypatch.setattr(QMessageBox, 'information',
                        lambda _parent, title, text: told.append((title, text)))
    window.import_paths([path])
    pump()
    assert told == [('Imported, with a note',
                     ('run.csv: B is constant at 0 g over every present '
                      'sample: a dead or unconnected channel'))]
