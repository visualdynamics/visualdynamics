"""Compare paired responses on a shared clock and keep the original data."""

import numpy as np
import pytest

from visualdynamics import Geometry, Project, TimeHistory
from visualdynamics.core.filters import Filtering
from visualdynamics.core.time_compare import scores, sides


def _trace(t, response, dof='10Z-'):
    return TimeHistory(t, [response, np.ones_like(t)],
                       response_dof=[dof, '20Z+'], ordinate_dim=['velocity', 'force'])


def _pair(project):
    t = np.arange(1200)/1000
    y = np.sin(2*np.pi*20*t)
    a = project.add('Experiment', _trace(t, y))
    b = project.add('Model', _trace(t, np.where(t < 0.6, -2*y, np.cos(2*np.pi*20*t))))
    return a, b


def test_the_window_scores_shape_separately_from_sign_and_amplitude(tmp_path):
    project = Project()
    a, b = _pair(project)
    name = project.compare_time_histories(a, b, start=0, stop=0.599)
    first = scores(project[name])[0]
    assert first['pair'] == [a, 0, b, 0]  # not the common force channel
    assert first['trac'] == pytest.approx(1)
    assert first['correlation'] == pytest.approx(-1)
    assert first['rms_ratio'] == pytest.approx(2)
    assert first['strac'] == pytest.approx(-0.5)
    project.set_time_comparison_window(name, 0.6, 1.199)
    assert scores(project[name])[0]['trac'] == pytest.approx(0, abs=1e-10)
    path = tmp_path/'paired.vdyn'
    project.save(path)
    reopened = Project.open(path)
    assert reopened[name].time_comparison['start'] == 0.6
    assert scores(reopened[name])[0]['samples'] == 600
    settings = reopened[name].time_comparison
    left, right = sides(reopened[name], (settings['start'], settings['stop']))
    np.testing.assert_array_equal(left.abscissa, np.arange(600, 1200) / 1000)
    np.testing.assert_array_equal(right.abscissa, left.abscissa)
    assert reopened[name].abscissa[0] == 0


def test_different_node_names_require_an_explicit_pair_and_clocks_share_overlap():
    project = Project()
    t = np.arange(100)/100
    u = np.arange(10, 90)/100
    a = project.add('Experiment', _trace(t, t))
    b = project.add('Model', _trace(u, 3*u, '12Z-'))
    with pytest.raises(ValueError, match='expected two responses'):
        project.compare_time_histories(a, b)
    name = project.compare_time_histories(a, b, pairs=[(a, 0, b, 0)])
    np.testing.assert_array_equal(project[name].abscissa, u)
    assert scores(project[name])[0]['rms_ratio'] == pytest.approx(3)
    with pytest.raises(ValueError, match='common time interval'):
        project.set_time_comparison_window(name, 0, 0.8)


@pytest.mark.parametrize(('replacement', 'reason'), [
    (0., 'Zero-energy'), (1., 'Constant'), (np.nan, 'Nonfinite')])
def test_unscorable_responses_report_the_reason(replacement, reason):
    project = Project()
    a, b = _pair(project)
    project[b].ordinate[0] = replacement
    name = project.compare_time_histories(a, b)
    row = scores(project[name])[0]
    assert row['trac'] is None
    assert reason in row['reason']


def test_batch_filter_creates_copies_and_leaves_the_sources_alone():
    project = Project()
    a, b = _pair(project)
    before = project[a].ordinate.copy()
    filtered = project.filter_histories(a, b, filtering=Filtering(high=10))
    assert len(filtered) == 2
    assert np.std(project[filtered[0]].ordinate[0]) < 0.1*np.std(before[0])
    np.testing.assert_array_equal(project[a].ordinate, before)


def _select(window, names):
    from PySide6.QtCore import QSignalBlocker

    with QSignalBlocker(window.tree):
        window.tree.setCurrentItem(window._item_for_object(names[0]))
        window.tree.clearSelection()
        for name in names:
            window._item_for_object(name).setSelected(True)
    window._selection_changed()


def test_two_selected_histories_offer_filter_and_trac(window, pump):
    names = _pair(window.project)
    for name in names:
        window.show_object(name, select=False)
    _select(window, names)
    assert window.data_pane.filter_action.isVisible()
    assert window.time_tools.compare_action.isVisible()
    window.data_pane.waterfall_action.setChecked(False)
    window.time_tools.compare_action.trigger()
    pump()
    tools = window.time_tools
    assert tools.table.rowCount() == 1
    tools.stop.setFocus()
    pump()
    assert tools.stop.hasFocus()
    tools.stop.setValue(0.599)
    assert tools.stop.hasFocus(), 'redrawing a score must not hide the editor'
    assert tools.table.item(0, 1).text() == '1'
    tools.apply.click()
    assert 'Time Response Comparison' in window.project
    assert window.project['Time Response Comparison'].time_comparison['stop'] == 0.599
    _select(window, names)
    window.data_pane.filter_action.trigger()
    assert window.data_pane.filter_panel.isVisible()
    window.filter_data()
    assert {'Experiment Filtered', 'Model Filtered'} <= set(window.project)


def test_waterfall_draws_two_sides_at_one_station_with_trac_note(window, pump):
    names = _pair(window.project)
    result = window.project.compare_time_histories(*names)
    window.show_object(result)
    window.data_pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    from visualdynamics.gui import opengl

    pane = window.data_pane
    assert pane.waterfall_plotter is not None, {
        'checked': pane.waterfall_action.isChecked(), 'flat_only': pane.flat_only,
        'refusal': opengl.refusal(),
        'note': window.time_tools.note.text()}
    actors = pane.waterfall_plotter.actors
    assert {'paired-loud', 'paired-quiet'} <= set(actors)
    assert any(key.startswith('paired-end-notes') for key in actors)
    a = actors['paired-loud'].mapper.dataset.points[:, 1]
    b = actors['paired-quiet'].mapper.dataset.points[:, 1]
    np.testing.assert_array_equal(np.unique(a), np.unique(b))


def test_a_geometry_node_finds_only_its_linked_response_records(window):
    from PySide6.QtCore import QSignalBlocker

    from visualdynamics.gui.project_tree import ROLE_REFERENCE

    names = _pair(window.project)
    geo = Geometry(node_id=[10, 20], node_xyz=[[0, 0, 0], [1, 0, 0]], length_unit='m')
    window.add_object('Mesh', geo)
    window.project.add('Unlinked', _trace(np.arange(10), np.arange(10)))
    window.project.link('Mesh', *names)
    assert window.project.time_histories_at_node('Mesh', 10) == [(n, 0) for n in names]
    assert not window.project.time_histories_at_node('Mesh', 20)
    mesh = window._item_for_object('Mesh')
    nodes = next(mesh.child(i) for i in range(mesh.childCount())
                 if mesh.child(i).data(0, ROLE_REFERENCE) == ('component', 'Mesh', 'nodes'))
    window._populate_entities(nodes)
    with QSignalBlocker(window.tree):
        window.tree.setCurrentItem(nodes.child(0))
        window.tree.clearSelection()
        nodes.child(0).setSelected(True)
    window._selection_changed()
    assert [n for n, _data, _rows in window._linked_node_series()] == list(names)
    assert window.time_tools.compare_action.isVisible()


def test_renaming_a_source_preserves_pair_mapping_and_recompute():
    project = Project()
    a, b = _pair(project)
    name = project.compare_time_histories(a, b)
    project.rename(b, 'Updated model')
    assert project[name].time_comparison['pairs'][0][2] == 'Updated model'
    project['Updated model'].ordinate[0] *= 2
    assert name in project.stale()
    project.refresh(name)
    assert project[name].time_comparison['pairs'][0][2] == 'Updated model'
    assert name not in project.stale()


def test_rescale_applies_one_factor_to_every_selected_history(window, pump):
    names = _pair(window.project)
    for name in names:
        window.show_object(name, select=False)
    _select(window, names)
    pane = window.data_pane
    assert pane.rescale_action.isVisible()
    pane.rescale_action.trigger()
    pump()
    assert pane.showing('rescale')
    panel = pane.rescale_panel
    panel.set_settings({**panel.settings(), 'way': 'factor', 'factor': -2.0})
    panel.apply_asked.emit()
    for name in names:
        old, new = window.project[name], window.project[f'{name} Rescaled']
        np.testing.assert_array_equal(new.ordinate, -2*old.ordinate)
        assert window.project.provenance[f'{name} Rescaled']['verb'] == 'rescale'


def test_filtered_batch_collapses_separately_from_raw_histories(window):
    from visualdynamics.gui.project_tree import ROLE_REFERENCE

    sources = _pair(window.project)
    for name in sources:
        window.show_object(name, select=False)
    window.group_time_histories_action.setChecked(True)
    results = window.project.filter_histories(*sources)
    window.time_tools.add_results(results)
    sections = [window.test_item.child(i) for i in range(window.test_item.childCount())
                if (window.test_item.child(i).data(0, ROLE_REFERENCE) or (None,))[0]
                == 'history_section']
    header = next(row for row in sections if row.text(0).startswith('Filtered batch'))
    assert header.childCount() == len(results)
    header.setExpanded(False)
    assert header.data(0, ROLE_REFERENCE)[1] in window._closed_history_sections
    assert all(window._item_for_object(name).parent() is not header for name in sources)
    window._reorder_tree()
    assert not window._item_for_object(results[0]).parent().isExpanded()


def test_opening_project_does_not_select_each_history(window, tmp_path):
    project = Project()
    _pair(project)
    path = tmp_path / 'comparison.vdyn'
    project.save(path)
    selections = []
    window.tree.currentItemChanged.connect(
        lambda item, _old: selections.append(item.text(0)) if item else None)
    assert len(window.import_path(str(path))) == 2
    # the rows arrive unselected; only the last is selected, once, so
    # the opened project shows something without a redraw per history
    assert selections == ['Model']


def test_import_sections_survive_reopen_and_select_a_whole_trac_pair(window, tmp_path, pump):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from visualdynamics.gui.project_tree import ROLE_REFERENCE

    project = Project()
    a, b = _pair(project)
    for filename, name in [('experimental.unv', a), ('analytical.unv', b)]:
        source = tmp_path / filename
        source.write_bytes(b'UNV source')
        project.data_index.add(source, 'time histories', objects=[name], source='import')
    comparison = project.compare_time_histories(a, b)
    path = tmp_path / 'comparison.vdyn'
    project.save(path)

    window.group_time_histories_action.setChecked(True)
    window.import_paths([str(path)])
    headers = [item for i in range(window.test_item.childCount())
               if (item := window.test_item.child(i)).data(0, ROLE_REFERENCE)
               and item.data(0, ROLE_REFERENCE)[0] == 'history_section']
    assert [item.text(0) for item in headers] == ['experimental.unv', 'analytical.unv']
    header = headers[0]
    assert header.child(0) is window._item_for_object(a)
    rect = window.tree.visualItemRect(header)
    arrow = QPoint(rect.left() - window.tree.indentation() // 2, rect.center().y())
    QTest.mouseClick(window.tree.viewport(), Qt.MouseButton.LeftButton, pos=arrow)
    assert not header.isExpanded()
    QTest.mouseClick(window.tree.viewport(), Qt.MouseButton.LeftButton, pos=arrow)
    assert header.isExpanded()
    from PySide6.QtWidgets import QLineEdit

    title = QPoint(rect.left() + 40, rect.center().y())
    QTest.mouseClick(window.tree.viewport(), Qt.MouseButton.LeftButton, pos=title)
    QTest.mouseDClick(window.tree.viewport(), Qt.MouseButton.LeftButton, pos=title)
    pump()
    editor = window.tree.findChild(QLineEdit)
    assert editor is not None and editor.isVisible()
    editor.setText('Experiment')
    QTest.keyClick(editor, Qt.Key.Key_Return)
    pump()
    assert header.text(0) == 'Experiment'
    assert header.isExpanded()
    # the name is the project's, so it travels with the file
    assert list(window.project.history_group_names.values()) == ['Experiment']
    for suffix in ('vdyn', 'mat'):
        window.project.save(tmp_path / f'named.{suffix}')
        reopened = Project.open(tmp_path / f'named.{suffix}')
        assert reopened.history_group_names == window.project.history_group_names
    window._reorder_tree()
    headers = [window._item_for_object(name).parent() for name in (a, b)]
    assert headers[0].text(0) == 'Experiment'
    for header, modifier in zip(headers, [Qt.KeyboardModifier.NoModifier,
                                          Qt.KeyboardModifier.ControlModifier], strict=True):
        window.tree.scrollToItem(header)
        rect = window.tree.visualItemRect(header)
        QTest.mouseClick(window.tree.viewport(), Qt.MouseButton.LeftButton, modifier,
                         QPoint(rect.left() + 60, rect.center().y()))
    QTest.keyRelease(window.tree, Qt.Key.Key_Control)
    pump()
    assert window.time_tools.table.rowCount() == 1
    assert window.time_tools.table.item(0, 0).text() == '10Z-'
    assert window.time_tools.draft.time_comparison['pairs'] == [[a, 0, b, 0]]
    assert not window.selected_references()

    window.show_object(comparison)
    assert window.time_tools.table.rowCount() == 1
    assert window.time_tools.saved == comparison


def test_group_time_histories_is_remembered_and_starts_off(window, window_factory):
    assert not window.group_time_histories_action.isChecked()
    window.group_time_histories_action.setChecked(True)
    from visualdynamics.gui.preferences import remembered_group_histories

    assert remembered_group_histories()
    assert window_factory().group_time_histories_action.isChecked()


def test_the_pair_editor_is_the_shared_table(window, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog

    from visualdynamics.gui.tables import CopyPasteTableView

    a, b = _pair(window.project)
    tools = window.time_tools
    tools.series = [(a, window.project[a], None), (b, window.project[b], None)]
    refused = []

    def edit(dialog):
        model = dialog.findChild(CopyPasteTableView).model()
        model.edit_rejected.connect(refused.append)
        labels = model.columns[0].choices
        assert [model.data(model.index(0, c)) for c in (0, 1)] == labels[:2]
        assert not model.setData(model.index(0, 0), 'nothing', Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 0), labels[1], Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 1), labels[0], Qt.ItemDataRole.EditRole)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, 'exec', edit)
    monkeypatch.setattr(window, 'render_current', lambda: None)
    tools._manual_pairs()
    assert refused
    assert tools.explicit == [[b, 0, a, 0]]


def test_the_score_map_is_drawn_in_the_windows_own_3d_view(window, pump):
    names = _pair(window.project)
    comparison = window.project.compare_time_histories(*names)
    geo = Geometry(node_id=[10, 20], node_xyz=[[0, 0, 0], [1, 0, 0]], length_unit='m')
    window.add_object('Stations', geo)
    window.project.link('Stations', comparison)
    window.show_object(comparison)
    pump()
    tools = window.time_tools
    assert window.data_pane.isVisible()
    assert not window.scene.isVisible(), 'no map asked for, no 3-D view'
    tools.map_mode.setCurrentIndex(1)                 # TRAC points
    window.render_current()
    pump()
    plotter = window.scene.plotter
    assert window.scene.isVisible() and window.data_pane.isVisible()
    assert 'correlation-average' in plotter.actors
    assert 'TRAC' in plotter.scalar_bars
    assert tools.correlation_count == 1
    assert '1 scored stations' in tools.note.text()
    # a new reading of the same stations keeps the camera where it was put
    plotter.camera.position = (5.0, 5.0, 5.0)
    held = plotter.camera.position
    tools.map_mode.setCurrentIndex(2)                 # sTRAC points
    window.render_current()
    pump()
    assert 'sTRAC' in plotter.scalar_bars and 'TRAC' not in plotter.scalar_bars
    assert plotter.camera.position == pytest.approx(held)
    tools.map_mode.setCurrentIndex(0)
    window.render_current()
    pump()
    assert not window.scene.isVisible()



def test_a_click_on_a_station_selects_its_row_and_plots_its_traces(window, pump):
    from pyqtgraph import PlotItem

    from visualdynamics.viz.time_correlation import station_pixel

    t = np.arange(1200)/1000
    y = np.sin(2*np.pi*20*t)
    project = window.project
    names = [project.add(f'{side} {dof}', _trace(t, y if side == 'Test' else 2*y + t, dof))
             for side in ('Test', 'Model') for dof in ('30Z-', '40Z-')]
    comparison = project.compare_time_histories(*names)
    geo = Geometry(node_id=[30, 40], node_xyz=[[0, 0, 0], [1, 0, 0]], length_unit='m')
    window.add_object('Stations', geo)
    project.link('Stations', comparison)
    window.show_object(comparison)
    pump()
    tools = window.time_tools
    tools.map_mode.setCurrentIndex(1)                 # TRAC points
    window.render_current()
    pump()
    plotter = window.scene.plotter
    plotter.render()
    assert [tools.table.item(i, 0).text() for i in range(2)] == ['30Z-', '40Z-']
    assert tools.table.currentRow() == 0
    x, y = station_pixel(plotter, 1)
    # through the press handler, as a mouse press arrives (the headless
    # view has no interactor to send one through); the handler stops
    # the camera from starting a turn when the press was a station's
    aborted = []

    class Press:
        def GetEventPosition(self):
            return round(x), round(y)

        def GetCommand(self, _observer):
            return self

        def SetAbortFlag(self, flag):
            aborted.append(flag)

    window.scene._cube_observer = None
    window.scene._on_cube_press(Press(), None)
    pump()
    assert aborted == [1]
    assert tools.table.currentRow() == 1
    assert tools.table.item(1, 0).isSelected()
    titles = {plot.titleLabel.text for plot in window.data_pane.graphics.ci.items
              if isinstance(plot, PlotItem)}
    assert titles and all('40Z-' in title for title in titles)
    # a click away from every station leaves the row alone
    assert not window.scene.click_station(2.0, 2.0)
    assert tools.table.currentRow() == 1

def test_a_group_row_wears_a_folder(window):
    from visualdynamics.gui.icons import group_icon, object_icon
    from visualdynamics.gui.project_tree import ROLE_REFERENCE

    sources = _pair(window.project)
    for name in sources:
        window.show_object(name, select=False)
    window.group_time_histories_action.setChecked(True)
    header = next(window.test_item.child(i) for i in range(window.test_item.childCount())
                  if (window.test_item.child(i).data(0, ROLE_REFERENCE) or (None,))[0]
                  == 'history_section')

    def ink(icon):
        image = icon.pixmap(16, 16).toImage()
        return [image.pixelColor(x, y).alpha() > 0
                for y in range(image.height()) for x in range(image.width())]

    shown = ink(header.icon(0))
    assert sum(shown) > 40, 'a glyph a person can see at 16 px'
    assert shown == ink(group_icon())
    assert shown != ink(object_icon(window.project[sources[0]]))


def test_a_history_group_name_is_set_cleared_and_journaled():
    project = Project()
    assert project.name_history_group('Rescaled time histories', ' Scaled ') == 'Scaled'
    assert project.history_group_names == {'Rescaled time histories': 'Scaled'}
    assert project.journal[-1] == (
        "project.name_history_group('Rescaled time histories', ' Scaled ')")
    assert project.name_history_group('Rescaled time histories', '') is None
    assert project.history_group_names == {}
