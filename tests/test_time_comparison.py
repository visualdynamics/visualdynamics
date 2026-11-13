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


def test_batch_filter_creates_copies_and_scaling_leaves_reference_force_alone():
    project = Project()
    a, b = _pair(project)
    before = project[a].ordinate.copy()
    scaled = project.scale_time_histories(a, b, factor=-3)
    np.testing.assert_array_equal(project[scaled[0]].ordinate[0], -3*before[0])
    np.testing.assert_array_equal(project[scaled[0]].ordinate[1], before[1])
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
    actors = window.data_pane.waterfall_plotter.actors
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


def test_units_list_waits_until_the_opening_mouse_button_is_released(window, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QComboBox

    from visualdynamics.gui import tables

    shown = []

    class MenuProbe(QComboBox):
        def showPopup(self):
            shown.append(True)

    editor = MenuProbe(window)
    editor.addItems(['m', 'mm'])
    editor.show()
    pending = []
    monkeypatch.setattr(tables.QTimer, 'singleShot', lambda _delay, call: pending.append(call))
    monkeypatch.setattr(QApplication, 'mouseButtons', lambda: Qt.MouseButton.LeftButton)
    tables._open_list(editor)
    assert not shown
    assert len(pending) == 1
    monkeypatch.setattr(QApplication, 'mouseButtons', lambda: Qt.MouseButton.NoButton)
    pending[0]()
    assert shown == [True]


def test_scale_preview_applies_to_both_selected_histories(window, pump):
    names = _pair(window.project)
    for name in names:
        window.show_object(name, select=False)
    _select(window, names)
    tools = window.time_tools
    tools.scale_action.trigger()
    pump()
    tools.factor.setValue(-2)
    assert tools.apply.isEnabled()
    tools.apply.click()
    for name in names:
        old, new = window.project[name], window.project[f'{name} Scaled']
        np.testing.assert_array_equal(new.ordinate[0], -2*old.ordinate[0])
        np.testing.assert_array_equal(new.ordinate[1], old.ordinate[1])


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
    header = next(row for row in sections if row.text(0).startswith('▾ Filtered batch'))
    window._toggle_history_section(header, 0)
    assert all(window._item_for_object(name).isHidden() for name in results)
    assert all(not window._item_for_object(name).isHidden() for name in sources)
    window._toggle_history_section(header, 0)
    assert all(not window._item_for_object(name).isHidden() for name in results)


def test_opening_project_does_not_select_each_history(window, tmp_path):
    project = Project()
    _pair(project)
    path = tmp_path / 'comparison.vdyn'
    project.save(path)
    selections = []
    window.tree.currentItemChanged.connect(
        lambda item, _old: selections.append(item.text(0)) if item else None)
    assert len(window.import_path(str(path))) == 2
    assert not {'Experiment', 'Model'}.intersection(selections)
    assert window.tree.currentItem() is window.test_item


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
    assert [item.text(0).split()[1] for item in headers] == ['experimental.unv', 'analytical.unv']
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


def test_windows_first_time_trace_waits_for_an_explicit_3d_request(window, monkeypatch):
    from visualdynamics.gui import opengl

    monkeypatch.setattr(window.data_pane, '_defer_first_waterfall', True, raising=False)
    monkeypatch.setattr(opengl, '_found', [])
    calls = []
    monkeypatch.setattr(opengl, 'ready', lambda: calls.append('OpenGL') or False)
    name, _ = _pair(window.project)
    window.data_pane.waterfall_action.setChecked(True)
    window.show_object(name)
    assert calls == []
    window.data_pane.waterfall_action.trigger()
    assert calls
