"""A report on its own is a template (Brandon, 2026-09-08).

Exported without its project, a report is its blocks and their
bindings: the same figures and text, bound afresh to whatever project
loads it. Saved into the templates folder it is offered by Generate
Report beside the built-in ones.
"""

from __future__ import annotations

import json

import pytest

import visualdynamics
from visualdynamics.core.report import Report, modal_template
from visualdynamics.io import report_template


@pytest.fixture
def templates(tmp_path, monkeypatch):
    """The templates folder, pointed at a scratch directory."""
    folder = tmp_path / 'Report Templates'
    monkeypatch.setenv('VISUALDYNAMICS_TEMPLATES', str(folder))
    return folder


def test_a_report_round_trips_as_a_template(tmp_path):
    report = modal_template({})
    report.marking = 'SECRET'
    report.marking_color = 'red'
    path = tmp_path / 'house.vdreport'
    report_template.save(report, path)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data[report_template.MARKER] == 1
    assert data['blocks'] == report.blocks, 'the bindings go out as they stand'
    back = report_template.load(path)
    assert back.title == report.title and back.blocks == report.blocks
    assert back.marking == 'SECRET' and back.marking_color == 'red'
    assert report_template.sniff(path)
    as_json = tmp_path / 'house.json'
    as_json.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
    assert report_template.sniff(as_json), 'recognized by its marker too'
    other = tmp_path / 'other.json'
    other.write_text('{"blocks": []}', encoding='utf-8')
    assert not report_template.sniff(other)
    with pytest.raises(ValueError, match='not a Visual Dynamics report template'):
        report_template.load(other)


def test_the_project_exports_and_imports_it(project, tmp_path):
    source = visualdynamics.Project()
    for name, obj in project.items():
        source.add(name, obj)
    made = source.generate_report('modal')
    source[made].blocks[1]['caption'] = 'House caption'
    path = source.export(made, tmp_path / 'house.vdreport')
    assert path.endswith('.vdreport')
    assert source.journal[-1] == \
        f"project.export({made!r}, {str(tmp_path / 'house.vdreport')!r})"
    # into another project, empty: every bound block is a card to rebind
    target = visualdynamics.Project()
    [added] = target.import_file(path)
    loaded = target[added]
    assert isinstance(loaded, Report)
    assert loaded.blocks == source[made].blocks
    assert loaded.blocks[1]['caption'] == 'House caption'
    assert loaded.unbound(target) != [], 'nothing to bind to yet'
    # into one with the objects: the symbolic bindings resolve
    for name, obj in project.items():
        target.add(name, obj)
    assert loaded.unbound(target, target.links) == source[made].unbound(
        source, source.links)


def test_generate_report_takes_a_saved_template_by_name_or_path(project,
                                                                 templates):
    p = visualdynamics.Project()
    for name, obj in project.items():
        p.add(name, obj)
    assert report_template.saved_templates() == []
    templates.mkdir(parents=True)
    report = Report('House Style', [{'kind': 'text', 'text': '# Ours'}])
    report_template.save(report, templates / 'House.vdreport')
    assert report_template.saved_templates() == [
        ('House', templates / 'House.vdreport')]
    by_name = p.generate_report('House', name='From Name')
    assert p[by_name].title == 'House Style'
    assert p[by_name].blocks == report.blocks
    by_path = p.generate_report(str(templates / 'House.vdreport'),
                                name='From Path')
    assert p[by_path].blocks == report.blocks
    assert p[by_path] is not p[by_name], 'each generation is its own report'
    with pytest.raises(ValueError, match='unknown template'):
        p.generate_report('Nobody')
    assert 'House' in str(pytest.raises(
        ValueError, p.generate_report, 'Nobody').value), \
        'the refusal names what is saved'


def test_the_report_bar_saves_a_template_and_the_menu_offers_it(
        window, pump, project, templates, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    for name, obj in project.items():
        window.add_object(name, obj)
    name = window.generate_report('modal')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    editor = window.report_editor
    labels = [a.text() for a in editor.export_menu.actions()]
    assert labels == ['HTML…', 'Report template…']
    asked = []

    def choose(_parent, _title, start, _filter):
        asked.append(start)
        return start, ''

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(choose))
    editor.export_menu.actions()[1].trigger()
    pump()
    assert asked == [str(templates / f'{name}.vdreport')], \
        'the dialog opens in the templates folder, named for the report'
    assert (templates / f'{name}.vdreport').is_file()
    assert 'Generate Report offers it' in window.statusBar().currentMessage()
    assert window.project.journal[-1].startswith(f'project.export({name!r}, ')
    # and Generate Report offers it — even in a typed project, where
    # the act would otherwise generate the typed report at once
    popped = []
    monkeypatch.setattr(window, '_pop_menu', lambda menu: popped.append(menu))
    window.tree.clearSelection()
    window.test_item.setSelected(True)
    window.tree.setCurrentItem(window.test_item)
    pump()
    window.generate_report_act()
    texts = [a.text() for a in popped[-1].actions() if not a.isSeparator()]
    assert texts[-1] == f'Generate {name} Report (saved template)'
    assert len(texts) == 9, 'the eight built-in, then the saved one'
    window.set_project_type('Modal Test')
    pump()
    window.generate_report_act()
    texts = [a.text() for a in popped[-1].actions() if not a.isSeparator()]
    assert texts == ['Generate Modal Test Report',
                     f'Generate {name} Report (saved template)']
    popped[-1].actions()[-1].trigger()
    pump()
    from visualdynamics.core.report import Report as R
    reports = [n for n, o in window.objects.items() if isinstance(o, R)]
    assert len(reports) == 2, 'the saved template generated a second report'
