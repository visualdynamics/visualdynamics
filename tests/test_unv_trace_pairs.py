"""Labeled acquisition pairs keep their reference beside each response."""

import numpy as np
import pytest

from visualdynamics import Project, TimeHistory
from visualdynamics.io import unv


def trace_file(tmp_path, *, units=True, incomplete=False):
    """An export with the measurement DOF repeated on both trace headers."""
    blocks = [unv._units_dataset()] if units else []
    t = np.arange(8) / 8
    for node in (730, 1125):
        data = TimeHistory(
            t, np.array([np.arange(8), -np.arange(8)]) + node,
            response_dof=[f'{node}Z-'] * 2, reference_dof=['1Z+'] * 2,
            ordinate_dim=['velocity', 'force'])
        for i, label in enumerate(('Response Time Trace', 'Reference Time Trace')):
            if incomplete and node == 1125 and i == 1:
                continue
            lines = unv._58_dataset(data, i).splitlines(keepends=True)
            lines[2] = label + '\n'
            blocks.append(''.join(lines))
    path = tmp_path / 'traces.unv'
    path.write_text(''.join(blocks))
    return path


@pytest.mark.parametrize('units', [True, False])
def test_each_response_keeps_its_own_reference_trace(tmp_path, units):
    path = trace_file(tmp_path, units=units)
    project = Project()
    names = project.import_file(path, reference_node=1013)
    assert names == ['Time History - 730Z- - ref 1013Z+',
                     'Time History - 1125Z- - ref 1013Z+']
    for node, name in zip((730, 1125), names):
        data = project[name]
        assert data.response_dof == [f'{node}Z-', '1013Z+']
        assert data.reference_dof is None
        assert data.roles == {(f'{node}Z-', 'velocity'): 'response',
                              ('1013Z+', 'force'): 'reference'}
        np.testing.assert_array_equal(data.ordinate,
                                      np.array([np.arange(8), -np.arange(8)]) + node)
        np.testing.assert_array_equal(data.abscissa, np.arange(8) / 8)
        assert data.units_defined == units
    saved = tmp_path / 'saved.vdyn'
    project.save(saved)
    restored = Project.open(saved)
    assert list(restored) == names
    assert restored[names[0]].roles == project[names[0]].roles


def test_without_an_override_the_file_reference_is_preserved(tmp_path):
    data = next(iter(unv.load(trace_file(tmp_path)).values()))
    assert data.response_dof == ['730Z-', '1Z+']


def test_an_incomplete_labeled_pair_names_the_measurement(tmp_path):
    with pytest.raises(ValueError, match='1125Z-: expected one response'):
        unv.load(trace_file(tmp_path, incomplete=True))


def test_ordinary_multichannel_histories_stay_together(tmp_path):
    data = TimeHistory(np.arange(8), np.ones((2, 8)),
                       response_dof=['1Z+', '2Z+'], ordinate_dim='velocity')
    path = tmp_path / 'ordinary.unv'
    unv.save(data, path)
    restored = unv.load(path)
    assert isinstance(restored, TimeHistory)
    assert restored.num_records == 2
    project = Project()
    assert project.import_file(path) == ['Time History']


def test_the_files_reference_is_shown_and_can_be_replaced(window, tmp_path,
                                                         monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    path = trace_file(tmp_path)
    assert unv.trace_pair_references(path) == ['1Z+']
    asked = []

    def answer(_parent, _title, label, value, *_limits):
        asked.append((label, value))
        return 1013, True

    monkeypatch.setattr(QInputDialog, 'getInt', answer)
    [names] = window.import_paths([str(path)])
    assert '1Z+' in asked[0][0] and asked[0][1] == 1
    assert names == ['Time History - 730Z- - ref 1013Z+',
                     'Time History - 1125Z- - ref 1013Z+']


def test_a_file_without_labeled_pairs_asks_nothing(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    data = TimeHistory(np.arange(8), np.ones((2, 8)),
                       response_dof=['1Z+', '2Z+'], ordinate_dim='velocity')
    path = tmp_path / 'ordinary.unv'
    unv.save(data, path)
    assert unv.trace_pair_references(path) == []
    monkeypatch.setattr(QInputDialog, 'getInt',
                        lambda *_a: pytest.fail('asked about a file with no pairs'))
    assert window.import_paths([str(path)]) == ['Time History']


def test_canceling_the_reference_question_skips_the_file(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, 'getInt', lambda *_a: (1, False))
    assert window.import_paths([str(trace_file(tmp_path))]) == []
    assert not list(window.project)
