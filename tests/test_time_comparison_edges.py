"""Boundaries of the scoring window and independent pair calculations."""

import numpy as np
import pytest

from visualdynamics import Project, TimeHistory
from visualdynamics.core.time_compare import scores


def _comparison():
    project = Project()
    a = TimeHistory([0., 1., 2.], [[1., 2., 0.], [0., 3., 2.]],
                    response_dof=['10Z-', '20Z-'], ordinate_dim='velocity')
    b = TimeHistory([0., 1., 2.], [[0., 3., 1.], [2., -1., 1.]],
                    response_dof=['10Z-', '20Z-'], ordinate_dim='velocity')
    project.add('A', a)
    project.add('B', b)
    return project[project.compare_time_histories('A', 'B')]


def test_each_pair_uses_its_own_rows_and_amplitude():
    first, second = scores(_comparison())
    assert first['trac'] == pytest.approx(36/(5*10))
    assert first['rms_ratio'] == pytest.approx(np.sqrt(10/5))
    assert second['dof'] == '20Z-'
    assert second['correlation'] == pytest.approx(-1/np.sqrt(13*6))
    assert second['rms_ratio'] == pytest.approx(np.sqrt(6/13))


def test_two_samples_at_the_window_edges_are_enough():
    row = scores(_comparison(), 0., 1.)[0]
    assert row['samples'] == 2
    assert row['trac'] == pytest.approx(4/5)
    with pytest.raises(ValueError, match='fewer than two samples'):
        scores(_comparison(), 0.1, 1.1)


@pytest.mark.parametrize(('start', 'stop', 'message'), [
    (float('nan'), 1., 'finite and greater'),
    (1., 1., 'finite and greater'),
    (-1., 1., 'common time interval'),
    (1., 3., 'common time interval'),
])
def test_invalid_window_limits_report_their_cause(start, stop, message):
    with pytest.raises(ValueError, match=message):
        scores(_comparison(), start, stop)


def test_a_constant_first_response_has_no_score():
    data = _comparison()
    data.ordinate[0] = 1.
    row = scores(data)[0]
    assert row['trac'] is None
    assert 'Constant' in row['reason']


def test_explicit_pairs_keep_each_left_label_when_the_right_labels_differ():
    project = Project()
    values = [[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]]
    left = ['10Z-', '20Z-', '30Z-']
    right = ['11Z-', '21Z-', '31Z-']
    for name, dofs in [('A', left), ('B', right)]:
        project.add(name, TimeHistory([0., 1., 2.], values,
                                      response_dof=dofs, ordinate_dim='velocity'))
    name = project.compare_time_histories('A', 'B', pairs=[('A', i, 'B', i) for i in range(3)])
    rows = scores(project[name])
    assert [row['dof'] for row in rows] == left
    assert [row['trac'] for row in rows] == pytest.approx([1., 1., 1.])
