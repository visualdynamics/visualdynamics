"""Truncating a record to the stretch that matters.

The core rules, pinned: the cut keeps the samples' measured instants
rather than re-zeroing the clock; both ends are inclusive; no marks
ride across, because they can refer to samples that no longer exist;
and the derived record's staleness follows the span the way a filtered
record's follows the filter.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.truncate import Truncation, truncate

RATE = 2048.0
SECONDS = 4.0


def _history():
    t = np.arange(int(RATE * SECONDS)) / RATE
    rng = np.random.default_rng(11)
    return TimeHistory(t, rng.standard_normal((2, len(t))),
                       response_dof=['101Z+', '9001X+'],
                       ordinate_dim=['acceleration', 'force'])


def _project():
    project = visualdynamics.Project('Cut')
    project.add('Run', _history())
    return project


def test_the_cut_keeps_the_span_and_the_clock():
    data = _history()
    out = truncate(data, Truncation(0.5, 2.0))
    assert out.abscissa[0] == pytest.approx(0.5), \
        'the first kept sample keeps its measured instant'
    assert out.abscissa[-1] == pytest.approx(2.0)
    kept = slice(int(0.5 * RATE), int(2.0 * RATE) + 1)
    assert np.array_equal(out.ordinate, data.ordinate[:, kept]), \
        'every channel cut at the same two instants, values untouched'
    assert list(out.response_dof) == list(data.response_dof)
    assert list(out.ordinate_dim) == list(data.ordinate_dim)


def test_both_ends_are_inclusive():
    data = _history()
    out = truncate(data, Truncation(1.0, 1.0 + 10 / RATE))
    assert out.ordinate.shape[-1] == 11, \
        'a sample sitting exactly on either edge is kept'


def test_a_span_holding_less_than_two_samples_is_refused():
    data = _history()
    with pytest.raises(ValueError, match='fewer than two samples'):
        truncate(data, Truncation(1.0 + 0.1 / RATE, 1.0 + 0.9 / RATE))
    with pytest.raises(ValueError, match='fewer than two samples'):
        truncate(data, Truncation(100.0, 200.0))    # off the record
    with pytest.raises(ValueError, match='fewer than two samples'):
        truncate(data, Truncation(1.0, 1.0 + 0.5 / RATE))   # one sample
    # and two is enough — the boundary itself (mutation testing,
    # 2026-10-10: `< 2` and `<= 2`, or `< 3`, passed before)
    assert truncate(data, Truncation(1.0, 1.0 + 1.0 / RATE)).ordinate.shape[-1] == 2


def test_the_settings_validate_at_entry():
    with pytest.raises(ValueError, match='runs forward'):
        Truncation(2.0, 0.5)
    with pytest.raises(ValueError, match='runs forward'):
        Truncation(1.0, 1.0)
    with pytest.raises(ValueError, match='finite'):
        Truncation(0.0, float('inf'))


def test_no_marks_are_carried():
    """`core.filters` carries marks because the abscissa is unchanged
    and they still say what they said. Here the abscissa changed —
    that is the point — so the honest answer is a clean record."""
    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.filters import Filtering
    from visualdynamics.core.shocks import Shock

    data = _history()
    data.averaging = Averaging(frame_length=1024, frames=4)
    data.shocks = (Shock(1.0, 0.5),)
    data.filtering = Filtering(high=200.0)
    out = truncate(data, Truncation(0.5, 2.0))
    assert out.averaging is None
    assert not out.shocks
    assert out.filtering is None
    assert out.truncation is None, \
        'the result of a cut has been cut; offering the same cut '\
        'again would be a button that does nothing'


def test_describe_is_the_one_wording():
    assert Truncation(0.5, 2.0).describe() == 'keeping 0.5 to 2 s'


# ---- through the project --------------------------------------------------


def test_truncate_data_refuses_without_a_span():
    """Unlike Filter Data there is no suggestion to adopt: the whole
    record is the only neutral span and keeping all of it is not an
    act."""
    project = _project()
    with pytest.raises(ValueError, match='no span set'):
        project.truncate_data('Run')


def test_the_derived_record_goes_stale_when_the_span_moves():
    project = _project()
    project['Run'].truncation = Truncation(0.5, 2.0)
    name = project.truncate_data('Run')
    assert project.stale() == {}
    project['Run'].truncation = Truncation(0.25, 2.0)
    assert name in project.stale()
    assert 'keeping 0.25 to 2 s' in project.stale()[name], \
        'the story says the span in the same words the status line does'


def test_the_settings_survive_a_save_and_a_load(tmp_path):
    project = _project()
    project['Run'].truncation = Truncation(0.5, 2.0)
    name = project.truncate_data('Run')
    project['Run'].truncation = Truncation(0.25, 2.0)
    back = visualdynamics.Project.open(
        project.save(tmp_path / 'cut.vdyn'))
    assert back['Run'].truncation == Truncation(0.25, 2.0)
    assert name in back.stale(), \
        'the fingerprint survived the file, so the badge still works'
    back.refresh(name)
    assert back[name].abscissa[0] == pytest.approx(0.25), \
        'refresh recomputes with the recorded verb and the new span'
    assert back.stale() == {}
