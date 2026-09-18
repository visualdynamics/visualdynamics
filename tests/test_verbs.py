"""What can be done with an object is the project's own answer.

`Project.verbs` is the one applicability table — the API lists what a
script can call on an object, and the window's calculator menu is a
presentation of the same table's answers (Brandon, 2026-08-31: a flat
method list on the project says nothing about what `integrate` is
*for*). The summaries are the first paragraph of each verb's own
docstring, so the table and the API reference cannot drift apart.
"""

from __future__ import annotations

import numpy as np

import visualdynamics
from visualdynamics.core.data import Psd, Specification
from visualdynamics.project import _JOURNALED_VERBS, _VERB_APPLIES, Project


def _accel_history(channels=('101Z+', '104Z+')):
    t = np.arange(4096) / 2048.0
    return visualdynamics.TimeHistory(
        t, np.random.default_rng(5).standard_normal((len(channels), len(t))),
        response_dof=list(channels), ordinate_dim='acceleration')


def _psd(cls=Psd, **extra):
    freq = np.linspace(10.0, 2000.0, 200)
    level = np.full((1, len(freq)), 1e-2)
    return cls(abscissa=freq, ordinate=level, response_dof=['101Z+'],
               ordinate_dim=['acceleration**2/frequency'], **extra)


def test_verbs_answer_what_a_time_history_offers():
    project = visualdynamics.Project('t')
    project.add('Run', _accel_history())
    verbs = dict(project.verbs('Run'))
    assert 'integrate' in verbs, 'acceleration integrates to velocity'
    assert 'compute_psds' in verbs
    assert 'fit_modes' not in verbs, 'that one wants an FRF'
    assert 'compute_octave' not in verbs, 'and that one a density'
    # the summary is the verb's own first paragraph, one line of prose
    assert 'time history' in verbs['integrate']
    assert '\n' not in verbs['integrate']


def test_a_record_with_nothing_to_integrate_is_not_offered_it():
    t = np.arange(1024) / 1024.0
    forces = visualdynamics.TimeHistory(
        t, np.ones((1, len(t))), response_dof=['101Z+'],
        ordinate_dim='force')
    project = visualdynamics.Project('t')
    project.add('Forces', forces)
    verbs = dict(project.verbs('Forces'))
    assert 'integrate' not in verbs
    assert 'differentiate' not in verbs
    assert 'compute_psds' in verbs, 'a force record still has spectra'


def test_frfs_need_drive_channels():
    project = visualdynamics.Project('t')
    project.add('Run', _accel_history())
    assert 'compute_frfs' not in dict(project.verbs('Run')), (
        'no excitation quantity anywhere, so nothing can be a reference')
    t = np.arange(1024) / 1024.0
    driven = visualdynamics.TimeHistory(
        t, np.ones((2, len(t))), response_dof=['101Z+', '104Z+'],
        ordinate_dim=['force', 'acceleration'])
    project.add('Driven', driven)
    offered = dict(project.verbs('Driven'))
    assert 'compute_frfs' in offered
    assert 'compute_multiple_coherence' in offered


def test_extract_sine_appears_with_the_specification():
    from visualdynamics.core.sine import SineSweepSpecification, SineTone

    project = visualdynamics.Project('t')
    project.add('Run', _accel_history())
    assert 'extract_sine' not in dict(project.verbs('Run'))
    tone = SineTone('Up', 1.0, [100.0, 800.0],
                    [[2.0, 3.0], [2.0, 3.0]], [0], [100.0])
    project.add('Sweep', SineSweepSpecification(
        [tone], ['101Z+', '104Z+'], ordinate_unit='m/s**2'))
    assert 'extract_sine' in dict(project.verbs('Run'))


def test_a_specification_is_banded_too():
    """It was excluded until 2026-09-18: Brandon wants the target and
    the measurement it judges convertible alike, limits and all."""
    project = visualdynamics.Project('t')
    project.add('PSD', _psd())
    level = np.full((1, 200), 1e-2)
    project.add('Spec', _psd(Specification,
                             warning_lower=level * 0.5,
                             warning_upper=level * 2.0,
                             abort_lower=level * 0.25,
                             abort_upper=level * 4.0))
    assert 'compute_octave' in dict(project.verbs('PSD'))
    assert 'compute_octave' in dict(project.verbs('Spec'))


def test_merge_needs_a_peer_of_the_same_kind():
    project = visualdynamics.Project('t')
    project.add('Run', _accel_history())
    assert 'merge' not in dict(project.verbs('Run'))
    project.add('Run 2', _accel_history(('110Z+', '111Z+')))
    assert 'merge' in dict(project.verbs('Run'))


def test_every_table_entry_is_a_real_journaled_verb():
    """A verb the table lists must exist on the project and journal
    when called — an entry pointing at nothing would teach a script
    writer a command that does not run."""
    for verb, _applies in _VERB_APPLIES:
        assert callable(getattr(Project, verb)), verb
        assert verb in _JOURNALED_VERBS, verb
    # and asked without an object, the table reads out whole
    listed = [verb for verb, _summary in visualdynamics.Project('t').verbs()]
    assert listed == [verb for verb, _applies in _VERB_APPLIES]


def test_the_bar_is_the_tables_presentation(window, pump):
    """The bar offers a verb exactly when the table says it applies —
    one implementation of the rule, worn by both surfaces."""
    window.add_object('Run', _accel_history())
    labels = [label for _v, label, *_rest in window.acts_for(['Run'])]
    assert 'Integrate' in labels
    applicable = {verb for verb, _s in window.project.selection_verbs('Run')}
    for verb, label, _icon, _handler in window.ACTS:
        assert (label in labels) == (verb in applicable), verb
