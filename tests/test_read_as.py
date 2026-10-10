"""Reading a PSD as a specification, and back.

A format with no word for a requirement — ESCDF's data types have none
— can only say a PSD is a target by its name, and a name is no
evidence. So the person who knows says so: an act on the bar and a
project verb, in place, the name and everything keyed on it kept
(Brandon, 2026-10-09).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Psd, Specification


def _bar(pane):
    actions = pane.__dict__.get('_acts', {}).get('actions', {}).values()
    return [a.text() for a in pane.toolbar.actions()
            if a in actions and a.isVisible()]


@pytest.fixture
def run():
    """A random run: a measurement, and its target imported as the
    specification it is — read back as a plain PSD, the way a file
    from a format with no specification type arrives."""
    project = visualdynamics.random_vibration_run(fixture_path('plate',
                                                               'random.nc4'))
    target = project['Specification']
    project['Target'] = Psd(target.abscissa, target.ordinate,
                            response_dof=target.response_dof,
                            reference_dof=target.reference_dof,
                            ordinate_dim=target.ordinate_dim,
                            ordinate_unit=target.ordinate_unit)
    return project


def test_a_psd_reads_as_a_specification_and_back_in_place(run):
    run.link('Target', 'Time History PSDs')
    group = run.object_group_of('Target')
    assert run.read_as_specification('Target') == 'Target'
    spec = run['Target']
    assert isinstance(spec, Specification) and not spec.has_limits
    assert np.array_equal(spec.ordinate, run['Specification'].ordinate)
    assert spec.ordinate_unit == run['Specification'].ordinate_unit
    assert spec.interpolation == 'bin', "a controller's lines step"
    assert run.object_group_of('Target') == group, 'its group kept'
    assert run.journal[-1] == "project.read_as_specification('Target')"
    run.read_as_psd('Target')
    assert type(run['Target']) is Psd
    assert run.journal[-1] == "project.read_as_psd('Target')"


def test_it_is_offered_only_where_it_applies(run):
    verbs = dict(run.verbs('Target'))
    assert 'read_as_specification' in verbs and 'read_as_psd' not in verbs
    assert 'read_as_psd' not in dict(run.verbs('Specification')), \
        'not with limits: a PSD has nowhere to keep them'
    assert 'read_as_specification' not in dict(run.verbs('Time History'))
    with pytest.raises(ValueError, match='limits'):
        run['Specification'].as_psd()


def test_a_banded_psd_reads_as_a_stepped_specification(run):
    banded = run['Target'].to_octave(6).as_specification()
    assert banded.interpolation == 'bin' and banded.bandwidth is not None


def test_the_bar_offers_it_and_the_comparison_follows(window, pump, run):
    """In the window: the act on the bar, the swap in place, and the
    pair then read as a comparison — the reason to do it."""
    window.add_object('Target', run['Target'])
    window.add_object('PSDs', run['Time History PSDs'])
    item = window._item_for_object('Target')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert 'Read as Specification' in _bar(window.data_pane)
    window.data_pane._acts['actions']['read_as_specification'].trigger()
    pump()
    assert isinstance(window.objects['Target'], Specification)
    assert 'read as a specification' in window.statusBar().currentMessage()
    assert 'Read as PSD' in _bar(window.data_pane)
    window.tree.clearSelection()
    for name in ('Target', 'PSDs'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.data_pane.scaling_edit.isVisible(), \
        'a specification beside a measurement is a comparison'
