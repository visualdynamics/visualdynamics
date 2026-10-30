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
from conftest import bar_acts, fixture_path

import visualdynamics
from visualdynamics.core.data import Psd, Specification


@pytest.fixture(scope='module')
def imported():
    """The random run, read once: its import computes PSDs, octave
    bands and coherence, over a second each time."""
    return visualdynamics.random_vibration_run(fixture_path('plate',
                                                            'random.nc4'))


@pytest.fixture
def run(imported):
    """A measurement and its target imported as the specification it
    is — and again as a plain PSD, the way a file from a format with no
    specification type delivers it — in a project of its own per test,
    since the tests swap objects in it."""
    project = visualdynamics.Project('Read as')
    for name in ('Time History', 'Specification', 'Time History PSDs'):
        project.add(name, imported[name])
    target = imported['Specification']
    project.add('Target', Psd(target.abscissa, target.ordinate,
                              response_dof=target.response_dof,
                              reference_dof=target.reference_dof,
                              ordinate_dim=target.ordinate_dim,
                              ordinate_unit=target.ordinate_unit))
    project.journal.clear()
    return project


def test_a_psd_reads_as_a_specification_and_back_in_place(run):
    run.link('Target', 'Time History PSDs')
    group = run.object_group_of('Target')
    assert run.convert_to_specification('Target') == 'Target'
    spec = run['Target']
    assert isinstance(spec, Specification) and not spec.has_limits
    assert np.array_equal(spec.ordinate, run['Specification'].ordinate)
    assert spec.ordinate_unit == run['Specification'].ordinate_unit
    assert spec.interpolation == 'bin', "a controller's lines step"
    assert run.object_group_of('Target') == group, 'its group kept'
    assert run.journal[-1] == "project.convert_to_specification('Target')"
    run.convert_to_psd('Target')
    assert type(run['Target']) is Psd
    assert run.journal[-1] == "project.convert_to_psd('Target')"


def test_it_is_offered_only_where_it_applies(run):
    verbs = dict(run.verbs('Target'))
    assert 'convert_to_specification' in verbs and 'convert_to_psd' not in verbs
    assert 'convert_to_specification' not in dict(run.verbs('Time History'))
    with pytest.raises(ValueError, match='limits'):
        run['Specification'].as_psd()


def test_a_specification_with_limits_converts_without_them(run):
    # the Specification sheet always writes its bands, so a target made
    # there could never be converted while limits refused it
    assert run['Specification'].has_limits
    assert 'convert_to_psd' in dict(run.verbs('Specification'))
    ordinate = run['Specification'].ordinate.copy()
    run.convert_to_psd('Specification')
    plain = run['Specification']
    assert type(plain) is Psd and 'limits' not in plain.__dict__
    assert np.array_equal(plain.ordinate, ordinate)


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
    assert 'Convert to Specification' in bar_acts(window.data_pane)
    window.data_pane._acts['actions']['convert_to_specification'].trigger()
    pump()
    assert isinstance(window.objects['Target'], Specification)
    assert 'converted to a specification' in window.statusBar().currentMessage()
    assert 'Convert to PSD' in bar_acts(window.data_pane)
    window.tree.clearSelection()
    for name in ('Target', 'PSDs'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.data_pane.scaling_edit.isVisible(), \
        'a specification beside a measurement is a comparison'


def test_the_label_follows_the_object_and_the_limits_are_said(window, pump, run):
    """Convert to CPSD on a specification with cross terms, Convert to
    PSD on autospectra alone, and the status line says the limits went."""
    import copy

    autos = run['Specification']
    assert autos.is_autospectrum() and autos.has_limits
    spec = copy.deepcopy(autos)
    spec.reference_dof[1] = spec.response_dof[0]     # one cross term
    assert not spec.is_autospectrum()
    window.add_object('Target', spec)
    window.add_object('Autos', autos)

    def show(name):
        window.tree.clearSelection()
        item = window._item_for_object(name)
        item.setSelected(True)
        window.tree.setCurrentItem(item)
        window.render_current()
        pump()
        return bar_acts(window.data_pane)

    acts = show('Target')
    assert 'Convert to CPSD' in acts and 'Convert to PSD' not in acts
    acts = show('Autos')
    assert 'Convert to PSD' in acts and 'Convert to CPSD' not in acts
    show('Target')
    window.data_pane._acts['actions']['convert_to_psd'].trigger()
    pump()
    assert type(window.objects['Target']) is Psd
    message = window.statusBar().currentMessage()
    assert 'converted to a CPSD' in message
    assert 'limits were left out' in message
