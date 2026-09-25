"""Project types: gray slots for what a report of that kind expects.

Setting a project type lists every object the report needs; missing
ones show as gray placeholders that disappear as the real objects
arrive, offer to compute themselves where the data allows, and ride
the project file so a reopened project remembers what it is.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics import io
from visualdynamics.core.data import Psd, TimeHistory
from visualdynamics.core.report import PROJECT_TYPES, Report, project_expectations

ROLE_REFERENCE = None  # resolved from the module in a fixture


def _placeholders(window):
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    out = []
    for i in range(window.test_item.childCount()):
        child = window.test_item.child(i)
        reference = child.data(0, ROLE_REFERENCE)
        if reference is not None and reference[0] == 'placeholder':
            out.append(child.text(0))
    return out


def _burst_time():
    t = np.arange(256) / 256.0
    rows = [np.sin(2 * np.pi * 16 * t)] * 2
    time = TimeHistory(t, np.array(rows), response_dof=['1X+', '1X+'],
                       block=['avg 1', 'avg 2'])
    time.define_units('m/s^2')
    return time


def test_expectations_name_what_the_templates_bind():
    expectations = project_expectations('Modal Test')
    names = [name for name, *_rest in expectations]
    # Matched Modes comes after both groups, not between them: it
    # belongs to neither side, and a group's rows have to stay
    # contiguous to be drawn as one bracket
    # every slot named for its type and nothing more (Brandon,
    # 2026-09-02): the comparison pair reads as a second Geometry and
    # a second Shape Set, on the side that has no name
    assert names == ['Geometry', 'Photos', 'Channel Table',
                     'Time History', 'PSD', 'FRF', 'Multiple Coherence',
                     'Shape Set', 'Geometry', 'Shape Set',
                     'Matched Modes', 'Report']
    from visualdynamics.core.report import OTHER_SIDE

    sides = [side for *_rest, side in expectations]
    assert sides == ['Basis'] * 8 + [OTHER_SIDE] * 2 + [None] * 2, (
        'eight Basis slots, the comparison pair on the other side, and '
        'two slots on neither')
    optional = [name for name, _cls, _icon, _count, opt, _group
                in expectations if opt]
    assert optional == ['Geometry', 'Shape Set', 'Matched Modes'], (
        'correlation is best practice, not required')
    groups = {name: group for name, _cls, _icon, _count, _opt, group
              in expectations if name not in ('Geometry', 'Shape Set')}
    assert groups['FRF'] == 'Basis'
    assert groups['Photos'] == 'Basis', (
        'setup photos document the test — they belong with its group')
    assert groups['Report'] is None, (
        'a report reads the whole project and belongs to no one side')
    assert groups['Matched Modes'] is None, (
        'and a comparison names a set on each side, so it is on neither')
    assert project_expectations(None) == []


@pytest.mark.parametrize('project_type', PROJECT_TYPES)
def test_every_type_expects_the_report_itself_last(project_type):
    """The report is the thing the rest of it is for.

    A project holding everything a report draws from and no report is
    not a finished project, so the report is a slot like the others —
    gray until it is generated, and last, because it is what the others
    add up to.
    """
    expectations = project_expectations(project_type)
    name, cls, icon, count, optional, group = expectations[-1]
    assert (name, icon, count, optional, group) == (
        'Report', 'Report', 1, False, None)
    assert cls is Report


@pytest.mark.parametrize('project_type', PROJECT_TYPES)
def test_a_type_has_a_slot_for_everything_its_report_binds(project_type):
    """Nothing the report reaches for is missing from the tree.

    The slots exist so that a project shows what its report will need
    before the report is generated — which only holds while the two
    lists agree. They drifted once already: the shock template binds
    '@basis:ShockSpecification' and '@basis:Srs', and neither token was
    a binding type at all, so both blocks resolved to nothing and the
    shock report rendered its specification and its spectra as empty
    slots however complete the project was.
    """
    from visualdynamics.core.report import (
        PROJECT_TEMPLATES,
        TEMPLATE_BUILDERS,
        binding_tokens,
    )

    report = TEMPLATE_BUILDERS[PROJECT_TEMPLATES[project_type]](
        visualdynamics.Project(), links=[])
    types = binding_tokens()
    wanted = {token for block in report.blocks
              for value in block.values()
              for token in re.findall(r'@\w+:(\w+)', str(value))}
    assert wanted, 'the template binds symbolically or this proves nothing'
    unknown = sorted(token for token in wanted if token not in types)
    assert not unknown, (
        f'{project_type} binds tokens that are not binding types at '
        f'all, so they resolve to nothing: {unknown}')
    slots = [cls for _name, cls, *_rest
             in project_expectations(project_type)]
    for token in sorted(wanted):
        cls, _banded = types[token]
        assert any(issubclass(slot, cls) or issubclass(cls, slot)
                   for slot in slots), (
            f'the {project_type} report binds {token} and the project '
            'tree never asks for one')


def test_a_typed_project_links_its_sides_as_they_arrive(window, pump,
                                                        survey):
    """The type's structure includes the links: the Basis objects
    link together as they arrive. The other side is never guessed —
    a second geometry and shape set stay loose until linked by hand
    or by their file (2026-09-02) — and an explicit Unlink is
    respected until something new arrives."""
    shapes, frfs = survey
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    window.set_project_type('Modal Test')
    window.add_object('Geometry', geometry)
    window.add_object('FRF', frfs)
    assert window.linked_group('FRF') == ['Geometry', 'FRF'], (
        'the experimental side links itself as it fills in')
    window.add_object('Shape Set', shapes)
    assert set(window.linked_group('Shape Set')) == {
        'Geometry', 'FRF', 'Shape Set'}
    # a second geometry and shape set are the FEM side — their own link
    fem_geometry = visualdynamics.import_file(fixture_path('plate',
                                                 'geometry.npz'))
    fem_shapes = visualdynamics.import_file(fixture_path('plate',
                                               'shapes.npy'))
    window.add_object('FEM Geometry', fem_geometry)
    window.add_object('FEM Shapes', fem_shapes)
    assert window.linked_group('FEM Shapes') is None, 'not guessed'
    assert window.linked_group('FEM Geometry') is None
    assert 'FEM Shapes' not in window.linked_group('FRF')
    # the type names one group: the measured one is the Basis
    assert window.link_role('FRF') == 'Basis'
    assert window.link_role('FEM Shapes') is None
    # unlink survives until a new satisfier shows up
    window.tree.clearSelection()
    window._item_for_object('FRF').setSelected(True)
    window.unlink_selected()
    assert window.linked_group('FRF') is None


def test_placeholders_track_the_missing_objects(window, pump, survey):
    shapes, frfs = survey
    window.set_project_type('Modal Test')
    # sides stay adjacent for their brackets, in expectation order
    assert _placeholders(window) == ['Geometry', 'Photos',
                                     'Channel Table', 'Time History',
                                     'PSD', 'FRF', 'Multiple Coherence',
                                     'Shape Set', 'Geometry',
                                     'Shape Set', 'Matched Modes',
                                     'Report']
    window.add_object('FRF', frfs)
    assert 'FRF' not in _placeholders(window), 'the real object filled it'
    window.add_object('Shape Set', shapes)
    assert _placeholders(window).count('Shape Set') == 1, (
        'the first set fills the Basis slot; the other side still '
        'wants a second set, not the first')
    window.tree.clearSelection()
    window._item_for_object('FRF').setSelected(True)
    window.delete_selected()
    assert 'FRF' in _placeholders(window), 'deleting reopens the slot'
    window.set_project_type(None)
    assert _placeholders(window) == [], 'no type, no slots'


def test_a_fresh_modal_project_shows_both_group_brackets(window, pump,
                                                         survey):
    """The default modal project includes the linked groups: the
    Basis-to-be and the model side stand bracketed over the
    placeholder slots before anything exists, and real arrivals join
    their side."""
    def texts(items):
        return [item.text(0) for item in items]

    window.set_project_type('Modal Test')
    pump()
    spans = {bold: items for _color, items, bold
             in window.tree.link_spans}
    assert set(spans) == {True, False}, 'the Basis bracket is the bold one'
    assert texts(spans[True]) == [
        'Geometry', 'Photos', 'Channel Table', 'Time History', 'PSD',
        'FRF', 'Multiple Coherence', 'Shape Set']
    assert texts(spans[False]) == ['Geometry', 'Shape Set']
    # real members lead their bracket; the open slots stay inside it
    _shapes, frfs = survey
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    window.add_object('Geometry', geometry)
    window.add_object('FRF', frfs)
    pump()
    spans = {bold: items for _color, items, bold
             in window.tree.link_spans}
    names = texts(spans[True])
    assert names[:2] == ['Geometry', 'FRF']
    assert 'Time History' in names and 'Photos' in names
    assert texts(spans[False]) == ['Geometry', 'Shape Set'], (
        'the other side keeps its bracket while still empty')


def test_a_psd_placeholder_computes_itself_from_time_data(window, pump):
    window.add_object('Time Data', _burst_time())
    window.set_project_type('Modal Test')
    assert 'PSD' in _placeholders(window)
    window._compute_from_time('Time Data', 'psds')
    assert any(isinstance(obj, Psd) for obj in window.objects.values())
    assert 'PSD' not in _placeholders(window)


def test_a_typed_project_generates_its_own_report(window, pump):
    window.set_project_type('Modal Test')
    window.generate_typed_report()
    report = next(obj for obj in window.objects.values()
                  if type(obj).__name__ == 'Report')
    assert report.title == 'Modal Test Report'


def test_the_report_slot_fills_itself(window, pump):
    """The gray Report slot is the one click that makes the report.

    The type already says which report, so there is nothing to choose —
    the slot names what is missing and offers to be it.
    """
    window.set_project_type('Random Vibration')
    assert 'Report' in _placeholders(window)
    window.generate_typed_report()
    report = next(obj for obj in window.objects.values()
                  if isinstance(obj, Report))
    assert report.title == 'Random Vibration Test Report'
    assert 'Report' not in _placeholders(window), (
        'the real report filled its slot')


def test_the_project_type_rides_the_project_file(tmp_path, window,
                                                 window_factory, pump):
    window.add_object('Time Data', _burst_time())
    window.set_project_type('Modal Test')
    path = tmp_path / 'typed.vdyn'
    io.save_test(str(path), 'Typed', dict(window.objects),
                 project_type=window.project_type)
    contents = io.load(str(path))
    assert contents.project_type == 'Modal Test'
    other = window_factory()
    other.import_paths([str(path)])
    assert other.project_type == 'Modal Test'
    assert 'PSD' in _placeholders(other), (
        'the reopened project shows its remaining slots')


def test_placeholders_never_pollute_the_selection(window, pump):
    window.set_project_type('Modal Test')
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    window.add_object('Geometry', geometry)
    for i in range(window.test_item.childCount()):
        window.test_item.child(i).setSelected(True)
    references = window.selected_references()
    assert all(kind != 'placeholder' for kind, *_rest in references)
    window.render_current()
    pump()     # nothing to assert beyond not crashing on gray slots


def test_a_random_report_is_not_finished_without_the_octave_bands():
    """The sixth-octave PSD is a deliverable, not an optional reading.

    It is the *second* `Psd` in the experimental group rather than a
    class of its own: `Psd.to_octave` returns a `Psd` because banding
    conserves the area and changes nothing about what the object is.
    """
    from visualdynamics.core.data import Psd
    from visualdynamics.core.report import project_expectations

    slots = project_expectations('Random Vibration')
    octave = next(s for s in slots if s[2] == 'OctavePsd')
    name, cls, icon, count, optional, tag = octave
    assert name == 'PSD', 'named for its type, like every slot'
    assert cls is Psd, 'banding does not make a different type'
    assert count == 2, 'the second PSD in the measured group'
    assert not optional, 'a random report is incomplete without it'
    assert icon == 'OctavePsd', 'and it does not wear the narrowband glyph'
    assert tag == 'Basis'

    icons = [s[2] for s in slots]
    assert icons.index('Psd') < icons.index('OctavePsd'), (
        'it is derived from the narrowband PSD and reads after it')


def test_two_psds_fill_both_psd_slots(window, pump):
    """The count is what makes the second slot want a second object, so
    banding one PSD has to fill it — and one PSD alone must not."""
    import numpy as np

    from visualdynamics.core.data import Psd

    window.set_project_type('Random Vibration')
    pump()
    assert _placeholders(window).count('PSD') == 2, 'narrowband and banded'

    f = np.linspace(10.0, 2000.0, 400)
    psd = Psd(f, np.ones((1, 400)), response_dof=['1Z+'],
              ordinate_dim='acceleration**2/frequency',
              ordinate_unit='m/s**2')
    window.add_object('PSD', psd)
    pump()
    assert _placeholders(window).count('PSD') == 1, (
        'one PSD fills one slot, not both')

    window.add_object('Octave Band PSD', psd.to_octave(6))
    pump()
    assert 'PSD' not in _placeholders(window)


def test_the_specification_fills_no_psd_slot(window, pump):
    """A Specification *is a* Psd, and raw isinstance let it stand in
    for the measured spectra: a random project's spec filled the 'PSD'
    slot on arrival, and the spec plus the computed PSDs filled
    'Octave Band PSD' — the standing deliverable vanished from the
    skeleton with no octave object anywhere in the project. Within a
    type, an object that satisfies a more specific expected class
    belongs to that class's slots and no other's."""
    import numpy as np

    from visualdynamics.core.data import Specification

    window.set_project_type('Random Vibration')
    pump()
    f = np.linspace(10.0, 2000.0, 400)
    window.add_object('Specification', Specification(
        f, np.ones((1, 400)), response_dof=['1Z+'],
        ordinate_dim='acceleration**2/frequency',
        ordinate_unit='m/s**2'))
    pump()
    assert _placeholders(window).count('PSD') == 2, (
        'the spec is not the measured spectra')

    window.add_object('PSD', Psd(
        f, np.ones((1, 400)), response_dof=['1Z+'],
        ordinate_dim='acceleration**2/frequency',
        ordinate_unit='m/s**2'))
    pump()
    assert _placeholders(window).count('PSD') == 1, (
        'the spec must not count as the second Psd either')

    window.add_object('Octave Band PSD',
                      window.objects['PSD'].to_octave(6))
    pump()
    assert 'PSD' not in _placeholders(window)


def test_a_sine_sweep_project_shows_its_slots(window, pump):
    """The new type's tree: the sine specification where the shock
    type keeps its SRS, no coherence (a swept tone is not a
    stationary average), and the report last as everywhere."""
    window.set_project_type('Sine Sweep')
    assert _placeholders(window) == ['Geometry', 'Photos',
                                     'Channel Table', 'Time History',
                                     'Sine Sweep Specification',
                                     'Sine Level Set', 'Report']


def test_a_random_report_wants_the_octave_band_specification_too():
    """The requirement on octave bands beside the PSD on them: the
    second Specification in the group, as the octave PSD is the second
    Psd (Brandon, 2026-09-18)."""
    from visualdynamics.core.data import Specification
    from visualdynamics.core.report import project_expectations

    slots = project_expectations('Random Vibration')
    octave = next(s for s in slots if s[2] == 'OctaveSpecification')
    name, cls, _icon, count, optional, tag = octave
    assert name == 'Specification'
    assert cls is Specification and count == 2 and not optional
    assert tag == 'Basis'
    icons = [s[2] for s in slots]
    assert icons.index('Specification') < icons.index('OctaveSpecification')


def test_a_banded_specification_fills_the_octave_slot_and_not_the_plain(window, pump):
    import numpy as np

    from visualdynamics.core.data import Specification

    window.set_project_type('Random Vibration')
    pump()
    assert _placeholders(window).count('Specification') == 2
    f = np.linspace(10.0, 2000.0, 400)
    level = np.ones((1, 400)) * 1e-3
    spec = Specification(f, level, response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency',
                         ordinate_unit='m/s**2',
                         abort_upper=level * 4, abort_lower=level / 4)
    window.add_object('Specification', spec)
    pump()
    assert _placeholders(window).count('Specification') == 1
    assert _placeholders(window).count('PSD') == 2, 'a specification is no PSD'
    window.add_object('Octave Band Specification', spec.to_octave(6))
    pump()
    assert 'Specification' not in _placeholders(window)


def test_a_second_narrowband_specification_does_not_fill_the_octave_slot(window, pump):
    """The octave slot wants the requirement on bands: a second plain
    specification — the same run imported again — was filling it
    (2026-09-18), and then the whole second run followed it into the
    Basis."""
    import numpy as np

    from visualdynamics.core.data import Specification

    window.set_project_type('Random Vibration')
    pump()
    f = np.linspace(10.0, 2000.0, 400)
    level = np.ones((1, 400)) * 1e-3
    spec = Specification(f, level, response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency',
                         ordinate_unit='m/s**2')
    window.add_object('Specification', spec)
    window.add_object('Specification (2)', spec)
    pump()
    assert _placeholders(window).count('Specification') == 1, \
        'the second plain one fills nothing'
    window.add_object('Octave Band Specification', spec.to_octave(6))
    pump()
    assert 'Specification' not in _placeholders(window)
