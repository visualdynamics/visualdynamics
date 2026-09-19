"""Writing and editing a specification, at any coordinates.

One sheet, every door: new at a shape set's modal coordinates, new at
a channel table's control channels, or opened from a specification
that already exists. The draft holds what an author states and
nothing else — breakpoints and a level per channel, every pair's
coherence and phase, the bands in decibels — and a pair left unstated
refuses the specification: independent channels are a statement too
(Brandon, 2026-09-04). The operations each give a new draft, so a
script reads as a sequence of statements.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.author import SpecificationDraft
from visualdynamics.core.rigid import MassProperties, rigid_body_shapes
from visualdynamics.core.transform import to_modal, to_physical


def _rigid_set(seed=1):
    rng = np.random.default_rng(seed)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    return geometry, rigid_body_shapes(geometry, MassProperties((0, 0, 0)))


def _draft(channels=None, dims=None):
    channels = channels or ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    dims = dims or ['acceleration'] * 3 + ['angular_acceleration'] * 3
    return SpecificationDraft(
        channels, dims, [20.0, 100.0, 2000.0],
        [[0.01, 0.04, 0.04], [0.01, 0.04, 0.04], [0.02, 0.08, 0.08],
         [1.0, 4.0, 4.0], [1.0, 4.0, 4.0], [2.0, 8.0, 8.0]],
        {(i, j): (0.0, 0.0) for i in range(6) for j in range(i + 1, 6)})


def _pairs(spec):
    return {(r, f): i for i, (r, f) in enumerate(
        zip(spec.response_dof, spec.reference_dof))}


# ---- the doors in --------------------------------------------------------------


def test_at_modal_coordinates_starts_unstated_and_wears_the_points_units():
    _geometry, rigid = _rigid_set()
    draft = SpecificationDraft.at_modal_coordinates(rigid)
    assert draft.channels == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    assert draft.dims == ['acceleration'] * 3 + ['angular_acceleration'] * 3
    assert len(draft.unset_pairs()) == 15
    alone = draft.make()
    assert alone.num_records == 6, \
        'an unstated pair is absent, not assumed and not a refusal'
    assert alone.response_dof == alone.reference_dof
    stated = draft.with_all_pairs(0.0)
    assert stated.unset_pairs() == []
    assert len(draft.unset_pairs()) == 15, \
        'with_all_pairs returns a new draft; the sheet is unchanged'
    assert stated.make().num_records == 36
    scaled = rigid_body_shapes(_geometry, MassProperties(
        (0, 0, 0), mass=2.0, inertia=(1.0, 1.0, 1.0, 0, 0, 0)))
    assert set(SpecificationDraft.at_modal_coordinates(scaled).dims) == \
        {'modal_acceleration'}
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    with pytest.raises(ValueError, match='no mass unit'):
        SpecificationDraft.at_modal_coordinates(shapes)
    # the shapes picked alone: a virtual point's three translations, the
    # rotations left out — expanded, a shape with no channel contributes
    # nothing, which is what a zero would say (Brandon, 2026-09-06)
    three = SpecificationDraft.at_modal_coordinates(rigid, modes=[0, 1, 2])
    assert three.channels == ['M1', 'M2', 'M3']
    assert three.dims == ['acceleration'] * 3
    assert len(three.unset_pairs()) == 3
    assert SpecificationDraft.at_modal_coordinates(rigid, modes=[5]).dims == \
        ['angular_acceleration']
    with pytest.raises(ValueError, match='no shape 7'):
        SpecificationDraft.at_modal_coordinates(rigid, modes=[6])
    with pytest.raises(ValueError, match='no shape picked'):
        SpecificationDraft.at_modal_coordinates(rigid, modes=[])


def test_a_specification_without_limits_opens_wearing_the_defaults_and_says_so():
    """Brandon, 2026-09-06: where a specification has no warning or
    abort limits, the user should be able to add them on the sheet.
    They are there to drag — the defaults — and the note says they are
    not on the object until an edit writes them."""
    axis = np.geomspace(10.0, 2000.0, 16)
    level = np.atleast_2d(np.full(16, 1e-3))
    bare = visualdynamics.Specification(
        abscissa=axis, ordinate=level, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])
    from visualdynamics.core.author import band_notes

    assert not bare.limits
    draft = SpecificationDraft.from_specification(bare)
    assert draft.bands[0]['warning'] == [(-3.0, 3.0)] * 15
    assert draft.bands[0]['abort'] == [(-6.0, 6.0)] * 15
    assert draft.notes == [], 'the saying is about the object, not the sheet'
    assert band_notes(bare) == [
        ('no warning band on the specification — the sheet shows -3/+3 dB, '
         'written with the first edit'),
        ('no abort band on the specification — the sheet shows -6/+6 dB, '
         'written with the first edit')]
    made = draft.with_band_edge('warning', 'upper', [0], 2.0).make()
    assert sorted(made.limits) == ['abort_lower', 'abort_upper',
                                   'warning_lower', 'warning_upper']
    assert band_notes(made) == [], 'on the object now'
    assert SpecificationDraft.from_specification(made).bands[0]['warning'] == \
        [(-2.0, 2.0)] * 15
    # one band there and the other not: the note names the missing one
    half = visualdynamics.Specification(
        abscissa=axis, ordinate=level, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'],
        warning_lower=level / 2, warning_upper=level * 2)
    assert [n[:13] for n in band_notes(half)] == ['no abort band']


def test_at_dofs_and_at_control_channels():
    draft = SpecificationDraft.at_dofs(['101Z+', '104Z+'], 'acceleration')
    assert draft.channels == ['101Z+', '104Z+']
    assert draft.dims == ['acceleration', 'acceleration']
    assert draft.unset_pairs() == [(0, 1)]
    project = visualdynamics.Project('t')
    project.import_file(fixture_path('plate', 'random.nc4'))
    table = project['Channel Table']
    draft = SpecificationDraft.at_control_channels(table)
    flagged = [dof for dof, on in zip(table.dof_strings(), table.controls())
               if on]
    assert draft.channels == flagged
    assert set(draft.dims) == {'acceleration'}


def test_from_specification_opens_what_the_sheet_can_hold():
    """Cross terms that hold one coherence and phase over frequency
    come in as such; ones that vary are left unstated and said; bands
    the same on every channel come in as decibels."""
    written = _draft()
    written.pairs[(0, 5)] = (0.25, 30.0)
    spec = written.with_band('warning', -2.0, 2.0).with_band(
        'abort', -5.0, 5.0).make()
    back = SpecificationDraft.from_specification(spec)
    assert back.channels == written.channels
    assert back.dims == written.dims
    assert back.frequencies == written.frequencies
    assert np.allclose(back.levels, written.levels)
    assert back.pairs[(0, 5)] == pytest.approx((0.25, 30.0))
    assert back.pairs[(1, 2)] == (0.0, 0.0)
    assert back.bands[0]['warning'] == [pytest.approx((-2.0, 2.0))] * 2
    assert back.bands[0]['abort'] == [pytest.approx((-5.0, 5.0))] * 2
    assert back.bands[0]['symmetric'] and back.bands[0]['uniform']
    assert back.notes == []
    # a cross term that changes with frequency does not fit the sheet
    pairs = _pairs(spec)
    spec.ordinate[pairs[('M1', 'M6')]] *= np.array([1.0, 0.5, 0.25])
    spec.ordinate[pairs[('M6', 'M1')]] = np.conj(spec.ordinate[pairs[('M1', 'M6')]])
    varied = SpecificationDraft.from_specification(spec)
    assert varied.pairs[(0, 5)] is None
    assert varied.notes == [('1 cross term vary with frequency and cannot be '
                             'one coherence and phase (M1–M6)')]
    # and the controller's own target opens, autos only, pairs unstated
    project = visualdynamics.Project('t')
    project.import_file(fixture_path('plate', 'random.nc4'))
    target = project['Specification']
    opened = SpecificationDraft.from_specification(target)
    assert opened.channels == target.response_dof
    kept = (target.abscissa > 0) & np.all(np.real(target.ordinate) > 0, axis=0)
    assert opened.frequencies == [float(f) for f in target.abscissa[kept]]
    assert 0.0 not in opened.frequencies and min(opened.levels[0]) > 0
    assert opened.notes == [(f'{int((~kept).sum())} lines at 0 Hz or with a '
                             'zero level left out — a sheet holds positive '
                             'levels at positive frequencies')]
    assert len(opened.unset_pairs()) == 28
    one = visualdynamics.Specification(
        target.abscissa[:3], np.zeros((1, 3)), response_dof=['1Z+'],
        reference_dof=['1Z+'], ordinate_dim=['acceleration**2/frequency'])
    with pytest.raises(ValueError, match='fewer than two lines'):
        SpecificationDraft.from_specification(one)


# ---- what it makes -----------------------------------------------------------------


def test_the_specification_is_what_was_stated():
    draft = _draft()
    draft.pairs[(0, 5)] = (0.25, 30.0)
    spec = draft.make('written on Modes')
    pairs = _pairs(spec)
    assert np.allclose(np.real(spec.ordinate[pairs[('M1', 'M1')]]),
                       [0.01, 0.04, 0.04])
    assert spec.ordinate_dim[pairs[('M1', 'M1')]] == \
        'acceleration**2/frequency'
    assert spec.ordinate_dim[pairs[('M6', 'M6')]] == \
        'angular_acceleration**2/frequency', 'a rotation, per radian'
    assert spec.ordinate_dim[pairs[('M1', 'M6')]] == \
        'acceleration*angular_acceleration/frequency'
    cross = spec.ordinate[pairs[('M1', 'M6')]]
    assert np.allclose(np.abs(cross), np.sqrt(0.25 * np.array(
        [0.01, 0.04, 0.04]) * np.array([2.0, 8.0, 8.0])))
    assert np.allclose(np.degrees(np.angle(cross)), 30.0)
    assert np.allclose(spec.ordinate[pairs[('M6', 'M1')]], np.conj(cross)), \
        'Hermitian by construction'
    assert np.allclose(spec.ordinate[pairs[('M1', 'M2')]], 0.0), \
        'independent: a stated zero'
    i = pairs[('M2', 'M2')]
    assert np.allclose(spec.limits['abort_upper'][i], 10 ** 0.6 * np.array(
        [0.01, 0.04, 0.04]))
    assert np.allclose(spec.limits['warning_lower'][i], 10 ** -0.3 * np.array(
        [0.01, 0.04, 0.04]))
    assert np.isnan(spec.limits['abort_upper'][pairs[('M1', 'M6')]]).all()
    assert spec.interpolation == 'log_log'
    assert spec.comment[0] == 'written on Modes'


def test_it_expands_to_an_exact_control_channel_target():
    """The virtual point door: the physical specification carries the
    author's cross terms, and transforms back to what was written."""
    _geometry, rigid = _rigid_set()
    spec = _draft().with_all_pairs(1.0, 0.0).make()     # one coherent motion
    physical, _report = to_physical(spec, rigid)
    assert isinstance(physical, visualdynamics.Specification)
    assert physical.num_records == rigid.num_dofs ** 2
    assert physical.has_limits
    again, report = to_modal(physical, rigid)
    pairs = _pairs(again)
    for i, (r, f) in enumerate(zip(spec.response_dof, spec.reference_dof)):
        assert np.allclose(again.ordinate[pairs[(r, f)]], spec.ordinate[i],
                           atol=1e-12), (r, f)
    assert report.worst_residual == pytest.approx(0.0, abs=1e-9)


def test_the_preview_is_the_autospectra_with_their_bands():
    _geometry, rigid = _rigid_set()
    preview = SpecificationDraft.at_modal_coordinates(rigid).preview()
    assert preview.num_records == 6 and preview.has_limits
    assert not np.iscomplexobj(preview.ordinate)
    assert preview.response_dof == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    assert preview.interpolation == 'log_log'


# ---- the operations ---------------------------------------------------------------


def test_the_operations_each_give_a_new_draft():
    draft = _draft()
    louder = draft.scaled_db(3.0)
    assert np.allclose(louder.levels[0], np.array([0.01, 0.04, 0.04]) * 10 ** 0.3)
    assert draft.levels[0] == [0.01, 0.04, 0.04], 'the original is unchanged'
    one = draft.scaled_db(-6.0, ['M3'])
    assert np.allclose(one.levels[2], np.array([0.02, 0.08, 0.08]) * 10 ** -0.6)
    assert one.levels[0] == draft.levels[0]
    with pytest.raises(ValueError, match="no channel 'M9'"):
        draft.scaled_db(1.0, ['M9'])
    banded = draft.with_band('abort', -9.0, 9.0)
    assert banded.bands[0]['abort'] == [(-9.0, 9.0)] * 2
    assert banded.bands[0]['warning'] == [(-3.0, 3.0)] * 2
    more = draft.with_breakpoint(50.0)
    assert more.frequencies == [20.0, 50.0, 100.0, 2000.0]
    # on the power law between 20 and 100 Hz: log-log interpolation
    expected = np.exp(np.interp(np.log(50.0), np.log([20.0, 100.0]),
                                np.log([0.01, 0.04])))
    assert more.levels[0][1] == pytest.approx(expected)
    assert more.with_breakpoint(50.0).frequencies == more.frequencies
    fewer = more.without_breakpoint(50.0)
    assert fewer.frequencies == draft.frequencies
    assert np.allclose(fewer.levels, draft.levels)
    with pytest.raises(ValueError, match='no breakpoint at 77'):
        draft.without_breakpoint(77.0)
    two = SpecificationDraft.at_dofs(['101Z+'])
    with pytest.raises(ValueError, match='at least two breakpoints'):
        two.without_breakpoint(20.0)


def test_what_no_specification_could_hold_is_refused():
    with pytest.raises(ValueError, match='at least two breakpoints'):
        SpecificationDraft(['M1'], ['acceleration'], [20.0], [[1.0]])
    with pytest.raises(ValueError, match='increase'):
        SpecificationDraft(['M1'], ['acceleration'], [20.0, 10.0], [[1.0, 1.0]])
    with pytest.raises(ValueError, match='positive level'):
        SpecificationDraft(['M1'], ['acceleration'], [20.0, 30.0], [[1.0, 0.0]])
    with pytest.raises(ValueError, match='coherence'):
        SpecificationDraft(['M1', 'M2'], ['acceleration'] * 2, [20.0, 30.0],
                           [[1.0, 1.0], [1.0, 1.0]], {(0, 1): (1.5, 0.0)})
    with pytest.raises(ValueError, match='below and above'):
        SpecificationDraft(['M1'], ['acceleration'], [20.0, 30.0], [[1.0, 1.0]],
                           bands=[{'warning': [(3.0, 6.0)],
                                   'abort': [(-6.0, 6.0)]}])
    with pytest.raises(ValueError, match='named twice'):
        SpecificationDraft(['M1', 'M1'], ['acceleration'] * 2, [20.0, 30.0],
                           [[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match='quantity'):
        SpecificationDraft(['M1'], ['unknown'], [20.0, 30.0], [[1.0, 1.0]])


def test_the_dict_form_round_trips():
    draft = _draft()
    draft.pairs[(2, 4)] = None
    draft.notes.append('a note')
    back = SpecificationDraft.from_dict(draft.as_dict())
    assert back == draft


# ---- through the project ---------------------------------------------------


def _project():
    geometry, rigid = _rigid_set()
    project = visualdynamics.Project('A')
    project.add('Plate', geometry)
    project.add('Modes', rigid)
    project.link('Plate', 'Modes')
    return project


def test_the_verb_links_journals_and_refreshes(tmp_path):
    project = _project()
    draft = _draft()
    name = project.author_specification('Modes', draft)
    assert name == 'Modes Specification'
    assert name in project.links[0]['members'], \
        'a modal object belongs with its set'
    assert project.journal[-1].startswith(
        "project.author_specification('Modes', SpecificationDraft(")
    assert project.provenance[name]['params']['draft'] == draft.as_dict()
    starter = project.author_specification(
        'Modes', SpecificationDraft.at_modal_coordinates(project['Modes']))
    assert project[starter].num_records == 6, 'autospectra alone'
    project.remove(starter)
    assert 'author_specification' in dict(project.verbs('Modes'))
    assert 'author_specification' in dict(project.verbs(name))
    assert 'author_specification' not in dict(project.verbs('Plate'))

    back = visualdynamics.Project.open(project.save(tmp_path / 'a.vdyn'))
    assert back[name] == project[name]
    back.refresh(name)
    assert back[name] == project[name], 'the recipe rebuilds it exactly'

    expanded = project.expand(name, 'Modes')
    assert expanded == 'Modes Specification at Physical DOFs'
    assert isinstance(project[expanded], visualdynamics.Specification)

    script = project.session_script()
    assert 'from visualdynamics.core.author import SpecificationDraft' \
        in script


def test_a_specification_is_replaced_in_place():
    project = _project()
    name = project.author_specification('Modes', _draft())
    project.link('Modes', name)
    opened = SpecificationDraft.from_specification(project[name])
    edited = opened.scaled_db(3.0).with_band('warning', -1.0, 1.0)
    same = project.author_specification(name, edited, replace=True)
    assert same == name
    assert name in project.links[0]['members'], 'links kept'
    pairs = _pairs(project[name])
    assert np.allclose(np.real(project[name].ordinate[pairs[('M1', 'M1')]]),
                       np.array([0.01, 0.04, 0.04]) * 10 ** 0.3)
    assert project.provenance[name]['source'] == name
    assert project.journal[-1].endswith(', replace=True)')
    with pytest.raises(TypeError, match='not a specification'):
        project.author_specification('Modes', edited, replace=True)
    project.refresh(name)
    assert np.allclose(np.real(project[name].ordinate[pairs[('M1', 'M1')]]),
                       np.array([0.01, 0.04, 0.04]) * 10 ** 0.3)


def test_a_channel_tables_control_channels_are_a_door(tmp_path):
    project = visualdynamics.Project('t')
    project.import_file(fixture_path('plate', 'random.nc4'))
    draft = SpecificationDraft.at_control_channels(
        project['Channel Table']).with_all_pairs(1.0)
    name = project.author_specification('Channel Table', draft)
    made = project[name]
    assert made.response_dof[:8:9] == draft.channels[:1]
    assert made.num_records == len(draft.channels) ** 2
    assert name in next(group['members'] for group in project.links
                        if 'Channel Table' in group['members'])


# ---- picks, several specifications, written into -----------------------------


def _target():
    project = visualdynamics.Project('t')
    project.import_file(fixture_path('plate', 'random.nc4'))
    return project, project['Specification']


def test_a_door_restricted_to_channels():
    _project, target = _target()
    dofs = target.response_dof
    draft = SpecificationDraft.from_specification(target, [dofs[3], dofs[0]])
    assert draft.channels == [dofs[0], dofs[3]], 'in the specification\'s order'
    assert draft.unset_pairs() == [(0, 1)]
    with pytest.raises(ValueError, match='999X\\+ has no autospectrum'):
        SpecificationDraft.from_specification(target, ['999X+'])


def test_written_into_keeps_the_lines_and_the_rest():
    """The sheet's channels take the sheet over its range at the
    specification's own lines; the lines it left out keep what they
    had; channels it does not hold are untouched; a cross term it
    states is made at every line from the autospectra there."""
    _project, target = _target()
    dofs = target.response_dof
    draft = SpecificationDraft.from_specification(target, dofs[:2])
    edited = draft.scaled_db(3.0).with_all_pairs(1.0, 90.0)
    out = edited.written_into(target, 'edited')
    assert np.array_equal(out.abscissa, target.abscissa)
    assert out.num_records == 10
    kept = (target.abscissa > 0) & np.all(np.real(target.ordinate) > 0, axis=0)
    for i in (0, 1):
        assert np.allclose(np.real(out.ordinate[i])[kept],
                           np.real(target.ordinate[i])[kept] * 10 ** 0.3)
        assert np.allclose(np.real(out.ordinate[i])[~kept],
                           np.real(target.ordinate[i])[~kept])
        assert np.allclose(out.limits['abort_upper'][i][kept],
                           np.real(out.ordinate[i])[kept] * 10 ** 0.6)
    for i in range(2, 8):
        assert np.array_equal(out.ordinate[i], target.ordinate[i])
        assert np.array_equal(out.limits['warning_lower'][i],
                              target.limits['warning_lower'][i],
                              equal_nan=True)
    pairs = {(r, f): i for i, (r, f) in enumerate(
        zip(out.response_dof, out.reference_dof))}
    cross = out.ordinate[pairs[(dofs[0], dofs[1])]]
    expected = np.sqrt(np.real(out.ordinate[0]) * np.real(out.ordinate[1]))
    assert np.allclose(np.abs(cross), expected), \
        'coherent at every line, the left-out ones from their own autos'
    assert np.allclose(np.angle(cross[kept]), np.pi / 2)
    assert np.allclose(out.ordinate[pairs[(dofs[1], dofs[0])]], np.conj(cross))
    assert out.comment[-1] == 'edited' and out.comment[0] == target.comment[0]
    assert out.interpolation == target.interpolation
    # a breakpoint that is not a line becomes one, the untouched
    # channels read off their own power law there — exact between
    # two lines of a power law
    with_line = edited.with_breakpoint(101.0).written_into(target)
    assert 101.0 in with_line.abscissa
    k = list(with_line.abscissa).index(101.0)
    lo, hi = list(target.abscissa).index(100.0), list(target.abscissa).index(102.0)
    for i in range(2, 8):
        a, b = np.real(target.ordinate[i][lo]), np.real(target.ordinate[i][hi])
        power = np.exp(np.interp(np.log(101.0), np.log([100.0, 102.0]),
                                 np.log([a, b])))
        assert np.real(with_line.ordinate[i][k]) == pytest.approx(power)
    # a pair the sheet leaves unstated keeps the object's own record
    # where it had one, and is absent where it had none
    half = SpecificationDraft.from_specification(target, dofs[:2])
    half.pairs[(0, 1)] = (0.5, 0.0)
    coherent = half.written_into(target)
    assert coherent.num_records == 10
    quiet = SpecificationDraft.from_specification(coherent, dofs[:2])
    assert quiet.pairs[(0, 1)] == pytest.approx((0.5, 0.0))
    quiet.pairs[(0, 1)] = None
    kept = quiet.scaled_db(1.0).written_into(coherent)
    assert kept.num_records == 10, 'the record it held stays as it was'
    pairs = {(r, f): i for i, (r, f) in enumerate(
        zip(kept.response_dof, kept.reference_dof))}
    assert np.allclose(kept.ordinate[pairs[(dofs[0], dofs[1])]],
                       coherent.ordinate[pairs[(dofs[0], dofs[1])]])
    assert draft.written_into(target).num_records == 8, \
        'unstated and never held: absent'


def test_a_whole_sheet_rewrites_the_object_in_its_own_form():
    """Fold a target to its breakpoints and the object *is* the
    breakpoints; a picked subset merges at the object's lines."""
    project, target = _target()
    whole = SpecificationDraft.from_specification(target).to_breakpoints()
    project.author_specification('Specification', whole, replace=True)
    folded = project['Specification']
    assert len(folded.abscissa) == len(whole.frequencies)
    assert folded.num_records == 8
    assert folded.interpolation == 'log_log'
    back = SpecificationDraft.from_specification(folded).to_lines(2.0)
    project.author_specification('Specification', back, replace=True)
    assert len(project['Specification'].abscissa) == len(back.frequencies)
    part = SpecificationDraft.from_specification(
        project['Specification'], target.response_dof[:2]).to_breakpoints()
    project.author_specification('Specification', part, replace=True)
    assert len(project['Specification'].abscissa) == len(back.frequencies), \
        'a subset merges at the lines the object has'


def test_written_into_the_whole_specification_is_the_refresh_recipe():
    project, target = _target()
    draft = SpecificationDraft.from_specification(target).with_all_pairs(0.0)
    same = project.author_specification('Specification', draft, replace=True)
    assert same == 'Specification'
    once = project['Specification']
    assert once.num_records == 64
    kept = (target.abscissa > 0) & np.all(np.real(target.ordinate) > 0, axis=0)
    assert np.array_equal(once.abscissa, target.abscissa[kept]), \
        'the whole sheet: the object takes the sheet\'s lines'
    project.refresh('Specification')
    again = project['Specification']
    assert np.allclose(again.ordinate, once.ordinate, equal_nan=True), \
        'the same sheet written in again lands on the same lines'
    assert project.provenance['Specification']['params']['into'] is True


def test_a_sheet_across_specifications():
    project, target = _target()
    dofs = target.response_dof
    project.add('Other', target)
    a = SpecificationDraft.from_specification(target, dofs[:2], source='Specification')
    b = SpecificationDraft.from_specification(project['Other'], dofs[2:3],
                                              source='Other')
    both = SpecificationDraft.across({'Specification': a, 'Other': b})
    assert both.channels == dofs[:3]
    assert both.sources == ['Specification', 'Specification', 'Other']
    assert both.pairable() == [(0, 1)], 'no cross term across specifications'
    assert both.unset_pairs() == [(0, 1)]
    assert both.frequencies == a.frequencies
    assert both.notes[0].startswith('Specification: ')
    with pytest.raises(ValueError, match='different specifications'):
        both._copy(pairs={(0, 2): (0.0, 0.0)})
    with pytest.raises(ValueError, match='spans several'):
        both.make()
    with pytest.raises(ValueError, match='spans several'):
        both.preview()
    assert both.preview  # the sheet previews whatever is stated
    own = both.for_source('Other')
    assert own.channels == dofs[2:3] and own.sources == []
    assert own.notes == b.notes
    with pytest.raises(ValueError, match='no channel of'):
        both.for_source('Nobody')
    stated = both.with_all_pairs(1.0).scaled_db(-3.0)
    assert stated.pairs == {(0, 1): (1.0, 0.0)}
    first = project.author_specification(['Specification', 'Other'], stated,
                                         replace=True)
    assert first == 'Specification'
    assert project['Specification'].num_records == 10
    assert project['Other'].num_records == 8, 'one channel: no pair to add'
    kept = (target.abscissa > 0) & np.all(np.real(target.ordinate) > 0, axis=0)
    assert np.allclose(np.real(project['Other'].ordinate[2])[kept],
                       np.real(target.ordinate[2])[kept] * 10 ** -0.3)
    assert np.array_equal(project['Other'].ordinate[3], target.ordinate[3])
    assert project.journal[-1].startswith(
        "project.author_specification(['Specification', 'Other'], ")
    assert SpecificationDraft.from_dict(stated.as_dict()) == stated
    # one draft spanning nothing is itself, named
    alone = SpecificationDraft.across({'Specification': a})
    assert alone.sources == ['Specification'] * 2
    # sheets that share no range cannot meet
    low = SpecificationDraft.at_dofs(['1X+']).with_breakpoint(
        30.0).with_breakpoint(40.0).without_breakpoint(
        2000.0).without_breakpoint(20.0)
    high = SpecificationDraft.at_dofs(['2X+']).with_breakpoint(
        3000.0).with_breakpoint(4000.0).without_breakpoint(
        2000.0).without_breakpoint(20.0)
    assert low.frequencies == [30.0, 40.0] and high.frequencies == [3000.0, 4000.0]
    with pytest.raises(ValueError, match='share fewer than two'):
        SpecificationDraft.across({'Low': low, 'High': high})
    with pytest.raises(ValueError, match='no sheets'):
        SpecificationDraft.across({})


# ---- the two forms ------------------------------------------------------------


def _controller_target():
    """A target as a controller writes it: four breakpoints read onto
    0.5 Hz lines from 0 to 4096 Hz, zero outside the band, +6 dB
    bands — the shape of the drone project's specification."""
    from visualdynamics.core.compliance import log_interpolate

    f = np.arange(0.0, 4096.5, 0.5)
    points, levels = [20.0, 40.0, 200.0, 2000.0], [0.01, 0.04, 0.04, 0.005]
    inside = (f >= 20.0) & (f <= 2000.0)
    rows = []
    for k in range(2):
        row = np.zeros(len(f))
        row[inside] = log_interpolate(f[inside], np.asarray(points),
                                      np.asarray(levels) * (k + 1))
        rows.append(row)
    rows = np.asarray(rows)
    return visualdynamics.Specification(
        f, rows, response_dof=['101Z+', '104Z+'],
        reference_dof=['101Z+', '104Z+'],
        ordinate_dim='acceleration**2/frequency',
        abort_upper=rows * 10 ** 0.6, abort_lower=rows * 10 ** -0.6)


def test_a_controllers_target_opens_as_lines_and_folds_to_its_breakpoints():
    from visualdynamics.core.author import uniform_spacing

    target = _controller_target()
    assert uniform_spacing(target.abscissa) == 0.5
    assert uniform_spacing([20.0, 40.0, 200.0]) is None
    assert uniform_spacing([1.0, 2.0]) is None, 'two lines say nothing'
    opened = SpecificationDraft.from_specification(target)
    assert opened.form == 'lines' and opened.spacing == 0.5
    assert len(opened.frequencies) == 3961, 'the band, 20 to 2000 at 0.5'
    few = opened.to_breakpoints()
    assert few.form == 'breakpoints' and few.spacing == 0.5, \
        'the spacing is remembered for the way back'
    assert few.frequencies == [20.0, 40.0, 200.0, 2000.0]
    assert np.allclose(few.levels[0], [0.01, 0.04, 0.04, 0.005])
    assert np.allclose(few.levels[1], [0.02, 0.08, 0.08, 0.01])
    assert few.notes[-1] == '3961 lines read as 4 breakpoints'
    assert few.bands[0]['abort'][0] == pytest.approx((-6.0, 6.0))
    assert opened.form == 'lines', 'a new draft; the sheet is unchanged'
    # and back onto the same lines, exactly
    back = few.to_lines()
    assert back.form == 'lines' and back.spacing == 0.5
    assert back.frequencies == opened.frequencies
    assert np.allclose(back.levels, opened.levels)
    assert SpecificationDraft.from_dict(back.as_dict()) == back
    # the whole sheet folded and written back: the object *is* the
    # breakpoints now (Brandon, 2026-09-06), and interpolated again it
    # is the lines in the band
    project = visualdynamics.Project('t')
    project.add('Target', target)
    edited = few.scaled_db(3.0).with_all_pairs(0.0)
    project.author_specification('Target', edited, replace=True)
    written = project['Target']
    assert list(written.abscissa) == [20.0, 40.0, 200.0, 2000.0]
    assert np.allclose(np.real(written.ordinate[0]),
                       np.array([0.01, 0.04, 0.04, 0.005]) * 10 ** 0.3)
    project.author_specification(
        'Target', SpecificationDraft.from_specification(written).to_lines(0.5),
        replace=True)
    lines = project['Target']
    kept = (target.abscissa >= 20.0) & (target.abscissa <= 2000.0)
    assert np.array_equal(lines.abscissa, target.abscissa[kept])
    assert np.allclose(np.real(lines.ordinate[0]),
                       np.real(target.ordinate[0])[kept] * 10 ** 0.3)


def test_lines_are_multiples_of_the_spacing_and_edges_off_the_grid_are_said():
    draft = SpecificationDraft.at_dofs(['1Z+']).with_breakpoint(100.0)
    lines = draft.to_lines(4.0)
    assert lines.frequencies[0] == 20.0 and lines.frequencies[-1] == 2000.0
    assert lines.frequencies[:3] == [20.0, 24.0, 28.0]
    assert all(f % 4.0 == 0 for f in lines.frequencies), \
        'a controller lays its lines at multiples of the spacing'
    assert np.allclose(lines.levels[0], 1.0)
    assert lines.notes == []
    off = draft.to_lines(3.0)
    assert off.frequencies[0] == 21.0 and off.frequencies[-1] == 1998.0
    assert off.notes == [('the band edges at 20 and 2000 Hz are not on the '
                          '3 Hz grid; the lines run 21 to 1998 Hz')]
    with pytest.raises(ValueError, match='no frequency spacing'):
        draft.to_lines()
    with pytest.raises(ValueError, match='fewer than two lines'):
        draft.to_lines(5000.0)
    with pytest.raises(ValueError, match='positive'):
        draft.to_lines(0.0)
    with pytest.raises(ValueError, match="'breakpoints' or 'lines'"):
        draft._copy(form='decimated')


def test_a_measured_shape_keeps_every_line_and_says_so():
    f = np.arange(10.0, 100.0, 10.0)
    rng = np.random.default_rng(3)
    draft = SpecificationDraft(['1Z+'], ['acceleration'], f,
                               [list(rng.uniform(0.5, 2.0, len(f)))])
    same = draft.to_breakpoints()
    assert same.frequencies == draft.frequencies
    assert same.notes == [('no power-law structure in its 9 lines: every '
                           'line kept as a breakpoint')]
    assert draft.to_breakpoints().form == 'breakpoints'
    two = SpecificationDraft.at_dofs(['1Z+']).to_breakpoints()
    assert two.frequencies == [20.0, 2000.0] and two.notes == []


# ---- bands per channel and per segment --------------------------------------------


def _shaped():
    """One channel with a plateau and two skirts, two segments each side."""
    return SpecificationDraft(
        ['1Z+', '2Z+'], ['acceleration'] * 2, [10.0, 40.0, 200.0, 2000.0],
        [[0.0002, 0.002, 0.002, 0.0002], [0.0001, 0.001, 0.001, 0.0001]],
        {(0, 1): (0.0, 0.0)})


def test_a_band_belongs_to_a_channel_and_a_segment():
    draft = _shaped()
    assert len(draft.bands) == 2
    assert draft.bands[0] == {'warning': [(-3.0, 3.0)] * 3,
                              'abort': [(-6.0, 6.0)] * 3,
                              'symmetric': True, 'uniform': True}
    assert draft.segment_of([10.0, 20.0, 40.0, 100.0, 200.0, 2000.0, 5000.0]
                            ).tolist() == [0, 0, 1, 1, 2, 2, 2], \
        'a breakpoint belongs to the segment on its right, the last to its left'
    assert draft.segment_of().tolist() == [0, 1, 2, 2]
    # stated outright, on one channel and one segment, once uniform is off
    loose = draft.constrained(['1Z+'], uniform=False)
    one = loose.with_band('abort', -9.0, 9.0, ['1Z+'], [1])
    assert one.bands[0]['abort'] == [(-6.0, 6.0), (-9.0, 9.0), (-6.0, 6.0)]
    assert one.bands[1]['abort'] == [(-6.0, 6.0)] * 3, 'the other channel untouched'
    _below, above = one.band_db('abort', 0, [20.0, 100.0, 1000.0])
    assert above.tolist() == [6.0, 9.0, 6.0]
    # on a uniform channel a segment cannot differ: the whole channel moves
    every = draft.with_band('abort', -9.0, 9.0, ['1Z+'], [1])
    assert every.bands[0]['abort'] == [(-9.0, 9.0)] * 3
    with pytest.raises(ValueError, match='symmetric'):
        draft.with_band('warning', -2.0, 3.0)
    with pytest.raises(ValueError, match='is uniform'):
        draft._copy(bands=[{'warning': [(-3.0, 3.0), (-2.0, 2.0), (-3.0, 3.0)],
                            'abort': [(-6.0, 6.0)] * 3}, draft.bands[1]])
    with pytest.raises(ValueError, match='is symmetric'):
        draft._copy(bands=[{'warning': [(-3.0, 2.0)] * 3,
                            'abort': [(-6.0, 6.0)] * 3}, draft.bands[1]])
    with pytest.raises(ValueError, match='3 segments'):
        draft._copy(bands=[{'warning': [(-3.0, 3.0)], 'abort': [(-6.0, 6.0)]},
                           draft.bands[1]])


def test_a_drag_lands_under_each_channels_constraints():
    """The drag on the plot: one edge, the segments under it, some
    decibels — and the channel's own rules decide what else moves."""
    draft = _shaped()
    # symmetric and uniform: dragging the upper abort edge on one
    # segment moves both edges of every segment, on every channel
    moved = draft.shifted_band('abort', 'upper', [1], 3.0)
    for entry in moved.bands:
        assert entry['abort'] == [(-9.0, 9.0)] * 3
        assert entry['warning'] == [(-3.0, 3.0)] * 3, 'the other band stays'
    # channels are edited by name: the selected ones
    one = draft.shifted_band('abort', 'lower', [0], -2.0, channels=['2Z+'])
    assert one.bands[0]['abort'] == [(-6.0, 6.0)] * 3
    assert one.bands[1]['abort'] == [(-8.0, 8.0)] * 3
    # neither constraint: the one edge of the one segment
    loose = draft.constrained(symmetric=False, uniform=False)
    edge = loose.shifted_band('abort', 'upper', [1], 3.0, channels=['1Z+'])
    assert edge.bands[0]['abort'] == [(-6.0, 6.0), (-6.0, 9.0), (-6.0, 6.0)]
    # an edge cannot cross the target: a step short of it at the least
    pinned = loose.shifted_band('abort', 'upper', [1], -20.0)
    assert pinned.bands[0]['abort'][1] == (-6.0, 1.0)
    # and it lands on whole decibels (Brandon, 2026-09-06: never 1.5)
    assert draft.shifted_band('abort', 'upper', [0], 0.7).bands[0]['abort'][0] == (-7.0, 7.0)
    assert draft.shifted_band('abort', 'upper', [0], 0.4).bands[0]['abort'][0] == (-6.0, 6.0)
    odd = draft._copy(bands=[{'warning': [(-3.0, 3.0)] * 3,
                              'abort': [(-6.3, 6.3)] * 3}, draft.bands[1]])
    assert odd.shifted_band('abort', 'upper', [0], 1.0).bands[0]['abort'][0] == (-7.0, 7.0), \
        'a band off the grid lands on it'
    assert odd.shifted_band('abort', 'upper', [0], 0.25, step=0.5).bands[0]['abort'][0] == (-6.5, 6.5)
    # the drag itself lands every channel *at* a level, not each a
    # step further from its own (Brandon, 2026-09-06: a channel at 3 dB
    # and one at 4 dB, dragged to 4, are both at 4 — not 4 and 5)
    uneven = draft.with_band('warning', -4.0, 4.0, ['2Z+'])
    assert [e['warning'][0] for e in uneven.bands] == [(-3.0, 3.0), (-4.0, 4.0)]
    landed = uneven.with_band_edge('warning', 'upper', [1], 4.0)
    assert all(e['warning'] == [(-4.0, 4.0)] * 3 for e in landed.bands)
    assert uneven.shifted_band('warning', 'upper', [1], 1.0).bands[1]['warning'][0] == \
        (-5.0, 5.0), 'the shift is still there for a script that wants it'
    below = uneven.with_band_edge('warning', 'lower', [0], -2.5, channels=['2Z+'])
    assert below.bands[1]['warning'] == [(-2.0, 2.0)] * 3, 'on the grid, mirrored'
    assert below.bands[0]['warning'] == [(-3.0, 3.0)] * 3
    assert loose.with_band_edge('abort', 'upper', [1], 0.2).bands[0]['abort'][1] == \
        (-6.0, 1.0), 'never nearer the target than a step'
    # constraints turned on settle the bands to their rule
    tidy = edge.constrained(['1Z+'], symmetric=True)
    assert tidy.bands[0]['abort'] == [(-6.0, 6.0), (-9.0, 9.0), (-6.0, 6.0)]
    same = tidy.constrained(['1Z+'], uniform=True)
    assert same.bands[0]['abort'] == [(-6.0, 6.0)] * 3, "the first segment's"
    assert draft.band_runs(0, 'abort') == [[0, 1, 2]]
    assert edge.band_runs(0, 'abort') == [[0], [1], [2]]
    assert SpecificationDraft.from_dict(edge.as_dict()) == edge


def test_the_object_carries_a_step_in_the_band_and_folds_back():
    """A limit array holds one value per line and a breakpoint belongs
    to the segment on its right, so a step needs a helper line a hair
    before the breakpoint carrying the left segment's band."""
    draft = _shaped().constrained(uniform=False).with_band(
        'abort', -9.0, 9.0, ['1Z+'], segments=[1])
    assert draft.bands_differ()
    assert not _shaped().bands_differ()
    cornered = draft.with_band_corners()
    assert cornered.frequencies == pytest.approx(
        [10.0, 40.0 * (1 - 1e-6), 40.0, 200.0 * (1 - 1e-6), 200.0, 2000.0])
    spec = draft.make()
    assert len(spec.abscissa) == 6
    ratio = 10 * np.log10(np.real(spec.limits['abort_upper'][0])
                          / np.real(spec.ordinate[0]))
    assert np.allclose(ratio, [6.0, 6.0, 9.0, 9.0, 6.0, 6.0]), \
        'the plateau from its corner, the skirts up to theirs'
    back = SpecificationDraft.from_specification(spec)
    # the helpers are the object's way of holding the step, not
    # breakpoints anyone wrote: opened, they fold away (Brandon,
    # 2026-09-06 — two rows at 40 Hz and two at 200 Hz on the sheet,
    # and a sliver handle left behind by a drag on the plateau)
    assert back.frequencies == [10.0, 40.0, 200.0, 2000.0], 'the helpers fold away'
    assert back.bands[0]['abort'] == [pytest.approx(p) for p in (
        (-6.0, 6.0), (-9.0, 9.0), (-6.0, 6.0))], 'the merged segment keeps the band'
    assert not back.bands[0]['uniform'] and back.bands[0]['symmetric']
    assert back.band_runs(0, 'abort') == [[0], [1], [2]], 'no sliver runs'
    few = back.to_breakpoints()
    assert few.frequencies == [10.0, 40.0, 200.0, 2000.0]
    assert few.bands[0]['abort'] == [pytest.approx(p) for p in (
        (-6.0, 6.0), (-9.0, 9.0), (-6.0, 6.0))]
    assert few.bands[1]['abort'] == [pytest.approx((-6.0, 6.0))] * 3
    # a band that steps where the target runs straight keeps its line
    flat = SpecificationDraft.at_dofs(['1Z+']).with_breakpoint(200.0).constrained(
        uniform=False).with_band('abort', -9.0, 9.0, segments=[1])
    folded = SpecificationDraft.from_specification(flat.make()).to_breakpoints()
    assert folded.frequencies == [20.0, 200.0, 2000.0]
    assert folded.bands[0]['abort'] == [pytest.approx((-6.0, 6.0)),
                                        pytest.approx((-9.0, 9.0))]
    # written into an object at its own lines, the bands follow there too
    project = visualdynamics.Project('t')
    lines = draft.to_lines(1.0)
    project.add('Spec', lines.make())
    project.author_specification('Spec', lines.shifted_band(
        'abort', 'upper', list(range(30, 190)), 3.0), replace=True)
    made = project['Spec']
    ratio = 10 * np.log10(np.real(made.limits['abort_upper'][0])
                          / np.real(made.ordinate[0]))
    f = made.abscissa
    assert np.allclose(ratio[(f >= 40.0) & (f < 200.0)], 12.0)
    assert np.allclose(ratio[f < 40.0], 6.0)


def test_the_bands_survive_the_forms_and_a_new_breakpoint():
    draft = _shaped().constrained(uniform=False).with_band(
        'abort', -9.0, 9.0, segments=[1])
    lines = draft.to_lines(1.0)
    f = np.asarray(lines.frequencies)
    _below, above = lines.band_db('abort', 0)
    assert np.allclose(above[(f >= 40.0) & (f < 200.0)], 9.0)
    assert np.allclose(above[f < 40.0], 6.0)
    assert np.allclose(above[f >= 200.0], 6.0)
    split = draft.with_breakpoint(100.0)
    assert split.bands[0]['abort'] == [(-6.0, 6.0), (-9.0, 9.0), (-9.0, 9.0),
                                       (-6.0, 6.0)], 'both halves inherit'
    merged = split.without_breakpoint(100.0)
    assert merged.bands[0]['abort'] == draft.bands[0]['abort']
    both = SpecificationDraft.across({'A': draft, 'B': _shaped()})
    assert both.bands[0]['abort'] == draft.bands[0]['abort']
    assert both.bands[2]['abort'] == [(-6.0, 6.0)] * 3
    assert both.for_source('A').bands == draft.bands


def test_uniform_off_gives_a_handle_per_linear_section():
    """Brandon, 2026-09-06: with Uniform off, one level per linear
    section of the requirement, each moved on its own."""
    draft = _shaped()
    assert draft.band_runs(0, 'abort') == [[0, 1, 2]], 'uniform: one run'
    loose = draft.constrained(uniform=False)
    assert loose.sections(0) == [[0], [1], [2]], 'breakpoints: every segment'
    assert loose.band_runs(0, 'abort') == [[0], [1], [2]]
    lines = loose.to_lines(1.0)
    f = lines.frequencies
    sections = lines.sections(0)
    assert [(f[s[0]], f[s[-1] + 1]) for s in sections] == [
        (10.0, 40.0), (40.0, 200.0), (200.0, 2000.0)], \
        'lines: the runs of one power law, as folding finds them'
    assert lines.band_runs(0, 'abort') == sections
    moved = lines.shifted_band('abort', 'upper', sections[1], 3.0)
    assert [moved.bands[0]['abort'][s[0]] for s in moved.band_runs(0, 'abort')] \
        == [(-6.0, 6.0), (-9.0, 9.0), (-6.0, 6.0)], 'the plateau alone'
    assert moved.constrained(uniform=True).band_runs(0, 'abort') == [
        list(range(len(f) - 1))]
    # a band already stepping inside a section splits the section
    split = loose.to_lines(1.0).with_band('abort', -9.0, 9.0,
                                          segments=list(range(10)))
    runs = split.band_runs(0, 'abort')
    assert runs[0] == list(range(10)) and runs[1] == list(range(10, 30))


def test_a_target_with_band_corners_still_opens_as_lines_with_few_runs():
    """The drone project's hang (Brandon, 2026-09-06): a non-uniform
    edit, interpolated, made the object carry two helper lines, which
    broke the even spacing, so it reopened as four thousand breakpoints
    each its own section — sixteen thousand handles. The spacing ignores
    helper lines, and the decibels are read to the microdecibel."""
    from visualdynamics.core.author import uniform_spacing

    draft = _shaped().constrained(['1Z+'], uniform=False).with_band(
        'abort', -9.0, 9.0, ['1Z+'], segments=[1])
    made = draft.to_lines(0.5).make()
    assert len(made.abscissa) == 3981 + 2, 'two helper lines at the corners'
    assert uniform_spacing(made.abscissa) == 0.5, 'helpers are not lines of the grid'
    back = SpecificationDraft.from_specification(made)
    assert back.form == 'lines' and back.spacing == 0.5
    assert len(back.sections(0)) == 3
    assert len(back.band_runs(0, 'abort')) == 3
    assert len(back.band_runs(1, 'abort')) == 1, 'the uniform channel, one run'
    assert set(back.bands[0]['abort']) == {(-6.0, 6.0), (-9.0, 9.0)}, \
        'no rounding noise between lines'
    # a ratio that is not on the grid is still read faithfully
    odd = draft.with_band('abort', -6.25, 6.25, ['2Z+'])
    again = SpecificationDraft.from_specification(odd.to_lines(0.5).make())
    assert set(again.bands[1]['abort']) == {(-6.25, 6.25)}
    # the drone target's own noise: its +3 dB warning ratio is stored a
    # float apart from line to line, 1.9952623149688793 and
    # 1.9952623149688797, which read exactly are 2.999999999999999 and
    # 3.0000000000000004 — two bands a rounding error apart
    f = np.arange(20.0, 2000.5, 0.5)
    y = np.full((1, len(f)), 0.002)
    ratio = np.where(np.arange(len(f)) % 2, 1.9952623149688793,
                     1.9952623149688797)
    noisy = visualdynamics.Specification(
        f, y, response_dof=['1Z+'], reference_dof=['1Z+'],
        ordinate_dim='acceleration**2/frequency',
        warning_upper=y * ratio, warning_lower=y / ratio)
    # on a channel that is not uniform — the settle-to-rule cannot hide
    # the noise there, and that was the drone case
    noisy.band_constraints = {'1Z+': {'symmetric': True, 'uniform': False}}
    read = SpecificationDraft.from_specification(noisy)
    assert not read.bands[0]['uniform']
    assert set(read.bands[0]['warning']) == {(-3.0, 3.0)}, \
        'one band, not two a rounding error apart'
    assert len(read.band_runs(0, 'warning')) == 1, 'one handle, not thousands'



def test_a_virtual_points_rows_open_in_the_sheet():
    """A controller's virtual response comes in numbered '1', '2', '3'
    with no direction (the file names the rows nowhere), and the sheet
    refused to open on it: "channel '1' names no direction" (Brandon,
    2026-09-18). What the object accepts, the sheet accepts."""
    freq = np.array([20.0, 80.0, 800.0, 2000.0])
    level = np.array([[1e-3, 4e-3, 4e-3, 1e-3], [2e-3, 8e-3, 8e-3, 2e-3]])
    spec = visualdynamics.Specification(
        freq, level, response_dof=['1', '2'],
        ordinate_dim=['acceleration**2/frequency'] * 2,
        ordinate_unit=['m/s**2'] * 2)
    spec.interpolation = 'log_log'
    draft = SpecificationDraft.from_specification(spec)
    assert draft.channels == ['1', '2']
    back = draft.make()
    assert back.response_dof == ['1', '2']
    assert np.allclose(back.ordinate.real, level)
    with pytest.raises(ValueError, match='every channel needs a name'):
        SpecificationDraft(channels=['1', ''], dims=['acceleration'] * 2,
                           frequencies=[20.0, 2000.0],
                           levels=[[1e-3, 1e-3], [1e-3, 1e-3]])
