"""The caption a report plot of a shock response spectrum, a sine level
and a time history carries, read off its records, and the built-in
templates that ask for it.

An SRS caption states the Q, the spacing, the kind of peak, the filter
the record came through, the sample rate and the points rule in each
polarity the project holds; a sine caption the sweep as specified, the
detector and the distortion check the report made; a time-history
caption the run, its span and stretch, the sample rate and each
channel's range. Pinned here against hand values on synthetic records,
on a project worked up the way the app does it, in the report's payload
and in the editor's pane.
"""

from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.data import ShockSpecification, Srs, TimeHistory
from visualdynamics.core.filters import Filtering
from visualdynamics.core.report import Report
from visualdynamics.core.report_caption import (
    NOT_RECORDED,
    SineCaption,
    SrsCaption,
    TimeCaption,
    caption_kind,
    lines_per_octave,
    sine_caption,
    srs_caption,
    sweep_segments,
    time_caption,
)
from visualdynamics.core.shocks import Shock
from visualdynamics.core.sine import SineTone
from visualdynamics.core.srs import octave_frequencies
from visualdynamics.report import block_caption, render_html
from visualdynamics.report import check_blocks as cb

MINUS = '−'


def _rows(caption):
    return dict(caption.rows())


def _payload(page):
    return json.loads(re.search(r'<script id="data"[^>]*>(.*?)</script>',
                                page, re.DOTALL).group(1))


@pytest.fixture(autouse=True)
def fresh():
    """Every test reads its own checks: nothing kept from another."""
    cb.forget()
    yield
    cb.forget()


# ---- a shock response spectrum ----------------------------------------------

#: twelve lines an octave from 20 Hz over two decades, as `compute_srs`
#: spaces them
F = octave_frequencies(20.0, 2000.0, 12)
REQUIRED = 10.0 * (F / 20.0) ** 0.5


def _pair(*factors, kind='maximax', dofs=None):
    """A requirement at 1Z+ and a measured SRS, one record per factor of
    the requirement, each an event of 1Z+ unless `dofs` names others."""
    spec = ShockSpecification(F, np.atleast_2d(REQUIRED),
                              response_dof=['1Z+'],
                              ordinate_dim=['acceleration'])
    rows = np.vstack([REQUIRED * factor for factor in factors])
    dofs = dofs or ['1Z+'] * len(factors)
    measured = Srs(F, rows, response_dof=dofs,
                   ordinate_dim=['acceleration'] * len(factors),
                   block=[f'shock {k + 1}' for k in range(len(factors))],
                   kind=kind)
    return spec, measured


def test_an_srs_caption_states_each_row_against_hand_values():
    spec, measured = _pair(1.1, 2.5)
    caption = srs_caption(measured, [0, 1], run='Drop 3', sample_rate=10240.0,
                          filters=('low-pass at 2000 Hz, order 4, zero phase',),
                          specification=spec, requirement='Spec')
    assert [label for label, _ in caption.rows()] == list(SrsCaption.ROWS)
    rows = _rows(caption)
    n = len(F)
    assert rows['Channel'] == '1Z+ acceleration', 'one channel, named once'
    assert rows['Run'] == 'Drop 3, shock 1, shock 2'
    assert rows['Q'] == '10 (5 % of critical damping)'
    assert rows['Spacing'] == f'12 lines per octave, 20 Hz to {F[-1]:.4g} Hz'
    assert rows['Type'] == 'maximax'
    assert rows['Filter'] == 'low-pass at 2000 Hz, order 4, zero phase'
    assert rows['Sample rate'] == '10240 Hz'
    # 1.1 times the requirement is +0.83 dB, inside -1.5/+3 dB at every
    # point; 2.5 times is +7.96 dB, outside at every point
    assert rows['Points rule'] == (
        f'Spec, band {MINUS}1.5/+3 dB, rule at least 90 % of the points '
        f'inside the inner band; maximax, shock 1: {n} of {n} points inside '
        f'the inner band, passes; maximax, shock 2: 0 of {n} points inside '
        'the inner band, fails')
    assert caption.text().startswith('Channel: 1Z+ acceleration. Location')


def test_an_srs_without_its_surroundings_says_what_is_not_recorded():
    _spec, measured = _pair(1.0)
    rows = _rows(srs_caption(measured))
    assert rows['Run'] == f'{NOT_RECORDED}, shock 1'
    assert rows['Filter'] == NOT_RECORDED
    assert rows['Sample rate'] == NOT_RECORDED
    assert rows['Points rule'] == 'no requirement drawn'
    assert _rows(srs_caption(measured, filters=()))['Filter'] == \
        'none applied in the project'


def test_the_points_rule_reads_the_bands_from_the_requirements_limits():
    spec, measured = _pair(1.0, 2.5)
    spec.limits = {'warning_lower': REQUIRED / 10 ** (3 / 20),
                   'warning_upper': REQUIRED * 10 ** (3 / 20),
                   'abort_lower': REQUIRED / 10 ** (6 / 20),
                   'abort_upper': REQUIRED * 10 ** (6 / 20)}
    rows = _rows(srs_caption(measured, [0, 1], specification=spec))
    n = len(F)
    # 2.5 times the requirement is +7.96 dB, outside the outer band too
    assert rows['Points rule'] == (
        f'specification, band {MINUS}3/+3 dB within {MINUS}6/+6 dB, rule at '
        'least 90 % of the points inside the inner band, the band read from '
        f"the requirement's limits; maximax, shock 1: {n} of {n} points "
        f'inside the inner band and {n} inside the outer, passes; maximax, '
        f'shock 2: 0 of {n} points inside the inner band and 0 inside the '
        'outer, fails')


def test_one_point_is_one_point_and_one_event_is_named_once():
    from visualdynamics.core.report_caption import PointsCount

    assert PointsCount('maximax', None, 1, 1, None, True, None, 0).text() == \
        'maximax: 1 of 1 point inside the inner band, passes'
    _spec, measured = _pair(1.0, 1.0, dofs=['1Z+', '2Z+'])
    measured.block = ['shock 1', 'shock 1']
    rows = _rows(srs_caption(measured, [0, 1]))
    assert rows['Run'] == f'{NOT_RECORDED}, shock 1'
    assert rows['Channel'] == '1Z+ acceleration; 2Z+ acceleration'
    measured.block = ['', 'shock 2']
    assert _rows(srs_caption(measured, [0, 1]))['Run'] == \
        f'{NOT_RECORDED}, shock 2', 'a record with no label names no event'


def test_each_polarity_given_is_judged_after_the_one_drawn():
    spec, maximax = _pair(1.1)
    _spec, positive = _pair(1.1, kind='positive')
    _spec, negative = _pair(0.5, kind='negative')
    caption = srs_caption(maximax, specification=spec,
                          polarities=[positive, negative])
    assert [point.kind for point in caption.points] == \
        ['maximax', 'positive', 'negative']
    said = _rows(caption)['Points rule']
    assert said.endswith(f'negative, shock 1: 0 of {len(F)} points inside '
                         'the inner band, fails')
    _spec, other = _pair(1.1, 1.1, kind='positive')
    assert len(srs_caption(maximax, specification=spec,
                           polarities=[other]).points) == 1, \
        'another count of records is not the same records'


@pytest.mark.parametrize('values, reason', [
    (np.zeros_like(REQUIRED), 'every point of the record is zero'),
    (np.full_like(REQUIRED, np.nan), "none of the record's points is a number"),
])
def test_a_record_with_no_point_to_judge_says_why_and_never_passes(values,
                                                                   reason):
    spec, measured = _pair(1.0)
    measured.ordinate = np.atleast_2d(values)
    caption = srs_caption(measured, specification=spec)
    (point,) = caption.points
    assert point.reason == reason and point.passed is None
    assert _rows(caption)['Points rule'].endswith(
        f'maximax, shock 1: not judged, {reason}')
    assert 'passes' not in caption.text()


def test_a_point_not_a_number_is_left_out_of_the_count_and_said():
    spec, measured = _pair(1.1)
    values = np.array(measured.ordinate, dtype=float)
    values[0, 5] = np.nan
    measured.ordinate = values
    said = _rows(srs_caption(measured, specification=spec))['Points rule']
    n = len(F)
    assert said.endswith(f'shock 1: {n - 1} of {n - 1} points inside the '
                         'inner band (1 point not a number, not judged), '
                         'passes')


def test_a_spectrum_outside_the_requirements_band_is_not_judged():
    spec, measured = _pair(1.0)
    measured.abscissa = F / 1000.0
    said = _rows(srs_caption(measured, specification=spec))['Points rule']
    assert said.endswith('not judged, no point of the record lies where the '
                         'requirement is written')


def test_a_requirement_at_another_q_or_of_no_channel_drawn_is_refused():
    spec, measured = _pair(1.0)
    measured.q = 20.0
    said = _rows(srs_caption(measured, specification=spec))['Points rule']
    assert 'maximax, shock 1: not judged, the requirement is at Q=10' in said
    spec, measured = _pair(1.0, dofs=['9X+'])
    assert _rows(srs_caption(measured, specification=spec))['Points rule'] == (
        'specification, not judged: the requirement names none of the '
        'channels drawn')


def test_an_srs_caption_refuses_a_requirement_and_an_empty_selection():
    spec, measured = _pair(1.0)
    with pytest.raises(TypeError, match='not one'):
        srs_caption(spec)
    with pytest.raises(IndexError, match='at least one record'):
        srs_caption(measured, [])
    with pytest.raises(IndexError, match='record 3 is not one of the 1'):
        srs_caption(measured, 3)
    with pytest.raises(IndexError, match='record 1 is not one of the 1'):
        srs_caption(measured, 1)
    with pytest.raises(IndexError, match='record -1 is not one'):
        srs_caption(measured, -1)


def test_the_spacing_is_read_off_the_natural_frequencies():
    assert lines_per_octave(octave_frequencies(10.0, 1000.0, 6)) == \
        pytest.approx(6.0)
    assert lines_per_octave(np.linspace(10.0, 1000.0, 50)) is None
    assert lines_per_octave([10.0, 20.0]) == pytest.approx(1.0), \
        'two lines are an octave apart'
    assert lines_per_octave([10.0]) is None
    assert lines_per_octave([0.0, 10.0]) is None
    assert lines_per_octave([0.0, 10.0, 20.0]) is None
    assert lines_per_octave([10.0, 10.0, 10.0]) is None
    assert lines_per_octave([20.0, 10.0]) is None
    assert lines_per_octave([10.0, np.nan]) is None
    _spec, measured = _pair(1.0)
    measured.abscissa = np.linspace(20.0, 2000.0, len(F))
    assert _rows(srs_caption(measured))['Spacing'].startswith(
        'lines not geometrically spaced')


RATE = 10240.0


@pytest.fixture
def shock_project():
    """A two-channel half-sine drop, filtered, its maximax and positive
    spectra computed, a requirement 5 % over the maximax, and a channel
    table that ranges both channels."""
    t = np.arange(int(0.5 * RATE)) / RATE
    pulse = np.where((t > 0.1) & (t < 0.111),
                     400.0 * np.sin(np.pi * (t - 0.1) / 0.011), 0.0)
    history = TimeHistory(t, np.vstack([pulse, 0.5 * pulse]),
                          response_dof=['1Z+', '2Z+'],
                          ordinate_dim=['acceleration'] * 2)
    history.shocks = (Shock(0.05, 0.2),)
    history.filtering = Filtering(high=2000.0, order=4)
    project = visualdynamics.Project()
    project.add('Drop', history)
    filtered = project.filter_data('Drop')
    srs = project.compute_srs(filtered, q=10.0)
    project.compute_srs(filtered, q=10.0, kind='positive')
    project.compute_srs(filtered, q=20.0, kind='negative')
    measured = project[srs]
    project.add('Spec', ShockSpecification(
        measured.abscissa, np.real(measured.ordinate) * 1.05,
        response_dof=['1Z+', '2Z+'], ordinate_dim=['acceleration'] * 2))
    project.add('Channels', ChannelTable({
        'channel': [1, 2], 'node': [1, 2], 'direction': ['Z+', 'Z+'],
        'unit': ['g', 'g'], 'channel_type': ['acceleration'] * 2,
        'comment': ['base', 'top'], 'sensitivity': ['10', '10'],
        'daq_range': ['10', '']}))
    return project, srs, filtered


def test_a_projects_srs_caption_reads_its_run_filter_and_polarities(
        shock_project):
    project, srs, filtered = shock_project
    caption = project.caption_of(srs, 0, specification='Spec')
    rows = _rows(caption)
    assert rows['Channel'] == 'channel 1, 1Z+ acceleration'
    assert rows['Location and axis'] == 'node 1 (base), Z+'
    assert rows['Run'] == filtered
    assert rows['Filter'] == 'low-pass at 2000 Hz, order 4, zero phase'
    assert rows['Sample rate'] == '10240 Hz'
    # the positive spectrum is of the same events at the same Q; the
    # negative one is at another Q and is a different curve
    assert [point.kind for point in caption.points] == ['maximax', 'positive']
    assert rows['Points rule'].startswith('Spec, band ')
    assert 'maximax: ' in rows['Points rule'] and 'passes' in rows['Points rule']


def test_an_imported_srs_names_no_run_and_no_filter(shock_project):
    project, srs, _filtered = shock_project
    project.add('Imported', copy.deepcopy(project[srs]))
    rows = _rows(project.caption_of('Imported', 0))
    assert (rows['Run'], rows['Filter'], rows['Sample rate']) == \
        (NOT_RECORDED,) * 3


def test_an_srs_drawn_unfiltered_says_no_filter_was_applied(shock_project):
    project, _srs, _filtered = shock_project
    raw = project.compute_srs('Drop', q=10.0)
    assert _rows(project.caption_of(raw, 1))['Filter'] == \
        'none applied in the project'


def test_the_page_carries_each_srs_channels_caption(shock_project):
    project, srs, _filtered = shock_project
    block = {'kind': 'plot', 'source': srs, 'specification': 'Spec',
             'record_caption': True, 'caption': 'SRS'}
    [figure] = _payload(render_html(Report('R', [block]), project))['blocks']
    channels = figure['channels']
    assert len(channels) == 2
    for k, channel in enumerate(channels):
        expected = project.caption_of(srs, [k], specification='Spec').rows()
        assert channel['facts'] == [list(row) for row in expected]
        assert '_record' not in channel, 'bookkeeping never reaches the page'
    assert figure['facts'] == channels[0]['facts']
    assert block_caption(block, project) == \
        project.caption_of(srs, [0], specification='Spec')


# ---- a sine level ---------------------------------------------------------------


@pytest.fixture(scope='module')
def sweep_project():
    """A sweep with a third harmonic over 250 to 350 Hz, its levels
    extracted at ten cycles of smoothing."""
    from test_sine_distortion import _sweep

    history, spec = _sweep(harmonic=0.5, band=(250.0, 350.0))
    project = visualdynamics.Project()
    project.add('Sweep', history)
    project.add('Sine Spec', spec)
    history.sine_extraction = history.suggest_sine_extraction()
    from dataclasses import replace

    history.sine_extraction = replace(history.sine_extraction, cycles=10.0)
    (levels,) = project.extract_sine('Sweep', 'Sine Spec')
    return project, levels


def test_a_sine_caption_states_the_sweep_the_detector_and_no_check(
        sweep_project):
    project, levels = sweep_project
    caption = project.caption_of(levels, 0, specification='Sine Spec',
                                 tone='Up')
    assert isinstance(caption, SineCaption)
    assert [label for label, _ in caption.rows()] == list(SineCaption.ROWS)
    rows = _rows(caption)
    assert rows['Channel'] == '101Z+ acceleration'
    assert rows['Run'] == 'Sweep, tone Up found at 0.5 s'
    assert rows['Sweep'] == 'linear, 100 Hz/s, ascending 100 Hz to 500 Hz'
    assert rows['Detector'] == ("Vold-Kalman filter tracking each tone, 10 "
                                "cycles of the tone's frequency per reading")
    assert rows['Distortion'] == 'not checked'


def test_a_sine_caption_states_the_check_the_report_made(sweep_project):
    from visualdynamics.core.sine_distortion import check_distortion

    project, levels = sweep_project
    block = {'kind': 'plot', 'source': levels, 'tone': 'Up',
             'specification': 'Sine Spec', 'record_caption': True,
             'caption': 'Up'}
    report = Report('R', [{'kind': 'plot', 'source': 'Sweep'}, block],
                    features=['check_blocks'])
    page = _payload(render_html(report, project))
    figure = next(b for b in page['blocks'] if b.get('caption') == 'Up')
    check = check_distortion(project['Sweep'], project['Sine Spec'],
                             project[levels], tone='Up')
    assert dict(figure['facts'])['Distortion'] == check.describe('101Z+')
    assert 'over the 32.0 % limit' in check.describe('101Z+')


def test_a_dead_recordings_check_is_stated_as_not_judged(sweep_project):
    project, levels = sweep_project
    history = project['Sweep']
    dead = TimeHistory(history.abscissa, np.zeros_like(history.ordinate),
                       response_dof=list(history.response_dof),
                       ordinate_dim=['acceleration'], ordinate_unit=['m/s**2'])
    other = visualdynamics.Project()
    other.add('Sweep', dead)
    other.add('Sine Spec', project['Sine Spec'])
    other.add('Levels', project[levels])
    other.provenance['Levels'] = dict(project.provenance[levels])
    item = cb.Item(cb.CHECKS['distortion'],
                   (('source', 'Sweep'), ('specification', 'Sine Spec'),
                    ('measured', 'Levels')), {'tone': 'Up'})
    cb.read(item, other)
    said = _rows(other.caption_of('Levels', 0, specification='Sine Spec',
                                  tone='Up'))['Distortion']
    assert said == 'Up at 101Z+: no line could be judged'


def test_a_sine_caption_without_its_requirement_or_run(sweep_project):
    project, levels = sweep_project
    level = project[levels].tone('Up')
    rows = _rows(sine_caption(level))
    assert rows['Channel'] == '101Z+ acceleration', 'the first record'
    assert rows['Run'] == f'{NOT_RECORDED}, tone Up'
    assert (rows['Sweep'], rows['Detector']) == (NOT_RECORDED, NOT_RECORDED)
    assert rows['Distortion'] == 'not checked'
    rows = _rows(sine_caption(project[levels], 0, distortion='no tone held'))
    assert rows['Distortion'] == 'not judged: no tone held'
    with pytest.raises(IndexError, match='at least one record'):
        sine_caption(level, [])
    with pytest.raises(KeyError):
        sine_caption(project[levels], 0, tone='Down')
    with pytest.raises(TypeError, match='not one'):
        sine_caption(project['Sine Spec'])


def test_each_segment_of_a_tone_is_said_with_its_rate_and_direction():
    tone = SineTone('Down', 0.0, [2000.0, 20.0, 40.0], [[1.0]] * 3, [1, 0],
                    [-1.0, 2.0])
    assert sweep_segments(tone) == (
        'logarithmic, 1 oct/min, descending 2000 Hz to 20 Hz',
        'linear, 2 Hz/s, ascending 20 Hz to 40 Hz')
    dwell = SineTone.dwell('Hold', 40.0, [1.0], 30.0)
    assert sweep_segments(dwell) == ('dwell at 40 Hz for 30 s',)


# ---- a time history ----------------------------------------------------------


def test_a_time_caption_states_its_run_rate_and_ranges(shock_project):
    project, _srs, _filtered = shock_project
    caption = project.caption_of('Drop', [0, 1])
    assert isinstance(caption, TimeCaption)
    assert [label for label, _ in caption.rows()] == list(TimeCaption.ROWS)
    rows = _rows(caption)
    assert rows['Run'] == 'Drop, 0 s to 0.4999 s recorded'
    assert rows['Sample rate'] == '10240 Hz'
    # 10 V through 10 mV/g is 1000 g; the second row states no range
    assert rows['Range'] == ('1Z+: ±1000 g full scale; 2Z+: no range in the '
                             'channel table: neither its DAQ Range nor its '
                             'Sensor Range is stated')


def test_a_time_caption_states_the_stretch_its_averaging_reads():
    t = 0.25 + np.arange(4096) / 1024.0
    history = TimeHistory(t, np.zeros((1, t.size)), response_dof=['1Z+'],
                          ordinate_dim=['acceleration'])
    history.averaging = Averaging(frame_length=1024, overlap=0.0,
                                  window='hann', frames=2, start=1.0)
    rows = _rows(time_caption(history, run='Run 4'))
    assert rows['Run'] == 'Run 4, 0.25 s to 4.249 s recorded, 1.25 s to 3.25 s analyzed'
    assert rows['Range'] == NOT_RECORDED


def test_a_time_caption_of_a_dead_or_broken_channel_is_still_read():
    """A caption reads the record's clock and the table, not its values:
    an all-zero channel and a sample that is not a number change none of
    it; a clock that is not even has no one sample rate."""
    t = np.arange(8) / 4.0
    values = np.zeros((2, 8))
    values[1, 3] = np.nan
    history = TimeHistory(t, values, response_dof=['1Z+', '2Z+'],
                          ordinate_dim=['acceleration'] * 2)
    rows = _rows(time_caption(history, [0, 1], run='R'))
    assert rows['Sample rate'] == '4 Hz'
    assert rows['Run'] == 'R, 0 s to 1.75 s recorded'
    uneven = TimeHistory(t ** 2, values, response_dof=['1Z+', '2Z+'],
                         ordinate_dim=['acceleration'] * 2)
    assert _rows(time_caption(uneven))['Sample rate'] == NOT_RECORDED
    two = TimeHistory(t[:2], values[:, :2], response_dof=['1Z+', '2Z+'],
                      ordinate_dim=['acceleration'] * 2)
    assert _rows(time_caption(two))['Sample rate'] == '4 Hz'
    one = TimeHistory(t[:1], values[:, :1], response_dof=['1Z+', '2Z+'],
                      ordinate_dim=['acceleration'] * 2)
    rows = _rows(time_caption(one))
    assert rows['Sample rate'] == NOT_RECORDED, 'one sample has no rate'
    assert rows['Run'] == f'{NOT_RECORDED}, 0 s to 0 s recorded'
    with pytest.raises(IndexError, match='at least one record'):
        time_caption(history, [])


def test_which_caption_a_record_carries():
    spec, measured = _pair(1.0)
    assert caption_kind(measured) == 'srs'
    assert caption_kind(spec) is None, 'a requirement is not a measurement'
    assert caption_kind(TimeHistory(np.arange(4.0), np.zeros((1, 4)),
                                    response_dof=['1Z+'])) == 'time'
    assert caption_kind('Spec') is None
    with pytest.raises(TypeError, match='reads a measured time history'):
        time_caption(measured)


# ---- the built-in templates ---------------------------------------------------


def _plot_blocks(report):
    return [block for block in report.blocks if block.get('kind') == 'plot']


def test_a_stage_of_spectra_carries_the_caption_of_the_records_it_draws():
    rng = np.random.default_rng(5)
    t = np.arange(4096) / 1024.0
    history = TimeHistory(t, rng.standard_normal((3, t.size)),
                          response_dof=['1Z+', '2Z+', '3Z+'],
                          ordinate_dim=['acceleration'] * 3)
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann', frames=7)
    project = visualdynamics.Project()
    project.add('Run', history)
    psds = project.compute_psds('Run')
    block = {'kind': 'plot', 'source': psds, 'mode': 'stage',
             'record_caption': True, 'caption': 'Spectra'}
    [figure] = _payload(render_html(Report('R', [block]), project))['blocks']
    assert figure['kind'] == 'stage' and '_records' not in figure
    assert figure['facts'] == [list(row) for row in
                               project.caption_of(psds, [0, 1, 2]).rows()]
    # seven half-overlapped frames of 1024 samples span all 4096
    assert dict(figure['facts'])['Run'] == 'Run, 0 s to 4 s'


def test_the_modal_and_transient_templates_caption_their_spectra():
    from visualdynamics.core.report import modal_template, transient_template

    stages = [block for block in _plot_blocks(modal_template({}))
              if block.get('mode') == 'stage'
              and block.get('select') in ('dim:force', 'dim:acceleration')]
    spectra = [block for block in stages if 'spectral' in block['caption']]
    assert len(spectra) == 2
    assert all(block.get('record_caption') for block in spectra)
    transient = [block for block in _plot_blocks(transient_template({}))
                 if block['source'] == '@basis:Psd']
    assert transient and all(block.get('record_caption')
                             for block in transient)


def test_the_shock_and_sine_templates_caption_their_spectra_and_levels():
    from visualdynamics.core.report import shock_template, sine_template

    shock = _plot_blocks(shock_template({}))
    captioned = [block for block in shock if block.get('record_caption')]
    assert [block['source'] for block in captioned] == ['@basis:Srs']
    assert all(block['source'] != '@basis:ShockSpecification'
               for block in captioned), 'a requirement carries none'
    sine = _plot_blocks(sine_template({}))
    levels = [block for block in sine
              if block['source'] == '@basis:SineLevelSet']
    assert levels and all(block.get('record_caption') for block in levels)


def test_the_random_template_captions_every_control_comparison():
    from test_report_caption import fixture_path

    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    report = project[project.generate_report('random')]
    comparisons = [block for block in _plot_blocks(report)
                   if block.get('specification')]
    assert comparisons and all(block.get('record_caption')
                               for block in comparisons)
    histories = [block for block in _plot_blocks(report)
                 if block['source'] in ('@basis:TimeHistory', 'Time History')]
    assert not any(block.get('record_caption') for block in histories), \
        'the templates leave the time histories as they were'
    page = _payload(render_html(report, project,
                                object_groups=project.object_groups))
    facts = [block for block in page['blocks'] if block.get('facts')
             or (block['kind'] == 'grid'
                 and any(cell.get('facts') for row in block['rows']
                         for column in row['cells'] for cell in column))]
    assert facts, 'a report generated from the template carries its captions'
