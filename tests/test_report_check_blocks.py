"""Check blocks (report feature `check_blocks`).

Which checks a report makes, from what it shows and from blocks of
their own; each check's verdict, words, table and figures against the
core function it calls; what a dead channel, a sample that is not a
number and an input the reading refuses say; the readings made once and
kept, and made on the window's worker; where the blocks fall on the page
and under the skeleton; saving with the report and its template; and
the promise the feature rests on: off, the page is the page it was.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import (
    Psd,
    ShockSpecification,
    Specification,
    Srs,
    TimeHistory,
)
from visualdynamics.core.report import Report
from visualdynamics.report import check_blocks as cb
from visualdynamics.report import features as rf
from visualdynamics.report import render_html

SI = visualdynamics.SI


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _page(report, objects, **kwargs):
    return _payload(render_html(report, objects, unit_system=SI,
                                **kwargs))['blocks']


def _heads(blocks):
    """(title, verdict word, html) of each check's head on the page."""
    out = []
    for block in blocks:
        html = block.get('html', '')
        if html.startswith('<h4 class="check">'):
            title = html.split('>', 1)[1].split('</h4>')[0]
            word = html.split('<strong>')[1].split('.</strong>')[0]
            out.append((title, word, html))
    return out


def _shown(report, objects, key, groups=None):
    """What the first item of check `key` shows."""
    item = next(item for item in cb.items(report, objects, groups)
                if item.check.key == key)
    shown = cb.show(item, objects, SI)
    width = len(shown.headers)
    assert all(len(row) == width for row in shown.rows)
    assert len(shown.marks) == len(shown.rows)
    assert all(len(row) == width for row in shown.marks)
    return shown


@pytest.fixture(autouse=True)
def fresh():
    """Every test reads its own checks: nothing kept from another."""
    cb.forget()
    yield
    cb.forget()


def _noise(channels=2, seconds=8.0, rate=2048.0, seed=4):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    return TimeHistory(np.arange(n) / rate,
                       rng.standard_normal((channels, n)),
                       response_dof=[f'{k + 1}Z+' for k in range(channels)],
                       ordinate_dim=['acceleration'] * channels)


def _random_report(history, features=('check_blocks',)):
    """A random test's report on `history`: a time plot and a PSD
    specification it is compared to, as the random template binds them."""
    spec = Specification(np.array([20.0, 1000.0]), np.ones((1, 2)),
                         response_dof=['1Z+'], ordinate_dim=['acceleration'])
    objects = {'History': history, 'Spec': spec}
    report = Report('R', [{'kind': 'text', 'text': '## Measured'},
                          {'kind': 'plot', 'source': 'History'},
                          {'kind': 'plot', 'source': 'Spec'},
                          {'kind': 'text', 'text': '## Conclusions\n\nEnd.'}],
                    features=list(features))
    return report, objects


# ---- the feature --------------------------------------------------------------


def test_the_feature_is_built_and_off_by_default():
    feature = rf.FEATURES[cb.FEATURE]
    assert feature.available and not feature.default
    assert feature.requires == ()
    assert cb.FEATURE in rf.PRESETS['test_report'].features
    assert cb.FEATURE in rf.configure(Report(), preset='test_report').enabled


def test_every_check_has_a_reader_and_a_view():
    assert set(cb._READERS) == set(cb._SHOWERS) == set(cb.CHECKS)
    assert cb.AUTOMATIC == ('time_statistics', 'shock_validity',
                            'srs_points', 'distortion', 'sweep_rate')
    for found in cb.CHECKS.values():
        fields = dict(found.fields)
        assert set(found.needs) <= set(fields)
        assert set(found.required) <= set(found.settings)
        assert found.description.endswith('.')
        if found.automatic:
            assert all(fields[name] for name in found.needs), \
                'a check made from what the report shows binds by role'


def test_off_a_report_with_a_check_of_its_own_draws_the_page_it_was():
    history = _noise()
    report, objects = _random_report(history, features=())
    before = render_html(report, objects)
    cb.add(report, 'time_statistics', {'source': 'History'})
    assert render_html(report, objects) == before, \
        'a check block is drawn only while the feature is on'
    edited = render_html(report, objects, edit=True)
    report.blocks.pop()
    assert edited == render_html(report, objects, edit=True)


def test_on_with_nothing_to_check_the_page_is_unchanged_but_its_style():
    objects = {'Spec': Specification(np.array([20.0, 1000.0]),
                                     np.ones((1, 2)), response_dof=['1Z+'],
                                     ordinate_dim=['acceleration'])}
    report = Report('R', [{'kind': 'plot', 'source': 'Spec'}])
    off = _page(report, objects)
    report.features = ['check_blocks']
    assert cb.items(report, objects) == []
    assert _page(report, objects) == off


# ---- which checks a report makes -----------------------------------------------


def test_a_random_report_carries_the_time_statistics_of_its_record():
    report, objects = _random_report(_noise())
    found = cb.items(report, objects)
    assert [(item.check.key, dict(item.bound)) for item in found] == [
        ('time_statistics', {'source': 'History'})]


def test_a_check_is_made_only_from_what_the_report_shows():
    report, objects = _random_report(_noise())
    objects['Other'] = _noise(seed=9)
    report.blocks[1]['source'] = 'Other'
    # the basis record is not drawn: the report shows another one
    report.blocks = [block for block in report.blocks
                     if block.get('source') != 'Other']
    assert cb.items(report, objects) == []


def test_time_statistics_are_not_made_for_a_shock_or_a_sine_report():
    from visualdynamics.core.sine import SineSweepSpecification, SineTone

    history = _noise()
    srs = Srs(np.geomspace(20, 2000, 10), np.ones((1, 10)),
              response_dof=['1Z+'], ordinate_dim=['acceleration'])
    report = Report('R', [{'kind': 'plot', 'source': 'History'},
                          {'kind': 'plot', 'source': 'SRS'}],
                    features=['check_blocks'])
    objects = {'History': history, 'SRS': srs}
    keys = [item.check.key for item in cb.items(report, objects)]
    assert keys == ['shock_validity'], 'a shock record is not Gaussian'
    tone = SineTone('Up', 0.0, [20.0, 200.0], [[1.0]] * 2, [0], [1.0])
    objects['SRS'] = SineSweepSpecification([tone], ['1Z+'],
                                            ordinate_unit='m/s**2')
    assert cb.items(report, objects) == [], 'nor is a swept tone'


def test_a_check_of_its_own_replaces_the_one_made_from_the_report():
    report, objects = _random_report(_noise())
    cb.add(report, 'time_statistics', {'source': 'History'},
           {'kurtosis_band': [2.9, 3.1]})
    (item,) = cb.items(report, objects)
    assert item.index == len(report.blocks) - 1
    assert item.settings == {'kurtosis_band': [2.9, 3.1]}


def test_a_check_whose_object_is_gone_is_not_made():
    report, objects = _random_report(_noise())
    cb.add(report, 'pulse', {'source': 'Missing'},
           {'shape': 'half-sine', 'peak': 490.0, 'duration': 0.011})
    assert [item.check.key for item in cb.items(report, objects)] == [
        'time_statistics']


# ---- a check of its own -----------------------------------------------------------


@pytest.mark.parametrize(('key', 'bindings', 'settings', 'message'), [
    ('notes', {}, {}, "unknown check 'notes'"),
    ('pulse', {'geometry': 'G'}, {}, "not 'geometry'"),
    ('pulse', {}, {'shape': 'half-sine', 'peak': 1.0, 'duration': 0.01,
                   'colour': 'red'}, "not 'colour'"),
    ('pulse', {}, {'shape': 'half-sine'}, 'needs peak, duration'),
    ('signature', {}, {}, 'reads source, measured; name what it reads'),
    ('sweep_rate', {}, {'modes': [(1.0, np.float32)]}, 'saved with the '),
])
def test_a_check_block_that_cannot_be_made_is_refused(key, bindings,
                                                      settings, message):
    report = Report('R', [{'kind': 'text', 'text': 'x'}])
    with pytest.raises(ValueError, match=message):
        cb.add(report, key, bindings, settings)
    assert report.blocks == [{'kind': 'text', 'text': 'x'}]


def test_a_check_is_named_by_its_key():
    with pytest.raises(TypeError, match='key, a str'):
        cb.check(3)


def test_the_project_verb_writes_the_block_and_journals_the_call():
    project = visualdynamics.Project('P')
    name = project.generate_report('empty')
    index = project.report_check_block(
        name, 'sweep_rate', source='Spec', modes=[[22.0, 40.0]], margin=1.5)
    assert project[name].blocks[index] == {
        'kind': 'text', 'text': '', 'check': 'sweep_rate', 'source': 'Spec',
        'shapes': '@basis:ShapeSet',
        'settings': {'modes': [[22.0, 40.0]], 'margin': 1.5}}
    assert project.journal[-1] == (
        f"project.report_check_block({name!r}, 'sweep_rate', "
        "source='Spec', modes=[[22.0, 40.0]], margin=1.5)")


def test_a_check_block_is_saved_with_the_report_and_its_template(tmp_path):
    from visualdynamics import io
    from visualdynamics.io import report_template

    report = Report('R', features=[cb.FEATURE])
    cb.add(report, 'pulse', None, {'shape': 'half-sine', 'peak': 490.0,
                                   'duration': 0.011, 'band': 0.15})
    io.save(report, str(tmp_path / 'r.vdyn'))
    assert io.load(str(tmp_path / 'r.vdyn')).blocks == report.blocks
    report_template.save(report, tmp_path / 'house.vdreport')
    back = report_template.load(tmp_path / 'house.vdreport')
    assert back.blocks == report.blocks and back.features == [cb.FEATURE]


def test_a_block_with_settings_the_check_does_not_take_is_not_judged():
    """A block written by hand or by a later version: said, not read."""
    report, objects = _random_report(_noise())
    report.blocks.append({'kind': 'text', 'text': '',
                          'check': 'time_statistics', 'source': 'History',
                          'settings': {'colour': 'red'}})
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.NOT_JUDGED
    assert "'colour' are not the check's" in shown.words
    report.blocks[-1] = {'kind': 'text', 'text': '', 'check': 'pulse',
                         'source': 'History', 'settings': {}}
    shown = _shown(report, objects, 'pulse')
    assert shown.status == cb.NOT_JUDGED
    assert 'does not say shape, peak, duration' in shown.words


def test_a_setting_the_reading_cannot_take_is_its_refusal():
    report, objects = _random_report(_noise())
    cb.add(report, 'time_statistics', {'source': 'History'},
           {'kurtosis_band': [4.0, 2.0]})
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.NOT_JUDGED
    assert 'runs from its low edge to its high edge' in shown.words


# ---- the verdict ------------------------------------------------------------------


@pytest.mark.parametrize(('verdicts', 'status'), [
    ([True, True], cb.PASSED),
    ([True, False], cb.FLAGGED),
    ([None, False], cb.FLAGGED),
    ([True, None], cb.PARTLY),
    ([None, None], cb.NOT_JUDGED),
    ([], cb.NOT_JUDGED),
])
def test_nothing_judged_is_never_a_pass(verdicts, status):
    assert cb.status_of(verdicts) == status


# ---- time statistics ----------------------------------------------------------------


def test_time_statistics_agree_with_the_reading_channel_by_channel():
    from visualdynamics.core.time_statistics import record_statistics

    history = _noise(channels=3)
    history.ordinate[2] = np.clip(history.ordinate[2], -1.5, 1.5)
    report, objects = _random_report(history)
    shown = _shown(report, objects, 'time_statistics')
    assert shown.headers == ['Channel', 'Mean', 'RMS', 'Skewness', 'Kurtosis',
                             'Crest factor', 'Frames out', 'Verdict']
    assert len(shown.rows) == 3
    for row in range(3):
        stats = record_statistics(history, row)
        cells = shown.rows[row]
        assert cells[0] == history.record_label(row)
        assert cells[3] == f'{stats.skewness:.3f}'
        assert cells[4] == f'{stats.kurtosis:.3f}'
        assert cells[5] == f'{stats.crest:.3f}'
        assert cells[-1] == stats.describe()
        assert (shown.marks[row][-1] == 'over') == bool(stats.flagged)
    assert shown.status == cb.FLAGGED, 'the clipped channel'
    assert shown.words.startswith('1 of 3 channels read outside a band: 3Z+')
    assert 'crest factor' in shown.words


def test_clean_noise_passes_and_shows_its_worst_channel():
    report, objects = _random_report(_noise())
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.PASSED
    assert shown.words == ('Every statistic is inside its band on all 2 '
                           'channels, read over the record.')
    assert 'kurtosis 2 to 4 over the record' in shown.criterion
    histogram, frames = shown.figures
    assert histogram['caption'].startswith(('1Z+: the histogram',
                                            '2Z+: the histogram'))
    assert frames['ylabel'] == 'Pearson kurtosis'
    assert 'does not vary' not in frames['caption']
    assert '% (Gaussian 0.27 %), 4σ' in histogram['caption']


def test_one_clean_channel_is_one_channel_and_its_frames_are_timed():
    from visualdynamics.core.time_statistics import record_statistics

    history = _noise(channels=1)
    report, objects = _random_report(history)
    shown = _shown(report, objects, 'time_statistics')
    assert shown.words == ('Every statistic is inside its band on 1 '
                           'channel, read over the record.')
    stats = record_statistics(history, 0)
    seconds = stats.frame_length / stats.sample_rate
    assert f'each {seconds:.3g} s frame' in shown.figures[1]['caption']
    assert shown.figures[0]['curves'][0]['color'] == 0


def test_with_none_flagged_the_channel_furthest_from_gaussian_is_drawn():
    from visualdynamics.core.time_statistics import record_statistics

    # a tone under the first channel's noise lowers its kurtosis, and the
    # second's level swelling and falling raises its own, both inside
    # their bands; the first is further from 3 and the second higher
    history = _noise(channels=2)
    t = history.abscissa
    history.ordinate[0] += 1.1 * np.sqrt(2.0) * np.sin(2 * np.pi * 97.3 * t)
    history.ordinate[1] *= 1.0 + 0.2 * np.sin(np.pi * t)
    report, objects = _random_report(history)
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.PASSED
    low, high = (record_statistics(history, row).kurtosis for row in (0, 1))
    assert abs(low - 3.0) > abs(high - 3.0) and high > 3.0
    assert shown.figures[0]['caption'].startswith('1Z+:')


def test_the_rms_is_shown_in_the_display_units():
    from visualdynamics.units import DEFAULT_SYSTEM as us

    history = _noise(channels=1)
    report, objects = _random_report(history)
    item = cb.items(report, objects)[0]
    rms = float(np.sqrt(np.mean(history.ordinate[0] ** 2)))
    row = cb.show(item, objects, us).rows[0]
    assert row[2] == (f'{float(us.from_si(rms, "acceleration")):.4g} '
                      f'{us.label_text("acceleration")}')


def test_the_bell_is_the_gaussian_of_the_records_spread():
    history = _noise(channels=1)
    history.ordinate[0] = history.ordinate[0] * 3.0
    report, objects = _random_report(history)
    figure = _shown(report, objects, 'time_statistics').figures[0]
    grid = np.array(figure['x'], dtype=float)
    bell = np.array(figure['curves'][1]['y'], dtype=float)
    sigma = float(np.std(history.ordinate[0]))
    mean = float(np.mean(history.ordinate[0]))
    at = float(np.interp(mean + sigma, grid, bell))
    assert at == pytest.approx(bell.max() * np.exp(-0.5), rel=1e-3)


def test_the_histogram_is_a_density_beside_its_gaussian():
    history = _noise(channels=1)
    history.ordinate[0] = history.ordinate[0] * 3.0 + 0.5
    report, objects = _random_report(history)
    figure = _shown(report, objects, 'time_statistics').figures[0]
    record, gauss = figure['curves']
    x = np.array(record['x'], dtype=float)
    y = np.array(record['y'], dtype=float)
    # the steps' area: each bar is a pair of points at its two edges
    area = np.sum((x[2:-1:2] - x[1:-1:2]) * y[1:-1:2])
    assert area == pytest.approx(1.0, abs=1e-6)
    grid = np.array(figure['x'], dtype=float)
    peak = grid[int(np.argmax(gauss['y']))]
    assert peak == pytest.approx(0.5, abs=0.05), 'the mean of the record'
    assert max(gauss['y']) == pytest.approx(1.0 / (3.0 * np.sqrt(2 * np.pi)),
                                            rel=0.02)
    assert gauss['gray'] and gauss['dash'], 'the criterion stands back'


def test_the_histogram_is_drawn_in_the_display_units():
    history = _noise(channels=1)
    report, objects = _random_report(history)
    item = cb.items(report, objects)[0]
    us = visualdynamics.units.SYSTEMS['in-slinch-lbf-s (g)'] \
        if hasattr(visualdynamics.units, 'SYSTEMS') else None
    if us is None:
        from visualdynamics.units import DEFAULT_SYSTEM as us
    si = cb.show(item, objects, SI).figures[0]
    shown = cb.show(item, objects, us).figures[0]
    factor = float(us.from_si(1.0, 'acceleration'))
    assert np.allclose(np.array(shown['x'], dtype=float),
                       np.array(si['x'], dtype=float) * factor, rtol=1e-5)
    assert 'g' in shown['xlabel']


def test_a_dead_channel_is_not_judged_and_named():
    history = _noise(channels=2)
    history.ordinate[1] = 0.0
    report, objects = _random_report(history)
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.PARTLY
    assert shown.rows[1][-1] == f'not judged: {cb.DOES_NOT_VARY}'
    assert shown.rows[1][1:-1] == [cb.DASH] * 6
    assert f'Not judged: 2Z+, {cb.DOES_NOT_VARY}.' in shown.words


def test_a_constant_channel_is_not_judged_either():
    history = _noise(channels=1)
    history.ordinate[0] = 2.0
    report, objects = _random_report(history)
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.NOT_JUDGED
    assert shown.words.startswith('No channel could be read.')
    assert shown.figures == []


def test_a_sample_that_is_not_a_number_is_flagged_by_name():
    history = _noise(channels=1)
    history.ordinate[0, 100] = np.nan
    report, objects = _random_report(history)
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.FLAGGED
    assert 'samples not finite' in shown.words


def test_white_noise_at_the_defaults_is_rarely_flagged():
    """The bands are the reading's; checked here against simulated white
    noise, read through the block, as a reader meets them."""
    flagged = 0
    for seed in range(20):
        report, objects = _random_report(_noise(channels=1, seed=seed))
        flagged += _shown(report, objects,
                          'time_statistics').status != cb.PASSED
        cb.forget()
    assert flagged <= 2


def test_figures_are_drawn_for_each_channel_outside_a_band_up_to_a_few():
    history = _noise(channels=6)
    history.ordinate[:] = np.clip(history.ordinate, -1.2, 1.2)
    report, objects = _random_report(history)
    shown = _shown(report, objects, 'time_statistics')
    assert len(shown.figures) == 2 * cb.MOST_FIGURES
    assert shown.figures[-1]['caption'].endswith(
        'The first 4 of 6 channels outside a band are drawn; the table '
        'holds every one.')


# ---- shock validity --------------------------------------------------------------------


def _shock_record(*rows):
    from test_shock_check import history

    return history(*rows)


def _shock_report(record):
    srs = Srs(np.geomspace(20, 2000, 10), np.ones((1, 10)),
              response_dof=['1Z+'], ordinate_dim=['acceleration'])
    report = Report('R', [{'kind': 'plot', 'source': 'History'},
                          {'kind': 'plot', 'source': 'SRS'}],
                    features=['check_blocks'])
    return report, {'History': record, 'SRS': srs}


def test_a_compensated_shock_returns_to_rest_and_an_offset_does_not():
    from test_shock_check import compensated, times

    from visualdynamics.core.shock_check import check_shock

    t = times(0.4)
    record = _shock_record(compensated(t, 0.2), compensated(t, 0.2) + 2.0)
    report, objects = _shock_report(record)
    shown = _shown(report, objects, 'shock_validity')
    checks = check_shock(record)
    assert shown.status == cb.FLAGGED
    assert shown.rows[0][-1] == 'returns to rest'
    assert shown.rows[1][-1] == ('does not return to rest: still changing '
                                 'at the end')
    assert shown.rows[1][3] == \
        f'{100.0 * checks[1].end_velocity_share:.1f} % of its peak'
    assert shown.marks[1][-1] == 'over' and shown.marks[0][-1] is None
    assert shown.words.startswith('1 of 2 do not return to rest.')
    assert 'within 10 % of its peak' in shown.criterion
    (figure,) = shown.figures
    assert [curve['label'] for curve in figure['curves']] == ['2Z+', '1Z+'], \
        'the channel that does not return first'
    assert figure['x'][0] == 0.0
    assert figure['x'][-1] == pytest.approx(t[-1], rel=1e-6)


def test_a_shock_channel_that_never_moves_is_not_judged():
    from test_shock_check import compensated, times

    t = times(0.4)
    record = _shock_record(compensated(t, 0.2), np.zeros_like(t))
    report, objects = _shock_report(record)
    shown = _shown(report, objects, 'shock_validity')
    assert shown.status == cb.PARTLY
    assert shown.rows[1][-1] == f'not judged: {cb.DOES_NOT_MOVE}'
    assert shown.rows[1][:2] == ['2Z+', '1']
    assert shown.words.endswith('Not judged: 2Z+, which does not move.')


def test_a_shock_record_with_a_sample_not_a_number_is_refused_by_name():
    from test_shock_check import compensated, times

    t = times(0.4)
    row = compensated(t, 0.2)
    row[50] = np.nan
    report, objects = _shock_report(_shock_record(row))
    shown = _shown(report, objects, 'shock_validity')
    assert shown.status == cb.NOT_JUDGED
    assert shown.words.startswith('1Z+ holds a sample that is not a finite '
                                  'number')
    assert shown.rows == [] and shown.figures == []


# ---- the SRS points rule ---------------------------------------------------------------


def _srs_pair(factor=1.0, q=None):
    f = np.geomspace(20.0, 2000.0, 40)
    required = 10.0 * (f / 20.0) ** 0.5
    spec = ShockSpecification(f, np.atleast_2d(required),
                              response_dof=['1Z+'],
                              ordinate_dim=['acceleration'])
    measured = Srs(f, np.vstack([required * factor, required * 2.5]),
                   response_dof=['1Z+', '1Z+'],
                   ordinate_dim=['acceleration'] * 2)
    if q is not None:
        spec.q, measured.q = 10.0, q
    report = Report('R', [{'kind': 'plot', 'source': 'SRS',
                           'specification': 'Spec'}],
                    features=['check_blocks'])
    return report, {'Spec': spec, 'SRS': measured}


def test_the_points_rule_rows_are_the_readings_own():
    from visualdynamics.core.compliance import srs_tolerance_all

    report, objects = _srs_pair(1.1)
    shown = _shown(report, objects, 'srs_points')
    rows = srs_tolerance_all(objects['Spec'], objects['SRS'])
    assert len(shown.rows) == len(rows) == 2
    first, _second = rows
    assert shown.rows[0][2] == (f'{first[2]["inside_count"]} of '
                                f'{first[2]["points"]} (100.0 %)')
    assert shown.rows[0][-1] == 'passes' and shown.rows[1][-1] == 'fails'
    assert shown.rows[0][1] == cb.DASH, 'a spectrum of no event'
    assert shown.status == cb.FLAGGED
    assert shown.criterion.startswith('Band −1.5/+3 dB about the '
                                      'requirement; rule: at least 90 % ')
    (figure,) = shown.figures
    assert figure['logx'] is True
    db = [v for v in figure['curves'][0]['y'] if v is not None]
    assert db == pytest.approx([20 * np.log10(1.1)] * len(db), abs=1e-6)


def test_a_pair_at_another_q_is_not_judged_with_the_reason():
    report, objects = _srs_pair(1.0, q=20.0)
    shown = _shown(report, objects, 'srs_points')
    assert shown.status == cb.NOT_JUDGED
    assert shown.rows[0][-1].startswith('not judged: the requirement is at '
                                        'Q=10')
    assert shown.rows[0][1] == cb.DASH
    assert 'Not judged: 1Z+, the requirement is at Q=10' in shown.words


def test_a_spectrum_the_requirement_does_not_name_is_not_judged():
    report, objects = _srs_pair()
    objects['SRS'].response_dof = np.array(['9X+', '9X+'])
    shown = _shown(report, objects, 'srs_points')
    assert shown.status == cb.NOT_JUDGED
    assert shown.words == ('No record of the spectrum is at a channel the '
                           'requirement names.')


# ---- the classical pulse ------------------------------------------------------------------


def _pulse_report(nominal_ms=11.0, peak=50.0):
    from test_classical_pulse import G, pulse, record

    t, opens = record()
    made = pulse().at(t - opens)
    history = TimeHistory(t, made[None], response_dof=['1Z+'],
                          ordinate_dim=['acceleration'])
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'pulse', {'source': 'History'},
           {'shape': 'half-sine', 'peak': peak * G,
            'duration': nominal_ms / 1000.0})
    return report, {'History': history}


def test_a_pulse_on_its_nominal_passes_with_the_readouts_numbers():
    from visualdynamics.core.classical_pulse import ClassicalPulse, check_pulse
    from visualdynamics.plot.classical_pulse import readout

    report, objects = _pulse_report()
    shown = _shown(report, objects, 'pulse')
    check = check_pulse(objects['History'],
                        ClassicalPulse('half-sine', 50.0 * 9.80665, 0.011))
    assert shown.status == cb.PASSED
    assert shown.words == check.describe()
    assert [row[1] for row in shown.rows] == list(readout(check, SI).values())
    assert shown.criterion == (
        'Nominal: 490.3 m/s², 11 ms half-sine. Envelope '
        f'±{100 * check.band:g} % of the peak, duration '
        f'±{100 * check.duration_tolerance:g} %, velocity change '
        f'±{100 * check.velocity_tolerance:g} %.')
    (figure,) = shown.figures
    assert [curve['label'] for curve in figure['curves']] == [
        'envelope, lower edge', 'envelope, upper edge', 'record']


def test_a_pulse_against_the_wrong_nominal_is_flagged():
    report, objects = _pulse_report(nominal_ms=15.0)
    shown = _shown(report, objects, 'pulse')
    assert shown.status == cb.FLAGGED
    assert shown.words.startswith('Flagged: ')
    assert any(mark[1] == 'over' for mark in shown.marks)


def test_a_pulse_read_from_a_record_with_no_acceleration_is_refused():
    report, objects = _pulse_report()
    objects['History'].ordinate_dim = np.array(['force'])
    shown = _shown(report, objects, 'pulse')
    assert shown.status == cb.NOT_JUDGED
    assert 'is read from an acceleration' in shown.words


# ---- sine distortion -------------------------------------------------------------------------


@pytest.fixture(scope='module')
def sweep():
    from test_sine_distortion import _sweep

    from visualdynamics.core.sine import extract_sine

    history, spec = _sweep(harmonic=0.5, band=(250.0, 350.0))
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    return history, spec, levels


def _sine_report(history, spec, levels):
    report = Report('R', [{'kind': 'plot', 'source': 'History'},
                          {'kind': 'plot', 'source': 'Levels',
                           'specification': 'Spec'}],
                    features=['check_blocks'])
    return report, {'History': history, 'Spec': spec, 'Levels': levels}


def test_the_distortion_check_is_made_per_tone_and_agrees_with_its_reading(
        sweep):
    from visualdynamics.core.sine_distortion import check_distortion

    report, objects = _sine_report(*sweep)
    (item,) = cb.items(report, objects)
    assert item.settings == {'tone': 'Up'}
    assert cb.title_of(item) == 'Sine distortion, Up'
    shown = cb.show(item, objects, SI)
    check = check_distortion(*sweep, tone='Up')
    assert shown.status == cb.FLAGGED
    assert shown.words.startswith(check.describe())
    _frequency, value = check.worst('101Z+')
    assert shown.rows[0][2].startswith(f'{100.0 * value:.1f} %')
    assert shown.rows[0][4] == (f'{int(check.over[0].sum())} of '
                                f'{int(check.judged[0].sum())}')
    assert 'at most 32.0 %' in shown.criterion
    assert "(the unfiltered RMS within 5 % of the tone's)" in shown.criterion
    (figure,) = shown.figures
    assert figure['logx'] and figure['curves'][-1]['label'] == \
        'limit, 32.0 %'
    limit = np.array(figure['curves'][-1]['y'], dtype=float)
    assert np.allclose(limit, 100.0 * check.limit, rtol=1e-6)
    drawn = np.array([np.nan if v is None else v
                      for v in figure['curves'][0]['y']])
    want = np.where(check.judged[0], 100.0 * check.distortion[0], np.nan)
    assert np.allclose(drawn, want, rtol=1e-5, equal_nan=True)


def test_a_dead_sine_channel_is_not_judged_and_does_not_pass(sweep):
    history, spec, levels = sweep
    dead = TimeHistory(history.abscissa, np.zeros_like(history.ordinate),
                       response_dof=list(history.response_dof),
                       ordinate_dim=['acceleration'],
                       ordinate_unit=['m/s**2'])
    report, objects = _sine_report(dead, spec, levels)
    shown = _shown(report, objects, 'distortion')
    assert shown.status == cb.NOT_JUDGED
    assert shown.rows[0][-1] == 'not judged: no line could be judged'


# ---- sweep rate -------------------------------------------------------------------------------


def _sweep_report(modes=None, shapes=None, top=400.0):
    from visualdynamics.core.sine import SineSweepSpecification, SineTone

    tone = SineTone('Up', 0.0, [10.0, top], [[1.0]] * 2, [1], [4.0])
    spec = SineSweepSpecification([tone], ['1Z+'], ordinate_unit='m/s**2')
    report = Report('R', [{'kind': 'plot', 'source': 'Spec'}],
                    features=['check_blocks'])
    objects = {'Spec': spec}
    if shapes is not None:
        objects['Shapes'] = shapes
    if modes is not None:
        cb.add(report, 'sweep_rate', {'source': 'Spec'}, {'modes': modes})
    return report, objects


def test_the_sweep_rate_rows_are_the_checks_own():
    from visualdynamics.core.sweep_rate import check_sweep_rate

    report, objects = _sweep_report([[22.0, 40.0], [180.0, 5.0]])
    shown = _shown(report, objects, 'sweep_rate')
    checks = check_sweep_rate(objects['Spec'], [(22.0, 40.0), (180.0, 5.0)])
    assert len(shown.rows) == len(checks) == 2
    assert [row[-1] for row in shown.rows] == [c.verdict for c in checks] \
        == ['too fast', 'builds']
    assert shown.status == cb.FLAGGED
    assert shown.words == f'1 of 2 passes too fast: {checks[0].describe()}.'
    (figure,) = shown.figures
    assert figure['logy'] is True
    assert sum(curve.get('gray', False) for curve in figure['curves']) == 4


def test_a_sweep_that_lets_every_mode_build_passes():
    report, objects = _sweep_report([[180.0, 5.0]])
    shown = _shown(report, objects, 'sweep_rate')
    assert shown.status == cb.PASSED
    assert shown.words == 'Every pass gives its mode time to build (1 pass).'


def test_a_mode_passed_twice_has_its_band_drawn_once():
    from visualdynamics.core.sine import SineSweepSpecification, SineTone

    tone = SineTone('Up', 0.0, [10.0, 400.0, 10.0], [[1.0]] * 3, [1, 1],
                    [4.0, -4.0])
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'sweep_rate', {'source': 'Spec'},
           {'modes': [[180.0, 5.0]]})
    objects = {'Spec': SineSweepSpecification([tone], ['1Z+'],
                                              ordinate_unit='m/s**2')}
    shown = _shown(report, objects, 'sweep_rate')
    assert [row[3] for row in shown.rows] == ['up', 'down']
    (figure,) = shown.figures
    assert sum(curve.get('gray', False) for curve in figure['curves']) == 2


def test_a_sweep_with_no_mode_in_reach_or_no_mode_named_is_not_judged():
    report, objects = _sweep_report([[2000.0, 10.0]])
    shown = _shown(report, objects, 'sweep_rate')
    assert shown.status == cb.NOT_JUDGED
    assert shown.words.startswith("The sweep reaches none of the modes'")
    report, objects = _sweep_report([])
    shown = _shown(report, objects, 'sweep_rate')
    assert shown.status == cb.NOT_JUDGED
    assert shown.words.startswith('No modes are named for the sweep')


def test_with_a_shape_set_the_sweep_rate_is_made_from_its_modes():
    from conftest import plate_geometry_and_shapes

    from visualdynamics.core.sweep_rate import check_sweep_rate

    _geometry, shapes = plate_geometry_and_shapes()
    report, objects = _sweep_report(shapes=shapes, top=5000.0)
    (item,) = cb.items(report, objects)
    assert dict(item.bound) == {'source': 'Spec', 'shapes': 'Shapes'}
    shown = cb.show(item, objects, SI)
    checks = check_sweep_rate(objects['Spec'], shapes)
    assert checks and len(shown.rows) == len(checks)
    assert [row[-1] for row in shown.rows] == [c.verdict for c in checks]
    report, objects = _sweep_report()
    assert cb.items(report, objects) == [], \
        'without modes there is nothing to check a sweep against'


# ---- signature shifts --------------------------------------------------------------------------


def _signature_report(after_modes):
    from test_signature import AFTER, BEFORE, MODES, signature

    before = signature(BEFORE)
    after = signature(AFTER if after_modes is None else after_modes)
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'signature', {'source': 'Before', 'measured': 'After'},
           {'modes': MODES})
    return report, {'Before': before, 'After': after}


def test_the_signature_rows_are_the_comparisons_own():
    from test_signature import MODES

    from visualdynamics.core.signature import compare_signatures

    report, objects = _signature_report(None)
    shown = _shown(report, objects, 'signature')
    comparison = compare_signatures(objects['Before'], objects['After'],
                                    MODES)
    assert [row[-1] for row in shown.rows] == [
        shift.verdict for shift in comparison.shifts]
    assert shown.status == cb.FLAGGED, 'the worked case fails on amplitude'
    counted, local = shown.marks
    assert counted == [None, None, None, None, 'over', 'over'], \
        'amplitude outside, frequency inside'
    assert local == [None] * 6, 'a mode that does not count is not marked'
    assert shown.words.startswith(
        f"{objects['Before'].record_label(0)}: {comparison.describe()}")
    (figure,) = shown.figures
    assert figure['logx'] and figure['logy']
    assert [curve['label'] for curve in figure['curves']] == ['before',
                                                              'after']


def test_a_signature_is_drawn_in_the_display_units():
    from test_signature import BEFORE, F, transmissibility

    from visualdynamics.core.data import Frf
    from visualdynamics.plot.signature import display_scale
    from visualdynamics.units import DEFAULT_SYSTEM as us

    def accelerance(modes):
        return Frf(F, transmissibility(F, modes)[None],
                   response_dof=['101X+'], reference_dof=['1X+'],
                   ordinate_dim=['acceleration/force'])

    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'signature', {'source': 'Before', 'measured': 'After'},
           {'modes': [84.0]})
    objects = {'Before': accelerance(BEFORE), 'After': accelerance(BEFORE)}
    item = cb.items(report, objects)[0]
    (figure,) = cb.show(item, objects, us).figures
    scale = display_scale('acceleration/force', us)
    assert scale != 1.0
    for curve in figure['curves']:
        drawn = 10.0 ** np.array(curve['y'][:5], dtype=float)
        want = np.abs(objects['Before'].ordinate[0][:5]) * scale
        assert np.allclose(drawn, want, rtol=1e-5)


def test_an_unchanged_article_passes():
    from test_signature import BEFORE

    report, objects = _signature_report(BEFORE)
    assert _shown(report, objects, 'signature').status == cb.PASSED


def test_signatures_of_two_quantities_are_not_judged_by_name():
    from test_signature import BEFORE, F, signature

    from visualdynamics.core.data import Frf

    report, objects = _signature_report(None)
    objects['After'] = Frf(F, signature(BEFORE).ordinate,
                           response_dof=['101X+'], reference_dof=['1X+'],
                           ordinate_dim=['acceleration/force'])
    shown = _shown(report, objects, 'signature')
    assert shown.status == cb.NOT_JUDGED
    assert shown.rows[0][-1].startswith('not judged: the signature before is')


# ---- replication segments ----------------------------------------------------------------------


def test_the_segments_table_is_the_readings_own():
    from test_replication_segments import SMALL, _small_pair

    from visualdynamics.core.replication_segments import replication_segments
    from visualdynamics.plot.replication_segments import (
        COLUMNS,
        describe_record,
        verdict_rows,
    )

    run, reference = _small_pair()
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'segments', {'source': 'Run', 'specification': 'Ref'},
           {'segments': [list(mark) for mark in SMALL]})
    objects = {'Run': run, 'Ref': reference}
    shown = _shown(report, objects, 'segments')
    result = replication_segments(run, reference, SMALL)
    assert shown.headers == list(COLUMNS)
    assert shown.rows == [list(row) for row in verdict_rows(result)]
    assert shown.words == describe_record(result) + '.'
    assert shown.status == {True: cb.PASSED, False: cb.FLAGGED}[result.passed]
    assert [mark[-1] == 'over' for mark in shown.marks] == [
        row[-1] == 'fails' for row in shown.rows]
    assert any(mark[-1] == 'over' for mark in shown.marks)
    (figure,) = shown.figures
    assert figure['curves'][0]['label'] == 'reference'


def test_a_segments_block_with_no_segment_marked_is_not_judged():
    from test_replication_segments import _small_pair

    run, reference = _small_pair()
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'segments', {'source': 'Run', 'specification': 'Ref'},
           {'segments': []})
    shown = _shown(report, {'Run': run, 'Ref': reference}, 'segments')
    assert shown.status == cb.NOT_JUDGED
    assert shown.words.startswith('No segment is marked')


# ---- the readings, made once ---------------------------------------------------------------------


def test_a_reading_is_made_once_and_again_when_its_record_changes():
    report, objects = _random_report(_noise())
    (item,) = cb.items(report, objects)
    first = cb.read(item, objects)
    assert cb.read(item, objects) is first
    objects['History'].ordinate = objects['History'].ordinate * 2.0
    assert cb.read(item, objects) is not first, 'a replaced array'
    again = cb.read(item, objects)
    objects['History'] = _noise(seed=11)
    assert cb.read(item, objects) is not again, 'a replaced object'


def test_a_reading_follows_the_stretch_read():
    from visualdynamics.core.shocks import Shock

    report, objects = _shock_report(_noise(seconds=1.0))
    (item,) = cb.items(report, objects)
    first = cb.read(item, objects)
    objects['History'].shocks = [Shock(0.1, 0.2)]
    assert cb.read(item, objects) is not first


def test_prepare_reads_what_is_pending_and_says_how_far_it_has_got():
    report, objects = _random_report(_noise())
    told = []
    assert len(cb.pending(report, objects)) == 1
    assert cb.prepare(report, objects,
                      progress=lambda done, total: told.append((done, total))
                      ) == 1
    assert told == [(0, 1), (1, 1)]
    assert cb.pending(report, objects) == []
    assert cb.prepare(report, objects) == 0
    report.features = []
    cb.forget()
    assert cb.pending(report, objects) == [], 'off, nothing to read'


def test_a_cancel_raised_by_progress_stops_the_reading():
    from visualdynamics.core.progress import Cancelled

    report, objects = _random_report(_noise())

    def cancel(done, total):
        raise Cancelled('reading')

    with pytest.raises(Cancelled):
        cb.prepare(report, objects, progress=cancel)
    assert len(cb.pending(report, objects)) == 1, 'asked for again'
    shown = _shown(report, objects, 'time_statistics')
    assert shown.status == cb.NOT_JUDGED
    assert shown.words.startswith('The reading was canceled')
    assert cb.prepare(report, objects) == 1
    assert _shown(report, objects, 'time_statistics').status == cb.PASSED


def test_the_kept_readings_are_bounded():
    report, objects = _random_report(_noise(channels=1, seconds=1.0))
    for seed in range(cb.KEPT + 3):
        objects['History'] = _noise(channels=1, seconds=1.0, seed=seed)
        cb.read(cb.items(report, objects)[0], objects)
    assert len(cb._KEPT) == cb.KEPT


# ---- the page --------------------------------------------------------------------------------------


def test_the_checks_close_the_results_before_the_conclusions():
    report, objects = _random_report(_noise())
    blocks = _page(report, objects)
    htmls = [block.get('html', '') for block in blocks]
    checks = htmls.index('<h3>Checks</h3>')
    conclusions = next(k for k, html in enumerate(htmls)
                       if html.startswith('<h3>Conclusions'))
    assert checks < conclusions
    (head,) = _heads(blocks)
    assert head[:2] == ('Time statistics', 'Passed')
    assert 'class="check-verdict check-passed"' in head[2]
    after = blocks[checks + 2:conclusions]
    assert [block['kind'] for block in after] == ['table', 'plot', 'plot']
    assert after[0]['label'] == 'Table 1'
    assert [block['label'] for block in after[1:]] == ['Figure 3',
                                                      'Figure 4']


@pytest.mark.parametrize(('kinds', 'at'), [
    ([], 0),
    (['text'], 1),
    (['plot', '## A', 'plot', '## B', 'text'], 3),
    (['plot', 'text', '## C'], 2),
    (['## A', 'plot'], 2),
    (['## A', 'text'], 0),
])
def test_the_checks_go_at_the_first_heading_after_the_last_figure(kinds, at):
    blocks = [({'kind': 'text', 'text': kind} if kind.startswith('#')
               or kind == 'text' else {'kind': kind}) for kind in kinds]
    assert cb.position(blocks) == at


def test_a_value_a_log_axis_cannot_hold_is_a_gap():
    figure = cb._figure('c', [0.0, 1.0, 10.0], 'x', 'y',
                        [('a', None, [0.0, 1.0, 100.0], {})],
                        logx=True, logy=True)
    assert figure['x'] == [None, 0.0, 1.0]
    assert figure['curves'][0]['y'] == [None, 0.0, 2.0]


def test_a_head_escapes_what_it_says():
    blocks = [{'kind': 'text', 'html': '', 'check_head': {
        'title': 'A <b>', 'status': cb.NOT_JUDGED, 'words': 'x < y & z',
        'criterion': ''}}]
    cb.finish(blocks)
    assert blocks[0]['html'] == (
        '<h4 class="check">A &lt;b&gt;</h4><p class="check-verdict '
        'check-not-judged"><strong>Not judged.</strong> x &lt; y &amp; z</p>')


def test_the_page_carries_the_feature_style_only_when_on():
    report, objects = _random_report(_noise())
    assert 'p.check-verdict' in render_html(report, objects)
    report.features = []
    assert 'p.check-verdict' not in render_html(report, objects)


def test_on_the_edit_page_a_check_of_its_own_is_selected_by_its_block():
    report, objects = _random_report(_noise())
    index = cb.add(report, 'time_statistics', {'source': 'History'})
    blocks = _page(report, objects, edit=True)
    mine = [block for block in blocks if block.get('index') == index]
    assert [block['kind'] for block in mine][:2] == ['text', 'table']
    report.blocks.pop()
    blocks = _page(report, objects, edit=True)
    checks = next(k for k, block in enumerate(blocks)
                  if block.get('html') == '<h3>Checks</h3>')
    assert all('index' not in block for block in blocks[checks:checks + 3]), \
        'a check made from what the report shows has no block to select'


def test_under_the_skeleton_the_checks_fall_under_results():
    report, objects = _random_report(_noise())
    report.features = ['skeleton_sections', 'check_blocks']
    blocks = _page(report, objects)
    htmls = [block.get('html', '') for block in blocks]
    results = next(k for k, html in enumerate(htmls)
                   if html.startswith('<h2 class="skeleton">8. Results'))
    after = next(k for k, html in enumerate(htmls)
                 if html.startswith('<h2 class="skeleton">9.'))
    assert results < htmls.index('<h3>Checks</h3>') < after


def test_a_check_that_judged_nothing_says_so_on_the_page():
    history = _noise(channels=1)
    history.ordinate[0] = 0.0
    report, objects = _random_report(history)
    ((_title, word, html),) = _heads(_page(report, objects))
    assert word == 'Not judged'
    assert 'No channel could be read.' in html


def test_the_random_run_report_carries_its_time_statistics():
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    name = project.generate_report('random')
    project.report_features(name, enable=['check_blocks'])
    blocks = _payload(render_html(project[name], project, unit_system=SI,
                                  object_groups=project.object_groups)
                      )['blocks']
    ((title, _word, _html),) = _heads(blocks)
    assert title == 'Time statistics'
    table = next(block for block in blocks if block['kind'] == 'table'
                 and block['caption'].startswith('Time statistics'))
    assert len(table['rows']) == project.time_history.num_records
    assert isinstance(project.psds[0], Psd)


# ---- the edges -----------------------------------------------------------------------------------------


def test_a_figure_holds_so_many_curves_and_says_how_many_it_left_off():
    from test_shock_check import compensated, times

    t = times(0.4)
    rows = [compensated(t, 0.2) * (1.0 + 0.1 * k) for k in range(9)]
    report, objects = _shock_report(_shock_record(*rows))
    shown = _shown(report, objects, 'shock_validity')
    assert len(shown.rows) == 9
    (figure,) = shown.figures
    assert len(figure['curves']) == cb.MOST_CURVES == 8
    assert figure['caption'].endswith('The first 8 of 9 channels are drawn; '
                                      'the table holds every one.')


def test_a_caption_says_nothing_when_nothing_was_left_off():
    assert cb._left_off(8, 8, 'channels') == ''
    assert cb._left_off(8, 7, 'channels') == ''
    assert cb._left_off(4, 6, 'shocks') == (
        ' The first 4 of 6 shocks are drawn; the table holds every one.')


def test_a_signed_number_has_a_minus_sign_and_a_dash_for_none():
    assert cb._signed(float('nan')) == cb.DASH
    assert cb._signed(-1.26) == '−1.3'
    assert cb._signed(0.5, 2, 'dB') == '+0.50 dB'


def test_two_shocks_are_told_apart_in_the_table_and_the_figures():
    from test_shock_check import compensated, times

    from visualdynamics.core.shocks import Shock

    t = times(0.8)
    record = _shock_record(compensated(t, 0.2) + compensated(t, 0.6))
    record.shocks = [Shock(0.0, 0.4), Shock(0.4, 0.4)]
    report, objects = _shock_report(record)
    shown = _shown(report, objects, 'shock_validity')
    assert [row[1] for row in shown.rows] == ['1', '2']
    assert [figure['caption'].split(',')[0] for figure in shown.figures] == [
        'Velocity through shock 1', 'Velocity through shock 2']
    single = _shock_report(_shock_record(compensated(times(0.4), 0.2)))
    (figure,) = _shown(*single, 'shock_validity').figures
    assert figure['caption'].startswith('Velocity through the window,')


def test_the_points_rule_marks_the_spectrum_that_fails():
    report, objects = _srs_pair(1.1)
    shown = _shown(report, objects, 'srs_points')
    assert shown.marks[0] == [None] * 5
    assert shown.marks[1] == [None, None, 'over', None, 'over']
    assert shown.rows[1][3].startswith('+8.0 dB at ')


def test_a_pulse_with_points_outside_its_envelope_marks_them():
    report, objects = _pulse_report(peak=30.0)
    shown = _shown(report, objects, 'pulse')
    assert shown.rows[-1][0] == 'Points outside the envelope'
    assert shown.marks[-1][1] == 'over'
    report, objects = _pulse_report()
    assert _shown(report, objects, 'pulse').marks[-1][1] is None


def test_a_check_block_is_not_what_the_report_shows():
    """An object only a check of its own reads does not make the checks
    made from what the report shows."""
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'pulse', {'source': 'History'},
           {'shape': 'half-sine', 'peak': 490.0, 'duration': 0.011})
    keys = [item.check.key for item in cb.items(report,
                                                {'History': _noise()})]
    assert keys == ['pulse']


def test_a_record_of_no_declared_quantity_is_shown_as_it_is_held():
    history = _noise(channels=1)
    unknown = TimeHistory(history.abscissa, history.ordinate * 3.0,
                          response_dof=['1Z+'])
    report, objects = _random_report(unknown)
    rows = _shown(report, objects, 'time_statistics').rows
    rms = float(np.sqrt(np.mean(unknown.ordinate[0] ** 2)))
    assert rows[0][2] == f'{rms:.4g}', 'no unit, no conversion'
    report, objects = _random_report(history)
    rows = _shown(report, objects, 'time_statistics').rows
    assert rows[0][2].endswith(' m/s²')


def test_the_histogram_keeps_its_area_in_the_display_units():
    from visualdynamics.units import DEFAULT_SYSTEM as us

    report, objects = _random_report(_noise(channels=1))
    item = cb.items(report, objects)[0]
    record = cb.show(item, objects, us).figures[0]['curves'][0]
    x = np.array(record['x'], dtype=float)
    y = np.array(record['y'], dtype=float)
    area = np.sum((x[2:-1:2] - x[1:-1:2]) * y[1:-1:2])
    assert area == pytest.approx(1.0, abs=1e-5)


def test_the_gaussian_reaches_as_far_as_the_record_does():
    history = _noise(channels=1)
    for low, high in ((-18.0, 0.0), (0.0, 18.0)):
        values = history.ordinate[0] + 5.0
        values[10] += low
        values[20] += high
        spiked = TimeHistory(history.abscissa, values[None],
                             response_dof=['1Z+'],
                             ordinate_dim=['acceleration'])
        report, objects = _random_report(spiked)
        figure = _shown(report, objects, 'time_statistics').figures[0]
        grid = np.array(figure['x'], dtype=float)
        edges = np.array(figure['curves'][0]['x'], dtype=float)
        mean = float(np.mean(values))
        reach = max(abs(edges[0] - mean), abs(edges[-1] - mean))
        assert grid[0] == pytest.approx(mean - reach, abs=1e-5)
        assert grid[-1] == pytest.approx(mean + reach, abs=1e-5)
        cb.forget()


def test_a_frame_that_does_not_vary_is_said_on_its_figure():
    history = _noise(channels=1, seconds=20.0)
    history.ordinate[0, 4096:8192] = 0.0
    report, objects = _random_report(history)
    figures = _shown(report, objects, 'time_statistics').figures
    frames = next(figure for figure in figures
                  if figure['ylabel'] == 'Pearson kurtosis')
    assert 'does not vary and has no kurtosis' in frames['caption']
    from visualdynamics.core.time_statistics import record_statistics

    when = record_statistics(history, 0).worst_frame[0]
    assert f'A frame at {when:.4g} s does not vary' in frames['caption']
    assert None in frames['curves'][0]['y'], 'a gap where it has none'


def test_a_signature_with_no_modes_named_compares_the_two_it_finds():
    from test_signature import AFTER, BEFORE, signature

    from visualdynamics.core.signature import compare_signatures, find_modes

    # a third peak, which the default of two leaves out
    third = (900.0, 25.0, 0.1)
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'signature', {'source': 'Before', 'measured': 'After'})
    objects = {'Before': signature([*BEFORE, third]),
               'After': signature([*AFTER, third])}
    shown = _shown(report, objects, 'signature')
    found = find_modes(objects['Before'], 0)
    assert len(found) == 2, 'two, as the reading finds them by default'
    comparison = compare_signatures(objects['Before'], objects['After'],
                                    found)
    assert [row[1] for row in shown.rows] == [
        shift.mode.label() for shift in comparison.shifts]


def test_a_segments_band_saved_as_a_list_is_read_as_given():
    from test_replication_segments import SMALL, _small_pair

    from visualdynamics.core.replication_segments import replication_segments
    from visualdynamics.plot.replication_segments import verdict_rows

    run, reference = _small_pair()
    report = Report('R', [], features=['check_blocks'])
    cb.add(report, 'segments', {'source': 'Run', 'specification': 'Ref'},
           {'segments': [list(mark) for mark in SMALL],
            'band': [5.0, 300.0]})
    shown = _shown(report, {'Run': run, 'Ref': reference}, 'segments')
    result = replication_segments(run, reference, SMALL, band=(5.0, 300.0))
    assert shown.rows == [list(row) for row in verdict_rows(result)]


def test_the_editor_shows_a_check_blocks_settings(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = window.generate_report('random')
    window.project.report_check_block(name, 'pulse', shape='half-sine',
                                      peak=490.0, duration=0.011)
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    editor = window.report_editor
    editor.selected = len(window.objects[name].blocks) - 1
    editor._show_selection()
    from PySide6.QtWidgets import QLabel

    shown = [label.text() for label in editor.pane.widget().findChildren(
        QLabel)]
    assert '<b>Classical pulse</b> — check' in shown
    assert cb.CHECKS['pulse'].description in shown
    assert "'half-sine'" in shown and '490.0' in shown


def test_export_reads_the_checks_on_the_worker_first(window, pump, tmp_path,
                                                     monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = window.generate_report('random')
    window.project.report_features(name, enable=['check_blocks'])
    window.tree.setCurrentItem(window._item_for_object(name))
    pump()
    cb.forget()
    ran = []
    real = window._run_long

    def spy(status, verb, *args, **kwargs):
        ran.append(verb)
        return real(status, verb, *args, **kwargs)

    window._run_long = spy
    path = tmp_path / 'out.html'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(path), ''))
    window.export_report()
    assert ran == [cb.prepare]
    assert '<h3>Checks<\\/h3>' in path.read_text(encoding='utf-8')


# ---- the window -------------------------------------------------------------------------------------


def test_the_window_reads_the_checks_on_its_worker(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = window.generate_report('random')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    ran = []
    real = window._run_long

    def spy(status, verb, *args, **kwargs):
        ran.append((status, verb))
        assert cb.pending(window.objects[name], window.objects,
                          window.object_groups), 'read before the render'
        return real(status, verb, *args, **kwargs)

    window._run_long = spy
    # turned on in the editor's panel, which redraws the page
    window.report_editor._operate({'op': 'features',
                                   'enable': ['check_blocks']})
    pump()
    assert [verb for _status, verb in ran] == [cb.prepare]
    assert ran[0][0] == 'Reading the report checks…'
    assert cb.pending(window.objects[name], window.objects,
                      window.object_groups) == []
    assert window.report_editor.failure is None
    with open(window.report_editor._page_path, encoding='utf-8') as page:
        drawn = _payload(page.read())['blocks']
    assert {'kind': 'text', 'html': '<h3>Checks</h3>'} in [
        {key: block.get(key) for key in ('kind', 'html')} for block in drawn]
    ran.clear()
    window.report_editor.rebuild()
    assert ran == [], 'kept: nothing to read again'


def test_a_canceled_reading_draws_the_page_not_judged(window, pump):
    from visualdynamics.core.progress import Cancelled

    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = window.generate_report('random')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()

    def cancel(done, total):
        raise Cancelled('Reading the report checks…')

    window._run_long = (lambda status, verb, *args, **kwargs:
                        verb(*args, progress=cancel, **kwargs))
    window.report_editor._operate({'op': 'features',
                                   'enable': ['check_blocks']})
    pump()
    with open(window.report_editor._page_path, encoding='utf-8') as page:
        drawn = _payload(page.read())['blocks']
    ((title, word, html),) = _heads(drawn)
    assert (title, word) == ('Time statistics', 'Not judged')
    assert 'The reading was canceled' in html
    assert cb.pending(window.objects[name], window.objects,
                      window.object_groups), 'asked for again'


def test_the_window_waits_while_its_worker_is_busy(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = window.generate_report('random')
    window.project.report_features(name, enable=['check_blocks'])
    window._importing = True
    try:
        assert window._prepare_report_checks(window.objects[name]) is False
        assert window._report_settle.isActive()
    finally:
        window._importing = False
        window._report_settle.stop()
