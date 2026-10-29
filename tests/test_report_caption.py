"""The caption a report plot of a spectrum carries, read off its record.

Every row is a fact some object already holds: the channel and the
channel table's row for it, the run the spectrum came from and the
stretch of it averaged, the estimate, the line spacing, the window and
overlap, the equivalent DOF, and the requirement drawn. Pinned here
against hand values on a synthetic record, against the plate's random
run worked up the way the app does it, in the report's payload, on a
saved plot, and in the report editor's pane.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from conftest import fixture_path, web_close, web_read, web_view

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.report_caption import (
    CHANNEL_ROWS,
    NOT_RECORDED,
    ROWS,
    CaptionChannel,
    RecordCaption,
    record_caption,
)

FS = 1024.0
#: Hann frames of 1024 samples at half overlap, nine of them, from
#: 0.5 s: 5120 samples, so the stretch is 0.5 s to 5.5 s, the lines are
#: 1 Hz apart, the nominal DOF is 18, and Welch's equivalent count is
#: 18 / (1 + 2 (1/6)² (8/9)) = 17.1529 (the Hann window's correlation
#: at half overlap is exactly 1/6)
AVERAGING = Averaging(frame_length=1024, overlap=0.5, window='hann',
                      frames=9, start=0.5)


@pytest.fixture(scope='module')
def measured():
    """Two channels at one DOF, an acceleration and a force, white,
    averaged by the averaging above."""
    rng = np.random.default_rng(7)
    t = np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((2, t.size)), response_dof=['1Z+', '1Z+'],
        ordinate_dim=['acceleration', 'force'])
    history.averaging = AVERAGING
    return history.compute_psds()


@pytest.fixture(scope='module')
def table():
    return ChannelTable({'channel': [3, 4], 'node': [1, 1],
                         'direction': ['Z+', 'Z+'],
                         'unit': ['m/s^2', 'N'],
                         'channel_type': ['acceleration', 'force'],
                         'comment': ['bracket', '1Z+']})


@pytest.fixture(scope='module')
def run():
    return visualdynamics.random_vibration_run(fixture_path('plate',
                                                            'random.nc4'))


def _rows(caption):
    return dict(caption.rows())


# ---- read off the record, against hand values ----------------------------


def test_every_row_is_read_off_the_record_and_its_averaging(measured, table):
    caption = record_caption(measured, 0, averaging=AVERAGING, run='Run 6',
                             sample_rate=FS, channel_table=table)
    rows = _rows(caption)
    assert [label for label, _ in caption.rows()] == list(ROWS)
    assert rows['Channel'] == 'channel 3, 1Z+ acceleration'
    assert rows['Location and axis'] == 'node 1 (bracket), Z+'
    assert rows['Run'] == 'Run 6, 0.5 s to 5.5 s'
    assert rows['Estimate'] == 'ASD, single channel, linear average of 9 frames'
    assert rows['Resolution'] == '1 Hz lines'
    assert rows['Window'] == 'hann, 50% overlap, frames of 1024 samples'
    assert rows['DOF'] == '17.15 equivalent (18 nominal)'
    assert rows['Requirement'] == 'none drawn'
    assert caption.dof == pytest.approx((18 * 324 / 340,))
    assert caption.stretch == pytest.approx((0.5, 5.5))
    assert caption.line_spacing == pytest.approx(1.0)


def test_records_of_one_channel_pooled_count_in_full():
    """A capture saved one record per average pools its records, each
    adding its frames' count in full: two records of nine frames are
    twice the nominal and twice the equivalent count."""
    rng = np.random.default_rng(11)
    t = np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((2, t.size)), response_dof=['1Z+', '1Z+'],
        ordinate_dim=['acceleration', 'acceleration'])
    history.averaging = AVERAGING
    pooled = history.compute_psds()
    caption = record_caption(pooled, 0, averaging=AVERAGING)
    assert caption.pooled == (2,) and caption.nominal_dof == (36,)
    rows = _rows(caption)
    assert rows['DOF'] == '34.31 equivalent (36 nominal)'
    assert rows['Estimate'].endswith('of 9 frames in each of 2 records')


def test_the_channel_table_row_is_the_one_of_the_records_quantity(measured,
                                                                  table):
    """A drive point's force channel shares its DOF with the
    acceleration beside it; only the type tells the two rows apart,
    and a comment that only repeats the DOF says nothing."""
    force = record_caption(measured, 1, channel_table=table).channels[0]
    assert (force.channel, force.quantity, force.description) == (4, 'force', '')
    assert force.place() == 'node 1, Z+'


def test_without_a_table_the_dof_still_names_the_channel(measured):
    channel = record_caption(measured, 0).channels[0]
    assert channel.name() == '1Z+ acceleration'
    assert (channel.node, channel.direction) == (1, 'Z+')


def test_the_nearest_global_axis_is_said_with_how_far_off_it_sits():
    square = CaptionChannel('7X+', 'acceleration', 1, 7, 'X+',
                            nearest_axis='X+', axis_angle=0.0)
    tilted = CaptionChannel('7X+', 'acceleration', 1, 7, 'X+',
                            nearest_axis='Y-', axis_angle=12.4)
    assert square.place() == 'node 7, X+ (nearest global X+)'
    assert tilted.place() == 'node 7, X+ (nearest global Y-, 12° off)'


def test_what_nothing_records_is_written_not_recorded():
    """A spectrum read from a file carries no averaging, no DOF and no
    run: each row says so rather than vanishing, so the gap shows."""
    bare = visualdynamics.Psd(np.arange(1.0, 101.0), np.ones((1, 100)),
                              response_dof='5Y-',
                              ordinate_dim='acceleration**2/frequency')
    rows = _rows(record_caption(bare))
    assert rows['Run'] == NOT_RECORDED
    assert rows['Window'] == NOT_RECORDED
    assert rows['DOF'] == NOT_RECORDED
    assert rows['Estimate'] == 'ASD, single channel, averaging not recorded'
    assert rows['Resolution'] == '1 Hz lines'


def test_a_block_label_is_the_run_when_nothing_names_one():
    labeled = visualdynamics.Psd(np.arange(1.0, 11.0), np.ones((2, 10)),
                                 response_dof=['5Y-', '5Y-'],
                                 ordinate_dim='acceleration**2/frequency',
                                 block=['Run 3', 'Run 4'])
    assert _rows(record_caption(labeled, 1))['Run'] == 'Run 4'


def test_uneven_lines_have_no_spacing_to_state():
    uneven = visualdynamics.Psd(np.array([1.0, 2.0, 4.0, 8.0]),
                                np.ones((1, 4)), response_dof='5Y-',
                                ordinate_dim='acceleration**2/frequency')
    assert _rows(record_caption(uneven))['Resolution'] == 'lines not evenly spaced'


def test_several_records_name_each_channel_and_a_span_of_dof(measured, table):
    caption = record_caption(measured, [0, 1], averaging=AVERAGING,
                             channel_table=table)
    rows = _rows(caption)
    assert rows['Channel'] == ('channel 3, 1Z+ acceleration; '
                               'channel 4, 1Z+ force')
    assert rows['Estimate'].startswith('ASD of each of 2 channels')
    spread = RecordCaption(channels=caption.channels, kind='ASD',
                           dof=(48.0, 53.15), nominal_dof=(50, 56))
    assert _rows(spread)['DOF'] == '48 to 53.15 equivalent (50 to 56 nominal)'


def test_a_banded_spectrum_states_its_bands_and_its_lines_dof(measured):
    """Banding sums lines and raises the count: each band carries the
    count of the lines it sums, from one line's 17.15 (a band narrower
    than a line holds part of one) up, and the narrowband lines' count
    is said beside it. Without the narrowband source neither is known."""
    banded = measured.to_octave(3)
    rows = _rows(record_caption(banded, 0, averaging=AVERAGING,
                                narrowband=measured))
    assert rows['Estimate'].endswith(', banded to 1/3 octave')
    assert rows['Resolution'] == '1/3-octave bands, from 1 Hz lines'
    assert re.fullmatch(r'17\.15 to \d+(\.\d+)? equivalent per band, from '
                        r'17\.15 equivalent \(18 nominal\) per narrowband '
                        r'line', rows['DOF']), rows['DOF']
    # without the averaging, the window's correlation is not known
    unknown = _rows(record_caption(banded, 0, narrowband=measured))
    assert unknown['DOF'] == ('17.15 equivalent per narrowband line, '
                              'before banding')
    # without its source, a banded spectrum has neither to give
    alone = _rows(record_caption(banded, 0))
    assert alone['Resolution'] == '1/3-octave bands'
    assert alone['DOF'] == NOT_RECORDED


def test_the_requirement_row_carries_the_comparisons_own_rms(measured):
    """The two rms values are the verdict table's: both over the band
    it judged, the measurement as it is scaled for the comparison."""
    from visualdynamics.core.compliance import compare

    level = float(np.median(measured.ordinate[0, 10:400]))
    spec = visualdynamics.Specification(
        abscissa=np.array([20.0, 400.0]), ordinate=np.full((1, 2), level),
        response_dof=['1Z+'], ordinate_dim=['acceleration**2/frequency'],
        abort_upper=np.full((1, 2), 4 * level),
        abort_lower=np.full((1, 2), level / 4))
    held = measured.scale_db
    measured.scale_db = 0
    try:
        caption = record_caption(measured, 0, specification=spec,
                                 requirement='Profile A',
                                 unit_system=visualdynamics.SI)
        table = compare(spec, measured, 0, 0, 0)
    finally:
        measured.scale_db = held
    assert caption.measured_rms == pytest.approx(table['measured_rms'])
    assert caption.reference_rms == pytest.approx(table['specification_rms'])
    assert caption.limits == ('abort',), 'no warning pair was written'
    said = _rows(caption)['Requirement']
    assert said.startswith('Profile A, reference with its abort limits drawn; ')
    assert ' rms measured against ' in said and said.count(' rms specified')
    assert 'as drawn' not in said and 'asked' not in said
    assert caption.drawn_rms is None


def test_a_scaled_comparison_gives_the_rms_as_measured_and_as_drawn(measured):
    """Scaled for the comparison, the record has two rms values: the one
    a reader holding the raw data computes, and the one the figure draws
    and the verdict judges. The scale itself is said in the figure's
    legend and caption, not again here."""
    from visualdynamics.core.compliance import compare

    level = float(np.median(measured.ordinate[0, 10:400]))
    spec = visualdynamics.Specification(
        abscissa=np.array([20.0, 400.0]), ordinate=np.full((1, 2), level),
        response_dof=['1Z+'], ordinate_dim=['acceleration**2/frequency'],
        abort_upper=np.full((1, 2), 4 * level),
        abort_lower=np.full((1, 2), level / 4))
    caption = record_caption(measured, 0, specification=spec, scale_db=6,
                             unit_system=visualdynamics.SI)
    raw = compare(spec, measured, 0, 0, 0)['measured_rms']
    drawn = compare(spec, measured, 0, 0, 6)['measured_rms']
    assert caption.measured_rms == pytest.approx(raw)
    assert caption.drawn_rms == pytest.approx(drawn)
    assert caption.drawn_rms == pytest.approx(raw * 10 ** (6 / 20))
    said = _rows(caption)['Requirement']
    assert (f'{raw:.3g} m/s² rms measured, {drawn:.3g} m/s² rms as drawn, '
            'against ') in said
    assert 'dB' not in said, 'the legend and the caption say the scale'


def test_a_caption_reads_only_a_measured_spectrum(measured):
    spec = visualdynamics.Specification(
        abscissa=np.array([20.0, 400.0]), ordinate=np.ones((1, 2)),
        response_dof=['1Z+'], ordinate_dim=['acceleration**2/frequency'])
    with pytest.raises(TypeError, match='measured spectrum'):
        record_caption(spec)
    with pytest.raises(TypeError, match='TimeHistory'):
        record_caption(visualdynamics.TimeHistory(
            np.arange(4.0), np.ones((1, 4)), response_dof='1Z+'))
    with pytest.raises(IndexError):
        record_caption(measured, 2)


def test_the_text_is_every_row_in_order(measured):
    caption = record_caption(measured, 0, averaging=AVERAGING)
    text = caption.text()
    assert text.startswith('Channel: 1Z+ acceleration. Location and axis: ')
    assert text.endswith('Requirement: none drawn.')
    starts = [text.index(f'{label}: ') for label in ROWS]
    assert starts == sorted(starts)


def test_unequal_pooling_is_said_channel_by_channel():
    """Two records at 1Z+ and one at 2Z+: the pooling clause read off
    the first channel said '9 frames in each of 2 records' of both."""
    rng = np.random.default_rng(13)
    t = np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((3, t.size)),
        response_dof=['1Z+', '1Z+', '2Z+'],
        ordinate_dim=['acceleration'] * 3)
    history.averaging = AVERAGING
    psds = history.compute_psds()
    both = record_caption(psds, [0, 1], averaging=AVERAGING)
    assert both.pooled == (2, 1) and both.nominal_dof == (36, 18)
    assert _rows(both)['Estimate'] == (
        'ASD of each of 2 channels, linear average of 9 frames in each '
        'record, pooled over 2 records at 1Z+, 1 record at 2Z+')
    assert _rows(both)['DOF'] == '17.15 to 34.31 equivalent (18 to 36 nominal)'
    alone = record_caption(psds, 1, averaging=AVERAGING)
    assert _rows(alone)['Estimate'].endswith('linear average of 9 frames')


def test_a_cross_term_names_both_its_channels(table):
    """A CPSD's cross term is of two channels; by its response alone,
    1Z+ against 2Z+ read as 1Z+ a second time beside 1Z+'s own ASD."""
    rng = np.random.default_rng(17)
    t = np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((2, t.size)), response_dof=['1Z+', '2Z+'],
        ordinate_dim=['acceleration', 'force'])
    history.averaging = AVERAGING
    cpsd = history.compute_cpsds()
    pairs = [cpsd.record_pair(record) for record in range(cpsd.num_records)]
    cross = pairs.index(('1Z+', '2Z+'))
    auto = pairs.index(('1Z+', '1Z+'))
    caption = record_caption(cpsd, [auto, cross], averaging=AVERAGING)
    assert caption.kind == 'ASD and CPSD'
    assert _rows(caption)['Channel'] == (
        '1Z+ acceleration; 1Z+ acceleration against 2Z+ force')
    assert _rows(caption)['Location and axis'] == (
        'node 1, Z+; node 1, Z+; against node 2, Z+')
    numbered = record_caption(cpsd, cross, channel_table=ChannelTable(
        {'channel': [3, 4], 'node': [1, 2], 'direction': ['Z+', 'Z+'],
         'unit': ['m/s^2', 'N'], 'channel_type': ['acceleration', 'force'],
         'comment': ['bracket', 'load cell']}))
    assert _rows(numbered)['Channel'] == (
        'channel 3, 1Z+ acceleration against channel 4, 2Z+ force')
    assert _rows(numbered)['Location and axis'] == (
        'node 1 (bracket), Z+; against node 2 (load cell), Z+')


def test_the_stretch_starts_where_the_averaging_does_on_the_runs_clock():
    """A run whose clock starts at 10 s, averaged from 0.5 s into it:
    the stretch is 10.5 s to 15.5 s, read through the project."""
    rng = np.random.default_rng(19)
    t = 10.0 + np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((1, t.size)), response_dof=['1Z+'],
        ordinate_dim=['acceleration'])
    history.averaging = AVERAGING
    project = visualdynamics.Project('Clock')
    project.add('Run 7', history)
    name = project.compute_psds('Run 7')
    caption = project.caption_of(name)
    assert caption.stretch == pytest.approx((10.5, 15.5))
    assert _rows(caption)['Run'] == 'Run 7, 10.5 s to 15.5 s'


def test_the_window_row_is_the_averagings_own_description():
    """One statement of an averaging, with the window's parameter and
    the detrend, which the row once left out; and two-thirds overlap
    reads the same here as everywhere else."""
    tukey = Averaging(frame_length=1024, overlap=2 / 3, window='tukey',
                      window_parameter=0.25, frames=9, detrend='mean')
    assert tukey.overlap_text() == '66.7%'
    assert tukey.window_text() == 'tukey (alpha 0.25)'
    assert tukey.describe() == ('tukey (alpha 0.25), 66.7% overlap, frames '
                                'of 1024 samples, mean removed from each '
                                'frame')
    bare = visualdynamics.Psd(np.arange(1.0, 101.0), np.ones((1, 100)),
                              response_dof='5Y-',
                              ordinate_dim='acceleration**2/frequency')
    assert _rows(record_caption(bare, averaging=tukey))['Window'] == \
        tukey.describe()
    assert AVERAGING.describe() == 'hann, 50% overlap, frames of 1024 samples'
    linear = Averaging(frame_length=64, detrend='linear')
    assert linear.describe().endswith(', linear trend removed from each frame')


def test_a_reports_overlap_reference_reads_the_same_as_the_caption():
    from visualdynamics.report import _field_value
    from visualdynamics.units import DEFAULT_SYSTEM

    history = visualdynamics.TimeHistory(
        np.arange(4096) / FS, np.zeros((1, 4096)), response_dof=['1Z+'])
    history.averaging = Averaging(frame_length=1024, overlap=2 / 3,
                                  frames=3)
    assert _field_value(history, 'overlap', DEFAULT_SYSTEM) == '66.7%'


# ---- read off a project --------------------------------------------------


def test_the_project_finds_the_run_the_averaging_and_the_table(run):
    rows = _rows(run.caption_of('Time History PSDs', 8))
    assert rows == {
        'Channel': 'channel 9, 101Z+ force',
        'Location and axis': 'node 101, Z+',
        'Run': 'Time History, 6 s to 13.25 s',
        'Estimate': 'ASD, single channel, linear average of 28 frames',
        'Resolution': '2 Hz lines',
        'Window': 'hann, 50% overlap, frames of 2048 samples',
        'DOF': '53.15 equivalent (56 nominal)',
        'Requirement': 'none drawn'}


def test_a_stored_banded_spectrum_reads_through_its_narrowband_source(run):
    stored = run.caption_of('Time History PSDs 1/6 Octave', 0,
                            specification='Specification 1/6 Octave')
    drawn = run.caption_of('Time History PSDs', 0,
                           specification='Specification', per_octave=6)
    for caption in (stored, drawn):
        rows = _rows(caption)
        assert rows['Window'] == 'hann, 50% overlap, frames of 2048 samples'
        assert rows['Resolution'] == '1/6-octave bands, from 2 Hz lines'
    assert stored.requirement == 'Specification 1/6 Octave'
    assert drawn.requirement == 'Specification'
    # banded for the plot, compared on the same bands: the same numbers
    # as the stored banded pair
    assert drawn.measured_rms == pytest.approx(stored.measured_rms)
    assert drawn.reference_rms == pytest.approx(stored.reference_rms)


def test_an_imported_spectrum_names_no_run(run):
    """The name an import or a person gave the object is not a run."""
    project = visualdynamics.Project('Imported')
    project.add('Run 9 ASD', run['Time History PSDs'])
    rows = _rows(project.caption_of('Run 9 ASD'))
    assert rows['Run'] == NOT_RECORDED
    assert rows['Window'] == NOT_RECORDED
    with pytest.raises(TypeError, match='not a measured record'):
        run.caption_of('Specification')


# ---- in the report -------------------------------------------------------


def _payload(page):
    return json.loads(re.search(r'<script id="data"[^>]*>(.*?)</script>',
                                page, re.DOTALL).group(1))


def test_a_block_that_asks_carries_the_caption_its_records_give(run):
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    blocks = [{'kind': 'plot', 'source': 'Time History PSDs',
               'specification': 'Specification', 'channel': '104Z+',
               'caption': 'one channel', 'record_caption': True},
              {'kind': 'plot', 'source': 'Time History PSDs',
               'select': 'dim:force', 'caption': 'the forces',
               'record_caption': True},
              {'kind': 'plot', 'source': 'Time History PSDs',
               'caption': 'not asked'}]
    page = render_html(Report('R', blocks), run,
                       object_groups=run.object_groups)
    one, forces, plain = _payload(page)['blocks']
    expected = run.caption_of('Time History PSDs', 1,
                              specification='Specification')
    assert one['facts'] == [list(row) for row in expected.rows()]
    assert one['channels'][0]['facts'] == one['facts']
    assert dict(forces['facts'])['Channel'].count('force') == 4
    assert 'facts' not in plain
    assert '_record' not in page, 'the bookkeeping never reaches the page'


def test_a_grid_says_once_what_its_cells_share(run):
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    block = {'kind': 'plot', 'source': 'Time History PSDs',
             'specification': 'Specification', 'grid': True,
             'caption': 'Control', 'record_caption': True}
    [grid] = _payload(render_html(Report('R', [block]), run,
                                  object_groups=run.object_groups))['blocks']
    assert [label for label, _ in grid['facts']] == [
        label for label in ROWS if label not in CHANNEL_ROWS]
    cells = [cell for row in grid['rows'] for column in row['cells']
             for cell in column]
    assert len(cells) == 8
    assert all([label for label, _ in cell['facts']] == list(CHANNEL_ROWS)
               for cell in cells)
    assert dict(grid['facts'])['Estimate'].startswith('ASD of each of 8 channels')


def test_without_a_project_there_is_nothing_to_read_the_window_from(run):
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    block = {'kind': 'plot', 'source': 'Time History PSDs', 'caption': 'c',
             'record_caption': True}
    page = render_html(Report('R', [block]), dict(run.items()))
    assert 'facts' not in _payload(page)['blocks'][0]


def test_the_editor_reads_out_what_the_figure_will_carry(run):
    from visualdynamics.report import MAX_REPORT_CURVES, block_caption

    block = {'kind': 'plot', 'source': '@basis:Psd',
             'specification': '@basis:Specification', 'channel': '110Z+'}
    caption = block_caption(block, run, run.object_groups)
    assert caption == run.caption_of('Time History PSDs', 2,
                                     specification='Specification')
    # a time history's figure carries its own caption, the run, its
    # sample rate and each channel's range
    history = block_caption({'kind': 'plot', 'source': 'Time History'}, run)
    assert history == run.caption_of(
        'Time History', list(range(min(run['Time History'].num_records,
                                       MAX_REPORT_CURVES))))
    assert block_caption({'kind': 'plot', 'source': 'Specification'},
                         run) is None, 'a requirement is not a measurement'
    assert block_caption({'kind': 'text', 'text': ''}, run) is None


def test_each_grid_cell_carries_its_own_channels_values(run):
    """Every cell's rows are its own record's: the channel it draws and
    the rms of that channel against its own reference."""
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    block = {'kind': 'plot', 'source': 'Time History PSDs',
             'specification': 'Specification', 'grid': True,
             'caption': 'Control', 'record_caption': True}
    [grid] = _payload(render_html(Report('R', [block]), run,
                                  object_groups=run.object_groups))['blocks']
    psds = run['Time History PSDs']
    cells = [cell for row in grid['rows'] for column in row['cells']
             for cell in column]
    named = []
    for cell in cells:
        said = dict(cell['facts'])
        dof = said['Channel'].split(', ')[-1].split(' ')[0]
        record = list(psds.response_dof).index(dof)
        expected = dict(run.caption_of('Time History PSDs', record,
                                       specification='Specification').rows())
        assert said == {label: expected[label] for label in CHANNEL_ROWS}
        named.append(dof)
    assert len(set(named)) == len(cells) == 8, 'eight cells, eight channels'


def _caption_page(run, tmp_path, drop=None):
    """The report page of one comparison block drawn a channel at a
    time with its record's caption, the facts of channel `drop` taken
    out of the payload, written to a file."""
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    block = {'kind': 'plot', 'source': 'Time History PSDs',
             'specification': 'Specification', 'caption': 'Control',
             'record_caption': True}
    page = render_html(Report('R', [block]), run,
                       object_groups=run.object_groups)
    found = re.search(r'(<script id="data"[^>]*>)(.*?)(</script>)', page,
                      re.DOTALL)
    payload = json.loads(found.group(2))
    channels = payload['blocks'][0]['channels']
    if drop is not None:
        channels[drop].pop('facts')
    page = (page[:found.start(2)] + json.dumps(payload)
            + page[found.end(2):])
    path = tmp_path / 'caption.html'
    path.write_text(page, encoding='utf-8')
    return path, channels


def test_the_page_shows_the_picked_channels_caption_and_none_for_none(
        run, tmp_path):
    """Executed in a browser engine: picking a channel puts its caption
    block under the figure, and a channel with none clears the last
    one's rather than leaving it standing under the wrong curve."""
    import os

    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication

    path, channels = _caption_page(run, tmp_path, drop=1)
    assert len(channels) > 2
    QApplication.instance() or QApplication(['x'])
    view = web_view()
    view.resize(1000, 800)
    facts = ("(() => { const f = document.querySelector('.caption .facts');"
             " const pick = document.querySelector('select.channel');"
             " if (!pick) return null;"
             " return JSON.stringify({facts: f ? f.textContent : null,"
             "  picked: pick.selectedIndex}); })()")

    def pick(index):
        return ("(() => { const pick = document.querySelector("
                "'select.channel');"
                f" pick.selectedIndex = {index};"
                " pick.dispatchEvent(new Event('change'));"
                " const f = document.querySelector('.caption .facts');"
                " return JSON.stringify({facts: f ? f.textContent : null,"
                "  count: document.querySelectorAll('.facts').length});"
                " })()")

    seen = {}
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    try:
        seen['open'] = json.loads(web_read(view, facts))
        seen['none'] = json.loads(web_read(view, pick(1)))
        seen['third'] = json.loads(web_read(view, pick(2)))
    finally:
        web_close(view)

    def text(rows):
        return ''.join(f'{label}: {said}' for label, said in rows)

    assert seen['open']['facts'] == text(channels[0]['facts'])
    assert seen['none'] == {'facts': None, 'count': 0}, \
        'a channel with no caption shows none, not the last one'
    assert seen['third']['facts'] == text(channels[2]['facts'])
    assert seen['third']['count'] == 1


# ---- on a saved plot -----------------------------------------------------


def test_the_saved_plot_carries_the_caption_under_it(run, qt_app, tmp_path):
    import pyqtgraph as pg

    from visualdynamics.plot import plot_captioned

    psds, spec = run['Time History PSDs'], run['Specification']
    caption = run.caption_of('Time History PSDs', 0, specification=spec)
    pane = plot_captioned(psds, 0, caption=caption, specification=spec,
                          show=False)
    try:
        assert pane.windowTitle() == 'channel 1, 101Z+ acceleration'
        blocks = [item for item in pane.findChildren(pg.GraphicsLayoutWidget)[0]
                  .scene().items() if getattr(item, 'is_report_caption', False)]
        assert len(blocks) == 1 and blocks[0].caption is caption
        text = blocks[0].text
        assert '<b>Window:</b> hann, 50% overlap' in text
        assert '<b>Requirement:</b> Specification' in text
    finally:
        pane.close()
        pane.deleteLater()
        qt_app.processEvents()
    path = tmp_path / 'captioned.png'
    assert plot_captioned(psds, 3, path=str(path), show=False) == str(path)
    assert path.stat().st_size > 10_000
    with pytest.raises(ValueError, match='does not bound'):
        plot_captioned(psds, 8, specification=spec, show=False)


def test_a_long_row_is_carried_on_rather_than_run_off_the_picture(run):
    from visualdynamics.plot.report_caption import caption_html

    caption = run.caption_of('Time History PSDs', 0,
                             specification='Specification')
    lines = caption_html(caption, width=60).split('<br>')
    assert len(lines) > len(ROWS)
    assert all(len(re.sub('<[^>]+>|&nbsp;', ' ', line)) <= 64 for line in lines)


# ---- in the report editor ------------------------------------------------


def test_the_editor_pane_offers_the_caption_and_reads_it_out(run, qt_app):
    from visualdynamics.core.report import Report
    from visualdynamics.gui.report_editor import ReportEditor
    from visualdynamics.units import DEFAULT_SYSTEM

    report = Report('R', [{'kind': 'text', 'text': 'Prose'},
                          {'kind': 'plot', 'source': 'Time History PSDs',
                           'specification': 'Specification',
                           'channel': '104Z+', 'caption': 'c'}])
    editor = ReportEditor()
    try:
        editor.show_report(report, lambda: run, DEFAULT_SYSTEM,
                           lambda: run.object_groups)
        editor._operate({'op': 'select', 'at': 0})
        assert editor.record_caption_check is None, 'prose has no record'
        editor._operate({'op': 'select', 'at': 1})
        check = editor.record_caption_check
        assert check is not None and not check.isChecked()
        said = editor.record_caption_label.text()
        assert '<b>Channel:</b> channel 2, 104Z+ acceleration' in said
        assert '<b>DOF:</b> 53.15 equivalent (56 nominal)' in said
        check.setChecked(True)
        assert report.blocks[1]['record_caption'] is True
        editor.record_caption_check.setChecked(False)
        assert 'record_caption' not in report.blocks[1]
    finally:
        editor.close()
        editor.deleteLater()
        qt_app.processEvents()


# ---- dead, not finite, empty, another quantity ----------------------------


def _one(x, dim='acceleration', dof='1Z+'):
    """One channel of `x` averaged by AVERAGING, as its ASD."""
    t = np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(t, np.atleast_2d(x),
                                         response_dof=[dof],
                                         ordinate_dim=[dim])
    history.averaging = AVERAGING
    return history.compute_psds()


@pytest.fixture(scope='module')
def flat():
    return visualdynamics.Specification(
        abscissa=np.array([20.0, 400.0]), ordinate=np.full((1, 2), 1e-3),
        response_dof=['1Z+'], ordinate_dim=['acceleration**2/frequency'])


def test_a_drive_point_cross_term_is_a_cpsd():
    """A force and an acceleration at one DOF: their cross term names
    one DOF twice, and is a CPSD, not an ASD (`Psd.is_autospectrum`)."""
    t = np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(3).standard_normal((2, t.size)),
        response_dof=['1Z+', '1Z+'], ordinate_dim=['acceleration', 'force'])
    history.averaging = AVERAGING
    cpsd = history.compute_cpsds()
    cross = next(r for r in range(cpsd.num_records)
                 if not cpsd.is_autospectrum(r))
    caption = record_caption(cpsd, cross, averaging=AVERAGING)
    assert caption.kind == 'CPSD'
    assert ' against 1Z+ ' in _rows(caption)['Channel']
    auto = next(r for r in range(cpsd.num_records) if cpsd.is_autospectrum(r))
    assert record_caption(cpsd, [auto, cross]).kind == 'ASD and CPSD'


def test_a_dead_channel_is_captioned_with_nothing_measured(flat):
    caption = record_caption(_one(np.zeros(int(6 * FS))), 0,
                             averaging=AVERAGING, specification=flat,
                             unit_system=visualdynamics.SI)
    assert caption.measured_rms == 0.0 and caption.not_compared is None
    assert ' 0 m/s² rms measured against ' in _rows(caption)['Requirement']


def test_a_record_with_no_finite_line_is_not_compared_and_says_why(flat):
    x = np.random.default_rng(4).standard_normal(int(6 * FS))
    x[1000] = np.nan                      # inside the stretch averaged
    caption = record_caption(_one(x), 0, averaging=AVERAGING,
                             specification=flat)
    assert caption.measured_rms is None
    assert _rows(caption)['Requirement'] == (
        "specification, reference drawn; not compared: none of the "
        "record's lines is a finite number")


def test_a_record_of_another_quantity_is_not_compared_and_says_why(flat):
    x = np.random.default_rng(5).standard_normal(int(6 * FS))
    caption = record_caption(_one(x, dim='velocity'), 0,
                             averaging=AVERAGING, specification=flat)
    assert caption.measured_rms is None and caption.reference_rms is None
    said = _rows(caption)['Requirement']
    assert '; not compared: the specification is of acceleration' in said
    assert 'velocity' in said


def test_a_channel_the_specification_does_not_hold_says_so(flat):
    x = np.random.default_rng(6).standard_normal(int(6 * FS))
    caption = record_caption(_one(x, dof='2Z+'), 0, averaging=AVERAGING,
                             specification=flat)
    assert _rows(caption)['Requirement'].endswith(
        'not compared: the specification holds no record for 2Z+ '
        'acceleration')


def test_an_empty_selection_is_refused(measured):
    with pytest.raises(IndexError, match='at least one record'):
        record_caption(measured, [])


# ---- the survivors of tools/mutate.py, each pinned -------------------------


def test_one_frame_is_said_in_the_singular():
    t = np.arange(1024) / FS
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(12).standard_normal((1, t.size)),
        response_dof=['1Z+'], ordinate_dim=['acceleration'])
    one = Averaging(frame_length=1024, window='hann', frames=1)
    history.averaging = one
    rows = _rows(record_caption(history.compute_psds(), 0, averaging=one))
    assert rows['Estimate'] == 'ASD, single channel, linear average of 1 frame'
    two = Averaging(frame_length=512, window='hann', frames=2)
    history.averaging = two
    rows = _rows(record_caption(history.compute_psds(), 0, averaging=two))
    assert rows['Estimate'].endswith('linear average of 2 frames')


def test_the_band_the_rms_is_taken_over_is_said_low_to_high(measured):
    spec = visualdynamics.Specification(
        abscissa=np.array([20.0, 400.0]), ordinate=np.full((1, 2), 1e-3),
        response_dof=['1Z+'], ordinate_dim=['acceleration**2/frequency'])
    caption = record_caption(measured, 0, specification=spec, scale_db=0)
    low, high = caption.rms_band
    assert low < high
    assert _rows(caption)['Requirement'].endswith(
        f', {low:.4g} Hz to {high:.4g} Hz')
    assert caption.scale_db == 0
    scaled = record_caption(measured, 0, specification=spec, scale_db=6)
    assert scaled.scale_db == 6
    assert record_caption(measured, 0, scale_db=6).scale_db == 0, \
        'no specification, nothing scaled'


@pytest.mark.parametrize('centers, expected', [
    (1000.0 * 2.0 ** (np.arange(6) / 3.0), 3),
    (1000.0 * 2.0 ** (np.arange(6) / 6.0), 6),
    (1000.0 * 2.0 ** np.arange(4.0), 1),
    (1000.0 * 2.0 ** (np.arange(2) / 3.0), 3),
    (np.array([1000.0]), None),
    (np.array([0.0, 10.0]), None),
    (np.array([20.0, 10.0]), None),
    (np.array([10.0, 10.0]), None),
])
def test_bands_per_octave_are_read_from_the_centers(centers, expected):
    from visualdynamics.core.report_caption import _per_octave

    assert _per_octave(centers) == expected


def test_a_banded_spectrum_with_no_count_given_reads_it_from_its_bands(
        measured):
    banded = measured.to_octave(3)
    rows = _rows(record_caption(banded, 0))
    assert rows['Resolution'] == '1/3-octave bands'
    assert rows['Estimate'].endswith('banded to 1/3 octave')


def test_a_geometry_gives_the_global_axis_nearest_the_channel(measured,
                                                              table):
    geometry = visualdynamics.Geometry(node_id=[1], node_xyz=[[0, 0, 0]])
    channel = record_caption(measured, 0, channel_table=table,
                             geometry=geometry).channels[0]
    assert channel.nearest_axis == 'Z+'
    assert channel.place() == 'node 1 (bracket), Z+ (nearest global Z+)'


def test_a_row_with_a_blank_type_answers_when_none_matches_both():
    # no unit either, or the table infers the type from it
    untyped = ChannelTable({'channel': [7], 'node': [1], 'direction': ['Z+'],
                            'unit': [''], 'channel_type': [''],
                            'comment': ['bracket']})
    assert untyped.frame['channel_type'].iloc[0] == ''
    t = np.arange(int(2 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(13).standard_normal((1, t.size)),
        response_dof=['1Z+'], ordinate_dim=['acceleration'])
    channel = record_caption(history.compute_psds(), 0,
                             channel_table=untyped).channels[0]
    assert channel.channel == 7 and channel.description == 'bracket'


def test_a_record_of_unknown_quantity_takes_the_one_row_at_its_dof(table):
    t = np.arange(int(2 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(14).standard_normal((1, t.size)),
        response_dof=['2Z+'])
    two = ChannelTable({'channel': [5, 6], 'node': [2, 3],
                        'direction': ['Z+', 'Z+'], 'unit': ['m/s^2', 'm/s^2'],
                        'channel_type': ['acceleration', 'acceleration'],
                        'comment': ['arm', 'base']})
    channel = record_caption(history.compute_psds(), 0,
                             channel_table=two).channels[0]
    assert channel.channel == 5 and channel.description == 'arm'
    shared = ChannelTable({'channel': [5, 6], 'node': [2, 2],
                           'direction': ['Z+', 'Z+'],
                           'unit': ['m/s^2', 'N'],
                           'channel_type': ['acceleration', 'force'],
                           'comment': ['arm', 'cell']})
    assert record_caption(history.compute_psds(), 0,
                          channel_table=shared).channels[0].channel is None, \
        'two rows at the DOF and no quantity to choose between them'


def test_a_record_past_the_last_is_refused_by_its_number(measured):
    with pytest.raises(IndexError, match='record 2 is not one of the 2'):
        record_caption(measured, 2)


# ---- the degrees of freedom of a band -------------------------------------------


def test_a_band_counts_the_lines_it_sums_by_hand():
    """Four lines of 10 each, whole inside a band: 40 when they are
    independent; under a Hann window's single-frame correlation of 4/9
    at one line and 1/36 at two, 16 x 10 over 4 + 2 (3 x 4/9 + 2 x 1/36),
    23.61. Half a line's bin inside a band weighs it by half; a band no
    line with power reaches has no count."""
    from visualdynamics.core.report_caption import band_dof

    f = np.arange(1.0, 9.0)
    left, right = np.array([0.5, 4.5, 20.0]), np.array([4.5, 8.5, 30.0])
    independent = band_dof(f, np.ones(8), 10.0, [1.0], left, right)
    assert independent[:2] == pytest.approx([40.0, 40.0])
    assert np.isnan(independent[2])
    hann = band_dof(f, np.ones(8), 10.0, [1.0, 4 / 9, 1 / 36], left, right)
    assert hann[0] == pytest.approx(160.0 / (4 + 2 * (3 * 4 / 9 + 2 / 36)))
    # a band holding half of line 1 and all of line 2: weights 1/2 and 1,
    # (3/2)^2 x 10 / (1/4 + 1)
    half = band_dof(f, np.ones(8), 10.0, [1.0], [1.0], [2.5])
    assert half[0] == pytest.approx(2.25 * 10.0 / 1.25)
    # lines weighted by their power: 1, 1 and 4 give 6^2 x 10 / 18 = 20
    uneven = band_dof(f[:3], [1.0, 1.0, 4.0], 10.0, [1.0], [0.5], [3.5])
    assert uneven[0] == pytest.approx(20.0)


def test_a_banded_spectrum_carries_its_bands_dof_against_white_noise():
    """The rule against what chance gives: three hundred independent
    white-noise channels under the default Hann window, nine frames at
    half overlap, banded to 1/3 octave. Each band's count read from the
    scatter of its three hundred estimates, 2 mean^2 / variance, sits
    on the count the caption works out; the count of the lines summed as
    if independent overstates it about twice at the higher bands."""
    from visualdynamics.core.octave import bin_bounds
    from visualdynamics.core.report_caption import band_dof

    rng = np.random.default_rng(11)
    count = 300
    t = np.arange(int(6 * FS)) / FS
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((count, t.size)),
        response_dof=[f'{k + 1}Z+' for k in range(count)],
        ordinate_dim='acceleration')
    history.averaging = AVERAGING
    narrow = history.compute_psds()
    banded = narrow.to_octave(3)
    left, right = bin_bounds(banded.abscissa, banded.bandwidth)
    one = AVERAGING.statistical_dof(records=1).equivalent
    flat = np.ones(len(narrow.abscissa))
    worked = band_dof(narrow.abscissa, flat, one,
                      AVERAGING.line_correlation(), left, right)
    naive = band_dof(narrow.abscissa, flat, one, [1.0], left, right)
    values = np.real(banded.ordinate)
    scatter = 2.0 * values.mean(axis=0) ** 2 / values.var(axis=0, ddof=1)
    inside = (banded.abscissa >= 20.0) & (banded.abscissa <= 400.0)
    ratio = scatter[inside] / worked[inside]
    assert 0.9 < np.median(ratio) < 1.1
    assert ratio.min() > 0.7 and ratio.max() < 1.4
    high = banded.abscissa[inside] >= 200.0
    assert np.all(scatter[inside][high] / naive[inside][high] < 0.65)
    caption = record_caption(banded, 0, averaging=AVERAGING,
                             narrowband=narrow)
    # the caption's own count, from the record's own powers, is the
    # worked one: its highest band within the scatter of one record
    assert caption.band_dof[1] > caption.dof[0]
    assert caption.band_dof[1] == pytest.approx(np.nanmax(worked), rel=0.2)
