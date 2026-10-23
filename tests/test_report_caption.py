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
from conftest import fixture_path

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
    assert caption.pooled == 2 and caption.nominal_dof == (36,)
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
    """Banding sums lines and raises the count, which is why a banded
    spectrum carries none; the caption gives the narrowband lines' and
    says it is that."""
    banded = measured.to_octave(3)
    rows = _rows(record_caption(banded, 0, averaging=AVERAGING,
                                narrowband=measured))
    assert rows['Estimate'].endswith(', banded to 1/3 octave')
    assert rows['Resolution'] == '1/3-octave bands, from 1 Hz lines'
    assert rows['DOF'] == ('17.15 equivalent (18 nominal) per narrowband '
                           'line, before banding')
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
    assert 'rms measured against' in said and 'scaled' not in said
    scaled = record_caption(measured, 0, specification=spec, scale_db=6)
    assert _rows(scaled)['Requirement'].endswith(
        'measured scaled +6 dB for the comparison')


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


def test_an_imported_spectrum_is_its_own_run(run):
    project = visualdynamics.Project('Imported')
    project.add('Run 9 ASD', run['Time History PSDs'])
    rows = _rows(project.caption_of('Run 9 ASD'))
    assert rows['Run'] == 'Run 9 ASD'
    assert rows['Window'] == NOT_RECORDED
    with pytest.raises(TypeError, match='not a measured spectrum'):
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
    from visualdynamics.report import block_caption

    block = {'kind': 'plot', 'source': '@basis:Psd',
             'specification': '@basis:Specification', 'channel': '110Z+'}
    caption = block_caption(block, run, run.object_groups)
    assert caption == run.caption_of('Time History PSDs', 2,
                                     specification='Specification')
    assert block_caption({'kind': 'plot', 'source': 'Time History'}, run) is None
    assert block_caption({'kind': 'text', 'text': ''}, run) is None


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
