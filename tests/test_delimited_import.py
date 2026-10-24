"""Time histories from delimited text: the layout found, the sampling
checked, the units declared, and every refusal said where it is.

The files are written here, small and in each layout a recorder or
spreadsheet writes: a header with units in brackets, a units line under
the names, a tab file with no header at all, a semicolon file with a
decimal comma, a space-aligned file with its step in a preamble. Each
test pins the thing that would differ if the reader were wrong: the
sample rate to the digit, the unit declared, the SI value, the line
number a refusal names.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.channel_health import ChannelHealth, channel_health
from visualdynamics.core.data import TimeHistory
from visualdynamics.io import delimited
from visualdynamics.io.delimited import read_delimited, text_layout
from visualdynamics.io.stated_units import declare_stated_units, unit_problem
from visualdynamics.units import DEFAULT_SYSTEM

G = 9.80665
FS = 5120.0


def _write(tmp_path, text, name='run.csv', encoding='utf-8'):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return str(path)


def _text(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read(path, **kwargs):
    """The history and the notes the import gave, as strings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        history = read_delimited(path, **kwargs)
    notes = [str(w.message) for w in caught
             if issubclass(w.category, visualdynamics.io.ImportNote)]
    return history, notes


def _signal(samples=256, fs=FS):
    t = np.arange(samples) / fs
    return t, np.sin(2 * np.pi * 100.0 * t), 0.5 * np.cos(2 * np.pi * 40.0 * t)


def _csv(tmp_path, samples=256, header='Time (s),Accel (g),Force [lbf]',
         decimals=12, preamble='', name='run.csv', delimiter=','):
    t, a, f = _signal(samples)
    rows = [delimiter.join(f'{v:.{decimals}f}' for v in row)
            for row in zip(t, a, f, strict=True)]
    text = preamble + (header + '\n' if header else '') + '\n'.join(rows) + '\n'
    return _write(tmp_path, text, name), (t, a, f)


# ---- the layout --------------------------------------------------------------

def test_a_header_with_units_in_brackets(tmp_path):
    path, (_t, a, f) = _csv(tmp_path)
    history, notes = _read(path)
    assert isinstance(history, TimeHistory)
    assert history.sample_rate == pytest.approx(FS, rel=1e-12)
    assert history.ordinate_unit == ['g', 'lbf']
    assert history.ordinate_dim == ['acceleration', 'force']
    assert np.allclose(history.ordinate[0], a * G), 'g declared, stored in SI'
    assert np.allclose(history.ordinate[1], f * 4.4482216152605)
    assert history.response_dof == ['', ''], 'a name that is not a DOF'
    assert history.block == ['Accel', 'Force'], 'labels the record instead'
    assert history.record_label(0) == 'Accel'
    assert history.comment == ['column 2 of run.csv', 'column 3 of run.csv']
    assert notes == [], 'a clean file says nothing'


def test_the_layout_is_found_and_described(tmp_path):
    path, _ = _csv(tmp_path, preamble='Recorder export\nSample Rate: 5120 Hz\n')
    layout = text_layout(path)
    assert layout.delimiter == ',' and layout.decimal == '.'
    assert (layout.names_line, layout.units_line, layout.data_line) == (3, 0, 4)
    assert layout.columns == ('Time', 'Accel', 'Force')
    assert layout.units == ('s', 'g', 'lbf')
    assert layout.time_column == 0
    assert (layout.sample_rate, layout.rate_line) == (5120.0, 2)
    assert layout.describe(path) == (
        'run.csv: 3 columns separated by comma, names on line 3, data from '
        'line 4, 5120 samples per second stated on line 2')


def test_a_units_line_under_the_names(tmp_path):
    path, (_t, a, _f) = _csv(tmp_path, header='Time,Accel,Force\ns,g,lbf')
    layout = text_layout(path)
    assert (layout.names_line, layout.units_line, layout.data_line) == (1, 2, 3)
    history, notes = _read(path)
    assert history.ordinate_unit == ['g', 'lbf']
    assert np.allclose(history.ordinate[0], a * G)
    assert notes == []


def test_a_tab_file_with_no_header_takes_its_rising_first_column_as_time(tmp_path):
    path, (_t, a, _f) = _csv(tmp_path, header='', name='run.tsv', delimiter='\t')
    layout = text_layout(path)
    assert layout.delimiter == '\t' and layout.names_line == 0
    assert layout.columns == ('column 1', 'column 2', 'column 3')
    history, notes = _read(path)
    assert history.sample_rate == pytest.approx(FS, rel=1e-12)
    assert history.block == ['column 2', 'column 3'], 'counted from 1'
    assert history.ordinate_dim == ['unknown', 'unknown']
    assert np.array_equal(history.ordinate[0], np.round(a, 12)), 'raw numbers'
    assert any('column 1 rises in an even step' in n for n in notes)
    assert any('names no unit: read as seconds, 5120 samples' in n for n in notes)


def test_a_semicolon_file_with_a_decimal_comma(tmp_path):
    t, a, f = _signal(64)
    rows = [f'{ti:.9f};{ai:.6f};{fi:.6f}'.replace('.', ',')
            for ti, ai, fi in zip(t, a, f, strict=True)]
    path = _write(tmp_path, 'Zeit;Beschl. (g);Kraft (N)\n' + '\n'.join(rows))
    layout = text_layout(path)
    assert (layout.delimiter, layout.decimal) == (';', ',')
    history, _notes = _read(path, time_column='Zeit')
    assert history.sample_rate == pytest.approx(FS, rel=1e-6)
    assert np.allclose(history.ordinate[0], a * G, atol=1e-5)
    assert history.ordinate_unit == ['g', 'N']


def test_a_space_aligned_file_with_its_step_in_the_preamble(tmp_path):
    _t, a, f = _signal(100, fs=2000.0)
    rows = [f'  {ai: .6e}   {fi: .6e}' for ai, fi in zip(a, f, strict=True)]
    path = _write(tmp_path, '% dt = 0.5 ms\n' + '\n'.join(rows), name='run.txt')
    layout = text_layout(path)
    assert layout.delimiter == ' ' and layout.sample_rate == pytest.approx(2000.0)
    history, _notes = _read(path)
    assert history.num_records == 2, 'no time column, so both are channels'
    assert history.abscissa[1] == pytest.approx(1 / 2000.0)


@pytest.mark.parametrize('line, rate', [
    ('Sample Rate: 5120', 5120.0), ('fs = 5.12 kHz', 5120.0),
    ('Sampling Frequency (Hz),5120', 5120.0), ('Delta_X\t1.953125E-4', 5120.0),
    ('Sample Interval: 0.2 ms', 5000.0)])
def test_a_preamble_states_its_rate_in_the_ways_exports_write_it(tmp_path, line, rate):
    path, _ = _csv(tmp_path, header='A,B,C', preamble=line + '\n')
    assert text_layout(path).sample_rate == pytest.approx(rate)


def test_a_trailing_delimiter_and_trailing_empty_rows_are_not_data(tmp_path):
    t, a, _f = _signal(32)
    rows = [f'{ti:.17g},{ai:.17g},' for ti, ai in zip(t, a, strict=True)]
    path = _write(tmp_path, 'Time (s),Accel (g),\n' + '\n'.join(rows) + '\n,,\n,,\n')
    history, notes = _read(path)
    assert history.ordinate.shape == (1, 32), 'one channel, 32 samples'
    assert notes == []


def test_a_quoted_name_keeps_its_comma(tmp_path):
    path, _ = _csv(tmp_path, header='Time,"Accel, X (g)",Force')
    history, _notes = _read(path)
    assert history.block == ['Accel, X', 'Force']
    assert history.ordinate_unit == ['g', None]


def test_a_byte_order_mark_and_a_windows_encoding_both_read(tmp_path):
    path, _ = _csv(tmp_path, header='﻿Time,Strain (microstrain),B')
    assert text_layout(path).columns[0] == 'Time'
    t, a, _f = _signal(16)
    text = 'Time,Disp (µm),B\n' + '\n'.join(f'{x:.17g},{y:.17g},0.5' for x, y in zip(t, a,
                                                                              strict=True))
    legacy = _write(tmp_path, text, name='legacy.csv', encoding='cp1252')
    history, _notes = _read(legacy)
    assert history.ordinate_dim[0] == 'length'
    assert np.allclose(history.ordinate[0], a * 1e-6)


def test_max_rows_reads_the_head_of_the_file(tmp_path):
    path, _ = _csv(tmp_path, samples=500)
    history, _notes = _read(path, max_rows=40)
    assert history.ordinate.shape == (2, 40)


# ---- the sampling --------------------------------------------------------------

def test_a_time_column_written_to_six_places_reads_as_its_even_step(tmp_path):
    """1/5120 s written to six places steps 0.000195, then 0.000196: the
    times lie within a unit of the last digit of an even step, which is
    taken, and the import says so."""
    path, _ = _csv(tmp_path, samples=1000, decimals=6)
    history, notes = _read(path)
    assert history.sample_rate == pytest.approx(FS, rel=1e-6)
    assert np.allclose(np.diff(history.abscissa), 1 / history.sample_rate,
                       rtol=1e-9), 'even, not as written'
    [note] = [n for n in notes if 'rounded' in n]
    assert note == ('run.csv: the time column Time was rounded when it was '
                    'written, and its times sit up to 5.03e-07 s off an even '
                    'step, within that rounding: read as an even 5120 samples '
                    'per second')


def test_a_time_column_uneven_beyond_its_digits_is_refused(tmp_path):
    t, a, _f = _signal(64)
    t = t + np.where(np.arange(64) == 30, 3e-5, 0.0)       # one late sample
    path = _write(tmp_path, 'Time,A\n' + '\n'.join(
        f'{x:.9f},{y:.6f}' for x, y in zip(t, a, strict=True)))
    with pytest.raises(ValueError, match='evenly spaced') as refusal:
        read_delimited(path)
    assert 'time_column=False and the sample_rate' in str(refusal.value)


def test_a_time_column_out_of_order_is_refused_by_sample(tmp_path):
    t, a, _f = _signal(16)
    t[[5, 6]] = t[[6, 5]]
    path = _write(tmp_path, 'Time,A\n' + '\n'.join(
        f'{x:.17g},{y:.17g}' for x, y in zip(t, a, strict=True)))
    with pytest.raises(ValueError, match='in order; sample 6'):
        read_delimited(path)


def test_a_stated_rate_that_disagrees_with_the_time_column_is_refused(tmp_path):
    path, _ = _csv(tmp_path)
    with pytest.raises(ValueError, match=r'steps at 0\.000195313 s \(5120 per '
                                         r'second\) and the stated sample rate '
                                         r'is 5000; one of them is wrong'):
        read_delimited(path, sample_rate=5000)
    history, _notes = _read(path, sample_rate=5120)
    assert history.sample_rate == 5120.0


def test_a_preamble_rate_that_disagrees_with_the_time_column_is_refused(tmp_path):
    path, _ = _csv(tmp_path, preamble='Sample Rate: 10240\n')
    with pytest.raises(ValueError, match='the rate stated on line 1 is 10240'):
        read_delimited(path)


def test_neither_a_time_column_nor_a_rate_is_refused_naming_the_columns(tmp_path):
    path, _ = _csv(tmp_path, header='A,B,C')
    with pytest.raises(ValueError, match=r"neither a time column nor a stated "
                                         r"sample rate.*\['A', 'B', 'C'\]"):
        read_delimited(path)
    history, _notes = _read(path, sample_rate=FS)
    assert history.num_records == 3
    history, _notes = _read(path, time_column='A')
    assert history.num_records == 2 and history.sample_rate == pytest.approx(FS)


def test_the_time_unit_scales_the_clock(tmp_path):
    t, a, _f = _signal(32)
    path = _write(tmp_path, 'Time (ms),A\n' + '\n'.join(
        f'{x * 1000:.17g},{y:.17g}' for x, y in zip(t, a, strict=True)))
    history, notes = _read(path)
    assert history.sample_rate == pytest.approx(FS)
    assert not [n for n in notes if 'seconds' in n], 'the unit was given'
    bare, _ = _csv(tmp_path, header='Time,A,B', name='bare.csv')
    assert _read(bare, time_unit='ms')[0].sample_rate == pytest.approx(FS * 1000)
    with pytest.raises(ValueError, match="in 'g', which is not a unit of time"):
        read_delimited(bare, time_unit='g')


def test_a_row_with_no_time_is_refused_by_line(tmp_path):
    path, _ = _csv(tmp_path, samples=8)
    lines = _text(path).splitlines()
    lines[4] = ',' + lines[4].split(',', 1)[1]
    path = _write(tmp_path, '\n'.join(lines))
    with pytest.raises(ValueError, match='line 5: the time column Time has no value'):
        read_delimited(path)


# ---- missing samples, dead channels and malformed rows -------------------------

def test_a_missing_sample_is_kept_as_nan_and_named(tmp_path):
    path, _ = _csv(tmp_path, samples=16)
    lines = _text(path).splitlines()
    for at in (6, 9):
        lines[at] = lines[at].rsplit(',', 1)[0] + ','           # Force empty
    lines[11] = lines[11].rsplit(',', 1)[0] + ',NaN'
    path = _write(tmp_path, '\n'.join(lines))
    history, notes = _read(path)
    assert np.isnan(history.ordinate[1]).sum() == 3
    assert not np.isnan(history.ordinate[0]).any()
    [note] = [n for n in notes if 'missing sample' in n]
    assert note == ('run.csv: Force has 3 missing samples of 16 (the first at '
                    'sample 6, 0.000976562 s), kept as NaN')
    with pytest.raises(ValueError, match='line 7: Force has a missing sample'):
        read_delimited(path, missing='refuse')


def test_a_short_row_is_padded_and_noted(tmp_path):
    path, _ = _csv(tmp_path, samples=8)
    lines = _text(path).splitlines()
    lines[3] = lines[3].rsplit(',', 1)[0]
    path = _write(tmp_path, '\n'.join(lines))
    history, notes = _read(path)
    assert np.isnan(history.ordinate[1, 2])
    assert any('1 row shorter than the header (the first on line 4)' in n
               for n in notes)


def test_a_long_row_is_refused_by_line(tmp_path):
    path, _ = _csv(tmp_path, samples=8)
    lines = _text(path).splitlines()
    lines[5] += ',7.0'
    path = _write(tmp_path, '\n'.join(lines))
    with pytest.raises(ValueError, match='line 6: 4 fields where the data above '
                                         'it have 3'):
        read_delimited(path)


def test_a_cell_that_is_not_a_number_is_refused_by_line_and_column(tmp_path):
    path, _ = _csv(tmp_path, samples=8)
    lines = _text(path).splitlines()
    lines[4] = lines[4].rsplit(',', 1)[0] + ',over'
    path = _write(tmp_path, '\n'.join(lines))
    with pytest.raises(ValueError, match="line 5: Force holds 'over', which is "
                                         "not a number$"):
        read_delimited(path)


def test_a_clock_time_column_is_refused_with_the_way_round_it(tmp_path):
    _t, a, _f = _signal(8)
    rows = [f'2026-10-23 12:00:{i:02d}.000,{y:.17g}' for i, y in enumerate(a)]
    path = _write(tmp_path, 'Stamp,A\n' + '\n'.join(rows))
    with pytest.raises(ValueError, match=r"not read as delimited text: .* dates "
                                         r"or clock times \('2026-10-23'\)"):
        read_delimited(path)
    # one stamp among numbers is a cell in the data, refused where it is
    t, a, _f = _signal(8)
    rows = [f'{x:.17g},{y:.17g}' for x, y in zip(t, a, strict=True)]
    rows[3] = '2026-10-23 12:00:03,' + rows[3].split(',')[1]
    mixed = _write(tmp_path, 'Time (s),A\n' + '\n'.join(rows), name='mixed.csv')
    with pytest.raises(ValueError, match=r"line 5: Time holds '2026-10-23 "
                                         r"12:00:03', which is not a number "
                                         r"\(a date or clock time"):
        read_delimited(mixed)


def test_a_dead_channel_is_named_with_its_value_as_written(tmp_path):
    t, a, _f = _signal(32)
    path = _write(tmp_path, 'Time,A (g),B (g),C\n' + '\n'.join(
        f'{x:.17g},{y:.17g},1.5,' for x, y in zip(t, a, strict=True)))
    history, notes = _read(path)
    assert history.ordinate[1, 0] == pytest.approx(1.5 * G)
    assert ('run.csv: B is constant at 1.5 g over every present sample: a dead '
            'or unconnected channel') in notes
    assert 'run.csv: C holds no values: all 32 samples are missing' in notes
    assert not [n for n in notes if n.startswith('run.csv: A ')]


# ---- units, quantities, names and columns --------------------------------------

def test_a_header_unit_that_cannot_be_declared_is_noted_and_left_raw(tmp_path):
    path, (_t, a, _f) = _csv(tmp_path, header='Time,Accel (Gs),Force (bananas)')
    history, notes = _read(path)
    assert history.ordinate_dim == ['unknown', 'unknown']
    assert np.allclose(history.ordinate[0], a), 'never scaled by a guess'
    assert ("run.csv: Accel: the unit 'Gs' in the header reads as a time, which "
            "no channel of a time history measures; left undeclared, holding "
            "the file's numbers") in notes
    assert any("'bananas' in the header is not a unit Visual Dynamics reads" in n
               for n in notes)


def test_stated_units_are_taken_over_the_header(tmp_path):
    path, (_t, a, _f) = _csv(tmp_path, header='Time (s),Accel (Gs),Force')
    history, notes = _read(path, units={'Accel': 'g', 2: 'N'})
    assert history.ordinate_unit == ['g', 'N']
    assert np.allclose(history.ordinate[0], a * G)
    assert notes == []
    assert _read(path, units='m/s**2')[0].ordinate_unit == ['m/s**2'] * 2
    assert _read(path, units=['g', None])[0].ordinate_unit == ['g', None]


def test_a_stated_unit_that_cannot_be_declared_is_refused(tmp_path):
    path, _ = _csv(tmp_path)
    with pytest.raises(ValueError, match="Accel is stated in 'furlong/s', which "
                                         "is a velocity unit, and the channel is "
                                         "stated to be an acceleration"):
        read_delimited(path, units={'Accel': 'furlong/s'}, quantities={'Accel':
                                                                      'acceleration'})
    with pytest.raises(ValueError, match="'zorks', which is not a unit"):
        read_delimited(path, units={'Force': 'zorks'})
    with pytest.raises(ValueError, match="units names 'Nope'"):
        read_delimited(path, units={'Nope': 'g'})
    with pytest.raises(ValueError, match='units has 3 entries for 2 channels'):
        read_delimited(path, units=['g', 'g', 'g'])


def test_a_quantity_claims_without_scaling_and_must_agree_with_the_unit(tmp_path):
    path, (_t, a, _f) = _csv(tmp_path, header='Time,Accel,Force (lbf)')
    history, _notes = _read(path, quantities={'Accel': 'acceleration'})
    assert history.ordinate_dim[0] == 'unknown'
    assert history.dimension_hint[0] == 'acceleration'
    assert np.allclose(history.ordinate[0], a), 'a claim scales nothing'
    assert _read(path, quantities=['displacement', None])[0].dimension_hint[0] == 'length'
    with pytest.raises(ValueError, match="Force is headed in 'lbf', which is a "
                                         "force unit, and the channel is stated "
                                         "to be an acceleration"):
        read_delimited(path, quantities='acceleration')
    with pytest.raises(ValueError, match="stated as 'happiness', which is not a "
                                         "quantity"):
        read_delimited(path, quantities={'Accel': 'happiness'})


def test_names_that_are_dofs_become_the_response(tmp_path):
    path, _ = _csv(tmp_path)
    history, _notes = _read(path, names=['101z', 'drive'])
    assert history.response_dof == ['101Z+', '']
    assert history.block == ['', 'drive']
    assert history.record_label(0) == '101Z+'
    plain, _ = _csv(tmp_path, header='Time,101X+,102Y-', name='dofs.csv')
    named, _notes = _read(plain)
    assert named.response_dof == ['101X+', '102Y-'] and named.block is None
    with pytest.raises(ValueError, match='names has 1 entries for 2 channels'):
        read_delimited(path, names=['x'])


def test_columns_choose_the_channels_in_the_order_asked(tmp_path):
    path, (_t, a, _f) = _csv(tmp_path)
    history, _notes = _read(path, columns=['Force', 1])
    assert history.block == ['Force', 'Accel']
    assert np.allclose(history.ordinate[1], a * G)
    with pytest.raises(ValueError, match="'Time', which is the time column"):
        read_delimited(path, columns=['Time'])
    with pytest.raises(ValueError, match="columns names 'Speed' is not a column"):
        read_delimited(path, columns=['Speed'])
    with pytest.raises(ValueError, match='columns names 9 is not a column'):
        read_delimited(path, columns=[9])


def test_the_arguments_are_checked_before_the_file_is_read(tmp_path):
    path, _ = _csv(tmp_path)
    for kwargs, said in (({'missing': 'fill'}, "missing must be 'keep'"),
                         ({'sample_rate': -1}, 'sample_rate must be a positive'),
                         ({'delimiter': ':'}, 'delimiter must be one of'),
                         ({'decimal': ',', 'delimiter': ','}, 'both the delimiter'),
                         ({'time_column': True}, 'not True'),
                         ({'names_line': 9}, 'names_line 9 is not above the data')):
        with pytest.raises(ValueError, match=said):
            read_delimited(path, **kwargs)


# ---- through the registry -------------------------------------------------------

def test_import_file_reads_it_and_a_project_journals_the_options(tmp_path):
    path, _ = _csv(tmp_path)
    assert isinstance(visualdynamics.import_file(path), TimeHistory)
    project = visualdynamics.Project()
    [added] = project.import_file(path, columns=['Accel'])
    assert project[added].block == ['Accel']
    assert visualdynamics.io.importers()[-1].name == 'delimited', (
        'tried last: a text file another reader knows by its content is its')


def test_the_sniffer_takes_numbers_in_columns_and_nothing_else(tmp_path):
    path, _ = _csv(tmp_path, name='run.txt')
    assert delimited.sniff(path)
    prose = _write(tmp_path, 'a specification, in words\nand more words\n',
                   name='notes.txt')
    assert not delimited.sniff(prose)
    assert not delimited.sniff(tmp_path)
    assert not delimited.sniff(tmp_path / 'absent.csv')
    other = _write(tmp_path, _text(path), name='run.json')
    assert not delimited.sniff(other), 'only the suffixes it names'
    with pytest.raises(ValueError, match='No importer recognizes'):
        visualdynamics.import_file(prose)


def test_a_history_written_as_csv_reads_back_the_same(tmp_path):
    t, a, f = _signal(300)
    history = TimeHistory(t, np.vstack([a * G, f * 4.4482216152605, f]),
                          response_dof=['101Z+', '', ''],
                          block=['', 'drive', 'raw'],
                          ordinate_unit=[None, None, None])
    history.define_units({0: 'g', 1: 'lbf'})
    target = str(tmp_path / 'out.csv')
    visualdynamics.io.export_file(history, target, unit_system=DEFAULT_SYSTEM)
    header = _text(target).splitlines()[1]
    assert header == 'time (s),101Z+ (g),drive (lbf),raw'
    back, notes = _read(target)
    assert back.response_dof == ['101Z+', '', ''] and back.block == ['', 'drive', 'raw']
    assert back.ordinate_unit == ['g', 'lbf', None]
    assert np.array_equal(back.ordinate, history.ordinate)
    assert back.sample_rate == history.sample_rate
    assert notes == []
    stored = str(tmp_path / 'si.csv')
    visualdynamics.io.export_file(history, stored)
    assert _text(stored).splitlines()[1] == \
        'time (s),101Z+ (m/s**2),drive (N),raw'
    from visualdynamics.io.exporters import units_written

    assert units_written('delimited', DEFAULT_SYSTEM) == DEFAULT_SYSTEM.name


# ---- the shared helpers -------------------------------------------------------

def test_channel_health_by_hand():
    values = np.array([[1.0, 2.0, 3.0, 4.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [1.0, np.nan, 3.0, np.inf],
                       [np.nan] * 4,
                       [2.0, np.nan, 2.0, 2.0]])
    found = channel_health(values)
    assert [f.record for f in found] == [1, 2, 3, 4], 'the clean one says nothing'
    assert found[0] == ChannelHealth(1, 4, 0, None, 0.0)
    assert found[1] == ChannelHealth(2, 4, 2, 1, None), 'infinite counts'
    assert found[2].empty and not found[2].dead
    assert found[3].dead and found[3].missing == 1
    assert found[3].describe('C', [0.0, 0.5, 1.0, 1.5], 'g') == (
        'C has 1 missing sample of 4 (the first at sample 2, 0.5 s), kept as '
        'NaN; C is constant at 2 g over every present sample: a dead or '
        'unconnected channel')
    assert channel_health(np.array([[5.0]])) == [], 'one sample cannot vary'
    assert channel_health(np.array([[5.0, 5.0]]))[0].dead, 'two can'


def test_one_rule_for_stated_units():
    assert unit_problem('g') is None
    assert unit_problem('Gs') == ('reads as a time, which no channel of a time '
                                  'history measures')
    assert unit_problem('lbf', 'acceleration') == (
        'is a force unit, and the channel is stated to be an acceleration')
    assert unit_problem('mm', 'length') is None
    history = TimeHistory(np.arange(4.0), np.ones((3, 4)), response_dof=['', '', ''])
    refused = declare_stated_units(history, ['g', 'Gs', None])
    assert refused == [(1, 'Gs', unit_problem('Gs'))]
    assert history.ordinate_unit == ['g', None, None]


def test_sep005_follows_the_same_rule():
    """The SEP 005 converter declares its units through the same helper:
    'Gs' is kept as the claim, not declared as gigaseconds."""
    history = visualdynamics.from_sep005({'data': np.ones((2, 8)), 'fs': 8.0,
                                          'unit_str': ['g', 'Gs']})
    assert history.ordinate_unit == ['g', None]
    assert history.dimension_hint[1] == 'Gs'
