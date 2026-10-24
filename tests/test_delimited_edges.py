"""The edges of the delimited-text reader, each one a boundary a
one-character change to the reader would move.

Written from a mutation run over `io/delimited.py` (`tools/mutate.py`):
every test here fails on a change the main tests let through, and pins
the boundary by its value rather than by "something was read".
"""

from __future__ import annotations

import numpy as np
import pytest
from test_delimited_import import FS, _csv, _read, _signal, _text, _write

from visualdynamics.core.data import TimeHistory
from visualdynamics.io import delimited
from visualdynamics.io.delimited import read_delimited, text_layout


def _rows(t, *columns, fmt='.17g', delimiter=','):
    return [delimiter.join(format(v, fmt) for v in row)
            for row in zip(t, *columns, strict=True)]


# ---- reading the text --------------------------------------------------------------

def test_a_head_read_holds_exactly_the_lines_asked_for(tmp_path):
    path, _ = _csv(tmp_path, samples=200)
    head = delimited._read_lines(path, lines=5)
    assert head == _text(path).splitlines()[:5]
    assert delimited._read_lines(path, lines=10_000) == _text(path).splitlines()


def test_a_read_cut_mid_character_is_still_utf8(tmp_path):
    """The layout reads a fixed number of bytes; a two-byte character
    the cut falls inside is not a sign of a foreign encoding."""
    path = _write(tmp_path, 'Run µ1\nabcµ\n')
    cut = len('Run µ1\nabc'.encode()) + 1          # half of the second µ
    assert delimited._read_lines(path, limit=cut) == ['Run µ1']
    assert delimited._read_lines(path, limit=cut - 1) == ['Run µ1'], (
        'a whole character, the line still cut short')
    legacy = tmp_path / 'legacy.csv'
    legacy.write_bytes('Run µ1\nabc\n'.encode('cp1252'))
    assert delimited._read_lines(legacy)[0] == 'Run µ1'


def test_a_large_file_is_read_a_megabyte_at_a_time(tmp_path):
    t, a, _f = _signal(60000)
    path = _write(tmp_path, 'Time (s),A\n' + '\n'.join(_rows(t, a)))
    assert delimited._read_lines(path, lines=50000)[-1] == _rows(t, a)[49998]
    assert len(delimited._read_lines(path, lines=50000)) == 50000


def test_a_file_of_a_header_and_two_rows_is_enough(tmp_path):
    path = _write(tmp_path, 'Time (s),A\n0,1\n0.5,2\n')
    history, _notes = _read(path)
    assert history.sample_rate == 2.0 and history.ordinate.tolist() == [[1.0, 2.0]]
    with pytest.raises(ValueError, match='holds 1 data row; a time history needs two'):
        read_delimited(path, max_rows=1)
    with pytest.raises(ValueError, match='holds 0 data rows'):
        read_delimited(path, max_rows=0)
    assert read_delimited(path, max_rows=2).ordinate.shape == (1, 2)


def test_one_column_of_numbers_split_by_nothing(tmp_path):
    _t, a, _f = _signal(32)
    path = _write(tmp_path, '\n'.join(format(v, '.17g') for v in a), name='one.txt')
    layout = text_layout(path)
    assert layout.delimiter == ' ' and layout.columns == ('column 1',)
    history, _notes = _read(path, sample_rate=1000)
    assert history.ordinate.shape == (1, 32)


def test_prose_with_a_few_numbers_at_its_end_is_not_data(tmp_path):
    """Most of the last lines must be numbers for them to be the data."""
    lines = [f'note {k}: the run was fine, nothing to add' for k in range(10)]
    numbers = [f'{k},{k + 1}' for k in range(7)]
    path = _write(tmp_path, '\n'.join(lines + numbers), name='notes.txt')
    assert text_layout(path).data_line == 11, '7 of the last 12'
    words = _write(tmp_path, '\n'.join(lines + numbers[:5]), name='words.txt')
    assert not delimited.sniff(words), '5 of the last 12'
    few = _write(tmp_path, '\n'.join(lines[:3] + numbers[:3]), name='few.txt')
    assert text_layout(few).data_line == 4, '3 of 6'


def test_a_row_ending_in_empty_fields_is_not_too_long(tmp_path):
    t, a, f = _signal(8)
    rows = _rows(t, a, f)
    rows[3] += ',,'
    path = _write(tmp_path, 'Time (s),A,B\n' + '\n'.join(rows))
    history, _notes = _read(path)
    assert history.ordinate.shape == (2, 8)


def test_a_header_without_the_trailing_delimiter_its_rows_have(tmp_path):
    t, a, _f = _signal(16)
    rows = [row + ',' for row in _rows(t, a)]
    path = _write(tmp_path, 'Time (s),Accel (g)\n' + '\n'.join(rows))
    layout = text_layout(path)
    assert layout.columns == ('Time', 'Accel') and layout.names_line == 1


# ---- stated lines and delimiters ---------------------------------------------------

def test_a_stated_data_line_finds_the_names_above_it(tmp_path):
    t, a, f = _signal(16)
    text = 'Run 4\nTime (s),A (g),B\n' + '\n'.join(_rows(t, a, f))
    path = _write(tmp_path, text)
    layout = text_layout(path, data_line=3)
    assert (layout.names_line, layout.data_line) == (2, 3)
    with pytest.raises(ValueError, match='line 2 does not split into numbers'):
        text_layout(path, data_line=2, delimiter=',')
    for line in (0, 99):
        with pytest.raises(ValueError, match=f'data_line {line} is past the 18 lines'):
            text_layout(path, data_line=line)


def test_a_stated_units_line_takes_the_names_from_above_it(tmp_path):
    t, a, f = _signal(16)
    text = 'Time,A,B\nseconds,g,lbf\n' + '\n'.join(_rows(t, a, f))
    path = _write(tmp_path, text)
    layout = text_layout(path, units_line=2)
    assert (layout.names_line, layout.units_line) == (1, 2)
    assert layout.units == ('seconds', 'g', 'lbf')
    bare = text_layout(path, units_line=0)
    assert bare.units_line == 0 and bare.names_line == 2, 'the nearest line then'


def test_a_names_line_on_the_data_is_refused(tmp_path):
    path, _ = _csv(tmp_path)
    with pytest.raises(ValueError, match='names_line 2 is not above the data, which '
                                         'start on line 2'):
        text_layout(path, names_line=2)


def test_a_stated_decimal_comma_reads_without_a_delimiter_stated(tmp_path):
    t, a, _f = _signal(16)
    rows = [row.replace('.', ',') for row in _rows(t, a, fmt='.9f', delimiter=';')]
    path = _write(tmp_path, 'Time (s);A\n' + '\n'.join(rows))
    assert text_layout(path, decimal=',').delimiter == ';'


# ---- the time column -----------------------------------------------------------------

def test_an_unnamed_first_column_that_does_not_rise_evenly_is_a_channel(tmp_path):
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((2, 32))
    path = _write(tmp_path, '\n'.join(_rows(a, b)))
    assert text_layout(path).time_column is None
    uneven = np.cumsum(rng.uniform(0.5, 1.5, 32))
    jagged = _write(tmp_path, '\n'.join(_rows(uneven, b)), name='jagged.csv')
    assert text_layout(jagged).time_column is None
    falling = _write(tmp_path, '\n'.join(_rows(-np.arange(32.0), b)), name='down.csv')
    assert text_layout(falling).time_column is None
    two = _write(tmp_path, '0,1\n1,2\n', name='two.csv')
    assert text_layout(two).time_column is None, 'two rows show no step'
    lone = _write(tmp_path, '\n'.join(f'{k}' for k in range(8)), name='lone.csv')
    assert text_layout(lone).time_column is None, 'a lone column is the data'
    gap = _write(tmp_path, '0,1\n1,2\nnan,3\n3,4\n', name='gap.csv')
    assert text_layout(gap).time_column is None


def test_a_repeated_time_is_out_of_order(tmp_path):
    t, a, _f = _signal(16)
    t[6] = t[5]
    path = _write(tmp_path, 'Time (s),A\n' + '\n'.join(_rows(t, a)))
    with pytest.raises(ValueError, match='in order; sample 6'):
        read_delimited(path)


def test_rounding_is_forgiven_to_a_unit_of_the_last_digit_and_no_further(tmp_path):
    t, a, _f = _signal(400)
    path = _write(tmp_path, 'Time (s),A\n' + '\n'.join(_rows(t, a, fmt='.6f')))
    assert _read(path)[0].sample_rate == pytest.approx(FS, rel=1e-5)
    late = np.round(t, 6)
    late[200] += 3e-6                       # three units of the last digit
    path = _write(tmp_path, 'Time (s),A\n' + '\n'.join(
        _rows(late, a, fmt='.6f')), name='late.csv')
    with pytest.raises(ValueError, match='more than the digits it is written to'):
        read_delimited(path)


def test_rounding_read_from_exponent_and_quoted_times(tmp_path):
    t, a, _f = _signal(300)
    rows = [f'"{x:.5e}",{y:.17g}' for x, y in zip(t, a, strict=True)]
    path = _write(tmp_path, 'Time (s),A\n' + '\n'.join(rows))
    history, notes = _read(path)
    assert history.sample_rate == pytest.approx(FS, rel=1e-5)
    assert any('rounded when it was written' in n for n in notes)
    comma = [f'{x:.6f};{y:.6f}'.replace('.', ',') for x, y in zip(t, a, strict=True)]
    path = _write(tmp_path, 'Time (s);A\n' + '\n'.join(comma), name='comma.csv')
    assert _read(path)[0].sample_rate == pytest.approx(FS, rel=1e-5)


def test_a_stated_rate_inside_its_tolerance_is_the_rate_used(tmp_path):
    path, _ = _csv(tmp_path)
    assert _read(path, sample_rate=5120.4)[0].sample_rate == pytest.approx(5120.4)
    with pytest.raises(ValueError, match='one of them is wrong'):
        read_delimited(path, sample_rate=5121.0)


def test_a_stated_rate_is_taken_over_the_preamble(tmp_path):
    path, _ = _csv(tmp_path, header='A,B,C', preamble='Sample Rate: 10240\n')
    assert _read(path)[0].sample_rate == 10240.0
    assert _read(path, sample_rate=5120)[0].sample_rate == 5120.0
    timed, _ = _csv(tmp_path, preamble='Sample Rate: 10240\n', name='timed.csv')
    assert _read(timed, sample_rate=5120)[0].sample_rate == 5120.0, (
        'the preamble wrong, the stated rate and the column agreeing')


@pytest.mark.parametrize('rate', [0.0, -5.0, float('nan'), float('inf')])
def test_a_rate_that_is_not_a_positive_number_is_refused(tmp_path, rate):
    path, _ = _csv(tmp_path)
    with pytest.raises(ValueError, match='sample_rate must be a positive number'):
        read_delimited(path, sample_rate=rate)


@pytest.mark.parametrize('line, rate, at', [
    ('Sample Rate: 0\nfs = 2000', 2000.0, 2),
    ('dt = 0.25\n', 4.0, 1), ('Note: none\nSample Interval: 0.2 ms', 5000.0, 2)])
def test_the_preamble_rate_and_its_line(tmp_path, line, rate, at):
    path, _ = _csv(tmp_path, header='A,B,C', preamble=line + '\n')
    layout = text_layout(path)
    assert (layout.sample_rate, layout.rate_line) == (pytest.approx(rate), at)


# ---- channels ------------------------------------------------------------------------

def test_a_column_index_out_of_range_is_refused(tmp_path):
    path, _ = _csv(tmp_path)
    for key in (-1, 3):
        with pytest.raises(ValueError, match=f'columns names {key} is not a column: '
                                             'the file has 3'):
            read_delimited(path, columns=[key])
    assert _read(path, columns=[2])[0].block == ['Force']
    assert _read(path, columns=[1, 1])[0].block == ['Accel'], 'once each'


def test_the_first_missing_sample_in_the_file_is_the_one_refused(tmp_path):
    t, a, f = _signal(8)
    rows = _rows(t, a, f)
    rows[5] = rows[5].rsplit(',', 2)[0] + ',,' + rows[5].rsplit(',', 1)[1]  # A
    rows[2] = rows[2].rsplit(',', 1)[0] + ','                                # B
    path = _write(tmp_path, 'Time (s),A,B\n' + '\n'.join(rows))
    with pytest.raises(ValueError, match='line 4: B has a missing sample'):
        read_delimited(path, missing='refuse')


def test_a_record_with_no_label_is_written_by_its_number(tmp_path):
    t, a, _f = _signal(16)
    history = TimeHistory(t, a, response_dof=[''])
    target = str(tmp_path / 'out.csv')
    delimited.save(history, target)
    assert _text(target).splitlines()[1] == 'time (s),record 1'
    back, _notes = _read(target)
    assert back.block == ['record 1']
    assert np.array_equal(back.ordinate, history.ordinate)
