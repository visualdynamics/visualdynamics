"""Time histories from delimited text: CSV, TSV and the column exports
of controllers, recorders and spreadsheets.

    history = visualdynamics.import_file('run.csv')
    history = visualdynamics.import_file('run.txt', sample_rate=5120,
                                         units='g')

The commonest way a recorded test leaves the machine that recorded it is
a text file with one column per channel. The file is laid out the way
its program chose, so the reader finds the layout first
(`text_layout`) and anything it finds can be stated instead:

- **The delimiter**: tab, semicolon, comma, pipe or a run of spaces,
  whichever splits the data rows into the same number of numeric fields.
  A semicolon file whose numbers carry a decimal comma (``0,25``) is read
  with the comma as the decimal point.
- **The header**: the line of column names directly above the data, and
  a line of units under it when every field of that line reads as a
  unit. A unit can also ride in the name, as ``Accel (g)`` or
  ``Accel [g]``. Lines above the header (a preamble) are read only for a
  sample rate: ``Sample Rate: 5120``, ``fs = 5120 Hz`` or a step such as
  ``dt, 1.953125e-4``.
- **The time column**: a column named for time (``Time``, ``t``,
  ``Seconds``, ``X_Value``, with or without a unit). In a file with no
  header, a first column that rises in an even step is taken as time,
  and the import says so.
- **The sample rate**: from the time column's step, the preamble, or
  `sample_rate`. When two of them are given and disagree the import is
  refused: one of them is wrong, and the reader cannot tell which.

A time column without a unit is read as seconds, and the import says
so with the sample rate it gives. A time column written to a fixed
number of decimal places steps unevenly by up to a unit of its last
digit (1/5120 s written to six places alternates between 0.000195 and
0.000196); when every time lies within that rounding of an even step,
the even step is taken and the import says so. A time column that is
uneven beyond its rounding, or out of order, is refused with the reason
`TimeHistory`'s own spectra would give.

**Channels.** Every column but the time column is a channel, or the
ones `columns` names. A channel's name is its header, or ``column 3``
counted from 1 as a spreadsheet counts. A name that is a DOF
(``101X+``) becomes the record's response DOF; any other name labels the
record (its block) and leaves its DOF empty, to be set in the channel
table. `names` renames the channels.

**Units.** A unit from the header is declared when it reads as a
quantity a channel measures, converting the values to SI; one that does
not (``counts``, ``Gs``, which reads as gigaseconds) leaves the channel
undeclared, holding the file's numbers, and the import says why. A unit
stated in `units` that cannot be declared is refused. `quantities`
claims what each channel measures without giving its unit, which labels
it and narrows the units offered; a unit that contradicts a claimed
quantity is refused. Values are never scaled by a guess.

**Missing samples and dead channels.** An empty cell, ``NaN`` or
``N/A`` is a missing sample, kept as NaN, and the import names each
channel that has one and where the first falls; with
``missing='refuse'`` the first one refuses the file. A row shorter than
the header is padded as missing and noted. A channel whose every
sample is one value is noted as dead (`core.channel_health`). Neither
drops the channel: what the file holds is imported, and the notes say
what is wrong with it.

**Refusals** say what was found and where: a cell that is not a number
(by line, as an editor numbers it, and column name), a row longer than
the header, a file with neither a time column nor a sample rate, two
sample rates that disagree, a time column that is not even, a column
named that is not there.

Every note is an `ImportNote` warning, which the window shows after the
import and a script sees as a warning. `save` writes a time history
back out as a CSV with one column per record and its unit in the
header, which this reader takes back in.
"""

from __future__ import annotations

import csv
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .notes import ImportNote
from .sniffing import text_head
from .stated_units import CHANNEL_QUANTITIES, declare_stated_units, unit_problem

# Asked for by the course audit (2026-10-23, P11): a reader's own
# recorded data most often arrives as text, and every workflow it walks
# starts with an import. Built on review guidance from the previous
# tranche: refusals in the API, notes for NaN and dead channels, unit and
# quantity checks, and one shared helper for stated units.

#: the suffixes `sniff` looks inside; any other file is not delimited
#: text to this reader, whatever it holds (`format='delimited'` reads it)
SUFFIXES = ('.csv', '.tsv', '.txt', '.tab', '.asc', '.prn')

#: the delimiters tried, in order. A semicolon is tried before the comma
#: because a semicolon file is the one that uses a decimal comma, and a
#: comma split of it would also come out even. ' ' is a run of spaces.
DELIMITERS = ('\t', ';', ',', '|', ' ')

#: what each delimiter is called where a person reads it
DELIMITER_NAMES = {'\t': 'tab', ';': 'semicolon', ',': 'comma', '|': 'pipe',
                   ' ': 'spaces'}

#: cell text read as a missing sample, compared lower-cased
MISSING = frozenset({'', 'nan', '+nan', '-nan', 'na', 'n/a', '#n/a', 'null'})

#: column names read as the time column, compared lower-cased with the
#: unit removed. 'x_value' is the abscissa column of a common
#: instrument-software text export.
TIME_NAMES = frozenset({'time', 't', 'times', 'sec', 'secs', 'second',
                        'seconds', 'elapsed', 'elapsed time', 'x_value',
                        'time stamp', 'relative time'})

#: how far a time column's rate may sit from a stated one, relative,
#: and still be the same rate: a step printed to six significant
#: figures is good to about one part in a million
RATE_TOLERANCE = 1e-4

#: the lines `text_layout` reads to find the layout
HEAD_LINES = 400

#: preamble keys (lower-cased, letters only) naming a sample rate, and
#: those naming a step
_RATE_KEYS = frozenset({'samplerate', 'samplingrate', 'samplingfrequency',
                        'samplefrequency', 'samplespersecond', 'fs', 'sps',
                        'rate', 'scanrate'})
_STEP_KEYS = frozenset({'dt', 'deltat', 'deltax', 'timestep', 'sampleinterval',
                        'samplinginterval', 'increment', 'timeincrement',
                        'sampleperiod', 'samplingperiod'})

_NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
_DECIMAL_COMMA = re.compile(r'^\s*[-+]?\d+,\d+(?:[eE][-+]?\d+)?\s*$')
_UNIT_IN_NAME = re.compile(r'^(.*?)\s*[\(\[]\s*([^\)\]]*?)\s*[\)\]]\s*$')
_DATE = re.compile(r'\d{1,4}[-/:]\d{1,2}[-/:]\d{1,4}')
_KEY_VALUE = re.compile(
    r'^\W*([A-Za-z][A-Za-z _\-.]*?)\s*(?:\(\s*([^)]*)\)|\[\s*([^\]]*)\])?\s*'
    r'[:=,;\t|]?\s*(' + _NUMBER.pattern + r')\s*([A-Za-z/µ]*)')


@dataclass(frozen=True)
class TextLayout:
    """How a delimited text file is laid out, found by `text_layout`.

    Line numbers count from 1, as an editor numbers them; 0 means the
    file has no such line.

    Attributes
    ----------
    delimiter : str
        What separates the fields: ``'\\t'``, ``';'``, ``','``, ``'|'``,
        or ``' '`` for a run of spaces.
    decimal : str
        The decimal point, ``'.'`` or ``','``.
    names_line : int
        The line of column names.
    units_line : int
        The line of units under the names.
    data_line : int
        The first line of data.
    columns : tuple of str
        Every column's name, the time column included: the header's,
        or ``column N`` counted from 1.
    units : tuple of str or None
        Each column's unit as the file wrote it, or None.
    time_column : int or None
        Which column (from 0) holds time, or None.
    sample_rate : float or None
        A sample rate the preamble states, per second.
    rate_line : int
        The line it was stated on.
    """

    delimiter: str
    decimal: str
    names_line: int
    units_line: int
    data_line: int
    columns: tuple[str, ...]
    units: tuple[str | None, ...]
    time_column: int | None
    sample_rate: float | None = None
    rate_line: int = 0

    def describe(self, path: str | os.PathLike | None = None) -> str:
        """The layout in one line, for a dialog's header."""
        named = DELIMITER_NAMES.get(self.delimiter, repr(self.delimiter))
        said = [f'{len(self.columns)} columns separated by {named}']
        if self.decimal == ',':
            said.append('decimal comma')
        if self.names_line:
            said.append(f'names on line {self.names_line}')
        if self.units_line:
            said.append(f'units on line {self.units_line}')
        said.append(f'data from line {self.data_line}')
        if self.sample_rate is not None:
            said.append(f'{self.sample_rate:g} samples per second stated on '
                        f'line {self.rate_line}')
        text = ', '.join(said)
        return f'{os.path.basename(str(path))}: {text}' if path else text


# ---- reading the text ------------------------------------------------------

def _read_lines(path: str | os.PathLike, limit: int | None = None,
                lines: int | None = None) -> list[str]:
    """The file's lines, decoded as UTF-8 (a byte-order mark dropped) or,
    failing that, as Windows-1252, the two encodings text exports use.

    `limit` reads no more than that many bytes; `lines` reads only as
    far as it takes to hold that many lines, so a preview of a large
    file reads its head and not the whole of it.
    """
    with open(path, 'rb') as f:
        if lines is not None:
            chunks, count = [], 0
            while count <= lines:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
                count += chunk.count(b'\n')
            raw = b''.join(chunks)
            if count > lines:
                # whole lines only: the read ends after the line asked for
                end = -1
                for _ in range(lines):
                    end = raw.index(b'\n', end + 1)
                raw = raw[:end + 1]
        else:
            raw = f.read() if limit is None else f.read(limit)
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        if limit is not None and len(raw) == limit and error.start >= limit - 3:
            # a read cut mid-character is not a wrong encoding
            text = raw[:error.start].decode('utf-8-sig')
        else:
            text = raw.decode('cp1252', errors='replace')
    out = text.splitlines()
    if limit is not None and out and len(raw) == limit:
        out = out[:-1]                      # the last line may be cut short
    return out


def _split(line: str, delimiter: str) -> list[str]:
    """One line's fields: quoted fields kept whole, a run of spaces one
    delimiter, and the fields stripped."""
    if delimiter == ' ':
        return line.split()
    return [field.strip() for field in
            next(csv.reader([line], delimiter=delimiter, quotechar='"'))]


def _blank(line: str, delimiter: str) -> bool:
    """Whether a line holds nothing but delimiters and spaces: an empty
    row, which is no more a sample than an empty line is."""
    return not line.replace(delimiter, '').replace('"', '').strip()


def _number(text: str, decimal: str) -> float | None:
    """A cell's value; NaN for a missing sample; None for text that is not
    a number."""
    text = text.strip()
    if text.lower() in MISSING:
        return float('nan')
    if decimal == ',':
        text = text.replace(',', '.')
    try:
        return float(text)
    except ValueError:
        return None


def _numeric(fields: Sequence[str], decimal: str) -> bool:
    """Whether every field is a number or a missing sample, and at least
    one is a number."""
    values = [_number(field, decimal) for field in fields]
    return (all(v is not None for v in values)
            and not all(np.isnan(v) for v in values))


def _decimal_for(lines: Sequence[str], delimiter: str) -> str:
    """',' when a file split by anything but a comma writes its numbers
    with a decimal comma, '.' otherwise."""
    if delimiter == ',':
        return '.'
    for line in lines[-20:]:
        if any(_DECIMAL_COMMA.match(field) for field in _split(line, delimiter)):
            return ','
    return '.'


def _data_start(lines: Sequence[str], delimiter: str, decimal: str
                ) -> tuple[int, int] | None:
    """(index of the first data line, fields per line) for a delimiter,
    or None when the delimiter does not split the data evenly.

    The field count is the commonest among the numeric lines that end
    the head, and most of those lines must be numeric for the delimiter
    to be the file's. The data then start at the first two consecutive
    numeric lines with that count, read from the top: a header and a
    preamble hold words, and a stray word further down is a cell in the
    data, which the read refuses by line, not a second header that would
    quietly drop every row above it.
    """
    body = [(i, line) for i, line in enumerate(lines)
            if not _blank(line, delimiter)]
    tail = [_split(line, delimiter) for _i, line in body[-12:]]
    numeric = [len(parts) for parts in tail if _numeric(parts, decimal)]
    if len(numeric) < max(2, (len(tail) + 1) // 2):
        return None
    fields = max(set(numeric), key=numeric.count)
    if fields < (1 if delimiter == ' ' else 2):
        return None
    run = []
    for i, line in body:
        parts = _split(line, delimiter)
        if len(parts) == fields and _numeric(parts, decimal):
            run.append(i)
            if len(run) == 2:
                return run[0], fields
        else:
            run = []
    return None


def _unit_like(text: str) -> bool:
    """Whether a header field reads as a unit rather than a name."""
    from ..units import dimension_of

    text = text.strip()
    if not text or not re.search(r'[A-Za-z%µ°]', text):
        return False
    text = text.strip('()[]')
    return dimension_of(text) is not None and text.lower() not in TIME_NAMES - {'s'}


def _name_and_unit(header: str) -> tuple[str, str | None]:
    """'Accel (g)' -> ('Accel', 'g'); a header with no bracketed unit is
    all name."""
    match = _UNIT_IN_NAME.match(header.strip())
    if match and match.group(2):
        return match.group(1).strip(), match.group(2).strip()
    return header.strip(), None


def _preamble_rate(lines: Sequence[str], end: int) -> tuple[float | None, int]:
    """(samples per second, line number) a preamble states, or (None, 0).

    A line whose key names a sample rate (``Sample Rate: 5120``) gives it
    directly; one naming a step (``dt = 1.953125e-4 s``) gives its
    inverse. A unit after the value is honored (``kHz``, ``ms``).
    """
    from ..units import dimension_of, si_factor

    for i in range(end):
        match = _KEY_VALUE.match(lines[i])
        if not match:
            continue
        key = re.sub(r'[^a-z]', '', match.group(1).lower())
        unit = (match.group(2) or match.group(3) or match.group(5) or '').strip()
        value = float(match.group(4))
        if not value > 0:
            continue
        if key in _RATE_KEYS:
            if unit and dimension_of(unit) == 'frequency':
                value *= si_factor(unit, 'frequency')
            return value, i + 1
        if key in _STEP_KEYS:
            if unit and dimension_of(unit) == 'time':
                value *= si_factor(unit, 'time')
            return 1.0 / value, i + 1
    return None, 0


def text_layout(path: str | os.PathLike, *, delimiter: str | None = None,
                decimal: str | None = None, names_line: int | None = None,
                units_line: int | None = None,
                data_line: int | None = None) -> TextLayout:
    """Find how a delimited text file is laid out.

    Reads the first `HEAD_LINES` lines. Any argument given is taken as
    stated and the rest is found around it; the module docstring says
    how each part is found. Line numbers count from 1; a `names_line` or
    `units_line` of 0 says the file has none.

    Parameters
    ----------
    path : str or os.PathLike
        The file.
    delimiter : str, optional
        ``'\\t'``, ``';'``, ``','``, ``'|'``, or ``' '`` for spaces.
    decimal : str, optional
        ``'.'`` or ``','``.
    names_line, units_line, data_line : int, optional
        The line of column names, of units, and the first line of data.

    Returns
    -------
    TextLayout
        What was found and stated.

    Raises
    ------
    ValueError
        When no delimiter splits the data rows evenly into numbers, a
        stated line is past the lines read, or a stated delimiter or
        decimal is not one this reader takes.
    """
    if delimiter is not None and delimiter not in DELIMITERS:
        raise ValueError(f'delimiter must be one of {list(DELIMITERS)}, '
                         f'not {delimiter!r}')
    if decimal is not None and decimal not in ('.', ','):
        raise ValueError(f"decimal must be '.' or ',', not {decimal!r}")
    if decimal == ',' and delimiter == ',':
        raise ValueError('a comma cannot be both the delimiter and the '
                         'decimal point')
    lines = _read_lines(path, limit=1 << 20)[:HEAD_LINES]
    name = os.path.basename(str(path))
    if data_line is not None and not 1 <= data_line <= len(lines):
        raise ValueError(f'data_line {data_line} is past the {len(lines)} '
                         f'lines read from {name}')

    found = None
    for candidate in ([delimiter] if delimiter else DELIMITERS):
        point = decimal or _decimal_for(lines, candidate)
        if data_line is not None:
            parts = _split(lines[data_line - 1], candidate)
            if len(parts) >= (1 if candidate == ' ' else 2) and _numeric(parts, point):
                found = (candidate, point, data_line - 1, len(parts))
                break
            continue
        start = _data_start(lines, candidate, point)
        if start is not None:
            found = (candidate, point, *start)
            break
    if found is None:
        said = (f'line {data_line} does not split into numbers'
                if data_line is not None else
                'no delimiter splits its last lines into the same number of '
                'numeric fields')
        said += f' with delimiter {delimiter!r}' if delimiter else ''
        dated = next((match.group(0) for line in lines[-12:]
                      for match in [_DATE.search(line)] if match), None)
        if dated:
            said += (f'; it holds dates or clock times ({dated!r}), and this '
                     'reader takes numbers only: export the time as seconds, '
                     'or leave that column out of the export')
        raise ValueError(f'{name} is not read as delimited text: {said}')
    delimiter, decimal, start, fields = found

    # the header: names directly above the data (blank lines aside),
    # and a line of units between them
    above = [i for i in range(start) if not _blank(lines[i], delimiter)]
    names_at = (names_line - 1 if names_line else None
                ) if names_line is not None else None
    units_at = (units_line - 1 if units_line else None
                ) if units_line is not None else None

    # many exports end every line in the delimiter, and the empty field
    # after the last one is not a column: it is dropped when it is empty
    # in every data row read and has no name, so a named column that
    # happens to be empty is kept, and noted as empty by the read
    sample = [_split(line, delimiter) for line in lines[start:start + 64]
              if not _blank(line, delimiter)]
    trailing = delimiter != ' ' and fields > 1 and all(
        len(parts) == fields and not parts[-1] for parts in sample)

    def header_like(i):
        parts = _split(lines[i], delimiter)
        return (len(parts) == fields or trailing and len(parts) == fields - 1
                ) and not _numeric(parts, decimal)

    if names_line is None:
        nearest = above[-1] if above else None
        if units_at is not None:
            # the names are the header-like line over the stated units
            over = [i for i in above if i < units_at and header_like(i)]
            names_at = over[-1] if over else None
        elif nearest is not None and header_like(nearest):
            parts = _split(lines[nearest], delimiter)
            second = above[-2] if len(above) > 1 else None
            if (units_line is None and second is not None
                    and header_like(second) and any(parts)
                    and all(_unit_like(part) or not part for part in parts)):
                names_at, units_at = second, nearest
            else:
                names_at = nearest
    for label, at in (('names_line', names_at), ('units_line', units_at)):
        if at is not None and not 0 <= at < start:
            raise ValueError(f'{label} {at + 1} is not above the data, which '
                             f'start on line {start + 1}')

    headers = (_split(lines[names_at], delimiter) if names_at is not None
               else [])
    if trailing and (len(headers) < fields or not headers[-1]):
        fields -= 1
    headers = (headers + [''] * fields)[:fields]
    columns, units = [], []
    for k, header in enumerate(headers):
        column, unit = _name_and_unit(header)
        columns.append(column or f'column {k + 1}')
        units.append(unit)
    if units_at is not None:
        row = (_split(lines[units_at], delimiter) + [''] * fields)[:fields]
        units = [cell.strip().strip('()[]') or units[k]
                 for k, cell in enumerate(row)]
    rate, rate_line = _preamble_rate(
        lines, min(i for i in (names_at, units_at, start) if i is not None))

    time_column = None
    if names_at is not None:
        time_column = next((k for k, column in enumerate(columns)
                            if column.strip().lower() in TIME_NAMES), None)
    else:
        time_column = _rising_first_column(lines[start:], delimiter, decimal)
    return TextLayout(delimiter, decimal,
                      0 if names_at is None else names_at + 1,
                      0 if units_at is None else units_at + 1,
                      start + 1, tuple(columns), tuple(units),
                      time_column, rate, rate_line)


def _rising_first_column(lines: Sequence[str], delimiter: str, decimal: str
                         ) -> int | None:
    """0 when the first column of an unnamed file rises in a step even
    to a few parts in a thousand over the first rows, else None."""
    values = []
    for line in lines[:64]:
        if not line.strip():
            continue
        parts = _split(line, delimiter)
        if len(parts) < 2:
            return None
        value = _number(parts[0], decimal)
        if value is None or np.isnan(value):
            return None
        values.append(value)
    if len(values) < 3:
        return None
    steps = np.diff(values)
    if not (steps > 0).all():
        return None
    return 0 if np.allclose(steps, steps.mean(), rtol=5e-3, atol=0) else None


def sniff(path: str | os.PathLike) -> bool:
    """Whether a file is delimited numeric text: a `SUFFIXES` name whose
    last lines split evenly into numbers."""
    if not str(path).lower().endswith(SUFFIXES):
        return False
    if text_head(path, 1) is None:
        return False
    try:
        text_layout(path)
    except Exception:  # noqa: BLE001 - sniffers must not raise on foreign files
        return False
    return True


# ---- the import -------------------------------------------------------------

def _per_channel(value: Any, channels: Sequence[str], indices: Sequence[int],
                 what: str) -> list[Any]:
    """A stated value spread over the channels kept: one value for all, a
    sequence with one per channel, or a mapping by column name or
    column index (from 0)."""
    if value is None:
        return [None] * len(channels)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str) or not isinstance(value, (Sequence, Mapping)):
        return [value] * len(channels)
    if isinstance(value, Mapping):
        out = [None] * len(channels)
        for key, entry in value.items():
            if isinstance(key, (int, np.integer)) and int(key) in indices:
                out[list(indices).index(int(key))] = entry
            elif key in channels:
                out[list(channels).index(key)] = entry
            else:
                raise ValueError(f'{what} names {key!r}, which is not a '
                                 f'channel read: {list(channels)}')
        return out
    value = list(value)
    if len(value) != len(channels):
        raise ValueError(f'{what} has {len(value)} entries for '
                         f'{len(channels)} channels — they must match')
    return value


def _column_index(key: Any, columns: Sequence[str], what: str) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        if not 0 <= int(key) < len(columns):
            raise ValueError(f'{what} {key} is not a column: the file has '
                             f'{len(columns)}, indexed from 0')
        return int(key)
    if key in columns:
        return list(columns).index(key)
    raise ValueError(f'{what} {key!r} is not a column: {list(columns)}')


def _parse(rows: Sequence[tuple[int, str]], layout: TextLayout, name: str
           ) -> tuple[np.ndarray, list[int]]:
    """(values, short lines): the data rows as a (rows, columns) array
    with NaN for missing samples, and the line numbers of rows shorter
    than the header.

    numpy reads a well-formed file in one call; a file it cannot read
    (an empty cell, a short row, a stray word) is read again a cell at a
    time, which is slower and says exactly where the trouble is.
    """
    fields = len(layout.columns)
    texts = [text for _n, text in rows]
    if layout.decimal == ',':
        texts = [text.replace(',', '.') for text in texts]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            values = np.loadtxt(
                texts, dtype=np.float64, ndmin=2, comments=None,
                delimiter=None if layout.delimiter == ' ' else layout.delimiter,
                quotechar='"', usecols=range(fields))
        # usecols reads past a row that is too long without a word; the
        # slow read below is what refuses one, so a file with any row
        # carrying more delimiters than its columns need goes there
        if layout.delimiter == ' ':
            widest = max(len(text.split()) for text in texts)
        else:
            widest = max(text.rstrip().rstrip(layout.delimiter).count(
                layout.delimiter) for text in texts) + 1
        if widest <= fields:
            return values, []
    except ValueError:
        pass
    values = np.full((len(rows), fields), np.nan)
    short = []
    for k, (line, text) in enumerate(rows):
        parts = _split(text, layout.delimiter)
        if len(parts) > fields and any(parts[fields:]):
            raise ValueError(f'{name}, line {line}: {len(parts)} fields where '
                             f'the data above it have {fields}')
        parts = parts[:fields]
        if len(parts) < fields:
            short.append(line)
        for j, part in enumerate(parts):
            value = _number(part, layout.decimal)
            if value is None:
                hint = (' (a date or clock time: leave the column out with '
                        'columns=, and state the sample rate)'
                        if _DATE.search(part) else '')
                raise ValueError(f'{name}, line {line}: {layout.columns[j]} '
                                 f'holds {part!r}, which is not a number{hint}')
            values[k, j] = value
    return values, short


_PRINTED = re.compile(r'\s*"?\s*[-+]?\d*(?:\.(\d*))?(?:[eE]([-+]?\d+))?\s*"?\s*')


def _resolution(rows: Sequence[tuple[int, str]], layout: TextLayout,
                column: int) -> np.ndarray:
    """One unit of the last printed digit of each row's cell in
    `column`, in the column's own unit: how far the file may have
    rounded that value when it wrote it.

    Per row, not one figure for the column: a value written in a
    shortest form ('1' for 1.0, '0.5') carries few digits because it
    needed few, and a single coarsest figure would read the whole column
    as rounded to the second.
    """
    out = np.full(len(rows), np.inf)
    delimiter = layout.delimiter
    for k, (_line, text) in enumerate(rows):
        if delimiter == ' ':
            parts = text.split()
        elif '"' in text:
            parts = _split(text, delimiter)
        else:
            parts = text.split(delimiter)
        if column >= len(parts):
            continue
        cell = parts[column]
        if layout.decimal == ',':
            cell = cell.replace(',', '.')
        match = _PRINTED.fullmatch(cell)
        if match:
            out[k] = 10.0 ** (int(match.group(2) or 0) - len(match.group(1) or ''))
    return out


def _even_time(time: np.ndarray, rows: Sequence[tuple[int, str]],
               layout: TextLayout, column: int, scale: float, name: str
               ) -> tuple[float, float, float]:
    """(start, step, rounding) of an evenly stepped time column, in
    seconds; `rounding` is how far the times sat from the even step,
    zero when they were even as written. Refused when the column is out
    of order, or when any time sits off the even step by more than a
    unit of the last digit it was written to, beyond the fit."""
    from ..core.data import _even_steps

    try:
        _even_steps(time, 'A delimited-text import')
        exact = True
    except ValueError as error:
        if (np.diff(time) <= 0).any():
            raise ValueError(f'{name}: {error} (the time column, '
                             f'{layout.columns[column]})') from None
        exact, refusal = False, error
    if exact:
        return (float(time[0]),
                float((time[-1] - time[0]) / (len(time) - 1)), 0.0)
    # the even step that fits every time best, not the one through the
    # first and last: those two are rounded like the rest, and over a
    # short record their rounding moved the rate by hundredths of a hertz
    index = np.arange(len(time), dtype=np.float64)
    middle = index - index.mean()
    step = float(middle @ (time - time.mean()) / (middle @ middle))
    start = float(time.mean() - step * index.mean())
    off = np.abs(time - (start + step * index))
    # each time may sit a unit of its own last digit off the even step,
    # and the fitted step itself by about a unit of the typical digit
    printed = _resolution(rows, layout, column) * scale
    if (off <= printed + np.median(printed)).all():
        return start, step, float(off.max())
    raise ValueError(
        f'{name}: {refusal} (the time column, {layout.columns[column]}), more '
        'than the digits it is written to can explain; if the rows were '
        'sampled evenly, read them without it: time_column=False and the '
        'sample_rate') from None


def read_delimited(path: str | os.PathLike, *, delimiter: str | None = None,
                   decimal: str | None = None, names_line: int | None = None,
                   units_line: int | None = None, data_line: int | None = None,
                   time_column: int | str | bool | None = None,
                   time_unit: str | None = None,
                   sample_rate: float | None = None,
                   columns: Sequence[int | str] | None = None,
                   names: Sequence[str] | None = None,
                   units: str | Sequence[str | None] | Mapping | None = None,
                   quantities: str | Sequence[str | None] | Mapping | None = None,
                   missing: str = 'keep', max_rows: int | None = None) -> Any:
    """Read a delimited text file as a `TimeHistory`.

        history = read_delimited('run.csv')
        history = read_delimited('run.txt', time_column=False,
                                 sample_rate=5120, units='g',
                                 names=['101Z+', '102Z+'])

    The layout is found by `text_layout` and anything stated here is
    taken instead; the module docstring says how each part is found, and
    what is noted and refused. Notes are `ImportNote` warnings.

    Parameters
    ----------
    path : str or os.PathLike
        The file.
    delimiter, decimal, names_line, units_line, data_line
        As for `text_layout`.
    time_column : int, str or bool, optional
        The time column, by index (from 0) or name; False when the file
        has none, which needs a sample rate. Found when omitted.
    time_unit : str, optional
        The time column's unit, when its header does not give one.
    sample_rate : float, optional
        Samples per second. With a time column, it must agree with the
        column's step.
    columns : sequence of int or str, optional
        The channels to read, by index (from 0) or name, in the order
        given. Every column but the time column when omitted.
    names : sequence of str, optional
        A name for each channel read: a DOF such as '101X+' becomes the
        record's response DOF.
    units : str, sequence or mapping, optional
        The unit of every channel, one per channel, or {name or index:
        unit}; taken over the header's.
    quantities : str, sequence or mapping, optional
        What each channel measures ('acceleration', 'force',
        'displacement'), in the same forms.
    missing : {'keep', 'refuse'}
        Whether a missing sample is kept as NaN, with a note, or refuses
        the file.
    max_rows : int, optional
        Read only the first so many data rows (a preview).

    Returns
    -------
    TimeHistory
        One record per channel.

    Raises
    ------
    ValueError
        For anything the module docstring lists as refused, a stated
        unit or quantity that cannot be used, or a stated argument that
        names no column.
    """
    from ..core.channel_health import channel_health
    from ..core.data import TimeHistory
    from ..core.unit_choices import stored_dimension
    from ..core.validate import dofs
    from ..units import dimension_of, from_si, si_factor

    if missing not in ('keep', 'refuse'):
        raise ValueError(f"missing must be 'keep' or 'refuse', not {missing!r}")
    if sample_rate is not None and not (np.isfinite(sample_rate)
                                        and sample_rate > 0):
        raise ValueError(f'sample_rate must be a positive number, not {sample_rate!r}')
    name = os.path.basename(str(path))
    layout = text_layout(path, delimiter=delimiter, decimal=decimal,
                         names_line=names_line, units_line=units_line,
                         data_line=data_line)
    # a preview reads the head of the file: its data rows, and room for
    # the blank lines among them
    lines = _read_lines(path, lines=None if max_rows is None else
                        layout.data_line + 2 * max(int(max_rows), 0) + 64)
    rows = [(n + 1, text) for n, text in enumerate(lines)
            if n + 1 >= layout.data_line and not _blank(text, layout.delimiter)]
    if max_rows is not None:
        rows = rows[:max(int(max_rows), 0)]
    values, short = _parse(rows, layout, name)
    # rows empty in every column at the end (',,,') are not samples
    filled = np.isfinite(values).any(axis=1)
    if not filled.all():
        last = int(np.flatnonzero(filled)[-1]) + 1 if filled.any() else 0
        values, rows = values[:last], rows[:last]
        short = [line for line in short if line <= (rows[-1][0] if rows else 0)]
    if len(values) < 2:
        raise ValueError(f'{name} holds {len(values)} data row'
                         f'{"s" * (len(values) != 1)}; a time history needs two')
    notes = []
    if short:
        notes.append(f'{len(short)} row{"s" * (len(short) != 1)} shorter than the '
                     f'header (the first on line {short[0]}): the cells missing '
                     'from the end of a short row read as missing samples')

    # ---- the time column and the sample rate ---------------------------
    if time_column is None:
        time_index = layout.time_column
        if time_index is not None and not layout.names_line:
            notes.append('column 1 rises in an even step and the file names no '
                         'columns: read as time')
    elif time_column is False:
        time_index = None
    elif time_column is True:
        raise ValueError('time_column takes an index, a name or False, not True')
    else:
        time_index = _column_index(time_column, layout.columns, 'time_column')
    stated_rate = sample_rate
    rate_source = 'the stated sample rate'
    if stated_rate is None and layout.sample_rate is not None:
        stated_rate = layout.sample_rate
        rate_source = f'the rate stated on line {layout.rate_line}'
    if time_index is not None:
        column = layout.columns[time_index]
        unit = time_unit or layout.units[time_index]
        if unit:
            if dimension_of(unit) != 'time':
                raise ValueError(f'{name}: the time column {column} is in '
                                 f'{unit!r}, which is not a unit of time')
            scale = si_factor(unit, 'time')
        else:
            scale = 1.0
        time = values[:, time_index] * scale
        gaps = np.flatnonzero(~np.isfinite(time))
        if len(gaps):
            raise ValueError(f'{name}, line {rows[gaps[0]][0]}: the time column '
                             f'{column} has no value, so the row has no time')
        start, step, rounding = _even_time(time, rows, layout, time_index,
                                           scale, name)
        rate = 1.0 / step
        if stated_rate is not None:
            if abs(rate - stated_rate) > RATE_TOLERANCE * stated_rate:
                raise ValueError(
                    f'{name}: the time column {column} steps at {step:g} s '
                    f'({rate:.6g} per second) and {rate_source} is '
                    f'{stated_rate:g}; one of them is wrong')
            rate = float(stated_rate)
        elif not unit:
            notes.append(f'the time column {column} names no unit: read as '
                         f'seconds, {rate:.6g} samples per second')
        if rounding:
            notes.append(f'the time column {column} was rounded when it was '
                         f'written, and its times sit up to {rounding:.3g} s off '
                         'an even step, within that rounding: read as an even '
                         f'{rate:.6g} samples per second')
        abscissa = start + np.arange(len(values)) / rate
    else:
        if stated_rate is None:
            raise ValueError(f'{name} says its sampling with neither a time '
                             'column nor a stated sample rate: give '
                             'sample_rate, or time_column if one of its '
                             f'columns {list(layout.columns)} holds time')
        rate = float(stated_rate)
        abscissa = np.arange(len(values)) / rate

    # ---- the channels ---------------------------------------------------
    if columns is None:
        indices = [k for k in range(len(layout.columns)) if k != time_index]
    else:
        indices = []
        for key in columns:
            k = _column_index(key, layout.columns, 'columns names')
            if k == time_index:
                raise ValueError(f'columns names {key!r}, which is the time column')
            if k not in indices:
                indices.append(k)
    if not indices:
        raise ValueError(f'{name} holds no channel beside its time column')
    channels = [layout.columns[k] for k in indices]
    if names is not None:
        names = [str(n) for n in names]
        if len(names) != len(indices):
            raise ValueError(f'names has {len(names)} entries for '
                             f'{len(indices)} channels — they must match')
    labels = names or channels

    stated_units = _per_channel(units, channels, indices, 'units')
    claimed = _per_channel(quantities, channels, indices, 'quantities')
    quantity = []
    for k, claim in enumerate(claimed):
        if claim is None:
            quantity.append(None)
            continue
        tag = stored_dimension(str(claim))
        if tag not in CHANNEL_QUANTITIES:
            raise ValueError(f'quantities: {labels[k]} is stated as {claim!r}, '
                             'which is not a quantity a channel measures')
        quantity.append(tag)
    for k, unit in enumerate(stated_units):
        if unit:
            reason = unit_problem(str(unit), quantity[k])
            if reason is not None:
                raise ValueError(f'units: {labels[k]} is stated in {unit!r}, '
                                 f'which {reason}')
    header_units = [layout.units[k] for k in indices]
    chosen = [stated or header for stated, header in zip(stated_units, header_units,
                                                         strict=True)]

    data = values[:, indices].T
    if missing == 'refuse':
        # the first missing sample in the file's order: by line, then
        # by column along it
        bad = ~np.isfinite(data.T)
        if bad.any():
            sample, channel = (int(v[0]) for v in np.nonzero(bad))
            raise ValueError(f'{name}, line {rows[sample][0]}: {labels[channel]} '
                             'has a missing sample')

    response, block = [], []
    for label in labels:
        try:
            [dof] = dofs([label])
        except ValueError:
            response.append('')
            block.append(label)
        else:
            response.append(dof)
            block.append('')
    history = TimeHistory(
        abscissa, data, response_dof=response,
        block=block if any(block) else None,
        dimension_hint=quantity,
        comment=[f'column {k + 1} of {name}' for k in indices])
    for k, unit, reason in declare_stated_units(history, chosen, quantity):
        # a stated unit passed the same check above, so what fails here
        # came from the header: a claimed quantity it contradicts is two
        # claims at odds, and a unit that is merely unreadable is noted
        if stated_units[k] or quantity[k] is not None:
            raise ValueError(f'{name}: {labels[k]} is headed in {unit!r}, '
                             f'which {reason}')
        notes.append(f'{labels[k]}: the unit {unit!r} in the header {reason}; '
                     'left undeclared, holding the file\'s numbers')
    for finding in channel_health(history):
        unit = history.ordinate_unit[finding.record]
        if finding.dead and unit:
            # the dead value as the file wrote it, not in SI
            finding = replace(finding, constant=float(
                from_si(finding.constant, unit)))
        notes.append(finding.describe(labels[finding.record], abscissa, unit))
    for note in notes:
        warnings.warn(ImportNote(f'{name}: {note}'), stacklevel=2)
    return history


def load(path: str | os.PathLike, **kwargs: Any) -> Any:
    """`read_delimited`, as the importer registry calls it."""
    return read_delimited(path, **kwargs)


# ---- writing ----------------------------------------------------------------

def handles(obj: Any) -> bool:
    """Whether `save` can write `obj`: a time history."""
    from ..core.data import TimeHistory

    return isinstance(obj, TimeHistory)


def save(history: Any, path: str | os.PathLike, unit_system: Any = None,
         **_kwargs: Any) -> None:
    """Write a time history as a CSV this module reads back.

    One line naming what the file holds, then a header of ``time (s)``
    and one column per record named by its label with its unit in
    brackets, then one row per sample, every value written to the
    digits that read back as the same number. With `unit_system` each
    record is written in that system's unit for its quantity (g
    included, since a header can say g); without one, as stored, in SI.
    A record with no units is written as its numbers, with no unit in
    its header.

    Parameters
    ----------
    history : TimeHistory
        What to write.
    path : str or os.PathLike
        Where.
    unit_system : UnitSystem, optional
        The units to write in.

    Returns
    -------
    None
    """
    from ..units import SI, UNKNOWN

    system = SI if unit_system is None else unit_system
    header = ['time (s)']
    columns = [np.asarray(history.abscissa, dtype=np.float64)]
    ordinate = np.real(np.asarray(history.ordinate))
    for i in range(history.num_records):
        label = history.record_label(i) or f'record {i + 1}'
        dimension = history.ordinate_dim[i]
        if dimension == UNKNOWN:
            header.append(label)
            columns.append(ordinate[i])
        else:
            header.append(f'{label} ({system.unit(dimension)})')
            columns.append(system.from_si(ordinate[i], dimension))
    rate = ''
    try:
        rate = f' at {history.sample_rate:.17g} samples per second'
    except (ValueError, AttributeError):
        pass
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(f'# Visual Dynamics time history: {history.num_records} records, '
                f'{len(history.abscissa)} samples{rate}\n')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        for row in np.column_stack(columns):
            writer.writerow([repr(float(v)) for v in row])
