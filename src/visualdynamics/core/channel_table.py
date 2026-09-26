"""Channel table: per-channel test metadata (sensor, DOF, engineering units).

One fixed set of columns, each with a type and a rule, rather than
whatever a source happened to carry. The table used to be a pandas
frame that took anything — a Rattlesnake import arrived with thirteen
columns of controller settings riding behind the ones visualdynamics
reads — and "anything" is exactly what a column cannot have logic
about. Now every column says what it holds, so a direction can be a
drop-down, a unit can be narrowed by the channel's type, and an
expiration can be a date.

Two strictnesses, on purpose, and they are not the same:

- **A file is read leniently.** A value that will not coerce is blanked
  rather than refused, because refusing the import leaves the user
  nothing to fix, and the whole point of this table is that it is the
  place to fix it. (`validate.dofs` makes the same trade for DOFs.)
- **An edit is refused.** Once a person is typing, invalid state is
  rejected at entry with the reason, never written and complained about
  afterwards.
"""

from __future__ import annotations

import datetime as _datetime
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np

from .validate import DIRECTIONS

if TYPE_CHECKING:                                    # pragma: no cover
    import pandas as pd


class Spec(NamedTuple):
    """What one column holds, and what it will accept."""

    kind: str                              # how the value is coerced
    default: str = ''
    choices: tuple[str, ...] = ()
    minimum: int | None = None             # ints and floats
    unique: bool = False


#: The quantities a channel can measure. The dimension vocabulary the
#: rest of the package uses (`units._DIMENSIONALITY`), so a channel's
#: type and a record's dimension are the same word and the unit
#: shortlist follows from it — not a second vocabulary to keep in step.
CHANNEL_TYPES = ('acceleration', 'velocity', 'length', 'force', 'pressure',
                 'strain', 'voltage', 'temperature')

#: what a channel is *for*: a reference drives, a response answers, a
#: monitor is recorded and judged as neither. One value, not three
#: booleans, so the one nonsense state — a channel that is both the
#: input and the output of the same estimation — cannot be written.
ROLES = ('reference', 'response', 'monitor')

#: Which leg of a triaxial accelerometer this channel is, for the
#: record. Not a DOF: the direction column says where it points.
TRIAX = ('X', 'Y', 'Z')


#: Headers, where title-casing the column name is not the answer.
#: Sensitivity says its units here because it has no column to say them
#: in — it is mV per whatever the Unit cell beside it holds, and the two
#: are one calibration statement. Range is always volts: the
#: instrumentation's limit, not the sensor's.
TITLES = {
    'triax_dof': 'Triax DOF',
    'sensitivity': 'Sensitivity (mV/Unit)',
    'range': 'Range (V)',
    'unit_x': 'Unit X',
    'unit_y': 'Unit Y',
    'unit_z': 'Unit Z',
    'nearest_axis': 'Nearest Axis',
    'axis_angle': 'Angle to Axis (deg)',
}

#: the columns a geometry adds beside the table's own, in order: the
#: channel's measured direction as a unit vector in the geometry's
#: global system, the global axis it is nearest, and how far off it
#: sits (Brandon, 2026-09-20). Derived — read from the geometry each
#: time, never stored — so they follow an edit to the node or the
#: direction at once and never disagree with the geometry.
DERIVED_COLUMNS = ('unit_x', 'unit_y', 'unit_z', 'nearest_axis', 'axis_angle')
AXES = ('X', 'Y', 'Z')


def title_of(name: str) -> str:
    """The header a column wears, wherever it is shown — the table in
    the window and the table in the report are the same table."""
    return TITLES.get(name) or name.replace('_', ' ').title()


def COLUMNS_KIND(name: str) -> str:
    """What kind of value a column holds."""
    return ChannelTable.COLUMNS[name].kind


def CHOICES_FOR(name: str) -> tuple[str, ...]:
    """The values a column limits itself to, empty when it does not.

    'control' is a choice in every sense a spreadsheet cares about,
    though the schema calls it a flag.
    """
    spec = ChannelTable.COLUMNS[name]
    if spec.kind == 'flag':
        return ('True', 'False')
    return spec.choices


def canonical_name(name: str) -> str:
    """A source's column name as this schema spells it.

    Tidy first — case, spaces, hyphens, a trailing colon someone typed,
    and a parenthetical unit — then the alias table for words that
    genuinely differ. The parenthetical matters twice over: it is how
    this schema writes its own headers ('Sensitivity (mV/Unit)'), so a
    spreadsheet exported from here has to read back in; and it is how a
    calibration lab writes theirs ('Sensitivity (mV/g)'), which lands in
    the same column for the same reason.
    """
    text = str(name).strip().rstrip(':')
    if '(' in text:
        text = text.split('(', 1)[0]
    key = '_'.join(text.lower().replace('-', ' ').split())
    return _ALIAS_OF.get(key, key)


def _parse_flag(text: str) -> bool:
    """'True'/'False' in any case, 1/0, yes/no, or blank for False —
    the spellings a spreadsheet paste actually contains."""
    value = text.strip().lower()
    if value in ('true', '1', 'yes', 'checked'):
        return True
    if value in ('', 'false', '0', 'no', 'unchecked'):
        return False
    raise ValueError(f'{text!r} is not a yes or a no')


#: Date forms a spreadsheet actually contains, tried in order. ISO
#: first because it is what we store and what round-trips unambiguously;
#: the American and European orders are ambiguous with each other, so
#: the American one wins by being the one a US calibration lab writes.
_DATE_FORMATS = ('%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%d.%m.%Y', '%Y/%m/%d')


def _parse_date(text: str) -> str:
    """A date as ISO text, or a refusal.

    Stored as text rather than a pandas datetime: this table round-trips
    through xlsx and HDF5, and a string that reads the same in all three
    is worth more here than a dtype.
    """
    value = str(text).strip()
    if not value:
        return ''
    value = value.split(' ')[0]        # a spreadsheet's midnight timestamp
    for form in _DATE_FORMATS:
        try:
            return _datetime.datetime.strptime(  # noqa: DTZ007
                value, form).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f'{text!r} is not a date — try YYYY-MM-DD')


class ChannelTable:
    """Per-channel metadata, one fixed column set, each column typed.

    Attributes:
        frame: The table, one row per channel, columns in `SCHEMA`
            order. Every cell is text except `channel` and `node`,
            which are integers — a serial number that happens to be all
            digits is not a number, and a column read back as int64
            would lose its leading zero and come out of a round trip a
            different string.
    """

    #: Every column, in the order the table shows them, each with the
    #: rule its cells answer to. A source's other columns are dropped:
    #: a Rattlesnake save carries its coupling, excitation and feedback
    #: wiring, which describe the controller's run rather than the
    #: measurement, and carrying them made this object a place data went
    #: to be ignored.
    COLUMNS: ClassVar[dict[str, Spec]] = {
        # 'int' is a number the column must have; 'index' is one it may
        # have. A channel without a number is not a channel — it is the
        # join key — but a node nobody recorded is the ordinary case
        # this table exists to let someone fix, and 0 cannot stand in
        # for 'unstated' when 0 is itself a legal node number.
        'channel': Spec('int', minimum=1, unique=True),
        'node': Spec('index', minimum=0),
        'direction': Spec('choice', choices=DIRECTIONS),
        'role': Spec('choice', choices=ROLES),
        'control': Spec('flag', default='False'),
        'channel_type': Spec('choice', choices=CHANNEL_TYPES),
        'unit': Spec('unit'),
        'sensitivity': Spec('float', minimum=0),
        'range': Spec('int', default='5', minimum=1),
        'serial_number': Spec('text'),
        'make': Spec('text'),
        'model': Spec('text'),
        'triax_dof': Spec('choice', choices=TRIAX),
        'expiration': Spec('date'),
        'comment': Spec('text'),
    }
    SCHEMA = tuple(COLUMNS)
    #: What other people call these columns. Applied after a generic
    #: tidy — 'Serial Number' and 'serial-number' both reach
    #: `serial_number` without help — so this holds only the genuinely
    #: different words. Every source goes through it, which is what
    #: makes "everything parses into these columns" true of a
    #: controller save, a calibration lab's spreadsheet and an older
    #: .vdyn alike, rather than of whichever importer remembered.
    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        'channel': ('channel_number', 'ch', 'chan', 'channel_index'),
        'node': ('node_number', 'node_id', 'point', 'point_number', 'grid'),
        'direction': ('dir', 'axis', 'component', 'dof_direction',
                      'node_direction'),
        'unit': ('units', 'engineering_unit', 'engineering_units', 'eu'),
        'channel_type': ('type', 'sensor_type', 'measurement', 'quantity'),
        'sensitivity': ('sens', 'sensitivity_mv_eu', 'cal_factor'),
        'range': ('voltage_range', 'full_scale', 'full_scale_voltage'),
        'serial_number': ('serial', 'sn', 'serial_no'),
        'make': ('manufacturer', 'vendor', 'brand'),
        'model': ('model_number', 'part_number', 'model_no'),
        'triax_dof': ('triax', 'triax_axis', 'triaxial_dof'),
        'expiration': ('cal_due', 'calibration_due', 'cal_date', 'expires',
                       'expiration_date', 'calibration_expiration'),
        'comment': ('comments', 'note', 'notes', 'description'),
    }
    #: the four a table is meaningless without; a source missing one is
    #: refused rather than filled in
    CORE = ('channel', 'node', 'direction', 'unit')
    ROLES = ROLES
    CHANNEL_TYPES = CHANNEL_TYPES
    TRIAX = TRIAX

    def __init__(self, columns: Mapping[str, Any] | pd.DataFrame) -> None:
        import pandas as pd

        if isinstance(columns, pd.DataFrame):
            frame = columns.copy()
        else:
            lengths = {name: len(values) for name, values in columns.items()}
            if len(set(lengths.values())) > 1:
                first = next(iter(lengths.values()))
                bad = next(name for name, n in lengths.items() if n != first)
                raise ValueError(
                    f'column {bad!r} has {lengths[bad]} entries, '
                    f'expected {first}')
            frame = pd.DataFrame(dict(columns))
        frame = frame.rename(columns={name: canonical_name(name)
                                      for name in frame.columns})
        # a source naming one column two ways keeps the first
        frame = frame.loc[:, ~frame.columns.duplicated()]
        for name in self.CORE:
            if name not in frame.columns:
                raise ValueError(f'missing core column {name!r}')
        for name, spec in self.COLUMNS.items():
            if name not in frame.columns:
                frame[name] = spec.default
        # the schema and nothing else: an older file's extras, and a
        # controller's run settings, stop here
        self.frame: pd.DataFrame = self._typed(frame[list(self.SCHEMA)])

    @classmethod
    def _typed(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce a whole frame, leniently: what will not read, blanks."""
        import pandas as pd

        frame = frame.reset_index(drop=True)
        for name, spec in cls.COLUMNS.items():
            values = []
            for raw in frame[name]:
                try:
                    values.append(cls._coerce(name, raw))
                except (ValueError, TypeError):
                    # lenient here, as the docstring says: an unreadable
                    # value takes the default rather than failing the
                    # import, and an int column's default is its own
                    values.append(spec.default or (0 if spec.kind == 'int'
                                                   else ''))
            if spec.kind == 'int':
                frame[name] = pd.Series(
                    [int(v) if str(v) != '' else 0 for v in values],
                    dtype='int64')
            else:
                frame[name] = pd.Series(values, dtype='string').fillna('')
        frame['channel_type'] = cls._typed_by_unit(frame)
        return frame

    @classmethod
    def _typed_by_unit(cls, frame) -> list[str]:
        """The type column with its blanks answered by the unit beside
        them.

        A source states a channel's type in its own words, and the
        coercion above is lenient: a spelling the schema does not know
        blanks rather than failing the import. So a controller writing
        'Accel' — or nothing at all — lost the type while its 'Voltage'
        channels kept theirs, and a table of accelerometers in G came
        in typeless (Brandon, 2026-09-20). The unit is the same claim
        said another way, and `dimension_of` already reads it, so a
        blank type takes the unit's answer. A stated type is never
        overruled: a disagreement between the two is the owner's to
        resolve, and `set_cell` refuses to create one.

        Only a unit that names a channel type answers. A unitless unit
        reads as 'dimensionless', which is no channel type — strain and
        a bare ratio share that dimensionality, which is the whole
        reason they are two names.
        """
        from ..units import dimension_of

        out = []
        for declared, unit in zip(frame['channel_type'], frame['unit']):
            declared = '' if declared is None else str(declared)
            unit = '' if unit is None else str(unit)
            if declared or not unit:
                out.append(declared)
                continue
            found = dimension_of(unit)
            out.append(found if found in CHANNEL_TYPES else '')
        return out

    @classmethod
    def _coerce(cls, name: str, value: object) -> Any:
        """One value, in the column's own terms, or a refusal.

        Uniqueness is not checked here — it is a property of the column
        rather than of the value, so `set_cell` owns it.
        """
        spec = cls.COLUMNS[name]
        text = '' if value is None else str(value).strip()
        if spec.kind == 'int':
            if not text:
                # an int column has no blank: a channel without a number
                # is not a channel, and a range of nothing is not a limit
                raise ValueError(f'{name} cannot be blank')
            number = float(text)
            if number != int(number):
                raise ValueError(f'{name} is a whole number')
            number = int(number)            # '5.0' from a spreadsheet
            if spec.minimum is not None and number < spec.minimum:
                what = ('positive' if spec.minimum == 1 else
                        f'at least {spec.minimum}')
                raise ValueError(f'{name} must be {what}')
            return number
        if spec.kind == 'index':
            if not text:
                return ''
            number = int(float(text))
            if spec.minimum is not None and number < spec.minimum:
                raise ValueError(f'{name} must be {spec.minimum} or more')
            return str(number)
        if spec.kind == 'float':
            if not text:
                return ''
            number = float(text)
            if spec.minimum is not None and not number > spec.minimum:
                raise ValueError(f'{name} must be greater than {spec.minimum}')
            return text
        if spec.kind == 'choice':
            if not text:
                return ''
            if name == 'channel_type':
                # the interface — and the spreadsheet it exports — says
                # 'displacement' where the schema stores 'length'
                from .unit_choices import stored_dimension
                text = stored_dimension(text)
            wanted = {choice.lower(): choice for choice in spec.choices}
            if text.lower() not in wanted:
                raise ValueError(
                    f'{name} is one of {", ".join(spec.choices)} '
                    '(or blank for undeclared)')
            return wanted[text.lower()]
        if spec.kind == 'flag':
            return str(_parse_flag(text))
        if spec.kind == 'date':
            return _parse_date(text)
        if spec.kind == 'unit':
            if text:
                from ..units import si_transform
                si_transform(text.replace('^', '**'))   # raises if unreadable
            return text
        return text

    def set_cell(self, name: str, row: int, value: object) -> None:
        """Write one cell; a value the column cannot hold is refused.

        Strict where `_typed` is lenient: a person typing gets the
        reason, where a file gets the benefit of the doubt.

        Parameters
        ----------
        name : str
            The column to write.
        row : int
            Which channel.
        value : object
            The value, refused if the column cannot hold it.

        Returns
        -------
        None
        """
        if name not in self.COLUMNS:
            raise KeyError(f'no column {name!r}')
        spec = self.COLUMNS[name]
        coerced = self._coerce(name, value)
        if spec.unique and coerced != '':
            clash = np.flatnonzero(self[name] == coerced)
            if len(clash) and clash[0] != row:
                raise ValueError(
                    f'{name} {coerced} already exists — it is the join key '
                    'and must be unique')
        # Control and role are two questions, answered independently
        # (Brandon, 2026-09-26): what the controller drove the test to
        # match, and what the channel is to an FRF. A controlled force
        # can be an FRF's reference, a controlled voltage too, and
        # either can be a monitor. The schema once refused every
        # control channel that was not a response, as "the one nonsense
        # pair"; force and voltage control are why it was not one.
        if name == 'unit' and coerced:
            self._agrees_with_type(row, str(coerced))
        self.frame.loc[row, name] = coerced
        if name == 'channel_type':
            # a unit of some other kind is no longer an answer here —
            # the same withdrawal the Imported Units pane makes
            unit = str(self.frame.loc[row, 'unit'])
            if unit and coerced:
                from ..units import dimension_of
                if dimension_of(unit) != coerced:
                    self.frame.loc[row, 'unit'] = ''

    def _agrees_with_type(self, row: int, unit: str) -> None:
        """A declared type and a unit have to describe one channel."""
        from ..units import dimension_of

        declared = str(self.frame.loc[row, 'channel_type'])
        if not declared:
            return
        actual = dimension_of(unit)
        # `dimension_of` never answers 'strain': strain and
        # dimensionless have the same (empty) dimensionality, and a unit
        # cannot tell them apart — which is the whole reason they are
        # two names. So a unitless unit answers for either.
        if declared in ('strain', 'dimensionless'):
            if actual == 'dimensionless':
                return
            raise ValueError(
                f'{unit} is a {actual}, and this channel is declared '
                f'{declared} — change the type first')
        if actual is not None and actual != declared:
            raise ValueError(
                f'{unit} is a {actual}, and this channel is declared '
                f'{declared} — change the type first')

    def units_for(self, row: int) -> list[str]:
        """The units this channel could be in, given its declared type.

        Parameters
        ----------
        row : int
            Which channel.

        Returns
        -------
        list of str
            The units this channel could be in, given its
            declared type.
        """
        from .unit_choices import ALL_ORDINATE_UNITS, ORDINATE_UNITS

        declared = str(self.frame.loc[row, 'channel_type'])
        return list(ORDINATE_UNITS.get(declared, ALL_ORDINATE_UNITS))

    def delete_channels(self, indices: Sequence[int]) -> None:
        """Remove the given rows in place; the last one is refused.

        Parameters
        ----------
        indices : sequence of int
            Which rows to remove.

        Returns
        -------
        None
        """
        doomed = {int(i) for i in indices}
        bad = [i for i in doomed if not 0 <= i < self.num_channels]
        if bad:
            raise IndexError(f'no such channels: {sorted(bad)}')
        keep = [i for i in range(self.num_channels) if i not in doomed]
        if not keep:
            raise ValueError('cannot delete every channel; delete the '
                             'object instead')
        self.frame = self.frame.iloc[keep].reset_index(drop=True)

    @property
    def num_channels(self) -> int:
        """How many channels the table describes — one per row."""
        return len(self.frame)

    @property
    def column_names(self) -> list[str]:
        """The table's column headings, in order."""
        return [str(name) for name in self.frame.columns]

    def orientation(self, row: int, geometry: Any
                    ) -> tuple[np.ndarray | None, str | None, float | None]:
        """One channel's measured direction against a geometry.

        The DOF the row names — node and direction — read through the
        frame that node is measured in (`Geometry.dof_direction`), as
        a unit vector in the geometry's global system; the global axis
        it is nearest, signed ('X+', 'Z-'); and the angle between the
        two in degrees. What the derived columns show.

        Parameters
        ----------
        row : int
            The channel's row.
        geometry : Geometry
            The geometry the channel is measured on.

        Returns
        -------
        tuple
            (vector, axis, angle), each None when the geometry cannot
            place the channel — a node it lacks, no direction, or a
            node measured in a frame that turns with position.
        """
        if geometry is None:
            return None, None, None
        vector = geometry.dof_direction(self.dof_strings()[row])
        if vector is None:
            return None, None, None
        k = int(np.argmax(np.abs(vector)))
        axis = f'{AXES[k]}{"+" if vector[k] >= 0 else "-"}'
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(vector[k]))))))
        return vector, axis, angle

    def derived_cells(self, row: int, geometry: Any) -> list[str]:
        """The derived columns' cells for one row, as text: three
        components to three decimals, the nearest axis, the angle to a
        tenth of a degree; empty where the geometry cannot place the
        channel.

        Parameters
        ----------
        row : int
            The channel's row.
        geometry : Geometry
            The geometry the channel is measured on.

        Returns
        -------
        list of str
            One cell per `DERIVED_COLUMNS` entry.
        """
        vector, axis, angle = self.orientation(row, geometry)
        if vector is None:
            return [''] * len(DERIVED_COLUMNS)
        return [f'{float(v):+.3f}' for v in vector] + [axis, f'{angle:.1f}']

    def dof_strings(self) -> list[str]:
        """Each channel's degree of freedom, as '101Z+' strings."""
        return [f'{node}{direction}' for node, direction
                in zip(self.frame['node'], self.frame['direction'])]

    def rename_dof(self, old: str, new: str,
                   quantity: str | None = None) -> int:
        """Give the channel at coordinate `old` the coordinate `new`,
        in place — its node and direction cells rewritten, since a DOF
        string is those two concatenated. The same correction a data
        array's `rename_dof` makes, for the table that names the
        channels (Brandon, 2026-09-06), and the same rule: the channel
        moves, not every channel at the point.

        Parameters
        ----------
        old : str
            The coordinate as the table has it, '101Z+'.
        new : str
            The coordinate to give it: a node number then a direction,
            normalized the way every DOF is, and refused when it is not
            one — a table row may lack a node or a direction, but a
            correction typed by a person is whole.
        quantity : str, optional
            Which channel at `old`, by the quantity its unit names
            ('acceleration', 'force' …; 'unknown' where the unit names
            none). Every channel at the coordinate when omitted.

        Returns
        -------
        int
            How many channels changed.
        """
        from ..units import dimension_of
        from .data import parse_dof
        from .validate import dofs

        old = str(old).strip()
        (new,) = dofs([str(new)], 'DOF')
        node, direction = parse_dof(new)
        if node is None:
            raise ValueError(f'{new!r} has no node number')
        units = [str(u).strip() for u in self.frame['unit']]
        named = [(dimension_of(u) if u else None) or 'unknown' for u in units]
        rows = [i for i, dof in enumerate(self.dof_strings())
                if dof == old and quantity in (None, named[i])]
        if not rows:
            raise ValueError(f'no {quantity + " " if quantity else ""}channel '
                             f'at {old!r}')
        if new == old:
            return 0
        for i in rows:
            self.set_cell('node', i, node)
            self.set_cell('direction', i, direction)
        return len(rows)

    def roles(self) -> list[str]:
        """Each channel's declared role, '' where undeclared."""
        return [str(v) for v in self.frame['role']]

    def controls(self) -> np.ndarray:
        """Which channels are control channels, as booleans."""
        return np.array([_parse_flag(str(v)) for v in self.frame['control']])

    def types(self) -> list[str]:
        """What each channel measures, '' where undeclared."""
        return [str(v) for v in self.frame['channel_type']]

    def sensitivities(self) -> np.ndarray:
        """mV per engineering unit, NaN where undeclared."""
        return np.array([float(v) if str(v).strip() else np.nan
                         for v in self.frame['sensitivity']])

    def ranges(self) -> np.ndarray:
        """The instrumentation voltage limit, per channel."""
        return np.array([int(v) for v in self.frame['range']])

    def __getitem__(self, name: str) -> np.ndarray:
        """A column as a numpy array, which is what every reader wants."""
        column = self.frame[name]
        if self.COLUMNS[name].kind == 'int':
            return column.to_numpy(dtype=np.int64)
        return column.to_numpy(dtype=object).astype(str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelTable):
            return NotImplemented
        return (self.column_names == other.column_names
                and all(np.array_equal(self[name], other[name])
                        for name in self.column_names))

    def __repr__(self) -> str:
        return (f'ChannelTable({self.num_channels} channels, '
                f'columns: {", ".join(self.column_names)})')

    def save(self, path: str | os.PathLike) -> None:
        """Write the table to a file of its own.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write it.

        Returns
        -------
        None
        """
        from ..io import native
        native.save(self, path)


#: built after the class, since it reads its own alias table
_ALIAS_OF = {spelling: canonical
             for canonical, spellings in ChannelTable.ALIASES.items()
             for spelling in spellings}
