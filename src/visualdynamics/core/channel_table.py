"""Channel table: per-channel test metadata (sensor, DOF, engineering units).

The table has one fixed set of columns, each with a type and a rule for
what it accepts, and a source's other columns are dropped. Because every
column says what it holds, a direction can be offered as a drop-down, a
unit can be limited by the channel's type, and an expiration is read as
a date.

A file and an edit are checked differently:

- A file is read leniently. A value that cannot be read takes the
  column's default (blank for most columns) and the import goes on,
  because refusing the import would leave the user nothing to correct,
  and this table is where they correct it. `validate.dofs` does the
  same for DOFs.
- An edit is refused. When a person types a value the column cannot
  hold, `set_cell` raises with the reason and writes nothing.
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

# One fixed set of columns rather than whatever a source carried: the
# table used to be a pandas frame that took anything, and a Rattlesnake
# import arrived with thirteen columns of controller settings behind the
# ones visualdynamics reads. A column that can hold anything is a column
# no logic can be written about.


class Spec(NamedTuple):
    """What one column holds, and what it accepts.

    `kind` is how a value is read ('int', 'index', 'float', 'choice',
    'flag', 'date', 'unit' or 'text'), `default` fills a missing or
    unreadable cell, `choices` limits a 'choice' column, `minimum`
    bounds a numeric one, and `unique` makes the column a key.
    """

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
    """The header shown for a column, in the window and in the report.

    Parameters
    ----------
    name : str
        The column name, such as 'serial_number'.

    Returns
    -------
    str
        The header, from `TITLES` or the name title-cased
        ('Serial Number').
    """
    return TITLES.get(name) or name.replace('_', ' ').title()


def COLUMNS_KIND(name: str) -> str:
    """What kind of value a column holds, its `Spec.kind`.

    Parameters
    ----------
    name : str
        The column name.

    Returns
    -------
    str
        The kind, such as 'int', 'choice' or 'text'.
    """
    return ChannelTable.COLUMNS[name].kind


def CHOICES_FOR(name: str) -> tuple[str, ...]:
    """The values a column is limited to, empty when it is not limited.

    A flag column such as 'control' returns ('True', 'False'), since a
    spreadsheet treats it as a choice of two.

    Parameters
    ----------
    name : str
        The column name.

    Returns
    -------
    tuple of str
        The allowed values.
    """
    spec = ChannelTable.COLUMNS[name]
    if spec.kind == 'flag':
        return ('True', 'False')
    return spec.choices


def canonical_name(name: str) -> str:
    """A source's column name as this schema spells it.

    The name is lower-cased, spaces and hyphens become underscores, and
    a trailing colon and anything from an opening parenthesis on are
    dropped; then `ChannelTable.ALIASES` maps the words that differ.
    Dropping the parenthetical unit reads this table's own headers back
    ('Sensitivity (mV/Unit)'), and a calibration sheet's
    ('Sensitivity (mV/g)') into the same column.

    Parameters
    ----------
    name : str
        The column name as the source wrote it.

    Returns
    -------
    str
        The schema's name for it, or the tidied name if it has none.
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
            return _datetime.datetime.strptime(
                value, form).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f'{text!r} is not a date — try YYYY-MM-DD')


#: what a table's channels can be chosen by: every channel, the control
#: channels (by their flag, whatever their role), or one role
CHANNEL_CHOICES = ('all', 'control', 'reference', 'response', 'monitor')


class ChannelTable:
    """Per-channel metadata, in one fixed set of typed columns.

    Parameters
    ----------
    columns : mapping or pandas.DataFrame
        The columns, by name, each with one entry per channel. Names are
        read through `canonical_name`, columns outside `COLUMNS` are
        dropped, missing ones take their default, and the values are
        read leniently (see the module description). A channel type left
        blank is filled in from the unit when the unit names one.

    Raises
    ------
    ValueError
        If a column in `CORE` is missing, or the columns differ in
        length.

    Attributes
    ----------
    frame : pandas.DataFrame
        The table, one row per channel, columns in `SCHEMA` order. Every
        cell is text except `channel` and `range`, which are integers.
        `node` is text too, blank where no node was recorded.
    """
    # Text and not numbers for the other columns: a serial number that
    # happens to be all digits is not a number, and a column read back
    # as int64 would lose a leading zero and come out of a round trip a
    # different string.

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
        # a Series of its own type, as the other text columns: a bare list
        # with no channels in it read as an empty float column
        frame['channel_type'] = pd.Series(cls._typed_by_unit(frame),
                                          dtype='string')
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
        """Write one cell, or refuse a value the column cannot hold.

        A unit that is not of the channel's declared type is refused;
        change the type first. Changing the type clears a unit of another
        kind.

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

        Raises
        ------
        KeyError
            If there is no column `name`.
        ValueError
            If the value cannot be read for that column, repeats a
            channel number, or disagrees with the channel's type.
        """
        # Strict where `_typed` is lenient: a person typing gets the
        # reason, where a file gets the benefit of the doubt.
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
        """The units to offer for this channel, given its declared type.

        Parameters
        ----------
        row : int
            Which channel.

        Returns
        -------
        list of str
            The shortlist for the declared type, or every unit in every
            shortlist when the type is blank.
        """
        from .unit_choices import ALL_ORDINATE_UNITS, ORDINATE_UNITS

        declared = str(self.frame.loc[row, 'channel_type'])
        return list(ORDINATE_UNITS.get(declared, ALL_ORDINATE_UNITS))

    def add_channels(self, count: int = 1) -> list[int]:
        """Add blank channels at the end of the table, numbered on from
        the highest channel in it (from 1 in an empty table).

        Every other cell takes its column's default, so the node,
        direction and unit are blank until they are typed in.

        Parameters
        ----------
        count : int, default 1
            How many channels to add.

        Returns
        -------
        list of int
            The new channels' numbers.

        Raises
        ------
        ValueError
            If `count` is less than 1.
        """
        import pandas as pd

        count = int(count)
        if count < 1:
            raise ValueError(f'cannot add {count} channels; add at least one')
        start = int(self.frame['channel'].max()) + 1 if self.num_channels else 1
        numbers = list(range(start, start + count))
        added = self._typed(pd.DataFrame(
            {name: numbers if name == 'channel' else [spec.default] * count
             for name, spec in self.COLUMNS.items()}))
        self.frame = pd.concat([self.frame, added], ignore_index=True)
        return numbers

    def chosen(self, choice: str = 'all') -> list[int]:
        """The rows of the channels `choice` picks, in table order.

        Parameters
        ----------
        choice : str, default 'all'
            One of `CHANNEL_CHOICES`: 'control' for the channels with
            Control checked, whatever their role, or a role.

        Returns
        -------
        list of int
            The rows.

        Raises
        ------
        ValueError
            If `choice` is not one of `CHANNEL_CHOICES`.
        """
        # One rule for every place channels are picked this way: the
        # arrows on a geometry and the shape values at a table's DOFs
        # (Brandon, 2026-10-22 and 2026-10-23).
        if choice not in CHANNEL_CHOICES:
            raise ValueError(f'{choice!r} is not a choice; use one of '
                             f'{CHANNEL_CHOICES}')
        if choice == 'all':
            return list(range(self.num_channels))
        if choice == 'control':
            return [int(i) for i in np.flatnonzero(self.controls())]
        return [i for i, role in enumerate(self.roles()) if role == choice]

    def choices(self) -> list[str]:
        """The `CHANNEL_CHOICES` that pick a channel, in that order:
        'all' always, then 'control' when a channel has Control checked,
        then each role a channel holds."""
        held = set(self.roles())
        return [choice for choice in CHANNEL_CHOICES
                if choice == 'all' or (choice == 'control' and self.controls().any())
                or choice in held]

    def delete_channels(self, indices: Sequence[int]) -> None:
        """Remove the given rows in place. At least one row must remain.

        Parameters
        ----------
        indices : sequence of int
            Which rows to remove.

        Returns
        -------
        None

        Raises
        ------
        IndexError
            If an index is out of range.
        ValueError
            If every row would be removed.
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
        """How many channels the table describes, one per row."""
        return len(self.frame)

    @property
    def column_names(self) -> list[str]:
        """The table's column names, in order (`title_of` gives the
        headers shown)."""
        return [str(name) for name in self.frame.columns]

    def orientation(self, row: int, geometry: Any
                    ) -> tuple[np.ndarray | None, str | None, float | None]:
        """One channel's measured direction on a geometry.

        The row's DOF (node and direction) is read through the coordinate
        system its node is measured in (`Geometry.dof_direction`). The
        result is that direction as a unit vector in the geometry's
        global system, the signed global axis nearest it ('X+', 'Z-'),
        and the angle between the two in degrees. These are the values
        the derived columns show.

        Parameters
        ----------
        row : int
            The channel's row.
        geometry : Geometry
            The geometry the channel is measured on.

        Returns
        -------
        tuple
            (vector, axis, angle), each None when the geometry is None
            or cannot place the channel: the node is not on it, the
            direction is missing, or the node is measured in a
            coordinate system that turns with position.
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
        """The derived columns' cells for one row, as text.

        The three components of the direction to three decimals, the
        nearest axis, and the angle to a tenth of a degree; all empty
        when the geometry cannot place the channel.

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
        """Each channel's degree of freedom, as '101Z+' strings: the node
        and direction cells joined, so a missing one is left out."""
        return [f'{node}{direction}' for node, direction
                in zip(self.frame['node'], self.frame['direction'])]

    def rename_dof(self, old: str, new: str,
                   quantity: str | None = None) -> int:
        """Give the channel at coordinate `old` the coordinate `new`, in
        place, by rewriting its node and direction cells.

        This is the same correction as `DataArray.rename_dof`, made in the
        table that names the channels. Only the channels at `old` that
        match `quantity` are changed, not every channel at the node.

        Parameters
        ----------
        old : str
            The coordinate as the table has it, '101Z+'.
        new : str
            The coordinate to give it: a node number then a direction,
            normalized as every DOF is. It is refused if either part is
            missing, although a table row may lack one.
        quantity : str, optional
            Which channel at `old`, by the quantity its unit names
            ('acceleration', 'force' and so on; 'unknown' where the unit
            names none). Every channel at the coordinate when omitted.

        Returns
        -------
        int
            How many channels changed; 0 when `new` equals `old`.

        Raises
        ------
        ValueError
            If `new` is not a whole DOF, or no channel matches.
        """
        # The table's version of a data array's `rename_dof` (Brandon,
        # 2026-09-06), with the same rule: the channel moves, not every
        # channel at the point.
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

    def identities(self, history: Any = None) -> list[Any]:
        """Each row as the (DOF, quantity) channel of `history` it names.

        The row's channel type is the quantity, the same word a time
        history's `ordinate_dim` uses. A row with no channel type names
        the history's channel at its DOF when there is exactly one
        there. A row that names none of the history's channels is None.

        Parameters
        ----------
        history : TimeHistory, optional
            The recording the table describes. Without one, each row's
            own (DOF, type) is returned.

        Returns
        -------
        list of (str, str) or None
            One entry per row.
        """
        # The rule the project's role pairing used privately, made the
        # table's so the overload check reads ranges by the same match.
        dofs, types = self.dof_strings(), self.types()
        if history is None:
            return list(zip(dofs, types))
        identities = history.channel_identities()
        known = set(identities)
        out: list[Any] = []
        for dof, kind in zip(dofs, types):
            if kind:
                out.append((dof, kind) if (dof, kind) in known else None)
            else:
                at = [identity for identity in identities
                      if identity[0] == dof]
                out.append(at[0] if len(at) == 1 else None)
        return out

    def roles(self) -> list[str]:
        """Each channel's declared role ('reference', 'response' or
        'monitor'), '' where undeclared."""
        return [str(v) for v in self.frame['role']]

    def controls(self) -> np.ndarray:
        """Which channels are control channels, as a boolean array."""
        return np.array([_parse_flag(str(v)) for v in self.frame['control']])

    def types(self) -> list[str]:
        """Each channel's type (what it measures), '' where undeclared."""
        return [str(v) for v in self.frame['channel_type']]

    def sensitivities(self) -> np.ndarray:
        """Each channel's sensitivity in mV per engineering unit, NaN where
        undeclared."""
        return np.array([float(v) if str(v).strip() else np.nan
                         for v in self.frame['sensitivity']])

    def ranges(self) -> np.ndarray:
        """Each channel's instrumentation voltage range, in volts, as
        integers (5 by default)."""
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
        """Write the table to a file of its own, in the `.vdyn` format.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write it; `.vdyn` is added when the suffix is not
            there.

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
