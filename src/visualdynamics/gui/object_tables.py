"""Table models for Visual Dynamics objects.

Each model is a list of columns over an existing object. It reads its
cells from the object and, where editing makes sense, writes an edit
straight back to it. In a mode table, the frequency and damping a fit
produced are read-only, while the description beside them is free text
to fill in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ..core.channel_table import title_of
from ..core.unit_choices import (
    ALL_ORDINATE_UNITS,
    shown_dimension,
    stored_dimension,
)
from ..units import UNKNOWN, dimension_of, parse_dimension, si_transform
from ..viz.geometry import COLOR_NAMES, color_name, color_rgb
from .icons import color_swatch
from .tables import Column, TableModel, TransposedTableModel

if TYPE_CHECKING:                                    # pragma: no cover
    from PySide6.QtCore import QObject

    from ..core.channel_table import ChannelTable
    from ..core.data import DataArray
    from ..core.geometry import Geometry
    from ..core.matches import MatchedModes
    from ..core.modal_fit import ModalFitSession
    from ..core.shapes import ShapeSet
    from ..units import UnitSystem

LEFT = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


def shape_table_model(shapes: ShapeSet,
                      parent: QObject | None = None) -> TableModel:
    """A mode set as a table: each mode's frequency, damping and modal
    mass, read-only, and a description to fill in."""
    columns = [
        Column('Mode', lambda s, r: r + 1),
        Column('Frequency [Hz]', lambda s, r: float(s.frequency[r]),
               format=lambda v: f'{v:.4f}'),
        Column('Damping [%]', lambda s, r: float(s.damping[r]) * 100.0,
               format=lambda v: f'{v:.3f}'),
        Column('Modal mass (unscaled)'
               if getattr(shapes, 'unscaled', False) else 'Modal mass',
               lambda s, r: float(s.modal_mass[r]),
               format=lambda v: f'{v:.4g}'),
        Column('Description', lambda s, r: s.description[r],
               set=_set_description,
               journal=lambda s, r, t: f'.description[{r}] = {t.strip()!r}',
               alignment=LEFT),
    ]
    return TableModel(shapes, columns, lambda s: s.num_shapes, parent)


def _set_description(shapes, row, text):
    shapes.description[row] = text.strip()


def matched_modes_model(matched: MatchedModes, objects: Mapping[str, Any],
                        parent: QObject | None = None) -> TableModel:
    """A MatchedModes object as a table, in the mode table's format.

    The frequencies and damping are read by name from the mode sets in
    `objects` when the model is built, so they show the sets as they
    are now. The MAC is the value stored on the object, as the
    comparison computed it (possibly from projected shapes). A set
    missing from the project, or a mode index past its end, shows
    dashes.
    """
    from ..core.tables import MATCHED_DECIMALS, cell_text, matched_rows

    headers, rows = matched_rows(matched, objects)
    columns = [Column(title, lambda s, r, i=i: s[r][i],
                      format=partial(cell_text, decimals=decimals))
               for i, (title, decimals)
               in enumerate(zip(headers, MATCHED_DECIMALS))]
    return TableModel(rows, columns, len, parent)


def _column_getter(name, index):
    def getter(table: ChannelTable, row: int) -> Any:
        return table[name][index(row)]
    return getter


def _column_setter(name, index):
    def setter(table: Any, row: int, text: str) -> None:
        # the interface says 'displacement' where the schema stores
        # 'length'; every other word is its own
        table.set_cell(name, index(row), stored_dimension(text)
                       if name == 'channel_type' else text)
    return setter


def _column_journal(name, index):
    def journal(_table, row, text):
        stored = (stored_dimension(text) if name == 'channel_type'
                  else text)
        return f'.set_cell({name!r}, {index(row)}, {stored!r})'
    return journal


def channel_table_model(table: ChannelTable, parent: QObject | None = None,
                        rows: Sequence[int] | None = None,
                        geometry: Geometry | None = None,
                        set_role: Callable | None = None) -> TableModel:
    """A channel table, with an editor of the right type for each
    column.

    The columns come from the table's own `COLUMNS` spec, so a schema
    change reaches the interface without a second edit. A choice column
    gets a drop-down, a date a calendar, a flag a check box, and the
    unit column offers only the units that fit what the channel says it
    measures.

    With a `geometry`, derived columns are added after the table's own:
    each channel's measured direction as a unit vector in the
    geometry's global system, the nearest global axis, and the angle to
    it. They are computed from the geometry each time they are shown,
    and again whenever the node or the direction in that row is edited,
    so they always agree with it. They are not editable, because the
    table does not store them.

    `rows`, when given, shows only those channels, so the table shows
    the rows picked in the tree's grid and nothing else.

    `set_role(row, role)`, when given, receives the Role column's edits
    in place of the table. A table linked with the time data it
    describes shows that data's roles, and a change made here goes to
    the data (`Project.set_channel_role`), which journals it.
    """
    # The derived direction columns: Brandon, 2026-09-20.
    rows = None if rows is None else [int(r) for r in rows]
    index = (lambda row: row) if rows is None else rows.__getitem__
    columns = []
    for name, spec in table.COLUMNS.items():
        title = title_of(name)
        if spec.kind == 'flag':
            columns.append(Column(
                title,
                lambda t, r, n=name, i=index: bool(t.controls()[i(r)])
                if n == 'control' else bool(t[n][i(r)]),
                set=_column_setter(name, index),
                journal=_column_journal(name, index), checkbox=True))
            continue
        if spec.kind == 'choice':
            shown = ([shown_dimension(c) for c in spec.choices]
                     if name == 'channel_type' else list(spec.choices))
            routed = name == 'role' and set_role is not None
            columns.append(Column(
                title, _shown_getter(name, index),
                set=((lambda _t, r, text, i=index: set_role(i(r), text))
                     if routed else _column_setter(name, index)),
                # the verb journals itself
                journal=((lambda _t, _r, _text: None) if routed
                         else _column_journal(name, index)),
                choices=shown,
                # a direction moves the derived columns beside it
                affects_row=name in ('channel_type', 'direction'),
                alignment=LEFT))
            continue
        if spec.kind == 'unit':
            # what this channel could be in, given what it says it is
            columns.append(Column(
                title, _column_getter(name, index),
                set=_column_setter(name, index),
                journal=_column_journal(name, index),
                choices=ALL_ORDINATE_UNITS,
                row_choices=lambda t, r, i=index: t.units_for(i(r)),
                alignment=LEFT))
            continue
        columns.append(Column(
            title, _column_getter(name, index),
            set=_column_setter(name, index),
            journal=_column_journal(name, index),
            date=spec.kind == 'date',
            # a node moves the derived columns beside it
            affects_row=name == 'node', alignment=LEFT))
    if geometry is not None:
        from ..core.channel_table import DERIVED_COLUMNS

        for k, name in enumerate(DERIVED_COLUMNS):
            columns.append(Column(
                title_of(name),
                lambda t, r, k=k, i=index, g=geometry: t.derived_cells(i(r), g)[k],
                alignment=LEFT))
    count = ((lambda t: t.num_channels) if rows is None
             else (lambda t: len(rows)))
    return TableModel(table, columns, count, parent)


def _shown_getter(name, index):
    """A choice cell, in the word the interface uses for it."""
    def getter(table: ChannelTable, row: int) -> Any:
        return shown_dimension(str(table[name][index(row)]))
    return getter


def modal_fit_table_model(session: ModalFitSession,
                          parent: QObject | None = None) -> TableModel:
    """The modes fitted so far, plus the one the cursor is placing.

    The last row, labeled "new", is the pending mode. Its frequency is
    the cursor's on the CMIF, its damping is the half-power estimate
    until a value is typed, and Confirm Mode makes it a fitted row. A
    fitted mode's damping stays editable, in percent between 0 and 100
    (exclusive), and the synthesis uses the new value.
    The description is free text.
    """
    def pending(row: int) -> bool:
        return row == len(session.modes)

    def entry(row: int, key: str) -> Any:
        return (session.pending[key] if pending(row)
                else session.modes[row][key])

    def set_damping(session: ModalFitSession, row: int, text: str) -> None:
        value = float(text.rstrip('%')) / 100.0
        if not 0.0 < value < 1.0:
            raise ValueError('damping is a fraction of critical, 0-100%')
        if pending(row):
            session.pending['damping'] = value
            session.pending['overridden'] = True
        else:
            session.modes[row]['damping'] = value

    def set_description(session: Any, row: int, text: str) -> None:
        if pending(row):
            session.pending['description'] = text.strip()
        else:
            session.modes[row]['description'] = text.strip()

    columns = [
        Column('Mode', lambda s, r: 'new' if pending(r) else r + 1),
        Column('Frequency [Hz]', lambda s, r: float(entry(r, 'frequency')),
               format=lambda v: f'{v:.4f}'),
        Column('Damping [%]', lambda s, r: float(entry(r, 'damping')) * 100.0,
               format=lambda v: f'{v:.3f}', set=set_damping,
               journal=_elsewhere),
        Column('Description', lambda s, r: entry(r, 'description'),
               set=set_description, journal=_elsewhere,
               alignment=LEFT),
    ]
    return TableModel(session, columns,
                          lambda s: len(s.modes) + 1, parent)


# ---- declaring the units data arrived in ------------------------------------

def units_table_model(obj: Any, records: Sequence[int] | None = None,
                      parent: QObject | None = None) -> TableModel:
    """A table with one row per item of `obj` that needs a unit.

    A geometry has one row, for its coordinates, and a mode set one,
    for its shapes. A data object has one row per channel among the
    records in `records` (all of them when None); for a PSD with cross
    records, a cross term's unit follows from its two channels'.

    It is a table rather than a dialog so the plot or the model stays on
    screen beside it, and units copy, paste and clear like any other
    column.
    """
    from ..core.data import Psd
    from ..core.geometry import Geometry
    from ..core.shapes import ShapeSet
    from ..core.unit_choices import LENGTH_UNITS, MASS_UNITS

    if isinstance(obj, Psd) and _has_cross_records(obj):
        return _matrix_units_model(obj, records, parent)
    if isinstance(obj, Geometry):
        return _one_unit_model(obj, 'Coordinates', 'length', LENGTH_UNITS,
                               'length_unit', parent)
    if isinstance(obj, ShapeSet):
        # What a mass-normalized shape is *in* is 1/sqrt(mass), and a cell
        # reading 'kg' does not say that — the same reasoning that shows a
        # PSD's unit as g**2/Hz. The mass unit is what is stored; the
        # radical is what is shown and offered.
        return _one_unit_model(obj, 'Mode shapes', 'mass',
                               [f'1/\u221a{unit}' for unit in MASS_UNITS],
                               'mass_unit', parent,
                               shown=lambda unit: f'1/\u221a{unit}',
                               given=_mass_unit_given,
                               type_label='modal mass')
    return _channel_units_model(obj, records, parent)


def _mass_unit_given(text):
    """'1/√kg' (or '1/sqrt(kg)', or plain 'kg') -> 'kg'."""
    text = text.strip()
    for prefix, suffix in (('1/\u221a', ''), ('1/sqrt(', ')'),
                           ('1/SQRT(', ')')):
        if text.startswith(prefix) and text.endswith(suffix):
            return text[len(prefix):len(text) - len(suffix) or None]
    return text


def _one_unit_model(obj, label, dimension, choices, attribute, parent=None,
                    shown=None, given=None, type_label=None):
    """An object measured in a single unit throughout: geometry, shapes.

    Still a table, and still the same table, so that clearing a unit or
    picking one works the way it does everywhere else. `shown`/`given` remap
    display and entry when the stored unit is not the quantity's own — a
    shape set stores its mass unit but *is* in 1/sqrt(mass). `type_label`
    names the quantity when the dimension that validates it does not —
    'modal mass' is validated as a mass.
    """
    shown = shown or (lambda unit: unit)
    given = given or (lambda text: text)
    type_label = type_label or dimension

    def set_unit(obj: Any, row: int, text: str) -> None:
        text = given(text.strip()).replace('^', '**')
        if not text:
            obj.undefine_units()
            return
        si_transform(text, dimension)   # refuse before anything is changed
        obj.define_units(text)

    def get_unit(obj: Any, _row: int) -> str:
        unit = getattr(obj, attribute)
        return shown(unit) if unit else ''

    columns = [
        Column('Item', lambda _obj, _row: label, alignment=LEFT),
        Column('Type', lambda _obj, _row: type_label, alignment=LEFT),
        Column('Unit', get_unit,
               set=set_unit, journal=_elsewhere,
               choices=choices, choices_editable=True,
               alignment=LEFT),
    ]
    return TableModel(obj, columns, lambda _obj: 1, parent)


def _has_cross_records(data):
    return (data.reference_dof is not None
            and any(r != s for r, s in zip(data.response_dof,
                                           data.reference_dof)))


def _matrix_units_model(data, records=None, parent=None):
    """A CPSD matrix has N channel units, not N² record units.

    A cross term is the product of two channels' spectra, so its unit is
    the product of theirs per Hz — it is never declared on its own, or two
    declarations could disagree about one channel. One row per channel;
    every record both of whose channels are named converts, however the
    pane was opened, and the rest wait for their other half.
    """
    from ..core.unit_choices import (
        ALL_ORDINATE_UNITS,
        ORDINATE_UNITS,
        engineering_unit,
        squared_per_hz,
        units_for,
    )

    wanted = (list(range(data.num_records)) if records is None
              else [int(i) for i in records])
    channels = []
    for i in wanted:
        for dof in (data.response_dof[i], data.reference_dof[i]):
            if dof not in channels:
                channels.append(dof)

    def involving(dof: str) -> list[int]:
        return [i for i in range(data.num_records)
                if dof in (data.response_dof[i], data.reference_dof[i])]

    def stored_unit(dof: str) -> str | None:
        """The unit a channel's records already carry, if any do."""
        for i in range(data.num_records):
            if data.response_dof[i] == dof and data.ordinate_unit[i]:
                return data.ordinate_unit[i]
            if data.reference_dof[i] == dof and data.reference_unit[i]:
                return data.reference_unit[i]
        return None

    declared = {dof: stored_unit(dof) for dof in channels}

    def unit_for(dof: str) -> str | None:
        """What a channel is declared in — including channels off the pane,
        whose records may carry units from an earlier declaration."""
        return declared[dof] if dof in declared else stored_unit(dof)

    def apply(dof: str) -> None:
        for i in involving(dof):
            unit = unit_for(data.response_dof[i])
            reference = unit_for(data.reference_dof[i])
            if unit and reference:
                data.define_units({i: unit}, {i: reference})

    def withdraw(dof: str) -> None:
        declared[dof] = None
        data.undefine_units(involving(dof))

    def set_unit(data: Any, row: int, text: str) -> None:
        text = engineering_unit(text.strip()).replace('^', '**')
        if not text:
            withdraw(channels[row])
            return
        si_transform(text)        # raises before anything is changed
        declared[channels[row]] = text
        apply(channels[row])
        # a chosen unit IS a statement of type (Brandon, 2026-08-30)
        types[row] = dimension_of(text) or types[row]

    def hinted_type(dof: str) -> str:
        """The quantity a channel was taken to hold, from any record's
        dimension — a channel referenced in a cross term is named by the
        trailing factor of the numerator, a response by the leading one."""
        for i in range(data.num_records):
            dim = data.known_dim(i)
            if dim == UNKNOWN:
                continue
            factors = [name for name, power in parse_dimension(dim)
                       if power > 0]
            if data.response_dof[i] == dof and factors:
                return factors[0]
            if data.reference_dof[i] == dof and factors:
                return factors[-1]
        return ''

    types = [hinted_type(dof) for dof in channels]

    def set_type(data: Any, row: int, text: str) -> None:
        text = stored_dimension(text)
        if text and text not in ORDINATE_UNITS:
            raise ValueError('expected one of ' + str(sorted(
                shown_dimension(d) for d in ORDINATE_UNITS)))
        types[row] = text
        unit = unit_for(channels[row])
        # a unit of some other kind is no longer an answer here
        if unit and (not text or dimension_of(unit) != text):
            withdraw(channels[row])

    columns = [
        Column('Channel', lambda d, r: channels[r], alignment=LEFT),
        Column('Type', lambda d, r: shown_dimension(types[r]),
               set=set_type, journal=_elsewhere,
               choices=[shown_dimension(d) for d in ORDINATE_UNITS],
               affects_row=True, alignment=LEFT),
        Column('Unit',
               lambda d, r: squared_per_hz(unit_for(channels[r]) or ''),
               set=set_unit, journal=_elsewhere,
               choices=[squared_per_hz(u) for u in ALL_ORDINATE_UNITS],
               row_choices=lambda d, r: [squared_per_hz(u)
                                         for u in units_for(types[r])],
               choices_editable=True, carries=('Type',),
               affects_row=True, alignment=LEFT),
    ]
    return TableModel(data, columns, lambda d: len(channels), parent)


def _channel_units_model(data, records=None, parent=None):
    """Channels that each name one unit: time histories, spectra, PSDs.

    Data naming two units per record — an FRF — is declared per side in
    `frf_units_models` instead, so nothing here is half of anything.
    """
    from ..core.data import Psd
    from ..core.unit_choices import (
        ALL_ORDINATE_UNITS,
        ORDINATE_UNITS,
        base_dimension,
        engineering_unit,
        squared_per_hz,
        units_for,
    )

    rows = (list(range(data.num_records)) if records is None
            else [int(i) for i in records])
    # A PSD holds the square-per-Hz of an engineering unit. A cell reading
    # 'g' does not say that, so the whole label is shown and picked instead.
    squared = isinstance(data, Psd)
    shown = squared_per_hz if squared else (lambda unit: unit)
    given = engineering_unit if squared else (lambda text: text)

    def set_unit(data: Any, row: int, text: str) -> None:
        text = given(text.strip()).replace('^', '**')
        if not text:
            data.undefine_units([rows[row]])
            return
        si_transform(text)        # raises before anything is changed
        data.define_units({rows[row]: text})
        # a chosen unit IS a statement of type (Brandon, 2026-08-30):
        # lbf on a typeless row must not leave Type reading empty
        types[row] = dimension_of(text) or types[row]

    # What each channel was taken to be when the pane opened, and what the
    # user has since said it is. Held here rather than recomputed from the
    # declared unit: narrowing to the unit just chosen would fight anyone
    # correcting a mistake, which is the one time the whole list is wanted.
    types = [base_dimension(data.known_dim(i)) for i in rows]
    types = ['' if t == 'unknown' else t for t in types]

    def set_type(data: Any, row: int, text: str) -> None:
        text = stored_dimension(text)
        if text and text not in ORDINATE_UNITS:
            raise ValueError('expected one of ' + str(sorted(
                shown_dimension(d) for d in ORDINATE_UNITS)))
        types[row] = text
        unit = data.ordinate_unit[rows[row]]
        # a unit of some other kind is no longer an answer here
        if unit and (not text or dimension_of(unit) != text):
            data.undefine_units([rows[row]])

    columns = [
        Column('Channel', lambda d, r: d.record_label(rows[r]), alignment=LEFT),
        Column('Type', lambda d, r: shown_dimension(types[r]),
               set=set_type, journal=_elsewhere,
               choices=[shown_dimension(d) for d in ORDINATE_UNITS],
               affects_row=True, alignment=LEFT),
        Column('Unit',
               lambda d, r: shown(d.ordinate_unit[rows[r]] or ''),
               set=set_unit, journal=_elsewhere,
               choices=[shown(u) for u in ALL_ORDINATE_UNITS],
               row_choices=lambda d, r: [shown(u) for u in units_for(types[r])],
               choices_editable=True, carries=('Type',),
               affects_row=True, alignment=LEFT),
    ]
    return TableModel(data, columns, lambda d: len(rows), parent)


def frf_units_models(data: DataArray, records: Sequence[int] | None = None,
                     parent: QObject | None = None
                     ) -> tuple[TableModel, TableModel]:
    """Two unit tables for an FRF: its response channels and its
    reference channels.

    An FRF matrix of N x M records has only N + M physical channels,
    and a unit belongs to a channel, so each channel's unit is declared
    once and every record converts as soon as both of its channels have
    one. The two sides stay separate even where the DOF labels are the
    same: a drive point's accelerometer and its force gauge share a
    label and are two different channels.
    """
    from ..core.unit_choices import (
        ALL_ORDINATE_UNITS,
        ORDINATE_UNITS,
        REFERENCE_UNITS,
        units_for,
    )

    wanted = (list(range(data.num_records)) if records is None
              else [int(i) for i in records])
    sides = {
        'response': (data.response_dof, data.ordinate_unit),
        'reference': (data.reference_dof, data.reference_unit),
    }

    channels = {side: [] for side in sides}
    for i in wanted:
        for side, (dofs, _) in sides.items():
            if dofs[i] not in channels[side]:
                channels[side].append(dofs[i])

    def involving(side: str, dof: str) -> list[int]:
        dofs = sides[side][0]
        return [i for i in range(data.num_records) if dofs[i] == dof]

    def stored_unit(side: str, dof: str) -> str | None:
        """The unit a channel's records already carry, if any do."""
        dofs, units = sides[side]
        for i in range(data.num_records):
            if dofs[i] == dof and units[i]:
                return units[i]
        return None

    declared = {(side, dof): stored_unit(side, dof)
                for side in sides for dof in channels[side]}

    def unit_for(side: str, dof: str) -> str | None:
        """What a channel is declared in — including channels off the pane,
        whose records may carry units from an earlier declaration."""
        key = (side, dof)
        return declared[key] if key in declared else stored_unit(side, dof)

    def apply(side: str, dof: str) -> None:
        for i in involving(side, dof):
            unit = unit_for('response', data.response_dof[i])
            reference = unit_for('reference', data.reference_dof[i])
            if unit and reference:
                data.define_units({i: unit}, {i: reference})

    def withdraw(side: str, dof: str) -> None:
        declared[(side, dof)] = None
        data.undefine_units(involving(side, dof))

    def hinted_type(side: str, dof: str) -> str:
        """The quantity a channel was taken to hold: an FRF's dimension is
        response over reference, and the side says which half is ours."""
        for i in involving(side, dof):
            response, _, reference = data.known_dim(i).partition('/')
            part = response if side == 'response' else reference
            if part and part != UNKNOWN:
                return part
        return ''

    def side_model(side: str,
                   unit_choices: Sequence[str]) -> TableModel:
        dofs = channels[side]
        types = [hinted_type(side, dof) for dof in dofs]

        def set_type(data: Any, row: int, text: str) -> None:
            text = stored_dimension(text)
            if text and text not in ORDINATE_UNITS:
                raise ValueError('expected one of ' + str(sorted(
                    shown_dimension(d) for d in ORDINATE_UNITS)))
            types[row] = text
            unit = unit_for(side, dofs[row])
            # a unit of some other kind is no longer an answer here
            if unit and (not text or dimension_of(unit) != text):
                withdraw(side, dofs[row])

        def set_unit(data: Any, row: int, text: str) -> None:
            text = text.strip().replace('^', '**')
            if not text:
                withdraw(side, dofs[row])
                return
            si_transform(text)    # raises before anything is changed
            declared[(side, dofs[row])] = text
            apply(side, dofs[row])
            # a chosen unit IS a statement of type (Brandon, 2026-08-30)
            types[row] = dimension_of(text) or types[row]

        columns = [
            Column('Channel', lambda d, r: dofs[r], alignment=LEFT),
            Column('Type', lambda d, r: shown_dimension(types[r]),
                   set=set_type, journal=_elsewhere,
                   choices=[shown_dimension(d) for d in ORDINATE_UNITS],
                   affects_row=True, alignment=LEFT),
            Column('Unit', lambda d, r: unit_for(side, dofs[r]) or '',
                   set=set_unit, journal=_elsewhere,
                   choices=unit_choices,
                   row_choices=lambda d, r: units_for(types[r], unit_choices),
                   choices_editable=True, carries=('Type',),
                   affects_row=True, alignment=LEFT),
        ]
        return TableModel(data, columns, lambda d: len(dofs), parent)

    return (side_model('response', ALL_ORDINATE_UNITS),
            side_model('reference', REFERENCE_UNITS))


# ---- geometry entities ------------------------------------------------------

#: a column journaled by another surface — the units pane — declares
#: this: editable, and deliberately quiet here
JOURNALED_ELSEWHERE = 'elsewhere'


def _elsewhere(_obj, _row, _text):
    return None


_elsewhere.tag = JOURNALED_ELSEWHERE


def _coordinate_setter(axis, unit_system):
    """Write a node coordinate, converting from the displayed unit."""
    def setter(geometry: Any, row: int, text: str) -> None:
        value = float(text)
        if geometry.units_defined:
            value = unit_system.to_si(value, 'length')
        geometry.node_xyz[row, axis] = value
    return setter


def _coordinate_getter(axis, unit_system):
    def getter(geometry: Geometry, row: int) -> Any:
        value = float(geometry.node_xyz[row, axis])
        if geometry.units_defined:
            value = unit_system.from_si(value, 'length')
        return value
    return getter


def _set_int_field(array_name, minimum=None):
    def setter(geometry: Any, row: int, text: str) -> None:
        value = int(float(text))
        if minimum is not None and value < minimum:
            raise ValueError(f'must be at least {minimum}')
        getattr(geometry, array_name)[row] = value
    return setter


def _int_field_journal(array_name):
    def journal(geometry, row, _text):
        return (f'.{array_name}[{row}] = '
                f'{int(getattr(geometry, array_name)[row])}')
    return journal


def _coordinate_journal(axis):
    def journal(geometry, row, _text):
        # post-state, in SI: the setter converted, and the echo reads
        # what actually landed
        return (f'.node_xyz[{row}, {axis}] = '
                f'{float(geometry.node_xyz[row, axis])!r}')
    return journal


def _renumber_journal(method_name):
    def journal(_geometry, row, text):
        return f'.{method_name}({row}, {int(float(text))})'
    return journal


def _connectivity_journal(attribute):
    def journal(geometry, row, _text):
        nodes = [int(n) for n in getattr(geometry, attribute)[row]]
        return f'.{attribute}[{row}] = np.array({nodes!r})'
    return journal


def _color_column(title, array_name):
    """A color as its name and swatch, chosen from the palette.

    The stored value stays the UFF-style palette index; the index itself is
    still accepted as text, so numeric colors paste in from a spreadsheet.
    """
    def setter(geometry: Any, row: int, text: str) -> None:
        text = text.strip()
        allowed = f'color must be a palette name or 0-{len(COLOR_NAMES) - 1}'
        if text.lower() in COLOR_NAMES:
            value = COLOR_NAMES.index(text.lower())
        else:
            try:
                value = int(float(text))
            except ValueError:
                raise ValueError(allowed) from None
            if not 0 <= value < len(COLOR_NAMES):
                raise ValueError(allowed)
        getattr(geometry, array_name)[row] = value

    return Column(
        title,
        lambda g, r: int(getattr(g, array_name)[r]),
        set=setter,
        journal=_int_field_journal(array_name),
        format=color_name,
        decoration=lambda g, r: QColor.fromRgbF(
            *color_rgb(getattr(g, array_name)[r])),
        choices=COLOR_NAMES,
        choice_icon=lambda name: color_swatch(color_rgb(COLOR_NAMES.index(name))),
        alignment=LEFT,
    )


def _renumber(method_name):
    """Set an id that has to stay unique.

    The geometry does the work: an id is what everything else references,
    so renumbering has to refuse collisions and carry the references over.
    """
    def setter(geometry: Any, row: int, text: str) -> None:
        value = int(float(text))
        if value < 1:
            raise ValueError('must be at least 1')
        getattr(geometry, method_name)(row, value)
    return setter


def _set_cs_reference(array_name):
    """Point a node at a coordinate system, which has to exist."""
    def setter(geometry: Any, row: int, text: str) -> None:
        value = int(float(text))
        if value not in geometry.cs_id:
            raise ValueError(f'no coordinate system {value}')
        getattr(geometry, array_name)[row] = value
    return setter


def node_table_model(geometry: Geometry, unit_system: UnitSystem,
                     parent: QObject | None = None) -> TableModel:
    unit = (unit_system.label_text('length') if geometry.units_defined
            else 'units undefined')
    columns = [
        Column('Node', lambda g, r: int(g.node_id[r]),
               set=_renumber('renumber_node'),
               journal=_renumber_journal('renumber_node')),
        *[Column(f'{axis} [{unit}]', _coordinate_getter(i, unit_system),
                 set=_coordinate_setter(i, unit_system),
                 journal=_coordinate_journal(i),
                 format=lambda v: f'{v:.6g}')
          for i, axis in enumerate('XYZ')],
        _color_column('Color', 'node_color'),
        Column('Placement CS', lambda g, r: int(g.node_def_cs[r]),
               set=_set_cs_reference('node_def_cs'),
               journal=_int_field_journal('node_def_cs')),
        Column('Displacement CS', lambda g, r: int(g.node_disp_cs[r]),
               set=_set_cs_reference('node_disp_cs'),
               journal=_int_field_journal('node_disp_cs')),
    ]
    return TableModel(geometry, columns, lambda g: g.num_nodes, parent)


def coordinate_system_table_model(geometry: Geometry,
                                  unit_system: UnitSystem,
                                  parent: QObject | None = None
                                  ) -> TableModel:
    from ..core.geometry import CS_TYPES

    unit = (unit_system.label_text('length') if geometry.units_defined
            else 'units undefined')

    def origin_getter(axis: int) -> Callable[[Geometry, int], float]:
        def getter(geometry: Geometry, row: int) -> Any:
            value = float(geometry.cs_matrix[row, 3, axis])
            if geometry.units_defined:
                value = unit_system.from_si(value, 'length')
            return value
        return getter

    def origin_setter(axis: int) -> Callable[[Geometry, int, str], None]:
        def setter(geometry: Any, row: int, text: str) -> None:
            value = float(text)
            if geometry.units_defined:
                value = unit_system.to_si(value, 'length')
            geometry.cs_matrix[row, 3, axis] = value
        return setter

    def set_name(geometry: Any, row: int, text: str) -> None:
        geometry.cs_name[row] = text

    def set_type(geometry: Any, row: int, text: str) -> None:
        wanted = text.strip().lower()
        for code, name in CS_TYPES.items():
            if name == wanted:
                geometry.cs_type[row] = code
                return
        raise ValueError(f'expected one of {sorted(CS_TYPES.values())}')

    columns = [
        Column('CS', lambda g, r: int(g.cs_id[r]),
               set=_renumber('renumber_coordinate_system'),
               journal=_renumber_journal('renumber_coordinate_system')),
        Column('Name', lambda g, r: g.cs_name[r], set=set_name,
               journal=lambda g, r, t: f'.cs_name[{r}] = {t!r}',
               alignment=LEFT),
        Column('Type', lambda g, r: CS_TYPES.get(int(g.cs_type[r]),
                                                 'cartesian'),
               set=set_type,
               journal=lambda g, r, _t:
               f'.cs_type[{r}] = {int(g.cs_type[r])}',
               alignment=LEFT,
               # the only three there are, in code order
               choices=[CS_TYPES[code] for code in sorted(CS_TYPES)]),
        *[Column(f'Origin {axis} [{unit}]', origin_getter(i),
                 set=origin_setter(i),
                 journal=lambda g, r, _t, i=i:
                 f'.cs_matrix[{r}, 3, {i}] = '
                 f'{float(g.cs_matrix[r, 3, i])!r}',
                 format=lambda v: f'{v:.6g}')
          for i, axis in enumerate('XYZ')],
    ]
    return TableModel(geometry, columns, lambda g: len(g.cs_id), parent)


def _connectivity_text(nodes):
    return ' '.join(str(int(node)) for node in nodes)


def _set_connectivity(attribute, geometry_check=True):
    def setter(geometry: Any, row: int, text: str) -> None:
        nodes = [int(part) for part in text.replace(',', ' ').split()]
        if geometry_check:
            missing = set(nodes) - set(geometry.node_id.tolist())
            if missing:
                raise ValueError(f'unknown nodes {sorted(missing)}')
        getattr(geometry, attribute)[row] = np.asarray(nodes, dtype=np.int64)
    return setter


def id_runs_text(ids) -> str:
    """Ids as sorted runs, without repeats: `[1,2,3,7,9,10]` ->
    `'1-3 7 9-10'`.

    An element group can hold hundreds of elements (478 in one group of
    the demonstration drone), and they come in runs because a mesh is
    built one part at a time. Written out one by one, the drone's 22
    element groups take 21 731 characters; as runs they take 595, which
    fits in a cell a person can read.
    """
    ids = np.unique(np.asarray(list(ids), dtype=np.int64))
    if not len(ids):
        return ''
    runs, start, previous = [], int(ids[0]), int(ids[0])
    for value in ids[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return ' '.join(str(a) if a == z else f'{a}-{z}' for a, z in runs)


def parse_id_runs(text: str) -> list[int]:
    """The inverse of `id_runs_text`: `'1-3, 7 9-10'` ->
    `[1, 2, 3, 7, 9, 10]`.

    Runs are separated by commas or spaces, and a run can be written
    either way round: `10-7` is the same four elements as `7-10`.
    """
    found: list[int] = []
    for part in str(text).replace(',', ' ').split():
        if '-' in part[1:]:
            head, _, tail = part[1:].partition('-')
            low, high = int(part[0] + head), int(tail)
            if low > high:
                low, high = high, low
            found.extend(range(low, high + 1))
        else:
            found.append(int(part))
    return found


def _group_elements_text(geometry: Geometry, row: int) -> str:
    group = int(geometry.group_id[row])
    return id_runs_text(geometry.elem_id[geometry.elem_group == group])


def _set_group_elements(geometry: Any, row: int, text: str) -> None:
    """Claim these elements for this element group.

    Editing the list *moves elements in*: an element named here leaves
    whatever element group it was in, because it can only be in one. That also
    means the list cannot be used to take an element *out* — every
    element is in exactly one element group, always (the constructor refuses
    strays, and deleting an element group rehouses its elements rather than
    orphaning them), so a removal with no destination is not a state
    the geometry can hold. Removing is done by adding: name the
    element in the element group it should go to, and it leaves this one by
    itself. The refusal says so rather than guessing a destination.
    """
    wanted = parse_id_runs(text)
    known = {int(i) for i in geometry.elem_id}
    missing = sorted(set(wanted) - known)
    if missing:
        raise ValueError(f'unknown elements {missing}')
    group = int(geometry.group_id[row])
    here = {int(i) for i in geometry.elem_id[geometry.elem_group == group]}
    dropped = sorted(here - set(wanted))
    if dropped:
        shown = ', '.join(str(i) for i in dropped[:4])
        more = f' (and {len(dropped) - 4} more)' if len(dropped) > 4 else ''
        raise ValueError(
            f'element {shown}{more} would be left with no element group — every '
            'element is in exactly one. Add it to the element group it belongs '
            'in instead; it leaves this one by itself')
    if wanted:
        geometry.elem_group[np.isin(geometry.elem_id,
                                    list(wanted))] = group


#: the Element Groups table's property columns: (title, field of the property
#: set or of its material/section, kind)
_PROPERTY_FIELDS = (
    # (title, holder, field, dimension — None where there is no unit)
    ('Material', 'material', 'name', None),
    ('E', 'material', 'youngs_modulus', 'pressure'),
    ('ν', 'material', 'poissons_ratio', None),
    ('ρ', 'material', 'density', 'mass/length**3'),
    ('Thickness', '', 'thickness', 'length'),
    ('Section', 'section', 'name', None),
    ('Shape', 'section', 'shape', None),
    ('Dimensions', 'section', 'dimensions', 'length'),
    ('A', 'section', 'area', 'length**2'),
    ('Iy', 'section', 'iy', 'length**4'),
    ('Iz', 'section', 'iz', 'length**4'),
    ('J', 'section', 'j', 'length**4'),
    ('Orientation', '', 'orientation', None),
    # an element group of point elements: the mass at each one's node, and
    # nothing else (2026-10-07)
    ('Mass', '', 'mass', 'mass'),
    # a group of springs: a stiffness per global direction, typed by
    # label in the display system (2026-10-08)
    ('Stiffness', '', 'stiffness', None),
    # a group of points held: the directions, 'all' for six
    ('Ground', '', 'ground', None),
)

#: the Material list's entry that makes a group of points ground
GROUND = 'ground (fixed)'
#: the Stiffness cell's labels, in `fem.DIRECTIONS` order
STIFFNESS_LABELS = ('Kx', 'Ky', 'Kz', 'Krx', 'Kry', 'Krz')


def _ground_text(ground):
    """'all', or the directions held: 'X, Y, Z'."""
    if not ground:
        return ''
    return 'all' if len(ground) == 6 else ', '.join(ground)


def _stiffness_dimension(index):
    return 'force/length' if index < 3 else 'force*length'


def _stiffness_text(stiffness, unit_system):
    """'Kz=1e+05, Kry=2000' in the display system, free directions left
    out."""
    shown = []
    for k, (label, value) in enumerate(zip(STIFFNESS_LABELS, stiffness)):
        if value is None:
            continue
        if unit_system is not None:
            value = float(unit_system.from_si(value, _stiffness_dimension(k)))
        shown.append(f'{label}={value:.6g}')
    return ', '.join(shown)


def _parse_stiffness(text, unit_system):
    """Six stiffnesses in SI from 'Kz=1e5, Kry=2e3', each labeled, in the
    display system; a direction not named is free."""
    values = [None] * 6
    lower = [label.lower() for label in STIFFNESS_LABELS]
    for token in text.replace(';', ',').split(','):
        token = token.strip()
        if not token:
            continue
        label, eq, number = token.partition('=')
        label = label.strip().lower()
        if not eq or label not in lower:
            raise ValueError('a stiffness is given by direction: '
                             + ', '.join(f'{name}=' for name in STIFFNESS_LABELS)
                             + ' (Kz=1e5, Kry=2e3)')
        k = lower.index(label)
        value = float(number)
        if not value > 0.0:
            raise ValueError(f'{STIFFNESS_LABELS[k]} is positive, or left out '
                             'for a direction that is free')
        if unit_system is not None:
            value = float(unit_system.to_si(value, _stiffness_dimension(k)))
        values[k] = value
    if not any(values):
        raise ValueError('a group of springs needs a stiffness in one '
                         'direction at least')
    return tuple(values)


def _properties_of(geometry, row):
    return geometry.group_properties.get(int(geometry.group_id[row]))


#: the short names a shape's dimensions are written with in the table,
#: in the constructor's order (`fem.SHAPES`)
_SHAPE_LABELS = {
    'round tube': ('D', 't'), 'rod': ('D',), 'rectangle': ('b', 'h'),
    'rectangular tube': ('b', 'h', 't'),
    'I-beam': ('d', 'bf', 'tf', 'tw'), 'channel': ('d', 'bf', 'tf', 'tw'),
    'angle': ('a', 'b', 't'),
}
CUSTOM = 'custom'


def _dimensions_text(section, unit_system):
    """'D=1, t=0.065' in the display system, the shape's own labels; a
    shape with nothing typed yet shows what it wants ('D=?, t=?'); an
    angle adds where its orientation points."""
    labels = _SHAPE_LABELS[section.shape]
    if not section.dimensions:
        return ', '.join(f'{label}=?' for label in labels)
    shown = [float(unit_system.from_si(v, 'length')) if unit_system else v
             for v in section.dimensions]
    text = ', '.join(f'{label}={value:.6g}'
                     for label, value in zip(labels, shown))
    if section.shape == 'angle':
        from ..core.fem import angle_major_axis

        text += (f'; orient {angle_major_axis(*section.dimensions):.1f}° '
                 'from the long leg')
    return text


def _property_value(geometry, row, holder, field, dimension=None,
                    unit_system=None):
    props = _properties_of(geometry, row)
    if props is None:
        return ''
    if field == 'name' and holder == 'material' and props.ground:
        return GROUND
    if field == 'stiffness':
        return (_stiffness_text(props.stiffness, unit_system)
                if props.stiffness is not None else '')
    if field == 'ground':
        return _ground_text(props.ground)
    owner = getattr(props, holder) if holder else props
    if owner is None:
        return ''
    if (holder == 'material' and field != 'name' and owner.is_rigid):
        # a link has no modulus, density or ratio: blank, not 'inf'
        return ''
    if holder == 'section' and field == 'shape':
        return owner.shape or CUSTOM
    if holder == 'section' and field == 'dimensions':
        return _dimensions_text(owner, unit_system) if owner.shape else ''
    value = getattr(owner, field)
    if value is None:
        return ''
    if field == 'orientation':
        return ', '.join(f'{v:g}' for v in value)
    if dimension is not None and unit_system is not None:
        # held in SI, shown in the display system
        return float(unit_system.from_si(float(value), dimension))
    return value


def _set_property(holder, field, dimension=None, unit_system=None):
    """A setter that rewrites the element group's property set with one field
    changed — a material or a section made on first touch with the
    other fields blank, so a person can fill a row cell by cell. A
    number is typed in the display system and stored in SI."""
    from dataclasses import replace

    from ..core.fem import MATERIALS, GroupProperties, Material, Section

    def set_value(geometry, row, text):
        group = int(geometry.group_id[row])
        props = geometry.group_properties.get(group)
        text = str(text).strip()
        if field == 'mass':
            # a mass makes the element group point masses, and a mass alone;
            # cleared, the element group has nothing left to say
            value = float(text) if text else None
            if value is not None and unit_system is not None:
                value = float(unit_system.to_si(value, 'mass'))
            if value is not None:
                geometry.group_properties[group] = GroupProperties(mass=value)
            elif props is not None and props.kind == 'mass':
                geometry.group_properties.pop(group, None)
            return
        if field == 'stiffness':
            # a stiffness makes the group springs, and a stiffness alone;
            # cleared, the group has nothing left to say
            if text:
                geometry.group_properties[group] = GroupProperties(
                    stiffness=_parse_stiffness(text, unit_system))
            elif props is not None and props.kind == 'spring':
                geometry.group_properties.pop(group, None)
            return
        if holder == 'material' and field == 'name' and text == GROUND:
            geometry.group_properties[group] = GroupProperties(ground=True)
            return
        if field == 'ground':
            # directions make the group ground, and nothing else; cleared,
            # it has nothing left to say
            if text:
                geometry.group_properties[group] = GroupProperties(
                    ground=True if text.lower() == 'all' else text)
            elif props is not None and props.kind == 'ground':
                geometry.group_properties.pop(group, None)
            return
        if props is not None and props.kind == 'mass':
            raise ValueError('an element group of point masses takes a mass alone — '
                             'clear its Mass to give it a material')
        if props is not None and props.kind == 'spring':
            raise ValueError('a group of springs takes a stiffness alone — '
                             'clear its Stiffness to give it a material')
        if (props is not None and props.kind == 'ground'
                and not (holder == 'material' and field == 'name'
                         and text in MATERIALS)):
            raise ValueError('a ground group holds its points fixed and takes '
                             'nothing else — pick a material to make it '
                             'something else')
        # a group of solids takes a material alone: a section, a thickness
        # or an orientation typed on it is refused here, and clearing one
        # left from before is not (Brandon, 2026-10-20: his bricks carried
        # an empty round tube, and the solve asked for its dimensions)
        if text and (holder == 'section' or field in ('thickness', 'orientation')):
            from ..core.geometry import SOLID_FAMILIES
            from ..core.mesh import group_families

            families = group_families(geometry, group)
            if families and families <= SOLID_FAMILIES:
                raise ValueError('a group of solid elements takes a material alone: '
                                 'no section, thickness or orientation')
        if holder == 'section' and field in ('shape', 'dimensions'):
            geometry.group_properties[group] = _without_empty_section(
                _set_section_shape(props, field, text, unit_system))
            return
        if (holder == 'section' and props is not None
                and props.section is not None and props.section.shape
                and field in ('area', 'iy', 'iz', 'j')):
            raise ValueError(f'computed from the {props.section.shape}’s '
                             'dimensions — choose Custom in Shape to type '
                             'it')
        if (holder == 'material' and field != 'name' and props is not None
                and props.material.is_rigid):
            raise ValueError('a rigid (massless) link has no '
                             'modulus, density or ratio — pick a material '
                             'to give it one')
        if holder == 'material' and field == 'name' and text in MATERIALS:
            # a library name fills the row; any other name is a name. A
            # ground group picked a material is a group of that material
            geometry.group_properties[group] = replace(
                props or GroupProperties(MATERIALS[text]),
                material=MATERIALS[text], ground=False)
            return
        if field == 'name':
            value = text
        elif field == 'orientation':
            parts = [float(v) for v in text.replace(',', ' ').split()]
            if text and len(parts) != 3:
                raise ValueError('an orientation is three numbers')
            value = tuple(parts) if text else None
        else:
            value = float(text) if text else None
            if (value is not None and dimension is not None
                    and unit_system is not None):
                value = float(unit_system.to_si(value, dimension))
        if props is None:
            props = GroupProperties(Material('', 0.0, 0.0, 0.3))
        if holder == 'material':
            material = props.material or Material('', 0.0, 0.0, 0.3)
            props = replace(props, material=replace(material,
                                                    **{field: value}))
        elif holder == 'section':
            section = props.section or Section('', 0.0, 0.0, 0.0, 0.0)
            if field == 'name' and not text and props.section is None:
                return
            props = _without_empty_section(replace(props, section=replace(section,
                                                                          **{field: value})))
        else:
            props = replace(props, **{field: value})
        geometry.group_properties[group] = props

    return set_value


def _without_empty_section(props):
    """The properties with a section that says nothing taken off: no
    name, no shape and no numbers. Clearing a section's cells clears the
    section, so a group given one by mistake becomes what it was
    (Brandon, 2026-10-20: clearing them left an empty one behind, and the
    group of bricks stayed a group of beams)."""
    from dataclasses import replace

    section = props.section if props is not None else None
    if (section is not None and not section.name and not section.shape
            and not section.dimensions
            and not any((section.area, section.iy, section.iz, section.j))):
        return replace(props, section=None)
    return props


def _set_section_shape(props, field, text, unit_system):
    """The element group's properties with its section's shape or dimensions
    changed. A shape picked keeps the dimensions it can (the same shape,
    or as many as the new one takes are there — not guessed); Custom
    keeps the four numbers and lets them be typed. Dimensions are typed
    in the display system as numbers in order or as the labels shown
    ('D=1, t=0.065'), anything after a ';' ignored."""
    from dataclasses import replace

    from ..core.fem import SHAPES, GroupProperties, Material, Section, with_article

    if props is None:
        props = GroupProperties(Material('', 0.0, 0.0, 0.3))
    section = props.section or Section('', 0.0, 0.0, 0.0, 0.0)
    if field == 'shape':
        shape = text.strip()
        if shape in ('', CUSTOM):
            return replace(props, section=replace(section, shape='',
                                                  dimensions=()))
        if shape not in SHAPES:
            raise ValueError(f'{shape!r} is not a section shape: '
                             + ', '.join([*SHAPES, CUSTOM]))
        if shape == section.shape:
            return props
        # a new shape starts empty: its dimensions mean different things
        return replace(props, section=Section(section.name, 0.0, 0.0, 0.0,
                                              0.0, shape, ()))
    if not section.shape:
        raise ValueError('choose a shape first — the Shape column — or '
                         'type A, Iy, Iz and J for a custom section')
    labels = _SHAPE_LABELS[section.shape]
    values = []
    for token in text.split(';')[0].replace(',', ' ').split():
        if '=' in token:
            label, _eq, token = token.partition('=')
            if label.strip() not in labels:
                raise ValueError(f'{with_article(section.shape)} is given '
                                 'as ' + ', '.join(labels))
        values.append(float(token))
    if len(values) != len(labels):
        names = SHAPES[section.shape][1]
        raise ValueError(f'{with_article(section.shape)} takes '
                         f'{len(labels)} dimensions: '
                         + ', '.join(f'{label} ({name})' for label, name
                                     in zip(labels, names)))
    if unit_system is not None:
        values = [float(unit_system.to_si(v, 'length')) for v in values]
    return replace(props, section=Section.of_shape(section.name,
                                                   section.shape, values))


def _property_journal(geometry, row, _text):
    """The element group's whole property set, as the line that rebuilds it."""
    group = int(geometry.group_id[row])
    props = geometry.group_properties.get(group)
    if props is None:
        return f'.group_properties.pop({group}, None)'
    if props.kind == 'mass':
        return f'.group_properties[{group}] = GroupProperties(mass={props.mass!r})'
    if props.kind == 'ground':
        held = (True if len(props.ground) == 6 else tuple(props.ground))
        return f'.group_properties[{group}] = GroupProperties(ground={held!r})'
    if props.kind == 'spring':
        return (f'.group_properties[{group}] = '
                f'GroupProperties(stiffness={tuple(props.stiffness)!r})')
    m = props.material
    if m.is_rigid:
        # the rigid link by name: its modulus is infinite, and 'inf' is
        # not a number a script can replay
        return f'.group_properties[{group}] = GroupProperties(RIGID)'
    material = (f'Material({m.name!r}, {m.youngs_modulus!r}, {m.density!r}, '
                f'{m.poissons_ratio!r}'
                + (f', {m.modulus_of_rigidity!r}'
                   if m.modulus_of_rigidity is not None else '') + ')')
    parts = [material]
    if props.thickness is not None:
        parts.append(f'thickness={props.thickness!r}')
    if props.section is not None:
        s = props.section
        if s.shape and s.dimensions:
            # a shaped section replays from its dimensions
            parts.append(f'section=Section.of_shape({s.name!r}, '
                         f'{s.shape!r}, {list(s.dimensions)!r})')
        else:
            parts.append(f'section=Section({s.name!r}, {s.area!r}, '
                         f'{s.iy!r}, {s.iz!r}, {s.j!r}'
                         + (f', {s.shape!r}' if s.shape else '') + ')')
    if props.orientation is not None:
        parts.append(f'orientation={tuple(props.orientation)!r}')
    return (f'.group_properties[{group}] = GroupProperties('
            + ', '.join(parts) + ')')


def _property_columns(unit_system=None):
    """The property columns, in the display system (Brandon,
    2026-09-26: the table should show and expect the current display
    units). Held in SI on the geometry, as the model is inside, and
    journaled in SI, as every table edit is; the header names the unit.
    Without a system, SI."""
    from ..core.fem import MATERIALS

    columns = []
    from ..core.fem import SHAPES

    for title, holder, field, dimension in _PROPERTY_FIELDS:
        kwargs = ({'alignment': LEFT}
                  if field in ('name', 'orientation', 'shape', 'dimensions',
                               'stiffness', 'ground')
                  else {'format': lambda v: v if v == '' else f'{v:.6g}'})
        if holder == 'material' and field == 'name':
            # the library as a shortlist, not a rule: pick one and the
            # row fills, or type any name and fill the row yourself; ground
            # last, beside rigid, as the other thing a group can be that
            # is no material
            kwargs.update(choices=[*MATERIALS, GROUND], choices_editable=True)
        if field == 'shape':
            kwargs.update(choices=[*SHAPES, CUSTOM])
        if dimension is not None:
            unit = (unit_system.label_text(dimension)
                    if unit_system is not None
                    else _SI_LABELS[dimension])
            title = f'{title} [{unit}]'
        if field == 'stiffness':
            # two units in one cell: along an axis, and about it
            along, about = (
                (unit_system.label_text('force/length'),
                 unit_system.label_text('force*length'))
                if unit_system is not None else ('N/m', 'N·m'))
            title = f'{title} [{along}, {about}/rad]'
        columns.append(Column(
            title, (lambda g, r, h=holder, f=field, d=dimension:
                    _property_value(g, r, h, f, d, unit_system)),
            set=_set_property(holder, field, dimension, unit_system),
            journal=_property_journal, **kwargs))
    return columns


#: the headers with no display system to ask
_SI_LABELS = {'pressure': 'Pa', 'mass/length**3': 'kg/m³', 'length': 'm',
              'length**2': 'm²', 'length**4': 'm⁴', 'mass': 'kg'}


def group_label(geometry: Geometry, row: int) -> str:
    """An element group's label in a list: its name, or
    `'element group N'` when it has none.

    Exodus files often have unnamed element groups, and "element group
    3" is what a person calls one. A made-up name would claim something
    the file does not say.
    """
    name = geometry.group_name[row].strip()
    return name or f'element group {int(geometry.group_id[row])}'


def _group_labels(geometry):
    return [group_label(geometry, row) for row in range(len(geometry.group_id))]


def _set_element_group(geometry, row, text):
    """Move an element into an element group, named or numbered.

    The element group has to exist: `add_element` may declare one because it is
    building the element in the first place, but retyping a cell into a
    number nobody has declared is a typo far more often than it is a new
    element group, and the Element Groups table is where one is made.
    """
    wanted = text.strip()
    for i in range(len(geometry.group_id)):
        if wanted in (group_label(geometry, i), geometry.group_name[i].strip(),
                      str(int(geometry.group_id[i]))):
            geometry.elem_group[row] = int(geometry.group_id[i])
            return
    raise ValueError(f'no element group {wanted!r}; add it in the Element Groups table')


def element_table_model(geometry: Geometry,
                        unit_system: UnitSystem | None = None,
                        parent: QObject | None = None) -> TableModel:
    from ..core.geometry import ELEMENT_TYPES

    names = {code: name for code, (name, _, _) in ELEMENT_TYPES.items()}

    def set_type(geometry: Any, row: int, text: str) -> None:
        wanted = text.strip().lower()
        for code, name in names.items():
            if name == wanted:
                geometry.elem_type[row] = code
                return
        raise ValueError(f'unknown element type {text!r}')

    def group_of_row(geometry: Geometry, row: int) -> str:
        where = np.flatnonzero(geometry.group_id == int(geometry.elem_group[row]))
        return group_label(geometry, int(where[0])) if len(where) else ''

    columns = [
        Column('Element', lambda g, r: int(g.elem_id[r]),
               set=_set_int_field('elem_id', minimum=1),
               journal=_int_field_journal('elem_id')),
        Column('Type', lambda g, r: names.get(int(g.elem_type[r]), 'unknown'),
               set=set_type,
               journal=lambda g, r, _t:
               f'.elem_type[{r}] = {int(g.elem_type[r])}',
               alignment=LEFT),
        _color_column('Color', 'elem_color'),
        # which part of the structure this element is, and the only way to
        # move one between element groups — an element group holds nothing itself
        Column('Element group', group_of_row, set=_set_element_group,
               journal=lambda g, r, _t:
               f'.elem_group[{r}] = {int(g.elem_group[r])}',
               alignment=LEFT,
               row_choices=lambda g, _r: _group_labels(g)),
        Column('Nodes', lambda g, r: _connectivity_text(g.elem_conn[r]),
               set=_set_connectivity('elem_conn'),
               journal=_connectivity_journal('elem_conn'),
               alignment=LEFT),
    ]
    return TableModel(geometry, columns,
                          lambda g: len(g.elem_conn), parent)


def element_group_table_model(geometry: Geometry,
                      unit_system: UnitSystem | None = None,
                      parent: QObject | None = None) -> TableModel:
    """The element groups, which divide the mesh into parts.

    Every column is editable: the group's number, its name, its
    elements, and the properties a model built from the geometry needs
    (a material, and a thickness, section or mass by element type, shown
    in the display units and blank until set). The elements are listed as runs
    (`1-36 156-245`), and naming an element here moves it into this
    group. A node can be in no element at all, but an element is always
    in exactly one element group, so an edit can add and move elements,
    and one that would leave an element in no group is refused with the
    reason (see `_set_group_elements`). The Element group column of the
    elements table moves elements one at a time; this one moves many at
    once.
    """
    def set_name(geometry: Any, row: int, text: str) -> None:
        geometry.group_name[row] = text

    columns = [
        Column('Element group', lambda g, r: int(g.group_id[r]),
               set=_renumber('renumber_group'),
               journal=_renumber_journal('renumber_group')),
        Column('Name', lambda g, r: g.group_name[r], set=set_name,
               journal=lambda g, r, t: f'.group_name[{r}] = {t!r}',
               alignment=LEFT),
        # The elements themselves, as runs, and editable: naming an
        # element here claims it for this element group. Editing the *count*
        # would have been meaningless — a number is not a thing you
        # can change — and the count is still readable at a glance
        # from the runs.
        Column('Elements', _group_elements_text, set=_set_group_elements,
               # membership moves elements between element groups; the honest
               # replay is the post-state of the whole assignment array
               journal=lambda g, _r, _t: '.elem_group = np.array('
               f'{[int(b) for b in g.elem_group]!r})',
               alignment=LEFT),
        # What the element group is made of, for a model built from the
        # geometry (Brandon, 2026-09-25): a material for any element group, a
        # thickness for an element group of plates, a section for an element group of
        # beams, a mass for an element group of point masses. SI throughout, as the model is inside; blank until
        # set. Every cell's journal line restates the whole property
        # set, so a replay lands on the same object whichever cell was
        # edited last.
        *_property_columns(unit_system),
    ]
    # the other way up (Brandon, 2026-10-20): a group's properties read
    # as a column of entries, and several groups side by side are
    # edited at once by selecting across them
    return TransposedTableModel(geometry, columns, lambda g: len(g.group_id), parent,
                                item_label=group_label)


ENTITY_TABLES = {
    'nodes': node_table_model,
    'coordinate_systems': coordinate_system_table_model,
    'elements': element_table_model,
    'groups': element_group_table_model,
}


class ComplianceRows:
    """The comparison of a specification with a measurement, one row
    per channel.

    `rows` is a list of `(channel, result)`, computed once when the
    selection changes rather than once per cell. `unit_label` is the
    level's unit for the column heading. `plotted` is the index of the
    row the plot is drawing, so the table can mark the channel on
    screen.
    """

    def __init__(self, rows: Sequence[tuple[str, Any]], unit_label: str = '',
                 plotted: int | None = None) -> None:
        self.rows: list[tuple[str, Any]] = list(rows)
        self.unit_label: str = unit_label
        self.plotted: int | None = plotted

    def __len__(self) -> int:
        return len(self.rows)

    def row_of(self, label: str) -> int | None:
        """The row of the channel named `label`, or None."""
        for row, (name, _result) in enumerate(self.rows):
            if name == label:
                return row
        return None


def _pair_of(data, index):
    """The (response, reference) a record is between — the data's own
    reading of it, so the labels agree with the drop-down's."""
    return data.record_pair(index)


def specification_model(rows: Any,
                        parent: QObject | None = None) -> TableModel:
    """A specification's channels, one row each: the channel's name
    and its RMS level.

    The plot draws one channel at a time, because six targets and their
    limits drawn together cannot be read, so this table lists every
    channel, and picking a row plots that one.

    The level is the specification's own, integrated from its own
    breakpoints over its own band, so it does not change with the
    measurement compared against it.
    """
    level = f' [{rows.unit_label}]' if rows.unit_label else ''
    return TableModel(rows, [
        Column('Channel', lambda h, r: h.rows[r][0], alignment=LEFT),
        Column(f'Specification RMS{level}',
               lambda h, r: h.rows[r][1].get('specification_rms'),
               format=_number),
    ], len, parent)


def _number(value):
    # words pass through: a cell that could not be scored says why
    if isinstance(value, str):
        return value
    return '' if value is None else f'{value:.4g}'


class ReplicationRows:
    """Waveform error as a grid: one control channel per row, one
    playing of the waveform (an event) per column.

    A transient record holds many attempts at one target, so its errors
    have two dimensions. The plot draws only what is picked and the bar
    chart reduces them to one value, so this grid is the one place that
    shows both. In it, a channel that is poor in every event and an
    event that was poor on every channel are easy to tell apart.

    `values` maps `(channel, event)` to the error, or to a sentence
    saying why that pair has none. `plotted` is the set
    of `(channel, event)` pairs on screen, because picking a cell picks
    both at once.
    """

    def __init__(self, channels: Sequence[str], events: Sequence[int],
                 values: Mapping[tuple[str, int], float | str],
                 plotted: Sequence[tuple[str, int]] | None = None) -> None:
        self.channels: list[str] = list(channels)
        self.events: list[int] = list(events)
        self.values: dict[tuple[str, int], float | str] = dict(values)
        self.plotted: set[tuple[str, int]] = set(plotted or ())

    def __len__(self) -> int:
        return len(self.channels)

    def value(self, row: int, column: int) -> float | str | None:
        """The value at a row and an event's column (counted from zero,
        without the channel column), the words for why it was not
        scored, or None where there is no value for that channel and
        event."""
        if not 0 <= row < len(self.channels):
            return None
        if not 0 <= column < len(self.events):
            return None
        return self.values.get((self.channels[row], self.events[column]))

    def pairs_at(self, row: int,
                 column: int) -> list[tuple[str, int]]:
        """The `(channel, event)` pairs one selected cell picks.

        Column zero holds the channel's name, so picking it picks that
        channel in every event, the whole row.
        """
        if not 0 <= row < len(self.channels):
            return []
        channel = self.channels[row]
        if column <= 0:
            return [(channel, event) for event in self.events]
        if column - 1 >= len(self.events):
            return []
        return [(channel, self.events[column - 1])]


def replication_model(rows: Any, parent: QObject | None = None,
                      units: str = '%') -> TableModel:
    """A table of a `ReplicationRows`: channels down, events across.

    Two comparisons have this shape and share it: the waveform error
    against a target time history, in percent, and the RMS deviation in
    dB of a shock spectrum from the one it had to meet. Only `units`, in
    each column heading, differs.

    A cell with no value is blank, not zero, because a zero would read
    as a channel that matched its target exactly.
    """
    columns = [Column('Channel', lambda h, r: h.channels[r], alignment=LEFT)]
    for index, event in enumerate(rows.events):
        columns.append(Column(
            f'Event {event + 1} [{units}]',
            lambda h, r, c=index: h.value(r, c),
            format=_number))
    return TableModel(rows, columns, len, parent)


def _pair_text(pair):
    """How a row names its channel — the plot's own reading of it, so
    the table and the drop-down beside it agree."""
    from ..plot import pair_label

    return pair_label(pair) if isinstance(pair, tuple) else str(pair)


class ComplianceGrid:
    """How far each control channel is from its specification, one
    row per channel.

    It does for a PSD comparison what `ReplicationRows` does for a
    transient, with one set of numbers per channel instead of a grid,
    because a PSD is already an average over the repeats.

    `rows` is a list of `(DOF pair, RMS error in dB, percent of the
    band outside the abort limits)`. `plotted` is the set of DOF pairs
    on screen.
    """

    def __init__(self, rows: Sequence[tuple[str, Any]],
                 plotted: Sequence[str] | None = None) -> None:
        self.rows: list[tuple[str, Any]] = list(rows)
        self.plotted: set[str] = set(plotted or ())

    def __len__(self) -> int:
        return len(self.rows)

    def pair_at(self, row: int) -> str | None:
        """The DOF pair a row names, or None."""
        if 0 <= row < len(self.rows):
            return self.rows[row][0]
        return None


def compliance_grid_model(rows: Any,
                          parent: QObject | None = None) -> TableModel:
    """A table of a `ComplianceGrid`: each channel, its RMS error in
    dB, and the percent of its band outside the abort limits.

    Both values are shown because they often disagree: a channel can
    have exactly the right level and still be outside the limits across
    half its band, and one that is 2 dB low everywhere may never cross
    an abort limit.
    """
    return TableModel(rows, [
        Column('Channel', lambda h, r: _pair_text(h.rows[r][0]),
               alignment=LEFT),
        Column('RMS error [dB]', lambda h, r: h.rows[r][1], format=_number),
        Column('Outside abort [%]', lambda h, r: h.rows[r][2],
               format=_number),
    ], len, parent)


def sweep_rate_rows(tone: str, checks: Sequence[Any],
                    swept: Sequence[tuple[str, float, float, Any]] = ()
                    ) -> list[dict[str, Any]]:
    """The sweep-rate table's rows: one per pass of the tone through a
    mode (`core.sweep_rate.SweepRateCheck`), then one per Q read off a
    tracked level at a mode, `swept` holding (DOF, frequency, Q,
    `SweptQ`).

    The two kinds share the columns: a level's row names the level's
    DOF where a pass names its direction, gives the rate at the level's
    peak and the rule's rate for the Q the reading implies, and leaves
    the time in the band empty, since a level is a reading of the sweep
    rather than a pass of it."""
    rows = []
    for check in checks:
        rows.append({
            'tone': tone, 'mode': check.frequency, 'q': check.q,
            'pass': check.direction + ('' if check.crossed else ', partial'),
            'rate': f'{abs(check.rate):.3g} {check.unit}',
            'limit': f'{check.limit:.3g} {check.unit}',
            'ratio': check.ratio, 'seconds': check.seconds,
            'cycles': check.cycles,
            'peak': 100.0 * check.peak_share, 'q_read': check.q_read,
            'verdict': check.verdict})
    for dof, frequency, q, reading in swept:
        if reading.q_true is None:
            verdict = 'too fast to read any Q'
        elif reading.flagged:
            verdict = f'reads low; Q about {reading.q_true:.3g}'
        else:
            verdict = 'slow enough'
        rows.append({
            'tone': tone, 'mode': frequency, 'q': q,
            'pass': f'level at {dof}',
            'rate': f'{abs(reading.rate):.3g} {reading.unit}',
            'limit': f'{reading.limit:.3g} {reading.unit}',
            'ratio': reading.ratio, 'seconds': None, 'cycles': None,
            'peak': None, 'q_read': reading.q_read, 'verdict': verdict})
    return rows


def sweep_rate_model(rows: Sequence[dict[str, Any]],
                     parent: QObject | None = None) -> TableModel:
    """A table of `sweep_rate_rows`: each pass of a sine tone through a
    mode, with the time and the cycles it spends in the mode's
    half-power band, its rate against the rule's, what a mode of that Q
    reads at that rate, and the verdict; then each Q read off a tracked
    level, judged against its sweep. Read-only: the modes are set in the
    panel beside the plot."""
    def field(key, places=None):
        def text(value):
            if value is None:
                return ''
            if isinstance(value, float):
                if not np.isfinite(value):
                    return ''
                return f'{value:.{places}f}' if places is not None \
                    else f'{value:.4g}'
            return str(value)
        return Column(key, lambda h, r: h[r][key], format=text)

    columns = [
        ('Tone', 'tone', None, LEFT), ('Mode [Hz]', 'mode', None, None),
        ('Q', 'q', None, None), ('Pass', 'pass', None, LEFT),
        ('Rate', 'rate', None, None), ("Rule's rate", 'limit', None, None),
        ('× Rule', 'ratio', 2, None), ('In band [s]', 'seconds', None, None),
        ('Cycles', 'cycles', 1, None), ('Peak read [%]', 'peak', 1, None),
        ('Q read', 'q_read', 1, None), ('Verdict', 'verdict', None, LEFT)]
    built = []
    for title, key, places, alignment in columns:
        column = field(key, places)
        column.title = title
        if alignment is not None:
            column.alignment = alignment
        built.append(column)
    return TableModel(list(rows), built, len, parent)
