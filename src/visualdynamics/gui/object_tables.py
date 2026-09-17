"""Table models for visualdynamics objects.

Each is a list of columns over an existing object — reading straight from it
and, where editing makes sense, writing straight back. Results that a fit
produced (frequency, damping) are read-only; the free-text description
beside them is not.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ..core.channel_table import title_of
from ..core.shapes import scale_ratios
from ..core.unit_choices import (
    ALL_ORDINATE_UNITS,
    shown_dimension,
    stored_dimension,
)
from ..units import UNKNOWN, dimension_of, parse_dimension, si_transform
from ..viz.geometry import COLOR_NAMES, color_name, color_rgb
from .icons import color_swatch
from .tables import Column, TableModel

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
    """Modes with their frequency, damping and a description to fill in."""
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

    The referenced sets are read live by name — frequencies and
    damping restate themselves as the sets change — while the MAC is
    the object's own stored value (it came from the comparison as
    displayed, possibly projected). A set gone from the project, or a
    mode index past its end, shows dashes rather than guessing.
    """
    a_name, b_name = matched.first, matched.second
    a, b = objects.get(a_name), objects.get(b_name)

    def parameters(shape_set: Any,
                   mode: int) -> tuple[float | None, float | None]:
        if shape_set is None or not 0 <= mode < shape_set.num_shapes:
            return None, None
        return (float(shape_set.frequency[mode]),
                float(shape_set.damping[mode]) * 100.0)

    # the scale each pair was normalized by, which the overlay
    # animation cannot show: it draws both shapes to their own peak, so
    # a set thirty times the other looks identical to one that agrees
    ratios = (scale_ratios(a, b, matched.pairs)
              if a is not None and b is not None
              else [None] * len(matched.pairs))
    rows = []
    for index, ((row, column), mac) in enumerate(
            zip(matched.pairs, matched.macs)):
        fa, da = parameters(a, row)
        fb, db = parameters(b, column)
        delta = (f'{(fb - fa) / fa * 100.0:+.2f}'
                 if fa and fb is not None else '—')
        rows.append((row + 1, fa, da, column + 1, fb, db, delta,
                     float(mac), ratios[index]))
    def number(decimals: int) -> Callable[[Any], str]:
        return lambda v: '—' if v is None else f'{v:.{decimals}f}'

    columns = [
        Column(f'{a_name} Mode', lambda s, r: s[r][0]),
        Column('Frequency [Hz]', lambda s, r: s[r][1],
               format=number(4)),
        Column('Damping [%]', lambda s, r: s[r][2], format=number(3)),
        Column(f'{b_name} Mode', lambda s, r: s[r][3]),
        Column('Frequency [Hz]', lambda s, r: s[r][4],
               format=number(4)),
        Column('Damping [%]', lambda s, r: s[r][5], format=number(3)),
        Column('Δf [%]', lambda s, r: s[r][6]),
        Column('MAC', lambda s, r: s[r][7], format=number(3)),
        Column(f'{b_name}/{a_name}', lambda s, r: s[r][8],
               format=number(2)),
    ]
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
                        rows: Sequence[int] | None = None) -> TableModel:
    """A channel table: every column editable, every column typed.

    The columns come from the table's own `COLUMNS` spec rather than
    being listed again here, so a schema change reaches the interface
    without a second edit — a choice column gets its drop-down, a date
    gets a calendar, a flag gets a check box, and the unit column is
    narrowed by what the channel says it measures.

    `rows` shows only those channels — picking cells in the tree's grid
    is picking rows here too, so the table beside the model shows what
    the selection says and nothing else.
    """
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
            columns.append(Column(
                title, _shown_getter(name, index),
                set=_column_setter(name, index),
                journal=_column_journal(name, index),
                choices=shown, affects_row=name == 'channel_type',
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
            date=spec.kind == 'date', alignment=LEFT))
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

    The last row is the pending mode: its frequency mirrors the cursor on
    the CMIF, its damping is the half-power estimate until the user types
    one, and Confirm Mode turns it into a fitted row. A fitted mode's
    damping stays editable — the synthesis restates itself — and its
    description is free text.
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
    """One row per thing that needs a unit, whatever kind of object it is.

    A table rather than a dialog, so the plot or the model stays on screen
    beside it and units copy, paste and clear like any other column.
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
    """Two tables for an FRF: response channels and reference channels.

    An FRF matrix is N x M records but only N + M physical channels, and a
    channel's unit is the channel's property — so each side is declared
    once, and every record converts as soon as both of its channels are
    named. The sides stay separate even where the DOF labels collide: a
    drive point's accelerometer and its force gauge share a label and are
    two different channels.
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
    """Sorted ids as runs: `[1,2,3,7,9,10]` -> `'1-3 7 9-10'`.

    A block's elements are not a handful like an element's nodes — the
    demonstration drone puts 478 in one — and they come in runs,
    because a mesh is built a part at a time. Written out one by one
    the drone's 22 blocks need 21 731 characters; as runs they need
    595, which is the difference between a cell you can read and one
    you cannot.
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
    """The inverse: `'1-3, 7 9-10'` -> `[1, 2, 3, 7, 9, 10]`.

    Commas or spaces, and a run either way round — `10-7` is the same
    six elements as `7-10`, because someone typing a range backwards
    means the range.
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


def _block_elements_text(geometry: Geometry, row: int) -> str:
    block = int(geometry.block_id[row])
    return id_runs_text(geometry.elem_id[geometry.elem_block == block])


def _set_block_elements(geometry: Any, row: int, text: str) -> None:
    """Claim these elements for this block.

    Editing the list *moves elements in*: an element named here leaves
    whatever block it was in, because it can only be in one. That also
    means the list cannot be used to take an element *out* — every
    element is in exactly one block, always (the constructor refuses
    strays, and deleting a block rehouses its elements rather than
    orphaning them), so a removal with no destination is not a state
    the geometry can hold. Removing is done by adding: name the
    element in the block it should go to, and it leaves this one by
    itself. The refusal says so rather than guessing a destination.
    """
    wanted = parse_id_runs(text)
    known = {int(i) for i in geometry.elem_id}
    missing = sorted(set(wanted) - known)
    if missing:
        raise ValueError(f'unknown elements {missing}')
    block = int(geometry.block_id[row])
    here = {int(i) for i in geometry.elem_id[geometry.elem_block == block]}
    dropped = sorted(here - set(wanted))
    if dropped:
        shown = ', '.join(str(i) for i in dropped[:4])
        more = f' (and {len(dropped) - 4} more)' if len(dropped) > 4 else ''
        raise ValueError(
            f'element {shown}{more} would be left with no block — every '
            'element is in exactly one. Add it to the block it belongs '
            'in instead; it leaves this one by itself')
    if wanted:
        geometry.elem_block[np.isin(geometry.elem_id,
                                    list(wanted))] = block


def traceline_table_model(geometry: Geometry,
                          unit_system: UnitSystem | None = None,
                          parent: QObject | None = None) -> TableModel:
    def set_description(geometry: Any, row: int, text: str) -> None:
        geometry.traceline_desc[row] = text

    columns = [
        Column('Traceline', lambda g, r: int(g.traceline_id[r]),
               set=_set_int_field('traceline_id', minimum=1),
               journal=_int_field_journal('traceline_id')),
        Column('Description', lambda g, r: g.traceline_desc[r],
               set=set_description,
               journal=lambda g, r, t: f'.traceline_desc[{r}] = {t!r}',
               alignment=LEFT),
        _color_column('Color', 'traceline_color'),
        Column('Nodes', lambda g, r: _connectivity_text(g.traceline_conn[r]),
               set=_set_connectivity('traceline_conn'),
               journal=_connectivity_journal('traceline_conn'),
               alignment=LEFT),
    ]
    return TableModel(geometry, columns,
                          lambda g: len(g.traceline_conn), parent)


def block_label(geometry: Geometry, row: int) -> str:
    """How a block reads in a list: its name, or its id when unnamed.

    Exodus files often carry unnamed blocks, and 'block 3' is what a
    person calls that one — inventing a name for it would be putting
    words in the file's mouth.
    """
    name = geometry.block_name[row].strip()
    return name or f'block {int(geometry.block_id[row])}'


def _block_labels(geometry):
    return [block_label(geometry, row) for row in range(len(geometry.block_id))]


def _set_element_block(geometry, row, text):
    """Move an element into a block, named or numbered.

    The block has to exist: `add_element` may declare one because it is
    building the element in the first place, but retyping a cell into a
    number nobody has declared is a typo far more often than it is a new
    block, and the Blocks table is where one is made.
    """
    wanted = text.strip()
    for i in range(len(geometry.block_id)):
        if wanted in (block_label(geometry, i), geometry.block_name[i].strip(),
                      str(int(geometry.block_id[i]))):
            geometry.elem_block[row] = int(geometry.block_id[i])
            return
    raise ValueError(f'no block {wanted!r}; add it in the Blocks table')


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

    def block_of_row(geometry: Geometry, row: int) -> str:
        where = np.flatnonzero(geometry.block_id == int(geometry.elem_block[row]))
        return block_label(geometry, int(where[0])) if len(where) else ''

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
        # move one between blocks — a block holds nothing itself
        Column('Block', block_of_row, set=_set_element_block,
               journal=lambda g, r, _t:
               f'.elem_block[{r}] = {int(g.elem_block[r])}',
               alignment=LEFT,
               row_choices=lambda g, _r: _block_labels(g)),
        Column('Nodes', lambda g, r: _connectivity_text(g.elem_conn[r]),
               set=_set_connectivity('elem_conn'),
               journal=_connectivity_journal('elem_conn'),
               alignment=LEFT),
    ]
    return TableModel(geometry, columns,
                          lambda g: len(g.elem_conn), parent)


def block_table_model(geometry: Geometry,
                      unit_system: UnitSystem | None = None,
                      parent: QObject | None = None) -> TableModel:
    """The element blocks: what the mesh is divided into.

    Three columns, all editable. The elements are listed as runs
    (`1-36 156-245`) and naming one here claims it for this block —
    the same gesture as typing a node into a traceline, with one
    difference the model forces: a node can be in no traceline at all,
    while an element is always in exactly one block. So the list adds
    and moves, and a removal that would orphan an element is refused
    with the reason (see `_set_block_elements`). The Elements column
    in the *elements* table moves them one at a time; this one moves
    them by the hundred.
    """
    def set_name(geometry: Any, row: int, text: str) -> None:
        geometry.block_name[row] = text

    columns = [
        Column('Block', lambda g, r: int(g.block_id[r]),
               set=_renumber('renumber_block'),
               journal=_renumber_journal('renumber_block')),
        Column('Name', lambda g, r: g.block_name[r], set=set_name,
               journal=lambda g, r, t: f'.block_name[{r}] = {t!r}',
               alignment=LEFT),
        # The elements themselves, as runs, and editable: naming an
        # element here claims it for this block. Editing the *count*
        # would have been meaningless — a number is not a thing you
        # can change — and the count is still readable at a glance
        # from the runs.
        Column('Elements', _block_elements_text, set=_set_block_elements,
               # membership moves elements between blocks; the honest
               # replay is the post-state of the whole assignment array
               journal=lambda g, _r, _t: '.elem_block = np.array('
               f'{[int(b) for b in g.elem_block]!r})',
               alignment=LEFT),
    ]
    return TableModel(geometry, columns, lambda g: len(g.block_id), parent)


ENTITY_TABLES = {
    'nodes': node_table_model,
    'coordinate_systems': coordinate_system_table_model,
    'tracelines': traceline_table_model,
    'elements': element_table_model,
    'blocks': block_table_model,
}


class ComplianceRows:
    """The comparison of a specification with a measurement, per channel.

    A plain holder so the table model has something to read: the rows
    are computed once, when the selection settles, rather than per cell.

    `plotted` is the row the plot is currently drawing, so the table can
    say which of its channels is the one on screen.
    """

    def __init__(self, rows: Sequence[tuple[str, Any]], unit_label: str = '',
                 plotted: int | None = None) -> None:
        self.rows: list[tuple[str, Any]] = list(rows)
        self.unit_label: str = unit_label
        self.plotted: int | None = plotted

    def __len__(self) -> int:
        return len(self.rows)

    def row_of(self, label: str) -> int | None:
        """Which row is this channel, or None."""
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
    """A row per channel a specification carries: its name and its level.

    What a specification says, in one number per channel. The plot draws
    one channel at a time — six targets and their limits stacked
    together are unreadable — so the table is where every channel is
    still accounted for, and picking a row is how you get to one.

    The level is the specification's own, integrated from its own
    breakpoints over its own band, so it does not move with whatever
    happens to be measured against it.
    """
    level = f' [{rows.unit_label}]' if rows.unit_label else ''
    return TableModel(rows, [
        Column('Channel', lambda h, r: h.rows[r][0], alignment=LEFT),
        Column(f'Specification RMS{level}',
               lambda h, r: h.rows[r][1].get('specification_rms'),
               format=_number),
    ], len, parent)


def _number(value):
    return '' if value is None else f'{value:.4g}'


class ReplicationRows:
    """Waveform error as a grid: a control channel per row, a playing of
    the waveform per column.

    A transient record is many attempts at one target, so the numbers
    are two-dimensional and the table is the only place that shows them
    that way — the plot draws what is picked and the bar chart reduces
    to one reading. Laid out this way a channel that is bad everywhere
    and a repeat that was bad for everything look different at a
    glance, which is the comparison a single column cannot make.

    `plotted` is the set of (channel, event) pairs on screen, because
    picking a cell picks both at once.
    """

    def __init__(self, channels: Sequence[str], events: Sequence[int],
                 values: Mapping[tuple[str, int], float],
                 plotted: Sequence[tuple[str, int]] | None = None) -> None:
        self.channels: list[str] = list(channels)
        self.events: list[int] = list(events)
        self.values: dict[tuple[str, int], float] = dict(values)
        self.plotted: set[tuple[str, int]] = set(plotted or ())

    def __len__(self) -> int:
        return len(self.channels)

    def value(self, row: int, column: int) -> float | None:
        """The cell at (row, column-of-events), or None where the
        channel cannot be scored against its target."""
        if not 0 <= row < len(self.channels):
            return None
        if not 0 <= column < len(self.events):
            return None
        return self.values.get((self.channels[row], self.events[column]))

    def pairs_at(self, row: int,
                 column: int) -> list[tuple[str, int]]:
        """The (channel, event) pairs one selected cell stands for.

        Column zero is the channel's name rather than a reading of it,
        so picking there asks for that channel across every playing —
        the row as a whole, which is what clicking a name should mean.
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
    """A grid of channels down and repeats across.

    Two comparisons share it, because they have the same shape: the
    waveform error against a target time history, in percent, and the
    RMS dB deviation of a shock spectrum against the one it had to
    meet. Only the unit in the heading differs.

    Blank rather than zero for a channel whose target asks for nothing.
    A zero would read as a channel that matched perfectly, which is the
    opposite of what is known about it.
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
    """A control channel per row: how far it is from its specification.

    The same job the transient's grid does, for a comparison that has
    no repeats to spread across — a PSD is already an average over
    them, so there is one column of numbers and not a grid of them.

    `plotted` is the set of DOF pairs on screen.
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
    """Channel, how far its level is out, and how much of its band is.

    Two readings and not one, because they disagree often enough to be
    worth seeing together: a channel can sit at exactly the right level
    and still be out of tolerance across half its band, and one that is
    2 dB low everywhere may never cross an abort limit at all.
    """
    return TableModel(rows, [
        Column('Channel', lambda h, r: _pair_text(h.rows[r][0]),
               alignment=LEFT),
        Column('RMS error [dB]', lambda h, r: h.rows[r][1], format=_number),
        Column('Outside abort [%]', lambda h, r: h.rows[r][2],
               format=_number),
    ], len, parent)
