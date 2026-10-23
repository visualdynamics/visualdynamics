"""An object's contents as a table: headers and rows of text.

The app has its own table models, which are editable, colored, aware of
units and built on Qt. This module is the read-only version: the table
that goes in a report, and the one a script gets when it asks what is in
a channel table without opening a window. The report and a script use
the same functions here, so they always show the same table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: decimals each matched-modes column is read to, None for as-is
MATCHED_DECIMALS = (None, 4, 3, None, 4, 3, None, 3, 2)


def matched_rows(matched: Any, objects: Mapping[str, Any]
                 ) -> tuple[list[str], list[tuple[Any, ...]]]:
    """The table of a `MatchedModes` object, as values.

    Returns the headers and one row per pair: each set's mode number
    (one-based), frequency [Hz] and damping [%], the frequency error of
    the second set against the first as signed text in percent, the
    pair's MAC, and the size of the second set's mode relative to the
    first's (`scale_ratios`).

    The two sets are looked up by name in `objects` each time, so the
    frequencies and damping follow any change to the sets. The MAC is
    the value stored in the object, because it came from the comparison
    as it was displayed, possibly after a projection. A set missing from
    `objects`, or a mode number past the end of its set, gives None (and
    a dash for the frequency error). The last column shows how the two
    shapes of each pair are scaled relative to each other, which an
    overlay cannot show because it draws each shape to its own peak; a
    column of 1.00 means the comparison is of shape alone.

    The window's table and the report's table are drawn from these
    rows; `table_of` gives the same rows as text.
    """
    from .shapes import scale_ratios

    a_name, b_name = matched.first, matched.second
    a, b = objects.get(a_name), objects.get(b_name)

    def parameters(shape_set: Any,
                   mode: int) -> tuple[float | None, float | None]:
        if shape_set is None or not 0 <= mode < shape_set.num_shapes:
            return None, None
        return (float(shape_set.frequency[mode]),
                float(shape_set.damping[mode]) * 100.0)

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
    headers = [f'{a_name} Mode', 'Frequency [Hz]', 'Damping [%]',
               f'{b_name} Mode', 'Frequency [Hz]', 'Damping [%]',
               'Δf [%]', 'MAC', f'{b_name}/{a_name}']
    return headers, rows


def cell_text(value: Any, decimals: int | None) -> str:
    """One matched-modes value as text, to `decimals` places; None is a dash."""
    if value is None:
        return '—'
    return str(value) if decimals is None else f'{value:.{decimals}f}'


def mode_heading(mode: int) -> str:
    """The column heading of mode `mode` (from 0) in a table of shape
    values: 'M1' for the first."""
    return f'M{mode + 1}'


def shape_value_text(value: Any) -> str:
    """One shape value as text, to six significant figures; a complex
    one as its real and imaginary parts, '0.5-0.25j'."""
    import numpy as np

    if np.iscomplexobj(value):
        return f'{value.real:.6g}{value.imag:+.6g}j'
    return f'{float(value):.6g}'


def shape_values(shapes: Any) -> tuple[list[str], list[list[str]]]:
    """``(headers, rows)`` of a shape set's values: a row per DOF, the
    DOF first, then a column per mode ('M1', 'M2', ...).

    Rows are DOFs and columns modes, the way a mode shape matrix is
    usually written, which is the transpose of `ShapeSet.shape_matrix`.
    """
    # Brandon, 2026-10-22, with the table of them beside the mode list.
    headers = ['DOF'] + [mode_heading(m) for m in range(shapes.num_shapes)]
    rows = [[str(dof)] + [shape_value_text(value)
                          for value in shapes.shape_matrix[:, i]]
            for i, dof in enumerate(shapes.coordinate)]
    return headers, rows


def table_of(obj: Any, geometry: Any = None,
             objects: Mapping[str, Any] | None = None
             ) -> tuple[list[str], list[list[str]]] | None:
    """``(headers, rows)`` for an object that has a table, else None.

    A shape set gives its modal parameters (mode, frequency, damping,
    description). A channel table gives its columns, plus the direction
    columns derived from `geometry` when one is given. Matched modes
    give the pairs beside both sets' parameters (`matched_rows`), with
    the sets looked up in `objects`. Every cell is text, formatted the
    way it is usually quoted rather than the way it is stored: a damping
    of 0.0213 is '2.130' percent.
    """
    from .channel_table import ChannelTable
    from .matches import MatchedModes
    from .shapes import ShapeSet

    if isinstance(obj, MatchedModes):
        headers, rows = matched_rows(obj, objects or {})
        return headers, [[cell_text(value, decimals) for value, decimals
                          in zip(row, MATCHED_DECIMALS)] for row in rows]

    if isinstance(obj, ShapeSet):
        return (['Mode', 'Frequency [Hz]', 'Damping [%]', 'Description'],
                [[str(m + 1), f'{float(obj.frequency[m]):.4f}',
                  f'{float(obj.damping[m]) * 100:.3f}', obj.description[m]]
                 for m in range(obj.num_shapes)])
    if isinstance(obj, ChannelTable):
        # every column, because the schema is already the curated set:
        # a Rattlesnake save's coupling and feedback wiring stop at the
        # importer now, so there is no acquisition plumbing left here to
        # cut and no second list to keep in step with the first
        from .channel_table import DERIVED_COLUMNS, title_of

        names = obj.column_names
        derived = list(DERIVED_COLUMNS) if geometry is not None else []
        return ([title_of(name) for name in names + derived],
                [[str(obj[name][r]) for name in names]
                 + (obj.derived_cells(r, geometry) if derived else [])
                 for r in range(obj.num_channels)])
    return None
