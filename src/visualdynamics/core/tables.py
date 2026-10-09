"""What an object reads as, as a table: headers and rows of text.

The app has table *models* — editable, colored, unit-aware, and Qt to
the core. This is the other kind: the read-only rendering that goes in a
report, and that a script wants when it asks what is in a channel table
without opening a window.

One implementation, because the report and a script asking the same
question must not answer it differently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: decimals each matched-modes column is read to, None for as-is
MATCHED_DECIMALS = (None, 4, 3, None, 4, 3, None, 3, 2)


def matched_rows(matched: Any, objects: Mapping[str, Any]
                 ) -> tuple[list[str], list[tuple[Any, ...]]]:
    """A MatchedModes object's table as values: headers, and a row per
    pair — each set's mode number, frequency [Hz] and damping [%], the
    frequency error against the first set (text, signed), the pair's
    MAC, and the second set's size against the first's.

    The sets are read live by name, so their parameters restate
    themselves as the sets change, while the MAC is the object's own
    stored value (it came from the comparison as displayed, possibly
    projected). A set gone from `objects`, or a mode past its end,
    reads None rather than a guess. The last column is the scale each
    pair was normalized by, which an overlay — both shapes drawn to
    their own peak — cannot show: 1.00 down it is the evidence that a
    comparison is of shape alone.

    The window's table and the report's draw these rows; `table_of`
    is them as text.
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
    """One matched-modes value as it is read: a dash for nothing."""
    if value is None:
        return '—'
    return str(value) if decimals is None else f'{value:.{decimals}f}'


def table_of(obj: Any, geometry: Any = None,
             objects: Mapping[str, Any] | None = None
             ) -> tuple[list[str], list[list[str]]] | None:
    """(headers, rows) for an object that reads as a table, else None.

    A shape set is its identified modal parameters; a channel table is
    its instrumentation — with the direction columns a geometry
    derives beside it, when one is given; matched modes are the pairs
    beside both sets' parameters (`matched_rows`), read from the sets
    in `objects`. Everything is text, formatted the way it is read
    rather than the way it is stored — a damping of 0.0213 is '2.130'
    percent, because that is the number an engineer quotes.
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
