"""What an object reads as, as a table: headers and rows of text.

The app has table *models* — editable, colored, unit-aware, and Qt to
the core. This is the other kind: the read-only rendering that goes in a
report, and that a script wants when it asks what is in a channel table
without opening a window.

One implementation, because the report and a script asking the same
question must not answer it differently.
"""

from __future__ import annotations

from typing import Any


def table_of(obj: Any, geometry: Any = None
             ) -> tuple[list[str], list[list[str]]] | None:
    """(headers, rows) for an object that reads as a table, else None.

    A shape set is its identified modal parameters; a channel table is
    its instrumentation — with the direction columns a geometry
    derives beside it, when one is given. Everything is text,
    formatted the way it is read rather than the way it is stored — a
    damping of 0.0213 is '2.130' percent, because that is the number
    an engineer quotes.
    """
    from .channel_table import ChannelTable
    from .shapes import ShapeSet

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
