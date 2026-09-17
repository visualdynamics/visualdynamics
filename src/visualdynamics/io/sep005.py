"""SEP 005 — the sdypy ecosystem's unified timeseries, in and out.

SEP 005 (`SEP 5 <https://github.com/sdypy/sdypy/blob/main/docs/seps/sep-0005.rst>`_)
is the interchange standard of the open-source sdypy project: one plain
dict per timeseries — ``data`` shaped ``(n,)`` or ``(m, n)``, a
``name``, per-channel ``unit_str``, and either a sampling frequency
``fs`` or a ``time`` vector — with a list of dicts for several series.
It is an **in-memory standard, not a file format**, so this module
converts objects rather than registering a file importer:
`from_sep005` turns the dicts a sdypy package hands over into
`TimeHistory` objects, and `TimeHistory.to_sep005` goes the other way.

Nothing here imports anything from sdypy — the standard *is* the dict,
and staying import-free is what keeps the license boundary trivial.

Timeseries are the standard's whole scope, which is why only
`TimeHistory` converts: sdypy has no richer exchange form for FRFs,
PSDs or geometry, so those stay in ``.vdyn`` / UNV.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: SEP 005 quantity letters -> the dimension recorded as the claim in
#: `dimension_hint` when a series arrives without a usable unit. The
#: names are the package's own dimension vocabulary (units.py), where
#: a displacement is a 'length' and a stress is a 'pressure' — a hint
#: in words the units machinery cannot read would narrow nothing.
QUANTITIES = {'f': 'force', 'a': 'acceleration', 'v': 'velocity',
              'd': 'length', 'e': 'strain', 's': 'pressure'}

#: and the way back, for the export's optional quantity letter
DIMENSION_TO_QUANTITY = {'force': 'f', 'acceleration': 'a',
                         'velocity': 'v', 'length': 'd',
                         'strain': 'e', 'pressure': 's'}


def from_sep005(timeseries: dict[str, Any] | list[dict[str, Any]]
                ) -> Any:
    """SEP 005 timeseries into `TimeHistory` objects.

        history = visualdynamics.from_sep005({'data': y, 'fs': 256.0,
                                              'name': 'run 4',
                                              'unit_str': 'm/s²'})

    One dict returns one `TimeHistory`; a list — the standard's form
    for several series — returns ``{name: TimeHistory}``, numbering a
    repeated name the way the project tree would.

    ``unit_str`` entries that parse are *declared* on the object
    (values converted to SI, exactly as `define_units` would), because
    the producer stated them; one that does not parse leaves that
    channel's values raw with the claim kept in `dimension_hint`, where
    ``quantity`` also lands when there is no unit at all. Nothing is
    ever scaled by a guess.

    Refused, with the reason: a series with no ``data``, with neither
    ``fs`` nor ``time``, a ``time`` vector of the wrong length, or a
    ``channel_name`` list that does not match the channel count.
    """
    if isinstance(timeseries, dict):
        return _one(timeseries)
    out: dict[str, Any] = {}
    for series in timeseries:
        name = str(series.get('name', 'Time History')) or 'Time History'
        unique, n = name, 1
        while unique in out:
            n += 1
            unique = f'{name} ({n})'
        out[unique] = _one(series)
    return out


def _one(series: dict[str, Any]) -> Any:
    from ..core.data import TimeHistory

    if 'data' not in series:
        raise ValueError("a SEP 005 timeseries carries its samples in "
                         "'data'; this one has none")
    data = np.atleast_2d(np.asarray(series['data'], dtype=float))
    if data.ndim != 2:
        raise ValueError(f"'data' must be (n,) or (m, n), "
                         f'not {np.asarray(series["data"]).shape}')
    channels, samples = data.shape

    if 'time' in series and series['time'] is not None:
        abscissa = np.asarray(series['time'], dtype=float)
        if abscissa.shape != (samples,):
            raise ValueError(f"'time' has {abscissa.size} entries for "
                             f'{samples} samples — they must match')
    elif series.get('fs'):
        abscissa = np.arange(samples) / float(series['fs'])
    else:
        raise ValueError("a SEP 005 timeseries says its sampling as "
                         "'fs' or as a 'time' vector; this one has "
                         'neither')

    names = series.get('channel_name')
    if names is None:
        names = [str(i + 1) for i in range(channels)]
    elif isinstance(names, str):
        names = [names]
    else:
        names = [str(name) for name in names]
    if len(names) != channels:
        raise ValueError(f"'channel_name' lists {len(names)} names for "
                         f'{channels} channels — they must match')

    hint = QUANTITIES.get(series.get('quantity'))
    out = TimeHistory(abscissa, data, response_dof=names,
                      dimension_hint=hint,
                      comment=str(series.get('name', '') or ''))

    units = series.get('unit_str')
    if isinstance(units, str) or units is None:
        units = [units] * channels
    for i, unit in enumerate(units[:channels]):
        if not unit:
            continue
        try:
            out.define_units({i: str(unit)})
        except Exception:  # noqa: BLE001 — pint's parse failures span
            # UndefinedUnitError (an AttributeError), DefinitionSyntaxError
            # and ValueError; whichever it is, the answer is the same:
            # the claim is kept as a hint and the values stay raw
            if out.dimension_hint[i] is None:
                out.dimension_hint[i] = str(unit)
    return out
