"""Convert between SEP 005 timeseries, the sdypy project's standard,
and `TimeHistory` objects.

SEP 005 (`SEP 5 <https://github.com/sdypy/sdypy/blob/main/docs/seps/sep-0005.rst>`_)
is the interchange standard of the open-source sdypy project: one plain
dict per timeseries, with ``data`` shaped ``(n,)`` or ``(m, n)``, a
``name``, per-channel ``unit_str``, and either a sampling frequency
``fs`` or a ``time`` vector, and a list of dicts for several series. It
is a standard for data in memory, not a file format, so this module
converts objects and does not register a file importer. `from_sep005`
turns the dicts into `TimeHistory` objects, and `TimeHistory.to_sep005`
converts the other way.

This module imports nothing from sdypy; the standard is the dict.

The standard covers timeseries only, so only `TimeHistory` converts.
FRFs, PSDs and geometry are exchanged as ``.vdyn`` or universal files.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .stated_units import declare_stated_units

# Staying free of imports keeps the license boundary simple.

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
    """Convert SEP 005 timeseries to `TimeHistory` objects.

        history = visualdynamics.from_sep005({'data': y, 'fs': 256.0,
                                              'name': 'run 4',
                                              'unit_str': 'm/s²'})

    One dict returns one `TimeHistory`. A list, the standard's form for
    several series, returns ``{name: TimeHistory}``, with a repeated
    name numbered 'name (2)' and so on, as in the project tree. The
    channels are named by ``channel_name``, or numbered from 1 without
    it, and the series ``name`` becomes the comment.

    A ``unit_str`` entry that parses is declared on its channel, so the
    values are converted to SI as `define_units` converts them, since the
    producer stated the unit. One that does not parse leaves that
    channel's values as they are and keeps the unit string in
    `dimension_hint`, unless the series ``quantity`` letter has already
    put a dimension there. Values are never scaled by a guess.

    Raises ValueError, with the reason, for a series with no ``data``,
    ``data`` with more than two dimensions, neither ``fs`` nor ``time``,
    a ``time`` vector of the wrong length, or a ``channel_name`` list
    that does not match the channel count.

    Parameters
    ----------
    timeseries : dict or list of dict
        One SEP 005 series, or a list of them.

    Returns
    -------
    TimeHistory or dict of str to TimeHistory
        One for one dict; for a list, every series by name.
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
    # a unit that cannot be declared keeps the claim as a hint and the
    # values raw; which units can be is the one rule every reader of
    # stated units follows (stated_units)
    for i, unit, _reason in declare_stated_units(out, units[:channels]):
        if out.dimension_hint[i] is None:
            out.dimension_hint[i] = unit
    return out
