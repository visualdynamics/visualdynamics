"""Time-history pairing, shared clocks, and time response assurance."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import numpy as np

from .data import TimeHistory


def response_records(data: TimeHistory) -> list[int]:
    """Records whose channel role is response, leaving force references out."""
    roles = data.channel_roles()
    return [i for i in range(data.num_records)
            if roles.get((data.response_dof[i], data.known_dim(i))) == 'response']


def retarget_pairs(settings: dict, mapping: dict[str, str]) -> None:
    """Keep saved pair identities and their recipes following renamed objects."""
    settings['sources'] = [mapping.get(n, n) for n in settings['sources']]
    settings['pairs'] = [[mapping.get(a, a), i, mapping.get(b, b), j]
                         for a, i, b, j in settings['pairs']]


def match_records(project: Any, sources: list[str], pairs: Any = None,
                  records: Any = None) -> list[list[Any]]:
    """Resolve explicit pairs, or match exactly two responses per channel key.

    Explicit pairs are (left name, record, right name, record). They can map
    different node labels, but never different quantities or undefined units.
    Automatic matching uses signed DOF, dimension, and dimension hint. A
    missing or repeated match is refused rather than guessed from a name.
    """
    if not sources or any(not isinstance(project[n], TimeHistory) for n in sources):
        raise ValueError('Select time histories to compare')
    if pairs is None:
        grouped = defaultdict(list)
        for name in sources:
            data = project[name]
            for i in (records or {}).get(name, response_records(data)):
                grouped[data.channel_key(i)].append((name, int(i)))
        if not grouped:
            raise ValueError('No response channels selected; force references are excluded')
        pairs = []
        for key, members in grouped.items():
            if len(members) != 2:
                raise ValueError(f'{key[0]} ({key[1]}): expected two responses, '
                                 f'found {len(members)}; select both sets or use manual pairing')
            pairs.append([*members[0], *members[1]])
    if not pairs:
        raise ValueError('No time-history pairs were supplied')
    resolved = []
    used = set()
    for left, i, right, j in pairs:
        if left not in sources or right not in sources:
            raise ValueError('A manual pair names a source outside this selection')
        a, b = project[left], project[right]
        if not (0 <= i < a.num_records and 0 <= j < b.num_records):
            raise ValueError('A pair has an invalid record index')
        if (left, i) == (right, j) or (left, i) in used or (right, j) in used:
            raise ValueError('Each selected record must occur in only one pair')
        if not a.ordinate_unit[i] or not b.ordinate_unit[j]:
            raise ValueError(f'{left} / {right}: define units before comparing')
        if np.iscomplexobj(a.ordinate[i]) or np.iscomplexobj(b.ordinate[j]):
            raise ValueError('TRAC needs real time responses')
        if a.channel_key(i)[1:] != b.channel_key(j)[1:]:
            raise ValueError(f'{left} / {right}: quantities or unit dimensions differ')
        used.update(((left, i), (right, j)))
        resolved.append([left, int(i), right, int(j)])
    return resolved


def compare(project: Any, sources: list[str], *, pairs: Any = None,
            records: Any = None, start: float | None = None,
            stop: float | None = None) -> TimeHistory:
    """Put paired SI responses on a shared clock without fitting their values.

    The coarsest source clock is retained inside the common overlap; other
    rows are linearly interpolated to it, without extrapolation. No lag,
    sign, amplitude fitting, detrending, or spectral filtering is performed.
    The full common overlap is stored; start/stop limit the score and view.
    """
    pairs = match_records(project, sources, pairs, records)
    entries = [(project[n], i) for a, i, b, j in pairs for n, i in ((a, i), (b, j))]
    for data, _index in entries:
        x = data.abscissa
        if len(x) < 2 or not np.isfinite(x).all() or not (np.diff(x) > 0).all():
            raise ValueError('Comparison needs at least two finite, increasing timestamps')
    low = max(float(d.abscissa[0]) for d, _i in entries)
    high = min(float(d.abscissa[-1]) for d, _i in entries)
    clock = max(entries, key=lambda e: float(np.median(np.diff(e[0].abscissa))))[0].abscissa
    clock = clock[(clock >= low) & (clock <= high)]
    if len(clock) < 2:
        raise ValueError('The selected time histories have no common time interval')
    rows = [np.interp(clock, d.abscissa, d.ordinate[i]) for d, i in entries]
    output = TimeHistory(clock.copy(), np.asarray(rows),
                         response_dof=[d.response_dof[i] for d, i in entries],
                         ordinate_dim=[d.ordinate_dim[i] for d, i in entries],
                         comment=[n for a, _i, b, _j in pairs for n in (a, b)])
    output.time_comparison = {'pairs': pairs, 'sources': list(sources),
                              'start': float(clock[0]) if start is None else float(start),
                              'stop': float(clock[-1]) if stop is None else float(stop)}
    scores(output)  # also validates the requested window before keeping anything
    return output


def sides(data: TimeHistory, window: tuple[float, float] | None = None) -> tuple[TimeHistory, TimeHistory]:
    """Return the two paired sides with shared station labels for plotting.

    The comparison's original right-side identities stay in its mapping;
    the temporary display arrays use the left DOFs on both sides so a
    manually mapped pair occupies one station. A window restricts the
    displayed samples to its inclusive bounds.
    """
    dofs = data.response_dof[::2]
    common = {'response_dof': dofs, 'ordinate_dim': data.ordinate_dim[::2],
              'block': [f'(pair {i+1})' if dofs.count(dof) > 1 else ''
                        for i, dof in enumerate(dofs)]}
    samples = slice(None)
    if window is not None:
        start, stop = window
        samples = slice(data.abscissa.searchsorted(start), data.abscissa.searchsorted(stop, side='right'))
    clock = data.abscissa[samples]
    return (TimeHistory(clock, data.ordinate[::2, samples], **common),
            TimeHistory(clock, data.ordinate[1::2, samples], **common))


def scores(data: TimeHistory, start: float | None = None,
           stop: float | None = None) -> list[dict[str, Any]]:
    """TRAC, signed correlation, and amplitude ratio for every saved pair.

    A zero, constant, or nonfinite record has no reported score; its reason
    travels with the row instead of a false pass. TRAC is the squared
    normalized inner product, so sign reversal and constant gain leave it
    unchanged. Signed correlation and RMS ratio make those limits visible.
    sTRAC is the signed least-squares multiplier from the second trace
    to the first, with an ideal value of one.

    References
    ----------
    The standard metric and its amplitude limitation are described in:

    1. Coble et al., *Mechanical Systems and Signal Processing*,
       [doi:10.1016/j.ymssp.2024.111522](https://doi.org/10.1016/j.ymssp.2024.111522).
    """
    setting = data.time_comparison
    start = setting['start'] if start is None else float(start)
    stop = setting['stop'] if stop is None else float(stop)
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError('TRAC end time must be finite and greater than start time')
    if start < data.abscissa[0] or stop > data.abscissa[-1]:
        raise ValueError('TRAC window must stay within the common time interval')
    keep = (data.abscissa >= start) & (data.abscissa <= stop)
    if keep.sum() < 2:
        raise ValueError('TRAC window contains fewer than two samples')
    rows = []
    for k, pair in enumerate(setting['pairs']):
        a, b = data.ordinate[2*k:2*k+2, keep]
        reason = ''
        if not np.isfinite([a, b]).all():
            reason = 'Nonfinite samples in TRAC window'
        elif not np.any(a) or not np.any(b):
            reason = 'Zero-energy signal in TRAC window'
        elif np.ptp(a) == 0 or np.ptp(b) == 0:
            reason = 'Constant signal in TRAC window'
        trac = correlation = ratio = strac = None
        if not reason:
            sa, sb = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
            u, v = a/sa, b/sb
            aa, bb = float(u@u), float(v@v)
            correlation = float(np.clip((u@v)/np.sqrt(aa*bb), -1, 1))
            trac = correlation**2
            ratio = float((sb/sa)*np.sqrt(bb/aa))
            # MATLAB fits the experimental response as a multiple of
            # the calculated response. Comparison pairs store experiment first.
            strac = float((sa/sb)*(v@u)/bb)
        rows.append({'pair': pair, 'dof': data.response_dof[2*k],
                     'trac': trac, 'strac': strac, 'correlation': correlation, 'rms_ratio': ratio,
                     'samples': int(keep.sum()), 'reason': reason})
    return rows


def scaled(data: TimeHistory, factor: float, records: Any = None,
           include_references: bool = False) -> TimeHistory:
    """Return an independent scaled copy, preserving SI units and timing."""
    if not np.isfinite(factor):
        raise ValueError('Scale factor must be finite')
    selected = (list(range(data.num_records)) if include_references else response_records(data))
    if records is not None:
        selected = list(records)
    if not selected:
        raise ValueError('No response records selected to scale')
    result = copy.deepcopy(data)
    if any(not isinstance(i, (int, np.integer)) or not 0 <= i < data.num_records
           for i in selected):
        raise ValueError('Invalid selected record index')
    result.ordinate[selected] *= float(factor)
    return result
