"""Responses against prediction in a report (report feature
`prediction`).

A `prediction` block reads one measured object, the response PSDs of a
run say, against the project's predictions (`core.prediction`). With
the feature on it draws, for each response channel with a prediction,
a figure of the measurement over its prediction scaled to the level
run, and after them one table: a row per predicted peak, giving the
channel, the frequency, level and Q predicted and measured, the
frequency difference, the ratio of the levels (named once, in the
column's heading, as an ASD ratio or an amplitude ratio, and given in
decibels) and the uncertainty the model stated. A note beneath says how
the predictions were scaled, where they came from (the customer's,
where they are), which predicted peaks have no measured peak in their
band, and what the half-power reader warned of.

Whether a response is as predicted, different and explained, or
different and not explained is the analyst's answer and is never
written here; the table is what lets a reader give it. The figures and
the table fall in the Results section of the skeleton, which the
program fills; nothing is offered as the author's text.

Where the comparison takes its predictions and its level is one
function, `inputs`, so a later source of run levels (a run log) feeds
the comparison there and nowhere else. With the feature off a
`prediction` block is not drawn at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: the feature's key in `report.features`
FEATURE = 'prediction'

#: the table's caption when the block gives none
CAPTION = 'Responses against prediction'


def inputs(block: Mapping[str, Any], objects: Mapping[str, Any],
           object_groups: Sequence[Mapping[str, Any]] | None = None
           ) -> tuple[Any, Any, str, float | None] | None:
    """What a block compares: (predictions, the measured object, its
    name, the level stated for the run), or None while either binding
    does not resolve.

    Today the predictions are the project's object the block names, and
    the level is the one the block states (`'level'`, dB relative to
    full level; None leaves it to the scaling a PSD holds,
    `core.prediction.level_run`). This is the one place the comparison
    takes them from, so a later source of predictions or run levels is
    read here.

    Raises
    ------
    TypeError
        When the block's predictions are not a predictions object, or
        its level is not a number.
    """
    from ..core.prediction import Predictions
    from ..core.report import resolve_binding

    source = resolve_binding(block.get('source'), objects, object_groups)
    name = resolve_binding(block.get('measured'), objects, object_groups)
    if source not in objects or name not in objects:
        return None
    predictions = objects[source]
    if not isinstance(predictions, Predictions):
        raise TypeError(f'{source!r} is a {type(predictions).__name__}, '
                        'not a predictions object')
    level = block.get('level')
    if level is not None and (isinstance(level, bool)
                              or not isinstance(level, (int, float))):
        raise TypeError(f'the level run is a number of dB, not {level!r}')
    return predictions, objects[name], str(name), level


def build(block: Mapping[str, Any], objects: Mapping[str, Any], us: Any,
          object_groups: Sequence[Mapping[str, Any]] | None = None
          ) -> list[dict[str, Any]] | None:
    """The page blocks for one `prediction` block: a figure per channel
    compared, then the table, then its note. None while a binding does
    not resolve; a single paragraph saying why when the comparison is
    refused, since a comparison that read nothing says so."""
    from ..core.prediction import compare

    try:
        found = inputs(block, objects, object_groups)
        if found is None:
            return None
        predictions, measured, name, level = found
        comparison = compare(predictions, measured, level)
    except (TypeError, ValueError) as refusal:
        return [{'kind': 'text',
                 'text': f'**{CAPTION}:** nothing compared; {refusal}.'}]
    compared = [channel for channel in comparison.channels
                if not channel.note]
    if not compared:
        # a numbered table with no rows is a handle to nothing: the
        # note alone says what was not compared and why (2026-10-28)
        return [{'kind': 'text',
                 'text': f'**{CAPTION}:** nothing compared.\n\n'
                         + _note(comparison)}]
    grid = _grid(measured)
    out = [_figure(channel, comparison, measured, us, grid)
           for channel in compared]
    out.append(_table(block, comparison, measured, name, us))
    out.append({'kind': 'text', 'text': _note(comparison)})
    return out


def _display(measured: Any, record: int, us: Any) -> tuple[str, Any]:
    """(the axis label, SI-to-display) for record `record`'s quantity."""
    dimension = measured.ordinate_dim[record]
    return us.label_text(dimension), (
        lambda values: us.from_si(np.asarray(values, dtype=float),
                                  dimension))


def _logs(values: Any) -> np.ndarray:
    """log10 of `values`, NaN (a gap) at or below zero."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(values > 0.0,
                        np.log10(np.where(values > 0.0, values, 1.0)),
                        np.nan)


def _grid(measured: Any) -> dict[str, Any]:
    """The measurement's frequency lines above 0 Hz, in decades, as
    every figure of one block draws them: worked out once, since every
    channel shares them. `edges` are the bins a density is stepped
    across, None for a curve read at its lines."""
    from ..plot import bin_edges, drawing_shape
    from . import _compact, _decades, _finite

    x = np.asarray(measured.abscissa, dtype=float)
    keep = x > 0.0
    lx = _decades(x[keep], True)
    edges = (_finite(_decades(bin_edges(x[keep]), True))
             if drawing_shape(measured) == 'steps' else None)
    return {'keep': keep, 'decades': lx, 'x': [_compact(v) for v in lx],
            'edges': edges}


def _figure(channel: Any, comparison: Any, measured: Any, us: Any,
            grid: Mapping[str, Any]) -> dict[str, Any]:
    """One channel's figure: the measurement over its prediction scaled
    to the level run, on log-log axes, the prediction dashed so the two
    read apart in a monochrome copy."""
    from . import _decades, _decimate, _finite

    prediction = channel.prediction
    record = channel.record
    label, shown = _display(measured, record, us)
    values = np.abs(np.asarray(measured.display_ordinate(us, [record])[0]))
    cx, cy = _decimate(grid['decades'], _logs(values[grid['keep']]))
    measured_curve: dict[str, Any] = {
        'label': f'{channel.label} measured', 'x': None, 'y': _finite(cy),
        'ink': True}
    built: dict[str, Any] = {
        'kind': 'plot', 'logx': True, 'logy': True, 'x': list(grid['x']),
        'xlabel': f'frequency [{us.label_text("frequency")}]',
        'ylabel': label, 'curves': [measured_curve]}
    if len(cx) != len(grid['x']):
        # thinned for the file: on its own lines, drawn through them
        measured_curve['x'] = _finite(cx)
    elif grid['edges'] is not None:
        built['steps'] = True
        built['edges'] = list(grid['edges'])
    if prediction.curve is not None:
        frequencies, predicted = prediction.curve
        on = frequencies > 0.0
        scaled = shown(prediction.si(predicted[on]) * channel.factor)
        px, py = _decimate(_decades(frequencies[on], True), _logs(scaled))
        drawn = {'x': _finite(px), 'y': _finite(py)}
    else:
        # the peaks alone: each predicted level drawn across the band it
        # is looked for in, nothing drawn between, since the model gave
        # no curve to draw there
        xs: list[float] = []
        ys: list[float] = []
        for row in channel.rows:
            level = float(_logs(shown(row.predicted_level)))
            xs += [*np.log10(row.band), np.nan]
            ys += [level, level, np.nan]
        drawn = {'x': _finite(xs[:-1]), 'y': _finite(ys[:-1])}
    level = ('' if comparison.level_db is None
             else f', scaled to {comparison.level_db:+g} dB')
    built['curves'].append({'label': f'prediction{level}', 'dash': True,
                            'steps': False, **drawn})
    noun = {'asd': 'ASD', 'amplitude': 'amplitude',
            'transmissibility': 'transmissibility'}[comparison.quantity]
    built['caption'] = (f'{channel.label}: measured {noun} over its '
                        f'prediction{level}')
    return built


def _cell(value: float | None, figures: int) -> str:
    from ..core.half_power import _figures

    return '—' if value is None else _figures(value, figures)


def _uncertainty(prediction: Any) -> str:
    parts = []
    if prediction.frequency_uncertainty is not None:
        parts.append(f'±{prediction.frequency_uncertainty:g} %')
    if prediction.level_uncertainty is not None:
        parts.append(f'±{prediction.level_uncertainty:g} dB')
    return ', '.join(parts) or 'not stated'


def _table(block: Mapping[str, Any], comparison: Any, measured: Any,
           name: str, us: Any) -> dict[str, Any]:
    """The table: a row per predicted peak, levels in the report's units,
    the ratio named once, in its column's heading."""
    from ..core.half_power import _figures

    rows = comparison.rows
    labels = {_display(measured, channel.record, us)[0]
              for channel, _row in rows}
    shared = labels.pop() if len(labels) == 1 else None
    unit = f' [{shared}]' if shared else ''
    # 'ASD ratio' keeps its capitals; 'amplitude ratio' takes one
    ratio = comparison.ratio_name
    headers = ['Channel', 'Predicted [Hz]', f'Predicted level{unit}',
               'Predicted Q', 'Measured [Hz]', f'Measured level{unit}',
               'Measured Q', 'Frequency difference [%]',
               f'{ratio[:1].upper()}{ratio[1:]} (dB)', 'Model uncertainty']
    body = []
    for channel, row in rows:
        label, shown = _display(measured, channel.record, us)
        suffix = '' if shared else f' {label}'

        def level(value, shown=shown, suffix=suffix):
            return ('—' if value is None
                    else _figures(float(shown(value)), 3) + suffix)

        ratio, decibels = row.ratio, row.ratio_db
        said = ('—' if ratio is None else _figures(ratio, 3)
                + ('' if decibels is None else f' ({decibels:+.1f} dB)'))
        difference = row.frequency_difference
        body.append([channel.label, _cell(row.predicted_frequency, 4),
                     level(row.predicted_level), _cell(row.predicted_q, 3),
                     _cell(row.measured_frequency, 4),
                     level(row.measured_level), _cell(row.measured_q, 3),
                     '—' if difference is None else f'{difference:+.1f}',
                     said, _uncertainty(channel.prediction)])
    caption = str(block.get('caption') or CAPTION)
    return {'kind': 'table', 'caption': f'{caption}: {name}, '
                                        f'{comparison.scaling}',
            'headers': headers, 'rows': body}


def _note(comparison: Any) -> str:
    """What the table cannot say in its cells, as Markdown: where the
    predictions came from, the peaks with no measured counterpart in
    their band, the channels not compared and why, the channels with no
    record, and the reader's warnings."""
    from ..core.half_power import _figures

    lines = []
    sources: dict[tuple[str, bool], list[str]] = {}
    for channel in comparison.channels:
        prediction = channel.prediction
        sources.setdefault((prediction.source, prediction.customer),
                           []).append(channel.label)
    for (source, customer), labels in sources.items():
        said = source or 'a source not stated'
        if customer:
            said += ', supplied by the customer'
        lines.append(f'- Predictions for {", ".join(labels)}: {said}.')
    for channel, row in comparison.rows:
        at = f'{channel.label} at {_figures(row.predicted_frequency, 4)} Hz'
        if row.missing:
            lines.append(f'- {at}: {row.missing}.')
        for warned in row.notes:
            lines.append(f'- {at}: {warned}.')
    for channel in comparison.channels:
        if channel.note:
            lines.append(f'- {channel.label}: not compared; {channel.note}.')
    if comparison.absent:
        lines.append(f'- No record of {", ".join(comparison.absent)} in '
                     'this measurement.')
    if not comparison.rows:
        lines.append('- Nothing was compared.')
    return '\n'.join(lines)
