"""Render a Report to one self-contained HTML file.

The whole point is the reader: they open the file in the browser they
already have — no install, no network, no third-party code inside the
deliverable. Everything interactive is a few hundred lines of our own
JavaScript: an orthographic trackball scene that animates mode shapes
with the same phase math as the desktop animator, and a zoomable
log-magnitude plot with visualdynamics's own axis labels. Data rides along as one
JSON payload, already converted to the display unit system, so the file
shows exactly what the screen showed.
"""

from __future__ import annotations

import html as html_escape
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..decimate import peak_decimate
from ..plot import as_power_law, bin_edges, drawing_shape
from ..theme import OVERLAY_ALPHA
from ..units import DEFAULT_SYSTEM

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.report import Report
    from ..units import UnitSystem

MAX_REPORT_CURVES = 24

#: points a 2-D time-history figure carries in all, shared by its
#: curves. The stage's two million per page made a 144-channel run at
#: 16 kHz a 200 MB report; a record's figure is read for where the
#: run was at level and how the frames sat on it, which a quarter of
#: a million peak-kept points draw at every zoom the page offers
#: (Brandon, 2026-09-20: "switch those plots to 2D so the report can
#: be smaller").
TIME_FIGURE_POINTS = 160_000

#: points one figure may carry, across every curve on it. A report is
#: a deliverable that gets zoomed into — Brandon found the shock
#: traces thinned when he looked closely (2026-08-24) — and a figure
#: whose data has been reduced cannot answer a question asked of it
#: later, however faithful the reduction was at the opening view. So
#: the budget is set where the data itself usually fits: every plate
#: run here draws untouched, sample for sample, and only a survey of
#: stress-test size is thinned at all.
#:
#: It is a bound on the *file*, not a preference: at roughly twenty
#: bytes of JSON a point this is some forty megabytes of geometry in
#: a self-contained page, and the drone's 13-million-point random run
#: would be 261 MB unthinned. Where it does bite, the thinning is
#: `decimate.peak_decimate` — every extreme kept, no spike walked
#: past — and the caption says it happened.
MAX_FIGURE_POINTS = 2_000_000

#: what one curve of a flat plot may carry, from that budget
MAX_POINTS_PER_CURVE = MAX_FIGURE_POINTS // MAX_REPORT_CURVES
MAX_MAP_COLUMNS = 1200

#: how many records one 3-D stage figure may hold. What each record
#: brings is `MAX_FIGURE_POINTS` shared between them, so a figure of
#: a few channels draws them whole and one of forty-eight thins only
#: if the records are very long.
MAX_STAGE_RECORDS = 48

#: how many records one stage *block* may draw, across as many
#: continuation figures as that takes. There is no cap on the figures
#: themselves (Brandon, 2026-08-24) — an object with more channels
#: than a stage holds legibly continues into the next figure, exactly
#: as the app's own pages do. This bounds the file instead, and for a
#: measured reason: a full figure is ~736 KB of geometry inside the
#: self-contained HTML, so the 25,380-record stress set would want
#: 529 figures and a 399 MB document that no browser opens and the
#: report editor cannot load. A thousand is ~21 figures and ~15 MB —
#: above every real survey block here (the largest is 560) and far
#: below the wall.
MAX_STAGE_TOTAL = 1000

#: time columns a report scalogram is thinned to. About the canvas's
#: own width: finer is invisible and weighs the file down — a hundred
#: frequency rows each carry this many points.
#: 240 rather than the 700 the line drawing carried: the sheet fills
#: a quad per cell per repaint, and a rotating figure has to stay
#: under the hand (principle 8) — the ridges survive peak-hold at
#: this width, and the note says the thinning either way
SCALOGRAM_COLUMNS = 240


def render_html(report: Report, objects: Mapping[str, Any],
                unit_system: UnitSystem | None = None, edit: bool = False,
                channel_js: str | None = None,
                links: Sequence[Mapping[str, Any]] | None = None,
                selected: int | None = None,
                labels: list[tuple[str, str]] | None = None) -> str:
    """The report as one HTML document string.

    `links` is the project's link groups: symbolic bindings like
    '@basis:Frf' resolve against them, so a report depends on the
    project's *structure*, never on what anyone named their objects.
    Reading mode skips blocks whose references cannot resolve —
    an unbound template block is a slot to fill, not an error to show a
    reader. Edit mode keeps them as cards to rebind, tags every block
    with its index, frames the block at `selected`, and wires one
    message to the app over Qt's web channel — which block was
    clicked. Every act lives on the application's bar and pane
    (2026-09-08); the page carries no insert bars, block toolbars or
    editors. Only the app ever loads edit mode; the exported file
    carries none of it.

    `labels`, when given, is filled with (label, caption) for every
    numbered figure and table in order — what the editor's Reference
    menu offers, from the same numbering the page shows.
    """
    us = unit_system or DEFAULT_SYSTEM
    from ..theme import DARK, LIGHT, VIRIDIS

    payload = {'title': report.title, 'edit': bool(edit),
               'marking': report.marking,
               'marking_color': report.marking_color,
               # the page is told the colors it draws with rather
               # than carrying its own copies: one color scale and
               # one pair of mark colors, from the app's own theme.
               # Both themes ride along because the reader can flip
               # between them in the page.
               'viridis': [list(stop) for stop in VIRIDIS],
               'marks_color': {
                   side['name']: {
                       'band': side['averaging_band'],
                       'window': side['averaging_window']}
                   for side in (LIGHT, DARK)},
               'blocks': []}
    if edit:
        payload['selected'] = -1 if selected is None else int(selected)
    figures = tables = 0
    for index, block in enumerate(report.blocks):
        built = _build_block(block, objects, us, links)
        if built is None:
            if not edit:
                continue
            # tell the truth on the card: a block whose bindings all
            # resolve but that still has nothing to draw is not
            # 'unbound' — rebinding it would change nothing
            from ..core.report import resolve_binding
            needed = [block.get(key) for key in
                      ('source', 'geometry', 'dofs_source', 'shapes')
                      if block.get(key)]
            resolvable = bool(needed) and all(
                resolve_binding(name, objects, links) in objects
                for name in needed)
            built = {'kind': 'unbound',
                     'was': block.get('kind', 'block'),
                     'empty': resolvable}
        # one block can answer with several figures — a stage with
        # more channels than it holds legibly continues into the next
        # one. They share the block's index, so the editor edits the
        # block whichever of its figures was clicked.
        drawn = built if isinstance(built, list) else [built]
        for built in drawn:
            if built['kind'] in ('plot', 'mac', 'map', 'scene', 'image',
                                 'bars', 'stage', 'grid'):
                figures += 1
                built['label'] = f'Figure {figures}'
            elif built['kind'] == 'table':
                tables += 1
                built['label'] = f'Table {tables}'
            if edit:
                # the block's index rides with each of its figures, so a
                # click on any of them selects the block
                built['index'] = index
            payload['blocks'].append(built)
    # text renders last: only now does every figure have its number, so
    # {{figure:...}} references can resolve — and renumber themselves
    # the next time a block is added, removed, or moved
    from .markdown import to_html

    labeled = [(built['label'], built.get('caption', ''))
               for built in payload['blocks'] if built.get('label')]
    if labels is not None:
        labels[:] = labeled
    for built in payload['blocks']:
        if built['kind'] == 'text':
            built['html'] = to_html(
                _resolve_figures(built.pop('text'), labeled))
    data = json.dumps(payload, allow_nan=False)
    scripts = _JS + (_EDIT_JS if edit else '')
    shell = _PAGE
    if edit:
        # the app hands over Qt's own qwebchannel.js to inline, because
        # the editor page loads from a file and file: pages cannot
        # reach qrc:; the qrc tag remains for anything rendering edit
        # HTML without the app
        channel = (f'<script>{channel_js}</script>' if channel_js else
                   '<script src="qrc:///qtwebchannel/qwebchannel.js">'
                   '</script>')
        # edit pages trap script errors where the app can read them —
        # a blank editor with no diagnosis cost a debugging session
        trap = ('<script>window.__err = [];'
                "window.onerror = (m, s, l) => __err.push(m + ' @' + l);"
                '</script>')
        shell = shell.replace('<script id="data"',
                              trap + channel + '\n<script id="data"')
    return (shell.replace('__TITLE__', html_escape.escape(report.title))
                 .replace('__DATA__', data.replace('</', '<\\/'))
                 .replace('__CSS__', _CSS + (_EDIT_CSS if edit else ''))
                 .replace('__JS__', scripts))


#: significant digits a number keeps on the page. A figure is a
#: picture: seven digits draw it to a part in ten million, where the
#: seventeen a float prints by default were most of what made a
#: 144-channel report 200 MB (Brandon, 2026-09-20).
PAGE_DIGITS = 7


def _compact(value) -> float:
    return float(f'{float(value):.{PAGE_DIGITS}g}')


def _finite(values):
    return [None if not np.isfinite(v) else _compact(v) for v in values]


def _decades(coords, logx):
    """Coordinates for a frequency axis in decades: log10, a line at
    or below 0 Hz a gap. Applied to a finished curve — bin edges, a
    power law's fill-in — never to the lines they are built from."""
    if not logx:
        return np.asarray(coords, dtype=float)
    coords = np.asarray(coords, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(coords > 0,
                        np.log10(np.where(coords > 0, coords, 1.0)), np.nan)


def _positive_lines(source):
    """The source without its lines at or below 0 Hz, when its axis
    reads in decades — a controller's target starts at 0 Hz, which no
    log axis can draw; the app's plot drops the line silently and the
    figure has to as well, since a NaN coordinate is not JSON. The
    source is untouched: a shallow copy carries the sliced arrays."""
    import copy

    if not getattr(source, 'log_abscissa', False):
        return source
    x = np.asarray(source.abscissa, dtype=float)
    keep = x > 0
    if keep.all():
        return source
    out = copy.copy(source)
    out.abscissa = x[keep]
    out.ordinate = np.asarray(source.ordinate)[:, keep]
    if getattr(source, 'limits', None):
        out.limits = {name: np.asarray(values)[:, keep]
                      for name, values in source.limits.items()}
    widths = getattr(source, 'bandwidth', None)
    if widths is not None and np.ndim(widths) == 1 and len(widths) == len(x):
        out.bandwidth = np.asarray(widths)[keep]
    return out


def _thinned_note(drawn: int, held: int) -> str:
    """What to say on a figure whose data was reduced to fit it.

    Every path that thins says the same sentence (Brandon,
    2026-08-24: any plot where the data gets decimated should be
    clearly marked). A reader zooming into a figure is entitled to
    know whether they are looking at the record or at a reading of
    it — and the thinning is `decimate.peak_decimate`, which keeps
    the smallest and largest value of every slice, so the envelope
    is exact and no spike is walked past even where detail between
    the extremes is gone.
    """
    if drawn >= held:
        return ''
    return (f'Thinned for the file: {drawn} points drawn of {held}, '
            'every extreme kept.')


def _decimate(x, y, budget=None):
    """Peak-keeping decimation: a report is a document, not a scope."""
    cx, cy = peak_decimate(np.asarray(x), np.asarray(y),
                           MAX_POINTS_PER_CURVE if budget is None else int(budget))
    return [_compact(v) for v in cx], [_compact(v) for v in cy]


_REFERENCE = re.compile(r'\{\{\s*([^{}]+?)\s*\}\}')


def resolve_references(text: str, objects: Mapping[str, Any], us: UnitSystem,
                       links: Sequence[Mapping[str, Any]] | None = None) -> str:
    """{{Object Name.field}} in report text becomes the live value.

    The whole point is templates: a summary that says how the data was
    sampled fills itself in whatever project the template lands in.
    The name may be a symbolic selector — {{@basis:TimeHistory.
    sample_rate}} — resolved against the link groups, so the text
    depends on no one's naming either. A reference that cannot
    resolve — no such object, or a field the object cannot answer —
    stays visible as written, the same way an unbound block stays a
    slot instead of an error.
    """
    from ..core.report import resolve_binding

    def swap(match: re.Match) -> str:
        name, dot, field = match.group(1).rpartition('.')
        obj = (objects.get(resolve_binding(name.strip(), objects,
                                           links) or '')
               if dot else None)
        if obj is not None:
            value = _field_value(obj, field.strip(), us)
            if value is not None:
                return value
        return match.group(0)

    return _REFERENCE.sub(swap, text or '')


def _field_value(obj, field, us):
    """A named fact about an object, formatted for prose; None when
    this object cannot answer it."""
    from ..core.channel_table import ChannelTable
    from ..core.geometry import Geometry
    from ..core.photos import Photos
    from ..core.shapes import ShapeSet

    def hz(value: float) -> str:
        return f'{float(value):g} {us.label_text("frequency")}'

    if isinstance(obj, ShapeSet):
        return {
            'num_modes': lambda: f'{obj.num_shapes}',
            'min_frequency': lambda: hz(np.min(obj.frequency)),
            'max_frequency': lambda: hz(np.max(obj.frequency)),
        }.get(field, lambda: None)()
    if isinstance(obj, Geometry):
        return {
            'num_nodes': lambda: f'{len(obj.node_id)}',
            'num_elements': lambda: f'{len(obj.elem_id)}',
            'num_tracelines': lambda: f'{len(obj.traceline_id)}',
        }.get(field, lambda: None)()
    if isinstance(obj, ChannelTable):
        return f'{obj.num_channels}' if field == 'num_channels' else None
    if isinstance(obj, Photos):
        return f'{obj.num_photos}' if field == 'num_photos' else None
    if not hasattr(obj, 'abscissa'):
        return None
    # a data array: counts first, then whatever its abscissa can say
    n = len(obj.abscissa)
    channels = len({(dof, obj.known_dim(i))
                    for i, dof in enumerate(obj.response_dof)})
    counts = {'num_records': obj.num_records,
              'num_channels': channels,
              'num_averages': obj.num_records // max(channels, 1),
              'num_samples': n}
    if field in counts:
        return f'{counts[field]}'
    # how the record is to be cut up, for a summary that states the
    # settings rather than leaving them to be filled in later
    # (Brandon, 2026-08-24). A record nobody has framed answers
    # nothing, and the reference stays visible as written.
    averaging = getattr(obj, 'averaging', None)
    if averaging is not None:
        answer = {
            'num_frames': lambda: f'{averaging.frames}',
            'frame_length': lambda: f'{averaging.frame_length}',
            'window': lambda: f'{averaging.window}',
            'overlap': lambda: f'{round(averaging.overlap * 100)}%',
        }.get(field, lambda: None)()
        if answer is not None:
            return answer
    # the filter riding the record, so a caption that names the
    # corners stays true after an edge moves and the chain refreshes
    filtering = getattr(obj, 'filtering', None)
    if filtering is not None:
        answer = {
            'filter_description': lambda: filtering.describe(),
            'filter_order': lambda: f'{filtering.order}',
        }.get(field, lambda: None)()
        if answer is not None:
            return answer
    if field == 'num_shocks':
        shocks = getattr(obj, 'shocks', None)
        return f'{len(shocks)}' if shocks else None
    if n < 2:
        return None
    if obj.abscissa_dim == 'time':
        dt = float(obj.abscissa[1] - obj.abscissa[0])
        return {
            'sample_rate': lambda: hz(1.0 / dt),
            'duration': lambda: f'{n * dt:g} s',
            'frequency_resolution': lambda: hz(1.0 / (n * dt)),
        }.get(field, lambda: None)()
    if obj.abscissa_dim == 'frequency':
        return {
            'frequency_resolution': lambda: hz(
                obj.abscissa[1] - obj.abscissa[0]),
            'min_frequency': lambda: hz(np.min(obj.abscissa)),
            'max_frequency': lambda: hz(np.max(obj.abscissa)),
        }.get(field, lambda: None)()
    return None


_FIGURE = re.compile(r'\{\{\s*(figure|table)\s*:\s*([^{}]+?)\s*\}\}',
                     re.IGNORECASE)


def _resolve_figures(text, labeled):
    """{{figure:<start of caption>}} becomes 'Figure 3' — whatever
    number that figure carries right now. Same for {{table:...}}. An
    unmatched reference stays visible as written."""
    def swap(match: re.Match) -> str:
        wanted = match.group(1).lower()
        target = match.group(2).strip().lower()
        for label, caption in labeled:
            if (label.lower().startswith(wanted)
                    and caption.lower().startswith(target)):
                return label
        return match.group(0)

    return _FIGURE.sub(swap, text)


def _build_block(block, objects, us, links=None):
    from ..core.report import resolve_binding

    # symbolic bindings resolve here, once, into a working copy — every
    # builder below sees plain names, and the stored block keeps its
    # selector
    resolved = {key: (resolve_binding(block.get(key), objects, links)
                      or '')
                for key in ('source', 'geometry', 'shapes', 'dofs_source',
                            'measured', 'specification') if block.get(key)}
    block = {**block, **resolved}
    if block.get('caption'):
        # captions carry live references too — a filtered-transients
        # figure names its corner, and the number must follow the
        # setting the way the prose does. Field references only:
        # `{{figure:...}}` has no dot, so the resolver leaves it be
        # (a caption is what figures are *labeled* by; one referring
        # to another resolves at markdown time like the prose)
        block['caption'] = resolve_references(
            block['caption'], objects, us, links)
    kind = block.get('kind')
    if kind == 'text':
        # markdown waits: figure references need every label assigned
        return {'kind': 'text', 'text': resolve_references(
            block.get('text', ''), objects, us, links)}
    if kind == 'plot':
        source = objects.get(block.get('source'))
        if source is None:
            return None
        if block.get('grid'):
            geometry = objects.get(
                resolve_binding('@basis:Geometry', objects, links) or '')
            return _grid_block(block, source, objects, us, geometry)
        return _plot_block(block, source, objects, us)
    if kind == 'scene':
        geometry = objects.get(block.get('geometry'))
        if geometry is None:
            return None
        shapes = objects.get(block.get('shapes')) \
            if block.get('shapes') else None
        return _scene_block(block, geometry, shapes, us, objects)
    if kind == 'table':
        source = objects.get(block.get('source'))
        if source is None:
            return None
        # the basis geometry brings the channel table its direction
        # columns, as it does in the window
        geometry = objects.get(
            resolve_binding('@basis:Geometry', objects, links) or '')
        return _table_block(block, source, geometry)
    if kind == 'photo':
        source = objects.get(block.get('source'))
        if source is None:
            return None
        return _photo_block(block, source)
    if kind == 'bars':
        source = objects.get(block.get('source'))
        if block.get('mode') == 'sine':
            from ..core.sine import (
                SineLevel,
                SineLevelSet,
                SineSweepSpecification,
            )
            if not isinstance(source, SineSweepSpecification):
                return None
            # every level in the project, not one binding: the chart
            # exists to say where the whole test stands. Grouped sets
            # and loose levels both count, so older projects keep
            # their bars
            levels = []
            for obj in objects.values():
                if isinstance(obj, SineLevelSet):
                    levels.extend(obj.levels)
                elif isinstance(obj, SineLevel):
                    levels.append(obj)
            return _sine_bars(block, source, levels)
        if block.get('mode') == 'kurtosis':
            return _kurtosis_bars(block, source)
        measured = objects.get(block.get('measured'))
        if source is None or measured is None:
            return None
        return _bars_block(block, source, measured, us)
    if kind == 'pairs':
        return _pairs_block(block, objects)
    if kind == 'overlay':
        return _overlay_block(block, objects, us, links)
    if kind == 'verdict':
        source = objects.get(block.get('source'))
        measured = objects.get(block.get('measured'))
        if source is None or measured is None:
            return None
        return _verdict_block(source, measured)
    return None


def _verdict_block(specification, measured):
    """The pass/fail box: `compliance.verdict` over the comparison of
    `measured` against `specification`, the scale resolved once for
    the pair the way the bar charts resolve it. None when the two
    have no channel in common or cannot be compared."""
    from ..core.compliance import compare_all, comparison_scale_db, verdict

    rows = compare_all(specification, measured,
                       scale_db=comparison_scale_db(specification, measured))
    read = verdict(rows)
    if read['passed'] is None:
        return None
    return {'kind': 'verdict', 'passed': bool(read['passed']),
            'channels': int(read['channels']),
            'lines_percent': _compact(read['lines_percent']),
            'rms_percent': _compact(read['rms_percent']),
            'lines_limit': float(read['lines_limit']),
            'rms_limit': float(read['rms_limit']),
            'lines_fail_percent': float(read['lines_fail_percent']),
            'rms_fail_percent': float(read['rms_fail_percent'])}


def _pairs_block(block, objects):
    """The matched-modes table: both sets' parameters side by side
    with the frequency error against the first set and the MAC of each
    pair. The block binds a MatchedModes object by name."""
    from ..core.matches import MatchedModes
    from ..core.shapes import scale_ratios

    bound = objects.get(block.get('source'))
    if not isinstance(bound, MatchedModes):
        return None
    source = objects.get(bound.first)
    other = objects.get(bound.second)
    pairs, macs = bound.pairs, bound.macs
    if source is None or other is None:
        return None
    # the scale each pair was normalized by. The overlay figure below
    # this table draws both shapes to their own peak, so it cannot show
    # one set being thirty times the other; this column is where that
    # shows, and 1.00 down the column is the reader's evidence that the
    # comparison is of shape alone.
    ratios = scale_ratios(source, other, pairs)
    rows = []
    for index, (row, column) in enumerate(pairs):
        if not (0 <= row < source.num_shapes
                and 0 <= column < other.num_shapes):
            continue
        fa = float(source.frequency[row])
        fb = float(other.frequency[column])
        delta = f'{(fb - fa) / fa * 100.0:+.2f}' if fa else '—'
        mac = macs[index]
        rows.append([
            str(row + 1), f'{fa:.4f}',
            f'{float(source.damping[row]) * 100:.3f}',
            str(column + 1), f'{fb:.4f}',
            f'{float(other.damping[column]) * 100:.3f}',
            delta, f'{mac:.3f}',
            '—' if ratios[index] is None else f'{ratios[index]:.2f}'])
    if not rows:
        return None
    a_name, b_name = bound.first, bound.second
    return {'kind': 'table', 'caption': block.get('caption', ''),
            'headers': [f'{a_name} Mode', 'Freq [Hz]',
                        'Damping [%]', f'{b_name} Mode',
                        'Freq [Hz]', 'Damping [%]', 'Δf [%]', 'MAC',
                        f'{b_name}/{a_name}'],
            'rows': rows}


def _photo_block(block, photos):
    """The photo as a data: URI — the image rides inside the file, like
    everything else in the deliverable."""
    import base64

    from ..core.photos import Photos

    name = block.get('photo', '')
    if not isinstance(photos, Photos) or name not in photos.names:
        return None
    i = photos.names.index(name)
    encoded = base64.b64encode(photos.images[i]).decode('ascii')
    return {'kind': 'image', 'caption': block.get('caption', ''),
            'src': f'data:image/{photos.formats[i]};base64,{encoded}'}


def _hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*tuple(round(255 * v) for v in rgb))


def _stage_budget(records: int) -> int:
    """Points per record for a stage figure of `records` of them.

    The figure's whole budget shared out, so a handful of channels
    draw sample for sample and only a very long or very wide set is
    thinned at all (Brandon, 2026-08-24: a report should not reduce
    the data it is reporting). Never below a floor — a curve of two
    hundred points is not a trace of anything.
    """
    return max(MAX_FIGURE_POINTS // max(records, 1), 512)


def _stage_page(block, source, objects, us, caption, page):
    """One stage block as the figures it needs — the app's own pages.

    A stage holds `MAX_STAGE_RECORDS` records legibly, and an object
    with more of them continues into further figures rather than
    being truncated at one (Brandon, 2026-08-24: there is no reason
    to cap the figure count). The pages are `waterfall_arrays`' own,
    so the report's continuation figures *are* the pages the app
    steps through on its bar.

    What is capped is the total drawn, and for a reason that is not
    taste: the geometry rides inside the self-contained HTML at about
    three quarters of a megabyte a figure, so the 25,380-record
    stress set would ask for 529 of them and a 399 MB file that no
    browser opens and the report editor cannot load.
    `MAX_STAGE_TOTAL` is set well above any real survey block and
    well below that, and whatever it leaves out is said in the last
    caption.

    Answers ``(figure, pages, held)`` for one page — the built
    figure, how many pages the object needs, and how many records the
    drawn quantity group holds — or ``(None, 0, 0)`` when there is
    nothing to draw.
    """
    # the app's 3-D reading, in the document: every record on the
    # stage, receding, colored by level (Brandon, 2026-08-23).
    # Many channels on one 2-D axis hide each other exactly where
    # it matters — the tenth curve lands on the first nine — and
    # a system ID's whole plant is thirty-two of them.
    #
    # With a `floor` it is the paired stage instead: the two
    # objects met station by station, overlaid or divided, which
    # is the same choice the app's bar offers over two PSDs.
    from ..viz.waterfall import (
        STAGE,
        stage_basis,
        stage_curves,
        waterfall_arrays,
        waterfall_group,
    )

    floor_object = objects.get(block.get('floor', '') or '')
    source = _positive_lines(source)
    if floor_object is not None:
        floor_object = _positive_lines(floor_object)
    reading = block.get('reading', 'overlay')
    note = ''
    if floor_object is not None:
        from ..core.report import base_quantity
        from ..viz.paired import paired_stage_curves
        from ..viz.waterfall import waterfall_groups

        # a paired stage picks its quantity by group, not by
        # filtering records: the stations are what the two objects
        # share, so both sides have to be asked the same question.
        # The block names the quantity in the interface's own word
        # ('acceleration'), which resolves here to the
        # (dimension, hint) key the stage groups by.
        wanted = block.get('quantity', '')
        group_key = None
        if wanted:
            for key, _members in waterfall_groups(source, None):
                if base_quantity(key[0]) == wanted:
                    group_key = key
                    break
            if group_key is None:
                return None, 0, 0   # nothing here measures that
        try:
            staged = paired_stage_curves(
                source, floor_object, mode=reading, unit_system=us,
                budget=_stage_budget(len(
                    getattr(source, 'response_dof', []) or [1])),
                quantity=group_key, page=page)
        except ValueError:
            return None, 0, 0   # no shared channel, or mismatched lines
        log = reading != 'ratio' and staged['log_scaled']
        labels = [run['label'] for run in staged['runs']]
        stations = int(staged['drawn'])
        pages, held = int(staged['pages']), stations
        if staged.get('unmatched'):
            note = (f'{staged["unmatched"]} records are held by only '
                    'one of the two and are not paired.')
        if reading != 'ratio' and not any(
                run['quiet'] for run in staged['runs']):
            # the same silence the flat figure explains: an
            # ambient channel recorded with its shaker off is
            # exactly zero, and zero has no place on a log axis
            note = (note + ' ' if note else '') + (
                'Nothing to draw for the quieter side here, which '
                'recorded exactly zero.')
    else:
        records = list(range(source.num_records))
        select = block.get('select', '')
        if select == 'drive' and source.reference_dof is not None:
            records = [i for i in records
                       if source.response_dof[i]
                       == source.reference_dof[i]]
        elif select.startswith('dim:'):
            from ..core.report import base_quantity

            records = [i for i in records
                       if base_quantity(source.known_dim(i))
                       == select[4:]]
        if not records:
            return None, 0, 0
        # group first, then page the group — never the other way
        # round. The stage draws one quantity group, and a cap taken
        # before the grouping spends its places on records that were
        # never going to be drawn (a 100-record mixed FRF came back
        # with 26 of them). Paged here rather than inside
        # `waterfall_arrays` because its own pages are a *vertex*
        # budget — right for the app, where the ceiling is memory,
        # wrong for a document, where 525 short records would land on
        # one stage as an unreadable wall of stations.
        chosen = waterfall_group(source, records, None)[0]
        held = len(chosen)
        pages = max(1, -(-held // MAX_STAGE_RECORDS))
        if page >= pages:
            return None, pages, held
        drawing = chosen[page * MAX_STAGE_RECORDS:
                         (page + 1) * MAX_STAGE_RECORDS]
        per_record = _stage_budget(len(drawing))
        arrays = waterfall_arrays(
            source, drawing, us, block.get('component', 'magnitude'),
            budget=per_record,
            # the records are already counted out above, so the scene
            # budget only has to be wide enough not to page them again
            scene_budget=MAX_FIGURE_POINTS * 4)
        staged = stage_curves(arrays, source.ordinate_limits)
        log = arrays['log_scaled']
        labels = list(arrays['labels'])
        stations = staged['stations']
        # what the *object* holds outside the drawn quantity, from
        # the whole record list rather than this page's slice
        left_out = len(waterfall_group(source, records, None)[1])
        if left_out:
            # one vertical axis holds one quantity — the app
            # offers a box to pick the others, and a document has
            # to say they are there
            caption = (caption + f' — {left_out} records of other '
                       'quantities are not shown').strip()
    if not staged['runs']:
        return None, 0, 0
    # Thinned, and said (Brandon, 2026-08-24). Measured off the drawing
    # rather than read off the budget, because the budget lies in two
    # directions at once: a stepped spectrum is never thinned at all —
    # cutting a staircase destroys the treads and the area under them —
    # and it draws *two* points a line, so a budget under the line
    # count means nothing there. Asking the budget claimed "512 points
    # of 1025" for a spectrum drawn whole in 2050. A curve also breaks
    # into a run per gap, so the points are summed per station before
    # the widest is taken.
    # a station is a *fractional* depth (0.0, 0.3, 0.6 …), so it keys
    # as itself — `int()` rounded four curves into one bucket and
    # summed their points, which read as four times the draw
    per_station: dict[float, int] = {}
    for run in staged['runs']:
        station = float(run['station'])
        per_station[station] = per_station.get(station, 0) + len(run['points'])
    whole = len(source.abscissa) * (
        2 if drawing_shape(source, False) == 'steps' else 1)
    thinning = _thinned_note(max(per_station.values(), default=0), whole)
    if thinning:
        note = (note + ' ' if note else '') + thinning
    x0, x1, z0, z1 = staged['extents']
    # levels normalized to the stage's own range, so the color
    # scale and the vertical axis are one scale — the app pins
    # `clim` to exactly this pair for the same reason
    span = (z1 - z0) or 1.0
    # (x, z) per point with the station carried once: a run stands
    # at one depth by construction, and a third coordinate
    # repeated seven hundred times is a third of the figure's
    # weight in the file for nothing. Three decimals on a stage
    # 1.6 wide is finer than a pixel of the canvas it draws on.
    runs = []
    for position, run in enumerate(staged['runs']):
        built_run = {
            'record': run.get('record', position),
            'station': round(float(run['station']), 4),
            'xz': [[round(float(point[0]), 3),
                    round(float(point[2]), 3)]
                   for point in run['points']],
            'levels': [round((float(v) - z0) / span, 3)
                       for v in run['levels']]}
        if run.get('quiet'):
            # the stood-back object: gray, not on the color
            # scale, exactly as the app stands it back
            built_run['quiet'] = True
            del built_run['levels']
        runs.append(built_run)
    # the marks the flat figure carries, given depth: the
    # analyzed span as a slab across every channel, and the rail
    # of window weights on the back wall — the app's own stage
    # marks, from `viz.marks`' shared geometry, so the two views
    # cannot disagree about where a frame is
    marks = {}
    if floor_object is None:
        from ..viz.marks import (
            averaging_stage_geometry,
            shock_stage_geometry,
        )

        averaging = getattr(source, 'averaging', None)
        if averaging is not None:
            try:
                rate = source.sample_rate
            except ValueError:
                rate = None     # unevenly sampled: no frames to draw
            if rate is not None:
                marks['averaging'] = averaging_stage_geometry(
                    averaging, rate, staged['extents'],
                    float(source.abscissa[0]))
                marks['averaging']['label'] = (
                    f'{averaging.frames} x {averaging.frame_length} '
                    f'samples, {averaging.window}, '
                    f'{round(averaging.overlap * 100)}% overlap')
        if getattr(source, 'shocks', None):
            marks['shocks'] = shock_stage_geometry(
                source.shocks, staged['extents'], float(source.abscissa[0]))
    right, up = stage_basis()
    # the canvas prints real values at decade ticks where VTK prints
    # the exponents it was given, so the 'log10' honesty prefix comes
    # off here and the flag goes on instead
    logx = bool(staged.get('log_abscissa', False))
    xlabel = staged['xlabel']
    if logx:
        xlabel = xlabel.removeprefix('log10 ')
    built = {'kind': 'stage', 'caption': caption,
             'runs': runs, 'labels': labels,
             'stations': stations,
             'extents': [float(x0), float(x1), float(z0), float(z1)],
             'log': bool(log), 'logx': logx,
             'xlabel': xlabel, 'zlabel': staged['zlabel'],
             'stage': [_compact(v) for v in STAGE],
             # the app's own opening view of data, derived from the
             # camera `place_camera` sets — the figure opens where
             # the reader last saw this object on screen
             'home': [[_compact(v) for v in right],
                      [_compact(v) for v in up]]}
    if marks:
        built['marks'] = marks
    if note:
        built['note'] = note
    return built, pages, held


def _scalogram_figure(block, source, us, caption):
    """One channel's scalogram as a stage figure: time across,
    frequency receding, amplitude up and in color.

    The app's wavelet reading, in the document (Brandon, 2026-08-28:
    the transient report should carry the 3-D plot). One channel, as
    the view shows one: the block's `channel` names a DOF — the
    editor offers the choice as a drop-down — and without one the
    first channel the `select` admits is drawn, the panel's own
    default. The rows ride a `sheet` payload the page fills as a
    surface, cell by cell on the level colormap, stationed by log
    frequency with the decade rows labeled in Hz — a surface, not a
    fan of lines, because that is what the app's own stage draws
    (Brandon, 2026-08-29).

    Time is thinned to `SCALOGRAM_COLUMNS` by peak-hold — the largest
    magnitude in each bin — because a scalogram's story is its ridges
    and a stride would land between them; the thinning is said in the
    note like the stage's own. The cone of influence is drawn the way
    the app's stage draws it — two translucent walls standing where
    the record's ends reach in (Brandon, 2026-08-29: the two views
    disagreed, and the walls are the reading he chose) — because
    inside the cone the picture is shaped by where the record was
    cut rather than by the event.
    """
    import numpy as np

    from ..core import wavelet
    from ..viz.waterfall import STAGE, stage_basis

    try:
        rate = source.sample_rate
    except (ValueError, AttributeError):
        return None
    candidates = _scalogram_candidates(block, source)
    named = block.get('channel')
    if named:
        candidates = [i for i in candidates
                      if str(source.response_dof[i]) == str(named)]
    if not candidates:
        return None
    channel = candidates[0]
    dims = list(np.atleast_1d(source.ordinate_dim))

    values = np.real(np.asarray(source.ordinate)[channel])
    duration = len(values) / rate
    low, high = wavelet.default_range(rate, duration)
    frequencies = wavelet.log_frequencies(low, high)
    frequencies = frequencies[frequencies < rate / 2.0]
    if frequencies.size < 2:
        return None
    # peak-held down to the columns a figure can carry, by the reading
    # the app's view draws from: the ridges are the story, and a stride
    # would land between the very samples a transient's energy lives in
    times, magnitude = wavelet.scalogram_peaks(
        values, rate, frequencies, columns=SCALOGRAM_COLUMNS)
    note = ''
    if magnitude.shape[1] < len(values):
        note = (f'time thinned to {magnitude.shape[1]} columns of '
                f'{len(values)}, keeping each bin\'s peak')

    sx, sy, _sz = STAGE
    z1 = float(magnitude.max()) or 1.0
    rows = np.log10(frequencies)
    span = float(rows[-1] - rows[0]) or 1.0
    labels, stations = [], []
    ticks = _decade_values(frequencies)
    for k in range(len(frequencies)):
        station = float((rows[k] - rows[0]) / span)
        labels.append(f'{ticks[k]:g} Hz' if k in ticks else '')
        stations.append(round(station * sy, 4))
    # the cone of influence, per row, as the share of the time axis
    # each end of the record reaches into at that row's frequency
    reach = wavelet.cone_of_influence(frequencies, rate)
    sheet = {
        'xs': [round(float(t) / duration * sx, 4) for t in times],
        'stations': stations,
        'levels': [[round(float(v) / z1, 3) for v in magnitude[k]]
                   for k in range(len(frequencies))],
        'cone': [round(float(c) / duration * sx, 4) for c in reach]}
    right, up = stage_basis()

    unit = None
    units = getattr(source, 'ordinate_unit', None)
    if units is not None:
        unit = list(np.atleast_1d(units))[min(channel, len(
            list(np.atleast_1d(units))) - 1)]
    zlabel = str(dims[min(channel, len(dims) - 1)])
    if unit:
        zlabel = f'{zlabel} [{unit}]'
    cone = float(wavelet.cone_of_influence([frequencies[0]], rate)[0])
    built = {'kind': 'stage',
             'caption': (f'{caption} — {source.record_label(channel)}, '
                         f'{frequencies[0]:.3g} to {frequencies[-1]:.3g} Hz'
                         f'; the walls mark where the record\'s ends '
                         f'reach in, {cone:.2g} s at the lowest '
                         f'row').strip(),
             'sheet': sheet, 'labels': labels, 'stations': stations,
             'extents': [0.0, float(duration), 0.0, z1],
             'log': False, 'logx': False,
             'xlabel': f'time [{us.label_text("time")}]',
             'zlabel': zlabel,
             'stage': [_compact(v) for v in STAGE],
             'home': [[_compact(v) for v in right], [_compact(v) for v in up]]}
    if note:
        built['note'] = note
    return built


def _scalogram_candidates(block, source):
    """Record indices the block's `select` admits — the one list the
    figure draws from and the editor's channel drop-down offers, so
    the two cannot disagree."""
    import numpy as np

    from ..core.report import base_quantity

    select = block.get('select', '')
    wanted_dim = select[4:] if select.startswith('dim:') else None
    dims = list(np.atleast_1d(source.ordinate_dim))
    return [i for i in range(len(source.response_dof))
            if wanted_dim is None
            or base_quantity(str(dims[min(i, len(dims) - 1)]))
            == wanted_dim]


def scalogram_channel_options(block, objects, links=()):
    """The DOF names a scalogram block may draw — for the editor's
    drop-down (Brandon, 2026-08-29: the figure shows one channel, so
    the reader chooses which).

    Parameters
    ----------
    block : dict
        The scalogram plot block.
    objects : mapping
        The report's objects, name to object.
    links : sequence, optional
        The project's link groups, for symbolic source bindings.

    Returns
    -------
    list of str
        The response DOFs the block's `select` admits.
    """
    from ..core.report import resolve_binding

    name = resolve_binding(block.get('source', ''), objects, links)
    source = objects.get(name)
    if source is None or not hasattr(source, 'response_dof'):
        return []
    return [str(source.response_dof[i])
            for i in _scalogram_candidates(block, source)]


def _decade_values(frequencies):
    """{row index: tick value}: 1, 2, 5 per decade, at the nearest row.

    The rows are twelfths of an octave and nobody reads 158.7 Hz off
    an axis — the same choice the app\'s 3-D scalogram makes. The
    label carries the round value, not the row\'s own frequency: a
    row within a twelfth of an octave of 5 Hz labeled "5.04 Hz"
    reads as a measurement where it is a ruling.
    """
    import numpy as np

    from ..core import wavelet

    return {int(np.argmin(np.abs(frequencies - value))): value
            for value in wavelet.decade_values(frequencies[0],
                                               frequencies[-1])}


def _stage_figures(block, source, objects, us, caption):
    """One stage block as the figures it needs.

    A stage holds `MAX_STAGE_RECORDS` records legibly, and an object
    with more of them continues into further figures rather than
    being truncated at one (Brandon, 2026-08-24: there is no reason
    to cap the figure count). The continuation figures are
    `waterfall_arrays`' own pages — the very pages the app steps
    through on its bar — so the document shows what the screen shows,
    all of it.

    What *is* bounded is the total drawn, for a reason that is not
    taste: the geometry rides inside the self-contained HTML at about
    three quarters of a megabyte a figure, so the 25,380-record
    stress set would ask for 529 of them and a 399 MB file that no
    browser opens and the report editor cannot load.
    `MAX_STAGE_TOTAL` sits well above any real survey block (the
    largest here is 560 records) and well below that; what it leaves
    out is said in the last caption.

    Returns the list of built figures, or None when there is nothing
    to draw at all.
    """
    figures = []
    page = 0
    while True:
        built, pages, held = _stage_page(block, source, objects, us,
                                         caption, page)
        if built is None:
            break
        figures.append(built)
        page += 1
        drawn = sum(len(f['labels']) for f in figures)
        if page >= pages:
            break
        if drawn >= MAX_STAGE_TOTAL:
            figures[-1]['caption'] = (
                figures[-1]['caption']
                + f' — {held - drawn} further records are not drawn; '
                  'this figure would not fit in the document'
            ).strip()
            break
    if not figures:
        return None
    if len(figures) > 1:
        # each figure says which stretch of the channels it holds, or
        # a reader has no way to tell one continuation from the next
        at = 0
        for figure in figures:
            count = len(figure['labels'])
            figure['caption'] = (
                f'{figure["caption"]} — records {at + 1} to '
                f'{at + count} of {held}').strip()
            at += count
    return figures


def _grid_block(block, source, objects, us, geometry=None):
    """Many control channels as one figure: a grid, a row per node and
    a column per direction (`core.report.channel_grid`), each cell the
    channel's own figure — the very block `_plot_block` draws for a
    block naming that channel, so a cell reads exactly as the single
    figure would (Brandon, 2026-09-19). The columns are the global
    axes when the basis geometry can place the channels, the DOF's own
    letters when it cannot."""
    from ..core.report import channel_grid, control_channels_of

    bounded = (objects.get(block['specification'])
               if block.get('specification') else source)
    labels = control_channels_of(bounded)
    if not labels:
        return None
    layout = channel_grid(labels, geometry)
    rows, drawn_any = [], False
    for row in layout['rows']:
        cells = []
        for column in row['cells']:
            drawn = []
            for entry in column:
                cell = _plot_block({**block, 'grid': False,
                                    'channel': entry['channel'],
                                    'caption': entry['channel']},
                                   source, objects, us)
                if cell is None:
                    continue
                if entry['note']:
                    cell['note'] = entry['note']
                drawn.append(cell)
                drawn_any = True
            cells.append(drawn)
        rows.append({'label': row['label'], 'cells': cells})
    if not drawn_any:
        return None
    return {'kind': 'grid', 'caption': block.get('caption', ''),
            'columns': layout['columns'], 'rows': rows}


def _plot_block(block, source, objects, us):
    from ..core.shapes import ShapeSet
    from ..core.sine import SineLevelSet

    if isinstance(source, SineLevelSet):
        # the set is grouping; a figure shows one tone's level, named
        # by the block, so the template emits one block per tone
        try:
            source = source.tone(block.get('tone', ''))
        except KeyError:
            return None
    mode = block.get('mode', 'curves')
    caption = block.get('caption', '')
    if mode == 'overlay':
        against = objects.get(block.get('specification')) \
            if block.get('specification') else None
        return _replication_overlay_block(block, source, against, us)
    if mode == 'mac':
        if not isinstance(source, ShapeSet):
            return None
        if block.get('shapes'):
            # a second set makes it the cross-MAC — test against the
            # projected finite element model, DOF-name aligned. A named
            # partner that is not here yet keeps this an unbound slot,
            # never a silent fall-back to the auto-MAC.
            from ..core.shapes import cross_mac

            other = objects.get(block.get('shapes'))
            if other is None:
                return None
            try:
                matrix = cross_mac(source, other)
            except ValueError:
                return None     # no shared DOFs: nothing to compare
            columns = [f'{float(f):.1f}' for f in other.frequency]
        else:
            matrix = source.auto_mac()
            columns = None
        labels = [f'{float(f):.1f}' for f in source.frequency]
        rows, columns = labels, columns or labels
        # the committed matches checker their squares, exactly as the
        # comparison screen marks them — found by the pair of set
        # names, in either order
        pairs = []
        if block.get('shapes'):
            from ..core.matches import MatchedModes
            for matched in objects.values():
                if not isinstance(matched, MatchedModes):
                    continue
                ends = (matched.first, matched.second)
                if ends == (block['source'], block['shapes']):
                    pairs = [list(pair) for pair in matched.pairs]
                    break
                if ends == (block['shapes'], block['source']):
                    pairs = [[c, r] for r, c in matched.pairs]
                    break
        # a cross-MAC reads tall: whichever set has more modes makes
        # the rows, matching the GUI's own orientation
        if len(columns) > len(rows):
            matrix = np.asarray(matrix).T
            rows, columns = columns, rows
            pairs = [[c, r] for r, c in pairs]
        pairs = [[r, c] for r, c in pairs
                 if 0 <= r < len(rows) and 0 <= c < len(columns)]
        return {'kind': 'mac', 'caption': caption,
                'rows': rows, 'columns': columns, 'pairs': pairs,
                'matrix': [[_compact(v) for v in row] for row in matrix]}
    if mode == 'stage':
        return _stage_figures(block, source, objects, us, caption)
    if mode == 'scalogram':
        return _scalogram_figure(block, source, us, caption)
    if mode == 'pair':
        # the two densities overlaid, one figure per quantity — the
        # flat reading of the app's paired stage (Brandon,
        # 2026-08-23). The pair is named the way the ratio names it,
        # and pairs its channels through the same rule, so this
        # figure and the ratio below it cannot disagree about which
        # channel is which.
        from ..core.data import paired_channels
        from ..core.report import base_quantity

        floor_object = objects.get(block.get('floor', '') or '')
        if floor_object is None:
            return None
        try:
            shared = paired_channels(source, floor_object)
        except (ValueError, AttributeError):
            return None
        select = block.get('select', '')
        if select.startswith('dim:'):
            wanted_dim = select[4:]
            shared = [row for row in shared
                      if base_quantity(row[3]) == wanted_dim]
        if not shared:
            return None
        shared = shared[:MAX_REPORT_CURVES // 2]
        x = np.asarray(source.display_abscissa(us), dtype=float)
        loud = np.asarray(source.display_ordinate(
            us, [i for i, _j, _d, _q in shared])).real
        quiet = np.asarray(floor_object.display_ordinate(
            us, [j for _i, j, _d, _q in shared])).real
        logy = source.log_scaled()

        thinned = [0, 0]

        def curve(values, position, label, dashed):
            """One channel's density, or None when it has nothing to
            draw — an ambient channel recorded while its shaker was
            off is exactly zero, and zero has no place on a log axis.
            A legend entry for a curve that draws nothing is a claim
            the figure does not support."""
            y = np.asarray(values[position], dtype=float)
            if logy:
                with np.errstate(divide='ignore', invalid='ignore'):
                    y = np.where(y > 0.0,
                                 np.log10(np.maximum(y, 1e-300)), np.nan)
            if not np.isfinite(y).any():
                return None
            cx, cy = _decimate(x, y)
            thinned[:] = [max(thinned[0], len(x)), max(thinned[1], len(cx))]
            built_curve = {'label': label, 'color': position,
                           'x': (cx if len(cx) != len(x) else None),
                           'y': _finite(cy)}
            if dashed:
                built_curve['dash'] = True
            return built_curve

        # the quieter set follows the louder one the way every
        # follower curve here does: the same color as the channel it
        # belongs to, dashed — so a channel and its own noise floor
        # read as one pair rather than as two unrelated curves
        loud_name = block.get('source', '')
        quiet_name = block.get('floor', '')
        curves, quiet_drawn = [], 0
        for position, (_i, _j, dof, _q) in enumerate(shared):
            for values, label, dashed in (
                    (loud, f'{loud_name}: {dof}', False),
                    (quiet, f'{quiet_name}: {dof}', True)):
                one = curve(values, position, label, dashed)
                if one is None:
                    continue
                curves.append(one)
                quiet_drawn += dashed
        if not curves:
            return None
        if not quiet_drawn:
            # the reader is promised a pair and is getting one side:
            # say why rather than leaving the missing half unexplained
            caption = (caption + ' — nothing to draw for the quieter '
                       'side on these channels, which recorded exactly '
                       'zero').strip()
        built = {'kind': 'plot', 'caption': caption, 'logy': bool(logy),
                 'x': [_compact(v) for v in x],
                 'xlabel': f'frequency [{us.label_text("frequency")}]',
                 'ylabel': _axis_text(source, us, shared[0][0]),
                 'curves': curves}
        thinning = _thinned_note(thinned[1], thinned[0])
        if thinning:
            built['note'] = thinning
        return built
    if mode == 'ratio':
        # two densities divided, in decibels — the signal-to-noise
        # reading (Brandon, 2026-08-25). The louder is the numerator;
        # the block names the pair, and either missing keeps this an
        # unbound slot rather than a half of a ratio.
        from ..core.data import density_ratio

        floor_object = objects.get(block.get('floor', '') or '')
        if floor_object is None:
            return None
        try:
            x, rows, dofs, _dims = density_ratio(source, floor_object)
        except ValueError:
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            decibels = 10.0 * np.log10(np.real(rows))
        curves = [{'label': dof, 'color': k, 'y': _finite(decibels[k])}
                  for k, dof in enumerate(dofs)
                  if np.isfinite(decibels[k]).any()]
        if not curves:
            return None
        return {'kind': 'plot', 'caption': caption, 'logy': False,
                'x': [_compact(v) for v in x],
                'xlabel': f'frequency [{us.label_text("frequency")}]',
                'ylabel': 'ratio [dB]', 'curves': curves}
    if mode == 'cmif':
        from ..plot import cmif_curves

        singular, x = cmif_curves(source, None, us)
        curves = [{'label': f'CMIF {k + 1}', 'color': k,
                   'y': _finite(np.log10(np.maximum(singular[k], 1e-300)))}
                  for k in range(singular.shape[0])]
        shapes = objects.get(block.get('shapes')) \
            if block.get('shapes') else None
        built = {'kind': 'plot', 'caption': caption, 'logy': True,
                 'x': [_compact(v) for v in x],
                 'xlabel': f'frequency [{us.label_text("frequency")}]',
                 'ylabel': _axis_text(source, us), 'curves': curves}
        if shapes is not None:
            synthesized = _synthesized_cmif(source, shapes, us)
            built['curves'] = curves + [
                {'label': f'Synthesized CMIF {k + 1}', 'color': k,
                 'dash': True,
                 'y': _finite(np.log10(np.maximum(synthesized[k], 1e-300)))}
                for k in range(synthesized.shape[0])]
            # open on the mode band: 80% of the lowest resynthesized
            # frequency to 120% of the highest (frequency is Hz in
            # every unit system, so this is already display units)
            freqs = np.asarray(shapes.frequency, dtype=float)
            if freqs.size:
                built['home_x'] = [0.8 * float(freqs.min()),
                                   1.2 * float(freqs.max())]
                # the fitting screen's gray bookmarks, in the report
                # too; coincident frequencies collapse to one line
                built['marks'] = list(
                    dict.fromkeys(float(f) for f in freqs))
        return built
    if mode == 'map':
        # the coherence map: frequency across, channel down, the color
        # scale pinned to 0..1 exactly as the GUI pins it
        x = np.asarray(source.display_abscissa(us), dtype=float)
        records = list(range(source.num_records))
        values = np.asarray(source.display_ordinate(us, records)).real
        # a map is read as color on a ~900px canvas: column count and
        # value precision beyond that only bloat the file
        stride = max(1, len(x) // MAX_MAP_COLUMNS)
        values = np.round(np.clip(values[:, ::stride], 0.0, 1.0), 3)
        built_map = {'kind': 'map', 'caption': caption,
                'x': [round(float(v), 3) for v in x[::stride]],
                'xlabel': f'frequency [{us.label_text("frequency")}]',
                'zlabel': 'coherence',
                'labels': [source.record_label(i) for i in records],
                'rows': [[_compact(v) for v in row] for row in values]}
        if stride > 1:
            # a map strides its columns where a curve keeps extremes:
            # a color field is read across, so the honest note names
            # what it is rather than borrowing the curve's sentence
            built_map['note'] = (
                f'Thinned for the file: every {stride}th frequency '
                f'line drawn of {len(x)}.')
        return built_map
    # curves: the records themselves, filtered the way the GUI's own
    # buttons filter, capped the way the screen caps
    select = block.get('select', '')
    indices = list(range(source.num_records))
    if select == 'drive' and source.reference_dof is not None:
        indices = [i for i in indices
                   if source.response_dof[i] == source.reference_dof[i]]
    elif select.startswith('dim:'):
        from ..core.report import base_quantity

        dimension = select[4:]
        indices = [i for i in indices
                   if base_quantity(source.known_dim(i)) == dimension]
    if not indices:
        return None     # a filter that keeps nothing has nothing to show
    from ..core.data import Specification

    # a plot naming a specification is the comparison: the response is
    # the source, the target and its zones come from the other one
    against = objects.get(block.get('specification')) \
        if block.get('specification') else None
    from ..core.data import ShockSpecification
    from ..core.sine import SineLevel, SineLevelSet, SineSweepSpecification

    if (against is not None
            and isinstance(against, SineSweepSpecification)):
        # the sine pair rides the SRS comparison machinery whole: the
        # tone's requirement, derived as a Bounded curve, is the
        # specification and the extracted level the measurement — one
        # figure per tone, because tones sweep different frequencies
        # and would draw different requirements over each other
        if not isinstance(source, SineLevel):
            return None
        try:
            target = against.tone_curve(source.tone)
        except KeyError:
            return None       # a level whose tone the spec never named
        return _srs_comparison_block(block, source, target, us, caption)
    if against is not None and isinstance(against, ShockSpecification):
        # the shock pair: not the PSD comparison, which bands, scales
        # and marks — an SRS is unscaled on principle (a spectrum
        # sitting 12 dB low must not be rescaled into agreement) and
        # its events are overlaid, so the marks would not say whose
        # line went out
        return _srs_comparison_block(block, source, against, us, caption)
    if against is not None and isinstance(against, Specification):
        # the scale is resolved on the measurement itself, before any
        # banding: the report bands the narrowband PSD for its octave
        # figures, and detecting again on the banded copy rounded to a
        # different decibel — one report, two claimed scalings
        from ..core.compliance import comparison_scale_db

        scale_db = comparison_scale_db(against, source)
        banded, against = _banded_pair(source, against, block)
        if banded is None:
            return None
        return _comparison_block(block, banded, against, us, caption,
                                 scale_db=scale_db)

    from ..core.data import Bounded, TransientSpecification

    # Bounded, not Specification: a shock specification is `Bounded,
    # Srs` — same four limit curves, different spectrum underneath —
    # and gating on the PSD kind left the shock report's Figure 2
    # drawing the bare target with its tolerance band silently dropped
    bounded = isinstance(source, Bounded) and source.has_limits
    # a transient specification reads one channel at a time too: its
    # waveforms overlaid are a thicket with no comparison in them
    paged = bounded or (isinstance(source, TransientSpecification)
                        and len(indices) > 1)
    channels = list(indices) if paged else []
    if paged and block.get('channel'):
        # a block that names its channel is that channel's figure and
        # carries no others: the random report writes one per control
        # channel, with no drop-down anywhere in it (Brandon, 2026-09-18)
        named = str(block['channel'])
        channels = [i for i in channels if _channel_label(source, i) == named]
        if not channels:
            return None
        indices = channels
    if paged:
        # One channel drawn — with its limits shaded around it when it
        # has any — the same reading the app gives. Six targets stacked
        # together are unreadable, so the rest are carried in the
        # payload and reached with the drop-down instead.
        chosen = block.get('record')
        indices = [indices[0] if chosen is None else int(chosen)]
    from ..core.data import TimeHistory

    # what this block draws from, before any cap: a figure filtered to
    # one quantity is a figure of that quantity's channels, and
    # counting the object's other records told a reader that 6 of 30
    # records were dropped from a figure that drew all 24 of its own
    available = len(indices)
    budget = None
    if isinstance(source, TimeHistory) and not paged:
        # a time history of many channels continues into further
        # figures rather than stopping at the first two dozen — the
        # stage's paging, in 2-D — and each figure shares one point
        # budget among its curves
        pages = max(1, -(-available // MAX_REPORT_CURVES))
        if pages > 1 and 'page' not in block:
            return [built for k in range(pages)
                    for built in [_plot_block({**block, 'page': k},
                                              source, objects, us)]
                    if built is not None]
        page = int(block.get('page', 0))
        first = page * MAX_REPORT_CURVES
        indices = indices[first:first + MAX_REPORT_CURVES]
        if pages > 1:
            caption = (f'{caption} — channels {first + 1}–'
                       f'{first + len(indices)} of {available}')
        budget = min(max(TIME_FIGURE_POINTS // max(len(indices), 1), 512),
                     MAX_POINTS_PER_CURVE)
    wanted = indices[:MAX_REPORT_CURVES]
    source = _positive_lines(source)
    x = np.asarray(source.display_abscissa(us), dtype=float)
    if budget is not None and x.size > budget:
        # the page's curves share one time axis: an envelope on common
        # bins, two points a bin (`decimate.envelope_rows`), so a page
        # of two dozen channels carries one axis and not two dozen
        from ..decimate import envelope_rows

        rows = np.real(np.asarray(source.display_ordinate(us, wanted)))
        axis, envelope = envelope_rows(x, rows, max(budget // 2, 256))
        built = {'kind': 'plot', 'caption': caption, 'logy': False,
                 'logx': False, 'x': [_compact(v) for v in axis],
                 'xlabel': f'{source.abscissa_dim} '
                           f'[{us.label_text(source.abscissa_dim)}]',
                 'ylabel': _axis_text(source, us, wanted[0]),
                 'curves': [{'label': source.record_label(i), 'x': None,
                             'y': _finite(envelope[k])}
                            for k, i in enumerate(wanted)],
                 'note': _thinned_note(int(envelope.shape[1]), int(x.size))}
        frames = _averaging_marks(source)
        if frames is not None:
            built['averaging'] = frames
        events = _shock_marks(source)
        if events is not None:
            built['shocks'] = events
        return built
    # the abscissa's own reading, the same flag the app consults: an
    # SRS lays natural frequency out in decades. The curves are built
    # in hertz — a band's edges, a power law's fill-in — and each is
    # put into decades as it is finished (`_decades`); transforming
    # the lines first put an octave PSD's steps and a specification's
    # law in the wrong places once the axis became the viewer's
    # choice (Brandon, 2026-09-05)
    logx = (bool(getattr(source, 'log_abscissa', False))
            # an octave-band figure reads on a log frequency axis, the
            # way bands are read (Brandon, 2026-09-20)
            or getattr(source, 'bandwidth', None) is not None)
    values = source.display_ordinate(us, wanted)
    logy = source.abscissa_dim == 'frequency' \
        if source.log_ordinate is None else source.log_ordinate
    # complex data reads four ways, exactly as the GUI's drop-down:
    # anything but the magnitude is signed, so the axis goes linear
    component = block.get('component', 'magnitude')
    ylabel = _axis_text(source, us, wanted[0])
    tag = (component if component != 'magnitude'
           and np.iscomplexobj(values) else None)
    if tag == 'real':
        values, ylabel = np.asarray(values).real, f'{ylabel} (real)'
    elif tag == 'imag':
        values, ylabel = (np.asarray(values).imag,
                          f'{ylabel} (imaginary)')
    elif tag == 'phase':
        values, ylabel = np.degrees(np.angle(values)), 'phase [deg]'
    if tag:
        logy = False
    # One rule for every reading. `drawing_shape` is what the flat
    # plot and the waterfall both draw by, so the exported figure
    # cannot say something the app does not: 'steps' for a density
    # (flat across its own bin, so the area drawn is the area summed),
    # 'law' for a specification's breakpoints, 'line' for a value at a
    # frequency. The JS is handed the answer rather than deciding it —
    # it has no `interpolation` field and no way to know.
    shape = drawing_shape(source, bool(tag))
    rounded_x = [_compact(v) for v in x]
    curves = []
    thinned_from = thinned_to = 0
    for position, i in enumerate(wanted):
        y = np.abs(values[position]) if np.iscomplexobj(values) \
            else np.asarray(values[position], dtype=float)
        if shape == 'law':
            # a breakpoint curve is a power law between its points and
            # the frequency axis here is linear, so the straight
            # segment a polyline would draw runs about 5 dB off
            # mid-decade. Filled in on the object's own grid, exactly
            # as the app fills it.
            law_x, y = as_power_law(x, y)
        else:
            law_x = x
        if logy:
            # a point at or below zero has no place on a log axis, and
            # log10(1e-300) is not "no place" — it is -300, which the
            # axis then has to reach down to. A specification is written
            # to zero outside its band, so this is most of its length.
            # Non-finite is what `_finite` turns into a gap.
            with np.errstate(divide='ignore', invalid='ignore'):
                y = np.where(y > 0.0, np.log10(np.maximum(y, 1e-300)),
                             np.nan)
        cx, cy = _decimate(law_x, np.asarray(y, dtype=float), budget)
        thinned_from = max(thinned_from, len(law_x))
        thinned_to = max(thinned_to, len(cx))
        # a curve claims its own grid only when it really is on one —
        # a power law's fill-in, or a decimation that dropped points.
        # Compared against the raw `x` this was true of every octave
        # figure: `_decimate` rounds to PAGE_DIGITS and a band center
        # is a geometric mean, so the rounded copy of the *same* grid
        # compared unequal. The curve then carried an `x` and no
        # `edges`, and the page reads that as "on its own grid, edges
        # unknown" — which dropped the warning and abort zones off
        # their steps (Brandon, 2026-09-21). Compared like with like.
        curves.append({'label': source.record_label(i),
                       'x': (_finite(_decades(cx, logx))
                             if len(cx) != len(x)
                             or not np.array_equal(cx, rounded_x) else None),
                       'y': _finite(cy)})
    dropped = 0 if paged or 'page' in block else available - len(wanted)
    if dropped:
        caption = (caption + f' (first {len(wanted)} of '
                             f'{source.num_records} records)').strip()
    elif paged and len(channels) > 1:
        caption = (caption + f' — one of {len(channels)} control '
                   'channels; the rest are on the drop-down').strip()
    built = {'kind': 'plot', 'caption': caption, 'logy': bool(logy),
             'logx': logx,
             'x': [_compact(v) for v in _decades(x, logx)],
             'xlabel': f'{source.abscissa_dim} '
                       f'[{us.label_text(source.abscissa_dim)}]',
             'ylabel': ylabel, 'curves': curves}
    thinning = _thinned_note(thinned_to, thinned_from)
    if thinning:
        built['note'] = thinning
    frames = _averaging_marks(source)
    if frames is not None:
        built['averaging'] = frames
    events = _shock_marks(source)
    if events is not None:
        built['shocks'] = events
    # flat across each bin exactly when the spectrum says it is one —
    # the same field its area is worked out from, so the exported file
    # shows the number it would compute. This used to be set only for a
    # specification *with limits*, which was wrong both ways round: a
    # measured PSD drew as a polyline through its bin centers, and a
    # written specification — whose points are breakpoints of a curve —
    # drew as the staircase it is not.
    if shape == 'steps':
        built['steps'] = True
        # the object's *own* edges, not midpoints between centers: an
        # octave band's center is the geometric mean of its edges, so
        # midpoints miss them by several percent of a band. The JS
        # cannot work these out — it never sees `bandwidth`.
        own = getattr(source, 'bin_edges', None)
        widths = own() if own is not None and getattr(
            source, 'bandwidth', None) is not None else None
        built['edges'] = _finite(_decades(bin_edges(x, widths), logx))
    if bounded:
        built['channels'] = _specification_channels(source, channels, x,
                                                    logy, us)
        built['curves'][0]['ink'] = True
        # and the figure opens on the band the target is written on,
        # as the comparison does: a controller's target sits on every
        # line to Nyquist, most of them empty (Brandon, 2026-09-19)
        band = _specified_band(source, channels[0], us)
        if band is not None:
            built['home_x'] = [_compact(v) for v in _decades(band, logx)]
    elif paged:
        # the transient specification's channels: waveforms, linear,
        # nothing to shade — a target carries no limits
        rows = np.asarray(source.display_ordinate(us, channels)).real
        built['channels'] = [
            {'label': source.record_label(i),
             'y': _finite(np.asarray(rows[k], dtype=float)), 'zones': []}
            for k, i in enumerate(channels)]
        built['curves'][0]['ink'] = True
    # a curves plot that names a shape set gets the fitting screen's
    # bookmarks: a vertical line at every identified mode — how a
    # drive point FRF reads against the modes found in it
    marked = (objects.get(block.get('shapes'))
              if block.get('shapes') else None)
    if isinstance(marked, ShapeSet) and marked.num_shapes:
        built['marks'] = list(dict.fromkeys(
            float(f) for f in marked.frequency))
    return built


def _synthesized_cmif(frf, shapes, us):
    """The modal model's CMIF over the FRF's own records — the dashed
    line the measured indicator is judged against, exactly as in the
    fitting screen. Records at DOFs the shapes do not cover are left
    out of the synthesis matrix (they contribute zero)."""
    from ..core.data import Frf
    from ..plot import cmif_curves

    powers = {'length': 0, 'velocity': 1, 'acceleration': 2}
    usable = [i for i in range(frf.num_records)
              if shapes.covers(frf.response_dof[i])
              and shapes.covers(frf.reference_dof[i])]
    power = next((powers[frf.known_dim(i).partition('/')[0]]
                  for i in usable
                  if frf.known_dim(i).partition('/')[0] in powers), 2)
    ordinate = shapes.synthesize_frf(
        frf.abscissa,
        [frf.response_dof[i] for i in usable],
        [frf.reference_dof[i] for i in usable], power=power)
    twin = Frf(frf.abscissa, ordinate,
               response_dof=[frf.response_dof[i] for i in usable],
               reference_dof=[frf.reference_dof[i] for i in usable],
               ordinate_dim=[frf.ordinate_dim[i] for i in usable],
               ordinate_unit=[frf.ordinate_unit[i] for i in usable],
               reference_unit=[frf.reference_unit[i] for i in usable],
               dimension_hint=[frf.dimension_hint[i] for i in usable])
    singular, _x = cmif_curves(twin, None, us)
    return singular


def _axis_text(source, us, record=0):
    """The axis label for the record actually plotted — a filtered
    block must not wear the first record's units."""
    from ..units import UNKNOWN

    dim = source.ordinate_dim[record]
    if dim == UNKNOWN:
        hint = source.dimension_hint[record]
        return f'{hint} [units undefined]' if hint else 'units undefined'
    return us.label_text(dim)


def _scene_geometry(geometry, us):
    """(points, node colors, lines, faces, axis unit) — one geometry
    as the scene payload draws it, the GUI's own palette entity for
    entity: the report is the GUI, emailed."""
    from ..viz.geometry import color_rgb, display_points

    points, axis_unit = display_points(geometry, us)
    points = np.asarray(points, dtype=float)
    node_row = {int(n): i for i, n in enumerate(geometry.node_id)}
    lines, faces = [], []
    for index, conn in enumerate(geometry.traceline_conn):
        chain = [node_row[int(n)] for n in conn if int(n) in node_row]
        color = _hex(color_rgb(geometry.traceline_color[index]))
        lines.extend([[a, b, color] for a, b in itertools.pairwise(chain)])
    for index, conn in enumerate(geometry.elem_conn):
        chain = [node_row[int(n)] for n in conn if int(n) in node_row]
        color = _hex(color_rgb(geometry.elem_color[index]))
        if len(chain) >= 3:
            faces.append({'nodes': chain, 'color': color})
        elif len(chain) == 2:
            lines.append(chain + [color])
    node_colors = [_hex(color_rgb(c)) for c in geometry.node_color]
    return points, node_colors, lines, faces, axis_unit or ''


def _scene_block(block, geometry, shapes, us, objects):
    points, node_colors, lines, faces, axis_unit = \
        _scene_geometry(geometry, us)
    node_row = {int(n): i for i, n in enumerate(geometry.node_id)}
    built = {'kind': 'scene', 'caption': block.get('caption', ''),
             'unit': axis_unit,
             'points': [[_compact(v) for v in p] for p in points],
             'node_colors': node_colors,
             'lines': lines, 'faces': faces, 'modes': []}
    quantity = block.get('dofs', '')
    if quantity:
        # a DOFs scene: labeled arrows where its one named data
        # object measures that quantity
        from ..core.data import parse_dof
        from ..core.report import (
            EXCITATION_QUANTITIES,
            series_quantity_dofs,
        )
        from ..viz.geometry import (
            AXIS_COLORS,
            DOF_DIRECTIONS,
            dof_arrow_length,
            dof_axis,
        )

        source = objects.get(block.get('dofs_source'))
        if source is None:
            return None
        arrows = []
        for dof in series_quantity_dofs([('', source, None)], quantity):
            node, direction = parse_dof(dof)
            vector = DOF_DIRECTIONS.get(str(direction).upper())
            axis = dof_axis(direction)
            if node is None or vector is None or axis is None \
                    or int(node) not in node_row:
                continue
            arrows.append({'node': node_row[int(node)],
                           'vector': [_compact(v) for v in vector],
                           'color': AXIS_COLORS[axis],
                           'label': dof})
        if not arrows:
            return None     # nothing measured as that quantity here
        center = points.mean(axis=0)
        extent = float(np.sqrt(((points - center) ** 2)
                               .sum(axis=1).max())) or 1.0
        built['arrows'] = arrows
        built['arrow_length'] = float(dof_arrow_length(
            points[[arrow['node'] for arrow in arrows]], extent))
        built['arrows_incoming'] = quantity in EXCITATION_QUANTITIES
    if shapes is not None:
        from ..deform import ShapeDeflection

        for m in range(shapes.num_shapes):
            deflection = ShapeDeflection(geometry, shapes.coordinate,
                                         shapes.shape_matrix[m])
            real = np.zeros_like(points)
            imag = np.zeros_like(points)
            real[deflection.rows] = deflection._real
            imag[deflection.rows] = deflection._imag
            built['modes'].append({
                'label': shapes.mode_label(m),
                # the corner table over the animation: what is moving
                'info': {'mode': m + 1,
                         'description':
                             str(shapes.description[m]).strip(),
                         'frequency': float(shapes.frequency[m]),
                         'damping': float(shapes.damping[m]) * 100.0},
                'peak': float(deflection.peak_magnitude) or 1.0,
                'real': [[_compact(v) for v in p] for p in real],
                'imag': [[_compact(v) for v in p] for p in imag]})
    return built


def _overlay_block(block, objects, us, links=None):
    """The matched pairs animated over each other: both geometries in
    one scene — the basis side blue, the other orange — each pair's
    two modes phase-aligned (through the projection when the
    geometries differ) and each normalized to its own peak, so the
    sparse test and the dense model swing comparably. Emits a 'scene'
    payload drawn flat: the colors tell the sets apart, so the
    deflection colormap stays out.

    That per-shape normalization is also why the payload carries a
    `note`: it makes the picture unable to show a scale difference
    between the two sets, and a scale difference is nearly always a
    mass unit or a normalization convention worth knowing about."""
    from ..core.geometry import Geometry
    from ..core.matches import MatchedModes
    from ..core.shapes import (
        ShapeSet,
        aligned_mode,
        alignment_factor,
        apply_alignment,
    )
    from ..deform import ShapeDeflection

    matched = objects.get(block.get('source'))
    if not isinstance(matched, MatchedModes) or not matched.pairs:
        return None
    a = objects.get(matched.first)
    b = objects.get(matched.second)

    def linked_geometry(name: str) -> Any:
        # matches committed before the object remembered its
        # geometries: the link groups still know
        group = next((g['members'] for g in (links or [])
                      if name in g['members']), [])
        return next((objects[member] for member in group
                     if isinstance(objects.get(member), Geometry)),
                    None)

    geo_a = (objects.get(matched.first_geometry or '')
             or linked_geometry(matched.first))
    geo_b = (objects.get(matched.second_geometry or '')
             or linked_geometry(matched.second) or geo_a)
    if not (isinstance(a, ShapeSet) and isinstance(b, ShapeSet)
            and isinstance(geo_a, Geometry)
            and isinstance(geo_b, Geometry)):
        return None
    first_color, second_color = '#4c92d9', '#ff8c2b'
    # The basis is the one being looked *at*; the other is drawn through
    # it, at the same quarter opacity the comparison screen uses. Which
    # of the pair that is depends on the project rather than on the
    # order they were matched in, so ask the links — the same question
    # the window asks. Without this the report drew both solid and the
    # near mesh simply hid the far one.
    def role_of(name: str) -> str | None:
        return next((group.get('role') for group in (links or [])
                     if name in group['members']), None)

    basis_is_second = (role_of(matched.second) == 'Basis'
                       and role_of(matched.first) != 'Basis')
    first_alpha, second_alpha = ((OVERLAY_ALPHA, 1.0) if basis_is_second
                                 else (1.0, OVERLAY_ALPHA))
    pa, _na, la, fa, unit = _scene_geometry(geo_a, us)
    pb, _nb, lb, fb = _scene_geometry(geo_b, us)[:4]
    offset = len(pa)
    points = np.vstack([pa, pb])
    lines = ([[i, j, first_color, first_alpha] for i, j, _c in la]
             + [[i + offset, j + offset, second_color, second_alpha]
                for i, j, _c in lb])
    faces = ([{'nodes': f['nodes'], 'color': first_color,
               'alpha': first_alpha} for f in fa]
             + [{'nodes': [n + offset for n in f['nodes']],
                 'color': second_color, 'alpha': second_alpha}
                for f in fb])
    node_colors = ([first_color] * len(pa)
                   + [second_color] * len(pb))
    node_alphas = [first_alpha] * len(pa) + [second_alpha] * len(pb)
    # phase alignment across geometries runs through the projection —
    # the raw sets share no DOFs there
    projected = None
    if geo_a is not geo_b:
        from ..core.correlate import project_shapes
        try:
            projected, _report = project_shapes(b, geo_b, a, geo_a)
        except ValueError:
            projected = None
    modes = []
    for (r, c), mac in zip(matched.pairs, matched.macs):
        if not (0 <= r < a.num_shapes and 0 <= c < b.num_shapes):
            continue
        if projected is None:
            second_shape = aligned_mode(a, r, b, c)
        else:
            second_shape = apply_alignment(
                b.shape_matrix[c],
                alignment_factor(a, r, projected, c))
        real = np.zeros_like(points)
        imag = np.zeros_like(points)
        for geometry, dofs, shape, shift in (
                (geo_a, a.coordinate, a.shape_matrix[r], 0),
                (geo_b, b.coordinate, second_shape, offset)):
            deflection = ShapeDeflection(geometry, dofs, shape)
            peak = float(deflection.peak_magnitude) or 1.0
            real[deflection.rows + shift] = deflection._real / peak
            imag[deflection.rows + shift] = deflection._imag / peak

        def info(name: str, shape_set: Any, mode: int) -> dict[str, Any]:
            return {'set': name, 'mode': mode + 1,
                    'description':
                        str(shape_set.description[mode]).strip(),
                    'frequency': float(shape_set.frequency[mode]),
                    'damping': float(shape_set.damping[mode]) * 100.0}

        modes.append({
            'label': (f'{matched.first} mode {r + 1} ↔ '
                      f'{matched.second} mode {c + 1} — MAC {mac:.2f}'),
            'info': [info(matched.first, a, r),
                     info(matched.second, b, c)],
            'peak': 1.0,
            'real': [[_compact(v) for v in p] for p in real],
            'imag': [[_compact(v) for v in p] for p in imag]})
    if not modes:
        return None
    # each shape is drawn to its own peak just above, which is what makes
    # the sparse test set and the dense model swing comparably — and also
    # what hides one set being thirty times the other. Say so when they
    # are: a reader looking at the picture cannot tell, and it is usually
    # a mass unit or a normalization rather than the structure.
    from ..core.shapes import compare_scaling

    caption = block.get('caption', '')
    note = compare_scaling(a, b, matched.pairs or None).message(
        matched.first, matched.second)
    return {'kind': 'scene', 'caption': caption, 'note': note,
            'unit': unit, 'flat': True,
            'points': [[_compact(v) for v in p] for p in points],
            'node_colors': node_colors, 'node_alphas': node_alphas,
            'lines': lines, 'faces': faces, 'modes': modes}


#: the zones a response must not be in, as the app shades them: between
#: warning and abort in warm, past abort in hot, read the same way above
#: the target and below it. `None` means the edge of the plot — the zone
#: past abort has no far side.
LIMIT_ZONES = (
    ('warning_upper', 'abort_upper', 'warning'),
    ('abort_upper', None, 'abort'),
    ('abort_lower', 'warning_lower', 'warning'),
    (None, 'abort_lower', 'abort'),
)


#: how finely the window shape is sampled for the report. The curve is
#: smooth and a few dozen points draw it to the pixel.
WINDOW_POINTS = 48


def _averaging_marks(source):
    """The frames a PSD would be averaged over, for the trace to carry.

    A time history in a report is not read for its values — nobody
    measures a volt off a printed trace. It is read for what part of
    the run was analyzed and how, which is exactly what the app draws
    over it and what a bare trace leaves out.

    One window shape for all of them, because they share one: it is the
    same window applied at each frame's own span.
    """
    from ..core.averaging import window_shape
    from ..core.data import TimeHistory

    averaging = getattr(source, 'averaging', None)
    if not isinstance(source, TimeHistory) or averaging is None:
        return None
    try:
        rate = source.sample_rate
    except ValueError:
        return None                 # unevenly sampled: no frames to speak of
    shape = window_shape(averaging.window, WINDOW_POINTS,
                         averaging.window_parameter)
    peak = float(np.max(np.abs(shape))) or 1.0
    levels, rows = averaging.levels(rate)
    # the rail's geometry comes from the averaging itself — the very
    # numbers the app's own overlay draws with. The page used to
    # restate these proportions in its JavaScript and was held to them
    # by matching strings; being told is the stronger seam, and it is
    # the same seam the 3-D figure already uses.
    rail = averaging.rail(rate)
    return {
        'frames': [[float(low), float(high)]
                   for low, high in averaging.frame_bounds(
                       rate, float(source.abscissa[0]))],
        # which row each frame sits on. Overlapping frames cannot share
        # one without their windows running through each other, which is
        # what put ten of them on top of each other in the figure.
        'levels': [int(level) for level in levels],
        'rows': int(rows),
        'rail': {key: rail[key] for key in
                 ('baselines', 'glyph', 'cap', 'foot',
                  'band_alpha', 'glyph_alpha')},
        'window': [round(float(v / peak), 4) for v in shape],
        'label': (f'{averaging.frames} x {averaging.frame_length} samples, '
                  f'{averaging.window}, '
                  f'{round(averaging.overlap * 100)}% overlap'),
    }


def _shock_marks(source):
    """The windows an SRS was computed from, for the trace to carry.

    The same argument as `_averaging_marks`, applied to the other way
    a record gets cut up: a time history in a report is read for what
    part of the run was analyzed, and for a shock test that is exactly
    where the events were found. A bare trace of a four-shock run
    shows four bumps and says nothing about which stretches the
    spectra came from — and the SRS figures beside it are one curve
    per event, so the reader has no way to tell which bump is which
    curve.

    Numbered from one, as the app numbers them on screen, so a caption
    or a table can say "shock 2" and mean something the reader can
    find in the figure.
    """
    from ..core.data import TimeHistory

    shocks = getattr(source, 'shocks', None)
    if not isinstance(source, TimeHistory) or not shocks:
        return None
    return {
        'windows': [[float(s.start), float(s.start + s.duration)]
                    for s in shocks],
        'label': (f'{len(shocks)} shock{"s" * (len(shocks) != 1)}, '
                  'each the window its spectrum was computed from'),
    }


def _shared_grid(x, cap=MAX_POINTS_PER_CURVE):
    """Indices to thin every array of one channel by, together.

    `_decimate` keeps the extremes of each bin, so which points it keeps
    depends on the values — two curves put through it separately come
    back on two different frequency grids. A band shaded between them
    would then be shaded between points that are not above each other.
    An even stride keeps one grid for the target and all four of its
    limits, which is what a filled band needs.
    """
    if len(x) <= cap:
        return None
    return np.linspace(0, len(x) - 1, cap).astype(int)


def _specification_channels(source, records, x, logy, us, measured=None):
    """Every control channel, with the zones shaded around each.

    All of them travel in the payload and one is drawn: the figure is
    the app's reading of a specification, and the drop-down reaches the
    rest without a figure per channel.

    Lines were what this drew before, four of them crowding the curve
    they bound. The app shades instead, and shading is the better
    reading — a response is either inside the band or it is not, and a
    filled band says that without asking anyone to tell four dashed
    lines apart.

    `measured` adds the response that answered each channel, matched by
    DOF, which is what turns the figure into the comparison.
    """
    from ..core.data import Specification

    keep = _shared_grid(x)

    def mapped(values: ArrayLike) -> np.ndarray:
        y = np.asarray(values, dtype=float)
        if keep is not None:
            y = y[keep]
        if not logy:
            return _finite(y)
        with np.errstate(divide='ignore', invalid='ignore'):
            return _finite(np.where(y > 0.0,
                                    np.log10(np.maximum(y, 1e-300)), np.nan))

    shown = {name: source.display_limit(name, us)
             for name in Specification.LIMITS}
    answers = (_measured_by_pair(measured, us,
                                 dimension=source.known_dim(records[0]))
               if measured is not None and records else {})
    out = []
    for record in records:
        zones = []
        for lower, upper, severity in LIMIT_ZONES:
            edges = []
            for name in (lower, upper):
                if name is None:
                    edges.append(None)          # the edge of the plot
                elif shown.get(name) is None:
                    break                       # a limit never written
                else:
                    edges.append(mapped(shown[name][record]))
            if len(edges) == 2 and not all(e is None for e in edges):
                zones.append({'lower': edges[0], 'upper': edges[1],
                              'severity': severity})
        channel = {
            'label': _channel_label(source, record),
            'y': mapped(np.asarray(
                source.display_ordinate(us, [record])[0]).real),
            'zones': zones,
        }
        answer = answers.get(_pair_key(source, record))
        if answer is not None:
            channel['response'] = mapped(answer)
        out.append(channel)
    return out


def _banded(measured, block):
    """The measurement on octave bands, when the block asks for it.

    Banded here rather than requiring the reader to have made an octave
    object first: the block says which fraction it wants and the report
    does the integration, so a template is a description of the
    document and not a list of prerequisites.
    """
    per_octave = block.get('octave')
    if not per_octave:
        return measured
    try:
        return measured.to_octave(int(per_octave))
    except (ValueError, AttributeError):
        return None


def _banded_pair(measured, specification, block):
    """Both objects on the bands the block asks for: bands compare only
    with the same bands (Brandon, 2026-09-19), so a block that bands
    the measurement bands the specification with it, limits and all,
    unless the specification is on bands already."""
    banded = _banded(measured, block)
    if banded is None or not block.get('octave'):
        return banded, specification
    if getattr(specification, 'bandwidth', None) is not None:
        return banded, specification
    try:
        return banded, specification.to_octave(int(block['octave']))
    except (ValueError, AttributeError):
        return None, specification


def _srs_comparison_block(block, measured, specification, us, caption):
    """The control channel's measured shock spectra over the
    specification and its band — the reading the app gives when the
    SRS and its specification are selected together.

    Every event of the channel on one plot: the events are what a
    shock series compares. One channel at a time, the rest on the
    drop-down, for the same reason the app pages them.

    Unlike the PSD comparison beside it: no comparison scaling — a
    spectrum sitting 12 dB under its requirement is the finding, and
    rescaling it into agreement would erase it — and no
    outside-the-limit marks, because with several events overlaid a
    mark cannot say whose line went out (skipped deliberately;
    Brandon, 2026-08-20).
    """
    from ..core.compliance import log_interpolate
    from ..core.data import Specification
    from ..plot import axis_label

    events = {}
    for i in range(measured.num_records):
        events.setdefault(_pair_key(measured, i), []).append(i)
    shared = [j for j in range(specification.num_records)
              if _pair_key(specification, j) in events]
    if not shared:
        return None
    x = np.asarray(measured.display_abscissa(us), dtype=float)
    keep = _shared_grid(x)
    grid = x if keep is None else x[keep]
    spec_x = np.asarray(specification.display_abscissa(us), dtype=float)
    limits = {name: specification.display_limit(name, us)
              for name in Specification.LIMITS}
    shown = np.asarray(measured.display_ordinate(us))

    def onto(values):
        return log_interpolate(x, spec_x, np.asarray(values, dtype=float))

    def mapped(values):
        y = np.asarray(values, dtype=float)
        if keep is not None:
            y = y[keep]
        with np.errstate(divide='ignore', invalid='ignore'):
            return _finite(np.where(y > 0.0,
                                    np.log10(np.maximum(y, 1e-300)), np.nan))

    channels = []
    for record in shared:
        target = onto(np.asarray(
            specification.display_ordinate(us, [record])[0]).real)
        edges = {name: (None if values is None else onto(values[record]))
                 for name, values in limits.items()}
        zones = []
        for lower, upper, severity in LIMIT_ZONES:
            pair = [None if name is None else edges.get(name)
                    for name in (lower, upper)]
            if any(name is not None and edges.get(name) is None
                   for name in (lower, upper)):
                continue                      # a limit never written
            zones.append(
                {'lower': None if pair[0] is None else mapped(pair[0]),
                 'upper': None if pair[1] is None else mapped(pair[1]),
                 'severity': severity})
        rows = events[_pair_key(specification, record)]
        responses = []
        for i in rows:
            name = None if measured.block is None else measured.block[i]
            responses.append({'label': str(name) if name
                              else measured.record_label(i),
                              'y': mapped(np.abs(shown[i]))})
        channels.append({'label': _channel_label(specification, record),
                         'y': mapped(target), 'zones': zones,
                         'responses': responses})

    first = channels[0]
    if len(channels) > 1:
        caption = (caption + f' — one of {len(channels)} control '
                   'channels; the rest are on the drop-down').strip()
    # decades on the x axis exactly when the measurement reads that
    # way — an SRS always, a sine level never. Transformed here at the
    # end, after every log_interpolate above has worked in real
    # frequency; transformed at the top, the interpolation would have
    # been onto exponents
    logx = bool(getattr(measured, 'log_abscissa', False))
    if logx:
        grid = np.log10(np.maximum(np.asarray(grid, dtype=float), 1e-300))
    built = {'kind': 'plot', 'caption': caption, 'logy': True,
             'logx': logx,
             'x': [_compact(v) for v in grid],
             'xlabel': f'frequency [{us.label_html("frequency")}]',
             'ylabel': axis_label(measured.ordinate_dim[0], us,
                                  measured.dimension_hint[0]),
             'curves': [{'label': first['label'], 'x': None,
                         'y': first['y'], 'gray': True}],
             'channels': channels, 'label': block.get('label')}
    return built


def _comparison_block(block, measured, specification, us, caption,
                      scale_db=None):
    """The response against its specification, a channel at a time.

    The figure the app draws when both are selected: the target behind,
    the response in front, the zones shaded around them, and the lines
    that went outside abort marked in red above and blue below.

    Built on the **measurement's** frequency axis, with the
    specification interpolated onto it in log-log. The two rarely share
    an axis — a specification is written at breakpoints or on the
    controller's lines — and drawing the response against the
    specification's grid only works while they happen to match, which
    they did here and will not always.

    `compliance.outside` decides what is out, the same call the app and
    the table make, so a line marked on the page is a line counted in
    the table.

    The response drawn is the **scaled** one whenever a comparison
    scale applies — the same curve the app draws and the same data the
    error bars judge — and the caption and the curve's own label both
    say by how much, because a reader who misses the scaling reads the
    figure as a claim about the raw data.
    """
    from ..core.compliance import (
        comparison_scale_db,
        judge,
        log_interpolate,
        matched_records,
    )
    from ..core.data import Specification

    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)
    answers = _measured_by_pair(
        measured, us, scale_db=scale_db,
        dimension=specification.known_dim(0))
    # a block that names its channel is one channel's figure — the
    # random report writes one per control channel rather than one
    # figure with a drop-down (Brandon, 2026-09-18)
    named = block.get('channel')
    shared = [i for i in range(specification.num_records)
              if _pair_key(specification, i) in answers
              and (not named
                   or _channel_label(specification, i) == str(named))]
    if not shared:
        return None
    measured = _positive_lines(measured)
    specification = _positive_lines(specification)
    x = np.asarray(measured.display_abscissa(us), dtype=float)
    keep = _shared_grid(x)
    grid = x if keep is None else x[keep]
    spec_x = np.asarray(specification.display_abscissa(us), dtype=float)
    limits = {name: specification.display_limit(name, us)
              for name in Specification.LIMITS}
    # A banded specification is drawn on its own bins — the target and
    # its zones flat across each of *its* bands, the shape its own
    # figure and the app give it — where every specification used to
    # be interpolated onto the measurement's grid and stepped on the
    # measurement's bins, which drew an octave-band requirement as a
    # staircase of the measurement's making (Brandon, 2026-09-18: "it
    # seems wrong to compare a stair-step psd to a non-stairstep
    # specification"). A specification on lines or at breakpoints is
    # still read onto the measurement's axis: its power law between
    # points needs the dense grid to be drawn as the curve it is.
    banded = getattr(specification, 'bandwidth', None) is not None
    spec_edges = specification.bin_edges() if banded else None

    def onto(values: ArrayLike) -> np.ndarray:
        """One of the specification's curves, on the axis it is drawn
        on: its own bins when banded, the measurement's otherwise."""
        values = np.asarray(values, dtype=float)
        return values if banded else log_interpolate(x, spec_x, values)

    # the measured record each specification record is judged
    # against, by the pairing the table uses
    measured_of = {si: mi for _label, si, mi
                   in matched_records(specification, measured)}

    def mapped(values: ArrayLike, own: bool = False) -> np.ndarray:
        y = np.asarray(values, dtype=float)
        if keep is not None and not own:
            y = y[keep]
        with np.errstate(divide='ignore', invalid='ignore'):
            return _finite(np.where(y > 0.0,
                                    np.log10(np.maximum(y, 1e-300)), np.nan))

    channels = []
    for record in shared:
        target = onto(np.asarray(
            specification.display_ordinate(us, [record])[0]).real)
        response = answers[_pair_key(specification, record)]
        edges = {name: (None if values is None else onto(values[record]))
                 for name, values in limits.items()}
        zones = []
        for lower, upper, severity in LIMIT_ZONES:
            pair = [None if name is None else edges.get(name)
                    for name in (lower, upper)]
            if any(name is not None and edges.get(name) is None
                   for name in (lower, upper)):
                continue                      # a limit never written
            zones.append({'lower': None if pair[0] is None
                          else mapped(pair[0], banded),
                          'upper': None if pair[1] is None
                          else mapped(pair[1], banded),
                          'severity': severity})
        channel = {'label': _channel_label(specification, record),
                   'y': mapped(target, banded), 'response': mapped(response),
                   'zones': zones}
        for name, key in (('abort_upper', 'over'), ('abort_lower', 'under')):
            if limits.get(name) is None or record not in measured_of:
                continue
            # the one judgment (`compliance.judge`, an area against an
            # area over each cell), so a mark on the page is a cell the
            # table counts; the mark stands at the limit's mean density
            # over its cell, in the page's units
            verdict = judge(specification, measured, record, measured_of[record],
                            name, key == 'over', scale_db)
            factor = _display_factor(limits[name][record],
                                     specification.limits[name][record])
            channel[key] = mapped(verdict['level'] * factor)
        channels.append(channel)

    first = channels[0]
    measured_label = (f'measured ({scale_db:+d} dB)' if scale_db
                      else 'measured')
    curves = [{'label': first['label'], 'x': None, 'y': first['y'],
               'gray': True},
              {'label': measured_label, 'x': None, 'y': first['response'],
               'ink': True}]
    if banded:
        # the target on its own grid, stepped on its own edges; the
        # response keeps the block's grid and edges
        curves[0]['x'] = [_compact(v) for v in spec_x]
        curves[0]['steps'] = True
        curves[0]['edges'] = [_compact(v) for v in spec_edges]
    if scale_db:
        caption = (caption + f' — measured data scaled {scale_db:+d} dB '
                   'to the specification').strip()
    if len(channels) > 1:
        caption = (caption + f' — one of {len(channels)} control '
                   'channels; the rest are on the drop-down').strip()
    # the view opens on the specification's own band — the lines its
    # target is written on, not the axis it is stored on, which for a
    # controller's target runs to Nyquist — and the measurement's
    # wider band is a zoom away (Brandon, 2026-09-18 and again
    # 2026-09-19)
    home = _specified_band(specification, shared[0], us) or (
        [spec_edges[0], spec_edges[-1]] if banded
        else [spec_x[0], spec_x[-1]])
    built = {'kind': 'plot', 'caption': caption, 'logy': True,
             'x': [_compact(v) for v in grid],
             'xlabel': f'frequency [{us.label_text("frequency")}]',
             'ylabel': _axis_text(specification, us, shared[0]),
             'steps': True,
             'home_x': [float(home[0]), float(home[1])],
             'curves': curves, 'channels': channels}
    if keep is None:
        # the grid *is* the measured object's own lines, so its own
        # edges apply — and for an octave comparison those are band
        # edges, which midpoints miss by several percent of a band.
        # Thinned (`_shared_grid` strides a huge grid down so the
        # target and its four limits stay on one x), the object's
        # widths no longer describe the points being drawn, so no
        # edges are claimed and the page falls back to midpoints of
        # what it was given — the honest reading of a thinned grid.
        own = getattr(measured, 'bin_edges', None)
        widths = own() if own is not None and getattr(
            measured, 'bandwidth', None) is not None else None
        built['edges'] = [_compact(v) for v in bin_edges(grid, widths)]
    if getattr(measured, 'bandwidth', None) is not None:
        # an octave-band comparison reads on a log frequency axis, the
        # way bands are read; the narrowband one stays linear (Brandon,
        # 2026-09-20)
        _into_decades(built)
    return built


def _into_decades(built):
    """A finished flat figure put on a log frequency axis: every x it
    carries — the grid, the opening window, the bin edges, a curve's
    own grid and edges — taken to decades, and `logx` set so the page
    labels the axis in hertz."""
    def decades(values):
        return _finite(_decades(np.asarray(values, dtype=float), True))

    for key in ('x', 'home_x', 'edges'):
        if built.get(key) is not None:
            built[key] = decades(built[key])
    for curve in built.get('curves', ()):
        for key in ('x', 'edges'):
            if curve.get(key) is not None:
                curve[key] = decades(curve[key])
    built['logx'] = True


def _specified_band(specification, record, us):
    """The band a specification's record actually specifies: from its
    first to its last line whose target is finite and positive — a
    banded one's outer bin edges — on the display axis.

    A controller's target is written on every FFT line of the run,
    NaN or zero outside the band it controlled (the plate's has 726
    positive lines of 1025), so the specification's *abscissa* runs to
    Nyquist and says nothing about where the requirement is. The
    figures open on this band instead (Brandon, 2026-09-19: the
    specification and comparison figures "zoomed across the entire
    data range"). None when no line is positive.
    """
    x = np.asarray(specification.display_abscissa(us), dtype=float)
    y = np.asarray(specification.display_ordinate(us, [record]),
                   dtype=complex).real[0]
    with np.errstate(invalid='ignore'):
        lines = np.flatnonzero(np.isfinite(y) & (y > 0.0))
    if not lines.size:
        return None
    first, last = int(lines[0]), int(lines[-1])
    if getattr(specification, 'bandwidth', None) is not None:
        edges = specification.bin_edges()
        return [float(edges[first]), float(edges[last + 1])]
    return [float(x[first]), float(x[last])]


def _display_factor(shown, raw):
    """What one curve's SI values are multiplied by to be the shown
    ones — a unit conversion is one factor, read off any written
    point. One when nothing is written."""
    shown = np.asarray(np.real(shown), dtype=float)
    raw = np.asarray(np.real(raw), dtype=float)
    with np.errstate(invalid='ignore'):
        good = np.isfinite(shown) & np.isfinite(raw) & (raw > 0.0)
    if not good.any():
        return 1.0
    k = int(np.flatnonzero(good)[0])
    return float(shown[k] / raw[k])


def _channel_label(data, record):
    """How a control channel reads: the DOF alone when it is its own
    reference.

    `record_label` gives '101Z+/101Z+' for an autospectrum, which says
    the same thing twice — and it is the compliance table's own label
    that the marks have to agree with, so they are the same call.
    """
    from ..plot import pair_label

    return pair_label(_pair_key(data, record))


def _pair_key(data, record):
    return data.record_pair(record)


def _measured_by_pair(measured, us, scale_db=0.0, dimension=None):
    """{DOF pair: values} for the records a measurement carries,
    scaled by the comparison's decibels when it has any.

    `dimension` filters to records measuring one quantity. It exists
    because a DOF pair alone is not an identity: a drive point that is
    also a control channel carries a force record and an acceleration
    record under the same pair, and without the filter whichever came
    last shadowed the other — which put a newtons-squared spectrum on
    an acceleration specification's plot and marked every line as
    under the abort floor."""
    keep = [i for i in range(measured.num_records)
            if dimension is None or measured.known_dim(i) == dimension]
    shown = np.asarray(measured.display_ordinate(us, keep))
    factor = 10.0 ** (scale_db / 10.0)
    return {_pair_key(measured, i): np.abs(shown[k]) * factor
            for k, i in enumerate(keep)}


def _kurtosis_bars(block, source):
    """How Gaussian each channel of a record is, a bar apiece.

    A spectrum says nothing about the shape of the distribution that
    produced it: two records with identical PSDs can be a smooth hiss
    and a train of rare hard peaks, and the second fatigues an article
    in a way the first never will. Every channel on one chart whatever
    it measures — kurtosis is dimensionless, so this is the one
    reading here where accelerations and forces share an axis honestly
    (Brandon, 2026-08-24).
    """
    from ..core.data import TimeHistory
    from ..core.kurtosis import (
        HIGH,
        LOW,
        NOMINAL,
        analyzed_span,
        channel_kurtosis,
    )

    if not isinstance(source, TimeHistory):
        return None
    read = channel_kurtosis(source)
    rows = [(label, value) for label, value in read if np.isfinite(value)]
    if not rows:
        return None
    # a dead channel has no shape to describe and drops out — said,
    # not dropped in silence, because a chart of eight bars from a
    # twelve-channel record otherwise reads as a chart of everything
    silent = len(read) - len(rows)
    # which stretch was read, said outright: the same record answers
    # 4.23 whole and 2.92 over its analyzed frames, and a reader
    # cannot reconcile the number with the spectra beside it unless
    # the figure says which one it is
    _spans, phrase = analyzed_span(source)
    caption = f"{block.get('caption', '')}, {phrase}".strip(', ')
    if silent:
        caption += (f' — {silent} channel{"s" * (silent != 1)} recorded '
                    'nothing and cannot be read')
    return {
        'kind': 'bars', 'caption': caption,
        'labels': [label for label, _v in rows],
        'values': [round(float(value), 4) for _l, value in rows],
        'low': LOW, 'high': HIGH,
        # the bars grow from the nominal, not from zero: what is being
        # read is the departure from Gaussian, and it runs both ways
        # (Brandon, 2026-08-24). The axis follows the bars rather than
        # reaching down to a zero no record ever sits at.
        'baseline': NOMINAL,
        'floor': None,
        'units': '',
        'ylabel': f'Pearson kurtosis ({NOMINAL:.0f} is Gaussian)',
    }


def _bars_block(block, specification, measured, us):
    """The comparison as a bar per control channel.

    A table of six channels is read; a table of sixty is scanned, and
    the one channel that matters is somewhere in it. The same numbers as
    bars answer which channel is worst and how many are out before
    anything is read at all — which is why the table these replaced is
    gone rather than sitting beside them.

    Two readings, chosen by `mode`. 'error' is the level each channel
    came out at, in dB, against a threshold either side. 'lines' is how
    much of each channel's band fell outside its abort limits, against
    a ceiling — no amount of staying inside is a fault, so it has no
    floor.

    A transient specification is compared a different way entirely —
    see `_replication_bars` — because it is a waveform rather than a
    statistic, and none of the machinery below applies to it.
    """
    from ..core.compliance import (
        ERROR_DB,
        LINES_PERCENT,
        channel_errors,
        compare_all,
    )
    from ..core.data import ShockSpecification, TransientSpecification

    if isinstance(specification, TransientSpecification):
        return _replication_bars(block, specification, measured)
    if isinstance(specification, ShockSpecification):
        return _srs_bars(block, specification, measured)

    # the scale is resolved before the banding, for the reason the
    # comparison figure resolves it before its own: one report, one
    # number, whichever grid a block reads the data on
    from ..core.compliance import comparison_scale_db

    scale_db = comparison_scale_db(specification, measured)
    measured, specification = _banded_pair(measured, specification, block)
    if measured is None:
        return None
    rows = channel_errors(compare_all(specification, measured,
                                      scale_db=scale_db))
    if not rows:
        return None
    # the bars judge the scaled data, so the caption says so — an RMS
    # error chart that quietly compared scaled data would read as a
    # claim about the raw run
    caption = block.get('caption', '')
    if scale_db:
        caption = (caption + f' — measured data scaled {scale_db:+d} dB '
                   'to the specification').strip()
    lines = block.get('mode') == 'lines'
    return {
        'kind': 'bars', 'caption': caption,
        'labels': [label for label, _db, _pct in rows],
        'values': [round(float(pct if lines else db), 4)
                   for _label, db, pct in rows],
        'low': LINES_PERCENT if lines else -ERROR_DB,
        'high': None if lines else ERROR_DB,
        # a share of a band cannot be negative, so the axis does not
        # pretend it might be; an error in dB runs both ways and gets
        # whatever the data needs
        'floor': 0.0 if lines else None,
        'units': '%' if lines else ' dB',
        'ylabel': ('band outside abort [%]' if lines
                   else 'RMS error [dB]'),
    }


#: mode -> (key, ylabel, units, two-sided). A waveform error cannot be
#: negative and no amount of matching is a fault, so it has a ceiling
#: and no floor; the two dB readings run either way and get both.
def _srs_bars(block, specification, measured):
    """How far each shock spectrum sits from the one it had to meet.

    RMS across the band, in decibels. An SRS spans decades and is read
    on log axes, so the linear distance a waveform gets is dominated by
    whatever bands are loudest and goes blind to a large ratio error
    where the target is small — see `core.compliance.srs_errors`.
    """
    from ..core.compliance import SRS_ERROR_DB, srs_errors

    rows = srs_errors(specification, measured)
    if not rows:
        return None
    many = len({block for _d, block, _v in rows if block}) > 1
    return {
        'kind': 'bars', 'caption': block.get('caption', ''),
        'labels': [f'{dof} {name}' if many and name else dof
                   for dof, name, _value in rows],
        'values': [round(float(value), 4) for _d, _n, value in rows],
        # both sides: the deviation is signed — over the ceiling the
        # shock was too hard, under the floor it under-hit — and the
        # unsigned reading that once made every miss look like an
        # over-test is what a signed value exists to prevent
        'low': -SRS_ERROR_DB, 'high': SRS_ERROR_DB,
        'units': ' dB', 'ylabel': 'SRS deviation, RMS across the band [dB]',
    }


def _sine_bars(block, specification, levels):
    """How far each tone's extracted level sits from its requirement.

    The same signed RMS-in-decibels reading the SRS deviation gives,
    for the same reasons — see `core.compliance.sine_errors`. Every
    tone at every control channel, one bar each.
    """
    from ..core.compliance import SINE_ERROR_DB, sine_errors

    rows = sine_errors(specification, levels)
    if not rows:
        return None
    many = len({tone for _d, tone, _v in rows}) > 1
    return {
        'kind': 'bars', 'caption': block.get('caption', ''),
        'labels': [f'{dof} {tone}' if many else dof
                   for dof, tone, _value in rows],
        'values': [round(float(value), 4) for _d, _t, value in rows],
        'low': -SINE_ERROR_DB, 'high': SINE_ERROR_DB,
        'units': ' dB',
        'ylabel': 'Sine level deviation, RMS across the sweep [dB]',
    }


def _replication_bars(block, specification, measured):
    """How far each control channel is from its target waveform.

    One bar per channel per repeat, and no repeat singled out. Which
    playing of the waveform was the bad one is a judgment about what
    the article is for rather than a measurement, so every one is here
    and the reader makes it.

    Only this reading. A level error wants two PSDs and an SRS
    deviation wants two spectra; both are reached by computing them —
    the specification's own PSD is a `Specification` and its SRS a
    `ShockSpecification`, so the report binds those the same way the
    random report binds its own. The waveform error is what the time
    data alone can answer, which is why it is the one that lives here.

    Every metric is a ratio against the target, so the unit system does
    not enter: both objects are held in SI and the units divide out.

    A channel whose target asks for nothing gets no bar. It cannot be
    given a fair one — the metrics are all ratios — and a bar of zero
    would read as a channel that replicated perfectly, which is the
    opposite of what is known about it. The caption says which channels
    those were rather than letting them vanish.
    """
    from ..core.replication import WAVEFORM_PERCENT, compare

    if block.get('mode', 'waveform') != 'waveform':
        return None
    rows = compare(measured, specification, metrics=('waveform',))
    shown = [r for r in rows if np.isfinite(r['waveform'])]
    if not shown:
        return None
    # the repeat only earns a place in the label when there is more
    # than one of them to tell apart
    many = len({r['frame'] for r in shown}) > 1
    caption = block.get('caption', '')
    silent = sorted({r['label'] for r in rows if r.get('zero_target')})
    if silent:
        caption = (f'{caption} — {", ".join(silent)} asked for nothing '
                   'and cannot be scored against it').strip(' —')
    return {
        'kind': 'bars', 'caption': caption,
        'labels': [f"{r['label']} e{r['frame'] + 1}" if many else r['label']
                   for r in shown],
        'values': [round(float(r['waveform']), 4) for r in shown],
        # a one-sided reading carries its only threshold in `low` and
        # leaves `high` null, which is how the page tells the two apart
        'low': WAVEFORM_PERCENT, 'high': None, 'floor': 0.0,
        'units': '%', 'ylabel': 'waveform error [%]',
    }


def _replication_overlay_block(block, measured, specification, us):
    """The transient comparison itself: the first playing of the
    waveform over the target it was aiming at, one control channel at
    a time on the drop-down — the report's rendering of the app's
    overlay view. Linear axes, because these are waveforms; the
    channel machinery is the specification plot's, with no zones,
    since a target waveform carries no limits."""
    from ..core.data import TimeHistory, TransientSpecification
    from ..core.replication import event_slice, playings

    if (not isinstance(measured, TimeHistory)
            or not isinstance(specification, TransientSpecification)):
        return None
    window = event_slice(measured, specification, 0)
    if window is None:
        return None
    count = len(playings(measured, specification))
    x = [_compact(v) for v in specification.display_abscissa(us)]
    rows = {dof: i for i, dof in enumerate(window.response_dof)}
    channels = []
    for i, dof in enumerate(specification.response_dof):
        if dof not in rows:
            continue
        target = _finite(np.asarray(
            specification.display_ordinate(us, [i])[0]).real)
        answer = _finite(np.asarray(
            window.display_ordinate(us, [rows[dof]])[0]).real)
        channels.append({'label': dof, 'y': target,
                         'response': answer, 'zones': []})
    if not channels:
        return None
    caption = block.get('caption', '')
    first = channels[0]
    return {'kind': 'plot', 'caption': caption, 'logy': False,
            'x': x, 'xlabel': f'time [{us.label_text("time")}]',
            'ylabel': _axis_text(specification, us),
            'curves': [{'label': first['label'], 'x': None,
                        'y': first['y'], 'gray': True},
                       {'label': f'playing 1 of {count}', 'x': None,
                        'y': first['response'], 'ink': True}],
            'channels': channels}


def _table_block(block, source, geometry=None):
    """An object as a table.

    The rows come from `core.tables.table_of`, so a report and a script
    asking what is in a channel table cannot answer differently. The
    caption is this end's business.
    """
    from ..core.shapes import ShapeSet
    from ..core.tables import table_of

    built = table_of(source, geometry)
    if built is None:
        return None
    headers, rows = built
    caption = block.get('caption', '')
    if isinstance(source, ShapeSet) and getattr(source, 'unscaled', False):
        # the reader must know: without a drive point the shapes carry
        # an arbitrary scale, and modal masses mean nothing
        caption = (caption + ' — mode shapes are unscaled (no drive '
                   'point was measured); frequencies, damping and MAC '
                   'are unaffected, but modal masses are not '
                   'physical').strip(' —')
    return {'kind': 'table', 'caption': caption,
            'headers': headers, 'rows': rows}


from .page import (
    _CSS,
    _EDIT_CSS,
    _EDIT_JS,
    _JS,
    _PAGE,
)
