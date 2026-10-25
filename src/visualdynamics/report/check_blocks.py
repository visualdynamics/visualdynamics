"""Check blocks: the checks Visual Dynamics reads, carried in the report
(report feature `check_blocks`).

The app reads a record or a spectrum in ways a test report has to carry:
the time statistics of a random record (the drive's skewness, kurtosis
and density, which MIL-STD-810H (2019), Method 514.8, in the text
under Table 514.8-II, asks for whenever the drive departs from a
Gaussian), whether
a shock record's velocity returns to rest, the points rule of a shock
response spectrum, a classical pulse against its tolerance, a sine record's
distortion, a sweep's rate against the modes it passes, two signatures
compared mode by mode, and a replicated record judged segment by
segment. Each is one function in `core`, which the reading's view, the
script and this module all call; nothing here computes a check a second
way.

With the feature on, each check the report can make becomes a block in
its results: a heading with the check's verdict and its words, the
criterion it was judged against, a table with a row per channel (or
mode, pass or segment), and its figures. A check that judged nothing
says "Not judged" and why, and is never counted a pass: a dead channel,
a sample that is not a number and an input the check refuses each name
their cause.

Which checks a report makes:

- **From what the report shows.** A check is made when the report
  already binds every object the check reads (`AUTOMATIC`): the time
  statistics of a random test's record, the velocity check of a shock
  record, the points rule of a shock response spectrum against its
  requirement, the distortion of each tone of a sine record. The sweep
  rate is made when the project also holds a shape set, whose modes it
  is checked against.
- **From what the author states.** Some checks need a fact no object
  holds: the nominal pulse, the modes a sweep must let build, which
  signature was taken before the test and which after, the segments of
  a replicated record. Those are made from a check's own block, which
  carries what it reads and its settings (`add`,
  `Project.report_check_block`), and saved with the report and its
  template.

The readings can take a while (the distortion check rebuilds every tone
on every channel), so they are made once and kept (`read`, keyed on the
objects they read and the settings), and the app makes them on its
worker before it draws the page (`prepare`, `pending`). A script that
renders a report makes them as it renders.

The blocks are laid out on the page being rendered (`page`), under a
Checks heading after the last figure of the results, so the report's
own blocks keep their order and turning the feature off gives back the
page it was. With skeleton sections on, the Checks fall under Results.
"""

from __future__ import annotations

import html
import json
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# P24 of the report-features plan (Kevin Cross, 2026-10-24): report
# blocks for the checks the app already reads, each with its verdict,
# its numbers and its plot, through the reading's own core function.

#: the feature's key in `report.features`
FEATURE = 'check_blocks'

#: the heading the checks are laid out under
HEADING = 'Checks'

#: a check's verdict: every item judged and inside its criterion
PASSED = 'passed'
#: at least one item judged and outside its criterion
FLAGGED = 'flagged'
#: some items judged and inside, others not judged
PARTLY = 'partly judged'
#: nothing judged
NOT_JUDGED = 'not judged'

#: how each verdict heads its block, in words a monochrome copy keeps
VERDICTS = {PASSED: 'Passed', FLAGGED: 'Flagged', PARTLY: 'Partly judged',
            NOT_JUDGED: 'Not judged'}

#: the most channels, windows, tones or records a check draws a figure
#: for; the table always holds every one
MOST_FIGURES = 4

#: the most curves on one figure; the caption names how many were left off
MOST_CURVES = 8

#: what a channel that does not vary is told, wherever it is read
DOES_NOT_VARY = ('the channel does not vary; a dead or constant channel '
                 'has no distribution to read')

#: what a shock channel that never moves is told
DOES_NOT_MOVE = ('the channel does not move over the window, so there is '
                 'no velocity to return to rest')

#: the most readings kept at once
KEPT = 64

#: the figures the dash for a missing number is
DASH = '—'


@dataclass(frozen=True)
class Check:
    """One check a report can carry.

    `fields` are the binding fields of the check's block, each with the
    symbolic binding it takes by default (`core.report.resolve_binding`);
    `needs` are the fields that must resolve for the check to be made.
    `automatic` checks are made whenever the report binds what they read;
    the others only from a block of their own. `settings` are the
    keyword arguments a block may carry to the reading, and `required`
    those it must.
    """

    key: str
    title: str
    description: str
    fields: tuple[tuple[str, str], ...]
    needs: tuple[str, ...]
    automatic: bool
    settings: tuple[str, ...] = ()
    required: tuple[str, ...] = ()


#: every check, in the order the report lays them out
CHECKS: dict[str, Check] = {check.key: check for check in (
    Check('time_statistics', 'Time statistics',
          "A random record's mean, rms, skewness and kurtosis over the "
          'record and frame by frame, crest factor, and histogram against '
          'the Gaussian, each against its band.',
          (('source', '@basis:TimeHistory'),), ('source',), True,
          ('frame', 'duration', 'kurtosis_band', 'frame_band',
           'skewness_band', 'offset_share', 'crest_band', 'peak_runs',
           'sigmas', 'tail_ratio')),
    Check('shock_validity', 'Shock validity',
          "Each acceleration channel integrated raw over each shock window: "
          'whether its velocity returns to rest.',
          (('source', '@basis:TimeHistory'),), ('source',), True,
          ('tolerance',)),
    Check('srs_points', 'SRS points rule',
          'Each measured shock response spectrum judged point by point '
          "against its requirement's tolerance band.",
          (('source', '@basis:ShockSpecification'),
           ('measured', '@basis:Srs')), ('source', 'measured'), True,
          ('inner', 'outer', 'rule', 'percent', 'limits')),
    Check('pulse', 'Classical pulse',
          'A shock record against the tolerance envelope of its nominal '
          'classical pulse, with its peak, duration, edges and velocity '
          'change.',
          (('source', '@basis:TimeHistory'),), ('source',), False,
          ('shape', 'peak', 'duration', 'edge', 'channel', 'band',
           'duration_tolerance', 'edge_limit', 'velocity_tolerance',
           'quiet', 'window'),
          ('shape', 'peak', 'duration')),
    Check('distortion', 'Sine distortion',
          'The RMS of the record less the tracked tones against the RMS of '
          'the tones, at every line of each tone.',
          (('source', '@basis:TimeHistory'),
           ('specification', '@basis:SineSweepSpecification'),
           ('measured', '@basis:SineLevelSet')),
          ('source', 'specification', 'measured'), True,
          ('tone', 'channels', 'excess', 'cycles')),
    Check('sweep_rate', 'Sweep rate',
          "Each pass of the sweep through a mode's half-power band against "
          "the rule's rate for the mode's Q.",
          (('source', '@basis:SineSweepSpecification'),
           ('shapes', '@basis:ShapeSet')), ('source',), True,
          ('modes', 'tone', 'margin')),
    Check('signature', 'Signature shifts',
          'Two signatures of one article, before and after the test, '
          'compared mode by mode against frequency and amplitude limits.',
          (('source', ''), ('measured', '')), ('source', 'measured'), False,
          ('modes', 'find', 'frequency_limit', 'amplitude_limit', 'search')),
    Check('segments', 'Replication segments',
          'A replicated record judged segment by segment, each by the rule '
          'for its kind, beside the whole-record rms error.',
          (('source', '@basis:TimeHistory'),
           ('specification', '@basis:TransientSpecification')),
          ('source', 'specification'), False,
          ('segments', 'channel', 'lag', 'playing', 'taper_percent', 'band',
           'lines', 'asd_db', 'asd_percent', 'variance_percent',
           'envelope_db', 'window', 'srs_db', 'q', 'per_octave',
           'peak_percent', 'peaks'),
          ('segments',)),
)}

#: the checks made from what the report shows, with no block of their own
AUTOMATIC = tuple(key for key, check in CHECKS.items() if check.automatic)


# ---- what a report carries ---------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One check a report will carry: the check, the names of the objects
    its fields resolved to (`bound`), its settings, and the index of its
    own block in the report (None for a check made from what the report
    shows)."""

    check: Check
    bound: tuple[tuple[str, str], ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    index: int | None = None

    def objects(self, objects: Mapping[str, Any]) -> dict[str, Any]:
        """The objects each field names, by field."""
        return {key: objects.get(name) for key, name in self.bound}


def check(key: Any) -> Check:
    """The check called `key`; `ValueError` naming the checks for any
    other key."""
    if not isinstance(key, str):
        raise TypeError(f'a check is named by its key, a str, not '
                        f'{type(key).__name__}')
    found = CHECKS.get(key)
    if found is None:
        raise ValueError(f'unknown check {key!r}; the checks are: '
                         f'{", ".join(CHECKS)}')
    return found


def add(report: Any, key: str, bindings: Mapping[str, str] | None = None,
        settings: Mapping[str, Any] | None = None) -> int:
    """Give `report` a check of its own, and return its block's index.

    The block is a text block carrying the check's key, its bindings
    (each field the check takes, by default its symbolic binding) and its
    settings, the keyword arguments the check's reading takes. It is
    drawn, with the other checks, only while the feature is on, and is
    saved with the report and its template. A check that is made from
    what the report shows, given a block of its own, is made from the
    block alone, so its settings hold.

    Raises `ValueError` for an unknown check, a field the check does not
    take, a setting it does not take, a required setting left out, a
    field with nothing bound to it, and settings that cannot be saved
    (anything but numbers, strings, booleans, None and lists or
    mappings of them). Nothing is changed when it raises.
    """
    found = check(key)
    given = dict(bindings or {})
    chosen = dict(settings or {})
    takes = dict(found.fields)
    for name in given:
        if name not in takes:
            raise ValueError(f'the {found.title.lower()} check reads '
                             f'{", ".join(takes)}, not {name!r}')
    for name in chosen:
        if name not in found.settings:
            raise ValueError(
                f'the {found.title.lower()} check takes the settings '
                f'{", ".join(found.settings)}, not {name!r}')
    missing = [name for name in found.required if name not in chosen]
    if missing:
        raise ValueError(f'the {found.title.lower()} check needs '
                         f'{", ".join(missing)}')
    bound = {name: str(given.get(name) or default)
             for name, default in found.fields}
    empty = [name for name in found.needs if not bound[name]]
    if empty:
        raise ValueError(f'the {found.title.lower()} check reads '
                         f'{", ".join(empty)}; name what it reads')
    try:
        stored = json.loads(json.dumps(chosen, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError('a check block is saved with the report, so its '
                         'settings are numbers, strings, booleans, None and '
                         'lists of them') from None
    block = {'kind': 'text', 'text': '', 'check': found.key,
             **{name: value for name, value in bound.items() if value},
             'settings': stored}
    report.blocks.append(block)
    return len(report.blocks) - 1


def _resolved(value: Any, objects: Mapping[str, Any],
              object_groups: Sequence[Mapping[str, Any]] | None) -> str:
    from ..core.report import resolve_binding

    name = resolve_binding(value, objects, object_groups) if value else None
    return name if name and name in objects else ''


#: the binding fields of a block, as `core.report` reads them
_BINDINGS = ('source', 'geometry', 'shapes', 'dofs_source', 'measured',
             'specification')


def _shown(report: Any, objects: Mapping[str, Any],
           object_groups: Sequence[Mapping[str, Any]] | None) -> set[str]:
    """The names of the objects the report's own blocks draw: what a
    check made from what the report shows may read."""
    out = set()
    for block in report.blocks:
        if 'check' in block or 'section' in block:
            continue
        for key in _BINDINGS:
            name = _resolved(block.get(key), objects, object_groups)
            if name:
                out.add(name)
    return out


def _applies(key: str, found: Mapping[str, Any], shown_types: set[type]
             ) -> bool:
    """Whether a check made from what the report shows suits the test the
    report is of, given the objects it would read and the classes of
    what the report draws."""
    from ..core.data import ShockSpecification, Srs, TransientSpecification
    from ..core.sine import SineSweepSpecification

    shock = bool(shown_types & {ShockSpecification, Srs})
    if key == 'time_statistics':
        # its bands are for a record meant to be Gaussian random: a
        # shock, a waveform replayed and a swept tone are none of them
        return (not getattr(found['source'], 'shocks', None) and not shock
                and not shown_types & {TransientSpecification,
                                       SineSweepSpecification})
    if key == 'shock_validity':
        return bool(getattr(found['source'], 'shocks', None)) or shock
    return True


def items(report: Any, objects: Mapping[str, Any],
          object_groups: Sequence[Mapping[str, Any]] | None = None
          ) -> list[Item]:
    """The checks `report` carries, in `CHECKS` order: each check of its
    own (a block carrying 'check') whose needed fields resolve, and each
    check in `AUTOMATIC` that the report has no block of and whose every
    needed object the report already draws. The distortion check is made
    once per tone of the level set it reads.
    """
    from ..core.sine import SineLevelSet

    own: list[Item] = []
    keys = set()
    for index, block in enumerate(report.blocks):
        key = block.get('check')
        if key not in CHECKS:
            continue
        keys.add(key)
        found = CHECKS[key]
        bound = tuple((name, _resolved(block.get(name), objects,
                                       object_groups))
                      for name, _default in found.fields)
        if not all(dict(bound)[name] for name in found.needs):
            continue
        settings = block.get('settings') or {}
        own.append(Item(found, bound, dict(settings), index))
    shown = _shown(report, objects, object_groups)
    shown_types = {type(objects[name]) for name in shown}
    made: list[Item] = []
    for key in AUTOMATIC:
        if key in keys:
            continue
        found = CHECKS[key]
        bound = tuple((name, _resolved(default, objects, object_groups))
                      for name, default in found.fields)
        names = dict(bound)
        if not all(names[name] in shown for name in found.needs):
            continue
        if key == 'sweep_rate' and not names['shapes']:
            # the modes a sweep is checked against are a fact about the
            # article; without a shape set the project holds none
            continue
        if not _applies(key, {name: objects[value]
                              for name, value in bound if value},
                        shown_types):
            continue
        if key == 'distortion':
            levels = objects[names['measured']]
            tones = (levels.tone_names if isinstance(levels, SineLevelSet)
                     else [])
            made.extend(Item(found, bound, {'tone': tone}) for tone in tones)
            continue
        made.append(Item(found, bound))
    order = list(CHECKS)
    return sorted(made + own, key=lambda item: order.index(item.check.key))


# ---- the readings, made once ---------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    """What a reading said instead of a result: why it could not be read."""

    reason: str


#: kept readings: key -> (weak references to the objects read, reading)
_KEPT: dict[tuple, tuple[tuple[Any, ...], Any]] = {}

#: the attributes of an object a reading depends on beyond its arrays:
#: the stretch read and the windows found, which change in place
_STATE = ('averaging', 'shocks', 'sine_extraction')


def _state(obj: Any) -> tuple:
    """What identifies `obj` as read: itself, each attribute by identity
    (an array replaced is a new reading), and the settings that say which
    part of a record is read, by value."""
    parts: list[Any] = [id(obj), type(obj).__name__]
    for name, value in sorted(getattr(obj, '__dict__', {}).items()):
        if isinstance(value, (int, float, str, bool, type(None))):
            parts.append((name, value))
        else:
            parts.append((name, id(value)))
    for name in _STATE:
        if hasattr(obj, name):
            parts.append((name, repr(getattr(obj, name))))
    return tuple(parts)


def _key(item: Item, objects: Mapping[str, Any]) -> tuple:
    found = item.objects(objects)
    return (item.check.key,
            tuple((name, _state(obj)) for name, obj in found.items()),
            json.dumps(item.settings, sort_keys=True, default=repr))


def _kept(key: tuple, found: Mapping[str, Any]) -> Any:
    """The kept reading for `key` while every object it read is still the
    one it read, else None."""
    held = _KEPT.get(key)
    if held is None:
        return None
    refs, reading = held
    if any(ref() is not obj for ref, obj in zip(refs, found.values(),
                                                strict=True)):
        del _KEPT[key]
        return None
    return reading


def _keep(key: tuple, found: Mapping[str, Any], reading: Any) -> None:
    try:
        refs = tuple(weakref.ref(obj) for obj in found.values())
    except TypeError:
        return          # an object that cannot be referred to weakly
    while len(_KEPT) >= KEPT:
        del _KEPT[next(iter(_KEPT))]
    _KEPT[key] = (refs, reading)


def forget() -> None:
    """Drop every kept reading, so the next render reads again."""
    _KEPT.clear()


def read(item: Item, objects: Mapping[str, Any]) -> Any:
    """The reading of one item: what its check's core function returns,
    or a `Refusal` with the reason when the function refuses. Made once
    and kept while the objects it read and its settings stay the same."""
    found = item.objects(objects)
    key = _key(item, objects)
    reading = _kept(key, found)
    if reading is not None:
        return reading
    unknown = [name for name in item.settings
               if name not in item.check.settings]
    missing = [name for name in item.check.required
               if name not in item.settings]
    if unknown or missing:
        # a block written by hand, or by a later version: said, not read
        reading = Refusal(
            f'the block\'s settings {", ".join(map(repr, unknown))} are not '
            f'the check\'s' if unknown else
            f'the block does not say {", ".join(missing)}')
    else:
        try:
            reading = _READERS[item.check.key](found, dict(item.settings))
        except (ValueError, TypeError) as refusal:
            # a value the reading cannot take, such as a word where a
            # number belongs, is refused as the reading refuses its own
            reading = Refusal(_reason(refusal))
    _keep(key, found, reading)
    return reading


#: what a reading canceled before it was made is kept as, until it is
#: asked for again
_CANCELED = Refusal('the reading was canceled; it is made when the report '
                    'is drawn again')


def _reason(refusal: BaseException) -> str:
    return str(refusal).strip() or type(refusal).__name__


def pending(report: Any, objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None
            ) -> list[Item]:
    """The items of `report` with no kept reading, or whose reading was
    canceled: what `prepare` would read. Empty when the feature is off."""
    from .features import enabled

    if FEATURE not in enabled(report):
        return []
    out = []
    for item in items(report, objects, object_groups):
        kept = _kept(_key(item, objects), item.objects(objects))
        if kept is None or kept is _CANCELED:
            out.append(item)
    return out


def prepare(report: Any, objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None = None, *,
            progress: Callable[[int, int], Any] | None = None) -> int:
    """Read every check `report` carries that has no kept reading, so the
    page can be drawn without waiting on any. `progress(done, total)` is
    told before each and at the end; what it raises (a cancel) stops the
    reading there, and every item not yet read is kept as not judged,
    the reading canceled, so the page can still be drawn; `pending`
    offers those again. Returns how many were read."""
    waiting = pending(report, objects, object_groups)
    for done, item in enumerate(waiting):
        try:
            if progress is not None:
                progress(done, len(waiting))
        except BaseException:
            for rest in waiting[done:]:
                _keep(_key(rest, objects), rest.objects(objects), _CANCELED)
            raise
        _KEPT.pop(_key(item, objects), None)
        read(item, objects)
    if progress is not None:
        progress(len(waiting), len(waiting))
    return len(waiting)


# ---- the readers: one call each into the check's own core function -----------


def _read_time_statistics(found, settings):
    from ..core.time_statistics import record_statistics
    from ..plot.time_statistics import histogram

    history = found['source']
    out = []
    for row in range(history.num_records):
        try:
            stats = record_statistics(history, row, **settings)
        except ValueError as refusal:
            out.append((row, None, _reason(refusal), None))
            continue
        if not stats.std > 0.0:
            out.append((row, None, DOES_NOT_VARY, None))
            continue
        out.append((row, stats, '', histogram(_spanned(history, row))))
    return out


def _spanned(history, row):
    """One channel's samples over the span its statistics were read
    over (`core.kurtosis.analyzed_span`), real, as a flat array."""
    from ..core.kurtosis import analyzed_span

    spans, _phrase = analyzed_span(history)
    values = np.real(np.asarray(history.ordinate)[row]).astype(float)
    return np.concatenate([values[low:high] for low, high in spans])


def _read_shock_validity(found, settings):
    from ..core.shock_check import check_shock

    return check_shock(found['source'], **settings)


def _read_srs_points(found, settings):
    from ..core.compliance import srs_tolerance_all

    return srs_tolerance_all(found['source'], found['measured'], **settings)


def _read_pulse(found, settings):
    from ..core import classical_pulse as pulses

    pulse = pulses.ClassicalPulse(
        settings.pop('shape'), float(settings.pop('peak')),
        float(settings.pop('duration')), settings.pop('edge', None))
    channel = settings.pop('channel', None)
    return pulses.check_pulse(found['source'], pulse, channel, **settings)


def _read_distortion(found, settings):
    from ..core.sine_distortion import check_distortion

    return check_distortion(found['source'], found['specification'],
                            found['measured'], **settings)


def _read_sweep_rate(found, settings):
    from ..core.sweep_rate import check_sweep_rate

    # the modes the block names, or a shape set's, which gives its own
    # with Q from their damping (`check_sweep_rate`)
    named = settings.pop('modes', None)
    modes = (found.get('shapes') if named is None
             else [tuple(mode) for mode in named])
    if modes is None or (named is not None and not modes):
        raise ValueError('no modes are named for the sweep to be checked '
                         'against; the test plan names them, or a shape set '
                         'does')
    return check_sweep_rate(found['source'], modes, **settings)


def _read_signature(found, settings):
    from ..core.signature import compare_signatures, find_modes

    before, after = found['source'], found['measured']
    named = settings.pop('modes', None)
    count = int(settings.pop('find', 2))
    out = []
    for record in range(before.num_records):
        try:
            modes = named if named is not None else find_modes(
                before, record, count)
            if not len(modes):
                raise ValueError('the signature before has no peak to '
                                 'compare')
            out.append((record, compare_signatures(
                before, after, modes, record=record, **settings)))
        except ValueError as refusal:
            out.append((record, Refusal(_reason(refusal))))
    return out


def _read_segments(found, settings):
    from ..core.replication_segments import aligned_traces, replication_segments

    marks = settings.pop('segments')
    if not len(marks):
        raise ValueError('no segment is marked, so there is nothing to judge '
                         'segment by segment')
    traces = aligned_traces(found['source'], found['specification'],
                            settings.get('channel'), settings.get('lag'),
                            settings.get('playing', 0))
    result = replication_segments(found['source'], found['specification'],
                                  marks, **settings)
    return result, traces


_READERS: dict[str, Callable[[Mapping[str, Any], dict[str, Any]], Any]] = {
    'time_statistics': _read_time_statistics,
    'shock_validity': _read_shock_validity,
    'srs_points': _read_srs_points,
    'pulse': _read_pulse,
    'distortion': _read_distortion,
    'sweep_rate': _read_sweep_rate,
    'signature': _read_signature,
    'segments': _read_segments,
}


# ---- what a reading shows --------------------------------------------------------


@dataclass
class Shown:
    """A reading laid out for the page: its verdict and words, the
    criterion, the table and the figures (each a built 'plot' block)."""

    status: str
    words: str
    criterion: str = ''
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    marks: list[list[str | None]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)


def status_of(verdicts: Sequence[bool | None]) -> str:
    """One verdict from the items' own: True inside, False outside, None
    not judged. Any outside is flagged; nothing judged is not judged;
    some judged and some not, with none outside, is partly judged; and
    only every item judged and inside is passed."""
    if any(verdict is False for verdict in verdicts):
        return FLAGGED
    if not any(verdict is True for verdict in verdicts):
        return NOT_JUDGED
    if any(verdict is None for verdict in verdicts):
        return PARTLY
    return PASSED


def _not_judged(reason: str) -> Shown:
    return Shown(NOT_JUDGED, reason[:1].upper() + reason[1:]
                 + ('' if reason.endswith('.') else '.'))


def _unit(dimension: str, us: Any) -> str:
    from ..units import UNKNOWN

    return '' if dimension == UNKNOWN else us.label_text(dimension)


def _level(value: Any, dimension: str, us: Any) -> Any:
    """SI values in display units, with the unit's offset; a record of
    no declared quantity is shown as it is held."""
    from ..units import UNKNOWN

    if dimension == UNKNOWN:
        return value
    return us.from_si(value, dimension)


def _amount(value: float, dimension: str, us: Any) -> str:
    """An SI value in display units with its unit, to four figures."""
    return f'{float(_level(value, dimension, us)):.4g} {_unit(dimension, us)}'


def _signed(value: float, places: int = 1, unit: str = '') -> str:
    if not np.isfinite(value):
        return DASH
    return (f'{value:+.{places}f}' + (f' {unit}' if unit else '')
            ).replace('-', '−')


def _decimated(x, y):
    from . import _decimate

    return _decimate(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def _figure(caption, x, xlabel, ylabel, curves, *, logx=False, logy=False):
    """A built 'plot' block: `curves` each (label, x or None, y, style),
    x and y in display units (put into decades here for a log axis)."""
    from . import _compact, _decades, _finite

    def axis(values, log):
        values = np.asarray(values, dtype=float)
        if not log:
            return values
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(values > 0.0,
                            np.log10(np.where(values > 0.0, values, 1.0)),
                            np.nan)

    built = []
    for label, cx, cy, style in curves:
        curve = {'label': label, 'y': _finite(axis(cy, logy)),
                 'x': None if cx is None else _finite(axis(cx, logx))}
        curve.update(style)
        built.append(curve)
    return {'kind': 'plot', 'caption': caption, 'logy': bool(logy),
            'logx': bool(logx),
            'x': [None if not np.isfinite(v) else _compact(v)
                  for v in _decades(np.asarray(x, dtype=float), logx)],
            'xlabel': xlabel, 'ylabel': ylabel, 'curves': built}


#: how a curve stands back: the criterion a reading is judged against
_LIMIT = {'gray': True, 'dash': True}


def _left_off(drawn: int, held: int, what: str) -> str:
    """The sentence a caption ends with when curves were left off."""
    if drawn >= held:
        return ''
    return (f' The first {drawn} of {held} {what} are drawn; the table '
            'holds every one.')


def _count(n: int, word: str, plural: str = '') -> str:
    return f'{n} {word if n == 1 else plural or word + "s"}'


# ---- time statistics -----------------------------------------------------------


#: the table's statistics: (key in `TimeStatistics.flags`, column)
_STAT_COLUMNS = (('mean', 'Mean'), ('rms', 'RMS'), ('skewness', 'Skewness'),
                 ('kurtosis', 'Kurtosis'), ('crest', 'Crest factor'),
                 ('frames', 'Frames out'))


def _show_time_statistics(reading, found, us):
    from ..core.kurtosis import NOMINAL, analyzed_span
    from ..core.time_statistics import statistic_name

    history = found['source']
    _spans, phrase = analyzed_span(history)
    headers = ['Channel'] + [name for _key, name in _STAT_COLUMNS] \
        + ['Verdict']
    rows, marks, verdicts, flagged, unread = [], [], [], [], []
    first = None
    for row, stats, cause, _hist in reading:
        label = history.record_label(row)
        if stats is None:
            rows.append([label] + [DASH] * len(_STAT_COLUMNS)
                        + [f'not judged: {cause}'])
            marks.append([None] * len(headers))
            verdicts.append(None)
            unread.append((label, cause))
            continue
        first = first or stats
        dimension = history.ordinate_dim[row]
        scale = us.transform(dimension)[0]
        unit = _unit(dimension, us)
        after = f' {unit}' if unit else ''
        found_flags = stats.flags()
        cells = {
            'mean': f'{_level(stats.mean, dimension, us):.4g}{after}',
            'rms': f'{abs(stats.rms / scale):.4g}{after}',
            'skewness': f'{stats.skewness:.3f}',
            'kurtosis': f'{stats.kurtosis:.3f}',
            'crest': f'{stats.crest:.3f}',
            'frames': (f'{stats.frames_outside} of '
                       f'{stats.frame_kurtosis.size}'
                       if stats.frame_kurtosis.size else DASH),
        }
        rows.append([label] + [cells[key] for key, _name in _STAT_COLUMNS]
                    + [stats.describe()])
        marks.append([None] + ['over' if found_flags.get(key) else None
                               for key, _name in _STAT_COLUMNS]
                     + ['over' if stats.flagged else None])
        verdicts.append(not stats.flagged)
        if stats.flagged:
            flagged.append((row, label, stats))
    status = status_of(verdicts)
    judged = sum(verdict is not None for verdict in verdicts)
    if flagged:
        words = (f'{len(flagged)} of {_count(judged, "channel")} read outside '
                 'a band: ' + '; '.join(
                     f'{label}, ' + ', '.join(statistic_name(key).lower()
                                              for key in stats.flagged)
                     for _row, label, stats in flagged) + '.')
    elif judged:
        words = (f'Every statistic is inside its band on '
                 f'{"all " if judged > 1 else ""}{_count(judged, "channel")}, '
                 f'read {phrase}.')
    else:
        words = 'No channel could be read.'
    if unread:
        words += (' Not judged: ' + '; '.join(
            f'{label}, {cause}' for label, cause in unread) + '.')
    criterion = ''
    if first is not None:
        low, high = first.kurtosis_band
        f_low, f_high = first.frame_band
        s_low, s_high = first.skewness_band
        c_low, c_high = first.crest_band
        criterion = (f'Read {phrase}. Bands: the mean within '
                     f'{first.offset_share:g} of a standard deviation of '
                     f'zero, skewness {s_low:g} to {s_high:g}, kurtosis '
                     f'{low:g} to {high:g} over the record and {f_low:g} to '
                     f'{f_high:g} frame by frame, crest factor {c_low:g} to '
                     f'{c_high:g}; Gaussian is a skewness of 0 and a kurtosis '
                     f'of {NOMINAL:g}.')
    # the drive's density and kurtosis against time are drawn for each
    # channel outside a band, and for the channel furthest from
    # Gaussian when none is, so the record's distribution is shown
    # either way
    judged_rows = [(row, history.record_label(row), stats, hist)
                   for row, stats, _cause, hist in reading
                   if stats is not None]
    if flagged:
        flagged_rows = {row for row, _label, _stats in flagged}
        drawn = [entry for entry in judged_rows if entry[0] in flagged_rows]
    else:
        drawn = sorted(judged_rows,
                       key=lambda entry: -abs(entry[2].kurtosis - NOMINAL))[:1]
    figures = []
    for row, label, stats, hist in drawn[:MOST_FIGURES]:
        figures.extend(_time_statistics_figures(history, row, label, stats,
                                                hist, phrase, us))
    if figures:
        figures[-1]['caption'] += _left_off(MOST_FIGURES, len(drawn),
                                            'channels outside a band')
    return Shown(status, words, criterion, headers, rows, marks, figures)


def _time_statistics_figures(history, row, label, stats, hist, phrase, us):
    """The histogram against the Gaussian and the kurtosis frame by frame,
    for one channel, as the Statistics reading draws them."""
    from . import _axis_text

    dimension = history.ordinate_dim[row]
    # a level converts with the unit's offset, a spread by its scale
    # alone; the area stays one, so the density scales inversely
    scale = abs(us.transform(dimension)[0])
    edges, density = hist
    edges = _level(np.asarray(edges), dimension, us)
    density = np.asarray(density) * scale
    mean, sigma = _level(stats.mean, dimension, us), stats.std / scale
    reach = max(abs(edges[0] - mean), abs(edges[-1] - mean), 5.5 * sigma)
    grid = np.linspace(mean - reach, mean + reach, 401)
    bell = (np.exp(-0.5 * ((grid - mean) / sigma) ** 2)
            / (sigma * np.sqrt(2.0 * np.pi)))
    steps_x = np.repeat(edges, 2)
    steps_y = np.concatenate([[0.0], np.repeat(density, 2), [0.0]])
    beyond = ', '.join(f'{tail.sigmas:g}σ {100.0 * tail.share:.3g} % '
                       f'(Gaussian {100.0 * tail.expected:.2g} %)'
                       for tail in stats.tails)
    level = _axis_text(history, us, row)
    figures = [_figure(
        f'{label}: the histogram of its samples, read {phrase}, against the '
        'Gaussian of the same mean and standard deviation. Skewness '
        f'{stats.skewness:.3f}, kurtosis {stats.kurtosis:.3f} (Gaussian 0 '
        f'and 3); beyond {beyond}.',
        grid, level, 'probability density',
        [('record', steps_x, steps_y, {'color': 0}),
         ('Gaussian, same mean and σ', None, bell, _LIMIT)])]
    times = stats.frame_times
    if times.size:
        unit = us.label_text('time')
        low, high = stats.frame_band
        seconds = _level(stats.frame_length / stats.sample_rate, 'time', us)
        caption = (f'{label}: kurtosis frame by frame, each {seconds:.3g} '
                   f'{unit} frame, against the band {low:g} to {high:g}; '
                   f'{stats.frames_outside} of {times.size} frames outside; '
                   f'the record reads {stats.kurtosis:.3f}.')
        # the record holds a frame, so it has a worst one
        worst = stats.worst_frame
        if not np.isfinite(worst[1]):
            when = _level(worst[0], 'time', us)
            caption += (f' A frame at {when:.4g} {unit} does not '
                        'vary and has no kurtosis; it is a gap in the curve '
                        'and counted outside.')
        t = _level(times, 'time', us)
        figures.append(_figure(
            caption, t, f'time [{unit}]', 'Pearson kurtosis',
            [('each frame', None, stats.frame_kurtosis, {'color': 0}),
             (f'band, {low:g}', None, np.full(t.size, low), _LIMIT),
             (f'band, {high:g}', None, np.full(t.size, high), _LIMIT)]))
    return figures


# ---- shock validity ------------------------------------------------------------


def _show_shock_validity(reading, found, us):
    from ..core.shock_check import summarize

    if not reading:
        return _not_judged('the record holds no window to check')
    windows = len({each.window for each in reading}) > 1
    headers = ['Channel', 'Shock', 'Velocity change', 'Ends at', 'Verdict']
    rows, marks, verdicts, moving, still = [], [], [], [], []
    for each in reading:
        if each.peak_velocity == 0.0:
            rows.append([each.dof, str(each.window + 1), DASH, DASH,
                         f'not judged: {DOES_NOT_MOVE}'])
            marks.append([None] * 5)
            verdicts.append(None)
            still.append(each)
            continue
        moving.append(each)
        if not each.flagged:
            verdict = 'returns to rest'
        elif each.drifting:
            verdict = 'does not return to rest: still changing at the end'
        else:
            verdict = 'does not return to rest: a held velocity change'
        rows.append([each.dof, str(each.window + 1),
                     _amount(each.velocity_change, 'velocity', us),
                     f'{100.0 * each.end_velocity_share:.1f} % of its peak',
                     verdict])
        marks.append([None, None, None, 'over' if each.flagged else None,
                      'over' if each.flagged else None])
        verdicts.append(not each.flagged)
    words = summarize(moving) if moving else 'No channel moves.'
    if still:
        names = sorted({each.dof for each in still})
        words += (f' Not judged: {", ".join(names)}, which '
                  f'{"does" if len(names) == 1 else "do"} not move.')
    tolerance = reading[0].tolerance
    criterion = ('Each acceleration channel integrated raw, nothing removed, '
                 "from rest at its window's start; the velocity ends within "
                 f'{100.0 * tolerance:g} % of its peak.')
    figures = []
    order = sorted({each.window for each in reading})
    unit = us.label_text('time')
    for window in order[:MOST_FIGURES]:
        mine = [each for each in moving if each.window == window]
        if not mine:
            continue
        # the channels that do not return come first, so a figure that
        # cannot hold every curve holds the ones the verdict rests on
        mine.sort(key=lambda each: not each.flagged)
        curves = []
        for k, each in enumerate(mine[:MOST_CURVES]):
            x, y = _decimated(_level(each.time, 'time', us),
                              us.from_si(each.velocity, 'velocity'))
            curves.append((each.dof, x, y, {'color': k}))
        where = f'shock {window + 1}' if windows else 'the window'
        figures.append(_figure(
            f'Velocity through {where}, integrated raw from rest at its '
            'start: a channel that returns to rest ends at zero.'
            + _left_off(MOST_CURVES, len(mine), 'channels'),
            curves[0][1], f'time [{unit}]',
            f'velocity [{us.label_text("velocity")}]',
            [(label, x, y, style) for label, x, y, style in curves]))
    if figures:
        figures[-1]['caption'] += _left_off(MOST_FIGURES, len(order),
                                            'shocks')
    return Shown(status_of(verdicts), words, criterion, headers, rows, marks,
                 figures)


# ---- the SRS points rule --------------------------------------------------------


def _show_srs_points(reading, found, us):
    from ..plot.srs_tolerance import band_text, describe

    if not reading:
        return _not_judged('no record of the spectrum is at a channel the '
                           'requirement names')
    headers = ['Channel', 'Event', 'Points inside', 'Worst point', 'Verdict']
    rows, marks, verdicts, refused = [], [], [], []
    judged = []
    for dof, event, result in reading:
        if 'refused' in result:
            rows.append([dof, event or DASH, DASH, DASH,
                         f'not judged: {result["refused"]}'])
            marks.append([None] * 5)
            verdicts.append(None)
            refused.append((dof, result['refused']))
            continue
        judged.append((dof, event, result))
        verdict = {True: 'passes', False: 'fails', None: 'not judged'}[
            result['passed']]
        worst = (f'{_signed(result["worst_db"], 1, "dB")} at '
                 f'{result["worst_frequency"]:.4g} Hz'
                 if np.isfinite(result['worst_db']) else DASH)
        rows.append([dof, event or DASH,
                     f'{result["inside_count"]} of {result["points"]}'
                     + (f' ({result["inside_percent"]:.1f} %)'
                        if result['points'] else ''),
                     worst, verdict])
        bad = 'over' if result['passed'] is False else None
        marks.append([None, None, bad, None, bad])
        verdicts.append(result['passed'])
    lines = [f'{dof}{f", {event}" if event else ""}: {describe(result)}'
             for dof, event, result in judged]
    words = '; '.join(lines) + '.' if lines else 'No record could be judged.'
    if refused:
        words += ' Not judged: ' + '; '.join(
            f'{dof}, {reason}' for dof, reason in refused) + '.'
    criterion = ''
    figures = []
    if judged:
        first = judged[0][2]
        rule = (f'at least {first["percent"]:g} % of the points inside'
                if first['rule'] == 'share'
                else 'every point at or above the lower edge')
        criterion = (f'Band {band_text(first["inner"], first["outer"])} about '
                     f'the requirement; rule: {rule}'
                     + ('; the band read from the requirement\'s limit curves'
                        if first.get('limits') else '') + '.')
        figures.append(_srs_points_figure(found['measured'], judged))
    return Shown(status_of(verdicts), words, criterion, headers, rows, marks,
                 figures)


def _srs_points_figure(measured, judged):
    """Each judged spectrum's distance from its requirement in dB at its
    own points, against natural frequency, with the band's edges."""
    frequencies = np.asarray(measured.abscissa, dtype=float)
    curves = []
    for k, (dof, event, result) in enumerate(judged[:MOST_CURVES]):
        db = np.where(result['judged'], result['db'], np.nan)
        curves.append((f'{dof}{f", {event}" if event else ""}', None, db,
                       {'color': k}))
    low, high = judged[0][2]['inner']
    for name, edge in (('band, lower edge', low), ('band, upper edge', high)):
        values = np.broadcast_to(np.asarray(edge, dtype=float),
                                 frequencies.shape)
        if np.isfinite(values).any():
            curves.append((name, None, values, _LIMIT))
    return _figure(
        'Each spectrum\'s distance from its requirement at its own points, '
        'in dB, against the band the points rule reads.'
        + _left_off(MOST_CURVES, len(judged), 'spectra'),
        frequencies, 'natural frequency [Hz]', 'distance from requirement [dB]',
        curves, logx=True)


# ---- the classical pulse ---------------------------------------------------------


#: what the table calls each of `plot.classical_pulse.readout`'s rows
_PULSE_ROWS = {'peak': 'Peak', 'duration': 'Duration', 'rise': 'Rise',
               'fall': 'Fall', 'velocity': 'Velocity change',
               'outside': 'Points outside the envelope'}


def _show_pulse(reading, found, us):
    from ..plot.classical_pulse import readout

    numbers = readout(reading, us)
    rows = [[_PULSE_ROWS[key], value] for key, value in numbers.items()]
    marks = [[None, 'over' if value.endswith(', out') else None]
             for value in numbers.values()]
    if not reading.envelope_ok:
        marks[-1][1] = 'over'
    pulse = reading.pulse
    factor = float(us.from_si(1.0, 'acceleration'))
    criterion = (f'Nominal: {pulse.describe(us.label_text("acceleration"), factor)}'
                 f'. Envelope ±{100.0 * reading.band:g} % of the peak, '
                 f'duration ±{100.0 * reading.duration_tolerance:g} %, '
                 f'velocity change ±{100.0 * reading.velocity_tolerance:g} %.')
    unit = us.label_text('time')
    t = _level(reading.time, 'time', us)

    def shown(values):
        return np.asarray(us.from_si(np.asarray(values), 'acceleration'),
                          dtype=float)

    figure = _figure(
        'The record over the tolerance envelope of its nominal pulse, '
        'placed on the measured start.',
        t, f'time [{unit}]',
        f'acceleration [{us.label_text("acceleration")}]',
        [('envelope, lower edge', None, shown(reading.lower), _LIMIT),
         ('envelope, upper edge', None, shown(reading.upper), _LIMIT),
         ('record', None, shown(reading.acceleration), {'color': 0})])
    return Shown(PASSED if reading.passed else FLAGGED, reading.describe(),
                 criterion, ['Reading', 'Measured'], rows, marks, [figure])


# ---- sine distortion --------------------------------------------------------------


def _show_distortion(reading, found, us):
    from ..core.sine_distortion import decibels

    headers = ['Tone', 'Channel', 'Worst', 'At', 'Lines over', 'Verdict']
    rows, marks, verdicts = [], [], []
    for r, dof in enumerate(reading.dofs):
        judged = int(reading.judged[r].sum())
        over = int(reading.over[r].sum())
        worst = reading.worst(dof)
        if worst is None:
            rows.append([reading.tone, dof, DASH, DASH, DASH,
                         'not judged: no line could be judged'])
            marks.append([None] * 6)
            verdicts.append(None)
            continue
        frequency, value = worst
        rows.append([reading.tone, dof,
                     f'{100.0 * value:.1f} % ({float(decibels(value)):.1f} dB)',
                     f'{frequency:.4g} Hz', f'{over} of {judged}',
                     'over the limit' if over else 'within the limit'])
        bad = 'over' if over else None
        marks.append([None, None, bad, None, bad, bad])
        verdicts.append(not over)
    words = reading.describe() + '.'
    if reading.notes:
        words += ' ' + ' '.join(note.rstrip('.') + '.'
                                for note in reading.notes)
    criterion = (f'Distortion, the RMS of the record less the tracked tones '
                 f'over the RMS of the tones, read over '
                 f'{reading.cycles:g} cycles at each line, at most '
                 f'{100.0 * reading.limit:.1f} % (the unfiltered RMS within '
                 f'{100.0 * reading.excess:g} % of the tone\'s).')
    curves = []
    for r, dof in enumerate(reading.dofs[:MOST_CURVES]):
        values = np.where(reading.judged[r], 100.0 * reading.distortion[r],
                          np.nan)
        curves.append((dof, None, values, {'color': r}))
    curves.append((f'limit, {100.0 * reading.limit:.1f} %', None,
                   np.full(reading.frequency.shape, 100.0 * reading.limit),
                   _LIMIT))
    figure = _figure(
        f'{reading.tone}: distortion along the sweep on each channel; a line '
        'that could not be judged is a gap.'
        + _left_off(MOST_CURVES, len(reading.dofs), 'channels'),
        reading.frequency, 'frequency [Hz]', 'distortion [%]', curves,
        logx=True)
    return Shown(status_of(verdicts), words, criterion, headers, rows, marks,
                 [figure])


# ---- sweep rate --------------------------------------------------------------------


def _show_sweep_rate(reading, found, us):
    if not reading:
        return _not_judged("the sweep reaches none of the modes' half-power "
                           'bands, so no pass can be judged')
    headers = ['Tone', 'Mode', 'Q', 'Direction', 'Rate', "Rule's rate",
               'Multiple', 'Cycles in band', 'Verdict']
    rows, marks = [], []
    for each in reading:
        part = '' if each.crossed else ' (not a full crossing)'
        rows.append([each.tone, f'{each.frequency:.4g} Hz', f'{each.q:.3g}',
                     each.direction, f'{abs(each.rate):.3g} {each.unit}',
                     f'{each.limit:.3g} {each.unit}', f'{each.ratio:.2g}',
                     f'{each.cycles:.3g} of {each.q:.3g}{part}',
                     each.verdict])
        bad = 'over' if each.too_fast else None
        marks.append([None] * 6 + [bad, None, bad])
    fast = [each for each in reading if each.too_fast]
    if fast:
        words = (f'{len(fast)} of {_count(len(reading), "pass", "passes")} too fast: '
                 + '; '.join(each.describe() for each in fast) + '.')
    else:
        words = (f'Every pass gives its mode time to build '
                 f'({_count(len(reading), "pass", "passes")}).')
    criterion = (f"Each pass at no more than {reading[0].margin:g} times the "
                 "rule's rate for the mode's Q, the rate at which a mode "
                 'builds to about 94 % of its steady height.')
    figures = _sweep_rate_figures(found['source'], reading)
    return Shown(status_of([not each.too_fast for each in reading]), words,
                 criterion, headers, rows, marks, figures)


def _sweep_rate_figures(specification, reading):
    from ..core.sine import SineTone
    from ..core.sweep_rate import half_power_band

    figures = []
    tones = list(dict.fromkeys(each.tone for each in reading))
    for name in tones[:MOST_FIGURES]:
        tone = (specification if isinstance(specification, SineTone)
                else specification.tone(name))
        duration = tone.duration()
        t, f = tone.trajectory(duration / 2000.0)
        curves = [(name, None, f, {'color': 0})]
        seen = []
        for each in reading:
            mode = (each.frequency, each.q)
            if each.tone != name or mode in seen:
                continue
            seen.append(mode)
            low, high = half_power_band(*mode)
            for edge in (low, high):
                band = f'{each.frequency:.4g} Hz band, Q {each.q:.3g}'
                curves.append((band, None, np.full(t.size, edge), _LIMIT))
        figures.append(_figure(
            f"{name}: the sweep's frequency against time, with each mode's "
            'half-power band; a pass is the time the sweep spends between '
            "a band's two edges.",
            t, 'time [s]', 'frequency [Hz]', curves, logy=True))
    return figures


# ---- signature shifts ---------------------------------------------------------------


def _show_signature(reading, found, us):
    from ..core.signature import matching_record
    from ..plot.signature import display_scale, display_unit, percent

    before, after = found['source'], found['measured']
    headers = ['Channel', 'Mode', 'Hz before, after', 'Frequency shift',
               'Height shift', 'Verdict']
    rows, marks, verdicts, lines, compared = [], [], [], [], []
    for record, comparison in reading:
        label = before.record_label(record)
        if isinstance(comparison, Refusal):
            rows.append([label, DASH, DASH, DASH, DASH,
                         f'not judged: {comparison.reason}'])
            marks.append([None] * 6)
            verdicts.append(None)
            lines.append(f'{label}: not judged, {comparison.reason}')
            continue
        compared.append((record, label, comparison))
        for shift in comparison.shifts:
            rows.append([label, shift.mode.label(),
                         (f'{shift.before.frequency:.1f}, '
                          f'{shift.after.frequency:.1f}'),
                         percent(shift.frequency_shift),
                         percent(shift.amplitude_shift), shift.verdict])
            bad = 'over' if shift.passed is False else None
            marks.append([None, None, None,
                          bad if not shift.frequency_inside else None,
                          bad if not shift.amplitude_inside else None, bad])
        verdicts.append(comparison.passed)
        lines.append(f'{label}: {comparison.describe()}'
                     + ''.join(f'; {note}' for note in comparison.notes))
    words = '. '.join(line.rstrip('.') for line in lines) + '.'
    criterion = ''
    if compared:
        f_low, f_high = compared[0][2].frequency_limit
        a_low, a_high = compared[0][2].amplitude_limit
        criterion = (f'Each mode that counts within {percent(f_low)} to '
                     f'{percent(f_high)} in frequency and {percent(a_low)} '
                     f'to {percent(a_high)} in amplitude; a mode that does '
                     'not count is shown and not judged.')
    figures = []
    for record, label, comparison in compared[:MOST_FIGURES]:
        dimension = before.ordinate_dim[record]
        scale = display_scale(dimension, us)
        unit = display_unit(dimension, us)
        other = matching_record(before, record, after)
        figures.append(_figure(
            f'{label}: the signature before the test and after it, '
            f'{comparison.describe()}.',
            np.asarray(before.abscissa, dtype=float), 'frequency [Hz]',
            f'magnitude [{unit}]' if unit else 'magnitude',
            [('before', None, np.abs(np.atleast_2d(before.ordinate)[record])
              * scale, {'color': 0}),
             ('after', np.asarray(after.abscissa, dtype=float),
              np.abs(np.atleast_2d(after.ordinate)[other]) * scale,
              {'color': 1})],
            logx=True, logy=True))
    if figures:
        figures[-1]['caption'] += _left_off(MOST_FIGURES, len(compared),
                                            'channels')
    return Shown(status_of(verdicts), words, criterion, headers, rows, marks,
                 figures)


# ---- replication segments --------------------------------------------------------------


def _show_segments(reading, found, us):
    from ..plot.replication_segments import COLUMNS, describe_record, verdict_rows

    result, traces = reading
    rows = [list(row) for row in verdict_rows(result)]
    marks = [[None] * (len(COLUMNS) - 1)
             + ['over' if row[-1] == 'fails' else None] for row in rows]
    # the segments' own verdicts, read as every check's are
    status = status_of([segment.passed for segment in result.segments])
    time, reference, control, dof, _shift = traces
    specification = found['specification']
    # the traces are read at a DOF of the reference (`aligned_traces`)
    dimension = specification.ordinate_dim[
        list(specification.response_dof).index(dof)]
    t = _level(np.asarray(time), 'time', us)
    x, ref = _decimated(t, _level(np.asarray(reference), dimension, us))
    cx, ctl = _decimated(t, _level(np.asarray(control), dimension, us))
    unit = _unit(dimension, us)
    figure = _figure(
        f'{dof}: the replicated record over its reference, aligned, the '
        'segments judged in the table.',
        x, f'time [{us.label_text("time")}]',
        f'{dimension} [{unit}]' if unit else dimension,
        [('reference', None, ref, {'gray': True}),
         ('replicated', cx, ctl, {'color': 0})])
    return Shown(status, describe_record(result) + '.',
                 'Each segment by the rule for its kind: a stationary '
                 'stretch by its ASD and variance, a transient by its '
                 'envelope and normalized ASD, a shock by its SRS and peaks.',
                 list(COLUMNS), rows, marks, [figure])


_SHOWERS: dict[str, Callable[[Any, Mapping[str, Any], Any], Shown]] = {
    'time_statistics': _show_time_statistics,
    'shock_validity': _show_shock_validity,
    'srs_points': _show_srs_points,
    'pulse': _show_pulse,
    'distortion': _show_distortion,
    'sweep_rate': _show_sweep_rate,
    'signature': _show_signature,
    'segments': _show_segments,
}


def show(item: Item, objects: Mapping[str, Any], us: Any) -> Shown:
    """One item's reading (`read`) laid out in the display units `us`."""
    reading = read(item, objects)
    if isinstance(reading, Refusal):
        return _not_judged(reading.reason)
    return _SHOWERS[item.check.key](reading, item.objects(objects), us)


# ---- the page ---------------------------------------------------------------------------


def title_of(item: Item) -> str:
    """What an item's block is headed: the check's title, and the tone
    for a distortion check of one tone."""
    tone = item.settings.get('tone') if item.check.key == 'distortion' else None
    return f'{item.check.title}, {tone}' if tone else item.check.title


def page(report: Any, objects: Mapping[str, Any], us: Any,
         object_groups: Sequence[Mapping[str, Any]] | None = None,
         edit: bool = False) -> list[dict[str, Any]]:
    """The page blocks of every check `report` carries, in order: the
    Checks heading, then for each check its head (verdict, words and
    criterion), its table and its figures. Empty when it carries none.

    A head is a text block carrying 'check_head' for `finish` to draw
    once the page's text is rendered. On the edit page, the blocks of a
    check of its own carry its block's index, so a click selects it.
    """
    found = items(report, objects, object_groups)
    if not found:
        return []
    out: list[dict[str, Any]] = [
        {'kind': 'text', 'text': '', 'check_head': {'heading': HEADING}}]
    for item in found:
        shown = show(item, objects, us)
        head: dict[str, Any] = {
            'kind': 'text', 'text': '',
            'check_head': {'title': title_of(item), 'status': shown.status,
                           'words': shown.words,
                           'criterion': shown.criterion}}
        blocks = [head]
        if shown.rows:
            table = {'kind': 'table',
                     'caption': f'{title_of(item)}: {shown.status}',
                     'headers': list(shown.headers),
                     'rows': [list(row) for row in shown.rows]}
            if any(any(mark for mark in row) for row in shown.marks):
                table['marks'] = shown.marks
            blocks.append(table)
        blocks.extend(shown.figures)
        if edit and item.index is not None:
            for block in blocks:
                block['index'] = item.index
        out.extend(blocks)
    return out


def origin(block: Mapping[str, Any]) -> dict[str, Any]:
    """What a check's page block stands for when the skeleton places it:
    the Checks heading as a level-two heading, and the rest as figures of
    no instrumentation, which fall under Results."""
    meta = block.get('check_head') or {}
    if 'heading' in meta:
        return {'kind': 'text', 'text': f'## {meta["heading"]}'}
    return {'kind': block.get('kind'), 'check': True}


def position(blocks: Sequence[Mapping[str, Any]]) -> int:
    """Where the checks go among the page's built blocks: at the first
    heading after the last figure (the section that closes the report),
    or at the end when no heading follows it."""
    figures = [k for k, block in enumerate(blocks)
               if block.get('kind') not in ('text', 'unbound', 'verdict')]
    last = figures[-1] if figures else -1
    for k in range(last + 1, len(blocks)):
        block = blocks[k]
        if block.get('kind') == 'text' and \
                str(block.get('text', '')).lstrip().startswith('## '):
            return k
    return len(blocks)


def finish(blocks: Sequence[dict[str, Any]]) -> None:
    """Draw each check's head in place: the Checks heading, or a check's
    title with its verdict in words, its words, and its criterion."""
    for block in blocks:
        meta = block.pop('check_head', None)
        if meta is None:
            continue
        if 'heading' in meta:
            block['html'] = f'<h3>{html.escape(meta["heading"])}</h3>'
            continue
        status = meta['status']
        parts = [f'<h4 class="check">{html.escape(meta["title"])}</h4>',
                 (f'<p class="check-verdict check-{status.replace(" ", "-")}">'
                  f'<strong>{html.escape(VERDICTS[status])}.</strong> '
                  f'{html.escape(meta["words"])}</p>')]
        if meta['criterion']:
            parts.append(f'<p class="check-criterion">'
                         f'{html.escape(meta["criterion"])}</p>')
        block['html'] = ''.join(parts)
