"""Report features: optional additions to a report, each turned on or off.

A report carries the keys of the features enabled on it
(`Report.features`). With none enabled, the report renders exactly as
it would without this module: the same page, byte for byte. Each
feature changes only what its description says, and only when it is
enabled, so a lab turns on what its reports need and nothing else
moves.

The registry (`FEATURES`) lists every feature, including those not yet
built, so the selection tool shows what is planned. A feature that is
not built is marked unavailable and cannot be enabled. A feature can
need another feature (`requires` names its key), or a capability the
package does not have yet (`CAPABILITIES`); while a requirement is not
met, the feature cannot be enabled and the tool says what is missing.

The three presets (`PRESETS`) are starting points: the default (no
features, today's report), a test report, and a flash report. A preset
may name features that are not available yet; applying it enables the
ones that are and says which it skipped and why.

`configure` is the one place a report's features change, and refuses
an unknown key, an unavailable feature or an unmet requirement before
changing anything. `Project.report_features` and the report editor's
Report Features panel both call it.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Kevin Cross's plan for the report features, 2026-10-24: every
# improvement to the generated report is an optional feature, off by
# default, chosen in a selection tool, with presets. This module is the
# framework; P23 (plot conventions) was the first feature built with
# it, and P18 (skeleton sections, `report/skeleton.py`) the second.
# P25 (responses against prediction, `report/prediction.py`) draws its
# figures and table under the skeleton's Results, and the data index
# (`report/data_index.py`) prints under its Record and Data Package.
# P21 (the findings register, `report/findings.py`) fills the
# skeleton's deviations section.
# P17 (the requirement and its result table, `report/requirement.py`)
# prints under the skeleton's Requirement and Results, and P30 (the
# control channels' uncertainty, drawn by the same module) under its
# Requirement; P26 (cross-axis and limit channels) adds rows to P17's
# result table. P29 (the flash report, `report/flash.py`) draws one
# preliminary page from P19's identity, P17's result rows and P21's
# register in place of the report.

#: the groups the selection tool lays the features out in, in order
GROUPS = ('Structure', 'Results', 'Records', 'Issue control', 'Figures')

#: what an unbuilt feature says in place of a requirement
NOT_YET = 'not yet available'


@dataclass(frozen=True)
class ReportFeature:
    """One optional addition to a report.

    `key` is what a report stores and a script names. `title` and
    `description` are what the selection tool shows: the title on the
    check box, the description beneath it and in its tooltip. `group` is
    one of `GROUPS`. `requires` names what must be in place first: the
    key of another feature, which must be enabled, or the key of a
    capability in `CAPABILITIES`, which the package must have. `default`
    is whether a new report starts with the feature on; every feature
    is off by default. `available` is False for a feature that is
    registered so the tool can show it but is not built yet.
    """

    key: str
    title: str
    group: str
    description: str
    requires: tuple[str, ...] = ()
    default: bool = False
    available: bool = True

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f'report feature {self.key!r}: group '
                             f'{self.group!r} is not one of {GROUPS}')


@dataclass(frozen=True)
class Capability:
    """Something a feature needs that the package itself must provide.

    `label` names it in a reason ("needs the multi-point composite").
    `probe` answers whether this installation has it.
    """

    key: str
    label: str
    probe: Callable[[], bool]


def _has_module(name: str) -> Callable[[], bool]:
    """A probe for a module of the package: True once it is installed."""
    def probe() -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False
    return probe


#: what the planned features wait on that is not a feature of its own
CAPABILITIES: dict[str, Capability] = {cap.key: cap for cap in (
    Capability('multi_point', 'the multi-point composite',
               _has_module('visualdynamics.core.multi_point')),
    Capability('chance', 'the rule for lines outside a band by chance',
               _has_module('visualdynamics.core.chance')),
    Capability('record_caption', "the caption read off a plot's record",
               _has_module('visualdynamics.core.report_caption')),
)}


#: every report feature, in the order the selection tool shows them
#: within their groups. Every feature registered here is built; a
#: feature registered before it is built carries `available=False`,
#: and the selection tool shows it grayed until it is.
FEATURES: dict[str, ReportFeature] = {feature.key: feature for feature in (
    ReportFeature(
        'skeleton_sections', 'Skeleton sections', 'Structure',
        'Every section a test report needs, in order, with what the '
        'project generates under the sections it answers. A section only '
        'the author can write is shown as owed until it is written; the '
        "deviations and the discussion can be set to 'None' in one step, "
        'and the export lists what is still owed.'),
    ReportFeature(
        'run_sections', 'Runs and chronology', 'Structure',
        'Run identity on each record (axis, run, level, start), read from '
        'the file name or the comparison and entered where it cannot be; '
        'the Results grouped by run; and a chronology of the runs with '
        'their time at level and the interruptions found, offered as '
        'draft text for the author to accept or edit.',
        requires=('skeleton_sections',)),
    ReportFeature(
        'test_requirement', 'Requirement and results', 'Results',
        'The test requirement in words, with the source of each number, '
        'and a result table with one row per criterion, worded met, not '
        'met or not judged, each miss with its band and size. The '
        'pass/fail box reads from the requirement; without one it is '
        'labeled a screening.',
        requires=('multi_point', 'chance')),
    ReportFeature(
        'findings', 'Findings register', 'Results',
        'Deviations, anomalies and nonconformances entered in a register, '
        'numbered, each in four parts: fact, cause, effect and approval. '
        'Section 9 draws them and the summary lists them by number; a '
        'blank part is owed. The fact can be offered as a draft from a '
        'measurement; the rest are typed.',
        requires=('skeleton_sections',)),
    ReportFeature(
        'check_blocks', 'Check blocks', 'Results',
        'A block for each check the report can make (time statistics, '
        'the rms check, lines by chance, shock validity, the SRS points '
        'rule, the pulse check, the sine distortion, frequency and sweep '
        'rate, signature shifts, replication segments): its verdict, a '
        'row per channel against its criterion, '
        'and its figures. A check that judged nothing says why.'),
    ReportFeature(
        'prediction', 'Responses against prediction', 'Results',
        'Each response channel against its prediction scaled to the '
        'level run: a figure of the measurement over the scaled '
        'prediction, and a table of peak frequency, level and Q measured '
        'and predicted, with their ratio named as an ASD ratio or an '
        "amplitude ratio and the model's stated uncertainty beside it."),
    ReportFeature(
        'cross_axis', 'Cross-axis and limit channels', 'Results',
        'Cross-axis response against its limit, and the band and depth '
        'over which each limit channel held the control below its '
        'reference, as rows of the result table; a notch the plan does '
        'not name is not met, and it and a cross-axis channel over its '
        'limit offer the findings register a fact.',
        requires=('test_requirement',)),
    ReportFeature(
        'control_uncertainty', 'Control-channel uncertainty', 'Results',
        "Each control channel's measurement uncertainty, from its "
        'contributions, against one third of the tolerance: a table under '
        "the requirement's words, a channel with nothing entered not "
        'judged.',
        requires=('test_requirement',)),
    ReportFeature(
        'data_index', 'Data index', 'Records',
        'Every file the report rests on, with what it holds, its format, '
        'size and SHA-256, checked again at export; attached documents '
        'with their checksums, a small text or CSV attachment printed '
        'whole; and the analysis recipe.'),
    ReportFeature(
        'calibration_in_date', 'Calibration in date', 'Records',
        "Whether each channel's sensor was in calibration on the test "
        'date: a column in the channel table, any channel out of date '
        'flagged and named with by how much, and a channel with no date '
        'said to be unchecked. The equipment list is offered to the '
        'setup and method section.'),
    ReportFeature(
        'report_identity', 'Identity and issue', 'Issue control',
        'Report number, revision and laboratory on every printed page '
        'with page n of N, the identification block on the first page, '
        'an end-of-report line, and a status: draft, issued or amended. '
        'Issued as a paginated PDF; a change after issue is an amendment '
        'with its own revision and a table of what changed, never a '
        'silent reissue.'),
    ReportFeature(
        'flash_report', 'Flash report', 'Issue control',
        "The report drawn as one preliminary page from the day's "
        'analysis: the identity marked Preliminary, the headline read '
        'off the result table for each run so far, the findings by '
        'number with their owners, and owed lines for what only the '
        'author can write: the condition on release, the opinions, what '
        'happens next and the report date.',
        requires=('test_requirement', 'report_identity', 'findings')),
    ReportFeature(
        'record_captions', 'Record captions', 'Figures',
        "A caption block on each spectrum, SRS and sine figure, read off "
        "the record: channel, run, analysis settings and requirement.",
        requires=('record_caption',)),
    ReportFeature(
        'plot_conventions', 'Plot conventions', 'Figures',
        'ASD and SRS figures open on log-log axes. Curves are told apart '
        'by line style as well as color, and a figure of up to eight '
        'curves names each one at its end, so it reads in a monochrome '
        'copy.'),
    ReportFeature(
        'shared_decades', 'Shared decades', 'Figures',
        'Every ASD figure in the report opens on the same whole decades, '
        'and every SRS figure on the same decades, so figures of a kind '
        'compare by eye. Figures of different quantities or units are '
        'scaled apart.',
        requires=('plot_conventions',)),
)}


@dataclass(frozen=True)
class Preset:
    """A named set of features to start from."""

    key: str
    title: str
    description: str
    features: tuple[str, ...]


#: the starting points the selection tool offers. Presets name features
#: that are not built yet: applying one enables what is available and
#: says what it skipped.
PRESETS: dict[str, Preset] = {preset.key: preset for preset in (
    Preset('default', 'Visual Dynamics default',
           'No features: the report as Visual Dynamics writes it.', ()),
    Preset('test_report', 'Test report',
           'What a report for a quality signer and a customer needs.',
           ('skeleton_sections', 'test_requirement', 'findings',
            'report_identity', 'data_index', 'run_sections',
            'record_captions', 'plot_conventions', 'shared_decades',
            'check_blocks', 'calibration_in_date')),
    Preset('flash_report', 'Flash report',
           'The one preliminary page a customer leaves the test with.',
           ('report_identity', 'test_requirement', 'skeleton_sections',
            'findings', 'flash_report')),
)}


@dataclass(frozen=True)
class FeatureSelection:
    """What `configure` left enabled, and what it has to say.

    `enabled` is the report's features afterwards, in registry order.
    `notes` says what was skipped or turned off along the way: a
    preset's unavailable features, or a feature turned off because a
    feature it needs was.
    """

    enabled: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())


def unmet(key: str, enabled: Iterable[str] = ()) -> list[str]:
    """What stands between feature `key` and being enabled, in words.

    Empty when nothing does. An unbuilt feature says 'not yet
    available' first and then what it will wait on; a built one names
    each feature that must be enabled (from `enabled`) and each
    capability this installation does not have. Raises `ValueError` for
    a key that is not a registered feature.
    """
    feature = _feature(key)
    on = set(enabled)
    missing = []
    for need in feature.requires:
        if need in FEATURES:
            if need not in on:
                missing.append(f'needs {FEATURES[need].title}')
        elif need in CAPABILITIES:
            if not CAPABILITIES[need].probe():
                missing.append(f'needs {CAPABILITIES[need].label}')
        else:
            missing.append(f'needs {need!r}, which is not known')
    if not feature.available:
        waits = [reason.removeprefix('needs ') for reason in missing]
        return [NOT_YET + (f'; waits on {", ".join(waits)}' if waits
                           else '')]
    return missing


def reason(key: str, enabled: Iterable[str] = ()) -> str:
    """`unmet` as one line for the selection tool: '' when the feature
    can be enabled."""
    return '; '.join(unmet(key, enabled))


def enabled(report: Any) -> tuple[str, ...]:
    """The features that act on `report` when it is rendered.

    Read from `report.features`, kept to the registered, available
    features whose requirements are met, in registry order. A key the
    registry does not know (written by a later version, say) is kept on
    the report but does nothing here.
    """
    stored = set(getattr(report, 'features', ()) or ())
    out: list[str] = []
    for key, feature in FEATURES.items():
        if key in stored and feature.available and not unmet(key, out):
            out.append(key)
    return tuple(out)


def defaults() -> tuple[str, ...]:
    """The features a new report starts with: those whose `default` is
    True. None are, so a new report starts as Visual Dynamics writes it."""
    return tuple(key for key, feature in FEATURES.items()
                 if feature.default and feature.available)


def _feature(key: Any) -> ReportFeature:
    if not isinstance(key, str):
        raise TypeError(f'a report feature is named by its key, a str, '
                        f'not {type(key).__name__}')
    feature = FEATURES.get(key)
    if feature is None:
        raise ValueError(f'unknown report feature {key!r}; the features '
                         f'are: {", ".join(FEATURES)}')
    return feature


def _keys(value: str | Iterable[str], what: str) -> list[str]:
    """`enable` or `disable` as a list of keys; a lone string is one key,
    not its letters."""
    if isinstance(value, str):
        return [value]
    try:
        keys = list(value)
    except TypeError:
        raise TypeError(f'{what} takes feature keys, not '
                        f'{type(value).__name__}') from None
    for key in keys:
        _feature(key)
    return keys


def _ordered(keys: Iterable[str]) -> list[str]:
    on = set(keys)
    return [key for key in FEATURES if key in on]


def configure(report: Any, *, enable: str | Iterable[str] = (),
              disable: str | Iterable[str] = (),
              preset: str | None = None) -> FeatureSelection:
    """Change which features are enabled on `report`.

    With `preset`, the report's features are replaced by the preset's
    available ones first, and every feature the preset names but cannot
    have is skipped with a note. Then `disable` turns features off, and
    a feature that needs one turned off goes with it, with a note. Then
    `enable` turns features on; a feature needing another may be
    enabled in the same call as the one it needs.

    Every request is checked before anything changes, so a refused call
    leaves the report as it was. Refused: an unknown feature or preset,
    a feature that is not yet available, a feature whose requirement is
    not met, and a key both enabled and disabled. Keys the registry
    does not know that are already on the report (from a later version)
    are kept, except when a preset replaces the set.

    Returns
    -------
    FeatureSelection
        The features enabled afterwards and the notes.

    Raises
    ------
    ValueError
        For a refusal; the message names the feature and the reason.
    TypeError
        When a key is not a string.
    """
    turn_on = _keys(enable, 'enable')
    turn_off = _keys(disable, 'disable')
    both = sorted(set(turn_on) & set(turn_off))
    if both:
        raise ValueError(f'{", ".join(map(repr, both))} cannot be enabled '
                         'and disabled in one call')
    notes: list[str] = []
    stored = list(getattr(report, 'features', ()) or ())
    if preset is not None:
        chosen = PRESETS.get(preset)
        if chosen is None:
            raise ValueError(f'unknown preset {preset!r}; the presets '
                             f'are: {", ".join(PRESETS)}')
        current: list[str] = []
        for key in chosen.features:
            if key not in FEATURES:
                notes.append(f'{chosen.title}: {key!r} skipped, not a '
                             'known feature')
                continue
            missing = unmet(key, current)
            if missing:
                notes.append(f'{chosen.title}: {FEATURES[key].title} '
                             f'skipped, {"; ".join(missing)}')
                continue
            current.append(key)
        unknown: list[str] = []
    else:
        current = [key for key in stored if key in FEATURES]
        unknown = [key for key in stored if key not in FEATURES]
    # off, and whatever stood on what went off
    off = set(turn_off)
    changed = True
    while changed:
        changed = False
        for key in current:
            if key in off:
                continue
            gone = [need for need in FEATURES[key].requires if need in off]
            if gone:
                off.add(key)
                changed = True
                # only features are ever turned off, so what went is
                # a feature
                notes.append(f'{FEATURES[key].title} turned off with '
                             f'{FEATURES[gone[0]].title}, which it needs')
    current = [key for key in current if key not in off]
    # on, in the registry's order, so a feature asked for with the one
    # it needs finds that one already on
    for key in _ordered(turn_on):
        if key in current:
            continue
        missing = unmet(key, current)
        if missing:
            raise ValueError(f'{FEATURES[key].title} ({key!r}) cannot be '
                             f'enabled: {"; ".join(missing)}')
        current.append(key)
    result = _ordered(current)
    report.features = result + unknown
    return FeatureSelection(tuple(result), tuple(notes))


def preset_of(keys: Sequence[str]) -> str | None:
    """The preset whose available features are exactly `keys`, or None
    when the set matches none of them (a custom choice)."""
    wanted = set(_ordered(keys))
    for preset in PRESETS.values():
        if wanted == set(configure(_Scratch(), preset=preset.key).enabled):
            return preset.key
    return None


class _Scratch:
    """A stand-in report for asking what a preset would enable."""

    features: list[str]

    def __init__(self) -> None:
        self.features = []
