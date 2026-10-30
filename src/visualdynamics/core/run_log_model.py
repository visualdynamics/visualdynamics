"""The run log read as tests, axes and runs: their states, the holds and
owed lists, the views a report reads, and the projection to full level.

A `RunLog` (`core.run_log`) is a list of entries, each with plain test,
axis and run references. `RunLogModel` reads a log's corrected view and
builds the rest from it, every time it is made: nothing here is stored
in the file, so a hold, an owed line or a credited duration is always
what the entries say now, and a correction changes it at once.

**Tests.** A test is opened by a `test_opened` entry, which carries its
kind (one of `TEST_KINDS`: the project types plus Functional and Other)
and its header (`run_log.HEADER_FIELDS`): what the test is, the article,
the documents and revisions, and the plan's numbers the log checks
against. A `header_change` changes one header field. A test has three
states: **open** (it takes runs), **closed to runs** (when it is closed,
or when the next test opens) and **released** (its release entry, a
post-test step 'release'). A test closed to runs still takes findings,
post-test steps and dispositions; a released test takes no new runs,
findings or header changes.

**The roll** (`roll`) closes the current test to runs and opens the next
with its header carried from the last. Every carried field is marked
carried until it is confirmed (`confirm`) or changed.

**Runs** are numbered within a test, and each sits on one axis. What a
run holds depends on the test's kind: random level rows ending at full
level with the counter, sine sweeps with their direction and count,
shock shots with their four checks and the pulse count against the
budget, and for a survey or functional run its result and who ran it.

**Holds** are what the log asks to be answered before the test goes on:
an abort with no cause, an over-test question unanswered, a restart with
no cause, a "not explained" answer with no later answer, a notch not in
the header's planned list with no deviation, a limit, tolerance or
abort-band change with no deviation, and a go at a hold-point level with
no customer agreement logged. They are shown, never enforced: the log
never drives the controller.

**Owed lists** are checklists with a line each, done by an entry (who
and when) or still owed: the setup before the first low level, the axis
change, the release checklist, an abort's first-minute fields, a shot's
four checks, and a day with no clock check.

**The projection** (`project`) takes readings at a level L in dB
relative to full level (negative) to full level by the factor
10^(-L/20), exactly, and judges the drive fraction against its window,
the peak current at the plan's sigma against the peak rating and the
peak voltage against the rail. A check that cannot be made reads "not
judged", with why, and is never a pass.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from .report import PROJECT_TYPES
from .run_log import (
    HEADER_FIELDS,
    KINDS,
    Entry,
    FileRef,
    RunLog,
    RunLogError,
    band_hz,
    check_fields,
)
from .run_log_export import ALL, _selected

#: the kinds of test: the project types, so a report's template follows
#: from one word, and Functional and Other
TEST_KINDS = (*PROJECT_TYPES, 'Functional', 'Other')

#: what a run of each kind of test holds: level rows, sweeps, shots, or
#: a result and who ran it
RUN_SHAPES = {'Random Vibration': 'levels', 'Sine Sweep': 'sweeps',
              'Shock': 'shots', 'Transient': 'shots',
              'Random and Sine': 'levels and sweeps', 'Modal Test': 'result',
              'System ID': 'result', 'Functional': 'result', 'Other': 'any'}

#: the states of a test, in the order it passes through them
STATES = ('open', 'closed to runs', 'released')

#: the kinds of entry that belong to a run, which a test closed to runs
#: no longer takes
RUN_KINDS = frozenset({'run_start', 'level_row', 'sweep', 'shot',
                       'run_stop', 'abort', 'restart', 'notch'})

#: the kinds of entry that start a run's running; the first of them on
#: an axis is the first low level the setup is owed before
DRIVE_KINDS = frozenset({'run_start', 'level_row', 'sweep', 'shot'})

#: what a released test no longer takes besides runs
RELEASED_REFUSES = frozenset({'deviation', 'anomaly_opened',
                              'header_change'})

#: two levels (or two notch depths) closer than this, in dB, are the
#: same: a level typed as -12.0 and a step of -12 are one step
LEVEL_MATCH_DB = 0.05

#: the drive fraction's window at full level when the header gives
#: none: a third to two thirds of the drive's range, enough to control
#: on and enough left over to correct with
DRIVE_WINDOW = (1.0 / 3.0, 2.0 / 3.0)

#: a change in the ratio of volts to amps between levels, or against
#: the last run of the same setup, larger than this fraction is flagged.
#: A reminder threshold, not a rule: a plant whose ratio moved has
#: changed, and the size worth a look is the lab's to set.
VA_CHANGE = 0.10

#: the header fields whose change once the test is running is a change
#: to the test, which a deviation answers
PLAN_CHANGE_FIELDS = ('limits', 'tolerance', 'abort_bands',
                      'duration_minus_pct', 'duration_plus_pct',
                      'notches_planned', 'hold_points_db')

#: the setting classes whose change is a change to the test
_PLAN_SETTINGS = ('limit', 'tolerance', 'abort band')

#: words that mark a setting change as a limit, tolerance or abort-band
#: change when the entry does not say what the setting governs
_PLAN_WORDS = ('limit', 'toleran', 'abort')

#: every hold rule by its key, with what it asks
HOLD_RULES = {
    'abort-cause': 'an abort with no cause written',
    'over-test': 'an abort whose over-test question is unanswered',
    'restart-cause': 'a restart with no cause written',
    'not-explained': 'a "not explained" answer with no later answer',
    'unplanned-notch': 'a notch not in the planned list, with no deviation',
    'plan-change': 'a limit, tolerance or abort-band change, with no '
                   'deviation',
    'hold-point': 'a go at a hold-point level with no customer agreement '
                  'logged',
}

#: the setup owed before a test's first low level, each line a setup
#: step: (key, what the line asks)
BEFORE_LOW_LEVEL = (
    ('receiving inspection', ('The article\'s receiving inspection, with '
     'its photographs')),
    ('configuration', 'The configuration checked on the article'),
    ('baseline functional', 'The article\'s baseline operational checkout'),
    ('mounting record', ('The mounting record: who mounted, the bolts with '
     'their torques and witness marks')),
    ('end-to-end check', ('The end-to-end check through its sixth step: '
     'taps, polarity and noise floors by channel')),
    ('surveys', 'The surveys: bare, loaded, and the pre-test signature'),
    ('photographs', 'The photographs of the setup'),
    ('second person', 'A second person\'s inspection of the setup'),
)

#: the change from one axis to the next, in order, each line a setup
#: step on the new axis
AXIS_CHANGE = (
    ('axis closed', ('The finished axis closed: post-test signature, data '
     'check and archive, functional, the attendance the plan names')),
    ('hardware', ('The hardware changed from its bolt lists: article, '
     'fixture, shaker rotation, slip table mated, aligned and oiled, the '
     'torque record')),
    ('controller', ('The controller: setup file and revision, shaker '
     'profile, moving mass')),
    ('channel table', ('The channel table: each triaxial\'s axis and '
     'polarity, the control sensor, cross-axis names, read back')),
    ('end-to-end check', ('The end-to-end check: tap and polarity by '
     'channel, noise floors')),
    ('surveys', 'The surveys: bare, loaded, pre-test signature'),
    ('photographs', 'New photographs, the axis readable in each'),
    ('second person', 'The second person\'s look before the first drive'),
)

#: the release checklist, in order, each line a post-test step. The line
#: 'release' releases the test.
RELEASE = (
    ('recorders stopped', ('The last run ended, the recorders stopped only '
     'after the amplifier was disabled')),
    ('signature compared', ('The post-test signature compared against its '
     'criterion')),
    ('data check', ('The data check on every channel, the files written, '
     'opened and archived')),
    ('post-test functional', ('The post-test operational checkout against '
     'the baseline')),
    ('inspection', 'The inspection to the procedure'),
    ('equipment inspected', ('The fixture, exciter and instrumentation '
     'inspected')),
    ('anomalies opened', 'Every anomaly photographed and opened'),
    ('post-test review', 'The post-test review held'),
    ('release', ('The article and the facility released, and the setup '
     'free to be broken')),
    ('article removed', ('The article removed and inspected, and its '
     'condition as it left')),
    ('machine log', 'The machine log updated'),
    ('flash report', 'The flash report sent'),
)

#: the four checks logged after every shot: (key, the shot's fields
#: that make it, what it asks)
SHOT_CHECKS = (
    ('tolerance', ('in_tolerance',), ('The pulse or spectrum against its '
     'tolerance, in both polarities')),
    ('stroke', ('within_stroke',), ('The displacement and velocity against '
     'the stroke')),
    ('amplifier', ('peak_current', 'peak_voltage'), ('The amplifier\'s '
     'current and voltage peaks')),
    ('prediction', ('prediction',), ('The controller\'s prediction for the '
     'next level')),
)


# ---- small readers -------------------------------------------------------------


def _norm(text: Any) -> str:
    if not isinstance(text, str):
        return ''
    return ' '.join(text.lower().replace('_', ' ').replace('-', ' ').split())


def _line_of(lines: Sequence[tuple[str, str]], step: Any) -> str | None:
    """The key of the checklist line `step` names, by its key or its
    words, ignoring case, hyphens and spacing; None for none."""
    wanted = _norm(step)
    for key, label in lines:
        if wanted in (_norm(key), _norm(label)):
            return key
    return None


def _same_level(a: Any, b: Any) -> bool:
    return (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and abs(a - b) <= LEVEL_MATCH_DB)


def notch_in_plan(planned: Sequence[Mapping[str, Any]] | None,
                  axis: str | None, fields: Mapping[str, Any]
                  ) -> tuple[bool, str]:
    """Whether a notch with `fields` (its band, and its depth_db where
    given) on `axis` is in the planned list `planned` (a header's
    `notches_planned`), and why: on its axis (a planned notch with no
    axis is on every axis), its band inside the planned band, and its
    depth no deeper than the planned depth where the plan gives one.
    What `RunLogModel.notch_planned` judges a written notch by, and what
    an entry form shows before the notch is written. A typed `planned`
    mark is not what decides it."""
    planned = planned or []
    marked = (' It is marked planned, but the header\'s list decides.'
              if fields.get('planned') else '')
    if 'band' not in fields:
        return False, 'the notch gives no band, so it cannot be found in ' \
                      'the planned list.' + marked
    try:
        low, high = band_hz(fields['band'], 'its band')
    except RunLogError as error:
        return False, f'{error}.' + marked
    if not planned:
        return False, 'the header plans no notches.' + marked
    reasons = []
    for number, notch in enumerate(planned, start=1):
        if notch.get('axis') is not None and notch['axis'] != axis:
            continue
        try:
            p_low, p_high = band_hz(notch['band'])
        except RunLogError:
            continue
        if low < p_low or high > p_high:
            continue
        depth = notch.get('depth_db')
        if depth is not None:
            if 'depth_db' not in fields:
                reasons.append(f'it is inside planned notch {number}, '
                               'which has a depth, but it gives none')
                continue
            if abs(fields['depth_db']) > abs(depth) + LEVEL_MATCH_DB:
                reasons.append(f'it is inside planned notch {number} but '
                               f'deeper than its {abs(depth):g} dB')
                continue
        return True, f'inside planned notch {number}.'
    why = '; '.join(reasons) if reasons else (
        f'{low:g} to {high:g} Hz on axis {axis or "(none)"} is in no planned '
        'notch')
    return False, why + '.' + marked


def full_level_factor(level_db: float) -> float:
    """The factor that takes a reading at `level_db` to full level:
    10^(-L/20), with L in dB relative to full level (-12 for twelve
    below, which gives 3.98).

    Refused with `RunLogError`: a level that is not a finite number, and
    a level above full (positive), which is not projected down.
    """
    if isinstance(level_db, bool) or not isinstance(level_db, (int, float)):
        raise RunLogError(f'a level is a number of dB relative to full level '
                          f'(-12 for twelve below), not {level_db!r}')
    if not math.isfinite(level_db):
        raise RunLogError(f'a level of {level_db} dB is not a finite number')
    if level_db > 0:
        raise RunLogError(f'a level of {level_db:+g} dB is above full level; '
                          'a level is given in dB relative to full level, '
                          'negative below it (-12 for twelve below)')
    return 10.0 ** (-level_db / 20.0)


def parse_counter(text: Any) -> float:
    """A controller's counter as seconds: '750', '750 s', '12:30' (minutes
    and seconds) or '1:02:30' (hours, minutes and seconds), with
    decimals on the seconds.

    Refused with `RunLogError` naming the text: anything else, and
    minutes or seconds of 60 or more after the first part.
    """
    if isinstance(text, bool) or not isinstance(text, (str, int, float)):
        raise RunLogError(f'a counter is a time such as 12:30, not {text!r}')
    raw = str(text).strip().lower()
    if raw.endswith('s'):
        raw = raw[:-1].rstrip()
    parts = raw.split(':')
    if not 1 <= len(parts) <= 3 or not all(
            re.fullmatch(r'\d+(\.\d+)?', part) for part in parts) or any(
            '.' in part for part in parts[:-1]):
        raise RunLogError(f'the counter {text!r} is not read as a time: give '
                          'seconds, minutes:seconds or hours:minutes:seconds')
    numbers = [float(part) for part in parts]
    if any(value >= 60 for value in numbers[1:]):
        raise RunLogError(f'the counter {text!r} has minutes or seconds of 60 '
                          'or more')
    seconds = 0.0
    for value in numbers:
        seconds = seconds * 60.0 + value
    return seconds


def clock_text(seconds: float) -> str:
    """Seconds as the counter reads them: m:ss, or h:mm:ss from an hour,
    to the second."""
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes}:{secs:02d}'


# ---- the projection ------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One headroom number at full level: what it is, its value at full
    level, the limit it is judged against, and the verdict ('within',
    'over', 'under' or 'not judged') with words that say why."""

    name: str
    at_full: float | None
    limit: float | tuple[float, float] | None
    verdict: str
    message: str

    @property
    def judged(self) -> bool:
        """Whether the check was made; one that was not is never a
        pass."""
        return self.verdict != 'not judged'

    def __str__(self) -> str:
        return f'{self.name}: {self.message}'


@dataclass(frozen=True)
class Projection:
    """Readings at a level projected to full level, and the headroom
    checks. `verdict` is 'within' only when every check was judged
    within; 'outside' when any is over or under; otherwise 'not
    judged'."""

    level_db: float
    factor: float
    voltage: float | None
    current: float | None
    drive_fraction: float | None
    checks: tuple[Check, ...]

    @property
    def verdict(self) -> str:
        """'within', 'outside' or 'not judged', as the class says."""
        verdicts = {check.verdict for check in self.checks}
        if verdicts & {'over', 'under'}:
            return 'outside'
        if 'not judged' in verdicts or not verdicts:
            return 'not judged'
        return 'within'

    def check(self, name: str) -> Check:
        """The check named `name` ('drive fraction', 'peak current',
        'rms current', 'peak voltage')."""
        for found in self.checks:
            if found.name == name:
                return found
        raise KeyError(f'no {name!r} check; the checks are: '
                       + ', '.join(check.name for check in self.checks))

    def __str__(self) -> str:
        head = (f'{self.level_db:+g} dB, factor {self.factor:.4g} to full '
                f'level: {self.verdict}')
        return '\n'.join([head, *(f'  {check}' for check in self.checks)])


def _reading(value: Any, what: str, *, at_most: float | None = None
             ) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunLogError(f'{what} is a number, not {value!r}')
    if not math.isfinite(value):
        raise RunLogError(f'{what} is not a finite number ({value!r})')
    if value < 0:
        raise RunLogError(f'{what} of {value:g} is negative')
    if at_most is not None and value > at_most:
        raise RunLogError(f'{what} of {value:g} is above {at_most:g}')
    return float(value)


def _rating(value: Any, what: str) -> float | None:
    value = _reading(value, what)
    if value == 0:
        raise RunLogError(f'{what} of zero is not a rating')
    return value


def _judge_peak(name: str, reading: float | None, read_what: str,
                factor: float, crest: float | None, crest_what: str,
                limit: float | None, limit_what: str, unit: str) -> Check:
    missing = []
    if reading is None:
        missing.append(f'no {read_what} was read')
    elif reading == 0:
        missing.append(f'the {read_what} reads zero, which says the meter or '
                       'the amplifier is not reading')
    if crest is None:
        missing.append(f'the header gives no {crest_what}')
    if limit is None:
        missing.append(f'the header gives no {limit_what}')
    if missing:
        return Check(name, None, limit, 'not judged',
                     'not judged: ' + '; '.join(missing))
    assert reading is not None and crest is not None and limit is not None
    at_full = reading * factor * crest
    verdict = 'over' if at_full > limit else 'within'
    word = 'above' if verdict == 'over' else 'within'
    return Check(name, at_full, limit, verdict,
                 f'{at_full:.4g} {unit} at full level, {word} the '
                 f'{limit_what} of {limit:.4g} {unit} '
                 f'({at_full / limit:.0%})')


def project(level_db: float, *, voltage: float | None = None,
            current: float | None = None,
            drive_fraction: float | None = None,
            sigma: float | None = None,
            rated_current_peak: float | None = None,
            rated_current_rms: float | None = None,
            rail_voltage: float | None = None,
            drive_window: Sequence[float] | None = None) -> Projection:
    """Readings at a level projected to full level, with the headroom
    checks.

    Each reading is multiplied by `full_level_factor` (10^(-L/20),
    exactly). The drive fraction at full level is judged against its
    window; the rms current times `sigma` (the peak a random drive
    reaches, at the plan's sigma) against the peak current rating; the
    rms voltage times `sigma` against the rail; and, when the header
    gives a continuous rating, the rms current against it, which is
    what a sine test budgets.

    Parameters
    ----------
    level_db : float
        The level the readings were taken at, in dB relative to full
        level: -12 for twelve below. Not above 0.
    voltage, current : float, optional
        The amplifier's rms voltage (V) and current (A) at the level.
    drive_fraction : float, optional
        The drive as a fraction of its range, 0 to 1.
    sigma : float, optional
        The plan's peak-to-rms ratio of the drive.
    rated_current_peak, rated_current_rms : float, optional
        The amplifier's peak and continuous current ratings (A).
    rail_voltage : float, optional
        The amplifier's rail, the peak voltage it can give (V).
    drive_window : (float, float), optional
        The drive fraction's window at full level; `DRIVE_WINDOW` when
        not given.

    Returns
    -------
    Projection
        The factor, the readings, and each check with its verdict. A
        check that cannot be made (a reading or a rating missing, or a
        reading of zero) reads "not judged", with why.

    Raises
    ------
    RunLogError
        A level above full or not finite, a reading that is negative or
        not finite, a drive fraction above 1, a sigma or rating that is
        zero, and a window that is not two fractions low then high.
    """
    factor = full_level_factor(level_db)
    voltage = _reading(voltage, 'a voltage')
    current = _reading(current, 'a current')
    drive_fraction = _reading(drive_fraction, 'a drive fraction', at_most=1.0)
    sigma = _rating(sigma, 'a sigma')
    rated_current_peak = _rating(rated_current_peak, 'a peak current rating')
    rated_current_rms = _rating(rated_current_rms,
                                'a continuous current rating')
    rail_voltage = _rating(rail_voltage, 'a rail voltage')
    defaulted = drive_window is None
    window = DRIVE_WINDOW if drive_window is None else tuple(drive_window)
    if len(window) != 2 or not all(
            isinstance(edge, (int, float)) and not isinstance(edge, bool)
            and math.isfinite(edge) for edge in window) or not (
            0 <= window[0] < window[1] <= 1):
        raise RunLogError(f'a drive window is two fractions of the range, low '
                          f'then high, between 0 and 1, not {drive_window!r}')
    low, high = float(window[0]), float(window[1])
    checks = []
    named = 'the default window' if defaulted else 'the window'
    if drive_fraction is None:
        checks.append(Check('drive fraction', None, (low, high), 'not judged',
                            'not judged: no drive fraction was read'))
    elif drive_fraction == 0:
        checks.append(Check('drive fraction', None, (low, high), 'not judged',
                            'not judged: the drive fraction reads zero, which '
                            'says the controller is not driving'))
    else:
        at_full = drive_fraction * factor
        verdict = ('over' if at_full > high else
                   'under' if at_full < low else 'within')
        word = {'over': 'above', 'under': 'below', 'within': 'within'}[verdict]
        checks.append(Check('drive fraction', at_full, (low, high), verdict,
                            f'{at_full:.3g} of the range at full level, {word} '
                            f'{named} {low:.3g} to {high:.3g}'))
    checks.append(_judge_peak('peak current', current, 'current', factor,
                              sigma, 'sigma', rated_current_peak,
                              'peak current rating', 'A'))
    if rated_current_rms is not None:
        checks.append(_judge_peak('rms current', current, 'current', factor,
                                  1.0, '', rated_current_rms,
                                  'continuous current rating', 'A'))
    checks.append(_judge_peak('peak voltage', voltage, 'voltage', factor,
                              sigma, 'sigma', rail_voltage, 'rail', 'V'))
    return Projection(float(level_db), factor, voltage, current,
                      drive_fraction, tuple(checks))


@dataclass(frozen=True)
class Comparison:
    """A level row compared with an earlier one: the earlier entry, what
    it is (the previous level, or the last run of the same setup), and
    the ratios of the readings now to then, each None where either
    reading is missing or zero. `flagged` names each ratio that moved
    more than `VA_CHANGE`."""

    against: Entry
    what: str
    ratios: Mapping[str, float | None]
    flagged: tuple[str, ...]

    def __str__(self) -> str:
        parts = [f'{name} x{ratio:.3g}' if ratio is not None else
                 f'{name} not compared' for name, ratio in self.ratios.items()]
        mark = (' FLAGGED: ' + ', '.join(self.flagged)) if self.flagged else ''
        return f'against {self.what} (#{self.against.id}): ' \
               + ', '.join(parts) + mark


@dataclass(frozen=True)
class LevelReading:
    """A level row read against its test's header: the projection to full
    level (None when it cannot be made, `message` saying why), the
    volts-per-amp ratio against the previous level, and the comparison
    with the last run of the same setup when the log holds one."""

    entry: Entry
    projection: Projection | None
    message: str
    previous: Comparison | None
    last_run: Comparison | None

    @property
    def verdict(self) -> str:
        """The projection's verdict, or 'not judged' without one."""
        return self.projection.verdict if self.projection else 'not judged'


def _ratio(now: Any, then: Any) -> float | None:
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and value > 0 for value in (now, then)):
        return None
    return now / then


def _va(fields: Mapping[str, Any]) -> float | None:
    return _ratio(fields.get('voltage'), fields.get('current'))


# ---- what the model holds ------------------------------------------------------


@dataclass(frozen=True)
class TestHeader:
    """A test's header as it stands at some entry: the value of each
    field, the entry that set it, and the fields carried from the last
    test and not yet confirmed or changed."""

    values: Mapping[str, Any]
    set_by: Mapping[str, int]
    carried: tuple[str, ...]

    def get(self, name: str, default: Any = None) -> Any:
        """The value of header field `name`, or `default`."""
        return self.values.get(name, default)

    def is_carried(self, name: str) -> bool:
        """Whether field `name` is carried and not yet confirmed."""
        return name in self.carried


@dataclass(frozen=True)
class Run:
    """One run of a test: its number, its axis, what its kind of test
    makes it hold (`shape`), and its entries in the order written."""

    test: int
    number: int
    axis: str | None
    shape: str
    entries: tuple[Entry, ...]

    def _kind(self, kind: str) -> tuple[Entry, ...]:
        return tuple(entry for entry in self.entries if entry.kind == kind)

    @property
    def levels(self) -> tuple[Entry, ...]:
        """The level rows, in the order written."""
        return self._kind('level_row')

    @property
    def sweeps(self) -> tuple[Entry, ...]:
        """The sweeps, in the order written."""
        return self._kind('sweep')

    @property
    def shots(self) -> tuple[Entry, ...]:
        """The shots, in the order written."""
        return self._kind('shot')

    @property
    def start(self) -> Entry | None:
        """The run's first start entry, or None."""
        starts = self._kind('run_start')
        return starts[0] if starts else None

    @property
    def stop(self) -> Entry | None:
        """The run's last stop or abort, or None."""
        stops = [entry for entry in self.entries
                 if entry.kind in ('run_stop', 'abort')]
        return stops[-1] if stops else None

    @property
    def outcome(self) -> str:
        """'aborted' (an abort since its last start or restart, whether or
        not a run stop was written beside it), 'stopped' (a scheduled
        stop), 'completed', or 'running' when no stop follows its last
        start or restart."""
        since: list[Entry] = []
        for entry in self.entries:
            if entry.kind in ('run_start', 'restart'):
                since = []
            elif entry.kind in ('run_stop', 'abort'):
                since.append(entry)
        if not since:
            return 'running'
        if any(entry.kind == 'abort' for entry in since):
            return 'aborted'
        return 'stopped' if since[-1].fields.get('scheduled') else 'completed'

    @property
    def reached_full(self) -> bool:
        """Whether any entry of the run gives full level (0 dB)."""
        return any(_same_level(entry.fields.get('level_db'), 0.0)
                   for entry in self.entries)

    @property
    def counter(self) -> str | None:
        """The counter at the run's last stop, as written, or None."""
        stop = self.stop
        return stop.fields.get('counter') if stop is not None else None

    @property
    def result(self) -> str | None:
        """A survey or functional run's result, from its last stop."""
        stop = self.stop
        return stop.fields.get('result') if stop is not None else None

    @property
    def by(self) -> str | None:
        """Who ran a survey or functional run, from its last stop."""
        stop = self.stop
        return stop.fields.get('by') if stop is not None else None

    def __str__(self) -> str:
        axis = f', axis {self.axis}' if self.axis else ''
        return f'test {self.test}, run {self.number}{axis}: {self.outcome}'


@dataclass(frozen=True)
class Test:
    """One test as the log holds it: its number and kind, the entries
    that opened, closed and released it, its header as it stands, its
    axes in the order first named, its runs, its entries in the order
    written, and what in them does not fit its states (`problems`)."""

    number: int
    kind: str
    opened: Entry | None
    closed: Entry | None
    released: Entry | None
    header: TestHeader
    axes: tuple[str, ...]
    runs: tuple[Run, ...]
    entries: tuple[Entry, ...]
    problems: tuple[str, ...]

    @property
    def title(self) -> str:
        """The header's title, or 'Test N'."""
        return self.header.get('title') or f'Test {self.number}'

    @property
    def state(self) -> str:
        """'open', 'closed to runs' or 'released' (`STATES`)."""
        if self.released is not None:
            return 'released'
        if self.closed is not None:
            return 'closed to runs'
        return 'open'

    @property
    def shape(self) -> str:
        """What its runs hold (`RUN_SHAPES`)."""
        return RUN_SHAPES[self.kind]

    def run(self, number: int) -> Run:
        """Run `number` of the test."""
        for run in self.runs:
            if run.number == number:
                return run
        raise KeyError(f'test {self.number} has no run {number}')


@dataclass(frozen=True)
class Hold:
    """A hold: its rule (a key of `HOLD_RULES`), its test, the entry that
    raised it and what it asks."""

    rule: str
    test: int | None
    entry: Entry
    message: str

    def __str__(self) -> str:
        where = f'test {self.test}, ' if self.test is not None else ''
        return f'HOLD ({where}#{self.entry.id}): {self.message}'


@dataclass(frozen=True)
class OwedLine:
    """One line of an owed list: its key, what it asks, the entry that
    did it (None while owed), and whether it was done after the list
    was due."""

    key: str
    label: str
    done: Entry | None = None
    late: bool = False

    @property
    def owed(self) -> bool:
        """Whether the line is still owed."""
        return self.done is None

    def __str__(self) -> str:
        if self.done is None:
            return f'[ ] {self.label}'
        mark = ' (after it was due)' if self.late else ''
        return (f'[x] {self.label}: {self.done.author}, '
                f'{self.done.event_time:%Y-%m-%d %H:%M}{mark}')


@dataclass(frozen=True)
class OwedList:
    """An owed list: its name, its title, its test and axis, its lines in
    order, and the entry it was due before (None when it has no due
    point)."""

    name: str
    title: str
    test: int | None
    axis: str | None
    lines: tuple[OwedLine, ...]
    due: Entry | None = None

    @property
    def owed(self) -> tuple[OwedLine, ...]:
        """The lines still owed."""
        return tuple(line for line in self.lines if line.owed)

    @property
    def complete(self) -> bool:
        """Whether every line is done."""
        return not self.owed

    def __str__(self) -> str:
        return '\n'.join([(f'{self.title}: {len(self.owed)} of '
                          f'{len(self.lines)} owed'),
                          *(f'  {line}' for line in self.lines)])


@dataclass(frozen=True)
class Interruption:
    """A stop the test did not plan to run through, or a scheduled one:
    the entry (an abort, or a scheduled run stop), its kind ('abort',
    'trip' or 'scheduled'), and the cause, path and restart that answer
    it, each None until logged."""

    entry: Entry
    kind: str
    cause: Entry | None
    path: Entry | None
    restart: Entry | None

    def __str__(self) -> str:
        cause = self.cause.fields.get('cause') if self.cause else 'no cause'
        return f'#{self.entry.id} {self.kind}: {cause}'


@dataclass(frozen=True)
class Credit:
    """The time credited on one axis of a test, from the counters, against
    the header's duration and its tolerance: the seconds credited,
    whether every segment could be read (`complete`), why not
    (`notes`), the duration and its bounds, and the verdict ('within',
    'short', 'over' or 'not judged') with words that say why."""

    test: int
    axis: str | None
    seconds: float
    complete: bool
    notes: tuple[str, ...]
    duration: float | None
    lower: float | None
    upper: float | None
    verdict: str
    message: str

    def __str__(self) -> str:
        axis = f', axis {self.axis}' if self.axis else ''
        return f'test {self.test}{axis}: {self.message}'


@dataclass(frozen=True)
class PulseCount:
    """The pulses a shock test has given, by axis and in all, against the
    header's budget: 'within', 'over' or 'not judged', with words."""

    test: int
    count: int
    by_axis: Mapping[str | None, int]
    budget: int | None
    verdict: str
    message: str


def judge_pulses(test: int, by_axis: Mapping[str | None, int],
                 budget: int | None) -> PulseCount:
    """The pulses counted by axis (`by_axis`) judged against `budget`, a
    test header's `pulse_budget`: what `RunLogModel.pulses` reports, and
    what a shot form shows with the shot being typed counted in."""
    by_axis = dict(by_axis)
    count = sum(by_axis.values())
    if budget is None:
        return PulseCount(test, count, by_axis, None, 'not judged',
                          f'{count} pulses; not judged: the header gives '
                          'no pulse budget')
    if count > budget:
        return PulseCount(test, count, by_axis, budget, 'over',
                          f'{count} pulses, over the budget of {budget} '
                          f'by {count - budget}')
    return PulseCount(test, count, by_axis, budget, 'within',
                      f'{count} of {budget} pulses, {budget - count} left')


@dataclass(frozen=True)
class Finding:
    """A deviation or anomaly: its number, kind ('deviation' or
    'anomaly'), the entry that opened it (corrected), its test, and the
    dispositions written against it."""

    number: str
    kind: str
    entry: Entry
    test: int | None
    dispositions: tuple[Entry, ...]

    @property
    def open(self) -> bool:
        """Whether an anomaly is still open (no disposition); a deviation
        is never held open waiting for an approval."""
        return self.kind == 'anomaly' and not self.dispositions

    @property
    def owner(self) -> str | None:
        """An anomaly's owner, as typed."""
        return self.entry.fields.get('owner')

    @property
    def fact(self) -> str | None:
        """The finding's fact, as typed at the console."""
        return self.entry.fields.get('fact')


@dataclass(frozen=True)
class FileRow:
    """A file an entry names: the entry and the file."""

    entry: Entry
    file: FileRef


@dataclass(frozen=True)
class ChronologyRow:
    """An entry in the order of events, with its time on the recorders'
    clock (None with no clock check before it) and whether it is
    late."""

    entry: Entry
    recorder_time: datetime | None
    late: bool


# ---- the model -----------------------------------------------------------------


class RunLogModel:
    """A run log read as tests, axes and runs, with its holds, owed lists
    and views. Built from the log's corrected view when made, and never
    stored: make a new one after the log changes.

    Parameters
    ----------
    log : RunLog
        The log, open for writing or read-only.
    """

    def __init__(self, log: RunLog) -> None:
        self.log: RunLog = log
        self.entries: tuple[Entry, ...] = tuple(log.view())
        self._by_id = {entry.id: entry for entry in self.entries}
        self._tests: dict[int, Test] = {}
        self._build()

    # ---- building -------------------------------------------------------------

    def _build(self) -> None:
        by_test: dict[int, list[Entry]] = {}
        opened: dict[int, Entry] = {}
        problems: dict[int, list[str]] = {}
        for entry in self.entries:
            if entry.test is None:
                continue
            by_test.setdefault(entry.test, []).append(entry)
            problems.setdefault(entry.test, [])
            if entry.kind == 'test_opened':
                if entry.test in opened:
                    problems[entry.test].append(
                        f'#{entry.id} opens test {entry.test} again; the '
                        f'first opening, #{opened[entry.test].id}, stands')
                else:
                    opened[entry.test] = entry
        openings = sorted(opened.values(), key=lambda entry: entry.id)
        for number, entries in by_test.items():
            first = opened.get(number)
            notes = problems[number]
            if first is None:
                notes.append(f'test {number} has entries but was never opened')
            closed = next((entry for entry in entries
                           if entry.kind == 'test_closed'), None)
            if first is not None:
                later = next((entry for entry in openings
                              if entry.id > first.id
                              and entry.test != number), None)
                if later is not None and (closed is None
                                          or later.id < closed.id):
                    closed = later
            released = next((entry for entry in entries
                             if entry.kind == 'post_test_step'
                             and _line_of(RELEASE, entry.fields.get('step'))
                             == 'release'), None)
            kind = 'Other'
            given = first.fields.get('test_kind') if first else None
            if given in TEST_KINDS:
                kind = given
            elif first is not None:
                notes.append(f'test {number}\'s kind '
                             + (f'{given!r} is not one of: ' if given else
                                'is not given; it is one of: ')
                             + ', '.join(TEST_KINDS) + '. Read as Other.')
            for entry in entries:
                if entry.kind in RUN_KINDS and closed is not None \
                        and entry.id > closed.id:
                    notes.append(f'#{entry.id} ({entry.label}) was written '
                                 f'after test {number} was closed to runs')
                elif released is not None and entry.id > released.id and (
                        entry.kind in RUN_KINDS | RELEASED_REFUSES):
                    notes.append(f'#{entry.id} ({entry.label}) was written '
                                 f'after test {number} was released')
            axes: list[str] = []
            for entry in entries:
                if entry.axis is not None and entry.axis not in axes:
                    axes.append(entry.axis)
            self._tests[number] = Test(
                number=number, kind=kind, opened=first, closed=closed,
                released=released, header=self._header(entries, first),
                axes=tuple(axes), runs=self._runs(number, entries, kind),
                entries=tuple(entries), problems=tuple(notes))

    @staticmethod
    def _header(entries: Sequence[Entry], opened: Entry | None,
                before: int | None = None) -> TestHeader:
        values: dict[str, Any] = {}
        set_by: dict[str, int] = {}
        carried: list[str] = []
        if opened is not None:
            for name, value in opened.fields.items():
                if name in HEADER_FIELDS:
                    values[name] = value
                    set_by[name] = opened.id
            carried = [name for name in opened.fields.get('carried', [])
                       if name in values]
        for entry in entries:
            if before is not None and entry.id >= before:
                break
            if entry.kind != 'header_change' or (
                    opened is not None and entry.id < opened.id):
                continue
            name = entry.fields.get('field')
            if name not in HEADER_FIELDS or 'value' not in entry.fields:
                continue
            values[name] = entry.fields['value']
            set_by[name] = entry.id
            if name in carried:
                carried.remove(name)
        return TestHeader(values, set_by, tuple(carried))

    @staticmethod
    def _runs(test: int, entries: Sequence[Entry], kind: str
              ) -> tuple[Run, ...]:
        grouped: dict[int, list[Entry]] = {}
        for entry in entries:
            if entry.run is not None:
                grouped.setdefault(entry.run, []).append(entry)
        runs = []
        for number, held in grouped.items():
            axis = next((entry.axis for entry in held
                         if entry.axis is not None), None)
            runs.append(Run(test, number, axis, RUN_SHAPES[kind], tuple(held)))
        return tuple(runs)

    # ---- tests ----------------------------------------------------------------

    def tests(self) -> list[Test]:
        """Every test, in the order first opened (or first named)."""
        return sorted(self._tests.values(),
                      key=lambda test: (test.opened or test.entries[0]).id)

    def test(self, number: int) -> Test:
        """Test `number`; `RunLogError` when the log has none."""
        if number not in self._tests:
            raise RunLogError(f'the log has no test {number}')
        return self._tests[number]

    @property
    def current(self) -> Test | None:
        """The test open to runs, the latest opened; None when none is."""
        open_tests = [test for test in self.tests() if test.state == 'open']
        return open_tests[-1] if open_tests else None

    def next_test(self) -> int:
        """The number the next test opened takes."""
        return max(self._tests, default=0) + 1

    def next_run(self, test: int) -> int:
        """The number the next run of test `test` takes."""
        return max((run.number for run in self.test(test).runs),
                   default=0) + 1

    def header_at(self, entry: Entry) -> TestHeader:
        """The header of `entry`'s test as it stood when `entry` was
        written: what a notch, a go or a reading is judged against, so a
        header changed later does not change the judgment."""
        test = self.test(entry.test) if entry.test is not None else None
        if test is None:
            return TestHeader({}, {}, ())
        return self._header(test.entries, test.opened, before=entry.id)

    def _selected_tests(self, test: Any) -> list[Test]:
        _selected([], test)                   # refuses what is not a test
        if test == ALL:
            return self.tests()
        if test is None:
            return []
        return [self.test(test)]

    def _selected_entries(self, test: Any) -> list[Entry]:
        return _selected(list(self.entries), test)

    def runs(self, test: Any = ALL) -> list[Run]:
        """The runs of one test (`test` its number) or of every test
        (`ALL`), test by test."""
        return [run for found in self._selected_tests(test)
                for run in found.runs]

    def _run_of(self, test: Test, entry: Entry) -> Run | None:
        if entry.run is None:
            return None
        return next((run for run in test.runs if run.number == entry.run),
                    None)

    def _axis_of(self, test: Test, entry: Entry) -> str | None:
        if entry.axis is not None:
            return entry.axis
        run = self._run_of(test, entry)
        return run.axis if run is not None else None

    # ---- answers --------------------------------------------------------------

    def _link(self, test: Test, targets: Iterable[Entry],
              kind: str) -> dict[int, Entry]:
        """The first entry of `kind` answering each target: one that names
        it (`refers_to`) wherever it is in the log, or, naming none, one
        in the same test written after it, which answers the latest
        target still unanswered before it."""
        targets = list(targets)
        wanted = {target.id for target in targets}
        found: dict[int, Entry] = {}
        for entry in self.entries:
            if entry.kind == kind and entry.fields.get('refers_to') in wanted:
                found.setdefault(entry.fields['refers_to'], entry)
        waiting: list[int] = []
        for entry in test.entries:
            if entry.id in wanted:
                waiting.append(entry.id)
            elif entry.kind == kind and 'refers_to' not in entry.fields:
                open_ones = [target for target in waiting
                             if target not in found]
                if open_ones:
                    found[open_ones[-1]] = entry
        return found

    def _deviated(self, entry: Entry) -> Entry | None:
        # A deviation answers a hold only when it names the entry: a
        # deviation is a change to the test, and which change it covers
        # is not guessed from where it was written.
        return next((found for found in self.entries
                     if found.kind == 'deviation'
                     and found.fields.get('refers_to') == entry.id), None)

    # ---- notches --------------------------------------------------------------

    def notch_planned(self, entry: Entry) -> tuple[bool, str]:
        """Whether notch `entry` is in its test's planned list as the
        header stood when it was written, and why: on its axis (a planned
        notch with no axis is on every axis), its band inside the planned
        band, and its depth no deeper than the planned depth where the
        plan gives one. The typed `planned` mark is not what decides it."""
        entry = self._by_id.get(entry.id, entry)
        return notch_in_plan(self.header_at(entry).get('notches_planned'),
                             entry.axis, entry.fields)

    # ---- holds ----------------------------------------------------------------

    def holds(self, test: Any = ALL) -> list[Hold]:
        """The holds of one test or every test, in the order of the
        entries that raised them (`HOLD_RULES`)."""
        found: list[Hold] = []
        for each in self._selected_tests(test):
            found.extend(self._holds_of(each))
        return sorted(found, key=lambda hold: hold.entry.id)

    def _holds_of(self, test: Test) -> list[Hold]:
        holds: list[Hold] = []
        number = test.number
        aborts = [entry for entry in test.entries if entry.kind == 'abort']
        causes = self._link(test, aborts, 'cause_found')
        for abort in aborts:
            if abort.id not in causes:
                holds.append(Hold('abort-cause', number, abort,
                                  f'the {abort.fields["stop"]} at #{abort.id} '
                                  'has no cause written; nothing restarts '
                                  'until it has'))
            if 'over_test' not in abort.fields:
                holds.append(Hold('over-test', number, abort,
                                  f'the over-test question for #{abort.id} is '
                                  'unanswered: did the article see more than '
                                  'the test? Answer it from the records, '
                                  'named'))
        first_run = next((entry for entry in test.entries
                          if entry.kind in DRIVE_KINDS), None)
        for entry in test.entries:
            if entry.kind == 'restart' and 'cause' not in entry.fields:
                holds.append(Hold('restart-cause', number, entry,
                                  f'the restart at #{entry.id} has no cause '
                                  'written'))
            elif entry.kind == 'level_row':
                holds.extend(self._level_holds(test, entry))
            elif entry.kind == 'notch':
                planned, why = self.notch_planned(entry)
                if not planned and self._deviated(entry) is None:
                    holds.append(Hold('unplanned-notch', number, entry,
                                      f'the notch at #{entry.id} is not '
                                      f'planned: {why} A deviation naming it '
                                      'answers this'))
            elif entry.kind in ('header_change', 'setting_change'):
                what = self._plan_change(test, entry, first_run)
                if what and self._deviated(entry) is None:
                    holds.append(Hold('plan-change', number, entry,
                                      f'{what} at #{entry.id} is a change to '
                                      'the test; a deviation naming it '
                                      'answers this'))
        return holds

    def _plan_change(self, test: Test, entry: Entry,
                     first_run: Entry | None) -> str:
        if entry.kind == 'setting_change':
            governs = entry.fields.get('governs')
            setting = entry.fields.get('setting', '')
            if governs in _PLAN_SETTINGS or (governs is None and any(
                    word in setting.lower() for word in _PLAN_WORDS)):
                return f'the change of {setting}'
            return ''
        name = entry.fields.get('field')
        # The header typed in before the test runs is the plan written
        # down; a change once it is running is a change on the floor.
        if name not in PLAN_CHANGE_FIELDS or first_run is None \
                or entry.id < first_run.id:
            return ''
        before = self._header(test.entries, test.opened, before=entry.id)
        if before.get(name) == entry.fields.get('value'):
            return ''                        # a confirmation, not a change
        return f'the change of the header\'s {name}'

    def _level_holds(self, test: Test, entry: Entry) -> list[Hold]:
        holds = []
        fields = entry.fields
        answers = [fields.get('answer'),
                   *fields.get('channel_answers', {}).values()]
        if 'not explained' in answers:
            axis = self._axis_of(test, entry)
            later = [found for found in test.entries
                     if found.kind == 'level_row' and found.id > entry.id
                     and self._axis_of(test, found) == axis
                     and 'answer' in found.fields
                     and 'not explained' not in (
                         found.fields['answer'],
                         *found.fields.get('channel_answers', {}).values())]
            if not later:
                channels = [name for name, answer in
                            fields.get('channel_answers', {}).items()
                            if answer == 'not explained']
                which = (f' (channel {", ".join(channels)})'
                         if channels else '')
                holds.append(Hold('not-explained', test.number, entry,
                                  f'the answer at #{entry.id} is "not '
                                  f'explained"{which}; the test holds at '
                                  'that level until a later answer'))
        if fields.get('go') is True and 'agreed_by' not in fields:
            points = self.header_at(entry).get('hold_points_db') or []
            level = fields.get('level_db')
            if any(_same_level(level, point) for point in points):
                holds.append(Hold('hold-point', test.number, entry,
                                  f'the go at #{entry.id} is at the hold '
                                  f'point {level:+g} dB with no customer '
                                  'agreement logged (agreed_by)'))
        return holds

    # ---- owed lists -----------------------------------------------------------

    def owed(self, test: Any = ALL) -> list[OwedList]:
        """Every list with a line still owed, for one test or the whole
        log: the setup before the first low level, each axis change, the
        release checklist once the test is closed to runs, each abort's
        first-minute fields, each shot's four checks, and the days with
        no clock check."""
        lists: list[OwedList] = []
        for each in self._selected_tests(test):
            lists.extend(self._checklists(each))
            lists.extend(self.abort_fields(each.number))
            lists.extend(self.shot_checks(each.number))
        lists.append(self.clock_days(test))
        return [found for found in lists if not found.complete]

    def _first_drive(self, test: Test, axis: str | None) -> Entry | None:
        return next((entry for entry in test.entries
                     if entry.kind in DRIVE_KINDS
                     and self._axis_of(test, entry) == axis), None)

    def _checklist(self, name: str, title: str, test: Test, axis: str | None,
                   lines: Sequence[tuple[str, str]], kind: str,
                   axes: Sequence[str | None] | None,
                   due: Entry | None) -> OwedList:
        done: dict[str, Entry] = {}
        for entry in test.entries:
            if entry.kind != kind or (axes is not None
                                      and entry.axis not in axes):
                continue
            key = _line_of(lines, entry.fields.get('step'))
            if key is not None:
                done.setdefault(key, entry)
        rows = []
        for key, label in lines:
            entry = done.get(key)
            late = (entry is not None and due is not None
                    and entry.event_time > due.event_time)
            rows.append(OwedLine(key, label, entry, late))
        return OwedList(name, title, test.number, axis, tuple(rows), due)

    def checklist(self, test: int, name: str,
                  axis: str | None = None) -> OwedList:
        """One checklist of test `test`, every line with who did it and
        when or still owed: 'before low level' (the test's first axis),
        'axis change' (to `axis`, an axis after the first) or 'release'.
        A line is done by a setup step (a post-test step for the release
        checklist) naming it by its key or its words; an axis change's
        steps are on the new axis, and its first line may name the axis
        it closes."""
        found = self.test(test)
        if name == 'before low level':
            first = found.axes[0] if found.axes else None
            return self._checklist(
                name, 'Before the first low level', found, first,
                BEFORE_LOW_LEVEL, 'setup_step', None,
                self._first_drive(found, first))
        if name == 'axis change':
            if axis not in found.axes[1:]:
                raise RunLogError(f'test {test} has no axis change to '
                                  f'{axis!r}; its axes are: '
                                  + (', '.join(found.axes) or 'none'))
            index = found.axes.index(axis)
            rows = self._checklist(name, f'The change to axis {axis}', found,
                                   axis, AXIS_CHANGE, 'setup_step', [axis],
                                   self._first_drive(found, axis))
            if rows.lines[0].owed:
                closing = self._checklist(
                    name, '', found, axis, AXIS_CHANGE[:1], 'setup_step',
                    [found.axes[index - 1]], rows.due).lines[0]
                rows = OwedList(rows.name, rows.title, rows.test, rows.axis,
                                (closing, *rows.lines[1:]), rows.due)
            return rows
        if name == 'release':
            return self._checklist(name, 'The release checklist', found, None,
                                   RELEASE, 'post_test_step', None, None)
        raise RunLogError(f'{name!r} is not a checklist; the checklists are: '
                          'before low level, axis change, release')

    def _checklists(self, test: Test) -> list[OwedList]:
        lists = [self.checklist(test.number, 'before low level')]
        lists += [self.checklist(test.number, 'axis change', axis)
                  for axis in test.axes[1:]]
        if test.state != 'open':
            lists.append(self.checklist(test.number, 'release'))
        return lists

    def abort_fields(self, test: Any = ALL) -> list[OwedList]:
        """Each abort's first-minute fields, done by the abort or by the
        correction that added the field: the level, the channel, the
        line or band, the panel fault, the counter, who called it, what
        was heard, the records read and the over-test answer. A partial
        abort is accepted; what it lacks is owed here."""
        lists = []
        optional = KINDS['abort'].optional
        for each in self._selected_tests(test):
            for abort in each.entries:
                if abort.kind != 'abort':
                    continue
                supplied = {name: abort for name in abort.fields}
                for correction in self.log.corrections_of(abort.id):
                    for name, value in correction.fields.items():
                        if name in optional:
                            if value is None:
                                supplied.pop(name, None)
                            else:
                                supplied[name] = correction
                lines = tuple(
                    OwedLine(name, name.replace('_', ' '),
                             supplied.get(name) if name in abort.fields
                             else None)
                    for name in optional)
                lists.append(OwedList('abort', f'The first minute of abort '
                                      f'#{abort.id}', each.number,
                                      self._axis_of(each, abort), lines))
        return lists

    def shot_checks(self, test: Any = ALL) -> list[OwedList]:
        """Each shot's four checks, a line done when the shot (as
        corrected) carries every field of the check."""
        lists = []
        for each in self._selected_tests(test):
            for shot in each.entries:
                if shot.kind != 'shot':
                    continue
                lines = tuple(
                    OwedLine(key, label,
                             shot if all(name in shot.fields
                                         for name in names) else None)
                    for key, names, label in SHOT_CHECKS)
                lists.append(OwedList('shot', f'The checks after shot '
                                      f'#{shot.id}', each.number,
                                      self._axis_of(each, shot), lines))
        return lists

    def clock_days(self, test: Any = ALL) -> OwedList:
        """A line for every day the selected entries fall on (by each
        entry's own date), done by that day's first clock check, which
        may be anywhere in the log."""
        checks: dict[date, Entry] = {}
        for entry in self.entries:
            if entry.kind == 'clock_check':
                checks.setdefault(entry.event_time.date(), entry)
        days: list[date] = []
        for entry in self._selected_entries(test):
            day = entry.event_time.date()
            if day not in days:
                days.append(day)
        lines = tuple(OwedLine(day.isoformat(), f'A clock check on {day}',
                               checks.get(day)) for day in sorted(days))
        return OwedList('clock', 'A clock check each day',
                        test if isinstance(test, int) else None, None, lines)

    # ---- views ----------------------------------------------------------------

    def interruptions(self, test: Any = ALL) -> list[Interruption]:
        """The aborts, trips and scheduled stops, each with the cause, the
        path and the restart that answer it (by `refers_to`, or the next
        written in the same test)."""
        found = []
        for each in self._selected_tests(test):
            stops = [entry for entry in each.entries
                     if entry.kind == 'abort' or (
                         entry.kind == 'run_stop'
                         and entry.fields.get('scheduled'))]
            causes = self._link(each, stops, 'cause_found')
            paths = self._link(each, stops, 'path_decided')
            restarts = self._link(each, stops, 'restart')
            for stop in stops:
                kind = ('scheduled' if stop.kind == 'run_stop' else
                        'trip' if stop.fields['stop'] in ('amplifier trip',
                                                          'power loss')
                        else 'abort')
                found.append(Interruption(stop, kind, causes.get(stop.id),
                                          paths.get(stop.id),
                                          restarts.get(stop.id)))
        return found

    def credited(self, test: Any = ALL) -> list[Credit]:
        """The time credited on each axis of one test or every test, from
        the counters, against the header's duration and tolerance.

        A run's time counts once it reaches full level (an entry giving
        0 dB). A segment runs from its start (a run start's counter, 0
        when not given, or a restart's counter) to the counter at its
        stop or abort. A restart that resumes continues the count; one
        that starts the test again discards what was credited before it
        on the axis. A segment whose counter is missing or unreadable
        leaves the total a lower bound, and the verdict not judged.
        """
        found = []
        for each in self._selected_tests(test):
            axes: list[str | None] = []
            for entry in each.entries:
                if entry.kind in RUN_KINDS:
                    axis = self._axis_of(each, entry)
                    if axis not in axes:
                        axes.append(axis)
            found.extend(self._credit(each, axis) for axis in axes)
        return found

    def _counter(self, entry: Entry, name: str, notes: list[str]
                 ) -> float | None:
        try:
            return parse_counter(entry.fields[name])
        except RunLogError as error:
            notes.append(f'#{entry.id}: {error}')
            return None

    def _credit(self, test: Test, axis: str | None) -> Credit:
        ordered = sorted((entry for entry in test.entries
                          if entry.kind in ('run_start', 'run_stop', 'abort',
                                            'restart')
                          and self._axis_of(test, entry) == axis),
                         key=lambda entry: (entry.event_time, entry.id))
        # A run stop and an abort written together for one stop are one
        # stop: the counter is taken from whichever of them gives it.
        events: list[Entry] = []
        for entry in ordered:
            last = events[-1] if events else None
            if entry.kind in ('run_stop', 'abort') and last is not None \
                    and last.kind in ('run_stop', 'abort') \
                    and last.run == entry.run:
                if 'counter' not in last.fields and 'counter' in entry.fields:
                    events[-1] = entry
                continue
            events.append(entry)
        credited = 0.0
        notes: list[str] = []
        complete = True
        start: float | None = None
        full = False
        running: Entry | None = None
        for entry in events:
            run = self._run_of(test, entry)
            reached = run.reached_full if run is not None else False
            if entry.kind == 'run_start':
                if running is not None:
                    notes.append(f'run {running.run} has no stop logged before '
                                 f'#{entry.id}; its time is not credited')
                    complete = False
                start = (self._counter(entry, 'counter', notes)
                         if 'counter' in entry.fields else 0.0)
                complete = complete and start is not None
                full, running = reached, entry
            elif entry.kind == 'restart':
                resume = entry.fields.get('resume')
                running = entry
                if resume is None:
                    notes.append(f'the restart at #{entry.id} does not say '
                                 'whether it resumes or starts again')
                    complete, start = False, None
                    continue
                if resume is False:
                    if credited:
                        notes.append(f'the restart at #{entry.id} starts the '
                                     f'test again; the {clock_text(credited)} '
                                     'credited before it is not counted')
                    credited = 0.0
                    start = (self._counter(entry, 'counter_at_restart', notes)
                             if 'counter_at_restart' in entry.fields else 0.0)
                    full = reached
                elif 'counter_at_restart' in entry.fields \
                        or 'counter_at_stop' in entry.fields:
                    start = self._counter(
                        entry, 'counter_at_restart'
                        if 'counter_at_restart' in entry.fields
                        else 'counter_at_stop', notes)
                    full = full or reached
                else:
                    notes.append(f'the restart at #{entry.id} gives no '
                                 'counter, so the time after it is not '
                                 'credited')
                    start = None
                complete = complete and start is not None
            else:
                if running is None:
                    notes.append(f'#{entry.id} stops a run with no start '
                                 'logged; its time is not credited')
                    complete = False
                    continue
                if start is None:
                    running = None             # already noted where it began
                    continue
                if 'counter' not in entry.fields:
                    notes.append(f'the {entry.label.lower()} at #{entry.id} '
                                 'gives no counter, so the time before it is '
                                 'not credited')
                    complete, running = False, None
                    continue
                end = self._counter(entry, 'counter', notes)
                if end is None:
                    complete = False
                elif end < start:
                    notes.append(f'the counter at #{entry.id} '
                                 f'({clock_text(end)}) is below the counter '
                                 f'at the segment\'s start '
                                 f'({clock_text(start)})')
                    complete = False
                elif full or reached:
                    credited += end - start
                else:
                    notes.append(f'run {entry.run} did not reach full level; '
                                 'its time is not credited')
                running = None
        if running is not None:
            notes.append(f'run {running.run} has no stop logged'
                         + (' yet' if test.state == 'open' else '')
                         + '; its time so far is not credited')
            complete = False
        header = test.header
        duration = header.get('duration_s')
        minus = header.get('duration_minus_pct')
        plus = header.get('duration_plus_pct')
        lower = duration * (1 - minus / 100) if (
            duration is not None and minus is not None) else None
        upper = duration * (1 + plus / 100) if (
            duration is not None and plus is not None) else None
        text = clock_text(credited)
        if duration is None:
            verdict, message = 'not judged', (
                f'{text} credited; not judged: the header gives no duration')
        elif not complete:
            verdict, message = 'not judged', (
                f'at least {text} credited; not judged: '
                + '; '.join(notes))
        elif lower is None or upper is None:
            verdict, message = 'not judged', (
                f'{text} credited of {clock_text(duration)}; not judged: the '
                'header gives no duration tolerance (minus and plus)')
        elif credited < lower:
            verdict, message = 'short', (
                f'{text} credited is short of {clock_text(lower)}, the least '
                f'the tolerance allows on {clock_text(duration)}')
        elif credited > upper:
            verdict, message = 'over', (
                f'{text} credited is over {clock_text(upper)}, the most the '
                f'tolerance allows on {clock_text(duration)}')
        else:
            verdict, message = 'within', (
                f'{text} credited, within {clock_text(lower)} to '
                f'{clock_text(upper)}')
        return Credit(test.number, axis, credited, complete, tuple(notes),
                      duration, lower, upper, verdict, message)

    def pulses(self, test: int) -> PulseCount:
        """The pulses test `test` has given (each shot one, or its
        `pulses`), by axis and in all, against the header's budget. Every
        shot entry counts, a missed one too: a missed pulse spends the
        budget."""
        found = self.test(test)
        by_axis: dict[str | None, int] = {}
        for shot in found.entries:
            if shot.kind == 'shot':
                axis = self._axis_of(found, shot)
                by_axis[axis] = by_axis.get(axis, 0) + shot.fields.get(
                    'pulses', 1)
        return judge_pulses(test, by_axis, found.header.get('pulse_budget'))

    def findings(self, test: Any = ALL) -> list[Finding]:
        """The deviations and anomalies, by number, each with the
        dispositions written against it anywhere in the log."""
        dispositions: dict[str, list[Entry]] = {}
        for entry in self.entries:
            if entry.kind == 'anomaly_dispositioned':
                dispositions.setdefault(entry.fields['number'], []).append(
                    entry)
        found = []
        for entry in self._selected_entries(test):
            if entry.kind in ('deviation', 'anomaly_opened') \
                    and 'number' in entry.fields:
                number = entry.fields['number']
                kind = 'deviation' if entry.kind == 'deviation' else 'anomaly'
                found.append(Finding(number, kind, entry, entry.test,
                                     tuple(dispositions.get(number, ())
                                           if kind == 'anomaly' else ())))
        return found

    def open_anomalies(self, test: Any = ALL) -> list[Finding]:
        """The anomalies with no disposition, with their owners."""
        return [found for found in self.findings(test) if found.open]

    def files(self, test: Any = ALL) -> list[FileRow]:
        """Every file the selected entries name, in the order written."""
        return [FileRow(entry, ref) for entry in self._selected_entries(test)
                for ref in entry.files]

    def late(self, test: Any = ALL) -> list[Entry]:
        """The selected entries written down late (`Entry.is_late`)."""
        return [entry for entry in self._selected_entries(test)
                if entry.is_late()]

    def chronology(self, test: Any = ALL) -> list[ChronologyRow]:
        """The selected entries in the order of their events (by event
        time, then by id), each with its time on the recorders' clock."""
        rows = sorted(self._selected_entries(test),
                      key=lambda entry: (entry.event_time, entry.id))
        return [ChronologyRow(entry, self.log.recorder_time(entry),
                              entry.is_late()) for entry in rows]

    # ---- readings -------------------------------------------------------------

    def reading(self, entry: Entry | int) -> LevelReading:
        """Level row `entry` read against its test's header as it stood
        when the row was written: projected to full level (`project`),
        its volts per amp against the previous level of the run, and its
        readings against the last run of the same setup (the same level
        on the same axis, in an earlier run of the test or of a test of
        the same kind and configuration). What cannot be read reads not
        judged, with why."""
        entry_id = entry.id if isinstance(entry, Entry) else entry
        if entry_id not in self._by_id:
            raise RunLogError(f'the log has no entry #{entry_id}')
        row = self._by_id[entry_id]
        if row.kind != 'level_row':
            raise RunLogError(f'#{row.id} is a {row.label} entry, not a level '
                              'row')
        fields = row.fields
        header = self.header_at(row)
        projection = None
        if 'level_db' not in fields:
            message = 'not judged: the row gives no level'
        elif row.test is None:
            message = 'not judged: the row belongs to no test, so no header'
        else:
            try:
                projection = project(
                    fields['level_db'], voltage=fields.get('voltage'),
                    current=fields.get('current'),
                    drive_fraction=fields.get('drive_fraction'),
                    sigma=header.get('sigma'),
                    rated_current_peak=header.get('rated_current_peak'),
                    rated_current_rms=header.get('rated_current_rms'),
                    rail_voltage=header.get('rail_voltage'),
                    drive_window=header.get('drive_window'))
                message = str(projection)
            except RunLogError as error:
                message = f'not judged: {error}'
        return LevelReading(row, projection, message, self._previous(row),
                            self._last_run(row))

    def _previous(self, row: Entry) -> Comparison | None:
        if row.test is None or row.run is None:
            return None
        before = [entry for entry in self.test(row.test).entries
                  if entry.kind == 'level_row' and entry.run == row.run
                  and entry.id < row.id and _va(entry.fields) is not None]
        if not before:
            return None
        then = before[-1]
        ratio = _ratio(_va(row.fields), _va(then.fields))
        flagged = (('volts per amp',) if ratio is not None
                   and abs(ratio - 1) > VA_CHANGE else ())
        return Comparison(then, 'the previous level', {'volts per amp': ratio},
                          flagged)

    def _last_run(self, row: Entry) -> Comparison | None:
        if row.test is None or 'level_db' not in row.fields:
            return None
        test = self.test(row.test)
        axis = self._axis_of(test, row)
        configuration = test.header.get('configuration')
        candidates = []
        for entry in self.entries:
            if entry.kind != 'level_row' or entry.id >= row.id \
                    or entry.test is None or not _same_level(
                        entry.fields.get('level_db'), row.fields['level_db']):
                continue
            other = self.test(entry.test)
            if self._axis_of(other, entry) != axis:
                continue
            if entry.test == row.test:
                if entry.run is None or entry.run == row.run:
                    continue
            elif not (configuration and other.kind == test.kind
                      and other.header.get('configuration') == configuration):
                continue
            candidates.append(entry)
        if not candidates:
            return None
        then = candidates[-1]
        ratios = {name: _ratio(row.fields.get(name), then.fields.get(name))
                  for name in ('current', 'voltage')}
        flagged = tuple(name for name, ratio in ratios.items()
                        if ratio is not None and abs(ratio - 1) > VA_CHANGE)
        return Comparison(then, f'the last run of the same setup (test '
                          f'{then.test}, run {then.run})', ratios, flagged)


# ---- writing through the model -------------------------------------------------


def admit(log: RunLog | RunLogModel, kind: str, test: int | None) -> None:
    """Refuse an entry of `kind` for `test` that the test's state does not
    take, with `RunLogError` naming the test and its state: a run entry
    with no test, any entry for a test never opened, a run entry for a
    test closed to runs, and a run, finding or header change for a
    released test. A test is opened with `open_test` or `roll`, and
    closed with `close_test`."""
    model = log if isinstance(log, RunLogModel) else RunLogModel(log)
    if kind in ('test_opened', 'test_closed'):
        raise RunLogError('a test is opened with open_test or roll, and '
                          'closed with close_test or roll')
    if test is None:
        if kind in RUN_KINDS:
            raise RunLogError(f'a {KINDS[kind].label} entry belongs to a run '
                              'of a test; give its test')
        return
    found = model._tests.get(test)
    if found is None or found.opened is None:
        raise RunLogError(f'test {test} has not been opened; open it first')
    if found.state == 'released' and kind in RUN_KINDS | RELEASED_REFUSES:
        raise RunLogError(f'test {test} is released; it takes no new runs, '
                          'findings or header changes')
    if found.state == 'closed to runs' and kind in RUN_KINDS:
        raise RunLogError(f'test {test} is closed to runs; it still takes '
                          'findings, post-test steps and dispositions')


def append(log: RunLog, kind: str, *, author: str,
           test: int | None = None, **kwargs: Any) -> Entry:
    """`RunLog.append`, refused first when the test's state does not take
    the entry (`admit`)."""
    admit(log, kind, test)
    return log.append(kind, author=author, test=test, **kwargs)


def _test_kind(test_kind: Any) -> str:
    if test_kind not in TEST_KINDS:
        raise RunLogError(f'{test_kind!r} is not a kind of test; the kinds '
                          'are: ' + ', '.join(TEST_KINDS))
    return test_kind


def open_test(log: RunLog, *, author: str, test_kind: str,
              header: Mapping[str, Any] | None = None, text: str = '',
              event_time: datetime | None = None,
              files: Iterable[Any] = ()) -> Entry:
    """Open the log's next test, when no test is open to runs.

    Parameters
    ----------
    log : RunLog
        The log, open for writing.
    author : str
        The author's typed initials.
    test_kind : str
        One of `TEST_KINDS`.
    header : mapping, optional
        The header's fields (`run_log.HEADER_FIELDS`).
    text : str, optional
        Free text.
    event_time : datetime, optional
        When it was opened, with its zone; now, if not given.
    files : iterable, optional
        Files the header attaches (the plan, the procedure, the
        profile), each a path or a `FileRef`.

    Returns
    -------
    Entry
        The test opened entry.

    Raises
    ------
    RunLogError
        When a test is open (roll to the next instead), the kind is not
        one of `TEST_KINDS`, or a header field is refused.
    """
    model = RunLogModel(log)
    if model.current is not None:
        raise RunLogError(f'test {model.current.number} is open; roll to the '
                          'next test, which closes it to runs')
    fields = {**(header or {}), 'test_kind': _test_kind(test_kind)}
    if 'carried' in fields:
        raise RunLogError('a test opened by hand carries nothing; roll to '
                          'carry a header')
    return log.append('test_opened', author=author, test=model.next_test(),
                      fields=fields, text=text, event_time=event_time,
                      files=files)


def roll(log: RunLog, *, author: str, test_kind: str | None = None,
         header: Mapping[str, Any] | None = None, reason: str = '',
         text: str = '', event_time: datetime | None = None,
         files: Iterable[Any] = ()) -> tuple[Entry, Entry]:
    """Close the current test to runs and open the next, its header
    carried from the current one.

    Every field of the current header not given in `header` (nor the
    kind, when `test_kind` is given) is carried, and marked carried
    until it is confirmed (`confirm`) or changed (`change_header`). The
    current test still takes findings, post-test steps and dispositions
    until its release entry; its release checklist is owed.

    Parameters
    ----------
    log : RunLog
        The log, open for writing.
    author : str
        The author's typed initials.
    test_kind : str, optional
        The next test's kind; carried when not given.
    header : mapping, optional
        The next test's header fields that differ, or are confirmed.
    reason : str, optional
        Why the current test closes; 'rolled to test N' when not given.
    text : str, optional
        Free text on the next test's opening.
    event_time : datetime, optional
        When, with its zone; now, if not given.
    files : iterable, optional
        Files the next test's header attaches.

    Returns
    -------
    (Entry, Entry)
        The current test's closing and the next test's opening.

    Raises
    ------
    RunLogError
        When no test is open, the kind is not one of `TEST_KINDS`, or a
        header field is refused; nothing is written then.
    """
    model = RunLogModel(log)
    current = model.current
    if current is None:
        raise RunLogError('no test is open to roll from; open one with '
                          'open_test')
    given = dict(header or {})
    if 'carried' in given:
        raise RunLogError('what is carried is worked out by the roll, not '
                          'given')
    if test_kind is not None:
        given['test_kind'] = _test_kind(test_kind)
    fields = {**current.header.values, **given}
    carried = [name for name in current.header.values if name not in given]
    if carried:
        fields['carried'] = carried
    fields = check_fields('test_opened', fields)      # refused before writing
    if fields.get('test_kind') not in TEST_KINDS:
        fields['test_kind'] = _test_kind(fields.get('test_kind'))
    number = model.next_test()
    closing = log.append('test_closed', author=author, test=current.number,
                         fields={'reason': reason or f'rolled to test '
                                                     f'{number}'},
                         event_time=event_time)
    opening = log.append('test_opened', author=author, test=number,
                         fields=fields, text=text,
                         event_time=closing.event_time, files=files)
    return closing, opening


def close_test(log: RunLog, test: int, *, author: str, reason: str = '',
               event_time: datetime | None = None) -> Entry:
    """Close test `test` to runs without opening another. It still takes
    findings, post-test steps and dispositions until its release."""
    found = RunLogModel(log).test(test)
    if found.state != 'open':
        raise RunLogError(f'test {test} is already {found.state}')
    return log.append('test_closed', author=author, test=test,
                      fields={'reason': reason}, event_time=event_time)


def change_header(log: RunLog, test: int, field: str, value: Any, *,
                  author: str, text: str = '',
                  event_time: datetime | None = None) -> Entry:
    """Change one field of test `test`'s header, the value it had
    recorded beside the new one. A carried field changed is confirmed.
    Once the test is running, a change of a limit, tolerance, abort band,
    planned notch or hold point raises a hold until a deviation names
    it."""
    model = RunLogModel(log)
    admit(model, 'header_change', test)
    was = model.test(test).header.get(field)
    fields = {'field': field, 'value': value}
    if was is not None:
        fields['was'] = was
    return log.append('header_change', author=author, test=test,
                      fields=fields, text=text, event_time=event_time)


def confirm(log: RunLog, test: int, names: Iterable[str] | None = None, *,
            author: str, event_time: datetime | None = None) -> list[Entry]:
    """Confirm carried fields of test `test` as they stand (every carried
    field when `names` is None): a header change of each to its own
    value, which is a confirmation, not a change. Refused for a field
    that is not carried."""
    model = RunLogModel(log)
    found = model.test(test)
    names = list(found.header.carried if names is None else names)
    for name in names:
        if not found.header.is_carried(name):
            raise RunLogError(f'{name!r} is not carried in test {test}; the '
                              'carried fields are: '
                              + (', '.join(found.header.carried) or 'none'))
    admit(model, 'header_change', test)
    return [log.append('header_change', author=author, test=test,
                       fields={'field': name,
                               'value': found.header.get(name),
                               'was': found.header.get(name)},
                       text='confirmed as carried', event_time=event_time)
            for name in names]


def release(log: RunLog, test: int, *, author: str, text: str = '',
            event_time: datetime | None = None) -> Entry:
    """Write test `test`'s release, the release checklist's 'release'
    line: the article and facility released and the setup free to be
    broken. The names typed are recorded; the log tracks no signatures.
    Refused while the test is open to runs."""
    found = RunLogModel(log).test(test)
    if found.state == 'open':
        raise RunLogError(f'test {test} is open to runs; close it (or roll) '
                          'before its release')
    if found.released is not None:
        raise RunLogError(f'test {test} is already released, at '
                          f'#{found.released.id}')
    return log.append('post_test_step', author=author, test=test,
                      fields={'step': 'release'}, text=text,
                      event_time=event_time)
