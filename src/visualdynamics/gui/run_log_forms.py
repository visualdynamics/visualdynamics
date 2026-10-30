"""The Run Log's forms: the test header in full, the readings each kind
of test takes, the findings, and the predictions read beside what is
observed.

Every form here is an `EntryForm` (`gui.run_log_dock`): its event time
taken when it opens, Enter writing it, Esc putting it away, and nothing
written when a box cannot be read. What a form computes from what is
typed is shown under its boxes as it is typed (`EntryForm.live`): a
sweep's current against the continuous rating, a shot's pulses against
the budget, a notch's place in the plan and the notch predicted there.
Each of those is read through the function the package keeps for it
(`run_log_model.project`, `judge_pulses`, `notch_in_plan`, and the
predictions' own readings), so a form says what the log will say once
the entry is written.

**The header** comes in three forms: the test opened, the roll to the
next test (every field offered as it stands and carried unless
changed), and the current test's header, where each field changed is a
header change and the fields carried from the last test are marked
until they are confirmed. Its fields are grouped as `HEADER_GROUPS`
lists them, and every field of `run_log.HEADER_FIELDS` is in one group.

**The predictions** are the ones the window drew for the project open
in it: the notch the limit channels would cut (`core.limit_prediction`)
and the drives the specification asks for (`core.drive_prediction`).
The window hands each one to the dock when it is drawn
(`RunLogPanel.offer_prediction`), and the dock shows it beside what is
read: a notch entry beside the predicted notch at its band, and the
level row's drive beside the predicted drive. Nothing is predicted here.
With none drawn, which is always so for a log kept with no project, the
place it would be shown says so.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..core.run_log import (
    DIRECTIONS,
    HEADER_FIELDS,
    SETTING_CLASSES,
    RunLogError,
    band_hz,
)
from ..core.run_log_model import (
    SHOT_CHECKS,
    TEST_KINDS,
    judge_pulses,
    notch_in_plan,
    project,
)
from .run_log_dock import EntryForm, Snapshot, notches_text

#: what a prediction field reads when the window has drawn none
NO_PREDICTION = 'no prediction in a project'

#: the header's fields in the groups the header form shows them in:
#: (group title, ((field, label, box kind), ...)). Every field of
#: `run_log.HEADER_FIELDS` is in exactly one group.
HEADER_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    ('Identity', (
        ('test_kind', 'Kind', 'choice'),
        ('title', 'Title', 'text'),
        ('test_class', 'Class', 'text'),
        ('purpose', 'Purpose', 'text'),
        ('article', 'Article', 'text'),
        ('serial', 'Serial', 'text'),
        ('configuration', 'Configuration', 'text'))),
    ('Governing documents', (
        ('document', 'Governing document', 'text'),
        ('document_revision', 'Its revision', 'text'),
        ('plan_revision', 'Plan revision', 'text'),
        ('procedure', 'Procedure', 'text'),
        ('procedure_revision', 'Procedure revision', 'text'),
        ('profile_source', 'Profile source', 'text'))),
    ('Step schedule and hold points', (
        ('steps_db', 'Step schedule (dB)', 'numbers'),
        ('hold_points_db', 'Hold points (dB)', 'numbers'))),
    ('Planned notches', (
        ('notches_planned', 'Planned notches (axis: band, depth)',
         'notches'),)),
    ('Limits and tolerances', (
        ('limits', 'Limits', 'text'),
        ('tolerance', 'Tolerance', 'text'),
        ('abort_bands', 'Abort bands', 'text'))),
    ('Duration', (
        ('duration_s', 'Duration (s)', 'number'),
        ('duration_minus_pct', 'Tolerance under (%)', 'number'),
        ('duration_plus_pct', 'Tolerance over (%)', 'number'))),
    ('Amplifier', (
        ('sigma', 'Sigma', 'number'),
        ('rated_current_peak', 'Peak current rating (A)', 'number'),
        ('rated_current_rms', 'Continuous current rating (A)', 'number'),
        ('rail_voltage', 'Rail (V)', 'number'),
        ('drive_window', 'Drive window at full level (low, high)',
         'numbers'))),
    ('Projected headroom', (
        ('projected_current', 'Projected peak current (A)', 'number'),
        ('projected_voltage', 'Projected peak voltage (V)', 'number'))),
    ('Shock', (
        ('pulse_budget', 'Pulse budget', 'whole'),)),
)

#: the readings forms by name, with the kind of entry each writes
READINGS = {
    'sweep': 'sweep', 'shot': 'shot', 'result': 'run_stop',
    'notch': 'notch', 'deviation': 'deviation', 'anomaly': 'anomaly_opened',
    'disposition': 'anomaly_dispositioned',
    'setting change': 'setting_change', 'opinion': 'opinion_given',
}

#: the readings forms that belong to one shape of run (`RUN_SHAPES`),
#: shown only for a test whose runs hold it
FOR_SHAPES = {
    'sweep': ('sweeps', 'levels and sweeps', 'any'),
    'shot': ('shots', 'any'),
    'result': ('result', 'any'),
}


@dataclass(frozen=True)
class Predictions:
    """The predictions the window drew for the project open in it: the
    notch prediction (`core.limit_prediction.LimitPrediction`) and the
    drive prediction (`core.drive_prediction.DrivePrediction`), each None
    until one is drawn, with the objects each was made from in words."""

    limit: Any = None
    limit_from: str = ''
    drive: Any = None
    drive_from: str = ''


# ---- the header ----------------------------------------------------------------


def header_form(name: str, taken: datetime, seen: Snapshot | None
                ) -> EntryForm:
    """The header form `name`: 'open test' (every field blank, the kind
    needed), 'roll' (every field offered as the current test has it,
    carried unless changed) or 'header' (the current test's header,
    each field changed written as a header change, the carried fields
    marked and confirmed with Confirm). Files named on the opening or
    the roll are attached by SHA-256."""
    header = dict(seen.header) if seen is not None and name != 'open test' \
        else {}
    carried = set(seen.carried) if seen is not None and name == 'header' \
        else set()
    fields: list[tuple[str, str, str, Any]] = []
    for group, rows in HEADER_GROUPS:
        kept = [row for row in rows
                if not (name == 'header' and row[0] == 'test_kind')]
        if not kept:
            continue
        fields.append(('', group, 'heading', None))
        for field, label, kind in kept:
            choices = None
            if field == 'test_kind':
                choices = ('', *TEST_KINDS) if name == 'roll' else TEST_KINDS
            if (name == 'roll' and field in header) or field in carried:
                label += ' (carried)'
            fields.append((field, label, kind, choices))
    if name == 'header':
        fields.append(('', 'Carried fields', 'heading', None))
        fields.append(('confirm', 'Confirm those left as they are', 'flag',
                       None))
    else:
        fields.append(('', 'Attached', 'heading', None))
        fields.append(('files', ('Files (the plan, the procedure, the '
                       'profile)'), 'files', None))
    if name == 'roll':
        fields.append(('reason', 'Why this test closes', 'text', None))
    if name == 'open test':
        title = 'Open the next test'
        note = ''
    elif name == 'roll':
        title = 'Roll to the next test'
        note = ('Every field left as it is offered is carried to the next '
                'test, and marked carried there until it is confirmed or '
                'changed. A box emptied is carried as it was.')
    else:
        test = seen.status if seen is not None else None
        title = (f'Header of test {test.test}: {test.title} ({test.kind})'
                 if test is not None and test.test is not None
                 else 'Header')
        note = ('Each field changed is written as a header change. Once the '
                'test is running, a change of a limit, tolerance, abort '
                'band, planned notch or hold point raises a hold until a '
                'deviation names it.')
        if carried:
            note += (' Carried from the last test and not yet confirmed: '
                     + ', '.join(field for field in HEADER_FIELDS
                                 if field in carried) + '.')
    return EntryForm(name, title, fields, taken,
                     required=('test_kind',) if name == 'open test' else (),
                     note=note, defaults=header,
                     text='Why' if name == 'header' else 'Note')


def header_changes(values: Mapping[str, Any], header: Mapping[str, Any]
                   ) -> dict[str, Any]:
    """The header fields among `values` (a header form's) that differ
    from `header`, the fields as they stand."""
    return {field: value for field, value in values.items()
            if field in HEADER_FIELDS and not same(field, value,
                                                   header.get(field))}


def same(field: str, value: Any, held: Any) -> bool:
    """Whether a header field typed as `value` is the `held` value: a
    planned notch list compared as it is written back, so a band stored
    as two numbers and one typed as text are the same band."""
    if field == 'notches_planned':
        return held is not None and notches_text(value) == notches_text(held)
    return value == held


# ---- the readings --------------------------------------------------------------


def _typed(form: EntryForm, *names: str) -> dict[str, Any]:
    found = {}
    for name in names:
        value = form.value(name)
        if value is not None:
            found[name] = value
    return found


def sweep_form(taken: datetime, seen: Snapshot | None) -> EntryForm:
    """A sine sweep: its direction, n of N, rate, span, level and the
    amplifier's rms current, judged live against the header's
    continuous rating, which is what a sine test budgets. n is offered
    as the next sweep of the run; N, the rate, the span and the level
    as the last sweep had them."""
    sweeps = seen.sweeps if seen is not None else ()
    last = dict(sweeps[-1]) if sweeps else {}
    header = dict(seen.header) if seen is not None else {}
    defaults = {'n': len(sweeps) + 1,
                **{name: last[name] for name in ('of', 'rate', 'span',
                                                 'level_db') if name in last}}
    fields = [('direction', 'Direction', 'choice', ('', *DIRECTIONS)),
              ('n', 'Sweep n', 'whole', None),
              ('of', 'of N', 'whole', None),
              ('rate', 'Rate (oct/min or Hz/s)', 'text', None),
              ('span', 'Span (Hz)', 'text', None),
              ('level_db', 'Level (dB)', 'number', None),
              ('current', 'Amplifier current, rms (A)', 'number', None)]
    return EntryForm('sweep', 'Sweep', fields, taken, defaults=defaults,
                     live=lambda form: sweep_text(header, form))


def sweep_text(header: Mapping[str, Any], form: EntryForm) -> str:
    """A sweep's rms current against the header's continuous rating, in
    words (`run_log_model.project`'s rms current check): not judged,
    with why, when the rating, the current or the level is missing."""
    typed = _typed(form, 'n', 'of', 'level_db', 'current')
    count = (f'Sweep {typed["n"]} of {typed["of"]}. '
             if 'n' in typed and 'of' in typed else '')
    rating = header.get('rated_current_rms')
    if rating is None:
        return (count + 'Continuous rating: not judged: the header gives no '
                'continuous current rating')
    if 'current' not in typed:
        return count + 'Continuous rating: not judged: no current was read'
    if 'level_db' not in typed:
        return count + 'Continuous rating: not judged: no level is given'
    projection = project(typed['level_db'], current=typed['current'],
                         rated_current_rms=rating)
    return count + 'Continuous rating: ' + \
        projection.check('rms current').message


def shot_form(taken: datetime, seen: Snapshot | None, axis: str | None
              ) -> EntryForm:
    """A shock shot: its polarity, n of N, level and pulses (one when
    blank), and the four checks made after every shot. Live: which
    checks are answered, and the pulses so far and with this shot
    against the header's budget (`run_log_model.judge_pulses`)."""
    fields = [('polarity', 'Polarity', 'choice', ('', 'positive',
                                                  'negative')),
              ('n', 'Shot n', 'whole', None),
              ('of', 'of N', 'whole', None),
              ('level_db', 'Level (dB)', 'number', None),
              ('pulses', 'Pulses (1 when blank)', 'whole', None),
              ('', 'The four checks', 'heading', None),
              ('in_tolerance', ('Pulse or spectrum in tolerance, both '
               'polarities'), 'flag', None),
              ('within_stroke', ('Displacement and velocity within the '
               'stroke'), 'flag', None),
              ('peak_current', 'Amplifier current peak (A)', 'number', None),
              ('peak_voltage', 'Amplifier voltage peak (V)', 'number', None),
              ('prediction', 'Controller\'s prediction for the next level',
               'text', None)]
    return EntryForm('shot', 'Shot', fields, taken,
                     live=lambda form: shot_text(seen, axis, form))


def shot_text(seen: Snapshot | None, axis: str | None,
              form: EntryForm) -> str:
    """The four checks answered and owed, and the pulse count so far and
    with the shot being typed, against the header's budget."""
    owed = [label for _key, names, label in SHOT_CHECKS
            if any(form.value(name) is None for name in names)]
    answered = len(SHOT_CHECKS) - len(owed)
    checks = f'Checks answered: {answered} of {len(SHOT_CHECKS)}'
    if owed:
        checks += '; owed: ' + '; '.join(label.lower() for label in owed)
    counted = seen.pulses if seen is not None else None
    if counted is None:
        return checks + '\nPulses: no test is open'
    pulses = form.value('pulses') or 1
    by_axis = dict(counted.by_axis)
    by_axis[axis] = by_axis.get(axis, 0) + pulses
    after = judge_pulses(counted.test, by_axis, counted.budget)
    return (f'{checks}\nPulses so far: {counted.message}\n'
            f'With this shot: {after.message}')


def result_form(taken: datetime) -> EntryForm:
    """A survey or functional run's result and who ran it, written as
    the run's stop."""
    return EntryForm('result', 'Survey or functional result',
                     [('result', 'Result', 'text', None),
                      ('by', 'Run by', 'text', None),
                      ('counter', 'Counter', 'text', None)], taken,
                     required=('result',))


def notch_form(taken: datetime, seen: Snapshot | None, axis: str | None,
               predictions: Callable[[], Predictions]) -> EntryForm:
    """A notch: its axis, channel, band and depth. Live: whether it is
    in the header's planned list (`run_log_model.notch_in_plan`, the
    rule a written notch is held by), and the notch predicted at its
    band beside it, from the predictions as `predictions()` gives them
    when the form is read."""
    header = dict(seen.header) if seen is not None else {}
    fields = [('axis', 'Axis', 'text', None),
              ('channel', 'Channel', 'text', None),
              ('band', 'Band (Hz)', 'text', None),
              ('depth_db', 'Depth (dB)', 'number', None)]
    return EntryForm('notch', 'Notch', fields, taken, required=('band',),
                     defaults={'axis': axis},
                     live=lambda form: notch_text(header, form,
                                                  predictions()))


def notch_text(header: Mapping[str, Any], form: EntryForm,
               predictions: Predictions) -> str:
    """A notch's place in the plan, then the prediction beside it."""
    typed = _typed(form, 'band', 'depth_db')
    planned, why = notch_in_plan(header.get('notches_planned'),
                                 form.value('axis'), typed)
    plan = (f'Planned: {why}' if planned else
            f'Not planned: {why} Written, it holds the test until a '
            'deviation names it.')
    return plan + '\n' + predicted_notch_text(predictions, typed.get('band'),
                                              typed.get('depth_db'))


def predicted_notch_text(predictions: Predictions, band: Any,
                         depth_db: float | None) -> str:
    """The notch the limit prediction cuts at `band` beside the notch
    observed there, each predicted notch in the prediction's own words
    (`Notch.describe`); with no band typed, the prediction's verdict.
    'no prediction in a project' when the window has drawn none."""
    prediction = predictions.limit
    if prediction is None:
        return f'Predicted: {NO_PREDICTION}'
    said = f'Predicted (from {predictions.limit_from})'
    if band is None:
        return f'{said}: {prediction.verdict()}'
    try:
        low, high = band_hz(band, 'the band')
    except RunLogError as error:
        return f'{said}: not compared: {error}'
    observed = (f'Observed: {low:g} to {high:g} Hz, '
                + (f'{abs(depth_db):g} dB deep' if depth_db is not None
                   else 'no depth given'))
    near = [notch for notch in prediction.notches
            if notch.low <= high and notch.high >= low]
    if not near:
        return (f'{said}: no notch from {low:g} to {high:g} Hz; '
                f'{prediction.verdict()}\n{observed}')
    return (f'{said}: '
            + '; '.join(notch.describe(prediction.dimension,
                                       prediction.tolerance)
                        for notch in near)
            + f'\n{observed}')


def drive_text(predictions: Predictions, header: Mapping[str, Any],
               fields: Mapping[str, Any]) -> str:
    """The drive predicted for the test's setup (each drive demand in
    the prediction's own words, `Demand.describe`) beside the drive the
    level row reads, its drive fraction projected to full level
    (`run_log_model.project`). 'no prediction in a project' when the
    window has drawn none."""
    prediction = predictions.drive
    if prediction is None:
        return f'Predicted drive: {NO_PREDICTION}'
    drives = [demand.describe() for demand in prediction.demands
              if demand.kind == 'drive']
    said = (f'Predicted drive (from {predictions.drive_from}): '
            + ('; '.join(drives) if drives
               else 'the prediction judges no drive'))
    level = fields.get('level_db')
    fraction = fields.get('drive_fraction')
    if fraction is None:
        return f'{said}\nRead drive: no drive fraction read'
    if level is None:
        return f'{said}\nRead drive: {fraction:g} of the range, at no level'
    try:
        check = project(level, drive_fraction=fraction,
                        drive_window=header.get('drive_window')
                        ).check('drive fraction')
    except RunLogError as error:
        return f'{said}\nRead drive: not projected: {error}'
    return (f'{said}\nRead drive: {fraction:g} of the range at '
            f'{level:+g} dB; {check.message}')


# ---- the findings and the rest -------------------------------------------------


def deviation_form(taken: datetime, seen: Snapshot | None) -> EntryForm:
    """A deviation: the fact at the console, and its cause, effect and
    approval when they are known (now, or later by correction); the
    names typed are recorded as agreed at the console. It answers the
    hold of the entry it names, offered as the latest an unplanned notch
    or a change to the test raised."""
    wanted = seen.deviation_wanted if seen is not None else ()
    holds = seen.holds if seen is not None else ()
    fields = [('fact', 'Fact at the console', 'text', None),
              ('cause', 'Cause (when known)', 'text', None),
              ('effect', 'Effect (when known)', 'text', None),
              ('approval', 'Approval (when known)', 'text', None),
              ('agreed_by', 'Agreed at the console by', 'text', None),
              ('refers_to', 'Names entry #', 'entry', None)]

    def live(form: EntryForm) -> str:
        named = form.value('refers_to')
        if named is None:
            return ('Names no entry: a deviation answers a hold only when it '
                    'names the entry that raised it')
        hold = next((found for found in holds if found.entry.id == named
                     and found.rule in ('unplanned-notch', 'plan-change')),
                    None)
        if hold is None:
            return f'#{named} raises no hold that a deviation answers'
        return f'Answers the hold at #{named}: {hold.message}'

    return EntryForm('deviation', 'Deviation', fields, taken,
                     defaults={'refers_to': wanted[-1] if wanted else None},
                     note='The names typed are recorded as agreed at the '
                          'console; the log does not track signatures or '
                          'approvals.', live=live)


def anomaly_form(taken: datetime) -> EntryForm:
    """An anomaly opened, numbered across the log, with its owner."""
    return EntryForm('anomaly', 'Anomaly opened',
                     [('fact', 'What was seen', 'text', None),
                      ('owner', 'Owner', 'text', None),
                      ('refers_to', 'Names entry #', 'entry', None)], taken)


def disposition_form(taken: datetime, seen: Snapshot | None) -> EntryForm:
    """An anomaly dispositioned, by its number: the latest open one is
    offered."""
    open_ones: Sequence[str] = seen.open_anomalies if seen is not None else ()
    return EntryForm('disposition', 'Anomaly dispositioned',
                     [('number', 'Anomaly number', 'text', None),
                      ('disposition', 'Disposition', 'text', None),
                      ('by', 'By', 'text', None)], taken,
                     required=('number',),
                     defaults={'number': open_ones[-1] if open_ones
                               else None},
                     note=('Open: ' + ', '.join(open_ones) if open_ones
                           else 'No anomaly is open.'))


def setting_form(taken: datetime) -> EntryForm:
    """A setting changed. A change to a limit, a tolerance or an abort
    band is a change to the test and holds it until a deviation names
    it."""
    return EntryForm('setting change', 'Setting changed',
                     [('setting', 'Setting', 'text', None),
                      ('value', 'Now', 'text', None),
                      ('was', 'Was', 'text', None),
                      ('governs', 'Governs', 'choice',
                       ('', *SETTING_CLASSES))], taken,
                     required=('setting',),
                     note='A limit, tolerance or abort band changed holds '
                          'the test until a deviation names the change.')


def opinion_form(taken: datetime) -> EntryForm:
    """An opinion given to the customer in conversation, and to whom."""
    return EntryForm('opinion', 'Opinion given',
                     [('to', 'Given to', 'text', None),
                      ('opinion', 'Opinion', 'text', None)], taken)
