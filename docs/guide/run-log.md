# The run log file

A run log is the record kept on the floor while a test runs: when each
level was reached, who called a stop and what was heard, when the
clocks were checked, who came and went, what was noted. It is kept by
a log keeper who is not at the controller, it rolls from one test to
the next on the same article, and it opens with or without a Visual
Dynamics project, so a lab that analyzes its data in other tools can
keep one too.

Each thing written down is one **entry**, and an entry is never
edited. A mistake is put right by a **correction**: a later entry that
names the one it corrects and carries the new values. Whoever reads
the log sees the corrected view, and the record keeps both.

## From a script

```python
from visualdynamics.core.run_log import RunLog

with RunLog.create('article-7.vdlog', title='Article 7',
                   mirror='//share/logs/article-7.vdlog') as log:
    log.append('clock_check', author='AB', fields={'offset_s': 1.25})
    log.append('test_opened', author='AB', test=1,
               fields={'title': 'Random, Z'})
    log.append('level_row', author='AB', test=1, axis='Z', run=1,
               fields={'level_db': -12.0, 'current': 63.0, 'go': True})
    stop = log.append('abort', author='AB', test=1, axis='Z', run=2,
                      fields={'stop': 'amplifier trip'})
    log.correct(stop, author='AB', fields={'channel': '4'},
                text='channel read off the panel')
```

`RunLog.open` opens an existing log to add to it, and `RunLog.read`
opens any log read-only. `log.view()` is the corrected view;
`log.entries` is every entry as stored, corrections included.

## The file

A `.vdlog` is JSON Lines: UTF-8 text, one JSON object per line, readable
in any text editor and by any language with a JSON reader. The first
line is the header, with the format's stamp, its version, the log's id
and when and where it was created:

    {"format": "visualdynamics-run-log", "version": 1, "log_id": "c450fa86…", "created": "2026-10-30T08:00:00.000-06:00", "title": "Article 7", "login": "tester", "computer": "bench"}

Each line after it is one entry:

    {"id": 5, "kind": "abort", "event_time": "2026-10-30T08:02:00.000-06:00", "entered_at": "2026-10-30T08:02:00.000-06:00", "author": "AB", "login": "tester", "computer": "bench", "test": 1, "axis": "Z", "run": 2, "fields": {"stop": "amplifier trip"}, "text": "", "files": [], "prev": "9b1f…", "hash": "e04c…"}

Every entry has its id (numbered from 1 across the whole log), its
kind, its event time (when the thing happened) and its entered-at time
(when it was written down), each with its date and its UTC offset, the
author's typed initials, the computer's login and machine name, the
test, axis and run it belongs to (any of them absent), the fields of
its kind, free text, the files it names by SHA-256, and its two links
in its test's chain.

Each entry is written, flushed and synced to the disk before the call
that wrote it returns, so a crash, a pulled plug or a killed program
loses at most the entry being written. A file cut off in the middle
of its last line is read up to that line, and the torn line is named
in `log.problems`; opened again for writing, the torn line is ended,
not removed, and the next entry starts on a line of its own. Any other
line that cannot be read is skipped and named the same way, and the
rest of the file is read.

## Versions

The header's version says which format the file is in. Each version
has its own reader, and a later Visual Dynamics keeps the reader of
every earlier version, so it reads every earlier file; a frozen file
of each version under `testdata/vdlog_corpus/` is opened by the tests
on every run. A file from a newer version is refused with a message to
update. Entries are only ever added to a file in the version this
build writes; an older file opens read-only.

The format may change while Visual Dynamics is an alpha. From the
first release that is not an alpha, an older file always opens.

## Entries

Each kind of entry asks for as little as it can, because the first
minute after a stop is the worst moment to be refused. An abort needs
only its time, the kind of stop and its author; the channel, the panel
fault, the counter, who called it and what was heard can all come
later, by correction. A field left blank is left out.

| Kind | Needs | May carry |
|---|---|---|
| `test_opened` | the test | the header's fields (below), carried |
| `test_closed` | the test | reason |
| `header_change` | field (a header field), value | was |
| `clock_check` | offset_s | recorder, method |
| `person_in`, `person_out` | person | role |
| `brief` | | attendees, topics |
| `checklist_reading` | item | reading, taken_by |
| `setup_step`, `post_test_step` | step | result |
| `run_start` | | level_db, counter |
| `level_row` | | level_db, voltage, current, drive_fraction, answer, channel_answers, go, agreed_by |
| `sweep` | | direction, n, of |
| `shot` | | polarity, n, of, level_db, pulses, in_tolerance, within_stroke, peak_current, peak_voltage, prediction |
| `run_stop` | | counter, reason, scheduled, result, by |
| `abort` | stop | channel, band, panel_fault, counter, called_by, heard, records_read, over_test, level_db |
| `cause_found` | cause | refers_to |
| `path_decided` | path | refers_to |
| `restart` | | resume, counter_at_stop, counter_at_restart, cause, decided_by, refers_to |
| `notch` | | channel, band, depth_db, planned |
| `deviation` | | number, fact, agreed_by, refers_to |
| `anomaly_opened` | | number, fact, owner, refers_to |
| `anomaly_dispositioned` | number | disposition, by |
| `adjustment` | | what |
| `photo` | | subject |
| `setting_change` | setting | value, was, governs |
| `opinion_given` | | to, opinion |
| `observation`, `note` | | |

The kind of stop is one of controller ramp, amplifier trip, power loss
and stop word: a hard shutdown may have over-tested the article, which
is why it is asked first. A level is a signed number of decibels
relative to full level (−12 for twelve below). `refers_to` names an
earlier entry by its id. A name typed into a deviation (`agreed_by`)
records who agreed at the console; the log does not track signatures or
approvals.

A few fields take one of a fixed set of words, and anything else is
refused: an answer (and each channel's answer in `channel_answers`) is
as predicted, explained or not explained; what a setting governs is
limit, tolerance, abort band or other; a sweep's direction is up or
down. A restart's `resume` is yes when the test continues with the time
already run counted, and no when it starts again.

Deviations and anomalies are numbered across the whole log, D-1, D-2,
A-1, in the same form as the report's findings register, so a number
names one finding however many tests the log spans.

## Tests, axes and runs

`visualdynamics.core.run_log_model` reads a log as tests, axes and
runs. `RunLogModel(log)` is built from the log's corrected view when it
is made; nothing it shows is stored in the file, so a hold, an owed
line or a credited duration is always what the entries say now. Make a
new one after the log changes.

```python
from visualdynamics.core import run_log_model as model

with RunLog.open('article-7.vdlog') as log:
    model.open_test(log, author='AB', test_kind='Random Vibration', header={
        'title': 'Random, Z', 'steps_db': [-12, -6, -3, 0],
        'hold_points_db': [-3], 'sigma': 3.0, 'rated_current_peak': 800.0,
        'rail_voltage': 250.0, 'duration_s': 600.0,
        'duration_minus_pct': 0.0, 'duration_plus_pct': 10.0,
        'notches_planned': [{'axis': 'Z', 'band': '180-220 Hz',
                             'depth_db': -6}]})
    model.append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
                 fields={'level_db': -12.0, 'voltage': 20.0, 'current': 63.0,
                         'drive_fraction': 0.12, 'answer': 'as predicted',
                         'go': True})
    state = model.RunLogModel(log)
    state.holds(), state.owed(), state.credited(1)
```

**A test** has a kind, one of the project types (Modal Test, Random
Vibration, Transient, Shock, Sine Sweep, Random and Sine, System ID)
or Functional or Other, so a report's template follows from one word.
Its **header** is what the test is (title, class, purpose), the article
and serial, its configuration, the governing document and its
revision, the plan and procedure revisions, the profile's source, and
the plan's numbers the log checks against:

| Field | Holds |
|---|---|
| `steps_db`, `hold_points_db` | the step schedule and the hold-point levels, in dB relative to full level |
| `notches_planned` | each planned notch: its band, and its axis and depth where the plan gives them |
| `limits`, `tolerance`, `abort_bands` | the limits, tolerances and abort bands, as text |
| `duration_s`, `duration_minus_pct`, `duration_plus_pct` | the duration per axis and its tolerance, minus and plus |
| `sigma` | the drive's peak-to-rms ratio |
| `rated_current_peak`, `rated_current_rms`, `rail_voltage` | the amplifier's peak and continuous current ratings and its rail |
| `drive_window` | the drive fraction's window at full level, low and high |
| `projected_current`, `projected_voltage` | the projected headroom from the plan's handover |
| `pulse_budget` | the pulses a shock test may give |

The files the header attaches (the plan, the procedure, the profile)
go on the test's opening entry by SHA-256. `model.change_header`
changes one field and records the value it had.

A test is **open** until it is closed (`model.close_test`) or the next
test opens. Then it is **closed to runs**, and still takes findings,
post-test steps and dispositions until its release (`model.release`,
the release checklist's release line), after which it is **released**
and takes no new runs, findings or header changes. An anomaly opened
before the release can still be dispositioned after it, since the
board that disposes of it meets later. `model.append` refuses what a
test's state does not take, with a message naming the test and its
state; an entry written past its state by other means is named in the
test's `problems`.

**The roll** (`model.roll`) closes the current test to runs and opens
the next with the header carried from the last. Every field not given
anew is carried, and the header marks it carried
(`test.header.carried`) until it is confirmed (`model.confirm`, which
records it unchanged) or changed:

```python
model.roll(log, author='AB', test_kind='Shock',
           header={'title': 'Release shock', 'pulse_budget': 12})
model.confirm(log, 2, ['article', 'configuration'], author='AB')
```

**Axes** are named by the entries of a test, in the order first named.
**Runs** are numbered within a test, each on one axis, and what a run
holds follows the test's kind: a random run's level rows, ending at full
level with the counter at its stop; a sine run's sweeps, each with its
direction and its number of how many; a shock run's shots, each with
its four checks (the pulse against its tolerance in both polarities,
the displacement and velocity against the stroke, the amplifier's
current and voltage peaks, the controller's prediction for the next
level); a survey or functional run's result and who ran it, on its
stop. Every shot counts against the header's pulse budget
(`state.pulses(test)`), a missed one too.

## Holds

A hold is something the log asks to be answered before the test goes
on. Holds are worked out from the entries, never stored, and shown
until answered; the log never stops the controller.

| Hold | Answered by |
|---|---|
| an abort with no cause | a cause found naming it, or the next cause found in the test |
| an abort whose over-test question is unanswered | the abort's `over_test`, written or added by correction |
| a restart with no cause | its `cause` |
| a "not explained" answer, for the run or a channel | a later level row on the same axis whose answers are all explained or as predicted |
| a notch not in the planned list | a deviation naming it (`refers_to`) |
| a limit, tolerance or abort-band change | a deviation naming it |
| a go at a hold-point level | the customer's agreement on the row (`agreed_by`) |

A notch is planned when it is on a planned notch's axis (a planned
notch with no axis is on every axis), its band is inside the planned
band, and it is no deeper than the planned depth where the plan gives
one. It is judged against the header as it stood when the notch was
written, so adding a notch to the plan afterwards does not plan it. A
notch typed as planned is still judged against the list.

A change to a limit, tolerance, abort band, planned notch, hold point
or duration tolerance in the header counts once the test is running
(the header typed in before the first run is the plan written down); a
header change to the value a field already has is a confirmation, not
a change. A setting change counts when it governs a limit, a tolerance
or an abort band, or, when it does not say, when its setting's name
says limit, tolerance or abort. A deviation answers such a hold only
when it names the entry: which change a deviation covers is not
guessed from where it was written.

## Owed lists

An owed list is a checklist, each line done by an entry, with who did
it and when, or still owed. `state.owed()` gives every list with a
line still owed, and `state.checklist(test, name)` one list whole:

- **Before the first low level** (`'before low level'`): the
  receiving inspection, the configuration checked on the article, the
  baseline operational checkout, the mounting record, the end-to-end
  check through its sixth step, the surveys, the photographs, and a
  second person's inspection of the setup.
- **The axis change** (`'axis change'`, to each axis after the first):
  the finished axis closed, the hardware from its bolt lists, the
  controller, the channel table, the end-to-end check, the surveys, the
  photographs, and the second person's look.
- **The release checklist** (`'release'`), owed once the test is
  closed to runs: the recorders stopped after the amplifier, the
  signature compared, the data check, the post-test functional, the
  inspection, the fixture, exciter and instrumentation inspected, every
  anomaly opened, the post-test review, the release, the article
  removed, the machine log and the flash report.
- **An abort's first minute** (`state.abort_fields()`): the channel,
  band, panel fault, counter, who called it, what was heard, the records
  read, the over-test answer and the level. A partial abort is
  accepted; what it lacks is owed here, done by a correction.
- **A shot's four checks** (`state.shot_checks()`).
- **A day with no clock check** (`state.clock_days()`).

A checklist line is done by a setup step (a post-test step for the
release checklist) whose `step` names the line by its key or its words,
ignoring case and hyphens: `'end-to-end check'`, `'End to end check'`.
An axis change's steps are written on the new axis; its first line may
name the axis it closes. A line done after the list was due (the first
drive on the axis) is marked so.

## Views

Each view is for one test (`test=1`), the entries between tests
(`test=None`), or the whole log (the default):

- `state.runs()`, with each run's outcome: completed, stopped (a
  scheduled stop), aborted, or running.
- `state.interruptions()`: the aborts, trips and scheduled stops, each
  with the cause, the path decided and the restart that answer it.
- `state.credited()`: the time credited on each axis from the counters,
  against the duration and its tolerance. A run's time counts once it
  reaches full level; a segment runs from its start (0 unless a counter
  is given) or a restart's counter to the counter at its stop. A resume
  continues the count, a restart that starts again discards what was
  credited before it, and a run stop and an abort written together for
  one stop are one stop. A counter is seconds, minutes:seconds or
  hours:minutes:seconds. A segment whose counter is missing leaves the
  total a lower bound, and the verdict not judged, with why.
- `state.holds()` and `state.owed()`.
- `state.chronology()`: the entries in the order of their events, each
  with its time on the recorders' clock.
- `state.findings()` and `state.open_anomalies()`: deviations and
  anomalies by number, with the dispositions against each, and the
  open anomalies with their owners.
- `state.files()` and `state.late()`.

## The projection to full level

A low-level reading is projected to full level to see whether the full
level will fit. The level L is in dB relative to full level, negative
below it, and every reading is multiplied by 10^(−L/20) exactly. From
the readings and the header, `state.reading(entry)` (or
`model.project(...)` with the numbers) judges three headroom numbers:

- the drive fraction at full level against its window (the header's
  `drive_window`, or a third to two thirds of the range);
- the peak current at full level, the rms current times the plan's
  sigma, against the amplifier's peak current rating (and, when the
  header gives a continuous rating, the rms current against it, which
  is what a sine test budgets);
- the peak voltage at full level, the rms voltage times sigma, against
  the rail.

A worked example at −12 dB, with a sigma of 3, a peak rating of
800 A and a rail of 250 V:

| Reading at −12 dB | × 10^(12/20) = 3.981 | At full level | Against |
|---|---|---|---|
| 63 A rms | 250.8 A rms | 752.4 A peak (× 3) | 800 A: within, 94 percent |
| 20 V rms | 79.6 V rms | 238.9 V peak (× 3) | 250 V: within, 96 percent |
| drive 0.12 | 0.478 | 0.478 | 1/3 to 2/3: within |

The factor is 3.981, not 4, so 63 A projects to 250.8 A rather than
252 A. At −6 dB the factor is 1.995 and at −3 dB 1.413.

A check that cannot be made reads **not judged**, with why: no reading,
a reading of zero (the meter or the amplifier is not reading), or a
rating or sigma the header does not give. The projection's verdict is
within only when every check was judged within; a projection with any
check not judged is not judged, never a pass. A level above full, a
reading that is negative or not a finite number, a drive fraction
above 1, or a sigma or rating of zero is refused with a message naming
it.

`state.reading(entry)` also compares the row's volts per amp with the
previous level of the same run, and its readings with the last run of
the same setup (the same level on the same axis in an earlier run of
the test, or of a test of the same kind and configuration), and flags a
ratio that moved more than 10 percent: a plant whose ratio of voltage
to current changed between levels has changed. The header used is the
header as it stood when the row was written.

## Corrections

`log.correct(entry, ...)` writes a correction carrying the new values:
any of the entry's fields (None strikes an optional one), and its event
time, free text, test, axis or run. The corrected view applies every
correction in the order written; `log.current(id)` is one entry
corrected, with the corrections it took in `corrected_by`, and
`log.entry(id)` is the entry as it was stored. A correction belongs to
the corrected entry's test. A required field cannot be struck, a
finding's number is never changed, and a correction is not itself
corrected: the original entry is corrected again.

## Times and the recorders' clock

Every time is stored with its UTC offset, so the hour that repeats
when daylight saving time ends is never ambiguous, and an interval
across the change is the real interval. An entry entered more than
ten minutes after its event is a **late entry**, marked as such
wherever the log is shown.

The computer's clock and the recorders' clock drift apart. A
`clock_check` entry records the recorders' clock less this computer's,
in seconds (`offset_s`), with the recorder it was read against.
`log.recorder_time(entry)` gives an entry's time on the recorders'
clock, using the latest clock check at or before it (of one recorder,
when named); with no clock check before it, there is no recorders'
time, and it says so (None) rather than guess.

## Integrity, and its limits

The entries of each test are chained by SHA-256. Each entry carries
the hash of the entry before it in the same test (`prev`), and its own
hash taken over its canonical form, which includes `prev`. The entries
between tests have a chain of their own, and every chain starts from
the log's header, so an edited header breaks them all. Because the
chain is per test, the entries of one test check on their own.

`log.verify()` checks every chain and names the first broken link by
test and entry: an entry whose hash does not match its contents, an
entry whose previous hash is not the entry before it (one was removed
or moved), and ids that skip or run backward.

What this detects is **accidental damage, a file cut short, and casual
edits**. It is **not proof against deliberate alteration**: anyone who
edits an entry can compute every hash after it again, and the edited
file then checks. A file cut off at the end of an entry also leaves a
valid chain. That is found only against a head recorded outside the
file: `log.head(test)` is the hash of the test's last entry, and
`log.verify(heads={1: head})` reports when a recorded head is no
longer in the log. A report's data index can carry a test's head, so
the record outside the file anchors the record inside it. The login
and machine name every entry records beside the typed initials are
recorded, not authenticated.

## One writer, and a second copy

A log open for writing holds a lock file beside it (`article-7.vdlog.lock`).
A second writer is refused, with a message naming the login and
machine that hold the log and since when; readers open it read-only
and are never refused. The lock is the operating system's, so a writer
that crashes leaves nothing holding the log.

Given a `mirror`, every entry is copied to a second path as well, the
lab's share or the data folder. A mirror that cannot be written (the
share is away) is reported in `log.mirror_problem` and never costs the
entry its place in the log itself; when the mirror can be written
again, the whole log is copied there, so it catches up.

## Exports

The `.vdlog` is its own JSON export, and the one format the log is read
back from. Three views are written out from it
(`visualdynamics.core.run_log_export`), each for the whole log, one
test (`test=1`), or the entries between tests (`test=None`):

- **Plain text** (`to_text`), for printing or attaching to a report:
  every entry in the order written, with its time and offset, author,
  login and machine, place and fields. A corrected entry is marked
  `[CORRECTED by #4]`, and the correction says what it changed and
  what the value was (`stop: power loss (was amplifier trip)`); a late
  entry is marked with when it was entered, and an entry under a clock
  check with its time on the recorders' clock. The lines that could not
  be read, the chains' heads and what verifying found close it.
- **CSV** (`to_csv`): a row per entry as stored, corrections included,
  with `corrected_by` and `corrects` columns, a `late` column, the time
  on the recorders' clock, and the fields as JSON.
- **The machine log** (`machine_log_csv`): the entries a lab's own
  machine log keeps (trips and aborts, setting changes, clock checks,
  people in and out, restarts, and the tests opened and closed, which
  say what was running), corrected, one row each: the local date,
  time and UTC offset, the recorders' time, the event in words, its
  details, the test and its title, the axis and run, the author, the
  entry's id, and whether a correction changed it.

```python
from visualdynamics.core import run_log_export

run_log_export.export(log, 'article-7', 'text')          # article-7.txt
run_log_export.export(log, 'test-1', 'csv', test=1)      # test-1.csv
run_log_export.export(log, 'machine', 'machine')         # machine.csv
```
