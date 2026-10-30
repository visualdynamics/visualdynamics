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
| `test_opened` | the test | title, test_kind, article, serial, procedure |
| `test_closed` | the test | reason |
| `header_change` | field, value | was |
| `clock_check` | offset_s | recorder, method |
| `person_in`, `person_out` | person | role |
| `brief` | | attendees, topics |
| `checklist_reading` | item | reading, taken_by |
| `setup_step`, `post_test_step` | step | result |
| `run_start` | | level_db, counter |
| `level_row` | | level_db, voltage, current, drive_fraction, answer, go |
| `sweep` | | direction, n, of |
| `shot` | | polarity, n, of, level_db |
| `run_stop` | | counter, reason |
| `abort` | stop | channel, band, panel_fault, counter, called_by, heard, records_read, over_test |
| `cause_found` | cause | refers_to |
| `path_decided` | path | refers_to |
| `restart` | | counter_at_stop, counter_at_restart, cause |
| `notch` | | channel, band, depth_db, planned |
| `deviation` | | number, fact, agreed_by |
| `anomaly_opened` | | number, fact, owner |
| `anomaly_dispositioned` | number | disposition, by |
| `adjustment` | | what |
| `photo` | | subject |
| `setting_change` | setting | value, was |
| `opinion_given` | | to, opinion |
| `observation`, `note` | | |

The kind of stop is one of controller ramp, amplifier trip, power loss
and stop word: a hard shutdown may have over-tested the article, which
is why it is asked first. A level is a signed number of decibels
relative to full level (−12 for twelve below). `refers_to` names an
earlier entry by its id. A name typed into a deviation (`agreed_by`)
records who agreed at the console; the log does not track signatures or
approvals.

Deviations and anomalies are numbered across the whole log, D-1, D-2,
A-1, in the same form as the report's findings register, so a number
names one finding however many tests the log spans.

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
