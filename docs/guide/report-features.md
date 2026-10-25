# Report Features

A report feature is an optional addition to a generated report. Each
one is off until it is turned on, and a report with none turned on is
exactly the report Visual Dynamics writes otherwise, byte for byte. A
lab turns on what its reports need; nothing else moves.

![The report editor with the Test report preset chosen: the Report
Features panel in the pane, and two control channels on the page drawn
on log-log axes with shared decades and named at their
ends](images/report-features.png)

## Choosing features

With no block selected, the report editor's pane holds the report's
title and marking and, under them, the **Report Features** panel. Each
feature has a check box, grouped by what it adds to the report:
Structure, Results, Records, Issue control and Figures. Its tooltip
says what it does. A feature that cannot be turned on is grayed out,
with the reason beneath it: *not yet available*, or the feature it
needs (Shared decades needs Plot conventions). Turning a feature off
also turns off any feature that needs it, and the panel says so.

From a script, the same choice is one call:

```python
project.report_features('Report', enable=['plot_conventions'])
project.report_features('Report', disable=['shared_decades'])
project.report_features('Report', preset='test_report')
```

The call returns what is enabled afterwards and any notes, and is
recorded in the journal as that line. It refuses, changing nothing, an
unknown feature or preset, a feature that is not available yet, and a
feature whose requirement is not met; a feature can be enabled in the
same call as the one it needs. The choice is saved with the report in
the project file, and in a `.vdreport` template, so a house template
carries its choice of features from one test to the next. A report
saved before features existed opens with none.

## Presets

The drop-down at the top of the panel sets a starting point.

| Preset | Features | For |
|---|---|---|
| Visual Dynamics default | none | quick looks, and every existing report |
| Test report | skeleton sections, requirement and results, findings, identity and issue, data index, runs and chronology, record captions, plot conventions, shared decades, check blocks, calibration in date | a report a quality signer and a customer will read |
| Flash report | identity and issue, requirement and results, findings, flash report | the one preliminary page a customer leaves the test with |

A preset names features that are not built yet. Choosing it turns on
the ones that are available and lists, under the panel, each one it
skipped and why. A set that matches no preset shows as *Custom*.

## Skeleton sections

**Skeleton sections** (group Structure) lays the report out on the
sections a test report needs, in the order its readers look for them,
and closes it with the line "End of report.", so a reader holding the
pages knows whether all of them are there.

| Section | What belongs in it | Filled by |
|---|---|---|
| 1. Identification | title, number and revision, the laboratory, the customer, the dates, who authorized it | the author |
| 2. Summary | what was tested and why, the result against each requirement, the deviations by number, the open anomalies | the author, above the pass/fail box and the generated description |
| 3. Requirement | the governing document, the plan and procedure, the tolerance clause, the completion criteria | the author |
| 4. Test Article | description, part and serial, configuration, condition | the author |
| 5. Setup and Method | facility, fixture, control strategy, settings, equipment and calibration | the author |
| 6. Instrumentation | the channel table and any sensor averaging | the geometry, DOF scenes, photographs and channel table |
| 7. Chronology | the runs in order, interruptions and failures, from the log | the author |
| 8. Results | the control against the tolerance and every other result | every other generated figure and its text |
| 9. Deviations, Anomalies and Nonconformances | each one in full, or None | the author |
| 10. Discussion and Interpretation | what the results mean, and by whom, or None | the author |
| 11. Conclusion | the statement of conformity and the rule it was decided by | the author, with the generated reading guide under it |
| 12. Record and Data Package | what is attached, referenced and kept | the author |

A generated report's own sections fall under the skeleton by what they
hold: a section with the pass/fail box, or one of text only before the
first figure, goes under the Summary; a section with the geometry, a
DOF scene, a photograph or the channel table goes under
Instrumentation; one of text only after the last figure goes under the
Conclusion; everything else goes under Results. Their headings sit one
level below the skeleton's. The report's blocks themselves are not
moved: the layout is made when the page is drawn, and turning the
feature off gives back the page as it was.

**Owed sections.** A section only the author can write shows, until it
says something, as a framed card reading NOT YET WRITTEN with what
belongs there. Instrumentation and Results are owed only when nothing
falls under them. The program never writes an author's words for them:
the article, the method, the chronology, the findings, the discussion
and the conclusion stay the engineer's. What it adds is fixed text:
the conclusion always ends "The results relate only to the item
tested.", and the discussion is labeled with who made the
interpretation, or *not stated* until someone is named.

In the editor, click a section's heading to write it. The pane shows
what belongs in the section above the Markdown editor; for Deviations,
Anomalies and Nonconformances and for Discussion and Interpretation a
**None** button writes "No deviations. No anomalies. No
nonconformances." or "No interpretation is offered beyond the
results." in one step, and the discussion has an **Interpretation, by**
field. A section's text can hold pipe tables:

```text
| Run | Level | Duration |
|-----|------:|---------:|
| 1   | -6 dB |    2 min |
| 2   |  0 dB |   10 min |
```

**Before export.** Export first lists every section still owed (not yet
written, nothing in it, or a discussion that does not say by whom) and
asks whether to export anyway. A script asks the same question:

```python
project.report_features('Report', enable=['skeleton_sections'])
project.report_section('Report', 'article', 'Bracket, part 1234, serial 7.')
project.report_section('Report', 'deviations', none=True)
project.report_section('Report', 'discussion', 'The notch is the fixture.',
                       by='J. Doe')
project.report_check('Report')
# ['1. Identification: not yet written', '2. Summary: not yet written', ...]
```

`report_section` refuses an unknown section, None for a section that
always has something to say, and an attribution outside the
discussion. `report_check` refuses a report without the skeleton,
since there is then nothing to check it against.

## Calibration in date

**Calibration in date** (group Records) answers the first thing a
quality signer checks: whether every sensor was in calibration on the
test date. MIL-STD-810H (2019), Part One, Task 406, asks a final report
for the test equipment with its calibration status (paragraph
406.2.2.2), and SMC-S-016 (2014) keeps calibration dates in the test
documentation file (paragraph 4.9.1). The channel table already holds
each sensor's expiration, its Expiration column; the feature compares
it with the dates of the test.

Each channel gets one of four answers:

| Answer | When |
|---|---|
| in date | the expiration is on or after the last test day |
| expired | the expiration is before the first test day; the report says how many days before |
| lapses during the test | in date on the first test day, out of date before the last |
| no date given | the Expiration cell is blank: nothing was judged, and the channel is never counted in date |

A sensor is taken as in calibration through its expiration date: one
due on the 24th may be used on the 24th, not on the 25th.

**Where it shows.** The channel table gains a **Calibration** column
with each channel's answer, and an expired or lapsing channel's
Expiration and Calibration cells are shaded red. Under the table a
paragraph gives the count in date and names each channel out of date
by its number, DOF and quantity (a force and an acceleration at one
DOF are two sensors), with how many days, and the channels with no
date, which were not checked:

> **Calibration on the test dates, 2026-10-24 to 2026-10-26: 8 of 12
> channels in date.**
>
> - Channel 2 (104Z+ acceleration, control) expired 2026-09-30, 24 days
>   before the test.
> - Channel 3 (110Z+ acceleration, control) lapses during the test: in
>   date through 2026-10-25, not on the last test day.
> - No expiration date given, so not checked: channel 4 (113Z+
>   acceleration, control), channel 11 (1304Z+ force).

"All channels in date" is written only when every channel had a date.
A table with no expirations at all, a table with no channels, and a
report with no test date entered each say that calibration was not
checked, and why. With Skeleton sections on, the column and the
paragraph fall under Instrumentation.

**The test date.** Select the channel table in the report editor and
type the **Test date**, and the **Last test day** for a test over
several days, as YYYY-MM-DD. A date written another way is refused
rather than guessed (10/04/2026 is October 4 in one country and April
10 in another), and so is a last day before the first. A file whose
Expiration cell holds something that is not a date names the cell on
the page: *channel table row 6 (channel 6), Expiration: 'June' is not a
date*. The dates are kept with the report; a `.vdreport` template
leaves them out, so a house template never judges the next test
against the last one's date.

**Setup and Method.** That section is the author's and is never
written by the program. With Skeleton sections on, its pane has an
**Insert Calibration Status** button, which puts the equipment list at
the cursor (a table of the sensors with their make, model, serial
number, expiration and answer, then the lines of the check) for the
author to keep, change or delete.

From a script:

```python
project.report_features('Report', enable=['calibration_in_date'])
project.report_test_date('Report', '2026-10-24', end='2026-10-26')
check = project.report_calibration('Report')
check.summary()          # the paragraph, line by line
[c.status for c in check.channels]
project.report_section('Report', 'setup',
                       project.report_calibration_text('Report'))
```

`report_test_date` is recorded in the journal and refuses a report
without the feature or without a channel table. `report_calibration`
refuses a report with no test date, since there is then nothing to
judge against. What to do about a lapsed sensor stays the engineer's.

## Plot conventions

**Plot conventions** (group Figures) puts a report's figures under
three habits a test report is expected to keep:

- An ASD or SRS figure opens on log-log axes. A specification's
  breakpoints are joined by straight lines, and its tolerance band has a
  constant width, only on log-log axes; without the feature, a
  narrowband ASD comparison opens on a linear frequency axis. A figure
  on linear ordinates (a phase, a ratio in decibels) keeps its axes.
- Curves are told apart by line style as well as color: solid, dashed,
  dotted, dash-dot and so on, drawn in the legend too. The gray
  specification behind a comparison stays solid, and a figure that
  already pairs each curve with a dashed companion keeps that pairing.
- A figure of up to eight curves names each curve at its right-hand
  end, so it reads in a monochrome copy and to a reader who does not
  see color.

**Shared decades**, its option, opens every ASD figure in the report on
the same whole decades, and every SRS figure likewise, so two figures
can be compared by eye. The window covers what each figure opens on (a
comparison opens on its specification's band) and is widened outward
to whole decades. Figures of different quantities or units, such as an
acceleration ASD and a force ASD, are scaled apart.

## What is planned

The other features are registered so the panel shows what is coming,
and are grayed out until each is built:

| Feature | Group | What it will add |
|---|---|---|
| Runs and chronology | Structure | run identity, a results section per run, and the runs in time order with duration at level |
| Requirement and results | Results | the requirement in words and a met or not met row per criterion; the pass/fail box labeled a screening without one |
| Findings register | Results | deviations, anomalies and nonconformances in four parts, the fact filled from the measurement |
| Check blocks | Results | a table for each check the project holds, a row per channel against its criterion |
| Responses against prediction | Results | each response against its prediction at the level run |
| Cross-axis and limit channels | Results | cross-axis against its limit, and where limit channels held the control down |
| Control-channel uncertainty | Results | each control channel's uncertainty against one third of the tolerance |
| Data index | Records | every file the report rests on, with its size and SHA-256, checked again at export |
| Identity and issue | Issue control | number, revision, status and page n of N on every page, a paginated PDF, and amendments |
| Flash report | Issue control | a one-page preliminary report from the day's analysis |
| Record captions | Figures | a caption block on each spectrum, SRS and sine figure, read off the record |

What stays the engineer's to write (the cause and effect of a finding,
its approval, the discussion and the conclusion, the signatures) is
never a feature.
