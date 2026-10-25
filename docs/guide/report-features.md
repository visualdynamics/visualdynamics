# Report features

A report feature is an optional addition to a generated report. Each
one is off until it is turned on, and a report with none turned on is
exactly the report Visual Dynamics writes otherwise, byte for byte.

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
| 12. Record and Data Package | what is attached, referenced and kept | the author, with the data index under it when Data index is on |

A generated report's own sections fall under the skeleton by what they
hold: a section with the pass/fail box, or one of text only before the
first figure, goes under the Summary; a section with the geometry, a
DOF scene, a photograph or the channel table goes under
Instrumentation; one of text only after the last figure goes under the
Conclusion; everything else goes under Results. Their headings sit one
level below the skeleton's. The report's blocks themselves are not
moved: the layout is made when the page is drawn, and turning the
feature off gives back the page as it was.

### Owed sections

A section only the author can write shows, until it
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

### Before export

Export first lists every section still owed (not yet
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

## Data index

**Data index** (group Records) puts in the report every file it rests
on: each file's name, what it holds, its format, its size in bytes and
its SHA-256 checksum, the fingerprint that changes if one byte of the
file changes, so a copy can be shown identical to the original years
later. A reader can then ask for the file that holds a channel by its
name and know that what arrives is the file the report was written
from.

| File | Holds | Format | Bytes | SHA-256 | Role | Check |
|---|---|---|---|---|---|---|
| run11.nc4 | Channel Table: 12 channels; Time History: 12 records, 114688 samples, channels 101Z+ to 1313Z+, 4096 Hz | Rattlesnake controller output (.nc4) | 17,735,629 | d1c7ab5b…59adc0 | referenced | matches |
| settings.csv | settings list | comma-separated text (.csv) | 412 | 6f0e21a4…b7c913 | attached | matches |

**Where the entries come from.** Every file imported into the project
is indexed as it is read, with what was read out of it, and the index
is saved in the project file. Importing a project file brings its
index with it. Removing the last object read from a file takes the
file out of the index, unless something still in the project was
derived from one of those objects. Documents that go with the report
(the settings list, the log, the as-run procedure, the flash report)
are added with **Insert, Attached File…** in the report editor, which
asks what the file is and whether it is *attached* (sent with the
report) or *referenced* (named in the report and kept in the archive).
An attached text file of up to 64 KiB is printed whole under the
index: a CSV or TSV as a table when every row has the header's width,
anything else as it was written. **Insert, Remove from Index** takes out
a file added by hand. A file over 32 MiB is hashed off the window's
event loop, with the progress bar and Cancel; a canceled import keeps
the file's objects and the status line says the file is not in the
index.

**The check at export.** Export hashes every indexed file again before
it writes the report. A file is looked for where it was indexed, then
by its name beside the project file. Each row's Check column says
*matches*, *CHANGED* or *NOT FOUND*, and a line under the table says
when the files were checked and how many matched. When any file does
not match, Export lists them and asks whether to export anyway, and the
status line after the export repeats what the check found, or that the
index is empty.

**Provenance.** Under the index, a line names the version of Visual
Dynamics and the project file with its size and checksum, as the check
found it. The analysis recipe follows: each derived object in the
project, the verb that computed it and what it was computed from, read
off the project's provenance, so every figure traces to an object and
every object to a file.

**What stays the author's.** Where the archive is, who keeps it, and
which records travel with the report are the author's words. With
Skeleton sections on, the index goes under section 12, Record and Data
Package, after what the author writes there, and the section stays
owed until the author writes it. Without the skeleton, the index
closes the report under the heading Record and Data Package.

From a script:

```python
project.index_file('settings.csv', 'settings list', role='attached')
project.index_file('Y_run11_full.csv', 'Y axis, run 11, all channels')
project.report_features('Report', enable=['data_index'])
project.verify_data_index(search=['/archive/test-0037'])
# ['random.nc4: matches', 'settings.csv: matches', 'Y_run11_full.csv: not found']
project.unindex_file('Y_run11_full.csv')
```

`index_file` refuses a file it cannot find, a folder, an unknown role,
and an entry that does not say what the file holds. The same file with
the same checksum indexed twice is one entry; the same path with
different contents is a second entry, since it is a different file.
Any other source of files adds its entries the same way, naming itself
in `source`, so a project keeps a single index. `project.export_report` checks the index before it
writes, as Export does.

## Check blocks

**Check blocks** (group Results) puts the checks Visual Dynamics reads
into the report. Each check the report can make becomes a block under a
**Checks** heading at the close of the results (under 8. Results when
skeleton sections are on): the check's title, its verdict in a word,
what it found in the reading's own words, the criterion it was judged
against, a table with a row per channel (or mode, pass or segment) and
its figures. The numbers are the reading's own: each check calls the
same function its view in the app and a script call.

| Check | What it reads | Made |
|---|---|---|
| Time statistics | a random test's record: mean, rms, skewness, kurtosis over the record and frame by frame, crest factor, each against its band; the histogram against the Gaussian and the kurtosis frame by frame for each channel outside a band, or for the channel furthest from Gaussian when none is | from the report |
| Shock validity | a shock record: each acceleration channel integrated raw over each shock window, and whether its velocity returns to rest; the velocity drawn | from the report |
| SRS points rule | each measured shock response spectrum against its requirement's band, point by point; the distance from the requirement drawn against the band | from the report |
| Classical pulse | a shock record against the envelope of its nominal pulse, with its peak, duration, edges and velocity change | from its own block |
| Sine distortion | each tone of a sine record: the record less the tracked tones against the tones, at every line; the distortion drawn along the sweep | from the report, one block per tone |
| Sweep rate | each pass of the sweep through a mode's half-power band against the rule's rate; the sweep drawn through the bands | from the report with a shape set, or from its own block with the modes the plan names |
| Signature shifts | two signatures before and after the test, mode by mode against frequency and amplitude limits; the two drawn over each other | from its own block |
| Replication segments | a replicated record judged segment by segment, beside the whole-record rms error | from its own block |

A check is made **from the report** when the report already draws
every object it reads, so a random report carries the time statistics
of its record, a shock report the velocity check and the points rule,
and a sine report the distortion of each tone; the time statistics are
not made for a shock, transient or sine report, whose records are not
meant to be Gaussian. MIL-STD-810H (2019), Method 514.8, paragraph
4.2.2.1, asks for a drive that departs from a Gaussian distribution to
be documented with its time history, its skewness and kurtosis, and its
probability density: the time statistics block carries the last three
beside the time histories the report already draws.

A check that needs a fact no object holds is made from a block of its
own, which carries what it reads and its settings and is saved with the
report and its template:

```python
project.report_features('Report', enable=['check_blocks'])
project.report_check_block('Report', 'pulse', shape='half-sine',
                           peak=490.3, duration=0.011)
project.report_check_block('Report', 'sweep_rate',
                           modes=[[22.0, 40.0], [180.0, 20.0]])
project.report_check_block('Report', 'signature', source='Survey before',
                           measured='Survey after')
project.report_check_block('Report', 'segments', segments=[
    [6.0, 30.0, 'stationary', 'rough road'],
    [31.9, 32.7, 'shock', 'pothole']])
```

The settings are the keyword arguments of the check's reading (the
pulse's peak in SI); a check made from the report, given a block of its
own, is made from the block, so its settings hold
(`kurtosis_band=[2.9, 3.1]` for the time statistics). The call refuses,
changing nothing, an unknown check, a field or setting the check does
not take, and a required setting left out. In the editor, clicking a
check's own block shows what it reads and its settings; Remove takes it
away.

**What judged nothing says so.** A check reads **Passed** only when it
judged every channel and each was inside its criterion. **Flagged**
means at least one was outside. A channel the check cannot read, such
as a dead or constant channel, is a row reading *not judged* with the
cause, and the check reads **Partly judged** when the others passed, or
**Not judged** when none could be read. A reading that refuses its
input (a shock record holding a sample that is not a number, two spectra at
different Q, two signatures of different quantities) gives the
reading's own reason in place of a verdict.

**When the readings are made.** Some readings take a while (the
distortion check rebuilds every tone on every channel), so the app
makes them with a progress bar and Cancel before it draws the page, and
keeps them until what they read or their settings change. A check whose
reading was canceled reads Not judged, and is read again the next time
the page is drawn. A script that renders or exports the report makes
them as it renders.

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

### Where it shows

The channel table gains a **Calibration** column
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

### The test date

Select the channel table in the report editor and
type the **Test date**, and the **Last test day** for a test over
several days, as YYYY-MM-DD. A date written another way is refused
rather than guessed (10/04/2026 is October 4 in one country and April
10 in another), and so is a last day before the first. A file whose
Expiration cell holds something that is not a date names the cell on
the page: *channel table row 6 (channel 6), Expiration: 'June' is not a
date*. The dates are kept with the report; a `.vdreport` template
leaves them out, so a house template never judges the next test
against the last one's date.

### Setup and Method

That section is the author's and is never
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

## Responses against prediction

**Responses against prediction** (group Results) reads each response
channel against the response its analyst predicted, scaled to the level
run, so the analyst can give the low-level run's answer (as predicted,
different and explained, or different and not explained) and the
report can carry the table that answer rests on.

**The predictions** are the project's: one predictions object holding,
per response channel, the predicted peaks (frequency, level and Q) or a
predicted curve with the frequencies of the modes it is compared at,
the input level the prediction is stated at, its source and whether the
customer supplied it, and the model's stated uncertainty. They are
entered from the console or a script, one channel a line, and saved in
the project file:

```python
project.predict('Predictions', '201Y+', 'asd', [(385, 5.8, 12)], unit='g',
                level_db=0, frequency_uncertainty=10, level_uncertainty=3,
                source='Stress analysis SR-12 rev B', customer=True)
project.predict('Predictions', '301Z+', 'transmissibility',
                [(560, 8, 8, (380, 600))], reference='1Z+')
```

An ASD's levels are in the unit squared per hertz (`unit='g'` for
g²/Hz, the way a PSD declares its unit), an amplitude's in the unit. A
transmissibility is a ratio and takes no unit and no level.

### The scaling

A prediction stated at full level and a run at L dB
relative to full level: the predicted ASD is multiplied by 10^(L/10)
and a predicted amplitude by 10^(L/20), so at minus 12 dB an ASD by
0.063 and an amplitude by 0.25. A transmissibility is not scaled. The
level run is stated on the block (minus 12 for a run at minus 12 dB);
left blank, the scaling a PSD holds for its comparison against the
specification is used, and a level stated that disagrees with it is
refused.

### Peaks are looked for in their band

Near each predicted peak, the
measured peak is read by the package's half-power reader inside the
peak's band: the predicted frequency plus or minus the model's stated
frequency uncertainty, or a band stated for the peak. A band that holds
no measured peak says so; the peak is never matched to one outside it.
To compare a mode the model placed far from where it was found, state a
band for it. The Q of an ASD is read where the ASD has fallen to half
its height, its half-power points.

**In the report**, a responses-against-prediction block (Insert,
Responses Against Prediction, once the feature is on and the project
holds predictions; or `project.report_prediction('Report', 'Response
PSD', 'Predictions', level=-12)`) draws a figure per channel, the
measurement over its prediction scaled to the level run on log-log
axes, the prediction dashed; then a table with a row per predicted
peak: the frequency, level and Q predicted and measured, the frequency
difference in percent, the ratio of the levels named once in its
column's heading as an ASD ratio or an amplitude ratio and given in
decibels, and the model's stated uncertainty. A note beneath names the
predictions' source (supplied by the customer, where it is), the peaks
with no measured peak in their band, and the channels not compared and
why. The figures and the table fall under the skeleton's Results.

Refused, with the cause named: a prediction of one quantity against a
record of another, or in units of another dimension; an ASD or
amplitude whose input level or run level is not stated; a peak with no
band to look in; two peaks of one channel whose bands overlap. A
channel with two records, a record that is zero or constant, or one
with its units undefined is listed as not compared. Whether a
difference is explained is never written by the program; the table
states the numbers, and the answer is the analyst's.

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

## Record captions

**Record captions** (group Figures) puts the caption block a record
carries under every figure of a measured spectrum in the report: the
channel, the run, the analysis settings and the requirement, read off
the record as [Report captions](report-captions.md) describes. Without
the feature the same block can be asked for one figure at a time, with
the figure's own check box in the editor; the feature asks for it on
every figure at once.

## What is planned

The other features are registered so the panel shows what is coming,
and are grayed out until each is built:

| Feature | Group | What it will add |
|---|---|---|
| Runs and chronology | Structure | run identity, a results section per run, and the runs in time order with duration at level |
| Requirement and results | Results | the requirement in words and a met or not met row per criterion; the pass/fail box labeled a screening without one |
| Findings register | Results | deviations, anomalies and nonconformances in four parts, the fact filled from the measurement |
| Cross-axis and limit channels | Results | cross-axis against its limit, and where limit channels held the control down |
| Control-channel uncertainty | Results | each control channel's uncertainty against one third of the tolerance |
| Identity and issue | Issue control | number, revision, status and page n of N on every page, a paginated PDF, and amendments |
| Flash report | Issue control | a one-page preliminary report from the day's analysis |

What stays the engineer's to write (the cause and effect of a finding,
its approval, the discussion and the conclusion, the signatures) is
never a feature.
