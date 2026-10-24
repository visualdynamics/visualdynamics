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
| Skeleton sections | Structure | every section a test report needs, owed sections marked until written or set to None, and a check before export |
| Runs and chronology | Structure | run identity, a results section per run, and the runs in time order with duration at level |
| Requirement and results | Results | the requirement in words and a met or not met row per criterion; the pass/fail box labeled a screening without one |
| Findings register | Results | deviations, anomalies and nonconformances in four parts, the fact filled from the measurement |
| Check blocks | Results | a table for each check the project holds, a row per channel against its criterion |
| Responses against prediction | Results | each response against its prediction at the level run |
| Cross-axis and limit channels | Results | cross-axis against its limit, and where limit channels held the control down |
| Control-channel uncertainty | Results | each control channel's uncertainty against one third of the tolerance |
| Data index | Records | every file the report rests on, with its size and SHA-256, checked again at export |
| Calibration in date | Records | whether each sensor was in calibration on the test date |
| Identity and issue | Issue control | number, revision, status and page n of N on every page, a paginated PDF, and amendments |
| Flash report | Issue control | a one-page preliminary report from the day's analysis |
| Record captions | Figures | a caption block on each spectrum, SRS and sine figure, read off the record |

What stays the engineer's to write (the cause and effect of a finding,
its approval, the discussion and the conclusion, the signatures) is
never a feature.
