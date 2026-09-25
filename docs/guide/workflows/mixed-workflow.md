# The random and sine workflow

A random and sine test runs both at once on the same shakers: a
broadband random held to its power spectral density at the control
channels, with one or more sine tones sweeping under it at a controlled
level. It is one recording that has to meet two requirements, and the
two are judged separately — the random by [its own
workflow](random-workflow.md), the sweep by [its own](sine-workflow.md)
— because a PSD says nothing about a tone's level and a tracked level
says nothing about the band around it. What this type adds is the
project that holds both halves and the report that reads both.

The walkthrough is the demonstration plate's mixed run: the random
specification of the random workflow, with a quiet sweep from 100 to
800 Hz riding about 10 dB under it — quiet enough that only the
extraction can read it, which is the point. The recording is generated
by the controller (`generate_plate_sine.py` in the generators
repository) and lands at `stressdata/plate/mixed.nc4`.

| Step | In the app | In a script |
|---|---|---|
| Import the run | drag the `.nc4` onto the window | `project.import_file(path)` |
| Both specifications | arrive with the run — the random's and the sweep's | (read from the file's two environments) |
| The random half | PSDs, octave bands and coherence, as the random workflow does | `visualdynamics.random_vibration_run(path)` does all of it |
| The sine half | Extract Sine Levels on the time history's bar | `project.extract_sine(project.time_history)` |
| Geometry and photos | drag them in, link them | `project.add(...)`, `project.link(...)` |
| Report | *Generate Report* on the bar, the project row selected | `project.generate_report('mixed')` |
| Save | Save Project As… | `project.save('mixed.vdyn')` |

## What the file says, and what the project becomes

A controller file names its environments, and a file with a random
environment and a sine environment that both drove the article is a
**Random and Sine** project the moment it is imported. Both
specifications arrive: the random's as the PSD target with its warning
and abort limits, the sweep's as its tones with their schedules and
bands. The tree shows the slots of both halves — the random's PSD,
octave bands and coherence, then the sine specification and the sine
levels — gray until each is filled, and the report last.

A sweep that controlled a *virtual point* through a response
transformation, while the random controlled the raw channels, imports
the same way: the sine specification is over the transformation's rows,
numbered from 1, and the rows' time histories ride the recording beside
the raw channels, which is what the levels are read from.

## The random half

Exactly the [random vibration workflow](random-workflow.md): average
the PSDs from the control channels with the frames the controller
used, band them and the specification onto proportional bands, and
measure the multiple coherence. The one-call `random_vibration_run`
does all of it from a script.

One thing to know that a random-only run would not raise: the control
spectra are of the whole recording, sweep included. A tone adds its
power across the band it swept, so a control channel reading high
there may be the tone rather than the random. The report says so where
that comparison is drawn.

## The sine half

Exactly the [sine sweep workflow](sine-workflow.md): Extract Sine
Levels reads each tone along its own trajectory out of the same
recording, with the random treated as the noise it is extracted from,
and the levels are judged against the tone's required amplitude with
its bands. The extraction is what makes this test readable at all —
the sweep here sits 10 dB under the random, where a spectrum shows
nothing and a controller's live tracker reads high.

## The report

*Generate Report* with the project row selected writes one report for
the run. It opens on the random half's verdict, then the front matter
and the recording once, then the random's control comparisons and
compliance readings, then the sweep's level figures and deviation bars,
then the data-quality readings and the conclusions, each written for a
run that was both. The prose carries the two facts a reader of one
half's report would not have: that the control spectra include the
sweep, and that the recording is not Gaussian by design — a pure tone's
kurtosis is 1.5, so a channel under three is the sweep, not clipping.

```python
import visualdynamics

project = visualdynamics.random_vibration_run('mixed.nc4')
project.extract_sine(project.time_history)
project.generate_report('mixed')
project.export_report(project.report, 'mixed_report.html')
project.save('mixed.vdyn')
```

## Where judgment lives

- **Two requirements, one recording.** Neither half's reading is
  corrected for the other: the random's PSDs include the tone's power,
  and the tone's level is read with the random as noise. The report
  states both rather than hiding either, because a reader deciding
  whether an exceedance is real needs to know what else was in the
  band.
- **The verdict is the random's.** It is read off the octave-band
  comparison, as the random report reads it. The sweep is judged tone
  by tone in its own figures and bars, which is how a sweep is judged
  anywhere.
- **Which specification the run was *for* is still the engineer's
  call.** Both are in the project and both are in the report; the
  type only says that both belong there.
