# The random and sine workflow

A random and sine test runs both at once on the same shakers: a
broadband random held to its power spectral density at the control
channels, with one or more sine tones sweeping under it at a controlled
level. It is one recording that has to meet two requirements, and the
two are judged separately — the random by [its own
workflow](random-workflow.md), the sweep by [its own](sine-workflow.md)
— because a PSD says nothing about a tone's level and a tracked level
says nothing about the band around it. What this type adds is the
project that holds both halves, and it makes two reports: a random
report and a sine report, each read the way its own test is.

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
| The random half | PSDs, octave bands and coherence, as the random workflow does | `visualdynamics.mixed_run(path)` does this and the next |
| The sine half | Extract Sine Levels on the time history's bar | `project.extract_sine(project.time_history)` |
| Geometry and photos | drag them in, link them | `project.add(...)`, `project.link(...)` |
| Reports | *Generate Report* on the bar, the project row selected: a random report and a sine report | `project.generate_report('random')`, `project.generate_report('sine')` |
| Or all of it at once | **Automatic** on the tree bar, once the project has its type | `project.work_up()` |
| Save | Save Project As… | `project.save('mixed.vdyn')` |

## What the file says, and what the project becomes

A controller file names its environments, and a file with a random
environment and a sine environment that both drove the article is a
**Random and Sine** project the moment it is imported. Both
specifications arrive: the random's as the PSD target with its warning
and abort limits, the sweep's as its tones with their schedules and
bands. The tree shows the slots of both halves — the random's PSD,
octave bands and coherence, then the sine specification and the sine
levels — gray until each is filled, and the two reports last.

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
there may be the tone rather than the random. The random report says
so where that comparison is drawn.

## The sine half

Exactly the [sine sweep workflow](sine-workflow.md): Extract Sine
Levels reads each tone along its own trajectory out of the same
recording, with the random treated as the noise it is extracted from,
and the levels are judged against the tone's required amplitude with
its bands. The extraction is what makes this test readable at all —
the sweep here sits 10 dB under the random, where a spectrum shows
nothing and a controller's live tracker reads high.

## The reports

*Generate Report* with the project row selected writes two reports for
the run, a random report and a sine report, the same two a random-only
run and a sweep-only run would get. Through 0.1.0a35 this type wrote
one combined report; the two separate reports let each half go to whoever
owns its specification, and let each read the stretch of the run it
needs.

The random report says what a reader of a random-only report would not
have to know. Its summary says a sweep ran under the random and is
judged in its own report. A section after the compliance charts says
the control spectra include the sweep, so a channel reading high
across the band the sweep passed through, and its RMS error and band
outside abort with it, may be the tone. Its kurtosis reading says a
channel under three is the sweep, not clipping, since a pure tone's
kurtosis is 1.5. The level beside its verdict is labeled **Random test
level**. The sine report's summary says the random is judged in its
own report.

```python
import visualdynamics

project = visualdynamics.mixed_run('mixed.nc4')      # both halves worked up
random_report = project.generate_report('random')
sine_report = project.generate_report('sine')
project.export_report(random_report, 'mixed_random.html')
project.export_report(sine_report, 'mixed_sine.html')
project.save('mixed.vdyn')
```

Or both in one call. `run_report` reads the run's type and writes
`<name>_random.html` and `<name>_sine.html`. Given `last`, the random
report reads only that many seconds from the end of the run, while the
sine report always reads the whole run, since a sweep cut short loses
tones:

```python
visualdynamics.run_report('mixed.nc4', 'reports/', last=120.0)
visualdynamics.run_report()                         # ask for the runs
```

A run with no sine specification is refused by name: `mixed_run` will
not work up half of what was asked for, and `random_vibration_run` is
the call for it.

## Where judgment lives

- **Two requirements, one recording.** Neither half's reading is
  corrected for the other: the random's PSDs include the tone's power,
  and the tone's level is read with the random as noise. The random
  report says so rather than hiding it, because a reader deciding
  whether an exceedance is real needs to know what else was in the
  band.
- **Each report has its own verdict.** The random report's is read off
  the octave-band comparison, and the level beside it is labeled
  **Random test level**: the random half's comparison is scaled to its
  specification when the run was at reduced level. The sine report
  judges each tone in its own figures and bars, always as measured.
- **Which specification the run was *for* is still the engineer's
  call.** Both are in the project, each judged in its own report; the
  type only says that both belong there.
