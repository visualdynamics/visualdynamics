# The transient workflow

A transient test names the motion itself, sample by sample — where a
shock test names a response spectrum and leaves the waveform to the
controller. So a transient run is judged as two waveforms of the same
thing: the target the controller was asked to reproduce, and what the
article actually did, compared at one control channel at a time.

The walkthrough is the demonstration plate's own replication run —
the plate's response to a burst at its corner drive, replicated by
two shakers under a real Rattlesnake controller — photographed at
every step. The run lands at `stressdata/plate/transient.nc4` and
regenerates with `generate_plate_runs.py --transient` in the
generators repository.

| Step | In the app | In a script |
|---|---|---|
| Import the run | drag the controller's `.nc4` onto the window | `project.import_file(path)` |
| Say what it is for | (the run says — the type switches itself) | `project.project_type == 'Transient'` |
| The target | (it arrives with the run) | `project.transient_specification` |
| Read the target | select it — one channel, drop-down for the rest | `spec.plot(...)` |
| Overlay | select the pair — the replication views appear | `plot_replication(history, spec, 'overlay')` |
| Waveform error | the plot bar's error view | `plot_replication(history, spec, 'waveform')` |
| PSDs of both | Compute PSDs in each averaging view | `project.compute_psds(name)` |
| Geometry and photos | drag them in, link them | `project.add(...)`, `project.link(...)` |
| Report | *Generate Report* on the bar, the project row selected | `project.generate_report('transient')` |
| Export | Export HTML… | `project.export_report(name, path)` |
| Save | Save Project As… | `project.save('transient.vdyn')` |

## In the app, step by step

### 1. Import the run

A transient replication run carries the measured control channels, the
drive forces, the channel table — and the target waveform itself,
which imports as the *Transient Specification*. The type switches to
*Transient* and the tree grows the slots a transient report needs.

The record is read as **playings**: the controller played the target
over and over, and the run's frames *are* those repeats — nothing is
detected or guessed. The time view shades them on the trace.

![The imported run: the control time histories with the playings
shaded, the Transient Specification in the tree
beside them](../images/transient-import.png)

### 2. Read the target

Select the specification on its own. One channel draws at a time —
its waveforms stacked are a thicket with no comparison in them — and
the drop-down on the bar reaches the rest.

![The target waveform at one control channel, the channel drop-down
on the bar](../images/transient-spec.png)

### 3. Compare waveforms

Select the measured history against the specification and the
replication views appear on the plot bar. **Waveforms** draws each
playing on the target, one control channel at a time; the grid below
is every channel through every playing, and picking cells points the
plot at them. Alignment is handled — a playing is compared where it
landed, not where the clock says it should have.

Picking works record by record too: expand both objects in the tree
and select one channel's records on each side, and the comparison
draws **the DOFs the two picks share**. Picks that share no channel —
the measured 104Z+ against the specification's 101Z+ — draw nothing,
with the reason in the plot, because overlaying two different
channels would be a comparison that looks like one and is not.

![The overlay: playing one of seventeen against the target at the
corner channel — the burst filling the window's first 0.6 s and the
ringdown decaying inside it — with the waveform-error grid for every
channel and every playing below](../images/transient-overlay.png)

The **waveform error** view reads the same grid as bars: the
difference between the two waveforms against the size of the target,
per channel per playing. Amplitude, phase and shape all move it, and
it is what the controller was minimizing.

![The waveform error by channel: two shakers replicating an
eight-channel motion land in the 6-26% band, and the bars say which
channels carry the error](../images/transient-error.png)

### 4. Then compare levels

Compute PSDs from both sides (each history's averaging view). The
target's own PSD arrives as a *specification* — a spectrum of a
specification is still a specification — so selecting the pair gives
the run a second reading: energy against frequency, the scale with
shape and phase divided out. No SRS here: a shock response spectrum
is how a *shock* test is judged, and that reading lives in the shock
workflow.

![The measured PSDs against the target's: the level comparison, with
the drive-force records filtered out because they have no
specification to answer to](../images/transient-level.png)

### 5. Geometry, photos, report

Drag the article's geometry in, declare its units, add the setup
photographs, link the group. The transient report is deliberately
shorter than the random one: there is no settled numerical compliance
measure for two transients, so the waveforms lead — the target, the
measured response, the overlay, the error bars — the level follows,
and the judgment is written in the conclusions.

![The exported report: the target waveform and the measured control
response, with the playings marked](../images/transient-report.png)

## The same run, headless

And the script never has to be reconstructed by hand: the **console tab** along the bottom of the window writes it live as you click — every act of the session as the line that replays it. Expand it, copy the stretch you want, and it runs as-is.


```python
import visualdynamics
from visualdynamics.plot import plot_replication

project = visualdynamics.Project('Plate Transient Replication')

# 1. import — the target arrives as the Transient Specification
project.import_file('transient.nc4')
assert project.project_type == 'Transient'

history = project.time_history
spec = project.transient_specification

# 2-3. the waveform views, headless
plot_replication(history, spec, 'overlay', path='overlay.png', show=False)
plot_replication(history, spec, 'waveform', path='error.png', show=False)

# 4. the spectral pair: the target's PSD is a Specification
project.compute_psds(spec)
project.compute_psds(history)

# 5. geometry, photos, report
project.add('Geometry', visualdynamics.import_file('geometry.npz'))
project.geometry.define_units('m')
project.link(*project.names)
project.set_basis(*project.names)
project.generate_report('transient')
project.export_report(project.report, 'transient_report.html')
project.save('transient.vdyn')
```

## Where judgment lives

- **Whether the reproduction is close enough.** There is no band to
  fall outside: the overlay is the evidence, and the waveform and RMS
  error bars quantify it without deciding it. The conclusions section
  is where the decision is written. The demo run's 6-26% errors are
  themselves informative: two shakers cannot exactly reproduce an
  eight-channel motion, and the numbers say how close they came.
- **Which channels are controls.** The controller's file says which
  channels it controlled to; monitors ride along. The report compares
  where a target exists and leaves the rest drawn but unjudged.
