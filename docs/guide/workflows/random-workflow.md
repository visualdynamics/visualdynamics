# The random vibration workflow

A controller run in, a compliance report out. A random test is judged
against a band: the specification says what the control PSD must be,
warning and abort lines say how far it may stray, and the report's job
is to show where the run sat inside that band, channel by channel.

The walkthrough is the demonstration plate's own random test — eight
control channels against four shakers, controlled to a shaped
specification by a real Rattlesnake run — photographed at every step.
The run ships with the repository (`testdata/plate/random.nc4`), so
every step can be followed exactly.

| Step | In the app | In a script |
|---|---|---|
| Import the run | drag the controller's `.nc4` onto the window | `project = visualdynamics.random_vibration_run(path)` |
| Say what it is for | (the run says — the type switches itself) | `project.project_type == 'Random Vibration'` |
| Control PSDs | Compute PSDs in the averaging view | `project.compute_psds('Time History')` |
| The octave-band view | Octave Bands on the PSD's bar | `project.compute_octave('Time History PSDs')` |
| Multiple coherence | Multiple Coherence in the averaging view | `project.compute_multiple_coherence('Time History')` |
| Compare against the spec | select PSD and specification together | `plot_comparison(psds, spec)` |
| Geometry and photos | drag them in, link them | `project.add(...)`, `project.link(...)` |
| Report | *Generate Report* on the bar, the project row selected | `project.generate_report('random')` |
| Export | Export HTML… | `project.export_report(name, path)` |
| Save | Save Project As… | `project.save('random.vdyn')` |

## In the app, step by step

### 1. Import the run

Drag the controller's save onto the window. A Rattlesnake random run
brings the control time histories, the specification with its warning
and abort lines, and the channel table in one file — and says which
environment drove it, so the project type switches to *Random
Vibration* by itself. The tree's grey slots then say what a finished
report still needs.

![The imported run: the control time histories drawn, the
specification and channel table beside them in the tree, and the
type's remaining slots in grey](../images/random-import.png)

### 2. Compute the control PSDs

**Compute PSDs** in the time history's averaging view averages over
the run's own frame length and framing — the numbers the controller
used, read from the file, not guessed — and the view shows exactly
those frames before the button is pressed.

![The control PSDs: eight acceleration channels and the drive forces,
averaged over the run's own frames](../images/random-psds.png)

### 3. Band onto sixth octaves

The octave-band PSD is a standing deliverable of a random report, not
an option. **Octave Bands** on the PSD's bar previews the banded
steps over the narrowband as the spacing is set (a sixth is the usual
answer) and the panel's own button bands conserving area — the octave
PSD is still a PSD, drawn as bars because its `bandwidth` says so.

![The sixth-octave view of the same PSDs](../images/random-octave.png)

### 4. Multiple coherence

How much of each response the drives account for — **Multiple
Coherence**, beside the PSDs in the averaging view, so it reads the
same frames. This is the figure that later separates
"the article exceeded" from "the cable did" — a channel the drives
explain (coherence near one) exceeded because the article did.

![The multiple coherence map: frequency across, channel down,
coherence as colour](../images/random-coherence.png)

### 5. Compare against the specification

Select the control PSDs and the specification together. One channel
draws at a time — the drop-down reaches the rest — with the tolerance
zones shaded and every line that went outside the abort band striped
red or blue for which way it went. The table below the plot accounts
for every channel at once; the plot shows the ones you pick in it.

Two things to notice in the bar above the plot. The **Scaling** field
is the comparison's dB offset — detected from the data in whole
decibels, for runs captured below the 0 dB requirement; this run was
at full level, so it reads 0. And the status bar notes the records
that had no specification to answer to: the drive force PSDs share
DOF names with control channels, and the comparison filters them out
rather than judging newtons against an acceleration band.

![One control channel against the specification: zones shaded,
exceedances striped, the Scaling field reading 0 dB, and the
compliance table accounting for all eight
channels](../images/random-comparison.png)

### 6. Geometry, photos, report

Drag the article's geometry in and declare its units; drop the setup
photographs beside it; link them into the group. Then the project's
row's *Generate Report* builds the random report — the measured data, the
specification comparison read three ways (spectra, RMS error bars,
lines-out bars), the coherence beside the compliance so the reader
can ask *why* a channel exceeded. *Export → HTML…* writes one
self-contained file.

![The exported report: control histories with the averaged frames
shaded, the specification with its tolerance
zones](../images/random-report.png)

## The same run, headless

And the script never has to be reconstructed by hand: the **console tab** along the bottom of the window writes it live as you click — every act of the session as the line that replays it. Expand it, copy the stretch you want, and it runs as-is.


The whole thing is one call when the defaults are right:

```python
import visualdynamics

visualdynamics.random_vibration_report('random.nc4', 'report.html')
```

…and the same call written out when the project should live on. It
is condensed from
[`examples/random_workflow.py`](https://github.com/visualdynamics/visualdynamics/blob/main/examples/random_workflow.py),
which the test suite executes; the step numbers match the walkthrough.

```python
import visualdynamics
from visualdynamics.plot import plot_bars, plot_coherence_map, plot_comparison

# 1-4. import and work up: PSDs over the run's own frames, the
# sixth-octave banding, the coherence — one verb does the sequence
project = visualdynamics.random_vibration_run('random.nc4')
assert project.project_type == 'Random Vibration'

history = project.time_history
spec = project.specification
psds, octave = project.psds        # the specification is not among them

# 5. the comparison, drawn headless
plot_comparison(psds, spec, path='control.png', show=False)
plot_comparison(octave, spec, path='control_octave.png', show=False)
plot_bars(psds, spec, 'error', path='error.png', show=False)
plot_bars(psds, spec, 'lines', path='lines.png', show=False)
plot_coherence_map(project.coherence, path='coherence.png', show=False)

# 6. geometry, photos, report
project.add('Geometry', visualdynamics.import_file('geometry.npz'))
project.geometry.define_units('m')
project.link(*project.names)
project.set_basis(*project.names)
project.generate_report('random')
project.export_report(project.report, 'random_report.html')
project.save('random.vdyn')
```

## Where judgement lives

- **Whether an exceedance is the article or the instrumentation.**
  The multiple coherence map answers it: a channel whose response the
  drives explain exceeded because the article did; a channel low on
  coherence may be a mounting or a cable. The report puts the map
  beside the compliance bars for exactly this reading.
- **Whether the comparison scale is right.** A run captured below the
  requirement is compared scaled up to the 0 dB specification —
  standard practice — and the scale is detected only when the data
  supports it: every channel at or above a corroborated offset. Type
  a number in the Scaling field to hold it (0 holds it unscaled),
  clear the field to detect again. The data itself is never changed.
- **What the run was allowed to do.** The warning and abort lines
  come from the specification; whether a brush against the warning
  line matters is the test engineer's sentence to write in the
  conclusions.
