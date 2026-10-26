# The shock workflow

A shock test specifies a response spectrum. The article must see an
event whose SRS falls inside a tolerance band around the required
curve, and the waveform that produces it is up to the shock machine.
The report therefore leads with the transients, since how long each
event lasted and how it rang down is only visible in the trace, and
answers the spectral question after.

The walkthrough uses the demonstration plate's shock series, four
half-sine events stepped up to level and recorded on six
accelerometers and a drive channel, with a screenshot at every step.
The recording ships with the repository (`testdata/plate/shock.nc4`).

| Step | In the app | In a script |
|---|---|---|
| Import the recording | drag the `.nc4` onto the window | `project.import_file(path)` |
| Say what it is for | right-click the root → Set Project Type → Shock | `project.project_type = 'Shock'` |
| The events | Detect in the Shocks view marks each event | `project.detect_shocks('Time History')` |
| Filter and integrate (optional) | Apply Filter in the Filter view; Integrate on the bar | `project.filter_data(name)`, `project.integrate(name)` |
| Compute the SRS | Compute SRS in the Shocks view | `project.compute_srs('Time History')` |
| The required SRS | import it, with its tolerance | `ShockSpecification(...)` from `visualdynamics.core.data` |
| Compare | select SRS and specification together | `plot_series([srs, spec])`, `plot_bars(srs, spec, 'srs')` from `visualdynamics.plot` |
| Judge the tolerance | Points Rule on the bar, with SRS and specification selected | `srs_tolerance(spec, srs)` from `visualdynamics.core.compliance`, `plot_srs_tolerance(srs, spec)` |
| Geometry and photos | drag them in, link them | `project.add(...)`, `project.link(...)` |
| Report | Generate Report on the bar, with the project row selected | `project.generate_report('shock')` |
| Or all of it at once | Automatic on the tree bar, once the project has its type | `project.work_up()` |
| Export | Export → HTML… | `project.export_report(name, path)` |
| Save | File → Save Project As… | `project.save('shock.vdyn')` |

## In the app, step by step

### 1. Import the recording, and say what it was

Drag the recording onto the window. At the file level a shock
recording cannot be told from a transient one (both are one stream of
events), so the run is imported as *Transient* and you declare it:
right-click the project root and choose **Set Project Type → Shock**.
The tree adds the slots a shock report draws from. There is no
coherence slot, because a shock is a single event and coherence
describes a stationary average.

![Where to click: right-click the project row (1), then choose Set
Project Type and Shock from the menu](../images/shock-type-click.png)

Select the time history and choose **Shocks** on its bar. Press
**Detect** to find the events; the view does not detect them until
you ask, and **Compute SRS** detects them for you if you go straight
to it. A record that is already read as frames (a controller's save
of a transient run, whose playings are its frames) keeps its own. The
time view shades each event on the trace, and **Add** and **Remove**
in the view add a window the detector missed or drop one.

Detected windows share one length by default, the longest any event
needs from its peak to a settled ringdown, so every event is analyzed
over the same window. Dragging a window's body moves that one event.
Dragging an edge resizes it, and while **Same length for all** is
ticked, resizing one window resizes the whole series, stopping before
any window would overlap its neighbor. Clear the box to resize windows
one at a time.

![Where to click: select the time history (1), choose Shocks on the bar
(2), and press Detect (3)](../images/shock-detect-click.png)

![The imported series: four events on one recording, each shaded
where it was found, the quiet between them
kept](../images/shock-import.png)

### 2. Filter and integrate (optional)

The shock project has optional slots for a filtered record and for
velocity and displacement integrated from it. Choose **Filter** on the
time history's bar, set the filter, and press **Apply Filter**; the
event windows are copied onto the filtered record. **Integrate** on
the bar then gives velocity, and integrating that gives displacement.
When a filtered record is present, compute the SRS from it, so the
spectra, the motion and the scalogram in the report all describe the
same signal. **Automatic** on the tree bar works in this order: it
filters the record, computes the SRS from the filtered record, and
integrates twice.

![Where to click: select the time history (1), choose Filter on the bar
(2), and press Apply Filter (3)](../images/shock-filter-click.png)

![Where to click: select the filtered record (1), then Integrate on the
bar (2)](../images/shock-integrate-click.png)

The Integrate act removes a drift below 2 Hz, which suits a record's
motion and not a stroke estimate: a compensated pulse's displacement
is content at a few hertz. To see whether a pulse fits on the shaker,
choose **Stroke** on the bar instead. It integrates the record raw and
reads the peak velocity, the travel with the shaker body counted and
the force against the shaker's shock ratings; [Will a pulse fit on the
shaker](../stroke.md) says how to read it.

![Where to click: select the time history (1), then Stroke on the bar
(2)](../images/shock-stroke-click.png)

### 3. Compute the shock response spectra

Before computing, read the velocity check under the SRS settings in
the Shocks view: every acceleration channel integrated to velocity with
nothing removed, over every window. A record whose velocity does not
come back to rest carries a zero shift or an offset, and its spectrum
is false at the low end; [Checking a shock record before its
spectrum](../shock-check.md) says how to read it.

Press Compute SRS in the Shocks view. It computes one SRS per
channel per event, at the **Q** (10 by default, 5 % damping), **Peak**
(maximax, positive or negative) and **Per octave** spacing (12 by
default) set in the view. A shock test is judged on the worst event,
so a recording holding four events gives four spectra rather than an
average. For one spectrum over the whole record, set a single window
over the whole record. The plot shows every event at one channel, so a
stepped-up series reads as a family of curves rising to level, and the
drop-down on the bar reaches the other channels.

![Where to click: with the time history selected (1), press Compute SRS
in the Shocks view (2)](../images/shock-srs-click.png)

![The computed SRS: all four events at the corner channel, the
channel drop-down reaching the others](../images/shock-srs.png)

A series of measured shocks can also be the start of a requirement:
**Field Limit** on the SRS's bar takes the envelope or a tolerance
limit over the events and writes the straight-line envelope a shock
specification is written from, at the spectra's own Q (see [Deriving a
specification from field data](../field-limit.md)).

### 4. The requirement, and the comparison

A shock specification is an SRS with a tolerance band around it, the
required curve with `abort_upper` and `abort_lower`. Import it, or
write it once in a script and save it. Select the measured SRS and the
specification together and the band is shaded around the spectra,
still every event at one control channel. **SRS Error** on the bar
shows the whole run as bars, one for every event at every control
channel: the RMS deviation from the requirement in decibels, signed,
shaded blue past −3 dB where the shock under-tested and red past
+3 dB where it over-tested. The same 3 dB is used by the random
workflow's RMS error bars.

![Where to click: select the SRS (1) and the specification (2) together;
SRS Error on the bar (3) shows the deviation
bars](../images/shock-error-click.png)

![The measured spectra against the required one: the tolerance band
shaded, the four events' deviations tabled, the stepped-up series
closing on the specification event by
event](../images/shock-spec.png)

The bars are an RMS summary and not the tolerance check, which counts
the points of each spectrum inside the band and applies the test
document's rule to the count. A spectrum with a narrow, deep dip can
read well inside 3 dB on the bars and still fail it. That check is
**Points Rule** on the bar, with its band and rule in the panel
beside the plot; in a script it is `srs_tolerance` and its picture
`plot_srs_tolerance`. [Judging an SRS against its
tolerance](../srs-tolerance.md) describes it.

![Where to click: with the SRS (1) and the specification (2) selected,
choose Points Rule on the bar (3)](../images/shock-points-click.png)

A classical pulse (a half-sine, sawtooth or trapezoid) is judged in
time rather than by its spectrum: **Pulse Check** on the time history's
bar draws the record on the nominal pulse's tolerance envelope and
reads its velocity change; [Checking a classical
pulse](../pulse-check.md) says how.

![Where to click: select the time history (1), then Pulse Check on the
bar (2)](../images/shock-pulse-click.png)

### 5. Geometry, photos, report

Drag the article's geometry in, declare its units, add the setup
photographs, and link the group. Then select the project row and press
**Generate Report**. The shock report starts with the front matter and
the measured time histories, with the event windows shaded and
numbered on them. How long each event lasted, how many there were, and
whether the article was still ringing when the next event arrived are
visible in the trace and not in the spectrum. When the project holds a
filtered record or integrated motion, a section on the motion follows.
Next is the scalogram of the filtered record (or of the measured
record when there is no filtered one), which shows where in frequency
each event's energy was and when, and is the way to find the source of
an unexpected SRS peak. Then come the specification with its band, the
measured SRS drawn over it as selecting both shows in the app, the
signed deviation bars, and the conclusions. There is no PSD in a shock
report: the level of a density computed from a transient recording
depends on how much quiet time was recorded around the events, and the
scalogram answers the frequency question with the time axis kept.

![Where to click: select the project row (1), then Generate Report on
the bar (2)](../images/shock-report-click.png)

![The exported report: the measured transients first, the
specification and its band after](../images/shock-report.png)

## The same series, headless

The **Console** tab along the bottom of the window shows the session
as Python while you work: every act as the line that replays it.
Expand it, copy the lines you want, and they run as they are.

```python
import numpy as np

import visualdynamics
from visualdynamics.core.data import ShockSpecification

project = visualdynamics.Project('Plate Shock Series')

# 1. import, and declare what the file cannot say
project.import_file('shock.nc4')
project.project_type = 'Shock'

# 2. one SRS per channel per event; the events are detected here
# because nothing else in the record says where they are
project.compute_srs(project.time_history)

# 3. the requirement: breakpoints with a tolerance band around them
srs = project.srs
frequencies = np.asarray(srs.abscissa, dtype=float)
level = ...                       # the required curve, from the test plan
project.add('Shock Specification', ShockSpecification(
    abscissa=frequencies, ordinate=level[None, :],
    response_dof=[srs.response_dof[0]],
    ordinate_dim='acceleration', ordinate_unit='m/s**2',
    q=srs.q, kind=srs.kind,
    abort_upper=level[None, :] * 10.0 ** 0.3,     # +6 dB: a factor of 2
    abort_lower=level[None, :] * 10.0 ** -0.3))   # -6 dB

# 4. geometry, photos, report
project.add('Geometry', visualdynamics.import_file('geometry.npz'))
project.geometry.define_units('m')
project.link(*project.names)
project.set_basis(*project.names)
project.generate_report('shock')
project.export_report(project.report, 'shock_report.html')
project.save('shock.vdyn')
```

## Where judgment lives

- Which events count. The demonstration series was stepped up to
  level, as a real one usually is. The report shows every event's
  spectrum, and which events were qualification and which were
  rehearsal is for the engineer to write.
- The tolerance itself. The demonstration specification uses ±6 dB
  (+6 dB and −3 dB is a common convention). A real specification
  carries its own band, and the shading in every comparison comes from
  the specification object, not from a setting. The rule about how
  many points must lie inside it is the test document's, and is given
  to `srs_tolerance` as its arguments; the band it counts against is
  the specification's own unless one is given.
- Q, and what the spectrum means. An SRS is computed at one
  amplification: a spectrum at Q = 10 says nothing about what the same
  shock does to a Q = 50 article. The Q is stored on the spectrum and
  on the specification. Compute the SRS at the Q the specification
  names, because the deviation is computed from the two curves as they
  are, with no adjustment for a difference in Q.
- The window length. Once a window is long enough to hold the peak and
  its ringdown, the SRS changes very little with more length, which is
  why one shared length is a safe default. The judgment is whether
  each window reached that length: a ringdown cut short reads low at
  the article's own frequencies, and the shaded windows on the trace
  are where to check.
