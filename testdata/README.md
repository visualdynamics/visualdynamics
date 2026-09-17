# Test data

One folder per source model, files named by content. Every model here
is built by the package itself — `visualdynamics.fem` and the two
demonstration articles in `visualdynamics.demo` — because a
demonstration of the toolset should be made by the toolset. The sdynpy
demo models (`beam_plate`, `beam_airplane`) that seeded the original
fixtures are gone, generators and all.

`plate/` carries every object type visualdynamics handles, so it is a
complete test on its own. It comes from `visualdynamics.demo.plate`:
the 12 in x 12 in x 0.5 in 6061-T6 free plate, the article whose
frequencies check against the classical tables. The drone is the
second model and different in kind: large, and living outside the
repository — see below.

The plate set is laid out like a **modal test**: four fixed reference
DOFs where the article would be struck or shaken, responses measured
over the survey grid. The FRFs are every response against each of the
four drive points, and the time data is one hammer hit — the impulse
response at every measured DOF, obtained by inverse-transforming those
FRFs, so the time data, the FRFs and the mode shapes all describe the
same structure.

```
plate/            visualdynamics.demo.plate — 12x12x0.5 in 6061, free
  geometry.npz        sdynpy-format geometry (no units in the file)
  geometry.unv        the same geometry as a universal file (no 164)
  geometry.exo        the same geometry as exodus, 144 SHELL4 quads
  test_geometry.npz   the modal run's nodes with survey tracelines
  shapes.npy          22 mass-normalized modes to 5 kHz over 1014 DOFs
  frfs.npz            25 responses x 4 references (m/s^2 per N), 2 Hz
                      lines to 2.5 kHz
  frfs.unv            the same FRFs as UFF dataset 58 with a 164
                      declaring SI
  time.npz            impulse response at all 25 responses, corner drive
  psd.npz             25 autospectral densities
  spectrum.npz        linear spectra
  channel_table.vdyn  29 channels: 25 accelerometers and 4 load cells
  shock.nc4           four shocks in one recording, synthesized in
                      Rattlesnake's format (see generate_plate.py)
  modal.nc4           a real Rattlesnake modal survey: 11 channels
                      against 2 shakers, burst random, 20 averages
  modal_spectra.nc4   that run's FRFs and multiple coherence
  random.nc4          a real Rattlesnake MIMO random test: 8 control
                      channels, 4 shakers, controlled to a shaped
                      specification
  random_spectra.nc4  that run's spectra, saved while at level: FRFs,
                      coherence, the 8x8 response and 4x4 drive
                      cross-spectral matrices, and the specification
```

References 101Z+, 110Z+, 1304Z+, 1310Z+ — one corner and three edge
stations at the quarter points, which are the *generic* positions on a
square plate. Not the grid's interior: every interior survey station
lies on a center line or a diagonal — the plate's own symmetry — and a
reference there is blind to whichever mode family nulls that line. The
first reference set had three of four references unable to see the
647 Hz mode at all, and its fitted shape could not resynthesize the
measurement. Responses are Z only: the plate lies in the XY plane and
its measured in-plane response is numerical noise, not data — except
one deliberate in-plane channel in the modal run, because a real
survey always has a sensor pointed somewhere nothing moves.

The channel layout is `plate_layout.py`, imported by both generators,
so the survey the display geometry describes and the survey the
controller runs cannot drift apart — `tests/test_correlate.py` holds
the identity.

## The drone set lives outside the repository

`generate_drone.py` — in the `visualdynamics-generators` repository,
because it drives the real controller — builds the demonstration
quadcopter and everything that comes from it, and writes the lot to
`stressdata/drone/`, which is gitignored in its entirety. Nothing of it
is committed. `generate_drone_projects.py` here assembles that data
into four ready-to-open projects.

That is a size decision, not a quality one. The runs are at 8192 Hz
over 28 channels, which is what makes them representative of real
controller output, and a stream at that rate costs about 1.8 MB a
second — the set comes to 600 MB. Git keeps blobs for ever, so the
rate stays and the data does not.

The plate's committed runs live with the same tension resolved the
other way: its physics needs 4096 Hz (modes at 439-1142 Hz in band),
sixteen times the old airplane's rate, so its four run files weigh
about 45 MB — kept, because the suite needs real controller files, and
timed tightly because of what they weigh.

```
stressdata/drone/
  fem_geometry.vdyn     the whole shell, 1422 nodes, quads and triangles
  fem_shapes.vdyn       942 modes to 10 kHz at 2% damping — the truth set
  test_geometry.vdyn    the 14 nodes a survey instruments, threaded with
                        tracelines: four arms, the body, the payload, two legs
  plant.vdyn            24 responses x 4 drives, accelerance, DC to 4096 Hz
  specification.vdyn    the random test's control target
  transient_target.vdyn the waveform the transient test replicates
  modal.nc4             burst random modal survey, 20 averages
  modal_spectra.nc4     that run's FRFs and multiple coherence
  random.nc4            MIMO random vibration, 4 shakers, 16 control channels
  transient.nc4         the same control set replicating a waveform
```

**Two geometries, on purpose.** The FEM half is the complete model and
every mode it has; the test half is the handful of nodes a real survey
would instrument. Correlating one against the other is the point of
having both.

**The inputs are all ours.** The plants, the specifications' PSDs and
the transient waveform are written by visualdynamics and exported to
sdynpy's format through the ordinary exporter — which is what the
import/export symmetry rule buys. Nothing here hand-rolls an sdynpy
object; sdynpy is only the container the controller happens to read,
and `rattlesnake` is imported to run the controller and nothing else.

## The Rattlesnake files are different in kind

Everything else here is synthesized by visualdynamics itself. The
modal and random `.nc4` files came out of the Rattlesnake controller
actually running — virtual hardware convolving the plate's FRFs in
real time, a control law closing the loop frame by frame, spectra
averaging. They took minutes to make rather than seconds, on purpose.

They are the only files here that **carry their own units**. A
Rattlesnake channel table names the engineering unit of every channel,
so they import fully defined and the Define Units step never appears —
and each holds two quantities at once, accelerations and forces side
by side, which is the case per-record units exist for. Everything else
imports unit-less by design. They are also the only check that the
`.nc4` reader copes with a real file rather than an idea of one.

**A run saves twice, and the two files share nothing.** Streaming
writes the time histories; a separate save writes what the environment
computed from them — FRFs and coherence, and for random the cross
spectra too. Neither file contains the other's contents, so both are
kept.

**Two of the random run's drives are control channels.** So a drive
point's force record and its acceleration record share a DOF name —
the case that flushed out two DOF-name-collision bugs the airplane's
layout (drives outside the control set) could never reach. The run
drives four shakers against its eight control channels: two were
tried first and genuinely could not hold the plate, which is worth
remembering as the thing the compliance report looks like when a
test is under-actuated.

**The modal test is burst random, which is three settings and not
one.** The excitation drives for half of each frame and holds zero for
the rest, so the response rings down inside the frame it started in.
That is what makes the signal self-windowing, so the FRF window is
`rectangle` — a Hann window would discard the start of the burst and
taper a decay already at zero — and the overlap is 0%, because one
frame is one burst cycle.

The controller-driving machinery lives in `rattlesnake_profiles.py`,
in the `visualdynamics-generators` repository (`run_profile`,
`run_transient_profile`, the completeness gates), and it carries the
scars of everything the controller's spectral side did during
development:

- **`SAVE_CONTROL_DATA` does not save the control data.** Read it: it
  writes `sysid_data` — the system identification measured once before
  the profile starts, at its own fixed excitation. It is the same
  bytes however long the test runs and whatever the test level is.
  Pointed at a two-level run it produced two identical files sitting
  82 dB under the specification, while the streamed time data from the
  same run showed the loop converged and the level change plainly. The
  committed fixture had this in it for a while and nothing caught it,
  because every test asked about shapes and units and none asked
  whether the numbers were the measurement.
- So **neither** environment's spectra come from a save command. Both
  push what they have computed at the interface they would otherwise
  draw it on — modal a `SPECTRAL_UPDATE`, random a `CONTROL_UPDATE`
  carrying the live `last_response_cpsd` and `last_drive_cpsd` — and
  the driver takes the latest off that queue and writes the file
  itself, in the layout the save command would have used. Draining
  that queue is not optional anyway: left alone it grows for the whole
  run.
- A modal environment counts every enabled channel as a **response**
  and never excludes its references, so the saved FRF matrix carries
  each drive against the drives. Those rows are identity and
  cross-talk by construction — 1 against itself, zero against the
  other — and the importer drops them.
- A modal spectral save also carries **time data**, and it is not a
  recording: `modal_ui.py` appends one frame per *accepted* average,
  so the samples are separate captures laid end to end. With
  `Accept All` and no overlap that is `num_averages *
  samples_per_frame` exactly. visualdynamics splits it into one record
  per capture rather than drawing twenty restarting bursts as one
  signal.
- **Nothing stops a headless modal environment when its averages are
  done.** `num_averages` appears nowhere in `modal_environment.py`
  except its own metadata; the *interface* is what counts accepted
  frames and calls `stop_environment`. Left to itself the generator
  bursts on until the profile pulls the hardware. The driver does what
  the interface does.
- **A starved run is short, not empty, and `_has_data` cannot see
  that.** On a loaded machine the driver's loop stalls and a run
  captures 1 average of 20 — a file that is structurally perfect,
  non-empty, and worthless. It overwrote a good set once.
  `_run_is_complete` is the gate: a modal save must hold every average
  it was asked for, and a streamed file must cover at least half its
  window.
- **Draining that queue is expensive in proportion to channel count.**
  An unbounded drain takes longer than the interval between drains and
  the driver's clock stops — 929 seconds in one iteration. The drain
  is bounded by wall clock as well as by count.
- Random's save command also opens its target in netCDF append mode
  and leaves nothing in it but its own environment group. Never point
  it — or anything — at the streamed file: the streaming process still
  holds it open until shutdown, and a second writer destroys the time
  data outright.

A channel table has no foreign format to come from — only a
rattlesnake run carries one — so the plate's is written with
visualdynamics itself, in the native `.vdyn` format.

Everything is written in SI. Only `plate/frfs.unv` declares its units
(via UFF dataset 164); every other non-controller file carries none,
which is deliberate — it exercises the unit-less import path and the
Define Units step.

The `.npz` files are saved compressed — the arrays are smooth enough
to shrink by nearly half, and they live in the repository.

## Regenerating

Everything but the controller runs regenerates in the project's own
venv, because every file goes out through visualdynamics's exporters:

```
./.venv/bin/python testdata/generate_plate.py     # the plate set
```

The controller runs live in the `visualdynamics-generators`
repository — they import rattlesnake, which must not happen in this
tree — and need a python that has rattlesnake and sdynpy. Run them
from this checkout, which is where they find `visualdynamics` and
`plate_layout`:

```
python3 ~/visualdynamics-generators/generate_plate_runs.py   # plate modal + random
python3 ~/visualdynamics-generators/generate_drone.py        # the drone set
```

Both pace to real time and take minutes; scratch goes to
`testdata/_rattlesnake_plate/` and `_rattlesnake_drone/`, neither
kept. Redirect their output to a file rather than piping it: the
controller's multiprocessing helpers outlive the main process holding
the pipe open, and a reader never sees EOF.
