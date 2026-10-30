# Guide

How the parts of Visual Dynamics fit together, and why they work the
way they do.

**This is an alpha release.** It has been checked against known cases,
but not yet against a broad set of tests, so verify the numbers it
produces by independent means before anything depends on them. The
application says so when it opens, and asks you to acknowledge it the
first time each new version is launched. Please report anything that looks wrong to
contact@visualdynamics.org. Projects are saved as `.vdyn` files, and
Engineering Sciences Common Data Format (`.h5`) files are imported and
exported for exchanging data with other tools. While Visual Dynamics
is an alpha, the `.vdyn` format may still change; the commitment that
a later version opens every earlier file starts with the first release
that is not an alpha.

## Start here

- [Projects, links, and the Basis](projects.md): the structure
  everything else reads
- [The project tree](project-tree.md): what a selection shows, whether
  it is one object, records picked in one, or several objects together
- [The objects](objects.md): every kind of object, what it holds, and
  what can be done with it
- [Units](units.md): SI inside, display units where values are shown
  or written out
- [The channel table](channel-table.md): the one object that is a
  spreadsheet, and its rules
- [Planning a test's instrumentation](plan-instrumentation.md): the
  accelerometers for a modal test chosen from the finite element modes,
  on the surfaces a person can reach
- [Importing a time history from delimited text](text-import.md): a
  CSV or a recorder's text export, its layout found and checked

## Plotting and figures

- [Plotting](plotting.md): every plot the app draws, from code
- [Composing a figure from a script](plot-composition.md): several
  readings, marks and panels on one figure, written to a file
- [A figure in a page of your own](figures-in-a-page.md): one figure
  exported as a data file and the renderer a page mounts, trimmed by
  channel and by peak

## Workflows

- [The modal workflow](workflows/modal-workflow.md): the app and a
  script, step by step
- [The random vibration workflow](workflows/random-workflow.md): from
  a controller run to a compliance report
- [The transient workflow](workflows/transient-workflow.md): a target
  waveform compared with what the test article did
- [The shock workflow](workflows/shock-workflow.md): events, the SRS,
  and its tolerance band
- [The sine sweep workflow](workflows/sine-workflow.md): tones tracked
  through a sweep and compared with the requirement
- [The random and sine workflow](workflows/mixed-workflow.md): a sweep
  on top of a random, both judged from one recording
- [The system identification workflow](workflows/sysid-workflow.md):
  measuring a plant, and how well it was measured
- [The finite element workflow: the BARC](workflows/fem-workflow.md):
  a model built from plates, and again from solid bricks, then solved
- [The solid workflow: the Frame-Wing](workflows/fem-solids.md):
  solid shapes added, patterned, subtracted and joined, then meshed
  into bricks and solved
- [Three shapes that are not extrusions](workflows/mesh-shapes.md):
  a cross-drilled shaft, a wye fitting and a V-block meshed into
  bricks fitted to the shape and into tetrahedra, with the numbers

## Modal

- [How the mode fitting works](modal-fitting.md): the algorithm, and
  how it differs from the classical batch methods
- [Reading Q from a peak's half-power width](half-power.md): the
  0.707 line, its crossings, and how many lines a peak needs
- [Comparing signatures before and after a test](signature-comparison.md):
  each mode's frequency and amplitude shift, judged against limits
- [Modal responses from forces](modal-responses.md): how each mode of
  a shape set answers the forces in a time history, appended to it

## Reading a spectrum

- [How a PSD means its area](psd-reading.md): bandwidth, integration,
  and what the numbers mean
- [What an average is worth](averaging-confidence.md): the degrees of
  freedom of an averaged PSD, and the confidence band they give
- [How a comparison is judged](compliance.md): the one cell rule,
  every pairing of PSD and specification, and the edge cases
- [Lines outside a band by chance](comparison-chance.md): the lines a
  comparison counts outside each band, beside how many the scatter of
  an average puts there, and lines combined for judging
- [Writing a random specification by slope](breakpoint-slopes.md): a
  level, a slope in dB per octave and the next frequency, each
  segment's area and the running rms
- [Judging a multi-point control run](multi-point.md): the composite
  of several control channels against its band, and each channel
  against its own
- [Checking a spectrum against its record](rms-check.md): the rms from
  the area under an ASD beside the rms of its time history, and the
  scaling errors the gap names
- [Deriving a specification from field data](field-limit.md): the
  envelope or a tolerance limit over a set of measured spectra, what it
  rests on, and the straight-line envelope a specification is written
  from
- [Fatigue damage, extreme response and time compression](fatigue.md):
  a record and a candidate test read through the same resonators, and
  the trade of time for level with its cap
- [Reading a tone off an ASD](tone-readout.md): why a spike's height
  is not the tone, and the tone in g peak from the lines it leaks into
- [Reading a tone's amplitude from a record](amplitude-spectrum.md):
  the amplitude spectrum in g peak, the window's correction stated, and
  the flat top for a tone between lines
- [The two transmissibilities, and what H1 is
  worth](transmissibility.md): the ASD ratio beside H1, and the band
  H1's coherence and averages give
- [A mixed-mode test as one specification](mixed-mode.md): the floor,
  the tones and the narrowbands, and a data sheet that judges each by
  its own rule
- [Judging a cross-spectral matrix against its
  specification](matrix-comparison.md): the levels on the diagonal and
  each cross term's coherence and phase, as a grid
- [Checking the matrices of a multi-input test](mimo-matrix-checks.md):
  the plant's singular values and condition number, and whether a
  cross-spectral specification is realizable, with its repair
- [Predicting the notch before the test](limit-prediction.md): where
  each limit channel would exceed its profile at full level, and the
  notch it would cut in the control
- [Predicting the drives before the test](drive-prediction.md): the
  drives a specification asks for through an FRF, one drive or several,
  against the amplifier and shaker limits
- [Which drive explains a response](partial-coherence.md): partial,
  reference and virtual coherence of a test with several drives
- [The transformation to a point, and where the rigid body
  holds](transformation.md): the matrix from channels to six motions,
  its conditioning, and the residual it leaves against frequency

## Reading a record

- [Reading a record's time statistics](time-statistics.md): the
  moments, the rail, the tails against the Gaussian, the expected
  largest peak and the kurtosis frame by frame
- [Checking each channel against its range](overload.md): the samples
  on the rail, the headroom over the peak and the peak at the next
  level, from the channel table's ranges
- [Taking a record to a lower sample rate](resample.md): the
  anti-alias filter first, the span and the dropped lines, and where
  content above the new Nyquist frequency folds
- [The wavelet reading](wavelet.md): a time history as a scalogram

## Sine

- [How a tracking filter reads a sweep](sine-tracking.md): the
  controller's band and detector, and why its level differs
- [Checking a sweep rate against the modes it passes](sweep-rate.md):
  the time and cycles in each half-power band, and what a fast sweep
  does to a peak and its Q
- [Reading a dwell against time](dwell.md): a held tone's build-up,
  steady level and ring-down, in cycles
- [Checking a sine record for distortion](sine-distortion.md): the
  record less the tracked tones against the tones, along the sweep,
  against a limit
- [Checking a sine tone's frequency](sine-frequency.md): the tone's own
  frequency on the record against the specification's, along the
  sweep, against a band
- [Writing a sine profile in segments](sine-profile.md): constant
  displacement, velocity and acceleration, every crossover solved and
  checked, and the tone a controller runs

## Shock and transient

- [Judging an SRS against its tolerance](srs-tolerance.md): the band,
  the count of points inside it, and the rule applied to the count
- [Checking a shock record before its spectrum](shock-check.md): the
  velocity integrated raw, and the positive and negative spectra
- [Checking a classical pulse](pulse-check.md): a half-sine, sawtooth
  or trapezoid against its envelope and its velocity change
- [Will a pulse fit on the shaker](stroke.md): a pulse integrated raw
  against the shaker's shock stroke, velocity and force
- [Synthesizing a shock to its spectrum](shock-synthesis.md): wavelets
  iterated until every point is inside the band, then read against the
  shaker's shock stroke, velocity and force
- [Checking a replication reference](replication-fit.md): whether a
  shaker can play a record, and what the high-pass that makes it
  playable costs
- [Judging a replication by segment](replication-segments.md): each
  stretch of a replicated record held to the rule for its kind
- [Generating reference signals](generating-signals.md): random,
  burst random, periodic random, an impulse or a pulse, and sine, made
  at the reference DOFs from a specification or a channel table

## Running a test

- [The run log file](run-log.md): what happened on the floor, entry by
  entry, corrected without editing, each test's entries chained by
  hash, kept by one writer with a second copy, and exported as text,
  CSV and the machine log

## Reports

- [Reports](reports.md): symbolic bindings, live text, and one file out
- [A report plot's caption](report-captions.md): the channel, run,
  resolution, window, DOF and requirement a spectrum's plot carries,
  read off its record
- [Report features](report-features.md): optional additions to a
  report, turned on in a panel or by preset: the skeleton sections,
  the requirement and its results, the data index, the findings
  register, calibration in date, identity and issue, responses against
  prediction, record captions and the plot conventions

For signatures, see the API reference in the site navigation. For the
layers of the package and the conventions behind them, see
[Architecture and conventions](../architecture.md).
