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

- [Projects, links, and the Basis](projects.md): the structure
  everything else reads
- [The project tree](project-tree.md): what a selection shows, whether
  it is one object, records picked in one, or several objects together
- [The objects](objects.md): every kind of object, what it holds, and
  what can be done with it
- [Units](units.md): SI inside, display units where values are shown
  or written out
- [Plotting](plotting.md): every plot the app draws, from code
- [The modal workflow](workflows/modal-workflow.md): the app and a
  script, step by step
- [How the mode fitting works](modal-fitting.md): the algorithm, and
  how it differs from the classical batch methods
- [The random vibration workflow](workflows/random-workflow.md): from
  a controller run to a compliance report
- [Reading a record's time statistics](time-statistics.md): the
  moments, the rail, the tails against the Gaussian, the expected
  largest peak and the kurtosis frame by frame
- [The transient workflow](workflows/transient-workflow.md): a target
  waveform compared with what the test article did
- [Judging a replication by segment](replication-segments.md): each
  stretch of a replicated record held to the rule for its kind
- [The shock workflow](workflows/shock-workflow.md): events, the SRS,
  and its tolerance band
- [Judging an SRS against its tolerance](srs-tolerance.md): the band,
  the count of points inside it, and the rule applied to the count
- [Checking a shock record before its spectrum](shock-check.md): the
  velocity integrated raw, and the positive and negative spectra
- [Checking a replication reference](replication-fit.md): whether a
  shaker can play a record, and what the high-pass that makes it
  playable costs
- [Will a pulse fit on the shaker](stroke.md): a pulse integrated raw
  against the shaker's shock stroke, velocity and force
- [The sine sweep workflow](workflows/sine-workflow.md): tones tracked
  through a sweep and compared with the requirement
- [The random and sine workflow](workflows/mixed-workflow.md): a sweep
  on top of a random, both judged from one recording
- [How a tracking filter reads a sweep](sine-tracking.md): the
  controller's band and detector, and why its level differs
- [Reading Q from a peak's half-power width](half-power.md): the
  0.707 line, its crossings, and how many lines a peak needs
- [Reading a tone off an ASD](tone-readout.md): why a spike's height
  is not the tone, and the tone in g peak from the lines it leaks into
- [Checking a sweep rate against the modes it passes](sweep-rate.md):
  the time and cycles in each half-power band, and what a fast sweep
  does to a peak and its Q
- [Comparing signatures before and after a test](signature-comparison.md):
  each mode's frequency and amplitude shift, judged against limits
- [Reading a dwell against time](dwell.md): a held tone's build-up,
  steady level and ring-down, in cycles
- [A mixed-mode test as one specification](mixed-mode.md): the floor,
  the tones and the narrowbands, and a data sheet that judges each by
  its own rule
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
- [Planning a test's instrumentation](plan-instrumentation.md): the
  accelerometers for a modal test chosen from the finite element modes,
  on the surfaces a person can reach
- [The wavelet reading](wavelet.md): a time history as a scalogram
- [Checking a classical pulse](pulse-check.md): a half-sine, sawtooth
  or trapezoid against its envelope and its velocity change
- [The channel table](channel-table.md): the one object that is a
  spreadsheet, and its rules
- [How a PSD means its area](psd-reading.md): bandwidth, integration,
  and what the numbers mean
- [What an average is worth](averaging-confidence.md): the degrees of
  freedom of an averaged PSD, and the confidence band they give
- [How a comparison is judged](compliance.md): the one cell rule,
  every pairing of PSD and specification, and the edge cases
- [Reports](reports.md): symbolic bindings, live text, and one file out

For signatures, see the API reference in the site navigation. For the
layers of the package and the conventions behind them, see
[Architecture and conventions](../architecture.md).
