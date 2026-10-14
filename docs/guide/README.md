# Guide

How the parts of Visual Dynamics fit together, and why they work the
way they do.

**This is an alpha release.** It has been checked against known cases,
but not yet against a broad set of tests, so verify the numbers it
produces by independent means before anything depends on them. The
application says so when it opens, and asks you to acknowledge it
every time. Please report anything that looks wrong to
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
- [The transient workflow](workflows/transient-workflow.md): a target
  waveform compared with what the test article did
- [The shock workflow](workflows/shock-workflow.md): events, the SRS,
  and its tolerance band
- [The sine sweep workflow](workflows/sine-workflow.md): tones tracked
  through a sweep and compared with the requirement
- [The random and sine workflow](workflows/mixed-workflow.md): a sweep
  on top of a random, both judged from one recording
- [How a tracking filter reads a sweep](sine-tracking.md): the
  controller's band and detector, and why its level differs
- [The system identification workflow](workflows/sysid-workflow.md):
  measuring a plant, and how well it was measured
- [The finite element workflow: the BARC](workflows/fem-workflow.md):
  a model built from plates, and again from solid bricks, then solved
- [The solid workflow: the Frame-Wing](workflows/fem-solids.md):
  solid shapes added, patterned, subtracted and joined, then meshed
  into bricks and solved
- [The wavelet reading](wavelet.md): a time history as a scalogram
- [The channel table](channel-table.md): the one object that is a
  spreadsheet, and its rules
- [How a PSD means its area](psd-reading.md): bandwidth, integration,
  and what the numbers mean
- [How a comparison is judged](compliance.md): the one cell rule,
  every pairing of PSD and specification, and the edge cases
- [Reports](reports.md): symbolic bindings, live text, and one file out

For signatures, see the API reference in the site navigation. For the
layers of the package and the conventions behind them, see
[Architecture and conventions](../architecture.md).
