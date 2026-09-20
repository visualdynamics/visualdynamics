# Guide

How the pieces fit, and why they are the way they are.

**This is an alpha release.** It has been checked against known
cases, not yet against a broad set of tests; numbers it produces are
to be verified by independent means before anything depends on them.
The application says so when it opens, and asks for that to be
acknowledged every time. Anything that looks wrong is worth
reporting to contact@visualdynamics.org. The project file, `.vdyn`,
is provisional: the next alpha will save projects in the Engineering
Sciences Common Data Format (`.escdf`), a public standard, and will
still open `.vdyn` files; the first release that is not an alpha will
not. Open each `.vdyn` once in that alpha and save it again.

- [Projects, links, and the Basis](projects.md) — the structure
  everything else reads
- [The project tree](project-tree.md) — what a selection shows: one
  object, records picked in one, or several together
- [The objects](objects.md) — every kind of object, what it holds,
  and what can be done with it
- [Units](units.md) — SI inside, display units at the boundary
- [Plotting](plotting.md) — every plot the app draws, from code
- [The modal workflow](workflows/modal-workflow.md) — the app and a script,
  step for step
- [How the mode fitting works](modal-fitting.md) — the algorithm, and
  how it differs from the classical batch methods
- [The random vibration workflow](workflows/random-workflow.md) — a controller
  run in, a compliance report out
- [The transient workflow](workflows/transient-workflow.md) — a target waveform
  against what the article did
- [The shock workflow](workflows/shock-workflow.md) — events, the SRS, and its
  tolerance band
- [The sine sweep workflow](workflows/sine-workflow.md) — tones tracked
  through a sweep, against what was asked for
- [The system identification workflow](workflows/sysid-workflow.md) —
  a plant measured, and how well
- [The wavelet reading](wavelet.md) — a time history as a scalogram
- [The channel table](channel-table.md) — the one object that is a
  spreadsheet, and its rules
- [How a PSD means its area](psd-reading.md) — bandwidth, integration,
  and what the numbers mean
- [How a comparison is judged](compliance.md) — the one cell rule,
  every pairing of PSD and specification, and the edges
- [Reports](reports.md) — symbolic bindings, live text, one file out

For signatures see the API reference in the site navigation; for the
design decisions behind all of it, `PLAN.md` in the repository root
is the long form.
