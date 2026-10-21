# Visual Dynamics

[![tests](https://github.com/visualdynamics/visualdynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/visualdynamics/visualdynamics/actions/workflows/ci.yml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)

A units-aware structural dynamics analysis toolset. It handles
geometry, test data (time histories, spectra, FRFs, PSDs, SRS), mode
shapes, finite element models and reports, with unit handling built
into every object and a desktop application for working with them
visually: 3-D geometry, shape animation and interactive plots.

Three names, on purpose: Visual Dynamics is what a person reads,
`visualdynamics` is the package and the import, and `.vdyn` is the
project file.

**This is an alpha.** The first launch of each version opens with a
notice that has to be acknowledged: the software has been checked against known cases, not
yet against a broad set of tests, and its numbers are to be verified
by independent means before anything depends on them. Projects are
saved as `.vdyn` files, and Engineering Sciences Common Data Format
(`.h5`) files are imported and exported for exchanging data with other
tools. While Visual Dynamics is an alpha the `.vdyn` format may still
change. The promise that a later version opens every earlier file
starts with the first release that is not an alpha.

[Core principles](PRINCIPLES.md): thirteen principles guide the
design. Among them, data should be visible and easy to handle, the
interface shows only what applies, the scripting API can do everything
the interface can, every bug that gets through becomes a test, and
dependencies are kept at their latest versions rather than pinned.

[Documentation](https://visualdynamics.org/documentation/): getting
started, the guides and an API reference generated from the
docstrings. It is part of [visualdynamics.org](https://visualdynamics.org)
and is rebuilt with every release; its [source](docs/README.md) is in
this repository and reads well on GitHub. The application and its
downloads are on [visualdynamics.org](https://visualdynamics.org) too.
To build the documentation locally:

```bash
pip install -e '.[docs]' && properdocs serve  # localhost:8000
```

The desktop application and `visualdynamics.Project` are two front ends
over one workflow. They hold the same objects and call the same verbs,
so anything you can do with the mouse you can do in a script, and a
`.vdyn` file saved by either opens in the other.

```python
import visualdynamics

project = visualdynamics.Project('Beam Airplane Modal Test')
project.import_file('modal_run.nc4')
project.set_basis(*project.names)
project.fit_modes(project.basis.frf, bounds=(5, 120))
project.basis.frf.plot()                  # the app's own plot
project.export_report(project.generate_report('modal'), 'report.html')
```

## What it does

- Imports universal files, Exodus, Nastran, Femap, I-DEAS associated
  data files, Rattlesnake runs and specifications, STEP, IGES, STL and
  3MF CAD, MATLAB, ESCDF and sdynpy files. Where a file does not
  declare its units, the data is marked as having unknown units until
  you define them. The
  [interchange matrix](docs/export.md) lists what is read and written.
- Writes a random specification back out in the form Rattlesnake loads,
  and cuts a Rattlesnake run to a stretch of time and saves it as a new
  `.nc4` with every setting and channel kept.
- Exports a project to MATLAB in the display units, each unit field
  naming the unit its values are in.
- Shows data as grids, plots, CMIF and coherence maps, and on a 3-D
  stage, where two FRF objects can be paired channel by channel.
  Animates shapes and
  time data on the geometry and marks measurement DOFs with labeled,
  axis-colored arrows.
- Fits modal models to FRFs interactively, one mode at a time on the
  residual CMIF, with the MAC beside the mode table, and compares shape
  sets by cross-MAC with phase-aligned overlay animation.
- Builds and solves finite element models from planes of plate
  elements, from solids meshed into bricks (blocks, cylinders, standard
  sections, holes, booleans and patterns), or from an imported mesh.
- Computes averaged spectra and PSDs, shock response spectra, sine
  levels and signal-to-noise from time data.
- Keeps test photographs in the project.
- Generates a complete report for each of its seven project types
  (modal, random, shock, transient, sine sweep, random and sine, and
  system identification). The report is edited with the exported page
  itself as the live preview, in markdown text with live
  `{{Object.field}}` values and figure references that renumber
  themselves, and is exported as one self-contained HTML file that
  animates mode shapes in any browser.
- Exports mode shapes with their geometry to Exodus files that
  ParaView can animate, and saves whole projects as one `.vdyn` file.

## Repository layout

The package is `src/visualdynamics`. The `src/` directory keeps the
checkout from shadowing the installed package: a directory named
`visualdynamics` would be imported in its place by any Python started
in its parent directory, and it would bind to an empty namespace
package (an `AttributeError` on everything, not an `ImportError`).

Moving a checkout takes more than `mv`. The virtual environment records
the absolute path in every console script's shebang and in
`pyvenv.cfg`, and an editable install records it too, so after a move
run `pip install -e .` again in each environment.

## Install

The packaged application, one download per platform with nothing else
to install, is on the [downloads
page](https://visualdynamics.org/downloads). The library is on PyPI
as `visualdynamics` (`pip install visualdynamics`, Python 3.12 or
newer). To work on it, install from a clone:

```bash
pip install -e .
visualdynamics                     # the desktop app
```

```python
import visualdynamics
```

STEP and IGES import need the OpenCASCADE kernel, which `pip` keeps
optional so a plain install stays small: `pip install -e '.[step]'`.
Every packaged build includes it.

The packaged builds are a `.dmg` for Apple silicon and one for Intel, a
Windows installer and a Linux AppImage. The release workflow
(`.github/workflows/release.yml`) builds all four on GitHub's runners,
and signs and notarizes the macOS images. `packaging/README.md` has the
recipes, including how to build each one locally.

On a Mac, `python tools/make_app.py` writes
`~/Applications/Visual Dynamics.app`, a real bundle with an icon, for
Finder, Spotlight and the Dock. It is a launcher, not a frozen
application: it runs the checkout's own interpreter, so an edit to the
source takes effect on the next launch. It finds the checkout by
absolute path, so run it again after moving the repository.

## Contributing

Bug reports are very welcome, and I will try to fix them promptly. Say
what you did, what happened, and what you expected. A `.vdyn` project
file or a small script that reproduces the problem makes the fastest
bug report.

Pull requests are welcome from anyone: bug fixes, improvements and new
additions alike, with no invitation needed. A pull request that keeps
the project's rules will be merged; [CONTRIBUTING.md](CONTRIBUTING.md)
says what they are and where they are written. An issue first is never
required, though for a large addition a short one can save you time.
Questions are welcome at
[contact@visualdynamics.org](mailto:contact@visualdynamics.org).

Help that is not code is welcome too: advice, review, measurements, or
a bug report that turns out to be a real one. It is thanked on the
website's [acknowledgments](https://visualdynamics.org/acknowledgments.html)
page, as thanks rather than authorship.

## License

Copyright (c) 2026 Brandon Zwink. Visual Dynamics is released under
the [MIT License](LICENSE): you may install it, use it, read it, change
it, build it into your own work, open or closed, and pass it on, as
long as the copyright notice goes with it. The outputs are yours:
plots, reports, exported files, and numbers you put in a paper or hand
to a customer. Nothing here claims any interest in your data or in
what the software tells you about it.

Versions 0.1.0a1 through 0.1.0a14 were released under the GNU General
Public License, version 3 or later. The copyright holder grants them
under the MIT License as well, so every version so far can be used on
MIT terms. Contributions are accepted under the same MIT License; see
[CONTRIBUTING.md](CONTRIBUTING.md). Third-party licenses are listed in
[NOTICE.md](NOTICE.md).
