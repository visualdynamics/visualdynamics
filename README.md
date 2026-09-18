# Visual Dynamics — Structural Dynamics Visualization

[![tests](https://github.com/visualdynamics/visualdynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/visualdynamics/visualdynamics/actions/workflows/ci.yml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)

Units-aware structural dynamics analysis toolset. Geometry, test data (time
histories, spectra, FRFs, PSDs, SRS), mode shapes, and composite test containers —
with first-class unit handling and a highly visual desktop GUI (3D geometry,
shape animation, interactive plotting).

Three names, on purpose: **Visual Dynamics** is what a person reads,
`visualdynamics` is the package and the import, `.vdyn` is the project
file.

**This is an alpha.** Every launch opens with a disclaimer to
acknowledge: the software is offered as is, and its numbers are the
user's to check against a tool they already trust. The project file,
`.vdyn`, is provisional: the next alpha will save projects in the
Engineering Sciences Common Data Format (`.escdf`), a public standard,
and will still open `.vdyn` files; the first release that is not an
alpha will not. Open each `.vdyn` once in that alpha and save it again.

**[Core principles](PRINCIPLES.md)** — thirteen principles
that shape every part of it: data you can see and handle, a workflow
that knows what comes next, an interface that shows only what applies,
a scripting API that can do everything the buttons can, a test for
every bug that ever got through, and dependencies tracked at their
latest versions rather than pinned.

**[Documentation](https://visualdynamics.org/documentation/)** —
getting started, guides, and a generated API reference, a page of
[visualdynamics.org](https://visualdynamics.org) rebuilt with every
release; the [source](docs/README.md) is in the tree and GitHub
renders it well enough to read. The application itself, and its
downloads, are at [visualdynamics.org](https://visualdynamics.org).
To build the site locally:

```bash
pip install -e '.[docs]' && properdocs serve  # localhost:8000
```

See [PRINCIPLES.md](PRINCIPLES.md) for what shapes the design.

Two front ends, one workflow: the desktop app and `visualdynamics.Project` hold the
same objects and call the same verbs, so anything you can click you can
script, and either saves a `.vdyn` file the other opens.

```python
import visualdynamics

project = visualdynamics.Project('Beam Airplane Modal Test')
project.import_file('modal_run.nc4')
project.set_basis(*project.names)
project.fit_modes(project.basis.frf, bounds=(5, 120))
project.basis.frf.plot()                  # the app's own plot
project.export_report(project.generate_report('modal'), 'report.html')
```

What it does today: import sdynpy/exodus/UNV/Rattlesnake files (declaring
units where the source is silent), write a specification back out as
the target Rattlesnake loads, view everything as grids, plots, CMIF
and coherence maps, animate shapes and time data on geometry, mark
measurement DOFs with labeled axis-colored arrows, **fit modal models to
FRFs interactively** (residual CMIF, one mode at a time, MAC beside the
table), compare shape sets by cross-MAC with phase-aligned overlay
animation, compute averaged spectra and PSDs from multi-average time
data, keep test photos in the project, and **generate complete modal test
reports** — a bar of acts and a settings pane beside the exported page
itself as the live preview, markdown text with live `{{Object.field}}`
values and self-renumbering figure references, exported as one
self-contained HTML file that animates mode
shapes in any browser. Export includes mode shapes riding their geometry
into exodus files ParaView animates, and whole projects as one `.vdyn`
file.

## Where this lives

The checkout is `~/visualdynamics`, and the package is
`src/visualdynamics` inside it. The `src/` is not decoration: a
directory named `visualdynamics` shadows the installed package for any
Python started in its *parent* directory, so before the move
`import visualdynamics` from `~` bound to an empty namespace package —
`__file__` of `None`, and an `AttributeError` on everything, rather than
the `ImportError` that would have explained itself. Under `src/` no
directory the checkout could be named collides with the import.

Moving it is not just a `mv`. The virtualenv records the absolute path
in every console script's shebang and in `pyvenv.cfg`, and both
editable installs record it too, so a move wants those rewritten and
`pip install -e .` run again in each environment.

## Install

The packaged application — one download per platform, nothing else
to install — is on the [downloads
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
optional so a plain install stays lean: `pip install -e '.[step]'`.
Every packaged build carries it.

**Packaged builds** — a `.dmg` for Apple silicon and one for Intel, a
Windows installer, a Linux AppImage — are made from this tree by
`packaging/refresh_builds.sh` (macOS natively, Windows under Wine,
Linux on a Debian box) and staged for the site preview at
http://127.0.0.1:8710/downloads.html; `packaging/README.md` has the
recipes and every trap they learned.

On a Mac, `python tools/make_app.py` writes
`~/Applications/Visual Dynamics.app` — a real bundle with a real icon,
for Finder, Spotlight and the Dock. It is a launcher and not a frozen
application: it runs the checkout's own interpreter, so an edit to the
source is live on the next launch, and it points at that checkout by
absolute path. Move the repository and run it again.

The distribution name and the import are the same word. The project
was called `vibe` until August 2026,
which could never have been published under that name — `vibe` belongs
to an unrelated package on PyPI.

## Contributing

**Bug reports are very welcome, and I will try to fix them
promptly.** Say what you did, what happened, and what you expected —
a `.vdyn` project file or a small script that reproduces it is the
fastest possible bug report.

**To contribute more than that, get in touch first:
[contact@visualdynamics.org](mailto:contact@visualdynamics.org).**
This is one person's project — code contributions are by invitation
for now, with community contributions to follow once the footing is
there — and a conversation before the work saves you writing something
already half-built or aimed somewhere it was not going.

See [CONTRIBUTING.md](CONTRIBUTING.md) for what a contribution has to
satisfy.

Help that is not code is welcome too — advice, review, measurements, a
bug report that turns out to be a real one. It is recorded in
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md), which is thanks rather
than authorship.

## License

Copyright (c) 2026 Brandon Zwink. Visual Dynamics is free software
under the [GNU General Public License, version 3 or any later
version](LICENSE): install it, use it, read it, change it, and pass it
on — a distributed derivative stays under the same terms. The outputs
are yours: plots, reports, exported files, numbers you put in a paper
or hand to a customer. Nothing here claims any interest in your data
or in what the software tells you about it.

Other terms are available from the copyright holder
(contact@visualdynamics.org). Contributions are accepted under
[CLA.md](CLA.md), which is what keeps that possible; see
[CONTRIBUTING.md](CONTRIBUTING.md). Third-party licenses are listed in
[NOTICE.md](NOTICE.md).
