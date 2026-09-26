# Contributing

Thanks for taking a look.

**Contributions are welcome** — bug reports, fixes, features,
documentation, tests. Open an issue or a pull request; nobody needs
an invitation.

**A bug report is the easiest place to start**, and I will try to fix
what is reported promptly: what you did, what happened, what you
expected. A report of something wrong is worth a great deal to this
project, because it is the part I cannot do alone.

**A pull request that keeps the project's rules will be merged.** The
rules are written down, and they are the same ones every change here
is held to, mine included: the principles (`PRINCIPLES.md`), the
architecture and its conventions (`docs/architecture.md`), the rules
for working in the tree (`AGENTS.md`) and the three conditions below.
Meeting them is what a review looks for. When a change is declined or
needs work, the review says which rule and why.

**For anything substantial, open an issue first.** Not a requirement
— a courtesy to your own time. A sentence about what you would like to
change and why can save you writing something already half-built, or
aimed somewhere the project is not going. You can also write to
[contact@visualdynamics.org](mailto:contact@visualdynamics.org).

How the package is put together, and the conventions a change is
expected to keep — the layers, the one-workflow rule, the docstring
standard, the test conventions, the tests that hold the rules — is
[`docs/architecture.md`](docs/architecture.md). Working with an AI
assistant? [`AGENTS.md`](AGENTS.md) is the same set of rules in the
form an agent reads first, and it is what any such tool should be
pointed at before it touches the tree.

Three things any contributed code has to satisfy:

- **The contributor license agreement** ([`CLA.md`](CLA.md)), signed
  once. You keep your copyright; the Project gets a license broad
  enough that every line in it stays licensable by one person, which
  is what keeps its future options open. A standard text, adapted from
  Apache's. **Signing happens on your first pull request**: a bot
  comments with the agreement and a sentence to reply with, and the
  check stays red until that reply is on record, and a red check
  cannot be merged. Once is enough.
- **It has to be yours to give.** Your own work, not something an
  employer already has a claim on, and carrying no third-party code
  with obligations of its own.
- **No code from any package whose license would restrict the
  Project's licensing.** Copyleft code in particular: taking it would
  put lines in the tree that the Project does not own — the licenses
  would be compatible and the ownership would not be, and that can
  never be undone. File formats and published methods may be
  reimplemented; code may not be copied or derived from.
  `tests/test_license_boundary.py` checks this on every run for the
  packages the Project is closest to.

Help that is not code counts too, and is recorded in
[`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) — thanks rather than a
rights record.

## Licensing

**GPL-3.0-or-later** (`LICENSE`): free to install, use, copy and
redistribute, with a distributed derivative staying under the same
terms — and other terms available from the owner
(contact@visualdynamics.org). Contributions arrive under
[`CLA.md`](CLA.md) so that stays possible: every line in the tree is
licensable by one person, which is what lets the owner offer the same
code under other terms or relicense a future version.

## Getting set up

```bash
git clone https://github.com/visualdynamics/visualdynamics
cd visualdynamics
pip install -e '.[dev,step]'
python -m pytest tests -q
```

Python 3.12 or newer. The desktop app is PySide6; the tests run headless
(`QT_QPA_PLATFORM=offscreen`), and on Linux under `xvfb-run`, because
VTK still wants a GL context for its screenshots. The `step` extra is
the OpenCASCADE kernel behind STEP/IGES import (about 220 MB
installed); without it those tests skip and the rest of the suite is
unaffected.

## How a change travels

Work in a fork. The repository itself holds no working branches —
`main`, the CLA bot's signature branch and the maintainer's publishing
branches — and a ruleset refuses the creation of any other, the
maintainer included. Your fork is your own space: nobody else can
name a branch there, and the fork's name says whose the branch is.

Name the branch for what it carries: `fix/`, `feature/`, `docs/` or
`tooling/`, then a short lower-case slug of letters, digits and
hyphens — `fix/decade-axis-ticks`, `docs/random-workflow`. Open the
pull request against `main`. A check on every pull request reads the
branch name and stays red until it fits; renaming the branch in your
fork is enough, the pull request follows the rename.

## Before you push

```bash
ruff check .                    # lint
python -m pytest tests -q       # the suite, about a minute
python -m pytest tests -q -m slow   # the demonstration model, another minute
properdocs build --strict       # the docs site, if you touched docstrings
```

CI runs the same on every pull request, on every push to `main`
and on demand, on both Pythons the package promises. Nothing reaches
`main` except through a pull request with four green checks — the
tests on each Python, the documentation build and the CLA — and
that holds for the owner too.

The `slow` mark is the demonstration airframe
(`visualdynamics.demo.drone`), which the website, the docs and the
examples are built from. Checking it means a 6732-degree-of-freedom
eigensolution — three quarters of a minute that is about the demo rather
than the package — so it is deselected by default and run when the
demonstration itself is touched. The rules it used to carry that belong
to the toolset are stated on a three-quad strip in `tests/test_fem.py`.

## What the project expects of a change

- **One implementation of a rule.** The desktop app and `visualdynamics.Project`
  are two front ends over one workflow: put behavior in the project (or
  in `core`) and let the GUI display it. A rule written twice is a rule
  that will disagree with itself.
- **Tests that would fail without the change.** Prefer a test that
  encodes the behavior to one that pins the implementation.
- **Docstrings that say why.** Reference documentation is generated from
  them (`docs/api`), and the guides in `docs/guide` explain reasoning.
  Comments earn their place by stating a constraint the code cannot.
- **No code the Project does not own in the package.** Importing a
  package whose license would restrict the Project's licensing — a
  copyleft one in particular — would put every future licensing choice
  out of reach: not because the licenses clash, but because the lines
  would not be the owner's to relicense. File formats are read as the
  documented formats they are, and the methods are written here.
  `tests/test_license_boundary.py` checks this on every run rather than
  trusting anyone to remember it. There is no exception, including for
  tooling that is never distributed; the scripts under `testdata/` that
  make the suite's data import nothing copyleft, and what the suite
  reads is their committed output.
- **Every bug fixed comes with a test that fails without the fix.**
  Break the change on purpose and watch the test fail before calling it
  done; a test that passes either way is testing nothing.

`PRINCIPLES.md` is the standard a change is measured against — worth
reading before a substantial one.

## Reporting a problem

Say what you did, what happened, and what you expected. A `.vdyn` project
file or a small script that reproduces it is the fastest possible bug
report.
