# Rules for AI agents working on Visual Dynamics

This file is for any AI assistant — Claude, or anything else that
reads `AGENTS.md` — working in a checkout of the public Visual
Dynamics repository. Read it, then **`CONTRIBUTING.md`** (how a
change reaches the project), **`PRINCIPLES.md`** (the thirteen
principles a change is measured against) and
**`docs/architecture.md`** (the package by layer, the one-workflow
rule, the docstring standard, the test conventions and the ratchet
tests that hold them), before doing real work. The guides under
`docs/` say what the application does; the docstrings are the API
reference.

## What this is

A from-scratch, units-aware structural dynamics toolset. Copyright
Brandon Zwink, published under **GPL-3.0-or-later with a contributor
license agreement** (`LICENSE`, `CLA.md`). Contributions are
welcome: bug reports, and pull requests from anyone, under the CLA. A
pull request that keeps the rules in this file, `PRINCIPLES.md`,
`docs/architecture.md` and `CONTRIBUTING.md` will be merged; for
anything substantial, an issue first is a courtesy to the
contributor's own time. The owner reviews and merges. An agent working
in this tree does not accept contributions, make licensing claims or
speak for the project; it helps the person it is working with make a
change that keeps those rules.

Three names, on purpose: **Visual Dynamics** is what a person reads
(window title, app bundle, file dialogs, report headings),
`visualdynamics` is the package and the import, `.vdyn` is the
project file — provisional for the alpha, as the README says.

## Hard rules

These are not preferences. Each one exists because breaking it costs
something that cannot be bought back.

1. **No code from any package whose license would restrict this
   project's licensing.** Copyleft code — GPL, AGPL, however
   compatible with this project's own GPL — must never be imported,
   copied, or derived from (subclassing is deriving, even in code
   that is never distributed). Under this project's GPL an import
   would be *legal*; it would also put lines in the tree that the
   project does not own, which ends every licensing option the owner
   keeps, and that cannot be undone. Read such a package to
   understand a method; write the method here yourself. Its
   *numbers* may be compared against as frozen data (a program's
   output is not a covered work); its code may not travel here.
   `tests/test_license_boundary.py` enforces the rule by AST walk on
   every run, `tests/` included, for the packages this project is
   closest to.

2. **Nothing lands without the maintainer.** The public `main`
   accepts no direct push from anyone. A change arrives as a pull
   request, under the CLA (a bot asks for the signature on your first
   one), with CI green on both Pythons, the documentation build and
   the CLA check — and the maintainer merges it, or does not. Do not
   try to route around that, and do not open a pull request on
   someone's behalf without their say-so.

3. **Read a dependency's license before adding it.** GPL is
   disqualifying; permissive (BSD/MIT/ISC) or LGPL with dynamic
   linking only. For Qt the line is **LGPL-available, not
   Essentials-only**: the sanctioned set is `QtCore`, `QtGui`,
   `QtWidgets`, `QtTest`, `QtWebEngineWidgets`, `QtWebChannel`. Before
   importing a Qt module that is not already used, read that module's
   own licensing page — some are GPL-or-commercial. Dependencies are
   declared by name with no pin and no upper bound (principle 12);
   never cap a version to work around a break.

4. **Never claim a license the tree does not grant.** Every file that
   says anything says GPL-3.0-or-later; nothing else. And the project
   is **unaffiliated**: never name an employer, client, laboratory or
   institution anywhere in it — code, tests, docs, commit messages.
   A test walks every tracked file for such names.

5. **Never commit generated data or large binaries.** Fixtures under
   `testdata/` are small and deliberate. Never `git add -A` without
   reading `git status` first, and never `git checkout` or
   `git restore` a file holding uncommitted work — copy it aside
   first.

## Branches

The repository holds no working branches: `main`, the CLA bot's
`cla-signatures`, the maintainer's publishing branches and, should a
backport ever be needed, `maintenance/X.Y.x`. A ruleset refuses the
creation of anything else, the maintainer included, and no
contributor has write access — so every working branch, yours and
the maintainer's alike, lives in a fork. Your fork is your folder:
named for you, writable by nobody else. That is the structure the
large scientific Python projects run on, and it is what keeps the
canonical repository readable.

A pull request's head branch is named `kind/short-slug`: `fix/`,
`feature/`, `docs/` or `tooling/`, then lower-case letters, digits
and hyphens — `fix/decade-axis-ticks`, `feature/comparison-grid`. CI
checks the name on every pull request and the check is required, so
a branch named otherwise cannot merge; rename it in your fork (the
pull request follows the rename) and push again. When you make a
branch for the person you are working with, name it by the scheme.

## The gate

Before saying anything is done:

```bash
./.venv/bin/python -m pytest tests -q -n 4
./.venv/bin/ruff check .
```

- **Use the repository's own virtual environment**, not whatever
  `python` resolves to; another interpreter can bind the wrong Qt.
- Judge the run by pytest's **own exit code**, with the output written
  to a file — never through a pipe that reports its own status.
- `-n 4` locally; `auto` pegs every core. The demonstration model is
  marked `slow` and deselected; run `-m slow` when the demo, the
  website or the examples are touched.
- Docs: `properdocs build --strict` when docstrings or guides changed.
- CI runs the same suite on every pull request, on both Pythons, and
  is what decides whether a change can merge. The local gate is still
  the gate.

## Verification discipline

- **Falsify every fix.** Break the change deliberately and watch a
  named test fail. A test that passes with the fix reverted is
  testing nothing. Pin the thing that differs (a label, a unit, a
  count), not merely "something was built".
- **A falsification that prints nothing is not a pass.** Assert on
  the summary line.
- **Clear `__pycache__` after restoring a broken file** — a break and
  a restore in the same second leave size and mtime unchanged, and the
  interpreter keeps running the broken bytecode.
- **An intermittent failure is a bug until proved otherwise.** Never
  call it a flake; capture the assertion's values first.
- **Verify, don't assume.** Claims about the GUI must be measured:
  `tests/conftest.py` gives `window` (a headless `MainWindow`), `pump`
  and `fixture_path`, so a behavior gets a named test that runs in
  milliseconds. A build that succeeds proves nothing about an app
  that runs.
- **Never edit source while a suite is running.**
- **Own test-authorship errors plainly.** When a failure was a bad
  test rather than bad code, say which it was.

## Style

- Comments and docstrings explain *why*, especially where the obvious
  choice was tried and rejected. Match the surrounding density; the
  codebase is heavily commented by intention.
- **One implementation of a rule.** The desktop app and
  `visualdynamics.Project` are two front ends over one workflow: put
  behavior in the project (or `core`) and let the GUI display it.
- Keep `core`, `units`, `io`, `deform`, `rotate` and `viz` free of Qt.
  Qt belongs in `gui/`, `plot/` and `theme.py`.
- The principles that bite most often while writing code: show only
  what applies (3), anything the interface can do the API can do (4),
  what can be read can be written (5), few and boring dependencies
  (7), effort goes where the user is waiting (8), one implementation
  of a rule (9), every bug becomes a falsified test (11). A new
  computation ships with its reading, its settings pane, its live
  preview and its apply button — all four (13).

## Proposing a change

Say what you would like to change and why, at
`contact@visualdynamics.org`, before writing much: it may be
half-built already, or aimed somewhere it was not going. A bug report
— what you did, what happened, what you expected, with the file if you
can share it — is the contribution that helps most. When a change is
agreed, `CONTRIBUTING.md` says how to set up, what the project expects
of it, and how the pull request is checked.

The maintainer's working records — the design record and the session
handoff — live elsewhere and are not needed here: this tree is
complete for building, testing and packaging the software.
