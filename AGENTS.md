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
Brandon Zwink, published under the **MIT License** (`LICENSE`).
Contributions are welcome: bug reports, and pull requests from anyone,
offered under the same MIT License. A
pull request that keeps the rules in this file, `PRINCIPLES.md`,
`docs/architecture.md` and `CONTRIBUTING.md` will be merged. Bug
fixes, improvements and additions are all wanted, and an issue first is
never required. The maintainers review and merge. An agent working
in this tree does not accept contributions, make licensing claims or
speak for the project; it helps the person it is working with make a
change that keeps those rules.

Three names, on purpose: **Visual Dynamics** is what a person reads
(window title, app bundle, file dialogs, report headings),
`visualdynamics` is the package and the import, `.vdyn` is the
project file — its format may still change during the alpha, as the
README says.

## Hard rules

These are not preferences. Each one exists because breaking it costs
something that cannot be bought back.

1. **No copyleft code.** GPL or AGPL code must never be imported,
   copied, or derived from (subclassing is deriving, even in code
   that is never distributed). It cannot live in an MIT project: one
   import would put every build that carries it under the copyleft
   terms, and a release cannot take that back. Read such a package to
   understand a method; write the method here yourself. Its
   *numbers* may be compared against as frozen data (a program's
   output is not a covered work); its code may not travel here.
   `tests/test_license_boundary.py` enforces the rule by AST walk on
   every run, `tests/` included, for the packages this project is
   closest to.

2. **Nothing lands without a maintainer.** The public `main`
   accepts no direct push from anyone. A change arrives as a pull
   request, with CI green on both Pythons, the documentation build and
   the branch-name check — and a maintainer merges it, or does
   not. Do not
   try to route around that, and do not open a pull request on
   someone's behalf without their say-so.

3. **Read a dependency's license before adding it.** GPL is
   disqualifying; permissive (BSD/MIT/ISC) or LGPL with dynamic
   linking only. For Qt the line is **LGPL-available, not
   Essentials-only**: the sanctioned set is `QtCore`, `QtGui`,
   `QtWidgets`, `QtTest`, `QtWebEngineWidgets`, `QtWebEngineCore`,
   `QtWebChannel`, `QtMultimedia`. Before
   importing a Qt module that is not already used, read that module's
   own licensing page — some are GPL-or-commercial. Dependencies are
   declared by name with no pin and no upper bound (principle 12);
   never cap a version to work around a break.

4. **Never claim a license the tree does not grant.** Every file that
   says anything says MIT; nothing else. And the project
   is **unaffiliated**: never name an employer, client, laboratory or
   institution anywhere in it — code, tests, docs, commit messages.
   A test walks every tracked file for such names. The one exception
   is credit: where third-party work is used, its source is cited in a
   fixed phrase the test knows, and nowhere else.

5. **Never commit generated data or large binaries.** Fixtures under
   `testdata/` are small and deliberate. Never `git add -A` without
   reading `git status` first, and never `git checkout` or
   `git restore` a file holding uncommitted work — copy it aside
   first.

## Branches

The repository holds no working branches: `main`, `cla-signatures`
(the record of a retired contributor agreement), the publishing
branches, the merge queue's own temporary branches and, should a
backport ever be needed, `maintenance/X.Y.x`.
A ruleset refuses the creation of anything else, the maintainers
included, and write access is only for merging — so every working
branch, yours and the maintainers' alike, lives in a fork. Your fork is your folder:
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

A pull request merges through a merge queue. A maintainer adds it once
its checks pass, and the queue tests it again on top of `main` and
everything queued ahead of it, merging only if that passes. There is no
need to keep a pull request rebased on `main` while it waits; rebase
only when it has a conflict.

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
- CI runs the same suite on every pull request, on Python 3.12, 3.13
  and 3.14, with the demonstration models in a job beside it, and is
  what decides whether a change can merge; the merge queue runs the
  suite once more on 3.14 on top of `main`. It also runs weekly
  against whatever the dependencies' latest releases are, since none
  of them is pinned, and the weekly run and every push to `main` check
  the guide's mesh table too (`-m guide`, minutes of meshing a shape,
  left off pull requests). The local gate is still the gate.

## Verification discipline

- **Falsify every fix.** Break the change deliberately and watch a
  named test fail. A test that passes with the fix reverted is
  testing nothing. Pin the thing that differs (a label, a unit, a
  count), not merely "something was built".
- **The same, a module at a time: `tools/mutate.py MODULE TESTS…`**
  makes one small change to the code at a time and prints every
  change the named tests do not notice. Each is a gap in the tests
  or a change that cannot alter behavior; reading which is the work.
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

## What review catches most often

Most of the work a pull request needs after it is opened comes from
the same few gaps. Check each one before asking for review.

- **Call what is already here.** Before writing a computation, search
  `core/` for it. Spectral framing and averaging
  (`TimeHistory._spectral_frame`), judging a spectrum against its
  limits (`compliance.judge`, `compliance.compare`), sine errors
  (`compliance.sine_errors`), a filter's response (`filters.response`),
  coherence (`Psd.coherence`), the correlation between overlapped
  frames (`averaging.frame_correlation`) and the square CSD matrix
  built from records (`transform._hermitian_block`) each exist once. A
  second copy of a formula drifts from the first, and the review will
  ask for it to be replaced by a call. Building on a function, keep
  the names of the keys it returns, so the helpers that read them
  (`compliance.readouts`, `compliance.verdict`) still apply.
- **Take a fact from where the package keeps it.** Control channels
  are the channel table's Control flag (`ChannelTable.controls()`); the
  level a channel clips at is `ChannelTable.clip_ranges()`; a record's
  identity is `channel_key`. A CPSD record is stored as
  conj(X_a)·X_b (`compute_cpsds`), so a response is
  conj(H) S Hᵀ, not H S Hᴴ.
- **Dead, NaN and empty inputs.** Test every new reading against an
  all-zero channel, a constant channel, a record with one NaN sample
  and an empty selection. Each must give a message that names the
  cause. A check that judged nothing says so everywhere its result is
  read (its words, the status line, the verdict); it never reports a
  pass, and never "0 lines outside".
- **Refuse what cannot be read one way.** A value the input does not
  pin down is refused with the line or cell named, not guessed: a
  short row in a whitespace-separated file, a name that matches two
  columns, a stated unit that contradicts the header's, a rate whose
  unit is not a frequency.
- **Match records by their full identity.** Two records at one DOF can
  be different quantities (an acceleration and a force at `1Z+`).
  Pair records by `channel_key` (DOF, dimension and hint), and refuse
  a record whose units do not match what it is compared against.
- **Test at the edges, with the defaults.** The easy case passes
  almost any implementation. Test a tone on the first spectral line,
  a frequency at the edge of the span, a record with a nonzero start
  time, noise on the input as well as the output, two tones that
  cross, and a band only partly covered by the data. Test under the
  default window (Hann), not only boxcar. A multi-input calculation
  is tested with a complex plant and complex cross terms against an
  independent truth (simulated signals run through the package's own
  estimators); a real-valued test cannot see a conjugation error.
- **Check a statistical claim against noise.** A degrees-of-freedom
  count, a confidence band or a tolerance is checked by simulating
  white noise with the default settings and counting how often the
  rule trips. Adjacent lines are correlated under any window but
  boxcar, a cell that spans several lines has those lines' degrees of
  freedom, and a tolerance tighter than the estimate's own scatter
  fails a correct test by chance.
- **Say a reading's limits where it is read.** If a result is
  unreliable in some range (a tone the window cannot resolve, the
  start of a sweep), the pane and the status line say so for that
  result, not only the docstring.
- **Refusals live in `core`.** A rule that only the GUI enforces is
  one the API skips. Put the check where both front ends call it, and
  let the GUI show its message. A setting that does nothing in some
  mode is refused in that mode, not ignored.
- **A claim is a test.** What a docstring, the guide or the pull
  request says ("every impossible entry is refused", "works under a
  strict content security policy", "within a sample") has a test that
  would fail if it were false, or it is reworded to what is tested.
- **Run `tools/mutate.py` over a new numeric module** against its
  tests, and list the survivors in the pull request with a line on
  each: a gap now closed, or a change that cannot alter behavior.
- **Views follow their inputs; slow work waits to be asked.** A pane,
  caption or plot that depends on a selection or a record clears or
  redraws when that input changes. A computation runs when its result
  is needed, not on every redraw, and work that takes more than about
  a tenth of a second does not run on the GUI thread.
- **The branch is yours whole.** Rebase onto current `main` before
  asking for review. A test that fails on your branch and passes on
  `main` is your branch's, whatever else fails on your machine. Tests
  pass on every platform CI runs; a test that reads pixels leaves a
  margin for antialiasing.
- **Published text** is American English (color, labeled, analyze),
  cites the edition of a standard that was actually read, and carries
  no decision history or notes about the author's own machine.
  Comments and test docstrings say why, not who asked: no reviewer's
  name, and nothing that points outside the repository. A guide figure
  added or changed is named in the pull request; a maintainer
  renders it again so every figure shares one typeface.

## Style

- Comments and docstrings explain *why*, especially where the obvious
  choice was tried and rejected. Match the surrounding density; the
  codebase is heavily commented by intention.
- **Published text reads as a plain manual.** The documentation and
  every public docstring (the API reference is generated from them)
  say what the code does and why, in plain words, with no dates or
  names. The history of a decision goes in a `#` comment beside the
  code it explains. `docs/architecture.md` shows the pattern.
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
