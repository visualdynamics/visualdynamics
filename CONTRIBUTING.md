# Contributing

Thanks for taking a look.

Pull requests are welcome from anyone. Bug fixes, improvements to
something that is already here, and new additions are all wanted, and
you do not need an invitation or anyone's permission to open one. If
something got in your way while you were using Visual Dynamics, the
fix for it is a good first pull request.

A pull request that keeps the project's rules will be merged. The rules
are written down, and every change here is held to them, mine
included: the principles (`PRINCIPLES.md`), the architecture and its
conventions (`docs/architecture.md`), the rules for working in the tree
(`AGENTS.md`) and the two conditions below. A review checks a change
against them. When a change needs more work, the review says which rule
and why, and helps get it there. Leave *Allow edits by maintainers*
ticked on the pull request, so that a small fix or a rebase onto `main`
can be made on your branch instead of being sent back to you.

Small contributions count as much as large ones: a fix for one wrong
number, a test for a case nobody covered, a clearer sentence in a
guide, a reader for one more file format. A bug report is welcome too,
and I will try to fix what is reported promptly. Say what you did, what
happened, and what you expected.

An issue before a pull request is never required. For a large
addition, a short issue saying what you plan can still save you time,
since it may already be half-built or there may be a simpler place for
it. Questions are welcome at
[contact@visualdynamics.org](mailto:contact@visualdynamics.org).

[`docs/architecture.md`](docs/architecture.md) describes how the
package is put together and the conventions a change is expected to
keep: the layers, the one-workflow rule, the docstring standard, the
test conventions and the tests that enforce the rules. If you work
with an AI assistant, [`AGENTS.md`](AGENTS.md) is the same set of rules
in the form an assistant reads first. Point the tool at it before it
touches the tree.

Any contributed code has to meet two conditions:

- It has to be yours to give. It must be your own work, not something
  an employer already has a claim on, and it must carry no third-party
  code with obligations of its own. By opening a pull request you offer
  the change under the Project's MIT License, the same terms everyone
  else receives it under.
- It may contain no code from any package under a copyleft license.
  GPL or AGPL code cannot be part of an MIT project: including it would
  make every build that contains it subject to the copyleft terms,
  which the Project's license does not allow and which cannot be undone
  once a release is out. File formats and published methods may be
  implemented here from their descriptions; code may not be copied or
  derived from. There is no exception for tooling that is never
  distributed: the scripts under `testdata/` that make the suite's data
  import nothing copyleft, and the suite reads their committed output.
  `tests/test_license_boundary.py` checks this on every run for the
  packages the Project is closest to.

Help that is not code counts too, and is thanked on the website's
[acknowledgments](https://visualdynamics.org/acknowledgments.html)
page. That page is a thank-you, not a record of rights
([`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) says so).

## Licensing

The project is under the MIT License (`LICENSE`): you are free to
install, use, copy, change and redistribute it, in open or closed work,
as long as the copyright notice travels with it. Contributions come in
under the same license they go out under, and there is no separate
agreement to sign.

## Getting set up

```bash
git clone https://github.com/visualdynamics/visualdynamics
cd visualdynamics
pip install -e '.[dev,step]'
pip install 'escdf @ git+https://github.com/sandialabs/Engineering-Sciences-Common-Data-Format.git'
python -m pytest tests -q
```

The second `pip install` is the reference implementation of the
Engineering Sciences Common Data Format, which the tests of the
`.escdf` reader and writer check them against. It is not on PyPI, so
it cannot be an extra, and without it those tests are skipped.

Visual Dynamics needs Python 3.12 or newer. The desktop app is built on
PySide6. The tests run headless (`QT_QPA_PLATFORM=offscreen`), and on
Linux under `xvfb-run`, because VTK still needs a GL context for its
screenshots. The `step` extra is the OpenCASCADE kernel used for STEP
and IGES import (about 220 MB installed); without it those tests are
skipped and the rest of the suite is unaffected.

## How a change travels

Work in a fork. The repository itself holds no working branches, only
`main` and the maintainer's publishing branches, and a ruleset refuses
the creation of any other branch, the maintainer's included. Your fork
is your own space: nobody else can create a branch there, and the
fork's name says whose branch it is.

Name the branch for what it carries: `fix/`, `feature/`, `docs/` or
`tooling/`, then a short lower-case name of letters, digits and
hyphens, such as `fix/decade-axis-ticks` or `docs/random-workflow`.
Open the pull request against `main`. A check on every pull request
reads the branch name and fails until the name fits the scheme.
Renaming the branch in your fork is enough; the pull request follows
the rename.

## Before you push

```bash
ruff check .                        # lint
python -m pytest tests -q           # the suite, a few minutes
python -m pytest tests -q -m slow   # the demonstration models, a few more
properdocs build --strict           # the docs site, if you changed docstrings or guides
```

CI runs the same checks on every pull request, on every push to `main`,
once a week, and on demand, on Python 3.12 and 3.13. Nothing reaches `main` except through a pull
request with its checks green (the tests on each Python, the
documentation build and the branch name), and that holds for the owner
too.

The `slow` mark covers the demonstration models in `visualdynamics.demo`
that the website, the docs and the examples are built from (the drone
airframe, the Frame-Wing and the BARC), the demonstration projects built
on the plate, and one comparison that needs files from outside the
repository. Checking the drone alone means a 6732-degree-of-freedom
eigensolution, which takes most of a minute and tests the demo more than
the package, so these tests are deselected by default. Run them when
you change the demonstrations, the website or the examples. The
finite element rules that belong to the toolset itself are tested on
small models in `tests/test_fem.py`, which runs every time.

## What the project expects of a change

- One implementation of a rule. The desktop app and
  `visualdynamics.Project` are two front ends over one workflow: put
  behavior in the project (or in `core`) and let the GUI display it. A
  rule written twice will eventually disagree with itself.
- Tests that would fail without the change. A test that checks the
  behavior is better than one that pins the implementation.
- Docstrings that say why. The reference documentation is generated
  from them, and the guides in `docs/guide` explain the reasoning.
  Write a comment when it states a constraint the code cannot show.
- No copyleft code in the package, as set out in the second condition
  above.
- Every bug fix comes with a test that fails without the fix. Break the
  change on purpose and watch the test fail before calling it done; a
  test that passes either way is testing nothing.

`PRINCIPLES.md` is the standard a change is measured against, and it is
worth reading before you start on a substantial one.

## Reporting a problem

Say what you did, what happened, and what you expected. A `.vdyn`
project file or a small script that reproduces the problem is the
fastest way to get it fixed.
