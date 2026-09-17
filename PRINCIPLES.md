# Core principles

Thirteen principles, learned while building it rather than decided
in advance. They are the reason the software looks the way it does, and
they are the standard a change is measured against — by a person
reading a pull request, or by an AI assistant working in this
repository (the repository's rulebook for assistants points here).

Where a principle has a sharp edge, the edge is stated. A principle
that only ever agrees with you is not doing any work.

---

## What it is for

**1. Data should be seen, and handled.**
A number in a table is a claim; a plot is the evidence. Every
measurement in a project can be drawn, and the drawings are
interactive — zoom into a shock transient, rotate a 3-D waterfall,
drag the corner of a filter and watch the trace change under it.
Where a reading has a natural 3-D form, it gets one: channels
receding along a depth axis rather than thirty curves overlaid into a
band.

**2. The workflow is the product.**
A structural dynamics analysis is a sequence — import a run, mark the
events, filter, integrate, compute the spectra, judge them against a
specification, write the report. The software knows that sequence.
A project skeleton shows the steps a test of its type expects, filled
or still empty, so what to do next is visible rather than
remembered. Derived objects know what they came from and say when
their source has moved beneath them.

**3. Show only what applies.**
The interface offers what the current selection can actually do, and
nothing else. Select a time history and the averaging, shock and
filter views appear; select a spectrum and they are gone, not grayed.
An option that cannot be used is a question the user has to answer
before they can get on, and a grayed control still costs a glance.
The corollaries: say a thing once, refuse invalid state at entry
rather than reporting it afterwards, and make every table the same
table.

**4. Anything the interface can do, code can do.**
The desktop application and the scripting API are two front ends over
one implementation. Every button calls a method on `Project` that a
script may call identically — importing, computing, refreshing,
generating a report, exporting. Nothing is reachable only by mouse.
This is what makes a result reproducible: an analysis done by hand on
Tuesday can be replayed exactly, in a loop, over a hundred runs.

**5. What can be read can be written.**
If Visual Dynamics imports a format, it exports that format, and the
round trip is tested rather than assumed. A tool that can only read
is a dead end — it strands your data inside it.

*The exceptions, stated so they are decisions and not gaps:*

- **Rattlesnake `.nc4`** is read and never written, permanently. It is
  the record of a test as the controller ran it, carrying hardware and
  environment settings this toolset never holds — a file written from
  here would be a controller run that never happened.
- **Femap `.neu` and Nastran `.pch`** are read-only *for now*, not on
  principle. Both are on `REMAINING-TASKS.md`; the readers exist
  because a correlation needs the modes, and the writers wait until
  somebody needs to send a model back the other way.

## How it is built

**6. Every function says what it does; the scripting surface also says
how to call it.**
Two standards, because one was the wrong answer. *Everything* public
carries a docstring and a typed signature — the annotations are the
published API, shipped as `py.typed`, so an editor driving a script
shows real types. On top of that, the **scripting surface** —
`Project`'s verbs and the classes the package exports — carries NumPy
`Parameters` and `Returns` sections, because that is where somebody
arrives with no context and a table earns its place.

Everywhere else the docstrings stay prose, deliberately. They explain
the *reasoning* — especially where the obvious approach was tried and
rejected, since several bugs here were caused by code that looked
right — and a Parameters table cannot say that. Mandating one
everywhere would trade the useful half of the documentation for the
mechanical half. `tests/test_docstrings.py` enforces both standards,
and the list of what must carry sections only ever grows.

**7. Few dependencies, and boring ones.**
The scientific mainstream — NumPy, SciPy, and the established readers
for the formats involved — plus Qt for the interface. A new
dependency has to earn its place against the cost of everyone
installing it, of it going unmaintained, and of its license
constraining what this project may do. Reimplementing a well-defined
method in fifty lines is usually the better trade.

**8. Interaction is where the time budget goes.**
Effort is spent where the user is waiting. What is under the hand —
a dragged filter corner, a dragged averaging span, a rotating
scene — updates immediately; heavy work that nobody is watching yet
is deferred or coalesced until the gesture settles. Slowness is
measured before it is optimized, and the fix is nearly always
algorithmic rather than clever.

**9. One implementation of a rule.**
Where two parts of the system would answer the same question, they
share the answer rather than each computing it. The 2-D plot, the 3-D
scene and the report renderer draw on different surfaces but consult
one decision layer — a picture that disagrees with the number
computed beside it is worse than no picture. Duplication that has
already diverged is a bug that has not been noticed yet.

**10. AI is a welcome way to build this, and the discipline is what
makes it one.**
Much of Visual Dynamics was written with AI assistance, deliberately:
it is how a single person keeps a toolset this size moving and turns
a reported bug around in an afternoon. That works *because* of the
rules around it, not in spite of them. AI writes code that looks
right — which is exactly the failure this codebase guards against —
so a change is not finished when it runs. It is finished when a test
fails without it (11), when the reasoning is in the docstring (6),
and when someone has checked it did not quietly import something it
should not have. The repository carries a rulebook an assistant works
under; the speed is the point, and the guardrails are what make the
speed safe.

**11. Every bug becomes a test, and every test is falsified.**
Coverage is kept as high as it can honestly be — the suite runs some
three thousand tests in about three minutes, so there is no
excuse not to run it. But the number matters less than the habit: when
a wrong answer is found, the fix comes with a test that would have
caught it, pinned to the specific thing that was wrong — an axis
label, a unit, a count — rather than to "something was built".

And the test is checked by breaking the fix on purpose and watching it
fail. **A test that still passes with the fix reverted is testing
nothing**, and this project has caught several of those. An
intermittent failure is treated as a bug until proved otherwise; the
word "flake" is not used, because twice here it was a real latch both
times.

**12. Track the latest version of everything, and never pin around a
break.**
Dependencies are declared by name with no upper bound and no pin —
all thirteen of them, today — and there is no version-gated
compatibility code in the package. When NumPy, Qt or VTK ships a
release that breaks something here, the fix is to work with the new
version, not to cap the old one and move on.

The reasoning is about where the pain lands. A pin defers a small
problem into a large one: caps accumulate, they conflict with each
other, and eventually the project can only be installed alongside a
museum. Taking each break as it arrives keeps every one of them
small, and keeps this installable beside whatever else a test
engineer already has in their environment — which for a tool meant to
sit in someone's existing workflow is the difference between usable
and not.

What makes it affordable is the suite. Three thousand tests in about
three minutes is an early-warning system, run on a clean
machine before every release precisely so that a dependency's
breaking change is found by a machine, ahead of time, rather than by
a person mid-analysis.
A green suite against current versions is the claim this principle
makes; anything less would be a pin wearing a disguise.

**13. Processing is select, see, set, apply.**
Every computation on data follows one workflow, stated here because
it emerged piecewise and is now the standard (Brandon, 2026-08-29):
select the data to process; toggle to the process's reading on the
bar — averaging, shock detection, filtering, truncation; set its
parameters in the pane on the right while the plot previews the
effect live, on the very data it will be made from; and press the
act's own button on that pane to create the derived object. The
settings ride the record, the preview is the evidence, and the
button runs the same project verb a script calls — so provenance,
staleness and the API never notice which door was used. A
computation with no settings is an act, not a reading — an icon
button on the same bar, the verb leading its tooltip, pressed once — so the bar is the one place
any act lives (Brandon, 2026-09-04; three homes confused a user). A
new computation added to the application arrives with all four
parts or it is not finished: settings with no preview ask for blind
trust, a preview with no apply is a tease, and an act whose
settings live elsewhere is the workflow this principle replaced.

---

## What these add up to

Visual Dynamics is for a test engineer who has data and needs an
answer they can defend. Its strengths follow from the list: you can
*see* what you have, follow a workflow that knows what comes next,
drive it by hand or by script interchangeably, and get your data back
out in the format you brought it in. It is not a general numerical
library and does not try to be one.
