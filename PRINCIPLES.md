# Core principles

These thirteen principles came out of building Visual Dynamics; they
were not decided in advance. They explain why the software looks the
way it does, and they are the standard a change is measured against,
whether a person is reviewing a pull request or an AI assistant is
working in the repository (the repository's rulebook for assistants
points here).

Where a principle has a hard edge case, the edge case is stated. A
principle that never rules anything out is not doing any work.

## What it is for

**1. Data should be seen, and handled.**
A number in a table is a claim, and a plot is the evidence for it.
Every measurement in a project can be drawn, and the plots are
interactive: you can zoom into a shock transient, rotate a 3-D
waterfall, or drag the corner of a filter and watch the trace change
under it. Where a reading has a natural 3-D form, it is drawn in 3-D,
for example as channels set one behind the other along a depth axis
instead of thirty curves overlaid into a band.

**2. The workflow is the product.**
A structural dynamics analysis is a sequence of steps: import a run,
mark the events, filter, integrate, compute the spectra, compare them
to a specification, and write the report. The software knows that
sequence. A project skeleton shows the steps a test of its type
expects, filled or still empty, so the next step is on the screen and
does not have to be remembered. Derived objects record what they were
made from and show when their source has changed since.

**3. Show only what applies.**
The interface offers what the current selection can do, and nothing
else. Select a time history and the averaging, shock and filter views
appear; select a spectrum and they are gone, not grayed out. An option
that cannot be used is a question the user has to answer before they
can carry on, and even a grayed control costs a glance. Three rules
follow from this: say a thing once, refuse invalid input when it is
entered instead of reporting it afterwards, and make every table work
the same way.

**4. Anything the interface can do, code can do.**
The desktop application and the scripting API are two front ends over
one implementation. Every button calls a method on `Project` that a
script can call the same way: importing, computing, refreshing,
generating a report, exporting. Nothing is reachable only with the
mouse. This is what makes a result reproducible. An analysis done by
hand on Tuesday can be replayed exactly, in a loop, over a hundred
runs.

**5. What can be read can be written.**
If Visual Dynamics imports a format, it exports that format, and the
round trip is tested. A tool that can only read a format is a dead
end, because the data cannot get back out of it.

The exceptions are listed here so that it is clear they are decisions
and not gaps:

- Rattlesnake `.nc4` is never written from what Visual Dynamics holds.
  The file is the record of a test as the controller ran it, with
  hardware and environment settings this toolset does not keep, so a
  file written from a project would describe a controller run that
  never happened. The one thing written is a truncated copy of an
  existing run: every group, attribute and variable copied as the
  file has it, with only the time data cut to the chosen stretch.
- STEP and IGES are import only. Visual Dynamics holds meshes, not
  surfaces, so writing B-rep geometry back out would mean inventing
  surfaces the project does not have.
- Femap `.neu` and Nastran `.pch` are read only for now, as a matter
  of priority. Both writers are listed in `REMAINING-TASKS.md`. The
  readers exist because a correlation needs the modes; the writers
  wait until somebody needs to send a model back the other way.

## How it is built

**6. Every function says what it does; the scripting surface also says
how to call it.**
There are two standards. Everything public has a docstring and a typed
signature. The annotations are the published API, shipped with
`py.typed`, so an editor shows real types while you write a script. In
addition, the scripting surface (the verbs on `Project` and the
classes the package exports) has NumPy `Parameters` and `Returns`
sections, because that is where somebody arrives with no context and
a table of parameters is useful.

Everywhere else the docstrings are prose, on purpose. They explain the
reasoning, especially where the obvious approach was tried and
rejected, since several bugs here were caused by code that looked
right. A Parameters table cannot say that, and requiring one
everywhere would replace the useful half of the documentation with the
mechanical half. `tests/test_docstrings.py` enforces both standards,
and the list of what must have the sections only grows.

**7. Few dependencies, and boring ones.**
The dependencies are the scientific mainstream (NumPy, SciPy, pandas,
pint), the established libraries for the file formats involved, and Qt
and VTK for the interface and the 3-D view. A new dependency has to be
worth the cost of everyone installing it, the risk of it going
unmaintained, and the limits its license may put on this project.
Writing a well-defined method in fifty lines is usually the better
choice.

**8. Interaction is where the time budget goes.**
Effort is spent where the user is waiting. Whatever is under the mouse
(a dragged filter corner, a dragged averaging span, a rotating scene)
updates immediately. Heavy work that nobody is watching yet is put off
or combined until the gesture ends. Slowness is measured before it is
optimized, and the fix has nearly always been a better algorithm
instead of a clever trick.

**9. One implementation of a rule.**
Where two parts of the system would answer the same question, they
share one answer instead of each computing its own. The 2-D plot, the
3-D scene and the report renderer draw on different surfaces but use
one decision layer, because a picture that disagrees with the number
computed beside it is worse than no picture. Duplicated code that has
already drifted apart is a bug that has not been noticed yet.

**10. AI is a welcome way to build this, and the discipline is what
makes it one.**
Much of Visual Dynamics was written with AI assistance, on purpose.
It is how one person keeps a toolset this size moving and can turn a
reported bug around in an afternoon. That works because of the rules
around it. AI writes code that looks right, which is exactly the
failure this codebase guards against, so a change is not finished when
it runs. It is finished when a test fails without it (11), when the
reasoning is in the docstring (6), and when someone has checked that
it did not import something it should not have. The repository has a
rulebook that an assistant works under. The speed is the point, and
the guardrails are what make the speed safe.

**11. Every bug becomes a test, and every test is falsified.**
Coverage is kept as high as real tests can make it. The suite runs nearly
five thousand tests in about four minutes, so there is no reason not
to run it. The habit matters more than the number: when a wrong
answer is found, the fix comes with a test that would have caught it,
pinned to the specific thing that was wrong (an axis label, a unit, a
count), never to "something was built".

Each test is then checked by breaking the fix on purpose and watching
the test fail. A test that still passes with the fix reverted is
testing nothing, and this project has caught several of those. An
intermittent failure is treated as a bug until proved otherwise. The
word "flake" is not used here, because one failure that was dismissed
that way twice turned out to be a real latch both times.

**12. Track the latest version of everything, and never pin around a
break.**
Every dependency, runtime and optional, is declared by name with no
upper bound and no pin, and the package has no code that branches on
a dependency's version to stay compatible with an old one. When NumPy,
Qt or VTK ships a release that breaks something here, the fix is to
work with the new version, not to cap the old one and move on.

The reasoning is about where the cost falls. A pin turns a small
problem now into a large one later: caps accumulate, they conflict
with each other, and eventually the project can only be installed
next to very old versions of everything else. Taking each break as it
arrives keeps every one of them small. It also keeps Visual Dynamics
installable next to whatever a test engineer already has in their
environment, and for a tool meant to fit into an existing workflow
that decides whether it can be used at all.

The suite is what makes this affordable. It runs on a clean machine
on every pull request and once a week even when nothing has changed,
so a breaking change in a dependency is found by a machine ahead of
time, not by a person in the middle of an analysis. A green suite
against current versions is what this principle promises; anything
less would be a pin by another name.

**13. Processing is select, see, set, apply.**
Every computation on data follows one workflow. Select the data to
process. Switch the bar to the process's reading (averaging, shock
detection, filtering, truncation). Set its parameters in the pane on
the right while the plot previews the effect live, on the same data it
will be made from. Then press the button on that pane to create the
derived object. The settings are stored with the result, the preview
is the evidence, and the button runs the same project verb a script
calls, so provenance, staleness and the API work the same whichever
way the computation was started. A computation with no settings is an
act, not a reading: an icon button on the same bar, with the verb at
the start of its tooltip, pressed once. That keeps every act in one
place. A new computation added to the application is not finished
until it has all four parts. Settings without a preview ask the user
to trust them blind, a preview without an apply button leaves the user
nothing to do with it, and an act whose settings are somewhere else
splits one task across two places.

## What these add up to

Visual Dynamics is for a test engineer who has data and needs an
answer they can defend. With it you can see what you have, follow a
workflow that knows what comes next, work by hand or by script with
the same results, and get your data back out in the format you brought
it in. It is not a general numerical library.
