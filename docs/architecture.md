---
title: Architecture and conventions
---

# Architecture and conventions

This page describes how the package is put together and the
conventions a change is expected to keep. It is written for someone
about to work on the code, not someone using it. The
[principles](principles.md) are the standard; this page is the map.
Most of what it describes is enforced by a test, and the last section
lists which.

## The package, by layer

Most of the package (about three lines in five) imports no Qt, and that
is a rule. The data model, the math, the file formats and the 3-D scene
are a library, and the desktop app is built on top of it. A change that
puts Qt into the wrong layer is refused in review, whatever else it
does.

| Layer | Modules | Qt? | What lives there |
|---|---|---|---|
| Data model | `core/` | no | The objects a test holds (`Geometry`, `TimeHistory`, `Spectrum`, `Frf`, `Psd`, `Srs`, `ShapeSet`, `ChannelTable`, the specifications) and every computation on them: spectral estimates, FRFs, shock response, kurtosis, the wavelet, filters, modal fitting, the checks a test is judged by (the SRS points rule, the shock and pulse checks, the time statistics, replication by segment, the sweep rate, the stroke and the fit to a shaker), rigid-body modes, transforms, the finite element model and the solids and meshes it is built from, and reports as data. |
| Units | `units.py`, `core/unit_choices.py` | no | Every quantity is stored in SI and displayed in the chosen system; an object imported without units stays unit-less until it is told its units. |
| Project | `project.py` | no | The `Project`: named objects, links, roles, the verbs and the journal (both below). This is the one workflow that the app and a script both drive. |
| Formats | `io/` | no | One module per format, each a reader and nearly always a writer, listed in a small registry so that a new format touches nothing else. `native.py` is `.vdyn`. |
| Geometry motion | `deform.py`, `rotate.py`, `decimate.py` | no | Data at DOFs turned into node motion; a coordinate system rotated by dragging its ring; decimation that keeps every peak. |
| 3-D | `viz/` | no | PyVista over VTK: the scene, the waterfall, the scalogram surface, the MAC bars, picking and animation, built on a fixed stage and relabeled with the real ranges. |
| 2-D | `plot/` | yes | pyqtgraph. One renderer serves the window and a script, so a figure saved to a file is the figure on screen. |
| App | `gui/` | yes | The window, the tree, the panes, the settings panels, the console and the report editor. It displays what the project does and decides nothing on its own. |
| Reports | `report/` | no | A `Report` rendered to one self-contained HTML file, with its viewers included. |
| Theme | `theme.py` | yes | The colors the package draws with, and which scheme applies. |

Two calls in `viz/` reach Qt when they run: an animation shown in its
own window (through pyvistaqt) and an animation saved as a movie
(through Qt Multimedia). Both import Qt inside the call, so importing
`viz` loads no Qt. The same holds for every layer marked "no": the
package imports without loading Qt at all.

## One workflow, two front ends

Everything the app can do, a script can do with the same call
(principle 4), because the app is effectively a script. The desktop
window and `visualdynamics.Project` drive one workflow, and every act
in the window is a verb on the project.

- Behavior goes on the project, or in `core`, and the GUI displays it.
  A new computation is a project verb (`compute_srs`, `filter_data`,
  `transform`, ...) with its math in a `core` module. The window's bar
  and menus present `Project.verbs()`, the one applicability table.
  Its summaries are the first paragraph of each verb's docstring, so
  the bar, the API and the reference cannot disagree. A verb the window
  offers and the API lacks is a bug, and `tests/test_verbs.py` finds
  it.
- The journal. Every verb the project runs, clicked or called, is
  recorded as runnable Python in `project.journal`. The console tab
  shows it, and `tests/test_journal.py` replays it and compares the
  projects. Nothing may be designed so that it cannot be replayed.
- Everything renders headless. Every view the window can show has a
  call that renders it to a file with no window and no event loop
  (`tests/test_headless.py` is the inventory). A view that gains a
  GUI-only path fails that test.
- Processing is select, see, set, apply (principle 13). A new
  computation ships with all four parts: a reading of the object it
  acts on, a settings panel (built from `gui/settings_panel.py`, beside
  the view, with its derived readings), a live preview drawn over the
  data, and an apply button that runs the verb. The filter, the
  truncation and the wavelet are good examples to follow.
- A reading of a time history is one row. The data bar's readings are
  `Reading` rows in `DataPane._READINGS` in `gui/panes.py`, and the
  pane offers and puts away every reading by looping over that table.
  A new reading needs its row and its action on the pane, and a render
  method in `main_window.py`.
- An animation shows one quantity. `deform.quantity_records` picks the
  records of one quantity for every deflection the app or a script
  draws, so a node is never moved by a force beside one moved by an
  acceleration.
- Staleness comes from provenance. A derived object records the
  settings it was made with. When its source's settings change, it
  shows a refresh badge with the reason. Recomputing is an act and is
  never automatic.

## Docstrings

The docstrings are the API reference. The site's reference pages are
generated from them at build time, and there is nothing else to keep
in step. There are three standards, enforced by
`tests/test_docstrings.py` and `tests/test_docstring_references.py`:

- Every public callable has a docstring. That is the minimum.
- The scripting surface (what `visualdynamics.__all__` exports and the
  public methods on those classes) also has NumPy `Parameters` and
  `Returns` sections, because that is where somebody arrives without
  context and a table of parameters is useful.
- Anything implementing published theory has a NumPy `References`
  section: the shock response spectrum's filter, the octave bands, the
  FRF estimators, the multiple coherence, the wavelet, the averaging,
  the kurtosis, the modal assurance criterion, the beam, plate and
  solid elements, the modal fit, the filters, the tracking filter, the
  half-power width and the signature comparison built on it, the
  classical shock pulses, the velocity check of a shock record, a tone
  read off an autospectrum, the time statistics, the replication read
  by segment and the fit of its reference to a shaker, the stroke, the
  sweep rate, and the build-up and decay of a dwell.
  A reader has to be able to check the equations against the source
  they came from, and the next person to work on the code has to know
  which of several formulations was chosen. Each entry says what it is
  used for, not only that it exists.

  Cite only what has been checked. A `References` section opens with a
  sentence saying which kind it is: the formulation this code follows,
  or a standard method and where it is described. Do not give chapter
  or section numbers. They look checkable, and when they are written
  from knowledge of a method instead of from the book in hand, they
  are often wrong. Give volume, issue and pages only next to a DOI
  link, where a reader can follow the link and check them. Everywhere
  else an entry stops at the venue.

  Write those entries as a Markdown numbered list, not as reST
  `.. [1]` citations. This site renders Markdown, so a `.. [1]` marker
  comes out literally and the entries run together in one paragraph.
  With `1.` the entries become an ordered list, `*title*` is
  italicized, and ordinary Markdown link syntax becomes a link a reader
  can follow.

Everywhere else, docstrings are prose. A published docstring says
what the function does and why it works the way it does, in plain
words; it carries no dates and no names. The history behind a decision
(why the obvious approach was tried and rejected, with the date and
the name of whoever decided) goes in a comment beside the code it
explains. Several bugs here were caused by code that looked right, and
a comment that says "this looks wrong, here is the measurement" is what
stops the next person from putting it back. Match the density of the
code around you. A typical one, from `core/wavelet.py`:

```python
def decade_values(low: float, high: float) -> list[float]:
    """The round frequencies to label a log axis with.

    1, 2 and 5 times each power of ten, every one inside
    ``[low, high]``, in ascending order. The transform's rows are
    twelfths of an octave and nobody reads 158.7 Hz off an axis, so the
    flat picture, the 3-D stage and the report figure all label these
    values instead, and all three call this function.
    """
    # One function rather than a loop in each caller, because the three
    # loops drifted apart once (2026-09-16: they were byte-for-byte the
    # same, and the fourth would not have been).
```

The docstring says what the function does and why it is here and not
somewhere else; the comment keeps the history that justifies it.
Neither restates the signature.

## Tests

The suite is the specification. About 4 850 tests run headless in
about four minutes on four workers, most of them through a real
`MainWindow` built off screen. The conventions:

- A test is named as a sentence about behavior, read from the user's
  side: `test_a_time_history_offers_the_wavelet`,
  `test_the_flat_picture_is_a_toggle_away`,
  `test_a_record_that_starts_late_is_drawn_on_its_own_clock`. The name
  is the claim, and the body pins it with a value (a label, a unit, a
  count), never just "something was built".
- A change ships with its test in the same commit. A feature without a
  test is not finished, and a bug fix without a test that fails on the
  old code is not a fix (principle 11).
- Every fix is falsified. Break the change on purpose and watch the
  named test fail; a test that passes with the fix reverted is testing
  nothing. Clear `__pycache__` between the break and the restore, or
  the interpreter keeps running the broken bytecode.
- A numeric module is falsified as a whole. `tools/mutate.py MODULE
  TESTS...` makes one small change at a time (a comparison flipped, an
  operator swapped, an integer constant moved by one), runs the named
  tests against each change in a private copy of `src/`, and prints
  every change the tests did not catch. Each survivor is either a gap
  in the tests or a change that cannot alter behavior. Deciding which
  is the work, and a gap gets a test pinned by value.
- GUI behavior is pinned by a named headless test, not a script.
  `tests/conftest.py` provides `window` (a `MainWindow` with an
  offscreen 3-D view), `pump` (runs the queued events), `fixture_path`,
  `project` and `survey`. Claims about the GUI are measured, never
  reasoned about: the 3-D view through `plotter.screenshot()`, the rest
  through `QWidget.grab()` or by reading the widgets.
- Fixtures are small and chosen on purpose. `testdata/` holds the plate
  (its geometry, mode shapes and FRFs, and controller runs recorded on
  it) and a few small frozen files, at sizes the suite can afford; the
  largest file is about 19 MB.
- An intermittent failure is a bug until proved otherwise. Record the
  assertion's values before forming a theory. The one failure here
  that was dismissed as a flake was a real latch both times.
- Own a bad test. When a failure was caused by the test and not the
  code, say so plainly.

Coverage is measured by CI on every pull request and held to a floor
of 90 percent. A change that adds code without tests fails CI the
same way a license violation does. The number is not the goal (the
falsified, behavior-named test is), but the floor keeps the number from
slipping without anyone noticing.

## The ratchets

These tests hold a rule instead of a feature. They run with the rest of
the suite, and they are why the rules above are more than preferences:

| Test | What it holds |
|---|---|
| `test_license_boundary.py` | The package and the tests import no code whose license would restrict the project's license, and no Qt module offered only under the GPL can load; the detector finds a planted violation. |
| `test_principles.py` | The thirteen principles are numbered, each is argued, the count is the same everywhere it is stated, and they can be reached from every audience; no dependency is capped or pinned. |
| `test_docstrings.py` | The first two docstring standards above. |
| `test_docstring_references.py` | The third: theory has its citations, in the form this site renders. |
| `test_verbs.py` | `Project.verbs` is the one applicability table the bar reads. |
| `test_headless.py` | Every view has a headless call. |
| `test_journal.py`, `test_workflow_journals.py` | The journal replays, for every project type's workflow. |
| `test_declared_dependencies.py` | What the modules import, what `pyproject.toml` declares and what the frozen build carries all agree. |
| `test_window_lifetime.py` | A closed window is destroyed. A leak here once grew CI's workers to gigabytes. |
| `test_public_tree.py` | What is published names no affiliation and no held-back material. |

## Where to go next

[Contributing](https://github.com/visualdynamics/visualdynamics/blob/main/CONTRIBUTING.md)
says how a change reaches the project. `AGENTS.md` at the repository
root is the same set of rules in the form an AI assistant reads first.
The [guide](guide/README.md) says what each part of the application
does for the person using it.
