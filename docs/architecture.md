---
title: Architecture and conventions
---

# Architecture and conventions

How the package is put together, and the conventions a change is
expected to keep — written for someone (or something) about to work
on the code rather than use it. The [principles](principles.md) are
the standard; this page is the map. Most of it is enforced by a test,
and the last section says which.

## The package, by layer

Half the codebase imports no Qt, and that is a rule rather than an
accident: the data model, the math, the file formats and the 3-D
scene are a library; the desktop app sits on it. A change that puts
Qt into the wrong layer is refused in review whatever else it does.

| Layer | Modules | Qt? | What lives there |
|---|---|---|---|
| Data model | `core/` | no | The objects a test holds — `Geometry`, `TimeHistory`, `Spectrum`, `Frf`, `Psd`, `Srs`, `ShapeSet`, `ChannelTable`, the specifications — and every computation on them: spectral estimates, FRFs, shock response, kurtosis, the wavelet, filters, modal fitting, rigid-body modes, transforms, reports as data. |
| Units | `units.py`, `core/unit_choices.py` | no | Every quantity stores SI and displays in the chosen system; objects import unit-less until told. |
| Project | `project.py` | no | The `Project`: named objects, links, roles, the **verbs** (below) and the **journal** — the one workflow the app and a script both drive. |
| Formats | `io/` | no | One module per format, each a reader and (nearly always) a writer, registered in a small registry so a new format touches nothing else. `native.py` is `.vdyn`. |
| Geometry motion | `deform.py`, `rotate.py`, `decimate.py` | no | Data at DOFs turned into node motion; a coordinate system dragged by its ring; peak-keeping thinning. |
| 3-D | `viz/` | no | PyVista over VTK: the scene, the waterfall, the scalogram surface, the MAC bars, picking, animation — built on a fixed stage and relabeled with real ranges. |
| 2-D | `plot/` | yes | pyqtgraph: one renderer serves the window and a script, so a figure in a file is the figure on screen. |
| App | `gui/` | yes | The window, the tree, the panes, the settings panels, the console, the report editor. Displays what the project does; decides nothing on its own. |
| Reports | `report/` | no | A `Report` rendered to one self-contained HTML file, viewers included. |
| Theme | `theme.py` | yes | The colors the package draws with, and which scheme applies. |

## One workflow, two front ends

Everything the app can do, a script can do, with the same call
(principle 4), because the app *is* a script: the desktop window and
`visualdynamics.Project` drive one workflow, and every act in the
window is a verb on the project.

- **Behavior goes on the project, or in `core`; the GUI displays
  it.** A new computation is a project verb (`compute_srs`,
  `filter_data`, `transform`…) with its math in a `core` module. The
  window's bar and menus are a *presentation* of `Project.verbs()` —
  the one applicability table, whose summaries are the first line of
  each verb's docstring, so the bar, the API and the reference cannot
  disagree. A verb the window offers and the API lacks is a bug, and
  `tests/test_verbs.py` finds it.
- **The journal.** Every verb the project runs — clicked or called —
  is recorded as runnable Python in `project.journal`; the console tab
  shows it, and `tests/test_journal.py` replays it and compares
  projects. Nothing may be un-replayable by design.
- **Headless is the promise.** Every view the window can put up has a
  call that renders it to a file with no window and no event loop
  (`tests/test_headless.py` is the inventory). A view that grows a
  GUI-only path fails that test.
- **Processing is select, see, set, apply** (principle 13). A new
  computation ships with all four: a reading of the object it acts on,
  a settings panel (built from `gui/settings_panel.py`, beside the
  view, with its derived readings), a live preview drawn over the
  data, and an apply button that runs the verb. Look at the filter,
  the truncation or the wavelet for the pattern.
- **Staleness is provenance.** A derived object records the settings
  it was made with; when its source's settings move it shows a
  refresh badge with the reason. Recomputing is an act, never
  automatic.

## Docstrings

The docstrings **are** the API reference: the site is generated from
them at build time, and there is nothing else to keep in step. Two
standards, both enforced by `tests/test_docstrings.py`:

- **Every public callable has a docstring.** That is the floor.
- **The scripting surface** — what `visualdynamics.__all__` exports and
  the public methods on those classes — also carries NumPy
  `Parameters` and `Returns` sections, because that is where somebody
  arrives without context and a table earns its place.

Everywhere else, prose. The house style is that a docstring or comment
explains *why* — especially why the obvious approach was tried and
rejected, dated, and with the name of whoever decided. Several bugs
here were caused by code that looked right, and the comment that
says "this looks wrong, here is the measurement" is what stops the
next person putting it back. Match the surrounding density. A
representative one, from `core/wavelet.py`:

```python
def decade_values(low: float, high: float) -> list[float]:
    """The round frequencies a person would label a log axis with.

    1, 2, 5 per decade, every one inside ``[low, high]``, ascending.
    The transform's rows are twelfths of an octave and nobody reads
    158.7 Hz off an axis, so the flat picture, the 3-D stage and the
    report figure all label these values instead — and they call this
    one function rather than each carrying the loop, which is how the
    three drifted apart once (2026-09-16: the three loops were
    byte-for-byte the same, and the fourth would not have been).
    """
```

What it does, then why it is here rather than somewhere else, then
the history that justifies it. No restating of the signature.

## Tests

The suite is the specification. About 3 300 tests run headless in
three minutes on four workers, most of them through a real
`MainWindow` built off screen. The conventions:

- **A test is named as a sentence about behavior**, read from the
  user's side: `test_a_time_history_offers_the_wavelet`,
  `test_the_flat_picture_is_a_toggle_away`,
  `test_a_record_that_starts_late_is_drawn_on_its_own_clock`. The name
  is the claim; the body pins it with a value (a label, a unit, a
  count), never merely "something was built".
- **A change ships with its test in the same commit.** A feature
  without a test is not finished; a bug fix without a test that fails
  on the old code is not a fix (principle 11).
- **Every fix is falsified.** Break the change deliberately and watch
  the named test fail; a test that passes with the fix reverted is
  testing nothing. Clear `__pycache__` between the break and the
  restore, or the interpreter keeps running the broken bytecode.
- **GUI behavior is pinned by a named headless test, not a script.**
  `tests/conftest.py` gives `window` (a `MainWindow` with an offscreen
  3-D view), `pump` (drain the event loop), `fixture_path`, `project`
  and `survey`. Claims about the GUI are measured — the 3-D view
  through `plotter.screenshot()`, the rest through `QWidget.grab()`
  or by reading the widgets — never reasoned about.
- **Fixtures are small and deliberate.** `testdata/` holds the plate
  survey and the drone run at sizes the suite can afford; nothing
  large is committed.
- **An intermittent failure is a bug until proved otherwise.**
  Capture the assertion's values before theorizing; the one dismissed
  as a flake was a real latch both times.
- **Own a bad test.** When a failure was the test and not the code,
  say so plainly.

Coverage is measured on every pull request and held to a floor by
CI; a change that adds code without tests goes red the way a license
violation does. The number is not the goal — the falsified,
behavior-named test is — but the floor is what stops the number
sliding while nobody is looking.

## The ratchets

Tests that hold a rule rather than a feature. They run with the rest
and are why the rules above are not merely preferences:

| Test | What it holds |
|---|---|
| `test_license_boundary.py` | The package, and the tests, import no code whose license would restrict the project's licensing; the detector finds a planted violation. |
| `test_principles.py` | The thirteen principles are numbered, stated, counted the same everywhere, reachable from every audience; no dependency is capped or pinned. |
| `test_docstrings.py` | The two docstring standards above. |
| `test_verbs.py` | `Project.verbs` is the one applicability table the bar reads. |
| `test_headless.py` | Every view has a headless call. |
| `test_journal.py`, `test_workflow_journals.py` | The journal replays, for every project type's workflow. |
| `test_declared_dependencies.py` | What the modules import, what `pyproject.toml` declares and what the frozen build carries agree. |
| `test_window_lifetime.py` | A closed window dies; a leak here grew CI's workers to gigabytes once. |
| `test_public_tree.py` | What is published names no affiliation and no held-back material. |

## Where to go next

[Contributing](https://github.com/visualdynamics/visualdynamics/blob/main/CONTRIBUTING.md)
says how a change reaches the project; `AGENTS.md` at the repository root is the same set of rules
in the form an AI assistant reads first; the
[guide](guide/README.md) says what each part of the application does
for the person using it.
