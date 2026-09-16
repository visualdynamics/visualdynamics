---
title: About
---

# Visual Dynamics

A units-aware structural dynamics toolset: import test and model data,
look at it, identify modal parameters, correlate a test against a
finite element model, and write the report — from a desktop app or
from Python, with the same objects and the same verbs behind both.

## Where to start

- **[Principles](principles.md)** — what the toolset is trying to be,
  and the standard every part of it is held to. Start here if you want
  to know whether it fits how you work.
- **[Getting started](getting-started.md)** — install, open the app,
  run the same workflow as a script.
- **[Guide](guide/README.md)** — how the pieces fit:
  [projects and links](guide/projects.md),
  [which geometry an object answers to](compatibility.md),
  [units](guide/units.md),
  [plotting](guide/plotting.md),
  [the modal workflow](guide/workflows/modal-workflow.md),
  [reports](guide/reports.md).
- **[Architecture and conventions](architecture.md)** — for someone
  about to change the code: the package by layer, the one-workflow
  rule, the docstring standard, the test conventions.
- **API reference** — every module, class, attribute and function in the
  package, generated from the code when the site is built, so it is never
  out of date. It is in the site's navigation rather than in the
  repository: run `properdocs serve` and it is the *API reference* section.

## The one idea worth knowing first

A **project** holds named objects — a geometry, some time data, FRFs,
mode shapes — and the structure around them: which objects are
**linked** together, which linked group is the **Basis** of
comparisons, what kind of test this is. The desktop app's tree *is* a
project, and so is `visualdynamics.Project` in a script; both save to the same
`.vdyn` file, and each opens the other's.

```python
import visualdynamics

project = visualdynamics.Project('Plate Modal Survey')
project.import_file('modal_spectra.nc4')
project.set_basis(*project.names)
project.compute_psds('Time History')
project.fit_modes(project.basis.frf, bounds=(300.0, 1300.0), limit=5)
print(project)
```

```
Project: Plate Modal Survey  [Modal Test]
  Basis
    Channel Table        ChannelTable  13 channels
    Time History         TimeHistory   260 records, 2048 samples
    Time History PSDs    Psd           13 records, 1025 samples
    FRF                  Frf           22 records, 1025 samples
    Multiple Coherence   MultipleCoherence 11 records, 1025 samples
    FRF Modes            ShapeSet      5 modes, 439.2-1142 Hz
```

Nothing above is a shortcut written for scripts: every line is the
verb behind a button in the app. What each object holds, and what can
be done with each kind, is one page: [The objects](guide/objects.md).

## Documentation policy

Guides are written by hand and explain *why*. The API reference is not
written down at all: `docs/gen_api.py` walks the package while the site
builds and mkdocstrings renders the docstrings, so the reference *is* the
code — a new module is a new page, a new method documents itself, and
there is nothing that can drift because there is nothing to keep in step.

That puts the weight on the docstrings, which is where it belongs.
`properdocs build --strict` fails on a broken cross-reference, and
`tests/test_docs.py` holds the generator to the whole package rather than
to a list of it.

The site is a page of visualdynamics.org: `properdocs build` writes it
into the website's own tree, and the release workflow deploys the two
together, so it answers at
[visualdynamics.org/documentation](https://visualdynamics.org/documentation/).
To work on it from a clone:

    pip install -e '.[docs]'
    properdocs serve        # localhost:8000, live reload
