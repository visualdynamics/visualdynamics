---
title: About
---

# Visual Dynamics

A units-aware structural dynamics toolset. It imports test and model
data, plots it, identifies modal parameters, correlates a test with a
finite element model, builds finite element models of its own, and
writes the report. All of it can be done from the desktop app or from
Python, and both use the same objects and the same verbs.

## Where to start

- [Principles](principles.md): what the toolset is meant to be, and
  the standard every part of it is held to. Read this first if you want
  to know whether it fits the way you work.
- [Getting started](getting-started.md): install it, open the app, and
  run the same workflow as a script.
- [Guide](guide/README.md): how the parts fit together, including
  [projects and links](guide/projects.md),
  [which geometry an object answers to](compatibility.md),
  [units](guide/units.md),
  [plotting](guide/plotting.md),
  [the modal workflow](guide/workflows/modal-workflow.md),
  [the finite element workflow](guide/workflows/fem-workflow.md) and
  [reports](guide/reports.md).
- [Architecture and conventions](architecture.md): for someone about
  to change the code. It covers the package by layer, the rule that the
  app and a script share one workflow, the docstring standard and the
  test conventions.
- API reference: every module, class, attribute and function in the
  package, generated from the code each time the site is built. It is
  in the site's navigation rather than in the repository; run
  `properdocs serve` and it is the *API reference* section.

## Projects

A **project** holds named objects (a geometry, some time data, FRFs,
mode shapes) and the structure around them: which objects are
**linked** together, which linked group is the **Basis** of
comparisons, and what kind of test it is. The tree in the desktop app
is a project, and so is `visualdynamics.Project` in a script. Both save
to the same `.vdyn` file, and each opens the other's.

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

Each line of the script is the verb behind a button in the app; none of
it is a separate interface written for scripts. What each object holds,
and what can be done with each kind, is on one page:
[The objects](guide/objects.md).

## How the documentation is written

The guides are written by hand and explain why things work the way
they do. The API reference is not written separately: `docs/gen_api.py`
walks the package while the site builds, and mkdocstrings renders the
docstrings. A new module gets a new page and a new method appears with
its docstring, so the reference always matches the code.

That makes the docstrings the place where the reference is kept.
`properdocs build --strict` fails on a broken cross-reference, and
`tests/test_docs.py` checks that the generator reaches every module in
the package.

The documentation is part of visualdynamics.org: `properdocs build`
writes it into the website's own tree, and the release workflow
deploys the two together, so it is at
[visualdynamics.org/documentation](https://visualdynamics.org/documentation/).
To work on it from a clone:

    pip install -e '.[docs]'
    properdocs serve        # localhost:8000, live reload
