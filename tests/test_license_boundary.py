"""The line visualdynamics must not cross, enforced rather than remembered.

visualdynamics is owned outright, and its licensing is deliberately
still open: nothing has been granted, and every option — permissive,
copyleft, commercial, or nothing at all — is still available. sdynpy,
rattlesnake and forcefinder are GPL-3.0, and one import would settle
the question by accident and for ever.

**Copyleft inherited is not copyleft chosen.** Code taken from
somebody else's GPL project can never be relicensed, because it is not
ours to relicense — so the choice would not merely be made, it would
be made permanently, by whoever wrote the import rather than by
anyone deciding. Permissive code may travel into a copyleft work; the
reverse never holds.

The rule has always been written down: use their *file formats*, which
are documented interfaces and not copyrightable expression, and never
their code. What it has not been is checked. This checks it.

There used to be a deliberate exception. Three generators under
`testdata/` drove the real controller and the real modal library to
produce the fixtures the suite reads, and the argument for them was
that they are not distributed — true, and checked, but an argument
nonetheless, and one that would have had to be made again in public
with a license note beside it. They live in the
`visualdynamics-generators` repository now. The exception is retired
rather than defended: no file in this tree combines with a GPL-3
package at runtime.

The packaging check stays regardless. It costs nothing, and it is the
half that is easy to break by accident — widening the packaged set is
one line in `pyproject.toml`.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

#: the packages whose license would reach into visualdynamics: the three
#: structural-dynamics libraries this code is likeliest to reach for, every
#: one of them GPL-3.0.
COPYLEFT = frozenset({'sdynpy', 'rattlesnake', 'forcefinder'})

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / 'src' / 'visualdynamics'
TESTS = ROOT / 'tests'


def imports_in(path):
    """[(line, module)] every module this file imports, at any depth.

    Parsed rather than grepped, for two reasons that both matter here.
    A grep for 'sdynpy' hits the dozens of comments and docstrings that
    *name* it — the file-format readers, the Qt binding notes — and
    would fail on a package that is perfectly clean. And visualdynamics imports
    lazily all through, inside functions and methods, where a grep for
    a line starting with `import` would miss every one of them.

    Relative imports have no module name to check and are skipped:
    `from ..core import data` is visualdynamics's own.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif (isinstance(node, ast.ImportFrom)
                and node.level == 0 and node.module):
            found.append((node.lineno, node.module))
    return found


def named_dynamically(path):
    """[(line, name)] modules imported by string — importlib, __import__.

    An import the AST cannot see as an import. Nothing in visualdynamics does
    this, and this is here so that nothing quietly starts to.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = (target.attr if isinstance(target, ast.Attribute)
                else target.id if isinstance(target, ast.Name) else '')
        if name not in ('import_module', '__import__'):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str):
                found.append((node.lineno, argument.value))
    return found


def package_files():
    # tests included, since 2026-08-28. The sdynpy oracle
    # (testdata/sdynpy_oracle, tests/test_sdynpy_oracle.py) compares
    # this package's spectral answers against sdynpy's, and the whole
    # design of that comparison is that sdynpy's *numbers* travel here
    # as data while sdynpy itself never does — the freezing script
    # lives in the private generators repository with the other GPL
    # tool-use. A test that imported sdynpy directly would be the easy
    # wrong way to write the next comparison, and this is what makes it
    # fail instead of merely being frowned at.
    return sorted(PACKAGE.rglob('*.py')) + sorted(TESTS.glob('*.py'))


def offending(pairs):
    """The (line, module) pairs naming a copyleft package."""
    return [(line, module) for line, module in pairs
            if module.split('.')[0] in COPYLEFT]


# ---- the rule -----------------------------------------------------------


def test_no_module_under_visualdynamics_imports_a_copyleft_package():
    """The whole point. The license stays a decision, not an inheritance."""
    trespasses = []
    for path in package_files():
        for line, module in offending(imports_in(path)):
            trespasses.append(f'{path.relative_to(ROOT)}:{line} imports '
                              f'{module}')
    assert not trespasses, (
        'these are GPL-3.0, and importing one carries its copyleft into '
        'everything distributed with visualdynamics — which would settle the '
        'license question permanently, in one direction. '
        'Use the file format, never the code:\n  '
        + '\n  '.join(trespasses))


def test_nothing_reaches_one_by_name_either():
    """importlib.import_module('sdynpy') is an import the AST does not
    read as one."""
    trespasses = []
    for path in package_files():
        for line, module in offending(named_dynamically(path)):
            trespasses.append(f'{path.relative_to(ROOT)}:{line} -> {module}')
    assert not trespasses, '\n  '.join(trespasses)


@pytest.mark.parametrize('module', sorted(COPYLEFT))
def test_the_check_would_catch_each_of_them(tmp_path, module):
    """A scan for something has to be shown capable of finding it, or
    the tests above pass by being blind.

    Synthetic samples rather than a real file: there is no longer a
    file in this repository that imports one of these, which is the
    point of the arrangement — so the proof of capability has to be
    manufactured. Every name in the list, in each of the three shapes
    the parser has to see through.
    """
    for source in (f'import {module}\n',
                   f'from {module} import something\n',
                   f'def later():\n    import {module}.submodule\n'):
        path = tmp_path / 'sample.py'
        path.write_text(source)
        assert offending(imports_in(path)), source


#: every directory holding this project's own Python. Named rather than
#: walked from the root, so a vendored copy or a build tree cannot
#: quietly widen what is being claimed.
OURS = ('src', 'testdata', 'tests', 'tools', 'examples')


def test_no_file_in_the_repository_imports_one():
    """The stronger claim, and now a true one: not the package alone —
    nothing here at all.

    The three generators that drove the real controller used to be the
    deliberate exception, kept honest by not being distributed. They
    live in their own repository now, which retires the exception
    rather than arguing it: there is no file in this tree that combines
    with a GPL-3 package at runtime, so there is none to explain, note
    or license differently when this is published.
    """
    trespasses = []
    for directory in OURS:
        # `rglob` on a directory that is not there yields nothing, so a
        # rename would leave this sweeping an empty set and passing
        # while checking nothing at all. Moving the package under src/
        # is exactly the rename that would have done it.
        sources = sorted((ROOT / directory).rglob('*.py'))
        assert sources, f'{directory}/ holds no Python — has it moved?'
        for path in sources:
            if '__pycache__' in path.parts:
                continue
            for line, module in offending(imports_in(path)):
                trespasses.append(f'{path.relative_to(ROOT)}:{line} '
                                  f'imports {module}')
    assert not trespasses, (
        'a copyleft import is back in the tree. If it is a generator '
        'that drives the real controller, it belongs in the '
        'visualdynamics-generators repository:\n  '
        + '\n  '.join(trespasses))


# ---- and the other half of the argument ---------------------------------


def test_the_generators_are_not_distributed():
    """They are the deliberate exception — they import the GPL packages
    to produce the fixtures — and the argument only holds while they
    are not shipped.

    Read from `pyproject.toml` rather than by asking setuptools to
    discover packages: the declaration is what a build honors, it is
    what a reviewer reads, and checking it needs nothing installed
    beyond the standard library.
    """
    import tomllib

    with open(ROOT / 'pyproject.toml', 'rb') as f:
        config = tomllib.load(f)
    found = config['tool']['setuptools']['packages']['find']
    assert found.get('include') == ['visualdynamics*'], (
        'the packaged set is no longer just visualdynamics, so a generator '
        f'could be shipped with it: {found}')
    # src layout narrows it a second way: discovery never looks outside
    # src/, so testdata/ and tools/ are not candidates to begin with
    assert found.get('where') == ['src'], (
        'discovery reaches outside src/, where the generators live: '
        f'{found}')
    assert 'exclude' not in found or 'visualdynamics' not in ' '.join(
        found['exclude']), found
    # and the package itself does not smuggle the fixtures in as data
    data = config['tool']['setuptools'].get('package-data', {})
    assert not any('testdata' in entry for entries in data.values()
                   for entry in entries), data


def test_the_readers_named_after_a_format_import_only_the_format():
    """`io/rattlesnake.py` and the `io/sdynpy_*` modules are named for
    the layouts they read, which is documented interface and not
    copyrightable expression. What they import is netCDF4 and numpy."""
    named = [path for path in package_files()
             if path.stem.split('_')[0] in COPYLEFT]
    assert named, 'the format readers should exist to be checked'
    for path in named:
        modules = {module.split('.')[0] for _line, module in imports_in(path)}
        assert not (modules & COPYLEFT), path.relative_to(ROOT)


def test_no_gpl_only_qt_module_can_load_into_the_process():
    """Qt Data Visualization is GPLv3-or-commercial, no LGPL — its own
    licensing page says so. Nothing here uses it, but qtpy imports it
    at startup for a Windows-Qt5 compatibility alias, which put a
    GPL-only module into the running app by accident of a dependency.
    The package root refuses the import before qtpy can make it; this
    holds that refusal, through the exact import that loaded it."""
    import importlib
    import sys

    # order is the point, so import_module rather than statements the
    # formatter would sort: the refusal has to be in place before the
    # dependency that made the import
    importlib.import_module('visualdynamics')
    importlib.import_module('pyvistaqt')

    gpl_only = ('PySide6.QtDataVisualization', 'PySide6.QtCharts')
    loaded = [name for name in gpl_only
              if isinstance(sys.modules.get(name), type(sys))]
    assert not loaded, (
        f'GPL-only Qt modules loaded into the process: {loaded} — '
        'the commercial half of the licensing plan cannot ship '
        'combined with these')
