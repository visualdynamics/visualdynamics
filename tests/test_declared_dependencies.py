"""What the package imports, it must declare — and must ship.

Three lists have to agree and had quietly stopped agreeing: what the
modules under `visualdynamics/` actually import, what `pyproject.toml`
declares as dependencies, and what `packaging/visualdynamics.spec`
leaves out of a frozen build.

scipy was in all the wrong halves of that (found 2026-08-27). It was
declared a *test* dependency and excluded from the build, while four
runtime paths called it — the low-pass, the integration, the filter
view's response curve and the sine matched filter. Every test passed,
because a developer's environment has scipy for the suite's own
cross-checks; the packaged application raised ImportError on Filter
Data, and nobody would have found that before whoever was handed the
build did.

That is the shape of failure this file exists for: a claim about the
*environment*, which the environment the tests run in happens to
satisfy. Neither check costs anything, and the second one — that
nothing imported is also excluded — is the one that was needed.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / 'src' / 'visualdynamics'
SPEC = ROOT / 'packaging' / 'visualdynamics.spec'

#: import names that are not the distribution's name. Only what is
#: actually imported here needs an entry.
DISTRIBUTION = {'PySide6': 'pyside6', 'netCDF4': 'netcdf4',
                'yaml': 'pyyaml', 'PIL': 'pillow',
                'dateutil': 'python-dateutil',
                # the `vtk` wheel is what puts `vtkmodules` on the path
                'vtkmodules': 'vtk',
                # the STEP kernel's bindings ride the cadquery-ocp wheel
                'OCP': 'cadquery-ocp'}


def third_party_imports():
    """{top-level module: [where it is imported]} across the package.

    Parsed rather than grepped, and at any depth, because this codebase
    imports lazily inside functions all through — which is exactly how
    scipy stayed invisible to a reading of the file heads.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif (isinstance(node, ast.ImportFrom)
                    and node.level == 0 and node.module):
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split('.')[0]
                if top in sys.stdlib_module_names or top == 'visualdynamics':
                    continue
                found.setdefault(top, []).append(
                    f'{path.relative_to(ROOT)}:{node.lineno}')
    return found


def declared():
    """The distribution names pyproject declares — the runtime
    `dependencies`, plus the `step` extra: its import lives behind a
    graceful refusal that names the extra, which is the one shape an
    optional import is allowed to take here. The dev/docs extras stay
    out — hiding a runtime import in them would be the bug this test
    exists to catch."""
    import tomllib

    with open(ROOT / 'pyproject.toml', 'rb') as handle:
        data = tomllib.load(handle)
    names = list(data['project']['dependencies'])
    names += data['project']['optional-dependencies'].get('step', [])
    return {name.split('[')[0].split('>')[0].split('=')[0].strip().lower()
            for name in names}


def excluded():
    """The module names the frozen build leaves out.

    Read from the spec's source: it is not importable, because
    PyInstaller injects its globals.
    """
    tree = ast.parse(SPEC.read_text(encoding='utf-8'), filename=str(SPEC))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, 'id', '') == 'EXCLUDED'
                        for t in node.targets)):
            return {element.value for element in node.value.elts
                    if isinstance(element, ast.Constant)}
    raise AssertionError('the spec no longer has an EXCLUDED list')


def test_everything_imported_is_declared():
    """A package imported but undeclared works on the machine that
    happens to have it and nowhere else."""
    undeclared = {
        module: where for module, where in third_party_imports().items()
        if DISTRIBUTION.get(module, module).lower() not in declared()}
    assert not undeclared, (
        'imported but not in pyproject dependencies:\n  '
        + '\n  '.join(f'{module} ({where[0]}'
                      + (f' and {len(where) - 1} more' if len(where) > 1
                         else '') + ')'
                      for module, where in sorted(undeclared.items())))


def test_nothing_imported_is_excluded_from_the_build():
    """The half that was needed, and the half a passing suite cannot
    otherwise notice: excluding something the package calls builds an
    application that fails where no test runs."""
    trespasses = {
        module: where for module, where in third_party_imports().items()
        if module in excluded()}
    assert not trespasses, (
        'these are imported at runtime and left out of the frozen '
        'build, so the packaged application raises ImportError where a '
        'developer machine does not:\n  '
        + '\n  '.join(f'{module} ({", ".join(where[:3])})'
                      for module, where in sorted(trespasses.items())))


def test_scipy_specifically():
    """Named, because it is the one that was wrong and the one most
    likely to be re-excluded for its size — a scipy wheel is the
    largest thing in the bundle after Qt and VTK."""
    assert 'scipy' in declared(), 'scipy is called at runtime'
    assert 'scipy' not in excluded(), 'and so must be packaged'


@pytest.mark.parametrize('module', ['scipy', 'numpy', 'pint'])
def test_the_check_can_see_the_package_use_these(module):
    """A scan for something has to be shown capable of finding it, or
    the tests above pass by being blind."""
    assert module in third_party_imports(), (
        f'{module} is imported somewhere under the package, so the '
        'walk should have found it')


def test_the_packages_own_name_is_the_command():
    """One window is the whole program, so `visualdynamics` launches it
    (Brandon, 2026-09-18); `visualdynamics-gui`, the first alphas'
    spelling, stays as an alias of the same entry point."""
    import tomllib
    from importlib.metadata import entry_points

    scripts = tomllib.loads(pathlib.Path('pyproject.toml').read_text(
        encoding='utf-8'))['project']['scripts']
    assert scripts['visualdynamics'] == 'visualdynamics.gui:main'
    assert scripts['visualdynamics-gui'] == 'visualdynamics.gui:main'
    installed = {e.name: e.value for e in entry_points(group='console_scripts')
                 if e.name.startswith('visualdynamics')}
    assert installed.get('visualdynamics') == 'visualdynamics.gui:main', \
        'reinstall the package (pip install -e .) to grow the script'

