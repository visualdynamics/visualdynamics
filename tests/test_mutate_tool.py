"""tools/mutate.py builds the mutants it says it builds.

The tool finds where the tests miss a change; if it quietly stopped
making a kind of change, every module would look better guarded than
it is. So each kind is made once here and read back."""

from __future__ import annotations

import ast
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _tool():
    spec = importlib.util.spec_from_file_location('mutate', ROOT / 'tools' / 'mutate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = 'def f(a, b):\n    return a < 2 and not b or a * b\n'


def test_every_kind_of_change_is_made_once_and_read_back():
    tool = _tool()
    tree = ast.parse(SOURCE)
    found = tool.sites(tree)
    kinds = sorted(how[0] for _k, how, _line, _what in found)
    assert kinds == ['arithmetic', 'boolean', 'boolean', 'compare',
                     'integer', 'not']
    made = {what: tool.mutant(tree, k, how) for k, how, _line, what in found}
    assert 'a <= 2' in made['Lt -> LtE']
    assert 'a < 3' in made['2 -> 3']
    assert 'a / b' in made['Mult -> Div']
    assert 'not b' not in made['not dropped'] and ' b or' in made['not dropped']
    assert any('a < 2 or not b' in text for what, text in made.items()
               if what == 'And swapped')
    # the tree itself is never changed: every mutant starts from it
    assert ast.unparse(tree) == ast.unparse(ast.parse(SOURCE))


def test_a_boolean_constant_is_not_an_integer_to_move():
    tool = _tool()
    found = tool.sites(ast.parse('x = True\ny = 3\n'))
    assert [what for _k, _how, _line, what in found] == ['3 -> 4']


def test_a_copy_finds_the_repositorys_data_beside_it(tmp_path):
    """A test may find testdata/ by where the package is; in the copy,
    that is beside the copy's src/."""
    tool = _tool()
    work = tool.workspace(tmp_path)
    assert (work / 'src' / 'visualdynamics' / '__init__.py').exists()
    assert (work / 'testdata').resolve() == ROOT / 'testdata'
    assert not list((work / 'src').rglob('__pycache__'))
