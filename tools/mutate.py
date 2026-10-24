"""Mutation testing for one module: which small changes do its tests miss?

    ./.venv/bin/python tools/mutate.py visualdynamics/core/srs.py tests/test_srs.py
    ./.venv/bin/python tools/mutate.py MODULE TEST [TEST ...] [--workers 4] [--lines 70-120]

Run it with the repository's venv interpreter (`.venv/Scripts/python.exe`
on Windows); the tests run under the same one.

Each mutant is one change to the module's syntax tree — a comparison
flipped (`<` to `<=`, `==` to `!=`, `in` to `not in`), an arithmetic
operator swapped (`+` and `-`, `*` and `/`, `**` to `*`), `and` and
`or` exchanged, a `not` dropped, or an integer constant moved by one —
and the named tests are run against it. A mutant the tests fail on is
*killed*; one they pass is a *survivor*, printed with its line, and
every survivor is either a gap in the tests or a change that cannot
alter behavior (a cache size, `>` against `>=` where equality never
happens). Telling which is the reading of the code; the tool only
finds where to read.

It is AGENTS.md's "falsify every fix" run over a whole module rather
than one line (2026-10-10, the first pass: srs.py went from 51 of 58
mutants killed to 58, and found its coefficients checked only through
a DC gain that is exact whatever the damped frequency is).

Each worker runs in its own copy of `src/`, put first on `PYTHONPATH`,
so the mutants never touch the tree and four run side by side; the
baseline is run first and must be green, and the copy must be what is
imported, or nothing is reported. The tests run with `-n 0`: the
project's `-n auto` would start a dozen workers per mutant. Pick tests
that run in seconds — the module's own file and the few that exercise
it — since every mutant runs them all.
"""

from __future__ import annotations

import argparse
import ast
import copy
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import tempfile
import threading

REPO = pathlib.Path(__file__).resolve().parent.parent
#: the interpreter this runs under, which is the one whose packages the
#: tests need; the repository's venv path was written in at first, and
#: on Windows a venv keeps it under Scripts/, so the tool did not run
#: there (2026-10-24, a contributor on Windows)
PYTHON = pathlib.Path(sys.executable)

SWAP_COMPARE = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
                ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
                ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn,
                ast.NotIn: ast.In}
SWAP_ARITHMETIC = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div,
                   ast.Div: ast.Mult, ast.FloorDiv: ast.Div,
                   ast.Pow: ast.Mult, ast.Mod: ast.FloorDiv}


def sites(tree: ast.AST) -> list[tuple[int, tuple, int, str]]:
    """Every place one mutation can be made: (node's position in
    `ast.walk` order, how, line, description)."""
    found = []
    for k, node in enumerate(ast.walk(tree)):
        line = getattr(node, 'lineno', 0)
        if isinstance(node, ast.Compare):
            for j, op in enumerate(node.ops):
                if type(op) in SWAP_COMPARE:
                    swapped = SWAP_COMPARE[type(op)].__name__
                    found.append((k, ('compare', j), line,
                                  f'{type(op).__name__} -> {swapped}'))
        elif isinstance(node, ast.BinOp) and type(node.op) in SWAP_ARITHMETIC:
            swapped = SWAP_ARITHMETIC[type(node.op)].__name__
            found.append((k, ('arithmetic',), line,
                          f'{type(node.op).__name__} -> {swapped}'))
        elif isinstance(node, ast.BoolOp):
            found.append((k, ('boolean',), line,
                          f'{type(node.op).__name__} swapped'))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            found.append((k, ('not',), line, 'not dropped'))
        elif (isinstance(node, ast.Constant) and type(node.value) is int):
            found.append((k, ('integer',), line,
                          f'{node.value} -> {node.value + 1}'))
    return found


def mutant(tree: ast.AST, position: int, how: tuple) -> str:
    """The module's source with the one change made."""
    tree = copy.deepcopy(tree)
    nodes = list(ast.walk(tree))
    node = nodes[position]
    if how[0] == 'compare':
        node.ops[how[1]] = SWAP_COMPARE[type(node.ops[how[1]])]()
    elif how[0] == 'arithmetic':
        node.op = SWAP_ARITHMETIC[type(node.op)]()
    elif how[0] == 'boolean':
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif how[0] == 'not':
        for parent in nodes:
            for field, value in ast.iter_fields(parent):
                if value is node:
                    setattr(parent, field, node.operand)
                elif isinstance(value, list) and any(v is node for v in value):
                    value[[v is node for v in value].index(True)] = node.operand
    elif how[0] == 'integer':
        node.value += 1
    return ast.unparse(ast.fix_missing_locations(tree))


#: the repository's directories a test may find beside the package, by
#: a path worked out from where the package is: linked into each copy,
#: since a test reading testdata/ from the copy's own parent found
#: nothing there (2026-10-11, the report-size tests)
BESIDE = ('testdata', 'examples')


def workspace(work: pathlib.Path) -> pathlib.Path:
    """A private copy of `src/` in `work`, the data beside it linked."""
    shutil.copytree(REPO / 'src', work / 'src',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in BESIDE:
        if (REPO / name).exists():
            try:
                (work / name).symlink_to(REPO / name,
                                         target_is_directory=True)
            except OSError:
                # Windows makes a symlink only in developer mode; a
                # copy is slower to make but reads the same
                shutil.copytree(REPO / name, work / name)
    return work


def passes(work: pathlib.Path, tests: list[str], timeout: float) -> bool:
    """Do the tests pass against the tree in `work`?"""
    environment = dict(os.environ, PYTHONPATH=str(work / 'src'))
    try:
        run = subprocess.run(
            [str(PYTHON), '-m', 'pytest', '-q', '-x', '-n', '0',
             '-p', 'no:randomly', '-p', 'no:cacheprovider', *tests],
            cwd=REPO, env=environment, capture_output=True, timeout=timeout,
            check=False)
    except subprocess.TimeoutExpired:
        return False            # a mutant that hangs the tests is caught
    return run.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('module', help='path under src/, e.g. '
                        'visualdynamics/core/srs.py')
    parser.add_argument('tests', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lines', help='only mutate these lines, A-B')
    parser.add_argument('--timeout', type=float, default=300.0)
    args = parser.parse_args(argv)

    source = (REPO / 'src' / args.module).read_text(encoding='utf-8')
    tree = ast.parse(source)
    found = sites(tree)
    if args.lines:
        low, high = (int(v) for v in args.lines.split('-'))
        found = [site for site in found if low <= site[2] <= high]
    print(f'{args.module}: {len(found)} mutants against {args.tests}',
          flush=True)

    copies = [workspace(pathlib.Path(tempfile.mkdtemp(prefix=f'mutate{w}-')))
              for w in range(args.workers)]
    try:
        imported = subprocess.run(
            [str(PYTHON), '-c',
             'import visualdynamics; print(visualdynamics.__file__)'],
            env=dict(os.environ, PYTHONPATH=str(copies[0] / 'src')),
            capture_output=True, text=True, check=False).stdout
        if str(copies[0]) not in imported:
            print(f'the copy is not what is imported ({imported.strip()})')
            return 2
        if not passes(copies[0], args.tests, args.timeout):
            print('the tests fail before anything is changed')
            return 2

        work_queue: queue.Queue = queue.Queue()
        for site in found:
            work_queue.put(site)
        survivors, killed = [], [0]
        lock = threading.Lock()

        def worker(work: pathlib.Path) -> None:
            target = work / 'src' / args.module
            while True:
                try:
                    position, how, line, description = work_queue.get_nowait()
                except queue.Empty:
                    return
                target.write_text(mutant(tree, position, how),
                                  encoding='utf-8')
                lived = passes(work, args.tests, args.timeout)
                target.write_text(source, encoding='utf-8')
                with lock:
                    if lived:
                        survivors.append((line, description))
                        print(f'survived  {line}: {description}', flush=True)
                    else:
                        killed[0] += 1

        threads = [threading.Thread(target=worker, args=(work,))
                   for work in copies]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for work in copies:
            shutil.rmtree(work, ignore_errors=True)
    print(f'{args.module}: {killed[0]} killed, {len(survivors)} survived '
          f'of {len(found)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
