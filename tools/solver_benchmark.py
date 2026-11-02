"""Time the sparse modal solve on blocks of bricks, PARDISO against SuperLU.

    python tools/solver_benchmark.py [--modes 20] [--superlu-up-to 32000]
        [--timeout 2700] 40x40x20 60x40x40 80x60x40

Each block is nx x ny x nz aluminum bricks, free, solved for its lowest
`--modes` modes through `Model.eigensolution(solver='sparse')`: with the
machine's own factorization (PARDISO on x86-64 Windows and Linux,
Accelerate on a Mac), and with SuperLU for the blocks of up to
`--superlu-up-to` bricks, which is what the packaged builds used before
oneMKL rode along. Each case runs in its own process, so each has a
fresh library and its own memory, and one that runs past `--timeout`
seconds is stopped and reported as such. The table goes to standard
output, and to the job summary when run by GitHub Actions.

The case it was written for: a 193,627-brick model, 20 modes, took 70
minutes on a Windows machine through SuperLU and two on a Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

CASE = r'''
import json, sys, time
from visualdynamics.core import fem, sparse_solve

nx, ny, nz, modes = (int(value) for value in sys.argv[1:5])
started = time.perf_counter()
model = fem.Model('block')
aluminum = fem.material('6061-T6')

def node(i, j, k):
    return 1 + i + (nx + 1) * (j + (ny + 1) * k)

for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            model.add_node(node(i, j, k), 0.01 * i, 0.01 * j, 0.01 * k)
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            model.add_solid([node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k),
                             node(i, j + 1, k), node(i, j, k + 1), node(i + 1, j, k + 1),
                             node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)], aluminum)
built = time.perf_counter()
used = []
real = sparse_solve.factor_symmetric

def spy(matrix):
    factor = real(matrix)
    used.append(factor.backend)
    return factor

sparse_solve.factor_symmetric = spy
shapes = model.eigensolution(num_modes=modes, solver='sparse')
solved = time.perf_counter()
elastic = [float(f) for f in shapes.frequency if f > 0.0]
print(json.dumps({'dof': int(model.num_dof), 'backend': used[0] if used else '?',
                  'build_s': built - started, 'solve_s': solved - built,
                  'first_hz': elastic[0] if elastic else None,
                  'last_hz': elastic[-1] if elastic else None}))
'''


def run_case(dims: tuple[int, int, int], modes: int, superlu: bool,
             timeout: float) -> dict:
    environment = dict(os.environ)
    if superlu:
        environment['VISUALDYNAMICS_SPARSE_SOLVER'] = 'superlu'
    started = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, '-c', CASE, *map(str, dims), str(modes)],
                              capture_output=True, text=True, timeout=timeout,
                              env=environment, check=False)
    except subprocess.TimeoutExpired:
        return {'outcome': f'stopped after {timeout / 60:.0f} min'}
    if done.returncode != 0:
        tail = (done.stderr.strip().splitlines() or ['no output'])[-1]
        return {'outcome': f'failed: {tail[:120]}',
                'wall_s': time.perf_counter() - started}
    result = json.loads(done.stdout.strip().splitlines()[-1])
    result['outcome'] = 'solved'
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('blocks', nargs='+', help='nx x ny x nz, e.g. 40x40x20')
    parser.add_argument('--modes', type=int, default=20)
    parser.add_argument('--superlu-up-to', type=int, default=32000,
                        help='solve blocks of up to this many bricks by SuperLU as well')
    parser.add_argument('--timeout', type=float, default=2700.0,
                        help='seconds a case may run before it is stopped')
    arguments = parser.parse_args(argv)
    print(f'{platform.platform()}, {platform.machine()}, {os.cpu_count()} logical CPUs, '
          f'Python {platform.python_version()}', flush=True)
    rows = ['| Bricks | DOF | Solver | Build | Solve | First, last elastic mode |',
            '|---:|---:|---|---:|---:|---|']
    for text in arguments.blocks:
        dims = tuple(int(part) for part in text.lower().split('x'))
        bricks = dims[0] * dims[1] * dims[2]
        cases = [False] + ([True] if bricks <= arguments.superlu_up_to else [])
        for superlu in cases:
            result = run_case(dims, arguments.modes, superlu, arguments.timeout)
            solver = 'SuperLU' if superlu else result.get('backend', 'default')
            if result['outcome'] == 'solved':
                elastic = ('none: every mode asked for is rigid' if result['first_hz'] is None
                           else f"{result['first_hz']:.2f}, {result['last_hz']:.2f} Hz")
                row = (f"| {bricks:,} | {result['dof']:,} | {result['backend']} | "
                       f"{result['build_s']:.1f} s | {result['solve_s']:.1f} s | {elastic} |")
            else:
                row = f"| {bricks:,} | | {solver} | | {result['outcome']} | |"
            rows.append(row)
            print(row, flush=True)
    table = '\n'.join(rows)
    print('\n' + table)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as handle:
            handle.write(f'## Sparse modal solve, {arguments.modes} modes\n\n'
                         f'{platform.platform()}, {os.cpu_count()} logical CPUs\n\n'
                         f'{table}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
