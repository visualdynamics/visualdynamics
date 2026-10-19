#!/usr/bin/env python
"""The pictures and the numbers of the three meshing shapes
(`visualdynamics.demo.shapes`), for the guide's page on them.

    ./.venv/bin/python tools/make_shape_pictures.py            # all three
    ./.venv/bin/python tools/make_shape_pictures.py v_block    # one

Each shape is meshed with bricks and with tetrahedra, each at the size
Detect Element Size gives for it, the
meshes are rendered by the toolset's own scene (`Geometry.plot`, off
screen) into docs/guide/images/shapes-<shape>-<elements>.png, and the
table the page carries is printed, one row per mesh: the count, the
worst element, how many under 0.3, the volume against the recipe's,
and the seconds. Paste the table into docs/guide/workflows/mesh-shapes.md
when the meshers change.
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from visualdynamics.demo import shapes

OUT = ROOT / 'docs' / 'guide' / 'images'


def main(keys: list[str]) -> None:
    rows = []
    for key in keys:
        for elements in ('hexes', 'tetras'):
            project = shapes.SHAPES[key]()
            got = shapes.mesh(project, elements)
            rows.append((shapes.TITLES[key], got['size'], elements, got))
            if got['method'] == 'refused':
                print(f'{key} {elements}: {got["refusal"]}', file=sys.stderr)
                continue
            geometry = project[project.names[0]]
            path = OUT / f'shapes-{key.replace("_", "-")}-{elements}.png'
            geometry.plot(screenshot=str(path), components=['elements'])
            print(f'{path.name}: {got["count"]:,} {elements}, worst {got["quality"]:.2f}, '
                  f'{got["seconds"]:.0f} s', file=sys.stderr)
    print('| Shape | Size (in) | Elements | Count | Worst | Under 0.3 | Volume (in³) | Recipe (in³) | Error | Seconds |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for title, size, elements, got in rows:
        if got['method'] == 'refused':
            print(f'| {title} | {size:.3g} | {elements} | refused: '
                  f'{got["refusal"].split(" fitting")[0]} | | | | '
                  f'{got["recipe_volume"]:.2f} | | {got["seconds"]:.0f} |')
            continue
        print(f'| {title} | {size:.3g} | {elements} | {got["count"]:,} | {got["quality"]:.2f} | '
              f'{got["poor"]:,} | {got["volume"]:.2f} | {got["recipe_volume"]:.2f} | '
              f'{got["volume_error"] * 100:+.2f} % | {got["seconds"]:.0f} |')


if __name__ == '__main__':
    main(sys.argv[1:] or list(shapes.SHAPES))
