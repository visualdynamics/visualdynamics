"""Three shapes that are not extrusions, for trying the meshers on.

A shaft with an oil hole drilled through it at an angle, a wye pipe
fitting, and a V-block with two banks of angled bores meeting a crank
bore. None can be swept along an axis, so Bricks meshes each on a grid
fitted to its shape (`core.hexmesh`) and Tets from its signed distance
(`core.tetmesh`). Each is built from the recipe alone, blocks and
cylinders and their booleans, in inches:

    from visualdynamics.demo import shapes

    project = shapes.v_block()
    found = shapes.mesh(project, 'hexes')        # or 'tetras'
    print(found['count'], found['quality'], found['volume_error'])

`report` meshes a shape both ways at its size and gives the numbers
the guide's table shows: the element count, the worst element (the
scaled Jacobian of a brick, the radius ratio of a tetrahedron, 1 for a
perfect element and 0 for a flat one), how many are under 0.3, the
mesh's volume against the recipe's, and the seconds taken.

Run it directly to print the three:

    python3 -m visualdynamics.demo.shapes
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import Project
from ..core import fem, hexmesh, solids, tetmesh

INCH = 0.0254

#: the element size each shape is meshed at for the guide, in inches:
#: three to five elements across the smallest feature
SIZES = {'shaft': 0.1, 'wye': 0.1, 'v_block': 0.2}
#: each shape's project, by its key
TITLES = {'shaft': 'Cross-drilled shaft', 'wye': 'Wye fitting', 'v_block': 'V-block'}


def shaft() -> Project:
    """A 2 in shaft, 6 in long along X, with a 0.6 in oil hole drilled
    through its middle at 30 degrees to its axis."""
    project = Project(TITLES['shaft'])
    name = project.new_geometry('Shaft', unit='in')
    body = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 6.0, angles=(0, 90, 0),
                                      unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.3, 5.0, angles=(0, 60, 0),
                                      unit='in')
    project.combine_solids(name, 'subtract', [body, hole], 'shaft')
    return project


def wye() -> Project:
    """A wye fitting: a 2 in run 6 in long along X with a 1.5 in branch
    leaving it at 45 degrees, both bored, the walls 0.3 in."""
    project = Project(TITLES['wye'])
    name = project.new_geometry('Wye', unit='in')
    run = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 6.0, angles=(0, 90, 0),
                                     unit='in')
    branch = project.add_solid_cylinder(name, (1.2, 0, 1.2), 0.75, 3.5, angles=(0, 45, 0),
                                        unit='in')
    outside = project.combine_solids(name, 'union', [run, branch])
    run_bore = project.add_solid_cylinder(name, (0, 0, 0), 0.7, 6.4, angles=(0, 90, 0),
                                          unit='in')
    branch_bore = project.add_solid_cylinder(name, (1.2, 0, 1.2), 0.45, 3.8,
                                             angles=(0, 45, 0), unit='in')
    bores = project.combine_solids(name, 'union', [run_bore, branch_bore])
    project.combine_solids(name, 'subtract', [outside, bores], 'wye')
    return project


def v_block() -> Project:
    """A 6 x 5 x 3 in block with a 1.2 in crank bore along X and two
    banks of three 1 in bores at 45 degrees from vertical, each bank's
    bores running from the crank bore out through the top face."""
    project = Project(TITLES['v_block'])
    name = project.new_geometry('V-block', unit='in')
    block = project.add_solid_block(name, (0, 0, 0), (6, 5, 3), unit='in')
    crank = project.add_solid_cylinder(name, (0, 0, -0.5), 0.6, 7.0, angles=(0, 90, 0),
                                       unit='in')
    bores = [crank]
    for x in (-2.0, 0.0, 2.0):
        for side in (-1.0, 1.0):
            bores.append(project.add_solid_cylinder(
                name, (x, side * 0.5, 0.5), 0.5, 4.0, angles=(-side * 45, 0, 0),
                unit='in'))
    holes = project.combine_solids(name, 'union', bores)
    project.combine_solids(name, 'subtract', [block, holes], 'V-block')
    return project


SHAPES = {'shaft': shaft, 'wye': wye, 'v_block': v_block}


def recipe_volume(geometry: Any, ids, samples: int = 2_000_000) -> float:
    """The parts' volume by sampling their recipe, in the geometry's
    length unit cubed: random points over the parts' extent, the share
    inside. Within about 0.1 percent at two million points."""
    ids = [int(i) for i in ids]
    low, high = tetmesh._extent(geometry, ids)
    at = np.random.default_rng(0).uniform(low, high, (int(samples), 3))
    inside = np.zeros(len(at), dtype=bool)
    for i in ids:
        inside |= solids.contains(geometry, i, at)
    return float(inside.mean() * np.prod(high - low))


def mesh_volume(geometry: Any) -> float:
    """The solid elements' volume, integrated as the solver integrates
    them: exact for bricks, and for tetrahedra straight or curved."""
    xyz = np.asarray(geometry.node_xyz)
    index = geometry.node_index
    total = 0.0
    by_count: dict[int, list] = {}
    for conn in geometry.elem_conn:
        by_count.setdefault(len(conn), []).append(xyz[index(list(conn))])
    for count, cells in by_count.items():
        if count in (4, 8, 10):
            total += float(np.abs(fem._solid_volumes(np.array(cells))).sum())
    return total


def mesh(project: Project, elements: str, size: float | None = None) -> dict:
    """Mesh the project's one geometry's parts into `elements`, 'hexes'
    or 'tetras', at `size` in inches (the shape's `SIZES` entry by
    default), and measure the result.

    Returns
    -------
    dict
        'count', the elements; 'quality', the worst element's scaled
        Jacobian (bricks) or radius ratio (tets); 'poor', how many are
        under 0.3; 'volume', the mesh's, and 'recipe_volume', the
        recipe's, in cubic inches; 'volume_error', the mesh's less the
        recipe's as a share of the recipe's; 'seconds'; and 'method',
        which mesher ran. A mesh refused (bricks that fold at the size)
        is recorded with 'method' 'refused', the reason in 'refusal'
        and a count of 0.
    """
    name = project.names[0]
    geometry = project[name]
    ids = [s.id for s in solids.top_level(geometry)]
    if size is None:
        size = SIZES[next(key for key, title in TITLES.items() if title == project.name)]
    started = time.perf_counter()
    try:
        found = project.mesh_solids(name, ids, size, unit='in', elements=elements)
    except ValueError as refusal:
        # recorded as it stands: the table says what was refused and why
        return {'count': 0, 'quality': float('nan'), 'poor': 0, 'volume': float('nan'),
                'recipe_volume': recipe_volume(geometry, ids) / INCH ** 3,
                'volume_error': float('nan'), 'seconds': time.perf_counter() - started,
                'method': 'refused', 'refusal': str(refusal)}
    seconds = time.perf_counter() - started
    recipe = recipe_volume(geometry, ids) / INCH ** 3
    volume = mesh_volume(geometry) / INCH ** 3
    quality = found['quality'] if elements == 'tetras' or found.get('method') == 'overlay' \
        else _brick_quality(geometry)
    poor = found['poor'] if 'poor' in found else int((_brick_qualities(geometry) < 0.3).sum())
    return {'count': found['tets'] if elements == 'tetras' else found['bricks'],
            'quality': float(quality), 'poor': int(poor), 'volume': volume,
            'recipe_volume': recipe, 'volume_error': (volume - recipe) / recipe,
            'seconds': seconds,
            'method': 'tetras' if elements == 'tetras' else found.get('method', 'extrusion')}


def _brick_qualities(geometry: Any) -> np.ndarray:
    xyz = np.asarray(geometry.node_xyz)
    conn = np.array([geometry.node_index(list(c)) for c in geometry.elem_conn
                     if len(c) == 8])
    return hexmesh._scaled_jacobians(xyz, conn).min(axis=1)


def _brick_quality(geometry: Any) -> float:
    return float(_brick_qualities(geometry).min())


def report(name: str, size: float | None = None) -> dict:
    """Both meshes of the shape `name` ('shaft', 'wye' or 'v_block') at
    `size` in inches (its `SIZES` entry by default): {'hexes': ...,
    'tetras': ...}, each as `mesh` gives it, each on a fresh build."""
    found = {}
    for elements in ('hexes', 'tetras'):
        project = SHAPES[name]()
        found[elements] = mesh(project, elements, size if size is not None else SIZES[name])
    return found


def _main() -> None:
    for name in SHAPES:
        found = report(name)
        print(f'{name} at {SIZES[name]} in:')
        for elements, got in found.items():
            print(f'  {elements:<7} {got["count"]:>8,} elements, worst {got["quality"]:.2f}, '
                  f'{got["poor"]:,} under 0.3, volume {got["volume"]:.2f} in^3 '
                  f'({got["volume_error"] * 100:+.2f} %), {got["seconds"]:.1f} s, '
                  f'{got["method"]}')


if __name__ == '__main__':
    _main()
