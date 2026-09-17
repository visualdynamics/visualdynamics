"""The two example-project tiles' pictures: the plate and the quadcopter.

The downloads page offers two example bundles, and each tile wears a
picture of the thing inside it rather than the application's mark
(Brandon, 2026-09-16) — a render of the geometry itself, from the same
scene the app draws, on a transparent ground so the tile's own panel
shows through. The plate is the test fixture's mesh; the quadcopter is
the demonstration airframe's drawing (no eigensolution, so this takes
seconds).

    python tools/make_example_icons.py          # write web/launch/{plate,drone}.png

Renders are not byte-stable across VTK builds, so there is no --check;
`tests/test_website.py` holds the files to their shape instead.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / 'web' / 'launch'

#: rendered square; the tile shows it at 46 CSS pixels, so this is
#: generous for any screen and still a few kilobytes
SIZE = 320


def plate():
    sys.path.insert(0, str(ROOT / 'src'))
    import visualdynamics
    return visualdynamics.import_file(
        str(ROOT / 'testdata' / 'plate' / 'geometry.exo'))


def drone():
    sys.path.insert(0, str(ROOT / 'src'))
    from visualdynamics.demo import drone as airframe
    shape = airframe.draw()
    shape.prune()
    shape.orient()
    return shape.geometry()


def render(geometry, path: pathlib.Path, *, azimuth: float, elevation: float,
           zoom: float) -> None:
    """The geometry alone — no axes, no labels, no ground — as a PNG."""
    import pyvista as pv

    from visualdynamics.theme import theme as resolve_theme
    from visualdynamics.viz.geometry import add_geometry

    colors = resolve_theme('dark')
    plotter = pv.Plotter(off_screen=True, window_size=(SIZE, SIZE))
    add_geometry(plotter, geometry, node_size=0.0, line_width=1.0,
                 show_edges=True, text_color=colors['scene_text'],
                 components=['elements', 'tracelines'])
    plotter.view_isometric()
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.camera.zoom(zoom)
    plotter.screenshot(str(path), transparent_background=True)
    plotter.close()


def main() -> int:
    render(plate(), SITE / 'plate.png', azimuth=30.0, elevation=25.0, zoom=1.0)
    render(drone(), SITE / 'drone.png', azimuth=45.0, elevation=50.0, zoom=1.2)
    for name in ('plate.png', 'drone.png'):
        print(f'wrote {SITE / name} ({(SITE / name).stat().st_size // 1024} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
