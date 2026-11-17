"""The orientation marker of an interactive 3-D view: the triad, with a
cube at its origin whose faces turn the view to look straight at them.

The marker is a small renderer of its own in a corner of the window,
turned with the main camera before every frame. It uses no Qt; the pane
that owns the view passes clicks to `ViewCube.face_at`.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

from .geometry import AXIS_COLORS

# Brandon, 2026-11-17: a way to line the view up with an axis without
# another button on the bar, as Blender's corner gizmo does. PyVista's
# own triad (`add_axes`) is a marker widget whose renderer is private,
# and a triad put in an assembly with a cube is not drawn at all, so the
# marker here is a renderer this module makes and can pick in.

#: the corner of the window the marker is drawn in, as PyVista's triad
VIEWPORT = (0.0, 0.0, 0.2, 0.2)
#: the cube's edge, where the triad's arrows are one long
EDGE = 0.6
#: half the height of what the marker's camera shows: the arrows and
#: their labels, turned any way
REACH = 1.3
#: how much of a negative face's color is white, to tell it from the
#: positive face of its axis
TINT = 0.6
#: looking this squarely at a face already, a click on it turns the
#: view to the opposite face
SQUARE = 0.999

_cubes: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _rgb(color: str, tint: float = 0.0) -> list[int]:
    color = color.lstrip('#')
    parts = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
    return [round(part + (255 - part) * tint) for part in parts]


class ViewCube:
    """The marker drawn in one plotter, until `remove`.

    Attributes
    ----------
    renderer : vtkRenderer
        The marker's own renderer, in the window's corner.
    faces : list of (int, int)
        Each of the cube's cells as (axis, sign): the face it is.
    """

    def __init__(self, plotter: Any, text_color: Any = None) -> None:
        import pyvista as pv
        import vtk
        from pyvista.plotting.tools import create_axes_marker

        self.plotter: Any = plotter
        self.text_color: Any = text_color
        cube = pv.Cube(x_length=EDGE, y_length=EDGE, z_length=EDGE)
        self.faces: list[tuple[int, int]] = []
        colors = []
        for normal in np.asarray(cube.cell_normals):
            axis = int(np.argmax(np.abs(normal)))
            sign = 1 if normal[axis] > 0 else -1
            self.faces.append((axis, sign))
            colors.append(_rgb(AXIS_COLORS[axis], 0.0 if sign > 0 else TINT))
        cube.cell_data['color'] = np.asarray(colors, dtype=np.uint8)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(cube)
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        self.cube: Any = vtk.vtkActor()
        self.cube.SetMapper(mapper)
        # mostly its own color: a face turned from the light is still
        # told from its neighbors by color, not by shade
        self.cube.GetProperty().SetAmbient(0.8)
        self.cube.GetProperty().SetDiffuse(0.25)
        self.axes: Any = create_axes_marker(label_color=text_color)

        self.renderer: Any = vtk.vtkRenderer()
        self.renderer.SetViewport(*VIEWPORT)
        self.renderer.InteractiveOff()
        self.renderer.AddViewProp(self.axes)
        self.renderer.AddActor(self.cube)
        self.renderer.GetActiveCamera().ParallelProjectionOn()
        window = plotter.render_window
        # on a layer over the scene's, where a renderer draws on what is
        # there; on the scene's own layer it painted its corner black
        window.SetNumberOfLayers(max(window.GetNumberOfLayers(), 2))
        self.renderer.SetLayer(window.GetNumberOfLayers() - 1)
        window.AddRenderer(self.renderer)
        self._picker: Any = vtk.vtkCellPicker()
        # the triad's arrows are not picked by cells, so a hit is the cube
        self._picker.SetTolerance(0.0)
        self._observer: Any = plotter.renderer.AddObserver(
            'StartEvent', lambda *_: self.follow())
        self.follow()

    def follow(self) -> None:
        """Turn the marker as the main camera is turned."""
        main = self.plotter.renderer.GetActiveCamera()
        camera = self.renderer.GetActiveCamera()
        direction = np.asarray(main.GetDirectionOfProjection(), dtype=float)
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(*(-10.0 * direction))
        camera.SetViewUp(*main.GetViewUp())
        camera.SetParallelScale(REACH)
        self.renderer.ResetCameraClippingRange(
            -REACH, REACH, -REACH, REACH, -REACH, REACH)

    def face_at(self, x: float, y: float) -> tuple[int, int] | None:
        """The cube's face under the window pixel `(x, y)`, measured from
        the bottom left as VTK does, as (axis, sign); None off the
        cube."""
        if self._picker.Pick(x, y, 0.0, self.renderer):
            cell = self._picker.GetCellId()
            if cell >= 0:
                return self.faces[cell]
        return None

    def face_pixel(self, axis: int, sign: int) -> tuple[float, float]:
        """The window pixel a face's center is drawn at."""
        center = np.zeros(3)
        center[axis] = sign * EDGE / 2.0
        self.renderer.SetWorldPoint(*center, 1.0)
        self.renderer.WorldToDisplay()
        x, y, _depth = self.renderer.GetDisplayPoint()
        return x, y

    def remove(self) -> None:
        """Take the marker out of the window."""
        self.plotter.renderer.RemoveObserver(self._observer)
        self.plotter.render_window.RemoveRenderer(self.renderer)


def show_view_cube(plotter: Any, shown: bool, text_color: Any = None) -> None:
    """Draw the marker in `plotter`'s corner, or take it away. Shown
    again, the one already there is kept, with its labels recolored."""
    cube = _cubes.get(plotter)
    if not shown:
        if cube is not None:
            cube.remove()
            del _cubes[plotter]
        return
    if cube is not None:
        if cube.text_color == text_color:
            return
        cube.remove()
    _cubes[plotter] = ViewCube(plotter, text_color)


def view_cube(plotter: Any) -> ViewCube | None:
    """The marker drawn in `plotter`, or None."""
    return _cubes.get(plotter)


def look_at_face(plotter: Any, axis: int, sign: int,
                 render: bool = True) -> tuple[int, int]:
    """Turn the camera to look straight at a face of the model's axes:
    from the +X side for (0, 1), from below for (2, -1). Looking at
    that face already, the view turns to the opposite one. What the
    camera looks at and how far it stands stay as they are, so the
    scene keeps its size. Z is up on screen from the sides; from above
    Y is up, and from below it is down.

    Returns the face now looked at.
    """
    camera = plotter.renderer.GetActiveCamera()
    toward = np.zeros(3)
    toward[axis] = sign
    if -np.dot(camera.GetDirectionOfProjection(), toward) > SQUARE:
        sign = -sign
        toward = -toward
    focus = np.asarray(camera.GetFocalPoint(), dtype=float)
    camera.SetPosition(*(focus + toward * camera.GetDistance()))
    camera.SetViewUp(*((0.0, 0.0, 1.0) if axis != 2
                       else (0.0, float(sign), 0.0)))
    plotter.renderer.ResetCameraClippingRange()
    if render:
        plotter.render()
    return axis, sign
