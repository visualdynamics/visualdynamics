"""The orientation marker of an interactive 3-D view: the triad, with a
cube at its origin whose faces turn the view to look straight at them,
and, while the view looks straight at one, two curved arrows that turn
it a quarter turn about the line of sight.

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
#: the turn arrows' arc: its radius and its width, in the marker's
#: units, and the angles each one runs between, from the top
ARC_RADIUS = 1.05
ARC_WIDTH = 0.12
ARC_SPAN = (22.0, 55.0)

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
        # Brandon, 2026-11-23: the quarter turn Callum Hammill's score map
        # offered on a right-click menu, moved onto the marker
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
        ink = pv.Color(text_color if text_color is not None else 'gray')
        #: the turn arrows, keyed by their turn: +1 clockwise on screen,
        #: top right, and -1 counterclockwise, top left
        self.arrows: dict[int, Any] = {}
        for turn in (1, -1):
            arrow = vtk.vtkActor()
            arrow_mapper = vtk.vtkPolyDataMapper()
            arrow_mapper.SetInputData(_turn_arrow(turn))
            arrow.SetMapper(arrow_mapper)
            arrow.GetProperty().SetColor(*ink.float_rgb)
            arrow.GetProperty().LightingOff()
            arrow.SetUserMatrix(vtk.vtkMatrix4x4())
            self.renderer.AddActor(arrow)
            self.arrows[turn] = arrow
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
        # the arrows stay put on screen as the cube turns under them, and
        # show only while the view looks straight along an axis
        up = np.asarray(main.GetViewUp(), dtype=float)
        right = np.cross(direction, up)
        square = bool(np.max(np.abs(direction)) > SQUARE)
        for arrow in self.arrows.values():
            matrix = arrow.GetUserMatrix()
            for row in range(3):
                for column, vector in enumerate((right, up, -direction)):
                    matrix.SetElement(row, column, vector[row])
            matrix.Modified()
            arrow.SetVisibility(square)
        self.renderer.ResetCameraClippingRange(
            -REACH, REACH, -REACH, REACH, -REACH, REACH)

    def face_at(self, x: float, y: float) -> tuple[int, int] | None:
        """The cube's face under the window pixel `(x, y)`, measured from
        the bottom left as VTK does, as (axis, sign); None off the
        cube."""
        if self._picker.Pick(x, y, 0.0, self.renderer):
            cell = self._picker.GetCellId()
            if cell >= 0 and self._picker.GetActor() is self.cube:
                return self.faces[cell]
        return None

    def arrow_at(self, x: float, y: float) -> int | None:
        """The turn arrow under the window pixel `(x, y)`, as its turn:
        +1 for the clockwise one, -1 for the other; None off both, and
        while they are hidden."""
        if self._picker.Pick(x, y, 0.0, self.renderer):
            for turn, arrow in self.arrows.items():
                if self._picker.GetActor() is arrow:
                    return turn
        return None

    def arrow_pixel(self, turn: int) -> tuple[float, float]:
        """The window pixel the middle of a turn arrow's arc is drawn at."""
        self.follow()
        angle = np.radians(90.0 - turn * sum(ARC_SPAN) / 2.0)
        local = np.array([np.cos(angle), np.sin(angle), 0.0, 1.0]) * \
            np.array([ARC_RADIUS, ARC_RADIUS, 1.0, 1.0])
        matrix = self.arrows[turn].GetUserMatrix()
        point = matrix.MultiplyPoint(tuple(local))
        self.renderer.SetWorldPoint(*point[:3], 1.0)
        self.renderer.WorldToDisplay()
        x, y, _depth = self.renderer.GetDisplayPoint()
        return x, y

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


def _turn_arrow(turn: int) -> Any:
    """A flat curved arrow in the screen's plane (x right, y up), on the
    arc around the cube: top right pointing clockwise for `turn` +1, its
    mirror top left for -1."""
    import pyvista as pv

    inner, outer = ARC_RADIUS - ARC_WIDTH / 2.0, ARC_RADIUS + ARC_WIDTH / 2.0
    start, stop = ARC_SPAN      # degrees clockwise from the top
    steps = np.linspace(start, stop, 12)
    points = []
    for radius in (inner, outer):
        for angle in np.radians(90.0 - steps):
            points.append((radius * np.cos(angle), radius * np.sin(angle), 0.0))
    count = len(steps)
    faces = []
    for i in range(count - 1):
        faces += [4, i, i + 1, count + i + 1, count + i]
    # the head: a triangle wider than the band, its tip further round
    base = np.radians(90.0 - stop)
    tip = np.radians(90.0 - (stop + 14.0))
    head = len(points)
    for radius, angle in ((inner - ARC_WIDTH, base), (outer + ARC_WIDTH, base),
                          (ARC_RADIUS, tip)):
        points.append((radius * np.cos(angle), radius * np.sin(angle), 0.0))
    faces += [3, head, head + 1, head + 2]
    points = np.asarray(points)
    if turn < 0:
        points[:, 0] *= -1.0
    return pv.PolyData(points, faces=np.asarray(faces))


def turn_view(plotter: Any, turn: int, render: bool = True) -> None:
    """Turn the view a quarter turn about the line of sight: the scene
    turns clockwise on screen for `turn` +1, counterclockwise for -1.
    The camera stays where it is; only which way is up changes."""
    camera = plotter.renderer.GetActiveCamera()
    direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
    up = np.asarray(camera.GetViewUp(), dtype=float)
    right = np.cross(direction, up)
    camera.SetViewUp(*(-turn * right))
    camera.OrthogonalizeViewUp()
    if render:
        plotter.render()


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
