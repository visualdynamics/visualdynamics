"""A frame of animation must not cost the size of the model.

Measured on a 202 500-node, 201 601-quad plate: a frame is 2.2 ms, which
is seven percent of the 33 ms a 30 fps timer has to spend. That headroom
is the *size ceiling* — it is what makes a real FE model animate at all —
and it comes from three structural properties rather than from anything
being fast:

- a frame is **one write** into a shared point buffer and one into a
  shared scalar buffer, seen by every mesh at once;
- meshes are grouped by **color**, so a scene has a dozen of them
  whatever the model size, and a dozen draw calls;
- nothing derived is rebuilt per frame — no actor added, no mesh
  replaced, no array reallocated.

None of that is asserted here in milliseconds. A wall-clock ceiling
measures the machine it runs on, and this suite has been taught that
lesson twice; what is pinned is the *shape* of the work, which is what
actually decays. Tubes and sphere glyphs were measured at 120 ms and
1 258 ms a frame on this model and rejected on these grounds, and the
way that decision gets quietly undone is a per-entity actor creeping
into the draw path.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.geometry import Geometry
from visualdynamics.deform import ShapeDeflection


def plate(side: int) -> Geometry:
    """A meshed square of quads, `side` nodes each way."""
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    nodes = np.arange(1, side * side + 1)
    xyz = np.column_stack([xs.ravel() * 0.01, ys.ravel() * 0.01,
                           np.zeros(side * side)])
    index = nodes.reshape(side, side)
    quads = [np.column_stack([index[r, :-1], index[r, 1:],
                              index[r + 1, 1:], index[r + 1, :-1]])
             for r in range(side - 1)]
    conn = np.vstack(quads)
    return Geometry(node_id=nodes, node_xyz=xyz, elem_conn=list(conn),
                    elem_type=[44] * len(conn), length_unit='m')


def animator_for(geometry, colormap=False):
    import pyvista as pv

    from visualdynamics.viz.animate import GeometryAnimator

    dofs = [f'{int(node)}Z+' for node in geometry.node_id]
    shape = np.sin(np.arange(geometry.num_nodes) / 20.0)
    plotter = pv.Plotter(off_screen=True)
    return GeometryAnimator(plotter, geometry,
                            ShapeDeflection(geometry, dofs, shape),
                            colormap=colormap)


@pytest.fixture(scope='module')
def small():
    return plate(20)          # 400 nodes


@pytest.fixture(scope='module')
def larger():
    return plate(60)          # 3 600 nodes, nine times the first


def test_the_scene_has_a_mesh_per_color_not_per_entity(small, larger,
                                                        qt_app):
    """Nine times the nodes, the same number of meshes — which is the
    whole reason a large model draws at all."""
    few = animator_for(small)
    many = animator_for(larger)
    assert len(few.meshes) == len(many.meshes)
    assert len(many.meshes) <= 16, (
        f'{len(many.meshes)} meshes for one plate: something is drawing '
        'per entity')


def test_a_frame_adds_no_actor_and_replaces_no_mesh(small, qt_app):
    """The scene is built once. A frame that added an actor would cost
    the driver a state change per frame and the model its ceiling."""
    moving = animator_for(small)
    before_actors = len(moving.plotter.renderer.actors)
    before_meshes = [id(mesh) for mesh in moving.meshes]
    for step in range(5):
        moving.set_parameter(step * 0.4)
    assert len(moving.plotter.renderer.actors) == before_actors
    assert [id(mesh) for mesh in moving.meshes] == before_meshes


def test_a_frame_writes_into_the_buffer_the_meshes_already_share(small,
                                                                 qt_app):
    """One upload per frame rather than one per mesh: the points object
    is the same object afterwards, and its contents have moved."""
    moving = animator_for(small)
    points = moving.points_source
    view = moving.view
    moving.set_parameter(0.0)
    at_rest = view.copy()
    moving.set_parameter(np.pi / 2)
    assert moving.points_source is points, 'the buffer was replaced'
    assert moving.view is view, 'and so was the view onto it'
    assert not np.array_equal(view, at_rest), 'but the nodes did move'


def test_coloring_by_displacement_reuses_one_array_for_every_mesh(small,
                                                                   qt_app):
    moving = animator_for(small, colormap=True)
    scalars = moving.scalars_source
    magnitude = moving.magnitude
    assert magnitude is not None
    assert all(mesh.GetPointData().GetScalars() is scalars
               for mesh in moving.meshes), 'one array, every mesh'
    moving.set_parameter(np.pi / 2)
    assert moving.scalars_source is scalars
    assert moving.magnitude is magnitude, 'written into, not reallocated'


def test_a_frame_allocates_nothing_that_grows_with_the_model(small, qt_app):
    """The per-frame work is a fixed set of buffers being written. If a
    frame allocated per node, the count of live arrays of that size would
    climb as it ran."""
    import gc

    moving = animator_for(small)
    moving.set_parameter(0.1)          # first frame: warm every cache
    gc.collect()
    nodes = small.num_nodes

    def big_arrays():
        # type(), not isinstance(): the heap holds weak proxies whose
        # referents have died (destroyed windows, now that they do),
        # and isinstance on a dead proxy raises ReferenceError
        return sum(1 for obj in gc.get_objects()
                   if type(obj) is np.ndarray and obj.size >= nodes)

    before = big_arrays()
    for step in range(10):
        moving.set_parameter(step * 0.3)
    gc.collect()
    assert big_arrays() <= before, (
        'a frame is leaving arrays the size of the model behind')
