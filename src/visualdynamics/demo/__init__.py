"""Models to try the toolset on, built with the toolset.

There are four: the quadcopter (`drone`), the free plate (`plate`), the
BARC (`barc`) and the Frame-Wing (`frame_wing`). Each is built with this
package alone (`visualdynamics.fem`, `visualdynamics.mesh` and the
project verbs), so every number they produce (geometry, mode shapes,
synthesized FRFs) is computed by this package.

    from visualdynamics.demo import drone

    model = drone.build()
    shapes = model.eigensolution(maximum_frequency=300, damping=0.02)
    model.geometry().plot()

A model is built by calling `build()`, not when the module is imported.
The quadcopter has 8532 degrees of freedom and its eigensolution takes
about a second, which an import that only wanted the mesh should not
have to spend.
"""

#: The submodules are not imported here. `from visualdynamics.demo import
#: drone` resolves one on its own, and importing it eagerly would mean
#: `python -m visualdynamics.demo.drone` loads the module twice — which Python
#: warns about, having already put the first copy in sys.modules.
__all__ = ['barc', 'drone', 'frame_wing', 'plate']
