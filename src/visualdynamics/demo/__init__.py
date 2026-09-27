"""Models to try the toolset on, built by the toolset.

A structural dynamics package needs something to demonstrate on, and a
from-scratch one has a particular reason to build its own: the demonstration
is the claim. These models are assembled with `visualdynamics.fem` and
nothing else, so every number they produce — geometry, mode shapes,
synthesized FRFs — is the package's own answer rather than another
library's, imported.

    from visualdynamics.demo import drone

    model = drone.build()
    shapes = model.eigensolution(maximum_frequency=300, damping=0.02)
    model.geometry().plot()

Building is deliberately a call rather than a module-level object. The dense
quadcopter is 1230 degrees of freedom and its eigensolution takes a second
or two, which is not something to spend on an import that may only have
wanted the mesh.
"""

#: The submodules are not imported here. `from visualdynamics.demo import
#: drone` resolves one on its own, and importing it eagerly would mean
#: `python -m visualdynamics.demo.drone` loads the module twice — which Python
#: warns about, having already put the first copy in sys.modules.
__all__ = ['barc', 'drone', 'plate']
