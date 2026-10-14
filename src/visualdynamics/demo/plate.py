"""The demonstration test article: a free aluminum plate.

A 12 in x 12 in x 0.5 in plate of 6061-T6, free-free. A completely free
square plate is a classical benchmark: its nondimensional frequencies
are tabulated (Leissa, NASA SP-160), and most test engineers have tapped
one with a hammer. The drone shows a large model; the plate is the one
whose answers can be checked against a book.

It is built with `visualdynamics.fem` only, from the rectangular MITC4
shell element, which `tests/test_fem.py` checks against the same tables.
The small test fixtures are made from this model (`testdata/generate_plate.py`),
as the large ones are made from the drone.

    from visualdynamics.demo import plate

    model = plate.build()
    shapes = model.eigensolution(maximum_frequency=2500, damping=0.02)

Run it directly to print what it weighs and where its modes land
against the classical table:

    python3 -m visualdynamics.demo.plate

The plate is defined in inches and pounds, as the drawing for a machine
shop would give it, and converted exactly to SI once, below, because
`fem.Model` works in SI like the rest of the package. The units shown
are set in the app's status bar.

The frequencies agree with the tables to about a percent, for two
reasons. One is the mesh: the frequencies converge from above as h^2,
which `tests/test_fem.py` measures per mode. The other is Poisson's
ratio: the classical values are computed at 0.3, 6061 is 0.33, and a
free plate's frequencies change a little with it.
"""

from __future__ import annotations

import math

from visualdynamics import fem

#: the drawing: 12 in x 12 in x 0.5 in, converted exactly
SIDE = 12.0 * 0.0254                      #: m
THICKNESS = 0.5 * 0.0254                  #: m

#: 6061-T6 as the handbook states it — E = 10.0e6 psi,
#: rho = 0.098 lb/in^3, nu = 0.33 — converted exactly to SI, which is
#: what the material library holds for it
ALUMINUM = fem.material('6061-T6')

#: elements per side. 12 puts the first six elastic modes within a
#: percent of converged and solves in a fifth of a second; the tests
#: build coarser when the mesh is not what they are checking.
MESH = 12


def build(mesh: int = MESH) -> fem.Model:
    """The plate's finite element model, `mesh` x `mesh` plate elements
    (12 by default), in meters.

    Node ids are numbered like a survey grid: row 1 is 101, 102, ...
    along the edge at y = 0, row 2 is 201, 202, ..., so node 704 is the
    fourth node of the seventh row, the numbering a channel table would
    use.
    """
    model = fem.Model('plate', length_unit='m')
    for j in range(mesh + 1):
        for i in range(mesh + 1):
            model.add_node((j + 1) * 100 + (i + 1),
                           i * SIDE / mesh, j * SIDE / mesh, 0.0)
    for j in range(mesh):
        for i in range(mesh):
            corner = (j + 1) * 100 + (i + 1)
            model.add_plate((corner, corner + 1,
                             corner + 101, corner + 100),
                            ALUMINUM, THICKNESS)
    return model


def instrumented(model: fem.Model) -> dict[str, list[int]]:
    """The nodes a modal survey of this plate would put sensors on, as
    {'grid': node ids, 'drive': [node id]}.

    The grid is five by five accelerometers, a common layout for a plate
    survey and dense enough that the sixth elastic mode (three
    half-waves) is not spatially aliased. The drive point is the corner
    node 101, because a corner of a free plate moves in every low mode
    while the center is on the node lines of all the antisymmetric ones.
    The nodes are found from the model, not written down, because their
    ids depend on the mesh.
    """
    ids = model.node_ids
    rows = round(math.sqrt(len(ids))) - 1
    step = rows // 4 if rows >= 4 else 1
    stations = sorted({0, step, 2 * step, 3 * step, rows})
    grid = [(j + 1) * 100 + (i + 1)
            for j in stations for i in stations]
    return {'grid': grid, 'drive': [grid[0]]}


def describe(model: fem.Model | None = None) -> None:
    """Print the plate's size, its mass, and its elastic frequencies
    beside the classical nondimensional values. `model` defaults to
    `build()`."""
    model = model or build()
    shapes = model.eigensolution(maximum_frequency=2500.0)
    bending = (ALUMINUM.youngs_modulus * THICKNESS ** 3
               / (12.0 * (1.0 - ALUMINUM.poissons_ratio ** 2)))
    scale = (2.0 * math.pi * SIDE ** 2
             * math.sqrt(ALUMINUM.density * THICKNESS / bending))
    classical = [13.468, 19.596, 24.271, 34.801, 34.801, 61.093]

    pounds = model.structural_mass / 0.45359237
    print(f'{model.num_nodes} nodes, {model.num_dof} DOF, '
          f'{len(model.plates)} plate elements')
    print(f'{model.structural_mass:.2f} kg ({pounds:.1f} lb)')
    print()
    print(f'{"f (Hz)":>9}  {"lambda^2":>9}  {"classical (nu=0.3)":>19}')
    elastic = [f for f in shapes.frequency if f > 0.0]
    for k, frequency in enumerate(elastic):
        book = (f'{classical[k]:>19.3f}' if k < len(classical)
                else f'{"—":>19}')
        print(f'{frequency:9.1f}  {frequency * scale:9.3f}  {book}')


if __name__ == '__main__':
    describe()
