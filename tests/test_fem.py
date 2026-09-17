"""The beam element, checked against answers that exist without it.

Every frequency here has a closed form — a cantilever's bending series, a
free-free rod's axial series, a lumped mass on a light beam — so a wrong
sign or a transposed inertia cannot pass by looking plausible. That is the
point of the file: a finite element model has no independent truth of its
own, and 'it ran and gave numbers' is not evidence.

Convergence is stated as an inequality against the analytic value, not as a
tolerance around it, wherever the discretization has a known direction: a
consistent mass matrix is a Rayleigh-Ritz reduction, so its frequencies come
out *above* the exact ones and approach from that side. A model that
undershoots is not a slightly coarse mesh, it is a bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visualdynamics.core import fem
from visualdynamics.core.geometry import Geometry

STEEL = fem.Material('steel', youngs_modulus=200e9, density=7850,
                     poissons_ratio=0.3)


def straight_beam(length, section, elements, material=STEEL, axis=(1.0, 0.0, 0.0),
                  first=100, orientation=None):
    """A beam of `elements` equal elements laid along `axis` from the origin."""
    model = fem.Model('beam')
    direction = np.asarray(axis, dtype=float)
    direction = direction / np.linalg.norm(direction)
    for i in range(elements + 1):
        point = direction * (length * i / elements)
        model.add_node(first + i, *point)
    model.add_chain(range(first, first + elements + 1), material, section,
                    orientation=orientation)
    return model


# ---- the sections themselves ---------------------------------------------


def test_round_tube_reduces_to_a_rod():
    """A tube whose wall reaches the center is a solid rod: pi d^4 / 64."""
    rod = fem.Section.rod('rod', 0.02)
    assert rod.area == pytest.approx(math.pi * 0.01 ** 2)
    assert rod.iy == pytest.approx(math.pi * 0.02 ** 4 / 64)
    # a circle does not warp, so torsion is exactly the polar moment
    assert rod.j == pytest.approx(rod.polar)


def test_a_rectangle_bends_differently_about_its_two_axes():
    """Width lies along local y, so it is `iz` that carries the width cubed."""
    section = fem.Section.rectangle('bar', width=0.04, height=0.01)
    assert section.iz == pytest.approx(0.04 ** 3 * 0.01 / 12)
    assert section.iy == pytest.approx(0.04 * 0.01 ** 3 / 12)
    # and torsion of a thin strip is well under the polar moment, which is
    # the error using the polar value would make
    assert section.j < 0.35 * section.polar


def test_a_square_tube_is_stiffer_in_torsion_than_an_open_section():
    tube = fem.Section.square_tube('tube', width=0.016, wall=0.0015)
    assert tube.area == pytest.approx(0.016 ** 2 - 0.013 ** 2)
    # Bredt's closed-section value lands near the polar moment for a
    # square tube; nowhere near the thin-strip figure of an open one
    assert 0.7 < tube.j / tube.polar < 1.3


# ---- the element, against closed forms -----------------------------------


def test_cantilever_bending_matches_the_analytic_series():
    """f_n = (beta_n L)^2 / (2 pi L^2) sqrt(EI / rho A)."""
    length, section = 1.0, fem.Section.rectangle('bar', 0.02, 0.02)
    model = straight_beam(length, section, 20)
    shapes = model.eigensolution(fixed=['100'])
    rigidity = math.sqrt(STEEL.youngs_modulus * section.iy
                         / (STEEL.density * section.area))
    exact = [(beta ** 2) * rigidity / (2 * math.pi * length ** 2)
             for beta in (1.87510407, 4.69409113, 7.85475744)]

    # bending happens in two planes at once on a square section, so each
    # analytic frequency appears twice; take the distinct ones
    found = _distinct(shapes.frequency)
    for computed, analytic in zip(found, exact):
        assert computed == pytest.approx(analytic, rel=2e-3)
        # Rayleigh-Ritz: the discretization can only stiffen
        assert computed >= analytic


def test_refining_the_mesh_closes_on_the_answer_from_above():
    length, section = 1.0, fem.Section.rectangle('bar', 0.02, 0.02)
    rigidity = math.sqrt(STEEL.youngs_modulus * section.iy
                         / (STEEL.density * section.area))
    exact = 7.85475744 ** 2 * rigidity / (2 * math.pi * length ** 2)

    errors = []
    for elements in (4, 8, 16, 32):
        model = straight_beam(length, section, elements)
        third = _distinct(model.eigensolution(fixed=['100']).frequency)[2]
        assert third >= exact
        errors.append(third / exact - 1.0)
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-4


def test_cantilever_axial_and_torsional_series():
    """Both are the same quarter-wave series, with different wave speeds."""
    length = 1.0
    section = fem.Section.rod('rod', 0.02)
    model = straight_beam(length, section, 24)
    shapes = model.eigensolution(fixed=['100'])

    axial = math.sqrt(STEEL.youngs_modulus / STEEL.density) / (4 * length)
    torsion = math.sqrt(STEEL.shear_modulus / STEEL.density) / (4 * length)

    # Identify them by what moves. On a round rod both are uncoupled from
    # bending, so each is recognizable without comparing a meter against a
    # radian: the axial mode translates along X and hardly at all across
    # it, and the torsional one twists about X while translating nothing.
    def find(recognize):
        for frequency, shape in zip(shapes.frequency, shapes.shape_matrix):
            if recognize(_participation(model, shape)):
                return frequency
        return None

    stretching = find(lambda s: s[0] > 10 * max(s[1], s[2]))
    assert stretching is not None, 'no axial mode found'
    assert stretching == pytest.approx(axial, rel=2e-3)
    assert stretching >= axial

    twisting = find(lambda s: s[3] > 10 * s[:3].max())
    assert twisting is not None, 'no torsional mode found'
    assert twisting == pytest.approx(torsion, rel=2e-3)
    assert twisting >= torsion


def test_a_lumped_mass_on_a_light_beam_is_a_single_degree_of_freedom():
    """f = sqrt(3 EI / m L^3) / 2 pi, when the beam's own mass is nothing."""
    length = 0.5
    section = fem.Section.rod('rod', 0.006)
    light = fem.Material('light', youngs_modulus=200e9, density=1.0)
    model = straight_beam(length, section, 8, material=light)
    model.add_mass(108, 2.0)

    stiffness = 3 * light.youngs_modulus * section.iy / length ** 3
    exact = math.sqrt(stiffness / 2.0) / (2 * math.pi)
    first = model.eigensolution(fixed=['100']).frequency[0]
    assert first == pytest.approx(exact, rel=5e-3)


def test_free_free_beam_has_six_rigid_modes_and_the_right_elastic_ones():
    length, section = 1.0, fem.Section.rectangle('bar', 0.02, 0.02)
    model = straight_beam(length, section, 20)
    shapes = model.eigensolution()

    rigid = shapes.frequency[:6]
    assert np.all(rigid == 0.0), f'rigid-body modes came out at {rigid}'
    assert shapes.frequency[6] > 0.0

    rigidity = math.sqrt(STEEL.youngs_modulus * section.iy
                         / (STEEL.density * section.area))
    exact = [(beta ** 2) * rigidity / (2 * math.pi * length ** 2)
             for beta in (4.73004074, 7.85320462)]
    elastic = _distinct(shapes.frequency[6:])
    for computed, analytic in zip(elastic, exact):
        assert computed == pytest.approx(analytic, rel=2e-3)
        assert computed >= analytic


def test_a_rigid_body_mode_is_exactly_zero_not_merely_small():
    """`synthesize_frf` cancels a rigid mode's 0/0 by testing frequency == 0,
    so 'small' would reach it as a resonance a hair above DC instead."""
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 12)
    frequency = model.eigensolution().frequency
    assert list(frequency[:6]) == [0.0] * 6


def test_the_rigid_modes_are_recognized_on_a_soft_structure_too():
    """The test is projection onto the null space, not a threshold on the
    frequency: a floppy model's first elastic mode can be lower than a
    stiff one's numerical zero, and a frequency cutoff would confuse them."""
    floppy = fem.Material('floppy', youngs_modulus=1e6, density=7850)
    model = straight_beam(2.0, fem.Section.rod('wire', 0.002), 16,
                          material=floppy)
    shapes = model.eigensolution()
    assert list(shapes.frequency[:6]) == [0.0] * 6
    assert shapes.frequency[6] < 5.0        # genuinely floppy


# ---- the things a wrong transformation would break ------------------------


def test_frequencies_do_not_depend_on_how_the_model_is_oriented():
    """The same beam pointed three ways gives the same answer, or the
    element-to-global rotation is wrong."""
    section = fem.Section.rectangle('bar', 0.03, 0.01)
    answers = []
    for axis, orientation in (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                              ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
                              ((1.0, 2.0, 3.0), (3.0, 0.0, -1.0))):
        model = straight_beam(1.0, section, 10, axis=axis,
                              orientation=orientation)
        answers.append(model.eigensolution(fixed=['100']).frequency[:8])
    for other in answers[1:]:
        assert np.allclose(answers[0], other, rtol=1e-9)


def test_the_orientation_vector_decides_which_way_is_stiff():
    """A rectangular section is four times stiffer edgewise, and rolling
    the section about the beam axis has to move that stiffness with it."""
    section = fem.Section.rectangle('bar', 0.04, 0.01)
    flat = straight_beam(1.0, section, 10, orientation=(0.0, 1.0, 0.0))
    edge = straight_beam(1.0, section, 10, orientation=(0.0, 0.0, 1.0))
    soft_flat = flat.eigensolution(fixed=['100']).frequency[0]
    soft_edge = edge.eigensolution(fixed=['100']).frequency[0]
    # the softest bending is the same number either way — it is the *plane*
    # that has rotated — so what must differ is which plane it happens in
    assert soft_flat == pytest.approx(soft_edge, rel=1e-9)
    assert _dominant_translation(flat, flat.eigensolution(fixed=['100'])
                                 .shape_matrix[0]) == 'Z+'
    assert _dominant_translation(edge, edge.eigensolution(fixed=['100'])
                                 .shape_matrix[0]) == 'Y+'


def test_an_orientation_along_the_beam_names_no_plane():
    model = fem.Model()
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match='names no plane'):
        model.add_beam(1, 2, STEEL, fem.Section.rod('r', 0.01),
                       orientation=(1.0, 0.0, 0.0))
        model.matrices()


def test_a_vertical_member_gets_a_usable_default_orientation():
    """Global Z is the default reference, which a vertical beam is parallel
    to; without the fallback its local axes would be a division by zero."""
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 6,
                          axis=(0.0, 0.0, 1.0))
    frequency = model.eigensolution(fixed=['100']).frequency
    assert np.all(np.isfinite(frequency))
    assert frequency[0] > 0.0


# ---- the matrices' own properties ----------------------------------------


def test_the_stiffness_matrix_annihilates_rigid_body_motion():
    """K r = 0 for all six rigid motions — the strongest single check on
    the element, since it fails for almost any sign error."""
    model = straight_beam(1.0, fem.Section.rectangle('bar', 0.03, 0.01), 7,
                          axis=(1.0, 1.0, 0.4))
    _mass, stiffness = model.matrices()
    rigid = model.rigid_body_vectors()
    residual = stiffness @ rigid
    assert np.max(np.abs(residual)) < 1e-6 * np.max(np.abs(stiffness))


def test_the_mass_matrix_carries_the_right_total_mass_and_inertia():
    # a rectangle, deliberately: on a round section the torsion constant
    # and the polar moment are the same number, so a rod cannot tell which
    # one the rotary inertia was built from. This test passed with the
    # wrong one until the section stopped being circular.
    length = 1.0
    section = fem.Section.rectangle('bar', 0.04, 0.01)
    assert section.j < 0.5 * section.polar, 'the section must distinguish them'
    model = straight_beam(length, section, 10, orientation=(0.0, 1.0, 0.0))
    model.add_mass(105, 3.0, inertia=(0.1, 0.2, 0.3))
    mass, _ = model.matrices()
    rigid = model.rigid_body_vectors()

    expected = STEEL.density * section.area * length + 3.0
    assert model.total_mass == pytest.approx(expected)
    for axis in range(3):
        moved = rigid[:, axis]
        assert float(moved @ mass @ moved) == pytest.approx(expected)

    # spinning about the beam's own axis: the rod's polar inertia plus
    # the lumped item's, and nothing from the translational mass
    spin = rigid[:, 3]
    polar = STEEL.density * section.polar * length + 0.1
    assert float(spin @ mass @ spin) == pytest.approx(polar, rel=1e-9)


def test_the_shapes_come_out_mass_normalized():
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 8,
                          axis=(0.3, 1.0, 0.2))
    mass, _ = model.matrices()
    shapes = model.eigensolution(maximum_frequency=5000)
    generalized = shapes.shape_matrix @ mass @ shapes.shape_matrix.T
    assert np.allclose(generalized, np.eye(len(shapes.frequency)), atol=1e-8)


def test_a_face_shades_without_stiffening():
    """Faces are drawing, and the matrices must not notice them."""
    def build(with_faces):
        model = fem.Model()
        for i, point in enumerate([(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0), (0, 0.2, 0)]):
            model.add_node(10 + i, *point)
        section = fem.Section.rod('rod', 0.004)
        model.add_chain([10, 11, 12, 13, 10], STEEL, section)
        if with_faces:
            model.add_face([10, 11, 12, 13])
        return model

    plain, shaded = build(False), build(True)
    assert np.allclose(plain.matrices()[1], shaded.matrices()[1])
    assert plain.total_mass == shaded.total_mass
    assert len(shaded.geometry().elem_id) == len(plain.geometry().elem_id) + 1


# ---- what comes out ------------------------------------------------------


def test_the_geometry_carries_the_beams_as_elements():
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 4)
    geometry = model.geometry()
    assert list(geometry.node_id) == [100, 101, 102, 103, 104]
    assert geometry.length_unit == 'm'
    assert list(geometry.elem_type) == [21] * 4
    assert list(geometry.elem_conn[0]) == [100, 101]


def test_the_shape_set_covers_every_direction_including_rotations():
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 3)
    shapes = model.eigensolution(num_modes=8)
    assert shapes.coordinate[:6] == ['100X+', '100Y+', '100Z+',
                                     '100RX+', '100RY+', '100RZ+']
    assert shapes.mass_unit == 'kg'
    assert shapes.num_shapes == 8


def test_the_shapes_synthesize_an_frf_that_peaks_at_the_frequencies():
    """The end-to-end claim: these modes are a modal model, so the FRF
    built from them resonates where the eigensolution said it would."""
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 12)
    shapes = model.eigensolution(maximum_frequency=400, damping=0.01)
    frequencies = np.arange(0.0, 400.0, 0.25)
    frf = shapes.synthesize_frf(frequencies, ['112Z+'], ['112Z+'], power=2)
    peak = frequencies[np.argmax(np.abs(frf[0]))]
    elastic = [f for f in shapes.frequency if f > 0]
    assert min(abs(peak - f) for f in elastic) < 0.5


def test_maximum_frequency_and_num_modes_both_cut_the_set():
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 12)
    everything = model.eigensolution()
    banded = model.eigensolution(maximum_frequency=500)
    assert 6 < banded.num_shapes < everything.num_shapes
    assert banded.frequency.max() <= 500
    assert model.eigensolution(num_modes=9).num_shapes == 9
    # the two together take the tighter of them
    assert model.eigensolution(maximum_frequency=500,
                               num_modes=3).num_shapes == 3


# ---- refusals ------------------------------------------------------------


def test_fixing_a_node_grounds_all_six_and_a_dof_grounds_one():
    section = fem.Section.rod('rod', 0.01)
    model = straight_beam(1.0, section, 6)
    clamped = model.eigensolution(fixed=['100'])
    pinned = model.eigensolution(fixed=['100X+', '100Y+', '100Z+'])
    # a pin lets the end rotate, so everything is softer
    assert pinned.frequency[0] < clamped.frequency[0]
    # and the grounded rows are motionless in every mode
    assert np.allclose(clamped.shape_matrix[:, :6], 0.0)


def test_grounding_ignores_the_sign_of_a_direction():
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 6)
    assert np.allclose(model.eigensolution(fixed=['100Z-']).frequency,
                       model.eigensolution(fixed=['100Z+']).frequency)


def test_a_model_that_is_entirely_fixed_says_so():
    model = fem.Model()
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_beam(1, 2, STEEL, fem.Section.rod('r', 0.01))
    with pytest.raises(ValueError, match='every degree of freedom is fixed'):
        model.eigensolution(fixed=['1', '2'])


def test_naming_something_that_is_not_in_the_model_is_refused():
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 2)
    with pytest.raises(ValueError, match='not in the model'):
        model.add_beam(100, 999, STEEL, fem.Section.rod('r', 0.01))
    with pytest.raises(ValueError, match='not in the model'):
        model.add_mass(999, 1.0)
    with pytest.raises(ValueError, match='does not name a node'):
        model.eigensolution(fixed=['999'])


def test_a_duplicate_node_id_is_refused_at_entry():
    model = fem.Model()
    model.add_node(7, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match='already in the model'):
        model.add_node(7, 1.0, 0.0, 0.0)


def test_a_beam_to_itself_is_refused():
    model = fem.Model()
    model.add_node(1, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match='no length'):
        model.add_beam(1, 1, STEEL, fem.Section.rod('r', 0.01))


def test_a_free_node_with_no_stiffness_is_named_rather_than_crashing():
    """A node nothing connects to has no mass either, so the Cholesky
    factorization fails — and the useful thing to say is which node."""
    model = straight_beam(1.0, fem.Section.rod('rod', 0.01), 3)
    model.add_node(500, 5.0, 0.0, 0.0)
    with pytest.raises(ValueError, match='500X\\+'):
        model.eigensolution()


# ---- helpers -------------------------------------------------------------


def _distinct(frequencies, tolerance=1e-3):
    """One entry per distinct frequency: a square section bends identically
    in two planes, so its modes come in pairs that are one answer."""
    out = []
    for value in frequencies:
        if value <= 0.0:
            continue
        if not out or abs(value - out[-1]) > tolerance * max(out[-1], 1.0):
            out.append(float(value))
    return out


def _participation(model, shape):
    """How much of the shape lies in each of the six directions.

    Kept as six numbers rather than reduced to a winner, because the first
    three are meters and the last three radians: comparing across that
    divide is meaningless, and a helper that did it silently called a
    cantilever's first bending mode a rotation."""
    total = np.zeros(6)
    for i in range(model.num_nodes):
        total += np.abs(shape[6 * i:6 * i + 6])
    return total


def _dominant_translation(model, shape):
    """Which of X, Y, Z this shape mostly moves along."""
    return fem.DIRECTIONS[int(np.argmax(_participation(model, shape)[:3]))]


# ---- a drawn shape, made into a structure ---------------------------------


def cube(model=None, size=0.05):
    """A closed box of six quads — the smallest thing with a surface."""
    model = model or fem.Model('box')
    corner = {}
    n = 1
    for dx in (-1, 1):
        for dy in (-1, 1):
            for dz in (-1, 1):
                corner[(dx, dy, dz)] = n
                model.add_node(n, dx * size, dy * size, dz * size)
                n += 1
    for face in (((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)),
                 ((-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1)),
                 ((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1)),
                 ((1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)),
                 ((-1, -1, -1), (-1, -1, 1), (1, -1, 1), (1, -1, -1)),
                 ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1))):
        model.add_face([corner[c] for c in face])
    return model


MASSLESS = fem.Material('frame', youngs_modulus=70e9, density=0.0)
MEMBER = fem.Section.round_tube('member', outer=0.006, wall=0.001)


def test_wiring_a_surface_mesh_makes_a_structure():
    """The shortest road from a shape to a set of modes: a member on every
    edge, the mass over the nodes, and it stands up."""
    model = cube()
    assert model.wire_faces(MASSLESS, MEMBER) == 12, 'a cube has 12 edges'
    model.distribute_mass(0.500)
    shapes = model.eigensolution(num_modes=10)
    assert list(shapes.frequency[:6]) == [0.0] * 6
    assert shapes.frequency[6] > 1.0


def test_a_shared_edge_is_wired_once():
    """Six faces meeting on twelve edges, not twenty-four."""
    model = cube()
    model.wire_faces(MASSLESS, MEMBER)
    pairs = {frozenset((b.node_a, b.node_b)) for b in model.beams}
    assert len(pairs) == len(model.beams) == 12


def test_wiring_leaves_members_that_are_already_there():
    """So an explicit strut keeps its own section rather than being
    duplicated by a second one along the same edge."""
    model = cube()
    fat = fem.Section.rod('strut', 0.01)
    model.add_beam(1, 2, MASSLESS, fat)
    added = model.wire_faces(MASSLESS, MEMBER)
    assert added == 11
    along = [b for b in model.beams if {b.node_a, b.node_b} == {1, 2}]
    assert len(along) == 1 and along[0].section is fat


def test_the_mass_is_shared_by_what_each_node_carries():
    """Shares follow the members meeting at a node, not the node count.

    On a cube every corner carries the same three edges, so the shares
    come out equal there — the point is that they are *computed* from the
    members. Sharing by node count put 42% of the drone's mass into its
    propellers, which need the finest mesh to look right and so collect
    the most nodes."""
    model = cube()
    model.wire_faces(MASSLESS, MEMBER)
    shares = model.distribute_mass(0.800)
    assert sum(shares.values()) == pytest.approx(0.800)
    assert np.allclose(list(shares.values()), 0.800 / model.num_nodes)
    assert model.total_mass == pytest.approx(0.800)


def test_an_uneven_mesh_shares_its_mass_unevenly():
    """The whole reason for weighting: refine one end of a bar and it must
    not become heavier than the other."""
    model = fem.Model()
    for i in range(9):
        model.add_node(i + 1, i * 0.05, 0.0, 0.0)     # fine half
    model.add_node(20, 0.8, 0.0, 0.0)                 # then one long span
    model.add_chain(list(range(1, 10)) + [20], MASSLESS, MEMBER)
    shares = model.distribute_mass(1.0)
    assert sum(shares.values()) == pytest.approx(1.0)
    # the node on the long span carries half of a 400 mm member; one in the
    # fine half carries half of two 50 mm members, so exactly four times less
    assert shares[20] == pytest.approx(4 * shares[5])
    mass, _ = model.matrices()
    assert np.all(np.diag(mass)[0::6] > 0), 'translations'
    assert np.all(np.diag(mass)[3::6] > 0), 'rotations'


def test_sharing_the_mass_again_replaces_rather_than_adds():
    model = cube()
    model.wire_faces(MASSLESS, MEMBER)
    model.distribute_mass(0.500)
    model.distribute_mass(0.500)
    assert model.total_mass == pytest.approx(0.500)


def test_pieces_finds_what_is_not_joined():
    """One piece is a structure; more is that many free bodies, each with
    its own six zero-frequency modes."""
    model = cube()
    model.wire_faces(MASSLESS, MEMBER)
    assert len(model.pieces()) == 1
    apart = cube()
    for offset, node in enumerate(range(101, 109)):
        apart.add_node(node, 1.0 + offset * 0.01, 0.0, 0.0)
    apart.add_chain(range(101, 109), MASSLESS, MEMBER)
    apart.wire_faces(MASSLESS, MEMBER)
    assert [len(p) for p in apart.pieces()] == [8, 8]


def test_from_geometry_turns_any_drawn_shape_into_modes():
    """It is not told what the shape is — only that its faces have edges."""
    drawn = cube().geometry()
    model = fem.Model.from_geometry(drawn, MASSLESS, MEMBER, total_mass=0.5)
    assert model.num_nodes == 8
    assert len(model.beams) == 12
    assert model.total_mass == pytest.approx(0.5)
    shapes = model.eigensolution(num_modes=8)
    assert list(shapes.frequency[:6]) == [0.0] * 6


def test_from_geometry_refuses_a_shape_in_pieces():
    """Silently returning twelve zero modes instead of six is the trap;
    the plate geometry is genuinely three pieces, and this is what
    says so rather than leaving it to be noticed."""
    model = cube()
    for offset, node in enumerate(range(101, 109)):
        model.add_node(node, 1.0 + offset * 0.01, 0.0, 0.0)
    model.add_chain(range(101, 109), MASSLESS, MEMBER)
    with pytest.raises(ValueError, match='disconnected pieces'):
        fem.Model.from_geometry(model.geometry(), MASSLESS, MEMBER)


def test_from_geometry_refuses_a_node_joined_to_nothing():
    model = cube()
    model.add_node(99, 5.0, 5.0, 5.0)
    with pytest.raises(ValueError, match='connected to nothing'):
        fem.Model.from_geometry(model.geometry(), MASSLESS, MEMBER)


def test_from_geometry_refuses_a_shape_with_no_connectivity():
    bare = fem.Model()
    for i in range(4):
        bare.add_node(i + 1, i * 0.1, 0.0, 0.0)
    with pytest.raises(ValueError, match='nothing to connect'):
        fem.Model.from_geometry(bare.geometry(), MASSLESS, MEMBER)


STIFF = fem.Section.round_tube('blade', outer=0.020, wall=0.004)


def two_block_strip():
    """Three quads in a row: two in 'frame', the last in 'prop'.

    The smallest mesh with a *seam* — nodes 5 and 6 are on the boundary
    between the two blocks — which is what every rule below turns on.
    The drone is this shape with a thousand more nodes, and pinning the
    rules here rather than there is the difference between a tenth of a
    second and a minute.

        1---3---5---7
        |   |   |   |     frame | frame | prop
        2---4---6---8

    The prop quad is listed **first**, so that at the seam the block
    claiming a node first and the block holding most of it are different
    answers. Drawn in reading order they agree, and a fixture where the
    two rules agree cannot fail the wrong one — which is how this was
    written the first time.
    """
    return Geometry(
        node_id=[1, 2, 3, 4, 5, 6, 7, 8],
        node_xyz=[[0, 0, 0], [0, 0.1, 0], [0.1, 0, 0], [0.1, 0.1, 0],
                  [0.2, 0, 0], [0.2, 0.1, 0], [0.3, 0, 0], [0.3, 0.1, 0]],
        elem_id=[12, 10, 11],
        elem_conn=[[5, 7, 8, 6], [1, 3, 4, 2], [3, 5, 6, 4]],
        elem_type=[44, 44, 44],
        elem_block=[2, 1, 1],
        block_id=[1, 2],
        block_name=['frame', 'prop'],
        length_unit='m')


def test_a_member_takes_its_section_from_the_block_of_its_element():
    """Not from labels on its end nodes. A node on a seam belongs to two
    parts and can answer for only one, so a node-based rule gives the
    edges touching the seam the wrong section — which is exactly how 24
    of the drone's blade members came back as ordinary frame, moving
    every mode by a couple of percent and failing nothing."""
    model = fem.Model.from_geometry(two_block_strip(), MASSLESS, MEMBER,
                                    total_mass=1.0,
                                    sections={'prop': STIFF})
    stiff = {frozenset((b.node_a, b.node_b)) for b in model.beams
             if b.section.name == 'blade'}
    assert stiff == {frozenset(e) for e in ((5, 7), (7, 8), (8, 6), (6, 5))}, (
        'the prop quad\'s own edges — including the two that touch the '
        'seam, whose end nodes are labeled frame')
    # 5-6 is in both quads and can only be one member, so it keeps the
    # section of whichever element wired it first — the prop here. An
    # edge, unlike a node, is at least never split between two answers.
    assert any(b.section.name == 'blade' for b in model.beams
               if {b.node_a, b.node_b} == {5, 6})
    assert len(model.beams) == 10, 'and the shared edges are wired once'


def test_a_seam_node_belongs_to_the_block_holding_most_of_its_elements():
    """The label names the part a node is *in* — what a mode is read
    against and where a sensor goes. First claim rather than a majority
    emptied the parts that sit between others: the drone's canopy is
    ringed by six neighbors and came back with a single node in it."""
    model = fem.Model.from_geometry(two_block_strip(), MASSLESS, MEMBER,
                                    total_mass=1.0)
    groups = {node: model.group(node) for node in model.node_ids}
    assert groups[1] == groups[3] == 'frame', 'wholly inside one block'
    assert groups[7] == groups[8] == 'prop'
    assert groups[5] == groups[6] == 'frame', (
        'the seam: two frame elements against one prop')


def test_a_saved_geometry_alone_rebuilds_the_model(tmp_path):
    """The geometry is the whole description — sections come back off the
    element blocks, with nothing passed alongside the file. A side channel
    would not survive being saved, which is the point of blocks riding it.

    The matrices rather than the modes: they are what the modes come out
    of, so equal ones say the same thing about every degree of freedom
    instead of about the modes under some ceiling, and they cost a
    thousandth of what two eigensolutions do.
    """
    from visualdynamics.io import load, save

    drawn = two_block_strip()
    built = fem.Model.from_geometry(drawn, MASSLESS, MEMBER, total_mass=1.0,
                                    sections={'prop': STIFF})
    path = tmp_path / 'strip.vdyn'
    save(drawn, path)
    rebuilt = fem.Model.from_geometry(load(path), MASSLESS, MEMBER,
                                      total_mass=1.0,
                                      sections={'prop': STIFF})
    assert rebuilt.dof_strings() == built.dof_strings(), (
        'a different DOF order would make the matrices differ by a '
        'permutation and say nothing')
    for name, mine, theirs in zip(('mass', 'stiffness'),
                                  built.matrices(), rebuilt.matrices()):
        assert np.array_equal(mine, theirs), (
            f'the {name} matrix came back different: '
            f'{np.abs(mine - theirs).max():.3g} at worst')


def test_a_geometry_with_no_blocks_still_rebuilds_as_one_part():
    """Most formats do not record the question, so the mesh arrives as one
    unnamed block — and the model must still build, every member on the
    one section and every node unlabeled. Nothing invents a division."""
    strip = two_block_strip()
    plain = Geometry(node_id=strip.node_id, node_xyz=strip.node_xyz,
                     elem_id=strip.elem_id, elem_conn=strip.elem_conn,
                     elem_type=strip.elem_type, length_unit='m')
    assert plain.block_name == [''], 'the fixture is the case'
    model = fem.Model.from_geometry(plain, MASSLESS, MEMBER, total_mass=1.0,
                                    sections={'prop': STIFF})
    assert {b.section.name for b in model.beams} == {'member'}
    assert {model.group(n) for n in model.node_ids} == {''}


def test_from_geometry_can_wire_tracelines_for_a_wireframe():
    """A geometry with no elements at all has only its display lines, and
    for a bare wireframe those are the connectivity."""
    wire = fem.Model()
    for i in range(6):
        wire.add_node(i + 1, i * 0.1, 0.0, 0.0)
    drawn = Geometry(node_id=list(range(1, 7)),
                     node_xyz=[[i * 0.1, 0.0, 0.0] for i in range(6)],
                     traceline_conn=[list(range(1, 7))], length_unit='m')
    with pytest.raises(ValueError, match='nothing to connect'):
        fem.Model.from_geometry(drawn, MASSLESS, MEMBER)
    model = fem.Model.from_geometry(drawn, MASSLESS, MEMBER, total_mass=1.0,
                                    tracelines=True)
    assert len(model.beams) == 5
    assert len(model.pieces()) == 1


# ---- the plate element ----------------------------------------------------

ALUMINUM = fem.Material('aluminum', youngs_modulus=70e9, density=2700,
                        poissons_ratio=0.3)


def square_plate(mesh, side=1.0, thickness=0.01, material=ALUMINUM):
    """A free square plate meshed `mesh` x `mesh`, nodes from 1."""
    model = fem.Model('plate')
    for j in range(mesh + 1):
        for i in range(mesh + 1):
            model.add_node(1 + j * (mesh + 1) + i,
                           i * side / mesh, j * side / mesh, 0.0)
    for j in range(mesh):
        for i in range(mesh):
            n = 1 + j * (mesh + 1) + i
            model.add_plate((n, n + 1, n + mesh + 2, n + mesh + 1),
                            material, thickness)
    return model


def plate_lambdas(shapes, side, thickness, material, count):
    """The classical nondimensional frequencies, lambda^2 = w a^2
    sqrt(rho h / D), for the first `count` elastic modes."""
    d = (material.youngs_modulus * thickness ** 3
         / (12.0 * (1.0 - material.poissons_ratio ** 2)))
    elastic = shapes.frequency[shapes.frequency > 0.0][:count]
    return (elastic * 2.0 * math.pi * side ** 2
            * math.sqrt(material.density * thickness / d))


#: the completely free square plate's first six elastic lambda^2 at
#: nu = 0.3 — the classical benchmark (Leissa, NASA SP-160, with the
#: later converged values agreeing to the digits given here)
FREE_SQUARE_PLATE = [13.468, 19.596, 24.271, 34.801, 34.801, 61.093]


def test_free_square_plate_matches_the_classical_values():
    """Six rigid modes exactly, and the elastic ones on the published
    free-plate benchmark — from above, because a conforming
    consistent-mass model is a Rayleigh-Ritz reduction."""
    shapes = square_plate(12).eigensolution(num_modes=12)
    assert int(np.sum(shapes.frequency == 0.0)) == 6, (
        'a free plate has six rigid modes, no more: a seventh is the '
        'drilling mechanism')
    lambdas = plate_lambdas(shapes, 1.0, 0.01, ALUMINUM, 6)
    # the allowance grows with the mode: discretization error runs as
    # h^2 times the mode's spatial frequency squared, and the measured
    # errors at this mesh are 0.3/0.8/1.3/1.0/1.0/4.0 percent
    for computed, exact, room in zip(lambdas, FREE_SQUARE_PLATE,
                                     (1.01, 1.02, 1.02, 1.02, 1.02, 1.05)):
        assert exact <= computed <= exact * room, (
            f'lambda^2 {computed:.3f} against the classical {exact}')


def test_refining_the_plate_mesh_closes_on_the_classical_values():
    coarse = plate_lambdas(square_plate(8).eigensolution(num_modes=12),
                           1.0, 0.01, ALUMINUM, 6)
    fine = plate_lambdas(square_plate(16).eigensolution(num_modes=12),
                         1.0, 0.01, ALUMINUM, 6)
    assert np.all(fine < coarse), 'convergence is from above'
    assert np.all(fine <= np.array(FREE_SQUARE_PLATE)
                  * (1.005, 1.01, 1.01, 1.01, 1.01, 1.03))


def test_plate_frequencies_do_not_depend_on_orientation():
    """The same plate stood on edge and yawed answers the same — the
    local-frame transform, checked the way the beams check theirs."""
    flat = square_plate(6).eigensolution(num_modes=10)
    tilted = fem.Model('tilted')
    spin = np.array([[0.36, -0.48, 0.8],
                     [0.8, 0.6, 0.0],
                     [-0.48, 0.64, 0.6]])
    for j in range(7):
        for i in range(7):
            point = spin @ np.array([i / 6.0, j / 6.0, 0.0]) + 2.5
            tilted.add_node(1 + j * 7 + i, *point)
    for j in range(6):
        for i in range(6):
            n = 1 + j * 7 + i
            tilted.add_plate((n, n + 1, n + 8, n + 7), ALUMINUM, 0.01)
    assert np.allclose(tilted.eigensolution(num_modes=10).frequency,
                       flat.frequency, rtol=1e-8, atol=1e-6)


def test_a_skewed_quad_is_refused():
    model = fem.Model('skewed')
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.3, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match='rectangle'):
        model.add_plate((1, 2, 3, 4), ALUMINUM, 0.01)


def test_a_warped_quad_is_refused():
    model = fem.Model('warped')
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.2)
    model.add_node(4, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match='rectangle'):
        model.add_plate((1, 2, 3, 4), ALUMINUM, 0.01)


def test_plate_structural_mass_is_density_times_volume():
    model = square_plate(4, side=2.0, thickness=0.05)
    assert model.structural_mass == pytest.approx(2700 * 2.0 * 2.0 * 0.05)


def test_plate_rigid_modes_are_exactly_zero():
    shapes = square_plate(4).eigensolution(num_modes=8)
    assert np.all(shapes.frequency[:6] == 0.0), (
        'rigid motion strains neither bending, shear, membrane nor '
        'the drilling tie — exactly, not approximately')


def test_the_plate_appears_in_the_geometry_as_a_structural_quad():
    geometry = square_plate(2).geometry()
    assert list(geometry.elem_type) == [44] * 4
    assert len(geometry.elem_conn) == 4


def test_drilling_artifacts_live_far_above_the_physical_band():
    """The drilling DOF's penalty-against-inertia modes are artifacts,
    and artifacts must not sit where a user can solve: at full rotary
    inertia they walled up at 4.2 kHz on the demonstration plate,
    mid-band. A theta-z-dominated mode below 50 kHz is a regression."""
    shapes = square_plate(6, thickness=0.0127).eigensolution()
    found = 0
    for frequency, mode in zip(shapes.frequency, shapes.shape_matrix):
        spin = np.abs(mode[5::6]).max()
        rest = np.abs(np.concatenate(
            [mode[d::6] for d in range(5)])).max()
        # a real mode's theta-z *follows* its displacement field (the
        # ratio runs 5-20 with the tiny drilling inertia); an artifact
        # is theta-z alone, orders of magnitude clear of everything
        if spin > 1000.0 * rest:
            found += 1
            assert frequency > 50e3, (
                f'drilling artifact at {frequency:.0f} Hz')
    assert found > 0, ('no pure-drilling modes at all: the classifier '
                       'is not seeing them, so it guards nothing')
