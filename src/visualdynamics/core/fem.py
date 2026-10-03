"""A beam finite element model, and the eigensolution it gives.

visualdynamics is an analysis toolset, so this is deliberately the smallest
modeling capability that produces something worth analyzing: three-dimensional
two-node beams, flat-shell plates — four-node rectangles (MITC4) and
three-node triangles (MITC3) — and lumped masses, assembled into mass and
stiffness matrices and solved for real normal modes. It exists because a demonstration needs a
*truth* model — a dense analytical answer the measured one can be compared
against — and because building one should not require reaching for another
package.

There are three ways in. Build a structure member by member, as the
example below does. Give a geometry's blocks their properties — a material
and a thickness for a block of plates, a material and a section for a
block of beams (`BlockProperties`) — and let `from_geometry` build every
element as the element it is, which is how an exodus file means a
structure and the way a user builds a model of a simple article. Or draw
a shape as a surface mesh and let `from_geometry`, given one material and
one section, put a member along every edge of it and share the mass over
its nodes: the shorter road from any geometry to *some* set of modes, and
how `visualdynamics.demo.drone` is built throughout.

What it is not: a general finite element code. The plate is rectangular
only — a skewed or warped quad is refused rather than solved badly — and
there are no curved shells, no solids, no constraints beyond fixing degrees
of freedom, and no static solution. Wiring a surface mesh's edges with
`from_geometry` makes a grillage of beams, not plates, which is a real
modeling choice with a known cost rather than an approximation hidden
inside an element: it answers what order of mode density and what mode
families a shape has, and it does not pretend to be shell theory.

Everything here is SI, because the mass and stiffness matrices are the one
place in visualdynamics where several dimensions have to be consistent with each
other at once — a length in millimeters beside a modulus in pascals is not
wrong in any single entry, it is wrong only in the answer. The objects that
come out (`Geometry`, `ShapeSet`) carry their units declared, so the
conversion happens once, at the boundary, as it does everywhere else.

    from visualdynamics import fem

    aluminum = fem.Material('aluminum', youngs_modulus=70e9, density=2700)
    tube = fem.Section.round_tube('16 mm tube', outer=0.016, wall=0.001)

    model = fem.Model('cantilever')
    for i in range(11):
        model.add_node(100 + i, i * 0.1, 0.0, 0.0)
    model.add_chain(range(100, 111), aluminum, tube)
    shapes = model.eigensolution(maximum_frequency=2000,
                                 fixed=['100'], damping=0.01)

The element formulation is Euler-Bernoulli with a consistent mass matrix:
no shear flexibility and no section rotary inertia, which is right while a
member is slender (length more than about ten times its depth) and
progressively optimistic when it is not. A drone arm at 180 mm long and
16 mm deep is comfortably inside that; a stubby mounting stub is not, and
frequencies there read high.

References
----------
The Euler-Bernoulli beam element and its consistent mass matrix are
textbook; these are where they are set out.

1. Przemieniecki, J. S. (1968). *Theory of Matrix Structural
   Analysis*. McGraw-Hill.
2. Cook, R. D., Malkus, D. S., Plesha, M. E., & Witt, R. J. (2002).
   *Concepts and Applications of Finite Element Analysis*, 4th ed.
   Wiley. The cubic Hermite shape functions the consistent mass
   follows from, and why consistent rather than lumped changes the
   frequencies returned.
3. Craig, R. R., & Kurdila, A. J. (2006). *Fundamentals of Structural
   Dynamics*, 2nd ed. Wiley. The generalized symmetric eigenproblem
   this solves, and normalization to unit modal mass.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from .data import direction_code
from .geometry import ELEMENT_TYPES, Geometry
from .progress import Ticker
from .shapes import ShapeSet

#: The six degrees of freedom every node carries, in order. A beam
#: transmits moments, so rotations are structural here rather than a
#: bookkeeping convenience — leaving them out would model pin joints.
DIRECTIONS = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')

#: When a mode's strain energy is called zero, so its frequency is set to
#: zero exactly rather than left at the 1e-4 Hz round-off gives it.
#: Downstream code distinguishes a rigid mode by `frequency == 0` — the FRF
#: synthesis cancels its 0/0 that way — so 'very small' would arrive as a
#: resonance a hair above DC instead.
#:
#: The measure is how much cancellation happened: phi^T K phi against the
#: same sum with every sign removed. A motion that strains nothing is a
#: total cancellation of large terms and lands at 1e-16; a real mode, even
#: the softest one a fine mesh has, keeps a part in a million of what it
#: adds up. Six orders of margin either side, and no knowledge of how stiff
#: the structure is.
#:
#: Projecting onto the analytically known rigid-body vectors was tried
#: first and is wrong: a cantilever's first bending mode is 99.04% rigid
#: translation in the mass metric, because that is what a cantilever mode
#: looks like. Being *shaped* like a rigid motion and *being* one are
#: different questions, and only the energy asks the second.
ZERO_ENERGY = 1e-12

#: past this many degrees of freedom `Model.eigensolution` solves sparse
#: rather than dense: the dense solve of a 662-node plate model (3972
#: DOF) already took ten seconds and 2.2 GB, peaking near ten times its
#: two matrices (2026-09-26)
SPARSE_ABOVE = 3000

#: the element type a solid is written back as, by its node count
#: (`Model.geometry`): hex8, wedge6, tet4 in `ELEMENT_TYPES`
SOLID_CODES = {8: 115, 6: 112, 4: 111}

#: the sparse solver polishes what Lanczos hands it until every mode
#: asked for has a relative residual below this (`polish_modes`), or a
#: step stops helping, or `POLISH_STEPS` have been spent. The residual
#: floors where the conditioning says it must — the rigid-link test
#: model, scaled to a condition number of 4e9, floors at 3e-7 on this
#: machine's builds — so the polish stops at the floor rather than
#: spending its steps there, and the threshold is what a clean start
#: passes on the way down. A runner's build once handed it a start it
#: did not clean in the three steps that used to be fixed (2026-09-30,
#: Python 3.12 on GitHub: the twelfth mode of that model, 1.45 % from
#: its neighbor, came back with a MAC of 1 - 2.6e-8 against the dense
#: solver's, while every build here gave 1 - 1e-14; a start corrupted
#: by a part in a thousand reproduces it, 3e-4 after three steps and the
#: floor after eight). A residual is measured; a step count is hoped for.
POLISH_RESIDUAL = 1e-8
POLISH_STEPS = 12
#: modes asked of Lanczos beyond the ones wanted. Block inverse
#: iteration cleans a wanted mode of the modes *outside* its block at
#: the rate (lambda_k - sigma) / (lambda_outside - sigma) per step, so
#: the highest wanted mode, whose nearest outside neighbor may be a
#: percent away, would barely move without a band of unwanted modes
#: above it to widen that gap. Eight is cheap: Lanczos computes them
#: nearly for free and the polish is a few extra solves.
#: the eigen stage's length on the bar is an estimate — shift-invert
#: Lanczos solves with the factor a few times per mode it is asked
#: for — and the bar is kept short of full until the stage ends
#: (`Ticker.extend_to`)
EIGEN_SOLVES_PER_MODE = 6
LANCZOS_GUARD = 8


#: how many elements of one family are rotated and scattered at once:
#: enough that the per-chunk cost vanishes, few enough that a fine mesh's
#: (E, 24, 24) arrays stay in the tens of megabytes
ASSEMBLY_CHUNK = 4096


@dataclass(frozen=True)
class Material:
    """An isotropic elastic material.

    Shear modulus is derived from the modulus and Poisson's ratio unless it
    is given: for carbon fiber laminates the isotropic relation is a poor
    guess and the torsional stiffness it implies can be out by a factor of
    two, so the door is left open to state it.
    """

    name: str
    youngs_modulus: float          #: Pa
    density: float                 #: kg/m^3
    poissons_ratio: float = 0.3
    modulus_of_rigidity: float | None = None    #: Pa, derived when None

    @property
    def is_rigid(self) -> bool:
        """Whether this is `RIGID`: a link, not a material. A modulus
        left blank in the Blocks table (None) is a material not yet
        finished, not a link."""
        return (self.youngs_modulus is not None
                and math.isinf(self.youngs_modulus))

    @property
    def shear_modulus(self) -> float:
        if self.modulus_of_rigidity is not None:
            return float(self.modulus_of_rigidity)
        return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))


#: US handbooks state a modulus in psi and a density in lb/in^3, and the
#: library keeps their numbers rather than a rounding of a conversion —
#: the same choice the demonstration plate made for its 6061-T6
PSI = 6894.757293168361                   #: Pa
LB_PER_IN3 = 27679.90471020312            #: kg/m^3


@dataclass(frozen=True)
class LibraryMaterial:
    """A material the library offers, with where its numbers came from.

    The note names the kind of source and what the values are typical
    of, because a handbook's typical modulus and density are what a
    modal solution wants and also not what a particular heat of a
    particular alloy measures: a person with the part's certification
    in hand types those numbers over the library's.
    """

    material: Material
    note: str


def _handbook(name, modulus_msi, density_lb_in3, poissons_ratio, note):
    return LibraryMaterial(
        Material(name, youngs_modulus=modulus_msi * 1e6 * PSI,
                 density=density_lb_in3 * LB_PER_IN3,
                 poissons_ratio=poissons_ratio), note)


_METALS = 'typical room-temperature handbook values for the wrought alloy'
_PLASTIC = ('typical room-temperature values; a plastic varies by grade '
            'and supplier more than a metal does')

#: The materials the Blocks table offers and `material()` answers to:
#: common structural alloys and a few plastics, each as the handbooks
#: state it. Typical values, to be checked against the part's own
#: certification when it matters; every entry says so in its note.
MATERIAL_LIBRARY: tuple[LibraryMaterial, ...] = (
    _handbook('6061-T6', 10.0, 0.098, 0.33, _METALS + ' (aluminum)'),
    _handbook('7075-T6', 10.4, 0.101, 0.33, _METALS + ' (aluminum)'),
    _handbook('2024-T3', 10.6, 0.100, 0.33, _METALS + ' (aluminum)'),
    _handbook('1018 steel', 29.0, 0.284, 0.29, _METALS + ' (carbon steel)'),
    _handbook('A36 steel', 29.0, 0.284, 0.26, _METALS + ' (carbon steel)'),
    _handbook('4130 steel', 29.7, 0.283, 0.29, _METALS + ' (alloy steel)'),
    _handbook('304 stainless', 28.0, 0.289, 0.29, _METALS + ' (austenitic)'),
    _handbook('316 stainless', 28.0, 0.289, 0.27, _METALS + ' (austenitic)'),
    _handbook('17-4 PH stainless', 28.5, 0.282, 0.27,
              _METALS + ' (precipitation hardening, H900)'),
    _handbook('Ti-6Al-4V', 16.5, 0.160, 0.342, _METALS + ' (titanium)'),
    _handbook('AZ31B magnesium', 6.5, 0.0639, 0.35, _METALS + ' (magnesium)'),
    _handbook('C26000 brass', 16.0, 0.308, 0.375, _METALS + ' (cartridge brass)'),
    _handbook('C11000 copper', 17.0, 0.323, 0.33, _METALS + ' (ETP copper)'),
    _handbook('Inconel 718', 29.0, 0.296, 0.29, _METALS + ' (nickel superalloy)'),
    _handbook('acrylic (PMMA)', 0.45, 0.0426, 0.37, _PLASTIC),
    _handbook('polycarbonate', 0.35, 0.0433, 0.37, _PLASTIC),
    _handbook('ABS', 0.33, 0.038, 0.35, _PLASTIC),
    _handbook('nylon 6/6', 0.41, 0.0412, 0.39, _PLASTIC),
)

#: A block of two-node lines made of this is a set of rigid, massless
#: links (Brandon, 2026-09-26): each joins two nodes so that one moves as
#: the other plus its rotation about it, and adds no mass. What a bolt
#: joining two plates whose mid-surfaces do not meet is, in a model built
#: from planes. Not a material with a very large modulus — that makes the
#: stiffness matrix ill-conditioned and the higher modes wrong — but an
#: exact constraint the eigensolution eliminates (`Model.add_rigid_link`).
RIGID = Material('rigid (massless)', youngs_modulus=math.inf, density=0.0,
                 poissons_ratio=0.0)

#: the library by name, for a lookup — and the rigid link, which is not a
#: handbook material but is picked where one is
MATERIALS: dict[str, Material] = {entry.material.name: entry.material
                                  for entry in MATERIAL_LIBRARY}
MATERIALS[RIGID.name] = RIGID


def material(name: str) -> Material:
    """A library material by name — `material('6061-T6')` — the same
    entry the Blocks table's Material drop-down fills a row from.

    Raises `KeyError` naming the library when the name is not in it,
    so a typo reads as one rather than as a missing material.
    """
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f'{name!r} is not in the material library; it '
                       f'has {", ".join(MATERIALS)}') from None


@dataclass(frozen=True)
class Section:
    """A beam cross section, as the four numbers the element needs.

    `iy` and `iz` are second moments of area about the element's own local
    y and z axes, and `j` is the torsion constant — St Venant's, which
    equals the polar second moment only for a circular section. For
    anything else it is smaller, and using the polar value overstates
    torsional stiffness; the constructors below carry the right formula for
    the shapes they build.

    **The shapes** (`SHAPES`): a section built from one remembers it —
    `shape` and its `dimensions` in meters, in the order the constructor
    takes them — so a table can show the dimensions and a file or a
    session script rebuilds the section from them. One rule for which way
    a shape faces: its width or flanges run along local **y** and its
    depth or height along local **z**, so the orientation vector (which
    names local y) points across the flanges, and `iy` is the
    strong-axis bending of an I-beam or a channel. An angle is the
    exception, because its leg axes are not principal and the element has
    no product term: its `iy` and `iz` are its principal moments, and the
    orientation vector points along the major principal axis
    (`angle_major_axis` says where that is relative to the legs).

    The element assumes the shear center is the centroid. For the
    symmetric shapes it is; a channel's and an angle's are not, and the
    twisting a load through their centroid causes is not in the model.

    `polar` is the *inertia* term, always the true polar second moment
    (iy + iz), because rotary inertia about the axis is a property of where
    the material is and has nothing to do with warping.
    """

    name: str
    area: float                    #: m^2
    iy: float                      #: m^4, bending about local y
    iz: float                      #: m^4, bending about local z
    j: float                       #: m^4, St Venant torsion constant
    shape: str = ''                #: one of `SHAPES`, '' for four numbers
    dimensions: tuple[float, ...] = ()   #: m, in the constructor's order

    @property
    def polar(self) -> float:
        return self.iy + self.iz

    @classmethod
    def of_shape(cls, name: str, shape: str,
                 dimensions: Sequence[float]) -> Section:
        """The section of a named shape (`SHAPES`), given its dimensions
        in meters in the order the shape's constructor takes them.

        Parameters
        ----------
        name : str
            What to call the section.
        shape : str
            A key of `SHAPES`.
        dimensions : sequence of float
            The shape's dimensions, in meters.

        Returns
        -------
        Section
        """
        if shape not in SHAPES:
            raise ValueError(f'{shape!r} is not a section shape: '
                             + ', '.join(SHAPES))
        method, names = SHAPES[shape]
        if len(dimensions) != len(names):
            raise ValueError(f'{with_article(shape)} takes {len(names)} '
                             'dimensions — ' + ', '.join(names))
        return getattr(cls, method)(name, *(float(v) for v in dimensions))

    @classmethod
    def round_tube(cls, name: str, outer: float, wall: float) -> Section:
        """A circular tube, given its outside diameter and wall thickness."""
        ro, ri = outer / 2.0, outer / 2.0 - wall
        if ri < 0 or wall <= 0 or outer <= 0:
            raise ValueError(f'{name}: a round tube needs a wall between 0 '
                             'and the radius')
        area = math.pi * (ro ** 2 - ri ** 2)
        i = math.pi * (ro ** 4 - ri ** 4) / 4.0
        # a closed circular section is the one case where torsion is the
        # polar moment exactly: it does not warp
        return cls(name, area, i, i, 2.0 * i, 'round tube',
                   (float(outer), float(wall)))

    @classmethod
    def rod(cls, name: str, diameter: float) -> Section:
        """A solid circular rod."""
        section = cls.round_tube(name, diameter, diameter / 2.0)
        return cls(name, section.area, section.iy, section.iz, section.j,
                   'rod', (float(diameter),))

    @classmethod
    def rectangle(cls, name: str, width: float, height: float) -> Section:
        """A solid rectangle, `width` along local y and `height` along z."""
        if width <= 0 or height <= 0:
            raise ValueError(f'{name}: a rectangle needs a width and a height')
        area = width * height
        iz = width ** 3 * height / 12.0     # bending in the local x-y plane
        iy = width * height ** 3 / 12.0     # bending in the local x-z plane
        long, short = max(width, height), min(width, height)
        # St Venant's constant for a solid rectangle, from the exact series
        # solution of the Prandtl stress function (Timoshenko & Goodier,
        # Theory of Elasticity, sec. 109). The odd terms fall as n^-5, so a
        # hundred of them leave under a part in 1e10. This replaced Roark's
        # closed approximation (2026-09-23), which had been described as
        # good to 0.1% and is 0.45% off at an aspect ratio of 1.2.
        tail = sum(math.tanh(n * math.pi * long / (2.0 * short)) / n ** 5
                   for n in range(1, 200, 2))
        j = long * short ** 3 / 3.0 * (
            1.0 - 192.0 / math.pi ** 5 * (short / long) * tail)
        return cls(name, area, iy, iz, j, 'rectangle',
                   (float(width), float(height)))

    @classmethod
    def rectangular_tube(cls, name: str, width: float, height: float,
                         wall: float) -> Section:
        """A rectangular tube of one wall thickness, `width` along local y
        and `height` along z, both outside dimensions."""
        inner_w, inner_h = width - 2.0 * wall, height - 2.0 * wall
        if wall <= 0 or inner_w <= 0 or inner_h <= 0:
            raise ValueError(f'{name}: wall {wall} closes the section')
        area = width * height - inner_w * inner_h
        iy = (width * height ** 3 - inner_w * inner_h ** 3) / 12.0
        iz = (width ** 3 * height - inner_w ** 3 * inner_h) / 12.0
        # Bredt's thin-wall formula, J = 4 A_m^2 t / s: A_m the area inside
        # the wall's centreline, s that centreline's length
        mean_w, mean_h = width - wall, height - wall
        j = 4.0 * (mean_w * mean_h) ** 2 * wall / (2.0 * (mean_w + mean_h))
        return cls(name, area, iy, iz, j, 'rectangular tube',
                   (float(width), float(height), float(wall)))

    @classmethod
    def square_tube(cls, name: str, width: float, wall: float) -> Section:
        """A square tube, given its outside width and wall thickness — a
        rectangular tube of equal sides."""
        return cls.rectangular_tube(name, width, width, wall)

    @classmethod
    def i_beam(cls, name: str, depth: float, flange_width: float,
               flange_thickness: float, web_thickness: float) -> Section:
        """A doubly symmetric I-beam: overall `depth` along local z,
        flanges `flange_width` wide along y. Fillets are left out, which
        puts area and torsion a few percent under a rolled shape's table
        values (a W8x31: 8.99 in² against 9.13, J 0.50 in⁴ against 0.54)."""
        d, bf, tf, tw = depth, flange_width, flange_thickness, web_thickness
        web = d - 2.0 * tf
        if min(d, bf, tf, tw) <= 0 or web <= 0 or tw > bf:
            raise ValueError(f'{name}: the flanges and web do not make an I')
        area = 2.0 * bf * tf + web * tw
        iy = (bf * d ** 3 - (bf - tw) * web ** 3) / 12.0
        iz = 2.0 * tf * bf ** 3 / 12.0 + web * tw ** 3 / 12.0
        # an open thin-walled section: the sum of b t^3 / 3 over its
        # plates, the web taken between the flanges' mid-planes
        j = (2.0 * bf * tf ** 3 + (d - tf) * tw ** 3) / 3.0
        return cls(name, area, iy, iz, j, 'I-beam',
                   (float(d), float(bf), float(tf), float(tw)))

    @classmethod
    def channel(cls, name: str, depth: float, flange_width: float,
                flange_thickness: float, web_thickness: float) -> Section:
        """A channel: overall `depth` along local z, the flanges
        `flange_width` wide (web included) running along +y from the web.
        `iz` is about the centroid, which sits off the web; the shear
        center sits further off it the other way, and the element does
        not know. The flanges are of one thickness: a rolled channel's
        taper toward their tips, and its table flange thickness is an
        average, so this `iz` runs high for one (a C6x10.5: 1.06 in⁴
        against the table's 0.86) — a bent-plate channel it gives
        exactly."""
        d, bf, tf, tw = depth, flange_width, flange_thickness, web_thickness
        web = d - 2.0 * tf
        if min(d, bf, tf, tw) <= 0 or web <= 0 or tw > bf:
            raise ValueError(f'{name}: the flanges and web do not make a '
                             'channel')
        flange_area, web_area = bf * tf, web * tw
        area = 2.0 * flange_area + web_area
        iy = (bf * d ** 3 - (bf - tw) * web ** 3) / 12.0
        # centroid along y, from the back of the web
        centroid = (2.0 * flange_area * bf / 2.0 + web_area * tw / 2.0) / area
        iz = (2.0 * (tf * bf ** 3 / 12.0
                     + flange_area * (bf / 2.0 - centroid) ** 2)
              + web * tw ** 3 / 12.0 + web_area * (tw / 2.0 - centroid) ** 2)
        j = (2.0 * bf * tf ** 3 + (d - tf) * tw ** 3) / 3.0
        return cls(name, area, iy, iz, j, 'channel',
                   (float(d), float(bf), float(tf), float(tw)))

    @classmethod
    def angle(cls, name: str, long_leg: float, short_leg: float,
              thickness: float) -> Section:
        """An angle of one thickness. Its leg axes are not principal and
        the element has no product term, so `iy` and `iz` are the
        principal moments — major and minor — and the orientation vector
        is to point along the major principal axis (`angle_major_axis`)."""
        a, b, t = long_leg, short_leg, thickness
        if min(a, b, t) <= 0 or t >= min(a, b) or b > a:
            raise ValueError(f'{name}: an angle needs a long leg, a short '
                             'leg no longer, and a thickness under both')
        area, (i_major, i_minor), _theta = _angle_properties(a, b, t)
        j = (a + b - t) * t ** 3 / 3.0
        return cls(name, area, i_major, i_minor, j, 'angle',
                   (float(a), float(b), float(t)))


def _angle_properties(a: float, b: float, t: float
                      ) -> tuple[float, tuple[float, float], float]:
    """(area, (I major, I minor), angle of the major axis) for an angle
    with its long leg `a` along y and short leg `b` along z from the
    heel, both `t` thick: two rectangles, their centroidal moments and
    product about the combined centroid, and the principal values of
    that tensor. The angle is in degrees from the long leg toward the
    short one."""
    # the long leg: y in [0, a], z in [0, t]; the short leg above it:
    # y in [0, t], z in [t, b]
    parts = [(a, t, a / 2.0, t / 2.0), (t, b - t, t / 2.0, t + (b - t) / 2.0)]
    area = sum(w * h for w, h, _y, _z in parts)
    cy = sum(w * h * y for w, h, y, _z in parts) / area
    cz = sum(w * h * z for w, h, _y, z in parts) / area
    iyy = sum(w * h ** 3 / 12.0 + w * h * (z - cz) ** 2
              for w, h, _y, z in parts)            # about y: z squared
    izz = sum(h * w ** 3 / 12.0 + w * h * (y - cy) ** 2
              for w, h, y, _z in parts)            # about z: y squared
    iyz = sum(w * h * (y - cy) * (z - cz) for w, h, y, z in parts)
    tensor = np.array([[iyy, -iyz], [-iyz, izz]])
    values, vectors = np.linalg.eigh(tensor)
    major = vectors[:, 1]                          # the larger moment's axis
    theta = math.degrees(math.atan2(major[1], major[0])) % 180.0
    return area, (float(values[1]), float(values[0])), theta


def with_article(shape: str) -> str:
    """'a channel', 'an I-beam', 'an angle' — a shape named in a
    sentence."""
    return ('an ' if shape[:1].lower() in 'aeio' or shape.startswith('I-')
            else 'a ') + shape


def angle_major_axis(long_leg: float, short_leg: float,
                     thickness: float) -> float:
    """Where an angle's major principal axis lies — the direction its
    orientation vector points — in degrees from the long leg, turning
    toward the short leg. 45 for an equal angle.

    Parameters
    ----------
    long_leg, short_leg, thickness : float
        The angle's dimensions, in any one unit.

    Returns
    -------
    float
        Degrees from the long leg, 0 to 180.
    """
    return _angle_properties(long_leg, short_leg, thickness)[2]


#: the shapes a section can be built from: {shape: (constructor, the
#: dimensions it takes, in order)} — what the Blocks table offers, and
#: what a file or a session script rebuilds a section from
SHAPES: dict[str, tuple[str, tuple[str, ...]]] = {
    'round tube': ('round_tube', ('outer diameter', 'wall')),
    'rod': ('rod', ('diameter',)),
    'rectangle': ('rectangle', ('width', 'height')),
    'rectangular tube': ('rectangular_tube', ('width', 'height', 'wall')),
    'I-beam': ('i_beam', ('depth', 'flange width', 'flange thickness',
                          'web thickness')),
    'channel': ('channel', ('depth', 'flange width', 'flange thickness',
                            'web thickness')),
    'angle': ('angle', ('long leg', 'short leg', 'thickness')),
}


@dataclass
class BlockProperties:
    """What a block of a geometry is made of, for `Model.from_geometry`.

    One property set per block of one element type, the way every
    finite element format states a structure: a material for any
    block, plus a thickness for a block of plates (triangles or
    quads) or a section — and an orientation vector for the roll, as
    `Model.add_beam` takes it — for a block of beams; a block of
    solids takes the material alone. A block given both, or plates or
    beams given neither, is refused when the model is built, by name.
    """

    material: Material
    thickness: float | None = None            #: m, for a block of plates
    section: Section | None = None            #: for a block of beams
    orientation: tuple[float, float, float] | None = None

    @property
    def kind(self) -> str:
        """'plate', 'beam', 'solid', 'rigid', or what is wrong with it.
        A rigid block takes no thickness and no section, and any left
        from before the material was picked are ignored; a material
        alone is a block of solids (2026-09-30), which take nothing
        else — a block of plates or beams given only a material is
        refused where the elements are built, by what they are."""
        if self.material.is_rigid:
            return 'rigid'
        if self.thickness is not None and self.section is None:
            return 'plate'
        if self.section is not None and self.thickness is None:
            return 'beam'
        if self.section is None and self.thickness is None:
            return 'solid'
        return 'both a thickness and a section'


@dataclass
class Beam:
    """One two-node beam element."""

    node_a: int
    node_b: int
    material: Material
    section: Section
    #: a vector defining the local x-y plane, so the section's iy and iz
    #: mean something on a member that is not round. None picks a default.
    orientation: tuple[float, float, float] | None = None
    color: int = 1
    group: str = ''


@dataclass
class Plate:
    """One four-node rectangular plate-bending element.

    A flat shell: plane-stress membrane action in its own plane and
    Mindlin bending out of it, with the transverse shear tied at the
    edge midpoints (MITC4). The tying is not optional finesse — a
    plain bilinear Mindlin element locks in shear as the plate gets
    thin, and a 12x12 mesh of the locked element puts the first
    elastic mode of a thin free plate several times too high.

    Rectangles only, and refused otherwise rather than silently
    mis-integrated: the tying directions assume the natural axes align
    with the sides, which is exactly true for a rectangle and only
    approximately for anything else. The models this module exists to
    build mesh rectangular panels; a skewed general quad earns its
    place when something needs it, with the covariant transforms and
    the validation that come with it.

    Nodes run around the perimeter: 1-2 is the first edge, 1-4 the
    second, corner 3 opposite corner 1.
    """

    nodes: tuple[int, int, int, int]
    material: Material
    thickness: float               #: m
    color: int = 1
    group: str = ''


@dataclass
class Triangle:
    """One three-node triangular plate-bending element.

    The quad's sibling, so a mesh of triangles and a mesh of rectangles
    are one theory: a flat shell with plane-stress membrane action in
    its own plane and Mindlin bending out of it, the transverse shear
    tied along the three edges (MITC3, Lee & Bathe 2004). The tying is
    what keeps a linear triangle from locking in shear as the plate
    gets thin — a plain linear Mindlin triangle is the worst locker
    there is. Any flat triangle is a valid element; only a degenerate
    one (zero area) is refused.

    Nodes run around the perimeter, counterclockwise about the normal
    the element takes as its own +z.
    """

    nodes: tuple[int, int, int]
    material: Material
    thickness: float               #: m
    color: int = 1
    group: str = ''


@dataclass
class Solid:
    """One solid element: a hexahedron, a wedge or a tetrahedron, told
    apart by how many nodes it names (8, 6 or 4).

    Three translations per node and no rotations — a solid has no
    rotational stiffness, and the rotations of a node only solids touch
    are grounded by the eigensolution rather than left as degrees of
    freedom with nothing on them (`Model.dangling_rotations`).

    The hexahedron is trilinear with Wilson's incompatible bending
    modes, Taylor's form (the extra modes' strains taken from the
    centroid's Jacobian, so a distorted brick still passes the patch
    test): a plain trilinear brick is far too stiff in bending, and a
    part meshed a few elements through its thickness would come out a
    third high. With the modes, one layer of bricks bends like a beam.
    The wedge is the linear six-node element and the tetrahedron the
    constant-strain one — transition and imported shapes, not what a
    part is meshed with here (`mesh.block` makes bricks); a linear
    tetrahedron locks in bending and a mesh of them is trusted only
    where it is fine.

    Nodes run around the bottom face and then the top, the same way
    round (UFF 2412 and Nastran's CHEXA/CPENTA/CTETRA order).
    """

    nodes: tuple[int, ...]
    material: Material
    color: int = 1
    group: str = ''


@dataclass
class RigidLink:
    """Two nodes held rigidly together, with no mass of their own: the
    second moves as the first does, translated by the first's rotation
    about it (`RIGID`)."""

    node_a: int
    node_b: int
    group: str = ''


@dataclass
class LumpedMass:
    """A rigid item carried at a node: a motor, a battery, a camera.

    The inertias are about the global axes through the node. A point mass
    leaves them zero, which is honest for something small against the
    members carrying it and wrong for a battery the size of the structure —
    a mass with no inertia cannot rock, so a rocking mode simply will not
    appear.
    """

    node: int
    mass: float                                     #: kg
    inertia: tuple[float, float, float] = (0.0, 0.0, 0.0)   #: kg m^2
    name: str = ''


@dataclass
class Face:
    """A cosmetic face: shading, not stiffness.

    A grillage of beams reads as a wireframe, and a wireframe of a drone
    deck reads as nothing much. Faces spanning nodes that are already
    there give the renderer something to shade and the animation something
    to deform, while contributing nothing to the matrices — which is
    exactly the truth about them, and is why they are a separate kind
    rather than a zero-stiffness element.
    """

    nodes: tuple[int, ...]
    color: int = 1
    group: str = ''


def connected_pieces(neighbors: dict[int, set[int]]) -> list[list[int]]:
    """The connected components of an adjacency map, largest first.

    One piece is a structure; more than one is that many free bodies,
    each bringing its own six zero-frequency modes. The flood fill is
    shared by the model (asking over its beams and plates) and the
    drone's drawing (asking over its face edges, before a model
    exists), so the two cannot disagree about what "joined" means.
    """
    seen: set[int] = set()
    found: list[list[int]] = []
    for start in neighbors:
        if start in seen:
            continue
        stack, piece = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            piece.append(node)
            stack.extend(neighbors[node] - seen)
        found.append(sorted(piece))
    return sorted(found, key=len, reverse=True)


def polish_modes(stiffness: Any, mass: Any, sigma: float, vectors: np.ndarray,
                 wanted: int, factor: Any = None, ticker: Any = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine a block of approximate modes until they are converged.

    Block inverse iteration with the shifted factor (K - sigma M), each
    step followed by Rayleigh-Ritz on the block — the small projected
    generalized problem solved densely — repeated until every one of the
    first `wanted` modes has a relative residual below `POLISH_RESIDUAL`,
    or a step no longer halves the worst of them (the floor the
    conditioning sets), or `POLISH_STEPS` are spent. The residual of a mode is
    |K v - lambda M v| over |(K - sigma M) v|, which is defined for a
    rigid-body mode too (its numerator and |K v| are both zero). The
    block is polished whole: the Ritz step resolves the modes *inside*
    it exactly, and iteration removes what leaks in from *outside*, at
    a rate set by the gap between the last wanted mode and the first
    beyond the block, which is why the caller hands over a guard band.

    Returns the eigenvalues, the mass-normal vectors, both for the whole
    block in ascending order, and the residual of each after the last
    step. Sparse `stiffness` and `mass` in CSC form.
    """
    from scipy.linalg import eigh as scipy_eigh
    from scipy.sparse.linalg import splu

    if factor is None:
        factor = splu((stiffness - sigma * mass).tocsc())
    wanted = min(int(wanted), vectors.shape[1])
    worst = math.inf
    if ticker is not None:
        ticker.add(POLISH_STEPS)
    for step in range(POLISH_STEPS):
        if ticker is not None:
            ticker.tick()
        if step:
            vectors = factor.solve(mass @ vectors)
        eigenvalues, ritz = scipy_eigh(vectors.T @ (stiffness @ vectors),
                                       vectors.T @ (mass @ vectors))
        vectors = vectors @ ritz
        k_v = stiffness @ vectors
        m_v = mass @ vectors
        residuals = (np.linalg.norm(k_v - m_v * eigenvalues, axis=0)
                     / np.linalg.norm(k_v - sigma * m_v, axis=0))
        before, worst = worst, float(residuals[:wanted].max())
        if worst < POLISH_RESIDUAL or worst > 0.5 * before:
            break
    return eigenvalues, vectors, residuals


class Model:
    """Nodes, beams and lumped masses, and the modes they imply.

    Two assemblies of one set of element matrices. Dense (`matrices`):
    (6 x nodes) square, every mode solved whole — exact, and the path for a
    model up to `SPARSE_ABOVE` degrees of freedom. Sparse
    (`sparse_matrices`): only the nonzeros, the lowest modes by
    shift-invert Lanczos. The dense path was the only one, and its ceiling
    (~1000 nodes) deliberate, until a plate model of a small real
    structure met it: the BARC at a quarter inch, 1,500 nodes, is ~13 GB
    dense; sparse, its eighth-inch mesh of 6,150 nodes solves in three
    seconds in under half a gigabyte (2026-09-26).

    Attributes:
        name: What the model is called; it becomes the geometry's name
            and the shape set's comment.
        length_unit: What the coordinates are in. Everything inside is
            SI; this is what the `Geometry` is told on the way out.
        beams: Every member, as `Beam` records naming two nodes, a
            material and a section.
        masses: Lumped masses, as `LumpedMass` records at a node.
        faces: Surfaces, as `Face` records naming three or four nodes.
            They carry no stiffness — a face is drawn, and its *edges*
            are what carry members.
    """

    def __init__(self, name: str = '', length_unit: str = 'm') -> None:
        self.name: str = name
        #: everything here is SI; this is what the Geometry is told
        self.length_unit: str = length_unit
        self._nodes: dict[int, np.ndarray] = {}
        self._node_group: dict[int, str] = {}
        self.beams: list[Beam] = []
        self.plates: list[Plate] = []
        self.triangles: list[Triangle] = []
        self.solids: list[Solid] = []
        self.rigid_links: list[RigidLink] = []
        self.masses: list[LumpedMass] = []
        self.faces: list[Face] = []

    # ---- building ---------------------------------------------------------

    def add_node(self, node_id: int, x: float, y: float, z: float,
                 group: str = '') -> int:
        node_id = int(node_id)
        if node_id in self._nodes:
            raise ValueError(f'node {node_id} is already in the model')
        self._nodes[node_id] = np.array([x, y, z], dtype=np.float64)
        self._node_group[node_id] = group
        return node_id

    def add_beam(self, node_a: int, node_b: int, material: Material,
                 section: Section,
                 orientation: Sequence[float] | None = None,
                 color: int = 1, group: str = '') -> Beam:
        for node in (node_a, node_b):
            if int(node) not in self._nodes:
                raise ValueError(f'beam names node {node}, which is not in the model')
        if int(node_a) == int(node_b):
            raise ValueError(f'beam from node {node_a} to itself has no length')
        beam = Beam(int(node_a), int(node_b), material, section,
                    None if orientation is None
                    else tuple(float(v) for v in orientation), color, group)
        self.beams.append(beam)
        return beam

    def add_chain(self, nodes: Sequence[int], material: Material,
                  section: Section,
                  orientation: Sequence[float] | None = None,
                  color: int = 1, group: str = '') -> list[Beam]:
        """A run of beams through consecutive nodes — a member, in one call."""
        nodes = [int(n) for n in nodes]
        return [self.add_beam(a, b, material, section, orientation, color, group)
                for a, b in pairwise(nodes)]

    def add_plate(self, nodes: Sequence[int], material: Material,
                  thickness: float, color: int = 1,
                  group: str = '') -> Plate:
        """One rectangular plate element over four existing nodes."""
        nodes = tuple(int(n) for n in nodes)
        if len(nodes) != 4 or len(set(nodes)) != 4:
            raise ValueError('a plate spans four distinct nodes')
        for node in nodes:
            if node not in self._nodes:
                raise ValueError(
                    f'plate names node {node}, which is not in the model')
        if float(thickness) <= 0.0:
            raise ValueError('a plate needs a positive thickness')
        # validate the rectangle at build time, not at solve time: the
        # person holding the bad corner coordinate is the one adding it
        _plate_frame(*[self._nodes[n] for n in nodes])
        plate = Plate(nodes, material, float(thickness), color, group)
        self.plates.append(plate)
        return plate

    def add_triangle(self, nodes: Sequence[int], material: Material,
                     thickness: float, color: int = 1,
                     group: str = '') -> Triangle:
        """One triangular plate element over three existing nodes."""
        nodes = tuple(int(n) for n in nodes)
        if len(nodes) != 3 or len(set(nodes)) != 3:
            raise ValueError('a triangle spans three distinct nodes')
        for node in nodes:
            if node not in self._nodes:
                raise ValueError(
                    f'triangle names node {node}, which is not in the model')
        if float(thickness) <= 0.0:
            raise ValueError('a triangle needs a positive thickness')
        _triangle_frame(*[self._nodes[n] for n in nodes])
        triangle = Triangle(nodes, material, float(thickness), color, group)
        self.triangles.append(triangle)
        return triangle

    def add_solid(self, nodes: Sequence[int], material: Material,
                  color: int = 1, group: str = '') -> Solid:
        """One solid element over eight, six or four existing nodes: a
        hexahedron, a wedge or a tetrahedron.

        Parameters
        ----------
        nodes : sequence of int
            The corners, bottom face then top, the same way round.
        material : Material
            What it is made of.
        color, group : optional
            As for a plate.

        Returns
        -------
        Solid
        """
        nodes = tuple(int(n) for n in nodes)
        if len(nodes) not in (8, 6, 4) or len(set(nodes)) != len(nodes):
            raise ValueError('a solid spans eight, six or four distinct nodes')
        for node in nodes:
            if node not in self._nodes:
                raise ValueError(
                    f'solid names node {node}, which is not in the model')
        if material.is_rigid:
            raise ValueError('a solid is made of a material, not rigid')
        # a flat element is caught here, by the person holding the bad
        # coordinate, not by the eigensolver. One numbered the other way
        # round (its volume negative) is the same element and is taken:
        # meshers disagree on the handedness, and Linderholt's frame
        # mesh arrived inside out to this convention (2026-09-30)
        xyz = np.array([self._nodes[n] for n in nodes])
        if abs(_solid_volume(xyz)) <= 1e-12 * float(np.ptp(xyz)) ** 3:
            raise ValueError(f'solid {nodes} has no volume')
        solid = Solid(nodes, material, color, group)
        self.solids.append(solid)
        return solid

    def add_rigid_link(self, node_a: int, node_b: int,
                       group: str = '') -> RigidLink:
        """Join two nodes rigidly, adding no mass.

        Links that share nodes join into one rigid body, however they are
        chained; each body moves as its first node does (the one added to
        the model first), and the others follow it exactly. The
        eigensolution eliminates the followers' degrees of freedom rather
        than stiffening anything, so the answer is the limit of an
        infinitely stiff member and the matrices stay well conditioned.

        Parameters
        ----------
        node_a, node_b : int
            The nodes, both already in the model, and different.
        group : str, optional
            The part the link belongs to — its block, from a geometry.

        Returns
        -------
        RigidLink
        """
        for node in (node_a, node_b):
            if int(node) not in self._nodes:
                raise ValueError(f'rigid link names node {node}, which is not '
                                 'in the model')
        if int(node_a) == int(node_b):
            raise ValueError(f'a rigid link joins two nodes; it names node '
                             f'{node_a} twice')
        link = RigidLink(int(node_a), int(node_b), group)
        self.rigid_links.append(link)
        return link

    def rigid_bodies(self) -> list[list[int]]:
        """The groups of nodes the rigid links join, each in the model's
        node order — its first node is the one the others follow."""
        neighbors: dict[int, set[int]] = {}
        for link in self.rigid_links:
            neighbors.setdefault(link.node_a, set()).add(link.node_b)
            neighbors.setdefault(link.node_b, set()).add(link.node_a)
        order = {node: i for i, node in enumerate(self._nodes)}
        return [sorted(piece, key=order.__getitem__)
                for piece in connected_pieces(neighbors)]

    def add_mass(self, node: int, mass: float,
                 inertia: Sequence[float] = (0.0, 0.0, 0.0),
                 name: str = '') -> LumpedMass:
        if int(node) not in self._nodes:
            raise ValueError(f'mass names node {node}, which is not in the model')
        item = LumpedMass(int(node), float(mass),
                          tuple(float(v) for v in inertia), name)
        self.masses.append(item)
        return item

    def add_face(self, nodes: Sequence[int], color: int = 1,
                 group: str = '') -> Face:
        nodes = tuple(int(n) for n in nodes)
        if len(nodes) not in (3, 4):
            raise ValueError('a face spans three or four nodes')
        for node in nodes:
            if node not in self._nodes:
                raise ValueError(f'face names node {node}, which is not in the model')
        face = Face(nodes, color, group)
        self.faces.append(face)
        return face

    @classmethod
    def from_geometry(cls, geometry: Geometry, material: Material | None = None,
                      section: Section | None = None,
                      total_mass: float | None = None,
                      name: str = '',
                      groups: dict[int, str] | None = None,
                      sections: dict[str, Section] | None = None) -> Model:
        """A structure from a drawn shape: members on its edges, mass at its
        nodes.

        The shortest route from *any* geometry to a set of modes. Every
        element contributes its own edges as beams — a face gives its
        perimeter, a line element gives its run — and the mass is shared
        equally over the nodes. What comes back is a model that can be
        solved like any other.

        This is a sanity-check tool, not a mesher. The members are a stand-in
        for whatever the real structure is: one section for the whole model,
        chosen to put the modes where they are wanted, and the answer scales
        as its square root. Shell bending, membrane action and any real
        thickness are simply not represented. What it *is* good for is
        looking at a shape and asking what order of mode density and what
        mode families it has — which is the question a display model usually
        raises first.

        **A geometry whose blocks carry properties builds itself.** When
        `geometry.block_properties` names what each block is made of
        (`BlockProperties`), every element becomes the element it is:
        a quad a plate, a triangle a triangle, a two-node line a beam,
        each with its block's material and thickness or section. That
        is the model an exodus file means, one property set per block
        of one element type (Brandon, 2026-09-25), and the demonstration
        plate rebuilt from its own geometry this way is the same model
        to the last digit (`tests/test_block_model.py`). A block with no
        properties, or an element type the solver has no element for,
        is refused by name; `material` and `section` are not consulted.
        Without block properties the grillage below is built, and
        `material` and `section` are required for it.

        A drawn line — a block of two-node line elements with no
        properties, what a traceline was — is an element like any other
        here and gives its run, which is what a wireframe geometry
        needs to hold together at all.
        `groups` labels the nodes by the part they belong to; a Geometry
        does not carry that, and the first question asked of any result is
        which part of the structure a mode lives in. `sections` then gives
        one part a section of its own — {'prop': stiffer} — applied where
        both ends of an edge belong to it, which is how a part is made
        stiffer or softer than the rest without redrawing anything.
        """
        model = cls(name or getattr(geometry, 'name', '') or 'geometry',
                    length_unit=geometry.length_unit or 'm')
        properties = dict(getattr(geometry, 'block_properties', {}) or {})
        if properties:
            return _from_blocks(model, geometry, properties, total_mass,
                                groups)
        if material is None or section is None:
            raise ValueError(
                'the geometry carries no block properties, so a material '
                'and a section are needed to make a grillage of it — or '
                'give each block its properties (fem.BlockProperties) and '
                'the elements build themselves')
        labels = dict(groups or {})
        if not labels:
            # A geometry that carries element blocks says for itself which
            # part each node belongs to, so nothing has to be passed
            # alongside it. That matters because a side-channel does not
            # survive being saved: a geometry written to a file and read
            # back could not reproduce the model it came from.
            labels = _labels_from_blocks(geometry)
        for node, xyz in zip(geometry.node_id, geometry.node_xyz):
            model.add_node(int(node), *[float(v) for v in xyz],
                           group=labels.get(int(node), ''))

        # A member takes its section from the block of the element it came
        # from, not from labels on its end nodes. A node on a seam belongs
        # to two parts and can only answer for one, which left 24 of the
        # drone's 1248 blade members reading as ordinary frame; an element
        # belongs to exactly one block and is never ambiguous.
        parts = _block_labels(geometry)
        runs: list[tuple[list[int], str]] = []
        for index, (kind, conn) in enumerate(zip(geometry.elem_type,
                                                 geometry.elem_conn)):
            nodes = [int(n) for n in conn]
            label = parts[index] if index < len(parts) else ''
            shape = ELEMENT_TYPES.get(int(kind), (None, 0, 'line'))[2]
            if shape == 'line':
                runs.append((nodes, label))
            else:
                # a face closes on itself; three or four of them also
                # become a Face, so the shape can be drawn back
                runs.append((nodes + nodes[:1], label))
                if len(nodes) in (3, 4):
                    model.add_face(nodes, group=label)
        seen: set[frozenset[int]] = set()
        for run, label in runs:
            chosen = section
            for part, alternative in (sections or {}).items():
                if label.startswith(part):
                    chosen = alternative
                    break
            for a, b in pairwise(run):
                edge = frozenset((a, b))
                if a == b or edge in seen:
                    continue
                seen.add(edge)
                model.add_beam(a, b, material, chosen, group=label)
        if not seen:
            raise ValueError(
                'the geometry has no elements to make members '
                'from, so there is nothing to connect its nodes')

        loose = sorted(set(model.node_ids)
                       - {n for edge in seen for n in edge})
        if loose:
            raise ValueError(
                f'{len(loose)} nodes are connected to nothing, so they would '
                'carry mass with no stiffness and the solution would not '
                'factorize: ' + ', '.join(str(n) for n in loose[:8]))
        pieces = model.pieces()
        if len(pieces) > 1:
            sizes = ', '.join(str(len(p)) for p in pieces[:6])
            raise ValueError(
                f'the geometry is {len(pieces)} disconnected pieces ({sizes} '
                'nodes), which would solve as that many free bodies and give '
                f'{6 * len(pieces)} zero-frequency modes rather than 6. Its '
                'elements do not join them: either they are '
                'meant to be separate structures, or the connectivity is '
                'incomplete')
        if total_mass is not None:
            model.distribute_mass(total_mass)
        return model

    def pieces(self) -> list[list[int]]:
        """The structure's disconnected parts, largest first.

        One piece is a structure; more than one is that many free bodies,
        each bringing its own six zero-frequency modes. It is the first
        thing to ask of any model that comes back too floppy, and the
        answer is almost never what was intended — the old airplane
        fixture, meshed and wired through its own drawn lines, turned out
        to be three: the fuselage, a wing and the tail, none joined.
        """
        neighbors: dict[int, set[int]] = {n: set() for n in self.node_ids}
        for beam in self.beams:
            if beam.node_a in neighbors and beam.node_b in neighbors:
                neighbors[beam.node_a].add(beam.node_b)
                neighbors[beam.node_b].add(beam.node_a)
        for plate in self.plates:
            for k, node in enumerate(plate.nodes):
                other = plate.nodes[(k + 1) % 4]
                neighbors[node].add(other)
                neighbors[other].add(node)
        for triangle in self.triangles:
            for k, node in enumerate(triangle.nodes):
                other = triangle.nodes[(k + 1) % 3]
                neighbors[node].add(other)
                neighbors[other].add(node)
        for solid in self.solids:
            first = solid.nodes[0]
            for node in solid.nodes[1:]:
                neighbors[first].add(node)
                neighbors[node].add(first)
        for link in self.rigid_links:
            neighbors[link.node_a].add(link.node_b)
            neighbors[link.node_b].add(link.node_a)
        return connected_pieces(neighbors)

    def wire_faces(self, material: Material, section: Section,
                   color: int = 1, group: str = '') -> int:
        """Put a beam along every edge of every face, and return how many.

        This is the shortest road from a shape to a structure: draw the
        thing as a surface mesh, and let its own edges be its members. The
        geometry then *is* the model, with nothing derived, nothing tied,
        and no second set of nodes that only exist to be looked at.

        Beams, not axial springs. A spring on each edge leaves a quad free
        to shear and a flat sheet free to fold — the edges never change
        length, so nothing resists it — and the model comes back a
        mechanism with as many zero-frequency modes as it has panels. A
        beam carries moment, so a wireframe of them is a space frame and
        stands up. It is the same element the rest of this module uses.

        Shared edges are wired once. An edge that already has a beam is
        left alone, so explicit members (a truss strut, a standoff) can be
        placed first and keep their own section.
        """
        seen = {frozenset((beam.node_a, beam.node_b)) for beam in self.beams}
        added = 0
        for face in self.faces:
            nodes = face.nodes
            for k, node in enumerate(nodes):
                other = nodes[(k + 1) % len(nodes)]
                edge = frozenset((node, other))
                if edge in seen or node == other:
                    continue
                seen.add(edge)
                self.add_beam(node, other, material, section, color=color,
                              group=group or f'edge {len(self.beams)}')
                added += 1
        return added

    def distribute_mass(self, total: float, spin: float = 1.0 / 12.0
                        ) -> dict[int, float]:
        """Share `total` over the nodes by the members meeting at each.

        A node's share is proportional to the length of member it carries —
        half of every member that reaches it — so mass follows the material
        rather than the mesh. Returns {node: mass}.

        Sharing it *equally* is the obvious thing and it is a trap, because
        a node count is a statement about how finely something was drawn
        rather than about how much of it there is. On the demonstration
        airframe that put 42% of the mass into the propellers — they need
        the finest mesh to look right, so they collect the most nodes — and
        117 g on each rotor buried every airframe mode below 1.8 kHz under
        blade motion, while the battery, the heaviest real item on the
        aircraft, was left with 51 g.

        Each node also gets a rotary inertia, without which the three
        rotational degrees of freedom carry nothing, the mass matrix is
        singular and the Cholesky factorization fails outright. It is
        taken as `spin * m * L^2` with L the mean length of the members
        meeting there — the patch of structure the node stands for, spun
        about its own middle. The default 1/12 is a uniform rod's.
        """
        nodes = self.node_ids
        if not nodes:
            raise ValueError('there are no nodes to share the mass over')
        reach: dict[int, list[float]] = {node: [] for node in nodes}
        for beam in self.beams:
            length = self._length(beam)
            for node in (beam.node_a, beam.node_b):
                if node in reach:
                    reach[node].append(length)
        carried = {node: sum(lengths) / 2.0 for node, lengths in reach.items()}
        span = sum(carried.values())
        if span <= 0.0:
            raise ValueError(
                'no node carries any member, so there is nothing to share '
                'the mass out in proportion to')

        self.masses = [item for item in self.masses if item.name != 'shared']
        shares = {}
        for node in nodes:
            share = float(total) * carried[node] / span
            lengths = reach[node]
            scale = float(np.mean(lengths)) if lengths else 0.0
            inertia = spin * share * scale ** 2
            self.add_mass(node, share, (inertia, inertia, inertia),
                          name='shared')
            shares[node] = share
        return shares

    # ---- what is in it ----------------------------------------------------

    @property
    def node_ids(self) -> list[int]:
        """In insertion order: the matrices' row order follows this."""
        return list(self._nodes)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_dof(self) -> int:
        return 6 * len(self._nodes)

    def position(self, node: int) -> np.ndarray:
        return self._nodes[int(node)]

    def group(self, node: int) -> str:
        """What this node was added as part of — 'arm 2', 'top deck'.

        A label for the modeler's own use: nothing here reads it, but
        working out which part of a structure a mode lives in is the first
        question asked of any result, and reconstructing it from
        coordinates afterwards is guesswork.
        """
        return self._node_group[int(node)]

    def dof_strings(self) -> list[str]:
        """'101X+', '101Y+', … in the matrices' own order."""
        return [f'{node}{direction}'
                for node in self._nodes for direction in DIRECTIONS]

    @property
    def structural_mass(self) -> float:
        """What the members weigh, before anything is hung on them."""
        beams = float(sum(beam.section.area * beam.material.density
                          * self._length(beam) for beam in self.beams))
        plates = 0.0
        for plate in self.plates:
            _, a, b = _plate_frame(*[self._nodes[n] for n in plate.nodes])
            plates += plate.material.density * plate.thickness * a * b
        for triangle in self.triangles:
            _, xy = _triangle_frame(*[self._nodes[n] for n in triangle.nodes])
            plates += (triangle.material.density * triangle.thickness
                       * _triangle_area(xy))
        solids = float(sum(
            solid.material.density
            * abs(_solid_volume(np.array([self._nodes[n] for n in solid.nodes])))
            for solid in self.solids))
        return beams + plates + solids

    @property
    def total_mass(self) -> float:
        return self.structural_mass + float(sum(m.mass for m in self.masses))

    def _length(self, beam: Beam) -> float:
        return float(np.linalg.norm(self._nodes[beam.node_b]
                                    - self._nodes[beam.node_a]))

    # ---- the matrices -----------------------------------------------------

    def _contributions(self):
        """Every element's (rows, stiffness, mass) in global coordinates,
        and every lumped mass's (rows, None, mass) — `_batches` one
        element at a time, for the dense assembly."""
        for rows, k, m in self._batches():
            for e in range(len(rows)):
                yield rows[e], (None if k is None else k[e]), m[e]

    def _batches(self, chunk: int = ASSEMBLY_CHUNK):
        """The elements' (rows, stiffness, mass) in global coordinates,
        as arrays of up to `chunk` elements of one family at a time —
        rows (E, r), stiffness and mass (E, r, r) — what both assemblies
        scatter, so the dense and the sparse matrices cannot differ in
        anything but storage.

        Each distinct element's local matrices are computed once (2026-10-03):
        a plate's depend only on its material, thickness and side lengths,
        a beam's on its material, section and length, a triangle's on its
        corners in its own plane, a solid's on its corners relative to its
        first, so a meshed model of twenty thousand identical plates built
        one 24x24 pair twenty thousand times in Python, half its solve
        (Brandon: is the solve multi-threaded? It was not, and threads were
        not the answer). The key is the exact inputs, so a cached matrix is
        the matrix; the rotations into the global axes are one einsum per
        chunk, and the chunk bounds the memory a fine mesh takes.
        """
        index = {node: 6 * i for i, node in enumerate(self.node_ids)}

        def dof_rows(nodes_per, per_node):
            idx = np.array([[index[n] for n in nodes] for nodes in nodes_per],
                           dtype=np.int64)
            return (idx[:, :, None] + np.arange(per_node)).reshape(len(idx), -1)

        def rotated(rotations, local):
            # k_g = T^T k_l T with T the rotation repeated down the diagonal
            count, size, _ = local.shape
            blocks = size // 3
            grid = local.reshape(count, blocks, 3, blocks, 3)
            out = np.einsum('epi,eapbq,eqj->eaibj', rotations, grid, rotations,
                            optimize=True)
            return out.reshape(count, size, size)

        def cached(compute, keys):
            store: dict = {}
            first = []
            for key in keys:
                if key not in store:
                    store[key] = len(first)
                    first.append(key)
            which = np.array([store[key] for key in keys], dtype=np.int64)
            pairs = [compute(key) for key in first]
            k = np.array([pair[0] for pair in pairs])
            m = np.array([pair[1] for pair in pairs])
            return which, k, m

        for lo in range(0, len(self.beams), chunk):
            beams = self.beams[lo:lo + chunk]
            lengths = []
            rotations = []
            for beam in beams:
                length = self._length(beam)
                if length == 0.0:
                    raise ValueError(f'beam {beam.node_a}-{beam.node_b} has zero length')
                lengths.append(length)
                rotations.append(_element_axes(self._nodes[beam.node_a],
                                               self._nodes[beam.node_b],
                                               beam.orientation))
            which, k, m = cached(
                lambda key: (_beam_stiffness(*key), _beam_mass(*key)),
                [(beam.material, beam.section, length)
                 for beam, length in zip(beams, lengths)])
            rotations = np.array(rotations)
            rows = dof_rows([(beam.node_a, beam.node_b) for beam in beams], 6)
            yield rows, rotated(rotations, k[which]), rotated(rotations, m[which])

        for lo in range(0, len(self.plates), chunk):
            plates = self.plates[lo:lo + chunk]
            corners = np.array([[self._nodes[n] for n in plate.nodes]
                                for plate in plates], dtype=np.float64)
            rotations, a, b = _plate_frames(corners)
            which, k, m = cached(
                lambda key: _plate_matrices(*key),
                [(plate.material, plate.thickness, float(a[e]), float(b[e]))
                 for e, plate in enumerate(plates)])
            rows = dof_rows([plate.nodes for plate in plates], 6)
            yield rows, rotated(rotations, k[which]), rotated(rotations, m[which])

        for lo in range(0, len(self.triangles), chunk):
            triangles = self.triangles[lo:lo + chunk]
            frames = [_triangle_frame(*[self._nodes[n] for n in triangle.nodes])
                      for triangle in triangles]
            which, k, m = cached(
                lambda key: _triangle_matrices(key[0], key[1],
                                               np.array(key[2]).reshape(3, 2)),
                [(triangle.material, triangle.thickness,
                  tuple(float(v) for v in xy.ravel()))
                 for triangle, (_rotation, xy) in zip(triangles, frames)])
            rotations = np.array([rotation for rotation, _xy in frames])
            rows = dof_rows([triangle.nodes for triangle in triangles], 6)
            yield rows, rotated(rotations, k[which]), rotated(rotations, m[which])

        for lo in range(0, len(self.solids), chunk):
            solids = self.solids[lo:lo + chunk]
            # tets, wedges and bricks side by side differ in size, so each
            # count is its own batch
            by_count: dict[int, list] = {}
            for solid in solids:
                by_count.setdefault(len(solid.nodes), []).append(solid)
            for members in by_count.values():
                # a solid's matrices are those of its corners relative to
                # its first: the same brick anywhere in the mesh is one brick
                shapes = [np.array([self._nodes[n] for n in solid.nodes])
                          for solid in members]
                which, k, m = cached(
                    lambda key: _solid_matrices(key[0],
                                                np.array(key[1]).reshape(-1, 3)),
                    [(solid.material, tuple(float(v) for v in (xyz - xyz[0]).ravel()))
                     for solid, xyz in zip(members, shapes)])
                # three translations per node: the rows of each node's
                # first three degrees of freedom, and nothing on its rotations
                rows = dof_rows([solid.nodes for solid in members], 3)
                yield rows, k[which], m[which]

        if self.masses:
            rows = dof_rows([(item.node,) for item in self.masses], 6)
            mass = np.array([np.diag([item.mass] * 3 + list(item.inertia))
                             for item in self.masses])
            yield rows, None, mass

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the global mass and stiffness matrices, dense.

        Rows and columns run structural node by structural node in
        insertion order, six per node in `DIRECTIONS` order, which is what
        `dof_strings()` spells out. Display nodes are absent: they carry
        nothing, so there is nothing of theirs to assemble.
        """
        n = self.num_dof
        mass = np.zeros((n, n), dtype=np.float64)
        stiffness = np.zeros((n, n), dtype=np.float64)
        for rows, k, m in self._contributions():
            grid = np.ix_(rows, rows)
            if k is not None:
                stiffness[grid] += k
            mass[grid] += m
        # assembly is symmetric by construction, but floating point addition
        # is not associative and the halves drift apart in the last bits;
        # eigh reads only one triangle, so an asymmetry here is silent
        return (mass + mass.T) / 2.0, (stiffness + stiffness.T) / 2.0

    def sparse_matrices(self, ticker: Any = None) -> tuple[Any, Any]:
        """The same mass and stiffness matrices as `matrices`, stored
        sparse (scipy CSR): each node couples only to the nodes of the
        elements it touches, so a row holds a few dozen entries of
        thousands, and the storage grows with the nodes rather than
        their square.

        Returns
        -------
        tuple of scipy.sparse.csr_matrix
            (mass, stiffness), symmetric.
        """
        from scipy import sparse

        n = self.num_dof
        rows_k, cols_k, vals_k, rows_m, cols_m, vals_m = [], [], [], [], [], []
        if ticker is not None:
            ticker.add(len(self.beams) + len(self.plates)
                       + len(self.triangles) + len(self.solids))
        for rows, k, m in self._batches():
            count, r = rows.shape
            i = np.broadcast_to(rows[:, :, None], (count, r, r)).ravel()
            j = np.broadcast_to(rows[:, None, :], (count, r, r)).ravel()
            if k is not None:
                rows_k.append(i)
                cols_k.append(j)
                vals_k.append(k.ravel())
                if ticker is not None:
                    ticker.tick(count)
            rows_m.append(i)
            cols_m.append(j)
            vals_m.append(m.ravel())

        def gather(rows, cols, vals):
            if not rows:
                return sparse.csr_matrix((n, n))
            matrix = sparse.coo_matrix(
                (np.concatenate(vals), (np.concatenate(rows),
                                        np.concatenate(cols))),
                shape=(n, n)).tocsr()
            return ((matrix + matrix.T) / 2.0).tocsr()

        return gather(rows_m, cols_m, vals_m), gather(rows_k, cols_k, vals_k)

    def rigid_body_vectors(self) -> np.ndarray:
        """The six rigid-body motions, as columns over the model's DOFs.

        Written down from the node positions rather than found from the
        matrices: they are what the null space of an unconstrained
        stiffness matrix *is*, and knowing them in advance is what lets a
        rigid mode be recognized as rigid rather than as a very soft one.
        """
        n = self.num_dof
        solved = self.node_ids
        vectors = np.zeros((n, 6), dtype=np.float64)
        center = np.mean(np.array([self._nodes[k] for k in solved]), axis=0)
        for i, node in enumerate(solved):
            offset = self._nodes[node] - center
            for axis in range(3):
                vectors[6 * i + axis, axis] = 1.0            # translations
                # a small rotation about `axis` moves a point by the cross
                # product of the axis with its offset, and rotates it by
                # the axis itself
                spin = np.zeros(3)
                spin[axis] = 1.0
                vectors[6 * i:6 * i + 3, 3 + axis] = np.cross(spin, offset)
                vectors[6 * i + 3 + axis, 3 + axis] = 1.0
        return vectors

    # ---- the answer -------------------------------------------------------

    def eigensolution(self, maximum_frequency: float | None = None,
                      num_modes: int | None = None, damping: float = 0.0,
                      fixed: Sequence[str] = (),
                      solver: str = 'auto',
                      progress: Callable[[int, int], None] | None = None
                      ) -> ShapeSet:
        """Real normal modes, mass-normalized, as a ShapeSet.

        `fixed` names degrees of freedom to ground: '101X+' fixes one,
        '101' fixes all six of that node. Rigid links are eliminated
        exactly (`constraint_transform`).

        Two solvers, one answer. **Dense** (a model of up to
        `SPARSE_ABOVE` degrees of freedom): the symmetric generalized
        problem K phi = lambda M phi solved whole, by factoring M
        (Cholesky), reducing to a standard symmetric problem and
        transforming back — every mode, mass-normalized to machine
        precision. **Sparse** (larger models): the matrices stored as
        their nonzeros (`sparse_matrices`) and the lowest modes found by
        shift-invert Lanczos (ARPACK, `scipy.sparse.linalg.eigsh`), the
        family the large finite element codes use; it finds the lowest
        `num_modes`, or every mode up to `maximum_frequency`, and one of
        the two must be said. Memory grows with the nodes instead of
        their square: a plate model of 1,500 nodes, ~13 GB dense, is tens
        of megabytes sparse (Brandon, 2026-09-26, for the BARC example —
        the dense ceiling, deliberate until then, measured and met).

        `damping` is a fraction of critical, applied uniformly. A model
        has no damping of its own; it is stated so the modes can
        synthesize an FRF that looks like a measurement.

        Parameters
        ----------
        maximum_frequency : float, optional
            Keep every mode up to this frequency, in Hz.
        num_modes : int, optional
            Keep this many, the lowest, rigid ones included.
        damping : float, default 0.0
            The fraction of critical damping every mode is given.
        fixed : sequence of str
            Degrees of freedom to ground.
        solver : {'auto', 'dense', 'sparse'}, default 'auto'
            Which solver; 'auto' is dense up to `SPARSE_ABOVE` degrees
            of freedom and sparse beyond.
        progress : callable, optional
            Told ``(done, total)`` as the solve advances: the assembly
            per element, the factorization, the eigen iterations (an
            estimated length, extended while they run), the polish.
            The window's strip bar reads it; anything it raises stops
            the solve.

        Returns
        -------
        ShapeSet
        """
        if solver not in ('auto', 'dense', 'sparse'):
            raise ValueError(f'{solver!r} is not a solver: auto, dense, sparse')
        ticker = Ticker(progress)
        if solver == 'sparse' or (solver == 'auto'
                                  and self.num_dof > SPARSE_ABOVE):
            eigenvalues, full, stiffness = self._sparse_modes(
                fixed, maximum_frequency, num_modes, ticker)
        else:
            eigenvalues, full, stiffness = self._dense_modes(fixed, ticker)
        if ticker.total:
            ticker.tick(ticker.total - ticker.done)
        # a rigid-body eigenvalue is zero plus round-off, and comes out
        # either side of it; the negative ones are not oscillations
        frequency = np.sqrt(np.clip(eigenvalues, 0.0, None)) / (2.0 * np.pi)
        frequency[_strains_nothing(full, stiffness)] = 0.0

        order = np.argsort(frequency, kind='stable')
        frequency, full = frequency[order], full[:, order]
        keep = np.ones(len(frequency), dtype=bool)
        if maximum_frequency is not None:
            keep &= frequency <= float(maximum_frequency)
        keep = np.flatnonzero(keep)
        if num_modes is not None:
            keep = keep[:int(num_modes)]

        return ShapeSet(frequency=frequency[keep],
                        damping=np.full(len(keep), float(damping)),
                        coordinate=self.dof_strings(),
                        shape_matrix=full[:, keep].T,
                        mass_unit='kg',
                        comment=self.name or '')

    def _dense_modes(self, fixed, ticker=None):
        """(eigenvalues, shapes at every DOF, stiffness) solved whole."""
        if ticker is not None:
            # the dense path is one factorization and one eigh: two
            # steps, the second the long one
            ticker.add(2)
        mass, stiffness = self.matrices()
        # u = T q: q the degrees of freedom that remain — every node's,
        # less those grounded and those following a rigid link's first
        # node — and T writes every one of the model's in terms of them
        transform = self.constraint_transform(fixed)
        free = np.arange(transform.shape[1])
        if not len(free):
            raise ValueError('every degree of freedom is fixed')
        reduced_m = transform.T @ mass @ transform
        reduced_k = transform.T @ stiffness @ transform

        try:
            factor = np.linalg.cholesky(reduced_m)
        except np.linalg.LinAlgError:
            names = self.dof_strings()
            kept = [names[int(np.flatnonzero(column)[0])]
                    for column in transform.T]
            starved = [kept[i]
                       for i in np.flatnonzero(np.diag(reduced_m) <= 0.0)]
            raise ValueError(
                'the mass matrix is not positive definite, so these degrees '
                'of freedom carry no mass and no rotary inertia: '
                + (', '.join(starved[:6]) or 'none on the diagonal, so the '
                   'model is nearly a mechanism')) from None

        # M = L L^T, so K phi = lambda M phi becomes A y = lambda y with
        # A = L^-1 K L^-T and phi = L^-T y. y orthonormal then gives
        # phi^T M phi = y^T y = I exactly, which is the normalization we
        # want and not something applied afterwards.
        temporary = np.linalg.solve(factor, reduced_k)
        standard = np.linalg.solve(factor, temporary.T).T
        if ticker is not None:
            ticker.tick()
        eigenvalues, vectors = np.linalg.eigh((standard + standard.T) / 2.0)
        if ticker is not None:
            ticker.tick()
        shapes = np.linalg.solve(factor.T, vectors)

        full = transform @ shapes
        return eigenvalues, full, stiffness

    def scaled_system(self, fixed: Sequence[str] = (),
                      ticker: Any = None) -> tuple[Any, ...]:
        """The sparse eigenproblem as the sparse solver poses it.

        Returns (K, M, T, S, sigma, stiffness): the constrained,
        symmetrically scaled stiffness and mass in CSC form, the
        constraint transform T and the scaling S that carry a solution
        back to every degree of freedom as T (S phi), the shift sigma, and
        the unconstrained sparse stiffness. Public so a test can hand the
        polish a start of its own choosing.

        The shift is just below zero: the rigid-body modes (eigenvalue 0)
        are nearest it, so they come first, and K - sigma M = K + |sigma| M
        is positive definite even when K alone is singular (free-free).
        The scaling is symmetric and diagonal, S K S and S M S with S =
        1/sqrt of the shifted diagonal: the same eigenvalues, the modes
        S phi'. A model of plates mixes meters with radians, and rigid
        links fold lever arms into the rotations — the rigid-link test
        model's shifted matrix had a condition number of 4e12, and
        Lanczos, which converges no better than its linear solves, left
        residuals of 1e-2. Scaled it is 4e9 (2026-09-26).
        """
        from scipy import sparse

        mass, stiffness = self.sparse_matrices(ticker)
        transform = self.constraint_transform(fixed, sparse=True)
        reduced_m = (transform.T @ mass @ transform).tocsc()
        reduced_k = (transform.T @ stiffness @ transform).tocsc()
        if not reduced_m.shape[0]:
            raise ValueError('every degree of freedom is fixed')
        sigma = -(2.0 * np.pi) ** 2
        scale = sparse.diags(1.0 / np.sqrt(
            (reduced_k - sigma * reduced_m).diagonal()))
        reduced_k = (scale @ reduced_k @ scale).tocsc()
        reduced_m = (scale @ reduced_m @ scale).tocsc()
        return reduced_k, reduced_m, transform, scale, sigma, stiffness

    def _sparse_modes(self, fixed, maximum_frequency, num_modes, ticker=None):
        """(eigenvalues, shapes at every DOF, stiffness) for the lowest
        modes, by shift-invert Lanczos on the sparse matrices."""
        from scipy.sparse.linalg import LinearOperator, eigsh, splu

        if num_modes is None and maximum_frequency is None:
            raise ValueError(
                f'a model of {self.num_dof} degrees of freedom is solved '
                'for its lowest modes: say how many (num_modes) or up to '
                'what frequency (maximum_frequency)')
        reduced_k, reduced_m, transform, scale, sigma, stiffness = \
            self.scaled_system(fixed, ticker)
        size = reduced_m.shape[0]
        # the shifted factor, once: eigsh's shift-invert and the polish
        # both solve with it, and eigsh built its own until the bar
        # needed to count the solves (2026-10-01) — the operator ticks
        # per application against an estimate the bar is kept short of
        if ticker is not None:
            ticker.add(1)
        try:
            factor = splu((reduced_k - sigma * reduced_m).tocsc())
        except RuntimeError as failure:
            empty = np.flatnonzero((reduced_m.diagonal() <= 0.0)
                                   & (reduced_k.diagonal() <= 0.0))
            raise ValueError(
                'the model could not be factored: degrees of freedom '
                'with neither mass nor stiffness'
                + (f' ({len(empty)} of them)' if len(empty) else '')
                + f' — {failure}') from None
        if ticker is not None:
            ticker.tick()

        def solve(vector):
            if ticker is not None:
                ticker.extend_to(1)
                ticker.tick()
            return factor.solve(np.asarray(vector, dtype=float))

        shifted_inverse = LinearOperator(factor.shape, matvec=solve,
                                         dtype=float)
        limit = (None if maximum_frequency is None
                 else (2.0 * np.pi * float(maximum_frequency)) ** 2)
        wanted = int(num_modes) if num_modes is not None else 24
        while True:
            wanted = max(1, min(wanted, size - 1))
            count = min(wanted + LANCZOS_GUARD, size - 1)
            try:
                if ticker is not None:
                    ticker.extend_to(EIGEN_SOLVES_PER_MODE * count)
                eigenvalues, vectors = eigsh(reduced_k, k=count, M=reduced_m,
                                             sigma=sigma, which='LM',
                                             OPinv=shifted_inverse)
            except RuntimeError as failure:
                empty = np.flatnonzero((reduced_m.diagonal() <= 0.0)
                                       & (reduced_k.diagonal() <= 0.0))
                raise ValueError(
                    'the model could not be factored: degrees of freedom '
                    'with neither mass nor stiffness'
                    + (f' ({len(empty)} of them)' if len(empty) else '')
                    + f' — {failure}') from None
            # the wanted modes, not the guard, have to reach the limit:
            # the guard is what keeps the last of them clean, and a
            # guard mode is never the answer to "every mode up to"
            if (limit is None or np.sort(eigenvalues)[wanted - 1] > limit
                    or wanted >= size - 1):
                break
            wanted *= 2
        # Polish: block inverse iteration with the shifted factor, each
        # step followed by Rayleigh-Ritz, until the residuals say the
        # modes are converged (`polish_modes`). What is left of the
        # conditioning after scaling still bounds Lanczos' own accuracy
        # (measured on the rigid-link model: 1e-4 residual before, 1.5e-6
        # after two fixed steps, 2026-09-26), and a fixed count of steps
        # was the flaw: a start the polish did not clean in three steps
        # slipped through on one runner's build. The guard band above
        # the wanted modes is trimmed by the caller's `keep`. The Ritz
        # step also makes the modes mass-normal exactly.
        eigenvalues, vectors, _residuals = polish_modes(
            reduced_k, reduced_m, sigma, vectors, wanted, factor=factor,
            ticker=ticker)
        return eigenvalues, transform @ (scale @ vectors), stiffness

    def constraint_transform(self, fixed: Sequence[str] = (),
                             sparse: bool = False) -> Any:
        """T, with u = T q: every degree of freedom of the model written
        in terms of those that remain free — grounded ones gone, and each
        rigid body's followers written through its first node, u_b = u_a +
        θ_a × (x_b − x_a) and θ_b = θ_a. The identity's columns when there
        is nothing to constrain.

        Parameters
        ----------
        fixed : sequence of str
            Degrees of freedom to ground, as `eigensolution` takes them. A
            rigid body is grounded through its first node; naming one of
            its followers is refused, since fixing a follower alone would
            fix part of a rigid body and not the rest.

        sparse : bool, default False
            Return it as a scipy CSR matrix, for the sparse solver.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            (num_dof, remaining) and real.
        """
        index = {node: 6 * i for i, node in enumerate(self.node_ids)}
        follows: dict[int, int] = {}
        for body in self.rigid_bodies():
            for node in body[1:]:
                follows[node] = body[0]
        free = self._free_dofs(fixed)
        followers = {index[node] + k for node in follows for k in range(6)}
        held = sorted(set(range(self.num_dof)) - {int(i) for i in free})
        clash = sorted({self.node_ids[i // 6] for i in held if i in followers})
        if clash:
            raise ValueError(
                'node ' + ', '.join(str(n) for n in clash) + ' follows a '
                'rigid link; ground the node it follows instead')
        kept = [int(i) for i in free if int(i) not in followers]
        column = {dof: k for k, dof in enumerate(kept)}
        entries = [(dof, k, 1.0) for dof, k in column.items()]
        for node, lead in follows.items():
            r = self._nodes[node] - self._nodes[lead]
            # θ × r = -[r]× θ: the lead's rotation moves the follower
            arm = np.array([[0.0, r[2], -r[1]],
                            [-r[2], 0.0, r[0]],
                            [r[1], -r[0], 0.0]])
            rows = index[node]
            lead_row = index[lead]
            for k in range(6):
                source = lead_row + k
                if source not in column:
                    continue                  # the lead is grounded there
                entries.append((rows + k, column[source], 1.0))
                if k >= 3:
                    entries.extend((rows + axis, column[source],
                                    float(arm[axis, k - 3]))
                                   for axis in range(3) if arm[axis, k - 3])
        shape = (self.num_dof, len(kept))
        if sparse:
            from scipy import sparse as sp

            if not entries:
                return sp.csr_matrix(shape)
            i, j, v = zip(*entries, strict=True)
            return sp.coo_matrix((v, (i, j)), shape=shape).tocsr()
        transform = np.zeros(shape, dtype=np.float64)
        for i, j, v in entries:
            transform[i, j] = v
        return transform

    def dangling_rotations(self) -> list[int]:
        """The nodes whose rotations nothing acts on: touched by solids
        and by nothing that carries a rotation — no beam, plate or
        triangle, and no rigid link, whose lead's rotation moves its
        followers. A solid has no rotational stiffness, so these
        rotations are grounded by the eigensolution; left free they
        would be degrees of freedom with neither mass nor stiffness,
        which no factorization survives.

        Returns
        -------
        list of int
            The nodes, in the model's order; empty for a model with no
            solids.
        """
        if not self.solids:
            return []
        rotating = self._rotating_nodes()
        touched = {n for solid in self.solids for n in solid.nodes}
        return [n for n in self.node_ids if n in touched and n not in rotating]

    def loose_nodes(self) -> list[int]:
        """The nodes nothing touches: no element, no rigid link, no
        lumped mass. A finite element deck carries them routinely — a
        reference point, a constraint's own grid — and a model built
        from its blocks grounds them whole rather than refusing the
        deck (2026-09-30); the grillage path still refuses, since there
        a loose node is a drawing that was never wired.

        Returns
        -------
        list of int
            The nodes, in the model's order.
        """
        touched = self._rotating_nodes()
        touched |= {n for solid in self.solids for n in solid.nodes}
        touched |= {m.node for m in self.masses}
        return [n for n in self.node_ids if n not in touched]

    def _rotating_nodes(self) -> set[int]:
        """Every node something with a rotation acts on."""
        rotating = {n for beam in self.beams for n in (beam.node_a, beam.node_b)}
        rotating |= {n for plate in self.plates for n in plate.nodes}
        rotating |= {n for triangle in self.triangles for n in triangle.nodes}
        # a rigid body's lead rotates its followers, so its rotation has
        # something to act on; the followers' rotations are written
        # through the lead's and are not free to ground
        rotating |= {n for body in self.rigid_bodies() for n in body}
        return rotating

    def _free_dofs(self, fixed) -> np.ndarray:
        """Which rows survive after grounding what `fixed` names, the
        rotations of the nodes only solids touch, and every degree of
        freedom of a node nothing touches."""
        index = {node: 6 * i for i, node in enumerate(self.node_ids)}
        held = set()
        for node in self.loose_nodes():
            held.update(range(index[node], index[node] + 6))
        dangling = self.dangling_rotations()
        if dangling:
            inert = sorted({m.node for m in self.masses
                            if any(m.inertia) and m.node in set(dangling)})
            if inert:
                raise ValueError(
                    f'the mass at node {inert[0]} has rotary inertia, but '
                    'only solids touch the node and a solid gives a '
                    'rotation nothing to act on; put it on a node a beam, '
                    'a plate or a rigid link holds')
            for node in dangling:
                held.update(range(index[node] + 3, index[node] + 6))
        for item in fixed:
            text = str(item).strip()
            digits = 0
            while digits < len(text) and text[digits].isdigit():
                digits += 1
            node = int(text[:digits]) if digits else None
            if node not in index:
                raise ValueError(f'{item!r} does not name a node in the model')
            direction = text[digits:]
            if not direction:
                held.update(range(index[node], index[node] + 6))
            else:
                # a sign is meaningless when grounding — X- is the same
                # constraint as X+ — so the code's magnitude is the axis
                held.add(index[node] + abs(direction_code(direction)) - 1)
        return np.array([i for i in range(self.num_dof) if i not in held],
                        dtype=np.int64)

    # ---- what it looks like -----------------------------------------------

    def geometry(self, beams: bool = True) -> Geometry:
        """The model as a Geometry: nodes, beams as elements, faces as faces.

        Beams become element type 21 (beam2) rather than tracelines,
        because they *are* elements — a traceline is a line drawn through
        nodes to make a display readable, and confusing the two would make
        the model's own connectivity indistinguishable from a drawing aid
        the moment anything edited it.

        `beams=False` leaves them out, for a model whose members are all
        wrapped in surfaces: there the beams run *inside* the shells, so
        drawing them puts a wireframe over the thing it is the skeleton of.
        """
        node_ids = self.node_ids
        connectivity, types, colors = [], [], []
        for beam in self.beams if beams else ():
            connectivity.append([beam.node_a, beam.node_b])
            types.append(21)
            colors.append(beam.color)
        # rigid links are two-node lines too, in blocks of their own — a
        # geometry given `RIGID` on those blocks rebuilds them
        for link in self.rigid_links:
            connectivity.append([link.node_a, link.node_b])
            types.append(21)
            colors.append(1)
        # plates are structural quads and appear as such — unlike faces,
        # which are drawings; both shade, only one carries stiffness
        for plate in self.plates:
            connectivity.append(list(plate.nodes))
            types.append(44)
            colors.append(plate.color)
        for triangle in self.triangles:
            connectivity.append(list(triangle.nodes))
            types.append(41)
            colors.append(triangle.color)
        for solid in self.solids:
            connectivity.append(list(solid.nodes))
            types.append(SOLID_CODES[len(solid.nodes)])
            colors.append(solid.color)
        for face in self.faces:
            connectivity.append(list(face.nodes))
            types.append(44 if len(face.nodes) == 4 else 41)
            colors.append(face.color)
        # Elements go into the block of the part they belong to, so the
        # geometry states its own regions and a saved file can rebuild the
        # structure. Without this the blocks live only on the drawing, and
        # the model's own geometry — which is what gets saved — carries
        # nothing: the file comes back with every member the same section.
        # The part comes off the element, never off its first node: a node
        # on a seam belongs to two parts and answers for one, which put
        # five of the drone's blade members into the frame block.
        parts, blocks = {}, []
        for part in ([b.group for b in (self.beams if beams else ())]
                     + [link.group or 'rigid links'
                        for link in self.rigid_links]
                     + [p.group for p in self.plates]
                     + [t.group for t in self.triangles]
                     + [s.group for s in self.solids]
                     + [f.group for f in self.faces]):
            blocks.append(parts.setdefault(part or 'body', len(parts) + 1))
        return Geometry(
            node_id=node_ids,
            node_xyz=np.array([self._nodes[node] for node in node_ids]),
            elem_conn=connectivity or None,
            elem_type=types or None,
            elem_color=colors or None,
            elem_block=blocks or None,
            block_id=list(parts.values()) or None,
            block_name=list(parts) or None,
            length_unit=self.length_unit)


def _from_blocks(model: Model, geometry: Geometry,
                 properties: dict[int, BlockProperties],
                 total_mass: float | None, groups) -> Model:
    """`from_geometry`'s block path: nodes, then every element as
    itself, then the same checks the grillage path makes."""
    labels = dict(groups or {}) or _labels_from_blocks(geometry)
    for node, xyz in zip(geometry.node_id, geometry.node_xyz):
        model.add_node(int(node), *[float(v) for v in xyz],
                       group=labels.get(int(node), ''))
    if not _element_by_block(model, geometry, properties):
        raise ValueError('the geometry has no elements to build from')
    joined = {n for beam in model.beams for n in (beam.node_a, beam.node_b)}
    joined |= {n for plate in model.plates for n in plate.nodes}
    joined |= {n for triangle in model.triangles for n in triangle.nodes}
    joined |= {n for solid in model.solids for n in solid.nodes}
    # nodes no element touches are grounded at solve time
    # (`Model.loose_nodes`), not refused: a deck's reference points
    pieces = [piece for piece in model.pieces()
              if len(piece) > 1 or piece[0] in joined]
    if len(pieces) > 1:
        sizes = ', '.join(str(len(p)) for p in pieces[:6])
        raise ValueError(
            f'the geometry is {len(pieces)} disconnected pieces ({sizes} '
            'nodes); a model is one structure')
    if total_mass is not None:
        model.distribute_mass(total_mass)
    return model


def _element_by_block(model: Model, geometry: Geometry,
                      properties: dict[int, BlockProperties]) -> int:
    """Every element as the element it is, with its block's properties.
    Returns how many were built."""
    ids = np.asarray(getattr(geometry, 'block_id', []), dtype=np.int64)
    names = list(getattr(geometry, 'block_name', []))
    block_name = {int(b): (names[i] if i < len(names) else '')
                  for i, b in enumerate(ids)}
    blocks = np.asarray(getattr(geometry, 'elem_block', []), dtype=np.int64)

    def named(block: int) -> str:
        label = block_name.get(block, '')
        return f'block {block}' + (f' ({label})' if label else '')

    built = 0
    for index, (kind, conn) in enumerate(zip(geometry.elem_type,
                                             geometry.elem_conn)):
        block = int(blocks[index]) if index < len(blocks) else 0
        props = properties.get(block)
        if props is None:
            raise ValueError(
                f'{named(block)} has no properties: give every block a '
                'material and a thickness or a section (fem.BlockProperties)')
        if props.kind not in ('plate', 'beam', 'solid', 'rigid'):
            raise ValueError(f'{named(block)} has {props.kind}: a block of '
                             'plates takes a thickness, a block of beams a '
                             'section, never both')
        if props.kind == 'beam' and not min(
                props.section.area, props.section.iy, props.section.iz,
                props.section.j) > 0.0:
            # a shape picked in the table and its dimensions not yet typed,
            # or a number left blank: a beam of no stiffness is a
            # singular model, said here rather than by the eigensolver
            what = (f'its {props.section.shape} has no dimensions yet'
                    if props.section.shape and not props.section.dimensions
                    else 'its A, Iy, Iz and J must all be positive')
            raise ValueError(f'{named(block)}: the section is not finished — '
                             f'{what}')
        nodes = [int(n) for n in conn]
        shape_name, _count, shape = ELEMENT_TYPES.get(
            int(kind), (f'type {int(kind)}', 0, 'unknown'))
        label = block_name.get(block, '')
        if shape == 'face' and len(nodes) == 3 and props.kind == 'plate':
            model.add_triangle(nodes, props.material, props.thickness,
                               group=label)
        elif shape == 'face' and len(nodes) == 4 and props.kind == 'plate':
            model.add_plate(nodes, props.material, props.thickness,
                            group=label)
        elif shape == 'line' and len(nodes) == 2 and props.kind == 'beam':
            model.add_beam(nodes[0], nodes[1], props.material, props.section,
                           props.orientation, group=label)
        elif shape == 'line' and len(nodes) == 2 and props.kind == 'rigid':
            model.add_rigid_link(nodes[0], nodes[1], group=label)
        elif (shape == 'volume' and len(nodes) in SOLID_CODES
                and props.kind == 'solid'):
            model.add_solid(nodes, props.material, group=label)
        elif props.kind == 'rigid':
            raise ValueError(f'{named(block)} holds {shape_name} elements; a '
                             'rigid (massless) block holds two-node lines, '
                             'one link each')
        else:
            wanted = ('a thickness' if shape == 'face' else 'a section'
                      if shape == 'line' else 'a material alone')
            solvable = ((shape in ('face', 'line') and len(nodes) in (2, 3, 4))
                        or (shape == 'volume' and len(nodes) in SOLID_CODES))
            raise ValueError(
                f'{named(block)} holds {shape_name} elements, which take '
                f'{wanted}; it was given {props.kind} properties'
                if solvable
                else f'{named(block)} holds {shape_name} elements, and the '
                     'solver has no element for them: two-node beams, '
                     'three-node triangles, four-node quads, and eight-node '
                     'hexahedra, six-node wedges and four-node tetrahedra only')
        built += 1
    return built


def _block_labels(geometry) -> list[str]:
    """The block name of each element, in the geometry's own order."""
    names = list(getattr(geometry, 'block_name', []))
    ids = np.asarray(getattr(geometry, 'block_id', []), dtype=np.int64)
    named = {int(b): (names[i] if i < len(names) else '')
             for i, b in enumerate(ids)}
    block = np.asarray(getattr(geometry, 'elem_block', []), dtype=np.int64)
    return [named.get(int(b), '') for b in block]


def _labels_from_blocks(geometry) -> dict[int, str]:
    """{node: block name} from a geometry's own element blocks.

    A node on the seam between two blocks belongs to the one holding most
    of its elements, ties going to the block declared first. It cannot
    belong to both — a node has one part in every format that records the
    question — and giving it to whichever block happened to be numbered
    first emptied the parts that sit *between* others: the drone's canopy
    is ringed by waist, camera mount and four arms, and came back with a
    single node in it, which is one accelerometer where four were meant.

    Nothing structural rides on this. Sections come off the elements,
    which are never ambiguous; these labels name the part a node is in,
    for reading a mode and for placing sensors.
    """
    names = list(getattr(geometry, 'block_name', []))
    ids = np.asarray(getattr(geometry, 'block_id', []), dtype=np.int64)
    named = {int(b): (names[i] if i < len(names) else '')
             for i, b in enumerate(ids)}
    order = {name: i for i, name in enumerate(names)}
    block = np.asarray(getattr(geometry, 'elem_block', []), dtype=np.int64)
    votes: dict[int, dict[str, int]] = {}
    for i, conn in enumerate(geometry.elem_conn):
        label = named.get(int(block[i]), '') if i < len(block) else ''
        if not label:
            continue
        for node in conn:
            counted = votes.setdefault(int(node), {})
            counted[label] = counted.get(label, 0) + 1
    return {node: max(counted, key=lambda n: (counted[n], -order.get(n, 0)))
            for node, counted in votes.items()}


def _strains_nothing(shapes: np.ndarray, stiffness: np.ndarray) -> np.ndarray:
    """Which of these modes store no strain energy, and so sit at 0 Hz.

    Zero-energy is asked of the mode rather than of the structure, so a
    constrained model simply has none — a cantilever's modes all strain
    something — and a genuine mechanism (a node nothing connects to, a
    hinge built by accident) is found on the same footing as rigid-body
    motion. Both really are at zero frequency; the distinction between
    them is a modeling question, not a numerical one.
    """
    energy = np.sum(shapes * (stiffness @ shapes), axis=0)
    # the same sum with every cancellation removed: how big the terms were
    # before they annihilated each other
    magnitude = np.sum(np.abs(shapes) * (abs(stiffness) @ np.abs(shapes)),
                       axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(energy) / magnitude
    return np.nan_to_num(ratio, nan=0.0) < ZERO_ENERGY


# ---- element matrices -----------------------------------------------------


def _element_axes(start: np.ndarray, end: np.ndarray, orientation) -> np.ndarray:
    """The 3x3 rotation whose rows are the element's local axes.

    Local x runs from the first node to the second. The orientation vector
    fixes the roll about it by naming a direction the local x-y plane must
    contain — the same job Nastran's orientation vector does. Without one,
    global Z is used, falling back to global Y for a member that is nearly
    vertical and would otherwise have no plane at all.
    """
    axis = end - start
    length = np.linalg.norm(axis)
    axis = axis / length
    if orientation is None:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(axis @ reference)) > 0.99:
            reference = np.array([0.0, 1.0, 0.0])
    else:
        reference = np.asarray(orientation, dtype=np.float64)
        if np.linalg.norm(reference) == 0.0:
            raise ValueError('the orientation vector has no direction')
        reference = reference / np.linalg.norm(reference)
        if abs(float(axis @ reference)) > 0.9999:
            raise ValueError('the orientation vector lies along the beam, so '
                             'it names no plane')
    local_y = reference - float(axis @ reference) * axis
    local_y = local_y / np.linalg.norm(local_y)
    local_z = np.cross(axis, local_y)
    return np.array([axis, local_y, local_z])


def _block_diagonal(rotation: np.ndarray, times: int) -> np.ndarray:
    out = np.zeros((3 * times, 3 * times), dtype=np.float64)
    for i in range(times):
        out[3 * i:3 * i + 3, 3 * i:3 * i + 3] = rotation
    return out


def _plate_frame(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                 p4: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The element's own axes and side lengths, refusing non-rectangles.

    Local x runs along edge 1-2, local y along edge 1-4, local z is
    their normal. The checks are one relative tolerance: the corner
    angle must be right and corner 3 must sit where the other three
    say it does — which also catches a warped (non-planar) quad,
    because a flat rectangle is the only shape that passes both.
    """
    edge_x = np.asarray(p2, dtype=np.float64) - p1
    edge_y = np.asarray(p4, dtype=np.float64) - p1
    a, b = float(np.linalg.norm(edge_x)), float(np.linalg.norm(edge_y))
    if a == 0.0 or b == 0.0:
        raise ValueError('a plate edge has zero length')
    e1, e2 = edge_x / a, edge_y / b
    tolerance = 1e-6 * max(a, b)
    if abs(float(np.dot(edge_x, edge_y))) > tolerance * max(a, b):
        raise ValueError(
            'plate corners do not form a rectangle: the corner angle at '
            'node 1 is not square. Rectangular elements only — see Plate')
    if float(np.linalg.norm(p3 - (p1 + edge_x + edge_y))) > tolerance:
        raise ValueError(
            'plate corners do not form a rectangle: corner 3 is not '
            'where corners 1, 2 and 4 put it. Rectangular elements '
            'only — see Plate')
    return np.array([e1, e2, np.cross(e1, e2)]), a, b


def _plate_frames(corners: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`_plate_frame` for many plates at once: corners (E, 4, 3) in,
    rotations (E, 3, 3) and side lengths a and b (E,) out. A plate that
    fails a check is handed to `_plate_frame`, so the refusal is the
    one-plate refusal, word for word."""
    edge_x = corners[:, 1] - corners[:, 0]
    edge_y = corners[:, 3] - corners[:, 0]
    a = np.linalg.norm(edge_x, axis=1)
    b = np.linalg.norm(edge_y, axis=1)
    tolerance = 1e-6 * np.maximum(a, b)
    bad = ((a == 0.0) | (b == 0.0)
           | (np.abs(np.einsum('ei,ei->e', edge_x, edge_y))
              > tolerance * np.maximum(a, b))
           | (np.linalg.norm(corners[:, 2] - (corners[:, 0] + edge_x + edge_y),
                             axis=1) > tolerance))
    if bad.any():
        _plate_frame(*corners[int(np.flatnonzero(bad)[0])])
    e1 = edge_x / a[:, None]
    e2 = edge_y / b[:, None]
    return np.stack([e1, e2, np.cross(e1, e2)], axis=1), a, b


#: transverse shear correction for a homogeneous section: the parabolic
#: shear distribution carries 5/6 of what a uniform one would
SHEAR_CORRECTION = 5.0 / 6.0

#: how much artificial drilling stiffness a plate carries, as a
#: fraction of G*t. Any small positive number serves: the penalty
#: exists so the in-plane rotation is not a mechanism. It acts on the
#: gap between the drilling rotation and the rotation the membrane
#: displacements themselves describe, (v,x - u,y)/2 — so a rigid
#: rotation, where the two agree exactly, strains it not at all and
#: the rigid-mode count stays six. Penalizing only the *differences*
#: between the four nodal drillings was tried first and leaves a
#: mesh-wide uniform spin with no displacement as a seventh
#: zero-frequency mode, which is how this constant earned its
#: sentence.
DRILLING_FRACTION = 1e-3


def _plate_matrices(material: Material, thickness: float, a: float,
                    b: float) -> tuple[np.ndarray, np.ndarray]:
    """Local stiffness and consistent mass for the rectangular shell.

    24x24, six DOFs per node in DIRECTIONS order, nodes at
    (0,0) (a,0) (a,b) (0,b) in the element's own plane. Membrane and
    bending integrate 2x2 Gauss (exact for the rectangle); the
    transverse shear is the MITC4 assumed field — gamma_xz tied at the
    midpoints of the two eta edges and interpolated linearly in eta,
    gamma_yz the mirror of that — which is what keeps the thin limit
    honest instead of shear-locked.

    Sign conventions, written down because they were derived once and
    should not be re-derived: the tilt of the normal toward +x is
    +theta_y and toward +y is -theta_x, so the curvatures are
    (theta_y,x | -theta_x,y | theta_y,y - theta_x,x) and the shears
    (w,x + theta_y | w,y - theta_x). Both vanish identically on every
    rigid motion, which the rigid-mode test holds to machine zero.
    """
    E, nu = material.youngs_modulus, material.poissons_ratio
    G, rho, t = material.shear_modulus, material.density, thickness
    plane = E / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0],
                                            [nu, 1.0, 0.0],
                                            [0.0, 0.0, (1.0 - nu) / 2.0]])
    d_membrane = t * plane
    d_bending = t ** 3 / 12.0 * plane
    d_shear = SHEAR_CORRECTION * G * t * np.eye(2)

    def basis(xi: float, eta: float):
        signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        n = np.array([(1 + xi * sx) * (1 + eta * sy) / 4.0
                      for sx, sy in signs])
        dx = np.array([sx * (1 + eta * sy) / 4.0 * 2.0 / a
                       for sx, sy in signs])
        dy = np.array([sy * (1 + xi * sx) / 4.0 * 2.0 / b
                       for sx, sy in signs])
        return n, dx, dy

    def shear_rows(xi: float, eta: float) -> np.ndarray:
        n, dx, dy = basis(xi, eta)
        rows = np.zeros((2, 24))
        for i in range(4):
            rows[0, 6 * i + 2] = dx[i]          # gamma_xz = w,x + theta_y
            rows[0, 6 * i + 4] = n[i]
            rows[1, 6 * i + 2] = dy[i]          # gamma_yz = w,y - theta_x
            rows[1, 6 * i + 3] = -n[i]
        return rows

    stiffness = np.zeros((24, 24))
    consistent = np.zeros((24, 24))
    gauss = 1.0 / math.sqrt(3.0)
    area_weight = a * b / 4.0                    # dA per unit natural area
    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            n, dx, dy = basis(xi, eta)
            b_membrane = np.zeros((3, 24))
            b_bending = np.zeros((3, 24))
            fields = np.zeros((6, 24))
            for i in range(4):
                col = 6 * i
                b_membrane[0, col] = dx[i]
                b_membrane[1, col + 1] = dy[i]
                b_membrane[2, col] = dy[i]
                b_membrane[2, col + 1] = dx[i]
                b_bending[0, col + 4] = dx[i]
                b_bending[1, col + 3] = -dy[i]
                b_bending[2, col + 3] = -dx[i]
                b_bending[2, col + 4] = dy[i]
                for field in range(6):
                    fields[field, col + field] = n[i]
            # the assumed shear: the raw rows evaluated at the tying
            # points, blended linearly across the element
            b_shear = np.zeros((2, 24))
            b_shear[0] = ((1 - eta) * shear_rows(0.0, -1.0)[0]
                          + (1 + eta) * shear_rows(0.0, 1.0)[0]) / 2.0
            b_shear[1] = ((1 - xi) * shear_rows(-1.0, 0.0)[1]
                          + (1 + xi) * shear_rows(1.0, 0.0)[1]) / 2.0

            # the drilling tie: theta_z minus the rotation the membrane
            # field describes, penalized softly (see DRILLING_FRACTION)
            b_drilling = np.zeros((1, 24))
            for i in range(4):
                col = 6 * i
                b_drilling[0, col] = dy[i] / 2.0
                b_drilling[0, col + 1] = -dx[i] / 2.0
                b_drilling[0, col + 5] = n[i]

            stiffness += area_weight * (
                b_membrane.T @ d_membrane @ b_membrane
                + b_bending.T @ d_bending @ b_bending
                + b_shear.T @ d_shear @ b_shear
                + DRILLING_FRACTION * G * t
                * (b_drilling.T @ b_drilling))
            # translations carry rho t, the bending rotations the
            # section's rotary inertia rho t^3/12. The drilling
            # rotation carries a thousandth of that — not zero, because
            # a massless DOF breaks the Cholesky factorization the
            # eigensolution stands on, and not the full value, because
            # inertia against penalty stiffness sets where the drilling
            # *artifact* modes land: at the full inertia a half-inch
            # plate grew a wall of a hundred and sixty fake modes at
            # 4.2 kHz, mid-band; at a thousandth they sit near 130 kHz,
            # far above anything the element is valid for. Rigid
            # rotation about the normal loses nothing measurable — its
            # inertia is the translations' r^2 terms, not this.
            spin = rho * t ** 3 / 12.0
            weights = np.diag([rho * t] * 3 + [spin, spin, spin * 1e-3])
            consistent += area_weight * (fields.T @ weights @ fields)

    return stiffness, consistent


def _triangle_frame(p1: np.ndarray, p2: np.ndarray,
                    p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The element's own axes and its corners in its own plane.

    Local x runs along edge 1-2, local z is the normal the corners
    turn counterclockwise about, local y completes the frame. Returns
    the rotation (rows are the axes) and the 3x2 corner coordinates in
    it, node 1 at the origin. A zero-area triangle is refused.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    edge_x = np.asarray(p2, dtype=np.float64) - p1
    edge_3 = np.asarray(p3, dtype=np.float64) - p1
    normal = np.cross(edge_x, edge_3)
    a = float(np.linalg.norm(edge_x))
    twice_area = float(np.linalg.norm(normal))
    if a == 0.0 or twice_area <= 1e-12 * max(a, float(np.linalg.norm(edge_3))) ** 2:
        raise ValueError('a triangle has no area: its corners are collinear '
                         'or coincide')
    e1 = edge_x / a
    e3 = normal / twice_area
    e2 = np.cross(e3, e1)
    rotation = np.array([e1, e2, e3])
    xy = np.array([[0.0, 0.0],
                   [a, 0.0],
                   [float(edge_3 @ e1), float(edge_3 @ e2)]])
    return rotation, xy


def _triangle_area(xy: np.ndarray) -> float:
    (x1, y1), (x2, y2), (x3, y3) = xy
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _triangle_matrices(material: Material, thickness: float,
                       xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local stiffness and consistent mass for the triangular shell.

    18x18, six DOFs per node in DIRECTIONS order, corners at `xy` in
    the element's own plane. Linear shape functions throughout, so
    the membrane strains and the curvatures are constant over the
    element and their products integrate exactly with the area. The
    transverse shear is the MITC3 assumed field: the covariant shear
    along each natural direction is taken at that edge's midpoint and
    the field is completed with the one linear term that makes the
    third edge's tangential shear agree at its midpoint —

        e_rt = e_rt(P1) + c s,   e_st = e_st(P2) - c r,
        c = e_rt(P3) - e_rt(P1) - e_st(P3) + e_st(P2),

    P1, P2, P3 the midpoints of the edges along r, along s, and the
    hypotenuse. Derived here by matching the linear raw field at the
    three tying points rather than copied, and the same field falls
    out of the constant Jacobian either way. Linear in (r, s), so its
    energy integrates exactly with the three-point midpoint rule, and
    the consistent mass with it.

    Sign conventions are the quad's, written down there: curvatures
    (theta_y,x | -theta_x,y | theta_y,y - theta_x,x), shears
    (w,x + theta_y | w,y - theta_x), the drilling tie penalizing
    theta_z against (v,x - u,y)/2, and the rotary inertias
    rho t^3/12 with a thousandth of that on the drilling rotation.
    Rigid motion strains none of it, which the rigid-mode test holds
    to machine zero.
    """
    E, nu = material.youngs_modulus, material.poissons_ratio
    G, rho, t = material.shear_modulus, material.density, thickness
    plane = E / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0],
                                            [nu, 1.0, 0.0],
                                            [0.0, 0.0, (1.0 - nu) / 2.0]])
    d_membrane = t * plane
    d_bending = t ** 3 / 12.0 * plane
    d_shear = SHEAR_CORRECTION * G * t * np.eye(2)

    area = _triangle_area(xy)
    (x1, y1), (x2, y2), (x3, y3) = xy
    # the linear shape functions' constant gradients (N_i = a_i + b_i x + c_i y)
    dx = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
    dy = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)
    # natural coordinates r, s with node 1 at the origin, node 2 at r = 1
    # and node 3 at s = 1: x = x1 + r (x2 - x1) + s (x3 - x1), so the
    # Jacobian rows are the two edges from node 1, constant everywhere
    jacobian = np.array([[x2 - x1, y2 - y1],
                         [x3 - x1, y3 - y1]])
    inverse = np.linalg.inv(jacobian)

    def basis(r: float, s: float) -> np.ndarray:
        return np.array([1.0 - r - s, r, s])

    def raw_shear(r: float, s: float) -> np.ndarray:
        """gamma_xz = w,x + theta_y and gamma_yz = w,y - theta_x at a
        point, as rows over the 18 DOFs."""
        n = basis(r, s)
        rows = np.zeros((2, 18))
        for i in range(3):
            rows[0, 6 * i + 2] = dx[i]
            rows[0, 6 * i + 4] = n[i]
            rows[1, 6 * i + 2] = dy[i]
            rows[1, 6 * i + 3] = -n[i]
        return rows

    def covariant(r: float, s: float) -> np.ndarray:
        # e_rt = x,r gamma_xz + y,r gamma_yz, e_st likewise with ,s
        return jacobian @ raw_shear(r, s)

    tie_r = covariant(0.5, 0.0)      # P1, the midpoint of the r edge
    tie_s = covariant(0.0, 0.5)      # P2, the midpoint of the s edge
    tie_h = covariant(0.5, 0.5)      # P3, the midpoint of the hypotenuse
    c = tie_h[0] - tie_r[0] - tie_h[1] + tie_s[1]

    def assumed_shear(r: float, s: float) -> np.ndarray:
        e = np.vstack([tie_r[0] + s * c, tie_s[1] - r * c])
        return inverse @ e                # back to gamma_xz, gamma_yz

    b_membrane = np.zeros((3, 18))
    b_bending = np.zeros((3, 18))
    b_drilling = np.zeros((1, 18))
    for i in range(3):
        col = 6 * i
        b_membrane[0, col] = dx[i]
        b_membrane[1, col + 1] = dy[i]
        b_membrane[2, col] = dy[i]
        b_membrane[2, col + 1] = dx[i]
        b_bending[0, col + 4] = dx[i]
        b_bending[1, col + 3] = -dy[i]
        b_bending[2, col + 3] = -dx[i]
        b_bending[2, col + 4] = dy[i]
        b_drilling[0, col] = dy[i] / 2.0
        b_drilling[0, col + 1] = -dx[i] / 2.0
    # the constant parts integrate exactly with the area; the drilling
    # tie's theta_z term and the assumed shear vary linearly, so they
    # and the mass take the three-point midpoint rule, exact for
    # quadratics
    stiffness = area * (b_membrane.T @ d_membrane @ b_membrane
                        + b_bending.T @ d_bending @ b_bending)
    consistent = np.zeros((18, 18))
    spin = rho * t ** 3 / 12.0
    weights = np.diag([rho * t] * 3 + [spin, spin, spin * 1e-3])
    for r, s in ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):
        n = basis(r, s)
        drilling = b_drilling.copy()
        fields = np.zeros((6, 18))
        for i in range(3):
            drilling[0, 6 * i + 5] = n[i]
            for field in range(6):
                fields[field, 6 * i + field] = n[i]
        b_shear = assumed_shear(r, s)
        stiffness += area / 3.0 * (
            b_shear.T @ d_shear @ b_shear
            + DRILLING_FRACTION * G * t * (drilling.T @ drilling))
        consistent += area / 3.0 * (fields.T @ weights @ fields)
    return stiffness, consistent


def _beam_stiffness(material: Material, section: Section, length: float) -> np.ndarray:
    """The 12x12 local stiffness of an Euler-Bernoulli beam.

    Degrees of freedom run u, v, w, rx, ry, rz at the first node then the
    second. Bending in the local x-y plane pairs v with rz and uses iz;
    bending in x-z pairs w with ry and uses iy, with the signs of the
    coupling terms reversed because a positive rotation about y moves a
    point in the *negative* z direction.
    """
    e, g = material.youngs_modulus, material.shear_modulus
    a, iy, iz, j = section.area, section.iy, section.iz, section.j
    length2, length3 = length ** 2, length ** 3
    k = np.zeros((12, 12), dtype=np.float64)

    k[0, 0] = k[6, 6] = e * a / length
    k[0, 6] = -e * a / length

    k[3, 3] = k[9, 9] = g * j / length
    k[3, 9] = -g * j / length

    k[1, 1] = k[7, 7] = 12.0 * e * iz / length3
    k[1, 7] = -12.0 * e * iz / length3
    k[1, 5] = k[1, 11] = 6.0 * e * iz / length2
    k[5, 7] = k[7, 11] = -6.0 * e * iz / length2
    k[5, 5] = k[11, 11] = 4.0 * e * iz / length
    k[5, 11] = 2.0 * e * iz / length

    k[2, 2] = k[8, 8] = 12.0 * e * iy / length3
    k[2, 8] = -12.0 * e * iy / length3
    k[2, 4] = k[2, 10] = -6.0 * e * iy / length2
    k[4, 8] = k[8, 10] = 6.0 * e * iy / length2
    k[4, 4] = k[10, 10] = 4.0 * e * iy / length
    k[4, 10] = 2.0 * e * iy / length

    return k + np.triu(k, 1).T


def _beam_mass(material: Material, section: Section, length: float) -> np.ndarray:
    """The 12x12 consistent mass matrix of the same element.

    Consistent rather than lumped: the cubic shape functions that gave the
    stiffness give the mass too, so the pair is a proper Rayleigh-Ritz
    reduction and the frequencies converge from above. A lumped diagonal
    would converge from below and give the rotations no inertia at all,
    which would leave the mass matrix singular wherever nothing else does.
    """
    density, area = material.density, section.area
    unit = density * area * length / 420.0
    m = np.zeros((12, 12), dtype=np.float64)

    m[0, 0] = m[6, 6] = 140.0 * unit
    m[0, 6] = 70.0 * unit

    # torsion carries the polar second moment of area, not the torsion
    # constant: warping changes the stiffness, not where the material is
    rotary = density * section.polar * length / 6.0
    m[3, 3] = m[9, 9] = 2.0 * rotary
    m[3, 9] = rotary

    m[1, 1] = m[7, 7] = 156.0 * unit
    m[1, 7] = 54.0 * unit
    m[1, 5] = 22.0 * length * unit
    m[1, 11] = -13.0 * length * unit
    m[5, 7] = 13.0 * length * unit
    m[7, 11] = -22.0 * length * unit
    m[5, 5] = m[11, 11] = 4.0 * length ** 2 * unit
    m[5, 11] = -3.0 * length ** 2 * unit

    m[2, 2] = m[8, 8] = 156.0 * unit
    m[2, 8] = 54.0 * unit
    m[2, 4] = -22.0 * length * unit
    m[2, 10] = 13.0 * length * unit
    m[4, 8] = -13.0 * length * unit
    m[8, 10] = 22.0 * length * unit
    m[4, 4] = m[10, 10] = 4.0 * length ** 2 * unit
    m[4, 10] = -3.0 * length ** 2 * unit

    return m + np.triu(m, 1).T


# ---- solids ------------------------------------------------------------

#: natural coordinates of the trilinear hexahedron's corners, bottom face
#: then top, the same way round
_HEX_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                        dtype=np.float64)
_GAUSS_2 = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
#: the three-point rule on the triangle, in area coordinates (L2, L3)
#: with weights summing to the unit triangle's half
_TRIANGLE_3 = (((1 / 6, 1 / 6), 1 / 6), ((2 / 3, 1 / 6), 1 / 6),
               ((1 / 6, 2 / 3), 1 / 6))


def _isotropic(material: Material) -> np.ndarray:
    """The 6x6 elasticity matrix, strains ordered xx yy zz xy yz xz with
    engineering shears."""
    E, nu = material.youngs_modulus, material.poissons_ratio
    lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = material.shear_modulus
    d = np.zeros((6, 6))
    d[:3, :3] = lame
    d[np.arange(3), np.arange(3)] += 2.0 * mu
    d[3:, 3:] = np.eye(3) * mu
    return d


def _strain_rows(derivatives: np.ndarray) -> np.ndarray:
    """B (6 x 3n) from each node's (dN/dx, dN/dy, dN/dz)."""
    n = len(derivatives)
    b = np.zeros((6, 3 * n))
    dx, dy, dz = derivatives.T
    cols = 3 * np.arange(n)
    b[0, cols], b[1, cols + 1], b[2, cols + 2] = dx, dy, dz
    b[3, cols], b[3, cols + 1] = dy, dx
    b[4, cols + 1], b[4, cols + 2] = dz, dy
    b[5, cols], b[5, cols + 2] = dz, dx
    return b


def _hex_basis(xi: float, eta: float, zeta: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """The eight trilinear functions and their natural derivatives (8, 3)."""
    c = _HEX_CORNERS
    n = (1 + xi * c[:, 0]) * (1 + eta * c[:, 1]) * (1 + zeta * c[:, 2]) / 8.0
    d = np.column_stack([
        c[:, 0] * (1 + eta * c[:, 1]) * (1 + zeta * c[:, 2]),
        (1 + xi * c[:, 0]) * c[:, 1] * (1 + zeta * c[:, 2]),
        (1 + xi * c[:, 0]) * (1 + eta * c[:, 1]) * c[:, 2]]) / 8.0
    return n, d


def _hex_matrices(material: Material, xyz: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Stiffness and consistent mass of the eight-node hexahedron with
    incompatible modes, 24x24, three translations per node.

    Wilson's three bubble modes (1 - xi^2, 1 - eta^2, 1 - zeta^2), each
    in three directions, ride along as nine internal degrees of freedom
    and are condensed out. Their strains are taken from the centroid's
    Jacobian and scaled by its determinant over the local one (Taylor,
    Beresford and Wilson, 1976), which is what lets a brick that is not
    a parallelepiped still represent a constant strain exactly. Without
    the modes a trilinear brick in bending carries parasitic shear and
    comes out much too stiff; with them one layer through a plate's
    thickness bends like the plate. Integrated 2x2x2, exact for the
    mass of a parallelepiped and standard for the stiffness. The
    Jacobian's sign is taken as read: an element numbered the other way
    round is the same element.
    """
    d = _isotropic(material)
    rho = material.density
    _n0, d0 = _hex_basis(0.0, 0.0, 0.0)
    jacobian_0 = d0.T @ xyz
    det_0 = abs(np.linalg.det(jacobian_0))
    inverse_0_t = np.linalg.inv(jacobian_0).T
    k_uu = np.zeros((24, 24))
    k_ua = np.zeros((24, 9))
    k_aa = np.zeros((9, 9))
    mass = np.zeros((24, 24))
    for xi in _GAUSS_2:
        for eta in _GAUSS_2:
            for zeta in _GAUSS_2:
                n, dn = _hex_basis(xi, eta, zeta)
                jacobian = dn.T @ xyz
                det = abs(np.linalg.det(jacobian))
                b_u = _strain_rows(dn @ np.linalg.inv(jacobian).T)
                # the bubble modes' natural derivatives, through the
                # centroid's Jacobian
                bubble = np.diag([-2.0 * xi, -2.0 * eta, -2.0 * zeta])
                b_a = _strain_rows(bubble @ inverse_0_t) * (det_0 / det)
                k_uu += det * (b_u.T @ d @ b_u)
                k_ua += det * (b_u.T @ d @ b_a)
                k_aa += det * (b_a.T @ d @ b_a)
                fields = np.zeros((3, 24))
                for i in range(8):
                    fields[:, 3 * i:3 * i + 3] = n[i] * np.eye(3)
                mass += det * rho * (fields.T @ fields)
    stiffness = k_uu - k_ua @ np.linalg.solve(k_aa, k_ua.T)
    return (stiffness + stiffness.T) / 2.0, (mass + mass.T) / 2.0


def _wedge_basis(l2: float, l3: float, zeta: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The six linear wedge functions — a linear triangle swept
    between zeta = -1 and 1 — and their natural derivatives (6, 3) with
    respect to (L2, L3, zeta)."""
    l1 = 1.0 - l2 - l3
    lower, upper = (1.0 - zeta) / 2.0, (1.0 + zeta) / 2.0
    n = np.array([l1 * lower, l2 * lower, l3 * lower,
                  l1 * upper, l2 * upper, l3 * upper])
    d = np.array([[-lower, -lower, -l1 / 2.0],
                  [lower, 0.0, -l2 / 2.0],
                  [0.0, lower, -l3 / 2.0],
                  [-upper, -upper, l1 / 2.0],
                  [upper, 0.0, l2 / 2.0],
                  [0.0, upper, l3 / 2.0]])
    return n, d


def _wedge_matrices(material: Material, xyz: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Stiffness and consistent mass of the six-node wedge, 18x18: the
    three-point triangle rule by two Gauss points along its length."""
    d = _isotropic(material)
    rho = material.density
    stiffness = np.zeros((18, 18))
    mass = np.zeros((18, 18))
    for (l2, l3), weight in _TRIANGLE_3:
        for zeta in _GAUSS_2:
            n, dn = _wedge_basis(l2, l3, zeta)
            jacobian = dn.T @ xyz
            det = abs(np.linalg.det(jacobian)) * weight
            b = _strain_rows(dn @ np.linalg.inv(jacobian).T)
            stiffness += det * (b.T @ d @ b)
            fields = np.zeros((3, 18))
            for i in range(6):
                fields[:, 3 * i:3 * i + 3] = n[i] * np.eye(3)
            mass += det * rho * (fields.T @ fields)
    return (stiffness + stiffness.T) / 2.0, (mass + mass.T) / 2.0


def _tet_matrices(material: Material, xyz: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Stiffness and consistent mass of the constant-strain tetrahedron,
    12x12, in closed form."""
    d = _isotropic(material)
    edges = xyz[1:] - xyz[0]
    volume = abs(np.linalg.det(edges)) / 6.0
    # dL_i/dx from the inverse of the edge matrix: the natural
    # coordinates L2, L3, L4 are affine in x, and L1 = 1 - the rest
    inverse = np.linalg.inv(edges)               # d(L2, L3, L4)/dx, by columns
    derivatives = np.vstack([-inverse.sum(axis=1), inverse.T])
    b = _strain_rows(derivatives)
    stiffness = volume * (b.T @ d @ b)
    mass = np.zeros((12, 12))
    for i in range(4):
        for j in range(4):
            mass[3 * i:3 * i + 3, 3 * j:3 * j + 3] = (
                np.eye(3) * (2.0 if i == j else 1.0))
    mass *= material.density * volume / 20.0
    return stiffness, mass


def _solid_matrices(material: Material, xyz: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(stiffness, mass) of a solid by its node count."""
    if len(xyz) == 8:
        return _hex_matrices(material, xyz)
    if len(xyz) == 6:
        return _wedge_matrices(material, xyz)
    return _tet_matrices(material, xyz)


def _solid_volume(xyz: np.ndarray) -> float:
    """The volume of a hexahedron, wedge or tetrahedron by its nodes,
    integrated the way its matrices are, so the two agree; negative for
    an element numbered inside out, zero for a flat one."""
    if len(xyz) == 4:
        return float(np.linalg.det(xyz[1:] - xyz[0]) / 6.0)
    if len(xyz) == 6:
        return float(sum(np.linalg.det((_wedge_basis(l2, l3, zeta)[1]).T @ xyz)
                         * weight
                         for (l2, l3), weight in _TRIANGLE_3
                         for zeta in _GAUSS_2))
    return float(sum(np.linalg.det((_hex_basis(xi, eta, zeta)[1]).T @ xyz)
                     for xi in _GAUSS_2 for eta in _GAUSS_2
                     for zeta in _GAUSS_2))

