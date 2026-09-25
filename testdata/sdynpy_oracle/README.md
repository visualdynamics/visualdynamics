# The sdynpy oracle

`oracle.npz` holds sdynpy's own answers — PSDs, the CPSD matrix, FRFs
(H1, H2, Hv), coherence, an SRS, and RMS values — for input signals
stored in the same file, so `tests/test_sdynpy_oracle.py` can check
Visual Dynamics' spectral computations against the tool its audience
already trusts.

## Why data, and never an import

sdynpy is GPL-3.0, and Visual Dynamics keeps every licensing option
open by never importing it (`AGENTS.md` hard rule 1;
`tests/test_license_boundary.py` enforces it over the package *and*
the tests). Running a GPL program privately carries no conditions
(GPLv3 §2), and a program's numerical output is not a covered work
(GPLv3 §0: output is covered only if it constitutes a covered work) —
so sdynpy runs in the private `visualdynamics-generators` repository,
where the GPL tools are already used as tools, and only its numbers
travel here. This is the same shape as the ADF corpus: the writing
tool's own readings frozen beside the sample files, the deriving tool
never in the tree.

Regenerate with the interpreter that has sdynpy:

    cd ~/visualdynamics-generators
    ~/.pyenv/versions/3.13.2/bin/python3 generate_sdynpy_oracle.py

The file's `provenance` key records the sdynpy/numpy/scipy versions,
the seed, and the date it was made.

`rigid.npz` is the same arrangement for rigid-body mode shapes
(`generate_rigid_oracle.py`): a five-node geometry with one node
measured in a turned frame, unit shapes about a point off the
centroid, and mass-normalized shapes with a *diagonal* inertia tensor.
A coupled tensor is deliberately not frozen — sdynpy takes its
diagonal unless asked for principal axes, and Visual Dynamics always
takes the principal axes (`tests/test_rigid.py` proves which is
orthonormal) — so the two would disagree there by design.

Three more files, each from its own `generate_*_oracle.py`, added
2026-09-23 when the computations were audited for what had only ever
been checked against themselves:

- `modal_frf.npz` — FRFs synthesized from a modal model (unequal
  modal masses, damping 1.2% to 8%) as displacement, velocity and
  acceleration, once with real shapes and once with complex ones
  (`tests/test_modal_frf_oracle.py`). Complex shapes take the
  pole-plus-conjugate form with modal A. sdynpy divides the conjugate
  term by modal A rather than its conjugate, which is right only for a
  real modal A, so a real one is what is frozen. Closed-form physics
  checks both forms independently, a complex modal A included
  (`tests/test_frf_synthesis_exact.py`).
- `octave.npz` — PSD banding at 1, 3, 6 and 12 bands per octave, real
  and complex, on sdynpy's own band edges (`tests/test_octave_oracle.py`).
  Bands are matched by edge, not label: sdynpy names a band by the
  arithmetic mean of its edges, Visual Dynamics by the geometric.
- `beam.npz` — the assembled 30x30 mass and stiffness matrices of a
  small 3D beam frame (`tests/test_beam_oracle.py`). The section
  numbers are handed to sdynpy rather than derived by it, because its
  rectangle helper's shear modulus and torsion constant are not the
  textbook ones; the generator's docstring has the details.

## What is covered, and what deliberately is not

Covered: PSDs (hann, rectangle, flattop; half- and no-overlap), the
full CPSD matrix, FRFs (H1, Hv, H2 single and square), multiple
coherence (both references, and the single-reference identity), the
CMIF, MAC on complex shapes, the SRS family (maximax at Q of 10 and
50, the signed peak types), and band RMS.

Then, in the files above: FRF synthesis from real modes, octave
banding, and the beam element with its assembly.

Deliberately not covered, with reasons at the implementation sites:
linear spectra (ours is numpy's rfft unscaled by convention), and
integrate/differentiate (ours applies a drift high-pass by design).

## An agreement oracle, not a truth oracle

sdynpy has around 13% test coverage (Brandon, 2026-08-28). Agreement
at machine precision — which is what these files show, worst case
1e-14 relative — demonstrates that two independent implementations of
the same published definitions produce the same numbers. A
*disagreement* is a flag to investigate from first principles, never a
verdict for either side; what the first build of this oracle found is
recorded in `tests/test_sdynpy_oracle.py`'s docstring, including the
one "discrepancy" that turned out to be the oracle script itself.
