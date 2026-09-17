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

## What is covered, and what deliberately is not

Covered: PSDs (hann, rectangle, flattop; half- and no-overlap), the
full CPSD matrix, FRFs (H1, Hv, H2 single and square), multiple
coherence (both references, and the single-reference identity), the
CMIF, MAC on complex shapes, the SRS family (maximax at Q of 10 and
50, the signed peak types), and band RMS.

Deliberately not covered, with reasons at the implementation sites:
octave banding (sdynpy's lives in a documentation-support class whose
conventions were not verified against ours), linear spectra (ours is
numpy's rfft unscaled by convention), and integrate/differentiate
(ours applies a drift high-pass by design).

## An agreement oracle, not a truth oracle

sdynpy has around 13% test coverage (Brandon, 2026-08-28). Agreement
at machine precision — which is what these files show, worst case
1e-14 relative — demonstrates that two independent implementations of
the same published definitions produce the same numbers. A
*disagreement* is a flag to investigate from first principles, never a
verdict for either side; what the first build of this oracle found is
recorded in `tests/test_sdynpy_oracle.py`'s docstring, including the
one "discrepancy" that turned out to be the oracle script itself.
