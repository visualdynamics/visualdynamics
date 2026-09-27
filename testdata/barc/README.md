# The BARC's reference modes

`reference_modes.npz` holds a finite element model's free-free modes of
the BARC (Box Assembly with Removable Component), as shared on the SEM
Dynamic Substructuring Focus Group wiki:
<https://wiki.sem.org/wiki/BARC> ("Random Vibration Data",
`BARC_field_and_lab_data.mat`, its `field_modes` and `field_geometry`).
Frozen by `testdata/freeze_barc_reference.py`.

Unlike everything else in `testdata/`, these numbers are not the
package's own: they are the published reference that
`visualdynamics.demo.barc` — built from planes with this package — is
checked against, frequency by frequency and by MAC.

| array | holds |
|---|---|
| `frequency` | 30 modes, Hz; the first six are rigid-body |
| `damping` | the fraction of critical each was given (0.01) |
| `dof` | 118 DOFs, `'101X+'`: node and a global axis |
| `shape` | 118 x 30, the modes at those DOFs (their own normalization; only MAC reads them) |
| `coordinates` | 118 x 3, where each DOF is, in inches, in the reference model's frame |

**Frames.** The reference model's frame is the BARC solid model's (the
STEP file shared on the same wiki) moved by (0, −3, −1.5) in: its
Bench top reads y = 2.128 where the STEP's reads 5.128. `demo.barc`
builds in the STEP's frame and shifts these points by (0, +3, +1.5) in
to compare.
