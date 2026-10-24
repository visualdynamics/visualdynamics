# Checking the matrices of a multi-input test

A multi-input test stands on two matrices at every frequency line. The
plant matrix \(H\), one row per control channel and one column per
drive, is what the controller inverts to turn a control error into a
drive correction. The cross-spectral density matrix \(S\), one row and
one column per control channel, is what the specification asks the
channels to hold: their autospectra on the diagonal and, off it, how
each pair moves together. Each can be wrong in a way that no single
record shows, and both can be checked at the desk before the article is
on the table.

## Conditioning: how near singular the plant is

The singular values of \(H\) at a line are its gains in the
combinations of drives it responds to most and least. Their ratio is
the condition number:

\[
\kappa = \frac{\sigma_\text{max}}{\sigma_\text{min}}
\]

It is dimensionless, and it bounds how far a relative error in the
measured plant can grow in its inverse: an FRF good to one percent,
inverted at a condition number of 100, can give a drive wrong by up to
a hundred percent. Two drives that reach the control channels in
nearly the same proportions make two nearly equal columns, a small
\(\sigma_\text{min}\) and a large \(\kappa\), and the inverse that
separates them is mostly noise. A line is flagged where \(\kappa\) is
over the limit, 100 by default (a controller has a setting of its own
for it). The effective rank at that limit counts the singular values at
least \(\sigma_\text{max}/100\), so a line is flagged exactly where the
effective rank falls under the full rank.

The same reading is made of the drives' own cross-spectral matrix,
\(G_{ff}\), whose singular values are its eigenvalues. Two drives with
equal power and a coherence of \(\rho^2\) give \((1+\rho)/(1-\rho)\):
99 at \(\rho = 0.98\), 199 at 0.99. Drives that are nearly copies of
each other make it nearly singular, and an FRF estimated from them
inverts it.

The matrix has to be in one unit. The singular values of a matrix whose
rows are in different units change when one row's unit does, so a
condition number of an acceleration row beside an angular-acceleration
row would describe the units and not the plant; such a selection is
refused. So is an FRF from one reference, which has nothing to compare,
and a matrix with a response missing against some reference.

![The singular values of a two-drive plant with the rank threshold
dotted, and the condition number beneath with the lines over 100
hatched](images/mimo-conditioning.png)

The figure is a plant of two drives and two control accelerometers
through two modes at 2 % damping, the first at 100 Hz moving both
drives together and the second at 400 Hz moving them against each other
(`tools/make_concept_figures.py`). At low frequency both modes answer
and the condition number is 16, the square of the ratio of the two
frequencies. At the first mode that mode alone answers, the two columns
are one, and the condition number reaches 375 between 92.5 and
106.2 Hz, hatched: at that resonance a controller cannot tell the two
drives apart. At 250 Hz it is 1.9.

## Realizability: whether any signals have the specification

The cross-spectral matrix of any set of signals is Hermitian (the term
for channel \(j\) against \(i\) is the conjugate of \(i\) against
\(j\)) and positive semi-definite (no combination of the channels has
negative power) at every line. A specification that is not cannot be
produced by any drive, and a controller asked for one fails at those
lines however long it runs. Every entry can look right on its own:
three channels with every coherence 1 and phases of 0, 0 and 180
degrees ask for 1 in step with 2, 2 in step with 3, and 1 opposite 3,
which no motion does.

The check is made on the coherence matrix, the cross-spectral matrix
scaled by the root of its diagonal:

\[
C_{ij} = \frac{S_{ij}}{\sqrt{S_{ii}\,S_{jj}}}
\]

It has ones on the diagonal and no units, and it is positive
semi-definite exactly where \(S\) is, so channels of different
quantities can be judged together and the levels do not change the
verdict. Its eigenvalues sum to the number of channels. A line is
realizable where the smallest is no lower than minus a tolerance times
the largest and the matrix is Hermitian to the same tolerance; the
default, one part in a million, passes a matrix written to six
significant figures. For the three channels above the eigenvalues are
\(-1\), 2 and 2.

A matrix that fails is repaired to the nearest realizable one, in the
Frobenius norm of the coherence matrix. By default the autospectra stay
exactly as written and only the cross terms move (the nearest
unit-diagonal matrix, found by alternating projections); lines that
pass are left as they are. For the three channels the nearest matrix
has every coherence 0.25, the phases kept. The reading reports the
largest change in coherence and in phase the repair made, and draws the
pair it moved most.

![The eigenvalues of a three-channel specification's coherence matrix
with the lines that are not realizable hatched, and the coherence and
phase of one pair as written and repaired](images/mimo-realizability.png)

The figure is three control channels at 0.01 g²/Hz with a coherence of
0.36 between every pair, 1 and 2 in step, 2 and 3 in step, and the
phase of 1 against 3 rising from 0 at 20 Hz to 180 degrees at
2000 Hz. Two pairs in step leave the third little room: its determinant
\(1 - 3a^2 + 2a^3\cos\varphi\), with \(a = 0.6\), goes negative past
79 degrees, and the smallest eigenvalue (solid) crosses zero at
155 Hz. Above that every line is hatched. At 1000 Hz the repair takes
the 1-3 coherence from 0.36 to 0.26 and its phase from 153 to 151
degrees; at 2000 Hz, where the phases are 0, 0 and 180, every coherence
goes to 0.25.

## From a script

```python
import visualdynamics

frf = project['Plant H1 FRFs']
reading = frf.conditioning()              # limit=100 by default
reading.bands()                           # [(92.5, 106.25)], in Hz
point = reading.at(100.0)
point.condition, point.rank               # 375.0, 1 at the first mode
print(reading.describe())
visualdynamics.plot.plot_conditioning(reading, path='conditioning.png')

spec = project['Specification']
check = spec.realizability()
check.realizable                          # False
check.bands()                             # the runs of failing lines
check.changes()                           # (largest coherence, phase change)
visualdynamics.plot.plot_realizability(check, path='realizability.png')
project.make_realizable('Specification')  # adds 'Specification Realizable'
```

`Frf.conditioning(records=None, limit=100)` reads the FRF's records as
the matrix of responses by references; `Psd.conditioning` reads a CSD
of the drives. `Psd.realizability(records=None, tolerance=1e-6,
keep_diagonal=True)` judges a CSD or a specification, and
`Psd.made_realizable` returns the repaired copy; `make_realizable`
adds it to the project beside its source, refreshable from it. A
specification's limits stay with its autospectra. Lines holding a value
that is not a number are not judged and are noted, and a reading that
judged nothing says so rather than passing. The functions underneath
are in `visualdynamics.core.matrix_checks`: `read_conditioning` and
`condition_matrix`, `read_realizability`, `check_spectral_matrix` and
`repair_coherence`.

A specification sheet (`SpecificationDraft`) runs the same check on its
coherence and phase entries and refuses a sheet whose entries cannot
hold together, naming the channels, the lines (every line of the
sheet, since the entries do not change with frequency) and the nearest
entries that would hold:

```python
draft = SpecificationDraft.at_dofs(['1Z+', '2Z+', '3Z+'], 'acceleration')
draft.with_all_pairs(1.0, 90.0)
# ValueError: the cross terms of 1Z+, 2Z+ and 3Z+ cannot hold together ...
```

A pair left unstated is not judged; each group of channels whose pairs
are all stated is judged on its own.

## In the app

Select an FRF computed from two references or more, or a CSD, and
press **Conditioning** on the plot's bar. The plot gives way to the
singular values and the condition number, and a dashed cursor starts on
the first flagged line; drag it, and the pane reads the condition
number against the limit, the effective rank and the singular values at
its line, with the verdict and the notes beneath. Select a CSD or a
specification and press **Realizability** for the eigenvalues, read the
same way. When the matrix is not realizable, **Make Realizable** in the
pane adds the repaired copy beside it and shows it. The specification
sheet refuses an entry that cannot hold together in the line under its
grids, and the specification is left as it was.
