# Judging a cross-spectral matrix against its specification

A multi-input random test is controlled to a matrix. Its diagonal is
each control channel's autospectrum, the level that channel is held
at; its cross terms say how each pair of channels moves together. A run
can hold every level inside its band and still not be the test that was
asked for: two channels that were to move together may have moved
independently, or in opposition. This page describes how Visual
Dynamics judges the whole matrix, in the app and from a script.

## What is compared

The specification is a `Specification` with cross terms, such as one
written on the specification sheet (`SpecificationDraft`), where each
pair is entered as a coherence and a phase, or one imported from a
controller. The measurement is the run's cross-spectral matrix,
`compute_cpsds` on the time history. The control channels are the
specification's autospectra, in its order, paired with the
measurement's records by DOF and quantity, as every comparison pairs
them. A set of autospectra with no cross terms is compared channel by
channel ([How a comparison is judged](compliance.md)), not as a matrix.

## The rule

1. The levels. Each autospectrum is compared with its specification
   exactly as a single channel is: power against power cell by cell,
   the share of the band outside abort and the RMS error in dB. A level
   is out when more than 10 percent of its band is outside abort or its
   RMS is more than 3 dB off, and the diagonal passes or fails by the
   rule a set of channels does. A measured autospectrum with a line
   that is not a number inside the band is refused, naming the count,
   since the line would otherwise be counted as no power.
2. The coherence. For each pair the specification writes a cross term
   for, the ordinary coherence, |S<sub>pq</sub>|² / (S<sub>pp</sub>
   S<sub>qq</sub>), is read from the specification and from the
   measurement at every measured line. A line is out when the two
   differ by more than the coherence tolerance, an absolute difference.
3. The phase. The phase of the same cross term, arg S<sub>pq</sub>, is
   read the same way, and a line is out when the two differ by more
   than the phase tolerance, taken the short way round the circle (a
   phase 350 degrees off is 10 degrees off). The phase is judged only
   where the specified coherence is at least the floor: two channels
   asked to be nearly independent have no phase to hold.
4. The share. A cross term is out when either reading is out on more
   than 10 percent of the lines it judged, the share a level is allowed
   outside abort.
5. The verdict. The matrix passes when the diagonal passes and no cross
   term is out.

The specification is read the way it is drawn. Between breakpoints
each term's magnitude is read log-log and its phase linearly, the way
the specification sheet reads a cross term, so a pair entered as one
coherence and one phase keeps exactly that coherence and phase at
every line between the breakpoints. On a specification of lines or
bands, a line takes the value of the bin it falls in. A coherence of
zero, two channels stated to be independent, is a value and is judged.
A pair the specification writes no cross term for is not judged and is
said to be not specified: an unstated pair is not a statement that the
channels are independent.

The defaults, a coherence tolerance of 0.1 either way, a phase
tolerance of 10 degrees either way and a floor of 0.5, are examples of
what test documents ask for. The test plan states its own.

### Why a share of the lines and not the worst line

A coherence and a phase read from averaged spectra scatter about their
true values. The normalized random error of a coherence estimate is
√2 (1 − γ²) / (|γ| √n) at `n` averages, and the standard deviation of
a phase estimate is √(1 − γ²) / (|γ| √(2n)) radians. At a coherence of
0.5 and a hundred averages the coherence scatters by about 0.05, so a
run that is exactly its specification puts a few percent of its lines
past a tolerance of 0.1 by chance, and below the floor the phase
scatters by tens of degrees. A cross term judged by its worst line
would fail a correct run. Judged by the share of its lines, it fails
when the departure is real. Fewer averages widen the scatter, and a
run judged against a tight tolerance needs enough of them.

Each measured line also carries two numbers for the whole matrix: the
RMS over the control channels of their level errors in dB, and how
many cross terms are out at that line. Watched at each step of a
step-up, they show where the matrix is diverging before the verdict
changes.

## In the app

Select the specification and the run's CPSD together, and choose
**Matrix** on the plot's bar. The plot becomes a grid with one row and
one column per control channel: each channel's autospectrum over its
specification and abort limits on the diagonal, each pair's coherence
above it and the same pair's phase below it, the measurement solid, the
specification dashed and the tolerances dotted. Every cell's title
names its reading and its verdict, with OUT in capitals for a cell
that is out, and the frame says it again: heavy and solid round a cell
that is out, thin round one that is in, dotted round one that was not
judged or not specified. The lines judged out are marked with a cross
on the measured curve. Under the grid, two strips give the per-line
summary.

The **Matrix** panel beside the plot holds the settings:

- **Coherence [±]**: the coherence tolerance.
- **Phase [±°]**: the phase tolerance, in degrees.
- **Phase from γ²**: the specified coherence a phase is judged from.

Under them the panel gives the verdict and every cell in words, the
levels first and then each pair's coherence and phase. Every edit
redraws the grid. The measured levels are compared at the comparison's
scale, the same one the channel comparison and the report use; the
coherence and phase do not change with it. Unchecking **Matrix** goes
back to the channels.

## From a script

```python
from visualdynamics.core.matrix_comparison import compare_matrix
from visualdynamics.plot import plot_matrix_comparison

cpsd = history.compute_cpsds()
result = compare_matrix(specification, cpsd)
print(result.describe())
for row in result.table():
    print(row['label'], row['reading'], row['words'])
plot_matrix_comparison(specification, cpsd, path='matrix.png', show=False)
```

For a three-channel run whose third channel's phase to the first
walks 40 degrees off its specification above 800 Hz, with every level
and every coherence held, this prints:

```
fails: levels pass on 3 channels; 2 of 3 cross terms out (coherence 0, phase 2)
101X+ level in: 0.0 % of lines outside abort, +0.0 dB rms
102X+ level in: 0.0 % of lines outside abort, +0.0 dB rms
103X+ level in: 0.0 % of lines outside abort, +0.0 dB rms
101X+/102X+ coherence coherence in: 0 of 1981 lines past ±0.10
101X+/102X+ phase phase in: 0 of 1981 lines past ±10°
101X+/103X+ coherence coherence in: 5 of 1981 lines past ±0.10
101X+/103X+ phase phase out: 45.0 % of 1981 lines past ±10°
102X+/103X+ coherence coherence in: 23 of 1981 lines past ±0.10
102X+/103X+ phase phase out: 45.1 % of 1981 lines past ±10°
```

The phase of the third channel to the first has walked out, and since
the second is coherent with the first, its phase to the third has
walked out with it. The coherences each have a few lines past 0.1, the
scatter of the estimate, and are inside.

![A three-channel matrix comparison: the three levels on the diagonal
inside their abort limits, the coherences above the diagonal inside
their tolerances, and below it the phases of the third channel to the
first and to the second walking out of their tolerance above about
1100 Hz, those two cells framed heavily and titled OUT; under the grid
the RMS level error and the count of cross terms out at each line, and
the table beside](images/matrix-comparison.png)

`compare_matrix` takes `coherence_tolerance=`, `phase_tolerance=`,
`coherence_floor=` and `scale_db=`, and `spec_records=` and
`measured_records=` to restrict either side. The result's `levels`
hold each channel's comparison, its `pairs` each cross term's
specified and measured coherence and phase at every line with the
lines out, and `cell(row, column)` what one cell of the grid shows.
`cross_term_lines` is the per-line judgement of one cross term on plain
arrays.

A cross term is never read as a level. `compare` refuses one, and
`compare_all` compares the autospectra only: a cross term's real part
is not the power of either channel, and it moves with the phase.
