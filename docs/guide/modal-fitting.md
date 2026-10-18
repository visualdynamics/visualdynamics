# How the mode fitting works

Visual Dynamics fits modes one at a time on the CMIF, and the method
differs from the batch polynomial methods (least-squares complex
exponential, polyreference and similar). There is no model order and
no stabilization diagram. You choose each pole while looking at the
CMIF, and the solver fits a mode at that pole. This page describes
each part of the fit in the order the loop runs.

The way of working, one mode at a time with the analyst choosing each
pole and judging a synthesis against the measurement, was inspired by
SMAC (Synthesize Modes And Correlate), described in the last section.
The computations here are different.

## The CMIF and the residual

The fitting screen shows the CMIF: at every frequency line, the
singular values of the FRF matrix (responses × references). The
largest singular value peaks at every mode, however the individual
FRFs interfere, which is why it is the curve the fit is done against.
The measured CMIF is drawn solid, and the synthesis of the fitted
modes is drawn dashed over it, so the plot shows how much of the
measurement the modes fitted so far account for.

The fit works on the residual: the FRF matrix with the contribution
of every confirmed mode subtracted. Each new mode is fit to what is
left, so confirming a mode removes its peak from the residual, and the
peaks that remain are the modes still to fit. The residual's own CMIF
is drawn with the **Residual** toggle on the plot bar, with the other
show/hide overlays.

The fitting screen opens with *Fit Modal Model* on one selected FRF.
Records picked in the tree limit the fit to those records. The table
under the plot lists the confirmed modes and, in its last row, the
pending mode at the cursor; the MAC of the fitted shapes is beside it.
A confirmed mode's damping and description can be typed into the
table, and selecting a row and pressing Delete removes the mode.

## Find Mode: choosing the next peak

**Find Mode** moves the cursor to the next pole to fit. Two rules
decide which peak it picks.

The first rule is prominence rather than height. The CMIF is usually
accelerance, so raw peak height grows with frequency squared, and the
tallest residual line is nearly always on the shoulder of the highest
dense cluster. Find Mode instead ranks peaks by how far each stands
above its own local floor (the median of the log CMIF over the
surrounding lines), which is how a peak looks on the log plot. Ranked
this way, a low fundamental is found before the lines of a loud
cluster above it.

The second rule keeps the loop from confirming the same peak over and
over. Subtracting a fitted mode is exact only when the mode is alone;
in a cluster it leaves enough of a ridge that the same line can stay
the tallest. So the neighborhood of each confirmed mode (its frequency
plus or minus ζf, the half-power half-width) is excluded from
suggestion. Frequency alone cannot tell that ridge from a real
neighbor, though, and the repeated pair of a symmetric structure puts
its second mode at the confirmed frequency. The shape can tell them
apart: a ridge lies in the span of the confirmed shapes and a real
neighbor does not. A residual peak inside a confirmed neighborhood is
therefore offered when less than a tenth of its direction lies in the
span of the confirmed shapes there and its residual still holds at
least a tenth of the measured CMIF at that line. You can always drag
the cursor anywhere by hand; the exclusion applies only to Find Mode
and the scripted loop.

After choosing a line, Find Mode passes it to the two-parameter search
described below, which returns a frequency between the spectral lines
and the damping that goes with it. The peak line is where the search
starts, not the frequency it reports.

The search band is the zoom: Find Mode searches only the frequencies
on the plot, so to limit the fit to the excited band, zoom to it. In a
script, `fit_modes(bounds=...)` sets the same limits.

## The cursor

The cursor is two controls in one. Moving it left and right sets the
frequency. Dragging it vertically holds the damping: the plot height
maps damping from 0.01 % at the top to 10 % at the bottom, log-linear,
and the two limits can be edited in the fields at the plot's corners.
While you drag, a dashed parabola shows the SDOF magnitude for the
cursor's frequency and damping, with its crown on the residual CMIF at
the cursor and the damping value printed at the crown. Match its
shoulders to the peak and the fit has the pole you intend. The held
damping stays until **Find Mode** suggests a new pole or a mode is
confirmed.

The parabola is a display, not an estimate. It is a closed-form SDOF
magnitude for the frequency and damping the cursor names, computed
with no SVD and no least squares so that it can be redrawn on every
step of a drag. The fit itself comes from the search and the solver
below.

With the damping not held, the damping estimate is automatic: a
one-dimensional search along the damping axis of that same cost
surface, seeded from the local half-power width. The estimate depends
only on the cursor position, so the same line always gives the same
damping, whichever direction the drag approached it from.

The fit runs as the cursor moves, and the MAC and anything the drag
skipped are brought up to date when the cursor stops. There is no
separate fit button.

The peak the cursor sits on is also read by its half-power width: the
0.707 level, its two crossings and the Q they give are drawn in the
cursor's red, and the reading ends the status line
(`ModalFitSession.half_power`). It is the curve's own width, beside the
damping the search found; where the two disagree the peak is not one
clean mode ([Reading Q from a peak's half-power width](half-power.md)).

## The equation being solved

The model behind everything on the screen is the residue form of a
real normal mode expansion. The synthesis draws, and the published
shape set holds,

\[
H_{jk}(\omega) \;=\; \sum_r
\frac{(i\omega)^{p}\,\phi_{jr}\,\phi_{kr}}
     {m_r\left(\omega_r^{2}-\omega^{2}
      + 2 i \zeta_r \omega_r \omega\right)}
\]

where \(H_{jk}\) is the FRF between response DOF \(j\) and
reference DOF \(k\), \(\omega_r\), \(\zeta_r\) and
\(\phi_{\cdot r}\) are mode \(r\)'s natural frequency, viscous
damping ratio and mass-normalized shape, \(m_r\) is the modal mass
(1 for a mass-normalized set), and \(p\) is set by the measured
quantity: 0 for displacement over force, 1 for velocity, 2 for
accelerance. The \((i\omega)^p\) factor is inside the sum because a
rigid-body mode's denominator is exactly \(-\omega^2\): at
\(\omega = 0\) its accelerance cancels to the finite residue, while
multiplying afterward would give 0/0.

The CMIF on the screen is the singular value spectrum of that matrix
at each line, \(\mathrm{CMIF}_n(\omega) =
\sigma_n\!\left(H(\omega)\right)\), and the residual the fit works on
is the measurement minus the term of every confirmed mode:

\[
R(\omega) \;=\; H^{\text{measured}}(\omega)
\;-\; \sum_{r\,\in\,\text{confirmed}}
\frac{(i\omega)^{p}\,\phi_{jr}\,\phi_{kr}}
     {\omega_r^{2}-\omega^{2}+2 i \zeta_r \omega_r \omega}.
\]

Fitting one mode is then a small, fully defined problem. With the pole
\((\omega_r, \zeta_r)\) chosen, by Find Mode or by the cursor, define
the SDOF kernel

\[
q(\omega) \;=\;
\frac{(i\omega)^{p}}
     {\omega_r^{2}-\omega^{2}+2 i \zeta_r \omega_r \omega},
\]

take the shape direction \(u\) as the first left singular vector of
\(R\) at the peak line, rotated to the nearest real vector
(\(u \leftarrow \operatorname{Re}\!\left[u\,
e^{-\tfrac{i}{2}\arg \sum_j u_j^{2}}\right]\), then normalized), and
project each reference's residual column onto it,
\(g_k(\omega) = u^{\mathsf T} R(\omega)\, e_k\). The residues are one
real least squares per reference over a band \(B\) of the lines within
\(3\zeta f\) of the peak (with \(\zeta\) read from the half-power width
of the residual CMIF there, and at least two lines either side):

\[
A_k \;=\; \arg\min_{A_k \in \mathbb{R}}
\sum_{\omega \in B}\bigl|\,g_k(\omega) - A_k\, q(\omega)
\bigr|^{2}
\;=\;
\frac{\operatorname{Re}\sum_{\omega \in B}
\overline{q(\omega)}\, g_k(\omega)}
{\sum_{\omega \in B} \left|q(\omega)\right|^{2}}.
\]

The \(A_k\) are real because the model is real normal modes. Finally
the residues become shape coefficients through the mass-normalized
identity \(A_{jk} = \phi_j\,\phi_k\). A drive point, a reference that
was also measured as a response, gives \(\phi_k = \sqrt{A_{kk}}\) and
with it the absolute scale. With several drive points, each gives a
scale estimate \(A_k/u_k\), and the estimates are averaged with weights
\(u_k^2\), the share of the mode at that drive. The weighting matters
because a drive point near a node of the mode carries mostly noise:
on the demonstration plate, one shaker sits near a node of the 647 Hz
mode, and averaging its estimate in with equal weight would cut the
synthesis to about half the measured peak. An estimate that is still negative after
the shape's sign convention is settled is noise as well (a true
drive-point residue is \(\phi_k^2\) and cannot be negative) and is
discarded. With no usable estimate the shape is correct, the scale is
a convention, and the shape set is marked unscaled.

The single-mode fit is unweighted. When a coherence covers the data,
the joint refinement (**Refine All**) and its scoring weight each line
by the inverse FRF variance, \(w \propto \gamma^{2}/(1-\gamma^{2})\).

That is the whole solver: the pole is chosen by you or by Find Mode,
the direction comes from an SVD, and each amplitude is a
one-parameter least squares. Nothing else is fitted.

## Choosing the pole: the two-parameter search

Find Mode's frequency and damping, and the cursor's automatic damping,
come from the same calculation, searched over two axes or one. Given a
line, the cost of a candidate pole is defined as follows:

- take the shape direction \(u\) from the residual's SVD at that
  line, exactly as above, and hold it fixed for the whole search (it
  is not estimated again for each candidate);
- project the residual over a band around the line onto it, giving
  \(g_k(\omega)\), one complex number per reference per line;
- for a candidate \((f, \zeta)\), build \(q(\omega)\), solve the
  residues \(A_k\) with the closed form above, and measure what is
  left over:

\[
\operatorname{cost}(f, \zeta) \;=\;
\frac{\sum_{\omega \in B}\sum_k
\bigl|\,g_k(\omega) - A_k\,q(\omega)\bigr|^{2}}
{\sum_{\omega \in B}\sum_k \bigl|\,g_k(\omega)\bigr|^{2}}
\]

The cost is normalized by the size of the projected residual, so 0 is
a pole that explains the band completely and 1 is a pole that explains
none of it.

A grid search is used instead of an optimizer because only the
frequency and the damping enter nonlinearly. Given those two, the
residues are a linear least squares with the closed form above, so one
candidate costs a single SDOF term over a couple of dozen lines, a dot
product and a norm. That takes microseconds, where a full single-mode
fit takes milliseconds, so the surface can be evaluated directly.

Find Mode searches both axes. Each pass is a 5×5 grid, over half a
frequency line either side of the current best and a damping span that
starts at the half-power estimate, and ends with a parabola through
the winner and its neighbors in each direction. The next pass is
centered on that result with a span a fifth as wide, and there are
four passes, about a hundred cost evaluations in all. The cost is not
exactly a parabola at a pass's spacing, so each vertex lands slightly
off the minimum; four passes reduce that error to about 0.001 % of the
damping on exact data. When a pass's best frequency lies on the edge
of its grid, the grid is re-centered and the pass repeated, but the
frequency never moves more than 1.5 lines from the line Find Mode
picked, and never into a confirmed mode's half-power neighborhood. The
damping span does not move this way: in a cluster a wider mode always
explains more of the band, and a free damping axis would fit blends
of several modes. Dragging the cursor searches the damping axis only,
since you are setting the frequency by hand.

The search replaces reading the damping from two half-power crossings.
Half-power uses two points on the curve and assumes the peak is one
isolated mode; where two modes are close, the width it measures is
the width of the pair, and the damping comes out too high with
nothing to show it. A residual minimum makes no such assumption. The
half-power width is still used to seed the damping span, where it is
a good starting value even when it is a poor answer.

Because the search works on a normalized residual, a cost near 1 means
the pole explains almost nothing of what is left at that line. That is
what a suggestion in the noise floor looks like, and it indicates that
the fit has taken everything the measurement supports.

## Fitting one mode

One mode is fit to the residual in three steps:

1. Shape direction: the first left singular vector of the residual
   matrix at the peak line, rotated to the nearest real vector. The
   direction is estimated from every record at once, so noise that is
   not coherent across channels has little effect on it.
2. Residues: a least-squares fit of the SDOF term over the band
   around the peak, one residue per reference, on the residual
   projected onto that direction.
3. Scale: a drive point (a response measured at an excitation DOF)
   sets the mass-normalized scale through \( A_{jk} = \phi_j \phi_k \).
   Without one the shapes are consistent with each other but the scale
   is a convention, and the shape set is marked unscaled wherever it
   appears.

When you hold the damping by hand, there is one more step: the
residues are scaled (by a factor between 0.25 and 4) so that the
mode's own CMIF at the fitted line meets the residual CMIF there,
which is the height the parabola's crown showed. Without this, a held
damping wider than the data's (matched by eye to a lobe that includes
the shoulders) puts the synthesis below the peak, up to 25 % low at
twice the true width. Automatic fits are not scaled this way. In a
dense cluster the residual peak holds several modes, and scaling each
rank-one fit to the whole peak makes the amplitudes far too large;
the least squares already accounts for how much of the
residual its one direction explains.

**Confirm Mode** adds the mode to the fit and moves the cursor to the
next suggestion. Each confirm publishes the fit as a shape set in the
project, and that shape set holds the complete fit state. To change
the fit later, select the shape set together with its FRF and choose
**Edit Fit** on the plot bar; the session reopens with every mode as
published. A published set does not keep the order the modes were
confirmed in, so the reopened session treats them in frequency order.

## Refine All: refitting close modes together

Sequential fitting never goes back. The first mode of a close pair is
fit on data that still contains the second mode's tail, so its
residues take up part of the neighbor. The sum of the pair still
matches the measurement, so the error is in how the response is split
between the two modes. It does not show on the CMIF, and it appears as
MAC leakage between the pair or as modal masses that depend on the
order the modes were fit in.

**Refine All** appears once two modes are confirmed. It refits every
mode's residues with the poles held exactly where they were placed:

- Modes are solved together only where their bands overlap. An
  accelerance SDOF term tends to a constant far above its pole (the
  mass line), so a single global solve would let a quiet low mode
  change to correct small errors in a loud cluster hundreds of lines
  away. Modes are grouped into clusters by band overlap, and each
  cluster is solved on the measurement less the other modes at their
  sequential residues.
- Each cluster is solved five ways and the CMIF picks the result. The
  candidates are the plain joint least squares, three ridge strengths
  pulled toward the sequential answer, and the sequential answer
  itself. Each candidate is carried through to the shapes it would
  publish and scored by how far its synthesized CMIF is from the
  measured CMIF over the cluster's own band. The residues are
  optimized on the FRFs, and the result is selected on the CMIF.
  Because the sequential answer is always a candidate, refining cannot
  make the CMIF match worse. When the sequential answer wins
  everywhere, the status bar says that the refine changed nothing.
- A cluster of one mode is not refit. A lone mode has no neighbor to
  share residue with, and refitting it can only follow noise and the
  smooth tails of louder modes, so its sequential fit stays.
- A coherence weights the solve. If the project holds a coherence
  covering the fit's response DOFs, the fitting screen uses it
  automatically and the status line names it. The variance of an FRF
  estimate goes as (1 − γ²)/γ², so its inverse weights the joint solve
  and the CMIF scoring, and a channel with low coherence has less
  influence on every mode's residues. In a script,
  `project.fit_coherence('FRF', records)` returns the coherence the
  screen would use, and `fit_modes(coherence=...)` applies it; a fit
  of records picked in the tree is `fit_modes(records=[...])`. The
  session's journal line records both, so a replayed session fits the
  same records with the same weights.

## Limitations

These are the limits of this approach compared with the polynomial
methods:

- Real normal modes only. Shape vectors are rotated to the nearest
  real vector. Strongly complex modes (heavily non-proportional
  damping) lose their phase; their frequencies and dampings are still
  usable.
- No global pole search. Poles are where you or Find Mode put them,
  refined locally. A stabilization diagram can find repeated roots
  under a peak that looks like one mode; here you find them by
  watching the residual after each confirm. If a peak remains after a
  confirm, there is another mode under it.
- One mode per fit step. Close pairs are handled by fitting both modes
  and pressing **Refine All**, not by a simultaneous high-order solve.

In exchange, every mode on the screen was chosen by a person and fit
at the pole that was chosen, there is no model order to sweep and no
computational poles to remove, and the complete fit state is kept in
the published shape set, which can be reopened and edited at any time.

## Where the approach comes from

SMAC, the Synthesize Modes And Correlate method of Klenke and Mayes,
also fits modes one at a time with the analyst in the loop: a trial
root's synthesized single-mode response is correlated with the
measured data, and the roots, dampings and shapes are found mode by
mode. The fitting screen grew from experience with that way of
working. The computations here are not SMAC's: the next mode is the
largest peak of the residual CMIF, its shape is the first singular
vector at that peak, and its pole is the one that leaves the least
residual over a band, as the sections above describe.

1. Klenke, S. E., & Mayes, R. L. (1999). "The SMAC Modal Parameter
   Extraction Package." *International Modal Analysis Conference*,
   Orlando, FL. [OSTI 1063](https://www.osti.gov/biblio/1063).
