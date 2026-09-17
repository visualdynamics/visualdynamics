# How the mode fitting works

Visual Dynamics does not fit modes the way the classical batch methods
do, and anyone arriving from LSCE, polyreference, or PolyMAX-style
tools deserves to know exactly what is different and why. The short
version: **there is no model order and no stabilization diagram. Poles
are chosen one at a time by a person looking at the CMIF, and the
solver's job is to make each choice land exactly where it was aimed.**
Everything below is the long version, in the order the loop runs.

## The curve everything happens on

The fitting screen shows the **CMIF** — at every frequency line, the
singular values of the FRF matrix (responses × references). The
largest singular value peaks at every mode, however the individual
FRFs interfere, which is what makes it the right single curve to fit
against. The measured CMIF stays solid throughout; the fit's synthesis
climbs onto it dashed, so the state of the fit is always read as *how
much of the measurement the model explains so far*.

The fit itself works on the **residual**: the FRF matrix with every
confirmed mode's contribution subtracted. Each new mode is fit to what
is *left*, not to the raw measurement — so confirming a mode visibly
collapses its peak, and what remains is the to-do list. (The residual's
own CMIF can be drawn with the **Residual** toggle on the plot bar,
beside the other show/hide overlays; by default it is legible from the
synthesis climbing.)

## Find Mode: where the loop looks next

Find Mode suggests the next pole, and two rules make the suggestion
match what an engineer would pick rather than what a ruler would:

- **Prominence, not height.** The CMIF is usually accelerance, so raw
  peak height grows as frequency squared, and the tallest residual
  line is nearly always a shoulder of the highest dense cluster.
  Ranked by height, ten confirms on a real survey spent every one
  inside a single cluster and never visited the 64 Hz fundamental.
  Suggestions are instead ranked by how far a peak stands above its
  own local floor, in log height — which is what a peak *looks like*
  on the plot. On that same survey, ten confirms then landed on the
  very set of modes the survey's engineer had picked by hand.
- **A confirmed peak is spoken for — unless the shape standing there
  is somebody else's.** Subtracting a fitted mode is exact only when
  the mode is alone; in a cluster it leaves enough ridge that the same
  line can stay the tallest, and a naive loop will confirm the same
  frequency forever. So a confirmed mode's half-power neighborhood is
  excluded from suggestion — but frequency alone cannot tell that
  ridge from a genuine neighbor, and a symmetric structure's repeated
  pair puts its second mode at exactly the confirmed frequency. Shape
  can tell: ridge lies in the span of the confirmed shapes, a real
  neighbor does not, so a residual peak inside the neighborhood is
  offered anyway when the confirmed modes explain almost none of the
  direction standing there. The cursor can still be dragged anywhere
  by hand — the exclusion binds the automatic loop, not the person.

Having chosen a line, Find Mode does not stop there: it hands that
line to the two-parameter search below, which returns a frequency
*between* the spectral lines and the damping that goes with it. The
peak line is where the search starts, not what it reports.

The search band is the **zoom**: what is on the plot is what gets
searched, so constraining a fit to the excited band is done by looking
at it. Scripted, `fit_modes(bounds=...)` says the same thing.

## The cursor: pole selection is a human act

The cursor is two controls in one. Left–right is frequency. Dragging
**vertically holds the damping**: the plot height maps the range
0.01 % (top) to 10 % (bottom), log-linear, editable at the plot's
corner fields. While you drag, a dashed **parabola** — the SDOF
magnitude your (frequency, damping) claims, crown pinned to the
residual CMIF at the cursor — shows the mode you are asking for
before anything is fit. Match its shoulders to the peak and you have
told the solver everything it needs.

**The parabola is a readout, not an estimator.** Nothing is measured
off it: it is a closed-form SDOF magnitude for whatever frequency and
damping the cursor currently names, drawn with no SVD and no least
squares so that it can be redrawn on every tick of a drag. It shows
what you are about to ask for. What answers is the search below.

With nothing held, the damping estimate is automatic: a
one-dimensional walk along that search's damping axis, seeded from the
local half-power width. The estimate is a pure function of the cursor
position — the same line always answers the same, whichever way the
drag approached it.

## The equation being solved

The model behind everything on the screen is the **residue form of a
real normal mode expansion**. What the synthesis draws — and what the
published shape set means — is

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
(1 for a mass-normalized set), and \(p\) picks the measured
quantity — 0 for displacement over force, 1 for velocity, 2 for
accelerance. The \((i\omega)^p\) sits *inside* the sum because a
rigid-body mode's denominator is exactly \(-\omega^2\): at
\(\omega = 0\) its accelerance cancels to the finite residue,
where an after-the-fact multiply is 0/0.

The CMIF the screen shows is the singular value spectrum of that
matrix at each line, \(\mathrm{CMIF}_n(\omega) =
\sigma_n\!\left(H(\omega)\right)\), and the **residual** the
fit works on is the measurement minus every confirmed mode's term:

\[
R(\omega) \;=\; H^{\text{measured}}(\omega)
\;-\; \sum_{r\,\in\,\text{confirmed}}
\frac{(i\omega)^{p}\,\phi_{jr}\,\phi_{kr}}
     {\omega_r^{2}-\omega^{2}+2 i \zeta_r \omega_r \omega}.
\]

Fitting one mode is then a small, exactly-stated problem. With the
pole \((\omega_r, \zeta_r)\) chosen — by Find Mode or by the
cursor — define the SDOF kernel

\[
q(\omega) \;=\;
\frac{(i\omega)^{p}}
     {\omega_r^{2}-\omega^{2}+2 i \zeta_r \omega_r \omega},
\]

take the shape *direction* \(u\) as the first left singular vector
of \(R\) at the peak line, rotated to the nearest real vector
(\(u \leftarrow \operatorname{Re}\!\left[u\,
e^{-\tfrac{i}{2}\arg \sum_j u_j^{2}}\right]\), then normalized),
and project each reference's residual column onto it,
\(g_k(\omega) = u^{\mathsf T} R(\omega)\, e_k\). The residues
are one real least squares per reference over a band \(B\) three
half-power widths either side of the peak:

\[
A_k \;=\; \arg\min_{A_k \in \mathbb{R}}
\sum_{\omega \in B}\bigl|\,g_k(\omega) - A_k\, q(\omega)
\bigr|^{2}
\;=\;
\frac{\operatorname{Re}\sum_{\omega \in B}
\overline{q(\omega)}\, g_k(\omega)}
{\sum_{\omega \in B} \left|q(\omega)\right|^{2}}.
\]

Real \(A_k\), because the model is real normal modes. Finally the
residues become shape coefficients through the mass-normalized
identity \(A_{jk} = \phi_j\,\phi_k\): a **drive point** — a
reference that was also measured as a response — gives
\(\phi_k = \sqrt{A_{kk}}\) and with it the absolute scale. With
several drive points each offers a scale estimate
\(A_k/u_k\), and they vote **weighted by \(u_k^2\)** — how much of
the mode lives at that drive — because a drive point near a node
has no standing on scale: the plate's 647 Hz mode sits on one
shaker's node, and averaging that drive's noise in halved the
synthesis. A vote still negative after the shape's sign convention
is settled is that same noise self-declared — a true drive-point
residue is \(\phi_k^2\), never negative — and is discarded. Without
any usable vote the shape is right and the scale is a convention,
and the set says so. The single-mode fit is unweighted; when a
coherence \(\gamma^2\) covers the data, it is the joint refinement
(*Refine All*) and its judging that weight each line by the inverse
FRF variance, \(w \propto \gamma^{2}/(1-\gamma^{2})\).

That is the whole solver: pole by a person, direction by an SVD,
amplitude by a one-parameter least squares. Nothing else is free.

## Choosing the pole: the two-parameter search

Find Mode's frequency and damping, and the cursor's automatic damping,
are the same calculation read along two axes or one. Given a line to
look at, the cost of a candidate pole is defined like this:

- take the shape direction \(u\) from the residual's SVD at that
  line, exactly as above — and **hold it fixed for the whole search**.
  It is not re-estimated per candidate;
- project the residual over a band around the line onto it, giving
  \(g_k(\omega)\) — one complex number per reference per line;
- for a candidate \((f, \zeta)\), build \(q(\omega)\), solve the
  residues \(A_k\) by the closed form above, and measure what is
  left over:

\[
\operatorname{cost}(f, \zeta) \;=\;
\frac{\sum_{\omega \in B}\sum_k
\bigl|\,g_k(\omega) - A_k\,q(\omega)\bigr|^{2}}
{\sum_{\omega \in B}\sum_k \bigl|\,g_k(\omega)\bigr|^{2}}
\]

normalized by the projected residual's own size, so **0 is a pole that
explains the band perfectly and 1 is one that explains nothing**.

**Why a grid rather than an optimizer.** Only the frequency and the
damping enter nonlinearly. Given them, the residues are a linear least
squares with the closed form written out above — so one candidate
costs a single SDOF term over a couple of dozen lines, a dot product
and a norm. Microseconds, where a full single-mode fit is
milliseconds. The
surface can therefore be walked directly rather than handed to an
optimizer that would have to rediscover the same structure on its way
to the same answer.

Find Mode walks both axes: a 5×5 grid over half a frequency line
either side and a damping span seeded from the half-power width, then
the same grid again around the winner at a fifth the span, then a
parabola through the winner and its neighbors in each direction
independently. About fifty cheap evaluations reach what a 25×25 grid
would, and the parabola lands *between* grid points, where the surface
really is quadratic and interpolating it is unbiased. Dragging the
cursor walks one axis of the same surface — the frequency is being
chosen by hand, so there is nothing to search along it.

**Why this replaced reading two half-power crossings.** Half-power is
two points on a curve, and it assumes the peak is one isolated mode.
Where two modes sit close, the width it measures is *the pair's* — the
damping comes back too high and nothing says so. A residual minimum
assumes none of that. The half-power width survives as the seed for
the search span, which is a fine starting guess even where it is a
poor answer.

One consequence worth knowing when reading a fit: because the search
is over a normalized residual, a cost near 1 means the pole on offer
explains almost nothing of what is left there. That is what a
suggestion into the noise floor looks like, and it is the signal that
the fit has taken everything the measurement supports.

## Fitting one mode: what the solver actually does

There is no button for this: the fit runs as the cursor moves and
settles when it stops. One mode is fit to the residual, in three
steps:

1. **Shape direction** — the first left singular vector of the
   residual matrix at the peak line, rotated to the nearest real
   vector. Working in the dominant singular direction is what makes
   the fit noise-tolerant: the direction is estimated from every
   record at once, and noise that does not cohere across channels
   cannot steer it.
2. **Residues** — a least-squares fit of the SDOF term over the band
   around the peak, one residue per reference, on the residual
   projected onto that direction.
3. **Scale** — a drive point (a response measured at an excitation
   DOF) pins the mass-normalized scale, via \( A_{jk} = \phi_j \phi_k \). Without
   one the shapes are consistent but the scale is a convention, and
   the set says so everywhere it appears.

When the damping was **held by hand**, one more step: the residues are
scaled so the mode's own CMIF at the fitted line meets the residual
CMIF there — the crown of the parabola is a promise the fit keeps.
Without that, a held damping wider than the data's (matched by eye to
a lobe that includes shoulders) landed the synthesis 15–25 % below the
peak it was aimed at. Automatic fits are *not* pinned: in a dense
cluster the residual peak is several modes deep, and pinning each
rank-one fit to the whole of it is how amplitudes run away. The least
squares knows how much of the residual its one direction explains; the
pin only knows what the user meant.

*Confirm Mode* adopts the mode and the loop moves on. The published
shape set **is** the complete fit state — *Edit Fit* on it later
reopens the session with every mode as published, replayed in
frequency order (the order they were confirmed in is not kept).

## Refine All: going back for the close pairs

Sequential peeling never goes back: the first of a close pair is fit
on data still containing the second's tail, so its residues absorb a
piece of the neighbor. The pair's *sum* tracks the measurement — the
error is in the decomposition, invisible on the one curve the screen
shows, and it surfaces as MAC leakage between the pair or modal masses
that depend on fitting order.

**Refine All** re-fits every mode's residues with the poles held
exactly where you put them. Its structure encodes several lessons
learned on real data:

- **Modes couple only where their bands overlap.** An accelerance
  SDOF term tends to a constant far above its pole — the mass line —
  so a naive global solve lets a quiet fundamental be bought as a shim
  against a loud cluster hundreds of lines away. Modes are grouped
  into clusters by band overlap; distant modes decouple exactly.
- **Each cluster runs a ladder and the CMIF judges it.** Five
  candidates — the pure joint least squares, three ridge strengths
  pulled toward the sequential answer, and the sequential answer
  itself — are each carried all the way to the shapes they would
  publish, and scored by how far their synthesized CMIF sits from the
  measured CMIF over the cluster's own band. Residues are optimized on
  the FRFs; the model among them is selected on the CMIF. With the
  sequential rung always on the ladder, refining cannot lose ground on
  the curve the fit is judged by — when it changes nothing, the status
  bar says so.
- **A cluster of one is not refit.** A lone mode has nobody to borrow
  from, and re-fitting it can only chase noise and the smooth tails of
  louder neighbors. Its sequential fit stands.
- **Coherence weights the whole thing.** If the project holds a
  coherence covering the fit's response DOFs, the fit picks it up
  automatically (the status line says so). The variance of an FRF
  estimate goes as (1 − γ²)/γ², so its inverse weights the solve, the
  shape extraction, and the judging alike — a channel the measurement
  itself distrusts cannot vote noise into every mode's residues.

## What this is not

Honesty about the trade against the polynomial methods:

- **Real normal modes only.** Shape vectors are rotated to the nearest
  real vector. Strongly complex modes (heavily non-proportional
  damping) lose their phase structure; frequencies and dampings are
  still serviceable.
- **No global pole search.** Poles are where you (or Find Mode) put
  them, refined locally. A stabilization diagram can dig repeated
  roots out of a peak that looks like one mode; here that judgment is
  yours, made by watching what the residual does after each confirm —
  if a peak survives a confirm, there is another mode under it.
- **One mode per fit step.** Close pairs are handled by fitting both
  and pressing Refine All, not by a simultaneous high-order solve.

What is bought with those trades: every mode on screen was chosen by a
person and lands where it was aimed; there is no model order to sweep,
no spurious computational poles to cull, and the entire fit state
survives in the published shape set, reopenable and editable at any
time.
