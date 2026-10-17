# Reading Q from a peak's half-power width

A resonance on a magnitude curve (an FRF, a transmissibility, a tracked
sine level) is read for two numbers: where its peak is and how sharp it
is. The sharpness is Q. Find the peak, go down to 0.707 of its height
(3 dB down; *half power*, because power goes with amplitude squared and
0.707 squared is one half), and read the two frequencies where the
curve crosses that level. Q is the peak frequency over the width
between them, and the damping ratio is one over twice Q:

\[
Q = \frac{f_n}{f_2 - f_1}, \qquad \zeta = \frac{1}{2Q}
\]

A peak at 180 Hz whose half-power points sit at 175.5 and 184.5 Hz is
9 Hz wide, and its Q is 20, a damping ratio of 2.5 %. For one mode well
clear of its neighbors the half-power Q and the mode's own damping agree
to within a few times the damping squared: read finely off a one-mass
accelerance, Q comes out 1 % low at a Q of 10, 0.25 % low at 20 and
0.04 % low at 50.

## From a script

`half_power` on an FRF or a spectrum (a tracked sine level is one)
reads the peak nearest a frequency on one record:

```python
reading = frf.half_power(180.0, record=0)
reading.q, reading.damping           # 19.9, 0.0251
reading.low, reading.high            # the two crossings, Hz
reading.lines                        # lines across the width
print(reading.describe())
# Q 19.9 at 180.1 Hz, half-power 175.8 to 184.8 Hz (9.03 Hz, 40 lines)

visualdynamics.plot.plot_half_power(frf, 180.0, path='q.png')
```

The peak is the one the nearest line sits on, found by climbing uphill
from it, so a frequency anywhere on a peak's flank reads that peak and
not a taller one elsewhere; `within=(low, high)` reads the tallest line
in a band instead. The top is placed between lines by a parabola through
the logarithms of the tallest line and its two neighbors, and the
crossings are interpolated between lines. The same reading works on any
curve as `visualdynamics.core.half_power.read_half_power(frequencies,
magnitude, near)`, and returns a `HalfPower`: the peak's frequency and
height, the crossings, the width, Q, the damping ratio, the lines
across the width, and any notes.

`plot_half_power` draws the record with the reading on it: a dotted
line at 0.707 of the peak, a hollow circle on each crossing, a filled
diamond on the top as read between lines, and the Q beside the lower
crossing. A reading is drawn so it can be checked; a crossing placed on
a shoulder is obvious on the plot and invisible in a number. It takes
an FRF or a spectrum and refuses a PSD, an SRS or a coherence. A PSD
is a power, whose half-power level is half its peak, not 0.707 of it;
read at 0.707, a PSD of a Q 20 resonance reads a Q of about 31. An SRS
and a coherence are not a response's magnitude at all.

## In the app

On the fitting screen the reading follows the cursor. Whatever peak the
cursor sits on is read from the residual CMIF the cursor works on, drawn
in the cursor's red, and written at the end of the status line. It moves
with every step of a drag, beside the dashed SDOF curve and the damping
the fit searched.

Until a mode is confirmed the residual CMIF is the measured one, and
the reading sits on the solid curve. A confirmed mode is taken out of
the residual, so near it the residual falls below the measured curve,
and so do the level and crossings read from it. The **Residual** toggle
draws the residual CMIF, gray and dashed; turn it on to see the curve
the reading was taken from.

The two dampings are different things and are shown together on
purpose. The half-power reading is the curve's own width. The fit's
damping is the pole that leaves the least residual
([How the mode fitting works](modal-fitting.md)). On one clean mode
they agree. Where they do not, the curve is not one clean mode: a close
pair reads as one wide peak, and a mode on a neighbor's shoulder reads
wide. That disagreement is worth seeing before a mode is confirmed.

## Resolution: lines across the peak

A peak sampled at a line spacing near its own width is read low and
wide. The tallest line misses the top, and the crossings are
interpolated across straight segments of a curved flank, so Q comes
out low. On a one-mass resonator, over every placement of the lines
against the peak:

| Lines across the half-power width | Q read, worst case |
|---|---|
| 40 | within 0.3 % |
| 8 | 0.7 % low |
| 4 | 3 % low |
| 2 | 14 % low |
| 1 | 44 % low, and the height 22 % low |

From two lines up the reading was low at every placement; nearer one
line a placement can read a little high as well as far low, so a
coarse reading is not a lower bound on Q, only a poor estimate of it.

The reading warns (`HalfPowerWarning`) when fewer than four lines span
the width; the plot's label says so, and the status line gives the
count. The
number is still returned, because it is what the curve says; the
warning says what it cannot be trusted for. The remedy is finer lines:
a longer frame for an FRF or a spectrum, more readings along the sweep
for a tracked level. A mode with a Q of 50 at 60 Hz is 1.2 Hz wide, so
an analysis at 2.5 Hz lines puts half a line across it and cannot read
it at all.

A crossing that is not inside the curve, because the measured band ends
before the curve falls to 0.707 of the peak on that side, is refused
rather than guessed: that side is None, so are the width, Q and
damping, and the warning says which side. (The modal fit's damping
seed borrows the other side's width in that case; a seed may guess, a
reading should not.) So is a peak so much narrower than its lines that
its top, placed between them, stands more than 3 dB above every line:
no crossing can be placed, and no width is read.

## A swept level reads the sweep as well as the mode

A tracked sine level read too fast through a resonance under-reads it:
the mode needs about Q cycles at its frequency to build, and a fast
sweep does not stay inside the half-power band that long. The peak comes
out late and low and the width wide, so the Q read from it is low. The
reading is faithful to the curve; it is the curve that is wrong. A
60 Hz mode with a Q of 20, tracked at 1 octave per minute, reads its Q
within one percent; swept at 20 octaves per minute the same mode reads a
Q near 13, with its peak carried up to almost 64 Hz. Read Q from a
swept level only when the sweep was slow for that mode: well under
\(87 f_n / Q^2\) octaves per minute.
