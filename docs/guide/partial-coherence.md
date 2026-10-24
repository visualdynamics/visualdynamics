# Which drive explains a response: partial and reference coherence

In a test with several drives, the multiple coherence says how much of
a response the drives account for together. It does not say which
drive does, and it does not say whether the drives were independent
enough for the identification to tell them apart. Three readings of
the same cross-spectral matrix answer those questions.

**The reference coherence** is the ordinary coherence between every
pair of drives,

\[
\gamma^2_{ij} = \frac{|G_{ij}|^2}{G_{ii}\,G_{jj}}
\]

Two drives coherent near one at a line are one drive there: the matrix
of the references is nearly singular, and an FRF computed from it is
whichever of many answers the pseudo-inverse picks. Drives are meant
to be independent random signals, so a reference coherence near one
says the drives were wired, generated or coupled through the article
in a way that defeats the identification.

**The partial coherence** of a response with one drive is the ordinary
coherence between the two after the linear effect of every other drive
has been removed from both. With the other drives \(r\), the
conditioned spectra are

\[
G_{ab\cdot r} = G_{ab} - G_{ar}\,G_{rr}^{-1}\,G_{rb}, \qquad
\gamma^2_{iy\cdot r} = \frac{|G_{iy\cdot r}|^2}{G_{ii\cdot r}\,G_{yy\cdot r}}
\]

It says how much of what the other drives leave of the response this
drive explains. That is not the share of the whole response: a drive
can explain nearly all of a small remainder. With one drive the
partial coherence is the ordinary coherence, and with two the multiple
coherence is built from an ordinary and a partial one,

\[
1 - \gamma^2_{y:x} = (1 - \gamma^2_{1y})(1 - \gamma^2_{2y\cdot 1})
\]

These are Bendat and Piersol's conditioned spectral analysis.

**The virtual coherence** is the coherence of a response with each
principal drive: the uncorrelated combinations of the drives given by
the eigenvectors of their matrix, \(G_{rr} = U \Lambda U^\mathsf{H}\).
The principal drives have autospectra \(\Lambda\) and cross spectra
\(U^\mathsf{H} G_{ry}\) with the response, and their coherences with it
sum to the multiple coherence. A principal drive whose power is small
next to the first is a direction the drives hardly excited, so the
count of large ones is how many independent drives the test had at
that line.

Where a drive is explained completely by the others, nothing of it is
left to be coherent with anything, and its partial coherence is not a
number rather than zero. The same holds for a channel that recorded
nothing.

## From a script

```python
reading = history.read_partial_coherence()   # the reference channels
reading.references, reading.responses        # ['1Z+', '2Z+'], ['101Z+']
point = reading.at(150.0, '101Z+')
point.multiple                               # 0.986
point.ordinary                               # [0.956, 0.057]
point.partial                                # [0.985, 0.683]
reading.reference[0, 1]                      # the drives' coherence, every line
reading.coupled()      # [('1Z+', '2Z+', 0.19, 0.998)]: past 0.9 on 19 % of lines
reading.effective_references()               # independent drives, every line
print(reading.describe('101Z+'))

visualdynamics.plot.plot_partial_coherence(reading, '101Z+',
                                           path='partial.png')
```

The reading is made from the frames `compute_multiple_coherence` and
`compute_frfs` use, against the same references, so its multiple
coherence is that method's to round-off. In a project,
`project.read_partial_coherence(name)` reads behind a multiple
coherence made by Compute Multiple Coherence, from its history with the
averaging it was computed with, or reads a time history directly. A
record with a sample that is not a number in the frames read is
refused, naming the channel. The same reading works on any
cross-spectral matrix whose first channels are the references, as
`core.partial_coherence.read_partial_coherence(frequency, matrix,
references, responses)`.

![The multiple coherence over each drive's partial coherence, the
coherence between the drives, and the principal drives'
powers](images/partial-coherence.png)

The figure is 64 seconds at 1024 Hz of two drives that are one signal
below 100 Hz and independent above it, and a response with a 150 Hz
path from the first and a 300 Hz path from the second, both at Q 8,
with a little noise, read in 255 Hann frames at half overlap
(`tools/make_concept_figures.py`). The multiple coherence is near one
everywhere. Below 100 Hz the drives' coherence is 0.997, the second
principal drive is 32 dB under the first, and the partial coherences
fall to a few hundredths: the response is explained, but not by either
drive that could be named. At 150 Hz the first drive's ordinary and
partial coherence are both near one, and the second's ordinary
coherence is 0.06 while its partial is 0.68, since with the first
drive's large share removed what is left is mostly the second's. At
300 Hz the roles are reversed.

## In the app

Select a multiple coherence made by Compute Multiple Coherence from a
history with two or more drives, and press **Partial Coherence** on the
bar. The figure above is drawn for the record picked in the tree, and
the pane reads, over the band, the response, the drives and the
averages, the median multiple coherence, the drive that explains most
once the others are removed, each drive's median partial coherence, the
pair of drives most alike and how many independent drives the test had
in effect, with the notes under them. Pick another record of the
coherence to read another response. The view is journaled as its
`plot_partial_coherence` call.
