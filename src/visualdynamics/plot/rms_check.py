"""The rms check, written on the autospectrum it checks.

The check (`core.rms_check`) is two numbers and the gap between them:
the rms from the area under the spectrum and the rms of the time
history it came from. A picture of the two would show nothing, since a
scaling error worth catching is a fraction of a decibel to a few
decibels and a windowed estimate is a few hundredths off, so the view
is the numbers themselves, in words, in the corner of the spectrum:

- the rms from the spectrum, then the rms from the time history, each
  in the display unit and named, never told apart by color;
- the gap in dB, signed, with the threshold it is judged against;
- the window's share of the gap, when the frames are known and the gap
  is over the threshold, so a gap that is all the window's reads as
  such;
- the verdict in words: agree, the window's gap, a scaling error named,
  or a gap to look for one behind, under a heading in bold that says
  'agree', 'the window's gap', 'flagged' or 'not judged'.

The words go in one label pinned to the plot's top-right corner, in
the plot's foreground color unless the caller passes another. It is
parented to the view box rather than added to it, so it stays where it
is through a zoom or a pan, takes no part in the plot's bounds, and is
not data: the legend, the copy and the plot's preview never include it.

`RmsCheckLabel` owns its label and rewrites it, so a reading that is
redone (another threshold, another record) updates the same item;
`add_rms_check` writes one check once, for a figure.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np

#: how wide the verdict is wrapped, in characters: the label sits over
#: the spectrum's corner and should not cover the middle of the plot
WRAP = 46

#: where the label sits, pixels in from the view box's top-right corner
OFFSET = (-10, 8)


def heading(check: Any) -> str:
    """The judgment in a word or two, the label's first line: 'agree'
    within the threshold, 'the window's gap' outside it when the window
    is the whole of the gap, 'flagged' outside it otherwise, and 'not
    judged' when there is no gap to judge (`RmsCheck.unjudged`)."""
    if not np.isfinite(check.gap_db):
        return 'rms check: not judged'
    if check.passed:
        return 'rms check: agree'
    return 'rms check: ' + ("the window's gap" if check.explained
                            else 'flagged')


def label_lines(check: Any, unit: str = '', scale: float = 1.0, *,
                brief: bool = False) -> list[str]:
    """The lines the label writes, as plain text: the heading, both rms
    (converted by `scale` into `unit`), the gap against its threshold,
    the window's share when it bears on the verdict, and the verdict,
    wrapped. `brief` keeps the heading and the gap alone, for a plot
    with a pane beside it that holds the rest."""
    def rms(value: float) -> str:
        return f'{value * scale:.4g} {unit} rms'.replace('  ', ' ')

    gap = check.gap_db
    lines = [heading(check),
             f'from the spectrum: {rms(check.spectrum_rms)}',
             f'from the time history: {rms(check.time_rms)}',
             (f'gap {gap:+.3f} dB, threshold {check.threshold_db:g} dB'
              if np.isfinite(gap) else 'no gap to judge')]
    if brief:
        return [lines[0], lines[3]]
    share = check.window_gap_db
    if not check.passed and share is not None and np.isfinite(share):
        lines.append(f"the window's share: {share:+.3f} dB")
    lines.extend(textwrap.wrap(check.verdict, WRAP))
    return lines


def label_html(check: Any, unit: str = '', scale: float = 1.0, *,
               brief: bool = False) -> str:
    """The label as rich text for a pyqtgraph `LabelItem`: the lines of
    `label_lines` (`brief` as there), the heading in bold so the
    judgment reads first."""
    from html import escape

    lines = [escape(line, quote=False)
             for line in label_lines(check, unit, scale,
                                brief=brief)]
    lines[0] = f'<b>{lines[0]}</b>'
    return '<br>'.join(lines)


class RmsCheckLabel:
    """The label that writes one rms check on a pyqtgraph plot.

    Made once per plot with `plot`, the PlotItem to write on, and
    `color`, a pyqtgraph color (the plot's foreground by default); then
    `show(check, ...)` writes it and `clear()` empties it. `brief`
    writes the heading and the gap alone, for the window's plot, whose
    pane holds the rest.
    """

    def __init__(self, plot: Any, color: Any = None, *,
                 brief: bool = False, size: str = '9pt',
                 z: float = 44.0) -> None:
        import pyqtgraph as pg

        if color is None:
            color = plot.getAxis('left').pen().color()
        self.plot: Any = plot
        self.color: Any = pg.mkColor(color)
        self.brief: bool = bool(brief)
        self.label: Any = pg.LabelItem(justify='right', color=self.color,
                                       size=size)
        self.label.setZValue(z)
        # parented, not added: in the view box's own pixels, so a zoom
        # leaves it in the corner and the bounds never see it
        self.label.setParentItem(plot.getViewBox())
        self.label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=OFFSET)
        self.check: Any = None

    def show(self, check: Any, *, unit: str = '', scale: float = 1.0
             ) -> None:
        """Write `check`, an `RmsCheck`, its rms converted by `scale`
        into `unit`."""
        self.check = check
        self.label.setText(label_html(check, unit, scale,
                                      brief=self.brief), color=self.color)
        self.label.setVisible(True)

    def text(self) -> str:
        """What the label says, as rich text; empty when cleared."""
        return self.label.text if self.check is not None else ''

    def clear(self) -> None:
        """Empty the label until the next `show`."""
        self.check = None
        self.label.setText('')
        self.label.setVisible(False)

    def remove(self) -> None:
        """Take the label off the plot for good."""
        scene = self.label.scene()
        self.label.setParentItem(None)
        if scene is not None:
            scene.removeItem(self.label)


def add_rms_check(plot: Any, check: Any, *, color: Any = None,
                  unit: str = '', scale: float = 1.0) -> RmsCheckLabel:
    """Write `check`, an `RmsCheck`, on `plot` and return the label.
    `color` is as for `RmsCheckLabel`, the others as for
    `RmsCheckLabel.show`."""
    label = RmsCheckLabel(plot, color)
    label.show(check, unit=unit, scale=scale)
    return label
