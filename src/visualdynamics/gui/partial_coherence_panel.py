"""The partial coherence reading, beside the multiple coherence it is
read behind.

The Partial Coherence toggle on a multiple coherence draws, for the
response picked, the multiple coherence over each reference's partial
coherence, the coherence between the references and the principal
references' powers (`plot.partial_coherence`). This panel says, over
the band, what the picture says: which reference explains most of the
response once the others are removed, each reference's median partial
coherence, the pair of references most nearly the same drive, and how
many independent drives the test had in effect.

Nothing is set and nothing is made, so the panel is read-only: the
reading is of a coherence that exists, from the frames it was computed
from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QLabel, QWidget

from .settings_panel import add_derived, panel_grid

#: the readings, in order: key and row name
ROWS = (('response', 'Response'), ('references', 'References'),
        ('averages', 'Averages'), ('multiple', 'Multiple, median'),
        ('explains', 'Explains most'), ('partials', 'Partial, median'),
        ('coupled', 'Most alike'), ('drives', 'Drives in effect'))


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float('nan')


class PartialCoherencePanel(QWidget):
    """The reading of one response over the band, in rows, with the
    notes under them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Partial Coherence')
        grid = panel_grid(self, self.title)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 1).items()}
        self.derived['partials'].setWordWrap(True)
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 1 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(2 + len(ROWS), 1)

    def show_reading(self, reading: Any, response: int = 0,
                     refusal: str = '') -> None:
        """Write one response of `reading`, a `PartialCoherence`, into
        the rows. None clears them, with `refusal` as the note."""
        from ..core.partial_coherence import COUPLED

        values = dict.fromkeys(key for key, _name in ROWS)
        notes: list[str] = [refusal] if refusal else []
        if reading is not None:
            k = reading.index(response)
            medians = [_median(row) for row in reading.partial[k]]
            values.update(
                response=reading.responses[k],
                references=', '.join(reading.references),
                averages=f'{reading.averages} frames',
                multiple=f'{_median(reading.multiple[k]):.3f}',
                partials='\n'.join(
                    f'{name}  {value:.3f}' if np.isfinite(value)
                    else f'{name}  undefined'
                    for name, value in zip(reading.references, medians)))
            if np.isfinite(medians).any():
                best = int(np.nanargmax(medians))
                values['explains'] = reading.references[best]
            pairs = [(reading.references[i], reading.references[j],
                      _median(reading.reference[i, j]))
                     for i in range(len(reading.references))
                     for j in range(i + 1, len(reading.references))]
            pairs = [pair for pair in pairs if np.isfinite(pair[2])]
            if pairs:
                one, other, value = max(pairs, key=lambda pair: pair[2])
                values['coupled'] = f'{one}, {other}: {value:.2f}'
            effective = reading.effective_references()
            live = effective[effective > 0]
            if live.size:
                values['drives'] = (f'{int(np.median(live))} of '
                                    f'{len(reading.references)}')
            notes.extend(reading.notes)
            coupled = reading.coupled()
            if coupled:
                one, other, share, _top = coupled[0]
                notes.append(f'{one} and {other} are coherent over '
                             f'{COUPLED:g} on '
                             f'{share:.0%} of the lines, where the '
                             'identification cannot tell them apart')
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        self.notes.setText('\n'.join(note[0].upper() + note[1:] + '.'
                                     for note in notes if note))
