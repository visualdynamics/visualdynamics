"""The rigid-body residual reading, beside the transformed density it
is read behind.

The Rigid-Body Residual toggle on a density transformed through rigid
shapes draws the measured power at the channels the transform used,
the part the six motions explain and the residual, with the residual's
share against frequency (`plot.transformation`). This panel says what
the transformation is and what the picture says: how many channels go
to how many motions, the conditioning and singular values, the length
the rotations were scaled by, the residual's share over the band, the
frequency up to which it stays under the threshold, and the channel
with the largest residual of its own.

Nothing is set and nothing is made, so the panel is read-only: the
reading is of a transform that exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QLabel, QWidget

from .settings_panel import add_derived, panel_grid

#: the readings, in order: key and row name
ROWS = (('channels', 'Channels'), ('condition', 'Conditioning'),
        ('singular', 'Singular values'), ('length', 'Length scale'),
        ('overall', 'Residual, band'), ('holds', 'Holds to'),
        ('threshold', 'Threshold'), ('worst', 'Worst channel'))


class TransformationPanel(QWidget):
    """The reading, in rows, with the notes under them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Rigid-Body Residual')
        grid = panel_grid(self, self.title)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 1).items()}
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 1 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(2 + len(ROWS), 1)

    def show_reading(self, reading: Any, refusal: str = '', *,
                     length_scale: float = 1.0,
                     length_unit: str = 'm') -> None:
        """Write `reading`, a `TransformationResidual`, into the rows,
        the length scale converted by `length_scale` (meters to
        `length_unit`). None clears them, with `refusal` as the note."""
        from ..core.transformation import MAX_CONDITION

        values = dict.fromkeys(key for key, _name in ROWS)
        notes: list[str] = [refusal] if refusal else []
        if reading is not None:
            t = reading.transformation
            condition = t.condition
            singular = t.singular_values
            values.update(
                channels=f'{len(t.inputs)} to {len(t.outputs)} motions',
                condition=(f'{condition:.3g}' if np.isfinite(condition)
                           else 'infinite'),
                singular=f'{singular[0]:.3g} to {singular[-1]:.3g}',
                length=f'{t.length_scale * length_scale:.4g} {length_unit}',
                threshold=f'{reading.threshold:.0%} of the power')
            if np.isfinite(reading.overall):
                values['overall'] = f'{reading.overall:.1%}'
                holds = reading.holds_to
                values['holds'] = (
                    'every line' if reading.holds_throughout
                    else f'{holds:.4g} Hz' if holds is not None
                    else 'not at the first line')
            worst = reading.worst_channel
            if worst is not None:
                share = reading.channel_fractions[
                    reading.channels.index(worst)]
                values['worst'] = f'{worst}, {share:.1%}'
            if not condition <= MAX_CONDITION:
                notes.append(
                    f'the conditioning is past {MAX_CONDITION:g}: an error '
                    'at the channels reaches the motions that many times '
                    'larger, and building the transformation is refused '
                    'there unless the limit is raised')
            notes.extend(reading.notes)
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        self.notes.setText('\n'.join(note[0].upper() + note[1:] + '.'
                                     for note in notes if note))
