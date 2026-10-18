"""Spin boxes that report a typed number only when the edit is finished.

A Qt spin box reports every keystroke by default (`keyboardTracking` is
on unless it is turned off), so typing `50` into an overlap field
reports 5 and then 50. Each report is a real edit here: it is applied to
the object, the shading moves, and the panel restates the clamped value.
Restating it takes the selection away in the middle of the number, so
the second digit has nowhere to go, and with tracking on 50 % overlap
cannot be entered at all.

So every spin box in the application commits when the edit is finished
(Enter, Tab, or clicking elsewhere) or when its own arrows step it. That
is Qt's `keyboardTracking(False)`. It is a rule for the whole interface,
so it is one call here instead of a setting repeated on each field.

A drag is different: it is continuous and stays live, because the
gesture is the value. A half-typed number is not a value the person
means.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

if TYPE_CHECKING:                                    # pragma: no cover
    from PySide6.QtWidgets import QAbstractSpinBox

# Brandon, 2026-08-27: the overlap field took the selection away after
# the first digit and 50 % could not be typed; the same on frame length.
# This does not conflict with principle 8 (effort where the user is
# waiting): a drag stays live, and only typing waits for the finished
# edit.


def commit_on_enter(*boxes: QAbstractSpinBox) -> None:
    """Make these spin boxes report a value only when an edit is
    finished.

    Parameters
    ----------
    *boxes : QAbstractSpinBox
        Spin boxes (`QSpinBox`, `QDoubleSpinBox`) whose `valueChanged`
        drives something. Each then reports only a finished edit or a
        step of its own arrows.
    """
    for box in boxes:
        box.setKeyboardTracking(False)


class _Clamping:
    """A spin box that lets a number be typed and then corrects it.

    Qt refuses the keystroke: with a range ending at 1638, typing 2000
    stops after the third digit, because the fourth would make a value
    out of range and the validator calls that invalid. The number never
    appears, nothing says why, and the box simply stops responding
    (Brandon, 2026-08-28).

    Here the typing is allowed and the *value* is corrected when the
    edit finishes — the same clamp-and-restate the panels already use
    for a frame count that will not fit or a resolution that is not
    reachable. What the person typed is a statement of intent; the
    right answer to "2000 Hz" on a record that reaches 1638 is 1638,
    not a box that ignores the last keypress.
    """

    def validate(self, text, position):
        """Accept anything that parses as a number in this box's format.

        Range is deliberately not checked. `valueFromText` is where a
        value out of range is brought back in, which is after the
        person has finished typing rather than during.
        """
        from PySide6.QtGui import QValidator

        stripped = self._number_in(text)
        if stripped == '':
            return (QValidator.State.Intermediate, text, position)
        try:
            float(stripped)
        except ValueError:
            # a lone sign or decimal point on the way to a number
            if stripped in ('-', '+', '.', '-.', '+.'):
                return (QValidator.State.Intermediate, text, position)
            return (QValidator.State.Invalid, text, position)
        return (QValidator.State.Acceptable, text, position)

    def valueFromText(self, text):     
        """The number typed, brought inside the range."""
        stripped = self._number_in(text)
        try:
            value = float(stripped)
        except ValueError:
            return self.value()
        return max(self.minimum(), min(self.maximum(), value))

    def _number_in(self, text):
        """The text with this box's prefix, suffix and separators gone."""
        cleaned = str(text).strip()
        for fixed in (self.prefix(), self.suffix()):
            if fixed and cleaned.endswith(fixed):
                cleaned = cleaned[:-len(fixed)].strip()
            if fixed and cleaned.startswith(fixed):
                cleaned = cleaned[len(fixed):].strip()
        return cleaned.replace(',', '').replace(' ', '').strip()


class SpinBox(_Clamping, QSpinBox):
    """A whole-number spin box that commits on a finished edit and
    brings a typed value inside its range.

    Qt's own spin box refuses a keystroke that would put the value out
    of range: with a maximum of 1638, typing 2000 stops after the third
    digit, and nothing says why. Here any number can be typed, and when
    the edit is finished it is clamped to the range and rounded, so 2000
    becomes 1638. Keyboard tracking is off, as `commit_on_enter` sets
    it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)

    def valueFromText(self, text):     
        return round(_Clamping.valueFromText(self, text))


#: the places a length box keeps: the guides' dimensions are inch
#: fractions to the 32nd and 64th (1.96875, -0.15625), and four places
#: turned the BARC's channel positions into gaps, the model in four
#: pieces (2026-10-17)
LENGTH_DECIMALS = 6


def number_box(low: float, high: float, *, value: float | None = None,
               decimals: int = 2, step: float | None = None, tip: str = '',
               suffix: str = '') -> DoubleSpinBox:
    """A `DoubleSpinBox` over [`low`, `high`] for a settings panel.

    The range and the places are set before the value, which is the
    order that keeps it: a value set first is clamped to the default
    range and rounded to the default two places. Like every
    `DoubleSpinBox`, it reports only a finished edit, since each report
    redraws a reading.
    """
    box = DoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    if step is not None:
        box.setSingleStep(step)
    if value is not None:
        box.setValue(value)
    box.setToolTip(tip)
    if suffix:
        box.setSuffix(suffix)
    return box


class DoubleSpinBox(_Clamping, QDoubleSpinBox):
    """A decimal spin box that commits on a finished edit and brings a
    typed value inside its range, as `SpinBox` does for whole numbers.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)
