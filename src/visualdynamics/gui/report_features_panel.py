"""The Report Features panel: which optional features a report carries.

Shown in the report editor's pane when no block is selected, under the
report's title and marking. A preset drop-down sets a starting point; a
check box per feature, grouped as the registry groups them
(`report.features.GROUPS`), turns one feature on or off. A feature that
cannot be enabled is greyed out with the reason beneath it: not yet
available, or the feature or capability it needs. The description of
each feature is its tooltip.

The panel decides nothing. Every change is emitted as `requested`, the
editor makes it through `report.features.configure` (in the app,
through `Project.report_features`, so it is journaled), and the panel
is shown the result with `show_state`.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..report.features import FEATURES, GROUPS, PRESETS, preset_of, reason

#: what the drop-down shows when the features match no preset
CUSTOM = 'Custom'


class ReportFeaturesPanel(QWidget):
    """The preset drop-down and a check box per report feature.

    `requested` carries one change, as the keyword arguments of
    `report.features.configure`: {'preset': key}, {'enable': [key]} or
    {'disable': [key]}.
    """

    requested = Signal(dict)

    def __init__(self, features: Sequence[str] = (),
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self._loading = False
        self.title: QLabel = QLabel('<b>Report Features</b>')
        layout.addWidget(self.title)
        intro = QLabel('Optional additions to the report. With none '
                       'enabled, the report is the one Visual Dynamics '
                       'writes.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.preset_box: QComboBox = QComboBox()
        for preset in PRESETS.values():
            self.preset_box.addItem(preset.title, preset.key)
            self.preset_box.setItemData(
                self.preset_box.count() - 1, preset.description,
                Qt.ItemDataRole.ToolTipRole)
        self.preset_box.addItem(CUSTOM, None)
        self.preset_box.setToolTip(
            'A starting point: the default (no features), a test report, '
            'or a flash report. A preset enables the features it names '
            'that are available and says which it skipped.')
        self.preset_box.currentIndexChanged.connect(self._preset_chosen)
        layout.addWidget(self.preset_box)
        #: a check box per feature, by key
        self.boxes: dict[str, QCheckBox] = {}
        #: the line beneath each box that says why it cannot be enabled
        self.reasons: dict[str, QLabel] = {}
        for group in GROUPS:
            members = [f for f in FEATURES.values() if f.group == group]
            if not members:
                continue
            heading = QLabel(f'<b>{group}</b>')
            heading.setContentsMargins(0, 6, 0, 0)
            layout.addWidget(heading)
            for feature in members:
                box = QCheckBox(feature.title)
                box.toggled.connect(
                    lambda on, key=feature.key: self._toggled(key, on))
                layout.addWidget(box)
                why = QLabel()
                why.setWordWrap(True)
                why.setEnabled(False)            # greyed, as the box is
                why.setContentsMargins(22, 0, 0, 0)
                layout.addWidget(why)
                self.boxes[feature.key] = box
                self.reasons[feature.key] = why
        #: what the last change had to say: a preset's skipped features,
        #: a feature turned off with the one it needs, or a refusal
        self.notes: QLabel = QLabel()
        self.notes.setWordWrap(True)
        layout.addWidget(self.notes)
        self.show_state(features)

    def show_state(self, features: Sequence[str],
                   notes: Sequence[str] = ()) -> None:
        """Show `features` as the report's enabled features, with each
        box greyed or live by what it needs, and `notes` beneath."""
        self._loading = True
        try:
            on = [key for key in features if key in FEATURES]
            for key, box in self.boxes.items():
                feature = FEATURES[key]
                box.setChecked(key in on)
                why = reason(key, on)
                # a box that is on stays live, so it can be turned off
                # even if what it needs has gone (a file from elsewhere)
                box.setEnabled(not why or key in on)
                tip = feature.description
                if why:
                    tip += f'\n\n{why[0].upper()}{why[1:]}.'
                box.setToolTip(tip)
                label = self.reasons[key]
                label.setText(why)
                label.setVisible(bool(why))
            preset = preset_of(on)
            index = self.preset_box.findData(preset)
            self.preset_box.setCurrentIndex(
                index if index >= 0 else self.preset_box.count() - 1)
            self.notes.setText('\n'.join(notes))
            self.notes.setVisible(bool(notes))
        finally:
            self._loading = False

    def _toggled(self, key: str, on: bool) -> None:
        if self._loading:
            return
        self.requested.emit({'enable': [key]} if on else {'disable': [key]})

    def _preset_chosen(self, _index: int) -> None:
        if self._loading:
            return
        key = self.preset_box.currentData()
        if key is None:
            return                       # 'Custom' is a state, not a choice
        self.requested.emit({'preset': key})
