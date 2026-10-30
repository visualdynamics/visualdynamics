"""The Run Log's header forms: the test header in full, for a test opened,
the roll to the next test and the current test.

Every form here is an `EntryForm` (`gui.run_log_dock`): its event time
taken when it opens, Enter writing it, Esc putting it away, and nothing
written when a box cannot be read.

**The header** comes in three forms: the test opened, the roll to the
next test (every field offered as it stands and carried unless
changed), and the current test's header, where each field changed is a
header change and the fields carried from the last test are marked
until they are confirmed. Its fields are grouped as `HEADER_GROUPS`
lists them, and every field of `run_log.HEADER_FIELDS` is in one group.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..core.run_log import HEADER_FIELDS
from ..core.run_log_model import TEST_KINDS
from .run_log_dock import EntryForm, Snapshot, notches_text

#: the header's fields in the groups the header form shows them in:
#: (group title, ((field, label, box kind), ...)). Every field of
#: `run_log.HEADER_FIELDS` is in exactly one group.
HEADER_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    ('Identity', (
        ('test_kind', 'Kind', 'choice'),
        ('title', 'Title', 'text'),
        ('test_class', 'Class', 'text'),
        ('purpose', 'Purpose', 'text'),
        ('article', 'Article', 'text'),
        ('serial', 'Serial', 'text'),
        ('configuration', 'Configuration', 'text'))),
    ('Governing documents', (
        ('document', 'Governing document', 'text'),
        ('document_revision', 'Its revision', 'text'),
        ('plan_revision', 'Plan revision', 'text'),
        ('procedure', 'Procedure', 'text'),
        ('procedure_revision', 'Procedure revision', 'text'),
        ('profile_source', 'Profile source', 'text'))),
    ('Step schedule and hold points', (
        ('steps_db', 'Step schedule (dB)', 'numbers'),
        ('hold_points_db', 'Hold points (dB)', 'numbers'))),
    ('Planned notches', (
        ('notches_planned', 'Planned notches (axis: band, depth)',
         'notches'),)),
    ('Limits and tolerances', (
        ('limits', 'Limits', 'text'),
        ('tolerance', 'Tolerance', 'text'),
        ('abort_bands', 'Abort bands', 'text'))),
    ('Duration', (
        ('duration_s', 'Duration (s)', 'number'),
        ('duration_minus_pct', 'Tolerance under (%)', 'number'),
        ('duration_plus_pct', 'Tolerance over (%)', 'number'))),
    ('Amplifier', (
        ('sigma', 'Sigma', 'number'),
        ('rated_current_peak', 'Peak current rating (A)', 'number'),
        ('rated_current_rms', 'Continuous current rating (A)', 'number'),
        ('rail_voltage', 'Rail (V)', 'number'),
        ('drive_window', 'Drive window at full level (low, high)',
         'numbers'))),
    ('Projected headroom', (
        ('projected_current', 'Projected peak current (A)', 'number'),
        ('projected_voltage', 'Projected peak voltage (V)', 'number'))),
    ('Shock', (
        ('pulse_budget', 'Pulse budget', 'whole'),)),
)

# ---- the header ----------------------------------------------------------------


def header_form(name: str, taken: datetime, seen: Snapshot | None
                ) -> EntryForm:
    """The header form `name`: 'open test' (every field blank, the kind
    needed), 'roll' (every field offered as the current test has it,
    carried unless changed) or 'header' (the current test's header,
    each field changed written as a header change, the carried fields
    marked and confirmed with Confirm). Files named on the opening or
    the roll are attached by SHA-256."""
    header = dict(seen.header) if seen is not None and name != 'open test' \
        else {}
    carried = set(seen.carried) if seen is not None and name == 'header' \
        else set()
    fields: list[tuple[str, str, str, Any]] = []
    for group, rows in HEADER_GROUPS:
        kept = [row for row in rows
                if not (name == 'header' and row[0] == 'test_kind')]
        if not kept:
            continue
        fields.append(('', group, 'heading', None))
        for field, label, kind in kept:
            choices = None
            if field == 'test_kind':
                choices = ('', *TEST_KINDS) if name == 'roll' else TEST_KINDS
            if (name == 'roll' and field in header) or field in carried:
                label += ' (carried)'
            fields.append((field, label, kind, choices))
    if name == 'header':
        fields.append(('', 'Carried fields', 'heading', None))
        fields.append(('confirm', 'Confirm those left as they are', 'flag',
                       None))
    else:
        fields.append(('', 'Attached', 'heading', None))
        fields.append(('files', ('Files (the plan, the procedure, the '
                       'profile)'), 'files', None))
    if name == 'roll':
        fields.append(('reason', 'Why this test closes', 'text', None))
    if name == 'open test':
        title = 'Open the next test'
        note = ''
    elif name == 'roll':
        title = 'Roll to the next test'
        note = ('Every field left as it is offered is carried to the next '
                'test, and marked carried there until it is confirmed or '
                'changed. A box emptied is carried as it was.')
    else:
        test = seen.status if seen is not None else None
        title = (f'Header of test {test.test}: {test.title} ({test.kind})'
                 if test is not None and test.test is not None
                 else 'Header')
        note = ('Each field changed is written as a header change. Once the '
                'test is running, a change of a limit, tolerance, abort '
                'band, planned notch or hold point raises a hold until a '
                'deviation names it.')
        if carried:
            note += (' Carried from the last test and not yet confirmed: '
                     + ', '.join(field for field in HEADER_FIELDS
                                 if field in carried) + '.')
    return EntryForm(name, title, fields, taken,
                     required=('test_kind',) if name == 'open test' else (),
                     note=note, defaults=header,
                     text='Why' if name == 'header' else 'Note')


def header_changes(values: Mapping[str, Any], header: Mapping[str, Any]
                   ) -> dict[str, Any]:
    """The header fields among `values` (a header form's) that differ
    from `header`, the fields as they stand."""
    return {field: value for field, value in values.items()
            if field in HEADER_FIELDS and not same(field, value,
                                                   header.get(field))}


def same(field: str, value: Any, held: Any) -> bool:
    """Whether a header field typed as `value` is the `held` value: a
    planned notch list compared as it is written back, so a band stored
    as two numbers and one typed as text are the same band."""
    if field == 'notches_planned':
        return held is not None and notches_text(value) == notches_text(held)
    return value == held
