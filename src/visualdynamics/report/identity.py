"""Identity and issue: which report, which revision, where it ends, and
no silent reissue (report feature `report_identity`).

A page that cannot say which report and which revision it belongs to,
or where the report ends, cannot be told apart from a page of another
report, and a reader cannot tell a complete copy from a partial one.
With the feature on, every printed page carries the laboratory, the
report number and revision, the title and page n of N, and the first
page carries the identification block: the laboratory's address, where
the work was done, the customer and contact, the dates of receipt, test
and issue, the people authorizing the report with a blank line for each
signature, and the statement that the report is not reproduced except
in full. The report closes with "End of report." (or "End of
amendment.").

Every field is entered by the author (`identify`). The program never
writes a report number, a revision, a date or a name of its own: a
field not yet entered shows as not entered, and the report cannot be
issued until each one the identification needs is there (`missing`).

A report is a draft until it is issued. Issuing it (`Project.issue_report`)
records the revision issued, with the SHA-256 of the issued document
and of its content. From then on the report is not changed under that
identity: an export whose content differs from what was issued is
refused (`guard`), whatever changed, a block, a number or the data
under a figure. A change after issue is an amendment (`amend`): its
own revision, the report it amends by number, revision and date, and a
table of what changed (where, what it said, what it now says, and
why), which is then issued in turn. The original's record is kept.

The paginated issued form is a PDF printed from the same page
(`report.pdf`); the page numbers are the print engine's own page
counters, so page n of N is what the PDF holds.
"""

from __future__ import annotations

import hashlib
import html
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: the feature's key in `report.features`
FEATURE = 'report_identity'

#: a report's status: a draft (including an amendment being prepared),
#: issued, or an issued amendment
STATUSES = ('draft', 'issued', 'amended')

#: the statement on the first page while `full_only` is on (the default)
FULL_ONLY = ('This report shall not be reproduced except in full without '
             'the written approval of the laboratory.')

#: what a field not yet entered shows, on the page and in a refusal
NOT_ENTERED = 'not entered'

#: the last line of an amendment
END_AMENDMENT = 'End of amendment.'

#: the printed page's margins in inches, (top and bottom, sides): room
#: for the identity line above and the page number below
MARGINS = (0.75, 0.6)


@dataclass(frozen=True)
class Field:
    """One identification field the author enters.

    `required` fields must be entered before the report is issued.
    """

    key: str
    label: str
    required: bool = True


#: the fields, in the order the identification block lists them
FIELDS: tuple[Field, ...] = (
    Field('number', 'Report number'),
    Field('revision', 'Revision'),
    Field('laboratory', 'Laboratory'),
    Field('address', 'Laboratory address'),
    Field('location', 'Location of the work'),
    Field('customer', 'Customer'),
    Field('contact', 'Customer contact'),
    Field('received', 'Date of receipt of the item'),
    Field('tested', 'Date of test'),
    Field('issued', 'Date of issue'),
    Field('authorized', 'Authorized by'),
    Field('distribution', 'Distribution', required=False),
)

#: the fields by key
BY_KEY: dict[str, Field] = {field.key: field for field in FIELDS}

#: the fields a report template carries: the laboratory's, which are
#: the same on every report it writes. A number, a revision, a status
#: or an issue record never passes from one report to the next.
TEMPLATE_FIELDS = ('laboratory', 'address', 'full_only')

#: what changes only by issuing (`record_issue`) and amending (`amend`)
_CONTROLLED = ('status', 'amends', 'changes', 'record')

#: the parts of one change in an amendment's table, and which must be
#: given (the reason where appropriate: a corrected typing error needs none)
CHANGE_PARTS = ('where', 'was', 'now', 'reason')
_CHANGE_REQUIRED = ('where', 'was', 'now')


def _identity(report: Any) -> dict[str, Any]:
    return dict(getattr(report, 'identity', None) or {})


def value(report: Any, key: str) -> Any:
    """What the author entered for field `key` ('' or [] when nothing),
    or the report's `status` ('draft' until issued) or `full_only`
    (True until turned off)."""
    held = _identity(report)
    if key == 'status':
        return held.get('status', 'draft')
    if key == 'full_only':
        return bool(held.get('full_only', True))
    if key == 'authorized':
        return list(held.get('authorized', []))
    return str(held.get(key, ''))


def record(report: Any) -> list[dict[str, Any]]:
    """The issue record: one entry per revision issued, oldest first,
    each with its number, revision, status, date of issue, the SHA-256
    of the issued PDF and of its content, the file written, and for an
    amendment the changes it made."""
    return [dict(entry) for entry in _identity(report).get('record', [])]


def _people(given: Any) -> list[str]:
    if isinstance(given, str):
        lines = given.splitlines()
    elif isinstance(given, Iterable):
        lines = list(given)
    else:
        raise TypeError(f'authorized is a list of names, not '
                        f'{type(given).__name__}')
    out = []
    for line in lines:
        if not isinstance(line, str):
            raise TypeError(f'authorized: each name is a str, not '
                            f'{type(line).__name__}')
        if line.strip():
            out.append(line.strip())
    return out


def identify(report: Any, **fields: Any) -> None:
    """Enter identification fields on `report`.

    Each keyword is a key of `FIELDS`, a string ('' clears it), except
    `authorized`, the people authorizing the report, one per entry or
    one per line ("name, role"), and `full_only`, whether the first
    page says the report is not reproduced except in full (True by
    default).

    Refused before anything changes: an unknown field; the status, the
    amendment and the issue record, which change only by issuing and
    amending; a value of the wrong type; and the number or revision of
    a report that has been issued, which change only by an amendment.

    Raises
    ------
    ValueError
        For a refusal; the message names the field and the reason.
    TypeError
        When a value is of the wrong type.
    """
    held = _identity(report)
    issued = bool(held.get('record'))
    changes: dict[str, Any] = {}
    for key, given in fields.items():
        if key in _CONTROLLED:
            raise ValueError(f'{key} is not entered: a report is issued with '
                             'Project.issue_report and amended with '
                             'Project.amend_report')
        if key == 'full_only':
            if not isinstance(given, bool):
                raise TypeError(f'full_only is True or False, not '
                                f'{type(given).__name__}')
            changes[key] = given
            continue
        if key not in BY_KEY:
            raise ValueError(f'unknown identification field {key!r}; the '
                             f'fields are: {", ".join([*BY_KEY, "full_only"])}')
        if key == 'authorized':
            changes[key] = _people(given)
            continue
        if not isinstance(given, str):
            raise TypeError(f'{BY_KEY[key].label} is a str, not '
                            f'{type(given).__name__}')
        if issued and key in ('number', 'revision') \
                and given.strip() != str(held.get(key, '')):
            raise ValueError(
                f'{BY_KEY[key].label} cannot change: the report was issued '
                f'as {_name(held)}, and a change after issue is an '
                'amendment (Project.amend_report)')
        changes[key] = given.strip()
    for key, given in changes.items():
        # a blank field, and the statement left on (its default), are
        # not stored, so a report with nothing entered stores nothing
        if given is True or (given in ('', []) and key != 'full_only'):
            held.pop(key, None)
        else:
            held[key] = given
    report.identity = held


def _name(held: Mapping[str, Any]) -> str:
    number = str(held.get('number', '')) or f'(number {NOT_ENTERED})'
    revision = str(held.get('revision', '')) or f'({NOT_ENTERED})'
    return f'Report {number}, Revision {revision}'


def name(report: Any) -> str:
    """'Report R-0037, Revision 1': the identity every page carries,
    with '(number not entered)' in place of what is missing."""
    return _name(_identity(report))


def missing(report: Any) -> list[str]:
    """The labels of the fields the identification needs that are not
    entered, in order; empty when it is complete."""
    held = _identity(report)
    return [field.label for field in FIELDS
            if field.required and not held.get(field.key)]


def amendment(report: Any) -> int:
    """Which amendment the report is (1 for the first), or 0 when it is
    not an amendment. Counted from the issue record, never entered."""
    held = _identity(report)
    if not held.get('amends'):
        return 0
    issued = sum(1 for entry in held.get('record', [])
                 if entry.get('status') == 'amended')
    return issued + (0 if held.get('status') == 'amended' else 1)


def end_mark(report: Any) -> str:
    """The report's last line: 'End of amendment.' for an amendment,
    otherwise the skeleton's 'End of report.'."""
    from .skeleton import END

    return END_AMENDMENT if _identity(report).get('amends') else END


def content_hash(page: str) -> str:
    """The SHA-256 (hex) of a rendered page: what is compared to tell
    the issued content from a changed one."""
    return hashlib.sha256(page.encode('utf-8')).hexdigest()


def _issued_entry(held: Mapping[str, Any]) -> dict[str, Any] | None:
    """The record entry of the revision the report now carries, when it
    has been issued."""
    if held.get('status', 'draft') == 'draft':
        return None
    for entry in reversed(held.get('record', [])):
        if (entry.get('number') == held.get('number')
                and entry.get('revision') == held.get('revision')):
            return dict(entry)
    return None


def guard(report: Any, page: str) -> None:
    """Refuse to write `page` (the report rendered for export) when the
    report has been issued and the page differs from what was issued.

    A report that is a draft, or the feature switched off, passes: a
    draft says so on every page. An issued report whose content is
    unchanged passes too, since that is a copy of the issued document.

    Raises
    ------
    ValueError
        Naming the issued revision and how to make the change.
    """
    from .features import enabled

    if FEATURE not in enabled(report):
        return
    held = _identity(report)
    entry = _issued_entry(held)
    if entry is None:
        return
    if entry.get('content') != content_hash(page):
        raise ValueError(
            f'{_name(held)} was issued on {entry.get("issued") or "its date"} '
            'and has changed since; an issued report is not written again '
            'under its identity. A change after issue is an amendment: '
            'Project.amend_report, then Project.issue_report.')


def issue_ready(report: Any, objects: Mapping[str, Any] | None = None,
                object_groups: Sequence[Mapping[str, Any]] | None = None
                ) -> str:
    """Check that `report` can be issued now, and return the status it
    will carry once issued: 'issued', or 'amended' for an amendment.

    Refused: the feature not enabled; a report already issued (a change
    is an amendment); a field the identification needs not entered;
    and, with skeleton sections on and `objects` given, a section still
    owed (section 9 read off the findings register while that feature
    is on, as `Project.report_check` reads it). With the flash report
    on, what that page still owes (`report.flash.check`) is read in
    place of the skeleton's sections, which it does not draw.

    Raises
    ------
    ValueError
        For a refusal; the message names what stands in the way.
    """
    from .features import enabled

    title = getattr(report, 'title', '')
    if FEATURE not in enabled(report):
        raise ValueError(f'report {title!r} does not have Identity and issue '
                         'enabled, so it has no identity to issue under')
    held = _identity(report)
    if held.get('status', 'draft') != 'draft':
        raise ValueError(f'{_name(held)} is already issued; a change after '
                         'issue is an amendment (Project.amend_report)')
    gaps = missing(report)
    if gaps:
        raise ValueError(f'{_name(held)} cannot be issued: '
                         f'{", ".join(gaps)} {"is" if len(gaps) == 1 else "are"}'
                         f' {NOT_ENTERED}')
    from .flash import FEATURE as FLASH
    from .skeleton import FEATURE as SKELETON

    if objects is not None and FLASH in enabled(report):
        # a flash report is its one page: what it owes is the page's,
        # not the skeleton's sections, which it does not draw
        from .flash import check as flash_check

        owed = flash_check(report, objects, object_groups)
        if owed:
            raise ValueError(f'{_name(held)} cannot be issued while the flash '
                             f'report owes {len(owed)} '
                             f'part{"s" * (len(owed) != 1)}: '
                             + '; '.join(owed))
    elif objects is not None and SKELETON in enabled(report):
        from .findings import check as register_check
        from .skeleton import check

        owed = register_check(report, check(report, objects, object_groups))
        if owed:
            raise ValueError(f'{_name(held)} cannot be issued while it owes '
                             f'{len(owed)} section{"s" * (len(owed) != 1)}: '
                             + '; '.join(map(str, owed)))
    return 'amended' if held.get('amends') else 'issued'


def mark_issued(report: Any, status: str) -> dict[str, Any]:
    """Set the status `issue_ready` returned, before the issued page is
    rendered (the page says it is issued). Returns the identity as it
    was, for `unmark` if writing the document fails."""
    if status not in ('issued', 'amended'):
        raise ValueError(f'a report is issued as issued or amended, not '
                         f'{status!r}')
    before = _identity(report)
    held = dict(before)
    held['status'] = status
    report.identity = held
    return before


def unmark(report: Any, before: Mapping[str, Any]) -> None:
    """Put back the identity `mark_issued` returned."""
    report.identity = dict(before)


def record_issue(report: Any, page: str, pdf: bytes, file: str = '') -> dict:
    """Add the revision just issued to the issue record: its number,
    revision, status and date of issue as entered, the SHA-256 of the
    PDF and of the page it was printed from, and the file written.
    Returns the entry."""
    held = _identity(report)
    entry = {'number': held.get('number', ''),
             'revision': held.get('revision', ''),
             'status': held.get('status', 'draft'),
             'issued': held.get('issued', ''),
             'sha256': hashlib.sha256(pdf).hexdigest(),
             'content': content_hash(page),
             'file': str(file)}
    if held.get('amends'):
        entry['amends'] = dict(held['amends'])
        entry['changes'] = [dict(change) for change in held.get('changes', [])]
    held['record'] = [*held.get('record', []), entry]
    report.identity = held
    return dict(entry)


def _changes(changes: Any) -> list[dict[str, str]]:
    if isinstance(changes, (Mapping, str)) \
            or not isinstance(changes, Iterable):
        raise TypeError('changes is a list of changes, each a mapping of '
                        f'{", ".join(CHANGE_PARTS)}')
    out = []
    for number, change in enumerate(changes, start=1):
        if not isinstance(change, Mapping):
            raise TypeError(f'change {number} is a mapping of '
                            f'{", ".join(CHANGE_PARTS)}, not '
                            f'{type(change).__name__}')
        unknown = sorted(set(change) - set(CHANGE_PARTS))
        if unknown:
            raise ValueError(f'change {number}: unknown part'
                             f'{"s" * (len(unknown) != 1)} '
                             f'{", ".join(map(repr, unknown))}; the parts '
                             f'are: {", ".join(CHANGE_PARTS)}')
        clean = {}
        for part in CHANGE_PARTS:
            given = change.get(part, '')
            if not isinstance(given, str):
                raise TypeError(f'change {number}: {part} is a str, not '
                                f'{type(given).__name__}')
            clean[part] = given.strip()
        blank = [part for part in _CHANGE_REQUIRED if not clean[part]]
        if blank:
            raise ValueError(f'change {number}: {", ".join(blank)} '
                             f'{"is" if len(blank) == 1 else "are"} '
                             f'{NOT_ENTERED}')
        out.append(clean)
    if not out:
        raise ValueError('an amendment records at least one change')
    return out


def amend(report: Any, changes: Any, *, revision: str,
          number: str | None = None) -> None:
    """Make the issued `report` an amendment, to be issued in its turn.

    `changes` lists what the amendment changes, each a mapping of
    `where` (the page or section), `was` (what it said), `now` (what it
    says) and, where appropriate, `reason`. `revision` is the
    amendment's own revision, entered by the author; `number`, when
    given, its own report number (otherwise it keeps the original's).
    The amendment refers to the revision it amends by number, revision
    and date of issue, taken from the issue record. Its date of issue
    is cleared, for the author to enter when it is issued.

    Refused before anything changes: a report not issued (a draft is
    changed in place); a change missing its where, was or now; no
    changes; and a number and revision already issued.

    Raises
    ------
    ValueError
        For a refusal; the message names the reason.
    TypeError
        When a value is of the wrong type.
    """
    from .features import enabled

    held = _identity(report)
    title = getattr(report, 'title', '')
    if FEATURE not in enabled(report):
        raise ValueError(f'report {title!r} does not have Identity and issue '
                         'enabled, so it has nothing issued to amend')
    entry = _issued_entry(held)
    if entry is None:
        raise ValueError(f'{_name(held)} has not been issued; a draft is '
                         'changed in place and needs no amendment')
    clean = _changes(changes)
    if not isinstance(revision, str):
        raise TypeError(f'revision is a str, not {type(revision).__name__}')
    if number is not None and not isinstance(number, str):
        raise TypeError(f'number is a str, not {type(number).__name__}')
    revision = revision.strip()
    new_number = held.get('number', '') if number is None else number.strip()
    if not revision:
        raise ValueError(f'the amendment needs its own revision; the '
                         f'revision is {NOT_ENTERED}')
    if not new_number:
        raise ValueError(f'the amendment needs a report number; the number '
                         f'is {NOT_ENTERED}')
    taken = {(old.get('number'), old.get('revision'))
             for old in held.get('record', [])}
    if (new_number, revision) in taken:
        raise ValueError(f'Report {new_number}, Revision {revision} has '
                         'already been issued; an amendment carries an '
                         'identity of its own')
    held.update(number=new_number, revision=revision, status='draft',
                changes=clean,
                amends={'number': entry.get('number', ''),
                        'revision': entry.get('revision', ''),
                        'issued': entry.get('issued', '')})
    held.pop('issued', None)
    report.identity = held


# ---- the page --------------------------------------------------------------


def _text(given: Any) -> str:
    """Escaped, or the caution 'not entered' when blank."""
    if given:
        return html.escape(str(given))
    return f'<span class="missing">{NOT_ENTERED}</span>'


def status_words(report: Any) -> str:
    """The status as the page says it: 'Draft', 'Draft amendment',
    'Issued' or 'Amendment 1'. A flash report (the report feature
    `flash_report`) is preliminary whatever its status, and says so
    first: 'Preliminary (draft)', 'Preliminary (issued)'."""
    from .features import enabled

    held = _identity(report)
    status = held.get('status', 'draft')
    if status == 'issued':
        words = 'Issued'
    elif status == 'amended':
        words = f'Amendment {amendment(report)}'
    else:
        words = 'Draft amendment' if held.get('amends') else 'Draft'
    if 'flash_report' in enabled(report):
        return f'Preliminary ({words.lower()})'
    return words


def block_html(report: Any) -> str:
    """The identification block for the first page, as HTML.

    The identity rows, then for an amendment the revision it amends and
    its table of changes, then each person authorizing the report with
    a blank line for the signature and the date, and the statement that
    the report is not reproduced except in full. Joined with no white
    space between tags, since the page keeps a text block's white
    space.
    """
    held = _identity(report)
    rows = [('Status', html.escape(status_words(report)))]
    for field in FIELDS:
        if field.key == 'authorized':
            continue
        given = held.get(field.key, '')
        if not field.required and not given:
            continue
        rows.append((field.label, _text(given)))
    parts = ['<div class="identity">',
             '<table class="identity"><tbody>']
    parts.extend(f'<tr><th>{html.escape(label)}</th><td>{cell}</td></tr>'
                 for label, cell in rows)
    parts.append('</tbody></table>')
    amends = held.get('amends')
    if amends:
        parts.append(
            f'<p class="amends">Amendment {amendment(report)} to Report '
            f'{_text(amends.get("number"))}, Revision '
            f'{_text(amends.get("revision"))}, issued '
            f'{_text(amends.get("issued"))}. The report it amends is not '
            'altered and is kept with this amendment.</p>')
        parts.append('<table class="changes"><thead><tr><th>Where</th>'
                     '<th>Was</th><th>Now</th><th>Reason</th></tr></thead>'
                     '<tbody>')
        for change in held.get('changes', []):
            parts.append('<tr>' + ''.join(
                f'<td>{html.escape(change.get(part, ""))}</td>'
                for part in CHANGE_PARTS) + '</tr>')
        parts.append('</tbody></table>')
    people = held.get('authorized', [])
    parts.append('<div class="authorized"><p class="label">Authorized by</p>')
    if people:
        for person in people:
            parts.append(f'<div class="signature"><span class="who">'
                         f'{html.escape(person)}</span><span class="line">'
                         'Signature</span><span class="line date">Date'
                         '</span></div>')
    else:
        parts.append(f'<p>{_text("")}</p>')
    parts.append('</div>')
    if value(report, 'full_only'):
        parts.append(f'<p class="fullonly">{html.escape(FULL_ONLY)}</p>')
    parts.append('</div>')
    return ''.join(parts)


def _css_string(given: str) -> str:
    """`given` as a quoted CSS string: quotes, backslashes, line breaks
    and '<' (which could close the style element) escaped."""
    out = []
    for char in given:
        if char in '"\\':
            out.append('\\' + char)
        elif char == '<' or ord(char) < 0x20 or ord(char) == 0x7f:
            out.append(f'\\{ord(char):x} ')
        else:
            out.append(char)
    return '"' + ''.join(out) + '"'


def page_css(report: Any) -> str:
    """The printed page's margins for `report`: the laboratory and the
    report's number and revision at the top of every page, its title and
    page n of N at the foot. The numbers are the print engine's own page
    counters, so they are true of whatever it prints. A draft says so
    beside its number. The classification marking, when there is one,
    moves from the page's fixed strips into the margins, top and foot
    center, where it is on every page and over nothing."""
    held = _identity(report)
    laboratory = (str(held.get('laboratory', ''))
                  or f'(laboratory {NOT_ENTERED})')
    identity = _name(held)
    status = status_words(report)
    if status != 'Issued':
        identity += f', {status}'
    title = str(getattr(report, 'title', ''))
    small = 'font-size: 8pt; color: #1f2328;'
    rules = [f'  @top-left {{ content: {_css_string(laboratory)}; {small} }}',
             f'  @top-right {{ content: {_css_string(identity)}; {small} }}',
             f'  @bottom-left {{ content: {_css_string(title)}; {small} }}',
             ('  @bottom-right { content: "Page " counter(page) " of " '
              f'counter(pages); {small} }}')]
    marking = str(getattr(report, 'marking', '')).strip()
    if marking:
        color = ('#d02b2b' if getattr(report, 'marking_color', 'ink') == 'red'
                 else '#1f2328')
        mark = (f'content: {_css_string(marking)}; font-size: 10pt; '
                f'font-weight: 700; letter-spacing: .12em; color: {color};')
        rules += [f'  @top-center {{ {mark} }}',
                  f'  @bottom-center {{ {mark} }}']
    top, side = MARGINS
    return (f'\n@page {{ margin: {top}in {side}in;\n' + '\n'.join(rules)
            + ' }\n')
