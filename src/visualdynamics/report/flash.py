"""The flash report: the one preliminary page a customer leaves the test
with (report feature `flash_report`).

The full report takes days, because the data are reviewed before it
is written; the program cannot wait that long to know where it stands.
With the feature on, the report is drawn as one page built from what
the project already holds, in the order its reader asks:

- PRELIMINARY across the top, with the sentence that the results are
  subject to data review and that the test report supersedes the page;
- the identification (`report.identity`), its status Preliminary on
  the page and in every printed margin, with the people authorizing it
  and a blank line for each signature;
- the headline: each requirement read against its run so far
  (`report.requirement.read`), one line per run, with how many of its
  criteria were met against which document and the criteria not met
  and not judged, by name. With the feature `run_sections` on, each
  line is labeled by its run's axis, run and level (`report.runs`) and
  the lines follow the order the runs were run. The lines are results
  as read on the day, never a statement of conformity, and the page
  says so;
- as run against the plan: the deviations of the findings register
  (`report.findings`) by number, or "No deviations." where the author
  set them to None;
- what is open: the anomalies and nonconformances by number, each with
  its status, its fact and its disposition (who owns it), then the
  condition of the article on release;
- the opinions given at the console, each marked as an opinion and
  attributed;
- what happens next, with the date the test report is due.

With a run log brought into the project (`Project.attach_run_log`),
the page gains a short part of its own, From the Run Log, before what
happens next, kept to what the page does not already say: for each
extract, which log, test and entry it holds (with its badge when the
log has newer entries), the runs so far with how each ended, the holds
open, the anomalies open that the findings register does not hold,
with their owners, the opinions given at the console that the page
does not record in the same words, and the checklists with lines still
owed. A long list stops at `LISTED` and says how many more the log
holds. Each is read from the extract as it was taken (only the badge
looks at the live file), and a kind the log holds none of is said to be
held none of by the log at its entry, never to have been none. The
page's own parts stay the author's: the opinions the page records, the
condition on release and what happens next are still typed, and still
owed until they are. So a day with a run log still prints on one
sheet, the part is set small and the page a little tighter while it is
on it (`LOG_CSS`); a page without a run log keeps its own style.

Nothing on the page is invented. The condition on release, the
opinions, what happens next and the report's date are typed by the
author (`enter`) and kept with the report's identity, so a report
template, which carries only the laboratory's fields, never passes one
test's page to the next. Anything the project does not hold is drawn as
owed (NOT YET WRITTEN, with what belongs there) and listed by `check`,
which the export asks about and which stands in the way of issuing the
page. The decision to send it stays the author's.

The feature needs Requirement and results, Identity and issue and the
Findings register; the register lives in the skeleton's deviations
section, so the skeleton is on too, but its sections are not drawn: the
flash report is the page.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from ..core import findings as cf
from . import findings as fr
from . import identity as ri
from . import skeleton

# P29 of the report-features plan (Kevin Cross, 2026-10-24): a flash
# report starter drafted from the day's analysis, built from the
# identity (P19), the result table (P17) and the register (P21), with
# nothing computed of its own.

#: the feature's key in `report.features`
FEATURE = 'flash_report'

#: the word across the top of the page
PRELIMINARY = 'PRELIMINARY'

#: the sentence beside it
SUPERSEDED = ('Results are subject to data review; the test report '
              'supersedes this page.')

#: what the headline's lines are, and are not
NOT_CONFORMITY = ('These are the results as read on the day, before the data '
                  'review; they are not a statement of conformity.')

#: the fields the author types, in the order the page draws them, with
#: the label each is listed by while it is owed, and what belongs there
TYPED: tuple[tuple[str, str, str], ...] = (
    ('condition', 'Condition on release',
     ('the post-test functional and inspection results as far as they go, '
      "the article's condition when it was released, and who received "
      'it.')),
    ('next_steps', 'What happens next',
     ('what is still to be done before the result is final: the data '
      "review, the analyst's comparison with the predictions and the open "
      "findings' dispositions.")),
    ('report_date', 'Report date',
     'the date the test report will be issued.'),
)

#: the typed fields by key
LABELS = {key: label for key, label, _what in TYPED}

#: what belongs in each typed field, by key
WHAT = {key: what for key, _label, what in TYPED}

#: where the identity keeps the page's typed fields
HELD = 'flash'

#: the opinions' owed line, and the page's line when the author set them
#: to None
OPINIONS_OWED = ('the opinions given at the console, each with who gave it, '
                 'or None.')
NO_OPINIONS = 'No opinions were given.'

#: the heading over what the run log holds
FROM_LOG = 'From the Run Log'

#: the most lines a part of the run log's block lists before it says
#: how many more the log holds
LISTED = 3

#: the headline's owed line
NO_HEADLINE = ('no requirement is read against a run the project holds; '
               'bind one with Requirement and results, and the headline is '
               'read off its result table.')


def _require(report: Any) -> None:
    from .features import enabled

    if FEATURE not in enabled(report):
        raise ValueError(f'report {getattr(report, "title", "")!r} does not '
                         'have the Flash report enabled')


def held(report: Any) -> dict[str, Any]:
    """What the author has typed for the page: each of `TYPED` that is
    entered, 'opinions' (a list of {'by', 'text'}) and 'opinions_none'
    (True when the author said none were given)."""
    return dict(ri._identity(report).get(HELD) or {})


def _opinion(given: Any, number: int) -> dict[str, str]:
    if isinstance(given, Mapping):
        unknown = sorted(set(given) - {'by', 'text'})
        if unknown:
            raise ValueError(f'opinion {number}: unknown part'
                             f'{"s" * (len(unknown) != 1)} '
                             f'{", ".join(map(repr, unknown))}; an opinion '
                             'is by and text')
        by, text = given.get('by', ''), given.get('text', '')
    elif isinstance(given, (tuple, list)) and len(given) == 2:
        by, text = given
    else:
        raise TypeError(f'opinion {number} is (by, text) or a mapping of by '
                        f'and text, not {type(given).__name__}')
    if not isinstance(by, str) or not isinstance(text, str):
        raise TypeError(f'opinion {number}: by and text are each a str')
    by, text = by.strip(), text.strip()
    if not text:
        raise ValueError(f'opinion {number}: the opinion is {ri.NOT_ENTERED}')
    if not by:
        raise ValueError(f'opinion {number}: an opinion is recorded with who '
                         'gave it; by is ' + ri.NOT_ENTERED)
    return {'by': by, 'text': text}


def enter(report: Any, *, condition: str | None = None,
          next_steps: str | None = None, report_date: str | None = None,
          opinions: Sequence[Any] | None = None,
          opinions_none: bool | None = None) -> None:
    """Type the parts of the flash report only the author can write.

    `condition` is the article's condition on release, `next_steps` what
    happens next, and `report_date` the date the test report is due,
    each a str ('' clears it). `opinions` replaces the opinions given at
    the console, each (by, text) or a mapping of `by` and `text`; an
    opinion is recorded with who gave it. `opinions_none` True says no
    opinion was given, False takes that back. A part left as None is
    not changed.

    Refused before anything changes: the feature off; a value of the
    wrong type; an opinion without its words or without who gave it;
    and opinions entered together with None said of them.

    Raises
    ------
    ValueError
        For a refusal; the message names the part and the reason.
    TypeError
        When a value is of the wrong type.
    """
    _require(report)
    page = held(report)
    texts = {'condition': condition, 'next_steps': next_steps,
             'report_date': report_date}
    for key, given in texts.items():
        if given is not None and not isinstance(given, str):
            raise TypeError(f'{LABELS[key]} is a str, not '
                            f'{type(given).__name__}')
    if opinions_none is not None and not isinstance(opinions_none, bool):
        raise TypeError(f'opinions_none is True or False, not '
                        f'{type(opinions_none).__name__}')
    if opinions is not None:
        if isinstance(opinions, (str, Mapping)):
            raise TypeError('opinions is a list of opinions, each (by, text)')
        cleaned = [_opinion(given, number)
                   for number, given in enumerate(opinions, start=1)]
    else:
        cleaned = page.get('opinions', [])
    none = (opinions_none if opinions_none is not None
            else bool(page.get('opinions_none')))
    if none and cleaned:
        raise ValueError('opinions are entered and also said to be none; '
                         'clear the opinions or take back the None')
    for key, given in texts.items():
        if given is None:
            continue
        if given.strip():
            page[key] = given.strip()
        else:
            page.pop(key, None)
    if cleaned:
        page['opinions'] = cleaned
    else:
        page.pop('opinions', None)
    if none:
        page['opinions_none'] = True
    else:
        page.pop('opinions_none', None)
    identity = ri._identity(report)
    if page:
        identity[HELD] = page
    else:
        identity.pop(HELD, None)
    report.identity = identity


# ---- what the page reads ----------------------------------------------------


def headline(report: Any, objects: Mapping[str, Any],
             object_groups: Sequence[Mapping[str, Any]] | None = None
             ) -> list[str]:
    """The headline's lines, plain text: one per requirement of the
    report read against its run, in the order the runs were run with
    the feature `run_sections` on (each labeled by its run identity),
    else in the report's order (each labeled by the record it reads).
    A requirement whose specification or run is not in the project is
    left out, as the page leaves out a block that does not resolve; one
    that cannot be read says nothing was judged, and why. Empty when no
    requirement is read."""
    from .features import enabled
    from .requirement import read, roles_on

    runs_on = 'run_sections' in enabled(report)
    entries: list[tuple[str | None, str, str]] = []
    for block in report.blocks:
        if 'requirement' not in block:
            continue
        name = _measured(block, objects, object_groups)
        if name is None:
            continue
        try:
            found = read(block, objects, object_groups,
                         roles=roles_on(report))
        except (TypeError, ValueError) as refusal:
            said = f'nothing judged; {refusal}'
        else:
            if found is None:
                continue
            said = _result(found)
        root = _root(objects, name) if runs_on else None
        entries.append((root, name, said))
    if not runs_on:
        return [f'{name}: {said}.' for _root_name, name, said in entries]
    return _by_run(entries, objects, object_groups)


def _measured(block: Mapping[str, Any], objects: Mapping[str, Any],
              object_groups: Sequence[Mapping[str, Any]] | None
              ) -> str | None:
    from ..core.report import resolve_binding

    names = [resolve_binding(block.get(key), objects, object_groups)
             for key in ('source', 'measured')]
    if any(name not in objects for name in names):
        return None
    return str(names[1])


def _root(objects: Mapping[str, Any], name: str) -> str | None:
    from .runs import root_of

    return root_of(objects, name)


def _result(found: Any) -> str:
    """One requirement's reading in words: the count met against the
    document, the scale it was read at, and the criteria not met and not
    judged, as the pass/fail box names them."""
    count = len(found.rows)
    said = (f'{found.met} of {count} criteri{"on" if count == 1 else "a"} '
            f'met against {found.requirement.document_text}')
    if found.scale_db:
        said += f', compared at {found.scale_db:+g} dB'
    for label, state in (('not met', 'not met'),
                         ('not judged', 'not judged')):
        named = [f'{row.name}, {row.required}' for row in found.rows
                 if row.state == state]
        if named:
            said += f'; {label}: ' + '; '.join(named)
    return said


def _by_run(entries: Sequence[tuple[str | None, str, str]],
            objects: Mapping[str, Any],
            object_groups: Sequence[Mapping[str, Any]] | None
            ) -> list[str]:
    """The lines of `entries` (run, record, result) in the order their
    runs were run, each led by its run's heading; a record of no run
    keeps its own name and comes first, as a section of no run does in
    the grouped Results."""
    from .runs import _groups, heading, run_order

    groups = _groups(objects, object_groups)
    roots = [root for root, _name, _said in entries if root is not None]
    ordered, found = run_order(objects, roots, groups)
    lines = [f'{name}: {said}.' for root, name, said in entries
             if root is None]
    for root in ordered:
        label = heading(root, found[root])
        for mine, _name, said in entries:
            if mine == root:
                lines.append(f'{label}: {said}.')
    return lines


def check(report: Any, objects: Mapping[str, Any],
          object_groups: Sequence[Mapping[str, Any]] | None = None
          ) -> list[str]:
    """What the flash report still owes before it is issued, one line
    each, in the page's order: the identification's fields not entered,
    the headline when no requirement is read, each kind of finding
    neither entered nor set to None, each finding's fact and its
    approval or disposition not written, the typed parts not written,
    and the opinions neither recorded nor set to None. Empty when the
    page owes nothing.

    Raises `ValueError` when the feature is off: there is then no flash
    report to check, and an empty list would read as one complete.
    """
    _require(report)
    owed = []
    gaps = ri.missing(report)
    if gaps:
        owed.append(f'Identification: {", ".join(gaps)} {ri.NOT_ENTERED}')
    if not headline(report, objects, object_groups):
        owed.append('Headline: no requirement is read against a run')
    entries = fr.register(report)
    said = cf.declared(skeleton.words(report, fr.SECTION))
    for kind in cf.KINDS:
        mine = [entry for entry in entries if entry['kind'] == kind]
        if not mine and kind not in said:
            owed.append(f'{cf.PLURAL[kind]}: neither entered nor set '
                        'to None')
        for entry in mine:
            missing = []
            if not entry['fact']:
                missing.append('fact')
            if not entry['approval']:
                missing.append(cf.FOURTH[entry['kind']].lower())
            if missing:
                owed.append(f'{entry["number"]}: {" and ".join(missing)} not '
                            'yet written')
    page = held(report)
    for key, label, _what in TYPED:
        if not page.get(key):
            owed.append(f'{label}: not yet written')
    if not page.get('opinions') and not page.get('opinions_none'):
        owed.append('Opinions: neither recorded nor set to None')
    return owed


# ---- the page ---------------------------------------------------------------


def _owed(what: str) -> str:
    return (f'<div class="owed"><strong>{skeleton.OWED}</strong> '
            f'{html.escape(what)}</div>')


def _head(title: str) -> str:
    return f'<h2 class="flash">{html.escape(title)}</h2>'


def _finding(entry: Mapping[str, Any]) -> str:
    """One finding in a line: its number, what it concerns, its fact and
    its approval or disposition (who owns it), each blank one owed."""
    where = fr._where(entry)
    fourth = cf.FOURTH[entry['kind']]
    fact = (fr._escape(entry['fact']) if entry['fact'] else fr._owed_part())
    owner = (fr._escape(entry['approval']) if entry['approval']
             else fr._owed_part())
    return (f'<p class="flash-finding"><strong>'
            f'{html.escape(entry["number"])}</strong>'
            + (f' <span class="finding-where">({html.escape(where)})</span>'
               if where else '')
            + f': {fact} <em>{fourth}:</em> {owner}</p>')


def _kinds(kinds: Sequence[str], entries: Sequence[Mapping[str, Any]],
           said: set[str]) -> str:
    out = []
    for kind in kinds:
        mine = [entry for entry in entries if entry['kind'] == kind]
        if mine:
            out.extend(_finding(entry) for entry in mine)
        elif kind in said:
            out.append(f'<p class="finding-none">'
                       f'{html.escape(cf.NONE_SENTENCES[kind])}</p>')
        else:
            out.append(_owed(f'{cf.PLURAL[kind]}: enter each in the '
                             'findings register, or set None.'))
    return ''.join(out)


def _typed(page: Mapping[str, Any], key: str) -> str:
    label = LABELS[key]
    what = WHAT[key]
    given = page.get(key)
    if not given:
        return _owed(f'{label}: {what}')
    return (f'<p class="flash-typed"><strong>{html.escape(label)}:</strong> '
            f'{fr._escape(given)}</p>')


def _capped(items: Sequence[str], most: int = LISTED) -> list[str]:
    """`items`, the first `most` of them and then how many more the run
    log holds, so a long day keeps the page to one sheet."""
    if len(items) <= most:
        return list(items)
    return [*items[:most], f'and {len(items) - most} more, in the run log.']


def log_lines(report: Any, extract: Any) -> tuple[list[tuple[str, list[str]]],
                                                  list[str]]:
    """What a run log extract adds to `report`'s flash page, kept to what
    the page does not say already: ([(title, lines)], [kinds held none
    of]).

    The parts are the runs so far with how each ended, the holds open
    (by what each asks), the anomalies open that the findings register
    does not hold under their number, with their owners, the opinions
    given at the console that the page does not record in the same
    words, and the checklists with lines still owed, each with how many.
    A part with more than `LISTED` lines lists that many and says how
    many more the log holds. A part the log holds none of is not drawn;
    its kind is named in the second list, so the page says the log
    holds none by its last entry, never that there was none.
    """
    from ..core.run_log_model import HOLD_RULES
    from .runs import _logged_outcome

    upto = extract.upto
    runs = [f'run {run.number}' + (f' ({run.axis})' if run.axis else '')
            + f' {_logged_outcome(run, upto)}'
            for run in sorted(extract.runs(), key=lambda run: run.entries[0].id)]
    holds = [f'{HOLD_RULES.get(hold.rule, hold.message)} (#{hold.entry.id})'
             for hold in extract.holds()]
    entered = {entry['number'] for entry in fr.register(report)}
    anomalies = [f'{found.number}, owner {found.owner or "not logged"}: '
                 f'{found.fact or "fact not logged"}'
                 for found in extract.open_anomalies()
                 if found.number not in entered]
    typed = {opinion['text'] for opinion in held(report).get('opinions', [])}
    opinions = []
    for entry in extract.entries:
        if entry.kind != 'opinion_given':
            continue
        words = entry.fields.get('opinion') or entry.text or 'not logged'
        if words in typed:
            continue
        to = entry.fields.get('to')
        opinions.append(f'by {entry.author}' + (f' to {to}' if to else '')
                        + f': {words}')
    owed = [f'{found.title.lower()}, {len(found.owed)} of {len(found.lines)}'
            for found in extract.owed()]
    parts = (('Runs so far', runs, 'run'), ('Holds open', holds, 'open hold'),
             ('Anomalies open', anomalies, 'open anomaly'),
             ('Opinions logged', opinions, 'opinion'),
             ('Still owed', owed, 'line still owed'))
    shown = [(title, _capped(lines)) for title, lines, _none in parts if lines]
    return shown, [none for _title, lines, none in parts if not lines]


def _nothing(kinds: Sequence[str], upto: int) -> str:
    """'The log holds no open hold or opinion by entry #8.': what the log
    holds none of, said of the log."""
    said = kinds[0] if len(kinds) == 1 else \
        f'{", ".join(kinds[:-1])} or {kinds[-1]}'
    return f'The log holds no {said} by entry #{upto}.'


def log_html(report: Any, objects: Mapping[str, Any]) -> str:
    """The run log's part of `report`'s flash page, drawn as HTML: for
    each run log extract in `objects`, a line citing its log, test and
    entry (with the extract's badge when the log has newer entries),
    then a line for each part of `log_lines` and one for what the log
    holds none of. '' when `objects` holds no extract, so the page is
    as it was."""
    from ..core.run_log_extract import held as extracts_in

    extracts = extracts_in(objects)
    if not extracts:
        return ''
    out = [_head(FROM_LOG)]
    for _name, extract in extracts:
        shown, none = log_lines(report, extract)
        lines = [f'<strong>{html.escape(title)}:</strong> '
                 f'{html.escape("; ".join(items))}' for title, items in shown]
        if none:
            lines.append(html.escape(_nothing(none, extract.upto)))
        out.append(f'<p class="flash-log-source">'
                   f'{html.escape(f"As recorded in {extract.cited()}.")}</p>'
                   + ''.join(f'<p class="flash-log">{line}</p>'
                             for line in lines))
    return ''.join(out)


#: the run log part's style, added to the page only when the part is
#: drawn: set smaller and tighter than the page's own lines, so a day
#: with a run log still prints on one sheet
LOG_CSS = """
body { font-size: 12px; line-height: 1.2; }
h1 { font-size: 1.15rem; margin: 0 0 .25rem; }
p.flash-preliminary { padding: .2rem .6rem; }
table.identity th, table.identity td { padding: 0 .4rem; font-size: .74rem;
  line-height: 1.15; }
.authorized { margin-top: .2rem; }
.signature .line { margin-top: .45rem; }
h2.flash { font-size: .92rem; margin: .1rem 0 .1rem; }
.flash-line, .flash-note, .flash-finding, .flash-typed, .flash-opinion,
p.finding-none, .authorized p.label { margin: .03rem 0; }
p.endmark { margin-top: .25rem; }
p.flash-log-source, p.flash-log { white-space: normal; margin: .03rem 0;
  font-size: .74rem; line-height: 1.18; }
p.flash-log-source { color: var(--faint); font-style: italic; }
"""


def page_blocks(report: Any, objects: Mapping[str, Any],
                object_groups: Sequence[Mapping[str, Any]] | None = None
                ) -> list[dict[str, Any]]:
    """The flash report as the page's blocks, each a text block already
    drawn as HTML (joined with no white space between tags, since the
    page keeps a text block's white space): the PRELIMINARY line across
    the top, the identification, the headline, the deviations, what is
    open, the opinions, what the run log holds (`log_html`, only with a
    run log extract in the project), and what happens next closed by the
    end mark."""
    entries = fr.register(report)
    said = cf.declared(skeleton.words(report, fr.SECTION))
    page = held(report)
    lines = headline(report, objects, object_groups)
    if lines:
        head = (''.join(f'<p class="flash-line">{html.escape(line)}</p>'
                        for line in lines)
                + f'<p class="flash-note">{html.escape(NOT_CONFORMITY)}</p>')
    else:
        head = _owed(f'Headline: {NO_HEADLINE}')
    opinions = page.get('opinions') or []
    if opinions:
        said_at = ''.join(
            f'<p class="flash-opinion"><em>Opinion, by '
            f'{html.escape(opinion["by"])}:</em> '
            f'{fr._escape(opinion["text"])}</p>' for opinion in opinions)
    elif page.get('opinions_none'):
        said_at = f'<p class="flash-opinion">{html.escape(NO_OPINIONS)}</p>'
    else:
        said_at = _owed(f'Opinions: {OPINIONS_OWED}')
    parts = [
        (f'<p class="flash-preliminary"><strong>{PRELIMINARY}</strong> '
         f'{html.escape(SUPERSEDED)}</p>'),
        ri.block_html(report),
        _head('Headline') + head,
        _head('As Run Against the Plan') + _kinds(('deviation',), entries,
                                                 said),
        (_head('What Is Open')
         + _kinds(('anomaly', 'nonconformance'), entries, said)
         + _typed(page, 'condition')),
        _head('Opinions Recorded as Opinions') + said_at,
        # the end mark closes the last block rather than a block of its
        # own, which would add a block's margin to a page kept to one
        (_head('What Happens Next') + _typed(page, 'next_steps')
         + _typed(page, 'report_date')
         + f'<p class="endmark">{html.escape(ri.end_mark(report))}</p>'),
    ]
    logged = log_html(report, objects)
    if logged:
        parts.insert(-1, logged)
    return [{'kind': 'text', 'html': part} for part in parts]
