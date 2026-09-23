"""Theory carries its citations.

The docstrings **are** the API reference, so a reader who wants to
check an equation against the source it came from has nowhere else to
look — and an implementer coming after has no other way to know which
of several formulations was chosen (Brandon, 2026-09-22: "how did you
know how to compute the SRS, or different FRF types, or a wavelet?").

Two rules, and the second is about rendering rather than scholarship:
this site renders Markdown, so a reST `.. [1]` citation marker comes
out literally and the whole list runs together in one paragraph.

Every docstring here is read through `inspect.cleandoc`, because
**Python 3.13 strips a docstring's leading whitespace at compile time
and 3.12 does not** — this project supports both. A test matching an
unindented `References` heading passes on one and fails on the other,
which is exactly what happened the first time (2026-09-22): green on
this desk, red on the 3.12 matrix job.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

#: modules whose docstring must cite the theory it implements, with
#: what each one is expected to answer for
THEORY = {
    'core.srs': 'the ramp-invariant shock response spectrum filter',
    'core.octave': 'the proportional band system',
    'core.wavelet': 'the Morlet transform',
    'core.averaging': 'the averaged periodogram and its windows',
    'core.kurtosis': 'the fourth standardized moment',
    'core.correlate': 'the modal assurance criterion',
    'core.fem': 'the beam element and its eigensolution',
    'core.modal_fit': 'the single-degree-of-freedom residue fit',
    'core.filters': 'filtering, integration and the drift it invents',
}

#: and methods, where the theory lives on the call rather than the module
THEORY_METHODS = (
    ('core.data', 'TimeHistory', 'compute_frfs'),
    ('core.data', 'TimeHistory', 'compute_multiple_coherence'),
)


def _module(name):
    return importlib.import_module(f'visualdynamics.{name}')


def _doc(obj) -> str:
    """A docstring with its indentation normalized away.

    3.13 dedents at compile time, 3.12 leaves the indentation in; both
    are supported, so neither form may be matched directly.
    """
    return inspect.cleandoc(obj.__doc__ or '')


@pytest.mark.parametrize('name', sorted(THEORY))
def test_a_module_implementing_theory_cites_it(name):
    doc = _doc(_module(name))
    assert 'References\n----------' in doc, (
        f'{name} implements {THEORY[name]} and cites nothing'
    )
    entries = doc.split('References\n----------', 1)[1]
    assert '\n1. ' in entries, (
        f'{name}: references are a numbered list, not prose'
    )
    assert '(19' in entries or '(20' in entries or 'ISO ' in entries \
        or 'ANSI' in entries or 'IEC ' in entries, (
        f'{name}: an entry names no year and no standard'
    )


@pytest.mark.parametrize('name,cls,method', THEORY_METHODS)
def test_a_method_implementing_theory_cites_it(name, cls, method):
    doc = _doc(getattr(getattr(_module(name), cls), method))
    assert 'References\n----------' in doc, (
        f'{name}.{cls}.{method} cites nothing'
    )


def _reference_sections():
    """(label, text) for every References section in the package."""
    out = []
    for name in sorted(THEORY):
        doc = _doc(_module(name))
        if 'References\n----------' in doc:
            out.append((name, doc.split('References\n----------', 1)[1]))
    for name, cls, method in THEORY_METHODS:
        doc = _doc(getattr(getattr(_module(name), cls), method))
        if 'References\n----------' in doc:
            out.append((f'{name}.{cls}.{method}',
                        doc.split('References\n----------', 1)[1]))
    return out


def test_no_citation_claims_a_chapter():
    """False precision is worse than none.

    A chapter or section number in a citation looks checkable, and
    every one of these was written from knowledge of the method rather
    than from the book in hand — so none of them could be checked
    (Brandon, 2026-09-23: "are those citations where you actually got
    the equations from?"). They came out. What is left is author,
    year, title and venue, which is true.
    """
    offenders = [label for label, text in _reference_sections()
                 if 'Chapter' in text or 'Section ' in text
                 or 'pp.' in text or 'p.' in text]
    assert not offenders, (
        'a citation claims a chapter or page nobody here has checked: '
        + ', '.join(offenders)
    )


def test_a_page_range_comes_only_with_a_stable_identifier():
    """Volume, issue and pages are quoted only where a DOI link is
    quoted beside them, because that is where they can be followed and
    found wrong. Everywhere else the entry stops at the venue."""
    import re

    for label, text in _reference_sections():
        for entry in re.split(r'\n(?=\d+\. )', text):
            paged = re.search(r'\d+\(\d+\),\s*\d+-\d+', entry)
            if paged:
                assert 'doi.org/' in entry, (
                    f'{label}: "{paged.group(0)}" is quoted with no '
                    'identifier to check it against'
                )


def test_every_reference_section_says_what_its_first_entry_is():
    """The honest distinction: a section either names the formulation
    the code follows, or says the method is standard and these are
    where it is set out. Both start with a sentence, not a list."""
    for label, text in _reference_sections():
        lead = text.strip().split('\n\n', 1)[0].strip()
        assert lead and not lead.startswith('1.'), (
            f'{label}: the list begins with no word on what it is'
        )
        assert lead.endswith('.'), f'{label}: the lead is not a sentence'


def test_no_docstring_uses_a_citation_marker_this_site_cannot_render():
    """The rendering rule. A `.. [1]` marker is the numpydoc
    convention and the right thing under Sphinx; under mkdocstrings
    rendering Markdown it appears verbatim and the entries collapse
    into one paragraph. Measured both ways before choosing
    (2026-09-22)."""
    import visualdynamics

    offenders = []
    seen = set()
    for name in dir(visualdynamics):
        obj = getattr(visualdynamics, name)
        if not (inspect.ismodule(obj) or inspect.isclass(obj)
                or inspect.isfunction(obj)):
            continue
        for label, doc in _docs_of(obj, seen):
            if '.. [' in inspect.cleandoc(doc or ''):
                offenders.append(label)
    for name in sorted(THEORY):
        if '.. [' in _doc(_module(name)):
            offenders.append(name)
    assert not offenders, (
        'reST citation markers render literally on this site: '
        + ', '.join(sorted(set(offenders)))
    )


def _docs_of(obj, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield getattr(obj, '__name__', repr(obj)), obj.__doc__
    if inspect.isclass(obj):
        for name, member in vars(obj).items():
            if name.startswith('_') or not callable(member):
                continue
            yield f'{obj.__name__}.{name}', getattr(member, '__doc__', '')


def test_the_check_reads_both_interpreters_docstrings():
    """The version difference, covered on whichever version is running.

    Python 3.13 strips a docstring's leading whitespace at compile
    time; 3.12 hands it over indented. This project supports both, and
    only one of them runs here — so the two forms are written out
    literally and the reading is held to answering the same on each.
    A bare substring match answers differently, which is the bug this
    replaced (green on 3.13, red on the 3.12 matrix job, 2026-09-22).
    """
    indented = ('One line.\n\n        References\n        ----------\n'
                '        1. Thing (1981).\n        ')
    dedented = 'One line.\n\nReferences\n----------\n1. Thing (1981).\n'

    class Carrier:
        pass

    for form in (indented, dedented):
        Carrier.__doc__ = form
        assert 'References\n----------' in _doc(Carrier), (
            'the reading finds the section in either interpreter'
        )
    assert 'References\n----------' not in indented, (
        'and the bare match really does miss the 3.12 form'
    )
