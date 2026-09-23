"""Theory carries its citations.

The docstrings **are** the API reference, so a reader who wants to
check an equation against the source it came from has nowhere else to
look — and an implementer coming after has no other way to know which
of several formulations was chosen (Brandon, 2026-09-22: "how did you
know how to compute the SRS, or different FRF types, or a wavelet?").

Two rules, and the second is about rendering rather than scholarship:
this site renders Markdown, so a reST `.. [1]` citation marker comes
out literally and the whole list runs together in one paragraph.
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


@pytest.mark.parametrize('name', sorted(THEORY))
def test_a_module_implementing_theory_cites_it(name):
    doc = _module(name).__doc__ or ''
    assert 'References\n----------' in doc, (
        f'{name} implements {THEORY[name]} and cites nothing'
    )
    entries = doc.split('References\n----------', 1)[1]
    assert entries.strip().startswith('1.'), (
        f'{name}: references are a numbered list, not prose'
    )
    assert '(19' in entries or '(20' in entries or 'ISO ' in entries \
        or 'ANSI' in entries or 'IEC ' in entries, (
        f'{name}: an entry names no year and no standard'
    )


@pytest.mark.parametrize('name,cls,method', THEORY_METHODS)
def test_a_method_implementing_theory_cites_it(name, cls, method):
    doc = getattr(getattr(_module(name), cls), method).__doc__ or ''
    assert 'References\n----------' in doc, (
        f'{name}.{cls}.{method} cites nothing'
    )


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
            if '.. [' in (doc or ''):
                offenders.append(label)
    for name in sorted(THEORY):
        doc = _module(name).__doc__ or ''
        if '.. [' in doc:
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
