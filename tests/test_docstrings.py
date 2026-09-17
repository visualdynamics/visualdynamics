"""Principle 6, checked rather than aspired to.

Two standards, because one size was the wrong answer (Brandon,
2026-08-25). *Every* public callable carries a docstring — that is the
floor, and it is enforced everywhere. The **scripting surface** —
what `visualdynamics.__all__` exports, and the public methods on those
classes — carries NumPy `Parameters` and `Returns` sections as well,
because that is where somebody arrives without context and a table
earns its place.

Prose stays the house style elsewhere on purpose. The docstrings in
this codebase explain *why the obvious approach was rejected*, and
several bugs here were caused by code that looked right; a Parameters
table cannot say that, and mandating one everywhere would trade the
useful half of the documentation for the mechanical half.

`ENFORCED` is a ratchet. Adding a module to it is how the standard
spreads; the list only ever grows.
"""

from __future__ import annotations

import inspect

import pytest

import visualdynamics

#: Where NumPy sections are required today. Grows as they are written;
#: nothing is ever removed, which is what makes it a ratchet.
ENFORCED = ('Project', 'TimeHistory', 'DataArray', 'Psd', 'Frf',
            'Spectrum', 'Srs', 'ShapeSet', 'Geometry', 'ChannelTable',
            'UnitSystem', 'Report', 'Specification', 'ShockSpecification',
            'TransientSpecification')


def public_surface():
    """[(label, object)] every callable a script can reach."""
    out = []
    for name in sorted(visualdynamics.__all__):
        obj = getattr(visualdynamics, name, None)
        if inspect.isclass(obj):
            for attr in sorted(vars(obj)):
                if attr.startswith('_'):
                    continue
                member = vars(obj)[attr]
                if inspect.isfunction(member) or isinstance(member, property):
                    out.append((f'{name}.{attr}', member))
        elif inspect.isfunction(obj):
            out.append((name, obj))
    return out


def docstring_of(obj):
    """The object's own docstring, or None.

    The `fget` branch is for properties, and it is written this way
    for a reason found the hard way: `getattr(getattr(obj, 'fget',
    None), '__doc__', None)` returns *NoneType's* docstring when
    there is no `fget`, which is a non-empty string and always
    truthy — so the test that used it could never fail, and reported
    a clean bill while thirteen public methods had no docstring at
    all (2026-08-25).
    """
    if isinstance(obj, property):
        return obj.fget.__doc__ if obj.fget else None
    return obj.__doc__


def takes_parameters(obj):
    target = obj.fget if isinstance(obj, property) else obj
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    return any(p not in ('self', 'cls') for p in sig.parameters)


# ---- the floor: everything public says what it does ----------------------


def test_every_public_callable_has_a_docstring():
    """A name can carry a one-line helper; it cannot carry an API."""
    bare = [label for label, obj in public_surface()
            if not docstring_of(obj)]
    assert not bare, ('undocumented on the public surface:\n  '
                      + '\n  '.join(bare))


# ---- the scripting surface: sections too ---------------------------------


def test_the_scripting_verbs_document_their_parameters():
    """Where a script arrives without context, the parameters are a
    table rather than something to infer from the prose."""
    missing = []
    for label, obj in public_surface():
        if not label.startswith(ENFORCED) or '.' not in label:
            continue
        if not takes_parameters(obj):
            continue          # no parameters, nothing to tabulate
        doc = docstring_of(obj) or ''
        if 'Parameters\n' not in doc:
            missing.append(label)
    covered = sum(1 for label, obj in public_surface()
                  if label.startswith(ENFORCED) and '.' in label
                  and takes_parameters(obj))
    # The ratchet reached zero on 2026-08-25: every parameterized
    # method on the scripting surface documents its parameters. It
    # stays at zero — a new public method arrives with its table, or
    # this fails.
    assert covered > 100, 'the surface shrank; check ENFORCED'
    assert not missing, (
        f'{len(missing)} of {covered} scripting methods lack a '
        'Parameters section:\n  ' + '\n  '.join(missing))


def test_a_documented_verb_names_every_parameter_it_takes():
    """A half-filled table is worse than none: a reader trusts it and
    is then missing an argument."""
    wrong = []
    for label, obj in public_surface():
        doc = docstring_of(obj) or ''
        if 'Parameters\n' not in doc or not label.startswith(ENFORCED):
            continue
        target = obj.fget if isinstance(obj, property) else obj
        params = [p for p in inspect.signature(target).parameters
                  if p not in ('self', 'cls')]
        section = doc.split('Parameters')[1]
        for p in params:
            bare = p.lstrip('*')
            if bare not in section:
                wrong.append(f'{label} does not document {p!r}')
    assert not wrong, '\n  '.join(wrong)


def test_a_documented_verb_says_what_it_returns():
    for label, obj in public_surface():
        doc = docstring_of(obj) or ''
        if 'Parameters\n' in doc and label.startswith(ENFORCED):
            assert 'Returns\n' in doc, \
                f'{label} tabulates its parameters but not its result'


def test_the_parser_matches_the_style_being_written():
    """mkdocs must parse what the docstrings are: sections written in
    NumPy form and read by a Google parser render as flat prose, which
    silently throws away the tables (2026-08-25)."""
    with open('properdocs.yml', encoding='utf-8') as handle:
        config = handle.read()
    assert 'docstring_style: numpy' in config, \
        'the scripting surface is written in NumPy sections; the docs ' \
        'build must parse them as such'


@pytest.mark.parametrize('verb', ['compute_psds', 'compute_srs',
                                  'filter_data', 'integrate',
                                  'import_file', 'export_report'])
def test_the_verbs_a_script_starts_with_are_documented(verb):
    """The ones a first script actually calls, named individually so
    the ratchet cannot drift past them."""
    doc = getattr(visualdynamics.Project, verb).__doc__ or ''
    assert 'Parameters\n' in doc and 'Returns\n' in doc, \
        f'Project.{verb} is where people start; it needs both sections'
