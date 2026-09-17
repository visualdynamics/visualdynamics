"""The principles are documented in three places and must agree.

`PRINCIPLES.md` is the canonical list (Brandon, 2026-08-25). It is
linked from `README.md` for someone arriving at the repository, from
`AGENTS.md` for an AI assistant working in it, and symlinked into
`docs/` so it appears on the documentation site — which is the whole
point of writing it down rather than leaving it implicit.

Three places to state a count is three places to get it wrong, so the
count is checked rather than trusted.
"""

from __future__ import annotations

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

WORDS = {9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
         13: 'thirteen', 14: 'fourteen', 15: 'fifteen'}


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as handle:
        return handle.read()


def numbered():
    """The principles, by their leading `**N.` marker."""
    return [int(n) for n in
            re.findall(r'^\*\*(\d+)\.', read('PRINCIPLES.md'), re.MULTILINE)]


def test_the_principles_are_numbered_without_gaps():
    found = numbered()
    assert found, 'no numbered principles found'
    assert found == list(range(1, len(found) + 1)), \
        f'numbering is not 1..n: {found}'


def test_every_principle_says_something():
    """A heading with no body is a slogan. Each one carries its
    reasoning, and several carry the edge case that tests it."""
    body = read('PRINCIPLES.md')
    blocks = re.split(r'^\*\*\d+\.', body, flags=re.MULTILINE)[1:]
    thin = [b.split('\n')[0][:50] for b in blocks if len(b.strip()) < 200]
    assert not thin, f'these principles are stated but not argued: {thin}'


def test_the_count_agrees_everywhere_it_is_stated():
    """README, AGENTS.md and the document itself each name a number.
    Three places to say it is three places to get it wrong."""
    count = len(numbered())
    word = WORDS[count]
    # AGENTS.md is the private rulebook and does not travel with the
    # published tree, so it is checked where it exists and skipped
    # where it does not — the same file has to pass in both trees.
    where = ['PRINCIPLES.md', 'README.md']
    if os.path.exists(os.path.join(ROOT, 'AGENTS.md')):
        where.append('AGENTS.md')
    for path in where:
        text = read(path).lower()
        assert word in text, \
            f'{path} does not say "{word}" — the count moved to ' \
            f'{count} and this file was not updated'


def test_it_is_reachable_from_every_audience():
    assert 'PRINCIPLES.md' in read('README.md'), 'a visitor cannot find it'
    if os.path.exists(os.path.join(ROOT, 'AGENTS.md')):
        assert 'PRINCIPLES.md' in read('AGENTS.md'), \
            'an assistant is not sent to it'
    link = os.path.join(ROOT, 'docs', 'principles.md')
    assert os.path.exists(link), 'the docs site has no principles page'
    if os.name == 'nt':
        # the page is a POSIX symlink; a Windows checkout materializes
        # it, so sameness is only checkable where symlinks exist —
        # the docs are built on the machines that have them
        return
    assert os.path.realpath(link) == os.path.join(ROOT, 'PRINCIPLES.md'), \
        'the docs page is a copy rather than the same file — which is ' \
        'principle 9 broken by the document that states it. The ' \
        'published tree keeps the link for the same reason.'


# ---- principle 12, which is checkable ------------------------------------

#: What a dependency specifier may not contain. `>=` is allowed — a
#: floor states the minimum feature set and freezes nothing; every
#: other operator caps, pins, or excludes, which is how a project ends
#: up installable only beside a museum.
CAPPING = ('<', '<=', '==', '~=', '!=')


def dependency_specs():
    """Every declared dependency, runtime and optional alike."""
    import tomllib

    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as handle:
        project = tomllib.load(handle)['project']
    specs = list(project.get('dependencies', []))
    for group in (project.get('optional-dependencies') or {}).values():
        specs.extend(group)
    return specs


def test_no_dependency_is_capped_or_pinned():
    """Principle 12. A cap defers a small problem into a large one:
    they accumulate, they conflict, and the package stops being
    installable beside whatever else the user already has."""
    offenders = []
    for spec in dependency_specs():
        # strip the extras bracket so 'mkdocstrings[python]' is clean
        bare = re.sub(r'\[.*?\]', '', spec)
        for op in CAPPING:
            if op in bare:
                offenders.append(f'{spec} (contains {op!r})')
                break
    assert not offenders, (
        'dependencies are capped or pinned, which principle 12 '
        'forbids:\n  ' + '\n  '.join(offenders))


def test_the_package_carries_no_version_gated_shims():
    """The other half of 12: no code that branches on a dependency's
    version to keep an old one working. `update.py` compares Visual
    Dynamics' *own* version against the latest release, which is a
    different thing and is why the check is scoped by content."""
    import pathlib

    shims = []
    for path in pathlib.Path(ROOT, 'src', 'visualdynamics').rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        for marker in ('pkg_resources', 'LooseVersion', 'distutils.version'):
            if marker in text:
                shims.append(f'{path.name} uses {marker}')
    assert not shims, '\n  '.join(shims)
