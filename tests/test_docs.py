"""The documentation cannot go stale without failing.

The API reference is no longer written down at all: `docs/gen_api.py`
walks the package at build time and mkdocstrings renders the docstrings,
so the reference *is* the code and there is nothing to drift. What can
still rot is the hand-written half — a guide pointing at a page that has
been renamed — and the generator itself, which has to keep naming every
module in the package.

The site build is the other half of this check and lives in CI
(`properdocs build --strict`, which fails on a broken cross-reference); it
is not run here because it wants the `docs` extra and eight seconds,
neither of which belongs in a per-edit suite.
"""

from __future__ import annotations

import pathlib
import pkgutil
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_every_documentation_link_resolves():
    """A link into `docs/api/` cannot be checked by existence: those
    pages are written into the source tree while the site builds
    (`docs/gen_api.py`), so on a fresh checkout the directory is not
    there at all. Checking existence passed for a week on machines that
    had built the site and failed on the first CI run to see a link into
    it. Such a link is instead held to the same walk the generator uses
    — it resolves exactly when the module it names is real — and the
    generated pages themselves are left out of the sweep, because what
    they link to is the generator's business and `properdocs build --strict`
    already fails on any broken cross-reference in the built site.
    """
    import sys

    sys.path.insert(0, str(ROOT / 'docs'))
    from api_layout import modules

    generated = {f'{name}.md' for name in modules()} | {'index.md',
                                                        'SUMMARY.md'}
    api = (ROOT / 'docs' / 'api').resolve()
    pages = [page for page in (ROOT / 'docs').rglob('*.md')
             if api not in page.resolve().parents]
    assert pages, 'there are docs to check'
    for page in pages:
        text = page.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)#:]+)\)', text):
            if target.startswith(('http', 'mailto')):
                continue
            resolved = (page.parent / target).resolve()
            if api in resolved.parents:
                assert resolved.name in generated, f'{page.name} -> {target}'
            else:
                assert resolved.exists(), f'{page.name} -> {target}'


def test_the_generator_reaches_every_module_in_the_package():
    """The reference covers the package, not a list of it.

    Its predecessor was a hand-kept table of twenty modules while the
    package had fifty-nine, and the octave bands, the shock response
    spectrum, the replication metrics and every file-format reader were
    simply absent. A walk cannot fall behind like that, and this says so
    rather than trusting it.
    """
    import sys

    import visualdynamics

    # the *layout* lives beside the docs it describes, not on the path.
    # Deliberately not `gen_api`: importing that runs it, which is how a
    # mkdocs-gen-files script works, and running it needs mkdocs — which
    # this environment does not have. That is exactly how this test
    # failed on CI twice.
    sys.path.insert(0, str(ROOT / 'docs'))
    from api_layout import modules, section_of

    covered = set(modules())
    present = {'visualdynamics'} | {
        module.name
        for module in pkgutil.walk_packages(visualdynamics.__path__,
                                            'visualdynamics.')
        if not module.name.rsplit('.', 1)[-1].startswith('_')}
    assert covered == present, (
        'the API reference and the package disagree about what is in it: '
        f'{sorted(present - covered)} missing, '
        f'{sorted(covered - present)} invented')
    assert len(covered) > 50, 'a walk that found almost nothing is a bug'
    # and every one of them lands in a section rather than a default bucket
    for module in covered:
        assert 0 <= section_of(module) < 8


def test_every_project_type_has_a_workflow_page():
    """Each project type should have a workflow (Brandon, 2026-08-28).

    Held to the template registry rather than to a list here: the day a
    seventh project type lands, this fails until its walkthrough
    exists — which is the moment the walkthrough is cheapest to write.
    The page must also be listed in the section's SUMMARY, because a
    page the navigation cannot reach documents nothing.
    """
    from visualdynamics.core.report import PROJECT_TEMPLATES

    workflows = ROOT / 'docs' / 'guide' / 'workflows'
    listed = (workflows / 'SUMMARY.md').read_text(encoding='utf-8')
    assert PROJECT_TEMPLATES, 'there are project types at all'
    for project_type, templates in PROJECT_TEMPLATES.items():
        # a type's page is named for its one report; the random-and-
        # sine type makes two (2026-10-05) and keeps its own page
        slug = templates[0] if len(templates) == 1 else 'mixed'
        page = workflows / f'{slug}-workflow.md'
        assert page.exists(), f'{project_type} has no workflow page'
        assert f'{slug}-workflow.md' in listed, \
            f'{project_type} is not in the workflows SUMMARY'
