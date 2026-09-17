"""The staged launch site, held to what the software actually does.

`web/launch/` is the version of visualdynamics.org for the day the
switch is flipped. It is prose rather than code, so nothing runs it
and nothing catches it going stale — which is how its format list
came to name Femap as something the toolset writes, when Femap has
only ever been read (Brandon, 2026-08-25).

So the one claim on the site that is checkable against the code is
checked: the table of supported formats, against the importer and
exporter registries. Add a reader and the test fails until the page
names it; take one away and it fails until the page stops promising
it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from visualdynamics import io

LAUNCH = pathlib.Path(__file__).resolve().parent.parent / 'web' / 'launch'
WORKFLOWS = LAUNCH / 'workflows.html'

#: Formats the page lists that are not in the file registries, and why
#: each one is nonetheless a true claim. Both are real routes in and
#: out of the software; neither goes through `import_file`.
OUTSIDE_THE_REGISTRY = {
    '.vdyn': 'the project file itself — Project.open and Project.save',
    '.png': 'photographs, added through Photos.add_file',
    '.jpg': 'the same',
    '.heic': 'the same',
}


def _rows():
    """The format table as (name, extensions, read, write) tuples."""
    html = WORKFLOWS.read_text(encoding='utf-8')
    body = re.search(r'<table class="formats">(.*?)</table>', html, re.DOTALL)
    assert body, 'the formats table is still on the page'
    rows = []
    for row in re.findall(r'<tr>(.*?)</tr>', body.group(1), re.DOTALL):
        cells = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
        if len(cells) == 4:
            plain = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            rows.append(tuple(plain))
    return rows


def _suffixes(text):
    return set(re.findall(r'\.[a-z0-9]+', text.lower()))


@pytest.mark.skipif(not WORKFLOWS.exists(),
                    reason='the launch site is not in this tree')
def test_the_website_lists_the_formats_that_exist():
    """Every reader and writer the registries hold is on the page, and
    every extension the page promises is one of them."""
    rows = _rows()
    assert rows, 'the table has rows'

    listed_read, listed_write = set(), set()
    for _, extensions, read, write in rows:
        found = _suffixes(extensions)
        if read.lower() not in ('no', 'not yet'):
            listed_read |= found
        if write.lower() in ('no', 'not yet'):
            continue
        # 'yes' means all of them; a suffix in the cell means that one
        listed_write |= _suffixes(write) or found

    real_read = set()
    for importer in io.importers():
        real_read |= _suffixes(importer.description)
    real_write = {e.suffix.lower() for e in io.exporters() if e.suffix}

    outside = set(OUTSIDE_THE_REGISTRY)
    assert real_read - listed_read == set(), (
        'the page does not name formats the software reads: '
        f'{sorted(real_read - listed_read)}')
    assert real_write - listed_write == set(), (
        'the page does not name formats the software writes: '
        f'{sorted(real_write - listed_write)}')
    assert listed_read - real_read - outside == set(), (
        'the page claims to read what nothing registers: '
        f'{sorted(listed_read - real_read - outside)}')
    assert listed_write - real_write - outside == set(), (
        'the page claims to write what nothing registers: '
        f'{sorted(listed_write - real_write - outside)}')


@pytest.mark.skipif(not WORKFLOWS.exists(),
                    reason='the launch site is not in this tree')
def test_the_website_says_which_formats_are_read_only():
    """A read-only format is marked read-only, not quietly left to be
    discovered at the export dialog — that is the mistake this file
    exists for."""
    real_write = {e.suffix.lower() for e in io.exporters() if e.suffix}
    for name, extensions, _, write in _rows():
        if _suffixes(extensions) & real_write:
            continue
        if _suffixes(extensions) <= set(OUTSIDE_THE_REGISTRY):
            continue
        assert write.lower() in ('no', 'not yet'), (
            f'{name} has no exporter, so the page must not offer it as '
            f'something written: it says {write!r}')


EXPORT_PAGE = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'export.md'
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]


def test_the_provisional_project_file_is_said_everywhere_a_user_reads():
    """`.vdyn` is alpha-only with a dated end (2026-09-13): the next
    alpha writes `.escdf` and still reads `.vdyn`, the first non-alpha
    does not. The disclaimer says so (test_disclaimer); so must the
    downloads page, the README and the guide, in the same terms."""
    for page in ('web/launch/downloads.html', 'README.md',
                 'docs/guide/README.md', 'docs/vdyn-format.md'):
        text = (ROOT_DIR / page).read_text(encoding='utf-8')
        for phrase in ('.vdyn', 'provisional', '.escdf', 'save it again'):
            assert phrase in text, f'{page} does not say {phrase!r}'


def test_the_export_page_names_every_format():
    """docs/export.md is where getting-started sends a reader for the
    full matrix, and it has drifted before (three formats short,
    2026-09-12) because nothing held it to the registries the way the
    website's table is held."""
    named = _suffixes(EXPORT_PAGE.read_text(encoding='utf-8'))
    real = set()
    for importer in io.importers():
        real |= _suffixes(importer.description)
    real |= {e.suffix.lower() for e in io.exporters() if e.suffix}
    real |= set(OUTSIDE_THE_REGISTRY)
    assert real - named == set(), (
        f'docs/export.md does not name: {sorted(real - named)}')


def test_the_navigation_wraps_on_a_phone():
    """Seven links in one unwrappable row were 647 px on a 375 px phone:
    every page became 669 px wide and the phone zoomed the whole site
    out to fit (Brandon, 2026-09-14). The bar wraps, and tightens
    under 40rem. Pinned on the stylesheet's own words, since no
    browser runs here."""
    css = (LAUNCH / 'site.css').read_text(encoding='utf-8')
    nav = re.search(r'\.top nav \{([^}]*)\}', css)
    assert nav and 'flex-wrap: wrap' in nav.group(1), 'the bar must wrap'
    assert re.search(r'@media \(max-width: 40rem\) \{\s*\.top nav', css), \
        'and tighten on a phone'
    code = re.search(r'\ncode \{([^}]*)\}', css)
    assert code and 'overflow-wrap: anywhere' in code.group(1), \
        'a URL in inline code must break rather than widen the page'


def test_the_downloads_page_links_every_example_set_the_tool_cuts():
    """The example projects are a release of their own, cut and
    uploaded by tools/cut_examples.py, and the downloads page links
    them (Brandon, 2026-09-14: examples on the website, the package
    kept small). One source for the asset names, so a set added or
    renamed in the tool cannot leave the page pointing at nothing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'cut_examples', ROOT_DIR / 'tools' / 'cut_examples.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    html = (LAUNCH / 'downloads.html').read_text(encoding='utf-8')
    for name in tool.SETS:
        assert tool.URL + name in html, f'the page does not link {name}'
    linked = set(re.findall(re.escape(tool.URL) + r'([^"]+)', html))
    assert linked == set(tool.SETS), 'the page links a set the tool does not cut'
    assert 'synthetic' in html, 'the page says the runs are synthetic'


# ---- the feature-requests page ---------------------------------------------

IDEAS = 'https://github.com/visualdynamics/visualdynamics/discussions/categories/ideas'


def test_the_requests_page_sends_people_to_github_to_ask_and_vote():
    """Requests and votes live in GitHub's Ideas discussions — one
    upvote per account is what keeps the count honest — and the page is
    the ranking plus the way there (Brandon, 2026-09-11)."""
    html = (LAUNCH / 'requests.html').read_text(encoding='utf-8')
    assert IDEAS in html, 'the button to ask or vote'
    assert "fetch('api/requests')" in html, 'the ranking, from the function'
    assert 'ghp_' not in html and 'github_pat_' not in html and \
        'api.github.com' not in html, 'the page carries no token and asks GitHub nothing'
    assert 'could not be loaded just now' in html
    for name in ('index', 'workflows', 'examples', 'downloads', 'about', 'requests'):
        page = (LAUNCH / f'{name}.html').read_text(encoding='utf-8')
        assert '<a href="requests.html"' in page, f'{name}: on the nav'


def test_the_requests_function_ranks_the_ideas_by_upvotes():
    """The function is the one place the token lives; it filters to the
    Ideas category, sorts by upvotes (GitHub orders by date only), keeps
    twenty, and answers open=false when the query fails — the private
    repository before release."""
    source = (LAUNCH / 'functions' / 'api' / 'requests.js').read_text(encoding='utf-8')
    assert 'context.env.GITHUB_TOKEN' in source
    assert "CATEGORY = 'ideas'" in source and 'LIMIT = 20' in source
    assert 'b.upvoteCount - a.upvoteCount' in source, 'sorted here, by votes'
    assert source.count('open: false') >= 3, 'no token, a failed query, a missing repository'
    assert 'max-age=' in source and 'caches.default' in source, 'cached at the edge'
    assert 'ghp_' not in source and 'github_pat_' not in source, 'the token is a secret, never a literal'


def test_the_example_tiles_wear_pictures_of_what_is_inside():
    """Each example bundle's tile shows the thing it holds — the plate,
    the quadcopter — rendered from the app's own scene on a transparent
    ground (Brandon, 2026-09-16), not the application's mark. Held to
    shape, not bytes: VTK renders differ across builds."""
    import os

    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtGui import QImage

    page = (ROOT_DIR / 'web' / 'launch' / 'downloads.html').read_text(encoding='utf-8')
    examples = page.split('class="downloads examples"')[1].split('</div>')[0]
    assert 'mark.png' not in examples
    seen = []
    for name in ('plate.png', 'drone.png'):
        assert f'src="{name}"' in examples, f'the tile shows {name}'
        image = QImage(str(ROOT_DIR / 'web' / 'launch' / name))
        assert not image.isNull(), name
        assert image.width() == image.height() >= 256, 'square, and big enough for any screen'
        assert image.hasAlphaChannel()
        w = image.width() - 1
        for x, y in ((0, 0), (w, 0), (0, w), (w, w)):
            assert image.pixelColor(x, y).alpha() == 0, f'{name}: corner ({x},{y}) is not transparent'
        opaque = sum(1 for x in range(0, image.width(), 8) for y in range(0, image.height(), 8)
                     if image.pixelColor(x, y).alpha() > 0)
        assert opaque > 40, f'{name} shows something'
        seen.append(image)
    assert seen[0] != seen[1], 'two different things'
