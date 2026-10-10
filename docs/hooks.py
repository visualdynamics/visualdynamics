"""Build-time hooks for the documentation, loaded by `hooks:` in properdocs.yml.

pymdown-extensions 12.2 (2026-10) made `md` a required first argument of
`pymdownx.highlight.Highlight`, and mkdocstrings 1.0.6 — the latest —
subclasses it and still calls `super().__init__(**config)` without one,
so every API page with a code block failed to build. Pinning pymdown
back is not how a break is handled here (PRINCIPLES.md, 12). This hands
the call the Markdown instance mkdocstrings' highlighter already holds,
and only while the installed `Highlight` requires it and the installed
mkdocstrings does not supply it — a no-op once either side is fixed, and
then this file and its `hooks:` line can go.
"""

from __future__ import annotations

import inspect


def on_startup(command, dirty):
    from mkdocstrings._internal.handlers.rendering import Highlighter
    from pymdownx.highlight import Highlight

    md = inspect.signature(Highlight.__init__).parameters.get('md')
    if md is None or md.default is not inspect.Parameter.empty:
        return
    configure = Highlighter.__init__
    if getattr(configure, 'passes_md', False):
        return

    def __init__(self, md):
        # `super().__init__` inside `configure` resolves to
        # `Highlight.__init__` at call time, so for the length of the
        # call it is a version that puts `md` first.
        bare = Highlight.__init__
        Highlight.__init__ = lambda inner, **config: bare(inner, md, **config)
        try:
            configure(self, md)
        finally:
            Highlight.__init__ = bare

    __init__.passes_md = True
    Highlighter.__init__ = __init__
