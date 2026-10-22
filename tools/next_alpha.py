"""Set `__version__` to the next alpha and print it.

    tools/next_alpha.py            # 0.1.0a66 -> 0.1.0a67, written and printed

`tools/sync_public.sh --release` runs this before it syncs: a sync that
carries a version with no release yet is how a release is asked for
(packaging/release_decision.py). Anything that is not an alpha
(`X.Y.ZaN`) is refused, since what follows a beta or a final release is
a choice and not a count.
"""

from __future__ import annotations

import pathlib
import re
import sys

VERSION_FILE = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'visualdynamics' / '__init__.py'
ALPHA = re.compile(r'^(\d+\.\d+\.\d+)a(\d+)$')
ASSIGNED = re.compile(r"^(__version__\s*=\s*)(['\"])([^'\"]+)\2", re.MULTILINE)


def next_alpha(version: str) -> str:
    """The alpha after `version`: 0.1.0a66 gives 0.1.0a67."""
    found = ALPHA.match(version)
    if found is None:
        raise ValueError(f'{version} is not an alpha (X.Y.ZaN): set the next version by hand')
    return f'{found.group(1)}a{int(found.group(2)) + 1}'


def bump(text: str) -> tuple[str, str]:
    """`text`, a module's source, with its `__version__` set to the next
    alpha, and that version."""
    assigned = list(ASSIGNED.finditer(text))
    if len(assigned) != 1:
        raise ValueError(f'expected one __version__ assignment, found {len(assigned)}')
    found = assigned[0]
    new = next_alpha(found.group(3))
    start, end = found.span(3)
    return text[:start] + new + text[end:], new


def main() -> int:
    text, new = bump(VERSION_FILE.read_text(encoding='utf-8'))
    VERSION_FILE.write_text(text, encoding='utf-8')
    print(new)
    return 0


if __name__ == '__main__':
    sys.exit(main())
