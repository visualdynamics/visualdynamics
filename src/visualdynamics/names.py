"""How an object is named in a project.

An importer's keys are for code — `time_data`, `Modal_frf`, `ChannelTable` —
and the tree is for reading. So the tree shows the object, spelled the way it
would be written down: words separated by spaces, each capitalized, and the
acronyms this field actually uses left as acronyms.

The name is the key: renaming an item renames the object, so these are what
`MainWindow.objects` is keyed by. Nothing here parses them back.
"""

from __future__ import annotations

import re

# 'Frf' is not a word, and neither is 'Cpsd'. Capitalizing the first letter of
# every word is the rule; these are the exceptions a structural dynamicist
# would otherwise have to read twice.
ACRONYMS = {
    'frf': 'FRF',
    'cpsd': 'CPSD',
    'psd': 'PSD',
    'asd': 'ASD',
    'srs': 'SRS',
    'dof': 'DOF',
    'rms': 'RMS',
}

# underscores, spaces, and the internal capitals of a class name — 'ShapeSet'
# and 'shape_set' are the same two words written two ways
_WORDS = re.compile(r'[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+')


def display_name(key: str) -> str:
    """'Modal_frf' -> 'Modal FRF'. 'ShapeSet' -> 'Shape Set'.

    Anything already spelled as a name comes back unchanged, so a user's own
    rename is never reformatted underneath them.
    """
    words = _WORDS.findall(key)
    if not words:
        return key
    return ' '.join(ACRONYMS.get(word.lower(), word[:1].upper() + word[1:])
                    for word in words)
