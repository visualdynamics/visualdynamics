"""How an object is named in a project.

An importer's keys are written for code (`time_data`, `Modal_frf`,
`ChannelTable`), and the tree is for reading. So the tree shows the
object spelled the way it would be written down: words separated by
spaces, each capitalized, and the acronyms of the field (FRF, CPSD, PSD,
ASD, SRS, DOF, RMS) in capitals.

The name is the key: renaming an item renames the object, so these
names are what `MainWindow.objects` is keyed by. Nothing here parses
them back.
"""

from __future__ import annotations

import re
from typing import Any

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


def imported_name(obj: Any) -> str:
    """Name an imported acquisition pair by its response and reference.

    Explicit channel roles distinguish a pair from an ordinary
    multichannel history, whose name remains its data type.
    """
    from .core.data import TimeHistory

    if isinstance(obj, TimeHistory) and obj.num_records == 2:
        roles = getattr(obj, 'roles', None) or {}
        response = [dof for (dof, _quantity), role in roles.items()
                    if role == 'response']
        reference = [dof for (dof, _quantity), role in roles.items()
                     if role == 'reference']
        if len(response) == len(reference) == 1:
            return f'Time History - {response[0]} - ref {reference[0]}'
    return display_name(type(obj).__name__)


def display_name(key: str) -> str:
    """A key from code as a name to read: 'Modal_frf' -> 'Modal FRF',
    'ShapeSet' -> 'Shape Set'.

    The key is split into words at underscores, spaces, digits and
    internal capitals. Each word is capitalized, or written as its
    acronym, and the words are joined with single spaces. A key already
    spelled that way, such as 'Shape Set', comes back unchanged.
    Punctuation is dropped and lower-case words are capitalized
    ('my frf data' -> 'My FRF Data'), so this is for keys from code, not
    for names a person typed. A key with no letters or digits comes back
    as given.
    """
    words = _WORDS.findall(key)
    if not words:
        return key
    return ' '.join(ACRONYMS.get(word.lower(), word[:1].upper() + word[1:])
                    for word in words)
