"""One normalization of Japanese text, shared by everything that writes it.

Three strings have to come from the same distribution or a run degrades with
nothing reporting why: the corpus the tokenizer is fitted on, the transcript the
DataLoader feeds at train time, and the text a user types at inference.
Sentencepiece encodes all three happily whatever they look like, so the only
symptom of a mismatch is a model that never quite becomes intelligible.

Everything here therefore lives in one function, and both prepare_ja_text.py
(tokenizer corpus) and the manifest builder call it.

The one subtlety is the ellipsis. Bare NFKC rewrites U+2026 as three ASCII
periods, which is the same damage the GOL dataset's own "normalized" column
does -- a run of full stops the model would learn to read aloud, and three
sentence boundaries where there was one. So U+2026 and U+2025 are held out
while NFKC folds everything else (halfwidth katakana, fullwidth latin and
digits, and the rest of the width variants, all of which we do want folded).

That also means the tokenizer has to be fitted with
`--normalization-rule identity`: sentencepiece's default applies nmt_nfkc
*inside* encode(), where no Python-side normalizer can see or prevent it.
"""

import re
import unicodedata

# Held out of NFKC. Private-use code points, so they cannot collide with text.
_PROTECTED = {"…": "", "‥": ""}
_RESTORE = {v: k for k, v in _PROTECTED.items()}

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RUN = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC, minus the ellipsis damage, with whitespace runs collapsed.

    Whitespace is collapsed rather than stripped out: Japanese carries none of
    its own, but a latin phrase inside a transcript needs its spaces, and the
    aligner's segmenter passes them through as readingless tokens either way.
    """
    for char, sentinel in _PROTECTED.items():
        text = text.replace(char, sentinel)
    text = unicodedata.normalize("NFKC", text)
    for sentinel, char in _RESTORE.items():
        text = text.replace(sentinel, char)
    text = _CONTROL.sub("", text)
    return _SPACE_RUN.sub(" ", text).strip()
