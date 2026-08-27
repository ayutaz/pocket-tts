"""One normalization of Japanese text, shared by everything that writes it.

Three strings have to come from the same distribution or a run degrades with
nothing reporting why: the corpus the tokenizer is fitted on, the transcript the
DataLoader feeds at train time, and the text a user types at inference.
Sentencepiece encodes all three happily whatever they look like, so the only
symptom of a mismatch is a model that never quite becomes intelligible.

Everything here therefore lives in one function, called by prepare_ja_text.py
for the tokenizer corpus and by align_data.py --segmenter japanese, which
rewrites each manifest transcript through it before segmenting.

NFKC does most of the work -- halfwidth katakana, fullwidth latin and digits and
the rest of the width variants all want folding -- but two of its rewrites are
wrong for Japanese speech and are held out of it:

* U+2026 and U+2025 become runs of ASCII periods. That is the same damage GOL's
  own "normalized" column does: a run of full stops the model would learn to
  read aloud, and three sentence boundaries where there was one.
* U+309B and U+309C, the standalone voiced sound marks, become a SPACE plus an
  orphan combining mark. The emphatic spelling they appear in is common in this
  kind of corpus, and injecting a space into text that has none is worse than
  leaving the mark as it was written.

That also means the tokenizer has to be fitted with
`--normalization-rule identity`: sentencepiece's default applies nmt_nfkc
*inside* encode(), where no Python-side normalizer can see or prevent it.

This module lives in pocket_tts/ rather than training/ because inference needs
it and the inference package cannot import from the training package.
"""

import re
import unicodedata
from collections.abc import Callable

# Held out of NFKC. Noncharacters: permanently unassigned and forbidden in
# interchange, so unlike the private-use area -- where legacy carrier emoji
# live, and this corpus already carries some -- they cannot occur in real text.
_PROTECTED = {
    "…": "﷐",  # horizontal ellipsis
    "‥": "﷑",  # two dot leader
    "゛": "﷒",  # standalone dakuten
    "゜": "﷓",  # standalone handakuten
}
_RESTORE = {v: k for k, v in _PROTECTED.items()}

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RUN = re.compile(r"\s+")


def normalize_japanese(text: str) -> str:
    """NFKC, minus the rewrites that damage Japanese, whitespace collapsed.

    Whitespace runs collapse to a single space rather than being stripped out:
    Japanese carries none of its own, but a latin phrase inside a transcript
    needs its spaces, and align_data.py's Japanese segmenter re-attaches the
    whitespace MeCab drops so the words still rebuild this exact string.
    """
    for char, sentinel in _PROTECTED.items():
        text = text.replace(char, sentinel)
    text = unicodedata.normalize("NFKC", text)
    for sentinel, char in _RESTORE.items():
        text = text.replace(sentinel, char)
    text = _CONTROL.sub("", text)
    return _SPACE_RUN.sub(" ", text).strip()


def _identity(text: str) -> str:
    return text


NORMALIZERS: dict[str, Callable[[str], str]] = {"japanese": normalize_japanese}


def resolve_normalizer(name: str | None) -> Callable[[str], str]:
    """The normalizer a config names. None means leave the text alone.

    An unknown name raises rather than falling back to the identity: a typo in
    a config would otherwise disable normalization silently, and the only
    symptom is a model that never quite becomes intelligible.
    """
    return _identity if name is None else NORMALIZERS[name]
