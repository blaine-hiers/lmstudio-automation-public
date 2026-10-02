"""Quote verification: is a model-produced "quote" really in the source text?

    verify_quote(quote, source_text) -> bool

Both sides are normalized (Unicode NFKC, case, curly/straight quotes and
apostrophes, dashes) and split into word/number tokens, so punctuation,
case, whitespace and quote marks never matter. The articles "a", "an" and
"the" are dropped from both sides. What's left of the quote must then appear
in the source **token for token, in order, contiguously** -- a character
ratio is not enough, because "we will withdraw" vs "we will not withdraw",
"rise" vs "fall" or "150" vs "15" are near-identical strings with opposite
meanings. Any differing negation word, number or other content word leaves
the quote unverified.

An ellipsis in the quote ("... ", "…") marks an omission: each part must
appear in the source, in order. A quote shorter than `MIN_WORDS` tokens is
never verified (it could match almost anywhere). A quote that fails is shown
flagged "unverified", never presented as checked.
"""

from __future__ import annotations

import re
import unicodedata

MIN_WORDS = 3
MAX_SOURCE_CHARS = 60_000  # bound the scan (source text is already capped upstream)
IGNORED_WORDS = frozenset({"a", "an", "the"})

_TRANS = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‒": "-", "−": "-", " ": " ",
})
_ELLIPSIS = re.compile(r"(?:\.\s*){3,}|…")
# words (with inner apostrophes: "don't") and numbers (with inner . or ,: "3.5", "1,200")
_TOKEN = re.compile(r"[^\W_]+(?:['.,][^\W_]+)*")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).translate(_TRANS)
    return " ".join(t.split()).lower()


_DIGIT_GROUP = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _canon(token: str) -> str:
    """Formatting-only variants compare equal: "1,200" == "1200", "u.s" ==
    "us", "café" == "cafe". A decimal point stays ("3.5" != "35")."""
    token = _DIGIT_GROUP.sub("", token)
    if "." in token and not any(c.isdigit() for c in token):
        token = token.replace(".", "")
    decomposed = unicodedata.normalize("NFKD", token)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    """Normalized word/number tokens, articles dropped."""
    return [_canon(t) for t in _TOKEN.findall(normalize(text)) if t not in IGNORED_WORDS]


def _find(needle: list[str], hay: list[str], start: int) -> int:
    """Index just past the first contiguous occurrence of `needle` in
    `hay[start:]`, or -1."""
    n = len(needle)
    first = needle[0]
    for i in range(start, len(hay) - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            return i + n
    return -1


def verify_quote(quote: str, source_text: str) -> bool:
    parts = [tokens(p) for p in _ELLIPSIS.split(unicodedata.normalize("NFKC", str(quote or "")))]
    parts = [p for p in parts if p]
    if sum(len(p) for p in parts) < MIN_WORDS:
        return False
    hay = tokens((source_text or "")[:MAX_SOURCE_CHARS])
    pos = 0
    for part in parts:
        pos = _find(part, hay, pos)
        if pos < 0:
            return False
    return True
