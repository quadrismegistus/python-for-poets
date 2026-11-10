"""Splitting text into words."""

import re
from functools import lru_cache

# A "word" is a run of letters, optionally joined by apostrophes: "don't",
# "o'er", "rock'n'roll". Numbers and punctuation are left out.
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

# The same, but numbers count as words too: "1776", "covid19".
WORD_OR_NUMBER_PATTERN = re.compile(r"\w+(?:'\w+)*")


def _normalize_apostrophes(text):
    # Curly apostrophes (’) become straight ones ('), so "don’t" == "don't".
    return text.replace("’", "'").replace("‘", "'")


def tokenize(text, lower=True, numbers=False, stopwords=None):
    """
    Split a text into a list of words.

    Words keep their internal apostrophes ("don't" stays one word); hyphenated
    words are split ("well-known" -> "well", "known"); punctuation is dropped.

    Args:
        text: the string to split.
        lower: lowercase every word (default True).
        numbers: keep numbers like "1776" as words (default False).
        stopwords: a set of words to leave out, e.g. p4p.get_stopwords().
            Use the language name as a shortcut: stopwords='english'.

    Example:
        >>> p4p.tokenize("I don't know—do you?")
        ['i', "don't", 'know', 'do', 'you']
    """
    text = _normalize_apostrophes(text)
    if lower:
        text = text.lower()
    pattern = WORD_OR_NUMBER_PATTERN if numbers else WORD_PATTERN
    words = pattern.findall(text)
    if stopwords:
        if isinstance(stopwords, str):
            stopwords = get_stopwords(stopwords)
        words = [w for w in words if w not in stopwords]
    return words


@lru_cache(maxsize=None)
def get_stopwords(language="english"):
    """
    Get NLTK's list of very common words ("the", "of", "and", ...) for a language.

    Returns a frozenset. Add your own words with `|`:

        >>> stops = p4p.get_stopwords() | {"applause"}

    Other languages: "german", "french", "spanish", "italian", ... (see
    nltk.corpus.stopwords.fileids()).
    """
    import nltk

    try:
        words = nltk.corpus.stopwords.words(language)
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words(language)
    return frozenset(_normalize_apostrophes(w) for w in words)
