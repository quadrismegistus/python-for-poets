"""Splitting text into words."""

import glob
import os
import re
from functools import lru_cache
from importlib.resources import files

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

    Other languages: "german", "french", "spanish", "italian", ... (all 33
    are listed by p4p.stopword_languages()).

    The lists come with this package (copied from NLTK's stopwords corpus;
    see python4poets/stopwords/README.txt), so nothing is downloaded.
    """
    path = files("python4poets").joinpath("stopwords", language.lower())
    if not path.is_file():
        raise ValueError(f"No stopword list for {language!r}. Choose one of: {', '.join(stopword_languages())}")
    words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return frozenset(_normalize_apostrophes(w) for w in words)


def stopword_languages():
    """The languages p4p.get_stopwords has lists for."""
    folder = files("python4poets").joinpath("stopwords")
    return sorted(p.name for p in folder.iterdir() if p.is_file() and "." not in p.name)


def download_nltk_data(*names, folder=os.path.join("p4p_data", "nltk_data")):
    """
    Download NLTK data (e.g. "punkt_tab", for nltk.word_tokenize) into `folder`,
    and tell NLTK to look there. Skips anything already downloaded.

    Use this instead of nltk.download(...): it keeps the data inside your
    project folder (AI assistants' sandboxes often can't write anywhere else),
    and works on a Mac without the "Install Certificates" step.

    Example:
        >>> p4p.download_nltk_data("punkt_tab")
        >>> nltk.word_tokenize("I don't know.")
    """
    import nltk

    from ._net import trusted_certificates

    folder = os.path.abspath(folder)
    if folder not in nltk.data.path:
        nltk.data.path.append(folder)
    for name in names:
        # Each package unzips to a folder of its name, e.g. tokenizers/punkt_tab.
        if glob.glob(os.path.join(folder, "*", name)):
            continue
        with trusted_certificates():
            ok = nltk.download(name, download_dir=folder, quiet=True, raise_on_error=True)
        if not ok:
            raise RuntimeError(f"Couldn't download NLTK data {name!r}")
