"""Keyword in context (KWIC): every place a word appears, with the words around it."""

import pandas as pd

from ._text import WORD_PATTERN, _normalize_apostrophes


def kwic(texts, word, window=8, text_col="txt"):
    """
    Show every occurrence of a word (or phrase) with the words around it.

    Args:
        texts: a single string, or a DataFrame with one text per row.
        word: the word or phrase to find, e.g. "liberty" or "united states".
            Matching ignores upper/lower case.
        window: how many words of context on each side (default 8).
        text_col: for a DataFrame, the column holding the text (default "txt").

    Returns a DataFrame with columns left, keyword, right; for a DataFrame
    input, each row also keeps that text's metadata (id, year, ...).

    Example:
        >>> p4p.kwic(df, "immigration")
        >>> p4p.kwic(df[df["year"] > 2000], "united states", window=5)
    """
    if isinstance(texts, str):
        return pd.DataFrame(_kwic_text(texts, word, window), columns=["left", "keyword", "right"])

    metadata = texts.drop(columns=[text_col])
    rows = []
    for (_, meta), text in zip(metadata.iterrows(), texts[text_col]):
        for left, keyword, right in _kwic_text(text, word, window):
            rows.append({**meta.to_dict(), "left": left, "keyword": keyword, "right": right})
    # Put the concordance right after the first column (usually the id), so
    # it's visible without scrolling; the rest of the metadata follows.
    first, *rest = metadata.columns
    return pd.DataFrame(rows, columns=[first, "left", "keyword", "right", *rest])


def _kwic_text(text, phrase, window):
    query = [w.lower() for w in WORD_PATTERN.findall(_normalize_apostrophes(phrase))]
    if not query or not isinstance(text, str):
        return []
    normalized = _normalize_apostrophes(text)  # same length as text
    # Quick check before the slower word-by-word search.
    if query[0] not in normalized.lower():
        return []

    # Find each word's position (start, end) in the original text, so the
    # context keeps its original punctuation and capitals.
    spans = [m.span() for m in WORD_PATTERN.finditer(normalized)]
    words = [normalized[start:end].lower() for start, end in spans]
    n = len(query)

    hits = []
    for i in range(len(words) - n + 1):
        if words[i : i + n] == query:
            start, end = spans[i][0], spans[i + n - 1][1]
            left_start = spans[max(i - window, 0)][0]
            right_end = spans[min(i + n - 1 + window, len(spans) - 1)][1]
            hits.append(
                (
                    _one_line(text[left_start:start]),
                    _one_line(text[start:end]),
                    _one_line(text[end:right_end]),
                )
            )
    return hits


def _one_line(s):
    return " ".join(s.split())
