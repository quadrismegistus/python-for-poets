"""Counting words."""

from collections import Counter
from functools import lru_cache

import pandas as pd
from tqdm.auto import tqdm

from ._text import get_stopwords, tokenize


def count_words(text, lower=True, numbers=False, stopwords=None):
    """
    Count how often each word appears in a text.

    Takes the same options as p4p.tokenize. Returns a Counter, which works like
    a dictionary of {word: count} and has a .most_common() method.

    Results are remembered, so counting the same text twice is instant.

    Example:
        >>> counts = p4p.count_words("the cat sat on the mat")
        >>> counts["the"]
        2
        >>> counts.most_common(1)
        [('the', 2)]
    """
    # Return a copy, so changing the result doesn't change the remembered one.
    return Counter(_remembered_counts(text, lower, numbers, stopwords))


def _remembered_counts(text, lower=True, numbers=False, stopwords=None):
    # The remembered Counter itself, not a copy: only for code that won't change it.
    if isinstance(stopwords, str):
        stopwords = get_stopwords(stopwords)
    elif stopwords is not None:
        stopwords = frozenset(stopwords)
    return _count_words(text, lower, numbers, stopwords)


@lru_cache(maxsize=None)
def _count_words(text, lower, numbers, stopwords):
    return Counter(tokenize(text, lower=lower, numbers=numbers, stopwords=stopwords))


def count_words_by_group(df, group_col, text_col="txt", **tokenize_kw):
    """
    Add up word counts for all the texts in each group.

    Args:
        df: a DataFrame with one text per row.
        group_col: the column to group by, e.g. "party".
        text_col: the column holding the text (default "txt").
        **tokenize_kw: options for p4p.tokenize, e.g. stopwords="english".

    Returns a dictionary of {group: Counter}.

    Example:
        >>> counts = p4p.count_words_by_group(df, "party")
        >>> counts["Democrat"].most_common(10)
    """
    groups = {}
    for group, text in tqdm(zip(df[group_col], df[text_col]), total=len(df)):
        if pd.isna(group):
            continue
        if group not in groups:
            groups[group] = Counter()
        groups[group].update(_remembered_counts(text, **tokenize_kw))
    return groups


def _count_texts_by_group(df, group_col, text_col="txt", **tokenize_kw):
    # {group: {word: how many texts in the group use it}}
    groups = {}
    for group, text in zip(df[group_col], df[text_col]):
        if pd.isna(group):
            continue
        if group not in groups:
            groups[group] = Counter()
        groups[group].update(_remembered_counts(text, **tokenize_kw).keys())
    return groups


def count_word_over_time(df, word, time_col="year", text_col="txt", **tokenize_kw):
    """
    Count a word in every text, as raw counts and per million words ("fpm").

    `word` can be one word or a list of words. Returns one row per text (per
    word), with the text's metadata plus columns: word, count, total, fpm.
    Sorted by `time_col`.

    Example:
        >>> p4p.count_word_over_time(df, "immigration")
        >>> p4p.count_word_over_time(df, ["war", "peace"])
    """
    words = [word] if isinstance(word, str) else list(word)
    if tokenize_kw.get("lower", True):
        words = [w.lower() for w in words]

    counts = [_remembered_counts(text, **tokenize_kw) for text in df[text_col]]
    totals = pd.Series([sum(c.values()) for c in counts], index=df.index)
    metadata = df.drop(columns=[text_col])

    parts = []
    for w in words:
        part = metadata.copy()
        part["word"] = w
        part["count"] = [c[w] for c in counts]
        part["total"] = totals
        part["fpm"] = (part["count"] / totals * 1_000_000).where(totals > 0, 0.0)
        parts.append(part)
    return pd.concat(parts).sort_values(time_col, kind="stable").reset_index(drop=True)
