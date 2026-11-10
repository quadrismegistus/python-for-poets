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
    if isinstance(stopwords, str):
        stopwords = get_stopwords(stopwords)
    elif stopwords is not None:
        stopwords = frozenset(stopwords)
    # Return a copy, so changing the result doesn't change the remembered one.
    return Counter(_count_words(text, lower, numbers, stopwords))


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
        groups[group].update(count_words(text, **tokenize_kw))
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
    lower = tokenize_kw.get("lower", True)
    if lower:
        words = [w.lower() for w in words]

    metadata = df.drop(columns=[text_col])
    rows = []
    for (_, meta), text in zip(metadata.iterrows(), df[text_col]):
        counts = count_words(text, **tokenize_kw)
        total = sum(counts.values())
        for w in words:
            row = meta.to_dict()
            row["word"] = w
            row["count"] = counts[w]
            row["total"] = total
            row["fpm"] = counts[w] / total * 1_000_000 if total else 0.0
            rows.append(row)
    return pd.DataFrame(rows).sort_values(time_col).reset_index(drop=True)
