"""Comparing word frequencies between two groups, and testing contingency tables."""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import chi2_contingency, false_discovery_control

from ._counts import _count_texts_by_group, count_words_by_group


def distinctive_words(
    counts_a, counts_b, names=("a", "b"), min_count=5, alpha=0.05, sort="log_ratio",
    texts_a=None, texts_b=None, min_texts=1,
):
    """
    Find the words that most distinguish one set of texts from another.

    Args:
        counts_a, counts_b: word counts for each group (Counters or dicts), e.g.
            from p4p.count_words_by_group.
        names: labels for the two groups, used in the column names
            (default ("a", "b") -> count_a, count_b, ...).
        min_count: skip words used fewer than this many times in total (default 5).
        alpha: the significance threshold for the `significant` column (default 0.05).
        sort: "log_ratio" (default: by how big the difference is) or "g2"
            (by how confident we are that the difference is real).
        texts_a, texts_b: optional {word: number of texts that use it} for
            each group. If given, adds columns texts_a and texts_b, and
            `min_texts` applies. (p4p.distinctive_words_by_group does this for you.)
        min_texts: skip words used in fewer than this many texts in total
            (default 1: no limit). A word used 20 times in a single speech
            can look "significant", but it tells us about that speech, not the group.

    Returns a DataFrame, one row per word, with columns:
        count_a, count_b  how many times each group uses the word
        texts_a, texts_b  how many texts in each group use it (if given)
        fpm_a, fpm_b      frequency per million words in each group
        obs_exp_a         group a's count divided by what we'd expect if the
                          word were spread evenly (above 1 = overused by a)
        log_ratio         log2 of fpm_a / fpm_b: +1 means twice as frequent in a,
                          -1 twice as frequent in b, 0 the same
        g2                log-likelihood (G²): how surprising the difference is
        p                 chance of a difference at least this big if there
                          were really none
        p_adj             p, corrected for testing thousands of words at once
                          (Benjamini–Hochberg)
        significant       p_adj < alpha

    Words most typical of group a are at the top, of group b at the bottom.

    Example:
        >>> counts = p4p.count_words_by_group(df, "party")
        >>> p4p.distinctive_words(counts["Democrat"], counts["Republican"],
        ...                       names=("D", "R"))
    """
    name_a, name_b = names
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    words = sorted(set(counts_a) | set(counts_b))
    a = np.array([counts_a.get(w, 0) for w in words], dtype=float)
    b = np.array([counts_b.get(w, 0) for w in words], dtype=float)
    keep = (a + b) >= min_count
    if texts_a is not None and texts_b is not None:
        n_texts_a = np.array([texts_a.get(w, 0) for w in words])
        n_texts_b = np.array([texts_b.get(w, 0) for w in words])
        keep &= (n_texts_a + n_texts_b) >= min_texts
    words, a, b = np.array(words, dtype=object)[keep], a[keep], b[keep]

    # Each word gets a 2x2 table: (this word, every other word) x (group a, group b).
    # Expected counts are what we'd see if the word were used at the same rate in both.
    total = total_a + total_b
    expected_a = (a + b) * total_a / total
    expected_b = (a + b) * total_b / total
    other_a, other_b = total_a - a, total_b - b
    expected_other_a = total_a - expected_a
    expected_other_b = total_b - expected_b

    # G² = 2 * sum(observed * ln(observed / expected)) over the four cells.
    # (xlogy treats 0 * ln(0) as 0, for words one group never uses.)
    g2 = 2 * (
        xlogy(a, a / expected_a)
        + xlogy(b, b / expected_b)
        + xlogy(other_a, other_a / expected_other_a)
        + xlogy(other_b, other_b / expected_other_b)
    )
    p = chi2_distribution.sf(g2, df=1)

    # Log ratio (Hardie 2014): a count of 0 is treated as 0.5, so words that
    # only one group uses still get a finite value.
    fpm_a = a / total_a * 1_000_000
    fpm_b = b / total_b * 1_000_000
    log_ratio = np.log2((np.where(a == 0, 0.5, a) / total_a) / (np.where(b == 0, 0.5, b) / total_b))

    result = pd.DataFrame(
        {
            "word": words,
            f"count_{name_a}": a.astype(int),
            f"count_{name_b}": b.astype(int),
            f"fpm_{name_a}": fpm_a,
            f"fpm_{name_b}": fpm_b,
            f"obs_exp_{name_a}": a / expected_a,
            "log_ratio": log_ratio,
            "g2": g2,
            "p": p,
            "p_adj": false_discovery_control(p) if len(p) else p,
        }
    )
    result["significant"] = result["p_adj"] < alpha
    if texts_a is not None and texts_b is not None:
        result.insert(3, f"texts_{name_a}", n_texts_a[keep])
        result.insert(4, f"texts_{name_b}", n_texts_b[keep])

    if sort == "log_ratio":
        result = result.sort_values("log_ratio", ascending=False)
    elif sort == "g2":
        # Signed, so group a's words still come first and group b's last.
        result = result.iloc[np.argsort(-(g2 * np.sign(log_ratio)), kind="stable")]
    else:
        raise ValueError(f'sort must be "log_ratio" or "g2", not {sort!r}')
    return result.reset_index(drop=True)


def distinctive_words_by_group(
    df, group_col, a, b, text_col="txt", min_count=5, min_texts=1, alpha=0.05, sort="log_ratio", **tokenize_kw
):
    """
    Find the words that most distinguish texts where `group_col` is `a` from those where it is `b`.

    Counts the words in each group, then runs p4p.distinctive_words. Column
    names use the group names, e.g. count_Democrat, fpm_Republican. Also
    counts how many texts in each group use each word (texts_Democrat, ...);
    `min_texts` skips words used in fewer texts than that.

    Example:
        >>> p4p.distinctive_words_by_group(df, "party", "Democrat", "Republican")
        >>> p4p.distinctive_words_by_group(df, "gender", "female", "male", stopwords="english")
        >>> p4p.distinctive_words_by_group(df, "party_main", "Conservative", "Labour", min_texts=20)
    """
    subset = df[df[group_col].isin([a, b])]
    counts = count_words_by_group(subset, group_col, text_col=text_col, **tokenize_kw)
    for group in (a, b):
        if group not in counts:
            raise ValueError(f"No rows where {group_col!r} is {group!r}")
    texts = _count_texts_by_group(subset, group_col, text_col=text_col, **tokenize_kw)
    return distinctive_words(
        counts[a], counts[b], names=(a, b), min_count=min_count, alpha=alpha, sort=sort,
        texts_a=texts[a], texts_b=texts[b], min_texts=min_texts,
    )


def crosstab_stats(df, col1, col2, method="chi2"):
    """
    Cross-tabulate two metadata columns and test whether they're related.

    Returns (statistic, p, obs_exp): the chi-squared (or G²) statistic, its
    p-value, and a table of observed / expected counts (above 1 = more
    common than chance would predict).

    Args:
        method: "chi2" (default) or "g2" for the log-likelihood statistic.

    Example:
        >>> chi2, p, obs_exp = p4p.crosstab_stats(df, "gender", "language")
    """
    if method not in ("chi2", "g2"):
        raise ValueError(f'method must be "chi2" or "g2", not {method!r}')
    table = pd.crosstab(df[col1], df[col2])
    lambda_ = "log-likelihood" if method == "g2" else None
    statistic, p, _dof, expected = chi2_contingency(table, lambda_=lambda_)
    obs_exp = table / expected
    return statistic, p, obs_exp
