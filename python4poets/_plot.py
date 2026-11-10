"""Plotting word frequencies."""

import numpy as np
import pandas as pd
import plotnine as p9

from ._counts import count_word_over_time

# Colors for groups, always used in this order (blue, orange, aqua, ...),
# chosen to stay distinguishable for colorblind readers.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

MIN_POINTS_FOR_TREND = 10


def plot_word_over_time(df, word, time_col="year", color=None, per_text=False, text_col="txt", **tokenize_kw):
    """
    Plot how often a word is used (per million words) over time.

    Each point is one year (or whatever `time_col` is): the word's frequency
    in all that year's texts put together, or, if `color` is set, in that
    year's texts from each group.

    Args:
        df: a DataFrame with one text per row.
        word: one word, or a list of words to compare.
        time_col: the column for the x-axis (default "year").
        color: a column to color points by, e.g. "party".
        per_text: if True, plot one point per text instead (fine for a few
            hundred texts; unreadable for many thousands).

    With a list of words, each word gets its own color, or (if `color` is
    also set) its own panel.

    Example:
        >>> p4p.plot_word_over_time(df, "immigration", color="party")
        >>> p4p.plot_word_over_time(df, ["war", "peace"])
    """
    data = count_word_over_time(df, word, time_col=time_col, text_col=text_col, **tokenize_kw)
    several_words = not isinstance(word, str)
    if not per_text:
        # Add up counts and totals for each time (x group, x word), then
        # divide: the frequency in all those texts put together.
        keys = list(dict.fromkeys([time_col] + ([color] if color else []) + ["word"]))
        data = data.groupby(keys, as_index=False, sort=False)[["count", "total"]].sum()
        data = data[data["total"] > 0]
        data["fpm"] = data["count"] / data["total"] * 1_000_000
    if several_words and color is None:
        color = "word"

    aes = p9.aes(x=time_col, y="fpm", color=color) if color else p9.aes(x=time_col, y="fpm")
    title = word if isinstance(word, str) else ", ".join(word)

    # Only draw trend lines for groups with enough points to have a trend.
    trend_groups = [g for g in [color, "word" if several_words else None] if g]
    if trend_groups:
        trend_data = data.groupby(trend_groups).filter(lambda group: len(group) >= MIN_POINTS_FOR_TREND)
    else:
        trend_data = data

    fig = (
        p9.ggplot(data, aes)
        + p9.geom_point(size=1.5, alpha=0.6)
        # Plain (non-robust: it=0) lowess. The robust default treats a word's
        # spikes as outliers and flattens mostly-zero series to nothing.
        + p9.geom_smooth(data=trend_data, method="lowess", span=0.4, method_args={"it": 0}, se=False, size=1)
        + p9.labs(x=time_col, y="frequency per million words", title=f'"{title}"')
        + p9.theme_minimal()
    )
    if color:
        groups = list(dict.fromkeys(data["word"])) if color == "word" else sorted(data[color].dropna().unique())
        # More groups than colors: the rest are drawn grey, not in made-up colors.
        fig += p9.scale_color_manual(values=COLORS, limits=groups[: len(COLORS)], na_value="#aaaaaa")
    if several_words and color != "word":
        fig += p9.facet_wrap("word")
    return fig


def plot_distinctive_words(result, n=15, significant_only=True, by="log_ratio"):
    """
    Bar chart of the most distinctive words for each group, from p4p.distinctive_words.

    By default, bars show the log ratio: how many times more frequent (on a
    log2 scale: 1 = twice as frequent, 2 = four times) a word is in one group
    than the other. With by="g2", the words with the highest G² are shown
    instead, with bars showing G²: the differences we're most sure are real,
    which tend to be commoner words.

    Args:
        result: the DataFrame returned by p4p.distinctive_words or
            p4p.distinctive_words_by_group.
        n: how many words to show for each group (default 15).
        significant_only: only show statistically significant words (default True).
        by: "log_ratio" (default) or "g2": which measure to choose and draw words by.

    Example:
        >>> result = p4p.distinctive_words_by_group(df, "party", "Democrat", "Republican")
        >>> p4p.plot_distinctive_words(result)
        >>> p4p.plot_distinctive_words(result, by="g2")
    """
    if by not in ("log_ratio", "g2"):
        raise ValueError(f'by must be "log_ratio" or "g2", not {by!r}')
    name_a, name_b = [c[len("fpm_") :] for c in result.columns if c.startswith("fpm_")]
    data = result[result["significant"]] if significant_only else result
    # G² is always positive: give it the log ratio's sign, so group b's words go left.
    data = data.assign(value=data["g2"] * np.sign(data["log_ratio"]) if by == "g2" else data["log_ratio"])
    top_a = data[data["value"] > 0].nlargest(n, "value")
    top_b = data[data["value"] < 0].nsmallest(n, "value")
    data = (
        pd.concat([top_a.assign(group=f"more {name_a}"), top_b.assign(group=f"more {name_b}")])
        .sort_values("value")
    )
    data["word"] = pd.Categorical(data["word"], categories=list(dict.fromkeys(data["word"])))

    return (
        p9.ggplot(data, p9.aes(x="word", y="value", fill="group"))
        + p9.geom_col(width=0.8)
        + p9.coord_flip()
        + p9.geom_hline(yintercept=0, color="#888888", size=0.5)
        + p9.scale_fill_manual(values=COLORS[:2], limits=[f"more {name_a}", f"more {name_b}"])
        + p9.labs(x="", y=f"{'G²' if by == 'g2' else 'log2 ratio'}  (left: more {name_b}  |  right: more {name_a})", fill="")
        + p9.theme_minimal()
        + p9.theme(figure_size=(6, max(3, 0.22 * len(data))), legend_position="top")
    )

