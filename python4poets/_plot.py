"""Plotting word frequencies."""

import pandas as pd
import plotnine as p9

from ._counts import count_word_over_time

# Colors for groups, always used in this order (blue, orange, aqua, ...),
# chosen to stay distinguishable for colorblind readers.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

MIN_POINTS_FOR_TREND = 10


def plot_word_over_time(df, word, time_col="year", color=None, text_col="txt", **tokenize_kw):
    """
    Plot how often a word is used (per million words) in each text over time.

    Args:
        df: a DataFrame with one text per row.
        word: one word, or a list of words to compare.
        time_col: the column for the x-axis (default "year").
        color: a column to color points by, e.g. "party".

    With a list of words, each word gets its own color, or (if `color` is
    also set) its own panel.

    Example:
        >>> p4p.plot_word_over_time(df, "immigration", color="party")
        >>> p4p.plot_word_over_time(df, ["war", "peace"])
    """
    data = count_word_over_time(df, word, time_col=time_col, text_col=text_col, **tokenize_kw)
    several_words = not isinstance(word, str)
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


def plot_distinctive_words(result, n=15, significant_only=True):
    """
    Bar chart of the most distinctive words for each group, from p4p.distinctive_words.

    Bars show the log ratio: how many times more frequent (on a log2 scale:
    1 = twice as frequent, 2 = four times) a word is in one group than the other.

    Args:
        result: the DataFrame returned by p4p.distinctive_words or
            p4p.distinctive_words_by_group.
        n: how many words to show for each group (default 15).
        significant_only: only show statistically significant words (default True).

    Example:
        >>> result = p4p.distinctive_words_by_group(df, "party", "Democrat", "Republican")
        >>> p4p.plot_distinctive_words(result)
    """
    name_a, name_b = [c[len("fpm_") :] for c in result.columns if c.startswith("fpm_")]
    data = result[result["significant"]] if significant_only else result
    top_a = data[data["log_ratio"] > 0].nlargest(n, "log_ratio")
    top_b = data[data["log_ratio"] < 0].nsmallest(n, "log_ratio")
    data = (
        pd.concat([top_a.assign(group=f"more {name_a}"), top_b.assign(group=f"more {name_b}")])
        .sort_values("log_ratio")
    )
    data["word"] = pd.Categorical(data["word"], categories=list(dict.fromkeys(data["word"])))

    return (
        p9.ggplot(data, p9.aes(x="word", y="log_ratio", fill="group"))
        + p9.geom_col(width=0.8)
        + p9.coord_flip()
        + p9.geom_hline(yintercept=0, color="#888888", size=0.5)
        + p9.scale_fill_manual(values=COLORS[:2], limits=[f"more {name_a}", f"more {name_b}"])
        + p9.labs(x="", y=f"log2 ratio  (left: more {name_b}  |  right: more {name_a})", fill="")
        + p9.theme_minimal()
        + p9.theme(figure_size=(6, max(3, 0.22 * len(data))), legend_position="top")
    )

