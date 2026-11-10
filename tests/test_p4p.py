import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

import python4poets as p4p


@pytest.fixture
def corpus():
    return pd.DataFrame(
        {
            "id": ["d1", "d2", "r1", "r2"],
            "year": [1990, 2000, 1995, 2005],
            "party": ["D", "D", "R", "R"],
            "txt": [
                "Health care for working families. Don't cut health care!",
                "Working families need health care and jobs.",
                "Lower taxes. Freedom and lower taxes for the nation.",
                "Taxes, freedom, and the nation’s strength.",
            ],
        }
    )


# --- namespace -------------------------------------------------------------


def test_namespace_shows_only_public_functions():
    public = [name for name in dir(p4p) if not name.startswith("_")]
    assert sorted(public) == sorted(p4p.__all__)


# --- tokenize --------------------------------------------------------------


def test_tokenize_keeps_contractions_and_drops_punctuation():
    assert p4p.tokenize("I don't know—do you?") == ["i", "don't", "know", "do", "you"]


def test_tokenize_curly_apostrophe_matches_straight():
    assert p4p.tokenize("don’t") == p4p.tokenize("don't") == ["don't"]


def test_tokenize_options():
    assert p4p.tokenize("In 1776 we", numbers=True) == ["in", "1776", "we"]
    assert p4p.tokenize("In 1776 we") == ["in", "we"]
    assert p4p.tokenize("The Cat", lower=False) == ["The", "Cat"]
    assert p4p.tokenize("the cat", stopwords={"the"}) == ["cat"]


def test_tokenize_non_english_letters():
    assert p4p.tokenize("Fräulein von Sternheim, élève") == ["fräulein", "von", "sternheim", "élève"]


def test_stopwords_contractions_are_normalized():
    stops = p4p.get_stopwords()
    assert "the" in stops and "don't" in stops
    assert p4p.tokenize("don't stop", stopwords="english") == ["stop"]


# --- counting --------------------------------------------------------------


def test_count_words_returns_independent_copies():
    c1 = p4p.count_words("a a b")
    c1["a"] = 100
    assert p4p.count_words("a a b")["a"] == 2


def test_count_words_by_group(corpus):
    counts = p4p.count_words_by_group(corpus, "party")
    assert counts["D"]["health"] == 3
    assert counts["R"]["taxes"] == 3
    assert counts["R"]["nation's"] == 1


def test_count_words_by_group_skips_missing_groups(corpus):
    corpus.loc[0, "party"] = None
    assert set(p4p.count_words_by_group(corpus, "party")) == {"D", "R"}


def test_count_word_over_time(corpus):
    result = p4p.count_word_over_time(corpus, "Health")
    assert list(result["year"]) == [1990, 1995, 2000, 2005]
    assert list(result["count"]) == [2, 0, 1, 0]
    assert "txt" not in result.columns
    row = result.iloc[0]
    assert row["fpm"] == pytest.approx(2 / row["total"] * 1_000_000)


def test_count_word_over_time_several_words(corpus):
    result = p4p.count_word_over_time(corpus, ["health", "taxes"])
    assert len(result) == 8
    assert set(result["word"]) == {"health", "taxes"}


# --- distinctive words -----------------------------------------------------


def test_g2_matches_scipy():
    a = Counter({"x": 30, "y": 5, "z": 965})
    b = Counter({"x": 10, "y": 20, "z": 1970})
    result = p4p.distinctive_words(a, b, min_count=1).set_index("word")
    for word in "xyz":
        table = [[a[word], b[word]], [1000 - a[word], 2000 - b[word]]]
        g2, p, _, _ = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        assert result.loc[word, "g2"] == pytest.approx(g2)
        assert result.loc[word, "p"] == pytest.approx(p)


def test_log_ratio_and_obs_exp():
    a = Counter({"x": 20, "other": 980})
    b = Counter({"x": 10, "other": 990})
    row = p4p.distinctive_words(a, b, min_count=1).set_index("word").loc["x"]
    assert row["log_ratio"] == pytest.approx(1.0)  # twice as frequent
    assert row["obs_exp_a"] == pytest.approx(20 / 15)
    assert row["fpm_a"] == pytest.approx(20_000)


def test_words_used_by_only_one_group_are_kept():
    a = Counter({"only_a": 50, "shared": 500})
    b = Counter({"shared": 500})
    result = p4p.distinctive_words(a, b, min_count=1)
    row = result.set_index("word").loc["only_a"]
    assert np.isfinite(row["log_ratio"]) and row["log_ratio"] > 0
    assert row["significant"]
    assert result.iloc[0]["word"] == "only_a"


def test_min_count_uses_totals_before_filtering():
    a = Counter({"rare": 1, "common": 99})
    b = Counter({"common": 100})
    result = p4p.distinctive_words(a, b, min_count=5)
    assert list(result["word"]) == ["common"]
    assert result.iloc[0]["fpm_a"] == pytest.approx(99 / 100 * 1_000_000)


def test_sort_by_signed_g2():
    a = Counter({"big_a": 300, "small_a": 12, "small_b": 8, "big_b": 100, "rest": 10_000})
    b = Counter({"big_a": 100, "small_a": 8, "small_b": 12, "big_b": 300, "rest": 10_000})
    result = p4p.distinctive_words(a, b, min_count=1, sort="g2")
    assert result.iloc[0]["word"] == "big_a"
    assert result.iloc[-1]["word"] == "big_b"
    with pytest.raises(ValueError):
        p4p.distinctive_words(a, b, sort="nonsense")


def test_distinctive_words_by_group(corpus):
    result = p4p.distinctive_words_by_group(corpus, "party", "D", "R", min_count=1)
    assert {"count_D", "count_R", "fpm_D", "fpm_R", "obs_exp_D"} <= set(result.columns)
    assert result.iloc[0]["log_ratio"] > 0 > result.iloc[-1]["log_ratio"]
    with pytest.raises(ValueError):
        p4p.distinctive_words_by_group(corpus, "party", "D", "Whig")


def test_crosstab_stats_matches_scipy():
    df = pd.DataFrame({"g": list("mmmmffff" * 5), "l": list("eeefeffe" * 5)})
    chi2, p, obs_exp = p4p.crosstab_stats(df, "g", "l")
    expected_chi2, expected_p, _, expected = chi2_contingency(pd.crosstab(df["g"], df["l"]))
    assert chi2 == pytest.approx(expected_chi2) and p == pytest.approx(expected_p)
    assert obs_exp.loc["m", "e"] == pytest.approx(15 / expected[1][0])
    g2, _, _ = p4p.crosstab_stats(df, "g", "l", method="g2")
    assert g2 != chi2


# --- kwic ------------------------------------------------------------------


def test_kwic_string_keeps_punctuation_and_case():
    result = p4p.kwic("One. Two, Three four five! Six", "four", window=2)
    assert result.to_dict("records") == [{"left": "Two, Three", "keyword": "four", "right": "five! Six"}]


def test_kwic_phrase_and_edges():
    text = "The United States of America. united   states!"
    result = p4p.kwic(text, "united states", window=3)
    assert list(result["keyword"]) == ["United States", "united states"]
    assert result.iloc[0]["left"] == "The"
    assert result.iloc[1]["right"] == ""


def test_kwic_curly_apostrophes():
    result = p4p.kwic("I don’t know.", "don't")
    assert list(result["keyword"]) == ["don’t"]


def test_kwic_dataframe_keeps_metadata(corpus):
    result = p4p.kwic(corpus, "health")
    assert list(result.columns) == ["id", "left", "keyword", "right", "year", "party"]
    assert list(result["id"]) == ["d1", "d1", "d2"]
    empty = p4p.kwic(corpus, "zebra")
    assert len(empty) == 0 and "left" in empty.columns


# --- corpus loading --------------------------------------------------------


def test_load_corpus_matches_unicode_filenames(tmp_path):
    (tmp_path / "txt").mkdir()
    # Filename in decomposed form (as zips made on a Mac store it), id in composed form.
    (tmp_path / "txt" / (unicodedata.normalize("NFD", "Fräulein") + ".txt")).write_text("hallo")
    pd.DataFrame({"id": [unicodedata.normalize("NFC", "Fräulein"), "missing"]}).to_csv(tmp_path / "m.csv", index=False)
    with pytest.warns(UserWarning, match="1 of 2"):
        df = p4p.load_corpus(tmp_path / "m.csv", tmp_path / "txt")
    assert list(df["txt"]) == ["hallo", ""]


# --- plots -----------------------------------------------------------------


def test_plots_render(corpus):
    p4p.plot_word_over_time(corpus, "health", color="party").draw()
    p4p.plot_word_over_time(corpus, ["health", "taxes"]).draw()
    p4p.plot_word_over_time(corpus, ["health", "taxes"], color="party").draw()
    result = p4p.distinctive_words_by_group(corpus, "party", "D", "R", min_count=1)
    p4p.plot_distinctive_words(result, significant_only=False).draw()


def test_load_hansard_reads_downloaded_file(tmp_path):
    # With the file already in the folder, nothing is downloaded.
    pd.DataFrame({"id": ["a"], "year": [1979], "txt": ["Order, order."]}).to_parquet(
        tmp_path / "hansard_sample.parquet"
    )
    df = p4p.load_hansard(folder=tmp_path)
    assert df["txt"].tolist() == ["Order, order."]
