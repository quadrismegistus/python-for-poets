"""Topic modelling."""

import numpy as np
import pandas as pd
import plotnine as p9
from scipy.stats import false_discovery_control, ttest_ind
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from ._plot import COLORS, MIN_POINTS_FOR_TREND
from ._text import get_stopwords, tokenize


def topic_model(
    df, n_topics=20, method="nmf", text_col="txt", min_texts=20, max_share=0.5,
    stopwords="english", extra_stopwords=(), seed=2026, n_words=10,
):
    """
    Find topics in a corpus: groups of words that tend to be used in the same texts.

    Each topic is a list of words, weighted by how typical they are of it.
    Each text is a mixture of topics: e.g. 60% topic 3, 30% topic 12, 10% others.

    Args:
        df: a DataFrame with one text per row.
        n_topics: how many topics to find (default 20).
        method: "nmf" (default: Non-negative Matrix Factorization; fast, and
            gives crisp topics on short texts) or "lda" (Latent Dirichlet
            Allocation, the classic topic model in digital humanities; several
            times slower).
        min_texts: ignore words used in fewer than this many texts (default 20).
        max_share: ignore words used in more than this share of texts
            (default 0.5: words in over half the texts say little about topic).
        stopwords: a language ("english", the default), a set of words, or None.
        extra_stopwords: more words to leave out, e.g. ["hon", "friend"].
        seed: for LDA, a number that fixes its random starting point, so the
            same settings give the same topics (default 2026). Change it to see
            how much the topics depend on chance. (NMF always starts from the
            same point, so the seed makes no difference to it.)
        n_words: how many words to show for each topic (default 10).

    Returns a TopicModel. Type `model.` and press Tab to see what it can do:
        model.topics                 one row per topic: its top words and overall share
        model.texts                  your DataFrame, plus each text's share of each topic
        model.top_words(3)           the words most typical of topic 3
        model.find_topic("nhs")      the number of the topic "nhs" belongs to most
        model.top_texts(3)           the texts with the most of topic 3
        model.text_topics(0)         the topics the text in row 0 is made of
        model.plot_over_time(3)      topic 3's share over time
        model.topics_by_group(col)   each topic's average share in each group
        model.distinctive_topics(col, a, b)   topics typical of group a vs b
        model.name_topic(3, "health")         give topic 3 a name for plots
        model.match(other_model)     which topics another model also found

    Example:
        >>> model = p4p.topic_model(df, n_topics=30)
        >>> model.topics
    """
    if method not in ("lda", "nmf"):
        raise ValueError(f'method must be "lda" or "nmf", not {method!r}')
    if isinstance(stopwords, str):
        stopwords = get_stopwords(stopwords)
    stopwords = frozenset(stopwords or ()) | {w.lower() for w in extra_stopwords}

    texts = df[text_col].fillna("")
    vectorizer = CountVectorizer(
        analyzer=lambda text: tokenize(text, stopwords=stopwords), min_df=min_texts, max_df=max_share
    )
    print(f"Counting words in {len(texts):,} texts ...")
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    print(f"Finding {n_topics} topics ({method.upper()}) using {len(vocab):,} different words ...")

    if method == "lda":
        print("(LDA can take several minutes on a big corpus; it prints a line after each of 10 passes.)")
        # n_jobs=2: two processes in parallel. More can be slower: each also
        # uses several processor cores of its own, and they compete.
        model = LatentDirichletAllocation(
            n_components=n_topics, learning_method="online", batch_size=2048, max_iter=10,
            random_state=seed, n_jobs=2, verbose=1,
        )
        # LDA learns from the texts in batches, in order: shuffle them first,
        # so it doesn't learn from (say) all the oldest texts first.
        model.fit(X[np.random.default_rng(seed).permutation(X.shape[0])])
        doc_topics = model.transform(X)
    else:
        model = NMF(n_components=n_topics, init="nndsvda", random_state=seed, max_iter=300)
        doc_topics = model.fit_transform(TfidfTransformer().fit_transform(X))

    # Make each text's topic weights add up to 1 (shares), and each topic's word weights too.
    row_sums = doc_topics.sum(axis=1, keepdims=True)
    doc_topics = np.divide(doc_topics, row_sums, out=np.zeros_like(doc_topics), where=row_sums > 0)
    topic_words = model.components_ / model.components_.sum(axis=1, keepdims=True)

    return TopicModel(df, doc_topics, topic_words, vocab, method, n_words, X.sum(axis=1).A1 > 0)


class TopicModel:
    """
    The topics found by p4p.topic_model. Type `model.` and press Tab to see
    what you can do with it, and add `?` to any of those for help.
    """

    def __init__(self, df, doc_topics, topic_words, vocab, method, n_words, has_words):
        self.method = method
        self.vocab = np.asarray(vocab)
        self.topic_words = topic_words
        self.n_topics = topic_words.shape[0]
        self.columns = [f"topic_{k}" for k in range(self.n_topics)]

        self.texts = df.copy()
        shares = pd.DataFrame(doc_topics, columns=self.columns, index=df.index)
        self.texts = pd.concat([self.texts, shares], axis=1)
        # A text with no counted words has no topics, not a "top" one.
        self.texts["top_topic"] = np.where(has_words, doc_topics.argmax(axis=1), -1)

        top = np.argsort(-topic_words, axis=1)[:, :n_words]
        self.topics = pd.DataFrame(
            {
                "topic": range(self.n_topics),
                "label": [f"{k}: {', '.join(self.vocab[top[k, :3]])}" for k in range(self.n_topics)],
                "words": [", ".join(self.vocab[row]) for row in top],
                "share": doc_topics[has_words].mean(axis=0),
            }
        ).set_index("topic", drop=False)
        self.topics.index.name = None

    def __repr__(self):
        return (
            f"<TopicModel: {self.n_topics} topics ({self.method.upper()}), "
            f"{len(self.texts):,} texts, {len(self.vocab):,} words. "
            f"See .topics and .texts; type `model.` and Tab for more.>"
        )

    def _label(self, topic):
        return self.topics.loc[topic, "label"]

    def name_topic(self, topic, name):
        """
        Give a topic a name, used in tables and plots from now on.

        Example:
            >>> model.name_topic(3, "health")
        """
        self.topics.loc[topic, "label"] = f"{topic}: {name}"

    def find_topic(self, word):
        """
        The number of the topic in which `word` has the most weight.

        Example:
            >>> health = model.find_topic("nhs")
            >>> model.top_words(health)
        """
        matches = np.flatnonzero(self.vocab == word.lower())
        if not len(matches):
            raise ValueError(f"{word!r} isn't one of the words the model used")
        return int(self.topic_words[:, matches[0]].argmax())

    def top_words(self, topic, n=20):
        """
        The words most typical of a topic, with their weights (out of 1).

        Example:
            >>> model.top_words(3)
        """
        weights = self.topic_words[topic]
        top = np.argsort(-weights)[:n]
        return pd.DataFrame({"word": self.vocab[top], "weight": weights[top]})

    def top_texts(self, topic, n=10):
        """
        The texts with the highest share of a topic: read these to understand it.

        Example:
            >>> model.top_texts(3)
        """
        column = f"topic_{topic}"
        result = self.texts.nlargest(n, column).drop(columns=self.columns + ["top_topic"])
        result.insert(0, "share", self.texts.loc[result.index, column])
        return result

    def text_topics(self, row, n=5):
        """
        The topics one text is made of, largest first.

        `row` is the text's row label in model.texts (and in your DataFrame).

        Example:
            >>> model.text_topics(0)
        """
        shares = self.texts.loc[row, self.columns].astype(float)
        top = shares.sort_values(ascending=False).head(n)
        return pd.DataFrame({"topic": [self._label(int(c[len("topic_"):])) for c in top.index], "share": top.to_numpy()})

    def topics_by_group(self, group_col):
        """
        Each topic's average share of the texts in each group.

        Returns one row per topic and one column per group.

        Example:
            >>> model.topics_by_group("party")
        """
        table = self.texts.groupby(group_col)[self.columns].mean().T
        table.index = [self._label(k) for k in range(self.n_topics)]
        table.columns.name = None
        return table

    def distinctive_topics(self, group_col, a, b, rows=None, alpha=0.05):
        """
        Find the topics that most distinguish texts where `group_col` is `a` from those where it is `b`.

        Like p4p.distinctive_words, but for topics: compares each topic's
        average share of the texts in each group. The significance test
        (Welch's t-test) treats each text as one piece of evidence, so one
        text that uses a topic a great deal can't swing the result.

        Args:
            rows: optionally, which texts to compare, e.g.
                model.texts["year"] >= 2010 (default: all of them).

        Returns a DataFrame, one row per topic, with columns:
            share_a, share_b   the topic's average share of a text in each group
            log_ratio          log2 of share_a / share_b: +1 means twice as much in a
            p, p_adj           the significance test, and p adjusted for testing every topic
            significant        p_adj < alpha

        With thousands of texts, even tiny differences come out significant:
        look at the size of the difference (log_ratio) too.

        Works with p4p.plot_distinctive_words.

        Example:
            >>> result = model.distinctive_topics("party", "Conservative", "Labour")
            >>> p4p.plot_distinctive_words(result)
            >>> model.distinctive_topics("gender", "female", "male", rows=model.texts["year"] >= 2010)
        """
        texts = self.texts if rows is None else self.texts[rows]
        shares_a = texts.loc[texts[group_col] == a, self.columns]
        shares_b = texts.loc[texts[group_col] == b, self.columns]
        for group, shares in ((a, shares_a), (b, shares_b)):
            if shares.empty:
                raise ValueError(f"No rows where {group_col!r} is {group!r}")
        mean_a, mean_b = shares_a.mean().to_numpy(), shares_b.mean().to_numpy()
        p = ttest_ind(shares_a.to_numpy(), shares_b.to_numpy(), equal_var=False, axis=0).pvalue

        result = pd.DataFrame(
            {
                "topic": range(self.n_topics),
                "label": [self._label(k) for k in range(self.n_topics)],
                f"share_{a}": mean_a,
                f"share_{b}": mean_b,
                "log_ratio": np.log2(mean_a / mean_b),
                "p": p,
                "p_adj": false_discovery_control(p),
            }
        )
        result["significant"] = result["p_adj"] < alpha
        return result.sort_values("log_ratio", ascending=False).reset_index(drop=True)

    def plot_over_time(self, topic, time_col="year", color=None):
        """
        Plot a topic's average share of the texts over time.

        `topic` can be one topic number or a list of them. With `color` (e.g.
        "party"), each group gets its own line.

        Example:
            >>> model.plot_over_time(3, color="party")
            >>> model.plot_over_time([3, 7])
        """
        topics = [topic] if np.isscalar(topic) else list(topic)
        keys = [time_col] + ([color] if color else [])
        data = (
            self.texts.groupby(keys)[[f"topic_{k}" for k in topics]].mean().reset_index()
            .melt(id_vars=keys, var_name="topic", value_name="share")
        )
        labels = [self._label(k) for k in topics]
        data["topic"] = pd.Categorical(data["topic"].map(dict(zip([f"topic_{k}" for k in topics], labels))), categories=labels)

        several = len(topics) > 1
        line_color = color or ("topic" if several else None)
        aes = p9.aes(x=time_col, y="share", color=line_color) if line_color else p9.aes(x=time_col, y="share")
        groups = [g for g in [line_color, "topic" if several else None] if g]
        trend_data = data.groupby(list(dict.fromkeys(groups)), observed=True).filter(
            lambda g: len(g) >= MIN_POINTS_FOR_TREND
        ) if groups else data

        fig = (
            p9.ggplot(data, aes)
            + p9.geom_point(size=1.5, alpha=0.6)
            + p9.geom_smooth(data=trend_data, method="lowess", span=0.4, method_args={"it": 0}, se=False, size=1)
            + p9.labs(x=time_col, y="average share of a text", title=labels[0] if not several else "")
            + p9.theme_minimal()
        )
        if line_color:
            values = labels if line_color == "topic" else sorted(data[line_color].dropna().unique())
            fig += p9.scale_color_manual(values=COLORS, limits=list(values)[: len(COLORS)], na_value="#aaaaaa")
        if several and color:
            fig += p9.facet_wrap("topic")
        return fig

    def match(self, other):
        """
        For each of this model's topics, the most similar topic in another model.

        Similarity (0 to 1) compares the two topics' word weights. Topics that
        both models find (similarity near 1) are more trustworthy than topics
        only one finds. Useful for comparing a different seed, number of
        topics, or method.

        Example:
            >>> model2 = p4p.topic_model(df, n_topics=30, seed=1)
            >>> model.match(model2)
        """
        # Line the two vocabularies up, then compare by cosine similarity.
        vocab = sorted(set(self.vocab) | set(other.vocab))
        index = {w: i for i, w in enumerate(vocab)}

        def aligned(model):
            matrix = np.zeros((model.n_topics, len(vocab)))
            matrix[:, [index[w] for w in model.vocab]] = model.topic_words
            return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        similarity = aligned(self) @ aligned(other).T
        best = similarity.argmax(axis=1)
        return pd.DataFrame(
            {
                "topic": [self._label(k) for k in range(self.n_topics)],
                "best_match": [other._label(j) for j in best],
                "similarity": similarity[np.arange(self.n_topics), best],
            }
        ).sort_values("similarity", ascending=False).reset_index(drop=True)
