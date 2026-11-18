"""Word vectors: words as points in space, placed by the company they keep."""

import os

import numpy as np
import pandas as pd

from ._corpus import DATA_FOLDER, download

# Made by scripts/train_word2vec.py: a zip of hansard_w2v_{name}.parquet files.
HANSARD_W2V_URL = "https://www.dropbox.com/scl/fi/9lod98z8dfigxulwstmok/hansard_w2v.zip?rlkey=ywt83w7id7trwls8z2b8wl3xp&dl=1"
HANSARD_W2V_MODELS = ["all", "conservative", "labour", "1979-1996", "1997-2009", "2010-2021"]


def load_word_vectors(name="all", folder=os.path.join(DATA_FOLDER, "hansard_w2v")):
    """
    Load word vectors (word2vec) trained on the full House of Commons corpus.

    `name` picks which speeches the model was trained on:
        "all"           every speech, 1979–2021 (~277 million words)
        "conservative"  Conservative MPs' speeches only
        "labour"        Labour MPs' speeches only
        "1979-1996", "1997-2009", "2010-2021"   one period only

    Each model has every word used at least 20 times in its speeches (from
    ~28,000 words in one period to ~46,000 in "all"). The models were trained
    with scripts/train_word2vec.py, on every speech of 50–1000 words by an MP
    with a party (the corpus that p4p.load_hansard samples from).

    Downloads the vectors the first time; later calls reuse them.

    Example:
        >>> wv = p4p.load_word_vectors()
        >>> wv.similar("fairness")
    """
    if name not in HANSARD_W2V_MODELS:
        raise ValueError(f"name must be one of {HANSARD_W2V_MODELS}, not {name!r}")
    path = os.path.join(folder, f"hansard_w2v_{name}.parquet")
    if not os.path.exists(path):
        if HANSARD_W2V_URL is None:
            raise FileNotFoundError(f"{path} doesn't exist, and there is no download link for the vectors yet.")
        download(HANSARD_W2V_URL, folder, "hansard_w2v.zip")
    return WordVectors.from_parquet(path, name=name)


class WordVectors:
    """
    A word2vec model: one vector (a list of numbers) per word. Words used in
    similar contexts get similar vectors. Type `wv.` and press Tab to see what
    you can do with it.
    """

    def __init__(self, words, vectors, counts=None, name=""):
        self.name = name
        self.words = pd.DataFrame({"word": list(words), "count": counts if counts is not None else np.nan})
        self._index = {w: i for i, w in enumerate(self.words["word"])}
        vectors = np.asarray(vectors, dtype="float32")
        # Unit length, so that a dot product is a cosine similarity.
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    @classmethod
    def from_parquet(cls, path, name=""):
        df = pd.read_parquet(path)
        columns = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
        return cls(df["word"], df[columns].to_numpy(), df["count"].to_numpy(), name=name)

    def __repr__(self):
        name = f" {self.name!r}" if self.name else ""
        return f"<WordVectors{name}: {len(self.words):,} words, {self.vectors.shape[1]} dimensions. Try .similar('word').>"

    def __contains__(self, word):
        return word in self._index

    def vector(self, word):
        """The (unit-length) vector of a word."""
        if word not in self._index:
            raise KeyError(f"{word!r} isn't one of the {len(self.words):,} words in this model")
        return self.vectors[self._index[word]]

    def similarity(self, a, b):
        """
        How similar two words are: the cosine of the angle between their vectors
        (1 = same direction; 0 = unrelated).

        Example:
            >>> wv.similarity("fairness", "equality")
        """
        return float(self.vector(a) @ self.vector(b))

    def _nearest(self, target, n, exclude=()):
        scores = self.vectors @ (target / np.linalg.norm(target))
        result = []
        for i in np.argsort(-scores):
            word = self.words.at[i, "word"]
            if word not in exclude:
                result.append((word, float(scores[i]), self.words.at[i, "count"]))
                if len(result) == n:
                    break
        return pd.DataFrame(result, columns=["word", "similarity", "count"])

    def similar(self, word, n=15):
        """
        The words whose vectors are closest to `word`'s: words used in similar
        contexts (`count` is how often each was used in the training speeches).

        Example:
            >>> wv.similar("fairness")
        """
        return self._nearest(self.vector(word), n, exclude={word})

    def analogy(self, a, b, c, n=10):
        """
        "a is to b as c is to ...?": the words closest to b - a + c.

        Example:
            >>> wv.analogy("man", "woman", "king")
        """
        target = self.vector(b) - self.vector(a) + self.vector(c)
        return self._nearest(target, n, exclude={a, b, c})


def compare_similar(word, models, n=15):
    """
    Compare a word's neighbours in two or more models (e.g. Conservative vs. Labour).

    Takes the `n` nearest neighbours of `word` in each model, and gives each
    one's similarity to `word` in every model (empty if a model doesn't have
    it). With two models, `difference` is the first's similarity minus the
    second's, and the table is sorted by it: the top rows are neighbours
    closer in the first model, the bottom rows closer in the second.

    Example:
        >>> con = p4p.load_word_vectors("conservative")
        >>> lab = p4p.load_word_vectors("labour")
        >>> p4p.compare_similar("fairness", [con, lab])
    """
    names = [m.name or f"model_{i}" for i, m in enumerate(models)]
    if len(set(names)) < len(names):
        raise ValueError("Give each model a different name (e.g. wv.name = 'Labour')")
    neighbours = []
    for model in models:
        if word in model:
            neighbours += [w for w in model.similar(word, n)["word"] if w not in neighbours]
    if not neighbours:
        raise KeyError(f"{word!r} isn't in any of the models")
    table = pd.DataFrame({"word": neighbours})
    for name, model in zip(names, models):
        table[name] = [model.similarity(word, w) if word in model and w in model else np.nan for w in neighbours]
    if len(models) == 2:
        table["difference"] = table[names[0]] - table[names[1]]
        table = table.sort_values("difference", ascending=False)
    else:
        table = table.sort_values(names, ascending=False)
    return table.reset_index(drop=True)
