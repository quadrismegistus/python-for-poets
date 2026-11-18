"""
Train word2vec models on the full House of Commons corpus, for p4p.load_word_vectors.

Input: hansard_full.parquet, written by scripts/prepare_hansard.py (every
speech of 50–1000 words by an MP with a party, 1979–2021: ~1.5M speeches,
~277M words).

For each subset of speeches below (all; each main party; each period), this:
    1. tokenizes every speech with p4p.tokenize, as in the course notebooks
       (lowercase, "don't" kept as one word, numbers dropped), and writes one
       speech per line to a plain-text corpus file;
    2. trains a word2vec model (gensim; skip-gram with negative sampling);
    3. keeps the vectors of the --n-words most frequent words and saves them,
       with each word's count, as a Parquet file: one row per word, columns
       word, count, v0 ... v99.

Speeches are not split into sentences: "hon." and "Mr." would break most of
them in the wrong places, and a speech is at most 1000 words anyway.

Requires gensim (pip install gensim), which the course package doesn't need:
students only download the finished vectors.

Usage (from the repo root):
    python scripts/train_word2vec.py
    python scripts/train_word2vec.py --models all conservative --workers 8

Outputs (in data/hansard/word2vec/ by default):
    corpus/{name}.txt              tokenized speeches, one per line
    hansard_w2v_{name}.parquet     the vectors to upload
"""

import argparse
import os
import time

import pandas as pd
from gensim.models import Word2Vec

import python4poets as p4p

# Model name -> which speeches it is trained on (a pandas query on hansard_full).
MODELS = {
    "all": None,
    "conservative": "party_main == 'Conservative'",
    "labour": "party_main == 'Labour'",
    "1979-1996": "year <= 1996",
    "1997-2009": "1997 <= year <= 2009",
    "2010-2021": "year >= 2010",
}


def write_corpora(df, names, folder):
    """Tokenize every speech once, writing it to the corpus file of each model it belongs to."""
    os.makedirs(folder, exist_ok=True)
    masks = {name: (df.eval(MODELS[name]) if MODELS[name] else pd.Series(True, index=df.index)) for name in names}
    files = {name: open(os.path.join(folder, f"{name}.txt"), "w", encoding="utf-8") for name in names}
    try:
        for i, (idx, text) in enumerate(df["txt"].items()):
            line = " ".join(p4p.tokenize(text)) + "\n"
            for name in names:
                if masks[name].at[idx]:
                    files[name].write(line)
            if i % 200_000 == 0:
                print(f"  tokenized {i:,} of {len(df):,} speeches", flush=True)
    finally:
        for f in files.values():
            f.close()
    for name in names:
        print(f"  {name}: {int(masks[name].sum()):,} speeches", flush=True)


def train(corpus_path, args):
    return Word2Vec(
        corpus_file=corpus_path,
        vector_size=args.dims,
        window=args.window,
        min_count=args.min_count,
        sg=1,
        negative=5,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )


def save_vectors(model, path, n_words):
    wv = model.wv
    words = wv.index_to_key[:n_words]  # gensim sorts the vocabulary by frequency
    vectors = pd.DataFrame(wv[words].astype("float32"), columns=[f"v{i}" for i in range(wv.vector_size)])
    vectors.insert(0, "count", [wv.get_vecattr(w, "count") for w in words])
    vectors.insert(0, "word", words)
    vectors.to_parquet(path, index=False)
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/hansard/hansard_full.parquet")
    parser.add_argument("--out", default="data/hansard/word2vec")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--dims", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-words", type=int, default=50_000, help="how many of the most frequent words to keep")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reuse-corpus", action="store_true", help="skip tokenizing if the corpus files exist")
    args = parser.parse_args()

    corpus_folder = os.path.join(args.out, "corpus")
    paths = {name: os.path.join(corpus_folder, f"{name}.txt") for name in args.models}
    if not (args.reuse_corpus and all(os.path.exists(p) for p in paths.values())):
        print(f"Reading {args.input} ...", flush=True)
        df = pd.read_parquet(args.input, columns=["year", "party_main", "txt"])
        print(f"Tokenizing {len(df):,} speeches ...", flush=True)
        write_corpora(df, args.models, corpus_folder)
        del df

    for name in args.models:
        t = time.time()
        print(f"Training {name} ...", flush=True)
        model = train(paths[name], args)
        out_path = os.path.join(args.out, f"hansard_w2v_{name}.parquet")
        vectors = save_vectors(model, out_path, args.n_words)
        size = os.path.getsize(out_path) / 1e6
        print(
            f"  {name}: {model.corpus_total_words / 1e6:.0f}M words, vocabulary {len(model.wv):,}; "
            f"kept {len(vectors):,} words (min count {vectors['count'].min():,}); "
            f"{time.time() - t:.0f}s -> {out_path} ({size:.0f} MB)",
            flush=True,
        )


if __name__ == "__main__":
    main()
