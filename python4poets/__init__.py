"""
python4poets: small, readable tools for the Python for Poets course.

    import python4poets as p4p

Then type `p4p.` and press Tab to see everything. Add `?` to any function
for help (`p4p.kwic?`), or `??` to read its code (`p4p.kwic??`).

Loading corpora
    p4p.load_sotu()                  U.S. State of the Union speeches, with party
    p4p.load_novel450()              450 English, French and German novels
    p4p.load_hansard()               ~96,000 House of Commons speeches, 1979–2021
    p4p.load_corpus(csv, folder)     your own: a spreadsheet + a folder of .txt files
    p4p.search_gutenberg(author=...) find books on Project Gutenberg
    p4p.load_gutenberg(ids)          download Gutenberg books by number
    p4p.split_texts(df, n_words)     split texts into paragraphs, or parts of n words
    p4p.download(url)                download (and unzip) a file

Words
    p4p.tokenize(text)               split a text into words
    p4p.get_stopwords()              common words ("the", "of", ...) to leave out
    p4p.stopword_languages()         the languages get_stopwords has lists for
    p4p.download_nltk_data(name)     NLTK data (e.g. "punkt_tab"), kept in p4p_data/

Counting
    p4p.count_words(text)                    {word: count} for one text
    p4p.count_words_by_group(df, col)        {group: {word: count}}
    p4p.count_word_over_time(df, word)       a word's frequency in every text

Comparing
    p4p.distinctive_words(counts_a, counts_b)            words typical of a vs b
    p4p.distinctive_words_by_group(df, col, a, b)        the same, straight from a DataFrame
    p4p.crosstab_stats(df, col1, col2)                   are two metadata columns related?

Reading
    p4p.kwic(df, word)               every occurrence of a word, in context

Plotting
    p4p.plot_word_over_time(df, word)
    p4p.plot_distinctive_words(result)

Topics
    p4p.topic_model(df, n_topics)    find topics; then type `model.` and Tab

Word vectors
    p4p.load_word_vectors(name)              word2vec trained on all of Hansard; then `wv.similar(word)`
    p4p.compare_similar(word, [wv1, wv2])    a word's neighbours in two models
"""

from ._corpus import download, load_corpus, load_hansard, load_novel450, load_sotu
from ._counts import count_word_over_time, count_words, count_words_by_group
from ._distinctive import crosstab_stats, distinctive_words, distinctive_words_by_group
from ._gutenberg import gutenberg_catalog, load_gutenberg, search_gutenberg, split_texts
from ._kwic import kwic
from ._plot import plot_distinctive_words, plot_word_over_time
from ._text import download_nltk_data, get_stopwords, stopword_languages, tokenize
from ._topics import topic_model
from ._vectors import WordVectors, compare_similar, load_word_vectors

__version__ = "0.1.0"

__all__ = [
    "compare_similar",
    "count_word_over_time",
    "count_words",
    "count_words_by_group",
    "crosstab_stats",
    "distinctive_words",
    "distinctive_words_by_group",
    "download",
    "download_nltk_data",
    "get_stopwords",
    "gutenberg_catalog",
    "kwic",
    "load_corpus",
    "load_gutenberg",
    "load_hansard",
    "load_novel450",
    "load_sotu",
    "load_word_vectors",
    "plot_distinctive_words",
    "plot_word_over_time",
    "search_gutenberg",
    "split_texts",
    "stopword_languages",
    "tokenize",
    "topic_model",
    "WordVectors",
]
