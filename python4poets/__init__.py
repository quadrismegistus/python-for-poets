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
    p4p.download(url)                download (and unzip) a file

Words
    p4p.tokenize(text)               split a text into words
    p4p.get_stopwords()              common words ("the", "of", ...) to leave out

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
"""

from ._corpus import download, load_corpus, load_hansard, load_novel450, load_sotu
from ._counts import count_word_over_time, count_words, count_words_by_group
from ._distinctive import crosstab_stats, distinctive_words, distinctive_words_by_group
from ._kwic import kwic
from ._plot import plot_distinctive_words, plot_word_over_time
from ._text import get_stopwords, tokenize

__version__ = "0.1.0"

__all__ = [
    "count_word_over_time",
    "count_words",
    "count_words_by_group",
    "crosstab_stats",
    "distinctive_words",
    "distinctive_words_by_group",
    "download",
    "get_stopwords",
    "kwic",
    "load_corpus",
    "load_hansard",
    "load_novel450",
    "load_sotu",
    "plot_distinctive_words",
    "plot_word_over_time",
    "tokenize",
]
