# python-for-poets
Course materials for "Python for Poets: introductory supervisions in digital humanities"

## The `python4poets` package

Small, readable tools for the course notebooks: loading corpora, counting words, finding distinctive words, keyword-in-context, and plotting.

```python
%pip install -q git+https://github.com/quadrismegistus/python-for-poets

import python4poets as p4p

df = p4p.load_sotu()
result = p4p.distinctive_words_by_group(df, "party", "Democrat", "Republican")
p4p.plot_distinctive_words(result)
p4p.kwic(df, "immigration")
p4p.plot_word_over_time(df, "immigration", color="party")
```

Type `p4p.` and press Tab to see every function; `p4p.kwic?` shows its help and `p4p.kwic??` its code.

Distinctive words are scored with log-likelihood (G²) for significance, Benjamini–Hochberg correction for testing many words at once, and Hardie's log ratio for effect size.
