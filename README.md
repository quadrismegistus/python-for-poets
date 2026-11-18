# Python for Poets

Course materials for "Python for Poets: introductory supervisions in digital humanities" (University of Cambridge), and `python4poets`, the small Python package the course notebooks use.

The package gives short, readable functions for the methods taught in the course: loading corpora, counting words, finding the words and topics that distinguish groups of texts, reading words in context, plotting, and exploring word meanings with word vectors. Everything else is ordinary [pandas](https://pandas.pydata.org/docs/user_guide/10min.html).

```python
import python4poets as p4p

df = p4p.load_sotu()                     # a corpus: one text per row, the text in a "txt" column
result = p4p.distinctive_words_by_group(df, "party", "Democrat", "Republican")
p4p.plot_distinctive_words(result)       # the words most typical of each party
p4p.kwic(df, "freedom")                  # every use of "freedom", in context
```

**Contents**

- [Getting started](#getting-started): in Google Colab, or on your own computer
- [How do I...?](#how-do-i): short answers to common questions, with code
- [The corpora](#the-corpora): what each built-in dataset contains
- [Reference](#reference): every function, its options, and what it returns
- [How the methods work](#how-the-methods-work), with references
- [Notes for AI coding assistants](#notes-for-ai-coding-assistants)
- [The workbooks](#the-workbooks)
- [For maintainers](#for-maintainers)

---

## Getting started

### In Google Colab

Put this in the first cell of a notebook:

```python
%pip install -q git+https://github.com/quadrismegistus/python-for-poets
import python4poets as p4p
```

To get the newest version after the package has changed, run the line below, then choose **Runtime → Restart session**, then run your notebook again from the top:

```python
%pip install -q --force-reinstall --no-deps git+https://github.com/quadrismegistus/python-for-poets
```

### On your own computer

1. Install **Python 3.13** from [python.org](https://www.python.org/downloads/release/python-31317/). On Windows, tick **"Add python.exe to PATH"** on the installer's first screen. (On a Mac, `python4poets` downloads its data without the extra "Install Certificates" step. If *other* packages fail to download with `CERTIFICATE_VERIFY_FAILED`, open the *Python 3.13* folder in Applications and double-click **Install Certificates.command**.)

   (The package works on any Python from 3.10. We recommend 3.13 because some other packages you might want, such as `gensim`, don't yet support 3.14 or 3.15.)

2. In a terminal, make a folder for your project, and a *virtual environment* in it (a private set of packages for this project):

   ```bash
   # Mac
   mkdir my_project && cd my_project
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

   ```powershell
   # Windows
   mkdir my_project
   cd my_project
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   ```

   If Windows says "running scripts is disabled on this system", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try `.venv\Scripts\activate` again.

3. Install the package (this works without `git`). `[all]` adds Jupyter (for notebooks) and Hugging Face `datasets`; leave it out for just the package:

   ```bash
   pip install "python4poets[all] @ https://github.com/quadrismegistus/python-for-poets/archive/refs/heads/main.zip"
   ```

4. Check that everything works:

   ```bash
   python -m python4poets
   ```

   This prints one line per check (Python version, downloads, plotting, and the optional extras), with advice for anything that fails.

Corpora and other data are downloaded the first time you load them, into a folder called `p4p_data` inside whatever folder you're working in. Later loads reuse them.

### With an AI coding assistant (Antigravity, Gemini, Claude Code, ...)

Make an empty folder, open it in your assistant, and paste this prompt. It sets up the folder step by step, and saves [`AGENTS.md`](setup/AGENTS.md), a file of rules the assistant will read in every later conversation in this folder (use Python 3.13 in `.venv`, use this package, keep downloads inside the folder, explain each step).

```text
I'm a humanities student learning Python in a course called Python for Poets. Please set up this folder for the course, following these steps exactly. Before each step, tell me in a sentence what you're about to do and why. Don't install anything else, don't use sudo, and don't change anything outside this folder.

1. Make a virtual environment called .venv in this folder, with Python 3.13 (Mac: python3.13 -m venv .venv; Windows: py -3.13 -m venv .venv). If Python 3.13 isn't installed, stop and tell me.

2. Using the .venv's Python, install the course package with Jupyter and Hugging Face datasets:
   python -m pip install "python4poets[all] @ https://github.com/quadrismegistus/python-for-poets/archive/refs/heads/main.zip"

3. Download these two files into this folder, keeping their names, and read both. AGENTS.md has the rules for all our future work here.
   https://raw.githubusercontent.com/quadrismegistus/python-for-poets/main/README.md
   https://raw.githubusercontent.com/quadrismegistus/python-for-poets/main/setup/AGENTS.md

4. Using the .venv's Python, run: python -m python4poets
   Show me its output. If any line says FAIL, tell me what it says; don't try to fix it yourself.

Then stop and summarise what you did.
```

The installation and downloads need internet access: if your assistant asks to allow network access for a command, say yes.

### Getting help inside Python

```python
help(p4p.kwic)      # the help text of any function
p4p.kwic?           # the same, in a notebook
p4p.kwic??          # the function's code, in a notebook
```

Type `p4p.` and press Tab to see every function.

---

## How do I...?

The examples use the course corpora; swap in your own DataFrame and column names. A *DataFrame* here always means a table with **one text per row**, the text in a column called `txt`, and other columns (`author`, `year`, ...) describing it.

### Getting texts

**...load one of the course corpora?**

```python
df = p4p.load_sotu()        # U.S. State of the Union addresses, 1790–2018
df = p4p.load_novel450()    # 450 English, French and German novels, 1771–1930
df = p4p.load_hansard()     # ~96,000 House of Commons speeches, 1979–2021
```

**...load my own texts, with a spreadsheet describing them?**

Make a spreadsheet (saved as CSV) with an `id` column and any others you like (`author`, `year`, ...), and a folder with one `.txt` file per row, named by its id (`id` "behn_oroonoko" → `behn_oroonoko.txt`):

```python
df = p4p.load_corpus("metadata.csv", "txt")
```

**...load a folder of .txt files, without a spreadsheet?**

```python
from pathlib import Path
import pandas as pd

df = pd.DataFrame([
    {"id": path.stem, "txt": path.read_text(encoding="utf-8")}
    for path in sorted(Path("my_texts").glob("*.txt"))
])
```

**...find books on Project Gutenberg?**

```python
p4p.search_gutenberg(author="behn")                 # by author
p4p.search_gutenberg(title="frankenstein")          # by title
p4p.search_gutenberg(subject="gothic")              # by subject or "bookshelf"
p4p.search_gutenberg("utopia", language="fr")       # anywhere, in French
```

Each result has an `id`: the book's Gutenberg number (also in its web address, e.g. gutenberg.org/ebooks/**51783**).

**...download books from Project Gutenberg?**

```python
books = p4p.load_gutenberg([51783, 29854])          # The Blazing World; The Works of Aphra Behn, vol. V
books[["id", "author", "title"]]
```

Gutenberg's header and licence are removed. Check what each file really contains before comparing books: a "Works" volume may hold many texts, plus an editor's introduction and notes.

**...split long texts into paragraphs, or into chunks?**

```python
paragraphs = p4p.split_texts(books)                 # one row per paragraph
chunks = p4p.split_texts(books, n_words=500)        # parts of ~500 words, split at paragraph breaks
chunks.groupby("title").size()                      # how many parts each book gave
```

Each part keeps its book's columns (`author`, `title`, ...), plus `part`, its number within the book.

### Looking at metadata

**...count how many texts there are in each group?**

```python
df = p4p.load_sotu()
df["party"].value_counts()
df.groupby("party")["year"].agg(["min", "max", "count"])
```

**...test whether two columns are related (e.g. gender and party)?**

```python
hansard = p4p.load_hansard()
chi2, p, obs_exp = p4p.crosstab_stats(hansard, "gender", "party_main")
obs_exp        # observed / expected: above 1 = more common than chance would predict
```

### Counting words

**...split a text into words?**

```python
p4p.tokenize("I don't know—do you?")                     # ['i', "don't", 'know', 'do', 'you']
p4p.tokenize("The cat sat on the mat", stopwords="english")   # ['cat', 'sat', 'mat']
```

**...find the most common words in a text?**

```python
text = df.loc[0, "txt"]
p4p.count_words(text, stopwords="english").most_common(20)
```

**...find the most common words in each group?**

```python
counts = p4p.count_words_by_group(df, "party", stopwords="english")
counts["Democrat"].most_common(20)
```

**...leave out common words ("the", "of"), or some of my own?**

```python
stops = p4p.get_stopwords("english") | {"applause", "mr"}
p4p.count_words(text, stopwords=stops).most_common(20)
```

Other languages: `p4p.get_stopwords("french")`, `"german"`, `"spanish"`, ... (`p4p.stopword_languages()` lists all 33).

**...see how often a word is used over time?**

```python
p4p.plot_word_over_time(df, "immigration")                    # one point per year
p4p.plot_word_over_time(df, "immigration", color="party")     # one color per party
p4p.plot_word_over_time(df, ["war", "peace"])                 # several words
p4p.count_word_over_time(df, "immigration")                   # the numbers, one row per text
```

**...read a word in context?**

```python
p4p.kwic(df, "liberty")                          # every use, with 8 words either side
p4p.kwic(df, "united states", window=5)          # phrases work too
p4p.kwic(df[df["year"] > 2000], "liberty")       # only some texts
p4p.kwic(df, "liberty").sample(20)               # 20 at random
```

### Comparing groups of texts

**...find the words that distinguish two groups?**

```python
result = p4p.distinctive_words_by_group(df, "party", "Democrat", "Republican")
p4p.plot_distinctive_words(result)
result.head(20)       # words most typical of Democrats
result.tail(20)       # words most typical of Republicans
```

**...stop one text (or one speaker) dominating the results?**

In a corpus of many short texts, a word one speaker repeats in a single speech can look highly "significant". `min_texts` keeps only words used in at least that many texts:

```python
hansard = p4p.load_hansard()
result = p4p.distinctive_words_by_group(hansard, "party_main", "Conservative", "Labour", min_texts=20)
```

**...see the differences we're most sure of, rather than the biggest?**

```python
p4p.plot_distinctive_words(result, by="g2")      # ranked by G² (confidence), not log ratio (size)
```

**...compare two long texts, like two novels?**

Split them into parts first, so the statistics have more than one observation per book, and use `min_texts` so a word must turn up in several parts:

```python
books = p4p.load_gutenberg([51783, 29854])
chunks = p4p.split_texts(books, n_words=500)
result = p4p.distinctive_words_by_group(chunks, "author", "Newcastle, Margaret Cavendish, Duchess of", "Behn, Aphra", min_texts=5)
p4p.plot_distinctive_words(result)
```

**...compare one group against everything else?**

Make a column with two values, then compare them:

```python
df["group"] = df["party"].where(df["party"] == "Democrat", "other")
result = p4p.distinctive_words_by_group(df, "group", "Democrat", "other")
```

**...compare only some of the texts (e.g. since 2010)?**

Filter the DataFrame first:

```python
recent = hansard[hansard["year"] >= 2010]
result = p4p.distinctive_words_by_group(recent, "gender", "female", "male", min_texts=20)
```

**...save a table of results?**

```python
result.to_csv("distinctive_words.csv", index=False)
```

### Topics

**...find the topics in a corpus?**

```python
model = p4p.topic_model(hansard, n_topics=30)
model.topics          # one row per topic: its top words, and its share of an average text
```

On ~100,000 short texts this takes about a minute. With few texts, split them first (`p4p.split_texts`), and lower `min_texts` (by default a word must be in 20 texts to count).

**...understand what a topic is about?**

```python
topic = model.find_topic("nhs")      # the topic a word belongs to most
model.top_words(topic)               # its words, with weights
model.top_texts(topic)               # the texts with the most of it: read these!
model.name_topic(topic, "health")    # name it, for tables and plots
```

**...see which topics a text is made of?**

```python
model.text_topics(0)                 # the text in row 0
```

**...compare topics between groups?**

```python
model.topics_by_group("party_main")                                   # average share in each group
result = model.distinctive_topics("party_main", "Conservative", "Labour")
p4p.plot_distinctive_words(result)
```

**...see how a topic changes over time?**

```python
model.plot_over_time(topic)                              # one topic
model.plot_over_time(topic, color="party_main")          # by group
model.plot_over_time([topic, model.find_topic("europe")])   # several topics
```

**...check whether my topics are trustworthy?**

Fit another model with different settings, and see which topics come back:

```python
other = p4p.topic_model(hansard, n_topics=30, method="lda", seed=1)
model.match(other)       # for each topic, its most similar topic in the other model (1 = identical)
```

Also try different numbers of topics, and leaving out more words (`extra_stopwords=["hon", "friend"]`).

### Word meanings (word vectors)

**...find words used like a given word?**

```python
wv = p4p.load_word_vectors()          # trained on all Commons speeches, 1979–2021
wv.similar("fairness")
wv.similarity("fairness", "equality")
```

**...see whether a word means something different to different groups, or in different periods?**

```python
con = p4p.load_word_vectors("conservative")
lab = p4p.load_word_vectors("labour")
p4p.compare_similar("austerity", [con, lab])

early = p4p.load_word_vectors("1979-1996")
late = p4p.load_word_vectors("2010-2021")
p4p.compare_similar("community", [early, late])
```

Then read the word in context (`p4p.kwic`) in each group or period, to see whether the neighbours match how it is really used.

**...train word vectors on my own texts?**

Word vectors need a lot of text: millions of words, ideally tens of millions. On a few novels, the "nearest neighbours" are mostly noise. The course's vectors were made with [`scripts/train_word2vec.py`](scripts/train_word2vec.py), which uses the `gensim` package (which, as of November 2026, needs Python 3.13 or earlier).

### Plots

All plots are [plotnine](https://plotnine.org/) plots, which work like R's ggplot2.

**...change a plot's title or size?**

```python
import plotnine as p9

fig = p4p.plot_word_over_time(df, "immigration", color="party")
fig = fig + p9.labs(title="'Immigration' in the State of the Union") + p9.theme(figure_size=(10, 5))
fig
```

**...save a plot as an image?**

```python
fig.save("immigration.png", dpi=300)
```

---

## The corpora

| loader | what | size | columns |
|---|---|---|---|
| `p4p.load_sotu()` | U.S. State of the Union addresses, 1790–2018 | 228 speeches | id, author, title, year, president, party, txt |
| `p4p.load_novel450()` | txtLAB [Novel450](https://txtlab.org/novel450/): novels in English, French and German, 1771–1930 | 450 novels | id, id_orig, author, title, year, language, date, gender, person, length, txt |
| `p4p.load_hansard()` | House of Commons speeches, 1979–2021 (a sample) | ~96,000 speeches, ~18M words | see below |
| `p4p.load_gutenberg(ids)` | any book on [Project Gutenberg](https://www.gutenberg.org/) | ~78,000 books | id, title, author, authors, language, issued, subjects, bookshelves, locc, txt |
| `p4p.load_word_vectors(name)` | word2vec models trained on all Commons speeches | 28,000–46,000 words each | (a `WordVectors` object) |

### Hansard

Each year has 1,000 Conservative and 1,000 Labour speeches, chosen at random; the other parties are sampled at the same rate, so they keep their real share of each year's speeches. Only speeches of 50–1000 words by MPs with a party are included. Because of the equal sampling, **the sample can't tell you which party spoke more**.

| column | meaning |
|---|---|
| date, year, decade | when the speech was given |
| speaker, mnis_id, gender | who gave it (`mnis_id`: Parliament's ID for the MP) |
| party, party_main | party as recorded; and grouped into Conservative, Labour, Liberal Democrat, SNP, Other |
| side | "government" or "opposition" on that date |
| frontbench, post | whether the MP held a ministerial or shadow post that day, and which |
| constituency | the MP's seat |
| major_heading, minor_heading | the debate's topic headings |
| word_count | the speech's length in words |
| id, txt | the speech's ID in Hansard, and the speech |

Cautions: opposition (shadow) posts are under-recorded before about 2000, so `frontbench` is reliable for the government side throughout but for the opposition only from around then. Party, side and period are tangled together (the Conservatives were in government for 29 of the 42 years), so a "party" difference may really be about government vs. opposition, or about when the speeches were given; compare within one side or one period to check.

Sources: Evan Odell, ["Hansard Speeches 1979–2021: Version 3.1.0"](https://doi.org/10.5281/zenodo.4843485) (Zenodo, 2021, CC-BY 4.0), built on mySociety's parlparse; gender and posts from the [UK Parliament Members API](https://members-api.parliament.uk/).

### Word vectors

| name | trained on | words |
|---|---|---|
| `"all"` | every speech, 1979–2021 (~277M words) | 46,222 |
| `"conservative"` | Conservative MPs' speeches (~134M words) | 35,654 |
| `"labour"` | Labour MPs' speeches (~111M words) | 33,450 |
| `"1979-1996"` | 1979–1996 (~111M words) | 32,120 |
| `"1997-2009"` | 1997–2009 (~84M words) | 28,492 |
| `"2010-2021"` | 2010–2021 (~82M words) | 28,590 |

Trained on every speech of 50–1000 words by an MP with a party (the corpus `p4p.load_hansard()` samples from), with gensim's word2vec (skip-gram, 100 dimensions, window of 5 words, 5 passes). Each model includes every word used at least 20 times in its speeches. Words were split with `p4p.tokenize`, so they match the words you count in the notebooks.

### Project Gutenberg

Books come from Gutenberg's mirror servers (first the University of Kent's, in the UK), not its website, as Project Gutenberg asks of programs. The catalog that `p4p.search_gutenberg` searches is Gutenberg's own (`pg_catalog.csv`). By default you get a snapshot of it made for the course (18 November 2026), which downloads in seconds; books added to Gutenberg since then won't turn up in searches, but `p4p.load_gutenberg` can still download them by number. For Gutenberg's latest catalog, run `p4p.gutenberg_catalog(latest=True)` once (a 20 MB download, which can be slow).

---

## Reference

Every function is called as `p4p.function_name(...)`. Arguments with `=` have a default and can be left out.

### Loading texts

#### `load_sotu(folder="p4p_data/sotu")`

Load the U.S. State of the Union corpus (1790–2018), with each speaker's party. Returns a DataFrame with columns id, author, title, year, president, party, txt.

#### `load_novel450(folder="p4p_data/novel450")`

Load the txtLAB Novel450 corpus: 450 novels in English, French and German (1771–1930). Returns a DataFrame with columns id, id_orig, author, title, year, language, date, gender, person, length, txt.

#### `load_hansard(folder="p4p_data/hansard")`

Load a sample of ~96,000 House of Commons speeches (1979–2021). Returns a DataFrame; see [Hansard](#hansard) for its columns and cautions.

#### `load_corpus(metadata_csv, txt_folder, id_col="id", text_col="txt")`

Load your own corpus: a spreadsheet (CSV) of metadata, plus one `.txt` file per row in `txt_folder`, named `{id}.txt`. Returns the spreadsheet as a DataFrame, with each file's text in a new column `text_col`. Rows with no matching file get empty text, with a warning.

#### `load_gutenberg(ids, folder="p4p_data/gutenberg")`

Download books from Project Gutenberg. `ids` is one Gutenberg number or a list of them. Returns a DataFrame with one row per book: the catalog's columns (id, title, author, authors, language, issued, subjects, bookshelves, locc) and `txt`, the book's text with Gutenberg's header and licence removed.

#### `search_gutenberg(query="", author="", title="", subject="", language="en", folder="p4p_data/gutenberg")`

Search the Gutenberg catalog, ignoring upper/lower case. `query` searches title, authors, subjects and bookshelves together; `author`, `title` and `subject` search one column each (`subject` covers subjects and bookshelves). `language` is a code such as `"en"`, `"fr"`, `"de"`, or `""` for any. Returns the matching rows of the catalog.

#### `gutenberg_catalog(folder="p4p_data/gutenberg", latest=False)`

The whole Gutenberg catalog of texts (~78,000 rows) as a DataFrame, for searching with pandas. Columns: id, title, author (the first author's name, e.g. "Behn, Aphra"), authors (all of them, with dates and roles such as "[Editor]"), language, issued, subjects, bookshelves, locc (Library of Congress class). Uses the course's snapshot of the catalog; `latest=True` downloads Gutenberg's current one instead (and keeps it for later calls).

#### `split_texts(df, n_words=None, text_col="txt")`

Split each text into paragraphs (separated by blank lines), or, with `n_words`, into parts of at least `n_words` words made of whole paragraphs (a text's last part may be shorter). Returns a DataFrame with one row per part, keeping each text's other columns, plus `part` (its number within the text, from 0).

#### `download(url, folder="p4p_data", filename=None)`

Download a file into `folder`, unless it's already there; zip files are unzipped. Returns the file's path. Refuses to save a web page (e.g. a "file not found" page from a broken link) in place of the file you asked for.

### Words

#### `tokenize(text, lower=True, numbers=False, stopwords=None)`

Split a text into a list of words. Words keep internal apostrophes ("don't" stays one word); hyphenated words are split ("well-known" → "well", "known"); punctuation is dropped. `lower`: lowercase every word. `numbers`: keep numbers such as "1776". `stopwords`: words to leave out: a set, or a language name such as `"english"`.

#### `get_stopwords(language="english")`

NLTK's list of very common words for a language, as a frozenset. Add your own with `|`: `p4p.get_stopwords() | {"applause"}`. The lists (33 languages) come with the package, so nothing is downloaded.

#### `download_nltk_data(*names, folder="p4p_data/nltk_data")`

Download NLTK data (e.g. `"punkt_tab"`, which `nltk.word_tokenize` needs) into `folder`, and tell NLTK to look there; skips anything already downloaded. Use it instead of `nltk.download(...)`: it keeps the data inside your project folder, and works on a Mac without the "Install Certificates" step.

#### `stopword_languages()`

The languages `get_stopwords` has lists for: arabic, chinese, english, french, german, greek, italian, spanish, ... (33 in all).

#### `count_words(text, lower=True, numbers=False, stopwords=None)`

Count each word in a text. Takes the same options as `tokenize`. Returns a `Counter` (a dictionary of {word: count} with a `.most_common(n)` method). Results are remembered, so counting the same text again is instant.

#### `count_words_by_group(df, group_col, text_col="txt", **tokenize_kw)`

Add up the word counts of all the texts in each group. Returns a dictionary of {group: Counter}. Extra options (e.g. `stopwords="english"`) are passed to `tokenize`.

#### `count_word_over_time(df, word, time_col="year", text_col="txt", **tokenize_kw)`

Count a word (or each of a list of words) in every text. Returns one row per text (per word), with the text's metadata plus columns word, count (times used), total (words in the text) and fpm (frequency per million words), sorted by `time_col`.

#### `kwic(texts, word, window=8, text_col="txt")`

"Keyword in context": every occurrence of a word or phrase, with `window` words on each side, ignoring upper/lower case. `texts` is one string or a DataFrame. Returns a DataFrame with columns left, keyword, right (and, for a DataFrame, each text's metadata).

### Comparing groups

#### `distinctive_words_by_group(df, group_col, a, b, text_col="txt", min_count=5, min_texts=1, alpha=0.05, sort="log_ratio", **tokenize_kw)`

The words that most distinguish texts where `group_col` is `a` from those where it is `b`. Counts the words in each group, then runs `distinctive_words` (below). Column names use the group names (count_Democrat, fpm_Republican, ...). Also counts how many texts in each group use each word, so `min_texts` can leave out words used in fewer texts than that. Extra options (e.g. `stopwords="english"`) are passed to `tokenize`.

#### `distinctive_words(counts_a, counts_b, names=("a", "b"), min_count=5, alpha=0.05, sort="log_ratio", texts_a=None, texts_b=None, min_texts=1)`

The same, from two sets of word counts (e.g. two Counters from `count_words_by_group`). `min_count`: leave out words used fewer times than this in total. `alpha`: the threshold for `significant`. `sort`: `"log_ratio"` (by the size of the difference) or `"g2"` (by confidence that it is real). `texts_a`, `texts_b`: optional {word: number of texts using it} for each group, which add the texts_ columns and make `min_texts` work.

Returns a DataFrame, one row per word, words most typical of a at the top and of b at the bottom:

| column | meaning |
|---|---|
| count_a, count_b | how many times each group uses the word |
| texts_a, texts_b | how many texts in each group use it (if known) |
| fpm_a, fpm_b | frequency per million words in each group |
| obs_exp_a | group a's count divided by what we'd expect if the word were spread evenly (above 1 = overused by a) |
| log_ratio | log2 of fpm_a / fpm_b: +1 = twice as frequent in a, −1 = twice as frequent in b, 0 = the same |
| g2 | log-likelihood (G²): how surprising the difference is |
| p | the chance of a difference at least this big if there were really none |
| p_adj | p, corrected for testing thousands of words at once (Benjamini–Hochberg) |
| significant | p_adj < alpha |

#### `crosstab_stats(df, col1, col2, method="chi2")`

Cross-tabulate two metadata columns and test whether they're related. Returns three things: the chi-squared statistic (or G², with `method="g2"`), its p-value, and a table of observed / expected counts (above 1 = more common than chance would predict).

### Plotting

#### `plot_word_over_time(df, word, time_col="year", color=None, per_text=False, text_col="txt", **tokenize_kw)`

Plot a word's frequency (per million words) over time, with a smoothed trend line (when there are at least 10 points). Each point is one year (or one value of `time_col`): the word's frequency in all that year's texts together, or in each group's texts if `color` names a column. `word` can be a list: each word then gets its own color (or, with `color`, its own panel). `per_text=True` plots one point per text instead (fine for a few hundred texts). Returns a plotnine plot.

#### `plot_distinctive_words(result, n=15, significant_only=True, by="log_ratio")`

Bar chart of the `n` most distinctive words for each group, from `distinctive_words`, `distinctive_words_by_group` or `model.distinctive_topics`. `by="log_ratio"` shows the biggest differences (which favours rarer words); `by="g2"` shows the differences we're most sure of (which favours commoner words). `significant_only`: leave out words that aren't significant. Returns a plotnine plot.

### Topics

#### `topic_model(df, n_topics=20, method="nmf", text_col="txt", min_texts=20, max_share=0.5, stopwords="english", extra_stopwords=(), seed=2026, n_words=10)`

Find topics: groups of words that tend to be used in the same texts. Each text becomes a mixture of topics (e.g. 60% topic 3, 30% topic 12, 10% others).

- `method`: `"nmf"` (default: Non-negative Matrix Factorization, on tf-idf weights; fast, and crisp on short texts) or `"lda"` (Latent Dirichlet Allocation, on raw counts; the classic in digital humanities; slower, and different with each `seed`).
- `min_texts`: ignore words used in fewer texts than this. `max_share`: ignore words used in more than this share of texts.
- `stopwords`: a language, a set of words, or None. `extra_stopwords`: more words to leave out.
- `seed`: fixes LDA's random start, so the same settings give the same topics (NMF doesn't need it).
- `n_words`: how many words to show per topic.

Returns a `TopicModel`:

| attribute or method | what it gives |
|---|---|
| `model.topics` | one row per topic: topic, label, words (top `n_words`), share (of an average text) |
| `model.texts` | your DataFrame, plus each text's share of each topic (`topic_0`, `topic_1`, ...) and `top_topic` |
| `model.find_topic(word)` | the number of the topic in which `word` has the most weight |
| `model.top_words(topic, n=20)` | the topic's words, with weights |
| `model.top_texts(topic, n=10)` | the texts with the highest share of the topic (read these to understand it) |
| `model.text_topics(row, n=5)` | the topics the text in row `row` is made of, largest first |
| `model.name_topic(topic, name)` | give a topic a name, used in tables and plots from then on |
| `model.topics_by_group(group_col)` | each topic's average share in each group |
| `model.distinctive_topics(group_col, a, b, rows=None, alpha=0.05)` | topics typical of group a vs. b: columns topic, label, share_{a}, share_{b} (named after the groups, e.g. share_Labour), log_ratio, p, p_adj, significant. `rows` (a True/False column, e.g. `model.texts["year"] >= 2010`) compares only some texts. Uses Welch's t-test on each text's share, so one prolific speaker can't swing it |
| `model.plot_over_time(topic, time_col="year", color=None)` | a topic's (or list of topics') average share over time |
| `model.match(other)` | for each topic, the most similar topic in another model, with their similarity (1 = identical) |

### Word vectors

#### `load_word_vectors(name="all", folder="p4p_data/hansard_w2v")`

Load word vectors trained on all House of Commons speeches 1979–2021: `"all"`, `"conservative"`, `"labour"`, `"1979-1996"`, `"1997-2009"` or `"2010-2021"` (see [Word vectors](#word-vectors)). Returns a `WordVectors` object:

| method | what it gives |
|---|---|
| `wv.similar(word, n=15)` | the `n` words whose vectors are closest to `word`'s (words used in similar contexts), with similarity and count |
| `wv.similarity(a, b)` | how similar two words are: the cosine of the angle between their vectors (1 = same direction, 0 = unrelated) |
| `wv.analogy(a, b, c, n=10)` | "a is to b as c is to ...?": the words closest to b − a + c |
| `wv.vector(word)` | the word's vector (unit length) |
| `word in wv` | whether the model has the word |
| `wv.words` | every word in the model, with its count |

#### `compare_similar(word, models, n=15)`

Compare a word's neighbours in two or more models. Takes the `n` nearest neighbours of `word` in each model and gives each one's similarity to `word` in every model (empty if a model lacks it). With two models, `difference` is the first's similarity minus the second's, and rows are sorted by it: neighbours closer in the first model at the top, in the second at the bottom.

#### `WordVectors(words, vectors, counts=None, name="")`

Make your own `WordVectors` from a list of words and a matching table (array) of vectors, e.g. from a model you trained yourself.

---

## How the methods work

**Distinctive words.** For each word, the package compares its frequency in two groups. *Log ratio* measures how big the difference is (Hardie 2014): log2 of the ratio of the two frequencies. *G²* (log-likelihood; Dunning 1993) measures how surprising the difference would be if the word were really used equally by both, and gives the p-value. Because thousands of words are tested at once, some would look significant by chance; *p_adj* corrects for this (Benjamini and Hochberg 1995). These tests treat every word as an independent draw, which language isn't: once a speaker says "Spitalfields", they are likely to say it again. In corpora of many short texts this makes one person's repeated word look significant; `min_texts` guards against it (Lijffijt et al. 2016).

**Topics.** A topic model treats each text as a "bag of words" and looks for groups of words that keep turning up in the same texts. NMF (Lee and Seung 1999) factors the table of texts × words (weighted by tf-idf: Spärck Jones 1972) into texts × topics and topics × words. LDA (Blei, Ng and Jordan 2003) imagines each text was written by picking topics, then words from each topic, and infers the topics most likely to have produced the texts. Topics depend on the method, the number of topics, the words left out and, for LDA, chance: trust topics that come back across settings (`model.match`).

**Word vectors.** word2vec (Mikolov et al. 2013) gives each word a list of 100 numbers, adjusted until words used in similar contexts (within 5 words either side) have similar lists. Words that could fill the same slot in a sentence end up close together, whether or not they are about the same thing ("immigration" is close to "sentencing": both fill "the ___ Bill"). Neighbours of rare words, and on small corpora, are unstable (Antoniak and Mimno 2018); read the word in context before trusting them.

References:

- Antoniak, M., and D. Mimno (2018). "Evaluating the Stability of Embedding-based Word Similarities." *Transactions of the Association for Computational Linguistics* 6: 107–119.
- Benjamini, Y., and Y. Hochberg (1995). "Controlling the False Discovery Rate." *Journal of the Royal Statistical Society B* 57(1): 289–300.
- Blei, D., A. Ng, and M. Jordan (2003). "Latent Dirichlet Allocation." *Journal of Machine Learning Research* 3: 993–1022.
- Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." *Computational Linguistics* 19(1): 61–74.
- Hardie, A. (2014). "Log Ratio: An Informal Introduction." ESRC Centre for Corpus Approaches to Social Science (CASS).
- Lee, D., and H. S. Seung (1999). "Learning the Parts of Objects by Non-negative Matrix Factorization." *Nature* 401: 788–791.
- Lijffijt, J., et al. (2016). "Significance Testing of Word Frequencies in Corpora." *Digital Scholarship in the Humanities* 31(2): 374–397.
- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
- Spärck Jones, K. (1972). "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." *Journal of Documentation* 28(1): 11–21.

---

## Notes for AI coding assistants

If you are an AI assistant (Gemini, Claude, Copilot, ...) helping a student with this package, please:

- **Import it as `import python4poets as p4p`**, and use its functions for what they cover (loading corpora and Gutenberg books, tokenizing, counting, distinctive words, keyword in context, topics, word vectors, plots) rather than writing your own versions: they are the methods taught in the course, and students are expected to recognize them. Use plain pandas for everything else.
- **Keep to its data shape**: a pandas DataFrame with one text per row, the text in a column called `txt`, and metadata in other columns. Most functions take such a DataFrame plus column names.
- **Write short, readable code, and explain it in plain language.** The students are humanities students learning Python; prefer a few clear steps over clever one-liners, and say what each step does and why.
- **For Project Gutenberg, use `p4p.search_gutenberg` and `p4p.load_gutenberg`**: they use Gutenberg's mirrors (as Gutenberg asks), remove its header and licence, and cache downloads. Don't scrape gutenberg.org, and don't install `gutenbergpy`, `gutenberg` or similar.
- **For word vectors, use `p4p.load_word_vectors`** (pre-trained on 277 million words of Hansard). Don't train word2vec on a handful of texts: the results are noise. If a student needs `gensim` for something else, it requires Python 3.13 or earlier.
- **Before comparing groups, check them**: how many texts and words each group has (`df.groupby(col).size()`), and whether the texts are what their titles say (Gutenberg "Works" volumes contain several texts, often with an editor's introduction and notes).
- **For a few long texts, split them first** (`p4p.split_texts(df, n_words=500)`) and pass `min_texts` to `distinctive_words_by_group`, so one passage can't dominate the result.
- **Always look at examples in context** (`p4p.kwic`) before interpreting a distinctive word, topic or word-vector neighbour, and encourage the student to do so.
- **Plots are plotnine (ggplot) objects**: modify with `+` (e.g. `+ p9.labs(title=...)`), save with `.save("name.png")`.
- **Don't change system certificate settings** (no `sudo`, no symlinking certificate files) to make p4p's downloads work: it brings its own certificates (`certifi`). If a p4p download fails, report the error to the student instead.
- **Keep downloads inside the student's folder.** For stopwords use `p4p.get_stopwords()` (no download), not `nltk.download("stopwords")`. For other NLTK data, `p4p.download_nltk_data(name)` (e.g. `"punkt_tab"`), not `nltk.download(name)`. For Hugging Face datasets, set `os.environ["HF_HOME"] = "p4p_data/huggingface"` before `import datasets`. Assistants' sandboxes often block writing elsewhere (e.g. `~/nltk_data`).
- **Data is cached in `p4p_data/`** in the working directory. It can be large; don't commit it to git.
- Each function's full documentation is in its docstring: `help(p4p.function_name)`.

---

## The workbooks

The course notebooks are in [`notebooks/`](notebooks/). Click a workbook to open it in Colab (File → Save a copy in Drive to keep your changes).

| workbook | topic |
|---|---|
| [1A](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%201A%20-%20Intro%20to%20strings%20and%20functions.ipynb) | Python basics: strings and functions |
| [1B](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%201B%20-%20Letters%20and%20numbers.ipynb) | Letters and numbers |
| [1C](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%201C%20-%20Lists%20and%20dictionaries.ipynb) | Lists and dictionaries |
| [1D](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%201D%20-%20Whens%20and%20ifs.ipynb) | Conditions ("whens and ifs") |
| [2](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%202%20-%20Analysing%20metadata.ipynb) | Analysing metadata (Novel450) |
| [3](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%203%20-%20Finding%20distinctive%20words%20%28v2%29.ipynb) | Finding distinctive words (State of the Union) |
| [4A](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%204A%20-%20Speeches%20in%20Parliament.ipynb) | Speeches in Parliament: metadata and distinctive words (Hansard) |
| [4B](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%204B%20-%20Topics%20in%20Parliament.ipynb) | Topics in Parliament (Hansard) |
| [4C](https://colab.research.google.com/github/quadrismegistus/python-for-poets/blob/main/notebooks/Python%20for%20Poets%202026%20-%20Workbook%204C%20-%20Practicum.ipynb) | Practicum: follow a word through Hansard |

---

## For maintainers

| path | what |
|---|---|
| `python4poets/` | the package (one module per area: `_corpus`, `_gutenberg`, `_text`, `_counts`, `_distinctive`, `_kwic`, `_plot`, `_topics`, `_vectors`) |
| `notebooks/` | the course workbooks |
| `scripts/prepare_hansard.py` | builds the Hansard sample (and the full filtered corpus) from Odell's dataset and the Members API |
| `scripts/train_word2vec.py` | trains the word2vec models on the full corpus (needs `gensim`, Python ≤ 3.13) |
| `setup/AGENTS.md` | the rules file the AI-assistant setup prompt downloads into students' folders |
| `python4poets/stopwords/` | NLTK's stopword lists (see its README.txt for sources), bundled so nothing is downloaded |
| `tests/` | tests: `pip install -e . pytest`, then `pytest` |

Large data files are hosted on Dropbox; their links are constants at the top of `python4poets/_corpus.py` and `python4poets/_vectors.py`. If you replace a file, upload the new version and update the link.

Licence: GPL-3.0 (see [LICENSE](LICENSE)). The corpora have their own licences and credits, given above.
