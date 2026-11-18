"""Searching and downloading books from Project Gutenberg."""

import os
import re
import urllib.error
import warnings

import pandas as pd

from ._corpus import DATA_FOLDER, download
from ._net import urlopen

# Where to get the catalog of every book, tried in order. First, the course's
# snapshot of it (pg_catalog.csv, zipped: ~6 MB, and quick to download); then
# Gutenberg's live copy, which is sometimes down for maintenance; then
# pglaf.org, Gutenberg's own file server (complete, but it can take 15 minutes).
GUTENBERG_CATALOG_SNAPSHOT_URL = "https://www.dropbox.com/scl/fi/4gj07xdp0udkyiqhbbqqc/pg_catalog.zip?rlkey=lner2bvlpswhr3410y2xmnh45&dl=1"  # snapshot of 18 November 2026
GUTENBERG_CATALOG_URLS = [
    "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
    "https://gutenberg.pglaf.org/cache/epub/feeds/pg_catalog.csv",
]

# Where to get the books themselves: Gutenberg asks programs to use mirrors,
# not its website. The first is in the UK (University of Kent), and fast.
GUTENBERG_MIRRORS = [
    "https://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg",
    "https://gutenberg.pglaf.org",
]

GUTENBERG_FOLDER = os.path.join(DATA_FOLDER, "gutenberg")

CATALOG_COLUMNS = {
    "Text#": "id", "Type": "type", "Issued": "issued", "Title": "title", "Language": "language",
    "Authors": "authors", "Subjects": "subjects", "LoCC": "locc", "Bookshelves": "bookshelves",
}


def gutenberg_catalog(folder=GUTENBERG_FOLDER, latest=False):
    """
    The catalog of every book on Project Gutenberg (~78,000), as a DataFrame.

    Columns: id, title, author, authors, language, issued, subjects,
    bookshelves, locc. `author` is the first author's name; `authors` lists
    all of them, with their dates and roles (e.g. "[Editor]").

    Downloads the catalog the first time (a snapshot made for the course, so
    books added to Gutenberg since then are missing; p4p.load_gutenberg can
    still download them by number); later calls reuse it. To get Gutenberg's
    latest catalog instead, use `latest=True` (a 20 MB download, which can be slow).

    Example:
        >>> catalog = p4p.gutenberg_catalog()
        >>> catalog[catalog["author"] == "Austen, Jane"]
    """
    path = os.path.join(folder, "pg_catalog.csv")
    if latest and os.path.exists(path):
        os.remove(path)
    if not os.path.exists(path):
        sources = [(url, "pg_catalog.csv") for url in GUTENBERG_CATALOG_URLS]
        if GUTENBERG_CATALOG_SNAPSHOT_URL and not latest:
            sources.insert(0, (GUTENBERG_CATALOG_SNAPSHOT_URL, "pg_catalog.zip"))  # unzips to pg_catalog.csv
        _download_first(sources, folder)
    df = pd.read_csv(path, dtype=str, keep_default_na=False).rename(columns=CATALOG_COLUMNS)
    df = df[df["type"] == "Text"].drop(columns="type")
    df["id"] = df["id"].astype(int)
    # Long titles are split over several lines in the catalog.
    for col in ["title", "authors", "subjects", "bookshelves"]:
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    df.insert(2, "author", df["authors"].map(_first_author))
    return df.reset_index(drop=True)


def search_gutenberg(query="", author="", title="", subject="", language="en", folder=GUTENBERG_FOLDER):
    """
    Search the Project Gutenberg catalog. Every search ignores upper/lower case.

    Args:
        query: words to find anywhere: in the title, authors, subjects or bookshelves.
        author, title, subject: words to find in that column only
            (`subject` searches both subjects and bookshelves).
        language: a language code, e.g. "en", "fr", "de"; or "" for any.

    Returns the matching rows of p4p.gutenberg_catalog(). Pass the `id` of a
    book you want to p4p.load_gutenberg.

    Example:
        >>> p4p.search_gutenberg(author="behn")
        >>> p4p.search_gutenberg("gothic", language="en")
    """
    df = gutenberg_catalog(folder)
    keep = pd.Series(True, index=df.index)
    if query:
        everything = df["title"] + " " + df["authors"] + " " + df["subjects"] + " " + df["bookshelves"]
        keep &= everything.str.contains(query, case=False, regex=False)
    if author:
        keep &= df["authors"].str.contains(author, case=False, regex=False)
    if title:
        keep &= df["title"].str.contains(title, case=False, regex=False)
    if subject:
        subjects = df["subjects"] + " " + df["bookshelves"]
        keep &= subjects.str.contains(subject, case=False, regex=False)
    if language:
        keep &= df["language"].str.split("; ").map(lambda codes: language in codes)
    return df[keep].reset_index(drop=True)


def load_gutenberg(ids, folder=GUTENBERG_FOLDER):
    """
    Download books from Project Gutenberg, as a DataFrame with one row per book.

    `ids` is a book's Gutenberg number (from its web address, or from
    p4p.search_gutenberg), or a list of them. Gutenberg's header and licence
    are removed, leaving the book's own text in the `txt` column; the other
    columns come from the catalog (see p4p.gutenberg_catalog).

    Books are downloaded the first time; later calls reuse them.

    Example:
        >>> df = p4p.load_gutenberg([51783, 29854])
        >>> df[["id", "author", "title"]]
    """
    if isinstance(ids, (int, str)):
        ids = [ids]
    ids = [int(i) for i in ids]
    texts = [_load_book(book_id, folder) for book_id in ids]

    try:
        catalog = gutenberg_catalog(folder).set_index("id")
        missing = [i for i in ids if i not in catalog.index]
        if missing:
            warnings.warn(f"Not in the catalog (it may be newer than your copy): {missing}", stacklevel=2)
        df = catalog.reindex(ids).fillna("").reset_index()
    except (OSError, RuntimeError, urllib.error.URLError) as err:
        warnings.warn(f"Couldn't get the catalog, so the books have no titles or authors: {err}", stacklevel=2)
        df = pd.DataFrame({"id": ids})
    df["txt"] = texts
    return df


def split_texts(df, n_words=None, text_col="txt"):
    """
    Split each text into paragraphs, or into parts of at least `n_words` words.

    Paragraphs are separated by blank lines. With `n_words`, consecutive
    paragraphs are joined until each part has at least that many words (the
    last part of a text may be shorter). Every part keeps its text's other
    columns, plus `part`: its number within the text, from 0.

    Splitting a long text into parts gives the statistics more than one
    observation per text: e.g. `min_texts` in p4p.distinctive_words_by_group
    then counts parts, so a word used in only one passage can be left out.

    Example:
        >>> parts = p4p.split_texts(df, n_words=500)
        >>> parts.groupby("title").size()
    """
    rows = []
    for _, row in df.iterrows():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", _normalize_newlines(row[text_col]))]
        paragraphs = [p for p in paragraphs if p]
        if n_words:
            paragraphs = _join_until(paragraphs, n_words)
        for part, text in enumerate(paragraphs):
            rows.append({**row.drop(text_col).to_dict(), "part": part, text_col: text})
    return pd.DataFrame(rows, columns=[c for c in df.columns if c != text_col] + ["part", text_col])


# --- helpers ----------------------------------------------------------------

# "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***", and older
# variants ("THIS PROJECT", "E-BOOK", no stars).
START_PATTERN = re.compile(r"^\W*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK.*$", re.IGNORECASE | re.MULTILINE)
END_PATTERN = re.compile(
    r"^\W*(?:END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK|End of (?:the )?Project Gutenberg).*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_gutenberg_header(text):
    """Remove Project Gutenberg's header and licence from a book's text, if they're there."""
    text = _normalize_newlines(text).lstrip("﻿")
    start = START_PATTERN.search(text)
    if start:
        text = text[start.end():]
    end = END_PATTERN.search(text)
    if end:
        text = text[: end.start()]
    text = text.strip()
    # The volunteers' credit line ("Produced by ...") isn't part of the book.
    first, _, rest = text.partition("\n\n")
    if re.match(r"(?:Produced by|E-?text prepared by|Transcribed from)\b", first, re.IGNORECASE):
        text = rest.strip()
    return text + "\n"


def _load_book(book_id, folder):
    path = os.path.join(folder, f"{book_id}.txt")
    if not os.path.exists(path):
        print(f"Downloading Gutenberg book {book_id} ...")
        text = strip_gutenberg_header(_fetch_book(book_id))
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fetch_book(book_id):
    # Mirrors keep book 12345 in 1/2/3/4/12345/ (and books 1-9 in 0/).
    s = str(book_id)
    subfolder = "/".join(s[:-1]) if len(s) > 1 else "0"
    # 12345-0.txt is UTF-8; 12345-8.txt Latin-1; 12345.txt ASCII (older books).
    candidates = [(f"{mirror}/{subfolder}/{s}/{s}{suffix}", encoding)
                  for mirror in GUTENBERG_MIRRORS
                  for suffix, encoding in [("-0.txt", "utf-8"), ("-8.txt", "latin-1"), (".txt", "latin-1")]]
    candidates.append((f"https://www.gutenberg.org/cache/epub/{s}/pg{s}.txt", "utf-8"))
    for url, encoding in candidates:
        try:
            with urlopen(url) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError):
            continue
        if data.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
            continue  # an error page, not a book
        return data.decode(encoding, errors="replace")
    raise RuntimeError(f"Couldn't download Gutenberg book {book_id}. Check the number at https://www.gutenberg.org/ebooks/{s}")


def _download_first(sources, folder):
    # sources: [(url, filename to save it as), ...]; the first that works wins.
    errors = []
    for url, filename in sources:
        try:
            return download(url, folder, filename)
        except (OSError, RuntimeError, urllib.error.URLError) as err:
            errors.append(f"{url}: {err}")
    raise RuntimeError("Couldn't download the Gutenberg catalog from any source:\n" + "\n".join(errors))


def _first_author(authors):
    # "Behn, Aphra, 1640-1689; Summers, Montague, 1880-1948 [Editor]" -> "Behn, Aphra"
    first = authors.split("; ")[0]
    first = re.sub(r"\s*\[[^\]]*\]", "", first)
    first = re.sub(r",\s*(?:active |approximately |-)?\d.*$", "", first)
    return first.strip()


def _normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _join_until(paragraphs, n_words):
    parts, current, count = [], [], 0
    for paragraph in paragraphs:
        current.append(paragraph)
        count += len(paragraph.split())
        if count >= n_words:
            parts.append("\n\n".join(current))
            current, count = [], 0
    if current:
        parts.append("\n\n".join(current))
    return parts
