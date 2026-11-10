"""Downloading and loading corpora."""

import os
import unicodedata
import urllib.parse
import urllib.request
import warnings
import zipfile

import pandas as pd

DATA_FOLDER = "p4p_data"

SOTU_METADATA_URL = "https://www.dropbox.com/s/6gyueael6smbxyg/sotu_metadata.zip?dl=1"
SOTU_TXT_URL = "https://www.dropbox.com/s/w73qio0thhfzdpx/sotu_txt.zip?dl=1"
SOTU_PARTIES_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSdeAmHi47hihQ1JxK3SmoBBVH7sOCK7qcBioALss9T2EeOFV57"
    "Gi0tI7qmX0_RS_AAdZtkKmi0cxyI/pub?gid=0&single=true&output=csv"
)

NOVEL450_METADATA_URL = (
    "https://www.dropbox.com/scl/fi/hgb3vm357s3zjmsxgi62h/txtlab_metadata.zip"
    "?rlkey=0zimp348e24qwlqjvxrt9u33z&st=74dmierx&dl=1"
)
NOVEL450_TXT_URL = (
    "https://www.dropbox.com/scl/fi/mapo8e2grwpszct9503xn/txtlab_txt.zip"
    "?rlkey=7wmn97nvm0zzsm4bzaihul3zs&st=98zmx6sg&dl=1"
)

# Made by scripts/prepare_hansard.py (the hansard_sample.parquet it writes).
HANSARD_SAMPLE_URL = (
    "https://www.dropbox.com/scl/fi/1axb96yfi99eqhrh62kah/hansard_sample.parquet"
    "?rlkey=t3pp50lq217c9b7duubhjgyeo&dl=1"
)


def download(url, folder=DATA_FOLDER, filename=None):
    """
    Download a file into `folder`, unless it's already there. Zip files are unzipped.

    Returns the path to the downloaded file.

    Example:
        >>> p4p.download("https://example.com/corpus.zip", folder="my_corpus")
        'my_corpus/corpus.zip'
    """
    if filename is None:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        print(f"Downloading {filename} ...")
        # Download to a temporary name first, so a half-finished download
        # isn't mistaken for a finished one next time.
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
            if not all(os.path.exists(os.path.join(folder, n)) for n in names):
                zf.extractall(folder, members=names)

    return path


def load_corpus(metadata_csv, txt_folder, id_col="id", text_col="txt"):
    """
    Load a corpus: a spreadsheet of metadata plus one .txt file per row.

    Reads `metadata_csv` and, for each row, the file `{txt_folder}/{id}.txt`,
    putting its text in a new column (`text_col`, default "txt").

    Returns a pandas DataFrame.

    Example:
        >>> df = p4p.load_corpus("metadata.csv", "txt")
    """
    df = pd.read_csv(metadata_csv)
    files = _list_txt_files(txt_folder)
    texts = []
    missing = []
    for doc_id in df[id_col]:
        path = files.get(unicodedata.normalize("NFC", str(doc_id)))
        if path is None:
            missing.append(doc_id)
            texts.append("")
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    if missing:
        warnings.warn(
            f"{len(missing)} of {len(df)} rows have no text file in {txt_folder!r} "
            f"(e.g. {missing[0]!r}); their text is left empty.",
            stacklevel=2,
        )
    df[text_col] = texts
    return df


def _list_txt_files(txt_folder):
    # Map each file's id (filename minus .txt) to its path. Filenames are
    # normalized because zips made on a Mac store accented letters
    # ("ä") in a different Unicode form from spreadsheets.
    files = {}
    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            doc_id = unicodedata.normalize("NFC", filename[: -len(".txt")])
            files[doc_id] = os.path.join(txt_folder, filename)
    return files


def load_sotu(folder=os.path.join(DATA_FOLDER, "sotu")):
    """
    Load the U.S. State of the Union corpus (1790–2018), with each speaker's party.

    Columns: id, author, title, year, president, party, txt.
    Downloads the corpus the first time; later calls reuse the files.

    Example:
        >>> df = p4p.load_sotu()
        >>> df["party"].value_counts()
    """
    download(SOTU_METADATA_URL, folder, "sotu_metadata.zip")
    download(SOTU_TXT_URL, folder, "sotu_txt.zip")
    download(SOTU_PARTIES_URL, folder, "parties.csv")

    df = load_corpus(os.path.join(folder, "metadata.csv"), os.path.join(folder, "txt"))
    parties = pd.read_csv(os.path.join(folder, "parties.csv"), keep_default_na=False)
    parties.columns = ["year", "president", "party"]

    # Match each speech to its speaker's party by surname, using the nearest
    # year on or before the speech (or else the first year after it, for a
    # president's first speech in the year he took office). Matching on year
    # alone gets handover years wrong: Truman's 1953 speech is not Eisenhower's.
    surnames = parties["president"].str.split().str[-1]
    presidents, party_names = [], []
    for author, year in zip(df["author"], df["year"]):
        rows = parties[surnames == author]
        before = rows[rows["year"] <= year]
        after = rows[rows["year"] > year]
        if len(before):
            match = before.sort_values("year").iloc[-1]
        elif len(after):
            match = after.sort_values("year").iloc[0]
        else:
            presidents.append("")
            party_names.append("")
            continue
        presidents.append(match["president"])
        party_names.append(match["party"].strip())
    df.insert(df.columns.get_loc("year") + 1, "president", presidents)
    df.insert(df.columns.get_loc("president") + 1, "party", party_names)

    return df.sort_values("year").reset_index(drop=True)


def load_novel450(folder=os.path.join(DATA_FOLDER, "novel450")):
    """
    Load the txtLAB Novel450 corpus: 450 novels in English, French and German (1771–1930).

    Columns: id, id_orig, author, title, year, language, date, gender, person,
    length, txt. Downloads the corpus the first time; later calls reuse the files.
    See https://txtlab.org/novel450/

    Example:
        >>> df = p4p.load_novel450()
        >>> df["language"].value_counts()
    """
    download(NOVEL450_METADATA_URL, folder, "txtlab_metadata.zip")
    download(NOVEL450_TXT_URL, folder, "txtlab_txt.zip")
    df = load_corpus(os.path.join(folder, "metadata.csv"), os.path.join(folder, "txt"))
    return df.sort_values("year").reset_index(drop=True)


def load_hansard(folder=os.path.join(DATA_FOLDER, "hansard")):
    """
    Load a sample of ~96,000 House of Commons speeches (1979–2021).

    Each year has 1,000 Conservative and 1,000 Labour speeches; the other
    parties are sampled at the same rate, so they keep their real share.
    Only speeches of 50–1000 words by MPs with a party are included.

    Columns:
        date, year, decade            when the speech was given
        speaker, mnis_id, gender      who gave it (mnis_id: Parliament's ID for the MP)
        party, party_main             party as recorded; and grouped into Conservative,
                                      Labour, Liberal Democrat, SNP, Other
        side                          "government" or "opposition" on that date
        frontbench, post              whether the MP held a ministerial or shadow post
                                      that day, and which
        constituency
        major_heading, minor_heading  the debate's topic headings
        word_count                    the speech's length in words
        id, txt                       the speech's ID in Hansard, and the speech

    Caution: opposition (shadow) posts are under-recorded before about 2000,
    so `frontbench` is reliable for the government side throughout, but for
    the opposition only from around then.

    Sources: Evan Odell, "Hansard Speeches 1979–2021: Version 3.1.0" (Zenodo, 2021,
    CC-BY 4.0, https://doi.org/10.5281/zenodo.4843485), built on mySociety's parlparse;
    gender and posts from the UK Parliament Members API (members-api.parliament.uk).

    Downloads the file (~60 MB) the first time; later calls reuse it.

    Example:
        >>> df = p4p.load_hansard()
        >>> df.groupby("year")["gender"].value_counts(normalize=True)
    """
    path = download(HANSARD_SAMPLE_URL, folder, "hansard_sample.parquet")
    df = pd.read_parquet(path)
    # Older Hansard markup leaves line breaks and tabs around some headings.
    for col in ["major_heading", "minor_heading"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
    # The ids are long and unreadable; put them at the end, next to the text,
    # so tables (and p4p.kwic) start with the date and speaker.
    if "id" in df.columns:
        df = df[[c for c in df.columns if c not in ("id", "txt")] + ["id", "txt"]]
    return df
