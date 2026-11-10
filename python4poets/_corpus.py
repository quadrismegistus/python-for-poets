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
