"""
Prepare a House of Commons corpus for the course.

Source: Evan Odell's "Hansard Speeches 1979–2021" as published on Hugging Face
(biglam/hansard_speech, CC-BY 4.0): every House of Commons speech, May 1979 to
April 2021, with party and other metadata.

What this script does:
    1. Downloads the dataset's Parquet files (once; ~1.4 GB).
    2. Keeps real speeches (not procedural lines) by MPs with a party (not
       the Speaker), between 50 and 1000 words long (counted with p4p.tokenize).
    3. Adds columns for the lessons: year, decade, party_main, side
       (government/opposition), frontbench, gender.
    4. Writes the full filtered corpus and a smaller sample: equal numbers
       of Conservative and Labour speeches each year, with the other parties
       in proportion to how much they actually spoke.

Gender and frontbench posts (with their dates) come from the UK Parliament
Members API, looked up once per MP and cached in mps.json. (The dataset's own
government_posts/opposition_posts columns are empty.)

Usage (from the repo root):
    python scripts/prepare_hansard.py
    python scripts/prepare_hansard.py --per-group 1000 --seed 2026

Outputs (in data/hansard/ by default):
    raw/*.parquet              the original dataset
    mps.json                   gender and dated posts for each MP
    hansard_full.parquet       every speech that passes the filters
    hansard_sample.parquet     up to --per-group Con and Lab speeches per year,
                               other parties in proportion
"""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request

import pandas as pd

import python4poets as p4p

DATASET_URL = "https://huggingface.co/datasets/biglam/hansard_speech/resolve/main/data/train-{i:05d}-of-00007.parquet"
N_FILES = 7
MEMBERS_API_URL = "https://members-api.parliament.uk/api/Members/{mnis_id}"
COMMONS = 1  # the Members API's code for the House of Commons

MIN_WORDS = 50
MAX_WORDS = 1000

COLUMNS = [
    "id", "speech", "display_as", "party", "constituency", "mnis_id", "date",
    "speech_class", "major_heading", "minor_heading",
]

# Parties as they appear in the dataset -> the main groupings used in lessons.
# Anything not listed becomes "Other".
PARTY_MAIN = {
    "Conservative": "Conservative",
    "Labour": "Labour",
    "Labour (Co-op)": "Labour",
    "Liberal Democrat": "Liberal Democrat",
    # The Liberal Democrats were formed in 1988 from the Liberal Party and
    # the SDP, which fought the 1983 and 1987 elections as the Alliance.
    "Liberal": "Liberal Democrat",
    "Social Democratic Party": "Liberal Democrat",
    "Scottish National Party": "SNP",
}

# The sample takes equal numbers from these each year (see balanced_sample).
MAJOR_PARTIES = ["Conservative", "Labour"]

# Which parties were in government, and when. Each entry starts on the day
# the new government took office and runs until the next entry.
GOVERNMENTS = [
    ("1979-05-04", {"Conservative"}),                      # Thatcher, Major
    ("1997-05-02", {"Labour"}),                            # Blair, Brown
    ("2010-05-11", {"Conservative", "Liberal Democrat"}),  # Cameron–Clegg coalition
    ("2015-05-08", {"Conservative"}),                      # Cameron, May, Johnson
]


def download_raw(raw_folder):
    os.makedirs(raw_folder, exist_ok=True)
    paths = []
    for i in range(N_FILES):
        url = DATASET_URL.format(i=i)
        path = os.path.join(raw_folder, os.path.basename(url))
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        paths.append(path)
    return paths


def load_and_filter(paths):
    parts = []
    for path in paths:
        df = pd.read_parquet(path, columns=COLUMNS)
        n_before = len(df)
        party = df["party"].str.strip()
        df = df[(df["speech_class"] == "Speech") & (party != "") & (party != "Speaker")]
        df = df.assign(word_count=[len(p4p.tokenize(s, lower=False)) for s in df["speech"]])
        df = df[df["word_count"].between(MIN_WORDS, MAX_WORDS)]
        print(f"  {os.path.basename(path)}: {n_before:,} rows -> {len(df):,} speeches")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def side_on(date, party_main):
    """'government' if party_main was in government on `date`, else 'opposition'."""
    in_power = set()
    for start, parties in GOVERNMENTS:
        if date >= pd.Timestamp(start):
            in_power = parties
    return "government" if party_main in in_power else "opposition"


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "python4poets-hansard-prep"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)["value"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_mp(mnis_id):
    """Gender and Commons frontbench posts (with dates) for one MP."""
    url = MEMBERS_API_URL.format(mnis_id=mnis_id)
    try:
        member = fetch_json(url)
        biography = fetch_json(url + "/Biography")
    except Exception:
        return None
    posts = [
        {"side": side, "name": post["name"], "start": post["startDate"], "end": post["endDate"]}
        for side, key in [("government", "governmentPosts"), ("opposition", "oppositionPosts")]
        for post in biography[key]
        if post["house"] == COMMONS
    ]
    return {"gender": member["gender"], "posts": posts}


def mp_lookup(mnis_ids, cache_path):
    """{mnis_id: {"gender": "M"/"F", "posts": [...]}}, from the cache or the Members API."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    missing = sorted(set(mnis_ids) - set(cache))
    if missing:
        print(f"Looking up {len(missing):,} MPs in the Members API ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for mnis_id, mp in zip(missing, pool.map(fetch_mp, missing)):
                if mp is not None:
                    cache[mnis_id] = mp
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=1, sort_keys=True)

    failed = [m for m in missing if m not in cache]
    if failed:
        print(f"  Could not look up {len(failed)} MPs (e.g. {failed[:5]}); rerun to retry.")
    return cache


def posts_on_dates(df, mps):
    """The frontbench post (if any) each speaker held on the day of each speech."""
    post_names = pd.Series("", index=df.index)
    for mnis_id, rows in df.groupby("mnis_id").groups.items():
        dates = df.loc[rows, "date"]
        for post in mps.get(mnis_id, {}).get("posts", []):
            start = pd.Timestamp(post["start"])
            end = pd.Timestamp(post["end"]) if post["end"] else pd.Timestamp.max
            held = dates[(dates >= start) & (dates <= end)].index
            # If two posts overlap, keep the first one found.
            unset = held[post_names[held] == ""]
            post_names[unset] = post["name"]
    return post_names


def add_columns(df, mps_cache_path):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["decade"] = df["year"] // 10 * 10
    df["party"] = df["party"].str.strip()
    df["party_main"] = df["party"].map(PARTY_MAIN).fillna("Other")
    df["side"] = [side_on(d, p) for d, p in zip(df["date"], df["party_main"])]

    df["mnis_id"] = df["mnis_id"].fillna("").astype(str).str.strip()
    mps = mp_lookup([m for m in df["mnis_id"].unique() if m], mps_cache_path)
    genders = {m: {"M": "male", "F": "female"}.get(mp["gender"], "") for m, mp in mps.items()}
    df["gender"] = df["mnis_id"].map(genders).fillna("")
    df["post"] = posts_on_dates(df, mps)
    df["frontbench"] = df["post"] != ""

    df = df.rename(columns={"display_as": "speaker", "speech": "txt"})
    return df[
        [
            "id", "date", "year", "decade", "speaker", "mnis_id", "gender",
            "party", "party_main", "side", "frontbench", "post", "constituency",
            "major_heading", "minor_heading", "word_count", "txt",
        ]
    ].sort_values(["date", "id"]).reset_index(drop=True)


def balanced_sample(df, per_group, seed):
    """
    Up to `per_group` Conservative and `per_group` Labour speeches per year;
    the other parties are sampled at the same rate as those two were that
    year, so they keep their real share of the debate.
    """
    # Shuffle, so that "the first n rows of a group" is a random sample.
    shuffled = df.sample(frac=1, random_state=seed)
    major = shuffled["party_main"].isin(MAJOR_PARTIES)
    major_sample = shuffled[major].groupby(["year", "party_main"]).head(per_group)

    # Each year's sampling rate for the major parties, applied to the rest.
    rate = major_sample["year"].value_counts() / shuffled.loc[major, "year"].value_counts()
    others = shuffled[~major]
    groups = others.groupby(["year", "party_main"])
    n_keep = (groups["id"].transform("size") * others["year"].map(rate)).round()
    others_sample = others[groups.cumcount() < n_keep]

    return (
        pd.concat([major_sample, others_sample])
        .sort_values(["date", "id"])
        .reset_index(drop=True)
    )


def summarize(name, df):
    print(f"\n{name}: {len(df):,} speeches, {df['word_count'].sum():,} words, {df['year'].min()}–{df['year'].max()}")
    print("  party_main:", df["party_main"].value_counts().to_dict())
    print("  side:      ", df["side"].value_counts().to_dict())
    print("  gender:    ", df["gender"].replace("", "(unknown)").value_counts().to_dict())
    print("  frontbench:", df["frontbench"].value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join("data", "hansard"), help="output folder")
    parser.add_argument("--per-group", type=int, default=1000, help="Conservative and Labour speeches per year in the sample")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for the sample")
    args = parser.parse_args()

    paths = download_raw(os.path.join(args.out, "raw"))

    print("Filtering speeches ...")
    df = load_and_filter(paths)
    df = add_columns(df, os.path.join(args.out, "mps.json"))

    full_path = os.path.join(args.out, "hansard_full.parquet")
    df.to_parquet(full_path, index=False)
    summarize(f"Full corpus -> {full_path}", df)

    sample = balanced_sample(df, args.per_group, args.seed)
    sample_path = os.path.join(args.out, "hansard_sample.parquet")
    sample.to_parquet(sample_path, index=False)
    summarize(f"Sample -> {sample_path}", sample)
    print(f"  file size: {os.path.getsize(sample_path) / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
