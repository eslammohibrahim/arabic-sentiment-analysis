"""Download, load, clean and split the Arabic 100K Reviews dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import normalize_arabic

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "arabic_100k_reviews.parquet"
DATA_URL = (
    "https://huggingface.co/api/datasets/arbml/arabic_100k_reviews"
    "/parquet/default/train/0.parquet"
)

LABELS = {0: "Positive", 1: "Mixed", 2: "Negative"}
RANDOM_STATE = 42
MIN_CHARS = 3  # reviews shorter than this after cleaning carry no signal


def download() -> Path:
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {RAW_PATH} ...")
        urlretrieve(DATA_URL, RAW_PATH)
    return RAW_PATH


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(download())
    df["sentiment"] = df["label"].map(LABELS)
    return df


def load_clean(verbose: bool = False) -> pd.DataFrame:
    """Normalize the text, then drop too-short, conflicting and duplicate reviews."""
    df = load_raw()
    n0 = len(df)
    df["clean"] = df["text"].map(normalize_arabic)

    df = df[df["clean"].str.len() >= MIN_CHARS]
    n_short = n0 - len(df)

    # The same review (after normalization) with different labels can't be learned: drop it.
    labels_per_text = df.groupby("clean")["sentiment"].transform("nunique")
    conflicting = labels_per_text > 1
    n_conflict = int(conflicting.sum())
    df = df[~conflicting]

    # The same review with the same label: keep one copy so it can't land in both train and test.
    n_before = len(df)
    df = df.drop_duplicates(subset="clean")
    n_dupes = n_before - len(df)

    if verbose:
        print(f"Raw reviews:                 {n0:,}")
        print(f"Too short after cleaning:   -{n_short:,}")
        print(f"Same text, different label: -{n_conflict:,}")
        print(f"Duplicates:                 -{n_dupes:,}")
        print(f"Final:                       {len(df):,}")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2):
    return train_test_split(
        df, test_size=test_size, stratify=df["sentiment"], random_state=RANDOM_STATE
    )
