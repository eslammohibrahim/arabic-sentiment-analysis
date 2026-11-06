"""Exploratory analysis: class balance, review length, and the effect of normalization."""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_clean, load_raw

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
COLORS = {"Negative": "#ef4444", "Mixed": "#f59e0b", "Positive": "#22c55e"}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df = load_clean(verbose=True)

    print("\nClass balance:", df["sentiment"].value_counts().to_dict())
    df["words"] = df["clean"].str.split().str.len()
    print("\nWords per review:")
    print(df.groupby("sentiment")["words"].describe().round(1))

    raw_vocab = Counter(w for t in raw["text"].str.lower() for w in t.split())
    norm_vocab = Counter(w for t in df["clean"] for w in t.split())
    cut = 1 - len(norm_vocab) / len(raw_vocab)
    print(f"\nUnique tokens: raw {len(raw_vocab):,} -> normalized {len(norm_vocab):,} ({cut:.0%} fewer)")

    fig, ax = plt.subplots(figsize=(7, 4))
    for label in ["Negative", "Mixed", "Positive"]:
        words = df.loc[df["sentiment"] == label, "words"]
        words = words[words <= 150]
        ax.hist(words, bins=50, histtype="step", linewidth=2, label=label, color=COLORS[label])
    ax.set_xlabel("Words per review (reviews up to 150 words)")
    ax.set_ylabel("Reviews")
    ax.set_title("Review length by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "review_length.png", dpi=120)
    print(f"\nFigure saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
