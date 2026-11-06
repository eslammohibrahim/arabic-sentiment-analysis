"""Put the BERT results next to the TF-IDF + SVM baseline (same test set) and draw one chart."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    base = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))["final"]
    rows = [{
        "name": "TF-IDF + linear SVM", "params": None,
        "acc": base["test"]["accuracy"], "f1": base["test"]["macro_f1"],
        "mixed": base["per_class"]["Mixed"]["f1-score"],
    }]
    results = [json.loads(p.read_text()) for p in (ROOT / "reports" / "bert").glob("bert-*.json")]
    for r in sorted(results, key=lambda r: r["parameters_millions"]):
        rows.append({
            "name": r["model"].split("/")[-1], "params": r["parameters_millions"],
            "acc": r["test"]["accuracy"], "f1": r["test"]["macro_f1"], "mixed": r["per_class_f1"]["Mixed"],
            "minutes": r["train_minutes"],
        })
    for r in rows:
        print(r)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names = [r["name"] for r in rows]
    f1 = [r["f1"] for r in rows]
    colors = ["#94a3b8"] + ["#2563eb"] * (len(rows) - 1)
    bars = ax.barh(names, f1, color=colors)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in f1], padding=4)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Macro F1 on the same 19,964 test reviews")
    ax.set_title("Classic baseline vs small Arabic BERTs", loc="left")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "bert_vs_svm.png", dpi=120)
    (ROOT / "reports" / "bert" / "comparison.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
