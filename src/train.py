"""Train and compare sentiment models on Arabic reviews.

Workflow:
1. Split the data into train / validation / test (64% / 16% / 20%, stratified).
2. Check whether Arabic normalization helps (same model, raw vs normalized text).
3. Give every model a small hyperparameter search on the validation set, so the
   comparison is fair. Text features are computed once and reused.
4. Retrain the best model on train + validation and evaluate it on the test set once.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from data import RANDOM_STATE, load_clean, split
from preprocess import basic_clean

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
CLASSES = ["Negative", "Mixed", "Positive"]


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)


def char_tfidf():
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
        max_features=400_000, dtype=np.float32,
    )


def feature_pipeline(kind: str):
    if kind == "words":
        return word_tfidf()
    if kind == "characters":
        return char_tfidf()
    return FeatureUnion([("words", word_tfidf()), ("characters", char_tfidf())])


# name: (features, classifier factory, hyperparameter name, values to try)
MODELS = {
    "Naive Bayes": ("words", lambda a: MultinomialNB(alpha=a), "alpha", [0.1, 0.3, 1.0, 3.0]),
    "Logistic Regression": ("words", lambda c: LogisticRegression(C=c, max_iter=3000), "C", [0.3, 1, 3]),
    "Linear SVM": ("words", lambda c: LinearSVC(C=c), "C", [0.02, 0.05, 0.1, 0.25]),
    "Linear SVM (characters)": ("characters", lambda c: LinearSVC(C=c), "C", [0.02, 0.05, 0.1, 0.25]),
    "Linear SVM (words + characters)": ("both", lambda c: LinearSVC(C=c), "C", [0.02, 0.05, 0.1, 0.25]),
}


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def polarity_flip_rate(y_true, y_pred) -> float:
    """Among truly Positive/Negative reviews, how often the model picks the opposite polarity."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    polar = np.isin(y_true, ["Positive", "Negative"])
    flipped = polar & np.isin(y_pred, ["Positive", "Negative"]) & (y_true != y_pred)
    return round(float(flipped.sum() / polar.sum()), 4)


def plot_comparison(results: dict) -> None:
    """Dot plot: the scores are close together, so dots on a zoomed axis read more honestly than bars."""
    names = sorted(results, key=lambda n: results[n]["macro_f1"])
    f1 = [results[n]["macro_f1"] for n in names]
    best = names[-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for y, (name, v) in enumerate(zip(names, f1)):
        color = "#2563eb" if name == best else "#94a3b8"
        ax.hlines(y, 0.66, v, color="#e2e8f0", linewidth=2, zorder=1)
        ax.scatter(v, y, s=90, color=color, zorder=2)
        ax.text(v + 0.0012, y, f"{v:.3f}", va="center", fontsize=10,
                color=color, fontweight="bold" if name == best else "normal")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0.66, max(f1) + 0.01)
    ax.set_xlabel("Macro F1 on the validation set (best setting per model)")
    ax.set_title("Model comparison", loc="left")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=120)


def plot_confusion(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix (test set)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=120)


def top_words(vectorizer, clf, n: int = 12) -> dict:
    names = vectorizer.get_feature_names_out()
    return {
        label: [names[i] for i in np.argsort(coefs)[::-1][:n]]
        for label, coefs in zip(clf.classes_, clf.coef_)
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean(verbose=True)
    train_full, test = split(df)
    train, val = train_test_split(
        train_full, test_size=0.2, stratify=train_full["sentiment"], random_state=RANDOM_STATE
    )
    y_tr, y_va = train["sentiment"], val["sentiment"]
    print(f"\nTrain: {len(train):,}  Validation: {len(val):,}  Test: {len(test):,}\n")
    report = {"data": {"total": len(df), "train": len(train), "validation": len(val), "test": len(test)}}

    # 1. Does Arabic normalization help? Same model (Linear SVM, words, C=0.25), only the text differs.
    print("Normalization check:")
    norm = {}
    for name, X_tr, X_va in [
        ("basic cleaning", train["text"].map(basic_clean), val["text"].map(basic_clean)),
        ("Arabic normalization", train["clean"], val["clean"]),
    ]:
        model = make_pipeline(word_tfidf(), LinearSVC(C=0.25)).fit(X_tr, y_tr)
        norm[name] = scores(y_va, model.predict(X_va))
        print(f"  {name:<22} {norm[name]}")
    report["normalization"] = norm

    # 2. Build the text features once
    print("\nBuilding features...")
    feats = {}
    for kind in ["words", "characters"]:
        start = time.time()
        vec = feature_pipeline(kind)
        feats[kind] = (vec.fit_transform(train["clean"]), vec.transform(val["clean"]))
        print(f"  {kind:<11} {feats[kind][0].shape[1]:>9,} features  ({time.time() - start:.0f}s)")
    feats["both"] = tuple(hstack([feats["words"][i], feats["characters"][i]]).tocsr() for i in range(2))

    # 3. Fair comparison: best setting per model on the validation set
    print("\nModel comparison (validation):")
    results, search_log = {}, {}
    for name, (kind, make_clf, param, values) in MODELS.items():
        X_tr, X_va = feats[kind]
        tried = {}
        for v in values:
            start = time.time()
            clf = make_clf(v).fit(X_tr, y_tr)
            tried[v] = scores(y_va, clf.predict(X_va)) | {"seconds": round(time.time() - start, 1)}
        best_v = max(tried, key=lambda v: tried[v]["macro_f1"])
        results[name] = tried[best_v] | {param: best_v}
        search_log[name] = {str(k): v for k, v in tried.items()}
        print(f"  {name:<32} best {param}={best_v:<5} {tried[best_v]}")
    report["comparison"] = results
    report["search"] = search_log
    plot_comparison(results)

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    kind, make_clf, param, _ = MODELS[best_name]
    best_value = results[best_name][param]
    print(f"\nBest on validation: {best_name} ({param}={best_value})")

    # 4. Final model: retrain on train + validation, evaluate on the test set once
    final = make_pipeline(feature_pipeline(kind), make_clf(best_value))
    final.fit(train_full["clean"], train_full["sentiment"])
    pred = final.predict(test["clean"])
    test_scores = scores(test["sentiment"], pred)
    test_scores["polarity_flip_rate"] = polarity_flip_rate(test["sentiment"], pred)
    per_class = classification_report(test["sentiment"], pred, labels=CLASSES, output_dict=True)
    print(f"\nTEST: {test_scores}")
    print(classification_report(test["sentiment"], pred, labels=CLASSES, digits=3))
    report["final"] = {
        "model": best_name, param: best_value, "test": test_scores,
        "per_class": {c: {k: round(float(v), 4) for k, v in per_class[c].items()} for c in CLASSES},
    }
    plot_confusion(test["sentiment"], pred)
    joblib.dump(final, MODEL_DIR / "sentiment_model.joblib", compress=3)

    # 5. Interpretability: most influential words per class (word-level Linear SVM)
    vec = word_tfidf()
    X = vec.fit_transform(train_full["clean"])
    svm_c = results["Linear SVM"]["C"]
    report["top_words"] = top_words(vec, LinearSVC(C=svm_c).fit(X, train_full["sentiment"]))

    # 6. Error examples for the error analysis
    errors = test.assign(pred=pred)
    errors = errors[(errors["pred"] != errors["sentiment"]) & errors["clean"].str.len().between(15, 90)]
    report["error_examples"] = (
        errors.sample(15, random_state=RANDOM_STATE)[["text", "sentiment", "pred"]]
        .to_dict(orient="records")
    )

    (ROOT / "reports" / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved reports/results.json, figures, and models/sentiment_model.joblib")


if __name__ == "__main__":
    main()
