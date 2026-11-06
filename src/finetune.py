"""Fine-tune a small Arabic BERT on the same split as train.py, to compare with the TF-IDF + SVM baseline.

Usage:
    python src/finetune.py --model asafaya/bert-mini-arabic --epochs 3
    python src/finetune.py --model asafaya/bert-medium-arabic --epochs 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import RANDOM_STATE, load_clean, split
from preprocess import basic_clean

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["Negative", "Mixed", "Positive"]
MAX_LEN = 128


class Encoded:
    """Token ids kept unpadded; each batch is padded only to its own longest review."""

    def __init__(self, tokenizer, texts, labels):
        self.ids = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]
        self.y = np.array([CLASSES.index(l) for l in labels])
        self.pad_id = tokenizer.pad_token_id

    def batches(self, batch_size, shuffle, rng=None):
        order = np.arange(len(self.ids))
        if shuffle:
            rng.shuffle(order)
            # sort by length inside chunks of 50 batches, so a batch holds reviews of similar length
            chunk = batch_size * 50
            order = np.concatenate([sorted(order[i:i + chunk], key=lambda j: len(self.ids[j]))
                                    for i in range(0, len(order), chunk)])
            starts = np.arange(0, len(order), batch_size)
            rng.shuffle(starts)
        else:
            order = np.array(sorted(order, key=lambda j: len(self.ids[j])))
            starts = np.arange(0, len(order), batch_size)
        for s in starts:
            idx = order[s:s + batch_size]
            width = max(len(self.ids[j]) for j in idx)
            ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long)
            for row, j in enumerate(idx):
                ids[row, :len(self.ids[j])] = torch.tensor(self.ids[j])
            yield idx, ids, (ids != self.pad_id).long(), torch.tensor(self.y[idx])

    def n_batches(self, batch_size):
        return (len(self.ids) + batch_size - 1) // batch_size


@torch.no_grad()
def predict(model, data, batch_size=128):
    model.eval()
    preds = np.zeros(len(data.y), dtype=int)
    for idx, ids, mask, _ in data.batches(batch_size, shuffle=False):
        preds[idx] = model(input_ids=ids, attention_mask=mask).logits.argmax(-1).numpy()
    return preds


def main(model_name: str, epochs: int, batch_size: int = 32, lr: float = 5e-5):
    torch.manual_seed(RANDOM_STATE)
    df = load_clean()
    train_full, test = split(df)
    train, val = train_test_split(train_full, test_size=0.2, stratify=train_full["sentiment"], random_state=RANDOM_STATE)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(CLASSES))
    # BERT has its own tokenizer, so it gets the raw text with only URLs removed
    sets = {name: Encoded(tokenizer, part["text"].map(basic_clean), part["sentiment"])
            for name, part in [("train", train), ("val", val), ("test", test)]}

    rng = np.random.default_rng(RANDOM_STATE)
    n_steps = sets["train"].n_batches(batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * n_steps
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best_f1, best_state, history, start = -1.0, None, [], time.time()
    params = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {params / 1e6:.1f}M parameters, {n_steps} steps per epoch", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (_, ids, mask, y) in enumerate(sets["train"].batches(batch_size, True, rng), 1):
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{n_steps} loss {loss.item():.3f} [{time.time() - start:.0f}s]", flush=True)
        val_pred = predict(model, sets["val"])
        val_f1 = f1_score(sets["val"].y, val_pred, average="macro")
        history.append({"epoch": epoch, "val_macro_f1": round(float(val_f1), 4), "minutes": round((time.time() - start) / 60, 1)})
        print(f"  epoch {epoch}: val macro F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_test = sets["test"].y
    t = time.time()
    test_pred = predict(model, sets["test"])
    ms_per_review = (time.time() - t) / len(y_test) * 1000
    report = classification_report(y_test, test_pred, target_names=CLASSES, output_dict=True)
    result = {
        "model": model_name,
        "parameters_millions": round(params / 1e6, 1),
        "epochs": epochs,
        "history": history,
        "best_val_macro_f1": round(float(best_f1), 4),
        "test": {"accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
                 "macro_f1": round(float(f1_score(y_test, test_pred, average="macro")), 4)},
        "per_class_f1": {c: round(report[c]["f1-score"], 4) for c in CLASSES},
        "train_minutes": history[-1]["minutes"],
        "cpu_ms_per_review_batched": round(ms_per_review, 2),
    }
    print(json.dumps(result, indent=2))
    out = ROOT / "reports" / "bert"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_name.split('/')[-1]}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="asafaya/bert-mini-arabic")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()
    main(args.model, args.epochs, lr=args.lr)
