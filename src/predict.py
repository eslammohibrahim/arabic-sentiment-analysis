"""Predict the sentiment of Arabic reviews with the trained model.

Usage:
    python src/predict.py "الفندق نظيف والموظفين محترمين جدا"
    python src/predict.py            # runs a few built-in examples
"""

import sys
from pathlib import Path

import joblib

from preprocess import normalize_arabic

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.joblib"

EXAMPLES = [
    "الفندق نظيف جدا والموظفين محترمين والفطار ممتاز، أكيد هرجعله تاني",
    "المكان حلو بس الخدمة بطيئة والأسعار غالية شوية",
    "تجربة سيئة جدا، الغرفة مش نضيفة والتكييف بايظ ومحدش رد علينا",
    "الكتاب ممتع وأسلوب الكاتب جميل",
    "الرواية مملة ومفيهاش أي جديد، ضيعت وقتي",
]


def predict(texts: list[str]) -> list[str]:
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run `python src/train.py` first.")
    model = joblib.load(MODEL_PATH)
    return list(model.predict([normalize_arabic(t) for t in texts]))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    texts = sys.argv[1:] or EXAMPLES
    for text, label in zip(texts, predict(texts)):
        print(f"{label:<9} | {text}")


if __name__ == "__main__":
    main()
