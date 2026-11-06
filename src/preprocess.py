"""Arabic text normalization.

Arabic text has many ways to write the same word: optional diacritics,
different alef forms, stretched letters, and repeated letters for emphasis.
Normalizing these makes "رائعــــة", "رائعة" and "رآئعه" look the same to the model.
"""

import re

DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")  # tashkeel + dagger alef
TATWEEL = "ـ"  # the stretching character: جميــــل
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
REPEATED = re.compile(r"(.)\1{2,}")  # 3+ of the same character: جداااا
NOT_ALLOWED = re.compile(r"[^ء-غف-يa-z0-9\s]")  # keep Arabic letters, latin, digits
SPACES = re.compile(r"\s+")

ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
YA_TA = str.maketrans({"ى": "ي", "ة": "ه"})
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic(text: str) -> str:
    """Return a normalized version of `text` for bag-of-words models."""
    text = str(text).lower()
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = text.translate(ALEF).translate(YA_TA).translate(ARABIC_DIGITS)
    text = REPEATED.sub(r"\1", text)
    text = NOT_ALLOWED.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def basic_clean(text: str) -> str:
    """Minimal cleaning only (used as the 'no normalization' baseline)."""
    text = URL.sub(" ", str(text).lower())
    return SPACES.sub(" ", text).strip()
