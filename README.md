# Arabic Sentiment Analysis

Classifying Arabic reviews as **Positive**, **Mixed**, or **Negative**, with an Arabic-specific text normalization step and a fair comparison of five classic NLP models.

**Result:** 69.2% accuracy and 0.689 macro F1 on 19,964 unseen test reviews (three balanced classes, so random guessing gets 33%). When a review is truly positive or negative, the model picks the opposite polarity only **7.5%** of the time. Most remaining errors involve the Mixed class.

```text
$ python src/predict.py
Positive  | الفندق نظيف جدا والموظفين محترمين والفطار ممتاز، أكيد هرجعله تاني
Mixed     | المكان حلو بس الخدمة بطيئة والأسعار غالية شوية
Negative  | تجربة سيئة جدا، الغرفة مش نضيفة والتكييف بايظ ومحدش رد علينا
Positive  | الكتاب ممتع وأسلوب الكاتب جميل
Negative  | الرواية مملة ومفيهاش أي جديد، ضيعت وقتي
```

These are sentences I wrote myself in Egyptian dialect as a quick sanity check, not a benchmark.

## Data

[Arabic 100K Reviews](https://huggingface.co/datasets/arbml/arabic_100k_reviews), 99,999 reviews with three balanced classes. It was originally published on Kaggle; this project uses the Hugging Face copy. The data isn't stored in this repo, and `src/data.py` downloads it on first run.

What I found while exploring:

- **Mostly book and hotel reviews.** A keyword check finds book words in about 39% of reviews and hotel words in about 26%. The rest are short or from other domains.
- **The labels seem to follow star ratings, not the text.** 6.6% of reviews mention stars directly ("نجمتان", "ثلاث نجمات"). Some reviews that are identical after normalization carry different labels: "استمتعت به كثيرا" is labeled Positive once and Mixed once. This sets a ceiling on how accurate any model can be, especially for Mixed.
- **Review length doesn't separate the classes.** Positive reviews are slightly shorter, but the distributions overlap almost completely.

Cleaning:

| Step | Reviews |
|---|---|
| Raw | 99,999 |
| Empty or too short after normalization | −15 |
| Same text with different labels (can't be learned) | −75 |
| Exact duplicates (kept one copy, so none leak between train and test) | −89 |
| **Final** | **99,820** |

![Review length](reports/figures/review_length.png)

## Arabic text normalization

Arabic can write the same word in many ways. [`src/preprocess.py`](src/preprocess.py) maps them to one form:

| Step | Example |
|---|---|
| Remove diacritics (tashkeel) | جَمِيلٌ → جميل |
| Remove tatweel (stretching) | رائعــــة → رائعه |
| Unify alef forms | أ إ آ ٱ → ا |
| Unify ya and ta marbuta | ى → ي, ة → ه |
| Collapse repeated letters | حلوووووو جداااا → حلو جدا |
| Convert Arabic digits | ٥ نجوم → 5 نجوم |
| Remove URLs, mentions, punctuation | |

This cut the number of unique words by **36%** (418,549 → 268,141). With the same model, it raised validation macro F1 from 0.679 to 0.684. That's a small but consistent gain. The steps are covered by unit tests in [`tests/`](tests/test_preprocess.py).

## Approach

1. Stratified split: 64% train, 16% validation, 20% test.
2. Features: TF-IDF on words (1 and 2 word phrases, so "لا انصح" and "مخيب للامل" count as units) and on character n-grams (2 to 5 characters, which handle spelling variants and dialect).
3. Every model gets its own hyperparameter search on the validation set, so the comparison is fair. Features are computed once and reused.
4. The best model is retrained on train + validation and evaluated on the test set **once**.

## Results

Validation set, best setting per model:

| Model | Features | Best setting | Macro F1 |
|---|---|---|---|
| Naive Bayes | words | alpha = 3.0 | 0.671 |
| Linear SVM | characters | C = 0.1 | 0.678 |
| Linear SVM | words | C = 0.1 | 0.684 |
| Logistic Regression | words | C = 1 | 0.686 |
| **Linear SVM** | **words + characters** | **C = 0.1** | **0.690** |

![Model comparison](reports/figures/model_comparison.png)

Final model on the test set (19,964 reviews):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.732 | 0.758 | 0.745 |
| Mixed | 0.630 | 0.552 | 0.589 |
| Positive | 0.705 | 0.767 | 0.735 |
| **Overall** | | **Accuracy 0.692** | **Macro F1 0.689** |

The test score (0.689) matches the validation score (0.690), so the model selection didn't overfit the validation set.

![Confusion matrix](reports/figures/confusion_matrix.png)

Positive and Negative are rarely confused with each other (7 to 8%). Mixed is the hard class: it gets mistaken for both sides, which makes sense for a review that is partly good and partly bad.

## What the model learned

Words with the strongest weight for each class (word-level Linear SVM, after normalization):

| Class | Top words |
|---|---|
| Positive | استثنائي، رائع، ممتاز، جيد، روعه، اروع، من اجمل، جيد جدا، تحفه |
| Mixed | مرضي، مقبول، حسن، متوسط، لاباس، **لكن**، **ولكن** |
| Negative | ضعيف، مخيب للامل، سيئه، اسوا، **غير**، **لم**، لا انصح |

Two patterns stand out:
- **Contrast words ("لكن", "ولكن") signal Mixed.** Reviewers use them to balance praise and complaints.
- **Negation words ("غير", "لم") signal Negative.** Using word pairs lets the model see "لا انصح" as one unit instead of an unrelated "لا".

## Error analysis

Some test-set mistakes and why they happen:

| Review | Label | Predicted | Why |
|---|---|---|---|
| كان ليكون كتابا رائعا لو انني قرأته قبل خمسة عشر عاما | Negative | Positive | "Would have been great if..." — a bag-of-words model sees "رائعا" but not the conditional |
| اقامة فخمه . كل شئ ماعدا التكيييييف. التكيييييييييييييف سئ | Positive | Negative | Overall praise, but one strong complaint dominates the words |
| السلعة سعرها ممتاز وجيدة لا بأس بها | Mixed | Positive | "ممتاز" outweighs the lukewarm "لا بأس بها" |
| الفندق جيد ولكن السعر مرتفع عليه . نعم. المعاملة من بعض الموظفين | Negative | Mixed | The text reads as mixed; the reviewer's rating said negative |
| خفيف جدا .. مسلي أيضا | Negative | Mixed | The text sounds positive; the label likely comes from a low star rating |

The first three are real model limits: conditionals, and one strong complaint outweighing general praise. The last two are label noise, where the text and the rating disagree.

## Limitations

- Labels seem to come from ratings, so some don't match the text, and the ceiling for Mixed is low.
- Mostly books and hotels: performance on other domains, such as restaurants or apps, isn't measured.
- Performance on Egyptian dialect isn't measured beyond a few hand-written examples.
- Bag-of-words models don't understand word order, sarcasm, or conditionals.

## Next steps

- Fine-tune an Arabic transformer (AraBERT or CAMeLBERT) and compare it against this baseline.
- Evaluate on an Egyptian-dialect dataset to measure how well the model transfers.
- Serve the model behind a small API.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/explore.py         # data stats and the review-length chart
python src/train.py           # downloads the data, compares models, saves the best one (a few minutes on CPU)
python src/predict.py "الخدمة ممتازة والأكل طعمه حلو"
pytest                        # normalization unit tests
```

## Project structure

```
src/
  preprocess.py   Arabic text normalization
  data.py         download, cleaning, deduplication, train/test split
  explore.py      exploratory analysis
  train.py        model comparison, tuning, final evaluation, figures
  predict.py      predictions with the saved model
tests/
  test_preprocess.py
reports/
  figures/        generated charts
  results.json    all metrics, search results, top words, and error examples
```
