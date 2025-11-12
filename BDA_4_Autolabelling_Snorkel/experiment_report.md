# Experiment Report — Autolabelling with Snorkel & Label Studio

Project: Amazon Review Polarity (polarity sentiment: positive / negative)
Date: 2025-11-12
Author: (auto-generated)

## Goal
Build an auto-labeling pipeline that combines weak supervision (Snorkel) with small hand-labeled data from Label Studio to produce high-quality labels for Amazon review sentiment. Provide a reproducible baseline and next steps for scaling to the full dataset (34,686,770 reviews).

## Data
- Small placeholder CSVs included in this repo for a reproducible toy experiment:
  - `train.csv` (6 examples)
  - `test.csv` (4 examples)
- Real dataset: Stanford SNAP Amazon Review Polarity subset described by the user (34,686,770 reviews). Not included in repo due to size.

CSV format used:
- `review_id` — unique id
- `text` — review text
- `label` — (optional) gold label, `positive` or `negative`

## Methods
1. Label Studio
   - Create a small human-labeled dev set (~500-2,000 items) to use as a gold set and for LF calibration.
   - Export labels to CSV (`labelstudio_export.csv`).
2. Snorkel
   - Define a set of interpretable labeling functions (LFs) that capture heuristics (presence of positive/negative words, negation patterns, punctuation signals, short-message heuristics).
   - Apply LFs to the unlabeled dataset to produce a label matrix.
   - Train Snorkel's LabelModel to learn LF accuracies and correlations and produce probabilistic labels.
   - Threshold or argmax the probabilistic labels to produce hard labels for downstream training.
3. (Optional) Use a small supervised classifier (e.g., logistic regression or a finetuned transformer) trained on Snorkel-labeled data and evaluated on the Label Studio dev set.

## Implementation
- See `snorkel_pipeline.py` for a runnable example: it defines several LFs and trains a LabelModel.
- See `label_studio_instructions.md` for steps to create a Label Studio project and export labels.
- Install dependencies from `requirements.txt`.

## Labeling Functions (LFs) used (toy set)
- lf_has_positive_words: matches tokens from a positive lexicon (returns POS).
- lf_has_negative_words: matches tokens from a negative lexicon (returns NEG).
- lf_exclamation_positive: heuristics using `!` combined with lexicons.
- lf_short_negative: short messages containing negative tokens often label negative.
- lf_has_not_good: negation patterns like "not good" → negative.

These are illustrative — for large-scale experiments, expand the LF set with product-category aware heuristics, emoticon handling, star-rating signals (if available), and entity-based patterns.

## How to run (local)
1. Create a Python venv and install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r BDA_4_Autolabelling_Snorkel/requirements.txt
```

2. Run the Snorkel pipeline:

```bash
python BDA_4_Autolabelling_Snorkel/snorkel_pipeline.py
```

3. Label with Label Studio (optional): follow `label_studio_instructions.md` to create a dev set and export labels. Replace `train.csv` or provide `labelstudio_export.csv` as needed.

## Results (toy)
- The included toy dataset is too small to produce meaningful performance figures. Use the Label Studio exported dev set and the full training subset to compute accuracy/precision/recall.

Example evaluation (after running pipeline and labeling ~1k dev samples):
- Accuracy: X.XX
- Precision / Recall / F1 (positive): X.XX / X.XX / X.XX
- Precision / Recall / F1 (negative): X.XX / X.XX / X.XX

(Replace above with real numbers once you run the pipeline on your labeled dev set.)

## Next steps and scaling to full dataset
1. Expand and refine LFs:
   - Use domain-specific token lists.
   - Add regexes for rating phrases ("one star", "five star"), shipping complaints, feature-specific patterns.
2. Add more supervision sources:
   - Use metadata such as star rating, helpfulness votes, or product categories as weak signals.
   - Use a pretrained sentiment model (zero-shot or fine-tuned) as an LF (careful about correlated errors).
3. Train a downstream classifier on Snorkel-labeled data (fine-tune a transformer or train a linear model on embeddings) and evaluate on the Label Studio dev set.
4. Compute calibration and expected label quality before consuming labels at scale.
5. For full-scale processing (millions of reviews), use distributed processing (Dask / Spark) for LF application and batch LabelModel training or train on representative subsamples.

## Artifacts created
- `train_labeled_by_snorkel.csv` — output of the Snorkel pipeline (probabilistic/hard labels).
- `snorkel_pipeline.py` — code for LFs and LabelModel training.
- `label_studio_instructions.md` — how to get human labels.

## Caveats
- Weak supervision quality depends heavily on LF coverage and accuracy. Evaluate on a human-labeled dev set before using labels for downstream model training.
- Some LFs may be highly correlated (e.g., lexicon-based LFs) and LabelModel accounts for this to some extent, but diverse LFs are better.


## Contact / Reproducibility
If you want, I can:
- Expand LFs and add a baseline supervised classifier that trains on Snorkel-labeled data.
- Add a small notebook that runs the pipeline and plots LF coverage/accuracy using a Label Studio dev set.

