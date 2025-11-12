"""
Simple Snorkel pipeline for labeling Amazon review polarity (positive/negative).

This script demonstrates:
- Loading CSVs (`train.csv`, `test.csv`).
- Defining labeling functions (LFs).
- Applying LFs to build a label matrix.
- Training a Snorkel LabelModel.
- Generating probabilistic labels and an estimated hard label.
- Evaluating against `test.csv` if labels are available.

Run:
python snorkel_pipeline.py

"""

import pandas as pd
import numpy as np
from snorkel.labeling import LabelModel, labeling_function, PandasLFApplier
from snorkel.labeling import LFAnalysis
from sklearn.metrics import classification_report, accuracy_score
import re

# Label constants
ABSTAIN = -1
POS = 1
NEG = 0

# --- Load data ---
train_path = "BDA_4_Autolabelling_Snorkel/train.csv"
test_path = "BDA_4_Autolabelling_Snorkel/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# If labels in train are present, we'll hide them from LF stage to mimic unlabeled data
if "label" not in df_train.columns:
    df_train["label"] = np.nan

# --- Helper functions ---
positive_words = set(["good","great","excellent","love","loved","amazing","perfect","best","nice","exceeded"])
negative_words = set(["bad","terrible","waste","worst","disappoint","broke","broken","awful","poor","stopped","not working","do not buy","dont buy"])

def contains_word_set(x, words):
    txt = str(x).lower()
    for w in words:
        if w in txt:
            return True
    return False

# --- Labeling functions ---
@labeling_function()
def lf_has_positive_words(x):
    return POS if contains_word_set(x.text, positive_words) else ABSTAIN

@labeling_function()
def lf_has_negative_words(x):
    return NEG if contains_word_set(x.text, negative_words) else ABSTAIN

@labeling_function()
def lf_exclamation_positive(x):
    # Exclamation often indicates positive enthusiasm but not always; heuristic
    txt = str(x.text)
    if "!" in txt:
        # Count positive vs negative words
        if contains_word_set(txt, positive_words):
            return POS
        elif contains_word_set(txt, negative_words):
            return NEG
        else:
            return ABSTAIN
    return ABSTAIN

@labeling_function()
def lf_short_negative(x):
    # Very short messages like "Bad" or "Terrible" often negative
    txt = str(x.text).strip()
    if len(txt.split()) <= 2:
        if contains_word_set(txt, negative_words):
            return NEG
        elif contains_word_set(txt, positive_words):
            return POS
        else:
            return ABSTAIN
    return ABSTAIN

@labeling_function()
def lf_has_not_good(x):
    # phrases like "not good", "not great" -> negative
    txt = str(x.text).lower()
    if re.search(r"not\s+(good|great|excellent)", txt):
        return NEG
    return ABSTAIN

lfs = [lf_has_positive_words, lf_has_negative_words, lf_exclamation_positive, lf_short_negative, lf_has_not_good]

# --- Apply LFs ---
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df_train)
print("Label matrix shape:", L_train.shape)

# LF analysis
coverage = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print("\nLF summary:\n", coverage)

# --- Train LabelModel ---
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=200, log_freq=50, seed=42)

# Predict probabilistic labels on train set
probs_train = label_model.predict_proba(L=L_train)
# Convert to hard labels by argmax
preds_train = probs_train.argmax(axis=1)

# Save labeled train set
df_train_out = df_train.copy()
df_train_out["snorkel_label"] = preds_train
# Convert numeric labels to string labels for readability
label_map = {POS:"positive", NEG:"negative"}
df_train_out["snorkel_label_str"] = df_train_out["snorkel_label"].map(label_map)

out_path = "BDA_4_Autolabelling_Snorkel/train_labeled_by_snorkel.csv"
df_train_out.to_csv(out_path, index=False)
print(f"Saved snorkel labeled train set to {out_path}")

# --- Evaluate on test if available ---
if "label" in df_test.columns and not df_test["label"].isnull().all():
    # Map string labels to numeric
    label_to_num = {"positive":POS, "negative":NEG}
    y_test = df_test["label"].map(label_to_num).values

    # Apply LFs to test set
    L_test = applier.apply(df_test)
    probs_test = label_model.predict_proba(L=L_test)
    preds_test = probs_test.argmax(axis=1)

    print("\nEvaluation on test set:")
    print("Accuracy:", accuracy_score(y_test, preds_test))
    print(classification_report(y_test, preds_test, target_names=["negative","positive"]))
else:
    print("Test labels not available for automatic evaluation. Export labeled train set and/or label a dev set using Label Studio.")

# --- Quick summary ---
print("\nQuick label distribution in snorkel output:")
print(df_train_out["snorkel_label_str"].value_counts(dropna=False))

# End
