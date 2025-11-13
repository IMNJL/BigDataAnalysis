#!/usr/bin/env python3
"""
snorkel_pipeline.py
Run: python snorkel_pipeline.py --data_dir ../data --labels_file ../labelstudio_export/labels.csv
"""

import argparse
import pandas as pd
import numpy as np
import re
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

# Label mapping
ABSTAIN = -1
NEG = 0
POS = 1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--labels_file", default="../labelstudio_export/labels.csv", help="Label Studio export CSV (optional)")
    p.add_argument("--out_dir", default="../outputs")
    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()

def load_data(data_dir):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test

# ---------- Preprocessing helpers ----------
def preprocess_text(s):
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Labeling functions ----------
# These are simple examples; expand / tune for your dataset.
@labeling_function()
def lf_has_positive_word(x):
    positive_words = ["great", "excellent", "love", "loved", "awesome", "perfect", "best", "amazing", "fantastic"]
    txt = x.review_text_pre
    for w in positive_words:
        if re.search(r"\b" + re.escape(w) + r"\b", txt):
            return POS
    return ABSTAIN

@labeling_function()
def lf_has_negative_word(x):
    negative_words = ["bad", "terrible", "awful", "hate", "hated", "worst", "disappointed", "disappointing"]
    txt = x.review_text_pre
    for w in negative_words:
        if re.search(r"\b" + re.escape(w) + r"\b", txt):
            return NEG
    return ABSTAIN

@labeling_function()
def lf_contains_exclamation_positive(x):
    # Many exclamation marks often indicate positive sentiment, but it's heuristic
    txt = x.review_text
    if isinstance(txt, str) and txt.count("!") >= 2:
        return POS
    return ABSTAIN

@labeling_function()
def lf_contains_rating_very_high(x):
    # If dataset includes rating column, use it to create simple heuristics
    try:
        if not np.isnan(x.rating):
            if float(x.rating) >= 4.0:
                return POS
            if float(x.rating) <= 2.0:
                return NEG
    except Exception:
        pass
    return ABSTAIN

@labeling_function()
def lf_short_negative(x):
    # Very short reviews like "Terrible" or "awful" => negative
    txt = x.review_text_pre
    if len(txt.split()) <= 2 and len(txt) > 0:
        if re.search(r"\b(bad|awful|terrible|hate|hated|worst)\b", txt):
            return NEG
        if re.search(r"\b(great|love|excellent|best)\b", txt):
            return POS
    return ABSTAIN

# More advanced LFs can use embeddings, regex for emoticons, star ratings, product categories, or domain-specific lexicons.

def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    train, test = load_data(args.data_dir)

    # Minimal cleaning
    for df in [train, test]:
        if 'review_text' not in df.columns:
            raise ValueError("Expect column 'review_text' in CSVs")
        df['review_text_pre'] = df['review_text'].astype(str).apply(preprocess_text)

    # Optional: load Label Studio human labels and merge into a small gold set
    gold_df = None
    try:
        gold_df = pd.read_csv(args.labels_file)
        # Expect columns: review_id, sentiment
        # Normalize column names:
        if 'review_id' in gold_df.columns and 'sentiment' in gold_df.columns:
            gold_df = gold_df[['review_id','sentiment']].rename(columns={'sentiment':'gold_sentiment'})
            # Map text labels to 0/1
            gold_df['gold_label'] = gold_df['gold_sentiment'].map({'negative': 0, 'positive': 1})
            print(f"Loaded {len(gold_df)} human-labeled examples from Label Studio.")
        else:
            gold_df = None
    except Exception as e:
        print("No Label Studio labels loaded:", e)
        gold_df = None

    # Apply LFs
    lfs = [lf_has_positive_word, lf_has_negative_word, lf_contains_exclamation_positive,
           lf_contains_rating_very_high, lf_short_negative]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(train)
    L_test = applier.apply(test)
    print("LF matrix shapes:", L_train.shape, L_test.shape)

    # LF analysis
    print("\nLF Analysis (train):")
    print(LFAnalysis(L_train, lfs).lf_summary())

    # Train Snorkel LabelModel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=500, log_freq=100, seed=args.random_seed)
    probs_train = label_model.predict_proba(L_train)
    probs_test = label_model.predict_proba(L_test)

    # Save probabilistic labels for train set
    train['prob_neg'] = probs_train[:,0]
    train['prob_pos'] = probs_train[:,1]
    train['pseudo_label'] = train['prob_pos'] >= 0.5

    # Optional: if you have small human-labeled gold set, calibrate threshold or evaluate label_model
    if gold_df is not None:
        merged = train.merge(gold_df, on='review_id', how='inner')
        if len(merged) > 0:
            preds_from_label_model = (merged['prob_pos'] >= 0.5).astype(int)
            from sklearn.metrics import accuracy_score, f1_score
            print("Label Model vs gold --- acc:", accuracy_score(merged['gold_label'], preds_from_label_model),
                  "f1:", f1_score(merged['gold_label'], preds_from_label_model))

    # Train a downstream classifier using pseudo-labels
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train['review_text_pre'])
    y_train = train['pseudo_label'].astype(int)

    # Use subset with confident probabilities for training (optional)
    # E.g., keep only examples with prob_pos >= 0.9 or <= 0.1
    confidence_mask = (train['prob_pos'] >= 0.9) | (train['prob_pos'] <= 0.1)
    if confidence_mask.sum() >= 1000:
        X_train_conf = X_train[confidence_mask.values]
        y_train_conf = y_train[confidence_mask.values]
        print(f"Training on {X_train_conf.shape[0]} confident pseudo-labeled examples.")
    else:
        X_train_conf = X_train
        y_train_conf = y_train
        print(f"Training on all pseudo-labeled examples: {X_train_conf.shape[0]}")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_conf, y_train_conf)

    # Evaluate on test set
    X_test = vectorizer.transform(test['review_text_pre'])
    # If test has gold labels (some datasets include polarity label), use them; else, use Label Studio exported test labels.
    y_test = None
    if 'label' in test.columns:
        y_test = test['label'].astype(int)
    if y_test is None:
        print("Test has no gold labels; evaluation requires human-labeled test set.")
    else:
        y_pred = clf.predict(X_test)
        print("Classifier evaluation on test set:")
        print(classification_report(y_test, y_pred, digits=4))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save artifacts
    joblib.dump(vectorizer, f"{args.out_dir}/tfidf_vectorizer.joblib")
    joblib.dump(clf, f"{args.out_dir}/logreg_clf.joblib")
    joblib.dump(label_model, f"{args.out_dir}/snorkel_label_model.joblib")
    print("Saved models and artifacts to", args.out_dir)

if __name__ == "__main__":
    main()
