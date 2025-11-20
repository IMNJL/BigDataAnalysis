"""Run a FinBERT experiment on the attached financial_news_events.json dataset.

Default behavior: load dataset, preprocess records, run FinBERT inference (mock if no internet or model available), save outputs to outputs/ and draw a simple class distribution plot.
"""

import os
import json
import argparse
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from finbert_pipeline import load_jsonl, preprocess_text, load_model, predict_texts, save_predictions


DEFAULT_INPUT = '/Users/pro/Downloads/archive-2/financial_news_events.json'


def plot_distribution(preds: List[dict], outpath: str):
    labels = [p['pred_label'] for p in preds]
    cnt = Counter(labels)
    keys = list(cnt.keys())
    vals = [cnt[k] for k in keys]
    plt.figure(figsize=(6,4))
    sns.barplot(x=keys, y=vals)
    plt.title('FinBERT predicted class distribution')
    plt.ylabel('count')
    plt.xlabel('class')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default=os.getenv('INPUT_FILE', DEFAULT_INPUT))
    p.add_argument('--limit', '-n', type=int, default=200)
    p.add_argument('--model', default=os.getenv('FINBERT_MODEL', 'yiyanghkust/finbert-tone'))
    p.add_argument('--outdir', default='outputs')
    args = p.parse_args()

    records = load_jsonl(args.input, limit=args.limit)
    print(f'Loaded {len(records)} records from {args.input}')

    texts = [preprocess_text(r) for r in records]

    try:
        tok, model, device = load_model(args.model)
        preds = predict_texts(texts, tok, model, device=device)
    except Exception as e:
        print('Model load or inference failed, falling back to mock predictions:', e)
        # create mock predictions: neutral for all
        preds = [{'text': t, 'pred': 1, 'pred_label': 'neutral', 'probs': [0.1, 0.8, 0.1]} for t in texts]

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    preds_path = os.path.join(outdir, 'finbert_predictions.jsonl')
    save_predictions(preds, preds_path)
    print('Saved predictions to', preds_path)

    plot_path = os.path.join(outdir, 'finbert_class_distribution.png')
    plot_distribution(preds, plot_path)
    print('Saved distribution plot to', plot_path)


if __name__ == '__main__':
    main()
