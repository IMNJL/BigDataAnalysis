"""
FinBERT pipeline scaffold

Provides:
- load_jsonl: read JSONL of news events
- preprocess_text: normalize and assemble text from record
- load_model: load FinBERT tokenizer+model
- predict_texts: run inference and return label/probabilities
- save_predictions: write JSONL predictions

Notes: This scaffold uses Hugging Face transformers. Default model is 'yiyanghkust/finbert-tone' but you can change to another FinBERT variant.
"""

import json
import os
import re
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_jsonl(path: str, limit: int = None) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)
            if limit and len(items) >= limit:
                break
    return items


def preprocess_text(rec: dict) -> str:
    # Combine Headline, Market_Event, Related_Company, Source and Date
    parts = []
    for k in ['Headline', 'Market_Event', 'Related_Company', 'Source']:
        v = rec.get(k)
        if v:
            parts.append(str(v))
    if rec.get('Date'):
        parts.append(f"Date: {rec.get('Date')}")
    text = ' - '.join(parts)
    # Normalize whitespace and control characters
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def load_model(model_name: str = 'yiyanghkust/finbert-tone', device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_texts(texts: List[str], tokenizer, model, device: str = 'cpu', batch_size: int = 32) -> List[Dict]:
    results = []
    label_map = None
    if hasattr(model.config, 'id2label'):
        label_map = {int(k): v for k, v in model.config.id2label.items()}
    else:
        # default mapping common for FinBERT (check your model)
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            preds = logits.argmax(dim=-1).cpu().tolist()
            for t, p, pred in zip(batch, probs, preds):
                results.append({'text': t, 'pred': int(pred), 'pred_label': label_map.get(int(pred), str(pred)), 'probs': p})
    return results


def save_predictions(preds: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # quick demo
    sample = ["Acme Inc. reported a 15% increase in profit for Q4."]
    tok, model, dev = load_model()
    out = predict_texts(sample, tok, model, device=dev)
    print(out)
