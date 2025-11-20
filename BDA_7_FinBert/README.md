BDA_7_FinBert

This folder contains a FinBERT experiment scaffold: a simple pipeline to run a FinBERT sentiment model on financial news, an experiment runner that processes the provided JSONL dataset, and a full experiment report following the Homework schema.

Files:
- `finbert_pipeline.py` - pipeline functions to load data, preprocess and run FinBERT inference.
- `experiment.py` - CLI runner that loads `/Users/pro/Downloads/archive-2/financial_news_events.json` by default and writes outputs to `outputs/`.
- `requirements.txt` - Python dependencies for the FinBERT experiment.
- `report.md` - Full experiment writeup following the required Homework schema.
- `architecture_diagram.dot` - Simple Graphviz DOT describing pipeline architecture.

Quick start (macOS zsh):

1. Create and activate venv in repo root (recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r BDA_7_FinBert/requirements.txt

3. Run experiment (mock or real FinBERT model from Hugging Face):

   python BDA_7_FinBert/experiment.py --input /Users/pro/Downloads/archive-2/financial_news_events.json --limit 200

Outputs will be saved to `BDA_7_FinBert/outputs/` (predictions and a class distribution plot).

Notes:
- The script uses a FinBERT model from Hugging Face (default `yiyanghkust/finbert-tone`).
- If you have GPU and want faster inference, ensure `torch` with CUDA is installed and available.
- For Chinese data, replace model with a Chinese Finance-adapted BERT if available or translate text before inference.