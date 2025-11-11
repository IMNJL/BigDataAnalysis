#!/usr/bin/env bash
# Minimal driver to run CF, generate tags, and build DOCX
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw/ml-100k"
OUT_DIR="$PROJECT_ROOT/reports"
mkdir -p "$OUT_DIR"

# optional venv creation (skip if you already have env)
python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scipy scikit-learn python-docx transformers sentence-transformers >/dev/null 2>&1 || true

# default user id (can be overridden by first positional arg)
USER_ID=${1:-196}

# Ensure scripts exist
python3 src/cf/user_cf.py --data "$DATA_DIR" --user-id "$USER_ID" --out "$OUT_DIR"
python3 src/llm/generate_tags.py --data "$DATA_DIR" --user-id "$USER_ID" --out "$OUT_DIR"
python3 src/report/generate_docx.py --out "$OUT_DIR" --project-root "$PROJECT_ROOT" --user-id "$USER_ID"

echo "Report generated at: $PROJECT_ROOT/BDA_9_Recommendation_report.docx"