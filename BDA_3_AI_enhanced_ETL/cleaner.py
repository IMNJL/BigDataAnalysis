"""
Cleaner / extractor script for PDFs downloaded by the crawler.
- Walks `data/raw_pdfs/`, extracts text, runs simple cleaning, extracts comment-like lines and metadata,
  and saves results as one JSON file per PDF under `data/processed/`.

Optional extension point: integrate an LLM or a semantic extractor (e.g. huggingface) to parse
comments/sections more accurately; a placeholder is marked where to call your model.
"""
import os
import argparse
import json
from pathlib import Path
from extract_pdf import extract_text_from_pdf, basic_clean_text, extract_comments_from_text

DEFAULT_IN = os.path.join(os.path.dirname(__file__), 'data', 'raw_pdfs')
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), 'data', 'processed')

os.makedirs(DEFAULT_IN, exist_ok=True)
os.makedirs(DEFAULT_OUT, exist_ok=True)


def process_file(path: str, out_dir: str):
    print(f"Processing {path}")
    text = extract_text_from_pdf(path)
    clean = basic_clean_text(text)
    comments = extract_comments_from_text(clean)

    # placeholder: AI-enhanced extraction
    # e.g. call an LLM with prompt: "Extract comments and metadata from the following text..."
    # For now we'll keep the heuristic comments and first 2000 chars as `summary`.
    summary = clean[:2000]

    out = {
        'source': os.path.basename(path),
        'summary': summary,
        'comments': comments,
        'full_text_length': len(clean),
    }
    out_path = os.path.join(out_dir, os.path.basename(path) + '.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', default=DEFAULT_IN)
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUT)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in in_dir.glob('**/*.pdf')])
    if not pdfs:
        print('No PDFs found in', in_dir)
    for p in pdfs:
        try:
            process_file(str(p), str(out_dir))
        except Exception as e:
            print('Failed to process', p, e)
