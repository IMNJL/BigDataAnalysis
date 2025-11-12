"""Run a small experiment with the LLM+Neo4j pipeline.

This script demonstrates the pipeline on mocked data. It will not attempt to contact an LLM unless
`LLM_API_KEY` env var is set. It will show generated triplets and Cypher statements. If you have
Neo4j running and `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` set, it can push the graph.
"""
import os
import json
import argparse
from typing import List

from pipeline import run_pipeline_on_text, nl_to_cypher, Triplet, push_to_neo4j


DEFAULT_INPUT = '/Users/pro/Downloads/archive-2/financial_news_events.json'


def read_jsonl(path: str, limit: int = None) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # try to recover by ignoring trailing commas or malformed lines
                continue
            items.append(obj)
            if limit and len(items) >= limit:
                break
    return items


def article_text_from_record(rec: dict) -> str:
    # Build a compact article text combining headline and market event fields
    parts = []
    for k in ['Headline', 'Market_Event', 'Related_Company', 'Source']:
        v = rec.get(k)
        if v:
            parts.append(str(v))
    # include Date as context
    if rec.get('Date'):
        parts.append(f"Date: {rec.get('Date')}")
    return ' - '.join(parts)


def save_jsonl(items: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description='Run KG pipeline on provided JSONL of news')
    p.add_argument('--input', '-i', default=os.getenv('INPUT_FILE', DEFAULT_INPUT))
    p.add_argument('--limit', '-n', type=int, default=100, help='max number of records to process')
    p.add_argument('--push', action='store_true', help='push resulting triplets to Neo4j if configured')
    p.add_argument('--outdir', default='outputs', help='output directory for triplets and cypher')
    args = p.parse_args()

    inpath = args.input
    if not os.path.exists(inpath):
        print(f'Input file not found: {inpath}')
        return

    records = read_jsonl(inpath, limit=args.limit)
    print(f'Loaded {len(records)} records from {inpath}')

    all_cleaned = []
    all_raw = []
    all_cypher = []

    for rec in records:
        text = article_text_from_record(rec)
        if not text.strip():
            continue
        out = run_pipeline_on_text(text, llm_api_key=os.getenv('LLM_API_KEY'))
        # out['cleaned_triplets'] is a list of dicts
        all_raw.extend(out.get('raw_triplets', []))
        cleaned = out.get('cleaned_triplets', [])
        all_cleaned.extend(cleaned)
        # Collect cypher statements
        all_cypher.extend(out.get('cypher', []))

    os.makedirs(args.outdir, exist_ok=True)
    triplets_path = os.path.join(args.outdir, 'triplets.jsonl')
    cypher_path = os.path.join(args.outdir, 'cypher.cypher')

    save_jsonl(all_cleaned, triplets_path)
    with open(cypher_path, 'w', encoding='utf-8') as f:
        for s in all_cypher:
            f.write(s + '\n')

    print(f'Wrote {len(all_cleaned)} cleaned triplets to {triplets_path}')
    print(f'Wrote {len(all_cypher)} cypher statements to {cypher_path}')

    # Optionally push to Neo4j
    if args.push:
        try:
            # Reconstruct Triplet objects from cleaned dicts
            triplet_objs = [Triplet(t['subject'], t['predicate'], t['object'], t.get('meta')) for t in all_cleaned]
            push_to_neo4j(triplet_objs)
            print('Pushed triplets to Neo4j')
        except Exception as e:
            print('Failed to push to Neo4j:', e)


if __name__ == '__main__':
    main()
