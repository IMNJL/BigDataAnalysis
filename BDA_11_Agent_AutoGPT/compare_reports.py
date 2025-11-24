"""Compare stats.json and basic report artifacts from auto and manual runs.

Writes `comparison_summary.md` into the `--out` directory.
"""
import argparse
import json
import os


def load_stats(path):
    p = os.path.join(path, 'stats.json')
    if not os.path.exists(p):
        return None
    with open(p, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def word_count_md(p):
    if not os.path.exists(p):
        return 0
    with open(p, 'r', encoding='utf-8') as fh:
        txt = fh.read()
    return len(txt.split())


def compare(auto_path, manual_path, out_dir):
    auto_stats = load_stats(auto_path) or {}
    manual_stats = load_stats(manual_path) or {}

    auto_report = os.path.join(auto_path, 'report_auto.md')
    manual_report = os.path.join(manual_path, 'report_manual.md')

    summary_lines = []
    summary_lines.append('# Comparison summary')

    summary_lines.append('## High-level metrics')
    ac_words = word_count_md(auto_report)
    mn_words = word_count_md(manual_report)
    summary_lines.append(f'- Auto report word count: {ac_words}')
    summary_lines.append(f'- Manual report word count: {mn_words}')

    summary_lines.append('\n## Numeric stats comparison')
    keys = set(list(auto_stats.keys()) + list(manual_stats.keys()))
    for k in sorted(keys):
        a = auto_stats.get(k)
        m = manual_stats.get(k)
        summary_lines.append(f'- **{k}**: auto={a} | manual={m}')

    # Basic interpretation: are the main survival rates equal?
    if 'survival_rate' in auto_stats and 'survival_rate' in manual_stats:
        diff = abs(auto_stats['survival_rate'] - manual_stats['survival_rate'])
        summary_lines.append('\n## Interpretation')
        summary_lines.append(f'- Survival rate difference: {diff:.6f}')
        if diff < 1e-6:
            summary_lines.append('- Numbers match closely (within floating noise).')
        else:
            summary_lines.append('- There is a measurable difference; investigate data handling and missing values.')

    out_path = os.path.join(out_dir, 'comparison_summary.md')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(summary_lines))

    print('Wrote comparison summary to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto', required=True)
    parser.add_argument('--manual', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    compare(args.auto, args.manual, args.out)

if __name__ == '__main__':
    main()
