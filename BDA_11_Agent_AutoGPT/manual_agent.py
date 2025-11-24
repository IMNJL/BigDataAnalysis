"""Manual (scripted) analysis for Titanic dataset.

This script performs an explicit step-by-step analysis and writes:
- stats.json
- report_manual.md
- plots saved under the given output directory

Usage:
    python manual_agent.py --csv titanic.csv --out outputs/manual
"""
import argparse
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_and_save(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    if 'Survived' in df.columns:
        df['Survived'] = pd.to_numeric(df['Survived'], errors='coerce')

    def find_col(df, candidates):
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        for col in cols:
            cl = col.lower()
            for cand in candidates:
                if cand.lower() in cl:
                    return col
        return None

    survived_col = find_col(df, ['Survived', 'survived', 'surv', '2urvived', 'target'])
    sex_col = find_col(df, ['Sex', 'sex', 'gender'])
    age_col = find_col(df, ['Age', 'age'])

    stats = {}
    stats['n_rows'] = int(len(df))
    if survived_col is not None:
        stats['survival_rate'] = float(pd.to_numeric(df[survived_col], errors='coerce').dropna().mean())
    if sex_col is not None and survived_col is not None:
        s = df.groupby(sex_col)[survived_col].mean().fillna(0).to_dict()
        stats['survival_by_sex'] = s

    # Save stats
    with open(os.path.join(out_dir, 'stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(stats, fh, indent=2)

    # Plots
    # Plots (robust column detection)
    if sex_col is not None and survived_col is not None:
        fig, ax = plt.subplots()
        df_plot = df.copy()
        df_plot[survived_col] = pd.to_numeric(df_plot[survived_col], errors='coerce')
    sns.barplot(x=sex_col, y=survived_col, data=df_plot, errorbar=None, estimator=lambda x: sum(x)/len(x))
        ax.set_ylabel('Survival rate')
        p = os.path.join(out_dir, 'survival_by_sex_manual.png')
        fig.savefig(p)
        plt.close(fig)

    if age_col is not None and survived_col is not None:
        fig, ax = plt.subplots()
        df_plot = df.copy()
        df_plot[survived_col] = pd.to_numeric(df_plot[survived_col], errors='coerce')
        sns.kdeplot(data=df_plot, x=age_col, hue=survived_col, common_norm=False, fill=True)
        p2 = os.path.join(out_dir, 'age_dist_by_survival_manual.png')
        fig.savefig(p2)
        plt.close(fig)

    # Report
    report_path = os.path.join(out_dir, 'report_manual.md')
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write('# Manual Analysis Report\n\n')
        fh.write('## Steps executed\n')
        fh.write('1. Load data\n2. Convert types\n3. Compute descriptive statistics\n4. Create plots\n\n')
        fh.write('## Key statistics\n')
        for k,v in stats.items():
            fh.write(f'- **{k}**: {v}\n')
        fh.write('\n## Visualizations\n')
        if os.path.exists(os.path.join(out_dir, 'survival_by_sex_manual.png')):
            fh.write('![](' + 'survival_by_sex_manual.png' + ')\n')
        if os.path.exists(os.path.join(out_dir, 'age_dist_by_survival_manual.png')):
            fh.write('![](' + 'age_dist_by_survival_manual.png' + ')\n')

    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rp = compute_and_save(args.csv, args.out)
    print('Manual analysis done. Report:', rp)


if __name__ == '__main__':
    main()
