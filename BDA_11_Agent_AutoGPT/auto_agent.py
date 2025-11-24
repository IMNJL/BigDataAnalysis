"""Simulated AutoGPT-style agent for Titanic analysis.

This script 'plans' tasks and executes them autonomously. It writes:
- stats.json (key numeric findings)
- report_auto.md (markdown report referencing saved plots)
- plots saved in the output directory

Usage:
    python auto_agent.py --csv titanic.csv --out outputs/auto
"""
import argparse
import os
import json
import textwrap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plan_steps():
    return [
        "Load dataset",
        "Clean / inspect missing values",
        "Compute descriptive statistics",
        "Plot survival by sex",
        "Plot age distribution by survival",
        "Summarize findings and write report",
    ]


def compute_stats(df):
    def find_col(df, candidates):
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}
        # exact match
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        # substring match
        for col in cols:
            cl = col.lower()
            for cand in candidates:
                if cand.lower() in cl:
                    return col
        return None

    stats = {}
    stats['n_rows'] = int(len(df))

    survived_col = find_col(df, ['Survived', 'survived', 'surv', '2urvived', 'target'])
    sex_col = find_col(df, ['Sex', 'sex', 'gender'])
    age_col = find_col(df, ['Age', 'age'])
    pclass_col = find_col(df, ['Pclass', 'pclass', 'class'])

    if survived_col is not None:
        stats['survival_rate'] = float(pd.to_numeric(df[survived_col], errors='coerce').dropna().mean())
    if sex_col is not None and survived_col is not None:
        stats['survival_by_sex'] = df.groupby(sex_col)[survived_col].mean().fillna(0).to_dict()
    if age_col is not None:
        stats['age_mean'] = float(pd.to_numeric(df[age_col], errors='coerce').dropna().mean())
        stats['age_median'] = float(pd.to_numeric(df[age_col], errors='coerce').dropna().median())
    if pclass_col is not None and survived_col is not None:
        stats['survival_by_pclass'] = df.groupby(pclass_col)[survived_col].mean().fillna(0).to_dict()

    # store detected column names for traceability
    stats['_detected_columns'] = {
        'survived': survived_col,
        'sex': sex_col,
        'age': age_col,
        'pclass': pclass_col,
    }

    return stats


def save_plots(df, out_dir):
    plots = {}
    sns.set(style="whitegrid")

    # try to detect column names (case-insensitive / substring)
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

    if sex_col is not None and survived_col is not None:
        fig, ax = plt.subplots()
        # ensure numeric
        df_plot = df.copy()
        df_plot[survived_col] = pd.to_numeric(df_plot[survived_col], errors='coerce')
    sns.barplot(x=sex_col, y=survived_col, data=df_plot, errorbar=None, estimator=lambda x: sum(x)/len(x))
        ax.set_ylabel('Survival rate')
        p1 = os.path.join(out_dir, 'survival_by_sex.png')
        fig.savefig(p1)
        plt.close(fig)
        plots['survival_by_sex'] = os.path.basename(p1)

    if age_col is not None and survived_col is not None:
        fig, ax = plt.subplots()
        df_plot = df.copy()
        df_plot[survived_col] = pd.to_numeric(df_plot[survived_col], errors='coerce')
        sns.kdeplot(data=df_plot, x=age_col, hue=survived_col, common_norm=False, fill=True)
        ax.set_title('Age distribution by survival')
        p2 = os.path.join(out_dir, 'age_dist_by_survival.png')
        fig.savefig(p2)
        plt.close(fig)
        plots['age_dist_by_survival'] = os.path.basename(p2)

    return plots


def write_report(out_dir, stats, plots, plan_log):
    report_path = os.path.join(out_dir, 'report_auto.md')
    lines = []
    lines.append('# Auto Agent Report')
    lines.append('\n')
    lines.append('## Plan executed')
    lines.append('\n')
    for i,step in enumerate(plan_log,1):
        lines.append(f"{i}. {step}")
    lines.append('\n')
    lines.append('## Key statistics')
    lines.append('\n')
    for k,v in stats.items():
        lines.append(f"- **{k}**: {v}")
    lines.append('\n')
    lines.append('## Visualizations')
    lines.append('\n')
    for key, fname in plots.items():
        lines.append(f"### {key}")
        lines.append(f"![]({fname})")
        lines.append('\n')
    lines.append('## Brief findings')
    lines.append('\n')
    findings = []
    if 'survival_rate' in stats:
        findings.append(f"Overall survival rate: {stats['survival_rate']:.3f}.")
    if 'survival_by_sex' in stats:
        sbs = stats['survival_by_sex']
        findings.append('Survival by sex: ' + ', '.join([f"{k}={v:.3f}" for k,v in sbs.items()]))
    if findings:
        lines.append('\n'.join(['- ' + f for f in findings]))

    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write('\n\n'.join(lines))

    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    plan = plan_steps()
    # log plan
    plan_log_path = os.path.join(out_dir, 'plan.txt')
    with open(plan_log_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(plan))

    # Step: load dataset
    df = pd.read_csv(args.csv)

    # Quick cleaning: ensure Survived column is numeric if present
    if 'Survived' in df.columns:
        df['Survived'] = pd.to_numeric(df['Survived'], errors='coerce')

    stats = compute_stats(df)

    with open(os.path.join(out_dir, 'stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(stats, fh, indent=2)

    plots = save_plots(df, out_dir)

    report_path = write_report(out_dir, stats, plots, plan)

    print(f"Auto agent done. Report: {report_path}")


if __name__ == '__main__':
    main()
