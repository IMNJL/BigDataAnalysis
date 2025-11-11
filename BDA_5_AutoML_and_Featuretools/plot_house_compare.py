#!/usr/bin/env python3
"""
Plot house price comparison results.
Reads the JSON produced by `run_house_price_compare.py` and creates bar charts for RMSE and R^2.
Saves PNG files and a short summary into the same outputs folder.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_metrics(results):
    strategies = []
    rmse = []
    r2 = []
    for k, v in results.items():
        if isinstance(v, dict) and 'summary' in v:
            strategies.append(k)
            rmse.append(v['summary'].get('rmse_mean'))
            r2.append(v['summary'].get('r2_mean'))
    return strategies, rmse, r2


def plot_bar(xlabels, values, ylabel, out_path, title=None):
    x = np.arange(len(xlabels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(x, values, color='tab:blue')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # add value labels
    for rect in bars:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(out_path, strategies, rmse, r2):
    lines = []
    lines.append('House price comparison — summary')
    lines.append('')
    for s, r_m, r2_v in zip(strategies, rmse, r2):
        lines.append(f'- {s}: RMSE={r_m:.4f}, R2={r2_v:.4f}')
    text = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help='Path to house_compare_results_*.json')
    args = parser.parse_args()
    results = load_results(args.json_path)
    strategies, rmse, r2 = prepare_metrics(results)
    out_dir = os.path.dirname(args.json_path)
    if not strategies:
        print('No strategies with summary metrics found in JSON:', args.json_path)
        return
    # Plot RMSE
    rmse_png = os.path.join(out_dir, 'house_compare_rmse.png')
    plot_bar(strategies, rmse, 'RMSE', rmse_png, title='RMSE by feature strategy')
    print('Wrote', rmse_png)
    # Plot R2
    r2_png = os.path.join(out_dir, 'house_compare_r2.png')
    plot_bar(strategies, r2, 'R^2', r2_png, title='R^2 by feature strategy')
    print('Wrote', r2_png)
    # write summary
    summary_txt = os.path.join(out_dir, 'house_compare_summary.txt')
    write_summary(summary_txt, strategies, rmse, r2)
    print('Wrote', summary_txt)


if __name__ == '__main__':
    main()
