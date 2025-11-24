"""Orchestrator for AutoGPT vs Manual analysis experiment.

Usage:
    python experiment.py --csv titanic.csv

This script runs both agents, times them, and calls the comparator.
"""
import argparse
import time
import subprocess
import sys
import os

HERE = os.path.dirname(__file__)
AUTO_OUT = os.path.join(HERE, "outputs", "auto")
MANUAL_OUT = os.path.join(HERE, "outputs", "manual")
COMP_OUT = os.path.join(HERE, "outputs")


def ensure_dirs():
    os.makedirs(AUTO_OUT, exist_ok=True)
    os.makedirs(MANUAL_OUT, exist_ok=True)
    os.makedirs(COMP_OUT, exist_ok=True)


def run_script(script, csv_path, out_dir):
    cmd = [sys.executable, script, "--csv", csv_path, "--out", out_dir]
    print("Running:", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, check=False)
    duration = time.time() - start
    return proc.returncode, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to titanic.csv")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(2)

    ensure_dirs()

    print("1) Running simulated AutoGPT agent...")
    rc_auto, t_auto = run_script(os.path.join(HERE, "auto_agent.py"), csv_path, AUTO_OUT)
    print(f"Auto agent finished (rc={rc_auto}) in {t_auto:.2f}s")

    print("2) Running manual guided analysis...")
    rc_man, t_man = run_script(os.path.join(HERE, "manual_agent.py"), csv_path, MANUAL_OUT)
    print(f"Manual analysis finished (rc={rc_man}) in {t_man:.2f}s")

    print("3) Comparing outputs...")
    cmp_cmd = [sys.executable, os.path.join(HERE, "compare_reports.py"), "--auto", AUTO_OUT, "--manual", MANUAL_OUT, "--out", COMP_OUT]
    subprocess.run(cmp_cmd)

    print('\nSummary:')
    print(f"  Auto agent:  rc={rc_auto} time={t_auto:.2f}s")
    print(f"  Manual run:  rc={rc_man} time={t_man:.2f}s")
    print(f"  Comparison written to {COMP_OUT}/comparison_summary.md")


if __name__ == '__main__':
    main()
