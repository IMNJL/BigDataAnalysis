# BDA_11_Agent_AutoGPT

This folder contains an experiment to compare an AutoGPT-like autonomous agent vs a manually guided analysis for the Titanic dataset.

Overview
- Experiment 1: Run the simulated AutoGPT agent which plans tasks and executes them autonomously, producing a markdown report and artifacts.
- Experiment 2: Run the manual guided analysis that performs the same analysis steps but with a scripted, step-by-step implementation. Produce a manual report and artifacts.
- Comparison: A small script compares generated `stats.json` outputs and the reports and produces a short comparison summary.

Files
- `experiment.py` — orchestrator: runs both agents (auto & manual), times them, and runs the comparator.
- `auto_agent.py` — simulated AutoGPT: prints a plan, executes analysis steps autonomously, logs actions, produces `outputs/auto/`.
- `manual_agent.py` — manual analysis: scripted step-by-step analysis, produces `outputs/manual/`.
- `compare_reports.py` — compares `stats.json` from each agent and writes `outputs/comparison_summary.md`.
- `requirements.txt` — Python dependencies.

How to prepare
1. Place your Titanic CSV in this folder and name it `titanic.csv`.
   - Expected format: typical Titanic dataset with columns such as `Survived`, `Sex`, `Age`, `Pclass`, etc.
2. (Optional) Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick run

```bash
python experiment.py --csv titanic.csv
```

This will run the simulated AutoGPT agent, the manual agent, then compare their outputs and produce `outputs/` containing reports, plots and a comparison summary.

Notes
- This repository contains a simulation of how an AutoGPT agent might plan and execute tasks locally. It does not call external AutoGPT projects or remote APIs.
- If you want to use an actual AutoGPT/agent framework, you can adapt `auto_agent.py` to invoke that system; instructions and hooks are provided in comments.

Experimental design and suggested evaluation criteria are included inside `experiment.py` and `compare_reports.py`.
