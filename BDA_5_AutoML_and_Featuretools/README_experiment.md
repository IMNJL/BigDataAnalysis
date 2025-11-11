House price prediction — experiment README

This experiment compares house price prediction performance for several feature strategies:

A) Original features (as in the CSV)
B) Original + manual engineered features (simple ratios)
C) Original + Featuretools deep features (automatic feature synthesis)
D) AutoML model (FLAML) trained on original features (light budget)

How to run
1. Activate the repository `.venv`:

```bash
# from repository root
source .venv/bin/activate
```

2. Install dependencies for this experiment (recommended inside the .venv):

```bash
pip install -r "BDA_5_AutoML_and_Featuretools/Topic5 [light version]/requirements.txt"
```

3. Run the script (example):

```bash
python3 "BDA_5_AutoML_and_Featuretools/Topic5 [light version]/run_house_price_compare.py" \
  --data california_housing.csv \
  --out "BDA_5_AutoML_and_Featuretools/Topic5 [light version]/outputs"
```

Outputs
- `outputs/house_compare_results_<ts>.json` — JSON file with per-strategy results and summaries
- `outputs/house_compare_details_<ts>.csv` — per-fold metrics for strategies that produce per-fold results

Notes
- Featuretools generation may be slow on CPU for large datasets; reduce data size or skip featuretools if necessary.
- FLAML AutoML run uses a very small time_budget (30s) for a quick demo; increase for better results.
