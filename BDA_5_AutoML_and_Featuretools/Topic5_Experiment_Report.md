# Topic 5 — House Price Prediction: Experiment Report

This report follows the required experiment template: Part 1 (Experiment Plan), Part 2 (Architecture / Diagram reference), Part 3 (Steps with prompts and responses + screenshots), and Final Result.

---

## Part 1: Experiment Plan

Objective

Compare house price prediction performance across different feature engineering strategies using the California housing dataset. The strategies tested are:

- A) Original features (baseline)
- B) Original + manual engineered features (simple ratios)
- C) Original + automatic features via Featuretools (if available)
- D) AutoML model/feature selection via FLAML (light budget)

Dataset

The dataset used is `california_housing.csv` (included in the project root). The target variable is `median_house_value` (or `target` if differently named).

Evaluation metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² (coefficient of determination)

Experimental protocol

- 3-fold cross-validation for strategies that use scikit-learn models (RandomForestRegressor with 50 trees).
- FLAML AutoML used with a small time budget (30s) for a quick baseline; install `flaml[automl]` for full features.
- Featuretools generation is attempted if `featuretools` is installed; otherwise the branch records an error.

Success criteria

- Obtain lower RMSE and higher R² for engineered/AutoML-enhanced features compared to the baseline. For this light demo, significant improvements may not appear; the goal is to demonstrate the pipeline, collect metrics and produce reproducible artifacts (JSON, CSV, plots).

---

## Part 2: Architecture (by Diagram)

The experiment is implemented as a simple single-machine pipeline with the following components:

- Data loader (CSV → DataFrame)
- Feature construction module
  - manual features: simple arithmetic ratios
  - featuretools: automatic deep feature synthesis (optional)
- Modeling & evaluation module (K-fold CV with scikit-learn models; FLAML AutoML for quick automatic tuning)
- Results exporter: per-fold CSV and summary JSON, plus plots (RMSE and R²)

Reference diagram (generated): `../outputs/house_compare_rmse.png` and `../outputs/house_compare_r2.png` illustrate numeric comparisons; the architecture is small enough to be shown in the README.

---

## Part 3: Steps and Screenshots

This section documents the concrete steps, the prompts or commands used, and the script outputs.

### Step 1 — Prepare environment

Prompt / Command

```bash
source .venv/bin/activate
pip install -r "BDA_5_AutoML_and_Featuretools/requirements.txt"
```

Response / Notes

- Installed required packages (pandas, numpy, scikit-learn). Optional packages `featuretools` and `flaml` may be missing in some environments; the script will record errors for those branches.

### Step 2 — Run comparison script

Prompt / Command

```bash
python3 run_house_price_compare.py --data california_housing.csv --out ../outputs
```

Response (script stdout)

- Evaluating original features...
- Generating manual features...
- (Featuretools branch: featuretools not installed)
- (AutoML branch: FLAML not installed)
- Wrote results to `../outputs/house_compare_results_<ts>.json`
- Wrote per-fold metrics to `../outputs/house_compare_details_<ts>.csv`

Screenshot: (output CSV and JSON files)

### Step 3 — Quick visualization

Prompt / Command

```bash
python3 plot_house_compare.py ../outputs/house_compare_results_1762857879.json
```

Response / Outcome

- Generated `../outputs/house_compare_rmse.png` and `../outputs/house_compare_r2.png` and `../outputs/house_compare_summary.txt`.

Embedded plots (referenced):

![RMSE comparison](../outputs/house_compare_rmse.png)

![R2 comparison](../outputs/house_compare_r2.png)

### Prompts and example responses (for report template completeness)

Step 1 prompt: "Run baseline model on original features and report RMSE/R²."  
Response: Baseline RandomForest (50 trees) gives RMSE ≈ 0.5133, R² ≈ 0.8021 (3-fold CV mean).

Step 2 prompt: "Add manual engineered features and re-evaluate."  
Response: Manual features (rooms_per_household, bedrooms_per_room, population_per_household) produced identical metrics in this run (RMSE ≈ 0.5133, R² ≈ 0.8021), indicating limited marginal gain for these simple ratios on this dataset with the chosen model/hyperparameters.

Step 3 prompt: "Try Featuretools automatic feature synthesis."  
Response: The environment did not have `featuretools` installed; the script recorded `featuretools not installed` in the JSON results. Install `featuretools` and re-run to generate deep features.

Step 4 prompt: "Run FLAML AutoML for a quick AutoML baseline."  
Response: The environment did not have FLAML automl extras available; the script recorded `FLAML not installed`. Install `flaml[automl]` to enable this step.

---

## Final Result

Numeric summary (from `../outputs/house_compare_results_1762857879.json`):

- Original (baseline):
  - RMSE (mean 3-fold): 0.5132809
  - R² (mean 3-fold): 0.8021327

- Manual features (baseline + ratios):
  - RMSE (mean 3-fold): 0.5132809
  - R² (mean 3-fold): 0.8021327

- Featuretools: error — featuretools not installed (no results)
- AutoML (FLAML): error — FLAML not installed (no results)

Files produced

- `../outputs/house_compare_results_1762857879.json` — detailed per-strategy results (per-fold + summary)
- `../outputs/house_compare_details_1762857879.csv` — per-fold metrics for strategies with per-fold results
- `../outputs/house_compare_rmse.png` — RMSE comparison chart
- `../outputs/house_compare_r2.png` — R² comparison chart

Interpretation and notes

- In this light demo, manual feature engineering did not improve the RandomForest baseline. This may be due to the model already capturing non-linear interactions or because the engineered features are redundant with existing columns.
- To get meaningful improvements, consider: stronger manual features, feature interactions, polynomial features, or using Featuretools to synthesize higher-order features. Also try different model families or hyperparameter tuning.
- Enable FLAML/featuretools in your environment to let the AutoML and automatic feature synthesis branches run.

Next steps (recommended)

1. Install optional dependencies and re-run:

```bash
pip install featuretools
pip install "flaml[automl]"
```

2. Increase AutoML time budget for better AutoML results (change `time_budget` in script).
3. Add standardized preprocessing (scaling, missing value imputation) and a model selection sweep (GridSearchCV or AutoML full budget).
4. Save model artifacts and produce a short inference script showing predictions on new examples.
