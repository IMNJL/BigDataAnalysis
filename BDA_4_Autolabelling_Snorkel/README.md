# BDA_4_Autolabelling_Snorkel

This folder contains an auto-labeling experiment setup using Snorkel and Label Studio for the Amazon Review Polarity dataset (polarity sentiment: positive/negative).

Contents:
- `train.csv`, `test.csv` - csv datasets from Kaggle: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
- `snorkel_pipeline.py` - a runnable script that demonstrates labeling functions, LabelModel training, and evaluation.
- `label_studio_instructions.md` - step-by-step guide to create a Label Studio project and export labels for use with Snorkel.
- `experiment_report.md` - brief experiment report with results template and next steps.
- `requirements.txt` - Python dependencies for running the pipeline locally.

Notes:
- The full Amazon dataset (34M reviews) is too large to include in the repo. Use the provided small CSVs for experimentation and adapt the pipeline to larger data as needed.
