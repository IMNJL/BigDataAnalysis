BDA_5_Topic5_Regression_Clustering

This project implements Topic 5 experiments and report generation:

Tasks implemented:
- Summaries: Linear, Lasso, Polynomial, Neural Network regression; KMeans vs DeepCluster.
- Experiment 1: Compare Linear Regression and Neural Network Regression on synthetic data and California Housing.
- Experiment 2: Compare KMeans and a DeepCluster-like method on MNIST and Fashion-MNIST.
- Report generation: `make_report.py` creates a DOCX following the BDA experiment template.

Quick start

1) create venv and install requirements

```bash
cd BDA_5_Topic5_Regression_Clustering
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run Experiment 1

```bash
python3 run_experiment1.py --output-dir outputs/exp1
```

3) Run Experiment 2 (may be slower)

```bash
python3 run_experiment2.py --output-dir outputs/exp2 --dataset mnist
```

4) Generate DOCX report

```bash
python3 make_report.py --exp1 outputs/exp1/results_exp1.json --exp2 outputs/exp2/results_exp2.json --out report_topic5.docx
```

Notes
- The DeepCluster implementation is a lightweight proxy (autoencoder + clustering) to avoid heavy DL dependencies; results are for comparative study.
- If you want full DeepCluster with PyTorch, I can add it as an optional script.
