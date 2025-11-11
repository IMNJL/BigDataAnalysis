# Big Data Analysis Technology — Course Materials

This repository contains course materials, code examples and experiments for the Big Data Analysis Technology course (English). The course follows Bloom's taxonomy to build a competency hierarchy: remembering, understanding, applying, analyzing, evaluating and creating. It combines theoretical explanations with engineering practice so learners gain both conceptual knowledge and hands-on skills.

This README provides a course overview (English and Chinese), the theory roadmap, a detailed list of experiments and practical topics, and a table of hands-on topics beginning from crawler and cleaner. Use the per-folder `make_report.py` scripts to generate DOCX reports summarizing experiments (requires `python-docx`).

---

## 1. Course Overview (English)

The course is organized around a three-layer architecture:

1. Data storing system
2. Data processing system
3. Data application system

Each layer is described below together with the subcomponents and typical technologies.

### 1.1 Data storing system

- 1.1 Data collection and modeling
- 1.2 Distributed file system
- 1.3 Distributed database and data warehouse
- 1.4 Unified data access interface

### 1.2 Data processing system

- 2.1 Data analysis algorithms
- 2.2 Computing models
- 2.3 Computing engines and platforms (Spark, Hadoop, Dask, Ray)

### 1.3 Data application system

- 3.1 Big data visualization
- 3.2 Big data products and services
- 3.3 Big data applications (recommendation systems, social network analysis, etc.)

The theoretical part of the course explains the principles and algorithms used in these layers. For complex topics we selected high-quality explanatory videos and visual materials to help learners understand the underlying ideas intuitively.

---

## 2. Experiments (English)

Five major experimental modules are designed to develop practical engineering skills:

1) Dynamic web crawler
2) Spark MLlib learning and application
3) TensorFlow learning and application
4) Recommendation system understanding and construction
5) Social network analysis and visualization

For each module we provide a set of experiments that go from simple to advanced. Every experiment includes:

- Experimental design and objectives
- Step-by-step manual
- Source code and sample data (where appropriate)
- Expected outputs and suggestions for evaluation

The experiments train students to apply theoretical knowledge to realistic problems and to build reproducible pipelines.

---

## 3. 教学目标与课程目标 (Chinese / 授课目标)

学习本课程后，学生应能够理解大数据分析的基本概念、理论与平台，并能在实验手册与代码示例的帮助下完成工程化的大数据应用开发。

---

## 4. 课程大纲 (Course Outline)

Below is a concise course outline translated to English plus Chinese key points for each lesson block.

### Module: Big Data Introduction
Objective: Let students know what big data is and why it matters.

Contents:
- 1.1 Basic concept
- 1.2 Structured vs Unstructured data
- 1.3 The Fourth Paradigm
- 1.4 Big data characteristics
- 1.5 Big data lifecycle
- 1.6 Processing flow
- 1.7 Architecture

### Module: Data Collection
Objective: Understand data sources and acquisition methods, including deep web and dynamic crawling.

Contents:
- 2.1 Data resources
- 2.2 Internal data acquisition
- 2.3 External data acquisition
- 2.4 Deep web and dynamic crawler

### Module: Data Preprocessing
Objective: Learn cleaning, normalization, feature extraction, tokenization and data shaping for ML pipelines.

Contents include standard preprocessing workflows and hands-on exercises using Python tools.

---

## 5. Experiments: brief descriptions

- Dynamic web crawler: building robust crawlers (selenium/requests/beautifulsoup), politeness, rate limiting, parsing JavaScript-driven sites, storing raw HTML / PDF / media.
- Spark MLlib: classical ML algorithms at scale (regression, classification, clustering), feature pipelines and model evaluation.
- TensorFlow: building and training neural networks, dataset APIs, TFRecord, and basic distributed training concepts.
- Recommendation systems: collaborative filtering, content-based and hybrid methods, offline evaluation (Precision@K, NDCG), and demo pipelines.
- Social network analysis and visualization: graph processing, centrality, community detection, and visualization with networkx / Gephi / plotting libraries.

Each experimental folder in this repository contains code, a README, and scripts to reproduce experiments. See the individual subfolders (e.g., `BDA_3_AI_enhanced_ETL`, `BDA_6_Topic5_Regression_Clustering`, `BDA_9_Reccomendation_System_Assigment`) for details.

---

## 6. Table of practical topics (beginning from Crawler and Cleaner)

The table below lists practical topics and sub-topics covered by the experiments. Use it as a checklist for hands-on skills and to navigate folders in the repository.

| Topic group | Subtopics / Exercises | Key files / folders |
|---|---|---|
| [Crawler & Data Ingestion](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_1_crawler_and_cleaner) | Dynamic web crawler (requests/selenium), politeness, link extraction, PDF/asset download | `BDA_3_AI_enhanced_ETL/crawler.py`, `BDA_3_AI_enhanced_ETL/examples/` |
| PDF Extraction & Cleaner | PDF text extraction, basic OCR notes, text cleaning, summary extraction | `BDA_3_AI_enhanced_ETL/extract_pdf.py`, `BDA_3_AI_enhanced_ETL/cleaner.py`, `data/processed/` |
| [ETL & Preprocessing Pipelines](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_3_AI_enhanced_ETL) | Tokenization, text normalization, Parquet export, Spark UDFs | `BDA_3_AI_enhanced_ETL/` and `BDA_8` examples |
| Feature Engineering | Numeric, categorical encoding, embeddings (TF-IDF, pretrained transformers) | `BDA_9_Reccomendation_System_Assigment/src/features/` |
| Spark MLlib | Regression, classification, pipelines, scaling, HyperParam tuning (basic) | `BDA_6_Topic5_Regression_Clustering/run_experiment1.py`, `make_report.py` |
| Deep Learning (TensorFlow / PyTorch) | Small model training, dataset API, mixed precision, saving checkpoints | `BDA_3`, `BDA_8/train_ddp.py` (examples), `BDA_6` experiment scripts |
| Recommendation Systems | User/item collaborative filtering, hybrid methods, diversity, popularity bias mitigation, evaluation | `BDA_9_Reccomendation_System_Assigment/src/`, `run/recommend_user.py`, `outputs/` |
| Clustering & Representation Learning | KMeans baseline, autoencoder/DeepCluster proxies, NMI evaluation | `BDA_6_Topic5_Regression_Clustering/outputs/exp2/` |
| Evaluation & Metrics | Precision@K, NDCG, MSE/RMSE, R^2, NMI | evaluation scripts (planned: `evaluate.py`) |
| Visualization & Reporting | Generating DOCX reports, architecture diagrams (Pillow), plots (matplotlib) | `make_report.py` per folder, `BDA_3/generate_architecture_diagram.py` |
| Distributed Training & Orchestration | PyTorch DDP, All-Reduce, TorchDistributor (Spark), Ray Train & Tune | `BDA_8` examples, `BDA_3` integration notes |

---

## 7. How to generate reports (quick)

Each project folder includes a `make_report.py` script which assembles textual answers, experiment outputs and images into a DOCX report. Basic steps:

1. Activate the repository `.venv` in the project root (you indicated you will use `.venv`).
2. Install Python dependencies for reporting:

```bash
pip install -r requirements.txt
# and explicitly ensure python-docx is available
pip install python-docx
```

3. Run the report generator in a project folder, for example:

```bash
cd BDA_8_Data_parallelism_and_Model_Parallelism
python3 make_report.py --out BDA_8_Report.docx
```

If the script reads external experiment outputs, ensure those output files exist; the generator will skip missing artifacts with an explanatory note.

---

## 8. Next steps & recommended improvements

- Implement `evaluate.py` to compute Precision@K and NDCG@K for the recommender and basic regression/clustering metrics for Topic5 experiments. Save results to `outputs/` as JSON.
- Add a lightweight `train_ddp.py` demo (already scaffolded in `BDA_8`) that can be run with `torchrun --nproc_per_node=2` on local GPUs.
- Expand DeepCluster to a proper PyTorch implementation and run on GPU for meaningful comparisons.
- Add unit/integration tests for reproducers and CI integration to validate report generation.

---

## 9. License & Credits

Course materials and code are provided for educational use. If you reuse or redistribute, please keep attribution to the original author(s) and the course.



