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

## 6. Practical topics checklist (beginning from Crawler and Cleaner)

The table below is a compact checklist of project/homework items (dates shown as in the course board screenshot). Use this to track which experiments are handed in or pending.

| DD| HW | Project / Exercise | Status | Note |
|---:|---:|---|:---:|---|
| 25.09 | HW1 | [Topic 1 - Crawler & cleaner + report -> lexie](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_1_crawler_and_cleaner) | 🟢 | handed |
| 16.10 | HW2 | Multi modal Homework | 🟢 | handed |
| 27.10 | HW3 | [Topic 3 - Use AI enhanced ETL to Extract Information](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_3_AI_enhanced_ETL) | 🟢 | handed |
| 01.11 | HW4 | Topic 4 — Snorkel, autoLabeling | 🔴 | !!! |
| 08.11 | HW5 | [Topic 9 - Recommendation System Assignment — Comparative study & LLM-enhanced recsys](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_3_AI_enhanced_ETL) | 🟢 | handed |
| 11.11 | HW6 | [Linear Regression, Lasso, Polynomial, Neural Network Regression](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_6_Topic5_Regression_Clustering) | 🟢 | handed |
| 11.11 | HW8 | ["Data Parallelism" and "Model Parallelism" — Spark MLlib, PyTorch](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_8_Data_parallelism_and_Model_Parallelism DDP) | 🟢 | handed |
| 12.11 | HW5 | [AutoML and Featuretools](https://github.com/IMNJL/BigDataAnalysis/tree/main/BDA_5_AutoML_and_Featuretools) | 🟢 | handed |
| 13.11 | HW10 | [HW-BDA-10-KG ]() | 🔴 | pending |
| 16.11 | HW11 | [HW-BDA-11-Agent]() | 🔴 | pending |
| 20.11 | HW7 | FinBert | 🔴 | pending |

*Note: Some HW numbers repeat in the course board (e.g., HW5 appears for different topics); use the date column to disambiguate.

If you prefer different emoji or want a column for links to the report scripts / outputs, tell me and I will add a "Files / Link" column with direct paths to the relevant folders/files.

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



