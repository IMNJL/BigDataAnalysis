BDA_8 — Data Parallelism and Model Parallelism
=================================================

Этот каталог содержит:
- `make_report.py` — генератор DOCX-отчёта по Part 1–4 (включает большой раздел «Эксперименты и результаты»).
- `train_ddp.py` — минимальный пример PyTorch DDP (демо на синтетических данных, BERT tokenizer).
- `spark_torch_distributor_example.py` — шаблон/скелет для интеграции TorchDistributor (Spark + PyTorch).
- `requirements.txt` — базовые зависимости для генерации отчёта (python-docx).

Как использовать (предполагается, что вы работаете внутри `.venv` в корне проекта):

1) Активируйте виртуальное окружение (если ещё не активировано):

```bash
# Mac zsh
source .venv/bin/activate
```

2) Установите зависимости для отчёта (уже выполнено, но перечислено для справки):

```bash
pip install -r requirements.txt
```

3) (Опционально) Для запуска `train_ddp.py` установите PyTorch и Transformers в окружение:

```bash
# Пример (подберите правильный индекс URL для вашей CUDA версии или CPU вариант):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets
```

4) Запустите генерацию отчёта (создаст `BDA_8_Report.docx`):

```bash
python3 make_report.py --out BDA_8_Report.docx
```

5) Запуск DDP локально (если есть >1 GPU):

```bash
# пример: 2 процесса
torchrun --nproc_per_node=2 train_ddp.py --model_name_or_path bert-base-uncased --epochs 1
```

Примечание: `spark_torch_distributor_example.py` — шаблон, требует настройки под конкретный Spark + TorchDistributor стек и не рассчитан на немедленный запуск.
BDA_8_Data_parallelism_and_Model_Parallelism

Purpose

This project contains a report generator that produces a DOCX file answering the assignment on Data Parallelism and Model Parallelism (Parts 1–4). The report is written in Russian and contains detailed explanations, diagrams references and design answers.

Quick start

1) Create and activate a Python virtualenv:

```bash
cd /Users/pro/Downloads/BigDataAnalysis/BDA_8_Data_parallelism_and_Model_Parallelism
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Generate the DOCX report:

```bash
python3 make_report.py --out BDA_8_Report.docx
```

Output: `BDA_8_Report.docx` in the project folder.

Notes
- The report content is self-contained and intended for coursework submission. If you want diagrams embedded or different language, I can update the generator.
