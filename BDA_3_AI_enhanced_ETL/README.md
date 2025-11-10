BDA_3_Ai_enhanced_ETL

Purpose
- Small scaffold for an AI-enhanced ETL experiment: crawl (PDF/HTML), extract text from PDFs, run simple cleaning/extraction, and generate an experiment report.

Quick start

1) Create and activate a Python virtualenv:

```bash
cd /Users/pro/Downloads/BigDataAnalysis/BDA_3_Ai_enhanced_ETL
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (adjust as needed):

```bash
pip install requests beautifulsoup4 PyPDF2 python-docx pandas
```

3) Run the crawler on a URL (downloads PDFs to `data/raw_pdfs`):

```bash
python3 crawler.py --input "https://example.com/some-page-with-pdfs"
```

Or point directly to a PDF:

```bash
python3 crawler.py --input "https://example.com/report.pdf"
```

4) Extract and clean PDFs:

```bash
python3 cleaner.py --input-dir data/raw_pdfs --output-dir data/processed
```

5) Generate the experiment report (DOCX):

```bash
python3 make_report.py --experiment-name "BDA3 ETL demo"
```

Notes
- This scaffold is intentionally lightweight. If you want to add LLM-based extraction, connect your preferred model in `cleaner.py` where noted.
- When running on a topic (not a URL) you should provide seed URLs manually or adapt the crawler to use a search API. The scaffold accepts either a URL or a direct PDF link.
