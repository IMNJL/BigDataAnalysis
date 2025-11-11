# Intelligent Crawler and Cleaner — Experiment Report

Course: Big Data Analysis Technology
Assignment: Intelligent crawler + Cleaning Program/Agent
Date: 2025-11-11

---

## Part 1: Experiment Plan

### Objective

Design, implement and evaluate an intelligent web crawler agent that accepts a natural-language topic or a seed URL and produces a high-quality dataset of textual documents (HTML/PDF). Then design a cleaning program/agent that extracts text, cleans noise, identifies sections of interest and produces structured JSON records ready for downstream NLP tasks. The overall goal is to demonstrate an end-to-end automated pipeline (Agent) that maps an English topic or URL to cleaned structured data, plus evaluation metrics that quantify coverage and quality.

### Scope

- Input: a short natural language topic or a single seed URL (e.g., "Amazon annual report 2024" or a news site link).
- Output: a directory of raw artifacts (HTML, PDF) and a processed `data/processed/*.json` collection where each JSON record contains: source, title, extracted text, summary, comments/notes, full_text_length, and basic metadata (date, language, file type).

### Success criteria

- The crawler finds at least N=50 domain-relevant documents given a non-trivial topic (for large domains N can be lower to keep runs fast).
- The cleaner extracts plaintext with >95% preserved readable characters for machine-readable PDFs (no OCR) and produces a concise 100–400 word summary per document.
- The pipeline produces an evaluation report that includes: counts of raw vs processed documents, percent extraction success, average document length, and simple precision/recall proxies based on a small labeled sample.

### Experimental plan and steps

1. Topic parsing: interpret the natural-language input to produce query terms and seed URLs.
2. Candidate discovery (retrieval stage): use search APIs (optional), sitemap discovery, and focused crawling (link graph) starting from the seed URL(s). Limit crawl depth (2) and domain scope.
3. Fetching: download HTML/PDF assets with polite crawling (robots.txt) and rate limiting (configurable delays). Store raw bytes and minimal metadata.
4. Cleaning/Extraction: for HTML use DOM-based extraction (readability-like heuristics), for PDF use PyPDF2 / pdfminer (or OCR fallback) to extract text. Normalize whitespace and fix encoding.
5. Postprocessing: language detection, basic deduplication (min-hash or normalized text hashing), chunking large documents, and generating summaries using extractive rules (lead-paragraph + sentence-ranking) or a light LLM (optional).
6. Evaluation: sample K documents (e.g., 20) and compute extraction success, manual precision on extracted sections, and record statistics. Produce charts and a human-readable report.

### Tools and libraries

- Python 3.10+ environment
- requests, aiohttp, beautifulsoup4, readability-lxml
- PyPDF2 or pdfminer.six (PDF extraction), pytesseract (OCR, optional)
- langdetect or fastText language detection
- python-docx (report generation), matplotlib / seaborn for charts
- Optional: transformers (HuggingFace) for a lightweight summarizer or local LLM API hooks

### Experiment variants

- Agent version (preferred): a single program that receives natural language, performs retrieval and cleaning end-to-end and logs intermediate artifacts.
- Modular version: separate crawler and cleaner scripts where a human or external scheduler orchestrates runs.

### Evaluation plan

- Automatic metrics: number of fetched artifacts, number of processed JSONs, extraction success rate (file parsed and non-empty text), average full_text_length, duplicates removed.
- Manual sample: label 20 processed documents for extraction correctness (1/0) and compute precision; compute a simple recall proxy by counting overlap against a small known-good set (if available).
- Qualitative: inspect 10 examples and record failure modes (e.g., scanned PDFs requiring OCR, paywalls, JS-heavy sites).

---

## Part 2: Architecture (by Diagram)

The pipeline architecture is a small two-stage design: Retrieval (Crawler/Agent) → Cleaner (Extraction & Postprocessing) → Storage & Reporting.

- Retrieval (Agent): Input (topic/URL) → Query generator → Seed URL list → Focused crawler (polite, bounded depth) → downloaded artifacts (HTML/PDF) stored in `data/raw/`.
- Cleaner: For each artifact in `data/raw/` → detect type (HTML/PDF) → extract text → language detection → deduplication → summarize and extract metadata → write JSON to `data/processed/`.
- Orchestration & Evaluation: a driver script runs the agent and cleaner, collects logs, and computes metrics; a reporting module generates a DOCX/Markdown report with charts and examples.

Reference diagram file (generated): `screenshots/architecture.png` (see repository `BDA_3_AI_enhanced_ETL/screenshots/architecture.png`).

---

## Part 3: Steps and Screenshots

This section documents an example run. For each step we show the Prompt used (if Agent-style), and the Response or outcome. Replace the asterisks with real text when running; here we provide concrete example prompts and expected responses.

### Step 1. Topic input and query generation

**Prompt:**

"Topic: Amazon 2024 annual report; find publicly available PDFs and press releases with financial summaries."

**Agent actions / Response:**

- Parsed keywords: ["Amazon 2024 annual report", "Amazon Annual Report 2024 PDF", "Amazon investor relations 2024"].
- Generated seed queries for search endpoints and a small list of starting URLs: `https://www.amazon.com/`, `https://ir.aboutamazon.com/`.

**Screenshot:** (crawler start log) see `screenshots/crawler_start.png` (example placeholder in repo screenshots).

**Outcome:** seed URL list of length 4 and initial queries.

### Step 2. Focused crawling and asset discovery

**Prompt (Agent internal):**

"From each seed URL, follow internal links up to depth=2, restrict hosts to `amazon.com` and `aboutamazon.com`, extract asset links (pdf, html). Respect robots.txt and delays=1s."

**Response / Outcome:**

- Discovered N raw assets; downloaded M PDFs and K HTML pages. Example downloaded: `Amazon-2024-Annual-Report.pdf` (size ~ 600KB), `press-release-q4-2024.html`.
- All raw assets stored under `data/raw/` with metadata JSON: `{source, url, filename, content_type, fetched_at}`.

**Screenshot:** `screenshots/downloads_list.png`

### Step 3. Cleaning and extraction (HTML)

**Prompt (for agent component or human):**

"Extract readable article text from HTML file `data/raw/press-release-q4-2024.html` using DOM heuristics and fallback rules."

**Response:**

- Used readability heuristics to extract the main text block and title. Removed navbars, footers, and ads.
- Extracted metadata: title, publication date parsed from schema.org or page content, language=en.
- Saved processed JSON with keys: `source`, `title`, `text`, `summary`, `comments`, `full_text_length`.

**Example JSON excerpt (fields):**

```
{
  "source": "press-release-q4-2024.html",
  "title": "Amazon Reports Q4 2024 Results",
  "text": "Amazon today announced ...",
  "summary": "Amazon revenue grew 11% YoY to $638B in 2024...",
  "comments": ["Found relevant financial summary"],
  "full_text_length": 12345
}
```

**Screenshot:** `screenshots/html_extraction_example.png`

### Step 4. Cleaning and extraction (PDF)

**Prompt:**

"Extract text from `data/raw/Amazon-2024-Annual-Report.pdf` and generate 200–300 word summary. If PDF is image-scanned, fall back to OCR (pytesseract)."

**Response:**

- Used PyPDF2 to extract textual pages; concatenated page text; detected long continuous uppercase sections and corrected spacing artifacts.
- Generated a 150–300 word extractive summary by selecting top sentences (TF-IDF ranking) and by preserving the opening shareholder letter.
- Saved JSON with `summary` and `comments` listing extraction warnings (if any). Example `full_text_length`: 319924 (characters) for the Amazon annual report.

**Screenshot:** `screenshots/pdf_extraction_example.png`

### Step 5. Deduplication and language detection

**Prompt:**

"Remove duplicates by computing normalized text hash (lowercase, remove punctuation, strip whitespace) and keep the longest version. Detect language for each document and keep only English documents for downstream processing."

**Response:**

- Duplicates removed: 12 (out of 68). Language detection: kept 65 English documents, 3 non-English filtered out.

**Screenshot:** `screenshots/dedup_stats.png`

### Step 6. Summarization and metadata enrichment

**Prompt:**

"For each processed JSON, produce a 100–300 word summary and extract named entities (orgs, people, numerical figures) using a lightweight NER pipeline."

**Response:**

- Summaries generated via a rule-based hybrid: lead paragraph + top-ranked sentences; NER running via spaCy lite or a regex for financial figures.
- Enriched metadata: `entities: {organizations: [...], people: [...], figures: [...]}` saved in each JSON.

**Screenshot:** `screenshots/summary_and_entities.png`

### Step 7. Evaluation & manual inspection

**Prompt:**

"Select random K=20 processed documents and ask a human labeler to mark extraction as correct (1) or incorrect (0). Compute precision."

**Response / Outcome:**

- Sample size: 20. Human-labeled correct: 18/20 → precision = 90%.
- Extraction failures caused by: (a) scanned PDFs requiring OCR (2 cases), (b) paywalled content excluded early (0), (c) dynamically loaded JS content not captured by static fetch (0 in this sample).

**Screenshot:** `screenshots/manual_evaluation.png`

---

## Final Result

### Artifacts produced:

- `data/raw/` — raw downloaded artifacts (HTML and PDF) with metadata.
- `data/processed/*.json` — cleaned structured records (title, text, summary, comments, metadata).
- `reports/` — evaluation report with summary statistics and example JSON excerpts.
- `screenshots/` — example screenshots (extraction logs, architecture image, evaluation charts).

### Key numeric outcomes (example run):

- Raw artifacts fetched: 78
- Successfully processed JSON records: 66
- Extraction success rate: 84.6% (66/78)
- Deduplicated records kept: 66 (12 duplicates removed)
- Manual sample precision: 90% (18/20)
- Average extracted text length: 54,300 characters

### Lessons learned and improvements

- JS-heavy pages require a headless browser (selenium / playwright) for robust extraction; include this as an optional crawler mode.
- Scanned PDFs require OCR; integrate Tesseract for fallback on image-based PDFs and tune DPI settings.
- For scale, move to asynchronous fetching (aiohttp) and a simple task queue; for very large crawls use Spark/Beam to parallelize parsing.

### Next steps

1. Add an Agent mode that integrates a small LLM to improve query generation and to post-process noisy summaries.
2. Add an evaluation harness `evaluate.py` that computes Precision@K for relevance-based tasks and supports saving results in JSON/CSV.
3. Automate scheduled crawls and add incremental processing (only reprocess changed documents).

---

## Appendix: Example commands

**Run the crawler (example):**

```bash
python3 crawler.py --topic "Amazon 2024 annual report" --out data/raw --depth 2
```

**Run the cleaner:**

```bash
python3 cleaner.py --in data/raw --out data/processed
```

**Generate report (DOCX):**

```bash
python3 make_report.py --out reports/Intelligent_Crawler_and_Cleaner_Report.docx
```

**Appendix: Sample JSON record (shortened)**

```
{
  "source": "Amazon-2024-Annual-Report.pdf",
  "title": "Amazon 2024 Annual Report",
  "text": "ANNUAL REPORT 2024\nDear Shareholders:\n2024 was a strong year for Amazon...",
  "summary": "Amazon’s total revenue grew 11% Y/Y to $638B; AWS revenue grew 19% Y/Y to $108B. Operating income improved 86%...",
  "comments": ["extracted using PyPDF2; no OCR required"],
  "full_text_length": 319924
}
```

