
Task 1. Intelligent crawler.

1. "Input" (Natural Language)  Topic or Link URL.
   - Example inputs:
     - "collect financial news about acquisitions and earnings reports for Q1 2025"
     - A link to a news site RSS feed: https://www.reuters.com/rssFeed

2. Generate crawler. Program or Agent (preferred, better score)
   - Implementation approach (program):
     - Use an intelligent crawler (Python agent) that accepts either a topic or a list of seed URLs / RSS feeds.
     - Key components:
       - Feed fetcher (RSS): use `feedparser` to pull feeds regularly.
       - Page fetcher: `requests` + `newspaper3k` or `readability-lxml` for content extraction.
       - Deduplication: compute fingerprint (hash of normalized title+text) and skip duplicates.
       - Scheduler: simple cron or `APScheduler` for periodic crawling.
     - Agent enhancements:
       - Accept natural language topic and expand into seed queries using query templates (e.g., "site:reuters.com acquisitions 2025").
       - Rate limiting, robots.txt respect and polite crawling delays.

3. Evaluate Data.
   - Evaluation checklist:
     - Coverage: percent of target sources successfully fetched.
     - Quality: percent of records with non-empty title and body.
     - Duplicates: duplicate rate (should be low after dedupe).
     - Timeliness: time lag between publication and crawl (for live feeds).

Task 2. Cleaning Program/Agent.

- Purpose: remove HTML noise, normalize whitespace, detect and fix encoding issues, standardize company names and dates, and remove non-news items.

Cleaning pipeline steps (program/agent):
1. Remove HTML and scripts using `BeautifulSoup` or `readability`.
2. Normalize whitespace and punctuation, fix unicode normalization (NFKC).
3. Remove boilerplate using heuristics (e.g., navigation links, related-articles sections).
4. Normalize dates into ISO format and map company names to canonical forms (strip suffixes like "Inc.", "Ltd.").
5. Basic language detection: if article is not in the expected language, mark or route to translation.

Experiment Report Format

Part 1: Experiment Plan

- Objective: Build a lightweight FinBERT-based sentiment classifier pipeline that ingests financial news, preprocesses it, runs FinBERT inference, cleans outputs and produces a small KG-ready output and visualizations. Evaluate how FinBERT classifies headlines on the attached dataset.

- Dataset: `/Users/pro/Downloads/archive-2/financial_news_events.json` (JSONL) containing fields: Date, Headline, Source, Market_Event, Market_Index, Index_Change_Percent, Trading_Volume, Sentiment (if present), Sector, Impact_Level, Related_Company, News_Url.

- Tools & Libraries:
  - Python 3.8+
  - Hugging Face `transformers` (FinBERT model: `yiyanghkust/finbert-tone`)
  - `torch`, `pandas`, `matplotlib`, `seaborn`

- Success criteria:
  - Run inference on 200 records and produce predictions saved to `BDA_7_FinBert/outputs/finbert_predictions.jsonl`.
  - Produce a class distribution plot `finbert_class_distribution.png`.

FinBERT — model overview and prompt example

- What is FinBERT?
  - FinBERT is a domain-adapted BERT model specialized for financial language. It is created by continuing pretraining of a BERT-family model on large financial corpora (news, filings, analyst reports) and then fine-tuning on finance-labeled tasks such as sentiment classification (Financial Phrasebank, Reuters headlines, proprietary datasets).
  - Typical advantages: better tokenization for finance terms, improved recognition of finance-specific collocations ("beat estimates", "downgrade", "acquisition"), and more calibrated predictions in finance tasks.

- How we use FinBERT in this experiment
  - We use a pre-trained FinBERT classification model (`yiyanghkust/finbert-tone`) from Hugging Face. The model expects a single input text (headline or combined headline+context) and returns logits for sentiment classes (commonly: 0=Neutral, 1=Positive, 2=Negative depending on model config — check `model.config.id2label`).

- Example inference input (how we format the text):

```text
Tech Giant's New Product Launch Sparks Sector-Wide Gains - IPO Launch - JP Morgan Chase - Barron's - Date: 2025-03-02
```

- Example (pseudo-)prompt / wrapper (Markdown) used around the inference call

```markdown
Run FinBERT sentiment classifier on the following headline (single string). Return label and probabilities.

Headline: "Major technology firm acquires a smaller rival in a multi-billion dollar deal - IPO Launch - Samsung Electronics - Barron's - Date: 2025-04-02"

Response format (JSON): {"pred": <int>, "pred_label": "<label>", "probs": [p_neg, p_neu, p_pos]}
```

- Example prompt for LLM-based entity/relationship extraction (if you use an LLM in the pipeline)

```markdown
Instruction: Extract factual triplets from the sentence. Output only valid JSON array of objects. Each object must have: subject, predicate, object, subject_type, object_type, confidence (0-1), span (the exact text span).

Allowed types: Company, Person, Index, Event, Attribute, Other

Example:
Input: "BigCo Ltd has completed the acquisition of SmallCo."
Output: [{"subject":"BigCo Ltd","predicate":"acquired","object":"SmallCo","subject_type":"Company","object_type":"Company","confidence":0.95,"span":"BigCo Ltd has completed the acquisition of SmallCo"}]

Now extract from: "{INSERT_SEGMENT}" 
```


Part 2: Architecture (by Diagram)

- See `architecture_diagram.dot` (simple Graphviz DOT) for components: DataSources -> Crawler -> Preprocessing -> FinBERT -> Cleaning -> Storage -> Visualization.

Part 3: Steps and Screenshots.

Step 1. Crawl / Prepare dataset

Prompt: "Collect RSS feed items from Reuters and Financial Times for 'acquisition' and 'earnings' keywords for the last 6 months. Save as JSONL with fields: Date, Headline, Source, Market_Event, Related_Company, News_Url."

Response: The crawler produced a JSONL file with N records. Example JSONL line:
```
{"Date":"2025-05-21","Headline":"Nikkei 225 index benefits from a weaker yen","Source":"Times of India","Market_Event":"Commodity Price Shock", ...}
```

![](outputs/image.png)

Step 2. Run cleaning program

Prompt: "Run cleaning agent: normalize dates, remove empty headlines, strip html, canonicalize company names."

Response: The cleaner produced `cleaned_financial_news.jsonl` with normalized `Date` in ISO format and `Headline` cleaned.

(Attach screenshot: head of cleaned file)

Step 3. Run FinBERT inference

Prompt: "Run FinBERT inference on cleaned headlines and write predictions to outputs/finbert_predictions.jsonl".

Response: The FinBERT pipeline loaded model `yiyanghkust/finbert-tone`, ran inference and produced predictions with labels and probabilities. Example output:
```
{"text":"Acme Inc. announced a new electric vehicle today.", "pred":2, "pred_label":"positive", "probs":[0.05,0.1,0.85]}
```

(Attach screenshot: sample of outputs/finbert_predictions.jsonl)

Step 4. Evaluate and visualize

Prompt: "Plot a bar chart with class counts and save as finbert_class_distribution.png"

Response: Chart saved. (Attach screenshot of the plot)

Final Result.

- Files produced:
  - `BDA_7_FinBert/finbert_pipeline.py` (pipeline)
  - `BDA_7_FinBert/experiment.py` (experiment runner)
  - `BDA_7_FinBert/outputs/finbert_predictions.jsonl` (predictions)
  - `BDA_7_FinBert/outputs/finbert_class_distribution.png` (plot)

Experiment results (run on 200 records)

- Predicted class counts (FinBERT `yiyanghkust/finbert-tone`):
  - Positive: 81
  - Neutral: 69
  - Negative: 50
  - Total processed: 200

- Prediction confidence (max softmax per sample):
  - Mean max-probability: 0.9597
  - Median max-probability: 0.9995
  - Std (population): 0.0987

- Short interpretation:
  - FinBERT produced a balanced but slightly positive-leaning distribution on this dataset (41% Positive, 34.5% Neutral, 25% Negative). The high median max-probability (0.9995) indicates the model is highly confident for many examples (often near-deterministic logits); mean and std show most predictions are confident but a minority have lower confidence (examples with mixed signals).



