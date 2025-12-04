# Pseudocode: Core Agents

## IngestAgent
```
function IngestAgent.run(task):
  sources = task.sources
  for s in sources:
    if s.type == 'api':
      data = fetch_api(s.url, params=s.params)
    elif s.type == 'rss' or s.type == 'scrape':
      data = scrape(s.url)
    elif s.type == 'pdf':
      pdf = download(s.url)
      send_to(ParseAgent, pdf)
    store_raw(DataBroker, data)
  notify(Coordinator, 'ingest_complete', task_id=task.id)
```

## ParseAgent
```
function ParseAgent.run(payload):
  if payload.type == 'pdf':
    text = pdf_to_text(payload.bytes)
    entities = ner_extract(text)
    attachments = extract_tables(payload)
  elif payload.type == 'image':
    text = ocr(payload.bytes)
  store_processed(DataBroker, {text, entities, tables})
  notify(Coordinator, 'parse_complete', id=payload.id)
```

## RAGAgent
```
function RAGAgent.index(docs):
  embeddings = embed(docs)
  upsert_to_vector_db(embeddings, metadata=docs.meta)

function RAGAgent.retrieve(query, mode='hybrid'):
  if mode == 'table':
    return table_retriever(query)
  else:
    return hybrid_retriever(query)
```

## AnalysisAgent
```
function AnalysisAgent.run(query):
  docs = RAGAgent.retrieve(query)
  features = compute_technical_indicators(docs.timeseries)
  model_results = run_backtest(features)
  explanation = LLM.prompt(template, context=docs + model_results)
  return {results: model_results, explanation}
```

## ReportAgent
```
function ReportAgent.compile(context):
  tables = assemble_tables(context.stats)
  plots = generate_plots(context.timeseries)
  report = render_template('report.jinja', tables=tables, plots=plots, narratives=context.explanations)
  store_report(repo, report)
  return report
```
