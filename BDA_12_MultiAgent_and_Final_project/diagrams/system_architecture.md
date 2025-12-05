# System Architecture — Detailed Explanation

This document explains the `system_architecture.puml` PlantUML diagram in detail: its components, their responsibilities, data flows, failure/fallback behaviors, security considerations, and possible extension points.

---

**Diagram at a glance**
- The system is a lightweight multi-agent prototype built around two MCP (Minimal Coordination Protocol) services:
  - **Coordinator** — registers agents and accepts tasks.
  - **DataBroker** — stores and serves artifacts (data files), indexes metadata, and provides search/download endpoints.
- Agents (ingest, RAG, analysis/reporting) interact with the MCP services to coordinate work, share artifacts and trigger pipelines.
- A minimal **UI** is provided to trigger demos, view status, and launch RAG runs.
- Local storage (`/tmp`), `outputs/` and optional external services (LLM provider, vector DB, object storage) are shown as integration points.

---

## Components (what they are and what they do)

- **Coordinator (MCP Coordinator)**
  - Responsibilities: agent registration, task submission, simple orchestration and notifications.
  - Endpoints (demo stub):
    - `POST /register` — register agent metadata (capabilities, id)
    - `POST /task` — submit a task spec
    - `GET /tasks` — list tasks
    - `POST /notify` — simple notification hook
    - `GET /dashboard` — HTML status page
  - Implementation notes: small Flask app (`mcp_stubs/stubs.py`), keeps `AGENTS` and `TASKS` in memory for demo convenience.

- **DataBroker (MCP DataBroker)**
  - Responsibilities: ingest artifacts, store files, maintain metadata, serve downloads and simple search.
  - Endpoints (demo stub):
    - `POST /store` — multipart upload (file + metadata)
    - `GET /artifact/<id>` — retrieve artifact metadata and path
    - `GET /artifact/<id>/download` — download artifact contents (binary stream)
    - `GET /search?q=...` — search artifacts by metadata
    - `GET /dashboard` — HTML listing of artifacts and download links
  - Storage model: files are saved to `/tmp/<artifact_id>`; metadata stored in an in-memory `ARTIFACTS` dict. (Design note: persist metadata in SQLite or an object store for production.)
  - Robustness additions in the demo:
    - Fallback: if `ARTIFACTS` lacks an id (server restart), the `/artifact/<id>` endpoint checks `/tmp/<id>` and returns minimal metadata so downloads remain possible.
    - Download serving is tolerant to Flask versions: `send_file(download_name)`, `send_file(attachment_filename)`, and a manual streaming fallback using `Response(stream_with_context(...))`.

- **Agents**
  - **Ingest Agent** (`demo/ingest_demo.py`): uploads datasets to the DataBroker and registers itself with the Coordinator.
  - **RAG Agent** (`demo/rag_demo_gemini.py`): downloads artifacts, builds embeddings, creates an index (FAISS or NumPy fallback), retrieves top-k items for a query, and summarizes results (Vertex AI/Gemini optional or a template fallback).
  - **Auto / Manual agents** (in BDA_11): run analytic pipelines, produce `stats.json`, plots, reports; used to compare automated vs manual processes.

- **UI** (`ui/app.py`)
  - Minimal Flask site to view status and trigger demos. It may launch background processes (demo scripts) and presents basic dashboards.

- **Local & External Storage**
  - `outputs/` — experiment outputs (plots, reports, triplets)
  - `/tmp/<artifact_id>` — temporary artifact file storage used by the DataBroker in the demo
  - External systems (recommended for production): S3/GCS for artifact storage, SQLite/Postgres for metadata, and a persistent vector DB for large-scale retrieval.

- **LLM & Embedding Services**
  - Optional cloud LLM (Vertex AI / Gemini) for summarization.
  - Embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) when available, with a fallback lightweight TF vectorization implemented for demo portability.
  - FAISS as an optional index with a NumPy brute-force fallback for environments without FAISS.

---

## Core data flows and workflows (step-by-step)

1) **Ingest flow** (how a dataset becomes an artifact):
   - Client (or `ingest_demo.py`) POSTs a file and metadata to `POST /store` on DataBroker.
   - DataBroker saves file to `/tmp/<uuid>` and records `ARTIFACTS[uuid] = {'meta': meta, 'path': '/tmp/<uuid>'}`.
   - DataBroker returns `{'artifact_id': uuid}` to the client.
   - Optionally, the agent registers with Coordinator (`/register`) and posts a `task` to `/task` to initiate downstream work.

2) **Download flow** (how agents retrieve artifacts):
   - Agent calls `GET /artifact/<id>/download`.
   - DataBroker attempts to locate an in-memory `ARTIFACTS` record. If found, it uses the stored path; if not, it checks `/tmp/<id>` and constructs fallback metadata for the response.
   - DataBroker streams the file back, using `send_file` or the manual streaming fallback if necessary.

3) **RAG demo flow** (end-to-end retrieval & summarization):
   - RAG Agent downloads the CSV artifact.
   - Reads rows into document strings (Date | Headline | Source | Related_Company).
   - Builds embeddings:
     - Preferred: `sentence-transformers` => dense vectors.
     - Fallback: lightweight TF-count vectors with vocabulary capped and L2 normalization.
   - Indexing:
     - Preferred: FAISS `IndexFlatL2`.
     - Fallback: keep embeddings in a NumPy array and perform brute-force L2 search.
   - Query embedding computed using same embedding pipeline.
   - Top-k retrieval performed, and retrieved rows are passed to the summarizer.
   - Summarizer: Vertex AI (Gemini) if configured and available, else `simple_template_summary()`.

4) **Experiment orchestration** (Auto vs Manual)
   - `experiment.py` calls `auto_agent.py` and `manual_agent.py` sequentially or in parallel.
   - Each agent executes its pipeline, writing outputs to `outputs/`.
   - `compare_reports.py` compares metrics and produces `comparison_summary.md`.
   - `generate_docx.py` collects diagrams and outputs into a DOCX report.

---

## Endpoints / Message formats (for reference)
- `POST /register` — JSON: `{"agent_id": "<id>", "capabilities": [...], "meta": {...}}`
- `POST /task` — JSON: `{"task_id": "<id>", "spec": {...}}`
- `POST /store` — multipart/form-data: file under `file`, form fields for metadata. Response `{"artifact_id": "<uuid>"}`
- `GET /artifact/<id>` — returns `{"meta": {...}, "path": "/tmp/<id>"}` or 404
- `GET /artifact/<id>/download` — binary response with `Content-Disposition` header

---

## Failure modes, fallbacks, and resilience strategies (demo vs production)

- **In-memory metadata loss (demo)**
  - Cause: server restart clears `ARTIFACTS` dict.
  - Demo mitigation: `GET /artifact/<id>` checks `/tmp/<id>` and returns minimal metadata so downloads still work.
  - Production fix: persist metadata in a database and store files in object storage (S3/GCS) with durable keys.

- **send_file compatibility issues**
  - Cause: differences across Flask versions (`download_name` vs `attachment_filename`) or environment-specific errors when streaming files.
  - Mitigation: code attempts `send_file(download_name=...)`, falls back to `send_file(attachment_filename=...)`, and finally uses a manual streaming `Response` generator.

- **Heavy dependency / TF/Keras import failures**
  - Problem: `sentence-transformers` may import TensorFlow/Keras and fail with Keras 3 incompatibility in some environments.
  - Mitigation: the RAG demo tries to import `sentence-transformers` and, on failure, uses a lightweight TF-count embedding fallback (tokenize → build limited vocabulary → L2-normalize TF vectors).

- **FAISS missing**
  - Mitigation: use brute-force NumPy search over embedding matrix for small datasets.

- **LLM call failures**
  - Mitigation: wrap LLM calls with try/except and fallback to a deterministic template summarizer.

---

## Deployment & operational notes
- Demo runs two Flask processes (Coordinator: `:5005`, DataBroker: `:5006`). Use `python mcp_stubs/stubs.py` to start both in threads (no reloader).
- For production:
  - Use separate process containers behind reverse proxies (with TLS).
  - Persist metadata in a DB and files in object storage.
  - Replace in-memory indices with a managed vector DB (Weaviate, Milvus, Pinecone) or persistent FAISS service.

---

## Security considerations
- Add authentication/authorization for ingest/download endpoints (API keys or OAuth).
- Ensure artifact uploads are scanned and sanitized; enforce size limits and content-type checks.
- Use signed URLs for direct downloads from object storage in production to avoid proxying large files through the DataBroker.

---

## Extension ideas (next steps)
- Persist `ARTIFACTS` to SQLite and provide migration to S3/GCS.
- Add an async task runner (RQ/Celery) to execute heavy jobs and avoid blocking the stub servers.
- Add a small metrics/telemetry endpoint for timing and memory usage.
- Surface live logs and run results in the `ui/` instead of launching background processes invisibly.

---

## Where to look in the repo
- PlantUML source: `diagrams/system_architecture.puml`
- Demo and stubs: `mcp_stubs/stubs.py`, `demo/ingest_demo.py`, `demo/rag_demo_gemini.py`
- UI: `ui/app.py`
- Presentation helpers: `tools/generate_puml_pptx.py`

---

If you'd like, I can:
- Embed this MD into the presentation as a speaker-notes slide, or
- Produce a cleaned PNG of the PlantUML and place it beside this MD in `presentation/` (the repo already contains a script to render PlantUML).

Which of those should I do next?