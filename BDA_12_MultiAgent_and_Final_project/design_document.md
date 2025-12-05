# Design Document — Multi-Agent Intelligent Financial Analysis

## 1. Project Summary
Design and implement a multi-agent intelligent financial analysis system. The system ingests multi-source, multi-modal financial data, runs LangChain RAG and algorithmic analyses, and outputs investment recommendation reports. Agents communicate via MCP and are orchestrated in CrewAI-style flows.

## 2. Requirements mapping
- Data ingestion: streaming and batch connectors for market prices, news, and financial reports (PDFs).
- Multi-modal: tabular price time-series, textual news and filings (PDF), and images (charts, screenshots).
- Multi-agent: modular agents for ingestion, parsing, analysis, RAG retrieval, report generation and monitoring.
- MCP protocol: design two MCP servers (Coordinator and DataBroker).

## 3. System Architecture (high level)
- Agents:
  - IngestAgent: connects to APIs (YahooFinance/AlphaVantage), scrapes news, downloads PDFs and images.
  - ParseAgent: extracts text from PDFs, OCR images, and converts raw tables to normalized CSV.
  - RAGAgent: indexes documents and tables into vector DB (FAISS or Weaviate), provides retrieval.
  - AnalysisAgent: runs algorithmic models (technical indicators, backtests) and LLM prompts for interpretation.
  - ReportAgent: composes the final report (tables, visualizations), ensures output schema.
  - MonitorAgent: monitors market indicators and triggers alerts.
- MCP Servers:
  - Coordinator MCP: task routing, global plan storage, and agent registry.
  - DataBroker MCP: provides unified data access to agents and caches frequently used artifacts.

## 4. Multi-agent design pattern
- Adopt a hybrid Group-Chat + Hand-off pattern: agents can propose ideas in a shared channel (group) and then hand off concrete tasks to specialized agents for execution.
- Reasoning: collaborative brainstorming (LLM-driven planning) benefits from multiple perspectives, while hand-off ensures single-agent responsibility for execution.

## 5. LangChain component design
- Chains to implement:
  1. IngestionChain: orchestrates data pulls, PDF downloads, and passes to ParseAgent.
  2. RAGChain: document embedding, vector index update, canonical retrieval wrapper.
  3. AnalysisChain: combines algorithmic analysis + LLM-assisted explanations and output parsing.
- RAG architecture: use FAISS (disk-backed) for dense vectors + a metadata store (SQLite) to support hybrid search.
- Prompt Templates: define structured prompts with placeholders for question type, retrieved docs, and required output format.
- Output Parsers: use strict JSON schema parsing to prevent hallucinations and ensure downstream consistency.

## 6. CrewAI process control
- Task execution flow: hierarchical (plan → subtask groups → execution), tasks may be parallel where independent.
- Async attributes: define each task's `can_run_parallel` flag; Coordinator schedules parallelizable tasks on worker pool.
- Monitoring: heartbeat checks and result validators; failed tasks are retried based on policy (max 3 retries).

## 7. MCP Integration
- Two MCP servers (Coordinator, DataBroker) with REST and WebSocket endpoints for agents to register and exchange messages.
- Data transmission: JSON-LD style messages with explicit `task_id`, `agent_from`, `agent_to`, `payload_type`, and `payload`.
- Advantages: decouples agents, supports pub/sub, and permits centralized policy enforcement.

## 8. Agentic RAG System
- Knowledge graph: build using extracted entities (companies, tickers, dates), relations (reported revenue, events), stored in Neo4j or as JSON-LD triples.
- Dynamic retrieval: for numerical queries use table-aware retriever; for textual queries use dense retriever + hybrid BM25.
- Hallucination prevention: include provenance in replies (source links), enforce output schema, and cross-validate numeric claims against time-series DB.

## 9. Memory management
- Short-term: in-memory per-session store (Redis or in-process dict) for current tasks and context.
- Long-term: vector DB + document store for persistent facts, with periodic compression via LLM summarization.
- Cross-session context: user profiles and strategy preferences saved in long-term memory and rehydrated per session.

## 10. Performance & evaluation
- Parallelization: use worker pools for independent data pulls and indexing.
- Balancing resources: tier tasks as cheap/expensive; schedule expensive analyses off-peak.
- Fault tolerance: supervisor process restarts failed agents; DataBroker maintains durable queue (Kafka or Redis streams).
- Evaluation metrics: precision/recall for retrieved docs, RMSE for numeric predictions, human evaluation score for report quality.

## 11. Self-evolution
- Auto-tuning: track which prompt templates produce better downstream evaluations and promote them.
- Role adaptation: scale worker agents when task queue backlog increases.
- Transfer learning: reuse embeddings or vector indexes between domains with domain-specific prompt adapters.

## 12. Deliverables in this folder
- `design_document.md` (this file)
- `diagrams/system_architecture.png` (generated)
- `pseudocode/` (pseudocode for key modules)
- `demo/` (runnable minimal demos)
- `presentation/` (pptx template)

*** Next steps: pick which module you'd like me to implement first (MCP servers, RAG agent, or Analysis agent demo).