# Project Implementation Notes

This file contains concrete next steps and commands to run the demo locally.

Quickstart

1. Create and activate a venv: 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the MCP stubs (runs two local servers):

```bash
python mcp_stubs/stubs.py
```

3. In a new terminal, run the ingest demo (adjust CSV path as needed):

```bash
python demo/ingest_demo.py --csv ../BDA_10_LLM_Neo4j/financial_news_events.csv
```

4. Use the rest of the demo folder to test RAG and simple retrieval.

Notes
- The code is scaffolded for demonstration and documentation. Replace stub endpoints and integrate production components as you implement.
