Demo instructions

1. Start MCP stubs:

   ```bash
   python mcp_stubs/stubs.py
   ```

2. Run the ingestion demo (uses sample CSVs from other folders):

   ```bash
   python ingest_demo.py --csv ../BDA_10_LLM_Neo4j/financial_news_events.csv
   ```

3. Run the rag demo:

   ```bash
   python rag_demo.py --docs demo_docs/
   ```

Note: these demos are lightweight and meant for local testing; they expect dependencies listed in `requirements.txt` (Flask, pandas, langchain, faiss-cpu, python-dotenv, python-docx).