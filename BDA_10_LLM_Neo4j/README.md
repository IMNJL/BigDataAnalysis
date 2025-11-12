LLM + Neo4j Knowledge Graph Pipeline

This folder contains a scaffolded pipeline to build a knowledge graph from financial news using an LLM for entity and relationship extraction, and Neo4j for storage and visualization.

Files:
- pipeline.py - main pipeline implementation (preprocessing, extraction, cleaning, conversion, Neo4j import, NL-to-Cypher translator)
- experiment.py - example run using mocked data and a mock LLM extractor if you don't have an API key
- requirements.txt - Python dependencies

Quick start
1. Create a Python virtual environment (recommended).
2. pip install -r requirements.txt
3. Configure environment variables (optional):
   - NEO4J_URI (e.g., bolt://localhost:7687)
   - NEO4J_USER
   - NEO4J_PASSWORD
   - LLM_API_KEY (if you want to use a real LLM API)
4. Run the experiment: python experiment.py

Notes
- The pipeline includes a mocked LLM extractor that returns example triplets. Replace with your preferred LLM call in `pipeline.py`.
- The script demonstrates how to convert triplets to Cypher MERGE statements and how to push to Neo4j using py2neo.
- Always test on a small dataset before scaling to larger corpora.