# Final Project — Multi-Agent Intelligent Financial Analysis

Project skeleton and deliverables for the course final project "Multi-Agent Intelligent System".

Deadline: December 5, 2025

Overview
- This folder contains the project scaffold: design documents, diagrams, pseudocode, demo scripts and templates you can build on.
- The system is a multi-agent financial analysis platform using MCP for agent communication, LangChain for chains and RAG, and CrewAI-like orchestration for process control.

How to use

1. Review `design_document.md` for the architecture and module descriptions.
2. Inspect `mcp_stubs/` for example MCP server stubs and how to run them locally.
3. Use `demo/` to run minimal data ingestion and RAG retrieval examples (these use datasets from other folders in the workspace).
4. Fill the `presentation/` folder with slides and `deliverables/` with final documents.

Structure

- `design_document.md` — detailed design and pseudocode for all required sections.
- `diagrams/` — system architecture and dataflow diagrams (PNG files).
- `mcp_stubs/` — example MCP server implementations (stubs) and instructions.
- `langchain_chains/` — chain templates and RAG connector examples.
- `memory/` — memory module design and helper scripts.
- `demo/` — runnable minimal demos showing ingestion, retrieval, and a simple agent loop.
- `presentation/` — PPTX template and assets for the final presentation.
- `deliverables/` — where final report, pseudocode, and recorded video notes should go.
