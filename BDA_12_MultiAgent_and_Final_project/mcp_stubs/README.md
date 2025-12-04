MCP stubs

This folder contains lightweight examples and interface specifications for two MCP servers:

1) Coordinator MCP Server
- Responsibilities: agent registry, task scheduling, global plan store, heartbeat monitoring.
- Endpoints:
  - `POST /register` — register agent {agent_id, capabilities, websocket_url}
  - `POST /task` — submit a task {task_id, spec}
  - `GET /tasks` — list tasks
  - `POST /notify` — send notification between agents

2) DataBroker MCP Server
- Responsibilities: store raw and processed data, serve cached artifacts, provide unified data access.
- Endpoints:
  - `POST /store` — store artifact (type, payload, metadata)
  - `GET /artifact/{id}` — fetch artifact
  - `GET /search` — search by metadata

Run
- These stubs are lightweight HTTP servers. Implementations are provided in `stubs.py`.
