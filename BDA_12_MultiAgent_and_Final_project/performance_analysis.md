# Performance Analysis — Expected Metrics and Optimization Strategies

This document lists key performance metrics for the Multi-Agent RAG prototype, baseline expectations for small/medium/large datasets, and recommended optimization strategies to improve throughput, latency and resource usage.

## 1. Key Metrics to Monitor
- **Artifact download latency**: time to stream artifact from DataBroker to agent (ms–s). Important for large files.
- **Parsing latency**: time to parse/convert raw artifact to structured text (s–min depending on PDF/OCR complexity).
- **Embedding throughput**: documents/sec (or tokens/sec) when producing dense vectors (CPU vs GPU).
- **Index build time**: time to add N vectors to index (seconds → minutes); affected by index type (Flat vs IVF/HNSW) and hardware.
- **Query latency**: time to encode query + retrieve top-k (ms–s). Target: <200 ms for snappy UX (approx. for small corpora) or <1s for larger local indices).
- **Memory usage**: RAM required for embeddings and index (GB). Dense embeddings: e.g., 768-d float32 ≈ 3 KB per vector (unquantized).
- **Disk usage**: storage for raw artifacts + vector index (GB).
- **Throughput**: concurrent queries/sec supported by the system under target latency SLA.
- **LLM call latency & cost**: time per prompt and tokens consumed (ms–s and $/request). Often the largest variable-cost item.

## 2. Baseline Expectations (small / medium / large)

The table below summarizes expected performance characteristics and resource implications across three deployment scales. Use these as starting estimates — your results will vary by model, hardware, and dataset characteristics.

| Metric / Scale | Small (<= 10k docs) | Medium (10k–1M docs) | Large (>1M docs) |
|---|---:|---:|---:|
| Embedding model (recommended) | all-MiniLM (CPU/GPU) | all-MiniLM or larger (GPU) | GPU-accelerated models / cloud embeddings |
| Embedding throughput (docs/sec) | 100–1,000 (CPU), 1k+ (GPU) | 1k–10k (GPU, batched) | 10k+ with GPU cluster / cloud batching |
| Index type | FAISS IndexFlatL2 or NumPy | ANN (HNSW / IVF+PQ) | Sharded ANN or managed vector DB |
| Index build time | seconds → tens of seconds | minutes → hours (depends on N) | hours → days (use incremental + sharding) |
| Query latency (encode+retrieve) | ~10–200 ms | ~10–200 ms (with ANN) | ~50–200 ms (with tuned ANN + caching) |
| Memory (approx.) | ~MBs → low GB (10k vectors) | tens of GB (100k–1M uncompressed) | tens → hundreds GB (use quantization/managed DB) |
| Disk/storage | small; local disk ok | SSD recommended; consider object storage | Object storage + persistent index storage (SSD/NVMe) |
| Cost drivers | CPU time, local disk | GPUs for embeddings, index hosting | GPUs, storage, LLM token costs, managed services |
| Fallback strategy | NumPy brute-force, TF-count fallback | FAISS + quantization, partial sharding | Managed vector DB, heavy quantization, caching |

Notes:
- For small corpora the NumPy brute-force approach is acceptable and simpler to operate.
- For medium and large corpora use ANN indexes (HNSW, IVF+PQ) and consider vector quantization to reduce memory and CPU footprint.
- LLM summarization cost can dominate per-query expense; use cheaper models for drafts and reserve large models for final synthesis.

## 3. Bottlenecks & Where They Arise
- Embedding generation is compute-heavy (CPU-bound without GPU).
- Index building and in-memory indices require RAM; naive in-memory indices won't scale beyond a few million vectors.
- LLM summarization latency and cost when using cloud LLMs.
- Disk I/O when streaming many large artifacts through the DataBroker.
- Python GIL limits CPU-bound parallelism for pure-Python loops (use multiprocessing or native libs).

## 4. Optimization Strategies (practical recommendations)
### A. Embedding & Model Inference
- Use GPU inference for large-scale embedding generation (NVIDIA GPUs, or cloud GPUs).
- Use smaller, faster models for first-pass retrieval (e.g., all-MiniLM), and larger cross-encoders only for re-ranking top candidates.
- Batch texts for inference to maximize throughput and reduce invocation overhead.
- Use mixed precision and model quantization where supported by runtime.

### B. Indexing & Retrieval
- Use ANN indexes (HNSW, IVF+PQ) for large corpora; tune `efConstruction`, `M`, `nlist`, and PQ size for speed/recall trade-offs.
- Persist indices to disk and memory-map (faiss supports mmap) to speed restarts.
- Shard index by dataset/namespace and route queries to relevant shard(s).
- Use incremental upserts when adding documents rather than rebuilding full index.
- Use vector quantization (e.g., 8-bit/4-bit or PQ) to reduce memory and cache footprint.

### C. Caching & Tiering
- Cache recent query results and expensive LLM outputs.
- Keep a hot index in memory for the most relevant subset and cold storage for archive data.
- Use a CDN or signed URLs for artifact download to offload DataBroker for large file transfers.

### D. Parallelism & Architecture
- Decouple long-running tasks with a task queue (Celery/RQ/Kafka) and worker pool; this avoids blocking the Flask stubs.
- Use asynchronous I/O for network-bound tasks (downloading artifacts, HTTP calls to LLMs or PlantUML server).
- For CPU-bound processing, run multiple worker processes (not threads) to bypass the GIL.

### E. Storage & I/O
- Store raw artifacts in object storage (S3/GCS) and stream only what's needed to parsers.
- Use compressed storage (gzip) where possible; decompress on worker nodes.
- Use SSDs for index storage and large dataset operations.

### F. LLM Cost & Latency Control
- Reduce prompt size by summarizing or extracting key facts before calling the LLM.
- Use a cheaper model or shorter max tokens for exploration; reserve large models for final synthesis.
- Batch LLM requests where supported, and set conservative `max_output_tokens` and timeouts.

### G. Monitoring & Autoscaling
- Collect metrics: embedding_latency, index_build_time, query_latency_p50/p95/p99, memory_usage, disk_io, LLM_cost_per_request.
- Autoscale embedding workers and index-serving pods based on queue length and p95 latency.

## 5. Engineering Practices & Benchmarks
- Add micro-benchmarks for each step:
  - `benchmark_embed.py` to measure tokens/sec and docs/sec for chosen models (CPU/GPU).
  - `benchmark_index_build.py` to time index creation for N vectors.
  - `benchmark_query_latency.py` to measure query p50/p95/p99 under load.

- Example quick local benchmark commands:

```bash
# measure embedding throughput for a file of documents
python -m timeit -n 10 -s "from demo.rag_demo_gemini import build_embeddings, read_rows; docs=read_rows('sample.csv', max_rows=1000); texts=[f'{r.get(\"Headline\")}' for r in docs]" "build_embeddings(texts)"

# simple query latency loop
python - <<'PY'
from demo.rag_demo_gemini import build_embeddings, build_faiss_index, retrieve, read_rows
rows = read_rows('sample.csv', max_rows=5000)
texts = [f"{r.get('Headline')}" for r in rows]
emb = build_embeddings(texts)
idx = build_faiss_index(emb)
q = build_embeddings(['market'])[0]
import time
start=time.time(); ids,dists=retrieve(idx, emb, q, top_k=5); print('latency', time.time()-start)
PY
```

## 6. Observability & Alerting
- Export Prometheus metrics (embedding time, index size, query latency) and set alerts for p95 latency SLO breaches.
- Log slow queries and heavy LLM requests to identify optimization targets.
- Track LLM spend by tagging requests and aggregating token usage.

## 7. Recommendations by Stage
- **Prototype / Local demo**: Use NumPy fallback, small embedding models, keep artifacts in `/tmp`, focus on correctness and UX.
- **Pilot (10k–100k docs)**: Add FAISS or a managed vector DB; enable GPU for embeddings; persist metadata to SQLite or small Postgres.
- **Production (>=100k docs)**: Use vector DB, object storage, autoscaled embedding & index-serving infrastructure, caching layers, rate limiting, and secured endpoints.

## 8. Cost Considerations
- GPU instances and cloud LLM tokens are the primary recurring costs.
- Trade recall vs cost by tuning ANN settings and using efficient model selection (small dense model + cross-encoder re-ranker only on top candidates).

## 9. Checklist for Performance Hardened Release
- [ ] Benchmarks for embedding throughput and index build time exist.
- [ ] Auto-scaling policy for workers is defined.
- [ ] Monitoring & alerting for p95/p99 latency configured.
- [ ] Persistence (object store + metadata DB) implemented.
- [ ] ANN index tuned, quantized, and persisted to disk.
- [ ] LLM cost controls (max tokens, model selection) implemented.