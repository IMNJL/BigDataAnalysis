"""RAG demo with optional Gemini (Vertex AI) summarization.

Usage:
    python rag_demo_gemini.py --artifact-id <artifact_id> --mcp-base http://127.0.0.1:5006

Notes:
- If environment variable `GEMINI_MODEL` is set and Google credentials are available
  (set `GOOGLE_APPLICATION_CREDENTIALS`), the demo will attempt to call Vertex AI
  TextGeneration (Gemini). Otherwise it will produce a simple template-based summary.
"""
import os
import argparse
import requests
import csv
import tempfile
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
import numpy as np
import re

# sentence-transformers can be heavy and may import TensorFlow/Keras which
# can fail on some environments (Keras 3 incompatibility). Try importing
# but fall back to a lightweight TF embedding implementation below.
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print('sentence_transformers unavailable, falling back to simple TF embeddings:', e)

try:
    from google.cloud import aiplatform
    AIP_AVAILABLE = True
except Exception:
    AIP_AVAILABLE = False


def download_artifact(mcp_base, artifact_id, out_path=None):
    url = f"{mcp_base.rstrip('/')}/artifact/{artifact_id}/download"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix=f"artifact_{artifact_id}_", suffix='.csv')
        os.close(fd)
    with open(out_path, 'wb') as fh:
        for chunk in r.iter_content(1024*8):
            fh.write(chunk)
    return out_path


def read_rows(path, max_rows=None):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(r)
    return rows


def build_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return emb

    # Lightweight fallback: simple term-frequency vectors (lowercased words)
    print('Using lightweight TF embedding fallback (no sentence-transformers).')
    # Build vocabulary from texts (limit vocab size to keep vectors small)
    token_lists = [re.findall(r"\w+", t.lower()) for t in texts]
    freq = {}
    for toks in token_lists:
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
    # keep most frequent tokens up to a cap
    max_vocab = 2000
    vocab_items = sorted(freq.items(), key=lambda kv: -kv[1])[:max_vocab]
    vocab = {w: i for i, (w, _) in enumerate(vocab_items)}

    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        for t in toks:
            idx = vocab.get(t)
            if idx is not None:
                mat[i, idx] += 1.0
    # L2-normalize rows to mimic embedding cosine behavior
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat


def build_faiss_index(embeddings):
    """Return a faiss index if available, otherwise return the numpy embeddings as the index object.

    The retrieve function below handles both cases.
    """
    if FAISS_AVAILABLE:
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index
    else:
        # fallback: return embeddings array and do brute-force search in retrieve()
        return embeddings


def retrieve(index, embeddings, query_emb, top_k=5):
    """Retrieve top_k indices and distances.

    If a faiss index is provided, use it. Otherwise `index` is expected to be a numpy
    array of embeddings and we perform a brute-force L2 search.
    """
    if FAISS_AVAILABLE:
        D, I = index.search(np.array([query_emb]), top_k)
        return I[0], D[0]
    else:
        emb_array = index  # numpy array
        # compute L2 distances
        diffs = emb_array - query_emb
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argsort(dists)[:top_k]
        return idx.tolist(), dists[idx].tolist()


def call_gemini(prompt, model_name=None, max_output_tokens=512):
    model = model_name or os.environ.get('GEMINI_MODEL')
    if not model or not AIP_AVAILABLE:
        return None
    # Initialize aiplatform if not already
    project = os.environ.get('GOOGLE_CLOUD_PROJECT')
    location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    aiplatform.init(project=project, location=location)
    tg_model = aiplatform.TextGenerationModel.from_pretrained(model)
    response = tg_model.predict(prompt, max_output_tokens=max_output_tokens)
    return response.text


def simple_template_summary(retrieved_rows):
    # Basic aggregation: list top headlines and count by sentiment
    headlines = [r.get('Headline') or '(no headline)' for r in retrieved_rows]
    sentiments = [r.get('Sentiment') or 'Unknown' for r in retrieved_rows]
    from collections import Counter
    sct = Counter(sentiments)
    summary = 'Top retrieved headlines:\n'
    for h in headlines[:5]:
        summary += f'- {h}\n'
    summary += '\nSentiment counts:\n'
    for k,v in sct.items():
        summary += f'- {k}: {v}\n'
    return summary


def run_demo(artifact_id, mcp_base='http://127.0.0.1:5006', query='market', top_k=5, use_gemini=True):
    print('Downloading artifact...')
    path = download_artifact(mcp_base, artifact_id)
    print('Saved to', path)

    rows = read_rows(path)
    docs = []
    for r in rows:
        # Create a short document per row
        doc = f"{r.get('Date','')} | {r.get('Headline','')} | {r.get('Source','')} | {r.get('Related_Company','')}"
        docs.append((doc, r))

    texts = [d[0] for d in docs]
    if not texts:
        print('No documents found in artifact')
        return

    print('Building embeddings...')
    emb = build_embeddings(texts)
    index = build_faiss_index(emb)

    # encode query
    print('Encoding query...')
    q_emb = build_embeddings([query])[0]
    ids, dists = retrieve(index, emb, q_emb, top_k=top_k)
    retrieved = [docs[i][1] for i in ids]

    print('\n--- Retrieval results ---')
    for i, r in enumerate(retrieved, 1):
        print(f"{i}. {r.get('Date')} | {r.get('Headline')} | {r.get('Source')}\n")

    # Summarize with Gemini if available and allowed
    summary = None
    if use_gemini and AIP_AVAILABLE:
        prompt = 'Summarize the following news items and provide a concise investment-focused summary:\n\n'
        for r in retrieved:
            prompt += f"- {r.get('Date')} {r.get('Headline')} ({r.get('Related_Company')}) -- Sentiment: {r.get('Sentiment')}\n"
        print('\nCalling Gemini/Vertex AI...')
        try:
            resp = call_gemini(prompt)
            if resp:
                summary = resp
        except Exception as e:
            print('Gemini call failed:', e)

    if not summary:
        print('\nFalling back to template summary...')
        summary = simple_template_summary(retrieved)

    print('\n--- Summary ---\n')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--mcp-base', default='http://127.0.0.1:5006')
    parser.add_argument('--query', default='market')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--no-gemini', action='store_true')
    args = parser.parse_args()
    run_demo(args.artifact_id, mcp_base=args.mcp_base, query=args.query, top_k=args.top_k, use_gemini=not args.no_gemini)
