"""
LLM + Neo4j Knowledge Graph Pipeline

Stages implemented:
- Data Preprocessing
- Entity and Relationship Extraction (LLM-backed with a mock fallback)
- Data Cleaning and Conversion
- Graph Construction (py2neo)
- Natural Language to Cypher translator

This file is a scaffold and contains a working experiment using mocked LLM output for offline testing.
"""

from typing import List, Tuple, Dict, Any
import os
import re
import json
from dataclasses import dataclass

# Optional imports for graph database
try:
    from py2neo import Graph, Node, Relationship
except Exception:
    Graph = None

# Setup a small dataclass for triplets
@dataclass
class Triplet:
    subject: str
    predicate: str
    object: str
    meta: Dict[str, Any] = None


# -----------------------
# Data Preprocessing
# -----------------------

def preprocess_text(text: str) -> List[str]:
    """Convert raw text into cleaned sentences/segments.

    Objective: Convert unstructured text into a processable format.
    Method: split into sentences, remove unwanted characters, normalize whitespace.
    """
    # Basic normalization
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence split (simple rule-based split)
    segments = re.split(r'(?<=[.!?]) +', text)

    # Clean segments
    cleaned = []
    for s in segments:
        s2 = s.strip()
        # remove weird unicode control chars
        s2 = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s2)
        if len(s2) > 10:
            cleaned.append(s2)
    return cleaned


# -----------------------
# Entity and Relationship Extraction (LLM)
# -----------------------

def llm_extract_triplets(segments: List[str], llm_api_key: str = None) -> List[Triplet]:
    """Call an LLM to extract triplets. Falls back to a mock extractor if no key.

    Objective: Extract structured triplets from the text.
    Method: Use an LLM for zero/few-shot extraction. Prompt must specify entity and relationship types.
    """
    # If no API key provided, return a mocked set for testing
    if not llm_api_key:
        return mock_llm_extractor(segments)

    # Placeholder for real LLM call: implement your provider's API here.
    # The prompt should include examples of desired JSON output: [{"subject":"...","predicate":"...","object":"..."}, ...]
    raise NotImplementedError("LLM integration not implemented in scaffold. Provide your provider wrapper here.")


def mock_llm_extractor(segments: List[str]) -> List[Triplet]:
    """A mocked extractor that returns example triplets for demo purposes."""
    triplets = []
    for s in segments:
        # Simple regex-based heuristics for mocked output
        if 'acquire' in s.lower() or 'acquisition' in s.lower():
            triplets.append(Triplet('Company A', 'acquired', 'Company B', {'source_sentence': s}))
        if 'profit' in s.lower() or 'revenue' in s.lower():
            triplets.append(Triplet('Company A', 'reported', 'increased profit', {'source_sentence': s}))
        if 'announced' in s.lower():
            triplets.append(Triplet('Company A', 'announced', 'new product', {'source_sentence': s}))
    if not triplets:
        triplets.append(Triplet('Company A', 'mentioned_in', 'news', {'source_sentence': segments[0] if segments else ''}))
    return triplets


# -----------------------
# Data Cleaning and Conversion
# -----------------------

def clean_triplets(raw_triplets: List[Triplet]) -> List[Triplet]:
    """Standardize triplet formatting and remove obvious errors.

    Objective: Standardize the output format and remove erroneous data.
    Method: handle incorrect relationship directions, normalize text, deduplicate.
    """
    cleaned = []
    seen = set()
    for t in raw_triplets:
        subj = normalize_entity(t.subject)
        pred = normalize_predicate(t.predicate)
        obj = normalize_entity(t.object)
        key = (subj, pred, obj)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(Triplet(subj, pred, obj, t.meta or {}))
    return cleaned


def normalize_entity(e: str) -> str:
    e = e.strip()
    e = re.sub(r'\s+', ' ', e)
    # Remove trailing punctuation
    e = e.strip(' .,:;')
    return e


def normalize_predicate(p: str) -> str:
    p = p.strip().lower()
    p = p.replace(' ', '_')
    p = re.sub(r'[^a-z0-9_]', '', p)
    return p


# Convert triplets to dicts for easy JSON export
def triplets_to_dicts(triplets: List[Triplet]) -> List[Dict[str, Any]]:
    return [{
        'subject': t.subject,
        'predicate': t.predicate,
        'object': t.object,
        'meta': t.meta or {}
    } for t in triplets]


# -----------------------
# Graph Construction and Visualization
# -----------------------

def triplets_to_cypher(triplets: List[Triplet]) -> List[str]:
    """Generate Cypher MERGE statements for triplets.

    Method: Use MERGE for nodes and MERGE/Create relationships. Use simple labeling heuristics.
    """
    stmts = []
    for t in triplets:
        subj_label = label_for_entity(t.subject)
        obj_label = label_for_entity(t.object)
        subj_prop = escape_cypher_literal(t.subject)
        obj_prop = escape_cypher_literal(t.object)
        rel = t.predicate
        stmt = (
            f"MERGE (a:{subj_label} {{name: '{subj_prop}'}})\n"
            f"MERGE (b:{obj_label} {{name: '{obj_prop}'}})\n"
            f"MERGE (a)-[r:{rel}]->(b);"
        )
        stmts.append(stmt)
    return stmts


def escape_cypher_literal(s: str) -> str:
    return s.replace("'", "\\'")


def label_for_entity(entity: str) -> str:
    # Very naive heuristics: if contains Company or Corp give Company label
    if any(x in entity.lower() for x in ['company', 'inc', 'corp', 'ltd']):
        return 'Company'
    # If contains date-like or percent return Attribute
    if re.search(r'\d{4}|%|\d+\.?\d*%', entity):
        return 'Attribute'
    return 'Entity'


def push_to_neo4j(triplets: List[Triplet], uri=None, user=None, password=None):
    """Push triplets to Neo4j using py2neo. Requires a running Neo4j instance and credentials.

    Method: Connect using py2neo.Graph and run generated Cypher statements.
    """
    if Graph is None:
        raise RuntimeError('py2neo not available. Install requirements or run in an env with py2neo.')
    uri = uri or os.getenv('NEO4J_URI')
    user = user or os.getenv('NEO4J_USER')
    password = password or os.getenv('NEO4J_PASSWORD')
    if not (uri and user and password):
        raise ValueError('NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD must be set or provided')
    g = Graph(uri, auth=(user, password))
    stmts = triplets_to_cypher(triplets)
    for s in stmts:
        g.run(s)
    return True


# -----------------------
# Natural Language to Cypher translator
# -----------------------

def nl_to_cypher(nl_query: str, sample_triplets: List[Triplet]) -> str:
    """Translate a short natural language query into a Cypher query.

    Objective: allow end users to query the KG in natural language.
    Method: simple rule-based mapping for demo; replace with LLM mapping for production.
    """
    q = nl_query.lower()
    # Example patterns
    if 'who acquired' in q or 'which company acquired' in q:
        # return matches for predicate 'acquired'
        return "MATCH (a)-[r:acquired]->(b) RETURN a.name, b.name LIMIT 50"
    if 'profits' in q or 'reported profit' in q:
        return "MATCH (a)-[r:reported_increased_profit]->(b) RETURN a.name, r, b.name LIMIT 50"
    if 'show relations' in q:
        return "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 200"
    # fallback: return a graph browse
    return "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 200"


# -----------------------
# Utilities and runner
# -----------------------

def run_pipeline_on_text(text: str, llm_api_key: str = None) -> Dict[str, Any]:
    segments = preprocess_text(text)
    raw = llm_extract_triplets(segments, llm_api_key=llm_api_key)
    cleaned = clean_triplets(raw)
    cypher = triplets_to_cypher(cleaned)
    return {
        'segments': segments,
        'raw_triplets': triplets_to_dicts(raw),
        'cleaned_triplets': triplets_to_dicts(cleaned),
        'cypher': cypher
    }


if __name__ == '__main__':
    # quick local demo
    demo_text = (
        "Acme Inc. announced a new product today. The company reported increased profit in Q4. "
        "In related news, BigCo Ltd announced the acquisition of SmallCo."
    )
    out = run_pipeline_on_text(demo_text)
    print(json.dumps(out, indent=2))
