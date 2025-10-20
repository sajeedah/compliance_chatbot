# src/retriever.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple

import numpy as np

from .config import Settings
from .utils import ensure_dir

def _embed_query(text: str, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vec = model.encode([text], normalize_embeddings=True)
    return vec.astype("float32")  # shape: (1, dim)

def load_index(artifacts_dir: str, index_name: str):
    """
    Reads:
      artifacts/{index_name}/index.faiss
      artifacts/{index_name}/metadata.jsonl
    Returns: (faiss.Index, List[Dict])
    """
    import faiss, json

    folder = os.path.join(artifacts_dir, index_name)
    index_path = os.path.join(folder, "index.faiss")
    meta_path = os.path.join(folder, "metadata.jsonl")

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Index or metadata not found. Run ingestion first.")

    index = faiss.read_index(index_path)

    records: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return index, records

def retrieve(query: str, top_k: int, min_sim_threshold: float, settings: Settings, artifacts_dir: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Returns (results, ok)
      results: list of dicts with keys: doc_name, anchor, text, score
      ok: True if best score >= threshold and results not empty
    """
    # Load index + metadata
    index, records = load_index(artifacts_dir, settings.index_name)

    # Embed query
    qvec = _embed_query(query, settings.embedding_model)  # (1, dim)

    # Search
    scores, idxs = index.search(qvec, top_k)  # scores shape (1, k), idxs shape (1, k)
    scores = scores[0]
    idxs = idxs[0]

    results: List[Dict[str, Any]] = []
    for score, i in zip(scores, idxs):
        if i < 0:
            continue
        rec = records[int(i)]
        results.append({
            "doc_name": rec["doc_name"],
            "anchor": rec["anchor"],
            "text": rec["text"],
            "score": float(score),
        })

    ok = len(results) > 0 and (results[0]["score"] >= min_sim_threshold)
    return results, ok
