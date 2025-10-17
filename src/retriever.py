from __future__ import annotations
import os, pickle, argparse
from typing import List, Dict, Any
import numpy as np

from .config import Paths, Settings
from .utils import Chunk

def load_index(artifacts_dir: str, index_name: str):
    import faiss
    index_path = os.path.join(artifacts_dir, f"{index_name}.faiss")
    meta_path = os.path.join(artifacts_dir, f"{index_name}.meta.pkl")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Index or metadata not found. Run ingestion first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        records = pickle.load(f)  # List[Record]
    return index, records

def embed_query(q: str, model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vec = model.encode([q], normalize_embeddings=True)
    return vec.astype('float32')

def retrieve(query: str, top_k: int, min_sim: float, settings: Settings, artifacts_dir: str):
    index, records = load_index(artifacts_dir, settings.index_name)
    q_vec = embed_query(query, settings.embedding_model)
    D, I = index.search(q_vec, top_k)  # inner product == cosine since normalized
    D = D[0]
    I = I[0]
    results = []
    for score, idx in zip(D, I):
        if idx == -1:
            continue
        rec = records[idx]
        results.append({
            "score": float(score),
            "doc_name": rec.doc_name,
            "anchor": rec.anchor,
            "text": rec.text
        })
    # guardrail decision
    has_signal = len(results) > 0 and results[0]["score"] >= min_sim
    return results, has_signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--artifacts_dir", default=Paths().artifacts_dir)
    args = parser.parse_args()
    settings = Settings()
    res, ok = retrieve(args.query, settings.top_k, settings.min_sim_threshold, settings, args.artifacts_dir)
    print({"ok": ok, "results": res})
