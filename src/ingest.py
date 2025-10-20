# src/ingest.py
from __future__ import annotations
import os, argparse, json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .config import Paths, Settings
from .utils import clean_text, word_chunks, md_sections_to_chunks, Chunk, ensure_dir

# -------- PDF parsing --------
def extract_pdf_text(path: str) -> List[Dict[str, Any]]:
    """Return list of pages: [{'page': 1, 'text': '...'}, ...]"""
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("Install pypdf to parse PDFs") from e
    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append({'page': i + 1, 'text': clean_text(txt)})
    return pages

def load_markdown(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

@dataclass
class Record:
    doc_name: str
    anchor: str
    text: str

def build_corpus(docs_dir: str) -> List[Record]:
    records: List[Record] = []
    for root, _, files in os.walk(docs_dir):
        for name in files:
            low = name.lower()
            full = os.path.join(root, name)
            if low.endswith('.pdf'):
                for page in extract_pdf_text(full):
                    if page['text'].strip():
                        for ch in word_chunks(page['text']):
                            records.append(Record(doc_name=name, anchor=f"p{page['page']}", text=ch))
            elif low.endswith('.md'):
                md = load_markdown(full)
                for anchor, ch in md_sections_to_chunks(md):
                    records.append(Record(doc_name=name, anchor=anchor, text=ch))
            # extendable: .txt, .html, etc.
    return records

def embed_records(records: List[Record], model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [r.text for r in records]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vectors  # numpy array

def save_jsonl(records: List[Record], out_path: str):
    # Save metadata as JSONL (plain dicts, cloud-safe)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def build_faiss_index(vectors, records: List[Record], artifacts_dir: str, index_name: str):
    """
    Writes to:
      artifacts/{index_name}/index.faiss
      artifacts/{index_name}/metadata.jsonl
    """
    import faiss, numpy as np

    out_dir = os.path.join(artifacts_dir, index_name)
    ensure_dir(out_dir)

    # FAISS index (IP with normalized vectors = cosine similarity)
    if hasattr(vectors, "astype"):
        vecs = vectors.astype("float32")
        dim = vecs.shape[1]
    else:
        import numpy as np
        vecs = np.array(vectors, dtype="float32")
        dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))

    # Metadata as JSONL (no pickle/classes)
    save_jsonl(records, os.path.join(out_dir, "metadata.jsonl"))

    return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_dir', default=str(Paths().docs_dir))
    parser.add_argument('--artifacts_dir', default=str(Paths().artifacts_dir))
    args = parser.parse_args()

    settings = Settings()
    print(f"Loading documents from: {args.docs_dir}")
    records = build_corpus(args.docs_dir)
    if not records:
        print("No documents found. Put PDFs/MD files in ./docs and rerun.")
        return
    print(f"Built {len(records)} chunks. Embedding...")
    vectors = embed_records(records, settings.embedding_model)

    print("Saving FAISS index + metadata...")
    build_faiss_index(vectors, records, args.artifacts_dir, settings.index_name)
    print("Done.")

if __name__ == '__main__':
    main()
