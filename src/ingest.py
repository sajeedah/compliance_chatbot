from __future__ import annotations
import os, argparse, json, pickle
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .config import Paths, Settings
from .utils import clean_text, word_chunks, md_sections_to_chunks, Chunk, ensure_dir

# PDF parsing
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
        pages.append({'page': i+1, 'text': clean_text(txt)})
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
            if name.lower().endswith('.pdf'):
                full = os.path.join(root, name)
                for page in extract_pdf_text(full):
                    if page['text'].strip():
                        for ch in word_chunks(page['text']):
                            records.append(Record(doc_name=name, anchor=f"p{page['page']}", text=ch))
            elif name.lower().endswith('.md'):
                full = os.path.join(root, name)
                md = load_markdown(full)
                for anchor, ch in md_sections_to_chunks(md):
                    records.append(Record(doc_name=name, anchor=anchor, text=ch))
            # you can extend for .txt if needed
    return records

def embed_records(records: List[Record], model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [r.text for r in records]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vectors

def build_faiss_index(vectors, records: List[Record], artifacts_dir: str, index_name: str):
    import faiss, numpy as np
    dim = vectors.shape[1] if hasattr(vectors, 'shape') else len(vectors[0])
    index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
    index.add(vectors.astype('float32'))
    # save
    ensure_dir(artifacts_dir)
    faiss.write_index(index, os.path.join(artifacts_dir, f"{index_name}.faiss"))
    with open(os.path.join(artifacts_dir, f"{index_name}.meta.pkl"), "wb") as f:
        pickle.dump(records, f)
    return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_dir', default=Paths().docs_dir)
    parser.add_argument('--artifacts_dir', default=Paths().artifacts_dir)
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
