from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Paths:
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    docs_dir: str = os.path.join(base_dir, "docs")
    artifacts_dir: str = os.path.join(base_dir, "artifacts")
    logs_dir: str = os.path.join(base_dir, "data", "logs")

@dataclass(frozen=True)
class Settings:
    # Embeddings & index
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    index_name: str = os.getenv("INDEX_NAME", "faiss_index")
    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_sim_threshold: float = float(os.getenv("MIN_SIM", "0.30"))
    # LLM
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
