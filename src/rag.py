from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass
import json, os, datetime

from .config import Paths, Settings
from .retriever import retrieve
from .llm import LLMClient
from .utils import select_short_quote, format_citations

SYSTEM_PROMPT = (
    "You are ComplianceBot. Answer compliance questions using ONLY the provided context. "
    "Rules: (1) Be concise. (2) Use bullet points for the main answer. (3) Then add a 1–2 sentence short summary. "
    "(4) Do not invent facts; if the context is insufficient, answer exactly: 'Insufficient context.' "
    "(5) NEVER include content not grounded in the context."
)

@dataclass
class Answer:
    text: str
    quotes: List[str]
    citations: List[str]
    used_contexts: List[Dict[str, Any]]

def answer(query: str, paths: Paths = Paths(), settings: Settings = Settings()) -> Answer:
    """Top-level API used by notebook and app."""
    results, ok = retrieve(query, settings.top_k, settings.min_sim_threshold, settings, paths.artifacts_dir)
    if not ok:
        # Guardrail: insufficient
        return Answer(text="Insufficient context.", quotes=[], citations=[], used_contexts=[])

    # Build context
    context_blocks = []
    chosen_quotes = []
    for r in results:
        context_blocks.append(f"[Source: {r['doc_name']}#{r['anchor']}]\n{r['text']}")
        q = select_short_quote(r['text'])
        if q:
            chosen_quotes.append(q)

        # Build the prompt (triple-quoted so bullets are on new lines)
    context_str = "\n\n---\n\n".join(context_blocks[:settings.top_k])
    user_prompt = f"""Question: {query}

Context (authoritative sources; cite only these):
{context_str}

Write:
- Bullet points with the key answer.
- Then a short 1–2 sentence summary.
- Respond only from the context. Do not speculate.
"""

    llm = LLMClient(settings)
    raw = llm.generate(SYSTEM_PROMPT, user_prompt)

    # Add quotes (at least 1) — ensure unique & <= 30 words each
    quotes: List[str] = []
    for q in chosen_quotes[:2]:
        words = q.split()
        if len(words) > 30:
            q = " ".join(words[:30])
        if q and q not in quotes:
            quotes.append(q)

    if not quotes and results:
        # Fallback: take a short quote from the top result
        fallback = select_short_quote(results[0]["text"])
        if fallback:
            quotes.append(fallback)

    citations = format_citations(results)

    # Compose final answer text with quotes and citations
    parts = [raw.strip()]  # model-written bullets + short summary
    for q in quotes:
        parts.append(f'> Quote: "{q}"')
    if citations:
        parts.append("Citations: " + "; ".join(citations))

    formatted = "\n\n".join(parts)
    return Answer(text=formatted, quotes=quotes, citations=citations, used_contexts=results)


# Simple audit logger for the app
def log_audit(question: str, citations: List[str], paths: Paths = Paths()):
    os.makedirs(paths.logs_dir, exist_ok=True)
    log_path = os.path.join(paths.logs_dir, "audit_log.csv")
    ts = datetime.datetime.utcnow().isoformat()
    row = f'"{ts}","{question.replace(chr(34), chr(39))}","{"; ".join(citations).replace(chr(34), chr(39))}"\n'
    header_needed = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,question,citations\n")
        f.write(row)
