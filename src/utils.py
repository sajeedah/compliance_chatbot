from __future__ import annotations
import re, os, math, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# ----------- Cleaning -----------

def clean_text(text: str) -> str:
    # Basic cleanup suitable for rules text
    text = text.replace('\x0c', ' ')  # form feeds
    text = re.sub(r'[ \t]+', ' ', text)       # collapse spaces/tabs
    text = re.sub(r'\s*\n\s*', '\n', text) # tidy newlines
    text = re.sub(r'-\n', '', text)           # de-hyphenate line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # limit blank lines
    return text.strip()

# ----------- Chunking -----------

@dataclass
class Chunk:
    doc_name: str
    anchor: str      # e.g., 'p12' for PDFs or 'section-client-money' for MD
    text: str

def word_chunks(text: str, words_per_chunk: int = 400, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words).strip())
        if end == len(words):
            break
        start = end - overlap
    return [c for c in chunks if len(c.split()) >= max(40, words_per_chunk // 5)]

def md_sections_to_chunks(md_text: str) -> List[Tuple[str, str]]:
    """Return list of (anchor, section_text) using markdown headings (#, ##, ###).
    Anchor is derived from last seen heading.
    """
    lines = md_text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_anchor = 'section-0'
    current_buf: List[str] = []
    for line in lines:
        m = re.match(r'^(#{1,6})\s+(.*)$', line.strip())
        if m:
            # flush previous
            if current_buf:
                sections.append((current_anchor, current_buf))
            # new anchor
            title = m.group(2).strip()
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', title).strip('-').lower()
            current_anchor = f'section-{slug}'
            current_buf = []
        else:
            current_buf.append(line)
    if current_buf:
        sections.append((current_anchor, current_buf))
    out = []
    for anchor, buf in sections:
        txt = clean_text('\n'.join(buf))
        if txt:
            for ch in word_chunks(txt):
                out.append((anchor, ch))
    return out

def select_short_quote(text: str, max_words: int = 30) -> str:
    # pick a sentence (<= max_words) - prioritize sentences with 'shall', 'must', 'should'
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    key_words = ('shall', 'must', 'should', 'required', 'prohibit', 'oblig', 'ensure')
    # Rank: shorter and contains key words
    scored = []
    for s in sentences:
        wc = len(s.split())
        if wc == 0 or wc > max_words:
            continue
        score = 0
        lower = s.lower()
        for kw in key_words:
            if kw in lower:
                score += 2
        score += max(0, 20 - abs(15 - wc))  # prefer around ~15 words
        scored.append((score, s.strip().strip('"')))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    # fallback: trim first sentence
    for s in sentences:
        if s.strip():
            words = s.split()[:max_words]
            return ' '.join(words).strip()
    return ''

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def format_citations(items):
    """
    Format citations from retrieved items into `doc#anchor` strings,
    deduplicated but in original order.

    Each item is expected to have keys:
      - 'doc_name': like 'VARA_Client_Money.md'
      - 'anchor' : like 'e-reconciliation' or 'p12'
    """
    citations = []
    for r in items:
        doc = r.get("doc_name", "").strip()
        anchor = r.get("anchor", "").strip().lstrip("#")
        if doc and anchor:
            citations.append(f"{doc}#{anchor}")
        elif doc:
            citations.append(doc)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

