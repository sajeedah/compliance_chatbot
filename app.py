# app.py — Streamlit mini app for ComplianceBot

from __future__ import annotations
import os
from pathlib import Path
import sys
import streamlit as st

# ---------- Ensure we can import from src/ ----------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.config import Paths
from src.rag import answer, log_audit

# ---------- Streamlit page config ----------
st.set_page_config(page_title="ComplianceBot", page_icon="⚖️", layout="centered")

# ---------- Header ----------
st.title("⚖️ ComplianceBot (FATF + VARA)")
st.caption("Grounded answers with quotes & citations. If context is missing → bot says 'Insufficient context.'")

# ---------- First-run: ensure FAISS index exists (auto-build if missing) ----------
from src.ingest import main as ingest_main

paths = Paths()
faiss_dir = Path(paths.artifacts_dir) / "faiss_index"
docs_dir = Path(paths.docs_dir)

# Quick sanity check for docs
required_docs = [
    docs_dir / "FATF_Recommendations.pdf",
    docs_dir / "Explanatory_Note_R16.pdf",
    docs_dir / "VARA_Client_Money.md",
]
missing = [str(p.name) for p in required_docs if not p.exists()]
if missing:
    st.error(
        "Required source files are missing in the `docs/` folder:\n\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n\nAdd them to `docs/` in your repo and reload the app."
    )
    st.stop()

# Build index only if missing/empty
if not faiss_dir.exists() or not any(faiss_dir.iterdir()):
    with st.spinner("Building index from docs (first-time setup, please wait)..."):
        try:
            ingest_main()  # builds embeddings + FAISS using files in docs/
        except Exception as e:
            st.error(f"Index build failed: {e}")
            st.stop()

# ---------- Input form ----------
with st.form("ask_form", clear_on_submit=False):
    query = st.text_input("Ask ComplianceBot a question", placeholder="e.g., What does FATF Rec 16 require for wire transfers?")
    submitted = st.form_submit_button("Answer")

# ---------- Handle submit ----------
if submitted:
    if not query.strip():
        st.warning("Please type a question.")
        st.stop()

    # Run the RAG pipeline
    with st.spinner("Searching authoritative sources and generating answer..."):
        res = answer(query)  # returns Answer(text=..., quotes=[...], citations=[...], used_contexts=[...])

    # ---------- Guardrail result ----------
    if res.text.strip() == "Insufficient context.":
        st.error("Insufficient context.")
        # Optional: show what sources were *almost* matched
        if res.used_contexts:
            with st.expander("Closest sources (for debugging)"):
                for r in res.used_contexts[:3]:
                    st.write(f"- {r['doc_name']}#{r['anchor']}")
        st.stop()

    # ---------- Answer panel ----------
    st.subheader("Answer")
    st.markdown(res.text)  # already formatted as bullets + short summary + quotes + citations

    # Optional: show structured sections separately
    with st.expander("Quotes (structured view)"):
        if res.quotes:
            for q in res.quotes:
                st.write(f'> "{q}"')
        else:
            st.write("No quotes captured.")

    with st.expander("Citations (structured view)"):
        if res.citations:
            for c in res.citations:
                st.code(c)
        else:
            st.write("No citations captured.")

    # ---------- Audit log ----------
    try:
        os.makedirs(paths.logs_dir, exist_ok=True)
        log_audit(query, res.citations, paths=paths)
        st.success("Logged to audit trail.")
        st.caption(f"Log file: {Path(paths.logs_dir) / 'audit_log.csv'}")
    except Exception as e:
        st.warning(f"Could not write audit log: {e}")

# ---------- Footer ----------
st.write("---")
st.caption("Tip: Try questions like “Under FATF Rec 16, what info must accompany a wire transfer?” or “How should a VARA firm handle a Client Money shortfall?”")
