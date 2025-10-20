# app.py — Streamlit mini app for ComplianceBot (stable version)

from __future__ import annotations
import os, sys, time
from pathlib import Path
import streamlit as st

# import src/
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.config import Paths
from src.rag import answer, log_audit
from openai import RateLimitError, APIError, APITimeoutError

st.set_page_config(page_title="ComplianceBot", page_icon="⚖️", layout="centered")

st.title("⚖️ ComplianceBot (FATF + VARA)")
st.caption("Grounded answers with quotes & citations. If context is missing → bot says 'Insufficient context.'")

# --- Simple per-session throttle (prevents rapid clicks) ---
if "last_call_ts" not in st.session_state:
    st.session_state.last_call_ts = 0.0

# --- Optional per-session cache (avoids re-calling API for repeated Qs) ---
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}

# ---------- Input form ----------
with st.form("ask_form", clear_on_submit=False):
    query = st.text_input("Ask ComplianceBot a question", placeholder="e.g., What does FATF Rec 16 require for wire transfers?")
    submitted = st.form_submit_button("Answer")

# ---------- Handle submit ----------
if submitted:
    if not query.strip():
        st.warning("Please type a question.")
        st.stop()

    # Throttle: 1 request per 3 seconds per browser session
    now = time.time()
    if now - st.session_state.last_call_ts < 3:
        st.info("Taking a short breather — please wait a moment between questions.")
        st.stop()
    st.session_state.last_call_ts = now

    # Serve from cache if available
    key = query.strip().lower()
    if key in st.session_state.qa_cache:
        st.subheader("Answer (cached)")
        st.markdown(st.session_state.qa_cache[key])
        st.stop()

    # Run the RAG pipeline with friendly error handling
    try:
        with st.spinner("Searching authoritative sources and generating answer..."):
            res = answer(query)  # Answer(text=..., quotes=[...], citations=[...], used_contexts=[...])
    except (RateLimitError, APITimeoutError):
        st.warning("⏳ High load detected. Please wait a few seconds and try again.")
        st.stop()
    except APIError as e:
        st.error(f"Upstream API error: {e.__class__.__name__}. Please try again shortly.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    # Guardrail: insufficient
    if res.text.strip() == "Insufficient context.":
        st.error("Insufficient context.")
        if res.used_contexts:
            with st.expander("Closest sources (for debugging)"):
                for r in res.used_contexts[:3]:
                    st.write(f"- {r['doc_name']}#{r['anchor']}")
        st.stop()

    # Answer panel
    st.subheader("Answer")
    st.markdown(res.text)

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

    # Audit log
    try:
        paths = Paths()
        os.makedirs(paths.logs_dir, exist_ok=True)
        log_audit(query, res.citations, paths=paths)
        st.success("Logged to audit trail.")
        st.caption(f"Log file: {Path(paths.logs_dir) / 'audit_log.csv'}")
    except Exception as e:
        st.warning(f"Could not write audit log: {e}")

    # Cache the final composed text to absorb repeat clicks
    st.session_state.qa_cache[key] = res.text

# Footer + Disclaimer
st.write("---")
st.caption("""
**Disclaimer:** This tool is for educational and research purposes only. 
Regulatory rules and guidance are frequently updated, and interpretations may vary. 
Always verify information against the latest official FATF and VARA publications before making compliance decisions.
""")
