"""
§10 Weeks 13-14 — Streamlit interface showing the reasoning trace live:
retrieved -> graded -> conflict detected -> resolved/surfaced -> generated -> verified.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import settings
from src.graph.build import run_pipeline

st.set_page_config(page_title="Reliable AI Knowledge Layer", page_icon="🛡️", layout="wide")

STATUS_STYLE = {
    "VERIFIED": ("✅", "#1a7f37", "Sufficient, non-conflicting evidence — all claims supported."),
    "PARTIALLY VERIFIED": ("🟡", "#9a6700", "Sufficient evidence, but some claims weren't fully supported."),
    "CONFLICTING": ("⚠️", "#c93c37", "Relevant evidence disagrees and could not be auto-resolved."),
    "INSUFFICIENT EVIDENCE": ("❔", "#57606a", "No adequate evidence found, even after correction attempts."),
    "UNVERIFIED": ("🚫", "#8250df", "Evidence looked adequate, but generated claims failed verification."),
}


def render_status_card(state: dict):
    status = state.get("evidence_status", "UNVERIFIED")
    icon, color, description = STATUS_STYLE.get(status, ("❓", "#57606a", ""))

    st.markdown(
        f"""
        <div style="border:2px solid {color}; border-radius:10px; padding:16px; margin-bottom:16px;">
            <div style="font-size:20px; font-weight:700; color:{color};">{icon} {status}</div>
            <div style="color:#555; margin-top:4px;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if state.get("answer"):
        st.markdown("**Answer**")
        st.write(state["answer"])

    if state.get("conflicts"):
        st.markdown("**Conflicts detected**")
        for c in state["conflicts"]:
            resolved_tag = "✅ resolved" if c["resolved"] else "⚠️ unresolved"
            st.write(f"- `{c['doc_ids']}` — {c['description']} ({resolved_tag})")
            if c["resolution"]:
                st.caption(c["resolution"])

    citations = state.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        for c in citations:
            meta = f"{c.get('title')}"
            if c.get("version"):
                meta += f" (v{c['version']}, effective {c.get('effective_date', '?')})"
            st.write(f"- {meta}")

    if state.get("claims"):
        with st.expander("Claim-level verification detail"):
            for claim in state["claims"]:
                tag = "✅" if claim["supported"] else "❌"
                st.write(f"{tag} {claim['text']}")
                if claim.get("reasoning"):
                    st.caption(claim["reasoning"])


def render_trace(state: dict):
    st.markdown("**Reasoning trace**")
    for line in state.get("trace", []):
        st.text(line)


def main():
    st.title("🛡️ Reliable AI Knowledge Layer")
    st.caption(
        "A self-correcting retrieval & verification layer — HR / procurement policy assistant demo. "
        f"Default LLM backend: **{settings.default_provider}**."
    )

    with st.sidebar:
        st.header("Settings")
        variant = st.selectbox(
            "Pipeline variant (ablation ladder)",
            ["E", "D", "C", "B", "A"],
            format_func=lambda v: {
                "A": "A — Plain RAG",
                "B": "B — + relevance grading",
                "C": "C — + self-correction loop",
                "D": "D — + conflict & authority resolution",
                "E": "E — Full pipeline (+ hallucination check)",
            }[v],
        )
        st.markdown("---")
        st.markdown("**Sample questions to try:**")
        st.code("Can I approve this purchase myself?", language=None)
        st.code("What's the self-approval threshold for a general purchase?", language=None)
        st.code("What is the maternity leave policy?", language=None)

    query = st.text_input("Ask a policy question", placeholder="e.g. Can I approve a ₹3,00,000 purchase myself?")
    run = st.button("Ask", type="primary")

    if run and query.strip():
        with st.spinner("Running pipeline..."):
            try:
                final_state = run_pipeline(query, variant=variant)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.info(
                    "Common causes: no API key set for the selected provider, or ChromaDB "
                    "hasn't been ingested yet (run `python -m scripts.ingest` first)."
                )
                return

        col1, col2 = st.columns([3, 2])
        with col1:
            render_status_card(final_state)
        with col2:
            render_trace(final_state)


if __name__ == "__main__":
    main()
