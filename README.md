# Reliable AI Knowledge Layer

A self-correcting retrieval and verification layer for enterprise AI assistants.
MVP vertical: an HR/procurement policy assistant with deliberately engineered
conflicting-policy and unanswerable questions (see the business/technical plan
this repo implements).

## What's here

```
src/
  config.py               # settings, provider selection (Groq default)
  retriever.py             # ChromaDB similarity search wrapper
  providers/                # pluggable LLM backends
    base.py                 # interface + factory (get_provider)
    groq_provider.py         # default
    openai_provider.py
    anthropic_provider.py
    gemini_provider.py
  graph/
    state.py                 # shared LangGraph state schema
    build.py                 # graph assembly + ablation variant routing (A-E)
    nodes/
      router.py              # §6.2 — needs retrieval?
      retrieve.py             # ChromaDB search
      grader.py                # §6.3 — per-chunk relevance grading
      rewrite.py                # query rewriting for the correction loop
      web_search.py              # Tavily fallback
      conflict.py                 # §6.4 — conflict & authority resolution (headline node)
      generator.py                 # answer generation (normal + conflict-surfacing modes)
      hallucination.py              # §6.5 — claim-level faithfulness grading
      evidence_status.py             # §6.6 — deterministic status derivation
data/
  documents.json            # synthetic policy corpus (deliberately conflicting)
  eval_questions.json       # 15 labeled Q&A pairs: straightforward / conflicting / unanswerable
scripts/
  ingest.py                 # chunk + embed + load into ChromaDB
eval/
  run_ablation.py           # run variants A-E, compute intervention rates by category
  ragas_eval.py              # optional deeper RAGAS faithfulness/relevancy pass
app/
  streamlit_app.py           # live reasoning-trace UI + evidence status card
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set DEFAULT_PROVIDER (groq by default) and the matching API key
```

Get a free Groq API key at https://console.groq.com (default backend — fast, free tier).
Set `DEFAULT_PROVIDER=openai|anthropic|gemini` in `.env` to switch backends; each
provider file documents its own required env var.

## Run it

```bash
# 1. Build the vector index from the synthetic policy corpus
python -m scripts.ingest

# 2. Launch the demo UI
streamlit run app/streamlit_app.py

# 3. (optional) Run the ablation study across all 5 variants
python -m eval.run_ablation --variants A B C D E

# 4. (optional) Deeper RAGAS scoring on top of the ablation results
python -m eval.ragas_eval
```

## Try these in the UI

- `"Can I approve this ₹3,00,000 purchase myself?"` → should trigger **CONFLICTING**
  (Engineering matrix vs. general 2025 policy — see `data/eval_questions.json` c01).
- `"What's the self-approval threshold for a general purchase?"` → should resolve
  cleanly to **VERIFIED** using the 2025-supersedes-2024 metadata.
- `"What is the maternity leave policy?"` → should resolve to **INSUFFICIENT EVIDENCE**
  (deliberately not in the corpus).

Switch the "Pipeline variant" dropdown in the sidebar to compare Variant A (plain
RAG, no safety net) against Variant E (full pipeline) on the same question.

## Design notes

- **Evidence status is deterministic**, not a self-reported confidence score — see
  `src/graph/nodes/evidence_status.py` for the exact signal → status table.
- **Conflict resolution never silently picks a side.** If chunk metadata (version,
  effective date, explicit "supersedes" / department-override language) can't
  resolve a disagreement, the generator switches into conflict-surfacing mode and
  the status is reported as CONFLICTING rather than guessed.
- **All LLM calls use structured JSON output** (native JSON mode where the
  provider supports it, defensive parsing otherwise) — no free-text parsing
  anywhere in the pipeline.
- **Retry loops are capped** (`MAX_REWRITE_ATTEMPTS`, `MAX_REGENERATIONS` in
  `.env`) to avoid infinite correction loops.

## Extending

- Swap in a different embedding model or a hosted vector DB by editing
  `src/retriever.py` + `scripts/ingest.py`.
- Add a new provider by implementing `LLMProvider` in `src/providers/` and
  registering it in `get_provider()` (`src/providers/base.py`).
- Add new eval categories by extending `data/eval_questions.json` — the ablation
  script automatically breaks out results by whatever `category` values it finds.
