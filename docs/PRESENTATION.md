# 10-minute presentation outline

## 0:00–1:00 · Problem and goal
- Clinicians/researchers ask compound questions ("warfarin + sertraline: bleeding risk and monitoring?").
- A plain LLM answers from memory: no sources, may invent interactions or thresholds.
- Goal: answers grounded **only** in 21 local documents (DailyMed labels, MedlinePlus, glossary), with line-level
  citations, a reviewer step, and an explicit "not medical advice" disclaimer.

## 1:00–2:00 · Technologies and why
- FastAPI + Pydantic (typed API and tool-argument validation), OpenAI SDK against an OpenAI-compatible proxy.
- NumPy in-memory cosine similarity: 525 × 1536 matrix, one dot product per query — a vector DB adds nothing here.
- No LangChain/LangGraph: a ~150-line loop on native function calling is easier to verify and explain.
- React + Vite UI, built inside Docker and served by FastAPI.

## 2:00–4:00 · Architecture and ingestion
- Show `docs/ARCHITECTURE.md` component diagram.
- Ingestion: loader (path-confined, encoding fallback) → heading-aware chunker (breadcrumbs, tables kept whole,
  exact line ranges) → batched embeddings → L2-normalised matrix.
- Decision: rebuild the index at startup (~15 s) instead of caching — never stale, works with the evaluator's model.

## 4:00–6:00 · RAG and agentic workflow
- PLAN: LLM produces sub-questions (show Q3 plan in the UI trace).
- RESEARCH: LLM chooses tools each iteration. Show Q3: 2 searches, then 5 targeted section reads across both labels.
- Evidence ledger: every passage stored with file + lines; de-duplicated; budgeted context.
- ANSWER: evidence-only prompt, mandatory `[file:Lx-Ly]` citations, explicit "documents do not state…" for gaps.

## 6:00–7:00 · Tool selection safety and critic
- Tool registry: Pydantic schemas → function-calling JSON schema; validation errors go back to the model.
- `read_source_document` uses an allow-list of in-memory documents: `../../.env` is impossible by construction (tests).
- Critic: code validates citations (fabricated file / guessed lines rejected); LLM checks each claim against evidence.
  Code can veto an LLM "approve". One revision max, then `needs_review` with visible notes.

## 7:00–8:00 · Evaluation
- `python -m eval.run_eval` → new run directory with full outputs + fingerprint; human check in `eval/MANUAL_REVIEW.md`.
- Reported run: Q1–Q4 `success`, expected sources cited, 0 invalid citations; out-of-corpus control (lisinopril)
  → `insufficient_evidence`, no citations.
- Critic at work: Q3 first draft implied routine warfarin dose adjustment when sertraline starts and omitted the
  sertraline protein-binding instruction → flagged → one revision fixed both.
- Q4: answer states that "25 mL/min" is not body-surface indexed while the label threshold is mL/min/1.73 m².

## 8:00–9:00 · Live demo
1. `docker run -p 8000:8000 --env-file .env agent`; `curl localhost:8000/health`.
2. UI at http://localhost:8000 → click the Q3 example → open "How the agent worked" (plan, tool calls, reviews).
3. Ask something outside the corpus (e.g. "What is the dose of lisinopril?") → insufficient evidence, no invention.

## 9:00–10:00 · Challenges and decisions
- Guessed line numbers (off by 2) → per-line numbering for exact reads + strict citation validator + regression test.
- Out-of-corpus answers cited unrelated passages → corpus-coverage note, critic `irrelevant_citations` removed by
  code, citation-free abstention treated as `insufficient_evidence`.
- Omissions/units: verified the evidence *was* in context (no truncation) → fixed in generator/critic rules, plus a
  deterministic number/unit detector. A first completeness rule was too broad (bloated answers) — narrowed.
- Evaluation reproducibility: runs were being overwritten → immutable run directories with code fingerprints.
- Provider errors leaked to the browser → generic client messages, details only in server logs.
- Trade-offs: latency (10–50 s, 5–12 LLM calls) vs grounding; LLM critic helps but is not ground truth.
