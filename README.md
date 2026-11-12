# Agentic Pharmacology Research Assistant

An agent that answers compound pharmacology questions **only from a local corpus** of FDA DailyMed drug labels,
MedlinePlus consumer pages and a glossary. It plans, chooses its own tools, retrieves evidence, writes an answer
with line-level citations, has the answer checked by a critic, and always adds a medical disclaimer.

> Research and educational use only — not medical advice.

## What it does

| Capability (assignment) | Implementation |
|---|---|
| **PLAN** — decompose the question | `app/agent/planner.py`: the LLM returns drugs + sub-questions (JSON) |
| **RETRIEVE** — in-memory similarity | `app/retrieval/`: OpenAI embeddings + NumPy cosine similarity over 525 chunks |
| **USE TOOLS** — ≥2 local tools, agent-selected | `app/agent/tools.py` + `researcher.py`: `search_corpus`, `read_source_document` via native function calling |
| **REVIEW** — critic checks grounding | `app/agent/critic.py`: deterministic citation checks + LLM claim-by-claim review; max one revision |
| **ANSWER** — cited, with disclaimer | `app/rag/generator.py`: citations `[file.md:Lx-Ly]`; disclaimer appended by code |

## Architecture

```
React UI ──POST /ask──▶ FastAPI ──▶ ResearchAgent
                                     1. Planner        (LLM → sub-questions)
                                     2. Research loop  (LLM picks tools; ≤6 iterations, ≤12 calls)
                                          ├─ search_corpus ──▶ Retriever ─▶ NumPy vector store
                                          └─ read_source_document ──▶ in-memory documents
                                        every passage → EvidenceLedger (file + line range)
                                     3. Generator      (answer from evidence only, cited)
                                     4. Critic         (citation validator + LLM review)
                                     5. ≤1 revision    → final status: success | needs_review | insufficient_evidence
```

Diagrams and design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Technologies

| Choice | Reason |
|---|---|
| Python 3.13, FastAPI, Pydantic v2 | Typed request/response and tool-argument validation |
| OpenAI Python SDK (any OpenAI-compatible endpoint) | Native tool calling; the evaluator's proxy/models are injected via env vars |
| NumPy | 525 vectors fit in memory; cosine similarity is one matrix product — no vector DB needed |
| tiktoken | Exact token counts for chunk sizing |
| React + Vite | Small UI, built in Docker and served by FastAPI from the same origin |
| No agent framework | The loop is ~150 lines and fully explainable; no LangChain/LangGraph |

## Corpus, ingestion and retrieval

- `app/data/corpus/`: 10 `dailymed-*.md`, 10 `medlineplus-*.md`, `GLOSSARY.md` (21 files, ~460 KB).
- `app/ingestion/loader.py`: discovers `.md`/`.txt`, confines reads to the corpus dir, skips empty files, UTF-8 with
  Latin-1 fallback, classifies source type and drug name.
- `app/ingestion/chunker.py`: heading-aware chunks (≤600 tokens, 80-token overlap) with the heading breadcrumb,
  table header repetition and exact source line ranges.
- `app/retrieval/`: batched embeddings with response validation → L2-normalised float32 matrix → cosine top-k.
- `app/retrieval/index.py`: the index is rebuilt at startup (~15 s, 9 embedding calls). No on-disk cache, so it can
  never be stale for a different embedding model.

## Agent, tools and critic

- **Tools** (`app/agent/tools.py`) — registry of name → Pydantic argument model → function. JSON schemas for
  function calling are generated from the models. `execute_tool` never raises: unknown tools, bad JSON, invalid
  arguments and tool failures become error observations the LLM can react to.
  - `search_corpus(query, top_k≤8)` — semantic search; returns passages with file, lines, section, score.
  - `read_source_document(filename, section? | start_line/end_line?)` — table of contents, a whole section, or a
    line range. Filenames must match an allow-list of loaded documents; no filesystem path is built from model input.
- **Research loop** (`app/agent/researcher.py`) — the LLM decides every call; Python only validates and executes.
  Limits: iterations, total tool calls, duplicate-call blocking, one nudge if the model tries to answer without evidence.
- **Corpus coverage** — for each drug in the plan, code checks whether the corpus has a dedicated document
  (`CorpusIndex.drug_coverage`). The result is given to the research loop, writer and critic, so "not in the corpus"
  is stated as missing coverage, never as evidence that a drug or effect does not exist.
- **Citations** (`app/rag/evidence.py`) — valid only if the file was retrieved and the range is a retrieved passage
  (or lies inside a line-numbered exact read). Fabricated files and guessed line numbers are rejected.
- **Number/unit check** (`check_numbers`) — every number in a cited sentence must occur in the cited passages (or in
  the question), and a number written with a unit must appear there with the same unit. A detector, not proof.
- **Critic** (`app/agent/critic.py`) — receives the coverage note, evidence, automatic checks and draft; reports
  unsupported claims, citation problems, irrelevant citations, clinical ambiguities (incl. unit/population
  mismatches), omissions that change the answer, and source conflicts, with verdict
  `approve | revise | insufficient_evidence`. Code overrides "approve" on invalid citations, unsupported claims,
  irrelevant citations or number problems; an approved answer without any valid citation becomes
  `insufficient_evidence`; a "revise" without any structured finding is treated as "approve". Citations the final review marks irrelevant are removed from the answer, so unrelated
  documents are never shown as sources. At most one revision; remaining problems appear as reviewer notes.

## API

```bash
curl http://localhost:8000/health
# {"status":"ok","index_ready":true,"documents":21,"chunks":525,"detail":""}

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What are the main contraindications for warfarin?"}'
```

`POST /ask` response:

| Field | Content |
|---|---|
| `answer` | Markdown answer with `[file.md:Lx-Ly]` citations, reviewer notes if any, and the disclaimer |
| `citations` | Validated citations `{source, start_line, end_line, section}` |
| `sources` | Cited files |
| `disclaimer` | The medical disclaimer |
| `status` | `success` (critic approved) · `needs_review` (problems remained after one revision) · `insufficient_evidence` |
| `review` | Final critic review |
| `trace` | Plan, every tool call with arguments, evidence list, all reviews, timing |

Errors: `422` invalid question (3–2000 characters) · `503` index not ready/misconfigured · `502` LLM provider
error · `500` unexpected. Error bodies never contain provider messages or stack traces (those go to the server log).
`/health` always returns `200` with `index_ready`. Interactive docs at `/docs`. No CORS headers are sent: the UI is
served from the same origin, so other web origins cannot call the API from a browser.

## Configuration

Copy `.env.example` to `.env`:

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | |
| `OPENAI_MODEL` | yes | Chat model with tool-calling support |
| `OPENAI_BASE_URL` | no | OpenAI-compatible proxy URL; empty = SDK default |
| `OPENAI_EMBEDDING_MODEL` | no | Default `text-embedding-3-small` |
| `LLM_TIMEOUT_SECONDS`, `MAX_AGENT_ITERATIONS` | no | Defaults 90 and 6 |

Variables already in the environment (e.g. `docker run --env-file`) take precedence over a local `.env`.

## Run with Docker (assessment commands)

```bash
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
# open http://localhost:8000
```

Multi-stage build: Node 20 builds the frontend; `python:3.13-slim` runs uvicorn as a non-root user. The tokenizer
is downloaded at build time; `.env`, tests and local files are excluded by `.dockerignore`.

## Run locally

```bash
git clone <repository-url> pharma-research-agent && cd pharma-research-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # fill in values
(cd frontend && npm ci && npm run build)   # optional: UI at http://localhost:8000
uvicorn app.main:app --port 8000
```

Frontend development with hot reload: `cd frontend && npm run dev` (proxies `/ask` to port 8000).

## Tests

```bash
pytest                         # offline: no credentials, no network
RUN_INTEGRATION=1 pytest -m integration   # live embedding endpoint checks
```

Offline tests use the real corpus and chunker with a deterministic hashed bag-of-words embedder and a scripted
fake chat client, covering: ingestion, chunking, vector math and ranking, tool validation and path traversal,
the agent loop (multi-hop tool choice, iteration/tool budgets, unknown tools, bad JSON, duplicates), planner
fallback, critic approval/rejection/overrides, the single-revision bound, insufficient evidence, citation
validation, citation relevance removal, number/unit fidelity, corpus coverage, out-of-corpus questions and the
HTTP API.

## Evaluation

```bash
python -m eval.run_eval                 # all questions through the agent (live LLM) → new run directory
python -m eval.run_eval --report [RUN]  # re-render a run's RESULTS.md from its JSON (no LLM calls)
```

- `eval/questions.json` — the four public assessment questions (Q1–Q4) with expected sources, plus one
  out-of-corpus control (X1, lisinopril) expected to return `insufficient_evidence` without citations.
- `eval/results/<run_id>/` — complete response per question, `summary.json` (models, limits, git HEAD, code+corpus
  fingerprint) and `RESULTS.md`. Runs are never overwritten; `latest.txt` names the reported run;
  `eval/results/README.md` explains every run.
- `eval/MANUAL_REVIEW.md` — human check of the reported run against the cited source lines.

Automatic checks are deterministic (expected sources cited, citation validity, status). There is no LLM-as-judge score.

Reported run `2026-11-12T11-04-16Z` (gpt-5.6-terra, same code as this tree): Q1–Q4 `success`, all expected
sources cited, 0 invalid citations, 15–40 s per question; X1 `insufficient_evidence` with no citations. Known
weakness in that run: Q3 omits the sertraline protein-binding instruction (see the manual review).

## Known limitations

- Answers vary between runs (LLM non-determinism); the critic does not always catch the same omissions
  (e.g. the Q3 protein-binding instruction is included in some runs and not others).
- The critic is an LLM: it judges meaning but is not ground truth. Only citation validity is checked by code.
- Unit/population differences are handled by prompts and the LLM critic; only the presence of numbers and units
  in the cited passages is checked by code.
- Corpus coverage is decided from filenames (one document per drug); a drug mentioned only inside another
  document (e.g. in an interaction table) is reported as "no dedicated document" although passages may exist.
- Citation granularity is a whole passage unless the model read the exact section.
- Each question costs ~5–12 LLM calls; latency 10–50 s. The index costs ~9 embedding calls per start.
- English-only; corpus limited to 10 drugs. No conversation memory.

## AI coding configuration

This project was built with Claude Code. Its configuration is committed: [`CLAUDE.md`](CLAUDE.md) (project
instructions and conventions) and [`.claude/`](.claude) (shared permission settings and a `verify` skill). No MCP
servers were used. See the presentation outline in [`docs/PRESENTATION.md`](docs/PRESENTATION.md).

## Repository layout

```
app/
  main.py                 FastAPI app (routes only)
  core/                   config, OpenAI client, LLM JSON helpers
  ingestion/              loader, chunker
  retrieval/              embedder, vector store, retriever, index builder
  rag/                    evidence ledger + citation validation, grounded generator
  agent/                  tools, planner, research loop, critic, orchestrator
  schemas/                Pydantic models (documents, API)
  data/corpus/            the 21 source documents
frontend/                 React + Vite UI
eval/                     evaluation questions, runner, results, manual review
tests/                    pytest suite (offline + integration-marked)
docs/                     architecture and presentation
```
