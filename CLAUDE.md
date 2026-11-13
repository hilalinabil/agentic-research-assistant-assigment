# CLAUDE.md — Agentic Pharmacology Research Assistant

Instructions for AI coding assistants working in this repository.

## Project
FastAPI service + React UI that answers pharmacology questions only from `app/data/corpus/` using an
LLM-driven agent (plan → tool-calling research → cited answer → critic → ≤1 revision). See `README.md` and
`docs/ARCHITECTURE.md`.

## Commands
- Tests (offline, no credentials): `.venv/bin/pytest -q`
- Live integration tests: `RUN_INTEGRATION=1 .venv/bin/pytest -m integration`
- Evaluation (live LLM, costs quota): `.venv/bin/python -m eval.run_eval` · re-render only: `--report`
- Run API: `.venv/bin/uvicorn app.main:app --port 8000`
- Frontend: `cd frontend && npm run build` (or `npm run dev`)
- Docker (assessment commands): `docker build -t agent . && docker run -p 8000:8000 --env-file .env agent`

## Rules
- Never read, print or commit `.env` or any API key. Credentials come only from environment variables.
- Never hardcode model names, answers to evaluation questions, or question-specific tool sequences.
  Tool selection must stay LLM-driven (native function calling); Python only validates and executes.
- Every tool argument from the LLM is validated with Pydantic before execution; tools must not raise to the caller.
- `read_source_document` must never build filesystem paths from model input (allow-list of loaded documents).
- Answers must be grounded in retrieved evidence with `[file.md:Lx-Ly]` citations; the disclaimer is appended by code.
- Keep unit tests offline (use `tests/conftest.py` fakes). Mark tests needing the live API `@pytest.mark.integration`.
- Do not claim tests, Docker or evaluation results without running them. Do not commit unless asked.
- Prefer small, explainable code over frameworks; no LangChain/LangGraph.

## Layout
`app/core` config + client · `app/ingestion` loader/chunker · `app/retrieval` embeddings + NumPy store ·
`app/rag` evidence/citations + generator · `app/agent` tools, planner, researcher, critic, agent ·
`app/main.py` HTTP only · `eval/` questions, runner, results · `tests/`.
