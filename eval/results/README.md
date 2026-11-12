# Evaluation results

`2026-11-12T11-04-16Z/` is the reported evaluation run, written by `python -m eval.run_eval`. It holds the complete response for
every question (`Q1–Q4.json`, `X1.json`), `summary.json` (models, agent limits, git HEAD, SHA-256 fingerprint of
`app/**/*.py` + corpus) and the rendered `RESULTS.md`. `latest.txt` names it. New runs are written to new
directories and never overwrite earlier ones. Intermediate runs from development were removed from the working
tree; they remain in git history (commit `13552aa`).

Fingerprint of the reported run: `512d5d9ab30a5160`. Code changed after the run, without re-running the
evaluation (to limit API usage):
- `app/main.py`, `app/core/config.py`, `app/schemas/api.py`: startup index build retried with backoff, on-demand
  rebuild from `/ask`, build status in `/health`. Not on the answering code path.
- `app/agent/researcher.py`: the research call no longer sends `tool_choice="auto"` (the API default when tools are
  given), and falls back to plain retrieval if a provider rejects the tool-calling request. With a provider that
  supports tools, requests are otherwise identical.

LLM output is not deterministic: re-running the same code can produce different wording, tool sequences and
critic verdicts.
