# Evaluation results

`2026-11-12T11-32-54Z/` is the reported evaluation run, written by `python -m eval.run_eval` on the submitted code (code+corpus
fingerprint `e53b19be1dd283d5`, identical to the committed tree; the run's `git_head` reads `fef0588 (uncommitted changes)`
because the only uncommitted changes at the time were documentation and these result files). It holds the
complete response for every question (`Q1–Q4.json`, `X1.json`), `summary.json` (models, agent limits, git HEAD,
fingerprint) and the rendered `RESULTS.md`. `latest.txt` names it.

New runs are written to new directories and never overwrite earlier ones. Earlier development runs were removed
from the working tree; they remain in git history (commits `13552aa`, `fef0588`).

LLM output is not deterministic: re-running the same code can produce different wording, tool sequences and
critic verdicts.
