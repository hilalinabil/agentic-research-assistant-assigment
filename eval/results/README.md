# Evaluation runs

Each run directory is written once by `python -m eval.run_eval` and never overwritten. It holds the complete
response for every question (`Q*.json`, `X1.json`), `summary.json` (models, agent limits, git HEAD, SHA-256
fingerprint of `app/**/*.py` + corpus) and a rendered `RESULTS.md`. `latest.txt` names the run that is reported.

| Run | Status of code | Purpose |
|---|---|---|
| `2026-11-12T10-31-31Z-superseded` | before the reliability fixes | earlier report's table; see its `NOTE.md` |
| `2026-11-12T10-46-37Z` | intermediate | first run with coverage note, citation relevance, number/unit check; completeness rule too broad → Q3/Q4 `needs_review` with over-long answers |
| `2026-11-12T10-50-24Z` | intermediate | completeness rule narrowed; exposed that a correct citation-free "not in corpus" answer was forced to `needs_review` |
| `2026-11-12T10-52-50Z` | intermediate | zero-citation override fixed; X1 still spent an unnecessary revision |
| `2026-11-12T10-56-00Z` | intermediate (fingerprint `30c5a27f1145515a`) | Q1–Q4 success; Q3 included the protein-binding instruction. Afterwards a live Docker request showed a "revise" verdict with no findings blocking a correct Q1 answer → critic rule added |
| **`2026-11-12T11-04-16Z`** | **final code** (fingerprint `512d5d9ab30a5160`, identical to the current tree) | **reported run** — reviewed in `../MANUAL_REVIEW.md` |

LLM output is not deterministic: re-running the same code can produce different wording, tool sequences and
critic verdicts. That is why every run is kept rather than only the "best" one.
