# Project context: follow-up instructions
Verbatim instructions given to the AI coding assistant after the master prompt, in order. The assistant's work and results for each are recorded in the repository (code, tests, `eval/`, docs).

---

## 1. Discrepancy check, then implement remaining requirements

I reviewed your latest implementation audit. Before proceeding, I need you to resolve an important discrepancy.

Your previous progress report claimed that the agent orchestration, dynamic tool calling, critic, and additional tests were implemented. This latest audit says that only `app/agent/tools.py` exists and that the entire agent layer is missing.

**First: verify the actual repository state.**
- Inspect the current working directory, Git branch, Git status, and actual source files.
- Confirm whether `agent.py`, the tool registry, LLM function-calling implementation, critic, and associated tests exist.
- Do not rely on previous progress reports or assumptions.
- Give me a short, factual explanation of the discrepancy.

**Then proceed with implementation.**

I do not want another roadmap or an extensive planning document. I want you to implement the remaining assessment requirements.

Priority order:

1. Fix the critical path traversal vulnerability in `read_source_document`.
2. Fix unsafe embedding caching and configuration validation.
3. Fix the broken/skipped tests and make the test suite independent of live API credentials.
4. Implement the complete agentic workflow:
   - LLM-driven question decomposition and planning.
   - Dynamic tool selection using actual provider-supported function/tool calling.
   - Safe tool argument validation and execution.
   - Bounded orchestration loop.
   - Evidence-based answer generation with source citations.
   - Critic that checks actual evidence and flags unsupported claims.
   - At most one controlled revision.
5. Implement FastAPI `/health` and `/ask`.
6. Implement a minimal static HTML/CSS/JavaScript frontend.
7. Implement the root Dockerfile and verify the exact assessment commands.
8. Implement an evaluation runner for the four public questions.
9. Complete README, architecture documentation, AI coding configuration, and presentation materials.

Important engineering constraints:
- Preserve working ingestion and retrieval functionality.
- Avoid unnecessary dependencies and abstractions.
- Do not use keyword matching as a substitute for evidence verification.
- Do not hardcode API credentials, provider-specific assumptions, or model names.
- Validate all LLM-generated tool arguments.
- Add meaningful automated tests, including multi-hop questions.
- Do not claim a feature is complete unless its implementation and tests have been verified.

**Work through the implementation without waiting for further approval.** At the end, provide a factual implementation report showing completed files, tests actually executed and their results, remaining limitations, and exact Docker verification results. If something cannot be verified, explicitly say so.


 this is the context of the project can u tell me whcih phase is complted so far and wt is not done yet

---

## 2. Authorization and phase-by-phase implementation brief

You have my authorization to proceed with implementation.

The assessment deadline is November 13, so I want you to complete the remaining project requirements in a focused and efficient manner.

Use the master prompt and project context as the authoritative specification.

### Implementation instructions

**1. Stabilize the existing foundation**
- Fix the path traversal vulnerability in `read_source_document`.
- Review and fix the embedding cache implementation. Prefer rebuilding the small in-memory vector index at startup rather than introducing unnecessary persistent caching.
- Fix the broken skipped test.
- Make tests independent of real API credentials.
- Validate the configuration, including the chat model.
- Run the existing test suite and establish a verified baseline.

**2. Implement Phase 3 — Basic RAG**
- Query embedding.
- Semantic retrieval.
- Evidence/context construction.
- LLM answer generation.
- Grounded citations.
- Insufficient-evidence handling.
- Medical disclaimer.

**3. Implement Phase 4 — Agentic orchestration**
- Question decomposition and concise planning.
- Native LLM tool/function calling.
- Dynamic selection between the available local tools.
- Tool registry and argument validation.
- Safe execution and observation handling.
- Bounded agent loop.
- Proper error handling.

**4. Implement Phase 5 — Critic**
- Validate source filenames and citation line ranges.
- Review factual claims against actual retrieved evidence.
- Detect unsupported claims and clinical ambiguity.
- Request a controlled revision or abstain when evidence is insufficient.
- Do not use simplistic keyword matching as proof of factual grounding.

**5. Implement Phase 6 — API, frontend and Docker**
- FastAPI `/health` and `/ask`.
- React + Vite frontend as specified in the project context.
- Dockerfile supporting the exact required commands.
- Ensure the frontend is accessible through the backend/container.
- Verify the application from a clean container environment.

**6. Implement Phase 7 — Evaluation and delivery**
- Evaluation runner for all four public questions.
- Record actual answers, citations and results.
- README with setup, architecture and execution instructions.
- Architecture diagram and technical presentation material.
- Required AI coding configuration.
- Ensure no credentials, local environments or generated secrets are committed.

### Engineering constraints

- Preserve existing working Phase 1 and Phase 2 code.
- Reuse existing components instead of duplicating functionality.
- Do not introduce LangChain or LangGraph.
- Do not hardcode answers to evaluation questions.
- Do not assume the LLM provider supports a feature without testing it.
- Keep the implementation simple enough for me to explain during the assessment.
- Do not commit automatically.
- Do not silently ignore failing tests.
- Do not claim Docker, API or evaluation success without actually executing and verifying them.

### Working approach

You may implement the remaining phases sequentially without waiting for approval after each phase.

However, maintain phase boundaries internally: finish and test each phase before building the next one.

If a problem blocks implementation, investigate and resolve it rather than stopping to request approval for routine engineering decisions.

### Final report

At completion, provide:

1. Actual files created and modified.
2. Implemented features mapped to assessment requirements.
3. Test results, including failures and skips.
4. Results of the four evaluation questions.
5. Exact Docker build/run verification.
6. Remaining limitations or unverified functionality.
7. Git status and instructions for preparing the public repository.

Prioritize a complete, working, demonstrable application over unnecessary architectural sophistication.

---

## 3. Final reliability and submission audit

The implementation report looks good. Before we publish the project, I want one final reliability and submission audit.

Do not rewrite the architecture or introduce unnecessary features.

### 1. Fix out-of-corpus citation behavior

The report mentions that questions about drugs absent from the corpus can still receive unrelated citations.

Investigate the complete retrieval → evidence → generation → critic workflow.

Ensure that:
- Insufficient evidence is represented explicitly.
- Unrelated documents cannot be presented as supporting evidence.
- Citations are included only when they support relevant claims.
- The system distinguishes missing corpus coverage from evidence that a drug or fact does not exist.
- Add regression tests for out-of-corpus questions.

### 2. Investigate medical answer consistency

Review the Q4 eGFR unit inconsistency and the Q3 omitted sertraline information.

Do not hardcode answers to the evaluation questions.

Instead:
- Inspect the original source passages.
- Verify how evidence is passed to the generator and critic.
- Ensure units and clinical thresholds are preserved accurately.
- Ensure important evidence is not silently omitted.
- Add tests for citation support and preservation of units.

### 3. Reconcile evaluation reporting

The report says Q3 triggered a critic revision, but the evaluation table says it did not.

Determine whether these refer to different runs. Make the evaluation results reproducible and document the actual run being reported.

Run all four public questions again and preserve their complete outputs.

### 4. Frontend verification

Inspect the React implementation and test the complete user interaction if browser testing is available.

At minimum verify:
- Successful question submission.
- Loading state.
- Answer rendering.
- Citation display.
- Insufficient-evidence response.
- API error handling.

If browser interaction is unavailable, explicitly document that limitation.

### 5. Final security and submission audit

Review:
- Path traversal protection.
- LLM-generated tool argument validation.
- API input validation.
- CORS configuration.
- Docker non-root execution.
- Secret handling and `.gitignore`.
- Git diff for accidental files, credentials or generated artifacts.
- README setup instructions.

Do not commit or push anything.

### Final deliverable

Provide:
1. Issues found and fixes applied.
2. Files modified.
3. Test results after fixes.
4. Updated evaluation results.
5. Docker verification results.
6. Remaining limitations.
7. Exact commands I should execute to review, commit and publish the repository.

Do not claim anything is verified unless you actually tested it.

---

## 4. Audit-only verification request

Stop implementation work. Produce an audit report only — do not modify code.

Verify by running commands, not by reading code alone. Paste real output.

1. Tests: run `pytest -v`. Paste the summary. Explain any skipped or failing test.
2. Embeddings: where are they stored? Are the embedding model name and dimension
   saved with them? What happens if the cache is missing, or OPENAI_EMBEDDING_MODEL
   differs from the one used to build it (graders inject their own .env)?
3. Startup: what happens if the API key is invalid or the proxy is unreachable?
   Does GET /health still return 200?
4. Agent loop: is tool selection driven by the LLM via tool calling, or hardcoded?
   Show a real trace (plan, tools called with arguments, critic result) for one question.
5. Evaluation: for each of these exact questions, list retrieved source files vs expected:
   - "What are the main contraindications for warfarin?" → dailymed-warfarin.md
   - "Compare the contraindications of warfarin and apixaban. What conditions are shared
     and which are unique to each drug?" → dailymed-warfarin.md, dailymed-apixaban.md
   - "A patient currently on warfarin is prescribed sertraline. What is the risk of
     bleeding and what monitoring is recommended?" → dailymed-warfarin.md, dailymed-sertraline.md
   - "A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin
     contraindicated at this level of renal function, and what is the threshold?"
     → dailymed-metformin.md
   If the full agent isn't built yet, report retrieval results only. Limit API calls.
6. Docker: was `docker build -t agent .` and `docker run -p 8000:8000 --env-file .env agent`
   actually run? Paste output, or say "not run".
7. Git: is .env ignored? Are dependencies pinned? Are AI coding config files
   (CLAUDE.md, prompts, .mcp.json) committed?
8. Checklist: mark each item COMPLETED / PARTIAL / NOT DONE:
   Plan, Retrieve (in-memory similarity), ≥2 tools with agent-driven choice,
   Critic (grounding + clinical ambiguity flags), Cited answer + not-medical-advice
   disclaimer, Dockerfile at repo root, GET /health → 200, POST /ask → 200,
   Minimal frontend calling /ask, Pinned deps, AI config in repo, README.
9. List known bugs, risks and technical debt.

Be concise. Never claim something works unless you ran it.

---

## 5. Resilience, portability, cleanup and commit

Audit accepted. Do the following, in order, then report briefly with real command output.

1. Check `git log --all --full-history -- .env` prints nothing (if .env was ever
   committed, stop and tell me). Confirm .env is in .gitignore. Commit all current work.
2. Index build resilience: if the startup index build fails, retry in the background
   with backoff (e.g. 5 attempts), and also attempt a rebuild on /ask if the index is
   still not ready. /health stays 200 and reports status. Add tests (mock the embedder).
3. Model portability: graders may set a different OPENAI_MODEL. Grep for model-specific
   params (temperature, max_tokens/max_completion_tokens, response_format, reasoning_effort,
   parallel_tool_calls). Make sure an unsupported param can't crash a request; keep the
   change minimal. Explain what you changed.
4. Run exactly 3 live questions through /ask and show status, cited files, and the
   first lines of each answer:
   - "What are the contraindications of Eliquis?"
   - "What are the contraindications of ibuprofen?" (or any drug not in the corpus)
   - "What is the mechanism of action of sertraline?"
5. Cleanup: replace the README <repository-url> placeholder with a TODO note I'll fill
   in, keep only the final eval run in eval/results/, add my project context and master
   prompt files under docs/ai-assistance/.
6. Run full pytest, then commit. Do not push.
