---
name: verify
description: Verify the pharmacology agent end to end — offline tests, Docker build/run with the assessment commands, /health and a real /ask. Use before claiming the project works or before submission.
---

# Verify the application

Run these steps in order and report actual output. Stop and investigate on the first failure; never report a step
as passing without having run it.

1. Offline tests: `.venv/bin/pytest -q` — expect 0 failures; the only skips must be `integration` tests.
2. Frontend build: `cd frontend && npm ci && npm run build`.
3. Docker build (exact command): `docker build -t agent .`
4. Docker run (exact command, in the background): `docker run -p 8000:8000 --env-file .env agent`
5. `curl -s localhost:8000/health` → HTTP 200; poll until `"index_ready":true` (≈15 s).
6. `curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"What are the main contraindications for warfarin?"}'`
   → HTTP 200, `status`, non-empty `citations` pointing to `dailymed-warfarin.md`, `disclaimer` present.
7. `curl -s -o /dev/null -w '%{http_code}' localhost:8000/` → 200 (frontend served).
8. Confirm `.env` is not inside the image: `docker exec <id> ls -a /app`.
9. Stop the container.

Live LLM calls cost quota: do not loop step 6, and run `python -m eval.run_eval` only when evaluation results must be refreshed.
