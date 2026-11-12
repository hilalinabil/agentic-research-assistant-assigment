# AI assistance

This project was built with an AI coding assistant (Claude Code), as the assignment allows. The author is
responsible for every line and reviewed the work; this folder records how the assistant was directed.

| File | Content |
|---|---|
| [`master-prompt.md`](master-prompt.md) | The master engineering prompt (full specification, constraints and acceptance checklist) given at the start |
| [`phase-prompts/`](phase-prompts) | The author's phase-by-phase prompts (originally `Pahse0–6.pdf`), each ending with "Stop after Phase N". Phases 0–2 were already implemented when the session in `master-prompt.md` / `project-context.md` started; phases 3–6 were completed in that session: |
| &nbsp;&nbsp;[`phase-0.pdf`](phase-prompts/phase-0.pdf) | Repository inspection and project foundation |
| &nbsp;&nbsp;[`phase-1.pdf`](phase-prompts/phase-1.pdf) | Document ingestion and chunking |
| &nbsp;&nbsp;[`phase-2.pdf`](phase-prompts/phase-2.pdf) | Embeddings and in-memory semantic retrieval |
| &nbsp;&nbsp;[`phase-3.pdf`](phase-prompts/phase-3.pdf) | Basic retrieval-augmented generation |
| &nbsp;&nbsp;[`phase-4.pdf`](phase-prompts/phase-4.pdf) | Agentic research orchestration (planner, tools, native tool calling) |
| &nbsp;&nbsp;[`phase-5.pdf`](phase-prompts/phase-5.pdf) | Critic and evidence verification |
| &nbsp;&nbsp;[`phase-6.pdf`](phase-prompts/phase-6.pdf) | API, frontend, Docker and final delivery |
| [`project-context.md`](project-context.md) | The follow-up instructions, in order: discrepancy check, implementation brief, reliability audit, audit-only verification, resilience/portability/cleanup |
| [`../../CLAUDE.md`](../../CLAUDE.md) | Standing project instructions and rules for the assistant |
| [`../../.claude/settings.json`](../../.claude/settings.json) | Shared permissions (test/build commands allowed; reading `.env` denied) |
| [`../../.claude/skills/verify/SKILL.md`](../../.claude/skills/verify/SKILL.md) | `verify` skill: end-to-end verification procedure (tests, Docker, /health, /ask) |

No MCP servers were used, so there is no `.mcp.json`.
