# AI assistance

This project was built with an AI coding assistant (Claude Code), as the assignment allows. The author is
responsible for every line and reviewed the work; this folder records how the assistant was directed.

| File | Content |
|---|---|
| [`master-prompt.md`](master-prompt.md) | The master engineering prompt (full specification, constraints and acceptance checklist) given at the start |
| [`project-context.md`](project-context.md) | The follow-up instructions, in order: discrepancy check, implementation brief, reliability audit, audit-only verification, resilience/portability/cleanup |
| [`../../CLAUDE.md`](../../CLAUDE.md) | Standing project instructions and rules for the assistant |
| [`../../.claude/settings.json`](../../.claude/settings.json) | Shared permissions (test/build commands allowed; reading `.env` denied) |
| [`../../.claude/skills/verify/SKILL.md`](../../.claude/skills/verify/SKILL.md) | `verify` skill: end-to-end verification procedure (tests, Docker, /health, /ask) |

No MCP servers were used, so there is no `.mcp.json`.
