<!-- Verbatim master prompt given to the AI coding assistant (Claude Code) at the start of the build session.
     The author's own short note that followed the paste is kept at the end. -->

# MASTER ENGINEERING PROMPT
# Syntax Systems — Agentic Pharmacology Research Assistant

## 1. ROLE AND MISSION

You are acting as a Senior Generative AI Engineer, Python Software Architect, RAG Engineer, Agentic AI Developer, and Technical Reviewer.

Your mission is to inspect, validate, complete, test, and prepare the existing repository for the Syntax Systems Generative AI Bootcamp Coding Assignment.

You are not being asked to generate another learning roadmap or a collection of implementation plans.

You are being asked to finish an actual working software engineering project.

Your priorities, in order:

1. Satisfy every mandatory assessment requirement.
2. Preserve and improve the existing implementation.
3. Build a genuinely agentic research assistant.
4. Ensure answers are grounded in the provided pharmacology corpus.
5. Ensure the application works reliably through its Docker interface.
6. Produce professional, understandable, maintainable code.
7. Prepare the project for technical demonstration and interview discussion.

Avoid unnecessary complexity, excessive abstraction, premature optimization, and unnecessary dependencies.

---

# 2. OFFICIAL ASSIGNMENT SPECIFICATION

The project is an Agentic Pharmacology Research Assistant.

It must answer compound questions using a provided local corpus consisting of approximately 20 Markdown and text documents derived from public FDA drug labels (DailyMed) and consumer medication information.

The corpus contains information such as:

- Indications.
- Dosage.
- Contraindications.
- Drug interactions.
- Mechanism of action.
- Other relevant medication information.

A GLOSSARY.md file is provided.

The application must demonstrate five fundamental capabilities:

1. PLAN: Decompose user questions into sub-questions or reasoning steps.
2. RETRIEVE: Fetch relevant passages using in-memory similarity.
3. USE TOOLS: Invoke at least two local tools, with tool selection driven by the agent rather than hardcoded.
4. REVIEW: Use a critic step to verify grounding and flag unsupported or clinically ambiguous claims.
5. ANSWER: Produce a cited response referencing source files, with an explicit disclaimer that the output is not medical advice.

The assignment permits any agent orchestration framework, including:

- LangChain.
- LangGraph.
- PydanticAI.
- A custom agentic loop.

Python 3.11+ is required.

Dependencies must be pinned in pyproject.toml or requirements.txt.

The mandatory interface consists of:

- A Dockerfile at the repository root.
- GET /health returning HTTP 200.
- POST /ask accepting {"question": "..."} and returning HTTP 200 for valid questions.
- A minimal frontend that calls POST /ask and displays the answer.

The evaluator must be able to execute:

docker build -t agent .
docker run -p 8000:8000 --env-file .env agent

The container must start the API and be self-contained.

The evaluator will inject its own environment variables and credentials.

The project must be published in a public Git repository, preferably GitLab.

A 10-minute presentation must demonstrate the problem, technologies, architecture, workflow, challenges, and technical decisions.

AI coding assistance is explicitly permitted. However, the candidate is accountable for every line of code and may be asked to explain any implementation decision.

AI coding configuration, including relevant skills and MCP configuration files, must also be pushed to the repository.

---

# 3. EXISTING PROJECT CONTEXT — PRESERVE IT

The repository already contains implementation work.

Previously completed work:

- Phase 0: Project initialization.
- Phase 1: Document ingestion and chunking.
- Phase 2: Embedding generation and in-memory vector retrieval.

The latest reported automated test results were:

49 passed, 1 skipped, 0 reported regressions.

This is historical information, not proof of current correctness.

Your first responsibility is to independently inspect and validate the implementation.

IMPORTANT RULES:

- Do not restart the project.
- Do not delete existing working functionality.
- Do not regenerate existing modules unnecessarily.
- Do not replace NumPy retrieval with a vector database without a compelling requirement.
- Do not regenerate embeddings unless they are missing, invalid, or incompatible.
- Do not change working architecture merely to introduce a framework.
- Do not duplicate existing functionality.
- Preserve existing naming conventions and project organization where reasonable.

The existing implementation is the foundation of the final application.

---

# 4. STAGE ONE — REPOSITORY AUDIT

Start by inspecting the entire repository.

Read:

- Project structure.
- README.
- Assessment PDF, if available.
- pyproject.toml or requirements.txt.
- Configuration files.
- Environment variable examples.
- Existing application modules.
- Ingestion and chunking implementation.
- Embedding implementation.
- Vector retrieval implementation.
- Existing tests.
- Frontend implementation.
- Docker configuration.

Produce an internal technical understanding of the repository before changing files.

Identify:

1. What is already implemented correctly.
2. What is partially implemented.
3. What is missing.
4. What is broken.
5. What is unnecessarily complicated.
6. What must be modified to satisfy the assessment.

Inspect the existing implementation rather than trusting previous completion reports.

## Phase 1 verification

Verify:

- All expected source documents are discovered.
- Document parsing works.
- Markdown and text files are handled correctly.
- Chunking preserves useful semantic boundaries.
- Chunk metadata is preserved.
- Source filenames remain traceable.
- Empty and malformed documents are handled appropriately.
- GLOSSARY.md is available to the retrieval system where appropriate.

## Phase 2 verification

Verify:

- Embeddings are generated successfully.
- Query and document embedding dimensions match.
- The intended embedding model is used consistently.
- Embeddings are stored and associated with their original chunks.
- Vector similarity is calculated correctly.
- Cosine similarity is implemented correctly.
- Top-K retrieval works.
- Results are ordered correctly.
- Source metadata is preserved.
- Empty queries and invalid parameters are handled.
- Existing embeddings can be reused without unnecessary API calls.

Run the existing test suite.

Investigate the skipped test and explain whether the skip is justified.

Perform actual retrieval tests using questions based on the provided corpus.

Do not move forward with a broken retrieval foundation.

---

# 5. STAGE TWO — COMPLETE THE RAG PIPELINE

Implement or complete the complete Retrieval-Augmented Generation pipeline.

Expected flow:

User Question
    |
    v
Query Processing
    |
    v
Query Embedding
    |
    v
In-Memory Vector Retrieval
    |
    v
Relevant Passages
    |
    v
Context Construction
    |
    v
LLM Generation
    |
    v
Grounded Answer

Requirements:

### Query processing

- Accept natural-language questions.
- Preserve important drug names and clinical terminology.
- Support compound questions.
- Avoid unnecessarily rewriting user intent.

### Context construction

- Build context from retrieved passages.
- Preserve source filename, chunk identifier, and relevant metadata.
- Avoid injecting irrelevant documents.
- Handle context limits intelligently.
- Avoid duplicate evidence where possible.

### Generation

- Use the configured LLM provider.
- Instruct the LLM to answer using supplied evidence.
- Require traceable source references.
- Distinguish explicitly between supported information and missing evidence.
- Do not invent clinical facts.
- Do not make assumptions that are not supported by the corpus.

### Citations

Every important factual claim should be traceable to the source document or documents supporting it.

Citations must reference actual local source files.

Do not fabricate filenames, sections, or quotations.

The final answer should have a readable structure.

---

# 6. STAGE THREE — IMPLEMENT THE AGENTIC WORKFLOW

This is the central requirement of the assessment.

Do not implement a conventional RAG pipeline and simply label it an agent.

Build a genuinely agentic system.

## 6.1 Planning

The agent must understand the user question and decide whether it requires multiple steps.

For example:

Question:
"A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?"

Expected planning behavior:

- Identify warfarin.
- Identify sertraline.
- Determine the relevant interaction question.
- Retrieve evidence about warfarin.
- Retrieve evidence about sertraline.
- Identify relevant bleeding-risk evidence.
- Identify monitoring recommendations.
- Synthesize the findings.

Planning should be adaptive.

Do not hardcode the above sequence for this particular question.

Use the LLM to determine the sub-questions based on the actual user input.

## 6.2 Tool registry

Implement a clean tool registration and execution mechanism.

At least two local tools are mandatory.

Recommended tools:

TOOL 1: search_corpus

Purpose:
Search the local in-memory vector index using semantic similarity.

Suggested input:
- query
- top_k

Expected output:
- Relevant text passages.
- Source filenames.
- Chunk identifiers.
- Similarity scores.
- Relevant metadata.

TOOL 2: read_source_document

Purpose:
Read an original local source document and retrieve its relevant contents.

Suggested input:
- filename
- optional section or search term

Expected output:
- Original document content or relevant excerpts.
- Source filename.
- Section information where available.

Security requirements:

- Restrict document reading to the configured corpus directory.
- Prevent path traversal.
- Reject unauthorized paths.
- Handle missing files safely.

You may introduce additional tools if they add real value, but do not introduce unnecessary tools just to increase the tool count.

## 6.3 Dynamic tool selection

This requirement is particularly important.

The agent must select tools based on the question and its current observations.

Use the configured LLM's native tool/function-calling capability through an OpenAI-compatible SDK.

The execution mechanism should:

1. Send the user request and available tool definitions to the LLM.
2. Receive the LLM's selected tool call.
3. Validate the tool name and arguments.
4. Execute the corresponding local Python function.
5. Return the observation to the LLM.
6. Allow the LLM to decide whether another tool is necessary.
7. Continue until a final answer is produced or a safe execution limit is reached.

Do not hardcode a fixed sequence of tool calls.

Do not create a fake tool-selection mechanism based on Python if/else statements matching particular question keywords.

Python should execute the tools selected by the agent, not decide the research plan through hardcoded question-specific logic.

## 6.4 Execution controls

Implement:

- Maximum agent iterations.
- Tool argument validation.
- Unknown-tool handling.
- Tool execution error handling.
- Logging of tool decisions.
- Protection against infinite loops.
- Appropriate handling of empty observations.

Keep the orchestration understandable and easy to demonstrate.

---

# 7. STAGE FOUR — CRITIC AND EVIDENCE VALIDATION

The critic is a mandatory assessment requirement.

Implement a distinct review step before the final answer is returned.

The critic must evaluate the candidate answer against the evidence gathered by the agent.

Review dimensions:

1. Evidence grounding.
2. Citation correctness.
3. Unsupported claims.
4. Clinical ambiguity.
5. Missing information.
6. Completeness of the original question.
7. Consistency between different source documents.

The critic must be able to:

- Approve a sufficiently grounded answer.
- Request a revision when necessary.
- Identify unsupported claims.
- Flag clinically ambiguous statements.
- Recommend an insufficient-evidence response.

The critic should not blindly approve an answer simply because it contains citations.

Implement a bounded review mechanism.

Avoid unnecessary repeated LLM calls.

Do not create an endless generate-review-regenerate loop.

If evidence is insufficient, the system must communicate that limitation rather than manufacture an answer.

Keep the critic's output structured and inspectable.

---

# 8. STAGE FIVE — PHARMACOLOGY SAFETY AND GROUNDING

The project concerns medication information.

It is a technical research assistant, not a medical diagnosis or treatment system.

Implement appropriate safeguards.

Requirements:

- Do not fabricate drug interactions.
- Do not invent contraindications.
- Do not generate unsupported dosage instructions.
- Do not confuse missing information with evidence that a condition is absent.
- Preserve clinically important qualifications and thresholds.
- Clearly distinguish evidence from interpretation.
- Cite all relevant supporting documents.
- Flag contradictions or ambiguities between sources.
- Explicitly state when the corpus does not contain sufficient information.

Every final response must include an explicit disclaimer:

"This information is provided for research and educational purposes only. It is not medical advice and should not replace consultation with a qualified healthcare professional."

The disclaimer must not be used as an excuse for generating unsupported medical information.

---

# 9. STAGE SIX — EVALUATION

The assignment describes five evaluation questions in its overview.

The detailed evaluation table explicitly displays four public questions and states that hidden questions are held back by reviewers.

Do not assume these are the complete same set.

Inspect all supplied files and identify every evaluation question actually available.

The four explicitly listed public questions are:

QUESTION 1 — WARFARIN CONTRAINDICATIONS

"What are the main contraindications for warfarin?"

Expected source:
dailymed-warfarin.md

QUESTION 2 — WARFARIN VS APIXABAN

"Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?"

Expected sources:
dailymed-warfarin.md
dailymed-apixaban.md

QUESTION 3 — WARFARIN AND SERTRALINE

"A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?"

Expected sources:
dailymed-warfarin.md
dailymed-sertraline.md

QUESTION 4 — METFORMIN AND RENAL FUNCTION

"A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?"

Expected source:
dailymed-metformin.md

Use the exact wording available in the provided evaluation files where applicable.

Do not invent answers or assume expected clinical conclusions.

Retrieve and inspect the actual corpus evidence.

For each evaluation question:

- Run the full application pipeline.
- Inspect the agent's plan.
- Inspect the tools selected.
- Inspect retrieved passages.
- Inspect the final answer.
- Inspect citations.
- Inspect critic decisions.

Create a reproducible evaluation runner.

Assess:

- Retrieval relevance.
- Answer correctness against source evidence.
- Citation validity.
- Completeness.
- Multi-document evidence coverage.
- Multi-hop reasoning.
- Unsupported-claim resistance.

Produce a results report with actual outputs.

Do not manufacture numerical evaluation scores.

If using an LLM-as-judge, explain its limitations and do not treat it as objective ground truth.

Include representative failure cases where useful.

---

# 10. STAGE SEVEN — FASTAPI INTERFACE

Implement the mandatory HTTP API.

Required:

GET /health

Expected:
HTTP 200.

POST /ask

Expected request:

{
  "question": "What are the main contraindications for warfarin?"
}

Expected:
HTTP 200 for valid requests.

The response should contain an understandable answer and source citations.

Recommended response structure:

{
  "answer": "...",
  "citations": [],
  "disclaimer": "...",
  "status": "success"
}

Adapt this schema to existing implementation where appropriate.

Requirements:

- Pydantic request/response validation.
- Clear HTTP error handling.
- Appropriate status codes.
- Input validation.
- Logging.
- Separation between HTTP controllers and agent orchestration.
- No direct business logic inside API route handlers.

Do not expose internal secrets or unnecessary implementation details.

The API must remain functional when accessed from the Docker container.

---

# 11. STAGE EIGHT — MINIMAL FRONTEND

A minimal frontend is mandatory.

Inspect the existing frontend before implementing another one.

The frontend must:

- Allow users to submit a question.
- Send it to POST /ask.
- Display the generated answer.
- Display citations and source filenames.
- Display the medical disclaimer.
- Handle loading states.
- Handle API errors.
- Be usable without requiring technical knowledge.

Do not waste time creating an elaborate dashboard.

A simple, professional interface is sufficient.

Prioritize a seamless demonstration.

---

# 12. STAGE NINE — DOCKER AND REPRODUCIBLE EXECUTION

A Dockerfile at the repository root is mandatory.

The exact evaluator workflow must work:

docker build -t agent .

docker run -p 8000:8000 --env-file .env agent

Requirements:

- Python 3.11+.
- Pinned dependencies.
- Correct application startup.
- Correct working directory.
- Required corpus files included in the image.
- Required local indexes/artifacts available.
- Environment variables read at runtime.
- No hardcoded credentials.
- Appropriate port exposure.
- Reliable API startup.
- Self-contained application runtime.

The evaluator will inject its own .env.

Use the provided OpenAI-compatible SDK configuration.

The supplied integration example uses:

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

Embedding model configuration:

OPENAI_EMBEDDING_MODEL

Default example:
text-embedding-3-small

Chat model configuration:

OPENAI_MODEL

The evaluator will provide the allowed model names and proxy configuration.

Read the API key and base URL from the environment.

Do not hardcode a particular model name if the environment already provides it.

Do not assume your personal API credentials will be available during evaluation.

Verify that the Docker image can build without access to personal local files outside the repository.

Also verify that the container can start with the evaluator's configuration and access the bundled corpus.

---

# 13. STAGE TEN — CONFIGURATION AND SECURITY

Implement clean configuration management.

Use environment variables for:

- OPENAI_API_KEY.
- OPENAI_BASE_URL.
- OPENAI_MODEL.
- OPENAI_EMBEDDING_MODEL.
- Other necessary configuration.

Use an appropriate .env.example.

Never commit actual credentials.

Ensure .gitignore excludes:

- .env.
- Virtual environments.
- Python cache files.
- Temporary artifacts.
- Sensitive local files.
- Unnecessary generated data.

Verify document-reading tools cannot access arbitrary filesystem locations.

Validate all LLM-generated tool arguments before execution.

Keep logs useful without exposing credentials or sensitive information.

---

# 14. STAGE ELEVEN — TESTING AND FINAL VERIFICATION

Run the complete test suite.

Preserve the existing tests.

Add missing tests where necessary.

Cover:

- Ingestion.
- Chunking.
- Embeddings.
- Vector retrieval.
- Similarity ranking.
- RAG generation.
- Planning.
- Dynamic tool selection.
- Tool execution.
- Invalid tool arguments.
- Tool execution failures.
- Critic approval.
- Critic rejection.
- Revision limits.
- Citation validity.
- Insufficient evidence.
- API endpoints.
- Docker startup.

Distinguish unit tests from tests requiring external LLM access.

Use mocks or deterministic fixtures for local tests wherever appropriate.

Do not require a paid API call for every unit test.

Do not silently remove, weaken, or skip tests to obtain a passing test report.

Perform a final Docker verification using the exact assignment commands.

Test both:

- GET /health.
- POST /ask with a real corpus question.

---

# 15. STAGE TWELVE — DOCUMENTATION AND PRESENTATION

Prepare a professional repository.

README.md must explain:

1. Project overview.
2. Problem statement.
3. Functional requirements.
4. Architecture.
5. Technologies and rationale.
6. Corpus structure.
7. Ingestion and chunking.
8. Embeddings and retrieval.
9. RAG workflow.
10. Agent planning and tool calling.
11. Critic mechanism.
12. Evidence grounding and citations.
13. API usage.
14. Frontend.
15. Environment configuration.
16. Local execution.
17. Docker execution.
18. Testing.
19. Evaluation methodology.
20. Known limitations.

Include architecture and workflow diagrams where useful.

## Ten-minute technical presentation

Prepare a presentation outline and supporting material.

Suggested structure:

Minute 0–1:
Problem and objectives.

Minute 1–2:
Technologies and rationale.

Minute 2–4:
Architecture and data ingestion.

Minute 4–6:
RAG and agentic workflow.

Minute 6–7:
Dynamic tool selection and critic.

Minute 7–8:
Evaluation questions and results.

Minute 8–9:
Live demonstration.

Minute 9–10:
Challenges, engineering decisions, and limitations.

The presentation must describe the actual implemented system.

---

# 16. AI CODING CONFIGURATION

The assessment explicitly requires relevant AI coding configuration to be pushed to the repository.

Inspect the current coding-agent environment.

Preserve and document relevant:

- Agent instructions.
- Project-specific skills.
- MCP configuration.
- Development conventions.
- Useful coding-agent configuration.

Do not commit API keys, access tokens, or private credentials.

Only include configuration files that are safe to publish.

---

# 17. EXECUTION RULES — IMPORTANT

You have permission to inspect and modify the existing repository, implement missing functionality, run tests, and prepare the deliverables.

Do not stop after writing a plan.

Do not create another collection of phase-specific prompts.

Do not ask me to approve every routine operation.

Use your engineering judgment.

However:

- Never delete existing working functionality without justification.
- Never invent test results.
- Never fabricate evaluation outcomes.
- Never claim dynamic tool selection if it is hardcoded.
- Never claim citations are correct without verifying their sources.
- Never claim Docker works without actually testing it.
- Never introduce unnecessary complexity.
- Never consume external API quota for repetitive or unnecessary checks.

If a requirement is already satisfied, verify it and leave it alone.

If a requirement is partially satisfied, improve the existing implementation.

If a requirement is missing, implement it.

If an external API key is unavailable, finish everything possible locally and explicitly report the limitation.

When an architectural decision must be made, choose the simplest technically sound solution that satisfies the assessment.

Prioritize completing the actual application over generating additional documentation about work that has not been implemented.

---

# 18. FINAL ACCEPTANCE CHECKLIST

The project is ready only when the following requirements have been verified:

[ ] Existing Phase 1 implementation works.
[ ] Existing Phase 2 implementation works.
[ ] Embedding and retrieval are correct.
[ ] In-memory similarity retrieval is implemented.
[ ] Basic RAG generation works.
[ ] Compound questions are decomposed.
[ ] Agent-driven planning works.
[ ] At least two local tools are implemented.
[ ] Tool selection is dynamically driven by the agent.
[ ] Tool execution is safe and validated.
[ ] Critic reviews evidence and claims.
[ ] Critic can reject or request revision.
[ ] Unsupported claims are handled safely.
[ ] Answers contain real source citations.
[ ] Medical disclaimer is present.
[ ] Evaluation questions are tested.
[ ] Multi-document reasoning works.
[ ] FastAPI GET /health works.
[ ] FastAPI POST /ask works.
[ ] Minimal frontend works.
[ ] Dockerfile exists at repository root.
[ ] Exact Docker build command succeeds.
[ ] Exact Docker run command succeeds.
[ ] Evaluator-provided environment variables are supported.
[ ] Full tests pass, with justified skips.
[ ] README is complete.
[ ] Architecture documentation exists.
[ ] Ten-minute presentation material exists.
[ ] Safe AI coding configuration is included.
[ ] No secrets are committed.
[ ] Public repository is ready for submission.

---

# 19. FINAL ENGINEERING REPORT

After implementation and verification, provide:

## A. Implementation summary
What existed, what was changed, and what was newly implemented.

## B. Architecture
Explain the final architecture and why it was selected.

## C. Assessment compliance
For every official requirement, report:

- COMPLETED AND VERIFIED
- IMPLEMENTED BUT NOT FULLY VERIFIED
- PARTIALLY COMPLETED
- NOT COMPLETED

## D. Testing
Provide actual test results and explain skipped tests.

## E. Evaluation
Show actual outputs for the supplied evaluation questions, including evidence and citations.

## F. Docker
Provide the exact commands tested and their results.

## G. Important files
Identify the important source files and explain their responsibilities.

## H. Limitations
Explicitly identify any remaining technical or external blockers.

## I. Demonstration
Provide the exact steps to demonstrate the application during the interview.

## J. Interview preparation
Provide a concise explanation of the most important architectural decisions and the files I should understand thoroughly.

---

# FINAL DIRECTIVE

Build a small, reliable, genuinely agentic pharmacology research assistant that satisfies the Syntax Systems assessment.

The objective is not to impress through unnecessary complexity.

The objective is to demonstrate real Generative AI engineering competence through:

- Correct Python engineering.
- Grounded RAG.
- Dynamic agentic tool use.
- Evidence-based reasoning.
- Critic validation.
- Reproducible evaluation.
- Reliable deployment.
- Clear architectural decisions.

Preserve existing work, complete missing functionality, verify everything, and prepare the repository for successful evaluation and technical discussion.

Do not finish by generating another plan.

Finish by delivering the working project.


 this is the master prompt of this project many feature are implemented can u analyze it understand the current one do execute ntg, just generate a full repport abt wt is done and not done according to the prompt
