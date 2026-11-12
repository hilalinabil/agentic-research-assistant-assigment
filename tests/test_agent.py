"""Phases 4-5: planning, LLM-driven tool loop, critic and bounded revision (offline, scripted LLM)."""
import json

import pytest

from app.agent.agent import ResearchAgent
from app.agent.critic import review_answer
from app.agent.planner import make_plan
from app.agent.researcher import run_research
from app.agent.tools import ToolContext
from app.agent.planner import Plan
from app.rag.evidence import Evidence, EvidenceLedger, validate_citations
from app.rag.generator import DISCLAIMER
from tests.conftest import FakeLLMClient, completion, tool_call

Q_MULTI = ("A patient currently on warfarin is prescribed sertraline. "
           "What is the risk of bleeding and what monitoring is recommended?")


def plan_json(subs, drugs=("warfarin",)):
    return completion(json.dumps({"drugs": list(drugs), "sub_questions": subs, "is_compound": len(subs) > 1}))


def review_json(verdict, **kw):
    return completion(json.dumps({"verdict": verdict, "unsupported_claims": [], "citation_problems": [],
                                  "clinical_ambiguities": [], "missing_aspects": [], "source_conflicts": [],
                                  "feedback": "", **kw}))


# --- planner ------------------------------------------------------------------

def test_planner_decomposes_compound_question(test_settings):
    client = FakeLLMClient([plan_json(["Bleeding risk of warfarin with sertraline?", "Recommended monitoring?"],
                                      drugs=("warfarin", "sertraline"))])
    plan = make_plan(client, test_settings, Q_MULTI)
    assert plan.sub_questions == ["Bleeding risk of warfarin with sertraline?", "Recommended monitoring?"]
    assert plan.drugs == ["warfarin", "sertraline"] and plan.is_compound and not plan.fallback
    assert client.requests[0]["messages"][1]["content"] == Q_MULTI   # question passed through verbatim


def test_planner_falls_back_on_malformed_output(test_settings):
    plan = make_plan(FakeLLMClient([completion("I think you should search for warfarin.")]), test_settings, "q?")
    assert plan.fallback and plan.sub_questions == ["q?"]


def test_planner_caps_sub_questions(test_settings):
    plan = make_plan(FakeLLMClient([plan_json([f"s{i}" for i in range(9)])]), test_settings, "q?")
    assert len(plan.sub_questions) == 5


# --- research loop ------------------------------------------------------------

@pytest.fixture
def ctx(corpus_index, test_settings):
    return ToolContext(index=corpus_index, settings=test_settings, ledger=EvidenceLedger())


def test_research_executes_tools_chosen_by_model_multi_hop(ctx):
    """The model picks different tools for different drugs; evidence spans both documents."""
    client = FakeLLMClient([
        completion(tool_calls=[tool_call("search_corpus", {"query": "warfarin drugs that increase bleeding risk SSRIs"}),
                               tool_call("search_corpus", {"query": "sertraline bleeding warfarin"})]),
        completion(tool_calls=[tool_call("read_source_document",
                                         {"filename": "dailymed-warfarin.md", "section": "7.3"})]),
        completion("Covered: bleeding risk (both labels), monitoring (warfarin 7.3)."),
    ])
    res = run_research(client, ctx, Q_MULTI, Plan(sub_questions=["risk?", "monitoring?"]))
    assert [t.tool for t in res.tool_calls] == ["search_corpus", "search_corpus", "read_source_document"]
    assert all(t.ok for t in res.tool_calls)
    assert res.stop_reason == "model_finished" and res.iterations == 3
    assert "dailymed-warfarin.md" in ctx.ledger.sources
    # tool schemas were offered to the model and observations returned with the right ids
    first = client.requests[0]
    assert {t["function"]["name"] for t in first["tools"]} == {"search_corpus", "read_source_document"}
    assert "tool_choice" not in first   # default ("auto") is used; no optional parameters are sent
    assert set(first) == {"model", "messages", "tools"}
    second_msgs = client.requests[1]["messages"]
    tool_msgs = [m for m in second_msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and all(m["tool_call_id"] for m in tool_msgs)


def test_research_stops_at_iteration_limit(ctx):
    ctx.settings.max_agent_iterations = 3
    client = FakeLLMClient([completion(tool_calls=[tool_call("search_corpus", {"query": f"warfarin topic {i}"})])
                            for i in range(3)])
    res = run_research(client, ctx, "q", Plan(sub_questions=["q"]))
    assert res.stop_reason == "max_iterations" and res.iterations == 3
    assert len(client.requests) == 3


def test_research_reports_bad_calls_back_to_model_and_continues(ctx):
    client = FakeLLMClient([
        completion(tool_calls=[tool_call("rm_rf", {"path": "/"}),
                               tool_call("read_source_document", {"filename": "../../.env"}),
                               tool_call("search_corpus", "{bad json")]),
        completion(tool_calls=[tool_call("search_corpus", {"query": "warfarin contraindications"})]),
        completion("done"),
    ])
    res = run_research(client, ctx, "q", Plan(sub_questions=["q"]))
    assert [t.ok for t in res.tool_calls] == [False, False, False, True]
    errors = [json.loads(m["content"])["error"] for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert "Unknown tool" in errors[0] and "Invalid arguments" in errors[1] and "valid JSON" in errors[2]


def test_research_blocks_duplicate_calls(ctx):
    call = {"query": "warfarin contraindications"}
    client = FakeLLMClient([
        completion(tool_calls=[tool_call("search_corpus", call, "a")]),
        completion(tool_calls=[tool_call("search_corpus", call, "b")]),
        completion("done"),
    ])
    res = run_research(client, ctx, "q", Plan(sub_questions=["q"]))
    assert res.tool_calls[1].ok is False and "Duplicate" in res.tool_calls[1].summary


def test_research_nudges_once_when_model_skips_tools(ctx):
    client = FakeLLMClient([completion("I know this already."), completion("Still no."),])
    res = run_research(client, ctx, "q", Plan(sub_questions=["q"]))
    assert len(client.requests) == 2 and len(ctx.ledger) == 0 and res.stop_reason == "model_finished"


def test_research_enforces_tool_budget(ctx):
    ctx.settings.max_tool_calls = 2
    client = FakeLLMClient([
        completion(tool_calls=[tool_call("search_corpus", {"query": f"warfarin q{i}"}) for i in range(4)]),
        completion("done"),
    ])
    res = run_research(client, ctx, "q", Plan(sub_questions=["q"]))
    assert [t.ok for t in res.tool_calls] == [True, True, False, False]


# --- critic ---------------------------------------------------------------------

def _ledger():
    led = EvidenceLedger()
    led.add(Evidence("dailymed-metformin.md", 44, 50, "Severe renal impairment (eGFR below 30 mL/min/1.73 m2)",
                     "4 CONTRAINDICATIONS"))
    return led


def test_critic_approves_grounded_answer(test_settings):
    led = _ledger()
    draft = "Contraindicated below eGFR 30 mL/min/1.73 m2 [dailymed-metformin.md:L44-L50]."
    r = review_answer(FakeLLMClient([review_json("approve")]), test_settings, "q", ["q"], draft, led,
                      validate_citations(draft, led))
    assert r.approved and not r.overrides


def test_critic_cannot_approve_with_fabricated_citation(test_settings):
    led = _ledger()
    draft = "Contraindicated [dailymed-metformin.md:L44-L50]. Also lowers INR [dailymed-made-up.md:L1-L2]."
    r = review_answer(FakeLLMClient([review_json("approve")]), test_settings, "q", ["q"], draft, led,
                      validate_citations(draft, led))
    assert r.verdict == "revise"
    assert any("made-up" in p for p in r.citation_problems)


def test_critic_cannot_approve_uncited_answer(test_settings):
    led = _ledger()
    draft = "Metformin is contraindicated."
    r = review_answer(FakeLLMClient([review_json("approve")]), test_settings, "q", ["q"], draft, led,
                      validate_citations(draft, led))
    assert r.verdict == "insufficient_evidence" and "no valid citations" in r.overrides[0]


def test_uncited_answer_with_unsupported_claims_is_revised(test_settings):
    led = _ledger()
    draft = "Metformin is safe at any eGFR."
    r = review_answer(FakeLLMClient([review_json("approve", unsupported_claims=["safe at any eGFR"])]),
                      test_settings, "q", ["q"], draft, led, validate_citations(draft, led))
    assert r.verdict == "revise"


def test_uncited_insufficient_answer_is_final_without_revision(corpus_index, test_settings):
    """A correct 'not in the corpus' answer has no citations; it must not be pushed into a revision loop."""
    client = FakeLLMClient([plan_json(["lisinopril dose?"], drugs=("lisinopril",)), *_research_script(),
                            completion("The provided documents do not contain information on lisinopril dosing."),
                            review_json("approve")])
    out = ResearchAgent(client, test_settings, corpus_index).run("lisinopril dose?")
    assert out.status == "insufficient_evidence" and out.citations == [] and not out.trace["revised"]


def test_critic_approve_with_unsupported_claims_is_overridden(test_settings):
    led = _ledger()
    draft = "Contraindicated [dailymed-metformin.md:L44-L50]."
    r = review_answer(FakeLLMClient([review_json("approve", unsupported_claims=["dose 500 mg"])]),
                      test_settings, "q", ["q"], draft, led, validate_citations(draft, led))
    assert r.verdict == "revise"


def test_critic_sees_evidence_and_draft(test_settings):
    led = _ledger()
    client = FakeLLMClient([review_json("approve")])
    review_answer(client, test_settings, "q", ["q"], "draft text [dailymed-metformin.md:L44-L50]", led,
                  validate_citations("draft text [dailymed-metformin.md:L44-L50]", led))
    user = client.requests[0]["messages"][1]["content"]
    assert "eGFR below 30" in user and "draft text" in user and "Valid citations" in user


def test_critic_unparseable_output_means_revise(test_settings):
    led = _ledger()
    draft = "x [dailymed-metformin.md:L44-L50]"
    r = review_answer(FakeLLMClient([completion("looks fine to me")]), test_settings, "q", ["q"], draft, led,
                      validate_citations(draft, led))
    assert r.verdict == "revise"


# --- full agent -----------------------------------------------------------------

def _research_script():
    return [
        completion(tool_calls=[tool_call("read_source_document",
                                         {"filename": "dailymed-metformin.md", "section": "4 CONTRAINDICATIONS"})]),
        completion("Covered."),
    ]


GOOD = "Contraindicated below eGFR 30 mL/min/1.73 m2 [dailymed-metformin.md:L44-L50]."


def test_agent_happy_path_single_review(corpus_index, test_settings):
    client = FakeLLMClient([plan_json(["Is it contraindicated?"]), *_research_script(),
                            completion(GOOD), review_json("approve")])
    out = ResearchAgent(client, test_settings, corpus_index).run("metformin eGFR 25?")
    assert out.status == "success"
    assert out.trace["revised"] is False and len(out.trace["reviews"]) == 1
    assert out.citations[0].source == "dailymed-metformin.md"
    assert out.citations[0].section == "4 CONTRAINDICATIONS"
    assert out.disclaimer == DISCLAIMER and DISCLAIMER in out.answer
    assert len(client.requests) == 5


def test_agent_revises_once_then_approves(corpus_index, test_settings):
    client = FakeLLMClient([plan_json(["q"]), *_research_script(),
                            completion("Contraindicated, use 500 mg instead [dailymed-metformin.md:L44-L50]."),
                            review_json("revise", unsupported_claims=["500 mg dose not in evidence"],
                                        feedback="Remove the dose."),
                            completion(GOOD), review_json("approve")])
    out = ResearchAgent(client, test_settings, corpus_index).run("metformin eGFR 25?")
    assert out.status == "success" and out.trace["revised"] is True
    revision_prompt = client.requests[5]["messages"][1]["content"]
    assert "500 mg dose not in evidence" in revision_prompt and "Previous draft" in revision_prompt
    assert "500 mg" not in out.answer


def test_agent_revision_is_bounded_and_flags_remaining_problems(corpus_index, test_settings):
    bad = review_json("revise", unsupported_claims=["claim X unsupported"])
    client = FakeLLMClient([plan_json(["q"]), *_research_script(), completion(GOOD), bad, completion(GOOD), bad])
    out = ResearchAgent(client, test_settings, corpus_index).run("q?")
    assert out.status == "needs_review"
    assert len(out.trace["reviews"]) == 2          # exactly one revision, then stop
    assert "claim X unsupported" in out.answer     # unresolved flags are shown to the user
    assert client.chat.completions.responses == []


def test_agent_insufficient_evidence_from_critic(corpus_index, test_settings):
    insuff = review_json("insufficient_evidence", missing_aspects=["corpus has no pediatric data"])
    client = FakeLLMClient([plan_json(["q"]), *_research_script(), completion(GOOD), insuff,
                            completion("The provided documents do not state this."), insuff])
    out = ResearchAgent(client, test_settings, corpus_index).run("q?")
    assert out.status == "insufficient_evidence" and DISCLAIMER in out.answer


def test_agent_no_evidence_skips_generation(corpus_index, test_settings):
    client = FakeLLMClient([plan_json(["q"]), completion("nothing"), completion("nothing")])
    out = ResearchAgent(client, test_settings, corpus_index).run("What is the capital of France?")
    assert out.status == "insufficient_evidence" and out.citations == []
    assert len(client.requests) == 3   # plan + research + one nudge; no answer/critic calls


# --- corpus coverage and out-of-corpus questions -----------------------------------

from app.agent.agent import coverage_text


def test_drug_coverage_uses_corpus_documents(corpus_index):
    cov = corpus_index.drug_coverage(["Warfarin sodium", "lisinopril"])
    assert cov["Warfarin sodium"] == ["dailymed-warfarin.md", "medlineplus-warfarin.md"]
    assert cov["lisinopril"] == []
    text = coverage_text(cov)
    assert "lisinopril: NO dedicated document" in text and "not evidence about the drug" in text


def test_out_of_corpus_question_has_no_unrelated_citations(corpus_index, test_settings):
    """Regression: a lisinopril question must not present atorvastatin/sertraline passages as support."""
    unrelated = "dailymed-atorvastatin.md:L17-L17"
    draft = ("The provided documents do not contain lisinopril dosing.\n\n## Evidence gaps\n"
             f"The passages only cover atorvastatin dosing [{unrelated}].")
    insuff = review_json("insufficient_evidence", irrelevant_citations=[unrelated],
                         citation_problems=[f"{unrelated} does not support a lisinopril claim"])
    client = FakeLLMClient([
        plan_json(["What is the starting dose of lisinopril?"], drugs=("lisinopril",)),
        completion(tool_calls=[tool_call("search_corpus", {"query": "lisinopril starting dose hypertension"}),
                               tool_call("read_source_document", {"filename": "dailymed-atorvastatin.md",
                                                                  "start_line": 17, "end_line": 17})]),
        completion("Lisinopril is not covered by the corpus."),
        completion(draft), insuff, completion(draft), insuff,
    ])
    out = ResearchAgent(client, test_settings, corpus_index).run("What is the starting dose of lisinopril?")
    assert out.status == "insufficient_evidence"
    assert out.citations == [] and out.sources == []
    assert "atorvastatin.md" not in out.answer
    assert out.trace["corpus_coverage"] == {"lisinopril": []}
    assert out.trace["removed_citations"] == [unrelated]
    # coverage information reached the research loop, the generator and the critic
    for i in (1, 3, 4):
        assert "lisinopril: NO dedicated document" in json.dumps(client.requests[i]["messages"])
    # the revision request told the writer to drop the irrelevant citation
    assert unrelated in client.requests[5]["messages"][1]["content"]


def test_critic_cannot_approve_with_irrelevant_citations(test_settings):
    led = _ledger()
    draft = "Contraindicated below eGFR 30 mL/min/1.73 m2 [dailymed-metformin.md:L44-L50]."
    r = review_answer(FakeLLMClient([review_json("approve", irrelevant_citations=["dailymed-metformin.md:L44-L50"])]),
                      test_settings, "q", ["q"], draft, led, validate_citations(draft, led))
    assert r.verdict == "revise" and "irrelevant" in r.overrides[0]


def test_critic_cannot_approve_altered_threshold(test_settings):
    led = _ledger()
    draft = "Contraindicated below eGFR 45 mL/min/1.73 m2 [dailymed-metformin.md:L44-L50]."
    client = FakeLLMClient([review_json("approve")])
    r = review_answer(client, test_settings, "q", ["q"], draft, led, validate_citations(draft, led))
    assert r.verdict == "revise" and r.number_problems and "numbers not found" in r.overrides[0]
    critic_input = client.requests[0]["messages"][1]["content"]
    assert "45" in critic_input and "does not appear in the cited passages" in critic_input  # critic saw the finding


def test_unresolved_number_problem_is_shown_to_user(corpus_index, test_settings):
    bad = "Contraindicated below eGFR 45 mL/min/1.73 m2 [dailymed-metformin.md:L44-L50]."
    client = FakeLLMClient([plan_json(["q"]), *_research_script(), completion(bad), review_json("approve"),
                            completion(bad), review_json("approve")])
    out = ResearchAgent(client, test_settings, corpus_index).run("metformin eGFR?")
    assert out.status == "needs_review" and "'45'" in out.answer


def test_revise_without_findings_is_treated_as_approve(test_settings):
    """Observed live: critic said 'revise' with empty finding lists and only style advice in feedback."""
    led = _ledger()
    draft = "Contraindicated below eGFR 30 mL/min/1.73 m2 [dailymed-metformin.md:L44-L50]."
    r = review_answer(FakeLLMClient([review_json("revise", feedback="Remove the Evidence gaps section.")]),
                      test_settings, "q", ["q"], draft, led, validate_citations(draft, led))
    assert r.approved and "without any structured finding" in r.overrides[0]
    assert r.feedback == "Remove the Evidence gaps section."


def test_revise_with_findings_stays_revise(test_settings):
    led = _ledger()
    draft = "Contraindicated below eGFR 30 mL/min/1.73 m2 [dailymed-metformin.md:L44-L50]."
    r = review_answer(FakeLLMClient([review_json("revise", missing_aspects=["discontinuation rule omitted"])]),
                      test_settings, "q", ["q"], draft, led, validate_citations(draft, led))
    assert r.verdict == "revise"


def test_unparseable_critic_output_still_forces_revision(test_settings):
    led = _ledger()
    draft = "x [dailymed-metformin.md:L44-L50]"
    r = review_answer(FakeLLMClient([completion("not json")]), test_settings, "q", ["q"], draft, led,
                      validate_citations(draft, led))
    assert r.verdict == "revise"   # the 'critic output unusable' override keeps it a revision


# --- model portability: provider rejects the tool-calling request -------------------

import httpx
import openai


def _bad_request():
    return openai.BadRequestError("tools are not supported for this model",
                                  response=httpx.Response(400, request=httpx.Request("POST", "http://llm/chat")),
                                  body=None)


def test_rejected_tool_request_falls_back_to_retrieval(ctx):
    client = FakeLLMClient([_bad_request()])
    res = run_research(client, ctx, "q", Plan(sub_questions=["warfarin contraindications", "apixaban contraindications"]))
    assert res.stop_reason == "tool_calling_unavailable_fallback_retrieval"
    assert [t.tool for t in res.tool_calls] == ["search_corpus", "search_corpus"] and len(ctx.ledger) > 0


def test_rejected_later_request_keeps_gathered_evidence(ctx):
    client = FakeLLMClient([completion(tool_calls=[tool_call("search_corpus", {"query": "warfarin contraindications"})]),
                            _bad_request()])
    res = run_research(client, ctx, "q", Plan(sub_questions=["q"]))
    assert res.stop_reason == "provider_rejected_request" and len(ctx.ledger) > 0


def test_every_chat_request_sends_only_portable_parameters(corpus_index, test_settings):
    client = FakeLLMClient([plan_json(["Is it contraindicated?"]), *_research_script(),
                            completion(GOOD), review_json("approve")])
    ResearchAgent(client, test_settings, corpus_index).run("metformin eGFR 25?")
    for req in client.requests:
        assert set(req) <= {"model", "messages", "tools"}, set(req)
