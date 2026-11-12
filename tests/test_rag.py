"""Phase 3: evidence ledger, context construction, citation validation and basic RAG."""
import pytest

from app.core.llm import LLMResponseError, as_str_list, parse_json_object
from app.rag.evidence import Evidence, EvidenceLedger, extract_citations, validate_citations
from app.rag.generator import INSUFFICIENT_EVIDENCE_ANSWER, basic_rag, generate_answer
from tests.conftest import FakeLLMClient, completion


def ev(source="dailymed-warfarin.md", start=129, end=145, text="Warfarin is contraindicated in pregnancy."):
    return Evidence(source=source, start_line=start, end_line=end, text=text, section="4 CONTRAINDICATIONS")


# --- ledger -------------------------------------------------------------------

def test_ledger_deduplicates_contained_passages():
    led = EvidenceLedger()
    assert led.add(ev(start=129, end=145))
    assert not led.add(ev(start=130, end=140))      # contained -> skipped
    assert led.add(ev(start=120, end=150))           # contains -> replaces
    assert len(led) == 1 and led.items[0].start_line == 120


def test_ledger_ignores_empty_text():
    led = EvidenceLedger()
    assert not led.add(ev(text="   "))


def test_context_contains_source_and_lines_and_respects_budget():
    led = EvidenceLedger()
    led.add(ev())
    led.add(ev(source="dailymed-sertraline.md", start=10, end=20, text="x" * 500))
    ctx = led.build_context(max_chars=10_000)
    assert "[E1] dailymed-warfarin.md:L129-L145" in ctx
    assert "section: 4 CONTRAINDICATIONS" in ctx
    small = led.build_context(max_chars=200)
    assert "omitted to fit context limit" in small


# --- citations ----------------------------------------------------------------

def test_extract_citation_formats():
    text = "A [dailymed-warfarin.md:L129-L145]. B [dailymed-apixaban.md:L10-12; GLOSSARY.md:L5]."
    refs = [c.ref for c in extract_citations(text)]
    assert refs == ["dailymed-warfarin.md:L129-L145", "dailymed-apixaban.md:L10-L12", "GLOSSARY.md:L5-L5"]


def test_validate_citations_detects_fabricated_file_and_wrong_lines():
    led = EvidenceLedger()
    led.add(ev())
    answer = ("Contraindicated in pregnancy [dailymed-warfarin.md:L129-L145]. "
              "Fake [dailymed-made-up.md:L1-L2]. Wrong lines [dailymed-warfarin.md:L400-L410].")
    rep = validate_citations(answer, led)
    assert [c.ref for c in rep.valid] == ["dailymed-warfarin.md:L129-L145"]
    reasons = {i["citation"]: i["reason"] for i in rep.invalid}
    assert "never retrieved" in reasons["dailymed-made-up.md:L1-L2"]
    assert "does not overlap" in reasons["dailymed-warfarin.md:L400-L410"]
    assert not rep.ok


def test_narrowed_citation_into_unnumbered_chunk_is_rejected():
    """Regression: the model guessed single-line numbers inside a search chunk it never saw numbered."""
    led = EvidenceLedger()
    led.add(ev(start=129, end=145))
    rep = validate_citations("Hypersensitivity [dailymed-warfarin.md:L141-L141].", led)
    assert not rep.valid and "not a retrieved passage range" in rep.invalid[0]["reason"]


def test_narrowed_citation_into_line_numbered_passage_is_accepted():
    led = EvidenceLedger()
    led.add(Evidence("dailymed-warfarin.md", 127, 146, "\n".join(f"line {n}" for n in range(127, 147)),
                     "4 CONTRAINDICATIONS", origin="read_source_document", line_numbered=True))
    rep = validate_citations("Hypersensitivity [dailymed-warfarin.md:L143-L143].", led)
    assert rep.ok
    assert "L143: line 143" in led.build_context(10_000)


def test_answer_without_citations_is_not_ok():
    led = EvidenceLedger()
    led.add(ev())
    assert not validate_citations("Warfarin is contraindicated in pregnancy.", led).ok


# --- JSON helpers ---------------------------------------------------------------

def test_parse_json_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 1} done') == {"a": 1}
    with pytest.raises(LLMResponseError):
        parse_json_object("no json here")
    with pytest.raises(LLMResponseError):
        parse_json_object("[1, 2]")


def test_as_str_list():
    assert as_str_list(None) == []
    assert as_str_list("x") == ["x"]
    assert as_str_list(["a", "", {"claim": "c"}]) == ["a", "claim: c"]


# --- generation -----------------------------------------------------------------

def test_generate_answer_without_evidence_does_not_call_llm(test_settings):
    client = FakeLLMClient([])
    assert generate_answer(client, test_settings, "q", EvidenceLedger()) == INSUFFICIENT_EVIDENCE_ANSWER
    assert client.requests == []


def test_generate_answer_sends_evidence_and_rules(test_settings):
    led = EvidenceLedger()
    led.add(ev())
    client = FakeLLMClient([completion("Contraindicated in pregnancy [dailymed-warfarin.md:L129-L145].")])
    out = generate_answer(client, test_settings, "Warfarin contraindications?", led, plan=["What are they?"])
    assert out.startswith("Contraindicated")
    req = client.requests[0]
    assert req["model"] == "test-chat-model"
    user = req["messages"][1]["content"]
    assert "dailymed-warfarin.md:L129-L145" in user and "What are they?" in user
    assert "[filename:Lstart-Lend]" in req["messages"][0]["content"]


def test_basic_rag_end_to_end_offline(corpus_index, test_settings):
    client = FakeLLMClient([completion("See [dailymed-metformin.md:L44-L50].")])
    answer, led = basic_rag(client, test_settings, corpus_index.retriever,
                            "metformin contraindications renal impairment eGFR")
    assert len(led) > 0
    assert "Evidence passages" in client.requests[0]["messages"][1]["content"]
    assert answer


# --- numeric and unit fidelity (detector, not proof of grounding) ----------------

from app.rag.evidence import check_numbers, remove_citations


def _metformin_ledger():
    led = EvidenceLedger()
    led.add(Evidence("dailymed-metformin.md", 44, 50,
                     "Severe renal impairment (eGFR below 30 mL/minute/1.73 m 2). Max 2,550 mg per day.",
                     "4 CONTRAINDICATIONS"))
    return led


@pytest.mark.parametrize("answer", [
    "Contraindicated below eGFR 30 mL/min/1.73 m² [dailymed-metformin.md:L44-L50].",
    "- Maximum 2550 mg per day [dailymed-metformin.md:L44-L50]",
    "1. Contraindicated below 30 mL/min/1.73 m² [dailymed-metformin.md:L44-L50]",   # list marker ignored
    "The documents do not state a pediatric dose of 10 mg.",                          # uncited line skipped
])
def test_number_check_accepts_faithful_numbers(answer):
    assert check_numbers(answer, _metformin_ledger(), "eGFR of 25 mL/min") == []


def test_number_check_allows_values_from_the_question():
    ans = "An eGFR of 25 mL/min is below 30 mL/min/1.73 m² [dailymed-metformin.md:L44-L50]."
    assert check_numbers(ans, _metformin_ledger(), "A patient with an eGFR of 25 mL/min") == []


def test_number_check_flags_invented_threshold():
    problems = check_numbers("Contraindicated below eGFR 45 [dailymed-metformin.md:L44-L50].", _metformin_ledger())
    assert len(problems) == 1 and "'45'" in problems[0]


@pytest.mark.parametrize("answer,unit", [
    ("Contraindicated below 30 mg [dailymed-metformin.md:L44-L50].", "30 mg"),
    ("Maximum 2,550 mcg per day [dailymed-metformin.md:L44-L50].", "2550 mcg"),
])
def test_number_check_flags_changed_units(answer, unit):
    problems = check_numbers(answer, _metformin_ledger())
    assert len(problems) == 1 and unit in problems[0]


def test_number_check_only_uses_the_cited_passage():
    led = _metformin_ledger()
    led.add(Evidence("dailymed-warfarin.md", 151, 157, "INR above 3.5 increases bleeding risk", "5.1 Hemorrhage"))
    # 3.5 exists in the warfarin passage, but the sentence cites the metformin passage
    assert check_numbers("Bleeding above INR 3.5 [dailymed-metformin.md:L44-L50].", led)


# --- citation removal -----------------------------------------------------------

def test_remove_citations_keeps_supporting_ones():
    text = ("A [dailymed-a.md:L1-L2].\nB [dailymed-x.md:L5-L9; dailymed-a.md:L1-L2].\n"
            "Not covered by the documents [dailymed-x.md:L5-L9]")
    out = remove_citations(text, ["dailymed-x.md:L5-L9"])
    assert "dailymed-x.md" not in out
    assert out.count("[dailymed-a.md:L1-L2]") == 2
    assert out.splitlines()[2] == "Not covered by the documents"


def test_remove_citations_noop_for_empty_list():
    assert remove_citations("A [a.md:L1-L2]", []) == "A [a.md:L1-L2]"


# --- generator prompt guards -------------------------------------------------------

def test_generator_prompt_contains_grounding_rules_and_coverage(test_settings):
    led = EvidenceLedger()
    led.add(ev())
    client = FakeLLMClient([completion("x [dailymed-warfarin.md:L129-L145]")])
    generate_answer(client, test_settings, "q", led, coverage_note="- lisinopril: NO dedicated document")
    system, user = (m["content"] for m in client.requests[0]["messages"])
    assert "never cite passages just to show what the corpus contains" in system
    assert "differs from how the evidence expresses it" in system
    assert "Corpus coverage:\n- lisinopril: NO dedicated document" in user
