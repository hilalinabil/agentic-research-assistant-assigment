# Manual review of the reported evaluation run

Reported run: [`results/2026-11-12T11-04-16Z/`](results/2026-11-12T11-04-16Z/RESULTS.md) — chat model `gpt-5.6-terra`,
embeddings `text-embedding-3-small`, code+corpus fingerprint `512d5d9ab30a5160` (code changed slightly afterwards; see [`results/README.md`](results/README.md)).

Method: each answer was read next to the cited lines of the source files; narrowed citations (≤2 lines) were also
checked mechanically against the cited source text. No numeric score is given; an LLM judge would not be ground truth.

| ID | Status | Expected sources cited | Valid / invalid citations | Tool calls | Revised | Human check |
|---|---|---|---|---|---|---|
| Q1 | success | yes | 11 / 0 | 2 | no | Correct and complete |
| Q2 | success | yes | 18 / 0 | 4 | yes | Correct |
| Q3 | success | yes | 6 / 0 | 8 | yes | Correct; one relevant instruction omitted (see below) |
| Q4 | success | yes | 3 / 0 | 1 | no | Correct; unit assumption stated |
| X1 (control) | insufficient_evidence | n/a | 0 / 0 | 1 | no | Correct abstention, no citations |

## Q1 — Warfarin contraindications
- Tools: search, then `read_source_document(dailymed-warfarin.md, "4 CONTRAINDICATIONS")` (L127–L146).
- All items of §4 are listed (pregnancy with the mechanical-heart-valve exception, hemorrhagic tendencies/blood
  dyscrasias, CNS/eye/traumatic surgery, bleeding tendencies, threatened abortion/eclampsia/preeclampsia, unsupervised
  non-compliant patients, spinal puncture, hypersensitivity, major regional/lumbar block anesthesia, malignant
  hypertension); single-line citations point to the exact list items.

## Q2 — Warfarin vs apixaban
- Tools: one search per drug, then the full §4 of each label.
- Apixaban §4 (L108–L114): active pathological bleeding and severe hypersensitivity — reported exactly. Shared:
  hypersensitivity, and bleeding-related contraindications at category level (different wording). Unique to warfarin:
  pregnancy, procedures/anesthesia, threatened abortion/eclampsia, non-compliance, malignant hypertension. Correct.
- Critic, first review: flagged that the draft presented warfarin-only bleeding conditions as definitively *not*
  contraindications for apixaban, and that "shared" holds only at category level → revised → approved.

## Q3 — Warfarin + sertraline (multi-hop)
- Tools: 4 searches, then 4 section reads across both labels (sertraline §5.3 and §7.1, warfarin §7.3 and §5.1).
- Verified: SRIs increase bleeding risk and warfarin may add to it; anticoagulant + sertraline may potentiate bleeding;
  "for patients taking warfarin, carefully monitor the INR" (sertraline L102–L107, L339–L342); warfarin Table 3 lists
  sertraline and says to closely monitor (warfarin L251–L260); more frequent INR monitoring when other drugs are
  started/stopped/dose-changed and therapeutic INR does not eliminate bleeding risk (warfarin L149–L155); bleeding
  symptoms to report (warfarin §17 L428–L445).
- Critic, first review: the draft cited `dailymed-warfarin.md:L428-L432`, a guessed range inside an unnumbered chunk;
  the validator rejected it and the revision cited the retrieved passage range.
- **Omission:** the sertraline protein-binding interaction ("monitor for adverse reactions and reduce the dosage of
  sertraline or the other protein-bound drug as warranted", warfarin as example; sertraline §7.1) was in the evidence
  but is not in the answer, and the critic did not request it. An earlier development run included it.
- Minor: the opening sentence applies the warfarin label's general "other drugs started/stopped" monitoring rule to
  sertraline; it cites both labels but is not marked "Interpretation:".

## Q4 — Metformin at eGFR 25 mL/min
- Correct: contraindicated below eGFR 30 mL/min/1.73 m² (§4 L46–L50); assess renal function before and periodically,
  discontinue if eGFR later falls below 30 (§2.3 L34–L38). Label units reproduced exactly.
- The answer states the unit assumption: the question's "25 mL/min" is not body-surface indexed, the label's threshold is.

## X1 — Lisinopril (out-of-corpus control, not an assessment question)
- Coverage note: no dedicated lisinopril document. The answer says the documents "do not contain information on the
  recommended starting dose of lisinopril" — it does not claim that the drug or a dose does not exist — and has no
  citations, so no unrelated file is listed as a source.

## Issues investigated in this audit
| Issue | Root cause (verified) | Fix |
|---|---|---|
| Out-of-corpus answer cited atorvastatin/sertraline passages | Generator cited passages to show "what the corpus contains"; nothing stopped citations that support no relevant claim | Generator rule (cite only supporting passages; no citations on gap statements); critic field `irrelevant_citations`, removed from the answer by code; corpus-coverage note per planned drug; an approved answer with no valid citations is treated as `insufficient_evidence` |
| Q3 omitted the sertraline protein-binding instruction | The passage **was** in the generator's context (sertraline §7.1 L322–L359, 23.6k chars total, no truncation); the generator left it out and the critic did not ask for it | Generator rule to include management instructions for the drugs/situation asked; critic checks passages one by one for omissions that change or qualify the answer |
| Q4 compared "25 mL/min" with "30 mL/min/1.73 m²" without comment | Evidence units were passed correctly; nothing required reconciling question units with label units | Generator rule to state unit/population differences and assumptions; critic instructed to flag them; deterministic `check_numbers` flags numbers/units not present in the cited passages |
| Reported "Q3 revised" vs table "not revised" | Two different runs: the first (overwritten) run revised Q3, the second did not | New run directories that are never overwritten; reported run named explicitly; earlier runs in git history |

| A correct answer ended `needs_review` because the critic said "revise" with no findings (only style advice: "remove the Evidence gaps section") | Verdict not tied to findings | A "revise" without any structured finding is treated as "approve"; the feedback stays in the trace |

A first version of the completeness rule was too broad (critic demanded unrelated safety information; Q3/Q4 ended
`needs_review` with bloated answers in an intermediate run). It was narrowed to omissions that change or
qualify the answer.

## Remaining observations
- Run-to-run variation: tool sequences, included details and critic findings differ between runs. The Q3
  protein-binding instruction was included in one of the last two runs and omitted in the reported one — the
  completeness rules reduce, but do not eliminate, omissions.
- The number/unit check is a detector only: it flags numbers or units absent from the cited passages but cannot
  tell whether a present number is used correctly.
