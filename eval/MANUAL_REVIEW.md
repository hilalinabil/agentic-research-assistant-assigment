# Manual review of the reported evaluation run

Reported run: [`results/2026-11-12T11-32-54Z/`](results/2026-11-12T11-32-54Z/RESULTS.md) — chat model `gpt-5.6-terra`, embeddings
`text-embedding-3-small`, code+corpus fingerprint `e53b19be1dd283d5` (identical to the submitted code).

Method: each answer was read next to the cited lines of the source files. No numeric score is given; an LLM judge
would not be ground truth.

| ID | Status | Expected sources cited | Valid / invalid citations | Tool calls | Revised | Human check |
|---|---|---|---|---|---|---|
| Q1 | success | yes | 1 / 0 | 1 | no | Correct and complete; one passage-level citation |
| Q2 | success | yes | 12 / 0 | 4 | no | Correct |
| Q3 | success | yes | 7 / 0 | 2 | yes | Correct and complete after one revision |
| Q4 | success | yes | 2 / 0 | 3 | no | Correct; unit assumption stated |
| X1 (control) | insufficient_evidence | n/a | 0 / 0 | 1 | no | Correct abstention, no citations |

## Q1 — Warfarin contraindications
- Tools: one search (`warfarin contraindications FDA label`); the top hit was the whole §4 chunk (L129–L145), so the
  model did not read the section separately.
- All ten §4 items are present: pregnancy (with the mechanical-heart-valve exception), hemorrhagic tendencies/blood
  dyscrasias, CNS/eye/traumatic surgery, bleeding tendencies (GI/GU/respiratory bleeding, CNS hemorrhage, cerebral
  aneurysms, dissecting aorta, pericarditis/effusions, bacterial endocarditis), threatened abortion/eclampsia/
  preeclampsia, unsupervised non-compliant patients, spinal puncture, hypersensitivity, major regional/lumbar block
  anesthesia, malignant hypertension.
- Every sentence cites the same passage range `dailymed-warfarin.md:L129-L145` — correct, but coarser than the
  per-line citations produced when the model reads the section (as in Q2).

## Q2 — Warfarin vs apixaban
- Tools: one search per drug, then the full §4 of each label (line-numbered reads).
- Apixaban §4 (L108–L114): active pathological bleeding and severe hypersensitivity — reported exactly. Warfarin §4
  items listed with narrowed citations inside the line-numbered read. Shared: hypersensitivity; bleeding-related
  contraindications as a category with different wording. Unique to warfarin: pregnancy, procedures/anesthesia,
  threatened abortion/eclampsia, non-compliance, malignant hypertension. Approved on first review.
- Several narrowed ranges start a few lines early (e.g. `L135-L138` for the surgery item on L138); each range
  contains the supporting line.

## Q3 — Warfarin + sertraline (multi-hop)
- Tools: two searches (one framed from each drug); evidence from both DailyMed labels.
- Verified: SRIs including sertraline increase bleeding risk and warfarin may add to it (sertraline L104–L106);
  anticoagulant + sertraline may potentiate bleeding and "for patients taking warfarin, carefully monitor the INR"
  (sertraline L324–L342); warfarin Table 3 lists sertraline and says to closely monitor (warfarin L253–L259); more
  frequent INR monitoring when other drugs are started/stopped/dose-changed, and therapeutic INR does not
  eliminate bleeding risk (warfarin L151–L157); additional INR tests when other medications start/stop (L93–L95);
  protein-binding interaction: "monitor for adverse reactions and reduce the dosage of sertraline or the other
  protein-bound drug as warranted" (sertraline L342–L357); bleeding symptoms to report (warfarin §17).
- Critic, first review: flagged that the draft implied routine warfarin dose adjustment when sertraline starts, and
  that the protein-binding instruction was missing → one revision fixed both.
- The warfarin label's general "other drugs started/stopped" monitoring rule is applied to sertraline; both labels
  are cited for that sentence.

## Q4 — Metformin at eGFR 25 mL/min
- Correct: contraindicated below eGFR 30 mL/min/1.73 m²; discontinue if eGFR falls below 30 during treatment;
  initiation not recommended at 30–45 (DailyMed §2.3 L34–L38). MedlinePlus statement ("tell your doctor if you have
  kidney disease … your doctor will probably tell you not to take metformin", L7–L17) reported accurately, with the
  note that it gives no numeric cutoff.
- The answer states the unit assumption: "25 mL/min" is not body-surface indexed, the label threshold is.

## X1 — Lisinopril (out-of-corpus control, not an assessment question)
- Retrieval returned unrelated passages (sertraline, atorvastatin dosing); none is cited. The answer states that the
  documents "do not contain information on the recommended starting dose of lisinopril" — missing coverage, not a
  claim that the drug or a dose does not exist.

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
- Run-to-run variation: tool sequences, citation granularity, included details and critic findings differ between
  runs. In development runs the Q3 protein-binding instruction was sometimes omitted without the critic noticing;
  in the reported run the critic caught it. The completeness rules reduce, but do not eliminate, omissions.
- The number/unit check is a detector only: it flags numbers or units absent from the cited passages but cannot
  tell whether a present number is used correctly.
