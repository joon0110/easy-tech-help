# Product quality improvements — 2026-09-21

## Scope

This pass fixes the concrete application failures found in the [initial full audit](full_evaluation.md). It does **not** retrain the model. The Qwen2.5-1.5B-Instruct base, selected PyTorch/PEFT LoRA weights, 200 training/validation/regression records and ten reference documents are unchanged. SHA-256 comparisons in the before/after reports confirm the adapter, dataset and corpus identities.

The raw adapter still matches 23/34 inspected regression observations, with signal recall 15/22 (68.2%). Public SMS raw signal recall remains 0/4. Application corrections and policy recovery must not be presented as improved fine-tuned model accuracy. All revised code remains uncommitted for review.

## Changes and rationale

| Area | Change | Verification focus |
| --- | --- | --- |
| Sensitive requests | Shared literal cues cover login-number paraphrases, compound password/code requests, code pronouns and explicit premium-call charges | Current requests versus negation, historical/educational text and ordinary prices |
| Ordinary controls | Code notifications, reported native Settings updates and informational URLs are distinguished from disclosure/install/link requests | Avoid unnecessary warnings while retaining actual account threats and payment requests |
| Observation grounding | Restore unique case-only quote differences; re-ground limited code/connection evidence using independently located exact input spans | Full strict schema validation remains; fabricated unsupported evidence is rejected and raw predictions preserved |
| Categories and Wi-Fi | Recognize explicit alert context, current connected/Airplane Mode state and contradictory Wi-Fi states; remove security/joining interpretations supported only by “No Internet Connection” | Exact observation comparisons, false-security controls and appropriate troubleshooting actions |
| RAG | Search up to 18 chunks, six per document; select up to three topic-compatible units from one reference | Reject mismatched billing/lottery/device-list passages, broad assurances, unqualified state changes and payment headings without useful explanation |
| Safety and handoff | Shared cue/context rules, fixed supported action templates, final explanation filtering | Allowed actions, literal local support, unchanged family-summary constraints and explicit privacy markers |
| UI | Darker warning text and removal of overlapping invisible Streamlit header | Desktop/mobile axe scan and real-model copy/download/reset flow |
| Project documentation | README, architecture and evaluation provenance updated | Current results separated from historical training and audit reports |

Rules are bounded English heuristics. Exact input evidence proves occurrence, not semantic truth. A user saying a prompt is in Settings is not authenticated device provenance. Sensitive cues retain priority over that context.

## Measured results

All actual-model runs used the existing adapter on MPS. The 70-case product report and 16-case verification report retain every output.

| Check | Before | After |
| --- | --- | --- |
| Raw adapter exact observations, inspected regression | 23/34 | **23/34**, unchanged |
| Application-reviewed exact observations, same regression | 23/34 | **34/34**, tuned development recovery |
| Sensitive regression requests: expected warning and flags | 4/7 | **7/7** |
| Original authored challenges: category, level, required flags and action | 19/24 | **24/24** |
| Known product development cases | 12/12 | **12/12** |
| Action/evidence/handoff output contracts | 70/70 | **70/70** |
| Additional authored verification | Not run | **16/16** after one inspected failure was fixed; first run 15/16 |
| Automated tests | 205 passed | **237 passed**, seven opt-in tests skipped |
| Separately enabled actual-model smoke tests | 7/7 | **7/7** |
| Ruff lint and formatting; Git whitespace | Passed | **Passed** |
| Explanations selected / displayed across 70 cases | 39 / 37 | **29 / 28**, greater abstention |

The final 70-case run took a mean **2.545 s** per case, nearest-rank p95 **4.825 s**, excluding model loading. All 16 additional cases also pass their eight output contracts. Required-flag checks are subset checks; they are not exact observation accuracy. The older regression metric above separately compares complete category/signal/issue sets.

### RAG evaluation correction

The 70-case report retains one false failure from the older eight-case RAG checker: `public_wifi/past_present_context_preserved`. That assertion demanded the literal “In the past,” and “Today,” passage even when the selected answer was complete HTTPS guidance containing neither historical claim. The checker was corrected **after** the 70-case run: if either temporal marker is quoted, both are required; a different complete passage is allowed. Four tests reject either half alone and accept the full contrast or unrelated complete guidance. Original case text, expected status and expected source documents were not relabeled. This is a checker correction, not a claimed model improvement. The separate current RAG check report records the rerun; its source-code fingerprints therefore differ from the earlier 70/16-case snapshots in the evaluator module.

### RAG and browser completion

The corrected RAG checker passes **8/8 live development cases** in [its separate report](rag_quality_checks.json). This does not replace independent relevance review.

Chrome/axe checks found **zero violations and zero incomplete checks** on the initial page at 1440, 390 and 320 px, and on the exercised result page with sources and family summary expanded. The previous warning contrast failure is resolved. Actual local-model submission, official source links, copying without hover, exact clipboard/download contents, reset and narrow-screen overflow checks all passed; no uncaught JavaScript errors were observed. These are automated checks of one result flow, not accessibility certification or an older-adult usability study. [UI previews](interface.md#previews) were refreshed from this run.

## Additional verification and discovered failure

Sixteen assistant-authored inputs were frozen after the initial fixes and before their first model run. None is an exact normalized duplicate of the 200 existing examples or 24 challenge inputs. They remain conceptually related and are not independent real-world data.

The [first verification run](verification_initial.json) passed **15/16**. In the failed home-network case, the raw model labeled “No Internet Connection” as both connected and unsecured. The unsecured flag suppressed useful connectivity actions. A bounded review rule now rejects that phrase as evidence for either property, retaining a connected state only when separate literal evidence exists. The input and expected result were not changed. Short payment section headings were also rejected after inspecting the selected explanations. These 16 cases are now development/regression material.

## Interpretation and remaining limits

- The original 24 challenge cases were inspected and used for application fixes. Their improved scores show regression recovery, not unseen generalization. The same limitation applies to the 34 older regression records and 12 development inputs.
- Safety contracts and classification/action quality have different denominators. Passing output constraints alone does not mean a missed scam was detected.
- Stronger relevance filtering deliberately reduces explanation coverage. When no suitable passage is available, the app abstains while independently supported fixed actions can still be shown. This is preferable to an unrelated explanation, but leaves coverage gaps, including some payment and remote-support inputs.
- Sentence topic gates and qualitative assistant review do not establish semantic accuracy. Public Wi-Fi HTTPS advice is accompanied by the policy's warning that encryption does not establish website or network trustworthiness.
- Three original challenge cases and three additional cases contain explicit privacy canaries. Other cases do not count as privacy probes.
- Contemporary real messages, independently reviewed labels, real iPhone alert/Wi-Fi data, screen-reader checks and sessions with older users remain necessary before unsupervised deployment. No new real-world accuracy or accessibility certification is claimed.

## Reproduction

Run from the repository with the installed environment and existing local model/adapter:

```bash
python -m pytest -q
EASY_TECH_HELP_RUN_LIVE_TESTS=1 python -m pytest tests/test_analysis_live.py -q
ruff check .
ruff format --check .
python -m easy_tech_help.product_evaluation --device mps --output results/product_improved.json
python -m easy_tech_help.verification_evaluation --device mps --output results/verification.json
python -m easy_tech_help.rag_evaluation --device mps --output results/rag_quality_checks.json
streamlit run app/app.py
```

The product evaluator saves failures even when it exits nonzero. Raw/reviewed observation metrics use full-pipeline case time, not extraction-only latency. Local model loading is excluded from the 70-case timings. Reports retain inputs, raw generations, corrected observations, reason traces, retrieved passages, selected quotes, policy decisions, handoffs and fingerprints. Diagnostic reports contain authored/redacted fixtures and must not be mistaken for a safe export format for real private user inputs.

## Evidence files

- [Full 70-case after-fix outputs and fingerprints](product_improved.json)
- [Additional 16-case final verification](verification.json) and [first run with failure](verification_initial.json)
- [Qualitative review of all 24 challenge reference outputs](rag_review_improved.json)
- [Current eight-case RAG checker rerun](rag_quality_checks.json)
- [Desktop/mobile input accessibility](ui_improved_accessibility.json) and [actual-model browser flow](ui_improved_flow.json)
- [Frozen inputs and provenance](../data/evaluation/README.md)

The broad initial `full_*` reports remain the before-fix record. Training runs and their historical reports were not replaced or removed. Temporary browser tooling and execution logs remain outside the repository.
