# Full-product challenge inputs

`product_challenge.jsonl` contains 24 assistant-authored English cases for the 2026-09-21 product audit. Inputs and expectations were written before running this audit. They are synthetic, not collected messages or verbatim iPhone UI. Names, contacts, credentials and URLs are fictional test markers.

This file is outside `data/text/` and is not loaded by supervised training, checkpoint selection or the trusted RAG corpus. At the initial audit, no model weights, prompts, safety rules or gold labels were adjusted to make these cases pass. Subsequent application fixes used these findings; the original inputs and gold labels remain unchanged. Model weights and training data also remain unchanged. Cases are now inspected development material; do not call them an independent final test in future runs.

## Expectations

- `category`: expected observation category.
- `levels`: acceptable product guidance levels. Ordinary controls deliberately expect no special warning; a source-check warning on an informational link is recorded as unnecessary caution, not as a definitive false scam accusation.
- `required_flags`: signals the application must retain or recover. This is a required subset, not exact signal-set accuracy.
- `actions_any`: at least one of these approved actions should be offered.
- `documents_any`: expected explanatory reference documents when applicable. Retrieval/citation coverage is reported separately from action safety and is not proof of semantic relevance.
- `private_markers`: strings that must not appear in the family summary. Only three cases contain explicit privacy markers; empty lists do not provide privacy-test evidence.
- `group`: sensitive request, concerning popup, ordinary control, connection description or uncertain/out-of-scope input.

The challenge includes code/password paraphrases, payment requests, historical/current context, prompt injection, legitimate notifications and links, Wi-Fi settings, working connections and unsupported context. Exact input duplicates across the 200-row training/validation/regression corpus are rejected. Conceptual overlap remains: the closest word-sequence matches were 0.741 for a working Wi-Fi description and 0.727 for a notification-permission alert. This small targeted set cannot estimate real-world prevalence or accuracy.

## Run

```bash
python -m easy_tech_help.product_evaluation --device mps --output results/product_improved.json
```

The runner combines 34 previously inspected regression cases, these 24 challenge cases and 12 known development cases. It saves raw/reviewed observations, RAG output, policy decisions, shareable summaries, timings and fingerprints. A nonzero exit indicates a failed output constraint or scored quality check; failed cases stay in the report. See [the full audit](../../results/full_evaluation.md).

## Additional verification

`verification.jsonl` contains 16 additional assistant-authored inputs written after the initial fixes and before their first model run. They exercise new wording for code notifications/requests, conjunctive secrets, paid calls and ordinary prices, negation, Wi-Fi states, native/browser update context, injection, and missing context. Three cases contain explicit privacy markers. Inputs and expectations are frozen; they are not training examples. Conceptual overlap and assistant authorship prevent an independent accuracy claim. After inspection they are regression material too.

```bash
python -m easy_tech_help.verification_evaluation --device mps --output results/verification.json
```

The [quality improvement report](../../results/quality_improvement.md) separates raw model, application review, contract and authored quality results. Baseline `full_*` reports are historical; do not overwrite them to erase failures.
