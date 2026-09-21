# Stage 6: family handoff

Run date: 2026-09-21. The family summary is implemented in Streamlit and the product CLI. No model, adapter, prompt or training dataset was changed for this stage.

## What changed

- `handoff.py` builds an English help request from reviewed category/flags and approved action IDs. Raw input, evidence quotes, generated prose and arbitrary guidance text are excluded.
- Before rendering, actions and their evidence are resolved again against local references. Modified actions, unrecognized metadata or missing supporting files produce a generic fallback.
- Summaries include the situation, unverified cues, what was checked, next steps, help needed and source links. Apple-based summaries are labeled. Actions already taken and resolution are explicitly unknown.
- Streamlit includes an **Ask family for help** panel with its native code-block copy control. CLI `--family-summary` exports only the shareable text. Nothing is sent automatically.

## Verification

| Check | Result | Scope |
| --- | --- | --- |
| Focused handoff tests | 18 passed | Private input, malicious model prose, modified actions/sources/metadata, unavailable sources, ordinary/Wi-Fi cases, completion claims, isolated CLI export |
| Full pytest suite | 198 passed, 7 opt-in live tests skipped | Includes Streamlit form submission and displayed share text |
| Ruff lint and format | Passed | Repository checks |
| Actual local model → RAG → safety → handoff | 12/12 cases passed all nine checks | Assistant-authored development inputs; not independent validation |

The real model run used MPS, PyTorch 2.14.0, Transformers 5.17.0 and PEFT 0.21.0 with the existing selected LoRA adapter. The twelve cases took 34.15 seconds in total, excluding initial model loading. [Complete traces, checks and fingerprints](handoff_development.json) preserve the actual outputs.

Live checks cover expected guidance levels, appropriate next actions, catalog template identity, prohibited action exclusion, literal action support, ready handoff status, preservation of reviewed actions, explicit unknown completion and absence of the complete original input in the summary. The last check alone is not a privacy proof; separate tests place names, email, phone, password, code, network identifier, malicious URL and injected instructions in raw input/model fields and verify their exclusion from shareable output. The builder's API excludes these free-text fields by construction.

## Reproduce

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m easy_tech_help.safety_evaluation --device mps --output results/handoff_development.json
python -m easy_tech_help.guidance --text "My iPhone Wi-Fi is off." --family-summary
```

## Limits

These are development checks, not evidence of real-world scam accuracy. Misclassified situations can produce an inaccurate family summary. Templates cannot establish that a sender is genuine or a network is safe, and the application has no record of what the user actually did.

The summary deliberately loses identifying/contextual detail to avoid forwarding private information. Review it with the user before sharing. Only the dedicated summary is intended for sharing: ordinary CLI JSON and evaluation traces retain raw input and model output.

Streamlit rendering was exercised with AppTest. The native copy control's operating-system clipboard interaction and readability with older users still need manual browser/usability testing. This stage does not add an external messaging service or a new model-generated rewrite.
