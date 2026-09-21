# Grounding and application classification improvements

Date: 2026-09-21. Local PyTorch Qwen2.5-1.5B-Instruct + existing `pytorch-adapter-v2`, MPS on the M4 Mac. **No retraining occurred.** The adapter checksum matches the [initial RAG run](rag_development.json); training data and the extraction prompt are unchanged.

## Changes

### Extractive explanations

The earlier free-form explanations could add claims absent from the cited excerpt even when the source ID was valid. The model now generates a **source ID and one numbered source-unit selection**, or abstains. Code copies the selected unit verbatim. It rejects additional prose, invented IDs, arrays/booleans in place of an integer, invalid JSON and unfinished output. The UI calls this "What the reference says" and provides the surrounding passage.

This is an explicit change to **extractive RAG**. It prevents newly invented wording from being displayed as a sourced explanation; it is not proof that arbitrary model paraphrases can now be verified. The local model still generates structured observations and source selections, but it does not write a new conversational explanation. Source relevance and usefulness still require evaluation.

Source units retain entire sentences, including conditions and consequences. Immediate dependent continuations stay with their antecedent, and an adjacent "In the past" / "Today" contrast is retained. This prevents showing only historical public Wi-Fi risks while omitting the passage's present-day encryption context. These are bounded English heuristics, not a complete linguistic parser.

### Retrieval relevance

- Original input terms have weight 1.0; model-signal expansions have weight 0.2, so a mistaken signal has less influence.
- Explicit password/passcode/verification/credential terms in the original input have weight 3.0.
- Generic account messages use general references; Apple Account material requires Apple/iCloud context.
- Within the selected passage, source units are ordered by original-input overlap (generic terms weight 0.2, other terms 1.0). This helps the small model select the actual Wi-Fi state or popup request rather than generic background.
- Source files remain unchanged. No new model, embeddings service, vector database or external API was added.

### Classification corrections

`observation_review.py` applies narrow application rules after raw model validation:

1. A whole-input generic help request, such as "Please help me with this.", becomes `unknown` with `insufficient_context`. An actual message containing that phrase or a description naming a device state is not matched by this rule.
2. An existing `payment_request` can become `credential_request` when its literal evidence begins with an explicit account-password request and contains no payment cues. Network-password and negated/historical/educational contexts are excluded. The rule does not add signals to every password mention or repair invalid model output.

`analysis.analyze_text()` and the RAG/app flow use the review. Raw `LocalRuntime.analyze()`, training selection and raw model evaluation remain unchanged. RAG JSON records `raw_observation`, reviewed `observation`, and `analysis_adjustments`, so a model improvement cannot be confused with a rule correction.

## Actual local-model results

Reproduction:

```bash
python -m easy_tech_help.rag_evaluation --device mps --output results/rag_improved.json
python -m easy_tech_help.review_evaluation
```

The eight inputs are the same assistant-authored development cases used to debug the initial connection. Expectations were strengthened to check exact source-unit identity, relevant topic phrases, the targeted classification corrections and preservation of the Wi-Fi past/present contrast. They are **not independent unseen evaluation**.

| Check | Initial connection | Current application |
| --- | --- | --- |
| Expected response status | 7/8 | 8/8 |
| Cited expected document on answered cases | 5/5 | 5/5 |
| Displayed explanation is literal selected source text | Not enforced; free-form paraphrases | 5/5, enforced by rendering |
| Password-request example | Model mislabeled request as payment | Corrected to credential request; original mistake retained in trace |
| Contextless help example | Classified as a message | Corrected to unknown / insufficient context |
| All current checks, including topic/context checks | Not measured by the old checker | 8/8 |

Current outputs: five answers, two explicit insufficient-evidence results, one uncertain-analysis result. Mean time per case after loading was **2.808 seconds**, range **1.154–4.452 seconds**. These are single local development measurements, not a service latency guarantee. Raw generations, exact excerpts, correction reasons, versions and source/code/adapter fingerprints are in [rag_improved.json](rag_improved.json).

### Review of the five displayed passages

| Input | What is now shown | Remaining interpretation limit |
| --- | --- | --- |
| Connected Wi-Fi, no internet | Apple-based summary's complete No Internet Connection sentence | General troubleshooting quotation, not a diagnosis or approved personalized action |
| Wi-Fi off | Apple-based summary's sentence about checking that Wi-Fi is on | Does not infer an additional cause |
| Virus popup requesting a call | FTC passage describing bogus warnings and a fake popup urging a phone call, with the antecedent retained | Describes a scam pattern; does not prove this specific popup is fraudulent |
| Password request | FTC spam-text passage explicitly mentioning passwords and personal information | Does not establish sender identity |
| Unsecured public Wi-Fi | Entire FTC past/present contrast, including current website encryption | General statement about public Wi-Fi; cannot certify this particular network as safe |

Review was by the coding assistant against the saved local sources. Exact source identity is mechanically checked; keyword matches and this review do not replace independent semantic relevance or safety evaluation.

## Regression and automated verification

[observation_review.json](observation_review.json) replays the new rules over **68 previously recorded raw predictions**: 34 validation and 34 regression examples. No prediction changed in these sets; observation matches remain **20/34** and **23/34**. This demonstrates unchanged behavior on those recorded examples, not improved general model accuracy or a fresh GPU evaluation. The known public-SMS misses remain unresolved.

```text
python -m pytest -q       135 passed, 7 opt-in live tests skipped
ruff check .             passed
ruff format --check .    passed
git diff --check         passed
```

The eight-case MPS evaluation ran separately. Automated tests include explicit password/payment controls, normal messages, negated and historical wording, Wi-Fi password exclusions, contextless and contextual requests, source selection validation, retention of qualifications and temporal contrast, retrieval specificity, CLI review integration and Streamlit rendering.

## Limits and next work

- Classification rules correct two observed failure types; they do not fix general model recall or all ambiguous wording. Broader gains need reviewed training examples and independent evaluation.
- Extractive wording is less conversational. Better paraphrases would need a separate faithfulness evaluation before replacing this constrained output.
- Lexical retrieval can miss synonyms or select an irrelevant passage. The corpus is still six FTC originals and four Apple-based summaries.
- Quoted source instructions have not passed a personalized action-safety policy. That policy, its normal/dangerous controls, and family handoff remain the next product work.
