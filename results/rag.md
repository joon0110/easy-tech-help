# Local RAG connection — development results

Run: 2026-09-21, Apple M4 / MPS, PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.0.

## Implemented

The Streamlit app and `python -m easy_tech_help.rag` now execute:

```text
English text -> trained PyTorch observation model -> validated observations
            -> local reference retrieval -> base-model explanation
            -> checked source ID -> literal excerpt and official catalog URL
```

The existing extraction adapter is temporarily disabled only during explanation generation; the same base weights remain in memory. No retraining or new model download was performed for this stage. A lock serializes model access and PEFT restores the adapter after failures. The extraction prompt, training data and adapter weights are unchanged.

Six original FTC HTML articles and four labeled Apple-based summaries produce **100 literal chunks**. Short adjacent paragraphs are grouped for context, then split on sentence boundaries with a 75-word target; a longer sentence stays intact. Retrieval uses category filtering, weighted document keywords, simple English plural normalization and BM25 chunk ranking. Three candidates at most are retained in the trace; only the top candidate is supplied for the short explanation. No API key, live website request, embeddings service or vector database is used.

The app renders generated prose and input quotes as plain text. Clickable sources come from the trusted catalog. Unknown analysis, absent evidence, explicit model abstention, invalid response/source ID, unfinished generation and runtime/file errors are handled explicitly. Rejected drafts are not displayed as answers. Apple summaries are labeled as summaries rather than original Apple pages.

## Actual local-model checks

Reproduce from the repository root with the existing model and adapter:

```bash
python -m easy_tech_help.rag_evaluation --device mps --output results/rag_development.json
```

[Full trace](rag_development.json) includes inputs, expected document IDs/statuses, analysis, retrieved excerpts, raw generated output, displayed explanation, source links, timings and source/code/adapter fingerprints. These eight cases are assistant-authored development inputs used during implementation and prompt debugging. They are **not an unseen benchmark or an accuracy claim**. Prompt demonstrations are separate from these cases; both belong to development, not independent evaluation.

| Case | Actual result | Cited document | Review |
| --- | --- | --- | --- |
| Connected Wi-Fi, no internet | Answered | `wifi_iphone` | Summarizes relevant troubleshooting facts, but is generic rather than explaining the distinction between joining and internet access |
| Wi-Fi off | Answered | `wifi_iphone` | Related excerpt; generated wording infers a cause rather than just reporting the setting |
| Virus popup requesting a call | Answered | `popup_tech_support` | Includes the source's useful fact about genuine warnings not requesting calls; also says the excerpt discusses reporting, which it does not |
| Message requesting a password | Answered | `message_phishing` | Relevant document, but selected excerpt does not mention passwords; explanation overstates its support |
| Unsecured public Wi-Fi | Answered | `wifi_public_safety` | Paraphrases the FTC passage on modern website encryption; the broad safety wording must not become a verdict that this particular network is safe |
| Ordinary book-club message | Insufficient evidence | None | No unsupported explanation generated |
| Unrelated astronomy alert | Insufficient evidence | None | No unsupported explanation generated |
| Missing context | Insufficient evidence | None | Expected uncertain analysis; analyzer instead returned `message` with no signals. Retrieval found nothing, so no answer was generated |

- **7/8** cases matched all mechanical checks: expected status, expected-document retrieval/citation and citation provenance.
- **5/5** answered cases cite a real retrieved excerpt from an expected document. This is a provenance result, **not 5/5 factually correct explanations**.
- All three negative cases produced no generated explanation. The missing-context case still failed the expected-status check.
- End-to-end time per case after model loading: **1.316–7.747 seconds**, mean **4.728 seconds**. Hardware and cache state affect timing; these are not service latency guarantees.
- The evaluation command currently exits **1** to preserve the known missing-context failure. Its expected label has not been relaxed to hide the error.

Review above was performed by the coding assistant against the saved excerpts, not an independent human/domain reviewer. No semantic entailment score is claimed. A source ID can pass structural validation while an explanation remains inaccurate.

## Verification

```text
python -m pytest -q       99 passed, 7 opt-in live tests skipped
ruff check .             passed
ruff format --check .    passed
git diff --check         passed
```

The real eight-case MPS evaluation above ran separately from pytest. Unit/integration checks cover literal/stable chunks, retrieval categories, missing evidence, invalid/fabricated citations, candidates excluded from model context, strict JSON validation, incomplete/runtime failures, source URL allowlisting, real PEFT adapter restoration after an error, and Streamlit source/summary/fallback rendering. They do not establish resistance to every prompt injection or correctness of generated claims.

## Next improvements after this RAG connection

1. Improve retrieval and grounding: prefer excerpts supporting the actual sensitive request, reduce query distortion from incorrect model signals, and evaluate generated claims against the selected excerpt. The password case is a concrete failure to address.
2. Improve uncertain-input and request classification. The password example was incorrectly analyzed as a payment request; RAG does not correct that automatically. Preserve these development failures and reserve independent examples for later evaluation.
3. Implement the separate application safety policy before presenting next actions: reviewed action IDs/templates, consequences, prohibited instructions, uncertainty handling and normal controls. Prompt requests to avoid actions are not enforcement.
4. Add family handoff only after the approved guidance result exists. Obtain independent real iPhone/popup examples and contemporary messages for full-pipeline evaluation.

**Stage 4 wiring is implemented; dependable grounding and the final safe-guidance product are not complete.** This report records the actual limitations rather than treating correct citation formatting as reliable advice.
