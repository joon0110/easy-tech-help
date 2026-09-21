# Architecture — text V1

V1 accepts English text and uses local PyTorch inference and text fine-tuning. Screenshots are outside the active scope.

## Current executable flow

```text
Streamlit text area / CLI --text or --file
  -> validate_input (nonempty text, at most 4,000 characters)
  -> shared prompt + Qwen2.5-1.5B-Instruct + trained PEFT adapter
  -> completed JSON response
  -> TextObservation validated against the original input
  -> conservative application review, retaining raw prediction and correction reasons
  -> category-filtered local documents and BM25-ranked sentence chunks
  -> same Qwen base model with extraction adapter temporarily disabled
  -> generated selection of one numbered source unit (or abstention)
  -> validated IDs, literal explanation and official URL resolved from the corpus
  -> deterministic safety policy and independently checked action support
  -> summary, cautions, allowed next steps, avoid actions and sources
```

`analysis.py` lazily caches a `LocalRuntime`. Transformers and PEFT load local files only; inference needs no model server or API key. Runtime checks the completed training manifest, model revision and prompt fingerprint before loading the adapter. It does not silently substitute the base model. The model has no tools and cannot browse, follow links or operate a phone. The app does not save inputs.

Input is JSON-encoded in a user message. Embedded role markers remain untrusted input; this is not a prompt-injection guarantee. Pydantic rejects unknown fields, duplicate signals, incompatible connection states and evidence absent from the input. Any issue forces `unknown` and clears signals. Incomplete or invalid generations also become `unknown`. Exact quotation cannot prove that an interpretation is correct. Displayed summaries and uncertainty are application templates excluded from training targets.

## Intended complete pipeline

```text
Text -> validated observations
     -> local document retrieval
     -> explanation grounded in retrieved evidence
     -> deterministic safety rules and allowed actions
     -> explanation, cautions, next action and sources
     -> optional family handoff from the approved result
```

`rag.py` implements retrieval-augmented explanation. It maps `alert` to knowledge category `popup` and expands the original input with terms for observed signals. `retrieval.py` filters documents by category and weighted title/tag/body keywords, then ranks literal sentence-aware chunks with BM25 (k1=1.5, b=0.75) plus 0.05 times the document score. It groups adjacent short paragraphs and applies simple English plural normalization to search terms. Chunk IDs contain document ID, ordinal and content hash. The target size is 75 words; a longer sentence remains intact to retain conditions and consequences. At most three candidates are retrieved, at most two per document. Only the highest-ranked excerpt enters the explanation prompt: the small model mixed source facts when supplied with several excerpts but asked to cite only one. Other candidates remain in the trace and fallback reference display. The ten-document corpus is rebuilt in memory for each request; there is no persistent vector index or online fetch.

Original query terms have weight 1.0 in chunk scoring; added signal terms have weight 0.2. Explicit password/passcode/verification/credential terms in the original input have weight 3.0. Apple Account message material is excluded from generic account queries without Apple/iCloud context. These are transparent lexical heuristics, not semantic embeddings. The original document-ranking API is retained.

The fine-tuned adapter was trained for extraction JSON. `generate_grounded()` temporarily disables it using PEFT's context manager while reusing the loaded base weights for source selection. A runtime lock serializes extraction and selection so concurrent sessions cannot observe the wrong adapter state. The context manager restores the adapter on errors. Source selection is greedy, with repetition penalty 1.1 and a 384-token cap; observation extraction retains penalty 1.0 and 256 tokens. Both share a 4,096-token context budget and fail explicitly rather than truncating input silently.

Free-form explanation output was replaced by an extractive contract after observed unsupported paraphrases. The model returns `source_id`, one strict integer `sentence_id`, and strict boolean `insufficient_evidence`. It receives numbered literal source units instead of an instruction to rewrite source text. Units retain complete sentences; immediate dependent continuations and an adjacent past/present contrast stay together (at most 120 words). Units are ranked by original-input term overlap, with generic terms weighted 0.2 and other terms 1.0. The highest-ranked usable candidate supplies the units. The application resolves the selected unit verbatim and rejects invented IDs, arrays/booleans as IDs, extra prose, inconsistent abstention and malformed output. An enclosing JSON markdown fence is tolerated. The product applies the stage 5 display guard before showing "What the reference says"; surrounding passages remain in diagnostic traces, while official links are available in the UI. Source URLs are resolved from the catalog and Apple summaries remain labeled. Raw selection output is recorded in CLI/development traces, never rendered as the explanation.

This is extractive RAG, not successful verification of arbitrary generated paraphrases. Every displayed explanation is copied from the selected source; relevance, context and source accuracy still need independent review. General source instructions are not personalized approved actions. Unknown analysis, missing/unreadable evidence, explicit abstention, incomplete output, invalid IDs and runtime failures remain distinct states. Related references can be inspected after a failure. User messages and training data never enter the trusted corpus automatically.

### Application observation review

`observation_review.py` applies two bounded corrections after schema validation: a whole-input generic help request becomes `unknown/insufficient_context`; an existing `payment_request` whose evidence starts with an explicit account-password request can become `credential_request` if payment/network/context exclusions do not apply. It never infers requests from every password mention, never repairs invalid model JSON, and skips negated/historical/educational contexts. URL text is excluded from the context-word check so `.example` domains do not count as educational examples. Replacement evidence remains an exact input span.

`analysis.analyze_text()` and `rag.explain_text()` apply these rules. `LocalRuntime.analyze()`, the training validator and raw model evaluator remain unchanged; raw training metrics are not inflated by application rules. RAG traces preserve `raw_observation`, the reviewed `observation`, and `analysis_adjustments`. Rules are conservative English heuristics, not a complete semantic classifier or a replacement for better training data.

## Stage 5 action policy

`guidance.py` is the product API/CLI and wraps `explain_text()` with `safety.build_guidance()`. The original RAG/extraction entrypoints remain diagnostic. Training, model weights and model prompts are unchanged by this stage.

`safety.py` keeps frozen action templates in a read-only catalog. The policy emits one of `attention`, `check_source`, `connection_check`, `no_specific_warning` or `uncertain`; none is a certified scam/safe verdict. Sensitive account/payment/remote requests take priority over connectivity checks. Unknown analysis requests context; explicit sensitive input cues can still cause a pause when model output is unknown. Network-password and receipt controls prevent two obvious semantic misclassifications. Clause-local negation and historical/educational checks reduce false alarms, while regex guards catch several explicit requests missed by the model. These are conservative heuristics, not a complete semantic parser.

The policy selects action IDs, never model prose. `resolve_actions()` rejects unknown/prohibited IDs, reads the bound local reference, verifies an exact support span and the official URL allowlist, and returns the fixed text plus `ActionEvidence`. The bindings were reviewed during implementation. They are deterministic source lookups, not the same excerpt selected by the explanatory model. Missing or modified support causes a pause/clarification fallback. `pause`, `clarify` and `no_change` are explicit application fallbacks with no external citation; their trace status is `policy_only`.

Allowed Wi-Fi actions check current state, compare another device on the same known network, or obtain a known network's password from its owner for use in Settings. Airplane Mode changes require understanding whether the state is intentional. The policy does not reset/forget networks, erase the device, disable security, join unknown networks, forward credentials, pay a requester, use message links/numbers, or grant remote access. There is no phone automation; all actions are instructions for the user to consider.

`displayable_reference()` checks that the explanation still equals literal retrieved citations and hides passages mentioning high-impact operations or broad public-network safety assurances. The app offers source titles and links instead of full unreviewed procedural passages. This additional conservative display filter can suppress helpful quotes too; the fixed action catalog is the enforcement boundary, not this word filter. Diagnostic JSON retains raw source/model data for inspection and must not be used directly as a product action panel.

`safety_evaluation.py` runs twelve local-model development cases through analysis, RAG and policy. It checks decision levels, appropriate actions, template identity, prohibited-action exclusion and literal source support. Unit tests cover missing/altered sources, normal/negated/historical cases, all prohibited IDs and injection through input/model/source content. These checks constrain output behavior; independent real-world classification, relevance and usability evaluation is still necessary. See [stage 5 results](../results/safety.md).

Family summaries remain a later step and must use only the reviewed result, excluding private credentials and unnecessary personal details.

## Data and provenance

`data/text/{train,validation,test}.jsonl` contains 200 examples: 120 revised synthetic records and 80 redacted public UCI SMS records. The splits contain 132/34/34 examples. Public records include 40 ham and 40 spam; their source labels are provenance, not fraud verdicts or model targets. Synthetic cases provide Wi-Fi, alert and adversarial coverage absent from the SMS source.

Every row includes a correct observation, a rejected alternative, rationale, provenance, review status and scenario group. Public records additionally identify the original source row and hash. `sms_source.py` verifies them against the checksum-pinned original ZIP and reproduces redactions. Attribution and the original author notice are in `data/sources/uci_sms/`. Review is automated assistant review, not independent human/domain review.

`dataset.py` checks schemas, literal evidence, references, metadata, duplicate IDs/text/source rows and scenario groups across splits. It rejects cross-split normalized word-sequence similarity of at least 0.85, or 0.75 for two public SMS records. One repeated prize campaign stays entirely within train. Literal held-out inputs in the system prompt are rejected. These checks cannot establish factual correctness or complete semantic independence.

SFT export contains only system/user/correct-assistant messages from train and validation. Rejected answers, explanations, source labels and test examples are excluded. Token checking includes the complete chat serialization and verifies the assistant boundary without truncation. Maximum sequence lengths are 630/612/610 tokens for train/validation/test, below the 1,536-token training limit. `preparation_report.json` is a preparation-stage snapshot; training and evaluation have separate reports.

## PyTorch training

The tested machine is an Apple M4 with 10 CPU cores and 16 GB unified memory. MPS is available outside the restricted execution sandbox.

| Setting | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-1.5B-Instruct, Apache-2.0 |
| Pinned revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Libraries | PyTorch, Transformers, PEFT; Accelerate installed for PEFT |
| Device | MPS; CPU supported |
| Precision | BF16 frozen base on MPS, FP32 adapters; FP32 base on CPU |
| LoRA | `q_proj` and `v_proj`, rank 8, alpha 16, dropout 0.05 |
| Trainable parameters | 1,089,536 |
| Microbatch / accumulation | 1 / 4 |
| Optimizer | AdamW, learning rate 2e-4, weight decay 0.01 |
| Schedule | 5 epochs, 10% warmup, linear decay, gradient clipping 1.0 |
| Loss | Assistant completion tokens only; accumulation weighted by token count |
| Memory control | Gradient checkpointing, cache disabled, project only supervised logit positions |
| Selection | Lowest validation loss |
| Reproducibility | Seed 42, model/data/prompt hashes, versions and run history |
| Inference | Greedy decoding, explicit repetition penalty 1.0, at most 256 new tokens, 4,096-token context budget |

`training.py` verifies the downloaded weights checksum, uses only train for gradient updates and validation for checkpoint selection, checks finite loss/gradients, and saves the best adapter plus a manifest. Test token lengths and integrity are checked, but test sequences never enter model forward/backward passes during training. A tiny real PyTorch test checks the optimized completion loss against full masked causal loss. GPU runs are not guaranteed bit-for-bit identical across platforms.

The table above describes initial training. The current default adapter is a three-epoch continuation at learning rate 5e-5, with epoch 2 selected; see [measured results](../results/improvement.md). For continuation, `--init-adapter` loads completed LoRA weights and starts a fresh optimizer/scheduler; it does not resume optimizer state. It requires unchanged prompt and data fingerprints. `--decision-weight 4` increases the loss weight of category values, signal names, issue lists and signal-list opening/closing/continuation tokens. Evidence tokens otherwise keep weight 1. Weights are assigned with the tokenizer's character offsets, and accumulation divides by the total supervised weight. Validation loss remains unweighted for comparison.

`--select-by-generation` compares actual validation outputs before training and after each epoch. Selection prioritizes observation matches, then fewer ordinary cases with extra signals, higher signal F1 and lower validation loss. The initial adapter is a candidate, so a worse continuation does not automatically replace it. The manifest records generated validation metrics, selection settings and the starting adapter hash. Teacher-forced loss alone can hide missing signals during generation.

## Evaluation

`evaluation.py` runs the same base and adapter with identical prompt, tokenizer, decoding and validation on a frozen split. It verifies split fingerprints against the training manifest and records raw generations, effective generation configuration and per-case outputs. Metrics include valid outputs, category accuracy, observation matches, signal precision/recall/F1, extra signals on ordinary controls and latency, with category and provenance breakdowns. The existing test split has already been inspected and is now reported as regression data; validation is development data.

Extraction disables the pinned model's default repetition penalty of 1.1 by explicitly setting 1.0. Repeated signal names, JSON keys and exact input quotes are expected in this task. In Transformers 5.17.0, unspecified generation fields inherit model defaults, so setting only `do_sample=False` does not remove this penalty. A tiny real PyTorch generation test verifies that the extraction settings preserve raw greedy token scores even when the model carries chat defaults. The first evaluation predates this fix; it remains a historical run, not a measurement of current inference settings.

An observation match requires a valid response and correct category, signal set and issue set. Different literal evidence spans can match; quote validation does not establish semantic grounding. Extra-signal rate is not a scam false-positive rate. `rag_evaluation.py` runs eight assistant-authored development inputs, checking statuses, relevant document IDs, source-unit identity, topic phrases, the two targeted classification corrections and preservation of the Wi-Fi past/present contrast. These are development checks, not independent semantic relevance metrics. `review_evaluation.py` separately replays application rules against the 68 previously recorded validation/regression predictions. See [current review](../results/rag_improvement.md) and [historical initial connection](../results/rag.md). Independent relevance review and action-safety evaluation remain necessary.

Keep test failures for reporting; do not tune on them and continue calling them unseen. This small curated public/synthetic test is only an initial check. UCI texts may have appeared in base-model pretraining, and stronger evaluation needs independently reviewed contemporary messages. Final training metadata and evaluation outputs belong in `results/`; large local weights remain ignored under `artifacts/`.

## Responsibilities

| File / directory | Responsibility |
| --- | --- |
| `app/app.py` | Text input, observations, cautions, next/avoid actions, filtered references and explicit errors |
| `analysis.py`, `config.py` | Shared prompt, app/CLI integration and local settings |
| `runtime.py` | Local base/adapter loading and PyTorch generation |
| `schemas.py` | Input and observation validation |
| `observation_review.py`, `review_evaluation.py` | Narrow application corrections and replay of recorded raw predictions |
| `dataset.py`, `sms_source.py` | Dataset integrity, source verification and SFT export |
| `training.py` | Completion-only LoRA training and validation selection |
| `evaluation.py` | Matched base/adapter evaluation |
| `data/text/`, `data/sources/` | Labeled examples, provenance and licensing |
| `retrieval.py`, `knowledge/` | Local corpus, literal chunks and BM25 search |
| `rag.py`, `rag_evaluation.py` | Retrieval-augmented explanations, citation validation and development checks |
| `safety.py`, `guidance.py`, `safety_evaluation.py` | Fixed action policy, source checks, product API/CLI and product development evaluation |
| `results/` | Training metadata and measured evaluation results |
| `artifacts/` | Ignored downloads, SFT exports and model weights |

The initial action safety policy is implemented. Family handoff, independent evaluation and further usability work remain; model recall and semantic relevance still need improvement.
