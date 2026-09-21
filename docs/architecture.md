# Architecture — text V1

V1 accepts English text and uses local PyTorch inference and text fine-tuning. Screenshots are outside the active scope.

## Current executable flow

```text
Streamlit text area / CLI --text or --file
  -> validate_input (nonempty text, at most 4,000 characters)
  -> shared prompt + Qwen2.5-1.5B-Instruct + trained PEFT adapter
  -> completed JSON response
  -> TextObservation validated against the original input
  -> template summary, uncertainty and quoted evidence
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

The independent retriever uses weighted keyword overlap on local documents. Connecting it requires mapping `alert` to knowledge category `popup`, creating useful queries from supported signals, validating citations and handling missing evidence. User messages and training examples are not trusted RAG documents. Retrieved text cannot override application safety rules.

The remaining action policy needs reviewed action IDs/templates, uncertainty handling, and tests for both missed concerning requests and false alarms. A Wi-Fi problem alone is not a scam. Network resets need consequence-aware handling. Family summaries must exclude private credentials and unnecessary personal details.

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

An observation match requires a valid response and correct category, signal set and issue set. Different literal evidence spans can match; quote validation does not establish semantic grounding. Extra-signal rate is not a scam false-positive rate. Full RAG relevance and action-safety metrics require the remaining pipeline.

Keep test failures for reporting; do not tune on them and continue calling them unseen. This small curated public/synthetic test is only an initial check. UCI texts may have appeared in base-model pretraining, and stronger evaluation needs independently reviewed contemporary messages. Final training metadata and evaluation outputs belong in `results/`; large local weights remain ignored under `artifacts/`.

## Responsibilities

| File / directory | Responsibility |
| --- | --- |
| `app/app.py` | Text input, observations, explicit errors |
| `analysis.py`, `config.py` | Shared prompt, app/CLI integration and local settings |
| `runtime.py` | Local base/adapter loading and PyTorch generation |
| `schemas.py` | Input and observation validation |
| `dataset.py`, `sms_source.py` | Dataset integrity, source verification and SFT export |
| `training.py` | Completion-only LoRA training and validation selection |
| `evaluation.py` | Matched base/adapter evaluation |
| `data/text/`, `data/sources/` | Labeled examples, provenance and licensing |
| `retrieval.py`, `knowledge/` | Independent local reference search and corpus |
| `results/` | Training metadata and measured evaluation results |
| `artifacts/` | Ignored downloads, SFT exports and model weights |

RAG-connected generation, action safety rules and family handoff remain to be implemented.
