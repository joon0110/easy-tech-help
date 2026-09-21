# PyTorch analysis improvement

Date: 2026-09-21. This is a development improvement, not a claim of reliable scam detection or completed RAG guidance. The input data and analysis prompt are unchanged.

## What changed

1. **Explicit extraction decoding.** The original runtime inherited `repetition_penalty=1.1` from Qwen's `generation_config.json`. The pinned Transformers version fills unspecified generation settings from model defaults. Extraction now explicitly sets the penalty to 1.0, uses greedy decoding, and records the effective configuration. Exact quotes and repeated signal names should not receive a repetition penalty. [Transformers generation documentation](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/text_generation) describes these controls.
2. **Decision-weighted continuation.** PyTorch + PEFT continued the original LoRA on the same 132 training examples for three epochs / 99 updates, with learning rate 5e-5 and weight 4 for category/signal/issue decisions. Evidence tokens otherwise retain weight 1. The frozen base, rank 8, microbatch 1, accumulation 4 and MPS BF16/FP32 setup remain the same. AdamW and the scheduler start fresh; optimizer state is not resumed.
3. **Checkpoint selection from actual answers.** Validation generations, rather than loss alone, determine selection. The order is observation matches, fewer ordinary examples with extra signals, higher signal F1, then lower validation loss. The starting adapter is also a candidate. Epoch 2 was selected; epoch 3 was worse. The continuation took 1,024.6 seconds (17.1 minutes), including validation and excluding loading.
4. **Regression protection.** Tests verify weighted causal loss against a manual calculation, character-offset decision spans, answer-based selection and unmodified greedy token scores despite inherited model defaults. Evaluation now records effective decoding and completion status, and identifies the inspected test split as regression data.

The initial weights, prompt and split fingerprints are recorded in [training_manifest_v2.json](training_manifest_v2.json). No new SMS or real iPhone screens were collected for this continuation. The current application defaults to `artifacts/pytorch-adapter-v2`; the first adapter remains available for reproduction.

## Development results

The same 34 validation examples were used throughout. They guide selection and are **not independent final evidence**. Reloading the saved selected adapter reproduced every non-latency validation metric from training.

| Metric | Original adapter + original decoding | Original adapter + corrected decoding | Selected continuation + corrected decoding |
| --- | ---: | ---: | ---: |
| Valid outputs | 33/34 | 34/34 | 34/34 |
| Correct category | 30/34 | 31/34 | 31/34 |
| Full observation matches | 17/34 | 20/34 | 20/34 |
| Signal precision | 83.3% | 73.1% | 88.9% |
| Signal recall | 15/27 (55.6%) | 19/27 (70.4%) | 16/27 (59.3%) |
| Signal F1 | 0.6667 | 0.7170 | 0.7111 |
| Ordinary examples with extra signals | 1/14 | 3/14 | 1/14 |

Corrected decoding alone increased recall but also false signals. Continuation improved precision and reduced ordinary-control errors relative to that configuration, while giving up some recall. It did not improve every metric. Against the original app settings, exact observations improved by three cases and signal recall by one signal. This is a modest gain on a small development set.

Raw results: [before](pytorch_validation_before.json), [decoding fix only](pytorch_validation_decoding_fix.json), [selected saved continuation](pytorch_v2_validation.json). The first two were produced before evaluation metadata was expanded; their raw predictions and metrics are preserved.

### Remaining validation failures

- Visible links, support numbers and premium-rate payment requests are still missed, especially in historical public SMS.
- A failed subscription renewal was incorrectly given an urgent security warning.
- A hidden-network setup description was incorrectly treated as an established connection. This case is outside the `ordinary` control denominator, so 1/14 does not summarize all false signals.
- Some unsupported or incomplete descriptions receive a message classification.
- An ordinary accessibility installation alert was rejected as unsupported.

Quote presence is not semantic correctness. An empty signal list is not proof of safety. Both Wi-Fi and popup/alert training examples remain synthetic; real device text and independent label review are still needed.

## Existing test set: regression results

After selection on validation, the saved adapter was evaluated once on the existing 34 test examples. This set was inspected during the first run, so these results are **regression evidence, not a new independent held-out claim**. No further prompt, label or model tuning followed this evaluation. Raw outputs and the completed-run marker are in [pytorch_v2_regression.json](pytorch_v2_regression.json).

| Metric | Original app (old decoding) | Base model (corrected decoding) | Selected continuation (corrected decoding) |
| --- | ---: | ---: | ---: |
| Valid outputs | 30/34 | 19/34 | 32/34 |
| Correct category | 26/34 | 7/34 | 30/34 |
| Full observation matches | 17/34 | 2/34 | 23/34 |
| Signal precision | 85.7% | 36.8% | 88.2% |
| Signal recall | 6/22 (27.3%) | 7/22 (31.8%) | 15/22 (68.2%) |
| Signal F1 | 0.4138 | 0.3415 | 0.7692 |
| Ordinary examples with extra signals | 0/12 | 1/12 | 0/12 |

The current base/adapter columns share the exact decoding configuration. The historical column differs in both adapter version and repetition penalty, so its improvement is the combined runtime/training effect, not training alone. Current adapter generation averaged 1.788 seconds per case; this is a single local run, not a performance guarantee.

The gains are concentrated in synthetic cases: observation matches improved from 6/20 to 12/20, and the current Wi-Fi group matches 5/6 descriptions. **Public SMS observation matches remain 11/14 and all four labeled public-SMS signals are still missed.** Their many no-signal answers must not be presented as successful real scam detection. This is a major unresolved limitation.

Two responses failed validation. Other remaining errors include missing verification-code/link requests, gift-card redemption codes mistaken for account credentials, alerts labeled as messages, and a role-injection example incorrectly marked as containing a visible URL. The validator rejects malformed/unsupported structures but cannot prove semantic grounding. No detected errors among 12 ordinary controls is too small a sample to establish real-world safety.

## Reproduce

First obtain the original adapter using the root README, then:

```bash
python -m easy_tech_help.training --device mps --init-adapter artifacts/pytorch-adapter --output artifacts/pytorch-adapter-v2 --epochs 3 --learning-rate 0.00005 --decision-weight 4 --select-by-generation
python -m easy_tech_help.evaluation --device mps --adapter-only --split validation --output results/pytorch_v2_validation.json
python -m easy_tech_help.evaluation --device mps --split test --output results/pytorch_v2_regression.json
```

Use a new output directory and corresponding `--adapter-dir` for a repeated training run; existing weights are not overwritten. The unit suite passes 68 tests, with seven opt-in live tests skipped; Ruff passes.

## Next: connect RAG with action safety

The existing local corpus and retriever can support a development prototype next. Map `alert` to the corpus category `popup`, retrieve relevant evidence, generate explanations with document IDs, and validate citations against the retrieved documents. Keep Apple-based summaries labeled as summaries. Handle missing evidence and uncertain analysis explicitly.

Add code-controlled allowed actions before showing next-step advice. RAG cannot establish sender authenticity or repair every missed signal. Test normal messages, Wi-Fi failures, dangerous requests and missing sources through the complete pipeline, then add family handoff from the approved result. A new independently reviewed final evaluation set is needed before claiming real-world reliability.
