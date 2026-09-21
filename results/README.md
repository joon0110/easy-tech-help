# First PyTorch text fine-tuning run

This is the historical first run from commit `38d1925`. See [the current improvement report](improvement.md) for the selected continuation. The first run's generation inherited the pinned model's `repetition_penalty=1.1`; specifying greedy decoding alone did not disable it. Current inference explicitly uses 1.0. Re-running the commands below with current code therefore does not reproduce these historical generation metrics exactly. The saved raw results remain unchanged.

Run date: 2026-09-21. **Training completed; the model is not ready for reliable user guidance.** This report evaluates observation extraction only. RAG-connected explanations, action safety and family handoff are not implemented yet.

## Data

200 labeled examples combine 120 revised synthetic cases with 80 redacted records from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 40 ham and 40 spam. Public spam includes advertising and is not a verified scam label. Synthetic data covers Wi-Fi, alerts and adversarial inputs. Normal examples are necessary controls; only correct analysis targets enter supervised training.

| Split | Synthetic | Public SMS | Total |
| --- | ---: | ---: | ---: |
| Train | 80 | 52 | 132 |
| Validation | 20 | 14 | 34 |
| Test | 20 | 14 | 34 |

Source hashes, redactions, labels, references, token lengths and split separation passed executable checks. The repeated public prize campaign stays in train. Labels have assistant review, not independent human review. Source labels and rejected answers are excluded from model targets. Public texts may have occurred in base-model pretraining.

## Training

Qwen2.5-1.5B-Instruct was trained locally with **PyTorch 2.14.0, Transformers 5.17.0 and PEFT 0.21.0** on Apple M4 MPS. Accelerate 1.15.0 is installed for PEFT; the training loop uses native PyTorch. LoRA updates 1,089,536 parameters in `q_proj`/`v_proj` (rank 8, alpha 16, dropout 0.05). The base stays frozen in BF16 and adapters use FP32. Training uses completion-only causal loss, learning rate 2e-4, microbatch 1, accumulation 4 and seed 42.

Five epochs / 165 updates took **1,291.5 seconds (21.5 minutes)**, including validation but excluding model loading. Epoch 4 was selected by validation loss. The final epoch's loss was slightly higher; test outputs were not used for checkpoint selection. The final MPS allocation recorded in the manifest is not a peak-memory measurement.

| Epoch | Train loss | Validation loss |
| --- | ---: | ---: |
| Before training | — | 0.26675 |
| 1 | 0.16757 | 0.11278 |
| 2 | 0.08644 | 0.07579 |
| 3 | 0.05435 | 0.06405 |
| **4 (selected)** | **0.03793** | **0.05867** |
| 5 | 0.02995 | 0.05930 |

The complete settings, versions, hashes and history are in [training_manifest.json](training_manifest.json). Weights remain local under `artifacts/pytorch-model` and `artifacts/pytorch-adapter`; they are not committed.

## Frozen test comparison

The base and selected adapter used the same prompt, tokenizer, greedy decoding with the inherited repetition penalty, 256-token output budget and validator. These were the frozen settings at the time of the first evaluation. Case-level generations and metrics are in [pytorch_test.json](pytorch_test.json). This split has since been inspected and is now a regression set for subsequent changes.

| Metric | Base | Trained adapter |
| --- | ---: | ---: |
| Outputs passing validation | 5/34 (14.7%) | 30/34 (88.2%) |
| Category correct | 4/34 (11.8%) | 26/34 (76.5%) |
| Full observation matches | 2/34 (5.9%) | 17/34 (50.0%) |
| Signal precision | 0.0% | 85.7% |
| Signal recall | 0/22 (0.0%) | 6/22 (27.3%) |
| Signal F1 | 0.0000 | 0.4138 |
| Extra signals on ordinary controls | 0/12 | 0/12 |
| Mean generation time | 1.654 s | 1.599 s |

An observation match requires valid category, signal set and issue set. Evidence must be a literal input quote, but alternate quote spans can match. Category accuracy alone can count an invalid response mapped to `unknown`; full observation matches cannot. Invalid outputs cause abstention rather than successful safe advice. The base failed validation on 29/34 inputs, so its zero ordinary extra-signal rate is not evidence of reliable behavior. Neither model's 12 ordinary controls establish a real-world false-alarm rate.

Adapter results differ sharply by source: synthetic observations match **6/20**, public SMS **11/14**. Every public SMS was classified as `message`, but the model missed all four labeled public-SMS signals. Many public messages have no defined signal; a high match rate on that subset does not establish scam detection.

### Observed failures

- Gift-card payment, verification-code and password requests were missed.
- Paid premium-call requests and a public SMS support number were missed.
- `wifi_recovered_state_en` incorrectly linked `no_internet` to the quote “the internet works.” Literal evidence checks cannot detect this semantic error.
- A support-number quote changed `Our` to `our`, causing the strict evidence validator to reject the whole response.
- Some Wi-Fi/alert cases were classified as messages; four responses failed output validation.
- Unknown/contradictory cases did not produce the correct issue type.

**The main remaining problem is missed signals, not training execution.** Improvement should use new, independently reviewed signal-bearing and normal examples; validation should guide subsequent changes. Keep this test as a reported regression set, and reserve a new independent final set for the next model version. Do not relabel evaluated failures to inflate results. Add deterministic action safety before offering advice; absence of model signals must never mean “safe.”

## Reproduce and inspect

Follow the root README to install dependencies and download the pinned model. From the repository root:

```bash
python -m easy_tech_help.dataset --check-tokens
python -m easy_tech_help.sms_source
python -m easy_tech_help.training --device mps --output artifacts/pytorch-adapter
python -m easy_tech_help.evaluation --device mps --split test --output results/pytorch_test.json
python -m pytest -q
ruff check .
```

Source verification requires the original ZIP download documented in `data/sources/uci_sms/README.md`. Training refuses to overwrite a nonempty output directory; use a new directory and the matching `--adapter-dir` when repeating a run. Seed and fingerprints support reproducibility, but cross-platform GPU execution is not guaranteed bit-identical. The current unit suite passed **64 tests**, with **7 opt-in live tests skipped**; a separate CLI check loads the real trained adapter.
