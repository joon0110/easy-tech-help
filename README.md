# EasyTechHelp

Local text-based GenAI tech support for older adults and their families.

My grandmother lives alone, and helping with confusing phone messages remotely can be difficult. EasyTechHelp is being built to explain a pasted message, alert, or iPhone Wi-Fi description, identify concerns, and suggest a safe next step with sources. If the issue remains difficult, it will prepare a summary to share with family.

**V1 direction: English-only text input and text fine-tuning, local inference, RAG, and application safety rules.** Screenshot input and image training are outside the active V1. This decision replaces the earlier screenshot plan.

## Current status

| Component | Status |
| --- | --- |
| Python package, configuration, pytest, Ruff | Implemented |
| Text input in Streamlit and CLI | Implemented |
| Local text model → validated category, signals, quoted evidence | Implemented |
| English labeled text examples and SFT export | 200 examples: 120 synthetic + 80 redacted public SMS; source/coverage/leakage/token checks |
| Local reference corpus and chunk retrieval | Implemented; 6 FTC originals + 4 Apple-based summaries, sentence-aware chunks and BM25 |
| RAG-connected explanation and source display | Implemented in Streamlit and CLI; retrieved excerpts, catalog URLs and explicit abstention |
| Risk assessment and safe next-action rules | Planned; observation validation is not a complete safety policy |
| Family handoff summary | Planned |
| PyTorch training and inference | Implemented: Qwen2.5 1.5B + PEFT LoRA on MPS/CPU |
| Baseline evaluation and fine-tuning comparison | Completed; [current results](results/improvement.md) and [first-run history](results/README.md); signal errors remain |

The current app shows **reviewed observations and model-selected source passages**, with surrounding context and official links. Explanations are extractive: the application copies a selected source passage rather than displaying a free-form model paraphrase. Narrow application rules correct contextless help requests and explicit password requests mislabeled as payments; the raw prediction and adjustments remain in the RAG trace. Action safety rules and family handoff are not implemented yet. Input is not automatically saved or added to training data. Raw adapter observation matches remain 20/34 development and 23/34 regression; public-SMS extraction remains weak. See the [grounding and classification improvements](results/rag_improvement.md), [analysis results](results/improvement.md), and historical [initial RAG results](results/rag.md).

## Intended V1 flow

```text
Pasted text / typed Wi-Fi status
  → local PyTorch text model with a trained LoRA adapter
  → validated category + signals + exact quotes from the input
  → retrieve relevant local FTC / Apple-based help documents
  → local model selects a source passage; code renders its exact words
  → application safety rules select allowed actions and reject unsafe guidance
  → explanation / cautions / next action / sources / optional family summary
```

Scope: SMS/email/chat (`message`), alerts/notifications/browser popups (`alert`), iPhone connectivity descriptions (`wifi`), and missing/unsupported/contradictory context (`unknown`). Ordinary messages and harmless alerts are necessary controls. A Wi-Fi problem is not automatically a scam. V1 supports English input and output. Other languages are outside the supported scope; the prompt requests an unsupported result, but language detection is not a deterministic input check.

## Run locally

Use Python 3.11+ from this repository root. The tested machine is an M4 Mac with 16 GB memory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,training]'
```

Download the pinned model once (about 3.1 GB; internet required):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_dir='artifacts/pytorch-model', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE'])"
```

Train the initial adapter, then the continuation selected with generated validation answers. Downloaded weights and trained adapters stay in ignored `artifacts/`; a new clone needs both training steps. An existing nonempty adapter directory will not be overwritten. If the initial adapter already exists, run only the continuation command. The app defaults to `artifacts/pytorch-adapter-v2`.

```bash
python -m easy_tech_help.training --device mps --output artifacts/pytorch-adapter
python -m easy_tech_help.training --device mps --init-adapter artifacts/pytorch-adapter --output artifacts/pytorch-adapter-v2 --epochs 3 --learning-rate 0.00005 --decision-weight 4 --select-by-generation
streamlit run app/app.py
```

CLI alternatives:

```bash
python -m easy_tech_help.rag --text 'My iPhone is connected to Wi-Fi but says No Internet Connection.'
python -m easy_tech_help.rag --file examples/message.txt
python -m easy_tech_help.analysis --text 'Text message: Reply with your verification code.'
python -m easy_tech_help.analysis --file examples/message.txt
python -m easy_tech_help.analysis --text 'iPhone Wi-Fi is off.'
python -m easy_tech_help.runtime --base --text 'Reply with your verification code.'
```

The model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Generation uses local files in the Python process and needs no hosted-model API key. Set `--device cpu` when MPS is unavailable; CPU execution is slower. `.env.example` documents optional model/adapter directories and the device. An old `EASY_TECH_HELP_LOCAL_MODEL=qwen3:4b` setting must be removed or changed to `artifacts/pytorch-model`. The app requires a completed adapter and will not silently substitute the base model.

The former positional image-path CLI and `analyze_screenshot()` have been replaced with `--text` / `--file` and `analyze_text()`. Image parsing, image request payloads, image tests and the direct Pillow dependency were removed. Streamlit may still install Pillow as its own dependency.

## Text data and training setup

The repository examples are in [data/text/](data/text/). Read the [labeling guide](data/text/README.md) before changing them.

| Split | Examples | Scenario groups | Purpose |
| --- | ---: | ---: | --- |
| [train.jsonl](data/text/train.jsonl) | 132 | 131 | Supervised fine-tuning |
| [validation.jsonl](data/text/validation.jsonl) | 34 | 34 | Model/adapter selection |
| [test.jsonl](data/text/test.jsonl) | 34 | 34 | Regression checks; a new independent final set is still needed |

The dataset contains **120 revised synthetic examples plus 80 redacted public SMS**. The UCI subset includes **40 normal (`ham`) and 40 spam messages**, with the original labels stored only as provenance. Public spam includes advertising; it is not a verified fraud label. [Source attribution, original notice and checksums](data/sources/uci_sms/README.md) make the selected records traceable to the downloaded archive. Synthetic examples cover iPhone Wi-Fi, alerts, sensitive requests, missing context and adversarial inputs. Misleading framing was removed, and most message inputs no longer announce their category. Synthetic iPhone prompts are not verified verbatim system text.

**More real-world data is needed for iPhone Wi-Fi states and popups/alerts.** Their current training examples are synthetic, not text collected and verified from actual devices. The public SMS dataset does not cover these categories. Before claiming reliable performance, collect de-identified English text from real iPhone Wi-Fi screens and actual popups/alerts, include normal and problematic cases, review the labels, and reserve separate real examples for evaluation.

Each record contains the input, correct structured answer (`expected`), an intentionally wrong answer (`rejected_output`) with an explanation, scenario group, language, review status and reference IDs. “Ordinary” means the described text has no selected suspicious request; it does not certify the sender as authentic. Wrong answers are for review/error analysis, **never the supervised training target**.

Validate and export:

```bash
python -m easy_tech_help.dataset
python -m easy_tech_help.dataset --check-tokens --report data/text/preparation_report.json --export artifacts/text-sft
```

The token check requires the `data` extra and the cached pinned Qwen tokenizer; installation/download commands are in [data/text/README.md](data/text/README.md#preparation-commands). Export creates `train.jsonl` and `valid.jsonl` with system/user/assistant messages. It shares the exact inference prompt and exports only the correct category/signals/issues as the assistant answer. Test examples, wrong answers, rationale and review metadata are excluded. `artifacts/` and `.venv-training/` are ignored by Git. This command prepares data; it **does not train a model**.

All 200 examples passed structural/evidence/reference checks, case metadata checks, split leakage checks and the pinned tokenizer length check. The longest full sequence is 630 tokens, within the 1,536-token limit. The [preparation report](data/text/preparation_report.json) records coverage, split hashes and token measurements. Review is `automated_reviewed`, not independent human/domain review. Source redactions are reproducible with `python -m easy_tech_help.sms_source` after downloading the original ZIP. Public benchmark messages may already have appeared in model pretraining; this small curated corpus cannot establish real-world accuracy.

The model is pinned to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with its weights checksum verified before training. PyTorch computes causal language-model loss only on assistant completion tokens; PEFT updates 1,089,536 LoRA parameters while the base remains frozen. Initial training uses 5 epochs, rank 8, learning rate 2e-4, microbatch 1 and accumulation 4; selection uses validation loss. The continuation uses 3 additional epochs, learning rate 5e-5 and 4× loss weight for category/signal/issue decisions, with unchanged data and prompt. It selects by generated validation observation matches, then ordinary-control errors, signal F1 and loss; epoch 2 was selected. BF16 base weights and FP32 adapters are used on MPS. Inference explicitly disables repetition penalties so signal names and input quotes can repeat. See [architecture](docs/architecture.md) for details.

Compare the same base and current adapter on the existing regression split (this test set has already been inspected):

```bash
python -m easy_tech_help.evaluation --device mps --split test --output results/pytorch_v2_regression.json
```

The evaluator records raw generations, invalid outputs, category accuracy, signal precision/recall, observation matches, ordinary-control extra signals, latency and separate synthetic/public results. These are analysis metrics; they do not measure RAG or the unfinished safety-action pipeline.

## References and retrieval

[knowledge/README.md](knowledge/README.md) describes the six original FTC HTML pages, four clearly labeled Apple-based summaries, and source URLs. The app reads these local files; it does not fetch their links at runtime. No vector database, embeddings service, new model download or API key is required.

`rag.py` now connects the full explanation path:

1. Analyze the text with the existing PyTorch LoRA adapter, validate its observations and apply narrow application corrections. The model, extraction prompt and training data are unchanged.
2. Map `alert` to corpus category `popup`; use the original text for retrieval, with lower-weight English expansions for observed signals. Generic account messages do not select Apple Account material unless Apple or iCloud is mentioned.
3. Filter documents using weighted title/tag/body keywords. Group adjacent short paragraphs to retain context, split at sentence boundaries (target 75 words, preserving longer sentences), then rank excerpts with BM25 plus a small document-score contribution. Retrieve at most three candidates, at most two per document.
4. Split the highest-ranked usable passage into complete sentences, retain dependent continuations and past/present contrasts, and rank these units against the original input. The same loaded Qwen base model, with the extraction adapter temporarily disabled, selects one numbered unit or abstains. Selection uses repetition penalty 1.1; observation extraction keeps 1.0.
5. Validate the source ID and integer selection. Copy the entire selected unit verbatim; reject free-form additions, invented IDs and malformed selections. Official URLs come only from the catalog. Apple summaries are explicitly labeled as summaries.

Unknown analysis, no matching evidence, model abstention, invalid source IDs, incomplete generation and runtime/file failures produce explicit messages. A failed draft is not displayed as an answer. The CLI includes raw output for debugging; the app shows only accepted explanations and source excerpts. Training examples and user input are never added to the trusted corpus automatically.

**Exact quotation prevents new claims being added to the explanation; it does not prove relevance or safety.** A selected passage can still be poorly matched, and a general reference cannot verify this sender, this network or the cause of this problem. Some official passages contain troubleshooting steps, but these quotations are not personalized, safety-approved actions. A reviewed application action policy remains the next stage. The wording is less conversational than free-form generation; that is the current grounding tradeoff.

Run the real local RAG development checks (eight assistant-authored inputs, not an independent accuracy benchmark):

```bash
python -m easy_tech_help.rag_evaluation --device mps --output results/rag_improved.json
python -m easy_tech_help.review_evaluation
```

The RAG report preserves raw and reviewed observations, correction reasons, retrieved excerpts, selected quotes, raw model output, timings and source/code/adapter fingerprints. `review_evaluation` separately replays the corrections on 68 recorded predictions; it does not run or retrain a model. See [reviewed results and limits](results/rag_improvement.md).

## Validation and limits

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

Optional local-model smoke tests (development inputs, not the held-out dataset):

```bash
EASY_TECH_HELP_RUN_LIVE_TESTS=1 python -m pytest tests/test_analysis_live.py -v -s
```

Inputs are limited to 4,000 characters. Pydantic rejects unknown fields, unsupported categories, duplicate signals, incompatible connection states, and evidence absent from the original input. Invalid or unfinished generations become `unknown`; runtime failures are reported explicitly. Application templates supply the displayed summary and uncertainty text.

Exact quotation proves only that words occurred in the input. It does not prove that a signal interpretation, sender claim, or network status is true. Prompt instructions are not a security guarantee. Future safety rules must account for false alarms, missed scams, uncertain context and unsafe actions. Remove passwords, verification codes and personal details before pasting text.

Active evaluation data lives in [data/text/validation.jsonl](data/text/validation.jsonl) for development and [data/text/test.jsonl](data/text/test.jsonl) for regression checks after its initial evaluation. The preparation report captures data checks before training; training manifests and evaluation results are separate. Normal controls are necessary to measure false alarms, and unseen independently reviewed recent messages remain necessary for a stronger evaluation.

## Next milestones

| Order | Work | Difficulty (1–5) |
| --- | --- | ---: |
| 1 | Implement code-enforced safe guidance; independently evaluate retrieval relevance and analysis errors | 5 |
| 2 | Build family handoff and finish the readable product interface | 3 |
| 3 | Add independent contemporary cases and evaluate the complete guidance pipeline | 4 |
| 4 | Publish demo/results/limits | 3 |

Architecture and implementation boundaries: [docs/architecture.md](docs/architecture.md).
