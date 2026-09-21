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
| English labeled text examples and SFT export | Prepared: 33 synthetic draft examples; no fine-tuning has run |
| Local reference corpus and keyword retrieval | Implemented separately; 6 FTC originals + 4 Apple-based summaries |
| RAG-connected explanation and source display | Next implementation stage |
| Risk assessment and safe next-action rules | Planned; observation validation is not a complete safety policy |
| Family handoff summary | Planned |
| Baseline evaluation, text fine-tuning, before/after comparison | Planned; no accuracy results yet |

The current app shows **observations only**, not scam verdicts or next-step advice. It does not automatically save user input or add it to training data.

## Intended V1 flow

```text
Pasted text / typed Wi-Fi status
  → local text model (later: same base model with a text-trained adapter)
  → validated category + signals + exact quotes from the input
  → retrieve relevant local FTC / Apple-based help documents
  → generate an explanation grounded in those documents
  → application safety rules select allowed actions and reject unsafe guidance
  → explanation / cautions / next action / sources / optional family summary
```

Scope: SMS/email/chat (`message`), alerts/notifications/browser popups (`alert`), iPhone connectivity descriptions (`wifi`), and missing/unsupported/contradictory context (`unknown`). Ordinary messages and harmless alerts are necessary controls. A Wi-Fi problem is not automatically a scam. V1 supports English input and output. Other languages are outside the supported scope; the prompt requests an unsupported result, but language detection is not a deterministic input check.

## Run locally

Use Python 3.11+ and [Ollama](https://docs.ollama.com/macos). Run from this repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ollama pull qwen3:4b
```

Start Ollama if the desktop app or another process is not already serving it:

```bash
ollama serve
```

In another terminal with the virtual environment activated:

```bash
streamlit run app/app.py
```

CLI alternatives:

```bash
python -m easy_tech_help.analysis --text 'Text message: Reply with your verification code.'
python -m easy_tech_help.analysis --file examples/message.txt
python -m easy_tech_help.analysis --text 'iPhone Wi-Fi is off.' --model qwen3:4b
```

The default is the local [Qwen3 4B text model](https://ollama.com/library/qwen3:4b). Initial model/package downloads require internet access; analysis calls only `127.0.0.1:11434`, bypassing proxy environment variables. No hosted-model API key is needed. `.env.example` documents the optional `EASY_TECH_HELP_LOCAL_MODEL` setting; creating `.env` is unnecessary for the default. If an existing `.env` still selects `qwen3-vl:4b-instruct`, update it to `qwen3:4b`.

The former positional image-path CLI and `analyze_screenshot()` have been replaced with `--text` / `--file` and `analyze_text()`. Image parsing, image request payloads, image tests and the direct Pillow dependency were removed. Streamlit may still install Pillow as its own dependency.

## Text data and training setup

The repository examples are in [data/text/](data/text/). Read the [labeling guide](data/text/README.md) before changing them.

| Split | Examples | Independent scenario groups | Purpose |
| --- | ---: | ---: | --- |
| [train.jsonl](data/text/train.jsonl) | 19 | 19 | Future supervised fine-tuning |
| [validation.jsonl](data/text/validation.jsonl) | 7 | 7 | Development and model/adapter selection |
| [test.jsonl](data/text/test.jsonl) | 7 | 7 | Final comparison after settings are frozen |

The dataset contains **33 authored English examples covering 33 scenarios**. The earlier Korean translations were removed; the existing scenario split assignments are preserved. Examples include suspicious payment/password/code requests, ordinary receipts and notifications, Wi-Fi problems, normal connections, missing context, and an instruction-injection case. They are newly authored synthetic text, not transcriptions of the previous 24 screenshots and not quotations from FTC or Apple.

Each record contains the input, correct structured answer (`expected`), an intentionally wrong answer (`rejected_output`) with an explanation, scenario group, language, review status and reference IDs. “Ordinary” means the described text has no selected suspicious request; it does not certify the sender as authentic. Wrong answers are for review/error analysis, **never the supervised training target**.

Validate and export:

```bash
python -m easy_tech_help.dataset
python -m easy_tech_help.dataset --export artifacts/text-sft
```

Export creates `train.jsonl` and `valid.jsonl` with system/user/assistant messages. It shares the exact inference prompt and exports only the correct category/signals/issues as the assistant answer. Test examples, wrong answers, rationale and review metadata are excluded. `artifacts/` and `.venv-training/` are ignored by Git. This command prepares data; it **does not train a model**.

These seed labels need human review and substantially more varied examples before claiming useful generalization. A training runner, token-length checks, adapter loading and performance comparison remain to be implemented. On this Mac, a text LoRA workflow is the intended next training approach. Choose and record the exact text base-model revision and runtime; compare the same model before and after training. Previously downloaded vision weights in `artifacts/base-model/` and the old MLX-VLM environment are not used by this text setup.

## References and retrieval

[knowledge/README.md](knowledge/README.md) describes the six original FTC HTML pages, four clearly labeled Apple-based summaries, and source URLs. `retrieval.py` searches local article text using weighted English keyword overlap. It does not fetch a source link at runtime, use a vector database, or yet feed retrieved documents to a generative model.

Connecting RAG is the next step. It must map analysis category `alert` to the existing knowledge category `popup`, turn validated signals into useful English search queries. A missing source must remain explicit. Training examples are not RAG evidence, and user input must never be added to the trusted corpus automatically.

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

The previous screenshot evaluation directory and its 24 images have been removed. Active evaluation data lives in [data/text/validation.jsonl](data/text/validation.jsonl) for development and [data/text/test.jsonl](data/text/test.jsonl) for final evaluation after settings are frozen. The historical demo `examples/synthetic-message.png` remains unused by text V1. No fine-tuning or final evaluation has been run.

## Next milestones

| Order | Work | Difficulty (1–5) |
| --- | --- | ---: |
| 1 | Review text labels; implement and record baseline evaluation | 3 |
| 2 | Connect local RAG and code-enforced safe guidance | 5 |
| 3 | Expand text training data; run LoRA and compare the same base model before/after | 5 |
| 4 | Build family handoff and finish the readable product interface | 3 |
| 5 | Freeze settings, run final held-out evaluation, publish demo/results/limits | 4 |

Architecture and implementation boundaries: [docs/architecture.md](docs/architecture.md).
