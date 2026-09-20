# EasyTechHelp

Local multimodal AI tech support for older adults and their families

My grandmother lives alone, and when she runs into something confusing on her phone, helping remotely can be difficult because she may not know how to describe exactly what she is seeing. I am building EasyTechHelp so an older adult can upload a screenshot, understand what is on the screen, identify potential risks, and get a safe next step. If they still need help, the system will create a concise summary that a family member can quickly understand.

## Current status

This repository contains the Milestone 1 project foundation and a small local knowledge base: six original FTC web pages and four original summaries based on Apple support pages. The Streamlit app starts and local document search works, but screenshot upload, image understanding, generated guidance, safety enforcement, family handoff, and benchmark evaluation are **not implemented yet**. The sections below describe the intended iPhone-only V1 product, not currently available features.

## V1 scope

- Safari pop-ups and security warnings on iPhone, including harmless examples
- Text messages and emails viewed on iPhone
- iPhone Wi-Fi and connectivity settings
- An `unknown` result when a screenshot is unclear or outside those categories

The planned local-only pipeline will read a screenshot using local image understanding, convert it into validated structured observations, retrieve relevant local help documents, apply application-level safety rules, and render short guidance with source links. The model's free-form answer will not be shown directly. A copyable family summary and a fixed 24-screenshot evaluation benchmark are also planned.

## Run the foundation locally

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
streamlit run app/app.py
```

On systems where `python3` is older than 3.11, use a newer Python executable in the first command. The app opens in the browser and displays the current project preview. V1 is intended to run only on your own computer and does not require a hosted-model API key. A local OCR engine and model still need to be selected and installed before screenshot analysis works.

Run the checks with:

```bash
python -m pytest
ruff check .
```

The ten local RAG documents are six original FTC HTML pages in [knowledge/original/](knowledge/original/) and four short Apple-based summaries in [knowledge/apple_summaries/](knowledge/apple_summaries/). `src/easy_tech_help/retrieval.py` extracts FTC article text and reads the summary text locally without a network connection or database. The catalog labels each item as an original or a summary and stores its source URL. Apple's pages themselves are not copied into this repository. The search results are not yet connected to the Streamlit screen.

`.env.example` contains an optional local model name reserved for later integration. `.env` is ignored by Git.

## Design and limitations

The planned architecture and module responsibilities are in [docs/architecture.md](docs/architecture.md). V1 will process one iPhone screenshot at a time on the same computer. Android screens are outside V1. The app cannot independently authenticate a message sender, website, phone number, or Wi-Fi network from an image alone. User screenshots will not become RAG source documents or be permanently stored.

There are no benchmark results or demo screenshots yet. Results will be published only after the fixed benchmark has actually run.
