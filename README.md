# EasyTechHelp

Multimodal AI tech support for older adults and their families

My grandmother lives alone, and when she runs into something confusing on her phone, helping remotely can be difficult because she may not know how to describe exactly what she is seeing. I am building EasyTechHelp so an older adult can upload a screenshot, understand what is on the screen, identify potential risks, and get a safe next step. If they still need help, the system will create a concise summary that a family member can quickly understand.

## Current status

This repository contains the Milestone 1 project foundation. The Streamlit app starts, but screenshot upload, AI analysis, safety guidance, family handoff, and benchmark evaluation are **not implemented yet**. The sections below describe the intended V1 product, not currently available features.

## V1 scope

- Pop-ups and security warnings, including harmless examples
- Text messages and emails
- Wi-Fi and connectivity settings
- An `unknown` result when a screenshot is unclear or outside those categories

The planned pipeline will convert a screenshot into validated structured observations, apply application-level safety rules, and then render short guidance. The model's free-form answer will not be shown directly. A copyable family summary and a fixed 24-screenshot evaluation benchmark are also planned.

## Run the foundation locally

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
streamlit run app/app.py
```

On systems where `python3` is older than 3.11, use a newer Python executable in the first command. The app opens in the browser and displays the current project preview. No API key is needed for this milestone.

Run the checks with:

```bash
python -m pytest
ruff check .
```

`.env.example` shows the configuration reserved for the later model integration. If needed in a later milestone, copy it to `.env` and supply your own credentials. `.env` is ignored by Git.

## Design and limitations

The planned architecture and module responsibilities are in [docs/architecture.md](docs/architecture.md). V1 will process one screenshot at a time and cannot independently authenticate a message sender, website, phone number, or Wi-Fi network from an image alone. Screenshots may be sent to a configured multimodal model provider when analysis is implemented; users should avoid uploading information they do not want to share with that provider.

There are no benchmark results or demo screenshots yet. Results will be published only after the fixed benchmark has actually run.
