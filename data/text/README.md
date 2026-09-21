# Text training and evaluation data

**Active V1: English text input, output and training.** The corpus contains 120 authored synthetic examples and 80 redacted messages from the public UCI SMS Spam Collection. The public subset has 40 original `ham` and 40 original `spam` labels. See [source attribution and reproduction](../sources/uci_sms/README.md). Synthetic examples use fictional contacts; public examples have contacts redacted and URLs replaced with a reserved `.example` address. No raw user submissions are collected.

| Split | Rows | Scenario groups | Use |
| --- | ---: | ---: | --- |
| `train.jsonl` | 132 | 131 | Weight updates: 80 synthetic + 52 public SMS |
| `validation.jsonl` | 34 | 34 | Checkpoint selection: 20 synthetic + 14 public SMS |
| `test.jsonl` | 34 | 34 | Previously inspected regression set: 20 synthetic + 14 public SMS |

There are 200 English examples in 199 scenario groups. Two versions of one UCI prize campaign share a group inside train. Synthetic scenario split assignments are retained, but misleading framing such as "pretending to be support" and many SMS/Email prefixes were removed. Prompt-injection cases now use `adversarial_input`, not `ordinary`. Simulated iPhone prompts are user descriptions, not verified verbatim iOS UI strings. Scenario IDs do not prove semantic independence: concepts necessarily recur across splits.

All expected labels, evidence quotes, rationales, reference IDs, and intentionally rejected answers received an automated authoring review and passed executable integrity checks on 2026-09-21. The records are marked `automated_reviewed`, not human or independent expert reviewed. Test labels were inspected for data preparation and token lengths, but no model predictions were generated or used to select these examples.

## Coverage and hard cases

| Category | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Message | 76 | 20 | 20 |
| Alert | 24 | 5 | 5 |
| Wi-Fi | 24 | 6 | 6 |
| Unknown | 8 | 3 | 3 |

Every split contains all 14 signal types and the three gold issue types (`unsupported`, `insufficient_context`, `contradictory_input`). Every signal has at least three positive training examples. Train contains 53 ordinary controls, 26 unsolicited promotions, 24 suspicious patterns, 13 connection issues, seven reported normal connections, seven unclear cases and two adversarial cases. These are design choices for a small development dataset, not estimated real-world prevalence.

The new cases distinguish code disclosure from code notices, payment requests from receipts, account passwords from Wi-Fi passwords, and remote access from installation. Negative controls include negation, historical or hypothetical requests, quoted educational content, ordinary informational links, legitimate installation prompts, airplane mode with working Wi-Fi, and searching for networks without a known connection state. A multiline popup with an embedded instruction checks that role markers remain data. Unknown cases cover missing context, unsupported devices, conflicting current states and instructions directed at the analyzer.

Labels describe observations. An installation request can occur in an ordinary Settings prompt, and an ordinary message can contain a visible link. These signals must not automatically become scam verdicts. Future safety guidance requires its own tests and expected actions.

[preparation_report.json](preparation_report.json) records counts, coverage, split SHA-256 hashes, the shared prompt hash, the exact tokenizer revision and measured sequence lengths. Pin these inputs when running the baseline and training. Changing a split or the prompt requires regenerating the report; keep the test set fixed after this preparation stage.

## Quick review examples (from training scenarios)

| Input excerpt | Correct observation | Incorrect interpretation to avoid |
| --- | --- | --- |
| Reply with the six-digit verification code you just received | Verification-code request | Treating it as an ordinary code notification |
| Do not share this code with anyone | No request to share the code | Treating a warning against sharing as a request |
| allow our technician to remotely control your iPhone | Remote-access request | Missing the remote-control instruction |
| Local News would like to send you notifications | Ordinary permission alert | Treating every popup as a scam |
| connected to HomeNet … No Internet Connection | Wi-Fi connected + no internet | Claiming that Wi-Fi must be switched off |
| websites load normally | Reported working connection, given the full context | Certifying the network identity as safe |

The JSONL files contain the full inputs and exact structured targets; these excerpts are for human review only.

## Record fields

- `id`, `scenario_group`, `language`: identity and split grouping; language must be `en`.
- `provenance`: `synthetic_authored` or `uci_sms_derived`; never relabel invented text as collected data.
- `source`: public messages include dataset ID, original row number, original `ham`/`spam` label, raw-text SHA-256 and sanitization version. These fields never enter the model prompt or target.
- `review_status`: `automated_reviewed` records this review. Use `human_reviewed` only after a person checks the semantics; automated review is not sufficient for a model-quality claim.
- `input_text`: synthetic situation or redacted original SMS. A pasted SMS need not announce its category with a heading.
- `expected`: correct structured category/signals/issues for supervised training.
- `case_kind`: scenario metadata for later analysis, not a model field or proof of authenticity.
- `rejected_output`, `rejection_reason`: a deliberately wrong answer and why it is wrong. Excluded from SFT; not a DPO dataset or a safety-response training set.
- `rationale`: labeling explanation, also excluded from the assistant training answer.
- `reference_ids`: optional background references in `knowledge/catalog.json`. They explain general principles; they do not establish that this particular fictional message is fraudulent. Ordinary reminders may have no reference.

## Labeling rules

Categories: `message` (SMS/email/chat), `alert` (notification/popup/warning), `wifi` (iPhone connectivity description), `unknown` (insufficient, unsupported or contradictory context). Explicit descriptions of ordinary phone alerts are supported. Inputs describing Android-specific settings or unrelated tasks should abstain.

| Case kind | Meaning |
| --- | --- |
| `suspicious_pattern` | Authored scenario contains a concerning solicitation or threat; identity still unverified |
| `ordinary` | Ordinary control scenario, not a certificate of legitimacy |
| `connection_issue` | Described connection/security-setting issue, not automatically a scam |
| `normal_connection` | User reports working/normal Wi-Fi, not independently verified |
| `unclear` | Requires abstention or clarification |
| `adversarial_input` | Contains instructions directed at the analyzer; not an ordinary safe control |
| `unsolicited_promotion` | Public source marked spam; not a verified scam verdict |

Signals describe explicit current text, not a final safe/unsafe decision. Evidence must be a literal, case-sensitive substring of the input. Preserve the original English wording, including punctuation. Use only supported signal IDs in `schemas.py`:

- `payment_request`: asks for payment, gift-card purchase or payment-card details, or explicitly invites a paid call/text/subscription. A price alone or a completed receipt is not a request.
- `credential_request`: asks for account-password disclosure; a known-network Wi-Fi password dialog uses `wifi_password_prompt`.
- `verification_code_request`: asks to share/enter a login code; a code notice saying not to share is a negative control.
- `urgent_security_warning`: claims an urgent account/device security threat; appointment, delivery and prize deadlines are not such threats.
- `support_phone_number`, `visible_link`, `install_request`, `remote_access_request`: require the corresponding explicit text; a link alone proves no wrongdoing.
- Wi-Fi signals: `wifi_off`, `airplane_mode_on`, `wifi_connected`, `no_internet`, `wifi_password_prompt`, `unsecured_network`. Connection to a router can coexist with no internet. Airplane mode alone does not establish Wi-Fi off. A reported checkmark can support connection; a model must not invent an unseen checkmark.

Negation, quoted instructions to the analyzer, and past/hypothetical states must not become current signals. Report incompatible current states as `unknown` / `contradictory_input`; ordinary state changes over time are not contradictions. Unknown outputs have no signals. System/runtime error codes are not gold labels for ordinary authored inputs.

## Preparation commands

```bash
python -m easy_tech_help.dataset
python -m pip install -e '.[dev,data]'
```

Download only the selected tokenizer once (internet required; no API key or model weights required):

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')"
```

Then check lengths and prepare training files offline:

```bash
python -m easy_tech_help.dataset --check-tokens --report data/text/preparation_report.json --export artifacts/text-sft
```

The exporter writes only `train.jsonl` and `valid.jsonl` chat records, using the runtime prompt and correct assistant answer. Test cases and wrong alternatives never enter the exported trainer files. The token check reads the locally cached tokenizer at the pinned revision, with no inference call. It includes the system prompt, chat markers, user text, assistant answer and end marker, and checks that the inference prefix exactly matches the start of each complete sequence. No input is silently truncated. A missing tokenizer or an overlength sample fails preparation before export.

| Token measurement | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Maximum full sequence | 630 | 612 | 610 |
| Maximum assistant completion | 107 | 89 | 91 |

All sequences fit the 1,536-token training limit and all completions fit the planned 256-token output budget. These measurements use the actual exported JSON serialization, which includes spaces. They are token-length checks, not a claim that a model will produce correct or complete responses within the budget.

Validation rejects incorrect structure, invented evidence, runtime-only gold issues, incompatible case metadata, duplicate IDs/text/source rows, unknown source IDs, scenario groups crossing splits, and held-out inputs copied literally into the system prompt. A word-sequence similarity check rejects cross-split pairs at or above 0.85 (0.75 between public SMS), after normalizing Unicode, case, URLs and numbers. The current corpus has zero cross-split flagged pairs after grouping the repeated campaign. This lexical heuristic cannot establish semantic independence or label truth. Human review remains necessary. The historical public corpus may have appeared in model pretraining and does not provide contemporary phishing or iPhone-specific examples.

Preparation itself does not update weights. `easy_tech_help.training` implements PyTorch LoRA with completion-only loss and validation checkpoint selection. `easy_tech_help.evaluation` compares base and adapter with identical prompts/decoding, including separate synthetic/public metrics. See the root README for commands. The preparation report records the pre-training integrity check, not model performance.

Do not report training-set performance as evaluation. The initial test results have been inspected; subsequent runs on these same 34 examples are regression checks, not independent final evaluation. The continuation keeps every input and label unchanged and selects checkpoints using validation only. Reserve fresh, independently reviewed texts before claiming generalization. Test labels are intentionally reviewable in the repository; a held-out claim concerns model/developer tuning use, not secret files.

## Background references

These principles are informed by the project's local corpus and public guidance, not copied text:

- [FTC: recognizing phishing](https://consumer.ftc.gov/articles/how-recognize-avoid-phishing-scams)
- [FTC: tech support scams](https://consumer.ftc.gov/articles/how-spot-avoid-and-report-tech-support-scams)
- [Apple: iPhone Wi-Fi troubleshooting](https://support.apple.com/en-us/111786)
- [Apple: social engineering and phishing](https://support.apple.com/en-us/102568)

The knowledge catalog contains the full source mappings. Keep synthetic samples separate from those documents.
