# Text training and evaluation seed data

**Active V1: English text input, output and training.** These examples were authored for EasyTechHelp on 2026-09-20. They are synthetic examples, not collected scam messages, official FTC/Apple examples, or transcriptions of the old screenshots. There are no real passwords or verification codes; links use reserved `.example` domains and the fictional support number uses a 555-01xx number.

| Split | Rows | Scenario groups | Use |
| --- | ---: | ---: | --- |
| `train.jsonl` | 19 | 19 | Future weight updates |
| `validation.jsonl` | 7 | 7 | Development and tuning |
| `test.jsonl` | 7 | 7 | Frozen final comparison |

There are 33 English examples covering 33 distinct authored scenarios. The Korean translations have been removed, retaining all original English examples and their split assignments. Splits include ordinary/suspicious messages and alerts, Wi-Fi situations, and uncertain/unsupported inputs. Some signals recur across splits intentionally; each scenario group belongs to only one split.

All 33 expected labels, evidence quotes, rationales, reference IDs, and intentionally rejected answers received a systematic automated review on 2026-09-21. No expected-label changes were required. This review checked consistency with the labeling rules and stored source catalog; it is not a human domain-expert review and does not establish real-world accuracy. The final test labels were inspected, but the model was not run on the final test inputs.

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
- `provenance`: `synthetic_authored`; never claim these are real incident records.
- `review_status`: `automated_reviewed` records this review. Use `human_reviewed` only after a person checks the semantics; automated review is not sufficient for a model-quality claim.
- `input_text`: exact user text, including necessary context such as “SMS” or “Safari popup.”
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

Signals describe explicit current text, not a final safe/unsafe decision. Evidence must be a literal, case-sensitive substring of the input. Preserve the original English wording, including punctuation. Use only supported signal IDs in `schemas.py`:

- `payment_request`: asks for payment, gift-card purchase or payment-card details; a completed receipt is not a request.
- `credential_request`: asks for account-password disclosure; a known-network Wi-Fi password dialog uses `wifi_password_prompt`.
- `verification_code_request`: asks to share/enter a login code; a code notice saying not to share is a negative control.
- `urgent_security_warning`: claims an urgent account/device security threat; an appointment or delivery delay is not one.
- `support_phone_number`, `visible_link`, `install_request`, `remote_access_request`: require the corresponding explicit text; a link alone proves no wrongdoing.
- Wi-Fi signals: `wifi_off`, `airplane_mode_on`, `wifi_connected`, `no_internet`, `wifi_password_prompt`, `unsecured_network`. Connection to a router can coexist with no internet. Airplane mode alone does not establish Wi-Fi off. A reported checkmark can support connection; a model must not invent an unseen checkmark.

Negation, quoted instructions to the analyzer, and past/hypothetical states must not become current signals. Report incompatible current states as `unknown` / `contradictory_input`; ordinary state changes over time are not contradictions. Unknown outputs have no signals. System/runtime error codes are not gold labels for ordinary authored inputs.

## Preparation commands

```bash
python -m easy_tech_help.dataset
python -m easy_tech_help.dataset --export artifacts/text-sft
```

The exporter writes only `train.jsonl` and `valid.jsonl` chat records, using the runtime prompt and correct assistant answer. Test cases and wrong alternatives never enter the exported trainer files. File validation checks gold and rejected-answer structure, exact quote presence, runtime-only issue codes, references and split grouping. These checks do not establish factual correctness. Training itself, completion masking, tokenizer-length validation, adapter loading and before/after evaluation remain to be implemented.

Do not report training-set performance as evaluation. Freeze prompts and model selection using validation before running the final test. Test labels are intentionally reviewable in the repository; a held-out claim concerns model/developer tuning use, not secret files. Add fresh independent texts after this small seed set to test generalization.

## Background references

These principles are informed by the project's local corpus and public guidance, not copied text:

- [FTC: recognizing phishing](https://consumer.ftc.gov/articles/how-recognize-avoid-phishing-scams)
- [FTC: tech support scams](https://consumer.ftc.gov/articles/how-spot-avoid-and-report-tech-support-scams)
- [Apple: iPhone Wi-Fi troubleshooting](https://support.apple.com/en-us/111786)
- [Apple: social engineering and phishing](https://support.apple.com/en-us/102568)

The knowledge catalog contains the full source mappings. Keep synthetic samples separate from those documents.
