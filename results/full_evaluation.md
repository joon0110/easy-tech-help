# Full product evaluation — 2026-09-21

This is the preserved **before-fix** audit. See [quality improvement results](quality_improvement.md) for the current implementation and follow-up verification.

## Assessment

**The local demonstration pipeline works, but it is not ready for unsupervised real-world guidance.** Output constraints and integration checks pass. Risk detection, normal-message handling and explanation relevance still fail on concrete inputs. A browser accessibility check also found insufficient contrast in the warning text.

This audit evaluated the product at commit `b3e3a77`. It added evaluation code, fixtures and reports; it did not retrain the model or change the prompt, application policy, UI or existing gold labels to improve results.

## Scope and measured results

| Area | Measured result | Interpretation |
| --- | --- | --- |
| Dataset integrity and token limits | 200 examples passed; 132 train / 34 validation / 34 regression | Structural/source checks, not independent label validation |
| Public SMS provenance | All 80 selected records reproduced from the original source | Historical public SMS, not verified modern fraud labels |
| Automated suite | 205 passed, 7 live checks skipped in the ordinary run | Includes three tests of the new evaluator |
| Opt-in actual-model smoke tests | Separately enabled: 7/7 passed | Total 212 passing tests across the two runs; development examples only |
| Raw adapter regression observations | 23/34 exact category/signal/issue matches (67.6%); 30/34 correct categories | Reproduces earlier current-adapter metrics |
| Raw regression signal extraction | Precision 88.2%; recall 68.2%; F1 0.7692 | Different from binary scam accuracy |
| Public SMS regression subset | 11/14 exact observations; **0/4 labeled signals recovered** | Many no-signal labels inflate the match rate |
| New authored product challenge | **19/24** pass all category, level, required-signal and action checks | 79.2% on a small targeted development set |
| Existing product development cases | 12/12 passed | Previously used examples; optimistic compared with new cases |
| Existing RAG development checks | 8/8 passed again | These checks do not establish general semantic relevance |
| Action and handoff output constraints | 70/70 pass all eight checks | No prohibited next-action ID, modified action template, bad action evidence or broken handoff contract was observed |
| Explanation provenance | 39 answered cases have exact source-unit citations; 37 displayed | Literal quotation does **not** guarantee an appropriate explanation |
| Explicit privacy-marker probes | 3/3 exclude the fictional secrets/contacts from the family summary | Empty marker lists on other cases are not privacy evidence |
| Browser interaction | Real model, official links, copy without hover, exact clipboard/download, reset passed | One exercised sensitive-request flow, not a usability study |
| Automated accessibility | Initial page: no confirmed violations at 1440/390/320px; result page: **one contrast violation** | Initial-page header contrast also had two incomplete checks at each width |
| Full-product latency | Mean 2.718 s; nearest-rank p95 4.628 s across 70 cases | Warm local MPS run; excludes loading; total measured case time 190.277 s |

The model is the existing Qwen2.5-1.5B-Instruct + selected LoRA adapter running on PyTorch 2.14.0, Transformers 5.17.0 and PEFT 0.21.0. Adapter hashes match prior validation/regression reports. The full run used MPS. Prior validation results were checked for adapter identity, not rerun; the 34-row regression set was rerun through the complete current product.

## What failed

### 1. Sensitive requests can still receive no warning

Of the seven existing regression cases with a gold password/code/payment/remote-access signal, only **four received the expected attention level and signal flags**. Three received `no_specific_warning`:

| Case | Missed cue |
| --- | --- |
| `message_auction_identity_en` | A request for the one-time login number, phrased as identity confirmation |
| `uci_sms_3829` | A prize claim with premium-call cost information and a visible link |
| `uci_sms_1943` | A holiday/prize claim with a per-minute call charge |

The fine-tuned model misses these signals and the current supplementary patterns do not recover them. Generic `no_change` remains an allowed action, so action-allowlist tests pass despite the missed risk. This is why 70/70 output-constraint success must not be presented as safe detection.

### 2. Five new cases fail product expectations

| Case | Observed failure | User impact |
| --- | --- | --- |
| `normal_code_notice` | A login-code notification with “Never share this code” is interpreted as a code request and urgent warning | Unnecessary attention-level warning |
| `private_details` | Extraction fails validation; the input guard recovers the password request but misses the accompanying code request | Safe pause/clarification, but incomplete analysis and no RAG explanation |
| `settings_update_control` | A genuine iPhone Settings update prompt is classified as a message with a concerning installation request | Unnecessary source-check guidance |
| `airplane_wifi_working` | Airplane Mode is missed despite being explicitly on; the approved Airplane Mode check is absent | Incomplete connection guidance |
| `wifi_no_access` | The blue-checkmark/no-internet description fails observation validation | Clarification instead of useful troubleshooting |

Challenge breakdown: messages 6/8, alerts 7/8, Wi-Fi 4/6, unknown/out-of-scope 2/2 pass all scored quality checks. All eight new sensitive-request cases receive `attention`, but one fails category/signal completeness. Two of six ordinary controls receive unnecessary caution (one attention, one source-check); this is not an estimate of real-world false-positive frequency.

A targeted raw-generation rerun reproduced both validation failures. `private_details` invents the contiguous quote “Send your verification code 684219” by combining nonadjacent parts of the input. `wifi_no_access` uses “connected” as evidence even though the input describes a blue checkmark and does not contain that word. The validator correctly rejects both: the improvement should be literal evidence extraction, not allowing invented quotes.

### 3. Exact citations can still be poorly matched

For the 16 challenge inputs with predeclared expected reference documents, retrieval found a listed document in **14/16**, and a listed document was cited in **12/16**. These are document-coverage proxies, not semantic accuracy. In one case, a legitimately relevant FTC spam-text source was omitted from the predeclared expected list; the fixture limitation is recorded rather than silently relabeled after the run.

Qualitative review of all 15 answered challenge cases found clear relevance problems, including:

- A parcel redelivery fee is explained with a lottery-fee example.
- A request for a sign-in secret is explained with an email billing-problem example.
- A working iPhone connection is explained with a list of home-network devices.
- With Airplane Mode explicitly on and Wi-Fi working, the displayed source says to check that Airplane Mode is off. It does not preserve the application's “confirm whether intentional” qualification.

The final case also shows that a general reference passage can expose an instruction whose context differs from the approved next steps. The current display filter catches some high-impact instructions, but it does not establish contextual compatibility for every quote. See [all fifteen qualitative reviews](full_rag_review.json). This is an assistant review, not blinded human adjudication or a new semantic-accuracy metric.

### 4. Warning text contrast needs correction

The real browser result-page scan reports foreground `#926c05` on `#fcf7d4`, contrast **4.43:1**, below the checker’s **4.5:1** target for this 18px normal-weight text. The input page did not have confirmed violations, but header overlap prevented two automated contrast determinations. No horizontal overflow was detected in the document or app scroll container at the tested widths.

The functional UI checks still pass. This distinction matters: a working copy button and an attractive white layout do not establish accessibility for older users. Physical device, screen-reader and representative-user checks remain.

## Improvements in priority order

1. **Risk detection and ordinary controls:** add independently reviewed examples of login-number requests, premium-call payment cues, code notifications and genuine Settings updates. Improve using development data, then evaluate a new untouched set. Treat the current challenge as regression material from now on.
2. **RAG relevance and context:** evaluate sentence selection separately from citation validity. Require explanations to address the current request/state and preserve necessary conditions; abstain when relevance is weak. Check source instructions against the approved action context.
3. **Structured extraction reliability:** inspect the two invalid outputs, improve relevant training examples and generation reliability, and retain useful failure diagnostics. Do not weaken exact-evidence validation merely to increase pass counts.
4. **UI contrast and usability:** darken warning text, rerun the result-page scan, then test keyboard/screen-reader/physical-phone use with representative users.

The project is useful as a GenAI/FDE demonstration of local fine-tuning, RAG, application policy, observability and evaluation. A defensible portfolio should show these failures and measured limitations alongside the architecture and successful flows.

## Evidence and reproduction

- [Complete 70-case outputs, metrics and fingerprints](full_product_evaluation.json)
- [Data/token audit](full_data_audit.json)
- [Frozen authored challenge and labeling notes](../data/evaluation/README.md)
- [Raw-generation diagnostics for the two invalid challenge outputs](full_failure_diagnostics.json)
- [Initial-page accessibility scan](full_ui_accessibility.json)
- [Full browser flow and result-page contrast finding](full_ui_flow.json)

```bash
python -m easy_tech_help.dataset --check-tokens --report results/full_data_audit.json
python -m easy_tech_help.sms_source
python -m pytest -q
EASY_TECH_HELP_RUN_LIVE_TESTS=1 python -m pytest tests/test_analysis_live.py -q
ruff check .
ruff format --check .
python -m easy_tech_help.product_evaluation --device mps --output results/full_product_evaluation.json
```

The product evaluator exits with status 1 because scored quality checks fail; it still writes the complete report. It does not assert that all regression rows passed a full end-to-end gold answer: regression observation accuracy is scored against existing labels, while sensitive-level recovery is separately checked on its seven applicable cases. Challenge and development denominators are kept separate. The `mean_seconds` fields under raw/reviewed regression metrics refer to full-pipeline case time in this report, not extraction-only latency.

The evaluation sets were already inspected or authored by the assistant, synthetic Wi-Fi/alert data remain prominent, and the public SMS corpus may have appeared in pretraining. No independent real-world accuracy, end-user resolution rate or accessibility certification is claimed.
