# Product interface checks

Date: 2026-09-21. The white, English-only product interface is implemented with Streamlit 1.64 and local CSS. No model weights, extraction prompts or safety rules were changed for this UI work.

## Design and interaction

- White background, dark text, local system fonts, restrained borders and one readable column.
- Large labeled input, three example buttons, one primary action, and visible keyboard focus.
- Next steps and actions to avoid stay visible. Supporting details, sources and family sharing are optional expandable sections.
- Results persist across session reruns. The checked text remains separate from a new draft. Reset clears the input/result/error; failed submissions remove outdated results and retain the draft for retry.
- Family summaries have a copy control visible without hovering and a text-file download. Neither action re-runs the model or sends a message.
- Diagnostic JSON is available in the CLI instead of the product page. Untrusted input is rendered as plain text.

## Verification

| Check | Result |
| --- | --- |
| Full pytest suite | 202 passed, 7 opt-in live tests skipped |
| Streamlit interaction tests | 11 passed, included above |
| Ruff lint / formatting / Git whitespace | Passed |
| Real local-model submission in Chrome | Sensitive-request result, next steps and official source links displayed |
| Clipboard | Copied text exactly matched the family summary; no hover required |
| Download | Downloaded UTF-8 text exactly matched the family summary |
| Input flow | Blank-input error, example filling without inference, submission and reset passed |
| Responsive layout | Desktop at 1440px; 390px and 320px widths checked without horizontal overflow in the document or app scroll container |
| Keyboard | Input could receive focus with a visible outline |
| Browser errors | No uncaught JavaScript errors during the checked flow |

Browser checks used temporary Playwright tooling with the installed Google Chrome and the actual local app/model. The screenshots contain only the built-in synthetic example. Caption opacity and the initially hover-only copy toolbar were corrected after browser inspection.

## Previews

- [Desktop input](ui/desktop.png)
- [Mobile input](ui/mobile.png)
- [Actual sensitive-request result](ui/result.png)

## Repeat the user flow

```bash
source .venv/bin/activate
streamlit run app/app.py
```

Submit an empty input, select **Text message**, then select **Help me understand**. Open **Sources and explanation** and **Ask family for help**. Copy/save the summary, compare its contents, then select **Start a new check**. Repeat at a narrow browser width and with keyboard navigation.

These checks establish behavior for the exercised flow, not comprehensive accessibility certification or usability with older adults. Screen-reader testing, physical phone/browser checks and sessions with representative users remain. UI styling uses Streamlit test-ID selectors and should be rechecked after a Streamlit upgrade. Upstream classification and source-relevance limitations still apply.

## Follow-up accessibility fixes

The full audit subsequently found warning text at 4.43:1 contrast and an overlapping invisible Streamlit header. The quality pass darkened warning text and removed that header. See [current quality results](quality_improvement.md), [initial-page scan](ui_improved_accessibility.json) and [real-model browser flow](ui_improved_flow.json). The earlier table above is a historical UI-stage snapshot; previews show the corrected current UI. Current checks found zero axe violations/incomplete checks on the tested initial and result pages, and all exercised copy/download/reset/source-link flows passed.
