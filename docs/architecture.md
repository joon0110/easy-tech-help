# Architecture

EasyTechHelp is being built as a single-iPhone-screenshot Streamlit application that runs on the user's own computer. The implemented pieces so far are the app shell, installable Python package, local configuration loader, local vision analysis with Pydantic validation, and an offline document retriever. The vision analysis is currently available from the command line.

## Planned V1 data flow

```text
Screenshot
  -> local vision model on 127.0.0.1
  -> Pydantic validation
  -> retrieval from reviewed local help documents
  -> local model drafts a grounded explanation
  -> application safety rules construct final actions
  -> short on-screen guidance with source links
  -> optional copyable family handoff
  -> fixed benchmark evaluation
```

The model will describe what it can observe and may use retrieved documents for a grounded explanation. Application code will decide which safe actions to show. Retrieved documents are evidence for explanations, not instructions that can override safety rules. Text visible inside a screenshot can be misleading, and a model can misread a screen or propose an unsafe action.

## Implemented screen-analysis boundary

`analyze_screenshot()` checks image bytes, calls only the local Ollama chat endpoint, checks that generation finished normally, and validates the response with Pydantic. Network failures remain explicit errors. Malformed, inconsistent, or incomplete responses produce `unknown`.

The model supplies only category, text fragments, signals with evidence quotes, and quality issues. The application creates the summary and uncertainty text from fixed templates. The earlier free-form `likely_user_goal` field was removed because it encouraged unsupported user-intent and authenticity claims. `unexpected_link` became `visible_link`: an image can show a link but cannot establish whether the user expected it.

Evidence must occur in an extracted text fragment; duplicate signals, conflicting Wi-Fi states, and connectivity signals outside a Wi-Fi screen are rejected. This checks internal consistency, not whether OCR or the signal interpretation is correct. Icon-only states may be omitted because they have no text evidence. When the model reports blur, missing context, or another quality issue, the application forces `unknown` and clears the signals. Recognizing those quality issues still depends on the model.

Text inside images is untrusted input, including apparent system prompts. The pipeline has no tools for acting on that text, and it cannot supply application-authored summaries or recommendations. Future retrieval and safety stages must preserve this boundary. `tests/test_analysis_live.py` exercises seven development images through the real model when explicitly enabled, separately from the final 24-image benchmark.

## Planned responsibilities

| Part | Responsibility |
| --- | --- |
| `app/app.py` | Streamlit presentation and interaction |
| `src/easy_tech_help/config.py` | Optional local model setting |
| `src/easy_tech_help/analysis.py` | Validate one image and request local Ollama vision analysis |
| `src/easy_tech_help/retrieval.py` | Search reviewed local documents without a service or database |
| `knowledge/original/` | Original FTC HTML pages searched locally after article-text extraction |
| `knowledge/apple_summaries/` | Original short summaries based on Apple support pages, clearly labeled in the catalog |
| `src/easy_tech_help/schemas.py` | Validate allowed screen, signal, and uncertainty values |
| Safety module | Derive risk and permitted guidance from validated observations |
| Handoff module | Build a copyable summary from the final safe state |
| `eval/` | Run a fixed 24-screenshot benchmark against the complete pipeline, including retrieval quality |

The safety, handoff, and evaluation modules are planned for later milestones. Screenshot analysis currently returns observations only and does not use retrieval or issue recommendations. V1 will support iPhone Safari pop-ups, iPhone messages, iPhone Wi-Fi, and an unknown outcome. Android screens are unsupported. Apple-based summaries cover common iPhone steps; retrieved content must still be checked against the visible screen and the application's safety rules. The retrieval module alone is not a complete RAG pipeline until a local generative model uses its results. V1 will not authenticate senders or networks from a screenshot, control a device, send messages to family members, or permanently store uploaded screenshots.
