# Architecture

EasyTechHelp is being built as a single-iPhone-screenshot Streamlit application that runs on the user's own computer. The implemented pieces so far are the app shell, installable Python package, local configuration loader, and a small offline document retriever.

## Planned V1 data flow

```text
Screenshot
  -> local OCR and/or local vision model
  -> Pydantic validation
  -> retrieval from reviewed local help documents
  -> local model drafts a grounded explanation
  -> application safety rules construct final actions
  -> short on-screen guidance with source links
  -> optional copyable family handoff
  -> fixed benchmark evaluation
```

The model will describe what it can observe and may use retrieved documents for a grounded explanation. Application code will decide which safe actions to show. Retrieved documents are evidence for explanations, not instructions that can override safety rules. Text visible inside a screenshot can be misleading, and a model can misread a screen or propose an unsafe action.

## Planned responsibilities

| Part | Responsibility |
| --- | --- |
| `app/app.py` | Streamlit presentation and interaction |
| `src/easy_tech_help/config.py` | Optional local model setting |
| Analysis module | Use local OCR/vision and return screen observations |
| `src/easy_tech_help/retrieval.py` | Search reviewed local documents without a service or database |
| `knowledge/original/` | Original FTC HTML pages searched locally after article-text extraction |
| `knowledge/apple_summaries/` | Original short summaries based on Apple support pages, clearly labeled in the catalog |
| Schema module | Validate allowed screen, signal, and uncertainty values |
| Safety module | Derive risk and permitted guidance from validated observations |
| Handoff module | Build a copyable summary from the final safe state |
| `eval/` | Run a fixed 24-screenshot benchmark against the complete pipeline, including retrieval quality |

The analysis, schema, safety, handoff, and evaluation modules are planned for later milestones. V1 will support iPhone Safari pop-ups, iPhone messages, iPhone Wi-Fi, and an unknown outcome. Android screens are unsupported. Apple-based summaries cover common iPhone steps; retrieved content must still be checked against the visible screen and the application's safety rules. The retrieval module alone is not a complete RAG pipeline until a local generative model uses its results. V1 will not authenticate senders or networks from a screenshot, control a device, send messages to family members, or permanently store uploaded screenshots.
