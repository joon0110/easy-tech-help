# Architecture

EasyTechHelp is being built as a single-screenshot Streamlit application. The only implemented pieces in Milestone 1 are the app shell, installable Python package, and local configuration loader.

## Planned V1 data flow

```text
Screenshot
  -> multimodal screen observations
  -> Pydantic validation
  -> application safety rules
  -> short on-screen guidance
  -> optional copyable family handoff
  -> fixed benchmark evaluation
```

The model will describe what it can observe. Application code will decide which safe actions to show. This separation matters because text visible inside a screenshot can be misleading, and a model can misread a screen or propose an unsafe action.

## Planned responsibilities

| Part | Responsibility |
| --- | --- |
| `app/app.py` | Streamlit presentation and interaction |
| `src/easy_tech_help/config.py` | Optional local environment settings; never display the API key |
| Analysis module | Call one multimodal provider and return screen observations |
| Schema module | Validate allowed screen, signal, and uncertainty values |
| Safety module | Derive risk and permitted guidance from validated observations |
| Handoff module | Build a copyable summary from the final safe state |
| `eval/` | Run a fixed 24-screenshot benchmark against the complete pipeline |

The analysis, schema, safety, handoff, and evaluation modules are planned for later milestones. V1 will support pop-ups, messages, Wi-Fi, and an unknown outcome. It will not authenticate senders or networks from a screenshot, control a device, or send messages to family members.
