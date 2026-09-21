"""Conservative application corrections; these do not change model predictions."""

import re

from easy_tech_help.schemas import TextObservation

# Match the whole input, not occurrences within an actual message or device state.
CONTEXTLESS = re.compile(
    r"(?:please\s+)?(?:help(?:\s+me)?(?:\s+(?:with\s+)?(?:this|that|it))?"
    r"|(?:can|could|would)\s+you\s+(?:please\s+)?help(?:\s+me)?"
    r"(?:\s+(?:with\s+)?(?:this|that|it))?"
    r"|(?:i\s+)?need\s+(?:some\s+)?help"
    r"|what\s+(?:should|do|can)\s+i\s+do(?:\s+(?:now|next))?)",
    re.IGNORECASE,
)
PASSWORD_REQUEST = re.compile(
    r"(?:please\s+)?(?:enter|send|share|provide|type|submit|confirm)\s+"
    r"(?:me\s+)?(?:your|the|my)\s+(?:account\s+|login\s+|email\s+)?password\b",
    re.IGNORECASE,
)
MONEY = re.compile(
    r"\b(?:pay\w*|money|card|cash|fee|charge\w*|cost\w*|premium|subscription|"
    r"gift|bank|credit|debit|dollars?|pounds?|euros?|usd|gbp|eur)\b|[$£€]",
    re.IGNORECASE,
)
NONCURRENT = re.compile(
    r"\b(?:never|not|don't|do not|avoid|yesterday|previously|earlier|used to|"
    r"example|hypothetical|training|lesson|article|warn\w*|learn\w*|"
    r"ignore|system|assistant|analyzer)\b",
    re.IGNORECASE,
)


def review_observation(
    text: str, observation: TextObservation
) -> tuple[TextObservation, list[str]]:
    """Apply narrow, auditable corrections after raw model schema validation.

    Do not infer requests from a password mention, recover invalid generations,
    or claim the sender is fraudulent. All replacement evidence stays literal.
    """
    if observation.category == "unknown" or observation.issues:
        return observation, []
    if CONTEXTLESS.fullmatch(text.strip().rstrip(".!? ")):
        return TextObservation.unknown("insufficient_context"), [
            "contextless_help_request"
        ]
    target = observation.training_target()
    notes = []
    context = re.sub(r"https?://\S+", "", text).replace("’", "'")
    if observation.category in {"message", "alert"} and not NONCURRENT.search(context):
        names = {s.signal for s in observation.signals}
        for item in target["signals"]:
            evidence = item["evidence"]
            request = PASSWORD_REQUEST.match(evidence)
            if (
                item["signal"] == "payment_request"
                and request
                and not MONEY.search(evidence)
                and not re.search(
                    r"\b(?:wi-?fi|network|router|hotspot)\b", text, re.IGNORECASE
                )
            ):
                # Only fix a misnamed existing explicit request. Do not add
                # credential signals to every sentence mentioning a password.
                if "credential_request" not in names:
                    item["signal"] = "credential_request"
                    item["evidence"] = evidence[request.start() : request.end()]
                    names.add("credential_request")
                else:
                    item["signal"] = None
                notes.append("password_request_mislabeled_as_payment")
    target["signals"] = [s for s in target["signals"] if s["signal"] is not None]
    return TextObservation.model_validate(target, context={"input_text": text}), notes
