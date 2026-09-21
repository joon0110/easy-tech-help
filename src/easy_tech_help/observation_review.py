"""Review extraction errors while keeping the raw prediction."""

import json
import re

from easy_tech_help.input_signals import (
    literal_cues,
    notification_only_signal,
    reported_alert,
    settings_update,
)
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
    text: str, observation: TextObservation, generation_text: str | None = None
) -> tuple[TextObservation, list[str]]:
    """Correct supported extraction errors after schema validation.

    Do not infer requests from password mentions or claim a sender is fraudulent.
    Recover only known evidence errors supported by independent literal cues.
    """
    notes = []
    cues = literal_cues(text)
    if {"wifi_off", "wifi_connected"} <= cues.keys():
        return TextObservation.unknown("contradictory_input"), [
            "conflicting_current_wifi_states"
        ]
    if (
        re.match(r"\s*(?:SYSTEM|ASSISTANT|ANALYZER)\s*:", text)
        and re.search(r"\b(?:schema|output|return|classify)\b", text, re.IGNORECASE)
        and not {
            "credential_request",
            "verification_code_request",
            "payment_request",
            "remote_access_request",
        }.intersection(cues)
    ):
        return TextObservation.unknown("unsupported"), [
            "analyzer_command_not_phone_context"
        ]
    if observation.issues == ["invalid_model_output"] and generation_text:
        # Recover supported quote errors, then validate the complete object again.
        try:
            target = json.loads(generation_text)
            for item in target["signals"]:
                if (
                    isinstance(item["evidence"], str)
                    and item["evidence"]
                    and item["evidence"] not in text
                ):
                    exact = list(
                        re.finditer(re.escape(item["evidence"]), text, re.IGNORECASE)
                    )
                    if len(exact) == 1:
                        item["evidence"] = exact[0].group()
                        notes.append("literal_evidence_case_restored:" + item["signal"])
                if (
                    item["evidence"] not in text
                    and item["signal"]
                    in {"wifi_connected", "verification_code_request"}
                    and item["signal"] in cues
                    and re.search(
                        r"connect|checkmark|code|number|passcode",
                        item["evidence"],
                        re.IGNORECASE,
                    )
                ):
                    item["evidence"] = cues[item["signal"]]
                    notes.append("literal_evidence_regrounded:" + item["signal"])
            observation = TextObservation.model_validate(
                target, context={"input_text": text}
            )
        except (ValueError, TypeError, KeyError):
            return observation, []
    if observation.category == "unknown" or observation.issues:
        return observation, []
    if CONTEXTLESS.fullmatch(text.strip().rstrip(".!? ")):
        return TextObservation.unknown("insufficient_context"), [
            "contextless_help_request"
        ]
    target = observation.training_target()
    if target["category"] == "wifi":
        corrected = []
        for item in target["signals"]:
            if item["signal"] in {
                "wifi_connected",
                "unsecured_network",
            } and re.fullmatch(
                r"no internet(?: connection| access)?[.!]?",
                item["evidence"].strip(),
                re.IGNORECASE,
            ):
                # Internet reachability does not establish network security or
                # joining state. Keep either only with independent input evidence.
                name = item["signal"]
                notes.append("no_internet_not_evidence_for:" + name)
                if name not in cues:
                    continue
                item["evidence"] = cues[name]
            corrected.append(item)
        target["signals"] = corrected
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
    kept = [
        s
        for s in target["signals"]
        if not notification_only_signal(text, s["signal"], s["evidence"], cues)
    ]
    if len(kept) != len(target["signals"]):
        notes.append("code_notification_not_disclosure_request")
    target["signals"] = kept
    target["signals"] = [
        s
        for s in target["signals"]
        if not (
            s["signal"] == "credential_request"
            and re.search(r"gift cards?", text, re.IGNORECASE)
            and re.search(r"redemption|redeem", s["evidence"], re.IGNORECASE)
            and not re.search(r"password|account", s["evidence"], re.IGNORECASE)
        )
    ]
    if "visible_link" not in cues:
        target["signals"] = [
            s
            for s in target["signals"]
            if s["signal"] != "visible_link"
            or re.search(r"https?://|www\.", s["evidence"], re.IGNORECASE)
        ]
    if settings_update(text) and target["category"] != "alert":
        target["category"] = "alert"
        notes.append("settings_update_is_alert")
    elif reported_alert(text) and target["category"] == "message":
        target["category"] = "alert"
        notes.append("explicit_alert_context")
    names = {s["signal"] for s in target["signals"]}
    for name, evidence in cues.items():
        if name in names or name == "credential_request" or len(target["signals"]) >= 8:
            continue
        if (
            name
            in {
                "verification_code_request",
                "payment_request",
                "remote_access_request",
                "visible_link",
                "support_phone_number",
            }
            or target["category"] == "wifi"
        ):
            target["signals"].append({"signal": name, "evidence": evidence})
            notes.append("literal_cue_added:" + name)
    if {"wifi_off", "wifi_connected"} <= {s["signal"] for s in target["signals"]}:
        return TextObservation.unknown("contradictory_input"), notes + [
            "conflicting_current_wifi_states"
        ]
    return TextObservation.model_validate(target, context={"input_text": text}), notes
