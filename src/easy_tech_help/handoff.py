"""Build a family summary from reviewed metadata and approved actions."""

from dataclasses import asdict, dataclass
from pathlib import Path

from easy_tech_help.retrieval import DEFAULT_KNOWLEDGE_DIR
from easy_tech_help.safety import POLICY_VERSION, Guidance, resolve_actions

SITUATIONS = {
    "message": "a message",
    "alert": "an alert or popup",
    "wifi": "an iPhone Wi-Fi situation",
    "unknown": "a situation that needs more context",
}
ASSESSMENTS = {
    "attention": "A sensitive request needs independent checking; this is not a confirmed scam.",
    "check_source": "The source of the request needs checking before acting.",
    "connection_check": "The text describes a connection setting or problem.",
    "uncertain": "There is not enough reliable context or supporting evidence for specific guidance.",
    "no_specific_warning": "No specific concerning request was identified; this is not a safety guarantee.",
}
FLAG_DESCRIPTIONS = {
    "credential_request": "a request for an account password",
    "verification_code_request": "a request for a verification code",
    "payment_request": "a request for payment",
    "remote_access_request": "a request for remote access",
    "install_request": "an installation request",
    "urgent_security_warning": "an urgent security warning",
    "support_phone_number": "a support phone number",
    "visible_link": "a link",
    "wifi_off": "Wi-Fi described as off",
    "airplane_mode_on": "Airplane Mode described as on",
    "wifi_connected": "Wi-Fi described as connected",
    "no_internet": "no internet access reported",
    "wifi_password_prompt": "a network password prompt",
    "unsecured_network": "an unsecured network label",
}
LIMITS = (
    "Actions already taken: not recorded. Resolution: not confirmed.\n"
    "EasyTechHelp checked text only; it did not inspect the phone or verify the sender or network.\n"
    "The original text, names, contact details, network names, passwords and codes are omitted."
)
FALLBACK = (
    "Family help request\n\n"
    "Please help me understand a phone message or connection problem. "
    "The reviewed guidance could not be verified for sharing.\n"
    "Pause before acting on an unexpected request. Keep passwords and verification codes private.\n\n"
    + LIMITS
)


@dataclass(frozen=True)
class FamilyHandoff:
    status: str
    suggested: bool
    text: str
    action_ids: tuple[str, ...] = ()
    source_document_ids: tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)


def build_handoff(
    category: str, guidance: Guidance, directory: Path = DEFAULT_KNOWLEDGE_DIR
) -> FamilyHandoff:
    """Never accept raw input, evidence quotes or model-authored prose.

    Re-resolve actions so altered instructions or source metadata cannot cross
    the sharing boundary. Free-text guidance fields are deliberately ignored.
    """
    try:
        if (
            category not in SITUATIONS
            or guidance.level not in ASSESSMENTS
            or guidance.policy_version != POLICY_VERSION
            or set(guidance.flags) - FLAG_DESCRIPTIONS.keys()
            or guidance.evidence_status
            not in {"verified_local_support", "policy_only", "unavailable"}
        ):
            raise ValueError("Unrecognized reviewed metadata")
        actions = resolve_actions([a.id for a in guidance.next_actions], directory)
        if actions != guidance.next_actions:
            raise ValueError("Actions differ from reviewed templates or evidence")
        sources = {a.evidence.document_id: a.evidence for a in actions if a.evidence}
        if bool(sources) != (guidance.evidence_status == "verified_local_support"):
            raise ValueError("Inconsistent evidence status")
    except (OSError, ValueError, KeyError, TypeError):
        return FamilyHandoff("fallback", True, FALLBACK)

    parts = [
        "Family help request",
        f"I would like help with {SITUATIONS[category]}.",
        ASSESSMENTS[guidance.level],
    ]
    if guidance.flags:
        parts.append(
            "Cues identified in the text (not independently verified): "
            + "; ".join(
                description
                for flag, description in FLAG_DESCRIPTIONS.items()
                if flag in guidance.flags
            )
            + "."
        )
    parts.append(
        "What was checked: the suggested steps have supporting passages in local reference files."
        if sources
        else "What was checked: general application precautions only; no source-backed specific steps are included."
    )
    parts.append(
        "Suggested next steps (not recorded as completed):\n"
        + "\n".join(f"{index}. {a.text}" for index, a in enumerate(actions, 1))
    )
    parts.append(
        "Keep private: account passwords and verification codes. Do not follow an unexpected request to pay, install software or grant remote access."
    )
    if category == "wifi":
        parts.append(
            "Do not join an unfamiliar network, reset network settings, erase the device or disable security."
        )
    parts.append(
        "Help needed: please review the current Wi-Fi status with me and help decide whether further support is needed."
        if category == "wifi"
        else "Help needed: please review the situation with me and help check any unexpected request through a contact route we already trust."
    )
    if sources:
        parts.append(
            "Sources for the suggested steps:\n"
            + "\n".join(
                f"- {'FTC original article' if s.source_type == 'original_html' else 'Apple-based summary; official source'}: {url}"
                for s in sources.values()
                for url in s.urls
            )
        )
    parts.append(LIMITS)
    return FamilyHandoff(
        "ready",
        guidance.level in {"attention", "check_source", "uncertain"},
        "\n\n".join(parts),
        tuple(a.id for a in actions),
        tuple(sources),
    )
