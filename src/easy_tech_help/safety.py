"""Choose fixed next actions and check their local source support."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from easy_tech_help.input_signals import (
    PATTERNS as GUARDS,
)
from easy_tech_help.input_signals import (
    active_context as _active_context,
)
from easy_tech_help.input_signals import (
    link_needs_check,
    literal_cues,
    notification_only_signal,
    settings_update,
)
from easy_tech_help.rag import RagResult
from easy_tech_help.reference_relevance import relevance_score
from easy_tech_help.retrieval import DEFAULT_KNOWLEDGE_DIR, load_documents

POLICY_VERSION = "2"


@dataclass(frozen=True)
class Action:
    id: str
    text: str
    document_id: str = ""
    support: str = ""


# Only application-authored text can enter the next-action panel. Nothing here
# interpolates a user URL, phone number, credential, model string or source HTML.
ACTIONS = MappingProxyType(
    {
        a.id: a
        for a in (
            Action(
                "pause",
                "Pause before acting on an unexpected request. Keep passwords and verification codes private.",
            ),
            Action(
                "clarify",
                "Include the exact wording and where it appeared. For Wi-Fi, describe what Settings > Wi-Fi shows. Leave out passwords, codes and account details.",
            ),
            Action(
                "no_change",
                "No special action is suggested from this text alone. If the situation is unexpected or unclear, ask for help before acting.",
            ),
            Action(
                "verify_independently",
                "If a company is named, contact it through a website or phone number you already know is genuine. Use a route you open yourself, not a link or number in the message.",
                "message_phishing",
                "contact the company using a phone number or website you know is real — not the information in the email.",
            ),
            Action(
                "close_popup",
                "For an unfamiliar popup, close the browser tab using the browser's controls. Avoid buttons inside the popup, including its close button.",
                "popup_safari",
                "If a pop-up is unfamiliar, avoid its links and buttons; even a close button may be fake. Close the Safari tab or window instead.",
            ),
            Action(
                "trusted_help",
                "If you are unsure or already followed the request, stop and discuss what happened with someone you know and trust.",
                "message_avoid_scam",
                "Stop and talk to someone you trust.",
            ),
            Action(
                "wifi_check",
                "Open Settings > Wi-Fi and check whether Wi-Fi is on. Check the displayed network name; this does not confirm internet access.",
                "wifi_iphone",
                "If an iPhone cannot connect to Wi-Fi, open Settings, Wi-Fi and check that Wi-Fi is on. A blue checkmark beside a network name indicates that the phone joined that network; it does not by itself prove that internet access works.",
            ),
            Action(
                "wifi_compare",
                "If this is a network you recognize, check whether another device on the same network can access the internet.",
                "wifi_iphone",
                "For a No Internet Connection message, try the same network on another device to see whether the problem is limited to the iPhone.",
            ),
            Action(
                "wifi_password",
                "For a network you recognize, use Settings > Wi-Fi to join. If you do not know its network password, ask the owner or administrator. Do not paste the password here or enter it into an unfamiliar webpage.",
                "wifi_join_iphone",
                "Choose a network that you recognize, then enter its password if prompted. Ask the network owner or administrator for the password when you do not know it.",
            ),
            Action(
                "wifi_airplane",
                "Check whether Airplane Mode is on. Confirm whether it is intentional before changing any connectivity setting.",
                "wifi_iphone",
                "Check that Airplane Mode is off.",
            ),
            Action(
                "check_encryption",
                "In the browser address bar, check whether the site uses https. Encryption alone does not prove that the website or this network is trustworthy.",
                "wifi_public_safety",
                "Look for a lock symbol or https in the address bar to the left of the website address.",
            ),
        )
    }
)

# These are deliberately absent from ACTIONS; they are audit identifiers, not
# a text blacklist that could be bypassed with different model wording.
PROHIBITED_ACTIONS = (
    "pay_requester",
    "share_password",
    "share_verification_code",
    "call_message_number",
    "open_message_link",
    "grant_remote_access",
    "install_from_popup",
    "disable_security",
    "join_unknown_network",
    "reset_network",
    "forget_network",
    "erase_device",
)


@dataclass(frozen=True)
class ActionEvidence:
    document_id: str
    title: str
    source_type: str
    urls: tuple[str, ...]
    quote: str


@dataclass(frozen=True)
class SafeAction:
    id: str
    text: str
    evidence: ActionEvidence | None = None


@dataclass(frozen=True)
class Guidance:
    level: str
    summary: str
    cautions: tuple[str, ...]
    next_actions: tuple[SafeAction, ...]
    avoid: tuple[str, ...]
    flags: tuple[str, ...]
    reasons: tuple[str, ...]
    evidence_status: str
    policy_version: str = POLICY_VERSION

    def to_dict(self):
        return asdict(self)


SENSITIVE = {
    "credential_request",
    "verification_code_request",
    "payment_request",
    "remote_access_request",
}
CONCERN = SENSITIVE | {
    "urgent_security_warning",
    "support_phone_number",
    "visible_link",
    "install_request",
}


def policy_flags(text: str, result: RagResult) -> tuple[set[str], list[str]]:
    """Combine reviewed observations with literal request cues."""
    flags, reasons = set(), []
    cues = literal_cues(text)
    for signal in result.observation.signals:
        start = text.find(signal.evidence)
        if notification_only_signal(text, signal.signal, signal.evidence, cues):
            reasons.append("ignored_code_notification")
            continue
        if (
            signal.signal == "install_request"
            and settings_update(text)
            and not SENSITIVE.intersection(cues)
        ):
            reasons.append("reported_settings_update_not_untrusted_install")
            continue
        if (
            signal.signal == "credential_request"
            and result.observation.category == "wifi"
            and not re.search(
                r"\b(?:account|login|email|apple)\b", signal.evidence, re.IGNORECASE
            )
        ):
            reasons.append("ignored_network_password_as_account_credential")
            continue
        if (
            signal.signal == "payment_request"
            and re.search(
                r"\b(?:receipt|paid|payment received|payment confirmed)\b",
                signal.evidence,
                re.IGNORECASE,
            )
            and not re.search(GUARDS["payment_request"], signal.evidence, re.IGNORECASE)
        ):
            reasons.append("ignored_receipt_as_payment_request")
            continue
        if start >= 0 and _active_context(text, start, start + len(signal.evidence)):
            flags.add(signal.signal)
        else:
            reasons.append(f"ignored_noncurrent_signal:{signal.signal}")
    for name in literal_cues(text):
        if name not in flags:
            reasons.append(f"input_guard:{name}")
        flags.add(name)
    return flags, reasons


def resolve_actions(
    ids: list[str], directory: Path = DEFAULT_KNOWLEDGE_DIR
) -> tuple[SafeAction, ...]:
    """Resolve allowlisted text and reviewed literal support from local files.

    Binding an action to a support span is deterministic evidence retrieval,
    separate from the model's explanatory source selection.
    """
    if not ids or any(
        name not in ACTIONS or name in PROHIBITED_ACTIONS for name in ids
    ):
        raise ValueError("Only allowlisted action IDs are accepted")
    documents = (
        {d.id: d for d in load_documents(directory)}
        if any(ACTIONS[n].document_id for n in ids)
        else {}
    )
    output = []
    for name in dict.fromkeys(ids):
        action = ACTIONS[name]
        evidence = None
        if action.document_id:
            doc = documents[action.document_id]
            if (
                not action.support
                or action.support not in doc.text
                or not doc.source_urls
                or any(
                    urlsplit(url).scheme != "https"
                    or urlsplit(url).netloc
                    not in {"consumer.ftc.gov", "support.apple.com"}
                    for url in doc.source_urls
                )
            ):
                raise ValueError(
                    "Action support could not be verified in the local corpus"
                )
            evidence = ActionEvidence(
                doc.id, doc.title, doc.source_type, doc.source_urls, action.support
            )
        output.append(SafeAction(action.id, action.text, evidence))
    return tuple(output)


def build_guidance(
    text: str, result: RagResult, directory: Path = DEFAULT_KNOWLEDGE_DIR
) -> Guidance:
    flags, reasons = policy_flags(text, result)
    cautions, avoid = [], []
    if flags & SENSITIVE:
        level = "attention"
        summary = "A sensitive request needs independent checking. This does not establish that the sender is a scammer."
        cautions.append(
            "Requests for money, account secrets or remote access can cause harm if the requester is not genuine."
        )
        avoid.extend(
            (
                "Do not share account passwords or verification codes with the requester.",
                "Do not pay, install software or grant remote access because of this message or popup.",
            )
        )
        ids = ["verify_independently", "trusted_help"]
    elif result.observation.category == "unknown" or result.observation.issues:
        level, summary, ids = (
            "uncertain",
            "There is not enough reliable context to choose a specific next action.",
            ["clarify"],
        )
    elif result.observation.category == "wifi":
        level, summary = (
            "connection_check",
            "This describes a connection setting or problem, not evidence of a scam.",
        )
        cautions.append(
            "Joining Wi-Fi does not guarantee internet access or a trustworthy network."
        )
        avoid.extend(
            (
                "Do not reset network settings, erase the device or disable security as a first step.",
                "Do not join a network you do not recognize.",
            )
        )
        ids = []
        if "airplane_mode_on" in flags:
            ids.append("wifi_airplane")
        if "unsecured_network" in flags:
            ids.append("check_encryption")
            cautions.append(
                "An unsecured network label does not prove an attack. HTTPS also does not prove a site is genuine."
            )
        elif "wifi_password_prompt" in flags:
            ids.append("wifi_password")
        else:
            ids.append("wifi_check")
            if "no_internet" in flags:
                ids.append("wifi_compare")
    elif flags & (CONCERN - {"visible_link"}) or (
        "visible_link" in flags and link_needs_check(text)
    ):
        level, summary = (
            "check_source",
            "Check where this request came from before following it. A link or update notice alone is not proof of a scam.",
        )
        cautions.append(
            "A warning on a webpage does not prove that your iPhone is infected."
            if result.observation.category == "alert"
            else "A sender name or a link does not establish that the message is genuine."
        )
        avoid.extend(
            (
                "Do not use a link or phone number in an unexpected message to sign in or seek support.",
                "Do not install software or allow remote access from an unfamiliar popup.",
            )
        )
        if result.observation.category == "alert" and re.search(
            r"\b(?:browser|safari|pop.?up)\b", text, re.IGNORECASE
        ):
            ids = ["close_popup", "verify_independently"]
        else:
            ids = ["verify_independently"]
    else:
        level, summary, ids = (
            "no_specific_warning",
            "No specific concerning request was identified. This is not a guarantee of safety.",
            ["no_change"],
        )
    if flags & SENSITIVE and result.observation.category == "unknown":
        ids = ["pause", "clarify"]
        reasons.append("uncertain_analysis_with_sensitive_cue")
    if result.status != "answered":
        cautions.append(
            "A sourced explanation was not available. Any action sources below were checked separately against the local reference files."
        )
    try:
        actions = resolve_actions(ids, directory)
        evidence_status = (
            "verified_local_support"
            if any(a.evidence for a in actions)
            else "policy_only"
        )
    except (OSError, ValueError, KeyError, TypeError):
        reasons.append("action_support_unavailable")
        actions = resolve_actions(
            ["pause", "clarify"] if flags & CONCERN else ["clarify"], directory
        )
        evidence_status = "unavailable"
        cautions.append(
            "The local evidence for specific steps could not be checked. Pause and provide more context instead."
        )
        if level != "attention":
            level = "uncertain"
    return Guidance(
        level,
        summary,
        tuple(cautions),
        actions,
        tuple(avoid),
        tuple(sorted(flags)),
        tuple(reasons),
        evidence_status,
    )


def displayable_reference(result: RagResult) -> str:
    """Prevent raw instructions from bypassing the next-action allowlist in the UI.

    Explanatory source text is optional. If it mentions a high-impact operation,
    display its official link only. This conservative filter is an extra display
    guard, not the mechanism that authorizes actions.
    """
    if result.status != "answered" or not result.citations:
        return ""
    if result.reference_topic and any(
        not relevance_score(result.reference_topic, c.quote) for c in result.citations
    ):
        return ""
    if result.explanation != "\n\n".join(c.quote for c in result.citations) or any(
        c.reference not in result.retrieved or c.quote not in c.reference.excerpt
        for c in result.citations
    ):
        return ""
    if re.search(
        r"\b(?:reset|erase|forget|disable|install|download|remote access|remote control|run a scan|usually safe|generally safe)\b",
        result.explanation,
        re.IGNORECASE,
    ):
        return ""
    return result.explanation
