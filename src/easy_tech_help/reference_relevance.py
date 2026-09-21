"""Conservative topic gates for the small English reference corpus.

These are explicit heuristics, not an entailment model. Unmatched topics abstain.
"""

import re

from easy_tech_help.input_signals import (
    code_notice,
    link_needs_check,
    literal_cues,
    settings_update,
)


def explanation_topic(text, observation):
    names = {s.signal for s in observation.signals} | set(literal_cues(text))
    if names == {"visible_link"} and not link_needs_check(text):
        return "none"
    if code_notice(text) and not names.intersection(
        {
            "credential_request",
            "verification_code_request",
            "payment_request",
            "remote_access_request",
        }
    ):
        return "none"
    if settings_update(text) and not names.intersection(
        {
            "credential_request",
            "verification_code_request",
            "payment_request",
            "remote_access_request",
        }
    ):
        return "none"
    if names.intersection({"credential_request", "verification_code_request"}):
        return "secrets"
    if "payment_request" in names:
        return (
            "gift_payment"
            if re.search(r"gift\s*cards?", text, re.IGNORECASE)
            else "payment"
        )
    if "remote_access_request" in names:
        return "remote_support"
    if observation.category == "wifi":
        for name in (
            "no_internet",
            "unsecured_network",
            "wifi_password_prompt",
            "wifi_off",
            "wifi_connected",
        ):
            if name in names:
                return name
        return "wifi_general"
    if observation.category == "alert" and names:
        return "popup"
    if names:
        return "verify_source"
    return "none"


def relevance_score(topic: str, sentence: str, query: str = "") -> int:
    s = sentence.casefold()
    # Do not expose unqualified state-changing instructions or broad guarantees.
    if re.search(
        r"\b(?:reset|erase|forget|disable|usually safe|generally safe)\b|airplane mode is off|is a scammer",
        s,
    ):
        return 0
    if re.search(r"lottery|sweepstakes|won money", s) and not re.search(
        r"lottery|sweepstakes|prize|won", query, re.IGNORECASE
    ):
        return 0
    if topic == "secrets":
        if re.search(r"billing|multi.factor|authentication makes|scan of your", s):
            return 0
        return (
            4
            if re.search(r"password|personal information|verification code", s)
            and re.search(r"steal|trick|do not share|never share|ask for your", s)
            else 0
        )
    if topic == "gift_payment":
        return 4 if "gift card" in s and re.search(r"pay|money|number", s) else 0
    if topic == "payment":
        if (
            "gift card" in s
            or "lottery" in s
            or "remote access" in s
            or len(s.split()) < 12
        ):
            return 0
        return (
            3
            if re.search(r"\bpay\b|\bfee\b|\bpayment\b", s)
            and re.search(r"scam|pressure|insist|impersonat", s)
            else 0
        )
    if topic == "remote_support":
        return 4 if "remote access" in s and re.search(r"scam|impersonat", s) else 0
    if topic == "popup":
        return (
            4
            if re.search(r"pop.?up|webpage|phone number", s)
            and re.search(r"not proof|scam|never|do not|don't|warning", s)
            else 0
        )
    if topic == "no_internet":
        return 5 if "no internet connection" in s and "another device" in s else 0
    if topic == "wifi_off":
        return 5 if re.search(r"wi-fi is on|turn wi-fi on", s) else 0
    if topic == "wifi_password_prompt":
        return (
            5
            if "password" in s and re.search(r"owner|administrator|recognize", s)
            else 0
        )
    if topic == "unsecured_network":
        if "https" in s and "address bar" in s:
            return 6
        return 2 if "encryption" in s and "website" in s and "devices" not in s else 0
    if topic in {"wifi_connected", "wifi_general"}:
        return (
            4
            if re.search(
                r"joining a network and having working internet|blue checkmark", s
            )
            else 0
        )
    if topic == "verify_source":
        return (
            3
            if re.search(
                r"contact the company|website you know|phone number.*know.*real", s
            )
            else 0
        )
    return 0
