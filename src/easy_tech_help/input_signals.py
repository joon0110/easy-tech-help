"""Extract literal request and Wi-Fi cues from English text."""

import re

BOUNDARY = re.compile(r"[.!?;\n](?!\d)")
INACTIVE = re.compile(
    r"\b(?:yesterday|previously|earlier|example|lesson|article|hypothetical)\b",
    re.IGNORECASE,
)
REQUEST = r"\b(?:enter|send|share|provide|submit|type|reply|tell|give|forward|read|pay|transfer|buy|allow|enable|grant|install|call)\b"
PATTERNS = {
    "credential_request": r"\b(?:enter|send|share|provide|submit|type|tell)\b[^.!?;\n]{0,80}?\b(?:(?:account|login|email|apple)\s+)?password\b",
    "verification_code_request": r"\b(?:enter|send|share|provide|reply|tell|give|forward|read)\b(?:(?![.!?;]\s|\n).){0,160}?\b(?:(?:one[- ]time|six[- ]digit|login|verification|security)\s+)+(?:code|number|passcode)\b",
    "payment_request": r"\b(?:pay|send|transfer)\s+(?:me\s+)?(?:[$£€]\s*\d|\d+\s*(?:dollars?|pounds?|euros?)|money\b)|\b(?:buy|pay with|send)\s+(?:(?:a|the|\d+|two|three)\s+)?gift cards?\b",
    "remote_access_request": r"\b(?:allow|enable|grant|give)\b[^.!?;\n]{0,60}\b(?:remote access|remote control|screen sharing)\b",
}
WIFI_PATTERNS = {
    "wifi_connected": r"\b(?:blue\s+)?check\s?mark\b|\b(?:wi-?fi\s+(?:is\s+)?connected|connected\s+to\s+(?:(?:my|the|a)\s+)?(?:home\s+)?wi-?fi)\b|\b(?:it|iphone|phone)\s+(?:is\s+)?(?:also\s+)?connected\s+to\s+[^.;\n]{1,60}",
    "no_internet": r"\bno internet(?: connection| access)?\b",
    "airplane_mode_on": r"\bairplane mode(?:\s+is)?\s+(?:turned\s+|switched\s+)?on\b",
    "wifi_off": r"\bwi-?fi\b[^.!?;\n]{0,35}\b(?:turned|switched) off\b|\bwi-?fi\s+(?:is\s+)?off\b",
    "wifi_password_prompt": r"\b(?:asks? for|enter)\b[^.!?;\n]{0,35}\b(?:wi-?fi|network) password\b",
    "unsecured_network": r"\bunsecured\s+(?:public\s+)?(?:wi-?fi\s+)?network\b",
}


def active_context(text: str, start: int, end: int) -> bool:
    left = max((m.end() for m in BOUNDARY.finditer(text, 0, start)), default=0)
    span = text[start:end]
    verb = re.search(REQUEST, span, re.IGNORECASE)
    cue = start + verb.start() if verb else start
    prefix = re.sub(r"https?://\S+", "", text[left:cue]).replace("’", "'")
    if INACTIVE.search(prefix) or INACTIVE.match(span.strip()):
        return False
    return not re.search(
        r"\b(?:never|not|don't|avoid|no need to)\s+(?:\w+\s+){0,3}$",
        prefix,
        re.IGNORECASE,
    )


def literal_cues(text: str) -> dict[str, str]:
    cues = {}
    for name, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if not active_context(text, *match.span()):
                continue
            if (
                name == "credential_request"
                and re.search(r"\b(?:wi-?fi|network|router)\b", text, re.IGNORECASE)
                and not re.search(
                    r"\b(?:account|login|email|apple)\b", match.group(), re.IGNORECASE
                )
            ):
                continue
            cues.setdefault(name, match.group())
    if code_notice(text):
        for match in re.finditer(
            r"\b(?:send|share|tell|give|forward|read)\s+(?:it|this code|that code)\b",
            text,
            re.IGNORECASE,
        ):
            if active_context(text, *match.span()):
                cues.setdefault("verification_code_request", match.group())
    # A price alone is not a request. Match a call instruction followed by
    # an explicit charge within the bounded passage.
    for match in re.finditer(
        r"\bcall\b[^!?;\n]{0,250}\b(?:cost|calls? cost|charged?)\s*[: ]?\s*[$£€]\s*\d+(?:\.\d+)?(?:\s*/\s*(?:p?m|min(?:ute)?s?))?",
        text,
        re.IGNORECASE,
    ):
        if active_context(text, *match.span()):
            cues.setdefault("payment_request", match.group())
    if re.search(r"\biphone\b", text, re.IGNORECASE) and re.search(
        r"\bwi-?fi\b", text, re.IGNORECASE
    ):
        for name, pattern in WIFI_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if active_context(text, *match.span()):
                    cues.setdefault(name, match.group())
    for match in re.finditer(r"https?://[^\s<>\"]+", text, re.IGNORECASE):
        if active_context(text, *match.span()):
            cues.setdefault("visible_link", match.group().rstrip(".,;!?)"))
    for match in re.finditer(
        r"\b(?:support|help)\b[^\n]{0,70}?(?:\[PHONE\]|\+?[\d(][\d ()+-]{6,}\d)",
        text,
        re.IGNORECASE,
    ):
        if active_context(text, *match.span()):
            cues.setdefault("support_phone_number", match.group())
    return cues


def code_notice(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:login|verification|security|one[- ]time)\s+(?:code|number|passcode)\s*(?:is|:)\s*\d{4,8}\b",
            text,
            re.IGNORECASE,
        )
    )


def settings_update(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:in|open(?:ed)?)\b[^.!?\n]{0,30}\b(?:iphone\s+)?settings\b[^.!?\n]{0,40}\bsoftware update\b",
            text,
            re.IGNORECASE,
        )
        and not re.search(
            r"https?://|\b(?:safari|browser|webpage|caller|email|text message)\b",
            text,
            re.IGNORECASE,
        )
    )


def notification_only_signal(text, name, evidence, cues):
    if not code_notice(text) or "verification_code_request" in cues:
        return False
    if name == "verification_code_request":
        return True
    return (
        name == "urgent_security_warning"
        and bool(
            re.search(
                r"\b(?:expires?|valid for|(?:login|verification|security) code (?:is|:))\b",
                evidence,
                re.IGNORECASE,
            )
        )
        and not re.search(
            r"\b(?:account|suspend|close|closed|lock|locked|virus|infected)\b",
            evidence,
            re.IGNORECASE,
        )
    )


def link_needs_check(text):
    context = re.sub(r"https?://\S+", "", text, flags=re.IGNORECASE)
    return len(context.split()) <= 3 or bool(
        re.search(
            r"\b(?:click|tap|open|verify|sign in|log in|update|confirm|claim|visit)\b",
            context,
            re.IGNORECASE,
        )
    )


def reported_alert(text):
    return bool(
        re.match(
            r"\s*(?:(?:a|an|my)\s+)?(?:(?:safari|browser|iphone)\s+)?(?:pop.?up|notification|alert|web warning)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:i get|i see|my iphone shows)\s+(?:a|an)\s+(?:prompt|popup|notification)\b",
            text,
            re.IGNORECASE,
        )
    )
