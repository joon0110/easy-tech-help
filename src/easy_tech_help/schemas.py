"""Structured text observations, checked against the actual user input."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    model_validator,
)

MAX_TEXT_CHARS = 4_000
Category = Literal["message", "alert", "wifi", "unknown"]
Signal = Literal[
    "urgent_security_warning",
    "support_phone_number",
    "payment_request",
    "credential_request",
    "verification_code_request",
    "visible_link",
    "install_request",
    "remote_access_request",
    "wifi_off",
    "airplane_mode_on",
    "wifi_connected",
    "no_internet",
    "wifi_password_prompt",
    "unsecured_network",
]
AnalysisIssue = Literal[
    "unsupported",
    "insufficient_context",
    "contradictory_input",
    "invalid_model_output",
    "incomplete_model_output",
]
CONNECTIVITY_SIGNALS = {
    "wifi_off",
    "airplane_mode_on",
    "wifi_connected",
    "no_internet",
    "wifi_password_prompt",
    "unsecured_network",
}
SUMMARIES = {
    "message": "This is a message or email.",
    "alert": "This is a notification or popup.",
    "wifi": "This describes an iPhone Wi-Fi connection.",
    "unknown": "There is not enough information to identify the situation reliably.",
}
ISSUE_TEXT = {
    "unsupported": "V1 supports English messages, alerts and iPhone Wi-Fi descriptions.",
    "insufficient_context": "Please include the message text or describe the current situation.",
    "contradictory_input": "The descriptions of the current state contradict each other.",
    "invalid_model_output": "The model output did not pass validation.",
    "incomplete_model_output": "The model did not finish its response.",
}


def validate_input(text: str) -> str:
    """Reject invalid inputs before contacting a model; preserve exact quotes."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter text to analyze.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Please enter no more than {MAX_TEXT_CHARS:,} characters.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        raise ValueError("Please enter plain text only.")
    return text


class ObservedSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    signal: Signal
    evidence: Annotated[str, Field(min_length=1, max_length=500)] = Field(
        description="An exact, case-sensitive substring of the user's input text."
    )


class TextObservation(BaseModel):
    """Validate structure and evidence presence, not authenticity or meaning."""

    model_config = ConfigDict(extra="forbid")
    category: Category
    signals: list[ObservedSignal] = Field(max_length=8)
    issues: list[AnalysisIssue] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_observation(self, info: ValidationInfo) -> "TextObservation":
        if self.category == "unknown" and not self.issues:
            self.issues = ["insufficient_context"]
        if self.issues:
            self.category = "unknown"
            self.signals = []
        names = [item.signal for item in self.signals]
        if len(names) != len(set(names)):
            raise ValueError("signals must be unique")
        if {"wifi_off", "wifi_connected"}.issubset(names):
            raise ValueError("Wi-Fi cannot currently be both off and connected")
        if self.category != "wifi" and CONNECTIVITY_SIGNALS.intersection(names):
            raise ValueError("connectivity signals require a Wi-Fi description")
        source = (info.context or {}).get("input_text")
        for item in self.signals:
            if not isinstance(source, str) or item.evidence not in source:
                raise ValueError("evidence must quote the actual input text")
        return self

    @computed_field
    @property
    def summary(self) -> str:
        return SUMMARIES[self.category]

    @computed_field
    @property
    def uncertainty(self) -> str:
        if self.issues:
            return " ".join(ISSUE_TEXT[issue] for issue in dict.fromkeys(self.issues))
        return (
            "This analysis describes the text you entered. It does not verify the sender "
            "or the actual connection. No detected warning signs does not guarantee safety."
        )

    def training_target(self) -> dict:
        """Only model-authored fields, shared by inference and fine-tuning."""
        return self.model_dump(include={"category", "signals", "issues"})

    @classmethod
    def unknown(cls, reason: AnalysisIssue) -> "TextObservation":
        return cls(category="unknown", signals=[], issues=[reason])
