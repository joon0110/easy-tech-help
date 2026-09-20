"""Constrained observations; application text is never written by the model."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

ScreenType = Literal["popup", "message", "wifi", "unknown"]
VisibleSignal = Literal[
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
]
AnalysisIssue = Literal[
    "blurred",
    "cropped",
    "unreadable",
    "unsupported",
    "insufficient_context",
    "invalid_model_output",
    "incomplete_model_output",
]
TextFragment = Annotated[str, Field(min_length=1, max_length=500)]

SUMMARIES = {
    "popup": "A browser pop-up or alert is visible.",
    "message": "A message or email screen is visible.",
    "wifi": "A Wi-Fi settings screen is visible.",
    "unknown": "The screen could not be identified reliably.",
}
ISSUE_TEXT = {
    "blurred": "The image is too blurred to interpret reliably.",
    "cropped": "The image is missing necessary screen context.",
    "unreadable": "The relevant text cannot be read reliably.",
    "unsupported": "The screen is outside the supported iPhone categories.",
    "insufficient_context": "There is not enough context to identify the screen.",
    "invalid_model_output": "The model output failed validation.",
    "incomplete_model_output": "The model stopped before completing its response.",
}


class ObservedSignal(BaseModel):
    """A proposed signal paired with a quote from the extracted screen text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    signal: VisibleSignal
    evidence: TextFragment = Field(description="An exact quote from visible_text.")


class ScreenObservation(BaseModel):
    """Validate consistency, not the factual accuracy of model observations."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    screen_type: ScreenType
    visible_text: list[TextFragment] = Field(max_length=30)
    visible_signals: list[ObservedSignal] = Field(max_length=5)
    quality_issues: list[AnalysisIssue] = Field(max_length=7)

    @model_validator(mode="after")
    def validate_observations(self) -> "ScreenObservation":
        if self.screen_type == "unknown" and not self.quality_issues:
            self.quality_issues = ["insufficient_context"]
        if self.quality_issues:
            # Unclear screens must not pass actionable signals to later stages.
            self.screen_type = "unknown"
            self.visible_signals = []
        elif not self.visible_text:
            raise ValueError("identified screens require readable text")
        signals = [item.signal for item in self.visible_signals]
        if len(signals) != len(set(signals)):
            raise ValueError("visible signals must be unique")
        if {"wifi_off", "wifi_connected"}.issubset(signals):
            raise ValueError("Wi-Fi cannot be both off and connected")
        connectivity = {
            "wifi_off",
            "airplane_mode_on",
            "wifi_connected",
            "no_internet",
            "wifi_password_prompt",
        }
        if self.screen_type != "wifi" and connectivity.intersection(signals):
            raise ValueError("connectivity signals require a Wi-Fi settings screen")
        fragments = [" ".join(text.split()) for text in self.visible_text]
        for item in self.visible_signals:
            if not any(" ".join(item.evidence.split()) in text for text in fragments):
                raise ValueError("signal evidence must quote visible_text")
        return self

    @computed_field
    @property
    def screen_summary(self) -> str:
        return SUMMARIES[self.screen_type]

    @computed_field
    @property
    def uncertainty(self) -> str:
        if self.quality_issues:
            return " ".join(
                ISSUE_TEXT[issue] for issue in dict.fromkeys(self.quality_issues)
            )
        return (
            "Text and signals are model observations and may be incorrect. "
            "User intent, authenticity, and actual connectivity cannot be verified "
            "from this screenshot alone."
        )

    @classmethod
    def unknown(cls, reason: AnalysisIssue) -> "ScreenObservation":
        return cls(
            screen_type="unknown",
            visible_text=[],
            visible_signals=[],
            quality_issues=[reason],
        )
