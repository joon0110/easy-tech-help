"""Corrections must preserve normal, negated, historical and payment controls."""

import pytest

from easy_tech_help.observation_review import review_observation
from easy_tech_help.schemas import TextObservation


def prediction(text, evidence=None, category="message"):
    return TextObservation.model_validate(
        {
            "category": category,
            "signals": []
            if evidence is None
            else [{"signal": "payment_request", "evidence": evidence}],
            "issues": [],
        },
        context={"input_text": text},
    )


@pytest.mark.parametrize(
    "text",
    [
        "Please help me with this.",
        "Could you help me?",
        "What should I do now?",
        "I need some help!",
    ],
)
def test_generic_help_requests_need_context(text):
    reviewed, notes = review_observation(text, prediction(text))
    assert reviewed.category == "unknown"
    assert reviewed.issues == ["insufficient_context"]
    assert notes == ["contextless_help_request"]


@pytest.mark.parametrize(
    "text",
    [
        "Text message: Please help me.",
        "I need help with Wi-Fi.",
        "Please help me move the table tomorrow.",
        "My phone says help.",
        "Hello!",
    ],
)
def test_real_content_is_not_removed_by_contextless_guard(text):
    original = prediction(text)
    assert review_observation(text, original) == (original, [])


@pytest.mark.parametrize(
    "evidence",
    [
        "Enter your password at https://account.example",
        "Please send me your login password",
        "Share your account password",
    ],
)
def test_password_request_is_not_a_payment(evidence):
    text = "Text message: " + evidence
    original = prediction(text, evidence)
    reviewed, notes = review_observation(text, original)
    assert reviewed.signals[0].signal == "credential_request"
    assert reviewed.signals[0].evidence in text
    assert original.signals[0].signal == "payment_request"
    assert notes == ["password_request_mislabeled_as_payment"]


@pytest.mark.parametrize(
    ("text", "evidence"),
    [
        ("Enter your password to pay a $5 fee.", "Enter your password to pay a $5 fee"),
        ("Never share your password.", "share your password"),
        ("Don’t share your password.", "share your password"),
        ("Yesterday it asked: Enter your password.", "Enter your password"),
        ("A lesson says scammers ask: Enter your password.", "Enter your password"),
        ("iPhone Wi-Fi: Enter your password.", "Enter your password"),
        ("Your password has been updated.", "Your password has been updated"),
        ("Pay $10 using your saved password.", "Pay $10 using your saved password"),
    ],
)
def test_ambiguous_or_noncurrent_predictions_are_not_reclassified(text, evidence):
    original = prediction(text, evidence)
    assert review_observation(text, original) == (original, [])


def test_invalid_model_output_is_not_recovered_by_rules():
    original = TextObservation.unknown("invalid_model_output")
    assert review_observation("Please help.", original) == (original, [])
