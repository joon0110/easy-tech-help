"""Verify text submission and errors through the real Streamlit interaction layer."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from easy_tech_help import analysis
from easy_tech_help.schemas import TextObservation

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_form_submits_text_and_shows_validated_result(monkeypatch):
    calls = []

    def fake_analyze(text, *, model):
        calls.append((text, model))
        return TextObservation.model_validate(
            {
                "category": "message",
                "signals": [
                    {"signal": "visible_link", "evidence": "https://notice.example"}
                ],
                "issues": [],
            },
            context={"input_text": text},
        )

    monkeypatch.setattr(analysis, "analyze_text", fake_analyze)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("Message: https://notice.example").run()
    app.button[0].click().run()
    assert not app.exception
    assert calls[0][0] == "Message: https://notice.example"
    assert app.text[0].value == "https://notice.example"
    assert len(app.json) == 1


def test_model_failure_is_displayed_without_app_crash(monkeypatch):
    def fail(*args, **kwargs):
        raise analysis.LocalModelError("Cannot reach local Ollama")

    monkeypatch.setattr(analysis, "analyze_text", fail)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("iPhone Wi-Fi is off.").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Cannot reach local Ollama"
    assert len(app.json) == 0
