"""Verify text submission and errors through the real Streamlit interaction layer."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from easy_tech_help import analysis, rag
from easy_tech_help.rag import Citation, RagResult, retrieve_references
from easy_tech_help.schemas import TextObservation

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_form_submits_text_and_shows_validated_result(monkeypatch):
    calls = []

    def fake_analyze(text, *, model):
        calls.append((text, model))
        observation = TextObservation.model_validate(
            {
                "category": "message",
                "signals": [
                    {"signal": "visible_link", "evidence": "https://notice.example"}
                ],
                "issues": [],
            },
            context={"input_text": text},
        )
        reference = retrieve_references(text, observation)[0]
        return RagResult(
            observation,
            "answered",
            explanation=reference.excerpt,
            citations=[Citation(reference, reference.excerpt)],
            retrieved=[reference],
        )

    monkeypatch.setattr(rag, "explain_text", fake_analyze)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("Message: https://notice.example").run()
    app.button[0].click().run()
    assert not app.exception
    assert calls[0][0] == "Message: https://notice.example"
    assert "https://notice.example" in [t.value for t in app.text]
    assert app.text[0].value == app.text[1].value
    assert any("What the reference says" == s.value for s in app.subheader)
    assert app.get("link_button")[0].proto.url.startswith("https://consumer.ftc.gov/")
    assert len(app.json) == 1


def test_model_failure_is_displayed_without_app_crash(monkeypatch):
    def fail(*args, **kwargs):
        raise analysis.LocalModelError("Local PyTorch adapter is missing")

    monkeypatch.setattr(rag, "explain_text", fail)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("iPhone Wi-Fi is off.").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Local PyTorch adapter is missing"
    assert len(app.json) == 0


def test_no_source_is_explicit_and_does_not_display_a_generated_answer(monkeypatch):
    observation = TextObservation(category="message", signals=[], issues=[])
    monkeypatch.setattr(
        rag,
        "explain_text",
        lambda *a, **k: RagResult(observation, "insufficient_evidence"),
    )
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("See you at the book club.").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("do not provide enough matching evidence" in i.value for i in app.info)
    assert not app.get("link_button")


def test_apple_summary_is_labeled_as_summary(monkeypatch):
    observation = TextObservation(category="wifi", signals=[], issues=[])
    reference = retrieve_references("iPhone Wi-Fi off", observation)[0]
    monkeypatch.setattr(
        rag,
        "explain_text",
        lambda *a, **k: RagResult(
            observation,
            "answered",
            explanation="Wi-Fi is described as off.",
            citations=[Citation(reference, reference.excerpt)],
            retrieved=[reference],
        ),
    )
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("iPhone Wi-Fi off").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Apple-based summary" in c.value for c in app.caption)
    assert app.get("link_button")[0].proto.url.startswith("https://support.apple.com/")


def test_failed_generation_can_show_multiple_chunks_from_same_document(monkeypatch):
    observation = TextObservation(category="wifi", signals=[], issues=[])
    references = retrieve_references(
        "iPhone connected Wi-Fi no internet connection", observation
    )
    assert len({r.document_id for r in references}) < len(references)
    monkeypatch.setattr(
        rag,
        "explain_text",
        lambda *a, **k: RagResult(
            observation,
            "invalid_citation",
            retrieved=references,
            generation={"text": "Untrusted hidden draft"},
        ),
    )
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("iPhone connected Wi-Fi no internet").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.get("link_button")) == sum(len(r.urls) for r in references)
    assert all("Untrusted hidden draft" not in t.value for t in app.text)
