"""Verify text submission and errors through the real Streamlit interaction layer."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from easy_tech_help import analysis, guidance
from easy_tech_help.rag import Citation, RagResult, retrieve_references
from easy_tech_help.schemas import TextObservation

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_form_submits_text_and_shows_validated_result(monkeypatch):
    calls = []

    def fake_analyze(text, *, model, **kwargs):
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

    monkeypatch.setattr(guidance, "explain_text", fake_analyze)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("Message: https://notice.example").run()
    app.button[0].click().run()
    assert not app.exception
    assert calls[0][0] == "Message: https://notice.example"
    assert "https://notice.example" in [t.value for t in app.text]
    assert any("Next steps" == s.value for s in app.subheader)
    assert any("route you open yourself" in t.value for t in app.text)
    assert app.get("link_button")[0].proto.url.startswith("https://consumer.ftc.gov/")
    assert len(app.json) == 0  # Diagnostic JSON is available through the CLI.


def test_model_failure_is_displayed_without_app_crash(monkeypatch):
    def fail(*args, **kwargs):
        raise analysis.LocalModelError("Local PyTorch adapter is missing")

    monkeypatch.setattr(guidance, "explain_text", fail)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("iPhone Wi-Fi is off.").run()
    app.button[0].click().run()
    assert not app.exception
    assert "Your text is still here" in app.error[0].value
    assert app.text_area[0].value == "iPhone Wi-Fi is off."
    assert len(app.json) == 0


def test_no_source_is_explicit_and_does_not_display_a_generated_answer(monkeypatch):
    observation = TextObservation(category="message", signals=[], issues=[])
    monkeypatch.setattr(
        guidance,
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
        guidance,
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
        guidance,
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
    assert len(app.get("link_button")) == len(
        {url for r in references for url in r.urls}
    )
    assert all("Untrusted hidden draft" not in t.value for t in app.text)


def test_source_reset_instruction_is_not_exposed_in_product_panels(monkeypatch):
    text = "iPhone Wi-Fi network reset"
    observation = TextObservation(category="wifi", signals=[], issues=[])
    refs = retrieve_references(text, observation)
    ref = next(r for r in refs if "Reset Network Settings" in r.excerpt)
    monkeypatch.setattr(
        guidance,
        "explain_text",
        lambda *a, **k: RagResult(
            observation,
            "answered",
            explanation=ref.excerpt,
            citations=[Citation(ref, ref.excerpt)],
            retrieved=refs,
        ),
    )
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input(text).run()
    app.button[0].click().run()
    assert not app.exception
    assert all(ref.excerpt not in t.value for t in app.text)
    assert any("Do not reset" in t.value for t in app.text)
    assert any("Open Settings > Wi-Fi" in t.value for t in app.text)


def test_sensitive_request_shows_attention_next_steps_and_avoid_actions(monkeypatch):
    text = "Enter your password."
    monkeypatch.setattr(
        guidance,
        "explain_text",
        lambda *a, **k: RagResult(
            TextObservation(category="message", signals=[], issues=[]),
            "insufficient_evidence",
        ),
    )
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input(text).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.warning
    assert {"Next steps", "What to avoid", "References"}.issubset(
        {h.value for h in app.subheader}
    )
    assert any("Do not share account passwords" in t.value for t in app.text)
    assert len(app.code) == 1
    assert "Family help request" in app.code[0].value
    assert "Actions already taken: not recorded" in app.code[0].value
    assert text not in app.code[0].value
    assert any("Nothing is sent automatically" in c.value for c in app.caption)


def test_example_fills_input_without_running_the_model(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Choosing an example must not run inference")

    monkeypatch.setattr(guidance, "explain_text", unexpected_call)
    app = AppTest.from_file(str(APP)).run()
    app.button(key="example_Wi-Fi problem").click().run()
    assert not app.exception
    assert "No Internet Connection" in app.text_area[0].value
    assert not app.code


def test_empty_input_does_not_run_inference(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Empty input must not load a model")

    monkeypatch.setattr(guidance, "explain_text", unexpected_call)
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert not app.exception
    assert "Please enter text" in app.error[0].value
    assert not app.code


def test_rerun_preserves_result_and_reset_clears_private_session_data(monkeypatch):
    calls = []

    def explain(text, **kwargs):
        calls.append(text)
        return RagResult(
            TextObservation(category="message", signals=[], issues=[]),
            "insufficient_evidence",
        )

    monkeypatch.setattr(guidance, "explain_text", explain)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("Send your verification code.").run()
    app.button[0].click().run()
    summary = app.code[0].value
    app.run()
    assert app.code[0].value == summary
    assert len(calls) == 1
    app.text_area[0].input("A different draft.").run()
    assert app.session_state["checked_text"] == calls[0]
    app.button(key="start_over").click().run()
    assert not app.exception
    assert app.text_area[0].value == ""
    assert not app.code
    assert not app.error
    assert "product" not in app.session_state
    assert "checked_text" not in app.session_state
    assert len(calls) == 1


def test_failed_second_check_never_shows_the_previous_summary(monkeypatch):
    def explain(text, **kwargs):
        if text == "Second message":
            raise analysis.LocalModelError("Private model path")
        return RagResult(
            TextObservation(category="message", signals=[], issues=[]),
            "insufficient_evidence",
        )

    monkeypatch.setattr(guidance, "explain_text", explain)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("First message").run()
    app.button[0].click().run()
    assert app.code
    app.text_area[0].input("Second message").run()
    app.button[0].click().run()
    assert app.error
    assert not app.code
    assert "Private model path" not in app.error[0].value
