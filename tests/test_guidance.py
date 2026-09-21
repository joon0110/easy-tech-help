"""The product entrypoint must enforce policy after RAG, including CLI output."""

import json

import pytest

from easy_tech_help import guidance
from easy_tech_help.analysis import LocalModelError
from easy_tech_help.rag import RagResult
from easy_tech_help.schemas import TextObservation


def test_product_cli_runs_safety_after_a_model_misses_a_sensitive_request(
    monkeypatch, capsys
):
    text = "Send your verification code."
    monkeypatch.setattr(
        guidance,
        "explain_text",
        lambda *a, **k: RagResult(
            TextObservation(category="message", signals=[], issues=[]),
            "insufficient_evidence",
        ),
    )
    monkeypatch.setattr("sys.argv", ["guidance", "--text", text])
    assert guidance.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["guidance"]["level"] == "attention"
    assert "input_guard:verification_code_request" in output["guidance"]["reasons"]
    assert all(a["evidence"] for a in output["guidance"]["next_actions"])


def test_cli_input_file_errors_are_explicit(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "sys.argv", ["guidance", "--file", str(tmp_path / "missing.txt")]
    )
    with pytest.raises(SystemExit) as exc:
        guidance.main()
    assert exc.value.code == 1
    assert "missing.txt" in capsys.readouterr().err


def test_model_failure_never_substitutes_a_model_authored_action(monkeypatch):
    def fail(*a, **k):
        raise LocalModelError("Model unavailable")

    monkeypatch.setattr(guidance, "explain_text", fail)
    with pytest.raises(LocalModelError):
        guidance.guide_text("Help me.")
