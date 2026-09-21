"""The audit must expose semantic failures even when output contracts pass."""

import json
from dataclasses import replace
from types import SimpleNamespace

from easy_tech_help import product_evaluation as audit
from easy_tech_help.guidance import GuidedResult
from easy_tech_help.handoff import build_handoff
from easy_tech_help.rag import RagResult
from easy_tech_help.safety import SafeAction, build_guidance
from easy_tech_help.schemas import TextObservation


def ordinary_result():
    result = RagResult(
        TextObservation(category="message", signals=[], issues=[]),
        "insufficient_evidence",
    )
    guidance = build_guidance("See you tomorrow.", result)
    return GuidedResult(result, guidance, build_handoff("message", guidance))


def test_safe_templates_do_not_count_as_correct_detection():
    result = ordinary_result()
    case = json.loads(audit.CHALLENGE.read_text().splitlines()[0])
    assert all(audit.contract_checks(result).values())
    checks = audit.challenge_checks(case, result)
    assert not checks["level"]
    assert not checks["required_flags"]
    assert not checks["appropriate_action"]


def test_invalid_actions_are_recorded_as_failure_instead_of_crashing():
    result = ordinary_result()
    result.guidance = replace(
        result.guidance, next_actions=(SafeAction("share_password", "Send a password"),)
    )
    checks = audit.contract_checks(result)
    assert not checks["approved_actions_only"]
    assert not checks["no_prohibited_actions"]
    assert not checks["action_evidence_verified"]


def test_complete_runner_records_all_denominators_and_returns_failure(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(audit, "LocalRuntime", lambda *a: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(audit, "guide_text", lambda *a, **k: ordinary_result())
    path = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["evaluation", "--output", str(path)])
    assert audit.main() == 1
    report = json.loads(path.read_text())
    assert len(report["cases"]) == 70
    assert report["summary"]["regression"]["cases"] == 34
    assert report["summary"]["challenge"]["cases"] == 24
    assert report["summary"]["development"]["cases"] == 12
    assert report["summary"]["challenge"]["quality_passed"] < 24
