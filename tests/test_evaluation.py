"""Meaningful metric denominators, abstentions and ordinary control behavior."""

from easy_tech_help.evaluation import summarize


def test_failed_generation_is_not_a_successful_unknown_prediction():
    row = {
        "case_kind": "unclear",
        "expected": {
            "category": "unknown",
            "signals": [],
            "issues": ["insufficient_context"],
        },
        "predicted": {
            "category": "unknown",
            "signals": [],
            "issues": ["invalid_model_output"],
        },
        "generation": {"seconds": 1.0},
    }
    result = summarize([row])
    assert result["category_accuracy"] == 1.0
    assert result["observation_match_rate"] == 0.0
    assert result["valid_outputs"] == 0
    assert result["signal_precision"] is None
    assert result["ordinary_extra_signal_rate"] is None


def test_ordinary_install_is_not_an_extra_signal_but_hallucinated_threat_is():
    signal = {"signal": "install_request", "evidence": "Install"}
    rows = [
        {
            "case_kind": "ordinary",
            "expected": {"category": "alert", "signals": [signal], "issues": []},
            "predicted": {"category": "alert", "signals": [signal], "issues": []},
            "generation": {"seconds": 1.0},
        },
        {
            "case_kind": "ordinary",
            "expected": {"category": "message", "signals": [], "issues": []},
            "predicted": {
                "category": "message",
                "signals": [{"signal": "urgent_security_warning", "evidence": "Hi"}],
                "issues": [],
            },
            "generation": {"seconds": 1.0},
        },
    ]
    result = summarize(rows)
    assert result["ordinary_extra_signal_rate"] == 0.5
    assert result["signal_precision"] == 0.5
    assert result["signal_recall"] == 1.0
