"""Source attribution, redaction and separation of spam labels from targets."""

import pytest
from pydantic import ValidationError

from easy_tech_help.dataset import TextExample, load_dataset
from easy_tech_help.sms_source import sanitize


def test_redaction_preserves_fee_and_help_meaning():
    text = sanitize(
        "Help: 0845 2814032 16 after 1st free. Visit www.test.example. Call cost 150p/min. TsCs08714740323"
    )
    assert "[PHONE] 16 after 1st free" in text
    assert "150p/min" in text
    assert "https://source-link.example" in text
    assert "TsCs[PHONE]" in text
    assert "0845" not in text
    assert "08714740323" not in text


def test_real_examples_require_source_and_do_not_relabel_spam_as_fraud():
    dataset = load_dataset()
    public = [r for rows in dataset.values() for r in rows if r.source]
    assert len(public) == 80
    assert sum(r.source.original_label == "ham" for r in public) == 40
    assert sum(r.source.original_label == "spam" for r in public) == 40
    assert all(
        r.case_kind == "unsolicited_promotion"
        for r in public
        if r.source.original_label == "spam"
    )
    # A prize deadline is not a current account/device threat.
    prize = next(r for r in public if r.id == "uci_sms_3469")
    assert prize.expected["signals"] == []
    broken = public[0].model_dump()
    broken["source"] = None
    with pytest.raises(ValidationError, match="source provenance"):
        TextExample.model_validate(broken)


def test_known_campaign_variants_share_a_training_group():
    dataset = load_dataset()
    campaign = [
        r for r in dataset["train"] if r.scenario_group == "uci_prize_draw_template"
    ]
    assert {r.id for r in campaign} == {"uci_sms_2375", "uci_sms_3469"}
    assert not any(
        r.scenario_group == "uci_prize_draw_template"
        for s in ["validation", "test"]
        for r in dataset[s]
    )


def test_adversarial_cases_are_not_ordinary_controls():
    rows = [r for split in load_dataset().values() for r in split]
    for id in [
        "alert_embedded_instruction_en",
        "unknown_role_injection_en",
        "unknown_schema_override_en",
    ]:
        assert next(r for r in rows if r.id == id).case_kind == "adversarial_input"
    assert not any("pretending to be" in r.input_text for r in rows)
