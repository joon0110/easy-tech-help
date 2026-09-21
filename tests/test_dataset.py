"""Guard labels, split leakage and positive-only training export."""

import json
import shutil
from typing import get_args

import pytest
from pydantic import ValidationError

from easy_tech_help.analysis import build_messages
from easy_tech_help.dataset import (
    DEFAULT_DATA_DIR,
    TextExample,
    dataset_report,
    export_sft,
    load_dataset,
    similar_cross_split_pairs,
    token_lengths,
)
from easy_tech_help.schemas import Signal


def test_dataset_labels_and_split_coverage():
    dataset = load_dataset()
    assert {key: len(rows) for key, rows in dataset.items()} == {
        "train": 132,
        "validation": 34,
        "test": 34,
    }
    for rows in dataset.values():
        assert {row.language for row in rows} == {"en"}
        assert {row.review_status for row in rows} == {"automated_reviewed"}
        assert {row.provenance for row in rows} == {
            "synthetic_authored",
            "uci_sms_derived",
        }
        assert {row.expected["category"] for row in rows} == {
            "message",
            "alert",
            "wifi",
            "unknown",
        }
        assert {row.case_kind for row in rows} >= {"suspicious_pattern", "ordinary"}
        assert {s["signal"] for row in rows for s in row.expected["signals"]} == set(
            get_args(Signal)
        )
        assert {i for row in rows for i in row.expected["issues"]} == {
            "unsupported",
            "insufficient_context",
            "contradictory_input",
        }


def test_export_excludes_wrong_answers_metadata_and_test_inputs(tmp_path):
    dataset = load_dataset()
    export_sft(dataset, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {"train.jsonl", "valid.jsonl"}
    for split, filename in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        records = [
            json.loads(line) for line in (tmp_path / filename).read_text().splitlines()
        ]
        for record, example in zip(records, dataset[split], strict=True):
            assert set(record) == {"messages"}
            assert record["messages"][:2] == build_messages(example.input_text)
            assert json.loads(record["messages"][2]["content"]) == example.expected
            assert "rejected_output" not in record
            assert "source" not in record
            assert "original_label" not in record["messages"][0]["content"]
        exported_inputs = {
            json.loads(r["messages"][1]["content"])["input_text"] for r in records
        }
        assert not exported_inputs.intersection(
            row.input_text for row in dataset["test"]
        )


def test_labels_cannot_silently_normalize_into_different_gold():
    example = load_dataset()["train"][0].model_dump()
    example["expected"]["issues"] = ["unsupported"]
    with pytest.raises(ValidationError, match="runtime normalization"):
        TextExample.model_validate(example)


def test_fabricated_gold_evidence_is_rejected():
    example = load_dataset()["train"][0].model_dump()
    example["expected"]["signals"][0]["evidence"] = "not in the input"
    with pytest.raises(ValidationError, match="actual input"):
        TextExample.model_validate(example)


def test_fabricated_rejected_evidence_is_rejected():
    example = next(
        row.model_dump()
        for row in load_dataset()["train"]
        if row.rejected_output["signals"]
    )
    example["rejected_output"]["signals"][0]["evidence"] = "not in the input"
    with pytest.raises(ValidationError, match="actual input"):
        TextExample.model_validate(example)


def test_runtime_failure_cannot_be_a_gold_label():
    example = load_dataset()["train"][0].model_dump()
    example["expected"] = {
        "category": "unknown",
        "signals": [],
        "issues": ["invalid_model_output"],
    }
    with pytest.raises(ValidationError, match="runtime-only"):
        TextExample.model_validate(example)


@pytest.mark.parametrize("leak", ["scenario", "text", "id", "near"])
def test_cross_split_leakage_rejected(tmp_path, leak):
    shutil.copytree(DEFAULT_DATA_DIR, tmp_path / "data")
    path = tmp_path / "data" / "validation.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    train = load_dataset()["train"][0]
    if leak == "scenario":
        records[0]["scenario_group"] = train.scenario_group
    elif leak == "id":
        records[0]["id"] = train.id
    elif leak == "near":
        records[0] = train.model_dump() | {
            "id": "near_copy_en",
            "scenario_group": "near_copy",
            "input_text": train.input_text + " Please.",
        }
    else:
        records[0]["input_text"] = train.input_text
        records[0]["expected"] = train.expected
        records[0]["rejected_output"] = train.rejected_output
        records[0]["case_kind"] = train.case_kind
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match="leaked|Duplicate|Near-duplicate"):
        load_dataset(tmp_path / "data")


def test_held_out_input_in_system_prompt_is_rejected(monkeypatch):
    from easy_tech_help import dataset as module

    text = load_dataset()["validation"][0].input_text
    monkeypatch.setattr(module, "build_messages", lambda _: [{"content": text}])
    with pytest.raises(ValueError, match="system prompt"):
        load_dataset()


def test_template_reuse_with_changed_url_and_amount_is_detected():
    original = load_dataset()["train"][0].model_copy(
        update={
            "input_text": "SMS: Pay $15 to restore delivery at https://one.example."
        }
    )
    copy = original.model_copy(
        update={
            "id": "copy",
            "scenario_group": "different_id_does_not_prevent_leakage",
            "input_text": "sms: pay $99 to restore delivery at https://two.example!",
        }
    )
    matches = similar_cross_split_pairs({"train": [original], "test": [copy]})
    assert len(matches) == 1
    assert matches[0]["similarity"] == 1.0


class FakeTokenizer:
    """Small known token sequences, so truncation and loss boundaries are testable."""

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"]
        assert "truncation" not in kwargs
        return {"input_ids": [1, 2, 3, 4, 5] if len(messages) == 3 else [1, 2, 3]}


def test_token_lengths_include_prompt_and_completion_without_truncating():
    one = {"train": load_dataset()["train"][:1]}
    report = token_lengths(one, FakeTokenizer(), max_tokens=5)
    assert report["train"]["max"] == 5
    assert report["train"]["max_completion_tokens"] == 2
    with pytest.raises(ValueError, match="Token limit exceeded"):
        token_lengths(one, FakeTokenizer(), max_tokens=4)
    with pytest.raises(ValueError, match="Token limit exceeded"):
        token_lengths(one, FakeTokenizer(), max_completion_tokens=1)


def test_invalid_completion_boundary_is_rejected():
    class BrokenTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            tokens = super().apply_chat_template(messages, **kwargs)
            if len(messages) == 3:
                tokens["input_ids"][0] = 99
            return tokens

    with pytest.raises(ValueError, match="completion boundary"):
        token_lengths({"train": load_dataset()["train"][:1]}, BrokenTokenizer())


def test_checked_in_preparation_report_matches_data_and_prompt():
    saved = json.loads((DEFAULT_DATA_DIR / "preparation_report.json").read_text())
    actual = dataset_report(load_dataset(), DEFAULT_DATA_DIR)
    for key, value in actual.items():
        assert saved[key] == value, f"Stale preparation report field: {key}"
