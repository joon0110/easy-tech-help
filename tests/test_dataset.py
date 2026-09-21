"""Guard labels, split leakage and positive-only training export."""

import json
import shutil

import pytest
from pydantic import ValidationError

from easy_tech_help.analysis import build_messages
from easy_tech_help.dataset import (
    DEFAULT_DATA_DIR,
    TextExample,
    export_sft,
    load_dataset,
)


def test_dataset_labels_and_split_coverage():
    dataset = load_dataset()
    assert {key: len(rows) for key, rows in dataset.items()} == {
        "train": 19,
        "validation": 7,
        "test": 7,
    }
    for rows in dataset.values():
        assert {row.language for row in rows} == {"en"}
        assert {row.review_status for row in rows} == {"automated_reviewed"}
        assert {row.expected["category"] for row in rows} == {
            "message",
            "alert",
            "wifi",
            "unknown",
        }
        assert {row.case_kind for row in rows} >= {"suspicious_pattern", "ordinary"}


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
    example = load_dataset()["train"][0].model_dump()
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


@pytest.mark.parametrize("leak", ["scenario", "text", "id"])
def test_cross_split_leakage_rejected(tmp_path, leak):
    shutil.copytree(DEFAULT_DATA_DIR, tmp_path / "data")
    path = tmp_path / "data" / "validation.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    train = load_dataset()["train"][0]
    if leak == "scenario":
        records[0]["scenario_group"] = train.scenario_group
    elif leak == "id":
        records[0]["id"] = train.id
    else:
        records[0]["input_text"] = train.input_text
        records[0]["expected"] = train.expected
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match="leaked|Duplicate"):
        load_dataset(tmp_path / "data")
