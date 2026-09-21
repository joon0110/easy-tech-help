"""Validate authored text labels and export positive-only chat SFT examples."""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easy_tech_help.analysis import build_messages
from easy_tech_help.retrieval import load_documents
from easy_tech_help.schemas import TextObservation, validate_input

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "text"
SPLITS = ("train", "validation", "test")


class TextExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    scenario_group: str = Field(min_length=1)
    language: Literal["en"]
    provenance: Literal["synthetic_authored"]
    review_status: Literal["draft_needs_human_review", "human_reviewed"]
    case_kind: Literal[
        "suspicious_pattern",
        "ordinary",
        "connection_issue",
        "normal_connection",
        "unclear",
    ]
    input_text: str
    expected: dict
    rejected_output: dict
    rejection_reason: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    reference_ids: list[str]

    @model_validator(mode="after")
    def validate_labels(self) -> "TextExample":
        validate_input(self.input_text)
        target = TextObservation.model_validate(
            self.expected, context={"input_text": self.input_text}
        )
        if target.training_target() != self.expected:
            raise ValueError("Gold labels must not rely on runtime normalization")
        if self.expected == self.rejected_output:
            raise ValueError("Rejected answer must differ from the gold label")
        return self


def load_dataset(directory: Path = DEFAULT_DATA_DIR) -> dict[str, list[TextExample]]:
    """Validate every split, including exact duplicate and scenario leakage checks."""
    result = {}
    ids: set[str] = set()
    texts: set[str] = set()
    group_splits: dict[str, str] = {}
    source_ids = {document.id for document in load_documents()}
    for split in SPLITS:
        examples = []
        for number, line in enumerate(
            (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                example = TextExample.model_validate_json(line)
                normalized = " ".join(example.input_text.casefold().split())
                if example.id in ids or normalized in texts:
                    raise ValueError("Duplicate example ID or input text")
                previous = group_splits.get(example.scenario_group)
                if previous is not None and previous != split:
                    raise ValueError("Scenario group leaked across dataset splits")
                if set(example.reference_ids) - source_ids:
                    raise ValueError("Unknown knowledge reference ID")
                ids.add(example.id)
                texts.add(normalized)
                group_splits[example.scenario_group] = split
                examples.append(example)
            except ValueError as exc:
                raise ValueError(f"{split}.jsonl:{number}: {exc}") from exc
        if not examples:
            raise ValueError(f"{split} must not be empty")
        result[split] = examples
    return result


def export_sft(dataset: dict[str, list[TextExample]], output: Path) -> None:
    """Export only training and validation; keep held-out test and wrong answers out."""
    output.mkdir(parents=True, exist_ok=True)
    for split, filename in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        records = []
        for example in dataset[split]:
            messages = build_messages(example.input_text)
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(example.expected, ensure_ascii=False),
                }
            )
            records.append(json.dumps({"messages": messages}, ensure_ascii=False))
        (output / filename).write_text("\n".join(records) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check text data; optionally export SFT."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--export", type=Path, help="Output directory for train/valid JSONL"
    )
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data_dir)
        if args.export:
            # Never overwrite hand-authored labels with trainer-only chat records.
            if args.export.resolve() == args.data_dir.resolve():
                raise ValueError(
                    "Export to a separate directory, e.g. artifacts/text-sft"
                )
            export_sft(dataset, args.export)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({split: len(rows) for split, rows in dataset.items()}))
    print(
        "Labels checked. Synthetic draft labels still need human review; no training ran."
    )
    if args.export:
        print(
            f"Exported positive-only train/valid messages to {args.export}. Test excluded."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
