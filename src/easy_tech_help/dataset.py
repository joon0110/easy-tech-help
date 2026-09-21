"""Validate authored text labels and export positive-only chat SFT examples."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easy_tech_help.analysis import build_messages
from easy_tech_help.retrieval import load_documents
from easy_tech_help.schemas import Signal, TextObservation, validate_input

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "text"
SPLITS = ("train", "validation", "test")
TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TOKENIZER_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MAX_SEQUENCE_TOKENS = 1536
NEAR_DUPLICATE_THRESHOLD = 0.85


class SmsSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["uci_sms_spam_collection"]
    record_number: int = Field(ge=1, le=5574)
    original_label: Literal["ham", "spam"]
    raw_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sanitization_version: Literal[1]


class TextExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    scenario_group: str = Field(min_length=1)
    language: Literal["en"]
    provenance: Literal["synthetic_authored", "uci_sms_derived"]
    source: SmsSource | None = None
    review_status: Literal[
        "draft_needs_human_review", "automated_reviewed", "human_reviewed"
    ]
    case_kind: Literal[
        "suspicious_pattern",
        "ordinary",
        "connection_issue",
        "normal_connection",
        "unclear",
        "adversarial_input",
        "unsolicited_promotion",
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
        rejected = TextObservation.model_validate(
            self.rejected_output, context={"input_text": self.input_text}
        )
        if target.training_target() != self.expected:
            raise ValueError("Gold labels must not rely on runtime normalization")
        if rejected.training_target() != self.rejected_output:
            raise ValueError("Rejected labels must not rely on runtime normalization")
        reserved_issues = {"invalid_model_output", "incomplete_model_output"}
        if reserved_issues.intersection(self.expected["issues"]):
            raise ValueError("Gold labels cannot contain runtime-only issues")
        if target.training_target() == rejected.training_target():
            raise ValueError("Rejected answer must differ from the gold label")
        if self.provenance == "uci_sms_derived" and self.source is None:
            raise ValueError("UCI-derived examples require source provenance")
        if self.provenance == "synthetic_authored" and self.source is not None:
            raise ValueError("Synthetic examples cannot claim a collected source")
        if self.case_kind == "unclear" and target.category != "unknown":
            raise ValueError("Unclear cases must use unknown")
        if target.category == "unknown" and self.case_kind not in {
            "unclear",
            "adversarial_input",
        }:
            raise ValueError("Unknown cases require unclear or adversarial metadata")
        if (
            self.case_kind in {"connection_issue", "normal_connection"}
            and target.category != "wifi"
        ):
            raise ValueError("Connection cases require a Wi-Fi category")
        if len(self.expected["issues"]) != len(set(self.expected["issues"])):
            raise ValueError("Gold issues must be unique")
        return self


def _comparison_tokens(text: str) -> list[str]:
    """Catch punctuation, URL and number substitutions in reused templates."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"https?://\S+", "url", text)
    text = re.sub(r"\b\d+(?:[.\-]\d+)*\b", "number", text)
    return re.findall(r"\w+", text)


def similar_cross_split_pairs(dataset: dict[str, list[TextExample]]) -> list[dict]:
    """Flag lexical reuse; this heuristic cannot prove semantic independence."""
    entries = [
        (split, row, _comparison_tokens(row.input_text))
        for split, rows in dataset.items()
        for row in rows
    ]
    matches = []
    for (left_split, left, a), (right_split, right, b) in combinations(entries, 2):
        if left_split == right_split:
            continue
        # SequenceMatcher can be asymmetric: use the larger ratio.
        score = max(
            SequenceMatcher(None, a, b, autojunk=False).ratio(),
            SequenceMatcher(None, b, a, autojunk=False).ratio(),
        )
        threshold = 0.75 if left.source and right.source else NEAR_DUPLICATE_THRESHOLD
        if score >= threshold:
            matches.append(
                {
                    "left": left.id,
                    "left_split": left_split,
                    "right": right.id,
                    "right_split": right_split,
                    "similarity": round(score, 4),
                }
            )
    return matches


def load_dataset(directory: Path = DEFAULT_DATA_DIR) -> dict[str, list[TextExample]]:
    """Validate labels and reject duplicate IDs, inputs, groups and near copies."""
    result = {}
    ids: set[str] = set()
    texts: set[str] = set()
    group_splits: dict[str, str] = {}
    source_records: set[int] = set()
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
                if example.source:
                    if example.source.record_number in source_records:
                        raise ValueError("Duplicate source record")
                    source_records.add(example.source.record_number)
                previous = group_splits.get(example.scenario_group)
                if previous is not None and previous != split:
                    raise ValueError("Scenario group leaked across dataset splits")
                if set(example.reference_ids) - source_ids:
                    raise ValueError("Unknown knowledge reference ID")
                if (
                    split != "train"
                    and example.input_text in build_messages("audit")[0]["content"]
                ):
                    raise ValueError(
                        "Held-out input is already present in the system prompt"
                    )
                ids.add(example.id)
                texts.add(normalized)
                group_splits[example.scenario_group] = split
                examples.append(example)
            except ValueError as exc:
                raise ValueError(f"{split}.jsonl:{number}: {exc}") from exc
        if not examples:
            raise ValueError(f"{split} must not be empty")
        result[split] = examples
    near_duplicates = similar_cross_split_pairs(result)
    if near_duplicates:
        raise ValueError(f"Near-duplicate inputs across splits: {near_duplicates}")
    return result


def sft_messages(example: TextExample) -> list[dict[str, str]]:
    """Use one serialization for export, length checking and future training."""
    return build_messages(example.input_text) + [
        {
            "role": "assistant",
            "content": json.dumps(example.expected, ensure_ascii=False),
        }
    ]


def token_lengths(
    dataset: dict[str, list[TextExample]],
    tokenizer,
    *,
    max_tokens: int = MAX_SEQUENCE_TOKENS,
    max_completion_tokens: int = 256,
) -> dict:
    """Measure full chat sequences without truncation or model inference."""
    if max_tokens < 1 or max_completion_tokens < 1:
        raise ValueError("Token limits must be positive")
    stats = {}
    for split, rows in dataset.items():
        lengths = []
        completions = []
        for row in rows:
            messages = sft_messages(row)
            full = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True
            )["input_ids"]
            prefix = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
            if full[: len(prefix)] != prefix or len(full) <= len(prefix):
                raise ValueError(f"Assistant completion boundary is invalid: {row.id}")
            completion = len(full) - len(prefix)
            if len(full) > max_tokens or completion > max_completion_tokens:
                raise ValueError(
                    f"Token limit exceeded: {row.id}: {len(full)} total, "
                    f"{completion} completion tokens; no truncation allowed"
                )
            lengths.append(len(full))
            completions.append(completion)
        stats[split] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
            "max_completion_tokens": max(completions),
        }
    return stats


def dataset_report(dataset: dict[str, list[TextExample]], directory: Path) -> dict:
    """Record coverage and fingerprints; structural checks are not accuracy."""
    splits = {}
    for split, rows in dataset.items():
        signal_counts = Counter(
            signal["signal"] for row in rows for signal in row.expected["signals"]
        )
        splits[split] = {
            "examples": len(rows),
            "scenario_groups": len({row.scenario_group for row in rows}),
            "categories": dict(
                sorted(Counter(r.expected["category"] for r in rows).items())
            ),
            "case_kinds": dict(sorted(Counter(r.case_kind for r in rows).items())),
            "provenance": dict(sorted(Counter(r.provenance for r in rows).items())),
            "source_labels": dict(
                sorted(
                    Counter(r.source.original_label for r in rows if r.source).items()
                )
            ),
            "signals": dict(sorted(signal_counts.items())),
            "missing_signals": sorted(set(get_args(Signal)) - signal_counts.keys()),
            "issues": dict(
                sorted(Counter(i for r in rows for i in r.expected["issues"]).items())
            ),
            "no_signal_examples": sum(not r.expected["signals"] for r in rows),
            "sha256": hashlib.sha256(
                (directory / f"{split}.jsonl").read_bytes()
            ).hexdigest(),
        }
    return {
        "report_version": 1,
        "review": "automated; authored synthetic and redacted public SMS; human review still needed",
        "model_training_run": False,
        "model_accuracy_evaluated": False,
        "splits": splits,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "public_sms_near_duplicate_threshold": 0.75,
        "cross_split_near_duplicates": similar_cross_split_pairs(dataset),
        "similarity_method": "max bidirectional word SequenceMatcher; NFKC, case, URL and number normalization",
        "semantic_independence_proven": False,
        "prompt_sha256": hashlib.sha256(
            build_messages("audit")[0]["content"].encode()
        ).hexdigest(),
    }


def export_sft(dataset: dict[str, list[TextExample]], output: Path) -> None:
    """Export only training and validation; keep held-out test and wrong answers out."""
    output.mkdir(parents=True, exist_ok=True)
    for split, filename in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        records = []
        for example in dataset[split]:
            messages = sft_messages(example)
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
    parser.add_argument("--report", type=Path, help="Write coverage and integrity JSON")
    parser.add_argument(
        "--check-tokens",
        action="store_true",
        help="Use the pinned, locally cached tokenizer",
    )
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data_dir)
        report = dataset_report(dataset, args.data_dir)
        if args.check_tokens:
            from importlib.metadata import version

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True
            )
            report["tokenizer"] = {
                "id": TOKENIZER_ID,
                "revision": TOKENIZER_REVISION,
                "transformers_version": version("transformers"),
                "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
                "max_completion_tokens": 256,
            }
            report["token_lengths"] = token_lengths(dataset, tokenizer)
        if args.report and args.report.resolve() in {
            (args.data_dir / f"{split}.jsonl").resolve() for split in SPLITS
        }:
            raise ValueError("Report must not overwrite dataset labels")
        if args.export:
            # Never overwrite hand-authored labels with trainer-only chat records.
            if args.export.resolve() == args.data_dir.resolve():
                raise ValueError(
                    "Export to a separate directory, e.g. artifacts/text-sft"
                )
            export_sft(dataset, args.export)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({split: len(rows) for split, rows in dataset.items()}))
    print(
        "Labels passed automated checks but still need human review; no training ran."
    )
    if args.export:
        print(
            f"Exported positive-only train/valid messages to {args.export}. Test excluded."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
