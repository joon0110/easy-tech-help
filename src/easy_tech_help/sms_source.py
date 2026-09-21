"""Verify the downloaded UCI SMS source and reproduce selected text redactions.

The source spam/ham label is metadata, never an EasyTechHelp analysis target.
No SMS links are opened. The unmodified ZIP stays in ignored artifacts/sources.
"""

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ARCHIVE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
DEFAULT_ARCHIVE = Path("artifacts/sources/uci_sms_spam_collection.zip")


def read_source(archive: Path = DEFAULT_ARCHIVE) -> list[tuple[str, str]]:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("UCI archive checksum differs from the reviewed snapshot")
    with ZipFile(archive) as zipped:
        lines = zipped.read("SMSSpamCollection").decode("utf-8").splitlines()
    rows = [tuple(line.split("\t", 1)) for line in lines]
    if len(rows) != 5574 or Counter(r[0] for r in rows) != {"ham": 4827, "spam": 747}:
        raise ValueError("Unexpected source counts or labels")
    return rows


def sanitize(text: str) -> str:
    """Version 1 redactions for the curated subset, not a universal PII detector."""
    text = html.unescape(text).replace("<fone no>", "[PHONE]")
    text = re.sub(r"https?://[^\s]+|www\.[^\s]+", "https://source-link.example", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"(?<!\d)0(?:[ ()-]*\d){10}", "[PHONE]", text)
    text = re.sub(r"\b0\d+x+\b", "[PHONE]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\d)\d{5,}(?!\d)", "[NUMBER]", text)
    text = re.sub(r"(?i)\b(?:po\s*box|pobox|box)\s*\d+[\w/]*", "[POSTBOX]", text)
    text = re.sub(r"(?i)\b[a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2}\b", "[POSTCODE]", text)
    return " ".join(text.split())


def verify_selected(data_dir: Path, archive: Path = DEFAULT_ARCHIVE) -> dict:
    source_rows = read_source(archive)
    counts = Counter()
    seen = set()
    for split in ("train", "validation", "test"):
        for line in (data_dir / f"{split}.jsonl").read_text().splitlines():
            example = json.loads(line)
            if example["provenance"] != "uci_sms_derived":
                continue
            source = example["source"]
            number = source["record_number"]
            if not 1 <= number <= len(source_rows) or number in seen:
                raise ValueError("Invalid or repeated UCI source record")
            seen.add(number)
            label, raw = source_rows[number - 1]
            if (
                source["original_label"] != label
                or source["raw_text_sha256"] != hashlib.sha256(raw.encode()).hexdigest()
                or example["input_text"] != sanitize(raw)
            ):
                raise ValueError(f"Source or redaction mismatch: {example['id']}")
            counts[f"{split}_{label}"] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--data-dir", type=Path, default=Path("data/text"))
    args = parser.parse_args()
    print(json.dumps(verify_selected(args.data_dir, args.archive), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
