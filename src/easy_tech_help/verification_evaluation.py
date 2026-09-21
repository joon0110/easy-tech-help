"""Additional authored verification after the quality fixes; not an independent test."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from easy_tech_help.guidance import guide_text
from easy_tech_help.product_evaluation import (
    challenge_checks,
    contract_checks,
    fingerprint,
)
from easy_tech_help.runtime import ADAPTER_DIR, LocalRuntime
from easy_tech_help.safety import displayable_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("data/evaluation/verification.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/verification.json")
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    runtime = LocalRuntime(device=args.device)
    rows = []
    for case in cases:
        start = time.perf_counter()
        result = guide_text(case["input"], runtime=runtime)
        checks = {
            **contract_checks(result, case["private_markers"]),
            **challenge_checks(case, result),
        }
        rows.append(
            {
                "id": case["id"],
                "input": case["input"],
                "expected": case,
                "checks": checks,
                "seconds": round(time.perf_counter() - start, 3),
                "result": result.to_dict(),
                "displayed_reference": displayable_reference(result.analysis),
            }
        )
        print(
            case["id"], "failed:", [k for k, v in checks.items() if not v], flush=True
        )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Assistant-authored verification inputs frozen before this run. Conceptual overlap with development data remains; no independent real-world accuracy claim.",
        "device": runtime.device,
        "cases_sha256": fingerprint(args.cases),
        "adapter_sha256": fingerprint(ADAPTER_DIR / "adapter_model.safetensors"),
        "code_sha256": {
            str(p): fingerprint(p)
            for p in sorted(Path("src/easy_tech_help").glob("*.py"))
        },
        "summary": {
            "cases": len(rows),
            "all_checks_passed": sum(all(r["checks"].values()) for r in rows),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["summary"])
    return 0 if report["summary"]["all_checks_passed"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
