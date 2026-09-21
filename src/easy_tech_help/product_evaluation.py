"""Audit the current full product on regression and authored challenge inputs."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from easy_tech_help.dataset import dataset_report, load_dataset
from easy_tech_help.evaluation import summarize
from easy_tech_help.guidance import guide_text
from easy_tech_help.rag import reference_sentences
from easy_tech_help.rag_evaluation import CASES as RAG_CASES
from easy_tech_help.rag_evaluation import case_checks
from easy_tech_help.retrieval import DEFAULT_KNOWLEDGE_DIR, load_documents
from easy_tech_help.runtime import ADAPTER_DIR, MODEL_DIR, LocalRuntime
from easy_tech_help.safety import (
    ACTIONS,
    PROHIBITED_ACTIONS,
    SENSITIVE,
    displayable_reference,
    resolve_actions,
)
from easy_tech_help.safety_evaluation import CASES, EXPECTED

CHALLENGE = Path("data/evaluation/product_challenge.jsonl")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_checks(result, private_markers=()):
    """Output constraints, separately from semantic detection and relevance."""
    analysis, guidance, handoff = (
        result.analysis,
        result.guidance,
        result.family_handoff,
    )
    ids = [a.id for a in guidance.next_actions]
    try:
        evidence_verified = guidance.next_actions == resolve_actions(ids)
    except (OSError, ValueError, KeyError, TypeError):
        evidence_verified = False
    return {
        "approved_actions_only": all(
            a.id in ACTIONS and a.text == ACTIONS[a.id].text
            for a in guidance.next_actions
        ),
        "no_prohibited_actions": not set(ids).intersection(PROHIBITED_ACTIONS),
        "action_evidence_verified": evidence_verified,
        "citation_provenance": all(
            c.reference in analysis.retrieved and c.quote in c.reference.excerpt
            for c in analysis.citations
        ),
        "literal_explanation": analysis.explanation
        == "\n\n".join(c.quote for c in analysis.citations)
        and all(
            c.quote in reference_sentences(c.reference) for c in analysis.citations
        ),
        "handoff_preserves_actions": handoff.status == "ready"
        and handoff.action_ids == tuple(ids)
        and all(a.text in handoff.text for a in guidance.next_actions),
        "handoff_no_completion_claim": "Actions already taken: not recorded. Resolution: not confirmed."
        in handoff.text,
        "private_markers_excluded": all(
            marker not in json.dumps(handoff.to_dict()) for marker in private_markers
        ),
    }


def challenge_checks(case, result):
    return {
        "category": result.analysis.observation.category == case["category"],
        "level": result.guidance.level in case["levels"],
        "required_flags": set(case["required_flags"]) <= set(result.guidance.flags),
        "appropriate_action": bool(
            set(case["actions_any"]) & {a.id for a in result.guidance.next_actions}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("results/full_product_evaluation.json")
    )
    args = parser.parse_args()
    dataset = load_dataset()
    data = dataset_report(dataset, Path("data/text"))
    manifest = json.loads((ADAPTER_DIR / "training_manifest.json").read_text())
    if manifest["data_sha256"] != {s: v["sha256"] for s, v in data["splits"].items()}:
        raise ValueError("Dataset differs from the training manifest")
    challenge = [json.loads(line) for line in CHALLENGE.read_text().splitlines()]
    known_inputs = {
        e.input_text.strip().casefold() for split in dataset.values() for e in split
    }
    if any(c["input"].strip().casefold() in known_inputs for c in challenge):
        raise ValueError("Challenge duplicates an existing dataset input")
    runtime = LocalRuntime(MODEL_DIR, ADAPTER_DIR, args.device)
    rag_expected = {row[0]: row for row in RAG_CASES}
    rows, raw_cases, reviewed_cases = [], [], []
    jobs = [("regression", e.id, e.input_text, e) for e in dataset["test"]]
    jobs += [("challenge", c["id"], c["input"], c) for c in challenge]
    jobs += [("development", name, text, None) for name, text in CASES]
    for suite, case_id, text, gold in jobs:
        start = time.perf_counter()
        result = guide_text(text, runtime=runtime)
        checks = contract_checks(
            result, gold["private_markers"] if suite == "challenge" else ()
        )
        quality = {}
        if suite == "regression":
            raw = result.analysis.raw_observation or result.analysis.observation
            base = {
                "id": gold.id,
                "expected": gold.expected,
                "case_kind": gold.case_kind,
                "provenance": gold.provenance,
                "generation": {"seconds": time.perf_counter() - start},
            }
            raw_cases.append({**base, "predicted": raw.training_target()})
            reviewed_cases.append(
                {**base, "predicted": result.analysis.observation.training_target()}
            )
            sensitive = {s["signal"] for s in gold.expected["signals"]} & SENSITIVE
            if sensitive:
                quality["sensitive_level"] = result.guidance.level == "attention"
                quality["sensitive_flags"] = sensitive <= set(result.guidance.flags)
        elif suite == "challenge":
            quality = challenge_checks(gold, result)
        else:
            levels, actions = EXPECTED[case_id]
            quality = {
                "level": result.guidance.level in levels,
                "appropriate_action": bool(
                    actions & {a.id for a in result.guidance.next_actions}
                ),
            }
        row = {
            "suite": suite,
            "id": case_id,
            "input": text,
            "seconds": round(time.perf_counter() - start, 3),
            "contracts": checks,
            "quality": quality,
            "result": result.to_dict(),
            "displayed_reference": displayable_reference(result.analysis),
        }
        if suite == "challenge":
            row["expected"] = gold
            if gold["documents_any"]:
                row["expected_document_retrieved"] = bool(
                    set(gold["documents_any"])
                    & {r.document_id for r in result.analysis.retrieved}
                )
                row["expected_document_cited"] = bool(
                    set(gold["documents_any"])
                    & {c.reference.document_id for c in result.analysis.citations}
                )
        if suite == "regression":
            row["expected"] = gold.expected
            row["case_kind"] = gold.case_kind
            row["provenance"] = gold.provenance
        if suite == "development" and case_id in rag_expected:
            _, _, status, docs = rag_expected[case_id]
            row["rag_development_checks"] = case_checks(
                case_id, result.analysis, status, docs
            )
        rows.append(row)
        print(
            f"{len(rows)}/{len(jobs)} {suite}/{case_id}: {result.guidance.level}; failed={[k for k, v in quality.items() if not v]}",
            flush=True,
        )
    metrics = {}
    for suite in ("regression", "challenge", "development"):
        selected = [r for r in rows if r["suite"] == suite]
        latencies = sorted(r["seconds"] for r in selected)
        metrics[suite] = {
            "cases": len(selected),
            "contracts_passed": sum(all(r["contracts"].values()) for r in selected),
            "rag_statuses": dict(
                Counter(r["result"]["analysis"]["status"] for r in selected)
            ),
            "mean_seconds": round(statistics.mean(latencies), 3),
            "p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
        }
        if suite != "regression":
            metrics[suite]["quality_passed"] = sum(
                all(r["quality"].values()) for r in selected
            )
            metrics[suite]["quality_by_check"] = {
                k: sum(r["quality"][k] for r in selected)
                for k in selected[0]["quality"]
            }
    metrics["raw_regression"] = summarize(raw_cases)
    metrics["reviewed_regression"] = summarize(reviewed_cases)
    metrics["raw_regression_by_provenance"] = {
        p: summarize([c for c in raw_cases if c["provenance"] == p])
        for p in sorted({c["provenance"] for c in raw_cases})
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "34 previously inspected regression examples, 24 assistant-authored challenge inputs frozen before this run, 12 known development cases. No retraining or output-driven label edits. No independent real-world accuracy claim; contract checks do not establish relevance or successful resolution.",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": runtime.device,
        "versions": {
            n: version(n) for n in ("torch", "transformers", "peft", "streamlit")
        },
        "adapter_sha256": fingerprint(ADAPTER_DIR / "adapter_model.safetensors"),
        "challenge_sha256": fingerprint(CHALLENGE),
        "data_sha256": manifest["data_sha256"],
        "code_sha256": {
            str(p): fingerprint(p)
            for p in sorted(Path("src/easy_tech_help").glob("*.py"))
        },
        "source_sha256": {
            str(p): fingerprint(p)
            for p in [
                DEFAULT_KNOWLEDGE_DIR / "catalog.json",
                *[
                    DEFAULT_KNOWLEDGE_DIR / d["file"]
                    for d in json.loads(
                        (DEFAULT_KNOWLEDGE_DIR / "catalog.json").read_text()
                    )["documents"]
                ],
            ]
        },
        "source_documents": len(load_documents()),
        "summary": metrics,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return (
        0
        if all(
            all(r["contracts"].values()) and all(r["quality"].values()) for r in rows
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
