"""Reproducible RAG development checks, not an independent accuracy benchmark."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from easy_tech_help.rag import RAG_PROMPT, explain_text
from easy_tech_help.retrieval import DEFAULT_KNOWLEDGE_DIR
from easy_tech_help.runtime import ADAPTER_DIR, MODEL_DIR, LocalRuntime

# Assistant-authored development cases. These do not enter SFT or model selection.
CASES = [
    (
        "wifi_no_internet",
        "My iPhone is connected to Wi-Fi but says No Internet Connection.",
        "answered",
        {"wifi_iphone", "wifi_join_iphone"},
    ),
    (
        "wifi_off",
        "In Settings on my iPhone, Wi-Fi is turned off.",
        "answered",
        {"wifi_iphone", "wifi_join_iphone"},
    ),
    (
        "popup_support",
        "A browser popup says: Virus detected! Call our technical support number immediately.",
        "answered",
        {"popup_tech_support", "popup_safari"},
    ),
    (
        "message_password",
        "Text message: Your account will be closed today. Enter your password at https://account-check.example to keep it open.",
        "answered",
        {"message_phishing", "message_spam_text", "message_avoid_scam"},
    ),
    (
        "public_wifi",
        "My iPhone shows an unsecured public Wi-Fi network.",
        "answered",
        {"wifi_public_safety", "wifi_home_security"},
    ),
    (
        "ordinary_message",
        "Text message from a friend: See you at the book club tomorrow.",
        "insufficient_evidence",
        set(),
    ),
    (
        "unrelated_alert",
        "An alert says my astronomy lecture starts at noon.",
        "insufficient_evidence",
        set(),
    ),
    ("missing_context", "Please help me with this.", "uncertain_analysis", set()),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("results/rag_development.json")
    )
    args = parser.parse_args()
    runtime = LocalRuntime(args.model_dir, args.adapter_dir, args.device)
    results = []
    for case_id, text, expected_status, expected_documents in CASES:
        start = time.perf_counter()
        result = explain_text(text, runtime=runtime)
        retrieved = {r.document_id for r in result.retrieved}
        cited = {c.reference.document_id for c in result.citations}
        checks = {
            "status_matches": result.status == expected_status,
            "expected_document_retrieved": bool(retrieved & expected_documents)
            if expected_documents
            else not retrieved,
            "citation_provenance_valid": all(
                c.reference in result.retrieved and c.quote == c.reference.excerpt
                for c in result.citations
            ),
            "expected_document_cited": bool(cited & expected_documents)
            if expected_documents
            else not cited,
        }
        results.append(
            {
                "id": case_id,
                "input": text,
                "expected_status": expected_status,
                "expected_documents": sorted(expected_documents),
                "checks": checks,
                "seconds": round(time.perf_counter() - start, 3),
                "result": result.to_dict(),
            }
        )
        print(f"{case_id}: {result.status}; checks={checks}", flush=True)
    paths = [DEFAULT_KNOWLEDGE_DIR / "catalog.json"]
    catalog = json.loads(paths[0].read_text())
    paths += [DEFAULT_KNOWLEDGE_DIR / doc["file"] for doc in catalog["documents"]]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Assistant-authored development checks; prompt/debugging inputs, not unseen evaluation. Provenance checks do not measure factual entailment or action safety.",
        "device": runtime.device,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "python": platform.python_version(),
        "model_directory": str(args.model_dir),
        "adapter_directory": str(args.adapter_dir),
        "adapter_sha256": hashlib.sha256(
            (args.adapter_dir / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "rag_prompt_sha256": hashlib.sha256(RAG_PROMPT.encode()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(DEFAULT_KNOWLEDGE_DIR)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in paths
        },
        "code_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("rag.py", "retrieval.py", "runtime.py", "rag_evaluation.py")
        },
        "generation": {
            "analysis_adapter_enabled": True,
            "explanation_adapter_enabled": False,
            "sampling": False,
            "analysis_repetition_penalty": 1.0,
            "explanation_repetition_penalty": 1.1,
        },
        "summary": {
            "cases": len(results),
            "statuses": dict(Counter(r["result"]["status"] for r in results)),
            "all_checks_passed": sum(all(r["checks"].values()) for r in results),
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")
    return 0 if report["summary"]["all_checks_passed"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
