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

from easy_tech_help.rag import RAG_PROMPT, explain_text, reference_sentences
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

# Explicit development expectations, separate from structural citation checks.
RELEVANT_PHRASES = {
    "wifi_no_internet": ("internet",),
    "wifi_off": ("Wi-Fi is on", "turn Wi-Fi on"),
    "popup_support": ("phone number",),
    "message_password": ("password",),
    "public_wifi": ("encryption", "encrypt"),
}


def case_checks(case_id, result, expected_status, expected_documents):
    retrieved = {r.document_id for r in result.retrieved}
    cited = {c.reference.document_id for c in result.citations}
    checks = {
        "status_matches": result.status == expected_status,
        "expected_document_retrieved": bool(retrieved & expected_documents)
        if expected_documents
        else not retrieved,
        "citation_provenance_valid": all(
            c.reference in result.retrieved and c.quote in c.reference.excerpt
            for c in result.citations
        ),
        "expected_document_cited": bool(cited & expected_documents)
        if expected_documents
        else not cited,
        "explanation_is_exact_selected_evidence": result.explanation
        == "\n\n".join(c.quote for c in result.citations)
        and all(c.quote in reference_sentences(c.reference) for c in result.citations),
        "relevant_phrase_present": any(
            p.casefold() in result.explanation.casefold()
            for p in RELEVANT_PHRASES[case_id]
        )
        if case_id in RELEVANT_PHRASES
        else not result.explanation,
    }
    if case_id == "message_password":
        signals = {s.signal for s in result.observation.signals}
        checks["credential_not_payment"] = (
            "credential_request" in signals and "payment_request" not in signals
        )
    if case_id == "missing_context":
        checks["context_classified_unknown"] = (
            result.observation.category == "unknown"
            and result.observation.issues == ["insufficient_context"]
        )
    if case_id == "public_wifi":
        # Require both sides when quoting the corpus's historical contrast.
        # A different complete passage (e.g. HTTPS guidance) need not contain it.
        checks["past_present_context_preserved"] = (
            "In the past," in result.explanation
        ) == ("Today," in result.explanation)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("results/rag_improved.json")
    )
    args = parser.parse_args()
    runtime = LocalRuntime(args.model_dir, args.adapter_dir, args.device)
    results = []
    for case_id, text, expected_status, expected_documents in CASES:
        start = time.perf_counter()
        result = explain_text(text, runtime=runtime)
        checks = case_checks(case_id, result, expected_status, expected_documents)
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
        "scope": "Assistant-authored development checks, not unseen evaluation. Explanations are extractive: exact source sentence identity is checked, not free-form paraphrase entailment. Keyword checks cannot establish semantic relevance or action safety. Classification corrections are application rules, not retraining.",
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
            for name in (
                "rag.py",
                "retrieval.py",
                "runtime.py",
                "rag_evaluation.py",
                "observation_review.py",
            )
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
