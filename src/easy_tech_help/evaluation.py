"""Compare the same PyTorch base model and trained adapter on a frozen split."""

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from easy_tech_help.dataset import DEFAULT_DATA_DIR, dataset_report, load_dataset
from easy_tech_help.runtime import (
    ADAPTER_DIR,
    MODEL_DIR,
    LocalRuntime,
    generation_settings,
)


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(cases: list[dict]) -> dict:
    tp = fp = fn = categories = matches = valid_count = controls = false_controls = 0
    for row in cases:
        gold, pred = row["expected"], row["predicted"]
        gold_signals = {s["signal"] for s in gold["signals"]}
        pred_signals = {s["signal"] for s in pred["signals"]}
        valid = not {"invalid_model_output", "incomplete_model_output"}.intersection(
            pred["issues"]
        )
        category_match = gold["category"] == pred["category"]
        tp += len(gold_signals & pred_signals)
        fp += len(pred_signals - gold_signals)
        fn += len(gold_signals - pred_signals)
        categories += category_match
        valid_count += valid
        matches += (
            valid
            and category_match
            and gold_signals == pred_signals
            and set(gold["issues"]) == set(pred["issues"])
        )
        if row["case_kind"] == "ordinary":
            controls += 1
            false_controls += bool(pred_signals - gold_signals)
    return {
        "examples": len(cases),
        "valid_outputs": valid_count,
        "category_correct": categories,
        "observation_matches": matches,
        "category_accuracy": ratio(categories, len(cases)),
        "observation_match_rate": ratio(matches, len(cases)),
        "signal_tp": tp,
        "signal_fp": fp,
        "signal_fn": fn,
        "signal_precision": ratio(tp, tp + fp),
        "signal_recall": ratio(tp, tp + fn),
        "signal_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "ordinary_controls": controls,
        "ordinary_with_extra_signals": false_controls,
        "ordinary_extra_signal_rate": ratio(false_controls, controls),
        "mean_seconds": round(
            sum(r["generation"]["seconds"] for r in cases) / len(cases), 3
        )
        if cases
        else None,
    }


def evaluate(args) -> dict:
    import torch
    from transformers import GenerationConfig

    dataset = load_dataset(args.data_dir)
    report = dataset_report(dataset, args.data_dir)
    manifest = json.loads((args.adapter_dir / "training_manifest.json").read_text())
    if manifest["data_sha256"] != {s: v["sha256"] for s, v in report["splits"].items()}:
        raise ValueError(
            "Dataset changed since training; evaluation must record a new version"
        )
    runs = (
        [("adapter", args.adapter_dir)]
        if args.adapter_only
        else [("base", None), ("adapter", args.adapter_dir)]
    )
    results = {
        "evaluation_completed": False,
        "requested_runs": [name for name, _ in runs],
        "split": args.split,
        "data_sha256": manifest["data_sha256"],
        "prompt_sha256": report["prompt_sha256"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "adapter_sha256": hashlib.sha256(
            (args.adapter_dir / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "settings": {**generation_settings(), "device": args.device},
        "evaluation_role": "development"
        if args.split == "validation"
        else "regression",
        "interpretation": "Previously inspected project split; not an independent final test or scam verdict accuracy. Public texts may occur in pretraining.",
        "runs": {},
    }
    for name, adapter in runs:
        runtime = LocalRuntime(args.model_dir, adapter, args.device)
        cases = []
        for example in dataset[args.split]:
            observation, generation = runtime.analyze(example.input_text)
            case = {
                "id": example.id,
                "input_text": example.input_text,
                "provenance": example.provenance,
                "case_kind": example.case_kind,
                "original_label": example.source.original_label
                if example.source
                else None,
                "expected": example.expected,
                "predicted": observation.training_target(),
                "generation": asdict(generation),
            }
            cases.append(case)
            print(
                json.dumps(
                    {
                        "run": name,
                        "id": example.id,
                        "seconds": round(generation.seconds, 2),
                    }
                ),
                flush=True,
            )
        results["runs"][name] = {
            "effective_generation_config": runtime.model._prepare_generation_config(
                GenerationConfig(
                    **generation_settings(),
                    eos_token_id=runtime.model.generation_config.eos_token_id,
                    pad_token_id=runtime.tokenizer.pad_token_id,
                )
            )[0].to_dict(),
            "metrics": summarize(cases),
            "by_provenance": {
                p: summarize([c for c in cases if c["provenance"] == p])
                for p in sorted({c["provenance"] for c in cases})
            },
            "by_category": {
                p: summarize([c for c in cases if c["expected"]["category"] == p])
                for p in sorted({c["expected"]["category"] for c in cases})
            },
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        del runtime
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    results["evaluation_completed"] = True
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--output", type=Path, default=Path("results/pytorch_validation.json")
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Skip a previously measured base run",
    )
    result = evaluate(parser.parse_args())
    print(json.dumps({k: v["metrics"] for k, v in result["runs"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
