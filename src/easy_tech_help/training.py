"""Reproducible PyTorch LoRA with assistant-only loss and validation selection."""

import argparse
import hashlib
import json
import math
import random
import re
import time
from importlib.metadata import version
from pathlib import Path

from easy_tech_help.dataset import (
    DEFAULT_DATA_DIR,
    MAX_SEQUENCE_TOKENS,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    dataset_report,
    load_dataset,
    sft_messages,
    token_lengths,
)
from easy_tech_help.runtime import (
    MODEL_DIR,
    LocalRuntime,
    generation_settings,
    load_base,
    select_device,
)


def encode_example(row, tokenizer, decision_weight: float = 1.0) -> dict:
    messages = sft_messages(row)
    full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)[
        "input_ids"
    ]
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    if full[: len(prefix)] != prefix or not 0 < len(prefix) < len(full):
        raise ValueError(f"Invalid completion boundary: {row.id}")
    if len(full) > MAX_SEQUENCE_TOKENS:
        raise ValueError(f"Training row exceeds context limit: {row.id}")
    encoded = {"ids": full, "start": len(prefix), "id": row.id}
    if decision_weight != 1.0:
        serialized = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(
            serialized, add_special_tokens=False, return_offsets_mapping=True
        )
        if tokenized["input_ids"] != full:
            raise ValueError("Token offsets differ from training serialization")
        content = messages[-1]["content"]
        offset = serialized.rindex(content)
        spans = decision_spans(content)
        encoded["weights"] = [
            decision_weight
            if any(a < offset + end and b > offset + start for start, end in spans)
            else 1.0
            for a, b in tokenized["offset_mapping"][len(prefix) :]
        ]
    return encoded


def decision_spans(content: str) -> list[tuple[int, int]]:
    """Emphasize category, signal/issue names and signal-list decisions."""
    spans = []
    for match in re.finditer(r'"(?:category|signal)":\s*("[^"\\]*")', content):
        spans.append(match.span(1))
    spans.extend(m.span(1) for m in re.finditer(r'"issues":\s*(\[[^\]]*\])', content))
    spans.extend(m.span(1) for m in re.finditer(r'"signals":\s*(\[)', content))
    spans.extend(m.span(1) for m in re.finditer(r'(\]), "issues"', content))
    # The separator is the decision to produce another signal, not stop the list.
    spans.extend(m.span() for m in re.finditer(r'\}, \{"signal"', content))
    return spans


def completion_loss(model, example: dict, device: str):
    """Predict only completion positions; avoid allocating full prompt logits."""
    import torch
    from torch.nn import functional as F

    ids = torch.tensor([example["ids"]], device=device)
    start = example["start"]
    positions = torch.arange(start - 1, ids.shape[1] - 1, device=device)
    logits = model(input_ids=ids, logits_to_keep=positions, use_cache=False).logits
    targets = ids[:, start:]
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    weights = torch.tensor(
        example.get("weights", [1.0] * targets.numel()), device=device
    )
    return (losses * weights).sum() / weights.sum(), float(weights.sum().item())


def generation_validation(model, tokenizer, rows, device) -> dict:
    """Use inference decoding for checkpoint selection; no test examples."""
    from easy_tech_help.evaluation import summarize

    runtime = LocalRuntime.from_loaded(model, tokenizer, device)
    was_training = model.training
    model.eval()
    try:
        cases = []
        for row in rows:
            observation, generation = runtime.analyze(row.input_text)
            cases.append(
                {
                    "expected": row.expected,
                    "predicted": observation.training_target(),
                    "case_kind": row.case_kind,
                    "generation": {"seconds": generation.seconds},
                }
            )
        return summarize(cases)
    finally:
        model.train(was_training)


def selection_key(metrics: dict, loss: float) -> tuple:
    return (
        metrics["observation_matches"],
        -metrics["ordinary_with_extra_signals"],
        metrics["signal_f1"] or 0.0,
        -loss,
    )


def validation_loss(model, examples: list[dict], device: str) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for example in examples:
            loss, n = completion_loss(model, example, device)
            total += loss.item() * n
            count += n
    model.train()
    return total / count


def train(args) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Training output must be empty; choose a new run directory")
    if args.epochs < 1 or args.accumulation < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, accumulation and learning rate must be positive")
    if not math.isfinite(args.decision_weight) or args.decision_weight < 1:
        raise ValueError("Decision weight must be finite and at least 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = select_device(args.device)
    with (args.model_dir / "model.safetensors").open("rb") as weights:
        weights_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
    if (
        weights_sha256
        != "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
    ):
        raise ValueError("Base weights do not match the pinned Qwen revision")
    dataset = load_dataset(args.data_dir)
    report = dataset_report(dataset, args.data_dir)
    model, tokenizer = load_base(args.model_dir, device)
    lengths = token_lengths(dataset, tokenizer)
    train_rows = [
        encode_example(r, tokenizer, args.decision_weight) for r in dataset["train"]
    ]
    valid_rows = [encode_example(r, tokenizer) for r in dataset["validation"]]
    # Test examples are integrity/length checked only, never forward/backwarded here.
    parent = None
    if args.init_adapter:
        parent = json.loads((args.init_adapter / "training_manifest.json").read_text())
        if (
            not parent.get("training_completed")
            or parent["model_revision"] != TOKENIZER_REVISION
            or parent["prompt_sha256"] != report["prompt_sha256"]
            or parent["data_sha256"]
            != {s: v["sha256"] for s, v in report["splits"].items()}
        ):
            raise ValueError(
                "Continuation requires the same completed model, prompt and data"
            )
        model = PeftModel.from_pretrained(
            model, args.init_adapter, is_trainable=True, local_files_only=True
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    for p in parameters:
        p.data = p.data.float()
    total_steps = math.ceil(len(train_rows) / args.accumulation) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(total_steps * 0.1), total_steps
    )
    args.output.mkdir(parents=True)
    manifest = {
        "model_id": TOKENIZER_ID,
        "model_revision": TOKENIZER_REVISION,
        "base_weights_sha256": weights_sha256,
        "prompt_sha256": report["prompt_sha256"],
        "data_sha256": {s: v["sha256"] for s, v in report["splits"].items()},
        "device": device,
        "base_dtype": str(next(model.parameters()).dtype),
        "adapter_dtype": "float32",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "gradient_accumulation": args.accumulation,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "decision_weight": args.decision_weight,
        "generation_settings": generation_settings()
        if args.select_by_generation
        else None,
        "initial_adapter_sha256": hashlib.sha256(
            (args.init_adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        if args.init_adapter
        else None,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "gradient_clipping": 1.0,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "token_lengths": lengths,
        "packages": {
            p: version(p) for p in ["torch", "transformers", "peft", "accelerate"]
        },
        "loss": "completion-only causal cross entropy; decision-token weighting; accumulation weighted by supervised weight sum",
        "selection": "highest validation observation matches; ties: fewer ordinary extra signals, higher signal F1, lower loss"
        if args.select_by_generation
        else "lowest validation loss; test never used for training/selection",
    }
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "loaded",
                "device": device,
                "train_rows": len(train_rows),
                "steps": total_steps,
                "trainable": manifest["trainable_parameters"],
            }
        ),
        flush=True,
    )
    begin = time.perf_counter()
    initial = validation_loss(model, valid_rows, device)
    best = initial
    best_key = None
    model.save_pretrained(args.output, safe_serialization=True)
    manifest["selected_epoch"] = 0
    if args.select_by_generation:
        metrics = generation_validation(model, tokenizer, dataset["validation"], device)
        best_key = selection_key(metrics, initial)
        manifest["initial_validation_generation"] = metrics
        manifest["selected_validation_generation"] = metrics
        print(json.dumps({"event": "initial_generation", **metrics}), flush=True)
    history = []
    steps = 0
    for epoch in range(args.epochs):
        order = list(train_rows)
        random.Random(args.seed + epoch).shuffle(order)
        epoch_loss, epoch_tokens = 0.0, 0
        for offset in range(0, len(order), args.accumulation):
            group = order[offset : offset + args.accumulation]
            tokens = sum(
                sum(r.get("weights", [1.0] * (len(r["ids"]) - r["start"])))
                for r in group
            )
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for row in group:
                loss, count = completion_loss(model, row, device)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss; training stopped")
                (loss * count / tokens).backward()
                step_loss += loss.item() * count
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Non-finite gradient; training stopped")
            optimizer.step()
            scheduler.step()
            steps += 1
            epoch_loss += step_loss
            epoch_tokens += tokens
            print(
                json.dumps(
                    {
                        "event": "step",
                        "step": steps,
                        "epoch": epoch + 1,
                        "loss": step_loss / tokens,
                        "seconds": time.perf_counter() - begin,
                    }
                ),
                flush=True,
            )
            if args.max_steps and steps >= args.max_steps:
                break
        valid = validation_loss(model, valid_rows, device)
        entry = {
            "epoch": epoch + 1,
            "step": steps,
            "train_loss": epoch_loss / epoch_tokens,
            "validation_loss": valid,
        }
        history.append(entry)
        better = valid < best
        if args.select_by_generation:
            metrics = generation_validation(
                model, tokenizer, dataset["validation"], device
            )
            entry["generation"] = metrics
            key = selection_key(metrics, valid)
            better = key > best_key
            if better:
                best_key = key
                manifest["selected_validation_generation"] = metrics
        if better:
            best = valid
            model.save_pretrained(args.output, safe_serialization=True)
            manifest["selected_epoch"] = epoch + 1
        manifest.update(
            {
                "initial_validation_loss": initial,
                "best_validation_loss": min(
                    initial, *(h["validation_loss"] for h in history)
                ),
                "selected_validation_loss": best,
                "history": history,
                "steps_completed": steps,
                "seconds": time.perf_counter() - begin,
            }
        )
        (args.output / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps({"event": "epoch", **entry}), flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    manifest["training_completed"] = True
    if device == "mps":
        manifest["final_mps_allocated_bytes"] = torch.mps.current_allocated_memory()
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/pytorch-adapter")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="Continue a completed adapter on identical data/prompt",
    )
    parser.add_argument("--decision-weight", type=float, default=1.0)
    parser.add_argument("--select-by-generation", action="store_true")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--max-steps", type=int, default=0, help="Optional bounded training run"
    )
    args = parser.parse_args()
    result = train(args)
    print(
        json.dumps(
            {
                "saved": str(args.output),
                "best_validation_loss": result["best_validation_loss"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
