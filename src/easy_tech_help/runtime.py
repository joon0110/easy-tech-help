"""Offline PyTorch generation with the pinned base model and optional LoRA."""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from easy_tech_help.analysis import build_messages
from easy_tech_help.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from easy_tech_help.schemas import TextObservation, validate_input

MODEL_DIR = Path("artifacts/pytorch-model")
ADAPTER_DIR = Path("artifacts/pytorch-adapter")


@dataclass
class Generation:
    text: str
    complete: bool
    tokens: int
    seconds: float


def select_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("PyTorch MPS is not available in this process")
    if requested not in {"mps", "cpu"}:
        raise ValueError("Device must be auto, mps or cpu")
    return requested


def load_base(model_dir: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # BF16 has a wider exponent range than FP16; M4/macOS supports BF16 on MPS.
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    return model, tokenizer


class LocalRuntime:
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        adapter_dir: Path | None = ADAPTER_DIR,
        device: str = "auto",
    ):
        self.device = select_device(device)
        if adapter_dir is not None:
            manifest = json.loads((adapter_dir / "training_manifest.json").read_text())
            if not manifest.get("training_completed"):
                raise ValueError("Adapter training has not completed")
            prompt_hash = hashlib.sha256(
                build_messages("audit")[0]["content"].encode()
            ).hexdigest()
            if (
                manifest["model_id"] != TOKENIZER_ID
                or manifest["model_revision"] != TOKENIZER_REVISION
                or manifest["prompt_sha256"] != prompt_hash
            ):
                raise ValueError("Adapter model or prompt does not match this runtime")
        self.model, self.tokenizer = load_base(model_dir, self.device)
        if adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, adapter_dir, is_trainable=False, local_files_only=True
            )
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> Generation:
        import torch
        from transformers import GenerationConfig

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if encoded["input_ids"].shape[1] + max_new_tokens > 4096:
            raise ValueError("Text exceeds the local context budget; shorten it")
        eos = self.model.generation_config.eos_token_id
        stop_ids = eos if isinstance(eos, list) else [eos]
        settings = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**encoded, generation_config=settings)
        tail = output[0, encoded["input_ids"].shape[1] :].tolist()
        return Generation(
            text=self.tokenizer.decode(tail, skip_special_tokens=True),
            complete=bool(tail and tail[-1] in stop_ids),
            tokens=len(tail),
            seconds=time.perf_counter() - start,
        )

    def analyze(self, text: str) -> tuple[TextObservation, Generation]:
        validate_input(text)
        result = self.generate(build_messages(text))
        if not result.complete:
            return TextObservation.unknown("incomplete_model_output"), result
        try:
            observation = TextObservation.model_validate_json(
                result.text, context={"input_text": text}
            )
        except ValueError:
            observation = TextObservation.unknown("invalid_model_output")
        return observation, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument(
        "--base", action="store_true", help="Use the untrained baseline"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()
    runtime = LocalRuntime(
        args.model_dir, None if args.base else args.adapter_dir, args.device
    )
    observation, generation = runtime.analyze(args.text)
    print(observation.model_dump_json(indent=2))
    print(json.dumps({"seconds": generation.seconds, "tokens": generation.tokens}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
