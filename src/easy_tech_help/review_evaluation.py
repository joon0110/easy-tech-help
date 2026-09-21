"""Replay application corrections on recorded raw predictions, without retraining."""

import hashlib
import json
from pathlib import Path

from easy_tech_help.evaluation import summarize
from easy_tech_help.observation_review import review_observation
from easy_tech_help.schemas import TextObservation


def main() -> int:
    results = {
        "scope": "Replay of unchanged recorded PyTorch outputs on previously inspected development/regression splits. Measures application corrections, not a new model run or independent accuracy.",
        "runs": {},
    }
    for split, path in [
        ("validation", Path("results/pytorch_v2_validation.json")),
        ("regression", Path("results/pytorch_v2_regression.json")),
    ]:
        original = json.loads(path.read_text())["runs"]["adapter"]["cases"]
        reviewed, changes = [], []
        for case in original:
            raw = TextObservation.model_validate(
                case["predicted"], context={"input_text": case["input_text"]}
            )
            observation, reasons = review_observation(case["input_text"], raw)
            reviewed.append({**case, "predicted": observation.training_target()})
            if reasons:
                changes.append(
                    {
                        "id": case["id"],
                        "input_text": case["input_text"],
                        "before": raw.training_target(),
                        "after": observation.training_target(),
                        "reasons": reasons,
                        "expected": case["expected"],
                    }
                )
        results["runs"][split] = {
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "before": summarize(original),
            "after": summarize(reviewed),
            "changes": changes,
        }
    results["review_code_sha256"] = hashlib.sha256(
        Path(__file__).with_name("observation_review.py").read_bytes()
    ).hexdigest()
    output = Path("results/observation_review.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
