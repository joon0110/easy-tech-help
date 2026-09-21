"""Run analysis, RAG, action selection, and the family summary."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from easy_tech_help.analysis import DEFAULT_MODEL, LocalModelError
from easy_tech_help.config import load_settings
from easy_tech_help.handoff import FamilyHandoff, build_handoff
from easy_tech_help.rag import RagResult, explain_text
from easy_tech_help.retrieval import DEFAULT_KNOWLEDGE_DIR
from easy_tech_help.safety import Guidance, build_guidance, displayable_reference


@dataclass
class GuidedResult:
    analysis: RagResult
    guidance: Guidance
    family_handoff: FamilyHandoff

    def to_dict(self):
        return {
            "guidance": self.guidance.to_dict(),
            "family_handoff": self.family_handoff.to_dict(),
            "reference_explanation": displayable_reference(self.analysis),
            "analysis": self.analysis.to_dict(),
        }


def guide_text(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    directory: Path = DEFAULT_KNOWLEDGE_DIR,
    runtime=None,
) -> GuidedResult:
    result = explain_text(text, model=model, directory=directory, runtime=runtime)
    guidance = build_guidance(text, result, directory)
    return GuidedResult(
        result,
        guidance,
        build_handoff(result.observation.category, guidance, directory),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    parser.add_argument(
        "--family-summary", action="store_true", help="Print only the shareable summary"
    )
    args = parser.parse_args()
    try:
        text = args.file.read_text(encoding="utf-8") if args.file else args.text
        result = guide_text(text, model=load_settings().local_model or DEFAULT_MODEL)
    except (OSError, ValueError, LocalModelError) as exc:
        parser.exit(1, f"{exc}\n")
    print(
        result.family_handoff.text
        if args.family_summary
        else json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
