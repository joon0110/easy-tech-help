"""Read optional configuration for the local-only application."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Local text model selection."""

    local_model: str | None
    adapter_dir: str
    device: str


def load_settings() -> Settings:
    """Load optional settings, preserving values already set in the environment."""

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    return Settings(
        local_model=os.getenv("EASY_TECH_HELP_LOCAL_MODEL") or None,
        adapter_dir=os.getenv(
            "EASY_TECH_HELP_ADAPTER_DIR", "artifacts/pytorch-adapter-v2"
        ),
        device=os.getenv("EASY_TECH_HELP_DEVICE", "auto"),
    )
