"""Read optional configuration for the local-only application."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Local model selection, to be used by the future analysis module."""

    local_model: str | None


def load_settings() -> Settings:
    """Load optional settings, preserving values already set in the environment."""

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    return Settings(local_model=os.getenv("EASY_TECH_HELP_LOCAL_MODEL") or None)
