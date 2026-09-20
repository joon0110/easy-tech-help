"""Read local configuration without exposing credentials in debug output."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration reserved for the multimodal integration."""

    api_key: str | None = field(repr=False)
    model: str | None


def load_settings() -> Settings:
    """Load optional settings, preserving values already set in the environment."""

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    return Settings(
        api_key=os.getenv("EASY_TECH_HELP_API_KEY") or None,
        model=os.getenv("EASY_TECH_HELP_MODEL") or None,
    )
