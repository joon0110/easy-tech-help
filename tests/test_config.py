"""Checks for the future local model configuration boundary."""

from easy_tech_help.config import load_settings


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("EASY_TECH_HELP_LOCAL_MODEL=file-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EASY_TECH_HELP_LOCAL_MODEL", "environment-model")

    settings = load_settings()

    assert settings.local_model == "environment-model"
