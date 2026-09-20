"""Checks for the configuration boundary used by future API integration."""

from easy_tech_help.config import load_settings


def test_environment_overrides_dotenv_without_exposing_api_key(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "EASY_TECH_HELP_API_KEY=file-secret\nEASY_TECH_HELP_MODEL=example-model\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EASY_TECH_HELP_API_KEY", "environment-secret")
    monkeypatch.delenv("EASY_TECH_HELP_MODEL", raising=False)

    settings = load_settings()

    assert settings.api_key == "environment-secret"
    assert settings.model == "example-model"
    assert "environment-secret" not in repr(settings)
